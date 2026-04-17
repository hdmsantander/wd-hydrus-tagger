"""Async Hydrus Network Client API wrapper.

Uses a shared ``httpx.AsyncClient`` per (api_url, access_key) so HTTP keep-alive
and connection pooling apply across many ``get_file`` / metadata calls.
"""

import asyncio
import json
import logging
from typing import Any

import httpx

_pool_log = logging.getLogger(__name__)

_pool_lock = asyncio.Lock()
_client_pool: dict[tuple[str, str], httpx.AsyncClient] = {}
_HYDRUS_HTTP_LIMITS = httpx.Limits(max_keepalive_connections=128, max_connections=192)


async def aclose_all_hydrus_clients() -> None:
    """Close every pooled client (app shutdown)."""
    async with _pool_lock:
        for client in _client_pool.values():
            await client.aclose()
        _client_pool.clear()


async def invalidate_hydrus_client_pool() -> None:
    """Close all pooled clients when Hydrus URL or API key changes."""
    await aclose_all_hydrus_clients()


class HydrusClient:
    def __init__(self, api_url: str, access_key: str):
        self.api_url = api_url.rstrip("/")
        self.access_key = access_key

    def _pool_key(self) -> tuple[str, str]:
        return (self.api_url, self.access_key)

    async def _shared(self) -> httpx.AsyncClient:
        key = self._pool_key()
        async with _pool_lock:
            if key not in _client_pool:
                _pool_log.debug(
                    "hydrus_http pool new_client api_url=%s keepalive_max=%s max_connections=%s",
                    self.api_url,
                    _HYDRUS_HTTP_LIMITS.max_keepalive_connections,
                    _HYDRUS_HTTP_LIMITS.max_connections,
                )
                _client_pool[key] = httpx.AsyncClient(
                    base_url=self.api_url,
                    headers={"Hydrus-Client-API-Access-Key": self.access_key},
                    timeout=httpx.Timeout(120.0, connect=15.0),
                    # Room for hydrus_download_parallel concurrent GETs + chunked metadata + search overlap.
                    limits=_HYDRUS_HTTP_LIMITS,
                    follow_redirects=True,
                )
            return _client_pool[key]

    async def _get(
        self,
        path: str,
        params: dict | None = None,
        *,
        timeout: httpx.Timeout | float | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        client = await self._shared()
        resp = await client.get(path, params=params, timeout=timeout, headers=extra_headers)
        resp.raise_for_status()
        return resp

    async def _post(self, path: str, json_data: dict | None = None) -> httpx.Response:
        client = await self._shared()
        resp = await client.post(path, json=json_data)
        resp.raise_for_status()
        return resp

    async def verify_access_key(self) -> dict:
        resp = await self._get("/api_version")
        version_info = resp.json()
        resp2 = await self._get("/verify_access_key")
        key_info = resp2.json()
        return {**version_info, **key_info}

    async def get_services(self) -> list[dict]:
        resp = await self._get("/get_services")
        data = resp.json()
        services = []
        for type_name, service_list in data.items():
            if not isinstance(service_list, list):
                continue
            for svc in service_list:
                if isinstance(svc, dict) and "service_key" in svc:
                    services.append({
                        "service_key": svc.get("service_key", ""),
                        "name": svc.get("name", ""),
                        "type": svc.get("type", 0),
                        "type_pretty": svc.get("type_pretty", type_name),
                    })
        return services

    async def search_files(
        self,
        tags: list[str],
        file_sort_type: int | None = None,
        file_sort_asc: bool | None = None,
    ) -> list[int]:
        params: dict[str, Any] = {"tags": json.dumps(tags)}
        if file_sort_type is not None:
            params["file_sort_type"] = file_sort_type
        if file_sort_asc is not None:
            params["file_sort_asc"] = str(file_sort_asc).lower()
        resp = await self._get("/get_files/search_files", params=params)
        data = resp.json()
        return data.get("file_ids", [])

    async def get_file_metadata(self, file_ids: list[int]) -> list[dict]:
        params = {"file_ids": json.dumps(file_ids)}
        resp = await self._get("/get_files/file_metadata", params=params)
        data = resp.json()
        return data.get("metadata", [])

    async def get_thumbnail(self, file_id: int) -> tuple[bytes, str]:
        resp = await self._get(
            "/get_files/thumbnail",
            params={"file_id": file_id},
            timeout=60.0,
            extra_headers={"Accept": "*/*"},
        )
        content_type = resp.headers.get("content-type", "image/jpeg")
        return resp.content, content_type

    async def get_file(self, file_id: int) -> tuple[bytes, str]:
        resp = await self._get(
            "/get_files/file",
            params={"file_id": file_id},
            timeout=120.0,
            extra_headers={"Accept": "*/*"},
        )
        content_type = resp.headers.get("content-type", "application/octet-stream")
        return resp.content, content_type

    async def add_tags(
        self,
        hash_: str,
        service_key: str,
        tags: list[str],
    ) -> None:
        await self.apply_tag_actions(
            hash_=hash_,
            service_key=service_key,
            add_tags=tags,
            remove_tags=[],
        )

    async def apply_tag_actions(
        self,
        hash_: str,
        service_key: str,
        *,
        add_tags: list[str],
        remove_tags: list[str],
    ) -> None:
        actions: dict[str, list[str]] = {}
        if add_tags:
            actions["0"] = add_tags
        if remove_tags:
            actions["1"] = remove_tags
        if not actions:
            return
        await self._post("/add_tags/add_tags", json_data={
            "hashes": [hash_],
            "service_keys_to_actions_to_tags": {
                service_key: actions,
            },
        })

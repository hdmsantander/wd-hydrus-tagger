"""Async Hydrus Network Client API wrapper."""

import json
from typing import Any

import httpx


class HydrusClient:
    def __init__(self, api_url: str, access_key: str):
        self.api_url = api_url.rstrip("/")
        self.access_key = access_key

    def _headers(self) -> dict[str, str]:
        return {"Hydrus-Client-API-Access-Key": self.access_key}

    async def _get(self, path: str, params: dict | None = None) -> Any:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{self.api_url}{path}",
                headers=self._headers(),
                params=params,
            )
            resp.raise_for_status()
            return resp

    async def _post(self, path: str, json_data: dict | None = None) -> Any:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.api_url}{path}",
                headers=self._headers(),
                json=json_data,
            )
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
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(
                f"{self.api_url}/get_files/file_metadata",
                headers=self._headers(),
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("metadata", [])

    async def get_thumbnail(self, file_id: int) -> tuple[bytes, str]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{self.api_url}/get_files/thumbnail",
                headers=self._headers(),
                params={"file_id": file_id},
            )
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "image/jpeg")
            return resp.content, content_type

    async def get_file(self, file_id: int) -> tuple[bytes, str]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(
                f"{self.api_url}/get_files/file",
                headers=self._headers(),
                params={"file_id": file_id},
            )
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "application/octet-stream")
            return resp.content, content_type

    async def add_tags(
        self,
        hash_: str,
        service_key: str,
        tags: list[str],
    ) -> None:
        await self._post("/add_tags/add_tags", json_data={
            "hashes": [hash_],
            "service_keys_to_tags": {
                service_key: tags,
            },
        })

"""Hydrus HTTP client pooling and API wrappers."""

import httpx
from unittest.mock import MagicMock

import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]

import backend.hydrus.client as hydrus_mod
from backend.hydrus.client import (
    HydrusClient,
    aclose_all_hydrus_clients,
    invalidate_hydrus_client_pool,
)


@pytest.mark.asyncio
async def test_aclose_all_hydrus_clients_clears_pool():
    hc = HydrusClient("http://example.invalid:45869", "key")
    await hc._shared()
    assert len(hydrus_mod._client_pool) == 1
    await aclose_all_hydrus_clients()
    assert len(hydrus_mod._client_pool) == 0


@pytest.mark.asyncio
async def test_same_pool_key_reuses_client():
    await aclose_all_hydrus_clients()
    a = HydrusClient("http://h.test:1", "k")
    b = HydrusClient("http://h.test:1", "k")
    ca = await a._shared()
    cb = await b._shared()
    assert ca is cb
    await aclose_all_hydrus_clients()


@pytest.mark.asyncio
async def test_invalidate_hydrus_client_pool_closes():
    await aclose_all_hydrus_clients()
    hc = HydrusClient("http://inv.test:2", "k")
    await hc._shared()
    assert len(hydrus_mod._client_pool) == 1
    await invalidate_hydrus_client_pool()
    assert len(hydrus_mod._client_pool) == 0


def _ok_response(payload: dict):
    r = MagicMock()
    r.json.return_value = payload
    r.raise_for_status = lambda: None
    r.content = b"x"
    r.headers = {"content-type": "image/jpeg"}
    return r


@pytest.mark.asyncio
async def test_verify_access_key_merges_json(monkeypatch):
    await aclose_all_hydrus_clients()
    c = HydrusClient("http://h.test", "key")
    calls: list[str] = []

    async def fake_get(path, params=None, timeout=None, extra_headers=None):
        calls.append(path)
        if path == "/api_version":
            return _ok_response({"hydrus_version": 1})
        if path == "/verify_access_key":
            return _ok_response({"access": "ok"})
        raise AssertionError(path)

    monkeypatch.setattr(c, "_get", fake_get)
    out = await c.verify_access_key()
    assert out["hydrus_version"] == 1
    assert out["access"] == "ok"
    assert calls == ["/api_version", "/verify_access_key"]


@pytest.mark.asyncio
async def test_get_services_skips_non_list_and_builds_rows(monkeypatch):
    await aclose_all_hydrus_clients()
    c = HydrusClient("http://h.test", "key")
    payload = {
        "local_tags": [
            {"service_key": "a", "name": "A", "type": 0, "type_pretty": "local"},
        ],
        "broken": "not-a-list",
        "empty": [],
    }

    async def fake_get(path, params=None, timeout=None, extra_headers=None):
        assert path == "/get_services"
        return _ok_response(payload)

    monkeypatch.setattr(c, "_get", fake_get)
    svcs = await c.get_services()
    assert len(svcs) == 1
    assert svcs[0]["service_key"] == "a"


@pytest.mark.asyncio
async def test_search_files_optional_sort_params(monkeypatch):
    await aclose_all_hydrus_clients()
    c = HydrusClient("http://h.test", "key")
    captured: dict = {}

    async def fake_get(path, params=None, timeout=None, extra_headers=None):
        captured["path"] = path
        captured["params"] = params
        return _ok_response({"file_ids": [5, 6]})

    monkeypatch.setattr(c, "_get", fake_get)
    ids = await c.search_files(["a", "b"], file_sort_type=2, file_sort_asc=False)
    assert ids == [5, 6]
    assert captured["params"]["file_sort_type"] == 2
    assert captured["params"]["file_sort_asc"] == "false"


@pytest.mark.asyncio
async def test_get_file_metadata(monkeypatch):
    await aclose_all_hydrus_clients()
    c = HydrusClient("http://h.test", "key")

    async def fake_get(path, params=None, timeout=None, extra_headers=None):
        return _ok_response({"metadata": [{"file_id": 1}]})

    monkeypatch.setattr(c, "_get", fake_get)
    rows = await c.get_file_metadata([1])
    assert rows == [{"file_id": 1}]


@pytest.mark.asyncio
async def test_get_thumbnail_and_get_file(monkeypatch):
    await aclose_all_hydrus_clients()
    c = HydrusClient("http://h.test", "key")

    async def fake_get(path, params=None, timeout=None, extra_headers=None):
        r = MagicMock()
        r.content = b"data"
        r.raise_for_status = lambda: None
        if "thumbnail" in path:
            r.headers = {"content-type": "image/png"}
        else:
            r.headers = {}
        return r

    monkeypatch.setattr(c, "_get", fake_get)
    blob, ct = await c.get_thumbnail(42)
    assert blob == b"data"
    assert ct == "image/png"
    blob2, ct2 = await c.get_file(99)
    assert blob2 == b"data"
    assert ct2 == "application/octet-stream"


@pytest.mark.asyncio
async def test_get_raises_for_status_propagates(monkeypatch):
    """500 from Hydrus → ``raise_for_status`` in ``_get`` / ``_post``."""
    await aclose_all_hydrus_clients()
    c = HydrusClient("http://h.test", "key")
    req = httpx.Request("GET", "http://h.test/x")
    bad = httpx.Response(500, request=req)

    async def fake_shared():
        cli = MagicMock()

        async def get(*_a, **_k):
            return bad

        cli.get = get
        return cli

    monkeypatch.setattr(c, "_shared", fake_shared)
    with pytest.raises(httpx.HTTPStatusError):
        await c.search_files(["x"])


@pytest.mark.asyncio
async def test_post_raises_for_status_propagates(monkeypatch):
    await aclose_all_hydrus_clients()
    c = HydrusClient("http://h.test", "key")
    req = httpx.Request("POST", "http://h.test/x")
    bad = httpx.Response(502, request=req)

    async def fake_shared():
        cli = MagicMock()

        async def post(*_a, **_k):
            return bad

        cli.post = post
        return cli

    monkeypatch.setattr(c, "_shared", fake_shared)
    with pytest.raises(httpx.HTTPStatusError):
        await c.add_tags("h", "svc", ["t"])


@pytest.mark.asyncio
async def test_add_tags_posts(monkeypatch):
    await aclose_all_hydrus_clients()
    c = HydrusClient("http://h.test", "key")
    posted: list = []

    async def fake_post(path, json_data=None):
        posted.append((path, json_data))
        r = MagicMock()
        r.raise_for_status = lambda: None
        return r

    monkeypatch.setattr(c, "_post", fake_post)
    await c.add_tags("deadbeef", "svc", ["t1"])
    assert posted[0][0] == "/add_tags/add_tags"
    assert "deadbeef" in posted[0][1]["hashes"]
    actions = posted[0][1]["service_keys_to_actions_to_tags"]["svc"]
    assert actions["0"] == ["t1"]


@pytest.mark.asyncio
async def test_apply_tag_actions_noop_when_both_empty(monkeypatch):
    await aclose_all_hydrus_clients()
    c = HydrusClient("http://h.test", "key")
    posted: list = []

    async def fake_post(path, json_data=None):
        posted.append((path, json_data))
        r = MagicMock()
        r.raise_for_status = lambda: None
        return r

    monkeypatch.setattr(c, "_post", fake_post)
    await c.apply_tag_actions("deadbeef", "svc", add_tags=[], remove_tags=[])
    assert posted == []


@pytest.mark.asyncio
async def test_apply_tag_actions_posts_add_and_remove(monkeypatch):
    await aclose_all_hydrus_clients()
    c = HydrusClient("http://h.test", "key")
    posted: list = []

    async def fake_post(path, json_data=None):
        posted.append((path, json_data))
        r = MagicMock()
        r.raise_for_status = lambda: None
        return r

    monkeypatch.setattr(c, "_post", fake_post)
    await c.apply_tag_actions(
        "deadbeef",
        "svc",
        add_tags=["new_tag"],
        remove_tags=["old_tag"],
    )
    assert posted[0][0] == "/add_tags/add_tags"
    payload = posted[0][1]
    actions = payload["service_keys_to_actions_to_tags"]["svc"]
    assert actions["0"] == ["new_tag"]
    assert actions["1"] == ["old_tag"]


@pytest.mark.asyncio
async def test_apply_tag_actions_remove_only_posts_delete_action(monkeypatch):
    await aclose_all_hydrus_clients()
    c = HydrusClient("http://h.test", "key")
    posted: list = []

    async def fake_post(path, json_data=None):
        posted.append((path, json_data))
        r = MagicMock()
        r.raise_for_status = lambda: None
        return r

    monkeypatch.setattr(c, "_post", fake_post)
    await c.apply_tag_actions("deadbeef", "svc", add_tags=[], remove_tags=["gone"])
    assert posted[0][0] == "/add_tags/add_tags"
    actions = posted[0][1]["service_keys_to_actions_to_tags"]["svc"]
    assert "0" not in actions
    assert actions["1"] == ["gone"]

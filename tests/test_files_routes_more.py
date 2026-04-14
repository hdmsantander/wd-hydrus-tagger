"""Additional ``/api/files`` routes: search errors, full file proxy, bad metadata ids."""

from unittest.mock import AsyncMock

import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]

from fastapi.testclient import TestClient

import backend.config as config_module
import backend.routes.files as files_routes
from backend.config import AppConfig


@pytest.fixture
def base_cfg(tmp_path):
    return AppConfig(
        models_dir=str(tmp_path / "models"),
        hydrus_api_key="k",
        hydrus_api_url="http://invalid.test",
        hydrus_metadata_chunk_size=100,
    )


def test_search_files_hydrus_error_json(base_cfg, monkeypatch):
    class BoomHydrus:
        def __init__(self, *a, **k):
            pass

        async def search_files(self, **kw):
            raise RuntimeError("hydrus search failed")

    monkeypatch.setattr(config_module, "_config", base_cfg)
    monkeypatch.setattr(files_routes, "get_config", lambda: base_cfg)
    monkeypatch.setattr(files_routes, "HydrusClient", BoomHydrus)

    from backend.app import app

    with TestClient(app) as c:
        r = c.post("/api/files/search", json={"tags": ["a"]})
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is False
    assert "hydrus search failed" in body["error"]


def test_get_file_proxy_success(base_cfg, monkeypatch):
    class OkHydrus:
        def __init__(self, *a, **k):
            pass

        async def get_file(self, file_id: int):
            return b"bin", "application/octet-stream"

    monkeypatch.setattr(config_module, "_config", base_cfg)
    monkeypatch.setattr(files_routes, "get_config", lambda: base_cfg)
    monkeypatch.setattr(files_routes, "HydrusClient", OkHydrus)

    from backend.app import app

    with TestClient(app) as c:
        r = c.get("/api/files/42")
    assert r.status_code == 200
    assert r.content == b"bin"


def test_get_file_proxy_error_502(base_cfg, monkeypatch):
    class BoomHydrus:
        def __init__(self, *a, **k):
            pass

        async def get_file(self, file_id: int):
            raise OSError("no file")

    monkeypatch.setattr(config_module, "_config", base_cfg)
    monkeypatch.setattr(files_routes, "get_config", lambda: base_cfg)
    monkeypatch.setattr(files_routes, "HydrusClient", BoomHydrus)

    from backend.app import app

    with TestClient(app) as c:
        r = c.get("/api/files/1")
    assert r.status_code == 502
    assert "no file" in r.json()["error"]


def test_metadata_invalid_file_id_element(base_cfg, monkeypatch):
    async def fake_metadata(file_ids: list[int]):
        return []

    class FakeHydrus:
        def __init__(self, *a, **k):
            pass

        get_file_metadata = AsyncMock(side_effect=fake_metadata)

    monkeypatch.setattr(config_module, "_config", base_cfg)
    monkeypatch.setattr(files_routes, "get_config", lambda: base_cfg)
    monkeypatch.setattr(files_routes, "HydrusClient", FakeHydrus)

    from backend.app import app

    with TestClient(app) as c:
        r = c.post("/api/files/metadata", json={"file_ids": ["nope"]})
    assert r.status_code == 200
    assert r.json()["success"] is False

"""POST /api/connection/test URL / key resolution."""

from unittest.mock import AsyncMock

import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]

from fastapi.testclient import TestClient

import backend.config as config_module
import backend.routes.connection as connection_routes
from backend.config import AppConfig


@pytest.fixture
def conn_client(monkeypatch, tmp_path):
    cfg = AppConfig(
        models_dir=str(tmp_path / "models"),
        hydrus_api_key="saved-key",
        hydrus_api_url="http://config-hydrus.example:45869",
    )
    monkeypatch.setattr(config_module, "_config", cfg)
    monkeypatch.setattr(connection_routes, "get_config", lambda: cfg)

    async def ok_verify():
        return {"ok": True}

    fake_client_instances = []

    class FakeHydrus:
        def __init__(self, url, api_key):
            fake_client_instances.append((url, api_key))
            self.verify_access_key = AsyncMock(side_effect=ok_verify)

    monkeypatch.setattr(connection_routes, "HydrusClient", FakeHydrus)

    def _noop_save(_c):
        return None

    monkeypatch.setattr(connection_routes, "save_config", _noop_save)

    from backend.app import app

    with TestClient(app) as client:
        yield client, cfg, fake_client_instances


def test_connection_test_empty_body_uses_config(conn_client):
    client, _, instances = conn_client
    r = client.post("/api/connection/test", json={})
    assert r.status_code == 200
    assert r.json()["success"] is True
    assert instances[-1] == ("http://config-hydrus.example:45869", "saved-key")


def test_connection_test_blank_url_falls_back_to_config(conn_client):
    client, _, instances = conn_client
    r = client.post(
        "/api/connection/test",
        json={"url": "   ", "api_key": "from-form"},
    )
    assert r.status_code == 200
    assert r.json()["success"] is True
    assert instances[-1][0] == "http://config-hydrus.example:45869"
    assert instances[-1][1] == "from-form"


def test_connection_test_blank_api_key_falls_back_to_config(conn_client):
    client, _, instances = conn_client
    r = client.post(
        "/api/connection/test",
        json={"url": "http://explicit.example", "api_key": ""},
    )
    assert r.status_code == 200
    assert r.json()["success"] is True
    assert instances[-1] == ("http://explicit.example", "saved-key")


def test_connection_test_no_json_body_uses_config(conn_client):
    client, _, instances = conn_client
    r = client.post("/api/connection/test")
    assert r.status_code == 200
    assert r.json()["success"] is True
    assert instances[-1][0] == "http://config-hydrus.example:45869"


def test_connection_test_verify_failure(monkeypatch, tmp_path):
    cfg = AppConfig(
        models_dir=str(tmp_path / "models"),
        hydrus_api_key="k",
        hydrus_api_url="http://h.test",
    )
    monkeypatch.setattr(config_module, "_config", cfg)
    monkeypatch.setattr(connection_routes, "get_config", lambda: cfg)

    class FakeHydrus:
        def __init__(self, *a, **k):
            pass

        verify_access_key = AsyncMock(side_effect=RuntimeError("refused"))

    monkeypatch.setattr(connection_routes, "HydrusClient", FakeHydrus)
    monkeypatch.setattr(connection_routes, "save_config", lambda _c: None)

    from backend.app import app

    with TestClient(app) as client:
        r = client.post("/api/connection/test", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is False
    assert "refused" in body["error"]


def test_get_services_requires_api_key(monkeypatch, tmp_path):
    cfg = AppConfig(
        models_dir=str(tmp_path / "models"),
        hydrus_api_key="",
        hydrus_api_url="http://h.test",
    )
    monkeypatch.setattr(config_module, "_config", cfg)
    monkeypatch.setattr(connection_routes, "get_config", lambda: cfg)

    from backend.app import app

    with TestClient(app) as client:
        r = client.get("/api/connection/services")
    assert r.json()["success"] is False


def test_get_services_hydrus_error(monkeypatch, tmp_path):
    cfg = AppConfig(
        models_dir=str(tmp_path / "models"),
        hydrus_api_key="k",
        hydrus_api_url="http://h.test",
    )
    monkeypatch.setattr(config_module, "_config", cfg)
    monkeypatch.setattr(connection_routes, "get_config", lambda: cfg)

    class FakeHydrus:
        def __init__(self, *a, **k):
            pass

        get_services = AsyncMock(side_effect=OSError("unavailable"))

    monkeypatch.setattr(connection_routes, "HydrusClient", FakeHydrus)

    from backend.app import app

    with TestClient(app) as client:
        r = client.get("/api/connection/services")
    assert r.status_code == 200
    assert r.json()["success"] is False
    assert "unavailable" in r.json()["error"]

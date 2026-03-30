"""POST /api/files/metadata chunks large file_id lists."""

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import backend.config as config_module
import backend.routes.files as files_routes
from backend.config import AppConfig


@pytest.fixture
def meta_client(monkeypatch, tmp_path):
    calls: list[list[int]] = []

    async def fake_metadata(file_ids: list[int]):
        calls.append(list(file_ids))
        return [{"file_id": int(i), "hash": f"h{i}"} for i in file_ids]

    class FakeHydrus:
        def __init__(self, *a, **k):
            pass

        get_file_metadata = AsyncMock(side_effect=fake_metadata)

    cfg = AppConfig(
        models_dir=str(tmp_path / "models"),
        hydrus_api_key="k",
        hydrus_api_url="http://invalid.test",
        hydrus_metadata_chunk_size=100,
    )
    monkeypatch.setattr(config_module, "_config", cfg)
    monkeypatch.setattr(files_routes, "get_config", lambda: cfg)
    monkeypatch.setattr(files_routes, "HydrusClient", FakeHydrus)

    from backend.app import app

    with TestClient(app) as client:
        yield client, calls


def test_metadata_endpoint_chunks_requests(meta_client):
    client, calls = meta_client
    ids = list(range(250))
    r = client.post("/api/files/metadata", json={"file_ids": ids})
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert len(body["metadata"]) == 250
    assert len(calls) == 3
    assert len(calls[0]) == 100
    assert len(calls[1]) == 100
    assert len(calls[2]) == 50


def test_metadata_rejects_non_list(meta_client):
    client, _ = meta_client
    r = client.post("/api/files/metadata", json={"file_ids": "bad"})
    assert r.status_code == 200
    assert r.json()["success"] is False

"""HTTP predict route batch_size override."""

from unittest.mock import MagicMock

import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]

from fastapi.testclient import TestClient

import backend.config as config_module
import backend.routes.tagger_http as tagger_http_routes


class FakePredictService:
    def __init__(self):
        self.last_batch_size = None

    def load_model(self, name: str, **kwargs) -> None:
        pass

    async def ensure_model(self, explicit_name, **kwargs):
        pass

    async def tag_files(self, client, file_ids, general_threshold, character_threshold, **kwargs):
        self.last_batch_size = kwargs.get("batch_size")
        return []


@pytest.fixture
def predict_client(monkeypatch):
    svc = FakePredictService()
    monkeypatch.setattr(tagger_http_routes, "get_config", config_module.get_config)
    monkeypatch.setattr(tagger_http_routes.TaggingService, "get_instance", lambda _c: svc)
    monkeypatch.setattr(tagger_http_routes, "HydrusClient", lambda *a, **k: MagicMock())

    from backend.app import app

    with TestClient(app) as client:
        yield client, svc


def test_predict_passes_batch_size_override(predict_client):
    client, svc = predict_client
    r = client.post(
        "/api/tagger/predict",
        json={
            "file_ids": [1, 2, 3],
            "general_threshold": 0.35,
            "character_threshold": 0.85,
            "batch_size": 7,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert svc.last_batch_size == 7


def test_predict_omits_invalid_batch_size(predict_client):
    client, svc = predict_client
    r = client.post(
        "/api/tagger/predict",
        json={
            "file_ids": [1],
            "batch_size": "not-a-number",
        },
    )
    assert r.status_code == 200
    assert r.json()["success"] is True
    assert svc.last_batch_size is None


def test_predict_returns_error_when_tag_files_raises(predict_client):
    client, svc = predict_client

    async def boom(*_a, **_k):
        raise RuntimeError("onnx failed")

    svc.tag_files = boom
    r = client.post("/api/tagger/predict", json={"file_ids": [1, 2]})
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is False
    assert "onnx failed" in body["error"]

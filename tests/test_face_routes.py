"""Face API route smoke tests (no InsightFace load)."""

import pytest
from fastapi.testclient import TestClient

pytestmark = [pytest.mark.full, pytest.mark.core]

import backend.config as config_module
from backend.face.service import FaceTaggingService


@pytest.fixture
def client(test_config):
    config_module._config = test_config
    FaceTaggingService._instance = None
    from backend.app import app

    return TestClient(app)


def test_face_providers_endpoint(client):
    r = client.get("/api/face/providers")
    assert r.status_code == 200
    data = r.json()
    assert data["success"] is True
    assert "CPUExecutionProvider" in data["providers"]


def test_face_status_endpoint(client):
    r = client.get("/api/face/status")
    assert r.status_code == 200
    data = r.json()
    assert data["success"] is True
    assert "faces" in data["status"]


def test_face_reset_endpoint(client):
    r = client.post("/api/face/reset")
    assert r.status_code == 200
    assert r.json()["success"] is True

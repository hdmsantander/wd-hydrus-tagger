"""Early WebSocket tagging validation (no full tagging session)."""

import json

import pytest
from fastapi.testclient import TestClient

import backend.config as config_module
import backend.routes.tagger_ws as tagger_ws_routes
from backend.app import app

pytestmark = [pytest.mark.full, pytest.mark.ws]


@pytest.fixture
def ws_validation_client(monkeypatch):
    monkeypatch.setattr(tagger_ws_routes, "get_config", config_module.get_config)
    with TestClient(app) as client:
        yield client


def test_ws_rejects_non_run_action(ws_validation_client):
    client = ws_validation_client
    with client.websocket_connect("/api/tagger/ws/progress") as ws:
        ws.send_text(json.dumps({"action": "pause", "file_ids": [1]}))
        msg = ws.receive_json()
    assert msg["type"] == "error"
    assert "expected action run" in msg.get("message", "").lower()


def test_ws_rejects_file_ids_not_array(ws_validation_client):
    client = ws_validation_client
    with client.websocket_connect("/api/tagger/ws/progress") as ws:
        ws.send_text(json.dumps({"file_ids": "not-a-list"}))
        msg = ws.receive_json()
    assert msg["type"] == "error"
    assert msg.get("code") == "invalid_file_ids"


def test_ws_rejects_file_ids_non_integer(ws_validation_client):
    client = ws_validation_client
    with client.websocket_connect("/api/tagger/ws/progress") as ws:
        ws.send_text(json.dumps({"file_ids": ["x"]}))
        msg = ws.receive_json()
    assert msg["type"] == "error"
    assert msg.get("code") == "invalid_file_ids"


def test_ws_rejects_empty_file_ids(ws_validation_client):
    client = ws_validation_client
    with client.websocket_connect("/api/tagger/ws/progress") as ws:
        ws.send_text(json.dumps({"file_ids": []}))
        msg = ws.receive_json()
    assert msg["type"] == "error"
    assert msg.get("code") == "empty_queue"

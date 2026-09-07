"""Face detect WebSocket progress forwarding."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

pytestmark = [pytest.mark.full, pytest.mark.ws]

import backend.config as config_module
import backend.routes.face as face_routes
from backend.face.service import FaceTaggingService


def _recv_until(ws, types: set[str], limit: int = 30):
    for _ in range(limit):
        msg = ws.receive_json()
        if msg.get("type") in types:
            return msg
    raise AssertionError(f"No message in {types!r} within {limit} receives")


def _collect_progress(ws, terminal: str, limit: int = 40):
    progress = []
    for _ in range(limit):
        msg = ws.receive_json()
        if msg.get("type") == "progress":
            progress.append(msg)
        if msg.get("type") == terminal:
            return progress, msg
    raise AssertionError(f"Terminal {terminal!r} not received")


@pytest.fixture
def face_ws_client(monkeypatch, test_config):
    config_module._config = test_config
    FaceTaggingService._instance = None

    async def fake_detect_batch(
        self,
        client,
        *,
        file_ids,
        service_key,
        det_threshold=None,
        replace_existing=False,
        progress_cb=None,
        cancel_event=None,
    ):
        total = len(file_ids)
        if progress_cb:
            await progress_cb(
                {
                    "type": "progress",
                    "processed": 0,
                    "total": total,
                    "phase": "metadata",
                    "detail": "Fetching file metadata from Hydrus…",
                }
            )
            await progress_cb(
                {
                    "type": "progress",
                    "processed": 0,
                    "total": total,
                    "phase": "model",
                    "detail": "Warming up face inference (first GPU run may compile kernels) (3s)",
                    "active_provider": "MIGraphXExecutionProvider",
                }
            )
            await progress_cb(
                {
                    "type": "progress",
                    "processed": 0,
                    "total": total,
                    "phase": "detect",
                    "detail": "Processing file 1/2…",
                    "active_provider": "MIGraphXExecutionProvider",
                }
            )
            await progress_cb(
                {
                    "type": "progress",
                    "processed": 1,
                    "total": total,
                    "phase": "detect",
                    "face_count": 0,
                    "skipped": True,
                    "detail": "File 1/2 · skipped",
                    "active_provider": "MIGraphXExecutionProvider",
                }
            )
        return [
            {"file_id": file_ids[0], "face_count": 0, "skipped": True},
            {"file_id": file_ids[1], "face_count": 1, "skipped": False},
        ]

    monkeypatch.setattr(FaceTaggingService, "detect_batch", fake_detect_batch)

    hydrus = MagicMock()
    hydrus.get_services = AsyncMock(
        return_value={
            "local_tags": [{"name": test_config.target_tag_service, "service_key": "sk-test"}],
            "all_known_tags": [],
        }
    )
    monkeypatch.setattr(face_routes, "HydrusClient", lambda *a, **k: hydrus)

    from backend.app import app

    with TestClient(app) as client:
        yield client


def test_face_ws_rejects_invalid_first_message(face_ws_client):
    with face_ws_client.websocket_connect("/api/face/ws/progress") as ws:
        ws.send_json({"action": "pause"})
        msg = ws.receive_json()
        assert msg["type"] == "error"


def test_face_ws_rejects_empty_file_ids(face_ws_client):
    with face_ws_client.websocket_connect("/api/face/ws/progress") as ws:
        ws.send_json({"action": "run", "file_ids": []})
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "file_ids" in msg.get("error", "").lower()


def test_face_ws_forwards_progress_phases_with_detail(face_ws_client):
    with face_ws_client.websocket_connect("/api/face/ws/progress") as ws:
        ws.send_json({"action": "run", "file_ids": [101, 102], "service_key": "sk-test"})
        started = _recv_until(ws, {"started"})
        assert started["total"] == 2

        progress, terminal = _collect_progress(ws, "complete")
        assert terminal["files_processed"] == 2
        assert terminal["faces_found"] == 1

        phases = [p.get("phase") for p in progress]
        assert "metadata" in phases
        assert "model" in phases
        assert "detect" in phases

        details = [p.get("detail", "") for p in progress]
        assert any("metadata" in d.lower() or "Metadata" in d for d in details)
        assert any("Warming up" in d for d in details)
        assert any("Processing file 1/2" in d for d in details)
        assert any(p.get("processed") == 1 for p in progress if p.get("phase") == "detect")


def test_face_ws_cancel_sends_stopped(face_ws_client, monkeypatch):
    async def slow_detect_batch(self, client, *, file_ids, service_key, progress_cb=None, cancel_event=None, **_kw):
        if progress_cb:
            await progress_cb(
                {
                    "type": "progress",
                    "processed": 0,
                    "total": len(file_ids),
                    "phase": "model",
                    "detail": "Loading…",
                }
            )
        if cancel_event:
            cancel_event.set()
        raise asyncio.CancelledError("user stop")

    monkeypatch.setattr(FaceTaggingService, "detect_batch", slow_detect_batch)

    with face_ws_client.websocket_connect("/api/face/ws/progress") as ws:
        ws.send_json({"action": "run", "file_ids": [1], "service_key": "sk-test"})
        _recv_until(ws, {"started"})
        ws.send_json({"action": "cancel"})
        stopping = _recv_until(ws, {"stopping", "stopped"})
        if stopping["type"] == "stopping":
            terminal = _recv_until(ws, {"stopped"})
        else:
            terminal = stopping
        assert terminal["type"] == "stopped"

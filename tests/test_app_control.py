"""POST /api/app/shutdown and tagging session registry."""

import asyncio

import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]

from fastapi.testclient import TestClient

import backend.config as config_module
import backend.routes.app_control as app_control
from backend.services.tagging_session_registry import (
    TaggingSessionHandle,
    active_tagging_sessions_count,
    announce_shutdown_to_tagging_sessions,
    register_shutdown_notifier,
    register_tagging_session,
    signal_all_sessions_cancel,
    signal_all_sessions_flush,
    unregister_shutdown_notifier,
    unregister_tagging_session,
)


def test_registry_flush_cancel_and_count():
    c = asyncio.Event()
    f = asyncio.Event()
    h = TaggingSessionHandle(cancel_event=c, flush_event=f)
    register_tagging_session(h)
    assert active_tagging_sessions_count() == 1
    assert signal_all_sessions_flush() == 1
    assert f.is_set()
    assert signal_all_sessions_cancel() == 1
    assert c.is_set()
    unregister_tagging_session(h)
    assert active_tagging_sessions_count() == 0


@pytest.mark.asyncio
async def test_registry_announce_shutdown():
    n_calls = 0

    async def notifier():
        nonlocal n_calls
        n_calls += 1

    register_shutdown_notifier(notifier)
    assert await announce_shutdown_to_tagging_sessions() == 1
    assert n_calls == 1
    unregister_shutdown_notifier(notifier)


@pytest.fixture
def app_control_client(monkeypatch):
    monkeypatch.setattr(app_control, "_schedule_process_exit", lambda _d=0.6: None)
    from backend.app import app

    with TestClient(app) as client:
        yield client


def test_get_app_status(app_control_client):
    r = app_control_client.get("/api/app/status")
    assert r.status_code == 200
    data = r.json()
    assert data["success"] is True
    assert "active_tagging_sessions" in data
    assert "loaded_model" in data
    assert "models_dir" in data


def test_shutdown_forbidden_when_disabled(monkeypatch, app_control_client):
    snap = config_module.get_config()
    off = snap.model_copy(update={"allow_ui_shutdown": False})
    monkeypatch.setattr(config_module, "get_config", lambda: off)
    monkeypatch.setattr(app_control, "get_config", lambda: off)

    r = app_control_client.post("/api/app/shutdown")
    assert r.status_code == 403
    assert r.json()["success"] is False


def test_shutdown_returns_metrics(monkeypatch, app_control_client):
    snap = config_module.get_config()
    on = snap.model_copy(update={"allow_ui_shutdown": True, "shutdown_tagging_grace_seconds": 0.01})
    monkeypatch.setattr(config_module, "get_config", lambda: on)
    monkeypatch.setattr(app_control, "get_config", lambda: on)

    r = app_control_client.post("/api/app/shutdown")
    assert r.status_code == 200
    data = r.json()
    assert data["success"] is True
    mt = data["metrics"]
    assert "flush_signaled_sessions" in mt
    assert "cancel_signaled_sessions" in mt
    assert mt.get("onnx_released") is True

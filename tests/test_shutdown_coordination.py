"""Coordinated tagging shutdown (UI + lifespan)."""

import logging
from unittest.mock import AsyncMock

import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]

import backend.config as config_module
from backend.shutdown_coordination import (
    reset_coordinated_tagging_shutdown_for_tests,
    run_coordinated_tagging_shutdown,
)


@pytest.mark.asyncio
async def test_coordinated_shutdown_idempotent(monkeypatch):
    reset_coordinated_tagging_shutdown_for_tests()
    calls = {"unload": 0, "face_unload": 0}

    def fake_unload(cls):
        calls["unload"] += 1
        return None

    def fake_face_unload(cls):
        calls["face_unload"] += 1

    from backend.face import service as face_service
    from backend.services import tagging_service as ts

    monkeypatch.setattr(ts.TaggingService, "unload_model_from_memory", classmethod(fake_unload))
    monkeypatch.setattr(face_service.FaceTaggingService, "unload_model_from_memory", classmethod(fake_face_unload))

    m1 = await run_coordinated_tagging_shutdown(reason="test_a")
    assert m1.get("skipped") is not True
    assert m1.get("completed") is True
    assert m1.get("face_model_released") is True

    m2 = await run_coordinated_tagging_shutdown(reason="test_b")
    assert m2.get("skipped") is True
    assert calls["unload"] == 1
    assert calls["face_unload"] == 1


@pytest.mark.asyncio
async def test_coordinated_shutdown_notifies_face_sessions(monkeypatch):
    reset_coordinated_tagging_shutdown_for_tests()
    notified = {"face": 0, "tagging": 0}

    async def fake_face_announce():
        notified["face"] += 1
        return 1

    async def fake_tagging_announce():
        notified["tagging"] += 1
        return 0

    monkeypatch.setattr(
        "backend.shutdown_coordination.announce_shutdown_to_face_sessions",
        fake_face_announce,
    )
    monkeypatch.setattr(
        "backend.shutdown_coordination.announce_shutdown_to_tagging_sessions",
        fake_tagging_announce,
    )
    monkeypatch.setattr("backend.shutdown_coordination.signal_all_sessions_flush", lambda: 0)
    monkeypatch.setattr("backend.shutdown_coordination.signal_all_sessions_cancel", lambda: 0)
    monkeypatch.setattr("backend.shutdown_coordination.signal_all_face_sessions_cancel", lambda: 2)

    from backend.face import service as face_service
    from backend.services import tagging_service as ts

    monkeypatch.setattr(ts.TaggingService, "unload_model_from_memory", classmethod(lambda cls: None))
    monkeypatch.setattr(face_service.FaceTaggingService, "unload_model_from_memory", classmethod(lambda cls: None))

    metrics = await run_coordinated_tagging_shutdown(reason="face_notify")
    assert notified["face"] == 1
    assert notified["tagging"] == 1
    assert metrics["face_cancel_signaled_sessions"] == 2


@pytest.mark.asyncio
async def test_coordinated_shutdown_grace_positive_logs_debug(monkeypatch, caplog):
    reset_coordinated_tagging_shutdown_for_tests()
    cfg = config_module.get_config().model_copy(update={"shutdown_tagging_grace_seconds": 1.5})
    monkeypatch.setattr("backend.shutdown_coordination.get_config", lambda: cfg)
    monkeypatch.setattr("backend.shutdown_coordination.asyncio.sleep", AsyncMock())

    def fake_unload(cls):
        return None

    from backend.face import service as face_service
    from backend.services import tagging_service as ts

    monkeypatch.setattr(ts.TaggingService, "unload_model_from_memory", classmethod(fake_unload))
    monkeypatch.setattr(face_service.FaceTaggingService, "unload_model_from_memory", classmethod(lambda cls: None))

    caplog.set_level(logging.DEBUG, logger="backend.shutdown_coordination")
    await run_coordinated_tagging_shutdown(reason="grace_dbg")
    assert "waiting grace" in caplog.text

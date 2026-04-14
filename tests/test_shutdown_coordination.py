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
    calls = {"unload": 0}

    def fake_unload(cls):
        calls["unload"] += 1
        return None

    from backend.services import tagging_service as ts

    monkeypatch.setattr(ts.TaggingService, "unload_model_from_memory", classmethod(fake_unload))

    m1 = await run_coordinated_tagging_shutdown(reason="test_a")
    assert m1.get("skipped") is not True
    assert m1.get("completed") is True

    m2 = await run_coordinated_tagging_shutdown(reason="test_b")
    assert m2.get("skipped") is True
    assert calls["unload"] == 1


@pytest.mark.asyncio
async def test_coordinated_shutdown_grace_positive_logs_debug(monkeypatch, caplog):
    reset_coordinated_tagging_shutdown_for_tests()
    cfg = config_module.get_config().model_copy(update={"shutdown_tagging_grace_seconds": 1.5})
    monkeypatch.setattr("backend.shutdown_coordination.get_config", lambda: cfg)
    monkeypatch.setattr("backend.shutdown_coordination.asyncio.sleep", AsyncMock())

    def fake_unload(cls):
        return None

    from backend.services import tagging_service as ts

    monkeypatch.setattr(ts.TaggingService, "unload_model_from_memory", classmethod(fake_unload))

    caplog.set_level(logging.DEBUG, logger="backend.shutdown_coordination")
    await run_coordinated_tagging_shutdown(reason="grace_dbg")
    assert "waiting grace" in caplog.text

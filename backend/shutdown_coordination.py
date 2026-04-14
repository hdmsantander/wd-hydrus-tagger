"""Coordinated tagging teardown: UI shutdown, Ctrl+C / SIGTERM, and FastAPI lifespan exit.

Runs the same sequence: notify active WebSocket sessions (``server_shutting_down``), signal
flush on all sessions, optional grace wait, signal cancel, unload ONNX. Safe to call multiple
times — subsequent calls are no-ops so UI-triggered shutdown and lifespan exit do not double
flush or double-release.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from backend.config import get_config
from backend.services.tagging_service import TaggingService
from backend.services.tagging_session_registry import (
    active_tagging_sessions_count,
    announce_shutdown_to_tagging_sessions,
    signal_all_sessions_cancel,
    signal_all_sessions_flush,
)

log = logging.getLogger(__name__)

_COORDINATED_TAGGING_SHUTDOWN_DONE = False
_LAST_COORDINATED_METRICS: dict | None = None


def reset_coordinated_tagging_shutdown_for_tests() -> None:
    """Reset idempotency state between tests."""
    global _COORDINATED_TAGGING_SHUTDOWN_DONE, _LAST_COORDINATED_METRICS
    _COORDINATED_TAGGING_SHUTDOWN_DONE = False
    _LAST_COORDINATED_METRICS = None


def last_coordinated_shutdown_metrics() -> dict | None:
    """Copy of metrics from the last successful coordinated run, or None."""
    if _LAST_COORDINATED_METRICS is None:
        return None
    return dict(_LAST_COORDINATED_METRICS)


async def run_coordinated_tagging_shutdown(*, reason: str) -> dict:
    """Notify WS clients, flush Hydrus queues, wait, cancel runs, release ONNX in RAM.

    Tagging loops drain ``flush_event`` between batches and run a final Hydrus flush after
    cancel; signaling cancel ends inference after the current batch boundary where possible.
    """
    global _COORDINATED_TAGGING_SHUTDOWN_DONE, _LAST_COORDINATED_METRICS

    if _COORDINATED_TAGGING_SHUTDOWN_DONE:
        log.debug(
            "coordinated_tagging_shutdown skipped: already completed (reason=%s)",
            reason,
        )
        return {"skipped": True, "reason": reason}

    config = get_config()
    active_before = active_tagging_sessions_count()
    metrics: dict = {
        "reason": reason,
        "active_tagging_sessions_before": active_before,
        "shutdown_notified_sessions": 0,
        "flush_signaled_sessions": 0,
        "cancel_signaled_sessions": 0,
        "onnx_released": False,
        "previous_loaded_model": None,
        "models_dir": str(Path(config.models_dir).resolve()),
    }

    log.info(
        "coordinated_tagging_shutdown begin reason=%s active_ws_sessions=%s grace_s=%.2f",
        reason,
        active_before,
        float(config.shutdown_tagging_grace_seconds),
    )

    metrics["shutdown_notified_sessions"] = await announce_shutdown_to_tagging_sessions()
    metrics["flush_signaled_sessions"] = signal_all_sessions_flush()
    log.info(
        "coordinated_tagging_shutdown phase1 WebSocket_notify=%s flush_signaled=%s grace_s=%.2f",
        metrics["shutdown_notified_sessions"],
        metrics["flush_signaled_sessions"],
        float(config.shutdown_tagging_grace_seconds),
    )

    grace = max(0.0, min(30.0, float(config.shutdown_tagging_grace_seconds)))
    if grace > 0:
        log.debug("coordinated_tagging_shutdown waiting grace_s=%.2f for in-flight batches", grace)
    await asyncio.sleep(grace)

    metrics["cancel_signaled_sessions"] = signal_all_sessions_cancel()
    log.info(
        "coordinated_tagging_shutdown phase2 cancel_signaled=%s",
        metrics["cancel_signaled_sessions"],
    )
    await asyncio.sleep(0.25)

    prev = TaggingService.unload_model_from_memory()
    metrics["onnx_released"] = True
    metrics["previous_loaded_model"] = prev
    log.info(
        "coordinated_tagging_shutdown phase3 ONNX released previous_model=%r models_dir=%s",
        prev,
        metrics["models_dir"],
    )

    _COORDINATED_TAGGING_SHUTDOWN_DONE = True
    metrics["completed"] = True
    _LAST_COORDINATED_METRICS = dict(metrics)
    return metrics

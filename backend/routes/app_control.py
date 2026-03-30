"""Process control: graceful shutdown from the UI."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import threading
import time
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.config import get_config
from backend.services.tagging_service import TaggingService
from backend.services.tagging_session_registry import (
    active_tagging_sessions_count,
    announce_shutdown_to_tagging_sessions,
    signal_all_sessions_cancel,
    signal_all_sessions_flush,
)

router = APIRouter()
log = logging.getLogger(__name__)


def _schedule_process_exit(delay_s: float = 0.6) -> None:
    """Ask uvicorn to shut down after the HTTP response is sent."""

    def worker() -> None:
        time.sleep(delay_s)
        log.info("app_shutdown: requesting process exit (SIGINT)")
        try:
            if hasattr(signal, "raise_signal"):
                signal.raise_signal(signal.SIGINT)
            else:
                os.kill(os.getpid(), signal.SIGINT)
        except Exception as e:
            log.warning("app_shutdown: signal failed (%s); using os._exit", e)
            os._exit(0)

    threading.Thread(target=worker, daemon=True).start()


@router.post("/shutdown")
async def shutdown_from_ui():
    """Graceful stop: notify tagging WebSockets, flush pending Hydrus writes, cancel runs, unload ONNX, exit."""
    config = get_config()
    if not config.allow_ui_shutdown:
        return JSONResponse(
            {"success": False, "error": "UI shutdown is disabled in config (allow_ui_shutdown: false)"},
            status_code=403,
        )

    metrics: dict = {
        "active_tagging_sessions_before": active_tagging_sessions_count(),
        "shutdown_notified_sessions": 0,
        "flush_signaled_sessions": 0,
        "cancel_signaled_sessions": 0,
        "onnx_released": False,
        "previous_loaded_model": None,
        "models_dir": str(Path(config.models_dir).resolve()),
    }

    metrics["shutdown_notified_sessions"] = await announce_shutdown_to_tagging_sessions()
    metrics["flush_signaled_sessions"] = signal_all_sessions_flush()
    log.info(
        "app_shutdown: phase1 notified=%s flush_signaled=%s (grace %.2fs)",
        metrics["shutdown_notified_sessions"],
        metrics["flush_signaled_sessions"],
        config.shutdown_tagging_grace_seconds,
    )

    grace = max(0.0, min(30.0, float(config.shutdown_tagging_grace_seconds)))
    await asyncio.sleep(grace)

    metrics["cancel_signaled_sessions"] = signal_all_sessions_cancel()
    log.info(
        "app_shutdown: phase2 cancel_signaled=%s",
        metrics["cancel_signaled_sessions"],
    )
    await asyncio.sleep(0.25)

    prev = TaggingService.unload_model_from_memory()
    metrics["onnx_released"] = True
    metrics["previous_loaded_model"] = prev
    log.info(
        "app_shutdown: ONNX released; on-disk models under %s unchanged",
        metrics["models_dir"],
    )

    _schedule_process_exit(0.5)

    return {
        "success": True,
        "message": "Shutdown scheduled: tagging sessions flushed/cancelled, model unloaded from RAM; process exits shortly.",
        "metrics": metrics,
    }


@router.get("/status")
async def app_status():
    """Lightweight health + tagging/model metrics for the UI."""
    config = get_config()
    svc = TaggingService.get_instance(config)
    return {
        "success": True,
        "active_tagging_sessions": active_tagging_sessions_count(),
        "loaded_model": svc._loaded_model,
        "models_dir": str(Path(config.models_dir).resolve()),
        "allow_ui_shutdown": config.allow_ui_shutdown,
        "shutdown_tagging_grace_seconds": config.shutdown_tagging_grace_seconds,
    }

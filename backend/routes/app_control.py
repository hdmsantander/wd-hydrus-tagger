"""Process control: graceful shutdown from the UI."""

from __future__ import annotations

import logging
import os
import signal
import threading
import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from pathlib import Path

from backend.config import get_config
from backend.services.tagging_service import TaggingService
from backend.services.tagging_session_registry import active_tagging_sessions_count
from backend.shutdown_coordination import run_coordinated_tagging_shutdown

router = APIRouter()
log = logging.getLogger(__name__)


def _schedule_process_exit(delay_s: float = 0.6) -> None:
    """Ask uvicorn to shut down after the HTTP response is sent."""

    def worker() -> None:
        time.sleep(delay_s)
        log.info("app_shutdown: requesting process exit via SIGINT (uvicorn graceful shutdown)")
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

    metrics = await run_coordinated_tagging_shutdown(reason="api_post_shutdown")

    _schedule_process_exit(0.5)

    return {
        "success": True,
        "message": "Shutdown scheduled: tagging sessions notified, flush/cancel signaled, model unloaded from RAM; process exits shortly.",
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

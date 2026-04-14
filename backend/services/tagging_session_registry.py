"""Track active tagging WebSocket sessions for coordinated graceful shutdown."""

from __future__ import annotations

import asyncio
import logging
import threading
import time

_log = logging.getLogger(__name__)
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

_lock = threading.Lock()
_sessions: list[TaggingSessionHandle] = []
_shutdown_notifiers: list[Callable[[], Awaitable[None]]] = []
_snapshot_lock = threading.Lock()
_public_snapshot: dict | None = None
_controller_paused = False


@dataclass(frozen=True)
class TaggingSessionHandle:
    cancel_event: asyncio.Event
    flush_event: asyncio.Event


def register_tagging_session(handle: TaggingSessionHandle) -> None:
    with _lock:
        _sessions.append(handle)


def unregister_tagging_session(handle: TaggingSessionHandle) -> None:
    with _lock:
        try:
            _sessions.remove(handle)
        except ValueError:
            pass
        idle = len(_sessions) == 0
    if idle:
        clear_tagging_public_snapshot()


def register_shutdown_notifier(notify: Callable[[], Awaitable[None]]) -> None:
    with _lock:
        _shutdown_notifiers.append(notify)


def unregister_shutdown_notifier(notify: Callable[[], Awaitable[None]]) -> None:
    with _lock:
        try:
            _shutdown_notifiers.remove(notify)
        except ValueError:
            pass


def active_tagging_sessions_count() -> int:
    with _lock:
        return len(_sessions)


def clear_tagging_public_snapshot() -> None:
    global _public_snapshot, _controller_paused
    with _snapshot_lock:
        _public_snapshot = None
        _controller_paused = False


def set_controller_paused(paused: bool) -> None:
    global _controller_paused, _public_snapshot
    with _snapshot_lock:
        _controller_paused = paused
        if _public_snapshot is not None:
            _public_snapshot = {**_public_snapshot, "paused": paused}


def update_tagging_public_snapshot(ws_payload: dict, *, model_name: str, total_files: int) -> None:
    """Last-known progress for other browser tabs (no result tensors / large arrays)."""
    global _public_snapshot
    typ = ws_payload.get("type")
    with _snapshot_lock:
        base: dict = dict(_public_snapshot) if _public_snapshot else {}
        base["updated_at"] = time.time()
        base["model_name"] = model_name
        base["total_files"] = total_files
        base["total"] = total_files
        base["paused"] = _controller_paused
        if typ in ("progress", "file"):
            base["phase"] = "tagging"
            for k in (
                "current",
                "total",
                "infer_total",
                "cumulative_inferred_non_skip",
                "in_marker_skip_tail",
                "progress_bar_current",
                "progress_bar_total",
                "inference_batch",
                "batch_inferred",
                "batch_skipped_inference",
                "batch_predicted",
                "batch_skipped_same_model_marker",
                "batch_skipped_higher_tier_model_marker",
                "batches_completed",
                "batches_total",
                "total_applied",
                "total_tags_written",
                "total_duplicates_skipped",
                "cumulative_skipped_same_model_marker",
                "cumulative_skipped_higher_tier_model_marker",
                "cumulative_wd_stale_markers_removed",
                "performance_tuning",
                "tuning_state",
                "calibration_phase",
            ):
                if k in ws_payload:
                    base[k] = ws_payload[k]
        elif typ == "queue_plan":
            base["phase"] = "tagging"
            for k in (
                "queue_total",
                "infer_total",
                "skip_same_marker",
                "skip_higher_tier",
                "missing_metadata",
                "infer_first",
                "metadata_chunk_used",
            ):
                if k in ws_payload:
                    base[k] = ws_payload[k]
        elif typ == "tags_applied":
            base["phase"] = "tagging"
            for k in (
                "total_applied",
                "total_tags_written",
                "total_duplicates_skipped",
                "pending_remaining",
            ):
                if k in ws_payload:
                    base[k] = ws_payload[k]
        elif typ == "stopping":
            base["phase"] = "stopping"
            base["stopping_source"] = "user"
            base["detail"] = str(ws_payload.get("message") or "")
            if "pending_hydrus_queue" in ws_payload:
                base["pending_hydrus_queue"] = ws_payload["pending_hydrus_queue"]
        elif typ == "server_shutting_down":
            base["phase"] = "stopping"
            base["stopping_source"] = "server"
            base["detail"] = str(ws_payload.get("message") or "")
        elif typ in ("complete", "stopped"):
            base["phase"] = "finishing"
            for k in (
                "total_processed",
                "total_applied",
                "total_tags_written",
                "total_duplicates_skipped",
                "batches_completed",
                "batches_total",
                "inference_batch",
                "pending_hydrus_files",
                "cumulative_skipped_same_model_marker",
                "cumulative_skipped_higher_tier_model_marker",
                "cumulative_wd_stale_markers_removed",
            ):
                if k in ws_payload:
                    base[k] = ws_payload[k]
        _public_snapshot = base


def get_public_session_status() -> dict:
    """REST helper: whether a tagging WebSocket session is active and last progress snapshot."""
    with _lock:
        active = len(_sessions) > 0
    with _snapshot_lock:
        snap = dict(_public_snapshot) if _public_snapshot else None
    return {"active": active, "snapshot": snap}


def signal_all_sessions_flush() -> int:
    """Set flush on every session (pending Hydrus queue drains at next await point)."""
    with _lock:
        handles = list(_sessions)
    for h in handles:
        h.flush_event.set()
    if handles:
        _log.debug("tagging_shutdown: flush_event set on %s session(s)", len(handles))
    return len(handles)


def signal_all_sessions_cancel() -> int:
    with _lock:
        handles = list(_sessions)
    for h in handles:
        h.cancel_event.set()
    if handles:
        _log.debug("tagging_shutdown: cancel_event set on %s session(s)", len(handles))
    return len(handles)


async def announce_shutdown_to_tagging_sessions() -> int:
    """Send server_shutting_down (or similar) on all registered notifiers."""
    with _lock:
        cbs: list[Callable[[], Awaitable[None]]] = list(_shutdown_notifiers)
    if not cbs:
        _log.debug("tagging_shutdown: no WebSocket shutdown notifiers registered")
        return 0
    _log.info(
        "tagging_shutdown: broadcasting server_shutting_down to %s active tagging session(s)",
        len(cbs),
    )
    await asyncio.gather(*[c() for c in cbs], return_exceptions=True)
    return len(cbs)

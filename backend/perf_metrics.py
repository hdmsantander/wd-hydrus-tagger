"""Lightweight performance summaries: log only at boundaries (no hot-path overhead).

Uses stdlib ``time.perf_counter`` and optional ``resource.getrusage`` at shutdown.
Thread-safe counters for cumulative tagging totals (updated once per session).
"""

from __future__ import annotations

import logging
import sys
import threading
import time

from backend.log_stats import log_stats

_log = logging.getLogger(__name__)

_lock = threading.Lock()
_process_start_s: float | None = None
_WS_VALIDATION_OUTCOMES = frozenset({"empty_queue", "invalid_request"})

_totals: dict[str, int] = {
    "tagging_sessions": 0,
    "files_processed": 0,
    "outer_batches": 0,
    "ws_validation_rejects": 0,
}


def mark_process_start() -> None:
    """Call once when the app is ready (e.g. FastAPI lifespan startup)."""
    global _process_start_s
    with _lock:
        _process_start_s = time.perf_counter()


def peak_rss_mb() -> float | None:
    """Best-effort peak RSS; Linux ru_maxrss is KiB, macOS is bytes."""
    try:
        import resource

        raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":
            return round(raw / (1024 * 1024), 2)
        return round(raw / 1024, 2)
    except Exception:
        return None


def log_process_shutdown() -> None:
    """Log one INFO line at process teardown (uptime, cumulative tagging, optional RSS)."""
    with _lock:
        t0 = _process_start_s
        snap = _totals.copy()
    if t0 is None:
        uptime_s = 0.0
    else:
        uptime_s = time.perf_counter() - t0
    rss = peak_rss_mb()
    parts = [
        f"uptime_s={uptime_s:.3f}",
        f"tagging_sessions={snap['tagging_sessions']}",
        f"ws_validation_rejects={snap['ws_validation_rejects']}",
        f"files_tagged_total={snap['files_processed']}",
        f"tagging_outer_batches_total={snap['outer_batches']}",
    ]
    if rss is not None:
        parts.append(f"peak_rss_mb≈{rss}")
    _stats_kw: dict[str, object] = {
        "uptime_s": round(uptime_s, 6),
        "tagging_sessions": snap["tagging_sessions"],
        "ws_validation_rejects": snap["ws_validation_rejects"],
        "files_tagged_total": snap["files_processed"],
        "tagging_outer_batches_total": snap["outer_batches"],
    }
    if rss is not None:
        _stats_kw["peak_rss_mb"] = rss
    log_stats(_log, "process_shutdown", **_stats_kw)
    try:
        from backend.shutdown_coordination import last_coordinated_shutdown_metrics

        cm = last_coordinated_shutdown_metrics()
        if cm is not None and not cm.get("skipped"):
            parts.append(
                "shutdown_coordinated="
                f"sessions_before={cm.get('active_tagging_sessions_before')}"
                f" ws_notify={cm.get('shutdown_notified_sessions')}"
                f" flush={cm.get('flush_signaled_sessions')}"
                f" cancel={cm.get('cancel_signaled_sessions')}"
                f" reason={cm.get('reason')!r}"
            )
    except Exception:
        pass
    _log.info("perf process_shutdown %s", " ".join(parts))


def record_tagging_session(
    *,
    wall_s: float,
    model_prepare_wall_s: float,
    total_processed: int,
    batches_completed: int,
    total_applied: int,
    total_tags_written: int,
    stopped: bool,
    outcome: str,
    model_name: str,
) -> None:
    """One INFO line per WebSocket tagging session + update running totals.

    Validation-only rejects (``empty_queue``, ``invalid_request``) log ``outcome=…/skipped``
    and increment ``ws_validation_rejects`` only, not ``tagging_sessions`` / file totals.
    """
    with _lock:
        if outcome in _WS_VALIDATION_OUTCOMES:
            _totals["ws_validation_rejects"] += 1
        else:
            _totals["tagging_sessions"] += 1
            _totals["files_processed"] += max(0, int(total_processed))
            _totals["outer_batches"] += max(0, int(batches_completed))
    status = "stopped" if stopped else "complete"
    if outcome == "error":
        status = "error"
    elif outcome in _WS_VALIDATION_OUTCOMES:
        status = "skipped"
    _log.info(
        "perf tagging_session wall_s=%.3f model_prepare_s=%.3f processed=%s outer_batches=%s "
        "hydrus_files=%s tag_strings=%s outcome=%s/%s model=%s",
        wall_s,
        model_prepare_wall_s,
        total_processed,
        batches_completed,
        total_applied,
        total_tags_written,
        outcome,
        status,
        model_name,
    )
    log_stats(
        _log,
        "tagging_session",
        wall_s=round(wall_s, 6),
        model_prepare_s=round(model_prepare_wall_s, 6),
        processed=total_processed,
        outer_batches=batches_completed,
        hydrus_files=total_applied,
        tag_strings=total_tags_written,
        outcome=outcome,
        status=status,
        model=model_name,
    )


def log_predict_wall(*, wall_s: float, file_count: int, inference_batch: int) -> None:
    """Optional one-shot line for HTTP predict (no global counters)."""
    _log.info(
        "perf predict_batch wall_s=%.3f files=%s inference_batch=%s",
        wall_s,
        file_count,
        inference_batch,
    )
    log_stats(
        _log,
        "predict_http",
        wall_s=round(wall_s, 6),
        files=file_count,
        inference_batch=inference_batch,
    )


def log_apply_tags_http(
    *, wall_s: float, result_rows: int, files_written: int, dups_skipped: int
) -> None:
    """One line after POST /api/tagger/apply completes."""
    _log.info(
        "perf apply_tags_http wall_s=%.3f result_rows=%s files_written=%s dups_skipped=%s",
        wall_s,
        result_rows,
        files_written,
        dups_skipped,
    )
    log_stats(
        _log,
        "apply_tags_http",
        wall_s=round(wall_s, 6),
        result_rows=result_rows,
        files_written=files_written,
        dups_skipped=dups_skipped,
    )


def totals_snapshot() -> dict[str, int]:
    """For tests / diagnostics."""
    with _lock:
        return _totals.copy()


def reset_totals_for_tests() -> None:
    with _lock:
        _totals["tagging_sessions"] = 0
        _totals["files_processed"] = 0
        _totals["outer_batches"] = 0
        _totals["ws_validation_rejects"] = 0

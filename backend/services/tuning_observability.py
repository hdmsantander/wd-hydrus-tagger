"""T-A+ observability: performance tuning rolling window + end-of-run ``tuning_report`` (pure helpers)."""

from __future__ import annotations

DEFAULT_PERFORMANCE_TUNING_WINDOW = 32
MAX_PERFORMANCE_TUNING_WINDOW = 128


def clamp_performance_tuning_window(raw: object) -> int:
    """Client ``performance_tuning_window``; default 32, clamped 1–128."""
    if raw is None:
        return DEFAULT_PERFORMANCE_TUNING_WINDOW
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_PERFORMANCE_TUNING_WINDOW
    return max(1, min(MAX_PERFORMANCE_TUNING_WINDOW, n))


def merge_performance_tuning_row(
    batch_metric: dict,
    *,
    hydrus_apply_batch_s: float,
    effective_batch: int,
    download_parallel: int,
    peak_rss_mb_sample: float | None = None,
    ort_intra_op_threads: int | None = None,
    ort_inter_op_threads: int | None = None,
) -> dict:
    """One row for history + report (shallow copy of metric + WS fields)."""
    row = {
        **batch_metric,
        "hydrus_apply_batch_s": round(float(hydrus_apply_batch_s), 4),
        "effective_batch": int(effective_batch),
        "download_parallel": int(download_parallel),
    }
    if peak_rss_mb_sample is not None:
        row["peak_rss_mb_sample"] = round(float(peak_rss_mb_sample), 2)
    if ort_intra_op_threads is not None:
        row["ort_intra_op_threads"] = int(ort_intra_op_threads)
    if ort_inter_op_threads is not None:
        row["ort_inter_op_threads"] = int(ort_inter_op_threads)
    return row


def build_tuning_report(
    batch_series: list[dict],
    *,
    stopped: bool,
    batches_completed: int,
    total_processed: int,
    effective_batch: int,
    download_parallel: int,
    model_name: str,
    history_window: int,
    session_auto_tune: bool = False,
    tuning_control_mode: str | None = None,
    supervised_gates_passed: int = 0,
    autotune_summary: dict | None = None,
) -> dict:
    """Structured end-of-run payload for clients / operators (T-A+ / T-Auto)."""
    n = len(batch_series)
    sum_f = sum(float(x.get("fetch_s") or 0) for x in batch_series)
    sum_p = sum(float(x.get("predict_s") or 0) for x in batch_series)
    sum_a = sum(float(x.get("hydrus_apply_batch_s") or 0) for x in batch_series)
    wall = sum_f + sum_p + sum_a
    files_per_wall = (total_processed / wall) if wall > 0 else None
    out: dict = {
        "schema_version": 1,
        "performance_tuning": True,
        "session_auto_tune": bool(session_auto_tune),
        "stopped": stopped,
        "batches_completed": int(batches_completed),
        "batches_recorded": n,
        "total_processed": int(total_processed),
        "effective_batch": int(effective_batch),
        "hydrus_download_parallel": int(download_parallel),
        "model_name": model_name,
        "history_window": int(history_window),
        "aggregate": {
            "sum_fetch_s": round(sum_f, 4),
            "sum_predict_s": round(sum_p, 4),
            "sum_apply_s": round(sum_a, 4),
            "sum_wall_s": round(wall, 4),
            "avg_fetch_s": round(sum_f / n, 4) if n else 0.0,
            "avg_predict_s": round(sum_p / n, 4) if n else 0.0,
            "avg_apply_s": round(sum_a / n, 4) if n else 0.0,
            "files_per_wall_s": round(files_per_wall, 4) if files_per_wall is not None else None,
        },
        "batch_series": list(batch_series),
    }
    if session_auto_tune:
        out["tuning_control_mode"] = tuning_control_mode or "auto_lucky"
        out["supervised_gates_passed"] = int(supervised_gates_passed)
        if autotune_summary:
            out["autotune"] = dict(autotune_summary)
    return out

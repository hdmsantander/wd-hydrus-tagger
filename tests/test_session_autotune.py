"""Unit tests for session auto-tune coordinate descent (§4.2 / §4.4)."""

import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]

from backend.config import AppConfig
from backend.services.session_autotune import (
    DEFAULT_WARM_UP_BATCHES,
    SessionAutoTune,
    clamp_supervised_timeout_s,
    normalize_tuning_control_mode,
    resolve_intra_thread_bounds,
    resolve_tuning_bounds,
)


def _row(bs: int, dlp: int, fetch: float, predict: float, apply: float, files: int) -> dict:
    return {
        "batch_index": 1,
        "fetch_s": fetch,
        "predict_s": predict,
        "hydrus_apply_batch_s": apply,
        "effective_batch": bs,
        "download_parallel": dlp,
        "files_in_batch": files,
        "predict_queue": files,
    }


def test_tuning_state_includes_phase_title_and_detail():
    cfg = AppConfig()
    bounds, _ = resolve_tuning_bounds(
        cfg,
        {"batch_size": {"min": 4, "max": 4}, "hydrus_download_parallel": {"min": 8, "max": 8}},
    )
    t = SessionAutoTune(
        mode="auto_lucky",
        baseline_batch=4,
        baseline_dlp=8,
        bounds=bounds,
    )
    r = t.after_batch(_row(4, 8, 0.1, 0.2, 0.0, 4))
    assert "phase_title" in r.tuning_state
    assert "phase_detail" in r.tuning_state
    assert r.tuning_state["phase_title"]
    assert r.tuning_state["phase_detail"]
    assert r.tuning_state.get("underlying_phase") == "warm_up"


def test_ui_snapshot_commit_phase_shape():
    cfg = AppConfig()
    bounds, _ = resolve_tuning_bounds(cfg, None)
    t = SessionAutoTune(
        mode="auto_lucky",
        baseline_batch=8,
        baseline_dlp=8,
        bounds=bounds,
    )
    snap = t.ui_snapshot_commit_phase()
    assert snap["phase"] == "commit_apply"
    assert snap["calibration_segment"] == "commit"
    assert snap["phase_title"]
    assert snap["phase_detail"]
    assert snap["best_batch_size"] == t.best_pair[0]
    assert snap["next_batch_size"] == snap["best_batch_size"]
    assert snap["awaiting_approval"] is False


def test_default_warm_up_batches_is_three():
    assert DEFAULT_WARM_UP_BATCHES == 3


def test_session_autotune_default_warm_up_in_tuning_state():
    cfg = AppConfig()
    bounds, _ = resolve_tuning_bounds(
        cfg,
        {"batch_size": {"min": 4, "max": 4}, "hydrus_download_parallel": {"min": 8, "max": 8}},
    )
    t = SessionAutoTune(
        mode="auto_lucky",
        baseline_batch=4,
        baseline_dlp=8,
        bounds=bounds,
    )
    r = t.after_batch(_row(4, 8, 0.1, 0.1, 0.1, 4))
    assert r.tuning_state["warm_up_batches"] == 3


def test_resolve_tuning_bounds_clamps_to_global():
    cfg = AppConfig()
    b, w = resolve_tuning_bounds(
        cfg,
        {"batch_size": {"min": 2, "max": 400}, "hydrus_download_parallel": {"min": 1, "max": 99}},
    )
    assert b.batch_min == 2
    assert b.batch_max == 256
    assert b.dlp_min == 1
    assert b.dlp_max == 32
    assert len(w) >= 1


def test_normalize_tuning_control_mode():
    assert normalize_tuning_control_mode(None) == ("auto_lucky", False)
    assert normalize_tuning_control_mode("supervised") == ("supervised", False)
    assert normalize_tuning_control_mode("auto_lucky") == ("auto_lucky", False)
    m, bad = normalize_tuning_control_mode("nope")
    assert m == "auto_lucky" and bad is True


def test_clamp_supervised_timeout_s():
    assert clamp_supervised_timeout_s(None) is None
    assert clamp_supervised_timeout_s(10) == 30.0
    assert clamp_supervised_timeout_s(120) == 120.0


def test_auto_lucky_warm_up_then_explores():
    cfg = AppConfig()
    bounds, _ = resolve_tuning_bounds(cfg, {"batch_size": {"min": 4, "max": 4}, "hydrus_download_parallel": {"min": 8, "max": 8}})
    t = SessionAutoTune(
        mode="auto_lucky",
        baseline_batch=4,
        baseline_dlp=8,
        bounds=bounds,
        warm_up_batches=3,
    )
    # warm-up 1–2: stay baseline
    r1 = t.after_batch(_row(4, 8, 0.1, 0.1, 0.1, 4))
    assert r1.next_batch_size == 4
    assert r1.next_download_parallel == 8
    assert r1.tuning_state["phase"] == "warm_up"

    r2 = t.after_batch(_row(4, 8, 0.1, 0.1, 0.1, 4))
    assert r2.next_batch_size == 4
    assert r2.tuning_state["phase"] == "warm_up"

    # third warm-up batch transitions to explore_bs (single candidate → same size)
    r3 = t.after_batch(_row(4, 8, 0.1, 0.1, 0.1, 4))
    assert r3.next_batch_size == 4
    assert r3.tuning_state["phase"] in ("explore_bs", "awaiting_approval")

    # only one bs candidate (4) — one explore_bs batch then move to explore_dlp
    r4 = t.after_batch(_row(4, 8, 0.05, 0.05, 0.05, 4))
    assert r4.next_batch_size == 4
    assert r4.next_download_parallel == 8


def test_merge_progress_ui_fields_commit_segment_and_eta():
    cfg = AppConfig()
    bounds, _ = resolve_tuning_bounds(
        cfg,
        {"batch_size": {"min": 4, "max": 4}, "hydrus_download_parallel": {"min": 8, "max": 8}},
    )
    t = SessionAutoTune(
        mode="auto_lucky",
        baseline_batch=4,
        baseline_dlp=8,
        bounds=bounds,
        warm_up_batches=1,
    )
    r = t.after_batch(_row(4, 8, 1, 1, 0.5, 4))
    series = [_row(4, 8, 1, 1, 0.5, 4)]
    merged = t.merge_progress_ui_fields(r.tuning_state, series, commit_segment=False)
    assert merged["tuning_search_total"] == t.tuning_search_batches_planned()
    rem = merged["tuning_search_total"] - merged["tuning_search_done"]
    if rem > 0:
        assert merged["tuning_eta_seconds"] is not None

    snap = t.ui_snapshot_commit_phase()
    m2 = t.merge_progress_ui_fields(snap, [], commit_segment=True)
    assert m2["tuning_search_complete"] is True
    assert m2["tuning_eta_seconds"] is None


def test_merge_progress_ui_fields_no_eta_while_awaiting_approval():
    cfg = AppConfig()
    bounds, _ = resolve_tuning_bounds(
        cfg,
        {"batch_size": {"min": 2, "max": 8}, "hydrus_download_parallel": {"min": 8, "max": 8}},
    )
    t = SessionAutoTune(
        mode="supervised",
        baseline_batch=4,
        baseline_dlp=8,
        bounds=bounds,
        warm_up_batches=1,
    )
    r = t.after_batch(
        {**_row(4, 8, 0.1, 0.1, 0.1, 4), "ort_intra_op_threads": 4, "ort_inter_op_threads": 1},
    )
    merged = t.merge_progress_ui_fields(
        r.tuning_state,
        [_row(4, 8, 0.1, 0.1, 0.1, 4)],
        commit_segment=False,
    )
    assert merged["awaiting_approval"] is True
    assert merged["tuning_eta_seconds"] is None


def test_supervised_requires_ack_when_knobs_change():
    cfg = AppConfig()
    bounds, _ = resolve_tuning_bounds(
        cfg,
        {"batch_size": {"min": 2, "max": 8}, "hydrus_download_parallel": {"min": 8, "max": 8}},
    )
    t = SessionAutoTune(
        mode="supervised",
        baseline_batch=4,
        baseline_dlp=8,
        bounds=bounds,
        warm_up_batches=1,
    )
    r1 = t.after_batch(
        {
            **_row(4, 8, 0.1, 0.1, 0.1, 4),
            "ort_intra_op_threads": 4,
            "ort_inter_op_threads": 1,
        },
    )
    assert r1.require_ack_before_next is True
    assert r1.tuning_state["awaiting_approval"] is True


def test_resolve_intra_thread_bounds_defaults():
    cfg = AppConfig()
    lo, hi, _ = resolve_intra_thread_bounds(cfg, None, default_hi=None)
    assert 1 <= lo <= hi <= 64


def test_tune_threads_explore_intra_proposes_triplet():
    cfg = AppConfig()
    bounds, _ = resolve_tuning_bounds(
        cfg,
        {"batch_size": {"min": 4, "max": 4}, "hydrus_download_parallel": {"min": 8, "max": 8}},
    )
    t = SessionAutoTune(
        mode="auto_lucky",
        baseline_batch=4,
        baseline_dlp=8,
        bounds=bounds,
        warm_up_batches=1,
        tune_threads=True,
        baseline_ort_intra=4,
        baseline_ort_inter=1,
        intra_bounds=(2, 8),
    )
    # warm-up batch
    t.after_batch({**_row(4, 8, 0.1, 0.1, 0.1, 4), "ort_intra_op_threads": 4, "ort_inter_op_threads": 1})
    # explore_bs (single candidate)
    t.after_batch({**_row(4, 8, 0.1, 0.1, 0.1, 4), "ort_intra_op_threads": 4, "ort_inter_op_threads": 1})
    # explore_dlp (single candidate) → enter explore_intra
    r = t.after_batch({**_row(4, 8, 0.1, 0.1, 0.1, 4), "ort_intra_op_threads": 4, "ort_inter_op_threads": 1})
    assert r.next_ort_inter == 1
    assert r.next_ort_intra in (2, 4, 5, 8)
    assert t.phase == "explore_intra"

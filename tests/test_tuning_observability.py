"""T-A+ tuning observability helpers."""

import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]

from backend.services.tuning_observability import (
    build_tuning_report,
    clamp_performance_tuning_window,
    merge_performance_tuning_row,
)


def test_clamp_performance_tuning_window_defaults_and_bounds():
    assert clamp_performance_tuning_window(None) == 32
    assert clamp_performance_tuning_window("nope") == 32
    assert clamp_performance_tuning_window(1) == 1
    assert clamp_performance_tuning_window(200) == 128


def test_merge_performance_tuning_row():
    row = merge_performance_tuning_row(
        {"batch_index": 2, "fetch_s": 0.1, "predict_s": 0.2},
        hydrus_apply_batch_s=0.05,
        effective_batch=4,
        download_parallel=8,
        peak_rss_mb_sample=123.45,
    )
    assert row["batch_index"] == 2
    assert row["hydrus_apply_batch_s"] == 0.05
    assert row["effective_batch"] == 4
    assert row["download_parallel"] == 8
    assert row["peak_rss_mb_sample"] == 123.45


def test_merge_performance_tuning_row_ort_intra_only():
    row = merge_performance_tuning_row(
        {"batch_index": 1},
        hydrus_apply_batch_s=0.0,
        effective_batch=2,
        download_parallel=4,
        ort_intra_op_threads=6,
    )
    assert row["ort_intra_op_threads"] == 6
    assert "ort_inter_op_threads" not in row


def test_merge_performance_tuning_row_ort_inter_only():
    row = merge_performance_tuning_row(
        {"batch_index": 1},
        hydrus_apply_batch_s=0.0,
        effective_batch=2,
        download_parallel=4,
        ort_inter_op_threads=2,
    )
    assert row["ort_inter_op_threads"] == 2
    assert "ort_intra_op_threads" not in row


def test_merge_performance_tuning_row_optional_ort_and_rss_omitted():
    row = merge_performance_tuning_row(
        {"batch_index": 1, "fetch_s": 0.0, "predict_s": 0.0},
        hydrus_apply_batch_s=0.0,
        effective_batch=2,
        download_parallel=4,
    )
    assert "peak_rss_mb_sample" not in row
    assert "ort_intra_op_threads" not in row
    assert "ort_inter_op_threads" not in row


def test_build_tuning_report_includes_autotune_when_summary_present():
    r = build_tuning_report(
        [{"fetch_s": 0.1, "predict_s": 0.1, "hydrus_apply_batch_s": 0.0}],
        stopped=False,
        batches_completed=1,
        total_processed=2,
        effective_batch=2,
        download_parallel=8,
        model_name="m",
        history_window=8,
        session_auto_tune=True,
        autotune_summary={"phase": "hold", "best_batch_size": 8},
    )
    assert r["autotune"]["phase"] == "hold"


def test_build_tuning_report_session_auto_tune_without_autotune_summary_dict():
    r = build_tuning_report(
        [{"fetch_s": 0.1, "predict_s": 0.1, "hydrus_apply_batch_s": 0.0}],
        stopped=False,
        batches_completed=1,
        total_processed=4,
        effective_batch=4,
        download_parallel=8,
        model_name="m",
        history_window=8,
        session_auto_tune=True,
        tuning_control_mode="auto_lucky",
        supervised_gates_passed=0,
        autotune_summary=None,
    )
    assert r["session_auto_tune"] is True
    assert r["tuning_control_mode"] == "auto_lucky"
    assert "autotune" not in r


def test_build_tuning_report_empty_series():
    r = build_tuning_report(
        [],
        stopped=False,
        batches_completed=0,
        total_processed=0,
        effective_batch=8,
        download_parallel=8,
        model_name="m",
        history_window=32,
    )
    assert r["schema_version"] == 1
    assert r["batches_recorded"] == 0
    assert r["aggregate"]["sum_wall_s"] == 0.0
    assert r["batch_series"] == []
    assert r["session_auto_tune"] is False


def test_build_tuning_report_aggregates():
    series = [
        {
            "batch_index": 1,
            "fetch_s": 1.0,
            "predict_s": 2.0,
            "hydrus_apply_batch_s": 0.5,
        },
        {
            "batch_index": 2,
            "fetch_s": 1.0,
            "predict_s": 2.0,
            "hydrus_apply_batch_s": 0.5,
        },
    ]
    r = build_tuning_report(
        series,
        stopped=False,
        batches_completed=2,
        total_processed=10,
        effective_batch=5,
        download_parallel=8,
        model_name="wd-vit-tagger-v3",
        history_window=16,
    )
    assert r["batches_recorded"] == 2
    assert r["total_processed"] == 10
    agg = r["aggregate"]
    assert agg["sum_fetch_s"] == 2.0
    assert agg["sum_predict_s"] == 4.0
    assert agg["sum_apply_s"] == 1.0
    assert agg["sum_wall_s"] == 7.0
    assert agg["files_per_wall_s"] is not None
    assert abs(agg["files_per_wall_s"] - 10 / 7.0) < 0.01

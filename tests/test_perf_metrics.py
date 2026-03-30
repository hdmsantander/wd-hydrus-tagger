"""Lightweight perf metrics: totals and shutdown line (stdlib only)."""

import logging

import pytest

from backend import perf_metrics as pm


@pytest.fixture(autouse=True)
def reset_perf_totals():
    pm.reset_totals_for_tests()
    yield
    pm.reset_totals_for_tests()


def test_record_tagging_session_updates_totals():
    pm.record_tagging_session(
        wall_s=1.5,
        model_prepare_wall_s=0.1,
        total_processed=12,
        batches_completed=3,
        total_applied=10,
        total_tags_written=40,
        stopped=False,
        outcome="ok",
        model_name="wd-vit-tagger-v3",
    )
    snap = pm.totals_snapshot()
    assert snap["tagging_sessions"] == 1
    assert snap["files_processed"] == 12
    assert snap["outer_batches"] == 3

    pm.record_tagging_session(
        wall_s=0.5,
        model_prepare_wall_s=0.0,
        total_processed=4,
        batches_completed=1,
        total_applied=0,
        total_tags_written=0,
        stopped=True,
        outcome="ok",
        model_name="wd-vit-tagger-v3",
    )
    snap = pm.totals_snapshot()
    assert snap["tagging_sessions"] == 2
    assert snap["files_processed"] == 16
    assert snap["outer_batches"] == 4


def test_peak_rss_mb_returns_number_or_none():
    r = pm.peak_rss_mb()
    assert r is None or (isinstance(r, float) and r >= 0)


def test_log_process_shutdown_logs(caplog):
    caplog.set_level(logging.INFO, logger="backend.perf_metrics")
    pm.mark_process_start()
    pm.record_tagging_session(
        wall_s=0.01,
        model_prepare_wall_s=0.0,
        total_processed=1,
        batches_completed=1,
        total_applied=0,
        total_tags_written=0,
        stopped=False,
        outcome="ok",
        model_name="m",
    )
    pm.log_process_shutdown()
    joined = " ".join(r.message for r in caplog.records if r.name == "backend.perf_metrics")
    assert "perf process_shutdown" in joined
    assert "uptime_s=" in joined
    assert "tagging_sessions=1" in joined
    assert "files_tagged_total=1" in joined

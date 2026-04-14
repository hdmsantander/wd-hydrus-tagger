"""Lightweight perf metrics: totals and shutdown line (stdlib only)."""

import logging

import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]

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


def test_record_validation_reject_increments_ws_validation_rejects_only():
    pm.record_tagging_session(
        wall_s=0.0,
        model_prepare_wall_s=0.0,
        total_processed=99,
        batches_completed=9,
        total_applied=0,
        total_tags_written=0,
        stopped=False,
        outcome="empty_queue",
        model_name="m",
    )
    snap = pm.totals_snapshot()
    assert snap["tagging_sessions"] == 0
    assert snap["files_processed"] == 0
    assert snap["outer_batches"] == 0
    assert snap["ws_validation_rejects"] == 1

    pm.record_tagging_session(
        wall_s=0.0,
        model_prepare_wall_s=0.0,
        total_processed=0,
        batches_completed=0,
        total_applied=0,
        total_tags_written=0,
        stopped=False,
        outcome="invalid_request",
        model_name="m",
    )
    snap = pm.totals_snapshot()
    assert snap["ws_validation_rejects"] == 2


def test_record_validation_reject_logs_skipped_status(caplog):
    caplog.set_level(logging.INFO, logger="backend.perf_metrics")
    pm.record_tagging_session(
        wall_s=0.0,
        model_prepare_wall_s=0.0,
        total_processed=0,
        batches_completed=0,
        total_applied=0,
        total_tags_written=0,
        stopped=False,
        outcome="invalid_request",
        model_name="m",
    )
    joined = " ".join(r.message for r in caplog.records if r.name == "backend.perf_metrics")
    assert "outcome=invalid_request/skipped" in joined
    assert "stats op=tagging_session" in joined
    assert "outcome=invalid_request" in joined


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
    assert "ws_validation_rejects=0" in joined
    assert "files_tagged_total=1" in joined
    assert "stats op=process_shutdown" in joined

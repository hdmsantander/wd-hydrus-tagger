"""Tests for recognize progress registry (HTTP session polling)."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]

from backend.services.face_recognize_registry import (
    begin_recognize,
    begin_recognize_apply,
    end_recognize,
    get_recognize_status,
    update_recognize_apply,
    update_recognize_stage,
)


def test_recognize_registry_lifecycle():
    end_recognize()
    begin_recognize(stage_total=4, faces_in_db=100)
    st = get_recognize_status()["recognize"]
    assert st["active"] is True
    assert st["faces_in_db"] == 100
    assert st["stage_total"] == 4

    update_recognize_stage(
        stage_idx=2,
        step_label="Recognize 2/4 — Cluster (min 5 faces)",
        assigned=10,
        cumulative_assigned=25,
    )
    st = get_recognize_status()["recognize"]
    assert st["stage_idx"] == 2
    assert st["assigned"] == 25

    begin_recognize_apply(files_total=50, detail="Apply person tags")
    update_recognize_apply(25, "Apply person tags · 25/50 files")
    st = get_recognize_status()["recognize"]
    assert st["phase"] == "apply"
    assert st["files_done"] == 25
    assert st["files_total"] == 50

    end_recognize()
    assert get_recognize_status()["recognize"]["active"] is False

"""Face UI flow helpers and panel wiring (static + mirrored progress math)."""

from pathlib import Path

import pytest

pytestmark = [pytest.mark.full, pytest.mark.ui]

REPO = Path(__file__).resolve().parents[1]
FACE_JS = (REPO / "frontend" / "js" / "components" / "face.js").read_text(encoding="utf-8")
INDEX_HTML = (REPO / "frontend" / "index.html").read_text(encoding="utf-8")


def _face_progress_counts(msg: dict, file_total: int) -> dict:
    """Mirror of faceProgressCounts in face.js for regression tests."""
    total = max(1, msg.get("total") or file_total or 1)
    step_total = msg.get("step_total") or 5
    step_num = msg.get("step_num") or 1
    is_scan = msg.get("step") == "scan" or msg.get("phase") == "detect"
    if is_scan:
        pre_weight = (step_total - 1) / step_total
        scan_weight = 1 / step_total
        processed = msg.get("processed") or 0
        cur = pre_weight * total + (processed / total) * scan_weight * total
        return {"cur": min(total, round(cur)), "tot": total}
    cur = (step_num / step_total) * total
    return {"cur": max(0, round(cur)), "tot": total}


def test_face_progress_counts_pre_scan_advances():
    assert _face_progress_counts({"step_num": 1, "step_total": 5}, 100)["cur"] == 20
    assert _face_progress_counts({"step_num": 3, "step_total": 5}, 100)["cur"] == 60


def test_face_progress_counts_scan_blends_file_index():
    mid = _face_progress_counts(
        {"step": "scan", "step_total": 5, "processed": 50, "total": 100},
        100,
    )
    assert mid["tot"] == 100
    assert mid["cur"] == 90


def test_index_primary_pipeline_buttons():
    for needle in (
        'id="btn-face-run-pipeline-selected"',
        'id="btn-face-run-pipeline-all"',
        'id="face-pipeline-selected-count"',
        'id="face-last-run-summary"',
        'id="check-face-replace"',
        'class="face-advanced-actions"',
        'id="face-stats-grid"',
    ):
        assert needle in INDEX_HTML
    panel = INDEX_HTML.split('id="panel-face"', 1)[1].split("</section>", 1)[0]
    assert 'id="check-face-replace"' in panel
    assert INDEX_HTML.count('id="check-face-replace"') == 1
    assert 'id="check-face-replace-person-tags"' in panel
    assert 'check-face-replace-person-tags" checked' in INDEX_HTML
    assert 'id="face-incremental-hint"' in panel
    assert 'face-recognize-mode-full' in panel
    assert 'check-face-refine-incremental' not in INDEX_HTML


def test_settings_no_duplicate_replace_checkbox():
    settings = INDEX_HTML.split('<h3 class="settings-subhead">Face tagging</h3>', 1)[1]
    settings = settings.split('<h3 class="settings-subhead">', 1)[0]
    assert 'id="check-face-replace"' not in settings


def test_face_js_exports_pipeline_flow():
    for needle in (
        "export async function runFacePipeline",
        "export function runFaceDetect",
        "export async function runRecognize",
        "export function faceProgressCounts",
        "export function faceProgressStatsLine",
        "export function faceRecognizeProgressCounts",
        "export function faceRecognizeStatsLine",
        "faceSessionStatus",
        "export function renderFaceDbStats",
        "export function formatFaceStatusText",
        "export function faceHydrusApplyHint",
        "export function formatFaceIncrementalHint",
        "export function faceRecognizeMode",
        "refine_incremental",
        "recluster_all",
        "keepOverlay",
        "fromPipeline",
        "syncFaceActionButtons",
        "function setFaceLastRunSummary",
    ):
        assert needle in FACE_JS


def test_face_js_progress_detail_prefers_detail_over_step_label():
    assert "if (msg.detail) return String(msg.detail)" in FACE_JS
    assert "msg.step_label && body" not in FACE_JS


def test_face_js_exports_incremental_helpers():
    assert "export function formatFaceIncrementalHint" in FACE_JS
    assert "export function faceRecognizeMode" in FACE_JS
    assert "updateFaceIncrementalHint" in FACE_JS


def test_face_js_disables_recognize_without_faces():
    assert "hasFaces" in FACE_JS
    assert "#btn-face-recognize" in FACE_JS


def test_gallery_syncs_face_action_buttons():
    gallery = (REPO / "frontend" / "js" / "components" / "gallery.js").read_text(encoding="utf-8")
    assert "syncFaceActionButtons" in gallery
    assert "face-pipeline-selected-count" in gallery

"""Frontend checks for face tagging panel."""

from pathlib import Path

import pytest

pytestmark = [pytest.mark.full, pytest.mark.ui]

REPO = Path(__file__).resolve().parents[1]


def test_index_includes_face_panel():
    html = (REPO / "frontend" / "index.html").read_text(encoding="utf-8")
    assert "face-pipeline-steps" in html
    assert 'data-step="detect"' in html
    assert 'data-step="recognize"' in html
    for needle in (
        'id="panel-face"',
        'id="btn-face-detect-selected"',
        'id="btn-face-recognize"',
        "Face tagging",
        'id="face-model-list"',
        'id="btn-verify-face-model"',
        'id="slider-face-det"',
        'id="select-gpu-backend"',
    ):
        assert needle in html


def test_face_component_module_exists():
    assert (REPO / "frontend" / "js" / "components" / "face.js").is_file()


def test_face_js_handles_server_shutdown():
    js = (REPO / "frontend" / "js" / "components" / "face.js").read_text(encoding="utf-8")
    assert "onServerShuttingDown" in js
    assert "expectServerShutdownSoon" in js


def test_face_js_uses_shared_progress_overlay():
    js = (REPO / "frontend" / "js" / "components" / "face.js").read_text(encoding="utf-8")
    assert "showFaceProgress" in js
    assert "setProgressControlMode" in js
    assert "showProgress(total" in js


def test_face_js_progress_uses_ws_detail_for_all_phases():
    js = (REPO / "frontend" / "js" / "components" / "face.js").read_text(encoding="utf-8")
    assert "facePhaseLabel" in js
    assert "faceProgressDetail" in js
    assert "setFacePipelineStep" in js
    assert "msg.step_label" in js
    assert "video_excluded" in js
    assert "marker_present" in js
    assert "phase === 'metadata'" in js
    assert "phase === 'model'" in js
    assert "onProgress" in js
    assert "setProgressActivityPhase" in js
    assert "'load'" in js or '"load"' in js
    assert "'inference'" in js or '"inference"' in js


def test_face_api_ws_handles_progress_types():
    js = (REPO / "frontend" / "js" / "api.js").read_text(encoding="utf-8")
    assert "msg.type === 'progress'" in js
    assert "onProgress" in js
    assert "/api/face/ws/progress" in js


def test_face_api_settles_websocket_on_close():
    js = (REPO / "frontend" / "js" / "api.js").read_text(encoding="utf-8")
    assert "Face WebSocket closed before the run finished" in js
    assert "faceListModels" in js
    assert "faceVerifyModels" in js

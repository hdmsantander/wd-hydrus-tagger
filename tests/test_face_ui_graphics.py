"""Face panel graphics, accessibility, and CSS contract tests."""

from pathlib import Path

import pytest

pytestmark = [pytest.mark.full, pytest.mark.ui]

REPO = Path(__file__).resolve().parents[1]
INDEX = (REPO / "frontend" / "index.html").read_text(encoding="utf-8")
FACE_JS = (REPO / "frontend" / "js" / "components" / "face.js").read_text(encoding="utf-8")
STYLE = (REPO / "frontend" / "css" / "style.css").read_text(encoding="utf-8")
CONNECTION_JS = (REPO / "frontend" / "js" / "components" / "connection.js").read_text(encoding="utf-8")
SETTINGS_JS = (REPO / "frontend" / "js" / "components" / "settings.js").read_text(encoding="utf-8")


def _face_panel_html() -> str:
    return INDEX.split('id="panel-face"', 1)[1].split("</section>", 1)[0]


def test_face_panel_stat_grid_markup():
    panel = _face_panel_html()
    for needle in (
        'class="face-stats-grid"',
        'id="face-stat-faces"',
        'id="face-stat-persons"',
        'id="face-stat-unassigned"',
        'id="face-stat-files"',
        'aria-label="Face database statistics"',
        'class="face-stat-value"',
        'class="face-stat-label"',
    ):
        assert needle in panel


def test_face_panel_accessibility_ids():
    panel = _face_panel_html()
    assert 'id="face-pipeline-step-detect"' in panel
    assert 'id="face-pipeline-step-recognize"' in panel
    assert 'class="visually-hidden"' in panel
    assert 'id="face-db-stats"' in panel
    assert 'face-last-run-summary' in panel
    assert 'face-last-run-summary" role="status" aria-live="polite" hidden' in panel


def test_face_panel_no_inline_button_styles():
    panel = _face_panel_html()
    assert 'style="margin-top' not in panel


def test_face_panel_primary_actions_and_advanced_section():
    panel = _face_panel_html()
    assert 'class="face-btn-stack"' in panel
    assert 'class="face-advanced-actions-body"' in panel
    assert 'class="face-checkbox-label"' in panel
    assert 'btn-face-run-pipeline-selected' in panel
    assert 'btn-face-run-pipeline-all' in panel


def test_face_panel_hydrus_workflow_copy():
    panel = _face_panel_html()
    for needle in (
        'id="face-hydrus-workflow"',
        'How face tags reach Hydrus',
        'no pending queue',
        'person:p1',
        'tag siblings',
        'hydrus-ai-taggers',
        'ai face detected',
        'face not visible',
    ):
        assert needle in panel


def test_face_css_graphics_contract():
    for needle in (
        ".face-stats-grid",
        ".face-stat-value",
        ".face-stat--warn",
        ".face-pipeline-step.is-done",
        ".face-pipeline-step.is-active",
        ".face-last-run-summary[hidden]",
        ".face-checkbox-label",
        ".face-btn-stack",
        ".face-advanced-actions-body",
        ".face-hydrus-workflow",
        ".face-incremental-options",
        ".face-incremental-hint",
        ".face-radio-label",
        ".face-recognize-mode-fieldset",
        ".visually-hidden",
        "font-variant-numeric: tabular-nums",
    ):
        assert needle in STYLE, f"missing CSS rule: {needle!r}"


def test_face_js_renders_stats_and_aria():
    for needle in (
        "export function renderFaceDbStats",
        "export function formatFaceStatusText",
        "export function faceHydrusApplyHint",
        "setAttribute('aria-current', 'step')",
        "face-stat--warn",
        "el.hidden = false",
        "el.hidden = true",
    ):
        assert needle in FACE_JS, f"missing in face.js: {needle!r}"


def test_connection_refreshes_face_status_on_connect():
    assert "void refreshFaceStatus()" in CONNECTION_JS
    assert "refreshFaceStatus" in CONNECTION_JS


def test_settings_owns_face_slider_live_labels():
    assert "$('#slider-face-det')?.addEventListener('input'" in SETTINGS_JS
    assert "$('#slider-face-distance')?.addEventListener('input'" in SETTINGS_JS
    assert "$('#slider-face-det')?.addEventListener('input'" not in FACE_JS

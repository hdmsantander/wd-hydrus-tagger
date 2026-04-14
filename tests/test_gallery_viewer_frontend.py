"""Static checks for gallery viewer UX, incremental DOM patching, and shared Hydrus tag helpers."""

from pathlib import Path

import pytest

pytestmark = [pytest.mark.full, pytest.mark.ui]

_REPO = Path(__file__).resolve().parents[1]
_GALLERY = _REPO / "frontend" / "js" / "components" / "gallery.js"
_VIEWER = _REPO / "frontend" / "js" / "components" / "viewer.js"
_HYDRUS_JS = _REPO / "frontend" / "js" / "utils" / "hydrus.js"
_STYLE = _REPO / "frontend" / "css" / "style.css"


def test_gallery_uses_double_click_detail_and_incremental_page_key():
    text = _GALLERY.read_text(encoding="utf-8")
    for needle in (
        "event.detail >= 2",
        "data-gallery-page-key",
        "buildGalleryPageKey",
        "syncGalleryCard",
        "extractTagsByService",
        "attachViewerHoldGesture",
        "_suppressCardClickUntil",
        "prefetchGalleryThumbnails",
    ):
        assert needle in text, f"expected {needle!r} in gallery.js"


def test_gallery_sync_does_not_stack_duplicate_click_handler():
    """syncGalleryCard must not assign the legacy onevent property on the card (el() already addEventListener)."""
    text = _GALLERY.read_text(encoding="utf-8")
    assert "card" + ".onclick =" not in text, "use a single click listener from el(..., { onClick: ... }) only"


def test_style_includes_hold_retro_and_cinema_viewer_rules():
    css = _STYLE.read_text(encoding="utf-8")
    for needle in (
        "gallery-card--hold-armed",
        "gallery-hold-radar",
        "image-viewer-overlay--cinema",
        ".gallery-card--viewer-shake .thumb",
        "inset: 0",
        ".image-viewer-toolbar-cluster--nav",
        ".image-viewer-chip--wd-marker",
        ".gallery-card .thumb",
        "object-fit: contain",
        ".image-viewer-zoomport",
        ".btn-viewer-theater",
        "scrollbar-gutter: stable",
        ".image-viewer-zoom-range",
        "minmax(0, 1fr) minmax(220px, min(22vw, 360px))",
    ):
        assert needle in css, f"expected {needle!r} in style.css"


def test_viewer_phased_image_predict_apply_navigation():
    text = _VIEWER.read_text(encoding="utf-8")
    for needle in (
        "new Image()",
        "image-viewer-thumb",
        "image-viewer-full",
        "fetchPriority",
        "thumbEl.decode",
        "buf.decode",
        "applyViewerImageLayout",
        "adjustViewerZoomWheel",
        "ZOOM_SLIDER_MIN_PCT",
        "syncViewerZoomSlider",
        "btn-viewer-theater",
        "image-viewer-zoomport",
        "navigateViewer",
        "api.predict",
        "api.applyTags",
        "ArrowLeft",
        "ArrowRight",
        "_viewerDisplayedFileId",
        "image-viewer-overlay--cinema",
        "toggleCinemaMode",
        "wd_tagger_viewer_cinema",
        "buildWdModelMarkerFromConfig",
        "tagMatchesViewerWdMarker",
    ):
        assert needle in text, f"expected {needle!r} in viewer.js"


def test_image_viewer_html_has_zoom_shell_and_theater_control():
    html = (_REPO / "frontend" / "index.html").read_text(encoding="utf-8")
    for needle in (
        "image-viewer-zoomport",
        "image-viewer-zoom-inner",
        "btn-viewer-theater",
        "btn-viewer-zoom-in",
        "image-viewer-zoom-range",
        "Theater off",
        "image-viewer-toolbar-cluster--nav",
    ):
        assert needle in html, f"expected {needle!r} in index.html"
    assert "btn-viewer-cinema" not in html


def test_tagged_badge_icon_is_pseudo_on_slot_for_stable_layout():
    css = _STYLE.read_text(encoding="utf-8")
    assert "tagged-badge-icon-slot::before" in css
    assert r"content: '\2660'" in css


def test_hydrus_js_exports_tag_helpers():
    text = _HYDRUS_JS.read_text(encoding="utf-8")
    assert "export function extractTagsByService" in text
    assert "export function getTagsForService" in text

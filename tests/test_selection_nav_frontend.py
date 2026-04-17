"""Static checks for gallery viewer selection-only navigation helpers."""

from pathlib import Path

import pytest

pytestmark = [pytest.mark.full, pytest.mark.ui]

_REPO = Path(__file__).resolve().parents[1]
_SEL = _REPO / "frontend" / "js" / "utils" / "selection_nav.js"
_TOAST = _REPO / "frontend" / "js" / "components" / "gallery_selection_toast.js"


def test_selection_nav_exports_and_ls_key():
    text = _SEL.read_text(encoding="utf-8")
    for needle in (
        "export function orderedSelectedFileIds",
        "export function viewerUsesSelectionOnlyNav",
        "export function readGalleryViewerCycleSelection",
        "export function writeGalleryViewerCycleSelection",
        "wd_tagger_gallery_viewer_cycle_selection",
        "if (v === null) return true",
    ):
        assert needle in text, f"expected {needle!r} in selection_nav.js"


def test_selection_mode_toast_module():
    text = _TOAST.read_text(encoding="utf-8")
    assert "export function hideGallerySelectionModeToast" in text
    assert "export function showGallerySelectionModeToast" in text
    assert "gallery-selection-mode-toast" in text

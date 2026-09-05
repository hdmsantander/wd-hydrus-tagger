"""Frontend checks for face tagging panel."""

from pathlib import Path

import pytest

pytestmark = [pytest.mark.full, pytest.mark.ui]

REPO = Path(__file__).resolve().parents[1]


def test_index_includes_face_panel():
    html = (REPO / "frontend" / "index.html").read_text(encoding="utf-8")
    for needle in (
        'id="panel-face"',
        'id="btn-face-detect-selected"',
        'id="btn-face-recognize"',
        "Face tagging",
    ):
        assert needle in html


def test_face_component_module_exists():
    assert (REPO / "frontend" / "js" / "components" / "face.js").is_file()

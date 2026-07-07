"""Static checks for optional hydrus-web companion links."""

from pathlib import Path

import pytest

pytestmark = [pytest.mark.full, pytest.mark.ui]

REPO = Path(__file__).resolve().parents[1]
_HYDRUS_WEB = REPO / "frontend" / "js" / "utils" / "hydrus_web.js"
_INDEX = REPO / "frontend" / "index.html"
_SETTINGS = REPO / "frontend" / "js" / "components" / "settings.js"


def test_hydrus_web_util_resolves_pages_route():
    text = _HYDRUS_WEB.read_text(encoding="utf-8")
    for needle in (
        "export function hydrusWebLibraryUrl",
        "new URL('pages'",
        "export function syncHydrusWebToolbarLink",
        "#link-gallery-hydrus-web",
        "a.hidden = true",
    ):
        assert needle in text, f"expected {needle!r} in hydrus_web.js"


def test_index_has_hydrus_web_controls():
    html = _INDEX.read_text(encoding="utf-8")
    for needle in (
        'id="link-gallery-hydrus-web"',
        'id="btn-viewer-hydrus-web"',
        'id="input-hydrus-web-url"',
    ):
        assert needle in html, f"expected {needle!r} in index.html"


def test_settings_persists_hydrus_web_url():
    text = _SETTINGS.read_text(encoding="utf-8")
    for needle in (
        "hydrus_web_url",
        "input-hydrus-web-url",
        "syncHydrusWebToolbarLink",
        "persisted === false",
    ):
        assert needle in text, f"expected {needle!r} in settings.js"

"""Ensure the SPA frontend stays English-only (no CJK in shipped UI assets)."""

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FRONTEND = _REPO_ROOT / "frontend"

# CJK Unified Ideographs + extensions used for Chinese (and shared Kanji).
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")

_UI_EXTENSIONS = {".html", ".js", ".css"}


def _iter_frontend_ui_files():
    for path in sorted(_FRONTEND.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in _UI_EXTENSIONS:
            continue
        yield path


def _cjk_offenders(text: str, path: Path) -> list[str]:
    out: list[str] = []
    for i, line in enumerate(text.splitlines(), start=1):
        if _CJK_RE.search(line):
            out.append(f"{path.relative_to(_REPO_ROOT)}:{i}:{line.strip()[:120]}")
    return out


def test_frontend_ui_files_have_no_cjk_characters():
    """Browser UI (HTML/JS/CSS) must not contain Chinese or other CJK ideographs."""
    all_offenders: list[str] = []
    for path in _iter_frontend_ui_files():
        text = path.read_text(encoding="utf-8")
        all_offenders.extend(_cjk_offenders(text, path))
    assert not all_offenders, "CJK found in frontend:\n" + "\n".join(all_offenders)


def test_frontend_index_html_english_document_and_key_labels():
    """Smoke-check main shell: language, title, and primary controls are English."""
    index = _FRONTEND / "index.html"
    assert index.is_file()
    html = index.read_text(encoding="utf-8")
    assert 'lang="en"' in html
    assert "<title>WD Tagger for Hydrus</title>" in html
    for needle in (
        "Hydrus connection",
        "Connect",
        "Search",
        "Tagger",
        "General threshold",
        "Character threshold",
        "Tag service",
        "Settings",
        "Stop server",
        "Server",
        "View progress (read-only)",
        "Verify cached models",
        "Compare revision with Hugging Face",
        "Defaults",
        "Default model",
        "Default tag service name",
        "WD model marker",
        "Apply all tags (HTTP)",
        "Files per apply request",
        "Allow",
        "Shutdown grace",
        "Hydrus metadata chunk (file IDs per API call)",
        "Server unreachable",
        "Reload page",
        "id=\"app-shell\"",
        "Toggle sidebar",
        'id="app-sidebar"',
        'id="app-layout"',
        'id="sidebar-backdrop"',
        'id="results-run-summary"',
        'Performance tuning overlay',
        'check-performance-tuning-tag-all',
        'progress-perf-tuning',
        'heavier WD model marker',
        'check-wd-skip-higher-tier',
    ):
        assert needle in html, f"missing English UI string: {needle!r}"


def test_tagger_component_contains_results_summary_copy():
    """Run summary and apply-button disable logic ship English strings used after tagging."""
    path = _FRONTEND / "js/components/tagger.js"
    text = path.read_text(encoding="utf-8")
    for needle in (
        "formatRunSummary",
        "Nothing is pending on the selected tag service",
        "noManualApplyNeeded",
        "results-run-summary",
    ):
        assert needle in text, f"missing in tagger.js: {needle!r}"


def test_core_ui_javascript_modules_exist():
    """Guards against accidental removal of primary UI modules covered by the CJK scan."""
    for rel in (
        "js/components/connection.js",
        "js/components/gallery.js",
        "js/components/tagger.js",
        "js/components/settings.js",
        "js/components/progress.js",
        "js/api.js",
        "js/server_offline.js",
        "js/app.js",
        "js/layout.js",
    ):
        assert (_FRONTEND / rel).is_file(), f"missing frontend/{rel}"

"""scripts/analyze_tagging_log.py smoke test."""

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]

_REPO = Path(__file__).resolve().parents[1]


def test_analyze_tagging_log_writes_markdown(tmp_path):
    fixture = _REPO / "tests" / "fixtures" / "tagging_log_sample.log"
    out = tmp_path / "report.md"
    r = subprocess.run(
        [sys.executable, str(_REPO / "scripts" / "analyze_tagging_log.py"), str(fixture), "--out", str(out)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    text = out.read_text(encoding="utf-8")
    assert "# Tagging log analysis" in text
    assert "Summary table" in text
    assert "queue_analysis" in text.lower() or "infer=" in text
    # session_config: batch 16, apply 16, ORT 8/1 (fixture line order varies from regex)
    assert "16/16/8/1" in text

"""scripts/check_requirements.py pre-flight validation."""

import os

import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "check_requirements.py"


def _run_script(*, cwd: Path, env: dict | None = None) -> subprocess.CompletedProcess:
    e = os.environ.copy()
    e.pop("WD_TAGGER_CONFIG_PATH", None)
    e.pop("WD_TAGGER_CHECK_ROOT", None)
    if env:
        e.update(env)
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=cwd,
        env=e,
        capture_output=True,
        text=True,
    )


def test_check_requirements_passes_on_repo_layout():
    r = _run_script(cwd=REPO)
    assert r.returncode == 0, (r.stdout, r.stderr)


def test_check_requirements_fails_on_invalid_config(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("batch_size: not_an_int\n", encoding="utf-8")
    r = _run_script(
        cwd=REPO,
        env={"WD_TAGGER_CONFIG_PATH": str(bad)},
    )
    assert r.returncode != 0
    assert "invalid" in r.stderr.lower() or "batch_size" in r.stderr.lower()


def test_check_requirements_fails_without_run_py(tmp_path):
    r = _run_script(cwd=tmp_path, env={"WD_TAGGER_CHECK_ROOT": str(tmp_path)})
    assert r.returncode != 0
    assert "run.py" in r.stderr

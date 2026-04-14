"""Shell helper: no requirements check on help / run.py --help / --generate-config."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "wd-hydrus-tagger.sh"

pytestmark = [
    pytest.mark.full,
    pytest.mark.core,
    pytest.mark.skipif(not shutil.which("bash"), reason="bash required for wd-hydrus-tagger.sh tests"),
]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        env=env,
    )


@pytest.mark.parametrize(
    "argv",
    [
        ("help",),
        ("usage",),
        ("-h",),
        ("--help",),
    ],
)
def test_help_like_commands_never_run_requirements_check(argv):
    r = _run(*argv)
    assert r.returncode == 0
    out = r.stdout + r.stderr
    assert "Running requirements check" not in out


def test_run_with_runpy_help_skips_requirements_check():
    r = _run("run", "--help")
    assert r.returncode == 0
    combined = r.stdout + r.stderr
    assert "Running requirements check" not in combined
    assert "Skipping requirements check (run.py --help)" in r.stderr


def test_implicit_run_with_flag_help_skips_requirements_check():
    r = _run("--help")
    assert r.returncode == 0
    combined = r.stdout + r.stderr
    assert "Running requirements check" not in combined


def test_generate_config_first_token_skips_requirements_check():
    r = _run("--generate-config", "--help")
    assert "Running requirements check" not in (r.stdout + r.stderr)


def test_test_without_m_full_skips_requirements_check():
    """Plain ``test`` forwards to pytest only (no preflight)."""
    r = _run("test", "--collect-only", "-q", "--no-cov")
    out = r.stdout + r.stderr
    assert "Running requirements check:" not in out


def test_test_with_m_full_runs_requirements_check():
    """``test -m full`` runs check_requirements before pytest."""
    r = _run("test", "-m", "full", "--collect-only", "-q", "--no-cov")
    out = r.stdout + r.stderr
    assert "Running requirements check:" in out
    assert "check_requirements:" in out


def test_test_m_full_skip_req_check_skips_requirements():
    r = _run("test", "-m", "full", "--skip-req-check", "--collect-only", "-q", "--no-cov")
    out = r.stdout + r.stderr
    assert "Running requirements check:" not in out
    assert "skipping requirements check before test -m full" in out

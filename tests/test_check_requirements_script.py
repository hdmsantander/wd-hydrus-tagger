"""scripts/check_requirements.py pre-flight validation."""

import os
import sys

import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]
import subprocess
from pathlib import Path
from unittest.mock import patch

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


def test_check_requirements_subprocess_fails_use_gpu_without_gpu_provider(tmp_path):
    """Integration: CPU-only ORT installs should fail preflight when use_gpu is true."""
    sys.path.insert(0, str(REPO))
    from backend.tagger.providers import available_gpu_providers

    if available_gpu_providers():
        pytest.skip("host has a registered ONNX GPU execution provider")

    cfg = tmp_path / "gpu.yaml"
    cfg.write_text(
        "use_gpu: true\nhydrus_api_url: http://localhost:45869\nmodels_dir: ./models\n",
        encoding="utf-8",
    )
    r = _run_script(cwd=REPO, env={"WD_TAGGER_CONFIG_PATH": str(cfg)})
    assert r.returncode != 0
    assert "use_gpu" in r.stderr.lower()


def test_check_gpu_inference_fails_without_provider():
    sys.path.insert(0, str(REPO))
    from backend.config import AppConfig

    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from scripts.check_requirements import _check_gpu_inference

    cfg = AppConfig(use_gpu=True, hydrus_api_url="http://localhost:45869")
    with patch("backend.tagger.providers.available_gpu_providers", return_value=[]):
        assert _check_gpu_inference(cfg) is False


def test_check_gpu_inference_passes_with_provider():
    sys.path.insert(0, str(REPO))
    from backend.config import AppConfig
    from scripts.check_requirements import _check_gpu_inference

    cfg = AppConfig(use_gpu=True, hydrus_api_url="http://localhost:45869")
    with patch(
        "backend.tagger.providers.available_gpu_providers",
        return_value=["MIGraphXExecutionProvider"],
    ), patch(
        "backend.tagger.providers.build_execution_providers",
        return_value=["MIGraphXExecutionProvider", "CPUExecutionProvider"],
    ):
        assert _check_gpu_inference(cfg) is True

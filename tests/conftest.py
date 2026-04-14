"""Pytest fixtures.

Markers (see ``docs/TESTING.md``): ``core``, ``ws``, ``ui``, ``slow``, ``full``.
Every test module declares ``pytest.mark.full`` so ``pytest -m full`` runs the **complete** suite
(including ``slow``) with the default coverage gate.
"""

import os

import pytest

# Tests use tmp_path for ONNX fixtures; production coerces temp models_dir to ./models.
os.environ.setdefault("WD_TAGGER_ALLOW_TMP_MODELS_DIR", "1")
os.environ.setdefault("WD_TAGGER_SKIP_PERF_RESULTS_SAVE", "1")

import backend.config as config_module
from backend.config import AppConfig
from backend.perf_metrics import reset_totals_for_tests
from backend.services.tagging_service import TaggingService
from backend.shutdown_coordination import reset_coordinated_tagging_shutdown_for_tests


@pytest.fixture
def tmp_models_dir(tmp_path):
    d = tmp_path / "models"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def test_config(tmp_models_dir):
    return AppConfig(
        models_dir=str(tmp_models_dir),
        hydrus_api_key="test-key",
        hydrus_api_url="http://test.invalid",
        batch_size=4,
        default_model="wd-vit-tagger-v3",
    )


@pytest.fixture(autouse=True)
def isolated_config_yaml_on_disk(tmp_path, monkeypatch):
    """Send real load/save paths to a temp file so stray save_config never touches repo config.yaml."""
    p = tmp_path / "wd-tagger-pytest-config.yaml"
    p.write_text("", encoding="utf-8")
    monkeypatch.setenv("WD_TAGGER_CONFIG_PATH", str(p))


@pytest.fixture(autouse=True)
def isolate_config_and_service(monkeypatch, test_config):
    """Avoid reading repo config.yaml and reset tagging singleton each test."""
    monkeypatch.setattr(config_module, "_config", test_config)

    def _get_config():
        return config_module._config

    monkeypatch.setattr(config_module, "get_config", _get_config)
    monkeypatch.setattr(config_module, "load_config", lambda: config_module._config)
    TaggingService._instance = None
    reset_totals_for_tests()
    reset_coordinated_tagging_shutdown_for_tests()
    yield
    TaggingService._instance = None
    reset_totals_for_tests()
    reset_coordinated_tagging_shutdown_for_tests()

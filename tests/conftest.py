"""Pytest fixtures."""

import os

import pytest

# Tests use tmp_path for ONNX fixtures; production coerces temp models_dir to ./models.
os.environ.setdefault("WD_TAGGER_ALLOW_TMP_MODELS_DIR", "1")

import backend.config as config_module
from backend.config import AppConfig
from backend.perf_metrics import reset_totals_for_tests
from backend.services.tagging_service import TaggingService


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
def isolate_config_and_service(monkeypatch, test_config):
    """Avoid reading repo config.yaml and reset tagging singleton each test."""
    monkeypatch.setattr(config_module, "_config", test_config)

    def _get_config():
        return config_module._config

    monkeypatch.setattr(config_module, "get_config", _get_config)
    monkeypatch.setattr(config_module, "load_config", lambda: config_module._config)
    TaggingService._instance = None
    reset_totals_for_tests()
    yield
    TaggingService._instance = None
    reset_totals_for_tests()

"""Configuration model."""

import pytest
import yaml
from pydantic import ValidationError

pytestmark = [pytest.mark.full, pytest.mark.core]

from pathlib import Path

import backend.config as config_module
from backend.config import (
    AppConfig,
    _REPO_ROOT,
    apply_runtime_config_overrides,
    clamp_hydrus_metadata_chunk_size,
    config_example_yaml_path,
    config_yaml_path,
    path_is_ephemeral_models_location,
    resolved_models_dir,
    resolved_ort_profile_dir,
    stable_models_dir_for_config,
)


def test_app_config_defaults():
    c = AppConfig()
    assert c.target_tag_service == "local tags"
    assert c.shutdown_tagging_grace_seconds == 0.0
    assert c.batch_size == 8
    assert c.cpu_intra_op_threads == 8
    assert c.cpu_inter_op_threads == 1
    assert c.host == "0.0.0.0"
    assert c.models_dir == "./models"
    assert c.wd_skip_inference_if_marker_present is True
    assert c.wd_append_model_marker_tag is True
    assert c.ort_enable_profiling is False
    assert c.ort_profile_dir == "./ort_traces"
    assert c.max_learning_cached_files == 400_000


def test_resolved_ort_profile_dir_relative_to_repo():
    p = resolved_ort_profile_dir("./ort_traces")
    assert p == (_REPO_ROOT / "ort_traces").resolve()


def test_wd_tagger_ort_profiling_env_enables_flag(monkeypatch):
    monkeypatch.setenv("WD_TAGGER_ORT_PROFILING", "1")
    c = apply_runtime_config_overrides(AppConfig())
    assert c.ort_enable_profiling is True


def test_apply_runtime_config_overrides_no_env_unchanged(monkeypatch):
    monkeypatch.delenv("WD_TAGGER_ORT_PROFILING", raising=False)
    c = apply_runtime_config_overrides(AppConfig())
    assert c.ort_enable_profiling is False


def test_batch_size_bounds():
    with pytest.raises(ValidationError):
        AppConfig(batch_size=0)
    with pytest.raises(ValidationError):
        AppConfig(batch_size=300)


def test_none_prefix_coerced():
    c = AppConfig(general_tag_prefix=None)
    assert c.general_tag_prefix == ""


def test_hydrus_and_apply_defaults():
    c = AppConfig()
    assert c.hydrus_download_parallel == 8
    assert c.hydrus_metadata_chunk_size == 512
    assert c.tagging_skip_tail_batch_size == 512
    assert c.apply_tags_every_n == 8
    assert c.apply_tags_http_batch_size == 100


def test_apply_tags_every_n_and_http_batch_are_independent_semantics():
    """Stride for WebSocket incremental writes vs chunk size for POST /api/tagger/apply."""
    c = AppConfig()
    assert c.apply_tags_every_n != c.apply_tags_http_batch_size
    d = c.model_dump()
    assert "WebSocket tagging" in (AppConfig.model_fields["apply_tags_every_n"].description or "")
    assert "POST /api/tagger/apply" in (AppConfig.model_fields["apply_tags_http_batch_size"].description or "")


def test_clamp_hydrus_metadata_chunk_size():
    assert clamp_hydrus_metadata_chunk_size(512) == 512
    assert clamp_hydrus_metadata_chunk_size(32) == 32
    assert clamp_hydrus_metadata_chunk_size(2048) == 2048
    assert clamp_hydrus_metadata_chunk_size(10) == 32
    assert clamp_hydrus_metadata_chunk_size(9000) == 2048
    assert clamp_hydrus_metadata_chunk_size(None) == 512
    assert clamp_hydrus_metadata_chunk_size("not-a-number") == 512


def test_resolved_models_dir_relative_to_repo():
    r = resolved_models_dir("./models")
    assert Path(r).is_absolute()
    assert Path(r).resolve() == (_REPO_ROOT / "models").resolve()


def test_resolved_models_dir_absolute_unchanged(tmp_path):
    p = tmp_path / "m"
    p.mkdir()
    assert resolved_models_dir(str(p)) == str(p.resolve())


def test_path_is_ephemeral_detects_pytest_substring(tmp_path):
    assert path_is_ephemeral_models_location(tmp_path / "pytest-of-user" / "models")


def test_stable_models_dir_coerces_system_temp(monkeypatch):
    import os
    import tempfile

    monkeypatch.delenv("WD_TAGGER_ALLOW_TMP_MODELS_DIR", raising=False)
    td = Path(tempfile.gettempdir()).resolve() / f"wd-coerce-{os.getpid()}"
    td.mkdir(parents=True, exist_ok=True)
    try:
        out = Path(stable_models_dir_for_config(str(td))).resolve()
        assert out == (_REPO_ROOT / "models").resolve()
    finally:
        try:
            td.rmdir()
        except OSError:
            pass


def test_stable_models_dir_respects_allow_tmp_env(monkeypatch):
    import os
    import tempfile

    monkeypatch.setenv("WD_TAGGER_ALLOW_TMP_MODELS_DIR", "1")
    td = Path(tempfile.gettempdir()).resolve() / f"wd-keep-{os.getpid()}"
    td.mkdir(parents=True, exist_ok=True)
    try:
        out = Path(stable_models_dir_for_config(str(td))).resolve()
        assert out == td.resolve()
    finally:
        try:
            td.rmdir()
        except OSError:
            pass


def test_stable_models_dir_keeps_project_relative_models():
    out = Path(stable_models_dir_for_config("./models")).resolve()
    assert out == (_REPO_ROOT / "models").resolve()


def test_config_yaml_path_defaults_to_repo_root(monkeypatch):
    monkeypatch.delenv("WD_TAGGER_CONFIG_PATH", raising=False)
    assert config_yaml_path() == (_REPO_ROOT / "config.yaml").resolve()


def test_config_yaml_path_respects_env_override(tmp_path, monkeypatch):
    p = tmp_path / "custom.yaml"
    monkeypatch.setenv("WD_TAGGER_CONFIG_PATH", str(p))
    assert config_yaml_path() == p.resolve()


def test_config_example_yaml_path_is_repo_file():
    assert config_example_yaml_path() == (_REPO_ROOT / "config.example.yaml").resolve()


def test_save_config_writes_to_config_yaml_path_not_cwd(tmp_path, monkeypatch):
    """Regression: save_config must use repo-root path or WD_TAGGER_CONFIG_PATH, not cwd-relative."""
    p = tmp_path / "persisted.yaml"
    monkeypatch.setenv("WD_TAGGER_CONFIG_PATH", str(p))
    cfg = AppConfig(hydrus_api_key="save-test-key", hydrus_api_url="http://save-test.invalid")
    config_module.save_config(cfg)
    assert p.is_file()
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert data["hydrus_api_key"] == "save-test-key"
    assert data["hydrus_api_url"] == "http://save-test.invalid"

"""Configuration model."""

import pytest
from pydantic import ValidationError

from pathlib import Path

from backend.config import (
    AppConfig,
    _REPO_ROOT,
    path_is_ephemeral_models_location,
    resolved_models_dir,
    stable_models_dir_for_config,
)


def test_app_config_defaults():
    c = AppConfig()
    assert c.batch_size == 8
    assert c.cpu_intra_op_threads == 8
    assert c.cpu_inter_op_threads == 1
    assert c.host == "0.0.0.0"
    assert c.models_dir == "./models"
    assert c.wd_skip_inference_if_marker_present is True
    assert c.wd_append_model_marker_tag is True


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
    assert c.hydrus_metadata_chunk_size == 256
    assert c.apply_tags_every_n == 0


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

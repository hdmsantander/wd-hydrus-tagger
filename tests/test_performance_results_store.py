"""performance_results.yaml writer."""

import yaml

import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]

from backend.services.performance_results_store import performance_results_path, save_performance_results


def test_save_performance_results_writes_patch(tmp_path, monkeypatch):
    monkeypatch.delenv("WD_TAGGER_SKIP_PERF_RESULTS_SAVE", raising=False)
    out = tmp_path / "perf.yaml"
    monkeypatch.setenv("WD_TAGGER_PERF_RESULTS_PATH", str(out))
    save_performance_results(
        model_name="wd-vit-tagger-v3",
        best_batch=12,
        best_dlp=10,
        best_intra=6,
        best_inter=1,
        tune_threads=True,
        tuning_control_mode="supervised",
        autotune_phase="hold",
    )
    assert out.is_file()
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    patch = doc["last_success"]["config_patch"]
    assert patch["batch_size"] == 12
    assert patch["hydrus_download_parallel"] == 10
    assert patch["cpu_intra_op_threads"] == 6
    assert patch["cpu_inter_op_threads"] == 1


def test_save_respects_skip_env(tmp_path, monkeypatch):
    monkeypatch.setenv("WD_TAGGER_SKIP_PERF_RESULTS_SAVE", "1")
    monkeypatch.setenv("WD_TAGGER_PERF_RESULTS_PATH", str(tmp_path / "nope.yaml"))
    save_performance_results(
        model_name="m",
        best_batch=1,
        best_dlp=2,
        best_intra=3,
        best_inter=1,
        tune_threads=False,
        tuning_control_mode="auto_lucky",
        autotune_phase="hold",
    )
    assert not (tmp_path / "nope.yaml").exists()


def test_save_preserves_previous_last_success(tmp_path, monkeypatch):
    monkeypatch.delenv("WD_TAGGER_SKIP_PERF_RESULTS_SAVE", raising=False)
    out = tmp_path / "perf.yaml"
    monkeypatch.setenv("WD_TAGGER_PERF_RESULTS_PATH", str(out))
    save_performance_results(
        model_name="a",
        best_batch=4,
        best_dlp=4,
        best_intra=4,
        best_inter=1,
        tune_threads=False,
        tuning_control_mode="auto_lucky",
        autotune_phase="hold",
    )
    save_performance_results(
        model_name="b",
        best_batch=8,
        best_dlp=8,
        best_intra=8,
        best_inter=1,
        tune_threads=False,
        tuning_control_mode="auto_lucky",
        autotune_phase="hold",
    )
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert doc["last_success"]["model_name"] == "b"
    assert doc["previous_last_success"]["model_name"] == "a"


def test_performance_results_path_env_override(tmp_path, monkeypatch):
    p = str(tmp_path / "x.yaml")
    monkeypatch.setenv("WD_TAGGER_PERF_RESULTS_PATH", p)
    assert str(performance_results_path()) == p


def test_performance_results_path_default_is_repo_root(monkeypatch):
    monkeypatch.delenv("WD_TAGGER_PERF_RESULTS_PATH", raising=False)
    path = performance_results_path()
    assert path.name == "performance_results.yaml"


def test_save_ignores_corrupt_existing_yaml(tmp_path, monkeypatch):
    """Corrupt performance_results.yaml is ignored; no previous_last_success merged."""
    monkeypatch.delenv("WD_TAGGER_SKIP_PERF_RESULTS_SAVE", raising=False)
    out = tmp_path / "perf.yaml"
    out.write_text("{\nnot valid yaml or json", encoding="utf-8")
    monkeypatch.setenv("WD_TAGGER_PERF_RESULTS_PATH", str(out))
    save_performance_results(
        model_name="m",
        best_batch=2,
        best_dlp=2,
        best_intra=2,
        best_inter=1,
        tune_threads=False,
        tuning_control_mode="auto_lucky",
        autotune_phase="hold",
    )
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert doc["last_success"]["model_name"] == "m"
    assert "previous_last_success" not in doc

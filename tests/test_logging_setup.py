"""Logging path selection, run_id, and per-run file layout."""

import logging
import logging.handlers
import os

import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]

from backend.logging_setup import (
    DEFAULT_LOG_LEVEL,
    configure_logging,
    parse_level,
    parse_server_args,
)


@pytest.fixture(autouse=True)
def clear_root_handlers():
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
        try:
            h.close()
        except OSError:
            pass
    root.setLevel(logging.WARNING)
    yield
    for h in root.handlers[:]:
        root.removeHandler(h)
        try:
            h.close()
        except OSError:
            pass


def test_parse_level_default():
    assert parse_level("") == logging.INFO
    assert parse_level("debug") == logging.DEBUG


def test_parse_level_unknown_warns():
    with pytest.warns(UserWarning, match="unknown log level"):
        assert parse_level("not_a_real_level_xyz") == logging.INFO


def test_parse_server_args_default_log_level(monkeypatch):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("WD_TAGGER_LOG_LEVEL", raising=False)
    args = parse_server_args([])
    assert args.log_level.upper() == DEFAULT_LOG_LEVEL


def test_configure_per_run_creates_runs_file_and_run_id(tmp_path, monkeypatch):
    monkeypatch.delenv("WD_TAGGER_LOG_FILE", raising=False)
    monkeypatch.setenv("WD_TAGGER_LOG_RUNS_MAX", "0")
    log = logging.getLogger()
    path = configure_logging("INFO", log_file=None, repo_root=tmp_path, reset=True)
    assert path.parent.name == "runs"
    assert path.name.startswith("run-")
    assert path.suffix == ".log"
    assert path.is_file()
    assert os.environ.get("WD_TAGGER_RUN_ID") in path.name
    log.info("hello")
    text = path.read_text(encoding="utf-8")
    assert os.environ["WD_TAGGER_RUN_ID"] in text
    assert "hello" in text


def test_configure_explicit_path_uses_rotating(tmp_path, monkeypatch):
    """Explicit ``--log-file`` uses RotatingFileHandler (init line is INFO)."""
    monkeypatch.setenv("WD_TAGGER_LOG_RUNS_MAX", "0")
    target = tmp_path / "single.log"
    configure_logging("INFO", log_file=str(target), repo_root=tmp_path, reset=True)
    for h in logging.getLogger().handlers:
        if isinstance(h, logging.handlers.RotatingFileHandler):
            h.flush()
    text = target.read_text(encoding="utf-8")
    assert "mode=rotating" in text


def test_second_configure_returns_same_path_if_handlers_exist(tmp_path, monkeypatch):
    monkeypatch.delenv("WD_TAGGER_LOG_FILE", raising=False)
    monkeypatch.setenv("WD_TAGGER_LOG_RUNS_MAX", "0")
    p1 = configure_logging("INFO", log_file=None, repo_root=tmp_path, reset=True)
    p2 = configure_logging("INFO", log_file=None, repo_root=tmp_path, reset=False)
    assert p1.resolve() == p2.resolve()


def test_prune_removes_oldest_run_logs(tmp_path, monkeypatch):
    monkeypatch.delenv("WD_TAGGER_LOG_FILE", raising=False)
    monkeypatch.setenv("WD_TAGGER_LOG_RUNS_MAX", "1")
    runs = tmp_path / "logs" / "runs"
    runs.mkdir(parents=True)
    stale = runs / "run-19990101T000000-1.log"
    stale.write_text("legacy", encoding="utf-8")
    os.utime(stale, (1, 1))
    configure_logging("INFO", log_file=None, repo_root=tmp_path, reset=True)
    assert not stale.exists()
    assert list(runs.glob("run-*.log"))


def test_per_run_writes_latest_pointer(tmp_path, monkeypatch):
    monkeypatch.delenv("WD_TAGGER_LOG_FILE", raising=False)
    monkeypatch.setenv("WD_TAGGER_LOG_RUNS_MAX", "0")
    configure_logging("INFO", log_file=None, repo_root=tmp_path, reset=True)
    latest = tmp_path / "logs" / "latest.log"
    assert latest.exists()
    run_files = list((tmp_path / "logs" / "runs").glob("run-*.log"))
    assert len(run_files) == 1
    run_file = run_files[0]
    if latest.is_symlink():
        assert os.readlink(latest) == os.path.relpath(run_file, latest.parent)
    else:
        assert str(run_file.resolve()) in latest.read_text(encoding="utf-8")

"""Central logging: console + per-run file (or explicit rotating file), third-party noise control."""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

# Default log level when env and CLI omit it (shell script also exports INFO).
DEFAULT_LOG_LEVEL = "INFO"
# Max per-run log files to keep under logs/runs/ (0 = keep all).
_DEFAULT_RUNS_MAX = 30


def parse_level(name: str) -> int:
    """Map string to logging level; default INFO on garbage input."""
    n = (name or DEFAULT_LOG_LEVEL).strip().upper()
    level = getattr(logging, n, None)
    if isinstance(level, int):
        return level
    warnings.warn(
        f"unknown log level {name!r}; using INFO",
        UserWarning,
        stacklevel=2,
    )
    return logging.INFO


def _log_runs_max() -> int:
    raw = os.environ.get("WD_TAGGER_LOG_RUNS_MAX", str(_DEFAULT_RUNS_MAX)).strip()
    try:
        n = int(raw)
        return max(0, n)
    except ValueError:
        return _DEFAULT_RUNS_MAX


def _make_run_id() -> str:
    return f"{datetime.now().strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"


def _prune_old_run_logs(runs_dir: Path, keep: int) -> None:
    if keep <= 0 or not runs_dir.is_dir():
        return
    files = sorted(
        runs_dir.glob("run-*.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for p in files[keep:]:
        try:
            p.unlink()
        except OSError:
            pass


def _update_latest_pointer(logs_dir: Path, run_file: Path) -> None:
    """Symlink logs/latest.log → this run (relative). Fallback: one-line path file (e.g. Windows)."""
    latest = logs_dir / "latest.log"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
    except OSError:
        return
    try:
        latest.symlink_to(os.path.relpath(run_file, logs_dir))
    except OSError:
        try:
            latest.write_text(f"{run_file.resolve()}\n", encoding="utf-8")
        except OSError:
            pass


class _RunIdFilter(logging.Filter):
    def __init__(self, run_id: str) -> None:
        super().__init__()
        self.run_id = run_id or "—"

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = self.run_id
        return True


def parse_server_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse --log-level / --log-file; unknown args are ignored (kept on sys.argv for other tools)."""
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(
        prog="wd-hydrus-tagger",
        add_help=True,
        description="WD Hydrus Tagger server (use run.py or wd-hydrus-tagger.sh run).",
    )
    p.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", os.environ.get("WD_TAGGER_LOG_LEVEL", DEFAULT_LOG_LEVEL)),
        metavar="LEVEL",
        help=(
            "Python log level: DEBUG, INFO, WARNING, ERROR "
            f"(default: {DEFAULT_LOG_LEVEL} or env LOG_LEVEL / WD_TAGGER_LOG_LEVEL)"
        ),
    )
    p.add_argument(
        "--log-file",
        default=os.environ.get("WD_TAGGER_LOG_FILE") or None,
        metavar="PATH",
        help=(
            "Log file path (default: per-run file under logs/runs/; "
            "rotating single file if set; env WD_TAGGER_LOG_FILE)"
        ),
    )
    args, _unknown = p.parse_known_args(argv)
    return args


def _handler_log_path(handler: logging.Handler) -> Path | None:
    base = getattr(handler, "baseFilename", None)
    if isinstance(base, str) and base:
        return Path(base)
    return None


def configure_logging(
    level_str: str,
    *,
    log_file: str | None = None,
    repo_root: Path | None = None,
    reset: bool = False,
) -> Path:
    """Attach console + file handlers to the root logger.

    * No explicit ``log_file`` / ``WD_TAGGER_LOG_FILE``: one file per process under
      ``logs/runs/run-<timestamp>-<pid>.log``, symlink ``logs/latest.log``, optional
      prune (``WD_TAGGER_LOG_RUNS_MAX``, default 30).
    * Explicit path: classic ``RotatingFileHandler`` (10 MiB × 5) for a stable file.

    Sets ``WD_TAGGER_LOG_FILE`` and ``WD_TAGGER_RUN_ID`` when ``reset=True``.

    Returns the resolved log file path used (for startup banner).
    """
    level = parse_level(level_str)
    root = logging.getLogger()
    run_id = _make_run_id()

    if reset:
        for h in root.handlers[:]:
            root.removeHandler(h)
            try:
                h.close()
            except OSError:
                pass
    elif root.handlers:
        for h in root.handlers:
            p = _handler_log_path(h)
            if p is not None:
                os.environ["WD_TAGGER_LOG_FILE"] = str(p.resolve())
                return p
        for h in root.handlers[:]:
            root.removeHandler(h)
            try:
                h.close()
            except OSError:
                pass

    root.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s [%(run_id)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run_filter = _RunIdFilter(run_id)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(fmt)
    sh.addFilter(run_filter)
    root.addHandler(sh)

    path, use_rotation = _resolve_log_path(repo_root, log_file, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    if use_rotation:
        fh: logging.Handler = logging.handlers.RotatingFileHandler(
            path,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
    else:
        fh = logging.FileHandler(path, encoding="utf-8", mode="a")

    fh.setLevel(level)
    fh.setFormatter(fmt)
    fh.addFilter(run_filter)
    root.addHandler(fh)

    os.environ["WD_TAGGER_LOG_FILE"] = str(path.resolve())
    os.environ["WD_TAGGER_RUN_ID"] = run_id
    if not use_rotation:
        logs_dir = path.parent.parent
        if path.parent.name == "runs" and logs_dir.name == "logs":
            _update_latest_pointer(logs_dir, path)
        _prune_old_run_logs(path.parent, _log_runs_max())

    # App packages
    logging.getLogger("backend").setLevel(level)

    for name in (
        "httpx",
        "httpcore",
        "httpcore.connection",
        "httpcore.http11",
        "h11",
        "urllib3",
        "asyncio",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)

    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("PIL.Image").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    if level <= logging.DEBUG:
        logging.getLogger("uvicorn").setLevel(logging.INFO)
        logging.getLogger("uvicorn.error").setLevel(logging.INFO)
        for name in (
            "websockets",
            "websockets.client",
            "websockets.server",
            "websockets.protocol",
            "uvicorn.protocols.websockets",
            "uvicorn.protocols.websockets.websockets_impl",
        ):
            logging.getLogger(name).setLevel(logging.WARNING)

    access_env = os.environ.get("WD_TAGGER_LOG_ACCESS", "").strip().lower() in ("1", "true", "yes", "on")
    if access_env:
        logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    else:
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING if level >= logging.INFO else logging.DEBUG)

    logging.getLogger("starlette").setLevel(logging.WARNING)

    log = logging.getLogger(__name__)
    mode = "rotating" if use_rotation else "per_run"
    log.info(
        "Logging initialized run_id=%s effective=%s (requested=%s) file=%s mode=%s",
        run_id,
        logging.getLevelName(level),
        (level_str or "").strip().upper() or logging.getLevelName(level),
        path,
        mode,
    )
    for h in root.handlers:
        flush = getattr(h, "flush", None)
        if callable(flush):
            try:
                flush()
            except OSError:
                pass
    return path


def _resolve_log_path(
    repo_root: Path | None,
    log_file: str | None,
    run_id: str,
) -> tuple[Path, bool]:
    """Return (path, use_rotation). Caller passes ``log_file`` only (CLI/env merged in argparse or lifespan)."""
    root = repo_root or Path.cwd()
    explicit = (log_file or "").strip()

    if explicit:
        p = Path(explicit)
        resolved = p if p.is_absolute() else (root / p)
        return resolved, True

    runs = root / "logs" / "runs"
    return runs / f"run-{run_id}.log", False

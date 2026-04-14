"""Persist last successful Session auto-tune outcome for copying into config.yaml."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]


def performance_results_path() -> Path:
    override = os.environ.get("WD_TAGGER_PERF_RESULTS_PATH", "").strip()
    if override:
        return Path(override)
    return _REPO_ROOT / "performance_results.yaml"


def save_performance_results(
    *,
    model_name: str,
    best_batch: int,
    best_dlp: int,
    best_intra: int,
    best_inter: int,
    tune_threads: bool,
    tuning_control_mode: str,
    autotune_phase: str,
) -> None:
    if os.environ.get("WD_TAGGER_SKIP_PERF_RESULTS_SAVE", "").lower() in ("1", "true", "yes"):
        return
    path = performance_results_path()
    doc: dict[str, Any] = {
        "schema_version": 1,
        "about": (
            "Written after a successful Tag all run with Session auto-tune. "
            "Copy config_patch into config.yaml — keys match config.example.yaml."
        ),
        "last_success": {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "model_name": model_name,
            "config_patch": {
                "batch_size": int(best_batch),
                "hydrus_download_parallel": int(best_dlp),
                "cpu_intra_op_threads": int(best_intra),
                "cpu_inter_op_threads": int(best_inter),
            },
            "source": {
                "tuning_control_mode": tuning_control_mode,
                "session_auto_tune_threads": bool(tune_threads),
                "autotune_phase": autotune_phase,
            },
        },
    }
    prev_last: dict[str, Any] | None = None
    if path.is_file():
        try:
            prev = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:
            prev = None
        if isinstance(prev, dict):
            pl = prev.get("last_success")
            if isinstance(pl, dict):
                prev_last = pl
    if prev_last is not None:
        doc["previous_last_success"] = prev_last
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(doc, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

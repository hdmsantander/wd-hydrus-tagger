"""In-memory recognize-run progress for HTTP polling (cluster + Hydrus apply)."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

_lock = threading.Lock()


@dataclass
class _RecognizeState:
    active: bool = False
    phase: str = ""
    stage_idx: int = 0
    stage_total: int = 0
    step_label: str = ""
    detail: str = ""
    faces_in_db: int = 0
    assigned: int = 0
    files_done: int = 0
    files_total: int = 0


_state = _RecognizeState()


def begin_recognize(*, stage_total: int, faces_in_db: int) -> None:
    with _lock:
        _state.active = True
        _state.phase = "cluster"
        _state.stage_idx = 0
        _state.stage_total = max(1, int(stage_total))
        _state.step_label = ""
        _state.detail = "Starting person clustering…"
        _state.faces_in_db = int(faces_in_db)
        _state.assigned = 0
        _state.files_done = 0
        _state.files_total = 0


def update_recognize_stage(
    *,
    stage_idx: int,
    step_label: str,
    assigned: int,
    cumulative_assigned: int,
) -> None:
    with _lock:
        _state.phase = "cluster"
        _state.stage_idx = int(stage_idx)
        _state.step_label = step_label
        _state.detail = step_label
        _state.assigned = int(cumulative_assigned)


def begin_recognize_apply(*, files_total: int, detail: str) -> None:
    with _lock:
        _state.phase = "apply"
        _state.detail = detail
        _state.files_total = max(0, int(files_total))
        _state.files_done = 0


def update_recognize_apply(files_done: int, detail: str | None = None) -> None:
    with _lock:
        _state.files_done = int(files_done)
        if detail:
            _state.detail = detail


def end_recognize() -> None:
    with _lock:
        _state.active = False
        _state.phase = ""
        _state.detail = ""


def get_recognize_status() -> dict:
    with _lock:
        s = _state
        return {
            "recognize": {
                "active": s.active,
                "phase": s.phase,
                "stage_idx": s.stage_idx,
                "stage_total": s.stage_total,
                "step_label": s.step_label,
                "detail": s.detail,
                "faces_in_db": s.faces_in_db,
                "assigned": s.assigned,
                "files_done": s.files_done,
                "files_total": s.files_total,
            },
        }

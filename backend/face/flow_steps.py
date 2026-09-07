"""Named pipeline steps for face detect / recognize progress (CLI + WebSocket + UI)."""

from __future__ import annotations

DETECT_PIPELINE = "detect"
RECOGNIZE_PIPELINE = "recognize"

_DETECT_STEPS: tuple[tuple[str, str, str], ...] = (
    ("metadata", "Fetch metadata", "metadata"),
    ("queue", "Analyze queue", "metadata"),
    ("model_load", "Load InsightFace model", "model"),
    ("warmup", "Warm up inference", "model"),
    ("scan", "Scan images", "detect"),
)

_STEP_INDEX = {name: i for i, (name, _, _) in enumerate(_DETECT_STEPS)}


def face_detect_progress(
    step: str,
    *,
    detail: str,
    processed: int = 0,
    total: int = 0,
    **fields,
) -> dict:
    """Build a WebSocket progress payload with step_num / step_label for the detect pipeline."""
    if step not in _STEP_INDEX:
        raise ValueError(f"unknown detect step: {step}")
    idx = _STEP_INDEX[step]
    _, label, default_phase = _DETECT_STEPS[idx]
    step_num = idx + 1
    step_total = len(_DETECT_STEPS)
    step_label = f"Detect {step_num}/{step_total} — {label}"
    payload = {
        "type": "progress",
        "pipeline": DETECT_PIPELINE,
        "step": step,
        "step_num": step_num,
        "step_total": step_total,
        "step_label": step_label,
        "processed": processed,
        "total": total,
        "phase": fields.pop("phase", default_phase),
        "face_count": 0,
        "skipped": False,
        "detail": detail or step_label,
    }
    payload.update(fields)
    return payload


def face_recognize_step_label(stage_idx: int, stage_total: int, min_faces: int) -> str:
    return f"Recognize {stage_idx}/{stage_total} — Cluster (min {min_faces} faces)"


def face_recognize_apply_label() -> str:
    return "Recognize — Apply person tags to Hydrus"

"""Prefetched-queue analysis: classify files (infer vs marker skip) and reorder for throughput."""

from __future__ import annotations

from dataclasses import dataclass

from backend.config import AppConfig
from backend.hydrus.tag_merge import build_wd_model_marker, inference_skip_decision


@dataclass(frozen=True)
class QueueAnalysisCounts:
    """Counts from a single pass over prefetched Hydrus metadata (same rules as ``tag_files``)."""

    infer: int
    skip_same_marker: int
    skip_higher_tier: int
    missing_metadata: int


def analyze_prefetched_queue(
    work_ids: list[int],
    meta_by_id: dict[int, dict],
    *,
    resolved_model: str,
    config: AppConfig,
    service_key: str,
) -> QueueAnalysisCounts:
    """Classify each file id using ``inference_skip_decision`` (metadata-only)."""
    marker = build_wd_model_marker(resolved_model, config.wd_model_marker_template)
    skip_marker = bool(marker) and config.wd_skip_inference_if_marker_present
    infer = 0
    skip_same = 0
    skip_hi = 0
    missing = 0
    for fid in work_ids:
        meta = meta_by_id.get(fid)
        if meta is None:
            missing += 1
            infer += 1
            continue
        do_skip, reason = inference_skip_decision(
            meta,
            current_model=resolved_model,
            canonical_marker=marker,
            skip_same_model_marker=skip_marker,
            skip_if_higher_tier_model=config.wd_skip_if_higher_tier_model_present,
            service_key=service_key,
            marker_prefix=config.wd_model_marker_prefix,
        )
        if not do_skip:
            infer += 1
        elif reason == "wd_model_marker_present":
            skip_same += 1
        elif reason == "wd_skip_higher_tier_model_present":
            skip_hi += 1
        else:
            infer += 1
    return QueueAnalysisCounts(
        infer=infer,
        skip_same_marker=skip_same,
        skip_higher_tier=skip_hi,
        missing_metadata=missing,
    )


def reorder_work_ids_inference_first(
    work_ids: list[int],
    meta_by_id: dict[int, dict],
    *,
    resolved_model: str,
    config: AppConfig,
    service_key: str,
) -> list[int]:
    """Place files that need ONNX + fetch first; marker skips follow in stable sub-order.

    Preserves relative order within each bucket by iterating ``work_ids`` once.
    """
    marker = build_wd_model_marker(resolved_model, config.wd_model_marker_template)
    skip_marker = bool(marker) and config.wd_skip_inference_if_marker_present
    infer_l: list[int] = []
    same_l: list[int] = []
    hi_l: list[int] = []
    for fid in work_ids:
        meta = meta_by_id.get(fid)
        if meta is None:
            infer_l.append(fid)
            continue
        do_skip, reason = inference_skip_decision(
            meta,
            current_model=resolved_model,
            canonical_marker=marker,
            skip_same_model_marker=skip_marker,
            skip_if_higher_tier_model=config.wd_skip_if_higher_tier_model_present,
            service_key=service_key,
            marker_prefix=config.wd_model_marker_prefix,
        )
        if not do_skip:
            infer_l.append(fid)
        elif reason == "wd_model_marker_present":
            same_l.append(fid)
        elif reason == "wd_skip_higher_tier_model_present":
            hi_l.append(fid)
        else:
            infer_l.append(fid)
    return infer_l + same_l + hi_l

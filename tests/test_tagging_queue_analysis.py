"""Prefetched queue analysis and infer-first reorder (tagging_ws throughput)."""

from unittest.mock import patch

import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]

from backend.config import AppConfig
from backend.hydrus.tag_merge import build_wd_model_marker
from backend.services.tagging_queue_analysis import (
    QueueAnalysisCounts,
    analyze_prefetched_queue,
    reorder_work_ids_inference_first,
)


def _meta_with_marker(file_id: int, marker: str, service_key: str = "svc") -> dict:
    return {
        "file_id": file_id,
        "hash": f"h{file_id}",
        "tags": {
            service_key: {
                "storage_tags": {"0": [marker]},
            },
        },
    }


def test_analyze_all_infer_when_no_marker_skip_config():
    cfg = AppConfig(
        wd_skip_inference_if_marker_present=False,
        wd_skip_if_higher_tier_model_present=False,
    )
    m = build_wd_model_marker("wd-vit-tagger-v3", "")
    meta = {
        1: _meta_with_marker(1, m),
        2: _meta_with_marker(2, m),
    }
    q = analyze_prefetched_queue(
        [1, 2],
        meta,
        resolved_model="wd-vit-tagger-v3",
        config=cfg,
        service_key="svc",
    )
    assert q.infer == 2
    assert q.skip_same_marker == 0
    assert q.skip_higher_tier == 0
    assert q.missing_metadata == 0


def test_analyze_skip_same_marker_counts():
    cfg = AppConfig(
        wd_skip_inference_if_marker_present=True,
        wd_skip_if_higher_tier_model_present=True,
    )
    m = build_wd_model_marker("wd-vit-tagger-v3", "")
    meta = {
        1: _meta_with_marker(1, m),
        2: {"file_id": 2, "hash": "a", "tags": {}},
    }
    q = analyze_prefetched_queue(
        [1, 2],
        meta,
        resolved_model="wd-vit-tagger-v3",
        config=cfg,
        service_key="svc",
    )
    assert q.skip_same_marker == 1
    assert q.infer == 1
    assert q.missing_metadata == 0


def test_analyze_missing_metadata_treated_as_infer():
    cfg = AppConfig()
    q = analyze_prefetched_queue(
        [1, 2],
        {1: {"file_id": 1, "hash": "a", "tags": {}}},
        resolved_model="wd-vit-tagger-v3",
        config=cfg,
        service_key="svc",
    )
    assert q.missing_metadata == 1
    assert q.infer == 2


def test_reorder_infer_before_skips():
    cfg = AppConfig(
        wd_skip_inference_if_marker_present=True,
        wd_skip_if_higher_tier_model_present=False,
    )
    m = build_wd_model_marker("wd-vit-tagger-v3", "")
    meta = {
        10: _meta_with_marker(10, m),
        11: {"file_id": 11, "hash": "x", "tags": {}},
        12: _meta_with_marker(12, m),
    }
    out = reorder_work_ids_inference_first(
        [10, 11, 12],
        meta,
        resolved_model="wd-vit-tagger-v3",
        config=cfg,
        service_key="svc",
    )
    assert out[0] == 11
    assert set(out) == {10, 11, 12}


def test_queue_analysis_counts_dataclass():
    q = QueueAnalysisCounts(infer=1, skip_same_marker=2, skip_higher_tier=3, missing_metadata=0)
    assert q.infer == 1


def test_analyze_skip_higher_tier_model():
    """File already tagged with a heavier WD model — skip ONNX for current smaller model."""
    cfg = AppConfig(
        wd_skip_inference_if_marker_present=True,
        wd_skip_if_higher_tier_model_present=True,
    )
    heavy_marker = "wd14:wd-eva02-large-tagger-v3"
    meta = {
        1: _meta_with_marker(1, heavy_marker),
    }
    q = analyze_prefetched_queue(
        [1],
        meta,
        resolved_model="wd-vit-tagger-v3",
        config=cfg,
        service_key="svc",
    )
    assert q.skip_higher_tier == 1
    assert q.infer == 0


def test_analyze_defensive_skip_reason_treated_as_infer():
    """If skip=True with an unknown reason, count as infer (matches tag_files safety)."""
    cfg = AppConfig()
    with patch(
        "backend.services.tagging_queue_analysis.inference_skip_decision",
        return_value=(True, "unexpected_reason"),
    ):
        q = analyze_prefetched_queue(
            [1],
            {1: {"file_id": 1, "tags": {}}},
            resolved_model="wd-vit-tagger-v3",
            config=cfg,
            service_key="svc",
        )
    assert q.infer == 1
    assert q.skip_same_marker == 0


def test_reorder_defensive_skip_reason_goes_to_infer_bucket():
    cfg = AppConfig()
    with patch(
        "backend.services.tagging_queue_analysis.inference_skip_decision",
        return_value=(True, "unexpected_reason"),
    ):
        out = reorder_work_ids_inference_first(
            [7],
            {7: {"file_id": 7, "tags": {}}},
            resolved_model="wd-vit-tagger-v3",
            config=cfg,
            service_key="svc",
        )
    assert out == [7]


def test_reorder_higher_tier_bucket_after_infer_and_same_marker():
    """Higher-tier skips append to ``hi_l`` (last bucket)."""
    cfg = AppConfig(
        wd_skip_inference_if_marker_present=True,
        wd_skip_if_higher_tier_model_present=True,
    )
    heavy = "wd14:wd-eva02-large-tagger-v3"
    m = build_wd_model_marker("wd-vit-tagger-v3", "")
    meta = {
        1: {"file_id": 1, "hash": "a", "tags": {}},
        2: _meta_with_marker(2, m),
        3: _meta_with_marker(3, heavy),
    }
    out = reorder_work_ids_inference_first(
        [3, 1, 2],
        meta,
        resolved_model="wd-vit-tagger-v3",
        config=cfg,
        service_key="svc",
    )
    assert out[0] == 1
    assert out[1] == 2
    assert out[2] == 3


def test_reorder_missing_metadata_id_first_in_infer_bucket():
    cfg = AppConfig(wd_skip_inference_if_marker_present=True)
    m = build_wd_model_marker("wd-vit-tagger-v3", "")
    meta = {1: _meta_with_marker(1, m)}
    out = reorder_work_ids_inference_first(
        [99, 1],
        meta,
        resolved_model="wd-vit-tagger-v3",
        config=cfg,
        service_key="svc",
    )
    assert out[0] == 99
    assert out[-1] == 1


def test_reorder_matches_analyze_buckets():
    cfg = AppConfig(
        wd_skip_inference_if_marker_present=True,
        wd_skip_if_higher_tier_model_present=False,
    )
    m = build_wd_model_marker("wd-vit-tagger-v3", "")
    meta = {
        1: _meta_with_marker(1, m),
        2: {"file_id": 2, "hash": "b", "tags": {}},
    }
    out = reorder_work_ids_inference_first(
        [1, 2],
        meta,
        resolved_model="wd-vit-tagger-v3",
        config=cfg,
        service_key="svc",
    )
    assert out == [2, 1]

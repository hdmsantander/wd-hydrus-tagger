"""ONNX compute device fields for tagging WebSocket progress."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]

from backend.services.tagging_service import TaggingService
from backend.services.tagging_shared import infer_batch_compute_activity, tagging_compute_payload
from backend.tagger.ort_providers import CPU_PROVIDER


@pytest.fixture
def tagging_svc(test_config):
    TaggingService._instance = None
    return TaggingService.get_instance(test_config)


def test_tagging_compute_payload_cpu_when_no_session(tagging_svc):
    row = tagging_compute_payload(tagging_svc, activity="cpu")
    assert row["compute_device"] == "cpu"
    assert row["compute_activity"] == "cpu"
    assert row["active_provider"] is None
    assert row["use_gpu"] is False


def test_tagging_compute_payload_gpu_when_session_on_gpu(tagging_svc):
    tagging_svc.engine._active_providers = ["MIGraphXExecutionProvider", CPU_PROVIDER]
    tagging_svc.engine.session = object()
    row = tagging_compute_payload(tagging_svc, batch_predicted=4, batch_skipped=0)
    assert row["compute_device"] == "gpu"
    assert row["compute_activity"] == "gpu"
    assert row["active_provider"] == "MIGraphXExecutionProvider"


def test_infer_batch_compute_activity_marker_skip_is_cpu(tagging_svc):
    tagging_svc.engine._active_providers = ["MIGraphXExecutionProvider", CPU_PROVIDER]
    tagging_svc.engine.session = object()
    assert infer_batch_compute_activity(tagging_svc, batch_predicted=0, batch_skipped=8) == "cpu"


def test_infer_batch_compute_activity_onnx_cpu(tagging_svc):
    tagging_svc.engine._active_providers = [CPU_PROVIDER]
    tagging_svc.engine.session = object()
    assert infer_batch_compute_activity(tagging_svc, batch_predicted=2, batch_skipped=0) == "cpu"

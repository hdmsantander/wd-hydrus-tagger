"""ORT provider resolution tests (no GPU required)."""

import pytest
from pydantic import ValidationError

pytestmark = [pytest.mark.full, pytest.mark.core]

from backend.config import AppConfig
from backend.tagger.ort_providers import (
    available_ort_providers,
    insightface_ctx_id,
    resolve_ort_providers,
)


def test_resolve_ort_providers_cpu_only():
    providers = resolve_ort_providers(use_gpu=False, gpu_backend="cpu")
    assert providers == ["CPUExecutionProvider"]


def test_resolve_ort_providers_auto_without_gpu():
    providers = resolve_ort_providers(use_gpu=False, gpu_backend="auto")
    assert providers == ["CPUExecutionProvider"]


def test_insightface_ctx_id_cpu():
    assert insightface_ctx_id(["CPUExecutionProvider"]) == -1


def test_insightface_ctx_id_cuda():
    assert insightface_ctx_id(["CUDAExecutionProvider", "CPUExecutionProvider"]) == 0


def test_available_ort_providers_includes_cpu():
    avail = available_ort_providers()
    assert "CPUExecutionProvider" in avail


def test_gpu_backend_config_validation():
    c = AppConfig(gpu_backend="rocm")
    assert c.gpu_backend == "rocm"
    with pytest.raises(ValidationError):
        AppConfig(gpu_backend="invalid")

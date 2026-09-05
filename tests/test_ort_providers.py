"""ORT provider resolution tests (no GPU required)."""

import sys
from unittest.mock import patch

import pytest
from pydantic import ValidationError

pytestmark = [pytest.mark.full, pytest.mark.core]

from backend.config import AppConfig
from backend.tagger.ort_providers import (
    available_gpu_providers,
    available_ort_providers,
    gpu_config_error_message,
    gpu_ep_priority_for_platform,
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


def test_insightface_ctx_id_migraphx():
    assert insightface_ctx_id(["MIGraphXExecutionProvider", "CPUExecutionProvider"]) == 0


def test_available_ort_providers_includes_cpu():
    avail = available_ort_providers()
    assert "CPUExecutionProvider" in avail


def test_gpu_backend_config_validation():
    c = AppConfig(gpu_backend="rocm")
    assert c.gpu_backend == "rocm"
    with pytest.raises(ValidationError):
        AppConfig(gpu_backend="invalid")


def test_gpu_ep_priority_linux_excludes_directml():
    order = gpu_ep_priority_for_platform("linux")
    assert "MIGraphXExecutionProvider" in order
    assert "DmlExecutionProvider" not in order


def test_gpu_ep_priority_windows_includes_directml():
    order = gpu_ep_priority_for_platform("win32")
    assert "DmlExecutionProvider" in order


def test_resolve_ort_providers_linux_skips_directml_even_if_registered():
    avail = [
        "DmlExecutionProvider",
        "MIGraphXExecutionProvider",
        "CPUExecutionProvider",
    ]
    with patch("backend.tagger.ort_providers.available_ort_providers", return_value=avail):
        assert resolve_ort_providers(use_gpu=True, gpu_backend="auto", platform="linux") == [
            "MIGraphXExecutionProvider",
            "CPUExecutionProvider",
        ]


def test_resolve_ort_providers_rocm_backend_prefers_migraphx_when_rocm_ep_missing():
    avail = ["MIGraphXExecutionProvider", "CPUExecutionProvider"]
    with patch("backend.tagger.ort_providers.available_ort_providers", return_value=avail):
        assert resolve_ort_providers(use_gpu=True, gpu_backend="rocm") == [
            "MIGraphXExecutionProvider",
            "CPUExecutionProvider",
        ]


def test_available_gpu_providers_respects_platform():
    avail = ["DmlExecutionProvider", "MIGraphXExecutionProvider", "CPUExecutionProvider"]
    with patch("backend.tagger.ort_providers.available_ort_providers", return_value=avail):
        assert available_gpu_providers("linux") == ["MIGraphXExecutionProvider"]


def test_gpu_config_error_message_linux_mentions_rocm():
    msg = gpu_config_error_message("linux")
    assert "rocm" in msg.lower()
    assert "use_gpu" in msg


@pytest.mark.skipif(sys.platform != "linux", reason="Linux-only ORT integration smoke test")
def test_linux_ort_registers_cpu_provider():
    assert "CPUExecutionProvider" in available_ort_providers()

"""ONNX execution provider selection for TaggerEngine."""

import sys
from unittest.mock import patch

import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]

from backend.tagger.providers import (
    build_execution_providers,
    gpu_config_error_message,
    gpu_provider_order_for_platform,
)


def test_gpu_provider_order_linux_excludes_directml():
    order = gpu_provider_order_for_platform("linux")
    assert "MIGraphXExecutionProvider" in order
    assert "DmlExecutionProvider" not in order


def test_gpu_provider_order_windows_includes_directml():
    order = gpu_provider_order_for_platform("win32")
    assert "DmlExecutionProvider" in order


def test_build_providers_cpu_only():
    with patch(
        "backend.tagger.providers.get_available_providers",
        return_value=["CPUExecutionProvider"],
    ):
        assert build_execution_providers(False) == ["CPUExecutionProvider"]


def test_build_providers_gpu_cuda_first():
    available = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    with patch("backend.tagger.providers.get_available_providers", return_value=available):
        assert build_execution_providers(True) == [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]


def test_build_providers_gpu_migraphx_when_no_cuda():
    available = [
        "MIGraphXExecutionProvider",
        "CPUExecutionProvider",
    ]
    with patch("backend.tagger.providers.get_available_providers", return_value=available):
        assert build_execution_providers(True) == [
            "MIGraphXExecutionProvider",
            "CPUExecutionProvider",
        ]


def test_build_providers_linux_skips_directml_even_if_registered():
    available = [
        "DmlExecutionProvider",
        "MIGraphXExecutionProvider",
        "CPUExecutionProvider",
    ]
    with patch("backend.tagger.providers.get_available_providers", return_value=available):
        assert build_execution_providers(True, platform="linux") == [
            "MIGraphXExecutionProvider",
            "CPUExecutionProvider",
        ]


def test_build_providers_gpu_directml_on_windows():
    available = [
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]
    with patch("backend.tagger.providers.get_available_providers", return_value=available):
        assert build_execution_providers(True, platform="win32") == [
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        ]


def test_build_providers_gpu_priority_order_windows():
    available = list(gpu_provider_order_for_platform("win32")) + ["CPUExecutionProvider"]
    with patch("backend.tagger.providers.get_available_providers", return_value=available):
        assert build_execution_providers(True, platform="win32") == list(
            gpu_provider_order_for_platform("win32")
        ) + ["CPUExecutionProvider"]


def test_build_providers_gpu_no_gpu_eps_falls_back_to_cpu():
    available = ["CPUExecutionProvider", "AzureExecutionProvider"]
    with patch("backend.tagger.providers.get_available_providers", return_value=available):
        assert build_execution_providers(True) == ["CPUExecutionProvider"]


def test_gpu_config_error_message_linux_mentions_migraphx():
    msg = gpu_config_error_message("linux")
    assert "MIGraphX" in msg
    assert "use_gpu" in msg


@pytest.mark.skipif(sys.platform != "linux", reason="Linux-only provider integration smoke test")
def test_linux_ort_registers_cpu_provider():
    from backend.tagger.providers import get_available_providers

    providers = get_available_providers()
    assert "CPUExecutionProvider" in providers

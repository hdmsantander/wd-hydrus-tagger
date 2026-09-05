"""ONNX execution provider selection for TaggerEngine."""

from unittest.mock import patch

import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]

from backend.tagger.providers import GPU_PROVIDER_ORDER, build_execution_providers


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


def test_build_providers_gpu_directml_when_only_dml():
    available = [
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]
    with patch("backend.tagger.providers.get_available_providers", return_value=available):
        assert build_execution_providers(True) == [
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        ]


def test_build_providers_gpu_priority_order():
    available = list(GPU_PROVIDER_ORDER) + ["CPUExecutionProvider"]
    with patch("backend.tagger.providers.get_available_providers", return_value=available):
        assert build_execution_providers(True) == list(GPU_PROVIDER_ORDER) + ["CPUExecutionProvider"]


def test_build_providers_gpu_no_gpu_eps_falls_back_to_cpu():
    available = ["CPUExecutionProvider", "AzureExecutionProvider"]
    with patch("backend.tagger.providers.get_available_providers", return_value=available):
        assert build_execution_providers(True) == ["CPUExecutionProvider"]

"""ONNX Runtime execution provider selection for WD tagger inference."""

from __future__ import annotations

import sys

ALL_GPU_PROVIDERS = frozenset(
    {
        "CUDAExecutionProvider",
        "MIGraphXExecutionProvider",
        "DmlExecutionProvider",
    }
)
CPU_PROVIDER = "CPUExecutionProvider"


def gpu_provider_order_for_platform(platform: str | None = None) -> tuple[str, ...]:
    """Return GPU EP preference order for the host OS.

    Linux: CUDA (NVIDIA) then MIGraphX (AMD/ROCm). DirectML is Windows-only.
    Windows: CUDA, DirectML, then MIGraphX (WinML may register MIGraphX on AMD).
    """
    plat = platform if platform is not None else sys.platform
    if plat == "linux":
        return ("CUDAExecutionProvider", "MIGraphXExecutionProvider")
    if plat == "win32":
        return (
            "CUDAExecutionProvider",
            "DmlExecutionProvider",
            "MIGraphXExecutionProvider",
        )
    return (
        "CUDAExecutionProvider",
        "MIGraphXExecutionProvider",
        "DmlExecutionProvider",
    )


# Backwards-compatible alias used in tests/docs (Windows order includes DirectML).
GPU_PROVIDER_ORDER = gpu_provider_order_for_platform("win32")


def get_available_providers() -> list[str]:
    """Return execution providers registered in the installed ONNX Runtime build."""
    import onnxruntime as ort

    return list(ort.get_available_providers())


def available_gpu_providers(platform: str | None = None) -> list[str]:
    """GPU providers both registered in ORT and applicable on ``platform``."""
    available = set(get_available_providers())
    return [p for p in gpu_provider_order_for_platform(platform) if p in available]


def build_execution_providers(use_gpu: bool, *, platform: str | None = None) -> list[str]:
    """Build an ONNX Runtime provider list for ``TaggerEngine``.

    When ``use_gpu`` is false, only ``CPUExecutionProvider`` is requested.
    When true, available GPU providers are tried in platform order with CPU last.
    """
    available = set(get_available_providers())
    cpu = CPU_PROVIDER

    if not use_gpu:
        if cpu in available:
            return [cpu]
        return get_available_providers()

    providers: list[str] = []
    for name in gpu_provider_order_for_platform(platform):
        if name in available:
            providers.append(name)

    if cpu in available and cpu not in providers:
        providers.append(cpu)

    if not providers:
        return get_available_providers()

    return providers


def gpu_config_error_message(platform: str | None = None) -> str:
    """Human-readable fix hint when ``use_gpu`` is enabled but no GPU EP is registered."""
    plat = platform if platform is not None else sys.platform
    if plat == "linux":
        return (
            "use_gpu is true but no GPU execution provider is registered. "
            "On Linux install a ROCm-matched MIGraphX ONNX Runtime wheel "
            "(https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html) "
            "or NVIDIA onnxruntime-gpu; otherwise set use_gpu: false for CPU inference."
        )
    if plat == "win32":
        return (
            "use_gpu is true but no GPU execution provider is registered. "
            "On Windows install onnxruntime-directml or onnxruntime-gpu; "
            "otherwise set use_gpu: false for CPU inference."
        )
    return (
        "use_gpu is true but no GPU execution provider is registered. "
        "Install onnxruntime-gpu, MIGraphX, or DirectML matching your hardware; "
        "otherwise set use_gpu: false for CPU inference."
    )

"""ONNX Runtime execution provider selection for WD tagger inference."""

from __future__ import annotations

# Preferred order when ``use_gpu`` is enabled. CUDA (NVIDIA), MIGraphX (AMD ROCm on
# Linux), DirectML (AMD/Intel/NVIDIA on Windows). CPU is always appended last as fallback.
GPU_PROVIDER_ORDER = (
    "CUDAExecutionProvider",
    "MIGraphXExecutionProvider",
    "DmlExecutionProvider",
)


def get_available_providers() -> list[str]:
    """Return execution providers registered in the installed ONNX Runtime build."""
    import onnxruntime as ort

    return list(ort.get_available_providers())


def build_execution_providers(use_gpu: bool) -> list[str]:
    """Build an ONNX Runtime provider list for ``TaggerEngine``.

    When ``use_gpu`` is false, only ``CPUExecutionProvider`` is requested.
    When true, available GPU providers are tried in priority order with CPU last.
    """
    available = set(get_available_providers())
    cpu = "CPUExecutionProvider"

    if not use_gpu:
        if cpu in available:
            return [cpu]
        return get_available_providers()

    providers: list[str] = []
    for name in GPU_PROVIDER_ORDER:
        if name in available:
            providers.append(name)

    if cpu in available and cpu not in providers:
        providers.append(cpu)

    if not providers:
        return get_available_providers()

    return providers

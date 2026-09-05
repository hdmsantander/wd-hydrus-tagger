"""ONNX Runtime execution provider selection (NVIDIA CUDA, AMD ROCm, DirectML, CPU)."""

from __future__ import annotations

import logging
from typing import Literal

log = logging.getLogger(__name__)

GpuBackend = Literal["auto", "cuda", "rocm", "directml", "cpu"]

_GPU_EPS = frozenset({
    "CUDAExecutionProvider",
    "ROCmExecutionProvider",
    "DmlExecutionProvider",
})

_PRIORITY_AUTO = (
    "CUDAExecutionProvider",
    "ROCmExecutionProvider",
    "DmlExecutionProvider",
    "CPUExecutionProvider",
)

_BACKEND_TO_EP: dict[str, tuple[str, ...]] = {
    "cuda": ("CUDAExecutionProvider",),
    "rocm": ("ROCmExecutionProvider",),
    "directml": ("DmlExecutionProvider",),
}


def available_ort_providers() -> list[str]:
    try:
        import onnxruntime as ort

        return list(ort.get_available_providers())
    except Exception:
        return ["CPUExecutionProvider"]


def resolve_ort_providers(
    *,
    use_gpu: bool = False,
    gpu_backend: str = "auto",
) -> list[str]:
    """Pick ONNX Runtime providers for WD and face models.

    ``gpu_backend``:
    - ``auto`` — first available GPU EP (CUDA → ROCm → DirectML), else CPU
    - ``cuda`` / ``rocm`` / ``directml`` — force a specific EP when installed
    - ``cpu`` — CPU only (ignores ``use_gpu``)
    """
    avail = available_ort_providers()
    backend = (gpu_backend or "auto").strip().lower()

    if backend == "cpu":
        return ["CPUExecutionProvider"]

    if not use_gpu and backend == "auto":
        return ["CPUExecutionProvider"]

    if backend in _BACKEND_TO_EP:
        preferred = _BACKEND_TO_EP[backend]
        out = [p for p in preferred if p in avail]
        if not out:
            log.warning(
                "Requested GPU backend %r not available (installed EPs: %s); falling back to CPU",
                backend,
                avail,
            )
            return ["CPUExecutionProvider"]
        if "CPUExecutionProvider" not in out:
            out.append("CPUExecutionProvider")
        return out

    # auto with use_gpu
    out: list[str] = []
    for ep in _PRIORITY_AUTO:
        if ep in avail and ep not in out:
            out.append(ep)
    if not out or out == ["CPUExecutionProvider"]:
        log.info(
            "No GPU execution provider available (installed: %s); using CPUExecutionProvider",
            avail,
        )
        return ["CPUExecutionProvider"]
    return out


def insightface_ctx_id(providers: list[str]) -> int:
    """InsightFace ``ctx_id``: 0 when a GPU EP is active, -1 for CPU-only."""
    if any(p in _GPU_EPS for p in providers):
        return 0
    return -1


def active_gpu_provider_label(providers: list[str]) -> str:
    for p in providers:
        if p != "CPUExecutionProvider":
            return p
    return "CPUExecutionProvider"

"""ONNX Runtime execution provider selection (NVIDIA CUDA, AMD ROCm/MIGraphX, DirectML, CPU)."""

from __future__ import annotations

import logging
import sys
from typing import Literal

log = logging.getLogger(__name__)

GpuBackend = Literal["auto", "cuda", "rocm", "directml", "cpu"]

_GPU_EPS = frozenset(
    {
        "CUDAExecutionProvider",
        "ROCmExecutionProvider",
        "MIGraphXExecutionProvider",
        "DmlExecutionProvider",
    }
)
CPU_PROVIDER = "CPUExecutionProvider"

_BACKEND_TO_EP: dict[str, tuple[str, ...]] = {
    "cuda": ("CUDAExecutionProvider",),
    # ROCm EP (legacy) and MIGraphX EP (ORT 1.23+) — try both when backend is rocm.
    "rocm": ("ROCmExecutionProvider", "MIGraphXExecutionProvider"),
    "directml": ("DmlExecutionProvider",),
}


def gpu_ep_priority_for_platform(platform: str | None = None) -> tuple[str, ...]:
    """GPU EP preference order for ``auto`` mode on the host OS."""
    plat = platform if platform is not None else sys.platform
    if plat == "linux":
        return (
            "CUDAExecutionProvider",
            "ROCmExecutionProvider",
            "MIGraphXExecutionProvider",
        )
    if plat == "win32":
        return (
            "CUDAExecutionProvider",
            "DmlExecutionProvider",
            "ROCmExecutionProvider",
            "MIGraphXExecutionProvider",
        )
    return (
        "CUDAExecutionProvider",
        "ROCmExecutionProvider",
        "MIGraphXExecutionProvider",
        "DmlExecutionProvider",
    )


def available_ort_providers() -> list[str]:
    try:
        import onnxruntime as ort

        return list(ort.get_available_providers())
    except Exception:
        return [CPU_PROVIDER]


def available_gpu_providers(platform: str | None = None) -> list[str]:
    """GPU EPs both registered in ORT and applicable on ``platform``."""
    avail = set(available_ort_providers())
    return [p for p in gpu_ep_priority_for_platform(platform) if p in avail]


def resolve_ort_providers(
    *,
    use_gpu: bool = False,
    gpu_backend: str = "auto",
    platform: str | None = None,
) -> list[str]:
    """Pick ONNX Runtime providers for WD and face models.

    ``gpu_backend``:
    - ``auto`` — first available GPU EP for the platform, else CPU
    - ``cuda`` / ``rocm`` / ``directml`` — force a specific EP family when installed
    - ``cpu`` — CPU only (ignores ``use_gpu``)
    """
    avail = available_ort_providers()
    backend = (gpu_backend or "auto").strip().lower()

    if backend == "cpu":
        return [CPU_PROVIDER]

    if not use_gpu and backend == "auto":
        return [CPU_PROVIDER]

    if backend in _BACKEND_TO_EP:
        preferred = _BACKEND_TO_EP[backend]
        out = [p for p in preferred if p in avail]
        if not out:
            log.warning(
                "Requested GPU backend %r not available (installed EPs: %s); falling back to CPU",
                backend,
                avail,
            )
            return [CPU_PROVIDER]
        if CPU_PROVIDER not in out:
            out.append(CPU_PROVIDER)
        return out

    # auto with use_gpu
    out: list[str] = []
    for ep in gpu_ep_priority_for_platform(platform):
        if ep in avail and ep not in out:
            out.append(ep)
    if CPU_PROVIDER in avail and CPU_PROVIDER not in out:
        out.append(CPU_PROVIDER)
    if not out or out == [CPU_PROVIDER]:
        log.info(
            "No GPU execution provider available (installed: %s); using CPUExecutionProvider",
            avail,
        )
        return [CPU_PROVIDER]
    return out


def gpu_config_error_message(platform: str | None = None) -> str:
    """Human-readable fix hint when GPU inference is requested but unavailable."""
    plat = platform if platform is not None else sys.platform
    if plat == "linux":
        return (
            "use_gpu is true but no GPU execution provider is registered. "
            "On Linux install onnxruntime-rocm (gpu_backend: rocm) or NVIDIA onnxruntime-gpu "
            "(gpu_backend: cuda); see docs/FACE_TAGGING.md and docs/DEPENDENCIES.md. "
            "Otherwise set use_gpu: false for CPU inference."
        )
    if plat == "win32":
        return (
            "use_gpu is true but no GPU execution provider is registered. "
            "On Windows install onnxruntime-directml (gpu_backend: directml) or "
            "onnxruntime-gpu (gpu_backend: cuda); otherwise set use_gpu: false."
        )
    return (
        "use_gpu is true but no GPU execution provider is registered. "
        "Install a matching ONNX Runtime GPU build and set gpu_backend appropriately; "
        "otherwise set use_gpu: false for CPU inference."
    )


def insightface_ctx_id(providers: list[str]) -> int:
    """InsightFace ``ctx_id``: 0 when a GPU EP is active, -1 for CPU-only."""
    if any(p in _GPU_EPS for p in providers):
        return 0
    return -1


def active_gpu_provider_label(providers: list[str]) -> str:
    for p in providers:
        if p != CPU_PROVIDER:
            return p
    return CPU_PROVIDER

"""Shared tagging helpers used by routes/services.

These are intentionally side-effect free to avoid coupling routes to TaggingService internals.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from backend.hydrus.client import HydrusClient
from backend.hydrus.metadata_maps import rows_to_file_id_map
from backend.tagger.ort_providers import CPU_PROVIDER

if TYPE_CHECKING:
    from backend.services.tagging_service import TaggingService

log = logging.getLogger(__name__)


def tagging_compute_payload(
    service: TaggingService,
    *,
    activity: str | None = None,
    batch_predicted: int | None = None,
    batch_skipped: int | None = None,
) -> dict:
    """WebSocket / UI fields describing ONNX compute device (CPU vs GPU EP)."""
    cfg = service.config
    eng = service.engine
    provider = eng.active_provider if eng.session else None
    on_gpu = bool(provider and provider != CPU_PROVIDER)
    device = "gpu" if on_gpu else "cpu"
    if activity is None:
        if batch_predicted is not None and batch_predicted > 0:
            activity = "gpu" if on_gpu else "cpu"
        elif batch_skipped is not None and batch_skipped > 0:
            activity = "cpu"
        else:
            activity = device
    return {
        "use_gpu": bool(cfg.use_gpu),
        "gpu_backend": (cfg.gpu_backend or "auto").strip().lower(),
        "active_provider": provider,
        "compute_device": device,
        "compute_activity": activity,
    }


def infer_batch_compute_activity(service: TaggingService, *, batch_predicted: int, batch_skipped: int) -> str:
    """Which compute chip should pulse for this outer-batch progress tick."""
    if batch_predicted > 0:
        provider = service.engine.active_provider if service.engine.session else CPU_PROVIDER
        return "gpu" if provider != CPU_PROVIDER else "cpu"
    return "cpu"


def clamp_inference_batch(n: int | None, fallback: int) -> int:
    base = fallback if n is None else n
    return max(1, min(256, int(base)))


async def load_metadata_by_file_id(
    client: HydrusClient,
    file_ids: list[int],
    *,
    chunk_sz: int,
    cancel_event: asyncio.Event | None = None,
    progress_cb=None,
) -> dict[int, dict]:
    """Hydrus get_file_metadata in chunks; returns file_id → row (empty dicts skipped)."""
    meta_by_id: dict[int, dict] = {}
    total = len(file_ids)
    for off in range(0, total, chunk_sz):
        if cancel_event is not None and cancel_event.is_set():
            log.info("load_metadata_by_file_id stopped early offset=%s (cancel)", off)
            break
        part = file_ids[off : off + chunk_sz]
        rows = await client.get_file_metadata(file_ids=part)
        meta_by_id.update(rows_to_file_id_map(rows))
        if progress_cb is not None:
            await progress_cb(off + len(part), total)
    return meta_by_id

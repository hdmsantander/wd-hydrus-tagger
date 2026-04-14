"""Shared tagging helpers used by routes/services.

These are intentionally side-effect free to avoid coupling routes to TaggingService internals.
"""

from __future__ import annotations

import asyncio
import logging

from backend.hydrus.client import HydrusClient
from backend.hydrus.metadata_maps import rows_to_file_id_map

log = logging.getLogger(__name__)


def clamp_inference_batch(n: int | None, fallback: int) -> int:
    base = fallback if n is None else n
    return max(1, min(256, int(base)))


async def load_metadata_by_file_id(
    client: HydrusClient,
    file_ids: list[int],
    *,
    chunk_sz: int,
    cancel_event: asyncio.Event | None = None,
) -> dict[int, dict]:
    """Hydrus get_file_metadata in chunks; returns file_id → row (empty dicts skipped)."""
    meta_by_id: dict[int, dict] = {}
    for off in range(0, len(file_ids), chunk_sz):
        if cancel_event is not None and cancel_event.is_set():
            log.info("load_metadata_by_file_id stopped early offset=%s (cancel)", off)
            break
        part = file_ids[off : off + chunk_sz]
        rows = await client.get_file_metadata(file_ids=part)
        meta_by_id.update(rows_to_file_id_map(rows))
    return meta_by_id

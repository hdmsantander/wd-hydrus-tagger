"""Async polling around blocking work (ONNX load/predict) with log + optional progress heartbeats."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

log = logging.getLogger(__name__)

ProgressTick = Callable[[int, str], Awaitable[None] | None]


async def run_blocking_with_heartbeat(
    fn: Callable[[], Any],
    *,
    log_label: str,
    interval_s: float = 15.0,
    poll_s: float = 2.0,
    cancel_event: asyncio.Event | None = None,
    progress_tick: ProgressTick | None = None,
    progress_detail: str = "",
    timeout_s: float | None = None,
) -> Any:
    """Run ``fn`` in a worker thread; log and optionally emit progress every ``interval_s``."""
    worker = asyncio.create_task(asyncio.to_thread(fn))
    t0 = time.monotonic()
    last_log_s = -interval_s

    async def _tick(elapsed: int, *, force_log: bool = False) -> None:
        nonlocal last_log_s
        if progress_tick is not None:
            detail = progress_detail
            if detail:
                detail = f"{detail} ({elapsed}s)"
            maybe = progress_tick(elapsed, detail)
            if maybe is not None:
                await maybe
        if force_log or elapsed >= last_log_s + interval_s:
            log.info("%s still running elapsed_s=%s", log_label, elapsed)
            last_log_s = elapsed

    await _tick(0, force_log=True)
    while not worker.done():
        if cancel_event is not None and cancel_event.is_set():
            worker.cancel()
            raise asyncio.CancelledError(f"{log_label} cancelled")
        elapsed = int(time.monotonic() - t0)
        await _tick(elapsed)
        if timeout_s is not None and (time.monotonic() - t0) >= timeout_s:
            worker.cancel()
            raise TimeoutError(f"{log_label} timed out after {timeout_s:.0f}s")
        try:
            await asyncio.wait_for(asyncio.shield(worker), timeout=poll_s)
        except asyncio.TimeoutError:
            continue
    if cancel_event is not None and cancel_event.is_set():
        raise asyncio.CancelledError(f"{log_label} cancelled")
    return await worker

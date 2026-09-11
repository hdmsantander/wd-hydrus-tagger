"""blocking_heartbeat.run_blocking_with_heartbeat."""

from __future__ import annotations

import asyncio
import time

import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]

from backend.services.blocking_heartbeat import run_blocking_with_heartbeat


@pytest.mark.asyncio
async def test_run_blocking_with_heartbeat_returns_result():
    ticks: list[int] = []

    async def tick(elapsed: int, _detail: str) -> None:
        ticks.append(elapsed)

    out = await run_blocking_with_heartbeat(
        lambda: 42,
        log_label="test worker",
        interval_s=0.05,
        poll_s=0.01,
        progress_tick=tick,
        progress_detail="working",
    )
    assert out == 42
    assert 0 in ticks


@pytest.mark.asyncio
async def test_run_blocking_with_heartbeat_honours_cancel():
    cancel = asyncio.Event()

    def slow():
        time.sleep(0.5)
        return 1

    cancel.set()
    with pytest.raises(asyncio.CancelledError):
        await run_blocking_with_heartbeat(
            slow,
            log_label="cancel test",
            poll_s=0.01,
            cancel_event=cancel,
        )

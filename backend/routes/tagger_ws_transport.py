"""Transport helpers for tagger WebSocket flows."""

from __future__ import annotations

import asyncio
import logging

import httpx
from fastapi import WebSocket, WebSocketDisconnect

from backend.hydrus.client import HydrusClient
from backend.log_stats import log_stats

# Client disconnected or socket already closed — safe to ignore for best-effort sends.
WS_SEND_CLIENT_GONE = (WebSocketDisconnect, OSError, RuntimeError)
log = logging.getLogger(__name__)


async def ws_send_json_ignore_closed(ws: WebSocket, payload: dict) -> None:
    """Send JSON; swallow expected disconnect / transport errors (no raise)."""
    try:
        await ws.send_json(payload)
    except WS_SEND_CLIENT_GONE:
        pass


async def wait_until_hydrus_responsive(
    *,
    client: HydrusClient,
    cancel_event: asyncio.Event,
    hydrus_manual_retry: asyncio.Event,
    ws_send,
    last_error: str,
    snapshot: dict,
    poll_s: float = 12.0,
) -> bool:
    """Poll Hydrus ``verify_access_key`` or wake on client ``retry_hydrus``. Returns False if cancelled."""
    if cancel_event.is_set():
        return False
    payload = {
        "type": "hydrus_unreachable",
        "message": last_error,
        **snapshot,
    }
    if not await ws_send(payload):
        return False
    poll_failures = 0
    while not cancel_event.is_set():
        try:
            await client.verify_access_key()
            if poll_failures:
                log.info(
                    "tagging_ws hydrus_recovered after_failed_polls=%s",
                    poll_failures,
                )
            log_stats(
                log,
                "hydrus_recovered",
                failed_polls=poll_failures,
            )
            return await ws_send(
                {"type": "hydrus_recovered", "message": "Hydrus API is reachable again."},
            )
        except (httpx.HTTPError, OSError, ValueError) as e:
            poll_failures += 1
            log.debug("hydrus recovery poll still_unreachable err=%s", e)
        t_retry = asyncio.create_task(hydrus_manual_retry.wait())
        t_sleep = asyncio.create_task(asyncio.sleep(poll_s))
        done, pending = await asyncio.wait(
            [t_retry, t_sleep],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()
        for t in done:
            try:
                await t
            except asyncio.CancelledError:
                pass
        hydrus_manual_retry.clear()
        if not await ws_send({"type": "hydrus_waiting", "next_poll_s": poll_s}):
            return False
    return False


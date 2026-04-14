"""Hydrus wait/recovery helper on the tagging WebSocket."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from backend.routes.tagger_ws import _wait_until_hydrus_responsive

pytestmark = [pytest.mark.full, pytest.mark.ws]


@pytest.mark.asyncio
async def test_wait_until_hydrus_responsive_first_verify_ok():
    client = MagicMock()
    client.verify_access_key = AsyncMock(return_value={"hydrus": 1})
    cancel = asyncio.Event()
    retry = asyncio.Event()
    sent: list[str] = []

    async def ws_send(p: dict) -> bool:
        sent.append(p.get("type", ""))
        return True

    ok = await _wait_until_hydrus_responsive(
        client=client,
        cancel_event=cancel,
        hydrus_manual_retry=retry,
        ws_send=ws_send,
        last_error="connection reset",
        snapshot={"pending_commit_count": 0},
        poll_s=0.05,
    )
    assert ok is True
    assert sent[0] == "hydrus_unreachable"
    assert sent[-1] == "hydrus_recovered"
    client.verify_access_key.assert_awaited_once()


@pytest.mark.asyncio
async def test_wait_until_hydrus_responsive_retries_then_ok(monkeypatch):
    n = 0

    async def flaky_verify():
        nonlocal n
        n += 1
        if n < 2:
            raise httpx.ConnectError("down")
        return {}

    client = MagicMock()
    client.verify_access_key = flaky_verify
    cancel = asyncio.Event()
    retry = asyncio.Event()
    sent: list[str] = []

    async def ws_send(p: dict) -> bool:
        sent.append(p.get("type", ""))
        return True

    async def instant_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", instant_sleep)

    ok = await _wait_until_hydrus_responsive(
        client=client,
        cancel_event=cancel,
        hydrus_manual_retry=retry,
        ws_send=ws_send,
        last_error="down",
        snapshot={},
        poll_s=0.05,
    )
    assert ok is True
    assert "hydrus_waiting" in sent
    assert sent[-1] == "hydrus_recovered"
    assert n == 2


@pytest.mark.asyncio
async def test_wait_until_hydrus_responsive_cancelled():
    client = MagicMock()
    client.verify_access_key = AsyncMock(side_effect=httpx.ConnectError("down"))
    cancel = asyncio.Event()
    cancel.set()
    retry = asyncio.Event()

    async def ws_send(_):
        return True

    ok = await _wait_until_hydrus_responsive(
        client=client,
        cancel_event=cancel,
        hydrus_manual_retry=retry,
        ws_send=ws_send,
        last_error="x",
        snapshot={},
        poll_s=60.0,
    )
    assert ok is False

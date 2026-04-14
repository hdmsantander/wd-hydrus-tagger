"""``_wait_until_hydrus_responsive`` (WebSocket Hydrus recovery loop)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = [pytest.mark.full, pytest.mark.core, pytest.mark.ws]

from backend.routes.tagger_ws import _wait_until_hydrus_responsive


@pytest.mark.asyncio
async def test_wait_hydrus_returns_false_when_already_cancelled():
    cancel = asyncio.Event()
    cancel.set()
    out = await _wait_until_hydrus_responsive(
        client=MagicMock(),
        cancel_event=cancel,
        hydrus_manual_retry=asyncio.Event(),
        ws_send=AsyncMock(return_value=True),
        last_error="e",
        snapshot={},
    )
    assert out is False


@pytest.mark.asyncio
async def test_wait_hydrus_returns_false_when_unreachable_payload_not_sent():
    cancel = asyncio.Event()
    out = await _wait_until_hydrus_responsive(
        client=MagicMock(),
        cancel_event=cancel,
        hydrus_manual_retry=asyncio.Event(),
        ws_send=AsyncMock(return_value=False),
        last_error="e",
        snapshot={"n": 1},
    )
    assert out is False


@pytest.mark.asyncio
async def test_wait_hydrus_recovered_after_verify():
    client = MagicMock()
    client.verify_access_key = AsyncMock()
    cancel = asyncio.Event()
    sent: list[dict] = []

    async def ws_send(p: dict) -> bool:
        sent.append(p)
        return True

    out = await _wait_until_hydrus_responsive(
        client=client,
        cancel_event=cancel,
        hydrus_manual_retry=asyncio.Event(),
        ws_send=ws_send,
        last_error="down",
        snapshot={"session": 1},
        poll_s=0.01,
    )
    assert out is True
    assert sent[0]["type"] == "hydrus_unreachable"
    assert sent[0]["session"] == 1
    assert any(x.get("type") == "hydrus_recovered" for x in sent)

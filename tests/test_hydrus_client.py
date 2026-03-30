"""Hydrus HTTP client pooling."""

import pytest

import backend.hydrus.client as hydrus_mod
from backend.hydrus.client import HydrusClient, aclose_all_hydrus_clients


@pytest.mark.asyncio
async def test_aclose_all_hydrus_clients_clears_pool():
    hc = HydrusClient("http://example.invalid:45869", "key")
    await hc._shared()
    assert len(hydrus_mod._client_pool) == 1
    await aclose_all_hydrus_clients()
    assert len(hydrus_mod._client_pool) == 0


@pytest.mark.asyncio
async def test_same_pool_key_reuses_client():
    await aclose_all_hydrus_clients()
    a = HydrusClient("http://h.test:1", "k")
    b = HydrusClient("http://h.test:1", "k")
    ca = await a._shared()
    cb = await b._shared()
    assert ca is cb
    await aclose_all_hydrus_clients()

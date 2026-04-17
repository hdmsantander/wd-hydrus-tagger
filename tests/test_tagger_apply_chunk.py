"""Unit tests for ``_apply_results_chunk`` (Hydrus apply helper)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]

from backend.routes.tagger_apply import _apply_results_chunk


@pytest.mark.asyncio
async def test_apply_chunk_returns_zeros_when_no_service_or_items():
    client = MagicMock()
    assert await _apply_results_chunk(client, "", [{"file_id": 1, "hash": "h", "tags": ["a"]}]) == (
        0,
        0,
        0,
    )
    assert await _apply_results_chunk(client, "sk", []) == (0, 0, 0)


@pytest.mark.asyncio
async def test_apply_chunk_skips_row_without_hash():
    client = MagicMock()
    client.get_file_metadata = AsyncMock(return_value=[])
    client.apply_tag_actions = AsyncMock()
    out = await _apply_results_chunk(
        client,
        "sk",
        [{"file_id": 1, "tags": ["a"]}],
    )
    assert out == (0, 0, 0)
    client.apply_tag_actions.assert_not_awaited()


@pytest.mark.asyncio
async def test_apply_chunk_when_metadata_fetch_fails_still_writes_tags():
    client = MagicMock()
    client.get_file_metadata = AsyncMock(side_effect=OSError("metadata down"))
    client.apply_tag_actions = AsyncMock()
    out = await _apply_results_chunk(
        client,
        "sk",
        [{"file_id": 1, "hash": "h1", "tags": ["new"]}],
    )
    assert out == (1, 1, 0)
    client.apply_tag_actions.assert_awaited_once()


@pytest.mark.asyncio
async def test_apply_chunk_skips_when_all_tags_are_duplicates_and_no_removals():
    client = MagicMock()
    client.get_file_metadata = AsyncMock(
        return_value=[
            {
                "file_id": 1,
                "hash": "h1",
                "tags": {"sk": {"storage_tags": {"0": ["dup"]}, "display_tags": {}}},
            },
        ],
    )
    client.apply_tag_actions = AsyncMock()
    out = await _apply_results_chunk(
        client,
        "sk",
        [{"file_id": 1, "hash": "h1", "tags": ["dup"]}],
    )
    assert out == (0, 0, 1)
    client.apply_tag_actions.assert_not_awaited()


@pytest.mark.asyncio
async def test_apply_chunk_propagates_hydrus_apply_failure():
    client = MagicMock()
    client.get_file_metadata = AsyncMock(return_value=[])
    client.apply_tag_actions = AsyncMock(side_effect=RuntimeError("hydrus apply failed"))
    with pytest.raises(RuntimeError, match="hydrus apply failed"):
        await _apply_results_chunk(
            client,
            "sk",
            [{"file_id": 1, "hash": "h1", "tags": ["x"]}],
        )

"""Final WebSocket results trimmed to pending tags vs Hydrus storage."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.config import AppConfig
from backend.routes.tagger import _trim_ws_results_to_pending_for_service


@pytest.mark.asyncio
async def test_trim_removes_tags_already_in_hydrus_storage():
    cfg = AppConfig()
    results = [
        {
            "file_id": 1,
            "hash": "h1",
            "tags": ["keep_me", "already_there"],
            "general_tags": {"keep_me": 0.9, "already_there": 0.8},
            "character_tags": {},
            "rating_tags": {},
        }
    ]
    client = MagicMock()
    client.get_file_metadata = AsyncMock(
        return_value=[
            {
                "file_id": 1,
                "hash": "h1",
                "tags": {
                    "svc": {
                        "storage_tags": {"0": ["already_there"]},
                        "display_tags": {},
                    }
                },
            }
        ]
    )
    n = await _trim_ws_results_to_pending_for_service(client, "svc", results, cfg)
    assert n == 1
    assert results[0]["tags"] == ["keep_me"]
    assert "already_there" not in results[0]["general_tags"]
    assert "keep_me" in results[0]["general_tags"]

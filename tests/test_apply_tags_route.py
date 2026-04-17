"""POST /api/tagger/apply deduplicates against Hydrus storage_tags."""

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]

from fastapi.testclient import TestClient

import backend.config as config_module
import backend.routes.tagger_http as tagger_http_routes


@pytest.fixture
def apply_client(monkeypatch):
    hydrus = MagicMock()
    hydrus.add_tags = AsyncMock()
    hydrus.apply_tag_actions = AsyncMock()
    hydrus.get_file_metadata = AsyncMock(
        return_value=[
            {
                "file_id": 42,
                "hash": "abc123",
                "tags": {
                    "svckey": {
                        "storage_tags": {"0": ["blue_hair", "1girl"]},
                        "display_tags": {},
                    }
                },
            }
        ]
    )
    monkeypatch.setattr(tagger_http_routes, "get_config", config_module.get_config)
    monkeypatch.setattr(tagger_http_routes, "HydrusClient", lambda *a, **k: hydrus)

    from backend.app import app

    with TestClient(app) as client:
        yield client, hydrus


def test_apply_skips_tags_already_in_storage(apply_client):
    client, hydrus = apply_client
    body = {
        "service_key": "svckey",
        "results": [
            {
                "file_id": 42,
                "hash": "abc123",
                "tags": ["blue hair", "1girl", "solo", "character:hatsune miku"],
            }
        ],
    }
    resp = client.post("/api/tagger/apply", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["applied"] == 1
    assert data["skipped_duplicate_tags"] == 2
    hydrus.apply_tag_actions.assert_awaited_once()
    call = hydrus.apply_tag_actions.await_args
    assert set(call.kwargs["add_tags"]) == {"solo", "character:hatsune miku"}
    assert call.kwargs["remove_tags"] == []


def test_apply_logs_debug_http_route_chunking(apply_client, caplog):
    caplog.set_level(logging.DEBUG, logger="backend.routes.tagger_http")
    client, _ = apply_client
    resp = client.post(
        "/api/tagger/apply",
        json={
            "service_key": "svckey",
            "results": [
                {
                    "file_id": 42,
                    "hash": "abc123",
                    "tags": ["solo"],
                }
            ],
        },
    )
    assert resp.status_code == 200
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "apply_tags http_route rows=" in joined
    assert "applies_to_websocket_tagging_only" in joined


def test_apply_logs_hydrus_duplicate_tag_metrics(apply_client, caplog):
    caplog.set_level(logging.INFO, logger="backend.routes.tagger_http")
    caplog.set_level(logging.INFO, logger="backend.routes.tagger_apply")
    client, _ = apply_client
    resp = client.post(
        "/api/tagger/apply",
        json={
            "service_key": "svckey",
            "results": [
                {
                    "file_id": 42,
                    "hash": "abc123",
                    "tags": ["blue hair", "1girl", "solo", "character:hatsune miku"],
                }
            ],
        },
    )
    assert resp.status_code == 200
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "apply_tags chunk" in joined
    assert "hydrus_duplicate_tag_strings_skipped=2" in joined


def test_apply_splits_payload_into_http_batches(monkeypatch, apply_client):
    snap = config_module.get_config()
    batched_cfg = snap.model_copy(update={"apply_tags_http_batch_size": 1})
    monkeypatch.setattr(config_module, "get_config", lambda: batched_cfg)
    monkeypatch.setattr(tagger_http_routes, "get_config", lambda: batched_cfg)

    client, hydrus = apply_client
    async def _meta(**kw):
        fids = kw.get("file_ids") or []
        return [
            {"file_id": int(x), "hash": f"h{x}", "tags": {}} for x in fids
        ]

    hydrus.get_file_metadata = AsyncMock(side_effect=_meta)

    resp = client.post(
        "/api/tagger/apply",
        json={
            "service_key": "svckey",
            "results": [
                {"file_id": 10, "hash": "h10", "tags": ["a"]},
                {"file_id": 20, "hash": "h20", "tags": ["b"]},
            ],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["applied"] == 2
    assert hydrus.get_file_metadata.await_count == 2
    assert hydrus.apply_tag_actions.await_count == 2


def test_apply_sends_single_http_chunk_when_results_fewer_than_batch_size(
    monkeypatch, apply_client,
):
    """When N < apply_tags_http_batch_size, one iteration still runs (range(0, N, bs) → off=0)."""
    snap = config_module.get_config()
    large_batch_cfg = snap.model_copy(update={"apply_tags_http_batch_size": 512})
    monkeypatch.setattr(config_module, "get_config", lambda: large_batch_cfg)
    monkeypatch.setattr(tagger_http_routes, "get_config", lambda: large_batch_cfg)

    client, hydrus = apply_client

    async def _meta(**kw):
        fids = kw.get("file_ids") or []
        return [
            {"file_id": int(x), "hash": f"h{x}", "tags": {}} for x in fids
        ]

    hydrus.get_file_metadata = AsyncMock(side_effect=_meta)

    resp = client.post(
        "/api/tagger/apply",
        json={
            "service_key": "svckey",
            "results": [
                {"file_id": 1, "hash": "h1", "tags": ["a"]},
                {"file_id": 2, "hash": "h2", "tags": ["b"]},
                {"file_id": 3, "hash": "h3", "tags": ["c"]},
            ],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["applied"] == 3
    assert hydrus.get_file_metadata.await_count == 1
    meta_call = hydrus.get_file_metadata.await_args
    assert len(meta_call.kwargs.get("file_ids") or []) == 3
    assert hydrus.apply_tag_actions.await_count == 3


def test_apply_forwards_remove_tags_to_hydrus_actions(apply_client):
    client, hydrus = apply_client
    hydrus.get_file_metadata = AsyncMock(return_value=[{"file_id": 42, "hash": "abc123", "tags": {}}])
    resp = client.post(
        "/api/tagger/apply",
        json={
            "service_key": "svckey",
            "results": [
                {
                    "file_id": 42,
                    "hash": "abc123",
                    "tags": ["new_tag"],
                    "remove_tags": ["old_tag"],
                }
            ],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    hydrus.apply_tag_actions.assert_awaited_once()
    call = hydrus.apply_tag_actions.await_args
    assert call.kwargs["add_tags"] == ["new_tag"]
    assert call.kwargs["remove_tags"] == ["old_tag"]


def test_apply_remove_only_still_calls_hydrus(apply_client):
    """Rows with no new tags but non-empty remove_tags must still write to Hydrus."""
    client, hydrus = apply_client
    hydrus.get_file_metadata = AsyncMock(
        return_value=[
            {
                "file_id": 42,
                "hash": "abc123",
                "tags": {
                    "svckey": {
                        "storage_tags": {"0": ["stale"]},
                        "display_tags": {},
                    }
                },
            }
        ],
    )
    resp = client.post(
        "/api/tagger/apply",
        json={
            "service_key": "svckey",
            "results": [
                {
                    "file_id": 42,
                    "hash": "abc123",
                    "tags": [],
                    "remove_tags": ["stale"],
                }
            ],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    hydrus.apply_tag_actions.assert_awaited_once()
    call = hydrus.apply_tag_actions.await_args
    assert call.kwargs["add_tags"] == []
    assert call.kwargs["remove_tags"] == ["stale"]

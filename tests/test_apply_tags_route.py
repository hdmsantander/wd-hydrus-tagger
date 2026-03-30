"""POST /api/tagger/apply deduplicates against Hydrus storage_tags."""

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import backend.config as config_module
import backend.routes.tagger as tagger_routes


@pytest.fixture
def apply_client(monkeypatch):
    hydrus = MagicMock()
    hydrus.add_tags = AsyncMock()
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
    monkeypatch.setattr(tagger_routes, "get_config", config_module.get_config)
    monkeypatch.setattr(tagger_routes, "HydrusClient", lambda *a, **k: hydrus)

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
    hydrus.add_tags.assert_awaited_once()
    call = hydrus.add_tags.await_args
    assert set(call.kwargs["tags"]) == {"solo", "character:hatsune miku"}


def test_apply_logs_hydrus_duplicate_tag_metrics(apply_client, caplog):
    caplog.set_level(logging.INFO, logger="backend.routes.tagger")
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
    assert "apply_tags metrics hydrus_duplicate_tag_strings_skipped=2" in joined


def test_apply_splits_payload_into_http_batches(monkeypatch, apply_client):
    snap = config_module.get_config()
    batched_cfg = snap.model_copy(update={"apply_tags_http_batch_size": 1})
    monkeypatch.setattr(config_module, "get_config", lambda: batched_cfg)
    monkeypatch.setattr(tagger_routes, "get_config", lambda: batched_cfg)

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
    assert hydrus.add_tags.await_count == 2

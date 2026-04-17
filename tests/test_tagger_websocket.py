"""WebSocket tagging session (pause / flush / batch metadata) with mocked inference."""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import backend.config as config_module
import backend.routes.tagger_ws as tagger_ws_routes
from backend.services.tagging_session_registry import (
    TaggingSessionHandle,
    clear_tagging_public_snapshot,
    get_public_session_status,
    register_tagging_session,
    unregister_tagging_session,
    update_tagging_public_snapshot,
)

pytestmark = [pytest.mark.full, pytest.mark.ws]


def _ws_recv_skip_plan(ws):
    """First post-run message may be queue_plan (prefetch analysis); tests expect progress/apply."""
    msg = ws.receive_json()
    while msg.get("type") == "queue_plan":
        msg = ws.receive_json()
    return msg


class FakeTaggingService:
    """Minimal stand-in for TaggingService.tag_files / load_model."""

    def __init__(self):
        self._perf_batch_seq = 0
        self.ensure_model_calls: list[dict] = []

    def _resolve_ort_threads(self, ort_intra_op_threads, ort_inter_op_threads):
        c = config_module.get_config()
        ia = c.cpu_intra_op_threads if ort_intra_op_threads is None else int(ort_intra_op_threads)
        ie = c.cpu_inter_op_threads if ort_inter_op_threads is None else int(ort_inter_op_threads)
        return max(1, min(64, ia)), max(1, min(16, ie))

    def load_model(self, name: str, **kwargs) -> None:
        pass

    async def ensure_model(self, explicit_name, **kwargs):
        self.ensure_model_calls.append(dict(kwargs))

    async def tag_files(self, client, file_ids, **kwargs):
        batch_metrics_out = kwargs.get("batch_metrics_out")
        if batch_metrics_out is not None:
            self._perf_batch_seq += 1
            batch_metrics_out.append(
                {
                    "batch_index": self._perf_batch_seq,
                    "fetch_s": 0.001,
                    "predict_s": 0.002,
                    "files_in_batch": len(file_ids),
                    "skipped_pre_infer": 0,
                    "predict_queue": len(file_ids),
                },
            )
        return [
            {
                "file_id": fid,
                "hash": f"h{fid}",
                "general_tags": {},
                "character_tags": {},
                "rating_tags": {},
                "formatted_tags": [f"tag:{fid}"],
                "tags": [f"tag:{fid}"],
            }
            for fid in file_ids
        ]


@pytest.fixture
def ws_client(monkeypatch):
    monkeypatch.setattr(tagger_ws_routes, "get_config", config_module.get_config)
    monkeypatch.setattr(
        tagger_ws_routes.TaggingService,
        "get_instance",
        lambda _cfg: FakeTaggingService(),
    )
    # Simulate Hydrus storage so final-result trimming matches incremental applies.
    storage: dict[str, dict[str, list[str]]] = {}

    async def add_tags_impl(hash_, service_key, tags):
        if hash_ not in storage:
            storage[hash_] = {}
        if service_key not in storage[hash_]:
            storage[hash_][service_key] = []
        storage[hash_][service_key].extend(tags)

    async def apply_tag_actions_impl(hash_, service_key, *, add_tags, remove_tags):
        if add_tags:
            await add_tags_impl(hash_, service_key, add_tags)
        if not remove_tags:
            return
        bucket = storage.get(hash_, {}).get(service_key)
        if not bucket:
            return
        for t in remove_tags:
            while t in bucket:
                bucket.remove(t)

    async def get_metadata_impl(file_ids):
        out = []
        for fid in file_ids:
            h = f"h{int(fid)}"
            tags_root = {}
            for sk, lst in storage.get(h, {}).items():
                tags_root[sk] = {"storage_tags": {"0": list(lst)}, "display_tags": {}}
            out.append({
                "file_id": int(fid),
                "hash": h,
                "size": 50_000 + int(fid),
                "tags": tags_root,
            })
        return out

    hydrus = MagicMock()
    hydrus.add_tags = AsyncMock(side_effect=add_tags_impl)
    hydrus.apply_tag_actions = AsyncMock(side_effect=apply_tag_actions_impl)
    hydrus.get_file_metadata = AsyncMock(side_effect=get_metadata_impl)
    monkeypatch.setattr(tagger_ws_routes, "HydrusClient", lambda *a, **k: hydrus)

    from backend.app import app

    with TestClient(app) as client:
        yield client, hydrus


class MixedMarkerFakeTaggingService(FakeTaggingService):
    """One file skipped (same model marker), one inferred with stale marker removals."""

    async def tag_files(self, client, file_ids, **kwargs):
        out = []
        for i, fid in enumerate(file_ids):
            if i == 0:
                out.append({
                    "file_id": fid,
                    "hash": f"h{fid}",
                    "general_tags": {},
                    "character_tags": {},
                    "rating_tags": {},
                    "formatted_tags": [],
                    "tags": [],
                    "skipped_inference": True,
                    "skip_reason": "wd_model_marker_present",
                    "wd_stale_markers_removed": 0,
                })
            else:
                out.append({
                    "file_id": fid,
                    "hash": f"h{fid}",
                    "general_tags": {},
                    "character_tags": {},
                    "rating_tags": {},
                    "formatted_tags": [f"tag:{fid}"],
                    "tags": [f"tag:{fid}"],
                    "wd_stale_markers_removed": 2,
                })
        return out


@pytest.fixture
def ws_client_marker_mixed(monkeypatch):
    monkeypatch.setattr(tagger_ws_routes, "get_config", config_module.get_config)
    monkeypatch.setattr(
        tagger_ws_routes.TaggingService,
        "get_instance",
        lambda _cfg: MixedMarkerFakeTaggingService(),
    )
    storage: dict[str, dict[str, list[str]]] = {}

    async def add_tags_impl(hash_, service_key, tags):
        if hash_ not in storage:
            storage[hash_] = {}
        if service_key not in storage[hash_]:
            storage[hash_][service_key] = []
        storage[hash_][service_key].extend(tags)

    async def apply_tag_actions_impl(hash_, service_key, *, add_tags, remove_tags):
        if add_tags:
            await add_tags_impl(hash_, service_key, add_tags)
        if not remove_tags:
            return
        bucket = storage.get(hash_, {}).get(service_key)
        if not bucket:
            return
        for t in remove_tags:
            while t in bucket:
                bucket.remove(t)

    async def get_metadata_impl(file_ids):
        out = []
        for fid in file_ids:
            h = f"h{int(fid)}"
            tags_root = {}
            for sk, lst in storage.get(h, {}).items():
                tags_root[sk] = {"storage_tags": {"0": list(lst)}, "display_tags": {}}
            out.append({
                "file_id": int(fid),
                "hash": h,
                "size": 50_000 + int(fid),
                "tags": tags_root,
            })
        return out

    hydrus = MagicMock()
    hydrus.add_tags = AsyncMock(side_effect=add_tags_impl)
    hydrus.apply_tag_actions = AsyncMock(side_effect=apply_tag_actions_impl)
    hydrus.get_file_metadata = AsyncMock(side_effect=get_metadata_impl)
    monkeypatch.setattr(tagger_ws_routes, "HydrusClient", lambda *a, **k: hydrus)

    from backend.app import app

    with TestClient(app) as client:
        yield client, hydrus


def test_ws_rejects_invalid_first_message(ws_client):
    client, _ = ws_client
    with client.websocket_connect("/api/tagger/ws/progress") as ws:
        ws.send_json({"action": "pause"})
        msg = _ws_recv_skip_plan(ws)
        assert msg["type"] == "error"


def test_ws_rejects_empty_file_ids(ws_client):
    client, _ = ws_client
    with client.websocket_connect("/api/tagger/ws/progress") as ws:
        ws.send_json({"action": "run", "file_ids": []})
        msg = _ws_recv_skip_plan(ws)
        assert msg["type"] == "error"
        assert msg.get("code") == "empty_queue"


def test_ws_rejects_non_list_file_ids(ws_client):
    client, _ = ws_client
    with client.websocket_connect("/api/tagger/ws/progress") as ws:
        ws.send_json({"action": "run", "file_ids": "bad"})
        msg = _ws_recv_skip_plan(ws)
        assert msg["type"] == "error"
        assert msg.get("code") == "invalid_file_ids"


def test_get_session_status_inactive():
    from backend.app import app

    with TestClient(app) as client:
        r = client.get("/api/tagger/session/status")
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert data["active"] is False
        assert data.get("snapshot") is None


def test_ws_rejects_second_run_while_session_registered(ws_client):
    """Only one tagging WebSocket run at a time; second client gets tagging_busy."""
    cancel = asyncio.Event()
    flush = asyncio.Event()
    h = TaggingSessionHandle(cancel_event=cancel, flush_event=flush)
    register_tagging_session(h)
    try:
        assert get_public_session_status()["active"] is True
        client, _ = ws_client
        with client.websocket_connect("/api/tagger/ws/progress") as ws:
            ws.send_json({
                "action": "run",
                "file_ids": [1],
                "batch_size": 1,
                "apply_tags_every_n": 0,
            })
            msg = _ws_recv_skip_plan(ws)
            assert msg["type"] == "error"
            assert msg.get("code") == "tagging_busy"
            assert "snapshot" in msg
    finally:
        unregister_tagging_session(h)
        assert get_public_session_status()["active"] is False


def test_ws_progress_includes_inference_batch(ws_client):
    client, hydrus = ws_client
    with client.websocket_connect("/api/tagger/ws/progress") as ws:
        ws.send_json({
            "action": "run",
            "file_ids": [101],
            "batch_size": 1,
            "apply_tags_every_n": 0,
        })
        p = _ws_recv_skip_plan(ws)
        assert p["type"] == "progress"
        assert p["inference_batch"] == 1
        assert p["batch_inferred"] == 1
        assert p["current"] == 1
        assert p["batches_completed"] == 1
        assert p["batches_total"] == 1
        assert p["total_applied"] == 0
        assert p["total_tags_written"] == 0
        assert p.get("batch_skipped_same_model_marker") == 0
        assert p.get("batch_skipped_higher_tier_model_marker") == 0
        assert p.get("batch_wd_stale_markers_removed") == 0
        assert p.get("cumulative_skipped_same_model_marker") == 0
        assert p.get("cumulative_skipped_higher_tier_model_marker") == 0
        assert p.get("cumulative_wd_stale_markers_removed") == 0
        done = _ws_recv_skip_plan(ws)
        assert done["type"] == "complete"
        assert done["inference_batch"] == 1
        assert done["total_processed"] == 1
        assert done["batches_completed"] == 1
        assert done["batches_total"] == 1
        assert done["total_tags_written"] == 0
        assert done.get("pending_hydrus_files") == 0
        assert done.get("cumulative_skipped_same_model_marker") == 0
        assert done.get("cumulative_skipped_higher_tier_model_marker") == 0
        assert done.get("cumulative_wd_stale_markers_removed") == 0
        assert done["results"][0]["tags"] == ["tag:101"]
    hydrus.apply_tag_actions.assert_not_awaited()


def test_ws_pause_resume_between_batches(ws_client):
    client, _ = ws_client
    with client.websocket_connect("/api/tagger/ws/progress") as ws:
        ws.send_json({
            "action": "run",
            "file_ids": [1, 2],
            "batch_size": 1,
            "apply_tags_every_n": 0,
        })
        m1 = _ws_recv_skip_plan(ws)
        assert m1["type"] == "progress"
        assert m1["current"] == 1

        ws.send_json({"action": "pause"})
        ack = _ws_recv_skip_plan(ws)
        assert ack["type"] == "control_ack"
        assert ack["action"] == "pause"

        ws.send_json({"action": "resume"})
        ack2 = _ws_recv_skip_plan(ws)
        assert ack2["type"] == "control_ack"
        assert ack2["action"] == "resume"

        m2 = _ws_recv_skip_plan(ws)
        assert m2["type"] == "progress"
        assert m2["current"] == 2

        end = _ws_recv_skip_plan(ws)
        assert end["type"] == "complete"


def test_ws_manual_flush_pending_to_hydrus(ws_client):
    client, hydrus = ws_client
    with client.websocket_connect("/api/tagger/ws/progress") as ws:
        ws.send_json({
            "action": "run",
            "file_ids": [10, 20],
            "batch_size": 1,
            "apply_tags_every_n": 99,
            "service_key": "my-service",
        })
        p1 = _ws_recv_skip_plan(ws)
        assert p1["type"] == "progress"
        assert p1["current"] == 1

        ws.send_json({"action": "flush"})
        applied = _ws_recv_skip_plan(ws)
        assert applied["type"] == "tags_applied"
        assert applied["manual_flush"] is True
        assert applied["count"] == 1
        assert applied["pending_remaining"] == 0

        p2 = _ws_recv_skip_plan(ws)
        assert p2["type"] == "progress"
        assert p2["current"] == 2

        # End-of-run flush for remaining pending when apply_every > 0 (second file).
        final_apply = _ws_recv_skip_plan(ws)
        assert final_apply["type"] == "tags_applied"
        assert final_apply["manual_flush"] is False
        assert final_apply["count"] == 1

        end = _ws_recv_skip_plan(ws)
        assert end["type"] == "complete"
        assert end.get("pending_hydrus_files") == 0
        assert end["results"][0]["tags"] == []
        assert end["results"][1]["tags"] == []

    hydrus.apply_tag_actions.assert_awaited()
    assert hydrus.apply_tag_actions.await_count >= 1


def test_ws_tag_all_final_flush_when_inference_batch_exceeds_file_count(ws_client):
    """Incremental Hydrus apply uses bs as threshold; remainder must flush at end-of-run."""
    client, hydrus = ws_client
    with client.websocket_connect("/api/tagger/ws/progress") as ws:
        ws.send_json({
            "action": "run",
            "tag_all": True,
            "file_ids": [1, 2, 3],
            "batch_size": 16,
            "apply_tags_every_n": 1,
            "service_key": "my-service",
        })
        # No per-batch tags_applied while len(pending) < inference batch (16).
        p1 = _ws_recv_skip_plan(ws)
        assert p1["type"] == "progress"
        assert p1["current"] == 3
        assert p1["batch_inferred"] == 3
        final_apply = _ws_recv_skip_plan(ws)
        assert final_apply["type"] == "tags_applied"
        assert final_apply["manual_flush"] is False
        assert final_apply["count"] == 3
        assert final_apply["pending_remaining"] == 0
        end = _ws_recv_skip_plan(ws)
        assert end["type"] == "complete"
        assert end.get("pending_hydrus_files") == 0
    assert hydrus.apply_tag_actions.await_count == 3


def test_ws_non_tag_all_final_flush_when_apply_every_exceeds_last_batch(ws_client):
    """Without tag_all, apply_every_n is fixed; a short final batch still flushes at end."""
    client, hydrus = ws_client
    with client.websocket_connect("/api/tagger/ws/progress") as ws:
        ws.send_json({
            "action": "run",
            "tag_all": False,
            "file_ids": [1, 2, 3],
            "batch_size": 8,
            "apply_tags_every_n": 16,
            "service_key": "my-service",
        })
        p1 = _ws_recv_skip_plan(ws)
        assert p1["type"] == "progress"
        assert p1["current"] == 3
        final_apply = _ws_recv_skip_plan(ws)
        assert final_apply["type"] == "tags_applied"
        assert final_apply["count"] == 3
        end = _ws_recv_skip_plan(ws)
        assert end["type"] == "complete"
    assert hydrus.apply_tag_actions.await_count == 3


def test_ws_incremental_apply_emits_tag_counts(ws_client):
    client, hydrus = ws_client
    with client.websocket_connect("/api/tagger/ws/progress") as ws:
        ws.send_json({
            "action": "run",
            "file_ids": [1, 2],
            "batch_size": 1,
            "apply_tags_every_n": 1,
            "service_key": "my-service",
        })
        a1 = _ws_recv_skip_plan(ws)
        assert a1["type"] == "tags_applied"
        assert a1["count"] == 1
        assert a1["chunk_tag_count"] == 1
        assert a1["total_tags_written"] == 1
        p1 = _ws_recv_skip_plan(ws)
        assert p1["type"] == "progress"
        assert p1["batches_completed"] == 1
        assert p1["total_applied"] == 1
        assert p1["total_tags_written"] == 1
        a2 = _ws_recv_skip_plan(ws)
        assert a2["type"] == "tags_applied"
        assert a2["total_tags_written"] == 2
        p2 = _ws_recv_skip_plan(ws)
        assert p2["type"] == "progress"
        assert p2["batches_completed"] == 2
        end = _ws_recv_skip_plan(ws)
        assert end["type"] == "complete"
        assert end["total_tags_written"] == 2
        assert end.get("pending_hydrus_files") == 0
        assert end["results"][0]["tags"] == []
        assert end["results"][1]["tags"] == []
    assert hydrus.apply_tag_actions.await_count == 2


def test_ws_cancel_stops_with_partial_results(ws_client):
    client, _ = ws_client
    with client.websocket_connect("/api/tagger/ws/progress") as ws:
        ws.send_json({
            "action": "run",
            "file_ids": [1, 2],
            "batch_size": 1,
            "apply_tags_every_n": 0,
        })
        p = _ws_recv_skip_plan(ws)
        assert p["type"] == "progress"
        assert p["current"] == 1
        ws.send_json({"action": "cancel"})
        stopping = _ws_recv_skip_plan(ws)
        assert stopping["type"] == "stopping"
        assert "pending_hydrus_queue" in stopping
        end = _ws_recv_skip_plan(ws)
        assert end["type"] == "stopped"
        assert end["stopped"] is True
        assert end["total_processed"] == 1


def test_ws_cancel_logs_user_winding_down(ws_client, caplog):
    caplog.set_level(logging.INFO, logger="backend.routes.tagger_ws")
    client, _ = ws_client
    with client.websocket_connect("/api/tagger/ws/progress") as ws:
        ws.send_json({
            "action": "run",
            "file_ids": [1, 2],
            "batch_size": 1,
            "apply_tags_every_n": 0,
        })
        _ws_recv_skip_plan(ws)
        ws.send_json({"action": "cancel"})
        assert _ws_recv_skip_plan(ws)["type"] == "stopping"
        assert _ws_recv_skip_plan(ws)["type"] == "stopped"
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "tagging_ws user_cancel received" in joined
    assert "tagging_ws winding_down exit_loop" in joined
    assert "reason=user_cancel" in joined
    assert "tagging_ws user_stop_complete" in joined


def test_ws_cancel_pending_incremental_final_flush(ws_client, caplog):
    caplog.set_level(logging.INFO, logger="backend.routes.tagger_ws")
    client, hydrus = ws_client
    with client.websocket_connect("/api/tagger/ws/progress") as ws:
        ws.send_json({
            "action": "run",
            "file_ids": [1, 2],
            "batch_size": 1,
            "apply_tags_every_n": 2,
            "service_key": "my-service",
        })
        _ws_recv_skip_plan(ws)
        ws.send_json({"action": "cancel"})
        st = _ws_recv_skip_plan(ws)
        assert st["type"] == "stopping"
        assert st.get("pending_hydrus_queue") == 1
        ta = _ws_recv_skip_plan(ws)
        assert ta["type"] == "tags_applied"
        end = _ws_recv_skip_plan(ws)
        assert end["type"] == "stopped"
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "tagging_ws winding_down final_hydrus_flush files=1" in joined
    assert hydrus.apply_tag_actions.await_count >= 1


def test_public_snapshot_user_stopping_fields():
    clear_tagging_public_snapshot()
    update_tagging_public_snapshot(
        {"type": "stopping", "message": "Stopping - test", "pending_hydrus_queue": 2},
        model_name="m1",
        total_files=5,
    )
    snap = get_public_session_status().get("snapshot") or {}
    assert snap.get("phase") == "stopping"
    assert snap.get("stopping_source") == "user"
    assert snap.get("detail") == "Stopping - test"
    assert snap.get("pending_hydrus_queue") == 2
    clear_tagging_public_snapshot()


def test_public_snapshot_server_shutting_down_source():
    clear_tagging_public_snapshot()
    update_tagging_public_snapshot(
        {"type": "server_shutting_down", "message": "shutdown"},
        model_name="m2",
        total_files=3,
    )
    snap = get_public_session_status().get("snapshot") or {}
    assert snap.get("stopping_source") == "server"
    assert snap.get("detail") == "shutdown"
    clear_tagging_public_snapshot()


@pytest.mark.slow
def test_ws_progress_aggregates_same_model_skip_and_stale_marker_removals(ws_client_marker_mixed):
    client, _ = ws_client_marker_mixed
    with client.websocket_connect("/api/tagger/ws/progress") as ws:
        ws.send_json({
            "action": "run",
            "file_ids": [201, 202],
            "batch_size": 2,
            "apply_tags_every_n": 0,
        })
        p = _ws_recv_skip_plan(ws)
        assert p["type"] == "progress"
        assert p["batch_skipped_same_model_marker"] == 1
        assert p["batch_wd_stale_markers_removed"] == 2
        assert p["cumulative_skipped_same_model_marker"] == 1
        assert p["cumulative_wd_stale_markers_removed"] == 2
        end = _ws_recv_skip_plan(ws)
        assert end["type"] == "complete"
        assert end["cumulative_skipped_same_model_marker"] == 1
        assert end["cumulative_wd_stale_markers_removed"] == 2


@pytest.mark.slow
def test_ws_tag_all_performance_tuning_includes_timings(ws_client):
    client, _ = ws_client
    with client.websocket_connect("/api/tagger/ws/progress") as ws:
        ws.send_json({
            "action": "run",
            "file_ids": [1, 2],
            "batch_size": 2,
            "apply_tags_every_n": 0,
            "tag_all": True,
            "performance_tuning": True,
        })
        p = _ws_recv_skip_plan(ws)
        assert p["type"] == "progress"
        assert "performance_tuning" in p
        pt = p["performance_tuning"]
        assert pt.get("fetch_s") is not None
        assert pt.get("predict_s") is not None
        assert "hydrus_apply_batch_s" in pt
        assert pt.get("effective_batch") == 2
        end = _ws_recv_skip_plan(ws)
        assert end["type"] == "complete"
        tr = end.get("tuning_report")
        assert tr is not None
        assert tr.get("schema_version") == 1
        assert tr.get("batches_recorded") == 1
        assert "aggregate" in tr
        assert tr["aggregate"].get("sum_wall_s") is not None


@pytest.mark.slow
def test_ws_performance_tuning_history_window_and_report_multi_batch(ws_client):
    client, _ = ws_client
    with client.websocket_connect("/api/tagger/ws/progress") as ws:
        ws.send_json({
            "action": "run",
            "file_ids": [1, 2, 3, 4],
            "batch_size": 2,
            "apply_tags_every_n": 0,
            "tag_all": True,
            "performance_tuning": True,
            "performance_tuning_window": 2,
        })
        p1 = _ws_recv_skip_plan(ws)
        assert p1["type"] == "progress"
        h1 = p1.get("performance_tuning_history") or []
        assert len(h1) == 1
        p2 = _ws_recv_skip_plan(ws)
        assert p2["type"] == "progress"
        h2 = p2.get("performance_tuning_history") or []
        assert len(h2) == 2
        assert h2[0]["batch_index"] == 1
        assert h2[1]["batch_index"] == 2
        end = _ws_recv_skip_plan(ws)
        assert end["type"] == "complete"
        tr = end["tuning_report"]
        assert tr["batches_recorded"] == 2
        assert tr["history_window"] == 2
        assert len(tr["batch_series"]) == 2


@pytest.mark.slow
def test_ws_session_auto_tune_threads_sets_autotune_summary_flag(ws_client):
    client, _ = ws_client
    with client.websocket_connect("/api/tagger/ws/progress") as ws:
        ws.send_json({
            "action": "run",
            "file_ids": [1],
            "batch_size": 1,
            "apply_tags_every_n": 0,
            "tag_all": True,
            "session_auto_tune": True,
            "performance_tuning": True,
            "session_auto_tune_threads": True,
            "tuning_bounds": {"cpu_intra_op_threads": {"min": 4, "max": 4}},
        })
        while True:
            m = _ws_recv_skip_plan(ws)
            if m.get("type") == "complete":
                tr = m.get("tuning_report") or {}
                at = tr.get("autotune") or {}
                assert at.get("session_auto_tune_threads") is True
                break


@pytest.mark.slow
def test_ws_learning_calibration_no_hydrus_during_learning_then_flush(ws_client):
    """Phase L defers Hydrus; Phase C applies; then learning prefix flushes without re-inference."""
    client, hydrus = ws_client
    ids = list(range(1, 41))
    with client.websocket_connect("/api/tagger/ws/progress") as ws:
        ws.send_json({
            "action": "run",
            "file_ids": ids,
            "batch_size": 8,
            "apply_tags_every_n": 1,
            "tag_all": True,
            "service_key": "my-service",
            "learning_phase_calibration": True,
            "learning_fraction": 0.1,
        })
        for _ in range(4):
            p = _ws_recv_skip_plan(ws)
            assert p["type"] == "progress"
            assert p.get("calibration_phase") == "learning"
            assert hydrus.apply_tag_actions.await_count == 0
        ta = _ws_recv_skip_plan(ws)
        assert ta["type"] == "tags_applied"
        assert ta.get("learning_calibration_flush") is not True
        p = _ws_recv_skip_plan(ws)
        assert p["type"] == "progress"
        assert p.get("calibration_phase") == "commit"
        for _ in range(4):
            ta = _ws_recv_skip_plan(ws)
            assert ta["type"] == "tags_applied"
            assert ta.get("learning_calibration_flush") is True
        end = _ws_recv_skip_plan(ws)
        assert end["type"] == "complete"
    assert hydrus.apply_tag_actions.await_count == 40


@pytest.mark.slow
def test_ws_learning_calibration_bytes_scope_reports_effective_bytes(ws_client):
    client, hydrus = ws_client
    ids = list(range(1, 41))
    with client.websocket_connect("/api/tagger/ws/progress") as ws:
        ws.send_json({
            "action": "run",
            "file_ids": ids,
            "batch_size": 8,
            "apply_tags_every_n": 1,
            "tag_all": True,
            "service_key": "my-service",
            "learning_phase_calibration": True,
            "learning_fraction": 0.1,
            "learning_scope": "bytes",
        })
        while True:
            m = _ws_recv_skip_plan(ws)
            if m.get("type") == "complete":
                lc = m.get("learning_calibration") or {}
                assert lc.get("learning_scope_effective") == "bytes"
                assert lc.get("total_bytes_known", 0) > 0
                break
    assert hydrus.apply_tag_actions.await_count == 40


@pytest.mark.slow
def test_ws_learning_calibration_prefix_cap_respects_max_cached(monkeypatch, ws_client):
    import backend.config as config_module
    import backend.routes.tagger_ws as tagger_ws_routes

    capped = config_module.get_config().model_copy(update={"max_learning_cached_files": 16})
    monkeypatch.setattr(tagger_ws_routes, "get_config", lambda: capped)

    client, hydrus = ws_client
    ids = list(range(1, 41))
    with client.websocket_connect("/api/tagger/ws/progress") as ws:
        ws.send_json({
            "action": "run",
            "file_ids": ids,
            "batch_size": 8,
            "apply_tags_every_n": 1,
            "tag_all": True,
            "service_key": "my-service",
            "learning_phase_calibration": True,
            "learning_fraction": 0.1,
        })
        while True:
            m = _ws_recv_skip_plan(ws)
            if m.get("type") == "complete":
                lc = m.get("learning_calibration") or {}
                assert lc.get("learning_prefix_capped") == 16
                break
    assert hydrus.apply_tag_actions.await_count == 40


@pytest.mark.slow
def test_ws_learning_calibration_cancel_in_learning_skips_learning_flush(ws_client):
    client, hydrus = ws_client
    ids = list(range(1, 41))
    with client.websocket_connect("/api/tagger/ws/progress") as ws:
        ws.send_json({
            "action": "run",
            "file_ids": ids,
            "batch_size": 8,
            "apply_tags_every_n": 1,
            "tag_all": True,
            "service_key": "my-service",
            "learning_phase_calibration": True,
            "learning_fraction": 0.1,
        })
        _ws_recv_skip_plan(ws)
        ws.send_json({"action": "cancel"})
        assert _ws_recv_skip_plan(ws)["type"] == "stopping"
        end = _ws_recv_skip_plan(ws)
        assert end["type"] == "stopped"
    assert hydrus.apply_tag_actions.await_count == 0


@pytest.mark.slow
def test_ws_session_auto_tune_tag_all_includes_autotune_in_report(ws_client):
    client, _ = ws_client
    with client.websocket_connect("/api/tagger/ws/progress") as ws:
        ws.send_json({
            "action": "run",
            "file_ids": [1, 2],
            "batch_size": 1,
            "apply_tags_every_n": 1,
            "service_key": "my tags",
            "tag_all": True,
            "session_auto_tune": True,
            "performance_tuning": True,
            "tuning_control_mode": "auto_lucky",
        })
        while True:
            m = _ws_recv_skip_plan(ws)
            if m.get("type") == "complete":
                tr = m.get("tuning_report") or {}
                assert tr.get("session_auto_tune") is True
                assert tr.get("tuning_control_mode") == "auto_lucky"
                assert "autotune" in tr
                assert tr["autotune"].get("best_batch_size") is not None
                break


def test_ws_performance_tuning_omitted_without_tag_all(ws_client):
    client, _ = ws_client
    with client.websocket_connect("/api/tagger/ws/progress") as ws:
        ws.send_json({
            "action": "run",
            "file_ids": [1],
            "batch_size": 1,
            "apply_tags_every_n": 0,
            "tag_all": False,
            "performance_tuning": True,
        })
        p = _ws_recv_skip_plan(ws)
        assert p["type"] == "progress"
        assert "performance_tuning" not in p
        _ws_recv_skip_plan(ws)


def test_ws_logs_model_prepare_and_session_metrics(ws_client, caplog):
    """INFO lines for model prepare wall time and end-of-session optimization totals."""
    caplog.set_level(logging.INFO, logger="backend.routes.tagger_ws")
    client, _ = ws_client
    with client.websocket_connect("/api/tagger/ws/progress") as ws:
        ws.send_json({
            "action": "run",
            "file_ids": [501],
            "batch_size": 1,
            "apply_tags_every_n": 0,
        })
        while True:
            msg = _ws_recv_skip_plan(ws)
            if msg["type"] in ("complete", "stopped", "error"):
                break
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "tagging_ws session_config apply_tags_http_batch=" in joined
    assert "apply_tags_every_n_effective=" in joined
    assert "tagging_ws metrics model_prepare_wall_s=" in joined
    assert "tagging_ws session_metrics onnx_skipped_same_marker=" in joined
    assert "onnx_skipped_higher_tier_marker=" in joined
    assert "hydrus_duplicate_tag_strings_skipped_session=" in joined

"""Face DB cleanup, video exclusion, and re-run skip behaviour."""

from unittest.mock import MagicMock

import numpy as np
import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]

from backend.face.service import FaceTaggingService, _analyze_detect_queue, _is_video_mime


@pytest.mark.asyncio
async def test_clean_orphans_uses_hash_metadata(test_config, tmp_path, monkeypatch, caplog):
    db_path = tmp_path / "faces.db"
    cfg = test_config.model_copy(update={"face_embeddings_db_path": str(db_path)})
    FaceTaggingService._instance = None
    svc = FaceTaggingService.get_instance(cfg)
    svc.db.store_face("alivehash", 0, (0, 0, 1, 1), np.zeros(512, dtype=np.float32))
    svc.db.store_face("gonehash", 0, (0, 0, 1, 1), np.zeros(512, dtype=np.float32))

    calls: list[list[str]] = []

    class DummyClient:
        async def get_file_metadata_by_hashes(self, hashes):
            calls.append(list(hashes))
            return [{"hash": "alivehash", "is_deleted": False}]

    caplog.set_level("INFO", logger="backend.face.service")
    out = await svc.clean_orphans(DummyClient())
    assert calls == [["alivehash", "gonehash"]]
    assert out["removed"] == 1
    assert out["checked"] == 2
    assert out["alive"] == 1
    assert svc.db.file_has_embeddings("alivehash")
    assert not svc.db.file_has_embeddings("gonehash")
    assert "face clean orphans done" in caplog.text


def test_analyze_detect_queue_counts_videos_and_markers():
    meta = {
        1: {"hash": "a", "mime": "image/jpeg", "tags": {"sk": {"storage_tags": {"0": ["ai face detected"]}}}},
        2: {"hash": "b", "mime": "video/mp4"},
        3: {"hash": "c", "mime": "image/png"},
    }
    stats = _analyze_detect_queue(
        [1, 2, 3],
        meta,
        service_key="sk",
        skip_if_detected=True,
        replace_existing=False,
        marker_detected="ai face detected",
        marker_not_visible="face not visible",
    )
    assert stats["videos"] == 1
    assert stats["marker_skip"] == 1
    assert stats["to_process"] == 1
    assert stats["images"] == 2


def test_is_video_mime():
    assert _is_video_mime("video/x-ms-wmv")
    assert not _is_video_mime("image/jpeg")
    assert not _is_video_mime("")


@pytest.mark.asyncio
async def test_detect_batch_skips_video_without_model_load(test_config, monkeypatch, caplog):
    FaceTaggingService._instance = None
    cfg = test_config.model_copy(update={"face_skip_if_detected": False})
    svc = FaceTaggingService.get_instance(cfg)
    load_calls: list[int] = []

    async def fake_meta(client, file_ids, *, chunk_sz, cancel_event=None, progress_cb=None):
        return {
            1: {"hash": "vid1", "mime": "video/mp4"},
            2: {"hash": "vid2", "mime": "video/x-ms-wmv"},
        }

    async def fake_ensure(**_kwargs):
        load_calls.append(1)

    monkeypatch.setattr("backend.face.service.load_metadata_by_file_id", fake_meta)
    monkeypatch.setattr(svc, "ensure_model_loaded", fake_ensure)

    caplog.set_level("INFO", logger="backend.face.service")
    rows = await svc.detect_batch(MagicMock(), file_ids=[1, 2], service_key="sk")
    assert load_calls == []
    assert all(r.get("skip_reason") == "video_excluded" for r in rows)
    assert "face detect_batch skip model load" in caplog.text


@pytest.mark.asyncio
async def test_detect_batch_second_run_skips_marker(test_config, monkeypatch):
    FaceTaggingService._instance = None
    cfg = test_config.model_copy(
        update={
            "face_skip_if_detected": True,
            "face_marker_detected": "ai face detected",
        }
    )
    svc = FaceTaggingService.get_instance(cfg)
    _loaded_engine = lambda: setattr(svc.engine, "_app", object())
    _loaded_engine()
    svc._warmup_done = True

    async def fake_meta(client, file_ids, *, chunk_sz, cancel_event=None, progress_cb=None):
        return {
            5: {
                "hash": "tagged",
                "mime": "image/jpeg",
                "tags": {
                    "sk": {"storage_tags": {"0": ["ai face detected"]}},
                },
            },
        }

    async def fake_ensure(**_kwargs):
        return None

    detect_calls: list[int] = []

    async def fake_detect_file(**_kwargs):
        detect_calls.append(1)
        return {"file_id": 5, "hash": "tagged", "face_count": 1, "tags": ["ai face detected"]}

    monkeypatch.setattr("backend.face.service.load_metadata_by_file_id", fake_meta)
    monkeypatch.setattr(svc, "ensure_model_loaded", fake_ensure)
    monkeypatch.setattr(svc, "detect_file", fake_detect_file)

    rows = await svc.detect_batch(MagicMock(), file_ids=[5], service_key="sk")
    assert detect_calls == []
    assert rows[0]["skip_reason"] == "marker_present"


@pytest.mark.asyncio
async def test_reset_assignments_keeps_embeddings(test_config, tmp_path):
    db_path = tmp_path / "faces.db"
    cfg = test_config.model_copy(update={"face_embeddings_db_path": str(db_path)})
    FaceTaggingService._instance = None
    svc = FaceTaggingService.get_instance(cfg)
    svc.db.store_face("abc", 0, (0, 0, 1, 1), np.zeros(512, dtype=np.float32))
    pid = svc.db.create_new_person()
    svc.db.assign_face_to_person(1, pid)
    await svc.reset_assignments()
    assert svc.db.stats()["persons"] == 0
    assert svc.db.stats()["faces"] == 1
    assert svc.db.stats()["unassigned_faces"] == 1

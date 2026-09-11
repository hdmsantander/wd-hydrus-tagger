"""Face embedding database reuse and incremental detect/recognize."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import numpy as np
import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]

from backend.face.clustering import cluster_faces
from backend.face.embeddings_db import FaceEmbeddingsDB
from backend.face.service import FaceTaggingService, _analyze_detect_queue


def test_analyze_detect_queue_counts_marker_skip_for_person_tag():
    meta = {
        1: {
            "hash": "aa" * 32,
            "mime": "image/jpeg",
            "tags": {"sk": {"storage_tags": {"0": ["person:p12"]}}},
        },
    }
    stats = _analyze_detect_queue(
        [1],
        meta,
        service_key="sk",
        skip_if_detected=True,
        skip_if_in_db=False,
        db_hashes=set(),
        replace_existing=False,
        marker_detected="ai face detected",
        marker_not_visible="face not visible",
        person_prefix="person:",
    )
    assert stats["marker_skip"] == 1
    assert stats["to_process"] == 0


def test_analyze_detect_queue_large_rerun_mostly_skipped():
    """Simulate a 2000-file re-detect where markers + DB cover almost everything."""
    file_ids = list(range(2000))
    meta_by_id = {}
    db_hashes: set[str] = set()
    for fid in file_ids:
        h = f"{fid:064x}"
        meta_by_id[fid] = {"hash": h, "mime": "image/jpeg"}
        if fid < 1500:
            meta_by_id[fid]["tags"] = {
                "sk": {"storage_tags": {"0": ["ai face detected"]}},
            }
        elif fid < 1900:
            db_hashes.add(h)
    stats = _analyze_detect_queue(
        file_ids,
        meta_by_id,
        service_key="sk",
        skip_if_detected=True,
        skip_if_in_db=True,
        db_hashes=db_hashes,
        replace_existing=False,
        marker_detected="ai face detected",
        marker_not_visible="face not visible",
        person_prefix="person:",
    )
    assert stats["total"] == 2000
    assert stats["marker_skip"] == 1500
    assert stats["db_skip"] == 400
    assert stats["to_process"] == 100


def test_analyze_detect_queue_counts_db_skip():
    meta = {1: {"hash": "aa" * 32, "mime": "image/jpeg"}}
    stats = _analyze_detect_queue(
        [1],
        meta,
        service_key="sk",
        skip_if_detected=False,
        skip_if_in_db=True,
        db_hashes={"aa" * 32},
        replace_existing=False,
        marker_detected="ai face detected",
        marker_not_visible="face not visible",
    )
    assert stats["db_skip"] == 1
    assert stats["to_process"] == 0


@pytest.mark.asyncio
async def test_detect_second_run_reuses_db_without_wiping(test_config, tmp_path, monkeypatch):
    cfg = test_config.model_copy(
        update={
            "face_embeddings_db_path": str(tmp_path / "faces.db"),
            "face_skip_if_detected": False,
            "face_skip_if_in_db": True,
        }
    )
    FaceTaggingService._instance = None
    svc = FaceTaggingService.get_instance(cfg)
    fhash = "bb" * 32
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    svc.db.store_face(fhash, 0, (0, 0, 10, 10), emb)

    class DummyClient:
        async def get_file(self, file_id):
            raise AssertionError("should skip inference when hash in DB")

        async def get_thumbnail(self, file_id):
            raise AssertionError("should skip inference when hash in DB")

        async def add_tags(self, *a, **k):
            pass

    row = await svc.detect_file(
        DummyClient(),
        file_id=1,
        file_hash=fhash,
        meta={"hash": fhash, "mime": "image/jpeg"},
        service_key="sk",
    )
    assert row["skipped"] is True
    assert row["skip_reason"] == "db_cached"
    assert svc.db.stats()["faces"] == 1


@pytest.mark.asyncio
async def test_detect_appends_new_hash_to_existing_db(test_config, tmp_path, monkeypatch):
    cfg = test_config.model_copy(
        update={
            "face_embeddings_db_path": str(tmp_path / "faces.db"),
            "face_skip_if_detected": False,
            "face_skip_if_in_db": True,
        }
    )
    FaceTaggingService._instance = None
    svc = FaceTaggingService.get_instance(cfg)
    svc.db.store_face("aa" * 32, 0, (0, 0, 1, 1), np.zeros(3, dtype=np.float32))

    detect_calls: list[str] = []

    async def fake_detect_file(self, client, **kwargs):
        detect_calls.append(kwargs["file_hash"])
        self.db.store_face(kwargs["file_hash"], 0, (0, 0, 1, 1), np.ones(3, dtype=np.float32))
        return {"file_id": kwargs["file_id"], "hash": kwargs["file_hash"], "face_count": 1, "skipped": False, "tags": []}

    monkeypatch.setattr(FaceTaggingService, "detect_file", fake_detect_file)
    async def fake_ensure_model_loaded(**kwargs):
        return None

    monkeypatch.setattr(svc, "ensure_model_loaded", fake_ensure_model_loaded)

    meta_by_id = {
        1: {"hash": "aa" * 32, "mime": "image/jpeg"},
        2: {"hash": "cc" * 32, "mime": "image/jpeg"},
    }

    async def fake_load_metadata(client, file_ids, **kwargs):
        return {fid: meta_by_id[fid] for fid in file_ids}

    monkeypatch.setattr("backend.face.service.load_metadata_by_file_id", fake_load_metadata)

    class DummyClient:
        async def add_tags(self, *a, **k):
            pass

    results = await svc.detect_batch(
        DummyClient(),
        file_ids=[1, 2],
        service_key="sk",
    )
    assert svc.db.stats()["faces"] == 2
    assert detect_calls == ["cc" * 32]
    assert sum(1 for r in results if r.get("skip_reason") == "db_cached") == 1


@pytest.mark.asyncio
async def test_recognize_uses_entire_database(test_config, tmp_path, monkeypatch):
    cfg = test_config.model_copy(update={"face_embeddings_db_path": str(tmp_path / "faces.db")})
    FaceTaggingService._instance = None
    svc = FaceTaggingService.get_instance(cfg)
    for i, prefix in enumerate(("aa", "bb", "cc")):
        svc.db.store_face(prefix * 32, 0, (0, 0, 1, 1), np.array([1.0, 0.01 * i, 0.0], dtype=np.float32))

    cluster_sizes: list[int] = []

    def fake_cluster(faces, **kwargs):
        cluster_sizes.append(len(faces))
        return 0

    monkeypatch.setattr("backend.face.service.cluster_faces", fake_cluster)
    monkeypatch.setattr(svc.db, "get_file_person_tags", lambda prefix: {})

    async def fake_reset():
        return None

    monkeypatch.setattr(svc, "reset_assignments", fake_reset)

    class DummyClient:
        async def add_tags(self, *a, **k):
            pass

    out = await svc.recognize(DummyClient(), service_key="sk", staged=False, min_faces=1)
    assert out["faces_in_db"] == 3
    assert out["database_reused"] is True
    assert all(n == 3 for n in cluster_sizes)


@pytest.mark.asyncio
async def test_incremental_recognize_only_applies_touched_files(test_config, tmp_path, monkeypatch):
    cfg = test_config.model_copy(update={"face_embeddings_db_path": str(tmp_path / "faces.db")})
    FaceTaggingService._instance = None
    svc = FaceTaggingService.get_instance(cfg)
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    svc.db.store_face("aa" * 32, 0, (0, 0, 1, 1), emb)
    pid = svc.db.create_new_person()
    svc.db.assign_face_to_person(svc.db.load_all_faces()[0]["id"], pid)
    svc.db.store_face("bb" * 32, 0, (0, 0, 1, 1), emb + np.array([0.02, 0, 0], dtype=np.float32))

    def fake_cluster(faces, **kwargs):
        for f in faces:
            if not f.get("person_id"):
                svc.db.assign_face_to_person(int(f["id"]), pid)
        return 1

    monkeypatch.setattr("backend.face.service.cluster_faces", fake_cluster)
    apply_calls: list[str] = []

    class DummyClient:
        async def apply_tag_actions(self, hash_, service_key, *, add_tags, remove_tags=None):
            apply_calls.append(hash_)

        async def add_tags(self, *a, **k):
            apply_calls.append(a[0])

    out = await svc.recognize(
        DummyClient(),
        service_key="sk",
        staged=False,
        min_faces=1,
        refine_incremental=True,
        replace_person_tags=True,
        recluster_all=False,
    )
    assert out["files_tagged"] == 1
    assert out["files_touched"] == 1
    assert apply_calls == ["bb" * 32]


def test_file_hashes_in_db_set(tmp_path):
    db = FaceEmbeddingsDB(tmp_path / "faces.db")
    db.init()
    db.store_face("dd" * 32, 0, (0, 0, 1, 1), np.zeros(3, dtype=np.float32))
    assert "dd" * 32 in db.file_hashes_in_db()


@pytest.mark.asyncio
async def test_staged_recognize_applies_hydrus_after_each_stage(test_config, tmp_path, monkeypatch):
    cfg = test_config.model_copy(update={"face_embeddings_db_path": str(tmp_path / "faces.db")})
    FaceTaggingService._instance = None
    svc = FaceTaggingService.get_instance(cfg)
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    svc.db.store_face("aa" * 32, 0, (0, 0, 1, 1), emb)

    stage_calls: list[int] = []

    def fake_cluster(faces, **kwargs):
        stage_calls.append(kwargs.get("min_faces"))
        pid = svc.db.create_new_person()
        svc.db.assign_face_to_person(int(faces[0]["id"]), pid)
        return 1

    monkeypatch.setattr("backend.face.service.cluster_faces", fake_cluster)
    monkeypatch.setattr("backend.face.service.staged_min_faces_list", lambda _stages: [3, 2])

    apply_calls: list[str] = []

    class DummyClient:
        async def apply_tag_actions(self, hash_, service_key, *, add_tags, remove_tags=None):
            apply_calls.append(hash_)

        async def add_tags(self, *a, **k):
            apply_calls.append(a[0])

        async def get_file_metadata_by_hashes(self, hashes):
            return [{"hash": h} for h in hashes]

    async def fake_reset():
        return None

    monkeypatch.setattr(svc, "reset_assignments", fake_reset)

    out = await svc.recognize(DummyClient(), service_key="sk", staged=True)
    assert len(stage_calls) == 2
    assert apply_calls == ["aa" * 32, "aa" * 32]
    assert out["files_tagged"] == 2


@pytest.mark.asyncio
async def test_detect_batch_cancel_at_boundary_preserves_results(test_config, monkeypatch):
    FaceTaggingService._instance = None
    svc = FaceTaggingService.get_instance(test_config)
    cancel_event = asyncio.Event()

    async def fake_meta(client, file_ids, *, chunk_sz, cancel_event=None, progress_cb=None):
        return {fid: {"hash": f"{fid:02d}" * 16, "mime": "image/jpeg"} for fid in file_ids}

    async def fake_ensure(**_kwargs):
        return None

    async def fake_detect_file(self, client, *, file_id, file_hash, **kwargs):
        return {
            "file_id": file_id,
            "hash": file_hash,
            "face_count": 1,
            "skipped": False,
            "tags": ["ai face detected"],
        }

    monkeypatch.setattr("backend.face.service.load_metadata_by_file_id", fake_meta)
    monkeypatch.setattr(svc, "ensure_model_loaded", fake_ensure)
    monkeypatch.setattr(FaceTaggingService, "detect_file", fake_detect_file)

    cancel_event.set()
    results = await svc.detect_batch(
        MagicMock(),
        file_ids=[1, 2, 3],
        service_key="sk",
        cancel_event=cancel_event,
    )
    assert results == []


def test_cluster_faces_seed_existing_preserves_person_links(tmp_path):
    db = FaceEmbeddingsDB(tmp_path / "faces.db")
    db.init()
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    db.store_face("ff" * 32, 0, (0, 0, 1, 1), emb)
    face_id = db.load_all_faces()[0]["id"]
    pid = db.create_new_person()
    db.assign_face_to_person(face_id, pid)
    db.store_face("ff" * 32, 1, (0, 0, 2, 2), emb + np.array([0.02, 0, 0], dtype=np.float32))
    faces = db.load_all_faces()
    assigned = cluster_faces(
        faces,
        max_distance=0.5,
        min_faces=1,
        create_person=db.create_new_person,
        assign_person=db.assign_face_to_person,
        seed_existing=True,
    )
    assert assigned == 1
    by_id = {f["id"]: f for f in db.load_all_faces()}
    assert by_id[face_id]["person_id"] == pid

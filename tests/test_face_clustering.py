"""Tests for face clustering and embeddings DB."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]

from backend.face.clustering import cluster_faces, embedding_distance, staged_min_faces_list
from backend.face.embeddings_db import FaceEmbeddingsDB


def test_embedding_distance_cosine():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0], dtype=np.float32)
    assert embedding_distance(a, b, "cosine_similarity") == pytest.approx(0.0)


def test_staged_min_faces_list_parses_string():
    assert staged_min_faces_list("20,5,3,1") == [20, 5, 3, 1]


def test_face_db_store_and_stats(tmp_path):
    db = FaceEmbeddingsDB(tmp_path / "faces.db")
    db.init()
    emb = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    db.store_face("aa" * 32, 0, (1, 2, 3, 4), emb)
    st = db.stats()
    assert st["faces"] == 1
    assert st["files_with_faces"] == 1


def test_cluster_faces_assigns_person(tmp_path):
    db = FaceEmbeddingsDB(tmp_path / "faces.db")
    db.init()
    base = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    for i in range(4):
        noise = base + np.array([0.01 * i, 0.0, 0.0], dtype=np.float32)
        noise = noise / np.linalg.norm(noise)
        db.store_face("bb" * 32, i, (0, 0, 10, 10), noise)

    faces = db.load_all_faces()
    assigned = cluster_faces(
        faces,
        max_distance=0.15,
        min_faces=2,
        allow_new=True,
        distance_method="cosine_similarity",
        create_person=db.create_new_person,
        assign_person=db.assign_face_to_person,
    )
    assert assigned >= 2
    tags = db.get_file_person_tags()
    assert "bb" * 32 in tags
    assert any(t.startswith("person:p") for t in tags["bb" * 32])

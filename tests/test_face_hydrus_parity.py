"""Face tag format parity with hydrus-ai-taggers (direct Hydrus add_tags apply)."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]

from backend.face.embeddings_db import FaceEmbeddingsDB


def _source_get_file_tags(db_rows: dict[str, set[str]]) -> dict[str, set[str]]:
    """Mirror hydrus-ai-taggers get_file_tags: person:{pid} per file hash."""
    return {fhash: {f"person:{pid}" for pid in pids} for fhash, pids in db_rows.items()}


def test_person_tag_format_matches_source_project(tmp_path):
    """person:p# tags use the same prefix+id pattern as hydrus-ai-taggers."""
    db = FaceEmbeddingsDB(tmp_path / "faces.db")
    db.init()
    fhash = "cc" * 32
    pid = db.create_new_person()
    assert pid.startswith("p")
    db.store_face(fhash, 0, (0, 0, 10, 10), __import__("numpy").zeros(3, dtype="float32"))
    face_id = db.load_all_faces()[0]["id"]
    db.assign_face_to_person(face_id, pid)

    ours = db.get_file_person_tags("person:")
    source_style = _source_get_file_tags({fhash: {pid}})
    assert ours == source_style
    assert ours[fhash] == {f"person:{pid}"}


def test_recognized_marker_tag_is_configurable_not_pending():
    """Recognize applies marker + person tags in one add_tags call (no review queue)."""
    from backend.config import get_config

    cfg = get_config()
    marker = cfg.face_marker_recognized
    assert marker  # default: face ai generated tags
    assert "pending" not in marker.lower()

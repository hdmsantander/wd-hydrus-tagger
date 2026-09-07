"""Face pipeline step labels and critical flow coverage."""

import logging
from unittest.mock import MagicMock

import numpy as np
import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]

from backend.face.flow_steps import (
    DETECT_PIPELINE,
    RECOGNIZE_PIPELINE,
    face_detect_progress,
    face_recognize_apply_label,
    face_recognize_step_label,
)
from backend.face.service import FaceTaggingService


def test_face_detect_progress_step_fields():
    msg = face_detect_progress("queue", detail="Queue: 2 image(s) to scan", total=10)
    assert msg["pipeline"] == DETECT_PIPELINE
    assert msg["step"] == "queue"
    assert msg["step_num"] == 2
    assert msg["step_total"] == 5
    assert msg["step_label"] == "Detect 2/5 — Analyze queue"
    assert msg["phase"] == "metadata"
    assert "Queue:" in msg["detail"]


def test_face_detect_progress_scan_phase():
    msg = face_detect_progress("scan", detail="File 1/3", processed=1, total=3)
    assert msg["phase"] == "detect"
    assert msg["step_label"] == "Detect 5/5 — Scan images"


def test_face_recognize_step_labels():
    assert face_recognize_step_label(1, 4, 20) == "Recognize 1/4 — Cluster (min 20 faces)"
    assert "Apply" in face_recognize_apply_label()


@pytest.mark.asyncio
async def test_detect_batch_progress_includes_step_labels(test_config, monkeypatch, caplog):
    FaceTaggingService._instance = None
    cfg = test_config.model_copy(update={"face_skip_if_detected": False})
    svc = FaceTaggingService.get_instance(cfg)
    messages: list[dict] = []

    async def cb(msg):
        messages.append(msg)

    async def fake_meta(client, file_ids, *, chunk_sz, cancel_event=None, progress_cb=None):
        if progress_cb:
            await progress_cb(len(file_ids), len(file_ids))
        return {10: {"hash": "aaa", "mime": "image/png"}}

    async def fake_ensure(progress_cb=None, **kwargs):
        if progress_cb:
            await progress_cb(
                face_detect_progress("model_load", detail="loaded", total=1),
            )
            await progress_cb(
                face_detect_progress("warmup", detail="warm", total=1, phase="model"),
            )

    buf_image = __import__("io").BytesIO()
    from PIL import Image

    Image.new("RGB", (8, 8)).save(buf_image, format="PNG")
    png = buf_image.getvalue()

    class DummyClient:
        async def get_file(self, file_id=None, **_kwargs):
            return png, "image/png"

        async def add_tags(self, hash_, service_key, tags):
            return None

    monkeypatch.setattr("backend.face.service.load_metadata_by_file_id", fake_meta)
    monkeypatch.setattr(svc, "ensure_model_loaded", fake_ensure)
    monkeypatch.setattr(svc.engine, "detect_faces", lambda image, conf=None: [])

    caplog.set_level(logging.INFO, logger="backend.face.service")
    await svc.detect_batch(DummyClient(), file_ids=[10], service_key="sk", progress_cb=cb)

    steps = {m.get("step") for m in messages if m.get("pipeline") == DETECT_PIPELINE}
    assert "metadata" in steps
    assert "queue" in steps
    assert "scan" in steps
    assert all(m.get("step_label") for m in messages if m.get("pipeline") == DETECT_PIPELINE)
    assert "face detect_batch queue" in caplog.text


@pytest.mark.asyncio
async def test_recognize_returns_stage_breakdown(test_config, tmp_path, monkeypatch, caplog):
    cfg = test_config.model_copy(
        update={
            "face_embeddings_db_path": str(tmp_path / "faces.db"),
            "face_recognition_stages": [3, 1],
        }
    )
    FaceTaggingService._instance = None
    svc = FaceTaggingService.get_instance(cfg)
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    for i in range(3):
        noise = emb + np.array([0.01 * i, 0.0, 0.0], dtype=np.float32)
        noise = noise / np.linalg.norm(noise)
        svc.db.store_face("hash" + str(i), 0, (0, 0, 1, 1), noise)

    calls: list = []

    class DummyClient:
        async def add_tags(self, hash_, service_key, tags):
            calls.append((hash_, list(tags)))

    caplog.set_level(logging.INFO, logger="backend.face.service")
    out = await svc.recognize(DummyClient(), service_key="sk", staged=True)
    assert out["stages"]
    assert len(out["stages"]) == 2
    assert out["stages"][0]["step_label"].startswith("Recognize 1/2")
    assert "face recognize start" in caplog.text
    assert "face recognize done" in caplog.text


@pytest.mark.asyncio
async def test_recognize_empty_db_returns_zero_assignments(test_config, tmp_path):
    cfg = test_config.model_copy(update={"face_embeddings_db_path": str(tmp_path / "empty.db")})
    FaceTaggingService._instance = None
    svc = FaceTaggingService.get_instance(cfg)

    class DummyClient:
        async def add_tags(self, hash_, service_key, tags):
            pass

    out = await svc.recognize(DummyClient(), service_key="sk", staged=True)
    assert out["assigned_faces"] == 0
    assert out["files_tagged"] == 0
    assert len(out["stages"]) == len(test_config.face_recognition_stages)
    assert all(s["assigned"] == 0 for s in out["stages"])
    assert all(s.get("step_label", "").startswith("Recognize") for s in out["stages"])

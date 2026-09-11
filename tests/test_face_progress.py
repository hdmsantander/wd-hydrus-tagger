"""Face detect progress: WebSocket payloads, CLI logs, warmup, and inference timeout."""

import asyncio
import logging
import time
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from PIL import Image

pytestmark = [pytest.mark.full, pytest.mark.core]

import backend.config as config_module
from backend.face.load_control import reset_face_load_control_for_tests
from backend.face.flow_steps import face_detect_progress
from backend.face.service import FaceTaggingService


@pytest.fixture(autouse=True)
def _reset_face_load_control():
    reset_face_load_control_for_tests()
    yield
    reset_face_load_control_for_tests()


def _loaded_engine(svc):
    svc.engine._app = object()
    svc.engine._providers = ["CPUExecutionProvider"]
    svc.engine._ctx_id = -1
    svc.engine._load_stage = "ready"


@pytest.mark.asyncio
async def test_warmup_emits_progress_and_logs(test_config, monkeypatch, caplog):
    FaceTaggingService._instance = None
    svc = FaceTaggingService.get_instance(test_config)
    _loaded_engine(svc)
    messages: list[dict] = []

    async def cb(msg):
        messages.append(msg)

    monkeypatch.setattr(svc.engine, "detect_faces", lambda image, conf=None: [])
    caplog.set_level(logging.INFO, logger="backend.face.service")
    await svc._warmup_inference(progress_cb=cb, total=4)

    assert svc._warmup_done is True
    assert any(m.get("phase") == "model" and "Warming up" in m.get("detail", "") for m in messages)
    assert "face model warmup start" in caplog.text
    assert "face model warmup ok" in caplog.text


@pytest.mark.asyncio
async def test_warmup_gpu_timeout_reloads_cpu(test_config, monkeypatch):
    FaceTaggingService._instance = None
    cfg = test_config.model_copy(
        update={"use_gpu": True, "gpu_backend": "rocm", "face_inference_timeout_seconds": 0.05}
    )
    svc = FaceTaggingService.get_instance(cfg)
    _loaded_engine(svc)
    svc.engine._providers = ["MIGraphXExecutionProvider"]
    svc.engine._ctx_id = 0
    calls = {"detect": 0, "reload": 0}

    def slow_detect(image, conf=None):
        calls["detect"] += 1
        if calls["detect"] == 1:
            time.sleep(2.5)
        return []

    monkeypatch.setattr(svc.engine, "detect_faces", slow_detect)

    async def fake_reload(**_kwargs):
        calls["reload"] += 1
        _loaded_engine(svc)
        svc.engine._loaded_with_cpu_fallback = True

    monkeypatch.setattr(svc, "_reload_engine_on_cpu", fake_reload)

    await svc._warmup_inference(total=1)
    assert calls["detect"] >= 2
    assert calls["reload"] == 1
    assert svc._warmup_done is True


@pytest.mark.asyncio
async def test_run_face_worker_reports_processed_index(test_config, monkeypatch):
    FaceTaggingService._instance = None
    svc = FaceTaggingService.get_instance(test_config)
    _loaded_engine(svc)
    messages: list[dict] = []

    async def cb(msg):
        messages.append(msg)

    monkeypatch.setattr(svc.engine, "detect_faces", lambda image, conf=None: time.sleep(0.05) or [])

    await svc._run_face_worker(
        lambda: svc.engine.detect_faces(Image.new("RGB", (8, 8)), conf=0.5),
        timeout_s=None,
        progress_cb=cb,
        progress_detail="File 3/10 · running face detection",
        total=10,
        processed=2,
        phase="detect",
        log_label="test",
    )

    assert messages
    assert messages[0]["processed"] == 2
    assert messages[0]["total"] == 10
    assert messages[0]["phase"] == "detect"
    assert "File 3/10" in messages[0]["detail"]


@pytest.mark.asyncio
async def test_detect_batch_progress_phases_and_cli_logs(test_config, monkeypatch, caplog):
    FaceTaggingService._instance = None
    cfg = test_config.model_copy(update={"face_skip_if_detected": False})
    svc = FaceTaggingService.get_instance(cfg)
    _loaded_engine(svc)
    svc._warmup_done = True
    messages: list[dict] = []

    async def cb(msg):
        messages.append(msg)

    async def fake_meta(client, file_ids, *, chunk_sz, cancel_event=None, progress_cb=None):
        if progress_cb:
            await progress_cb(len(file_ids), len(file_ids))
        return {
            10: {"hash": "aaa", "mime": "image/png"},
            11: {"hash": "bbb", "mime": "image/png"},
        }

    async def fake_ensure(progress_cb=None, **kwargs):
        if progress_cb:
            await progress_cb(
                face_detect_progress(
                    "model_load",
                    detail="Loading InsightFace · ready (1s)",
                    total=2,
                )
            )

    buf = BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="PNG")
    png = buf.getvalue()

    class DummyClient:
        async def get_file(self, file_id=None, **_kwargs):
            return png, "image/png"

        async def add_tags(self, hash_, service_key, tags):
            return None

    monkeypatch.setattr("backend.face.service.load_metadata_by_file_id", fake_meta)
    monkeypatch.setattr(svc, "ensure_model_loaded", fake_ensure)
    monkeypatch.setattr(svc.engine, "detect_faces", lambda image, conf=None: [])

    caplog.set_level(logging.INFO, logger="backend.face.service")
    rows = await svc.detect_batch(
        DummyClient(),
        file_ids=[10, 11],
        service_key="sk",
        progress_cb=cb,
    )

    assert len(rows) == 2
    phases = [m.get("phase") for m in messages]
    assert phases.count("metadata") >= 1
    assert phases.count("model") >= 1
    assert phases.count("detect") >= 1
    assert all(m.get("step_label") for m in messages if m.get("type") == "progress")
    assert any("Detect 5/5" in m.get("step_label", "") for m in messages)
    assert any("Processing file 1/2" in m.get("detail", "") for m in messages)
    assert any("Processing file 2/2" in m.get("detail", "") for m in messages)
    assert any(m.get("processed") == 2 and m.get("phase") == "detect" for m in messages)

    text = caplog.text
    assert "face detect_batch start" in text
    assert "face detect_batch queue" in text
    assert "face detect progress" in text
    assert "face detect progress" in text
    assert "face detect_batch done" in text


@pytest.mark.asyncio
async def test_detect_file_inference_progress_uses_file_index(test_config, monkeypatch):
    FaceTaggingService._instance = None
    cfg = test_config.model_copy(
        update={"face_skip_if_detected": False, "face_skip_if_in_db": False},
    )
    svc = FaceTaggingService.get_instance(cfg)
    _loaded_engine(svc)
    messages: list[dict] = []

    async def cb(msg):
        messages.append(msg)

    buf = BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="PNG")
    png = buf.getvalue()

    class DummyClient:
        async def get_file(self, file_id=None, **_kwargs):
            return png, "image/png"

        async def add_tags(self, hash_, service_key, tags):
            return None

    def slow_detect(image, conf=None):
        time.sleep(2.05)
        return [{"bbox": [0, 0, 2, 2], "embedding": np.zeros(512, dtype=np.float32)}]

    monkeypatch.setattr(svc.engine, "detect_faces", slow_detect)

    await svc.detect_file(
        DummyClient(),
        file_id=42,
        file_hash="deadbeef",
        meta=None,
        service_key="sk",
        progress_cb=cb,
        file_index=3,
        total=10,
    )

    infer_msgs = [m for m in messages if "running face detection" in m.get("detail", "")]
    assert infer_msgs
    assert infer_msgs[0]["processed"] == 2
    assert infer_msgs[0]["total"] == 10


@pytest.mark.asyncio
async def test_detect_batch_cancel_during_file_returns_partial(test_config, monkeypatch):
    FaceTaggingService._instance = None
    svc = FaceTaggingService.get_instance(test_config)
    _loaded_engine(svc)
    svc._warmup_done = True
    cancel_event = asyncio.Event()

    async def fake_meta(client, file_ids, *, chunk_sz, cancel_event=None, progress_cb=None):
        return {
            1: {"hash": "aa" * 32, "mime": "image/png"},
            2: {"hash": "bb" * 32, "mime": "image/png"},
        }

    async def fake_ensure(**_kwargs):
        return None

    call_count = 0

    async def slow_detect_file(self, client, *, file_id, file_hash, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            cancel_event.set()
            raise asyncio.CancelledError("cancelled in test")
        return {
            "file_id": file_id,
            "hash": file_hash,
            "face_count": 1,
            "skipped": False,
            "tags": ["ai face detected"],
        }

    monkeypatch.setattr("backend.face.service.load_metadata_by_file_id", fake_meta)
    monkeypatch.setattr(svc, "ensure_model_loaded", fake_ensure)
    monkeypatch.setattr(FaceTaggingService, "detect_file", slow_detect_file)

    results = await svc.detect_batch(
        MagicMock(),
        file_ids=[1, 2],
        service_key="sk",
        cancel_event=cancel_event,
    )
    assert len(results) == 1
    assert results[0]["hash"] == "aa" * 32
    assert call_count == 2

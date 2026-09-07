"""Face API route smoke tests (no InsightFace load)."""

import pytest
from fastapi.testclient import TestClient

pytestmark = [pytest.mark.full, pytest.mark.core]

import backend.config as config_module
from backend.face.service import FaceTaggingService


@pytest.fixture
def client(test_config):
    config_module._config = test_config
    FaceTaggingService._instance = None
    from backend.app import app

    return TestClient(app)


def test_face_providers_endpoint(client):
    r = client.get("/api/face/providers")
    assert r.status_code == 200
    data = r.json()
    assert data["success"] is True
    assert "CPUExecutionProvider" in data["providers"]


def test_face_status_endpoint(client):
    r = client.get("/api/face/status")
    assert r.status_code == 200
    data = r.json()
    assert data["success"] is True
    assert "faces" in data["status"]


def test_face_session_status_endpoint(client):
    r = client.get("/api/face/session/status")
    assert r.status_code == 200
    data = r.json()
    assert data["success"] is True
    assert data["active"] is False
    assert data["sessions"] == 0


def test_face_models_list_endpoint(client):
    r = client.get("/api/face/models")
    assert r.status_code == 200
    data = r.json()
    assert data["success"] is True
    assert data["models"]
    assert data["models"][0]["name"] == "buffalo_l"
    assert "downloaded" in data["models"][0]


def test_face_models_verify_endpoint(client):
    r = client.post("/api/face/models/verify")
    assert r.status_code == 200
    data = r.json()
    assert data["success"] is True
    assert data["results"][0]["name"] == "buffalo_l"


def test_face_reset_endpoint(client):
    r = client.post("/api/face/reset")
    assert r.status_code == 200
    assert r.json()["success"] is True


def test_face_clean_endpoint(client, monkeypatch):
    async def fake_clean(self, client):
        return {"removed": 2, "checked": 5, "alive": 3, "failed_batches": 0}

    monkeypatch.setattr(FaceTaggingService, "clean_orphans", fake_clean)
    r = client.post("/api/face/clean")
    assert r.status_code == 200
    data = r.json()
    assert data["success"] is True
    assert data["removed"] == 2
    assert data["checked"] == 5


@pytest.mark.asyncio
async def test_detect_batch_passes_chunk_sz(test_config, monkeypatch):
    FaceTaggingService._instance = None
    svc = FaceTaggingService.get_instance(test_config)
    captured = {}

    async def fake_meta(client, file_ids, *, chunk_sz, cancel_event=None, progress_cb=None):
        captured["chunk_sz"] = chunk_sz
        captured["n"] = len(file_ids)
        return {}

    async def fake_loaded(**_kwargs):
        return None

    monkeypatch.setattr("backend.face.service.load_metadata_by_file_id", fake_meta)
    monkeypatch.setattr(svc, "ensure_model_loaded", fake_loaded)

    class DummyClient:
        pass

    rows = await svc.detect_batch(DummyClient(), file_ids=[1, 2], service_key="sk")
    assert captured["chunk_sz"] == test_config.hydrus_metadata_chunk_size
    assert captured["n"] == 2
    assert all(r.get("error") == "missing_hash" for r in rows)


@pytest.mark.asyncio
async def test_detect_file_applies_marker_via_add_tags(test_config, tmp_path, monkeypatch):
    import numpy as np
    from io import BytesIO
    from PIL import Image

    cfg = test_config.model_copy(
        update={
            "face_embeddings_db_path": str(tmp_path / "faces.db"),
            "face_skip_if_detected": False,
        }
    )
    FaceTaggingService._instance = None
    svc = FaceTaggingService.get_instance(cfg)

    async def _loaded():
        return None

    monkeypatch.setattr(svc, "ensure_model_loaded", _loaded)
    monkeypatch.setattr(
        svc.engine,
        "detect_faces",
        lambda image, conf=0.5: [
            {"bbox": [0, 0, 2, 2], "embedding": np.zeros(512, dtype=np.float32)}
        ],
    )

    buf = BytesIO()
    Image.new("RGB", (8, 8), (12, 34, 56)).save(buf, format="PNG")
    png = buf.getvalue()
    calls: list = []

    class DummyClient:
        async def get_file(self, file_id=None, **_kwargs):
            return png, "image/png"

        async def add_tags(self, hash_, service_key, tags):
            calls.append({"hash": hash_, "service_key": service_key, "tags": list(tags)})

        async def apply_tag_actions(self, hash_, service_key, *, add_tags, remove_tags=None):
            raise AssertionError("detect_file must use add_tags(), not apply_tag_actions")

    row = await svc.detect_file(
        DummyClient(),
        file_id=9,
        file_hash="abc123",
        meta=None,
        service_key="sk",
    )
    assert row["face_count"] == 1
    assert "error" not in row
    assert calls == [{"hash": "abc123", "service_key": "sk", "tags": [cfg.face_marker_detected]}]


@pytest.mark.asyncio
async def test_recognize_applies_person_tags_via_add_tags(test_config, tmp_path, monkeypatch):
    cfg = test_config.model_copy(update={"face_embeddings_db_path": str(tmp_path / "faces.db")})
    FaceTaggingService._instance = None
    svc = FaceTaggingService.get_instance(cfg)
    monkeypatch.setattr(svc.db, "load_all_faces", lambda: [])
    monkeypatch.setattr(svc.db, "get_file_person_tags", lambda prefix: {"deadbeef": {"person:p1"}})
    calls: list = []

    class DummyClient:
        async def add_tags(self, hash_, service_key, tags):
            calls.append((hash_, service_key, list(tags)))

        async def apply_tag_actions(self, hash_, service_key, *, add_tags, remove_tags=None):
            raise AssertionError("recognize must use add_tags(), not apply_tag_actions")

    out = await svc.recognize(DummyClient(), service_key="sk", staged=False, min_faces=1)
    assert out["files_tagged"] == 1
    assert "stages" in out
    assert calls[0][0] == "deadbeef"
    assert "person:p1" in calls[0][2]
    assert cfg.face_marker_recognized in calls[0][2]


@pytest.mark.asyncio
async def test_ensure_model_loaded_warmup_runs(test_config, monkeypatch):
    FaceTaggingService._instance = None
    svc = FaceTaggingService.get_instance(test_config)
    warmup_calls: list[int] = []

    async def fake_warmup(**kwargs):
        warmup_calls.append(1)

    monkeypatch.setattr(svc, "_warmup_inference", fake_warmup)

    class ReadyEngine:
        loaded_with_cpu_fallback = False
        active_provider = "CPUExecutionProvider"
        providers = ["CPUExecutionProvider"]
        load_stage = "ready"
        _loaded = True

        @property
        def loaded(self):
            return self._loaded

    monkeypatch.setattr(svc, "engine", ReadyEngine())
    await svc.ensure_model_loaded(total=1)
    assert warmup_calls == [1]


@pytest.mark.asyncio
async def test_ensure_model_loaded_sends_progress_while_loading(test_config, monkeypatch):
    FaceTaggingService._instance = None
    svc = FaceTaggingService.get_instance(test_config)
    messages: list[dict] = []

    class SlowEngine:
        loaded_with_cpu_fallback = False
        active_provider = "CPUExecutionProvider"
        providers = ["CPUExecutionProvider"]
        load_stage = "preparing"
        _loaded = False

        @property
        def loaded(self):
            return self._loaded

        def run_load(self, *, force_cpu: bool, generation: int):
            import time
            time.sleep(0.05)
            self._loaded = True
            self.load_stage = "ready"

        def detect_faces(self, image, conf=None):
            return []

    monkeypatch.setattr(svc, "engine", SlowEngine())

    async def cb(msg):
        messages.append(msg)

    await svc.ensure_model_loaded(progress_cb=cb, total=3)
    assert any(m.get("phase") == "model" and m.get("detail") for m in messages)
    assert any(m.get("step") == "model_load" and m.get("step_label", "").startswith("Detect 3/5") for m in messages)


def test_face_engine_cpu_fallback_on_gpu_error(test_config, tmp_path, monkeypatch):
    from backend.face.engine import FaceEngine

    calls: list[bool] = []

    def fake_load_impl(self, *, force_cpu: bool = False, generation: int = 0):
        calls.append(force_cpu)
        if not force_cpu:
            raise RuntimeError("gpu boom")
        self._app = object()
        self._providers = ["CPUExecutionProvider"]
        self._ctx_id = -1
        self._load_stage = "ready"
        if force_cpu:
            self._loaded_with_cpu_fallback = True

    monkeypatch.setattr(FaceEngine, "_load_impl", fake_load_impl)
    eng = FaceEngine(
        use_gpu=True,
        gpu_backend="rocm",
        models_root=tmp_path / "face",
    )
    eng.load()
    assert calls == [False, True]
    assert eng.loaded_with_cpu_fallback
    assert eng.loaded


def test_face_service_rebuilds_engine_on_gpu_change(test_config):
    FaceTaggingService._instance = None
    cfg_cpu = test_config.model_copy(update={"use_gpu": False, "gpu_backend": "cpu"})
    svc = FaceTaggingService.get_instance(cfg_cpu)
    assert svc.engine.gpu_backend == "cpu"
    assert svc.engine.use_gpu is False

    cfg_gpu = test_config.model_copy(update={"use_gpu": True, "gpu_backend": "auto"})
    svc2 = FaceTaggingService.get_instance(cfg_gpu)
    assert svc2 is svc
    assert svc2.engine.gpu_backend == "auto"
    assert svc2.engine.use_gpu is True
    assert not svc2.engine.loaded

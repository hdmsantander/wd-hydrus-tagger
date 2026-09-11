"""Model unload and resource cleanup on coordinated shutdown / app lifespan exit."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]

from backend.face.load_control import (
    get_face_load_executor,
    reset_face_load_control_for_tests,
    shutdown_face_load_executor,
)
from backend.face.service import FaceTaggingService
from backend.hydrus.client import _client_pool, aclose_all_hydrus_clients
from backend.services.tagging_service import TaggingService
from backend.shutdown_coordination import (
    reset_coordinated_tagging_shutdown_for_tests,
    run_coordinated_tagging_shutdown,
)


def _prime_loaded_models(test_config) -> tuple[TaggingService, FaceTaggingService]:
    TaggingService._instance = None
    FaceTaggingService._instance = None
    ts = TaggingService.get_instance(test_config)
    fs = FaceTaggingService.get_instance(test_config)
    ts.engine.session = MagicMock(name="onnx_session")
    ts.engine._active_providers = ["CPUExecutionProvider"]
    ts.engine.model_name = test_config.default_model
    ts._loaded_model = test_config.default_model
    fs.engine._app = MagicMock(name="insightface_app")
    fs.engine._providers = ["CPUExecutionProvider"]
    fs.engine._ctx_id = -1
    fs.engine._load_stage = "ready"
    fs._warmup_done = True
    return ts, fs


def test_tagging_service_unload_clears_onnx_session(test_config):
    ts, _ = _prime_loaded_models(test_config)
    prev = TaggingService.unload_model_from_memory()
    assert prev == test_config.default_model
    assert ts.engine.session is None
    assert ts.engine._active_providers == []
    assert ts._loaded_model is None
    assert ts._loaded_ort_threads is None


def test_face_service_unload_clears_engine(test_config):
    _, fs = _prime_loaded_models(test_config)
    FaceTaggingService.unload_model_from_memory()
    assert not fs.engine.loaded
    assert fs.engine._providers == []
    assert fs._warmup_done is False


@pytest.mark.asyncio
async def test_coordinated_shutdown_releases_both_models(test_config):
    reset_coordinated_tagging_shutdown_for_tests()
    ts, fs = _prime_loaded_models(test_config)
    reset_face_load_control_for_tests()
    get_face_load_executor()

    metrics = await run_coordinated_tagging_shutdown(reason="test_release")

    assert metrics.get("completed") is True
    assert metrics.get("onnx_released") is True
    assert metrics.get("face_model_released") is True
    assert metrics.get("previous_loaded_model") == test_config.default_model
    assert ts.engine.session is None
    assert ts._loaded_model is None
    assert not fs.engine.loaded

    with pytest.raises(RuntimeError, match="shut down"):
        get_face_load_executor()


@pytest.mark.asyncio
async def test_coordinated_shutdown_idempotent_after_release(test_config):
    reset_coordinated_tagging_shutdown_for_tests()
    _prime_loaded_models(test_config)
    await run_coordinated_tagging_shutdown(reason="first")
    metrics = await run_coordinated_tagging_shutdown(reason="second")
    assert metrics.get("skipped") is True


@pytest.mark.asyncio
async def test_aclose_all_hydrus_clients_clears_pool():
    class _FakeClient:
        async def aclose(self) -> None:
            return None

    _client_pool[("http://test.invalid", "key")] = _FakeClient()
    assert len(_client_pool) == 1
    await aclose_all_hydrus_clients()
    assert len(_client_pool) == 0


def test_app_lifespan_exit_runs_coordinated_shutdown(test_config):
    """TestClient lifespan shutdown must unload models (Ctrl+C / SIGTERM path)."""
    reset_coordinated_tagging_shutdown_for_tests()
    reset_face_load_control_for_tests()
    ts, fs = _prime_loaded_models(test_config)

    from fastapi.testclient import TestClient

    from backend.app import app

    with TestClient(app) as client:
        status = client.get("/api/app/status").json()
        assert status["success"] is True
        assert status["loaded_model"] == test_config.default_model
        assert status["face_model_loaded"] is True

    assert ts.engine.session is None
    assert ts._loaded_model is None
    assert not fs.engine.loaded
    assert fs._warmup_done is False

    reset_face_load_control_for_tests()


def test_face_load_executor_shutdown_is_idempotent():
    reset_face_load_control_for_tests()
    get_face_load_executor()
    shutdown_face_load_executor(wait=False, reason="test")
    shutdown_face_load_executor(wait=False, reason="test_again")
    reset_face_load_control_for_tests()

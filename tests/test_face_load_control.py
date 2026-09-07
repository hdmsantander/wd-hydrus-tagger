"""Face model load executor and timeout behaviour."""

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]

from backend.face.load_control import (
    abandon_face_load_worker,
    bump_load_generation,
    cancel_pending_face_loads,
    is_load_generation_current,
    reset_face_load_control_for_tests,
    submit_face_load,
)
from backend.face.service import FaceTaggingService


@pytest.fixture(autouse=True)
def _reset_load_control():
    reset_face_load_control_for_tests()
    yield
    reset_face_load_control_for_tests()


def test_load_generation_invalidation():
    g1 = bump_load_generation()
    assert is_load_generation_current(g1)
    g2 = abandon_face_load_worker("test")
    assert not is_load_generation_current(g1)
    assert is_load_generation_current(g2)


@pytest.mark.asyncio
async def test_ensure_model_loaded_timeout_retries_cpu(test_config, monkeypatch):
    FaceTaggingService._instance = None
    cfg = test_config.model_copy(
        update={"use_gpu": True, "gpu_backend": "rocm", "face_model_load_timeout_seconds": 0.05}
    )
    FaceTaggingService._instance = None
    svc = FaceTaggingService.get_instance(cfg)
    attempts: list[bool] = []

    def fake_run_load(self, *, force_cpu: bool, generation: int):
        attempts.append(force_cpu)
        if not force_cpu:
            time.sleep(0.2)
            return
        self._app = object()
        self._providers = ["CPUExecutionProvider"]
        self._ctx_id = -1
        self._load_stage = "ready"
        self._loaded_with_cpu_fallback = True

    monkeypatch.setattr(svc.engine, "run_load", fake_run_load.__get__(svc.engine, type(svc.engine)))
    monkeypatch.setattr(svc, "_warmup_inference", AsyncMock())

    await svc.ensure_model_loaded(total=1)
    assert attempts == [False, True]
    assert svc.engine.loaded
    assert svc.engine.loaded_with_cpu_fallback


@pytest.mark.asyncio
async def test_ensure_model_loaded_cancel_aborts(test_config, monkeypatch):
    FaceTaggingService._instance = None
    cfg = test_config.model_copy(update={"use_gpu": False})
    svc = FaceTaggingService.get_instance(cfg)
    cancel_event = asyncio.Event()

    def slow_run_load(self, *, force_cpu: bool, generation: int):
        time.sleep(0.3)

    monkeypatch.setattr(svc.engine, "run_load", slow_run_load.__get__(svc.engine, type(svc.engine)))

    async def _cancel_soon():
        await asyncio.sleep(0.05)
        cancel_event.set()

    task = asyncio.create_task(_cancel_soon())
    with pytest.raises(asyncio.CancelledError):
        await svc.ensure_model_loaded(total=1, cancel_event=cancel_event)
    await task
    assert not svc.engine.loaded


def test_submit_face_load_runs_in_worker():
    seen: list[int] = []

    def work():
        seen.append(1)

    fut = submit_face_load(work)
    fut.result(timeout=5)
    assert seen == [1]


def test_cancel_pending_bumps_generation():
    g = bump_load_generation()
    g2 = cancel_pending_face_loads("unit")
    assert g2 != g
    assert not is_load_generation_current(g)

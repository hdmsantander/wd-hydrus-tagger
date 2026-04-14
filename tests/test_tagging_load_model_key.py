"""ORT load key (model + GPU + thread counts) and optional session overrides."""

import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]

from backend.services.model_manager import ModelVerifyResult
from backend.services.tagging_service import TaggingService


@pytest.fixture
def tagging_disk_hit_service(test_config, monkeypatch):
    """TaggingService with model_manager mocked for fast load_model paths."""
    TaggingService._instance = None
    svc = TaggingService.get_instance(test_config)
    monkeypatch.setattr(svc.model_manager, "is_downloaded", lambda _n: True)
    monkeypatch.setattr(
        svc.model_manager,
        "verify_model",
        lambda name, check_remote=False: ModelVerifyResult(name=name, ok=True, issues=[]),
    )
    monkeypatch.setattr(svc.model_manager, "repair_manifest_if_missing", lambda _n: False)
    monkeypatch.setattr(svc.model_manager, "get_model_path", lambda n: svc.model_manager.models_dir / n)
    return svc


def test_resolve_ort_threads_uses_config_when_none(test_config):
    TaggingService._instance = None
    svc = TaggingService.get_instance(test_config)
    a, b = svc._resolve_ort_threads(None, None)
    assert a == test_config.cpu_intra_op_threads
    assert b == test_config.cpu_inter_op_threads


def test_resolve_ort_threads_clamps(test_config):
    TaggingService._instance = None
    svc = TaggingService.get_instance(test_config)
    a, b = svc._resolve_ort_threads(200, -5)
    assert a == 64
    assert b == 1


def test_load_model_skips_reload_same_ort_key(tagging_disk_hit_service, monkeypatch):
    svc = tagging_disk_hit_service
    loads: list[tuple[int, int]] = []

    def track_load(models_root, name, *, intra_op_threads, inter_op_threads, **kwargs):
        loads.append((intra_op_threads, inter_op_threads))

    monkeypatch.setattr(svc.engine, "load", track_load)
    name = svc.config.default_model
    svc.load_model(name)
    svc.load_model(name)
    assert loads == [
        (svc.config.cpu_intra_op_threads, svc.config.cpu_inter_op_threads),
    ]


def test_load_model_reloads_when_ort_threads_change(tagging_disk_hit_service, monkeypatch):
    svc = tagging_disk_hit_service
    loads: list[tuple[int, int]] = []

    def track_load(models_root, name, *, intra_op_threads, inter_op_threads, **kwargs):
        loads.append((intra_op_threads, inter_op_threads))

    monkeypatch.setattr(svc.engine, "load", track_load)
    name = svc.config.default_model
    svc.load_model(name, ort_intra_op_threads=4, ort_inter_op_threads=1)
    svc.load_model(name, ort_intra_op_threads=6, ort_inter_op_threads=1)
    assert loads == [(4, 1), (6, 1)]


@pytest.mark.asyncio
async def test_ensure_model_override_matches_loaded(test_config, monkeypatch):
    TaggingService._instance = None
    svc = TaggingService.get_instance(test_config)
    called: list[bool] = []

    def fake_load(name: str, **kwargs):
        called.append(True)
        svc._loaded_model = name
        svc._loaded_ort_threads = svc._resolve_ort_threads(
            kwargs.get("ort_intra_op_threads"),
            kwargs.get("ort_inter_op_threads"),
        )

    monkeypatch.setattr(svc, "load_model", fake_load)
    await svc.ensure_model(test_config.default_model, ort_intra_op_threads=3, ort_inter_op_threads=2)
    assert len(called) == 1
    await svc.ensure_model(test_config.default_model, ort_intra_op_threads=3, ort_inter_op_threads=2)
    assert len(called) == 1

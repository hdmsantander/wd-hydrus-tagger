"""Structured INFO metrics for model cache, marker pre-skip, and load timings."""

import logging

import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]

import backend.config as config_module
from backend.config import AppConfig
from backend.services.model_manager import ModelVerifyResult
from backend.services.tagging_service import TaggingService


@pytest.mark.asyncio
async def test_ensure_model_memory_cache_hit_logs_metrics(test_config, caplog):
    caplog.set_level(logging.INFO, logger="backend.services.tagging_service")
    service = TaggingService.get_instance(test_config)
    service._loaded_model = test_config.default_model
    await service.ensure_model(test_config.default_model)
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "ensure_model metrics" in joined
    assert "memory_cache_hit=True" in joined


@pytest.mark.asyncio
async def test_ensure_model_load_logs_memory_cache_false(test_config, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="backend.services.tagging_service")
    service = TaggingService.get_instance(test_config)
    service._loaded_model = None

    def fake_load(name: str, **kwargs) -> None:
        service._loaded_model = name

    monkeypatch.setattr(service, "load_model", fake_load)
    await service.ensure_model(test_config.default_model)
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "memory_cache_hit=False" in joined


def test_load_model_metrics_disk_cache_hit(test_config, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="backend.services.tagging_service")
    service = TaggingService.get_instance(test_config)
    service._loaded_model = None
    monkeypatch.setattr(service.model_manager, "is_downloaded", lambda _n: True)
    monkeypatch.setattr(
        service.model_manager,
        "verify_model",
        lambda name, check_remote=False: ModelVerifyResult(name=name, ok=True, issues=[]),
    )
    monkeypatch.setattr(service.model_manager, "repair_manifest_if_missing", lambda _n: False)
    monkeypatch.setattr(service.model_manager, "get_model_path", lambda n: service.model_manager.models_dir / n)
    monkeypatch.setattr(service.engine, "load", lambda *a, **k: None)

    service.load_model(test_config.default_model)
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "load_model metrics" in joined
    assert "disk_cache_hit=True" in joined


@pytest.mark.asyncio
async def test_tag_files_metrics_skipped_pre_infer_marker(
    tmp_path, monkeypatch, caplog,
):
    caplog.set_level(logging.INFO, logger="backend.services.tagging_service")
    md = tmp_path / "models"
    md.mkdir(parents=True, exist_ok=True)
    cfg = AppConfig(
        models_dir=str(md),
        hydrus_api_key="k",
        hydrus_api_url="http://invalid.test",
        batch_size=8,
        default_model="wd-vit-tagger-v3",
        wd_skip_inference_if_marker_present=True,
        wd_append_model_marker_tag=False,
    )
    monkeypatch.setattr(config_module, "_config", cfg)
    TaggingService._instance = None
    service = TaggingService.get_instance(cfg)
    service._loaded_model = cfg.default_model
    service.engine.predict = lambda imgs, g, c: [
        {"general_tags": {"x": 1.0}, "character_tags": {}, "rating_tags": {}}
        for _ in imgs
    ]

    class MetaHydrus:
        async def get_file_metadata(self, file_ids: list[int]):
            return [
                {
                    "file_id": 1,
                    "hash": "h1",
                    "tags": {
                        "svc": {
                            "storage_tags": {"0": ["wd14:wd-vit-tagger-v3"]},
                            "display_tags": {},
                        },
                    },
                },
                {"file_id": 2, "hash": "h2", "tags": {}},
            ]

        async def get_file(self, file_id: int):
            from io import BytesIO
            from PIL import Image

            buf = BytesIO()
            Image.new("RGB", (4, 4), color=(1, 2, 3)).save(buf, format="JPEG")
            return buf.getvalue(), "image/jpeg"

        async def get_thumbnail(self, file_id: int):
            return await self.get_file(file_id)

    await service.tag_files(MetaHydrus(), [1, 2], 0.35, 0.85, batch_size=8, service_key="svc")
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "tag_files metrics" in joined
    assert "skipped_pre_infer_marker_files=1" in joined
    assert "inferred_files=1" in joined

"""Timing / profiling checks: parallel Hydrus fetches vs sequential upper bound.

These tests use mocked delays (no real Hydrus). They guard regressions where
gather+semaphore would accidentally serialize.
"""

import asyncio
import time
from io import BytesIO

import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]

from PIL import Image

import backend.config as config_module
import backend.services.tagging_service as tagging_service_module
from backend.config import AppConfig
from backend.services.tagging_service import TaggingService


def _tiny_jpeg() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (4, 4), color=(1, 2, 3)).save(buf, format="JPEG")
    return buf.getvalue()


class FakeHydrusForProfile:
    async def get_file_metadata(self, file_ids: list[int]):
        return [{"file_id": fid, "hash": f"h{fid}"} for fid in file_ids]

    async def get_file(self, file_id: int):
        return _tiny_jpeg(), "image/jpeg"

    async def get_thumbnail(self, file_id: int):
        return _tiny_jpeg(), "image/jpeg"


@pytest.mark.asyncio
async def test_tag_files_parallel_fetch_under_serial_upper_bound(tmp_path, monkeypatch):
    """4 files × per-file sleep should complete in ~one sleep (parallel=4), not ~4×."""
    delay = 0.06
    lock = asyncio.Lock()
    concurrent = 0
    peak = 0

    class DelayedHydrus:
        async def get_file_metadata(self, file_ids: list[int]):
            return [{"file_id": fid, "hash": f"h{fid}"} for fid in file_ids]

        async def get_file(self, file_id: int):
            nonlocal concurrent, peak
            async with lock:
                concurrent += 1
                peak = max(peak, concurrent)
            try:
                await asyncio.sleep(delay)
                return _tiny_jpeg(), "image/jpeg"
            finally:
                async with lock:
                    concurrent -= 1

        async def get_thumbnail(self, file_id: int):
            return _tiny_jpeg(), "image/jpeg"

    # `tmp_models_dir` (via autouse `test_config`) already created tmp_path / "models".
    md = tmp_path / "models"
    md.mkdir(parents=True, exist_ok=True)
    cfg = AppConfig(
        models_dir=str(md),
        hydrus_api_key="k",
        hydrus_api_url="http://invalid.test",
        batch_size=4,
        hydrus_download_parallel=4,
        default_model="wd-vit-tagger-v3",
    )
    monkeypatch.setattr(config_module, "_config", cfg)
    monkeypatch.setattr(config_module, "get_config", lambda: config_module._config)
    monkeypatch.setattr(config_module, "load_config", lambda: config_module._config)
    TaggingService._instance = None

    service = TaggingService.get_instance(cfg)
    service._loaded_model = cfg.default_model

    def fake_predict(images, g, c):
        return [
            {"general_tags": {}, "character_tags": {}, "rating_tags": {}}
            for _ in images
        ]

    service.engine.predict = fake_predict

    # tag_files ends with gc.collect(); a real full collection can dominate wall time on CI.
    monkeypatch.setattr(tagging_service_module.gc, "collect", lambda *a, **k: 0)

    t0 = time.perf_counter()
    await service.tag_files(DelayedHydrus(), [1, 2, 3, 4], 0.35, 0.85, batch_size=4)
    elapsed = time.perf_counter() - t0

    assert peak >= 2, "expected overlapping get_file calls"
    # Serial upper bound ~4× delay; parallel should be ~1× delay (+ jitter).
    assert elapsed < delay * 2.5, (
        f"elapsed {elapsed:.3f}s — expected parallel fetch ~{delay}s, not ~{4 * delay}s serial"
    )


@pytest.mark.asyncio
async def test_tag_files_logs_fetch_and_predict_durations_ordering(test_config, monkeypatch, caplog):
    """Smoke: tag_files emits fetch timing before predict-done (structure only)."""
    import logging

    service = TaggingService.get_instance(test_config)
    service._loaded_model = test_config.default_model
    service.engine.predict = lambda imgs, g, c: [
        {"general_tags": {}, "character_tags": {}, "rating_tags": {}}
        for _ in imgs
    ]

    caplog.set_level(logging.INFO, logger="backend.services.tagging_service")
    await service.tag_files(FakeHydrusForProfile(), [10], 0.35, 0.85, batch_size=4)

    text = caplog.text
    assert "fetched_ok=" in text
    assert "fetch " in text and "s)" in text
    assert "predict done" in text
    assert "tag_files metrics" in text
    assert "skipped_pre_infer_marker_files=0" in text
    assert "wall_hydrus_fetch_s=" in text

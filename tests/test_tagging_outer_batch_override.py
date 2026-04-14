"""tag_files outer_batch_override (marker-skip tail large batches)."""

from io import BytesIO

import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]

from PIL import Image

from backend.hydrus.tag_merge import build_wd_model_marker
from backend.services.tagging_service import TaggingService


def _tiny_jpeg() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (4, 4), color=(9, 9, 9)).save(buf, format="JPEG")
    return buf.getvalue()


class HC:
    async def get_file_metadata(self, file_ids: list[int]):
        return [{"file_id": fid, "hash": f"h{fid}"} for fid in file_ids]

    async def get_file(self, file_id: int):
        return _tiny_jpeg(), "image/jpeg"

    async def get_thumbnail(self, file_id: int):
        return _tiny_jpeg(), "image/jpeg"


@pytest.mark.asyncio
async def test_outer_batch_override_one_outer_loop_skip_only(test_config):
    """All files skip ONNX — large override should use a single outer batch (no predict)."""
    service = TaggingService.get_instance(test_config)
    service._loaded_model = test_config.default_model
    predict_calls: list[int] = []

    def _predict(imgs, g, c):
        predict_calls.append(len(imgs))
        return [
            {"general_tags": {}, "character_tags": {}, "rating_tags": {}}
            for _ in imgs
        ]

    service.engine.predict = _predict
    marker = build_wd_model_marker(test_config.default_model, "")
    meta = {}
    for i in range(80):
        meta[i] = {
            "file_id": i,
            "hash": f"h{i}",
            "tags": {
                "svc": {
                    "storage_tags": {"0": [marker]},
                    "display_tags": {},
                },
            },
        }
    client = HC()
    results = await service.tag_files(
        client,
        list(range(80)),
        0.35,
        0.85,
        batch_size=8,
        service_key="svc",
        prefetched_meta_by_id=meta,
        outer_batch_override=80,
    )
    assert len(results) == 80
    assert predict_calls == []


@pytest.mark.asyncio
async def test_outer_batch_override_ignored_when_none_matches_default_clamp(test_config):
    service = TaggingService.get_instance(test_config)
    service._loaded_model = test_config.default_model
    service.engine.predict = lambda imgs, g, c: [
        {"general_tags": {"x": 1.0}, "character_tags": {}, "rating_tags": {}}
        for _ in imgs
    ]
    meta = {0: {"file_id": 0, "hash": "h0", "tags": {}}}
    client = HC()
    r = await service.tag_files(
        client,
        [0],
        0.35,
        0.85,
        batch_size=1,
        service_key="svc",
        prefetched_meta_by_id=meta,
    )
    assert len(r) == 1

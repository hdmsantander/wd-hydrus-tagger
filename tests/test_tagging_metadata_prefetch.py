"""Session-style metadata prefetch: WebSocket passes prefetched maps into tag_files."""

from io import BytesIO

import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]

from PIL import Image

from backend.services.tagging_service import TaggingService
from backend.services.tagging_shared import load_metadata_by_file_id


def _tiny_jpeg() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (4, 4), color=(9, 9, 9)).save(buf, format="JPEG")
    return buf.getvalue()


class CountingHydrus:
    def __init__(self):
        self.metadata_calls = 0

    async def get_file_metadata(self, file_ids: list[int]):
        self.metadata_calls += 1
        return [{"file_id": fid, "hash": f"h{fid}"} for fid in file_ids]

    async def get_file(self, file_id: int):
        return _tiny_jpeg(), "image/jpeg"

    async def get_thumbnail(self, file_id: int):
        return _tiny_jpeg(), "image/jpeg"


@pytest.mark.asyncio
async def test_tag_files_prefetched_avoids_metadata_calls(test_config):
    service = TaggingService.get_instance(test_config)
    service._loaded_model = test_config.default_model
    service.engine.predict = lambda imgs, g, c: [
        {"general_tags": {}, "character_tags": {}, "rating_tags": {}}
        for _ in imgs
    ]
    client = CountingHydrus()
    pref = {i: {"file_id": i, "hash": f"h{i}"} for i in range(16)}
    await service.tag_files(
        client,
        list(range(8)),
        0.35,
        0.85,
        batch_size=8,
        prefetched_meta_by_id=pref,
    )
    await service.tag_files(
        client,
        list(range(8, 16)),
        0.35,
        0.85,
        batch_size=8,
        prefetched_meta_by_id=pref,
    )
    assert client.metadata_calls == 0


@pytest.mark.asyncio
async def test_tag_files_prefetched_partial_fetches_only_missing(test_config):
    service = TaggingService.get_instance(test_config)
    service._loaded_model = test_config.default_model
    service.engine.predict = lambda imgs, g, c: [
        {"general_tags": {"a": 1.0}, "character_tags": {}, "rating_tags": {}}
        for _ in imgs
    ]
    client = CountingHydrus()
    pref = {1: {"file_id": 1, "hash": "h1"}}
    results = await service.tag_files(
        client,
        [1, 2],
        0.35,
        0.85,
        batch_size=8,
        prefetched_meta_by_id=pref,
    )
    assert client.metadata_calls == 1
    assert len(results) == 2


@pytest.mark.asyncio
async def test_load_metadata_by_file_id_chunks_and_index():
    class HC:
        def __init__(self):
            self.calls: list[list[int]] = []

        async def get_file_metadata(self, file_ids: list[int]):
            self.calls.append(list(file_ids))
            return [{"file_id": fid, "hash": str(fid)} for fid in file_ids]

    h = HC()
    m = await load_metadata_by_file_id(h, [5, 9, 12, 20], chunk_sz=2)
    assert len(h.calls) == 2
    assert h.calls[0] == [5, 9]
    assert h.calls[1] == [12, 20]
    assert m[5]["hash"] == "5"
    assert m[20]["hash"] == "20"

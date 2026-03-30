"""Tagging orchestration (mocked Hydrus + ONNX)."""

from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

import backend.config as config_module
from backend.config import AppConfig
from backend.services.tagging_service import TaggingService


def _tiny_jpeg() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (4, 4), color=(128, 64, 32)).save(buf, format="JPEG")
    return buf.getvalue()


class FakeHydrus:
    def __init__(self, file_ids):
        self.file_ids = file_ids

    async def get_file_metadata(self, file_ids: list[int]):
        return [{"file_id": fid, "hash": f"hash{fid}"} for fid in file_ids]

    async def get_file(self, file_id: int):
        return _tiny_jpeg(), "image/jpeg"

    async def get_thumbnail(self, file_id: int):
        return _tiny_jpeg(), "image/jpeg"


@pytest.mark.asyncio
async def test_tag_files_returns_formatted_tags(test_config, monkeypatch):
    service = TaggingService.get_instance(test_config)
    service._loaded_model = test_config.default_model

    def fake_predict(images, general_threshold, character_threshold):
        return [
            {
                "general_tags": {"cat": 0.9},
                "character_tags": {},
                "rating_tags": {"safe": 0.99},
            }
            for _ in images
        ]

    service.engine.predict = fake_predict

    client = FakeHydrus([])
    results = await service.tag_files(client, [101, 102], 0.35, 0.85)

    assert len(results) == 2
    assert results[0]["hash"] == "hash101"
    assert "cat" in results[0]["general_tags"]
    assert results[0]["formatted_tags"]
    assert "wd14:wd-vit-tagger-v3" in results[0]["tags"]


@pytest.mark.asyncio
async def test_tag_files_batch_size_override(test_config):
    test_config.batch_size = 10
    service = TaggingService.get_instance(test_config)
    service._loaded_model = test_config.default_model

    calls = []

    def fake_predict(images, g, c):
        calls.append(len(images))
        return [
            {"general_tags": {}, "character_tags": {}, "rating_tags": {}}
            for _ in images
        ]

    service.engine.predict = fake_predict
    client = FakeHydrus(list(range(25)))

    await service.tag_files(client, list(range(25)), batch_size=7)

    assert calls == [7, 7, 7, 4]


@pytest.mark.asyncio
async def test_tag_files_skips_failed_download(test_config):
    service = TaggingService.get_instance(test_config)
    service._loaded_model = test_config.default_model

    def fake_predict(images, g, c):
        return [
            {"general_tags": {"x": 1.0}, "character_tags": {}, "rating_tags": {}}
            for _ in images
        ]

    service.engine.predict = fake_predict

    class FlakyHydrus(FakeHydrus):
        async def get_file(self, file_id: int):
            if file_id == 2:
                raise OSError("boom")
            return await super().get_file(file_id)

    client = FlakyHydrus([1, 2, 3])
    results = await service.tag_files(client, [1, 2, 3], batch_size=8)

    assert len(results) == 2
    assert {r["file_id"] for r in results} == {1, 3}


@pytest.mark.asyncio
async def test_tag_files_pairs_metadata_by_file_id_not_list_order(test_config):
    service = TaggingService.get_instance(test_config)
    service._loaded_model = test_config.default_model

    def fake_predict(images, g, c):
        return [
            {"general_tags": {}, "character_tags": {}, "rating_tags": {}}
            for _ in images
        ]

    service.engine.predict = fake_predict

    class OrderHydrus(FakeHydrus):
        async def get_file_metadata(self, file_ids: list[int]):
            return [
                {"file_id": 202, "hash": "hashB"},
                {"file_id": 101, "hash": "hashA"},
            ]

    client = OrderHydrus([])
    results = await service.tag_files(client, [101, 202], 0.35, 0.85, batch_size=8)
    assert len(results) == 2
    by_id = {r["file_id"]: r["hash"] for r in results}
    assert by_id[101] == "hashA"
    assert by_id[202] == "hashB"


@pytest.mark.asyncio
async def test_tag_files_video_mime_skips_full_file_download(test_config):
    """Hydrus metadata mime video/* → only thumbnail is fetched (no get_files/file)."""
    service = TaggingService.get_instance(test_config)
    service._loaded_model = test_config.default_model

    def fake_predict(images, g, c):
        return [
            {"general_tags": {"v": 1.0}, "character_tags": {}, "rating_tags": {}}
            for _ in images
        ]

    service.engine.predict = fake_predict

    class VideoHydrus(FakeHydrus):
        def __init__(self, file_ids):
            super().__init__(file_ids)
            self.get_file_called = False

        async def get_file_metadata(self, file_ids: list[int]):
            return [{"file_id": fid, "hash": f"hash{fid}", "mime": "video/mp4"} for fid in file_ids]

        async def get_file(self, file_id: int):
            self.get_file_called = True
            raise AssertionError("get_file must not be called when mime is video/*")

    client = VideoHydrus([99])
    results = await service.tag_files(client, [99], 0.35, 0.85)
    assert not client.get_file_called
    assert len(results) == 1
    assert results[0]["file_id"] == 99


@pytest.mark.asyncio
async def test_tag_files_predict_failure_skips_batch_without_raising(test_config):
    service = TaggingService.get_instance(test_config)
    service._loaded_model = test_config.default_model

    def boom(_images, _g, _c):
        raise RuntimeError("simulated ONNX failure")

    service.engine.predict = boom
    client = FakeHydrus([1, 2])
    results = await service.tag_files(client, [1, 2], 0.35, 0.85, batch_size=8)
    assert results == []


@pytest.mark.asyncio
async def test_tag_files_thumbnail_fallback_when_full_file_not_image(test_config):
    service = TaggingService.get_instance(test_config)
    service._loaded_model = test_config.default_model

    def fake_predict(images, g, c):
        return [
            {"general_tags": {"a": 1.0}, "character_tags": {}, "rating_tags": {}}
            for _ in images
        ]

    service.engine.predict = fake_predict

    not_an_image = b"not a jpeg or png header"

    class ThumbHydrus(FakeHydrus):
        async def get_file(self, file_id: int):
            return not_an_image, "application/octet-stream"

    client = ThumbHydrus([10])
    results = await service.tag_files(client, [10], 0.35, 0.85)
    assert len(results) == 1
    assert results[0]["file_id"] == 10


def test_get_instance_refreshes_model_manager_when_models_dir_changes(tmp_path, monkeypatch):
    """Changing models_dir must not keep a stale ModelManager / ONNX path on the singleton."""
    md1 = tmp_path / "models_a"
    md2 = tmp_path / "models_b"
    md1.mkdir(parents=True)
    md2.mkdir(parents=True)
    base = dict(
        hydrus_api_key="k",
        hydrus_api_url="http://invalid.test",
        batch_size=4,
        default_model="wd-vit-tagger-v3",
    )
    cfg1 = AppConfig(models_dir=str(md1), **base)
    cfg2 = AppConfig(models_dir=str(md2), **base)
    monkeypatch.setattr(config_module, "_config", cfg1)
    TaggingService._instance = None
    s = TaggingService.get_instance(cfg1)
    assert Path(s.model_manager.models_dir).resolve() == md1.resolve()
    monkeypatch.setattr(config_module, "_config", cfg2)
    s2 = TaggingService.get_instance(cfg2)
    assert s2 is s
    assert Path(s.model_manager.models_dir).resolve() == md2.resolve()
    assert s._loaded_model is None


@pytest.mark.asyncio
async def test_tag_files_skips_inference_when_marker_present_on_service(
    test_config, monkeypatch,
):
    cfg = test_config.model_copy(update={
        "wd_skip_inference_if_marker_present": True,
        "wd_append_model_marker_tag": False,
    })
    monkeypatch.setattr(config_module, "_config", cfg)
    TaggingService._instance = None
    service = TaggingService.get_instance(cfg)
    service._loaded_model = cfg.default_model

    predict_calls: list[int] = []

    def fake_predict(images, g, c):
        predict_calls.append(len(images))
        return [
            {"general_tags": {"x": 1.0}, "character_tags": {}, "rating_tags": {}}
            for _ in images
        ]

    service.engine.predict = fake_predict

    class MetaHydrus(FakeHydrus):
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

    client = MetaHydrus([])
    results = await service.tag_files(
        client, [1, 2], 0.35, 0.85, batch_size=8, service_key="svc",
    )
    assert predict_calls == [1]
    assert len(results) == 2
    assert results[0]["skipped_inference"] is True
    assert results[0]["tags"] == []
    assert not results[1].get("skipped_inference")
    assert results[1]["general_tags"].get("x") == 1.0


@pytest.mark.asyncio
async def test_tag_files_marker_on_wrong_service_still_infers(
    test_config, monkeypatch,
):
    cfg = test_config.model_copy(update={
        "wd_skip_inference_if_marker_present": True,
        "wd_append_model_marker_tag": False,
    })
    monkeypatch.setattr(config_module, "_config", cfg)
    TaggingService._instance = None
    service = TaggingService.get_instance(cfg)
    service._loaded_model = cfg.default_model

    predict_calls: list[int] = []

    def fake_predict(images, g, c):
        predict_calls.append(len(images))
        return [
            {"general_tags": {}, "character_tags": {}, "rating_tags": {}}
            for _ in images
        ]

    service.engine.predict = fake_predict

    class MetaHydrus(FakeHydrus):
        async def get_file_metadata(self, file_ids: list[int]):
            return [
                {
                    "file_id": 1,
                    "hash": "h1",
                    "tags": {
                        "other": {
                            "storage_tags": {"0": ["wd14:wd-vit-tagger-v3"]},
                            "display_tags": {},
                        },
                    },
                },
            ]

    client = MetaHydrus([])
    await service.tag_files(client, [1], 0.35, 0.85, service_key="svc")
    assert predict_calls == [1]


@pytest.mark.asyncio
async def test_tag_files_marker_any_service_when_no_service_key(
    test_config, monkeypatch,
):
    cfg = test_config.model_copy(update={
        "wd_skip_inference_if_marker_present": True,
        "wd_append_model_marker_tag": False,
    })
    monkeypatch.setattr(config_module, "_config", cfg)
    TaggingService._instance = None
    service = TaggingService.get_instance(cfg)
    service._loaded_model = cfg.default_model

    predict_calls: list[int] = []

    def fake_predict(images, g, c):
        predict_calls.append(len(images))
        return [
            {"general_tags": {}, "character_tags": {}, "rating_tags": {}}
            for _ in images
        ]

    service.engine.predict = fake_predict

    class MetaHydrus(FakeHydrus):
        async def get_file_metadata(self, file_ids: list[int]):
            return [
                {
                    "file_id": 1,
                    "hash": "h1",
                    "tags": {
                        "z": {
                            "storage_tags": {"0": ["wd14:wd-vit-tagger-v3"]},
                            "display_tags": {},
                        },
                    },
                },
            ]

    client = MetaHydrus([])
    await service.tag_files(client, [1], 0.35, 0.85, service_key="")
    assert predict_calls == []


@pytest.mark.asyncio
async def test_tag_files_skips_inference_when_marker_uses_underscores_in_hydrus(
    test_config, monkeypatch,
):
    """Hydrus may store wd14:wd_vit_tagger_v3 while the app builds wd14:wd-vit-tagger-v3."""
    cfg = test_config.model_copy(update={
        "wd_skip_inference_if_marker_present": True,
        "wd_append_model_marker_tag": False,
    })
    monkeypatch.setattr(config_module, "_config", cfg)
    TaggingService._instance = None
    service = TaggingService.get_instance(cfg)
    service._loaded_model = cfg.default_model

    predict_calls: list[int] = []

    def fake_predict(images, g, c):
        predict_calls.append(len(images))
        return [
            {"general_tags": {"x": 1.0}, "character_tags": {}, "rating_tags": {}}
            for _ in images
        ]

    service.engine.predict = fake_predict

    class MetaHydrus(FakeHydrus):
        async def get_file_metadata(self, file_ids: list[int]):
            return [
                {
                    "file_id": 1,
                    "hash": "h1",
                    "tags": {
                        "svc": {
                            "storage_tags": {"0": ["wd14:wd_vit_tagger_v3"]},
                            "display_tags": {},
                        },
                    },
                },
            ]

    client = MetaHydrus([])
    await service.tag_files(client, [1], 0.35, 0.85, service_key="svc")
    assert predict_calls == []


@pytest.mark.asyncio
async def test_tag_files_dedupe_stale_wd_markers_in_formatted_list(
    test_config, monkeypatch,
):
    cfg = test_config.model_copy(update={
        "wd_append_model_marker_tag": True,
        "wd_skip_inference_if_marker_present": False,
    })
    monkeypatch.setattr(config_module, "_config", cfg)
    TaggingService._instance = None
    service = TaggingService.get_instance(cfg)
    service._loaded_model = cfg.default_model

    def fake_predict(images, g, c):
        return [
            {"general_tags": {"z": 1.0}, "character_tags": {}, "rating_tags": {}}
            for _ in images
        ]

    service.engine.predict = fake_predict

    def fake_format(self, prediction):
        return ["z", "wd14:other-model", "character:foo"]

    monkeypatch.setattr(TaggingService, "_format_tags", fake_format)

    client = FakeHydrus([1])
    results = await service.tag_files(client, [1], 0.35, 0.85, batch_size=8)
    assert results[0]["wd_stale_markers_removed"] == 1
    assert results[0]["tags"][-1] == "wd14:wd-vit-tagger-v3"
    assert "wd14:other-model" not in results[0]["tags"]

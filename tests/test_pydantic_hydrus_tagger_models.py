"""Pydantic models for Hydrus API and tagger API responses (serialization contract)."""

import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]

from backend.hydrus.models import FileMetadata, ServiceInfo
from backend.tagger.models import TagResult


def test_service_info_roundtrip():
    s = ServiceInfo(
        service_key="abc",
        name="my tags",
        type=0,
        type_pretty="local tags",
    )
    d = s.model_dump()
    assert ServiceInfo.model_validate(d) == s


def test_file_metadata_optional_fields():
    m = FileMetadata(file_id=1, hash="deadbeef", width=1920, height=1080)
    assert m.size is None
    assert m.mime is None
    assert m.width == 1920


def test_tag_result_formatted_tags():
    r = TagResult(
        file_id=1,
        hash="h",
        general_tags={"a": 0.9},
        character_tags={},
        rating_tags={"safe": 0.99},
        formatted_tags=["a", "rating:safe"],
    )
    assert r.formatted_tags[0] == "a"

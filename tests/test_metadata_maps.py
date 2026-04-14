"""Tests for backend.hydrus.metadata_maps."""

import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]

from backend.hydrus.metadata_maps import rows_to_file_id_map


def test_rows_to_file_id_map_empty():
    assert rows_to_file_id_map([]) == {}


def test_rows_to_file_id_map_skips_non_dict():
    assert rows_to_file_id_map([None, "x", 1, []]) == {}


def test_rows_to_file_id_map_skips_missing_file_id():
    assert rows_to_file_id_map([{"hash": "abc"}]) == {}


def test_rows_to_file_id_map_skips_bad_file_id():
    assert rows_to_file_id_map([{"file_id": "not-int"}]) == {}


def test_rows_to_file_id_map_coerces_int_like():
    assert rows_to_file_id_map([{"file_id": "42", "k": "v"}]) == {42: {"file_id": "42", "k": "v"}}


def test_rows_to_file_id_map_duplicate_last_wins():
    rows = [
        {"file_id": 1, "a": 1},
        {"file_id": 1, "a": 2},
    ]
    assert rows_to_file_id_map(rows)[1]["a"] == 2


def test_rows_to_file_id_map_multiple_ids():
    rows = [{"file_id": 10}, {"file_id": 20, "mime": "image/png"}]
    m = rows_to_file_id_map(rows)
    assert set(m) == {10, 20}
    assert m[20]["mime"] == "image/png"

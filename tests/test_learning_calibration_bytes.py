"""Bytes-based learning split and edge cases (T-Learn follow-up)."""

import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]

from backend.services.learning_calibration import (
    MIN_LEARNING_PREFIX_FILES,
    compute_learning_split,
    compute_learning_split_by_bytes,
)


def test_compute_learning_split_invalid_scope_coerces_to_count():
    ids = [1, 2, 3, 4, 50]
    learn, commit, info = compute_learning_split(ids, learning_fraction=0.2, learning_scope="not_a_scope")
    assert learn + commit == ids
    assert info["learning_scope_effective"] == "count"
    assert "bytes_fallback" not in info or info.get("bytes_fallback") is not True


def test_compute_learning_split_by_bytes_empty_ids():
    la, co, info = compute_learning_split_by_bytes([], meta_by_id={}, learning_fraction=0.1)
    assert la == [] and co == []
    assert info["learning_count"] == 0


def test_compute_learning_split_by_bytes_empty_row_dict_triggers_missing_size():
    ids = [1, 2]
    meta = {1: {}, 2: {"size": 100}}
    _, _, info = compute_learning_split_by_bytes(ids, meta_by_id=meta, learning_fraction=0.5)
    assert info["bytes_fallback"] == "missing_size"


def test_compute_learning_split_by_bytes_no_metadata_fallback():
    ids = list(range(40))
    la, co, info = compute_learning_split_by_bytes(ids, meta_by_id=None, learning_fraction=0.1)
    assert info["bytes_fallback"] == "no_metadata"
    assert info["learning_scope_effective"] == "count"
    assert len(la) == MIN_LEARNING_PREFIX_FILES


def test_compute_learning_split_by_bytes_missing_size_fallback():
    ids = [1, 2, 3]
    meta = {1: {"size": 100}, 2: {"size": 100}, 3: {"hash": "x"}}
    la, co, info = compute_learning_split_by_bytes(ids, meta_by_id=meta, learning_fraction=0.2)
    assert info["bytes_fallback"] == "missing_size"


def test_compute_learning_split_by_bytes_zero_total_fallback():
    ids = [1, 2, 40]
    meta = {1: {"size": 0}, 2: {"size": 0}, 40: {"size": 0}}
    la, co, info = compute_learning_split_by_bytes(ids, meta_by_id=meta, learning_fraction=0.5)
    assert info["bytes_fallback"] == "zero_total_bytes"
    assert len(la) + len(co) == 3


def test_compute_learning_split_by_bytes_invalid_size_treated_missing():
    ids = [1, 2, 33]
    meta = {1: {"size": 100}, 2: {"size": "bad"}, 33: {"size": 100}}
    la, co, info = compute_learning_split_by_bytes(ids, meta_by_id=meta, learning_fraction=0.5)
    assert info["bytes_fallback"] == "missing_size"


def test_compute_learning_split_by_bytes_weighted_prefix_differs_from_count():
    """Small leading files, large tail: byte target hits later in the queue than pure count %."""
    ids = list(range(1, 41))
    meta = {i: {"size": 10} for i in range(1, 33)}
    for i in range(33, 41):
        meta[i] = {"size": 5000}
    la_b, co_b, inf_b = compute_learning_split_by_bytes(ids, meta_by_id=meta, learning_fraction=0.1)
    la_c, co_c, _ = compute_learning_split(ids, learning_fraction=0.1, learning_scope="count")
    assert inf_b["learning_scope_effective"] == "bytes"
    assert len(la_b) != len(la_c) or la_b != la_c
    assert inf_b["target_bytes"] >= 1
    assert inf_b["bytes_cumulative_to_split"] >= inf_b["target_bytes"]


def test_compute_learning_split_by_bytes_single_file():
    la, co, info = compute_learning_split_by_bytes([7], meta_by_id={7: {"size": 999}}, learning_fraction=0.5)
    assert la == [7] and co == []
    assert info["learning_scope_effective"] == "bytes"


def test_min_learning_prefix_constant():
    assert MIN_LEARNING_PREFIX_FILES == 32

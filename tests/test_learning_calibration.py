"""Unit tests for T-Learn queue split (``learning_calibration``)."""

import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]

from backend.services.learning_calibration import (
    compute_learning_split,
    parse_learning_fraction,
)


def test_parse_learning_fraction_defaults_and_clamp():
    assert parse_learning_fraction(None) == 0.1
    assert parse_learning_fraction("0.2") == 0.2
    assert parse_learning_fraction(99) == 0.5
    assert parse_learning_fraction(-1) == 0.01
    assert parse_learning_fraction("bad") == 0.1


def test_compute_learning_split_empty():
    a, b, info = compute_learning_split([], learning_fraction=0.1, learning_scope="count")
    assert a == [] and b == []
    assert info["learning_count"] == 0


def test_compute_learning_split_single_file_all_learning():
    a, b, info = compute_learning_split([42], learning_fraction=0.1, learning_scope="count")
    assert a == [42] and b == []
    assert info["learning_count"] == 1 and info["commit_count"] == 0


def test_compute_learning_split_respects_floor_and_commit_nonempty():
    ids = list(range(100))
    learn, commit, info = compute_learning_split(ids, learning_fraction=0.05, learning_scope="count")
    assert len(learn) == 32
    assert len(commit) == 68
    assert learn + commit == ids
    assert info["split_index"] == 32


def test_compute_learning_split_bytes_scope_falls_back():
    ids = [1, 2, 3]
    _, _, info = compute_learning_split(ids, learning_fraction=0.2, learning_scope="bytes")
    assert info.get("bytes_fallback") is True
    assert info["learning_scope_effective"] == "count"


@pytest.mark.parametrize("n", [2, 10, 50])
def test_compute_learning_split_partition_covers_all(n):
    ids = list(range(n))
    learn, commit, _ = compute_learning_split(ids, learning_fraction=0.1, learning_scope="count")
    assert learn + commit == ids
    if n > 1:
        assert len(commit) >= 1

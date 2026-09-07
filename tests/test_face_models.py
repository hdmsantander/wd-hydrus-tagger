"""Face pack cache inspect / verify (no InsightFace download)."""

from pathlib import Path

import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]

from backend.face.models import inspect_face_pack, verify_face_pack


def test_inspect_missing_pack(tmp_path: Path):
    row = inspect_face_pack(tmp_path, "buffalo_l")
    assert row["downloaded"] is False
    assert row["cache_ok"] is False
    assert any("missing" in i or "directory" in i for i in row["cache_issues"])


def test_inspect_valid_required_files(tmp_path: Path):
    pack = tmp_path / "models" / "buffalo_l"
    pack.mkdir(parents=True)
    (pack / "det_10g.onnx").write_bytes(b"x" * 1_500_000)
    (pack / "w600k_r50.onnx").write_bytes(b"y" * 1_500_000)
    row = inspect_face_pack(tmp_path, "buffalo_l")
    assert row["downloaded"] is True
    assert row["cache_ok"] is True
    assert row["cache_issues"] == []
    verified = verify_face_pack(tmp_path, "buffalo_l")
    assert verified["ok"] is True


def test_inspect_tiny_onnx_fails(tmp_path: Path):
    pack = tmp_path / "models" / "buffalo_l"
    pack.mkdir(parents=True)
    (pack / "det_10g.onnx").write_bytes(b"tiny")
    (pack / "w600k_r50.onnx").write_bytes(b"y" * 1_500_000)
    row = inspect_face_pack(tmp_path, "buffalo_l")
    assert row["downloaded"] is True
    assert row["cache_ok"] is False
    assert any("too small" in i for i in row["cache_issues"])

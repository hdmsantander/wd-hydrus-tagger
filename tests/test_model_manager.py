"""Model disk cache helpers."""

from pathlib import Path

from backend.services.model_manager import ModelManager, SUPPORTED_MODELS


def _valid_csv(path: Path, rows: int = 101) -> None:
    lines = ["tag_id,name,category,count"]
    for i in range(rows):
        lines.append(f"{i},n{i},0,1")
    path.write_text("\n".join(lines), encoding="utf-8")


def test_is_downloaded_requires_onnx_and_csv(tmp_path):
    name = next(iter(SUPPORTED_MODELS))
    mm = ModelManager(tmp_path)
    d = tmp_path / name
    d.mkdir(parents=True)
    assert not mm.is_downloaded(name)
    (d / "model.onnx").write_bytes(b"x")
    assert not mm.is_downloaded(name)
    (d / "selected_tags.csv").write_text("a\n")
    assert mm.is_downloaded(name)


def test_is_downloaded_unknown_model(tmp_path):
    mm = ModelManager(tmp_path)
    assert not mm.is_downloaded("not-a-real-model-id")


def test_verify_model_fails_on_tiny_onnx(tmp_path):
    name = next(iter(SUPPORTED_MODELS))
    d = tmp_path / name
    d.mkdir(parents=True)
    (d / "model.onnx").write_bytes(b"small")
    _valid_csv(d / "selected_tags.csv")
    mm = ModelManager(tmp_path)
    r = mm.verify_model(name, check_remote=False)
    assert r.ok is False
    assert any("onnx_too_small" in i for i in r.issues)


def test_verify_model_ok_with_manifest_and_sizes(monkeypatch, tmp_path):
    name = next(iter(SUPPORTED_MODELS))
    d = tmp_path / name
    d.mkdir(parents=True)
    onnx_sz = 2_000_000
    (d / "model.onnx").write_bytes(b"y" * onnx_sz)
    _valid_csv(d / "selected_tags.csv")
    mm = ModelManager(tmp_path)
    monkeypatch.setattr(
        "backend.services.model_manager.fetch_repo_head_sha",
        lambda repo, revision="main": "deadbeef",
    )
    mm._write_cache_manifest(name, SUPPORTED_MODELS[name])
    r = mm.verify_model(name, check_remote=False)
    assert r.ok is True
    assert r.manifest_present is True
    assert r.local_revision == "deadbeef"


def test_verify_model_stale_on_hub(monkeypatch, tmp_path):
    name = next(iter(SUPPORTED_MODELS))
    d = tmp_path / name
    d.mkdir(parents=True)
    (d / "model.onnx").write_bytes(b"z" * 2_000_000)
    _valid_csv(d / "selected_tags.csv")
    mm = ModelManager(tmp_path)
    manifest = {
        "schema": 1,
        "files": {
            "model.onnx": {"size": 2_000_000},
            "selected_tags.csv": {"size": (d / "selected_tags.csv").stat().st_size},
        },
        "revision_sha": "aaa",
    }
    mm._manifest_path(name).write_text(__import__("json").dumps(manifest))
    monkeypatch.setattr(
        "backend.services.model_manager.fetch_repo_head_sha",
        lambda repo, revision="main": "bbb",
    )
    r = mm.verify_model(name, check_remote=True)
    assert r.ok is True
    assert r.stale_on_hub is True
    assert "newer_revision_on_hub" in r.issues


def test_download_model_skips_existing_files(monkeypatch, tmp_path):
    name = next(iter(SUPPORTED_MODELS))
    d = tmp_path / name
    d.mkdir(parents=True)
    (d / "model.onnx").write_bytes(b"a" * 2_000_000)
    _valid_csv(d / "selected_tags.csv")
    calls = []

    def fake_hf(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("should not re-download when files exist")

    monkeypatch.setattr("backend.services.model_manager.hf_hub_download", fake_hf)
    monkeypatch.setattr(
        "backend.services.model_manager.fetch_repo_head_sha",
        lambda repo, revision="main": "sha1",
    )
    mm = ModelManager(tmp_path)
    mm.download_model(name)
    assert not calls
    assert mm.read_manifest(name) is not None


def test_repair_manifest_if_missing(monkeypatch, tmp_path):
    name = next(iter(SUPPORTED_MODELS))
    d = tmp_path / name
    d.mkdir(parents=True)
    (d / "model.onnx").write_bytes(b"b" * 2_000_000)
    _valid_csv(d / "selected_tags.csv")
    monkeypatch.setattr(
        "backend.services.model_manager.fetch_repo_head_sha",
        lambda repo, revision="main": "cafef00d",
    )
    mm = ModelManager(tmp_path)
    assert mm.read_manifest(name) is None
    assert mm.repair_manifest_if_missing(name) is True
    assert mm.read_manifest(name)["revision_sha"] == "cafef00d"

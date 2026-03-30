"""Model cache verification API and list_models cache fields."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import backend.config as config_module
import backend.routes.tagger as tagger_routes
from backend.services.model_manager import SUPPORTED_MODELS


def _write_min_valid_cache(model_dir: Path, *, onnx_bytes: int = 2_000_000, csv_rows: int = 101) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.onnx").write_bytes(b"x" * onnx_bytes)
    lines = ["tag_id,name,category,count"]
    for i in range(csv_rows):
        lines.append(f"{i},t{i},0,1")
    (model_dir / "selected_tags.csv").write_text("\n".join(lines), encoding="utf-8")


def test_post_models_verify_all_local_ok(test_config, monkeypatch):
    monkeypatch.setattr(tagger_routes, "get_config", lambda: test_config)
    name = next(iter(SUPPORTED_MODELS))
    _write_min_valid_cache(Path(test_config.models_dir) / name)

    from backend.app import app

    with TestClient(app) as client:
        r = client.post("/api/tagger/models/verify", json={"check_remote": False})
    assert r.status_code == 200
    data = r.json()
    assert data["success"] is True
    assert data["check_remote"] is False
    assert len(data["results"]) == len(SUPPORTED_MODELS)
    by_name = {x["name"]: x for x in data["results"]}
    assert by_name[name]["ok"] is True
    assert by_name[name]["issues"] == []


def test_post_models_verify_unknown_model(test_config, monkeypatch):
    monkeypatch.setattr(tagger_routes, "get_config", lambda: test_config)
    from backend.app import app

    with TestClient(app) as client:
        r = client.post(
            "/api/tagger/models/verify",
            json={"model_name": "not-a-model", "check_remote": False},
        )
    assert r.status_code == 200
    assert r.json()["success"] is False


def test_get_models_includes_cache_fields(test_config, monkeypatch):
    monkeypatch.setattr(tagger_routes, "get_config", lambda: test_config)
    name = next(iter(SUPPORTED_MODELS))
    _write_min_valid_cache(Path(test_config.models_dir) / name)

    from backend.app import app

    with TestClient(app) as client:
        r = client.get("/api/tagger/models")
    assert r.status_code == 200
    data = r.json()
    assert data["success"] is True
    m0 = next(m for m in data["models"] if m["name"] == name)
    assert m0["downloaded"] is True
    assert m0["cache_ok"] is True
    assert m0["manifest_present"] is False


@pytest.mark.parametrize("check_remote", [False, True])
def test_post_models_verify_with_remote_monkeypatched(test_config, monkeypatch, check_remote):
    monkeypatch.setattr(tagger_routes, "get_config", lambda: test_config)
    name = next(iter(SUPPORTED_MODELS))
    _write_min_valid_cache(Path(test_config.models_dir) / name)

    import backend.services.model_manager as mm

    monkeypatch.setattr(mm, "fetch_repo_head_sha", lambda repo, revision="main": "abc123deadbeef")

    from backend.app import app

    with TestClient(app) as client:
        r = client.post("/api/tagger/models/verify", json={"check_remote": check_remote})
    assert r.status_code == 200
    data = r.json()
    assert data["success"] is True
    by_name = {x["name"]: x for x in data["results"]}
    row = by_name[name]
    if check_remote:
        assert row["remote_revision_sha"] == "abc123deadbeef"
    else:
        assert row.get("remote_revision_sha") in (None, "")

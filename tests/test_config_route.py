"""HTTP config API."""

import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]

from fastapi.testclient import TestClient

import backend.config as config_module
import backend.routes.config_routes as config_routes
from backend.app import app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(config_routes, "get_config", config_module.get_config)

    def _save(c):
        config_module._config = c

    # Routes bind save_config at import time; patching only backend.config would still write disk.
    monkeypatch.setattr(config_module, "save_config", _save)
    monkeypatch.setattr(config_routes, "save_config", _save)
    return TestClient(app)


def test_patch_config_validates_batch_size(client):
    r = client.patch("/api/config", json={"batch_size": 9999})
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is False
    assert "error" in body


def test_patch_config_updates_allowed_field(client):
    r = client.patch("/api/config", json={"batch_size": 6, "apply_tags_every_n": 4})
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert "batch_size" in body["updated"]
    r2 = client.get("/api/config")
    cfg = r2.json()["config"]
    assert cfg["batch_size"] == 6
    assert cfg["apply_tags_every_n"] == 4


def test_patch_max_learning_cached_files(client):
    r = client.patch("/api/config", json={"max_learning_cached_files": 100_000})
    assert r.status_code == 200
    assert r.json()["success"] is True
    cfg = client.get("/api/config").json()["config"]
    assert cfg["max_learning_cached_files"] == 100_000


def test_patch_hydrus_metadata_chunk_size(client):
    r = client.patch("/api/config", json={"hydrus_metadata_chunk_size": 512})
    assert r.status_code == 200
    assert r.json()["success"] is True
    cfg = client.get("/api/config").json()["config"]
    assert cfg["hydrus_metadata_chunk_size"] == 512


def test_patch_tagging_skip_tail_batch_size(client):
    r = client.patch("/api/config", json={"tagging_skip_tail_batch_size": 1024})
    assert r.status_code == 200
    assert r.json()["success"] is True
    assert "tagging_skip_tail_batch_size" in r.json()["updated"]
    cfg = client.get("/api/config").json()["config"]
    assert cfg["tagging_skip_tail_batch_size"] == 1024


def test_patch_ort_profiling_settings(client):
    r = client.patch(
        "/api/config",
        json={
            "ort_enable_profiling": True,
            "ort_profile_dir": "./ort_traces",
        },
    )
    assert r.status_code == 200
    assert r.json()["success"] is True
    cfg = client.get("/api/config").json()["config"]
    assert cfg["ort_enable_profiling"] is True
    assert cfg["ort_profile_dir"] == "./ort_traces"


def test_patch_config_default_model_and_wd_markers(client):
    r = client.patch(
        "/api/config",
        json={
            "default_model": "wd-vit-large-tagger-v3",
            "target_tag_service": "local tags",
            "wd_skip_inference_if_marker_present": False,
            "wd_skip_if_higher_tier_model_present": False,
            "wd_append_model_marker_tag": True,
            "wd_model_marker_template": "",
            "wd_model_marker_prefix": "wd14:",
            "apply_tags_http_batch_size": 50,
            "allow_ui_shutdown": False,
            "shutdown_tagging_grace_seconds": 2.5,
        },
    )
    assert r.status_code == 200
    assert r.json()["success"] is True
    cfg = client.get("/api/config").json()["config"]
    assert cfg["default_model"] == "wd-vit-large-tagger-v3"
    assert cfg["target_tag_service"] == "local tags"
    assert cfg["wd_skip_inference_if_marker_present"] is False
    assert cfg["wd_skip_if_higher_tier_model_present"] is False
    assert cfg["apply_tags_http_batch_size"] == 50
    assert cfg["allow_ui_shutdown"] is False
    assert cfg["shutdown_tagging_grace_seconds"] == 2.5

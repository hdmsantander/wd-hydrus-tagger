"""FastAPI dependency helpers."""

import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]

import backend.dependencies as deps
from backend.config import AppConfig


def test_get_app_config_returns_get_config(monkeypatch):
    cfg = AppConfig(hydrus_api_url="http://x", hydrus_api_key="k")
    monkeypatch.setattr(deps, "get_config", lambda: cfg)
    assert deps.get_app_config() is cfg


def test_get_hydrus_client_reuses_instance_for_same_credentials(monkeypatch):
    cfg = AppConfig(hydrus_api_url="http://h.test", hydrus_api_key="secret")
    monkeypatch.setattr(deps, "get_config", lambda: cfg)
    deps._hydrus_client = None
    c1 = deps.get_hydrus_client()
    c2 = deps.get_hydrus_client()
    assert c1 is c2
    assert c1.api_url == "http://h.test"
    deps._hydrus_client = None


def test_get_hydrus_client_rebuilds_when_key_changes(monkeypatch):
    state = {"cfg": AppConfig(hydrus_api_url="http://h.test", hydrus_api_key="a")}

    def _gc():
        return state["cfg"]

    monkeypatch.setattr(deps, "get_config", _gc)
    deps._hydrus_client = None
    first = deps.get_hydrus_client()
    state["cfg"] = AppConfig(hydrus_api_url="http://h.test", hydrus_api_key="b")
    second = deps.get_hydrus_client()
    assert first is not second
    deps._hydrus_client = None

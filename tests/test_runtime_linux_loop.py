"""Uvicorn loop selection on Linux vs other platforms."""

import sys
import types

import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]


def test_uvicorn_loop_setting_non_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    from backend.runtime_linux import uvicorn_loop_setting

    assert uvicorn_loop_setting() == "auto"


def test_uvicorn_loop_setting_linux_no_uvloop(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("PYTHONPATH", raising=False)
    sys.modules.pop("uvloop", None)
    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "uvloop":
            raise ImportError("test: no uvloop")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from backend.runtime_linux import uvicorn_loop_setting

    assert uvicorn_loop_setting() == "auto"


def test_uvicorn_loop_setting_linux_with_uvloop(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    # Satisfy ``import uvloop`` without optional dependency installed
    monkeypatch.setitem(sys.modules, "uvloop", types.ModuleType("uvloop"))
    from backend.runtime_linux import uvicorn_loop_setting

    assert uvicorn_loop_setting() == "uvloop"

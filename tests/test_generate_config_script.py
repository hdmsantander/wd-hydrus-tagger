"""Unit tests for scripts/generate_config.py (Linux tuning helpers)."""

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "generate_config.py"


def _load_generate_config():
    spec = importlib.util.spec_from_file_location("generate_config_wizard", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gc_mod():
    return _load_generate_config()


def test_suggest_batch_and_parallel_large_model_low_ram(gc_mod):
    assert gc_mod._suggest_batch_and_parallel(10.0, True) == (2, 4)
    assert gc_mod._suggest_batch_and_parallel(20.0, True) == (4, 4)
    assert gc_mod._suggest_batch_and_parallel(32.0, True) == (4, 6)


def test_suggest_batch_and_parallel_vit_and_unknown_mem(gc_mod):
    assert gc_mod._suggest_batch_and_parallel(4.0, False) == (4, 4)
    assert gc_mod._suggest_batch_and_parallel(16.0, False) == (8, 8)
    assert gc_mod._suggest_batch_and_parallel(None, False) == (8, 8)
    assert gc_mod._suggest_batch_and_parallel(None, True) == (6, 6)


def test_parse_physical_cores_two_cores_one_socket(gc_mod):
    text = """
processor\t: 0
physical id\t: 0
core id\t: 0
cpu cores\t: 2

processor\t: 1
physical id\t: 0
core id\t: 1
cpu cores\t: 2
"""
    assert gc_mod._parse_physical_cores_from_cpuinfo_text(text) == 2


def test_parse_physical_cores_smt_duplicate_pairs(gc_mod):
    """Same (physical id, core id) from two logical CPUs → one physical core."""
    text = """
processor\t: 0
physical id\t: 0
core id\t: 0

processor\t: 1
physical id\t: 0
core id\t: 0
"""
    assert gc_mod._parse_physical_cores_from_cpuinfo_text(text) == 1


def test_parse_physical_cores_fallback_cpu_cores_times_sockets(gc_mod):
    text = """
processor\t: 0
cpu cores\t: 8
physical id\t: 0

processor\t: 1
cpu cores\t: 8
physical id\t: 1
"""
    n = gc_mod._parse_physical_cores_from_cpuinfo_text(text)
    assert n == 16


def test_parse_physical_cores_empty_returns_none(gc_mod):
    assert gc_mod._parse_physical_cores_from_cpuinfo_text("") is None


def test_main_non_linux_exits_2(gc_mod, monkeypatch, tmp_path):
    monkeypatch.setattr(gc_mod.sys, "platform", "darwin")
    monkeypatch.setattr(gc_mod.sys, "argv", ["generate_config", "-f", "-o", str(tmp_path / "out.yaml")])
    assert gc_mod.main() == 2


def test_main_linux_missing_example(gc_mod, monkeypatch, tmp_path):
    monkeypatch.setattr(gc_mod.sys, "platform", "linux")
    root = tmp_path / "empty"
    root.mkdir()
    monkeypatch.setattr(gc_mod, "_repo_root", lambda: root)
    monkeypatch.setattr(gc_mod.sys, "argv", ["generate_config", "-f", "-o", str(tmp_path / "c.yaml")])
    assert gc_mod.main() == 1

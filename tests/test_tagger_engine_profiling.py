"""TaggerEngine ONNX Runtime profiling hooks (Tier D)."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]

from backend.tagger.engine import TaggerEngine
from backend.tagger.labels import LabelData


@pytest.fixture
def minimal_model_dir(tmp_path: Path) -> Path:
    m = tmp_path / "wd-vit-tagger-v3"
    m.mkdir(parents=True)
    (m / "model.onnx").write_bytes(b"")
    (m / "selected_tags.csv").write_text("tag_id,name,category,count\n0,tag0,0,1\n", encoding="utf-8")
    return tmp_path


def test_tagger_engine_load_sets_profiling_on_session_options(minimal_model_dir, monkeypatch):
    import onnxruntime as ort

    captured: dict = {}

    class FakeSession:
        def __init__(self, path, sess_options=None, providers=None):
            captured["enable_profiling"] = bool(sess_options.enable_profiling)
            captured["profile_file_prefix"] = getattr(sess_options, "profile_file_prefix", "")

        def get_inputs(self):
            inp = MagicMock()
            inp.shape = [None, 448, 448, 3]
            return [inp]

        def get_outputs(self):
            out = MagicMock()
            out.name = "out"
            return [out]

    ld = LabelData(names=["tag0"], general_indices=[0], character_indices=[], rating_indices=[])

    monkeypatch.setattr(ort, "InferenceSession", FakeSession)
    eng = TaggerEngine(use_gpu=False)
    eng.load(
        minimal_model_dir,
        "wd-vit-tagger-v3",
        enable_profiling=True,
        profile_file_prefix="/tmp/wd_test_prefix",
    )
    assert captured["enable_profiling"] is True
    assert captured["profile_file_prefix"] == "/tmp/wd_test_prefix"
    assert eng._profiling_active is True


def test_tagger_engine_finalize_calls_end_profiling(minimal_model_dir, monkeypatch):
    import onnxruntime as ort

    ended: list[bool] = []

    class FakeSession:
        def __init__(self, path, sess_options=None, providers=None):
            pass

        def end_profiling(self):
            ended.append(True)
            return "/tmp/fake_profile.json"

        def get_inputs(self):
            inp = MagicMock()
            inp.shape = [None, 448, 448, 3]
            return [inp]

        def get_outputs(self):
            out = MagicMock()
            out.name = "out"
            return [out]

    monkeypatch.setattr(ort, "InferenceSession", FakeSession)
    eng = TaggerEngine(use_gpu=False)
    eng.load(minimal_model_dir, "wd-vit-tagger-v3", enable_profiling=True)
    path = eng.finalize_ort_profiling()
    assert path == "/tmp/fake_profile.json"
    assert ended == [True]
    assert eng.session is None
    assert eng._profiling_active is False

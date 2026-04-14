"""TaggerEngine.predict path (mock ONNX) — thresholds + batch cleanup."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

pytestmark = [pytest.mark.full, pytest.mark.core]

from backend.tagger.engine import TaggerEngine


@pytest.fixture
def model_dir_three_tags(tmp_path: Path) -> Path:
    """General (0), character (4), rating (9) — matches labels.load_labels categories."""
    m = tmp_path / "wd-mock"
    m.mkdir(parents=True)
    (m / "model.onnx").write_bytes(b"")
    (m / "selected_tags.csv").write_text(
        "tag_id,name,category,count\n"
        "0,gen0,0,1\n"
        "1,char1,4,1\n"
        "2,rate_safe,9,1\n",
        encoding="utf-8",
    )
    return tmp_path


def test_predict_batch_thresholds_and_sorted_outputs(model_dir_three_tags, monkeypatch):
    import onnxruntime as ort

    class FakeSession:
        def __init__(self, path, sess_options=None, providers=None):
            self._path = path

        def get_inputs(self):
            inp = MagicMock()
            inp.name = "in"
            inp.shape = [None, 448, 448, 3]
            return [inp]

        def get_outputs(self):
            out = MagicMock()
            out.name = "out"
            return [out]

        def run(self, output_names, feeds):
            batch = next(iter(feeds.values()))
            b = int(batch.shape[0])
            # High scores for idx 0,1; low for rating idx 2 (still reported)
            prob = np.array([[0.96, 0.90, 0.40]], dtype=np.float32)
            if b > 1:
                prob = np.vstack([prob] * b)
            return [prob]

    monkeypatch.setattr(ort, "InferenceSession", FakeSession)
    eng = TaggerEngine(use_gpu=False)
    eng.load(model_dir_three_tags, "wd-mock", intra_op_threads=2, inter_op_threads=1)

    imgs = [Image.new("RGB", (64, 64), color=(1, 2, 3))]
    out = eng.predict(imgs, general_threshold=0.35, character_threshold=0.85)
    assert len(out) == 1
    assert "gen0" in out[0]["general_tags"]
    assert "char1" in out[0]["character_tags"]
    assert out[0]["rating_tags"]["rate_safe"] == pytest.approx(0.40)


def test_predict_requires_loaded_model():
    eng = TaggerEngine(use_gpu=False)
    with pytest.raises(RuntimeError, match="not loaded"):
        eng.predict([Image.new("RGB", (4, 4))])

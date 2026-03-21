"""ONNX-based WD v3 tagger inference engine."""

from pathlib import Path

import numpy as np
from PIL import Image

from backend.tagger.labels import LabelData, load_labels
from backend.tagger.preprocess import preprocess_batch



class TaggerEngine:
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self.session = None
        self.labels: LabelData | None = None
        self.model_name: str | None = None
        self.target_size: int = 448

    def load(self, model_dir: Path, model_name: str) -> None:
        """Load an ONNX model and its labels."""
        import onnxruntime as ort

        model_path = model_dir / model_name / "model.onnx"
        csv_path = model_dir / model_name / "selected_tags.csv"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not csv_path.exists():
            raise FileNotFoundError(f"Labels not found: {csv_path}")

        providers = []
        if self.use_gpu:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.labels = load_labels(csv_path)
        self.model_name = model_name

        # Detect input size from model
        input_shape = self.session.get_inputs()[0].shape
        if len(input_shape) == 4:
            self.target_size = input_shape[1]  # NHWC: (N, H, W, C)

    def predict(
        self,
        images: list[Image.Image],
        general_threshold: float = 0.35,
        character_threshold: float = 0.85,
    ) -> list[dict]:
        """Run inference on a batch of images.

        Returns list of dicts with keys: general_tags, character_tags, rating_tags.
        Each is a dict of {tag_name: confidence}.
        """
        if self.session is None or self.labels is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        batch = preprocess_batch(images, self.target_size)
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name

        raw_output = self.session.run([output_name], {input_name: batch})[0]
        # WD v3 ONNX models apply sigmoid internally — output is already probabilities [0, 1]
        probs = raw_output

        results = []
        for i in range(len(images)):
            p = probs[i]

            general_tags = {}
            for idx in self.labels.general_indices:
                if p[idx] >= general_threshold:
                    general_tags[self.labels.names[idx]] = float(p[idx])

            character_tags = {}
            for idx in self.labels.character_indices:
                if p[idx] >= character_threshold:
                    character_tags[self.labels.names[idx]] = float(p[idx])

            rating_tags = {}
            for idx in self.labels.rating_indices:
                rating_tags[self.labels.names[idx]] = float(p[idx])

            results.append({
                "general_tags": dict(sorted(general_tags.items(), key=lambda x: -x[1])),
                "character_tags": dict(sorted(character_tags.items(), key=lambda x: -x[1])),
                "rating_tags": dict(sorted(rating_tags.items(), key=lambda x: -x[1])),
            })

        return results

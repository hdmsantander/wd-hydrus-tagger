"""ONNX-based WD v3 tagger inference engine."""

import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image

from backend.tagger.labels import LabelData, load_labels
from backend.tagger.preprocess import preprocess_batch

log = logging.getLogger(__name__)



class TaggerEngine:
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self.session = None
        self.labels: LabelData | None = None
        self.model_name: str | None = None
        self.target_size: int = 448

    def load(
        self,
        model_dir: Path,
        model_name: str,
        *,
        intra_op_threads: int = 8,
        inter_op_threads: int = 1,
    ) -> None:
        """Load an ONNX model and its labels."""
        import onnxruntime as ort

        model_path = model_dir / model_name / "model.onnx"
        csv_path = model_dir / model_name / "selected_tags.csv"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not csv_path.exists():
            raise FileNotFoundError(f"Labels not found: {csv_path}")

        old = self.session
        self.session = None
        if old is not None:
            del old

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = intra_op_threads
        sess_opts.inter_op_num_threads = inter_op_threads
        # Single-stream CPU graphs: sequential + intra_op threads avoids extra scheduler work
        # when inter_op_num_threads is 1 (typical WD batch inference on Linux/CPU).
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # Defaults are usually True; explicit for CPU throughput on multi-GB models (32 GB RAM class).
        sess_opts.enable_mem_pattern = True
        sess_opts.enable_cpu_mem_arena = True

        providers = []
        if self.use_gpu:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        log.info(
            "TaggerEngine loading ONNX model=%s providers=%s path=%s",
            model_name,
            providers,
            model_path,
        )
        t_sess = time.perf_counter()
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_opts,
            providers=providers,
        )
        sess_s = time.perf_counter() - t_sess
        self.labels = load_labels(csv_path)
        self.model_name = model_name
        log.info(
            "TaggerEngine metrics model=%s session_init_wall_s=%.3f labels=%s "
            "threads_intra=%s threads_inter=%s",
            model_name,
            sess_s,
            len(self.labels.names),
            intra_op_threads,
            inter_op_threads,
        )

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

        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name

        log.debug(
            "TaggerEngine.predict batch=%s target_size=%s thresholds g=%s c=%s",
            len(images),
            self.target_size,
            general_threshold,
            character_threshold,
        )
        batch = None
        raw_output = None
        try:
            batch = preprocess_batch(images, self.target_size)
            # C-contiguous float32 helps ORT CPU EP avoid a copy on some Linux builds.
            batch = np.ascontiguousarray(batch, dtype=np.float32)
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
        finally:
            if batch is not None:
                del batch
            if raw_output is not None:
                del raw_output

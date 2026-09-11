"""InsightFace-based face detection and embedding engine."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from backend.face.load_control import is_load_generation_current
from backend.face.video_frames import pil_rgb_to_bgr
from backend.tagger.ort_providers import (
    active_gpu_provider_label,
    insightface_ctx_id,
    resolve_ort_providers,
)

log = logging.getLogger(__name__)

LOAD_STAGE_LABELS = {
    "idle": "idle",
    "inspecting_cache": "checking model cache",
    "downloading": "downloading model",
    "constructing": "building FaceAnalysis",
    "preparing": "preparing ONNX sessions",
    "ready": "ready",
}


class FaceEngine:
    """Wraps InsightFace ``buffalo_l`` (SCRFD + ArcFace) with multi-vendor GPU support."""

    def __init__(
        self,
        *,
        use_gpu: bool = False,
        gpu_backend: str = "auto",
        model_pack: str = "buffalo_l",
        det_threshold: float = 0.5,
        models_root: Path,
    ):
        self.use_gpu = use_gpu
        self.gpu_backend = gpu_backend
        self.model_pack = model_pack
        self.det_threshold = det_threshold
        self.models_root = models_root
        self._app: Any = None
        self._providers: list[str] = []
        self._ctx_id: int = -1
        self._loaded_with_cpu_fallback = False
        self._load_stage = "idle"

    @property
    def loaded(self) -> bool:
        return self._app is not None

    @property
    def load_stage(self) -> str:
        return self._load_stage

    @property
    def providers(self) -> list[str]:
        return list(self._providers)

    @property
    def active_provider(self) -> str:
        return active_gpu_provider_label(self._providers)

    @property
    def loaded_with_cpu_fallback(self) -> bool:
        return self._loaded_with_cpu_fallback

    def _check_generation(self, generation: int) -> bool:
        if not is_load_generation_current(generation):
            log.info(
                "FaceEngine load aborted (superseded) stage=%s generation=%s",
                self._load_stage,
                generation,
            )
            return False
        return True

    def _load_impl(self, *, force_cpu: bool = False, generation: int = 0) -> None:
        from insightface.app import FaceAnalysis

        if not self._check_generation(generation):
            return

        self.unload()
        self._loaded_with_cpu_fallback = False
        use_gpu = self.use_gpu and not force_cpu
        gpu_backend = "cpu" if force_cpu else self.gpu_backend
        self.models_root.mkdir(parents=True, exist_ok=True)
        self._providers = resolve_ort_providers(
            use_gpu=use_gpu,
            gpu_backend=gpu_backend,
        )
        self._ctx_id = insightface_ctx_id(self._providers)
        t0 = time.monotonic()
        log.info(
            "FaceEngine load start pack=%s root=%s use_gpu=%s gpu_backend=%s providers=%s ctx_id=%s generation=%s",
            self.model_pack,
            self.models_root,
            use_gpu,
            gpu_backend,
            self._providers,
            self._ctx_id,
            generation,
        )
        from backend.face.models import download_face_pack, inspect_face_pack

        self._load_stage = "inspecting_cache"
        if not self._check_generation(generation):
            return

        cache = inspect_face_pack(self.models_root, self.model_pack)
        if not cache["cache_ok"]:
            self._load_stage = "downloading"
            log.info(
                "FaceEngine cache miss pack=%s issues=%s; downloading",
                self.model_pack,
                cache["cache_issues"],
            )
            if not self._check_generation(generation):
                return
            download_face_pack(self.models_root, self.model_pack)
        log.info(
            "FaceEngine cache ok pack=%s elapsed_s=%.2f",
            self.model_pack,
            time.monotonic() - t0,
        )

        self._load_stage = "constructing"
        if not self._check_generation(generation):
            return
        t1 = time.monotonic()
        self._app = FaceAnalysis(
            name=self.model_pack,
            root=str(self.models_root),
            providers=self._providers,
        )
        log.info(
            "FaceEngine FaceAnalysis constructed elapsed_s=%.2f",
            time.monotonic() - t1,
        )

        self._load_stage = "preparing"
        if not self._check_generation(generation):
            self.unload()
            return
        t2 = time.monotonic()
        self._app.prepare(ctx_id=self._ctx_id, det_thresh=self.det_threshold)
        if not self._check_generation(generation):
            self.unload()
            return

        self._load_stage = "ready"
        if force_cpu and self.use_gpu:
            self._loaded_with_cpu_fallback = True
        log.info(
            "FaceEngine ready active_provider=%s ctx_id=%s prepare_s=%.2f total_s=%.2f cpu_fallback=%s",
            self.active_provider,
            self._ctx_id,
            time.monotonic() - t2,
            time.monotonic() - t0,
            self._loaded_with_cpu_fallback,
        )

    def run_load(self, *, force_cpu: bool, generation: int) -> None:
        """Load from the face worker thread; discards result when ``generation`` is stale."""
        try:
            self._load_impl(force_cpu=force_cpu, generation=generation)
        except Exception:
            if is_load_generation_current(generation):
                self.unload()
                raise
            log.info("FaceEngine load error ignored (superseded) generation=%s", generation)
            self.unload()

    def load(self) -> None:
        """Synchronous load for scripts/tests."""
        from backend.face.load_control import bump_load_generation

        generation = bump_load_generation()
        if self.use_gpu:
            try:
                self._load_impl(force_cpu=False, generation=generation)
                if self.loaded:
                    return
            except Exception:
                log.exception(
                    "FaceEngine GPU load failed (providers=%s); retrying on CPU",
                    self._providers,
                )
                self._loaded_with_cpu_fallback = True
                self.unload()
        self._load_impl(force_cpu=True, generation=generation)

    def unload(self) -> None:
        app = self._app
        self._app = None
        self._providers = []
        self._ctx_id = -1
        self._loaded_with_cpu_fallback = False
        self._load_stage = "idle"
        if app is not None:
            del app

    def detect_faces(self, image: Image.Image, *, conf: float | None = None) -> list[dict]:
        if self._app is None:
            raise RuntimeError("Face model not loaded. Call load() first.")
        threshold = self.det_threshold if conf is None else conf
        bgr = pil_rgb_to_bgr(image)
        raw_faces = self._app.get(bgr, max_num=0)
        out: list[dict] = []
        for face in raw_faces:
            score = float(face.det_score)
            if score < threshold:
                continue
            x1, y1, x2, y2 = face.bbox.astype(int)
            embedding = face.normed_embedding
            if embedding is None:
                continue
            out.append(
                {
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "score": score,
                    "landmarks": face.kps.astype(np.float32) if face.kps is not None else None,
                    "embedding": np.asarray(embedding, dtype=np.float32),
                }
            )
        return out


def load_stage_label(stage: str) -> str:
    return LOAD_STAGE_LABELS.get(stage, stage)

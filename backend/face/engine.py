"""InsightFace-based face detection and embedding engine."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from backend.face.video_frames import pil_rgb_to_bgr
from backend.tagger.ort_providers import (
    active_gpu_provider_label,
    insightface_ctx_id,
    resolve_ort_providers,
)

log = logging.getLogger(__name__)


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

    @property
    def loaded(self) -> bool:
        return self._app is not None

    @property
    def providers(self) -> list[str]:
        return list(self._providers)

    @property
    def active_provider(self) -> str:
        return active_gpu_provider_label(self._providers)

    def load(self) -> None:
        from insightface.app import FaceAnalysis

        self.unload()
        self.models_root.mkdir(parents=True, exist_ok=True)
        self._providers = resolve_ort_providers(
            use_gpu=self.use_gpu,
            gpu_backend=self.gpu_backend,
        )
        self._ctx_id = insightface_ctx_id(self._providers)
        log.info(
            "FaceEngine loading model_pack=%s root=%s providers=%s ctx_id=%s",
            self.model_pack,
            self.models_root,
            self._providers,
            self._ctx_id,
        )
        self._app = FaceAnalysis(
            name=self.model_pack,
            root=str(self.models_root),
            providers=self._providers,
        )
        self._app.prepare(ctx_id=self._ctx_id, det_thresh=self.det_threshold)

    def unload(self) -> None:
        self._app = None
        self._providers = []
        self._ctx_id = -1

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

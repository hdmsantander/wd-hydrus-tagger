"""Face detection and recognition orchestration."""

from __future__ import annotations

import asyncio
import logging
import time
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from backend.config import AppConfig
from backend.face.clustering import cluster_faces, staged_min_faces_list
from backend.face.embeddings_db import FaceEmbeddingsDB
from backend.face.flow_steps import (
    face_detect_progress,
    face_recognize_apply_label,
    face_recognize_step_label,
)
from backend.face.engine import FaceEngine, load_stage_label
from backend.face.load_control import (
    abandon_face_load_worker,
    bump_load_generation,
    cancel_pending_face_loads,
    is_load_generation_current,
    submit_face_load,
)
from backend.face.models import inspect_face_pack
from backend.hydrus.client import HydrusClient
from backend.hydrus.tag_merge import existing_storage_tag_keys, filter_new_tags
from backend.services.tagging_shared import load_metadata_by_file_id
from backend.tagger.ort_providers import available_ort_providers, resolve_ort_providers

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def resolved_face_db_path(raw: str) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (_REPO_ROOT / p).resolve()
    else:
        p = p.resolve()
    return p


def resolved_face_models_root(raw: str) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (_REPO_ROOT / p).resolve()
    else:
        p = p.resolve()
    return p


def _hydrus_mime(meta: dict | None) -> str:
    if not meta:
        return ""
    m = meta.get("mime")
    return m.strip().lower() if isinstance(m, str) else ""


def _is_video_mime(mime: str) -> bool:
    return bool(mime) and mime.startswith("video/")


def _decode_image(raw: bytes) -> Image.Image:
    im = Image.open(BytesIO(raw))
    im.load()
    return im


def _storage_has_marker(meta: dict | None, service_key: str, markers: set[str]) -> bool:
    if not meta or not service_key or not markers:
        return False
    existing = existing_storage_tag_keys(meta, service_key)
    normalized = {t.strip().lower() for t in markers if t.strip()}
    return any(t in normalized for t in existing)


def _analyze_detect_queue(
    file_ids: list[int],
    meta_by_id: dict[int, dict],
    *,
    service_key: str,
    skip_if_detected: bool,
    replace_existing: bool,
    marker_detected: str,
    marker_not_visible: str,
) -> dict:
    """Classify queued files before model load (images only; videos excluded)."""
    markers = {marker_detected, marker_not_visible}
    stats = {
        "total": len(file_ids),
        "images": 0,
        "videos": 0,
        "marker_skip": 0,
        "missing_hash": 0,
        "to_process": 0,
    }
    for fid in file_ids:
        meta = meta_by_id.get(fid)
        fhash = (meta or {}).get("hash") or ""
        if not fhash:
            stats["missing_hash"] += 1
            continue
        mime = _hydrus_mime(meta)
        if _is_video_mime(mime):
            stats["videos"] += 1
            continue
        stats["images"] += 1
        if skip_if_detected and not replace_existing and _storage_has_marker(meta, service_key, markers):
            stats["marker_skip"] += 1
            continue
        stats["to_process"] += 1
    return stats


def _queue_summary_detail(stats: dict) -> str:
    parts = [f"{stats['to_process']} image(s) to scan"]
    if stats["videos"]:
        parts.append(f"{stats['videos']} video(s) skipped")
    if stats["marker_skip"]:
        parts.append(f"{stats['marker_skip']} already tagged")
    if stats["missing_hash"]:
        parts.append(f"{stats['missing_hash']} missing hash")
    return "Queue: " + " · ".join(parts)


class FaceTaggingService:
    _instance: "FaceTaggingService | None" = None

    @staticmethod
    def _engine_from_config(config: AppConfig) -> FaceEngine:
        return FaceEngine(
            use_gpu=config.use_gpu,
            gpu_backend=config.gpu_backend,
            model_pack=config.face_model_pack,
            det_threshold=config.face_det_threshold,
            models_root=resolved_face_models_root(config.face_models_dir),
        )

    def __init__(self, config: AppConfig):
        self.config = config
        self.engine = self._engine_from_config(config)
        self.db = FaceEmbeddingsDB(resolved_face_db_path(config.face_embeddings_db_path))
        self.db.init()
        self._warmup_done = False

    @classmethod
    def get_instance(cls, config: AppConfig) -> "FaceTaggingService":
        if config is None:
            raise RuntimeError("AppConfig is not loaded")
        if cls._instance is None:
            cls._instance = cls(config)
            return cls._instance

        prev = cls._instance.config
        reload_engine = (
            prev.use_gpu != config.use_gpu
            or prev.gpu_backend != config.gpu_backend
            or prev.face_model_pack != config.face_model_pack
            or prev.face_det_threshold != config.face_det_threshold
            or prev.face_models_dir != config.face_models_dir
        )
        if reload_engine:
            cancel_pending_face_loads(reason="engine_config_change")
            cls._instance.engine.unload()
            cls._instance.engine = cls._engine_from_config(config)
            cls._instance._warmup_done = False

        if prev.face_embeddings_db_path != config.face_embeddings_db_path:
            cls._instance.db = FaceEmbeddingsDB(resolved_face_db_path(config.face_embeddings_db_path))
            cls._instance.db.init()

        cls._instance.config = config
        return cls._instance

    @classmethod
    def unload_model_from_memory(cls) -> None:
        cancel_pending_face_loads(reason="unload_model_from_memory")
        if cls._instance is not None:
            cfg = cls._instance.config
            cls._instance.engine.unload()
            cls._instance.engine = cls._engine_from_config(cfg)
            cls._instance._warmup_done = False

    def face_tag_service_name(self) -> str:
        name = (self.config.face_target_tag_service or "").strip()
        return name or self.config.target_tag_service

    async def _run_face_worker(
        self,
        fn,
        *,
        timeout_s: float | None,
        cancel_event: asyncio.Event | None = None,
        progress_cb=None,
        progress_detail: str = "",
        total: int = 0,
        processed: int = 0,
        phase: str = "detect",
        step: str | None = None,
        log_label: str = "face worker",
    ):
        """Run a blocking InsightFace call on the dedicated worker with timeout + cancel."""
        t0 = time.monotonic()
        detect_step = step or ("warmup" if phase == "model" else "scan")

        async def _emit_worker_progress() -> None:
            if progress_cb and progress_detail:
                elapsed = int(time.monotonic() - t0)
                detail = f"{progress_detail} ({elapsed}s)"
                await progress_cb(
                    face_detect_progress(
                        detect_step,
                        detail=detail,
                        processed=processed,
                        total=total,
                        phase=phase,
                        face_count=0,
                        skipped=False,
                        active_provider=self.engine.active_provider if self.engine.loaded else None,
                    )
                )

        cfut = submit_face_load(fn)
        worker_future = asyncio.wrap_future(cfut)
        last_log_s = -15
        await _emit_worker_progress()
        while not worker_future.done():
            if cancel_event is not None and cancel_event.is_set():
                cancel_pending_face_loads(reason="cancel_event")
                raise asyncio.CancelledError(f"{log_label} cancelled")
            await _emit_worker_progress()
            elapsed = int(time.monotonic() - t0)
            if elapsed >= last_log_s + 15:
                log.info("%s still running elapsed_s=%s provider=%s", log_label, elapsed, self.engine.active_provider)
                last_log_s = elapsed
            if timeout_s is not None and (time.monotonic() - t0) >= timeout_s:
                abandon_face_load_worker(reason=f"{log_label}_timeout")
                raise TimeoutError(f"{log_label} timed out after {timeout_s:.0f}s")
            try:
                await asyncio.wait_for(asyncio.shield(worker_future), timeout=2.0)
            except asyncio.TimeoutError:
                continue
        if cancel_event is not None and cancel_event.is_set():
            cancel_pending_face_loads(reason="cancel_event")
            raise asyncio.CancelledError(f"{log_label} cancelled")
        return await worker_future

    async def _reload_engine_on_cpu(
        self,
        *,
        progress_cb=None,
        total: int = 0,
        cancel_event: asyncio.Event | None = None,
    ) -> None:
        cfg = self.config
        cancel_pending_face_loads(reason="cpu_reload")
        self.engine.unload()
        self._warmup_done = False
        generation = bump_load_generation()
        cfut = submit_face_load(
            lambda: self.engine.run_load(force_cpu=True, generation=generation),
        )
        load_future = asyncio.wrap_future(cfut)
        t0 = time.monotonic()
        while not load_future.done():
            if cancel_event is not None and cancel_event.is_set():
                cancel_pending_face_loads(reason="cancel_event")
                self.engine.unload()
                raise asyncio.CancelledError("face CPU reload cancelled")
            if progress_cb:
                elapsed = int(time.monotonic() - t0)
                await progress_cb(
                    face_detect_progress(
                        "model_load",
                        detail=f"Reloading InsightFace on CPU ({elapsed}s)…",
                        total=total,
                    )
                )
            try:
                await asyncio.wait_for(asyncio.shield(load_future), timeout=2.0)
            except asyncio.TimeoutError:
                continue
        await load_future
        if not self.engine.loaded:
            raise RuntimeError("Face CPU reload finished without a loaded engine")

    async def _warmup_inference(
        self,
        *,
        progress_cb=None,
        total: int = 0,
        cancel_event: asyncio.Event | None = None,
    ) -> None:
        if self._warmup_done:
            return
        cfg = self.config
        dummy = Image.new("RGB", (64, 64), (128, 128, 128))
        gpu_path = cfg.use_gpu and not self.engine.loaded_with_cpu_fallback
        timeout = (
            cfg.face_inference_timeout_seconds
            if gpu_path
            else min(60.0, cfg.face_inference_timeout_seconds)
        )
        log.info(
            "face model warmup start provider=%s timeout_s=%s",
            self.engine.active_provider,
            timeout,
        )
        try:
            await self._run_face_worker(
                lambda: self.engine.detect_faces(dummy, conf=0.99),
                timeout_s=timeout,
                cancel_event=cancel_event,
                progress_cb=progress_cb,
                progress_detail="Warming up face inference (first GPU run may compile kernels)",
                total=total,
                phase="model",
                step="warmup",
                log_label="face warmup",
            )
        except TimeoutError:
            if gpu_path:
                log.warning(
                    "face GPU warmup timed out after %.0fs; reloading on CPU",
                    timeout,
                )
                if progress_cb:
                    await progress_cb(
                        face_detect_progress(
                            "warmup",
                            detail="GPU warmup timed out; reloading on CPU…",
                            total=total,
                        )
                    )
                await self._reload_engine_on_cpu(
                    progress_cb=progress_cb,
                    total=total,
                    cancel_event=cancel_event,
                )
                await self._run_face_worker(
                    lambda: self.engine.detect_faces(dummy, conf=0.99),
                    timeout_s=min(60.0, cfg.face_inference_timeout_seconds),
                    cancel_event=cancel_event,
                    progress_cb=progress_cb,
                    progress_detail="Warming up face inference on CPU",
                    total=total,
                    phase="model",
                    log_label="face warmup_cpu",
                )
            else:
                raise RuntimeError(f"Face inference warmup timed out after {timeout:.0f}s on CPU")
        except Exception:
            if gpu_path:
                log.exception("face GPU warmup failed; reloading on CPU")
                await self._reload_engine_on_cpu(
                    progress_cb=progress_cb,
                    total=total,
                    cancel_event=cancel_event,
                )
                await self._run_face_worker(
                    lambda: self.engine.detect_faces(dummy, conf=0.99),
                    timeout_s=min(60.0, cfg.face_inference_timeout_seconds),
                    cancel_event=cancel_event,
                    log_label="face warmup_cpu",
                )
            else:
                raise
        self._warmup_done = True
        log.info("face model warmup ok provider=%s", self.engine.active_provider)

    async def ensure_model_loaded(
        self,
        *,
        progress_cb=None,
        total: int = 0,
        cancel_event: asyncio.Event | None = None,
    ) -> None:
        if self.engine.loaded and self._warmup_done:
            return
        cfg = self.config
        if not self.engine.loaded:
            planned = resolve_ort_providers(use_gpu=cfg.use_gpu, gpu_backend=cfg.gpu_backend)
            installed = available_ort_providers()
            log.info(
                "face model load requested use_gpu=%s gpu_backend=%s timeout_s=%s planned_providers=%s installed_ort=%s",
                cfg.use_gpu,
                cfg.gpu_backend,
                cfg.face_model_load_timeout_seconds,
                planned,
                installed,
            )
            t0 = time.monotonic()
            migraphx_hint = cfg.use_gpu and "MIGraphXExecutionProvider" in planned
            last_log_s = -15

            async def _emit_progress(extra: str = "") -> None:
                elapsed = int(time.monotonic() - t0)
                stage = load_stage_label(self.engine.load_stage)
                detail = f"Loading InsightFace · {stage} ({elapsed}s)"
                if extra:
                    detail += f" — {extra}"
                elif migraphx_hint and self.engine.load_stage == "preparing" and elapsed >= 5:
                    detail += " — MIGraphX first compile can take several minutes"
                if progress_cb:
                    await progress_cb(
                        face_detect_progress(
                            "model_load",
                            detail=detail,
                            total=total,
                            load_stage=self.engine.load_stage,
                            providers=planned,
                            active_provider=self.engine.active_provider if self.engine.loaded else None,
                        )
                    )

            async def _wait_future(load_future: asyncio.Future, *, timeout_s: float | None) -> str:
                nonlocal last_log_s
                while not load_future.done():
                    if cancel_event is not None and cancel_event.is_set():
                        cancel_pending_face_loads(reason="cancel_event")
                        self.engine.unload()
                        raise asyncio.CancelledError("face model load cancelled")
                    await _emit_progress()
                    elapsed = int(time.monotonic() - t0)
                    if elapsed >= last_log_s + 15:
                        log.info(
                            "face model load still running stage=%s elapsed_s=%s providers=%s",
                            self.engine.load_stage,
                            elapsed,
                            planned,
                        )
                        last_log_s = elapsed
                    if timeout_s is not None and (time.monotonic() - t0) >= timeout_s:
                        return "timeout"
                    try:
                        await asyncio.wait_for(asyncio.shield(load_future), timeout=2.0)
                    except asyncio.TimeoutError:
                        continue
                if cancel_event is not None and cancel_event.is_set():
                    cancel_pending_face_loads(reason="cancel_event")
                    self.engine.unload()
                    raise asyncio.CancelledError("face model load cancelled")
                await load_future
                return "ok"

            async def _run_load(force_cpu: bool) -> str:
                generation = bump_load_generation()
                cfut = submit_face_load(
                    lambda: self.engine.run_load(force_cpu=force_cpu, generation=generation),
                )
                load_future = asyncio.wrap_future(cfut)
                timeout = cfg.face_model_load_timeout_seconds if cfg.use_gpu and not force_cpu else None
                status = await _wait_future(load_future, timeout_s=timeout)
                if status == "timeout":
                    abandon_face_load_worker(reason="gpu_load_timeout")
                    self.engine.unload()
                    return "timeout"
                if not self.engine.loaded and is_load_generation_current(generation):
                    raise RuntimeError("Face model load finished without a loaded engine")
                return status

            if cfg.use_gpu:
                try:
                    result = await _run_load(force_cpu=False)
                except Exception:
                    log.exception("face GPU load failed; retrying on CPU")
                    cancel_pending_face_loads(reason="gpu_load_exception")
                    self.engine.unload()
                    result = "error"
                if result == "timeout":
                    log.warning(
                        "face GPU load timed out after %.0fs; retrying on CPU",
                        cfg.face_model_load_timeout_seconds,
                    )
                    await _emit_progress("GPU load timed out; retrying on CPU…")
                    await _run_load(force_cpu=True)
                elif result == "error" or not self.engine.loaded:
                    await _emit_progress("GPU load failed; retrying on CPU…")
                    await _run_load(force_cpu=True)
            else:
                await _run_load(force_cpu=True)

            if self.engine.loaded_with_cpu_fallback:
                log.warning(
                    "face model loaded on CPU after GPU failure active_provider=%s",
                    self.engine.active_provider,
                )
            elif self.engine.loaded:
                log.info(
                    "face model loaded active_provider=%s providers=%s elapsed_s=%.1f",
                    self.engine.active_provider,
                    self.engine.providers,
                    time.monotonic() - t0,
                )

        await self._warmup_inference(progress_cb=progress_cb, total=total, cancel_event=cancel_event)

    async def _detect_faces_with_timeout(
        self,
        image: Image.Image,
        *,
        threshold: float,
        cancel_event: asyncio.Event | None = None,
        progress_cb=None,
        progress_detail: str = "",
        total: int = 0,
        file_index: int = 0,
    ) -> list[dict]:
        cfg = self.config
        timeout = (
            None
            if self.engine.loaded_with_cpu_fallback or not cfg.use_gpu
            else cfg.face_inference_timeout_seconds
        )
        try:
            return await self._run_face_worker(
                lambda: self.engine.detect_faces(image, conf=threshold),
                timeout_s=timeout,
                cancel_event=cancel_event,
                progress_cb=progress_cb,
                progress_detail=progress_detail,
                total=total,
                processed=max(0, file_index - 1),
                phase="detect",
                log_label=f"face infer file={file_index}",
            )
        except TimeoutError:
            if cfg.use_gpu and not self.engine.loaded_with_cpu_fallback:
                log.warning(
                    "face GPU inference timed out on file %s; reloading on CPU for remainder",
                    file_index,
                )
                await self._reload_engine_on_cpu(cancel_event=cancel_event)
                await self._warmup_inference(cancel_event=cancel_event)
                return await self._run_face_worker(
                    lambda: self.engine.detect_faces(image, conf=threshold),
                    timeout_s=min(60.0, cfg.face_inference_timeout_seconds),
                    cancel_event=cancel_event,
                    progress_cb=progress_cb,
                    progress_detail=progress_detail,
                    total=total,
                    processed=max(0, file_index - 1),
                    phase="detect",
                    log_label=f"face infer_cpu file={file_index}",
                )
            raise

    async def status(self) -> dict:
        st = self.db.stats()
        cache = inspect_face_pack(
            resolved_face_models_root(self.config.face_models_dir),
            self.config.face_model_pack,
        )
        installed = available_ort_providers()
        st.update(
            {
                "model_loaded": self.engine.loaded,
                "model_pack": self.config.face_model_pack,
                "active_provider": self.engine.active_provider if self.engine.loaded else None,
                "providers": self.engine.providers if self.engine.loaded else [],
                "installed_ort_providers": installed,
                "gpu_backend": self.config.gpu_backend,
                "use_gpu": self.config.use_gpu,
                "cpu_fallback": self.engine.loaded_with_cpu_fallback,
                "downloaded": cache["downloaded"],
                "cache_ok": cache["cache_ok"],
                "cache_issues": cache["cache_issues"],
                "model_path": cache["path"],
            }
        )
        return st

    async def detect_file(
        self,
        client: HydrusClient,
        *,
        file_id: int,
        file_hash: str,
        meta: dict | None,
        service_key: str,
        det_threshold: float | None = None,
        replace_existing: bool = False,
        cancel_event: asyncio.Event | None = None,
        progress_cb=None,
        file_index: int = 0,
        total: int = 0,
    ) -> dict:
        """Detect faces in one file, store embeddings, apply detection marker tags."""
        cfg = self.config
        threshold = cfg.face_det_threshold if det_threshold is None else det_threshold

        if cancel_event and cancel_event.is_set():
            raise asyncio.CancelledError("face detect cancelled")

        if cfg.face_skip_if_detected and not replace_existing:
            markers = {cfg.face_marker_detected, cfg.face_marker_not_visible}
            if _storage_has_marker(meta, service_key, markers):
                return {
                    "file_id": file_id,
                    "hash": file_hash,
                    "face_count": 0,
                    "skipped": True,
                    "skip_reason": "marker_present",
                    "tags": [],
                }

        mime = _hydrus_mime(meta)
        if _is_video_mime(mime):
            log.info("face detect skip file_id=%s reason=video_excluded mime=%s", file_id, mime)
            return {
                "file_id": file_id,
                "hash": file_hash,
                "face_count": 0,
                "skipped": True,
                "skip_reason": "video_excluded",
                "tags": [],
            }

        faces: list[dict] = []
        try:
            log.info("face detect file start file_id=%s mime=%s index=%s", file_id, mime or "unknown", file_index)
            try:
                raw, _ = await client.get_file(file_id=file_id)
                if cancel_event and cancel_event.is_set():
                    raise asyncio.CancelledError("face detect cancelled")
                image = await asyncio.to_thread(_decode_image, raw)
            except (UnidentifiedImageError, OSError):
                thumb, _ = await client.get_thumbnail(file_id=file_id)
                if cancel_event and cancel_event.is_set():
                    raise asyncio.CancelledError("face detect cancelled")
                image = await asyncio.to_thread(_decode_image, thumb)
            faces = await self._detect_faces_with_timeout(
                image,
                threshold=threshold,
                cancel_event=cancel_event,
                progress_cb=progress_cb,
                progress_detail=f"File {file_index}/{total} · running face detection",
                total=total,
                file_index=file_index,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("face detect failed file_id=%s", file_id)
            return {
                "file_id": file_id,
                "hash": file_hash,
                "face_count": 0,
                "error": str(exc),
                "tags": [],
            }

        log.info(
            "face detect file done file_id=%s faces=%s provider=%s",
            file_id,
            len(faces),
            self.engine.active_provider,
        )

        if replace_existing:
            self.db.delete_faces_for_hash(file_hash)

        for i, det in enumerate(faces):
            self.db.store_face(file_hash, i, det["bbox"], det["embedding"])

        if faces:
            marker_tag = cfg.face_marker_detected
        else:
            marker_tag = cfg.face_marker_not_visible

        tags = [marker_tag]
        if service_key:
            existing = existing_storage_tag_keys(meta, service_key) if meta else set()
            new_tags, _ = filter_new_tags(tags, existing)
            if new_tags:
                await client.add_tags(file_hash, service_key, new_tags)

        return {
            "file_id": file_id,
            "hash": file_hash,
            "face_count": len(faces),
            "skipped": False,
            "tags": tags,
        }

    async def detect_batch(
        self,
        client: HydrusClient,
        *,
        file_ids: list[int],
        service_key: str,
        det_threshold: float | None = None,
        replace_existing: bool = False,
        progress_cb=None,
        cancel_event: asyncio.Event | None = None,
    ) -> list[dict]:
        total = len(file_ids)
        t_batch = time.monotonic()
        log.info(
            "face detect_batch start files=%s replace_existing=%s use_gpu=%s gpu_backend=%s",
            total,
            replace_existing,
            self.config.use_gpu,
            self.config.gpu_backend,
        )
        if progress_cb:
            await progress_cb(
                face_detect_progress(
                    "metadata",
                    detail="Fetching file metadata from Hydrus…",
                    total=total,
                )
            )

        async def metadata_chunk_cb(done: int, chunk_total: int) -> None:
            if progress_cb:
                await progress_cb(
                    face_detect_progress(
                        "metadata",
                        detail=f"Metadata {done}/{chunk_total}",
                        total=chunk_total,
                    )
                )

        meta_by_id = await load_metadata_by_file_id(
            client,
            file_ids,
            chunk_sz=self.config.hydrus_metadata_chunk_size,
            cancel_event=cancel_event,
            progress_cb=metadata_chunk_cb,
        )
        log.info(
            "face detect_batch metadata ready rows=%s elapsed_s=%.2f",
            len(meta_by_id),
            time.monotonic() - t_batch,
        )
        cfg = self.config
        queue_stats = _analyze_detect_queue(
            file_ids,
            meta_by_id,
            service_key=service_key,
            skip_if_detected=cfg.face_skip_if_detected,
            replace_existing=replace_existing,
            marker_detected=cfg.face_marker_detected,
            marker_not_visible=cfg.face_marker_not_visible,
        )
        queue_detail = _queue_summary_detail(queue_stats)
        log.info(
            "face detect_batch queue total=%s images=%s videos=%s marker_skip=%s missing_hash=%s to_process=%s",
            queue_stats["total"],
            queue_stats["images"],
            queue_stats["videos"],
            queue_stats["marker_skip"],
            queue_stats["missing_hash"],
            queue_stats["to_process"],
        )
        if progress_cb:
            await progress_cb(
                face_detect_progress(
                    "queue",
                    detail=queue_detail,
                    total=total,
                    queue=queue_stats,
                )
            )
        if queue_stats["to_process"] == 0:
            log.info("face detect_batch skip model load: nothing to process (%s)", queue_detail)
            results: list[dict] = []
            for fid in file_ids:
                meta = meta_by_id.get(fid)
                fhash = (meta or {}).get("hash") or ""
                if not fhash:
                    results.append({"file_id": fid, "hash": "", "face_count": 0, "error": "missing_hash"})
                    continue
                mime = _hydrus_mime(meta)
                if _is_video_mime(mime):
                    results.append(
                        {
                            "file_id": fid,
                            "hash": fhash,
                            "face_count": 0,
                            "skipped": True,
                            "skip_reason": "video_excluded",
                            "tags": [],
                        }
                    )
                    continue
                markers = {cfg.face_marker_detected, cfg.face_marker_not_visible}
                if cfg.face_skip_if_detected and not replace_existing and _storage_has_marker(
                    meta, service_key, markers
                ):
                    results.append(
                        {
                            "file_id": fid,
                            "hash": fhash,
                            "face_count": 0,
                            "skipped": True,
                            "skip_reason": "marker_present",
                            "tags": [],
                        }
                    )
                    continue
            log.info(
                "face detect_batch done processed=%s faces=0 tagged=0 skipped=%s videos=%s marker_skip=%s errors=%s elapsed_s=%.1f provider=none",
                len(results),
                len(results),
                queue_stats["videos"],
                queue_stats["marker_skip"],
                sum(1 for r in results if r.get("error")),
                time.monotonic() - t_batch,
            )
            return results
        if progress_cb:
            await progress_cb(
                face_detect_progress(
                    "model_load",
                    detail="Loading InsightFace model…",
                    total=total,
                    providers=resolve_ort_providers(
                        use_gpu=self.config.use_gpu,
                        gpu_backend=self.config.gpu_backend,
                    ),
                )
            )
        await self.ensure_model_loaded(progress_cb=progress_cb, total=total, cancel_event=cancel_event)
        results: list[dict] = []
        faces_total = 0
        tagged_total = 0
        skipped_total = 0
        video_skip_total = 0
        marker_skip_total = 0
        for idx, fid in enumerate(file_ids):
            if cancel_event and cancel_event.is_set():
                log.info("face detect_batch cancelled at %s/%s", idx, total)
                break
            meta = meta_by_id.get(fid)
            fhash = (meta or {}).get("hash") or ""
            mime = _hydrus_mime(meta)
            if progress_cb:
                await progress_cb(
                    face_detect_progress(
                        "scan",
                        detail=f"Processing file {idx + 1}/{total}…",
                        processed=idx,
                        total=total,
                        file_id=fid,
                        active_provider=self.engine.active_provider,
                        queue=queue_stats,
                    )
                )
            if not fhash:
                row = {"file_id": fid, "hash": "", "face_count": 0, "error": "missing_hash"}
                results.append(row)
                log.warning("face detect skip file_id=%s reason=missing_hash", fid)
                continue
            if _is_video_mime(mime):
                row = {
                    "file_id": fid,
                    "hash": fhash,
                    "face_count": 0,
                    "skipped": True,
                    "skip_reason": "video_excluded",
                    "tags": [],
                }
                results.append(row)
                video_skip_total += 1
                skipped_total += 1
                log.info("face detect skip file_id=%s reason=video_excluded mime=%s", fid, mime)
                if progress_cb:
                    await progress_cb(
                        face_detect_progress(
                            "scan",
                            detail=f"File {idx + 1}/{total} · video skipped",
                            processed=idx + 1,
                            total=total,
                            file_id=fid,
                            face_count=0,
                            skipped=True,
                            skip_reason="video_excluded",
                            active_provider=self.engine.active_provider,
                        )
                    )
                continue
            markers = {cfg.face_marker_detected, cfg.face_marker_not_visible}
            if cfg.face_skip_if_detected and not replace_existing and _storage_has_marker(
                meta, service_key, markers
            ):
                row = {
                    "file_id": fid,
                    "hash": fhash,
                    "face_count": 0,
                    "skipped": True,
                    "skip_reason": "marker_present",
                    "tags": [],
                }
                results.append(row)
                marker_skip_total += 1
                skipped_total += 1
                if progress_cb:
                    await progress_cb(
                        face_detect_progress(
                            "scan",
                            detail=f"File {idx + 1}/{total} · already tagged (skipped)",
                            processed=idx + 1,
                            total=total,
                            file_id=fid,
                            face_count=0,
                            skipped=True,
                            skip_reason="marker_present",
                            active_provider=self.engine.active_provider,
                        )
                    )
                continue
            try:
                row = await self.detect_file(
                    client,
                    file_id=fid,
                    file_hash=fhash,
                    meta=meta,
                    service_key=service_key,
                    det_threshold=det_threshold,
                    replace_existing=replace_existing,
                    cancel_event=cancel_event,
                    progress_cb=progress_cb,
                    file_index=idx + 1,
                    total=total,
                )
            except asyncio.CancelledError:
                log.info("face detect_batch cancelled during file %s/%s", idx + 1, total)
                raise
            results.append(row)
            faces_total += int(row.get("face_count", 0) or 0)
            if row.get("skipped"):
                skipped_total += 1
            elif row.get("tags"):
                tagged_total += 1
            if (idx + 1) % 10 == 0 or idx == 0 or idx + 1 == total:
                log.info(
                    "face detect progress %s/%s file_id=%s faces=%s skipped=%s tagged=%s provider=%s",
                    idx + 1,
                    total,
                    fid,
                    row.get("face_count", 0),
                    row.get("skipped", False),
                    bool(row.get("tags")),
                    self.engine.active_provider,
                )
            if progress_cb:
                row_detail = (
                    f"File {idx + 1}/{total}"
                    + (f" · {row.get('face_count', 0)} face(s)" if not row.get("skipped") else " · skipped")
                )
                await progress_cb(
                    face_detect_progress(
                        "scan",
                        detail=row_detail,
                        processed=idx + 1,
                        total=total,
                        file_id=fid,
                        face_count=row.get("face_count", 0),
                        skipped=row.get("skipped", False),
                        active_provider=self.engine.active_provider,
                    )
                )
        log.info(
            "face detect_batch done processed=%s faces=%s tagged=%s skipped=%s videos=%s marker_skip=%s errors=%s elapsed_s=%.1f provider=%s",
            len(results),
            faces_total,
            tagged_total,
            skipped_total,
            video_skip_total,
            marker_skip_total,
            sum(1 for r in results if r.get("error")),
            time.monotonic() - t_batch,
            self.engine.active_provider,
        )
        return results

    async def recognize(
        self,
        client: HydrusClient,
        *,
        service_key: str,
        max_distance: float | None = None,
        min_faces: int | None = None,
        allow_new: bool = True,
        staged: bool = True,
        extra_marker_tag: str | None = None,
    ) -> dict:
        cfg = self.config
        max_dist = cfg.face_recognition_max_distance if max_distance is None else max_distance
        distance_method = cfg.face_recognition_distance_method
        stages = (
            staged_min_faces_list(cfg.face_recognition_stages)
            if staged
            else [cfg.face_recognition_min_faces if min_faces is None else min_faces]
        )
        if not stages:
            stages = [3]

        faces_in_db = len(self.db.load_all_faces())
        stage_total = len(stages)
        log.info(
            "face recognize start stages=%s faces_in_db=%s service=%s",
            stages,
            faces_in_db,
            service_key[:8] + "…" if len(service_key) > 8 else service_key,
        )
        total_assigned = 0
        stage_results: list[dict] = []
        for stage_idx, stage_min in enumerate(stages, start=1):
            step_label = face_recognize_step_label(stage_idx, stage_total, int(stage_min))
            log.info("face recognize %s", step_label)
            faces = self.db.load_all_faces()
            assigned = await asyncio.to_thread(
                cluster_faces,
                faces,
                max_distance=max_dist,
                min_faces=int(stage_min),
                allow_new=allow_new,
                distance_method=distance_method,
                create_person=self.db.create_new_person,
                assign_person=self.db.assign_face_to_person,
            )
            total_assigned += assigned
            stage_results.append(
                {
                    "stage": stage_idx,
                    "stage_total": stage_total,
                    "min_faces": int(stage_min),
                    "assigned": assigned,
                    "step_label": step_label,
                }
            )
            log.info(
                "face recognize stage done %s assigned=%s cumulative=%s",
                step_label,
                assigned,
                total_assigned,
            )

        apply_label = face_recognize_apply_label()
        log.info("face recognize %s", apply_label)
        file_tags = self.db.get_file_person_tags(cfg.face_person_tag_prefix)
        files_written = 0
        tag_strings = 0
        marker = extra_marker_tag if extra_marker_tag is not None else cfg.face_marker_recognized

        for fhash, person_tags in file_tags.items():
            tag_list = sorted(person_tags)
            if marker:
                tag_list.append(marker)
            try:
                await client.add_tags(fhash, service_key, tag_list)
                files_written += 1
                tag_strings += len(tag_list)
            except Exception:
                log.exception("face recognize apply failed hash=%s", fhash[:16])

        log.info(
            "face recognize done assigned_faces=%s files_tagged=%s persons=%s",
            total_assigned,
            files_written,
            self.db.stats()["persons"],
        )
        return {
            "assigned_faces": total_assigned,
            "files_tagged": files_written,
            "tag_strings": tag_strings,
            "persons": self.db.stats()["persons"],
            "stages": stage_results,
            "faces_in_db": faces_in_db,
        }

    async def reset_assignments(self) -> None:
        await asyncio.to_thread(self.db.reset_person_assignments)

    async def clean_orphans(self, client: HydrusClient) -> dict:
        db_hashes = self.db.distinct_file_hashes()
        if not db_hashes:
            log.info("face clean orphans: db empty")
            return {"removed": 0, "checked": 0}

        alive: set[str] = set()
        chunk = self.config.hydrus_metadata_chunk_size
        failed_batches = 0
        for i in range(0, len(db_hashes), chunk):
            batch = db_hashes[i : i + chunk]
            try:
                rows = await client.get_file_metadata_by_hashes(batch)
                seen: set[str] = set()
                for item in rows:
                    h = item.get("hash")
                    if not h:
                        continue
                    seen.add(h)
                    if item.get("is_deleted") is False:
                        alive.add(h)
                # Hashes absent from Hydrus response are not alive.
            except Exception:
                failed_batches += 1
                log.exception("face clean metadata batch failed hashes=%s", len(batch))

        missing = set(db_hashes) - alive
        removed = self.db.delete_orphan_hashes(missing)
        log.info(
            "face clean orphans done checked=%s alive=%s removed=%s failed_batches=%s",
            len(db_hashes),
            len(alive),
            removed,
            failed_batches,
        )
        return {
            "removed": removed,
            "checked": len(db_hashes),
            "alive": len(alive),
            "failed_batches": failed_batches,
        }

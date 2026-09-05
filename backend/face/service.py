"""Face detection and recognition orchestration."""

from __future__ import annotations

import asyncio
import logging
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from backend.config import AppConfig
from backend.face.clustering import cluster_faces, staged_min_faces_list
from backend.face.embeddings_db import FaceEmbeddingsDB
from backend.face.engine import FaceEngine
from backend.face.video_frames import extract_video_frames
from backend.hydrus.client import HydrusClient
from backend.hydrus.tag_merge import existing_storage_tag_keys, filter_new_tags
from backend.services.tagging_shared import load_metadata_by_file_id

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


class FaceTaggingService:
    _instance: "FaceTaggingService | None" = None

    def __init__(self, config: AppConfig):
        self.config = config
        self.engine = FaceEngine(
            use_gpu=config.use_gpu,
            gpu_backend=config.gpu_backend,
            model_pack=config.face_model_pack,
            det_threshold=config.face_det_threshold,
            models_root=resolved_face_models_root(config.face_models_dir),
        )
        self.db = FaceEmbeddingsDB(resolved_face_db_path(config.face_embeddings_db_path))
        self.db.init()

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
            cls._instance.engine.unload()

        if prev.face_embeddings_db_path != config.face_embeddings_db_path:
            cls._instance.db = FaceEmbeddingsDB(resolved_face_db_path(config.face_embeddings_db_path))
            cls._instance.db.init()

        cls._instance.config = config
        return cls._instance

    @classmethod
    def unload_model_from_memory(cls) -> None:
        if cls._instance is not None:
            cls._instance.engine.unload()

    def face_tag_service_name(self) -> str:
        name = (self.config.face_target_tag_service or "").strip()
        return name or self.config.target_tag_service

    async def ensure_model_loaded(self) -> None:
        if not self.engine.loaded:
            await asyncio.to_thread(self.engine.load)

    async def status(self) -> dict:
        st = self.db.stats()
        st.update(
            {
                "model_loaded": self.engine.loaded,
                "model_pack": self.config.face_model_pack,
                "active_provider": self.engine.active_provider if self.engine.loaded else None,
                "providers": self.engine.providers if self.engine.loaded else [],
                "gpu_backend": self.config.gpu_backend,
                "use_gpu": self.config.use_gpu,
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
    ) -> dict:
        """Detect faces in one file, store embeddings, apply detection marker tags."""
        cfg = self.config
        await self.ensure_model_loaded()
        threshold = cfg.face_det_threshold if det_threshold is None else det_threshold

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
        faces: list[dict] = []
        try:
            if mime.startswith("video/"):
                raw, _ = await client.get_file(file_id=file_id)
                frames = extract_video_frames(raw, num_frames=cfg.face_video_frame_count)
                for frame in frames:
                    faces.extend(await asyncio.to_thread(self.engine.detect_faces, frame, conf=threshold))
            else:
                try:
                    raw, _ = await client.get_file(file_id=file_id)
                    image = await asyncio.to_thread(_decode_image, raw)
                except (UnidentifiedImageError, OSError):
                    thumb, _ = await client.get_thumbnail(file_id=file_id)
                    image = await asyncio.to_thread(_decode_image, thumb)
                faces = await asyncio.to_thread(self.engine.detect_faces, image, conf=threshold)
        except Exception as exc:
            log.exception("face detect failed file_id=%s", file_id)
            return {
                "file_id": file_id,
                "hash": file_hash,
                "face_count": 0,
                "error": str(exc),
                "tags": [],
            }

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
                await client.apply_tag_actions(
                    hash_=file_hash,
                    service_key=service_key,
                    add_tags=new_tags,
                )

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
        meta_by_id = await load_metadata_by_file_id(
            client,
            file_ids,
            chunk_size=self.config.hydrus_metadata_chunk_size,
        )
        results: list[dict] = []
        total = len(file_ids)
        for idx, fid in enumerate(file_ids):
            if cancel_event and cancel_event.is_set():
                break
            meta = meta_by_id.get(fid)
            fhash = (meta or {}).get("hash") or ""
            if not fhash:
                results.append({"file_id": fid, "hash": "", "face_count": 0, "error": "missing_hash"})
                continue
            row = await self.detect_file(
                client,
                file_id=fid,
                file_hash=fhash,
                meta=meta,
                service_key=service_key,
                det_threshold=det_threshold,
                replace_existing=replace_existing,
            )
            results.append(row)
            if progress_cb:
                await progress_cb(
                    {
                        "type": "progress",
                        "processed": idx + 1,
                        "total": total,
                        "file_id": fid,
                        "face_count": row.get("face_count", 0),
                        "skipped": row.get("skipped", False),
                    }
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

        total_assigned = 0
        for stage_min in stages:
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
            log.info("face recognize stage min_faces=%s assigned=%s", stage_min, assigned)

        file_tags = self.db.get_file_person_tags(cfg.face_person_tag_prefix)
        files_written = 0
        tag_strings = 0
        marker = extra_marker_tag if extra_marker_tag is not None else cfg.face_marker_recognized

        for fhash, person_tags in file_tags.items():
            tag_list = sorted(person_tags)
            if marker:
                tag_list.append(marker)
            try:
                await client.apply_tag_actions(
                    hash_=fhash,
                    service_key=service_key,
                    add_tags=tag_list,
                )
                files_written += 1
                tag_strings += len(tag_list)
            except Exception:
                log.exception("face recognize apply failed hash=%s", fhash[:16])

        return {
            "assigned_faces": total_assigned,
            "files_tagged": files_written,
            "tag_strings": tag_strings,
            "persons": self.db.stats()["persons"],
        }

    async def reset_assignments(self) -> None:
        await asyncio.to_thread(self.db.reset_person_assignments)

    async def clean_orphans(self, client: HydrusClient) -> dict:
        db_hashes = self.db.distinct_file_hashes()
        if not db_hashes:
            return {"removed": 0}

        alive: set[str] = set()
        chunk = self.config.hydrus_metadata_chunk_size
        for i in range(0, len(db_hashes), chunk):
            batch = db_hashes[i : i + chunk]
            try:
                rows = await client.get_file_metadata(hashes=batch)
                for item in rows:
                    if item.get("is_deleted") is False:
                        alive.add(item["hash"])
            except Exception:
                log.exception("face clean metadata batch failed")

        missing = set(db_hashes) - alive
        removed = self.db.delete_orphan_hashes(missing)
        return {"removed": removed, "checked": len(db_hashes)}

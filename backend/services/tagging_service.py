"""Orchestrates the batch tagging workflow."""

import asyncio
import gc
import logging
import time
from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image, UnidentifiedImageError

log = logging.getLogger(__name__)

from backend.config import AppConfig, clamp_hydrus_metadata_chunk_size, resolved_ort_profile_dir
from backend.hydrus.client import HydrusClient
from backend.hydrus.tag_merge import (
    build_wd_model_marker,
    dedupe_wd_model_markers_in_tags,
    inference_skip_decision,
)
from backend.services.model_manager import ModelManager
from backend.services.tagging_shared import clamp_inference_batch, load_metadata_by_file_id
from backend.tagger.engine import TaggerEngine


def _decode_raster_bytes(raw: bytes) -> Image.Image:
    """Decode image bytes off the event loop thread."""
    im = Image.open(BytesIO(raw))
    im.load()
    return im


def _hydrus_mime(meta: dict | None) -> str:
    if not meta:
        return ""
    m = meta.get("mime")
    return m.strip().lower() if isinstance(m, str) else ""


def _prefer_thumbnail_only(meta: dict | None) -> bool:
    """Avoid downloading multi‑GB originals when Hydrus already exposes a raster thumbnail."""
    mime = _hydrus_mime(meta)
    return mime.startswith("video/")


class TaggingService:
    _instance: "TaggingService | None" = None

    def __init__(self, config: AppConfig):
        self.config = config
        self.engine = TaggerEngine(use_gpu=config.use_gpu)
        self.model_manager = ModelManager(config.models_dir)
        self._loaded_model: str | None = None
        # When set, ONNX was built with these ORT thread counts (intra, inter). None + _loaded_model set → tests / legacy.
        self._loaded_ort_threads: tuple[int, int] | None = None
        self._last_load_used_hub: bool = False

    @classmethod
    def get_instance(cls, config: AppConfig) -> "TaggingService":
        if cls._instance is None:
            cls._instance = cls(config)
            return cls._instance

        try:
            new_md = Path(config.models_dir).expanduser().resolve()
            old_md = Path(cls._instance.model_manager.models_dir).expanduser().resolve()
            if new_md != old_md:
                log.info(
                    "TaggingService models_dir changed %s -> %s; refreshing ModelManager and unloading ONNX",
                    old_md,
                    new_md,
                )
                cls._instance.engine.finalize_ort_profiling()
                cls._instance.model_manager = ModelManager(config.models_dir)
                cls._instance._loaded_model = None
                cls._instance._loaded_ort_threads = None
        except OSError:
            cls._instance.engine.finalize_ort_profiling()
            cls._instance.model_manager = ModelManager(config.models_dir)
            cls._instance._loaded_model = None
            cls._instance._loaded_ort_threads = None

        prev = cls._instance.config
        if (
            prev.use_gpu != config.use_gpu
            or prev.cpu_intra_op_threads != config.cpu_intra_op_threads
            or prev.cpu_inter_op_threads != config.cpu_inter_op_threads
        ):
            cls._instance.engine.finalize_ort_profiling()
            cls._instance.engine = TaggerEngine(use_gpu=config.use_gpu)
            cls._instance._loaded_model = None
            cls._instance._loaded_ort_threads = None

        cls._instance.config = config
        return cls._instance

    @classmethod
    def unload_model_from_memory(cls) -> str | None:
        """Release ONNX session; on-disk files under ``models_dir`` are unchanged."""
        if cls._instance is None:
            return None
        inst = cls._instance
        prev = inst._loaded_model
        cfg = inst.config
        inst.engine.finalize_ort_profiling()
        inst.engine = TaggerEngine(use_gpu=cfg.use_gpu)
        inst._loaded_model = None
        inst._loaded_ort_threads = None
        gc.collect()
        log.info(
            "unload_model_from_memory: released ONNX (was %r); disk cache dir=%s",
            prev,
            inst.model_manager.models_dir,
        )
        return prev

    @staticmethod
    def _clamp_ort_threads(intra: int, inter: int) -> tuple[int, int]:
        return (max(1, min(64, int(intra))), max(1, min(16, int(inter))))

    def _resolve_ort_threads(
        self,
        ort_intra_op_threads: int | None,
        ort_inter_op_threads: int | None,
    ) -> tuple[int, int]:
        """Effective ORT thread counts: optional overrides else ``AppConfig`` (session-local tuning)."""
        ia = self.config.cpu_intra_op_threads if ort_intra_op_threads is None else ort_intra_op_threads
        ie = self.config.cpu_inter_op_threads if ort_inter_op_threads is None else ort_inter_op_threads
        return self._clamp_ort_threads(int(ia), int(ie))

    def _loaded_threads_effective(self) -> tuple[int, int] | None:
        if self._loaded_model is None:
            return None
        if self._loaded_ort_threads is not None:
            return self._loaded_ort_threads
        return (
            self.config.cpu_intra_op_threads,
            self.config.cpu_inter_op_threads,
        )

    def _model_already_loaded(
        self,
        name: str,
        intra: int,
        inter: int,
    ) -> bool:
        if self._loaded_model != name or self.engine.use_gpu != self.config.use_gpu:
            return False
        loaded = self._loaded_threads_effective()
        return loaded is not None and loaded == (intra, inter)

    async def ensure_model(
        self,
        explicit_name: str | None,
        *,
        ort_intra_op_threads: int | None = None,
        ort_inter_op_threads: int | None = None,
    ) -> None:
        """Load ONNX + labels if needed (no-op when same model + ORT thread key already in memory).

        Optional thread overrides apply session-local tuning without mutating ``config.yaml``.
        """
        raw = (explicit_name or "").strip()
        target = raw if raw else self.config.default_model
        eff_intra, eff_inter = self._resolve_ort_threads(ort_intra_op_threads, ort_inter_op_threads)
        t0 = time.perf_counter()
        if self._model_already_loaded(target, eff_intra, eff_inter):
            ms = (time.perf_counter() - t0) * 1000.0
            log.info(
                "ensure_model metrics model=%s memory_cache_hit=True duration_ms=%.2f",
                target,
                ms,
            )
            return
        log.info("ensure_model loading: %s", target)
        await asyncio.to_thread(
            self.load_model,
            target,
            ort_intra_op_threads=ort_intra_op_threads,
            ort_inter_op_threads=ort_inter_op_threads,
        )
        ms = (time.perf_counter() - t0) * 1000.0
        log.info(
            "ensure_model metrics model=%s memory_cache_hit=False duration_ms=%.2f",
            target,
            ms,
        )

    def load_model(
        self,
        name: str,
        *,
        ort_intra_op_threads: int | None = None,
        ort_inter_op_threads: int | None = None,
    ) -> None:
        """Download if needed, verify disk cache, then load ONNX (reuse RAM if same load key).

        Load key: ``(model_name, use_gpu, intra_op_threads, inter_op_threads)``.
        """
        eff_intra, eff_inter = self._resolve_ort_threads(ort_intra_op_threads, ort_inter_op_threads)
        t0 = time.perf_counter()
        self._last_load_used_hub = False
        if self._model_already_loaded(name, eff_intra, eff_inter):
            log.info(
                "load_model metrics model=%s memory_already_loaded=True disk_cache_hit=n/a "
                "hf_wall_s=0.000 onnx_init_wall_s=0.000 total_wall_s=0.000 "
                "threads_intra=%s threads_inter=%s gpu=%s",
                name,
                eff_intra,
                eff_inter,
                self.config.use_gpu,
            )
            return

        on_disk_before = self.model_manager.is_downloaded(name)
        used_hub = False
        if not on_disk_before:
            log.info("load_model disk cache miss — fetching from HuggingFace: %s", name)
            self.model_manager.download_model(name)
            used_hub = True
        else:
            vr = self.model_manager.verify_model(name, check_remote=False)
            if not vr.ok:
                log.warning(
                    "load_model disk_cache_invalid model=%s refetch_from_hub issues=%s",
                    name,
                    vr.issues,
                )
                self.model_manager.download_model(name)
                used_hub = True
            else:
                if self.model_manager.repair_manifest_if_missing(name):
                    log.info("load_model wrote_missing_cache_manifest model=%s", name)
                log.debug(
                    "load_model disk cache hit (verified): %s dir=%s",
                    name,
                    self.model_manager.models_dir / name,
                )

        self._last_load_used_hub = used_hub
        t_after_hf = time.perf_counter()
        self.model_manager.get_model_path(name)
        models_root = self.model_manager.models_dir
        profile_prefix = None
        if self.config.ort_enable_profiling:
            trace_root = resolved_ort_profile_dir(self.config.ort_profile_dir)
            trace_root.mkdir(parents=True, exist_ok=True)
            safe_name = name.replace("/", "_").replace("\\", "_")
            profile_prefix = str(trace_root / f"wd_{safe_name}_{int(t0 * 1000)}")
        self.engine.load(
            models_root,
            name,
            intra_op_threads=eff_intra,
            inter_op_threads=eff_inter,
            enable_profiling=self.config.ort_enable_profiling,
            profile_file_prefix=profile_prefix,
        )
        t1 = time.perf_counter()
        self._loaded_model = name
        self._loaded_ort_threads = (eff_intra, eff_inter)
        hf_s = t_after_hf - t0
        onnx_s = t1 - t_after_hf
        disk_hit = on_disk_before and not used_hub
        log.info(
            "load_model metrics model=%s memory_already_loaded=False disk_cache_hit=%s "
            "hf_wall_s=%.3f onnx_init_wall_s=%.3f total_wall_s=%.3f "
            "threads_intra=%s threads_inter=%s gpu=%s hub_fetch_this_call=%s",
            name,
            disk_hit,
            hf_s,
            onnx_s,
            t1 - t0,
            eff_intra,
            eff_inter,
            self.config.use_gpu,
            used_hub,
        )

    async def tag_files(
        self,
        client: HydrusClient,
        file_ids: list[int],
        general_threshold: float = 0.35,
        character_threshold: float = 0.85,
        *,
        batch_size: int | None = None,
        model_name: str | None = None,
        cancel_event: asyncio.Event | None = None,
        download_parallel: int | None = None,
        service_key: str | None = None,
        batch_metrics_out: list | None = None,
        prefetched_meta_by_id: dict[int, dict] | None = None,
        outer_batch_override: int | None = None,
    ) -> list[dict]:
        """Tag a list of files from Hydrus.

        Returns list of dicts with file_id, hash, tags, formatted_tags, tag dicts.

        ``prefetched_meta_by_id``: when set (e.g. WebSocket session prefetch), avoids
        repeating get_file_metadata for IDs already present; missing IDs are fetched
        in one merge pass. Second passes over mostly-tagged libraries save Hydrus
        round-trips on every outer batch.

        ``outer_batch_override``: when set (e.g. Tag all marker-skip tail), outer batch
        size up to this cap — skips ONNX work and benefits from large chunks without
        changing normal inference batch limits.
        """
        chunk_sz = clamp_hydrus_metadata_chunk_size(self.config.hydrus_metadata_chunk_size)
        if prefetched_meta_by_id is not None:
            meta_by_id = {fid: prefetched_meta_by_id[fid] for fid in file_ids if fid in prefetched_meta_by_id}
            missing = [fid for fid in file_ids if fid not in meta_by_id]
            if missing:
                log.info(
                    "tag_files metadata merge missing=%s file_id(s) (partial prefetch or new ids)",
                    len(missing),
                )
                extra = await load_metadata_by_file_id(
                    client, missing, chunk_sz=chunk_sz, cancel_event=cancel_event,
                )
                meta_by_id.update(extra)
            log.info(
                "tag_files metadata rows=%s file_ids=%s source=prefetch chunk_size=%s",
                len(meta_by_id),
                len(file_ids),
                chunk_sz,
            )
        else:
            meta_by_id = await load_metadata_by_file_id(
                client, file_ids, chunk_sz=chunk_sz, cancel_event=cancel_event,
            )
            log.info(
                "tag_files metadata rows=%s file_ids=%s source=fetch chunk_size=%s",
                len(meta_by_id),
                len(file_ids),
                chunk_sz,
            )

        resolved_model = (model_name or "").strip() or self.config.default_model
        marker = build_wd_model_marker(resolved_model, self.config.wd_model_marker_template)
        skip_marker = bool(marker) and self.config.wd_skip_inference_if_marker_present
        append_marker = bool(marker) and self.config.wd_append_model_marker_tag
        svc_key = (service_key or "").strip()

        results: list[dict] = []
        if outer_batch_override is not None:
            effective_batch = max(1, min(int(outer_batch_override), 2048))
        else:
            effective_batch = clamp_inference_batch(batch_size, self.config.batch_size)
        parallel = download_parallel
        if parallel is None:
            parallel = self.config.hydrus_download_parallel
        parallel = max(1, min(32, int(parallel)))
        wall_fetch_s = 0.0
        wall_predict_s = 0.0
        skipped_pre_infer_marker_files = 0
        log.info(
            "tag_files start total_files=%s model_name=%s effective_batch=%s download_parallel=%s "
            "skip_if_marker=%s skip_higher_tier=%s append_marker=%s service_key_set=%s marker=%r",
            len(file_ids),
            resolved_model,
            effective_batch,
            parallel,
            skip_marker,
            self.config.wd_skip_if_higher_tier_model_present,
            append_marker,
            bool(svc_key),
            marker if marker else "",
        )
        sem = asyncio.Semaphore(parallel)

        async def load_raster_for_tagging(fid: int, meta: dict | None):
            """Raster for WD: full file for images; thumbnail-only for video mime (saves huge downloads)."""
            decode_s = 0.0
            thumb_only = _prefer_thumbnail_only(meta)
            if thumb_only:
                log.debug(
                    "tag_files thumbnail-only path file_id=%s mime=%s",
                    fid,
                    _hydrus_mime(meta),
                )
                tdata, tctype = await client.get_thumbnail(file_id=fid)
                try:
                    t0 = time.perf_counter()
                    im = await asyncio.to_thread(_decode_raster_bytes, tdata)
                    decode_s += time.perf_counter() - t0
                    return im, decode_s
                except (UnidentifiedImageError, OSError) as e:
                    log.warning("tag_files thumbnail decode failed file_id=%s: %s", fid, e)
                    return None, decode_s

            file_data, ctype = await client.get_file(file_id=fid)
            try:
                t0 = time.perf_counter()
                im = await asyncio.to_thread(_decode_raster_bytes, file_data)
                decode_s += time.perf_counter() - t0
                return im, decode_s
            except (UnidentifiedImageError, OSError) as e:
                log.debug(
                    "tag_files full file not a raster file_id=%s ctype=%s len=%s: %s; trying thumbnail",
                    fid,
                    ctype,
                    len(file_data),
                    e,
                )
            tdata, tctype = await client.get_thumbnail(file_id=fid)
            try:
                t0 = time.perf_counter()
                im = await asyncio.to_thread(_decode_raster_bytes, tdata)
                decode_s += time.perf_counter() - t0
                return im, decode_s
            except (UnidentifiedImageError, OSError) as e:
                log.warning(
                    "tag_files cannot decode image file_id=%s (full file and thumbnail failed): %s",
                    fid,
                    e,
                )
                if log.isEnabledFor(logging.DEBUG):
                    log.debug(
                        "tag_files decode debug file_id=%s head_file=%r head_thumb=%r",
                        fid,
                        file_data[:24],
                        tdata[:24],
                    )
                return None, decode_s

        async def fetch_one(fid: int):
            async with sem:
                meta = meta_by_id.get(fid)
                if meta is None:
                    meta = {"file_id": fid, "hash": ""}
                try:
                    img, decode_s = await load_raster_for_tagging(fid, meta)
                    if img is None:
                        return None
                    return (fid, img, meta, decode_s)
                except httpx.HTTPError as e:
                    log.warning("tag_files Hydrus HTTP error file_id=%s: %s", fid, e)
                    return None
                except OSError as e:
                    log.warning("tag_files fetch I/O error file_id=%s: %s", fid, e)
                    return None
                except Exception as e:
                    log.warning(
                        "tag_files fetch unexpected error file_id=%s: %s",
                        fid,
                        e,
                        exc_info=log.isEnabledFor(logging.DEBUG),
                    )
                    return None

        batch_index = 0
        for batch_start in range(0, len(file_ids), effective_batch):
            if cancel_event is not None and cancel_event.is_set():
                log.info("tag_files cancel_event set before batch offset=%s", batch_start)
                break

            batch_ids = file_ids[batch_start:batch_start + effective_batch]
            batch_index += 1
            log.info(
                "tag_files batch #%s offset=%s size=%s file_ids=%s…",
                batch_index,
                batch_start,
                len(batch_ids),
                batch_ids[:8],
            )

            skipped_by_fid: dict[int, dict] = {}
            if skip_marker or self.config.wd_skip_if_higher_tier_model_present:
                for fid in batch_ids:
                    meta = meta_by_id.get(fid)
                    do_skip, skip_reason = inference_skip_decision(
                        meta,
                        current_model=resolved_model,
                        canonical_marker=marker,
                        skip_same_model_marker=skip_marker,
                        skip_if_higher_tier_model=self.config.wd_skip_if_higher_tier_model_present,
                        service_key=svc_key,
                        marker_prefix=self.config.wd_model_marker_prefix,
                    )
                    if do_skip and skip_reason:
                        skipped_by_fid[fid] = {
                            "file_id": int(meta.get("file_id", fid)) if meta else fid,
                            "hash": (meta.get("hash") if meta else "") or "",
                            "general_tags": {},
                            "character_tags": {},
                            "rating_tags": {},
                            "formatted_tags": [],
                            "tags": [],
                            "skipped_inference": True,
                            "skip_reason": skip_reason,
                            "wd_model_marker": marker,
                            "wd_stale_markers_removed": 0,
                        }

            to_infer = [fid for fid in batch_ids if fid not in skipped_by_fid]
            n_same = sum(
                1 for r in skipped_by_fid.values() if r.get("skip_reason") == "wd_model_marker_present"
            )
            n_higher = sum(
                1
                for r in skipped_by_fid.values()
                if r.get("skip_reason") == "wd_skip_higher_tier_model_present"
            )
            skipped_pre_infer_marker_files += len(skipped_by_fid)
            log.info(
                "tag_files batch #%s skip_inference=%s same_model_marker=%s higher_tier_marker=%s "
                "predict_queue=%s",
                batch_index,
                len(skipped_by_fid),
                n_same,
                n_higher,
                len(to_infer),
            )

            infer_by_fid: dict[int, dict] = {}
            predictions: list | None = None
            rows: list = []
            fetched: list = []
            images: list = []
            valid_meta: list = []
            batch_fetch_s = 0.0
            batch_decode_s = 0.0
            batch_predict_s = 0.0

            try:
                if to_infer:
                    t0 = time.perf_counter()
                    fetched = await asyncio.gather(*[fetch_one(fid) for fid in to_infer])
                    fetch_s = time.perf_counter() - t0
                    batch_fetch_s = fetch_s
                    rows = [r for r in fetched if r is not None]
                    batch_decode_s = sum(float(r[3]) for r in rows) if rows else 0.0
                    wall_fetch_s += fetch_s
                    log.info(
                        "tag_files batch #%s fetched_ok=%s of %s (fetch %.2fs decode %.2fs)",
                        batch_index,
                        len(rows),
                        len(to_infer),
                        fetch_s,
                        batch_decode_s,
                    )
                    if not rows and not skipped_by_fid:
                        log.warning("tag_files batch #%s no decodable images; skipping", batch_index)
                        if batch_metrics_out is not None:
                            batch_metrics_out.append(
                                {
                                    "batch_index": batch_index,
                                    "fetch_s": round(batch_fetch_s, 4),
                                    "decode_s": round(batch_decode_s, 4),
                                    "predict_s": 0.0,
                                    "files_in_batch": len(batch_ids),
                                    "skipped_pre_infer": len(skipped_by_fid),
                                    "predict_queue": len(to_infer),
                                }
                            )
                        continue

                    images = [r[1] for r in rows]
                    valid_meta = [r[2] for r in rows]

                    if images:
                        log.debug("tag_files batch #%s predict n=%s", batch_index, len(images))
                        t1 = time.perf_counter()
                        try:
                            predictions = await asyncio.to_thread(
                                self.engine.predict,
                                images,
                                general_threshold,
                                character_threshold,
                            )
                        except Exception:
                            log.exception(
                                "tag_files batch #%s ONNX predict failed (n=%s); skipping infer slice",
                                batch_index,
                                len(images),
                            )
                            predictions = []
                        else:
                            pred_s = time.perf_counter() - t1
                            batch_predict_s = pred_s
                            wall_predict_s += pred_s
                            log.info(
                                "tag_files batch #%s predict done (%.2fs for %s images)",
                                batch_index,
                                pred_s,
                                len(images),
                            )

                            for meta, pred in zip(valid_meta, predictions):
                                base_tags = self._format_tags(pred)
                                stale_rm = 0
                                if append_marker and marker:
                                    tags, stale_rm = dedupe_wd_model_markers_in_tags(
                                        list(base_tags),
                                        marker,
                                        marker_prefix=self.config.wd_model_marker_prefix,
                                    )
                                else:
                                    tags = list(base_tags)
                                out_tags = list(tags)
                                fid = int(meta.get("file_id", 0))
                                if stale_rm and log.isEnabledFor(logging.DEBUG):
                                    log.debug(
                                        "tag_files batch #%s file_id=%s removed_stale_wd_markers=%s",
                                        batch_index,
                                        fid,
                                        stale_rm,
                                    )
                                infer_by_fid[fid] = {
                                    "file_id": fid,
                                    "hash": meta.get("hash", ""),
                                    "general_tags": pred["general_tags"],
                                    "character_tags": pred["character_tags"],
                                    "rating_tags": pred["rating_tags"],
                                    "formatted_tags": out_tags,
                                    "tags": out_tags,
                                    "wd_stale_markers_removed": stale_rm,
                                }

                for fid in batch_ids:
                    if fid in skipped_by_fid:
                        results.append(skipped_by_fid[fid])
                    elif fid in infer_by_fid:
                        results.append(infer_by_fid[fid])

                if batch_metrics_out is not None:
                    batch_metrics_out.append(
                        {
                            "batch_index": batch_index,
                            "fetch_s": round(batch_fetch_s, 4),
                            "decode_s": round(batch_decode_s, 4),
                            "predict_s": round(batch_predict_s, 4),
                            "files_in_batch": len(batch_ids),
                            "skipped_pre_infer": len(skipped_by_fid),
                            "predict_queue": len(to_infer),
                        }
                    )
            finally:
                predictions = None
                for r in rows:
                    try:
                        r[1].close()
                    except (OSError, ValueError, AttributeError):
                        pass
                del rows
                del images
                del fetched
                del valid_meta
                del infer_by_fid
                del skipped_by_fid
                # Full GC every N inference batches avoids pausing after every batch (CPU cost on huge Tag all).
                if to_infer and batch_index > 0 and batch_index % 4 == 0:
                    gc.collect()

        gc.collect()

        stale_total = sum(int(r.get("wd_stale_markers_removed") or 0) for r in results)
        inferred_files = sum(1 for r in results if not r.get("skipped_inference"))
        log.info(
            "tag_files metrics model=%s files_requested=%s files_returned=%s "
            "skipped_pre_infer_marker_files=%s inferred_files=%s "
            "stale_wd_markers_dropped=%s wall_hydrus_fetch_s=%.3f wall_onnx_predict_s=%.3f "
            "effective_batch=%s hydrus_download_parallel=%s",
            resolved_model,
            len(file_ids),
            len(results),
            skipped_pre_infer_marker_files,
            inferred_files,
            stale_total,
            wall_fetch_s,
            wall_predict_s,
            effective_batch,
            parallel,
        )
        return results

    def _format_tags(self, prediction: dict) -> list[str]:
        """Format tags with configured prefixes."""
        tags = []

        prefix = self.config.general_tag_prefix
        for tag in prediction["general_tags"]:
            formatted = tag.replace("_", " ")
            tags.append(f"{prefix}{formatted}" if prefix else formatted)

        prefix = self.config.character_tag_prefix
        for tag in prediction["character_tags"]:
            formatted = tag.replace("_", " ")
            tags.append(f"{prefix}{formatted}")

        prefix = self.config.rating_tag_prefix
        if prediction["rating_tags"]:
            top_rating = max(prediction["rating_tags"], key=prediction["rating_tags"].get)
            tags.append(f"{prefix}{top_rating}")

        return tags

"""Tagger endpoints."""

import asyncio
import json
import logging
import time

from fastapi import APIRouter, Body, WebSocket, WebSocketDisconnect

from backend.config import AppConfig, get_config
from backend.hydrus.client import HydrusClient
from backend.hydrus.tag_merge import (
    coalesce_wd_result_tag_strings,
    existing_storage_tag_keys,
    filter_new_tags,
    prune_wd_result_to_pending_tags,
)
from backend.services.model_manager import ModelManager, ModelVerifyResult, SUPPORTED_MODELS
from backend.services.tagging_service import TaggingService, _clamp_inference_batch, load_metadata_by_file_id
from backend.perf_metrics import (
    log_apply_tags_http,
    log_predict_wall,
    record_tagging_session,
)
from backend.services.tagging_session_registry import (
    TaggingSessionHandle,
    active_tagging_sessions_count,
    get_public_session_status,
    register_shutdown_notifier,
    register_tagging_session,
    set_controller_paused,
    unregister_shutdown_notifier,
    unregister_tagging_session,
    update_tagging_public_snapshot,
)

router = APIRouter()
log = logging.getLogger(__name__)


def _get_model_manager() -> ModelManager:
    config = get_config()
    return ModelManager(config.models_dir)


async def _apply_results_chunk(
    client: HydrusClient,
    service_key: str,
    items: list[dict],
) -> tuple[int, int, int]:
    """Write new tags to Hydrus (skips tags already in storage_tags for that service).

    Returns (files_written, new_tag_strings_sent, duplicate_tags_skipped).
    """
    if not service_key or not items:
        return (0, 0, 0)

    ids = [int(i["file_id"]) for i in items if i.get("file_id") is not None]
    meta_by_id: dict[int, dict] = {}
    if ids:
        try:
            rows = await client.get_file_metadata(file_ids=ids)
            for row in rows:
                if isinstance(row, dict) and row.get("file_id") is not None:
                    meta_by_id[int(row["file_id"])] = row
        except Exception:
            log.exception("get_file_metadata failed during apply; applying without deduplication")
            meta_by_id = {}

    files = 0
    tag_strings = 0
    duplicates_skipped = 0
    items_with_tags = 0
    items_all_duplicates = 0
    for item in items:
        tags = item.get("tags") or item.get("formatted_tags", [])
        if not item.get("hash") or not tags:
            continue
        items_with_tags += 1
        tag_list = list(tags)
        fid = item.get("file_id")
        meta = meta_by_id.get(int(fid)) if fid is not None else None
        existing = existing_storage_tag_keys(meta, service_key) if meta is not None else set()
        new_tags, skipped = filter_new_tags(tag_list, existing)
        duplicates_skipped += skipped
        if not new_tags:
            items_all_duplicates += 1
            continue
        try:
            await client.add_tags(
                hash_=item["hash"],
                service_key=service_key,
                tags=new_tags,
            )
        except Exception:
            log.exception(
                "add_tags failed file_id=%s hash=%s… new_tag_count=%s",
                fid,
                (item.get("hash") or "")[:12],
                len(new_tags),
            )
            raise
        files += 1
        tag_strings += len(new_tags)
    log.info(
        "apply_tags chunk summary items=%s with_tags=%s files_written=%s new_tag_strings=%s "
        "dup_tags_skipped=%s items_unchanged_all_dupes=%s",
        len(items),
        items_with_tags,
        files,
        tag_strings,
        duplicates_skipped,
        items_all_duplicates,
    )
    log.info(
        "apply_tags metrics hydrus_duplicate_tag_strings_skipped=%s new_tag_strings_sent=%s "
        "files_written=%s",
        duplicates_skipped,
        tag_strings,
        files,
    )
    return (files, tag_strings, duplicates_skipped)


_METADATA_TRIM_CHUNK = 256


def _prefix_kwargs(cfg: AppConfig) -> dict[str, str]:
    return {
        "general_prefix": cfg.general_tag_prefix or "",
        "character_prefix": cfg.character_tag_prefix or "",
        "rating_prefix": cfg.rating_tag_prefix or "",
    }


async def _trim_ws_results_to_pending_for_service(
    client: HydrusClient,
    service_key: str,
    results: list[dict],
    config: AppConfig,
) -> int:
    """Set each result's tags / structured fields to tags not yet in Hydrus for ``service_key``.

    Returns how many files still have at least one pending tag string.
    """
    if not service_key or not results:
        return 0

    kw = _prefix_kwargs(config)
    ids_unique: list[int] = []
    seen: set[int] = set()
    for r in results:
        if r.get("skipped_inference"):
            continue
        fid = r.get("file_id")
        if fid is None:
            continue
        i = int(fid)
        if i not in seen:
            seen.add(i)
            ids_unique.append(i)

    meta_by_id: dict[int, dict] = {}
    try:
        for off in range(0, len(ids_unique), _METADATA_TRIM_CHUNK):
            chunk = ids_unique[off : off + _METADATA_TRIM_CHUNK]
            rows = await client.get_file_metadata(file_ids=chunk)
            for row in rows:
                if isinstance(row, dict) and row.get("file_id") is not None:
                    meta_by_id[int(row["file_id"])] = row
    except Exception:
        log.exception("trim_ws_results: get_file_metadata failed; leaving results unchanged")
        return 0

    pending_files = 0
    for r in results:
        if r.get("skipped_inference"):
            prune_wd_result_to_pending_tags(r, [], **kw)
            continue
        fid = r.get("file_id")
        if fid is None:
            continue
        meta = meta_by_id.get(int(fid))
        existing = existing_storage_tag_keys(meta, service_key) if meta is not None else set()
        proposed = coalesce_wd_result_tag_strings(r, **kw)
        pending, _ = filter_new_tags(proposed, existing)
        prune_wd_result_to_pending_tags(r, pending, **kw)
        if pending:
            pending_files += 1

    log.info(
        "trim_ws_results_to_pending service_key_set=%s results=%s files_pending_apply=%s",
        bool(service_key),
        len(results),
        pending_files,
    )
    return pending_files


@router.get("/models")
async def list_models():
    """List available models, disk cache status, and which one is loaded in memory."""
    config = get_config()
    manager = _get_model_manager()
    service = TaggingService.get_instance(config)
    models = manager.list_models()
    loaded = service._loaded_model
    for m in models:
        m["loaded_in_memory"] = m["name"] == loaded
    return {
        "success": True,
        "models": models,
        "loaded_model": loaded,
        "default_model": config.default_model,
    }


@router.get("/session/status")
async def tagging_session_status():
    """Whether a tagging WebSocket run is active; snapshot for read-only progress in other tabs."""
    return {"success": True, **get_public_session_status()}


def _verify_result_to_dict(v: ModelVerifyResult) -> dict:
    return {
        "name": v.name,
        "ok": v.ok,
        "issues": v.issues,
        "manifest_present": v.manifest_present,
        "revision_sha": v.local_revision,
        "remote_revision_sha": v.remote_revision,
        "stale_on_hub": v.stale_on_hub,
    }


@router.post("/models/verify")
async def verify_models(body: dict | None = Body(default=None)):
    """Check cached ONNX+CSV (structure, sizes); optional Hub ``main`` revision comparison (network)."""
    payload = body if isinstance(body, dict) else {}
    check_remote = bool(payload.get("check_remote", False))
    raw_name = payload.get("model_name")
    manager = _get_model_manager()
    try:
        if raw_name is not None and str(raw_name).strip() != "":
            n = str(raw_name).strip()
            if n not in SUPPORTED_MODELS:
                return {"success": False, "error": f"Unknown model: {n}"}
            results = [manager.verify_model(n, check_remote=check_remote)]
        else:
            results = manager.verify_all(check_remote=check_remote)
        log.info(
            "models_verify check_remote=%s count=%s ok_count=%s",
            check_remote,
            len(results),
            sum(1 for r in results if r.ok),
        )
        return {
            "success": True,
            "check_remote": check_remote,
            "models_dir": str(manager.models_dir),
            "results": [_verify_result_to_dict(r) for r in results],
        }
    except Exception as e:
        log.exception("models_verify failed")
        return {"success": False, "error": str(e)}


@router.post("/models/{name}/download")
async def download_model(name: str):
    """Download a model from HuggingFace."""
    manager = _get_model_manager()
    try:
        await asyncio.to_thread(manager.download_model, name)
        log.info(
            "download_model endpoint ok name=%s cache_dir=%s",
            name,
            manager.models_dir / name,
        )
        return {"success": True, "message": f"Model {name} downloaded"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/models/{name}/load")
async def load_model(name: str):
    """Load a model into memory."""
    config = get_config()
    service = TaggingService.get_instance(config)
    try:
        on_disk_before = service.model_manager.is_downloaded(name)
        await asyncio.to_thread(service.load_model, name)
        hub_this_call = service._last_load_used_hub
        return {
            "success": True,
            "message": f"Model {name} loaded",
            "downloaded_from_hub": hub_this_call,
            "model_files_cached_on_disk": on_disk_before and not hub_this_call,
        }
    except Exception as e:
        log.exception("load_model endpoint failed name=%s", name)
        return {"success": False, "error": str(e)}


@router.post("/predict")
async def predict(body: dict):
    """Run WD14 tagging on files."""
    config = get_config()
    file_ids = body.get("file_ids", [])
    general_threshold = body.get("general_threshold", config.general_threshold)
    character_threshold = body.get("character_threshold", config.character_threshold)
    raw_bs = body.get("batch_size")
    batch_size = None
    if raw_bs is not None:
        try:
            batch_size = _clamp_inference_batch(int(raw_bs), config.batch_size)
        except (TypeError, ValueError):
            batch_size = None

    eff = _clamp_inference_batch(batch_size, config.batch_size)
    log.info(
        "predict file_count=%s inference_batch=%s (override=%s)",
        len(file_ids),
        eff,
        batch_size,
    )

    service = TaggingService.get_instance(config)
    client = HydrusClient(config.hydrus_api_url, config.hydrus_api_key)
    model_name = body.get("model_name") or None
    service_key = body.get("service_key") or None

    t_wall = time.perf_counter()
    try:
        await service.ensure_model(model_name)
        results = await service.tag_files(
            client=client,
            file_ids=file_ids,
            general_threshold=general_threshold,
            character_threshold=character_threshold,
            batch_size=batch_size,
            model_name=model_name,
            service_key=service_key,
        )
        wall = time.perf_counter() - t_wall
        log.info(
            "predict metrics total_wall_s=%.3f files=%s inference_batch=%s",
            wall,
            len(file_ids),
            eff,
        )
        log_predict_wall(wall_s=wall, file_count=len(file_ids), inference_batch=eff)
        return {"success": True, "results": results}
    except Exception as e:
        log.exception("predict failed file_count=%s", len(file_ids))
        return {"success": False, "error": str(e)}


@router.post("/apply")
async def apply_tags(body: dict):
    """Apply tags to files in Hydrus."""
    config = get_config()
    results = body.get("results", [])
    service_key = body.get("service_key", "")

    client = HydrusClient(config.hydrus_api_url, config.hydrus_api_key)

    try:
        t_wall = time.perf_counter()
        n = len(results)
        bs = max(1, min(512, int(config.apply_tags_http_batch_size)))
        log.info(
            "apply_tags request count=%s service_key_set=%s http_batch=%s",
            n,
            bool(service_key),
            bs,
        )
        applied = 0
        skipped_dups = 0
        for off in range(0, len(results), bs):
            chunk = results[off : off + bs]
            a, _ts, sk = await _apply_results_chunk(client, service_key, chunk)
            applied += a
            skipped_dups += sk
        wall = time.perf_counter() - t_wall
        log.info(
            "apply_tags hydrus_write files=%s skipped_dup_tags=%s",
            applied,
            skipped_dups,
        )
        log_apply_tags_http(
            wall_s=wall,
            result_rows=n,
            files_written=applied,
            dups_skipped=skipped_dups,
        )
        return {
            "success": True,
            "applied": applied,
            "skipped_duplicate_tags": skipped_dups,
        }
    except Exception as e:
        log.exception("apply_tags failed count=%s", len(results))
        return {"success": False, "error": str(e)}


@router.websocket("/ws/progress")
async def progress_ws(websocket: WebSocket):
    """WebSocket: batched tagging, pause/resume, manual flush to Hydrus, cancel."""
    await websocket.accept()
    cancel_event = asyncio.Event()
    pause_event = asyncio.Event()
    flush_event = asyncio.Event()

    try:
        first = await websocket.receive_text()
        request = json.loads(first)
    except WebSocketDisconnect:
        log.info("tagging_ws client disconnected before first message")
        return

    if request.get("action") not in (None, "run"):
        try:
            await websocket.send_json({"type": "error", "message": "expected action run"})
        except Exception:
            pass
        return

    config = get_config()
    file_ids = request.get("file_ids", [])
    general_threshold = float(request.get("general_threshold", config.general_threshold))
    character_threshold = float(request.get("character_threshold", config.character_threshold))
    model_name = request.get("model_name") or None
    raw_bs = request.get("batch_size")
    batch_size = None
    if raw_bs is not None:
        try:
            batch_size = _clamp_inference_batch(int(raw_bs), config.batch_size)
        except (TypeError, ValueError):
            batch_size = None

    service_key = request.get("service_key") or ""
    apply_every = int(request.get("apply_tags_every_n", config.apply_tags_every_n))
    apply_every = max(0, min(256, apply_every))
    download_parallel = request.get("hydrus_download_parallel")
    if download_parallel is not None:
        try:
            download_parallel = max(1, min(32, int(download_parallel)))
        except (TypeError, ValueError):
            download_parallel = None

    verbose = bool(request.get("stream_verbose", False))
    tag_all = bool(request.get("tag_all", False))
    performance_tuning = bool(request.get("performance_tuning", False))
    if performance_tuning and not tag_all:
        performance_tuning = False
        log.info("tagging_ws performance_tuning ignored (only valid for tag_all runs)")

    service = TaggingService.get_instance(config)
    client = HydrusClient(config.hydrus_api_url, config.hydrus_api_key)

    all_results: list[dict] = []
    total_applied = 0
    total_tags_written = 0
    total_duplicates_skipped = 0
    pending_apply: list[dict] = []
    total = len(file_ids)
    stopped = False
    cumulative_skipped_same_model_marker = 0
    cumulative_skipped_higher_tier_marker = 0
    cumulative_wd_stale_markers_removed = 0
    perf_batch_metrics: list[dict] = []

    effective_batch = _clamp_inference_batch(batch_size, config.batch_size)
    batches_total = (total + effective_batch - 1) // effective_batch if total > 0 else 0
    completed_batches = 0
    resolved_model = model_name or config.default_model

    if active_tagging_sessions_count() > 0:
        busy = get_public_session_status()
        try:
            await websocket.send_json({
                "type": "error",
                "code": "tagging_busy",
                "message": (
                    "Another tab already has an active tagging session. "
                    "Use that tab to stop, pause, or flush; you can browse the gallery here."
                ),
                "snapshot": busy.get("snapshot"),
            })
        except Exception:
            pass
        return

    async def ws_send(payload: dict) -> bool:
        try:
            await websocket.send_json(payload)
        except Exception as e:
            log.debug("tagging_ws send skipped (client gone or closed): %s", e)
            cancel_event.set()
            return False
        t = payload.get("type")
        if t in (
            "progress",
            "file",
            "tags_applied",
            "stopping",
            "server_shutting_down",
            "complete",
            "stopped",
        ):
            update_tagging_public_snapshot(payload, model_name=resolved_model, total_files=total)
        return True

    async def control_listener():
        try:
            while True:
                raw = await websocket.receive_text()
                msg = json.loads(raw)
                action = msg.get("action")
                if action == "cancel":
                    cancel_event.set()
                    log.info(
                        "tagging_ws user_cancel received pending_hydrus_queue=%s "
                        "(winding down: current batch may finish; then final Hydrus flush if any)",
                        len(pending_apply),
                    )
                    await ws_send(
                        {
                            "type": "stopping",
                            "message": (
                                "Stopping — finishing the current inference batch if in progress, "
                                "then flushing any pending Hydrus writes for this run."
                            ),
                            "pending_hydrus_queue": len(pending_apply),
                        }
                    )
                elif action == "pause":
                    pause_event.set()
                    set_controller_paused(True)
                    await ws_send({"type": "control_ack", "action": "pause"})
                elif action == "resume":
                    pause_event.clear()
                    set_controller_paused(False)
                    await ws_send({"type": "control_ack", "action": "resume"})
                elif action == "flush":
                    flush_event.set()
        except WebSocketDisconnect:
            log.info(
                "tagging_ws control_disconnect pending_hydrus_queue=%s (cancel signaled)",
                len(pending_apply),
            )
            cancel_event.set()

    async def wait_while_paused():
        while pause_event.is_set() and not cancel_event.is_set():
            if flush_event.is_set():
                await drain_flush("while_paused")
            await asyncio.sleep(0.05)

    async def drain_flush(context: str) -> None:
        nonlocal pending_apply, total_applied, total_tags_written, total_duplicates_skipped
        if not flush_event.is_set():
            return
        flush_event.clear()
        if not service_key or not pending_apply:
            log.info("tagging_ws flush_%s skipped (no pending or no service)", context)
            await ws_send({
                "type": "tags_applied",
                "count": 0,
                "total_applied": total_applied,
                "chunk_tag_count": 0,
                "chunk_duplicates_skipped": 0,
                "total_tags_written": total_tags_written,
                "total_duplicates_skipped": total_duplicates_skipped,
                "pending_remaining": len(pending_apply),
                "manual_flush": True,
            })
            return
        chunk = list(pending_apply)
        pending_apply.clear()
        n, nt, nd = await _apply_results_chunk(client, service_key, chunk)
        total_applied += n
        total_tags_written += nt
        total_duplicates_skipped += nd
        log.info(
            "tagging_ws hydrus_manual_flush count=%s total_applied=%s context=%s",
            n,
            total_applied,
            context,
        )
        await ws_send({
            "type": "tags_applied",
            "count": n,
            "total_applied": total_applied,
            "chunk_tag_count": nt,
            "chunk_duplicates_skipped": nd,
            "total_tags_written": total_tags_written,
            "total_duplicates_skipped": total_duplicates_skipped,
            "manual_flush": True,
            "pending_remaining": len(pending_apply),
        })

    listener = asyncio.create_task(control_listener())

    session_handle = TaggingSessionHandle(cancel_event=cancel_event, flush_event=flush_event)
    register_tagging_session(session_handle)

    async def notify_server_shutdown():
        await ws_send({
            "type": "server_shutting_down",
            "message": "Server stop requested. Pending Hydrus tags will flush where possible; tagging will cancel.",
        })

    register_shutdown_notifier(notify_server_shutdown)

    preview = file_ids[:8] if isinstance(file_ids, list) else []
    log.info(
        "tagging_ws start files=%d model=%s thresholds=(g=%s,c=%s) inference_batch=%d "
        "apply_every_n=%d dl_parallel=%s verbose=%s tag_all=%s performance_tuning=%s "
        "service_key_set=%s first_file_ids=%s",
        total,
        model_name or config.default_model,
        general_threshold,
        character_threshold,
        effective_batch,
        apply_every,
        download_parallel,
        verbose,
        tag_all,
        performance_tuning,
        bool(service_key),
        preview,
    )

    session_t0 = time.perf_counter()
    model_prepare_wall_s = 0.0
    try:
        t_model = time.perf_counter()
        await service.ensure_model(model_name)
        model_prepare_wall_s = time.perf_counter() - t_model
        log.info(
            "tagging_ws metrics model_prepare_wall_s=%.3f model=%s",
            model_prepare_wall_s,
            model_name or config.default_model,
        )

        meta_chunk = max(32, min(2048, int(config.hydrus_metadata_chunk_size)))
        session_meta_by_id: dict[int, dict] | None
        try:
            session_meta_by_id = await load_metadata_by_file_id(
                client,
                file_ids,
                chunk_sz=meta_chunk,
                cancel_event=cancel_event,
            )
            log.info(
                "tagging_ws metadata_prefetch rows=%s file_ids=%s chunk=%s",
                len(session_meta_by_id),
                len(file_ids),
                meta_chunk,
            )
        except Exception:
            log.exception(
                "tagging_ws metadata_prefetch failed; falling back to per-batch metadata in tag_files",
            )
            session_meta_by_id = None

        for batch_start in range(0, len(file_ids), effective_batch):
            await wait_while_paused()
            if cancel_event.is_set():
                stopped = True
                break

            await drain_flush("between_batches")
            if cancel_event.is_set():
                stopped = True
                break

            batch_ids = file_ids[batch_start:batch_start + effective_batch]
            batch_idx = batch_start // effective_batch + 1
            try:
                batch_results = await service.tag_files(
                    client=client,
                    file_ids=batch_ids,
                    general_threshold=general_threshold,
                    character_threshold=character_threshold,
                    batch_size=len(batch_ids),
                    model_name=model_name,
                    download_parallel=download_parallel,
                    cancel_event=cancel_event,
                    service_key=service_key,
                    batch_metrics_out=perf_batch_metrics if performance_tuning else None,
                    prefetched_meta_by_id=session_meta_by_id,
                )
            except Exception as e:
                log.exception("tagging_ws batch failed batch_idx=%s batch_ids_preview=%s", batch_idx, batch_ids[:8])
                await ws_send({
                    "type": "error",
                    "message": str(e),
                    "partial_results": all_results,
                })
                stopped = True
                break

            all_results.extend(batch_results)
            batch_skipped_same = sum(
                1 for r in batch_results if r.get("skip_reason") == "wd_model_marker_present"
            )
            batch_skipped_higher = sum(
                1 for r in batch_results if r.get("skip_reason") == "wd_skip_higher_tier_model_present"
            )
            batch_stale_rm = sum(int(r.get("wd_stale_markers_removed") or 0) for r in batch_results)
            cumulative_skipped_same_model_marker += batch_skipped_same
            cumulative_skipped_higher_tier_marker += batch_skipped_higher
            cumulative_wd_stale_markers_removed += batch_stale_rm
            if batch_stale_rm:
                log.info(
                    "tagging_ws batch #%s stale_wd_model_markers_removed=%s (session cumulative=%s)",
                    batch_idx,
                    batch_stale_rm,
                    cumulative_wd_stale_markers_removed,
                )
            log.info(
                "tagging_ws inferred batch_size=%d files_inferred=%d cumulative=%d/%d "
                "skipped_same_model_marker_batch=%s skipped_higher_tier_marker_batch=%s",
                effective_batch,
                len(batch_results),
                len(all_results),
                total,
                batch_skipped_same,
                batch_skipped_higher,
            )

            if verbose:
                for r in batch_results:
                    if cancel_event.is_set():
                        stopped = True
                        break
                    if not await ws_send({
                        "type": "file",
                        "current": len(all_results),
                        "total": total,
                        "inference_batch": effective_batch,
                        "batch_inferred": len(batch_results),
                        "batches_completed": batch_idx,
                        "batches_total": batches_total,
                        "total_applied": total_applied,
                        "total_tags_written": total_tags_written,
                        "total_duplicates_skipped": total_duplicates_skipped,
                        "file_id": r.get("file_id"),
                        "result": r,
                    }):
                        stopped = True
                        break
                if stopped:
                    break

            # Short yield so control_listener can read pause / flush / cancel.
            await asyncio.sleep(0.01)

            batch_apply_s = 0.0
            if apply_every > 0 and service_key:
                t_apply_batch = time.perf_counter()
                pending_apply.extend(batch_results)
                while len(pending_apply) >= apply_every:
                    chunk = pending_apply[:apply_every]
                    del pending_apply[:apply_every]
                    n, nt, nd = await _apply_results_chunk(client, service_key, chunk)
                    total_applied += n
                    total_tags_written += nt
                    total_duplicates_skipped += nd
                    log.info(
                        "tagging_ws hydrus_auto_apply count=%s total_applied=%s pending_left=%s",
                        n,
                        total_applied,
                        len(pending_apply),
                    )
                    if not await ws_send({
                        "type": "tags_applied",
                        "count": n,
                        "total_applied": total_applied,
                        "chunk_tag_count": nt,
                        "chunk_duplicates_skipped": nd,
                        "total_tags_written": total_tags_written,
                        "total_duplicates_skipped": total_duplicates_skipped,
                        "manual_flush": False,
                        "pending_remaining": len(pending_apply),
                    }):
                        stopped = True
                        break
                batch_apply_s = time.perf_counter() - t_apply_batch
                if performance_tuning:
                    log.debug(
                        "tagging_ws performance_tuning batch_idx=%s hydrus_apply_wall_s=%.4f",
                        batch_idx,
                        batch_apply_s,
                    )
                if stopped:
                    break

            await drain_flush("after_batch")
            if cancel_event.is_set():
                stopped = True
                break

            batch_skipped = sum(1 for r in batch_results if r.get("skipped_inference"))
            prog_payload: dict = {
                "type": "progress",
                "current": len(all_results),
                "total": total,
                "inference_batch": effective_batch,
                "batch_inferred": len(batch_results),
                "batch_skipped_inference": batch_skipped,
                "batch_predicted": len(batch_results) - batch_skipped,
                "batch_skipped_same_model_marker": batch_skipped_same,
                "batch_skipped_higher_tier_model_marker": batch_skipped_higher,
                "batch_wd_stale_markers_removed": batch_stale_rm,
                "cumulative_skipped_same_model_marker": cumulative_skipped_same_model_marker,
                "cumulative_skipped_higher_tier_model_marker": cumulative_skipped_higher_tier_marker,
                "cumulative_wd_stale_markers_removed": cumulative_wd_stale_markers_removed,
                "batches_completed": batch_idx,
                "batches_total": batches_total,
                "total_applied": total_applied,
                "total_tags_written": total_tags_written,
                "total_duplicates_skipped": total_duplicates_skipped,
            }
            if performance_tuning and perf_batch_metrics:
                last_m = perf_batch_metrics[-1]
                prog_payload["performance_tuning"] = {
                    **last_m,
                    "hydrus_apply_batch_s": round(batch_apply_s, 4),
                    "effective_batch": effective_batch,
                    "download_parallel": download_parallel
                    if download_parallel is not None
                    else config.hydrus_download_parallel,
                }
            if verbose:
                prog_payload["batch_summary"] = True
            else:
                prog_payload["results"] = batch_results
            if not await ws_send(prog_payload):
                stopped = True
                break
            completed_batches += 1

            # Yield before the next batch so cancel / flush can be processed first.
            await asyncio.sleep(0.01)

            if cancel_event.is_set():
                stopped = True
                break

        if stopped:
            log.info(
                "tagging_ws winding_down exit_loop processed_files=%s pending_hydrus_queue=%s",
                len(all_results),
                len(pending_apply),
            )

        if apply_every > 0 and service_key and pending_apply:
            if stopped:
                log.info(
                    "tagging_ws winding_down final_hydrus_flush files=%s",
                    len(pending_apply),
                )
            n, nt, nd = await _apply_results_chunk(client, service_key, pending_apply)
            total_applied += n
            total_tags_written += nt
            total_duplicates_skipped += nd
            log.info(
                "tagging_ws hydrus_final_flush count=%s total_applied=%s",
                n,
                total_applied,
            )
            pending_apply.clear()
            await ws_send({
                "type": "tags_applied",
                "count": n,
                "total_applied": total_applied,
                "chunk_tag_count": nt,
                "chunk_duplicates_skipped": nd,
                "total_tags_written": total_tags_written,
                "total_duplicates_skipped": total_duplicates_skipped,
                "manual_flush": False,
                "pending_remaining": 0,
            })

        pending_hydrus_files = 0
        if service_key and all_results:
            pending_hydrus_files = await _trim_ws_results_to_pending_for_service(
                client, service_key, all_results, config,
            )

        final_type = "stopped" if stopped else "complete"
        final_sent = await ws_send({
            "type": final_type,
            "stopped": stopped,
            "total_processed": len(all_results),
            "total_applied": total_applied,
            "total_tags_written": total_tags_written,
            "total_duplicates_skipped": total_duplicates_skipped,
            "batches_completed": completed_batches,
            "batches_total": batches_total,
            "inference_batch": effective_batch,
            "pending_hydrus_files": pending_hydrus_files,
            "cumulative_skipped_same_model_marker": cumulative_skipped_same_model_marker,
            "cumulative_skipped_higher_tier_model_marker": cumulative_skipped_higher_tier_marker,
            "cumulative_wd_stale_markers_removed": cumulative_wd_stale_markers_removed,
            "results": all_results,
        })
        if final_sent:
            log.info(
                "tagging_ws %s processed=%s applied=%s batches=%s/%s tags_written=%s dups_skipped=%s "
                "skipped_same_model_marker=%s skipped_higher_tier_marker=%s "
                "stale_wd_markers_removed_from_proposals=%s%s",
                final_type,
                len(all_results),
                total_applied,
                completed_batches,
                batches_total,
                total_tags_written,
                total_duplicates_skipped,
                cumulative_skipped_same_model_marker,
                cumulative_skipped_higher_tier_marker,
                cumulative_wd_stale_markers_removed,
                " reason=user_cancel" if stopped else "",
            )
            if stopped:
                log.info(
                    "tagging_ws user_stop_complete processed=%s applied=%s pending_hydrus_after_trim=%s",
                    len(all_results),
                    total_applied,
                    pending_hydrus_files,
                )
            log.info(
                "tagging_ws session_metrics onnx_skipped_same_marker=%s onnx_skipped_higher_tier_marker=%s "
                "hydrus_duplicate_tag_strings_skipped_session=%s "
                "stale_wd_model_markers_dropped_from_proposals=%s "
                "files_processed=%s tags_new_strings_applied=%s",
                cumulative_skipped_same_model_marker,
                cumulative_skipped_higher_tier_marker,
                total_duplicates_skipped,
                cumulative_wd_stale_markers_removed,
                len(all_results),
                total_tags_written,
            )
        record_tagging_session(
            wall_s=time.perf_counter() - session_t0,
            model_prepare_wall_s=model_prepare_wall_s,
            total_processed=len(all_results),
            batches_completed=completed_batches,
            total_applied=total_applied,
            total_tags_written=total_tags_written,
            stopped=stopped,
            outcome="ok",
            model_name=resolved_model,
        )
    except Exception as e:
        record_tagging_session(
            wall_s=time.perf_counter() - session_t0,
            model_prepare_wall_s=model_prepare_wall_s,
            total_processed=len(all_results),
            batches_completed=completed_batches,
            total_applied=total_applied,
            total_tags_written=total_tags_written,
            stopped=True,
            outcome="error",
            model_name=resolved_model,
        )
        log.exception(
            "tagging_ws session failed files=%s model=%s batches_done=%s",
            total,
            model_name or config.default_model,
            completed_batches,
        )
        pending_err = 0
        if service_key and all_results:
            try:
                pending_err = await _trim_ws_results_to_pending_for_service(
                    client, service_key, all_results, config,
                )
            except Exception:
                log.exception("trim_ws_results failed on error path")
        await ws_send({
            "type": "error",
            "message": str(e),
            "partial_results": all_results,
            "pending_hydrus_files": pending_err,
        })
    finally:
        unregister_shutdown_notifier(notify_server_shutdown)
        unregister_tagging_session(session_handle)
        listener.cancel()
        try:
            await listener
        except asyncio.CancelledError:
            pass

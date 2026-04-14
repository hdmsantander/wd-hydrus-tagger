"""WebSocket tagging progress endpoint."""

import asyncio
import gc
import json
import logging
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.config import clamp_hydrus_metadata_chunk_size, get_config
from backend.log_stats import log_stats
from backend.hydrus.client import HydrusClient
from backend.hydrus.transport_errors import is_hydrus_transport_error
from backend.perf_metrics import peak_rss_mb, record_tagging_session
from backend.routes.tagger_apply import _apply_results_chunk, _trim_ws_results_to_pending_for_service
from backend.routes.tagger_ws_transport import (
    wait_until_hydrus_responsive as _wait_until_hydrus_responsive_transport,
    ws_send_json_ignore_closed as _ws_send_json_ignore_closed_transport,
)
from backend.services.tagging_queue_analysis import (
    analyze_prefetched_queue,
    reorder_work_ids_inference_first,
)
from backend.services.tagging_service import TaggingService
from backend.services.tagging_shared import clamp_inference_batch, load_metadata_by_file_id
from backend.services.session_autotune import (
    SessionAutoTune,
    clamp_supervised_timeout_s,
    normalize_tuning_control_mode,
    resolve_intra_thread_bounds,
    resolve_tuning_bounds,
)
from backend.services.learning_calibration import (
    compute_learning_split,
    compute_learning_split_by_bytes,
    parse_learning_fraction,
)
from backend.services.tuning_observability import (
    build_tuning_report,
    clamp_performance_tuning_window,
    merge_performance_tuning_row,
)
from backend.services.performance_results_store import save_performance_results
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

# Client disconnected or socket already closed — safe to ignore for best-effort sends.
_WS_SEND_CLIENT_GONE = (WebSocketDisconnect, OSError, RuntimeError)


async def _ws_send_json_ignore_closed(ws: WebSocket, payload: dict) -> None:
    """Backward-compatible wrapper around extracted transport helper."""
    await _ws_send_json_ignore_closed_transport(ws, payload)


async def _wait_until_hydrus_responsive(
    *,
    client: HydrusClient,
    cancel_event: asyncio.Event,
    hydrus_manual_retry: asyncio.Event,
    ws_send,
    last_error: str,
    snapshot: dict,
    poll_s: float = 12.0,
) -> bool:
    """Backward-compatible wrapper around extracted transport helper."""
    return await _wait_until_hydrus_responsive_transport(
        client=client,
        cancel_event=cancel_event,
        hydrus_manual_retry=hydrus_manual_retry,
        ws_send=ws_send,
        last_error=last_error,
        snapshot=snapshot,
        poll_s=poll_s,
    )


@router.websocket("/ws/progress")
async def progress_ws(websocket: WebSocket):
    """WebSocket: batched tagging, pause/resume, manual flush to Hydrus, cancel."""
    await websocket.accept()
    cancel_event = asyncio.Event()
    pause_event = asyncio.Event()
    flush_event = asyncio.Event()
    tuning_ack_event = asyncio.Event()
    hydrus_manual_retry = asyncio.Event()

    try:
        first = await websocket.receive_text()
        request = json.loads(first)
    except WebSocketDisconnect:
        log.info("tagging_ws client disconnected before first message")
        return

    if request.get("action") not in (None, "run"):
        await _ws_send_json_ignore_closed(
            websocket,
            {"type": "error", "message": "expected action run"},
        )
        return

    config = get_config()

    def _record_validation_reject(outcome: str) -> None:
        record_tagging_session(
            wall_s=0.0,
            model_prepare_wall_s=0.0,
            total_processed=0,
            batches_completed=0,
            total_applied=0,
            total_tags_written=0,
            stopped=False,
            outcome=outcome,
            model_name=config.default_model,
        )

    file_ids = request.get("file_ids", [])
    if file_ids is None:
        file_ids = []
    if not isinstance(file_ids, list):
        await _ws_send_json_ignore_closed(
            websocket,
            {
                "type": "error",
                "message": "file_ids must be a JSON array",
                "code": "invalid_file_ids",
            },
        )
        _record_validation_reject("invalid_request")
        return
    try:
        file_ids = [int(x) for x in file_ids]
    except (TypeError, ValueError):
        await _ws_send_json_ignore_closed(
            websocket,
            {
                "type": "error",
                "message": "file_ids must be integers",
                "code": "invalid_file_ids",
            },
        )
        _record_validation_reject("invalid_request")
        return
    if len(file_ids) == 0:
        await _ws_send_json_ignore_closed(
            websocket,
            {
                "type": "error",
                "message": "No files to tag (file_ids is empty)",
                "code": "empty_queue",
            },
        )
        _record_validation_reject("empty_queue")
        return

    general_threshold = float(request.get("general_threshold", config.general_threshold))
    character_threshold = float(request.get("character_threshold", config.character_threshold))
    model_name = request.get("model_name") or None
    raw_bs = request.get("batch_size")
    batch_size = None
    if raw_bs is not None:
        try:
            batch_size = clamp_inference_batch(int(raw_bs), config.batch_size)
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

    session_auto_tune = bool(request.get("session_auto_tune", False))
    if session_auto_tune and not tag_all:
        session_auto_tune = False
        log.info("tagging_ws session_auto_tune ignored (requires tag_all)")
    if session_auto_tune and not bool(request.get("performance_tuning", False)):
        log.debug(
            "tagging_ws session_auto_tune implies performance_tuning for this session",
        )
        performance_tuning = True

    tuning_control_mode = "auto_lucky"
    tuning_bound_warnings: list[str] = []
    bounds = None
    autotune: SessionAutoTune | None = None
    supervised_timeout_s: float | None = None
    session_auto_tune_threads_req = bool(request.get("session_auto_tune_threads", False))
    tune_threads_effective = (
        session_auto_tune
        and session_auto_tune_threads_req
        and not bool(config.use_gpu)
    )
    if session_auto_tune_threads_req and config.use_gpu:
        log.info(
            "tagging_ws session_auto_tune_threads ignored (GPU inference — CPU thread search N/A)",
        )
    intra_bounds_tuple: tuple[int, int] | None = None
    if session_auto_tune:
        tuning_control_mode, mode_invalid = normalize_tuning_control_mode(
            request.get("tuning_control_mode"),
        )
        if mode_invalid:
            log.info("tagging_ws invalid tuning_control_mode; using auto_lucky")
        bounds, tuning_bound_warnings = resolve_tuning_bounds(
            config,
            request.get("tuning_bounds"),
        )
        supervised_timeout_s = clamp_supervised_timeout_s(
            request.get("tuning_supervised_timeout_s"),
        )
        if tune_threads_effective:
            ilo, ihi, iw = resolve_intra_thread_bounds(
                config,
                request.get("tuning_bounds"),
                default_hi=None,
            )
            intra_bounds_tuple = (ilo, ihi)
            tuning_bound_warnings.extend(iw)

    def _ws_opt_int(key: str) -> int | None:
        v = request.get(key)
        if v is None:
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            log.info("tagging_ws ignored invalid %s=%r", key, v)
            return None

    ws_ort_intra = _ws_opt_int("cpu_intra_op_threads")
    ws_ort_inter = _ws_opt_int("cpu_inter_op_threads")

    service = TaggingService.get_instance(config)
    baseline_ort_intra, baseline_ort_inter = service._resolve_ort_threads(ws_ort_intra, ws_ort_inter)

    client = HydrusClient(config.hydrus_api_url, config.hydrus_api_key)

    build_hydrus_recovery_snapshot: object | None = None

    learning_phase_calibration = bool(request.get("learning_phase_calibration", False))
    if learning_phase_calibration and not tag_all:
        learning_phase_calibration = False
        log.info("tagging_ws learning_phase_calibration ignored (requires tag_all)")
    learning_split_info: dict = {}
    work_ids: list[int] = list(file_ids)
    split_at = 0
    learning_rows_pending_flush: list[dict] = []

    all_results: list[dict] = []
    total_applied = 0
    total_tags_written = 0
    total_duplicates_skipped = 0
    pending_apply: list[dict] = []
    total = len(work_ids)
    stopped = False
    cumulative_skipped_same_model_marker = 0
    cumulative_skipped_higher_tier_marker = 0
    cumulative_wd_stale_markers_removed = 0
    perf_batch_metrics: list[dict] = []
    perf_tuning_series: list[dict] = []
    pt_history_window = (
        clamp_performance_tuning_window(request.get("performance_tuning_window"))
        if performance_tuning
        else 32
    )
    autotune_phase_logged: str | None = None
    autotune_prev_phase: str | None = None

    effective_batch = clamp_inference_batch(batch_size, config.batch_size)
    if session_auto_tune and bounds is not None:
        dlp_base = download_parallel if download_parallel is not None else config.hydrus_download_parallel
        autotune = SessionAutoTune(
            mode=tuning_control_mode,
            baseline_batch=effective_batch,
            baseline_dlp=int(dlp_base),
            bounds=bounds,
            tune_threads=tune_threads_effective and intra_bounds_tuple is not None,
            baseline_ort_intra=baseline_ort_intra,
            baseline_ort_inter=baseline_ort_inter,
            intra_bounds=intra_bounds_tuple,
        )
    next_bs = effective_batch
    next_dlp = download_parallel if download_parallel is not None else config.hydrus_download_parallel
    next_ort_intra = baseline_ort_intra
    next_ort_inter = baseline_ort_inter
    loaded_ort = (baseline_ort_intra, baseline_ort_inter)
    ort_reload_count = 0
    batches_total = (total + next_bs - 1) // next_bs if total > 0 else 0
    completed_batches = 0
    supervised_gates_passed = 0
    resolved_model = model_name or config.default_model

    _meta_chunk_cfg = clamp_hydrus_metadata_chunk_size(config.hydrus_metadata_chunk_size)
    log.info(
        "tagging_ws session_config apply_tags_http_batch=%s hydrus_metadata_chunk=%s "
        "ort_cpu_threads_intra_inter=%s/%s config_inference_batch_saved=%s "
        "apply_tags_every_n_effective=%s hydrus_download_parallel_effective=%s",
        config.apply_tags_http_batch_size,
        _meta_chunk_cfg,
        baseline_ort_intra,
        baseline_ort_inter,
        config.batch_size,
        apply_every,
        next_dlp,
    )
    log.debug(
        "tagging_ws session_profile tag_all=%s queue_files=%s effective_inference_batch=%s "
        "estimated_outer_batches=%s note=apply_tags_http_batch_is_for_http_apply_route_only",
        tag_all,
        total,
        effective_batch,
        batches_total,
    )

    if active_tagging_sessions_count() > 0:
        busy = get_public_session_status()
        await _ws_send_json_ignore_closed(
            websocket,
            {
                "type": "error",
                "code": "tagging_busy",
                "message": (
                    "Another tab already has an active tagging session. "
                    "Use that tab to stop, pause, or flush; you can browse the gallery here."
                ),
                "snapshot": busy.get("snapshot"),
            },
        )
        return

    async def ws_send(payload: dict) -> bool:
        try:
            await websocket.send_json(payload)
        except _WS_SEND_CLIENT_GONE as e:
            log.debug("tagging_ws send skipped (client gone or closed): %s", e)
            cancel_event.set()
            return False
        except Exception:
            log.exception(
                "tagging_ws send_json unexpected error payload_type=%s",
                payload.get("type"),
            )
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
            "queue_plan",
        ):
            update_tagging_public_snapshot(payload, model_name=resolved_model, total_files=total)
        return True

    async def _apply_chunk_with_recovery(
        chunk: list[dict],
        *,
        apply_sz: int,
        restore_to_pending: bool = True,
    ) -> tuple[int, int, int]:
        """Apply tags to Hydrus; on transport errors wait for Hydrus to return (poll + UI retry)."""
        nonlocal pending_apply, stopped
        while True:
            try:
                return await _apply_results_chunk(client, service_key, chunk)
            except Exception as e:
                if not is_hydrus_transport_error(e) or cancel_event.is_set():
                    if restore_to_pending:
                        pending_apply = chunk + pending_apply
                    raise
                if restore_to_pending:
                    pending_apply = chunk + pending_apply
                log.warning("tagging_ws Hydrus transport error during apply: %s", e)
                snap_fn = build_hydrus_recovery_snapshot
                snap = snap_fn() if callable(snap_fn) else {}
                ok = await _wait_until_hydrus_responsive(
                    client=client,
                    cancel_event=cancel_event,
                    hydrus_manual_retry=hydrus_manual_retry,
                    ws_send=ws_send,
                    last_error=str(e),
                    snapshot=snap,
                )
                if not ok or cancel_event.is_set():
                    stopped = True
                    raise RuntimeError("Hydrus unavailable; tagging stopped") from e
                if restore_to_pending:
                    chunk = pending_apply[:apply_sz]
                    del pending_apply[:apply_sz]

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
                    tuning_ack_event.set()
                    await ws_send({"type": "control_ack", "action": "resume"})
                elif action == "tuning_ack":
                    tuning_ack_event.set()
                    await ws_send({"type": "control_ack", "action": "tuning_ack"})
                elif action == "flush":
                    flush_event.set()
                elif action == "retry_hydrus":
                    hydrus_manual_retry.set()
                    await ws_send({"type": "control_ack", "action": "retry_hydrus"})
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
        nonlocal pending_apply, total_applied, total_tags_written, total_duplicates_skipped, stopped
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
        apply_sz = len(chunk)
        try:
            n, nt, nd = await _apply_chunk_with_recovery(chunk, apply_sz=apply_sz)
        except RuntimeError:
            stopped = True
            return
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

    preview = file_ids[:8]
    log.info(
        "tagging_ws start files=%d model=%s thresholds=(g=%s,c=%s) inference_batch=%d "
        "apply_every_n=%d dl_parallel=%s verbose=%s tag_all=%s performance_tuning=%s "
        "session_auto_tune=%s service_key_set=%s ort_intra=%s ort_inter=%s first_file_ids=%s",
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
        session_auto_tune,
        bool(service_key),
        ws_ort_intra,
        ws_ort_inter,
        preview,
    )

    session_t0 = time.perf_counter()
    model_prepare_wall_s = 0.0
    try:
        t_model = time.perf_counter()
        await service.ensure_model(
            model_name,
            ort_intra_op_threads=ws_ort_intra,
            ort_inter_op_threads=ws_ort_inter,
        )
        model_prepare_wall_s = time.perf_counter() - t_model
        log.info(
            "tagging_ws metrics model_prepare_wall_s=%.3f model=%s",
            model_prepare_wall_s,
            model_name or config.default_model,
        )

        meta_chunk = clamp_hydrus_metadata_chunk_size(config.hydrus_metadata_chunk_size)
        session_meta_by_id: dict[int, dict] | None
        try:
            t_prefetch = time.perf_counter()
            session_meta_by_id = await load_metadata_by_file_id(
                client,
                file_ids,
                chunk_sz=meta_chunk,
                cancel_event=cancel_event,
            )
            prefetch_wall_s = time.perf_counter() - t_prefetch
            log.info(
                "tagging_ws metadata_prefetch wall_s=%.3f rows=%s file_ids=%s chunk=%s",
                prefetch_wall_s,
                len(session_meta_by_id),
                len(file_ids),
                meta_chunk,
            )
        except Exception:
            log.exception(
                "tagging_ws metadata_prefetch failed; falling back to per-batch metadata in tag_files",
            )
            session_meta_by_id = None

        if learning_phase_calibration:
            lf = parse_learning_fraction(request.get("learning_fraction"))
            lscope = str(request.get("learning_scope") or "count").strip().lower()
            if lscope == "bytes":
                learn_part, commit_part, learning_split_info = compute_learning_split_by_bytes(
                    file_ids,
                    meta_by_id=session_meta_by_id,
                    learning_fraction=lf,
                )
            else:
                learn_part, commit_part, learning_split_info = compute_learning_split(
                    file_ids,
                    learning_fraction=lf,
                    learning_scope=lscope if lscope in ("count", "bytes") else "count",
                )
            max_c = int(config.max_learning_cached_files)
            if len(learn_part) > max_c:
                overflow = learn_part[max_c:]
                learn_part = learn_part[:max_c]
                commit_part = overflow + commit_part
                learning_split_info["learning_prefix_capped"] = max_c
                log.warning(
                    "tagging_ws learning_calibration prefix capped to max_learning_cached_files=%s",
                    max_c,
                )
            work_ids = learn_part + commit_part
            split_at = len(learn_part)
            log.info(
                "tagging_ws learning_calibration split learning=%s commit=%s scope=%s fraction=%s "
                "bytes_fallback=%s prefix_capped=%s",
                split_at,
                len(commit_part),
                learning_split_info.get("learning_scope_effective"),
                learning_split_info.get("learning_fraction"),
                learning_split_info.get("bytes_fallback"),
                learning_split_info.get("learning_prefix_capped"),
            )

        total = len(work_ids)
        resolved_model_ws = (model_name or "").strip() or config.default_model
        queue_counts = None
        infer_cut = 0
        if session_meta_by_id is not None and len(work_ids) > 0:
            q_counts = analyze_prefetched_queue(
                work_ids,
                session_meta_by_id,
                resolved_model=resolved_model_ws,
                config=config,
                service_key=service_key,
            )
            queue_counts = q_counts
            log.info(
                "tagging_ws queue_analysis infer=%s skip_same_marker=%s skip_higher_tier=%s "
                "missing_metadata=%s queue_total=%s reorder=infer_first "
                "(optional Hydrus filter: refine gallery search; API search_files tags JSON — "
                "last resort: negative system predicates if your client supports them)",
                q_counts.infer,
                q_counts.skip_same_marker,
                q_counts.skip_higher_tier,
                q_counts.missing_metadata,
                len(work_ids),
            )
            work_ids = reorder_work_ids_inference_first(
                work_ids,
                session_meta_by_id,
                resolved_model=resolved_model_ws,
                config=config,
                service_key=service_key,
            )
            total = len(work_ids)
            infer_cut = int(q_counts.infer)
            await ws_send(
                {
                    "type": "queue_plan",
                    "queue_total": total,
                    "infer_total": q_counts.infer,
                    "skip_same_marker": q_counts.skip_same_marker,
                    "skip_higher_tier": q_counts.skip_higher_tier,
                    "missing_metadata": q_counts.missing_metadata,
                    "infer_first": True,
                    "metadata_chunk_used": meta_chunk,
                }
            )

        def _hydrus_snap():
            rem_infer = work_ids[len(all_results) :] if len(all_results) <= len(work_ids) else []
            pids = [int(r["file_id"]) for r in pending_apply if r.get("file_id") is not None]
            return {
                "pending_commit_count": len(pending_apply),
                "pending_commit_file_ids": pids[:500],
                "remaining_infer_count": len(rem_infer),
                "remaining_infer_file_ids": rem_infer[:500],
                "inferred_so_far": len(all_results),
                "total_queue": len(work_ids),
            }

        build_hydrus_recovery_snapshot = _hydrus_snap

        skip_tail_cap = max(32, min(2048, int(config.tagging_skip_tail_batch_size)))
        cumulative_inferred_non_skip = 0
        pos = 0
        batch_idx = 0
        first_onnx_batch_logged = False
        while pos < len(work_ids):
            await wait_while_paused()
            if cancel_event.is_set():
                stopped = True
                break

            await drain_flush("between_batches")
            if cancel_event.is_set():
                stopped = True
                break

            batch_idx += 1
            iter_pos_start = pos
            if learning_phase_calibration and autotune is not None and iter_pos_start == split_at and split_at > 0:
                next_bs, next_dlp = autotune.best_pair
                next_ort_intra, next_ort_inter = autotune.best_ort_threads
                log.info(
                    "tagging_ws learning_calibration_commit_lock bs=%s dlp=%s ort_intra=%s ort_inter=%s "
                    "autotune_phase=%s (apply locked knobs for remainder of queue; incremental Hydrus on)",
                    next_bs,
                    next_dlp,
                    next_ort_intra,
                    next_ort_inter,
                    autotune.phase,
                )
            ort_intra_now = next_ort_intra
            ort_inter_now = next_ort_inter
            if (ort_intra_now, ort_inter_now) != loaded_ort:
                log.info(
                    "tagging_ws ort_session_reload target_intra=%s target_inter=%s previous=%s",
                    ort_intra_now,
                    ort_inter_now,
                    loaded_ort,
                )
                await service.ensure_model(
                    model_name,
                    ort_intra_op_threads=ort_intra_now,
                    ort_inter_op_threads=ort_inter_now,
                )
                ort_reload_count += 1
                loaded_ort = (ort_intra_now, ort_inter_now)
            bs_now = next_bs
            dlp_now = next_dlp
            rem_all = len(work_ids) - pos
            use_skip_tail_bulk = (
                tag_all
                and infer_cut > 0
                and pos >= infer_cut
                and rem_all > 0
            )
            if use_skip_tail_bulk:
                take = min(max(bs_now, min(skip_tail_cap, rem_all)), rem_all)
                if iter_pos_start == infer_cut:
                    log.info(
                        "tagging_ws skip_tail_fast start pos=%s outer_batch=%s cap=%s "
                        "(marker-skip / higher-tier only; ONNX starts earlier in queue via infer_first)",
                        iter_pos_start,
                        take,
                        skip_tail_cap,
                    )
            else:
                take = bs_now
                if learning_phase_calibration and split_at > 0 and pos < split_at < pos + bs_now:
                    take = split_at - pos
            batch_ids = work_ids[pos : pos + take]
            if not batch_ids:
                break
            in_learning_phase = bool(
                learning_phase_calibration and split_at > 0 and iter_pos_start < split_at
            )
            commit_seg = bool(
                learning_phase_calibration and split_at > 0 and iter_pos_start >= split_at
            )
            if learning_phase_calibration and split_at > 0 and in_learning_phase and batch_idx == 1:
                log.info(
                    "tagging_ws learning_calibration segment=L first_batch: "
                    "session_auto_tune=%s (explores knobs on this segment; Hydrus writes deferred until commit)",
                    session_auto_tune,
                )
            pos_after = pos + len(batch_ids)
            rem_pre = total - pos_after
            if rem_pre > 0:
                batches_total = batch_idx + (rem_pre + next_bs - 1) // max(1, next_bs)
            else:
                batches_total = batch_idx
            batch_results = None
            while batch_results is None:
                try:
                    tf_kw: dict = {}
                    if use_skip_tail_bulk:
                        tf_kw["outer_batch_override"] = min(len(batch_ids), skip_tail_cap)
                    batch_results = await service.tag_files(
                        client=client,
                        file_ids=batch_ids,
                        general_threshold=general_threshold,
                        character_threshold=character_threshold,
                        batch_size=len(batch_ids),
                        model_name=model_name,
                        download_parallel=dlp_now,
                        cancel_event=cancel_event,
                        service_key=service_key,
                        batch_metrics_out=perf_batch_metrics if performance_tuning else None,
                        prefetched_meta_by_id=session_meta_by_id,
                        **tf_kw,
                    )
                except Exception as e:
                    if is_hydrus_transport_error(e) and not cancel_event.is_set():
                        log.warning(
                            "tagging_ws tag_files transport error batch_idx=%s: %s",
                            batch_idx,
                            e,
                        )
                        snap_fn = build_hydrus_recovery_snapshot
                        snap = snap_fn() if callable(snap_fn) else {}
                        ok = await _wait_until_hydrus_responsive(
                            client=client,
                            cancel_event=cancel_event,
                            hydrus_manual_retry=hydrus_manual_retry,
                            ws_send=ws_send,
                            last_error=str(e),
                            snapshot=snap,
                        )
                        if ok and not cancel_event.is_set():
                            continue
                        stopped = True
                        break
                    log.exception(
                        "tagging_ws batch failed batch_idx=%s batch_ids_preview=%s",
                        batch_idx,
                        batch_ids[:8],
                    )
                    await ws_send({
                        "type": "error",
                        "message": str(e),
                        "partial_results": all_results,
                    })
                    stopped = True
                    break
            if stopped or batch_results is None:
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
            infer_non_marker = len(batch_results) - batch_skipped_same - batch_skipped_higher
            cumulative_inferred_non_skip += infer_non_marker
            if not first_onnx_batch_logged and infer_non_marker > 0:
                first_onnx_batch_logged = True
                log.info(
                    "tagging_ws first_onnx_inference_batch batch_idx=%s elapsed_since_session_s=%.3f "
                    "onnx_inferred_in_batch=%s cumulative_skipped_same_model_marker=%s",
                    batch_idx,
                    time.perf_counter() - session_t0,
                    infer_non_marker,
                    cumulative_skipped_same_model_marker,
                )
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
                bs_now,
                len(batch_results),
                len(all_results),
                total,
                batch_skipped_same,
                batch_skipped_higher,
            )

            # Short yield so control_listener can read pause / flush / cancel.
            await asyncio.sleep(0.01)

            # §4.3: Tag all uses apply chunk size = current inference batch so apply wall matches batch granularity.
            apply_every_run = apply_every
            if tag_all and apply_every > 0:
                apply_every_run = bs_now

            batch_apply_s = 0.0
            suppress_incremental_hydrus = (
                learning_phase_calibration and in_learning_phase and tag_all and bool(service_key)
            )
            if not suppress_incremental_hydrus and apply_every_run > 0 and service_key:
                t_apply_batch = time.perf_counter()
                pending_apply.extend(batch_results)
                while len(pending_apply) >= apply_every_run:
                    chunk = pending_apply[:apply_every_run]
                    del pending_apply[:apply_every_run]
                    try:
                        n, nt, nd = await _apply_chunk_with_recovery(chunk, apply_sz=apply_every_run)
                    except RuntimeError:
                        stopped = True
                        break
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
            elif suppress_incremental_hydrus:
                learning_rows_pending_flush.extend(batch_results)

            await drain_flush("after_batch")
            if cancel_event.is_set():
                stopped = True
                break

            batch_skipped = sum(1 for r in batch_results if r.get("skipped_inference"))
            row = None
            ab = None
            if performance_tuning and perf_batch_metrics:
                last_m = perf_batch_metrics[-1]
                rss_sample = peak_rss_mb() if batch_idx > 0 and batch_idx % 8 == 0 else None
                row = merge_performance_tuning_row(
                    last_m,
                    hydrus_apply_batch_s=batch_apply_s,
                    effective_batch=bs_now,
                    download_parallel=dlp_now,
                    peak_rss_mb_sample=rss_sample,
                    ort_intra_op_threads=ort_intra_now,
                    ort_inter_op_threads=ort_inter_now,
                )
                perf_tuning_series.append(row)
                if autotune is not None:
                    if not commit_seg:
                        ab = autotune.after_batch(row)
                        next_bs = ab.next_batch_size
                        next_dlp = ab.next_download_parallel
                        next_ort_intra = ab.next_ort_intra
                        next_ort_inter = ab.next_ort_inter
                        cur_ph = autotune.phase
                        if autotune_prev_phase is not None and cur_ph != autotune_prev_phase:
                            log.info(
                                "tagging_ws session_autotune sequential_transition %s -> %s; gc.collect()",
                                autotune_prev_phase,
                                cur_ph,
                            )
                            gc.collect()
                            if autotune_prev_phase == "explore_dlp" and cur_ph == "explore_intra":
                                log.info(
                                    "tagging_ws session_autotune unload_onnx_before_thread_sweep "
                                    "(sequential phase; model reloads next batch)",
                                )
                                await asyncio.to_thread(TaggingService.unload_model_from_memory)
                        autotune_prev_phase = cur_ph
                    else:
                        ab = None

            rem2 = total - pos_after
            if rem2 > 0:
                batches_total = batch_idx + (rem2 + next_bs - 1) // max(1, next_bs)
            else:
                batches_total = batch_idx

            infer_total_q = int(queue_counts.infer) if queue_counts is not None else 0
            tail_total_q = max(0, total - infer_total_q)
            if infer_total_q > 0:
                if cumulative_inferred_non_skip < infer_total_q:
                    pbc, pbt = cumulative_inferred_non_skip, infer_total_q
                else:
                    tail_done = max(0, len(all_results) - infer_total_q)
                    pbc, pbt = tail_done, max(1, tail_total_q)
            else:
                pbc, pbt = len(all_results), total
            prog_payload: dict = {
                "type": "progress",
                "current": len(all_results),
                "total": total,
                "infer_total": infer_total_q,
                "cumulative_inferred_non_skip": cumulative_inferred_non_skip,
                "in_marker_skip_tail": bool(infer_cut > 0 and iter_pos_start >= infer_cut),
                "progress_bar_current": pbc,
                "progress_bar_total": pbt,
                "inference_batch": bs_now,
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
            if performance_tuning and row is not None:
                prog_payload["performance_tuning"] = row
                prog_payload["performance_tuning_history"] = perf_tuning_series[
                    -pt_history_window:
                ]
            if session_auto_tune and autotune is not None:
                if commit_seg:
                    prog_payload["tuning_state"] = autotune.ui_snapshot_commit_phase()
                elif ab is not None:
                    prog_payload["tuning_state"] = ab.tuning_state
                    ts = ab.tuning_state
                    ph = str(ts.get("phase") or "")
                    if ph and ph != autotune_phase_logged:
                        autotune_phase_logged = ph
                        log.info(
                            "tagging_ws session_autotune tuning_phase=%s batch_idx=%s "
                            "learning_segment=%s next_bs=%s next_dlp=%s best_bs=%s best_dlp=%s",
                            ph,
                            batch_idx,
                            in_learning_phase,
                            ts.get("next_batch_size"),
                            ts.get("next_download_parallel"),
                            ts.get("best_batch_size"),
                            ts.get("best_download_parallel"),
                        )
                ts_merge = prog_payload.get("tuning_state")
                if ts_merge is not None:
                    prog_payload["tuning_state"] = autotune.merge_progress_ui_fields(
                        ts_merge,
                        perf_tuning_series,
                        commit_segment=commit_seg,
                    )
            if learning_phase_calibration:
                prog_payload["calibration_phase"] = "learning" if in_learning_phase else "commit"
            if batch_idx == 1 and tuning_bound_warnings:
                prog_payload["tuning_warnings"] = list(tuning_bound_warnings)
            if verbose:
                prog_payload["batch_summary"] = True
            else:
                prog_payload["results"] = batch_results
            ts_for_files = prog_payload.get("tuning_state")
            cal_for_files = prog_payload.get("calibration_phase")
            if verbose:
                for r in batch_results:
                    if cancel_event.is_set():
                        stopped = True
                        break
                    file_msg: dict = {
                        "type": "file",
                        "current": len(all_results),
                        "total": total,
                        "inference_batch": bs_now,
                        "batch_inferred": len(batch_results),
                        "batches_completed": batch_idx,
                        "batches_total": batches_total,
                        "total_applied": total_applied,
                        "total_tags_written": total_tags_written,
                        "total_duplicates_skipped": total_duplicates_skipped,
                        "file_id": r.get("file_id"),
                        "result": r,
                    }
                    if ts_for_files is not None:
                        file_msg["tuning_state"] = ts_for_files
                    if cal_for_files is not None:
                        file_msg["calibration_phase"] = cal_for_files
                    if not await ws_send(file_msg):
                        stopped = True
                        break
                if stopped:
                    break
            if not await ws_send(prog_payload):
                stopped = True
                break

            if ab is not None and ab.require_ack_before_next:
                tuning_ack_event.clear()
                try:
                    if supervised_timeout_s is not None:
                        await asyncio.wait_for(
                            tuning_ack_event.wait(),
                            supervised_timeout_s,
                        )
                    else:
                        await tuning_ack_event.wait()
                    supervised_gates_passed += 1
                except asyncio.TimeoutError:
                    pause_event.set()
                    set_controller_paused(True)
                    await ws_send({
                        "type": "tuning_timeout",
                        "message": (
                            "Supervised tuning approval timed out; paused. "
                            "Resume to approve the pending knob change or cancel."
                        ),
                    })
                    await wait_while_paused()
                if cancel_event.is_set():
                    stopped = True
                    break

            pos += len(batch_ids)
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

        if (
            learning_phase_calibration
            and service_key
            and learning_rows_pending_flush
            and not stopped
        ):
            flush_bs = max(1, int(next_bs))
            p = learning_rows_pending_flush
            off = 0
            while off < len(p):
                if cancel_event.is_set():
                    break
                chunk = p[off : off + flush_bs]
                try:
                    n, nt, nd = await _apply_chunk_with_recovery(
                        chunk,
                        apply_sz=len(chunk),
                        restore_to_pending=False,
                    )
                except RuntimeError:
                    stopped = True
                    break
                off += len(chunk)
                total_applied += n
                total_tags_written += nt
                total_duplicates_skipped += nd
                log.info(
                    "tagging_ws learning_calibration_flush count=%s total_applied=%s off=%s/%s",
                    n,
                    total_applied,
                    off,
                    len(p),
                )
                await ws_send({
                    "type": "tags_applied",
                    "count": n,
                    "total_applied": total_applied,
                    "chunk_tag_count": nt,
                    "chunk_duplicates_skipped": nd,
                    "total_tags_written": total_tags_written,
                    "total_duplicates_skipped": total_duplicates_skipped,
                    "manual_flush": False,
                    "pending_remaining": len(pending_apply),
                    "learning_calibration_flush": True,
                })
            if not cancel_event.is_set():
                learning_rows_pending_flush.clear()

        if apply_every > 0 and service_key and pending_apply:
            if stopped:
                log.info(
                    "tagging_ws winding_down final_hydrus_flush files=%s",
                    len(pending_apply),
                )
            chunk = list(pending_apply)
            pending_apply.clear()
            try:
                n, nt, nd = await _apply_chunk_with_recovery(
                    chunk,
                    apply_sz=len(chunk),
                    restore_to_pending=True,
                )
            except RuntimeError:
                n, nt, nd = 0, 0, 0
            total_applied += n
            total_tags_written += nt
            total_duplicates_skipped += nd
            log.info(
                "tagging_ws hydrus_final_flush count=%s total_applied=%s",
                n,
                total_applied,
            )
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
        if session_auto_tune and autotune is not None:
            eff_final, dlp_final = autotune.best_pair
        else:
            eff_final = next_bs
            dlp_final = (
                download_parallel
                if download_parallel is not None
                else config.hydrus_download_parallel
            )
        final_payload: dict = {
            "type": final_type,
            "stopped": stopped,
            "total_processed": len(all_results),
            "total_applied": total_applied,
            "total_tags_written": total_tags_written,
            "total_duplicates_skipped": total_duplicates_skipped,
            "batches_completed": completed_batches,
            "batches_total": batches_total,
            "inference_batch": eff_final,
            "pending_hydrus_files": pending_hydrus_files,
            "cumulative_skipped_same_model_marker": cumulative_skipped_same_model_marker,
            "cumulative_skipped_higher_tier_model_marker": cumulative_skipped_higher_tier_marker,
            "cumulative_wd_stale_markers_removed": cumulative_wd_stale_markers_removed,
            "results": all_results,
        }
        if learning_phase_calibration and learning_split_info:
            final_payload["learning_calibration"] = dict(learning_split_info)
        if performance_tuning:
            at_summary = None
            if autotune is not None:
                at_summary = dict(autotune.summary_for_report())
                at_summary["ort_session_reloads"] = ort_reload_count
            if learning_phase_calibration and learning_split_info:
                if at_summary is None:
                    at_summary = {}
                else:
                    at_summary = dict(at_summary)
                at_summary["learning_calibration"] = dict(learning_split_info)
            final_payload["tuning_report"] = build_tuning_report(
                perf_tuning_series,
                stopped=stopped,
                batches_completed=completed_batches,
                total_processed=len(all_results),
                effective_batch=int(eff_final),
                download_parallel=int(dlp_final),
                model_name=resolved_model,
                history_window=pt_history_window,
                session_auto_tune=session_auto_tune,
                tuning_control_mode=tuning_control_mode if session_auto_tune else None,
                supervised_gates_passed=supervised_gates_passed,
                autotune_summary=at_summary,
            )
        final_sent = await ws_send(final_payload)
        if final_sent:
            if (
                final_type == "complete"
                and session_auto_tune
                and autotune is not None
                and not stopped
            ):
                bi, bo = autotune.best_ort_threads
                bb, bd = autotune.best_pair
                save_performance_results(
                    model_name=str(resolved_model),
                    best_batch=int(bb),
                    best_dlp=int(bd),
                    best_intra=int(bi),
                    best_inter=int(bo),
                    tune_threads=tune_threads_effective,
                    tuning_control_mode=tuning_control_mode,
                    autotune_phase=str(autotune.phase),
                )
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
            wall_total = time.perf_counter() - session_t0
            inferred_non_skip = max(
                0,
                len(all_results)
                - cumulative_skipped_same_model_marker
                - cumulative_skipped_higher_tier_marker,
            )
            log.debug(
                "tagging_ws session_perf_rates wall_s=%.3f inferred_non_skip=%s "
                "inferred_non_skip_per_s=%.4f tags_new_strings_per_s=%.4f outer_batches=%s",
                wall_total,
                inferred_non_skip,
                inferred_non_skip / wall_total if wall_total > 0 else 0.0,
                total_tags_written / wall_total if wall_total > 0 else 0.0,
                completed_batches,
            )
            if performance_tuning and perf_tuning_series:
                tr = final_payload.get("tuning_report") or {}
                agg = tr.get("aggregate") or {}
                log.info(
                    "tagging_ws tuning_report batches_recorded=%s sum_wall_s=%s files_per_wall_s=%s",
                    tr.get("batches_recorded"),
                    agg.get("sum_wall_s"),
                    agg.get("files_per_wall_s"),
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
        err_payload: dict = {
            "type": "error",
            "message": str(e),
            "partial_results": all_results,
            "pending_hydrus_files": pending_err,
        }
        if performance_tuning:
            if session_auto_tune and autotune is not None:
                eff_e, dlp_e = autotune.best_pair
            else:
                eff_e = next_bs
                dlp_e = (
                    download_parallel
                    if download_parallel is not None
                    else config.hydrus_download_parallel
                )
            at_err = None
            if autotune is not None:
                at_err = dict(autotune.summary_for_report())
                at_err["ort_session_reloads"] = ort_reload_count
            err_payload["tuning_report"] = build_tuning_report(
                perf_tuning_series,
                stopped=True,
                batches_completed=completed_batches,
                total_processed=len(all_results),
                effective_batch=int(eff_e),
                download_parallel=int(dlp_e),
                model_name=resolved_model,
                history_window=pt_history_window,
                session_auto_tune=session_auto_tune,
                tuning_control_mode=tuning_control_mode if session_auto_tune else None,
                supervised_gates_passed=supervised_gates_passed,
                autotune_summary=at_err,
            )
        await ws_send(err_payload)
    finally:
        unregister_shutdown_notifier(notify_server_shutdown)
        unregister_tagging_session(session_handle)
        listener.cancel()
        try:
            await listener
        except asyncio.CancelledError:
            pass

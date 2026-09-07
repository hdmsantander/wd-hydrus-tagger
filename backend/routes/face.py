"""Face detection and recognition API routes."""

from __future__ import annotations

import asyncio
import json
import logging
import time

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

import backend.config as config_module
from backend.config import AppConfig
from backend.dependencies import get_hydrus_client
from backend.face.models import download_face_pack, inspect_face_pack, verify_face_pack
from backend.face.load_control import cancel_pending_face_loads
from backend.face.service import FaceTaggingService, resolved_face_models_root
from backend.hydrus.client import HydrusClient
from backend.services.face_session_registry import (
    FaceSessionHandle,
    active_face_sessions_count,
    get_face_session_status,
    register_face_session,
    register_face_shutdown_notifier,
    unregister_face_session,
    unregister_face_shutdown_notifier,
)
from backend.tagger.ort_providers import available_ort_providers, resolve_ort_providers

log = logging.getLogger(__name__)

router = APIRouter()

_face_detect_lock = asyncio.Lock()


class FaceDetectRequest(BaseModel):
    file_ids: list[int] = Field(min_length=1)
    service_key: str = ""
    det_threshold: float | None = None
    replace_existing: bool = False


class FaceRecognizeRequest(BaseModel):
    service_key: str = ""
    max_distance: float | None = None
    min_faces: int | None = None
    allow_new: bool = True
    staged: bool = True


async def _resolve_service_key(client: HydrusClient, body_key: str, config: AppConfig) -> str:
    if body_key.strip():
        return body_key.strip()
    name = (config.face_target_tag_service or config.target_tag_service).strip()
    services = await client.get_services()
    for bucket in ("local_tags", "all_known_tags"):
        for svc in services.get(bucket, []):
            if svc.get("name") == name:
                return svc["service_key"]
    raise ValueError(f"Tag service {name!r} not found")


@router.get("/providers")
async def list_ort_providers():
    return {"success": True, "providers": available_ort_providers()}


@router.get("/status")
async def face_status():
    svc = FaceTaggingService.get_instance(config_module.get_config())
    st = await svc.status()
    return {"success": True, "status": st}


@router.get("/models")
async def list_face_models():
    config = config_module.get_config()
    svc = FaceTaggingService.get_instance(config)
    row = inspect_face_pack(resolved_face_models_root(config.face_models_dir), config.face_model_pack)
    row["loaded_in_memory"] = svc.engine.loaded
    row["active_provider"] = svc.engine.active_provider if svc.engine.loaded else None
    return {
        "success": True,
        "models": [row],
        "loaded_model": config.face_model_pack if svc.engine.loaded else None,
        "default_model": config.face_model_pack,
        "models_dir": str(resolved_face_models_root(config.face_models_dir)),
    }


@router.post("/models/verify")
async def verify_face_models():
    config = config_module.get_config()
    row = verify_face_pack(resolved_face_models_root(config.face_models_dir), config.face_model_pack)
    return {
        "success": True,
        "models_dir": str(resolved_face_models_root(config.face_models_dir)),
        "results": [row],
    }


@router.post("/models/download")
async def download_face_model():
    config = config_module.get_config()
    root = resolved_face_models_root(config.face_models_dir)
    try:
        dest = await asyncio.to_thread(download_face_pack, root, config.face_model_pack)
    except Exception as exc:
        log.exception("face model download failed")
        return {"success": False, "error": str(exc)}
    return {
        "success": True,
        "message": f"Face pack {config.face_model_pack} downloaded",
        "path": str(dest),
        **inspect_face_pack(root, config.face_model_pack),
    }


@router.post("/models/load")
async def load_face_model():
    svc = FaceTaggingService.get_instance(config_module.get_config())
    try:
        await svc.ensure_model_loaded()
    except Exception as exc:
        log.exception("face model load failed")
        return {"success": False, "error": str(exc)}
    return {
        "success": True,
        "model_loaded": svc.engine.loaded,
        "model_pack": svc.config.face_model_pack,
        "providers": svc.engine.providers,
        "active_provider": svc.engine.active_provider,
    }


@router.post("/models/unload")
async def unload_face_model():
    svc = FaceTaggingService.get_instance(config_module.get_config())
    svc.engine.unload()
    return {"success": True, "model_loaded": False}


@router.post("/detect")
async def detect_faces_http(
    body: FaceDetectRequest,
    client: HydrusClient = Depends(get_hydrus_client),
):
    config = config_module.get_config()
    svc = FaceTaggingService.get_instance(config)
    try:
        service_key = await _resolve_service_key(client, body.service_key, config)
    except ValueError as e:
        return {"success": False, "error": str(e)}

    results = await svc.detect_batch(
        client,
        file_ids=body.file_ids,
        service_key=service_key,
        det_threshold=body.det_threshold,
        replace_existing=body.replace_existing,
    )
    faces_found = sum(r.get("face_count", 0) for r in results)
    return {
        "success": True,
        "results": results,
        "files_processed": len(results),
        "faces_found": faces_found,
    }


@router.post("/recognize")
async def recognize_faces_http(
    body: FaceRecognizeRequest,
    client: HydrusClient = Depends(get_hydrus_client),
):
    config = config_module.get_config()
    svc = FaceTaggingService.get_instance(config)
    try:
        service_key = await _resolve_service_key(client, body.service_key, config)
    except ValueError as e:
        return {"success": False, "error": str(e)}

    summary = await svc.recognize(
        client,
        service_key=service_key,
        max_distance=body.max_distance,
        min_faces=body.min_faces,
        allow_new=body.allow_new,
        staged=body.staged,
    )
    return {"success": True, **summary}


@router.post("/reset")
async def reset_face_assignments():
    svc = FaceTaggingService.get_instance(config_module.get_config())
    await svc.reset_assignments()
    return {"success": True}


@router.post("/clean")
async def clean_face_db(
    client: HydrusClient = Depends(get_hydrus_client),
):
    svc = FaceTaggingService.get_instance(config_module.get_config())
    summary = await svc.clean_orphans(client)
    return {"success": True, **summary}


@router.get("/session/status")
async def face_session_status():
    """Whether a face-detection WebSocket run is active."""
    return {"success": True, **get_face_session_status()}


@router.websocket("/ws/progress")
async def face_detect_ws(websocket: WebSocket):
    await websocket.accept()
    config = config_module.get_config()
    svc = FaceTaggingService.get_instance(config)

    try:
        raw = await websocket.receive_text()
        payload = json.loads(raw)
    except (WebSocketDisconnect, json.JSONDecodeError):
        await websocket.close()
        return

    if payload.get("action") != "run":
        await websocket.send_json({"type": "error", "error": "First message must be action=run"})
        await websocket.close()
        return

    file_ids = [int(x) for x in payload.get("file_ids") or []]
    if not file_ids:
        await websocket.send_json({"type": "error", "error": "file_ids required"})
        await websocket.close()
        return

    async with _face_detect_lock:
        if active_face_sessions_count() > 0:
            await websocket.send_json({"type": "error", "error": "Face detection already running"})
            await websocket.close()
            return

    cancel_event = asyncio.Event()
    session_handle = FaceSessionHandle(cancel_event=cancel_event)
    register_face_session(session_handle)

    async def ws_send(msg: dict) -> None:
        await websocket.send_json(msg)

    async def notify_server_shutdown():
        await ws_send(
            {
                "type": "server_shutting_down",
                "message": "Server stop requested. Face detection will cancel at the next file boundary.",
            }
        )

    register_face_shutdown_notifier(notify_server_shutdown)

    async def control_listener():
        try:
            while True:
                msg = await websocket.receive_text()
                data = json.loads(msg)
                if data.get("action") == "cancel":
                    cancel_event.set()
                    await ws_send({"type": "stopping"})
                    break
        except (WebSocketDisconnect, json.JSONDecodeError):
            cancel_event.set()

    listener = asyncio.create_task(control_listener())

    try:
        client = HydrusClient(config.hydrus_api_url, config.hydrus_api_key)
        try:
            service_key = await _resolve_service_key(client, str(payload.get("service_key") or ""), config)
        except ValueError as e:
            await ws_send({"type": "error", "error": str(e)})
            return

        await ws_send({"type": "started", "total": len(file_ids)})

        async def progress_cb(msg: dict):
            await ws_send(msg)

        t0 = time.monotonic()
        planned = resolve_ort_providers(use_gpu=config.use_gpu, gpu_backend=config.gpu_backend)
        log.info(
            "face detect ws start files=%s use_gpu=%s gpu_backend=%s installed_ort=%s planned_providers=%s",
            len(file_ids),
            config.use_gpu,
            config.gpu_backend,
            available_ort_providers(),
            planned,
        )
        try:
            results = await svc.detect_batch(
                client,
                file_ids=file_ids,
                service_key=service_key,
                det_threshold=payload.get("det_threshold"),
                replace_existing=bool(payload.get("replace_existing")),
                progress_cb=progress_cb,
                cancel_event=cancel_event,
            )
        except asyncio.CancelledError:
            log.info("face detect ws cancelled")
            await ws_send(
                {
                    "type": "stopped",
                    "results": [],
                    "files_processed": 0,
                    "faces_found": 0,
                }
            )
            return
        except Exception as exc:
            log.exception("face detect ws failed")
            cancel_pending_face_loads(reason="face_ws_error")
            await ws_send({"type": "error", "error": str(exc)})
            return

        terminal = "stopped" if cancel_event.is_set() else "complete"
        faces_found = sum(r.get("face_count", 0) for r in results)
        skipped = sum(1 for r in results if r.get("skipped"))
        videos = sum(1 for r in results if r.get("skip_reason") == "video_excluded")
        marker_skip = sum(1 for r in results if r.get("skip_reason") == "marker_present")
        errors = sum(1 for r in results if r.get("error"))
        log.info(
            "face detect ws %s files=%s faces=%s skipped=%s videos=%s marker_skip=%s errors=%s elapsed_s=%.1f active_provider=%s",
            terminal,
            len(results),
            faces_found,
            skipped,
            videos,
            marker_skip,
            errors,
            time.monotonic() - t0,
            svc.engine.active_provider if svc.engine.loaded else None,
        )
        await ws_send(
            {
                "type": terminal,
                "results": results,
                "files_processed": len(results),
                "faces_found": faces_found,
                "skipped": skipped,
                "videos_skipped": videos,
                "marker_skipped": marker_skip,
                "errors": errors,
            }
        )
    finally:
        if cancel_event.is_set():
            cancel_pending_face_loads(reason="face_ws_cancelled")
        unregister_face_shutdown_notifier(notify_server_shutdown)
        unregister_face_session(session_handle)
        listener.cancel()
        try:
            await listener
        except asyncio.CancelledError:
            pass

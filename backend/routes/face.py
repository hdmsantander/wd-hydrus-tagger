"""Face detection and recognition API routes."""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

import backend.config as config_module
from backend.config import AppConfig as load_app_config
from backend.dependencies import get_hydrus_client
from backend.face.service import FaceTaggingService
from backend.hydrus.client import HydrusClient
from backend.tagger.ort_providers import available_ort_providers

log = logging.getLogger(__name__)

router = APIRouter()

_face_detect_lock = asyncio.Lock()
_face_detect_active = False


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


@router.post("/models/load")
async def load_face_model():
    svc = FaceTaggingService.get_instance(config_module.get_config())
    await svc.ensure_model_loaded()
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


@router.websocket("/ws/progress")
async def face_detect_ws(websocket: WebSocket):
    global _face_detect_active
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
        if _face_detect_active:
            await websocket.send_json({"type": "error", "error": "Face detection already running"})
            await websocket.close()
            return
        _face_detect_active = True

    cancel_event = asyncio.Event()

    async def control_listener():
        try:
            while True:
                msg = await websocket.receive_text()
                data = json.loads(msg)
                if data.get("action") == "cancel":
                    cancel_event.set()
                    await websocket.send_json({"type": "stopping"})
                    break
        except (WebSocketDisconnect, json.JSONDecodeError):
            cancel_event.set()

    listener = asyncio.create_task(control_listener())

    try:
        client = get_hydrus_client()
        try:
            service_key = await _resolve_service_key(client, str(payload.get("service_key") or ""), config)
        except ValueError as e:
            await websocket.send_json({"type": "error", "error": str(e)})
            return

        await websocket.send_json({"type": "started", "total": len(file_ids)})

        async def progress_cb(msg: dict):
            await websocket.send_json(msg)

        results = await svc.detect_batch(
            client,
            file_ids=file_ids,
            service_key=service_key,
            det_threshold=payload.get("det_threshold"),
            replace_existing=bool(payload.get("replace_existing")),
            progress_cb=progress_cb,
            cancel_event=cancel_event,
        )

        terminal = "stopped" if cancel_event.is_set() else "complete"
        faces_found = sum(r.get("face_count", 0) for r in results)
        await websocket.send_json(
            {
                "type": terminal,
                "results": results,
                "files_processed": len(results),
                "faces_found": faces_found,
            }
        )
    finally:
        _face_detect_active = False
        listener.cancel()
        try:
            await listener
        except asyncio.CancelledError:
            pass

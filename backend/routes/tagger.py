"""Tagger endpoints."""

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.config import get_config
from backend.hydrus.client import HydrusClient
from backend.services.model_manager import ModelManager
from backend.services.tagging_service import TaggingService

router = APIRouter()


def _get_model_manager() -> ModelManager:
    config = get_config()
    return ModelManager(config.models_dir)


@router.get("/models")
async def list_models():
    """List available models and their download status."""
    manager = _get_model_manager()
    models = manager.list_models()
    return {"success": True, "models": models}


@router.post("/models/{name}/download")
async def download_model(name: str):
    """Download a model from HuggingFace."""
    manager = _get_model_manager()
    try:
        await asyncio.to_thread(manager.download_model, name)
        return {"success": True, "message": f"Model {name} downloaded"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/models/{name}/load")
async def load_model(name: str):
    """Load a model into memory."""
    config = get_config()
    service = TaggingService.get_instance(config)
    try:
        await asyncio.to_thread(service.load_model, name)
        return {"success": True, "message": f"Model {name} loaded"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/predict")
async def predict(body: dict):
    """Run WD14 tagging on files."""
    config = get_config()
    file_ids = body.get("file_ids", [])
    general_threshold = body.get("general_threshold", config.general_threshold)
    character_threshold = body.get("character_threshold", config.character_threshold)

    service = TaggingService.get_instance(config)
    client = HydrusClient(config.hydrus_api_url, config.hydrus_api_key)

    try:
        results = await service.tag_files(
            client=client,
            file_ids=file_ids,
            general_threshold=general_threshold,
            character_threshold=character_threshold,
        )
        return {"success": True, "results": results}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/apply")
async def apply_tags(body: dict):
    """Apply tags to files in Hydrus."""
    config = get_config()
    results = body.get("results", [])
    service_key = body.get("service_key", "")

    client = HydrusClient(config.hydrus_api_url, config.hydrus_api_key)

    try:
        for item in results:
            file_hash = item["hash"]
            tags = item["tags"]
            await client.add_tags(
                hash_=file_hash,
                service_key=service_key,
                tags=tags,
            )
        return {"success": True, "applied": len(results)}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.websocket("/ws/progress")
async def progress_ws(websocket: WebSocket):
    """WebSocket for batch processing progress."""
    await websocket.accept()
    config = get_config()
    try:
        while True:
            data = await websocket.receive_text()
            request = json.loads(data)

            file_ids = request.get("file_ids", [])
            general_threshold = request.get("general_threshold", config.general_threshold)
            character_threshold = request.get("character_threshold", config.character_threshold)

            service = TaggingService.get_instance(config)
            client = HydrusClient(config.hydrus_api_url, config.hydrus_api_key)

            total = len(file_ids)
            all_results = []

            for i, file_id in enumerate(file_ids):
                try:
                    results = await service.tag_files(
                        client=client,
                        file_ids=[file_id],
                        general_threshold=general_threshold,
                        character_threshold=character_threshold,
                    )
                    if results:
                        all_results.append(results[0])
                    await websocket.send_json({
                        "type": "progress",
                        "current": i + 1,
                        "total": total,
                        "file_id": file_id,
                        "result": results[0] if results else None,
                    })
                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "file_id": file_id,
                        "message": str(e),
                    })

            await websocket.send_json({
                "type": "complete",
                "total_processed": len(all_results),
                "results": all_results,
            })
    except WebSocketDisconnect:
        pass

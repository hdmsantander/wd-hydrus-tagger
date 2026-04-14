"""HTTP routes for tagger: models, predict, apply, session snapshot."""

import asyncio
import logging
import time

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict

from backend.config import get_config
from backend.hydrus.client import HydrusClient
from backend.perf_metrics import log_apply_tags_http, log_predict_wall
from backend.routes.tagger_apply import _apply_results_chunk
from backend.services.model_manager import ModelManager, ModelVerifyResult, SUPPORTED_MODELS
from backend.services.tagging_service import TaggingService
from backend.services.tagging_shared import clamp_inference_batch
from backend.services.tagging_session_registry import get_public_session_status

router = APIRouter()
log = logging.getLogger(__name__)


class VerifyModelsRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    check_remote: bool = False
    model_name: str | None = None


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    file_ids: list[int]
    general_threshold: float | None = None
    character_threshold: float | None = None
    batch_size: object | None = None
    model_name: str | None = None
    service_key: str | None = None


class ApplyTagsRow(BaseModel):
    model_config = ConfigDict(extra="ignore")
    file_id: int
    hash: str
    tags: list[str]


class ApplyTagsRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    results: list[ApplyTagsRow] = []
    service_key: str = ""


def _get_model_manager() -> ModelManager:
    config = get_config()
    return ModelManager(config.models_dir)


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
async def verify_models(body: VerifyModelsRequest | None = None):
    """Check cached ONNX+CSV (structure, sizes); optional Hub ``main`` revision comparison (network)."""
    payload = body or VerifyModelsRequest()
    check_remote = bool(payload.check_remote)
    raw_name = payload.model_name
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
        log.exception("download_model endpoint failed name=%s", name)
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
async def predict(body: PredictRequest):
    """Run WD14 tagging on files."""
    config = get_config()
    file_ids = body.file_ids
    general_threshold = (
        config.general_threshold if body.general_threshold is None else body.general_threshold
    )
    character_threshold = (
        config.character_threshold if body.character_threshold is None else body.character_threshold
    )
    raw_bs = body.batch_size
    batch_size = None
    if raw_bs is not None:
        try:
            batch_size = clamp_inference_batch(int(raw_bs), config.batch_size)
        except (TypeError, ValueError):
            batch_size = None

    eff = clamp_inference_batch(batch_size, config.batch_size)
    log.info(
        "predict file_count=%s inference_batch=%s (override=%s)",
        len(file_ids),
        eff,
        batch_size,
    )

    service = TaggingService.get_instance(config)
    client = HydrusClient(config.hydrus_api_url, config.hydrus_api_key)
    model_name = body.model_name or None
    service_key = body.service_key or None

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
async def apply_tags(body: ApplyTagsRequest):
    """Apply tags to files in Hydrus."""
    config = get_config()
    results = [row.model_dump() for row in body.results]
    service_key = body.service_key

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
        n_chunks = (n + bs - 1) // bs if bs else 0
        log.debug(
            "apply_tags http_route rows=%s chunk_size=%s num_chunks=%s "
            "(apply_tags_every_n=%s applies_to_websocket_tagging_only)",
            n,
            bs,
            n_chunks,
            config.apply_tags_every_n,
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

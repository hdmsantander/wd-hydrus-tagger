"""Configuration endpoints."""

from fastapi import APIRouter
from pydantic import ValidationError

from backend.config import AppConfig, get_config, save_config
from backend.hydrus.client import invalidate_hydrus_client_pool

router = APIRouter()


@router.get("")
async def get_configuration():
    """Get current configuration."""
    config = get_config()
    data = config.model_dump()
    # Don't expose full API key
    if data.get("hydrus_api_key"):
        key = data["hydrus_api_key"]
        data["hydrus_api_key_masked"] = key[:8] + "..." + key[-4:] if len(key) > 12 else "***"
    data.pop("hydrus_api_key", None)
    return {"success": True, "config": data}


@router.patch("")
async def update_configuration(body: dict):
    """Update configuration fields with Pydantic validation."""
    config = get_config()
    hydrus_sig_before = (config.hydrus_api_url, config.hydrus_api_key)
    updatable_fields = {
        "general_threshold", "character_threshold", "target_tag_service",
        "general_tag_prefix", "character_tag_prefix", "rating_tag_prefix",
        "batch_size", "default_model", "use_gpu",
        "cpu_intra_op_threads", "cpu_inter_op_threads",
        "hydrus_download_parallel", "hydrus_metadata_chunk_size", "apply_tags_every_n",
        "wd_skip_inference_if_marker_present", "wd_skip_if_higher_tier_model_present",
        "wd_append_model_marker_tag",
        "wd_model_marker_template", "wd_model_marker_prefix",
        "apply_tags_http_batch_size",
        "allow_ui_shutdown", "shutdown_tagging_grace_seconds",
    }

    merged = config.model_dump()
    updated = []
    for key, value in body.items():
        if key in updatable_fields:
            merged[key] = value
            updated.append(key)

    if not updated:
        return {"success": True, "updated": []}

    try:
        new_config = AppConfig.model_validate(merged)
    except ValidationError as e:
        return {"success": False, "error": e.errors(), "updated": []}

    save_config(new_config)

    hydrus_sig_after = (new_config.hydrus_api_url, new_config.hydrus_api_key)
    if hydrus_sig_after != hydrus_sig_before:
        await invalidate_hydrus_client_pool()

    return {"success": True, "updated": updated}

"""Configuration endpoints."""

from fastapi import APIRouter

from backend.config import AppConfig, get_config, save_config

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
    """Update configuration fields."""
    config = get_config()
    updatable_fields = {
        "general_threshold", "character_threshold", "target_tag_service",
        "general_tag_prefix", "character_tag_prefix", "rating_tag_prefix",
        "batch_size", "default_model", "use_gpu",
    }

    updated = []
    for key, value in body.items():
        if key in updatable_fields:
            setattr(config, key, value)
            updated.append(key)

    if updated:
        save_config(config)

    return {"success": True, "updated": updated}

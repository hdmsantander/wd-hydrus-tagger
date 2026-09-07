"""Configuration endpoints."""

from fastapi import APIRouter
from pydantic import ValidationError
from pydantic import BaseModel, ConfigDict

from backend.config import AppConfig, get_config, save_config
from backend.hydrus.client import invalidate_hydrus_client_pool

router = APIRouter()


class ConfigPatchRequest(BaseModel):
    model_config = ConfigDict(extra="allow")


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
async def update_configuration(body: ConfigPatchRequest):
    """Update configuration fields with Pydantic validation."""
    config = get_config()
    hydrus_sig_before = (config.hydrus_api_url, config.hydrus_api_key)
    updatable_fields = {
        "general_threshold", "character_threshold", "target_tag_service",
        "hydrus_web_url",
        "general_tag_prefix", "character_tag_prefix", "rating_tag_prefix",
        "batch_size", "default_model", "use_gpu", "gpu_backend",
        "face_det_threshold", "face_target_tag_service", "face_skip_if_detected",
        "face_marker_detected", "face_marker_not_visible", "face_marker_recognized",
        "face_person_tag_prefix",
        "face_recognition_max_distance", "face_recognition_min_faces",
        "face_recognition_stages", "face_recognition_distance_method",
        "face_video_frame_count", "face_model_load_timeout_seconds",
        "face_inference_timeout_seconds",
        "cpu_intra_op_threads", "cpu_inter_op_threads",
        "hydrus_download_parallel", "hydrus_metadata_chunk_size",
        "tagging_skip_tail_batch_size", "apply_tags_every_n",
        "wd_skip_inference_if_marker_present", "wd_skip_if_higher_tier_model_present",
        "wd_append_model_marker_tag",
        "wd_model_marker_template", "wd_model_marker_prefix",
        "apply_tags_http_batch_size",
        "allow_ui_shutdown", "shutdown_tagging_grace_seconds",
        "max_learning_cached_files",
        "ort_enable_profiling", "ort_profile_dir",
    }

    merged = config.model_dump()
    updated = []
    for key, value in body.model_dump(exclude_unset=True).items():
        if key in updatable_fields:
            merged[key] = value
            updated.append(key)

    if not updated:
        return {"success": True, "updated": []}

    try:
        new_config = AppConfig.model_validate(merged)
    except ValidationError as e:
        return {"success": False, "error": e.errors(), "updated": []}

    persisted = save_config(new_config)

    hydrus_sig_after = (new_config.hydrus_api_url, new_config.hydrus_api_key)
    if hydrus_sig_after != hydrus_sig_before:
        await invalidate_hydrus_client_pool()

    result: dict = {"success": True, "updated": updated}
    if not persisted:
        result["persisted"] = False
        result["warning"] = (
            "Settings applied for this session only; config file is read-only "
            "(common in Docker). Edit config.yaml on the host to persist."
        )
    return result

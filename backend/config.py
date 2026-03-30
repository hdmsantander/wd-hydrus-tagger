"""Application configuration loaded from YAML."""

import logging
import os
import tempfile
import warnings
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator

# Repository root (parent of ``backend/``). Used to resolve relative ``models_dir``.
_REPO_ROOT = Path(__file__).resolve().parent.parent

_log = logging.getLogger(__name__)


def resolved_models_dir(path: str | Path) -> str:
    """Resolve ``./models``-style paths against the repo root so caches survive CWD changes."""
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (_REPO_ROOT / p).resolve()
    else:
        p = p.resolve()
    return str(p)


def _temp_anchor_paths() -> list[Path]:
    """Roots used to detect ONNX cache paths that would be wiped or unmanaged (tmp, pytest)."""
    roots: list[Path] = []
    for key in ("TMPDIR", "TEMP", "TMP"):
        v = os.environ.get(key)
        if v:
            try:
                roots.append(Path(v).expanduser().resolve())
            except OSError:
                pass
    try:
        roots.append(Path(tempfile.gettempdir()).expanduser().resolve())
    except OSError:
        pass
    out: list[Path] = []
    seen: set[str] = set()
    for p in roots:
        try:
            r = p.resolve()
        except OSError:
            continue
        k = str(r)
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def path_is_ephemeral_models_location(resolved: Path) -> bool:
    """True if this directory should not host a long-lived ONNX disk cache."""
    s = str(resolved).replace("\\", "/").lower()
    if "pytest" in s:
        return True
    try:
        r = resolved.resolve()
    except OSError:
        return False
    for root in _temp_anchor_paths():
        try:
            if r == root:
                return True
            if hasattr(r, "is_relative_to") and r.is_relative_to(root):
                return True
        except (OSError, ValueError, TypeError):
            continue
    return False


def _allow_tmp_models_dir_env() -> bool:
    return os.environ.get("WD_TAGGER_ALLOW_TMP_MODELS_DIR", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def stable_models_dir_for_config(raw_models_dir: str) -> str:
    """Resolve ``models_dir`` and redirect temp/pytest trees to ``<repo>/models`` unless tests opt out.

    Ephemeral locations cause cache misses and uncontrolled eviction; production should use ``./models``
    or another persistent path in ``config.yaml``.
    """
    resolved_str = resolved_models_dir(raw_models_dir)
    try:
        resolved = Path(resolved_str).resolve()
    except OSError:
        return resolved_str
    if _allow_tmp_models_dir_env() or not path_is_ephemeral_models_location(resolved):
        return str(resolved)
    fallback = (_REPO_ROOT / "models").resolve()
    _log.warning(
        "models_dir pointed to a temporary/ephemeral location (%s); using %s so ONNX and "
        "``.wd_model_cache.json`` survive reboots. Set ``models_dir`` in config.yaml to a stable path "
        "(recommended: ./models). For pytest only, set WD_TAGGER_ALLOW_TMP_MODELS_DIR=1.",
        resolved,
        fallback,
    )
    return str(fallback)


class AppConfig(BaseModel):
    hydrus_api_url: str = "http://localhost:45869"
    hydrus_api_key: str = ""

    default_model: str = "wd-vit-tagger-v3"
    models_dir: str = "./models"
    use_gpu: bool = False

    general_threshold: float = 0.35
    character_threshold: float = 0.85

    target_tag_service: str = "my tags"

    general_tag_prefix: str = ""
    character_tag_prefix: str = "character:"
    rating_tag_prefix: str = "rating:"

    @field_validator(
        "target_tag_service", "general_tag_prefix",
        "character_tag_prefix", "rating_tag_prefix",
        mode="before",
    )
    @classmethod
    def coerce_none_to_str(cls, v, info):
        """Convert YAML null values to string defaults."""
        defaults = {
            "target_tag_service": "my tags",
            "general_tag_prefix": "",
            "character_tag_prefix": "character:",
            "rating_tag_prefix": "rating:",
        }
        if v is None:
            return defaults.get(info.field_name, "")
        return v

    batch_size: int = Field(default=8, ge=1, le=256)

    # ONNX Runtime CPUExecutionProvider thread pools (ignored for pure-GPU graphs in practice).
    # Ryzen 7 5700X3D: 8 physical cores → intra_op 8, inter_op 1 for typical single-batch inference.
    cpu_intra_op_threads: int = Field(default=8, ge=1, le=64)
    cpu_inter_op_threads: int = Field(default=1, ge=1, le=16)

    # Concurrent Hydrus file downloads per inference batch (HTTP layer).
    hydrus_download_parallel: int = Field(default=8, ge=1, le=32)

    # Chunk size for get_file_metadata (tagging + gallery API). Large searches avoid one huge Hydrus call.
    hydrus_metadata_chunk_size: int = Field(default=256, ge=32, le=2048)

    # During WebSocket tagging: push tags to Hydrus every N successfully tagged files (0 = off).
    apply_tags_every_n: int = Field(default=0, ge=0, le=256)

    # Skip ONNX for files that already carry the model marker tag (see build_wd_model_marker).
    wd_skip_inference_if_marker_present: bool = True
    # Skip ONNX when storage already has a strictly higher-tier WD marker (see WD_MODEL_CAPABILITY_TIER
    # in tag_merge). Speeds up Tag all over collections already processed with a heavier model.
    wd_skip_if_higher_tier_model_present: bool = True
    # Append marker to tag lists after inference so future runs can skip (same service as other WD tags).
    wd_append_model_marker_tag: bool = True
    # Empty → default tag ``wd14:{model_name}``. May include ``{model_name}`` placeholder.
    wd_model_marker_template: str = ""
    # Normalized prefix for stripping stale model markers from proposed tags (must match built markers).
    wd_model_marker_prefix: str = "wd14:"

    # Chunk size for POST /api/tagger/apply (many files in one click).
    apply_tags_http_batch_size: int = Field(default=100, ge=1, le=512)

    # POST /api/app/shutdown from the Settings UI (disable if the app is reachable untrusted clients).
    allow_ui_shutdown: bool = True
    # After signaling flush to active tagging sessions, wait this long before cancel + process exit.
    shutdown_tagging_grace_seconds: float = Field(default=1.5, ge=0.0, le=30.0)

    # 0.0.0.0 = all IPv4 interfaces (LAN + localhost). Use 127.0.0.1 to block remote browsers.
    host: str = "0.0.0.0"
    port: int = 8199


CONFIG_PATH = Path("config.yaml")
EXAMPLE_CONFIG_PATH = Path("config.example.yaml")

_config: Optional[AppConfig] = None


def load_config() -> AppConfig:
    global _config
    if _config is not None:
        return _config

    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    elif EXAMPLE_CONFIG_PATH.exists():
        with open(EXAMPLE_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    _config = AppConfig(**data)
    _config = _config.model_copy(update={"models_dir": stable_models_dir_for_config(_config.models_dir)})
    md = str(_config.models_dir).replace("\\", "/").lower()
    if _allow_tmp_models_dir_env() and ("pytest" in md or "/tmp/pytest" in md):
        warnings.warn(
            f"models_dir is under pytest/tmp ({_config.models_dir!r}) with WD_TAGGER_ALLOW_TMP_MODELS_DIR; "
            "ONNX cache may not persist across runs.",
            UserWarning,
            stacklevel=2,
        )
    return _config


def save_config(config: AppConfig) -> None:
    global _config
    _config = config
    data = config.model_dump()
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def get_config() -> AppConfig:
    global _config
    if _config is None:
        return load_config()
    return _config

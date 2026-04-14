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


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _allow_tmp_models_dir_env() -> bool:
    return _env_truthy("WD_TAGGER_ALLOW_TMP_MODELS_DIR")


def apply_runtime_config_overrides(config: AppConfig) -> AppConfig:
    """Env wins for diagnostic flags (Tier D); keep merge logic testable without ``load_config`` cache."""
    if _env_truthy("WD_TAGGER_ORT_PROFILING"):
        return config.model_copy(update={"ort_enable_profiling": True})
    return config


def resolved_ort_profile_dir(path: str) -> Path:
    """Resolve ``ort_profile_dir`` relative to the repo root (same convention as ``./models``)."""
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (_REPO_ROOT / p).resolve()
    else:
        p = p.resolve()
    return p


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

    target_tag_service: str = "local tags"

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
            "target_tag_service": "local tags",
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

    # Tier D (§8): ONNX Runtime session profiling — off by default; large trace files; throughput hit.
    ort_enable_profiling: bool = False
    ort_profile_dir: str = "./ort_traces"

    # T-Learn (§14.2): cap in-memory learning prefix rows (file count) before Phase C.
    max_learning_cached_files: int = Field(default=400_000, ge=32, le=2_000_000)

    # Concurrent Hydrus file downloads per inference batch (HTTP layer).
    hydrus_download_parallel: int = Field(default=8, ge=1, le=32)

    # Chunk size for get_file_metadata (tagging + gallery API). Large searches avoid one huge Hydrus call.
    hydrus_metadata_chunk_size: int = Field(default=512, ge=32, le=2048)

    # WebSocket Tag all: after metadata prefetch, marker-skip files are batched this large (no ONNX) so
    # the tail clears quickly. Independent of inference batch_size; typically >= hydrus_metadata_chunk_size.
    tagging_skip_tail_batch_size: int = Field(default=512, ge=32, le=2048)

    # WebSocket tagging (Tag selected / Tag all): when incremental Hydrus writes are on, push every N
    # processed files. 0 = off. Unrelated to ``apply_tags_http_batch_size`` (HTTP apply route only).
    apply_tags_every_n: int = Field(
        default=8,
        ge=0,
        le=256,
        description="WebSocket tagging: Hydrus write stride when incremental apply is enabled; 0 disables.",
    )

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

    # Chunk size for POST /api/tagger/apply only (results screen “Apply all tags to Hydrus”).
    # Does not control WebSocket tagging; see ``apply_tags_every_n``.
    apply_tags_http_batch_size: int = Field(
        default=100,
        ge=1,
        le=512,
        description="HTTP POST /api/tagger/apply: rows per request chunk from the results list.",
    )

    # POST /api/app/shutdown from the Settings UI (disable if the app is reachable untrusted clients).
    allow_ui_shutdown: bool = True
    # After signaling flush to active tagging sessions, wait this long before cancel + process exit.
    shutdown_tagging_grace_seconds: float = Field(default=0.0, ge=0.0, le=30.0)

    # 0.0.0.0 = all IPv4 interfaces (LAN + localhost). Use 127.0.0.1 to block remote browsers.
    host: str = "0.0.0.0"
    port: int = 8199


# Bounds for ``hydrus_metadata_chunk_size`` (must match ``Field(ge=, le=)`` on ``AppConfig``).
HYDRUS_METADATA_CHUNK_MIN = 32
HYDRUS_METADATA_CHUNK_MAX = 2048


def clamp_hydrus_metadata_chunk_size(value: object) -> int:
    """Clamp Hydrus ``get_file_metadata`` chunk size (defensive for config and call sites)."""
    try:
        n = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 512
    return max(HYDRUS_METADATA_CHUNK_MIN, min(HYDRUS_METADATA_CHUNK_MAX, n))


_config: Optional[AppConfig] = None


def config_yaml_path() -> Path:
    """Resolved path to the main YAML config file.

    ``WD_TAGGER_CONFIG_PATH`` (absolute or relative to CWD) overrides the default
    ``<repo>/config.yaml``. Tests set this so accidental ``save_config`` calls do not
    overwrite the developer's real config. Normal runs resolve against the repository
    root, not the process working directory.
    """
    override = (os.environ.get("WD_TAGGER_CONFIG_PATH") or "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (_REPO_ROOT / "config.yaml").resolve()


def config_example_yaml_path() -> Path:
    return (_REPO_ROOT / "config.example.yaml").resolve()


def load_config() -> AppConfig:
    global _config
    if _config is not None:
        return _config

    cfg_path = config_yaml_path()
    ex_path = config_example_yaml_path()
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    elif ex_path.exists():
        with open(ex_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    _config = AppConfig(**data)
    _config = _config.model_copy(update={"models_dir": stable_models_dir_for_config(_config.models_dir)})
    _config = apply_runtime_config_overrides(_config)
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
    path = config_yaml_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def get_config() -> AppConfig:
    global _config
    if _config is None:
        return load_config()
    return _config

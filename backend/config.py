"""Application configuration loaded from YAML."""

from __future__ import annotations

import errno
import logging
import os
import tempfile
import warnings
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, urlunparse

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


def _env_optional_bool(name: str) -> bool | None:
    """Parse a tri-state env flag: True / False / unset (None)."""
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return None
    v = str(raw).strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return None


def _allow_tmp_models_dir_env() -> bool:
    return _env_truthy("WD_TAGGER_ALLOW_TMP_MODELS_DIR")


def _running_in_docker() -> bool:
    return Path("/.dockerenv").exists()


def _docker_host_gateway_hydrus_url(url: str) -> str:
    """Map loopback Hydrus URLs to the host gateway when the tagger runs in a container."""
    p = urlparse(url)
    host = (p.hostname or "").lower()
    if host not in ("localhost", "127.0.0.1", "::1"):
        return url
    port = p.port or 45869
    gateway = os.environ.get("WD_TAGGER_DOCKER_HOST", "host.docker.internal").strip() or "host.docker.internal"
    netloc = f"{gateway}:{port}"
    return urlunparse((p.scheme or "http", netloc, p.path or "", p.params, p.query, p.fragment))


def apply_runtime_config_overrides(config: AppConfig) -> AppConfig:
    """Env wins for diagnostic flags (Tier D); keep merge logic testable without ``load_config`` cache."""
    updates: dict[str, object] = {}
    if _env_truthy("WD_TAGGER_ORT_PROFILING"):
        updates["ort_enable_profiling"] = True
    web = os.environ.get("HYDRUS_WEB_URL", "").strip()
    if web:
        validated = AppConfig.model_validate({**config.model_dump(), "hydrus_web_url": web})
        updates["hydrus_web_url"] = validated.hydrus_web_url
    api_env = os.environ.get("HYDRUS_API_URL", "").strip()
    if api_env:
        validated = AppConfig.model_validate({**config.model_dump(), "hydrus_api_url": api_env})
        updates["hydrus_api_url"] = validated.hydrus_api_url
    gpu_env = os.environ.get("WD_TAGGER_GPU_BACKEND", "").strip()
    if gpu_env:
        validated = AppConfig.model_validate({**config.model_dump(), "gpu_backend": gpu_env})
        updates["gpu_backend"] = validated.gpu_backend
    elif _running_in_docker():
        remapped = _docker_host_gateway_hydrus_url(config.hydrus_api_url)
        if remapped != config.hydrus_api_url:
            updates["hydrus_api_url"] = remapped
    use_gpu_env = _env_optional_bool("WD_TAGGER_USE_GPU")
    if use_gpu_env is not None:
        updates["use_gpu"] = use_gpu_env
    if updates:
        return config.model_copy(update=updates)
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

    Also redirects **missing** directories that resolve **outside** the repository root (common when
    ``config.yaml`` copied from the host uses an absolute host path inside Docker).
    """
    resolved_str = resolved_models_dir(raw_models_dir)
    try:
        resolved = Path(resolved_str).resolve()
    except OSError:
        return resolved_str

    if path_is_ephemeral_models_location(resolved) and not _allow_tmp_models_dir_env():
        fallback = (_REPO_ROOT / "models").resolve()
        _log.warning(
            "models_dir pointed to a temporary/ephemeral location (%s); using %s so ONNX and "
            "``.wd_model_cache.json`` survive reboots. Set ``models_dir`` in config.yaml to a stable path "
            "(recommended: ./models). For pytest only, set WD_TAGGER_ALLOW_TMP_MODELS_DIR=1.",
            resolved,
            fallback,
        )
        return str(fallback)

    try:
        repo_root = _REPO_ROOT.resolve()
    except OSError:
        repo_root = _REPO_ROOT
    try:
        under_repo = resolved == repo_root or resolved.is_relative_to(repo_root)
    except (ValueError, TypeError):
        under_repo = False
    if not under_repo and not resolved.exists():
        fallback = (_REPO_ROOT / "models").resolve()
        _log.warning(
            "models_dir %s is outside the application root (%s) and does not exist; using %s. "
            "This usually means config.yaml lists a host-only absolute path while the app runs in Docker. "
            "Use models_dir: ./models and bind-mount your host cache to the container (e.g. ./models:/app/models).",
            resolved,
            repo_root,
            fallback,
        )
        return str(fallback)

    return str(resolved)


class AppConfig(BaseModel):
    hydrus_api_url: str = "http://localhost:45869"
    hydrus_api_key: str = ""
    # Optional floogulinc/hydrus-web base URL (e.g. http://127.0.0.1:8080) for links in the tagger UI.
    hydrus_web_url: str = Field(default="", max_length=2048)

    default_model: str = "wd-vit-tagger-v3"
    models_dir: str = "./models"
    use_gpu: bool = False
    # ONNX GPU EP selection: auto (CUDA → ROCm → DirectML), cuda, rocm, directml, cpu
    gpu_backend: str = "auto"

    # --- AI face detection & unsupervised person tagging (InsightFace buffalo_l) ---
    face_model_pack: str = "buffalo_l"
    face_models_dir: str = "./models/face"
    face_embeddings_db_path: str = "./face_embeddings.db"
    face_det_threshold: float = Field(default=0.6, ge=0.1, le=1.0)
    face_target_tag_service: str = ""
    face_marker_detected: str = "ai face detected"
    face_marker_not_visible: str = "face not visible"
    face_marker_recognized: str = "face ai generated tags"
    face_person_tag_prefix: str = "person:"
    face_skip_if_detected: bool = True
    # Skip detect inference when embeddings for this file hash already exist in the local DB.
    face_skip_if_in_db: bool = True
    face_recognition_max_distance: float = Field(default=0.5, ge=0.05, le=2.0)
    face_recognition_min_faces: int = Field(default=3, ge=1, le=100)
    face_recognition_stages: list[int] = Field(default_factory=lambda: [20, 5, 3, 1])
    face_recognition_distance_method: str = "cosine_similarity"
    face_video_frame_count: int = Field(default=30, ge=1, le=120)
    # GPU InsightFace load: abandon hung MIGraphX/CUDA compile after this many seconds and retry CPU.
    face_model_load_timeout_seconds: float = Field(default=300.0, ge=30.0, le=900.0)
    face_inference_timeout_seconds: float = Field(default=300.0, ge=30.0, le=900.0)

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

    @field_validator("gpu_backend", mode="before")
    @classmethod
    def normalize_gpu_backend(cls, v: object) -> str:
        s = str(v or "auto").strip().lower()
        allowed = {"auto", "cuda", "rocm", "directml", "cpu"}
        if s not in allowed:
            raise ValueError(f"gpu_backend must be one of {sorted(allowed)}")
        return s

    @field_validator("face_recognition_distance_method", mode="before")
    @classmethod
    def normalize_face_distance_method(cls, v: object) -> str:
        s = str(v or "cosine_similarity").strip().lower()
        if s not in ("cosine_similarity", "euclidean"):
            raise ValueError("face_recognition_distance_method must be cosine_similarity or euclidean")
        return s

    @field_validator("face_recognition_stages", mode="before")
    @classmethod
    def normalize_face_stages(cls, v: object) -> list[int]:
        if v is None:
            return [20, 5, 3, 1]
        if isinstance(v, str):
            parts = [p.strip() for p in v.split(",") if p.strip()]
            return [int(p) for p in parts]
        return [int(x) for x in v]

    @field_validator("hydrus_web_url", mode="before")
    @classmethod
    def normalize_hydrus_web_url(cls, v: object) -> str:
        s = str(v or "").strip()
        if not s:
            return ""
        if not (s.startswith("http://") or s.startswith("https://")):
            raise ValueError("hydrus_web_url must be empty or start with http:// or https://")
        return s.rstrip("/")

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


def save_config(config: AppConfig) -> bool:
    """Update in-memory config; persist to YAML when the path is writable.

    Returns True when written to disk, False when only the in-memory copy was updated
    (e.g. read-only bind mount in Docker).
    """
    global _config
    _config = config
    data = config.model_dump()
    path = config_yaml_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        return True
    except OSError as e:
        if e.errno in (errno.EROFS, errno.EACCES, errno.EPERM):
            _log.warning(
                "config not persisted to %s (%s); in-memory config updated only",
                path,
                e,
            )
            return False
        raise


def get_config() -> AppConfig:
    global _config
    if _config is None:
        load_config()
    return apply_runtime_config_overrides(_config)

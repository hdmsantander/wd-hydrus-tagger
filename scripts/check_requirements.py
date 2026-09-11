#!/usr/bin/env python3
"""Pre-flight checks: Python version, runtime imports, optional config validation, writable dirs.

Exit 0 if OK; 1 on failure.

* ``WD_TAGGER_CHECK_ROOT`` — treat this directory as repo root (tests / unusual layouts).
* ``WD_TAGGER_CONFIG_PATH`` — validate this file instead of ``<root>/config.yaml`` (tests only).

When ``use_gpu: true`` or an explicit ``gpu_backend`` (cuda/rocm/directml) is set, validates
that a matching ONNX GPU execution provider is registered.

Environment reads use ``os.environ`` (not ``sys.environ``) for compatibility across Python builds.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

MIN_PY = (3, 10)


def _root() -> Path:
    env = (os.environ.get("WD_TAGGER_CHECK_ROOT") or "").strip()
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parent.parent


def _fail(msg: str) -> None:
    print(f"check_requirements: error: {msg}", file=sys.stderr)


def _ok(msg: str) -> None:
    print(f"check_requirements: ok: {msg}", file=sys.stderr)


def _check_python_version() -> bool:
    if sys.version_info < MIN_PY:
        _fail(f"need Python {MIN_PY[0]}.{MIN_PY[1]}+, got {sys.version_info.major}.{sys.version_info.minor}")
        return False
    _ok(f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True


def _installed_ort_wheel_names() -> list[str]:
    """Return installed onnxruntime distribution names (at most one is recommended)."""
    names = (
        "onnxruntime",
        "onnxruntime-gpu",
        "onnxruntime-rocm",
        "onnxruntime-migraphx",
    )
    found: list[str] = []
    try:
        import importlib.metadata as md

        for name in names:
            try:
                md.version(name)
                found.append(name)
            except md.PackageNotFoundError:
                continue
    except Exception:
        return []
    return found


def _check_imports() -> bool:
    modules = [
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("pydantic", "pydantic"),
        ("yaml", "pyyaml"),
        ("httpx", "httpx"),
        ("PIL", "Pillow"),
        ("numpy", "numpy"),
        ("onnxruntime", "onnxruntime"),
        ("huggingface_hub", "huggingface_hub"),
        ("websockets", "websockets"),
    ]
    for mod, pip_name in modules:
        try:
            __import__(mod)
        except ImportError as e:
            _fail(f"missing import {mod} (pip package: {pip_name}): {e}")
            return False
    try:
        import onnxruntime as ort  # noqa: F401

        _ = ort.InferenceSession
    except Exception as e:
        _fail(f"onnxruntime broken: {e}")
        return False
    wheels = _installed_ort_wheel_names()
    if len(wheels) > 1:
        print(
            "check_requirements: hint: multiple ONNX Runtime wheels installed "
            f"({', '.join(wheels)}) — keep only one (CPU or GPU) to avoid EP conflicts",
            file=sys.stderr,
        )
    elif wheels:
        _ok(f"onnxruntime package: {wheels[0]}")
    _ok("runtime libraries (FastAPI, uvicorn, httpx, ONNX Runtime, …)")
    return True


def _check_config_and_paths(root: Path) -> bool:
    sys.path.insert(0, str(root))
    cfg_override = (os.environ.get("WD_TAGGER_CONFIG_PATH") or "").strip()
    if cfg_override:
        cfg_path = Path(cfg_override).expanduser().resolve()
        if not cfg_path.is_file():
            _fail(f"WD_TAGGER_CONFIG_PATH not found: {cfg_path}")
            return False
    else:
        cfg_path = root / "config.yaml"
    example_path = root / "config.example.yaml"

    import yaml
    from pydantic import ValidationError

    from backend.config import AppConfig, stable_models_dir_for_config

    if cfg_path.is_file():
        try:
            raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            cfg = AppConfig.model_validate(raw)
        except (yaml.YAMLError, ValidationError, OSError) as e:
            _fail(f"config.yaml invalid: {e}")
            return False
        _ok("config.yaml parses and matches AppConfig")
    else:
        if example_path.is_file():
            _ok(
                "config.yaml absent — app will use config.example.yaml / code defaults "
                "(copy config.example.yaml to config.yaml to persist your settings)",
            )
        else:
            _ok("config.yaml absent — using code defaults for path checks")
        cfg = AppConfig()

    models_dir = Path(stable_models_dir_for_config(cfg.models_dir)).expanduser().resolve()
    try:
        models_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        _fail(f"models_dir not writable: {models_dir} ({e})")
        return False
    _ok(f"models_dir ready: {models_dir}")

    # Log directory (per-run files under logs/runs/)
    logs_runs = root / "logs" / "runs"
    try:
        logs_runs.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        _fail(f"cannot create logs/runs: {e}")
        return False
    _ok(f"log dir writable: {logs_runs}")

    if not _check_gpu_inference(cfg):
        return False

    return True


def _check_gpu_inference(cfg) -> bool:
    """Validate ONNX GPU execution providers for WD tagging and face models."""
    from backend.tagger.ort_providers import (
        active_gpu_provider_label,
        available_gpu_providers,
        available_ort_providers,
        gpu_config_error_message,
        insightface_ctx_id,
        resolve_ort_providers,
    )

    installed = available_ort_providers()
    _ok(f"ONNX Runtime providers: {', '.join(installed)}")
    gpu_eps = available_gpu_providers()
    if gpu_eps:
        _ok(f"GPU providers (this platform): {', '.join(gpu_eps)}")
    else:
        print(
            "check_requirements: hint: no GPU execution provider registered "
            f"(installed: {', '.join(installed)})",
            file=sys.stderr,
        )

    backend = (cfg.gpu_backend or "auto").strip().lower()
    if backend == "cpu" or (not cfg.use_gpu and backend == "auto"):
        planned = resolve_ort_providers(use_gpu=False, gpu_backend="auto")
        _ok(f"inference plan (CPU): {', '.join(planned)}")
        return True

    planned = resolve_ort_providers(use_gpu=cfg.use_gpu, gpu_backend=cfg.gpu_backend)
    gpu_active = [p for p in planned if p != "CPUExecutionProvider"]
    if gpu_active:
        label = active_gpu_provider_label(planned)
        ctx = insightface_ctx_id(planned)
        _ok(
            f"GPU inference plan — WD + face providers: {', '.join(planned)}; "
            f"active EP: {label}; InsightFace ctx_id={ctx}"
        )
        return True

    installed_joined = ", ".join(installed)
    if backend in ("cuda", "rocm", "directml"):
        _fail(
            f"gpu_backend={backend!r} requested but no matching execution provider is registered "
            f"(installed EPs: {installed_joined}). See docs/FACE_TAGGING.md and docs/DEPENDENCIES.md.",
        )
        return False

    if cfg.use_gpu:
        _fail(gpu_config_error_message())
        return False

    return True


def _check_optional_perf() -> None:
    if sys.platform != "linux":
        return
    try:
        import uvloop  # noqa: F401

        _ok("uvloop installed (optional [perf] extra — faster asyncio on Linux)")
    except ImportError:
        print(
            "check_requirements: hint: on Linux, pip install -e '.[perf]' for uvloop (optional)",
            file=sys.stderr,
        )


def _check_optional_face() -> None:
    """Face tagging routes are always mounted; hint when optional extras are missing."""
    try:
        import insightface  # noqa: F401
        import sklearn  # noqa: F401

        _ok("face tagging libraries (insightface, scikit-learn)")
    except ImportError:
        print(
            "check_requirements: hint: pip install -e '.[face]' for /api/face (InsightFace + clustering)",
            file=sys.stderr,
        )


def main() -> int:
    root = _root()
    if not (root / "run.py").is_file():
        _fail(f"run.py not found under {root} (wrong WD_TAGGER_CHECK_ROOT?)")
        return 1

    if not _check_python_version():
        return 1
    if not _check_imports():
        return 1
    if not _check_config_and_paths(root):
        return 1
    _check_optional_face()
    _check_optional_perf()
    print("check_requirements: all checks passed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

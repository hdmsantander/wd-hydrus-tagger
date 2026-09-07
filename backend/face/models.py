"""InsightFace pack cache inspect / download (buffalo_l under ``face_models_dir``)."""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

FACE_PACKS = ("buffalo_l",)

# Detection + ArcFace embedding are required for detect/recognize. Landmark/gender files are optional.
_REQUIRED_ONNX: dict[str, tuple[str, ...]] = {
    "buffalo_l": ("det_10g.onnx", "w600k_r50.onnx"),
}
_OPTIONAL_ONNX: dict[str, tuple[str, ...]] = {
    "buffalo_l": ("1k3d68.onnx", "2d106det.onnx", "genderage.onnx"),
}

# Failed / partial zips are tiny; buffalo_l ONNX files are tens of MB.
_MIN_ONNX_BYTES = 1_000_000


def face_pack_dir(models_root: Path, pack: str) -> Path:
    """InsightFace stores packs at ``{root}/models/{pack}``."""
    return Path(models_root) / "models" / pack


def inspect_face_pack(models_root: Path, pack: str) -> dict:
    """Disk status for one InsightFace pack (no download, no ONNX load)."""
    pack = (pack or "buffalo_l").strip() or "buffalo_l"
    root = Path(models_root)
    directory = face_pack_dir(root, pack)
    required = _REQUIRED_ONNX.get(pack, ())
    optional = _OPTIONAL_ONNX.get(pack, ())
    issues: list[str] = []
    files: list[dict] = []

    if not directory.is_dir():
        issues.append("pack directory missing")
    for name in required:
        path = directory / name
        size = path.stat().st_size if path.is_file() else 0
        files.append({"name": name, "present": path.is_file(), "bytes": size, "required": True})
        if not path.is_file():
            issues.append(f"missing {name}")
        elif size < _MIN_ONNX_BYTES:
            issues.append(f"{name} too small ({size} bytes)")
    for name in optional:
        path = directory / name
        size = path.stat().st_size if path.is_file() else 0
        files.append({"name": name, "present": path.is_file(), "bytes": size, "required": False})

    downloaded = bool(directory.is_dir()) and all(f["present"] for f in files if f["required"])
    cache_ok = downloaded and not issues
    return {
        "name": pack,
        "downloaded": downloaded,
        "cache_ok": cache_ok,
        "cache_issues": issues,
        "files": files,
        "path": str(directory) if directory.exists() else None,
        "repo": "insightface buffalo_l (GitHub releases v0.7)",
    }


def verify_face_pack(models_root: Path, pack: str) -> dict:
    """Same as inspect; kept as a named verify step for the Settings UI."""
    row = inspect_face_pack(models_root, pack)
    return {
        "name": row["name"],
        "ok": bool(row["cache_ok"]),
        "issues": list(row["cache_issues"]),
        "downloaded": row["downloaded"],
        "path": row["path"],
    }


def download_face_pack(models_root: Path, pack: str) -> Path:
    """Download and unzip the InsightFace pack into ``models_root/models/{pack}``."""
    from insightface.utils.storage import download

    pack = (pack or "buffalo_l").strip() or "buffalo_l"
    if pack not in FACE_PACKS:
        raise ValueError(f"Unsupported face model pack: {pack}")
    root = Path(models_root)
    root.mkdir(parents=True, exist_ok=True)
    log.info("face_pack_download start pack=%s root=%s", pack, root)
    dest = Path(download("models", pack, force=False, root=str(root)))
    checked = inspect_face_pack(root, pack)
    if not checked["cache_ok"]:
        raise FileNotFoundError(
            f"Face pack {pack} download incomplete: {'; '.join(checked['cache_issues'])}"
        )
    log.info("face_pack_download complete pack=%s dir=%s", pack, dest)
    return dest

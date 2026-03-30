#!/usr/bin/env python3
"""Interactive wizard: write config.yaml from config.example.yaml + Linux hardware hints."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("error: PyYAML is required (pip install pyyaml)", file=sys.stderr)
    sys.exit(1)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_mem_gib() -> float | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024)
    except (OSError, ValueError, IndexError):
        return None
    return None


def _cpu_threads() -> int:
    n = os.cpu_count() or 8
    return max(1, min(64, n))


def _parse_physical_cores_from_cpuinfo_text(text: str) -> int | None:
    """Count unique (physical id, core id) pairs; else cpu cores × socket count.

    Matches typical x86 /proc/cpuinfo. Returns None if no usable signal.
    """
    blocks = [b.strip() for b in text.strip().split("\n\n") if b.strip()]
    pairs: set[tuple[int, int]] = set()
    for block in blocks:
        phys: int | None = None
        core: int | None = None
        for line in block.splitlines():
            ls = line.strip().lower()
            if ls.startswith("physical id"):
                try:
                    phys = int(line.split(":", 1)[1].strip())
                except (ValueError, IndexError):
                    pass
            elif ls.startswith("core id"):
                try:
                    core = int(line.split(":", 1)[1].strip())
                except (ValueError, IndexError):
                    pass
        if phys is not None and core is not None:
            pairs.add((phys, core))
    if pairs:
        return max(1, min(64, len(pairs)))
    m = re.search(r"^cpu cores\s*:\s*(\d+)", text, re.MULTILINE)
    if m:
        per_socket = int(m.group(1))
        phys_ids = re.findall(r"^physical id\s*:\s*(\d+)", text, re.MULTILINE)
        sockets = len(set(phys_ids)) if phys_ids else 1
        return max(1, min(64, per_socket * max(1, sockets)))
    return None


def _physical_cores_hint() -> int:
    """Linux /proc/cpuinfo; else logical thread count."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return _cpu_threads()
    n = _parse_physical_cores_from_cpuinfo_text(text)
    return n if n is not None else _cpu_threads()


def _nvidia_gpu_present() -> bool:
    try:
        r = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return r.returncode == 0 and bool((r.stdout or "").strip())
    except (OSError, subprocess.TimeoutExpired):
        return False


def _prompt(msg: str, default: str = "") -> str:
    if default:
        raw = input(f"{msg} [{default}]: ").strip()
        return raw if raw else default
    return input(f"{msg}: ").strip()


def _prompt_yes_no(msg: str, default_yes: bool) -> bool:
    suf = "Y/n" if default_yes else "y/N"
    raw = input(f"{msg} ({suf}): ").strip().lower()
    if not raw:
        return default_yes
    return raw in ("y", "yes", "1", "true")


def _prompt_int(msg: str, default: int, *, min_v: int, max_v: int) -> int:
    raw = _prompt(msg, str(default)).strip()
    try:
        v = int(raw, 10)
    except ValueError:
        v = default
    return max(min_v, min(max_v, v))


def _suggest_batch_and_parallel(mem_gib: float | None, large_model: bool) -> tuple[int, int]:
    if mem_gib is None:
        return (6 if large_model else 8, 6 if large_model else 8)
    if large_model:
        if mem_gib < 14:
            return (2, 4)
        if mem_gib < 24:
            return (4, 4)
        return (4, 6)
    if mem_gib < 8:
        return (4, 4)
    return (8, 8)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate config.yaml interactively from config.example.yaml.")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output path (default: <repo>/config.yaml)",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Overwrite output without asking (still backs up existing config.yaml to .bak)",
    )
    args = parser.parse_args()

    if sys.platform != "linux":
        print(
            "error: generate_config.py is supported on Linux only "
            "(reads /proc/meminfo and /proc/cpuinfo; optional nvidia-smi).",
            file=sys.stderr,
        )
        print("  Copy config.example.yaml to config.yaml and edit manually on other platforms.", file=sys.stderr)
        return 2

    root = _repo_root()
    example = root / "config.example.yaml"
    out = args.output or (root / "config.yaml")

    if not example.is_file():
        print(f"error: missing {example}", file=sys.stderr)
        return 1

    with open(example, encoding="utf-8") as f:
        cfg: dict = yaml.safe_load(f) or {}

    mem = _read_mem_gib()
    threads = _cpu_threads()
    phys = max(1, min(64, _physical_cores_hint()))
    gpu = _nvidia_gpu_present()

    print()
    print("WD Hydrus Tagger — config wizard (Linux)")
    print("Uses values from config.example.yaml as a base, then adjusts for this machine.")
    if mem is not None:
        print(f"Detected ~{mem:.1f} GiB RAM, ~{phys} CPU core(s) (physical hint), nvidia GPU: {gpu}")
    else:
        print(f"Could not read MemTotal; using ~{threads} logical thread(s). nvidia GPU: {gpu}")
    print()

    cfg["hydrus_api_url"] = _prompt("Hydrus Client API URL", str(cfg.get("hydrus_api_url") or "http://localhost:45869"))
    key_default = str(cfg.get("hydrus_api_key") or "")
    cfg["hydrus_api_key"] = _prompt("Hydrus API key (paste yours)", key_default)

    print("\nModels: wd-vit-tagger-v3 (light), wd-swinv2-tagger-v3, wd-vit-large-tagger-v3, wd-eva02-large-tagger-v3 (heavy)")
    model_default = str(cfg.get("default_model") or "wd-vit-tagger-v3")
    cfg["default_model"] = _prompt("Default ONNX model id", model_default)
    large = cfg["default_model"] in ("wd-vit-large-tagger-v3", "wd-eva02-large-tagger-v3")

    use_gpu_default = bool(cfg.get("use_gpu")) or gpu
    cfg["use_gpu"] = _prompt_yes_no("Enable CUDA / GPU for ONNX (requires onnxruntime-gpu)", use_gpu_default)

    cfg["cpu_intra_op_threads"] = _prompt_int(
        "ONNX CPU intra_op threads (physical cores recommended)", phys, min_v=1, max_v=64,
    )
    cfg["cpu_inter_op_threads"] = _prompt_int(
        "ONNX CPU inter_op threads (usually 1)", 1, min_v=1, max_v=16,
    )

    bs, par = _suggest_batch_and_parallel(mem, large)
    print(f"\nSuggested batch_size={bs}, hydrus_download_parallel={par} for this model/RAM hint.")
    cfg["batch_size"] = _prompt_int("Inference batch_size (ONNX)", bs, min_v=1, max_v=256)
    cfg["hydrus_download_parallel"] = _prompt_int(
        "Parallel Hydrus image downloads per batch", par, min_v=1, max_v=32,
    )

    chunk_default = 512 if _prompt_yes_no("Mostly huge Tag all / gallery searches (fewer Hydrus metadata round-trips)?", False) else 256
    cfg["hydrus_metadata_chunk_size"] = _prompt_int(
        "get_file_metadata chunk size (32–2048)", chunk_default, min_v=32, max_v=2048,
    )

    cfg["apply_tags_every_n"] = _prompt_int(
        "Default apply_tags_every_n for incremental Hydrus writes (0=off)",
        int(cfg.get("apply_tags_every_n") or 8),
        min_v=0,
        max_v=256,
    )

    cfg["wd_skip_inference_if_marker_present"] = _prompt_yes_no(
        "Skip ONNX when wd14:<model> marker already on file (recommended)", True,
    )
    cfg["wd_skip_if_higher_tier_model_present"] = _prompt_yes_no(
        "Skip ONNX when a heavier WD model marker is already on file (fast Tag all on mixed libraries)", True,
    )

    cfg["target_tag_service"] = _prompt("Default Hydrus tag service display name", str(cfg.get("target_tag_service") or ""))

    host_default = str(cfg.get("host") or "127.0.0.1")
    if _prompt_yes_no("Listen on all interfaces (0.0.0.0) for LAN browsers?", host_default == "0.0.0.0"):
        cfg["host"] = "0.0.0.0"
    else:
        cfg["host"] = "127.0.0.1"
    cfg["port"] = _prompt_int("Web UI port", int(cfg.get("port") or 8199), min_v=1, max_v=65535)

    if out.exists() and not args.force:
        if not _prompt_yes_no(f"{out} exists — overwrite?", False):
            print("Aborted.")
            return 1

    if out.exists():
        bak = out.with_suffix(out.suffix + ".bak")
        shutil.copy2(out, bak)
        print(f"Backed up existing file to {bak}")

    with open(out, "w", encoding="utf-8") as f:
        f.write("# Generated by scripts/generate_config.py — review and adjust.\n")
        yaml.safe_dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"\nWrote {out}")
    print("Next: ./wd-hydrus-tagger.sh check && ./wd-hydrus-tagger.sh run")
    print("See docs/PERFORMANCE_AND_TUNING.md for Tag all, markers, and performance overlay.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

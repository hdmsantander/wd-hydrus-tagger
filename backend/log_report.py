"""Parse run logs for tracing: errors, ONNX/disk cache lines, Hydrus metadata fetches."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LogDigest:
    """Aggregates grep-style signals from one log file."""

    path: str = ""
    lines: int = 0
    error_count: int = 0
    warning_count: int = 0
    # ensure_model
    memory_cache_hit_true: int = 0
    memory_cache_hit_false: int = 0
    # load_model metrics (disk_cache_hit=…)
    disk_cache_hit_true: int = 0
    disk_cache_hit_false: int = 0
    disk_cache_hit_na: int = 0
    disk_cache_miss_lines: int = 0
    disk_cache_invalid_lines: int = 0
    hub_fetch_true: int = 0
    hub_fetch_false: int = 0
    # Hydrus metadata
    tag_files_metadata_fetched: int = 0
    files_metadata_hydrus: int = 0
    application_ready_lines: int = 0
    sample_errors: list[str] = field(default_factory=list)

    def max_error_samples(self) -> int:
        return 8


_RE_MEMORY_TRUE = re.compile(r"\bmemory_cache_hit=True\b")
_RE_MEMORY_FALSE = re.compile(r"\bmemory_cache_hit=False\b")
_RE_DISK_TRUE = re.compile(r"\bdisk_cache_hit=True\b")
_RE_DISK_FALSE = re.compile(r"\bdisk_cache_hit=False\b")
_RE_DISK_NA = re.compile(r"\bdisk_cache_hit=n/a\b")
_RE_HUB_TRUE = re.compile(r"\bhub_fetch_this_call=True\b")
_RE_HUB_FALSE = re.compile(r"\bhub_fetch_this_call=False\b")


def analyze_log_text(text: str, *, path: str = "") -> LogDigest:
    d = LogDigest(path=path)
    for line in text.splitlines():
        d.lines += 1
        if " ERROR [" in line:
            d.error_count += 1
            if len(d.sample_errors) < d.max_error_samples():
                d.sample_errors.append(line.strip()[:500])
        if " WARNING [" in line:
            d.warning_count += 1
        if _RE_MEMORY_TRUE.search(line):
            d.memory_cache_hit_true += 1
        if _RE_MEMORY_FALSE.search(line):
            d.memory_cache_hit_false += 1
        if _RE_DISK_TRUE.search(line):
            d.disk_cache_hit_true += 1
        if _RE_DISK_FALSE.search(line):
            d.disk_cache_hit_false += 1
        if _RE_DISK_NA.search(line):
            d.disk_cache_hit_na += 1
        if "disk cache miss" in line:
            d.disk_cache_miss_lines += 1
        if "disk_cache_invalid" in line:
            d.disk_cache_invalid_lines += 1
        if _RE_HUB_TRUE.search(line):
            d.hub_fetch_true += 1
        if _RE_HUB_FALSE.search(line):
            d.hub_fetch_false += 1
        if (
            "tag_files metadata_fetched" in line
            or "tag_files metadata rows=" in line
            or "tagging_ws metadata_prefetch" in line
        ):
            d.tag_files_metadata_fetched += 1
        if "files metadata_hydrus" in line:
            d.files_metadata_hydrus += 1
        if "Application ready host=" in line:
            d.application_ready_lines += 1
    return d


def analyze_log_path(path: Path) -> LogDigest:
    text = path.read_text(encoding="utf-8", errors="replace")
    return analyze_log_text(text, path=str(path.resolve()))


def format_digest(d: LogDigest) -> str:
    lines_out = [
        f"Log digest: {d.path or '(stdin)'}",
        f"  lines: {d.lines}",
        f"  ERROR [{d.error_count}]  WARNING [{d.warning_count}]",
        "  ONNX / disk cache (log keyword counts):",
        f"    ensure_model memory_cache_hit=True  → {d.memory_cache_hit_true}",
        f"    ensure_model memory_cache_hit=False → {d.memory_cache_hit_false}",
        f"    load_model disk_cache_hit=True      → {d.disk_cache_hit_true}",
        f"    load_model disk_cache_hit=False     → {d.disk_cache_hit_false}",
        f"    load_model disk_cache_hit=n/a       → {d.disk_cache_hit_na}",
        f"    load_model 'disk cache miss' lines  → {d.disk_cache_miss_lines}",
        f"    load_model disk_cache_invalid       → {d.disk_cache_invalid_lines}",
        f"    hub_fetch_this_call=True            → {d.hub_fetch_true}",
        f"    hub_fetch_this_call=False           → {d.hub_fetch_false}",
        "  Hydrus metadata:",
        f"    tag_files / tagging_ws metadata lines → {d.tag_files_metadata_fetched}",
        f"    files metadata_hydrus (gallery API)   → {d.files_metadata_hydrus}",
        f"  Application ready lines → {d.application_ready_lines}",
    ]
    if d.sample_errors:
        lines_out.append("  Sample ERROR lines:")
        for s in d.sample_errors:
            lines_out.append(f"    {s}")
    return "\n".join(lines_out)

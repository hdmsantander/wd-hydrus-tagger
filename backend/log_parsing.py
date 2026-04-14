"""Shared log parsing patterns for reporting scripts/tools."""

from __future__ import annotations

import re

RE_MEMORY_TRUE = re.compile(r"\bmemory_cache_hit=True\b")
RE_MEMORY_FALSE = re.compile(r"\bmemory_cache_hit=False\b")
RE_DISK_TRUE = re.compile(r"\bdisk_cache_hit=True\b")
RE_DISK_FALSE = re.compile(r"\bdisk_cache_hit=False\b")
RE_DISK_NA = re.compile(r"\bdisk_cache_hit=n/a\b")
RE_HUB_TRUE = re.compile(r"\bhub_fetch_this_call=True\b")
RE_HUB_FALSE = re.compile(r"\bhub_fetch_this_call=False\b")


def parse_tag_files_metrics_line(line: str) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    m = re.search(r"wall_onnx_predict_s=([\d.]+)", line)
    if m:
        out["wall_onnx_predict_s"] = float(m.group(1))
    m = re.search(r"wall_hydrus_fetch_s=([\d.]+)", line)
    if m:
        out["wall_hydrus_fetch_s"] = float(m.group(1))
    m = re.search(r"skipped_pre_infer_marker_files=(\d+)", line)
    if m:
        out["skipped_pre_infer_marker_files"] = int(m.group(1))
    m = re.search(r"inferred_files=(\d+)", line)
    if m:
        out["inferred_files"] = int(m.group(1))
    return out


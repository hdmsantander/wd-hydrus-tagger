"""T-Learn (§5): split Tag all queue into a learning prefix and commit suffix.

Bytes scope falls back to count in v1 (logged in ``split_info``).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Literal

log = logging.getLogger(__name__)

DEFAULT_LEARNING_FRACTION = 0.1
# §5.2: at least ~one "mini-epoch" of batches; cap by N−1 so commit stays non-empty when N>1.
MIN_LEARNING_PREFIX_FILES = 32


def parse_learning_fraction(raw: object | None) -> float:
    """Client ``learning_fraction``; default 0.1, clamped 0.01–0.5."""
    if raw is None:
        return DEFAULT_LEARNING_FRACTION
    try:
        f = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_LEARNING_FRACTION
    return max(0.01, min(0.5, f))


def compute_learning_split(
    file_ids: list[int],
    *,
    learning_fraction: float,
    learning_scope: Literal["count", "bytes"] | str = "count",
) -> tuple[list[int], list[int], dict[str, Any]]:
    """Partition ``file_ids`` into ``(learning_prefix, commit_suffix)`` and metadata.

    Guarantees: learning and commit are contiguous slices; ``len(learning)+len(commit)==len(file_ids)``.
    For ``n>1``, at least one file remains in commit when possible (§5.2).
    """
    n = len(file_ids)
    info: dict[str, Any] = {
        "learning_fraction": float(learning_fraction),
        "learning_scope_requested": str(learning_scope),
    }
    if n == 0:
        info["learning_count"] = 0
        info["commit_count"] = 0
        return [], [], info

    scope = str(learning_scope).strip().lower()
    if scope not in ("count", "bytes"):
        scope = "count"
    if scope == "bytes":
        log.info(
            "learning_calibration learning_scope=bytes requires metadata prefetch; "
            "use compute_learning_split_by_bytes after Hydrus metadata (count split here for API symmetry only)",
        )
        info["learning_scope_effective"] = "count"
        info["bytes_fallback"] = True
    else:
        info["learning_scope_effective"] = scope

    frac = parse_learning_fraction(learning_fraction)

    if n == 1:
        info["learning_count"] = 1
        info["commit_count"] = 0
        info["split_index"] = 1
        return list(file_ids), [], info

    k_raw = int(math.ceil(frac * n))
    k = max(MIN_LEARNING_PREFIX_FILES, k_raw)
    k = min(k, n - 1)
    k = max(1, k)

    learning = list(file_ids[:k])
    commit = list(file_ids[k:])
    info["learning_count"] = len(learning)
    info["commit_count"] = len(commit)
    info["split_index"] = k
    return learning, commit, info


def compute_learning_split_by_bytes(
    file_ids: list[int],
    *,
    meta_by_id: dict[int, dict] | None,
    learning_fraction: float,
    min_prefix: int = MIN_LEARNING_PREFIX_FILES,
) -> tuple[list[int], list[int], dict[str, Any]]:
    """Learning prefix by cumulative Hydrus ``size`` (bytes) until ``ceil(fraction * total_bytes)``.

    Falls back to :func:`compute_learning_split` (count) when metadata is missing, sizes are absent,
    or total known bytes is zero.
    """
    frac = parse_learning_fraction(learning_fraction)
    info: dict[str, Any] = {
        "learning_fraction": float(learning_fraction),
        "learning_scope_requested": "bytes",
    }
    n = len(file_ids)
    if n == 0:
        info["learning_count"] = 0
        info["commit_count"] = 0
        return [], [], info

    if meta_by_id is None:
        log.info("learning_calibration bytes: no metadata map; count fallback")
        la, co, inf2 = compute_learning_split(file_ids, learning_fraction=frac, learning_scope="count")
        inf2["learning_scope_requested"] = "bytes"
        inf2["learning_scope_effective"] = "count"
        inf2["bytes_fallback"] = "no_metadata"
        return la, co, inf2

    sizes: list[int | None] = []
    for fid in file_ids:
        row = meta_by_id.get(int(fid))
        if not row:
            sizes.append(None)
            continue
        sz = row.get("size")
        if sz is None:
            sizes.append(None)
        else:
            try:
                sizes.append(int(sz))
            except (TypeError, ValueError):
                sizes.append(None)

    if any(s is None for s in sizes):
        log.info("learning_calibration bytes: missing size on one or more files; count fallback")
        la, co, inf2 = compute_learning_split(file_ids, learning_fraction=frac, learning_scope="count")
        inf2["learning_scope_requested"] = "bytes"
        inf2["learning_scope_effective"] = "count"
        inf2["bytes_fallback"] = "missing_size"
        return la, co, inf2

    total_b = sum(sizes)
    info["total_bytes_known"] = total_b
    if total_b <= 0:
        la, co, inf2 = compute_learning_split(file_ids, learning_fraction=frac, learning_scope="count")
        inf2["learning_scope_requested"] = "bytes"
        inf2["learning_scope_effective"] = "count"
        inf2["bytes_fallback"] = "zero_total_bytes"
        return la, co, inf2

    target = max(1, int(math.ceil(frac * total_b)))
    info["target_bytes"] = target

    if n == 1:
        info["learning_count"] = 1
        info["commit_count"] = 0
        info["split_index"] = 1
        info["learning_scope_effective"] = "bytes"
        return list(file_ids), [], info

    acc = 0
    k = 0
    for i, s in enumerate(sizes):
        acc += s
        k = i + 1
        if acc >= target:
            break

    k = max(min_prefix, k)
    k = min(k, n - 1)
    k = max(1, k)

    learning = list(file_ids[:k])
    commit = list(file_ids[k:])
    info["learning_scope_effective"] = "bytes"
    info["learning_count"] = len(learning)
    info["commit_count"] = len(commit)
    info["split_index"] = k
    info["bytes_cumulative_to_split"] = sum(sizes[:k])
    return learning, commit, info

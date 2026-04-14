"""Pure helpers: Hydrus API metadata rows → file_id → row maps."""

from __future__ import annotations

from typing import Iterable


def rows_to_file_id_map(rows: Iterable[object]) -> dict[int, dict]:
    """Build ``file_id`` → metadata row from Hydrus ``get_file_metadata`` rows.

    - Skips non-dict rows, rows without ``file_id``, and non-coercible ids.
    - On duplicate ``file_id``, the **last** row in iteration order wins.
    """
    out: dict[int, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        fid = row.get("file_id")
        if fid is None:
            continue
        try:
            key = int(fid)
        except (TypeError, ValueError):
            continue
        out[key] = row
    return out

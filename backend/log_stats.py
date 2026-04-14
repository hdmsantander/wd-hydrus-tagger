"""Structured ``stats`` log lines for timing and counters (grep/AI-friendly, stable key order).

Use ``log_stats(logger, "operation", duration_ms=12, files=50)`` — message is always::

    stats op=<operation> <sorted key=value pairs>

Values are escaped so one line stays one record; booleans become ``true``/``false``.
"""

from __future__ import annotations

import logging
import math


def _fmt_stats_value(v: object) -> str:
    if v is True:
        return "true"
    if v is False:
        return "false"
    if v is None:
        return ""
    if isinstance(v, float):
        if not math.isfinite(v):
            return str(v)
        s = f"{v:.6f}".rstrip("0").rstrip(".")
        return s if s else "0"
    if isinstance(v, int):
        return str(v)
    s = str(v).replace("\n", " ").replace("\r", " ")
    if not s:
        return ""
    if " " in s or "=" in s or '"' in s:
        return repr(s)
    return s


def log_stats(
    logger: logging.Logger,
    op: str,
    /,
    *,
    level: int = logging.INFO,
    **kwargs: object,
) -> None:
    """Emit one line ``stats op=… k=v …`` with keys sorted (excluding ``op``)."""
    if not kwargs:
        logger.log(level, "stats op=%s", op)
        return
    tail = " ".join(f"{k}={_fmt_stats_value(v)}" for k, v in sorted(kwargs.items()))
    logger.log(level, "stats op=%s %s", op, tail)

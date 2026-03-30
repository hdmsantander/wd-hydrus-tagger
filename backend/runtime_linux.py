"""Linux-oriented runtime helpers (optional faster asyncio, documented CPU thread hygiene)."""

from __future__ import annotations

import sys


def uvicorn_loop_setting() -> str:
    """Event loop policy for Uvicorn.

    On Linux, **uvloop** (install optional extra ``perf``) replaces the default asyncio
    loop and typically improves concurrent I/O (Hydrus HTTP + WebSockets). On other
    platforms or when uvloop is absent, returns ``\"auto\"``.
    """
    if sys.platform != "linux":
        return "auto"
    try:
        import uvloop  # noqa: F401
    except ImportError:
        return "auto"
    return "uvloop"

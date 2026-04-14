"""Classify httpx / network failures talking to Hydrus (daemon down, TCP reset, etc.)."""

from __future__ import annotations

import httpx


def is_hydrus_transport_error(exc: BaseException) -> bool:
    """True when Hydrus is likely unreachable or the connection broke mid-request."""
    if isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.ReadError,
            httpx.WriteError,
            httpx.TimeoutException,
            httpx.RemoteProtocolError,
            httpx.ProxyError,
        ),
    ):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        resp = exc.response
        if resp is None:
            return False
        code = resp.status_code
        return code >= 502 or code == 408
    return False

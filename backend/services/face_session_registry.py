"""Track active face-detection WebSocket sessions for coordinated graceful shutdown."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

_log = logging.getLogger(__name__)

_lock = threading.Lock()
_sessions: list[FaceSessionHandle] = []
_shutdown_notifiers: list[Callable[[], Awaitable[None]]] = []


@dataclass(frozen=True)
class FaceSessionHandle:
    cancel_event: asyncio.Event


def register_face_session(handle: FaceSessionHandle) -> None:
    with _lock:
        _sessions.append(handle)


def unregister_face_session(handle: FaceSessionHandle) -> None:
    with _lock:
        try:
            _sessions.remove(handle)
        except ValueError:
            pass


def active_face_sessions_count() -> int:
    with _lock:
        return len(_sessions)


def register_face_shutdown_notifier(notify: Callable[[], Awaitable[None]]) -> None:
    with _lock:
        _shutdown_notifiers.append(notify)


def unregister_face_shutdown_notifier(notify: Callable[[], Awaitable[None]]) -> None:
    with _lock:
        try:
            _shutdown_notifiers.remove(notify)
        except ValueError:
            pass


async def announce_shutdown_to_face_sessions() -> int:
    """Send server_shutting_down on all registered face WebSocket sessions."""
    with _lock:
        cbs: list[Callable[[], Awaitable[None]]] = list(_shutdown_notifiers)
    if not cbs:
        _log.debug("face_shutdown: no WebSocket shutdown notifiers registered")
        return 0
    _log.info(
        "face_shutdown: broadcasting server_shutting_down to %s active face session(s)",
        len(cbs),
    )
    await asyncio.gather(*[c() for c in cbs], return_exceptions=True)
    return len(cbs)


def signal_all_face_sessions_cancel() -> int:
    with _lock:
        handles = list(_sessions)
    for h in handles:
        h.cancel_event.set()
    if handles:
        _log.debug("face_shutdown: cancel_event set on %s session(s)", len(handles))
    return len(handles)


def get_face_session_status() -> dict:
    return {"active": active_face_sessions_count() > 0, "sessions": active_face_sessions_count()}

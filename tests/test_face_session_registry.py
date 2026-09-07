"""Face WebSocket session registry."""

import asyncio

import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]

from backend.services.face_session_registry import (
    FaceSessionHandle,
    active_face_sessions_count,
    announce_shutdown_to_face_sessions,
    register_face_session,
    register_face_shutdown_notifier,
    signal_all_face_sessions_cancel,
    unregister_face_session,
    unregister_face_shutdown_notifier,
)


def test_face_registry_cancel_and_count():
    cancel = asyncio.Event()
    handle = FaceSessionHandle(cancel_event=cancel)
    register_face_session(handle)
    assert active_face_sessions_count() == 1
    assert signal_all_face_sessions_cancel() == 1
    assert cancel.is_set()
    unregister_face_session(handle)
    assert active_face_sessions_count() == 0


@pytest.mark.asyncio
async def test_face_registry_announce_shutdown():
    calls = 0

    async def notifier():
        nonlocal calls
        calls += 1

    register_face_shutdown_notifier(notifier)
    assert await announce_shutdown_to_face_sessions() == 1
    assert calls == 1
    unregister_face_shutdown_notifier(notifier)

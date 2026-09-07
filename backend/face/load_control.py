"""Dedicated executor + generation tokens for InsightFace model loads.

A hung GPU compile cannot be interrupted in-process; on timeout or shutdown we bump the
generation, replace the worker executor (abandoning the stale thread as a daemon), and
unload any partial model state so a CPU retry or process exit can proceed cleanly.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")

_lock = threading.Lock()
_generation = 0
_executor: ThreadPoolExecutor | None = None
_shutdown = False


def _make_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="face-ort-load")


def get_face_load_executor() -> ThreadPoolExecutor:
    global _executor
    with _lock:
        if _shutdown:
            raise RuntimeError("face load executor shut down")
        if _executor is None:
            _executor = _make_executor()
        return _executor


def bump_load_generation() -> int:
    global _generation
    with _lock:
        _generation += 1
        return _generation


def current_load_generation() -> int:
    with _lock:
        return _generation


def is_load_generation_current(generation: int) -> bool:
    with _lock:
        return generation == _generation


def abandon_face_load_worker(reason: str) -> int:
    """Invalidate in-flight loads and replace the single-worker executor."""
    global _executor, _generation
    with _lock:
        _generation += 1
        new_gen = _generation
        old = _executor
        if old is not None:
            log.warning(
                "face load worker abandoned reason=%s generation=%s (stale thread may linger until exit)",
                reason,
                new_gen,
            )
            try:
                old.shutdown(wait=False, cancel_futures=True)
            except Exception:
                log.exception("face load executor shutdown failed")
            _executor = None
        else:
            log.info("face load generation bumped reason=%s generation=%s", reason, new_gen)
        return new_gen


def submit_face_load(fn: Callable[[], T]) -> Future[T]:
    return get_face_load_executor().submit(fn)


def cancel_pending_face_loads(reason: str = "cancel") -> int:
    """Bump generation and drop the worker pool so hung loads cannot block the next attempt."""
    return abandon_face_load_worker(reason)


def shutdown_face_load_executor(wait: bool = False, reason: str = "shutdown") -> None:
    global _executor, _shutdown, _generation
    with _lock:
        _shutdown = True
        _generation += 1
        old = _executor
        _executor = None
    if old is not None:
        log.info("face load executor shutting down wait=%s reason=%s", wait, reason)
        try:
            old.shutdown(wait=wait, cancel_futures=True)
        except Exception:
            log.exception("face load executor shutdown failed")


def reset_face_load_control_for_tests() -> None:
    """Re-enable executor after tests that call shutdown."""
    global _executor, _shutdown, _generation
    with _lock:
        if _executor is not None:
            try:
                _executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
        _executor = None
        _shutdown = False
        _generation = 0

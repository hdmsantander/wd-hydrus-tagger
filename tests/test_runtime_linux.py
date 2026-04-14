"""Optional Linux asyncio loop selection."""

import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]

from backend.runtime_linux import uvicorn_loop_setting


def test_uvicorn_loop_setting_is_auto_or_uvloop():
    v = uvicorn_loop_setting()
    assert v in ("auto", "uvloop")

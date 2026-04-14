"""Structured ``stats`` log helper (machine-parseable lines)."""

import logging

import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]

from backend.log_stats import log_stats


def test_log_stats_sorted_keys_and_op(caplog):
    caplog.set_level(logging.INFO, logger="stats_test")
    log = logging.getLogger("stats_test")
    log_stats(log, "unit_op", z_last=1, a_first=2, m_mid=3)
    assert len(caplog.records) == 1
    msg = caplog.records[0].getMessage()
    assert msg.startswith("stats op=unit_op ")
    assert "a_first=2" in msg
    assert "m_mid=3" in msg
    assert "z_last=1" in msg
    # sorted order: a_first, m_mid, z_last
    assert msg.index("a_first") < msg.index("m_mid") < msg.index("z_last")


def test_log_stats_escapes_spaces(caplog):
    caplog.set_level(logging.INFO, logger="stats_test")
    log = logging.getLogger("stats_test")
    log_stats(log, "x", path="a b")
    msg = caplog.records[0].getMessage()
    assert "path=" in msg
    assert "'a b'" in msg or '"a b"' in msg

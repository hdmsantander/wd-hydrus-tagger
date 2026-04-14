"""Startup listen hints (LAN URLs when binding 0.0.0.0)."""

import io
import socket
from unittest.mock import patch

import pytest

pytestmark = [pytest.mark.full, pytest.mark.core]

from backend.listen_hints import (
    binds_all_ipv4_interfaces,
    non_loopback_ipv4_addresses,
    print_startup_listen_hint,
)


def test_binds_all_ipv4_interfaces():
    assert binds_all_ipv4_interfaces("0.0.0.0") is True
    assert binds_all_ipv4_interfaces("") is True
    assert binds_all_ipv4_interfaces("127.0.0.1") is False


def test_non_loopback_ipv4_addresses_swallows_oserror():
    with patch("socket.gethostname", side_effect=OSError("no host")):
        assert non_loopback_ipv4_addresses() == []


def test_non_loopback_ipv4_respects_max_n(monkeypatch):
    def fake_getaddrinfo(host, port, family, type, proto=0, flags=0):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", (f"10.1.0.{i}", 0))
            for i in range(30)
        ]

    monkeypatch.setattr(socket, "gethostname", lambda: "box")
    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    assert len(non_loopback_ipv4_addresses(max_n=4)) == 4


def test_non_loopback_ipv4_from_getaddrinfo(monkeypatch):
    def fake_getaddrinfo(host, port, family, type, proto=0, flags=0):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.2", 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 0)),
        ]

    monkeypatch.setattr(socket, "gethostname", lambda: "box")
    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    ips = non_loopback_ipv4_addresses(max_n=12)
    assert ips == ["10.0.0.2"]


def test_print_startup_listen_hint_skips_non_wildcard():
    buf = io.StringIO()
    print_startup_listen_hint("127.0.0.1", 8199, stream=buf)
    assert buf.getvalue() == ""


def test_print_startup_listen_hint_lists_ips(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setenv("WD_TAGGER_LISTEN_HINT_VERBOSE", "1")
    monkeypatch.setattr(
        "backend.listen_hints.non_loopback_ipv4_addresses",
        lambda max_n=12: ["10.0.0.5"],
    )
    print_startup_listen_hint("0.0.0.0", 8199, stream=buf)
    text = buf.getvalue()
    assert "0.0.0.0:8199" in text
    assert "http://10.0.0.5:8199/" in text


def test_print_startup_listen_hint_default_stderr_one_line(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr(
        "backend.listen_hints.non_loopback_ipv4_addresses",
        lambda max_n=12: ["10.0.0.5"],
    )
    print_startup_listen_hint("0.0.0.0", 8199, stream=buf)
    text = buf.getvalue()
    assert "0.0.0.0:8199" in text
    assert "WD_TAGGER_LISTEN_HINT_VERBOSE" in text
    assert "http://10.0.0.5" not in text


def test_print_startup_listen_hint_no_ips_fallback(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setenv("WD_TAGGER_LISTEN_HINT_VERBOSE", "1")
    monkeypatch.setattr("backend.listen_hints.non_loopback_ipv4_addresses", lambda max_n=12: [])
    print_startup_listen_hint("0.0.0.0", 9000, stream=buf)
    assert "hostname lookup" in buf.getvalue() or "ip -4 addr" in buf.getvalue()

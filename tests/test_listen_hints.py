"""Bind-address helpers for LAN access."""

from backend.listen_hints import binds_all_ipv4_interfaces, non_loopback_ipv4_addresses


def test_binds_all_ipv4_interfaces():
    assert binds_all_ipv4_interfaces("0.0.0.0") is True
    assert binds_all_ipv4_interfaces("") is True
    assert binds_all_ipv4_interfaces("127.0.0.1") is False


def test_non_loopback_ipv4_returns_list():
    ips = non_loopback_ipv4_addresses(max_n=3)
    assert isinstance(ips, list)
    for ip in ips:
        assert not ip.startswith("127.")

"""Log how to reach the app from other machines when binding all interfaces."""

from __future__ import annotations

import socket
import sys


def non_loopback_ipv4_addresses(max_n: int = 12) -> list[str]:
    """Best-effort LAN IPv4s for this host (via ``getaddrinfo(hostname)``)."""
    seen: set[str] = set()
    out: list[str] = []
    try:
        hostname = socket.gethostname()
        for res in socket.getaddrinfo(hostname, None, socket.AF_INET, socket.SOCK_STREAM):
            ip = res[4][0]
            if ip.startswith("127."):
                continue
            if ip not in seen:
                seen.add(ip)
                out.append(ip)
                if len(out) >= max_n:
                    break
    except OSError:
        pass
    return out


def binds_all_ipv4_interfaces(host: str) -> bool:
    return host in ("0.0.0.0", "")


def print_startup_listen_hint(host: str, port: int, *, stream=None) -> None:
    """Print stderr hints when the server listens on all IPv4 interfaces."""
    stream = stream or sys.stderr
    if not binds_all_ipv4_interfaces(host):
        return
    ips = non_loopback_ipv4_addresses()
    print(
        f"wd-hydrus-tagger: listening on 0.0.0.0:{port} — from another device use "
        f"http://<this-machine-ip>:{port}/ (allow TCP {port} in the firewall if needed)",
        file=stream,
    )
    for ip in ips:
        print(f"  e.g. http://{ip}:{port}/", file=stream)
    if not ips:
        print(
            "  (no non-loopback IPv4 from hostname lookup; run `ip -4 addr` or `hostname -I` for your LAN IP)",
            file=stream,
        )

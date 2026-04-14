"""Log how to reach the app from other machines when binding all interfaces."""

from __future__ import annotations

import logging
import os
import socket
import sys

from backend.log_stats import log_stats


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


def _listen_hint_verbose() -> bool:
    return os.environ.get("WD_TAGGER_LISTEN_HINT_VERBOSE", "").strip().lower() in ("1", "true", "yes", "on")


def log_startup_listen_hint(logger: logging.Logger, host: str, port: int) -> None:
    """One INFO ``stats`` line + per-IP DEBUG when bound to all interfaces."""
    if not binds_all_ipv4_interfaces(host):
        return
    ips = non_loopback_ipv4_addresses()
    examples = ",".join(f"http://{ip}:{port}/" for ip in ips[:6]) if ips else ""
    log_stats(
        logger,
        "startup_listen",
        bind="0.0.0.0",
        port=port,
        example_urls=examples or "(none_from_hostname)",
    )
    for ip in ips:
        logger.debug("startup_listen candidate url=http://%s:%s/", ip, port)


def print_startup_listen_hint(host: str, port: int, *, stream=None) -> None:
    """Stderr: one line by default; per-IP lines only if ``WD_TAGGER_LISTEN_HINT_VERBOSE=1``."""
    stream = stream or sys.stderr
    if not binds_all_ipv4_interfaces(host):
        return
    ips = non_loopback_ipv4_addresses()
    print(
        f"wd-hydrus-tagger: listening on 0.0.0.0:{port} — use http://<LAN-IP>:{port}/ "
        f"(set WD_TAGGER_LISTEN_HINT_VERBOSE=1 for example URLs on stderr)",
        file=stream,
    )
    if not _listen_hint_verbose():
        return
    for ip in ips:
        print(f"  e.g. http://{ip}:{port}/", file=stream)
    if not ips:
        print(
            "  (no non-loopback IPv4 from hostname lookup; run `ip -4 addr` or `hostname -I` for your LAN IP)",
            file=stream,
        )

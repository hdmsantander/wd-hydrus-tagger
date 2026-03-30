#!/usr/bin/env python3
"""Summarize logs/latest.log: errors, cache hit/miss signals, Hydrus metadata logs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.log_report import analyze_log_path, format_digest  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "log_path",
        nargs="?",
        default=None,
        help="Log file (default: <repo>/logs/latest.log)",
    )
    ap.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit 1 if any ERROR-level log lines were counted",
    )
    args = ap.parse_args()
    path = Path(args.log_path) if args.log_path else ROOT / "logs" / "latest.log"
    if not path.is_file():
        print(f"error: log file not found: {path}", file=sys.stderr)
        sys.exit(2)
    digest = analyze_log_path(path)
    print(format_digest(digest))
    if args.fail_on_error and digest.error_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

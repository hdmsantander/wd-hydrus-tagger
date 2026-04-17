#!/usr/bin/env python3
"""Fail if critical modules are below line coverage thresholds (run after ``coverage run -m pytest``).

Usage (from repo root)::

    .venv/bin/python -m coverage run -m pytest -q
    .venv/bin/python scripts/check_critical_coverage.py

Or one shot (``--no-cov`` avoids pytest-cov fighting ``coverage run``)::

    .venv/bin/python -m coverage run -m pytest -q --no-cov && .venv/bin/python scripts/check_critical_coverage.py

After a normal ``pytest`` run (with dev extras, coverage is enabled via ``addopts``), the same script reads the existing ``.coverage`` data::

    .venv/bin/pytest -q && .venv/bin/python scripts/check_critical_coverage.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Module substrings as reported by ``coverage report`` (path-style).
# Line+branch combined ``coverage`` %; learning split has one defensive for-loop branch (byte walk)
# that is provably always exited via ``break`` when ``target <= sum(sizes)`` — residual ~99.3%.
CRITICAL_THRESHOLDS: dict[str, float] = {
    "backend/services/learning_calibration.py": 99.0,
    "backend/hydrus/metadata_maps.py": 100.0,
    "backend/hydrus/transport_errors.py": 100.0,
    "backend/services/tuning_observability.py": 100.0,
}


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    failed: list[str] = []
    for mod, need in CRITICAL_THRESHOLDS.items():
        fu = int(need) if float(need).is_integer() else need
        r = subprocess.run(
            [
                sys.executable,
                "-m",
                "coverage",
                "report",
                f"--include={mod}",
                f"--fail-under={fu}",
            ],
            cwd=root,
            capture_output=True,
            text=True,
        )
        sys.stdout.write(r.stdout)
        if r.stderr:
            sys.stderr.write(r.stderr)
        if r.returncode != 0:
            failed.append(f"{mod} (need {need}% lines)")
    if failed:
        print("Critical coverage failures:", file=sys.stderr)
        for f in failed:
            print(f"  - {f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env bash
# Run Ruff with the same rule set as [tool.ruff.lint] in pyproject.toml.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if ! command -v ruff >/dev/null 2>&1; then
  if [[ -x "$ROOT/.venv/bin/ruff" ]]; then
    exec "$ROOT/.venv/bin/ruff" check backend tests "$@"
  fi
  echo "error: ruff not found — install dev extras: pip install -e '.[dev]'" >&2
  exit 1
fi
exec ruff check backend tests "$@"

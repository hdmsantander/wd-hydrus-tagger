#!/usr/bin/env bash
# Public entry name documented in README / CLAUDE.md; implementation lives in start.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/start.sh" "$@"

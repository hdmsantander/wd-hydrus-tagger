#!/usr/bin/env bash
# Quick HTTP smoke for docker-compose services (run on the host after compose up).
set -euo pipefail

TAGGER_URL="${TAGGER_URL:-http://127.0.0.1:8199}"
HYDRUS_WEB_URL="${HYDRUS_WEB_URL:-http://127.0.0.1:${HYDRUS_WEB_PORT:-8080}}"

check() {
    local name="$1"
    local url="$2"
    if curl -sf --max-time 10 "$url" >/dev/null; then
        echo "OK  $name  $url"
    else
        echo "FAIL  $name  $url" >&2
        return 1
    fi
}

check "wd-tagger" "${TAGGER_URL}/api/app/status"
check "hydrus-web" "${HYDRUS_WEB_URL}/"

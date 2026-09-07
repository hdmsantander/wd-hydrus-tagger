#!/usr/bin/env bash
# Smoke test for face tagging API (server must be running on 8199).
set -euo pipefail

BASE="${WD_TAGGER_SMOKE_URL:-http://127.0.0.1:8199}"

fail() { echo "face_smoke: $*" >&2; exit 1; }

echo "face_smoke: probing ${BASE} …"

curl -sf "${BASE}/api/app/status" >/dev/null || fail "tagger /api/app/status not reachable"

providers="$(curl -sf "${BASE}/api/face/providers")"
echo "$providers" | grep -q '"success":true' || fail "face providers bad response"
echo "$providers" | grep -q 'CPUExecutionProvider' || fail "CPUExecutionProvider missing"

status="$(curl -sf "${BASE}/api/face/status")"
echo "$status" | grep -q '"success":true' || fail "face status bad response"
echo "$status" | grep -q '"faces"' || fail "face status missing faces count"

session="$(curl -sf "${BASE}/api/face/session/status")"
echo "$session" | grep -q '"success":true' || fail "face session status bad response"
echo "$session" | grep -q '"active"' || fail "face session status missing active flag"

models="$(curl -sf "${BASE}/api/face/models")"
echo "$models" | grep -q '"success":true' || fail "face models bad response"
echo "$models" | grep -q 'buffalo_l' || fail "face models missing buffalo_l"

verify="$(curl -sf -X POST "${BASE}/api/face/models/verify")"
echo "$verify" | grep -q '"success":true' || fail "face models verify failed"

reset="$(curl -sf -X POST "${BASE}/api/face/reset")"
echo "$reset" | grep -q '"success":true' || fail "face reset failed"

# Static UI assets
curl -sf "${BASE}/" | grep -q 'panel-face' || fail "index.html missing face panel"
curl -sf "${BASE}/js/components/face.js" | grep -q 'startFaceDetectWebSocket' || fail "face.js not served"

echo "face_smoke: OK"

#!/usr/bin/env bash
set -euo pipefail
# Batch staging script for CPU-first optimization/refactor work.
# Run from repo root: /home/fabian/Storage/Privado/Hydrus/wd-hydrus-tagger
run_tests_for_batch() {
  local batch="$1"
  case "$batch" in
    1)
      .venv/bin/pytest --no-cov -q \
        tests/test_tagging_service.py \
        tests/test_tagging_batching.py \
        tests/test_tagger_ws_recovery.py
      ;;
    2)
      .venv/bin/pytest --no-cov -q \
        tests/test_predict_route.py \
        tests/test_config_route.py \
        tests/test_files_routes_more.py \
        tests/test_connection_route.py
      ;;
    3)
      .venv/bin/pytest --no-cov -q \
        tests/test_gallery_viewer_frontend.py \
        tests/test_frontend_english.py
      ;;
    4)
      .venv/bin/pytest --no-cov -q \
        tests/test_tagging_batching.py \
        tests/test_tagging_metadata_prefetch.py \
        tests/test_log_report.py \
        tests/test_analyze_tagging_log_script.py \
        tests/test_frontend_english.py \
        tests/test_gallery_viewer_frontend.py
      ;;
    5)
      .venv/bin/pytest --no-cov -q \
        tests/test_tagger_websocket.py \
        tests/test_tagger_ws_recovery.py \
        tests/test_tagger_ws_hydrus_wait.py
      ;;
    *)
      echo "Unknown batch: $batch" >&2
      exit 1
      ;;
  esac
}
stage_batch() {
  local batch="$1"
  git reset
  case "$batch" in
    1)
      git add \
        backend/services/tagging_service.py
      ;;
    2)
      git add \
        backend/routes/tagger_http.py \
        backend/routes/config_routes.py \
        backend/routes/files.py
      ;;
    3)
      git add \
        frontend/js/components/progress.js \
        frontend/js/components/gallery.js \
        frontend/js/utils/hydrus.js
      ;;
    4)
      git add \
        backend/services/tagging_shared.py \
        backend/log_parsing.py \
        backend/log_report.py \
        scripts/analyze_tagging_log.py \
        frontend/js/config_mapper.js \
        frontend/js/utils/dom.js \
        frontend/js/app.js \
        frontend/js/components/settings.js \
        tests/test_tagging_batching.py \
        tests/test_tagging_metadata_prefetch.py
      ;;
    5)
      git add \
        backend/routes/tagger_ws_transport.py \
        backend/routes/tagger_ws.py \
        frontend/js/components/tagger_progress.js \
        frontend/js/components/tagger.js
      ;;
  esac
  echo
  echo "=== Staged files for batch $batch ==="
  git diff --staged --name-only
  echo
}
commit_batch() {
  local batch="$1"
  local msg="$2"
  git commit -m "$(cat <<EOF
$msg
EOF
)"
}
# Usage:
#   ./stage_cpu_refactor_batches.sh 1
#   ./stage_cpu_refactor_batches.sh 2 --test
#   ./stage_cpu_refactor_batches.sh 3 --test --commit "your message"
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <batch:1-5> [--test] [--commit \"message\"]" >&2
  exit 1
fi
BATCH="$1"
shift
DO_TEST=0
DO_COMMIT=0
COMMIT_MSG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --test)
      DO_TEST=1
      shift
      ;;
    --commit)
      DO_COMMIT=1
      COMMIT_MSG="${2:-}"
      if [[ -z "$COMMIT_MSG" ]]; then
        echo "--commit requires a message" >&2
        exit 1
      fi
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done
stage_batch "$BATCH"
if [[ "$DO_TEST" -eq 1 ]]; then
  echo "=== Running tests for batch $BATCH ==="
  run_tests_for_batch "$BATCH"
fi
if [[ "$DO_COMMIT" -eq 1 ]]; then
  echo "=== Committing batch $BATCH ==="
  commit_batch "$BATCH" "$COMMIT_MSG"
fi
echo "Done."

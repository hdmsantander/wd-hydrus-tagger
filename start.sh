#!/usr/bin/env bash
# WD Hydrus Tagger — run the server or invoke tests from the repository root.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

usage() {
    cat <<'EOF'
WD Hydrus Tagger helper script.

Usage:
  ./start.sh              Start the app (same as: run); runs requirements check first
  ./start.sh run          Start uvicorn via run.py (foreground; logs to this terminal)
  ./start.sh check        Validate Python, dependencies, config.yaml, ONNX GPU stack (when configured), writable dirs
  ./start.sh test         Run pytest (any pytest options). Runs **check** first unless ``--skip-req-check``
  ./start.sh log-report   Summarize logs/latest.log (cache hits, metadata lines, errors); optional path, --fail-on-error
  ./start.sh tagging-report   Write Markdown tagging session table (default: logs/latest.log); optional log path, --out FILE
  ./start.sh generate-config   Interactive config.yaml wizard (Linux only: /proc + optional nvidia-smi)
  ./start.sh run-native-web Native tagger (GPU on host) + hydrus-web in Docker; stops hydrus-web on exit
  ./start.sh docker-run       Tagger only: docker compose up --build (runs check first; add -d to detach)
  ./start.sh docker-run-all   Tagger + hydrus-web: compose --profile hydrus-web up --build
  ./start.sh docker-down      Stop and remove compose containers + networks (add --profile hydrus-web for both services)
                              AMD: auto-includes docker-compose.amd.yml when /dev/kfd exists (WD_TAGGER_DOCKER_AMD=0 to skip)
  ./start.sh --generate-config Same as generate-config (must be the first argument)
  ./start.sh help         Show this help (same: usage, -h, --help as the command word)
  ./start.sh usage        Same as help

  Pass extra args to run.py (log level, etc.). You can omit the word run if the first arg is a flag:
  ./start.sh --log-level DEBUG
  ./start.sh run --log-level WARNING
  ./start.sh run --skip-req-check --log-level DEBUG   # skip pre-flight (not recommended)
  ./start.sh --log-file /tmp/wd-tagger.log

  **Native tagger + Docker hydrus-web** (recommended on AMD ROCm hosts for GPU inference):
  ./start.sh run-native-web
  ./start.sh run-native-web --log-level DEBUG
  Set hydrus_web_url to http://127.0.0.1:8080 (or HYDRUS_WEB_PORT) in config.

  **Log level** for run.py: any name accepted by Python logging (e.g. DEBUG, INFO, WARNING, ERROR, CRITICAL).

  **Tests:** all arguments are forwarded to pytest (markers, -k, -x, --no-cov, -q, etc.):
  ./start.sh test
  ./start.sh test --no-cov -q
  ./start.sh test -m core --skip-req-check
  ./start.sh test -o log_cli=true --log-cli-level=DEBUG

Environment:
  PYTHONPATH is set to the repo root.
  Uses .venv/bin/python when present, otherwise python3.
  LOG_LEVEL or WD_TAGGER_LOG_LEVEL   Default: INFO; run.py: DEBUG, INFO, WARNING, ERROR, CRITICAL, …
  WD_TAGGER_LOG_FILE                 Optional explicit log file path
  HYDRUS_WEB_PORT                    Host port for hydrus-web (default 8080; run-native-web / compose)
  WD_TAGGER_SKIP_REQ_CHECK=1         Skip requirements check (same as --skip-req-check)

Requires:
  run: pip install -r requirements.txt or pip install -e .

  The requirements check runs automatically before run, test, run-native-web, and docker-run*
  (except when using --skip-req-check / WD_TAGGER_SKIP_REQ_CHECK, or when you only pass
  run.py help flags (-h / --help) so a broken venv can still show usage). The check lists
  installed ONNX Runtime providers, validates the WD + face GPU plan when use_gpu or an
  explicit gpu_backend is set, and warns on multiple onnxruntime wheels.
EOF
}

# True if run.py will only be used for argparse help (skip preflight; works with broken deps).
_wants_runpy_help() {
    local a
    for a in "$@"; do
        case "$a" in
            -h | --help) return 0 ;;
        esac
    done
    return 1
}

die() {
    echo "error: $*" >&2
    exit 1
}

pick_python() {
    if [[ -x "$ROOT/.venv/bin/python" ]]; then
        echo "$ROOT/.venv/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        command -v python3
    else
        die "python3 not found (create .venv or install Python 3)"
    fi
}

PY="$(pick_python)"

[[ -f "$ROOT/run.py" ]] || die "run.py not found (wrong directory?)"

run_requirements_check() {
    if [[ ! -f "$ROOT/scripts/check_requirements.py" ]]; then
        die "scripts/check_requirements.py missing (incomplete checkout?)"
    fi
    echo "Running requirements check: $PY scripts/check_requirements.py" >&2
    if ! "$PY" "$ROOT/scripts/check_requirements.py"; then
        die "requirements check failed — fix errors above, or use pip install -r requirements.txt / pip install -e ."
    fi
}

maybe_run_requirements_check() {
    if [[ "${SKIP_REQ_CHECK:-0}" == "1" ]]; then
        echo "warning: skipping requirements check (WD_TAGGER_SKIP_REQ_CHECK or --skip-req-check)" >&2
        return 0
    fi
    run_requirements_check
}

run_server() {
    export LOG_LEVEL="${LOG_LEVEL:-${WD_TAGGER_LOG_LEVEL:-INFO}}"
    export WD_TAGGER_LOG_LEVEL="${WD_TAGGER_LOG_LEVEL:-$LOG_LEVEL}"
    if [[ "${SKIP_REQ_CHECK:-0}" == "1" ]]; then
        echo "warning: skipping requirements check (WD_TAGGER_SKIP_REQ_CHECK or --skip-req-check)" >&2
    elif _wants_runpy_help "$@"; then
        echo "Skipping requirements check (run.py --help)." >&2
    else
        run_requirements_check
    fi
    local log_hint="${WD_TAGGER_LOG_FILE:-$ROOT/logs/latest.log}"
    echo "Starting WD Hydrus Tagger: $PY run.py $* (env LOG_LEVEL=$LOG_LEVEL)" >&2
    echo "PYTHONPATH=$PYTHONPATH" >&2
    echo "Log file (default): $log_hint — override with WD_TAGGER_LOG_FILE or run.py --log-file …" >&2
    if ! "$PY" "$ROOT/run.py" "$@"; then
        local ec=$?
        echo "error: server exited unexpectedly (exit $ec)." >&2
        echo "error: Check $log_hint for ERROR lines (run.py also prints log_file= on startup)." >&2
        echo "error: ./start.sh log-report  # summarize recent log" >&2
        exit "$ec"
    fi
}

run_log_report() {
    if [[ ! -f "$ROOT/scripts/summarize_latest_log.py" ]]; then
        die "scripts/summarize_latest_log.py missing"
    fi
    "$PY" "$ROOT/scripts/summarize_latest_log.py" "${RUN_ARGS[@]}"
}

run_tagging_report() {
    if [[ ! -f "$ROOT/scripts/analyze_tagging_log.py" ]]; then
        die "scripts/analyze_tagging_log.py missing"
    fi
    "$PY" "$ROOT/scripts/analyze_tagging_log.py" "${RUN_ARGS[@]}"
}

pick_pytest() {
    if [[ -x "$ROOT/.venv/bin/pytest" ]]; then
        echo "$ROOT/.venv/bin/pytest"
    elif "$PY" -m pytest --version >/dev/null 2>&1; then
        echo "$PY -m pytest"
    else
        die "pytest not found. Install: pip install -e '.[dev]' (or: pip install pytest pytest-asyncio pytest-cov)"
    fi
}

run_tests() {
    local pytest_cmd
    pytest_cmd="$(pick_pytest)"
    # shellcheck disable=SC2086
    if ! $pytest_cmd "$@"; then
        die "tests failed (pytest exit non-zero)"
    fi
    echo "tests: OK" >&2
}

require_docker() {
    if ! command -v docker >/dev/null 2>&1; then
        die "Docker is not installed or not in PATH — install Docker first"
    fi
    if ! docker compose version >/dev/null 2>&1; then
        die "Docker Compose plugin is not installed"
    fi
}

# Compose files: add AMD overlay when the host has ROCm (/dev/kfd), unless WD_TAGGER_DOCKER_AMD=0.
docker_compose() {
    local files=(-f "$ROOT/docker-compose.yml")
    if [[ "${WD_TAGGER_DOCKER_AMD:-}" != "0" ]]; then
        if [[ "${WD_TAGGER_DOCKER_AMD:-}" == "1" || -e /dev/kfd ]]; then
            if [[ -f "$ROOT/docker-compose.amd.yml" ]]; then
                files+=(-f "$ROOT/docker-compose.amd.yml")
                if command -v getent >/dev/null 2>&1; then
                    if [[ -z "${VIDEO_GID:-}" ]]; then
                        _vgid="$(getent group video | cut -d: -f3 || true)"
                        if [[ -n "$_vgid" ]]; then
                            export VIDEO_GID="$_vgid"
                        fi
                    fi
                    if [[ -z "${RENDER_GID:-}" ]]; then
                        _rgid="$(getent group render | cut -d: -f3 || true)"
                        if [[ -n "$_rgid" ]]; then
                            export RENDER_GID="$_rgid"
                        fi
                    fi
                fi
                echo "AMD GPU: using docker-compose.amd.yml (ROCm devices + onnxruntime-migraphx)." >&2
            fi
        fi
    fi
    docker compose "${files[@]}" "$@"
}

_NATIVE_WEB_STARTED=0

cleanup_native_web() {
    if [[ "$_NATIVE_WEB_STARTED" == "1" ]]; then
        echo "Stopping hydrus-web container..." >&2
        docker_compose --profile hydrus-web stop hydrus-web 2>/dev/null || true
        _NATIVE_WEB_STARTED=0
    fi
}

run_native_with_web() {
    require_docker
    maybe_run_requirements_check
    echo "Starting hydrus-web in Docker (WD tagger runs natively on this host)..." >&2
    if ! docker_compose --profile hydrus-web up -d hydrus-web; then
        die "failed to start hydrus-web container"
    fi
    _NATIVE_WEB_STARTED=1
    trap cleanup_native_web EXIT INT TERM
    local web_port="${HYDRUS_WEB_PORT:-8080}"
    echo "hydrus-web: http://127.0.0.1:${web_port}/ — set hydrus_web_url in config if needed" >&2
    echo "tagger (native): see config host/port (default http://127.0.0.1:8199/)" >&2
    SKIP_REQ_CHECK=1 run_server "${RUN_ARGS[@]}"
}

run_docker_up() {
    local profile=("$@")
    require_docker
    maybe_run_requirements_check
    if ! docker_compose "${profile[@]}" up --build "${RUN_ARGS[@]}"; then
        die "docker compose exited with errors or it timed out"
    fi
}

# --- Parse command and trailing args; strip --skip-req-check from run.py argv only ---
SKIP_REQ_CHECK=0
case "${WD_TAGGER_SKIP_REQ_CHECK:-}" in
    1|true|yes|TRUE|YES) SKIP_REQ_CHECK=1 ;;
esac

if [[ $# -eq 0 ]]; then
    cmd="run"
    ARGS=()
elif [[ "$1" == "--generate-config" ]]; then
    cmd="generate-config"
    shift
    ARGS=("$@")
elif [[ "$1" == -* ]]; then
    cmd="run"
    ARGS=("$@")
else
    cmd="$1"
    shift
    ARGS=("$@")
fi

RUN_ARGS=()
for a in "${ARGS[@]}"; do
    if [[ "$a" == "--skip-req-check" ]]; then
        SKIP_REQ_CHECK=1
        continue
    fi
    RUN_ARGS+=("$a")
done

case "$cmd" in
    run|start|server)
        run_server "${RUN_ARGS[@]}"
        ;;
    check|doctor|verify)
        run_requirements_check
        echo "check: OK" >&2
        ;;
    test|tests)
        maybe_run_requirements_check
        run_tests "${RUN_ARGS[@]}"
        ;;
    log-report|logs|log-summary)
        run_log_report
        ;;
    tagging-report|tagging-analysis)
        run_tagging_report
        ;;
    generate-config|generate_config)
        if [[ ! -f "$ROOT/scripts/generate_config.py" ]]; then
            die "scripts/generate_config.py missing"
        fi
        exec "$PY" "$ROOT/scripts/generate_config.py" "${RUN_ARGS[@]}"
        ;;
    run-native-web|native-web|run-native-with-web)
        run_native_with_web
        ;;
    docker|docker-run)
        echo "Updating/Building and starting Docker container (wd-tagger only)..." >&2
        run_docker_up
        ;;
    docker-run-all|docker-all)
        echo "Updating/Building and starting wd-tagger + hydrus-web (--profile hydrus-web)..." >&2
        run_docker_up --profile hydrus-web
        ;;
    docker-down|docker-stop)
        require_docker
        echo "Stopping Docker compose stack (containers + networks)..." >&2
        docker_compose --profile hydrus-web down --remove-orphans "${RUN_ARGS[@]}"
        ;;
    help | usage | -h | --help)
        usage
        exit 0
        ;;
    *)
        die "unknown command '$cmd' (try: help)"
        ;;
esac

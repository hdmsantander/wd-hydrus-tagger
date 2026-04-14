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
  ./wd-hydrus-tagger.sh              Start the app (same as: run); runs requirements check first
  ./wd-hydrus-tagger.sh run          Start uvicorn via run.py (foreground; logs to this terminal)
  ./wd-hydrus-tagger.sh check        Validate Python, dependencies, config.yaml, and writable dirs (exit non-zero on failure)
  ./wd-hydrus-tagger.sh test         Run pytest (any pytest options). **Default:** complete suite (all tests, same as plain ``pytest``). **``test -m full``** also runs **check** first (deps + config); use ``--skip-req-check`` to skip.
  ./wd-hydrus-tagger.sh log-report   Summarize logs/latest.log (cache hits, metadata lines, errors); optional path, --fail-on-error
  ./wd-hydrus-tagger.sh tagging-report   Write Markdown tagging session table (default: logs/latest.log); optional log path, --out FILE
  ./wd-hydrus-tagger.sh generate-config   Interactive config.yaml wizard (Linux only: /proc + optional nvidia-smi)
  ./wd-hydrus-tagger.sh --generate-config Same as generate-config (must be the first argument)
  ./wd-hydrus-tagger.sh help         Show this help (same: usage, -h, --help as the command word)
  ./wd-hydrus-tagger.sh usage        Same as help

  Pass extra args to run.py (log level, etc.). You can omit the word run if the first arg is a flag:
  ./wd-hydrus-tagger.sh --log-level DEBUG
  ./wd-hydrus-tagger.sh run --log-level WARNING
  ./wd-hydrus-tagger.sh run --skip-req-check --log-level DEBUG   # skip pre-flight (not recommended)
  ./wd-hydrus-tagger.sh --log-file /tmp/wd-tagger.log

  **Log level** for run.py: any name accepted by Python logging (e.g. DEBUG, INFO, WARNING, ERROR, CRITICAL).

  **Tests:** all arguments are forwarded to pytest (markers, -k, -x, --no-cov, -q, etc.):
  ./wd-hydrus-tagger.sh test
  ./wd-hydrus-tagger.sh test --no-cov -q
  ./wd-hydrus-tagger.sh test -m full
  ./wd-hydrus-tagger.sh test -m full --skip-req-check
  ./wd-hydrus-tagger.sh test -m core --no-cov
  ./wd-hydrus-tagger.sh test -o log_cli=true --log-cli-level=DEBUG

Environment:
  PYTHONPATH is set to the repo root.
  Uses .venv/bin/python when present, otherwise python3.
  LOG_LEVEL or WD_TAGGER_LOG_LEVEL   Default: INFO; run.py: DEBUG, INFO, WARNING, ERROR, CRITICAL, …
  WD_TAGGER_LOG_FILE                 Optional explicit log file path
  WD_TAGGER_SKIP_REQ_CHECK=1         Skip requirements check before **run** or before **test -m full** (same as --skip-req-check)

Requires:
  run: pip install -r requirements.txt or pip install -e .

  The requirements check runs automatically before starting the server (run / default),
  except when using --skip-req-check / WD_TAGGER_SKIP_REQ_CHECK, or when you only pass
  run.py help flags (-h / --help) so a broken venv can still show usage.

  test: pytest + pytest-cov (pip install -e ".[dev]"). Plain **test** runs the **complete** suite (no ``-m``); **test -m full** is the same selection plus a prior **check**.
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

# True if pytest argv selects marker ``full`` (``-m full`` or ``-mfull``).
_test_args_include_m_full() {
    local prev=""
    for a in "$@"; do
        if [[ "$prev" == "-m" && "$a" == "full" ]]; then
            return 0
        fi
        case "$a" in
            -mfull) return 0 ;;
        esac
        prev="$a"
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
        echo "error: ./wd-hydrus-tagger.sh log-report  # summarize recent log" >&2
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

run_tests() {
    if [[ -x "$ROOT/.venv/bin/pytest" ]]; then
        if ! "$ROOT/.venv/bin/pytest" "$@"; then
            die "tests failed (pytest exit non-zero)"
        fi
    elif "$PY" -m pytest --version >/dev/null 2>&1; then
        if ! "$PY" -m pytest "$@"; then
            die "tests failed (pytest exit non-zero)"
        fi
    else
        die "pytest not found. Install: pip install -e '.[dev]' (or: pip install pytest pytest-asyncio pytest-cov)"
    fi
    echo "tests: OK" >&2
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
        if _test_args_include_m_full "${RUN_ARGS[@]}"; then
            if [[ "${SKIP_REQ_CHECK:-0}" == "1" ]]; then
                echo "warning: skipping requirements check before test -m full (WD_TAGGER_SKIP_REQ_CHECK or --skip-req-check)" >&2
            else
                run_requirements_check
            fi
        fi
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
    help | usage | -h | --help)
        usage
        exit 0
        ;;
    *)
        die "unknown command '$cmd' (try: help)"
        ;;
esac

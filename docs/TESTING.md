# Testing

The suite uses **pytest** with **pytest-asyncio** and **pytest-cov** (see `pyproject.toml`). The **default** invocation runs **all** tests with coverage — same selection as **`pytest -m full`** (see below).

**Shell entrypoints:** [`wd-hydrus-tagger.sh`](../wd-hydrus-tagger.sh) at the repo root delegates to [`start.sh`](../start.sh) (single implementation). Use either name; docs and CI refer to `wd-hydrus-tagger.sh`.

**Warnings baseline:** see [WARNINGS.md](WARNINGS.md) for how to capture pytest warnings and tiered backlog expectations.

## Markers

| Marker | Scope | When to run first |
|--------|--------|-------------------|
| **`full`** | **Every** test (declared on each module alongside `core` / `ws` / `ui`) | **`pytest -m full`** — complete suite **including** `slow` tests, **with** the default coverage gate (`fail_under`) |
| **`core`** | Unit tests, HTTP routes, services, scripts (not the `ws` / `ui`-only modules) | After edits outside `backend/routes/tagger_ws.py` and `frontend/` |
| **`ws`** | WebSocket tagging tests (`test_tagger_websocket.py`, `test_tagger_ws_recovery.py`, `test_tagger_ws_validation.py`, …) | After changing WebSocket tagging or recovery |
| **`ui`** | `test_frontend_english.py`, static frontend checks | After changing `frontend/` copy or layout |
| **`slow`** | Subset of **`ws`** / **`ui`** (multi-batch WS, learning calibration, full CJK scan) | Optional; use **`pytest -m "not slow"`** to skip these while keeping coverage high |

Markers are **composable**: slow WebSocket tests are **`ws`**, **`full`**, and **`slow`**.

## Commands

```bash
# Complete suite + coverage (default addopts) — equivalent to plain pytest
pytest
pytest -m full
./wd-hydrus-tagger.sh test                    # complete suite; no preflight
./wd-hydrus-tagger.sh test -m full            # same tests + runs ``check`` first (deps/config)
./wd-hydrus-tagger.sh test -m full --skip-req-check   # skip that preflight

# Targeted runs — use --no-cov (or --cov-fail-under=0) so partial selection does not trip fail_under
pytest -m core --no-cov
pytest -m ws --no-cov
pytest -m ui --no-cov
pytest -m slow --no-cov
./wd-hydrus-tagger.sh test -m core --no-cov -q

# All tests except slow (faster; coverage may still meet fail_under depending on selection)
pytest -m "not slow"

# Combine markers (examples)
pytest -m "ws and not slow" --no-cov
pytest -m "core or ui" --no-cov

# Docker / hydrus-web static regression (no docker daemon)
pytest tests/test_docker_artifacts.py tests/test_hydrus_web_frontend.py --no-cov -q
```

**Coverage:** `pyproject.toml` sets **`fail_under=82`** (total line+branch coverage for `backend/`). Narrow markers alone often collect fewer lines and can fall below the gate; use **`--no-cov`** or **`--cov-fail-under=0`** for quick partial runs.

**Lint:** with dev extras, run **`./scripts/run_ruff.sh`** or **`ruff check backend tests`** (see `[tool.ruff]` in `pyproject.toml`). The default rule set is **syntax and undefined-name checks** (`E9`, `F821`–`F823`); broaden to full **`E`/`F`** incrementally as cleanups land (`ruff check backend tests --select E,F`).

## Layout

- **`tests/conftest.py`** — Autouse isolation: config singleton, `TaggingService` reset, perf totals reset.
- **`tests/test_docker_artifacts.py`** — Compose/Dockerfile/dockerignore smoke strings (no daemon).
- **`tests/test_hydrus_web_frontend.py`** — Static checks for optional hydrus-web UI wiring.
- **`scripts/check_critical_coverage.py`** — Optional stricter line coverage on selected modules (run after `coverage run`).

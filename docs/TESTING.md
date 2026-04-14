# Testing

The suite uses **pytest** with **pytest-asyncio** and **pytest-cov** (see `pyproject.toml`). The **default** invocation runs **all** tests with coverage — same selection as **`pytest -m full`** (see below).

## Markers

| Marker | Scope | When to run first |
|--------|--------|-------------------|
| **`full`** | **Every** test (declared on each module alongside `core` / `ws` / `ui`) | **`pytest -m full`** — complete suite **including** `slow` tests, **with** the default coverage gate (`fail_under`) |
| **`core`** | Unit tests, HTTP routes, services, scripts (not the `ws` / `ui`-only modules) | After edits outside `backend/routes/tagger_ws.py` and `frontend/` |
| **`ws`** | `test_tagger_websocket.py`, `test_tagger_ws_recovery.py` | After changing WebSocket tagging or recovery |
| **`ui`** | `test_frontend_english.py` | After changing `frontend/` copy or layout |
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

# All tests except slow (~227 tests; coverage usually still ≥ fail_under)
pytest -m "not slow"

# Combine markers (examples)
pytest -m "ws and not slow" --no-cov
pytest -m "core or ui" --no-cov
```

**Coverage:** Default **`addopts`** run **pytest-cov** with **`fail_under=70`**. **`pytest`** and **`pytest -m full`** both run **237** tests and satisfy the gate (~76%). Narrow markers alone often drop below 70%; use **`--no-cov`** for quick runs.

## Layout

- **`tests/conftest.py`** — Autouse isolation: config singleton, `TaggingService` reset, perf totals reset.
- **`scripts/check_critical_coverage.py`** — Optional stricter line coverage on selected modules (run after `coverage run`).

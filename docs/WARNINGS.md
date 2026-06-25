# Pytest warnings baseline and tiers

## Capturing a fresh baseline

From the repository root (with dev extras installed):

```bash
.venv/bin/pytest -m full -q --tb=no -W default 2>&1 | tee reports/pytest-warnings-baseline.log
```

Group the `warnings summary` block by **message** (not only by test file). Prefer fixing sources over blanket `filterwarnings`.

## Baseline (2026-04 — after ORJSONResponse removal)

| Category | Count (before → after) | Source | Action |
|----------|------------------------|--------|--------|
| `FastAPIDeprecationWarning` (`ORJSONResponse`) | 36 → **0** | `default_response_class=ORJSONResponse` in [backend/app.py](../backend/app.py) | **Fixed:** rely on FastAPI default JSON (Pydantic serialization); `orjson` remains an optional transitive/runtime speedup only if used elsewhere. |

## Severity tiers (backlog template)

| Tier | Meaning | Typical examples | Priority |
|------|---------|------------------|----------|
| **P0** | Correctness, leaks, teardown | `ResourceWarning` (unclosed socket), `PytestUnraisableExceptionWarning` | Fix immediately; audit httpx clients / WS tests. |
| **P1** | Future breakages | `DeprecationWarning` (stdlib, Pydantic, FastAPI) | Fix at source or bump deps in a dedicated PR. |
| **P2** | Hygiene / intentional noise | `UserWarning` from validated code paths (e.g. logging level tests) | Document or narrow filters with a comment. |
| **P3** | Test harness / plugins | pytest-asyncio scope hints, collection noise | Tune `pyproject.toml` / fixtures. |

**Policy:** Do not add a global `ignore::DeprecationWarning`. If a third-party warning is unavoidable short-term, use a **module-scoped** `warnings.filterwarnings` in `[tool.pytest.ini_options]` with a link to an upstream issue.

## CI suggestion

Optionally fail on new warning families:

```bash
pytest -m full -W error::ResourceWarning
```

Export test results (and keep stderr with warnings) for CI artifacts:

```bash
.venv/bin/pytest -m full -q --tb=no -W default \
  --junitxml=reports/pytest-junit.xml 2>&1 | tee reports/pytest-ci.log
```

Warnings still appear in the log text; parse the `warnings summary` section or re-run with `-W error::DeprecationWarning` once the baseline is clean.

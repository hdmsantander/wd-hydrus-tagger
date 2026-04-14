# Python dependencies

This document describes **direct** dependencies, **how the app uses them**, and **how to maintain** versions. The canonical list is **`[project.dependencies]`** in `pyproject.toml` (mirrored in `requirements.txt` for `pip install -r` workflows).

_Last reviewed: 2026-03 — PyPI “latest” matched the versions below on a clean `pip install -e ".[dev]"`; re-check periodically with the commands in [Maintenance](#maintenance)._

## Runtime (`pip install -e .`)

| Package | Role in this project |
|---------|----------------------|
| **fastapi** | HTTP API, WebSocket tagging route, `lifespan` for startup/shutdown (`backend/app.py`). Starlette is pulled in as FastAPI’s ASGI layer (do not pin Starlette separately). |
| **uvicorn[standard]** | ASGI server (`run.py`, `backend.app:main`). **`[standard]`** pulls **httptools**, **uvloop** (where applicable), **websockets**, etc. |
| **pydantic** | `AppConfig` and Hydrus/tagger response models (`backend/config.py`, `backend/hydrus/models.py`, …). Uses **Pydantic v2** (`model_validate`, `field_validator`). |
| **pyyaml** | Load `config.yaml` (stdlib-style usage; safe load for config). |
| **httpx** | Async Hydrus Client API: one shared **`AsyncClient`** per `(api_url, access_key)` with pooling and long timeouts (`backend/hydrus/client.py`). |
| **onnxruntime** | WD ONNX inference (`backend/tagger/engine.py`). Use **`onnxruntime-gpu`** instead via **`[gpu]`** extra when using CUDA (CPU and GPU wheels must not be installed together). |
| **Pillow** | Decode/resize images before ONNX (`backend/tagger/preprocess.py`). |
| **numpy** | Batch tensors for ONNX I/O. NumPy **1.26+** and **2.x** are supported by current ONNX Runtime builds; floors stay flexible for older distros. |
| **huggingface-hub** | Download / cache models under `models_dir` (`backend/services/model_manager.py`). |
| **websockets** | Declared explicitly so WebSocket protocol handling stays within a tested range; **uvicorn** / **Starlette** also depend on **websockets** transitively. |

### Optional extras

- **`[gpu]`** — `onnxruntime-gpu` (replace CPU `onnxruntime`; do not install both).
- **`[perf]`** (Linux) — **`uvloop`** for a faster event loop (`backend/runtime_linux.py`, `run.py` / `backend.app:main`).

### Dev (`pip install -e ".[dev]"`)

**pytest**, **pytest-asyncio**, **pytest-cov** — test runner and coverage (`pyproject.toml` `[tool.pytest.ini_options]`: `asyncio_mode = auto`, `asyncio_default_fixture_loop_scope = function`). **httpx** is duplicated in `[dev]` only so tools that resolve extras see a consistent floor (**`>=0.28`**, aligned with the main dependency).

## Implementation notes (best-practice alignment)

- **FastAPI / Starlette:** Routers are registered **before** the `/` static mount so `/api/*` wins. **`CORSMiddleware`** uses permissive defaults suitable for a local/LAN tool; tighten if you expose the UI to untrusted networks.
- **Lifespan:** Hydrus HTTP clients are closed on shutdown (`aclose_all_hydrus_clients`), and perf totals are logged (`backend/perf_metrics.py`).
- **httpx:** A single **`AsyncClient`** per pool key is reused (keep-alive, bounded connection limits). **`PATCH /api/config`** calls **`invalidate_hydrus_client_pool()`** when Hydrus URL or API key changes so old connections are not reused.
- **ONNX Runtime:** **`SessionOptions`** use full graph optimizations, **`ORT_SEQUENTIAL`** execution with **`inter_op_num_threads=1`** for typical single-stream CPU inference, and explicit intra-op threads from config (`backend/tagger/engine.py`).
- **WebSocket:** Tagging session runs in **`progress_ws`**; CPU-bound ONNX runs via **`asyncio.to_thread`** in the tagging service (see `backend/services/tagging_service.py`).

## Maintenance

**Upgrade installed packages** (virtualenv active):

```bash
pip install -U -e ".[dev]"
```

**See what is outdated:**

```bash
pip list --outdated
```

**Inspect a single package’s published versions:**

```bash
pip index versions fastapi
```

**Optional — vulnerability scan** (install [pip-audit](https://pypi.org/project/pip-audit/) separately):

```bash
pip install pip-audit
pip-audit
```

After upgrading **FastAPI**, **Starlette**, **uvicorn**, or **httpx**, run the full test suite (`pytest` or `./wd-hydrus-tagger.sh test`). ONNX Runtime upgrades may change performance characteristics or provider behavior; re-verify GPU installs if you use **`[gpu]`**.

## Related docs

- [TESTING.md](TESTING.md) — pytest markers (`core`, `ws`, `ui`, `slow`) and commands.
- [PERFORMANCE_AND_TUNING.md](PERFORMANCE_AND_TUNING.md) — throughput, metadata chunking, ORT profiling.
- [UPGRADE.md](UPGRADE.md) — feature and API upgrade notes.

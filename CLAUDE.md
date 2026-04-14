# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

Prefer a venv at the repo root so the shell helper picks it up: **`python3 -m venv .venv`** then **`source .venv/bin/activate`** (Windows: **`.venv\Scripts\activate`**). **`./wd-hydrus-tagger.sh`** uses **`.venv/bin/python`** when present.

```bash
# Install dependencies (CPU)
pip install -e .

# Install with GPU support (CUDA)
pip install -e ".[gpu]"

# Run the server (default bind: 0.0.0.0:8199 — use http://127.0.0.1:8199 on this PC or http://<LAN-IP>:8199 elsewhere)
python run.py

# Or helper script (runs scripts/check_requirements.py before run/start unless skipped)
./wd-hydrus-tagger.sh check   # validate deps + config only
./wd-hydrus-tagger.sh run     # check first; skip with --skip-req-check or WD_TAGGER_SKIP_REQ_CHECK=1
# No check for: help | usage | -h | --help (shell), run.py -h/--help, or --generate-config as first arg
```

Configuration lives in `config.yaml` (copy from `config.example.yaml` on first setup). The app requires a running Hydrus Network instance with API access enabled. **`models_dir`**: use `./models` (repo-relative); temp/pytest paths are coerced to `<repo>/models` unless `WD_TAGGER_ALLOW_TMP_MODELS_DIR=1` (tests). Defaults: **`wd_skip_inference_if_marker_present`** and **`wd_append_model_marker_tag`** are **true** (skip ONNX when `wd14:` marker present; append marker after run).

## Architecture

**FastAPI backend + vanilla JavaScript frontend, no build step.**

The backend runs ONNX-based image classification models (WD14 Tagger v3) and proxies all communication with the Hydrus Network API. The frontend is a single-page app served as static files.

### Backend (`backend/`)

- **`app.py`** — FastAPI app factory, CORS middleware, route mounting, static file serving; lifespan calls **`perf_metrics.mark_process_start`** after ready and **`log_process_shutdown`** on teardown
- **`perf_metrics.py`** — stdlib-only perf lines at boundaries: WebSocket tagging session end, HTTP predict/apply, process shutdown (uptime + cumulative totals + optional peak RSS)
- **`config.py`** — YAML config loader with Pydantic validation; config is cached globally
- **`dependencies.py`** — Dependency injection for singleton `HydrusClient`

**Routes** (`backend/routes/`) — routers mounted under `/api`:
- `connection.py` → `/api/connection/*` — Hydrus API credential verification, service listing
- `files.py` → `/api/files/*` — search, metadata, thumbnail/file proxy
- `tagger.py` → `/api/tagger/*` — model management, inference, tag application, WebSocket progress, `GET /api/tagger/session/status` for multi-tab read-only progress
- `config_routes.py` → `/api/config/*` — runtime config get/patch (thresholds, prefixes, `default_model`, `target_tag_service`, WD marker flags, `apply_tags_http_batch_size`, `allow_ui_shutdown`, grace seconds, etc.)
- `app_control.py` → `/api/app/*` — status metrics, graceful UI shutdown (flush/cancel tagging, unload ONNX, exit)

**Services** (`backend/services/`):
- `tagging_service.py` — orchestrates batch tagging: fetch metadata → download images → infer → format tags. Singleton instance.
- `model_manager.py` — downloads models from HuggingFace Hub, manages local cache in `models/` directory

**Tagger engine** (`backend/tagger/`):
- `engine.py` — ONNX session management with CUDA→CPU provider fallback, sigmoid inference
- `preprocess.py` — image normalization pipeline: RGB→BGR, pad to square, resize to 448×448
- `labels.py` — CSV label parser mapping tag names to categories (general/character/rating)

**Hydrus client** (`backend/hydrus/`):
- `client.py` — async httpx wrapper for Hydrus Network API
- `models.py` — Pydantic models for Hydrus responses

### Frontend (`frontend/`)

Vanilla JS with ES modules, no bundler. State management via simple pub/sub pattern in `js/state.js`.

- `js/app.js` — entry point, initializes all components
- `js/layout.js` — collapsible sidebar (desktop collapse + mobile drawer); `localStorage` key `wd_tagger_sidebar_hidden`
- `js/api.js` — fetch-based HTTP client for backend API
- `js/components/` — UI components (connection, gallery, tagger, progress, settings)
- `js/utils/dom.js` — lightweight DOM helpers (`$`, `el`, `show`, `hide`)

## Key Patterns

- **All backend I/O is async.** CPU-bound ONNX inference runs via `asyncio.to_thread`.
- **WebSocket** at `/api/tagger/ws/progress` streams real-time progress during batch tagging; user **Stop** sends `cancel` → server emits `stopping` then `stopped` after winding down (logs `user_cancel`, `winding_down`, `user_stop_complete`).
- **Singleton services** — `HydrusClient` and `TaggingService` are created once and reused via FastAPI dependency injection.
- **Config masking** — API responses mask sensitive fields (API keys) before sending to the frontend.
- **Model providers** — ONNX runtime tries CUDA first, falls back to CPU. Controlled by `use_gpu` config flag.

## Data Flow

1. User connects to Hydrus → credentials stored in config.yaml
2. Search files by tags → gallery displays paginated thumbnails (metadata fetched in chunks of `hydrus_metadata_chunk_size`, default 512)
3. Select images → load ONNX model (downloaded from HuggingFace if needed)
4. Batch inference → results filtered by thresholds (general: 0.35, character: 0.85); files with the model marker in Hydrus `storage_tags` skip fetch+ONNX when enabled
5. Tags formatted with configurable prefixes → user edits in UI → applied to Hydrus

## Defaults & hard-coded I/O (summary)

- Config defaults: `hydrus_download_parallel` **8** (1–32), `hydrus_metadata_chunk_size` **512** (32–2048), `tagging_skip_tail_batch_size` **512** (32–2048), `batch_size` **8**. See `backend/config.py` and README “Hard-coded limits & regression tests”.
- Hydrus client: `httpx` timeout 120 s / connect 15 s; pool limits 128 keep-alive / 192 max connections (`backend/hydrus/client.py`).
- Frontend: Tagger batches rapid progress counter updates to one paint per frame (`frontend/js/components/progress.js`).
- Web bind: default `host` `0.0.0.0` in `config.yaml` for LAN access; `127.0.0.1` for localhost-only. Startup prints example URLs (`backend/listen_hints.py`, `run.py` / `backend.app:main`).
- Log tracing: `./wd-hydrus-tagger.sh log-report` → `backend/log_report.py` (cache hit/miss counts, `files metadata_hydrus`, errors).
- **UI offline:** `frontend/js/server_offline.js` — full-page end state after `POST /api/app/shutdown` success; `GET /api/app/status` polling when the process disappears (CLI kill). `api.js` notifies on fetch network errors (debounced).

## Tests

Install dev extras (`pip install -e ".[dev]"`), then run **`pytest`** or **`pytest -m full`** from the repo root — **complete** suite with coverage. **`./wd-hydrus-tagger.sh test`** is the same. Markers: **`core`**, **`ws`**, **`ui`**, **`slow`**, **`full`** (every test module; **`-m full`** = all tests including slow) — see **`docs/TESTING.md`**. Use `pytest --no-cov` for a faster run without coverage; use `pytest -m core` (etc.) for targeted feedback after local edits.

## Linux performance (optional)

- **`pip install -e ".[perf]"`** pulls **uvloop**; `run.py` / `backend.app:main` pass `loop="uvloop"` to Uvicorn on Linux when import succeeds.
- ONNX CPU path: **ORT_SEQUENTIAL** + contiguous batch tensors in `backend/tagger/engine.py`.

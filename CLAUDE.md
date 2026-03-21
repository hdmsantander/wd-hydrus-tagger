# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

```bash
# Install dependencies (CPU)
pip install -e .

# Install with GPU support (CUDA)
pip install -e ".[gpu]"

# Run the server (default: http://127.0.0.1:8199)
python run.py
```

Configuration lives in `config.yaml` (copy from `config.example.yaml` on first setup). The app requires a running Hydrus Network instance with API access enabled.

## Architecture

**FastAPI backend + vanilla JavaScript frontend, no build step.**

The backend runs ONNX-based image classification models (WD14 Tagger v3) and proxies all communication with the Hydrus Network API. The frontend is a single-page app served as static files.

### Backend (`backend/`)

- **`app.py`** — FastAPI app factory, CORS middleware, route mounting, static file serving
- **`config.py`** — YAML config loader with Pydantic validation; config is cached globally
- **`dependencies.py`** — Dependency injection for singleton `HydrusClient`

**Routes** (`backend/routes/`) — four routers mounted under `/api`:
- `connection.py` → `/api/connection/*` — Hydrus API credential verification, service listing
- `files.py` → `/api/files/*` — search, metadata, thumbnail/file proxy
- `tagger.py` → `/api/tagger/*` — model management, inference, tag application, WebSocket progress
- `config_routes.py` → `/api/config/*` — runtime config get/patch

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
- `js/api.js` — fetch-based HTTP client for backend API
- `js/components/` — UI components (connection, gallery, tagger, progress, settings)
- `js/utils/dom.js` — lightweight DOM helpers (`$`, `el`, `show`, `hide`)

## Key Patterns

- **All backend I/O is async.** CPU-bound ONNX inference runs via `asyncio.to_thread`.
- **WebSocket** at `/api/tagger/ws/progress` streams real-time progress during batch tagging.
- **Singleton services** — `HydrusClient` and `TaggingService` are created once and reused via FastAPI dependency injection.
- **Config masking** — API responses mask sensitive fields (API keys) before sending to the frontend.
- **Model providers** — ONNX runtime tries CUDA first, falls back to CPU. Controlled by `use_gpu` config flag.

## Data Flow

1. User connects to Hydrus → credentials stored in config.yaml
2. Search files by tags → gallery displays paginated thumbnails
3. Select images → load ONNX model (downloaded from HuggingFace if needed)
4. Batch inference → results filtered by thresholds (general: 0.35, character: 0.85)
5. Tags formatted with configurable prefixes → user edits in UI → applied to Hydrus

[Traditional Chinese](README.zh-TW.md)

# WD Tagger for Hydrus

A web tool that automatically generates tags for images in Hydrus Network using WD14 Tagger v3.

The **web UI is English by default**. Traditional Chinese documentation is in [README.zh-TW.md](README.zh-TW.md). Technical upgrade notes live in [docs/UPGRADE.md](docs/UPGRADE.md).

---

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Development & tests](#development--tests)
- [Python dependencies & upgrades](#python-dependencies--upgrades)
- [Testing (markers, targeted runs)](#development--tests) · [docs/TESTING.md](docs/TESTING.md)
- [Hydrus Network Setup](#hydrus-network-setup)
- [Configuration](#configuration)
- [Starting the Server](#starting-the-server)
- [Workflow](#workflow)
  - [Step 1: Connect to Hydrus](#step-1-connect-to-hydrus)
  - [Step 2: Search for Images](#step-2-search-for-images)
  - [Step 3: Select Images](#step-3-select-images)
  - [Step 4: Run Auto-Tagging](#step-4-run-auto-tagging)
  - [Step 5: Review & Edit Results](#step-5-review--edit-results)
  - [Step 6: Apply Tags to Hydrus](#step-6-apply-tags-to-hydrus)
  - [Settings Panel](#settings-panel)
  - [Model Management](#model-management)
  - [Tag Prefix Settings](#tag-prefix-settings)
  - [Performance (CPU / GPU)](#performance-cpu--gpu)
  - [ONNX Runtime session profiling](#onnx-runtime-session-profiling)
- [Available Models](#available-models)
- [Threshold Tuning Guide](#threshold-tuning-guide)
- [Configuration Reference](#configuration-reference)
- [Hard-coded limits & regression tests](#hard-coded-limits--regression-tests)
- [Performance, markers, Tag all](#performance-markers-tag-all)
- [FAQ](#faq)
- [Project Structure](#project-structure)

---

## System Requirements

| Item | Requirement |
|------|-------------|
| Python | 3.10+ |
| Hydrus Network | Any recent version (Client API must be enabled) |
| OS | Windows / Linux / macOS |
| Disk Space | ~400 MB (ViT base) – ~1.3 GB (ViT Large) per model; use a **stable** `models_dir` (e.g. `./models`) |
| RAM | 4 GB+ recommended; **Large** models: 16 GB+ for `batch_size` 8; with **32 GB** you can keep defaults (`batch_size` 8, `hydrus_download_parallel` 8, `hydrus_metadata_chunk_size` 512) and rely on ONNX staying loaded between runs when `models_dir` is stable (e.g. `./models`) |
| GPU (optional) | NVIDIA GPU with CUDA support |

---

## Installation

### 1. Install Python Dependencies

```bash
# CPU version (default)
pip install -r requirements.txt

# GPU version: onnxruntime-gpu includes CPU support
# Note: CPU and GPU versions cannot coexist — remove CPU version first
pip install -r requirements.txt --ignore-requires-python
pip uninstall -y onnxruntime
pip install onnxruntime-gpu
```

### 2. Create Configuration File

```bash
# Copy the example config
cp config.example.yaml config.yaml
```

Then edit `config.yaml` and fill in your Hydrus API Key (see "Hydrus Network Setup" below).

**Optional (Linux only):** run an interactive wizard that reads `/proc` for RAM/CPU hints and optionally detects NVIDIA via `nvidia-smi`:

```bash
./wd-hydrus-tagger.sh generate-config
# or: ./wd-hydrus-tagger.sh --generate-config
```

On other operating systems, copy `config.example.yaml` manually; the wizard exits with code 2 if `sys.platform` is not `linux`.

---

## Development & tests

**Virtual environment:** `./wd-hydrus-tagger.sh` uses **`.venv/bin/python`** when that path exists; otherwise it falls back to **`python3`**. Create the venv at the repository root and install the package in editable mode:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # runtime + pytest, pytest-cov, …
```

If **`import onnxruntime`** fails after an interrupted install, repair with **`pip install --force-reinstall onnxruntime`** (then **`pip install -e ".[dev]"`** again if needed).

```bash
# With .venv activated and ``pip install -e ".[dev]"`` done:
PYTHONPATH=. pytest       # coverage on ``backend`` by default (see ``pyproject.toml`` addopts)
PYTHONPATH=. pytest -m full   # complete suite (all tests, including ``slow``); same as plain pytest
PYTHONPATH=. pytest --no-cov -q   # faster runs without coverage
# or, from the repo root (sets PYTHONPATH):
./wd-hydrus-tagger.sh check   # deps + config + dirs only
./wd-hydrus-tagger.sh test                 # complete suite + coverage (no preflight)
./wd-hydrus-tagger.sh test -m full         # same tests; runs ``check`` first (use ``--skip-req-check`` to skip)
./wd-hydrus-tagger.sh test -m core --no-cov -q   # targeted run (add --no-cov when using -m; see docs/TESTING.md)
./wd-hydrus-tagger.sh log-report           # summarize logs/latest.log (cache + metadata + errors)
./wd-hydrus-tagger.sh log-report --fail-on-error   # exit 1 if any ERROR lines
```

**Markers (`core`, `ws`, `ui`, `slow`, `full`):** **`pytest -m full`** runs the **complete** suite (all tests including **`slow`**) with the default coverage gate — equivalent to unfiltered **`pytest`**. Use **`pytest -m "not slow"`** to skip heavier tests. Use **`pytest -m core --no-cov`** (etc.) for faster feedback; add **`--no-cov`** when filtering so the coverage **`fail_under`** gate is not applied to a partial run. See **[docs/TESTING.md](docs/TESTING.md)**.

**Faster tests and lower dev memory:** `pytest --no-cov -q` skips coverage collection (biggest win). To drop **`pytest-cov`** from the default `addopts` entirely, run `pytest --override-ini addopts=` (or `addopts=-q`). Profiling or micro-optimizing by “disabling test libraries” is usually unnecessary: dev extras are not installed in production deployments (`pip install -e .` without `[dev]`). Runtime log volume is controlled by **`--log-level`** / **`LOG_LEVEL`** / **`WD_TAGGER_LOG_LEVEL`** (e.g. **`WARNING`** in production); this does not change ONNX memory, only I/O.

### Python dependencies & upgrades

Direct runtime and dev dependencies are listed in **`pyproject.toml`** (and mirrored in **`requirements.txt`**). For **how they are used**, **optional extras** (`[gpu]`, `[perf]`, `[dev]`), **checking for updates**, and **security scanning**, see **[docs/DEPENDENCIES.md](docs/DEPENDENCIES.md)**.

Coverage is configured under **`[tool.coverage.*]`** in `pyproject.toml` (branch coverage, `fail_under` on total `backend` coverage). If you only installed `pytest` without **`pytest-cov`**, run `pytest --override-ini addopts=` or add `pytest-cov` so the default `addopts` apply. The suite includes API and WebSocket tagging tests, **`tests/test_logging_setup.py`** (per-run log files, pruning, `latest` pointer), **`tests/test_perf_metrics.py`** (perf totals + shutdown log line), **`tests/test_frontend_english.py`** (UI locale / asset scan), **`tests/test_tag_merge.py`** / **`tests/test_apply_tags_route.py`** (Hydrus deduplication and marker union for “any service”), **`tests/test_files_metadata_chunk.py`** (large `POST /api/files/metadata` lists are chunked per `hydrus_metadata_chunk_size`), **`tests/test_tagging_service.py`** (thumbnail fallback, video thumbnail-only path, ONNX batch failure handling, `models_dir` singleton refresh), **`tests/test_tagging_profile.py`** (parallel Hydrus fetch timing vs sequential bound, log structure for fetch/predict steps), **`tests/test_model_manager.py`** (disk cache detection), plus **`tests/test_tagger_websocket.py`**. See [Hard-coded limits & regression tests](#hard-coded-limits--regression-tests) for a short map of fixed backend values vs. config.

### Frontend ↔ backend API map

| UI / flow | Client call | Backend |
|-----------|-------------|---------|
| Connect, list tag services | `api.testConnection`, `api.getServices` | `POST /api/connection/test`, `GET /api/connection/services` |
| Search, metadata, thumbs, full file | `api.searchFiles`, `api.getMetadata`, thumbnail/file URLs | `POST /api/files/search`, `POST /api/files/metadata`, `GET /api/files/{id}/thumbnail`, `GET /api/files/{id}` |
| Models list / download | `api.listModels`, `api.downloadModel` | `GET /api/tagger/models`, `POST /api/tagger/models/{name}/download` |
| Load model before tagging | `api.loadModel` | `POST /api/tagger/models/{name}/load` |
| Tag selected / tag all (inference + optional Hydrus) | `api.startTaggingWebSocket` | WebSocket **`/api/tagger/ws/progress`** (first message = run payload including optional **`tag_all`**, **`performance_tuning`**; then `pause` / `resume` / `flush` / `cancel`). The server prefetches **`get_file_metadata`** once for the full queue, then reuses it for each inference batch (see [docs/PERFORMANCE_AND_TUNING.md](docs/PERFORMANCE_AND_TUNING.md)). After **`cancel`**, server sends **`stopping`** (wind-down hint + `pending_hydrus_queue`), then **`stopped`** or **`complete`**. |
| Multi-tab read-only progress | `api.getTaggingSessionStatus` | `GET /api/tagger/session/status` → `{ active, snapshot }` |
| Verify ONNX/CSV cache (optional Hub revision) | `api.verifyModels` | `POST /api/tagger/models/verify` body `{ check_remote?, model_name? }` |
| Apply all tags (results screen) | `api.applyTags` | `POST /api/tagger/apply` |
| Settings save | `api.getConfig`, `api.updateConfig` | `GET` / `PATCH /api/config` (includes **`ort_enable_profiling`**, **`ort_profile_dir`**, learning caps, etc.) |
| Server status / stop | `api.getAppStatus`, `api.shutdownApp` | `GET /api/app/status`, `POST /api/app/shutdown` (after stop, SPA shows full-page offline view; tab also probes this URL if the process dies elsewhere) |
| *(unused by SPA)* | `api.predict` | `POST /api/tagger/predict` (batch tagging without WebSocket; scripts / tools) |

---

## Hydrus Network Setup

Before using this tool, you need to enable the Client API in Hydrus Network and obtain an API Key.

### Enable Client API

1. Open Hydrus Client
2. Go to **services → manage services**
3. Ensure the **client api** service is enabled (default port 45869)

### Obtain API Key

1. Go to **services → review services → client api**
2. Click **add → from api request**, or manually create a new access key
3. Name the key (e.g., `WD Tagger`)
4. Grant the following permissions:
   - **search and fetch files**
   - **edit file tags**
5. Click **apply** to get a 64-character hex key
6. Paste this key into the `hydrus_api_key` field in `config.yaml`

### (Recommended) Create a Dedicated Tag Service

To separate AI-generated tags from manual tags, consider creating a dedicated local tag service:

1. Go to **services → manage services**
2. Add a new **local tag service**, e.g., `AI Tags`
3. Set `target_tag_service` to `"AI Tags"` in `config.yaml`

---

## Configuration

The configuration file is `config.yaml` (YAML format). Copy from `config.example.yaml` on first use:

```yaml
# Hydrus Network connection
hydrus_api_url: "http://localhost:45869"
hydrus_api_key: "your-64-character-api-key"

# WD Tagger settings
default_model: "wd-vit-tagger-v3"    # Default model
models_dir: "./models"                # Model storage directory
use_gpu: false                        # Enable GPU acceleration

# Tag thresholds (0.0 - 1.0)
general_threshold: 0.35               # General tag confidence threshold
character_threshold: 0.85             # Character tag confidence threshold

# Hydrus tag service name
target_tag_service: "my tags"

# Tag prefixes
general_tag_prefix: ""                # General tag prefix (empty = no prefix)
character_tag_prefix: "character:"    # Character tag prefix
rating_tag_prefix: "rating:"          # Rating tag prefix

# Batch processing & Hydrus I/O (see config.example.yaml for a CPU-tuned example)
batch_size: 8                         # Images per ONNX batch
hydrus_download_parallel: 8           # Concurrent Hydrus file downloads per batch
hydrus_metadata_chunk_size: 256        # File IDs per get_file_metadata (gallery + tagging prefetch)
apply_tags_every_n: 0                 # Default N for incremental Hydrus writes (0 = off in UI until enabled)
cpu_intra_op_threads: 8               # ONNX Runtime CPU threads (see docs/UPGRADE.md)
cpu_inter_op_threads: 1

# Web UI server
host: "0.0.0.0"                       # All IPv4 interfaces (LAN). Use 127.0.0.1 for localhost-only.
port: 8199
```

---

## Starting the Server

Validate Python, installed libraries, `config.yaml` (if present), and writable `models_dir` / `logs/runs` **without** starting the server:

```bash
./wd-hydrus-tagger.sh check    # aliases: doctor, verify
```

The **`run`** helper (and the default no-subcommand start) runs this check **first**. To skip it: `./wd-hydrus-tagger.sh run --skip-req-check …` or `WD_TAGGER_SKIP_REQ_CHECK=1`. The check is **not** run for shell help (`help`, `usage`, `-h`, `--help` as the command), for `run.py` help only (`./wd-hydrus-tagger.sh --help` or `run --help`), or for **`--generate-config`** (first argument).

```bash
python run.py
# or (repo helper; forwards flags to run.py; pre-flight check unless skipped):
./wd-hydrus-tagger.sh run
./wd-hydrus-tagger.sh run --log-level DEBUG
./wd-hydrus-tagger.sh --log-level DEBUG   # same as above if the first arg is a flag
```

You should see:

```
INFO:     Uvicorn running on http://0.0.0.0:8199 (Press CTRL+C to quit)
```

On the **same machine**, open **http://127.0.0.1:8199** or **http://localhost:8199**. From **another device on your LAN**, use `http://<this-machine-LAN-IP>:8199` (stderr lists examples when `host` is `0.0.0.0`). Allow TCP **8199** through the host firewall if the page does not load remotely.

### Logging

- **Level:** `run.py` honors `--log-level` / `LOG_LEVEL` / `WD_TAGGER_LOG_LEVEL` (default INFO). After startup, the process environment is aligned with the CLI level.
- **Files:** Each run normally writes to `logs/runs/run-<timestamp>-<pid>.log` with `logs/latest.log` pointing at it; set `--log-file` or `WD_TAGGER_LOG_FILE` for a single rotating file instead. See `backend/logging_setup.py`.
- **Optimization metrics (INFO):**
  - **`ensure_model metrics`** — `memory_cache_hit=True` when ONNX was already loaded; else `False` and `duration_ms` includes disk/ONNX load.
  - **`load_model metrics`** — `disk_cache_hit=True` when `model.onnx` + `selected_tags.csv` were on disk before load (no HuggingFace fetch); `hf_wall_s` / `onnx_init_wall_s` / `threads_intra` / `threads_inter`.
  - **`tag_files metrics`** — end of a tagging call: `skipped_pre_infer_marker_files` (ONNX skipped via `wd14:` marker), `wall_hydrus_fetch_s`, `wall_onnx_predict_s`, `stale_wd_markers_dropped`, `hydrus_download_parallel`.
  - **`tagging_ws metrics model_prepare_wall_s`** — time to satisfy `ensure_model` at WebSocket session start.
  - **`tagging_ws session_metrics`** — session totals: `onnx_skipped_same_marker`, `onnx_skipped_higher_tier_marker`, `hydrus_duplicate_tag_strings_skipped_session`, `stale_wd_model_markers_dropped_from_proposals`.
  - **User Stop** — `tagging_ws user_cancel received` (includes `pending_hydrus_queue`), `winding_down exit_loop`, optional `winding_down final_hydrus_flush files=N`, summary line with `reason=user_cancel`, `tagging_ws user_stop_complete`.
  - **`apply_tags metrics`** — per Hydrus chunk: `hydrus_duplicate_tag_strings_skipped` (tags already in `storage_tags`).
  - **`TaggerEngine metrics`** — ONNX `session_init_wall_s` and thread counts.
- **`POST .../load`** JSON still includes `model_files_cached_on_disk` and `downloaded_from_hub` (API mirrors disk cache, independent of log lines).
- **Tag apply:** chunk summary lines include `dup_tags_skipped` / `items_unchanged_all_dupes` (proposed tags already in Hydrus for that service).
- **Perf summaries (INFO, stdlib-only):** `backend/perf_metrics.py` logs at **boundaries only** (no per-file overhead):
  - **`perf tagging_session`** — after each WebSocket tagging run: wall-clock session time, `model_prepare_s`, files processed, outer batch count, Hydrus apply totals, outcome (`ok`/`error` × `complete`/`stopped`/`error`), model name; also updates in-memory cumulative totals.
  - **`perf predict_batch`** — after `POST /api/tagger/predict` completes.
  - **`perf apply_tags_http`** — after `POST /api/tagger/apply` completes.
  - **`perf process_shutdown`** — when the FastAPI app lifespan ends (normal exit, Ctrl+C, etc.): uptime since **Application ready**, cumulative tagging session/file/batch totals, optional **peak RSS** (`resource.getrusage`, Linux/macOS).

### Models directory

Set `models_dir` in `config.yaml` to a **persistent** path (recommended: **`./models`** under the repo, resolved against the project root). If `models_dir` resolves under the OS temp directory (including paths containing **`pytest`**) or other ephemeral locations, **`load_config()` redirects it to `<repo>/models`** and logs a warning so ONNX files and **`.wd_model_cache.json`** survive reboots. **Pytest** sets **`WD_TAGGER_ALLOW_TMP_MODELS_DIR=1`** so tests can still use `tmp_path`. Each cached model folder includes a manifest with **`kind`: `wd_hydrus_tagger_model_cache`**, Hub revision metadata, and per-file sizes for integrity checks.

### Large models and big jobs

- Lower **`batch_size`** in Settings (e.g. 4) if you OOM on **wd-vit-large-tagger-v3** / **wd-eva02-large-tagger-v3**.
- For **very large searches** (many thousands of hits), raise **`hydrus_metadata_chunk_size`** in Settings (up to 2048) to cut Hydrus round-trips when loading gallery metadata or prefetching before tagging; lower it (e.g. 128) if Hydrus returns errors or very large JSON bodies time out.
- Optional: set **`HF_TOKEN`** for faster, higher-quota Hugging Face downloads.

### Performance, markers, Tag all

For **skip rules** (same model vs heavier WD marker), **per-batch timings** on **Tag all**, the **`generate-config`** wizard, and example outcomes on big libraries, see **[docs/PERFORMANCE_AND_TUNING.md](docs/PERFORMANCE_AND_TUNING.md)**.

### Web UI: sidebar, mobile, and memory

- **Sidebar:** Use **Toggle sidebar** in the header to hide or show the connection / search / tagger column. On **wide** viewports the panel collapses horizontally; on **narrow** ones (about phone / small tablet width) it opens as a **drawer** over the gallery. Tap the dimmed backdrop or press **Escape** to close the drawer. The choice is stored in **localStorage** as **`wd_tagger_sidebar_hidden`** (`1` = hidden, `0` = visible). The first visit on a narrow screen starts with the drawer **closed** so thumbnails use the full width.
- **Mobile layout:** Gallery tiles use a slightly smaller minimum column width; the settings modal can use a bottom sheet style; safe-area insets are respected where supported; primary controls use larger tap targets where it mattered most.
- **Long runs:** The server **releases each batch** of decoded images and ONNX I/O arrays after processing (see `tagging_service.tag_files` and `TaggerEngine.predict`). **Full `gc.collect()`** runs periodically every **four** inference batches (plus once after the run) to drop large decoder buffers without pausing after every batch. A WebSocket tagging session still keeps **cumulative results** in memory until the run completes or stops—very large **Tag all** jobs can grow that structure; if RAM is tight, lower **`batch_size`** (and see [Large models and big jobs](#large-models-and-big-jobs) above).

### Linux: speed-oriented defaults

The stack is already tuned for throughput: **parallel Hydrus fetches** (`hydrus_download_parallel`), **batched ONNX** (`batch_size`), **CPU EP** with explicit intra/inter thread counts, **`asyncio.to_thread`** for inference so the event loop stays responsive, and a shared **httpx** client with keep-alive (connection pool limits raised for many parallel GETs).

Further options:

- **`pip install -e ".[perf]"`** (Linux only): enables **uvloop** in Uvicorn for faster asyncio (Hydrus I/O + WebSocket tagging). No effect if the extra is not installed.
- **ONNX Runtime** uses **sequential execution mode** with **`intra_op_num_threads`** from config (good default when `inter_op_num_threads` is 1). Inputs are forced **C-contiguous float32** before `session.run` to avoid extra copies on some builds.
- If you see **CPU oversubscription** or unstable latency, experiment with **`OMP_NUM_THREADS`** / **`OPENBLAS_NUM_THREADS`** / **`MKL_NUM_THREADS`** (often `1` or match physical cores) so OpenMP-based libraries do not stack on top of ORT’s thread pool—see [ONNX Runtime](https://onnxruntime.ai/docs/performance/tune-performance/threading.html) and your BLAS build.

---

## Workflow

### Step 1: Connect to Hydrus

1. Open your browser on this PC to `http://127.0.0.1:8199` (or from another device on the network to `http://<server-ip>:8199` when `host` is `0.0.0.0` in `config.yaml`)
2. If the left panel is hidden, use **Toggle sidebar** in the header (on a phone it opens a drawer you can close with the backdrop or Escape). In the **Hydrus connection** panel:
   - **API URL**: Enter the Hydrus Client API address (default `http://localhost:45869`)
   - **API Key**: Enter your API key
3. Click the "Connect" button
4. On success, the status indicator changes from **red** to **green**
5. Connection info is saved in the browser's localStorage for automatic reconnection

### Step 2: Search for Images

After connecting, the "Search Images" panel appears:

1. Enter search terms in the search box
2. Separate multiple terms with commas
3. Click "Search"

**Common search syntax examples:**

| Syntax | Purpose |
|--------|---------|
| `system:archive` | Archived files |
| `system:inbox` | Inbox files |
| `system:filetype is image` | Image files only |
| `system:archive, system:filetype is image` | Archived images |
| `system:archive, -character:hatsune_miku` | Archived, excluding a character |
| `system:filesize < 10MB` | Files smaller than 10MB |

The number of found images is displayed below the search bar.

### Step 3: Select Images

Search results are displayed as a thumbnail grid:

- **Click**: Select / deselect a single image
- **Ctrl + Click**: Multi-select (toggle individual images)
- **Shift + Click**: Range select (select all images between two clicks)
- **"Select All" button**: Select all images on the current page
- **"Deselect All" button**: Clear all selections

Selected images are highlighted with a purple border and a checkmark.

Pagination is available at the bottom, showing 50 images per page.

### Step 4: Run Auto-Tagging

In the **Tagger** panel on the left:

1. **Model**: Choose a model (downloaded from HuggingFace on first use)
2. **General / Character threshold**: Confidence cutoffs for tag categories
3. **Tag service**: Target Hydrus tag service (**required** for incremental writes and for **Tag all search results**)
4. **Inference batch** (optional): Override `batch_size` for this run only (empty = use config)
5. **Push tags to Hydrus while tagging**: When enabled, writes tags every **N** successful files (set N or use the default from Settings). Does not apply to **Tag all** (see below).
6. **Per-file WebSocket progress**: More granular progress messages (slightly more overhead)
7. **Show progress bar**: Toggle the bar; the stats block below still updates when the bar is hidden
8. **Performance tuning overlay**: Optional; **Tag all** only — adds last-batch Hydrus fetch, ONNX, and apply seconds to the progress overlay (see [docs/PERFORMANCE_AND_TUNING.md](docs/PERFORMANCE_AND_TUNING.md))
9. **Tag selected** or **Tag all search results** (confirmation if 100+ files)

**Tag selected** uses the checkbox and **N** above: tags are pushed while the run proceeds only when incremental mode is on.

**Tag all search results** always pushes **each inference batch** to the selected tag service as soon as that batch is tagged (N is forced to the inference batch size). You do not need to enable the checkbox for that path. When that run ends (including **Stop** after the server’s final flush), if **nothing is left pending** on the selected tag service, **Apply all tags to Hydrus** stays **disabled** and a short **run summary** above the results explains why — there is nothing extra to flush.

During a run, the progress overlay shows **batches completed / total**, **files inferred**, **pre-infer skips** (same model marker vs heavier WD marker when applicable), **files and tag strings already sent to Hydrus**, and the **pending write queue** when relevant. **Pause** / **Resume** and **Flush pending to Hydrus** are available when incremental writes are active. **Stop** sends cancel to the server: the UI switches to **Stopping…**, disables the action buttons, and shows a short **winding-down** note; the server may still finish the **current inference batch**, then runs a **final Hydrus flush** for any **pending** queued files (important for **Tag all** with per-batch writes). Logs: **`tagging_ws user_cancel received`**, **`winding_down exit_loop`**, **`winding_down final_hydrus_flush`** (when a queue remains), **`reason=user_cancel`** on the session summary line, and **`user_stop_complete`**.

**Multiple browser tabs:** The server allows **one** active tagging WebSocket session at a time. If you open the app in another tab while a run is in progress, a banner explains that tagging is already running elsewhere; **Tag selected** and **Tag all** stay disabled until that run finishes. You can still **search, browse thumbnails, and view metadata** in the second tab. Use **View progress (read-only)** to reopen the progress overlay with live stats (polled from `GET /api/tagger/session/status`). Only the tab that opened the WebSocket can **Stop**, **Pause**, **Resume**, or **Flush**.

> **First-time note**: Model files are ~300–600 MB and download from HuggingFace on first load.

### Step 5: Review & Edit Results

After tagging completes or stops, a **brief summary** appears under the **Tagging results** heading (files processed, Hydrus apply counts, whether anything is still pending on the selected tag service). There is **no separate alert** for a normal stop or completion — read that summary and the list below.

After tagging completes, the results view shows:

For each image:
- **Thumbnail**
- **General tags** (blue): Descriptive tags like `1girl`, `blue hair`, `outdoors`
- **Character tags** (purple): Identified characters like `hatsune miku`
- **Rating tags** (orange): Ratings like `general`, `sensitive`

Each tag shows its confidence percentage.

**Editing:**
- Click the **×** button on any tag to remove it
- Removed tags will not be applied to Hydrus

### Step 6: Apply Tags to Hydrus

After reviewing:

1. Click "Apply All Tags to Hydrus" to push the **current** result list (including any edits you made in the UI).
2. Tags are written to the Tag Service selected in Step 4.
3. A success message is displayed (including how many tag strings were skipped because they were **already** on that file in Hydrus for that service).

If you already used incremental writes or **Tag all** with Hydrus, most files may already be updated; use this step for **corrections** or for runs where incremental mode was off.

**Duplicate-safe writes:** Before each `add_tags` call, the server loads Hydrus **`storage_tags`** for the target service and **only sends tag strings that are not already stored** (matching is case-insensitive and treats `_` and spaces as equivalent). Existing tags on the file are never removed.

Tags are written with configured prefixes:
- General tags: `blue hair` (or with custom prefix)
- Character tags: `character:hatsune miku`
- Rating tags: `rating:general`

Click "Back to Gallery" to return and process more images.

---

## Settings Panel

Click the **gear icon (⚙)** in the top right to open the settings panel. Saved fields are written to **`config.yaml`** via **`PATCH /api/config`** (validated server-side). **Diagnostics (ONNX Runtime)** controls optional **session profiling**; see [ONNX Runtime session profiling](#onnx-runtime-session-profiling) for behavior, env overrides, and trace viewing.

### Model Management

Shows all available models and their download status:

- **Downloaded**: Green "Downloaded" text
- **Not downloaded**: "Download" button to fetch from HuggingFace
- **Cache check failed**: Files exist but failed integrity checks (size, CSV shape); use **Download** again or **Verify cached models**
- **No manifest**: Older cache without `.wd_model_cache.json`; the next successful load or download writes it

**Verify cached models** checks every model folder for `model.onnx` + `selected_tags.csv` (minimum sizes, CSV columns and row count). Enable **Compare revision with Hugging Face** to see whether the Hub `main` branch has moved since your manifest was written (informational; local files still run until you re-download).

Models are stored under `models_dir` (default `./models` in the repo). That path is resolved against the **project root** so the cache stays stable when you start the server from different working directories. ONNX files already on disk are **not** re-fetched unless a file is missing or verification fails during **Load model** / **Download**.

### Defaults

- **Default model** — written to `config.yaml` as `default_model`; the Tagger panel dropdown is kept in sync when you save.
- **Default tag service name** — stored as `target_tag_service`. After you **Connect** to Hydrus, the app tries to select a tag service whose **display name** matches this string (case-insensitive). You can still change the dropdown manually.

### WD model marker

Mirrors `wd_skip_inference_if_marker_present`, `wd_skip_if_higher_tier_model_present`, `wd_append_model_marker_tag`, optional `wd_model_marker_template`, and `wd_model_marker_prefix`. These control skipping ONNX when a file already has the marker (or a heavier WD model marker), and appending the marker after inference.

### Apply all tags (HTTP)

**Files per apply request** maps to `apply_tags_http_batch_size` (1–512): chunk size for **Apply all tags to Hydrus** on the results screen. The server and UI use `range(0, N, batch)`-style chunking, so if you have fewer results than the configured size, a **single** request still carries all of them (no “wait for a full batch”).

**Tag all (WebSocket)** with incremental Hydrus writes uses the same idea: in-loop applies only fire when the pending queue reaches the threshold (`inference batch` for tag-all, or `apply_tags_every_n` otherwise); **any remainder is flushed once at end-of-run** (and on cancel/stop where applicable).

### Server

- **Allow “Stop server” from this UI** — `allow_ui_shutdown`. When off, the Stop button is hidden (useful if the web UI is reachable from untrusted networks).
- **Shutdown grace** — `shutdown_tagging_grace_seconds` (0–30): delay after notifying active tagging WebSockets before cancel and process exit.

**After Stop server:** the settings modal closes and the app switches to a **full-page “Server stopped”** view with shutdown metrics and a **Reload page** button (no blocking alert). The Python process exits shortly after; refresh only works once you start the server again (`python run.py`, `./wd-hydrus-tagger.sh run`, etc.).

**If the server stops elsewhere** (terminal Ctrl+C, `kill`, crash): while a tab stays open, a **background probe** hits **`GET /api/app/status`** about every 22 seconds (only when the tab is visible). Failed requests show the same **“Server unreachable”** screen so the UI does not look interactive when the backend is gone. Active tagging also starts a **short burst of probes** after a **`server_shutting_down`** WebSocket message so disconnects are noticed within about a second.

The tagging session banner in other tabs uses **polled** status; when the browser tab is in the background, polling slows down to reduce load unless you have a read-only progress overlay open.

### Tag Prefix Settings

| Field | Description | Default |
|-------|-------------|---------|
| General Prefix | Text prepended to general tags | (empty, no prefix) |
| Character Prefix | Text prepended to character tags | `character:` |
| Rating Prefix | Text prepended to rating tags | `rating:` |

**Custom examples:**
- Add a namespace to all tags: Set General Prefix to `wd:` → produces `wd:blue hair`
- Remove character prefix: Clear Character Prefix → produces `hatsune miku`

### Performance (CPU / GPU)

- **Inference batch size**, **CPU intra/inter-op threads**, **concurrent Hydrus downloads**, **Hydrus metadata chunk** (file IDs per `get_file_metadata`), and **default “write every N files”** are editable here. Invalid values are rejected when saving (validated server-side).
- **Use GPU (CUDA)** turns on GPU providers in ONNX Runtime when a CUDA build of `onnxruntime-gpu` is installed and visible to the app. This project is tested on **NVIDIA + CUDA**; **AMD discrete GPUs** are not wired in-tree today. A practical path on Linux is a **ROCm**-enabled ONNX Runtime wheel (or custom build) plus matching `ROCM_PATH` / driver stack, then extending `backend/tagger/engine.py` to try ROCm execution providers in the same way CUDA is tried—expect environment-specific tuning. Until then, **CPU** tuning (`cpu_intra_op_threads` ≈ physical cores, `cpu_inter_op_threads` usually **1**, `batch_size` 4–8 for large models) is the supported path on AMD Ryzen class hosts with **32 GB RAM** (see `config.example.yaml`).

**Model load and threading (best practice):** ONNX loads on **Load model** or the **first tagging run** and stays in RAM until shutdown or explicit unload (`TaggerEngine.load` uses **ORT_SEQUENTIAL** with **`intra_op_num_threads`** from config and **`inter_op_num_threads`** at **1** for typical WD graphs). After changing CPU thread fields, **save** and force a **fresh load** (toggle model or restart the server). **Batch size** drives activation memory more than thread count.

**Rough CPU timing (order of magnitude):** With **WD ViT Large v3**, batch size **8**, and **8** intra-op threads, wall time is often on the order of **~2–4 s per image** once the model is warm (excluding Hydrus I/O and marker-skipped files). ViT base is several times faster; your logs under `logs/runs/` include per-batch metrics for calibration on your machine.

The Hydrus HTTP client **reuses keep-alive connections** per API URL + key, which speeds up large batches (see `backend/hydrus/client.py`).

### ONNX Runtime session profiling

**Purpose:** Record an ONNX Runtime **session trace** (per-op timing) to see where CPU/GPU time goes — useful when tuning **`cpu_intra_op_threads`**, batch size, or investigating stalls. This is **not** the same as the Tag all **performance tuning overlay** (which only logs wall-clock fetch / predict / Hydrus apply per batch in the UI and logs).

**Default:** **Off** (`ort_enable_profiling: false`). Turning it on **slows inference** and writes **large** files; use short diagnostic runs only.

**How to enable (pick one):**

1. **Settings (gear) → Diagnostics (ONNX Runtime)** — check **Enable ONNX Runtime session profiling**, set **Trace output directory** (default `./ort_traces`, resolved like `models_dir` to the **project root**), **Save settings**. The **next** time the app loads an ONNX session (`Load model` or first tagging run), profiling is active for that session.
2. **`config.yaml`** — set `ort_enable_profiling: true` and optionally `ort_profile_dir: ./ort_traces`, then restart the server (or rely on the next load if config is re-read per your deployment).
3. **Environment variable** — `export WD_TAGGER_ORT_PROFILING=1` (or `true` / `yes` / `on`) **before** `python run.py`. This forces profiling **on at config load**, regardless of the saved checkbox, until the variable is unset. Useful for one-off captures without editing `config.yaml`.

**When traces are written:** ONNX flushes profiling when the **`InferenceSession`** is torn down — e.g. **switching models**, **`unload_model_from_memory`**, or **Stop server** / process exit (see `TaggerEngine.finalize_ort_profiling` / `end_profiling`). After a run, check the server log for **`TaggerEngine ORT profiling finalized path=...`** at **INFO**.

**Viewing traces:** See ONNX Runtime’s [Profiling tools](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html) and [Tune performance](https://onnxruntime.ai/docs/performance/tune-performance/) (Chrome trace / ORT-specific viewers depending on build).

**Repository note:** `ort_traces/` (or your chosen directory) is listed in **`.gitignore`** so traces are not committed.

More context: [docs/PERFORMANCE_AND_TUNING.md](docs/PERFORMANCE_AND_TUNING.md) (markers, Tag all prefetch, overlay, profiling).

---

## Available Models

All models are from [SmilingWolf](https://huggingface.co/SmilingWolf)'s WD Tagger v3 series.

| Model | Size | Description |
|-------|------|-------------|
| **WD ViT v3** | ~300 MB | Base ViT model, balanced speed & quality, **recommended for general use** |
| **WD SwinV2 v3** | ~300 MB | SwinTransformer V2, similar quality to ViT |
| **WD ViT Large v3** | ~600 MB | Large ViT model, higher accuracy but slower |
| **WD EVA02 Large v3** | ~600 MB | EVA02 large model, highest accuracy |

**Recommendations:**
- General use, batch processing → **WD ViT v3** or **WD SwinV2 v3**
- Maximum accuracy → **WD EVA02 Large v3**

---

## Threshold Tuning Guide

Thresholds determine the minimum confidence level for keeping tags.

### General Threshold (default 0.35)

| Range | Effect |
|-------|--------|
| **0.20–0.30** | Loose — more tags, may include inaccurate ones |
| **0.35–0.45** | Balanced — suitable for most use cases |
| **0.50–0.70** | Strict — only high-confidence tags |
| **0.70+** | Very strict — only the most obvious features |

### Character Threshold (default 0.85)

Higher thresholds are recommended for character recognition to avoid false positives:

| Range | Effect |
|-------|--------|
| **0.70–0.80** | Loose — may produce false character matches |
| **0.85–0.90** | Balanced — suitable for most cases |
| **0.90+** | Only highly confident character identifications |

---

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hydrus_api_url` | string | `http://localhost:45869` | Hydrus Client API address |
| `hydrus_api_key` | string | `""` | 64-character API key |
| `default_model` | string | `wd-vit-tagger-v3` | Default model name |
| `models_dir` | string | `./models` | Model storage (resolved **relative to the project root**). Temp/pytest paths are **coerced** to `<repo>/models` unless `WD_TAGGER_ALLOW_TMP_MODELS_DIR=1` (tests only). |
| `use_gpu` | bool | `false` | Enable GPU inference (CUDA build of ONNX Runtime; not AMD ROCm out of the box) |
| `general_threshold` | float | `0.35` | General tag threshold |
| `character_threshold` | float | `0.85` | Character tag threshold |
| `target_tag_service` | string | `my tags` | Default tag service |
| `general_tag_prefix` | string | `""` | General tag prefix |
| `character_tag_prefix` | string | `character:` | Character tag prefix |
| `rating_tag_prefix` | string | `rating:` | Rating tag prefix |
| `batch_size` | int | `8` | Images per inference batch |
| `cpu_intra_op_threads` | int | `8` | ONNX Runtime CPU intra-op threads |
| `cpu_inter_op_threads` | int | `1` | ONNX Runtime CPU inter-op threads |
| `ort_enable_profiling` | bool | `false` | **Diagnostic:** ONNX Runtime session profiling (large trace files, slower inference). Overridden to `true` if env **`WD_TAGGER_ORT_PROFILING`** is `1` / `true` / `yes` / `on`. |
| `ort_profile_dir` | string | `./ort_traces` | Directory for profiling output (resolved **relative to the project root**), prefix `wd_<model>_<timestamp>`. |
| `max_learning_cached_files` | int | `400000` | T-Learn: max files in the learning prefix held in RAM before commit (allowed 32–2 000 000). |
| `hydrus_download_parallel` | int | `8` | Parallel Hydrus `get_file` calls per batch (allowed 1–32) |
| `hydrus_metadata_chunk_size` | int | `512` | Max file IDs per Hydrus `get_file_metadata` call for gallery loads and tagging prefetch (allowed 32–2048; server clamps out-of-range values) |
| `tagging_skip_tail_batch_size` | int | `512` | **Tag all (WebSocket):** max outer batch size for marker-only tail work after metadata prefetch (no ONNX; allowed 32–2048). Larger values clear skip-only batches faster. See [docs/PERFORMANCE_AND_TUNING.md](docs/PERFORMANCE_AND_TUNING.md). |
| `apply_tags_every_n` | int | `0` | Default N for incremental Hydrus writes when enabled in the Tagger panel (Tag all ignores this and uses N = inference batch) |
| `wd_skip_inference_if_marker_present` | bool | `true` | Skip ONNX when the file already has the model marker tag (see FAQ) |
| `wd_skip_if_higher_tier_model_present` | bool | `true` | Skip ONNX when `storage_tags` already include a **heavier** WD model marker (`wd14:*` tier table in [docs/PERFORMANCE_AND_TUNING.md](docs/PERFORMANCE_AND_TUNING.md)) |
| `wd_append_model_marker_tag` | bool | `true` | After inference, append the marker tag (default `wd14:<model_name>`) so later runs can skip |
| `wd_model_marker_template` | string | `""` | Empty uses `wd14:{model_name}`; may include a `{model_name}` placeholder |
| `apply_tags_http_batch_size` | int | `100` | Max files per HTTP request for “Apply all tags” (server and UI chunk the same way) |
| `allow_ui_shutdown` | bool | `true` | Allow **Settings → Stop server** (`POST /api/app/shutdown`); set `false` if the UI is exposed to untrusted networks |
| `shutdown_tagging_grace_seconds` | float | `1.5` | After a shutdown request, wait this long after signaling flush before cancel + exit (0–30) |
| `host` | string | `0.0.0.0` | Web UI bind address (`0.0.0.0` = all IPv4 interfaces for LAN access; `127.0.0.1` = local only) |
| `port` | int | `8199` | Web UI port |

---

## Hard-coded limits & regression tests

These are fixed in code (not exposed in `config.yaml`); change them only by editing the backend.

| Location | What | Value / behavior |
|----------|------|------------------|
| `backend/hydrus/client.py` | Hydrus `httpx` timeout | 120 s overall, 15 s connect |
| `backend/hydrus/client.py` | Connection pool | `max_keepalive_connections=128`, `max_connections=192` |
| `backend/config.py` | `hydrus_download_parallel` | Validated **1–32** (default **8**) |
| `backend/config.py` | `hydrus_metadata_chunk_size` | Validated **32–2048** (default **512**) |
| `backend/config.py` | `tagging_skip_tail_batch_size` | Validated **32–2048** (default **512**) |
| `backend/routes/files.py`, `tagging_service.py` | Metadata chunk at runtime | Same 32–2048 clamp even if config were bypassed |
| `backend/config.py` | Ephemeral `models_dir` | Temp/pytest paths → **`<repo>/models`**; tests use env **`WD_TAGGER_ALLOW_TMP_MODELS_DIR=1`** |
| `backend/services/session_autotune.py` | Session auto-tune warm-up batches before exploring knobs | **3** (`DEFAULT_WARM_UP_BATCHES`) |

**Regression tests worth running after I/O or UI changes:** `tests/test_files_metadata_chunk.py` (metadata chunking), `tests/test_tag_merge.py` (marker detection / dedupe), `tests/test_tagger_websocket.py` (includes **`stopping`** / cancel wind-down, incremental final flush, session snapshot `stopping_source`) + `tests/test_tagging_service.py` (batch tagging pipeline), `tests/test_config_route.py` (PATCH allowlist including `hydrus_metadata_chunk_size`, `tagging_skip_tail_batch_size`), `tests/test_config.py` (defaults + **`stable_models_dir_for_config`** / temp coercion), `tests/test_generate_config_script.py` (Linux wizard helpers + non-Linux exit), `tests/test_frontend_english.py` (English UI strings and core module list), `tests/test_log_report.py` / `tests/test_listen_hints.py` (log digest + bind hints), `tests/test_learning_calibration_bytes.py` (T-Learn count/bytes split), `tests/test_dependencies.py` (Hydrus client singleton), `scripts/check_critical_coverage.py` (after `coverage run -m pytest` — see `docs/UPGRADE_V2.md` §18).

**Acceptance (operator):** See **`docs/UPGRADE_V2.md` §18.5** — `./wd-hydrus-tagger.sh check`, full **`pytest`**, Hydrus connect + Tag selected / Tag all + optional tuning and learning-phase calibration. Use **`config.example.yaml`** as the baseline for **8-core / 32 GB / NVMe** (e.g. **512** metadata chunk, **128** HTTP apply chunk); code defaults remain **256** / **100** for smaller machines.

**Tracing the last run:** `./wd-hydrus-tagger.sh log-report` reads `logs/latest.log` (symlink to the current `logs/runs/run-*.log`) and prints counts of **ERROR**/WARNING lines, **ensure_model** / **load_model** cache keywords (`memory_cache_hit`, `disk_cache_hit`, `disk cache miss`, `hub_fetch_this_call`), and Hydrus **metadata** lines (`tag_files metadata rows=…`, `tagging_ws metadata_prefetch`, legacy `tag_files metadata_fetched`; plus **`files metadata_hydrus`** from `POST /api/files/metadata` / gallery).

The Tagger UI **coalesces** rapid WebSocket **progress** counter updates to one DOM refresh per animation frame (`requestAnimationFrame` with timeout fallback) so long runs stay responsive; verbose per-file messages still update immediately.

---

## FAQ

### Q: Connection failed, showing red indicator

- Ensure Hydrus Client is running
- Ensure Client API service is enabled (services → manage services)
- Verify the API URL and port are correct (default 45869)
- Verify the API Key is correct and has sufficient permissions

### Q: How does “already tagged by this model” skipping work?

- After a successful inference run, the app can append a **marker tag** (default `wd14:<model_id>`, e.g. `wd14:wd-vit-large-tagger-v3`) to the same tag list that gets applied to Hydrus.
- On later runs, if that marker is already present in Hydrus **`storage_tags`**, **ONNX inference is skipped** for that file (no image fetch, no predict), which saves CPU time.
- With **Tag service** selected in the UI, the marker is detected on that service only. With no service selected, any service on the file is checked.
- Turn off with `wd_skip_inference_if_marker_present: false` or stop appending with `wd_append_model_marker_tag: false` in `config.yaml`.

### Q: Model download is slow or fails

- Models are downloaded from HuggingFace and require a stable internet connection
- Large models are ~600 MB, please be patient
- If download fails, manually download `model.onnx` and `selected_tags.csv` from HuggingFace and place them in `models/{model-name}/`
- Manual download URL example: `https://huggingface.co/SmilingWolf/wd-vit-tagger-v3`

### Q: Tagging is slow

- Use smaller models (ViT v3 or SwinV2 v3)
- Tune `batch_size`, `cpu_intra_op_threads`, `hydrus_download_parallel`, and `hydrus_metadata_chunk_size` in config or Settings
- Reduce `batch_size` if you run out of RAM
- Enable GPU acceleration (NVIDIA GPU + `onnxruntime-gpu`)
- The backend reuses HTTP connections to Hydrus; very slow networks benefit from a local Hydrus client

### Q: What is the difference between “Tag selected” and “Tag all” for Hydrus?

- **Tag selected**: Hydrus writes during the run only if **Push tags to Hydrus while tagging** is on and **N** is set (plus a tag service).
- **Tag all**: Always writes **after each inference batch** to the selected tag service, without using that checkbox.

### Q: Too many / too few tags

- Adjust the General threshold: higher = fewer but more accurate tags, lower = more but possibly inaccurate tags
- You can manually remove unwanted tags before applying

### Q: Character recognition is inaccurate

- Increase Character threshold (e.g., 0.90+)
- Use a Large model for better accuracy
- Note: The model can only recognize characters present in its training data (primarily from Danbooru)

### Q: How to process only specific file types?

Add file type filters in the search box:
- `system:archive, system:filetype is image` — images only
- `system:archive, system:filetype is png` — PNG only
- `system:archive, system:width > 512` — width greater than 512

### Q: How are underscores in tags handled?

The tool automatically converts underscores `_` to spaces. For example, `blue_hair` becomes `blue hair`.

### Q: Can I use this from another PC or phone on my Wi‑Fi?

The default **`host`** is **`0.0.0.0`**, so the web UI listens on all IPv4 interfaces. Use `http://<server-LAN-IP>:8199` from other devices (and open the port in the OS firewall if needed). The API and WebSocket use **relative URLs**, so the browser talks to whichever host loaded the page.

### Q: Can I use this remotely (internet)?

Binding **`0.0.0.0`** is intended for **trusted LANs**. For wider exposure:
- This tool has no built-in authentication
- Not recommended to expose on public networks
- For remote use, consider SSH tunnels or VPN

---

## Project Structure

```
wd-hydrus-tagger/
├── run.py                    # Entry point
├── scripts/
│   └── check_requirements.py # Pre-flight checks (also: ./wd-hydrus-tagger.sh check)
├── wd-hydrus-tagger.sh       # ./wd-hydrus-tagger.sh run | check | test
├── config.yaml               # Config file (create from example)
├── config.example.yaml       # Example config
├── pyproject.toml             # Python project metadata
├── backend/                   # FastAPI backend
│   ├── app.py                 #   Application factory
│   ├── config.py              #   Config loader
│   ├── dependencies.py        #   Dependency injection
│   ├── hydrus/                #   Hydrus API client
│   │   ├── client.py          #     Async HTTP client (pooled)
│   │   ├── tag_merge.py       #     storage_tags vs proposed tags (dedupe)
│   │   └── models.py          #     Data models
│   ├── tagger/                #   WD Tagger inference engine
│   │   ├── engine.py          #     ONNX inference
│   │   ├── preprocess.py      #     Image preprocessing
│   │   ├── labels.py          #     Label CSV parser
│   │   └── models.py          #     Result data models
│   ├── routes/                #   API routes
│   │   ├── connection.py      #     Connection management
│   │   ├── files.py           #     File browsing
│   │   ├── tagger.py          #     Tagging endpoints
│   │   └── config_routes.py   #     Config management
│   └── services/              #   Service layer
│       ├── tagging_service.py #     Batch tagging orchestrator
│       └── model_manager.py   #     Model download manager
├── tests/                     # Pytest suite (run: PYTHONPATH=. pytest)
├── docs/                      # Extra documentation (e.g. UPGRADE.md)
├── frontend/                  # Web frontend
│   ├── index.html             #   Single-page application
│   ├── css/style.css          #   Stylesheet (dark theme)
│   └── js/                    #   JavaScript modules
│       ├── app.js             #     App initialization
│       ├── api.js             #     Backend API client
│       ├── state.js           #     State management
│       ├── components/        #     UI components
│       └── utils/             #     Utility functions
└── models/                    # Model files (auto-created)
```

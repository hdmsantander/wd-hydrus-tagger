# Performance, batching, and tagging session upgrade

## 0. Dependency lower bounds

`requirements.txt` and `pyproject.toml` track minimum versions together (e.g. **FastAPI** ≥ 0.115.6, **httpx** ≥ 0.28, **ONNX Runtime** ≥ 1.20). Refresh a venv with `pip install -U -r requirements.txt` or `pip install -e ".[dev]"` after pulling.

## 1. Hardware target (reference)

AMD Ryzen 7 5700X3D, 32 GB RAM: prefer **8** ONNX intra-op threads (physical cores), **inter-op 1**, **inference batch 8**, **8** parallel Hydrus downloads per batch, **`hydrus_metadata_chunk_size` 256** (fewer Hydrus metadata round-trips on huge searches). See `config.example.yaml` for a matching profile (`use_gpu: false` unless you use CUDA).

## 2. Implemented behavior (current tree)

### 2.1 ONNX Runtime

- `SessionOptions`: graph optimizations, `intra_op_num_threads`, `inter_op_num_threads` from config, plus explicit `enable_mem_pattern` / `enable_cpu_mem_arena` for CPU-friendly reuse of large weights (`backend/tagger/engine.py`).

### 2.2 Configuration (`backend/config.py`)

| Field | Default | Notes |
|-------|---------|--------|
| `batch_size` | 8 | Images per ONNX run / outer WebSocket step. |
| `cpu_intra_op_threads` | 8 | 5700X3D-oriented. |
| `cpu_inter_op_threads` | 1 | |
| `hydrus_download_parallel` | 8 | Concurrent `get_file` calls inside each batch (`asyncio.gather` + semaphore); max **32**. |
| `hydrus_metadata_chunk_size` | 256 | File IDs per `get_file_metadata` for gallery metadata and tagging prefetch; max **2048**, min **32**. |
| `apply_tags_every_n` | 0 | Default **N** pre-filled in the Tagger panel when the user enables incremental Hydrus writes. `config.example.yaml` may use a non-zero example (e.g. 8). |

`PATCH /api/config` merges into a full `AppConfig` via **`AppConfig.model_validate`**; invalid values return `success: false` and an `error` payload (`backend/routes/config_routes.py`).

### 2.3 Tagging service (`backend/services/tagging_service.py`)

- **`load_metadata_by_file_id`**: shared chunked `get_file_metadata` helper (cancel-aware).
- **`tag_files(..., prefetched_meta_by_id=...)`**: optional map from a **WebSocket session prefetch** so outer batches do not repeat metadata for the same file IDs (see `tagging_ws metadata_prefetch` + `tag_files metadata rows=… source=prefetch` in logs).

- Optional **`batch_size`** override per call (clamped 1–256).
- Optional **`model_name`** to load before inferring.
- Optional **`cancel_event`**: checked between internal batches.
- Optional **`service_key`**: used with **`wd_skip_inference_if_marker_present`** to detect the **`wd14:{model}`** marker on that Hydrus service; if empty, any service’s `storage_tags` are checked.
- **`wd_append_model_marker_tag`** (config): append marker to inferred results so Hydrus apply writes it; **`wd_skip_inference_if_marker_present`**: skip fetch+ONNX when marker already stored.
- Parallel image fetch per batch (bounded by `hydrus_download_parallel`), only for files not skipped.
- Skipped files return **`skipped_inference`: true**, empty **`tags`**, and **`skip_reason`**: `wd_model_marker_present` or `wd_skip_higher_tier_model_present` (when **`wd_skip_if_higher_tier_model_present`** is on and a strictly higher-tier `wd14:*` marker exists — see `backend/hydrus/tag_merge.py` and [PERFORMANCE_AND_TUNING.md](PERFORMANCE_AND_TUNING.md)).
- Each inferred result includes **`tags`** / **`formatted_tags`** for Hydrus `add_tags`.
- **Memory:** after batches that run inference, references to decoded images are cleared; **`gc.collect()`** runs every **four** such batches and **once** after `tag_files` finishes, trading a small amount of retained RAM between cycles for fewer full-GC pauses than collecting after every batch.

### 2.4 HTTP `POST /api/tagger/predict`

- Optional body field **`batch_size`** overrides config for that request only.
- Optional **`service_key`** — same marker detection as WebSocket (skips inference when marker already present).

### 2.5 WebSocket `/api/tagger/ws/progress`

After **`ensure_model`**, the server runs **`load_metadata_by_file_id`** once for the full **`file_ids`** queue (same chunk size as config), logs **`tagging_ws metadata_prefetch`**, and passes the map into each **`tag_files`** batch as **`prefetched_meta_by_id`**. If prefetch raises, logging falls back to per-batch metadata inside **`tag_files`**.

**First message** (JSON) — effectively `action: run`:

- `file_ids`, `general_threshold`, `character_threshold`
- `model_name` (optional; the UI normally loads the model before connecting)
- `batch_size` (optional override)
- `service_key` (Tag service dropdown) — used to detect the model marker on that service; Hydrus **`add_tags`** runs only when `apply_tags_every_n > 0` and `service_key` is non-empty (incremental apply).
- `apply_tags_every_n` — chunk size for those Hydrus writes; each chunk triggers **`tags_applied`**.
- `stream_verbose` — if true, one **`file`** message per inferred file (plus a batch **`progress`** summary after each outer batch). If false, one **`progress`** per outer batch (includes `results` for that batch).
- `hydrus_download_parallel` (optional) overrides config for that session.
- `tag_all` — boolean; when true with incremental apply, behavior matches **Tag all search results** in the UI.
- `performance_tuning` — boolean; only honored when **`tag_all`** is true. Adds **`performance_tuning`** object to batch **`progress`** messages (fetch / predict / apply seconds). See [PERFORMANCE_AND_TUNING.md](PERFORMANCE_AND_TUNING.md).

**Control messages** (JSON): `{"action":"cancel"}`, `pause`, `resume`, `flush`. The server acknowledges `pause` / `resume` with **`control_ack`**. **`flush`** applies any **pending** queued results to Hydrus immediately (manual flush). On **`cancel`**, the server logs **`tagging_ws user_cancel received`** (with `pending_hydrus_queue`), pushes a WebSocket **`stopping`** message (`message`, `pending_hydrus_queue`) for the UI / **`GET /api/tagger/session/status`** snapshot (`stopping_source`: **`user`**), then exits the outer loop when safe, logs **`winding_down exit_loop`**, may log **`winding_down final_hydrus_flush`** if incremental apply left a queue, and ends with **`stopped`** (summary line includes **`reason=user_cancel`**) plus **`tagging_ws user_stop_complete`**.

**Loop behavior (outer batch):**

1. Infer a batch of files; optionally stream per-file events.
2. Apply Hydrus writes according to `apply_tags_every_n` (extend `pending_apply`, drain when `len(pending) >= N`).
3. Run `drain_flush` if a flush was requested.
4. Send **`progress`** with cumulative stats **after** writes for that batch, so `total_applied` / `total_tags_written` are up to date.
5. Short **`asyncio.sleep(0.01)`** so the control task can process cancel/flush before the next batch.

**`progress` / `file` payload highlights:**

- `batches_completed`, `batches_total` (planned outer batches for the run)
- `total_applied` (files successfully written to Hydrus so far)
- `total_tags_written` (sum of tag string counts sent via `add_tags`)
- `inference_batch`, `batch_inferred`, `batch_skipped_inference`, `batch_predicted`, `current`, `total`

**`tags_applied` payload:**

- `count` (files in this chunk), `chunk_tag_count` (new tag strings in this chunk)
- `chunk_duplicates_skipped`, `total_duplicates_skipped` (proposed tags already in Hydrus `storage_tags` for that service)
- `total_applied`, `total_tags_written`, `pending_remaining`, `manual_flush`

**Terminal messages:** optional mid-session **`stopping`** (user cancel or mirrored in snapshot for observers); then **`complete`** or **`stopped`** with `results`, `total_processed`, `total_applied`, `total_tags_written`, `total_duplicates_skipped`, `batches_completed`, `batches_total`, `inference_batch`.

**Hydrus apply helper (`_apply_results_chunk`):** One **`get_file_metadata`** per chunk, then **`filter_new_tags`** using **`storage_tags`** for the target `service_key` (`backend/hydrus/tag_merge.py`: normalized compare, `_` vs space). Returns **`(files_written, new_tag_strings, duplicates_skipped)`**. **`POST /api/tagger/apply`** splits large bodies using **`apply_tags_http_batch_size`** and returns `skipped_duplicate_tags` summed across chunks.

**End-of-session results:** When a **`service_key`** was set, **`complete`** / **`stopped`** (and **`error`** with `partial_results`) run **`_trim_ws_results_to_pending_for_service`**: one batched metadata fetch, then each result’s **`tags`** / **`formatted_tags`** / structured tag dicts are reduced to tags **not** already in Hydrus for that service—so the results screen and “Apply all” only show **pending** strings (no duplicates of what incremental apply already wrote). Payload field **`pending_hydrus_files`** counts files that still have ≥1 pending tag.

### 2.6 Frontend

- **English UI** by default (`index.html` `lang="en"`). Automated check: `tests/test_frontend_english.py`.
- **Sidebar:** header **Toggle sidebar** — wide viewports collapse the left column; narrow viewports use a drawer + backdrop (Escape closes). Preference: **`wd_tagger_sidebar_hidden`** in `localStorage`. **Progress overlay** is `position: fixed` with a high z-index so it stays above the drawer during tagging.
- **Tagger panel**: inference batch override, incremental Hydrus + N, verbose per-file progress, show/hide progress bar.
- **Tag all search results**: requires a tag service; forces incremental behavior with **`apply_tags_every_n` = inference batch** so each ONNX batch is written to Hydrus as it completes (checkbox not required).
- **Tag selected**: uses the **Push tags to Hydrus** checkbox and N as before.
- **Progress overlay**: stats block (batches, inferred files, Hydrus file/tag counts, pending queue), **Stop**, **Pause** / **Resume**, **Flush** when incremental applies are active.
- **Results**: **`results-run-summary`** (status line) after each run; **`pending_hydrus_files`** from **`complete` / `stopped`** drives whether **Apply all tags** is enabled when a **service_key** was used (trim found nothing left → button disabled).

### 2.7 App control (`backend/routes/app_control.py`)

- **`GET /api/app/status`** — `active_tagging_sessions`, `loaded_model`, `models_dir`, `allow_ui_shutdown`, grace seconds.
- **`POST /api/app/shutdown`** (when `allow_ui_shutdown`): notifies open tagging WebSockets (`server_shutting_down`), sets **flush** on all sessions, waits **`shutdown_tagging_grace_seconds`**, sets **cancel**, calls **`TaggingService.unload_model_from_memory()`** (RAM only; on-disk `models_dir` unchanged), then schedules process exit (SIGINT). Response **`metrics`** summarize sessions and ONNX release.

Active tagging sessions register in **`backend/services/tagging_session_registry.py`** from **`progress_ws`**.

### 2.8 Hydrus HTTP client (`backend/hydrus/client.py`)

- **Shared `httpx.AsyncClient` per** `(api_url, access_key)` with keep-alive limits (`max_keepalive_connections=64`, `max_connections=96`).
- **`aclose_all_hydrus_clients()`** on app shutdown (`backend/app.py` lifespan).
- **`invalidate_hydrus_client_pool()`** when `hydrus_api_url` or `hydrus_api_key` changes after a validated `PATCH /api/config`.

### 2.9 Tests & helper script

- **`tests/`** — preprocess, config, tagging service (mocked), batch clamp, config API, Hydrus pool, **`test_tagger_websocket.py`** (batched progress, pause/resume, flush, cancel, tag counts), **`test_predict_route.py`**, **`test_frontend_english.py`**, **`test_tagging_service.py`** (thumbnail fallback, metadata order, video thumbnail-only path, predict failure skip), **`test_check_requirements_script.py`** (pre-flight script).
- Run: `PYTHONPATH=. pytest` or **`./wd-hydrus-tagger.sh test`** (repo root; sets `PYTHONPATH`).

### 2.10 Disk cache, video fetch path, apply logging, predict resilience

- **`models_dir`**: use **`./models`** (repo root). Ephemeral/temp/pytest paths are **coerced** to **`<repo>/models`** at load (see §2.11); pytest sets **`WD_TAGGER_ALLOW_TMP_MODELS_DIR=1`**.
- **`load_model`**: INFO lines **disk cache hit** vs **disk cache miss** (Hugging Face). **`POST /api/tagger/models/{name}/load`** JSON adds **`model_files_cached_on_disk`** (files were present before this call) alongside **`downloaded_from_hub`**.
- **Video**: when Hydrus **`mime`** is `video/*`, tagging uses **`get_files/thumbnail` only** (avoids downloading huge originals). Other types still try the full file first, then thumbnail on decode failure.
- **`tag_files`**: per-batch INFO logs include **fetch** and **predict** durations (seconds). If **ONNX `predict` raises**, the error is logged with traceback and **that batch is skipped** (WebSocket session continues). Each batch **finally** closes PIL handles, drops **batch-local** dicts (`infer_by_fid`, `skipped_by_fid`, fetch lists), clears the **predictions** reference, and may **`gc.collect()`** when inference was attempted.
- **`TaggerEngine.predict`**: after **`session.run`**, input/output ndarray references are released in **`finally`** so long CPU runs do not retain redundant tensor buffers longer than necessary (ORT still owns its arena).
- **`_apply_results_chunk`**: end-of-chunk **summary** log (`items`, `with_tags`, `files_written`, `new_tag_strings`, `dup_tags_skipped`, `items_unchanged_all_dupes`). **`add_tags`** failures log and **re-raise** so the client sees the error.
- **Libraries**: deprecated **`resume_download`** removed from `hf_hub_download`; **`filelock`** / **`huggingface_hub`** root noise capped; extra **websockets** loggers capped when app log level is DEBUG.

### 2.11 Model cache manifest, verify API, and resolved `models_dir`

- **`backend/config.py`**: `load_config()` resolves relative **`models_dir`** (e.g. `./models`) against the **repository root** so ONNX files stay in one place regardless of process CWD. Paths under the system temp tree or containing **`pytest`** are **redirected** to **`<repo>/models`** via **`stable_models_dir_for_config()`** (override with **`WD_TAGGER_ALLOW_TMP_MODELS_DIR=1`** for pytest). **`scripts/check_requirements.py`** uses the same helper so preflight matches runtime.
- **Disk manifest** (`.wd_model_cache.json`): includes **`kind`: `wd_hydrus_tagger_model_cache`** plus Hub revision and file sizes for reuse/verification.
- **`backend/services/model_manager.py`**: After a successful download, writes **`.wd_model_cache.json`** per model (Hub `main` revision SHA, file sizes). **`hf_hub_download`** skips files that already exist; **`revision=main`** is explicit. **`verify_model`** checks minimum ONNX/CSV sizes, CSV columns/row count, manifest size consistency, and optionally compares the manifest SHA to the current Hub tip.
- **`TaggingService.load_model`**: Runs local verification before trusting disk cache; on failure, re-downloads from Hub; **`repair_manifest_if_missing`** when ONNX+CSV exist but the manifest is absent. INFO metrics include **`hub_fetch_this_call`**.
- **`POST /api/tagger/models/verify`**: Body `{ "check_remote": bool, "model_name": string? }` — validate cache(s) without loading ONNX into RAM.
- **`GET /api/tagger/models`**: Each entry includes **`cache_ok`**, **`cache_issues`**, **`manifest_present`**, **`revision_sha`**.
- **`models/README.md`** is tracked; **`models/*`** remains gitignored.

### 2.12 Pre-flight requirements check (`scripts/check_requirements.py`)

- **`./wd-hydrus-tagger.sh check`** (aliases `doctor`, `verify`) runs **`scripts/check_requirements.py`**: Python ≥ 3.10, imports for all runtime deps (FastAPI, uvicorn, httpx, ONNX Runtime, …), optional **`config.yaml`** validation via **`AppConfig`**, **`models_dir`** and **`logs/runs`** writable under repo root, Linux hint for **`uvloop`** if `[perf]` installed.
- **`run` / `start` / default** invoke the same check **before** `exec run.py`. Skip with **`--skip-req-check`** on the run.py argv or **`WD_TAGGER_SKIP_REQ_CHECK=1`** (not recommended).
- Tests may set **`WD_TAGGER_CONFIG_PATH`** to point at a temp invalid YAML (validation failure) or **`WD_TAGGER_CHECK_ROOT`** for a fake repo root missing **`run.py`**.

## 3. Resolved items (formerly “technical debt” in earlier drafts)

| Topic | Resolution |
|-------|------------|
| WebSocket one-file loops | **Batched** outer loop + optional verbose per-file events. |
| Per-request batch size | **`predict` body** + **WebSocket** `batch_size`. |
| Config PATCH safety | **Pydantic validation** on merge. |
| Hydrus download parallelism | **Semaphore + gather** in `tag_files`; tunable via config / WS. |
| New `AsyncClient` per Hydrus request | **Pooled keep-alive clients** + shutdown / config invalidation. |
| Progress vs Hydrus ordering | **`progress` sent after** per-batch Hydrus applies so UI shows current totals. |

**Still optional / future**

- Finer-grained `cancel_event` checks inside a single very large internal batch (cancel is still primarily between outer batches).

## 4. Operator notes

1. **Partial tagging**: **Stop** or disconnect; if you used **incremental Hydrus writes** (or **Tag all**) with a **tag service** selected and the server reports **no pending tags** on that service after trim, **Apply all tags** is **disabled** and the results **summary** explains it. If you ran **without** incremental writes (or without a service), use **Apply all tags** for anything still listed.
2. **Tag selected + Hydrus**: choose tag service, enable **Push tags to Hydrus while tagging**, set N (defaults from Settings / config).
3. **Tag all**: pick tag service first; each inference batch is written automatically (N = batch size); when nothing remains pending, **Apply all tags** stays off.
4. **GPU**: set `use_gpu: true` in `config.yaml` if using CUDA; CPU thread options still apply for CPU fallback.

## 5. Summary

The app uses **WebSocket-first tagging** with **configurable inference batches**, **pause / resume / flush / cancel**, **optional Hydrus writes every N files** (or **per batch for Tag all**), **live progress metrics** (batches, files, tag strings), **validated configuration updates**, **parallel Hydrus downloads**, and **reused HTTP connections** to Hydrus.

---

## Post (changelog)

| Date (approx.) | Change |
|----------------|--------|
| **Hydrus dedupe** | `tag_merge.py` + `_apply_results_chunk` skips tags already in **storage_tags**; WS + apply metrics for duplicates skipped. |
| **Docs + UX metrics** | README API map; **Tag all** → per-batch Hydrus; progress payloads and overlay (batch %, file %, duplicate skips). |
| **Code** | Single `asyncio.sleep(0.01)` yield points (replacing paired `sleep(0)` + `sleep(0.01)`); `_apply_results_chunk` tuple unpack cleanup in `apply` route. |
| **Pool + docs** | `httpx.AsyncClient` reuse per Hydrus base URL + API key (`Limits` 128 keep-alive / 192 max connections; 120 s / 15 s connect timeout); close all on shutdown; invalidate pool when Hydrus URL/key changes after PATCH. |
| **Large metadata** | `POST /api/files/metadata` and tagging prefetch chunk `file_ids` by `hydrus_metadata_chunk_size`; `test_files_metadata_chunk.py`. |
| **Marker check** | `all_normalized_storage_tag_keys` avoids repeated service scans when detecting `wd14:` marker on any service. |
| **UI progress** | Tagger coalesces frequent `progress` WebSocket updates via `requestAnimationFrame` (`progress.js`). |
| **LAN bind** | Default `host` `0.0.0.0`; stderr lists example `http://<ip>:port/` (`listen_hints.py`). `./wd-hydrus-tagger.sh log-report` summarizes `logs/latest.log` (`log_report.py`). |
| **Gallery metadata log** | `files metadata_hydrus` INFO after each `POST /api/files/metadata` (chunk count + rows). |
| **Stop / wind-down** | WebSocket `stopping` + INFO `user_cancel` / `winding_down` / `user_stop_complete`; UI overlay + observer snapshot `stopping_source` user vs server. |
| **Offline SPA** | After UI **Stop server**, full-page “Server stopped” + metrics; `GET /api/app/status` poll (~22s, visible tabs) for CLI/crash; burst probe after `server_shutting_down`. |
| **models_dir coercion** | Temp/pytest → `<repo>/models`; manifest `kind: wd_hydrus_tagger_model_cache`; `WD_TAGGER_ALLOW_TMP_MODELS_DIR` for pytest. |
| **UI** | English-default SPA; `tests/test_frontend_english.py`. |
| **Tests** | `test_hydrus_client.py`, extended WebSocket and predict tests, `wd-hydrus-tagger.sh test`. |
| **Prior** | WebSocket batched tagging, incremental `add_tags`, `predict` `batch_size`, ONNX thread options, `hydrus_download_parallel`, validated PATCH — see sections 2–3 above. |
| **Tagging ops** | Video → thumbnail-only; batch timings; predict batch skip on ONNX error; apply chunk summary + `model_files_cached_on_disk`; stable `models_dir` warning. |
| **UI + memory** | Collapsible sidebar / mobile drawer (`layout.js`); per-batch cleanup + ONNX tensor drop in `predict`; README notes on WebSocket cumulative `results` RAM. |
| **Results UX** | Run summary text + disable **Apply all tags** when `service_key` was set and **`pending_hydrus_files` === 0** after trim (typical **Tag all** / incremental completion or stop after final flush). |
| **Perf metrics** | `backend/perf_metrics.py`: **`perf tagging_session`** after each WS run; **`perf predict_batch`** / **`perf apply_tags_http`**; **`perf process_shutdown`** on app lifespan exit; tests `test_perf_metrics.py`; totals reset in pytest `conftest`. |

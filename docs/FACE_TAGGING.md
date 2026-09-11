# Face tagging (AI-assisted)

WD Hydrus Tagger integrates the two-step face pipeline from [hydrus-ai-taggers](https://github.com/lannashelton/hydrus-ai-taggers):

1. **Detect** — InsightFace `buffalo_l` finds faces, stores 512-d embeddings in SQLite, tags files with `ai face detected` or `face not visible`.
2. **Recognize** — Unsupervised clustering assigns `person:p1`, `person:p2`, … tags and applies them **directly** to Hydrus via the API (same as hydrus-ai-taggers — no pending queue or in-app confirmation). Map real names in Hydrus via **tag siblings** (e.g. `person:p1` → `person:alice`).

**Hydrus apply model:** Face tagging does **not** use the WD tagger’s “review pending tags” flow. Each detect/recognize step calls Hydrus `add_tags` on the selected **face tag service**; tags appear in storage immediately. Use a dedicated local service (e.g. “ai faces”) so face markers and `person:p#` tags stay separate from WD results.

## Install

Native (from repo root):

```bash
pip install -e ".[face]"
# Optional GPU (pick one ONNX GPU package — do not mix cuda + rocm wheels):
pip install -e ".[gpu]"          # NVIDIA CUDA
pip install -e ".[rocm]"         # AMD ROCm (Linux)
pip install -e ".[directml]"     # Windows AMD/Intel via DirectML
```

Docker: the official image installs `.[face]` automatically. Bind-mount `./face_data` for the embedding database (see `docker-compose.yml`). AMD GPU: rebuild with **`docker-compose.amd.yml`** (`./start.sh docker-run` does this when `/dev/kfd` is present).

## GPU backends (WD + face)

| `gpu_backend` | ONNX package | Hardware |
|---------------|--------------|----------|
| `auto` | whichever EP is installed | CUDA → ROCm → DirectML → CPU |
| `cuda` | `onnxruntime-gpu` | NVIDIA |
| `rocm` | `onnxruntime-rocm` | AMD Radeon (Linux + ROCm) |
| `directml` | `onnxruntime-directml` | Windows AMD/Intel |
| `cpu` | `onnxruntime` | CPU only |

Set in `config.yaml` or env **`WD_TAGGER_GPU_BACKEND=rocm`**. Enable **`use_gpu: true`** (or env **`WD_TAGGER_USE_GPU=true`**, which wins over a read-only Docker `config.yaml`).

Verify installed providers: **`GET /api/face/providers`**.

### AMD ROCm (Linux)

1. Install ROCm drivers on the host. Native: `pip install -e ".[rocm,face]"` **or** AMD’s wheel `pip install onnxruntime-migraphx -f https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/` (ROCm 7.2.x; do not mix with `onnxruntime` / `onnxruntime-gpu`).
2. Set `use_gpu: true` and `gpu_backend: rocm`.
3. Docker: `./start.sh docker-run-all -d` on a ROCm host installs the **`onnxruntime-migraphx`** wheel and maps `/dev/kfd` + `/dev/dri`. Arch/Manjaro **`/opt/rocm` must not be bind-mounted** into the Debian slim image (glibc mismatch). For GPU inference on this machine, run the tagger **natively**: `sudo pacman -S migraphx`, replace CPU ORT with `pip install onnxruntime-migraphx -f https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/`, set `use_gpu: true` and `gpu_backend: rocm`.

### Windows DirectML

1. `pip install -e ".[directml,face]"`.
2. `use_gpu: true`, `gpu_backend: directml`.

## Web UI workflow

1. In **Settings → Face models**, refresh/verify the `buffalo_l` cache (Download if missing). Load into memory is optional — the first detect also loads it.
2. Connect to Hydrus and search files.
3. Open the **Face tagging** sidebar panel (thresholds and markers live under **Settings → Face tagging**).
4. **Detect faces** on selection or all search results (same progress overlay as WD tagging, Stop only). Marker tags (`ai face detected` / `face not visible`) are written straight to Hydrus.
5. **Recognize persons** — staged clustering + immediate Hydrus apply (`person:p#` + recognized marker). No pending review step.
6. In Hydrus, add tag siblings from `person:p#` to real character names.

Settings sliders:

- **Detection threshold** — minimum InsightFace det score (default 0.6).
- **Cluster max distance** — neighbour radius for person grouping (default 0.5, cosine).

## API

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/face/status` | DB + model + disk-cache stats |
| GET | `/api/face/providers` | Installed ONNX EPs |
| GET | `/api/face/models` | Face pack cache (on disk / in memory), same shape as WD models |
| POST | `/api/face/models/verify` | Check required `buffalo_l` ONNX files |
| POST | `/api/face/models/download` | Download InsightFace pack into `face_models_dir` |
| POST | `/api/face/models/load` | Load InsightFace |
| POST | `/api/face/detect` | HTTP batch detect |
| WS | `/api/face/ws/progress` | Batch detect with progress |
| GET | `/api/face/session/status` | Whether a detect WebSocket is active |
| POST | `/api/face/recognize` | Cluster + apply person tags |
| POST | `/api/face/reset` | Clear person assignments |
| POST | `/api/face/clean` | Remove orphans deleted in Hydrus |

## Configuration (`config.yaml`)

| Key | Default | Notes |
|-----|---------|--------|
| `face_models_dir` | `./models/face` | InsightFace model cache |
| `face_embeddings_db_path` | `./face_data/face_embeddings.db` | **Back up** this file |
| `face_target_tag_service` | `""` | Falls back to `target_tag_service` |
| `face_skip_if_detected` | `true` | Skip files already marked detected/not visible, or with any `person:` tag (second run is fast) |
| `face_recognition_stages` | `[20,5,3,1]` | Staged `min_faces` for clustering |
| `face_model_load_timeout_seconds` | `180` | GPU model load timeout before CPU retry |
| `face_inference_timeout_seconds` | `180` | Per-image inference timeout (GPU) |

**Videos:** face **detect** processes **still images only** (`image/*`). `video/*` files in the queue are skipped immediately (no download, no model frames). `face_video_frame_count` is reserved for a future optional video path.

Marker tags (configurable): `face_marker_detected`, `face_marker_not_visible`, `face_marker_recognized`, `face_person_tag_prefix`.

| Key | Default | Notes |
|-----|---------|-------|
| `face_skip_if_detected` | `true` | Skip when Hydrus already has detect markers (`ai face detected`, `face not visible`) **or** any `person:` tag on the target service |
| `face_skip_if_in_db` | `true` | Skip when embeddings for that file hash exist in SQLite |

**Incremental workflow (default):** **Detect** appends embeddings to the local database. Already-processed files are skipped via Hydrus markers and/or the embedding DB (`face_skip_if_in_db`). **Recognize** clusters the **entire** database (not the current gallery selection). Default mode **Full recluster** (`recluster_all: true`) regroups every face as the DB grows; **Incremental refine** (`refine_incremental: true`) links only new/unassigned faces and writes Hydrus tags only for files touched in that run.

**Interrupted runs (tag-as-you-go):** Unlike WD tagging’s optional pending queue, face tags commit to Hydrus immediately. **Detect** writes `ai face detected` / `face not visible` after each file finishes (embeddings are stored in SQLite per face). **Stop** cancels at the next file boundary — completed files keep their DB rows and Hydrus markers. Re-run Detect on the same search: skipped files (`face_skip_if_in_db` / marker skip) avoid re-inference. **Recognize** (staged, default) pushes `person:p#` tags to Hydrus after **each clustering stage**, so a stopped recognize run still leaves person tags from completed stages.

**Second run:** with skip flags on, files already processed are skipped without re-inference. Use **Replace existing** on detect to force a rescan.

**Recognize again:** **Replace person tags** (default on) removes previous `person:p#` on each file in Hydrus before writing the latest set. **Re-cluster all** (default when incremental refine is off) clears local person links and re-groups every embedding in the DB.

**Clean DB:** `POST /api/face/clean` removes embedding rows whose file hash no longer exists in Hydrus. **Reset assignments** clears `person:*` links but keeps embeddings.

## Docker checklist

1. `mkdir -p face_data models/face logs ort_traces && sudo chown -R 1000:1000 face_data models logs ort_traces`
2. `./start.sh docker-run -d --build`
3. First detect downloads `buffalo_l` into `./models/face` (may take a minute).
4. Embeddings persist in `./face_data/`.

## Tuning

- Run **Recognize** after large detect batches; staged clustering reduces false new persons.
- Use **Reset assignments** to re-cluster without re-running detection.
- Use **Clean DB** after deleting files in Hydrus.
- Adjust `face_recognition_max_distance` if persons split/merge incorrectly (see hydrus-ai-taggers `diagnose_distances` workflow conceptually).

## Reference

Ported from [lannashelton/hydrus-ai-taggers](https://github.com/lannashelton/hydrus-ai-taggers) (InsightFace + SQLite + sklearn radius clustering).

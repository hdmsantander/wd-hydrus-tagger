# Face tagging (AI-assisted)

WD Hydrus Tagger integrates the two-step face pipeline from [hydrus-ai-taggers](https://github.com/lannashelton/hydrus-ai-taggers):

1. **Detect** — InsightFace `buffalo_l` finds faces, stores 512-d embeddings in SQLite, tags files with `ai face detected` or `face not visible`.
2. **Recognize** — Unsupervised clustering assigns `person:p1`, `person:p2`, … tags. Map real names in Hydrus via **tag siblings** (e.g. `person:p1` → `person:alice`).

## Install

Native (from repo root):

```bash
pip install -e ".[face]"
# Optional GPU (pick one ONNX GPU package — do not mix cuda + rocm wheels):
pip install -e ".[gpu]"          # NVIDIA CUDA
pip install -e ".[rocm]"         # AMD ROCm (Linux)
pip install -e ".[directml]"     # Windows AMD/Intel via DirectML
```

Docker: the official image installs `.[face]` automatically. Bind-mount `./face_data` for the embedding database (see `docker-compose.yml`).

## GPU backends (WD + face)

| `gpu_backend` | ONNX package | Hardware |
|---------------|--------------|----------|
| `auto` | whichever EP is installed | CUDA → ROCm → DirectML → CPU |
| `cuda` | `onnxruntime-gpu` | NVIDIA |
| `rocm` | `onnxruntime-rocm` | AMD Radeon (Linux + ROCm) |
| `directml` | `onnxruntime-directml` | Windows AMD/Intel |
| `cpu` | `onnxruntime` | CPU only |

Set in `config.yaml` or env **`WD_TAGGER_GPU_BACKEND=rocm`**. Enable **`use_gpu: true`**.

Verify installed providers: **`GET /api/face/providers`**.

### AMD ROCm (Linux)

1. Install ROCm drivers and `pip install -e ".[rocm,face]"`.
2. Set `use_gpu: true` and `gpu_backend: rocm`.
3. Docker: map `/dev/kfd` and `/dev/dri`, rebuild with rocm extra, set `WD_TAGGER_GPU_BACKEND=rocm` (see `docker-compose.yml` comments).

### Windows DirectML

1. `pip install -e ".[directml,face]"`.
2. `use_gpu: true`, `gpu_backend: directml`.

## Web UI workflow

1. Connect to Hydrus and search files.
2. Open the **Face tagging** sidebar panel.
3. **Detect faces** on selection or all search results (WebSocket progress).
4. **Recognize persons** — staged clustering + Hydrus apply.
5. In Hydrus, add tag siblings from `person:p#` to real character names.

Sliders:

- **Detection threshold** — minimum InsightFace det score (default 0.6).
- **Cluster max distance** — neighbour radius for person grouping (default 0.5, cosine).

## API

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/face/status` | DB + model stats |
| GET | `/api/face/providers` | Installed ONNX EPs |
| POST | `/api/face/models/load` | Load InsightFace |
| POST | `/api/face/detect` | HTTP batch detect |
| WS | `/api/face/ws/progress` | Batch detect with progress |
| POST | `/api/face/recognize` | Cluster + apply person tags |
| POST | `/api/face/reset` | Clear person assignments |
| POST | `/api/face/clean` | Remove orphans deleted in Hydrus |

## Configuration (`config.yaml`)

| Key | Default | Notes |
|-----|---------|--------|
| `face_models_dir` | `./models/face` | InsightFace model cache |
| `face_embeddings_db_path` | `./face_data/face_embeddings.db` | **Back up** this file |
| `face_target_tag_service` | `""` | Falls back to `target_tag_service` |
| `face_skip_if_detected` | `true` | Skip files already marked detected/not visible |
| `face_recognition_stages` | `[20,5,3,1]` | Staged `min_faces` for clustering |
| `face_video_frame_count` | `30` | Frames sampled per video |

Marker tags (configurable): `face_marker_detected`, `face_marker_not_visible`, `face_marker_recognized`, `face_person_tag_prefix`.

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

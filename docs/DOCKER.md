# Docker

## Files

- [`Dockerfile`](../Dockerfile) — Python 3.11-slim image; installs `requirements.txt` then `pip install .` (runtime only; no `[dev]` extras). Runs as non-root **`tagger` (UID/GID 1000)** for predictable bind-mount permissions. **HEALTHCHECK** probes **`GET http://127.0.0.1:8199/api/app/status`** via stdlib `urllib` (no extra packages).
- [`docker-compose.yml`](../docker-compose.yml) — service **`wd-tagger`** (port **8199**, bind-mounts **`./config.yaml`** and **`./config.example.yaml`** read-only; optional **`WD_TAGGER_CONFIG_PATH`**), shared bridge **`hydrus_net`**, **`restart: unless-stopped`**, **`extra_hosts`** for **`host.docker.internal`** (Linux host-gateway). Optional **`hydrus-web`** (profile **`hydrus-web`**) serves the [floogulinc/hydrus-web](https://github.com/floogulinc/hydrus-web) UI from **`ghcr.io/floogulinc/hydrus-web`** via [`docker/hydrus-web/Dockerfile`](../docker/hydrus-web/Dockerfile).
- [`docker-compose.amd.yml`](../docker-compose.amd.yml) — AMD overlay: **`GPU_BACKEND=rocm`**, `/dev/kfd` + `/dev/dri`, **`onnxruntime-migraphx`** wheel.
- [`.dockerignore`](../.dockerignore) — shrinks build context (excludes `.venv`, tests, mounted dirs, **`config.yaml`** so secrets are not copied into image layers).

## Services and ports

| Service     | Default host port | Container | Profile        |
|------------|-------------------|------------|----------------|
| `wd-tagger` | 8199              | 8199       | (default)      |
| `hydrus-web` | **8080** (`HYDRUS_WEB_PORT`) | 80 | `hydrus-web` |

Start tagger only:

```bash
docker compose up -d
```

Start tagger + hydrus-web SPA:

```bash
docker compose --profile hydrus-web up -d
```

## Tagger UI link to Hydrus web

When the **`hydrus-web`** profile exposes the SPA on the host (default **8080**), set **`hydrus_web_url: 'http://127.0.0.1:8080'`** in **`config.yaml`** or under **Settings → Hydrus web URL**. The tagger gallery toolbar and the full-screen image viewer then show **Hydrus web**, opening the companion app’s **`/pages`** route in a new tab.

Without editing YAML in Docker, pass **`HYDRUS_WEB_URL=http://127.0.0.1:8080`** (or your **`HYDRUS_WEB_PORT`**) in the **`wd-tagger`** service environment — see **`docker-compose.yml`**.

## Read-only config in Docker

Compose bind-mounts **`./config.yaml:/app/config.yaml:ro`**. **`PATCH /api/config`** and connection-test credential saves still update **in-memory** config for the running process; when the file is not writable the API returns **`persisted: false`** and a short **`warning`**. Edit **`config.yaml` on the host** to persist changes across restarts.

## Shell helpers

From the repo root:

```bash
./start.sh docker-run -d          # wd-tagger only
./start.sh docker-run-all -d      # wd-tagger + hydrus-web profile
```

Extra compose flags are forwarded (e.g. **`--build`**, **`--force-recreate`**).

## Automated smoke (host)

After **`docker compose --profile hydrus-web up -d`** (or **`./start.sh docker-run-all -d`**):

```bash
./scripts/docker_smoke.sh
```

Exits non-zero if tagger **`/api/app/status`** or hydrus-web **`/`** does not return HTTP 200.

## Prerequisites on the host

1. **`config.yaml`** must exist next to `docker-compose.yml` before `docker compose up`, because compose bind-mounts `./config.yaml:/app/config.yaml:ro`. Create it from `config.example.yaml` and set Hydrus URL/key.
2. In that file, keep **`models_dir: './models'`** (or `./models` under the app root). **Absolute host paths** resolve inside the container and typically **break startup** (the process cannot create directories on your host tree outside the bind mounts). If you still point `models_dir` at a host-only path that is missing in the container, the server **falls back to `<app>/models`** (`/app/models` in the official image) and logs a warning.
3. Directories **`models`**, **`logs`**, **`ort_traces`**, **`face_data`** are mounted from the host; they must be **writable by UID 1000** inside the container (the image runs as **`tagger` = `1000:1000`**). If you see **`PermissionError`** on log files, run once on the host (from the compose directory):

   ```bash
   mkdir -p models models/face face_data logs ort_traces
   sudo chown -R 1000:1000 models face_data logs ort_traces
   ```

   Face embeddings live at **`./face_data/face_embeddings.db`** (see **`face_embeddings_db_path`** in `config.example.yaml`). The Docker image installs **`.[face]`** (InsightFace + OpenCV headless + scikit-learn).

4. Optional GPU:
   - **NVIDIA:** build with **`GPU_BACKEND=cuda`**, set **`WD_TAGGER_USE_GPU=true`** / **`WD_TAGGER_GPU_BACKEND=cuda`**, and uncomment the NVIDIA `deploy.resources` block in **`docker-compose.yml`**.
   - **AMD ROCm (Linux):** use the overlay **[`docker-compose.amd.yml`](../docker-compose.amd.yml)**. **`./start.sh docker-run`** / **`docker-run-all`** include it automatically when **`/dev/kfd`** exists (set **`WD_TAGGER_DOCKER_AMD=0`** to skip). Manual:

     ```bash
     docker compose -f docker-compose.yml -f docker-compose.amd.yml --profile hydrus-web up -d --build
     ```

     The overlay rebuilds with **`onnxruntime-migraphx`** (AMD ROCm 7.2.x index) and maps **`/dev/kfd`** + **`/dev/dri`**. **Do not bind-mount Arch/Manjaro `/opt/rocm`** into this Debian image (those HIP libs need glibc 2.43+). **`GET /api/face/providers`** should list **`MIGraphXExecutionProvider`**. GPU kernels still need a matching ROCm userspace — on this host use a **native** run (`sudo pacman -S migraphx`, then `pip install onnxruntime-migraphx -f https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/` and `use_gpu: true`). See **[FACE_TAGGING.md](FACE_TAGGING.md)**.

## Ports and who talks to whom

| Port (host) | Service | Client | Reaches Hydrus how |
|-------------|---------|--------|-------------------|
| **8199** | `wd-tagger` API/UI | Your browser | N/A (tagger backend) |
| **8080** | `hydrus-web` SPA | Your browser | Browser → **`http://localhost:45869`** (host loopback) |
| **45869** | Hydrus Client API | Hydrus client + above | Listens on the **host** |

**Why hydrus-web works but the tagger did not:** hydrus-web runs in the browser on your PC, so `localhost:45869` is correct. The **tagger backend** runs inside Docker; its `localhost` is the container, not Hydrus.

**Fix (automatic):** `docker-compose.yml` sets **`HYDRUS_API_URL=http://host.docker.internal:45869`** and adds **`extra_hosts: host.docker.internal:host-gateway`**. The tagger also remaps `localhost` / `127.0.0.1` in `hydrus_api_url` when `/.dockerenv` is present. Keep **`hydrus_api_url: http://localhost:45869`** in `config.yaml` for native runs.

Override if needed: `HYDRUS_API_URL=http://192.168.1.10:45869 docker compose up -d`

## Hydrus API URL from inside the container

`hydrus_api_url: http://localhost:45869` in `config.yaml` is fine on the host. Inside **`wd-tagger`**, that URL would point at the container unless remapped.

The stack handles this by default:

- Compose env **`HYDRUS_API_URL`** (default **`http://host.docker.internal:45869`**)
- Runtime remap of loopback URLs when the process sees **`/.dockerenv`**
- **`extra_hosts: host.docker.internal:host-gateway`** on Linux (Docker 20.10+)

Manual alternatives:

- Your machine’s **LAN IP** (e.g. `http://192.168.1.10:45869`) if `host.docker.internal` is unavailable.
- If Hydrus runs in the same Compose stack, use that **service name** as the hostname instead.

**hydrus-web** is a static SPA in the browser: the **browser** talks to Hydrus (and needs CORS / access as in the [hydrus-web wiki](https://github.com/floogulinc/hydrus-web/wiki)). The tagger container talks to Hydrus using the **effective** `hydrus_api_url` after the overrides above.

**Verify from the container:**

```bash
docker compose exec wd-tagger python -c "import urllib.request; print(urllib.request.urlopen('http://host.docker.internal:45869/', timeout=5).status)"
curl -sf http://127.0.0.1:8199/api/config | jq -r '.config.hydrus_api_url'
```

Both should show **`http://host.docker.internal:45869`** (or your override), not `localhost`.

## Health checks

| Service      | Probe |
|-------------|--------|
| `wd-tagger` | `GET /api/app/status` (in-process metrics; no Hydrus round-trip) |
| `hydrus-web` | `GET /` on nginx port 80 (inside container) |

**Smoke without editing `config.yaml`:** the compose file mounts [`config.example.yaml`](../config.example.yaml) at **`/app/config.example.yaml`**. It uses a relative **`models_dir`** suitable for containers. One-shot:

```bash
WD_TAGGER_CONFIG_PATH=/app/config.example.yaml docker compose --profile hydrus-web up -d
```

Inspect tagger health after `up`:

```bash
docker compose ps
docker inspect --format='{{json .State.Health}}' "$(docker compose ps -q wd-tagger)"
```

From the host (mapped port):

```bash
curl -sf "http://127.0.0.1:8199/api/app/status" | head -c 200
```

With **`hydrus-web`** profile:

```bash
curl -sfI "http://127.0.0.1:${HYDRUS_WEB_PORT:-8080}/"
```

## Pinning the hydrus-web image

The wrapper image accepts build args (see [`docker/hydrus-web/Dockerfile`](../docker/hydrus-web/Dockerfile)). Example:

```bash
HYDRUS_WEB_TAG=dev docker compose --profile hydrus-web build hydrus-web --no-cache
```

To pin by digest, set **`HYDRUS_WEB_IMAGE`** to your registry mirror or use a **`FROM`** digest in a local fork of that Dockerfile, e.g. `FROM ghcr.io/floogulinc/hydrus-web@sha256:…`.

## Validate configuration (no build)

```bash
docker compose -f docker-compose.yml config
```

## Troubleshooting: `network … not found`

If **`docker compose up`** fails with:

```text
failed to set up container networking: network <id> not found
```

a container still references a **removed** `hydrus_net` bridge (common after `docker network prune`, Docker restarts, or partial `compose down`). The **`hydrus-web`** container is often the stale one.

**Fix** — tear down and recreate the stack (removes containers and the project network):

```bash
docker compose --profile hydrus-web down --remove-orphans
docker compose --profile hydrus-web up -d --build
```

Or:

```bash
./start.sh docker-down
./start.sh docker-run-all -d
```

Verify:

```bash
docker compose --profile hydrus-web ps
./scripts/docker_smoke.sh
```

## Build smoke test

```bash
docker compose build --no-cache wd-tagger
docker compose --profile hydrus-web build --no-cache hydrus-web
```

Runtime image does not install pytest; run the full test suite on the host with **`./wd-hydrus-tagger.sh test`** or **`pytest -m full`**.

Static regression tests (no daemon): **`tests/test_docker_artifacts.py`**, **`tests/test_hydrus_web_frontend.py`**, config override tests in **`tests/test_config.py`**.

## Pre-merge / operator checklist

1. **`cp config.example.yaml config.yaml`** and set a real **`hydrus_api_key`** (never commit `config.yaml`).
2. **`models_dir: './models'`**; **`mkdir -p models logs ort_traces`** and **`chown -R 1000:1000`** those dirs if bind mounts fail with permission errors.
3. **`./start.sh docker-run-all -d --build`**
4. **`./scripts/docker_smoke.sh`**
5. **`curl -sf http://127.0.0.1:8199/api/config`** — `hydrus_api_url` should be **`http://host.docker.internal:45869`** (not `localhost`) inside Docker.
6. **`curl -sf -X POST http://127.0.0.1:8199/api/connection/test -H 'Content-Type: application/json' -d '{}'`** — expect **`success: true`** when Hydrus is running on the host.
7. Open **http://127.0.0.1:8199** — gallery toolbar **Hydrus web** link when **`hydrus_web_url`** or **`HYDRUS_WEB_URL`** is set; hydrus-web SPA at **http://127.0.0.1:8080**.
8. Host CI: **`./wd-hydrus-tagger.sh test`** (full pytest).

## Python version note

`Dockerfile` uses **Python 3.11**; `pyproject.toml` declares **`requires-python = ">=3.10"`**. Application code should remain compatible with 3.10+ even when the container tracks 3.11.

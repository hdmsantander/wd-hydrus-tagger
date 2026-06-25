# Docker

## Files

- [`Dockerfile`](../Dockerfile) — Python 3.11-slim image; installs `requirements.txt` then `pip install .` (runtime only; no `[dev]` extras). Runs as non-root **`tagger` (UID/GID 1000)** for predictable bind-mount permissions. **HEALTHCHECK** probes **`GET http://127.0.0.1:8199/api/app/status`** via stdlib `urllib` (no extra packages).
- [`docker-compose.yml`](../docker-compose.yml) — service **`wd-tagger`** (port **8199**, bind-mounts **`./config.yaml`** and **`./config.example.yaml`** read-only; optional **`WD_TAGGER_CONFIG_PATH`**), shared bridge **`hydrus_net`**, **`restart: unless-stopped`**, **`extra_hosts`** for **`host.docker.internal`** (Linux host-gateway). Optional **`hydrus-web`** (profile **`hydrus-web`**) serves the [floogulinc/hydrus-web](https://github.com/floogulinc/hydrus-web) UI from **`ghcr.io/floogulinc/hydrus-web`** via [`docker/hydrus-web/Dockerfile`](../docker/hydrus-web/Dockerfile).
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
3. Directories **`models`**, **`logs`**, **`ort_traces`** are mounted from the host; they must be **writable by UID 1000** inside the container (the image runs as **`tagger` = `1000:1000`**). If you see **`PermissionError`** on log files, run once on the host (from the compose directory):

   ```bash
   mkdir -p models logs ort_traces
   sudo chown -R 1000:1000 models logs ort_traces
   ```

## Hydrus API URL from inside the container

`hydrus_api_url: http://localhost:45869` in `config.yaml` points at **the same network namespace as the process**. Inside **`wd-tagger`**, `localhost` is the container, **not** your host where Hydrus usually runs.

Use one of:

- **`http://host.docker.internal:45869`** — Docker Desktop provides this hostname; **`docker-compose.yml`** adds **`host.docker.internal:host-gateway`** for **Linux** (Docker 20.10+).
- Your machine’s **LAN IP** (e.g. `http://192.168.1.10:45869`) if `host.docker.internal` is unavailable.
- If you later run Hydrus in the same Compose stack, use that **service name** as the hostname instead.

**hydrus-web** is a static SPA in the browser: the **browser** talks to Hydrus (and needs CORS / access as in the [hydrus-web wiki](https://github.com/floogulinc/hydrus-web/wiki)). The tagger container talks to Hydrus using **`hydrus_api_url`** only.

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

## Build smoke test

```bash
docker compose build --no-cache wd-tagger
docker compose --profile hydrus-web build --no-cache hydrus-web
```

Runtime image does not install pytest; run the full test suite on the host with **`./wd-hydrus-tagger.sh test`** or **`pytest -m full`**.

## Python version note

`Dockerfile` uses **Python 3.11**; `pyproject.toml` declares **`requires-python = ">=3.10"`**. Application code should remain compatible with 3.10+ even when the container tracks 3.11.

"""FastAPI application factory."""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import load_config
from backend.hydrus.client import aclose_all_hydrus_clients
from backend.perf_metrics import log_process_shutdown, mark_process_start
from backend.routes.connection import router as connection_router
from backend.routes.files import router as files_router
from backend.routes.tagger import router as tagger_router
from backend.routes.config_routes import router as config_router
from backend.routes.app_control import router as app_control_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    log = logging.getLogger("backend.app")
    if not logging.root.handlers:
        from backend.logging_setup import configure_logging, DEFAULT_LOG_LEVEL

        repo_root = Path(__file__).resolve().parent.parent
        level = os.environ.get("LOG_LEVEL", os.environ.get("WD_TAGGER_LOG_LEVEL", DEFAULT_LOG_LEVEL))
        configure_logging(
            level,
            log_file=os.environ.get("WD_TAGGER_LOG_FILE"),
            repo_root=repo_root,
            reset=False,
        )
    config = load_config()
    models_dir = Path(config.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    log.info(
        "Application ready host=%s port=%s run_id=%s log_file=%s",
        config.host,
        config.port,
        os.environ.get("WD_TAGGER_RUN_ID", "—"),
        os.environ.get("WD_TAGGER_LOG_FILE", "—"),
    )
    mark_process_start()
    yield
    log.info("lifespan: shutdown starting (Ctrl+C / SIGTERM / stop after UI shutdown)")
    try:
        from backend.shutdown_coordination import run_coordinated_tagging_shutdown

        await run_coordinated_tagging_shutdown(reason="lifespan_exit")
    except Exception:
        log.exception("lifespan: coordinated tagging shutdown failed; continuing with client close")
    await aclose_all_hydrus_clients()
    log.info("lifespan: Hydrus HTTP clients closed")
    log_process_shutdown()


app = FastAPI(title="WD Hydrus Tagger", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(connection_router, prefix="/api/connection", tags=["connection"])
app.include_router(files_router, prefix="/api/files", tags=["files"])
app.include_router(tagger_router, prefix="/api/tagger", tags=["tagger"])
app.include_router(config_router, prefix="/api/config", tags=["config"])
app.include_router(app_control_router, prefix="/api/app", tags=["app"])

frontend_dir = Path(__file__).parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


def main():
    """Console script entry (``wd-hydrus-tagger``); mirrors ``run.py``."""
    from backend.logging_setup import configure_logging, parse_server_args

    repo_root = Path(__file__).resolve().parent.parent
    args = parse_server_args(sys.argv[1:])
    os.environ["LOG_LEVEL"] = args.log_level.strip()
    os.environ["WD_TAGGER_LOG_LEVEL"] = args.log_level.strip()
    log_path = configure_logging(
        args.log_level,
        log_file=args.log_file,
        repo_root=repo_root,
        reset=True,
    )
    import uvicorn

    from backend.listen_hints import log_startup_listen_hint, print_startup_listen_hint
    from backend.runtime_linux import uvicorn_loop_setting

    config = load_config()
    boot = logging.getLogger("wd_tagger.bootstrap")
    boot.info(
        "startup run_id=%s log_file=%s",
        os.environ.get("WD_TAGGER_RUN_ID", "—"),
        log_path,
    )
    log_startup_listen_hint(boot, config.host, config.port)
    print_startup_listen_hint(config.host, config.port, stream=sys.stderr)
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=args.log_level.lower(),
        loop=uvicorn_loop_setting(),
    )

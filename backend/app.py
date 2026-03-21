"""FastAPI application factory."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import load_config
from backend.routes.connection import router as connection_router
from backend.routes.files import router as files_router
from backend.routes.tagger import router as tagger_router
from backend.routes.config_routes import router as config_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    models_dir = Path(config.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    yield


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

frontend_dir = Path(__file__).parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


def main():
    import uvicorn
    config = load_config()
    uvicorn.run(app, host=config.host, port=config.port)

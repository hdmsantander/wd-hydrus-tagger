"""File browsing endpoints."""

import logging
import time

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from backend.config import clamp_hydrus_metadata_chunk_size, get_config
from backend.log_stats import log_stats
from backend.hydrus.client import HydrusClient

router = APIRouter()
log = logging.getLogger(__name__)


class FileSearchRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    tags: list[str] = Field(default_factory=lambda: ["system:archive"])
    file_sort_type: int | None = None
    file_sort_asc: bool | None = None


class MetadataRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    file_ids: list = Field(default_factory=list)


def _get_client() -> HydrusClient:
    config = get_config()
    return HydrusClient(config.hydrus_api_url, config.hydrus_api_key)


@router.post("/search")
async def search_files(body: FileSearchRequest, client: HydrusClient = Depends(_get_client)):
    """Search for files in Hydrus."""
    tags = body.tags
    file_sort_type = body.file_sort_type
    file_sort_asc = body.file_sort_asc
    try:
        file_ids = await client.search_files(
            tags=tags,
            file_sort_type=file_sort_type,
            file_sort_asc=file_sort_asc,
        )
        return {"success": True, "file_ids": file_ids, "count": len(file_ids)}
    except Exception as e:
        log.warning("Hydrus search_files failed: %s", e)
        return {"success": False, "error": str(e)}


@router.post("/metadata")
async def get_metadata(body: MetadataRequest, client: HydrusClient = Depends(_get_client)):
    """Get metadata for file IDs (chunked for large lists — see ``hydrus_metadata_chunk_size``)."""
    file_ids = body.file_ids
    if not isinstance(file_ids, list):
        return {"success": False, "error": "file_ids must be a list"}
    try:
        t0 = time.perf_counter()
        cfg = get_config()
        chunk = clamp_hydrus_metadata_chunk_size(cfg.hydrus_metadata_chunk_size)
        metadata: list[dict] = []
        for i in range(0, len(file_ids), chunk):
            part = [int(x) for x in file_ids[i : i + chunk]]
            if not part:
                continue
            rows = await client.get_file_metadata(file_ids=part)
            metadata.extend(rows)
        n_chunks = max(1, (len(file_ids) + chunk - 1) // chunk) if file_ids else 0
        wall_ms = (time.perf_counter() - t0) * 1000.0
        log.debug(
            "files metadata_hydrus file_ids=%s chunk_size=%s chunks=%s rows_returned=%s",
            len(file_ids),
            chunk,
            n_chunks,
            len(metadata),
        )
        log_stats(
            log,
            "hydrus_metadata_fetch",
            duration_ms=round(wall_ms, 3),
            file_ids=len(file_ids),
            chunk_size=chunk,
            chunks=n_chunks,
            rows=len(metadata),
        )
        return {"success": True, "metadata": metadata}
    except (TypeError, ValueError) as e:
        return {"success": False, "error": f"invalid file_ids: {e}"}
    except Exception as e:
        log.warning("Hydrus get_file_metadata failed: %s", e)
        return {"success": False, "error": str(e)}


@router.get("/{file_id}/thumbnail")
async def get_thumbnail(file_id: int, client: HydrusClient = Depends(_get_client)):
    """Proxy thumbnail from Hydrus."""
    try:
        data, content_type = await client.get_thumbnail(file_id=file_id)
        return Response(content=data, media_type=content_type)
    except Exception as e:
        return JSONResponse(
            status_code=502,
            content={"success": False, "error": str(e)},
        )


@router.get("/{file_id}")
async def get_file(file_id: int, client: HydrusClient = Depends(_get_client)):
    """Proxy full file from Hydrus."""
    try:
        data, content_type = await client.get_file(file_id=file_id)
        return Response(content=data, media_type=content_type)
    except Exception as e:
        return JSONResponse(
            status_code=502,
            content={"success": False, "error": str(e)},
        )

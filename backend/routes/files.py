"""File browsing endpoints."""

import logging

from fastapi import APIRouter, Query
from fastapi.responses import Response

from backend.config import get_config
from backend.hydrus.client import HydrusClient

router = APIRouter()
log = logging.getLogger(__name__)


def _get_client() -> HydrusClient:
    config = get_config()
    return HydrusClient(config.hydrus_api_url, config.hydrus_api_key)


@router.post("/search")
async def search_files(body: dict):
    """Search for files in Hydrus."""
    client = _get_client()
    tags = body.get("tags", ["system:archive"])
    file_sort_type = body.get("file_sort_type", None)
    file_sort_asc = body.get("file_sort_asc", None)
    try:
        file_ids = await client.search_files(
            tags=tags,
            file_sort_type=file_sort_type,
            file_sort_asc=file_sort_asc,
        )
        return {"success": True, "file_ids": file_ids, "count": len(file_ids)}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/metadata")
async def get_metadata(body: dict):
    """Get metadata for file IDs (chunked for large lists — see ``hydrus_metadata_chunk_size``)."""
    client = _get_client()
    file_ids = body.get("file_ids", [])
    if not isinstance(file_ids, list):
        return {"success": False, "error": "file_ids must be a list"}
    try:
        cfg = get_config()
        chunk = max(32, min(2048, int(cfg.hydrus_metadata_chunk_size)))
        metadata: list[dict] = []
        for i in range(0, len(file_ids), chunk):
            part = [int(x) for x in file_ids[i : i + chunk]]
            if not part:
                continue
            rows = await client.get_file_metadata(file_ids=part)
            metadata.extend(rows)
        n_chunks = max(1, (len(file_ids) + chunk - 1) // chunk) if file_ids else 0
        log.info(
            "files metadata_hydrus file_ids=%s chunk_size=%s chunks=%s rows_returned=%s",
            len(file_ids),
            chunk,
            n_chunks,
            len(metadata),
        )
        return {"success": True, "metadata": metadata}
    except (TypeError, ValueError) as e:
        return {"success": False, "error": f"invalid file_ids: {e}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/{file_id}/thumbnail")
async def get_thumbnail(file_id: int):
    """Proxy thumbnail from Hydrus."""
    client = _get_client()
    try:
        data, content_type = await client.get_thumbnail(file_id=file_id)
        return Response(content=data, media_type=content_type)
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/{file_id}")
async def get_file(file_id: int):
    """Proxy full file from Hydrus."""
    client = _get_client()
    try:
        data, content_type = await client.get_file(file_id=file_id)
        return Response(content=data, media_type=content_type)
    except Exception as e:
        return {"success": False, "error": str(e)}

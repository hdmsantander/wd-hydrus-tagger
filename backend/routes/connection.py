"""Connection management endpoints."""

import logging

from fastapi import APIRouter

from backend.config import get_config, save_config
from backend.hydrus.client import HydrusClient

router = APIRouter()
log = logging.getLogger(__name__)


@router.post("/test")
async def test_connection(body: dict | None = None):
    """Test connection to Hydrus Network."""
    config = get_config()
    if body:
        raw_url = body.get("url")
        if isinstance(raw_url, str) and raw_url.strip():
            url = raw_url.strip()
        else:
            url = config.hydrus_api_url
        raw_key = body.get("api_key")
        if isinstance(raw_key, str) and raw_key.strip():
            api_key = raw_key.strip()
        else:
            api_key = config.hydrus_api_key
    else:
        url = config.hydrus_api_url
        api_key = config.hydrus_api_key

    if not api_key:
        return {"success": False, "error": "API key is required"}

    client = HydrusClient(url, api_key)
    try:
        result = await client.verify_access_key()
        # Save working credentials to config
        config.hydrus_api_url = url
        config.hydrus_api_key = api_key
        save_config(config)
        log.info("Hydrus API access verified url=%s", url)
        return {"success": True, "result": result}
    except Exception as e:
        log.warning("Hydrus connection test failed url=%s: %s", url, e)
        return {"success": False, "error": str(e)}


@router.get("/services")
async def get_services():
    """List Hydrus tag services."""
    config = get_config()
    if not config.hydrus_api_key:
        return {"success": False, "error": "Not connected"}

    client = HydrusClient(config.hydrus_api_url, config.hydrus_api_key)
    try:
        services = await client.get_services()
        n = len(services) if isinstance(services, list) else 0
        log.debug("Hydrus tag services listed count=%s", n)
        return {"success": True, "services": services}
    except Exception as e:
        log.warning("Hydrus get_services failed: %s", e)
        return {"success": False, "error": str(e)}

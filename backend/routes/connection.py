"""Connection management endpoints."""

from fastapi import APIRouter, Depends

from backend.config import AppConfig, get_config, save_config
from backend.hydrus.client import HydrusClient

router = APIRouter()


@router.post("/test")
async def test_connection(body: dict | None = None):
    """Test connection to Hydrus Network."""
    config = get_config()
    url = body.get("url", config.hydrus_api_url) if body else config.hydrus_api_url
    api_key = body.get("api_key", config.hydrus_api_key) if body else config.hydrus_api_key

    if not api_key:
        return {"success": False, "error": "API key is required"}

    client = HydrusClient(url, api_key)
    try:
        result = await client.verify_access_key()
        # Save working credentials to config
        config.hydrus_api_url = url
        config.hydrus_api_key = api_key
        save_config(config)
        return {"success": True, "result": result}
    except Exception as e:
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
        return {"success": True, "services": services}
    except Exception as e:
        return {"success": False, "error": str(e)}

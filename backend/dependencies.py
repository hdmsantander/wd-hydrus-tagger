"""FastAPI dependency injection."""

from backend.config import AppConfig, get_config
from backend.hydrus.client import HydrusClient

_hydrus_client: HydrusClient | None = None


def get_app_config() -> AppConfig:
    return get_config()


def get_hydrus_client() -> HydrusClient:
    global _hydrus_client
    config = get_config()
    if _hydrus_client is None or _hydrus_client.api_url != config.hydrus_api_url or _hydrus_client.access_key != config.hydrus_api_key:
        _hydrus_client = HydrusClient(config.hydrus_api_url, config.hydrus_api_key)
    return _hydrus_client

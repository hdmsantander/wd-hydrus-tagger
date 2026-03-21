"""Application configuration loaded from YAML."""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, field_validator


class AppConfig(BaseModel):
    hydrus_api_url: str = "http://localhost:45869"
    hydrus_api_key: str = ""

    default_model: str = "wd-vit-tagger-v3"
    models_dir: str = "./models"
    use_gpu: bool = False

    general_threshold: float = 0.35
    character_threshold: float = 0.85

    target_tag_service: str = "my tags"

    general_tag_prefix: str = ""
    character_tag_prefix: str = "character:"
    rating_tag_prefix: str = "rating:"

    @field_validator(
        "target_tag_service", "general_tag_prefix",
        "character_tag_prefix", "rating_tag_prefix",
        mode="before",
    )
    @classmethod
    def coerce_none_to_str(cls, v, info):
        """Convert YAML null values to string defaults."""
        defaults = {
            "target_tag_service": "my tags",
            "general_tag_prefix": "",
            "character_tag_prefix": "character:",
            "rating_tag_prefix": "rating:",
        }
        if v is None:
            return defaults.get(info.field_name, "")
        return v

    batch_size: int = 4

    host: str = "127.0.0.1"
    port: int = 8199


CONFIG_PATH = Path("config.yaml")
EXAMPLE_CONFIG_PATH = Path("config.example.yaml")

_config: Optional[AppConfig] = None


def load_config() -> AppConfig:
    global _config
    if _config is not None:
        return _config

    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    elif EXAMPLE_CONFIG_PATH.exists():
        with open(EXAMPLE_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    _config = AppConfig(**data)
    return _config


def save_config(config: AppConfig) -> None:
    global _config
    _config = config
    data = config.model_dump()
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def get_config() -> AppConfig:
    global _config
    if _config is None:
        return load_config()
    return _config

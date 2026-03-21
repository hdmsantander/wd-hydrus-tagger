"""Pydantic models for Hydrus API data."""

from pydantic import BaseModel


class ServiceInfo(BaseModel):
    service_key: str
    name: str
    type: int
    type_pretty: str


class FileMetadata(BaseModel):
    file_id: int
    hash: str
    size: int | None = None
    mime: str | None = None
    width: int | None = None
    height: int | None = None
    has_audio: bool | None = None
    ext: str | None = None

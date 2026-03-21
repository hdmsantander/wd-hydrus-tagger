"""Pydantic models for tagger results."""

from pydantic import BaseModel


class TagResult(BaseModel):
    file_id: int
    hash: str
    general_tags: dict[str, float]    # tag_name -> confidence
    character_tags: dict[str, float]
    rating_tags: dict[str, float]
    formatted_tags: list[str]         # Final tags with prefixes, ready to apply

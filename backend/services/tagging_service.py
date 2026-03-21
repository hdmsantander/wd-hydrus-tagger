"""Orchestrates the batch tagging workflow."""

from io import BytesIO
from pathlib import Path

from PIL import Image

from backend.config import AppConfig
from backend.hydrus.client import HydrusClient
from backend.services.model_manager import ModelManager
from backend.tagger.engine import TaggerEngine


class TaggingService:
    _instance: "TaggingService | None" = None

    def __init__(self, config: AppConfig):
        self.config = config
        self.engine = TaggerEngine(use_gpu=config.use_gpu)
        self.model_manager = ModelManager(config.models_dir)
        self._loaded_model: str | None = None

    @classmethod
    def get_instance(cls, config: AppConfig) -> "TaggingService":
        if cls._instance is None:
            cls._instance = cls(config)
            return cls._instance

        # If use_gpu changed, recreate the engine so the new provider takes effect
        if cls._instance.config.use_gpu != config.use_gpu:
            cls._instance.engine = TaggerEngine(use_gpu=config.use_gpu)
            cls._instance._loaded_model = None  # Force model reload with new provider

        cls._instance.config = config
        return cls._instance

    def load_model(self, name: str) -> None:
        """Download if needed and load model."""
        if self._loaded_model == name and self.engine.use_gpu == self.config.use_gpu:
            return

        if not self.model_manager.is_downloaded(name):
            self.model_manager.download_model(name)

        model_path = self.model_manager.get_model_path(name)
        self.engine.load(Path(self.config.models_dir), name)
        self._loaded_model = name

    async def tag_files(
        self,
        client: HydrusClient,
        file_ids: list[int],
        general_threshold: float = 0.35,
        character_threshold: float = 0.85,
    ) -> list[dict]:
        """Tag a list of files from Hydrus.

        Returns list of dicts with file_id, hash, tags, and formatted_tags.
        """
        # Ensure model is loaded
        if self._loaded_model is None:
            self.load_model(self.config.default_model)

        # Fetch metadata to get hashes
        metadata_list = await client.get_file_metadata(file_ids=file_ids)

        results = []
        batch_size = self.config.batch_size

        for batch_start in range(0, len(file_ids), batch_size):
            batch_ids = file_ids[batch_start:batch_start + batch_size]
            batch_meta = metadata_list[batch_start:batch_start + batch_size]

            # Download images
            images = []
            valid_meta = []
            for fid, meta in zip(batch_ids, batch_meta):
                try:
                    file_data, _ = await client.get_file(file_id=fid)
                    img = Image.open(BytesIO(file_data))
                    images.append(img)
                    valid_meta.append(meta)
                except Exception:
                    continue

            if not images:
                continue

            # Run inference
            import asyncio
            predictions = await asyncio.to_thread(
                self.engine.predict,
                images,
                general_threshold,
                character_threshold,
            )

            # Format results
            for meta, pred in zip(valid_meta, predictions):
                formatted_tags = self._format_tags(pred)
                results.append({
                    "file_id": meta.get("file_id", 0),
                    "hash": meta.get("hash", ""),
                    "general_tags": pred["general_tags"],
                    "character_tags": pred["character_tags"],
                    "rating_tags": pred["rating_tags"],
                    "formatted_tags": formatted_tags,
                })

        return results

    def _format_tags(self, prediction: dict) -> list[str]:
        """Format tags with configured prefixes."""
        tags = []

        prefix = self.config.general_tag_prefix
        for tag in prediction["general_tags"]:
            formatted = tag.replace("_", " ")
            tags.append(f"{prefix}{formatted}" if prefix else formatted)

        prefix = self.config.character_tag_prefix
        for tag in prediction["character_tags"]:
            formatted = tag.replace("_", " ")
            tags.append(f"{prefix}{formatted}")

        # Rating: take the highest confidence one
        prefix = self.config.rating_tag_prefix
        if prediction["rating_tags"]:
            top_rating = max(prediction["rating_tags"], key=prediction["rating_tags"].get)
            tags.append(f"{prefix}{top_rating}")

        return tags

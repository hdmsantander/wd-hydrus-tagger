"""Model download and cache management."""

from pathlib import Path

from huggingface_hub import hf_hub_download

SUPPORTED_MODELS = {
    "wd-vit-tagger-v3": "SmilingWolf/wd-vit-tagger-v3",
    "wd-swinv2-tagger-v3": "SmilingWolf/wd-swinv2-tagger-v3",
    "wd-vit-large-tagger-v3": "SmilingWolf/wd-vit-large-tagger-v3",
    "wd-eva02-large-tagger-v3": "SmilingWolf/wd-eva02-large-tagger-v3",
}

REQUIRED_FILES = ["model.onnx", "selected_tags.csv"]


class ModelManager:
    def __init__(self, models_dir: str | Path):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def list_models(self) -> list[dict]:
        """List all supported models and their download status."""
        result = []
        for name, repo in SUPPORTED_MODELS.items():
            model_dir = self.models_dir / name
            downloaded = all(
                (model_dir / f).exists() for f in REQUIRED_FILES
            )
            result.append({
                "name": name,
                "repo": repo,
                "downloaded": downloaded,
                "path": str(model_dir) if downloaded else None,
            })
        return result

    def is_downloaded(self, name: str) -> bool:
        if name not in SUPPORTED_MODELS:
            return False
        model_dir = self.models_dir / name
        return all((model_dir / f).exists() for f in REQUIRED_FILES)

    def download_model(self, name: str) -> Path:
        """Download model files from HuggingFace Hub."""
        if name not in SUPPORTED_MODELS:
            raise ValueError(f"Unknown model: {name}. Available: {list(SUPPORTED_MODELS.keys())}")

        repo = SUPPORTED_MODELS[name]
        model_dir = self.models_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)

        for filename in REQUIRED_FILES:
            dest = model_dir / filename
            if not dest.exists():
                downloaded_path = hf_hub_download(
                    repo_id=repo,
                    filename=filename,
                    local_dir=str(model_dir),
                )

        return model_dir

    def get_model_path(self, name: str) -> Path:
        model_dir = self.models_dir / name
        if not self.is_downloaded(name):
            raise FileNotFoundError(f"Model {name} not downloaded")
        return model_dir

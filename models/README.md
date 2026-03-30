# Model cache directory

ONNX weights and `selected_tags.csv` for each WD Tagger v3 model are stored here after download from [SmilingWolf’s Hugging Face repos](https://huggingface.co/SmilingWolf).

- **Path:** Set `models_dir` in `config.yaml` (default `./models`). It is resolved **relative to the project root** (the folder that contains `backend/`), not the current shell working directory.
- **Reuse:** Files are only fetched when missing or when a load-time integrity check fails. Each model folder may contain `.wd_model_cache.json` recording the Hugging Face `main` revision and file sizes.
- **Checks:** Use **Settings → Verify cached models** (optional: compare with Hugging Face) to validate files without starting a tagging run.

This directory is gitignored except for this file.

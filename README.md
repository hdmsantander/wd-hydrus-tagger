[繁體中文](README.zh-TW.md)

# WD Tagger for Hydrus

A web tool that automatically generates tags for images in Hydrus Network using WD14 Tagger v3.

---

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Hydrus Network Setup](#hydrus-network-setup)
- [Configuration](#configuration)
- [Starting the Server](#starting-the-server)
- [Workflow](#workflow)
  - [Step 1: Connect to Hydrus](#step-1-connect-to-hydrus)
  - [Step 2: Search for Images](#step-2-search-for-images)
  - [Step 3: Select Images](#step-3-select-images)
  - [Step 4: Run Auto-Tagging](#step-4-run-auto-tagging)
  - [Step 5: Review & Edit Results](#step-5-review--edit-results)
  - [Step 6: Apply Tags to Hydrus](#step-6-apply-tags-to-hydrus)
- [Settings Panel](#settings-panel)
  - [Model Management](#model-management)
  - [Tag Prefix Settings](#tag-prefix-settings)
  - [GPU Acceleration](#gpu-acceleration)
- [Available Models](#available-models)
- [Threshold Tuning Guide](#threshold-tuning-guide)
- [Configuration Reference](#configuration-reference)
- [FAQ](#faq)
- [Project Structure](#project-structure)

---

## System Requirements

| Item | Requirement |
|------|-------------|
| Python | 3.10+ |
| Hydrus Network | Any recent version (Client API must be enabled) |
| OS | Windows / Linux / macOS |
| Disk Space | ~300–600 MB per model |
| RAM | 4 GB+ recommended (8 GB+ for Large models) |
| GPU (optional) | NVIDIA GPU with CUDA support |

---

## Installation

### 1. Install Python Dependencies

```bash
# CPU version (default)
pip install -r requirements.txt

# GPU version: onnxruntime-gpu includes CPU support
# Note: CPU and GPU versions cannot coexist — remove CPU version first
pip install -r requirements.txt --ignore-requires-python
pip uninstall -y onnxruntime
pip install onnxruntime-gpu
```

### 2. Create Configuration File

```bash
# Copy the example config
cp config.example.yaml config.yaml
```

Then edit `config.yaml` and fill in your Hydrus API Key (see "Hydrus Network Setup" below).

---

## Hydrus Network Setup

Before using this tool, you need to enable the Client API in Hydrus Network and obtain an API Key.

### Enable Client API

1. Open Hydrus Client
2. Go to **services → manage services**
3. Ensure the **client api** service is enabled (default port 45869)

### Obtain API Key

1. Go to **services → review services → client api**
2. Click **add → from api request**, or manually create a new access key
3. Name the key (e.g., `WD Tagger`)
4. Grant the following permissions:
   - **search and fetch files**
   - **edit file tags**
5. Click **apply** to get a 64-character hex key
6. Paste this key into the `hydrus_api_key` field in `config.yaml`

### (Recommended) Create a Dedicated Tag Service

To separate AI-generated tags from manual tags, consider creating a dedicated local tag service:

1. Go to **services → manage services**
2. Add a new **local tag service**, e.g., `AI Tags`
3. Set `target_tag_service` to `"AI Tags"` in `config.yaml`

---

## Configuration

The configuration file is `config.yaml` (YAML format). Copy from `config.example.yaml` on first use:

```yaml
# Hydrus Network connection
hydrus_api_url: "http://localhost:45869"
hydrus_api_key: "your-64-character-api-key"

# WD Tagger settings
default_model: "wd-vit-tagger-v3"    # Default model
models_dir: "./models"                # Model storage directory
use_gpu: false                        # Enable GPU acceleration

# Tag thresholds (0.0 - 1.0)
general_threshold: 0.35               # General tag confidence threshold
character_threshold: 0.85             # Character tag confidence threshold

# Hydrus tag service name
target_tag_service: "my tags"

# Tag prefixes
general_tag_prefix: ""                # General tag prefix (empty = no prefix)
character_tag_prefix: "character:"    # Character tag prefix
rating_tag_prefix: "rating:"          # Rating tag prefix

# Batch processing
batch_size: 4                         # Images per batch

# Web UI server
host: "127.0.0.1"
port: 8199
```

---

## Starting the Server

```bash
python run.py
```

You should see:

```
INFO:     Uvicorn running on http://127.0.0.1:8199 (Press CTRL+C to quit)
```

Open your browser and navigate to **http://127.0.0.1:8199**.

---

## Workflow

### Step 1: Connect to Hydrus

1. Open your browser and go to `http://127.0.0.1:8199`
2. In the "Hydrus Connection" panel on the left sidebar:
   - **API URL**: Enter the Hydrus Client API address (default `http://localhost:45869`)
   - **API Key**: Enter your API key
3. Click the "Connect" button
4. On success, the status indicator changes from **red** to **green**
5. Connection info is saved in the browser's localStorage for automatic reconnection

### Step 2: Search for Images

After connecting, the "Search Images" panel appears:

1. Enter search terms in the search box
2. Separate multiple terms with commas
3. Click "Search"

**Common search syntax examples:**

| Syntax | Purpose |
|--------|---------|
| `system:archive` | Archived files |
| `system:inbox` | Inbox files |
| `system:filetype is image` | Image files only |
| `system:archive, system:filetype is image` | Archived images |
| `system:archive, -character:hatsune_miku` | Archived, excluding a character |
| `system:filesize < 10MB` | Files smaller than 10MB |

The number of found images is displayed below the search bar.

### Step 3: Select Images

Search results are displayed as a thumbnail grid:

- **Click**: Select / deselect a single image
- **Ctrl + Click**: Multi-select (toggle individual images)
- **Shift + Click**: Range select (select all images between two clicks)
- **"Select All" button**: Select all images on the current page
- **"Deselect All" button**: Clear all selections

Selected images are highlighted with a purple border and a checkmark.

Pagination is available at the bottom, showing 50 images per page.

### Step 4: Run Auto-Tagging

In the "Tagger Settings" panel on the left sidebar:

1. **Model**: Choose a model from the dropdown (automatically downloaded on first use)
2. **General Threshold**: Adjust the confidence threshold for general tags (default 0.35)
3. **Character Threshold**: Adjust the confidence threshold for character tags (default 0.85)
4. **Tag Service**: Select the target Hydrus tag service
5. Click one of:
   - **"Tag Selected Images (N)"**: Process only selected images
   - **"Tag All Search Results"**: Process all results (confirmation dialog for 100+ images)

A progress bar shows current progress and percentage.

> **First-time note**: Model files are ~300–600 MB and will be downloaded from HuggingFace on first load.

### Step 5: Review & Edit Results

After tagging completes, the results view shows:

For each image:
- **Thumbnail**
- **General tags** (blue): Descriptive tags like `1girl`, `blue hair`, `outdoors`
- **Character tags** (purple): Identified characters like `hatsune miku`
- **Rating tags** (orange): Ratings like `general`, `sensitive`

Each tag shows its confidence percentage.

**Editing:**
- Click the **×** button on any tag to remove it
- Removed tags will not be applied to Hydrus

### Step 6: Apply Tags to Hydrus

After reviewing:

1. Click "Apply All Tags to Hydrus"
2. Tags are written to the Tag Service selected in Step 4
3. A success message is displayed

Tags are written with configured prefixes:
- General tags: `blue hair` (or with custom prefix)
- Character tags: `character:hatsune miku`
- Rating tags: `rating:general`

Click "Back to Gallery" to return and process more images.

---

## Settings Panel

Click the **gear icon (⚙)** in the top right to open the settings panel.

### Model Management

Shows all available models and their download status:

- **Downloaded**: Green "Downloaded" text
- **Not downloaded**: "Download" button to fetch from HuggingFace

### Tag Prefix Settings

| Field | Description | Default |
|-------|-------------|---------|
| General Prefix | Text prepended to general tags | (empty, no prefix) |
| Character Prefix | Text prepended to character tags | `character:` |
| Rating Prefix | Text prepended to rating tags | `rating:` |

**Custom examples:**
- Add a namespace to all tags: Set General Prefix to `wd:` → produces `wd:blue hair`
- Remove character prefix: Clear Character Prefix → produces `hatsune miku`

### GPU Acceleration

Check "Use GPU Acceleration" to enable CUDA GPU inference. Requires:
- NVIDIA GPU
- `onnxruntime-gpu` installed (`pip install onnxruntime-gpu`)

---

## Available Models

All models are from [SmilingWolf](https://huggingface.co/SmilingWolf)'s WD Tagger v3 series.

| Model | Size | Description |
|-------|------|-------------|
| **WD ViT v3** | ~300 MB | Base ViT model, balanced speed & quality, **recommended for general use** |
| **WD SwinV2 v3** | ~300 MB | SwinTransformer V2, similar quality to ViT |
| **WD ViT Large v3** | ~600 MB | Large ViT model, higher accuracy but slower |
| **WD EVA02 Large v3** | ~600 MB | EVA02 large model, highest accuracy |

**Recommendations:**
- General use, batch processing → **WD ViT v3** or **WD SwinV2 v3**
- Maximum accuracy → **WD EVA02 Large v3**

---

## Threshold Tuning Guide

Thresholds determine the minimum confidence level for keeping tags.

### General Threshold (default 0.35)

| Range | Effect |
|-------|--------|
| **0.20–0.30** | Loose — more tags, may include inaccurate ones |
| **0.35–0.45** | Balanced — suitable for most use cases |
| **0.50–0.70** | Strict — only high-confidence tags |
| **0.70+** | Very strict — only the most obvious features |

### Character Threshold (default 0.85)

Higher thresholds are recommended for character recognition to avoid false positives:

| Range | Effect |
|-------|--------|
| **0.70–0.80** | Loose — may produce false character matches |
| **0.85–0.90** | Balanced — suitable for most cases |
| **0.90+** | Only highly confident character identifications |

---

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hydrus_api_url` | string | `http://localhost:45869` | Hydrus Client API address |
| `hydrus_api_key` | string | `""` | 64-character API key |
| `default_model` | string | `wd-vit-tagger-v3` | Default model name |
| `models_dir` | string | `./models` | Model file storage directory |
| `use_gpu` | bool | `false` | Enable CUDA GPU |
| `general_threshold` | float | `0.35` | General tag threshold |
| `character_threshold` | float | `0.85` | Character tag threshold |
| `target_tag_service` | string | `my tags` | Default tag service |
| `general_tag_prefix` | string | `""` | General tag prefix |
| `character_tag_prefix` | string | `character:` | Character tag prefix |
| `rating_tag_prefix` | string | `rating:` | Rating tag prefix |
| `batch_size` | int | `4` | Images per inference batch |
| `host` | string | `127.0.0.1` | Web UI bind address |
| `port` | int | `8199` | Web UI port |

---

## FAQ

### Q: Connection failed, showing red indicator

- Ensure Hydrus Client is running
- Ensure Client API service is enabled (services → manage services)
- Verify the API URL and port are correct (default 45869)
- Verify the API Key is correct and has sufficient permissions

### Q: Model download is slow or fails

- Models are downloaded from HuggingFace and require a stable internet connection
- Large models are ~600 MB, please be patient
- If download fails, manually download `model.onnx` and `selected_tags.csv` from HuggingFace and place them in `models/{model-name}/`
- Manual download URL example: `https://huggingface.co/SmilingWolf/wd-vit-tagger-v3`

### Q: Tagging is slow

- Use smaller models (ViT v3 or SwinV2 v3)
- Reduce `batch_size` (lower memory pressure)
- Enable GPU acceleration (requires NVIDIA GPU + `onnxruntime-gpu`)

### Q: Too many / too few tags

- Adjust the General threshold: higher = fewer but more accurate tags, lower = more but possibly inaccurate tags
- You can manually remove unwanted tags before applying

### Q: Character recognition is inaccurate

- Increase Character threshold (e.g., 0.90+)
- Use a Large model for better accuracy
- Note: The model can only recognize characters present in its training data (primarily from Danbooru)

### Q: How to process only specific file types?

Add file type filters in the search box:
- `system:archive, system:filetype is image` — images only
- `system:archive, system:filetype is png` — PNG only
- `system:archive, system:width > 512` — width greater than 512

### Q: How are underscores in tags handled?

The tool automatically converts underscores `_` to spaces. For example, `blue_hair` becomes `blue hair`.

### Q: Can I use this remotely?

Change `host` in `config.yaml` to `0.0.0.0` to accept external connections. However:
- This tool has no built-in authentication
- Not recommended to expose on public networks
- For remote use, consider SSH tunnels or VPN

---

## Project Structure

```
wd-hydrus-tagger/
├── run.py                    # Entry point
├── config.yaml               # Config file (create from example)
├── config.example.yaml       # Example config
├── pyproject.toml             # Python project metadata
├── backend/                   # FastAPI backend
│   ├── app.py                 #   Application factory
│   ├── config.py              #   Config loader
│   ├── dependencies.py        #   Dependency injection
│   ├── hydrus/                #   Hydrus API client
│   │   ├── client.py          #     Async HTTP client
│   │   └── models.py          #     Data models
│   ├── tagger/                #   WD Tagger inference engine
│   │   ├── engine.py          #     ONNX inference
│   │   ├── preprocess.py      #     Image preprocessing
│   │   ├── labels.py          #     Label CSV parser
│   │   └── models.py          #     Result data models
│   ├── routes/                #   API routes
│   │   ├── connection.py      #     Connection management
│   │   ├── files.py           #     File browsing
│   │   ├── tagger.py          #     Tagging endpoints
│   │   └── config_routes.py   #     Config management
│   └── services/              #   Service layer
│       ├── tagging_service.py #     Batch tagging orchestrator
│       └── model_manager.py   #     Model download manager
├── frontend/                  # Web frontend
│   ├── index.html             #   Single-page application
│   ├── css/style.css          #   Stylesheet (dark theme)
│   └── js/                    #   JavaScript modules
│       ├── app.js             #     App initialization
│       ├── api.js             #     Backend API client
│       ├── state.js           #     State management
│       ├── components/        #     UI components
│       └── utils/             #     Utility functions
└── models/                    # Model files (auto-created)
```

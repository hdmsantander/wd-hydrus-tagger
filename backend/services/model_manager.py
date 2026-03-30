"""Model download, disk cache, and integrity verification (SmilingWolf WD v3)."""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

log = logging.getLogger(__name__)

SUPPORTED_MODELS = {
    "wd-vit-tagger-v3": "SmilingWolf/wd-vit-tagger-v3",
    "wd-swinv2-tagger-v3": "SmilingWolf/wd-swinv2-tagger-v3",
    "wd-vit-large-tagger-v3": "SmilingWolf/wd-vit-large-tagger-v3",
    "wd-eva02-large-tagger-v3": "SmilingWolf/wd-eva02-large-tagger-v3",
}

REQUIRED_FILES = ("model.onnx", "selected_tags.csv")
CACHE_MANIFEST_NAME = ".wd_model_cache.json"
# WD v3 ONNX files are hundreds of MB; tiny files are failed / partial downloads.
MIN_ONNX_BYTES = 1_000_000
# Real WD v3 CSVs are ~100KB+; this catches empty/truncated files without rejecting small test fixtures.
MIN_CSV_BYTES = 800
MIN_CSV_DATA_ROWS = 100
REQUIRED_CSV_FIELDS = frozenset({"tag_id", "name", "category"})
DEFAULT_HF_REVISION_REF = "main"


@dataclass
class ModelVerifyResult:
    name: str
    ok: bool
    issues: list[str] = field(default_factory=list)
    manifest_present: bool = False
    local_revision: str | None = None
    remote_revision: str | None = None
    stale_on_hub: bool | None = None


def fetch_repo_head_sha(repo_id: str, revision: str = DEFAULT_HF_REVISION_REF) -> str | None:
    """Best-effort: current ``revision`` tip on the Hub (needs network)."""
    try:
        api = HfApi()
        info = api.repo_info(repo_id=repo_id, repo_type="model", revision=revision)
        sha = getattr(info, "sha", None)
        return str(sha) if sha else None
    except Exception as e:
        log.warning("fetch_repo_head_sha failed repo_id=%s revision=%s: %s", repo_id, revision, e)
        return None


class ModelManager:
    def __init__(self, models_dir: str | Path):
        self.models_dir = Path(models_dir).expanduser().resolve()
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def _manifest_path(self, name: str) -> Path:
        return self.models_dir / name / CACHE_MANIFEST_NAME

    def read_manifest(self, name: str) -> dict[str, Any] | None:
        p = self._manifest_path(name)
        if not p.is_file():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            log.warning("read_manifest failed model=%s path=%s: %s", name, p, e)
            return None

    def _write_cache_manifest(self, name: str, repo: str) -> None:
        """Record repo, Hub revision, and file sizes after a successful cache."""
        model_dir = self.models_dir / name
        if not all((model_dir / f).is_file() for f in REQUIRED_FILES):
            return
        rev = fetch_repo_head_sha(repo, DEFAULT_HF_REVISION_REF)
        files_meta: dict[str, dict[str, int]] = {}
        for fn in REQUIRED_FILES:
            fp = model_dir / fn
            files_meta[fn] = {"size": fp.stat().st_size}
        payload = {
            "schema": 1,
            "kind": "wd_hydrus_tagger_model_cache",
            "model_name": name,
            "repo_id": repo,
            "revision_sha": rev or "",
            "revision_ref": DEFAULT_HF_REVISION_REF,
            "files": files_meta,
        }
        out = self._manifest_path(name)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log.info(
            "model_cache_manifest_written model=%s repo=%s revision_sha=%s path=%s",
            name,
            repo,
            (rev or "unknown")[:12] if rev else "unknown",
            out,
        )

    def repair_manifest_if_missing(self, name: str) -> bool:
        """If ONNX+CSV exist but manifest is absent, write manifest (no re-download)."""
        if name not in SUPPORTED_MODELS or not self.is_downloaded(name):
            return False
        if self.read_manifest(name) is not None:
            return False
        repo = SUPPORTED_MODELS[name]
        self._write_cache_manifest(name, repo)
        return True

    def list_models(self) -> list[dict]:
        """List supported models, disk presence, and local verification (no network)."""
        result: list[dict] = []
        for name, repo in SUPPORTED_MODELS.items():
            model_dir = self.models_dir / name
            downloaded = self.is_downloaded(name)
            vr = self.verify_model(name, check_remote=False)
            result.append({
                "name": name,
                "repo": repo,
                "downloaded": downloaded,
                "path": str(model_dir) if downloaded else None,
                "cache_ok": vr.ok,
                "cache_issues": vr.issues,
                "revision_sha": vr.local_revision,
                "manifest_present": vr.manifest_present,
            })
        return result

    def is_downloaded(self, name: str) -> bool:
        if name not in SUPPORTED_MODELS:
            return False
        model_dir = self.models_dir / name
        return all((model_dir / f).is_file() for f in REQUIRED_FILES)

    def _check_csv_structure(self, csv_path: Path) -> list[str]:
        issues: list[str] = []
        try:
            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    issues.append("csv_missing_header")
                    return issues
                fields = {h.strip() for h in reader.fieldnames if h}
                missing = REQUIRED_CSV_FIELDS - fields
                if missing:
                    issues.append(f"csv_missing_columns:{sorted(missing)}")
                    return issues
                n = 0
                for _ in reader:
                    n += 1
                    if n >= MIN_CSV_DATA_ROWS:
                        break
                if n < MIN_CSV_DATA_ROWS:
                    issues.append(f"csv_too_few_rows:{n}")
        except OSError as e:
            issues.append(f"csv_read_error:{e}")
        return issues

    def verify_model(self, name: str, *, check_remote: bool = False) -> ModelVerifyResult:
        """Validate cached files; optionally compare recorded Hub revision to current ``main``."""
        if name not in SUPPORTED_MODELS:
            return ModelVerifyResult(name=name, ok=False, issues=["unknown_model"])

        repo = SUPPORTED_MODELS[name]
        model_dir = self.models_dir / name
        issues: list[str] = []
        manifest = self.read_manifest(name)
        manifest_present = manifest is not None
        local_rev = None
        if isinstance(manifest, dict):
            local_rev = manifest.get("revision_sha") or None
            if local_rev == "":
                local_rev = None

        if not all((model_dir / f).is_file() for f in REQUIRED_FILES):
            return ModelVerifyResult(
                name=name,
                ok=False,
                issues=["missing_required_files"],
                manifest_present=manifest_present,
                local_revision=local_rev,
            )

        onnx_p = model_dir / "model.onnx"
        csv_p = model_dir / "selected_tags.csv"
        onnx_sz = onnx_p.stat().st_size
        csv_sz = csv_p.stat().st_size

        if onnx_sz < MIN_ONNX_BYTES:
            issues.append(f"onnx_too_small:{onnx_sz}")
        if csv_sz < MIN_CSV_BYTES:
            issues.append(f"csv_too_small:{csv_sz}")

        issues.extend(self._check_csv_structure(csv_p))

        if manifest and isinstance(manifest.get("files"), dict):
            mf = manifest["files"]
            for fn in REQUIRED_FILES:
                entry = mf.get(fn) if isinstance(mf.get(fn), dict) else None
                if not entry:
                    continue
                expected = entry.get("size")
                if expected is None:
                    continue
                actual = (model_dir / fn).stat().st_size
                if int(expected) != actual:
                    issues.append(f"size_mismatch_vs_manifest:{fn}")

        remote_rev: str | None = None
        stale: bool | None = None
        if check_remote:
            remote_rev = fetch_repo_head_sha(repo, DEFAULT_HF_REVISION_REF)
            if remote_rev and local_rev and local_rev != remote_rev:
                stale = True
                issues.append("newer_revision_on_hub")
            elif remote_rev and not local_rev:
                issues.append("manifest_missing_revision_cannot_compare_hub")

        # Hub-only notes do not invalidate a working local cache for inference.
        hard_issues = [
            i
            for i in issues
            if not i.startswith("newer_revision_on_hub")
            and not i.startswith("manifest_missing_revision_cannot_compare_hub")
        ]
        ok = len(hard_issues) == 0

        return ModelVerifyResult(
            name=name,
            ok=ok,
            issues=issues,
            manifest_present=manifest_present,
            local_revision=local_rev,
            remote_revision=remote_rev,
            stale_on_hub=stale,
        )

    def verify_all(self, *, check_remote: bool = False) -> list[ModelVerifyResult]:
        return [self.verify_model(n, check_remote=check_remote) for n in SUPPORTED_MODELS]

    def download_model(self, name: str) -> Path:
        """Download required files from HuggingFace Hub; skip files already on disk."""
        if name not in SUPPORTED_MODELS:
            raise ValueError(f"Unknown model: {name}. Available: {list(SUPPORTED_MODELS.keys())}")

        repo = SUPPORTED_MODELS[name]
        model_dir = self.models_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)

        prev_pb = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        try:
            for filename in REQUIRED_FILES:
                dest = model_dir / filename
                if dest.exists():
                    log.debug(
                        "model_hf_skip_existing file=%s model=%s bytes=%s",
                        filename,
                        name,
                        dest.stat().st_size,
                    )
                    continue
                log.info("model_hf_download start file=%s repo=%s dest_dir=%s", filename, repo, model_dir)
                hf_hub_download(
                    repo_id=repo,
                    filename=filename,
                    local_dir=str(model_dir),
                    revision=DEFAULT_HF_REVISION_REF,
                )
                log.info(
                    "model_hf_download done file=%s model=%s bytes=%s",
                    filename,
                    name,
                    (model_dir / filename).stat().st_size,
                )
        finally:
            if prev_pb is None:
                os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
            else:
                os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = prev_pb

        vr = self.verify_model(name, check_remote=False)
        if not vr.ok:
            log.error("model_download_verify_failed model=%s issues=%s", name, vr.issues)
            raise RuntimeError(f"Downloaded model failed verification: {name} ({vr.issues})")

        self._write_cache_manifest(name, repo)
        log.info(
            "model_download_complete model=%s dir=%s cache_ok=True",
            name,
            model_dir,
        )
        return model_dir

    def get_model_path(self, name: str) -> Path:
        model_dir = self.models_dir / name
        if not self.is_downloaded(name):
            raise FileNotFoundError(f"Model {name} not downloaded")
        return model_dir

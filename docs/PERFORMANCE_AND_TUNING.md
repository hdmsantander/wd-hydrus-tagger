# Performance, WD markers, and Tag all tuning

This document describes how the tagger skips work already done, how to read throughput, and how to tune **Tag all search results** on large Hydrus libraries.

## WD model markers (`wd14:<model_name>`)

After inference (when **Append model marker** is enabled), each file receives a stable tag such as `wd14:wd-vit-tagger-v3`. The next run can skip downloading pixels and ONNX for that file if the same marker is present on the **target tag service** (or on any service when no service key is used for the check).

### Skip rules (in order)

1. **Same model marker** — If `wd_skip_inference_if_marker_present` is true and the canonical marker for the **current** model is already in `storage_tags`, the file is skipped **before** image fetch and inference. Reason code: `wd_model_marker_present`.
2. **Heavier model already tagged** — If `wd_skip_if_higher_tier_model_present` is true and the file has a **strictly higher** capability tier among `wd14:*` markers (see tier table below), inference is skipped. Reason code: `wd_skip_higher_tier_model_present`. This speeds up **Tag all** when most files were already processed with a larger model.
3. **Upgrade path** — If the file has only a **lower** tier marker (or unknown slug → tier 0), inference **runs**. Stale `wd14:*` markers are deduped from the proposed tag list and replaced by the current model’s marker (see `dedupe_wd_model_markers_in_tags` in `backend/hydrus/tag_merge.py`).

### Model capability tiers

Defined in `WD_MODEL_CAPABILITY_TIER` (`backend/hydrus/tag_merge.py`):

| Model id | Tier |
|----------|------|
| `wd-vit-tagger-v3` | 1 |
| `wd-swinv2-tagger-v3` | 2 |
| `wd-vit-large-tagger-v3` | 3 |
| `wd-eva02-large-tagger-v3` | 4 |

Unknown slugs use tier **0** so inference still runs (safe default for custom or future models).

## Hydrus metadata: one prefetch per WebSocket session

**Tag all / Tag selected over WebSocket** loads **`get_file_metadata` once** for the full queued `file_ids` list (chunked like the gallery) **before** the outer batch loop. Each call to `tag_files` for an inference batch then reuses that map (`prefetched_meta_by_id`), so Hydrus is **not** asked again for the same IDs on every batch.

- **Second pass** (most files already carry a WD marker): you still pay **one** metadata scan up front for skip decisions, not *N batches* of redundant metadata calls.
- **`POST /api/tagger/predict`** already issued a single `tag_files` call over all IDs; behavior is unchanged.
- If session prefetch fails (network error), the server logs **`tagging_ws metadata_prefetch failed`** and **`tag_files`** falls back to fetching metadata per call as before.

## Default settings (sensible baselines)

From `config.example.yaml` / `backend/config.py`:

- **`batch_size`** / **`hydrus_download_parallel`** — Often keep them similar (e.g. 8×8 on an 8‑core desktop with 32 GB RAM). Reduce both for **Large** / **EVA02** models or tight RAM.
- **`hydrus_metadata_chunk_size`** — 256 default; raise toward **2048** for huge searches to cut Hydrus round-trips; lower if metadata responses are slow or huge.
- **`cpu_intra_op_threads`** — Match **physical** cores for ONNX CPU EP; keep **`cpu_inter_op_threads`** at **1** unless you know the graph benefits.
- **`wd_skip_inference_if_marker_present`** — **true** (skip duplicate work for the same model).
- **`wd_skip_if_higher_tier_model_present`** — **true** (fast-forward already–heavily-tagged files on mixed libraries).

Disable **heavier-model skip** if you intentionally re-run a smaller model over files tagged with a larger one and expect new predictions to replace behavior (rare; normally you use the larger model only).

## Performance tuning overlay (Tag all only)

In the Tagger panel, **Performance tuning overlay** sends `performance_tuning: true` with **`tag_all: true`** on the tagging WebSocket. The server:

- Records per-batch **Hydrus fetch** and **ONNX predict** wall time (`tag_files` metrics list).
- Adds **`hydrus_apply_batch_s`** for time spent in automatic Hydrus applies during that outer batch.

The UI shows:

- Extra lines in **progress stats** (fetch / ONNX / apply seconds for the last batch).
- A dedicated line under the stats block (`#progress-perf-tuning`).

**Tag selected** runs ignore this flag (server forces `performance_tuning` off).

**Logs:** batch-level apply duration is logged at **DEBUG** (`tagging_ws performance_tuning batch_idx=…`). Use `./wd-hydrus-tagger.sh run --log-level DEBUG` when diagnosing stalls.

## Interactive `config.yaml` wizard (Linux only)

**Supported on Linux only.** The wizard reads **`/proc/meminfo`** and **`/proc/cpuinfo`** (physical cores are estimated from unique `physical id` / `core id` pairs, with fallbacks) and optionally probes **`nvidia-smi`**. On macOS or Windows, copy **`config.example.yaml`** to **`config.yaml`** and edit by hand.

From the repo root:

```bash
./wd-hydrus-tagger.sh generate-config
# or
./wd-hydrus-tagger.sh --generate-config
```

The script reads **`config.example.yaml`**, prints hardware hints, then asks for Hydrus URL/key, model, GPU toggle, threads, batch sizes, marker skips, and bind address. Integer answers are clamped to the same bounds as the server config. Existing **`config.yaml`** is copied to **`config.yaml.bak`** before overwrite.

## Expected behavior (optimization scenarios)

| Scenario | Expected |
|----------|----------|
| Tag all with ViT base; files already have `wd14:wd-vit-tagger-v3` | High **skipped pre-infer** counts; little ONNX time; progress shows same-model skips. |
| Tag all with ViT base; files tagged with EVA02 Large | Skipped as **heavier model marker** if that option is on; very fast scan over those files. |
| First pass on untagged files | No marker skips; full fetch + predict; markers appended after apply. |
| Re-tag with **larger** model after **smaller** | Inference runs; old `wd14:*` markers removed from proposal; new marker appended. |

## Session log line (INFO)

At end of a WebSocket run, **`tagging_ws session_metrics`** includes:

- `onnx_skipped_same_marker` — cumulative same-model marker skips.
- `onnx_skipped_higher_tier_marker` — cumulative heavier-model marker skips.

Together with `files_processed` and tag write counts, this validates that large libraries are mostly skipping rather than re-inferring.

## See also

- [README.md](../README.md) — installation, Configuration Reference, logging overview.
- [TUNING_MODE_UPGRADE_PLAN.md](TUNING_MODE_UPGRADE_PLAN.md) — historical design notes for tuning payloads.

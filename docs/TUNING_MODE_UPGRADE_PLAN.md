# Tuning / auto-profiling mode — upgrade plan (estimate)

This document estimates effort, runtime impact, and design for an optional **throughput tuning mode** aimed at **large “Tag all search results”** runs (the UI path that writes each inference batch to Hydrus incrementally). It is **not implemented**; it is a planning reference.

**Terminology:** There is no “import all pictures” mode in this repo. The closest match is **Tag all search results** (`tagAll` in `frontend/js/components/tagger.js`), which forces `apply_tags_every_n = inference_batch` and requires a tag service.

---

## 1. Goals vs. current architecture

| Goal | Reality today |
|------|----------------|
| “CPU agents” | Single ONNX `InferenceSession`; **parallelism** is `cpu_intra_op_threads` / `cpu_inter_op_threads` (ORT) plus **`hydrus_download_parallel`** HTTP fetches and asyncio concurrency—not separate OS processes. |
| Live parameter changes | Changing ORT thread counts requires **`TaggerEngine.load()`** (new `SessionOptions`) → **model reload** (seconds, disk/CPU spike). **Batch size** and **`hydrus_download_parallel`** can change **per WebSocket run** without reload (already partially supported). |
| “Real-time” proposals | Feasible as **advisory** (UI + logs) or **session-local overrides**; risky as silent `PATCH /api/config` without user confirm. |
| Large datasets | Bottleneck is often **Hydrus I/O + ONNX**; tuning must measure **both** (see §4). |

**Industry references (high level):**

- **ONNX Runtime:** [Tune performance](https://onnxruntime.ai/docs/performance/tune-performance/), [Thread management](https://onnxruntime.ai/docs/performance/tune-performance/threading.html) — intra vs inter threads, sequential vs parallel execution mode, optional profiling (heavy).
- **OLive:** Microsoft’s tooling for automated model/runtime optimization pipelines — heavier than this app needs; useful as a **pattern** (search over configs offline), not as a runtime dependency for the SPA.
- **Typical production pattern:** **Offline** benchmark (fixed hardware, representative batch sizes) → commit `config.yaml`; **online** only light metrics (counters, wall time per stage) unless debugging.

---

## 2. Effort estimate (engineering weeks, one developer)

Assumes familiarity with this codebase.

| Tier | Scope | Rough effort | Risk |
|------|--------|--------------|------|
| **A — Metrics only** | Extend WebSocket payloads / logs with structured **per-batch** stats (already have fetch/predict walls in `tagging_service`); add a **Tag-all-only** checkbox that enables a **rolling window** (e.g. last N batches) exposed in UI (read-only table + sparkline optional). No auto-change. | **1–2 weeks** | Low |
| **B — Advisory tuner** | Tier A + **heuristic suggestions** (“try batch_size 4”, “try download_parallel 6”) from simple rules (e.g. predict wall ≫ fetch wall → CPU-bound vs I/O-bound). User applies via Settings or YAML; optional **one-click copy** of suggested `PATCH` body. | **+1–2 weeks** (2–4 total) | Medium (wrong advice on mixed bottlenecks) |
| **C — Session auto-tune** | Tier B + **bounded experiments** inside one Tag-all run (e.g. try alternate `batch_size` / `hydrus_download_parallel` for next K batches, pick best throughput). **ORT thread** changes gated behind **explicit reload** step. Persist **run report** JSON. | **+2–4 weeks** (4–8 total) | High (stability, fairness, user expectations) |
| **D — ORT built-in profiling** | `SessionOptions.enable_profiling` + trace export — powerful for **debug** sessions only; **not** for default Tag-all. | **~3–5 days** (isolated mode) | Profiling **hurts throughput** and **disk**; must be strictly opt-in |

**Recommendation:** Ship **A**, prototype **B** behind a feature flag; treat **C** as a second phase after validated metrics.

---

## 3. Performance and memory impact

| Mechanism | CPU | RAM | Notes |
|-----------|-----|-----|--------|
| Extra `perf_counter` + rolling stats (fixed N batches) | Negligible | +O(N) small dicts | Align with existing `perf_metrics` / batch logs. |
| More WebSocket JSON (tuning panel) | Small | Small | Throttle updates (e.g. on batch boundary only). |
| ORT `enable_profiling` | **High** | **Higher** + trace files | Use only in dedicated “diagnostic” mode, not Tag-all production. |
| Mid-run `engine.load()` for new thread counts | Spike | Transient peak (old+new session briefly) | Coordinate with `unload_model` / single-flight lock. |
| Increasing `batch_size` | — | **↑** activations | Already the main RAM knob. |

**Guaranteeing “maximum efficiency”** is not realistically automatic: efficiency depends on Hydrus latency, disk, network, model size, and OS scheduler. The product goal should be **“measurable improvement on this machine for this run”** with **safe bounds** (clamp min/max, never OOM).

---

## 4. Key metrics to track (Tag-all, large queue)

Prioritize **stage-separated** times (already partially logged):

1. **Hydrus fetch wall** per outer batch (`wall_hydrus_fetch_s` aggregate in `tag_files` metrics).
2. **ONNX predict wall** per batch (`wall_onnx_predict_s`).
3. **Hydrus apply wall** per incremental chunk (derive from `tagging_ws` around `_apply_results_chunk` or add timers).
4. **Files/sec effective** — `batch_size / (fetch + predict + apply_overhead)`.
5. **Queue depth** — pending apply count, skipped-marker ratio (if most files skip ONNX, CPU tuning matters less).
6. **Optional:** `ru_maxrss` sample at batch boundary — **coarse**; cheap on Linux if sampled every M batches only.

**Derived KPIs:** throughput (files/s), “predict share” vs “Hydrus share” of batch time, duplicate-tag ratio (Hydrus-bound apply).

---

## 5. Code / flow changes (conceptual)

**Backend**

- `AppConfig`: optional `tuning_mode_enabled` (default false) or **no persisted flag** — only WebSocket `tuning: true` for Tag-all to avoid accidental persistence.
- `tagger.py` `progress_ws`: when `tag_all && tuning`, attach **`tuning_snapshot`** to `progress` messages (rolling window) and/or new message type `tuning_stats`.
- `tagging_service.tag_files`: ensure per-batch timings are **structured** (dict) for machine consumption, not only log strings.
- `TaggerEngine`: optional **diagnostic** session path for ORT profiling; never default.
- **Safety:** any auto-applied numeric change must stay within existing Pydantic bounds (`batch_size`, `hydrus_download_parallel`, threads).

**Frontend**

- Tagger panel: checkbox **only enabled when Tag all** (or when `tagAll` flow is selected) + **tuning dashboard** section (table: batch index, fetch_s, predict_s, apply_s, throughput).
- Clear UX: “Suggestions are advisory” / “Reload model to apply thread changes.”

**Tests**

- Unit: rolling window reducer, heuristic function (if B).
- Integration: WebSocket receives `tuning_stats` when flag set; does not when Tag selected only.
- Load / stress: optional manual benchmark script (not CI) comparing two configs.

---

## 6. Comparison to “similar” systems

| Approach | This app |
|----------|----------|
| **Offline autotune (OLive-style)** | Best for **shipping** a default `config.example.yaml` profile per CPU class; run once per release or machine class. |
| **Online adaptive batching** | Used in some serving stacks (dynamic batching); here batches are tied to **Hydrus incremental apply** — changing batch size also changes **apply granularity**; must stay consistent with user expectations. |
| **Full profilers (PyTorch/ORT trace)** | Great for **debug**; keep out of hot path for end users. |

---

## 7. Existing codebase: quick optimization opportunities (non–tuning-mode)

These are **incremental** improvements unrelated to the tuning UI but aligned with “main loop” quality:

- **WebSocket loop** (`tagger.py`): already yields between batches; any extra work should stay **O(1)** per batch (avoid large JSON copies in verbose mode on huge batches).
- **`tag_files`**: batch `finally` already drops references; keep **no** per-file logging at DEBUG in huge runs unless behind a flag.
- **Tests:** `pytest --no-cov` for speed; ensure WebSocket tests **mock** Hydrus to avoid accidental network; consider **parametrize** batch sizes for regression on clamp logic only (not full ONNX in CI).

---

## 8. Summary

- **Realistic v1:** Tag-all-only **observability** + rolling metrics in UI (**Tier A**, ~1–2 weeks).
- **Advisory suggestions:** +1–2 weeks with careful heuristics (**Tier B**).
- **Live auto-reconfiguration:** significant complexity + **model reload** for ORT threads; budget **4–8+ weeks** for **Tier C** with strong safeguards.
- **Memory/CPU:** light metrics are safe; **ORT profiling** and **frequent reloads** are not.
- Align language with the product: **“Tag all search results”**, not “import all.”

When implementing, add a short entry to `docs/UPGRADE.md` and link this file from the README **Configuration** or **Performance** section.

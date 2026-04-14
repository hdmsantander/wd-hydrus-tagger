#!/usr/bin/env python3
"""Parse wd-hydrus-tagger run logs and write a Markdown report (tagging sessions + metrics table).

Reads perf lines, metadata_prefetch, queue_analysis (if present), session_metrics, DEBUG rates.
Default input: ``logs/latest.log``. Override with first positional arg or ``--log PATH``.

Example:
  python scripts/analyze_tagging_log.py
  python scripts/analyze_tagging_log.py logs/runs/run-20260331T133803-494324.log
  python scripts/analyze_tagging_log.py --out reports/my-analysis.md logs/latest.log
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

from backend.log_parsing import parse_tag_files_metrics_line

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _parse_sessions(lines: list[str]) -> list[dict]:
    prefetch_idx = [i for i, l in enumerate(lines) if "tagging_ws metadata_prefetch" in l]
    perf_idx = [i for i, l in enumerate(lines) if "perf tagging_session" in l and "wall_s=" in l]

    runs: list[dict] = []
    for r in range(len(prefetch_idx)):
        start = prefetch_idx[r]
        end = perf_idx[r] if r < len(perf_idx) else len(lines)
        seg = lines[start : end + 1]

        pf = re.search(r"metadata_prefetch wall_s=([\d.]+).*rows=(\d+).*file_ids=(\d+)", lines[start])
        prefetch_s = float(pf.group(1)) if pf else 0.0
        rows_n = int(pf.group(2)) if pf else 0

        onnx = hydrus_f = 0.0
        skip_m = infer = 0
        for l in seg:
            if "tag_files metrics" in l and "wall_onnx_predict_s=" in l:
                parsed = parse_tag_files_metrics_line(l)
                onnx += float(parsed.get("wall_onnx_predict_s") or 0.0)
                hydrus_f += float(parsed.get("wall_hydrus_fetch_s") or 0.0)
                skip_m += int(parsed.get("skipped_pre_infer_marker_files") or 0)
                infer += int(parsed.get("inferred_files") or 0)

        pl = lines[end]
        pm = re.search(
            r"perf tagging_session wall_s=([\d.]+) model_prepare_s=([\d.]+) processed=(\d+) outer_batches=(\d+) "
            r"hydrus_files=(\d+) tag_strings=(\d+) outcome=(\S+)/(\S+) model=(\S+)",
            pl,
        )
        perf = pm.groups() if pm else None

        sm = None
        for j in range(end, max(start, end - 8), -1):
            if "tagging_ws session_metrics" in lines[j]:
                sm = re.search(
                    r"onnx_skipped_same_marker=(\d+).*onnx_skipped_higher_tier_marker=(\d+).*"
                    r"files_processed=(\d+).*tags_new_strings_applied=(\d+)",
                    lines[j],
                )
                break

        dbg = None
        for j in range(end, min(end + 5, len(lines))):
            if "session_perf_rates" in lines[j]:
                dbg = re.search(
                    r"wall_s=([\d.]+) inferred_non_skip=(\d+) inferred_non_skip_per_s=([\d.]+) "
                    r"tags_new_strings_per_s=([\d.]+) outer_batches=(\d+)",
                    lines[j],
                )
                break

        qline = None
        for l in seg:
            if "tagging_ws queue_analysis" in l:
                qline = l
                break
        qa = None
        if qline:
            qm = re.search(
                r"queue_analysis infer=(\d+) skip_same_marker=(\d+) skip_higher_tier=(\d+) "
                r"missing_metadata=(\d+)",
                qline,
            )
            if qm:
                qa = {
                    "infer": int(qm.group(1)),
                    "skip_same": int(qm.group(2)),
                    "skip_hi": int(qm.group(3)),
                    "missing": int(qm.group(4)),
                }

        cfg = None
        head = lines[max(0, start - 40) : start + 1]
        for l in head:
            if "tagging_ws session_config" in l:
                bi = re.search(r"config_inference_batch_saved=(\d+)", l)
                ae = re.search(r"apply_tags_every_n_effective=(\d+)", l)
                ort = re.search(r"ort_cpu_threads_intra_inter=(\d+)/(\d+)", l)
                if bi and ae and ort:
                    cfg = {
                        "batch": int(bi.group(1)),
                        "apply_every": int(ae.group(1)),
                        "ort_intra": int(ort.group(1)),
                        "ort_inter": int(ort.group(2)),
                    }
                break

        runs.append(
            {
                "prefetch_s": prefetch_s,
                "prefetch_rows": rows_n,
                "sum_onnx_wall_s": round(onnx, 3),
                "sum_hydrus_fetch_s": round(hydrus_f, 3),
                "sum_skipped_pre_infer": skip_m,
                "sum_inferred_files": infer,
                "perf": perf,
                "session_metrics": sm.groups() if sm else None,
                "debug_rates": dbg.groups() if dbg else None,
                "queue_analysis": qa,
                "session_config": cfg,
            }
        )
    return runs


def _markdown_table(runs: list[dict]) -> str:
    lines = [
        "| Run | cfg batch/apply/ORT | prefetch s | wall s | ΣONNX s | Σfetch s | batches | processed | skip_same | non_skip | tags | inf/s | tags/s | outcome |",
        "|-----|---------------------|-----------|--------|---------|---------|---------|-----------|-----------|----------|------|-------|--------|---------|",
    ]
    for i, run in enumerate(runs, start=1):
        p = run["perf"]
        d = run["debug_rates"]
        sm = run["session_metrics"]
        cfg = run["session_config"] or {}
        cfg_s = ""
        if cfg:
            cfg_s = f"{cfg.get('batch', '?')}/{cfg.get('apply_every', '?')}/{cfg.get('ort_intra', '?')}/{cfg.get('ort_inter', '?')}"
        if p:
            wall_s, _mp, proc, ob, _hf, ts, oc, st, _model = p
            inf_ns = d[1] if d else "—"
            inf_rate = d[2] if d else "—"
            tag_rate = d[3] if d else "—"
            oc_st = f"{oc}/{st}"
            lines.append(
                f"| {i} | {cfg_s} | {run['prefetch_s']:.3f} | {wall_s} | {run['sum_onnx_wall_s']} | "
                f"{run['sum_hydrus_fetch_s']:.3f} | {ob} | {proc} | {sm[0] if sm else '—'} | {inf_ns} | {ts} | "
                f"{inf_rate} | {tag_rate} | {oc_st} |"
            )
        else:
            lines.append(f"| {i} | {cfg_s} | … | (no perf line in segment) | | | | | | | | | | |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate Markdown report from tagging log.")
    ap.add_argument(
        "log_path",
        nargs="?",
        default=str(_REPO_ROOT / "logs" / "latest.log"),
        help="Log file (default: logs/latest.log)",
    )
    ap.add_argument(
        "--out",
        "-o",
        help="Output Markdown path (default: reports/tagging-analysis-<timestamp>.md)",
    )
    args = ap.parse_args()
    log_path = Path(args.log_path)
    if not log_path.is_file():
        print(f"error: log not found: {log_path}", flush=True)
        return 1

    text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    runs = _parse_sessions(lines)

    out_dir = _REPO_ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = Path(args.out) if args.out else out_dir / f"tagging-analysis-{ts}.md"
    if not out_path.is_absolute():
        out_path = _REPO_ROOT / out_path

    shutdown = None
    for l in reversed(lines):
        if "perf process_shutdown" in l:
            shutdown = l.strip()
            break

    body = [
        f"# Tagging log analysis",
        "",
        f"- **Source:** `{log_path.relative_to(_REPO_ROOT) if log_path.is_relative_to(_REPO_ROOT) else log_path}`",
        f"- **Sessions detected:** {len(runs)}",
        "",
        "## Summary table",
        "",
        _markdown_table(runs),
        "",
        "## Queue analysis lines (prefetch)",
        "",
    ]
    for i, run in enumerate(runs, start=1):
        qa = run.get("queue_analysis")
        if qa:
            body.append(f"- Run {i}: infer={qa['infer']} skip_same={qa['skip_same']} "
                        f"skip_higher_tier={qa['skip_hi']} missing_meta={qa['missing']}")
        else:
            body.append(f"- Run {i}: (no `queue_analysis` in log — older binary)")
    body.extend(["", "## Shutdown", "", f"`{shutdown}`" if shutdown else "(no shutdown line)", ""])

    out_path.write_text("\n".join(body), encoding="utf-8")
    print(f"Wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

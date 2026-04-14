# Tagging log reports

This directory holds **generated Markdown analyses** of run logs: session tables (prefetch wall time, ONNX/Hydrus totals, `queue_analysis`, `session_metrics`, inferred/tags rates). The canonical generator is `scripts/analyze_tagging_log.py`.

## `tagging-report` (shell)

From the repository root (same as other `./wd-hydrus-tagger.sh` commands):

| Invocation | Behavior |
|------------|----------|
| `./wd-hydrus-tagger.sh tagging-report` | Analyzes **`logs/latest.log`**. Writes **`reports/tagging-analysis-<UTC-timestamp>.md`** unless you pass `--out`. |
| `./wd-hydrus-tagger.sh tagging-report path/to/run.log` | Uses that log file as input. |
| `./wd-hydrus-tagger.sh tagging-report --out reports/my-run.md logs/runs/run-….log` | Fixed output path (created under repo if relative). |

- **Alias:** `tagging-analysis` is the same command.
- **Forwarding:** Any extra arguments are passed to the Python script (`--help` shows full options).
- **Git:** Generated `*.md` files here are **ignored** by default (see `.gitignore`); keep **`README.md`** for docs. Commit a generated report only when you want a permanent before/after record (e.g. tuning).

## Direct Python

```bash
python scripts/analyze_tagging_log.py
python scripts/analyze_tagging_log.py logs/runs/run-20260331T133803-494324.log
python scripts/analyze_tagging_log.py --out reports/manual.md logs/latest.log
```

Defaults match the shell wrapper: input **`logs/latest.log`**, output **`reports/tagging-analysis-<timestamp>.md`** when `--out` is omitted.

## Tests & coverage

Backend coverage is enforced in **`pyproject.toml`** (`[tool.coverage.report]` `fail_under`, currently **82%**). Tests focus on critical paths: Hydrus client and transport errors, queue analysis, tagger engine inference, `tag_merge`, coordinated shutdown, connection/files routes, and pieces of `tagger_ws` (for example `_wait_until_hydrus_responsive`). Pushing toward **85%** mostly needs additional WebSocket session scenarios in `tagger_ws.py` (large surface area), not more smoke tests elsewhere.

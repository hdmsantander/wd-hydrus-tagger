"""Log digest helper for tracing runs (cache + metadata keywords)."""

from pathlib import Path

from backend.log_report import LogDigest, analyze_log_path, analyze_log_text, format_digest


def test_analyze_log_text_counts_cache_and_metadata():
    text = """2025-01-01 12:00:00 INFO [rid] [backend] Application ready host=0.0.0.0 port=8199 run_id=x log_file=y
2025-01-01 12:00:01 INFO [rid] [backend] ensure_model metrics model=m memory_cache_hit=True duration_ms=0.10
2025-01-01 12:00:02 INFO [rid] [backend] ensure_model metrics model=m memory_cache_hit=False duration_ms=100
2025-01-01 12:00:03 INFO [rid] [backend] load_model metrics model=m disk_cache_hit=True hub_fetch_this_call=False
2025-01-01 12:00:04 INFO [rid] [backend] load_model metrics model=m disk_cache_hit=False hub_fetch_this_call=True
2025-01-01 12:00:05 INFO [rid] [backend] load_model disk cache miss — fetching from HuggingFace: x
2025-01-01 12:00:06 WARNING [rid] [backend] load_model disk_cache_invalid model=x refetch_from_hub issues=[]
2025-01-01 12:00:07 INFO [rid] [backend] tag_files metadata rows=10 file_ids=10 source=fetch chunk_size=256
2025-01-01 12:00:07 INFO [rid] [backend] tagging_ws metadata_prefetch rows=10 file_ids=10 chunk=256
2025-01-01 12:00:08 INFO [rid] [backend] files metadata_hydrus file_ids=5 chunk_size=256 chunks=1 rows_returned=5
2025-01-01 12:00:09 ERROR [rid] [backend] something failed
"""
    d = analyze_log_text(text, path="/tmp/x.log")
    assert d.lines == 11
    assert d.error_count == 1
    assert d.warning_count == 1
    assert d.memory_cache_hit_true == 1
    assert d.memory_cache_hit_false == 1
    assert d.disk_cache_hit_true == 1
    assert d.disk_cache_hit_false == 1
    assert d.disk_cache_miss_lines == 1
    assert d.disk_cache_invalid_lines == 1
    assert d.hub_fetch_true == 1
    assert d.hub_fetch_false == 1
    assert d.tag_files_metadata_fetched == 2
    assert d.files_metadata_hydrus == 1
    assert d.application_ready_lines == 1
    assert len(d.sample_errors) == 1


def test_format_digest_includes_path(tmp_path):
    p = tmp_path / "run.log"
    p.write_text(
        "2025-01-01 12:00:00 INFO [r] [n] ensure_model metrics model=x memory_cache_hit=True duration_ms=1\n",
        encoding="utf-8",
    )
    d = analyze_log_path(p)
    out = format_digest(d)
    assert str(p.resolve()) in out
    assert "memory_cache_hit=True" in out


def test_log_digest_dataclass_defaults():
    d = LogDigest()
    assert d.lines == 0
    assert d.error_count == 0

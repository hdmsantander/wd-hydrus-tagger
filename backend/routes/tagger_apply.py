"""Hydrus apply and result-trim helpers shared by tagger HTTP and WebSocket routes."""

import logging

from backend.config import AppConfig
from backend.hydrus.client import HydrusClient
from backend.hydrus.metadata_maps import rows_to_file_id_map
from backend.hydrus.tag_merge import (
    coalesce_wd_result_tag_strings,
    existing_storage_tag_keys,
    filter_new_tags,
    prune_wd_result_to_pending_tags,
)

log = logging.getLogger(__name__)

_METADATA_TRIM_CHUNK = 256


def _prefix_kwargs(cfg: AppConfig) -> dict[str, str]:
    return {
        "general_prefix": cfg.general_tag_prefix or "",
        "character_prefix": cfg.character_tag_prefix or "",
        "rating_prefix": cfg.rating_tag_prefix or "",
    }


async def _apply_results_chunk(
    client: HydrusClient,
    service_key: str,
    items: list[dict],
) -> tuple[int, int, int]:
    """Write new tags to Hydrus (skips tags already in storage_tags for that service).

    Returns (files_written, new_tag_strings_sent, duplicate_tags_skipped).
    """
    if not service_key or not items:
        return (0, 0, 0)

    ids = [int(i["file_id"]) for i in items if i.get("file_id") is not None]
    meta_by_id: dict[int, dict] = {}
    if ids:
        try:
            rows = await client.get_file_metadata(file_ids=ids)
            meta_by_id = rows_to_file_id_map(rows)
        except Exception:
            log.exception("get_file_metadata failed during apply; applying without deduplication")
            meta_by_id = {}

    files = 0
    tag_strings = 0
    duplicates_skipped = 0
    items_with_tags = 0
    items_all_duplicates = 0
    for item in items:
        tags = item.get("tags") or item.get("formatted_tags", [])
        if not item.get("hash") or not tags:
            continue
        items_with_tags += 1
        tag_list = list(tags)
        fid = item.get("file_id")
        meta = meta_by_id.get(int(fid)) if fid is not None else None
        existing = existing_storage_tag_keys(meta, service_key) if meta is not None else set()
        new_tags, skipped = filter_new_tags(tag_list, existing)
        duplicates_skipped += skipped
        if not new_tags:
            items_all_duplicates += 1
            continue
        try:
            await client.add_tags(
                hash_=item["hash"],
                service_key=service_key,
                tags=new_tags,
            )
        except Exception:
            log.exception(
                "add_tags failed file_id=%s hash=%s… new_tag_count=%s",
                fid,
                (item.get("hash") or "")[:12],
                len(new_tags),
            )
            raise
        files += 1
        tag_strings += len(new_tags)
    log.info(
        "apply_tags chunk items=%s with_tags=%s files_written=%s new_tag_strings=%s "
        "hydrus_duplicate_tag_strings_skipped=%s items_unchanged_all_dupes=%s",
        len(items),
        items_with_tags,
        files,
        tag_strings,
        duplicates_skipped,
        items_all_duplicates,
    )
    return (files, tag_strings, duplicates_skipped)


async def _trim_ws_results_to_pending_for_service(
    client: HydrusClient,
    service_key: str,
    results: list[dict],
    config: AppConfig,
) -> int:
    """Set each result's tags / structured fields to tags not yet in Hydrus for ``service_key``.

    Returns how many files still have at least one pending tag string.
    """
    if not service_key or not results:
        return 0

    kw = _prefix_kwargs(config)
    ids_unique: list[int] = []
    seen: set[int] = set()
    for r in results:
        if r.get("skipped_inference"):
            continue
        fid = r.get("file_id")
        if fid is None:
            continue
        i = int(fid)
        if i not in seen:
            seen.add(i)
            ids_unique.append(i)

    meta_by_id: dict[int, dict] = {}
    try:
        for off in range(0, len(ids_unique), _METADATA_TRIM_CHUNK):
            chunk = ids_unique[off : off + _METADATA_TRIM_CHUNK]
            rows = await client.get_file_metadata(file_ids=chunk)
            meta_by_id.update(rows_to_file_id_map(rows))
    except Exception:
        log.exception("trim_ws_results: get_file_metadata failed; leaving results unchanged")
        return 0

    pending_files = 0
    for r in results:
        if r.get("skipped_inference"):
            prune_wd_result_to_pending_tags(r, [], **kw)
            continue
        fid = r.get("file_id")
        if fid is None:
            continue
        meta = meta_by_id.get(int(fid))
        existing = existing_storage_tag_keys(meta, service_key) if meta is not None else set()
        proposed = coalesce_wd_result_tag_strings(r, **kw)
        pending, _ = filter_new_tags(proposed, existing)
        prune_wd_result_to_pending_tags(r, pending, **kw)
        if pending:
            pending_files += 1

    log.info(
        "trim_ws_results_to_pending service_key_set=%s results=%s files_pending_apply=%s",
        bool(service_key),
        len(results),
        pending_files,
    )
    return pending_files

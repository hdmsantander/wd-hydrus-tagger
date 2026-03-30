"""Compare WD output tags with Hydrus storage_tags to avoid duplicate add_tags calls."""

from __future__ import annotations

# Higher = heavier / generally more capable WD v3 variants (used for skip-if-already-tagged).
WD_MODEL_CAPABILITY_TIER: dict[str, int] = {
    "wd-vit-tagger-v3": 1,
    "wd-swinv2-tagger-v3": 2,
    "wd-vit-large-tagger-v3": 3,
    "wd-eva02-large-tagger-v3": 4,
}
WD_UNKNOWN_MODEL_TIER = 0


def normalize_tag_for_compare(tag: str) -> str:
    """Lowercase, strip; treat underscores and hyphens like spaces (Hydrus vs WD naming)."""
    return tag.replace("_", " ").replace("-", " ").strip().lower()


def all_normalized_storage_tag_keys(metadata: dict | None) -> set[str]:
    """All ``storage_tags`` strings on the file across every service (single pass).

    Used for marker checks when no target service is set — avoids scanning each
    service with repeated tree walks.
    """
    if not metadata:
        return set()
    tags_root = metadata.get("tags")
    if not isinstance(tags_root, dict):
        return set()
    out: set[str] = set()
    for block in tags_root.values():
        if not isinstance(block, dict):
            continue
        storage = block.get("storage_tags")
        if not isinstance(storage, dict):
            continue
        for tag_list in storage.values():
            if not isinstance(tag_list, list):
                continue
            for t in tag_list:
                if isinstance(t, str) and t.strip():
                    out.add(normalize_tag_for_compare(t))
    return out


def existing_storage_tag_keys(metadata: dict | None, service_key: str) -> set[str]:
    """Normalized keys for tags already stored on the file for this service (storage_tags only)."""
    if not metadata or not service_key:
        return set()
    tags_root = metadata.get("tags")
    if not isinstance(tags_root, dict):
        return set()
    block = tags_root.get(service_key)
    if not isinstance(block, dict):
        return set()
    storage = block.get("storage_tags")
    if not isinstance(storage, dict):
        return set()
    out: set[str] = set()
    for tag_list in storage.values():
        if not isinstance(tag_list, list):
            continue
        for t in tag_list:
            if isinstance(t, str) and t.strip():
                out.add(normalize_tag_for_compare(t))
    return out


def build_wd_model_marker(model_name: str, template: str = "") -> str:
    """Stable tag string marking that WD inference was applied for this model id.

    Default ``wd14:<model_name>``. If ``template`` contains ``{model_name}``, it is
    formatted; otherwise the non-empty template is used as a literal tag (same for
    every model — rarely useful).
    """
    name = (model_name or "").strip()
    if not name:
        return ""
    t = (template or "").strip()
    if not t:
        return f"wd14:{name}"
    if "{model_name}" in t:
        try:
            return t.format(model_name=name)
        except Exception:
            return f"wd14:{name}"
    return t


def tag_list_contains_normalized(tags: list[str], needle: str) -> bool:
    """True if any list entry matches ``needle`` after normalization."""
    if not needle or not needle.strip():
        return False
    n = normalize_tag_for_compare(needle.strip())
    if not n:
        return False
    return any(normalize_tag_for_compare(t) == n for t in tags if isinstance(t, str))


def dedupe_wd_model_markers_in_tags(
    tags: list[str],
    canonical_marker: str,
    *,
    marker_prefix: str = "wd14:",
) -> tuple[list[str], int]:
    """Remove ``marker_prefix``* tags that are not the current model; keep one canonical marker.

    Returns ``(new_tags, removed_count)``. Appends ``canonical_marker`` once if no matching wd14 tag remains.
    """
    if not (canonical_marker or "").strip():
        return list(tags), 0
    cur = normalize_tag_for_compare(canonical_marker.strip())
    pfx = normalize_tag_for_compare(marker_prefix)
    if not cur.startswith(pfx):
        return list(tags), 0
    removed = 0
    out: list[str] = []
    seen_cur = False
    for t in tags:
        if not isinstance(t, str) or not t.strip():
            continue
        n = normalize_tag_for_compare(t)
        if n.startswith(pfx):
            if n == cur:
                if not seen_cur:
                    out.append(canonical_marker.strip())
                    seen_cur = True
                else:
                    removed += 1
            else:
                removed += 1
        else:
            out.append(t)
    if not seen_cur:
        out.append(canonical_marker.strip())
    return out, removed


def model_capability_tier(model_name: str) -> int:
    """Numeric tier for a WD model id (unknown models → 0 so we still run inference)."""
    name = (model_name or "").strip()
    if not name:
        return WD_UNKNOWN_MODEL_TIER
    if name in WD_MODEL_CAPABILITY_TIER:
        return WD_MODEL_CAPABILITY_TIER[name]
    alt = name.replace("_", "-")
    return WD_MODEL_CAPABILITY_TIER.get(alt, WD_UNKNOWN_MODEL_TIER)


def _iter_storage_tag_strings(metadata: dict | None, service_key: str) -> list[str]:
    """Raw storage_tags strings for one service (if set) or all services."""
    if not metadata:
        return []
    tags_root = metadata.get("tags")
    if not isinstance(tags_root, dict):
        return []
    out: list[str] = []
    sk = (service_key or "").strip()
    blocks: list[dict] = []
    if sk:
        b = tags_root.get(sk)
        if isinstance(b, dict):
            blocks.append(b)
    else:
        for b in tags_root.values():
            if isinstance(b, dict):
                blocks.append(b)
    for block in blocks:
        storage = block.get("storage_tags")
        if not isinstance(storage, dict):
            continue
        for tag_list in storage.values():
            if not isinstance(tag_list, list):
                continue
            for t in tag_list:
                if isinstance(t, str) and t.strip():
                    out.append(t.strip())
    return out


def max_wd_marker_tier_on_file(
    metadata: dict | None,
    service_key: str,
    marker_prefix: str,
) -> tuple[int, str | None]:
    """Highest capability tier among ``wd14:*``-style model markers on the file.

    Parses raw ``storage_tags`` strings (before normalization) so model slugs stay
    identifiable. Returns ``(max_tier, slug_or_none)`` for the winning marker.
    """
    pfx = (marker_prefix or "wd14:").strip().lower()
    if not pfx.endswith(":"):
        pfx = f"{pfx}:"
    best_t = WD_UNKNOWN_MODEL_TIER
    best_slug: str | None = None
    for raw in _iter_storage_tag_strings(metadata, service_key):
        low = raw.lower()
        if not low.startswith(pfx):
            continue
        slug = raw[len(pfx) :].strip()
        if not slug:
            continue
        t = model_capability_tier(slug)
        if t > best_t:
            best_t = t
            best_slug = slug
    return best_t, best_slug


def inference_skip_decision(
    metadata: dict | None,
    *,
    current_model: str,
    canonical_marker: str,
    skip_same_model_marker: bool,
    skip_if_higher_tier_model: bool,
    service_key: str,
    marker_prefix: str,
) -> tuple[bool, str | None]:
    """Whether to skip ONNX+fetch for this file (metadata-only).

    * ``skip_same_model_marker``: canonical ``wd14:…`` for *this* run matches storage.
    * ``skip_if_higher_tier_model``: file already has a **strictly higher** tier WD marker
      (see ``WD_MODEL_CAPABILITY_TIER``) — skip so large libraries fast-forward past
      files already tagged with a heavier model.

    Either flag can be enabled independently. Returns ``(skip, reason_code)``.
    """
    sk = (service_key or "").strip()
    if skip_same_model_marker and canonical_marker and marker_present_on_file(
        metadata, canonical_marker, sk
    ):
        return True, "wd_model_marker_present"
    if skip_if_higher_tier_model:
        max_t, _slug = max_wd_marker_tier_on_file(metadata, sk, marker_prefix)
        cur_t = model_capability_tier(current_model)
        if max_t > cur_t:
            return True, "wd_skip_higher_tier_model_present"
    return False, None


def marker_present_on_file(
    metadata: dict | None,
    marker_tag: str,
    service_key: str = "",
) -> bool:
    """True if ``marker_tag`` (normalized) exists in Hydrus ``storage_tags``.

    If ``service_key`` is set, only that tag service is checked. If empty, any
    service on the file is checked (useful when the UI has not sent a service key).
    """
    if not metadata or not (marker_tag or "").strip():
        return False
    needle = normalize_tag_for_compare(marker_tag.strip())
    if not needle:
        return False
    sk = (service_key or "").strip()
    if sk:
        return needle in existing_storage_tag_keys(metadata, sk)
    return needle in all_normalized_storage_tag_keys(metadata)


def coalesce_wd_result_tag_strings(
    result: dict,
    *,
    general_prefix: str = "",
    character_prefix: str = "character:",
    rating_prefix: str = "rating:",
) -> list[str]:
    """Build the flat tag list the same way the results screen / apply route expect."""
    raw = result.get("tags")
    if raw is not None and len(raw) > 0:
        return list(raw)
    raw = result.get("formatted_tags")
    if raw is not None and len(raw) > 0:
        return list(raw)
    tags: list[str] = []
    gp = general_prefix or ""
    for name in (result.get("general_tags") or {}):
        if not isinstance(name, str):
            continue
        fmt = name.replace("_", " ")
        tags.append(f"{gp}{fmt}" if gp else fmt)
    cp = character_prefix or ""
    for name in (result.get("character_tags") or {}):
        if not isinstance(name, str):
            continue
        tags.append(f"{cp}{name.replace('_', ' ')}")
    rt = result.get("rating_tags") or {}
    if isinstance(rt, dict) and rt:
        top = max(rt, key=rt.get)
        tags.append(f"{rating_prefix or ''}{top}")
    return tags


def prune_wd_result_to_pending_tags(
    result: dict,
    pending_tags: list[str],
    *,
    general_prefix: str = "",
    character_prefix: str = "character:",
    rating_prefix: str = "rating:",
) -> None:
    """Mutate ``result`` so ``tags`` / structured fields match only ``pending_tags`` (for UI + apply)."""
    pending = list(pending_tags)
    result["tags"] = pending
    result["formatted_tags"] = list(pending)
    pend_norm = {normalize_tag_for_compare(t) for t in pending}

    gp = general_prefix or ""
    newg: dict = {}
    for name, conf in (result.get("general_tags") or {}).items():
        if not isinstance(name, str):
            continue
        fmt = name.replace("_", " ")
        tagged = f"{gp}{fmt}" if gp else fmt
        if normalize_tag_for_compare(tagged) in pend_norm:
            newg[name] = conf
    result["general_tags"] = newg

    newc: dict = {}
    cp = character_prefix or ""
    for name, conf in (result.get("character_tags") or {}).items():
        if not isinstance(name, str):
            continue
        tagged = f"{cp}{name.replace('_', ' ')}"
        if normalize_tag_for_compare(tagged) in pend_norm:
            newc[name] = conf
    result["character_tags"] = newc

    rt = result.get("rating_tags") or {}
    newr: dict = {}
    if isinstance(rt, dict) and rt:
        top = max(rt, key=rt.get)
        tagged = f"{rating_prefix or ''}{top}"
        if normalize_tag_for_compare(tagged) in pend_norm:
            newr = {k: v for k, v in rt.items() if k == top}
    result["rating_tags"] = newr


def filter_new_tags(
    proposed: list[str],
    existing_keys: set[str],
) -> tuple[list[str], int]:
    """Return (tags not already on file, count of skipped duplicates).

    ``existing_keys`` is not mutated; duplicates inside ``proposed`` are collapsed.
    """
    seen = set(existing_keys)
    new: list[str] = []
    skipped = 0
    for t in proposed:
        k = normalize_tag_for_compare(t)
        if k in seen:
            skipped += 1
            continue
        new.append(t)
        seen.add(k)
    return new, skipped

"""Hydrus storage_tags vs proposed tag deduplication helpers."""

from backend.hydrus.tag_merge import (
    all_normalized_storage_tag_keys,
    build_wd_model_marker,
    coalesce_wd_result_tag_strings,
    dedupe_wd_model_markers_in_tags,
    existing_storage_tag_keys,
    filter_new_tags,
    inference_skip_decision,
    marker_present_on_file,
    max_wd_marker_tier_on_file,
    model_capability_tier,
    normalize_tag_for_compare,
    prune_wd_result_to_pending_tags,
)


def test_normalize_tag_for_compare():
    assert normalize_tag_for_compare("Blue_Hair") == "blue hair"
    assert normalize_tag_for_compare("  rating:general ") == "rating:general"
    assert normalize_tag_for_compare("wd14:Foo-Bar") == "wd14:foo bar"


def test_existing_storage_tag_keys_flattens_statuses():
    meta = {
        "tags": {
            "deadbeef": {
                "storage_tags": {
                    "0": ["blonde_hair", "1girl"],
                    "1": ["bodysuit"],
                },
                "display_tags": {},
            }
        }
    }
    keys = existing_storage_tag_keys(meta, "deadbeef")
    assert "blonde hair" in keys
    assert "1girl" in keys
    assert "bodysuit" in keys


def test_filter_new_tags_skips_existing_and_dupes_in_proposed():
    existing = {"blue hair", "1girl"}
    new, skipped = filter_new_tags(
        ["blue_hair", "red hair", "blue hair", "red hair"],
        existing,
    )
    assert new == ["red hair"]
    assert skipped == 3


def test_existing_missing_service_returns_empty():
    assert existing_storage_tag_keys({"tags": {}}, "nope") == set()


def test_build_wd_model_marker_default_and_template():
    assert build_wd_model_marker("wd-vit-tagger-v3", "") == "wd14:wd-vit-tagger-v3"
    assert build_wd_model_marker("m", "x:{model_name}:y") == "x:m:y"


def test_coalesce_prefers_tags_array_then_dicts():
    r = {"tags": ["x"], "general_tags": {"y": 1.0}}
    assert coalesce_wd_result_tag_strings(r) == ["x"]
    r2 = {"general_tags": {"foo_bar": 0.9}, "character_tags": {}, "rating_tags": {}}
    assert coalesce_wd_result_tag_strings(r2) == ["foo bar"]


def test_prune_structured_matches_pending_list():
    r = {
        "general_tags": {"a": 0.5, "b": 0.6},
        "character_tags": {"c": 0.7},
        "rating_tags": {},
    }
    kw = {"general_prefix": "", "character_prefix": "character:", "rating_prefix": "rating:"}
    prune_wd_result_to_pending_tags(r, ["a", "character:c"], **kw)
    assert r["tags"] == ["a", "character:c"]
    assert "a" in r["general_tags"] and "b" not in r["general_tags"]
    assert "c" in r["character_tags"]


def test_marker_present_treats_hyphens_and_underscores_as_equivalent():
    meta = {
        "tags": {
            "svc": {
                "storage_tags": {"0": ["wd14:wd_vit_tagger_v3"]},
                "display_tags": {},
            },
        },
    }
    assert marker_present_on_file(meta, "wd14:wd-vit-tagger-v3", "svc")


def test_dedupe_wd_model_markers_keeps_canonical_and_counts_stale():
    canon = "wd14:model-a"
    tags, n = dedupe_wd_model_markers_in_tags(
        ["x", "wd14:model-b", "y", "wd14:legacy-z"],
        canon,
    )
    assert n == 2
    assert tags == ["x", "y", canon]


def test_dedupe_wd_model_markers_collapses_duplicate_canonical():
    canon = "wd14:my-model"
    tags, n = dedupe_wd_model_markers_in_tags(
        ["wd14:my_model", canon, "z"],
        canon,
    )
    assert n == 1
    assert tags == [canon, "z"]


def test_all_normalized_storage_tag_keys_unions_services():
    meta = {
        "tags": {
            "a": {"storage_tags": {"0": ["foo"]}, "display_tags": {}},
            "b": {"storage_tags": {"0": ["bar", "wd14:x"]}, "display_tags": {}},
        },
    }
    keys = all_normalized_storage_tag_keys(meta)
    assert "foo" in keys
    assert "bar" in keys
    assert "wd14:x" in keys


def test_marker_present_on_file_service_scoped_and_any_service():
    meta = {
        "tags": {
            "svc_a": {"storage_tags": {"0": ["wd14:foo"]}, "display_tags": {}},
            "svc_b": {"storage_tags": {"0": ["other"]}, "display_tags": {}},
        }
    }
    assert marker_present_on_file(meta, "wd14:foo", "svc_a")
    assert not marker_present_on_file(meta, "wd14:foo", "svc_b")
    assert marker_present_on_file(meta, "WD14:FOO", "")
    assert not marker_present_on_file(meta, "wd14:missing", "")


def test_model_capability_tier_known_and_unknown():
    assert model_capability_tier("wd-vit-tagger-v3") == 1
    assert model_capability_tier("wd-eva02-large-tagger-v3") == 4
    assert model_capability_tier("wd_vit_tagger_v3") == 1
    assert model_capability_tier("custom-unknown-model") == 0


def test_max_wd_marker_tier_on_file():
    meta = {
        "tags": {
            "sk": {
                "storage_tags": {"0": ["wd14:wd-vit-tagger-v3", "wd14:wd-eva02-large-tagger-v3"]},
                "display_tags": {},
            },
        },
    }
    t, slug = max_wd_marker_tier_on_file(meta, "sk", "wd14:")
    assert t == 4
    assert "eva02" in (slug or "")


def test_inference_skip_same_marker_before_higher_tier():
    meta = {
        "tags": {
            "sk": {"storage_tags": {"0": ["wd14:wd-vit-tagger-v3"]}, "display_tags": {}},
        },
    }
    skip, reason = inference_skip_decision(
        meta,
        current_model="wd-vit-tagger-v3",
        canonical_marker="wd14:wd-vit-tagger-v3",
        skip_same_model_marker=True,
        skip_if_higher_tier_model=True,
        service_key="sk",
        marker_prefix="wd14:",
    )
    assert skip and reason == "wd_model_marker_present"


def test_inference_skip_higher_tier_when_file_has_heavier_model():
    meta = {
        "tags": {
            "sk": {"storage_tags": {"0": ["wd14:wd-eva02-large-tagger-v3"]}, "display_tags": {}},
        },
    }
    skip, reason = inference_skip_decision(
        meta,
        current_model="wd-vit-tagger-v3",
        canonical_marker="wd14:wd-vit-tagger-v3",
        skip_same_model_marker=False,
        skip_if_higher_tier_model=True,
        service_key="sk",
        marker_prefix="wd14:",
    )
    assert skip and reason == "wd_skip_higher_tier_model_present"


def test_inference_runs_when_current_model_is_heavier_than_marker():
    meta = {
        "tags": {
            "sk": {"storage_tags": {"0": ["wd14:wd-vit-tagger-v3"]}, "display_tags": {}},
        },
    }
    skip, reason = inference_skip_decision(
        meta,
        current_model="wd-eva02-large-tagger-v3",
        canonical_marker="wd14:wd-eva02-large-tagger-v3",
        skip_same_model_marker=False,
        skip_if_higher_tier_model=True,
        service_key="sk",
        marker_prefix="wd14:",
    )
    assert not skip and reason is None

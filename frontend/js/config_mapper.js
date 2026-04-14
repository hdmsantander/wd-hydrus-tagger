/**
 * Shared config-to-UI mapper used by app bootstrap and settings modal.
 */

import { $, setInputValueIfPresent } from './utils/dom.js';
import { clampHydrusMetadataChunkSize } from './utils/hydrus.js';
import { setState } from './state.js';

function setSelectByValue(selectEl, value) {
    if (!selectEl || value == null) return;
    if ([...selectEl.options].some((o) => o.value === value)) {
        selectEl.value = value;
    }
}

export function applySharedConfigToUi(cfg, { syncIncrementalVisibility } = {}) {
    if (!cfg || typeof cfg !== 'object') return;
    setInputValueIfPresent('#input-general-prefix', cfg.general_tag_prefix || '');
    setInputValueIfPresent('#input-character-prefix', cfg.character_tag_prefix || 'character:');
    setInputValueIfPresent('#input-rating-prefix', cfg.rating_tag_prefix || 'rating:');
    const gpu = $('#check-gpu');
    if (gpu) gpu.checked = cfg.use_gpu || false;

    const inc = $('#check-incremental-hydrus');
    const applyN = $('#input-config-apply-every');
    if (cfg.apply_tags_every_n != null && inc && applyN) {
        const n = Number(cfg.apply_tags_every_n);
        inc.checked = n > 0;
        applyN.value = n > 0 ? String(n) : '8';
    }
    if (typeof syncIncrementalVisibility === 'function') {
        syncIncrementalVisibility();
    }

    const cHi = $('#check-wd-skip-higher-tier');
    if (cHi && cfg.wd_skip_if_higher_tier_model_present != null) {
        cHi.checked = cfg.wd_skip_if_higher_tier_model_present !== false;
    }
    const cSkip = $('#check-wd-skip-marker');
    if (cSkip && cfg.wd_skip_inference_if_marker_present != null) {
        cSkip.checked = cfg.wd_skip_inference_if_marker_present !== false;
    }
    const cApp = $('#check-wd-append-marker');
    if (cApp && cfg.wd_append_model_marker_tag != null) {
        cApp.checked = cfg.wd_append_model_marker_tag !== false;
    }
    const tpl = $('#input-wd-marker-template');
    if (tpl && cfg.wd_model_marker_template != null) tpl.value = cfg.wd_model_marker_template || '';
    const pfx = $('#input-wd-marker-prefix');
    if (pfx && cfg.wd_model_marker_prefix != null) {
        pfx.value = cfg.wd_model_marker_prefix || 'wd14:';
    }

    setSelectByValue($('#select-model'), cfg.default_model);
    setSelectByValue($('#select-settings-default-model'), cfg.default_model);

    const mcs = cfg.hydrus_metadata_chunk_size;
    if (mcs != null && Number.isFinite(Number(mcs))) {
        setState({ hydrusMetadataChunkSize: clampHydrusMetadataChunkSize(mcs) });
    }
}


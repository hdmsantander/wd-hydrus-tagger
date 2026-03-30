/**
 * Main application initialization.
 */

import { api, setFetchNetworkErrorHandler } from './api.js';
import { $ } from './utils/dom.js';
import { setState } from './state.js';
import { initConnection } from './components/connection.js';
import { initGallery } from './components/gallery.js';
import { initTagger } from './components/tagger.js';
import { initSettings } from './components/settings.js';
import { notifyFetchFailed, startServerWatch } from './server_offline.js';
import { initLayoutSidebar } from './layout.js';

async function loadAndApplyConfig() {
    let result;
    try {
        result = await api.getConfig();
    } catch {
        return;
    }
    if (!result.success) return;
    const cfg = result.config;

    // Thresholds
    $('#slider-general').value = cfg.general_threshold;
    $('#val-general').textContent = cfg.general_threshold.toFixed(2);
    $('#slider-character').value = cfg.character_threshold;
    $('#val-character').textContent = cfg.character_threshold.toFixed(2);

    // Model (Tagger panel + Settings default selector when modal exists)
    if (cfg.default_model) {
        const taggerSel = $('#select-model');
        if (taggerSel && [...taggerSel.options].some((o) => o.value === cfg.default_model)) {
            taggerSel.value = cfg.default_model;
        }
        const setDef = $('#select-settings-default-model');
        if (setDef && [...setDef.options].some((o) => o.value === cfg.default_model)) {
            setDef.value = cfg.default_model;
        }
    }

    // Prefixes & GPU (settings modal inputs)
    $('#input-general-prefix').value = cfg.general_tag_prefix || '';
    $('#input-character-prefix').value = cfg.character_tag_prefix || 'character:';
    $('#input-rating-prefix').value = cfg.rating_tag_prefix || 'rating:';
    $('#check-gpu').checked = cfg.use_gpu || false;

    const ib = $('#input-inference-batch');
    if (ib && cfg.batch_size != null) {
        ib.value = String(cfg.batch_size);
    }
    const cHi = $('#check-wd-skip-higher-tier');
    if (cHi && cfg.wd_skip_if_higher_tier_model_present != null) {
        cHi.checked = cfg.wd_skip_if_higher_tier_model_present !== false;
    }

    const mcs = cfg.hydrus_metadata_chunk_size;
    if (mcs != null && Number.isFinite(Number(mcs))) {
        setState({ hydrusMetadataChunkSize: Math.max(32, Math.min(2048, Math.floor(Number(mcs)))) });
    }
}

document.addEventListener('DOMContentLoaded', () => {
    setFetchNetworkErrorHandler(notifyFetchFailed);
    initLayoutSidebar();
    initConnection();
    initGallery();
    initTagger();
    initSettings();
    startServerWatch();
    void loadAndApplyConfig();
});

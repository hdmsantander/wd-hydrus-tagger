/**
 * Main application initialization.
 */

import { api, setFetchNetworkErrorHandler } from './api.js';
import { $ } from './utils/dom.js';
import { applySharedConfigToUi } from './config_mapper.js';
import { initConnection } from './components/connection.js';
import { initGallery } from './components/gallery.js';
import { initImageViewer } from './components/viewer.js';
import { initTagger } from './components/tagger.js';
import { initSettings, syncIncrementalHydrusApplyEveryVisibility } from './components/settings.js';
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

    applySharedConfigToUi(cfg, {
        syncIncrementalVisibility: syncIncrementalHydrusApplyEveryVisibility,
    });
}

document.addEventListener('DOMContentLoaded', () => {
    setFetchNetworkErrorHandler(notifyFetchFailed);
    initLayoutSidebar();
    initConnection();
    initGallery();
    initImageViewer();
    initTagger();
    initSettings();
    startServerWatch();
    void loadAndApplyConfig();
});

/**
 * Main application initialization.
 */

import { api, setFetchNetworkErrorHandler } from './api.js';
import { $ } from './utils/dom.js';
import { applySharedConfigToUi } from './config_mapper.js';
import { setState } from './state.js';
import { syncHydrusWebToolbarLink } from './utils/hydrus_web.js';
import { initConnection } from './components/connection.js';
import { initGallery } from './components/gallery.js';
import { initImageViewer } from './components/viewer.js';
import { initTagger } from './components/tagger.js';
import { initFace } from './components/face.js';
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

    setState({ hydrusWebUrl: String(cfg.hydrus_web_url || '').trim() });
    syncHydrusWebToolbarLink();

    if (cfg.face_det_threshold != null) {
        const det = $('#slider-face-det');
        const detVal = $('#val-face-det');
        if (det) det.value = cfg.face_det_threshold;
        if (detVal) detVal.textContent = Number(cfg.face_det_threshold).toFixed(2);
    }
    if (cfg.face_recognition_max_distance != null) {
        const dist = $('#slider-face-distance');
        const distVal = $('#val-face-distance');
        if (dist) dist.value = cfg.face_recognition_max_distance;
        if (distVal) distVal.textContent = Number(cfg.face_recognition_max_distance).toFixed(2);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    setFetchNetworkErrorHandler(notifyFetchFailed);
    initLayoutSidebar();
    initConnection();
    initGallery();
    initImageViewer();
    initTagger();
    initFace();
    initSettings();
    startServerWatch();
    void loadAndApplyConfig();
});

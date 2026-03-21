/**
 * Main application initialization.
 */

import { api } from './api.js';
import { $ } from './utils/dom.js';
import { initConnection } from './components/connection.js';
import { initGallery } from './components/gallery.js';
import { initTagger } from './components/tagger.js';
import { initSettings } from './components/settings.js';

async function loadAndApplyConfig() {
    const result = await api.getConfig();
    if (!result.success) return;
    const cfg = result.config;

    // Thresholds
    $('#slider-general').value = cfg.general_threshold;
    $('#val-general').textContent = cfg.general_threshold.toFixed(2);
    $('#slider-character').value = cfg.character_threshold;
    $('#val-character').textContent = cfg.character_threshold.toFixed(2);

    // Model
    if (cfg.default_model) {
        $('#select-model').value = cfg.default_model;
    }

    // Prefixes & GPU (settings modal inputs)
    $('#input-general-prefix').value = cfg.general_tag_prefix || '';
    $('#input-character-prefix').value = cfg.character_tag_prefix || 'character:';
    $('#input-rating-prefix').value = cfg.rating_tag_prefix || 'rating:';
    $('#check-gpu').checked = cfg.use_gpu || false;
}

document.addEventListener('DOMContentLoaded', () => {
    initConnection();
    initGallery();
    initTagger();
    initSettings();
    loadAndApplyConfig();
});

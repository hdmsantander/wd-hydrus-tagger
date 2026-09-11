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
    setSelectByValue($('#select-gpu-backend'), cfg.gpu_backend || 'auto');

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

    setInputValueIfPresent('#input-hydrus-web-url', cfg.hydrus_web_url ?? '');

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
    const skipDet = $('#check-face-skip-detected');
    if (skipDet && cfg.face_skip_if_detected != null) {
        skipDet.checked = cfg.face_skip_if_detected !== false;
    }
    const skipDb = $('#check-face-skip-in-db');
    if (skipDb && cfg.face_skip_if_in_db != null) {
        skipDb.checked = cfg.face_skip_if_in_db !== false;
    }
    setInputValueIfPresent('#input-face-target-service', cfg.face_target_tag_service ?? '');
    setInputValueIfPresent('#input-face-person-prefix', cfg.face_person_tag_prefix || 'person:');
    setInputValueIfPresent('#input-face-marker-detected', cfg.face_marker_detected || 'ai face detected');
    setInputValueIfPresent('#input-face-marker-not-visible', cfg.face_marker_not_visible || 'face not visible');
    setInputValueIfPresent('#input-face-marker-recognized', cfg.face_marker_recognized || 'face ai generated tags');
    if (cfg.face_video_frame_count != null) {
        setInputValueIfPresent('#input-face-video-frames', String(cfg.face_video_frame_count));
    }

    const mcs = cfg.hydrus_metadata_chunk_size;
    if (mcs != null && Number.isFinite(Number(mcs))) {
        setState({ hydrusMetadataChunkSize: clampHydrusMetadataChunkSize(mcs) });
    }
}


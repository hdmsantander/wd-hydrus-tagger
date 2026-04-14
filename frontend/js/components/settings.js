/**
 * Settings modal component.
 */

import { api } from '../api.js';
import { applySharedConfigToUi } from '../config_mapper.js';
import { setState } from '../state.js';
import { $, el, show, hide } from '../utils/dom.js';
import { showServerOfflineScreen } from '../server_offline.js';

function syncTaggerPanelDefaultModel() {
    const m = $('#select-settings-default-model')?.value;
    const taggerSel = $('#select-model');
    if (m && taggerSel && [...taggerSel.options].some((o) => o.value === m)) {
        taggerSel.value = m;
    }
}

/** Show the N field only while “Push tags to Hydrus while tagging” is enabled. */
export function syncIncrementalHydrusApplyEveryVisibility() {
    const wrap = $('#wrap-config-apply-every');
    const chk = $('#check-incremental-hydrus');
    if (!wrap || !chk) return;
    wrap.hidden = !chk.checked;
}

/** Align Settings defaults with the Tagger sidebar (model + tag service display name). */
function syncAdvancedDefaultsFromSidebar() {
    const taggerModel = $('#select-model')?.value;
    const sdm = $('#select-settings-default-model');
    if (taggerModel && sdm && [...sdm.options].some((o) => o.value === taggerModel)) {
        sdm.value = taggerModel;
    }
    const svc = $('#select-service');
    const tts = $('#input-target-tag-service');
    if (svc && tts && svc.selectedIndex >= 0) {
        const opt = svc.options[svc.selectedIndex];
        const name = String(opt?.textContent || '').trim();
        if (name) tts.value = name;
    }
}

async function loadModels() {
    const list = $('#model-list');
    const summary = $('#model-status-summary');
    list.innerHTML = '<div class="info-text">Loading…</div>';
    summary.textContent = '';

    const result = await api.listModels();
    list.innerHTML = '';

    if (!result.success) {
        list.innerHTML = '<div class="info-text">Could not load model list</div>';
        return;
    }

    const loaded = result.loaded_model || null;
    const def = result.default_model || '';
    if (loaded) {
        summary.textContent =
            `Loaded in memory: ${loaded}` + (def && def !== loaded ? ` (config default: ${def})` : '');
    } else {
        summary.textContent =
            'No model loaded in memory yet — the first tagging run loads ONNX into RAM.' +
            (def ? ` Config default: ${def}.` : '');
    }

    for (const model of result.models) {
        const badges = [];
        if (model.loaded_in_memory) {
            badges.push(el('span', { className: 'model-status loaded', textContent: 'In memory' }));
        }
        if (model.downloaded) {
            badges.push(el('span', { className: 'model-status downloaded', textContent: 'On disk' }));
            if (model.cache_ok === false) {
                badges.push(el('span', {
                    className: 'model-status cache-warn',
                    textContent: 'Cache check failed',
                    title: (model.cache_issues || []).join('; ') || 'Run Verify cached models',
                }));
            } else if (model.cache_ok === true && model.manifest_present === false) {
                badges.push(el('span', {
                    className: 'model-status cache-note',
                    textContent: 'No manifest',
                    title: 'Files look valid; manifest will be added on next download or load.',
                }));
            }
        } else {
            badges.push(el('button', {
                className: 'btn btn-sm btn-primary',
                textContent: 'Download',
                onClick: async (e) => {
                    e.target.disabled = true;
                    e.target.textContent = 'Downloading…';
                    const dlResult = await api.downloadModel(model.name);
                    if (dlResult.success) {
                        await loadModels();
                    } else {
                        e.target.textContent = 'Download';
                        e.target.disabled = false;
                        alert('Download failed: ' + dlResult.error);
                    }
                },
            }));
        }

        const rev = model.revision_sha && String(model.revision_sha).length >= 7
            ? ` · Hub rev ${String(model.revision_sha).slice(0, 7)}`
            : '';
        const item = el('div', { className: 'model-item' }, [
            el('span', { className: 'model-name', textContent: model.name }),
            el('span', { className: 'model-item-badges' }, badges),
            el('span', { className: 'model-repo', textContent: (model.repo || '') + rev }),
        ]);
        list.appendChild(item);
    }
}

async function loadAppStatus() {
    const summary = $('#app-status-summary');
    const hint = $('#shutdown-disabled-hint');
    const btnStop = $('#btn-stop-server');
    if (!summary) return;
    let res;
    try {
        res = await api.getAppStatus();
    } catch {
        summary.textContent = 'Could not reach server (is it still running?).';
        return;
    }
    if (!res.success) {
        summary.textContent = 'Could not load server status.';
        return;
    }
    const m = res.loaded_model;
    summary.textContent =
        `Active tagging sessions: ${res.active_tagging_sessions}. Model in RAM: ${m || 'none'}. `
        + `Models directory: ${res.models_dir || ''}. Multi-tab: other tabs can watch progress read-only.`;

    const allowShutdown = res.allow_ui_shutdown !== false;
    if (btnStop) {
        btnStop.style.display = allowShutdown ? 'inline-block' : 'none';
        btnStop.disabled = !allowShutdown;
    }
    if (hint) {
        hint.style.display = allowShutdown ? 'none' : 'block';
    }
}

async function loadConfig() {
    const result = await api.getConfig();
    if (!result.success) return;
    const cfg = result.config;

    applySharedConfigToUi(cfg, {
        syncIncrementalVisibility: syncIncrementalHydrusApplyEveryVisibility,
    });

    const sdm = $('#select-settings-default-model');
    if (sdm && cfg.default_model && [...sdm.options].some((o) => o.value === cfg.default_model)) {
        sdm.value = cfg.default_model;
    }

    const tts = $('#input-target-tag-service');
    if (tts) tts.value = cfg.target_tag_service || '';

    $('#slider-general').value = cfg.general_threshold;
    $('#val-general').textContent = cfg.general_threshold.toFixed(2);
    $('#slider-character').value = cfg.character_threshold;
    $('#val-character').textContent = cfg.character_threshold.toFixed(2);

    if (cfg.batch_size != null) {
        $('#input-batch-size').value = String(cfg.batch_size);
    }
    if (cfg.cpu_intra_op_threads != null) {
        $('#input-cpu-intra').value = String(cfg.cpu_intra_op_threads);
    }
    if (cfg.cpu_inter_op_threads != null) {
        $('#input-cpu-inter').value = String(cfg.cpu_inter_op_threads);
    }
    if (cfg.hydrus_download_parallel != null) {
        $('#input-hydrus-parallel').value = String(cfg.hydrus_download_parallel);
    }
    const hmc = $('#input-hydrus-metadata-chunk');
    if (hmc && cfg.hydrus_metadata_chunk_size != null) {
        hmc.value = String(cfg.hydrus_metadata_chunk_size);
    }
    const stb = $('#input-tagging-skip-tail-batch');
    if (stb && cfg.tagging_skip_tail_batch_size != null) {
        stb.value = String(cfg.tagging_skip_tail_batch_size);
    }
    const httpB = $('#input-apply-http-batch');
    if (httpB && cfg.apply_tags_http_batch_size != null) {
        httpB.value = String(cfg.apply_tags_http_batch_size);
    }

    const cOrt = $('#check-ort-enable-profiling');
    if (cOrt) cOrt.checked = cfg.ort_enable_profiling === true;
    const opd = $('#input-ort-profile-dir');
    if (opd && cfg.ort_profile_dir != null) {
        opd.value = String(cfg.ort_profile_dir);
    }

    const cShut = $('#check-allow-ui-shutdown');
    if (cShut) cShut.checked = cfg.allow_ui_shutdown !== false;
}

export function initSettings() {
    $('#check-incremental-hydrus')?.addEventListener('change', syncIncrementalHydrusApplyEveryVisibility);
    syncIncrementalHydrusApplyEveryVisibility();

    $('#btn-settings').addEventListener('click', () => {
        show('#modal-settings');
        void Promise.all([loadModels(), loadConfig(), loadAppStatus()]).then(() => {
            syncAdvancedDefaultsFromSidebar();
        });
    });

    $('#btn-refresh-models').addEventListener('click', () => {
        void Promise.all([loadModels(), loadAppStatus()]);
    });

    $('#btn-verify-models')?.addEventListener('click', async () => {
        const btn = $('#btn-verify-models');
        const remote = $('#check-verify-remote')?.checked === true;
        btn.disabled = true;
        const prev = btn.textContent;
        btn.textContent = 'Verifying…';
        try {
            const r = await api.verifyModels({ checkRemote: remote });
            if (!r.success) {
                alert('Verify failed: ' + (r.error || 'unknown error'));
                return;
            }
            const lines = (r.results || []).map((row) => {
                const st = row.ok ? 'OK' : 'FAIL';
                const iss = (row.issues && row.issues.length) ? ` — ${row.issues.join(', ')}` : '';
                const hub = row.stale_on_hub ? ' [newer on Hub]' : '';
                return `${row.name}: ${st}${iss}${hub}`;
            });
            alert(
                `Models directory:\n${r.models_dir || ''}\n\n`
                + lines.join('\n'),
            );
            await Promise.all([loadModels(), loadAppStatus()]);
        } finally {
            btn.disabled = false;
            btn.textContent = prev;
        }
    });

    $('#btn-close-settings').addEventListener('click', () => {
        hide('#modal-settings');
    });

    $('#modal-settings .modal-backdrop').addEventListener('click', () => {
        hide('#modal-settings');
    });

    const btnStop = $('#btn-stop-server');
    if (btnStop) {
        btnStop.addEventListener('click', async () => {
            const res = await api.shutdownApp();
            if (res.success) {
                hide('#modal-settings');
                showServerOfflineScreen({ reason: 'ui_shutdown' });
            } else {
                alert(res.error || 'Shutdown failed');
            }
        });
    }

    $('#btn-save-settings').addEventListener('click', async () => {
        const batchSize = parseInt($('#input-batch-size').value, 10);
        const cpuIntra = parseInt($('#input-cpu-intra').value, 10);
        const cpuInter = parseInt($('#input-cpu-inter').value, 10);
        const hyPar = parseInt($('#input-hydrus-parallel').value, 10);
        const metaChunk = parseInt($('#input-hydrus-metadata-chunk')?.value || '512', 10);
        const skipTailBatch = parseInt($('#input-tagging-skip-tail-batch')?.value || '512', 10);
        const incrementalUi = $('#check-incremental-hydrus')?.checked === true;
        const appEveryRaw = parseInt($('#input-config-apply-every').value, 10);
        const appEvery = incrementalUi ? appEveryRaw : 0;
        const httpBatch = parseInt($('#input-apply-http-batch')?.value || '100', 10);
        if (
            !Number.isFinite(batchSize) || batchSize < 1 || batchSize > 256
            || !Number.isFinite(cpuIntra) || cpuIntra < 1 || cpuIntra > 64
            || !Number.isFinite(cpuInter) || cpuInter < 1 || cpuInter > 16
            || !Number.isFinite(hyPar) || hyPar < 1 || hyPar > 32
            || !Number.isFinite(metaChunk) || metaChunk < 32 || metaChunk > 2048
            || !Number.isFinite(skipTailBatch) || skipTailBatch < 32 || skipTailBatch > 2048
            || !Number.isFinite(appEveryRaw) || appEveryRaw < 0 || appEveryRaw > 256
            || !Number.isFinite(httpBatch) || httpBatch < 1 || httpBatch > 512
        ) {
            alert('Invalid number (check limits: batch 1–256, apply HTTP 1–512, etc.)');
            return;
        }
        if (incrementalUi && (!Number.isFinite(appEveryRaw) || appEveryRaw < 1)) {
            alert('When push tags to Hydrus while tagging is on, set N to at least 1.');
            return;
        }
        const genTh = parseFloat($('#slider-general').value);
        const charTh = parseFloat($('#slider-character').value);
        if (
            !Number.isFinite(genTh) || genTh < 0 || genTh > 1
            || !Number.isFinite(charTh) || charTh < 0 || charTh > 1
        ) {
            alert('General and character thresholds must be numbers between 0 and 1.');
            return;
        }
        const updates = {
            general_tag_prefix: $('#input-general-prefix').value,
            character_tag_prefix: $('#input-character-prefix').value,
            rating_tag_prefix: $('#input-rating-prefix').value,
            use_gpu: $('#check-gpu').checked,
            general_threshold: genTh,
            character_threshold: charTh,
            batch_size: batchSize,
            cpu_intra_op_threads: cpuIntra,
            cpu_inter_op_threads: cpuInter,
            hydrus_download_parallel: hyPar,
            hydrus_metadata_chunk_size: metaChunk,
            tagging_skip_tail_batch_size: skipTailBatch,
            apply_tags_every_n: appEvery,
            default_model: $('#select-settings-default-model')?.value,
            target_tag_service: ($('#input-target-tag-service')?.value || '').trim(),
            wd_skip_inference_if_marker_present: $('#check-wd-skip-marker')?.checked ?? true,
            wd_skip_if_higher_tier_model_present: $('#check-wd-skip-higher-tier')?.checked ?? true,
            wd_append_model_marker_tag: $('#check-wd-append-marker')?.checked ?? true,
            wd_model_marker_template: ($('#input-wd-marker-template')?.value || '').trim(),
            wd_model_marker_prefix: ($('#input-wd-marker-prefix')?.value || '').trim() || 'wd14:',
            apply_tags_http_batch_size: httpBatch,
            allow_ui_shutdown: $('#check-allow-ui-shutdown')?.checked ?? true,
            shutdown_tagging_grace_seconds: 0,
            ort_enable_profiling: $('#check-ort-enable-profiling')?.checked ?? false,
            ort_profile_dir: ($('#input-ort-profile-dir')?.value || '').trim() || './ort_traces',
        };
        const result = await api.updateConfig(updates);
        if (result.success) {
            syncTaggerPanelDefaultModel();
            setState({ hydrusMetadataChunkSize: metaChunk });
            await loadAppStatus();
            const modal = $('#modal-settings');
            const content = modal?.querySelector('.modal-content');
            const fb = $('#settings-save-feedback');
            if (fb) {
                fb.textContent = 'Settings saved';
                fb.classList.add('settings-save-feedback--success');
            }
            modal?.classList.add('modal-settings-dismissing');
            content?.classList.add('modal-content--minimize-out');
            window.setTimeout(() => {
                hide('#modal-settings');
                modal?.classList.remove('modal-settings-dismissing');
                content?.classList.remove('modal-content--minimize-out');
                if (fb) {
                    fb.textContent = '';
                    fb.classList.remove('settings-save-feedback--success');
                }
            }, 480);
        } else if (result.error) {
            alert('Save failed: ' + JSON.stringify(result.error));
        } else {
            alert('Save failed');
        }
    });
}

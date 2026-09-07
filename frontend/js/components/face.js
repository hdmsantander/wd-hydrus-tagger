/**
 * AI face detection and unsupervised person tagging controls.
 */

import { api } from '../api.js';
import { getState, setState } from '../state.js';
import { $ } from '../utils/dom.js';
import {
    showProgress,
    hideProgress,
    updateProgress,
    setProgressActivityPhase,
    setProgressControlMode,
    requestProgressFrame,
} from './progress.js';
import { expectServerShutdownSoon } from '../server_offline.js';

let faceWsCancel = null;

function setFacePipelineStep(step) {
    const steps = document.querySelectorAll('#face-pipeline-steps .face-pipeline-step');
    for (const el of steps) {
        const name = el.getAttribute('data-step');
        el.classList.remove('is-active', 'is-done');
        if (!step) continue;
        if (name === step) el.classList.add('is-active');
        else if (name === 'detect' && step === 'recognize') el.classList.add('is-done');
    }
}

function clearFacePipelineStep() {
    setFacePipelineStep(null);
}

function faceProgressDetail(msg) {
    const body = facePhaseLabel(msg.phase, msg);
    if (msg.step_label && body) return `${msg.step_label} — ${body}`;
    if (msg.step_label) return String(msg.step_label);
    return body;
}

function faceActivityPhase(msg) {
    if (msg.pipeline === 'detect') {
        if (msg.step === 'scan' || msg.phase === 'detect') return 'inference';
        return 'load';
    }
    return msg.phase === 'metadata' || msg.phase === 'model' ? 'load' : 'inference';
}

function facePhaseLabel(phase, msg = {}) {
    if (msg.detail) return String(msg.detail);
    if (msg.skip_reason === 'video_excluded') return 'Video skipped (face detect is images only)';
    if (msg.skip_reason === 'marker_present') return 'Skipped (already face-tagged)';
    if (phase === 'metadata') return 'Loading file metadata…';
    if (phase === 'model') return 'Loading InsightFace model (downloads on first use)…';
    if (phase === 'detect') return null;
    return null;
}

export async function refreshFaceStatus() {
    const el = $('#face-db-stats');
    const hint = $('#face-model-hint');
    if (!el && !hint) return;
    const res = await api.faceStatus();
    if (!res.success) {
        if (el) el.textContent = res.error || 'Face status unavailable';
        if (hint) hint.textContent = '';
        return;
    }
    const st = res.status || {};
    if (el) {
        el.textContent =
            `${st.faces ?? 0} faces · ${st.persons ?? 0} persons · ${st.unassigned_faces ?? 0} unassigned · ${st.files_with_faces ?? 0} files`;
    }
    if (hint) {
        const disk = st.downloaded ? (st.cache_ok === false ? 'On disk (cache check failed)' : 'On disk') : 'Not downloaded';
        const ram = st.model_loaded ? `In memory (${st.active_provider || 'CPU'})` : 'Not loaded';
        const ort = Array.isArray(st.installed_ort_providers) && st.installed_ort_providers.length
            ? st.installed_ort_providers.join(', ')
            : '';
        const gpuLine = st.use_gpu
            ? `GPU ${st.gpu_backend}${ort ? ` · ${ort}` : ''}${st.cpu_fallback ? ' (CPU fallback)' : ''}`
            : 'CPU';
        hint.textContent = `${st.model_pack || 'buffalo_l'}: ${disk} · ${ram} · ${gpuLine}`;
    }
}

function resolveFaceServiceKey() {
    const sel = $('#select-face-service');
    if (sel && sel.value) return sel.value;
    const wd = $('#select-service');
    return wd?.value || '';
}

function populateFaceServiceSelect() {
    const select = $('#select-face-service');
    if (!select) return;
    const services = getState().services || [];
    select.innerHTML = '';
    for (const svc of services) {
        const opt = document.createElement('option');
        opt.value = svc.service_key;
        opt.textContent = svc.name;
        select.appendChild(opt);
    }
    const wd = $('#select-service');
    if (wd && wd.value) {
        select.value = wd.value;
    }
}

export function syncFaceServiceSelect() {
    populateFaceServiceSelect();
}

function clearFaceProgressHandlers() {
    const stop = $('#btn-progress-stop');
    if (stop) stop.onclick = null;
}

function showFaceProgress(total, title, detail) {
    setProgressControlMode({ controller: true });
    showProgress(total, { title, detail });
    for (const sel of ['#btn-progress-pause', '#btn-progress-resume', '#btn-progress-flush', '#btn-tuning-approve']) {
        const btn = $(sel);
        if (btn) btn.style.display = 'none';
    }
    const stop = $('#btn-progress-stop');
    if (stop) stop.style.display = 'inline-block';
    setProgressActivityPhase('load');
    if (stop) {
        stop.disabled = false;
        stop.textContent = 'Stop';
        stop.onclick = () => {
            stop.disabled = true;
            stop.textContent = 'Stopping…';
            faceWsCancel?.();
        };
    }
}

function startFaceDetectWs(fileIds, { tagAll = false } = {}) {
    if (!fileIds.length) {
        alert('No files selected.');
        return;
    }
    if (!resolveFaceServiceKey()) {
        alert('Choose a face tag service before detecting faces.');
        return;
    }
    setState({ processing: true });
    setFacePipelineStep('detect');
    const title = tagAll ? 'Step 1/2 — Detect faces (all results)' : 'Step 1/2 — Detect faces';
    showFaceProgress(fileIds.length, title, 'Detect 1/5 — Fetch metadata…');

    const threshold = parseFloat($('#slider-face-det')?.value || '0.6');
    const serviceKey = resolveFaceServiceKey();
    let lastProcessed = 0;

    const { ws, cancel, done } = api.startFaceDetectWebSocket(
        {
            file_ids: fileIds,
            service_key: serviceKey,
            det_threshold: threshold,
            replace_existing: $('#check-face-replace')?.checked ?? false,
        },
        {
            onStarted: (msg) => {
                requestProgressFrame(() => {
                    const tot = msg.total ?? fileIds.length;
                    updateProgress(0, tot, title, 'Starting detection…');
                    setProgressActivityPhase('load');
                });
            },
            onProgress: (msg) => {
                requestProgressFrame(() => {
                    lastProcessed = msg.processed ?? lastProcessed;
                    const cur = msg.processed ?? 0;
                    const tot = msg.total ?? fileIds.length;
                    const detail = faceProgressDetail(msg);
                    const providerNote = msg.active_provider ? ` · ${msg.active_provider}` : '';
                    const line = detail
                        || `Detected ${cur}/${tot} · last faces: ${msg.face_count ?? 0}${providerNote}`;
                    updateProgress(
                        cur,
                        tot,
                        title,
                        line,
                    );
                    setProgressActivityPhase(faceActivityPhase(msg), {
                        titleSuffix: msg.step_label || msg.active_provider || (msg.providers && msg.providers[0]) || '',
                    });
                });
            },
            onStopping: () => {
                requestProgressFrame(() => {
                    setProgressActivityPhase('stopping');
                    updateProgress(lastProcessed, fileIds.length, title, 'Stopping face detection…');
                });
            },
            onServerShuttingDown: (msg) => {
                expectServerShutdownSoon();
                requestProgressFrame(() => {
                    setProgressActivityPhase('stopping');
                    updateProgress(
                        lastProcessed,
                        fileIds.length,
                        'Server stopping',
                        msg.message || 'Face detection will cancel at the next file.',
                    );
                });
            },
        },
    );

    faceWsCancel = cancel;

    done
        .then((final) => {
            const faces = final?.faces_found ?? 0;
            const processed = final?.files_processed ?? fileIds.length;
            const videos = final?.videos_skipped ?? 0;
            const tagged = final?.marker_skipped ?? 0;
            const errN = final?.errors ?? 0;
            let tail = `Complete · ${faces} face(s) found`;
            if (videos) tail += ` · ${videos} video(s) skipped`;
            if (tagged) tail += ` · ${tagged} already tagged`;
            if (errN) tail += ` · ${errN} error(s)`;
            updateProgress(
                processed,
                fileIds.length,
                title,
                tail,
            );
            setProgressActivityPhase('done');
            setFacePipelineStep('recognize');
            document.querySelectorAll('#face-pipeline-steps .face-pipeline-step[data-step="detect"]')
                .forEach((el) => { el.classList.add('is-done'); el.classList.remove('is-active'); });
            void refreshFaceStatus();
        })
        .catch((err) => {
            alert('Face detection failed: ' + (err?.message || err));
        })
        .finally(() => {
            faceWsCancel = null;
            clearFaceProgressHandlers();
            clearFacePipelineStep();
            setState({ processing: false });
            hideProgress();
        });

    void ws;
}

async function runRecognize() {
    const serviceKey = resolveFaceServiceKey();
    if (!serviceKey) {
        alert('Choose a face tag service before recognizing persons.');
        return;
    }
    const maxDistance = parseFloat($('#slider-face-distance')?.value || '0.5');
    const stageHint = 'Stages: 20 → 5 → 3 → 1 faces per cluster';
    setState({ processing: true });
    setFacePipelineStep('recognize');
    showFaceProgress(1, 'Step 2/2 — Recognize persons', `Recognize 1/4 — Cluster (min 20 faces) · ${stageHint}`);
    setProgressActivityPhase('run');
    try {
        const res = await api.faceRecognize({
            service_key: serviceKey,
            max_distance: maxDistance,
            staged: true,
        });
        if (!res.success) {
            alert(res.error || 'Recognition failed');
            return;
        }
        const stages = Array.isArray(res.stages) ? res.stages : [];
        const stageSummary = stages.length
            ? stages.map((s) => `${s.step_label || `stage ${s.stage}`}: +${s.assigned}`).join(' · ')
            : '';
        updateProgress(
            1,
            1,
            'Step 2/2 — Recognize persons',
            `Tagged ${res.files_tagged ?? 0} file(s) · ${res.persons ?? 0} persons`
                + (stageSummary ? ` · ${stageSummary}` : ''),
        );
        setProgressActivityPhase('done');
        document.querySelectorAll('#face-pipeline-steps .face-pipeline-step[data-step="recognize"]')
            .forEach((el) => { el.classList.add('is-done'); el.classList.remove('is-active'); });
        await refreshFaceStatus();
    } finally {
        clearFaceProgressHandlers();
        clearFacePipelineStep();
        setState({ processing: false });
        hideProgress();
    }
}

export function initFace() {
    populateFaceServiceSelect();

    $('#slider-face-det')?.addEventListener('input', (e) => {
        const v = parseFloat(e.target.value);
        const lbl = $('#val-face-det');
        if (lbl) lbl.textContent = v.toFixed(2);
    });

    $('#slider-face-distance')?.addEventListener('input', (e) => {
        const v = parseFloat(e.target.value);
        const lbl = $('#val-face-distance');
        if (lbl) lbl.textContent = v.toFixed(2);
    });

    $('#btn-face-detect-selected')?.addEventListener('click', () => {
        const ids = [...getState().selectedIds];
        startFaceDetectWs(ids);
    });

    $('#btn-face-detect-all')?.addEventListener('click', () => {
        const ids = getState().fileIds || [];
        if (!ids.length) {
            alert('Run a search first.');
            return;
        }
        startFaceDetectWs(ids, { tagAll: true });
    });

    $('#btn-face-recognize')?.addEventListener('click', () => {
        void runRecognize();
    });

    $('#btn-face-reset')?.addEventListener('click', async () => {
        if (!confirm('Clear all person assignments? Embeddings are kept.')) return;
        const res = await api.faceReset();
        if (!res.success) alert(res.error || 'Reset failed');
        await refreshFaceStatus();
    });

    $('#btn-face-clean')?.addEventListener('click', async () => {
        const res = await api.faceClean();
        if (!res.success) {
            alert(res.error || 'Clean failed');
            return;
        }
        alert(`Removed ${res.removed ?? 0} orphaned face row(s).`);
        await refreshFaceStatus();
    });

    void refreshFaceStatus();
}

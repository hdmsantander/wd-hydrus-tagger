/**
 * AI face detection and unsupervised person tagging controls.
 */

import { api } from '../api.js';
import { getState, setState } from '../state.js';
import { $, show, hide } from '../utils/dom.js';
import {
    showProgress,
    hideProgress,
    updateProgress,
    setProgressActivityPhase,
    requestProgressFrame,
} from './progress.js';

let faceWsCancel = null;

async function refreshFaceStatus() {
    const el = $('#face-db-stats');
    if (!el) return;
    const res = await api.faceStatus();
    if (!res.success) {
        el.textContent = res.error || 'Face status unavailable';
        return;
    }
    const st = res.status || {};
    el.textContent =
        `${st.faces ?? 0} faces · ${st.persons ?? 0} persons · ${st.unassigned_faces ?? 0} unassigned · ${st.files_with_faces ?? 0} files`;
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

function startFaceDetectWs(fileIds, { tagAll = false } = {}) {
    if (!fileIds.length) {
        alert('No files selected.');
        return;
    }
    setState({ processing: true });
    showProgress({
        title: tagAll ? 'Detecting faces (all results)' : 'Detecting faces',
        detail: 'Loading InsightFace model…',
    });
    setProgressActivityPhase('inference');

    const threshold = parseFloat($('#slider-face-det')?.value || '0.6');
    const serviceKey = resolveFaceServiceKey();

    const { ws, cancel, done } = api.startFaceDetectWebSocket(
        {
            file_ids: fileIds,
            service_key: serviceKey,
            det_threshold: threshold,
            replace_existing: $('#check-face-replace')?.checked ?? false,
        },
        {
            onProgress: (msg) => {
                requestProgressFrame(() => {
                    const cur = msg.processed ?? 0;
                    const tot = msg.total ?? fileIds.length;
                    updateProgress(cur, tot, `Detected ${cur}/${tot} · last faces: ${msg.face_count ?? 0}`);
                    setProgressActivityPhase('inference');
                });
            },
        },
    );

    faceWsCancel = cancel;

    done
        .then((final) => {
            const faces = final?.faces_found ?? 0;
            updateProgress(
                final?.files_processed ?? fileIds.length,
                fileIds.length,
                `Complete · ${faces} face(s) found`,
            );
            setProgressActivityPhase('done');
            void refreshFaceStatus();
        })
        .catch((err) => {
            alert('Face detection failed: ' + (err?.message || err));
        })
        .finally(() => {
            faceWsCancel = null;
            setState({ processing: false });
            hideProgress();
        });

    void ws;
}

async function runRecognize() {
    const serviceKey = resolveFaceServiceKey();
    const maxDistance = parseFloat($('#slider-face-distance')?.value || '0.5');
    setState({ processing: true });
    showProgress({ title: 'Recognizing persons', detail: 'Clustering embeddings…' });
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
        updateProgress(1, 1, `Tagged ${res.files_tagged ?? 0} file(s) · ${res.persons ?? 0} persons`);
        setProgressActivityPhase('done');
        await refreshFaceStatus();
    } finally {
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

    $('#btn-face-load-model')?.addEventListener('click', async () => {
        const res = await api.faceLoadModel();
        if (!res.success) {
            alert(res.error || 'Model load failed');
            return;
        }
        alert(`Face model loaded (${res.active_provider || 'CPU'}).`);
    });

    void refreshFaceStatus();
}

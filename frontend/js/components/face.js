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
let faceDbStats = { faces: 0, unassigned_faces: 0 };

/** Map WebSocket progress to overlay counts (pre-scan steps use step_num; scan uses file index). */
export function faceProgressCounts(msg, fileTotal) {
    const total = Math.max(1, msg.total || fileTotal || 1);
    const stepTotal = msg.step_total || 5;
    const stepNum = msg.step_num || 1;
    const isScan = msg.step === 'scan' || msg.phase === 'detect';
    if (isScan) {
        const preWeight = (stepTotal - 1) / stepTotal;
        const scanWeight = 1 / stepTotal;
        const processed = msg.processed ?? 0;
        const cur = preWeight * total + (processed / total) * scanWeight * total;
        return { cur: Math.min(total, Math.round(cur)), tot: total };
    }
    const cur = (stepNum / stepTotal) * total;
    return { cur: Math.max(0, Math.round(cur)), tot: total };
}

export function faceProgressDetail(msg) {
    if (msg.detail) return String(msg.detail);
    if (msg.step_label) return String(msg.step_label);
    return facePhaseLabel(msg.phase, msg);
}

/** One-line queue summary for the progress stats row during detect. */
export function faceProgressStatsLine(msg) {
    const q = msg?.queue;
    if (!q || typeof q !== 'object') return '';
    const parts = [];
    if (q.images != null) parts.push(`${q.images} images`);
    if (q.videos) parts.push(`${q.videos} videos skipped`);
    if (q.marker_skip) parts.push(`${q.marker_skip} already tagged`);
    if (q.db_skip) parts.push(`${q.db_skip} in embedding DB`);
    if (q.to_process != null) parts.push(`${q.to_process} to scan`);
    return parts.join(' · ');
}

/** Map recognize session poll payload to overlay bar counts (stages + apply). */
export function faceRecognizeProgressCounts(recognize) {
    const r = recognize || {};
    const stageTotal = Math.max(1, r.stage_total || 4);
    const barTotal = stageTotal + 1;
    if (r.phase === 'apply') {
        const ft = Math.max(1, r.files_total || 1);
        const fd = r.files_done || 0;
        const cur = stageTotal + (fd / ft);
        return {
            cur: Math.min(barTotal, Math.round(cur * 10) / 10),
            tot: barTotal,
            detail: r.detail || 'Applying person tags in Hydrus…',
        };
    }
    const stageIdx = Math.max(0, r.stage_idx || 0);
    return {
        cur: Math.min(barTotal, stageIdx),
        tot: barTotal,
        detail: r.step_label || r.detail || 'Clustering faces…',
    };
}

export function faceRecognizeStatsLine(recognize) {
    const r = recognize || {};
    const parts = [];
    if (r.faces_in_db) parts.push(`${r.faces_in_db} faces in DB`);
    if (r.unassigned_before != null) parts.push(`${r.unassigned_before} were unassigned`);
    if (r.assigned) parts.push(`${r.assigned} newly assigned`);
    if (r.phase === 'apply' && r.files_total) {
        parts.push(`${r.files_done || 0}/${r.files_total} files`);
    }
    return parts.join(' · ');
}

export function faceActivityPhase(msg) {
    if (msg.pipeline === 'detect') {
        if (msg.step === 'scan' || msg.phase === 'detect') return 'inference';
        return 'load';
    }
    return msg.phase === 'metadata' || msg.phase === 'model' ? 'load' : 'inference';
}

export function facePhaseLabel(phase, msg = {}) {
    if (msg.skip_reason === 'video_excluded') return 'Video skipped (face detect is images only)';
    if (msg.skip_reason === 'marker_present') return 'Skipped (already face-tagged)';
    if (msg.skip_reason === 'db_cached') return 'Skipped (embedding already in local DB)';
    if (phase === 'metadata') return 'Loading file metadata…';
    if (phase === 'model') return 'Loading InsightFace model (downloads on first use)…';
    if (phase === 'detect') return null;
    return null;
}

function setFacePipelineStep(step) {
    const steps = document.querySelectorAll('#face-pipeline-steps .face-pipeline-step');
    for (const el of steps) {
        const name = el.getAttribute('data-step');
        el.classList.remove('is-active', 'is-done');
        el.removeAttribute('aria-current');
        if (!step) continue;
        if (name === step) {
            el.classList.add('is-active');
            el.setAttribute('aria-current', 'step');
        } else if (name === 'detect' && step === 'recognize') {
            el.classList.add('is-done');
        } else if (step === 'done' && name === 'detect') {
            el.classList.add('is-done');
        } else if (step === 'done' && name === 'recognize') {
            el.classList.add('is-done');
        }
    }
}

function clearFacePipelineStep() {
    setFacePipelineStep(null);
}

/** Screen-reader summary + stat grid values. */
export function formatFaceStatusText(st) {
    const faces = st?.faces ?? 0;
    const persons = st?.persons ?? 0;
    const unassigned = st?.unassigned_faces ?? 0;
    const files = st?.files_with_faces ?? 0;
    return `${faces} faces, ${persons} persons, ${unassigned} unassigned, ${files} files with faces`;
}

/** Actionable hint for incremental detect → recognize workflow. */
export function formatFaceIncrementalHint(st) {
    const faces = st?.faces ?? 0;
    const unassigned = st?.unassigned_faces ?? 0;
    const files = st?.files_with_faces ?? 0;
    if (faces === 0) {
        return 'No embeddings yet — run Detect on a search or selection. Each file writes detection markers to Hydrus immediately; stop anytime and resume later.';
    }
    if (unassigned > 0) {
        return `${unassigned} unassigned face(s) in ${files} file(s) — Recognize uses all ${faces} embeddings in the DB (staged clustering writes person tags after each stage). Full recluster is recommended after large detect batches.`;
    }
    return `All ${faces} face(s) linked to persons across ${files} file(s). Detect more images (skipped when already in DB), then Recognize again to refine grouping.`;
}

export function faceRecognizeMode() {
    const incremental = document.querySelector('#face-recognize-mode-incremental')?.checked;
    return incremental ? 'incremental' : 'full';
}

export function faceRefineIncremental() {
    return faceRecognizeMode() === 'incremental';
}

export function renderFaceDbStats(st) {
    const setVal = (id, value, warn = false) => {
        const el = $(id);
        if (!el) return;
        el.textContent = String(value ?? 0);
        const card = el.closest('.face-stat');
        if (card) card.classList.toggle('face-stat--warn', warn);
    };
    setVal('#face-stat-faces', st?.faces ?? 0);
    setVal('#face-stat-persons', st?.persons ?? 0);
    setVal('#face-stat-unassigned', st?.unassigned_faces ?? 0, (st?.unassigned_faces ?? 0) > 0);
    setVal('#face-stat-files', st?.files_with_faces ?? 0);
    const sr = $('#face-db-stats');
    if (sr) sr.textContent = formatFaceStatusText(st);
}

/** Same as hydrus-ai-taggers: tags commit to Hydrus storage immediately (no pending queue). */
export function faceHydrusApplyHint() {
    return 'Detect and recognize write tags to Hydrus after each file (detect) or clustering stage (recognize) — interrupted runs keep prior progress. Map person:p# to real names with tag siblings in Hydrus.';
}

function setFaceLastRunSummary(text) {
    const el = $('#face-last-run-summary');
    if (!el) return;
    const value = (text || '').trim();
    if (value) {
        el.textContent = value;
        el.hidden = false;
    } else {
        el.textContent = '';
        el.hidden = true;
    }
}

function resolveFaceServiceKey() {
    const sel = $('#select-face-service');
    if (sel && sel.value) return sel.value;
    const wd = $('#select-service');
    return wd?.value || '';
}

function faceReplaceExisting() {
    return $('#check-face-replace')?.checked ?? false;
}

function faceReplacePersonTags() {
    return $('#check-face-replace-person-tags')?.checked ?? true;
}

function updateFaceIncrementalHint(st) {
    const el = $('#face-incremental-hint');
    if (!el) return;
    el.textContent = formatFaceIncrementalHint(st);
}

function updateFaceActionButtons() {
    const state = getState();
    const lock = state.taggingLockedByOtherTab;
    const busy = state.processing;
    const selected = state.selectedIds?.size ?? 0;
    const hasSearch = (state.fileIds?.length ?? 0) > 0;
    const hasFaces = (faceDbStats.faces ?? 0) > 0;

    const disable = lock || busy;
    for (const id of [
        '#btn-face-run-pipeline-selected',
        '#btn-face-run-pipeline-all',
        '#btn-face-detect-selected',
        '#btn-face-detect-all',
        '#btn-face-recognize',
    ]) {
        const btn = $(id);
        if (!btn) continue;
        if (id === '#btn-face-run-pipeline-selected' || id === '#btn-face-detect-selected') {
            btn.disabled = disable || selected === 0;
        } else if (id === '#btn-face-run-pipeline-all' || id === '#btn-face-detect-all') {
            btn.disabled = disable || !hasSearch;
        } else if (id === '#btn-face-recognize') {
            btn.disabled = disable || !hasFaces;
        }
    }
    for (const id of ['#btn-face-reset', '#btn-face-clean']) {
        const btn = $(id);
        if (btn) btn.disabled = disable;
    }
}

export function syncFaceActionButtons() {
    updateFaceActionButtons();
}

export async function refreshFaceStatus() {
    const el = $('#face-db-stats');
    const hint = $('#face-model-hint');
    if (!el && !hint && !$('#face-stats-grid')) return;
    const res = await api.faceStatus();
    if (!res.success) {
        if (el) el.textContent = res.error || 'Face status unavailable';
        if (hint) hint.textContent = '';
        renderFaceDbStats({ faces: 0, persons: 0, unassigned_faces: 0, files_with_faces: 0 });
        updateFaceIncrementalHint({ faces: 0, unassigned_faces: 0, files_with_faces: 0 });
        faceDbStats = { faces: 0, unassigned_faces: 0 };
        updateFaceActionButtons();
        return;
    }
    const st = res.status || {};
    faceDbStats = st;
    renderFaceDbStats(st);
    updateFaceIncrementalHint(st);
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
    updateFaceActionButtons();
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

function validateFaceRun(fileIds) {
    if (!fileIds.length) {
        alert('No files selected.');
        return false;
    }
    if (!resolveFaceServiceKey()) {
        alert('Choose a face tag service before detecting faces.');
        return false;
    }
    return true;
}

/**
 * Run face detect over WebSocket. Resolves with terminal WS message; rejects on error/cancel.
 */
export function runFaceDetect(fileIds, { tagAll = false, keepOverlay = false } = {}) {
    if (!validateFaceRun(fileIds)) {
        return Promise.reject(new Error('invalid face detect request'));
    }

    setState({ processing: true });
    updateFaceActionButtons();
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
            replace_existing: faceReplaceExisting(),
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
                    const { cur, tot } = faceProgressCounts(msg, fileIds.length);
                    const detail = faceProgressDetail(msg);
                    const providerNote = msg.active_provider ? ` · ${msg.active_provider}` : '';
                    const line = detail
                        || `Detected ${cur}/${tot} · last faces: ${msg.face_count ?? 0}${providerNote}`;
                    const stats = faceProgressStatsLine(msg);
                    updateProgress(cur, tot, title, line, stats);
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

    return done
        .then((final) => {
            const faces = final?.faces_found ?? 0;
            const processed = final?.files_processed ?? fileIds.length;
            const videos = final?.videos_skipped ?? 0;
            const tagged = final?.marker_skipped ?? 0;
            const dbSkip = final?.db_skipped ?? 0;
            const errN = final?.errors ?? 0;
            const stopped = final?.type === 'stopped';
            let tail = stopped
                ? `Stopped · ${processed} file(s) saved · ${faces} face(s) found`
                : `Complete · ${faces} face(s) found`;
            if (videos) tail += ` · ${videos} video(s) skipped`;
            if (tagged) tail += ` · ${tagged} already tagged`;
            if (dbSkip) tail += ` · ${dbSkip} from embedding DB`;
            if (errN) tail += ` · ${errN} error(s)`;
            if (stopped) tail += ' — resume later; completed files keep Hydrus markers and DB rows';
            updateProgress(processed, fileIds.length, title, tail);
            setProgressActivityPhase(stopped ? 'stopping' : 'done');
            if (stopped) {
                clearFacePipelineStep();
            } else {
                setFacePipelineStep('recognize');
            }
            setFaceLastRunSummary(`Detect: ${tail}. ${faceHydrusApplyHint()}`);
            void refreshFaceStatus();
            return { final, stopped: final?.type === 'stopped' };
        })
        .catch((err) => {
            clearFacePipelineStep();
            setFaceLastRunSummary('');
            if (!keepOverlay) hideProgress();
            throw err;
        })
        .finally(() => {
            faceWsCancel = null;
            clearFaceProgressHandlers();
            if (!keepOverlay) {
                setState({ processing: false });
                updateFaceActionButtons();
                hideProgress();
            }
        });
}

export async function runRecognize({ keepOverlay = false, fromPipeline = false } = {}) {
    const serviceKey = resolveFaceServiceKey();
    if (!serviceKey) {
        alert('Choose a face tag service before recognizing persons.');
        return null;
    }
    const maxDistance = parseFloat($('#slider-face-distance')?.value || '0.5');
    const stageHint = 'Stages: 20 → 5 → 3 → 1 faces per cluster';
    const title = 'Step 2/2 — Recognize persons';
    setState({ processing: true });
    updateFaceActionButtons();
    setFacePipelineStep('recognize');
    showFaceProgress(5, title, `Recognize 1/5 — Cluster (min 20 faces) · ${stageHint}`);
    setProgressActivityPhase('run');
    let recognizePoll = null;
    const stopRecognizePoll = () => {
        if (recognizePoll != null) {
            clearInterval(recognizePoll);
            recognizePoll = null;
        }
    };
    recognizePoll = setInterval(() => {
        api.faceSessionStatus().then((st) => {
            if (!st?.success || !st.recognize?.active) return;
            requestProgressFrame(() => {
                const { cur, tot, detail } = faceRecognizeProgressCounts(st.recognize);
                const stats = faceRecognizeStatsLine(st.recognize);
                updateProgress(cur, tot, title, detail, stats);
                setProgressActivityPhase(
                    st.recognize.phase === 'apply' ? 'hydrus' : 'inference',
                    { titleSuffix: st.recognize.step_label || '' },
                );
            });
        }).catch(() => {});
    }, 600);
    try {
        const res = await api.faceRecognize({
            service_key: serviceKey,
            max_distance: maxDistance,
            staged: true,
            replace_person_tags: faceReplacePersonTags(),
            recluster_all: !faceRefineIncremental(),
            refine_incremental: faceRefineIncremental(),
        });
        stopRecognizePoll();
        if (!res.success) {
            alert(res.error || 'Recognition failed');
            if (!fromPipeline) clearFacePipelineStep();
            return null;
        }
        const stages = Array.isArray(res.stages) ? res.stages : [];
        const stageSummary = stages.length
            ? stages.map((s) => `${s.step_label || `stage ${s.stage}`}: +${s.assigned}`).join(' · ')
            : '';
        const modeLabel = res.refine_incremental ? 'Incremental refine' : 'Full recluster';
        const touchNote = res.refine_incremental && res.files_touched != null
            ? ` · ${res.files_touched} file(s) updated in Hydrus`
            : '';
        const tail = `${modeLabel}: tagged ${res.files_tagged ?? 0} file(s) · ${res.persons ?? 0} persons`
            + ` · ${res.assigned_faces ?? 0} face(s) assigned`
            + touchNote
            + (stageSummary ? ` · ${stageSummary}` : '');
        const barTotal = (stages.length || 4) + 1;
        updateProgress(barTotal, barTotal, title, tail, faceRecognizeStatsLine({
            faces_in_db: res.faces_in_db,
            assigned: res.assigned_faces,
            files_done: res.files_tagged,
            files_total: res.files_tagged,
        }));
        setProgressActivityPhase('done');
        setFacePipelineStep('done');
        const hint = faceHydrusApplyHint();
        setFaceLastRunSummary(
            fromPipeline ? `Pipeline complete · ${tail}. ${hint}` : `Recognize: ${tail}. ${hint}`,
        );
        await refreshFaceStatus();
        return res;
    } catch (err) {
        stopRecognizePoll();
        if (!fromPipeline) clearFacePipelineStep();
        throw err;
    } finally {
        stopRecognizePoll();
        clearFaceProgressHandlers();
        if (!keepOverlay) {
            setState({ processing: false });
            updateFaceActionButtons();
            hideProgress();
        }
    }
}

/** Primary flow: detect selected/all files, then cluster and apply person tags. */
export async function runFacePipeline(fileIds, { tagAll = false } = {}) {
    if (!validateFaceRun(fileIds)) return;
    try {
        const detectOut = await runFaceDetect(fileIds, { tagAll, keepOverlay: true });
        if (detectOut?.stopped) {
            clearFacePipelineStep();
            return;
        }
        await runRecognize({ keepOverlay: false, fromPipeline: true });
    } catch (err) {
        alert('Face pipeline failed: ' + (err?.message || err));
    } finally {
        setState({ processing: false });
        updateFaceActionButtons();
        hideProgress();
    }
}

export function initFace() {
    populateFaceServiceSelect();

    $('#btn-face-run-pipeline-selected')?.addEventListener('click', () => {
        const ids = [...getState().selectedIds];
        void runFacePipeline(ids);
    });

    $('#btn-face-run-pipeline-all')?.addEventListener('click', () => {
        const ids = getState().fileIds || [];
        if (!ids.length) {
            alert('Run a search first.');
            return;
        }
        void runFacePipeline(ids, { tagAll: true });
    });

    $('#btn-face-detect-selected')?.addEventListener('click', () => {
        const ids = [...getState().selectedIds];
        runFaceDetect(ids).catch((err) => {
            alert('Face detection failed: ' + (err?.message || err));
        });
    });

    $('#btn-face-detect-all')?.addEventListener('click', () => {
        const ids = getState().fileIds || [];
        if (!ids.length) {
            alert('Run a search first.');
            return;
        }
        runFaceDetect(ids, { tagAll: true }).catch((err) => {
            alert('Face detection failed: ' + (err?.message || err));
        });
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
        const removed = res.removed ?? 0;
        setFaceLastRunSummary(`Clean DB: removed ${removed} orphaned row(s).`);
        await refreshFaceStatus();
    });

    void refreshFaceStatus();
}

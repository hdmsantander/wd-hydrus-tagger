/**
 * Progress bar component.
 */

import { $ } from '../utils/dom.js';

const _ACTIVITY_CLASSES = [
    'progress-activity--load',
    'progress-activity--run',
    'progress-activity--marker_skip',
    'progress-activity--inference',
    'progress-activity--hydrus',
    'progress-activity--stopping',
    'progress-activity--paused',
    'progress-activity--wait_hydrus',
    'progress-activity--done',
];

/** Stable colored pill (avoids flashing titles when progress messages arrive quickly). */
export function setProgressActivityPhase(phase, { titleSuffix = '' } = {}) {
    const el = $('#progress-activity-indicator');
    if (!el) return;
    const labels = {
        load: 'Loading',
        run: 'Ready',
        marker_skip: 'Skipping markers',
        inference: 'Inference',
        hydrus: 'Hydrus write',
        stopping: 'Stopping',
        paused: 'Paused',
        wait_hydrus: 'Waiting for Hydrus',
        done: 'Done',
    };
    const key = labels[phase] != null ? phase : 'run';
    el.classList.remove(..._ACTIVITY_CLASSES);
    el.classList.add(`progress-activity--${key}`);
    el.textContent = labels[key];
    const suf = titleSuffix && String(titleSuffix).trim() ? String(titleSuffix).trim() : '';
    el.title = suf ? `Phase: ${labels[key]} · ${suf}` : `Phase: ${labels[key]}`;
}

let _progressRafScheduled = false;
let _progressRafCallback = null;
let _hideProgressTimer = null;
let _els = null;

function progressEls() {
    if (_els) return _els;
    _els = {
        overlay: $('#progress-overlay'),
        bar: $('#progress-bar'),
        text: $('#progress-text'),
        title: $('#progress-title'),
        detail: $('#progress-detail'),
        stats: $('#progress-stats'),
        perf: $('#progress-perf-tuning'),
        learningLine: $('#progress-learning-line'),
        sessionTuneLine: $('#progress-session-tune-line'),
        trainingWrap: $('#progress-training-wrap'),
        trainingBar: $('#progress-bar-training'),
        trainingText: $('#progress-training-text'),
        trainingEta: $('#progress-training-eta'),
        actions: $('#progress-actions'),
        observerNote: $('#progress-observer-note'),
    };
    return _els;
}

/** Coalesce rapid WebSocket updates to one paint per animation frame (smoother UI, less DOM work). */
export function requestProgressFrame(callback) {
    _progressRafCallback = callback;
    if (_progressRafScheduled) {
        return;
    }
    _progressRafScheduled = true;
    const run = () => {
        _progressRafScheduled = false;
        const fn = _progressRafCallback;
        _progressRafCallback = null;
        if (typeof fn === 'function') {
            fn();
        }
    };
    if (typeof requestAnimationFrame === 'function') {
        requestAnimationFrame(run);
    } else {
        setTimeout(run, 0);
    }
}

export function showProgress(total, { sessionAutoTune = false } = {}) {
    if (_hideProgressTimer != null) {
        clearTimeout(_hideProgressTimer);
        _hideProgressTimer = null;
    }
    const els = progressEls();
    if (!els.overlay || !els.bar || !els.text || !els.title) return;
    els.overlay.style.display = 'flex';
    els.bar.style.width = '0%';
    els.text.textContent = `0 / ${total}`;
    els.title.textContent = 'Working…';
    setProgressActivityPhase('load');
    const detail = els.detail;
    if (detail) detail.textContent = '';
    const stats = els.stats;
    if (stats) stats.textContent = '';
    const perf = els.perf;
    if (perf) {
        perf.textContent = '';
        perf.style.display = 'none';
    }
    const learn = els.learningLine;
    if (learn) {
        learn.textContent = '';
        learn.style.display = 'none';
    }
    const st = els.sessionTuneLine;
    if (st) {
        st.textContent = '';
        st.style.display = 'none';
    }
    const tw = els.trainingWrap;
    const tb = els.trainingBar;
    const ttxt = els.trainingText;
    const teta = els.trainingEta;
    if (tw) {
        tw.style.display = sessionAutoTune ? 'block' : 'none';
        if (sessionAutoTune && tb && ttxt) {
            tb.style.width = '0%';
            ttxt.textContent = 'Tuning search: waiting for first batch…';
            if (teta) teta.textContent = '';
        }
    }
}

/**
 * Second bar: session auto-tune search batches (not the file queue).
 * ETA is a rough projection from recent batch wall times; cleared while awaiting supervised approval.
 */
export function syncTrainingProgressBar(tuningState, { sessionAutoTune = false, learningCalibration = false } = {}) {
    const wrap = $('#progress-training-wrap');
    const bar = $('#progress-bar-training');
    const txt = $('#progress-training-text');
    const etaEl = $('#progress-training-eta');
    if (!wrap || !bar || !txt) return;
    if (!sessionAutoTune || !tuningState || typeof tuningState !== 'object') {
        wrap.style.display = 'none';
        bar.style.width = '0%';
        txt.textContent = '';
        if (etaEl) etaEl.textContent = '';
        return;
    }
    wrap.style.display = 'block';
    const total = Number(tuningState.tuning_search_total);
    const done = Number(tuningState.tuning_search_done);
    const complete = Boolean(tuningState.tuning_search_complete);
    if (!Number.isFinite(total) || total <= 0) {
        txt.textContent = 'Tuning search: starting…';
        bar.style.width = '0%';
        if (etaEl) etaEl.textContent = '';
        return;
    }
    if (learningCalibration && tuningState.phase === 'commit_apply') {
        txt.textContent = 'Tuning search complete — commit segment uses locked best settings';
        bar.style.width = '100%';
        if (etaEl) etaEl.textContent = '';
        return;
    }
    const pct = complete ? 100 : Math.min(100, Math.round((done / total) * 1000) / 10);
    bar.style.width = `${pct}%`;
    txt.textContent = complete
        ? `Tuning search done (${done} / ${total} batches)`
        : `Tuning search batches: ${done} / ${total}`;
    if (etaEl) {
        const eta = tuningState.tuning_eta_seconds;
        if (complete || eta == null || !Number.isFinite(Number(eta))) {
            etaEl.textContent = '';
        } else {
            etaEl.textContent = `Estimated time left (rough): ~${Number(eta).toFixed(0)}s`;
        }
    }
}

export function updateProgress(
    current,
    total,
    message = null,
    detail = null,
    statsLines = null,
) {
    const els = progressEls();
    if (!els.bar || !els.text) return;
    const pct = total > 0 ? (current / total * 100) : 0;
    els.bar.style.width = `${pct}%`;
    els.text.textContent = `${current} / ${total}`;
    if (message) {
        if (els.title) els.title.textContent = message;
    }
    const detailEl = els.detail;
    if (detailEl) {
        detailEl.textContent = detail == null ? '' : String(detail);
    }
    const statsEl = els.stats;
    if (statsEl) {
        statsEl.textContent = statsLines == null ? '' : String(statsLines);
    }
}

export function hideProgress() {
    if (_hideProgressTimer != null) {
        clearTimeout(_hideProgressTimer);
    }
    _hideProgressTimer = setTimeout(() => {
        _hideProgressTimer = null;
        const els = progressEls();
        if (!els.overlay) return;
        els.overlay.style.display = 'none';
        setProgressActivityPhase('done');
        const pa = els.actions;
        if (pa) pa.style.display = 'none';
        const note = els.observerNote;
        if (note) note.style.display = 'none';
        const tw = els.trainingWrap;
        const tb = els.trainingBar;
        const ttxt = els.trainingText;
        const teta = els.trainingEta;
        if (tw) tw.style.display = 'none';
        if (tb) tb.style.width = '0%';
        if (ttxt) ttxt.textContent = '';
        if (teta) teta.textContent = '';
    }, 500);
}

/** Controller tab shows Stop/Pause/Flush; observer tabs hide actions and show a short note. */
export function setProgressControlMode({ controller } = {}) {
    const els = progressEls();
    const pa = els.actions;
    const note = els.observerNote;
    if (pa) {
        pa.style.display = controller ? 'flex' : 'none';
    }
    if (note) {
        note.style.display = controller ? 'none' : 'block';
    }
}

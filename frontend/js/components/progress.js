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

/** CPU/GPU activity chips for ONNX tagging (and reusable for face runs). */
export function setProgressComputeActivity({
    useGpu = false,
    computeDevice = 'cpu',
    computeActivity = 'cpu',
    activeProvider = '',
    gpuBackend = 'auto',
} = {}) {
    const wrap = $('#progress-compute-indicator');
    const cpu = $('#progress-compute-cpu');
    const gpu = $('#progress-compute-gpu');
    if (!wrap || !cpu || !gpu) return;
    wrap.hidden = false;
    const act = computeActivity === 'gpu' ? 'gpu' : 'cpu';
    cpu.classList.toggle('progress-compute-chip--active', act === 'cpu');
    cpu.classList.toggle('progress-compute-chip--idle', act !== 'cpu');
    gpu.classList.toggle('progress-compute-chip--gpu', true);
    gpu.classList.toggle('progress-compute-chip--active', act === 'gpu');
    gpu.classList.toggle('progress-compute-chip--idle', act !== 'gpu');
    const prov = (activeProvider || '').trim()
        || (computeDevice === 'gpu' ? 'GPU' : 'CPUExecutionProvider');
    const backendNote = useGpu ? ` · backend ${gpuBackend || 'auto'}` : '';
    wrap.title = `ONNX compute: ${prov}${backendNote}`;
}

export function hideProgressComputeActivity() {
    const wrap = $('#progress-compute-indicator');
    if (wrap) wrap.hidden = true;
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

export function showProgress(total, opts = {}) {
    if (_hideProgressTimer != null) {
        clearTimeout(_hideProgressTimer);
        _hideProgressTimer = null;
    }
    let options = opts && typeof opts === 'object' ? { ...opts } : {};
    let n = total;
    if (total != null && typeof total === 'object' && !Array.isArray(total)) {
        options = { ...total, ...options };
        n = options.total;
    }
    const count = Number.isFinite(Number(n)) ? Math.max(0, Number(n)) : 0;
    const sessionAutoTune = options.sessionAutoTune === true;
    const els = progressEls();
    if (!els.overlay || !els.bar || !els.text || !els.title) return;
    els.overlay.style.display = 'flex';
    els.bar.style.width = '0%';
    els.text.textContent = `0 / ${count}`;
    els.title.textContent = options.title ? String(options.title) : 'Working…';
    setProgressActivityPhase('load');
    const detail = els.detail;
    if (detail) detail.textContent = options.detail != null ? String(options.detail) : '';
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
    const cur = Number.isFinite(Number(current)) ? Number(current) : 0;
    const tot = Number.isFinite(Number(total)) ? Number(total) : 0;
    const pct = tot > 0 ? Math.min(100, (cur / tot * 100)) : 0;
    els.bar.style.width = `${pct}%`;
    els.text.textContent = `${cur} / ${tot}`;
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
        hideProgressComputeActivity();
    }, 500);
}

/** Show only the requested progress-card buttons (reused by tagger and face). */
export function setProgressActionButtons({
    stop = false,
    pause = false,
    resume = false,
    flush = false,
    tuningApprove = false,
} = {}) {
    const any = stop || pause || resume || flush || tuningApprove;
    setProgressControlMode({ controller: any });
    const map = [
        ['#btn-progress-stop', stop],
        ['#btn-progress-pause', pause],
        ['#btn-progress-resume', resume],
        ['#btn-progress-flush', flush],
        ['#btn-tuning-approve', tuningApprove],
    ];
    for (const [sel, on] of map) {
        const btn = $(sel);
        if (btn) btn.style.display = on ? 'inline-block' : 'none';
    }
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

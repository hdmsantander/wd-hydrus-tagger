/**
 * Progress bar component.
 */

import { $ } from '../utils/dom.js';

let _progressRafScheduled = false;
let _progressRafCallback = null;

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

export function showProgress(total) {
    $('#progress-overlay').style.display = 'flex';
    $('#progress-bar').style.width = '0%';
    $('#progress-text').textContent = `0 / ${total}`;
    $('#progress-title').textContent = 'Working…';
    const detail = $('#progress-detail');
    if (detail) detail.textContent = '';
    const stats = $('#progress-stats');
    if (stats) stats.textContent = '';
    const perf = $('#progress-perf-tuning');
    if (perf) {
        perf.textContent = '';
        perf.style.display = 'none';
    }
}

export function updateProgress(
    current,
    total,
    message = null,
    detail = null,
    statsLines = null,
    { showBar = true } = {},
) {
    if (showBar) {
        const pct = total > 0 ? (current / total * 100) : 0;
        $('#progress-bar').style.width = `${pct}%`;
        $('#progress-text').textContent = `${current} / ${total}`;
    } else {
        $('#progress-text').textContent = `${current} / ${total}`;
    }
    if (message) {
        $('#progress-title').textContent = message;
    }
    const detailEl = $('#progress-detail');
    if (detailEl) {
        detailEl.textContent = detail == null ? '' : String(detail);
    }
    const statsEl = $('#progress-stats');
    if (statsEl) {
        statsEl.textContent = statsLines == null ? '' : String(statsLines);
    }
}

export function hideProgress() {
    setTimeout(() => {
        $('#progress-overlay').style.display = 'none';
        const pa = $('#progress-actions');
        if (pa) pa.style.display = 'none';
        const note = $('#progress-observer-note');
        if (note) note.style.display = 'none';
    }, 500);
}

/** Controller tab shows Stop/Pause/Flush; observer tabs hide actions and show a short note. */
export function setProgressControlMode({ controller, showBar = true } = {}) {
    const pa = $('#progress-actions');
    const note = $('#progress-observer-note');
    if (pa) {
        pa.style.display = controller ? 'flex' : 'none';
    }
    if (note) {
        note.style.display = controller ? 'none' : 'block';
    }
    if (!showBar && $('#progress-overlay')?.style.display === 'flex') {
        $('#progress-bar').style.width = '0%';
    }
}

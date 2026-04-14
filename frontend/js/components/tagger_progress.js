/**
 * Progress/status formatting helpers extracted from tagger component.
 */

import { $ } from '../utils/dom.js';

/** One-line summary from WebSocket `performance_tuning` object (Tag all + tuning only). */
export function formatPerfTuningSummary(pt, historyLen = 0) {
    if (!pt || typeof pt !== 'object') return '';
    const parts = [];
    if (pt.fetch_s != null && Number.isFinite(Number(pt.fetch_s))) {
        parts.push(`Hydrus fetch ${Number(pt.fetch_s).toFixed(2)}s`);
    }
    if (pt.predict_s != null && Number.isFinite(Number(pt.predict_s))) {
        parts.push(`ONNX ${Number(pt.predict_s).toFixed(2)}s`);
    }
    if (pt.hydrus_apply_batch_s != null && Number.isFinite(Number(pt.hydrus_apply_batch_s))) {
        parts.push(`Hydrus apply ${Number(pt.hydrus_apply_batch_s).toFixed(2)}s`);
    }
    if (parts.length === 0) return '';
    const bi = pt.batch_index != null ? `batch ${pt.batch_index}` : 'batch';
    let line = `${bi}: ${parts.join(' · ')}`;
    if (historyLen > 1) {
        line += ` · rolling ${historyLen} batches`;
    }
    return line;
}

export function syncProgressPerfElement(pt, historyLen = 0) {
    const perfEl = $('#progress-perf-tuning');
    if (!perfEl) return;
    const line = formatPerfTuningSummary(pt, historyLen);
    if (line) {
        perfEl.textContent = line;
        perfEl.style.display = 'block';
    } else {
        perfEl.textContent = '';
        perfEl.style.display = 'none';
    }
}

/** Learning-phase + session auto-tune: prominent line under stats (not ML training — online knob search). */
export function syncProgressLearningLine(
    calibrationPhase,
    tuningState,
    sessionAutoTune,
    learningCalibration,
) {
    const el = $('#progress-learning-line');
    if (!el) return;
    if (!learningCalibration) {
        el.textContent = '';
        el.style.display = 'none';
        return;
    }
    if (!sessionAutoTune) {
        if (calibrationPhase === 'learning') {
            el.textContent =
                'Learning segment: incremental Hydrus writes deferred until the commit segment. '
                + 'Enable Session auto-tune to search batch size and download parallelism during this segment.';
        } else if (calibrationPhase === 'commit') {
            el.textContent =
                'Commit segment: applying tags to Hydrus; queue was split using your learning fraction and scope.';
        } else {
            el.textContent = '';
        }
        el.style.display = el.textContent ? 'block' : 'none';
        return;
    }
    const seg =
        calibrationPhase === 'learning'
            ? 'Learning segment'
            : calibrationPhase === 'commit'
              ? 'Commit segment'
              : 'Calibration';
    const ts = tuningState && typeof tuningState === 'object' ? tuningState : null;
    if (!ts) {
        el.textContent = `${seg}: auto-tune starting…`;
        el.style.display = 'block';
        return;
    }
    if (ts.phase_title) {
        el.textContent = `${seg} · ${ts.phase_title}`;
        el.style.display = 'block';
        return;
    }
    const ph = ts.phase != null ? String(ts.phase) : '—';
    const bb = ts.best_batch_size != null ? ts.best_batch_size : '—';
    const bd = ts.best_download_parallel != null ? ts.best_download_parallel : '—';
    const nb = ts.next_batch_size != null ? ts.next_batch_size : '—';
    const nd = ts.next_download_parallel != null ? ts.next_download_parallel : '—';
    const bo = ts.best_ort_intra_op_threads != null ? ts.best_ort_intra_op_threads : null;
    let line = `${seg} · auto-tune phase: ${ph} · best so far: batch ${bb}, Hydrus parallel ${bd}`;
    if (bo != null && ts.session_auto_tune_threads) {
        line += ` · best ORT intra ${bo}`;
    }
    if (ph === 'commit_apply') {
        line += ` · applying locked batch ${nb} / parallel ${nd}`;
    } else if (ph !== 'warm_up') {
        line += ` · next trial: batch ${nb}, parallel ${nd}`;
    }
    el.textContent = line;
    el.style.display = 'block';
}

export function formatCalibrationOneLine(calibrationPhase, tuningState, { observer = false } = {}) {
    if (!calibrationPhase) return '';
    const ts = tuningState && typeof tuningState === 'object' ? tuningState : null;
    const seg = calibrationPhase === 'learning' ? 'Learning' : 'Commit';
    if (!ts) return `Calibration: ${seg} segment · auto-tune running…`;
    if (observer && ts.awaiting_approval === true) {
        const p = ts.proposal && typeof ts.proposal === 'object' ? ts.proposal : {};
        return (
            `${seg}: Supervised tuning — approval is only possible in the tab that started Tag all. `
            + `Next proposal: batch ${p.batch_size ?? '—'}, Hydrus parallel ${p.hydrus_download_parallel ?? '—'}, `
            + `ORT intra ${p.cpu_intra_op_threads ?? '—'}, inter ${p.cpu_inter_op_threads ?? '—'}.`
        );
    }
    if (ts.phase_title && ts.phase_detail) {
        return `${seg}: ${ts.phase_title} — ${ts.phase_detail}`;
    }
    const ph = ts.phase != null ? String(ts.phase) : '—';
    const bb = ts.best_batch_size != null ? ts.best_batch_size : '—';
    const bd = ts.best_download_parallel != null ? ts.best_download_parallel : '—';
    return `Calibration: ${seg} · auto-tune ${ph} · best batch ${bb} · best Hydrus parallel ${bd}`;
}

/** Session auto-tune (Tag all): title + detail from server; sequential phases, current params. */
export function syncProgressSessionTuneLine(tuningState, sessionAutoTune, observerMode = false) {
    const el = $('#progress-session-tune-line');
    if (!el) return;
    if (!sessionAutoTune || !tuningState || typeof tuningState !== 'object') {
        el.textContent = '';
        el.style.display = 'none';
        return;
    }
    if (observerMode && tuningState.awaiting_approval === true) {
        const p = tuningState.proposal && typeof tuningState.proposal === 'object' ? tuningState.proposal : {};
        el.textContent = [
            'Supervised tuning — approval only in the tab that started Tag all',
            `Next proposal: inference batch ${p.batch_size ?? '—'}, Hydrus parallel ${p.hydrus_download_parallel ?? '—'}, `
                + `ORT intra-op ${p.cpu_intra_op_threads ?? '—'}, inter-op ${p.cpu_inter_op_threads ?? '—'}.`,
            'This read-only view cannot approve.',
        ].join('\n');
        el.style.display = 'block';
        return;
    }
    const title = tuningState.phase_title ? String(tuningState.phase_title) : '';
    const detail = tuningState.phase_detail ? String(tuningState.phase_detail) : '';
    if (!title && !detail) {
        el.textContent = '';
        el.style.display = 'none';
        return;
    }
    el.textContent = title ? `${title}\n${detail}` : detail;
    el.style.display = 'block';
}


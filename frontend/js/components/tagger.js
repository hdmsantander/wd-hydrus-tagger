/**
 * Tagger controls and results display component.
 */

import { api } from '../api.js';
import { syncConnectionShell } from './connection.js';
import { getState, setState } from '../state.js';
import { $, el, show, hide } from '../utils/dom.js';
import {
    showProgress,
    hideProgress,
    updateProgress,
    setProgressActivityPhase,
    setProgressControlMode,
    requestProgressFrame,
    syncTrainingProgressBar,
} from './progress.js';
import { expectServerShutdownSoon } from '../server_offline.js';
import { syncIncrementalHydrusApplyEveryVisibility } from './settings.js';
import {
    formatCalibrationOneLine,
    formatPerfTuningSummary,
    syncProgressLearningLine,
    syncProgressPerfElement,
    syncProgressSessionTuneLine,
} from './tagger_progress.js';

let currentResults = [];
/** Page index for the post-tagging results list (thumbnails + tags). */
let resultsCurrentPage = 0;
/**
 * Files per page on the results screen — keeps DOM size and concurrent Hydrus thumbnail
 * requests bounded (each card loads `/api/files/:id/thumbnail`).
 */
const RESULTS_PAGE_SIZE = 25;
let observerOverlayOpen = false;
let refreshRemoteTaggingUi = async () => {};
let taggingSessionPollTimer = null;


function formatTuningReportFooter(final) {
    if (!final || typeof final !== 'object') return '';
    const lines = [];
    const tr = final.tuning_report;
    if (tr && typeof tr === 'object') {
        const agg = tr.aggregate || {};
        if (agg.files_per_wall_s != null && Number.isFinite(Number(agg.files_per_wall_s))) {
            lines.push(
                `Estimated throughput: ${Number(agg.files_per_wall_s).toFixed(2)} files/s wall (sum of fetch + ONNX + apply per recorded batches).`,
            );
        }
        const at = tr.autotune;
        if (at && typeof at === 'object') {
            if (at.best_batch_size != null) {
                const dlp = at.best_download_parallel != null ? at.best_download_parallel : '—';
                lines.push(`Auto-tune best batch size: ${at.best_batch_size} · best Hydrus download parallel: ${dlp}.`);
            }
            if (at.best_ort_intra_op_threads != null) {
                lines.push(`Auto-tune best ORT intra-op threads: ${at.best_ort_intra_op_threads}.`);
            }
            if (at.phase) {
                lines.push(`Auto-tune finished in phase: ${at.phase}.`);
            }
        }
    }
    const lc = final.learning_calibration;
    if (lc && typeof lc === 'object') {
        const lrn = lc.learning_count != null ? lc.learning_count : '—';
        const com = lc.commit_count != null ? lc.commit_count : '—';
        const sc = lc.learning_scope_effective || lc.learning_scope_requested || '—';
        lines.push(`Learning split (${sc}): ${lrn} file(s) in learning prefix, ${com} in commit suffix.`);
    }
    if (lines.length === 0) return '';
    return ' ' + lines.join(' ');
}

function setTuningApproveAttention(on) {
    const btn = $('#btn-tuning-approve');
    if (!btn) return;
    btn.classList.toggle('tuning-approve-attention', Boolean(on));
}

/**
 * Main bar uses server progress_bar_* (ONNX units, then marker-skip tail units). Before the first
 * progress message, queue_plan infer_total is the denominator so Tag all does not show 0/1000 when
 * only 600 files need ONNX.
 */
function resolveProgressBarCounts({
    fileIdsLen,
    lastProgressBarCurrent,
    lastProgressBarTotal,
    lastProgressCurrent,
    lastInferTotal,
}) {
    if (lastProgressBarTotal != null && Number.isFinite(Number(lastProgressBarTotal))) {
        const tot = Math.max(1, Number(lastProgressBarTotal));
        const rawCur =
            lastProgressBarCurrent != null ? Number(lastProgressBarCurrent) : Number(lastProgressCurrent);
        const cur = Number.isFinite(rawCur) ? Math.min(Math.max(0, rawCur), tot) : 0;
        return { cur, tot };
    }
    if (lastInferTotal != null && lastInferTotal > 0) {
        const rawCur = lastProgressBarCurrent != null ? Number(lastProgressBarCurrent) : 0;
        const cur = Number.isFinite(rawCur) ? Math.max(0, rawCur) : 0;
        return { cur, tot: lastInferTotal };
    }
    const tot = Math.max(1, fileIdsLen);
    const cur = Math.min(Math.max(0, lastProgressCurrent), tot);
    return { cur, tot };
}

function computeThroughputRates(epochMs, filesDone, tagsWritten) {
    if (epochMs == null) {
        return { filesPerSec: null, tagsPerSec: null };
    }
    const elapsedSec = (Date.now() - epochMs) / 1000;
    if (elapsedSec < 0.35) {
        return { filesPerSec: null, tagsPerSec: null };
    }
    const fd = Number(filesDone);
    const tw = Number(tagsWritten);
    const fps = Number.isFinite(fd) && fd >= 0 ? fd / elapsedSec : null;
    const tps = Number.isFinite(tw) && tw >= 0 ? tw / elapsedSec : null;
    if (!Number.isFinite(fps) || !Number.isFinite(tps)) {
        return { filesPerSec: null, tagsPerSec: null };
    }
    return { filesPerSec: fps, tagsPerSec: tps };
}

/** Tooltip line for the phase pill (status indicator). */
function formatThroughputTitleSuffix(epochMs, filesDone, tagsWritten) {
    const rates = computeThroughputRates(epochMs, filesDone, tagsWritten);
    if (rates.filesPerSec == null || rates.tagsPerSec == null) {
        return '';
    }
    return `~${rates.filesPerSec.toFixed(1)} files/s · ~${rates.tagsPerSec.toFixed(1)} tag strings/s`;
}

function resetTaggingProgressChrome() {
    const stopBtn = $('#btn-progress-stop');
    if (stopBtn) {
        stopBtn.disabled = false;
        stopBtn.textContent = 'Stop';
    }
    for (const sel of ['#btn-progress-pause', '#btn-progress-resume', '#btn-progress-flush']) {
        const b = $(sel);
        if (b) b.disabled = false;
    }
    const hint = $('#progress-stop-hint');
    if (hint) {
        hint.textContent = '';
        hint.style.display = 'none';
    }
    setTuningApproveAttention(false);
}

function msForTaggingPoll() {
    if (typeof document !== 'undefined' && document.hidden && !observerOverlayOpen) {
        return 12000;
    }
    return 2500;
}

function restartTaggingSessionPoll() {
    if (taggingSessionPollTimer != null) {
        clearInterval(taggingSessionPollTimer);
        taggingSessionPollTimer = null;
    }
    void refreshRemoteTaggingUi();
    taggingSessionPollTimer = setInterval(() => {
        void refreshRemoteTaggingUi();
    }, msForTaggingPoll());
}

function isControllerTab() {
    try {
        return sessionStorage.getItem('wd_tagger_controller') === '1';
    } catch {
        return false;
    }
}

function statsFromSnapshot(snap, fallbackTotal = 1) {
    if (!snap) return '';
    const total = snap.total ?? snap.total_files ?? fallbackTotal;
    const cal = snap.calibration_phase;
    const ts = snap.tuning_state;
    const obs = observerOverlayOpen;
    let calibrationLine = '';
    if (cal && ts) {
        calibrationLine = formatCalibrationOneLine(cal, ts, { observer: obs });
    }
    return formatTaggingStats({
        current: snap.current ?? snap.total_processed ?? 0,
        totalFiles: total,
        inferTotal: snap.infer_total,
        queueSkipSame: snap.skip_same_marker,
        queueSkipHi: snap.skip_higher_tier,
        cumulativeInferredNonSkip: snap.cumulative_inferred_non_skip,
        inferenceBatch: snap.inference_batch ?? 8,
        lastBatchInferred: snap.batch_inferred ?? '—',
        batchesCompleted: snap.batches_completed ?? '—',
        batchesTotal: snap.batches_total ?? '—',
        totalApplied: snap.total_applied ?? 0,
        totalTagsWritten: snap.total_tags_written ?? 0,
        totalDuplicatesSkipped: snap.total_duplicates_skipped ?? 0,
        pendingRemaining: snap.pending_remaining ?? 0,
        cumulativeSkippedSameModelMarker: snap.cumulative_skipped_same_model_marker ?? 0,
        cumulativeSkippedHigherTierMarker: snap.cumulative_skipped_higher_tier_model_marker ?? 0,
        cumulativeWdStaleMarkersRemoved: snap.cumulative_wd_stale_markers_removed ?? 0,
        lastBatchSkippedMarker: snap.batch_skipped_inference ?? 0,
        lastBatchSkippedSameModelMarker: snap.batch_skipped_same_model_marker ?? 0,
        lastBatchSkippedHigherTier: snap.batch_skipped_higher_tier_model_marker ?? 0,
        inMarkerSkipTail: Boolean(snap.in_marker_skip_tail),
        throughputFilesPerSec: null,
        throughputTagsPerSec: null,
        perfTuningSummary: formatPerfTuningSummary(snap.performance_tuning),
        calibrationLine,
    });
}

function applySnapshotToObserverOverlay(snap) {
    if (!observerOverlayOpen || !snap) return;
    const total = Math.max(1, snap.total ?? snap.total_files ?? 1);
    const cur = Math.min(snap.current ?? snap.total_processed ?? 0, total);
    const { cur: barCur, tot: barTot } = resolveProgressBarCounts({
        fileIdsLen: total,
        lastProgressBarCurrent: snap.progress_bar_current,
        lastProgressBarTotal: snap.progress_bar_total,
        lastProgressCurrent: snap.current ?? snap.total_processed ?? 0,
        lastInferTotal: snap.infer_total,
    });
    const title = snap.paused
        ? 'Paused (read-only)'
        : snap.phase === 'stopping'
          ? snap.stopping_source === 'server'
            ? 'Server stopping…'
            : 'Stopping tagging…'
          : `Tagging in another tab · ${snap.model_name || 'model'}`;
    const detail =
        snap.phase === 'stopping'
            ? (snap.detail
                || 'Winding down: current batch may finish; pending Hydrus writes will flush if any.')
            : snap.phase === 'finishing'
            ? 'Run is finishing in the controller tab…'
            : 'Updates every few seconds. You cannot start another run until this one finishes.';
    updateProgress(barCur, barTot, title, detail, statsFromSnapshot(snap, total));
    syncProgressPerfElement(snap.performance_tuning);
    const hasTune = Boolean(snap.tuning_state);
    const tw = $('#progress-training-wrap');
    if (tw) tw.style.display = hasTune ? 'block' : 'none';
    syncProgressLearningLine(
        snap.calibration_phase,
        snap.tuning_state,
        hasTune,
        Boolean(snap.calibration_phase),
    );
    syncProgressSessionTuneLine(snap.tuning_state, hasTune, true);
    syncTrainingProgressBar(snap.tuning_state, {
        sessionAutoTune: hasTune,
        learningCalibration: Boolean(snap.calibration_phase),
    });
}

function formatTaggingStats(s) {
    const inferTot = s.inferTotal;
    const qSame = s.queueSkipSame;
    const qHi = s.queueSkipHi;
    const cinf = s.cumulativeInferredNonSkip;
    const bc = s.batchesCompleted ?? '—';
    const bt = s.batchesTotal ?? '—';
    const binf = s.lastBatchInferred ?? '—';
    const ta = s.totalApplied ?? 0;
    const tt = s.totalTagsWritten ?? 0;
    const td = s.totalDuplicatesSkipped ?? 0;
    const skipSame = s.cumulativeSkippedSameModelMarker ?? 0;
    const skipHi = s.cumulativeSkippedHigherTierMarker ?? 0;
    const staleRm = s.cumulativeWdStaleMarkersRemoved ?? 0;
    let batchLine = `Inference batches done: ${bc} / ${bt}`;
    const nBt = Number(bt);
    const nBc = Number(bc);
    if (Number.isFinite(nBt) && nBt > 0 && Number.isFinite(nBc)) {
        batchLine += ` (${Math.round((nBc / nBt) * 100)}% of planned batches)`;
    }
    const skipM = s.lastBatchSkippedMarker ?? 0;
    const skipSameB = s.lastBatchSkippedSameModelMarker ?? 0;
    const skipHiB = s.lastBatchSkippedHigherTier ?? 0;
    let onnxDetail = `ONNX batch size: ${s.inferenceBatch} · last batch: ${binf} inferred`;
    if (skipM > 0) {
        const parts = [];
        if (skipSameB > 0) parts.push(`${skipSameB} same model marker`);
        if (skipHiB > 0) parts.push(`${skipHiB} heavier WD model marker`);
        const other = skipM - skipSameB - skipHiB;
        if (other > 0) parts.push(`${other} other pre-infer skip`);
        onnxDetail += ` · ${skipM} skipped pre-infer (${parts.join(', ')})`;
    }
    const queuePlanLine =
        inferTot != null && inferTot > 0
            ? `Queue (after metadata prefetch): ${inferTot} file(s) need ONNX · ${qSame ?? '—'} skip (same model marker) · ${qHi ?? '—'} skip (heavier WD marker) — infer-first order`
            : '';
    let onnxProgressLine = '';
    if (inferTot != null && inferTot > 0 && cinf != null) {
        const done = Math.min(100, Math.round((cinf / inferTot) * 100));
        const rem = Math.max(0, 100 - done);
        onnxProgressLine = `ONNX inference queue: ${cinf} / ${inferTot} (${done}% of inference work done · ${rem}% of ONNX queue remaining)`;
    }
    const tailHint = s.inMarkerSkipTail
        ? 'Marker-skip tail: processing large batches without ONNX (fast path).'
        : '';
    const lines = [batchLine];
    if (queuePlanLine) lines.push(queuePlanLine);
    if (onnxProgressLine) lines.push(onnxProgressLine);
    if (tailHint) lines.push(tailHint);
    lines.push(
        `All selected files processed in this run: ${s.current} / ${s.totalFiles} (${s.totalFiles > 0 ? Math.round((s.current / s.totalFiles) * 100) : 0}% of gallery selection)`,
        onnxDetail,
        `Hydrus: ${ta} file(s) received new tags · ${tt} new tag string(s) sent`,
    );
    const tp = s.throughputFilesPerSec;
    const ttags = s.throughputTagsPerSec;
    if (tp != null && ttags != null && Number.isFinite(tp) && Number.isFinite(ttags)) {
        lines.push(
            `Throughput (approx.): ~${tp.toFixed(1)} files/s · ~${ttags.toFixed(1)} new tag strings/s`,
        );
    }
    if (skipSame > 0) {
        lines.push(`Skipped ONNX (already has this model marker): ${skipSame} file(s)`);
    }
    if (skipHi > 0) {
        lines.push(
            `Skipped ONNX (file already tagged with a heavier WD model marker): ${skipHi} file(s)`,
        );
    }
    if (staleRm > 0) {
        lines.push(
            `Stale wd model markers dropped from proposals (re-tag with current model): ${staleRm} tag string(s)`,
        );
    }
    if (td > 0) {
        lines.push(`Already on file in Hydrus (skipped, not re-applied): ${td} tag string(s)`);
    }
    if (s.pendingRemaining != null && s.pendingRemaining > 0) {
        lines.push(`Pending write queue: ${s.pendingRemaining} file(s)`);
    }
    if (s.perfTuningSummary) {
        lines.push(s.perfTuningSummary);
    }
    if (s.calibrationLine) {
        lines.push(s.calibrationLine);
    }
    return lines.join('\n');
}

function formatRunSummary({
    tagAll,
    stopped,
    final,
    queuedTotal,
    serviceKeyForRun,
    pendingN,
}) {
    const proc = final.total_processed ?? (Array.isArray(final.results) ? final.results.length : 0);
    const applied = final.total_applied ?? 0;
    const tagsWritten = final.total_tags_written ?? 0;
    const dups = final.total_duplicates_skipped ?? 0;
    const skippedMarker = final.cumulative_skipped_same_model_marker ?? 0;
    const skippedHigher = final.cumulative_skipped_higher_tier_model_marker ?? 0;

    const head = tagAll
        ? stopped
            ? 'Tag all — stopped early.'
            : 'Tag all — finished.'
        : stopped
          ? 'Tagging stopped.'
          : 'Tagging finished.';

    const lines = [
        head,
        `Processed ${proc} of ${queuedTotal} queued file(s).`,
        `Hydrus: ${applied} file(s) received new tags · ${tagsWritten} new tag string(s) sent${
            dups > 0 ? ` (${dups} skipped as already on file).` : '.'
        }`,
    ];
    const tuningFoot = formatTuningReportFooter(final);
    if (tuningFoot) {
        lines.push(tuningFoot.trim());
    }
    if (skippedMarker > 0) {
        lines.push(`Skipped ONNX (model marker already on file): ${skippedMarker} file(s).`);
    }
    if (skippedHigher > 0) {
        lines.push(
            `Skipped ONNX (heavier WD model marker already on file): ${skippedHigher} file(s).`,
        );
    }
    if (serviceKeyForRun) {
        if (pendingN === 0) {
            lines.push(
                'Nothing is pending on the selected tag service — tags were already pushed during the run (or they match Hydrus). Apply all tags to Hydrus is disabled.',
            );
        } else {
            lines.push(
                `${pendingN} file(s) still have tags to apply on that service — use Apply all tags to Hydrus.`,
            );
        }
    } else {
        lines.push('Select a tag service, then use Apply all tags to Hydrus to write tags from the list below.');
    }
    return lines.join(' ');
}

function buildResultCard(result) {
    const g = result.general_tags || {};
    const ch = result.character_tags || {};
    const rt = result.rating_tags || {};
    const hasStruct =
        Object.keys(g).length + Object.keys(ch).length + Object.keys(rt).length > 0;
    const flat = Array.isArray(result.tags) ? result.tags : [];

    const tagBlockChildren = [];
    if (hasStruct) {
        tagBlockChildren.push(
            renderTagCategory('General', 'general', g),
            renderTagCategory('Character', 'character', ch),
            renderTagCategory('Rating', 'rating', rt),
        );
    } else if (flat.length > 0) {
        tagBlockChildren.push(
            el('div', { className: 'tag-category' }, [
                el('div', {
                    className: 'tag-category-label general',
                    textContent: 'Tags to apply',
                }),
                el(
                    'div',
                    { className: 'tag-chips' },
                    flat.map((t) =>
                        el('span', { className: 'tag-chip general', textContent: t }),
                    ),
                ),
            ]),
        );
    } else {
        tagBlockChildren.push(
            el('p', {
                className: 'result-empty',
                textContent: result.skipped_inference
                    ? 'Skipped (already processed by this model).'
                    : 'Nothing left to apply (already in Hydrus).',
            }),
        );
    }

    return el('div', { className: 'result-card' }, [
        el('img', {
            className: 'result-thumb',
            src: api.thumbnailUrl(result.file_id),
            alt: '',
            loading: 'lazy',
            decoding: 'async',
        }),
        el('div', { className: 'result-tags' }, tagBlockChildren),
    ]);
}

function updateResultsPagination() {
    const total = currentResults.length;
    const totalPages = total === 0 ? 0 : Math.ceil(total / RESULTS_PAGE_SIZE);
    const info = $('#results-page-info');
    if (info) {
        if (total === 0) {
            info.textContent = '';
        } else {
            info.textContent = `${total} file(s) · page ${resultsCurrentPage + 1} / ${totalPages}`;
        }
    }

    const container = $('#results-pagination');
    if (!container) return;
    container.innerHTML = '';

    if (totalPages <= 1) return;

    if (resultsCurrentPage > 0) {
        container.appendChild(el('button', {
            type: 'button',
            className: 'btn btn-sm',
            textContent: '<',
            onClick: () => {
                resultsCurrentPage -= 1;
                renderResultsPage();
            },
        }));
    }

    const startPage = Math.max(0, resultsCurrentPage - 3);
    const endPage = Math.min(totalPages, startPage + 7);
    for (let i = startPage; i < endPage; i++) {
        const pageIndex = i;
        container.appendChild(el('button', {
            type: 'button',
            className: `btn btn-sm${pageIndex === resultsCurrentPage ? ' btn-primary' : ''}`,
            textContent: String(pageIndex + 1),
            onClick: () => {
                resultsCurrentPage = pageIndex;
                renderResultsPage();
            },
        }));
    }

    if (resultsCurrentPage < totalPages - 1) {
        container.appendChild(el('button', {
            type: 'button',
            className: 'btn btn-sm',
            textContent: '>',
            onClick: () => {
                resultsCurrentPage += 1;
                renderResultsPage();
            },
        }));
    }
}

function renderResultsPage() {
    const list = $('#results-list');
    if (!list) return;

    const total = currentResults.length;
    const totalPages = total === 0 ? 0 : Math.ceil(total / RESULTS_PAGE_SIZE);
    if (totalPages > 0 && resultsCurrentPage >= totalPages) {
        resultsCurrentPage = Math.max(0, totalPages - 1);
    }

    list.innerHTML = '';
    const start = resultsCurrentPage * RESULTS_PAGE_SIZE;
    const slice = currentResults.slice(start, start + RESULTS_PAGE_SIZE);
    for (const result of slice) {
        list.appendChild(buildResultCard(result));
    }
    list.scrollTop = 0;
    updateResultsPagination();
}

function renderResults(results) {
    currentResults = results;
    resultsCurrentPage = 0;
    renderResultsPage();
}

function renderTagCategory(label, type, tags) {
    const entries = Object.entries(tags);
    if (entries.length === 0 && type !== 'rating') return el('div');

    return el('div', { className: 'tag-category' }, [
        el('div', { className: `tag-category-label ${type}`, textContent: `${label} (${entries.length})` }),
        el('div', { className: 'tag-chips' },
            entries.map(([name, conf]) =>
                el('span', { className: `tag-chip ${type}` }, [
                    document.createTextNode(name.replace(/_/g, ' ')),
                    el('span', { className: 'conf', textContent: ` ${(conf * 100).toFixed(1)}%` }),
                    el('span', {
                        className: 'remove-tag',
                        textContent: '\u00d7',
                        onClick: (e) => {
                            e.target.closest('.tag-chip').remove();
                            delete tags[name];
                        },
                    }),
                ])
            )
        ),
    ]);
}

function resolveInferenceBatch(cfg) {
    const n = Number(cfg.batch_size);
    if (!Number.isFinite(n) || n < 1 || n > 256) {
        return null;
    }
    return Math.floor(n);
}

async function runTagging(fileIds, options = {}) {
    const pre = await api.getTaggingSessionStatus();
    if (pre.success && pre.active && !isControllerTab()) {
        alert(
            'Tagging is already running in another tab. Use that tab to stop, pause, or flush, '
            + 'or use “View progress (read-only)” in the banner.',
        );
        return;
    }

    if (!Array.isArray(fileIds) || fileIds.length === 0) {
        alert('No files to tag. Select images in the gallery or run a search first.');
        return;
    }

    const tagAll = options.tagAll === true;
    const sessionAutoTune = tagAll && $('#check-session-auto-tune')?.checked === true;
    const learningCalibration =
        tagAll && $('#check-learning-phase-calibration')?.checked === true;
    const performanceTuning = tagAll && sessionAutoTune;
    let generalThreshold = parseFloat($('#slider-general').value);
    let characterThreshold = parseFloat($('#slider-character').value);
    if (!Number.isFinite(generalThreshold)) generalThreshold = 0.35;
    if (!Number.isFinite(characterThreshold)) characterThreshold = 0.85;
    generalThreshold = Math.min(1, Math.max(0, generalThreshold));
    characterThreshold = Math.min(1, Math.max(0, characterThreshold));
    const modelName = $('#select-model').value;

    const [cfgRes, lmRes] = await Promise.all([
        api.getConfig(),
        api.listModels(),
    ]);
    const cfg = cfgRes.success ? cfgRes.config : {};
    const sessionAutoTuneThreads =
        sessionAutoTune &&
        cfg.use_gpu !== true &&
        $('#check-session-auto-tune-threads')?.checked === true;
    const modelEntry =
        lmRes.success && Array.isArray(lmRes.models)
            ? lmRes.models.find((m) => m.name === modelName)
            : null;
    const mustDownloadFromHub = Boolean(modelEntry && modelEntry.downloaded === false);

    const effectiveBatch = resolveInferenceBatch(cfg);
    if (effectiveBatch == null) {
        alert('Inference batch size must be between 1 and 256.');
        return;
    }

    let incremental = $('#check-incremental-hydrus')?.checked === true;
    const applyEveryRaw = $('#input-config-apply-every')?.value?.trim();
    let applyEvery = 0;
    if (incremental) {
        applyEvery = parseInt(applyEveryRaw, 10);
        if (!Number.isFinite(applyEvery) || applyEvery < 1) {
            applyEvery = Math.max(1, Math.floor(Number(cfg.batch_size) || 8));
        }
        if (applyEvery > 256) applyEvery = 256;
    }

    if (tagAll) {
        if (!$('#select-service').value) {
            alert(
                'Choose a tag service. When tagging all search results, each inference batch is sent to Hydrus as soon as it is tagged.',
            );
            return;
        }
        incremental = true;
        applyEvery = effectiveBatch;
    }

    if (incremental && applyEvery > 0 && !$('#select-service').value) {
        alert('Choose a tag service before using push tags to Hydrus while tagging.');
        return;
    }

    observerOverlayOpen = false;
    setProgressControlMode({ controller: true });
    showProgress(fileIds.length, { sessionAutoTune });
    const actions = $('#progress-actions');
    actions.style.display = 'flex';
    $('#btn-progress-pause').style.display = 'inline-block';
    $('#btn-progress-resume').style.display = 'none';
    const showFlush = incremental && applyEvery > 0;
    $('#btn-progress-flush').style.display = showFlush ? 'inline-block' : 'none';
    resetTaggingProgressChrome();

    let lastProgressCurrent = 0;
    let lastProgressBarCurrent = null;
    let lastProgressBarTotal = null;
    let lastInferTotal = null;
    let lastQueuePlanSkipSame = null;
    let lastQueuePlanSkipHi = null;
    let lastCumulativeInferredNonSkip = null;
    let lastPending = 0;
    let lastBatchesCompleted = '—';
    let lastBatchesTotal = '—';
    let lastTotalApplied = 0;
    let lastTotalTagsWritten = 0;
    let lastTotalDuplicatesSkipped = 0;
    let lastBatchInferred = '—';
    let lastCumulativeSkippedSameModelMarker = 0;
    let lastCumulativeSkippedHigherTierMarker = 0;
    let lastCumulativeWdStaleMarkersRemoved = 0;
    let lastBatchSkippedMarker = 0;
    let lastBatchSkippedSameModelMarker = 0;
    let lastBatchSkippedHigherTier = 0;
    let lastPerfTuning = null;
    let lastPerfHistoryLen = 0;
    let lastCalibrationPhase = null;
    let lastTuningState = null;
    let throughputEpochMs = null;
    let lastInMarkerSkipTail = false;

    const progressObserverUi = observerOverlayOpen || !isControllerTab();

    const barAmounts = () =>
        resolveProgressBarCounts({
            fileIdsLen: fileIds.length,
            lastProgressBarCurrent,
            lastProgressBarTotal,
            lastProgressCurrent,
            lastInferTotal,
        });

    const statsSnapshot = () => {
        syncProgressSessionTuneLine(lastTuningState, sessionAutoTune, progressObserverUi);
        syncProgressLearningLine(
            lastCalibrationPhase,
            lastTuningState,
            sessionAutoTune,
            learningCalibration,
        );
        const rates = computeThroughputRates(throughputEpochMs, lastProgressCurrent, lastTotalTagsWritten);
        return formatTaggingStats({
            current: lastProgressCurrent,
            totalFiles: fileIds.length,
            inferTotal: lastInferTotal,
            queueSkipSame: lastQueuePlanSkipSame,
            queueSkipHi: lastQueuePlanSkipHi,
            cumulativeInferredNonSkip: lastCumulativeInferredNonSkip,
            inferenceBatch: effectiveBatch,
            lastBatchInferred,
            lastBatchSkippedMarker,
            lastBatchSkippedSameModelMarker,
            lastBatchSkippedHigherTier,
            batchesCompleted: lastBatchesCompleted,
            batchesTotal: lastBatchesTotal,
            totalApplied: lastTotalApplied,
            totalTagsWritten: lastTotalTagsWritten,
            totalDuplicatesSkipped: lastTotalDuplicatesSkipped,
            pendingRemaining: lastPending,
            cumulativeSkippedSameModelMarker: lastCumulativeSkippedSameModelMarker,
            cumulativeSkippedHigherTierMarker: lastCumulativeSkippedHigherTierMarker,
            cumulativeWdStaleMarkersRemoved: lastCumulativeWdStaleMarkersRemoved,
            inMarkerSkipTail: lastInMarkerSkipTail,
            throughputFilesPerSec: rates.filesPerSec,
            throughputTagsPerSec: rates.tagsPerSec,
            perfTuningSummary: formatPerfTuningSummary(lastPerfTuning, lastPerfHistoryLen),
            calibrationLine:
                learningCalibration && sessionAutoTune
                    ? formatCalibrationOneLine(lastCalibrationPhase, lastTuningState, {
                          observer: progressObserverUi,
                      })
                    : '',
        });
    };

    const armProgressUi = () => {
        requestProgressFrame(() => {
            const ib = effectiveBatch;
            const binf = lastBatchInferred;
            const skipSum = lastBatchSkippedSameModelMarker + lastBatchSkippedHigherTier;
            const phase =
                binf > 0 && skipSum >= binf ? 'marker_skip' : 'inference';
            setProgressActivityPhase(phase, {
                titleSuffix: formatThroughputTitleSuffix(
                    throughputEpochMs,
                    lastProgressCurrent,
                    lastTotalTagsWritten,
                ),
            });
            const detail = `Inference batch: ${ib} · last run: ${binf} file(s)${
                lastBatchSkippedMarker > 0
                    ? ` · ${lastBatchSkippedMarker} skipped pre-infer`
                    : ''
            }`;
            const label = 'Tagging';
            const { cur: barCur, tot: barTot } = barAmounts();
            updateProgress(barCur, barTot, label, detail, statsSnapshot());
            syncProgressPerfElement(lastPerfTuning, lastPerfHistoryLen);
            syncTrainingProgressBar(lastTuningState, {
                sessionAutoTune,
                learningCalibration,
            });
        });
    };

    try {
        const loadTitle = mustDownloadFromHub
            ? 'Downloading model from HuggingFace…'
            : 'Loading model into memory…';
        const loadDetail = [
            mustDownloadFromHub
                ? 'First run: ~300–600 MB saved under your models folder. Watch the server terminal for per-file download logs.'
                : 'Model files are reused from disk when already downloaded.',
            tagAll
                ? `Tag all: each batch of ${effectiveBatch} files is written to Hydrus when inferred.`
                : `Inference batch size: ${effectiveBatch} (from Settings → Performance).`,
            learningCalibration && sessionAutoTune
                ? 'Learning-phase calibration: the first queue segment explores batch/Hydrus/ORT settings without writing tags; the rest of the run applies tags using the best settings found (see progress panel).'
                : '',
        ].filter(Boolean).join(' ');
        {
            const b0 = barAmounts();
            updateProgress(b0.cur, b0.tot, loadTitle, loadDetail, statsSnapshot());
        }
        setProgressActivityPhase('load');

        const loadResult = await api.loadModel(modelName);
        if (!loadResult.success) {
            alert('Failed to load model: ' + loadResult.error);
            return;
        }

        if (loadResult.downloaded_from_hub) {
            const bdl = barAmounts();
            updateProgress(
                bdl.cur,
                bdl.tot,
                'Model downloaded and loaded',
                'Starting tagging session…',
                statsSnapshot(),
            );
        }
        setProgressActivityPhase('run');

        const payload = {
            file_ids: fileIds,
            general_threshold: generalThreshold,
            character_threshold: characterThreshold,
            model_name: modelName,
            batch_size: effectiveBatch,
            // Sent for every run so the server can detect the model marker on the intended service.
            // Hydrus writes still only occur when incremental apply is enabled.
            service_key: $('#select-service').value || '',
            apply_tags_every_n: incremental ? applyEvery : 0,
            stream_verbose: false,
            hydrus_download_parallel: cfg.hydrus_download_parallel ?? 8,
            tag_all: tagAll,
            performance_tuning: performanceTuning,
        };
        if (sessionAutoTune) {
            payload.session_auto_tune = true;
            payload.tuning_control_mode = $('#select-tuning-control-mode')?.value || 'auto_lucky';
        }
        if (sessionAutoTuneThreads) {
            payload.session_auto_tune_threads = true;
        }
        if (learningCalibration) {
            payload.learning_phase_calibration = true;
            const lfRaw = $('#input-learning-fraction')?.value?.trim();
            const lf = parseFloat(lfRaw || '0.1');
            payload.learning_fraction = Number.isFinite(lf) ? lf : 0.1;
            const lscope = ($('#select-learning-scope')?.value || 'count').trim().toLowerCase();
            payload.learning_scope = lscope === 'bytes' ? 'bytes' : 'count';
        }
        if (performanceTuning) {
            payload.performance_tuning_window = 32;
        }

        const showHydrusRecovery = (msg) => {
            const ov = $('#hydrus-recovery-overlay');
            const lead = $('#hydrus-recovery-lead');
            const stats = $('#hydrus-recovery-stats');
            const poll = $('#hydrus-recovery-poll');
            const title = $('#hydrus-recovery-title');
            if (!ov || !lead || !stats) return;
            if (title) title.textContent = 'Hydrus unreachable';
            setProgressActivityPhase('wait_hydrus');
            lead.textContent =
                msg.message
                || 'The Hydrus client or API became unreachable. Tagging will continue automatically when Hydrus responds.';
            stats.innerHTML = '';
            const rows = [
                `Inferred so far: ${msg.inferred_so_far ?? '—'} file(s)`,
                `Still queued for inference: ${msg.remaining_infer_count ?? '—'} file(s)`,
                `Pending write to Hydrus (tags not committed yet): ${msg.pending_commit_count ?? '—'} file(s)`,
            ];
            const pc = msg.pending_commit_file_ids;
            const ri = msg.remaining_infer_file_ids;
            if (Array.isArray(pc) && pc.length) {
                const sample = pc.slice(0, 14).join(', ');
                rows.push(`Pending commit IDs (sample): ${sample}${pc.length > 14 ? '…' : ''}`);
            }
            if (Array.isArray(ri) && ri.length) {
                const sample = ri.slice(0, 14).join(', ');
                rows.push(`Remaining infer IDs (sample): ${sample}${ri.length > 14 ? '…' : ''}`);
            }
            for (const t of rows) {
                stats.appendChild(el('li', { textContent: t }));
            }
            if (poll) {
                poll.style.display = 'none';
                poll.textContent = '';
            }
            ov.style.display = 'flex';
            const pt = $('#progress-title');
            if (pt) pt.textContent = 'Waiting for Hydrus…';
        };

        const hideHydrusRecovery = () => {
            const ov = $('#hydrus-recovery-overlay');
            if (ov) ov.style.display = 'none';
        };

        const { cancel, pause, resume, flush, retryHydrus, tuningAck, done } = api.startTaggingWebSocket(payload, {
            onQueuePlan(msg) {
                lastInferTotal = msg.infer_total ?? null;
                lastQueuePlanSkipSame = msg.skip_same_marker ?? null;
                lastQueuePlanSkipHi = msg.skip_higher_tier ?? null;
                throughputEpochMs = throughputEpochMs ?? Date.now();
                armProgressUi();
            },
            onControlAck(msg) {
                if (msg.action === 'pause') {
                    $('#btn-progress-pause').style.display = 'none';
                    $('#btn-progress-resume').style.display = 'inline-block';
                    setProgressActivityPhase('paused', {
                        titleSuffix: formatThroughputTitleSuffix(
                            throughputEpochMs,
                            lastProgressCurrent,
                            lastTotalTagsWritten,
                        ),
                    });
                    {
                        const bp = barAmounts();
                        updateProgress(
                            bp.cur,
                            bp.tot,
                            'Paused',
                            `Resume to continue. ONNX batch size: ${effectiveBatch}.`,
                            statsSnapshot(),
                        );
                    }
                } else if (msg.action === 'resume') {
                    $('#btn-progress-pause').style.display = 'inline-block';
                    $('#btn-progress-resume').style.display = 'none';
                } else if (msg.action === 'tuning_ack') {
                    const ap = $('#btn-tuning-approve');
                    if (ap) {
                        ap.style.display = 'none';
                        setTuningApproveAttention(false);
                    }
                }
            },
            onTuningTimeout(msg) {
                const detail =
                    msg.message
                    || 'Supervised tuning approval timed out — session paused; resume to continue.';
                setProgressActivityPhase('paused', {
                    titleSuffix: formatThroughputTitleSuffix(
                        throughputEpochMs,
                        lastProgressCurrent,
                        lastTotalTagsWritten,
                    ),
                });
                {
                    const bt = barAmounts();
                    updateProgress(bt.cur, bt.tot, 'Tuning approval timed out', detail, statsSnapshot());
                }
            },
            onProgress(msg) {
                throughputEpochMs = throughputEpochMs ?? Date.now();
                if (msg.tuning_state != null) lastTuningState = msg.tuning_state;
                if (msg.calibration_phase != null) lastCalibrationPhase = msg.calibration_phase;
                if (msg.in_marker_skip_tail != null) {
                    lastInMarkerSkipTail = Boolean(msg.in_marker_skip_tail);
                }
                lastProgressCurrent = msg.current ?? lastProgressCurrent;
                if (msg.infer_total != null) lastInferTotal = msg.infer_total;
                if (msg.cumulative_inferred_non_skip != null) {
                    lastCumulativeInferredNonSkip = msg.cumulative_inferred_non_skip;
                }
                if (msg.progress_bar_current != null) {
                    lastProgressBarCurrent = msg.progress_bar_current;
                }
                if (msg.progress_bar_total != null) {
                    lastProgressBarTotal = msg.progress_bar_total;
                }
                if (msg.batch_inferred != null) lastBatchInferred = msg.batch_inferred;
                if (msg.batch_skipped_inference != null) {
                    lastBatchSkippedMarker = msg.batch_skipped_inference;
                }
                if (msg.batch_skipped_same_model_marker != null) {
                    lastBatchSkippedSameModelMarker = msg.batch_skipped_same_model_marker;
                }
                if (msg.batch_skipped_higher_tier_model_marker != null) {
                    lastBatchSkippedHigherTier = msg.batch_skipped_higher_tier_model_marker;
                }
                if (msg.cumulative_skipped_higher_tier_model_marker != null) {
                    lastCumulativeSkippedHigherTierMarker = msg.cumulative_skipped_higher_tier_model_marker;
                }
                if (msg.performance_tuning != null) {
                    lastPerfTuning = msg.performance_tuning;
                }
                if (Array.isArray(msg.performance_tuning_history)) {
                    lastPerfHistoryLen = msg.performance_tuning_history.length;
                }
                if (msg.batches_completed != null) lastBatchesCompleted = msg.batches_completed;
                if (msg.batches_total != null) lastBatchesTotal = msg.batches_total;
                if (msg.total_applied != null) lastTotalApplied = msg.total_applied;
                if (msg.total_tags_written != null) lastTotalTagsWritten = msg.total_tags_written;
                if (msg.total_duplicates_skipped != null) {
                    lastTotalDuplicatesSkipped = msg.total_duplicates_skipped;
                }
                if (msg.cumulative_skipped_same_model_marker != null) {
                    lastCumulativeSkippedSameModelMarker = msg.cumulative_skipped_same_model_marker;
                }
                if (msg.cumulative_wd_stale_markers_removed != null) {
                    lastCumulativeWdStaleMarkersRemoved = msg.cumulative_wd_stale_markers_removed;
                }
                const ts = msg.tuning_state;
                const approveBtn = $('#btn-tuning-approve');
                if (approveBtn && ts && ts.awaiting_approval === true) {
                    approveBtn.style.display = 'inline-block';
                    setTuningApproveAttention(true);
                } else if (approveBtn && (!ts || !ts.awaiting_approval)) {
                    approveBtn.style.display = 'none';
                    setTuningApproveAttention(false);
                }
                armProgressUi();
            },
            onStopping(msg) {
                const detail =
                    msg.message
                    || 'Finishing the current batch if in progress, then flushing any pending Hydrus writes.';
                setProgressActivityPhase('stopping', {
                    titleSuffix: formatThroughputTitleSuffix(
                        throughputEpochMs,
                        lastProgressCurrent,
                        lastTotalTagsWritten,
                    ),
                });
                {
                    const bs = barAmounts();
                    updateProgress(bs.cur, bs.tot, 'Stopping…', detail, statsSnapshot());
                }
            },
            onServerShuttingDown(msg) {
                expectServerShutdownSoon();
                const detail =
                    msg.message
                    || 'Server stop requested — pending Hydrus writes will flush where possible; tagging will cancel.';
                setProgressActivityPhase('stopping', {
                    titleSuffix: formatThroughputTitleSuffix(
                        throughputEpochMs,
                        lastProgressCurrent,
                        lastTotalTagsWritten,
                    ),
                });
                {
                    const bx = barAmounts();
                    updateProgress(bx.cur, bx.tot, 'Server stopping', detail, statsSnapshot());
                }
            },
            onHydrusUnreachable(msg) {
                showHydrusRecovery(msg);
            },
            onHydrusWaiting(msg) {
                const poll = $('#hydrus-recovery-poll');
                if (poll) {
                    poll.style.display = 'block';
                    const s = msg.next_poll_s != null ? Number(msg.next_poll_s) : 12;
                    poll.textContent = `Polling every ${s}s — or click Retry Hydrus now.`;
                }
            },
            onHydrusRecovered(msg) {
                hideHydrusRecovery();
                const pt = $('#progress-title');
                if (pt) pt.textContent = 'Working…';
                setProgressActivityPhase('run');
                {
                    const br = barAmounts();
                    updateProgress(
                        br.cur,
                        br.tot,
                        'Tagging',
                        msg.message || 'Continuing tagging…',
                        statsSnapshot(),
                    );
                }
            },
            onTagsApplied(msg) {
                lastTotalApplied = msg.total_applied ?? lastTotalApplied;
                lastTotalTagsWritten = msg.total_tags_written ?? lastTotalTagsWritten;
                if (msg.total_duplicates_skipped != null) {
                    lastTotalDuplicatesSkipped = msg.total_duplicates_skipped;
                }
                lastPending = msg.pending_remaining ?? 0;
                const kind = msg.manual_flush ? 'Manual flush to Hydrus' : 'Wrote batch to Hydrus';
                const chunk = msg.chunk_tag_count != null ? msg.chunk_tag_count : msg.count ?? 0;
                const dchunk = msg.chunk_duplicates_skipped != null ? msg.chunk_duplicates_skipped : 0;
                const dupPart = dchunk > 0 ? ` · skipped ${dchunk} already present` : '';
                requestProgressFrame(() => {
                    setProgressActivityPhase('hydrus', {
                        titleSuffix: formatThroughputTitleSuffix(
                            throughputEpochMs,
                            lastProgressCurrent,
                            lastTotalTagsWritten,
                        ),
                    });
                    const bh = barAmounts();
                    updateProgress(
                        bh.cur,
                        bh.tot,
                        'Tagging',
                        `${kind}: +${chunk} new tag strings${dupPart} · Hydrus files updated: ${lastTotalApplied} · pending: ${lastPending}`,
                        statsSnapshot(),
                    );
                });
            },
        });

        $('#btn-progress-stop').onclick = () => {
            const stopBtn = $('#btn-progress-stop');
            if (stopBtn) {
                stopBtn.disabled = true;
                stopBtn.textContent = 'Stopping…';
            }
            for (const sel of [
                '#btn-progress-pause',
                '#btn-progress-resume',
                '#btn-progress-flush',
                '#btn-tuning-approve',
            ]) {
                const b = $(sel);
                if (b) b.disabled = true;
            }
            const hint = $('#progress-stop-hint');
            if (hint) {
                hint.style.display = 'block';
                hint.textContent = tagAll
                    ? 'Winding down: current batch may finish; the server flushes any partial queue to Hydrus, then stops. Tag all already wrote completed batches — you may not need Apply all tags afterward.'
                    : 'Winding down: current inference may finish; pending Hydrus queue will flush when safe.';
            }
            setProgressActivityPhase('stopping', {
                titleSuffix: formatThroughputTitleSuffix(
                    throughputEpochMs,
                    lastProgressCurrent,
                    lastTotalTagsWritten,
                ),
            });
            {
                const bq = barAmounts();
                updateProgress(
                    bq.cur,
                    bq.tot,
                    'Stopping…',
                    'Request sent — waiting for the server to finish this batch and flush pending writes…',
                    statsSnapshot(),
                );
            }
            cancel();
        };
        $('#btn-progress-pause').onclick = () => pause();
        $('#btn-progress-resume').onclick = () => resume();
        $('#btn-progress-flush').onclick = () => flush();
        $('#btn-tuning-approve').onclick = () => tuningAck(true);

        const btnHr = $('#btn-hydrus-retry-now');
        const btnHc = $('#btn-hydrus-cancel-recovery');
        if (btnHr) btnHr.onclick = () => retryHydrus();
        if (btnHc) {
            btnHc.onclick = () => {
                hideHydrusRecovery();
                cancel();
                syncConnectionShell('disconnected');
                setState({ connected: false, services: [] });
                hideProgress();
            };
        }

        const final = await done;

        const doneCurrent = final.total_processed ?? final.results?.length ?? 0;
        lastProgressCurrent = doneCurrent;
        lastBatchesCompleted = final.batches_completed ?? lastBatchesCompleted;
        lastBatchesTotal = final.batches_total ?? lastBatchesTotal;
        lastTotalApplied = final.total_applied ?? lastTotalApplied;
        lastTotalTagsWritten = final.total_tags_written ?? lastTotalTagsWritten;
        if (final.total_duplicates_skipped != null) {
            lastTotalDuplicatesSkipped = final.total_duplicates_skipped;
        }
        if (final.cumulative_skipped_same_model_marker != null) {
            lastCumulativeSkippedSameModelMarker = final.cumulative_skipped_same_model_marker;
        }
        if (final.cumulative_skipped_higher_tier_model_marker != null) {
            lastCumulativeSkippedHigherTierMarker = final.cumulative_skipped_higher_tier_model_marker;
        }
        if (final.cumulative_wd_stale_markers_removed != null) {
            lastCumulativeWdStaleMarkersRemoved = final.cumulative_wd_stale_markers_removed;
        }
        lastPending = final.pending_hydrus_files ?? 0;
        setProgressActivityPhase('done', {
            titleSuffix: formatThroughputTitleSuffix(
                throughputEpochMs,
                lastProgressCurrent,
                lastTotalTagsWritten,
            ),
        });
        {
            const n = Math.max(1, fileIds.length);
            const dc = Math.min(doneCurrent, n);
            updateProgress(
                dc,
                n,
                final.stopped ? 'Stopped' : 'Done',
                `Inference batch size was ${final.inference_batch ?? effectiveBatch}.`,
                statsSnapshot(),
            );
        }
        syncProgressPerfElement(lastPerfTuning);

        const allResults = final.results || [];
        setState({ tagResults: allResults });
        currentResults = allResults;

        hide('#section-gallery');
        show('#section-results');

        const serviceKeyForRun = payload.service_key || '';
        const pendingN = final.pending_hydrus_files ?? 0;
        const noManualApplyNeeded = Boolean(serviceKeyForRun) && pendingN === 0;

        const summaryEl = $('#results-run-summary');
        if (summaryEl) {
            summaryEl.textContent = formatRunSummary({
                tagAll,
                stopped: Boolean(final.stopped),
                final,
                queuedTotal: fileIds.length,
                serviceKeyForRun,
                pendingN,
            });
        }

        const btnApply = $('#btn-apply-tags');
        if (btnApply) {
            if (noManualApplyNeeded) {
                btnApply.disabled = true;
                btnApply.title =
                    'Nothing pending on the selected tag service for this run (tags were written during tagging or already match Hydrus).';
            } else {
                btnApply.disabled = false;
                btnApply.removeAttribute('title');
            }
        }

        renderResults(allResults);
    } catch (err) {
        if (err.code === 'tagging_busy') {
            alert(err.message);
            await refreshRemoteTaggingUi();
        } else {
            alert('Tagging error: ' + err.message);
        }
    } finally {
        actions.style.display = 'none';
        $('#btn-progress-stop').onclick = null;
        $('#btn-progress-pause').onclick = null;
        $('#btn-progress-resume').onclick = null;
        $('#btn-progress-flush').onclick = null;
        $('#btn-tuning-approve').onclick = null;
        const ap = $('#btn-tuning-approve');
        if (ap) {
            ap.style.display = 'none';
            ap.disabled = false;
            setTuningApproveAttention(false);
        }
        resetTaggingProgressChrome();
        hideProgress();
        const hro = $('#hydrus-recovery-overlay');
        if (hro) hro.style.display = 'none';
        const btnHrF = $('#btn-hydrus-retry-now');
        const btnHcF = $('#btn-hydrus-cancel-recovery');
        if (btnHrF) btnHrF.onclick = null;
        if (btnHcF) btnHcF.onclick = null;
    }
}

async function applyTags() {
    const serviceKey = $('#select-service').value;
    if (!serviceKey) {
        alert('Please select a tag service');
        return;
    }

    const generalPrefix = $('#input-general-prefix')?.value || '';
    const characterPrefix = $('#input-character-prefix')?.value || 'character:';
    const ratingPrefix = $('#input-rating-prefix')?.value || 'rating:';

    const rows = currentResults
        .map((r) => {
            let tags;
            if (Array.isArray(r.tags) && r.tags.length > 0) {
                tags = [...r.tags];
            } else {
                tags = [];
                for (const [name] of Object.entries(r.general_tags || {})) {
                    const tag = name.replace(/_/g, ' ');
                    tags.push(generalPrefix ? `${generalPrefix}${tag}` : tag);
                }
                for (const [name] of Object.entries(r.character_tags || {})) {
                    tags.push(`${characterPrefix}${name.replace(/_/g, ' ')}`);
                }
                const ratingEntries = Object.entries(r.rating_tags || {});
                if (ratingEntries.length > 0) {
                    const topRating = ratingEntries.reduce((a, b) => (a[1] > b[1] ? a : b))[0];
                    tags.push(`${ratingPrefix}${topRating}`);
                }
            }
            return {
                file_id: r.file_id,
                hash: r.hash,
                tags,
            };
        })
        .filter((row) => row.hash && row.tags.length > 0);

    if (rows.length === 0) {
        alert('Nothing to apply — all tags are already in Hydrus or results are empty.');
        return;
    }

    const cfgRes = await api.getConfig();
    let batch = 100;
    if (cfgRes.success && cfgRes.config && cfgRes.config.apply_tags_http_batch_size != null) {
        const n = Number(cfgRes.config.apply_tags_http_batch_size);
        if (Number.isFinite(n)) {
            batch = Math.max(1, Math.min(512, Math.floor(n)));
        }
    }

    showProgress(rows.length);
    let applied = 0;
    let skippedDup = 0;
    for (let off = 0; off < rows.length; off += batch) {
        const chunk = rows.slice(off, off + batch);
        updateProgress(
            Math.min(off + chunk.length, rows.length),
            rows.length,
            'Applying tags…',
            `Batch ${Math.floor(off / batch) + 1} (${chunk.length} file(s))`,
            '',
        );
        const result = await api.applyTags(chunk, serviceKey);
        if (!result.success) {
            hideProgress();
            alert('Apply failed: ' + result.error);
            return;
        }
        applied += result.applied ?? 0;
        skippedDup += result.skipped_duplicate_tags ?? 0;
    }

    hideProgress();

    let msg = `Applied tags to ${applied} file(s).`;
    if (skippedDup > 0) {
        msg += ` (${skippedDup} tag string(s) were already on the file in Hydrus and were skipped.)`;
    }
    alert(msg);
}

export function initTagger() {
    const banner = $('#tagging-busy-banner');
    const btnObs = $('#btn-observer-progress');

    refreshRemoteTaggingUi = async () => {
        const r = await api.getTaggingSessionStatus();
        if (!r.success) return;
        const controller = isControllerTab();
        const remoteBusy = r.active && !controller;
        setState({ taggingLockedByOtherTab: remoteBusy });
        const tagAllBtn = $('#btn-tag-all');
        if (tagAllBtn) tagAllBtn.disabled = remoteBusy;
        if (remoteBusy) {
            banner.style.display = 'flex';
            const m = r.snapshot?.model_name;
            $('#tagging-busy-text').textContent = m
                ? `Tagging is running in another tab (${m}).`
                : 'Tagging is running in another tab.';
        } else {
            banner.style.display = 'none';
            if (observerOverlayOpen && !r.active) {
                observerOverlayOpen = false;
                setProgressControlMode({ controller: true });
                hideProgress();
                restartTaggingSessionPoll();
            }
        }
        if (observerOverlayOpen && r.snapshot) {
            applySnapshotToObserverOverlay(r.snapshot);
        }
    };

    btnObs?.addEventListener('click', async () => {
        const r = await api.getTaggingSessionStatus();
        if (!r.success || !r.active || !r.snapshot) {
            alert('No active tagging session — the other tab may have finished.');
            return;
        }
        observerOverlayOpen = true;
        const total = Math.max(1, r.snapshot.total ?? r.snapshot.total_files ?? 1);
        showProgress(total);
        setProgressControlMode({ controller: false });
        applySnapshotToObserverOverlay(r.snapshot);
        restartTaggingSessionPoll();
    });

    restartTaggingSessionPoll();
    document.addEventListener('visibilitychange', () => {
        restartTaggingSessionPoll();
    });

    $('#slider-general').addEventListener('input', (e) => {
        $('#val-general').textContent = parseFloat(e.target.value).toFixed(2);
    });
    $('#slider-character').addEventListener('input', (e) => {
        $('#val-character').textContent = parseFloat(e.target.value).toFixed(2);
    });

    api.getConfig().then(res => {
        if (!res.success || !res.config) return;
        const c = res.config;
        const n = c.apply_tags_every_n;
        const elN = $('#input-config-apply-every');
        const inc = $('#check-incremental-hydrus');
        if (n != null && elN && inc) {
            const num = Number(n);
            inc.checked = num > 0;
            elN.value = num > 0 ? String(num) : '8';
        }
        syncIncrementalHydrusApplyEveryVisibility();
    });

    $('#btn-tag-selected').addEventListener('click', () => {
        const state = getState();
        const ids = Array.from(state.selectedIds);
        if (ids.length === 0) {
            alert('Select one or more images first');
            return;
        }
        runTagging(ids);
    });

    $('#btn-tag-all').addEventListener('click', () => {
        const state = getState();
        if (state.fileIds.length === 0) {
            alert('Search for images first');
            return;
        }
        if (state.fileIds.length > 100) {
            if (!confirm(`Tag all ${state.fileIds.length} search results? This may take a while.`)) return;
        }
        runTagging(state.fileIds, { tagAll: true });
    });

    $('#btn-apply-tags').addEventListener('click', applyTags);

    $('#btn-back-gallery').addEventListener('click', () => {
        const sum = $('#results-run-summary');
        if (sum) sum.textContent = '';
        const rpi = $('#results-page-info');
        if (rpi) rpi.textContent = '';
        const rpg = $('#results-pagination');
        if (rpg) rpg.innerHTML = '';
        resultsCurrentPage = 0;
        const btnApply = $('#btn-apply-tags');
        if (btnApply) {
            btnApply.disabled = false;
            btnApply.removeAttribute('title');
        }
        hide('#section-results');
        show('#section-gallery');
    });
}

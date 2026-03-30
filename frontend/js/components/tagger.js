/**
 * Tagger controls and results display component.
 */

import { api } from '../api.js';
import { getState, setState } from '../state.js';
import { $, el, show, hide } from '../utils/dom.js';
import {
    showProgress,
    hideProgress,
    updateProgress,
    setProgressControlMode,
    requestProgressFrame,
} from './progress.js';
import { expectServerShutdownSoon } from '../server_offline.js';

let currentResults = [];
let observerOverlayOpen = false;
let refreshRemoteTaggingUi = async () => {};
let taggingSessionPollTimer = null;

/** One-line summary from WebSocket `performance_tuning` object (Tag all + tuning only). */
function formatPerfTuningSummary(pt) {
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
    return `${bi}: ${parts.join(' · ')}`;
}

function syncProgressPerfElement(pt) {
    const perfEl = $('#progress-perf-tuning');
    if (!perfEl) return;
    const line = formatPerfTuningSummary(pt);
    if (line) {
        perfEl.textContent = line;
        perfEl.style.display = 'block';
    } else {
        perfEl.textContent = '';
        perfEl.style.display = 'none';
    }
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
    return formatTaggingStats({
        current: snap.current ?? snap.total_processed ?? 0,
        totalFiles: total,
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
        perfTuningSummary: formatPerfTuningSummary(snap.performance_tuning),
    });
}

function applySnapshotToObserverOverlay(snap) {
    if (!observerOverlayOpen || !snap) return;
    const total = Math.max(1, snap.total ?? snap.total_files ?? 1);
    const cur = Math.min(snap.current ?? snap.total_processed ?? 0, total);
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
    const showBar = $('#check-show-progress-bar')?.checked !== false;
    updateProgress(cur, total, title, detail, statsFromSnapshot(snap, total), { showBar });
    syncProgressPerfElement(snap.performance_tuning);
}

function formatTaggingStats(s) {
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
    const lines = [
        batchLine,
        `Files processed: ${s.current} / ${s.totalFiles} (${s.totalFiles > 0 ? Math.round((s.current / s.totalFiles) * 100) : 0}% of queue)`,
        onnxDetail,
        `Hydrus: ${ta} file(s) received new tags · ${tt} new tag string(s) sent`,
    ];
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
                'Nothing is pending on the selected tag service — incremental writes already covered these results (or tags match Hydrus). Apply all tags to Hydrus is disabled.',
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

function renderResults(results) {
    currentResults = results;
    const list = $('#results-list');
    list.innerHTML = '';

    for (const result of results) {
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

        const card = el('div', { className: 'result-card' }, [
            el('img', {
                className: 'result-thumb',
                src: api.thumbnailUrl(result.file_id),
                alt: '',
            }),
            el('div', { className: 'result-tags' }, tagBlockChildren),
        ]);
        list.appendChild(card);
    }
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
    const cfgBatch = Number(cfg.batch_size) > 0 ? Number(cfg.batch_size) : 8;
    const raw = $('#input-inference-batch')?.value?.trim();
    if (raw === '' || raw === undefined) {
        return cfgBatch;
    }
    const n = parseInt(raw, 10);
    if (!Number.isFinite(n) || n < 1 || n > 256) {
        return null;
    }
    return n;
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

    const tagAll = options.tagAll === true;
    const performanceTuning = tagAll && $('#check-performance-tuning-tag-all')?.checked === true;
    const generalThreshold = parseFloat($('#slider-general').value);
    const characterThreshold = parseFloat($('#slider-character').value);
    const modelName = $('#select-model').value;

    const [cfgRes, lmRes] = await Promise.all([
        api.getConfig(),
        api.listModels(),
    ]);
    const cfg = cfgRes.success ? cfgRes.config : {};
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

    let incremental = $('#check-incremental-hydrus')?.checked;
    const applyEveryRaw = $('#input-apply-every-n')?.value?.trim();
    let applyEvery = 0;
    if (incremental) {
        if (applyEveryRaw === '' || applyEveryRaw === undefined) {
            applyEvery = Number(cfg.apply_tags_every_n) || 0;
        } else {
            applyEvery = parseInt(applyEveryRaw, 10);
        }
        if (!Number.isFinite(applyEvery) || applyEvery < 0) applyEvery = 0;
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

    const verbose = $('#check-verbose-ws')?.checked;
    const showBar = $('#check-show-progress-bar')?.checked !== false;

    if (incremental && applyEvery > 0 && !$('#select-service').value) {
        alert('Choose a tag service before enabling incremental writes to Hydrus.');
        return;
    }

    observerOverlayOpen = false;
    setProgressControlMode({ controller: true });
    showProgress(fileIds.length);
    const actions = $('#progress-actions');
    actions.style.display = 'flex';
    $('#btn-progress-pause').style.display = 'inline-block';
    $('#btn-progress-resume').style.display = 'none';
    const showFlush = incremental && applyEvery > 0;
    $('#btn-progress-flush').style.display = showFlush ? 'inline-block' : 'none';
    resetTaggingProgressChrome();

    let lastProgressCurrent = 0;
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

    const statsSnapshot = () =>
        formatTaggingStats({
            current: lastProgressCurrent,
            totalFiles: fileIds.length,
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
            perfTuningSummary: formatPerfTuningSummary(lastPerfTuning),
        });

    const armProgressUi = () => {
        requestProgressFrame(() => {
            const ib = effectiveBatch;
            const binf = lastBatchInferred;
            const detail = `Inference batch: ${ib} · last run: ${binf} file(s)${
                lastBatchSkippedMarker > 0
                    ? ` · ${lastBatchSkippedMarker} skipped pre-infer`
                    : ''
            }`;
            const label = `Tagged ${lastProgressCurrent} / ${fileIds.length}`;
            updateProgress(lastProgressCurrent, fileIds.length, label, detail, statsSnapshot(), {
                showBar,
            });
            syncProgressPerfElement(lastPerfTuning);
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
                : `Inference batch size: ${effectiveBatch} (this run).`,
        ].join(' ');
        updateProgress(0, fileIds.length, loadTitle, loadDetail, statsSnapshot(), { showBar });

        const loadResult = await api.loadModel(modelName);
        if (!loadResult.success) {
            alert('Failed to load model: ' + loadResult.error);
            return;
        }

        if (loadResult.downloaded_from_hub) {
            updateProgress(
                0,
                fileIds.length,
                'Model downloaded and loaded',
                'Starting tagging session…',
                statsSnapshot(),
                { showBar },
            );
        }

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
            stream_verbose: verbose,
            hydrus_download_parallel: cfg.hydrus_download_parallel ?? 8,
            tag_all: tagAll,
            performance_tuning: performanceTuning,
        };

        const { cancel, pause, resume, flush, done } = api.startTaggingWebSocket(payload, {
            onControlAck(msg) {
                if (msg.action === 'pause') {
                    $('#btn-progress-pause').style.display = 'none';
                    $('#btn-progress-resume').style.display = 'inline-block';
                    updateProgress(
                        lastProgressCurrent,
                        fileIds.length,
                        'Paused',
                        `Resume to continue. ONNX batch size: ${effectiveBatch}.`,
                        statsSnapshot(),
                        { showBar },
                    );
                } else if (msg.action === 'resume') {
                    $('#btn-progress-pause').style.display = 'inline-block';
                    $('#btn-progress-resume').style.display = 'none';
                }
            },
            onProgress(msg) {
                lastProgressCurrent = msg.current ?? lastProgressCurrent;
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
                if (msg.type === 'file') {
                    const ib = msg.inference_batch ?? effectiveBatch;
                    const binf = msg.batch_inferred != null ? msg.batch_inferred : '—';
                    updateProgress(
                        lastProgressCurrent,
                        fileIds.length,
                        `File ${lastProgressCurrent} / ${fileIds.length}`,
                        `Inference batch: ${ib} · last run: ${binf} file(s)`,
                        statsSnapshot(),
                        { showBar },
                    );
                    syncProgressPerfElement(lastPerfTuning);
                } else {
                    armProgressUi();
                }
            },
            onStopping(msg) {
                const detail =
                    msg.message
                    || 'Finishing the current batch if in progress, then flushing any pending Hydrus writes.';
                updateProgress(
                    lastProgressCurrent,
                    fileIds.length,
                    'Stopping…',
                    detail,
                    statsSnapshot(),
                    { showBar },
                );
            },
            onServerShuttingDown(msg) {
                expectServerShutdownSoon();
                const detail =
                    msg.message
                    || 'Server stop requested — pending Hydrus writes will flush where possible; tagging will cancel.';
                updateProgress(
                    lastProgressCurrent,
                    fileIds.length,
                    'Server stopping',
                    detail,
                    statsSnapshot(),
                    { showBar },
                );
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
                    updateProgress(
                        lastProgressCurrent,
                        fileIds.length,
                        `${kind} (+${chunk} new tag strings${dupPart})`,
                        `Hydrus files with new tags: ${lastTotalApplied} · pending queue: ${lastPending}`,
                        statsSnapshot(),
                        { showBar },
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
            for (const sel of ['#btn-progress-pause', '#btn-progress-resume', '#btn-progress-flush']) {
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
            updateProgress(
                lastProgressCurrent,
                fileIds.length,
                'Stopping…',
                'Request sent — waiting for the server to finish this batch and flush pending writes…',
                statsSnapshot(),
                { showBar },
            );
            cancel();
        };
        $('#btn-progress-pause').onclick = () => pause();
        $('#btn-progress-resume').onclick = () => resume();
        $('#btn-progress-flush').onclick = () => flush();

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
        updateProgress(
            doneCurrent,
            fileIds.length,
            final.stopped ? 'Stopped' : 'Done',
            `Inference batch size was ${final.inference_batch ?? effectiveBatch}.`,
            statsSnapshot(),
            { showBar },
        );
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
        resetTaggingProgressChrome();
        hideProgress();
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
            { showBar: true },
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
        setProgressControlMode({
            controller: false,
            showBar: $('#check-show-progress-bar')?.checked !== false,
        });
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
        const ib = $('#input-inference-batch');
        if (ib && c.batch_size != null) {
            ib.value = String(c.batch_size);
        }
        const n = c.apply_tags_every_n;
        const elN = $('#input-apply-every-n');
        if (elN && n != null) {
            elN.value = String(n);
            if (n > 0 && $('#check-incremental-hydrus')) {
                $('#check-incremental-hydrus').checked = true;
            }
        }
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
        const btnApply = $('#btn-apply-tags');
        if (btnApply) {
            btnApply.disabled = false;
            btnApply.removeAttribute('title');
        }
        hide('#section-results');
        show('#section-gallery');
    });
}

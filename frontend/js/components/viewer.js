/**
 * Full-screen image viewer: phased image load, Hydrus tag editing, WD predict/apply.
 */

import { api } from '../api.js';
import { getState, setState, subscribe } from '../state.js';
import { $, el } from '../utils/dom.js';
import { extractTagsByService, getTagsForService } from '../utils/hydrus.js';
import {
    orderedSelectedFileIds,
    viewerUsesSelectionOnlyNav,
    writeGalleryViewerCycleSelection,
} from '../utils/selection_nav.js';
import { hideGallerySelectionModeToast, showGallerySelectionModeToast } from './gallery_selection_toast.js';

export function resetViewerTripleClickState() {
    /* Reserved for future gesture state; search still calls this to reset UI assumptions. */
}

let _viewerGlobalIndex = -1;
/** File currently shown (predict/apply even when not in `fileIds`). */
let _viewerDisplayedFileId = null;
let _draftTags = [];
let _pendingAddTags = new Set();
let _pendingRemoveTags = new Set();
let _baseServiceTags = new Set();
let _loadGeneration = 0;
let _predictBusy = false;
let _applyBusy = false;

const CINEMA_LS_KEY = 'wd_tagger_viewer_cinema';

/** Normalized tag key for WD marker match (same rules as server ``normalize_tag_for_compare``). */
let _viewerWdMarkerNorm = '';
let _viewerWdMarkerFetched = false;

/** First wrap onto last / last item / wrap to first while in multi-select viewer nav — show toast once. */
let _viewerSelectionBoundaryToastShown = false;

function normalizeTagForViewerCompare(tag) {
    return String(tag || '')
        .replace(/_/g, ' ')
        .replace(/-/g, ' ')
        .trim()
        .toLowerCase();
}

/** Mirrors ``build_wd_model_marker`` (``backend/hydrus/tag_merge.py``) using saved config. */
function buildWdModelMarkerFromConfig(cfg) {
    const model = String(cfg?.default_model || '').trim();
    if (!model) return '';
    const tpl = String(cfg?.wd_model_marker_template || '').trim();
    if (!tpl) return `wd14:${model}`;
    if (tpl.includes('{model_name}')) return tpl.split('{model_name}').join(model);
    return tpl;
}

async function ensureViewerWdModelMarker() {
    if (_viewerWdMarkerFetched) return;
    try {
        const res = await api.getConfig();
        if (res.success && res.config) {
            const raw = buildWdModelMarkerFromConfig(res.config);
            if (raw) _viewerWdMarkerNorm = normalizeTagForViewerCompare(raw);
        }
    } finally {
        _viewerWdMarkerFetched = true;
    }
}

function tagMatchesViewerWdMarker(tag) {
    if (!_viewerWdMarkerNorm) return false;
    return normalizeTagForViewerCompare(tag) === _viewerWdMarkerNorm;
}

function hasPendingDraftChanges() {
    return _pendingAddTags.size > 0 || _pendingRemoveTags.size > 0;
}

function refreshApplyButtonState() {
    const ba = $('#btn-viewer-apply');
    if (!ba || _applyBusy) return;
    const pending = hasPendingDraftChanges();
    ba.disabled = !pending;
    ba.classList.toggle('is-dimmed', !pending);
    ba.title = pending
        ? 'Apply pending tag changes to Hydrus'
        : 'No pending tag changes to apply';
}

function syncPendingSetsFromDraft() {
    const nextDraft = new Set(_draftTags);
    _pendingAddTags = new Set();
    _pendingRemoveTags = new Set();
    for (const t of nextDraft) {
        if (!_baseServiceTags.has(t)) _pendingAddTags.add(t);
    }
    for (const t of _baseServiceTags) {
        if (!nextDraft.has(t)) _pendingRemoveTags.add(t);
    }
}

/** Slider / +− “scale” is 100%–500% of fit (1.0–5.0×). Ctrl+scroll can go beyond via `_viewerZoom`. */
const ZOOM_SLIDER_MIN_PCT = 100;
const ZOOM_SLIDER_MAX_PCT = 500;
const ZOOM_WHEEL_MIN = 0.05;
const ZOOM_WHEEL_MAX = 20;

/** User zoom multiplier on top of “fit to viewport” (1 = fit). */
let _viewerZoom = 1;
let _viewerLayoutRaf = 0;
let _viewerLayoutObserver = null;

function viewerZoomPort() {
    return $('#image-viewer-zoomport');
}

function viewerZoomInner() {
    return $('#image-viewer-zoom-inner');
}

function updateTheaterButtonUi(on) {
    const btn = $('#btn-viewer-theater');
    if (!btn) return;
    btn.setAttribute('aria-pressed', on ? 'true' : 'false');
    btn.classList.toggle('is-active', on);
    const lab = btn.querySelector('.btn-viewer-theater-label');
    if (lab) lab.textContent = on ? 'Theater on' : 'Theater off';
    btn.setAttribute(
        'aria-label',
        on ? 'Theater mode on. Click to return to default layout.' : 'Theater mode off. Click for wide stage and dim edges.',
    );
    btn.title = on
        ? 'Theater on — image maximized, tags on the right. Click or C to exit.'
        : 'Theater off — click or C for wide stage and dim edges.';
}

function syncViewerZoomSlider() {
    const range = $('#image-viewer-zoom-range');
    if (!range) return;
    const actualPct = Math.round(_viewerZoom * 100);
    const peg = Math.min(ZOOM_SLIDER_MAX_PCT, Math.max(ZOOM_SLIDER_MIN_PCT, actualPct));
    range.value = String(peg);
    range.setAttribute('aria-valuenow', String(peg));
    range.setAttribute('aria-valuetext', `${actualPct}%`);
    range.dataset.overflow = actualPct > ZOOM_SLIDER_MAX_PCT || actualPct < ZOOM_SLIDER_MIN_PCT ? '1' : '0';
}

function updateViewerZoomControls() {
    const lab = $('#image-viewer-zoom-label');
    if (lab) lab.textContent = `${Math.round(_viewerZoom * 100)}%`;
    syncViewerZoomSlider();
}

function setZoomFromSlider(pctRaw) {
    const pct = Math.min(ZOOM_SLIDER_MAX_PCT, Math.max(ZOOM_SLIDER_MIN_PCT, Number(pctRaw) || 100));
    _viewerZoom = pct / 100;
    updateViewerZoomControls();
    scheduleViewerImageLayout();
}

function scheduleViewerImageLayout() {
    if (_viewerLayoutRaf) cancelAnimationFrame(_viewerLayoutRaf);
    _viewerLayoutRaf = requestAnimationFrame(() => {
        _viewerLayoutRaf = 0;
        applyViewerImageLayout();
    });
}

/**
 * Size the zoom inner box from decoded pixels × fit × user zoom so overflow scroll/pan works
 * (same pattern as progressive decode: paint only after dimensions are known).
 */
function applyViewerImageLayout() {
    const port = viewerZoomPort();
    const inner = viewerZoomInner();
    const thumbEl = $('#image-viewer-thumb');
    const fullEl = $('#image-viewer-full');
    if (!port || !inner) return;

    const csp = window.getComputedStyle(port);
    const padX = (parseFloat(csp.paddingLeft) || 0) + (parseFloat(csp.paddingRight) || 0);
    const padY = (parseFloat(csp.paddingTop) || 0) + (parseFloat(csp.paddingBottom) || 0);
    const W = Math.max(0, port.clientWidth - padX);
    const H = Math.max(0, port.clientHeight - padY);
    if (W < 8 || H < 8) return;

    let nw = fullEl?.naturalWidth || 0;
    let nh = fullEl?.naturalHeight || 0;
    if (nw < 2 || nh < 2) {
        nw = thumbEl?.naturalWidth || 0;
        nh = thumbEl?.naturalHeight || 0;
    }
    if ((nw < 2 || nh < 2) && _viewerDisplayedFileId != null) {
        const m = getState().metadata[_viewerDisplayedFileId];
        if (m && m.width > 0 && m.height > 0) {
            nw = m.width;
            nh = m.height;
        }
    }
    if (nw < 2 || nh < 2) {
        nw = W;
        nh = H;
    }

    const fit = Math.min(W / nw, H / nh);
    const z = Math.max(ZOOM_WHEEL_MIN, Math.min(ZOOM_WHEEL_MAX, _viewerZoom));
    const dw = nw * fit * z;
    const dh = nh * fit * z;
    inner.style.width = `${dw}px`;
    inner.style.height = `${dh}px`;
}

function resetViewerZoom() {
    _viewerZoom = 1;
    updateViewerZoomControls();
    scheduleViewerImageLayout();
}

function adjustViewerZoom(delta) {
    const next = Math.max(ZOOM_WHEEL_MIN, Math.min(ZOOM_WHEEL_MAX, _viewerZoom + delta));
    if (next === _viewerZoom) return;
    _viewerZoom = next;
    updateViewerZoomControls();
    scheduleViewerImageLayout();
}

/** Multiplicative zoom for wheel (any amount within wide min/max); does not move the viewer window. */
function adjustViewerZoomWheel(deltaY) {
    const z = _viewerZoom;
    const factor = Math.exp(-deltaY * 0.0018);
    let next = z * factor;
    next = Math.max(ZOOM_WHEEL_MIN, Math.min(ZOOM_WHEEL_MAX, next));
    if (Math.abs(next - z) < 1e-9) return;
    _viewerZoom = next;
    updateViewerZoomControls();
    scheduleViewerImageLayout();
}

function ensureViewerLayoutObserver() {
    const port = viewerZoomPort();
    if (!port || _viewerLayoutObserver || typeof ResizeObserver === 'undefined') return;
    _viewerLayoutObserver = new ResizeObserver(() => {
        if (isViewerVisible()) scheduleViewerImageLayout();
    });
    _viewerLayoutObserver.observe(port);
}

function applyCinemaFromStorage() {
    const overlay = viewerOverlay();
    if (!overlay) return;
    let on = false;
    try {
        on = localStorage.getItem(CINEMA_LS_KEY) === '1';
    } catch {
        on = false;
    }
    overlay.classList.toggle('image-viewer-overlay--cinema', on);
    updateTheaterButtonUi(on);
}

function toggleCinemaMode() {
    const overlay = viewerOverlay();
    if (!overlay) return;
    const on = !overlay.classList.contains('image-viewer-overlay--cinema');
    overlay.classList.toggle('image-viewer-overlay--cinema', on);
    try {
        localStorage.setItem(CINEMA_LS_KEY, on ? '1' : '0');
    } catch {
        /* private mode */
    }
    updateTheaterButtonUi(on);
    scheduleViewerImageLayout();
}

function viewerOverlay() {
    return $('#image-viewer-overlay');
}

function isViewerVisible() {
    const o = viewerOverlay();
    return o && o.style.display !== 'none' && o.classList.contains('image-viewer-overlay--visible');
}

function closeViewer() {
    const overlay = viewerOverlay();
    if (!overlay) return;
    hideGallerySelectionModeToast();
    _viewerSelectionBoundaryToastShown = false;
    _loadGeneration += 1;
    overlay.classList.remove('image-viewer-overlay--visible');
    let finished = false;
    const done = () => {
        if (finished) return;
        finished = true;
        overlay.style.display = 'none';
        overlay.removeEventListener('transitionend', done);
    };
    overlay.addEventListener('transitionend', done);
    window.setTimeout(done, 320);
    document.body.classList.remove('image-viewer-open');
    _viewerGlobalIndex = -1;
    _viewerDisplayedFileId = null;
    _draftTags = [];
    _pendingAddTags = new Set();
    _pendingRemoveTags = new Set();
    _baseServiceTags = new Set();
    _viewerZoom = 1;
    updateViewerZoomControls();
    _viewerWdMarkerFetched = false;
    _viewerWdMarkerNorm = '';
    const inner = viewerZoomInner();
    if (inner) {
        inner.style.width = '';
        inner.style.height = '';
    }
}

function readPrefixConfig() {
    return {
        general: $('#input-general-prefix')?.value || '',
        character: $('#input-character-prefix')?.value || 'character:',
        rating: $('#input-rating-prefix')?.value || 'rating:',
    };
}

function tagsFromPredictRow(row) {
    if (Array.isArray(row.tags) && row.tags.length > 0) {
        return [...row.tags];
    }
    const tags = [];
    const pfx = readPrefixConfig();
    for (const [name] of Object.entries(row.general_tags || {})) {
        const tag = name.replace(/_/g, ' ');
        tags.push(pfx.general ? `${pfx.general}${tag}` : tag);
    }
    for (const [name] of Object.entries(row.character_tags || {})) {
        tags.push(`${pfx.character}${name.replace(/_/g, ' ')}`);
    }
    const ratingEntries = Object.entries(row.rating_tags || {});
    if (ratingEntries.length > 0) {
        const topRating = ratingEntries.reduce((a, b) => (a[1] > b[1] ? a : b))[0];
        tags.push(`${pfx.rating}${topRating}`);
    }
    return tags;
}

function setViewerStatus(msg) {
    const s = $('#image-viewer-status');
    if (s) s.textContent = msg || '';
}

function renderReadonlyServices(meta, selectedServiceKey) {
    const root = $('#image-viewer-readonly-services');
    if (!root) return;
    root.innerHTML = '';
    const state = getState();
    const services = state.services || [];
    if (!meta) return;
    const bySvc = extractTagsByService(meta, services);
    const keys = Object.keys(bySvc).filter((k) => k !== selectedServiceKey);
    if (keys.length === 0) return;
    root.appendChild(el('h3', { className: 'image-viewer-subhead', textContent: 'Other services' }));
    for (const sk of keys) {
        const svcData = bySvc[sk];
        root.appendChild(el('div', { className: 'image-viewer-service', textContent: svcData.name }));
        const row = el('div', { className: 'image-viewer-tag-row image-viewer-tag-row--readonly' });
        for (const t of svcData.tags.slice(0, 200)) {
            const cls = tagMatchesViewerWdMarker(t)
                ? 'image-viewer-tag image-viewer-tag--wd-marker'
                : 'image-viewer-tag';
            row.appendChild(el('span', { className: cls, textContent: t }));
        }
        root.appendChild(row);
    }
}

function renderChips() {
    const list = $('#image-viewer-chip-list');
    if (!list) return;
    list.innerHTML = '';
    for (const tag of _draftTags) {
        const pendingAdd = _pendingAddTags.has(tag);
        const chipClass = [
            'image-viewer-chip',
            tagMatchesViewerWdMarker(tag) ? 'image-viewer-chip--wd-marker' : '',
            pendingAdd ? 'image-viewer-chip--pending' : '',
        ].filter(Boolean).join(' ');
        const chip = el('span', { className: chipClass }, [
            el('span', { className: 'image-viewer-chip-text', textContent: tag }),
            el('button', {
                type: 'button',
                className: 'image-viewer-chip-remove',
                textContent: '\u00d7',
                'aria-label': `Remove ${tag}`,
                onClick: () => {
                    _draftTags = _draftTags.filter((t) => t !== tag);
                    syncPendingSetsFromDraft();
                    renderChips();
                    refreshApplyButtonState();
                },
            }),
        ]);
        list.appendChild(chip);
    }
    for (const tag of _pendingRemoveTags) {
        const chipClass = [
            'image-viewer-chip',
            tagMatchesViewerWdMarker(tag) ? 'image-viewer-chip--wd-marker' : '',
            'image-viewer-chip--remove-pending',
        ].join(' ');
        const chip = el('span', { className: chipClass }, [
            el('span', { className: 'image-viewer-chip-text', textContent: tag }),
            el('button', {
                type: 'button',
                className: 'image-viewer-chip-remove',
                textContent: '\u21ba',
                'aria-label': `Restore ${tag}`,
                onClick: () => {
                    if (!_draftTags.includes(tag)) _draftTags.push(tag);
                    syncPendingSetsFromDraft();
                    renderChips();
                    refreshApplyButtonState();
                },
            }),
        ]);
        list.appendChild(chip);
    }
}

function addDraftTag(raw) {
    const t = String(raw || '').trim();
    if (!t || _draftTags.includes(t)) return;
    _draftTags.push(t);
    syncPendingSetsFromDraft();
    renderChips();
    refreshApplyButtonState();
}

function resetDraftFromMeta(meta, serviceKey) {
    _draftTags = serviceKey ? getTagsForService(meta, serviceKey) : [];
    _baseServiceTags = new Set(_draftTags);
    _pendingAddTags = new Set();
    _pendingRemoveTags = new Set();
    renderChips();
    refreshApplyButtonState();
}

function setupPhasedImage(fileId, gen) {
    const stack = $('#image-viewer-stack');
    const thumbEl = $('#image-viewer-thumb');
    const fullEl = $('#image-viewer-full');
    if (!stack || !thumbEl || !fullEl) return;

    stack.classList.add('loading-full', 'image-viewer-stack--preview');
    fullEl.classList.remove('is-visible');
    fullEl.removeAttribute('src');

    /* Preview first: eager thumb + optional decode() so the LQIP frame paints cleanly (MDN: HTMLImageElement.decode). */
    thumbEl.removeAttribute('src');
    thumbEl.loading = 'eager';
    thumbEl.decoding = 'async';
    if ('fetchPriority' in thumbEl) thumbEl.fetchPriority = 'high';
    thumbEl.src = api.thumbnailUrl(fileId);
    thumbEl.alt = '';
    const runThumbDecode = () => {
        if (gen !== _loadGeneration) return;
        void thumbEl.decode?.().catch(() => {});
        scheduleViewerImageLayout();
    };
    if (thumbEl.complete) runThumbDecode();
    else thumbEl.addEventListener('load', runThumbDecode, { once: true });

    const fullUrl = api.fileUrl(fileId);
    const buf = new Image();
    buf.decoding = 'async';
    if ('fetchPriority' in buf) buf.fetchPriority = 'high';
    buf.onload = async () => {
        if (gen !== _loadGeneration) return;
        try {
            if (buf.decode) await buf.decode();
        } catch {
            /* ignore decode failures; still attempt display */
        }
        fullEl.src = fullUrl;
        const done = () => {
            if (gen !== _loadGeneration) return;
            stack.classList.remove('loading-full');
            fullEl.classList.add('is-visible');
            stack.classList.remove('image-viewer-stack--preview');
            scheduleViewerImageLayout();
        };
        if (fullEl.decode) {
            fullEl.decode().then(done).catch(done);
        } else {
            fullEl.onload = done;
        }
    };
    buf.onerror = () => {
        if (gen !== _loadGeneration) return;
        stack.classList.remove('loading-full');
        fullEl.src = fullUrl;
        fullEl.classList.add('is-visible');
        scheduleViewerImageLayout();
    };
    buf.src = fullUrl;
}

async function ensureMetadata(fileId) {
    let meta = getState().metadata[fileId];
    if (meta) return meta;
    const result = await api.getMetadata([fileId]);
    if (!result.success || !Array.isArray(result.metadata) || result.metadata.length === 0) {
        return null;
    }
    const row = result.metadata[0];
    const m = row.file_id != null ? row : { ...row, file_id: fileId };
    const next = { ...getState().metadata, [fileId]: m };
    setState({ metadata: next });
    return m;
}

const VIEWER_NAV_HINT_TAIL = ' — Arrows · C theater · slider 100–500% · Ctrl+wheel any zoom';

/**
 * Nav hint, prev/next enabled state, and scope toggle visibility (selection vs full gallery).
 * Safe to call when metadata is still loading; uses current `fileId` only.
 */
function updateViewerNavigationChrome(fileId) {
    const hint = $('#image-viewer-nav-hint');
    const prevB = $('#btn-viewer-prev');
    const nextB = $('#btn-viewer-next');
    const scopeBtn = $('#btn-viewer-nav-scope');
    const st = getState();
    const orderSel = orderedSelectedFileIds(st);
    const multiSel = orderSel.length > 1;

    if (scopeBtn) {
        if (multiSel) {
            scopeBtn.style.display = 'inline-flex';
            const on = st.galleryViewerCycleSelection;
            scopeBtn.setAttribute('aria-pressed', on ? 'true' : 'false');
            scopeBtn.classList.toggle('is-active', on);
            scopeBtn.title = on
                ? 'Selection navigation on: arrows loop only selected images. Click for full gallery search order.'
                : 'Gallery navigation: arrows follow the full search. Click to loop only selected images.';
            scopeBtn.setAttribute(
                'aria-label',
                on
                    ? 'Selection navigation on. Toggle to use full gallery search order.'
                    : 'Gallery navigation on. Toggle to loop only selected images.',
            );
        } else {
            scopeBtn.style.display = 'none';
        }
    }

    if (hint) {
        const selNav = viewerUsesSelectionOnlyNav(st, fileId);
        if (selNav) {
            const order = orderedSelectedFileIds(st);
            const si = order.indexOf(fileId);
            const i = si >= 0 ? si + 1 : 1;
            hint.textContent = `${i} / ${order.length} selected${VIEWER_NAV_HINT_TAIL}`;
        } else {
            const pos = st.fileIds.indexOf(fileId);
            if (pos >= 0 && st.fileIds.length > 0) {
                let line = `${pos + 1} / ${st.fileIds.length} in search${VIEWER_NAV_HINT_TAIL}`;
                if (multiSel && st.selectedIds?.has?.(fileId)) {
                    line =
                        `${pos + 1} / ${st.fileIds.length} in search · ${orderSel.length} selected — ` +
                        'use the cycle button (before Prev) to loop only the selection';
                }
                hint.textContent = line;
            } else {
                hint.textContent = 'Not in current search — Prev/Next disabled';
            }
        }
    }

    if (prevB && nextB) {
        const selNav2 = viewerUsesSelectionOnlyNav(st, fileId);
        if (selNav2) {
            const ord = orderedSelectedFileIds(st);
            const dis = ord.length <= 1;
            prevB.disabled = dis;
            nextB.disabled = dis;
        } else {
            const pos2 = st.fileIds.indexOf(fileId);
            prevB.disabled = pos2 <= 0 || pos2 < 0;
            nextB.disabled = pos2 < 0 || pos2 >= st.fileIds.length - 1;
        }
    }
}

async function loadViewerFile(fileId) {
    const gen = ++_loadGeneration;
    _viewerDisplayedFileId = fileId;
    await ensureViewerWdModelMarker();
    if (gen !== _loadGeneration) return;
    _viewerZoom = 1;
    updateViewerZoomControls();
    const title = $('#image-viewer-title');
    const serviceKey = $('#select-service')?.value || '';

    const meta = await ensureMetadata(fileId);
    if (gen !== _loadGeneration) return;

    if (meta && meta.width != null && meta.height != null) {
        title.textContent = `#${fileId} · ${meta.width}×${meta.height}`;
    } else {
        title.textContent = `File #${fileId}`;
    }

    setupPhasedImage(fileId, gen);

    renderReadonlyServices(meta, serviceKey);
    resetDraftFromMeta(meta, serviceKey);

    updateViewerNavigationChrome(fileId);

    setViewerStatus('');
    refreshApplyButtonState();
}

function maybeShowSelectionModeBoundaryToast(oldIdx, newIdx, len, delta) {
    if (_viewerSelectionBoundaryToastShown || len <= 1) return;
    const reachedLastForward = delta > 0 && newIdx === len - 1;
    const wrappedToFirst = delta > 0 && oldIdx === len - 1 && newIdx === 0;
    if (reachedLastForward || wrappedToFirst) {
        _viewerSelectionBoundaryToastShown = true;
        showGallerySelectionModeToast();
    }
}

export async function openImageViewer(fileId) {
    const overlay = viewerOverlay();
    if (!overlay) return;

    hideGallerySelectionModeToast();
    _viewerSelectionBoundaryToastShown = false;

    const state = getState();
    _viewerGlobalIndex = state.fileIds.indexOf(fileId);

    _viewerWdMarkerFetched = false;
    _viewerWdMarkerNorm = '';

    overlay.style.display = 'flex';
    document.body.classList.add('image-viewer-open');
    applyCinemaFromStorage();
    await loadViewerFile(fileId);
    requestAnimationFrame(() => {
        overlay.classList.add('image-viewer-overlay--visible');
        scheduleViewerImageLayout();
    });
}

async function navigateViewer(delta) {
    const state = getState();
    if (state.fileIds.length === 0) return;

    const selNav = viewerUsesSelectionOnlyNav(state, _viewerDisplayedFileId);
    if (selNav) {
        const order = orderedSelectedFileIds(state);
        if (order.length <= 1) return;
        let idx = order.indexOf(_viewerDisplayedFileId);
        if (idx < 0) idx = order.indexOf(state.fileIds[_viewerGlobalIndex]);
        if (idx < 0) idx = 0;
        const len = order.length;
        const oldIdx = idx;
        const newIdx = ((idx + delta) % len + len) % len;
        maybeShowSelectionModeBoundaryToast(oldIdx, newIdx, len, delta);
        const fid = order[newIdx];
        _viewerGlobalIndex = state.fileIds.indexOf(fid);
        setViewerStatus('');
        await loadViewerFile(fid);
        return;
    }

    if (_viewerGlobalIndex < 0) return;
    const next = Math.max(0, Math.min(state.fileIds.length - 1, _viewerGlobalIndex + delta));
    if (next === _viewerGlobalIndex) return;
    _viewerGlobalIndex = next;
    const fid = state.fileIds[next];
    setViewerStatus('');
    await loadViewerFile(fid);
}

async function onPredict() {
    const fileId = _viewerDisplayedFileId;
    if (fileId == null || _predictBusy) return;
    const model = $('#select-model')?.value;
    if (!model) {
        setViewerStatus('Select a model in the tagger panel first.');
        return;
    }
    const g = parseFloat($('#slider-general')?.value || '0.35');
    const c = parseFloat($('#slider-character')?.value || '0.85');
    _predictBusy = true;
    const bp = $('#btn-viewer-predict');
    if (bp) {
        bp.disabled = true;
        bp.textContent = 'Predicting…';
    }
    try {
        const result = await api.predict([fileId], g, c, 1);
        if (!result.success) {
            setViewerStatus(result.error || 'Predict failed.');
            return;
        }
        const rows = result.results || [];
        const row = rows.find((r) => r.file_id === fileId) || rows[0];
        if (!row) {
            setViewerStatus('No prediction row returned.');
            return;
        }
        const incoming = tagsFromPredictRow(row);
        const uniqueIncoming = [];
        const set = new Set(_draftTags);
        for (const t of incoming) {
            if (!set.has(t)) {
                set.add(t);
                uniqueIncoming.push(t);
            }
        }
        if (uniqueIncoming.length === 0) {
            setViewerStatus('No new suggestions to add.');
            return;
        }
        const ok = window.confirm(
            `Add ${uniqueIncoming.length} predicted tag(s) to current draft tags?`,
        );
        if (!ok) {
            setViewerStatus('Prediction kept unchanged (not merged).');
            return;
        }
        _draftTags.push(...uniqueIncoming);
        syncPendingSetsFromDraft();
        renderChips();
        refreshApplyButtonState();
        setViewerStatus(`Added ${uniqueIncoming.length} predicted tag(s), pending apply.`);
    } finally {
        _predictBusy = false;
        if (bp) {
            bp.disabled = false;
            bp.textContent = 'Predict';
        }
    }
}

async function onApply() {
    const fileId = _viewerDisplayedFileId;
    const serviceKey = $('#select-service')?.value || '';
    if (fileId == null || _applyBusy) return;
    if (!serviceKey) {
        setViewerStatus('Select a tag service in the tagger panel.');
        return;
    }
    const meta = getState().metadata[fileId] || (await ensureMetadata(fileId));
    const hash = meta?.hash;
    if (!hash) {
        setViewerStatus('Missing file hash; cannot apply.');
        return;
    }
    if (!hasPendingDraftChanges()) {
        setViewerStatus('No tags to apply.');
        return;
    }
    const removeTags = Array.from(_pendingRemoveTags);
    if (removeTags.length > 0) {
        const confirmRemove = window.confirm(
            `Apply includes removing ${removeTags.length} tag(s) from Hydrus. Continue?`,
        );
        if (!confirmRemove) {
            setViewerStatus('Apply cancelled.');
            return;
        }
    }
    _applyBusy = true;
    const ba = $('#btn-viewer-apply');
    if (ba) {
        ba.disabled = true;
        ba.classList.remove('is-dimmed');
        ba.textContent = 'Applying…';
    }
    try {
        const result = await api.applyTags(
            [{
                file_id: fileId,
                hash,
                tags: Array.from(_pendingAddTags),
                remove_tags: removeTags,
            }],
            serviceKey,
        );
        if (!result.success) {
            setViewerStatus(result.error || 'Apply failed.');
            return;
        }
        setViewerStatus('Applied. Refreshing metadata…');
        const metaRes = await api.getMetadata([fileId]);
        if (metaRes.success && Array.isArray(metaRes.metadata) && metaRes.metadata.length > 0) {
            const row = metaRes.metadata[0];
            const m = row.file_id != null ? row : { ...row, file_id: fileId };
            const next = { ...getState().metadata, [fileId]: m };
            setState({ metadata: next });
            resetDraftFromMeta(m, serviceKey);
            renderReadonlyServices(m, serviceKey);
        }
        setViewerStatus('Saved to Hydrus.');
    } finally {
        _applyBusy = false;
        if (ba) {
            ba.disabled = false;
            ba.textContent = 'Apply to Hydrus';
        }
        refreshApplyButtonState();
    }
}

function onResetDraft() {
    _draftTags = [];
    syncPendingSetsFromDraft();
    renderChips();
    refreshApplyButtonState();
    if (_pendingRemoveTags.size > 0) {
        setViewerStatus(
            `Marked ${_pendingRemoveTags.size} tag(s) for removal. Apply to Hydrus to confirm delete.`,
        );
    } else {
        setViewerStatus('No tags to reset.');
    }
}

function onViewerKeydown(e) {
    if (!isViewerVisible()) return;

    const panel = $('#image-viewer-panel');
    if (panel && e.key === 'Escape') {
        const ae = document.activeElement;
        const input = $('#image-viewer-tag-input');
        if (input && ae === input) {
            input.blur();
            e.preventDefault();
            return;
        }
        closeViewer();
        e.preventDefault();
        return;
    }

    if (e.key === 'ArrowLeft') {
        const input = $('#image-viewer-tag-input');
        if (document.activeElement === input) return;
        e.preventDefault();
        void navigateViewer(-1);
        return;
    }
    if (e.key === 'ArrowRight') {
        const input = $('#image-viewer-tag-input');
        if (document.activeElement === input) return;
        e.preventDefault();
        void navigateViewer(1);
        return;
    }

    if (e.key === 'c' || e.key === 'C') {
        const input = $('#image-viewer-tag-input');
        if (document.activeElement === input) return;
        e.preventDefault();
        toggleCinemaMode();
        return;
    }

    if (e.key === '+' || e.key === '=') {
        const input = $('#image-viewer-tag-input');
        if (document.activeElement === input) return;
        e.preventDefault();
        adjustViewerZoom(0.25);
        return;
    }
    if (e.key === '-' || e.key === '_') {
        const input = $('#image-viewer-tag-input');
        if (document.activeElement === input) return;
        e.preventDefault();
        adjustViewerZoom(-0.25);
        return;
    }
    if (e.key === '0') {
        const input = $('#image-viewer-tag-input');
        if (document.activeElement === input) return;
        e.preventDefault();
        resetViewerZoom();
    }
}

export function initImageViewer() {
    const overlay = viewerOverlay();
    if (!overlay) return;

    overlay.querySelector('.image-viewer-backdrop')?.addEventListener('click', () => {
        closeViewer();
    });
    $('#btn-image-viewer-close')?.addEventListener('click', () => {
        closeViewer();
    });
    ensureViewerLayoutObserver();

    $('#btn-viewer-theater')?.addEventListener('click', () => {
        toggleCinemaMode();
    });
    $('#btn-viewer-zoom-in')?.addEventListener('click', () => {
        adjustViewerZoom(0.25);
    });
    $('#btn-viewer-zoom-out')?.addEventListener('click', () => {
        adjustViewerZoom(-0.25);
    });
    $('#btn-viewer-zoom-reset')?.addEventListener('click', () => {
        resetViewerZoom();
    });

    const zoomPort = viewerZoomPort();
    zoomPort?.addEventListener(
        'wheel',
        (e) => {
            if (!isViewerVisible()) return;
            if (!(e.ctrlKey || e.metaKey)) return;
            e.preventDefault();
            adjustViewerZoomWheel(e.deltaY);
        },
        { passive: false },
    );

    $('#image-viewer-zoom-range')?.addEventListener('input', (e) => {
        const t = e.target;
        if (t instanceof HTMLInputElement) setZoomFromSlider(parseInt(t.value, 10));
    });

    $('#image-viewer-stack')?.addEventListener('dblclick', (e) => {
        if (!isViewerVisible()) return;
        if (e.target instanceof HTMLImageElement) {
            e.preventDefault();
            resetViewerZoom();
        }
    });
    $('#btn-viewer-nav-scope')?.addEventListener('click', () => {
        const st = getState();
        const next = !st.galleryViewerCycleSelection;
        writeGalleryViewerCycleSelection(next);
        setState({ galleryViewerCycleSelection: next });
        hideGallerySelectionModeToast();
    });

    subscribe('galleryViewerCycleSelection', (on) => {
        if (on === true) {
            _viewerSelectionBoundaryToastShown = false;
        }
        if (!isViewerVisible() || _viewerDisplayedFileId == null) return;
        hideGallerySelectionModeToast();
        updateViewerNavigationChrome(_viewerDisplayedFileId);
    });

    $('#btn-viewer-prev')?.addEventListener('click', () => {
        void navigateViewer(-1);
    });
    $('#btn-viewer-next')?.addEventListener('click', () => {
        void navigateViewer(1);
    });
    $('#btn-viewer-predict')?.addEventListener('click', () => {
        void onPredict();
    });
    $('#btn-viewer-apply')?.addEventListener('click', () => {
        void onApply();
    });
    $('#btn-viewer-reset')?.addEventListener('click', () => {
        onResetDraft();
    });
    $('#btn-viewer-add-chip')?.addEventListener('click', () => {
        const input = $('#image-viewer-tag-input');
        if (input) {
            addDraftTag(input.value);
            input.value = '';
            input.focus();
        }
    });

    const input = $('#image-viewer-tag-input');
    input?.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            addDraftTag(input.value);
            input.value = '';
        } else if (e.key === 'Backspace' && input.value === '') {
            if (_draftTags.length > 0) {
                _draftTags.pop();
                renderChips();
            }
        }
    });

    document.addEventListener('keydown', onViewerKeydown, true);
}

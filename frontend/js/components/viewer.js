/**
 * Full-screen image viewer: phased image load, Hydrus tag editing, WD predict/apply.
 */

import { api } from '../api.js';
import { getState, setState } from '../state.js';
import { $, el } from '../utils/dom.js';
import { extractTagsByService, getTagsForService } from '../utils/hydrus.js';

export function resetViewerTripleClickState() {
    /* Reserved for future gesture state; search still calls this to reset UI assumptions. */
}

let _viewerGlobalIndex = -1;
/** File currently shown (predict/apply even when not in `fileIds`). */
let _viewerDisplayedFileId = null;
let _draftTags = [];
let _loadGeneration = 0;
let _predictBusy = false;
let _applyBusy = false;

const CINEMA_LS_KEY = 'wd_tagger_viewer_cinema';

/** Normalized tag key for WD marker match (same rules as server ``normalize_tag_for_compare``). */
let _viewerWdMarkerNorm = '';
let _viewerWdMarkerFetched = false;

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
        const chipClass = tagMatchesViewerWdMarker(tag)
            ? 'image-viewer-chip image-viewer-chip--wd-marker'
            : 'image-viewer-chip';
        const chip = el('span', { className: chipClass }, [
            el('span', { className: 'image-viewer-chip-text', textContent: tag }),
            el('button', {
                type: 'button',
                className: 'image-viewer-chip-remove',
                textContent: '\u00d7',
                'aria-label': `Remove ${tag}`,
                onClick: () => {
                    _draftTags = _draftTags.filter((t) => t !== tag);
                    renderChips();
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
    renderChips();
}

function resetDraftFromMeta(meta, serviceKey) {
    _draftTags = serviceKey ? getTagsForService(meta, serviceKey) : [];
    renderChips();
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

    const hint = $('#image-viewer-nav-hint');
    if (hint) {
        const st = getState();
        const pos = st.fileIds.indexOf(fileId);
        if (pos >= 0 && st.fileIds.length > 0) {
            hint.textContent = `${pos + 1} / ${st.fileIds.length} — Arrows · C theater · slider 100–500% · Ctrl+wheel any zoom`;
        } else {
            hint.textContent = 'Not in current search — Prev/Next disabled';
        }
    }

    const prevB = $('#btn-viewer-prev');
    const nextB = $('#btn-viewer-next');
    const st2 = getState();
    const pos2 = st2.fileIds.indexOf(fileId);
    if (prevB) prevB.disabled = pos2 <= 0 || pos2 < 0;
    if (nextB) nextB.disabled = pos2 < 0 || pos2 >= st2.fileIds.length - 1;

    setViewerStatus('');
}

export async function openImageViewer(fileId) {
    const overlay = viewerOverlay();
    if (!overlay) return;

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
    if (_viewerGlobalIndex < 0 || state.fileIds.length === 0) return;
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
        const set = new Set(_draftTags);
        for (const t of incoming) {
            if (!set.has(t)) {
                set.add(t);
                _draftTags.push(t);
            }
        }
        renderChips();
        setViewerStatus(`Merged ${incoming.length} suggestion(s) from the model.`);
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
    if (_draftTags.length === 0) {
        setViewerStatus('No tags to apply.');
        return;
    }
    _applyBusy = true;
    const ba = $('#btn-viewer-apply');
    if (ba) {
        ba.disabled = true;
        ba.textContent = 'Applying…';
    }
    try {
        const result = await api.applyTags(
            [{ file_id: fileId, hash, tags: [..._draftTags] }],
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
    }
}

function onResetDraft() {
    const fileId = _viewerDisplayedFileId;
    const serviceKey = $('#select-service')?.value || '';
    const meta = fileId != null ? getState().metadata[fileId] : null;
    resetDraftFromMeta(meta, serviceKey);
    setViewerStatus('Reverted to Hydrus tags for this service.');
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

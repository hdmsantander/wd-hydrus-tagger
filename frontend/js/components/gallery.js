/**
 * Image gallery component with selection and pagination.
 */

import { api } from '../api.js';
import { getState, setState, subscribe } from '../state.js';
import { $, el, show, hide } from '../utils/dom.js';
import {
    buildServiceNameMap,
    clampHydrusMetadataChunkSize,
    extractTagsByService,
} from '../utils/hydrus.js';
import {
    readGalleryViewerCycleSelection,
    writeGalleryViewerCycleSelection,
} from '../utils/selection_nav.js';
import { hideGallerySelectionModeToast } from './gallery_selection_toast.js';
import { openImageViewer, resetViewerTripleClickState } from './viewer.js';

let lastClickIndex = -1;

/** After a touch long-press opens the viewer, ignore the following synthetic click. */
let _suppressCardClickUntil = 0;

/**
 * Warm the browser cache for the current page thumbnails (idle / short timeout fallback).
 * Does not replace visible <img> loads; reduces decode jank when scrolling back to a page.
 */
function prefetchGalleryThumbnails(fileIds) {
    const ids = (fileIds || []).slice(0, 32);
    if (ids.length === 0) return;
    const run = () => {
        for (const id of ids) {
            const im = new Image();
            im.decoding = 'async';
            if ('fetchPriority' in im) im.fetchPriority = 'low';
            im.src = api.thumbnailUrl(id);
        }
    };
    if (typeof window.requestIdleCallback === 'function') {
        window.requestIdleCallback(run, { timeout: 1200 });
    } else {
        window.setTimeout(run, 200);
    }
}

function hasTouchScreen() {
    return ('ontouchstart' in window) || ((navigator.maxTouchPoints || 0) > 0);
}

/**
 * Touch: hold on a card to open the viewer (mobile). Retro hold feedback via CSS classes.
 */
function attachViewerHoldGesture(card, fileId, globalIndex) {
    if (!hasTouchScreen()) return;
    if (card.dataset.viewerHoldBound === '1') return;
    card.dataset.viewerHoldBound = '1';

    let holdTimer = null;
    let sx = 0;
    let sy = 0;

    const cancelTimer = () => {
        if (holdTimer != null) {
            window.clearTimeout(holdTimer);
            holdTimer = null;
        }
    };

    const clearHoldVisual = () => {
        cancelTimer();
        card.classList.remove('gallery-card--hold-armed', 'gallery-card--hold-retro');
    };

    card.addEventListener('touchstart', (e) => {
        if (e.touches.length !== 1) return;
        clearHoldVisual();
        const t = e.touches[0];
        sx = t.clientX;
        sy = t.clientY;
        card.classList.add('gallery-card--hold-armed');
        holdTimer = window.setTimeout(() => {
            holdTimer = null;
            card.classList.remove('gallery-card--hold-armed');
            card.classList.add('gallery-card--hold-retro');
            armViewerHint(card, 3);
            _suppressCardClickUntil = Date.now() + 500;
            void openImageViewer(fileId);
            lastClickIndex = globalIndex;
            window.setTimeout(() => {
                card.classList.remove('gallery-card--hold-retro');
            }, 340);
        }, 560);
    }, { passive: true });

    card.addEventListener('touchmove', (e) => {
        if (holdTimer == null || e.touches.length !== 1) return;
        const t = e.touches[0];
        if (Math.abs(t.clientX - sx) > 14 || Math.abs(t.clientY - sy) > 14) {
            clearHoldVisual();
        }
    }, { passive: true });

    const onTouchEnd = () => {
        cancelTimer();
        card.classList.remove('gallery-card--hold-armed');
    };
    card.addEventListener('touchend', onTouchEnd, { passive: true });
    card.addEventListener('touchcancel', onTouchEnd, { passive: true });

    card.addEventListener('contextmenu', (e) => {
        if (Date.now() < _suppressCardClickUntil) e.preventDefault();
    });
}

/** 1×1 transparent GIF — used when Hydrus thumbnail proxy fails so the grid stays stable. */
const BLANK_THUMB =
    'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';

const ONBOARDING_KEY = 'wd_tagger_onboarding_done';

function isOnboardingDone() {
    try {
        return localStorage.getItem(ONBOARDING_KEY) === '1';
    } catch {
        return false;
    }
}

function markOnboardingDone() {
    try {
        localStorage.setItem(ONBOARDING_KEY, '1');
    } catch {
        /* quota / private mode */
    }
}

function _inlineGear() {
    return '<span class="inline-gear" title="Settings" aria-hidden="true">&#9881;</span>';
}

/** Empty gallery when connected but user has not run a search this session (lastSearchResultCount is still null). */
function emptyGalleryConnectedPreSearchHtml() {
    return (
        '<div class="empty-state empty-state--compact">' +
        '<p>Connected to Hydrus.</p>' +
        '<p class="empty-state-sub">Search for images to start tagging, or review the advanced settings ' +
        _inlineGear() +
        ' beforehand.</p>' +
        '</div>'
    );
}

/**
 * Empty gallery copy: first-run welcome (disconnected), pre-search (connected, no search yet),
 * zero-result search, or post-search empty state.
 *
 * lastSearchResultCount: null until the first successful Search click this session — must not be
 * conflated with 0 (a search that returned no files). Previously only `=== 0` was handled, so null
 * fell through to “Search for more…”, which never matched the pre-search case.
 */
function emptyGalleryHtml() {
    const state = getState();
    if (state.fileIds.length > 0) return '';

    if (state.lastSearchResultCount === 0) {
        return '<div class="empty-state">No images found</div>';
    }

    if (state.lastSearchResultCount === null) {
        if (!state.connected) {
            if (!isOnboardingDone()) {
                return (
                    '<div class="empty-state empty-state--welcome">' +
                    '<p class="empty-state-lead">Tag generator for Hydrus using WD14 ONNX. Select a model and adequate thresholds for it in the tagger panel and choose the tag service to be used for the operation. Settings for tuning performance and for tagging images which are already tagged can be found in the advanced settings ' +
                    _inlineGear() +
                    '. Connect to Hydrus, then search for images to tag.</p>' +
                    '</div>'
                );
            }
            return (
                '<div class="empty-state empty-state--compact">' +
                '<p>Not connected to Hydrus. Expand <strong>Hydrus connection</strong> in the sidebar to connect.</p>' +
                '<p class="empty-state-sub">After you connect, search for images to start tagging, or review the advanced settings ' +
                _inlineGear() +
                ' beforehand.</p>' +
                '</div>'
            );
        }
        return emptyGalleryConnectedPreSearchHtml();
    }

    const connected = state.connected;
    const line1 = connected
        ? 'Connected to Hydrus.'
        : 'Not connected to Hydrus. Expand <strong>Hydrus connection</strong> in the sidebar to connect.';
    return (
        '<div class="empty-state empty-state--compact">' +
        `<p>${line1}</p>` +
        '<p class="empty-state-sub">Search for more images to keep tagging.</p>' +
        '</div>'
    );
}

function armViewerHint(cardEl, level) {
    if (!cardEl) return;
    cardEl.classList.remove(
        'gallery-card--viewer-arm-1',
        'gallery-card--viewer-arm-2',
        'gallery-card--viewer-shake',
        'gallery-card--viewer-shake-strong',
    );
    void cardEl.offsetWidth;
    if (level >= 1) cardEl.classList.add('gallery-card--viewer-arm-1');
    if (level >= 2) cardEl.classList.add('gallery-card--viewer-arm-2');
    cardEl.classList.add(level >= 3 ? 'gallery-card--viewer-shake-strong' : 'gallery-card--viewer-shake');
    window.setTimeout(() => {
        cardEl.classList.remove('gallery-card--viewer-shake', 'gallery-card--viewer-shake-strong');
        if (level < 3) {
            cardEl.classList.remove('gallery-card--viewer-arm-1', 'gallery-card--viewer-arm-2');
        }
    }, level >= 3 ? 420 : 260);
    if (level >= 3) {
        window.setTimeout(() => {
            cardEl.classList.remove('gallery-card--viewer-arm-1', 'gallery-card--viewer-arm-2');
        }, 520);
    }
}

function onThumbError(ev) {
    const img = ev.target;
    if (!(img instanceof HTMLImageElement)) return;
    if (img.dataset.thumbFallback === '1') return;
    img.dataset.thumbFallback = '1';
    img.src = BLANK_THUMB;
    img.classList.add('thumb-error');
}

function buildGalleryPageKey(state) {
    const start = state.currentPage * state.pageSize;
    const pageIds = state.fileIds.slice(start, Math.min(start + state.pageSize, state.fileIds.length));
    return `${start}|${pageIds.join('|')}|sz${state.pageSize}`;
}

function buildTagTooltipNodes(tagsByService) {
    const tooltipChildren = [];
    for (const [, svcData] of Object.entries(tagsByService)) {
        tooltipChildren.push(
            el('div', { className: 'tag-tooltip-service', textContent: svcData.name })
        );
        const tagsToShow = svcData.tags.slice(0, 20);
        tooltipChildren.push(
            el('div', { className: 'tag-tooltip-tags' },
                tagsToShow.map(t =>
                    el('span', { className: 'tag-tooltip-item', textContent: t })
                )
            )
        );
    }
    return tooltipChildren;
}

function buildGalleryCardPresentation(meta, state, selectedServiceKey, serviceNameMap) {
    const tagsByService = meta
        ? extractTagsByService(meta, state.services, serviceNameMap)
        : {};
    const hasServiceTags = selectedServiceKey && tagsByService[selectedServiceKey]?.tags.length > 0;
    const hasAnyTags = Object.keys(tagsByService).length > 0;
    const tagCountOnService = hasServiceTags ? tagsByService[selectedServiceKey].tags.length : 0;
    return { tagsByService, hasServiceTags, hasAnyTags, tagCountOnService };
}

/**
 * Update one gallery card in place (thumbnail <img> kept stable to avoid reload flicker).
 * @param {HTMLElement} card
 * @param {number} fileId
 * @param {number} globalIndex
 * @param {ReturnType<typeof getState>} state
 */
function syncGalleryCard(card, fileId, globalIndex, state, selectedServiceKey, serviceNameMap) {
    const meta = state.metadata[fileId];
    const { tagsByService, hasServiceTags, hasAnyTags, tagCountOnService } =
        buildGalleryCardPresentation(meta, state, selectedServiceKey, serviceNameMap);
    const isSelected = state.selectedIds.has(fileId);

    card.dataset.fileId = String(fileId);
    card.className = `gallery-card${isSelected ? ' selected' : ''}${hasServiceTags ? ' has-tags' : ''}`;
    /* Do not replace the click handler here: el() already registered one; a second listener would double-toggle selection. */

    const info = card.querySelector('.card-info');
    if (info) {
        info.textContent = meta ? `${meta.width || '?'}x${meta.height || '?'} ${meta.ext || ''}` : `#${fileId}`;
    }

    const badge = card.querySelector('.tagged-badge');
    if (badge) {
        badge.title = `Tagged (${tagCountOnService})`;
        badge.setAttribute(
            'aria-label',
            `Tagged, ${tagCountOnService} tag${tagCountOnService === 1 ? '' : 's'} on selected service`,
        );
        const slot = badge.querySelector('.tagged-badge-icon-slot');
        slot?.querySelector('.tagged-badge-icon')?.remove();
        const txt = badge.querySelector('.tagged-badge-text');
        if (txt) txt.textContent = 'Tagged';
        const cnt = badge.querySelector('.tagged-badge-count');
        if (cnt) cnt.textContent = `(${tagCountOnService})`;
    }

    let tip = card.querySelector('.tag-tooltip');
    if (hasAnyTags) {
        const nodes = buildTagTooltipNodes(tagsByService);
        if (!tip) {
            tip = el('div', { className: 'tag-tooltip' }, nodes);
            card.appendChild(tip);
        } else {
            tip.innerHTML = '';
            for (const n of nodes) {
                tip.appendChild(n);
            }
        }
    } else if (tip) {
        tip.remove();
    }
    attachViewerHoldGesture(card, fileId, globalIndex);
}

function appendGalleryCard(grid, fileId, globalIndex, state, selectedServiceKey, serviceNameMap) {
    const meta = state.metadata[fileId];
    const isSelected = state.selectedIds.has(fileId);
    const { tagsByService, hasServiceTags, hasAnyTags, tagCountOnService } =
        buildGalleryCardPresentation(meta, state, selectedServiceKey, serviceNameMap);

    const children = [
        el('img', {
            className: 'thumb',
            src: api.thumbnailUrl(fileId),
            loading: 'lazy',
            decoding: 'async',
            alt: '',
            onerror: onThumbError,
        }),
        el('div', { className: 'card-info' }, [
            meta ? `${meta.width || '?'}x${meta.height || '?'} ${meta.ext || ''}` : `#${fileId}`,
        ]),
        el('div', { className: 'check-mark', textContent: '\u2713' }),
        el('div', {
            className: 'tagged-badge',
            title: `Tagged (${tagCountOnService})`,
            tabindex: 0,
            'aria-label': `Tagged, ${tagCountOnService} tag${tagCountOnService === 1 ? '' : 's'} on selected service`,
        }, [
            el('span', { className: 'tagged-badge-icon-slot', 'aria-hidden': 'true' }),
            el('span', { className: 'tagged-badge-expanded', 'aria-hidden': 'true' }, [
                el('span', { className: 'tagged-badge-text', textContent: 'Tagged' }),
                el('span', { className: 'tagged-badge-count', textContent: `(${tagCountOnService})` }),
            ]),
        ]),
    ];

    if (hasAnyTags) {
        children.push(el('div', { className: 'tag-tooltip' }, buildTagTooltipNodes(tagsByService)));
    }

    const card = el('div', {
        className: `gallery-card${isSelected ? ' selected' : ''}${hasServiceTags ? ' has-tags' : ''}`,
        'data-file-id': String(fileId),
        onClick: (e) => handleCardClick(fileId, globalIndex, e, card),
    }, children);

    attachViewerHoldGesture(card, fileId, globalIndex);
    grid.appendChild(card);
}

function renderGrid() {
    const state = getState();
    const grid = $('#gallery-grid');
    const selectedServiceKey = $('#select-service')?.value || '';
    const serviceNameMap = buildServiceNameMap(state.services);

    if (state.fileIds.length === 0) {
        grid.removeAttribute('data-gallery-page-key');
        grid.innerHTML = emptyGalleryHtml();
        updateToolbar();
        updatePagination();
        return;
    }

    markOnboardingDone();

    const pageKey = buildGalleryPageKey(state);
    const start = state.currentPage * state.pageSize;
    const end = Math.min(start + state.pageSize, state.fileIds.length);
    const pageIds = state.fileIds.slice(start, end);

    const prevKey = grid.getAttribute('data-gallery-page-key');
    const cards = grid.querySelectorAll('.gallery-card');
    let patchOk =
        prevKey === pageKey &&
        cards.length === pageIds.length &&
        pageIds.length > 0;

    if (patchOk) {
        for (let i = 0; i < pageIds.length; i += 1) {
            if (cards[i].dataset.fileId !== String(pageIds[i])) {
                patchOk = false;
                break;
            }
        }
    }

    if (patchOk) {
        for (let i = 0; i < pageIds.length; i += 1) {
            syncGalleryCard(
                cards[i],
                pageIds[i],
                start + i,
                state,
                selectedServiceKey,
                serviceNameMap,
            );
        }
        updateToolbar();
        updatePagination();
        prefetchGalleryThumbnails(pageIds);
        return;
    }

    grid.setAttribute('data-gallery-page-key', pageKey);
    grid.innerHTML = '';

    for (let idx = 0; idx < pageIds.length; idx += 1) {
        appendGalleryCard(
            grid,
            pageIds[idx],
            start + idx,
            state,
            selectedServiceKey,
            serviceNameMap,
        );
    }

    updateToolbar();
    updatePagination();
    prefetchGalleryThumbnails(pageIds);
}

function handleCardClick(fileId, globalIndex, event, cardEl) {
    if (Date.now() < _suppressCardClickUntil) {
        event.preventDefault();
        event.stopPropagation();
        return;
    }

    const state = getState();
    const selected = new Set(state.selectedIds);

    if (event.ctrlKey || event.metaKey) {
        resetViewerTripleClickState();
        armViewerHint(cardEl, 3);
        window.requestAnimationFrame(() => {
            window.requestAnimationFrame(() => {
                void openImageViewer(fileId);
            });
        });
        lastClickIndex = globalIndex;
        return;
    }

    if (event.shiftKey) {
        resetViewerTripleClickState();
        if (lastClickIndex >= 0) {
            const start = Math.min(lastClickIndex, globalIndex);
            const end = Math.max(lastClickIndex, globalIndex);
            for (let i = start; i <= end; i++) {
                selected.add(state.fileIds[i]);
            }
        } else if (selected.has(fileId)) selected.delete(fileId);
        else selected.add(fileId);
    } else if (event.detail >= 2) {
        /**
         * Second click of a double-click: open viewer only if the first click left the file
         * selected (otherwise the user double-clicked to deselect — no shake, no viewer).
         */
        resetViewerTripleClickState();
        event.preventDefault();
        if (!getState().selectedIds.has(fileId)) {
            return;
        }
        armViewerHint(cardEl, 3);
        window.requestAnimationFrame(() => {
            window.requestAnimationFrame(() => {
                void openImageViewer(fileId);
            });
        });
        lastClickIndex = globalIndex;
        return;
    } else {
        if (selected.has(fileId)) selected.delete(fileId);
        else selected.add(fileId);
    }

    lastClickIndex = globalIndex;
    const st = getState();
    setState({
        selectedIds: selected,
        metadata: pruneMetadata(st.metadata, { ...st, selectedIds: selected }),
    });
    renderGrid();
    updateSelectedCount();
}

function updateCycleSelectionToolbarButton() {
    const btn = $('#btn-gallery-cycle-selection');
    if (!btn) return;
    const on = getState().galleryViewerCycleSelection;
    btn.setAttribute('aria-pressed', on ? 'true' : 'false');
    btn.classList.toggle('is-active', on);
    btn.title = on
        ? 'Selection mode on: in the viewer, Prev/Next cycle only selected images. Click for full search order.'
        : 'Selection mode off. Click to cycle only among selected images in the viewer when several are selected.';
    btn.setAttribute(
        'aria-label',
        on
            ? 'Selection mode on. Toggle to navigate the full search order in the viewer.'
            : 'Selection mode off. Toggle to cycle only selected images in the viewer.',
    );
}

function updateToolbar() {
    const state = getState();
    if (state.fileIds.length > 0) {
        show('#btn-select-all');
        show('#btn-deselect-all');
        show('#btn-gallery-cycle-selection');
        updateCycleSelectionToolbarButton();
    } else {
        hide('#btn-gallery-cycle-selection');
    }
    $('#gallery-page-info').textContent =
        `${state.fileIds.length} images · page ${state.currentPage + 1} / ${Math.ceil(state.fileIds.length / state.pageSize)}`;
}

function updatePagination() {
    const state = getState();
    const totalPages = Math.ceil(state.fileIds.length / state.pageSize);
    const container = $('#gallery-pagination');
    container.innerHTML = '';

    if (totalPages <= 1) return;

    if (state.currentPage > 0) {
        container.appendChild(el('button', {
            className: 'btn btn-sm',
            textContent: '<',
            onClick: () => { setState({ currentPage: state.currentPage - 1 }); },
        }));
    }

    const startPage = Math.max(0, state.currentPage - 3);
    const endPage = Math.min(totalPages, startPage + 7);
    for (let i = startPage; i < endPage; i++) {
        container.appendChild(el('button', {
            className: `btn btn-sm${i === state.currentPage ? ' btn-primary' : ''}`,
            textContent: String(i + 1),
            onClick: () => { setState({ currentPage: i }); },
        }));
    }

    if (state.currentPage < totalPages - 1) {
        container.appendChild(el('button', {
            className: 'btn btn-sm',
            textContent: '>',
            onClick: () => { setState({ currentPage: state.currentPage + 1 }); },
        }));
    }
}

function updateSelectedCount() {
    const state = getState();
    $('#selected-count').textContent = state.selectedIds.size;
    const lock = state.taggingLockedByOtherTab;
    $('#btn-tag-selected').disabled = lock || state.selectedIds.size === 0;
}

/** Keep metadata only for the visible page and any selected files (large searches otherwise retain every visited file forever). */
function pruneMetadata(meta, state) {
    const keep = new Set(state.selectedIds);
    const start = state.currentPage * state.pageSize;
    const end = Math.min(start + state.pageSize, state.fileIds.length);
    for (let i = start; i < end; i += 1) {
        keep.add(state.fileIds[i]);
    }
    const pruned = {};
    for (const id of keep) {
        if (meta[id]) pruned[id] = meta[id];
    }
    return pruned;
}

async function loadMetadata(fileIds) {
    const state = getState();
    const meta = { ...state.metadata };
    const chunkSize = clampHydrusMetadataChunkSize(state.hydrusMetadataChunkSize);
    for (let i = 0; i < fileIds.length; i += chunkSize) {
        const chunk = fileIds.slice(i, i + chunkSize);
        const needLoad = chunk.filter(id => !meta[id]);
        if (needLoad.length === 0) continue;

        const result = await api.getMetadata(needLoad);
        if (result.success) {
            for (const m of result.metadata) {
                meta[m.file_id] = m;
            }
        }
    }
    setState({ metadata: pruneMetadata(meta, getState()) });
}

let _galleryMetadataRaf = null;
function scheduleRenderGridFromMetadata() {
    if (_galleryMetadataRaf != null) return;
    _galleryMetadataRaf = window.requestAnimationFrame(() => {
        _galleryMetadataRaf = null;
        if (getState().fileIds.length > 0) {
            renderGrid();
        }
    });
}

export function initGallery() {
    setState({ galleryViewerCycleSelection: readGalleryViewerCycleSelection() });
    updateCycleSelectionToolbarButton();

    subscribe('taggingLockedByOtherTab', () => updateSelectedCount());
    subscribe('galleryViewerCycleSelection', () => updateCycleSelectionToolbarButton());
    subscribe('metadata', () => scheduleRenderGridFromMetadata());
    subscribe('connected', () => {
        if (getState().fileIds.length === 0) renderGrid();
    });

    $('#btn-search').addEventListener('click', async () => {
        const tagsStr = $('#input-search-tags').value.trim();
        if (!tagsStr) return;

        const tags = tagsStr.split(',').map(t => t.trim()).filter(Boolean);
        $('#btn-search').disabled = true;
        $('#btn-search').textContent = 'Searching…';

        const result = await api.searchFiles(tags);
        $('#btn-search').disabled = false;
        $('#btn-search').textContent = 'Search';

        if (result.success) {
            resetViewerTripleClickState();
            markOnboardingDone();
            const cfgRes = await api.getConfig();
            if (cfgRes.success && cfgRes.config?.hydrus_metadata_chunk_size != null) {
                const n = cfgRes.config.hydrus_metadata_chunk_size;
                if (Number.isFinite(Number(n))) {
                    setState({
                        hydrusMetadataChunkSize: clampHydrusMetadataChunkSize(n),
                    });
                }
            }
            const large = result.count > 5000;
            $('#search-info').textContent = large
                ? `Large result (${result.count.toLocaleString()} files): metadata loads in chunks from Hydrus; use pages in the gallery toolbar.`
                : '';
            setState({
                fileIds: result.file_ids,
                currentPage: 0,
                selectedIds: new Set(),
                metadata: {},
                lastSearchResultCount: result.file_ids.length,
            });
            renderGrid();
        } else {
            alert('Search failed: ' + result.error);
        }
    });

    $('#btn-select-all').addEventListener('click', () => {
        const state = getState();
        const start = state.currentPage * state.pageSize;
        const end = Math.min(start + state.pageSize, state.fileIds.length);
        const selected = new Set(state.selectedIds);
        for (let i = start; i < end; i++) {
            selected.add(state.fileIds[i]);
        }
        setState({
            selectedIds: selected,
            metadata: pruneMetadata(state.metadata, { ...state, selectedIds: selected }),
        });
        renderGrid();
        updateSelectedCount();
    });

    $('#btn-deselect-all').addEventListener('click', () => {
        const st = getState();
        const empty = new Set();
        setState({
            selectedIds: empty,
            metadata: pruneMetadata(st.metadata, { ...st, selectedIds: empty }),
        });
        renderGrid();
        updateSelectedCount();
    });

    $('#btn-gallery-cycle-selection')?.addEventListener('click', () => {
        const st = getState();
        const next = !st.galleryViewerCycleSelection;
        writeGalleryViewerCycleSelection(next);
        setState({ galleryViewerCycleSelection: next });
    });

    $('#btn-gallery-selection-toast-dismiss')?.addEventListener('click', () => {
        hideGallerySelectionModeToast();
    });
    $('#btn-gallery-selection-toast-turn-off')?.addEventListener('click', () => {
        writeGalleryViewerCycleSelection(false);
        setState({ galleryViewerCycleSelection: false });
        hideGallerySelectionModeToast();
    });

    subscribe('currentPage', async () => {
        const state = getState();
        const start = state.currentPage * state.pageSize;
        const pageIds = state.fileIds.slice(start, start + state.pageSize);
        renderGrid();
        await loadMetadata(pageIds);
    });

    renderGrid();
}

export { renderGrid };

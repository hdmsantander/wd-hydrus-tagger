/**
 * Image gallery component with selection and pagination.
 */

import { api } from '../api.js';
import { getState, setState, subscribe } from '../state.js';
import { $, el, show, hide } from '../utils/dom.js';

let lastClickIndex = -1;

function extractTagsByService(meta, services) {
    const result = {};
    const tagData = meta.tags || meta.service_keys_to_statuses_to_display_tags;
    if (!tagData || typeof tagData !== 'object') return result;

    const nameMap = {};
    for (const svc of services) {
        nameMap[svc.service_key] = svc.name;
    }

    for (const serviceKey of Object.keys(tagData)) {
        const statuses = tagData[serviceKey];
        if (!statuses || typeof statuses !== 'object') continue;

        const tagSource = statuses.storage_tags || statuses.display_tags || statuses;
        if (!tagSource || typeof tagSource !== 'object') continue;

        const tags = [];
        for (const status of Object.keys(tagSource)) {
            const tagList = tagSource[status];
            if (Array.isArray(tagList)) {
                tags.push(...tagList);
            }
        }
        if (tags.length > 0) {
            result[serviceKey] = {
                name: nameMap[serviceKey] || serviceKey.slice(0, 8) + '...',
                tags: [...new Set(tags)],
            };
        }
    }
    return result;
}

function renderGrid() {
    const state = getState();
    const grid = $('#gallery-grid');
    grid.innerHTML = '';

    if (state.fileIds.length === 0) {
        grid.innerHTML = '<div class="empty-state">No images found</div>';
        return;
    }

    const start = state.currentPage * state.pageSize;
    const end = Math.min(start + state.pageSize, state.fileIds.length);
    const pageIds = state.fileIds.slice(start, end);

    const selectedServiceKey = $('#select-service')?.value || '';

    pageIds.forEach((fileId, idx) => {
        const meta = state.metadata[fileId];
        const isSelected = state.selectedIds.has(fileId);
        const tagsByService = meta ? extractTagsByService(meta, state.services) : {};
        const hasServiceTags = selectedServiceKey && tagsByService[selectedServiceKey]?.tags.length > 0;
        const hasAnyTags = Object.keys(tagsByService).length > 0;

        const children = [
            el('img', {
                className: 'thumb',
                src: api.thumbnailUrl(fileId),
                loading: 'lazy',
                alt: '',
            }),
            el('div', { className: 'card-info' }, [
                meta ? `${meta.width || '?'}x${meta.height || '?'} ${meta.ext || ''}` : `#${fileId}`,
            ]),
            el('div', { className: 'check-mark', textContent: '\u2713' }),
            el('div', { className: 'tagged-badge', textContent: '\u2660' }),
        ];

        if (hasAnyTags) {
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
            children.push(el('div', { className: 'tag-tooltip' }, tooltipChildren));
        }

        const card = el('div', {
            className: `gallery-card${isSelected ? ' selected' : ''}${hasServiceTags ? ' has-tags' : ''}`,
            onClick: (e) => handleCardClick(fileId, start + idx, e),
        }, children);

        grid.appendChild(card);
    });

    updateToolbar();
    updatePagination();
}

function handleCardClick(fileId, globalIndex, event) {
    const state = getState();
    const selected = new Set(state.selectedIds);

    if (event.shiftKey && lastClickIndex >= 0) {
        const start = Math.min(lastClickIndex, globalIndex);
        const end = Math.max(lastClickIndex, globalIndex);
        for (let i = start; i <= end; i++) {
            selected.add(state.fileIds[i]);
        }
    } else if (event.ctrlKey || event.metaKey) {
        if (selected.has(fileId)) selected.delete(fileId);
        else selected.add(fileId);
    } else {
        if (selected.has(fileId)) selected.delete(fileId);
        else selected.add(fileId);
    }

    lastClickIndex = globalIndex;
    setState({ selectedIds: selected });
    renderGrid();
    updateSelectedCount();
}

function updateToolbar() {
    const state = getState();
    if (state.fileIds.length > 0) {
        show('#btn-select-all');
        show('#btn-deselect-all');
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
            onClick: () => { setState({ currentPage: state.currentPage - 1 }); renderGrid(); },
        }));
    }

    const startPage = Math.max(0, state.currentPage - 3);
    const endPage = Math.min(totalPages, startPage + 7);
    for (let i = startPage; i < endPage; i++) {
        container.appendChild(el('button', {
            className: `btn btn-sm${i === state.currentPage ? ' btn-primary' : ''}`,
            textContent: String(i + 1),
            onClick: () => { setState({ currentPage: i }); renderGrid(); },
        }));
    }

    if (state.currentPage < totalPages - 1) {
        container.appendChild(el('button', {
            className: 'btn btn-sm',
            textContent: '>',
            onClick: () => { setState({ currentPage: state.currentPage + 1 }); renderGrid(); },
        }));
    }
}

function updateSelectedCount() {
    const state = getState();
    $('#selected-count').textContent = state.selectedIds.size;
    const lock = state.taggingLockedByOtherTab;
    $('#btn-tag-selected').disabled = lock || state.selectedIds.size === 0;
}

async function loadMetadata(fileIds) {
    const state = getState();
    const meta = { ...state.metadata };
    const chunkSize = Math.max(32, Math.min(2048, Number(state.hydrusMetadataChunkSize) || 256));
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
    setState({ metadata: meta });
}

export function initGallery() {
    subscribe('taggingLockedByOtherTab', () => updateSelectedCount());

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
            setState({
                fileIds: result.file_ids,
                currentPage: 0,
                selectedIds: new Set(),
            });
            const cfgRes = await api.getConfig();
            if (cfgRes.success && cfgRes.config?.hydrus_metadata_chunk_size != null) {
                const n = Number(cfgRes.config.hydrus_metadata_chunk_size);
                if (Number.isFinite(n)) {
                    setState({
                        hydrusMetadataChunkSize: Math.max(32, Math.min(2048, Math.floor(n))),
                    });
                }
            }
            let info = `${result.count} images found`;
            if (result.count > 5000) {
                info += ' — large result: metadata loads in server-sized chunks; browse by page.';
            }
            $('#search-info').textContent = info;

            const firstPage = result.file_ids.slice(0, getState().pageSize);
            await loadMetadata(firstPage);
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
        setState({ selectedIds: selected });
        renderGrid();
        updateSelectedCount();
    });

    $('#btn-deselect-all').addEventListener('click', () => {
        setState({ selectedIds: new Set() });
        renderGrid();
        updateSelectedCount();
    });

    subscribe('currentPage', async () => {
        const state = getState();
        const start = state.currentPage * state.pageSize;
        const pageIds = state.fileIds.slice(start, start + state.pageSize);
        await loadMetadata(pageIds);
        renderGrid();
    });
}

export { renderGrid };

/**
 * Tagger controls and results display component.
 */

import { api } from '../api.js';
import { getState, setState } from '../state.js';
import { $, el, show, hide } from '../utils/dom.js';
import { showProgress, hideProgress, updateProgress } from './progress.js';

let currentResults = [];

function renderResults(results) {
    currentResults = results;
    const list = $('#results-list');
    list.innerHTML = '';

    for (const result of results) {
        const card = el('div', { className: 'result-card' }, [
            el('img', {
                className: 'result-thumb',
                src: api.thumbnailUrl(result.file_id),
                alt: '',
            }),
            el('div', { className: 'result-tags' }, [
                renderTagCategory('General', 'general', result.general_tags),
                renderTagCategory('Character', 'character', result.character_tags),
                renderTagCategory('Rating', 'rating', result.rating_tags),
            ]),
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
                            // Remove from current results
                            delete tags[name];
                        },
                    }),
                ])
            )
        ),
    ]);
}

async function runTagging(fileIds) {
    const generalThreshold = parseFloat($('#slider-general').value);
    const characterThreshold = parseFloat($('#slider-character').value);
    const modelName = $('#select-model').value;

    // Show progress
    showProgress(fileIds.length);

    try {
        // Ensure model is loaded
        updateProgress(0, fileIds.length, '載入模型中...');
        const loadResult = await api.loadModel(modelName);
        if (!loadResult.success) {
            alert('模型載入失敗: ' + loadResult.error);
            hideProgress();
            return;
        }

        // Process in chunks to show progress
        const chunkSize = 8;
        const allResults = [];

        for (let i = 0; i < fileIds.length; i += chunkSize) {
            const chunk = fileIds.slice(i, i + chunkSize);
            updateProgress(i, fileIds.length, `處理中 ${i + 1}/${fileIds.length}...`);

            const result = await api.predict(chunk, generalThreshold, characterThreshold);
            if (result.success) {
                allResults.push(...result.results);
            }
        }

        updateProgress(fileIds.length, fileIds.length, '完成!');

        setState({ tagResults: allResults });
        currentResults = allResults;

        // Switch to results view
        hide('#section-gallery');
        show('#section-results');
        renderResults(allResults);
    } catch (err) {
        alert('標記過程發生錯誤: ' + err.message);
    } finally {
        hideProgress();
    }
}

async function applyTags() {
    const serviceKey = $('#select-service').value;
    if (!serviceKey) {
        alert('請選擇 Tag Service');
        return;
    }

    // Read prefixes from settings inputs
    const generalPrefix = $('#input-general-prefix')?.value || '';
    const characterPrefix = $('#input-character-prefix')?.value || 'character:';
    const ratingPrefix = $('#input-rating-prefix')?.value || 'rating:';

    // Build apply payload from currentResults (after user edits)
    const payload = currentResults.map(r => {
        const tags = [];

        for (const [name] of Object.entries(r.general_tags)) {
            const tag = name.replace(/_/g, ' ');
            tags.push(generalPrefix ? `${generalPrefix}${tag}` : tag);
        }
        for (const [name] of Object.entries(r.character_tags)) {
            tags.push(`${characterPrefix}${name.replace(/_/g, ' ')}`);
        }
        // Top rating
        const ratingEntries = Object.entries(r.rating_tags);
        if (ratingEntries.length > 0) {
            const topRating = ratingEntries.reduce((a, b) => a[1] > b[1] ? a : b)[0];
            tags.push(`${ratingPrefix}${topRating}`);
        }

        return { hash: r.hash, tags };
    });

    showProgress(payload.length);
    updateProgress(0, payload.length, '套用標籤中...');

    const result = await api.applyTags(payload, serviceKey);

    hideProgress();

    if (result.success) {
        alert(`成功套用標籤到 ${result.applied} 張圖片!`);
    } else {
        alert('套用失敗: ' + result.error);
    }
}

export function initTagger() {
    // Threshold slider live display
    $('#slider-general').addEventListener('input', (e) => {
        $('#val-general').textContent = parseFloat(e.target.value).toFixed(2);
    });
    $('#slider-character').addEventListener('input', (e) => {
        $('#val-character').textContent = parseFloat(e.target.value).toFixed(2);
    });

    // Tag selected
    $('#btn-tag-selected').addEventListener('click', () => {
        const state = getState();
        const ids = Array.from(state.selectedIds);
        if (ids.length === 0) {
            alert('請先選擇圖片');
            return;
        }
        runTagging(ids);
    });

    // Tag all
    $('#btn-tag-all').addEventListener('click', () => {
        const state = getState();
        if (state.fileIds.length === 0) {
            alert('請先搜尋圖片');
            return;
        }
        if (state.fileIds.length > 100) {
            if (!confirm(`確定要標記全部 ${state.fileIds.length} 張圖片？這可能需要一些時間。`)) return;
        }
        runTagging(state.fileIds);
    });

    // Apply tags
    $('#btn-apply-tags').addEventListener('click', applyTags);

    // Back to gallery
    $('#btn-back-gallery').addEventListener('click', () => {
        hide('#section-results');
        show('#section-gallery');
    });
}

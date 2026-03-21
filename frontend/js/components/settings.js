/**
 * Settings modal component.
 */

import { api } from '../api.js';
import { $, el, show, hide } from '../utils/dom.js';

async function loadModels() {
    const list = $('#model-list');
    list.innerHTML = '<div class="info-text">載入中...</div>';

    const result = await api.listModels();
    list.innerHTML = '';

    if (!result.success) {
        list.innerHTML = '<div class="info-text">無法載入模型列表</div>';
        return;
    }

    for (const model of result.models) {
        const item = el('div', { className: 'model-item' }, [
            el('span', { className: 'model-name', textContent: model.name }),
            model.downloaded
                ? el('span', { className: 'model-status downloaded', textContent: '已下載' })
                : el('button', {
                    className: 'btn btn-sm btn-primary',
                    textContent: '下載',
                    onClick: async (e) => {
                        e.target.disabled = true;
                        e.target.textContent = '下載中...';
                        const dlResult = await api.downloadModel(model.name);
                        if (dlResult.success) {
                            e.target.textContent = '已下載';
                            e.target.className = 'model-status downloaded';
                        } else {
                            e.target.textContent = '失敗';
                            e.target.disabled = false;
                            alert('下載失敗: ' + dlResult.error);
                        }
                    },
                }),
        ]);
        list.appendChild(item);
    }
}

async function loadConfig() {
    const result = await api.getConfig();
    if (result.success) {
        const cfg = result.config;
        $('#input-general-prefix').value = cfg.general_tag_prefix || '';
        $('#input-character-prefix').value = cfg.character_tag_prefix || 'character:';
        $('#input-rating-prefix').value = cfg.rating_tag_prefix || 'rating:';
        $('#check-gpu').checked = cfg.use_gpu || false;

        // Thresholds
        $('#slider-general').value = cfg.general_threshold;
        $('#val-general').textContent = cfg.general_threshold.toFixed(2);
        $('#slider-character').value = cfg.character_threshold;
        $('#val-character').textContent = cfg.character_threshold.toFixed(2);
    }
}

export function initSettings() {
    $('#btn-settings').addEventListener('click', () => {
        show('#modal-settings');
        loadModels();
        loadConfig();
    });

    $('#btn-close-settings').addEventListener('click', () => {
        hide('#modal-settings');
    });

    // Close on backdrop click
    $('#modal-settings .modal-backdrop').addEventListener('click', () => {
        hide('#modal-settings');
    });

    $('#btn-save-settings').addEventListener('click', async () => {
        const updates = {
            general_tag_prefix: $('#input-general-prefix').value,
            character_tag_prefix: $('#input-character-prefix').value,
            rating_tag_prefix: $('#input-rating-prefix').value,
            use_gpu: $('#check-gpu').checked,
            general_threshold: parseFloat($('#slider-general').value),
            character_threshold: parseFloat($('#slider-character').value),
        };
        const result = await api.updateConfig(updates);
        if (result.success) {
            alert('設定已儲存');
            hide('#modal-settings');
        } else {
            alert('儲存失敗');
        }
    });
}

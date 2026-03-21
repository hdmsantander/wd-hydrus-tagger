/**
 * Connection panel component.
 */

import { api } from '../api.js';
import { getState, setState } from '../state.js';
import { $, show, hide } from '../utils/dom.js';

const STORAGE_KEY = 'wd-hydrus-connection';

function loadSaved() {
    try {
        const saved = localStorage.getItem(STORAGE_KEY);
        if (saved) return JSON.parse(saved);
    } catch {}
    return null;
}

function saveCredentials(url, key) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ url, key }));
}

async function connect(url, apiKey) {
    const statusDot = $('#connection-status');
    statusDot.className = 'status-dot connecting';
    statusDot.title = '連線中...';

    const result = await api.testConnection(url, apiKey);
    if (result.success) {
        statusDot.className = 'status-dot connected';
        statusDot.title = '已連線';
        saveCredentials(url, apiKey);

        // Load services
        const svcResult = await api.getServices();
        if (svcResult.success) {
            // Filter to local tag services (type 5) and tag repositories (type 1)
            const tagServices = svcResult.services.filter(s =>
                s.type === 5 || s.type_pretty.toLowerCase().includes('tag')
            );
            setState({ connected: true, services: tagServices });
            populateServiceSelect(tagServices);
        }

        show('#panel-search');
        show('#panel-tagger');
        return true;
    } else {
        statusDot.className = 'status-dot disconnected';
        statusDot.title = '連線失敗';
        alert('連線失敗: ' + (result.error || '未知錯誤'));
        return false;
    }
}

function populateServiceSelect(services) {
    const select = $('#select-service');
    select.innerHTML = '';
    for (const svc of services) {
        const opt = document.createElement('option');
        opt.value = svc.service_key;
        opt.textContent = svc.name;
        select.appendChild(opt);
    }
}

export function initConnection() {
    const saved = loadSaved();
    if (saved) {
        $('#input-api-url').value = saved.url;
        $('#input-api-key').value = saved.key;
    }

    $('#btn-connect').addEventListener('click', async () => {
        const url = $('#input-api-url').value.trim();
        const apiKey = $('#input-api-key').value.trim();
        if (!apiKey) {
            alert('請輸入 API Key');
            return;
        }
        await connect(url, apiKey);
    });

    // Auto-connect if saved credentials exist
    if (saved && saved.key) {
        connect(saved.url, saved.key);
    }
}

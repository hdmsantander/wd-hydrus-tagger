/**
 * Connection panel component.
 */

import { api } from '../api.js';
import { setState } from '../state.js';
import { $, show } from '../utils/dom.js';

const STORAGE_KEY = 'wd-hydrus-connection';

function loadSaved() {
    try {
        const saved = localStorage.getItem(STORAGE_KEY);
        if (saved) return JSON.parse(saved);
    } catch {
        /* ignore */
    }
    return null;
}

function saveCredentials(url, key) {
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify({ url, key }));
    } catch {
        /* quota / private mode */
    }
}

/** Visual state for the collapsible Hydrus connection block (green when healthy, red when attention needed). */
export function syncConnectionShell(mode) {
    const details = $('#connection-details');
    const hint = $('#connection-summary-hint');
    if (!details) return;
    details.classList.remove(
        'connection-shell--ok',
        'connection-shell--error',
        'connection-shell--connecting',
    );
    if (mode === 'connected') {
        details.classList.add('connection-shell--ok');
        details.open = false;
        if (hint) hint.textContent = 'Connected';
    } else if (mode === 'connecting') {
        details.classList.add('connection-shell--connecting');
        details.open = true;
        if (hint) hint.textContent = 'Connecting…';
    } else {
        details.classList.add('connection-shell--error');
        details.open = true;
        if (hint) hint.textContent = 'Not connected';
    }
}

async function connect(url, apiKey) {
    syncConnectionShell('connecting');
    const statusDot = $('#connection-status');
    statusDot.className = 'status-dot connecting';
    statusDot.title = 'Connecting…';

    const result = await api.testConnection(url, apiKey);
    if (result.success) {
        statusDot.className = 'status-dot connected';
        statusDot.title = 'Connected';
        saveCredentials(url, apiKey);
        syncConnectionShell('connected');

        const svcResult = await api.getServices();
        if (svcResult.success) {
            const tagServices = svcResult.services.filter(s =>
                s.type === 5 || s.type_pretty.toLowerCase().includes('tag')
            );
            setState({ connected: true, services: tagServices });
            populateServiceSelect(tagServices);
        }

        show('#panel-search');
        show('#panel-tagger');
        return true;
    }
    statusDot.className = 'status-dot disconnected';
    statusDot.title = 'Connection failed';
    setState({ connected: false, services: [] });
    syncConnectionShell('disconnected');
    alert('Connection failed: ' + (result.error || 'Unknown error'));
    return false;
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
    applyDefaultTagServiceFromConfig();
}

/** Prefer Hydrus service whose display name matches config ``target_tag_service`` (case-insensitive). */
function applyDefaultTagServiceFromConfig() {
    api.getConfig().then((r) => {
        if (!r.success || !r.config?.target_tag_service) return;
        const want = String(r.config.target_tag_service).trim().toLowerCase();
        if (!want) return;
        const sel = $('#select-service');
        if (!sel?.options?.length) return;
        for (const o of sel.options) {
            if (String(o.textContent || '').trim().toLowerCase() === want) {
                sel.value = o.value;
                break;
            }
        }
    });
}

export function initConnection() {
    syncConnectionShell('disconnected');

    const saved = loadSaved();
    if (saved) {
        $('#input-api-url').value = saved.url;
        $('#input-api-key').value = saved.key;
    }

    const btnKey = $('#btn-toggle-api-key');
    const inpKey = $('#input-api-key');
    if (btnKey && inpKey) {
        btnKey.addEventListener('click', () => {
            const showPlain = inpKey.type === 'password';
            inpKey.type = showPlain ? 'text' : 'password';
            btnKey.textContent = showPlain ? 'Hide' : 'Show';
            btnKey.setAttribute('aria-pressed', showPlain ? 'true' : 'false');
            btnKey.setAttribute('aria-label', showPlain ? 'Hide API key' : 'Show API key');
        });
    }

    $('#btn-connect').addEventListener('click', async () => {
        const url = $('#input-api-url').value.trim();
        const apiKey = $('#input-api-key').value.trim();
        if (!apiKey) {
            alert('Please enter an API key');
            return;
        }
        if (!url) {
            alert('Please enter the Hydrus API URL (or leave the default and only paste the key).');
            return;
        }
        await connect(url, apiKey);
    });

    if (saved && saved.key && String(saved.url || '').trim()) {
        connect(String(saved.url).trim(), saved.key);
    }
}

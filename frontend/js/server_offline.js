/**
 * Full-page "server stopped / unreachable" view and periodic connectivity checks.
 */

import { $ } from './utils/dom.js';

let overlayShown = false;
let watchTimer = null;
let fetchFailureNotifiedAt = 0;
const FETCH_FAILURE_DEBOUNCE_MS = 1800;

/** Called when any API fetch throws (network error, connection refused, etc.). */
export function notifyFetchFailed() {
    if (overlayShown) return;
    const now = Date.now();
    if (now - fetchFailureNotifiedAt < FETCH_FAILURE_DEBOUNCE_MS) return;
    fetchFailureNotifiedAt = now;
    showServerOfflineScreen({ reason: 'connection_lost' });
}

/**
 * After Settings → Stop server succeeds, or optional metrics text from shutdown response.
 */
export function showServerOfflineScreen({ reason = 'connection_lost', metricsLines = '' } = {}) {
    if (overlayShown) return;
    overlayShown = true;
    stopServerWatch();

    const shell = $('#app-shell');
    const off = $('#server-offline-screen');
    if (shell) shell.style.display = 'none';
    if (!off) return;

    const titleEl = $('#server-offline-title');
    const leadEl = $('#server-offline-lead');
    const metricsEl = $('#server-offline-metrics');

    if (reason === 'ui_shutdown') {
        if (titleEl) titleEl.textContent = 'Server stopped';
        if (leadEl) {
            leadEl.textContent =
                'The wd-hydrus-tagger process is exiting. This page cannot talk to the app until you start the server again '
                + '(e.g. python run.py or ./wd-hydrus-tagger.sh run).';
        }
    } else {
        if (titleEl) titleEl.textContent = 'Server unreachable';
        if (leadEl) {
            leadEl.textContent =
                'The browser cannot reach this app. The process may have been stopped in a terminal (Ctrl+C), '
                + 'crashed, or the network path failed. If the server is running again, reload this page.';
        }
    }

    if (metricsEl) {
        metricsEl.textContent = metricsLines ? metricsLines.trim() : '';
        metricsEl.style.display = metricsLines && metricsLines.trim() ? 'block' : 'none';
    }

    off.style.display = 'flex';

    const retry = $('#btn-server-offline-retry');
    if (retry) {
        retry.onclick = () => {
            window.location.reload();
        };
    }
}

/** When tagging receives server_shutting_down, probe soon so CLI/UI shutdown is detected without waiting for the long interval. */
export function expectServerShutdownSoon() {
    let attempts = 0;
    const maxAttempts = 12;
    const t = setInterval(async () => {
        if (overlayShown) {
            clearInterval(t);
            return;
        }
        attempts += 1;
        if (attempts > maxAttempts) {
            clearInterval(t);
            return;
        }
        try {
            const ac = new AbortController();
            const to = setTimeout(() => ac.abort(), 4000);
            const r = await fetch(`${window.location.origin}/api/app/status`, { signal: ac.signal });
            clearTimeout(to);
            if (!r.ok) throw new Error(String(r.status));
        } catch {
            clearInterval(t);
            showServerOfflineScreen({ reason: 'connection_lost' });
        }
    }, 1000);
}

export function stopServerWatch() {
    if (watchTimer != null) {
        clearInterval(watchTimer);
        watchTimer = null;
    }
}

/** Lightweight poll while the tab is visible (CLI kill, crash, remote host down). */
export function startServerWatch(intervalMs = 22000) {
    stopServerWatch();
    watchTimer = setInterval(async () => {
        if (overlayShown) return;
        if (typeof document !== 'undefined' && document.visibilityState !== 'visible') return;
        try {
            const ac = new AbortController();
            const to = setTimeout(() => ac.abort(), 4500);
            const r = await fetch(`${window.location.origin}/api/app/status`, { signal: ac.signal });
            clearTimeout(to);
            if (!r.ok) throw new Error(String(r.status));
        } catch {
            showServerOfflineScreen({ reason: 'connection_lost' });
        }
    }, intervalMs);
}

export function isOfflineScreenShown() {
    return overlayShown;
}

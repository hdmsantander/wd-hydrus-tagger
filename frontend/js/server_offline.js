/**
 * Full-page server status view (offline / online) and connectivity checks.
 */

import { $ } from './utils/dom.js';

let overlayShown = false;
let overlayReason = 'connection_lost';
let watchTimer = null;
let statusPollTimer = null;
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

function clearStatusPoll() {
    if (statusPollTimer != null) {
        clearInterval(statusPollTimer);
        statusPollTimer = null;
    }
}

function setStatusBadge(online) {
    const badge = $('#server-offline-status-badge');
    if (!badge) return;
    badge.textContent = online ? 'Online' : 'Offline';
    badge.classList.toggle('server-status-badge--online', online);
    badge.classList.toggle('server-status-badge--offline', !online);
}

function applyReasonCopy(reason) {
    const titleEl = $('#server-offline-title');
    const leadEl = $('#server-offline-lead');

    if (reason === 'ui_shutdown') {
        if (titleEl) titleEl.textContent = 'Server status';
        if (leadEl) {
            leadEl.textContent =
                'The wd-hydrus-tagger process has stopped or is exiting. Reload after you start the server again '
                + '(for example: python run.py or ./wd-hydrus-tagger.sh run).';
        }
    } else if (reason === 'shutting_down') {
        if (titleEl) titleEl.textContent = 'Server status';
        if (leadEl) {
            leadEl.textContent =
                'The server is shutting down. This page will show Online when the app responds again — then reload to continue.';
        }
    } else {
        if (titleEl) titleEl.textContent = 'Server status';
        if (leadEl) {
            leadEl.textContent =
                'The browser cannot reach this app. The process may have been stopped, crashed, or the network path failed. '
                + 'When the server is running again, this status will show Online — or use Check server.';
        }
    }
}

async function probeServerOnline() {
    const ac = new AbortController();
    const to = setTimeout(() => ac.abort(), 4000);
    try {
        const r = await fetch(`${window.location.origin}/api/app/status`, { signal: ac.signal });
        clearTimeout(to);
        return r.ok;
    } catch {
        clearTimeout(to);
        return false;
    }
}

function startStatusPoll() {
    clearStatusPoll();
    const tick = async () => {
        if (!overlayShown) return;
        const ok = await probeServerOnline();
        setStatusBadge(ok);
        const leadEl = $('#server-offline-lead');
        if (leadEl && ok) {
            leadEl.textContent =
                'The server is online. Reload the page to continue using the app.';
        } else if (leadEl && !ok) {
            applyReasonCopy(overlayReason);
        }
    };
    void tick();
    statusPollTimer = setInterval(tick, 2500);
}

/**
 * After Settings → Stop server, lost connection, or optional legacy metrics (ignored).
 */
export function showServerOfflineScreen({ reason = 'connection_lost', metricsLines: _metricsLines = '' } = {}) {
    if (overlayShown) return;
    overlayShown = true;
    overlayReason = reason;
    stopServerWatch();
    clearStatusPoll();

    const shell = $('#app-shell');
    const off = $('#server-offline-screen');
    if (shell) shell.style.display = 'none';
    if (!off) return;

    applyReasonCopy(overlayReason);
    setStatusBadge(false);

    off.style.display = 'flex';

    const retry = $('#btn-server-offline-retry');
    if (retry) {
        retry.onclick = () => {
            void checkServerThenReloadOrShake();
        };
    }

    startStatusPoll();
}

const BADGE_FLASH_ONLINE_MS = 560;

/** Probe /api/app/status; reload only if online (after green flash on badge). If offline, shake badge + red flash. */
async function checkServerThenReloadOrShake() {
    const badge = $('#server-offline-status-badge');
    const btn = $('#btn-server-offline-retry');
    if (btn) {
        btn.disabled = true;
        const prev = btn.textContent;
        btn.textContent = 'Checking…';
        let reloadScheduled = false;
        try {
            const ok = await probeServerOnline();
            if (ok) {
                reloadScheduled = true;
                setStatusBadge(true);
                const leadEl = $('#server-offline-lead');
                if (leadEl) {
                    leadEl.textContent = 'Server is online — reloading…';
                }
                if (badge) {
                    badge.classList.remove('server-status-badge--flash-online');
                    void badge.offsetWidth;
                    badge.classList.add('server-status-badge--flash-online');
                }
                window.setTimeout(() => {
                    window.location.reload();
                }, BADGE_FLASH_ONLINE_MS);
                return;
            }
            setStatusBadge(false);
            applyReasonCopy(overlayReason);
            if (badge) {
                badge.classList.remove('server-status-badge--retry-fail');
                void badge.offsetWidth;
                badge.classList.add('server-status-badge--retry-fail');
                window.setTimeout(() => {
                    badge.classList.remove('server-status-badge--retry-fail');
                }, 650);
            }
        } finally {
            if (btn && !reloadScheduled) {
                btn.disabled = false;
                btn.textContent = prev;
            }
        }
    }
}

/** When tagging receives server_shutting_down, show status immediately; poll updates online/offline. */
export function expectServerShutdownSoon() {
    showServerOfflineScreen({ reason: 'shutting_down' });
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

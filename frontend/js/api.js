/**
 * Backend API client module.
 */

const BASE = '';

/** Set from app bootstrap to show offline UI on network failures (debounced in server_offline). */
let _onFetchNetworkError = null;

export function setFetchNetworkErrorHandler(fn) {
    _onFetchNetworkError = typeof fn === 'function' ? fn : null;
}

async function request(method, path, body = null) {
    const opts = {
        method,
        headers: { 'Content-Type': 'application/json' },
    };
    if (body) opts.body = JSON.stringify(body);
    let resp;
    try {
        resp = await fetch(`${BASE}${path}`, opts);
    } catch (err) {
        _onFetchNetworkError?.();
        throw err;
    }
    const text = await resp.text();
    if (!text) {
        return { success: false, error: `Empty response (HTTP ${resp.status})` };
    }
    try {
        return JSON.parse(text);
    } catch {
        return { success: false, error: text.length > 200 ? `${text.slice(0, 200)}…` : text };
    }
}

export const api = {
    // Connection
    testConnection: (url, apiKey) =>
        request('POST', '/api/connection/test', { url, api_key: apiKey }),

    getServices: () =>
        request('GET', '/api/connection/services'),

    // Files
    searchFiles: (tags, fileSortType = null) =>
        request('POST', '/api/files/search', { tags, file_sort_type: fileSortType }),

    getMetadata: (fileIds) =>
        request('POST', '/api/files/metadata', { file_ids: fileIds }),

    thumbnailUrl: (fileId) =>
        `${BASE}/api/files/${fileId}/thumbnail`,

    fileUrl: (fileId) =>
        `${BASE}/api/files/${fileId}`,

    // Tagger
    listModels: () =>
        request('GET', '/api/tagger/models'),

    getTaggingSessionStatus: () =>
        request('GET', '/api/tagger/session/status'),

    downloadModel: (name) =>
        request('POST', `/api/tagger/models/${name}/download`),

    /**
     * Validate ONNX+CSV on disk; optional Hub revision check (network).
     * @param {{ checkRemote?: boolean, modelName?: string }} [opts]
     */
    verifyModels(opts = {}) {
        const body = { check_remote: !!opts.checkRemote };
        if (opts.modelName) {
            body.model_name = opts.modelName;
        }
        return request('POST', '/api/tagger/models/verify', body);
    },

    loadModel: (name) =>
        request('POST', `/api/tagger/models/${name}/load`),

    predict: (fileIds, generalThreshold, characterThreshold, batchSize = null) => {
        const body = {
            file_ids: fileIds,
            general_threshold: generalThreshold,
            character_threshold: characterThreshold,
        };
        if (batchSize != null && Number.isFinite(batchSize) && batchSize > 0) {
            body.batch_size = batchSize;
        }
        return request('POST', '/api/tagger/predict', body);
    },

    applyTags: (results, serviceKey) =>
        request('POST', '/api/tagger/apply', { results, service_key: serviceKey }),

    // Config
    getConfig: () =>
        request('GET', '/api/config'),

    updateConfig: (updates) =>
        request('PATCH', '/api/config', updates),

    getAppStatus: () => request('GET', '/api/app/status'),

    shutdownApp: () => request('POST', '/api/app/shutdown'),

    /**
     * Batched tagging with optional cancel. First outbound message is the run payload.
     * @param {object} payload — action: 'run' added if missing
     * @param {object} [callbacks] — onProgress, onTagsApplied
     */
    startTaggingWebSocket(payload, callbacks = {}) {
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const ws = new WebSocket(`${proto}//${location.host}/api/tagger/ws/progress`);
        const body = { action: 'run', ...payload };

        const sendControl = (action) => {
            try {
                if (ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ action }));
                }
            } catch (_) {
                /* ignore */
            }
        };

        const cancel = () => sendControl('cancel');
        const pause = () => sendControl('pause');
        const resume = () => sendControl('resume');
        const flush = () => sendControl('flush');

        const markController = () => {
            try {
                sessionStorage.setItem('wd_tagger_controller', '1');
            } catch (_) {
                /* private mode */
            }
        };
        const clearController = () => {
            try {
                sessionStorage.removeItem('wd_tagger_controller');
            } catch (_) {
                /* ignore */
            }
        };

        const done = new Promise((resolve, reject) => {
            ws.onmessage = (ev) => {
                let msg;
                try {
                    msg = JSON.parse(ev.data);
                } catch {
                    return;
                }

                if (
                    msg.type === 'progress'
                    || msg.type === 'file'
                    || msg.type === 'tags_applied'
                    || msg.type === 'stopping'
                    || msg.type === 'server_shutting_down'
                ) {
                    markController();
                }

                if (msg.type === 'control_ack') {
                    callbacks.onControlAck?.(msg);
                }
                if (msg.type === 'progress' || msg.type === 'file') {
                    callbacks.onProgress?.(msg);
                }
                if (msg.type === 'tags_applied') {
                    callbacks.onTagsApplied?.(msg);
                }
                if (msg.type === 'stopping') {
                    callbacks.onStopping?.(msg);
                }
                if (msg.type === 'server_shutting_down') {
                    callbacks.onServerShuttingDown?.(msg);
                }

                if (msg.type === 'error') {
                    if (msg.code === 'tagging_busy') {
                        const err = new Error(
                            msg.message || 'Tagging already in progress in another tab',
                        );
                        err.code = 'tagging_busy';
                        err.snapshot = msg.snapshot;
                        reject(err);
                        ws.close();
                        return;
                    }
                    if (Array.isArray(msg.partial_results)) {
                        clearController();
                        resolve({
                            type: 'stopped',
                            stopped: true,
                            results: msg.partial_results,
                            total_processed: msg.partial_results.length,
                            total_applied: 0,
                            total_duplicates_skipped: 0,
                            pending_hydrus_files: msg.pending_hydrus_files ?? 0,
                        });
                    } else {
                        clearController();
                        reject(new Error(msg.message || 'WebSocket error'));
                    }
                    ws.close();
                    return;
                }

                if (msg.type === 'complete' || msg.type === 'stopped') {
                    clearController();
                    resolve(msg);
                    ws.close();
                }
            };

            ws.onerror = () => {
                clearController();
                reject(new Error('WebSocket connection failed'));
                try {
                    ws.close();
                } catch (_) {
                    /* ignore */
                }
            };

            ws.onopen = () => {
                /* Same-tab session: mark controller before first server message so polling does not flash the “other tab” banner. */
                markController();
                ws.send(JSON.stringify(body));
            };
        });

        return { ws, cancel, pause, resume, flush, done };
    },
};

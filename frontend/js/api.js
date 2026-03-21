/**
 * Backend API client module.
 */

const BASE = '';

async function request(method, path, body = null) {
    const opts = {
        method,
        headers: { 'Content-Type': 'application/json' },
    };
    if (body) opts.body = JSON.stringify(body);
    const resp = await fetch(`${BASE}${path}`, opts);
    return resp.json();
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

    downloadModel: (name) =>
        request('POST', `/api/tagger/models/${name}/download`),

    loadModel: (name) =>
        request('POST', `/api/tagger/models/${name}/load`),

    predict: (fileIds, generalThreshold, characterThreshold) =>
        request('POST', '/api/tagger/predict', {
            file_ids: fileIds,
            general_threshold: generalThreshold,
            character_threshold: characterThreshold,
        }),

    applyTags: (results, serviceKey) =>
        request('POST', '/api/tagger/apply', { results, service_key: serviceKey }),

    // Config
    getConfig: () =>
        request('GET', '/api/config'),

    updateConfig: (updates) =>
        request('PATCH', '/api/config', updates),

    // WebSocket for progress
    createProgressWs: () => {
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        return new WebSocket(`${proto}//${location.host}/api/tagger/ws/progress`);
    },
};

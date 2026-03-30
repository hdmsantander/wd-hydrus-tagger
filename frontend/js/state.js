/**
 * Simple reactive state store with pub/sub.
 */

const state = {
    connected: false,
    services: [],
    fileIds: [],
    metadata: {},       // fileId -> metadata object
    selectedIds: new Set(),
    currentPage: 0,
    pageSize: 50,
    tagResults: [],     // Array of tag result objects
    processing: false,
    progressCurrent: 0,
    progressTotal: 0,
    /** True when server reports active tagging and this tab is not the WebSocket controller */
    taggingLockedByOtherTab: false,
    /** Mirrors server ``hydrus_metadata_chunk_size`` for gallery metadata requests */
    hydrusMetadataChunkSize: 256,
};

const listeners = {};

export function getState() {
    return state;
}

export function setState(updates) {
    Object.assign(state, updates);
    for (const key of Object.keys(updates)) {
        if (listeners[key]) {
            listeners[key].forEach(fn => fn(state[key], state));
        }
    }
    // Also fire a generic 'change' event
    if (listeners['*']) {
        listeners['*'].forEach(fn => fn(state));
    }
}

export function subscribe(key, fn) {
    if (!listeners[key]) listeners[key] = [];
    listeners[key].push(fn);
    return () => {
        listeners[key] = listeners[key].filter(f => f !== fn);
    };
}

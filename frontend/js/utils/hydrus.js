/**
 * Hydrus-related limits shared with the backend (see backend.config clamp_hydrus_metadata_chunk_size).
 */

/** Default matches server ``AppConfig.hydrus_metadata_chunk_size`` (512). */
export const HYDRUS_METADATA_CHUNK_DEFAULT = 512;

/** Clamp gallery metadata chunk size to 32–2048 (same bounds as config.yaml). */
export function clampHydrusMetadataChunkSize(raw) {
    const n = Number(raw);
    if (!Number.isFinite(n)) return HYDRUS_METADATA_CHUNK_DEFAULT;
    return Math.max(32, Math.min(2048, Math.floor(n)));
}

export function buildServiceNameMap(services) {
    const nameMap = {};
    for (const svc of services || []) {
        nameMap[svc.service_key] = svc.name;
    }
    return nameMap;
}

function collectServiceTags(meta, serviceKey) {
    const tagData = meta?.tags || meta?.service_keys_to_statuses_to_display_tags;
    if (!tagData || typeof tagData !== 'object') return [];
    const statuses = tagData[serviceKey];
    if (!statuses || typeof statuses !== 'object') return [];
    const tagSource = statuses.storage_tags || statuses.display_tags || statuses;
    if (!tagSource || typeof tagSource !== 'object') return [];
    const tags = [];
    for (const status of Object.keys(tagSource)) {
        const tagList = tagSource[status];
        if (Array.isArray(tagList)) tags.push(...tagList);
    }
    return tags.length > 0 ? [...new Set(tags)] : [];
}

/**
 * @param {object} meta Hydrus file metadata row
 * @param {{ service_key: string, name?: string }[]} services
 * @param {Record<string, string>} [serviceNameMap]
 * @returns {Record<string, { name: string, tags: string[] }>}
 */
export function extractTagsByService(meta, services, serviceNameMap = null) {
    const result = {};
    const tagData = meta.tags || meta.service_keys_to_statuses_to_display_tags;
    if (!tagData || typeof tagData !== 'object') return result;
    const nameMap = serviceNameMap || buildServiceNameMap(services);

    for (const serviceKey of Object.keys(tagData)) {
        const tags = collectServiceTags(meta, serviceKey);
        if (tags.length > 0) {
            result[serviceKey] = {
                name: nameMap[serviceKey] || serviceKey.slice(0, 8) + '...',
                tags,
            };
        }
    }
    return result;
}

/**
 * Tags stored on a single service (Hydrus storage_tags / display_tags merge).
 * @param {object} meta
 * @param {string} serviceKey
 * @returns {string[]}
 */
export function getTagsForService(meta, serviceKey) {
    if (!meta || !serviceKey) return [];
    return collectServiceTags(meta, serviceKey);
}

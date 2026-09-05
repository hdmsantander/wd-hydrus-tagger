/**
 * Optional floogulinc/hydrus-web companion UI (base URL from server config).
 */

import { getState } from '../state.js';
import { $ } from './dom.js';

/** Escape a string for use inside an HTML double-quoted attribute. */
export function escapeAttrForHtml(s) {
    return String(s || '')
        .replace(/&/g, '&amp;')
        .replace(/"/g, '&quot;')
        .replace(/</g, '&lt;');
}

/**
 * Resolve the hydrus-web “pages” route (main library UI) from a configured base URL.
 * @param {string} baseRaw e.g. http://127.0.0.1:8080
 * @returns {string} absolute URL or '' if unset/invalid
 */
export function hydrusWebLibraryUrl(baseRaw) {
    const base = String(baseRaw || '').trim();
    if (!base) return '';
    try {
        return new URL('pages', base.endsWith('/') ? base : `${base}/`).href;
    } catch {
        return '';
    }
}

/** Update the gallery toolbar link visibility and href from global state. */
export function syncHydrusWebToolbarLink() {
    const a = $('#link-gallery-hydrus-web');
    if (!a) return;
    const lib = hydrusWebLibraryUrl(getState().hydrusWebUrl);
    if (!lib) {
        a.hidden = true;
        a.removeAttribute('href');
        return;
    }
    a.hidden = false;
    a.setAttribute('href', lib);
}

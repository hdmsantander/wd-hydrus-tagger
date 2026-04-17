/**
 * Non-blocking toast for viewer “selection only” navigation.
 */

import { $ } from '../utils/dom.js';

export function hideGallerySelectionModeToast() {
    const t = $('#gallery-selection-mode-toast');
    if (!t) return;
    t.classList.remove('gallery-toast--visible');
    window.setTimeout(() => {
        if (!t.classList.contains('gallery-toast--visible')) {
            t.style.display = 'none';
        }
    }, 220);
}

export function showGallerySelectionModeToast() {
    const t = $('#gallery-selection-mode-toast');
    if (!t) return;
    t.style.display = 'flex';
    requestAnimationFrame(() => {
        t.classList.add('gallery-toast--visible');
    });
}

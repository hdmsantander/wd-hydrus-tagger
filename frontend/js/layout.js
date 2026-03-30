/**
 * Collapsible sidebar: desktop (narrow strip) vs mobile (drawer + backdrop).
 * Preference: localStorage key wd_tagger_sidebar_hidden — "1" hidden, "0" visible.
 * Defaults: desktop starts expanded; narrow view starts with drawer closed.
 */

const STORAGE_KEY = 'wd_tagger_sidebar_hidden';
const DESKTOP_MQ = '(min-width: 769px)';

export function initLayoutSidebar() {
    const layout = document.getElementById('app-layout');
    const toggle = document.getElementById('btn-sidebar-toggle');
    const backdrop = document.getElementById('sidebar-backdrop');
    if (!layout || !toggle || !backdrop) return;

    const mq = window.matchMedia(DESKTOP_MQ);

    function isMobileLayout() {
        return !mq.matches;
    }

    function readStoredHidden() {
        try {
            const raw = localStorage.getItem(STORAGE_KEY);
            if (raw === null) return isMobileLayout();
            return raw === '1';
        } catch {
            return isMobileLayout();
        }
    }

    function persistHidden(hidden) {
        try {
            localStorage.setItem(STORAGE_KEY, hidden ? '1' : '0');
        } catch {
            /* private mode or quota */
        }
    }

    function applyVisual(hidden) {
        if (isMobileLayout()) {
            layout.classList.toggle('sidebar-open', !hidden);
            layout.classList.remove('sidebar-collapsed');
            backdrop.hidden = hidden;
            document.body.classList.toggle('layout-drawer-open', !hidden);
        } else {
            layout.classList.toggle('sidebar-collapsed', hidden);
            layout.classList.remove('sidebar-open');
            backdrop.hidden = true;
            document.body.classList.remove('layout-drawer-open');
        }
        toggle.setAttribute('aria-expanded', hidden ? 'false' : 'true');
    }

    function syncFromStorage() {
        applyVisual(readStoredHidden());
    }

    toggle.addEventListener('click', () => {
        const hidden = isMobileLayout()
            ? layout.classList.contains('sidebar-open')
            : !layout.classList.contains('sidebar-collapsed');
        persistHidden(hidden);
        applyVisual(hidden);
    });

    backdrop.addEventListener('click', () => {
        if (!isMobileLayout()) return;
        persistHidden(true);
        applyVisual(true);
    });

    document.addEventListener('keydown', (e) => {
        if (e.key !== 'Escape') return;
        if (!isMobileLayout() || !layout.classList.contains('sidebar-open')) return;
        persistHidden(true);
        applyVisual(true);
    });

    mq.addEventListener('change', syncFromStorage);
    syncFromStorage();
}

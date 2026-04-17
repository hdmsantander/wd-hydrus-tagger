/**
 * Gallery viewer navigation limited to the current multi-selection (search order).
 */

const CYCLE_SELECTION_LS_KEY = 'wd_tagger_gallery_viewer_cycle_selection';

export function readGalleryViewerCycleSelection() {
    try {
        const v = localStorage.getItem(CYCLE_SELECTION_LS_KEY);
        /* Unset: default on so multi-select loops selected files without hunting the toolbar toggle. */
        if (v === null) return true;
        return v === '1';
    } catch {
        return true;
    }
}

export function writeGalleryViewerCycleSelection(on) {
    try {
        localStorage.setItem(CYCLE_SELECTION_LS_KEY, on ? '1' : '0');
    } catch {
        /* quota / private mode */
    }
}

/**
 * @param {{ fileIds: number[], selectedIds: Set<number> }} state
 * @returns {number[]}
 */
export function orderedSelectedFileIds(state) {
    if (!state?.fileIds?.length) return [];
    return state.fileIds.filter((id) => state.selectedIds?.has?.(id));
}

/**
 * When true, Prev/Next and arrow keys wrap only among selected files (multi-select).
 * @param {{ fileIds: number[], selectedIds: Set<number>, galleryViewerCycleSelection?: boolean }} state
 * @param {number|null|undefined} displayedFileId
 */
export function viewerUsesSelectionOnlyNav(state, displayedFileId) {
    if (!state?.galleryViewerCycleSelection) return false;
    const order = orderedSelectedFileIds(state);
    if (order.length <= 1) return false;
    if (displayedFileId == null) return false;
    if (!state.selectedIds?.has?.(displayedFileId)) return false;
    return true;
}

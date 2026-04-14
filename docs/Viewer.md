# Image viewer (“booru-style”) — design proposal

This document describes the **gallery image viewer**: a full-screen visor for inspecting the Hydrus-backed file, editing tags inline (booru-style chips), and running **Predict / Apply** on the same HTTP API as the main tagger.

## Goals

- **Inspect** one file at full resolution (same proxy as the gallery: `GET /api/files/{file_id}`) with **current Hydrus tags** visible.
- **Edit** tags in-place (booru-style: tag chips, add/remove, optional rating/general/character grouping) without leaving the viewer.
- **Reuse** existing HTTP endpoints and the same tag payload shapes as the main tagger flow.
- **Preserve** normal gallery selection: plain click and shift-range selection stay as today.
- **Open** the viewer deliberately: **Ctrl+click** (or **Cmd+click** on macOS) **or** a **double-click** on a card (the browser’s native double-click interval, so a select followed by a slower deselect does not open the viewer).

## Current implementation (shipped)

| Area | Behaviour |
|------|-----------|
| **Tagged badge** | On cards that already have tags on the **selected tag service** (`#select-service`), the badge shows a compact marker; on **hover** or **keyboard focus** it expands to **Tagged (N)**. The pill uses a **fixed 11px label** and **scale transform** on hover so text does not reflow awkwardly. |
| **Gestures** | **Ctrl/Cmd+click**: opens the viewer (no selection change). **Double-click** (`click` with `event.detail >= 2`): opens the viewer **without** applying the second click’s selection toggle. **Touch**: **press-and-hold** (~560 ms) with **retro frame + scanline overlay** on the card, then the viewer opens; the following click is suppressed so selection does not flip. |
| **Cinema mode** | **Cinema** button or **C** key: **radial dimmed backdrop**, **edge-to-edge panel**, **black letterbox** image stage, **taller tag column** (up to ~42dvh) for a theater-style layout. Preference stored in `localStorage` (`wd_tagger_viewer_cinema`). |
| **Gallery performance** | When the **current page’s file id list** is unchanged, the grid **patches cards in place** (selection, badge counts, tooltips) so **thumbnail `<img>` nodes are not recreated**, reducing flicker during multi-select or metadata refreshes. |
| **Viewer UI** | Full-screen overlay: **thumbnail first** (Hydrus proxy thumbnail), light blur while the **full file** prefetches via `Image()` then decodes on the visible `<img>`. Sidebar: **read-only tags** for other Hydrus services; **editable chips** for the selected write service; **Predict**, **Apply to Hydrus**, **Reset**. **Previous / Next** and **ArrowLeft / ArrowRight** move within the current **search `fileIds`** order (disabled when the file is not in that list). **Esc** blurs the tag field first, then closes. |
| **Data** | Shared **`extractTagsByService` / `getTagsForService`** in `frontend/js/utils/hydrus.js`. `POST /api/files/metadata`, `GET` thumbnail + file URLs, `POST /api/tagger/predict`, `POST /api/tagger/apply` as in the main tagger. |

### Trade-off: Ctrl+click no longer toggles multi-selection

Previously, **Ctrl/Cmd+click** toggled membership in the selection set (common desktop pattern). It is now reserved for **opening the viewer**. Plain click still toggles a single card; **Shift+click** still extends a range from `lastClickIndex`.

**Possible follow-up (not implemented):** reintroduce additive selection with **Alt+click** (or a toolbar “selection mode”) and document it in the gallery help text.

## Layout (implemented + optional follow-ups)

- **Left:** phased image (thumbnail → full), **Previous / Next** for mobile-friendly navigation.
- **Right:** read-only **other services**, **editable chips** for the selected write service, **Predict** (merges model output), **Apply to Hydrus**, **Reset** to Hydrus tags. Enter adds a chip; Backspace on an empty field pops the last chip.
- **Optional later:** zoom/pan, per-tag confidence display, swipe between files, “open in tagger results”.

### Endpoints to reuse (no new backend required for a first editable version)

| Operation | Endpoint | Notes |
|-----------|----------|--------|
| Full image | `GET /api/files/{file_id}` | Already used by the viewer. |
| Metadata / tags | `POST /api/files/metadata` | Refresh after apply; same chunking rules as the gallery. |
| ONNX predict | `POST /api/tagger/predict` | Body: `file_ids`, `general_threshold`, `character_threshold`, optional `batch_size`. For single-file UX, send one id. |
| Write tags | `POST /api/tagger/apply` | Same payload as results apply in `tagger.js`: `{ file_id, hash, tags[] }` rows + `service_key`. Requires **hash** from metadata (already present on Hydrus rows). |

Optional later enhancements (only if product needs them):

- **WebSocket progress** (`/api/tagger/ws/progress`): not required for single-file predict/apply; keep HTTP-only for simplicity.
- **Dedicated “patch tags” API**: today `apply` merges with Hydrus semantics already; a separate “replace all tags” API is **not** required unless Hydrus-side behaviour demands it.

### Frontend modules

1. **`viewer.js`** — `_viewerDisplayedFileId`, draft tags, phased `Image()` prefetch, Predict/Apply/Reset, arrow navigation.
2. **`gallery.js`** — `buildGalleryPageKey` + `syncGalleryCard` incremental updates; `extractTagsByService` from `utils/hydrus.js`.
3. **`utils/hydrus.js`** — `extractTagsByService`, `getTagsForService`.
4. **`style.css`** — viewer stack, chips, 44px nav buttons, gallery `touch-action: manipulation`.

### Interaction details

- **Service key:** use `#select-service` (same as tagger) for writes; show read-only blocks for other services from metadata.
- **After Apply:** call `getMetadata([fileId])`, merge into `state.metadata`, re-render chips and gallery card badge counts on `renderGrid()` if the page is visible.
- **Predict:** show a non-blocking loading state on the predict button; on success, map `general_tags` / `character_tags` / `rating_tags` to chips using the same prefix rules as `applyTags()` in `tagger.js` (read thresholds from sliders or config).
- **Keyboard:** Esc closes; when tag input focused, Esc could clear focus first — document behaviour to avoid accidental close.

### Accessibility & motion

- **Focus trap** inside the dialog when it contains interactive controls (phase 2+).
- **`prefers-reduced-motion`:** gallery shake already disabled in CSS; extend to viewer panel transition if needed.

### Testing

- Manual: Ctrl+click and double-click on tagged vs untagged files; phased image load; Predict/Apply/Reset; Prev/Next and arrow keys; metadata missing path; backdrop close; body scroll lock; mobile nav buttons (44px min height).
- Automated: `tests/test_gallery_viewer_frontend.py` (static guards); `tests/test_frontend_english.py` includes `viewer.js` in the core module list.

## Summary

The viewer now includes **inline tag editing** (chips + Predict/Apply) and **incremental gallery DOM updates** to keep thumbnails stable. Further polish (zoom/pan, swipe gestures, focus trap) can build on the same module layout.

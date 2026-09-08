package com.bk.drawing

object NativeRenderer {
    init { System.loadLibrary("drawing") }

    /**
     * Configure where on disk the document is persisted. Tiles are read from
     * and written to `<dir>/tile_<tx>_<ty>.bin`. Call before any draw work;
     * loading happens lazily on the first GL operation.
     */
    external fun setDocumentDir(path: String)

    /**
     * Switch to a different document at runtime. Frees the current doc's
     * GL state, clears selection/undo/pending writes, points g_docDir at
     * the new path, and lazy-loads the new doc on the next render. Queued
     * through the action drain so all GL work runs on the GL thread.
     *
     * Call drawingView.forceRedraw() afterward to trigger the load and
     * subsequent compositor pass.
     */
    external fun loadDocument(path: String)

    /**
     * Block the calling thread until every queued tile write/delete
     * has been drained. Tile saves run on a background thread; this is
     * the synchronization point used at doc switches and on app pause
     * so the user's last strokes can't be lost if the process is
     * killed shortly after.
     */
    external fun flushTileWrites()

    /**
     * Drain deferred per-tile glReadPixels work that commitStroke
     * queues. Must be invoked from the GL thread (uses GL functions).
     * Called from the no-stroke onDrawMultiBufferedLayer path so the
     * cost is paid on a quiet idle frame instead of during commits.
     */
    external fun flushPendingSaveTiles()

    /** Reset emitter state and start a new in-progress stroke. */
    external fun beginStroke()

    /**
     * Batched stroke extension. xyp packs `[x,y,p, x,y,p, ...]` in
     * doc-pixels; the first `realCount` triples are real samples
     * (persisted for the bake and used to update the coverage mirror)
     * and the remainder are predicted (rendered to the front-buffer
     * preview only and reverted at the next batch). One mirror blit
     * and one preview-overlay render per batch regardless of sample
     * count.
     */
    external fun extendStrokeBatch(
        width: Int, height: Int,
        transform: FloatArray,
        xyp: FloatArray, realCount: Int
    )


    /**
     * Bake the in-progress stroke into the tiles its bbox touches, then drop
     * the stroke samples. The tiles ARE the document state from now on.
     */
    external fun commitStroke()

    /**
     * Clear the bound (multi-buffered) layer to white and composite every
     * allocated tile onto it.
     */
    external fun renderDocument(
        width: Int, height: Int,
        transform: FloatArray
    )

    /**
     * Append a new (empty) layer above the current top, and switch the
     * active layer to it. Safe to call from any thread; the action is
     * applied on the GL thread at the start of the next operation.
     */
    external fun addLayer()

    /**
     * Move the active-layer pointer to the next layer, wrapping. Safe to
     * call from any thread.
     */
    external fun cycleActiveLayer()

    /**
     * Delete every tile in the active layer (GL textures + on-disk files).
     * For a vector layer, drops every shape and rewrites the empty marker.
     * Safe to call from any thread.
     */
    external fun clearActiveLayer()

    /**
     * Append a new (empty) vector layer above the current top, switch the
     * active layer to it. Vector layers hold parametric shapes (lines for
     * now; circles/rects later) instead of a tile grid.
     */
    external fun addVectorLayer()

    /**
     * Delete the layer at [idx] on the active page. No-op if only one layer
     * remains. The undo stack is cleared because indices shift. Safe to
     * call from any thread; runs on the GL thread on the next operation.
     */
    external fun deleteLayer(idx: Int)

    /**
     * Move the layer at [fromIdx] to position [toIdx] on the active page.
     * Both indices must be in range. Reorders both the in-memory stack
     * and the on-disk layer_<n>/ dirs. Clears the undo stack.
     */
    external fun moveLayer(fromIdx: Int, toIdx: Int)

    /**
     * Rasterize the vector layer at [idx]: bake every shape into raster
     * tiles in place, drop the shape arrays, and flip the layer type
     * from vector to raster. No-op if the layer is already raster, the
     * idx is out of range, or no page bounds are set. Queued onto the
     * GL thread; clears the undo stack since shape indices become stale.
     */
    external fun rasterizeLayer(idx: Int)

    /**
     * Rasterize the currently-selected vector shape onto the raster
     * layer directly below the source. No-op if there's no selection,
     * the source isn't a vector layer, the layer below doesn't exist or
     * isn't raster, or page bounds are unset. Queued; clears undo.
     */
    external fun rasterizeSelectionToLayerBelow()

    /**
     * Merge the raster layer at [idx] down onto the raster layer
     * directly below it using premultiplied "src over dst" — the top
     * layer's pixels stay on top. The source layer is then deleted and
     * trailing layer dirs renumber to fill the gap. No-op when idx==0
     * (nothing below) or either layer isn't raster. Queued; clears the
     * undo stack since target tile pixels change in a way prior entries
     * can't reverse. Caller should forceRedraw + resync layer state.
     */
    external fun mergeLayerWithBelow(idx: Int)

    /**
     * Append shapes to the active layer (only effective if the active
     * layer is a vector layer). Color and width follow the current brush
     * color and a fixed line width. Each gesture's (x0, y0, x1, y1)
     * interpretation:
     *   addLine:      endpoints
     *   addRectangle: opposite bbox corners
     *   addEllipse:   opposite bbox corners (oval inscribed)
     *   addCircle:    p0 = center, p1 = point on circle
     * Safe to call from any thread; takes effect on the next render.
     */
    external fun addLine     (x0: Float, y0: Float, x1: Float, y1: Float)
    external fun addRectangle(x0: Float, y0: Float, x1: Float, y1: Float)
    external fun addEllipse  (x0: Float, y0: Float, x1: Float, y1: Float)
    external fun addCircle   (x0: Float, y0: Float, x1: Float, y1: Float)

    /**
     * Live preview for the shape tools. shapeType:
     *   0 = line, 1 = rectangle, 2 = circle, 3 = ellipse
     * Renders into the currently-bound framebuffer (intended to be the
     * front-buffered layer, called from inside onDrawFrontBufferedLayer).
     * Clears the buffer first so successive previews replace rather than
     * accumulate.
     */
    external fun renderShapePreview(
        width: Int, height: Int,
        transform: FloatArray,
        shapeType: Int,
        x0: Float, y0: Float, x1: Float, y1: Float,
        snapped: Boolean
    )

    /**
     * Find the nearest snap target to (x, y) within the snap radius.
     * Fills output[0..2] with [snapX, snapY, didSnap (1.0/0.0)]. Output
     * is overwritten in place; pre-allocate one FloatArray(3) and reuse.
     * Snap targets are vector-shape vertices/centers and (when grid is
     * on) grid intersections. No-op + didSnap=0 if snap is disabled.
     */
    external fun snapPoint(x: Float, y: Float, output: FloatArray)

    /** Toggle whether snapping is active. Default on. */
    external fun setSnapEnabled(enabled: Boolean)

    /**
     * Mirror of DrawingSurfaceView.angleSnapEnabled. When on,
     * applyRotateTo locks rotation to 15° increments, and dragging a
     * Line endpoint via the scale handle locks the line direction to
     * 15° rays from the anchor (other endpoint).
     */
    external fun setAngleSnapEnabled(enabled: Boolean)

    /**
     * Runtime toggle for motion-predicted dabs in the front-buffer
     * preview. When off, the Kotlin side stops dispatching predicted
     * batches; when on, predicted samples ride along in the
     * extendStrokeBatch payload after the real samples and are
     * reverted from the live preview before the next real batch is
     * applied.
     */
    external fun setPredictionEnabled(enabled: Boolean)
    external fun isPredictionEnabled(): Boolean

    /**
     * Set the page-boundary rectangle (in doc-pixels). Drawn during
     * composite under the layers so user strokes locally occlude it but
     * the rest of the outline stays visible — useful as a visual anchor
     * when zoomed/rotated. Pass a zero-size rect to disable.
     */
    external fun setPageBounds(x0: Float, y0: Float, x1: Float, y1: Float)

    /**
     * Update the cached view scale (doc-px per view-px). Native uses it
     * to keep snap/hit-test radii and selection handles screen-relative
     * across pan/zoom. Default 1.0 (no zoom).
     */
    external fun setViewScale(scale: Float)

    /**
     * Drop the in-progress brush/eraser stroke without baking. Used by
     * the touch handler when a 2-finger gesture interrupts a stroke.
     */
    external fun discardStroke()

    /**
     * Selection helpers — operate on the active vector layer.
     *
     *  selectShapeAt(x, y): hit-test, set selection on hit, returns true
     *     on hit; on miss clears any prior selection and returns false.
     *  hasSelection: whether anything is currently selected.
     *  clearSelection: drop the current selection.
     *  translateSelection: move the selected shape by (dx, dy) doc px.
     *  moveSelectionTo:    snap-aware absolute move; drives an in-progress
     *     Move drag using the captured pen-to-center offset.
     *  deleteSelection: remove the selected vector shape(s) from their
     *     layer, OR — if a floating raster selection is active — discard
     *     its lifted pixels (source layer keeps the hole). The raster
     *     path needs a GL context, so callers must queue this onto the
     *     GL thread (via DrawingSurfaceView.queueDeleteSelection) when
     *     a raster selection is active.
     *  persistActiveVectorLayer: write shapes.bin for the active layer
     *     (used after a transform drag completes).
     *
     * All are safe to call from any thread.
     */
    external fun selectShapeAt(x: Float, y: Float): Boolean
    external fun hasSelection(): Boolean
    external fun clearSelection()
    external fun translateSelection(dx: Float, dy: Float)
    external fun moveSelectionTo(x: Float, y: Float)
    external fun deleteSelection()
    external fun persistActiveVectorLayer()

    /**
     * Begin a SELECT-tool interaction at (x, y). Returns the drag mode:
     *   0 = none (tap missed everything; selection cleared if there was one)
     *   1 = move  (drive moveSelectionTo with absolute pen pos; native
     *              uses the offset captured here so the grab-point follows)
     *   2 = scale (drive updateInteractionAt with absolute pen pos)
     *   3 = rotate (drive updateInteractionAt with absolute pen pos)
     */
    external fun beginInteractionAt(x: Float, y: Float): Int

    /** Drive an in-progress scale, rotate, or move interaction to (x, y).
     *  No-op if no interaction is active. (For Move, prefer the dedicated
     *  moveSelectionTo entry point — both go through the same path.) */
    external fun updateInteractionAt(x: Float, y: Float)

    /** Mark the current interaction complete. */
    external fun endInteraction()

    /**
     * Set the active drawing tool: 0 = brush, 1 = eraser. The next stroke
     * (i.e. the next beginStroke) snapshots this value, so toggling
     * mid-stroke doesn't split a stroke. Safe to call from any thread.
     */
    external fun setTool(tool: Int)

    /**
     * Set the brush RGB color (alpha is fixed). `rgb` is 0xRRGGBB; any
     * upper bits are ignored. Snapshotted at the next beginStroke. Safe
     * to call from any thread.
     */
    external fun setBrushColor(rgb: Int)

    /**
     * Brush size scale — multiplier on the per-pressure dab radius
     * (1.0 = the default 2..18 doc-px range). Snapshotted at beginStroke
     * so a slider change mid-stroke doesn't visibly split a stroke.
     */
    external fun setBrushSize(scale: Float)

    /**
     * Brush opacity in [0, 1]. Out-of-range values are clamped;
     * non-finite values are ignored. Snapshotted at beginStroke so a
     * mid-stroke change doesn't split a stroke. Affects brush only —
     * the eraser uses its own historical strength.
     */
    external fun setBrushAlpha(alpha: Float)

    /**
     * Stroke alpha mode. When true, overlapping dabs within a single
     * stroke clamp to the brush alpha (no buildup where the stroke
     * crosses itself) — useful for shading with low-alpha colors.
     * When false (default), opacity stacks within a stroke. Either way
     * the value is snapshotted at beginStroke, so a mid-stroke toggle
     * doesn't split a stroke between the two modes.
     */
    external fun setStrokeUniformAlpha(enabled: Boolean)

    /**
     * Dab hardness in [0, 1] — the radial fraction at which a single
     * dab becomes fully opaque. 1 = solid disc (hard edge); 0 = full
     * gradient from opaque center to transparent rim (smooth edge).
     * Snapshotted at beginStroke; applies to brush and eraser alike
     * (the same accumulation curve drives both, so the same per-dab
     * shape is appropriate).
     */
    external fun setBrushHardness(hardness: Float)

    /**
     * Bucket-fill "bleed" in pixels — how far the filled region grows
     * outward past the flood-fill tolerance match. 0 stops cleanly at
     * the boundary; larger values bridge the anti-aliased edge so the
     * fill bleeds under adjacent strokes. Read at fill time, so the
     * next bucket tap picks up changes. Clamped to [0, 64] in native.
     */
    external fun setBucketBleed(px: Int)

    /**
     * Vector-tool line width in doc-pixels. Captured at the time a shape
     * is added (addLine / addRectangle / etc.); subsequent changes don't
     * affect existing shapes.
     */
    external fun setVectorLineWidth(width: Float)

    /**
     * Raster selection (Phase 1: rectangular, translate-only).
     *
     *  beginRasterSelection: lift a doc-coord rectangle from the active
     *     raster layer into a floating selection. Returns false if the
     *     active layer isn't raster, the rect is empty, or a selection
     *     is already active (caller should commit/cancel first).
     *  translateRasterSelection: incrementally pan the floating selection.
     *  commitRasterSelection: bake the selection at its current position.
     *     Pushes a single undo entry covering both lift and drop.
     *  cancelRasterSelection: restore the lifted pixels to their original
     *     position; no undo entry recorded.
     *  hasRasterSelection / rasterSelectionContains: queries for UI.
     */
    external fun beginRasterSelection(
        x0: Float, y0: Float, x1: Float, y1: Float
    ): Boolean

    /**
     * Lasso (freeform polygon) lift. `points` is a flat
     * [x0,y0,x1,y1,...] FloatArray of doc-coord vertices; the polygon
     * closing edge from points[n-1] → points[0] is implicit. Returns
     * false if the active layer isn't raster, the polygon has fewer
     * than 3 vertices, or a selection is already active.
     *
     * Same lifecycle as beginRasterSelection — needs a current GL
     * context, so callers should queue this onto a multi-buffer pass.
     */
    external fun beginLassoSelection(points: FloatArray): Boolean

    /**
     * Live preview for the lasso tool during drag. Renders the
     * in-progress polyline into the bound (front-buffered) framebuffer,
     * clearing first so successive previews replace rather than
     * accumulate. `closed` controls whether the implicit closing edge
     * is drawn (typically false during drag, true on release).
     */
    external fun renderLassoPathPreview(
        width: Int, height: Int,
        transform: FloatArray,
        points: FloatArray,
        closed: Boolean
    )

    external fun translateRasterSelection(dx: Float, dy: Float)
    external fun commitRasterSelection()
    external fun cancelRasterSelection()
    external fun hasRasterSelection(): Boolean
    external fun rasterSelectionContains(x: Float, y: Float): Boolean

    /**
     * Copy / paste for the floating raster selection.
     *
     *  copySelection: snapshot the active selection's pixels and OBB
     *     into a process-global native clipboard. Selection is left
     *     unchanged. No-op if no selection is active. Needs a current
     *     GL context (uses a temp FBO + glReadPixels).
     *  pasteSelection: create a fresh floating selection from the
     *     clipboard, placed at the same doc-coord OBB as the original
     *     copy (so Copy → Paste is duplicate-in-place; the user can
     *     drag the new floating sel to reveal the source). Auto-commits
     *     any existing floating selection first. Returns false if the
     *     clipboard is empty or the active layer isn't raster. Needs a
     *     current GL context.
     *  hasClipboardContent: true if the clipboard holds something.
     *     Cheap; safe to call from any thread for UI button enabling.
     */
    external fun copySelection()
    /**
     * Cut: snapshot the active floating raster selection's pixels into
     * the global clipboard, then discard the floating selection
     * without restoring its lifted tiles — the source layer keeps the
     * hole. No-op if no selection is active. Same GL-thread / current-
     * context requirements as copySelection. Pair with pasteSelection
     * later to drop the cut content elsewhere.
     */
    external fun cutSelection()
    external fun pasteSelection(): Boolean
    external fun hasClipboardContent(): Boolean

    /**
     * What's currently in the clipboard:
     *   0 = empty
     *   1 = raster pixels (paste creates a floating raster selection)
     *   2 = vector shape (paste appends to active vector layer)
     * UI uses this to pick the right select tool after a paste.
     */
    external fun getClipboardKind(): Int

    /**
     * Raster floating-selection transforms (Phase 3). Mirror the vector
     * selection JNIs but operate on the active floating raster
     * selection's OBB (centerX/Y + halfW/H + rotation).
     *
     *  beginRasterInteractionAt: hit-test the selection's handles +
     *     body. Returns drag mode:
     *       0 = none   (no active selection, or tap missed all handles
     *                   AND fell outside the selection body)
     *       1 = move   (tap inside body)
     *       2 = scale  (corner handle)
     *       3 = rotate (rotate handle)
     *     Snapshots whatever drag-anchor / pen-offset state subsequent
     *     updateRasterInteractionAt calls need.
     *  updateRasterInteractionAt: drive the in-progress drag with the
     *     pen's current doc-coord position. No-op if mode is None.
     *  endRasterInteraction: clear the drag mode. Does NOT push an
     *     undo entry — the eventual commitRasterSelection bakes a
     *     single RasterStroke entry covering lift + final placement.
     *
     * Safe to call from any thread; no GL access.
     */
    external fun beginRasterInteractionAt(x: Float, y: Float): Int
    external fun updateRasterInteractionAt(x: Float, y: Float)
    external fun endRasterInteraction()

    /**
     * Bucket fill at (x, y) in doc-pixels using the current brush color.
     * Uses the visible page composite as the flood-fill source (so vector
     * outlines on other layers act as boundaries) and writes pixels into
     * the active raster layer. No-op if the active layer is vector or
     * the seed is outside the page rectangle. Synchronous on the GL
     * thread — call from inside an onDrawMultiBufferedLayer callback or
     * be prepared for a brief freeze.
     */
    external fun bucketFillAt(x: Float, y: Float)

    /**
     * Page-background grid controls. Read at multi-buffer composite time;
     * call forceRedraw() afterward to make a change appear immediately.
     */
    external fun setGridEnabled(enabled: Boolean)
    external fun setGridStyle(style: Int)   // 1 = lines, 2 = dots
    external fun setGridSpacing(spacingDocPx: Float)   // minor-line pitch; also the snap pitch

    /**
     * Page margins — rose rules inset from the page edge: across the
     * top and bottom at [top] doc-px, down each side at [side] doc-px.
     * Drawn in the page background with the grid (under the layers,
     * included in exports). Per-document; MainActivity mirrors them
     * from page_setup.txt on every doc switch.
     */
    external fun setMarginsEnabled(enabled: Boolean)
    external fun setMarginSizes(top: Float, side: Float)

    /**
     * Per-pixel grid overlay. When enabled AND the view is zoomed
     * enough that one doc-pixel covers several buffer-pixels, renders
     * a 1-buffer-pixel line at every integer doc-pixel boundary on top
     * of the layers. Below the zoom threshold it's a no-op (the grid
     * would be denser than the resolution). Call forceRedraw() to
     * make a toggle take effect immediately.
     */
    external fun setPixelGridEnabled(enabled: Boolean)

    /**
     * Layer-state read accessors for UI display. Values may lag queued
     * addLayer/cycleActiveLayer calls by up to one stroke/render; that's
     * fine for status text.
     */
    external fun getLayerCount(): Int
    external fun getActiveLayer(): Int

    /**
     * True once the GL thread has fully loaded the document from disk.
     * Polled by the UI on launch / after loadDocument so the layer panel
     * doesn't render with default-constructed Layer slots before disk
     * metadata (names, types, visibility) lands.
     */
    external fun isFullyLoaded(): Boolean

    /** Page rect dimensions in doc-px. Returns 0 when no page bounds
     *  are active; in that case the caller should fall back to the
     *  current SurfaceView size (the document behaves as an infinite
     *  plane). Used by the exporter to pick a 1:1 output resolution. */
    external fun getPageWidth(): Int
    external fun getPageHeight(): Int

    /**
     * Per-layer metadata accessors for the active page.
     *
     *  getLayerType: 0 = raster, 1 = vector. Returns 0 for out-of-range
     *     indices so the UI can fail soft.
     *  getLayerName: user-set display name, or "" if none. The UI
     *     should fall back to a default like "layer N" / "vector N".
     *  setLayerName: store + persist the name. Pass "" to clear.
     *
     * Names persist as plain UTF-8 in &lt;layerDir&gt;/name.txt; absence of
     * the file means "no custom name". Safe to call from any thread.
     */
    external fun getLayerType(idx: Int): Int
    external fun getLayerName(idx: Int): String
    external fun setLayerName(idx: Int, name: String)

    /**
     * Per-layer visibility. Hidden layers are skipped by both the main
     * compositor and the eraser preview's "below" snapshot. Visibility
     * persists as the *presence* of &lt;layerDir&gt;/hidden.flag (absence
     * = visible; that's the safe default for layers created before the
     * feature existed). Caller should forceRedraw() after a setVisible
     * to make the change appear immediately.
     */
    external fun getLayerVisible(idx: Int): Boolean
    external fun setLayerVisible(idx: Int, visible: Boolean)

    /**
     * Per-layer opacity in [0, 1]. The composite shaders multiply
     * their premultiplied output by this scalar. Persisted at
     * &lt;layerDir&gt;/opacity.txt; absence = 1.0 (the safe default for
     * layers created before this feature existed). Caller should
     * forceRedraw() after setLayerOpacity.
     */
    external fun getLayerOpacity(idx: Int): Float
    external fun setLayerOpacity(idx: Int, opacity: Float)

    /**
     * Multi-page document support. Each page has its own independent
     * layer stack and on-disk subdirectory (`page_<idx>/`).
     *
     *  getPageCount   - number of pages currently in the document.
     *  getActivePage  - index of the currently-shown page.
     *  addPage        - append an empty page and switch to it. Queued.
     *  switchPage     - change the active page. Queued; clears selection
     *                   and the undo stack (undo entries are scoped to
     *                   the previous page).
     *
     * After invoking addPage/switchPage, call drawingView.forceRedraw()
     * and re-sync any UI state mirrors.
     */
    external fun getPageCount(): Int
    external fun getActivePage(): Int
    external fun addPage()
    external fun switchPage(idx: Int)

    /**
     * Delete the page at [idx]. No-op if only one page remains. Frees
     * GL resources for every layer on that page, recursively wipes the
     * on-disk page_<idx>/ dir, renumbers trailing dirs to keep them
     * contiguous, and clears undo + selection. Queued through the
     * pending-action drain.
     */
    external fun deletePage(idx: Int)

    /**
     * Move the page at [fromIdx] to position [toIdx]. Reorders both
     * the in-memory page vector and the on-disk page_<n>/ dirs.
     * Per-page contents are preserved. Queued.
     */
    external fun movePage(fromIdx: Int, toIdx: Int)

    /**
     * Eyedropper. Queue a 1×1 sample request at [docX], [docY] in
     * doc pixels (the same space tile FBOs use). The GL thread runs
     * the composite at the end of the next compositeAllLayers pass —
     * walks visible raster layers, reads tile FBOs directly, and
     * Porter-Duff "over"-composites the layer pixels onto the page
     * background. After submitting, call [getLastSampledColor]
     * roughly one frame later to retrieve the result.
     */
    external fun requestColorSample(docX: Float, docY: Float)

    /**
     * Returns the most recently sampled color as 0xRRGGBB, or -1 if no
     * sample has been completed since the last requestColorSample.
     * Single-shot: subsequent calls return -1 until another request.
     */
    external fun getLastSampledColor(): Int

    /**
     * Import [argbPixels] (Android Color int format: 0xAARRGGBB) as a
     * floating raster selection on a fresh layer of the active page.
     * The drain creates a new layer at the top of the stack, uploads
     * the pixels into a content texture, and parks the result on
     * g_rasterSel with fixedAspect=true so the user can scale it
     * uniformly via the existing transform handles. Returns false if
     * the input is malformed (zero dims, mismatched array length).
     */
    external fun importImageAsSelection(width: Int, height: Int,
                                        argbPixels: IntArray): Boolean

    /**
     * Render a thumbnail of the given page into the provided Bitmap
     * (must be Bitmap.Config.ARGB_8888). Y-axis is oriented for
     * straight-into-Bitmap consumption (doc-top → bitmap-top). Letter-
     * boxes the page rect into the bitmap dimensions.
     *
     * [drawChrome] controls whether the on-screen page-boundary outline
     * is included. Sidebar thumbnails want it (it reads as a clear
     * "this is a page" frame at small sizes); PNG / PDF export wants
     * it off so the saved image has no 1-pixel border at the edges.
     *
     * Must be called from the GL thread (i.e. inside a render callback).
     */
    external fun renderPageThumbnail(pageIdx: Int, bitmap: android.graphics.Bitmap,
                                     drawChrome: Boolean)

    /**
     * Undo/redo. Both are queued onto the GL thread alongside layer
     * actions so the inverse mutation (which touches GL state and disk)
     * runs on the right thread. Call forceRedraw() afterward to make
     * the change appear immediately. canUndo/canRedo can be called from
     * any thread for UI button state.
     */
    external fun undo()
    external fun redo()
    external fun canUndo(): Boolean
    external fun canRedo(): Boolean

    // ---- Test-only access ------------------------------------------------
    //
    // Exposed for the androidTest fidelity suite. Driving the renderer
    // from tests instead of from real input lets us verify that the
    // bake is deterministic across implementation changes — same
    // inputs in, same pixels out. Not used by production code.

    /** Drain queued undo/redo/layer ops without doing a composite pass. */
    external fun flushPendingActions()

    /** Number of existing tiles in the layer (after any pending drain). */
    external fun getLayerTileCount(layerIdx: Int): Int

    /** Flat `[tx, ty, tx, ty, …]` of every existing tile in the layer.
     *  Insertion-ordered; sort before comparing if order matters. */
    external fun getLayerTileCoords(layerIdx: Int): IntArray

    /** kTileBytes (256×256×4) interior pixels of a tile, or null if the
     *  tile doesn't exist. */
    external fun readTileBytes(layerIdx: Int, tx: Int, ty: Int): ByteArray?
}

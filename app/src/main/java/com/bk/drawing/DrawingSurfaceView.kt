package com.bk.drawing

import android.content.Context
import android.opengl.Matrix
import android.os.SystemClock
import android.util.AttributeSet
import android.util.Log
import android.view.MotionEvent
import android.view.SurfaceView
import androidx.graphics.lowlatency.BufferInfo
import androidx.graphics.lowlatency.GLFrontBufferedRenderer
import androidx.graphics.opengl.egl.EGLManager
import androidx.input.motionprediction.MotionEventPredictor
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.sin

// Param type for the front-buffered renderer. Brush/eraser strokes flow
// through Sample (a stream per stroke); the shape tools (line, rect,
// circle, ellipse) flow through ShapePreview (one entry per pen position
// during the drag), with shapeType selecting which shape to render.
sealed class StrokeAction {
    /** All stroke samples coming out of one `MotionEvent` (historical
     *  rows + the current one) bundled into a single front-buffer
     *  dispatch. One renderFrontBufferedLayer call → one
     *  extendStrokeBatch JNI → one mirror blit + one preview overlay
     *  render, regardless of sample count. xyp is [x0,y0,p0, ...] in
     *  doc-coords; isNewStroke applies to the first sample only.
     *  inputAgeMs is the age of the *latest* sample in the batch
     *  (the freshest input); receivedNs is when this MotionEvent
     *  reached us. */
    class BatchSamples(
        val xyp: FloatArray,
        // First realCount triples are real samples (pushed to
        // g_current.samples and baked at commit). Triples after are
        // predicted, rendered into the front-buffer preview only and
        // reverted from g_coverage at the start of the next real batch.
        // Packing real + predicted into one BatchSamples means one
        // renderFrontBufferedLayer call per MotionEvent regardless of
        // whether prediction is on — preserves the appPrep win from
        // batching.
        val realCount: Int,
        val isNewStroke: Boolean,
        val inputAgeMs: Long = 0L,
        val receivedNs:  Long = 0L,
    ) : StrokeAction() {
        val count: Int get() = xyp.size / 3
        override fun equals(other: Any?): Boolean =
            other is BatchSamples
                && isNewStroke == other.isNewStroke
                && realCount   == other.realCount
                && xyp.contentEquals(other.xyp)
        override fun hashCode(): Int {
            var h = xyp.contentHashCode()
            h = 31 * h + isNewStroke.hashCode()
            h = 31 * h + realCount.hashCode()
            return h
        }
    }

    data class ShapePreview(
        val shapeType: Int,             // matches NativeRenderer.renderShapePreview
        val x0: Float, val y0: Float,
        val x1: Float, val y1: Float,
        val snapped: Boolean            // shows snap marker at (x1, y1)
    ) : StrokeAction()

    /** Live preview of the lasso (freeform selection) path during drag.
     *  Points are doc-coords [x0,y0,x1,y1,...]. `closed` is typically
     *  false during drag (no implicit closing edge while still drawing). */
    data class LassoPreview(
        val points: FloatArray,
        val closed: Boolean
    ) : StrokeAction() {
        override fun equals(other: Any?): Boolean =
            other is LassoPreview
                && closed == other.closed
                && points.contentEquals(other.points)
        override fun hashCode(): Int =
            31 * points.contentHashCode() + closed.hashCode()
    }
}

// Pen-to-render latency aggregator. Records three deltas per stroke
// batch on the GL thread:
//   inputAgeMs — sensor timestamp → app onTouchEvent receive (kernel
//                + InputDispatcher).
//   appPrepUs  — onTouchEvent receive → onDrawFrontBufferedLayer
//                start (queue onto GL thread + framework callback).
//   nativeUs   — duration of the extendStrokeBatch JNI call.
// When `enabled` is true, logs a p50/p95/max summary once kBucket
// samples are recorded, then resets. Off by default — flip this flag
// in code (not from UI) when investigating a regression. The hot-path
// nanoTime captures in the GL callback are cheap enough to leave in.
internal object LatencyProbe {
    /** Flip to `true` in source when investigating latency. Off in
     *  shipped builds so a long drawing session doesn't spam logcat. */
    var enabled: Boolean = false

    private const val kBucket = 50
    private val inputAgeMs = LongArray(kBucket)
    private val appPrepUs  = LongArray(kBucket)
    private val nativeUs   = LongArray(kBucket)
    private var n = 0

    @Synchronized
    fun record(inputAgeMsVal: Long, appPrepNs: Long, nativeNs: Long) {
        if (!enabled) return
        if (n >= kBucket) return
        inputAgeMs[n] = inputAgeMsVal
        appPrepUs[n]  = appPrepNs / 1000
        nativeUs[n]   = nativeNs  / 1000
        n++
        if (n >= kBucket) dumpLocked()
    }

    private fun dumpLocked() {
        if (n == 0) return
        val ia = inputAgeMs.copyOf(n).also { it.sort() }
        val ap = appPrepUs.copyOf(n).also { it.sort() }
        val nv = nativeUs.copyOf(n).also { it.sort() }
        val p50 = n / 2
        val p95 = (n * 95) / 100
        val mx  = n - 1
        Log.i("DrawingApp", String.format(
            "LATENCY n=%d  input(ms) p50=%d p95=%d max=%d  appPrep(us) p50=%d p95=%d max=%d  native(us) p50=%d p95=%d max=%d",
            n,
            ia[p50], ia[p95], ia[mx],
            ap[p50], ap[p95], ap[mx],
            nv[p50], nv[p95], nv[mx],
        ))
        n = 0
    }
}

// Only BRUSH and ERASER correspond to native "stroke tools" (running
// through beginStroke / extendStrokeBatch / commitStroke). The shape tools
// have their own gesture path and don't update the native stroke tool.
// BUCKET is a click-to-act tool with its own native entrypoint; nativeId
// is unused for it.
enum class Tool(val nativeId: Int) {
    BRUSH      (0),
    ERASER     (1),
    BUCKET     (-1),
    LINE       (2),
    RECTANGLE  (3),
    CIRCLE     (4),
    ELLIPSE    (5),
    // Text boxes on vector layers. Tap places an auto-width box, drag
    // places a fixed-width one, tap on a box edits it. No native tool
    // id — placement / editing go through TextEditController.
    TEXT       (-1),
    SELECT     (6),
    SELECT_RECT (-1),    // raster rectangle marquee; no native tool id
    SELECT_LASSO(-1),    // raster freeform marquee; no native tool id
    // Closed-path fill: traces a brush stroke, joins start to end with a
    // straight chord, and fills the enclosed region. Same raster stroke
    // pipeline as BRUSH — see kToolShade in renderer.cpp.
    SHADE      (7);

    /** Shape-type code passed to NativeRenderer for shape tools. */
    val shapeType: Int
        get() = when (this) {
            LINE      -> 0
            RECTANGLE -> 1
            CIRCLE    -> 2
            ELLIPSE   -> 3
            else      -> -1
        }

    val isShape: Boolean
        get() = this == LINE || this == RECTANGLE || this == CIRCLE || this == ELLIPSE

    /** Tools that drive the raster stroke pipeline (begin/extend/commit). */
    val isRasterStroke: Boolean
        get() = this == BRUSH || this == ERASER || this == SHADE
}

class DrawingSurfaceView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : SurfaceView(context, attrs) {

    private val callback = object : GLFrontBufferedRenderer.Callback<StrokeAction> {
        override fun onDrawFrontBufferedLayer(
            eglManager: EGLManager,
            width: Int,
            height: Int,
            bufferInfo: BufferInfo,
            transform: FloatArray,
            param: StrokeAction
        ) {
            android.os.Trace.beginSection("DrawingApp.onDrawFrontBufferedLayer")
            try {
            // Native shaders expect `transform` to be doc-pixel →
            // buffer-pixel; compose the framework's view→buffer with our
            // current doc→view here.
            val composed = composedTransform(transform)
            when (param) {
                is StrokeAction.BatchSamples -> {
                    if (param.isNewStroke) {
                        NativeRenderer.beginStroke()
                    }
                    val measure = param.receivedNs != 0L
                    val enterNs = if (measure) System.nanoTime() else 0L
                    NativeRenderer.extendStrokeBatch(
                        bufferInfo.width, bufferInfo.height,
                        composed,
                        param.xyp, param.realCount
                    )
                    if (measure) {
                        val afterNs = System.nanoTime()
                        LatencyProbe.record(
                            param.inputAgeMs,
                            enterNs - param.receivedNs,
                            afterNs - enterNs,
                        )
                    }
                }
                is StrokeAction.ShapePreview -> {
                    NativeRenderer.renderShapePreview(
                        bufferInfo.width, bufferInfo.height,
                        composed,
                        param.shapeType,
                        param.x0, param.y0, param.x1, param.y1,
                        param.snapped
                    )
                }
                is StrokeAction.LassoPreview -> {
                    NativeRenderer.renderLassoPathPreview(
                        bufferInfo.width, bufferInfo.height,
                        composed,
                        param.points, param.closed
                    )
                }
            }
            } finally {
                android.os.Trace.endSection()
            }
        }

        override fun onDrawMultiBufferedLayer(
            eglManager: EGLManager,
            width: Int,
            height: Int,
            bufferInfo: BufferInfo,
            transform: FloatArray,
            params: Collection<StrokeAction>
        ) {
            android.os.Trace.beginSection("DrawingApp.onDrawMultiBufferedLayer")
            try {
            val composed = composedTransform(transform)
            // Only commit a brush/eraser stroke if this batch actually
            // contained Sample entries. Line previews go through here too
            // when the line tool's commit calls renderer.commit(); we
            // mustn't bake an empty stroke in that case. cancelNextCommit
            // (set when a 2-finger gesture interrupts a stroke) suppresses
            // the bake entirely so the in-progress stroke is discarded.
            val hadStrokeSamples = params.any { it is StrokeAction.BatchSamples }
            if (hadStrokeSamples && !cancelNextCommit) {
                NativeRenderer.commitStroke()
            }
            cancelNextCommit = false
            // Drain a queued bucket fill before the main render so the
            // newly-painted tiles show up in this very pass.
            val bucket = pendingBucketFill
            if (bucket != null) {
                pendingBucketFill = null
                NativeRenderer.bucketFillAt(bucket.x, bucket.y)
            }
            // Drain raster-selection ops in the order they're allowed:
            // cancel → commit → begin. Commit before begin so a "tap
            // outside, define a new rect" gesture in one swipe drops the
            // old selection before lifting the new one.
            if (pendingCancelRasterSel) {
                pendingCancelRasterSel = false
                NativeRenderer.cancelRasterSelection()
            }
            if (pendingCommitRasterSel) {
                pendingCommitRasterSel = false
                NativeRenderer.commitRasterSelection()
            }
            val beginSel = pendingBeginRasterSel
            if (beginSel != null) {
                pendingBeginRasterSel = null
                NativeRenderer.beginRasterSelection(
                    beginSel.x0, beginSel.y0, beginSel.x1, beginSel.y1
                )
            }
            val beginLasso = pendingBeginLassoSel
            if (beginLasso != null) {
                pendingBeginLassoSel = null
                NativeRenderer.beginLassoSelection(beginLasso)
            }
            // Copy / cut are snapshots of the current selection, drained
            // BEFORE paste (which may auto-commit and replace the active
            // selection). Drain after begin so a fresh just-lifted
            // selection is also copyable in the same gesture. Cut also
            // discards the floating selection (source keeps the hole).
            if (pendingCopySel) {
                pendingCopySel = false
                NativeRenderer.copySelection()
            }
            if (pendingCutSel) {
                pendingCutSel = false
                NativeRenderer.cutSelection()
            }
            if (pendingDeleteSel) {
                pendingDeleteSel = false
                NativeRenderer.deleteSelection()
            }
            if (pendingPasteSel) {
                pendingPasteSel = false
                NativeRenderer.pasteSelection()
            }
            NativeRenderer.renderDocument(
                bufferInfo.width, bufferInfo.height,
                composed
            )
            // Text boxes whose raster is missing (fresh load, undo,
            // paste) or stale for this zoom get re-rasterized on the
            // UI thread; the upload lands on the next pass.
            if (NativeRenderer.textRasterNeeded()) {
                post { onTextRasterNeeded?.invoke() }
            }
            // Refresh page thumbnails for the sidebar. By default we only
            // re-render the ACTIVE page (the only one whose pixels can
            // have changed since the last multi-buffer pass under normal
            // drawing). setThumbnailTargets / requestFullThumbnailRefresh
            // sets a one-shot flag to refresh all entries the next time
            // around (used after addPage / switchPage / sidebar rebuild).
            //
            // Stroke commits SKIP the refresh here and defer it to a
            // trailing-edge forceRedraw — renderPageThumbnail's full
            // re-composite + glReadPixels is ~8 ms, enough to push the
            // commit past one vsync and cause a flash on fast handwriting.
            // Schedule a trailing-edge forceRedraw after every stroke
            // commit. The next no-stroke commit drains deferred tile
            // saves and (if the sidebar is open) refreshes thumbnails —
            // both costs we keep out of the stroke commit critical
            // path. dedup via removeCallbacks so a flurry of strokes
            // only triggers one trailing redraw.
            if (hadStrokeSamples) {
                removeCallbacks(thumbnailRefreshRunnable)
                postDelayed(thumbnailRefreshRunnable, kThumbnailDeferMs)
            } else {
                // Quiet frame — drain deferred saves and (if needed)
                // refresh thumbnails. Cheap if nothing's queued.
                NativeRenderer.flushPendingSaveTiles()
                removeCallbacks(thumbnailRefreshRunnable)

                val targets = thumbnailTargets
                if (targets != null && targets.isNotEmpty()) {
                    val pageCount  = NativeRenderer.getPageCount()
                    val activePage = NativeRenderer.getActivePage()
                    if (thumbnailRefreshAllOnce) {
                        thumbnailRefreshAllOnce = false
                        // Queue every non-active page; the active one is
                        // always rendered below, so queuing it would just
                        // make us composite it twice.
                        thumbnailRefreshQueue.clear()
                        for (idx in targets.keys) {
                            if (idx in 0 until pageCount && idx != activePage) {
                                thumbnailRefreshQueue.addLast(idx)
                            }
                        }
                    }

                    // Sidebar thumbnails include the page-edge outline so
                    // the page reads as a framed sheet at small sizes.
                    if (activePage in 0 until pageCount) {
                        targets[activePage]?.let { bitmap ->
                            NativeRenderer.renderPageThumbnail(activePage, bitmap,
                                drawChrome = true)
                        }
                    }

                    var budget = kThumbnailsPerPass
                    while (budget > 0 && thumbnailRefreshQueue.isNotEmpty()) {
                        val idx = thumbnailRefreshQueue.removeFirst()
                        val bitmap = targets[idx] ?: continue
                        if (idx !in 0 until pageCount) continue
                        NativeRenderer.renderPageThumbnail(idx, bitmap,
                            drawChrome = true)
                        budget--
                    }

                    // Unconditional, even when nothing rendered: this is
                    // also how MainActivity notices page-count / active-page
                    // drift and rebuilds the sidebar. After addPage the new
                    // active page isn't in `targets` yet, so gating this on
                    // "did we render something" would strand the sidebar.
                    post { onThumbnailsUpdated?.invoke() }
                    // More queued: come back next pass and keep draining.
                    // post() rather than calling forceRedraw() directly —
                    // commit() from inside the framework's own MB callback
                    // is the re-entrant case the deferred-refresh path
                    // already avoids.
                    if (thumbnailRefreshQueue.isNotEmpty()) post { forceRedraw() }
                }
            }

            // Drain any queued export render. Same pattern as the
            // thumbnail targets — bitmaps allocated on the UI thread, GL
            // thread fills the pixels, then we post the completion to
            // the UI thread to do the actual file write / encoding.
            val exportReq = pendingExportRequest
            if (exportReq != null) {
                pendingExportRequest = null
                for ((idx, bitmap) in exportReq.pages) {
                    if (idx in 0 until NativeRenderer.getPageCount()) {
                        // PNG / PDF exports drop the page outline so the
                        // saved image has no 1-pixel border at the edges.
                        NativeRenderer.renderPageThumbnail(idx, bitmap,
                            drawChrome = false)
                    }
                }
                post { exportReq.onComplete() }
            }
            } finally {
                android.os.Trace.endSection()
            }
        }
    }

    // Export render queue. queueExportRender stashes a request from the
    // UI thread; the next onDrawMultiBufferedLayer drains it and fills
    // each (pageIdx, bitmap) pair on the GL thread. After the pixels are
    // written, the completion callback runs on the UI thread.
    data class ExportPage(val pageIdx: Int, val bitmap: android.graphics.Bitmap)
    private data class ExportRequest(
        val pages: List<ExportPage>,
        val onComplete: () -> Unit,
    )
    @Volatile
    private var pendingExportRequest: ExportRequest? = null

    fun queueExportRender(pages: List<ExportPage>, onComplete: () -> Unit) {
        pendingExportRequest = ExportRequest(pages, onComplete)
        forceRedraw()
    }

    // Map of page index → Bitmap to write thumbnail pixels into. Set by
    // MainActivity each time the sidebar is (re)built. Read on the GL
    // thread; the map itself is only ever replaced wholesale, never
    // mutated in place.
    @Volatile
    private var thumbnailTargets: Map<Int, android.graphics.Bitmap>? = null
    @Volatile
    private var thumbnailRefreshAllOnce = false

    // Pending non-active pages awaiting a thumbnail render, drained a few
    // per multi-buffer pass.
    //
    // Pages load their raster tiles lazily now (see ensurePageContentLoaded
    // in renderer.cpp), and renderPageThumbnail is a load trigger — it
    // composites the page it's given. Rendering every thumbnail in one
    // pass therefore pulls every page's tiles off disk in a single GL
    // callback, which is the cost the lazy load exists to avoid. Spreading
    // the refresh means switching documents pays for the active page and
    // the rest fill in over the following frames.
    //
    // GL thread only.
    private val thumbnailRefreshQueue = ArrayDeque<Int>()
    // Per-pass budget. The active page is always refreshed on top of this.
    private val kThumbnailsPerPass = 2

    // Trailing-edge debounce: when a stroke commits we DON'T refresh the
    // thumbnail in the same multi-buffer pass (renderPageThumbnail does a
    // full compositeAllLayers + glReadPixels into a Bitmap — ~8 ms on the
    // MovinkPad, enough to push the commit past one 90 Hz vsync and cause
    // a flash during fast handwriting). Instead we mark the thumbnail
    // pending and schedule a trailing forceRedraw — when no new strokes
    // arrive for kThumbnailDeferMs, the next MB pass has no stroke samples
    // and the refresh runs there at no commit-latency cost. postDelayed on
    // a View is thread-safe (posts to the View's Handler), so calling it
    // from the GL thread is fine.
    private val thumbnailRefreshRunnable = Runnable { forceRedraw() }

    /** Notified on the UI thread after every batch of thumbnails finishes
     *  rendering. The sidebar uses this to ImageView.invalidate() each item. */
    var onThumbnailsUpdated: (() -> Unit)? = null

    fun setThumbnailTargets(targets: Map<Int, android.graphics.Bitmap>?) {
        thumbnailTargets = targets
        thumbnailRefreshAllOnce = true
    }

    /** Force every registered thumbnail to refresh on the next multi-buffer
     *  pass, even those for non-active pages. */
    fun requestFullThumbnailRefresh() {
        thumbnailRefreshAllOnce = true
    }

    init {
        // Force the panel to its highest refresh rate (90 Hz on the
        // MovinkPad). MainActivity sets window.attributes
        // .preferredDisplayModeId = <90 Hz mode>, but the framework was
        // ignoring it (dumpsys display showed the panel stuck at the
        // 60 Hz mode despite our preference). Surface.setFrameRate on
        // the rendering surface is a stronger signal — it tells the
        // framework this specific surface needs that rate, which under
        // CHANGE_FRAME_RATE_ALWAYS forces the panel mode switch for the
        // duration this surface is visible.
        //
        // Direct impact on the flash: each MB transition spans one
        // vsync of blackout window (FB hidden while the framework's
        // commit transaction propagates to the display). At 60 Hz that
        // window is ~16.7 ms and visibly flashes; at 90 Hz it shrinks
        // to ~11.1 ms and is much less perceptible.
        // Best-effort frame-rate hint. Under the system "90 Hz" display
        // setting this is redundant (the panel is already at 90); under
        // "Adaptive" it's ignored because GLFrontBufferedRenderer's child
        // SurfaceControls — not this surface — are what SurfaceFlinger
        // consults for rate selection, and androidx.graphics-core alpha05
        // doesn't expose setFrameRate on SurfaceControlCompat. Kept in
        // place so it engages on newer library versions / future code
        // paths that DO render directly through this Surface.
        holder.addCallback(object : android.view.SurfaceHolder.Callback {
            override fun surfaceCreated(holder: android.view.SurfaceHolder) {
                holder.surface.setFrameRate(
                    90f,
                    android.view.Surface.FRAME_RATE_COMPATIBILITY_DEFAULT,
                    android.view.Surface.CHANGE_FRAME_RATE_ALWAYS
                )
            }
            override fun surfaceChanged(
                holder: android.view.SurfaceHolder,
                format: Int, width: Int, height: Int
            ) {}
            override fun surfaceDestroyed(holder: android.view.SurfaceHolder) {}
        })
    }

    private var renderer: GLFrontBufferedRenderer<StrokeAction>? =
        GLFrontBufferedRenderer(this, callback)

    // Motion-prediction support. Lazily created on first stroke event.
    // When enabled, every real MotionEvent is fed to the predictor and a
    // predicted MotionEvent is dispatched as a separate BatchSamples
    // (predicted=true) so its dabs render into the front buffer ahead
    // of where the pen actually is. Native side reverts the predicted
    // dabs from g_coverage before applying the next real dab.
    private var motionPredictor: MotionEventPredictor? = null

    /** Runtime toggle. MainActivity flips this from a status-bar item.
     *  Default on — prediction has been validated as a clear win and
     *  the toggle remains so we can drop back to raw pen tracking if a
     *  corner case appears. Setter mirrors to native. */
    var predictionEnabled: Boolean = true
        set(value) {
            field = value
            NativeRenderer.setPredictionEnabled(value)
        }

    // -------------------------------------------------------------------
    // View transform: doc-pixel → view-pixel, parameterized as scale,
    // rotation (radians, applied around origin), and translation.
    // Identity by default. Updated by 2-finger gestures.
    //
    // The composition viewBufferTransform * docToViewMatrix is the matrix
    // we send to native shaders, so all native rendering operates in
    // doc-px and the view transform is "free" downstream.
    // -------------------------------------------------------------------
    private var viewScale = 1.0f
    private var viewRotation = 0.0f
    private var viewPanX = 0.0f
    private var viewPanY = 0.0f
    /** Public read-only mirror — UI overlays (brush preview) need it
     *  to convert doc-px brush radii to on-screen pixels. */
    val currentViewScale: Float get() = viewScale

    // Reused scratch buffers; touch handling runs at high rate, avoid alloc.
    private val tmpDoc = FloatArray(2)
    private val tmpDocToView = FloatArray(16)
    private val tmpComposed = FloatArray(16)

    // Most recent framework-supplied view→buffer matrix, captured every
    // onDraw{Front,Multi}BufferedLayer callback. The eyedropper handler
    // reads it on the UI thread to map a view-px touch into buffer-px
    // before submitting the sample request to native. Only changes on
    // surface rotation, so a one-frame stale snapshot is fine.
    private var lastFramebufferTransform: FloatArray? = null

    /** Convert a view-pixel coordinate to its doc-pixel equivalent. */
    private fun viewToDoc(vx: Float, vy: Float, out: FloatArray) {
        val tx = vx - viewPanX
        val ty = vy - viewPanY
        val invS = 1f / viewScale
        // Inverse rotation = R(-rotation).
        val c = cos(-viewRotation)
        val s = sin(-viewRotation)
        out[0] = invS * (c * tx - s * ty)
        out[1] = invS * (s * tx + c * ty)
    }

    /** Build a column-major 4x4 of the current doc→view affine. */
    private fun fillDocToViewMatrix(out: FloatArray) {
        val c = cos(viewRotation)
        val s = sin(viewRotation)
        val a = viewScale * c
        val b = viewScale * s
        // Column 0
        out[0]  =  a;  out[1]  =  b;  out[2]  = 0f; out[3]  = 0f
        // Column 1
        out[4]  = -b;  out[5]  =  a;  out[6]  = 0f; out[7]  = 0f
        // Column 2 (z)
        out[8]  = 0f;  out[9]  = 0f;  out[10] = 1f; out[11] = 0f
        // Column 3 (translation)
        out[12] = viewPanX; out[13] = viewPanY; out[14] = 0f; out[15] = 1f
    }

    /** Compose framework's view→buffer transform with our doc→view, giving
     *  the doc→buffer matrix the native shaders consume as `uTransform`.
     *  Side effect: caches the framebufferTransform for the eyedropper. */
    private fun composedTransform(framebufferTransform: FloatArray): FloatArray {
        // Snapshot for the eyedropper's view-px → buffer-px mapping.
        if (lastFramebufferTransform == null ||
            !lastFramebufferTransform.contentEqualsLocal(framebufferTransform)) {
            lastFramebufferTransform = framebufferTransform.copyOf()
        }
        fillDocToViewMatrix(tmpDocToView)
        Matrix.multiplyMM(tmpComposed, 0,
            framebufferTransform, 0,
            tmpDocToView, 0)
        return tmpComposed
    }

    private fun FloatArray?.contentEqualsLocal(other: FloatArray): Boolean {
        val a = this ?: return false
        if (a.size != other.size) return false
        for (i in a.indices) if (a[i] != other[i]) return false
        return true
    }

    /** Visible canvas inset in view-px (typically the right edge of the
     *  panels overlay). MainActivity updates this whenever the layer +
     *  sidebar columns resize so resetView's fit math accounts for the
     *  obscured strip on the left. Zero = no inset (full SurfaceView is
     *  visible to the user). */
    var visibleLeftInset: Int = 0

    /** Frame the page rect in the visible canvas area. If page bounds
     *  aren't set we fall back to identity (the doc behaves as an
     *  infinite plane in that mode and there's no natural "frame"). */
    fun resetView() {
        val pageW = NativeRenderer.getPageWidth()
        val pageH = NativeRenderer.getPageHeight()
        viewRotation = 0f
        if (pageW > 0 && pageH > 0 && width > 0 && height > 0) {
            // Visible canvas region after subtracting the left-side
            // overlay (sidebar + layer panel). Page is fitted edge-to-
            // edge — the dominant axis hits the visible bounds exactly,
            // and the orthogonal axis centers the leftover slack.
            val avail  = (width - visibleLeftInset).toFloat()
            val availH = height.toFloat()
            val sx = avail  / pageW.toFloat()
            val sy = availH / pageH.toFloat()
            val s  = minOf(sx, sy).coerceAtLeast(0.01f)
            viewScale = s
            // Center the scaled page within the visible region. The
            // doc→view mapping is view = scale*doc + viewPan, so the
            // page top-left lands at (viewPanX, viewPanY) in view-px.
            viewPanX = visibleLeftInset + (avail  - s * pageW) * 0.5f
            viewPanY = (availH - s * pageH) * 0.5f
        } else {
            viewScale = 1f
            viewPanX = 0f
            viewPanY = 0f
        }
        NativeRenderer.setViewScale(viewScale)
        forceRedraw()
        onViewTransformChanged?.invoke()
    }

    // Shape-tool drag state (doc-pixels, post-snap). Shared by Line,
    // Rectangle, Circle, Ellipse — only the interpretation differs.
    private var shapeP0X = 0f
    private var shapeP0Y = 0f
    private var shapeP1X = 0f
    private var shapeP1Y = 0f

    private var currentTool = Tool.BRUSH

    // Eyedropper. Single-shot: once `eyedropperPending` is true, the next
    // ACTION_DOWN inside this view samples the pixel under the touch and
    // fires `onColorSampled` with the resulting RGB. The flag clears
    // automatically after the sample completes (or fails).
    var eyedropperPending: Boolean = false
    var onColorSampled: ((rgb: Int) -> Unit)? = null
    private val tmpBufferPx = FloatArray(2)

    /** Notified after the active tool changes (from the UI button or
     *  the stylus side-button). MainActivity uses this to refresh the
     *  on-screen tool button label. */
    var onToolChanged: ((Tool) -> Unit)? = null

    // ---- Text tool hooks (see TextEditController) ----------------------
    /** TEXT tool released on empty canvas: (docX, docY, width, autoWidth). */
    var onTextPlaceRequested: ((Float, Float, Float, Boolean) -> Unit)? = null
    /** TEXT tool tapped an existing box: its id. */
    var onTextEditRequested: ((Int) -> Unit)? = null
    /** The last composite found a text box with no / stale raster. */
    var onTextRasterNeeded: (() -> Unit)? = null
    /** View scale / rotation / pan changed (gesture or reset). */
    var onViewTransformChanged: (() -> Unit)? = null
    /** Any single-pointer DOWN reaching the tool dispatch. Return true
     *  to consume the whole gesture (DOWN through UP): the text editor
     *  does this when a tap outside the open box closes it, so the
     *  same tap doesn't also start a new box — like Esc on a desktop. */
    var onCanvasTouchDown: (() -> Boolean)? = null
    private var swallowGestureUntilUp = false

    /** Vector selection may have changed (tap / marquee / drag end). */
    var onSelectionChanged: (() -> Unit)? = null

    val currentViewRotation: Float get() = viewRotation
    val isGestureActive: Boolean get() = gestureActive

    /** Doc-px point at the centre of the visible canvas (excluding the
     *  panel overlay on the left). */
    fun viewCenterDoc(): Pair<Float, Float> {
        val vx = visibleLeftInset + (width - visibleLeftInset) * 0.5f
        val vy = height * 0.5f
        val out = FloatArray(2)
        viewToDoc(vx, vy, out)
        return Pair(out[0], out[1])
    }

    /** Convert a doc-pixel coordinate to view pixels (inverse of viewToDoc). */
    fun docToView(dx: Float, dy: Float, out: FloatArray) {
        val c = cos(viewRotation)
        val s = sin(viewRotation)
        out[0] = viewPanX + viewScale * (c * dx - s * dy)
        out[1] = viewPanY + viewScale * (s * dx + c * dy)
    }

    /** Snapshot the active floating raster selection's pixels into the
     *  native clipboard. Selection is left unchanged. Queued onto the
     *  next multi-buffer pass since the snapshot uses a temp FBO +
     *  glReadPixels. No-op (silently) if no selection is active. */
    fun queueCopySelection() {
        pendingCopySel = true
        forceRedraw()
    }

    /** Cut: snapshot the active floating raster selection into the
     *  clipboard AND discard the floating selection without restoring
     *  its lifted tiles. The source layer keeps the hole. Pair with
     *  queuePasteSelection to drop the content elsewhere. */
    fun queueCutSelection() {
        pendingCutSel = true
        forceRedraw()
    }

    /** Delete the active selection (raster floating selection OR vector
     *  shape(s)). For raster, this discards the lifted pixels — the
     *  source layer keeps the hole punched out at lift time. For
     *  vectors, this erases the shape(s) from their layer. Queued onto
     *  the next multi-buffer pass since the raster path needs a GL
     *  context (texture delete + glReadPixels for the redo snapshot).
     *  No-op (silently) if nothing is selected. */
    fun queueDeleteSelection() {
        pendingDeleteSel = true
        forceRedraw()
    }

    /** Drop the clipboard's content. Switches to SELECT_RECT (the
     *  unified marquee/select tool) so the user can immediately
     *  interact with the pasted content. The tool dispatches by active
     *  layer type, so the same selection on a vector layer does
     *  tap-to-select / marquee multi-select, and on a raster layer
     *  drives the floating-selection handles. Auto-commits any
     *  existing floating raster sel first. Queued onto the next multi-
     *  buffer pass since paste needs GL state. Empty-clipboard taps
     *  fall through as a no-op. */
    fun queuePasteSelection() {
        if (NativeRenderer.getClipboardKind() != 0) {
            if (currentTool != Tool.SELECT_RECT
                && currentTool != Tool.SELECT_LASSO) {
                currentTool = Tool.SELECT_RECT
                onToolChanged?.invoke(currentTool)
            }
        }
        pendingPasteSel = true
        forceRedraw()
    }

    fun currentToolIs(tool: Tool): Boolean = currentTool == tool

    /** Switch directly to `tool`. Used by the tool-rail buttons. Same
     *  auto-commit-on-switch behavior as toggleTool, and notifies
     *  onToolChanged so the UI mirror updates. No-op if already on it. */
    fun setTool(tool: Tool) {
        if (currentTool == tool) return
        if (NativeRenderer.hasRasterSelection()) {
            pendingCommitRasterSel = true
            forceRedraw()
        }
        currentTool = tool
        if (currentTool.isRasterStroke) {
            NativeRenderer.setTool(currentTool.nativeId)
        }
        Log.i("DrawingApp", "tool -> ${currentTool.name.lowercase()}")
        onToolChanged?.invoke(currentTool)
    }

    /** Public so the stylus side-button (and the matching key-event path
     *  in MainActivity) can route through the same code. Toggles the
     *  brush/eraser pair: brush → eraser, eraser → brush, anything else
     *  → brush. The pair are the only two raster stroke tools, so this
     *  is the most common in-flow swap. */
    fun toggleTool() {
        // Repurpose the middle stylus button as the angle-snap toggle
        // whenever brush/eraser swap doesn't apply: either the active
        // layer is vector (raster brush/eraser don't paint there
        // anyway), or the current tool is a shape tool (where the
        // angle constraint is what the user is most likely after).
        // Both checks together cover "LINE on a raster layer" — a
        // valid setup that previously fell through to the brush
        // swap and got blocked by the mid-stroke guard below.
        val activeIsVector =
            NativeRenderer.getLayerType(NativeRenderer.getActiveLayer()) == 1
        val isShapeTool = currentTool == Tool.LINE
                       || currentTool == Tool.RECTANGLE
                       || currentTool == Tool.CIRCLE
                       || currentTool == Tool.ELLIPSE
        if (activeIsVector || isShapeTool) {
            angleSnapEnabled = !angleSnapEnabled
            Log.i("DrawingApp",
                  "angle snap -> ${if (angleSnapEnabled) "on" else "off"}")
            onAngleSnapChanged?.invoke(angleSnapEnabled)
            return
        }
        // Auto-commit any floating raster selection before swapping tools
        // so the user doesn't lose their work or end up with a stranded
        // floating overlay belonging to a tool they can no longer interact
        // with. Queued because commit needs a live GL context.
        if (NativeRenderer.hasRasterSelection()) {
            pendingCommitRasterSel = true
            forceRedraw()
        }
        val nextTool = when (currentTool) {
            Tool.BRUSH  -> Tool.ERASER
            Tool.ERASER -> Tool.BRUSH
            // Shade drives the same raster stroke pipeline, so swapping
            // to the brush is safe even mid-stroke (the in-flight
            // StrokeAction keeps its type).
            Tool.SHADE  -> Tool.BRUSH
            else        -> {
                // Cross-family swap (e.g. LINE → BRUSH) mid-stroke is
                // unsafe — the in-flight StrokeAction batch on the
                // front-buffered renderer changes type from
                // ShapePreview to Sample, and the next multi-buffer
                // commit ends up baking the abandoned remnants into a
                // black canvas (see logs from Wacom MovinkPad first
                // press after launch). If the pen is in contact, skip
                // the swap; the user can lift the pen and try again.
                if (penInContact) return
                Tool.BRUSH
            }
        }
        currentTool = nextTool
        // Both sides of the toggle are raster stroke tools — push the
        // ID to native so the bake path uses the right blend mode.
        NativeRenderer.setTool(currentTool.nativeId)
        Log.i("DrawingApp", "tool -> ${currentTool.name.lowercase()}")
        onToolChanged?.invoke(currentTool)
    }

    // Last-seen stylus button bitmask, used to detect press transitions
    // even on devices/states that don't fire ACTION_BUTTON_PRESS (notably
    // Wacom EMR while the pen is hovering — buttonState is reported on
    // hover events too, but the discrete press action isn't always).
    private var prevButtonState = 0
    // Until we've seen at least one stylus event, prevButtonState is a
    // synthetic 0; the FIRST event would otherwise look like every set
    // bit was "newly pressed", which can mis-fire one of the button
    // handlers (e.g. on launch, the closest-button press registers as
    // a STYLUS_SECONDARY transition because the OS reports both bits
    // briefly). Skip the transition logic until we've established a
    // baseline.
    private var prevButtonStateSeen = false
    // True from ACTION_DOWN to ACTION_UP/CANCEL of the pen. Used to
    // suppress the brush↔eraser tool swap when the user accidentally
    // hits the middle stylus button mid-stroke from a non-stroke tool
    // like LINE — that transition switches the StrokeAction type
    // (ShapePreview → Sample) inside the same multi-buffer batch and
    // corrupts the next commit. Same-family swaps (BRUSH↔ERASER) stay
    // safe because they share the Sample StrokeAction.
    private var penInContact = false

    /**
     * Detect stylus side-button presses two ways:
     *   1. ACTION_BUTTON_PRESS with actionButton == BUTTON_STYLUS_SECONDARY
     *      (the standard path; fires while pen is in contact).
     *   2. A 0→1 transition of the BUTTON_STYLUS_SECONDARY bit in
     *      event.buttonState (catches presses during hover that don't
     *      generate ACTION_BUTTON_PRESS).
     */
    /** Fired when the stylus's furthest-from-nib button is pressed.
     *  MainActivity wires this to userUndo so the layer panel sync
     *  fires alongside the native undo (matches the on-screen chip). */
    var onUndoRequested: (() -> Unit)? = null

    /** Fired when the stylus's closest-to-nib button is pressed.
     *  Toggles snapping on/off — handy mid-stroke when starting a
     *  vector action snapped but wanting to finish it free-hand. */
    var onSnapToggleRequested: (() -> Unit)? = null

    /** Constrain a freshly-drawn LINE-tool stroke's angle to a multiple
     *  of 15°, AND lock rotation/Line-endpoint drags to the same
     *  step. Off by default. Toggleable from the status bar and from
     *  the stylus middle button when a vector layer or shape tool is
     *  active. The setter mirrors the new value into native via
     *  setAngleSnapEnabled so applyRotateTo / applyScaleTo can read
     *  it directly on the GL thread. */
    var angleSnapEnabled: Boolean = false
        set(value) {
            field = value
            NativeRenderer.setAngleSnapEnabled(value)
        }
    /** Notified after toggleTool flips angleSnapEnabled, so the UI
     *  status indicator can re-paint. */
    var onAngleSnapChanged: ((Boolean) -> Unit)? = null

    /** Fires while a line-tool stroke is in progress AND angle snap
     *  is on, carrying the line's currently-snapped angle in degrees
     *  (0°=right, 90°=up, 180°=left, 270°=down). Null on UP/CANCEL or
     *  whenever angle snap is off. The status bar uses it to show
     *  e.g. "angle: 45°" so the user knows what they're locking to. */
    var onLineAngleChanged: ((Float?) -> Unit)? = null

    /** Fired on every stylus event so an overlay (e.g. brush preview)
     *  can track the pen. xy in view-px, hovering=true for hover-only
     *  events (pen near surface, not in contact). Null = pen has left
     *  the surface and any overlay should hide. */
    var onPenPosition: ((x: Float, y: Float, hovering: Boolean) -> Unit)? = null
    var onPenLeft: (() -> Unit)? = null

    /** Pen-pressure saturation point in (0, 1]. Raw pen pressure is
     *  divided by this and clamped to [0, 1] before being baked into a
     *  dab's radius/alpha — so 1.0 leaves the full pen range intact,
     *  0.5 makes the pen "max out" at half its raw range, etc. 0.0 is
     *  treated as "always full pressure" (pressure-insensitive mode).
     *  Set from MainActivity's brush-press slider. */
    var brushPressureSaturation: Float = 1.0f

    private fun mapPressure(raw: Float): Float {
        val sat = brushPressureSaturation
        if (sat <= 0f) return 1f
        val v = raw / sat
        return when {
            v < 0f -> 0f
            v > 1f -> 1f
            else   -> v
        }
    }

    private fun handleStylusButton(event: MotionEvent): Boolean {
        if (event.actionMasked == MotionEvent.ACTION_BUTTON_PRESS) {
            val ab = event.actionButton
            if (ab == MotionEvent.BUTTON_STYLUS_SECONDARY) {
                toggleTool()
                prevButtonState = event.buttonState
                prevButtonStateSeen = true
                return true
            }
            if (ab == MotionEvent.BUTTON_TERTIARY) {
                // MovinkPad's furthest-from-nib button reports via the
                // legacy BUTTON_TERTIARY bit. ACTION_BUTTON_PRESS and
                // the state-transition path below both cover it.
                onUndoRequested?.invoke()
                prevButtonState = event.buttonState
                prevButtonStateSeen = true
                return true
            }
            // Closest-to-nib button — covered both standard
            // BUTTON_STYLUS_PRIMARY (0x20) and the legacy BUTTON_SECONDARY
            // (0x2) aliases, since EMR devices report it inconsistently.
            if (ab == MotionEvent.BUTTON_STYLUS_PRIMARY
                || ab == MotionEvent.BUTTON_SECONDARY) {
                onSnapToggleRequested?.invoke()
                prevButtonState = event.buttonState
                prevButtonStateSeen = true
                return true
            }
        }
        val state = event.buttonState
        if (!prevButtonStateSeen) {
            // First stylus event — establish a baseline without firing
            // any handlers. See prevButtonStateSeen's comment above.
            prevButtonState = state
            prevButtonStateSeen = true
            return false
        }
        val newlyPressed = state and prevButtonState.inv()
        prevButtonState = state
        // Diagnostic: unmapped newly-pressed bits get logged so future
        // unmapped buttons surface in logcat without code changes.
        val mapped = MotionEvent.BUTTON_STYLUS_SECONDARY or
                     MotionEvent.BUTTON_TERTIARY or
                     MotionEvent.BUTTON_STYLUS_PRIMARY or
                     MotionEvent.BUTTON_SECONDARY or
                     MotionEvent.BUTTON_PRIMARY  // pen tip; expected
        val unmapped = newlyPressed and mapped.inv()
        if (unmapped != 0) {
            Log.i("DrawingApp",
                "stylus button: unmapped newly=0x${unmapped.toString(16)} " +
                "state=0x${state.toString(16)}")
        }
        if (newlyPressed and MotionEvent.BUTTON_STYLUS_SECONDARY != 0) {
            toggleTool()
            return true
        }
        if (newlyPressed and MotionEvent.BUTTON_TERTIARY != 0) {
            onUndoRequested?.invoke()
            return true
        }
        if (newlyPressed and
            (MotionEvent.BUTTON_STYLUS_PRIMARY
             or MotionEvent.BUTTON_SECONDARY) != 0) {
            onSnapToggleRequested?.invoke()
            return true
        }
        return false
    }

    override fun onGenericMotionEvent(event: MotionEvent): Boolean {
        if (handleStylusButton(event)) return true
        return super.onGenericMotionEvent(event)
    }

    override fun onHoverEvent(event: MotionEvent): Boolean {
        if (handleStylusButton(event)) return true
        // Push the hover position to any overlay (brush preview).
        // EXIT means the pen has lifted clear of the surface, so the
        // overlay should hide.
        when (event.actionMasked) {
            MotionEvent.ACTION_HOVER_ENTER,
            MotionEvent.ACTION_HOVER_MOVE -> {
                onPenPosition?.invoke(event.x, event.y, true)
            }
            MotionEvent.ACTION_HOVER_EXIT -> {
                onPenLeft?.invoke()
            }
        }
        return super.onHoverEvent(event)
    }

    // ---- 2-finger pan/zoom/rotate gesture state ----------------------
    //
    // While a gesture is active, viewScale/Rotation/Pan are recomputed
    // each MOVE so the two anchor doc-points (captured at the moment the
    // 2nd finger touched) stay locked under their respective fingers.
    // suppressTouches stays true after the gesture ends until every finger
    // has lifted, so a finger left from the gesture doesn't accidentally
    // start a stroke.
    private var gestureActive = false
    private var gestureP1Id = -1
    private var gestureP2Id = -1
    private var gestureP1DocX = 0f; private var gestureP1DocY = 0f
    private var gestureP2DocX = 0f; private var gestureP2DocY = 0f
    private var suppressTouches = false

    // Set by the gesture path when an in-progress brush/eraser stroke is
    // discarded; checked in onDrawMultiBufferedLayer to skip the bake.
    private var cancelNextCommit = false

    /** Palm-rejection gate. When true, single-pointer touches whose
     *  tool type isn't STYLUS are swallowed before tool dispatch — i.e.
     *  no finger/palm-driven strokes, taps, or marquee drags reach the
     *  active tool. 2-finger gestures (pan/zoom/rotate) still work
     *  because they have their own pre-dispatch branch. Set from
     *  MainActivity (persisted in SharedPreferences). */
    var stylusOnlyDrawing: Boolean = false

    private fun anyStylusPointer(event: MotionEvent): Boolean {
        for (i in 0 until event.pointerCount) {
            if (event.getToolType(i) == MotionEvent.TOOL_TYPE_STYLUS) return true
        }
        return false
    }

    private fun beginGesture(event: MotionEvent, p1Index: Int, p2Index: Int) {
        // Cancel any in-progress draw / interaction so the user's drag
        // doesn't bleed into a stroke or transform on gesture release.
        when (currentTool) {
            Tool.BRUSH, Tool.ERASER, Tool.SHADE -> {
                NativeRenderer.discardStroke()
                cancelNextCommit = true
                renderer?.commit()
            }
            Tool.BUCKET -> {
                // Click-to-act, nothing in flight to cancel.
            }
            Tool.TEXT -> {
                // Drop the drag-to-place preview; placement needs UP.
                renderer?.commit()
                textDragging = false
            }
            Tool.LINE, Tool.RECTANGLE, Tool.CIRCLE, Tool.ELLIPSE -> {
                renderer?.commit()
                p1Snapped = false
            }
            Tool.SELECT, Tool.SELECT_RECT, Tool.SELECT_LASSO -> {
                // Vector-select side: end any in-flight transform/marquee
                // drag so its mode flag doesn't carry over to the next
                // gesture. Safe even when the active layer is raster
                // (selectMode would just be 0).
                if (selectMode != 0) {
                    NativeRenderer.endInteraction()
                    selectMode = 0
                    selectChanged = false
                }
                // Raster-select side: cancel the marquee preview (rect
                // or polyline) if mid-define; end any handle drag if
                // mid-interact. The floating selection itself survives.
                if (selRectMode == SelRectMode.DEFINE) {
                    lassoPathBuf.clear()
                    renderer?.commit()
                }
                if (selRectMode == SelRectMode.INTERACT) {
                    NativeRenderer.endRasterInteraction()
                }
                selRectMode = SelRectMode.NONE
            }
        }
        gestureP1Id = event.getPointerId(p1Index)
        gestureP2Id = event.getPointerId(p2Index)
        viewToDoc(event.getX(p1Index), event.getY(p1Index), tmpDoc)
        gestureP1DocX = tmpDoc[0]; gestureP1DocY = tmpDoc[1]
        viewToDoc(event.getX(p2Index), event.getY(p2Index), tmpDoc)
        gestureP2DocX = tmpDoc[0]; gestureP2DocY = tmpDoc[1]
        gestureActive   = true
        suppressTouches = true
    }

    private fun updateGesture(event: MotionEvent) {
        val i1 = event.findPointerIndex(gestureP1Id)
        val i2 = event.findPointerIndex(gestureP2Id)
        if (i1 < 0 || i2 < 0) return
        val v1x = event.getX(i1); val v1y = event.getY(i1)
        val v2x = event.getX(i2); val v2y = event.getY(i2)

        // Solve the 2-point similarity problem: find scale, rotation, pan
        // such that view(D1) = V1' and view(D2) = V2', where view(D) =
        // pan + scale * R(rotation) * D.
        val viewDx = v2x - v1x; val viewDy = v2y - v1y
        val docDx  = gestureP2DocX - gestureP1DocX
        val docDy  = gestureP2DocY - gestureP1DocY
        val viewLen = hypot(viewDx, viewDy)
        val docLen  = hypot(docDx,  docDy)
        if (viewLen < 1e-3f || docLen < 1e-3f) return

        // Clamp scale to a sane range. Without this, zooming far out makes
        // any natural finger movement span enormous doc-pixel distances,
        // and the stroke bake then has to materialize a tile FBO for every
        // cell along the way — which can OOM the GPU.
        val rawScale    = (viewLen / docLen).toFloat()
        val newScale    = rawScale.coerceIn(kMinViewScale, kMaxViewScale)
        val newRotation = (atan2(viewDy, viewDx) - atan2(docDy, docDx)).toFloat()
        val c = cos(newRotation); val s = sin(newRotation)
        val newPanX = v1x - newScale * (c * gestureP1DocX - s * gestureP1DocY)
        val newPanY = v1y - newScale * (s * gestureP1DocX + c * gestureP1DocY)

        viewScale    = newScale
        viewRotation = newRotation
        viewPanX     = newPanX
        viewPanY     = newPanY
        NativeRenderer.setViewScale(viewScale)
        forceRedraw()
        onViewTransformChanged?.invoke()
    }

    private companion object {
        /** View-px the pen must travel before a TEXT tap becomes a
         *  drag-to-place. */
        const val kTextDragSlopPx = 14f
        // Limits on the gesture-driven view scale.
        const val kMinViewScale = 0.25f
        const val kMaxViewScale = 8.0f
        // Trailing-edge debounce window for the deferred thumbnail refresh.
        const val kThumbnailDeferMs = 250L
    }

    private fun endGesture() {
        gestureActive = false
        gestureP1Id = -1
        gestureP2Id = -1
        // suppressTouches stays true until the last finger lifts.
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (handleStylusButton(event)) return true

        val r = renderer ?: return super.onTouchEvent(event)
        val action = event.actionMasked

        // Track pen-down state for toggleTool's mid-stroke guard. We
        // only flip this on stylus events so finger touches don't
        // confuse the gating.
        if (event.getToolType(0) == MotionEvent.TOOL_TYPE_STYLUS) {
            when (action) {
                MotionEvent.ACTION_DOWN   -> penInContact = true
                MotionEvent.ACTION_UP,
                MotionEvent.ACTION_CANCEL -> penInContact = false
            }
        }

        // Push the pen position to any overlay. During contact the pen
        // is "drawing"; on UP/CANCEL it's lifted (any overlay should
        // hide). ACTION_POINTER_* events ignored — the overlay tracks
        // the primary pointer only.
        if (event.getToolType(0) == MotionEvent.TOOL_TYPE_STYLUS) {
            when (action) {
                MotionEvent.ACTION_DOWN,
                MotionEvent.ACTION_MOVE -> {
                    onPenPosition?.invoke(event.x, event.y, false)
                }
                MotionEvent.ACTION_UP,
                MotionEvent.ACTION_CANCEL -> {
                    onPenLeft?.invoke()
                }
            }
        }

        // ACTION_DOWN = transition from 0 → 1 pointers on screen, so no
        // prior gesture state should leak through. Reset defensively in
        // case an earlier ACTION_UP / ACTION_POINTER_UP was dropped (palm
        // rejection edge cases can do this); without this, a stuck flag
        // would silently swallow the new stroke.
        if (action == MotionEvent.ACTION_DOWN) {
            gestureActive   = false
            suppressTouches = false
            cancelNextCommit = false
            // Cancel any trailing-edge forceRedraw scheduled by the
            // prior stroke. If it fired mid-stroke it would trigger
            // an MB transition with no samples, briefly hiding the
            // FB-rendered in-progress stroke until the next move
            // sample restored it — visible to the user as a "skip"
            // mid-letter. The drain it would have performed gets
            // picked up on the next post-stroke trailing-edge anyway.
            removeCallbacks(thumbnailRefreshRunnable)
        }

        // While a gesture is active, all events drive the gesture path.
        if (gestureActive) {
            when (action) {
                MotionEvent.ACTION_MOVE -> updateGesture(event)
                MotionEvent.ACTION_POINTER_UP -> {
                    val upId = event.getPointerId(event.actionIndex)
                    if (upId == gestureP1Id || upId == gestureP2Id) endGesture()
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    endGesture()
                    suppressTouches = false
                }
            }
            return true
        }

        // After gesture, swallow remaining-finger events until all fingers
        // have lifted. Otherwise the trailing finger would start a stroke.
        if (suppressTouches) {
            if (action == MotionEvent.ACTION_UP || action == MotionEvent.ACTION_CANCEL) {
                suppressTouches = false
            }
            return true
        }

        // 2nd finger down (with no stylus involved) starts a gesture.
        if (action == MotionEvent.ACTION_POINTER_DOWN
            && event.pointerCount == 2
            && !anyStylusPointer(event)) {
            beginGesture(event, 0, 1)
            return true
        }

        // Single-pointer path: dispatch to the active tool.
        val tt = event.getToolType(0)
        if (tt != MotionEvent.TOOL_TYPE_STYLUS && tt != MotionEvent.TOOL_TYPE_FINGER) {
            return super.onTouchEvent(event)
        }
        // Palm-rejection gate: in stylus-only mode, finger/palm contacts
        // are consumed but not dispatched. Stylus events fall through.
        if (stylusOnlyDrawing && tt != MotionEvent.TOOL_TYPE_STYLUS) {
            return true
        }
        // Eyedropper takes precedence over the active tool. While the
        // mode is armed, every touch event is swallowed — the FIRST DOWN
        // fires the sample, MOVE / extra DOWNs do nothing, UP / CANCEL
        // disarms. Critically we DO NOT disarm before UP, otherwise the
        // trailing MOVE+UP fall through to handleStrokeEvent and a dab
        // gets painted at the touch point (which then becomes whatever
        // glReadPixels reads back, defeating the eyedropper).
        if (eyedropperPending) {
            when (action) {
                MotionEvent.ACTION_DOWN -> handleEyedropperTap(event.x, event.y)
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    eyedropperPending = false
                }
            }
            return true
        }
        if (swallowGestureUntilUp) {
            if (action == MotionEvent.ACTION_UP || action == MotionEvent.ACTION_CANCEL) {
                swallowGestureUntilUp = false
            }
            return true
        }
        if (action == MotionEvent.ACTION_DOWN && onCanvasTouchDown?.invoke() == true) {
            swallowGestureUntilUp = true
            return true
        }
        return when (currentTool) {
            Tool.BRUSH, Tool.ERASER, Tool.SHADE -> handleStrokeEvent(r, event)
            Tool.BUCKET             -> handleBucketEvent(event)
            Tool.TEXT               -> handleTextEvent(r, event)
            Tool.SELECT             -> handleSelectEvent(event)
            // The marquee/rectangle selection tool dispatches by active
            // layer type: raster → lift pixels into a floating raster
            // selection; vector → tap-to-select / marquee multi-select
            // shapes. One icon, two behaviors — eliminates the prior
            // "vector select" rail entry.
            Tool.SELECT_RECT        -> {
                val activeIsVector = NativeRenderer.getLayerType(
                    NativeRenderer.getActiveLayer()) == 1
                if (activeIsVector) handleSelectEvent(event)
                else                handleSelectRectEvent(r, event)
            }
            Tool.SELECT_LASSO       -> handleSelectLassoEvent(r, event)
            else                    -> handleShapeEvent(r, event, currentTool.shapeType)
        }
    }

    // ---- Raster selection (rectangle marquee) ------------------------
    //
    // Rectangle marquee gesture mode.
    //   NONE     : not currently dragging
    //   DEFINE   : rubber-banding the marquee that will become the lift rect
    //   INTERACT : driving an in-progress move/scale/rotate of the floating
    //              selection (mode chosen by beginRasterInteractionAt at
    //              ACTION_DOWN — body=move, corner=scale, top=rotate).
    //              ACTION_DOWN that misses both the body and any handle
    //              commits the existing selection and falls through to
    //              DEFINE in the same gesture.
    private enum class SelRectMode { NONE, DEFINE, INTERACT }
    private var selRectMode = SelRectMode.NONE
    // Live preview of the rect being defined, in doc-px (only valid while
    // selRectMode == DEFINE).
    private var selRectPreviewX0 = 0f
    private var selRectPreviewY0 = 0f
    private var selRectPreviewX1 = 0f
    private var selRectPreviewY1 = 0f

    // ---- INTERACT drag slop -----------------------------------------
    //
    // A tap must not transform the selection. The corner handles have a
    // hit radius that extends OUTSIDE the selection body, so a tap just
    // past a corner registers as a scale grab; without a movement
    // threshold, a single pixel of pen jitter between DOWN and UP is
    // then enough to resize. Combined with the native-side grab offset
    // (DragState.grabOffsetX/Y) this makes a handle drag start exactly
    // where the pen is and only move by the pen's own delta.
    //
    // Tracked in view-px so the threshold is a constant physical
    // distance regardless of zoom.
    private var interactDownX = 0f
    private var interactDownY = 0f
    private var interactMoved = false
    // Did the ACTION_DOWN land inside the selection body, as opposed to
    // merely inside a handle's hit radius (which reaches outside it)?
    // Decides what an unmoved tap means on ACTION_UP.
    private var interactInsideBody = false
    // Mode beginRasterInteractionAt chose: 1 = move, 2 = scale, 3 = rotate.
    private var interactHit = 0
    private val kInteractSlopViewPx = 6f

    /** Shared ACTION_DOWN bookkeeping for a rect/lasso INTERACT drag. */
    private fun beginSelInteract(
        event: MotionEvent, docX: Float, docY: Float, hit: Int
    ) {
        selRectMode = SelRectMode.INTERACT
        interactDownX = event.x
        interactDownY = event.y
        interactMoved = false
        interactHit = hit
        interactInsideBody = NativeRenderer.rasterSelectionContains(docX, docY)
    }

    /** Drive an in-progress INTERACT drag, holding still until the pen
     *  has travelled past the slop threshold. */
    private fun updateSelInteract(event: MotionEvent, docX: Float, docY: Float) {
        if (!interactMoved) {
            val ddx = event.x - interactDownX
            val ddy = event.y - interactDownY
            if (ddx * ddx + ddy * ddy
                < kInteractSlopViewPx * kInteractSlopViewPx) return
            interactMoved = true
        }
        NativeRenderer.updateRasterInteractionAt(docX, docY)
        forceRedraw()
    }

    /** Finish an INTERACT drag. An unmoved tap that landed outside the
     *  selection body but got claimed by a CORNER handle was the user
     *  reaching for "tap outside to finalize" and merely clipping that
     *  handle's hit radius — honour it and commit, matching what a
     *  cleaner miss (hit == 0) would have done.
     *
     *  Deliberately scoped to corner handles (hit == 2). Their radius is
     *  centred on the corners themselves, so half of it spills into the
     *  space just outside the selection — which is exactly where a
     *  dismissing tap lands. The rotate handle is a distinct affordance
     *  floating above the top edge; a tap there stays inert rather than
     *  silently dropping the selection. */
    private fun endSelInteract(r: GLFrontBufferedRenderer<StrokeAction>) {
        NativeRenderer.endRasterInteraction()
        if (!interactMoved && !interactInsideBody && interactHit == 2) {
            pendingCommitRasterSel = true
            r.commit()
        }
    }

    private fun handleSelectRectEvent(
        r: GLFrontBufferedRenderer<StrokeAction>,
        event: MotionEvent
    ): Boolean {
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                viewToDoc(event.x, event.y, tmpDoc)
                val dx = tmpDoc[0]; val dy = tmpDoc[1]
                // Try a handle / body hit on the active floating selection
                // first. Returns 1 (move), 2 (scale), or 3 (rotate) on hit;
                // 0 means no active selection OR the tap fell outside it.
                val hit = NativeRenderer.beginRasterInteractionAt(dx, dy)
                if (hit != 0) {
                    beginSelInteract(event, dx, dy, hit)
                    forceRedraw()
                } else {
                    // Tap missed the selection (or there is none). Commit
                    // any existing selection so it bakes before a new lift.
                    if (NativeRenderer.hasRasterSelection()) {
                        pendingCommitRasterSel = true
                    }
                    selRectMode = SelRectMode.DEFINE
                    selRectPreviewX0 = dx; selRectPreviewY0 = dy
                    selRectPreviewX1 = dx; selRectPreviewY1 = dy
                    forceRedraw()
                }
            }
            MotionEvent.ACTION_MOVE -> {
                viewToDoc(event.x, event.y, tmpDoc)
                val dx = tmpDoc[0]; val dy = tmpDoc[1]
                when (selRectMode) {
                    SelRectMode.DEFINE -> {
                        selRectPreviewX1 = dx
                        selRectPreviewY1 = dy
                        // Render the marquee outline through the same
                        // thin-black path the lasso uses, so the
                        // selection chrome reads as system UI rather
                        // than user content (it shouldn't pick up the
                        // brush color or the vector-tool line width).
                        // Construct a 4-point rectangle as a closed
                        // polyline.
                        val pts = floatArrayOf(
                            selRectPreviewX0, selRectPreviewY0,
                            selRectPreviewX1, selRectPreviewY0,
                            selRectPreviewX1, selRectPreviewY1,
                            selRectPreviewX0, selRectPreviewY1
                        )
                        r.renderFrontBufferedLayer(
                            StrokeAction.LassoPreview(pts, closed = true)
                        )
                    }
                    SelRectMode.INTERACT -> updateSelInteract(event, dx, dy)
                    SelRectMode.NONE -> {}
                }
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                when (selRectMode) {
                    SelRectMode.DEFINE -> {
                        // Queue the lift; r.commit() below clears the
                        // preview rectangle and triggers the multi-buffer
                        // pass that drains the queued operations on the
                        // GL thread (where they actually have a context).
                        pendingBeginRasterSel = PendingBeginRasterSel(
                            selRectPreviewX0, selRectPreviewY0,
                            selRectPreviewX1, selRectPreviewY1
                        )
                        r.commit()
                    }
                    SelRectMode.INTERACT -> {
                        // Selection stays floating unless this was a tap
                        // outside it; user can drag again or tap outside
                        // to commit.
                        endSelInteract(r)
                    }
                    SelRectMode.NONE -> {}
                }
                selRectMode = SelRectMode.NONE
            }
        }
        return true
    }

    // Live polyline buffer for the lasso DEFINE phase. Cleared at each
    // ACTION_DOWN / ACTION_UP. Held as a flat list to avoid allocating
    // a Pair per sample; converted to a FloatArray for the native call.
    private val lassoPathBuf = ArrayList<Float>(64)
    // Throttle: only append a new point if at least this many doc-px from
    // the last one. Keeps the per-MOVE polyline render cost bounded on
    // long, dense gestures.
    private val lassoMinSpacingDoc: Float
        get() = 1.0f / viewScale

    private fun handleSelectLassoEvent(
        r: GLFrontBufferedRenderer<StrokeAction>,
        event: MotionEvent
    ): Boolean {
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                viewToDoc(event.x, event.y, tmpDoc)
                val dx = tmpDoc[0]; val dy = tmpDoc[1]
                // Same handle/body hit-test as the rect tool: any active
                // floating selection takes priority over starting a new
                // lasso path.
                val hit = NativeRenderer.beginRasterInteractionAt(dx, dy)
                if (hit != 0) {
                    beginSelInteract(event, dx, dy, hit)
                    forceRedraw()
                } else {
                    if (NativeRenderer.hasRasterSelection()) {
                        pendingCommitRasterSel = true
                    }
                    selRectMode = SelRectMode.DEFINE
                    lassoPathBuf.clear()
                    lassoPathBuf.add(dx); lassoPathBuf.add(dy)
                    forceRedraw()
                }
            }
            MotionEvent.ACTION_MOVE -> {
                viewToDoc(event.x, event.y, tmpDoc)
                val dx = tmpDoc[0]; val dy = tmpDoc[1]
                when (selRectMode) {
                    SelRectMode.DEFINE -> {
                        val n = lassoPathBuf.size
                        val lastX = lassoPathBuf[n - 2]
                        val lastY = lassoPathBuf[n - 1]
                        val ddx = dx - lastX; val ddy = dy - lastY
                        val minSp = lassoMinSpacingDoc
                        if (ddx * ddx + ddy * ddy >= minSp * minSp) {
                            lassoPathBuf.add(dx); lassoPathBuf.add(dy)
                        }
                        // Re-render the entire path; the front-buffer
                        // shader clears first so this isn't additive.
                        r.renderFrontBufferedLayer(
                            StrokeAction.LassoPreview(
                                points = lassoPathBuf.toFloatArray(),
                                closed = false
                            )
                        )
                    }
                    SelRectMode.INTERACT -> updateSelInteract(event, dx, dy)
                    SelRectMode.NONE -> {}
                }
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                when (selRectMode) {
                    SelRectMode.DEFINE -> {
                        // Need at least 3 distinct points to form a polygon.
                        if (lassoPathBuf.size >= 6) {
                            pendingBeginLassoSel = lassoPathBuf.toFloatArray()
                        }
                        lassoPathBuf.clear()
                        // r.commit() clears the front-buffered preview and
                        // drives the multi-buffer pass that drains the
                        // lasso lift queue on the GL thread.
                        r.commit()
                    }
                    SelRectMode.INTERACT -> endSelInteract(r)
                    SelRectMode.NONE -> {}
                }
                selRectMode = SelRectMode.NONE
            }
        }
        return true
    }

    /** Bucket tool: tap-to-fill. The native fill needs a current GL
     *  context (it runs a full-page composite, glReadPixels, etc.), and
     *  no GL context is current on the UI thread — so we just stash the
     *  request and trigger a multi-buffer pass. The pass's GL-thread
     *  callback (onDrawMultiBufferedLayer) actually runs the fill. */

    /** Eyedropper: convert the view-px touch to buffer-px via the cached
     *  framework view→buffer matrix, queue a sample request to native,
     *  then poll for the result one frame later. The cached transform is
     *  identity on a non-rotated SurfaceView (the MovinkPad's case), but
     *  applying it makes us future-proof against orientation changes. */
    private fun handleEyedropperTap(viewX: Float, viewY: Float) {
        // The caller (onTouchEvent) keeps eyedropperPending=true through
        // the rest of the gesture; clearing happens on UP/CANCEL. Don't
        // touch the flag here.
        // The native sampler reads tile FBOs directly, so we want
        // doc-space coordinates. viewToDoc handles scale/rotation/pan.
        viewToDoc(viewX, viewY, tmpDoc)
        NativeRenderer.requestColorSample(tmpDoc[0], tmpDoc[1])
        // Trigger a multi-buffer pass so the GL thread actually performs
        // the sample. requestColorSample already cleared the prior result.
        forceRedraw()

        // Poll the result a couple of frames later — at 90 Hz a single
        // frame is ~11ms, but the framework can defer the multi-buffer
        // pass slightly. ~60ms is generous and still feels instant.
        val cb = onColorSampled
        postDelayed({
            val rgb = NativeRenderer.getLastSampledColor()
            if (rgb >= 0 && cb != null) cb(rgb)
        }, 60L)
    }

    private fun handleBucketEvent(event: MotionEvent): Boolean {
        if (event.actionMasked == MotionEvent.ACTION_DOWN) {
            viewToDoc(event.x, event.y, tmpDoc)
            pendingBucketFill = PendingBucketFill(tmpDoc[0], tmpDoc[1])
            forceRedraw()
        }
        return true
    }

    private data class PendingBucketFill(val x: Float, val y: Float)
    // Single volatile reference packages both coords + request flag — UI
    // thread writes a fully-constructed object, GL thread reads it whole.
    @Volatile
    private var pendingBucketFill: PendingBucketFill? = null

    // Raster selection lift / commit / cancel — same pattern as bucket
    // fill. The native impls all need a current GL context (they run
    // glCopyTexSubImage2D, scissor-clear, snapshot read-backs, etc.) so
    // they can't run from the touch-handler thread.
    private data class PendingBeginRasterSel(
        val x0: Float, val y0: Float, val x1: Float, val y1: Float
    )
    @Volatile
    private var pendingBeginRasterSel: PendingBeginRasterSel? = null
    @Volatile
    private var pendingCommitRasterSel = false
    @Volatile
    private var pendingCancelRasterSel = false
    // Lasso lift queue: a flat [x0,y0,x1,y1,...] doc-coord polyline.
    @Volatile
    private var pendingBeginLassoSel: FloatArray? = null
    @Volatile
    private var pendingCopySel = false
    private var pendingCutSel  = false
    @Volatile
    private var pendingDeleteSel = false
    @Volatile
    private var pendingPasteSel = false

    // Drag state for the SELECT tool.
    //   0=none, 1=move, 2=scale, 3=rotate, 4=marquee multi-select.
    private var selectMode = 0
    private var selectChanged = false   // true if we should persist on UP

    // Text box that was the primary selection when the SELECT-tool
    // DOWN landed. A tap (no drag) on it again opens the editor —
    // first tap selects, second tap edits, like a slide deck.
    private var selectDownTextId = 0

    private fun handleSelectEvent(event: MotionEvent): Boolean {
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                selectDownTextId = NativeRenderer.getSelectedTextBoxId()
                viewToDoc(event.x, event.y, tmpDoc)
                selectMode = NativeRenderer.beginInteractionAt(tmpDoc[0], tmpDoc[1])
                selectChanged = false
                forceRedraw()
                onSelectionChanged?.invoke()
            }
            MotionEvent.ACTION_MOVE -> {
                viewToDoc(event.x, event.y, tmpDoc)
                when (selectMode) {
                    1 -> { // move — snap-aware absolute, native uses captured offset
                        NativeRenderer.moveSelectionTo(tmpDoc[0], tmpDoc[1])
                        selectChanged = true
                        forceRedraw()
                    }
                    2, 3 -> { // scale / rotate — absolute pen position
                        NativeRenderer.updateInteractionAt(tmpDoc[0], tmpDoc[1])
                        selectChanged = true
                        forceRedraw()
                    }
                    4 -> { // marquee — drag updates rectangle, no shape change
                        NativeRenderer.updateInteractionAt(tmpDoc[0], tmpDoc[1])
                        forceRedraw()
                    }
                }
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                NativeRenderer.endInteraction()
                if (selectChanged) {
                    NativeRenderer.persistActiveVectorLayer()
                }
                // Marquee finalization needs a redraw to clear the
                // rect overlay + paint the new selection halos.
                if (selectMode == 4) forceRedraw()
                val tapOnSelected = selectMode == 1 && !selectChanged
                    && event.actionMasked == MotionEvent.ACTION_UP
                selectMode = 0
                selectChanged = false
                onSelectionChanged?.invoke()
                if (tapOnSelected && selectDownTextId != 0
                    && NativeRenderer.getSelectedTextBoxId() == selectDownTextId) {
                    onTextEditRequested?.invoke(selectDownTextId)
                }
                selectDownTextId = 0
            }
        }
        return true
    }

    private fun handleStrokeEvent(
        r: GLFrontBufferedRenderer<StrokeAction>,
        event: MotionEvent
    ): Boolean {
        // Sensor → handler delta is computed against the ms-resolution
        // event.eventTime (same epoch as SystemClock.uptimeMillis).
        // recvNs is the System.nanoTime() at this dispatch and is used
        // by the GL thread to compute app-prep delta.
        val recvNs = System.nanoTime()
        val recvMs = SystemClock.uptimeMillis()

        // Motion predictor lifecycle: lazy init on first event, fed
        // every real touch (DOWN + MOVE) so it can build a velocity
        // model. We don't tear it down between strokes — it self-resets
        // when the gap between events exceeds its internal threshold.
        if (predictionEnabled && motionPredictor == null) {
            motionPredictor = MotionEventPredictor.newInstance(this)
        }
        motionPredictor?.record(event)

        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                viewToDoc(event.x, event.y, tmpDoc)
                val xyp = floatArrayOf(
                    tmpDoc[0], tmpDoc[1], mapPressure(event.pressure)
                )
                r.renderFrontBufferedLayer(
                    StrokeAction.BatchSamples(
                        xyp, realCount = 1, isNewStroke = true,
                        inputAgeMs = recvMs - event.eventTime,
                        receivedNs = recvNs,
                    )
                )
                // No prediction on DOWN — the predictor needs at least
                // one move's worth of history before its output is
                // useful, and predicting from a single point can
                // overshoot wildly.
            }
            MotionEvent.ACTION_MOVE -> {
                val realN = event.historySize + 1
                // Pull predicted samples first so we know the final array
                // size up front. dispatchPredictedSamples() returns the
                // predictor's MotionEvent (caller-recycled) or null if
                // prediction is off / unavailable.
                val pe = pullPrediction()
                val predN = pe?.let { it.historySize + 1 } ?: 0
                val xyp = FloatArray((realN + predN) * 3)

                // Real region: historical (oldest first) then current.
                for (i in 0 until event.historySize) {
                    viewToDoc(
                        event.getHistoricalX(i), event.getHistoricalY(i), tmpDoc
                    )
                    val k = i * 3
                    xyp[k]     = tmpDoc[0]
                    xyp[k + 1] = tmpDoc[1]
                    xyp[k + 2] = mapPressure(event.getHistoricalPressure(i))
                }
                viewToDoc(event.x, event.y, tmpDoc)
                val curK = event.historySize * 3
                xyp[curK]     = tmpDoc[0]
                xyp[curK + 1] = tmpDoc[1]
                xyp[curK + 2] = mapPressure(event.pressure)

                // Predicted tail (if any). Recycled below in finally.
                if (pe != null) {
                    try {
                        for (i in 0 until pe.historySize) {
                            viewToDoc(
                                pe.getHistoricalX(i), pe.getHistoricalY(i), tmpDoc
                            )
                            val k = (realN + i) * 3
                            xyp[k]     = tmpDoc[0]
                            xyp[k + 1] = tmpDoc[1]
                            xyp[k + 2] = mapPressure(pe.getHistoricalPressure(i))
                        }
                        viewToDoc(pe.x, pe.y, tmpDoc)
                        val k = (realN + pe.historySize) * 3
                        xyp[k]     = tmpDoc[0]
                        xyp[k + 1] = tmpDoc[1]
                        xyp[k + 2] = mapPressure(pe.pressure)
                    } finally {
                        pe.recycle()
                    }
                }

                r.renderFrontBufferedLayer(
                    StrokeAction.BatchSamples(
                        xyp, realCount = realN, isNewStroke = false,
                        inputAgeMs = recvMs - event.eventTime,
                        receivedNs = recvNs,
                    )
                )
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                r.commit()
            }
        }
        return true
    }

    /** Return the predictor's next MotionEvent, or null if prediction is
     *  off / unavailable. Caller must recycle the returned event. */
    private fun pullPrediction(): MotionEvent? {
        if (!predictionEnabled) return null
        return motionPredictor?.predict()
    }

    // Reused output buffer for NativeRenderer.snapPoint — avoids
    // allocating a fresh FloatArray per pen sample.
    private val snapOut = FloatArray(3)
    private var p1Snapped = false

    /** Snap the given doc-space point to the nearest snap target if any.
     *  Updates p1Snapped as a side effect. Inputs and outputs are doc-px. */
    private fun snap(x: Float, y: Float): Pair<Float, Float> {
        NativeRenderer.snapPoint(x, y, snapOut)
        p1Snapped = snapOut[2] > 0.5f
        return if (p1Snapped) Pair(snapOut[0], snapOut[1]) else Pair(x, y)
    }

    /** Snap the LINE-tool endpoint's *direction* from p0 to a multiple
     *  of 15°, preserving distance from p0. Returns (x, y, stepIdx)
     *  where stepIdx is the integer multiple of 15° that was selected
     *  — caller uses it to display the exact snapped angle without
     *  going through an atan2(cos, sin) round-trip that can shave a
     *  degree off due to float precision. stepIdx == Int.MIN_VALUE
     *  means "no snap" (pen too close to p0). */
    private data class AngleSnapResult(val x: Float, val y: Float, val stepIdx: Int)
    private fun applyAngleSnap(p0x: Float, p0y: Float,
                               p1x: Float, p1y: Float): AngleSnapResult {
        val dx = p1x - p0x
        val dy = p1y - p0y
        val dist = kotlin.math.sqrt(dx * dx + dy * dy)
        if (dist < 1.0f) return AngleSnapResult(p1x, p1y, Int.MIN_VALUE)
        val step = kotlin.math.PI.toFloat() / 12f          // 15°
        val ang  = kotlin.math.atan2(dy, dx)
        val k    = kotlin.math.round(ang / step).toInt()
        val snappedAng = k * step
        return AngleSnapResult(
            p0x + (kotlin.math.cos(snappedAng) * dist).toFloat(),
            p0y + (kotlin.math.sin(snappedAng) * dist).toFloat(),
            k,
        )
    }

    // TEXT tool drag state (doc px). textHitId is the box under the
    // DOWN point, if any — a tap on it opens the editor.
    private var textP0X = 0f
    private var textP0Y = 0f
    private var textP0ViewX = 0f
    private var textP0ViewY = 0f
    private var textCurX = 0f
    private var textCurY = 0f
    private var textDragging = false
    private var textHitId = 0

    /** TEXT tool: tap on a box → edit; tap on empty canvas → auto-width
     *  box at the point; drag → fixed-width box spanning the drag's x
     *  range (rectangle preview while dragging). Placement is snapped
     *  like the shape tools so labels line up with the grid. */
    private fun handleTextEvent(
        r: GLFrontBufferedRenderer<StrokeAction>,
        event: MotionEvent
    ): Boolean {
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                viewToDoc(event.x, event.y, tmpDoc)
                textHitId = NativeRenderer.hitTestTextBoxAt(tmpDoc[0], tmpDoc[1])
                val (sx, sy) = if (textHitId == 0) snap(tmpDoc[0], tmpDoc[1])
                               else Pair(tmpDoc[0], tmpDoc[1])
                textP0X = sx; textP0Y = sy
                textCurX = sx; textCurY = sy
                textP0ViewX = event.x; textP0ViewY = event.y
                textDragging = false
            }
            MotionEvent.ACTION_MOVE -> {
                if (textHitId != 0) return true
                viewToDoc(event.x, event.y, tmpDoc)
                val (sx, sy) = snap(tmpDoc[0], tmpDoc[1])
                textCurX = sx; textCurY = sy
                val dvx = event.x - textP0ViewX
                val dvy = event.y - textP0ViewY
                if (!textDragging && dvx * dvx + dvy * dvy > kTextDragSlopPx * kTextDragSlopPx) {
                    textDragging = true
                }
                if (textDragging) {
                    r.renderFrontBufferedLayer(
                        StrokeAction.ShapePreview(1,
                            textP0X, textP0Y, textCurX, textCurY, p1Snapped)
                    )
                }
            }
            MotionEvent.ACTION_UP -> {
                if (textDragging) {
                    r.commit()
                    val x = minOf(textP0X, textCurX)
                    val y = minOf(textP0Y, textCurY)
                    val w = kotlin.math.abs(textCurX - textP0X)
                    onTextPlaceRequested?.invoke(x, y, w, false)
                } else if (textHitId != 0) {
                    onTextEditRequested?.invoke(textHitId)
                } else {
                    onTextPlaceRequested?.invoke(textP0X, textP0Y, 0f, true)
                }
                textDragging = false
                textHitId = 0
                p1Snapped = false
            }
            MotionEvent.ACTION_CANCEL -> {
                if (textDragging) r.commit()
                textDragging = false
                textHitId = 0
                p1Snapped = false
            }
        }
        return true
    }

    private fun handleShapeEvent(
        r: GLFrontBufferedRenderer<StrokeAction>,
        event: MotionEvent,
        shapeType: Int
    ): Boolean {
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                viewToDoc(event.x, event.y, tmpDoc)
                val (sx, sy) = snap(tmpDoc[0], tmpDoc[1])
                shapeP0X = sx; shapeP0Y = sy
                shapeP1X = sx; shapeP1Y = sy
                r.renderFrontBufferedLayer(
                    StrokeAction.ShapePreview(shapeType,
                        shapeP0X, shapeP0Y, shapeP1X, shapeP1Y, p1Snapped)
                )
            }
            MotionEvent.ACTION_MOVE -> {
                viewToDoc(event.x, event.y, tmpDoc)
                val (sx, sy) = snap(tmpDoc[0], tmpDoc[1])
                // Angle snap applies only to the LINE tool, only when
                // point-snap didn't already lock the endpoint to a
                // vertex (point snaps win — they're a deliberate hit on
                // a known target; angle snap is just a direction
                // constraint).
                val angleSnapping = (shapeType == 0
                                     && angleSnapEnabled
                                     && !p1Snapped)
                if (angleSnapping) {
                    val r = applyAngleSnap(shapeP0X, shapeP0Y, sx, sy)
                    shapeP1X = r.x; shapeP1Y = r.y
                    if (r.stepIdx != Int.MIN_VALUE) {
                        // 15° increments. Convert from doc-coord
                        // convention (y down, so a positive stepIdx
                        // means clockwise from +x) to display
                        // convention (90°=up). dispDeg = -k * 15
                        // wrapped into [0, 360).
                        var deg = (-r.stepIdx * 15) % 360
                        if (deg < 0) deg += 360
                        onLineAngleChanged?.invoke(deg.toFloat())
                    }
                } else {
                    shapeP1X = sx; shapeP1Y = sy
                }
                r.renderFrontBufferedLayer(
                    StrokeAction.ShapePreview(shapeType,
                        shapeP0X, shapeP0Y, shapeP1X, shapeP1Y, p1Snapped)
                )
            }
            MotionEvent.ACTION_UP -> {
                when (shapeType) {
                    0 -> NativeRenderer.addLine     (shapeP0X, shapeP0Y, shapeP1X, shapeP1Y)
                    1 -> NativeRenderer.addRectangle(shapeP0X, shapeP0Y, shapeP1X, shapeP1Y)
                    2 -> NativeRenderer.addCircle   (shapeP0X, shapeP0Y, shapeP1X, shapeP1Y)
                    3 -> NativeRenderer.addEllipse  (shapeP0X, shapeP0Y, shapeP1X, shapeP1Y)
                }
                // commit() clears the front buffer preview and triggers a
                // multi-buffer redraw via onDrawMultiBufferedLayer, which
                // applies the queued shape and re-renders the document.
                r.commit()
                p1Snapped = false
                onLineAngleChanged?.invoke(null)
            }
            MotionEvent.ACTION_CANCEL -> {
                r.commit()
                p1Snapped = false
                onLineAngleChanged?.invoke(null)
            }
        }
        return true
    }

    private var pageBoundsInitialized = false

    /** Optional callback fired once the surface has dims for the first
     *  time. MainActivity uses it to consult the active doc's saved
     *  page_size.txt and call setPageBounds with the right dimensions
     *  (rather than the surface dims, which can be larger now that the
     *  side panels overlay the SurfaceView). */
    var onSurfaceFirstSize: ((Int, Int) -> Unit)? = null

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        Log.i("DrawingApp", "onSizeChanged ${oldw}x${oldh} -> ${w}x${h}")
        // First time we know the surface size, hand control over to
        // MainActivity so it can decide what page bounds to apply (saved
        // dims for an existing doc, surface fallback for a legacy one).
        // We still kick a multi-buffer pass either way so the saved
        // document loads and shows immediately on app launch.
        if (!pageBoundsInitialized && w > 0 && h > 0) {
            pageBoundsInitialized = true
            val cb = onSurfaceFirstSize
            if (cb != null) {
                cb(w, h)
            } else {
                NativeRenderer.setPageBounds(0f, 0f, w.toFloat(), h.toFloat())
            }
            renderer?.commit()
        }
    }

    /** Force a multi-buffer redraw (used after layer-state changes that
     *  need to be reflected on screen, e.g. clearing a layer). */
    fun forceRedraw() {
        renderer?.commit()
    }

    fun release() {
        renderer?.release(true)
        renderer = null
    }

    override fun onDetachedFromWindow() {
        release()
        super.onDetachedFromWindow()
    }
}

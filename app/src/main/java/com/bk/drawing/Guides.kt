package com.bk.drawing

import android.content.Context
import android.graphics.Canvas
import android.graphics.DashPathEffect
import android.graphics.Paint
import android.graphics.Path
import android.view.View
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.sin

// Drafting guides: a ruler (straight edge) and a compass, both physical
// stand-ins that constrain the raster stroke tools (brush / eraser /
// shade) without changing anything about how a stroke is drawn.
//
// Everything here is doc-px and pure Kotlin. The constraint is applied
// to each sample in DrawingSurfaceView.handleStrokeEvent *before* the
// samples are packed into a BatchSamples, so native never knows a guide
// was involved: pressure, prediction, bake and undo are untouched, and
// the cost is a few flops per sample.
//
// Guides are per document (<docDir>/guides.txt), like a ruler left on
// the desk while you turn pages. They are drawn by GuideOverlayView, a
// plain View stacked over the SurfaceView, so they never reach the page
// pixels, thumbnails or exports.

/** Ruler model. p0→p1 is the *working edge* (edge A, the one with tick
 *  marks, the one position snapping lines up with targets). The body
 *  extends `width` doc-px to the left of p0→p1 (normal n = (-u.y, u.x)
 *  with y down), and its far side is edge B, also usable. */
class RulerGuide {
    var visible = false
    var x0 = 0f; var y0 = 0f
    var x1 = 0f; var y1 = 0f
    var width = kDefaultWidthDoc

    val length: Float get() = hypot(x1 - x0, y1 - y0)

    /** Along-edge coordinate t and normal coordinate s of (px, py), plus
     *  the unit axes, into out = [t, s, ux, uy, nx, ny]. False when the
     *  ruler is degenerate. */
    fun local(px: Float, py: Float, out: FloatArray): Boolean {
        val len = length
        if (len < 1e-3f) return false
        val ux = (x1 - x0) / len
        val uy = (y1 - y0) / len
        val nx = -uy
        val ny = ux
        val dx = px - x0
        val dy = py - y0
        out[0] = dx * ux + dy * uy
        out[1] = dx * nx + dy * ny
        out[2] = ux; out[3] = uy; out[4] = nx; out[5] = ny
        return true
    }

    /** Doc-px distance from (px, py) to the ruler's body rectangle
     *  (0 inside). Used to pick a guide when both are showing. */
    fun distanceTo(px: Float, py: Float, tmp: FloatArray): Float {
        if (!local(px, py, tmp)) return Float.MAX_VALUE
        val t = tmp[0]; val s = tmp[1]
        val dt = when { t < 0f -> -t; t > length -> t - length; else -> 0f }
        val ds = when { s < 0f -> -s; s > width -> s - width; else -> 0f }
        return hypot(dt, ds)
    }

    /** Displayed angle of the working edge in degrees, [0, 360),
     *  0° = right, 90° = up (doc y grows down). */
    fun displayAngleDeg(): Float {
        val a = Math.toDegrees(atan2(-(y1 - y0), x1 - x0).toDouble()).toFloat()
        return ((a % 360f) + 360f) % 360f
    }

    companion object {
        const val kDefaultWidthDoc = 64f
    }
}

/** Compass model: centre and radius. The radius handle sits on the
 *  circle at `handleAngle` (radians, doc convention) so it stays where
 *  the user last dragged it. */
class CompassGuide {
    var visible = false
    var cx = 0f; var cy = 0f
    var radius = 0f
    var handleAngle = 0f

    val handleX: Float get() = cx + radius * cos(handleAngle)
    val handleY: Float get() = cy + radius * sin(handleAngle)

    fun distanceTo(px: Float, py: Float): Float =
        abs(hypot(px - cx, py - cy) - radius)
}

class Guides {
    val ruler = RulerGuide()
    val compass = CompassGuide()

    val anyVisible: Boolean get() = ruler.visible || compass.visible

    // ---- Stroke constraint ---------------------------------------------
    //
    // Chosen once per stroke at pen-down, then applied to every sample
    // (real and predicted) of that stroke. The per-stroke snapshot of the
    // geometry means a guide edit can't move a stroke mid-flight.

    private enum class Mode { NONE, RULER_A, RULER_B, COMPASS }
    private var mode = Mode.NONE
    // Snapshot of the active guide (ruler: origin, axes, length, width;
    // compass: centre, radius).
    private var sx0 = 0f; private var sy0 = 0f
    private var sux = 0f; private var suy = 0f
    private var snx = 0f; private var sny = 0f
    private var sLen = 0f; private var sWidth = 0f
    private var sR = 0f
    // Last direction used for a compass sample that landed exactly on
    // the centre (no direction of its own).
    private var lastDirX = 1f; private var lastDirY = 0f
    private val tmp = FloatArray(6)

    val strokeConstrained: Boolean get() = mode != Mode.NONE

    /** Pen-down at doc (x, y): pick the guide and, for the ruler, the
     *  side whose edge the pen will slide along. A pen-down far from
     *  both still gets constrained — a real ruler on the desk blocks a
     *  stroke no matter where it started. When both guides are showing,
     *  the nearer one wins. */
    fun beginStroke(x: Float, y: Float) {
        mode = Mode.NONE
        val r = ruler
        val c = compass
        val rulerOk = r.visible && r.length >= 1f
        val compassOk = c.visible && c.radius >= 1f
        val useRuler = when {
            rulerOk && compassOk -> r.distanceTo(x, y, tmp) <= c.distanceTo(x, y)
            else -> rulerOk
        }
        if (useRuler) {
            if (!r.local(x, y, tmp)) return
            sx0 = r.x0; sy0 = r.y0
            sux = tmp[2]; suy = tmp[3]; snx = tmp[4]; sny = tmp[5]
            sLen = r.length; sWidth = r.width
            val s = tmp[1]
            mode = when {
                s <= 0f -> Mode.RULER_A
                s >= sWidth -> Mode.RULER_B
                // Pen-down on the body: slide along the nearer edge.
                s < sWidth * 0.5f -> Mode.RULER_A
                else -> Mode.RULER_B
            }
        } else if (compassOk) {
            sx0 = c.cx; sy0 = c.cy; sR = c.radius
            mode = Mode.COMPASS
            val dx = x - sx0; val dy = y - sy0
            val d = hypot(dx, dy)
            if (d > 1e-4f) { lastDirX = dx / d; lastDirY = dy / d }
            else { lastDirX = cos(c.handleAngle); lastDirY = sin(c.handleAngle) }
        }
    }

    fun endStroke() { mode = Mode.NONE }

    /** Constrain `count` [x, y, p] triples of `xyp` in place. */
    fun constrain(xyp: FloatArray, count: Int) {
        when (mode) {
            Mode.NONE -> return
            Mode.COMPASS -> {
                for (i in 0 until count) {
                    val k = i * 3
                    val dx = xyp[k] - sx0
                    val dy = xyp[k + 1] - sy0
                    val d = hypot(dx, dy)
                    if (d > 1e-4f) { lastDirX = dx / d; lastDirY = dy / d }
                    xyp[k]     = sx0 + lastDirX * sR
                    xyp[k + 1] = sy0 + lastDirY * sR
                }
            }
            Mode.RULER_A, Mode.RULER_B -> {
                val sideA = mode == Mode.RULER_A
                for (i in 0 until count) {
                    val k = i * 3
                    val dx = xyp[k] - sx0
                    val dy = xyp[k + 1] - sy0
                    val t = dx * sux + dy * suy
                    // Past the ends there's no straight edge to push on.
                    if (t < 0f || t > sLen) continue
                    val s = dx * snx + dy * sny
                    // Edge A is s = 0 with the free side s < 0; edge B is
                    // s = width with the free side s > width. Anything on
                    // the wrong side is pushed back onto the edge along
                    // its normal, so sliding along it draws a straight
                    // line and pulling away is free.
                    val push = if (sideA) { if (s > 0f) -s else 0f }
                               else { if (s < sWidth) sWidth - s else 0f }
                    if (push != 0f) {
                        xyp[k]     += push * snx
                        xyp[k + 1] += push * sny
                    }
                }
            }
        }
    }

    // ---- Persistence -----------------------------------------------------

    fun serialize(): String {
        val r = ruler; val c = compass
        return "ruler=${if (r.visible) 1 else 0}\n" +
               "ruler_x0=${r.x0}\nruler_y0=${r.y0}\n" +
               "ruler_x1=${r.x1}\nruler_y1=${r.y1}\n" +
               "ruler_width=${r.width}\n" +
               "compass=${if (c.visible) 1 else 0}\n" +
               "compass_cx=${c.cx}\ncompass_cy=${c.cy}\n" +
               "compass_r=${c.radius}\ncompass_handle=${c.handleAngle}\n"
    }

    /** Reset to "nothing placed", then apply whatever keys `text` has. */
    fun load(text: String?) {
        ruler.visible = false; ruler.x0 = 0f; ruler.y0 = 0f; ruler.x1 = 0f; ruler.y1 = 0f
        ruler.width = RulerGuide.kDefaultWidthDoc
        compass.visible = false; compass.cx = 0f; compass.cy = 0f
        compass.radius = 0f; compass.handleAngle = 0f
        if (text == null) return
        for (line in text.lines()) {
            val eq = line.indexOf('=')
            if (eq <= 0) continue
            val key = line.substring(0, eq).trim()
            val raw = line.substring(eq + 1).trim()
            val f = raw.toFloatOrNull()?.takeIf { it.isFinite() }
            when (key) {
                "ruler"          -> ruler.visible = raw == "1"
                "ruler_x0"       -> f?.let { ruler.x0 = it }
                "ruler_y0"       -> f?.let { ruler.y0 = it }
                "ruler_x1"       -> f?.let { ruler.x1 = it }
                "ruler_y1"       -> f?.let { ruler.y1 = it }
                "ruler_width"    -> f?.let { if (it > 1f) ruler.width = it }
                "compass"        -> compass.visible = raw == "1"
                "compass_cx"     -> f?.let { compass.cx = it }
                "compass_cy"     -> f?.let { compass.cy = it }
                "compass_r"      -> f?.let { compass.radius = maxOf(0f, it) }
                "compass_handle" -> f?.let { compass.handleAngle = it }
            }
        }
        // A degenerate saved guide is as good as none.
        if (ruler.length < 1f) ruler.visible = false
        if (compass.radius < 1f) compass.visible = false
    }
}

/**
 * Draws the ruler and compass over the canvas. Touches fall straight
 * through (not clickable), like BrushPreviewView. Geometry is doc-px,
 * mapped to view-px through [docToView] on every draw, so a pan / zoom
 * only needs an invalidate().
 */
class GuideOverlayView(
    context: Context,
    private val guides: Guides,
    private val docToView: (Float, Float, FloatArray) -> Unit,
    private val viewScale: () -> Float,
) : View(context) {

    /** Draw the editing handles (ruler ends, compass centre / radius
     *  knob). True while the matching guide tool is active. */
    var showRulerHandles = false
        set(v) { if (field != v) { field = v; invalidate() } }
    var showCompassHandles = false
        set(v) { if (field != v) { field = v; invalidate() } }

    private val density = context.resources.displayMetrics.density
    private val p = FloatArray(2)
    private val q = FloatArray(2)
    private val path = Path()

    private val bodyFill = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = 0x2E3A4F6B           // accent ink-blue, translucent
    }
    private val edgePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        color = 0xD03A4F6B.toInt()
    }
    private val tickPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        color = 0xB03A4F6B.toInt()
    }
    private val ringPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        color = 0xC03A4F6B.toInt()
    }
    private val haloPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        color = 0x80FFFFFF.toInt()
    }
    private val handleFill = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = 0xFFF5F0E6.toInt()   // paper
    }
    private val handleRing = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        color = 0xFFB5482E.toInt()   // hot
    }

    init {
        setWillNotDraw(false)
        isClickable = false
        isFocusable = false
    }

    override fun onDraw(canvas: Canvas) {
        if (guides.ruler.visible) drawRuler(canvas)
        if (guides.compass.visible) drawCompass(canvas)
    }

    private fun drawRuler(canvas: Canvas) {
        val r = guides.ruler
        val len = r.length
        if (len < 1f) return
        val ux = (r.x1 - r.x0) / len
        val uy = (r.y1 - r.y0) / len
        val nx = -uy
        val ny = ux
        val w = r.width

        // Body.
        path.reset()
        docToView(r.x0, r.y0, p); path.moveTo(p[0], p[1])
        docToView(r.x1, r.y1, p); path.lineTo(p[0], p[1])
        docToView(r.x1 + nx * w, r.y1 + ny * w, p); path.lineTo(p[0], p[1])
        docToView(r.x0 + nx * w, r.y0 + ny * w, p); path.lineTo(p[0], p[1])
        path.close()
        canvas.drawPath(path, bodyFill)

        // Edges: A (working) heavier than B.
        haloPaint.strokeWidth = 3.0f * density
        edgePaint.strokeWidth = 1.2f * density
        docToView(r.x0, r.y0, p); docToView(r.x1, r.y1, q)
        canvas.drawLine(p[0], p[1], q[0], q[1], haloPaint)
        canvas.drawLine(p[0], p[1], q[0], q[1], edgePaint)
        edgePaint.strokeWidth = 0.8f * density
        docToView(r.x0 + nx * w, r.y0 + ny * w, p)
        docToView(r.x1 + nx * w, r.y1 + ny * w, q)
        canvas.drawLine(p[0], p[1], q[0], q[1], edgePaint)
        docToView(r.x0, r.y0, p); docToView(r.x0 + nx * w, r.y0 + ny * w, q)
        canvas.drawLine(p[0], p[1], q[0], q[1], edgePaint)
        docToView(r.x1, r.y1, p); docToView(r.x1 + nx * w, r.y1 + ny * w, q)
        canvas.drawLine(p[0], p[1], q[0], q[1], edgePaint)

        // Ticks along edge A, measured from p0, into the body. Pitch
        // coarsens as you zoom out so they never smear into a band.
        val scale = viewScale()
        var pitch = 10f
        while (pitch * scale < 6f * density && pitch < 1e6f) pitch *= 5f
        val major = pitch * 10f
        val mid = pitch * 5f
        tickPaint.strokeWidth = 0.8f * density
        val n = floor(len / pitch).toInt()
        for (i in 0..n) {
            val t = i * pitch
            val tl = when {
                isMultiple(t, major) -> w * 0.45f
                isMultiple(t, mid)   -> w * 0.30f
                else                 -> w * 0.16f
            }
            docToView(r.x0 + ux * t, r.y0 + uy * t, p)
            docToView(r.x0 + ux * t + nx * tl, r.y0 + uy * t + ny * tl, q)
            canvas.drawLine(p[0], p[1], q[0], q[1], tickPaint)
        }

        if (showRulerHandles) {
            drawHandle(canvas, r.x0, r.y0)
            drawHandle(canvas, r.x1, r.y1)
        }
    }

    private fun isMultiple(t: Float, m: Float): Boolean {
        val k = t / m
        return abs(k - Math.round(k)) < 1e-3f
    }

    private val dash = DashPathEffect(floatArrayOf(6f, 5f), 0f)

    private fun drawCompass(canvas: Canvas) {
        val c = guides.compass
        if (c.radius < 1f) return
        val scale = viewScale()
        docToView(c.cx, c.cy, p)
        val rv = c.radius * scale
        haloPaint.strokeWidth = 3.0f * density
        ringPaint.strokeWidth = 1.2f * density
        canvas.drawCircle(p[0], p[1], rv, haloPaint)
        canvas.drawCircle(p[0], p[1], rv, ringPaint)

        // Centre pivot crosshair.
        val arm = 7f * density
        ringPaint.strokeWidth = 1.0f * density
        canvas.drawLine(p[0] - arm, p[1], p[0] + arm, p[1], ringPaint)
        canvas.drawLine(p[0], p[1] - arm, p[0], p[1] + arm, ringPaint)

        if (showCompassHandles) {
            // Radius arm, dashed, centre to knob.
            docToView(c.handleX, c.handleY, q)
            ringPaint.pathEffect = dash
            canvas.drawLine(p[0], p[1], q[0], q[1], ringPaint)
            ringPaint.pathEffect = null
            drawHandle(canvas, c.cx, c.cy)
            drawHandle(canvas, c.handleX, c.handleY)
        }
    }

    private fun drawHandle(canvas: Canvas, dx: Float, dy: Float) {
        docToView(dx, dy, q)
        val rad = kHandleDrawViewDp * density
        canvas.drawCircle(q[0], q[1], rad, handleFill)
        handleRing.strokeWidth = 1.6f * density
        canvas.drawCircle(q[0], q[1], rad, handleRing)
    }

    companion object {
        const val kHandleDrawViewDp = 7f
    }
}

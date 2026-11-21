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
// pixels, thumbnails or exports. A finger on a guide holds it (see
// DrawingSurfaceView.routeFingerGuide), so one hand can slide the ruler
// while the pen draws.

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
    //
    // The ruler is a solid rectangle the pen can't enter, from any side.
    // It's solved as a sliding contact in ruler-local (t, s) coordinates:
    // each sample moves the pen from where it last was towards the raw
    // point; if that path enters the body, the pen stops on the face it
    // hit and slides along it (the raw point projected onto that face).
    // Pulling away, or rounding a corner, is free. That makes the edge a
    // straight line to draw along, and an end a wall rather than a
    // teleport onto the nearer long edge.

    private enum class Mode { NONE, RULER, COMPASS }
    private var mode = Mode.NONE
    // Snapshot of the active guide (ruler: origin, axes, length, width;
    // compass: centre, radius).
    private var sx0 = 0f; private var sy0 = 0f
    private var sux = 0f; private var suy = 0f
    private var snx = 0f; private var sny = 0f
    private var sLen = 0f; private var sWidth = 0f
    private var sR = 0f
    // Contact state, advanced by real samples only. Predicted samples
    // run from a copy so the revert-and-replace of the predicted tail
    // can't leave the solver somewhere the pen never really went.
    // Ruler: pen position in local (t, s). Compass: last direction, for
    // a sample that lands exactly on the centre.
    private var penT = 0f; private var penS = 0f
    private var lastDirX = 1f; private var lastDirY = 0f
    private val tmp = FloatArray(6)
    private val out2 = FloatArray(2)
    private val lbP = FloatArray(4)
    private val lbQ = FloatArray(4)

    val strokeConstrained: Boolean get() = mode != Mode.NONE

    /** Pen-down at doc (x, y): pick the guide. A pen-down far from both
     *  still gets constrained (a ruler on the desk blocks a stroke no
     *  matter where it started); when both guides are showing, the
     *  nearer one wins. */
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
            mode = Mode.RULER
            penT = tmp[0]; penS = tmp[1]
            if (insideBody(penT, penS)) {
                // Pen-down on the body: start on the nearest face.
                val dA = penS; val dB = sWidth - penS
                val d0 = penT; val d1 = sLen - penT
                when (minOf(minOf(dA, dB), minOf(d0, d1))) {
                    dA -> penS = 0f
                    dB -> penS = sWidth
                    d0 -> penT = 0f
                    else -> penT = sLen
                }
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

    /** Constrain the first `count` [x, y, p] triples of `xyp` in place.
     *  The first `realCount` advance the contact state; the rest
     *  (prediction) are solved from a copy of it. */
    fun constrain(xyp: FloatArray, realCount: Int, count: Int = realCount) {
        if (mode == Mode.NONE) return
        val keepT = penT; val keepS = penS
        val keepDx = lastDirX; val keepDy = lastDirY
        var savedT = penT; var savedS = penS
        var savedDx = lastDirX; var savedDy = lastDirY
        for (i in 0 until count) {
            if (i == realCount) {
                // Real samples done: remember where they left the solver.
                savedT = penT; savedS = penS; savedDx = lastDirX; savedDy = lastDirY
            }
            val k = i * 3
            if (mode == Mode.COMPASS) {
                val dx = xyp[k] - sx0
                val dy = xyp[k + 1] - sy0
                val d = hypot(dx, dy)
                if (d > 1e-4f) { lastDirX = dx / d; lastDirY = dy / d }
                xyp[k]     = sx0 + lastDirX * sR
                xyp[k + 1] = sy0 + lastDirY * sR
            } else {
                val dx = xyp[k] - sx0
                val dy = xyp[k + 1] - sy0
                stepRuler(dx * sux + dy * suy, dx * snx + dy * sny)
                xyp[k]     = sx0 + penT * sux + penS * snx
                xyp[k + 1] = sy0 + penT * suy + penS * sny
            }
        }
        if (count > realCount) {
            penT = savedT; penS = savedS; lastDirX = savedDx; lastDirY = savedDy
        } else if (realCount == 0) {
            penT = keepT; penS = keepS; lastDirX = keepDx; lastDirY = keepDy
        }
    }

    private fun insideBody(t: Float, s: Float): Boolean =
        t > kEps && t < sLen - kEps && s > kEps && s < sWidth - kEps

    /** Move the pen from (penT, penS) towards raw (qt, qs). */
    private fun stepRuler(qt: Float, qs: Float) {
        val face = blockingFace(penT, penS, qt, qs)
        if (face < 0) { penT = qt; penS = qs; return }
        // Slide: the raw point projected onto the face it hit.
        slideOnto(face, qt, qs, out2)
        // Past the face's end (rounding a corner): if the rest of the way
        // is clear, the pen gets there.
        if (blockingFace(out2[0], out2[1], qt, qs) < 0) { penT = qt; penS = qs }
        else { penT = out2[0]; penS = out2[1] }
    }

    private fun slideOnto(face: Int, qt: Float, qs: Float, out: FloatArray) {
        when (face) {
            kFaceT0 -> { out[0] = 0f;   out[1] = qs.coerceIn(0f, sWidth) }
            kFaceT1 -> { out[0] = sLen; out[1] = qs.coerceIn(0f, sWidth) }
            kFaceS0 -> { out[0] = qt.coerceIn(0f, sLen); out[1] = 0f }
            else    -> { out[0] = qt.coerceIn(0f, sLen); out[1] = sWidth }
        }
    }

    /** Liang-Barsky clip of the segment (pt, ps) to (qt, qs) against the
     *  body [0, len] x [0, width]. Returns the face it enters through
     *  when the segment passes through the body's interior (not merely
     *  along a face or through a corner), else -1. */
    private fun blockingFace(pt: Float, ps: Float, qt: Float, qs: Float): Int {
        val dt = qt - pt
        val ds = qs - ps
        var u0 = 0f
        var u1 = 1f
        var face = -1
        // Each constraint is p * u <= q. Reused scratch: this runs per pen
        // sample, so no allocation here.
        val ps4 = lbP; val qs4 = lbQ
        ps4[0] = -dt; ps4[1] = dt; ps4[2] = -ds; ps4[3] = ds
        qs4[0] = pt; qs4[1] = sLen - pt; qs4[2] = ps; qs4[3] = sWidth - ps
        for (i in 0 until 4) {
            val p = ps4[i]
            val q = qs4[i]
            if (p == 0f) {
                if (q < 0f) return -1          // parallel and outside
                continue
            }
            val r = q / p
            if (p < 0f) {
                if (r >= u0) { u0 = r; face = i }
            } else if (r < u1) {
                u1 = r
            }
            if (u0 > u1) return -1
        }
        if (u1 - u0 <= 1e-6f) return -1
        // Grazing along a face or clipping a corner never enters the
        // interior; test the middle of the clipped span.
        val um = (u0 + u1) * 0.5f
        if (!insideBody(pt + dt * um, ps + ds * um)) return -1
        if (face < 0) {
            // Started inside (can't normally happen): nearest face.
            val dA = ps; val dB = sWidth - ps; val d0 = pt; val d1 = sLen - pt
            face = when (minOf(minOf(dA, dB), minOf(d0, d1))) {
                d0 -> kFaceT0; d1 -> kFaceT1; dA -> kFaceS0; else -> kFaceS1
            }
        }
        return face
    }

    private companion object {
        // Face ids match the Liang-Barsky constraint order:
        // t >= 0, t <= len, s >= 0 (edge A), s <= width (edge B).
        const val kFaceT0 = 0
        const val kFaceT1 = 1
        const val kFaceS0 = 2
        const val kFaceS1 = 3
        const val kEps = 1e-3f
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
    /** Ruler grip centre (doc-px) into the array; false = no grip now. */
    private val rulerGrip: (FloatArray) -> Boolean = { false },
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

        // Edges: all four the same thin line, inset into the body by half
        // its width so none of it lies over the drawing side. A stroke
        // run along any edge is centred on it, so a line centred on the
        // edge (or a halo) would cover half the stroke. The ticks are
        // what mark edge A as the working edge.
        edgePaint.strokeWidth = 0.8f * density
        val inset = edgePaint.strokeWidth * 0.5f / viewScale().coerceAtLeast(1e-3f)
        // Edge A (inward = +n), edge B (inward = -n).
        drawInsetEdge(canvas, r.x0, r.y0, r.x1, r.y1, nx * inset, ny * inset)
        drawInsetEdge(canvas, r.x0 + nx * w, r.y0 + ny * w,
                      r.x1 + nx * w, r.y1 + ny * w, -nx * inset, -ny * inset)
        // Ends (inward = +u at p0, -u at p1).
        drawInsetEdge(canvas, r.x0, r.y0, r.x0 + nx * w, r.y0 + ny * w,
                      ux * inset, uy * inset)
        drawInsetEdge(canvas, r.x1, r.y1, r.x1 + nx * w, r.y1 + ny * w,
                      -ux * inset, -uy * inset)

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
        drawRulerGrip(canvas, nx, ny)
    }

    /** Doc segment a→b shifted by doc offset (ox, oy) into the body. */
    private fun drawInsetEdge(canvas: Canvas, ax: Float, ay: Float,
                              bx: Float, by: Float, ox: Float, oy: Float) {
        docToView(ax + ox, ay + oy, p)
        docToView(bx + ox, by + oy, q)
        canvas.drawLine(p[0], p[1], q[0], q[1], edgePaint)
    }

    private val gripDoc = FloatArray(2)

    /** The pen grip: a paper disc with a double arrow across the ruler
     *  (the direction it slides in hatch steps). */
    private fun drawRulerGrip(canvas: Canvas, nx: Float, ny: Float) {
        if (!rulerGrip(gripDoc)) return
        docToView(gripDoc[0], gripDoc[1], p)
        val cx = p[0]; val cy = p[1]
        // View-space direction of the ruler normal.
        docToView(gripDoc[0] + nx, gripDoc[1] + ny, q)
        var vx = q[0] - cx; var vy = q[1] - cy
        val vl = kotlin.math.sqrt(vx * vx + vy * vy)
        if (vl < 1e-6f) return
        vx /= vl; vy /= vl
        val rad = kGripDrawDp * density
        canvas.drawCircle(cx, cy, rad, handleFill)
        edgePaint.strokeWidth = 1.4f * density
        canvas.drawCircle(cx, cy, rad, edgePaint)
        // Double arrow along the normal.
        val arm = rad * 0.6f
        val head = rad * 0.28f
        val ax = vx * arm; val ay = vy * arm
        canvas.drawLine(cx - ax, cy - ay, cx + ax, cy + ay, edgePaint)
        val px = -vy * head; val py = vx * head
        for (sgn in floatArrayOf(1f, -1f)) {
            val tx = cx + ax * sgn; val ty = cy + ay * sgn
            val bx = tx - vx * head * sgn; val by = ty - vy * head * sgn
            canvas.drawLine(tx, ty, bx + px, by + py, edgePaint)
            canvas.drawLine(tx, ty, bx - px, by - py, edgePaint)
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
        /** Ruler grip: drawn radius and pen grab radius (dp). */
        const val kGripDrawDp = 13f
        const val kGripHitDp = 24f
    }
}

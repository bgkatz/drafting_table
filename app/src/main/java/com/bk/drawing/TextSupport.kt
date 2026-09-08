package com.bk.drawing

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Typeface
import android.text.Layout
import android.text.SpannableString
import android.text.Spanned
import android.text.StaticLayout
import android.text.TextPaint
import android.text.style.ForegroundColorSpan
import android.text.style.StyleSpan
import android.text.style.UnderlineSpan
import androidx.core.content.res.ResourcesCompat
import java.nio.ByteBuffer
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min

/**
 * Fonts available to text boxes. Each entry is a bundled font family
 * (res/font/family_*.xml declaring regular/bold/italic/bold-italic
 * TTFs). Boxes persist the [FontEntry.key] string, never an index, so
 * entries can be added, removed, or reordered without touching saved
 * documents; an unknown key falls back to the first entry.
 *
 * To add a font: drop its TTFs into res/font (lowercase, underscores),
 * write a family XML next to the existing ones, add one entry here.
 */
object FontRegistry {
    data class FontEntry(val key: String, val displayName: String, val familyRes: Int)

    val entries: List<FontEntry> = listOf(
        FontEntry("inter",          "Inter",          R.font.family_inter),
        FontEntry("jetbrains_mono", "JetBrains Mono", R.font.family_jetbrains_mono),
        FontEntry("source_serif",   "Source Serif",   R.font.family_source_serif),
        FontEntry("routed_gothic",  "Routed Gothic",  R.font.family_routed_gothic),
    )
    const val defaultKey = "inter"

    private val cache = HashMap<String, Typeface>()

    fun entry(key: String): FontEntry =
        entries.firstOrNull { it.key == key } ?: entries[0]

    /** Resolved typeface for a key + style. Cached; falls back to the
     *  system sans if the resource fails to load so text never
     *  vanishes. */
    fun typeface(ctx: Context, key: String, bold: Boolean, italic: Boolean): Typeface {
        val cacheKey = "$key/$bold/$italic"
        cache[cacheKey]?.let { return it }
        val base = try {
            ResourcesCompat.getFont(ctx, entry(key).familyRes)
        } catch (e: Exception) { null } ?: Typeface.SANS_SERIF
        val style = (if (bold) Typeface.BOLD else 0) or (if (italic) Typeface.ITALIC else 0)
        val tf = Typeface.create(base, style)
        cache[cacheKey] = tf
        return tf
    }
}

/** A per-range style override. Offsets are UTF-16 indices into the
 *  box text. Flags add to the box's base style; [color] null = inherit.
 *  See [TextStyling.normalize] for the base/run split. */
data class TextRun(
    val start: Int, val end: Int,
    val bold: Boolean, val italic: Boolean, val underline: Boolean,
    val color: Int?,
) {
    fun flags(): Int = (if (bold) 1 else 0) or (if (italic) 2 else 0) or
        (if (underline) 4 else 0) or (if (color != null) 8 else 0)

    companion object {
        fun fromFlat(a: IntArray?): MutableList<TextRun> {
            val out = mutableListOf<TextRun>()
            if (a == null) return out
            var i = 0
            while (i + 3 < a.size) {
                val f = a[i + 2]
                out += TextRun(a[i], a[i + 1], f and 1 != 0, f and 2 != 0, f and 4 != 0,
                               if (f and 8 != 0) (a[i + 3] and 0xFFFFFF) else null)
                i += 4
            }
            return out
        }
        fun toFlat(runs: List<TextRun>): IntArray {
            val out = IntArray(runs.size * 4)
            for ((k, r) in runs.withIndex()) {
                out[k * 4] = r.start; out[k * 4 + 1] = r.end
                out[k * 4 + 2] = r.flags(); out[k * 4 + 3] = r.color ?: 0
            }
            return out
        }
    }
}

/** Kotlin mirror of the native TextBox. Sizes are doc px; [color] is
 *  0xRRGGBB; [align] is 0 left / 1 centre / 2 right. */
data class TextBoxModel(
    val id: Int,
    var x: Float, var y: Float,
    var w: Float, var h: Float,
    var rotation: Float,
    var color: Int,
    var fontSize: Float,
    var bold: Boolean, var italic: Boolean,
    var align: Int,
    var lineSpacing: Float,
    var autoWidth: Boolean,
    var fontKey: String,
    var text: String,
    val layerIdx: Int,
    var runs: MutableList<TextRun> = mutableListOf(),
) {
    companion object {
        /** Read a box back from native; null if the id isn't on the
         *  active page yet (a just-queued add). */
        fun fromNative(id: Int): TextBoxModel? {
            val n = NativeRenderer.getTextBoxNumeric(id) ?: return null
            val bytes = NativeRenderer.getTextBoxText(id) ?: ByteArray(0)
            val font = NativeRenderer.getTextBoxFont(id) ?: FontRegistry.defaultKey
            return TextBoxModel(
                id = id,
                x = n[0], y = n[1], w = n[2], h = n[3], rotation = n[4],
                color = n[5].toInt(), fontSize = n[6],
                bold = n[7] > 0.5f, italic = n[8] > 0.5f,
                align = n[9].toInt(), lineSpacing = n[10],
                autoWidth = n[11] > 0.5f,
                fontKey = font,
                text = String(bytes, Charsets.UTF_8),
                layerIdx = n[12].toInt(),
                runs = TextRun.fromFlat(NativeRenderer.getTextBoxRuns(id)),
            )
        }
    }

    /** True when some run carries a colour other than the base — the
     *  raster then has to carry colour itself (RGBA) rather than being
     *  tinted by native. */
    fun hasColorRuns(): Boolean = runs.any { it.color != null && it.color != color }

    /** Push the whole state to native. */
    fun pushToNative(undoable: Boolean) {
        NativeRenderer.updateTextBox(
            id, text.toByteArray(Charsets.UTF_8), fontKey, fontSize,
            bold, italic, align, lineSpacing, color,
            x, y, w, h, rotation, autoWidth, undoable,
            TextRun.toFlat(runs),
        )
    }
}

/** Effective style of one character. */
class CharStyle(var bold: Boolean, var italic: Boolean, var underline: Boolean, var color: Int)

/**
 * Per-character styling. The model is base style + additive runs;
 * editing works on a per-character array and [normalize] folds it
 * back: base bold/italic = what every character has, base colour =
 * the most common colour, runs = the exceptions.
 */
object TextStyling {
    fun expand(box: TextBoxModel): Array<CharStyle> {
        val n = box.text.length
        val arr = Array(n) { CharStyle(box.bold, box.italic, false, box.color) }
        for (r in box.runs) {
            for (i in max(0, r.start) until min(n, r.end)) {
                val c = arr[i]
                if (r.bold) c.bold = true
                if (r.italic) c.italic = true
                if (r.underline) c.underline = true
                if (r.color != null) c.color = r.color
            }
        }
        return arr
    }

    fun normalize(box: TextBoxModel, styles: Array<CharStyle>) {
        val n = styles.size
        if (n == 0) { box.runs = mutableListOf(); return }
        box.bold = styles.all { it.bold }
        box.italic = styles.all { it.italic }
        box.color = styles.groupingBy { it.color }.eachCount().maxByOrNull { it.value }!!.key
        fun key(c: CharStyle) = listOf(
            c.bold && !box.bold, c.italic && !box.italic, c.underline,
            if (c.color != box.color) c.color else null)
        val runs = mutableListOf<TextRun>()
        var i = 0
        while (i < n) {
            val k = key(styles[i])
            if (k[0] == false && k[1] == false && k[2] == false && k[3] == null) { i++; continue }
            var j = i + 1
            while (j < n && key(styles[j]) == k) j++
            runs += TextRun(i, j, k[0] as Boolean, k[1] as Boolean, k[2] as Boolean, k[3] as Int?)
            i = j
        }
        box.runs = runs
    }

    /** The box text with its runs as spans (StyleSpan / UnderlineSpan /
     *  ForegroundColorSpan). Base bold/italic/colour are NOT spans —
     *  they come from the paint / EditText typeface and colour. */
    fun spannable(box: TextBoxModel): SpannableString {
        val s = SpannableString(box.text)
        val n = box.text.length
        val fl = Spanned.SPAN_EXCLUSIVE_INCLUSIVE
        for (r in box.runs) {
            val a = r.start.coerceIn(0, n); val b = r.end.coerceIn(0, n)
            if (b <= a) continue
            if (r.bold || r.italic) {
                val st = (if (r.bold) Typeface.BOLD else 0) or (if (r.italic) Typeface.ITALIC else 0)
                s.setSpan(StyleSpan(st), a, b, fl)
            }
            if (r.underline) s.setSpan(UnderlineSpan(), a, b, fl)
            if (r.color != null) s.setSpan(ForegroundColorSpan((0xFF shl 24) or (r.color and 0xFFFFFF)), a, b, fl)
        }
        return s
    }

    /** Per-character styles read back from an edited Spanned (the
     *  overlay's text). IME composing underlines are ignored. */
    fun stylesFromSpanned(box: TextBoxModel, sp: Spanned): Array<CharStyle> {
        val n = sp.length
        val arr = Array(n) { CharStyle(box.bold, box.italic, false, box.color) }
        for (span in sp.getSpans(0, n, StyleSpan::class.java)) {
            val a = sp.getSpanStart(span).coerceIn(0, n); val b = sp.getSpanEnd(span).coerceIn(0, n)
            val bold = span.style and Typeface.BOLD != 0
            val italic = span.style and Typeface.ITALIC != 0
            for (i in a until b) { if (bold) arr[i].bold = true; if (italic) arr[i].italic = true }
        }
        for (span in sp.getSpans(0, n, UnderlineSpan::class.java)) {
            if (sp.getSpanFlags(span) and Spanned.SPAN_COMPOSING != 0) continue
            val a = sp.getSpanStart(span).coerceIn(0, n); val b = sp.getSpanEnd(span).coerceIn(0, n)
            for (i in a until b) arr[i].underline = true
        }
        for (span in sp.getSpans(0, n, ForegroundColorSpan::class.java)) {
            val a = sp.getSpanStart(span).coerceIn(0, n); val b = sp.getSpanEnd(span).coerceIn(0, n)
            val c = span.foregroundColor and 0xFFFFFF
            for (i in a until b) arr[i].color = c
        }
        return arr
    }
}

/**
 * Layout + rasterization for text boxes. StaticLayout does the work
 * (wrapping, alignment, spacing); we mirror its settings onto the edit
 * overlay's EditText so what you type is what gets rendered.
 *
 * The raster is an 8-bit coverage bitmap at [desiredScale] texels per
 * doc px — native tints it with the box colour, so recolouring never
 * needs a re-raster.
 */
object TextLayout {
    // Mirror of the native constants (kTextRasterMaxDim etc).
    private const val kMaxTexDim = 2048
    private const val kMinScale = 0.25f
    private const val kMaxScale = 8.0f
    /** Measuring width for auto-width boxes: wide enough that only
     *  explicit newlines break. */
    private const val kUnboundedWidthPx = 1 shl 20

    fun desiredScale(viewScale: Float, w: Float, h: Float): Float {
        var s = viewScale.coerceIn(kMinScale, kMaxScale)
        val maxDim = max(w, h)
        if (maxDim * s > kMaxTexDim) s = kMaxTexDim / maxDim
        return s
    }

    fun paint(ctx: Context, box: TextBoxModel, scale: Float,
              tint: Int = Color.WHITE): TextPaint =
        TextPaint(TextPaint.ANTI_ALIAS_FLAG or TextPaint.SUBPIXEL_TEXT_FLAG).apply {
            typeface = FontRegistry.typeface(ctx, box.fontKey, box.bold, box.italic)
            textSize = box.fontSize * scale
            color = tint   // WHITE for coverage-only rasters (native tints)
        }

    fun alignment(align: Int): Layout.Alignment = when (align) {
        1 -> Layout.Alignment.ALIGN_CENTER
        2 -> Layout.Alignment.ALIGN_OPPOSITE
        else -> Layout.Alignment.ALIGN_NORMAL
    }

    private fun build(text: CharSequence, paint: TextPaint, widthPx: Int,
                      align: Int, lineSpacing: Float): StaticLayout =
        StaticLayout.Builder.obtain(text, 0, text.length, paint, max(1, widthPx))
            .setAlignment(alignment(align))
            .setLineSpacing(0f, lineSpacing)
            .setIncludePad(true)
            .setBreakStrategy(Layout.BREAK_STRATEGY_SIMPLE)
            .setHyphenationFrequency(Layout.HYPHENATION_FREQUENCY_NONE)
            .build()

    /** Layout at [scale]; for auto-width boxes the width is measured
     *  first (widest line) and [box].w is updated to match. */
    fun layout(ctx: Context, box: TextBoxModel, scale: Float,
               tint: Int = Color.WHITE): StaticLayout {
        val p = paint(ctx, box, scale, tint)
        val text: CharSequence =
            if (box.text.isEmpty()) " "
            else if (box.runs.isEmpty()) box.text
            else TextStyling.spannable(box)
        if (box.autoWidth) {
            val probe = build(text, p, kUnboundedWidthPx, 0, box.lineSpacing)
            var maxW = 0f
            for (i in 0 until probe.lineCount) maxW = max(maxW, probe.getLineMax(i))
            // +2 px of slack so float rounding never wraps the last glyph.
            val wPx = ceil(maxW).toInt() + 2
            box.w = max(wPx / scale, 16f)
        }
        val widthPx = ceil(box.w * scale).toInt()
        return build(text, p, widthPx, box.align, box.lineSpacing)
    }

    /** Re-flow [box] at the current view scale and update its height
     *  (and width for auto-width boxes). Returns true if a dimension
     *  changed by more than a hair. */
    fun reflow(ctx: Context, box: TextBoxModel, viewScale: Float): Boolean {
        val scale = desiredScale(viewScale, box.w, box.h)
        val oldW = box.w; val oldH = box.h
        val l = layout(ctx, box, scale)
        box.h = max(l.height / scale, 1f)
        return kotlin.math.abs(box.h - oldH) > 0.25f || kotlin.math.abs(box.w - oldW) > 0.25f
    }

    class Raster(val alpha: ByteArray, val w: Int, val h: Int, val scale: Float,
                 val channels: Int)

    /** Rasterize [box] at [scale] into an 8-bit coverage bitmap whose
     *  dimensions are ceil(w*scale) x ceil(h*scale). Also updates the
     *  box height to the layout's height so the model and the raster
     *  agree. */
    fun rasterize(ctx: Context, box: TextBoxModel, scale: Float): Raster {
        // Coverage-only (tinted by native) unless some run has its own
        // colour, in which case the raster carries premultiplied RGBA.
        val rgba = box.hasColorRuns()
        val tint = if (rgba) ((0xFF shl 24) or (box.color and 0xFFFFFF)) else Color.WHITE
        val l = layout(ctx, box, scale, tint)
        box.h = max(l.height / scale, 1f)
        val bw = min(max(ceil(box.w * scale).toInt(), 1), kMaxTexDim * 2)
        val bh = min(max(ceil(box.h * scale).toInt(), 1), kMaxTexDim * 2)
        val channels = if (rgba) 4 else 1
        val bmp = Bitmap.createBitmap(bw, bh,
            if (rgba) Bitmap.Config.ARGB_8888 else Bitmap.Config.ALPHA_8)
        val canvas = Canvas(bmp)
        l.draw(canvas)
        val rowBytes = bmp.rowBytes
        val buf = ByteBuffer.allocate(rowBytes * bh)
        bmp.copyPixelsToBuffer(buf)   // ARGB_8888 lands as premultiplied R,G,B,A bytes
        bmp.recycle()
        val src = buf.array()
        val stride = bw * channels
        val out: ByteArray
        if (rowBytes == stride) {
            out = src
        } else {
            out = ByteArray(stride * bh)
            for (row in 0 until bh) {
                System.arraycopy(src, row * rowBytes, out, row * stride, stride)
            }
        }
        return Raster(out, bw, bh, scale, channels)
    }

    /** Rasterize + hand to native. Returns true if a dimension changed
     *  (caller decides whether to push the new geometry). */
    fun rasterizeAndUpload(ctx: Context, box: TextBoxModel, viewScale: Float): Boolean {
        val oldW = box.w; val oldH = box.h
        val scale = desiredScale(viewScale, box.w, box.h)
        val r = rasterize(ctx, box, scale)
        NativeRenderer.uploadTextRaster(box.id, r.scale, r.w, r.h, r.alpha, r.channels)
        return kotlin.math.abs(box.h - oldH) > 0.25f || kotlin.math.abs(box.w - oldW) > 0.25f
    }

    /** Export variant: rasterize at exactly [scale] (the PNG's doc→pixel
     *  factor) so the saved image gets crisp text, allowing the larger
     *  export texel cap. The normal upkeep loop re-rasters for the
     *  screen afterwards if the zoom differs enough. */
    fun rasterizeAndUploadAt(ctx: Context, box: TextBoxModel, scale: Float) {
        val maxDim = max(box.w, box.h)
        var s = scale.coerceIn(kMinScale, kMaxScale)
        if (maxDim * s > kMaxTexDim * 2) s = (kMaxTexDim * 2) / maxDim
        val r = rasterize(ctx, box, s)
        NativeRenderer.uploadTextRaster(box.id, r.scale, r.w, r.h, r.alpha, r.channels)
    }

    /** A text box read from a page (any page, not just the active one),
     *  with the layer state needed to draw it into an export. */
    class PageTextBox(val box: TextBoxModel, val layerVisible: Boolean, val layerOpacity: Float)

    fun loadPageBoxes(pageIdx: Int): List<PageTextBox> {
        val f = NativeRenderer.getPageTextBoxes(pageIdx)
        val out = ArrayList<PageTextBox>()
        var i = 0
        while (i + 15 <= f.size) {
            val id = f[i].toInt()
            val bytes = NativeRenderer.getPageTextBoxText(pageIdx, id) ?: ByteArray(0)
            val font = NativeRenderer.getPageTextBoxFont(pageIdx, id) ?: FontRegistry.defaultKey
            val box = TextBoxModel(
                id = id, x = f[i + 1], y = f[i + 2], w = f[i + 3], h = f[i + 4],
                rotation = f[i + 5], color = f[i + 6].toInt(), fontSize = f[i + 7],
                bold = f[i + 8] > 0.5f, italic = f[i + 9] > 0.5f, align = f[i + 10].toInt(),
                lineSpacing = f[i + 11], autoWidth = f[i + 12] > 0.5f,
                fontKey = font, text = String(bytes, Charsets.UTF_8), layerIdx = 0,
                runs = TextRun.fromFlat(NativeRenderer.getPageTextBoxRuns(pageIdx, id)),
            )
            out += PageTextBox(box, f[i + 13] > 0.5f, f[i + 14])
            i += 15
        }
        return out
    }

    /** Draw a page's text boxes onto a PDF page canvas as real text.
     *  [scale] and [offsetX]/[offsetY] replicate the letterboxing the
     *  export composite used for the page bitmap (doc → bitmap px), so
     *  text lands exactly where the raster would have. Z-order caveat:
     *  this draws above every raster stroke on the page. */
    fun drawPageTextToCanvas(ctx: Context, canvas: Canvas, pageIdx: Int,
                             scale: Float, offsetX: Float, offsetY: Float) {
        for (p in loadPageBoxes(pageIdx)) {
            if (!p.layerVisible || p.box.text.isEmpty()) continue
            val box = p.box
            val l = layout(ctx, box, scale, tint = (0xFF shl 24) or (box.color and 0xFFFFFF))
            l.paint.alpha = (p.layerOpacity.coerceIn(0f, 1f) * 255f).toInt()
            val cx = box.x + box.w * 0.5f
            val cy = box.y + box.h * 0.5f
            canvas.save()
            canvas.translate(offsetX + cx * scale, offsetY + cy * scale)
            canvas.rotate(Math.toDegrees(box.rotation.toDouble()).toFloat())
            canvas.translate(-box.w * 0.5f * scale, -box.h * 0.5f * scale)
            l.draw(canvas)
            canvas.restore()
        }
    }
}

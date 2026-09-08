package com.bk.drawing

import android.content.Context
import android.text.InputType
import android.text.Layout
import android.util.TypedValue
import android.view.Gravity
import android.view.KeyEvent
import android.view.View
import android.view.ViewGroup
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputMethodManager
import android.widget.EditText
import android.widget.FrameLayout
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

/**
 * Text-box editing and raster upkeep.
 *
 * Editing happens in a real [EditText] laid over the SurfaceView,
 * positioned, scaled, and rotated to sit exactly on the box — so the
 * IME, caret, in-text selection, hardware keyboards, and the system
 * clipboard all come for free. While the overlay is open native hides
 * that box; on commit the text is laid out with the same StaticLayout
 * settings the overlay uses, rasterized, and pushed to native as one
 * undo entry.
 *
 * Raster upkeep: after every multi-buffer render native reports
 * whether any box on screen lacks a raster (fresh load, undo, paste)
 * or has one rendered at a scale too far from the current zoom. We
 * answer by re-rasterizing those boxes at the current view scale,
 * skipping while a 2-finger gesture is in flight so a pinch doesn't
 * churn textures every frame.
 */
class TextEditController(
    private val ctx: Context,
    private val container: FrameLayout,
    private val view: DrawingSurfaceView,
) {
    /** Called when a box is placed on a raster layer: the host adds a
     *  vector layer and we retry once it's active. */
    var onNeedVectorLayer: (() -> Unit)? = null

    // Defaults for new boxes; updated from the last box edited so a
    // series of labels share a style without a trip to the panel.
    var defaultFontKey = FontRegistry.defaultKey
    var defaultFontSize = 24f
    var defaultBold = false
    var defaultItalic = false
    var defaultAlign = 0
    var defaultLineSpacing = 1.0f

    private var editText: OverlayEditText? = null
    private var editing: TextBoxModel? = null
    private var original: TextBoxModel? = null
    private var editingIsNew = false
    private val tmpView = FloatArray(2)

    val isEditing: Boolean get() = editing != null

    /** Fired when the box that style edits would apply to may have
     *  changed (editor opened/closed, selection changed). The host
     *  refreshes the TEXT panel from [currentStyleBox]. */
    var onStyleTargetChanged: (() -> Unit)? = null

    // ---- Style target ------------------------------------------------------

    /** The box style edits apply to: the one being edited, else the
     *  selected one, else null (edits then only update the defaults). */
    fun currentStyleBox(): TextBoxModel? {
        editing?.let { return it }
        val id = NativeRenderer.getSelectedTextBoxId()
        if (id == 0) return null
        return TextBoxModel.fromNative(id)
    }

    fun hasStyleTarget(): Boolean =
        editing != null || NativeRenderer.getSelectedTextBoxId() != 0

    /** A model carrying just the defaults, for panels with no target. */
    fun defaultsModel(): TextBoxModel = TextBoxModel(
        id = 0, x = 0f, y = 0f, w = 0f, h = 0f, rotation = 0f, color = 0,
        fontSize = defaultFontSize, bold = defaultBold, italic = defaultItalic,
        align = defaultAlign, lineSpacing = defaultLineSpacing, autoWidth = true,
        fontKey = defaultFontKey, text = "", layerIdx = 0,
    )

    /** Apply a style mutation to the current target and remember it as
     *  the default for new boxes. Editing: restyles the overlay live
     *  (the commit carries it into the undo history). Selected: one
     *  undoable update + re-raster right away. */
    fun applyStyleChange(mutate: (TextBoxModel) -> Unit) {
        val ed = editing
        if (ed != null) {
            mutate(ed)
            rememberDefaults(ed)
            applyStyle()
            reposition()
            return
        }
        val id = NativeRenderer.getSelectedTextBoxId()
        if (id != 0) {
            val box = TextBoxModel.fromNative(id)
            if (box != null) {
                mutate(box)
                rememberDefaults(box)
                TextLayout.rasterizeAndUpload(ctx, box, view.currentViewScale)
                box.pushToNative(undoable = true)
                view.forceRedraw()
                return
            }
        }
        val scratch = defaultsModel()
        mutate(scratch)
        rememberDefaults(scratch)
    }

    /** Colour follows the brush colour while a box is targeted — the
     *  selected characters if the editor has a selection, else the
     *  whole box. */
    fun setColor(rgb: Int) {
        if (!hasStyleTarget()) return
        applyCharStyle { it.color = rgb and 0xFFFFFF }
    }

    // ---- Per-character style -----------------------------------------------

    /** The overlay's selection as a range, or null when it's just a
     *  caret (or no editor is open). */
    fun selectionRange(): IntRange? {
        val et = editText ?: return null
        val a = min(et.selectionStart, et.selectionEnd)
        val b = max(et.selectionStart, et.selectionEnd)
        return if (b > a) a until b else null
    }

    /** Apply a per-character mutation to the target range: the overlay
     *  selection if there is one, otherwise every character of the
     *  edited / selected box. The result is normalized back into base
     *  style + runs, so toggling bold on a fully-bold box simply clears
     *  it and toggling it on one word adds a run. */
    fun applyCharStyle(mutate: (CharStyle) -> Unit) {
        val ed = editing
        val et = editText
        if (ed != null && et != null) {
            val selS = et.selectionStart; val selE = et.selectionEnd
            ed.text = et.text.toString()
            val styles = TextStyling.stylesFromSpanned(ed, et.text)
            val range = selectionRange() ?: (0 until styles.size)
            for (i in range) if (i in styles.indices) mutate(styles[i])
            TextStyling.normalize(ed, styles)
            rememberDefaults(ed)
            applyStyle()
            et.setText(TextStyling.spannable(ed))
            val n = et.text.length
            et.setSelection(selS.coerceIn(0, n), selE.coerceIn(0, n))
            reposition()
            onStyleTargetChanged?.invoke()
            return
        }
        val id = NativeRenderer.getSelectedTextBoxId()
        if (id != 0) {
            val box = TextBoxModel.fromNative(id)
            if (box != null) {
                val styles = TextStyling.expand(box)
                for (c in styles) mutate(c)
                TextStyling.normalize(box, styles)
                rememberDefaults(box)
                TextLayout.rasterizeAndUpload(ctx, box, view.currentViewScale)
                box.pushToNative(undoable = true)
                view.forceRedraw()
                return
            }
        }
        // No target: mutate the defaults through a one-character scratch.
        val scratch = defaultsModel()
        val c = CharStyle(scratch.bold, scratch.italic, false, scratch.color)
        mutate(c)
        scratch.bold = c.bold; scratch.italic = c.italic
        rememberDefaults(scratch)
    }

    /** [bold, italic, underline] as the TEXT panel should show them:
     *  for an overlay selection, true only if every selected character
     *  has it; for a caret, the character before it (falling back to
     *  the base); for a selected box, every character; else defaults. */
    fun panelFlags(): BooleanArray {
        val ed = editing
        val et = editText
        if (ed != null && et != null) {
            val styles = TextStyling.stylesFromSpanned(ed, et.text)
            val range = selectionRange()
            if (range != null && styles.isNotEmpty()) {
                val sub = range.filter { it in styles.indices }.map { styles[it] }
                return booleanArrayOf(sub.all { it.bold }, sub.all { it.italic }, sub.all { it.underline })
            }
            val at = (min(et.selectionStart, et.selectionEnd) - 1)
            if (at in styles.indices) {
                val c = styles[at]
                return booleanArrayOf(c.bold, c.italic, c.underline)
            }
            return booleanArrayOf(ed.bold, ed.italic, false)
        }
        val id = NativeRenderer.getSelectedTextBoxId()
        if (id != 0) {
            val box = TextBoxModel.fromNative(id)
            if (box != null) {
                val styles = TextStyling.expand(box)
                if (styles.isEmpty()) return booleanArrayOf(box.bold, box.italic, false)
                return booleanArrayOf(styles.all { it.bold }, styles.all { it.italic }, styles.all { it.underline })
            }
        }
        return booleanArrayOf(defaultBold, defaultItalic, false)
    }

    // ---- System clipboard --------------------------------------------------

    /** Plain text of the selected box (for the system clipboard), or null. */
    fun selectedText(): String? {
        val id = NativeRenderer.getSelectedTextBoxId()
        if (id == 0) return null
        return TextBoxModel.fromNative(id)?.text
    }

    /** Drop [text] into a new auto-width box centred on the visible
     *  canvas, committed immediately and left selected. */
    fun pasteText(text: String, color: Int) {
        commitIfOpen()
        val activeIsVector =
            NativeRenderer.getLayerType(NativeRenderer.getActiveLayer()) == 1
        if (!activeIsVector) {
            onNeedVectorLayer?.invoke()
            view.postDelayed({ pasteText(text, color) }, 90L)
            return
        }
        val (cx, cy) = view.viewCenterDoc()
        val id = NativeRenderer.addTextBox(
            cx, cy, 16f, true, defaultFontKey, defaultFontSize,
            defaultBold, defaultItalic, defaultAlign, defaultLineSpacing,
        )
        val box = TextBoxModel(
            id = id, x = cx, y = cy, w = 16f, h = defaultFontSize * 1.3f,
            rotation = 0f, color = color and 0xFFFFFF, fontSize = defaultFontSize,
            bold = defaultBold, italic = defaultItalic, align = defaultAlign,
            lineSpacing = defaultLineSpacing, autoWidth = true,
            fontKey = defaultFontKey, text = text,
            layerIdx = NativeRenderer.getActiveLayer(),
        )
        TextLayout.rasterizeAndUpload(ctx, box, view.currentViewScale)
        box.x = cx - box.w * 0.5f
        box.y = cy - box.h * 0.5f
        box.pushToNative(undoable = true)
        view.forceRedraw()
        view.postDelayed({
            NativeRenderer.selectTextBox(id)
            view.forceRedraw()
            onStyleTargetChanged?.invoke()
        }, 60L)
    }

    // ---- Export ------------------------------------------------------------

    /** Before a PNG export: commit any open edit and re-raster every
     *  box on the active page at the export's doc→pixel scale so the
     *  saved image isn't limited to screen resolution. Uploads land on
     *  the same GL pass that renders the export. */
    fun prepareForPngExport(exportScale: Float) {
        commitIfOpen()
        val page = NativeRenderer.getActivePage()
        for (p in TextLayout.loadPageBoxes(page)) {
            if (p.box.text.isEmpty()) continue
            TextLayout.rasterizeAndUploadAt(ctx, p.box, exportScale)
        }
    }

    // ---- Defaults persistence ---------------------------------------------

    fun loadDefaults(prefs: android.content.SharedPreferences) {
        defaultFontKey = prefs.getString("text_font", defaultFontKey) ?: defaultFontKey
        defaultFontSize = prefs.getFloat("text_size", defaultFontSize)
        defaultBold = prefs.getBoolean("text_bold", defaultBold)
        defaultItalic = prefs.getBoolean("text_italic", defaultItalic)
        defaultAlign = prefs.getInt("text_align", defaultAlign)
        defaultLineSpacing = prefs.getFloat("text_spacing", defaultLineSpacing)
    }

    private var prefsForDefaults: android.content.SharedPreferences? = null
    fun bindPrefs(prefs: android.content.SharedPreferences) {
        prefsForDefaults = prefs
        loadDefaults(prefs)
    }

    // ---- Placement / open ------------------------------------------------

    /** Start a new box at doc (x, y). [w] is the fixed width for a
     *  drag-placed box; [autoWidth] boxes grow with their text. */
    fun placeNew(x: Float, y: Float, w: Float, autoWidth: Boolean, color: Int) {
        commitIfOpen()
        val activeIsVector =
            NativeRenderer.getLayerType(NativeRenderer.getActiveLayer()) == 1
        if (!activeIsVector) {
            // Add a vector layer (queued; becomes active on the GL
            // thread) and come back once it has landed.
            onNeedVectorLayer?.invoke()
            view.postDelayed({ placeNew(x, y, w, autoWidth, color) }, 90L)
            return
        }
        val width = if (autoWidth) 16f else max(w, 40f)
        val id = NativeRenderer.addTextBox(
            x, y, width, autoWidth, defaultFontKey, defaultFontSize,
            defaultBold, defaultItalic, defaultAlign, defaultLineSpacing,
        )
        // Drain the add now so a quick commit finds the box.
        view.forceRedraw()
        val box = TextBoxModel(
            id = id, x = x, y = y, w = width, h = defaultFontSize * 1.3f,
            rotation = 0f, color = color and 0xFFFFFF, fontSize = defaultFontSize,
            bold = defaultBold, italic = defaultItalic, align = defaultAlign,
            lineSpacing = defaultLineSpacing, autoWidth = autoWidth,
            fontKey = defaultFontKey, text = "",
            layerIdx = NativeRenderer.getActiveLayer(),
        )
        editingIsNew = true
        openOverlay(box)
    }

    /** Open the overlay on an existing box. */
    fun editExisting(id: Int) {
        if (editing?.id == id) return
        commitIfOpen()
        val box = TextBoxModel.fromNative(id) ?: return
        editingIsNew = false
        openOverlay(box)
    }

    private fun openOverlay(box: TextBoxModel) {
        editing = box
        original = box.copy()
        NativeRenderer.setTextEditing(box.id)
        view.forceRedraw()
        onStyleTargetChanged?.invoke()

        val et = OverlayEditText(ctx).apply {
            setText(TextStyling.spannable(box))
            setSelection(box.text.length)
            onSelectionChangedCb = { onStyleTargetChanged?.invoke() }
            background = null
            setPadding(0, 0, 0, 0)
            includeFontPadding = true
            breakStrategy = Layout.BREAK_STRATEGY_SIMPLE
            hyphenationFrequency = Layout.HYPHENATION_FREQUENCY_NONE
            setHorizontallyScrolling(false)
            isSingleLine = false
            setRawInputType(InputType.TYPE_CLASS_TEXT
                or InputType.TYPE_TEXT_FLAG_MULTI_LINE
                or InputType.TYPE_TEXT_FLAG_CAP_SENTENCES)
            imeOptions = EditorInfo.IME_ACTION_DONE or
                EditorInfo.IME_FLAG_NO_EXTRACT_UI or
                EditorInfo.IME_FLAG_NO_FULLSCREEN
            setOnEditorActionListener { _, actionId, _ ->
                if (actionId == EditorInfo.IME_ACTION_DONE) { commit(); true } else false
            }
            setOnKeyListener { _, keyCode, event ->
                if (event.action != KeyEvent.ACTION_DOWN) return@setOnKeyListener false
                when {
                    keyCode == KeyEvent.KEYCODE_ESCAPE -> { cancel(); true }
                    keyCode == KeyEvent.KEYCODE_ENTER && event.isCtrlPressed -> { commit(); true }
                    else -> false
                }
            }
            onBackPressed = { commit() }
            pivotX = 0f
            pivotY = 0f
            // Stylus handwriting (Android 14+, Gboard): writing with the
            // pen over the box converts to text. Pad the initiation
            // bounds so a stroke that starts a little outside the box
            // still counts as writing into it rather than a canvas tap.
            isAutoHandwritingEnabled = true
            if (android.os.Build.VERSION.SDK_INT >= 34) {
                val pad = 24f * ctx.resources.displayMetrics.density
                setHandwritingBoundsOffsets(pad, pad, pad, pad)
            }
        }
        editText = et
        container.addView(et, FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        applyStyle()
        reposition()
        et.requestFocus()
        et.post {
            val imm = ctx.getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager
            imm.showSoftInput(et, InputMethodManager.SHOW_IMPLICIT)
        }
    }

    /** Mirror the box's style onto the overlay at the current zoom. */
    private fun applyStyle() {
        val et = editText ?: return
        val box = editing ?: return
        val s = view.currentViewScale
        et.typeface = FontRegistry.typeface(ctx, box.fontKey, box.bold, box.italic)
        et.setTextSize(TypedValue.COMPLEX_UNIT_PX, box.fontSize * s)
        et.setLineSpacing(0f, box.lineSpacing)
        et.setTextColor((0xFF shl 24) or (box.color and 0xFFFFFF))
        et.gravity = when (box.align) {
            1 -> Gravity.CENTER_HORIZONTAL
            2 -> Gravity.END
            else -> Gravity.START
        } or Gravity.TOP
    }

    /** Place the overlay on the box: view-px position of the box's
     *  rotated top-left, rotation = box + view rotation, width =
     *  box width at the current zoom (auto-width boxes wrap content). */
    fun reposition() {
        val et = editText ?: return
        val box = editing ?: return
        val s = view.currentViewScale
        // Re-apply the size in case the zoom changed.
        et.setTextSize(TypedValue.COMPLEX_UNIT_PX, box.fontSize * s)
        val lp = et.layoutParams as FrameLayout.LayoutParams
        lp.width = if (box.autoWidth) ViewGroup.LayoutParams.WRAP_CONTENT
                   else max(ceil(box.w * s).toInt(), 1)
        lp.height = ViewGroup.LayoutParams.WRAP_CONTENT
        et.layoutParams = lp
        et.minWidth = if (box.autoWidth) ceil(box.fontSize * s).toInt() else 0

        // Rotated top-left of the box in doc px, then to view px.
        val cx = box.x + box.w * 0.5f
        val cy = box.y + box.h * 0.5f
        val c = cos(box.rotation); val sn = sin(box.rotation)
        val hx = -box.w * 0.5f; val hy = -box.h * 0.5f
        val ox = cx + hx * c - hy * sn
        val oy = cy + hx * sn + hy * c
        view.docToView(ox, oy, tmpView)
        et.translationX = tmpView[0]
        et.translationY = tmpView[1]
        et.rotation = Math.toDegrees((box.rotation + view.currentViewRotation).toDouble()).toFloat()
    }

    // ---- Close --------------------------------------------------------------

    /** Commit if an editor is open. Returns true if one was. */
    fun commitIfOpen(): Boolean {
        if (editing == null) return false
        commit()
        return true
    }

    fun commit() {
        val box = editing ?: return
        val et = editText ?: return
        box.text = et.text.toString()
        // Fold the overlay's spans back into base style + runs.
        TextStyling.normalize(box, TextStyling.stylesFromSpanned(box, et.text))
        closeOverlay()
        if (box.text.isBlank()) {
            // Nothing to keep: drop the box (no undo entry for a never-
            // committed provisional box; VectorDelete otherwise).
            NativeRenderer.removeTextBox(box.id)
            view.forceRedraw()
            return
        }
        // Lay out at the current zoom (sets h, and w for auto-width),
        // upload the raster, then push the model. Both are queued and
        // drained together on the GL thread.
        TextLayout.rasterizeAndUpload(ctx, box, view.currentViewScale)
        val changed = editingIsNew || box != original
        box.pushToNative(undoable = changed)
        rememberDefaults(box)
        view.forceRedraw()
    }

    fun cancel() {
        val box = editing ?: return
        closeOverlay()
        if (editingIsNew) NativeRenderer.removeTextBox(box.id)
        view.forceRedraw()
    }

    private fun closeOverlay() {
        val et = editText
        if (et != null) {
            val imm = ctx.getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager
            imm.hideSoftInputFromWindow(et.windowToken, 0)
            container.removeView(et)
        }
        editText = null
        editing = null
        original = null
        NativeRenderer.setTextEditing(0)
        onStyleTargetChanged?.invoke()
    }

    private fun rememberDefaults(box: TextBoxModel) {
        defaultFontKey = box.fontKey
        defaultFontSize = box.fontSize
        defaultBold = box.bold
        defaultItalic = box.italic
        defaultAlign = box.align
        defaultLineSpacing = box.lineSpacing
        prefsForDefaults?.edit()
            ?.putString("text_font", defaultFontKey)
            ?.putFloat("text_size", defaultFontSize)
            ?.putBoolean("text_bold", defaultBold)
            ?.putBoolean("text_italic", defaultItalic)
            ?.putInt("text_align", defaultAlign)
            ?.putFloat("text_spacing", defaultLineSpacing)
            ?.apply()
    }

    // ---- Raster upkeep -----------------------------------------------------

    private var rasterRunQueued = false

    /** Native flagged a box with a missing / stale raster. Debounced;
     *  deferred while a view gesture is in progress. */
    fun rasterizeRequested() {
        if (rasterRunQueued) return
        rasterRunQueued = true
        view.postDelayed({ rasterRunQueued = false; rasterizeMissing() }, 40L)
    }

    private fun rasterizeMissing() {
        if (view.isGestureActive) {
            rasterizeRequested()
            return
        }
        val vs = view.currentViewScale
        val req = NativeRenderer.getTextRasterRequests(vs)
        if (req.isEmpty()) return
        var any = false
        var i = 0
        while (i + 1 < req.size) {
            val id = req[i].toInt()
            i += 2
            if (id == editing?.id) continue
            val box = TextBoxModel.fromNative(id) ?: continue
            val geomChanged = TextLayout.rasterizeAndUpload(ctx, box, vs)
            if (geomChanged) box.pushToNative(undoable = false)
            any = true
        }
        if (any) view.forceRedraw()
    }

    /** EditText that commits on the IME's Back rather than merely
     *  hiding the keyboard and stranding the overlay, and reports
     *  selection changes so the TEXT panel can track the caret. */
    private class OverlayEditText(ctx: Context) : EditText(ctx) {
        var onBackPressed: (() -> Unit)? = null
        var onSelectionChangedCb: (() -> Unit)? = null
        override fun onSelectionChanged(selStart: Int, selEnd: Int) {
            super.onSelectionChanged(selStart, selEnd)
            onSelectionChangedCb?.invoke()
        }
        override fun onKeyPreIme(keyCode: Int, event: KeyEvent): Boolean {
            if (keyCode == KeyEvent.KEYCODE_BACK && event.action == KeyEvent.ACTION_UP) {
                onBackPressed?.invoke()
                return true
            }
            return super.onKeyPreIme(keyCode, event)
        }
    }
}

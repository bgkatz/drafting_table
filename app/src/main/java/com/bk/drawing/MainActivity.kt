package com.bk.drawing

import android.app.AlertDialog
import android.content.Context
import android.content.res.ColorStateList
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Typeface
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.view.KeyEvent
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.GridLayout
import android.widget.HorizontalScrollView
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.TextView
import android.net.Uri
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.res.ResourcesCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import java.io.File

/**
 * Main UI host. Layout follows the Concept A v4 design:
 *
 *   ┌─────┬───────┬──────────┬─────────────────────────────┐
 *   │tool │ pages │  layer   │           canvas            │
 *   │rail │  bar  │  panel   │         (drawing)           │
 *   │56dp │ 84dp  │  180dp   │       (weight = 1)          │
 *   ├─────┴───────┴──────────┴─────────────────────────────┤
 *   │                  status bar (28dp)                   │
 *   └──────────────────────────────────────────────────────┘
 *
 * Floating chrome (undo/redo chip) is positioned over the canvas frame.
 */
class MainActivity : AppCompatActivity() {

    private var drawingView: DrawingSurfaceView? = null

    // ---- Tool rail ------------------------------------------------------
    // Rail tiles are pure Views with isSelected() driving their bg drawable
    // and tint. We keep references so onToolChanged can flip selection
    // without rebuilding the rail.
    private val railToolTiles  = mutableMapOf<Tool, ImageView>()
    private lateinit var pagesRailTile:  ImageView
    private lateinit var layersRailTile: ImageView
    private lateinit var colorRailTile:  ImageView
    // Layer panel sections — kept as lateinit refs so the layers / color
    // rail tiles can flip them between VISIBLE and GONE. The wrappers
    // include their adjacent panelDivider so toggling collapses cleanly.
    private lateinit var layersSection:  View
    private lateinit var colorSection:   View

    // ---- Layer panel ----------------------------------------------------
    private lateinit var layerListContainer: LinearLayout
    private lateinit var layerActiveHeading: TextView
    private lateinit var layerOpacitySlider: SeekBar
    private lateinit var layerOpacityValue: TextView

    // Layer drag-to-reorder state. dragSourceIdx is the native idx (not
    // ui position); -1 means not dragging. Other rows are displaced via
    // translationY as the dragged row passes over their centers.
    private var dragSourceIdx: Int = -1
    private var dragSourceView: View? = null
    private var dragStartRawY: Float = 0f
    private var dragStartTranslationY: Float = 0f
    private var dragRowHeightPx: Int = 0
    private var dragCurrentTargetUiPos: Int = -1

    // Page drag-to-reorder state. Same shape as the layer fields. Page
    // sidebar layout is top-down with idx 0 at top, so uiPos == pageIdx
    // and no conversion is needed.
    private var dragPageSourceIdx: Int = -1
    private var dragPageSourceView: View? = null
    private var dragPageStartRawY: Float = 0f
    private var dragPageStartTranslationY: Float = 0f
    private var dragPageSlotHeightPx: Int = 0
    private var dragPageCurrentTargetIdx: Int = -1
    private lateinit var sizeSlider: SeekBar
    private lateinit var sizeSliderLabel: TextView
    private lateinit var sizeValueLabel: TextView
    private lateinit var brushSectionHeader: TextView
    // Per-row refs so updateSizeSliderForTool can hide rows that don't
    // apply to the active tool (e.g. for BUCKET only alpha is shown).
    private lateinit var brushSizeRow: View
    private lateinit var brushAlphaRow: View
    private lateinit var strokeUniformAlphaRow: View
    private lateinit var brushHardRow: View
    private lateinit var brushPressRow: View
    private lateinit var bucketBleedRow: View
    private lateinit var colorChip: View

    // ---- Status bar -----------------------------------------------------
    private lateinit var statusToolText:  TextView
    private lateinit var statusDocText:   TextView
    private lateinit var statusGridText:    TextView
    private lateinit var statusMarginText:  TextView
    private lateinit var statusPixelGridText: TextView
    private lateinit var statusSnapText:    TextView
    private lateinit var statusAngleText:   TextView
    private lateinit var statusPreviewText: TextView
    private lateinit var statusPredictText: TextView
    private lateinit var statusPageText:  TextView

    // ---- Page sidebar (restyled to 84dp) -------------------------------
    private lateinit var sidebarScroll: ScrollView
    private lateinit var sidebarLayout: LinearLayout
    private lateinit var docsButton:    LinearLayout
    private lateinit var docsButtonLabel: TextView
    private val pageItems = mutableListOf<PageSidebarItem>()
    private val kSidebarWidthDp   = 84
    // Page-thumbnail bounding box. Actual thumbnail dimensions are
    // derived from the canvas aspect ratio inside thumbDimensions() so
    // the thumb is always the same shape as the page — no letterboxing.
    // The design ships portrait (60×78); we honor whichever fit produces
    // the canvas aspect within these bounds.
    private val kThumbMaxWidthDp  = 60
    private val kThumbMaxHeightDp = 78
    private var lastBuiltActivePage = -1
    private var lastBuiltPageCount  = -1

    private data class PageSidebarItem(
        val container: LinearLayout,
        val frame: FrameLayout,
        val imageView: ImageView,
        val numberBadge: TextView,
        val bitmap: Bitmap,
        var pageIdx: Int
    )

    // ---- App state mirrors ---------------------------------------------
    private val kDefaultGridSpacing = 50
    private val kGridSpacingMin     = 8
    private val kGridSpacingMax     = 256
    private val kDefaultMarginTop   = 96
    private val kDefaultMarginSide  = 72
    private val kMarginMax          = 400
    private var gridState = 0          // 0 = off, 1 = lines, 2 = dots
    // Page setup — per-document, persisted in <docDir>/page_setup.txt
    // (see loadPageSetup / writePageSetup). gridState is part of it so
    // a doc set up as graph paper reopens as graph paper. Margins are
    // rose rules inset from the page edge: across the top and bottom
    // at marginTop, down each side at marginSide. All sizes are doc px.
    private var gridSpacing    = kDefaultGridSpacing
    private var marginsEnabled = false
    private var marginTop      = kDefaultMarginTop
    private var marginSide     = kDefaultMarginSide
    // Pixel-grid overlay (1-doc-pixel boundaries). Native self-gates on
    // zoom — only renders when view scale is high enough that pixel
    // boundaries are visibly distinct. Persists across launches.
    private var pixelGridEnabled = false
    private var snapEnabled = true
    private var stylusOnly  = true
    private var layerCount = 1
    private var activeLayerIndex = 0
    private var currentToolMirror: Tool = Tool.BRUSH
    private val kPrefsName = "drawing_app_prefs"
    private val kPrefLastDoc = "last_doc"
    // Top-level folder name on shared storage. Visible in the Files
    // app and through USB; matches the "AppName/" convention other
    // drawing apps use (Clip Studio Paint/, Wacom Canvas/, etc.).
    private val kDocumentsRootName = "Drafting Table"
    // Set true once setDocumentDir has been called with a real path.
    // The whole document-init flow (migration + initial doc choice
    // + sidebar rebuild) waits for the MANAGE_EXTERNAL_STORAGE
    // permission to be granted; if the user grants from the settings
    // screen and returns, onResume retries.
    private var docsInitialized = false
    // Modal "permission required" dialog. Held so onResume can dismiss
    // it once permission is granted.
    private var storagePermissionDialog: android.app.AlertDialog? = null
    private var currentDocName: String = ""

    // ---- Brush + vector width (single slider routes to the right one) --
    private var brushSizeScale = 1.0f
    private var brushPreviewEnabled = false
    private lateinit var brushPreviewView: BrushPreviewView
    // Motion-prediction toggle. On by default — prediction is the
    // validated low-latency path; the toggle is kept so we can drop
    // back to raw pen tracking if an edge case appears. Not persisted
    // across launches (intentional — every session starts on).
    private var predictionEnabled = true
    private var vectorLineWidth = 2.0f
    private var brushAlpha = 1.0f
    private var brushHardness = 1.0f
    // When true, overlapping dabs within a stroke clamp to brushAlpha
    // instead of building up — useful for shading with low-alpha colors
    // (see Clip Studio's default behavior). Persisted; defaults to off
    // so existing behavior is preserved for users who upgrade.
    private var strokeUniformAlpha = false
    // Pressure saturation in (0, 1]. Slider value / 100 — see
    // DrawingSurfaceView.brushPressureSaturation for the mapping
    // semantics. 1.0 = full pen range (default), 0.5 = pen maxes out
    // at half its raw range, etc.
    private var brushPressureSaturation = 1.0f
    private var bucketBleed = 2
    private val kBucketBleedMax = 16
    private val kPrefBrushSize     = "brush_size_scale"
    private val kPrefVectorWidth   = "vector_line_width"
    private val kPrefStylusOnly    = "stylus_only"
    private val kPrefPixelGrid     = "pixel_grid_enabled"
    private val kPrefBrushAlpha    = "brush_alpha"
    private val kPrefStrokeUniformAlpha = "stroke_uniform_alpha"
    private val kPrefBrushHardness = "brush_hardness"
    private val kPrefBrushPressure = "brush_pressure_sat"
    private val kPrefBucketBleed   = "bucket_bleed"
    private val kBrushSizeMin = 0.25f
    private val kBrushSizeMax = 4.0f
    // Mirror of renderer.cpp's kMaxRadius (the brush's max dab radius
    // at full pressure, in doc-px). Used by the brush-preview overlay
    // to compute its on-screen circle. Keep in sync if the native
    // value changes.
    private val kBrushMaxRadiusDoc = 18.0f
    private val kVectorWidthMin = 0.5f
    private val kVectorWidthMax = 16.0f
    // Tool-rail tile geometry. Sized so every tool fits on screen at
    // once — see toolTile() for the height budget. Adding a rail entry
    // eats ~40dp of the remaining headroom.
    private val kRailTileSizeDp = 38
    private val kRailTilePadDp  = 6
    // Brush-alpha mapping: invert the dab-accumulation curve so the
    // slider position equals target *stroke* opacity (not per-dab α).
    // A stroke at position (x, y) is roughly N overlapping dabs deep
    // (kSpacing = 0.18 × radius → ~1/0.18 ≈ 5.5 overlap at the
    // stroke spine). With per-dab α and premultiplied-over compose,
    // cumulative coverage = 1 - (1 - α)^N. Inverting:
    //     α(target) = 1 - (1 - target)^(1/N)
    // gives a per-dab α whose stroke result matches `target`. So
    // slider=50 produces a 50%-opaque-looking stroke instead of one
    // that's nearly identical to 100% (which is what the prior
    // mappings gave because dab accumulation saturates quickly).
    private val kAlphaDabsPerOverlap = 6.0

    // 32-cell default palette — 4 rows × 8 columns. Reads top-to-bottom as
    // a value gradient (deep → standard → mid-tint → pale) with a
    // consistent warm-to-cool hue order across each row. Same drafting-
    // table family as the original 16; rows 1 and 3 are new.
    //   Row 1 — deepest hues          (heavy ink, dense fills)
    //   Row 2 — standard chromatics   (original row 1)
    //   Row 3 — mid-light tints       (muted, mid-saturation)
    //   Row 4 — pale neutrals + tints (original row 2)
    // Column 1 is always a grayscale anchor (black → ink → mid gray → paper).
    private val palette = intArrayOf(
        // Row 1 — deepest
        0x000000, 0x6E2218, 0x8A3F0F, 0x886C18,
        0x3D5E26, 0x195049, 0x1A3D60, 0x4A2A65,
        // Row 2 — standard chromatics
        0x1A1A1A, 0xB5482E, 0xC77A1F, 0xC8A030,
        0x5A8C3A, 0x2F7E78, 0x2A5D8F, 0x6B3A8A,
        // Row 3 — mid-light tints
        0x6E6457, 0xC07A60, 0xD0A270, 0xCAB870,
        0x95B070, 0x6FA59E, 0x6F95C0, 0xA088B5,
        // Row 4 — pale neutrals + tints
        0xFFFFFF, 0x7A7368, 0xA89E8A, 0xD9CFB8,
        0xF2D89A, 0xF2A48F, 0x9DB8D8, 0xC7D2A8
    )
    private var currentColorRgb = palette[0]
    // Tracked so the picker / palette can render the Photoshop-style
    // active+previous swatch pair. Updated whenever the active color
    // changes via setActiveColor().
    private var previousColorRgb = palette[1]

    // LRU recents (max 8) — filled implicitly every time the user picks a
    // color. mySlots are user-curated (long-press a slot to set it to the
    // current active color). Both are persisted in SharedPreferences.
    private val recentColors = ArrayDeque<Int>()
    private val kRecentsMax  = 8
    // null entry = empty slot (renders as a dashed +).
    private val mySlots: Array<Int?> = arrayOf(0x2F7E78, 0x9DB8D8, null, null)
    private val kMySlotsCount = 4
    private val kPrefRecents  = "color_recents"
    private val kPrefMySlots  = "color_myslots"

    // ---- Fonts (downloadable) ------------------------------------------
    private var fontMono:          Typeface? = null
    private var fontMonoSemibold:  Typeface? = null
    private var fontInter:         Typeface? = null
    private var fontInterSemibold: Typeface? = null

    // Image-import flow. Registered in onCreate; the contract returns the
    // selected URI or null on cancel. The Activity Result API requires
    // the launcher to be created before STARTED, so we bind it here.
    private lateinit var imagePickerLauncher:  ActivityResultLauncher<String>
    private lateinit var canvasPngLauncher:    ActivityResultLauncher<String>
    private lateinit var documentPdfLauncher:  ActivityResultLauncher<String>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        imagePickerLauncher = registerForActivityResult(
            ActivityResultContracts.GetContent()
        ) { uri: Uri? ->
            if (uri != null) decodeAndImportImage(uri)
        }
        canvasPngLauncher = registerForActivityResult(
            ActivityResultContracts.CreateDocument("image/png")
        ) { uri: Uri? ->
            if (uri != null) exportActivePageToPng(uri)
        }
        documentPdfLauncher = registerForActivityResult(
            ActivityResultContracts.CreateDocument("application/pdf")
        ) { uri: Uri? ->
            if (uri != null) exportDocumentToPdf(uri)
        }
        loadFonts()

        // Document init is deferred to ensureStoragePermissionThenInitDocs()
        // at the end of onCreate — documents live in /sdcard/Drafting Table/
        // now, which requires the MANAGE_EXTERNAL_STORAGE permission.

        // Restore brush size + vector width + opacity and push to native.
        brushSizeScale = prefs().getFloat(kPrefBrushSize, 1.0f)
            .coerceIn(kBrushSizeMin, kBrushSizeMax)
        vectorLineWidth = prefs().getFloat(kPrefVectorWidth, 2.0f)
            .coerceIn(kVectorWidthMin, kVectorWidthMax)
        brushAlpha = prefs().getFloat(kPrefBrushAlpha, 1.0f)
            .coerceIn(0.0f, 1.0f)
        brushHardness = prefs().getFloat(kPrefBrushHardness, 1.0f)
            .coerceIn(0.0f, 1.0f)
        brushPressureSaturation = prefs().getFloat(kPrefBrushPressure, 1.0f)
            .coerceIn(0.0f, 1.0f)
        bucketBleed = prefs().getInt(kPrefBucketBleed, 2)
            .coerceIn(0, kBucketBleedMax)
        strokeUniformAlpha = prefs().getBoolean(kPrefStrokeUniformAlpha, false)
        NativeRenderer.setBrushSize(brushSizeScale)
        NativeRenderer.setVectorLineWidth(vectorLineWidth)
        NativeRenderer.setBrushAlpha(brushAlpha)
        NativeRenderer.setBrushHardness(brushHardness)
        NativeRenderer.setBucketBleed(bucketBleed)
        NativeRenderer.setStrokeUniformAlpha(strokeUniformAlpha)
        // Palm-rejection mode persists across launches; defaults on so
        // accidental finger touches don't draw out of the box.
        stylusOnly = prefs().getBoolean(kPrefStylusOnly, true)
        pixelGridEnabled = prefs().getBoolean(kPrefPixelGrid, false)
        NativeRenderer.setPixelGridEnabled(pixelGridEnabled)
        // Color picker state — recents + my-slots. Stored as comma-
        // separated 6-digit hex. "" entries in mySlots persist as empty.
        loadColorState()

        WindowCompat.setDecorFitsSystemWindows(window, false)
        WindowInsetsControllerCompat(window, window.decorView).apply {
            hide(WindowInsetsCompat.Type.systemBars())
            systemBarsBehavior =
                WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
        }

        // Outer column: main horizontal row (rail + sidebar + panel + canvas)
        // on top, status bar pinned across the bottom.
        val outer = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(getColor(R.color.paper))
        }

        val mainRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
        }

        mainRow.addView(buildToolRail(),
            LinearLayout.LayoutParams(56.dp, ViewGroup.LayoutParams.MATCH_PARENT))

        // Body: a single FrameLayout that holds the SurfaceView at full
        // width plus an overlay row of [sidebar | layerPanel] in front.
        // Keeping the SurfaceView at a fixed size avoids the black flash
        // that the framework shows for one frame when its EGL surface is
        // recreated on resize. Toggling the sidebar now shrinks/expands
        // the overlay's width while the GL surface beneath is untouched.
        val bodyContainer = FrameLayout(this).apply {
            setBackgroundColor(getColor(R.color.bezel))
        }

        val canvas = DrawingSurfaceView(this).also { v ->
            v.onToolChanged = { tool -> onToolChanged(tool) }
            v.onThumbnailsUpdated = { onThumbnailsUpdated() }
            v.stylusOnlyDrawing = stylusOnly
            v.onSurfaceFirstSize = { w, h -> applyInitialPageBounds(w, h) }
            v.onUndoRequested = { userUndo() }
            v.onSnapToggleRequested = { toggleSnap() }
            v.onAngleSnapChanged = { refreshAngleSnapStatus() }
            v.onLineAngleChanged = { deg -> showLineAngle(deg) }
            v.onPenPosition = { x, y, _ -> updateBrushPreview(x, y) }
            v.onPenLeft     = { brushPreviewView.hide() }
            v.brushPressureSaturation = brushPressureSaturation
            bodyContainer.addView(
                v,
                FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT
                )
            )
        }
        drawingView = canvas

        // Brush preview overlay — sits ON TOP of the SurfaceView in
        // bodyContainer so its circle outline draws above the canvas.
        // isClickable/isFocusable are false in BrushPreviewView so
        // touches fall through to the SurfaceView underneath.
        brushPreviewView = BrushPreviewView(this)
        bodyContainer.addView(
            brushPreviewView,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
        )

        // panelsRow itself is NOT clickable — instead each child (sidebar,
        // layer panel) consumes touches inside its own bounds. That way
        // when the layer panel collapses to WRAP_CONTENT height, the empty
        // area below it falls through to the SurfaceView at the back of
        // bodyContainer and the user gets canvas room down there.
        sidebarScroll = buildPageSidebar().apply { isClickable = true }
        val panelsRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
        }
        panelsRow.addView(sidebarScroll,
            LinearLayout.LayoutParams(kSidebarWidthDp.dp,
                ViewGroup.LayoutParams.MATCH_PARENT))
        // Layer panel height is WRAP_CONTENT so collapsing sections (LAYERS
        // / COLOR via the rail tiles) frees up canvas area below the panel.
        // gravity=TOP on the LinearLayout keeps it pinned to the top of
        // its slot so the canvas reveal happens at the bottom.
        panelsRow.addView(buildLayerPanel().apply { isClickable = true },
            LinearLayout.LayoutParams(180.dp, ViewGroup.LayoutParams.WRAP_CONTENT))
        bodyContainer.addView(panelsRow, FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT,
            ViewGroup.LayoutParams.MATCH_PARENT,
            Gravity.START))

        // Undo/redo chip floats at the visible canvas's top-left, which
        // is panelsRow.right + 16dp. The marginStart updates whenever
        // panelsRow's width changes (sidebar toggle).
        val undoChip = buildUndoRedoChip()
        val undoChipParams = undoRedoChipParams()
        bodyContainer.addView(undoChip, undoChipParams)
        panelsRow.addOnLayoutChangeListener { _, _, _, right, _, _, _, oldRight, _ ->
            if (right != oldRight) {
                val lp = undoChip.layoutParams as FrameLayout.LayoutParams
                lp.marginStart = right + 16.dp
                undoChip.layoutParams = lp
                // resetView uses this to fit the page into the
                // panel-free portion of the canvas.
                drawingView?.visibleLeftInset = right
            }
        }

        mainRow.addView(
            bodyContainer,
            LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1.0f)
        )

        outer.addView(
            mainRow,
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1.0f
            )
        )
        outer.addView(buildStatusBar(),
            LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 28.dp))

        setContentView(outer)

        // Opt into the highest refresh rate the panel offers (90 Hz on the MovinkPad).
        val highest = display?.supportedModes?.maxByOrNull { it.refreshRate }
        if (highest != null) {
            val attrs = window.attributes
            attrs.preferredDisplayModeId = highest.modeId
            window.attributes = attrs
        }
        // One-shot log of available refresh modes + the one we asked for.
        // Useful for the latency probe — confirms whether we're actually
        // running at 90 Hz on this device (frame time ≈ 11.1 ms vs the
        // 16.7 ms 60 Hz fallback).
        display?.let { d ->
            val modes = d.supportedModes.joinToString(", ") {
                "${it.modeId}:${it.physicalWidth}x${it.physicalHeight}@${it.refreshRate}"
            }
            Log.i("DrawingApp",
                "DISPLAY currentRate=${d.refreshRate}  preferred=${highest?.modeId}  modes=[$modes]")
        }

        // Apply initial selection state to the brush tile.
        onToolChanged(Tool.BRUSH)
        // Push the persisted color into native and reflect it in the chip.
        NativeRenderer.setBrushColor(currentColorRgb)
        updateColorChip()

        // Gate the rest of the document init on the
        // MANAGE_EXTERNAL_STORAGE permission. If granted, picks the
        // initial doc and rebuilds the sidebar; otherwise shows a
        // modal dialog and retries from onResume after the user comes
        // back from settings.
        ensureStoragePermissionThenInitDocs()
    }

    override fun onResume() {
        super.onResume()
        // User may have just returned from the system settings page
        // after granting "All files access". Retry doc init.
        if (!docsInitialized
            && android.os.Environment.isExternalStorageManager()) {
            storagePermissionDialog?.dismiss()
            storagePermissionDialog = null
            initializeDocuments()
        }
    }

    override fun onPause() {
        // Tile writes are queued onto a background thread for the
        // duration of a session; force them to disk before we hand
        // control back to the system so a process kill (low-memory
        // reaper, user task swipe) doesn't lose the last few strokes.
        //
        // The forceRedraw triggers one final no-stroke MB pass which
        // drains commitStroke's deferred glReadPixels queue on the GL
        // thread. This is best-effort — if the system tears the GL
        // thread down before that pass completes, the recently
        // deferred saves may not make it to disk. Normal back/home
        // transitions give plenty of time.
        drawingView?.forceRedraw()
        NativeRenderer.flushTileWrites()
        super.onPause()
    }

    override fun onDestroy() {
        drawingView?.release()
        drawingView = null
        super.onDestroy()
    }

    /** Volume keys cycle/add layers; stylus side-button cycles tools. */
    override fun dispatchKeyEvent(event: KeyEvent): Boolean {
        val keyCode = event.keyCode
        when (keyCode) {
            KeyEvent.KEYCODE_VOLUME_UP, KeyEvent.KEYCODE_VOLUME_DOWN -> {
                if (event.action == KeyEvent.ACTION_DOWN) {
                    if (keyCode == KeyEvent.KEYCODE_VOLUME_UP) userCycleLayer()
                    else                                       userAddLayer()
                }
                return true
            }
            KeyEvent.KEYCODE_STYLUS_BUTTON_PRIMARY,
            KeyEvent.KEYCODE_STYLUS_BUTTON_SECONDARY,
            KeyEvent.KEYCODE_STYLUS_BUTTON_TERTIARY -> {
                if (event.action == KeyEvent.ACTION_DOWN) {
                    when (keyCode) {
                        // Closest-to-nib button: toggle snap.
                        KeyEvent.KEYCODE_STYLUS_BUTTON_PRIMARY   -> toggleSnap()
                        // Middle button: brush ⇄ eraser toggle.
                        KeyEvent.KEYCODE_STYLUS_BUTTON_SECONDARY -> drawingView?.toggleTool()
                        // Furthest-from-nib button: undo.
                        KeyEvent.KEYCODE_STYLUS_BUTTON_TERTIARY  -> userUndo()
                    }
                }
                return true
            }
        }
        return super.dispatchKeyEvent(event)
    }

    // ====================================================================
    // Tool rail
    // ====================================================================

    /** Vertical rail of icon tiles. Sections separated by hairline rules.
     *  Wrapped in a ScrollView so it gracefully degrades on shorter screens. */
    private fun buildToolRail(): View {
        val scroll = ScrollView(this).apply {
            isVerticalScrollBarEnabled = false
            overScrollMode = View.OVER_SCROLL_NEVER
            setBackgroundColor(getColor(R.color.paperDeep))
        }
        val rail = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER_HORIZONTAL
            setPadding(0, 4.dp, 0, 4.dp)
        }
        // Right-edge hairline so the rail visually separates from the
        // sidebar / layer panel even when paperDeep is the same tone.
        rail.addView(View(this).apply {
            setBackgroundColor(getColor(R.color.rule))
        }, LinearLayout.LayoutParams(0, 0))   // placeholder, replaced below

        // Menu (top of rail): opens overflow popover with rarely-used actions.
        rail.addView(toolTile(R.drawable.ic_menu, "menu", isToggle = false) {
            showOverflowMenu(it)
        })
        rail.addView(railRule())

        rail.addView(railSectionLabel("DRAW"))
        rail.addView(makeRailToolTile(Tool.BRUSH,   R.drawable.ic_pen,    "brush"))
        rail.addView(makeRailToolTile(Tool.ERASER,  R.drawable.ic_eraser, "eraser"))
        rail.addView(makeRailToolTile(Tool.BUCKET,  R.drawable.ic_bucket, "bucket"))
        rail.addView(makeRailToolTile(Tool.SHADE,   R.drawable.ic_shade,  "shade"))

        rail.addView(railRule())
        rail.addView(railSectionLabel("VECTOR"))
        rail.addView(makeRailToolTile(Tool.LINE,        R.drawable.ic_line,          "line"))
        rail.addView(makeRailToolTile(Tool.RECTANGLE,   R.drawable.ic_rect,          "rectangle"))
        rail.addView(makeRailToolTile(Tool.CIRCLE,      R.drawable.ic_circle,        "circle"))
        rail.addView(makeRailToolTile(Tool.ELLIPSE,     R.drawable.ic_ellipse,       "ellipse"))

        rail.addView(railRule())
        rail.addView(railSectionLabel("SELECT"))
        // The rect/marquee tile dispatches by active layer type:
        // raster → floating-selection lift, vector → tap-to-select +
        // marquee multi-select. The dedicated "vector select" tile was
        // collapsed into this one to free up rail space.
        rail.addView(makeRailToolTile(Tool.SELECT_RECT, R.drawable.ic_select,        "select"))
        rail.addView(makeRailToolTile(Tool.SELECT_LASSO,R.drawable.ic_lasso,         "lasso"))

        rail.addView(railRule())
        // Panel toggles — each flips visibility of one section in the
        // layer column or the page sidebar. The selected state on the
        // tile mirrors whether its section is currently shown, so the
        // rail reads at a glance.
        layersRailTile = toolTile(R.drawable.ic_layers, "layers", isToggle = false) { tile ->
            toggleLayersSection()
            tile.isSelected = layersSection.visibility == View.VISIBLE
        } as ImageView
        layersRailTile.isSelected = true
        rail.addView(layersRailTile)
        pagesRailTile = toolTile(R.drawable.ic_pages, "pages", isToggle = false) { tile ->
            togglePageSidebar()
            tile.isSelected = sidebarScroll.visibility == View.VISIBLE
        } as ImageView
        pagesRailTile.isSelected = true
        rail.addView(pagesRailTile)
        colorRailTile = toolTile(R.drawable.ic_color, "color", isToggle = false) { tile ->
            toggleColorSection()
            tile.isSelected = colorSection.visibility == View.VISIBLE
        } as ImageView
        colorRailTile.isSelected = true
        rail.addView(colorRailTile)
        // Grid + snap have moved to the bottom status bar (their text
        // labels there double as toggleable buttons), keeping the rail
        // tight enough that no scrolling is needed for the tools.

        // View controls live below the panel toggles, separated by a
        // rule so they read as their own group (not part of the panel
        // selectors above).
        rail.addView(railRule())
        rail.addView(toolTile(R.drawable.ic_reset_view, "reset view",
            isToggle = false) { drawingView?.resetView() })

        scroll.addView(rail, FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ))
        return scroll
    }

    private fun railRule(): View = View(this).apply {
        setBackgroundColor(getColor(R.color.rule))
        layoutParams = LinearLayout.LayoutParams(30.dp, 1.dp).apply {
            topMargin = 3.dp; bottomMargin = 3.dp
        }
    }

    private fun railSectionLabel(text: String): TextView = TextView(this).apply {
        this.text = text
        textSize = 8f
        typeface = fontMonoSemibold ?: Typeface.MONOSPACE
        letterSpacing = 0.12f
        setTextColor(getColor(R.color.inkFaint))
        gravity = Gravity.CENTER
        setPadding(0, 1.dp, 0, 1.dp)
        layoutParams = LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
    }

    private fun makeRailToolTile(tool: Tool, iconRes: Int, label: String): ImageView {
        val tile = toolTile(iconRes, label, isToggle = false) { drawingView?.setTool(tool) }
                as ImageView
        railToolTiles[tool] = tile
        return tile
    }

    /**
     * Generic icon tile. `isToggle` toggles `isSelected` on tap;
     * non-toggle tiles fire `onClick` and the caller manages state.
     *
     * Geometry is sized so the whole rail fits the MovinkPad's 822dp
     * landscape height without scrolling — a rail you have to scroll
     * hides tools and costs a gesture mid-sketch. The 26dp content box
     * still exceeds the icons' 24dp intrinsic size, so trimming the
     * tile trims dead space only; CENTER_INSIDE never upscales, so the
     * glyphs render at exactly the same size they did at 44dp.
     * Budget: 15 tiles x 40dp pitch + rules/labels ~= 685dp.
     */
    private fun toolTile(iconRes: Int, label: String,
                         isToggle: Boolean,
                         onClick: (ImageView) -> Unit): View {
        val tile = ImageView(this).apply {
            setImageResource(iconRes)
            scaleType = ImageView.ScaleType.CENTER_INSIDE
            background = ResourcesCompat.getDrawable(
                resources, R.drawable.tool_tile_bg, theme)
            setPadding(kRailTilePadDp.dp, kRailTilePadDp.dp,
                       kRailTilePadDp.dp, kRailTilePadDp.dp)
            contentDescription = label
            imageTintList = ColorStateList(
                arrayOf(intArrayOf(android.R.attr.state_selected),
                        intArrayOf()),
                intArrayOf(getColor(R.color.hot), getColor(R.color.ink))
            )
            isClickable = true
            isFocusable = true
            setOnClickListener {
                if (isToggle) isSelected = !isSelected
                onClick(this)
            }
        }
        tile.layoutParams = LinearLayout.LayoutParams(
            kRailTileSizeDp.dp, kRailTileSizeDp.dp).apply {
            topMargin = 1.dp; bottomMargin = 1.dp
        }
        return tile
    }

    /** Reflect the active tool by toggling `isSelected` on the rail tiles. */
    private fun onToolChanged(tool: Tool) {
        for ((t, tile) in railToolTiles) tile.isSelected = (t == tool)
        currentToolMirror = tool
        if (::statusToolText.isInitialized) {
            statusToolText.text = "◇ ${tool.displayName.lowercase()}"
        }
        if (::sizeSliderLabel.isInitialized) {
            updateSizeSliderForTool()
        }
        // Brush preview only makes sense for raster stroke tools; hide
        // immediately on switch so a stale circle doesn't linger until
        // the next pen event.
        if (::brushPreviewView.isInitialized && !tool.isRasterStroke) {
            brushPreviewView.hide()
        }
    }

    private val Tool.displayName: String
        get() = when (this) {
            Tool.BRUSH        -> "brush"
            Tool.ERASER       -> "eraser"
            Tool.BUCKET       -> "bucket"
            Tool.SHADE        -> "shade"
            Tool.LINE         -> "line"
            Tool.RECTANGLE    -> "rect"
            Tool.CIRCLE       -> "circle"
            Tool.ELLIPSE      -> "ellipse"
            Tool.SELECT       -> "select"
            Tool.SELECT_RECT  -> "select"   // unified raster + vector
            Tool.SELECT_LASSO -> "lasso"
        }

    // ====================================================================
    // Layer panel: LAYERS + BRUSH + COLOR
    // ====================================================================

    private fun buildLayerPanel(): View {
        // Right-edge hairline is baked into the panel's background as a
        // LayerDrawable (paper layer + 1dp rule layer inset to the right
        // edge) instead of a separate sibling View. The previous design
        // used a FrameLayout wrapper with a MATCH_PARENT rule view, but
        // the rule's MATCH_PARENT height resolved to the full body height
        // even when the wrapper was WRAP_CONTENT — leaking a thin vertical
        // line down into the canvas area. With the rule as part of the
        // background, it naturally clips to the panel's WRAP_CONTENT
        // height as sections collapse.
        val panel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            background = android.graphics.drawable.LayerDrawable(arrayOf(
                android.graphics.drawable.ColorDrawable(getColor(R.color.paper)),
                android.graphics.drawable.ColorDrawable(getColor(R.color.rule))
            )).apply {
                // Inset the rule layer from the left so only a 1dp strip
                // along the right edge remains visible.
                setLayerInset(1, 180.dp - 1.dp.coerceAtLeast(1), 0, 0, 0)
            }
        }
        // The wrapper is now a no-op pass-through (kept so callers still
        // get a View) — left in place for symmetry with the rest of the
        // layer-panel build pipeline.
        val wrapper = FrameLayout(this)
        wrapper.addView(panel, FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ))

        // ---- LAYERS section (collapsible via the rail's layers tile) ----
        // Wrapped in its own column so toggling visibility takes the
        // header + list + opacity slider + trailing divider down together.
        val layersWrap = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
        }
        layersWrap.addView(buildLayerHeader(),
            LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 28.dp))
        layerListContainer = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
        }
        layersWrap.addView(layerListContainer,
            LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT))
        // Active-layer opacity slider, sitting just under the layer list
        // so it visually belongs to the LAYERS section rather than the
        // BRUSH section below.
        layersWrap.addView(buildLayerOpacityRow(),
            LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT))
        layersWrap.addView(panelDivider())
        layersSection = layersWrap
        panel.addView(layersWrap, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT))

        // ---- BRUSH section (always shown) ----
        panel.addView(buildBrushSection(),
            LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT))

        // ---- COLOR section (collapsible via the rail's color tile) ----
        // Wrapped together with the divider above it so toggling cleanly
        // removes both the rule and the section content.
        val colorWrap = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
        }
        colorWrap.addView(panelDivider())
        colorWrap.addView(buildColorSection(),
            LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT))
        colorSection = colorWrap
        panel.addView(colorWrap, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT))

        return wrapper
    }

    private fun panelDivider(): View = View(this).apply {
        setBackgroundColor(getColor(R.color.rule))
        layoutParams = LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, 1.dp)
    }

    private fun buildLayerHeader(): View {
        val header = FrameLayout(this).apply {
            background = ResourcesCompat.getDrawable(
                resources, R.drawable.panel_section_header, theme)
        }
        layerActiveHeading = TextView(this).apply {
            text = "LAYERS"
            typeface = fontMonoSemibold ?: Typeface.MONOSPACE
            textSize = 10f
            letterSpacing = 0.08f
            setTextColor(getColor(R.color.ink))
            gravity = Gravity.CENTER_VERTICAL
            setPadding(10.dp, 0, 0, 0)
        }
        header.addView(layerActiveHeading, FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT,
            ViewGroup.LayoutParams.MATCH_PARENT,
            Gravity.START or Gravity.CENTER_VERTICAL
        ))

        // + raster layer
        val plus = ImageView(this).apply {
            setImageResource(R.drawable.ic_plus)
            imageTintList = ColorStateList.valueOf(getColor(R.color.inkSoft))
            setPadding(8.dp, 4.dp, 4.dp, 4.dp)
            contentDescription = "add raster layer"
            isClickable = true; isFocusable = true
            setOnClickListener { userAddLayer() }
        }
        // + vector layer — promoted out of the ⋯ overflow so it's a
        // first-class action alongside raster. Padding mirrors `plus`
        // so both icons get the same image-area size, otherwise the
        // identical SVG paths inside render at different scales.
        val plusVector = ImageView(this).apply {
            setImageResource(R.drawable.ic_plus_vector)
            imageTintList = ColorStateList.valueOf(getColor(R.color.inkSoft))
            setPadding(8.dp, 4.dp, 4.dp, 4.dp)
            contentDescription = "add vector layer"
            isClickable = true; isFocusable = true
            setOnClickListener { userAddVectorLayer() }
        }
        // ⋯ overflow → clear / delete (vector add lives next to the
        // raster +; the menu still keeps it as a backup option).
        val more = ImageView(this).apply {
            setImageResource(R.drawable.ic_more)
            imageTintList = ColorStateList.valueOf(getColor(R.color.inkSoft))
            setPadding(4.dp, 4.dp, 8.dp, 4.dp)
            contentDescription = "more"
            isClickable = true; isFocusable = true
            setOnClickListener { showLayerOverflow(it) }
        }
        val rightRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            addView(plus,       LinearLayout.LayoutParams(28.dp, 28.dp))
            addView(plusVector, LinearLayout.LayoutParams(28.dp, 28.dp))
            addView(more,       LinearLayout.LayoutParams(32.dp, 28.dp))
        }
        header.addView(rightRow, FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT,
            ViewGroup.LayoutParams.MATCH_PARENT,
            Gravity.END or Gravity.CENTER_VERTICAL
        ))
        return header
    }

    /** Re-render the layer list. Called whenever layer state changes. */
    private fun rebuildLayerList() {
        layerListContainer.removeAllViews()
        // Rows have a fixed height: the active row's MATCH_PARENT sienna
        // left bar would otherwise fight WRAP_CONTENT and inflate the
        // whole row to fill the panel. List is top-down: highest-index
        // (visually-topmost) layer first.
        for (idx in (layerCount - 1) downTo 0) {
            layerListContainer.addView(buildLayerRow(idx),
                LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT, 32.dp))
            layerListContainer.addView(View(this).apply {
                setBackgroundColor(getColor(R.color.rule))
            }, LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 1.dp))
        }
        // Sync the opacity slider to whatever the active layer is now.
        refreshLayerOpacitySlider()
    }

    private fun buildLayerRow(idx: Int): View {
        val isActive  = (idx == activeLayerIndex)
        val isVector  = NativeRenderer.getLayerType(idx) == 1
        val isVisible = NativeRenderer.getLayerVisible(idx)
        val customName = NativeRenderer.getLayerName(idx)
        val displayName = if (customName.isNotEmpty()) customName
                          else (if (isVector) "vector ${idx + 1}" else "layer ${idx + 1}")

        val row = FrameLayout(this).apply {
            setBackgroundColor(
                if (isActive) getColor(R.color.paperDeep) else Color.TRANSPARENT)
            // 2dp sienna left bar for active; transparent otherwise.
            if (isActive) {
                addView(View(this@MainActivity).apply {
                    setBackgroundColor(getColor(R.color.hot))
                }, FrameLayout.LayoutParams(2.dp,
                    ViewGroup.LayoutParams.MATCH_PARENT))
            }
            isClickable = true; isFocusable = true
            setOnClickListener {
                // Cycle to this layer. Native cycleActiveLayer goes one at
                // a time, so we cycle until we land on idx.
                while (activeLayerIndex != idx) userCycleLayer()
            }
            setOnLongClickListener {
                showRenameLayerDialog(idx); true
            }
        }
        val rowContent = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(10.dp, 0, 10.dp, 0)
            minimumHeight = 28.dp
        }
        // Eye icon — toggles layer visibility. Has its own click handler
        // so taps here don't bubble up to the row's set-active listener.
        // Hidden layers get a faint tint to read as inactive at a glance.
        val eye = ImageView(this).apply {
            setImageResource(if (isVisible) R.drawable.ic_eye else R.drawable.ic_eye_off)
            imageTintList = ColorStateList.valueOf(
                getColor(if (isVisible) R.color.ink else R.color.inkFaint))
            scaleType = ImageView.ScaleType.CENTER_INSIDE
            isClickable = true; isFocusable = true
            setOnClickListener {
                NativeRenderer.setLayerVisible(idx, !isVisible)
                drawingView?.forceRedraw()
                rebuildLayerList()
            }
        }
        rowContent.addView(eye, LinearLayout.LayoutParams(20.dp, 20.dp).apply {
            rightMargin = 8.dp
        })
        val name = TextView(this).apply {
            text = displayName
            typeface = if (isActive) fontMonoSemibold ?: Typeface.MONOSPACE
                       else fontMono ?: Typeface.MONOSPACE
            textSize = 11f
            setTextColor(getColor(R.color.ink))
            ellipsize = android.text.TextUtils.TruncateAt.END
            maxLines = 1
        }
        rowContent.addView(name, LinearLayout.LayoutParams(
            0, ViewGroup.LayoutParams.WRAP_CONTENT, 1.0f))
        val tag = TextView(this).apply {
            text = if (isVector) "V" else "R"
            typeface = fontMono ?: Typeface.MONOSPACE
            textSize = 8f
            setTextColor(getColor(R.color.inkFaint))
            setPadding(4.dp, 1.dp, 4.dp, 1.dp)
            background = ResourcesCompat.getDrawable(
                resources, R.drawable.chrome_chip_bg, theme)
        }
        rowContent.addView(tag)

        // Per-row ⋯ overflow → rename / delete (move is via drag handle).
        // Has its own click handler so taps here don't bubble to the
        // row's "set active" listener.
        val rowMore = ImageView(this).apply {
            setImageResource(R.drawable.ic_more)
            imageTintList = ColorStateList.valueOf(getColor(R.color.inkSoft))
            scaleType = ImageView.ScaleType.CENTER_INSIDE
            isClickable = true; isFocusable = true
            setOnClickListener { showLayerRowOverflow(it, idx) }
        }
        rowContent.addView(rowMore, LinearLayout.LayoutParams(20.dp, 20.dp).apply {
            leftMargin = 6.dp
        })

        // Drag handle for reorder. Captures touch on DOWN and grabs the
        // row's parent (FrameLayout) so we can translate the entire row
        // as the user drags. Other rows shift aside in updateDragRowDisplacement.
        val dragHandle = ImageView(this).apply {
            setImageResource(R.drawable.ic_drag_handle)
            imageTintList = ColorStateList.valueOf(getColor(R.color.inkSoft))
            scaleType = ImageView.ScaleType.CENTER_INSIDE
            contentDescription = "drag to reorder"
            isClickable = true; isFocusable = true
        }
        installDragHandleListener(dragHandle, idx, row)
        rowContent.addView(dragHandle, LinearLayout.LayoutParams(20.dp, 28.dp).apply {
            leftMargin = 4.dp
        })

        row.addView(rowContent, FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT,
            Gravity.CENTER_VERTICAL))
        return row
    }

    /** Per-row overflow menu: rename / rasterize / merge-down / delete.
     *  Reordering is done with the drag handle; Delete is disabled when
     *  only one layer remains. "Merge with layer below" is shown for
     *  raster layers when there's a raster layer beneath; greyed-out
     *  otherwise so the menu shape stays stable. */
    private fun showLayerRowOverflow(anchor: View, idx: Int) {
        val isVector = NativeRenderer.getLayerType(idx) == 1
        val belowIsRaster = idx > 0 &&
            NativeRenderer.getLayerType(idx - 1) == 0
        showPaperPopupMenu(anchor, listOf(
            PaperMenuItem("Rename…")              { showRenameLayerDialog(idx) },
            // Rasterize is only meaningful for vector layers; greyed
            // out otherwise so the menu shape stays stable.
            PaperMenuItem("Rasterize layer", enabled = isVector)
                                                  { userRasterizeLayer(idx) },
            // Merge-with-below requires source AND target to be raster.
            PaperMenuItem("Merge with layer below",
                          enabled = !isVector && belowIsRaster)
                                                  { userMergeLayerWithBelow(idx) },
            PaperMenuItem("Delete", enabled = layerCount > 1)
                                                  { confirmDeleteLayer(idx) },
        ))
    }

    /** Composite the raster layer [idx] onto the raster layer below using
     *  premultiplied "src over dst", then delete the source layer. The
     *  op is undoable (the native impl captures the source layer's full
     *  state plus the target's tile diff). */
    private fun userMergeLayerWithBelow(idx: Int) {
        if (idx <= 0) return
        NativeRenderer.mergeLayerWithBelow(idx)
        drawingView?.forceRedraw()
        drawingView?.postDelayed({ syncLayerStateFromNative() }, 60L)
    }

    /** Bake the vector shapes on layer [idx] into raster tiles in place.
     *  Undoable — the native impl captures the pre-rasterize shape lists
     *  + the post-rasterize tiles in a RasterizeLayer entry. */
    private fun userRasterizeLayer(idx: Int) {
        NativeRenderer.rasterizeLayer(idx)
        drawingView?.forceRedraw()
        drawingView?.postDelayed({ syncLayerStateFromNative() }, 60L)
    }

    /** Confirm-then-delete a layer. There is no undo for layer ops, so
     *  the dialog spells that out before the destructive call. */
    private fun confirmDeleteLayer(idx: Int) {
        val customName = NativeRenderer.getLayerName(idx)
        val isVector = NativeRenderer.getLayerType(idx) == 1
        val displayName = if (customName.isNotEmpty()) customName
                          else (if (isVector) "vector ${idx + 1}" else "layer ${idx + 1}")
        showPaperConfirmDialog(
            title = "Delete layer",
            message = "Delete “$displayName”? This can't be undone.",
            confirmLabel = "Delete",
        ) { userDeleteLayer(idx) }
    }

    private fun userDeleteLayer(idx: Int) {
        if (layerCount <= 1) return
        NativeRenderer.deleteLayer(idx)
        drawingView?.forceRedraw()
        // Native applies the delete on the GL thread next op; sync after a
        // beat so getLayerCount/getActiveLayer reflect the new state.
        drawingView?.postDelayed({ syncLayerStateFromNative() }, 60L)
    }

    private fun userMoveLayer(from: Int, to: Int) {
        if (from == to || from < 0 || to < 0
            || from >= layerCount || to >= layerCount) return
        NativeRenderer.moveLayer(from, to)
        drawingView?.forceRedraw()
        // Active layer index may shift; resync so the highlighted row is
        // correct after the move lands on the GL thread.
        drawingView?.postDelayed({ syncLayerStateFromNative() }, 60L)
    }

    /** Layer rows render top-to-bottom with the highest idx at the top, so
     *  ui-position 0 corresponds to native idx (count-1). This pair maps
     *  between the two so the drag math can stay in ui-position space. */
    private fun uiPosToIdx(uiPos: Int): Int = (layerCount - 1) - uiPos
    private fun idxToUiPos(idx: Int): Int   = (layerCount - 1) - idx

    /** Layer rows are interleaved with 1dp dividers in layerListContainer.
     *  Children at even indices (0, 2, 4, ...) are rows; odd indices are
     *  dividers. */
    private fun rowViewAtUiPos(uiPos: Int): View? {
        if (!::layerListContainer.isInitialized) return null
        val childIdx = uiPos * 2
        if (childIdx < 0 || childIdx >= layerListContainer.childCount) return null
        return layerListContainer.getChildAt(childIdx)
    }

    /** Bind the drag-to-reorder touch handler onto the handle ImageView for
     *  the row at native index [idx]. Handles its own touch stream — no
     *  click listener — so the row's "set active" tap can't fire when the
     *  user is grabbing the handle. */
    private fun installDragHandleListener(handle: View, idx: Int, rowView: View) {
        handle.setOnTouchListener { v, ev ->
            when (ev.actionMasked) {
                MotionEvent.ACTION_DOWN -> {
                    if (layerCount <= 1) return@setOnTouchListener false
                    dragSourceIdx = idx
                    dragSourceView = rowView
                    dragStartRawY = ev.rawY
                    dragStartTranslationY = rowView.translationY
                    // Row + 1dp divider = the per-slot height we use to
                    // displace other rows by integer multiples.
                    dragRowHeightPx = rowView.height + 1.dp
                    dragCurrentTargetUiPos = idxToUiPos(idx)
                    // Float the row above the others while dragging.
                    rowView.elevation = 6.dp.toFloat()
                    rowView.alpha = 0.95f
                    // We need parent disallowInterceptTouchEvent so a
                    // ScrollView ancestor doesn't steal the gesture.
                    v.parent?.requestDisallowInterceptTouchEvent(true)
                    true
                }
                MotionEvent.ACTION_MOVE -> {
                    if (dragSourceIdx != idx || dragSourceView !== rowView) {
                        return@setOnTouchListener false
                    }
                    val dy = ev.rawY - dragStartRawY
                    rowView.translationY = dragStartTranslationY + dy
                    updateDragRowDisplacement(rowView)
                    true
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    if (dragSourceIdx != idx || dragSourceView !== rowView) {
                        return@setOnTouchListener false
                    }
                    val targetUiPos = dragCurrentTargetUiPos
                    val sourceIdx = dragSourceIdx
                    // Reset visual state for every row before either
                    // committing the move (which rebuilds the list) or
                    // bouncing back.
                    rowView.elevation = 0f
                    rowView.alpha = 1.0f
                    rowView.translationY = 0f
                    resetAllRowDisplacements()
                    dragSourceIdx = -1
                    dragSourceView = null
                    dragCurrentTargetUiPos = -1

                    val sourceUiPos = idxToUiPos(sourceIdx)
                    if (targetUiPos in 0 until layerCount && targetUiPos != sourceUiPos) {
                        val targetIdx = uiPosToIdx(targetUiPos)
                        userMoveLayer(sourceIdx, targetIdx)
                    }
                    true
                }
                else -> false
            }
        }
    }

    /** While the user drags [draggedRow], shift the other rows so a visual
     *  insertion gap follows the dragged row. Called on every MOVE event;
     *  also recomputes [dragCurrentTargetUiPos] (used at UP to choose the
     *  destination). */
    private fun updateDragRowDisplacement(draggedRow: View) {
        val sourceIdx = dragSourceIdx
        if (sourceIdx < 0) return
        val sourceUiPos = idxToUiPos(sourceIdx)
        // Center of the dragged row in container-local coords.
        val draggedCenter = draggedRow.top + draggedRow.translationY +
                            draggedRow.height / 2f
        // Target ui-position: the count of *other* rows whose original
        // center lies above the dragged center. This is the stable
        // formulation — independent of which way the user crossed.
        var newUiPos = 0
        for (uiPos in 0 until layerCount) {
            if (uiPos == sourceUiPos) continue
            val row = rowViewAtUiPos(uiPos) ?: continue
            val originalCenter = row.top + row.height / 2f
            if (originalCenter < draggedCenter) newUiPos++
        }
        if (newUiPos == dragCurrentTargetUiPos) return
        dragCurrentTargetUiPos = newUiPos
        // Displace other rows so a gap opens at newUiPos. A row's *display*
        // ui-position after a hypothetical commit determines its translation.
        for (uiPos in 0 until layerCount) {
            if (uiPos == sourceUiPos) continue
            val row = rowViewAtUiPos(uiPos) ?: continue
            val displayedUiPos = when {
                sourceUiPos < newUiPos ->
                    if (uiPos in (sourceUiPos + 1)..newUiPos) uiPos - 1 else uiPos
                sourceUiPos > newUiPos ->
                    if (uiPos in newUiPos..(sourceUiPos - 1)) uiPos + 1 else uiPos
                else -> uiPos
            }
            val target = ((displayedUiPos - uiPos) * dragRowHeightPx).toFloat()
            // Animate so the shifts feel smooth instead of teleporting.
            row.animate().translationY(target).setDuration(120L).start()
        }
    }

    /** Clear translationY on every layer row. Called at the end of a drag
     *  so the visible state matches the (possibly committed) data. */
    private fun resetAllRowDisplacements() {
        if (!::layerListContainer.isInitialized) return
        for (i in 0 until layerListContainer.childCount) {
            layerListContainer.getChildAt(i).translationY = 0f
        }
    }

    /** Paper-styled rename dialog prepopulated with the current layer
     *  name (or empty if none was set). Empty input clears the custom
     *  name and reverts to the "layer N" / "vector N" default. */
    private fun showRenameLayerDialog(idx: Int) {
        val current = NativeRenderer.getLayerName(idx)
        showPaperInputDialog(
            title = "Rename layer",
            initial = current,
            hint = "layer name",
        ) { name ->
            NativeRenderer.setLayerName(idx, name)
            rebuildLayerList()
        }
    }

    /** "α" slider that always edits the active layer's opacity. The
     *  slider's progress is in [0, 100]; the value is divided by 100
     *  before it goes to native. We avoid pushing native writes for
     *  programmatic progress changes (active-layer-changed sync) by
     *  gating on the listener's `fromUser` flag. */
    private fun buildLayerOpacityRow(): View {
        val label = TextView(this).apply {
            text = "α"
            typeface = fontMono ?: Typeface.MONOSPACE
            textSize = 11f
            setTextColor(getColor(R.color.inkSoft))
        }
        layerOpacityValue = TextView(this).apply {
            typeface = fontMono ?: Typeface.MONOSPACE
            textSize = 11f
            setTextColor(getColor(R.color.ink))
            gravity = Gravity.END
            text = "100"
        }
        layerOpacitySlider = SeekBar(this).apply {
            max = 100
            progress = 100
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(sb: SeekBar?, p: Int, fromUser: Boolean) {
                    layerOpacityValue.text = p.toString()
                    if (!fromUser) return
                    NativeRenderer.setLayerOpacity(activeLayerIndex, p / 100f)
                    drawingView?.forceRedraw()
                }
                override fun onStartTrackingTouch(sb: SeekBar?) {}
                override fun onStopTrackingTouch(sb: SeekBar?) {}
            })
        }
        return buildSliderRow(label, layerOpacitySlider, layerOpacityValue)
    }

    /** Push the native opacity value for the currently-active layer
     *  into the slider. Called whenever the active layer changes (or
     *  rebuildLayerList runs) so the slider always reflects the right
     *  layer. fromUser=false on this programmatic set, so the listener
     *  won't echo it back to native. */
    private fun refreshLayerOpacitySlider() {
        if (!::layerOpacitySlider.isInitialized) return
        val o = NativeRenderer.getLayerOpacity(activeLayerIndex)
        layerOpacitySlider.progress = (o * 100f).toInt().coerceIn(0, 100)
    }

    private fun buildBrushSection(): View {
        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
        }
        // Header
        val header = FrameLayout(this).apply {
            background = ResourcesCompat.getDrawable(
                resources, R.drawable.panel_section_header, theme)
        }
        val headerLabel = TextView(this).apply {
            text = "BRUSH"
            typeface = fontMonoSemibold ?: Typeface.MONOSPACE
            textSize = 10f
            letterSpacing = 0.08f
            setTextColor(getColor(R.color.ink))
            gravity = Gravity.CENTER_VERTICAL
            setPadding(10.dp, 0, 10.dp, 0)
        }
        brushSectionHeader = headerLabel
        header.addView(headerLabel, FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT,
            Gravity.START or Gravity.CENTER_VERTICAL))
        container.addView(header,
            LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 28.dp))

        // Active "size" slider routes to brush vs vector width based on tool.
        sizeSliderLabel = TextView(this).apply {
            typeface = fontMono ?: Typeface.MONOSPACE
            textSize = 11f
            setTextColor(getColor(R.color.inkSoft))
            text = "size"
        }
        sizeValueLabel = TextView(this).apply {
            typeface = fontMono ?: Typeface.MONOSPACE
            textSize = 11f
            setTextColor(getColor(R.color.ink))
            gravity = Gravity.END
        }
        sizeSlider = SeekBar(this).apply {
            max = 100
            progress = brushScaleToProgress(brushSizeScale)
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(sb: SeekBar?, p: Int, fromUser: Boolean) {
                    if (!fromUser) return
                    if (currentToolEditsVector()) {
                        val w = progressToVectorWidth(p)
                        vectorLineWidth = w
                        sizeValueLabel.text = "%.1f".format(w)
                        NativeRenderer.setVectorLineWidth(w)
                    } else {
                        val s = progressToBrushScale(p)
                        brushSizeScale = s
                        sizeValueLabel.text = "%.2fx".format(s)
                        NativeRenderer.setBrushSize(s)
                    }
                }
                override fun onStartTrackingTouch(sb: SeekBar?) {}
                override fun onStopTrackingTouch(sb: SeekBar?) {
                    if (currentToolEditsVector()) {
                        prefs().edit().putFloat(kPrefVectorWidth, vectorLineWidth).apply()
                    } else {
                        prefs().edit().putFloat(kPrefBrushSize, brushSizeScale).apply()
                    }
                }
            })
        }
        brushSizeRow = buildSliderRow(sizeSliderLabel, sizeSlider, sizeValueLabel)
        container.addView(brushSizeRow)

        // α slider — controls brush opacity. Active for any tool, but
        // only affects brush strokes (eraser keeps a fixed strength).
        brushAlphaRow = buildBrushAlphaRow()
        container.addView(brushAlphaRow)

        // uniform-α toggle. When on, overlapping dabs within a single
        // stroke clamp to brushAlpha (no buildup) — useful for shading
        // and color blending. When off, stacking (the historical behavior).
        strokeUniformAlphaRow = buildStrokeUniformAlphaRow()
        container.addView(strokeUniformAlphaRow)

        // hard slider — radial dab hardness. Replaces the old "smth"
        // stub. 0 = full radial gradient (smooth dab), 100 = solid
        // disc (hard dab). Applies to brush + eraser.
        brushHardRow = buildBrushHardnessRow()
        container.addView(brushHardRow)

        // press slider — pen-pressure saturation point. 100 = full pen
        // range; lower values make the pen reach max effective pressure
        // sooner (e.g. 50 = max at half the raw pen output). 0 = always
        // full pressure (pressure-insensitive).
        brushPressRow = buildBrushPressureRow()
        container.addView(brushPressRow)

        // Bucket-only bleed slider. Hidden by default; shown when the
        // active tool is BUCKET (see updateSizeSliderForTool).
        bucketBleedRow = buildBucketBleedRow()
        container.addView(bucketBleedRow)

        // Push the initial value display.
        updateSizeSliderForTool()
        return container
    }

    private fun buildStrokeUniformAlphaRow(): View {
        // Two-column toggle: label "uniform α" on the left, on/off
        // indicator on the right. Matches the brush panel's padding and
        // font; the value column width mirrors the slider rows' so the
        // on/off chip lines up with the value labels above and below.
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(10.dp, 4.dp, 10.dp, 4.dp)
        }
        val label = TextView(this).apply {
            text = "uniform α"
            typeface = fontMono ?: Typeface.MONOSPACE
            textSize = 11f
            setTextColor(getColor(R.color.inkSoft))
        }
        val valueLabel = TextView(this).apply {
            typeface = fontMono ?: Typeface.MONOSPACE
            textSize = 11f
            gravity = Gravity.END
            text = if (strokeUniformAlpha) "on" else "off"
            setTextColor(getColor(
                if (strokeUniformAlpha) R.color.hot else R.color.inkSoft))
        }
        row.addView(label, LinearLayout.LayoutParams(
            0, ViewGroup.LayoutParams.WRAP_CONTENT, 1.0f))
        row.addView(valueLabel, LinearLayout.LayoutParams(36.dp,
            ViewGroup.LayoutParams.WRAP_CONTENT))
        row.isClickable = true
        row.isFocusable = true
        row.setOnClickListener {
            strokeUniformAlpha = !strokeUniformAlpha
            NativeRenderer.setStrokeUniformAlpha(strokeUniformAlpha)
            prefs().edit()
                .putBoolean(kPrefStrokeUniformAlpha, strokeUniformAlpha)
                .apply()
            valueLabel.text = if (strokeUniformAlpha) "on" else "off"
            valueLabel.setTextColor(getColor(
                if (strokeUniformAlpha) R.color.hot else R.color.inkSoft))
        }
        return row
    }

    private fun buildBrushAlphaRow(): View {
        val label = TextView(this).apply {
            text = "α"
            typeface = fontMono ?: Typeface.MONOSPACE
            textSize = 11f
            setTextColor(getColor(R.color.inkSoft))
        }
        val valueLabel = TextView(this).apply {
            typeface = fontMono ?: Typeface.MONOSPACE
            textSize = 11f
            setTextColor(getColor(R.color.ink))
            gravity = Gravity.END
            // Initial display matches the initial slider position
            // (target stroke opacity %), not the per-dab α value.
            text = brushAlphaToProgress(brushAlpha).toString()
        }
        val slider = SeekBar(this).apply {
            max = 100
            progress = brushAlphaToProgress(brushAlpha)
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(sb: SeekBar?, p: Int, fromUser: Boolean) {
                    // p is the target stroke opacity (0..100); display
                    // matches slider position so the user sees what
                    // they get.
                    valueLabel.text = p.toString()
                    if (!fromUser) return
                    brushAlpha = progressToBrushAlpha(p)
                    NativeRenderer.setBrushAlpha(brushAlpha)
                }
                override fun onStartTrackingTouch(sb: SeekBar?) {}
                override fun onStopTrackingTouch(sb: SeekBar?) {
                    prefs().edit().putFloat(kPrefBrushAlpha, brushAlpha).apply()
                }
            })
        }
        return buildSliderRow(label, slider, valueLabel)
    }

    /** Bucket-only bleed slider — pixels to dilate the fill mask after
     *  flood-fill, bridging the boundary's anti-aliased gradient. Only
     *  shown while the BUCKET tool is active. Range 0..kBucketBleedMax. */
    private fun buildBucketBleedRow(): View {
        val label = TextView(this).apply {
            text = "bleed"
            typeface = fontMono ?: Typeface.MONOSPACE
            textSize = 11f
            setTextColor(getColor(R.color.inkSoft))
        }
        val valueLabel = TextView(this).apply {
            typeface = fontMono ?: Typeface.MONOSPACE
            textSize = 11f
            setTextColor(getColor(R.color.ink))
            gravity = Gravity.END
            text = bucketBleed.toString()
        }
        val slider = SeekBar(this).apply {
            max = kBucketBleedMax
            progress = bucketBleed.coerceIn(0, kBucketBleedMax)
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(sb: SeekBar?, p: Int, fromUser: Boolean) {
                    valueLabel.text = p.toString()
                    if (!fromUser) return
                    bucketBleed = p
                    NativeRenderer.setBucketBleed(bucketBleed)
                }
                override fun onStartTrackingTouch(sb: SeekBar?) {}
                override fun onStopTrackingTouch(sb: SeekBar?) {
                    prefs().edit().putInt(kPrefBucketBleed, bucketBleed).apply()
                }
            })
        }
        return buildSliderRow(label, slider, valueLabel)
    }

    private fun buildBrushPressureRow(): View {
        val label = TextView(this).apply {
            text = "press"
            typeface = fontMono ?: Typeface.MONOSPACE
            textSize = 11f
            setTextColor(getColor(R.color.inkSoft))
        }
        val valueLabel = TextView(this).apply {
            typeface = fontMono ?: Typeface.MONOSPACE
            textSize = 11f
            setTextColor(getColor(R.color.ink))
            gravity = Gravity.END
            text = (brushPressureSaturation * 100).toInt().toString()
        }
        val slider = SeekBar(this).apply {
            max = 100
            progress = (brushPressureSaturation * 100).toInt().coerceIn(0, 100)
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(sb: SeekBar?, p: Int, fromUser: Boolean) {
                    valueLabel.text = p.toString()
                    if (!fromUser) return
                    brushPressureSaturation = p / 100f
                    drawingView?.brushPressureSaturation = brushPressureSaturation
                }
                override fun onStartTrackingTouch(sb: SeekBar?) {}
                override fun onStopTrackingTouch(sb: SeekBar?) {
                    prefs().edit()
                        .putFloat(kPrefBrushPressure, brushPressureSaturation)
                        .apply()
                }
            })
        }
        return buildSliderRow(label, slider, valueLabel)
    }

    private fun buildBrushHardnessRow(): View {
        val label = TextView(this).apply {
            text = "hard"
            typeface = fontMono ?: Typeface.MONOSPACE
            textSize = 11f
            setTextColor(getColor(R.color.inkSoft))
        }
        val valueLabel = TextView(this).apply {
            typeface = fontMono ?: Typeface.MONOSPACE
            textSize = 11f
            setTextColor(getColor(R.color.ink))
            gravity = Gravity.END
            text = (brushHardness * 100).toInt().toString()
        }
        val slider = SeekBar(this).apply {
            max = 100
            progress = (brushHardness * 100).toInt().coerceIn(0, 100)
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(sb: SeekBar?, p: Int, fromUser: Boolean) {
                    valueLabel.text = p.toString()
                    if (!fromUser) return
                    brushHardness = p / 100f
                    NativeRenderer.setBrushHardness(brushHardness)
                }
                override fun onStartTrackingTouch(sb: SeekBar?) {}
                override fun onStopTrackingTouch(sb: SeekBar?) {
                    prefs().edit().putFloat(kPrefBrushHardness, brushHardness).apply()
                }
            })
        }
        return buildSliderRow(label, slider, valueLabel)
    }

    private fun buildSliderRow(label: View, slider: View, valueLabel: View): View {
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(10.dp, 4.dp, 10.dp, 4.dp)
        }
        row.addView(label, LinearLayout.LayoutParams(36.dp,
            ViewGroup.LayoutParams.WRAP_CONTENT))
        row.addView(slider, LinearLayout.LayoutParams(
            0, ViewGroup.LayoutParams.WRAP_CONTENT, 1.0f))
        row.addView(valueLabel, LinearLayout.LayoutParams(36.dp,
            ViewGroup.LayoutParams.WRAP_CONTENT))
        return row
    }

    private fun makeStubSliderLabel(text: String): TextView = TextView(this).apply {
        this.text = text
        typeface = fontMono ?: Typeface.MONOSPACE
        textSize = 11f
        setTextColor(getColor(R.color.inkDisabled))
    }

    private fun makeStubSlider(): SeekBar = SeekBar(this).apply {
        max = 100
        progress = 100
        isEnabled = false
        alpha = 0.55f
    }

    private fun makeStubValueLabel(text: String): TextView = TextView(this).apply {
        this.text = text
        typeface = fontMono ?: Typeface.MONOSPACE
        textSize = 11f
        setTextColor(getColor(R.color.inkDisabled))
        gravity = Gravity.END
    }

    private fun currentToolEditsVector(): Boolean = when (currentToolMirror) {
        Tool.LINE, Tool.RECTANGLE, Tool.CIRCLE, Tool.ELLIPSE -> true
        else -> false
    }

    /** Refresh the size slider's progress / value display when the tool
     *  changes (or after a one-shot sync at startup). */
    private fun updateSizeSliderForTool() {
        if (currentToolEditsVector()) {
            sizeSliderLabel.text = "width"
            sizeSlider.progress = vectorWidthToProgress(vectorLineWidth)
            sizeValueLabel.text = "%.1f".format(vectorLineWidth)
        } else {
            sizeSliderLabel.text = "size"
            sizeSlider.progress = brushScaleToProgress(brushSizeScale)
            sizeValueLabel.text = "%.2fx".format(brushSizeScale)
        }

        // Match the section header to the tool family — same sliders
        // drive each, but the wording shouldn't say "BRUSH" while the
        // user is editing a vector line width or eraser settings.
        if (::brushSectionHeader.isInitialized) {
            brushSectionHeader.text = when (currentToolMirror) {
                Tool.ERASER                                              -> "ERASER"
                Tool.BUCKET                                              -> "BUCKET"
                Tool.SHADE                                               -> "SHADE"
                Tool.LINE, Tool.RECTANGLE, Tool.CIRCLE, Tool.ELLIPSE     -> "VECTOR"
                Tool.SELECT, Tool.SELECT_RECT, Tool.SELECT_LASSO         -> "SELECT"
                else                                                     -> "BRUSH"
            }
        }

        // Hide slider rows that don't apply to the active tool. Only
        // alpha is shared across most tools; size is brush/eraser/
        // vector; hardness is brush/eraser; press is a stub that only
        // makes sense alongside size.
        if (::brushSizeRow.isInitialized) {
            val tool = currentToolMirror
            val showSize  = tool.isRasterStroke || currentToolEditsVector()
            val showAlpha = tool.isRasterStroke || tool == Tool.BUCKET
            // Uniform-α toggle is meaningful only for the stroke-based
            // raster tools (brush + eraser). Bucket is a single fill,
            // not a stroke, so the mode has no effect there — and shade
            // forces uniform alpha so its outline and fill read as one
            // flat tone, which leaves the user nothing to toggle.
            val showUniformAlpha = tool == Tool.BRUSH || tool == Tool.ERASER
            val showHard  = tool.isRasterStroke
            val showPress = tool.isRasterStroke
            val showBleed = tool == Tool.BUCKET
            brushSizeRow.visibility   = if (showSize)         View.VISIBLE else View.GONE
            brushAlphaRow.visibility  = if (showAlpha)        View.VISIBLE else View.GONE
            strokeUniformAlphaRow.visibility =
                if (showUniformAlpha) View.VISIBLE else View.GONE
            brushHardRow.visibility   = if (showHard)         View.VISIBLE else View.GONE
            brushPressRow.visibility  = if (showPress)        View.VISIBLE else View.GONE
            bucketBleedRow.visibility = if (showBleed)        View.VISIBLE else View.GONE
        }
    }

    // The COLOR section follows the Design A "shared palette" layout:
    //   - active+previous swatch pair (Photoshop stack)  +  hex  +  eyedropper
    //   - 16-cell default palette (2×8)
    //   - recents strip (8 slots, auto, LRU)
    //   - my-slots strip (4 slots, long-press to set)
    // The full picker (HSV square + hue slider) opens via tap on the
    // active swatch — see ColorPickerDialog. The header still keeps the
    // tiny chip that mirrors the active color.
    private lateinit var colorActiveSwatch: View
    private lateinit var colorPreviousSwatch: View
    private lateinit var colorHexLabel: TextView
    private lateinit var colorPaletteGrid: GridLayout
    private lateinit var colorRecentsRow: LinearLayout
    private lateinit var colorMySlotsRow: LinearLayout

    private fun buildColorSection(): View {
        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
        }
        // Header with a small color chip on the right (mirrors active).
        val header = FrameLayout(this).apply {
            background = ResourcesCompat.getDrawable(
                resources, R.drawable.panel_section_header, theme)
        }
        val headerLabel = TextView(this).apply {
            text = "COLOR"
            typeface = fontMonoSemibold ?: Typeface.MONOSPACE
            textSize = 10f
            letterSpacing = 0.08f
            setTextColor(getColor(R.color.ink))
            setPadding(10.dp, 0, 0, 0)
            gravity = Gravity.CENTER_VERTICAL
        }
        header.addView(headerLabel, FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.MATCH_PARENT,
            Gravity.START or Gravity.CENTER_VERTICAL))
        colorChip = View(this).apply {
            setBackgroundColor(0xFF000000.toInt() or currentColorRgb)
        }
        header.addView(colorChip, FrameLayout.LayoutParams(18.dp, 14.dp,
            Gravity.END or Gravity.CENTER_VERTICAL).apply {
                marginEnd = 10.dp
            })
        container.addView(header,
            LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 28.dp))

        // Body padding matches the brush + layer sections.
        val body = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(10.dp, 8.dp, 10.dp, 12.dp)
        }

        // ---- Row 1: active+previous swatches | hex readout | eyedropper -
        body.addView(buildActiveColorRow(), LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply { bottomMargin = 12.dp })

        // ---- Row 2: 16-cell default palette ------------------------------
        body.addView(makePaletteSubsectionLabel("palette"))
        colorPaletteGrid = GridLayout(this).apply {
            columnCount = 8
        }
        rebuildPaletteGrid()
        body.addView(colorPaletteGrid, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply { bottomMargin = 12.dp })

        // ---- Row 3: recents strip ----------------------------------------
        body.addView(makePaletteSubsectionLabel("recent"))
        colorRecentsRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
        }
        rebuildRecentsRow()
        body.addView(colorRecentsRow, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply { bottomMargin = 12.dp })

        // ---- Row 4: my-slots strip --------------------------------------
        body.addView(makePaletteSubsectionLabel("my slots", "long-press to set"))
        colorMySlotsRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
        }
        rebuildMySlotsRow()
        body.addView(colorMySlotsRow, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT))

        container.addView(body)
        return container
    }

    /** Active+previous swatch pair (Photoshop stack) + hex + eyedropper. */
    private fun buildActiveColorRow(): View {
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }

        // Stacked swatch pair (active 28dp on top, previous 22dp behind).
        // Each swatch handles its own taps: tapping the active swatch opens
        // the picker; tapping the exposed corner of the previous swatch
        // swaps the two so the user can flip back to the prior color
        // without re-picking. Touch dispatch in FrameLayout goes top-down,
        // so taps in the overlap region land on the active swatch and the
        // previous swatch only sees the L-shape of pixels still visible.
        val stack = FrameLayout(this)
        colorPreviousSwatch = View(this).apply {
            setBackgroundColor(0xFF000000.toInt() or previousColorRgb)
            isClickable = true; isFocusable = true
            contentDescription = "previous color · tap to swap"
            setOnClickListener { swapPreviousAndCurrent() }
        }
        stack.addView(colorPreviousSwatch, FrameLayout.LayoutParams(22.dp, 22.dp,
            Gravity.END or Gravity.BOTTOM))
        colorActiveSwatch = View(this).apply {
            isClickable = true; isFocusable = true
            contentDescription = "active color · tap to open picker"
            // Active swatch with a 1.5dp ink border (we approximate via a
            // GradientDrawable to avoid needing an XML resource).
            background = android.graphics.drawable.GradientDrawable().apply {
                shape = android.graphics.drawable.GradientDrawable.RECTANGLE
                setColor(0xFF000000.toInt() or currentColorRgb)
                setStroke(2.dp.coerceAtLeast(1), getColor(R.color.ink))
            }
            setOnClickListener { showColorPickerDialog() }
        }
        stack.addView(colorActiveSwatch, FrameLayout.LayoutParams(28.dp, 28.dp,
            Gravity.START or Gravity.TOP))
        row.addView(stack, LinearLayout.LayoutParams(36.dp, 36.dp).apply {
            rightMargin = 8.dp
        })

        // Hex readout — tappable too so the user can also open the picker
        // by tapping the value.
        colorHexLabel = TextView(this).apply {
            typeface = fontMonoSemibold ?: Typeface.MONOSPACE
            textSize = 11f
            setTextColor(getColor(R.color.ink))
            text = String.format("#%06X", currentColorRgb)
            isClickable = true; isFocusable = true
            setOnClickListener { showColorPickerDialog() }
        }
        row.addView(colorHexLabel, LinearLayout.LayoutParams(
            0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))

        // Eyedropper button — arms a single-shot sample-from-canvas mode
        // in DrawingSurfaceView. The next tap on the canvas reads the
        // pixel under the touch and applies it as the active color.
        val eyedropper = ImageView(this).apply {
            setImageResource(R.drawable.ic_eyedropper)
            imageTintList = ColorStateList.valueOf(getColor(R.color.ink))
            scaleType = ImageView.ScaleType.CENTER_INSIDE
            setPadding(4.dp, 4.dp, 4.dp, 4.dp)
            background = android.graphics.drawable.GradientDrawable().apply {
                shape = android.graphics.drawable.GradientDrawable.RECTANGLE
                setStroke(1.dp.coerceAtLeast(1), getColor(R.color.rule))
                setColor(getColor(R.color.paper))
            }
            isClickable = true; isFocusable = true
            contentDescription = "eyedropper"
            setOnClickListener { armEyedropper() }
        }
        row.addView(eyedropper, LinearLayout.LayoutParams(28.dp, 28.dp))
        return row
    }

    private fun makePaletteSubsectionLabel(label: String, hint: String? = null): View {
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.BOTTOM
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply { bottomMargin = 4.dp }
        }
        row.addView(TextView(this).apply {
            text = label
            typeface = fontMono ?: Typeface.MONOSPACE
            textSize = 9f
            letterSpacing = 0.08f
            setTextColor(getColor(R.color.inkSoft))
        }, LinearLayout.LayoutParams(0,
            ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        if (hint != null) {
            row.addView(TextView(this).apply {
                text = hint
                typeface = fontMono ?: Typeface.MONOSPACE
                textSize = 8f
                setTextColor(getColor(R.color.inkFaint))
            })
        }
        return row
    }

    private fun rebuildPaletteGrid() {
        if (!::colorPaletteGrid.isInitialized) return
        colorPaletteGrid.removeAllViews()
        val swatchSize = 16.dp
        val gap = 2.dp
        for (rgb in palette) {
            val swatch = View(this).apply {
                setBackgroundColor(0xFF000000.toInt() or rgb)
                isClickable = true; isFocusable = true
                setOnClickListener { setActiveColor(rgb) }
            }
            val lp = GridLayout.LayoutParams().apply {
                width  = swatchSize
                height = swatchSize
                setMargins(gap, gap, gap, gap)
            }
            colorPaletteGrid.addView(swatch, lp)
        }
    }

    private fun rebuildRecentsRow() {
        if (!::colorRecentsRow.isInitialized) return
        colorRecentsRow.removeAllViews()
        val swatchSize = 18.dp
        val gap = 2.dp
        for (i in 0 until kRecentsMax) {
            val rgb = recentColors.getOrNull(i)
            val view = if (rgb != null) View(this).apply {
                setBackgroundColor(0xFF000000.toInt() or rgb)
                isClickable = true; isFocusable = true
                setOnClickListener { setActiveColor(rgb) }
            } else View(this).apply {
                background = android.graphics.drawable.GradientDrawable().apply {
                    shape = android.graphics.drawable.GradientDrawable.RECTANGLE
                    setStroke(1.dp.coerceAtLeast(1), getColor(R.color.rule),
                        4f * resources.displayMetrics.density,
                        3f * resources.displayMetrics.density)
                    setColor(0)
                }
            }
            colorRecentsRow.addView(view, LinearLayout.LayoutParams(swatchSize, swatchSize)
                .apply { if (i > 0) leftMargin = gap })
        }
    }

    private fun rebuildMySlotsRow() {
        if (!::colorMySlotsRow.isInitialized) return
        colorMySlotsRow.removeAllViews()
        val swatchSize = 28.dp
        val gap = 4.dp
        for (i in 0 until kMySlotsCount) {
            val rgb = mySlots[i]
            val view = if (rgb != null) View(this).apply {
                setBackgroundColor(0xFF000000.toInt() or rgb)
                isClickable = true; isFocusable = true
                setOnClickListener { setActiveColor(rgb) }
                setOnLongClickListener {
                    setMySlot(i, currentColorRgb); true
                }
            } else TextView(this).apply {
                text = "+"
                gravity = Gravity.CENTER
                typeface = fontMono ?: Typeface.MONOSPACE
                textSize = 14f
                setTextColor(getColor(R.color.inkFaint))
                background = android.graphics.drawable.GradientDrawable().apply {
                    shape = android.graphics.drawable.GradientDrawable.RECTANGLE
                    setStroke(1.dp.coerceAtLeast(1), getColor(R.color.rule),
                        4f * resources.displayMetrics.density,
                        3f * resources.displayMetrics.density)
                    setColor(0)
                }
                isClickable = true; isFocusable = true
                setOnClickListener { setMySlot(i, currentColorRgb) }
            }
            colorMySlotsRow.addView(view, LinearLayout.LayoutParams(swatchSize, swatchSize)
                .apply { if (i > 0) leftMargin = gap })
        }
    }

    /** Single source of truth for "the user picked a color". Updates the
     *  native brush color, mirrors into UI state, prepends to recents
     *  (dedup), and persists. previous tracks the most recent prior value
     *  so the Photoshop-style stack still reads. */
    private fun setActiveColor(rgb: Int) {
        val masked = rgb and 0xFFFFFF
        if (masked != currentColorRgb) {
            previousColorRgb = currentColorRgb
            currentColorRgb  = masked
        }
        NativeRenderer.setBrushColor(currentColorRgb)
        // Prepend to recents (dedup).
        recentColors.remove(currentColorRgb)
        recentColors.addFirst(currentColorRgb)
        while (recentColors.size > kRecentsMax) recentColors.removeLast()
        saveColorState()
        refreshColorUi()
    }

    private fun setMySlot(idx: Int, rgb: Int) {
        if (idx !in 0 until kMySlotsCount) return
        mySlots[idx] = rgb and 0xFFFFFF
        saveColorState()
        rebuildMySlotsRow()
    }

    /** Arm the eyedropper. The next tap on the canvas inside
     *  DrawingSurfaceView samples that pixel and applies it as the
     *  active color (single-shot). Toast lets the user know the mode
     *  is armed since there's no other affordance on the canvas. */
    private fun armEyedropper() {
        val v = drawingView ?: return
        v.onColorSampled = { rgb -> setActiveColor(rgb) }
        v.eyedropperPending = true
        android.widget.Toast.makeText(this,
            "tap canvas to sample color",
            android.widget.Toast.LENGTH_SHORT).show()
    }

    /** Flip active and previous. Goes through native (so the brush color
     *  follows) and refreshes the UI; intentionally bypasses recents,
     *  since the swap doesn't represent a fresh choice — the previous
     *  color is already in the user's history. */
    private fun swapPreviousAndCurrent() {
        if (currentColorRgb == previousColorRgb) return
        val tmp = currentColorRgb
        currentColorRgb = previousColorRgb
        previousColorRgb = tmp
        NativeRenderer.setBrushColor(currentColorRgb)
        refreshColorUi()
    }

    /** Open the full picker dialog — Design A's HSV square + hue slider. */
    private fun showColorPickerDialog() {
        ColorPickerDialog(
            context = this,
            initialRgb = currentColorRgb,
            mono = fontMono,
            monoSemibold = fontMonoSemibold,
            onColorPicked = { rgb -> setActiveColor(rgb) }
        ).show()
    }

    /** Refresh every UI element that mirrors the active color. */
    private fun refreshColorUi() {
        updateColorChip()
        if (::colorActiveSwatch.isInitialized) {
            (colorActiveSwatch.background as?
                android.graphics.drawable.GradientDrawable)
                ?.setColor(0xFF000000.toInt() or currentColorRgb)
        }
        if (::colorPreviousSwatch.isInitialized) {
            colorPreviousSwatch.setBackgroundColor(
                0xFF000000.toInt() or previousColorRgb)
        }
        if (::colorHexLabel.isInitialized) {
            colorHexLabel.text = String.format("#%06X", currentColorRgb)
        }
        rebuildRecentsRow()
        rebuildMySlotsRow()
    }

    private fun updateColorChip() {
        if (::colorChip.isInitialized) {
            colorChip.setBackgroundColor(0xFF000000.toInt() or currentColorRgb)
        }
    }

    private fun loadColorState() {
        val recentsStr = prefs().getString(kPrefRecents, "") ?: ""
        recentColors.clear()
        for (token in recentsStr.split(',')) {
            val t = token.trim()
            if (t.length == 6) {
                runCatching { Integer.parseInt(t, 16) }
                    .getOrNull()?.let { recentColors.addLast(it and 0xFFFFFF) }
                if (recentColors.size >= kRecentsMax) break
            }
        }
        val slotsStr = prefs().getString(kPrefMySlots, null)
        if (slotsStr != null) {
            val parts = slotsStr.split(',')
            for (i in 0 until kMySlotsCount) {
                val t = parts.getOrNull(i)?.trim().orEmpty()
                mySlots[i] = if (t.length == 6) {
                    runCatching { Integer.parseInt(t, 16) }.getOrNull()
                        ?.let { it and 0xFFFFFF }
                } else null
            }
        }
    }

    private fun saveColorState() {
        val recentsStr = recentColors.joinToString(",") { String.format("%06X", it) }
        val slotsStr = mySlots.joinToString(",") { rgb ->
            if (rgb != null) String.format("%06X", rgb) else ""
        }
        prefs().edit()
            .putString(kPrefRecents, recentsStr)
            .putString(kPrefMySlots, slotsStr)
            .apply()
    }

    // ====================================================================
    // Page sidebar (84dp wide, restyled)
    // ====================================================================

    private fun buildPageSidebar(): ScrollView {
        val scroll = ScrollView(this).apply {
            isFillViewport = true
            isVerticalScrollBarEnabled = false
            overScrollMode = View.OVER_SCROLL_NEVER
            setBackgroundColor(getColor(R.color.paper))
        }
        sidebarLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
        }
        // Right-edge hairline so the sidebar visually ends.
        val container = FrameLayout(this)
        container.addView(sidebarLayout, FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ))
        container.addView(View(this).apply {
            setBackgroundColor(getColor(R.color.rule))
        }, FrameLayout.LayoutParams(1.dp, ViewGroup.LayoutParams.MATCH_PARENT,
            Gravity.END))
        scroll.addView(container, FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ))
        return scroll
    }

    /** DOCS button at the top of the sidebar — opens the doc picker. */
    private fun buildDocsButton(): View {
        val tile = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            background = ResourcesCompat.getDrawable(
                resources, R.drawable.panel_section_header, theme)
            setPadding(0, 8.dp, 0, 8.dp)
            isClickable = true; isFocusable = true
            setOnClickListener { showDocsMenu(it) }
        }
        val icon = ImageView(this).apply {
            setImageResource(R.drawable.ic_docs)
            imageTintList = ColorStateList.valueOf(getColor(R.color.ink))
        }
        tile.addView(icon, LinearLayout.LayoutParams(16.dp, 16.dp))
        docsButtonLabel = TextView(this).apply {
            text = "DOCS"
            typeface = fontMonoSemibold ?: Typeface.MONOSPACE
            textSize = 10f
            letterSpacing = 0.1f
            setTextColor(getColor(R.color.ink))
            setPadding(6.dp, 0, 6.dp, 0)
        }
        tile.addView(docsButtonLabel)
        val chev = ImageView(this).apply {
            setImageResource(R.drawable.ic_chev_d)
            imageTintList = ColorStateList.valueOf(getColor(R.color.inkSoft))
        }
        tile.addView(chev, LinearLayout.LayoutParams(12.dp, 12.dp))
        docsButton = tile
        return tile
    }

    /** Recreate every sidebar entry from current page state. */
    private fun rebuildSidebar() {
        sidebarLayout.removeAllViews()
        pageItems.clear()

        // DOCS header — replaces the old name-label + new/open/delete row.
        sidebarLayout.addView(buildDocsButton(),
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT))

        val pageCount = NativeRenderer.getPageCount().coerceAtLeast(1)
        val active    = NativeRenderer.getActivePage()
        for (i in 0 until pageCount) {
            val item = createPageSidebarItem(i, isActive = (i == active))
            pageItems.add(item)
            sidebarLayout.addView(item.container, sidebarItemParams())
        }

        // + Page row, designed as a dashed-frame thumbnail.
        sidebarLayout.addView(buildAddPageRow(), sidebarItemParams())

        lastBuiltPageCount  = pageCount
        lastBuiltActivePage = active

        drawingView?.setThumbnailTargets(
            pageItems.associate { it.pageIdx to it.bitmap }
        )
        drawingView?.forceRedraw()
    }

    /** Move the "active page" highlight without touching the view tree
     *  or the thumbnail bitmaps.
     *
     *  Switching pages used to route through rebuildSidebar(), which
     *  allocates a fresh white Bitmap per item and re-registers them as
     *  thumbnail targets — so every preview blanked and then repainted.
     *  That was invisible back when the whole refresh happened inside a
     *  single GL pass; now that it's spread across several (see
     *  thumbnailRefreshQueue in DrawingSurfaceView), the blank frames
     *  read as a flash across every thumbnail. A page switch changes
     *  nothing structural, so don't rebuild anything.
     *
     *  Thumbnails don't go stale from skipping the refresh: a page's
     *  pixels can only change while it's the active one, and the active
     *  page is re-rendered every quiet pass regardless. */
    private fun updateSidebarActivePage(active: Int) {
        for (item in pageItems) {
            val isActive = item.pageIdx == active
            item.frame.isSelected = isActive
            item.numberBadge.setTextColor(
                getColor(if (isActive) R.color.hot else R.color.inkSoft))
        }
        lastBuiltActivePage = active
    }

    private fun buildAddPageRow(): View {
        val frame = FrameLayout(this).apply {
            isClickable = true; isFocusable = true
            setOnClickListener { userAddPage() }
        }
        val plus = ImageView(this).apply {
            setImageResource(R.drawable.ic_plus)
            imageTintList = ColorStateList.valueOf(getColor(R.color.inkSoft))
        }
        // Render via a thin-bordered View so the dashed-frame impression
        // comes through without needing a custom dashed drawable.
        val box = View(this).apply {
            background = makeDashedFrame()
        }
        frame.addView(box, FrameLayout.LayoutParams(kThumbMaxWidthDp.dp, 36.dp,
            Gravity.CENTER))
        frame.addView(plus, FrameLayout.LayoutParams(20.dp, 20.dp,
            Gravity.CENTER))
        return frame
    }

    private fun makeDashedFrame(): android.graphics.drawable.Drawable {
        val s = android.graphics.drawable.GradientDrawable()
        s.shape = android.graphics.drawable.GradientDrawable.RECTANGLE
        s.setColor(Color.TRANSPARENT)
        s.setStroke(1.dp,
            getColor(R.color.inkFaint),
            /* dashWidth */ 4f * resources.displayMetrics.density,
            /* dashGap   */ 3f * resources.displayMetrics.density)
        return s
    }

    /** Pick the thumbnail bitmap dimensions that match the canvas aspect
     *  ratio, fit inside the design's max box (60×78 dp). Without this
     *  we'd letterbox a wide canvas into a portrait thumb and end up
     *  with gray bars top + bottom. Falls back to the raw maximums when
     *  the SurfaceView hasn't been laid out yet. */
    private fun thumbDimensions(): Pair<Int, Int> {
        val maxW = kThumbMaxWidthDp.dp
        val maxH = kThumbMaxHeightDp.dp
        val pageW = drawingView?.width ?: 0
        val pageH = drawingView?.height ?: 0
        if (pageW <= 0 || pageH <= 0) return Pair(maxW, maxH)
        val aspect = pageH.toFloat() / pageW.toFloat()
        return if (maxW * aspect <= maxH) {
            Pair(maxW, (maxW * aspect).toInt().coerceAtLeast(1))
        } else {
            Pair((maxH / aspect).toInt().coerceAtLeast(1), maxH)
        }
    }

    private fun createPageSidebarItem(idx: Int, isActive: Boolean): PageSidebarItem {
        val (thumbW, thumbH) = thumbDimensions()
        val bitmap = Bitmap.createBitmap(thumbW, thumbH, Bitmap.Config.ARGB_8888)
        bitmap.eraseColor(Color.WHITE)

        val container = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            isClickable = true; isFocusable = true
            setOnClickListener {
                NativeRenderer.switchPage(idx)
                drawingView?.forceRedraw()
            }
        }

        // Thumbnail frame holds the bitmap + a corner page-number badge.
        val frame = FrameLayout(this).apply {
            background = ResourcesCompat.getDrawable(
                resources, R.drawable.page_thumb_bg, theme)
            isSelected = isActive
            setPadding(2.dp, 2.dp, 2.dp, 2.dp)
        }
        val imageView = ImageView(this).apply {
            setImageBitmap(bitmap)
            scaleType = ImageView.ScaleType.FIT_CENTER
        }
        frame.addView(imageView, FrameLayout.LayoutParams(thumbW, thumbH))

        val numberBadge = TextView(this).apply {
            text = String.format("%02d", idx + 1)
            typeface = fontMonoSemibold ?: Typeface.MONOSPACE
            textSize = 8f
            setTextColor(getColor(if (isActive) R.color.hot else R.color.inkSoft))
            setPadding(2.dp, 0, 0, 0)
        }
        frame.addView(numberBadge, FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT,
            ViewGroup.LayoutParams.WRAP_CONTENT,
            Gravity.START or Gravity.TOP))

        container.addView(frame)

        // Per-thumbnail icons stacked vertically in the dead space to the
        // right of the thumbnail. Drag handle on top, overflow below;
        // same idiom as layer rows so gesture vocabulary stays consistent.
        val iconCol = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
        }
        val dragHandle = ImageView(this).apply {
            setImageResource(R.drawable.ic_drag_handle)
            imageTintList = ColorStateList.valueOf(getColor(R.color.inkSoft))
            scaleType = ImageView.ScaleType.CENTER_INSIDE
            contentDescription = "drag to reorder"
            isClickable = true; isFocusable = true
        }
        installPageDragHandleListener(dragHandle, idx, container)
        iconCol.addView(dragHandle, LinearLayout.LayoutParams(18.dp, 18.dp))

        val pageMore = ImageView(this).apply {
            setImageResource(R.drawable.ic_more)
            imageTintList = ColorStateList.valueOf(getColor(R.color.inkSoft))
            scaleType = ImageView.ScaleType.CENTER_INSIDE
            contentDescription = "more"
            isClickable = true; isFocusable = true
            setOnClickListener { showPageOverflow(it, idx) }
        }
        iconCol.addView(pageMore, LinearLayout.LayoutParams(18.dp, 18.dp).apply {
            topMargin = 2.dp
        })
        container.addView(iconCol, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply {
            leftMargin = 4.dp
        })

        return PageSidebarItem(container, frame, imageView, numberBadge, bitmap, idx)
    }

    private fun sidebarItemParams() = LinearLayout.LayoutParams(
        ViewGroup.LayoutParams.MATCH_PARENT,
        ViewGroup.LayoutParams.WRAP_CONTENT
    ).apply {
        topMargin = 6.dp
        bottomMargin = 0
    }

    private fun togglePageSidebar() {
        // GONE so the layer panel can shift left and the canvas can claim
        // the freed space. The canvas right edge stays put (it's already
        // pinned to the screen edge), and DrawingSurfaceView.onSizeChanged
        // shifts viewPanX by the width delta so existing strokes don't
        // appear to slide on screen — only new room appears on the left.
        sidebarScroll.visibility =
            if (sidebarScroll.visibility == View.VISIBLE) View.GONE else View.VISIBLE
    }

    /** Show / hide the LAYERS section in the layer panel. The wrapper
     *  contains the header, list, opacity slider, and the trailing
     *  divider — toggling visibility removes them all together so the
     *  remaining sections close up cleanly. */
    private fun toggleLayersSection() {
        if (!::layersSection.isInitialized) return
        layersSection.visibility =
            if (layersSection.visibility == View.VISIBLE) View.GONE else View.VISIBLE
    }

    /** Show / hide the COLOR section. Same wrapper pattern as layers —
     *  the leading divider is included so the panel doesn't end on a
     *  stray rule when the section is collapsed. */
    private fun toggleColorSection() {
        if (!::colorSection.isInitialized) return
        colorSection.visibility =
            if (colorSection.visibility == View.VISIBLE) View.GONE else View.VISIBLE
    }

    // ====================================================================
    // Floating chrome (undo / redo chip)
    // ====================================================================

    private fun buildUndoRedoChip(): View {
        val chip = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            background = ResourcesCompat.getDrawable(
                resources, R.drawable.chrome_chip_bg, theme)
            setPadding(8.dp, 6.dp, 8.dp, 6.dp)
        }
        val undo = ImageView(this).apply {
            setImageResource(R.drawable.ic_undo)
            imageTintList = ColorStateList.valueOf(getColor(R.color.ink))
            isClickable = true; isFocusable = true
            contentDescription = "undo"
            setOnClickListener { userUndo() }
            setPadding(2.dp, 2.dp, 6.dp, 2.dp)
        }
        val redo = ImageView(this).apply {
            setImageResource(R.drawable.ic_redo)
            imageTintList = ColorStateList.valueOf(getColor(R.color.ink))
            isClickable = true; isFocusable = true
            contentDescription = "redo"
            setOnClickListener { userRedo() }
            setPadding(6.dp, 2.dp, 2.dp, 2.dp)
        }
        chip.addView(undo, LinearLayout.LayoutParams(28.dp, 22.dp))
        chip.addView(redo, LinearLayout.LayoutParams(28.dp, 22.dp))
        return chip
    }

    private fun undoRedoChipParams() = FrameLayout.LayoutParams(
        ViewGroup.LayoutParams.WRAP_CONTENT,
        ViewGroup.LayoutParams.WRAP_CONTENT
    ).apply {
        gravity = Gravity.TOP or Gravity.START
        topMargin  = 16.dp
        marginStart = 16.dp
    }

    // ====================================================================
    // Status bar
    // ====================================================================

    private fun buildStatusBar(): View {
        val bar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setBackgroundColor(getColor(R.color.paperDeep))
            setPadding(12.dp, 0, 12.dp, 0)
        }
        // Top hairline so the bar feels separated from the canvas above.
        val wrapper = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
        }
        wrapper.addView(View(this).apply {
            setBackgroundColor(getColor(R.color.rule))
        }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 1.dp))
        wrapper.addView(bar, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, 0, 1.0f))

        statusDocText  = makeStatusText("doc · $currentDocName")
        statusToolText = makeStatusText("◇ ${currentToolMirror.displayName.lowercase()}")
        // grid/snap texts double as toggles. Tap to flip; the active
        // state colors them with the same hot accent the rail's active
        // tile uses, so they read as enabled at a glance.
        statusGridText = makeStatusText(gridChipLabel()).apply {
            isClickable = true; isFocusable = true
            setOnClickListener { cycleGrid() }
            // Long-press opens the spacing slider — config sits with
            // the toggle so the two don't live on opposite screen edges.
            setOnLongClickListener { showGridConfigPopup(this); true }
            setTextColor(getColor(
                if (gridState != 0) R.color.hot else R.color.inkSoft))
        }
        // Margins — same tap-toggles / long-press-configures pattern.
        statusMarginText = makeStatusText(marginChipLabel()).apply {
            isClickable = true; isFocusable = true
            setOnClickListener { toggleMargins() }
            setOnLongClickListener { showMarginConfigPopup(this); true }
            setTextColor(getColor(
                if (marginsEnabled) R.color.hot else R.color.inkSoft))
        }
        // Pixel grid — 1-doc-pixel boundaries on top of strokes. Native
        // self-gates on zoom so the chip can read "on" while no grid
        // shows at low zoom; the user enables once and the grid appears
        // automatically when they zoom in past the threshold.
        statusPixelGridText = makeStatusText(
            if (pixelGridEnabled) "px: on" else "px: off"
        ).apply {
            isClickable = true; isFocusable = true
            setOnClickListener { togglePixelGrid() }
            setTextColor(getColor(
                if (pixelGridEnabled) R.color.hot else R.color.inkSoft))
        }
        statusSnapText = makeStatusText(
            if (snapEnabled) "snap: on" else "snap: off"
        ).apply {
            isClickable = true; isFocusable = true
            setOnClickListener { toggleSnap() }
            setTextColor(getColor(
                if (snapEnabled) R.color.hot else R.color.inkSoft))
        }
        // Angle snap — line tool's endpoint locks to multiples of 15°.
        // Toggleable here AND from the stylus middle button when a
        // vector layer is active.
        val angleOn = drawingView?.angleSnapEnabled == true
        statusAngleText = makeStatusText(
            if (angleOn) "angle: on" else "angle: off"
        ).apply {
            isClickable = true; isFocusable = true
            setOnClickListener { toggleAngleSnap() }
            setTextColor(getColor(
                if (angleOn) R.color.hot else R.color.inkSoft))
        }
        // Brush preview — thin outline circle on hover/while drawing
        // shows the brush's effective dab radius. Off by default.
        statusPreviewText = makeStatusText(
            if (brushPreviewEnabled) "preview: on" else "preview: off"
        ).apply {
            isClickable = true; isFocusable = true
            setOnClickListener { toggleBrushPreview() }
            setTextColor(getColor(
                if (brushPreviewEnabled) R.color.hot else R.color.inkSoft))
        }
        // Motion-prediction toggle. Forwards to drawingView, which
        // mirrors the flag to native.
        statusPredictText = makeStatusText(
            if (predictionEnabled) "predict: on" else "predict: off"
        ).apply {
            isClickable = true; isFocusable = true
            setOnClickListener { togglePrediction() }
            setTextColor(getColor(
                if (predictionEnabled) R.color.hot else R.color.inkSoft))
        }
        statusPageText = makeStatusText("page —")

        bar.addView(statusDocText,    statusItemLp())
        bar.addView(statusToolText,   statusItemLp())
        bar.addView(statusGridText,   statusItemLp())
        bar.addView(statusMarginText, statusItemLp())
        bar.addView(statusPixelGridText, statusItemLp())
        bar.addView(statusSnapText,   statusItemLp())
        bar.addView(statusAngleText,  statusItemLp())
        bar.addView(statusPreviewText, statusItemLp())
        bar.addView(statusPredictText, statusItemLp())
        // Spacer pushes pageText to the right edge.
        bar.addView(View(this), LinearLayout.LayoutParams(
            0, 0, 1.0f))
        bar.addView(statusPageText, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT,
            ViewGroup.LayoutParams.WRAP_CONTENT))
        return wrapper
    }

    private fun makeStatusText(initial: String): TextView = TextView(this).apply {
        text = initial
        typeface = fontMono ?: Typeface.MONOSPACE
        textSize = 9f
        setTextColor(getColor(R.color.inkSoft))
    }

    private fun statusItemLp() = LinearLayout.LayoutParams(
        ViewGroup.LayoutParams.WRAP_CONTENT,
        ViewGroup.LayoutParams.WRAP_CONTENT
    ).apply { marginEnd = 18.dp }

    private fun updateStatusBarPage() {
        if (!::statusPageText.isInitialized) return
        val pc = NativeRenderer.getPageCount().coerceAtLeast(1)
        val ap = NativeRenderer.getActivePage().coerceIn(0, pc - 1)
        statusPageText.text = "page ${String.format("%02d", ap + 1)} / $pc"
    }

    // ====================================================================
    // Overflow menus
    // ====================================================================

    /** One row in a paper-styled popup menu. */
    private data class PaperMenuItem(
        val title: String,
        val enabled: Boolean = true,
        val onClick: () -> Unit,
    )

    /** Paper-themed replacement for android.widget.PopupMenu. Builds a
     *  PopupWindow whose content is a vertical column of monospace
     *  TextView rows on a paper background, with hairline dividers and
     *  ink/inkSoft for enabled/disabled. Anchored to [anchor] with
     *  end-gravity (matches the PopupMenu calls being replaced). */
    private fun showPaperPopupMenu(anchor: View, items: List<PaperMenuItem>) {
        val ink     = getColor(R.color.ink)
        val inkSoft = getColor(R.color.inkSoft)
        val paper   = getColor(R.color.paper)
        val rule    = getColor(R.color.rule)

        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(paper)
        }

        val popup = android.widget.PopupWindow(
            container,
            ViewGroup.LayoutParams.WRAP_CONTENT,
            ViewGroup.LayoutParams.WRAP_CONTENT,
            /*focusable=*/ true,
        )
        // Required so taps outside the popup dismiss it; ColorDrawable
        // also gives the elevation shadow something to clip against.
        popup.setBackgroundDrawable(
            android.graphics.drawable.ColorDrawable(paper))
        popup.isOutsideTouchable = true
        popup.elevation = 8.dp.toFloat()

        for ((idx, item) in items.withIndex()) {
            val tv = TextView(this).apply {
                text = item.title
                typeface = fontMono ?: Typeface.MONOSPACE
                textSize = 13f
                setTextColor(if (item.enabled) ink else inkSoft)
                setPadding(14.dp, 9.dp, 14.dp, 9.dp)
                if (item.enabled) {
                    isClickable = true; isFocusable = true
                    setOnClickListener {
                        popup.dismiss()
                        item.onClick()
                    }
                }
            }
            container.addView(tv, LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT))
            if (idx < items.size - 1) {
                container.addView(View(this).apply {
                    setBackgroundColor(rule)
                }, LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT, 1.dp))
            }
        }

        // Default PopupWindow width on this OS theme is much wider
        // than the longest item — measure the container's natural
        // width and pin the popup to it so the items hug their text.
        container.measure(
            View.MeasureSpec.makeMeasureSpec(0, View.MeasureSpec.UNSPECIFIED),
            View.MeasureSpec.makeMeasureSpec(0, View.MeasureSpec.UNSPECIFIED),
        )
        popup.width = container.measuredWidth

        popup.showAsDropDown(anchor, 0, 0, Gravity.END)
    }

    /** Build a horizontal Cancel/<confirm> button row styled like the
     *  rest of the app — mono ink Cancel, mono semibold sienna primary
     *  action, end-aligned. Returns the row + the two TextViews so the
     *  caller can wire click handlers. */
    private fun makePaperDialogButtons(
        confirmLabel: String,
    ): Triple<LinearLayout, TextView, TextView> {
        val ink = getColor(R.color.ink)
        val hot = getColor(R.color.hot)
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.END
        }
        val cancel = TextView(this).apply {
            text = "Cancel"
            typeface = fontMono ?: Typeface.MONOSPACE
            textSize = 12f
            setTextColor(ink)
            setPadding(16.dp, 10.dp, 16.dp, 10.dp)
            isClickable = true; isFocusable = true
        }
        val confirm = TextView(this).apply {
            text = confirmLabel
            typeface = fontMonoSemibold ?: Typeface.MONOSPACE
            textSize = 12f
            setTextColor(hot)
            setPadding(16.dp, 10.dp, 16.dp, 10.dp)
            isClickable = true; isFocusable = true
        }
        row.addView(cancel, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT,
            ViewGroup.LayoutParams.WRAP_CONTENT))
        row.addView(confirm, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT,
            ViewGroup.LayoutParams.WRAP_CONTENT))
        return Triple(row, cancel, confirm)
    }

    /** Build a paper-themed dialog title — small monospace caps with a
     *  hairline letter-space, matching the LAYERS / BRUSH / COLOR
     *  section headers in the side panel. */
    private fun makePaperDialogTitle(text: String): TextView =
        TextView(this).apply {
            this.text = text.uppercase()
            typeface = fontMonoSemibold ?: Typeface.MONOSPACE
            textSize = 11f
            letterSpacing = 0.08f
            setTextColor(getColor(R.color.ink))
        }

    /** Wrap [content] in an AlertDialog whose window is paper-colored.
     *  Returns the dialog so the caller can dismiss it from button
     *  click handlers. */
    private fun showPaperDialog(content: View): android.app.AlertDialog {
        val dialog = AlertDialog.Builder(this)
            .setView(content)
            .create()
        dialog.show()
        dialog.window?.setBackgroundDrawable(
            android.graphics.drawable.ColorDrawable(getColor(R.color.paper)))
        return dialog
    }

    /** Paper-styled replacement for AlertDialog with title + message +
     *  Cancel/<confirmLabel>. */
    private fun showPaperConfirmDialog(
        title: String,
        message: String,
        confirmLabel: String = "OK",
        onConfirm: () -> Unit,
    ) {
        val pad = 18.dp
        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(getColor(R.color.paper))
            setPadding(pad, pad, pad, pad)
        }
        container.addView(makePaperDialogTitle(title),
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply { bottomMargin = 12.dp })
        container.addView(TextView(this).apply {
            text = message
            typeface = fontMono ?: Typeface.MONOSPACE
            textSize = 12f
            setTextColor(getColor(R.color.ink))
        }, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply { bottomMargin = 14.dp })

        val (buttons, cancelBtn, confirmBtn) = makePaperDialogButtons(confirmLabel)
        container.addView(buttons, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT))

        val dialog = showPaperDialog(container)
        cancelBtn.setOnClickListener  { dialog.dismiss() }
        confirmBtn.setOnClickListener { dialog.dismiss(); onConfirm() }
    }

    /** Paper-styled replacement for AlertDialog-with-EditText. The
     *  [onConfirm] callback receives the trimmed text; the dialog is
     *  dismissed before [onConfirm] runs (matches the legacy
     *  AlertDialog flow). */
    private fun showPaperInputDialog(
        title: String,
        initial: String,
        hint: String? = null,
        confirmLabel: String = "OK",
        onConfirm: (String) -> Unit,
    ) {
        val pad = 18.dp
        val ink     = getColor(R.color.ink)
        val inkSoft = getColor(R.color.inkSoft)
        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(getColor(R.color.paper))
            setPadding(pad, pad, pad, pad)
        }
        container.addView(makePaperDialogTitle(title),
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply { bottomMargin = 12.dp })

        val input = android.widget.EditText(this).apply {
            setText(initial)
            setSelection(initial.length)
            if (hint != null) this.hint = hint
            typeface = fontMono ?: Typeface.MONOSPACE
            textSize = 13f
            setTextColor(ink)
            backgroundTintList =
                android.content.res.ColorStateList.valueOf(inkSoft)
            isSingleLine = true
        }
        container.addView(input, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply { bottomMargin = 14.dp })

        val (buttons, cancelBtn, confirmBtn) = makePaperDialogButtons(confirmLabel)
        container.addView(buttons, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT))

        val dialog = showPaperDialog(container)
        cancelBtn.setOnClickListener  { dialog.dismiss() }
        confirmBtn.setOnClickListener {
            val text = input.text.toString().trim()
            dialog.dismiss()
            onConfirm(text)
        }
    }

    private fun showOverflowMenu(anchor: View) {
        showPaperPopupMenu(anchor, listOf(
            PaperMenuItem("Backup all documents…")              { backupAllDocuments() },
            PaperMenuItem("Import image…")                      { launchImageImport() },
            PaperMenuItem("Export canvas as PNG…")              { launchCanvasPngExport() },
            PaperMenuItem("Export document as PDF…")            { launchDocumentPdfExport() },
            PaperMenuItem("Stylus only: ${if (stylusOnly) "on" else "off"}")
                                                                { toggleStylusOnly() },
            PaperMenuItem("Delete selection")                   { userDeleteSelection() },
            PaperMenuItem("Rasterize selection to layer below") { userRasterizeSelectionBelow() },
            PaperMenuItem("Cut")                                { drawingView?.queueCutSelection() },
            PaperMenuItem("Copy")                               { drawingView?.queueCopySelection() },
            PaperMenuItem("Paste")                              { drawingView?.queuePasteSelection() },
        ))
    }

    private fun userRasterizeSelectionBelow() {
        if (!NativeRenderer.hasSelection()) {
            android.widget.Toast.makeText(this,
                "No vector selection",
                android.widget.Toast.LENGTH_SHORT).show()
            return
        }
        NativeRenderer.rasterizeSelectionToLayerBelow()
        drawingView?.forceRedraw()
        drawingView?.postDelayed({ syncLayerStateFromNative() }, 60L)
    }

    /** Default name for the active page's PNG export. Pages are 1-indexed
     *  in the user-facing name to match the page-number badge in the
     *  sidebar (which also displays "01", "02", etc.). */
    private fun defaultPngFilename(): String {
        val pageIdx = NativeRenderer.getActivePage()
        val docName = currentDocName.ifBlank { "Untitled" }
        return "${docName}_p${pageIdx + 1}.png"
    }

    private fun defaultPdfFilename(): String {
        val docName = currentDocName.ifBlank { "Untitled" }
        return "${docName}.pdf"
    }

    private fun launchCanvasPngExport() {
        canvasPngLauncher.launch(defaultPngFilename())
    }

    private fun launchDocumentPdfExport() {
        documentPdfLauncher.launch(defaultPdfFilename())
    }

    /**
     * Zip every document under documentsRoot() and write the archive
     * to the public Downloads folder. Survives uninstall of the app —
     * a safety net against accidental data loss from rebuild/reinstall
     * cycles (see the painful 2026-05-10 incident). Runs on a
     * background thread; UI is notified via Toast on completion.
     */
    private fun backupAllDocuments() {
        Thread {
            val docsRoot = documentsRoot()
            val message: String = try {
                // Make sure any in-flight tile saves have hit disk
                // before we snapshot the documents tree.
                NativeRenderer.flushTileWrites()

                if (!docsRoot.exists() ||
                    docsRoot.listFiles()?.isEmpty() != false) {
                    "No documents to back up."
                } else {
                    val ts = java.text.SimpleDateFormat(
                        "yyyy-MM-dd-HHmmss", java.util.Locale.US
                    ).format(java.util.Date())
                    val filename = "drawing-backup-$ts.zip"
                    val bytes = writeDocumentsZipToDownloads(docsRoot, filename)
                    "Saved Downloads/$filename (${humanBytes(bytes)})"
                }
            } catch (t: Throwable) {
                Log.e("DrawingApp", "backup failed", t)
                "Backup failed: ${t.message}"
            }
            runOnUiThread {
                android.widget.Toast.makeText(
                    this, message, android.widget.Toast.LENGTH_LONG
                ).show()
            }
        }.start()
    }

    /** Stream [docsRoot]'s tree into a zip in the system Downloads
     *  collection. Returns the total uncompressed byte count for the
     *  result toast. */
    private fun writeDocumentsZipToDownloads(
        docsRoot: File, filename: String
    ): Long {
        val values = android.content.ContentValues().apply {
            put(android.provider.MediaStore.Downloads.DISPLAY_NAME, filename)
            put(android.provider.MediaStore.Downloads.MIME_TYPE, "application/zip")
            put(android.provider.MediaStore.Downloads.RELATIVE_PATH,
                android.os.Environment.DIRECTORY_DOWNLOADS)
            put(android.provider.MediaStore.Downloads.IS_PENDING, 1)
        }
        val collection = android.provider.MediaStore.Downloads.getContentUri(
            android.provider.MediaStore.VOLUME_EXTERNAL_PRIMARY
        )
        val uri = contentResolver.insert(collection, values)
            ?: throw java.io.IOException("could not create Downloads entry")
        var totalBytes = 0L
        try {
            contentResolver.openOutputStream(uri).use { rawOut ->
                if (rawOut == null) throw java.io.IOException(
                    "could not open output stream for $uri")
                java.util.zip.ZipOutputStream(rawOut).use { zip ->
                    val basePath = docsRoot.absolutePath
                    docsRoot.walkTopDown()
                        .filter { it.isFile }
                        .forEach { file ->
                            val rel = file.absolutePath
                                .removePrefix(basePath)
                                .removePrefix(File.separator)
                                .replace(File.separatorChar, '/')
                            val entry = java.util.zip.ZipEntry("documents/$rel")
                            entry.time = file.lastModified()
                            zip.putNextEntry(entry)
                            java.io.FileInputStream(file).use { fin ->
                                totalBytes += fin.copyTo(zip)
                            }
                            zip.closeEntry()
                        }
                }
            }
            // Drop the pending flag so the file becomes visible to
            // the user in the Files app.
            val done = android.content.ContentValues().apply {
                put(android.provider.MediaStore.Downloads.IS_PENDING, 0)
            }
            contentResolver.update(uri, done, null, null)
        } catch (t: Throwable) {
            // Tear down the half-written file so the user doesn't end
            // up with a corrupt zip in Downloads.
            try { contentResolver.delete(uri, null, null) } catch (_: Throwable) {}
            throw t
        }
        return totalBytes
    }

    private fun humanBytes(n: Long): String {
        if (n < 1024) return "$n B"
        val kb = n / 1024.0
        if (kb < 1024) return String.format(java.util.Locale.US, "%.1f KB", kb)
        val mb = kb / 1024.0
        return String.format(java.util.Locale.US, "%.1f MB", mb)
    }

    /** Compute the natural export resolution. Falls back to the SurfaceView
     *  size when no page bounds are set (the doc behaves as an infinite
     *  plane in that mode). Capped to a sane upper bound so we don't try
     *  to allocate gigabytes for a thousand-page export. */
    private fun computeExportDimensions(): Pair<Int, Int> {
        val pw = NativeRenderer.getPageWidth()
        val ph = NativeRenderer.getPageHeight()
        if (pw > 0 && ph > 0) return Pair(pw.coerceAtMost(kExportMaxDim),
                                          ph.coerceAtMost(kExportMaxDim))
        val v = drawingView
        val w = (v?.width  ?: 1024).coerceAtLeast(1).coerceAtMost(kExportMaxDim)
        val h = (v?.height ?: 1024).coerceAtLeast(1).coerceAtMost(kExportMaxDim)
        return Pair(w, h)
    }

    private val kExportMaxDim = 4096

    /** Render the active page into a fresh bitmap on the GL thread,
     *  then write it to [uri] as a PNG. Failure → toast; the bitmap is
     *  always recycled. */
    private fun exportActivePageToPng(uri: Uri) {
        val v = drawingView ?: return
        val (w, h) = computeExportDimensions()
        val bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val pageIdx = NativeRenderer.getActivePage()
        v.queueExportRender(
            listOf(DrawingSurfaceView.ExportPage(pageIdx, bitmap))
        ) {
            try {
                contentResolver.openOutputStream(uri)?.use { os ->
                    bitmap.compress(Bitmap.CompressFormat.PNG, 100, os)
                }
                android.widget.Toast.makeText(this,
                    "Exported PNG", android.widget.Toast.LENGTH_SHORT).show()
            } catch (e: Exception) {
                android.util.Log.e("DrawingApp", "PNG export failed", e)
                android.widget.Toast.makeText(this,
                    "Export failed: ${e.message}",
                    android.widget.Toast.LENGTH_LONG).show()
            } finally {
                bitmap.recycle()
            }
        }
    }

    /** Render every page into its own bitmap on the GL thread, then
     *  build a multi-page PdfDocument with one bitmap per page (1 doc-px
     *  = 1 PDF point — keeps the on-paper geometry intact). */
    private fun exportDocumentToPdf(uri: Uri) {
        val v = drawingView ?: return
        val (w, h) = computeExportDimensions()
        val pageCount = NativeRenderer.getPageCount().coerceAtLeast(1)
        val bitmaps = (0 until pageCount).map {
            Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        }
        val renders = bitmaps.mapIndexed { i, b ->
            DrawingSurfaceView.ExportPage(i, b)
        }
        v.queueExportRender(renders) {
            try {
                val pdf = android.graphics.pdf.PdfDocument()
                for ((i, bmp) in bitmaps.withIndex()) {
                    val info = android.graphics.pdf.PdfDocument.PageInfo
                        .Builder(w, h, i + 1).create()
                    val page = pdf.startPage(info)
                    page.canvas.drawBitmap(bmp, 0f, 0f, null)
                    pdf.finishPage(page)
                }
                contentResolver.openOutputStream(uri)?.use { os ->
                    pdf.writeTo(os)
                }
                pdf.close()
                android.widget.Toast.makeText(this,
                    "Exported PDF · $pageCount page${if (pageCount == 1) "" else "s"}",
                    android.widget.Toast.LENGTH_SHORT).show()
            } catch (e: Exception) {
                android.util.Log.e("DrawingApp", "PDF export failed", e)
                android.widget.Toast.makeText(this,
                    "Export failed: ${e.message}",
                    android.widget.Toast.LENGTH_LONG).show()
            } finally {
                bitmaps.forEach { it.recycle() }
            }
        }
    }

    /** Open the system image picker. Result handled in the launcher
     *  registered in onCreate, which calls [decodeAndImportImage]. */
    private fun launchImageImport() {
        // GetContent's input is the MIME type filter.
        imagePickerLauncher.launch("image/*")
    }

    /** Decode a picked image, downsample if oversized, and hand the
     *  ARGB pixel array to native. The new layer + floating selection
     *  appear once the GL thread drains the action; we sync the layer
     *  panel + force a redraw so the user sees the import immediately. */
    private fun decodeAndImportImage(uri: Uri) {
        val bitmap = try {
            decodeBitmapBoundedTo(uri, kImportMaxDim)
        } catch (e: Exception) {
            android.util.Log.e("DrawingApp", "image import: decode failed", e)
            null
        }
        if (bitmap == null) {
            android.widget.Toast.makeText(this,
                "Couldn't load that image",
                android.widget.Toast.LENGTH_SHORT).show()
            return
        }
        val w = bitmap.width
        val h = bitmap.height
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)
        bitmap.recycle()
        val ok = NativeRenderer.importImageAsSelection(w, h, pixels)
        if (!ok) {
            android.widget.Toast.makeText(this,
                "Image import failed",
                android.widget.Toast.LENGTH_SHORT).show()
            return
        }
        drawingView?.forceRedraw()
        // The new layer becomes active in the drain; the panel needs to
        // reflect it. ~60ms gives the GL thread time to run the action.
        drawingView?.postDelayed({ syncLayerStateFromNative() }, 60L)
    }

    /** Decode the URI, downsampling on the way in if either dimension
     *  exceeds [maxDim]. Two-pass: first inJustDecodeBounds to get the
     *  raw size, then a real decode with inSampleSize. Returns ARGB_8888. */
    private fun decodeBitmapBoundedTo(uri: Uri, maxDim: Int): Bitmap? {
        // Probe size.
        val opts = android.graphics.BitmapFactory.Options().apply {
            inJustDecodeBounds = true
        }
        contentResolver.openInputStream(uri)?.use {
            android.graphics.BitmapFactory.decodeStream(it, null, opts)
        }
        val w0 = opts.outWidth
        val h0 = opts.outHeight
        if (w0 <= 0 || h0 <= 0) return null
        // Largest power-of-2 inSampleSize that keeps both dims ≥ maxDim.
        var sample = 1
        while (w0 / (sample * 2) >= maxDim || h0 / (sample * 2) >= maxDim) {
            sample *= 2
        }
        val realOpts = android.graphics.BitmapFactory.Options().apply {
            inSampleSize = sample
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        val bmp = contentResolver.openInputStream(uri)?.use {
            android.graphics.BitmapFactory.decodeStream(it, null, realOpts)
        } ?: return null
        // Final clamp: even after inSampleSize, the dimension can still
        // exceed maxDim by up to 2x. Scale down if so.
        val scale = (maxDim.toFloat() / maxOf(bmp.width, bmp.height)).coerceAtMost(1f)
        return if (scale < 1f) {
            val nw = (bmp.width  * scale).toInt().coerceAtLeast(1)
            val nh = (bmp.height * scale).toInt().coerceAtLeast(1)
            val scaled = Bitmap.createScaledBitmap(bmp, nw, nh, true)
            if (scaled !== bmp) bmp.recycle()
            scaled
        } else bmp
    }

    private val kImportMaxDim = 2048

    private fun showLayerOverflow(anchor: View) {
        showPaperPopupMenu(anchor, listOf(
            PaperMenuItem("+ Vector layer")     { userAddVectorLayer() },
            PaperMenuItem("Clear active layer") { userClearLayer() },
        ))
    }

    private fun showDocsMenu(anchor: View) {
        val items = mutableListOf<PaperMenuItem>()
        // Prefix the active doc with • so the user sees what's open.
        for (name in listDocumentNames()) {
            val label = if (name == currentDocName) "• $name" else "  $name"
            items += PaperMenuItem(label) {
                if (name != currentDocName) switchToDocument(name)
            }
        }
        items += PaperMenuItem("+ New document")  { userNewDocument() }
        items += PaperMenuItem("Rename current…") { showRenameDocDialog() }
        items += PaperMenuItem("Delete current")  { userDeleteCurrentDocument() }
        showPaperPopupMenu(anchor, items)
    }

    /** Rename the currently-open document. Validates non-empty, no
     *  collision with another doc, and that the underlying directory
     *  rename succeeds. After rename, points native at the new path
     *  (without going through loadDocument so undo state survives) and
     *  refreshes the status bar / sidebar mirrors. */
    private fun showRenameDocDialog() {
        val current = currentDocName
        showPaperInputDialog(
            title = "Rename document",
            initial = current,
            hint = "document name",
        ) { raw ->
            if (raw.isEmpty() || raw == current) return@showPaperInputDialog
            if (raw in listDocumentNames()) {
                android.widget.Toast.makeText(this,
                    "A document named “$raw” already exists",
                    android.widget.Toast.LENGTH_SHORT).show()
                return@showPaperInputDialog
            }
            renameCurrentDocument(raw)
        }
    }

    private fun renameCurrentDocument(newName: String) {
        val oldDir = docDirFor(currentDocName)
        val newDir = docDirFor(newName)
        if (!oldDir.exists() || !oldDir.renameTo(newDir)) {
            android.widget.Toast.makeText(this,
                "Couldn't rename document",
                android.widget.Toast.LENGTH_SHORT).show()
            return
        }
        currentDocName = newName
        rememberDocName(newName)
        // Point native I/O at the new path. setDocumentDir is safe to call
        // mid-session — it just stores the path; the in-memory layer
        // state stays put, and future tile saves go to the new location.
        // Avoids the undo-stack reset that loadDocument would trigger.
        NativeRenderer.setDocumentDir(newDir.absolutePath)
        if (::statusDocText.isInitialized) statusDocText.text = "doc · $newName"
    }

    // ====================================================================
    // State sync + action helpers (preserved from previous implementation)
    // ====================================================================

    private fun syncLayerStateFromNative() {
        val count = NativeRenderer.getLayerCount().coerceAtLeast(1)
        val active = NativeRenderer.getActiveLayer().coerceIn(0, count - 1)
        layerCount = count
        activeLayerIndex = active
        if (::layerListContainer.isInitialized) rebuildLayerList()
        updateStatusBarPage()
    }

    private fun userAddLayer() {
        NativeRenderer.addLayer()
        // Native creates the layer on the GL thread when the action
        // queue drains; sync after a beat so getLayerType / Count /
        // ActiveLayer reflect the new state. Without the delayed sync
        // the panel can render with stale type info — e.g. a freshly-
        // added vector layer briefly showing as raster.
        drawingView?.forceRedraw()
        drawingView?.postDelayed({ syncLayerStateFromNative() }, 60L)
    }

    private fun userAddVectorLayer() {
        NativeRenderer.addVectorLayer()
        drawingView?.forceRedraw()
        drawingView?.postDelayed({ syncLayerStateFromNative() }, 60L)
    }

    private fun userCycleLayer() {
        NativeRenderer.cycleActiveLayer()
        activeLayerIndex = (activeLayerIndex + 1) % layerCount
        rebuildLayerList()
    }

    private fun userClearLayer() {
        NativeRenderer.clearActiveLayer()
        drawingView?.forceRedraw()
    }

    private fun userDeleteSelection() {
        if (NativeRenderer.hasSelection() || NativeRenderer.hasRasterSelection()) {
            drawingView?.queueDeleteSelection()
        }
    }

    private fun toggleSnap() {
        snapEnabled = !snapEnabled
        NativeRenderer.setSnapEnabled(snapEnabled)
        if (::statusSnapText.isInitialized) {
            statusSnapText.text = if (snapEnabled) "snap: on" else "snap: off"
            statusSnapText.setTextColor(getColor(
                if (snapEnabled) R.color.hot else R.color.inkSoft))
        }
    }

    /** Flip the line-tool's angle-snap (15° increments). The state
     *  lives on DrawingSurfaceView since it's read on every shape-tool
     *  pen sample; we just mirror it into the status bar text. */
    private fun toggleAngleSnap() {
        val v = drawingView ?: return
        v.angleSnapEnabled = !v.angleSnapEnabled
        refreshAngleSnapStatus()
    }

    private fun refreshAngleSnapStatus() {
        if (!::statusAngleText.isInitialized) return
        val on = drawingView?.angleSnapEnabled == true
        statusAngleText.text = if (on) "angle: on" else "angle: off"
        statusAngleText.setTextColor(getColor(
            if (on) R.color.hot else R.color.inkSoft))
    }

    /** Push the latest pen position to the brush-preview overlay. Hides
     *  the overlay when the preview is disabled or the current tool
     *  isn't a raster stroke tool. Radius = max-pressure dab size in
     *  view-px (kBrushMaxRadiusDoc * brushSizeScale * viewScale). */
    private fun updateBrushPreview(viewPxX: Float, viewPxY: Float) {
        if (!::brushPreviewView.isInitialized) return
        if (!brushPreviewEnabled || !currentToolMirror.isRasterStroke) {
            brushPreviewView.hide()
            return
        }
        val viewScale = drawingView?.currentViewScale ?: 1.0f
        val radiusViewPx = kBrushMaxRadiusDoc * brushSizeScale * viewScale
        brushPreviewView.show(viewPxX, viewPxY, radiusViewPx)
    }

    private fun toggleBrushPreview() {
        brushPreviewEnabled = !brushPreviewEnabled
        if (!brushPreviewEnabled && ::brushPreviewView.isInitialized) {
            brushPreviewView.hide()
        }
        refreshBrushPreviewStatus()
    }

    private fun refreshBrushPreviewStatus() {
        if (!::statusPreviewText.isInitialized) return
        val on = brushPreviewEnabled
        statusPreviewText.text = if (on) "preview: on" else "preview: off"
        statusPreviewText.setTextColor(getColor(
            if (on) R.color.hot else R.color.inkSoft))
    }

    private fun togglePrediction() {
        predictionEnabled = !predictionEnabled
        drawingView?.predictionEnabled = predictionEnabled
        refreshPredictionStatus()
    }

    private fun refreshPredictionStatus() {
        if (!::statusPredictText.isInitialized) return
        val on = predictionEnabled
        statusPredictText.text = if (on) "predict: on" else "predict: off"
        statusPredictText.setTextColor(getColor(
            if (on) R.color.hot else R.color.inkSoft))
    }

    /** While the user is drawing a line with angle-snap on, the status
     *  bar shows the current snap angle (e.g. "angle: 45°") instead of
     *  the on/off label. Null = drag ended; revert. */
    private fun showLineAngle(deg: Float?) {
        if (!::statusAngleText.isInitialized) return
        if (deg == null) {
            refreshAngleSnapStatus()
            return
        }
        // Display as integer degrees in [0, 360).
        val rounded = ((deg.toInt() % 360) + 360) % 360
        statusAngleText.text = "angle: ${rounded}°"
        statusAngleText.setTextColor(getColor(R.color.hot))
    }

    /** Flip palm-rejection mode. Persisted across launches; the view's
     *  touch dispatcher reads stylusOnlyDrawing every event so the
     *  change takes effect immediately. */
    private fun toggleStylusOnly() {
        stylusOnly = !stylusOnly
        drawingView?.stylusOnlyDrawing = stylusOnly
        prefs().edit().putBoolean(kPrefStylusOnly, stylusOnly).apply()
    }

    private fun userUndo() {
        NativeRenderer.undo()
        drawingView?.forceRedraw()
        drawingView?.postDelayed({ syncLayerStateFromNative() }, 60L)
    }

    private fun userRedo() {
        NativeRenderer.redo()
        drawingView?.forceRedraw()
        drawingView?.postDelayed({ syncLayerStateFromNative() }, 60L)
    }

    private fun cycleGrid() {
        gridState = (gridState + 1) % 3
        applyPageSetupToNative()
        refreshPageSetupChips()
        writePageSetup(docDirFor(currentDocName))
        drawingView?.forceRedraw()
    }

    private fun toggleMargins() {
        marginsEnabled = !marginsEnabled
        applyPageSetupToNative()
        refreshPageSetupChips()
        writePageSetup(docDirFor(currentDocName))
        drawingView?.forceRedraw()
    }

    private fun gridChipLabel() = when (gridState) {
        0 -> "grid: off"; 1 -> "grid: lines"; else -> "grid: dots"
    }
    private fun marginChipLabel() =
        if (marginsEnabled) "margins: on" else "margins: off"

    /** Mirror the whole per-doc page setup into native. Called after
     *  any field changes and after a document switch. */
    private fun applyPageSetupToNative() {
        when (gridState) {
            0 -> NativeRenderer.setGridEnabled(false)
            1 -> { NativeRenderer.setGridStyle(1); NativeRenderer.setGridEnabled(true) }
            else -> { NativeRenderer.setGridStyle(2); NativeRenderer.setGridEnabled(true) }
        }
        NativeRenderer.setGridSpacing(gridSpacing.toFloat())
        NativeRenderer.setMarginsEnabled(marginsEnabled)
        NativeRenderer.setMarginSizes(marginTop.toFloat(), marginSide.toFloat())
    }

    private fun refreshPageSetupChips() {
        if (::statusGridText.isInitialized) {
            statusGridText.text = gridChipLabel()
            statusGridText.setTextColor(getColor(
                if (gridState != 0) R.color.hot else R.color.inkSoft))
        }
        if (::statusMarginText.isInitialized) {
            statusMarginText.text = marginChipLabel()
            statusMarginText.setTextColor(getColor(
                if (marginsEnabled) R.color.hot else R.color.inkSoft))
        }
    }

    /** Long-press on the grid chip: one slider for the minor-line
     *  spacing. Live — the canvas redraws as the thumb moves; the doc
     *  file is written when the thumb lifts. */
    private fun showGridConfigPopup(anchor: View) {
        showPaperSliderPopup(anchor, "Grid", listOf(
            PaperSliderSpec("size", kGridSpacingMin, kGridSpacingMax,
                            get = { gridSpacing },
                            set = { gridSpacing = it }),
        ))
    }

    /** Long-press on the margins chip: top/bottom and side insets. Opening
     *  the popup turns margins on if they're off, so the sliders have
     *  something to show. */
    private fun showMarginConfigPopup(anchor: View) {
        if (!marginsEnabled) {
            marginsEnabled = true
            applyPageSetupToNative()
            refreshPageSetupChips()
            drawingView?.forceRedraw()
        }
        showPaperSliderPopup(anchor, "Margins", listOf(
            PaperSliderSpec("top",  0, kMarginMax,
                            get = { marginTop },
                            set = { marginTop = it }),
            PaperSliderSpec("side", 0, kMarginMax,
                            get = { marginSide },
                            set = { marginSide = it }),
        ))
    }

    /** One slider row in a [showPaperSliderPopup]. [set] is called with
     *  the new value on every user drag step. */
    private class PaperSliderSpec(
        val label: String,
        val min: Int,
        val max: Int,
        val get: () -> Int,
        val set: (Int) -> Unit,
    )

    /** Paper-styled popup of labelled sliders, anchored to a status
     *  chip. Every drag step mirrors to native and redraws so the user
     *  sees the result live; the per-doc file is written on thumb lift
     *  (same cadence as the brush sliders' pref writes). */
    private fun showPaperSliderPopup(
        anchor: View, title: String, specs: List<PaperSliderSpec>,
    ) {
        val paper = getColor(R.color.paper)
        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(paper)
            setPadding(14.dp, 12.dp, 14.dp, 8.dp)
        }
        container.addView(makePaperDialogTitle(title),
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply { bottomMargin = 6.dp })

        for (spec in specs) {
            val label = TextView(this).apply {
                text = spec.label
                typeface = fontMono ?: Typeface.MONOSPACE
                textSize = 11f
                setTextColor(getColor(R.color.inkSoft))
            }
            val valueLabel = TextView(this).apply {
                typeface = fontMono ?: Typeface.MONOSPACE
                textSize = 11f
                setTextColor(getColor(R.color.ink))
                gravity = Gravity.END
                text = spec.get().toString()
            }
            val slider = SeekBar(this).apply {
                max = spec.max - spec.min
                progress = (spec.get() - spec.min).coerceIn(0, spec.max - spec.min)
                setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                    override fun onProgressChanged(sb: SeekBar?, p: Int, fromUser: Boolean) {
                        val v = p + spec.min
                        valueLabel.text = v.toString()
                        if (!fromUser) return
                        spec.set(v)
                        applyPageSetupToNative()
                        drawingView?.forceRedraw()
                    }
                    override fun onStartTrackingTouch(sb: SeekBar?) {}
                    override fun onStopTrackingTouch(sb: SeekBar?) {
                        writePageSetup(docDirFor(currentDocName))
                    }
                })
            }
            container.addView(buildSliderRow(label, slider, valueLabel),
                LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT))
        }

        val popup = android.widget.PopupWindow(
            container, 280.dp, ViewGroup.LayoutParams.WRAP_CONTENT,
            /*focusable=*/ true,
        )
        popup.setBackgroundDrawable(
            android.graphics.drawable.ColorDrawable(paper))
        popup.isOutsideTouchable = true
        popup.elevation = 8.dp.toFloat()
        // The chips live in the bottom status bar, so a plain drop-down
        // lands off-screen (the framework doesn't flip a PopupWindow
        // whose parent is the full-screen decor). Measure and shove it
        // up by its own height plus the anchor's so it sits just above
        // the chip.
        container.measure(
            View.MeasureSpec.makeMeasureSpec(280.dp, View.MeasureSpec.EXACTLY),
            View.MeasureSpec.makeMeasureSpec(0, View.MeasureSpec.UNSPECIFIED),
        )
        val yOff = -(anchor.height + container.measuredHeight + 6.dp)
        popup.showAsDropDown(anchor, 0, yOff, Gravity.START)
    }

    private fun togglePixelGrid() {
        pixelGridEnabled = !pixelGridEnabled
        NativeRenderer.setPixelGridEnabled(pixelGridEnabled)
        prefs().edit().putBoolean(kPrefPixelGrid, pixelGridEnabled).apply()
        if (::statusPixelGridText.isInitialized) {
            statusPixelGridText.text =
                if (pixelGridEnabled) "px: on" else "px: off"
            statusPixelGridText.setTextColor(getColor(
                if (pixelGridEnabled) R.color.hot else R.color.inkSoft))
        }
        drawingView?.forceRedraw()
    }

    // ---- Slider geometric mappings (unchanged from previous version) ----

    private fun progressToBrushScale(p: Int): Float =
        (kBrushSizeMin * Math.pow(
            (kBrushSizeMax / kBrushSizeMin).toDouble(), p / 100.0)).toFloat()

    private fun brushScaleToProgress(scale: Float): Int =
        (Math.log((scale / kBrushSizeMin).toDouble())
            / Math.log((kBrushSizeMax / kBrushSizeMin).toDouble())
            * 100).toInt().coerceIn(0, 100)

    private fun progressToVectorWidth(p: Int): Float =
        (kVectorWidthMin * Math.pow(
            (kVectorWidthMax / kVectorWidthMin).toDouble(), p / 100.0)).toFloat()

    private fun vectorWidthToProgress(w: Float): Int =
        (Math.log((w / kVectorWidthMin).toDouble())
            / Math.log((kVectorWidthMax / kVectorWidthMin).toDouble())
            * 100).toInt().coerceIn(0, 100)

    /** Slider position (0..100, target stroke opacity %) → per-dab α. */
    private fun progressToBrushAlpha(p: Int): Float {
        if (p >= 100) return 1.0f
        if (p <= 0)   return 0.0f
        val target = p / 100.0
        return (1.0 - Math.pow(1.0 - target, 1.0 / kAlphaDabsPerOverlap)).toFloat()
    }

    /** Per-dab α → slider position (target stroke opacity %). */
    private fun brushAlphaToProgress(a: Float): Int {
        if (a >= 1.0f) return 100
        if (a <= 0.0f) return 0
        val target = 1.0 - Math.pow(1.0 - a.toDouble(), kAlphaDabsPerOverlap)
        return (target * 100).toInt().coerceIn(0, 100)
    }

    // ---- Document I/O ---------------------------------------------------

    /** Top-level folder on the device's shared storage, browsable from
     *  the system Files app. Survives app uninstall. Requires the
     *  MANAGE_EXTERNAL_STORAGE permission to access — see
     *  ensureStoragePermissionThenInitDocs. */
    private fun documentsRoot(): File =
        File(android.os.Environment.getExternalStorageDirectory(),
             kDocumentsRootName).apply { mkdirs() }

    /** Legacy private location used before the shared-storage move.
     *  Read on first launch with the new build so existing documents
     *  migrate forward; written only by migration. */
    private fun legacyDocumentsRoot(): File =
        File(filesDir, "documents")

    private fun docDirFor(name: String): File = File(documentsRoot(), name)

    private fun listDocumentNames(): List<String> =
        documentsRoot().listFiles { f -> f.isDirectory }
            ?.map { it.name }
            ?.sorted()
            ?: emptyList()

    private fun nextUntitledName(): String {
        val existing = listDocumentNames().toSet()
        var n = 1
        while ("Untitled $n" in existing) n++
        return "Untitled $n"
    }

    /** Move filesDir/document/ → documentsRoot()/document/ on first run.
     *  Inherited from the original single-doc layout. */
    private fun migrateLegacyDocumentIfNeeded() {
        val legacy = File(filesDir, "document")
        if (!legacy.isDirectory) return
        val target = docDirFor("document")
        if (target.exists()) return
        if (!documentsRoot().exists()) documentsRoot().mkdirs()
        // Cross-volume rename usually fails — copy + delete is the
        // robust path.
        try {
            legacy.copyRecursively(target, overwrite = false)
            legacy.deleteRecursively()
        } catch (t: Throwable) {
            Log.e("DrawingApp", "failed to migrate legacy document/", t)
        }
    }

    /** Move docs from the old private filesDir/documents/ location to
     *  the new shared-storage location at /sdcard/Drafting Table/.
     *  Runs once on first launch after the relocation; subsequent runs
     *  find an empty legacy dir and no-op. Verifies the file count
     *  matches before deleting the source so a partial copy leaves
     *  both copies intact. */
    private fun migrateToExternalStorageIfNeeded() {
        val legacy = legacyDocumentsRoot()
        if (!legacy.isDirectory) return
        val newRoot = documentsRoot()
        val srcDocs = legacy.listFiles { f -> f.isDirectory } ?: return
        if (srcDocs.isEmpty()) return
        var migrated = 0
        for (srcDoc in srcDocs) {
            val targetDoc = File(newRoot, srcDoc.name)
            if (targetDoc.exists()) continue        // name collision; skip
            try {
                srcDoc.copyRecursively(targetDoc, overwrite = false)
                val srcCount = srcDoc.walkTopDown().count { it.isFile }
                val dstCount = targetDoc.walkTopDown().count { it.isFile }
                if (srcCount > 0 && srcCount == dstCount) {
                    srcDoc.deleteRecursively()
                    migrated++
                } else {
                    Log.e("DrawingApp",
                        "migration verify failed for ${srcDoc.name}: " +
                        "$srcCount → $dstCount; leaving both copies")
                }
            } catch (t: Throwable) {
                Log.e("DrawingApp", "migration failed for ${srcDoc.name}", t)
            }
        }
        if (migrated > 0) {
            Log.i("DrawingApp",
                "migrated $migrated document(s) to ${newRoot.absolutePath}")
        }
    }

    /** First-launch (and post-grant) doc-init entry point. Gated on
     *  the MANAGE_EXTERNAL_STORAGE permission — if it isn't granted,
     *  shows the permission dialog and returns; onResume retries
     *  after the user comes back from settings. */
    private fun ensureStoragePermissionThenInitDocs() {
        if (android.os.Environment.isExternalStorageManager()) {
            initializeDocuments()
        } else {
            showStorageAccessRequiredDialog()
        }
    }

    /** Run migrations, pick the initial doc, point native at it, and
     *  refresh the doc/layer sidebar. Idempotent — second invocation
     *  (e.g. if onResume races a partial first call) is a no-op. */
    private fun initializeDocuments() {
        if (docsInitialized) return
        docsInitialized = true

        migrateLegacyDocumentIfNeeded()
        migrateToExternalStorageIfNeeded()

        val available = listDocumentNames()
        val initialDoc = lastOpenedDocName()?.takeIf { it in available }
            ?: available.firstOrNull()
            ?: nextUntitledName()
        currentDocName = initialDoc
        rememberDocName(initialDoc)
        val docDir = docDirFor(initialDoc).apply { mkdirs() }
        NativeRenderer.setDocumentDir(docDir.absolutePath)
        loadPageSetup(docDir)

        drawingView?.post {
            syncLayerStateFromNative()
            rebuildSidebar()
            rebuildLayerList()
        }
    }

    /** Paper-styled modal dialog explaining the storage permission
     *  requirement and linking to the system "All files access"
     *  settings page. Not cancellable by tapping outside; the only
     *  ways out are "Open Settings" (deep-links to the granting page)
     *  or "Quit" (closes the app). */
    private fun showStorageAccessRequiredDialog() {
        if (storagePermissionDialog?.isShowing == true) return
        val pad = 18.dp
        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(getColor(R.color.paper))
            setPadding(pad, pad, pad, pad)
        }
        container.addView(makePaperDialogTitle("Storage permission required"),
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply { bottomMargin = 12.dp })
        container.addView(TextView(this).apply {
            text = "Documents live in /sdcard/$kDocumentsRootName/ so " +
                "they survive uninstalling the app and are browsable " +
                "from the Files app over USB.\n\n" +
                "Tap \"Open Settings\" and turn on \"Allow access to " +
                "manage all files\". Return here when done."
            typeface = fontMono ?: Typeface.MONOSPACE
            textSize = 12f
            setTextColor(getColor(R.color.ink))
        }, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply { bottomMargin = 14.dp })

        val (buttons, cancelBtn, confirmBtn) =
            makePaperDialogButtons("Open Settings")
        cancelBtn.text = "Quit"
        container.addView(buttons, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT))

        val dialog = showPaperDialog(container)
        dialog.setCancelable(false)
        dialog.setCanceledOnTouchOutside(false)
        cancelBtn.setOnClickListener {
            dialog.dismiss()
            finish()
        }
        confirmBtn.setOnClickListener {
            // Don't dismiss; the user comes back via Recents/back
            // after granting, and onResume picks up from there.
            val pkgUri = android.net.Uri.parse("package:$packageName")
            try {
                startActivity(android.content.Intent(
                    android.provider.Settings
                        .ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
                    pkgUri))
            } catch (_: android.content.ActivityNotFoundException) {
                // Some OEM builds don't deep-link to the per-app
                // toggle; fall back to the global settings page.
                startActivity(android.content.Intent(
                    android.provider.Settings
                        .ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION))
            }
        }
        storagePermissionDialog = dialog
    }

    private fun prefs() = getSharedPreferences(kPrefsName, Context.MODE_PRIVATE)
    private fun rememberDocName(name: String) {
        prefs().edit().putString(kPrefLastDoc, name).apply()
    }
    private fun lastOpenedDocName(): String? = prefs().getString(kPrefLastDoc, null)

    /** Switch to (or open) a document. If [sizeOverride] is non-null,
     *  the doc's page rect is reset to those dimensions and persisted —
     *  used by the new-document dialog. Otherwise we read the saved
     *  size from &lt;docDir&gt;/page_size.txt; legacy docs (no file) get
     *  their current surface dims written back as a self-heal step so
     *  every doc has a stable page rect from then on. */
    private fun switchToDocument(name: String, sizeOverride: Pair<Int, Int>? = null) {
        val dir = docDirFor(name).apply { mkdirs() }
        if (sizeOverride != null) {
            writePageSize(dir, sizeOverride.first, sizeOverride.second)
        }
        var size = readPageSize(dir)
        if (size == null) {
            // Legacy / never-saved doc: lock in the current canvas dims.
            // Falls back to a sensible default if the surface isn't laid
            // out yet (e.g. very early in onCreate).
            val v = drawingView
            val w = (v?.width  ?: 1024).coerceAtLeast(1)
            val h = (v?.height ?: 1024).coerceAtLeast(1)
            writePageSize(dir, w, h)
            size = Pair(w, h)
        }
        NativeRenderer.setPageBounds(0f, 0f, size.first.toFloat(), size.second.toFloat())
        currentDocName = name
        rememberDocName(name)
        loadPageSetup(dir)
        if (::statusDocText.isInitialized) statusDocText.text = "doc · $name"
        NativeRenderer.loadDocument(dir.absolutePath)
        drawingView?.forceRedraw()
        // Frame the new doc's page in the visible canvas, same as the
        // launch-time path does. post() so the layout listeners (which
        // update visibleLeftInset on panel toggles) get a chance to fire
        // first if anything is mid-resize when we get here.
        drawingView?.post { drawingView?.resetView() }
        lastBuiltPageCount = -1
        lastBuiltActivePage = -1
    }

    /** Open the new-document size dialog, then create the doc with the
     *  chosen size. Cancel falls back to no-op (no document is created). */
    private fun userNewDocument() {
        showNewDocumentSizeDialog { size ->
            switchToDocument(nextUntitledName(), size)
        }
    }

    /** Build and show the new-document size picker. Presets cover the
     *  common cases (current device default, US Letter, A4, a 2× hi-res
     *  default) plus a Custom row for arbitrary dimensions. The chosen
     *  size is delivered to [onPick]; Cancel does nothing. */
    private fun showNewDocumentSizeDialog(onPick: (Pair<Int, Int>) -> Unit) {
        val v = drawingView
        val defaultW = (v?.width  ?: 1024).coerceAtLeast(1)
        val defaultH = (v?.height ?: 1024).coerceAtLeast(1)

        // (label, w, h) — null entry = the Custom row.
        data class Preset(val label: String, val w: Int, val h: Int)
        val presets = listOf(
            Preset("Default · ${defaultW} × ${defaultH}",         defaultW, defaultH),
            Preset("Letter portrait · 1700 × 2200",               1700, 2200),
            Preset("Letter landscape · 2200 × 1700",              2200, 1700),
            Preset("A4 portrait · 1654 × 2339",                   1654, 2339),
            Preset("A4 landscape · 2339 × 1654",                  2339, 1654),
            Preset("High-res default · ${defaultW * 2} × ${defaultH * 2}",
                                                                  defaultW * 2, defaultH * 2)
        )

        val ink     = getColor(R.color.ink)
        val inkSoft = getColor(R.color.inkSoft)
        val paper   = getColor(R.color.paper)
        val hot     = getColor(R.color.hot)

        val pad = 18.dp
        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(paper)
            setPadding(pad, pad, pad, pad)
        }

        // Title row, styled like the layer/brush/color section headers
        // elsewhere in the app — small monospace caps with a hairline
        // letter-space.
        val title = TextView(this).apply {
            text = "NEW DOCUMENT"
            typeface = fontMonoSemibold ?: Typeface.MONOSPACE
            textSize = 11f
            letterSpacing = 0.08f
            setTextColor(ink)
        }
        container.addView(title, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT).apply { bottomMargin = 14.dp })

        // Selected preset index (or null if Custom is being edited).
        var selectedIdx: Int? = 0
        val radioGroup = android.widget.RadioGroup(this).apply {
            orientation = LinearLayout.VERTICAL
        }
        val radios = presets.mapIndexed { i, p ->
            android.widget.RadioButton(this).apply {
                text = p.label
                typeface = fontMono ?: Typeface.MONOSPACE
                textSize = 12f
                setTextColor(ink)
                buttonTintList = android.content.res.ColorStateList.valueOf(ink)
                isChecked = (i == 0)
                radioGroup.addView(this, LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT))
            }
        }
        // Custom radio + two number inputs.
        val customRadio = android.widget.RadioButton(this).apply {
            text = "Custom"
            typeface = fontMono ?: Typeface.MONOSPACE
            textSize = 12f
            setTextColor(ink)
            buttonTintList = android.content.res.ColorStateList.valueOf(ink)
            radioGroup.addView(this, LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT))
        }
        container.addView(radioGroup, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT))

        val customRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(28.dp, 0, 0, 0)
        }
        val widthEdit = android.widget.EditText(this).apply {
            setText(defaultW.toString())
            inputType = android.text.InputType.TYPE_CLASS_NUMBER
            isSingleLine = true
            filters = arrayOf<android.text.InputFilter>(android.text.InputFilter.LengthFilter(5))
            typeface = fontMono ?: Typeface.MONOSPACE
            setTextColor(ink)
            backgroundTintList = android.content.res.ColorStateList.valueOf(inkSoft)
        }
        val heightEdit = android.widget.EditText(this).apply {
            setText(defaultH.toString())
            inputType = android.text.InputType.TYPE_CLASS_NUMBER
            isSingleLine = true
            filters = arrayOf<android.text.InputFilter>(android.text.InputFilter.LengthFilter(5))
            typeface = fontMono ?: Typeface.MONOSPACE
            setTextColor(ink)
            backgroundTintList = android.content.res.ColorStateList.valueOf(inkSoft)
        }
        val xLabel = TextView(this).apply {
            text = " × "
            typeface = fontMono ?: Typeface.MONOSPACE
            textSize = 12f
            setTextColor(inkSoft)
        }
        customRow.addView(widthEdit,  LinearLayout.LayoutParams(0,
            ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        customRow.addView(xLabel)
        customRow.addView(heightEdit, LinearLayout.LayoutParams(0,
            ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        container.addView(customRow, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT))

        // Manage radio exclusivity manually since RadioGroup only owns
        // one set; clicking a preset deselects Custom and vice versa.
        val allButtons = radios + customRadio
        for ((i, btn) in allButtons.withIndex()) {
            btn.setOnClickListener {
                for (other in allButtons) other.isChecked = (other === btn)
                selectedIdx = if (btn === customRadio) null else i
                if (btn === customRadio) widthEdit.requestFocus()
            }
        }
        // Editing either field auto-selects Custom.
        val selectCustom = {
            for (other in allButtons) other.isChecked = (other === customRadio)
            selectedIdx = null
        }
        val watcher = object : android.text.TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
            override fun onTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
            override fun afterTextChanged(s: android.text.Editable?) {
                if (selectedIdx != null) selectCustom()
            }
        }
        widthEdit.addTextChangedListener(watcher)
        heightEdit.addTextChangedListener(watcher)

        // In-content button row — replaces the default AlertDialog
        // Cancel/Create chrome so the buttons inherit the same paper
        // background + monospace typography as the rest of the dialog.
        val buttonRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.END
        }
        val cancelBtn = TextView(this).apply {
            text = "Cancel"
            typeface = fontMono ?: Typeface.MONOSPACE
            textSize = 12f
            setTextColor(ink)
            setPadding(16.dp, 10.dp, 16.dp, 10.dp)
            isClickable = true; isFocusable = true
        }
        val createBtn = TextView(this).apply {
            text = "Create"
            typeface = fontMonoSemibold ?: Typeface.MONOSPACE
            textSize = 12f
            setTextColor(hot)            // sienna accent — primary action
            setPadding(16.dp, 10.dp, 16.dp, 10.dp)
            isClickable = true; isFocusable = true
        }
        buttonRow.addView(cancelBtn, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT,
            ViewGroup.LayoutParams.WRAP_CONTENT))
        buttonRow.addView(createBtn, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT,
            ViewGroup.LayoutParams.WRAP_CONTENT))
        container.addView(buttonRow, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT).apply { topMargin = 12.dp })

        val dialog = showPaperDialog(container)
        cancelBtn.setOnClickListener { dialog.dismiss() }
        createBtn.setOnClickListener {
            val sizeIdx = selectedIdx
            val size = if (sizeIdx != null) {
                Pair(presets[sizeIdx].w, presets[sizeIdx].h)
            } else {
                val w = widthEdit.text.toString().toIntOrNull()
                    ?.coerceIn(64, 8192)
                val h = heightEdit.text.toString().toIntOrNull()
                    ?.coerceIn(64, 8192)
                if (w == null || h == null) {
                    android.widget.Toast.makeText(this,
                        "Custom size must be 64–8192",
                        android.widget.Toast.LENGTH_SHORT).show()
                    return@setOnClickListener
                }
                Pair(w, h)
            }
            dialog.dismiss()
            onPick(size)
        }
    }

    private val kPageSizeFile = "page_size.txt"

    private fun readPageSize(dir: java.io.File): Pair<Int, Int>? {
        val f = java.io.File(dir, kPageSizeFile)
        if (!f.exists()) return null
        val txt = runCatching { f.readText().trim() }.getOrNull() ?: return null
        val parts = txt.split('x', 'X', '×')
        if (parts.size != 2) return null
        val w = parts[0].trim().toIntOrNull() ?: return null
        val h = parts[1].trim().toIntOrNull() ?: return null
        if (w <= 0 || h <= 0) return null
        return Pair(w, h)
    }

    private fun writePageSize(dir: java.io.File, w: Int, h: Int) {
        runCatching {
            java.io.File(dir, kPageSizeFile).writeText("${w}x${h}")
        }.onFailure { android.util.Log.e("DrawingApp", "writePageSize failed", it) }
    }

    /** Per-document page setup (grid state + spacing, margins) lives
     *  in <docDir>/page_setup.txt as key=value lines. Unknown keys are
     *  ignored and missing keys keep their current value, so the format
     *  can grow without a migration. */
    private val kPageSetupFile = "page_setup.txt"

    /** Load [dir]'s page setup into the mirrors, push it to native, and
     *  refresh the chips. A doc with no file (legacy, or freshly
     *  created) inherits whatever setup is currently active — so a new
     *  notebook starts out like the one you were just in — and gets the
     *  file written so it's stable from then on. */
    private fun loadPageSetup(dir: java.io.File) {
        val f = java.io.File(dir, kPageSetupFile)
        val text = if (f.exists()) runCatching { f.readText() }.getOrNull() else null
        if (text != null) {
            for (line in text.lines()) {
                val eq = line.indexOf('=')
                if (eq <= 0) continue
                val key = line.substring(0, eq).trim()
                val v = line.substring(eq + 1).trim()
                when (key) {
                    "grid"         -> v.toIntOrNull()?.let { gridState = it.coerceIn(0, 2) }
                    "grid_spacing" -> v.toIntOrNull()?.let {
                        gridSpacing = it.coerceIn(kGridSpacingMin, kGridSpacingMax) }
                    "margins"      -> marginsEnabled = (v == "1" || v == "true")
                    "margin_top"   -> v.toIntOrNull()?.let { marginTop = it.coerceIn(0, kMarginMax) }
                    "margin_side"  -> v.toIntOrNull()?.let { marginSide = it.coerceIn(0, kMarginMax) }
                }
            }
        } else {
            writePageSetup(dir)
        }
        applyPageSetupToNative()
        refreshPageSetupChips()
    }

    private fun writePageSetup(dir: java.io.File) {
        runCatching {
            java.io.File(dir, kPageSetupFile).writeText(
                "grid=$gridState\n" +
                "grid_spacing=$gridSpacing\n" +
                "margins=${if (marginsEnabled) 1 else 0}\n" +
                "margin_top=$marginTop\n" +
                "margin_side=$marginSide\n")
        }.onFailure { android.util.Log.e("DrawingApp", "writePageSetup failed", it) }
    }

    /** Called once, the first time the SurfaceView has dims. Reads the
     *  initial document's saved page size if present; otherwise locks
     *  in the surface dims as the doc's page size and writes the file
     *  so future loads of this doc are deterministic. Schedules a fit-
     *  to-canvas reset on the next frame so launches open with the
     *  whole page visible (mirrors the rail's reset-view button) — the
     *  post() defers until panelsRow's layout listener has updated the
     *  visibleLeftInset, otherwise the fit math is off by the panel
     *  width. */
    private fun applyInitialPageBounds(surfaceW: Int, surfaceH: Int) {
        val dir = docDirFor(currentDocName).apply { mkdirs() }
        val saved = readPageSize(dir)
        val (w, h) = if (saved != null) saved else {
            writePageSize(dir, surfaceW, surfaceH)
            Pair(surfaceW, surfaceH)
        }
        NativeRenderer.setPageBounds(0f, 0f, w.toFloat(), h.toFloat())
        drawingView?.post { drawingView?.resetView() }
    }

    private fun userDeleteCurrentDocument() {
        val toDelete = currentDocName
        if (toDelete.isEmpty()) return
        showPaperConfirmDialog(
            title = "Delete document",
            message = "Delete “$toDelete”? This can't be undone.",
            confirmLabel = "Delete",
        ) { performDelete(toDelete) }
    }

    private fun performDelete(name: String) {
        val others = listDocumentNames().filter { it != name }
        val target = others.firstOrNull() ?: nextUntitledName()
        switchToDocument(target)
        val ok = docDirFor(name).deleteRecursively()
        if (!ok) android.util.Log.e("DrawingApp", "deleteRecursively($name) failed")
    }

    // ---- Page sidebar wiring -------------------------------------------

    private fun userAddPage() {
        NativeRenderer.addPage()
        drawingView?.forceRedraw()
    }

    /** Per-thumbnail overflow menu: just Delete for now. Disabled when
     *  there's only one page (the document needs at least one). */
    private fun showPageOverflow(anchor: View, idx: Int) {
        showPaperPopupMenu(anchor, listOf(
            PaperMenuItem("Delete page",
                          enabled = NativeRenderer.getPageCount() > 1)
                { confirmDeletePage(idx) },
        ))
    }

    private fun confirmDeletePage(idx: Int) {
        showPaperConfirmDialog(
            title = "Delete page",
            message = "Delete page ${idx + 1}? This can't be undone.",
            confirmLabel = "Delete",
        ) { userDeletePage(idx) }
    }

    private fun userDeletePage(idx: Int) {
        if (NativeRenderer.getPageCount() <= 1) return
        NativeRenderer.deletePage(idx)
        drawingView?.forceRedraw()
        // Native applies the delete on the GL thread next op. Rebuild the
        // sidebar after a beat so getPageCount/getActivePage reflect the
        // post-drain state. syncLayerStateFromNative covers the layer
        // mirror because the active page's layers may differ.
        drawingView?.postDelayed({
            rebuildSidebar()
            syncLayerStateFromNative()
        }, 60L)
    }

    private fun userMovePage(from: Int, to: Int) {
        val count = NativeRenderer.getPageCount()
        if (from == to || from < 0 || to < 0 || from >= count || to >= count) return
        NativeRenderer.movePage(from, to)
        drawingView?.forceRedraw()
        drawingView?.postDelayed({
            rebuildSidebar()
            syncLayerStateFromNative()
        }, 60L)
    }

    /** Bind the drag-to-reorder touch handler onto the handle ImageView for
     *  the page item at native [idx]. Same gesture vocabulary as the
     *  layer panel's handle, but operates on pageItems / sidebarLayout
     *  for displacement. */
    private fun installPageDragHandleListener(handle: View, idx: Int, container: View) {
        handle.setOnTouchListener { v, ev ->
            when (ev.actionMasked) {
                MotionEvent.ACTION_DOWN -> {
                    if (NativeRenderer.getPageCount() <= 1) return@setOnTouchListener false
                    dragPageSourceIdx = idx
                    dragPageSourceView = container
                    dragPageStartRawY = ev.rawY
                    dragPageStartTranslationY = container.translationY
                    // Slot height = container + topMargin (sidebarItemParams).
                    dragPageSlotHeightPx = container.height + 6.dp
                    dragPageCurrentTargetIdx = idx
                    container.elevation = 6.dp.toFloat()
                    container.alpha = 0.95f
                    // Stop the ScrollView ancestor from intercepting the
                    // gesture mid-drag (it would otherwise grab on
                    // vertical movement).
                    v.parent?.requestDisallowInterceptTouchEvent(true)
                    true
                }
                MotionEvent.ACTION_MOVE -> {
                    if (dragPageSourceIdx != idx || dragPageSourceView !== container) {
                        return@setOnTouchListener false
                    }
                    val dy = ev.rawY - dragPageStartRawY
                    container.translationY = dragPageStartTranslationY + dy
                    updatePageDragDisplacement(container)
                    true
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    if (dragPageSourceIdx != idx || dragPageSourceView !== container) {
                        return@setOnTouchListener false
                    }
                    val targetIdx = dragPageCurrentTargetIdx
                    val sourceIdx = dragPageSourceIdx
                    container.elevation = 0f
                    container.alpha = 1.0f
                    container.translationY = 0f
                    resetAllPageDisplacements()
                    dragPageSourceIdx = -1
                    dragPageSourceView = null
                    dragPageCurrentTargetIdx = -1

                    val count = NativeRenderer.getPageCount()
                    if (targetIdx in 0 until count && targetIdx != sourceIdx) {
                        userMovePage(sourceIdx, targetIdx)
                    }
                    true
                }
                else -> false
            }
        }
    }

    /** While a page item is being dragged, shift the other items so a
     *  visual gap follows the dragged thumb. Called on every MOVE event;
     *  also updates [dragPageCurrentTargetIdx] (used at UP). Mirrors the
     *  layer-row version but operates on pageItems where uiPos == idx. */
    private fun updatePageDragDisplacement(draggedContainer: View) {
        val sourceIdx = dragPageSourceIdx
        if (sourceIdx < 0) return
        val draggedCenter = draggedContainer.top + draggedContainer.translationY +
                            draggedContainer.height / 2f
        var newIdx = 0
        for (item in pageItems) {
            if (item.pageIdx == sourceIdx) continue
            val originalCenter = item.container.top + item.container.height / 2f
            if (originalCenter < draggedCenter) newIdx++
        }
        if (newIdx == dragPageCurrentTargetIdx) return
        dragPageCurrentTargetIdx = newIdx
        for (item in pageItems) {
            if (item.pageIdx == sourceIdx) continue
            val origIdx = item.pageIdx
            val displayedIdx = when {
                sourceIdx < newIdx ->
                    if (origIdx in (sourceIdx + 1)..newIdx) origIdx - 1 else origIdx
                sourceIdx > newIdx ->
                    if (origIdx in newIdx..(sourceIdx - 1)) origIdx + 1 else origIdx
                else -> origIdx
            }
            val target = ((displayedIdx - origIdx) * dragPageSlotHeightPx).toFloat()
            item.container.animate().translationY(target).setDuration(120L).start()
        }
    }

    private fun resetAllPageDisplacements() {
        for (item in pageItems) item.container.translationY = 0f
    }

    private fun onThumbnailsUpdated() {
        for (item in pageItems) item.imageView.invalidate()
        val pc = NativeRenderer.getPageCount()
        val ap = NativeRenderer.getActivePage()
        if (pc != lastBuiltPageCount) {
            syncLayerStateFromNative()
            rebuildSidebar()
        } else if (ap != lastBuiltActivePage) {
            // Page switch only. Layers are per-page so the layer panel
            // still needs a resync, but the sidebar's view tree and
            // thumbnails are unchanged — just move the highlight.
            syncLayerStateFromNative()
            updateSidebarActivePage(ap)
        }
        // Layer count / active-layer drift: catches the case where a
        // queued action (merge, rasterize, delete, add) drains AFTER
        // the postDelayed-60ms sync from its caller. Without this
        // check, e.g. a merge that takes 100+ms can leave the panel
        // showing the just-merged source layer until the next
        // unrelated user action triggers another sync.
        val lc = NativeRenderer.getLayerCount()
        val al = NativeRenderer.getActiveLayer()
        if (lc != layerCount || al != activeLayerIndex) {
            syncLayerStateFromNative()
        }
        updateStatusBarPage()
    }

    // ---- Fonts (downloadable) ------------------------------------------

    private fun loadFonts() {
        // The synchronous ResourcesCompat.getFont throws Resources$
        // NotFoundException when downloadable fonts can't be resolved
        // on the calling thread (e.g. provider unreachable, GMS query
        // pending, signature mismatch). We never want that to crash
        // the activity — fall back to the system mono/sans Typeface,
        // which the rest of the UI consumes through `?:` defaults.
        //
        // The async path (ResourcesCompat.getFont with FontCallback)
        // would let the real fonts swap in once they download; we can
        // wire that in a follow-up if the system mono isn't acceptable.
        fontMono          = tryLoadFont(R.font.jetbrains_mono)
        fontMonoSemibold  = tryLoadFont(R.font.jetbrains_mono_semibold)
        fontInter         = tryLoadFont(R.font.inter)
        fontInterSemibold = tryLoadFont(R.font.inter_semibold)
    }

    private fun tryLoadFont(@androidx.annotation.FontRes id: Int): Typeface? = try {
        ResourcesCompat.getFont(this, id)
    } catch (t: Throwable) {
        android.util.Log.w("DrawingApp",
            "downloadable font ${resources.getResourceEntryName(id)} unavailable: $t")
        null
    }

    // ---- Helpers --------------------------------------------------------

    private val Int.dp: Int
        get() = (this * resources.displayMetrics.density).toInt()
}

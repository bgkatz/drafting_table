# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Custom Android sketching / technical-notebook app for the Wacom MovinkPad 11 (standalone Android tablet, EMR pen). Single-developer, single-device target. The user's overriding priority is **input responsiveness** — feature breadth is secondary. See `NOTES.md` for the original design intent and `README.md` for first-time setup.

## Build & run

Builds and runs are driven from Android Studio. There is no Gradle wrapper checked in; if `gradlew.bat` is missing, opening the project in Android Studio regenerates it (or run `gradle wrapper --gradle-version 8.7` once).

```powershell
.\gradlew.bat installDebug
adb shell am start -n com.bk.drawing/.MainActivity
adb logcat -s DrawingApp        # native log tag for renderer.cpp
```

There is no test suite. Validation is done by running on the tablet and observing behavior. UI-impacting changes need to be tested on-device — type-checking and "compiles" do not constitute validation for this codebase.

Build details that bite if you forget them:
- **arm64-v8a only** (`abiFilters` in `app/build.gradle.kts`); the MovinkPad is the sole target. Don't add other ABIs.
- **min SDK 33** so `GLFrontBufferedRenderer` is available without backports.
- **C++20**, NDK side-by-side, OpenGL ES 3.2.

## Architecture

Two-language split: Kotlin owns lifecycle, UI, and touch dispatch; C++ owns all GL state and pixel work. The interesting code is concentrated in four files.

### Source layout

```
app/src/main/
├── java/com/bk/drawing/
│   ├── MainActivity.kt          Activity, button panel, key handling, layer-state mirror
│   ├── DrawingSurfaceView.kt    Touch dispatch, view transform, gesture handling, GLFrontBufferedRenderer callbacks
│   └── NativeRenderer.kt        JNI surface (object NativeRenderer; loads libdrawing.so)
└── cpp/
    ├── CMakeLists.txt
    └── renderer.cpp             Single-TU native renderer (~4000 lines). All shaders, persistence, undo, etc.
```

### Document model (mirror in `renderer.cpp` top comment)

```
Document
  └── Layer[]                  z-ordered, bottom-up
        ├── Raster: sparse map<(tx,ty), Tile>   256×256 RGBA8 tiles, lazily allocated
        └── Vector: Line[] / Rect[] / Ellipse[] / Circle[]
```

Tiles are stored **premultiplied alpha** on disk and on the GPU; both bake and composite use `glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA)`. On-disk layout is `<docDir>/layer_<idx>/tile_<tx>_<ty>.bin` (raw RGBA bytes) plus `shapes.bin` for vector layers.

The "paper" is implicit — the multi-buffer is cleared to off-canvas gray, then a paper-white quad is drawn over the page bounds, then layers composite on top.

### Coordinate spaces — critically important

There are three coordinate spaces and confusing them is the most common bug in this codebase:

- **doc-px**: document coordinates. The natural unit for shapes, tiles, snap targets, stroke samples, and persistence.
- **view-px**: on-screen pixels in the SurfaceView (`MotionEvent.x/y`).
- **buffer-px**: pixels of the GL buffer; may be rotated relative to view-px on some device orientations (the framework provides a `view→buffer` mat4 in each callback).

**Kotlin owns the doc→view view transform** (scale/rotation/pan from 2-finger gestures) and converts touch coords to doc-px before any JNI call. **The `transform: FloatArray` parameter passed into render JNIs is always doc→buffer**, composed in `DrawingSurfaceView.composedTransform()` via `Matrix.multiplyMM`. Native consumes only this composed matrix, so the view transform is invisible to most native code.

The current view scale is mirrored into native via `setViewScale` so snap radii, selection handle sizes, and hit-test thresholds stay constant in *screen* pixels (constants ending in `ViewPx` are divided by `currentViewScale()` at use time).

### Threading & cross-thread mutation

- **GL thread** (managed by `GLFrontBufferedRenderer`): runs both `onDrawFrontBufferedLayer` and `onDrawMultiBufferedLayer`. All GL state — texture creation, FBO binds, draw calls — must happen here.
- **UI thread**: runs touch dispatch and JNI calls invoked by buttons.

JNI methods that mutate `g_layers` from the UI thread (`addLayer`, `cycleActiveLayer`, `clearActiveLayer`, `addVectorLayer`, `addLine`/`addRectangle`/etc., `undo`, `redo`) **enqueue actions** rather than mutating directly. The GL thread drains the queue at the start of every operation via `applyPendingLayerActions()` and `applyPendingShapes()`. Constants:

```
kActionAddLayer / kActionCycleActive / kActionClearActive
kActionAddVectorLayer / kActionUndo / kActionRedo
```

`undo`/`redo` go through this queue specifically because the inverse mutations may touch GPU textures and disk — they must run on the GL thread.

Selection mutations (`hitTestActiveVectorLayer`, `deleteSelection`, transform drags) read/write `g_selection` under `g_selectionMutex` and *do* mutate from the UI thread directly (a pre-existing race that is acceptable for this single-user app).

### Front-buffer / multi-buffer pipeline

`GLFrontBufferedRenderer<StrokeAction>` provides:
- `renderFrontBufferedLayer(action)` — low-latency render for live stroke / shape preview.
- `commit()` — flushes to multi-buffer; framework calls `onDrawMultiBufferedLayer` with the accumulated actions.

`StrokeAction` is a sealed class: `BatchSamples(xyp, realCount, isNewStroke, …)` for brush/eraser, `ShapePreview(shapeType, x0..y1, snapped)` for shape tools, and `LassoPreview(points, closed)` for raster lasso. Sample coordinates are doc-px (Kotlin converts before packing the action).

`BatchSamples` packs every sample from a single `MotionEvent` into one dispatch: real samples (the historical rows plus the current) followed by an optional predicted tail from `MotionEventPredictor`. `realCount` marks the real/predicted boundary. One `renderFrontBufferedLayer` call per `MotionEvent`, one native call (`extendStrokeBatch`), and one preview-overlay render per batch — keeps GL-thread work constant-time regardless of how many samples or how long the stroke. Motion prediction is gated by `g_predictionEnabled` (mirrored from `predictionEnabled` on the surface view, status-bar toggle; defaults on). The native side keeps a `g_coverageReal` FBO mirror and an emitter snapshot so it can revert predicted dabs from the live preview before applying the next real batch.

Stroke lifecycle:
1. `ACTION_DOWN` → `BatchSamples(realCount=1, isNewStroke=true)` → native `beginStroke` clears `g_current.samples` and snapshots `g_predictionEnabled` into a per-stroke flag.
2. `ACTION_MOVE` → `BatchSamples(real + predicted tail)` → native `extendStrokeBatch` reverts any pending prediction, applies real dabs (mirror after), applies predicted dabs (if active), and renders the front-buffer overlay once.
3. `ACTION_UP` → `r.commit()` → native `commitStroke` bakes the real samples into tile FBOs (predicted samples are never persisted), snapshots tiles for undo, saves to disk.

### Shade tool (closed-path fill)

`Tool.SHADE` (native id 7, `kToolShade`) runs the *same* stroke pipeline as the brush; the stroke's samples additionally describe a closed path (implicit chord from the last sample back to the first) whose interior is filled. Rules, all deliberate: the chord is **not** stroked, only filled to; the region uses **nonzero winding** so self-crossing scribbles come out solid; and outline + fill are one flat coverage mask (`beginStroke` forces `g_strokeUniformAlpha`) so the shared border can't darken at partial opacity.

Filling is `kPolyVS/FS` + a two-pass stencil (winding via `GL_INCR/DECR_WRAP` on a triangle fan, then a cover quad with `GL_NOTEQUAL 0`). Consequences worth knowing:
- The two FBOs it fills through — `g_shadeCoverage` (screen-sized) and `g_strokeCoverageTile` — are the only ones created with a stencil renderbuffer (`ensureViewFbo(..., withStencil=true)`). **Without a stencil attachment the stencil test always passes and the whole path AABB fills**, so keep that flag if you touch those call sites.
- The dab half of the coverage accumulates in `g_coverage` exactly as a normal stroke; the fill half is **rebuilt every batch** into `g_shadeCoverage` (the chord sweeps, so the region can shrink) and merged with the dabs under `GL_MAX`. The rebuild is one fan draw scissored to a monotonically-growing bbox — don't "optimize" it into accumulating, and don't rebuild the dabs, which would make per-batch cost O(stroke length).
- Bake reuses the uniform-alpha per-tile coverage path, drawing the same uploaded path per tile with the tile origin folded into `uTransform` (so the poly program's page clip is doc-space, unlike the dab program's tile-local one). Interior tiles hold no samples, so the tile-touched test additionally checks corner/centre-inside-path and chord-crosses-tile.
- `g_strokeTool` is no longer "0 = brush, anything else = eraser" — use `strokeToolIsEraser()`.
- The chord is the only stretch of the fill boundary with no dabs on it, so `kPolyFS` feathers it: distance to the chord **segment** (not its line — a concave region can wrap past the chord's ends), faded on the same `(d/band)^4` curve as `kDabFS`. `band = widest dab radius in the stroke * (1 - hardness)`, computed CPU-side by `chordFeatherBand()`. **Do not size it from the dab radii at the chord's own endpoints** — those are the pen-down and pen-lift samples, `mapPressure` returns ~0 there and `kMinRadius` is 0, so the band collapses and the feather silently vanishes (shipped that bug once). The stencil pass must run with alpha 1 and the feather off — it shares the fragment shader, and its `discard` at zero alpha would otherwise drop stencil writes and punch holes in the fill.
- The chord bridges the two sample *centres*, so dabs spill past it and read as nubs on the straight edge. Fixed by offsetting the fill's edge outward by `uChordOffset`, computed by `chordOutwardOffset()` as `max over samples of (outward distance from the chord line + that sample's dab radius)` — i.e. tangent to the outermost dab. Constant along the chord, so the bridge stays straight. Two traps here, both already hit once: it must be a **max, not an average**, because pressure ramps up at pen-down and falls at pen-lift so the biggest dab near either end is several samples in from the tip (an average sits inside it and it protrudes); and it must measure **reach, not radius**, so dabs set back inside the shape don't inflate it. Capped at the stroke's widest dab, which bounds the spill when a path swings outside the chord line. The offset region lies outside the polygon, so `drawShadeFill` adds a third pass — an oriented box around the chord, stencil off, `uChordMode = 2`.
- `chordOutwardNormal()` resolves which side is "outward" from the polygon's signed area, then confirms it with `pointInPolygonNonzero` at a probe point — the confirmation is what covers self-intersecting paths, where signed area alone lies.
- The fill (polygon + chord) is built from **real samples only** — `extendStrokeBatch` deliberately withholds the predicted tail from it while the dabs still get it. Prediction buys tip latency; the chord is a long edge anchored at the far end of the shape, so the predictor's overshoot just makes the bridge and its offset width shimmer, worse the larger the brush. It also keeps the preview's chord identical to the one the bake commits. Withholding it also steadies the offset, which is derived from those same samples and scales with brush size — a jittery reading makes the bridge's edge breathe, badly on a large brush.
- Both chord passes measure **unsigned** distance to the chord segment. That's load-bearing: a signed/half-plane test would let the fade reach across a self-crossing scribble that legitimately spans the chord and gouge it, while an unsigned distance puts every far-off interior fragment deep in the solid range.

### Undo / redo

Single global stack on the native side (`g_undoStack` / `g_redoStack`, `std::deque<UndoEntry>`, mutex-guarded). Bounded by **50 entries / 200 MB combined**; oldest entries evict first. Not persisted across app restart.

Six op types (`UndoOp`): `RasterStroke`, `VectorAdd`, `VectorDelete`, `VectorMutate`, `LayerClear`, `LayerAdd`. Each entry stores enough to apply forward and reverse — for raster strokes that's tile snapshots before+after the bake (the dominant memory cost). For vector ops it's a `ShapeData` tagged union.

Push points (search for `pushUndoEntry`):
- `commitStroke` after bake (RasterStroke)
- `applyPendingShapes` per shape applied (VectorAdd)
- `deleteSelection` (VectorDelete)
- `endInteraction` if shape changed during a transform drag (VectorMutate)
- `applyPendingLayerActions` for `kActionClearActive` and add-layer paths

### Page bounds (canvas rectangle)

The doc is mathematically infinite, but `setPageBounds(x0, y0, x1, y1)` defines a fixed-size canvas. When active:

- Tiles fully outside the page are not created or baked.
- Boundary tiles use intersected dab clip bounds AND fragment-shader-level discard (`vDocPos` varying + `uPageMin/Max/Active` uniforms in dab/line/grid programs) for clean truncation.
- Compositor clears to gray and paints paper-white over the page rect via a tiny `kFillVS/FS` program.
- UI affordances (selection handles, snap markers, the page outline itself) explicitly **un-set** the page-clip uniforms so they stay visible at edges.

Default page bounds = initial surface dimensions, set once from `onSizeChanged`.

### Page setup (grid + margins)

Grid on/off/style, grid spacing, and the margin rules are **per-document**, stored in `<docDir>/page_setup.txt` as `key=value` lines (`grid`, `grid_spacing`, `margins`, `margin_top`, `margin_side`) next to `page_size.txt`. `loadPageSetup` runs on every doc switch and at startup; a doc with no file inherits the currently active setup and gets the file written. Native mirrors are atomics (`g_gridSpacing`, `g_marginEnabled`, `g_marginTop/Side`) read at composite time; `g_gridSpacing` is also the grid-snap pitch. Margins are rose line segments (top/bottom share `margin_top`, left/right share `margin_side`) drawn right after the grid in both `compositeAllLayers` and the eraser-preview "below" snapshot, so they behave exactly like the grid (under the layers, exported). The status-bar chips toggle on tap and open a slider popup on long-press (`showPaperSliderPopup`).

### Text boxes (vector layers)

`TextBox` is the fifth vector shape (`ShapeKind::Text`, tag 5, `shapes.bin` magic `VEC2`; V0/V1 files still load). **Kotlin owns layout and rasterization** (`TextSupport.kt`: `FontRegistry`, `TextBoxModel`, `TextLayout` over `StaticLayout`); native owns the model, persistence, selection/undo/clipboard, and draws one textured quad per box from an R8 coverage texture cached by box id (`g_textTextures`, `kTextVS/FS`, `drawTextBoxes`). Colour is a uniform, so recolouring never re-rasterizes.

- **Raster upkeep loop**: the compositor raises `g_textRasterNeeded` when a box has no texture or its texture is stale (`textRasterStale`: view scale drifted past `kTextRasterBand`, or box width changed by > 1 px — the latter is what makes a handle-resize re-wrap live). `DrawingSurfaceView` polls `textRasterNeeded()` after every multi-buffer render and `TextEditController.rasterizeMissing` answers via `getTextRasterRequests` + `uploadTextRaster`. Thumbnail/export composites set `g_suppressTextRasterRequests` so their scale doesn't trigger churn. `g_textRasterScales` is the UI-thread-readable mirror of texture scale/width; never read `g_textTextures` off the GL thread.
- **Editing** is a real `EditText` overlaid on the SurfaceView (`TextEditController`), positioned/scaled/rotated onto the box; native hides that box via `g_textEditingId`. Commit = layout → raster upload → `updateTextBox(undoable)`. Both are queued and drained in `applyPendingShapes`, which applies text **adds before edits** so a fast commit finds its box. A new box is provisional: no undo entry until the first non-empty commit (then `VectorAdd`); an abandoned empty box vanishes without a trace.
- **Ids** are random 24-bit (`newTextBoxId`) so they survive the float round-trip in `getTextRasterRequests` and can't collide across lazily-loaded pages. Paste assigns a fresh id. Text crosses JNI as UTF-8 `ByteArray`, not `jstring` (modified UTF-8 mangles emoji).
- **Resize is width-only** (`applyScaleTo` Text case): the opposite corner stays fixed, `w` changes, height follows content after Kotlin re-lays-out. Font size never changes on resize.
- **Fonts** are bundled OFL TTFs (`res/font/family_*.xml`, licenses in `fonts_licenses/`). Boxes persist the registry *key*; unknown keys fall back to the first entry. Adding a font = TTFs + family XML + one `FontRegistry` entry.
- Every `switch (kind)` / four-list iteration in the file needed a text case; if you add another shape list, grep for `layer.texts` to find them all.
- **Style target**: the TEXT panel rows (`buildText*Row` in MainActivity) edit whichever box `TextEditController.currentStyleBox()` names — the box being edited (live overlay restyle, committed with the text), else the selected box (one undoable update + immediate re-raster), else only the defaults for the next box (persisted in prefs as `text_*`). The brush colour follows the same rule via `setColor`. The rows show for the TEXT tool, or for a SELECT tool with a text box selected; native reports a queued edit one frame late, so the panel refreshes again after ~80 ms.
- **Editing from SELECT**: first tap selects, a second tap on the already-selected box opens the editor (`selectDownTextId` in `handleSelectEvent`). A canvas tap while the editor is open only closes it (`onCanvasTouchDown` returns true and the gesture is swallowed).
- **System clipboard**: Copy/Cut of a text box mirrors plain text to the Android clipboard; Paste with an empty in-app clipboard turns clipboard text into a new auto-width box at the visible centre (`pasteText`).
- **Export dialog** (`showExportDialog` → `runExport`): one flow for PNG / PDF, current / all / range pages, PNG scale 1–3x and transparent background, guide hiding, and Save (SAF file picker for single files; a multi-page PNG set is written straight to `<Drafting Table>/exports/<doc>/` because Android's folder picker refuses the root and Download folders) or Share (staged in `cacheDir/exports`, served by the manifest's `FileProvider`, `res/xml/export_paths.xml`). Pages render **one at a time** (a 3x page is ~64 MB); PNG pages get a tiny probe render first so the page is loaded before its text is re-rastered at the export scale. Native flags: `g_exportTransparent`, `g_exportSkipGuides`, `g_exportSkipText`, all cleared in `finish`. Format / scale / background / guides persist in prefs; pages always reopen as "current".
- **PDF text**: PDF pages render with `g_exportSkipText` set, then Kotlin draws each page's boxes as real text onto the page canvas (`TextLayout.drawPageTextToCanvas`, geometry from `exportTransform` mirroring `renderPageThumbnail`'s letterbox) — selectable in viewers, but above every raster stroke on the page. PNG export first re-rasters the active page's boxes at the export scale (`prepareForPngExport`, texel cap 4096). Per-page box enumeration for any loaded page: `getPageTextBoxes` + `getPageTextBoxText/Font/Runs`.
- **Per-character formatting** (`TextRun`, `shapes.bin` magic `VEC3`; V2 files load with no runs): a box is base style + additive runs (bold / italic / underline / colour) over UTF-16 ranges. Kotlin `TextStyling` expands to per-character styles, mutates a range (the overlay selection, else the whole box), and `normalize`s back: base bold/italic = what every character has, base colour = the most common, runs = the exceptions. The overlay carries runs as StyleSpan / UnderlineSpan / ForegroundColorSpan and reads them back on commit (IME composing underlines are skipped via `SPAN_COMPOSING`). Font, size, align, and leading stay per box. A box with a colour run is rasterized as premultiplied RGBA (`uMode 1` in `kTextFS`, `channels = 4` on upload); all others stay R8 + tint. `panelFlags()` drives the bold/italic/underline rows: selection → all-of, caret → the character before it, selected box → all-of.

### renderer.cpp navigation

The file is single-TU and order-sensitive. When adding helpers, watch the forward-declaration block around line ~830 (`saveVectorLayer`, `loadVectorLayerShapes`, the tile-snapshot helpers, `applyUndo` / `applyRedo` / `applyPendingShapes`). The undo apply functions (`applyEntryReverse`/`Forward`/`applyUndo`/`applyRedo`) live near the bottom of the anonymous namespace so all their dependencies are defined.

Major sections, roughly:
- Tunables, shaders (kDabVS/FS, kCompVS/FS, kPreviewVS/FS, kGridVS/FS, kLineVS/FS, kFillVS/FS)
- POD types (Tile, Line, Rect, Ellipse, Circle, Layer, Selection, DragState, UndoEntry, TileSnap, ShapeData)
- Globals + `currentViewScale()`, `readPageClip()`, `vpxToDoc()`
- Pending action drain
- `ensureInited` (program linking + uniform lookup) — must add new uniforms here
- Persistence (tile load/save, vector layer save/load with V0→V1 migration)
- Tile management + snapshot helpers
- Bake (`bakeCurrentStrokeIntoTiles`)
- Compose (`compositeAllLayers`, `compositeRasterLayer`, `compositeVectorLayer`)
- Selection / OBB / hit-test / handles
- Snapping (`forEachShapeSnapTarget`, `findSnap`, `drawSnapMarker`)
- Shape rendering helpers (`drawLineSegment`, `drawRectangleAsLines`, `drawEllipseAsLines`)
- Undo/redo apply (`applyEntryReverse`, `applyEntryForward`, `applyUndo`, `applyRedo`)
- `extern "C"` JNI exports

### Things that are easy to break

- **Forward declarations**: this file has had several rounds of "undeclared identifier" build errors when helpers were added in the wrong order. If you call something not yet visible, add a forward decl in the block around line ~830.
- **FBO leaks**: any helper that binds a tile FBO via `glReadPixels` / `glReadPixels`-style work must save and restore `GL_DRAW_FRAMEBUFFER_BINDING` — otherwise calls inside `applyPendingLayerActions` (drained at the *start* of `compositeAllLayers`) will leave the multi-buffer FBO replaced, and the subsequent clear/grid/composite go to a tile. Symptom: canvas turns black after undo until the next stroke commit. `saveTileToDisk` and `uploadTileBytesAndSave` already do this — preserve it.
- **`shapes.bin` format**: tagged with `kShapesMagicV1`. The V0→V1 migration (rotation field added to Rect/Ellipse) is in `loadVectorLayerShapes`. Bumping the format requires adding a new magic and a migration path.
- **Premultiplied alpha**: don't accidentally output straight RGBA in a fragment shader and blend it premultiplied — the `*color * alpha, alpha` pattern is everywhere on purpose.
- **View-px constants**: anything that needs to feel constant on screen at any zoom is named `*ViewPx` and divided by `currentViewScale()` at use time. Don't hard-code doc-px sizes for UI affordances.
- **Bake performance at low zoom**: `bakeCurrentStrokeIntoTiles` skips bbox cells that no sample's brush radius touches, AND `DabEmitter::emit` skips dabs whose footprint is fully outside the per-tile clip bounds. Both optimizations are critical — without them, a long stroke at 0.25× zoom can allocate thousands of tile FBOs and lock up the GL thread.
- **Tool-rail height**: the rail lives in a `ScrollView`, but it is meant never to scroll — a scrolling rail hides tools and costs a gesture mid-sketch. The device gives it ~810dp of usable height; the current rail is ~685dp at a 40dp per-tile pitch (`kRailTileSizeDp` / `kRailTilePadDp` in `MainActivity`). Each new rail entry eats ~40dp of that headroom, so check it on-device (`adb exec-out screencap -p`) after adding one, and trim tile geometry rather than letting it scroll. Tiles use `CENTER_INSIDE` with 24dp icons, so shrinking a tile below a 24dp content box starts shrinking the glyph too.

## Memory

There is a `.claude/projects/.../memory/` directory with prior-session notes about user preferences and project context. Use it via the auto-memory protocol — check `MEMORY.md` for relevance before doing major work.

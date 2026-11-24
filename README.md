# Drafting Table
<img src="tools/drafting_table.png">
Ben's notes:

I wanted a drawing app for my [Wacom MovinkPad 11](https://estore.wacom.com/en-us/wacom-movinkpad-11-dtha116cl0z.html) tablet that combines features from conventional drawing programs, slide decks, and paper engineering notebooks.  This is a  attempt at creating that.  This is *not* intended to be a full-blown painting program: it's designed for making technical sketches.

"Documents" are organized into pages, with a sidebar for previewing and navigating between pages, like a slide deck.  Tapping on a page in the sidebar loads that page onto the canvas.  Each page has its own set of layers.  

In addition to raster brush tools, vector primitives (lines, rectangles, circles, ellipses) can be constructed on vector layers. I find these useful for creating construction geometry in technical sketches.  Vector objects can be transformed after creation, or rastered down to the raster layer below. Vector "snapping" can be toggled, so the start/end/midpoints of vector objects snap to points on other vector objects, or to nodes on the (toggleable) graph paper background.  Snapping can be toggled from the pen, so it's easy to snap one end of a vector but not the other, mid-creation.

Pen-to-glass latency is significantly less than other drawig apps I've measured (25-35ms, vs ~75ms for other programs)

I have only tested on the MovinkPad 11, and it will almost certainly not work out-of-the-box on any other tablet. But Claude can probably fix it for you :P

Beware:  100% of the code is written by Claude.  

Claude Code generated Readme below:

## Features

**Documents and pages**
- Multiple documents, each with multiple pages. A sidebar shows page thumbnails; pages can be added, reordered and deleted.
- Page size is set per document (device default, Letter, A4, hi-res, or custom) and can be changed later without losing anything.
- Graph-paper background (lines or dots, adjustable spacing) and margin guides, per document.
- Documents are saved to `/sdcard/Drafting Table/` as you draw, so they survive an uninstall. "Backup all documents" is in the menu.
- Only what's needed is loaded: the page you're on, and only its visible layers. Switching documents takes well under a second.

**Layers**
- Raster and vector layers, mixed freely, per page.
- Name, visibility, opacity; drag to reorder; merge down; delete.
- Rasterize a vector layer, or a single selected shape onto the layer below.

**Drawing**
- Brush with size, opacity, hardness and pressure response. "Uniform α" keeps overlapping passes within one stroke from building up.
- Eraser.
- Bucket fill, with adjustable bleed.
- Shade: draw an outline and the enclosed area fills as you go.
- Eyedropper.

**Ruler and compass**
- Ruler: a straight edge you place on the page. The pen can't cross it, so a stroke run along it comes out straight, with normal pressure response. Either long edge and both ends work.
- Compass: set a centre and radius, and strokes follow the circle.
- With a drawing tool active, drag the grip on the ruler to slide it. With snap on it steps by the grid spacing, for evenly spaced hatching.
- Both follow point snap and angle snap while you position them.

**Vector shapes and text**
- Line, rectangle, circle, ellipse. Move, scale and rotate them after drawing.
- Text boxes, with a choice of fonts, size, alignment, line spacing, and bold / italic / underline / colour per character.
- Snap to the grid and to other shapes. Angle snap with an adjustable step (long-press the angle chip).

**Selection**
- Rectangle or lasso. On a raster layer it lifts the pixels so you can move, scale and rotate them; on a vector layer it selects shapes.
- Cut, copy, paste, delete. "Keep aspect" option when scaling.
- Import an image onto a new layer.

**Colour**
- HSV picker with hex / RGB entry, a palette, recent colours and your own saved slots.

**Export**
- PNG or PDF; current page, all pages or a range.
- PNG at 1–3× with optional transparent background. PDF at a chosen dpi, with text boxes as real, selectable text.
- Save to a file or share to another app.

**Pen and touch**
- Pressure-sensitive EMR pen. Pen-to-glass latency is low, helped by motion prediction (can be turned off).
- Pen buttons: nearest the tip toggles snap; middle swaps brush / eraser (or toggles angle snap with vector tools); far one is undo.
- Two fingers pan, zoom and rotate the canvas. "Stylus only" mode ignores fingers and palm for drawing.
- Undo / redo: 50 steps.

**Status bar**
- Tap to toggle grid, margins, pixel grid, snap, angle snap, brush preview and prediction. Long-press grid, margins or angle for their settings.
- The `load:` chip shows how long the last document took to open; tap it for a breakdown.

## Build and install

**With Android Studio** (the usual way)
1. Open this folder in Android Studio and let it sync. Install any SDK / NDK / CMake versions it asks for.
2. Connect the tablet over USB and accept the debugging prompt on it.
3. Press **Run**.

**From the command line**
```powershell
.\gradlew.bat installDebug
adb shell am start -n com.bk.drawing/.MainActivity
adb logcat -s DrawingApp
```
If `gradlew.bat` is missing, open the project in Android Studio once, or run `gradle wrapper --gradle-version 8.7`.

**Without a computer**
Every push to `master` (or a `claude/**` branch) builds an APK on GitHub and publishes it to the [`ci-latest` release](../../releases/tag/ci-latest). Open that page on the tablet, download the APK and tap it. It installs as a separate app, "Drafting Table CI", next to the regular one. Both use the same documents folder, so don't run them at the same time.

Build facts: arm64 only, Android 13 or newer (min SDK 33), C++20, OpenGL ES 3.2.

## How the code is organised

Kotlin handles the app shell, panels and touch input. C++ does all the drawing, saving and undo.

```
app/src/main/
├── java/com/bk/drawing/
│   ├── MainActivity.kt          panels, menus, dialogs, export
│   ├── DrawingSurfaceView.kt    pen and touch input, pan / zoom
│   ├── Guides.kt                ruler and compass
│   ├── TextEditController.kt    text box editing
│   ├── TextSupport.kt           fonts and text layout
│   ├── ColorPicker*.kt          colour picker
│   └── NativeRenderer.kt        bridge to the C++ side
└── cpp/renderer.cpp             the renderer
```

`CLAUDE.md` has the full architecture notes: coordinate spaces, threading, file formats, and the things that are easy to break. `NOTES.md` has the original design intent.

## Testing and debugging

Most testing is done by using the app on the tablet. There are also:

- **Instrumented tests** (`app/src/androidTest/`): check that redrawing a stroke gives bit-identical pixels, which undo / redo depends on. Run with `.\gradlew.bat connectedDebugAndroidTest`.
- **Trace capture**: `.\tools\capture-trace.ps1` records 10 seconds for [ui.perfetto.dev](https://ui.perfetto.dev), for chasing latency or dropped frames. See `tools/README.md`.
- **Logs**: `adb logcat -s DrawingApp`.

// Tile-based document renderer with on-disk persistence and layers.
//
// Document model:
//   Document
//     └── Layer[]                  (z-ordered, bottom-up)
//           └── tile grid          (sparse: unordered_map<(tx, ty), Tile>)
//
//   - Document is an infinite 2D plane in floating-point coordinates
//     (origin top-left, y-down — matches MotionEvent).
//   - Each layer holds its own sparse 256x256 RGBA tile grid; tiles are
//     allocated lazily when a stroke first touches them and cleared to
//     fully-transparent.
//   - The multi-buffer is cleared to paper-white before tiles composite,
//     so the bottom-most "paper" is implicit (no dedicated layer for it).
//
// Premultiplied alpha:
//   Tile contents are stored premultiplied. Dab fragment shader outputs
//   vec4(rgb*α, α). Both intra-tile baking and tile compositing use
//   glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA), which is the correct
//   blend mode for premultiplied alpha.
//
// Persistence:
//   <docDir>/layer_<idx>/tile_<tx>_<ty>.bin holds raw 256x256x4 RGBA bytes
//   per tile (premultiplied). Single-layer documents from before the layer
//   refactor are migrated into layer_0/ on first load.
//
// Stroke lifecycle:
//   beginStroke        - reset emitter, snapshot active layer as stroke target
//   extendStrokeBatch  - append a batch of samples (real + optional predicted
//                        tail), emit dabs additively into the front-buffered
//                        layer; one preview-overlay render per batch
//   commitStroke       - bake the stroke into the snapshotted target layer's
//                        tiles, save those tiles to disk, drop samples
//   renderDocument   - clear the multi-buffer to white, composite every
//                      tile of every layer in z-order
//
// Cross-thread layer ops:
//   addLayer / cycleActiveLayer can be called from the UI thread; they
//   enqueue an action under a mutex and the GL thread drains the queue at
//   the start of each operation. This avoids racy access to layers().
//
// Coordinate spaces:
//   doc-px:    document coordinates. The natural unit for shapes, tiles,
//              snap targets, and stroke samples. Persistent.
//   view-px:   on-screen pixels in the SurfaceView (MotionEvent coords).
//   buffer-px: pixels of the GL buffer (may be rotated relative to view
//              by the framework on certain device orientations).
//   The `transform` mat4 passed into render JNIs is doc→buffer (Kotlin
//   composes its doc→view view-transform with the framework's view→buffer
//   matrix). Native consumes only this composed form, so 2-finger pan/
//   zoom/rotate doesn't require any native-side coordinate plumbing.
//   View scale is mirrored into g_viewScaleBits via setViewScale so snap
//   radii and selection-handle sizes stay constant in screen-space.

#include <jni.h>
#include <GLES3/gl3.h>
#include <android/bitmap.h>
#include <android/log.h>
#include <android/trace.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <dirent.h>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#define LOG_TAG "DrawingApp"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// RAII wrapper for ATrace sections. Sections nest naturally with C++
// scopes, and the destructor closes them so early returns don't leak
// an unmatched begin. ATrace events show up alongside the rest of the
// system in perfetto / systrace captures, time-correlated with vsync,
// SurfaceFlinger transactions, scheduler events, etc. Cost when
// tracing is OFF: one branch (ATrace_isEnabled is cheap).
struct ATraceScope {
    explicit ATraceScope(const char* name) {
        if (ATrace_isEnabled()) ATrace_beginSection(name);
    }
    ~ATraceScope() {
        if (ATrace_isEnabled()) ATrace_endSection();
    }
};
#define ATRACE_CONCAT_(a, b) a##b
#define ATRACE_CONCAT(a, b)  ATRACE_CONCAT_(a, b)
#define ATRACE_SCOPE(name) ATraceScope ATRACE_CONCAT(_atrace_scope_, __LINE__)(name)

namespace {

// ---- Tunables -------------------------------------------------------------

constexpr int    kTileSize  = 256;
constexpr float  kTileSizeF = 256.0f;
constexpr float  kTileHalfF = 128.0f;
constexpr size_t kTileBytes = static_cast<size_t>(kTileSize) * kTileSize * 4;
// Apron padding around each tile texture. Storing a 1-pixel border
// of the neighbor's edge content lets the composite shader's LINEAR
// filter blend across tile boundaries without producing a seam (the
// alternative — sampling at the edge with CLAMP_TO_EDGE — repeats the
// edge texel and misses the neighboring tile's content entirely).
constexpr int   kApron       = 1;
constexpr int   kTileTexSize = kTileSize + 2 * kApron;

constexpr float kSpacing    = 0.18f;
// Pressure → dab-radius range. kMinRadius = 0 lets the lightest pen
// contact taper to nothing, so the brush has dynamic range across the
// pen's full pressure curve. The bake/emit path floors spacing
// independently below (kMinSpacing) so a near-zero radius can't
// degenerate the dab loop.
constexpr float kMinRadius  = 0.0f;
constexpr float kMaxRadius  = 18.0f;
// Dab spacing floor in doc-px. Ordinary spacing is kSpacing * radius;
// at very low pressure that product approaches zero, which would emit
// thousands of redundant sub-pixel dabs and (at exactly zero) infinite
// loop the extend(). 0.5 doc-px is below human-perceptible resolution
// at all sane zooms.
constexpr float kMinSpacing = 0.5f;
// Dabs whose radius is below this are skipped — they'd contribute no
// visible coverage at any zoom and the GL rasterizer has nothing to do.
constexpr float kMinDabRadius = 0.05f;

// Brush dabs render with a runtime-settable color and alpha. The
// dab's per-fragment alpha (g_strokeBrushAlpha * radial falloff)
// controls stroke buildup as overlapping dabs accumulate; the *color*
// is g_strokeBrushColor (captured from g_currentBrushColor +
// g_brushAlphaBits at beginStroke).

// During an erase stroke, the front-buffered live preview shows a pink
// indicator (blended additively, like a brush) so the user can see where
// they're about to erase. The actual erase happens at commit, in the
// tile bake path, with a different blend mode.
constexpr float kEraserPreviewColor[4] = { 1.00f, 0.40f, 0.45f, 0.55f };

constexpr float kIdentity[16] = {
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
};

// ---- Shaders --------------------------------------------------------------

const char* kDabVS = R"(#version 300 es
layout(location = 0) in vec2 aQuad;
out vec2 vUv;
out vec2 vDocPos;
uniform mat4  uTransform;
uniform vec2  uScreen;
uniform vec2  uCenter;
uniform float uRadius;
void main() {
    vUv = aQuad;
    // docPos is in the SAME frame as uCenter (tile-local during bake,
    // doc-pixels during live preview); the page-clip rect uniforms must
    // be set in that same frame.
    vec2 docPos = uCenter + aQuad * uRadius;
    vDocPos = docPos;
    vec4 bufPx = uTransform * vec4(docPos, 0.0, 1.0);
    vec2 ndc = (bufPx.xy / uScreen) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
)";

const char* kDabFS = R"(#version 300 es
// highp here, not mediump: the dab is rendered separately into each
// tile it touches, and a stroke that crosses a tile boundary expects
// the same alpha at fragments on either side of the seam. With
// mediump (10-bit mantissa) the smoothstep evaluation differs by ~1
// ULP between adjacent tiles' rasterizations, and LINEAR composite
// sampling surfaces that as a faint visible seam at the boundary.
// highp gives us enough precision for the two sides to round to the
// same alpha value.
precision highp float;
in vec2 vUv;
in vec2 vDocPos;
out vec4 outColor;
uniform vec4 uColor;
uniform vec2 uPageMin;
uniform vec2 uPageMax;
uniform int  uPageActive;     // 0 = no page clip
// Hardness in [0, 1]: the radial fraction at which the dab becomes
// fully opaque. The dab is opaque for r ≤ uHardness and falls off via
// smoothstep(uHardness, 1.0, r) outside that. uHardness=1 → solid
// disc (hard); uHardness=0 → full gradient (smooth).
uniform float uHardness;
void main() {
    if (uPageActive != 0 && (
        vDocPos.x < uPageMin.x || vDocPos.x > uPageMax.x ||
        vDocPos.y < uPageMin.y || vDocPos.y > uPageMax.y)) {
        discard;
    }
    float r = length(vUv);
    if (r >= 1.0) discard;
    float a;
    if (uHardness >= 1.0 || r <= uHardness) {
        // Inside the opaque core (and the whole disc when fully hard).
        a = 1.0;
    } else {
        // Outer falloff: rescale (uHardness..1) to (0..1) and apply a
        // 4th-power curve. Steeper than smoothstep — necessary because
        // dabs overlap ~6× at the spine; with smoothstep, accumulation
        // saturates to opaque almost everywhere and only a thin rim
        // looks "soft" no matter how low the user sets the slider. The
        // quartic drops fast enough that cumulative alpha tapers
        // through a wide visible gradient when stacked.
        float t = (1.0 - r) / (1.0 - uHardness);
        float t2 = t * t;
        a = t2 * t2;
    }
    float alpha = uColor.a * a;
    // Premultiplied output, blended with GL_ONE / GL_ONE_MINUS_SRC_ALPHA.
    outColor = vec4(uColor.rgb * alpha, alpha);
}
)";

const char* kCompVS = R"(#version 300 es
layout(location = 0) in vec2 aQuad;
out vec2 vUv;
uniform mat4  uTransform;
uniform vec2  uScreen;
uniform vec2  uTileCenter;
uniform float uTileHalf;
void main() {
    vec2 viewPx = uTileCenter + aQuad * uTileHalf;
    vec4 bufPx  = uTransform * vec4(viewPx, 0.0, 1.0);
    vec2 ndc = (bufPx.xy / uScreen) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
    vUv = aQuad * 0.5 + 0.5;
}
)";

const char* kCompFS = R"(#version 300 es
precision mediump float;
in vec2 vUv;
out vec4 outColor;
uniform sampler2D uTileTex;
// Per-layer opacity (1.0 = fully opaque). Multiplying a premultiplied
// RGBA by a scalar yields a correctly-premultiplied RGBA at the
// scaled alpha — that's why we don't need separate rgb / a math.
uniform float uOpacity;
// Apron-aware UV remap. The tile texture is 258x258 with the visible
// 256x256 content at texels [1..256]; the outer 1-pixel ring is filled
// from neighbors so LINEAR sampling near the edge can blend across
// tile boundaries cleanly (no seams). We map the visible quad UV
// [0..1] to texture UV [1/258, 257/258].
const float kApronTex = 1.0 / 258.0;
const float kInner    = 256.0 / 258.0;
void main() {
    vec2 uv = vUv * kInner + kApronTex;
    outColor = texture(uTileTex, uv) * uOpacity;
}
)";

// Unified live-preview composite for both brush and eraser. Output is
// premultiplied with alpha = c (accumulated coverage), so the framework's
// premultiplied composite over the multi-buffer yields:
//     result = fullColor * c + multi * (1 - c)
//            = lerp(multi, fullColor, c)
// where fullColor is "the displayed pixel value if the active layer were
// fully painted (brush) / fully erased (eraser) at this point". By the
// linearity of premultiplied compositing in any single layer's
// contribution, this lerp equals the true WYSIWYG display target for any
// 0 ≤ c ≤ 1. Both tools therefore correctly render strokes UNDER any
// layers above active, instead of overlaying them on top of multi.
//
//   eraser: fullColor = above + below * (1 - above.a)   // active gone
//   brush:  fullColor = above + brushRgb * (1 - above.a) // active fully painted
const char* kPreviewVS = R"(#version 300 es
layout(location = 0) in vec2 aQuad;
out vec2 vUv;
void main() {
    gl_Position = vec4(aQuad, 0.0, 1.0);
    vUv = aQuad * 0.5 + 0.5;
}
)";

const char* kPreviewFS = R"(#version 300 es
precision mediump float;
in vec2 vUv;
uniform sampler2D uBelow;     // paper + layers strictly below active; alpha=1
uniform sampler2D uAbove;     // layers strictly above active, premultiplied
uniform sampler2D uCoverage;  // accumulated coverage in alpha channel
uniform int       uMode;      // 0 = brush, 1 = eraser
uniform vec3      uBrushRgb;  // un-premultiplied brush color (ignored for eraser)
out vec4 outColor;
void main() {
    vec4 above = texture(uAbove, vUv);
    float c    = texture(uCoverage, vUv).a;

    vec3 fullColor;
    if (uMode == 1) {
        vec4 below = texture(uBelow, vUv);
        fullColor = above.rgb + below.rgb * (1.0 - above.a);
    } else {
        fullColor = above.rgb + uBrushRgb * (1.0 - above.a);
    }
    outColor = vec4(fullColor * c, c);
}
)";

// "Uniform-stroke-alpha" stamp shader. Used by the bake when the active
// stroke is in uniform-alpha mode: dabs build a per-tile coverage texture
// via MAX blending, then this shader stamps the brush color (or the
// eraser cut-out) onto the tile in a single OVER pass. That removes the
// dab-on-dab accumulation that produces the dark spots wherever a
// low-alpha stroke crosses itself.
const char* kStampVS = R"(#version 300 es
layout(location = 0) in vec2 aQuad;
out vec2 vUv;
void main() {
    gl_Position = vec4(aQuad, 0.0, 1.0);
    vUv = aQuad * 0.5 + 0.5;
}
)";

const char* kStampFS = R"(#version 300 es
precision mediump float;
in vec2 vUv;
uniform sampler2D uCoverage;
uniform vec4      uColor;       // straight rgb + ignored alpha
uniform int       uMode;        // 0 = brush (OVER), 1 = eraser (DESTOUT)
out vec4 outColor;
void main() {
    float c = texture(uCoverage, vUv).a;
    if (c <= 0.0) discard;
    if (uMode == 1) {
        // Eraser: caller binds glBlendFunc(GL_ZERO, GL_ONE_MINUS_SRC_ALPHA),
        // so output alpha c scales dest by (1 - c). RGB unused.
        outColor = vec4(0.0, 0.0, 0.0, c);
    } else {
        // Brush: premultiplied (color * c, c), blended OVER.
        outColor = vec4(uColor.rgb * c, c);
    }
}
)";

// Vector-layer line. Vertex shader morphs the unit quad into a
// capsule-shaped quad oriented along the line, with an extra halfWidth
// of margin past each endpoint so the rounded caps fit. Fragment shader
// computes a capsule SDF (signed distance to the segment with rounded
// ends) and anti-aliases over a 1-pixel band at the edge.
const char* kLineVS = R"(#version 300 es
layout(location = 0) in vec2 aQuad;
out vec2 vLineSpace;       // (along, across) from line midpoint, in doc-pixels
out vec2 vDocPos;          // for page-bounds discard in FS (doc-pixels)
uniform mat4  uTransform;
uniform vec2  uScreen;
uniform vec2  uP0;
uniform vec2  uP1;
uniform float uHalfWidth;
void main() {
    vec2 dir = uP1 - uP0;
    float len = length(dir);
    vec2 ndir = (len > 1e-6) ? dir / len : vec2(1.0, 0.0);
    vec2 nrm  = vec2(-ndir.y, ndir.x);
    vec2 mid  = (uP0 + uP1) * 0.5;

    float halfLen   = len * 0.5;
    float along     = aQuad.x * (halfLen + uHalfWidth);
    float across    = aQuad.y * uHalfWidth;
    vec2  docPos    = mid + ndir * along + nrm * across;
    vDocPos = docPos;

    vec4 bufPx = uTransform * vec4(docPos, 0.0, 1.0);
    vec2 ndc   = (bufPx.xy / uScreen) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);

    vLineSpace = vec2(along, across);
}
)";

const char* kLineFS = R"(#version 300 es
// highp throughout — uP0/uP1 carry full-buffer-magnitude pixel
// coordinates (up to ~2200), and at that magnitude the default mediump
// (10-bit mantissa) has ~1.7 unit precision. length(uP1 - uP0) for
// short lines then loses most of its accuracy, which warps the SDF
// enough that alpha can't ramp correctly.
precision highp float;
in vec2 vLineSpace;
in vec2 vDocPos;
uniform vec2  uP0;
uniform vec2  uP1;
uniform float uHalfWidth;
uniform vec4  uColor;        // straight RGBA (premultiplied at output)
uniform vec2  uPageMin;
uniform vec2  uPageMax;
uniform int   uPageActive;
// Per-layer opacity for vector lines. Defaulted to 1.0 by every
// non-layer caller (handles, snap markers, page outline, shape
// preview); only the layer compositor (drawLineSegment etc. inside
// compositeVectorLayer) sets it to a layer-specific value.
uniform float uOpacity;
out vec4 outColor;
void main() {
    if (uPageActive != 0 && (
        vDocPos.x < uPageMin.x || vDocPos.x > uPageMax.x ||
        vDocPos.y < uPageMin.y || vDocPos.y > uPageMax.y)) {
        discard;
    }
    float halfLen = length(uP1 - uP0) * 0.5;
    float dAlong  = max(abs(vLineSpace.x) - halfLen, 0.0);
    float dist    = sqrt(dAlong * dAlong + vLineSpace.y * vLineSpace.y);
    float alpha   = 1.0 - smoothstep(uHalfWidth - 0.5, uHalfWidth + 0.5, dist);
    if (alpha <= 0.0) discard;
    float a = uColor.a * alpha * uOpacity;
    outColor = vec4(uColor.rgb * a, a);
}
)";

// Vector-layer ellipse/circle outline drawn as a single SDF quad. Replaces
// the N-segment polyline approach (drawEllipseAsLines), which left dark
// junction artifacts at sub-1 alpha — each segment's rounded cap overlaps
// the next segment's cap, and overlapping caps double-cover the same
// pixels. Single-primitive SDF avoids the overlap entirely so the stroke
// renders at exactly the requested alpha everywhere.
const char* kEllipseVS = R"(#version 300 es
layout(location = 0) in vec2 aQuad;
out vec2 vLocal;       // ellipse-local (axis-aligned), doc-pixels from center
out vec2 vDocPos;      // for page-bounds discard
uniform mat4  uTransform;
uniform vec2  uScreen;
uniform vec2  uCenter;
uniform vec2  uRadii;       // (rx, ry)
uniform float uRotation;
uniform float uHalfWidth;
void main() {
    // Enclose the outer edge of the stroke (rx+halfW, ry+halfW) with a
    // 1-pixel slack for the AA ramp.
    vec2 outer = uRadii + vec2(uHalfWidth + 1.0);
    vec2 local = aQuad * outer;
    vLocal = local;

    float c = cos(uRotation), s = sin(uRotation);
    vec2 docPos = uCenter + vec2(local.x * c - local.y * s,
                                 local.x * s + local.y * c);
    vDocPos = docPos;

    vec4 bufPx = uTransform * vec4(docPos, 0.0, 1.0);
    vec2 ndc   = (bufPx.xy / uScreen) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
)";

const char* kEllipseFS = R"(#version 300 es
precision highp float;
in vec2 vLocal;
in vec2 vDocPos;
uniform vec2  uRadii;
uniform float uHalfWidth;
uniform vec4  uColor;     // straight RGBA (premultiplied at output)
uniform float uOpacity;
uniform vec2  uPageMin;
uniform vec2  uPageMax;
uniform int   uPageActive;
out vec4 outColor;

// Approximate signed distance to the ellipse outline. For rx == ry this
// reduces to length(p) - rx (exact circle). For unequal radii the relative
// error grows with eccentricity but stays well under a pixel at the radii
// / stroke widths used here. Sign is positive outside, negative inside.
float ellipseDistApprox(vec2 p, vec2 ab) {
    if (ab.x < 1e-4 || ab.y < 1e-4) return 1e9;
    float k1 = length(p / ab);
    float k2 = length(p / (ab * ab));
    if (k2 < 1e-8) return -min(ab.x, ab.y);
    return k1 * (k1 - 1.0) / k2;
}

void main() {
    if (uPageActive != 0 && (
        vDocPos.x < uPageMin.x || vDocPos.x > uPageMax.x ||
        vDocPos.y < uPageMin.y || vDocPos.y > uPageMax.y)) {
        discard;
    }
    float d  = ellipseDistApprox(vLocal, uRadii);
    float ad = abs(d);
    float alpha = 1.0 - smoothstep(uHalfWidth - 0.5, uHalfWidth + 0.5, ad);
    if (alpha <= 0.0) discard;
    float a = uColor.a * alpha * uOpacity;
    outColor = vec4(uColor.rgb * a, a);
}
)";

// Vector-layer rectangle outline drawn as a single SDF quad. Same
// motivation as the ellipse program: drawing the outline as 4 line
// segments leaves overlapping caps at the corners, which double-cover and
// render darker than requested whenever alpha < 1.
const char* kRectVS = R"(#version 300 es
layout(location = 0) in vec2 aQuad;
out vec2 vLocal;       // rect-local (axis-aligned), doc-pixels from center
out vec2 vDocPos;
uniform mat4  uTransform;
uniform vec2  uScreen;
uniform vec2  uCenter;
uniform vec2  uHalfExt;       // (hw, hh)
uniform float uRotation;
uniform float uHalfWidth;
void main() {
    vec2 outer = uHalfExt + vec2(uHalfWidth + 1.0);
    vec2 local = aQuad * outer;
    vLocal = local;

    float c = cos(uRotation), s = sin(uRotation);
    vec2 docPos = uCenter + vec2(local.x * c - local.y * s,
                                 local.x * s + local.y * c);
    vDocPos = docPos;

    vec4 bufPx = uTransform * vec4(docPos, 0.0, 1.0);
    vec2 ndc   = (bufPx.xy / uScreen) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
)";

const char* kRectFS = R"(#version 300 es
precision highp float;
in vec2 vLocal;
in vec2 vDocPos;
uniform vec2  uHalfExt;
uniform float uHalfWidth;
uniform vec4  uColor;
uniform float uOpacity;
uniform vec2  uPageMin;
uniform vec2  uPageMax;
uniform int   uPageActive;
out vec4 outColor;

void main() {
    if (uPageActive != 0 && (
        vDocPos.x < uPageMin.x || vDocPos.x > uPageMax.x ||
        vDocPos.y < uPageMin.y || vDocPos.y > uPageMax.y)) {
        discard;
    }
    // Standard axis-aligned box SDF (positive outside, negative inside).
    vec2 q = abs(vLocal) - uHalfExt;
    float d = length(max(q, vec2(0.0))) + min(max(q.x, q.y), 0.0);
    float ad = abs(d);
    float alpha = 1.0 - smoothstep(uHalfWidth - 0.5, uHalfWidth + 0.5, ad);
    if (alpha <= 0.0) discard;
    float a = uColor.a * alpha * uOpacity;
    outColor = vec4(uColor.rgb * a, a);
}
)";

// Page-background grid. A fullscreen NDC quad whose fragment shader
// computes each pixel's document position via the inverse of the
// framework's transform matrix, then evaluates a periodic line/dot
// pattern. Output is premultiplied so it composites cleanly onto the
// paper-white-cleared multi-buffer ahead of the layer tiles.
const char* kGridVS = R"(#version 300 es
layout(location = 0) in vec2 aQuad;
void main() {
    gl_Position = vec4(aQuad, 0.0, 1.0);
}
)";

const char* kGridFS = R"(#version 300 es
precision mediump float;
uniform mat4  uInverseTransform;   // buffer-pixel -> document-pixel
uniform float uSpacing;            // minor grid spacing in document pixels
uniform float uSubdivisions;       // major lines every Nth minor
uniform float uMinorWidth;         // half-width of minor line/dot in doc pixels
uniform float uMajorWidth;
uniform vec4  uMinorColor;         // straight RGBA
uniform vec4  uMajorColor;
uniform int   uStyle;              // 1 = lines, 2 = dots
uniform vec2  uPageMin;
uniform vec2  uPageMax;
uniform int   uPageActive;
out vec4 outColor;

void main() {
    vec2 docPos = (uInverseTransform * vec4(gl_FragCoord.xy, 0.0, 1.0)).xy;
    if (uPageActive != 0 && (
        docPos.x < uPageMin.x || docPos.x > uPageMax.x ||
        docPos.y < uPageMin.y || docPos.y > uPageMax.y)) {
        discard;
    }

    float sMin = uSpacing;
    float sMaj = uSpacing * uSubdivisions;

    // Distance to the nearest grid line per axis, in document pixels.
    vec2 dMin = abs(mod(docPos + sMin * 0.5, sMin) - sMin * 0.5);
    vec2 dMaj = abs(mod(docPos + sMaj * 0.5, sMaj) - sMaj * 0.5);

    float aMinor = 0.0;
    float aMajor = 0.0;

    if (uStyle == 1) {
        // Lines — close to ANY axis lights up.
        aMinor = max(
            1.0 - smoothstep(uMinorWidth, uMinorWidth + 1.0, dMin.x),
            1.0 - smoothstep(uMinorWidth, uMinorWidth + 1.0, dMin.y));
        aMajor = max(
            1.0 - smoothstep(uMajorWidth, uMajorWidth + 1.0, dMaj.x),
            1.0 - smoothstep(uMajorWidth, uMajorWidth + 1.0, dMaj.y));
    } else if (uStyle == 2) {
        // Dots — only at intersections (both axes close).
        aMinor = 1.0 - smoothstep(uMinorWidth, uMinorWidth + 1.0, length(dMin));
        aMajor = 1.0 - smoothstep(uMajorWidth, uMajorWidth + 1.0, length(dMaj));
    }

    vec4 col = vec4(0.0);
    if (aMajor >= aMinor && aMajor > 0.001) {
        col = vec4(uMajorColor.rgb, uMajorColor.a * aMajor);
    } else if (aMinor > 0.001) {
        col = vec4(uMinorColor.rgb, uMinorColor.a * aMinor);
    }

    // Premultiplied output for blending against the paper-white clear.
    outColor = vec4(col.rgb * col.a, col.a);
}
)";

// Per-pixel grid overlay. Drawn on top of the layers when the user
// toggles it on AND the view scale is high enough that one doc-pixel
// covers several buffer-pixels (otherwise the grid is denser than the
// resolution and just fills the screen as a solid wash). Uses fwidth
// to keep the line at ~1 buffer-pixel wide regardless of zoom — the
// shared grid shader's 1-doc-pixel AA band would blur to many buffer-
// pixels at the zoom levels this feature targets.
const char* kPixelGridVS = R"(#version 300 es
layout(location = 0) in vec2 aQuad;
void main() {
    gl_Position = vec4(aQuad, 0.0, 1.0);
}
)";

const char* kPixelGridFS = R"(#version 300 es
precision highp float;
uniform mat4  uInverseTransform;
uniform vec4  uColor;        // straight RGBA
uniform vec2  uPageMin;
uniform vec2  uPageMax;
uniform int   uPageActive;
out vec4 outColor;
void main() {
    vec2 docPos = (uInverseTransform * vec4(gl_FragCoord.xy, 0.0, 1.0)).xy;
    if (uPageActive != 0 && (
        docPos.x < uPageMin.x || docPos.x > uPageMax.x ||
        docPos.y < uPageMin.y || docPos.y > uPageMax.y)) {
        discard;
    }
    // Distance from the nearest integer doc-px boundary, per axis.
    vec2 dEdge = 0.5 - abs(fract(docPos) - 0.5);
    // Scale into buffer-pixel space. fwidth gives 1 buffer-pixel ≈
    // (zoom-recip) doc-px, so dividing rescales the AA band to ~1
    // buffer-pixel regardless of zoom.
    vec2 bufD = dEdge / max(fwidth(docPos), vec2(1e-6));
    // Fade over a single buffer-pixel centered on the boundary — keeps
    // the line at sub-pixel thickness where the screen permits.
    float a = 1.0 - smoothstep(0.0, 1.0, min(bufD.x, bufD.y));
    if (a <= 0.0) discard;
    float alpha = uColor.a * a;
    outColor = vec4(uColor.rgb * alpha, alpha);
}
)";

// Solid-color fill of an axis-aligned doc-coord rectangle. Used to paint
// the page-area paper-white over the gray (off-canvas) clear.
const char* kFillVS = R"(#version 300 es
layout(location = 0) in vec2 aQuad;
uniform mat4 uTransform;
uniform vec2 uScreen;
uniform vec2 uMin;
uniform vec2 uMax;
void main() {
    vec2 t = aQuad * 0.5 + 0.5;
    vec2 docPos = mix(uMin, uMax, t);
    vec4 bufPx  = uTransform * vec4(docPos, 0.0, 1.0);
    vec2 ndc = (bufPx.xy / uScreen) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
)";

const char* kFillFS = R"(#version 300 es
precision mediump float;
uniform vec4 uFillColor;       // straight RGBA, opaque expected
out vec4 outColor;
void main() {
    outColor = uFillColor;
}
)";

// Closed-path (shade tool) fill. Unlike every other program here the
// vertex data is a real buffer of positions rather than the shared unit
// quad: pass 1 draws the sample path as a triangle fan to accumulate a
// winding number in the stencil buffer, pass 2 draws the path's AABB
// with a stencil test to convert "winding != 0" into coverage alpha.
// Positions arrive in doc-pixels; uTransform maps them into the target
// (buffer-px for the live preview, tile-local for the bake), so
// vDocPos — and therefore the page clip — is always doc-space.
const char* kPolyVS = R"(#version 300 es
layout(location = 0) in vec2 aPos;
out vec2 vDocPos;
uniform mat4 uTransform;
uniform vec2 uScreen;
void main() {
    vDocPos = aPos;
    vec4 bufPx = uTransform * vec4(aPos, 0.0, 1.0);
    vec2 ndc = (bufPx.xy / uScreen) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
)";

const char* kPolyFS = R"(#version 300 es
precision highp float;
in vec2 vDocPos;
uniform vec4 uColor;          // straight RGBA; premultiplied at output
uniform vec2 uPageMin;
uniform vec2 uPageMax;
uniform int  uPageActive;
// Chord edge. The closing chord is the one stretch of the fill's
// boundary that no dab covers, so left alone it reads as a hard polygon
// edge butted against the brush's soft one — and the stroke's own dabs
// at the two endpoints spill past it as nubs on an otherwise straight
// edge. Both are fixed by treating the chord as if a stroke ran along
// it: push the shape's edge out to uChordOffset — far enough that the
// bridge is tangent to the outermost dab rather than letting it poke
// through, and constant along the chord so the bridge stays straight —
// then fade to zero across uFeatherBand on kDabFS's quartic.
//
// Both are sized CPU-side, and neither is read off the chord's own
// endpoint samples: pressure at pen-down / pen-lift is near zero and
// kMinRadius is 0, so endpoint-derived widths collapse to nothing (the
// feather silently vanishes) or sit inside the largest nearby dab (it
// protrudes). See chordOutwardOffset / chordFeatherBand.
uniform vec2  uChord0;        // chord start = last path point, doc-px
uniform vec2  uChord1;        // chord end   = first path point, doc-px
uniform float uChordOffset;   // outward offset of the edge, doc-px
uniform float uFeatherBand;   // soft-edge width, doc-px
// 0 = ignore the chord (stencil pass), 1 = polygon interior,
// 2 = the band the chord's offset adds outside the polygon.
uniform int   uChordMode;
out vec4 outColor;
void main() {
    if (uPageActive != 0 && (
        vDocPos.x < uPageMin.x || vDocPos.x > uPageMax.x ||
        vDocPos.y < uPageMin.y || vDocPos.y > uPageMax.y)) {
        discard;
    }
    float alpha = uColor.a;
    if (uChordMode != 0) {
        // Distance to the chord SEGMENT, not its infinite line — a
        // concave region can wrap around past the chord's ends, and
        // those parts are nowhere near the chord and must not fade.
        // Unsigned distance is what keeps that safe: a far-off interior
        // fragment lands deep in the solid range either way, so the
        // fade can never reach across the shape and gouge it.
        vec2  seg  = uChord1 - uChord0;
        float len2 = dot(seg, seg);
        float t    = (len2 > 1e-8)
                   ? clamp(dot(vDocPos - uChord0, seg) / len2, 0.0, 1.0)
                   : 0.0;
        float d    = distance(vDocPos, uChord0 + seg * t);
        // The shape's edge now sits uChordOffset outside the chord, so
        // moving into the polygon adds to the depth below that edge
        // while moving out into the offset band subtracts from it. Both
        // modes agree at d = 0, which keeps the two passes seamless.
        float depth = (uChordMode == 1) ? (uChordOffset + d)
                                        : (uChordOffset - d);
        // max() guards the hard-brush case (band 0) against a divide by
        // zero; 1e-3 doc-px reads as the same crisp step its dabs give.
        float f  = clamp(depth / max(uFeatherBand, 1e-3), 0.0, 1.0);
        float f2 = f * f;
        alpha *= f2 * f2;
    }
    if (alpha <= 0.0) discard;
    outColor = vec4(uColor.rgb * alpha, alpha);
}
)";

// Render a sampled, premultiplied texture to an axis-aligned doc-coord
// rectangle (uMin..uMax). Used to draw the floating raster selection.
// Translation lives in (uMin, uMax) — the caller adjusts these by the
// pan offset, so this shader doesn't need to know about transforms
// beyond the standard doc→buffer matrix.
//
// Page-clip via FS discard: lets the caller render at the unclamped
// drop bounds (preserving aspect ratio) while clipping the off-page
// portion at the fragment level. Caller passes uPageMin/Max in the
// SAME coord frame as uMin/uMax (doc-coords for the overlay path,
// tile-local for the commit-into-tile path).
const char* kSelVS = R"(#version 300 es
layout(location = 0) in vec2 aQuad;
out vec2 vUv;
out vec2 vDocPos;
uniform mat4 uTransform;
uniform vec2 uScreen;
// Four placement corners — one per quad UV corner. Layout:
//   uC0 = UV (0,0) = top-left,   uC1 = UV (1,0) = top-right,
//   uC2 = UV (1,1) = bottom-right, uC3 = UV (0,1) = bottom-left.
// During the doc-space overlay these are doc-pixels; during the
// per-tile commit bake they are tile-local pixels (origin subtracted
// out by the caller). The mapping is bilinear over the quad UV; for an
// affine placement (scale + rotation) this collapses to exactly the
// affine transform of the source rect.
uniform vec2 uC0;
uniform vec2 uC1;
uniform vec2 uC2;
uniform vec2 uC3;
void main() {
    vec2 t = aQuad * 0.5 + 0.5;
    vec2 top = mix(uC0, uC1, t.x);
    vec2 bot = mix(uC3, uC2, t.x);
    vec2 docPos = mix(top, bot, t.y);
    vDocPos = docPos;
    vec4 bufPx  = uTransform * vec4(docPos, 0.0, 1.0);
    vec2 ndc = (bufPx.xy / uScreen) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
    vUv = t;
}
)";

const char* kSelFS = R"(#version 300 es
precision mediump float;
in vec2 vUv;
in vec2 vDocPos;
uniform sampler2D uContent;     // premultiplied selection content
uniform vec2 uPageMin;
uniform vec2 uPageMax;
uniform int  uPageActive;
// Source-layer opacity at overlay time. Bake (commit drop) uses 1.0
// so pixels go into the tile at full opacity; the layer's opacity is
// applied later at composite time.
uniform float uOpacity;
out vec4 outColor;
void main() {
    if (uPageActive != 0 && (
        vDocPos.x < uPageMin.x || vDocPos.x > uPageMax.x ||
        vDocPos.y < uPageMin.y || vDocPos.y > uPageMax.y)) {
        discard;
    }
    outColor = texture(uContent, vUv) * uOpacity;
}
)";

// ---- Data structures ------------------------------------------------------

struct Sample { float x, y, p; };
struct Stroke { std::vector<Sample> samples; };

// Reference-counted, kTileBytes-sized byte buffer. Used as the
// immutable carrier for a tile's interior pixels in both the per-tile
// cache and the undo system's TileSnap. Sharing is a refcount inc;
// new buffers are allocated only on tile mutation (bake, paste, etc.)
// — without sharing, allocating + zero-filling kTileBytes per snap
// took ~4 ms each and dominated commitStroke. Convention: treat the
// vector as immutable once it's been stored anywhere shared.
using TileBytes = std::shared_ptr<std::vector<uint8_t>>;

struct Tile {
    GLuint texture = 0;
    GLuint fbo     = 0;
    // Set whenever this tile's content (or any of its 8 neighbors')
    // changes. The composite path lazy-syncs the apron from neighbors
    // on the next composite pass when this is true.
    bool   apronStale = true;
    // CPU-side mirror of the FBO interior. Populated on
    // creation/load and replaced on every mutation (bake, paste,
    // selection lift, bucket fill, undo restore). The undo system's
    // BEFORE snapshot points at this exact shared_ptr instead of
    // copying — see snapshotTilesInBbox.
    TileBytes cachedBytes;
};

// ---- Tile-bytes buffer pool ---------------------------------------------
//
// commitStroke's AFTER pass allocates one kTileBytes-sized vector per
// touched tile. On this device, that single allocation costs ~4 ms
// (jemalloc's mmap path + 256 KB zero-init), and a 4-tile commit
// spends 16 ms just on allocations. By recycling vectors that the
// undo system has finished with (via shared_ptr custom deleter), the
// allocation cost is paid once and amortizes to zero after steady
// state. Capped at kMaxPoolSize buffers (~16 MB) to bound memory.
//
// Thread-safe — eviction can race against commit on the GL thread
// because the disk writer thread may be the one releasing the last
// shared_ptr reference (DiskTask owns a vector<uint8_t> copy, not
// the shared TileBytes, so this is conservative; cheap lock-take
// anyway).
class TileBufferPool {
    std::mutex mu_;
    std::vector<std::vector<uint8_t>*> free_;
    static constexpr size_t kMaxPoolSize = 64;
    // Pre-warmed at first acquire so the early commits (before any
    // undo eviction recycles buffers) pay zero allocation cost.
    static constexpr size_t kPrewarmCount = 32;
    bool warmed_ = false;
    void prewarmLocked_() {
        warmed_ = true;
        free_.reserve(kPrewarmCount);
        for (size_t i = 0; i < kPrewarmCount; ++i) {
            free_.push_back(new std::vector<uint8_t>(kTileBytes));
        }
    }
public:
    ~TileBufferPool() {
        for (auto* p : free_) delete p;
    }
    TileBytes acquire() {
        std::vector<uint8_t>* raw = nullptr;
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (!warmed_) prewarmLocked_();
            if (!free_.empty()) {
                raw = free_.back();
                free_.pop_back();
            }
        }
        if (!raw) raw = new std::vector<uint8_t>(kTileBytes);
        return std::shared_ptr<std::vector<uint8_t>>(
            raw,
            [this](std::vector<uint8_t>* p) {
                std::lock_guard<std::mutex> lock(mu_);
                if (free_.size() < kMaxPoolSize) free_.push_back(p);
                else delete p;
            });
    }
};

inline TileBufferPool& tilePool() {
    static TileBufferPool p;
    return p;
}

// Convenience: acquire a buffer pre-filled with kTileBytes from src.
inline TileBytes acquireTileBytesFrom(const uint8_t* src) {
    auto b = tilePool().acquire();
    std::memcpy(b->data(), src, kTileBytes);
    return b;
}

// Convenience: acquire a buffer zeroed (for freshly-cleared tiles).
inline TileBytes acquireZeroedTileBytes() {
    auto b = tilePool().acquire();
    std::memset(b->data(), 0, kTileBytes);
    return b;
}

// Vector-layer primitives. All coordinates are in document pixels. Each
// struct is plain-old-data so we can fwrite it byte-for-byte during
// persistence. If you change a layout, bump kShapesMagic.
enum class LayerType : int { Raster = 0, Vector = 1 };

struct Line {
    float    x0, y0;
    float    x1, y1;
    uint32_t color;     // 0xRRGGBB
    float    width;     // full line width in doc pixels
};

struct Rect {
    float    x0, y0;    // axis-aligned bbox in shape-local frame (rotation
    float    x1, y1;    //   = 0 means these are also world-space corners).
    float    rotation;  // radians, applied around the bbox center.
    uint32_t color;
    float    width;
};

struct Ellipse {
    float    cx, cy;    // center
    float    rx, ry;    // semi-axes (in shape-local frame)
    float    rotation;  // radians, applied around the center
    uint32_t color;
    float    width;
};

struct Circle {
    float    cx, cy;    // center
    float    radius;
    uint32_t color;
    float    width;
};

struct Layer {
    LayerType type = LayerType::Raster;
    std::unordered_map<int64_t, Tile> tiles;   // populated for raster
    // Vector-layer shape lists (populated for vector). Kept as separate
    // typed vectors rather than a tagged union — shape counts are small
    // and per-type iteration in the compositor is cleaner.
    std::vector<Line>    lines;
    std::vector<Rect>    rects;
    std::vector<Ellipse> ellipses;
    std::vector<Circle>  circles;
    // User-set display name. Empty = caller renders a default like
    // "layer N" / "vector N". Persisted at <layerDir>/name.txt; reads/
    // writes are guarded by g_layerNameMutex (see below).
    std::string name;
    // User-toggleable visibility. The compositor (GL thread) reads it
    // every frame; the UI thread writes it from the eye-icon tap, so we
    // use atomic to avoid a mutex on the per-layer hot path.
    // Persisted as the *presence* of <layerDir>/hidden.flag — absence
    // means visible, which is the unambiguous default for layers
    // created before this feature existed.
    std::atomic_bool visible{true};
    // Per-layer opacity in [0, 1]. The composite shaders multiply their
    // premultiplied output by this scalar. Same atomic access pattern
    // as `visible`. Persisted at <layerDir>/opacity.txt; absence means
    // 1.0 (the default for layers created before this feature existed).
    std::atomic<float> opacity{1.0f};
};

// A document Page wraps a layer stack plus an active-layer index. All
// pages live in g_pages; the active one is g_activePageIdx. Most existing
// code references the active page's slots through layers() / activeLayer()
// accessor functions so the historical single-page logic could stay put.
struct Page {
    std::vector<std::unique_ptr<Layer>> layers;
    size_t activeLayer = 0;
    // False until this page's *content* (raster tile uploads, vector
    // shape tables) has been pulled off disk. Layer metadata — type,
    // name, visibility, opacity — is always loaded for every page at
    // document-open time because it's cheap; content is not, because a
    // single raster layer on a Letter-size page is ~63 tiles x 256 KB
    // of file I/O plus a texture and an FBO each. Loading every page's
    // content up front is what made switching documents take seconds.
    // See ensurePageContentLoaded.
    bool contentLoaded = false;
};

struct DabProg {
    GLuint program    = 0;
    GLint  uTransform = -1;
    GLint  uScreen    = -1;
    GLint  uCenter    = -1;
    GLint  uRadius    = -1;
    GLint  uColor     = -1;
    GLint  uPageMin    = -1;
    GLint  uPageMax    = -1;
    GLint  uPageActive = -1;
    GLint  uHardness   = -1;
};

struct CompProg {
    GLuint program     = 0;
    GLint  uTransform  = -1;
    GLint  uScreen     = -1;
    GLint  uTileCenter = -1;
    GLint  uTileHalf   = -1;
    GLint  uTileTex    = -1;
    GLint  uOpacity    = -1;
};

struct PreviewProg {
    GLuint program   = 0;
    GLint  uBelow    = -1;
    GLint  uAbove    = -1;
    GLint  uCoverage = -1;
    GLint  uMode     = -1;
    GLint  uBrushRgb = -1;
};

struct GridProg {
    GLuint program           = 0;
    GLint  uInverseTransform = -1;
    GLint  uSpacing          = -1;
    GLint  uSubdivisions     = -1;
    GLint  uMinorWidth       = -1;
    GLint  uMajorWidth       = -1;
    GLint  uMinorColor       = -1;
    GLint  uMajorColor       = -1;
    GLint  uStyle            = -1;
    GLint  uPageMin          = -1;
    GLint  uPageMax          = -1;
    GLint  uPageActive       = -1;
};

struct LineProg {
    GLuint program    = 0;
    GLint  uTransform = -1;
    GLint  uScreen    = -1;
    GLint  uP0        = -1;
    GLint  uP1        = -1;
    GLint  uHalfWidth = -1;
    GLint  uColor     = -1;
    GLint  uPageMin    = -1;
    GLint  uPageMax    = -1;
    GLint  uPageActive = -1;
    GLint  uOpacity    = -1;
};

struct EllipseProg {
    GLuint program    = 0;
    GLint  uTransform  = -1;
    GLint  uScreen     = -1;
    GLint  uCenter     = -1;
    GLint  uRadii      = -1;
    GLint  uRotation   = -1;
    GLint  uHalfWidth  = -1;
    GLint  uColor      = -1;
    GLint  uOpacity    = -1;
    GLint  uPageMin    = -1;
    GLint  uPageMax    = -1;
    GLint  uPageActive = -1;
};

struct RectProg {
    GLuint program     = 0;
    GLint  uTransform  = -1;
    GLint  uScreen     = -1;
    GLint  uCenter     = -1;
    GLint  uHalfExt    = -1;
    GLint  uRotation   = -1;
    GLint  uHalfWidth  = -1;
    GLint  uColor      = -1;
    GLint  uOpacity    = -1;
    GLint  uPageMin    = -1;
    GLint  uPageMax    = -1;
    GLint  uPageActive = -1;
};

struct PixelGridProg {
    GLuint program          = 0;
    GLint  uInverseTransform = -1;
    GLint  uColor            = -1;
    GLint  uPageMin          = -1;
    GLint  uPageMax          = -1;
    GLint  uPageActive       = -1;
};

struct StampProg {
    GLuint program   = 0;
    GLint  uCoverage = -1;
    GLint  uColor    = -1;
    GLint  uMode     = -1;
};

struct FillProg {
    GLuint program    = 0;
    GLint  uTransform = -1;
    GLint  uScreen    = -1;
    GLint  uMin       = -1;
    GLint  uMax       = -1;
    GLint  uFillColor = -1;
};

struct PolyProg {
    GLuint program      = 0;
    GLint  uTransform   = -1;
    GLint  uScreen      = -1;
    GLint  uColor       = -1;
    GLint  uPageMin     = -1;
    GLint  uPageMax     = -1;
    GLint  uPageActive  = -1;
    GLint  uChord0      = -1;
    GLint  uChord1      = -1;
    GLint  uChordOffset = -1;
    GLint  uFeatherBand = -1;
    GLint  uChordMode   = -1;
};

struct SelProg {
    GLuint program    = 0;
    GLint  uTransform = -1;
    GLint  uScreen    = -1;
    GLint  uC0        = -1;
    GLint  uC1        = -1;
    GLint  uC2        = -1;
    GLint  uC3        = -1;
    GLint  uContent   = -1;
    GLint  uOpacity   = -1;
    GLint  uPageMin    = -1;
    GLint  uPageMax    = -1;
    GLint  uPageActive = -1;
};

struct ViewFbo {
    GLuint texture = 0;
    GLuint fbo     = 0;
    int    width   = 0;
    int    height  = 0;
    // Optional stencil renderbuffer. Only the two FBOs the shade tool
    // fills through (the screen-sized shade coverage and the tile-sized
    // stroke coverage) carry one — a stencil attachment on every scratch
    // FBO would be pure memory overhead for the paths that never use it.
    GLuint stencil = 0;
};

// ---- State (GL-thread-only unless noted) ----------------------------------

DabProg     g_dab;
CompProg    g_comp;
PreviewProg g_preview;
GridProg    g_grid;
LineProg    g_lineProg;
EllipseProg g_ellipseProg;
RectProg    g_rectProg;
PixelGridProg g_pixelGridProg;
StampProg   g_stamp;
FillProg    g_fill;
PolyProg    g_poly;
SelProg     g_sel;
GLuint      g_quadVao = 0;
GLuint      g_quadVbo = 0;
// Dynamic vertex buffer for the shade tool's closed-path fill. Holds the
// stroke's sample path followed by 4 AABB corners (see uploadShadePolygon).
GLuint      g_polyVao = 0;
GLuint      g_polyVbo = 0;
bool        g_inited  = false;

// Grid overlay state. Settable from any thread; read at multi-buffer
// composite time. Style = 0 means "use most recent non-zero style"
// internally, but the public setter only sends 1 (lines) or 2 (dots).
std::atomic<int> g_gridEnabled{0};   // 0 = off, 1 = on
std::atomic<int> g_gridStyle{1};     // 1 = lines, 2 = dots
// Minor-line spacing in doc px. Per-document: Kotlin reads it from the
// doc's page_setup.txt on switch and mirrors it here. Also the pitch
// for grid-intersection snapping (findSnap).
std::atomic<float> g_gridSpacing{50.0f};

// Page margins — rose rules inset from the page edge: across the top
// and bottom at g_marginTop doc-px, down each side at g_marginSide.
// Drawn in the page background right after the grid (under the layers,
// exported like the grid). Per-document like the grid spacing.
std::atomic<int>   g_marginEnabled{0};
std::atomic<float> g_marginTop{96.0f};
std::atomic<float> g_marginSide{72.0f};

// Pixel grid overlay — 1-buffer-pixel lines at every integer doc-px
// boundary. Only renders when enabled AND view scale is at least
// kPixelGridMinViewScale (below that the grid would be denser than the
// resolution can show, just darkening the canvas). UI toggle persists
// across launches; the zoom gate is automatic, not user-visible.
std::atomic<int> g_pixelGridEnabled{0};
// GL-thread-only flag: when true, compositeAllLayers skips the page-
// boundary outline (the 1.5-view-px gray rectangle at the page edges).
// Toggled around renderPageThumbnail's compositeAllLayers call for the
// export path so the saved image has no border at the bitmap edges.
bool g_skipPageOutline = false;
// Below this zoom the grid would be denser than the screen can show;
// keep some headroom below the gesture max (kMaxViewScale = 8.0) so the
// grid is visible across a useful zoom range, not just the very top.
constexpr float kPixelGridMinViewScale = 4.0f;

// Snapping is on by default. Settable from any thread.
std::atomic<int> g_snapEnabled{1};
// Angle-snap (LINE-tool draw + rotate handle), off by default. Mirror
// of DrawingSurfaceView.angleSnapEnabled. When on, rotations and line
// draws lock to 15° increments.
std::atomic<int> g_angleSnapEnabled{0};

// Page bounds in doc-pixels — a fixed rectangle drawn during composite
// to give the user a visual anchor when zoomed/rotated. Optional;
// disabled when w/h are 0. Set from Kotlin via setPageBounds. Stored
// under a mutex (the rect's 4 floats need to update atomically together).
std::mutex g_pageBoundsMutex;
float      g_pageX0 = 0.0f, g_pageY0 = 0.0f;
float      g_pageX1 = 0.0f, g_pageY1 = 0.0f;

struct PageClip {
    bool  active;
    float minX, minY, maxX, maxY;
};

PageClip readPageClip() {
    PageClip out;
    std::lock_guard<std::mutex> lock(g_pageBoundsMutex);
    out.minX = g_pageX0; out.minY = g_pageY0;
    out.maxX = g_pageX1; out.maxY = g_pageY1;
    out.active = (out.maxX > out.minX && out.maxY > out.minY);
    return out;
}

// Apply the page-clip uniforms to the currently-bound program. The
// caller passes the program's uPageMin/Max/Active locations and the
// page rect in whatever coord frame the program's vDocPos varying uses
// (tile-local for bake, doc-pixels everywhere else).
void uploadPageClip(GLint locMin, GLint locMax, GLint locActive,
                    const PageClip& p) {
    if (locActive >= 0) glUniform1i(locActive, p.active ? 1 : 0);
    if (p.active) {
        if (locMin >= 0) glUniform2f(locMin, p.minX, p.minY);
        if (locMax >= 0) glUniform2f(locMax, p.maxX, p.maxY);
    }
}

// Current view-zoom factor (doc-px per view-px). Set from Kotlin whenever
// the 2-finger gesture changes scale; used to keep snap/hit-test radii
// screen-relative — otherwise zooming out would make snap useless and
// zooming in would make it overshoot. Stored as raw bits in an atomic
// uint32 to avoid std::atomic<float> on platforms where it requires
// special linkage.
std::atomic<uint32_t> g_viewScaleBits{0x3F800000u};   // 1.0f
inline float currentViewScale() {
    uint32_t bits = g_viewScaleBits.load();
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f > 1e-6f ? f : 1.0f;
}

// Mirror of the user's view scale that is NEVER overwritten by the
// transient render scales used during page-thumbnail rendering and
// bucket-fill compositing (both paths save/restore g_viewScaleBits).
// UI-thread code paths — snap, hit-test, velocity gating — must read
// this instead of currentViewScale(), otherwise a thumbnail render
// landing between MotionEvent dispatches makes snap radii alternate
// between sane (e.g. 4 doc-px at 3× zoom) and absurd (240 doc-px at
// the thumbnail's 0.05× scale), which manifests as snap engagement
// flickering on every other event.
std::atomic<uint32_t> g_userViewScaleBits{0x3F800000u};   // 1.0f
inline float userViewScale() {
    uint32_t bits = g_userViewScaleBits.load();
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f > 1e-6f ? f : 1.0f;
}

// All values below are *view-pixel* targets — divided by currentViewScale()
// at use time to get the equivalent doc-pixel radius/threshold.
// Engagement zone — pen must be within this distance (in view-px,
// scaled to doc-px at use time) of a snap target to START snapping.
// Smaller is more deliberate; vector vertices farther than this on
// screen no longer pull the selection.
constexpr float    kSnapRadiusViewPx   = 12.0f;
// Release zone — once locked, the pen has to drift past this
// distance (in view-px) before unlocking. Two values: a wide one
// when the pen is slow/stationary (so natural hand tremor can't
// flick the lock off — the dominant cause of stationary-pen
// vibration when zoomed in) and a tight one once the pen is moving
// deliberately (so dragging the object clear feels responsive,
// not "rubber-banded" against the snap point).
constexpr float    kSnapReleaseSlowViewPx = 60.0f;
constexpr float    kSnapReleaseFastViewPx = 22.0f;
// Velocity gate — fresh snap engagement is suppressed when the pen
// is moving faster than this (view-px per applyMoveTo call). Existing
// locks still hold via hysteresis. The intent: "snap when pen settles
// near a target", not "snap to every vertex the pen sweeps past".
// 6 view-px/call ≈ a few cm/sec on screen at 90Hz — slow, deliberate.
constexpr float    kSnapVelocityViewPx = 6.0f;
constexpr uint32_t kSnapMarkerColor    = 0xFFC020u;       // amber
constexpr float    kSnapMarkerRViewPx  = 9.0f;            // ring radius (view px)

constexpr float kGridSubdivisions = 5.0f;        // major every Nth minor
// Half-widths in doc pixels. Lines look fine at sub-pixel widths since they're
// extended; isolated dots need to be chunkier to read.
constexpr float kGridMinorLineWidth = 0.5f;
constexpr float kGridMajorLineWidth = 1.0f;
constexpr float kGridMinorDotWidth  = 1.5f;
constexpr float kGridMajorDotWidth  = 2.5f;
constexpr float kGridMinorColor[4] = { 0.55f, 0.60f, 0.70f, 0.45f };  // straight RGBA
constexpr float kGridMajorColor[4] = { 0.40f, 0.45f, 0.55f, 0.70f };

// Margin rules. Width is in doc px (not view px) so the rules zoom and
// export exactly like the grid lines rather than like UI chrome.
constexpr uint32_t kMarginColor      = 0xD07A8Cu;   // muted notebook rose
constexpr float    kMarginAlpha      = 0.75f;
constexpr float    kMarginWidthDocPx = 2.0f;

// Live eraser preview state. Two layer-stack snapshots and one coverage
// map, all sized to the buffer dimensions and lazily (re)allocated.
//   - belowFbo: paper + layers strictly below active (alpha=1)
//   - aboveFbo: layers strictly above active, premultiplied over transparent
//   - coverage: accumulated erase-coverage alpha as the user drags
ViewFbo g_belowFbo;
ViewFbo g_aboveFbo;
ViewFbo g_coverage;
// Mirror of g_coverage AFTER the most recent real-sample dab (i.e. real
// dabs only, no predicted). Used to revert g_coverage when a new real
// sample arrives after one or more predicted samples — otherwise the
// predicted dabs persist in the live preview and visibly thicken the
// stroke as real samples catch up. Allocated lazily alongside g_coverage.
ViewFbo g_coverageReal;
bool    g_needsPreviewPrep = false;  // set at beginStroke; cleared on first extendStrokeBatch
// Set after a batch whose predicted tail produced any dabs. Triggers
// a coverage + emitter restore at the start of the next batch's real
// region, before any new real dabs are applied.
bool    g_predictionInFlight = false;
// Snapshot of g_predictionEnabled taken at beginStroke and held for the
// life of the stroke. The mirror blit and the predicted-dab path are
// gated on this — when off, no mirror cost is paid. Toggling mid-stroke
// is a no-op so the coverage mirror can't desync.
bool    g_strokePredictionActive = false;
// Tile-sized scratch coverage FBO used by the uniform-alpha bake path:
// dabs MAX-blend into it (one channel — only alpha matters), then a
// single stamp pass composites the brush color + max coverage onto the
// tile. Allocated lazily in bakeCurrentStrokeIntoTiles.
ViewFbo g_strokeCoverageTile;

// ---- Shade tool (closed-path fill) --------------------------------------
//
// The shade tool traces a brush stroke like the pen, but the start and
// end points are joined by an implicit straight chord and the enclosed
// region is filled. Outline + fill are one flat coverage mask at the
// brush alpha (no darker rim where they overlap), so the stroke always
// runs through the uniform-alpha path.
//
// The dab half of the coverage accumulates in g_coverage exactly as a
// normal stroke does — including the prediction mirror. The fill half
// can SHRINK between batches (the chord sweeps as the pen moves), so it
// can't accumulate; it is rebuilt every batch into g_shadeCoverage,
// which is the union (per-pixel MAX) of the freshly-filled polygon and
// the accumulated dab coverage. Rebuilding the fill is one fan draw
// call; rebuilding the dabs would be O(samples) per batch, which is
// exactly the per-event cost the batch path exists to avoid.
ViewFbo g_shadeCoverage;
// Per-stroke flag: snapshot of "current tool is shade" taken at
// beginStroke, so a mid-stroke tool switch can't split the stroke.
bool    g_strokeFillClosed = false;
// The stroke's path in doc-px, kept as interleaved x,y for the fill's
// vertex buffer. Mirrors g_current.samples (real samples only) and is
// temporarily extended with the predicted tail inside a batch.
std::vector<float> g_shadePolyDoc;
// Monotonically-growing doc-px bbox of every sample in the stroke
// (predicted ones included). Drives a scissor that bounds the per-batch
// clear + merge to the shaded region instead of the whole surface.
// Monotonic on purpose: the fill can shrink between batches, and a
// scissor that shrank with it would strand stale coverage outside.
bool  g_shadeBboxValid = false;
float g_shadeMinX = 0.0f, g_shadeMinY = 0.0f;
float g_shadeMaxX = 0.0f, g_shadeMaxY = 0.0f;
// Highest pressure seen among the stroke's REAL samples, i.e. its
// widest dab. Sizes the chord's soft band (see chordFeatherBand).
// Real samples only — a predicted tail is transient, and a running max
// would let one overshooting prediction widen the band for good.
float g_shadeMaxPressure = 0.0f;

// Extend the shade path with one doc-px sample, growing its bbox.
inline void shadePathPush(float x, float y) {
    g_shadePolyDoc.push_back(x);
    g_shadePolyDoc.push_back(y);
    if (!g_shadeBboxValid) {
        g_shadeMinX = g_shadeMaxX = x;
        g_shadeMinY = g_shadeMaxY = y;
        g_shadeBboxValid = true;
        return;
    }
    g_shadeMinX = std::min(g_shadeMinX, x);
    g_shadeMaxX = std::max(g_shadeMaxX, x);
    g_shadeMinY = std::min(g_shadeMinY, y);
    g_shadeMaxY = std::max(g_shadeMaxY, y);
}

// ---- Multi-buffer cache (partial re-composite path) ---------------------
//
// GLFrontBufferedRenderer hands us a freshly-cleared MB.back every
// commit, with no setting to preserve the previous frame. To avoid
// re-rendering the entire document each commit (~11 ms at our doc
// size, the dominant slice of the commit-flash window we measured via
// perfetto), we maintain our own copy of MB's pixels and restore them
// at the start of every renderDocument. Then we only re-composite the
// region the just-finished stroke actually touched, scissored to a
// dirty bbox. Saves ~9 ms per commit on typical strokes — enough to
// land commits inside one 90 Hz vsync and eliminate the flash.
ViewFbo g_mbCache;
// Set true once g_mbCache contains a valid copy of the previous
// frame's MB content. Reset whenever something invalidates that
// cache (transform/zoom changed, layer count changed, doc switched,
// etc.) — those force a full re-render that refreshes the cache.
bool    g_mbCacheValid = false;
// Doc-px bbox of the region affected by the most recent stroke /
// bucket-fill / tile-mutating op. populated set true means
// renderDocument can take the partial path; consumed (cleared) at
// the start of renderDocument. Other render triggers (layer toggle,
// doc switch, view transform change) leave this null and get a full
// re-composite.
struct PendingDirtyBbox {
    bool  populated = false;
    float minX = 0, minY = 0;
    float maxX = 0, maxY = 0;
};
PendingDirtyBbox g_pendingDirtyBbox;
// Last transform passed to renderDocument; compared against the
// current one to detect view scale / pan / rotation changes. The
// stored cache is keyed to a specific transform — if the user pans
// or zooms, the cached pixels are at the wrong screen position and
// we must do a full re-render.
float g_mbCacheTransform[16] = { 0 };
int   g_mbCacheWidth  = 0;
int   g_mbCacheHeight = 0;

std::vector<std::unique_ptr<Page>> g_pages;
size_t g_activePageIdx = 0;
size_t g_strokeTarget  = 0;          // captured at beginStroke

// Per-layer user-set names live in Layer.name. Reads from the UI thread
// (for the layer panel) and writes (rename dialogs) cross threads with
// the GL thread (which mutates layers via the pending-action queue).
// We guard string access here rather than holding a wider lock.
std::mutex g_layerNameMutex;

// Accessors for the active page's layer state. Reseat as g_activePageIdx
// changes — most existing code uses these as if they were the original
// `layers()` / `activeLayer()` globals. Both return non-const references
// (write-through). Caller must ensure at least one page exists; see
// ensureAtLeastOnePage().
inline std::vector<std::unique_ptr<Layer>>& layers() {
    return g_pages[g_activePageIdx]->layers;
}
inline size_t& activeLayer() {
    return g_pages[g_activePageIdx]->activeLayer;
}

// Active tool: 0 = brush, 1 = eraser, 7 = shade (closed-path fill; see
// kToolShade). Settable from any thread (UI thread flips it via
// setTool() on stylus-button press); the GL thread reads it at
// beginStroke into g_strokeTool so a stroke uses one consistent tool
// even if the user toggles mid-stroke.
constexpr int kToolEraser = 1;
constexpr int kToolShade  = 7;
std::atomic<int> g_currentTool{0};
int g_strokeTool = 0;

// Every raster-stroke branch is really "eraser or not" — the shade tool
// paints like the brush, so testing `== 0` for brush would silently
// route it down the eraser path.
inline bool strokeToolIsEraser() { return g_strokeTool == kToolEraser; }

// Brush color: RGB packed into low 24 bits (0xRRGGBB). Settable from any
// thread; the GL thread snapshots into g_strokeBrushColor at beginStroke.
// The alpha component of the snapshot comes from g_brushAlphaBits
// (user-controlled via the brush opacity slider).
std::atomic<uint32_t> g_currentBrushColor{0x14171Fu};
float g_strokeBrushColor[4] = { 0.08f, 0.09f, 0.12f, 1.0f };

// Brush opacity in [0, 1]. Same atomic-bits-of-float pattern as
// g_brushSizeScaleBits. Snapshotted at beginStroke so mid-stroke
// slider changes don't split a stroke; the snapshot is what the bake
// uses for g_strokeBrushColor[3] AND what the live coverage pass
// uses (so live preview matches the post-commit appearance).
std::atomic<uint32_t> g_brushAlphaBits{0x3F800000u};       // 1.0f
float g_strokeBrushAlpha = 1.0f;
inline float currentBrushAlpha() {
    uint32_t bits = g_brushAlphaBits.load();
    float v;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

// Brush "hardness" in [0, 1]: the radial fraction at which a single
// dab is fully opaque (everything outside falls off via smoothstep to
// r=1). 1.0 = solid disc, 0.0 = full gradient. Snapshotted into
// g_strokeBrushHardness at beginStroke so the live coverage pass and
// the bake see the same value.
std::atomic<uint32_t> g_brushHardnessBits{0x3F800000u};    // 1.0f
float g_strokeBrushHardness = 1.0f;
inline float currentBrushHardness() {
    uint32_t bits = g_brushHardnessBits.load();
    float v;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

// Bucket-fill bleed: how many extra pixels the fill mask grows outward
// after flood-fill, to bridge the boundary's anti-aliased gradient. 0
// means no dilation (fill stops at exact tolerance match); larger
// values produce a heavier "ink soak" into adjacent edges. UI-tunable.
std::atomic<int> g_bucketBleedPx{2};

// Brush size scale: a multiplier applied to the per-pressure dab radius.
// Snapshotted into g_strokeBrushSizeScale at beginStroke so mid-stroke
// slider changes don't split a stroke. Stored as raw bits in atomic<u32>
// for the same reason as g_viewScaleBits — std::atomic<float> needs
// special linkage on some platforms.
std::atomic<uint32_t> g_brushSizeScaleBits{0x3F800000u};   // 1.0f
float g_strokeBrushSizeScale = 1.0f;
inline float currentBrushSizeScale() {
    uint32_t bits = g_brushSizeScaleBits.load();
    float f; std::memcpy(&f, &bits, sizeof(f));
    return f > 1e-4f ? f : 1.0f;
}

// Vector tool line width (doc-pixels). Used by addLine / addRectangle /
// addEllipse / addCircle / renderShapePreview. Read at call time on the
// UI thread; subsequent changes don't retroactively affect existing
// shapes (their width is captured into the shape struct at add time).
std::atomic<uint32_t> g_vectorLineWidthBits{0x40000000u}; // 2.0f
inline float currentVectorLineWidth() {
    uint32_t bits = g_vectorLineWidthBits.load();
    float f; std::memcpy(&f, &bits, sizeof(f));
    return f > 1e-4f ? f : 2.0f;
}

Stroke     g_current;
struct DabEmitter;                  // forward-decl
extern DabEmitter g_liveEmitter;

std::string g_docDir;               // empty = persistence disabled
// True only after loadAllLayersFromDisk() has fully populated g_pages.
// Atomic + write-after-load so the UI thread can poll for "is the
// layer panel safe to render?" — without that, a syncLayerStateFromNative
// call landing mid-load reads default-constructed Layer slots (empty
// name, visible=true, type=Raster) and the layer panel shows stale
// placeholders until the user taps something.
std::atomic<bool> g_loaded{false};

// On-disk layout helpers — every page lives in its own directory and
// hosts its layer subdirs. Using these consistently is what made the
// single-page → multi-page transition feasible without changing every
// call site. Defined up here (rather than next to the persistence code)
// so they're visible to earlier callers like applyPendingLayerActions.
inline std::string pageDirOf(size_t pageIdx) {
    return g_docDir + "/page_" + std::to_string(pageIdx);
}
inline std::string layerDirIn(size_t pageIdx, size_t layerIdx) {
    return pageDirOf(pageIdx) + "/layer_" + std::to_string(layerIdx);
}
// Most callers operate on the active page implicitly (stroke commit,
// vector mutations, undo apply). They use this convenience wrapper.
inline std::string activeLayerDir(size_t layerIdx) {
    return layerDirIn(g_activePageIdx, layerIdx);
}

// Per-document memory of which page the user was last on. Without this
// every document reopens on page 1, because closeCurrentDocument resets
// g_activePageIdx to 0 and nothing restores it. Stored as a plain
// decimal index in <docDir>/active_page.txt; a missing or unparseable
// file means page 0, and the value is clamped to the page count at read
// time so a deleted page can't strand us out of range.
inline std::string activePagePath() {
    return g_docDir + "/active_page.txt";
}
inline void saveActivePageToDisk() {
    if (g_docDir.empty()) return;
    if (FILE* f = std::fopen(activePagePath().c_str(), "wb")) {
        std::fprintf(f, "%zu", g_activePageIdx);
        std::fclose(f);
    }
}
inline size_t readActivePageFromDisk(size_t pageCount) {
    if (g_docDir.empty() || pageCount == 0) return 0;
    size_t idx = 0;
    if (FILE* f = std::fopen(activePagePath().c_str(), "rb")) {
        char buf[32] = {};
        size_t n = std::fread(buf, 1, sizeof(buf) - 1, f);
        std::fclose(f);
        if (n > 0) {
            long v = std::strtol(buf, nullptr, 10);
            if (v > 0) idx = static_cast<size_t>(v);
        }
    }
    return idx < pageCount ? idx : pageCount - 1;
}

// Cross-thread queue: UI-thread layer ops enqueue, GL thread drains at the
// start of each operation.
std::mutex g_pendingMutex;
std::vector<int> g_pendingActions;
constexpr int kActionAddLayer       = 1;
constexpr int kActionCycleActive    = 2;
constexpr int kActionClearActive    = 3;
constexpr int kActionAddVectorLayer = 4;
constexpr int kActionUndo           = 5;
constexpr int kActionRedo           = 6;
constexpr int kActionAddPage        = 7;
constexpr int kActionSwitchPage     = 8;
constexpr int kActionLoadDocument   = 9;
constexpr int kActionDeleteLayer    = 10;
constexpr int kActionMoveLayer      = 11;
constexpr int kActionDeletePage     = 12;
constexpr int kActionMovePage       = 13;
constexpr int kActionImportImage    = 14;
constexpr int kActionRasterizeLayer       = 15;
constexpr int kActionRasterizeShapeBelow  = 16;
constexpr int kActionMergeLayerDown       = 17;
// Target index for the next kActionSwitchPage drained. Stored separately
// because the action queue is just `vector<int>`. Last-write-wins on
// rapid taps: exchange(-1) at drain time picks up whichever target was
// most recently written.
std::atomic<int>     g_pendingSwitchPage{-1};
// Path for the next kActionLoadDocument drained. Same last-write-wins
// shape as g_pendingSwitchPage but for an arbitrary string.
std::mutex   g_pendingDocPathMutex;
std::string  g_pendingDocPath;
// Side channels for parameterized layer ops (same last-write-wins
// pattern as g_pendingSwitchPage). The action code in the queue
// triggers the drain to read these.
std::atomic<int> g_pendingDeleteLayerIdx{-1};
std::mutex       g_pendingMoveLayerMutex;
int              g_pendingMoveLayerFrom = -1;
int              g_pendingMoveLayerTo   = -1;
// Same shape for page-level ops.
std::atomic<int> g_pendingDeletePageIdx{-1};
std::mutex       g_pendingMovePageMutex;
int              g_pendingMovePageFrom  = -1;
int              g_pendingMovePageTo    = -1;
// Rasterize-layer side channel. The action queue carries the action
// code; the layer index is stored here last-write-wins.
std::atomic<int> g_pendingRasterizeLayerIdx{-1};

// Merge-layer-down side channel. Same shape as rasterize: the action
// code goes on the queue and the index of the source layer (the one
// being merged into the layer below) is stored here last-write-wins.
std::atomic<int> g_pendingMergeLayerIdx{-1};

// Eyedropper: UI thread queues a sample-this-pixel request in DOC-px
// (floats; matches doc convention y-down). The GL thread drains it at
// the end of compositeAllLayers, walks the visible raster layers, reads
// the relevant tile FBO directly, and composites Porter-Duff "over" to
// produce the final color. Bypassing the multi-buffer is more reliable
// than glReadPixels there — some framework configurations present a
// different buffer than the one bound during the render callback. Tile
// FBOs are the source of truth; we trust them.
std::atomic<int>   g_pendingSampleHasReq{0};   // 0 = none, 1 = pending
std::atomic<int>   g_pendingSampleDocXBits{0}; // float bits via memcpy
std::atomic<int>   g_pendingSampleDocYBits{0};
std::atomic<int>   g_lastSampledRgb{-1};

// Image-import side channel. UI thread decodes the bitmap, premultiplies
// the bytes, and stores them here under g_pendingImportMutex. The drain
// in applyPendingLayerActions creates a fresh raster layer, uploads the
// bytes into a content texture, and parks the result on g_rasterSel as
// a floating selection so the user can position/scale/rotate it.
struct PendingImageImport {
    bool                 active = false;
    int                  width  = 0;
    int                  height = 0;
    std::vector<uint8_t> rgba;   // RGBA8 premultiplied, width*height*4
};
std::mutex          g_pendingImportMutex;
PendingImageImport  g_pendingImport;

// Shapes added from the UI thread are queued separately because they
// carry per-shape data that doesn't fit in the int-tagged action queue.
// Drained on the GL thread alongside the layer-action queue. One mutex
// covers all four queues — adds are infrequent and the lock is brief.
std::mutex           g_pendingShapesMutex;
std::vector<Line>    g_pendingLines;
std::vector<Rect>    g_pendingRects;
std::vector<Ellipse> g_pendingEllipses;
std::vector<Circle>  g_pendingCircles;

// Current selection — at most one shape on a vector layer. Mutated from
// the UI thread (tap to select) and read from the GL thread (highlight
// on composite, transforms during drag). Mutex-protected.
enum class ShapeKind : int {
    None = 0,
    Line = 1,
    Rect = 2,
    Ellipse = 3,
    Circle = 4,
};

struct Selection {
    ShapeKind kind     = ShapeKind::None;
    size_t    layerIdx = 0;
    size_t    shapeIdx = 0;
};

std::mutex g_selectionMutex;
Selection  g_selection;
// Additional vector shapes selected via marquee. Empty = single-select
// (only g_selection applies). Non-empty = multi-select (g_selection is
// the "primary" + these are the rest). Move and delete iterate both;
// transform handles (scale/rotate) only show when this is empty.
std::vector<Selection> g_extraSelections;

// Marquee-drag state: while the user drags on empty canvas with the
// SELECT tool, these track the in-progress rectangle for previewing
// + final hit-test on release. doc-coords; under g_selectionMutex.
bool  g_marqueeActive = false;
float g_marqueeX0 = 0.0f, g_marqueeY0 = 0.0f;
float g_marqueeX1 = 0.0f, g_marqueeY1 = 0.0f;

// Active drag interaction state (for SELECT-tool gestures). Only valid
// when g_selection.kind != None and an interaction is in progress.
// Mutex-protected by g_selectionMutex.
enum class DragMode : int {
    None = 0,
    Move = 1,
    Scale = 2,
    Rotate = 3,
    Marquee = 4,
};
struct DragState {
    DragMode mode = DragMode::None;
    int      handleIdx = 0;          // for Scale: 0..3 (or 0..1 for Line)
    // Scale: world-space anchor (opposite handle) and initial OBB
    // rotation, snapshotted at drag start so subsequent moves don't
    // accumulate float drift.
    float    anchorX = 0.0f, anchorY = 0.0f;
    float    initialRotation = 0.0f;
    // Initial half-extents at scale-drag start. Only used by raster
    // selections with fixedAspect set; lets applyRasterScaleTo derive
    // a uniform scale factor preserving the original aspect ratio.
    float    initialHalfW = 0.0f, initialHalfH = 0.0f;
    // Scale: offset from the pen to the grabbed handle, captured at drag
    // start. The dragged corner tracks (pen − offset) so it moves by
    // exactly the pen delta. Without this the corner teleports to the
    // pen on the first move — a grab anywhere inside the handle's hit
    // radius yanks the corner up to that radius away. Same role
    // moveOffset plays for Move and initialPenAngle plays for Rotate.
    float    grabOffsetX = 0.0f, grabOffsetY = 0.0f;
    // Rotate: shape center (fixed), initial pen-to-center angle, and the
    // shape's initial rotation. Also a snapshot of the initial Line
    // endpoints (for rotating Lines, which have no rotation field).
    float    centerX = 0.0f, centerY = 0.0f;
    float    initialPenAngle = 0.0f;
    Line     initialLine{};
    // Move: offset from pen to shape center, captured at drag start.
    // The shape's center tracks (pen − offset) so the user's grab-point
    // follows the pen. Snap is applied to the would-be center.
    float    moveOffsetX = 0.0f, moveOffsetY = 0.0f;
    // Snap indicator state, set by transforms when a snap is engaged.
    // Read by compositeAllLayers to draw an amber marker.
    bool     snapActive = false;
    float    snapX = 0.0f, snapY = 0.0f;
    // Previous query position (in doc-px). Used to compute pen
    // velocity per call so snap engagement can be gated — the user
    // wants snap to activate only when the pen has settled near a
    // target, not when it sweeps through the area at speed.
    bool     hasPrevQuery = false;
    float    prevQueryX = 0.0f, prevQueryY = 0.0f;
};
DragState g_drag;

// ---- Undo / redo ---------------------------------------------------------
//
// Single global stack of reversible actions. Entries are pushed at the
// point each mutation is realized (so e.g. queued vector-shape adds push
// from applyPendingShapes after they actually land in layers(), not from
// the JNI thread when they're queued). Bounded by entry count and total
// memory; oldest entries evict first.
//
// Threading: pushes happen from the GL thread (commitStroke, applyPending*,
// applyUndo) and the UI thread (deleteSelection, end of a transform drag).
// All access is protected by g_undoMutex. The actual undo() / redo() JNI
// calls just enqueue a kActionUndo/Redo into the pending-action queue;
// the GL thread runs the inverse mutation in applyPendingLayerActions so
// GL state stays single-threaded.

enum class UndoOp : int {
    RasterStroke = 0,   // tile snapshots before/after a stroke bake
    VectorAdd,          // shape was appended to a vector layer
    VectorDelete,       // shape was removed from a vector layer
    VectorMutate,       // shape was transformed (move/scale/rotate)
    LayerClear,         // active layer was cleared
    LayerAdd,           // a new layer was appended
    RasterizeShapeBelow,// vector shape was baked onto the raster layer below
                        // (combines a VectorDelete on layerIdx with a
                        // RasterStroke-like tile diff on targetLayerIdx)
    RasterizeLayer,     // a vector layer's shapes were baked into raster
                        // tiles in place; type flipped Vector → Raster
    MergeLayerDown,     // a raster layer was composited onto the layer
                        // below ("src over dst") and then deleted; the
                        // deleted layer is captured in srcSnapshot
    VectorMutateGroup,  // multiple shapes transformed together (multi-
                        // select Move). One entry, one undo step.
};

struct TileSnap {
    int  tx = 0, ty = 0;
    bool existed = false;             // tile present at snapshot time
    // Reference-counted; null when `existed` is false. See TileBytes
    // declaration for the sharing convention. Reads via `bytes->data()`
    // / `bytes->size()`. Allocate via `tilePool().acquire()` (or
    // `acquireTileBytesFrom` for a pre-filled buffer).
    TileBytes bytes;
};

struct ShapeData {
    ShapeKind kind = ShapeKind::None;
    Line    line{};
    Rect    rect{};
    Ellipse ellipse{};
    Circle  circle{};
};

// Complete snapshot of a layer's state. Used by destructive layer ops
// (merge, future delete) that need to fully recreate a layer on undo:
// type, name, visibility, opacity, plus content (raster tiles or vector
// shapes — only one applies per type).
struct LayerSnapshot {
    std::string           name;
    LayerType             type    = LayerType::Raster;
    bool                  visible = true;
    float                 opacity = 1.0f;
    std::vector<TileSnap> tiles;        // raster: every existing tile
    std::vector<Line>     lines;        // vector: shape lists
    std::vector<Rect>     rects;
    std::vector<Ellipse>  ellipses;
    std::vector<Circle>   circles;
};

struct UndoEntry {
    UndoOp op = UndoOp::RasterStroke;
    size_t layerIdx = 0;

    // RasterStroke: tile state before the bake. Redo doesn't need an
    //   afterTiles snapshot — it re-bakes from rebakeSamples instead,
    //   which is orders of magnitude cheaper in memory (samples + a
    //   brush snapshot are ~tens to hundreds of bytes vs kTileBytes
    //   per tile for full pixel storage).
    // LayerClear / RasterizeShapeBelow / MergeLayerDown still use
    //   beforeTiles + afterTiles in the classic before/after style.
    std::vector<TileSnap> beforeTiles;
    std::vector<TileSnap> afterTiles;

    // RasterStroke: inputs needed to reproduce the bake on redo. The
    // bake is deterministic given these + the pre-stroke tile state
    // restored by beforeTiles, so redo runs bakeCurrentStrokeIntoTiles
    // against the freshly-undone tiles to recreate the post-stroke
    // pixels bit-identically.
    std::vector<Sample> rebakeSamples;
    float rebakeBrushColor[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float rebakeBrushAlpha    = 1.0f;
    float rebakeBrushSize     = 1.0f;
    float rebakeBrushHardness = 1.0f;
    int   rebakeTool          = 0;
    bool  rebakeUniformAlpha  = false;
    // Shade tool: the samples also describe a closed path whose interior
    // was filled. Without this the redo would re-lay the outline and
    // drop the fill.
    bool  rebakeFillClosed    = false;
    // Page-clip snapshot — the bake's edge clipping depends on the
    // page bounds at bake time, so we capture and restore them.
    bool  rebakePageActive    = false;
    float rebakePageX0        = 0.0f;
    float rebakePageY0        = 0.0f;
    float rebakePageX1        = 0.0f;
    float rebakePageY1        = 0.0f;

    // VectorAdd:    afterShape  = pushed shape; shapeIdx = its index.
    // VectorDelete: beforeShape = removed shape; shapeIdx = original index.
    // VectorMutate: beforeShape / afterShape; shapeIdx = the shape's index.
    ShapeData beforeShape;
    ShapeData afterShape;
    size_t    shapeIdx = 0;

    // LayerClear (vector path): full pre-clear shape lists.
    std::vector<Line>    beforeLines;
    std::vector<Rect>    beforeRects;
    std::vector<Ellipse> beforeEllipses;
    std::vector<Circle>  beforeCircles;
    LayerType            layerTypeBefore = LayerType::Raster;

    // LayerAdd: type of the layer that was appended; previous active idx.
    LayerType addedLayerType  = LayerType::Raster;
    size_t    prevActiveLayer = 0;

    // RasterizeShapeBelow: layerIdx is the SOURCE vector layer (where the
    //   shape used to live); targetLayerIdx is the RASTER layer that
    //   received the bake. shapeIdx + beforeShape carry the deleted
    //   shape; beforeTiles / afterTiles carry the target's tile diff
    //   over the shape's bbox.
    // MergeLayerDown: layerIdx is the SOURCE (deleted) layer's original
    //   index; targetLayerIdx is the layer below (merge target).
    //   srcSnapshot carries the deleted layer's full state (so undo
    //   can re-create it). beforeTiles / afterTiles carry the target's
    //   tile diff.
    size_t targetLayerIdx = 0;

    // RasterizeLayer: layerTypeAfter is Raster (post-bake); layerType-
    //   Before (the existing field, also used by LayerClear) is Vector.
    LayerType layerTypeAfter = LayerType::Raster;

    // MergeLayerDown: full snapshot of the deleted source layer.
    LayerSnapshot srcSnapshot;

    // VectorMutateGroup: parallel arrays of selection + before/after
    // shape data for each shape that moved together. One entry covers
    // the entire multi-select drag so undo restores them all at once.
    std::vector<Selection> mutateGroupSels;
    std::vector<ShapeData> mutateGroupBefore;
    std::vector<ShapeData> mutateGroupAfter;

    // MergeLayerDown: undo entries that were pinned to the source
    // layer at merge time (RasterStroke, RasterizeLayer, etc.).
    // Captured before the merge's deleteLayerImpl drops them; re-
    // pushed onto the undo stack when the merge is undone so the
    // source's prior history is reachable.
    std::vector<UndoEntry> srcLayerUndoEntries;

    size_t bytes = 0;                // approximate memory cost
};

constexpr size_t kMaxUndoEntries = 50;
constexpr size_t kMaxUndoBytes   = 200u * 1024u * 1024u;   // combined cap

std::mutex            g_undoMutex;
std::deque<UndoEntry> g_undoStack;
std::deque<UndoEntry> g_redoStack;
size_t                g_undoTotalBytes = 0;
size_t                g_redoTotalBytes = 0;

// Drag-scoped pre-transform shape snapshot. Captured at beginInteractionAt
// and consumed at endInteraction to build a VectorMutate entry. Both are
// guarded by g_selectionMutex.
Selection g_transformBeforeSel;
ShapeData g_transformBeforeShape;
// Pre-transform snapshots for the extras of a multi-select Move drag.
// Parallel arrays — same length, same order. Captured at begin*; each
// non-trivial change pushes its own VectorMutate entry at end-time.
std::vector<Selection> g_transformBeforeExtraSels;
std::vector<ShapeData> g_transformBeforeExtraShapes;

// Floating raster selection ("marquee"). Lives across the SELECT_RECT
// gesture: lift → translate (one or more drags) → commit/cancel.
// While active, the source tiles have a hole where the selection was;
// the contentTex carries the lifted pixels. Mutations to GL resources
// happen on the GL thread; the active flag and pan offsets can be read
// from the UI thread under g_rasterSelMutex.
struct RasterSelection {
    bool   active = false;
    size_t layerIdx = 0;
    // Original bbox of the lift, in doc-pixels — the rectangle from which
    // pixels were copied into contentTex. Doesn't change after lift.
    float  bboxMinX = 0.0f, bboxMinY = 0.0f;
    float  bboxMaxX = 0.0f, bboxMaxY = 0.0f;
    // Current placement OBB in doc-pixels — the floating selection's
    // position/size/orientation. Initial state mirrors the lift bbox
    // (centerX/Y at bbox midpoint, halfW/H at half the lift dimensions,
    // rotation 0). Translate/scale/rotate drags mutate these in place;
    // the source contentTex is unchanged.
    float  centerX = 0.0f, centerY = 0.0f;
    float  halfW   = 0.0f, halfH   = 0.0f;
    float  rotation = 0.0f;
    // RGBA texture holding the lifted pixels, sized to the bbox in
    // doc-pixels (premultiplied to match tile storage).
    GLuint contentTex = 0;
    int    contentW = 0, contentH = 0;
    // Pre-lift snapshots of every tile that the bbox touched, used to
    // build the undo entry on commit and to restore on cancel.
    std::vector<TileSnap> liftedTiles;
    // When true, scale-handle drags preserve the OBB's initial aspect
    // ratio (uniform scaling). Set on imported images so the user can't
    // accidentally squash the picture.
    bool   fixedAspect = false;
};
std::mutex      g_rasterSelMutex;
RasterSelection g_rasterSel;

// Cross-document raster clipboard. Holds the content bytes (RGBA8,
// premultiplied, contentW × contentH) plus the placement OBB at the
// time of copy. Paste creates a fresh floating selection at the same
// doc-coord OBB. Persists for the process lifetime; cleared only by an
// explicit overwrite (i.e. another copy).
struct RasterClipboard {
    bool   present  = false;
    int    w        = 0, h = 0;
    float  centerX  = 0.0f, centerY = 0.0f;
    float  halfW    = 0.0f, halfH   = 0.0f;
    float  rotation = 0.0f;
    std::vector<uint8_t> bytes;
};
std::mutex       g_rasterClipboardMutex;
RasterClipboard  g_rasterClipboard;

// Parallel clipboard for vector shapes — copy/cut/paste of one or
// more selected vector shapes. Last-write-wins between this and the
// raster clipboard via g_clipboardKind, so paste knows which one to
// drop.
struct VectorClipboard {
    bool                   present = false;
    std::vector<ShapeData> shapes;
};
std::mutex      g_vectorClipboardMutex;
VectorClipboard g_vectorClipboard;

// 0=empty, 1=raster, 2=vector. Set by each copy/cut, read by paste.
std::atomic<int> g_clipboardKind{0};

size_t computeEntrySize(const UndoEntry& e) {
    size_t s = sizeof(UndoEntry);
    // TileSnap.bytes is a shared_ptr — over-counts when buffers are
    // shared across snaps (e.g. a BEFORE that shares storage with a
    // sibling op's AFTER), which means the undo budget evicts a bit
    // more aggressively than physical memory dictates. Acceptable;
    // accurate accounting would require pointer-set bookkeeping.
    for (const auto& t : e.beforeTiles) s += t.bytes ? t.bytes->size() : 0;
    for (const auto& t : e.afterTiles)  s += t.bytes ? t.bytes->size() : 0;
    // Rebake inputs (RasterStroke). Tiny compared to tile pixels —
    // a stroke of N samples costs ~12 N bytes.
    s += e.rebakeSamples.size() * sizeof(Sample);
    s += e.beforeLines.size()    * sizeof(Line);
    s += e.beforeRects.size()    * sizeof(Rect);
    s += e.beforeEllipses.size() * sizeof(Ellipse);
    s += e.beforeCircles.size()  * sizeof(Circle);
    // srcSnapshot — used by MergeLayerDown (and any future op that
    // captures a full layer for re-creation on undo).
    for (const auto& t : e.srcSnapshot.tiles) s += t.bytes ? t.bytes->size() : 0;
    s += e.srcSnapshot.lines.size()    * sizeof(Line);
    s += e.srcSnapshot.rects.size()    * sizeof(Rect);
    s += e.srcSnapshot.ellipses.size() * sizeof(Ellipse);
    s += e.srcSnapshot.circles.size()  * sizeof(Circle);
    s += e.srcSnapshot.name.size();
    // Nested entries (MergeLayerDown's per-layer history capture).
    for (const auto& nested : e.srcLayerUndoEntries) {
        s += computeEntrySize(nested);
    }
    // VectorMutateGroup arrays — small structs, but count them.
    s += e.mutateGroupSels.size()   * sizeof(Selection);
    s += e.mutateGroupBefore.size() * sizeof(ShapeData);
    s += e.mutateGroupAfter.size()  * sizeof(ShapeData);
    return s;
}

void clearRedoStack_locked() {
    g_redoTotalBytes = 0;
    g_redoStack.clear();
}

// Drop oldest entries until under both the count and byte budgets.
void enforceUndoBudget_locked() {
    while ((g_undoStack.size() > kMaxUndoEntries
            || g_undoTotalBytes + g_redoTotalBytes > kMaxUndoBytes)
           && !g_undoStack.empty()) {
        g_undoTotalBytes -= g_undoStack.front().bytes;
        g_undoStack.pop_front();
    }
}

// Push a fresh action onto the undo stack. Clears the redo stack (any
// re-do history is invalidated by a new mutation).
void pushUndoEntry(UndoEntry e) {
    e.bytes = computeEntrySize(e);
    std::lock_guard<std::mutex> lock(g_undoMutex);
    clearRedoStack_locked();
    g_undoTotalBytes += e.bytes;
    g_undoStack.push_back(std::move(e));
    enforceUndoBudget_locked();
}

bool shapeDataEqual(const ShapeData& a, const ShapeData& b) {
    if (a.kind != b.kind) return false;
    switch (a.kind) {
        case ShapeKind::Line:    return std::memcmp(&a.line,    &b.line,    sizeof(Line))    == 0;
        case ShapeKind::Rect:    return std::memcmp(&a.rect,    &b.rect,    sizeof(Rect))    == 0;
        case ShapeKind::Ellipse: return std::memcmp(&a.ellipse, &b.ellipse, sizeof(Ellipse)) == 0;
        case ShapeKind::Circle:  return std::memcmp(&a.circle,  &b.circle,  sizeof(Circle))  == 0;
        case ShapeKind::None:    return true;
    }
    return true;
}

// Snapshot the active selection's shape into a ShapeData (used at drag
// start and drag end for VectorMutate entries). Returns false if there's
// no selection or the layer is gone.
bool snapshotSelectionShape(const Selection& sel, ShapeData& out) {
    out.kind = ShapeKind::None;
    if (sel.kind == ShapeKind::None) return false;
    if (sel.layerIdx >= layers().size() || !layers()[sel.layerIdx]) return false;
    const Layer& layer = *layers()[sel.layerIdx];
    out.kind = sel.kind;
    switch (sel.kind) {
        case ShapeKind::Line:
            if (sel.shapeIdx >= layer.lines.size()) return false;
            out.line = layer.lines[sel.shapeIdx]; return true;
        case ShapeKind::Rect:
            if (sel.shapeIdx >= layer.rects.size()) return false;
            out.rect = layer.rects[sel.shapeIdx]; return true;
        case ShapeKind::Ellipse:
            if (sel.shapeIdx >= layer.ellipses.size()) return false;
            out.ellipse = layer.ellipses[sel.shapeIdx]; return true;
        case ShapeKind::Circle:
            if (sel.shapeIdx >= layer.circles.size()) return false;
            out.circle = layer.circles[sel.shapeIdx]; return true;
        case ShapeKind::None:
            return false;
    }
    return false;
}

constexpr float kDefaultLineWidth = 2.0f;

// ---- Helpers --------------------------------------------------------------

inline int64_t tileKey(int tx, int ty) {
    uint32_t ux = static_cast<uint32_t>(tx);
    uint32_t uy = static_cast<uint32_t>(ty);
    return static_cast<int64_t>(static_cast<uint64_t>(ux) |
                                (static_cast<uint64_t>(uy) << 32));
}

inline void unpackTileKey(int64_t k, int& tx, int& ty) {
    uint64_t uk = static_cast<uint64_t>(k);
    tx = static_cast<int32_t>(static_cast<uint32_t>(uk & 0xFFFFFFFFu));
    ty = static_cast<int32_t>(static_cast<uint32_t>(uk >> 32));
}

inline float clamp01(float v) { return v < 0 ? 0 : (v > 1 ? 1 : v); }

inline float radiusOf(float pressure) {
    // The base curve maps pressure to [kMinRadius, kMaxRadius]; the
    // current stroke's brush-size scale (snapshotted at beginStroke into
    // g_strokeBrushSizeScale) multiplies that. Both bake-time dab emitters
    // and the live preview emitter call this — they share the snapshot,
    // so live preview matches what'll be baked.
    float base = kMinRadius + clamp01(pressure) * (kMaxRadius - kMinRadius);
    return base * g_strokeBrushSizeScale;
}

// ---- Drawing primitive ----------------------------------------------------

void drawDab(float x, float y, float radius) {
    glUniform2f(g_dab.uCenter, x, y);
    glUniform1f(g_dab.uRadius, radius);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}

// ---- DabEmitter -----------------------------------------------------------

struct DabEmitter {
    bool  active        = false;
    float lastX         = 0.0f;
    float lastY         = 0.0f;
    float lastP         = 0.0f;
    float distToNextDab = 0.0f;
    // Optional clip bounds (in the same coord frame extend() is called in).
    // Dabs whose square footprint lies entirely outside these bounds are
    // skipped — saving the GL draw call. Used by bake (per-tile bounds);
    // live preview leaves the bounds at +/-inf so all dabs render.
    float clipMinX = -1e30f, clipMinY = -1e30f;
    float clipMaxX =  1e30f, clipMaxY =  1e30f;

    void reset() {
        active = false;
        distToNextDab = 0.0f;
        clipMinX = clipMinY = -1e30f;
        clipMaxX = clipMaxY =  1e30f;
    }

    void setClipBounds(float minX, float minY, float maxX, float maxY) {
        clipMinX = minX; clipMinY = minY;
        clipMaxX = maxX; clipMaxY = maxY;
    }

    void emit(float x, float y, float radius) {
        if (radius < kMinDabRadius) return;
        if (x + radius < clipMinX || x - radius > clipMaxX ||
            y + radius < clipMinY || y - radius > clipMaxY) {
            return;
        }
        drawDab(x, y, radius);
    }

    static inline float spacingOf(float pressure) {
        return std::max(kSpacing * radiusOf(pressure), kMinSpacing);
    }

    void extend(float x, float y, float p) {
        if (!active) {
            emit(x, y, radiusOf(p));
            active = true;
            lastX = x; lastY = y; lastP = p;
            distToNextDab = spacingOf(p);
            return;
        }
        float dx = x - lastX, dy = y - lastY;
        float dist = std::sqrt(dx * dx + dy * dy);
        if (dist < 1e-4f) return;
        float ux = dx / dist, uy = dy / dist;

        float traveled = 0.0f;
        while (traveled + distToNextDab <= dist) {
            traveled += distToNextDab;
            float t    = traveled / dist;
            float dabX = lastX + ux * traveled;
            float dabY = lastY + uy * traveled;
            float dabP = lastP + (p - lastP) * t;
            emit(dabX, dabY, radiusOf(dabP));
            distToNextDab = spacingOf(dabP);
        }
        distToNextDab -= (dist - traveled);
        lastX = x; lastY = y; lastP = p;
    }
};

DabEmitter g_liveEmitter;
// Snapshot of g_liveEmitter taken after the most recent real-sample
// dab. Restored before the next real dab if any predicted samples have
// been emitted in between, so the predicted dabs don't pollute the
// emitter's last-position state and bend the next real dab interpolation.
DabEmitter g_liveEmitterReal;
// Runtime toggle for motion prediction. On by default — validated as
// a clear win; the toggle remains for the rare case of needing raw
// pen tracking.
std::atomic<bool> g_predictionEnabled{true};

// "Uniform stroke alpha" mode. When true, overlapping dabs within a single
// stroke do NOT accumulate opacity; the stroke's effective per-pixel alpha
// is the max of any dab's coverage at that pixel. Atomic for cross-thread
// UI toggling; snapshotted into g_strokeUniformAlpha at beginStroke so a
// mid-stroke toggle doesn't split rendering between the two modes.
std::atomic<bool> g_strokeUniformAlphaEnabled{false};
bool              g_strokeUniformAlpha = false;

// Forward declarations; defined down with the persistence and composite
// helpers.
void saveVectorLayer(size_t layerIdx, const Layer& layer);
void loadVectorLayerShapes(Layer& layer, const std::string& dir);
void saveTileToDisk(size_t layerIdx, int64_t tileK);
void writeTileBytesToDisk(size_t layerIdx, int tx, int ty,
                          const uint8_t* bytes);
void enqueueDeferredSave(size_t layerIdx, int64_t tileK);
void drainPendingSaveTiles();
void drainPendingSaveTilesForBbox(size_t layerIdx,
                                  int tx0, int tx1, int ty0, int ty1);
static void flushDiskWriter();
void snapshotAllTiles(size_t layerIdx, std::vector<TileSnap>& out);
void snapshotTilesInBbox(size_t layerIdx, int tx0, int tx1, int ty0, int ty1,
                         std::vector<TileSnap>& out);
void uploadTileBytesAndSave(size_t layerIdx, int tx, int ty,
                            const uint8_t* bytes);
void deleteTileIfExists(size_t layerIdx, int tx, int ty);
void applyTileSnap(size_t layerIdx, const TileSnap& snap);
void deleteLayerDirIfExists(size_t layerIdx);
void deleteLayerImpl(size_t idx);
void moveLayerImpl(size_t from, size_t to);
void deletePageImpl(size_t idx);
void movePageImpl(size_t from, size_t to);
void rasterizeShapesIntoTiles(size_t targetLayerIdx,
                              const std::vector<Line>&    lines,
                              const std::vector<Rect>&    rects,
                              const std::vector<Ellipse>& ellipses,
                              const std::vector<Circle>&  circles);
void rasterizeVectorLayerImpl(size_t layerIdx);
void rasterizeShapeBelowImpl();
void mergeRasterLayerDownImpl(size_t layerIdx);
// Shape draw primitives — defined down with the composite helpers but
// called from rasterizeShapesIntoTiles up here.
void drawLineSegment(float x0, float y0, float x1, float y1,
                     uint32_t rgb, float width, float alpha);
void drawRectangleAsLines(float x0, float y0, float x1, float y1,
                          float rotation, uint32_t rgb, float width,
                          float alpha);
void drawEllipseAsLines(float cx, float cy, float rx, float ry,
                        float rotation, uint32_t rgb, float width,
                        float alpha);
void applyUndo();
void applyRedo();
void applyPendingShapes();
void ensureAtLeastOnePage();
void closeCurrentDocument();
void ensureLoaded();
void ensurePageContentLoaded(size_t pageIdx);
bool liftRasterSelectionRect(float x0, float y0, float x1, float y1);
bool liftRasterSelectionPolygon(const float* points, size_t nPoints);
void commitRasterSelectionImpl();
void cancelRasterSelectionImpl();
void discardRasterSelectionImpl();
void copyVectorSelectionImpl();
void cutVectorSelectionImpl();
bool pasteVectorSelectionImpl();
void deleteAllSelectionsImpl();
void copyRasterSelectionImpl();
bool pasteRasterSelectionImpl();
void bindRasterCompositePipeline(JNIEnv* env, jint width, jint height,
                                 jfloatArray transform);
void compositeRasterLayer(const Layer& layer, float opacityOverride = -1.0f);
void compositeVectorLayer(JNIEnv* env, const Layer& layer, size_t layerIdx,
                          jint width, jint height, jfloatArray transform);

// ---- Pending layer actions ------------------------------------------------

void enqueuePendingAction(int a) {
    std::lock_guard<std::mutex> lock(g_pendingMutex);
    g_pendingActions.push_back(a);
}

void applyPendingLayerActions() {
    std::vector<int> actions;
    {
        std::lock_guard<std::mutex> lock(g_pendingMutex);
        actions.swap(g_pendingActions);
    }
    // Every action below uses layers() / activeLayer(), which need a page.
    if (!actions.empty()) {
        ensureAtLeastOnePage();
        // Any pending action (layer add/move/delete, undo/redo, doc
        // switch, etc.) potentially changes MB content outside what
        // any single stroke could touch. Invalidate the partial-
        // recomposite cache so the next renderDocument does a full
        // re-render and refreshes the cache.
        g_mbCacheValid = false;
        // Drain deferred saves now, BEFORE the actions execute. The
        // queue's keys are layerIdx-relative-to-active-page; layer or
        // page mutations would re-map them to the wrong destinations.
        drainPendingSaveTiles();
    }
    for (int a : actions) {
        if (a == kActionAddLayer) {
            size_t prevActive = activeLayer();
            layers().push_back(std::make_unique<Layer>());
            activeLayer() = layers().size() - 1;
            LOGI("layer added (count=%zu, active=%zu)",
                 layers().size(), activeLayer());
            UndoEntry e;
            e.op = UndoOp::LayerAdd;
            e.layerIdx = activeLayer();
            e.addedLayerType = LayerType::Raster;
            e.prevActiveLayer = prevActive;
            pushUndoEntry(std::move(e));
        } else if (a == kActionCycleActive && !layers().empty()) {
            activeLayer() = (activeLayer() + 1) % layers().size();
            LOGI("active layer cycled to %zu/%zu",
                 activeLayer(), layers().size() - 1);
        } else if (a == kActionAddVectorLayer) {
            size_t prevActive = activeLayer();
            auto layer = std::make_unique<Layer>();
            layer->type = LayerType::Vector;
            layers().push_back(std::move(layer));
            activeLayer() = layers().size() - 1;
            LOGI("vector layer added (count=%zu, active=%zu)",
                 layers().size(), activeLayer());
            // Marker file so loadAllLayersFromDisk recognizes the type
            // even before any line is drawn. saveVectorLayer creates the
            // dir if needed and writes a header with zero shapes.
            saveVectorLayer(activeLayer(), *layers()[activeLayer()]);
            UndoEntry e;
            e.op = UndoOp::LayerAdd;
            e.layerIdx = activeLayer();
            e.addedLayerType = LayerType::Vector;
            e.prevActiveLayer = prevActive;
            pushUndoEntry(std::move(e));
        } else if (a == kActionClearActive
                   && activeLayer() < layers().size()
                   && layers()[activeLayer()]) {
            // A floating selection on this layer wouldn't survive the
            // clear; cancel it (restoring its pre-lift bytes) so the
            // pre-clear snapshot below captures a consistent state.
            cancelRasterSelectionImpl();
            Layer& layer = *layers()[activeLayer()];
            // Snapshot pre-clear state for undo (raster tiles or vector
            // shapes, depending on layer type).
            UndoEntry undo;
            undo.op = UndoOp::LayerClear;
            undo.layerIdx = activeLayer();
            undo.layerTypeBefore = layer.type;
            if (layer.type == LayerType::Raster) {
                snapshotAllTiles(activeLayer(), undo.beforeTiles);
            } else {
                undo.beforeLines    = layer.lines;
                undo.beforeRects    = layer.rects;
                undo.beforeEllipses = layer.ellipses;
                undo.beforeCircles  = layer.circles;
            }
            bool somethingToUndo = !undo.beforeTiles.empty()
                                || !undo.beforeLines.empty()
                                || !undo.beforeRects.empty()
                                || !undo.beforeEllipses.empty()
                                || !undo.beforeCircles.empty();
            // Drop GL resources for raster tiles.
            for (auto& kv : layer.tiles) {
                if (kv.second.fbo)     glDeleteFramebuffers(1, &kv.second.fbo);
                if (kv.second.texture) glDeleteTextures(1, &kv.second.texture);
            }
            layer.tiles.clear();
            // Drop all vector shapes too.
            layer.lines.clear();
            layer.rects.clear();
            layer.ellipses.clear();
            layer.circles.clear();
            // Wipe the on-disk copy so the cleared state persists.
            // Metadata files (name.txt, opacity.txt, hidden.flag) belong
            // to the layer itself, not its content — keep them so a
            // user-set name / opacity / hidden state survives a clear.
            if (!g_docDir.empty()) {
                std::string layerDir = activeLayerDir(activeLayer());
                DIR* d = opendir(layerDir.c_str());
                if (d) {
                    struct dirent* e;
                    while ((e = readdir(d)) != nullptr) {
                        const char* n = e->d_name;
                        if (n[0] == '.') continue;
                        if (std::strcmp(n, "name.txt") == 0
                            || std::strcmp(n, "opacity.txt") == 0
                            || std::strcmp(n, "hidden.flag") == 0) continue;
                        std::string p = layerDir + "/" + n;
                        unlink(p.c_str());
                    }
                    closedir(d);
                }
                // For vector layers, re-create the empty marker so the
                // type is preserved across launches.
                if (layer.type == LayerType::Vector) {
                    saveVectorLayer(activeLayer(), layer);
                }
            }
            LOGI("active layer %zu cleared", activeLayer());
            if (somethingToUndo) pushUndoEntry(std::move(undo));
        } else if (a == kActionUndo) {
            // Realize any queued shapes first so they end up on the undo
            // stack and can themselves be undone in the natural order.
            applyPendingShapes();
            applyUndo();
        } else if (a == kActionRedo) {
            applyPendingShapes();
            applyRedo();
        } else if (a == kActionAddPage) {
            // Apply any pending shapes to the current page first.
            applyPendingShapes();
            // Snapshot pre-state for clearing selection / undo.
            {
                std::lock_guard<std::mutex> lock(g_selectionMutex);
                g_selection = Selection{};
            }
            {
                std::lock_guard<std::mutex> lock(g_undoMutex);
                g_undoStack.clear(); g_undoTotalBytes = 0;
                g_redoStack.clear(); g_redoTotalBytes = 0;
            }
            g_pages.push_back(std::make_unique<Page>());
            // Brand-new page — nothing on disk to lazily load, so mark
            // it resident rather than leaving ensurePageContentLoaded
            // to scan an empty directory on every frame.
            g_pages.back()->contentLoaded = true;
            g_activePageIdx = g_pages.size() - 1;
            saveActivePageToDisk();
            LOGI("page added (count=%zu, active=%zu)",
                 g_pages.size(), g_activePageIdx);
        } else if (a == kActionLoadDocument) {
            std::string newPath;
            {
                std::lock_guard<std::mutex> lock(g_pendingDocPathMutex);
                newPath = g_pendingDocPath;
                g_pendingDocPath.clear();
            }
            if (newPath.empty()) continue;
            // Make sure all pending tile writes for the OLD doc reach
            // disk before we drop its GL state. The queued tasks hold
            // absolute paths captured at enqueue time, so they'd land
            // correctly even without this flush — but flushing here
            // guarantees the user's last strokes are persisted before
            // the doc switch is observable.
            flushDiskWriter();
            closeCurrentDocument();
            g_docDir = newPath;
            // The render entry point already called ensureLoaded() before
            // we got here (g_loaded was true at that point so it no-op'd).
            // Re-run it now so the new doc's pages/tiles are available
            // before the SAME pass tries to composite — otherwise layers()
            // dereferences an empty g_pages and we crash.
            ensureLoaded();
            LOGI("loaded document: %s", g_docDir.c_str());
        } else if (a == kActionDeleteLayer) {
            int idx = g_pendingDeleteLayerIdx.exchange(-1);
            if (idx >= 0) {
                applyPendingShapes();
                deleteLayerImpl(static_cast<size_t>(idx));
            }
        } else if (a == kActionMoveLayer) {
            int from = -1, to = -1;
            {
                std::lock_guard<std::mutex> lock(g_pendingMoveLayerMutex);
                from = g_pendingMoveLayerFrom;
                to   = g_pendingMoveLayerTo;
                g_pendingMoveLayerFrom = -1;
                g_pendingMoveLayerTo   = -1;
            }
            if (from >= 0 && to >= 0) {
                applyPendingShapes();
                moveLayerImpl(static_cast<size_t>(from),
                              static_cast<size_t>(to));
            }
        } else if (a == kActionDeletePage) {
            int idx = g_pendingDeletePageIdx.exchange(-1);
            if (idx >= 0) {
                applyPendingShapes();
                deletePageImpl(static_cast<size_t>(idx));
            }
        } else if (a == kActionRasterizeLayer) {
            int idx = g_pendingRasterizeLayerIdx.exchange(-1);
            if (idx >= 0) {
                applyPendingShapes();
                rasterizeVectorLayerImpl(static_cast<size_t>(idx));
            }
        } else if (a == kActionRasterizeShapeBelow) {
            applyPendingShapes();
            rasterizeShapeBelowImpl();
        } else if (a == kActionMergeLayerDown) {
            int idx = g_pendingMergeLayerIdx.exchange(-1);
            if (idx >= 0) {
                applyPendingShapes();
                mergeRasterLayerDownImpl(static_cast<size_t>(idx));
            }
        } else if (a == kActionImportImage) {
            // Take ownership of the pending image bytes under the mutex
            // so the UI thread can post a fresh import without racing.
            PendingImageImport pending;
            {
                std::lock_guard<std::mutex> lock(g_pendingImportMutex);
                if (!g_pendingImport.active) continue;
                pending = std::move(g_pendingImport);
                g_pendingImport.active = false;
                g_pendingImport.rgba.clear();
            }
            // A floating raster selection from earlier work needs to be
            // committed first so we don't lose it (and so g_rasterSel is
            // free for the new import).
            commitRasterSelectionImpl();

            // Append a fresh raster layer and make it active.
            size_t prevActive = activeLayer();
            layers().push_back(std::make_unique<Layer>());
            activeLayer() = layers().size() - 1;
            UndoEntry undo;
            undo.op = UndoOp::LayerAdd;
            undo.layerIdx = activeLayer();
            undo.addedLayerType = LayerType::Raster;
            undo.prevActiveLayer = prevActive;
            pushUndoEntry(std::move(undo));

            // Upload the image into a fresh content texture.
            GLuint contentTex = 0;
            glGenTextures(1, &contentTex);
            glBindTexture(GL_TEXTURE_2D, contentTex);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,
                         pending.width, pending.height, 0,
                         GL_RGBA, GL_UNSIGNED_BYTE, pending.rgba.data());
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glBindTexture(GL_TEXTURE_2D, 0);

            // Compute an initial OBB centered on the page (or at the
            // doc origin if no page bounds are set), scaled so the
            // longer image dimension is at most 80% of the longer page
            // dimension — fits with breathing room.
            float pcx = 0.0f, pcy = 0.0f;
            float pageW = static_cast<float>(pending.width);
            float pageH = static_cast<float>(pending.height);
            PageClip page = readPageClip();
            if (page.active) {
                pcx = (page.minX + page.maxX) * 0.5f;
                pcy = (page.minY + page.maxY) * 0.5f;
                pageW = page.maxX - page.minX;
                pageH = page.maxY - page.minY;
            }
            float fitW = pageW * 0.8f;
            float fitH = pageH * 0.8f;
            float fx = fitW / static_cast<float>(pending.width);
            float fy = fitH / static_cast<float>(pending.height);
            // If the image already fits inside the budget at 1:1, don't
            // upscale it — show actual size. Otherwise scale to fit.
            float scale = std::min(1.0f, std::min(fx, fy));
            float halfW = pending.width  * scale * 0.5f;
            float halfH = pending.height * scale * 0.5f;

            {
                std::lock_guard<std::mutex> lock(g_rasterSelMutex);
                g_rasterSel.active   = true;
                g_rasterSel.layerIdx = activeLayer();
                g_rasterSel.bboxMinX = pcx - halfW;
                g_rasterSel.bboxMinY = pcy - halfH;
                g_rasterSel.bboxMaxX = pcx + halfW;
                g_rasterSel.bboxMaxY = pcy + halfH;
                g_rasterSel.centerX  = pcx;
                g_rasterSel.centerY  = pcy;
                g_rasterSel.halfW    = halfW;
                g_rasterSel.halfH    = halfH;
                g_rasterSel.rotation = 0.0f;
                g_rasterSel.contentTex = contentTex;
                g_rasterSel.contentW   = pending.width;
                g_rasterSel.contentH   = pending.height;
                // Imported image isn't lifted from anywhere; canceling
                // simply drops it (and the empty layer remains).
                g_rasterSel.liftedTiles.clear();
                g_rasterSel.fixedAspect = true;
            }
            LOGI("import image: %dx%d → layer %zu (scale=%.3f)",
                 pending.width, pending.height, activeLayer(),
                 static_cast<double>(scale));
        } else if (a == kActionMovePage) {
            int from = -1, to = -1;
            {
                std::lock_guard<std::mutex> lock(g_pendingMovePageMutex);
                from = g_pendingMovePageFrom;
                to   = g_pendingMovePageTo;
                g_pendingMovePageFrom = -1;
                g_pendingMovePageTo   = -1;
            }
            if (from >= 0 && to >= 0) {
                applyPendingShapes();
                movePageImpl(static_cast<size_t>(from),
                             static_cast<size_t>(to));
            }
        } else if (a == kActionSwitchPage) {
            int target = g_pendingSwitchPage.exchange(-1);
            if (target >= 0
                && static_cast<size_t>(target) < g_pages.size()
                && static_cast<size_t>(target) != g_activePageIdx) {
                // Drain pending shapes onto the OLD page first.
                applyPendingShapes();
                // A floating raster selection refers to a specific layer
                // on the current page; cancel-restore before swapping.
                cancelRasterSelectionImpl();
                // Selection and undo are scoped to the active page; reset.
                {
                    std::lock_guard<std::mutex> lock(g_selectionMutex);
                    g_selection = Selection{};
                }
                {
                    std::lock_guard<std::mutex> lock(g_undoMutex);
                    g_undoStack.clear(); g_undoTotalBytes = 0;
                    g_redoStack.clear(); g_redoTotalBytes = 0;
                }
                g_activePageIdx = static_cast<size_t>(target);
                // Pages load their content lazily; this is the moment
                // the target page's pixels are actually needed.
                ensurePageContentLoaded(g_activePageIdx);
                saveActivePageToDisk();
                LOGI("switched to page %d", target);
            }
        }
    }
}

// Guarantees g_pages has at least one page and g_activePageIdx is in
// range. layers() / activeLayer() are unsafe to call before this. Run
// from every JNI entry point that touches the active-page layer state.
void ensureAtLeastOnePage() {
    if (g_pages.empty()) {
        g_pages.push_back(std::make_unique<Page>());
        // Synthesized, not read from disk — see the addPage path.
        g_pages.back()->contentLoaded = true;
        g_activePageIdx = 0;
    }
    if (g_activePageIdx >= g_pages.size()) {
        g_activePageIdx = g_pages.size() - 1;
    }
}

void ensureAtLeastOneLayer() {
    ensureAtLeastOnePage();
    if (layers().empty()) {
        layers().push_back(std::make_unique<Layer>());
        activeLayer() = 0;
    }
    if (activeLayer() >= layers().size()) {
        activeLayer() = layers().size() - 1;
    }
}

// Tear down all in-memory state belonging to the currently-loaded
// document so a different document can take its place. Run on the GL
// thread (frees tile FBOs/textures). Does not touch g_docDir — caller
// changes that after this returns.
void closeCurrentDocument() {
    // A floating selection is tied to a specific layer; discard it
    // before tearing down the layer it points at. We bake-back via
    // cancel so the source pixels are restored — but the tiles get
    // freed below anyway, so this just keeps the in-memory state clean.
    cancelRasterSelectionImpl();
    for (auto& page : g_pages) {
        if (!page) continue;
        for (auto& layer : page->layers) {
            if (!layer) continue;
            for (auto& kv : layer->tiles) {
                if (kv.second.fbo)     glDeleteFramebuffers(1, &kv.second.fbo);
                if (kv.second.texture) glDeleteTextures(1, &kv.second.texture);
            }
            layer->tiles.clear();
        }
    }
    g_pages.clear();
    g_activePageIdx = 0;

    {
        std::lock_guard<std::mutex> lock(g_selectionMutex);
        g_selection = Selection{};
    }
    {
        std::lock_guard<std::mutex> lock(g_undoMutex);
        g_undoStack.clear(); g_undoTotalBytes = 0;
        g_redoStack.clear(); g_redoTotalBytes = 0;
    }
    {
        std::lock_guard<std::mutex> lock(g_pendingShapesMutex);
        g_pendingLines.clear();
        g_pendingRects.clear();
        g_pendingEllipses.clear();
        g_pendingCircles.clear();
    }
    g_current.samples.clear();
    g_liveEmitter.reset();
    g_loaded.store(false, std::memory_order_release);
}

// ---- Shader / program helpers --------------------------------------------

GLuint compile(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetShaderInfoLog(s, sizeof(log), nullptr, log);
        LOGE("shader compile failed: %s", log);
    }
    return s;
}

GLuint linkProgram(const char* vs, const char* fs) {
    GLuint v = compile(GL_VERTEX_SHADER,   vs);
    GLuint f = compile(GL_FRAGMENT_SHADER, fs);
    GLuint p = glCreateProgram();
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetProgramInfoLog(p, sizeof(log), nullptr, log);
        LOGE("program link failed: %s", log);
    }
    glDeleteShader(v);
    glDeleteShader(f);
    return p;
}

void ensureInited() {
    if (g_inited) return;

    g_dab.program    = linkProgram(kDabVS, kDabFS);
    g_dab.uTransform = glGetUniformLocation(g_dab.program, "uTransform");
    g_dab.uScreen    = glGetUniformLocation(g_dab.program, "uScreen");
    g_dab.uCenter    = glGetUniformLocation(g_dab.program, "uCenter");
    g_dab.uRadius    = glGetUniformLocation(g_dab.program, "uRadius");
    g_dab.uColor     = glGetUniformLocation(g_dab.program, "uColor");
    g_dab.uHardness  = glGetUniformLocation(g_dab.program, "uHardness");
    g_dab.uPageMin    = glGetUniformLocation(g_dab.program, "uPageMin");
    g_dab.uPageMax    = glGetUniformLocation(g_dab.program, "uPageMax");
    g_dab.uPageActive = glGetUniformLocation(g_dab.program, "uPageActive");

    g_comp.program     = linkProgram(kCompVS, kCompFS);
    g_comp.uTransform  = glGetUniformLocation(g_comp.program, "uTransform");
    g_comp.uScreen     = glGetUniformLocation(g_comp.program, "uScreen");
    g_comp.uTileCenter = glGetUniformLocation(g_comp.program, "uTileCenter");
    g_comp.uTileHalf   = glGetUniformLocation(g_comp.program, "uTileHalf");
    g_comp.uTileTex    = glGetUniformLocation(g_comp.program, "uTileTex");
    g_comp.uOpacity    = glGetUniformLocation(g_comp.program, "uOpacity");

    g_preview.program   = linkProgram(kPreviewVS, kPreviewFS);
    g_preview.uBelow    = glGetUniformLocation(g_preview.program, "uBelow");
    g_preview.uAbove    = glGetUniformLocation(g_preview.program, "uAbove");
    g_preview.uCoverage = glGetUniformLocation(g_preview.program, "uCoverage");
    g_preview.uMode     = glGetUniformLocation(g_preview.program, "uMode");
    g_preview.uBrushRgb = glGetUniformLocation(g_preview.program, "uBrushRgb");

    g_grid.program           = linkProgram(kGridVS, kGridFS);
    g_grid.uInverseTransform = glGetUniformLocation(g_grid.program, "uInverseTransform");
    g_grid.uSpacing          = glGetUniformLocation(g_grid.program, "uSpacing");
    g_grid.uSubdivisions     = glGetUniformLocation(g_grid.program, "uSubdivisions");
    g_grid.uMinorWidth       = glGetUniformLocation(g_grid.program, "uMinorWidth");
    g_grid.uMajorWidth       = glGetUniformLocation(g_grid.program, "uMajorWidth");
    g_grid.uMinorColor       = glGetUniformLocation(g_grid.program, "uMinorColor");
    g_grid.uMajorColor       = glGetUniformLocation(g_grid.program, "uMajorColor");
    g_grid.uStyle            = glGetUniformLocation(g_grid.program, "uStyle");
    g_grid.uPageMin          = glGetUniformLocation(g_grid.program, "uPageMin");
    g_grid.uPageMax          = glGetUniformLocation(g_grid.program, "uPageMax");
    g_grid.uPageActive       = glGetUniformLocation(g_grid.program, "uPageActive");

    g_lineProg.program    = linkProgram(kLineVS, kLineFS);
    g_lineProg.uTransform = glGetUniformLocation(g_lineProg.program, "uTransform");
    g_lineProg.uScreen    = glGetUniformLocation(g_lineProg.program, "uScreen");
    g_lineProg.uP0        = glGetUniformLocation(g_lineProg.program, "uP0");
    g_lineProg.uP1        = glGetUniformLocation(g_lineProg.program, "uP1");
    g_lineProg.uHalfWidth = glGetUniformLocation(g_lineProg.program, "uHalfWidth");
    g_lineProg.uColor     = glGetUniformLocation(g_lineProg.program, "uColor");
    g_lineProg.uPageMin    = glGetUniformLocation(g_lineProg.program, "uPageMin");
    g_lineProg.uPageMax    = glGetUniformLocation(g_lineProg.program, "uPageMax");
    g_lineProg.uPageActive = glGetUniformLocation(g_lineProg.program, "uPageActive");
    g_lineProg.uOpacity    = glGetUniformLocation(g_lineProg.program, "uOpacity");
    // GLES uniforms default to 0 at link time, but compositeVectorLayer
    // is the only place that sets uOpacity to a non-1 value (and it
    // resets to 1.0 on the way out). Without this initial set, a frame
    // that draws line-program content BEFORE any vector layer
    // composites — e.g. the raster selection's dashed marquee + handles
    // when there's no vector layer in the doc at all — would render
    // with uOpacity = 0 and disappear. Same defensiveness for g_comp.
    glUseProgram(g_lineProg.program);
    glUniform1f(g_lineProg.uOpacity, 1.0f);
    glUseProgram(g_comp.program);
    glUniform1f(g_comp.uOpacity, 1.0f);
    glUseProgram(0);

    g_ellipseProg.program     = linkProgram(kEllipseVS, kEllipseFS);
    g_ellipseProg.uTransform  = glGetUniformLocation(g_ellipseProg.program, "uTransform");
    g_ellipseProg.uScreen     = glGetUniformLocation(g_ellipseProg.program, "uScreen");
    g_ellipseProg.uCenter     = glGetUniformLocation(g_ellipseProg.program, "uCenter");
    g_ellipseProg.uRadii      = glGetUniformLocation(g_ellipseProg.program, "uRadii");
    g_ellipseProg.uRotation   = glGetUniformLocation(g_ellipseProg.program, "uRotation");
    g_ellipseProg.uHalfWidth  = glGetUniformLocation(g_ellipseProg.program, "uHalfWidth");
    g_ellipseProg.uColor      = glGetUniformLocation(g_ellipseProg.program, "uColor");
    g_ellipseProg.uOpacity    = glGetUniformLocation(g_ellipseProg.program, "uOpacity");
    g_ellipseProg.uPageMin    = glGetUniformLocation(g_ellipseProg.program, "uPageMin");
    g_ellipseProg.uPageMax    = glGetUniformLocation(g_ellipseProg.program, "uPageMax");
    g_ellipseProg.uPageActive = glGetUniformLocation(g_ellipseProg.program, "uPageActive");
    glUseProgram(g_ellipseProg.program);
    glUniform1f(g_ellipseProg.uOpacity, 1.0f);
    glUseProgram(0);

    g_rectProg.program     = linkProgram(kRectVS, kRectFS);
    g_rectProg.uTransform  = glGetUniformLocation(g_rectProg.program, "uTransform");
    g_rectProg.uScreen     = glGetUniformLocation(g_rectProg.program, "uScreen");
    g_rectProg.uCenter     = glGetUniformLocation(g_rectProg.program, "uCenter");
    g_rectProg.uHalfExt    = glGetUniformLocation(g_rectProg.program, "uHalfExt");
    g_rectProg.uRotation   = glGetUniformLocation(g_rectProg.program, "uRotation");
    g_rectProg.uHalfWidth  = glGetUniformLocation(g_rectProg.program, "uHalfWidth");
    g_rectProg.uColor      = glGetUniformLocation(g_rectProg.program, "uColor");
    g_rectProg.uOpacity    = glGetUniformLocation(g_rectProg.program, "uOpacity");
    g_rectProg.uPageMin    = glGetUniformLocation(g_rectProg.program, "uPageMin");
    g_rectProg.uPageMax    = glGetUniformLocation(g_rectProg.program, "uPageMax");
    g_rectProg.uPageActive = glGetUniformLocation(g_rectProg.program, "uPageActive");
    glUseProgram(g_rectProg.program);
    glUniform1f(g_rectProg.uOpacity, 1.0f);
    glUseProgram(0);

    g_pixelGridProg.program          = linkProgram(kPixelGridVS, kPixelGridFS);
    g_pixelGridProg.uInverseTransform = glGetUniformLocation(g_pixelGridProg.program, "uInverseTransform");
    g_pixelGridProg.uColor            = glGetUniformLocation(g_pixelGridProg.program, "uColor");
    g_pixelGridProg.uPageMin          = glGetUniformLocation(g_pixelGridProg.program, "uPageMin");
    g_pixelGridProg.uPageMax          = glGetUniformLocation(g_pixelGridProg.program, "uPageMax");
    g_pixelGridProg.uPageActive       = glGetUniformLocation(g_pixelGridProg.program, "uPageActive");

    g_stamp.program   = linkProgram(kStampVS, kStampFS);
    g_stamp.uCoverage = glGetUniformLocation(g_stamp.program, "uCoverage");
    g_stamp.uColor    = glGetUniformLocation(g_stamp.program, "uColor");
    g_stamp.uMode     = glGetUniformLocation(g_stamp.program, "uMode");

    g_fill.program    = linkProgram(kFillVS, kFillFS);
    g_fill.uTransform = glGetUniformLocation(g_fill.program, "uTransform");
    g_fill.uScreen    = glGetUniformLocation(g_fill.program, "uScreen");
    g_fill.uMin       = glGetUniformLocation(g_fill.program, "uMin");
    g_fill.uMax       = glGetUniformLocation(g_fill.program, "uMax");
    g_fill.uFillColor = glGetUniformLocation(g_fill.program, "uFillColor");

    g_poly.program     = linkProgram(kPolyVS, kPolyFS);
    g_poly.uTransform  = glGetUniformLocation(g_poly.program, "uTransform");
    g_poly.uScreen     = glGetUniformLocation(g_poly.program, "uScreen");
    g_poly.uColor      = glGetUniformLocation(g_poly.program, "uColor");
    g_poly.uPageMin    = glGetUniformLocation(g_poly.program, "uPageMin");
    g_poly.uPageMax    = glGetUniformLocation(g_poly.program, "uPageMax");
    g_poly.uPageActive = glGetUniformLocation(g_poly.program, "uPageActive");
    g_poly.uChord0      = glGetUniformLocation(g_poly.program, "uChord0");
    g_poly.uChord1      = glGetUniformLocation(g_poly.program, "uChord1");
    g_poly.uChordOffset = glGetUniformLocation(g_poly.program, "uChordOffset");
    g_poly.uFeatherBand = glGetUniformLocation(g_poly.program, "uFeatherBand");
    g_poly.uChordMode   = glGetUniformLocation(g_poly.program, "uChordMode");

    g_sel.program    = linkProgram(kSelVS, kSelFS);
    g_sel.uTransform = glGetUniformLocation(g_sel.program, "uTransform");
    g_sel.uScreen    = glGetUniformLocation(g_sel.program, "uScreen");
    g_sel.uC0        = glGetUniformLocation(g_sel.program, "uC0");
    g_sel.uC1        = glGetUniformLocation(g_sel.program, "uC1");
    g_sel.uC2        = glGetUniformLocation(g_sel.program, "uC2");
    g_sel.uC3        = glGetUniformLocation(g_sel.program, "uC3");
    g_sel.uContent   = glGetUniformLocation(g_sel.program, "uContent");
    g_sel.uOpacity   = glGetUniformLocation(g_sel.program, "uOpacity");
    g_sel.uPageMin    = glGetUniformLocation(g_sel.program, "uPageMin");
    g_sel.uPageMax    = glGetUniformLocation(g_sel.program, "uPageMax");
    g_sel.uPageActive = glGetUniformLocation(g_sel.program, "uPageActive");
    // Default to opaque; the overlay path overrides with the source
    // layer's opacity, the bake path explicitly sets 1.0.
    glUseProgram(g_sel.program);
    glUniform1f(g_sel.uOpacity, 1.0f);
    glUseProgram(0);

    const float verts[] = {
        -1.0f, -1.0f,
         1.0f, -1.0f,
        -1.0f,  1.0f,
         1.0f,  1.0f,
    };
    glGenVertexArrays(1, &g_quadVao);
    glGenBuffers(1, &g_quadVbo);
    glBindVertexArray(g_quadVao);
    glBindBuffer(GL_ARRAY_BUFFER, g_quadVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);

    // Shade-tool path buffer. Contents are re-uploaded per stroke batch
    // (and once per bake); the VAO just fixes the attribute layout.
    glGenVertexArrays(1, &g_polyVao);
    glGenBuffers(1, &g_polyVbo);
    glBindVertexArray(g_polyVao);
    glBindBuffer(GL_ARRAY_BUFFER, g_polyVbo);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);

    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);  // premultiplied

    g_inited = true;
    LOGI("renderer initialized");
}

void uploadMat4(JNIEnv* env, GLint loc, jfloatArray transform) {
    jfloat* arr = env->GetFloatArrayElements(transform, nullptr);
    glUniformMatrix4fv(loc, 1, GL_FALSE, arr);
    env->ReleaseFloatArrayElements(transform, arr, JNI_ABORT);
}

// General 4x4 matrix inverse via cofactor expansion (column-major in/out).
// Returns true on success; on a singular matrix, leaves out untouched and
// returns false. Used to map buffer-pixel coords back to document-pixel
// coords for the grid overlay shader.
bool invertMat4(const float* m, float* out) {
    float inv[16];
    inv[0]  =  m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15] + m[9]*m[7]*m[14] + m[13]*m[6]*m[11] - m[13]*m[7]*m[10];
    inv[4]  = -m[4]*m[10]*m[15] + m[4]*m[11]*m[14] + m[8]*m[6]*m[15] - m[8]*m[7]*m[14] - m[12]*m[6]*m[11] + m[12]*m[7]*m[10];
    inv[8]  =  m[4]*m[9]*m[15]  - m[4]*m[11]*m[13] - m[8]*m[5]*m[15] + m[8]*m[7]*m[13] + m[12]*m[5]*m[11] - m[12]*m[7]*m[9];
    inv[12] = -m[4]*m[9]*m[14]  + m[4]*m[10]*m[13] + m[8]*m[5]*m[14] - m[8]*m[6]*m[13] - m[12]*m[5]*m[10] + m[12]*m[6]*m[9];
    inv[1]  = -m[1]*m[10]*m[15] + m[1]*m[11]*m[14] + m[9]*m[2]*m[15] - m[9]*m[3]*m[14] - m[13]*m[2]*m[11] + m[13]*m[3]*m[10];
    inv[5]  =  m[0]*m[10]*m[15] - m[0]*m[11]*m[14] - m[8]*m[2]*m[15] + m[8]*m[3]*m[14] + m[12]*m[2]*m[11] - m[12]*m[3]*m[10];
    inv[9]  = -m[0]*m[9]*m[15]  + m[0]*m[11]*m[13] + m[8]*m[1]*m[15] - m[8]*m[3]*m[13] - m[12]*m[1]*m[11] + m[12]*m[3]*m[9];
    inv[13] =  m[0]*m[9]*m[14]  - m[0]*m[10]*m[13] - m[8]*m[1]*m[14] + m[8]*m[2]*m[13] + m[12]*m[1]*m[10] - m[12]*m[2]*m[9];
    inv[2]  =  m[1]*m[6]*m[15]  - m[1]*m[7]*m[14]  - m[5]*m[2]*m[15] + m[5]*m[3]*m[14] + m[13]*m[2]*m[7]  - m[13]*m[3]*m[6];
    inv[6]  = -m[0]*m[6]*m[15]  + m[0]*m[7]*m[14]  + m[4]*m[2]*m[15] - m[4]*m[3]*m[14] - m[12]*m[2]*m[7]  + m[12]*m[3]*m[6];
    inv[10] =  m[0]*m[5]*m[15]  - m[0]*m[7]*m[13]  - m[4]*m[1]*m[15] + m[4]*m[3]*m[13] + m[12]*m[1]*m[7]  - m[12]*m[3]*m[5];
    inv[14] = -m[0]*m[5]*m[14]  + m[0]*m[6]*m[13]  + m[4]*m[1]*m[14] - m[4]*m[2]*m[13] - m[12]*m[1]*m[6]  + m[12]*m[2]*m[5];
    inv[3]  = -m[1]*m[6]*m[11]  + m[1]*m[7]*m[10]  + m[5]*m[2]*m[11] - m[5]*m[3]*m[10] - m[9]*m[2]*m[7]   + m[9]*m[3]*m[6];
    inv[7]  =  m[0]*m[6]*m[11]  - m[0]*m[7]*m[10]  - m[4]*m[2]*m[11] + m[4]*m[3]*m[10] + m[8]*m[2]*m[7]   - m[8]*m[3]*m[6];
    inv[11] = -m[0]*m[5]*m[11]  + m[0]*m[7]*m[9]   + m[4]*m[1]*m[11] - m[4]*m[3]*m[9]  - m[8]*m[1]*m[7]   + m[8]*m[3]*m[5];
    inv[15] =  m[0]*m[5]*m[10]  - m[0]*m[6]*m[9]   - m[4]*m[1]*m[10] + m[4]*m[2]*m[9]  + m[8]*m[1]*m[6]   - m[8]*m[2]*m[5];

    float det = m[0]*inv[0] + m[1]*inv[4] + m[2]*inv[8] + m[3]*inv[12];
    if (det == 0.0f) return false;
    float invDet = 1.0f / det;
    for (int i = 0; i < 16; ++i) out[i] = inv[i] * invDet;
    return true;
}

void renderGridOverlay(JNIEnv* env, int width, int height,
                       jfloatArray transform) {
    if (g_gridEnabled.load() == 0) return;

    jfloat* m = env->GetFloatArrayElements(transform, nullptr);
    float invM[16];
    bool ok = invertMat4(m, invM);
    env->ReleaseFloatArrayElements(transform, m, JNI_ABORT);
    if (!ok) return;

    glViewport(0, 0, width, height);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(g_grid.program);
    glBindVertexArray(g_quadVao);

    int style = g_gridStyle.load();
    glUniformMatrix4fv(g_grid.uInverseTransform, 1, GL_FALSE, invM);
    glUniform1f(g_grid.uSpacing, g_gridSpacing.load());
    glUniform1f(g_grid.uSubdivisions, kGridSubdivisions);
    glUniform1f(g_grid.uMinorWidth,
                style == 2 ? kGridMinorDotWidth : kGridMinorLineWidth);
    glUniform1f(g_grid.uMajorWidth,
                style == 2 ? kGridMajorDotWidth : kGridMajorLineWidth);
    glUniform4fv(g_grid.uMinorColor, 1, kGridMinorColor);
    glUniform4fv(g_grid.uMajorColor, 1, kGridMajorColor);
    glUniform1i(g_grid.uStyle, style);

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);
}

// Page margin rules: a horizontal rule g_marginTop in from the top and
// bottom edges, and a vertical rule g_marginSide in from each side
// edge, all spanning the full page. Part of the page background like the grid — drawn
// under the layers so strokes occlude them, and included in exports.
// Requires active page bounds (the rules are inset from page edges);
// a margin of 0 draws nothing on that edge rather than sitting on the
// page outline.
void renderMarginRules(JNIEnv* env, int width, int height,
                       jfloatArray transform, const PageClip& pageClip) {
    if (g_marginEnabled.load() == 0 || !pageClip.active) return;
    const float top  = g_marginTop.load();
    const float side = g_marginSide.load();
    const float pageW = pageClip.maxX - pageClip.minX;
    const float pageH = pageClip.maxY - pageClip.minY;

    glViewport(0, 0, width, height);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(g_lineProg.program);
    glBindVertexArray(g_quadVao);
    uploadMat4(env, g_lineProg.uTransform, transform);
    glUniform2f(g_lineProg.uScreen, (float)width, (float)height);
    // uOpacity is per-layer state left behind by the vector compositor;
    // reset it like the page outline does or a faint/hidden vector layer
    // on the previous frame silently dims these.
    glUniform1f(g_lineProg.uOpacity, 1.0f);
    // Page-clipped: the rules live inside the page anyway, and the clip
    // trims their round caps flush with the page edge.
    uploadPageClip(g_lineProg.uPageMin, g_lineProg.uPageMax,
                   g_lineProg.uPageActive, pageClip);

    if (top > 0.0f && top * 2.0f < pageH) {
        const float yt = pageClip.minY + top;
        const float yb = pageClip.maxY - top;
        drawLineSegment(pageClip.minX, yt, pageClip.maxX, yt,
                        kMarginColor, kMarginWidthDocPx, kMarginAlpha);
        drawLineSegment(pageClip.minX, yb, pageClip.maxX, yb,
                        kMarginColor, kMarginWidthDocPx, kMarginAlpha);
    }
    if (side > 0.0f && side * 2.0f < pageW) {
        const float xl = pageClip.minX + side;
        const float xr = pageClip.maxX - side;
        drawLineSegment(xl, pageClip.minY, xl, pageClip.maxY,
                        kMarginColor, kMarginWidthDocPx, kMarginAlpha);
        drawLineSegment(xr, pageClip.minY, xr, pageClip.maxY,
                        kMarginColor, kMarginWidthDocPx, kMarginAlpha);
    }
    glBindVertexArray(0);
}

// Pixel-grid overlay — draws a 1-buffer-pixel-wide line at every integer
// doc-pixel boundary. Gated on the enable flag AND the current view scale;
// at low zoom the grid would be denser than the screen can show and just
// darkens the canvas. Drawn AFTER the layer composite so users can see the
// grid sitting on top of their pixels (vs the regular grid which renders
// underneath as a background).
void renderPixelGridOverlay(JNIEnv* env, int width, int height,
                            jfloatArray transform) {
    if (g_pixelGridEnabled.load() == 0) return;
    if (currentViewScale() < kPixelGridMinViewScale) return;

    jfloat* m = env->GetFloatArrayElements(transform, nullptr);
    float invM[16];
    bool ok = invertMat4(m, invM);
    env->ReleaseFloatArrayElements(transform, m, JNI_ABORT);
    if (!ok) return;

    glViewport(0, 0, width, height);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(g_pixelGridProg.program);
    glBindVertexArray(g_quadVao);
    glUniformMatrix4fv(g_pixelGridProg.uInverseTransform, 1, GL_FALSE, invM);
    // Dark gray at ~25% alpha — visible against both paper-white and
    // dark strokes, but subtle enough that the per-pixel lines don't
    // dominate the actual pixel content they overlay.
    float color[4] = { 0.15f, 0.15f, 0.15f, 0.25f };
    glUniform4fv(g_pixelGridProg.uColor, 1, color);
    PageClip pageClip = readPageClip();
    uploadPageClip(g_pixelGridProg.uPageMin, g_pixelGridProg.uPageMax,
                   g_pixelGridProg.uPageActive, pageClip);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);
}

// ---- Eraser-preview FBO helpers ------------------------------------------

void ensureViewFbo(ViewFbo& v, int width, int height,
                   bool withStencil = false) {
    if (v.texture != 0 && v.width == width && v.height == height
        && (!withStencil || v.stencil != 0)) {
        return;
    }

    if (v.fbo)     glDeleteFramebuffers(1, &v.fbo);
    if (v.texture) glDeleteTextures(1, &v.texture);
    if (v.stencil) glDeleteRenderbuffers(1, &v.stencil);
    v.fbo = 0; v.texture = 0; v.stencil = 0;

    glGenTextures(1, &v.texture);
    glBindTexture(GL_TEXTURE_2D, v.texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glGenFramebuffers(1, &v.fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, v.fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, v.texture, 0);
    if (withStencil) {
        glGenRenderbuffers(1, &v.stencil);
        glBindRenderbuffer(GL_RENDERBUFFER, v.stencil);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_STENCIL_INDEX8,
                              width, height);
        glBindRenderbuffer(GL_RENDERBUFFER, 0);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT,
                                  GL_RENDERBUFFER, v.stencil);
    }
    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        LOGE("view FBO incomplete: 0x%x (size %dx%d)", status, width, height);
    }

    v.width  = width;
    v.height = height;
}

// Composite the layers in [startIdx, endExclusive) into target FBO,
// dispatching by layer type so both raster tiles and vector shapes get
// rendered. If clearWhite is true the FBO is first cleared to opaque
// paper-white; otherwise to transparent. Result is premultiplied either
// way. Blends with GL_ONE / GL_ONE_MINUS_SRC_ALPHA throughout.
void renderLayerRangeIntoFbo(JNIEnv* env, ViewFbo& target,
                             size_t startIdx, size_t endExclusive,
                             int width, int height, jfloatArray transform,
                             bool clearWhite) {
    ensureViewFbo(target, width, height);
    glBindFramebuffer(GL_FRAMEBUFFER, target.fbo);
    glViewport(0, 0, width, height);
    if (clearWhite) {
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    } else {
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    }
    glClear(GL_COLOR_BUFFER_BIT);

    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

    // The "below" snapshot represents the page background + everything
    // below the active layer, mirroring what compositeAllLayers paints
    // before the active-layer pass. The grid belongs in that background
    // — without this, the eraser preview formula (display = lerp(multi,
    // above + below*(1-above.a), c)) hides the grid in erased regions
    // until commit, since `multi` has the grid but `below` didn't.
    if (clearWhite) {
        PageClip pageClip = readPageClip();
        glUseProgram(g_grid.program);
        uploadPageClip(g_grid.uPageMin, g_grid.uPageMax,
                       g_grid.uPageActive, pageClip);
        renderGridOverlay(env, width, height, transform);
        // Margins are background too — same eraser-preview reasoning.
        renderMarginRules(env, width, height, transform, pageClip);
    }

    if (startIdx >= endExclusive) return;

    bindRasterCompositePipeline(env, width, height, transform);

    for (size_t i = startIdx; i < endExclusive && i < layers().size(); ++i) {
        if (!layers()[i]) continue;
        if (!layers()[i]->visible.load(std::memory_order_relaxed)) continue;
        if (layers()[i]->type == LayerType::Raster) {
            compositeRasterLayer(*layers()[i]);
        } else { // Vector
            compositeVectorLayer(env, *layers()[i], i, width, height, transform);
            // compositeVectorLayer leaves the line program bound; restore
            // the raster pipeline for the next iteration.
            bindRasterCompositePipeline(env, width, height, transform);
        }
    }

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindVertexArray(0);
}

// Populate the scratch buffers used by the live-preview composite. Both
// the brush and the eraser paths share the same plumbing — only the
// preview shader's mode + uniforms differ. The active layer is
// intentionally omitted: compositing is linear in any single layer's
// contribution, so the framework's alpha-blend recovers the true
// disp(c) for free as a lerp between multi and fullColor.
void preparePreviewBuffers(JNIEnv* env, int width, int height,
                           jfloatArray transform) {
    // below: paper + layers strictly below active (paper-backed, alpha=1)
    renderLayerRangeIntoFbo(env, g_belowFbo, 0, g_strokeTarget,
                            width, height, transform, /*clearWhite=*/true);
    // above: layers strictly above active (premultiplied over transparent;
    // empty range when active is the top layer is handled by early-return)
    renderLayerRangeIntoFbo(env, g_aboveFbo, g_strokeTarget + 1, layers().size(),
                            width, height, transform, /*clearWhite=*/false);

    // Reset coverage to zero.
    ensureViewFbo(g_coverage, width, height);
    glBindFramebuffer(GL_FRAMEBUFFER, g_coverage.fbo);
    glViewport(0, 0, width, height);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    // Mirror coverage. Same dimensions; cleared so the first real dab's
    // restore (if a prediction sneaks in before any real sample, which
    // shouldn't happen but defensively) reads zero.
    ensureViewFbo(g_coverageReal, width, height);
    glBindFramebuffer(GL_FRAMEBUFFER, g_coverageReal.fbo);
    glViewport(0, 0, width, height);
    glClear(GL_COLOR_BUFFER_BIT);
    // Shade tool: merged (dabs ∪ fill) coverage. Per-batch rebuilds are
    // scissored to the shaded region, so the one full-surface clear has
    // to happen here — it's what makes the untouched remainder of the
    // texture reliably zero for the whole stroke.
    if (g_strokeFillClosed) {
        ensureViewFbo(g_shadeCoverage, width, height, /*withStencil=*/true);
        glBindFramebuffer(GL_FRAMEBUFFER, g_shadeCoverage.fbo);
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT);
    }
    g_predictionInFlight = false;
    g_liveEmitterReal = g_liveEmitter;   // both freshly reset()
}

// ---- Tile management ------------------------------------------------------

Tile& getOrCreateTile(Layer& layer, int tx, int ty,
                      const uint8_t* initial = nullptr) {
    int64_t k = tileKey(tx, ty);
    auto it = layer.tiles.find(k);
    if (it != layer.tiles.end()) return it->second;

    Tile t;
    glGenTextures(1, &t.texture);
    glBindTexture(GL_TEXTURE_2D, t.texture);
    // 258×258 storage; interior 256×256 sits at texels [kApron..kApron+255]
    // and the outer ring is the apron filled lazily from neighbors.
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, kTileTexSize, kTileTexSize, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glGenFramebuffers(1, &t.fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, t.fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, t.texture, 0);
    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        LOGE("tile (%d, %d) FBO incomplete: 0x%x", tx, ty, status);
    }

    // Always clear the full 258×258 to transparent first (covers both
    // the interior on the no-initial path AND the apron in either case).
    glViewport(0, 0, kTileTexSize, kTileTexSize);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);   // transparent — premultiplied
    glClear(GL_COLOR_BUFFER_BIT);
    if (initial) {
        // Upload the 256×256 of interior bytes into texels [kApron..kApron+255].
        glBindTexture(GL_TEXTURE_2D, t.texture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, kApron, kApron,
                        kTileSize, kTileSize,
                        GL_RGBA, GL_UNSIGNED_BYTE, initial);
        // Mirror the upload into the BEFORE-snapshot cache. Shared
        // refcounted buffer (from the pool), see TileBytes.
        t.cachedBytes = acquireTileBytesFrom(initial);
    } else {
        // Freshly cleared (interior is all zero). Allocate the cache
        // to a zeroed buffer — same memory cost as if it had been
        // populated and the eventual mutation will replace it. Pool
        // returns a buffer with undefined contents, so zero-fill.
        t.cachedBytes = acquireZeroedTileBytes();
    }

    layer.tiles[k] = t;
    return layer.tiles[k];
}

// Mark this tile's apron stale, plus all 8 surrounding tiles' aprons
// (neighbors' aprons hold copies of this tile's edge data, so a change
// here invalidates them too). Cheap — flag flips, no GL work.
void markApronStaleAround(Layer& layer, int tx, int ty) {
    static const int dx[9] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
    static const int dy[9] = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };
    for (int i = 0; i < 9; ++i) {
        auto it = layer.tiles.find(tileKey(tx + dx[i], ty + dy[i]));
        if (it != layer.tiles.end()) it->second.apronStale = true;
    }
}

// Pull edge data from existing neighbors into this tile's apron. No-op
// for sides where there is no neighbor — the apron is cleared to
// transparent, which is the right behavior at the canvas edge: LINEAR
// composite samples blend the interior edge texel with transparent,
// fading the stroke out smoothly rather than producing a hard step.
//
// Uses glBlitFramebuffer (and a scissored glClear for missing neighbors)
// rather than glCopyTexSubImage2D / glTexSubImage2D. The CopyTex path
// forces a tile-memory flush on TBDR mobile GPUs (Adreno/Mali) — each
// call costs hundreds of µs regardless of pixel count, and a single
// commit can issue tens of these (9 stale tiles × 8 neighbors). Blits
// go through dedicated hardware and don't pay that cost. Measured drop
// from ~10 ms to <1 ms on the MovinkPad's flash repro.
void syncTileApron(Layer& layer, int tx, int ty) {
    auto it = layer.tiles.find(tileKey(tx, ty));
    if (it == layer.tiles.end()) return;
    Tile& t = it->second;
    if (!t.apronStale) return;

    // Save state we may clobber. The caller may have:
    //   - Both FBO bindings set (the partial-recomposite path binds
    //     MB.back as draw and may have a neighbor as read).
    //   - Scissor enabled (partial-recomposite confines composite to
    //     the dirty bbox in MB.back coords — meaningless for tile FBOs).
    GLint prevDrawFbo = 0;
    GLint prevReadFbo = 0;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &prevDrawFbo);
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &prevReadFbo);
    GLboolean scissorWas = glIsEnabled(GL_SCISSOR_TEST);
    GLint scissorBox[4];
    glGetIntegerv(GL_SCISSOR_BOX, scissorBox);

    // Bind dest tile's FBO as the draw target for blits and clears.
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, t.fbo);
    glDisable(GL_SCISSOR_TEST);

    auto copyFromNeighbor = [&](int nx, int ny,
                                int srcX, int srcY,
                                int dstX, int dstY,
                                int w,    int h) {
        auto nIt = layer.tiles.find(tileKey(nx, ny));
        if (nIt != layer.tiles.end()) {
            glBindFramebuffer(GL_READ_FRAMEBUFFER, nIt->second.fbo);
            glBlitFramebuffer(srcX, srcY, srcX + w, srcY + h,
                              dstX, dstY, dstX + w, dstY + h,
                              GL_COLOR_BUFFER_BIT, GL_NEAREST);
        } else {
            // No neighbor — clear that apron strip to transparent so a
            // recently-deleted neighbor's stale data doesn't linger.
            glEnable(GL_SCISSOR_TEST);
            glScissor(dstX, dstY, w, h);
            glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            glDisable(GL_SCISSOR_TEST);
        }
    };

    // Sides — `kApron` is the offset of "interior origin", and
    // `kApron+kTileSize-1` (== kTileSize) is the rightmost / bottom
    // interior texel. Neighbors copy their opposite-side edge column
    // (or row) into our matching apron strip.
    const int last = kApron + kTileSize - 1;            // rightmost / bottom interior texel
    const int aprL = 0;                                 // left apron column
    const int aprR = kTileTexSize - 1;                  // right apron column
    const int aprT = 0;                                 // top apron row
    const int aprB = kTileTexSize - 1;                  // bottom apron row

    // Left neighbor: its rightmost interior column → our left apron column.
    copyFromNeighbor(tx - 1, ty, /*src*/ last, kApron,
                     /*dst*/ aprL, kApron,
                     /*w*/ 1, /*h*/ kTileSize);
    // Right neighbor: its leftmost interior column → our right apron column.
    copyFromNeighbor(tx + 1, ty, /*src*/ kApron, kApron,
                     /*dst*/ aprR, kApron,
                     /*w*/ 1, /*h*/ kTileSize);
    // Top neighbor: its bottom interior row → our top apron row.
    copyFromNeighbor(tx, ty - 1, /*src*/ kApron, last,
                     /*dst*/ kApron, aprT,
                     /*w*/ kTileSize, /*h*/ 1);
    // Bottom neighbor: its top interior row → our bottom apron row.
    copyFromNeighbor(tx, ty + 1, /*src*/ kApron, kApron,
                     /*dst*/ kApron, aprB,
                     /*w*/ kTileSize, /*h*/ 1);

    // Diagonal corners: each is a single 1×1 pixel from the
    // diagonally-adjacent tile's far interior corner.
    copyFromNeighbor(tx - 1, ty - 1, last,   last,   aprL, aprT, 1, 1);
    copyFromNeighbor(tx + 1, ty - 1, kApron, last,   aprR, aprT, 1, 1);
    copyFromNeighbor(tx - 1, ty + 1, last,   kApron, aprL, aprB, 1, 1);
    copyFromNeighbor(tx + 1, ty + 1, kApron, kApron, aprR, aprB, 1, 1);

    // Restore scissor + FBO bindings.
    glScissor(scissorBox[0], scissorBox[1], scissorBox[2], scissorBox[3]);
    if (scissorWas) glEnable(GL_SCISSOR_TEST);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, prevReadFbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, prevDrawFbo);
    t.apronStale = false;
}

// Snapshot every tile in the inclusive bbox (tx0..tx1, ty0..ty1) of the
// given layer into `out`. Tiles that don't currently exist are recorded
// with `existed=false` and zero bytes (so undo of a stroke that created
// new tiles deletes them rather than leaving zero-alpha leftovers). Tile
// pixels are read via the tile's FBO with glReadPixels.
void snapshotTilesInBbox(size_t layerIdx, int tx0, int tx1, int ty0, int ty1,
                         std::vector<TileSnap>& out) {
    out.clear();
    if (layerIdx >= layers().size() || !layers()[layerIdx]) return;
    Layer& layer = *layers()[layerIdx];
    if (layer.type != LayerType::Raster) return;
    GLint prevFbo = 0;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &prevFbo);
    for (int ty = ty0; ty <= ty1; ++ty) {
        for (int tx = tx0; tx <= tx1; ++tx) {
            TileSnap snap;
            snap.tx = tx; snap.ty = ty;
            auto it = layer.tiles.find(tileKey(tx, ty));
            if (it != layer.tiles.end()) {
                snap.existed = true;
                if (it->second.cachedBytes) {
                    // Fast path: share the cache's buffer by refcount.
                    // No allocation, no copy.
                    snap.bytes = it->second.cachedBytes;
                } else {
                    // Defensive fallback — cache should be populated
                    // post-load and after every mutation. Allocate
                    // fresh, read back, and seed the cache.
                    auto fresh = tilePool().acquire();
                    glBindFramebuffer(GL_FRAMEBUFFER, it->second.fbo);
                    glReadPixels(kApron, kApron, kTileSize, kTileSize,
                                 GL_RGBA, GL_UNSIGNED_BYTE,
                                 fresh->data());
                    it->second.cachedBytes = fresh;
                    snap.bytes = fresh;
                }
            }
            out.push_back(std::move(snap));
        }
    }
    glBindFramebuffer(GL_FRAMEBUFFER, prevFbo);
}

// Capture a complete layer snapshot — type, name, visibility, opacity,
// plus content (raster tiles or vector shapes per the type). Used by
// destructive layer ops that need to re-create the layer on undo.
// Reads tile FBOs directly, so it must run on the GL thread.
void captureLayerSnapshot(size_t layerIdx, LayerSnapshot& out) {
    out = LayerSnapshot{};
    if (layerIdx >= layers().size() || !layers()[layerIdx]) return;
    Layer& layer = *layers()[layerIdx];
    {
        std::lock_guard<std::mutex> lock(g_layerNameMutex);
        out.name = layer.name;
    }
    out.type    = layer.type;
    out.visible = layer.visible.load(std::memory_order_relaxed);
    out.opacity = layer.opacity.load(std::memory_order_relaxed);
    if (layer.type == LayerType::Raster) {
        snapshotAllTiles(layerIdx, out.tiles);
    } else {
        out.lines    = layer.lines;
        out.rects    = layer.rects;
        out.ellipses = layer.ellipses;
        out.circles  = layer.circles;
    }
}

// Insert a fresh Layer at [idx] populated from `snap`, shifting any
// trailing layers (and their on-disk dirs) up by one. Restores tiles
// or vector shapes per the snapshot's type, plus name/visibility/
// opacity to disk so the layer survives the next launch. Must run on
// the GL thread (creates tile textures).
void insertLayerWithSnapshot(size_t idx, const LayerSnapshot& snap) {
    auto& ls = layers();
    if (idx > ls.size()) idx = ls.size();

    // Renumber trailing layer dirs UP by one — top-down to avoid
    // clobbering. Mirror image of the renumber-after-delete path in
    // deleteLayerImpl.
    if (!g_docDir.empty()) {
        std::string pageDir = pageDirOf(g_activePageIdx);
        for (long long j = static_cast<long long>(ls.size()) - 1;
             j >= static_cast<long long>(idx); --j) {
            std::string oldDir = pageDir + "/layer_" + std::to_string(j);
            std::string newDir = pageDir + "/layer_" + std::to_string(j + 1);
            struct stat st;
            if (stat(oldDir.c_str(), &st) == 0) {
                rename(oldDir.c_str(), newDir.c_str());
            }
        }
    }

    auto layer = std::make_unique<Layer>();
    layer->type = snap.type;
    layer->visible.store(snap.visible, std::memory_order_relaxed);
    layer->opacity.store(snap.opacity, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(g_layerNameMutex);
        layer->name = snap.name;
    }
    layer->lines    = snap.lines;
    layer->rects    = snap.rects;
    layer->ellipses = snap.ellipses;
    layer->circles  = snap.circles;
    ls.insert(ls.begin() + idx, std::move(layer));

    // Restore raster tiles (creates GL textures + saves to disk).
    if (snap.type == LayerType::Raster) {
        for (const auto& t : snap.tiles) {
            applyTileSnap(idx, t);
        }
    }
    // Persist vector shapes (also creates the marker file used to
    // detect Vector type at load time).
    if (snap.type == LayerType::Vector) {
        saveVectorLayer(idx, *ls[idx]);
    }

    // Persist metadata files. mkdir is no-op if the dir already exists
    // (e.g. raster-layer tile saves above already created it).
    if (!g_docDir.empty()) {
        std::string dir = activeLayerDir(idx);
        mkdir(dir.c_str(), 0755);
        std::string namePath = dir + "/name.txt";
        if (snap.name.empty()) {
            std::remove(namePath.c_str());
        } else if (FILE* f = std::fopen(namePath.c_str(), "wb")) {
            std::fwrite(snap.name.data(), 1, snap.name.size(), f);
            std::fclose(f);
        }
        std::string flagPath = dir + "/hidden.flag";
        if (snap.visible) {
            std::remove(flagPath.c_str());
        } else if (FILE* f = std::fopen(flagPath.c_str(), "wb")) {
            std::fclose(f);
        }
        std::string opath = dir + "/opacity.txt";
        if (snap.opacity >= 0.999f) {
            std::remove(opath.c_str());
        } else if (FILE* f = std::fopen(opath.c_str(), "wb")) {
            std::fprintf(f, "%.4f", snap.opacity);
            std::fclose(f);
        }
    }
}

// Snapshot every existing tile in the layer (for full-layer-clear undo).
void snapshotAllTiles(size_t layerIdx, std::vector<TileSnap>& out) {
    out.clear();
    if (layerIdx >= layers().size() || !layers()[layerIdx]) return;
    Layer& layer = *layers()[layerIdx];
    if (layer.type != LayerType::Raster) return;
    GLint prevFbo = 0;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &prevFbo);
    out.reserve(layer.tiles.size());
    for (auto& kv : layer.tiles) {
        TileSnap snap;
        unpackTileKey(kv.first, snap.tx, snap.ty);
        snap.existed = true;
        if (kv.second.cachedBytes) {
            snap.bytes = kv.second.cachedBytes;
        } else {
            auto fresh = tilePool().acquire();
            glBindFramebuffer(GL_FRAMEBUFFER, kv.second.fbo);
            glReadPixels(kApron, kApron, kTileSize, kTileSize,
                         GL_RGBA, GL_UNSIGNED_BYTE, fresh->data());
            kv.second.cachedBytes = fresh;
            snap.bytes = fresh;
        }
        out.push_back(std::move(snap));
    }
    glBindFramebuffer(GL_FRAMEBUFFER, prevFbo);
}

// Upload bytes into a tile's texture (creating it if necessary), then
// persist to disk. Used by undo/redo to restore tile state.
//
// Save+restore the FBO binding so this is safe to call from inside the
// pending-action drain at the start of compositeAllLayers, where the
// caller's bound FBO is the multi-buffer and any leak would cause the
// subsequent clear/grid/composite to write into a tile FBO instead.
// (getOrCreateTile binds the new tile's FBO during its clear path.)
void uploadTileBytesAndSave(size_t layerIdx, int tx, int ty,
                            const uint8_t* bytes) {
    if (layerIdx >= layers().size() || !layers()[layerIdx]) return;
    Layer& layer = *layers()[layerIdx];
    GLint prevFbo = 0;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &prevFbo);
    Tile& tile = getOrCreateTile(layer, tx, ty);
    glBindTexture(GL_TEXTURE_2D, tile.texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, kApron, kApron, kTileSize, kTileSize,
                    GL_RGBA, GL_UNSIGNED_BYTE, bytes);
    glBindTexture(GL_TEXTURE_2D, 0);
    // Mirror into the BEFORE-snapshot cache. Caller's `bytes` is the
    // authoritative new content for the tile.
    tile.cachedBytes = acquireTileBytesFrom(bytes);
    saveTileToDisk(layerIdx, tileKey(tx, ty));
    // Tile content changed — its 8 neighbors hold copies of its old
    // edge data in their aprons, so flag them (and self) for resync
    // before next composite.
    markApronStaleAround(layer, tx, ty);
    glBindFramebuffer(GL_FRAMEBUFFER, prevFbo);
}

// ---- Async tile-disk writer ---------------------------------------------
//
// commitStroke used to do fwrite+fclose+rename synchronously on the GL
// thread for every tile a stroke touched. With strokes covering many
// tiles, disk I/O latency caused commit to overrun the vsync budget
// and the GLFrontBufferedRenderer's transition occasionally presented
// a frame where the front buffer was hidden but the multi-buffer
// hadn't yet been updated — visible as a brief white flash on commit.
//
// The fix: a single-threaded FIFO writer. Tile writes (and the unlink
// done by deleteTileIfExists) are queued from the GL thread and
// drained on a background thread. Single consumer keeps ordering for
// the same tile path so a later write can't be clobbered by an
// earlier-queued one. Defined here (rather than next to
// writeTileBytesToDisk) so deleteTileIfExists, which constructs a
// DiskTask by value, can see the type.
struct DiskTask {
    enum Op { kWrite, kDelete };
    Op op;
    std::string path;       // final path
    std::string tmpPath;    // write target (renamed atomically); unused for kDelete
    std::vector<uint8_t> bytes;  // owned; copied at enqueue time
};

std::deque<DiskTask>     g_diskQueue;
std::mutex               g_diskQueueMutex;
std::condition_variable  g_diskQueueCv;
std::thread              g_diskWriter;
std::atomic<bool>        g_diskWriterStarted{false};
std::atomic<bool>        g_diskWriterStop{false};

static void diskWriterLoop() {
    for (;;) {
        DiskTask task;
        {
            std::unique_lock<std::mutex> lock(g_diskQueueMutex);
            g_diskQueueCv.wait(lock, [] {
                return !g_diskQueue.empty() || g_diskWriterStop.load();
            });
            if (g_diskQueue.empty() && g_diskWriterStop.load()) return;
            task = std::move(g_diskQueue.front());
            g_diskQueue.pop_front();
        }
        switch (task.op) {
            case DiskTask::kWrite: {
                FILE* f = fopen(task.tmpPath.c_str(), "wb");
                if (!f) {
                    LOGE("disk writer: fopen %s failed (errno=%d)",
                         task.tmpPath.c_str(), errno);
                    break;
                }
                size_t written = fwrite(task.bytes.data(), 1,
                                        task.bytes.size(), f);
                fclose(f);
                if (written != task.bytes.size()) {
                    LOGE("disk writer: short write %zu to %s",
                         written, task.tmpPath.c_str());
                    unlink(task.tmpPath.c_str());
                    break;
                }
                if (rename(task.tmpPath.c_str(), task.path.c_str()) != 0) {
                    LOGE("disk writer: rename %s → %s failed (errno=%d)",
                         task.tmpPath.c_str(), task.path.c_str(), errno);
                }
                break;
            }
            case DiskTask::kDelete:
                unlink(task.path.c_str());
                break;
        }
    }
}

static void ensureDiskWriterStarted() {
    bool expected = false;
    if (g_diskWriterStarted.compare_exchange_strong(expected, true)) {
        g_diskWriter = std::thread(diskWriterLoop);
        // Detached so static destruction at process exit (rare on
        // Android — usually SIGKILL — but possible) doesn't call
        // std::terminate on a still-joinable thread.
        g_diskWriter.detach();
    }
}

static void enqueueDiskTask(DiskTask&& task) {
    ensureDiskWriterStarted();
    {
        std::lock_guard<std::mutex> lock(g_diskQueueMutex);
        g_diskQueue.push_back(std::move(task));
    }
    g_diskQueueCv.notify_one();
}

// Block until every queued disk task has been drained. Called at doc
// boundaries (switch / close / app pause) so the user can't navigate
// away with pending writes in memory.
static void flushDiskWriter() {
    if (!g_diskWriterStarted.load()) return;
    for (;;) {
        {
            std::lock_guard<std::mutex> lock(g_diskQueueMutex);
            if (g_diskQueue.empty()) return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

// Drop a tile's GL resources, remove it from the layer's map, and unlink
// its on-disk file. No-op if the tile isn't present.
void deleteTileIfExists(size_t layerIdx, int tx, int ty) {
    if (layerIdx >= layers().size() || !layers()[layerIdx]) return;
    Layer& layer = *layers()[layerIdx];
    int64_t k = tileKey(tx, ty);
    auto it = layer.tiles.find(k);
    auto queueUnlink = [&](void) {
        if (g_docDir.empty()) return;
        DiskTask task;
        task.op = DiskTask::kDelete;
        task.path = activeLayerDir(layerIdx)
                  + "/tile_" + std::to_string(tx)
                  + "_"      + std::to_string(ty) + ".bin";
        // FIFO single-consumer queue keeps this in order with any
        // earlier-queued writes for the same tile path.
        enqueueDiskTask(std::move(task));
    };
    if (it == layer.tiles.end()) {
        // Even if no GPU tile, an orphan disk file from a partially-saved
        // state should still be cleaned up; harmless if absent.
        queueUnlink();
        return;
    }
    if (it->second.fbo)     glDeleteFramebuffers(1, &it->second.fbo);
    if (it->second.texture) glDeleteTextures(1, &it->second.texture);
    layer.tiles.erase(it);
    // Neighbors' aprons hold copies of this tile's edge data. Mark
    // them stale so the next composite pulls in zero (no-neighbor)
    // for those sides instead of leaving the deleted content behind.
    markApronStaleAround(layer, tx, ty);
    queueUnlink();
}

// Apply one tile snapshot (existed=true → upload bytes; false → delete).
void applyTileSnap(size_t layerIdx, const TileSnap& snap) {
    if (snap.existed && snap.bytes) {
        uploadTileBytesAndSave(layerIdx, snap.tx, snap.ty, snap.bytes->data());
    } else {
        deleteTileIfExists(layerIdx, snap.tx, snap.ty);
    }
}

// Compute the tile-space bbox a stroke's samples will touch, the same
// way bakeCurrentStrokeIntoTiles does. Returns false if there are no
// samples (caller should skip snapshotting).
bool currentStrokeTileBbox(int& tx0, int& tx1, int& ty0, int& ty1) {
    if (g_current.samples.empty()) return false;
    // Match bakeCurrentStrokeIntoTiles — pad by the actual maximum dab
    // radius (scales with the stroke's brush-size snapshot), not the
    // unscaled kMaxRadius constant. Otherwise undo snapshotting misses
    // tiles for large brushes and an undo can leave stale pixels.
    float pad = kMaxRadius * g_strokeBrushSizeScale;
    float minX = g_current.samples.front().x, maxX = minX;
    float minY = g_current.samples.front().y, maxY = minY;
    for (const auto& s : g_current.samples) {
        if (s.x < minX) minX = s.x;
        if (s.x > maxX) maxX = s.x;
        if (s.y < minY) minY = s.y;
        if (s.y > maxY) maxY = s.y;
    }
    minX -= pad; maxX += pad;
    minY -= pad; maxY += pad;
    tx0 = static_cast<int>(std::floor(minX / kTileSizeF));
    tx1 = static_cast<int>(std::floor(maxX / kTileSizeF));
    ty0 = static_cast<int>(std::floor(minY / kTileSizeF));
    ty1 = static_cast<int>(std::floor(maxY / kTileSizeF));
    return true;
}

// Wipe the on-disk dir for a layer (used when undoing a layer add to
// remove stranded shapes.bin / tile files).
void deleteLayerDirIfExists(size_t layerIdx) {
    if (g_docDir.empty()) return;
    std::string layerDir = activeLayerDir(layerIdx);
    DIR* d = opendir(layerDir.c_str());
    if (!d) return;
    struct dirent* e;
    while ((e = readdir(d)) != nullptr) {
        const char* n = e->d_name;
        if (n[0] == '.') continue;
        std::string p = layerDir + "/" + n;
        unlink(p.c_str());
    }
    closedir(d);
    rmdir(layerDir.c_str());
}

// Free GL resources owned by a layer's raster tiles. Vector layers have
// no GPU state, so this is a no-op for them. Caller is responsible for
// removing the Layer from the page's vector afterward.
void freeLayerGLResources(Layer& layer) {
    for (auto& kv : layer.tiles) {
        if (kv.second.fbo)     glDeleteFramebuffers(1, &kv.second.fbo);
        if (kv.second.texture) glDeleteTextures(1, &kv.second.texture);
    }
    layer.tiles.clear();
}

// Index-permutation helpers used to rewrite indices in undo entries and
// the selection after a structural change. Keeping these as small free
// functions makes the remap logic in delete/move impls a one-liner per
// reference site.
//
// After a move (from -> to):
//   - the moved layer ends up at `to`;
//   - every other index in (from, to] (or [to, from) for upward moves)
//     shifts by one in the opposite direction.
size_t mapLayerIdxAfterMove(size_t i, size_t from, size_t to) {
    if (i == from) return to;
    if (from < to) {
        if (i > from && i <= to) return i - 1;
    } else {
        if (i >= to && i < from) return i + 1;
    }
    return i;
}

// After a delete: returns SIZE_MAX for "this index is the deleted layer
// itself" (caller should drop the entry). Indices above the deleted slot
// shift down by one.
constexpr size_t kInvalidLayerIdx = SIZE_MAX;
size_t mapLayerIdxAfterDelete(size_t i, size_t deletedIdx) {
    if (i == deletedIdx) return kInvalidLayerIdx;
    if (i > deletedIdx) return i - 1;
    return i;
}

// Walks an UndoEntry (and any MergeLayerDown's nested entries) applying
// the post-delete index remap. Returns false when the entry should be
// dropped (its primary or target layer was the deleted slot, or any
// nested entry that survives is impossible).
bool remapEntryAfterDelete(UndoEntry& e, size_t deletedIdx) {
    size_t newIdx = mapLayerIdxAfterDelete(e.layerIdx, deletedIdx);
    if (newIdx == kInvalidLayerIdx) return false;
    if (e.op == UndoOp::RasterizeShapeBelow
     || e.op == UndoOp::MergeLayerDown) {
        size_t newTgt = mapLayerIdxAfterDelete(e.targetLayerIdx, deletedIdx);
        if (newTgt == kInvalidLayerIdx) return false;
        e.targetLayerIdx = newTgt;
    }
    e.layerIdx = newIdx;
    if (e.op == UndoOp::LayerAdd) {
        size_t prev = mapLayerIdxAfterDelete(e.prevActiveLayer, deletedIdx);
        e.prevActiveLayer = (prev == kInvalidLayerIdx) ? 0 : prev;
    }
    if (e.op == UndoOp::MergeLayerDown) {
        for (auto it = e.srcLayerUndoEntries.begin();
             it != e.srcLayerUndoEntries.end(); ) {
            if (!remapEntryAfterDelete(*it, deletedIdx)) {
                it = e.srcLayerUndoEntries.erase(it);
            } else {
                ++it;
            }
        }
    }
    if (e.op == UndoOp::VectorMutateGroup) {
        // Drop any group members on the deleted layer; shift the rest.
        size_t w = 0;
        for (size_t r = 0; r < e.mutateGroupSels.size(); ++r) {
            size_t mapped = mapLayerIdxAfterDelete(
                e.mutateGroupSels[r].layerIdx, deletedIdx);
            if (mapped == kInvalidLayerIdx) continue;
            e.mutateGroupSels[r].layerIdx = mapped;
            if (w != r) {
                e.mutateGroupSels[w]   = std::move(e.mutateGroupSels[r]);
                e.mutateGroupBefore[w] = std::move(e.mutateGroupBefore[r]);
                e.mutateGroupAfter[w]  = std::move(e.mutateGroupAfter[r]);
            }
            ++w;
        }
        e.mutateGroupSels.resize(w);
        e.mutateGroupBefore.resize(w);
        e.mutateGroupAfter.resize(w);
        if (w == 0) return false;   // nothing left to mutate
    }
    return true;
}

// Walks an UndoEntry applying the post-move index remap. No drop case —
// move is order-preserving, never invalidates an entry.
void remapEntryAfterMove(UndoEntry& e, size_t from, size_t to) {
    e.layerIdx = mapLayerIdxAfterMove(e.layerIdx, from, to);
    if (e.op == UndoOp::LayerAdd) {
        e.prevActiveLayer =
            mapLayerIdxAfterMove(e.prevActiveLayer, from, to);
    }
    if (e.op == UndoOp::RasterizeShapeBelow
     || e.op == UndoOp::MergeLayerDown) {
        e.targetLayerIdx =
            mapLayerIdxAfterMove(e.targetLayerIdx, from, to);
    }
    if (e.op == UndoOp::MergeLayerDown) {
        for (auto& nested : e.srcLayerUndoEntries) {
            remapEntryAfterMove(nested, from, to);
        }
    }
    if (e.op == UndoOp::VectorMutateGroup) {
        for (auto& s : e.mutateGroupSels) {
            s.layerIdx = mapLayerIdxAfterMove(s.layerIdx, from, to);
        }
    }
}

// Delete the layer at `idx` on the active page. No-op if there's only one
// layer left (the document needs at least one). Frees GL resources, wipes
// the on-disk dir, renumbers the trailing layer dirs to keep them
// 0..N-1 contiguous, and adjusts the active-layer index.
//
// Undo / redo entries that reference the deleted layer are dropped (their
// data is gone, so they can't apply); entries with a higher layerIdx are
// decremented. The vector selection is remapped or cleared the same way.
// The floating raster selection is cancel-restored — preserving it across
// a structural change isn't worth the complexity.
void deleteLayerImpl(size_t idx) {
    if (g_pages.empty() || g_activePageIdx >= g_pages.size()) return;
    auto& ls = g_pages[g_activePageIdx]->layers;
    if (idx >= ls.size() || !ls[idx]) return;
    if (ls.size() <= 1) {
        LOGI("delete layer %zu refused — last layer", idx);
        return;
    }

    cancelRasterSelectionImpl();

    // Remap undo / redo: drop entries pinned to the deleted layer, shift
    // the rest. Note: `bytes` is per-entry and totals must be kept in
    // sync as we erase.
    {
        std::lock_guard<std::mutex> lock(g_undoMutex);
        auto remap = [&](std::deque<UndoEntry>& stack, size_t& totalBytes) {
            for (auto it = stack.begin(); it != stack.end(); ) {
                if (!remapEntryAfterDelete(*it, idx)) {
                    totalBytes -= it->bytes;
                    it = stack.erase(it);
                } else {
                    // Bytes may have changed (nested entries dropped);
                    // re-account for it.
                    totalBytes -= it->bytes;
                    it->bytes = computeEntrySize(*it);
                    totalBytes += it->bytes;
                    ++it;
                }
            }
        };
        remap(g_undoStack, g_undoTotalBytes);
        remap(g_redoStack, g_redoTotalBytes);
    }

    // Vector selection: remap or drop.
    {
        std::lock_guard<std::mutex> lock(g_selectionMutex);
        if (g_selection.kind != ShapeKind::None) {
            size_t newSel = mapLayerIdxAfterDelete(g_selection.layerIdx, idx);
            if (newSel == kInvalidLayerIdx) g_selection = Selection{};
            else                            g_selection.layerIdx = newSel;
        }
        // Same remap for multi-select extras: drop those on the
        // deleted layer; shift the rest down by one.
        for (auto it = g_extraSelections.begin();
             it != g_extraSelections.end(); ) {
            size_t newSel = mapLayerIdxAfterDelete(it->layerIdx, idx);
            if (newSel == kInvalidLayerIdx) {
                it = g_extraSelections.erase(it);
            } else {
                it->layerIdx = newSel;
                ++it;
            }
        }
    }

    freeLayerGLResources(*ls[idx]);
    deleteLayerDirIfExists(idx);

    ls.erase(ls.begin() + static_cast<ptrdiff_t>(idx));

    // Renumber trailing layer dirs so they stay 0..N-1. Each rename moves
    // every file under that dir, so the cost is O(remaining-layers * dir
    // contents) — fine for the layer counts we care about (<20).
    if (!g_docDir.empty()) {
        std::string pageDir = pageDirOf(g_activePageIdx);
        for (size_t j = idx; j < ls.size(); ++j) {
            std::string oldDir = pageDir + "/layer_" + std::to_string(j + 1);
            std::string newDir = pageDir + "/layer_" + std::to_string(j);
            struct stat st;
            if (stat(oldDir.c_str(), &st) == 0) {
                rename(oldDir.c_str(), newDir.c_str());
            }
        }
    }

    size_t& active = g_pages[g_activePageIdx]->activeLayer;
    if (active == idx) {
        // Prefer the layer that took the deleted index's spot; fall back
        // to the new last layer if we deleted the top.
        active = std::min(idx, ls.size() - 1);
    } else if (active > idx) {
        --active;
    }

    LOGI("layer %zu deleted (count=%zu, active=%zu)",
         idx, ls.size(), active);
}

// Move the layer at `from` to position `to` on the active page. Both the
// in-memory vector and on-disk layer_<n> dirs are reordered; the layer's
// content (tiles / shapes) is preserved unchanged.
//
// Undo / redo entries are kept — their layerIdx (and prevActiveLayer for
// LayerAdd entries) is rewritten through mapLayerIdxAfterMove, so a
// post-reorder undo still hits the correct (now-relocated) layer. Same
// remap is applied to the vector selection. The floating raster
// selection is cancel-restored before the move (its lifted pixels go
// back to the source layer object, which is what the user intends).
void moveLayerImpl(size_t from, size_t to) {
    if (g_pages.empty() || g_activePageIdx >= g_pages.size()) return;
    auto& ls = g_pages[g_activePageIdx]->layers;
    if (from >= ls.size() || to >= ls.size() || from == to) return;
    if (!ls[from]) return;

    cancelRasterSelectionImpl();

    // Remap undo / redo entries' layer indices to the post-move slots.
    {
        std::lock_guard<std::mutex> lock(g_undoMutex);
        auto remap = [&](std::deque<UndoEntry>& stack) {
            for (auto& e : stack) {
                remapEntryAfterMove(e, from, to);
            }
        };
        remap(g_undoStack);
        remap(g_redoStack);
    }
    {
        std::lock_guard<std::mutex> lock(g_selectionMutex);
        if (g_selection.kind != ShapeKind::None) {
            g_selection.layerIdx =
                mapLayerIdxAfterMove(g_selection.layerIdx, from, to);
        }
        for (auto& s : g_extraSelections) {
            s.layerIdx = mapLayerIdxAfterMove(s.layerIdx, from, to);
        }
    }

    // Reorder in-memory: erase + insert keeps the unique_ptrs intact so
    // GL textures and tile maps move with their layer.
    auto holder = std::move(ls[from]);
    ls.erase(ls.begin() + static_cast<ptrdiff_t>(from));
    ls.insert(ls.begin() + static_cast<ptrdiff_t>(to), std::move(holder));

    // Reorder on disk via temp staging. Rename the moving layer aside,
    // shift the in-between dirs into the slot it just vacated, then
    // rename the staged dir into its final position. Same approach as
    // moving an array element in place but at the filesystem level.
    if (!g_docDir.empty()) {
        std::string pageDir = pageDirOf(g_activePageIdx);
        std::string staged = pageDir + "/layer_tmp_move";
        std::string fromDir = pageDir + "/layer_" + std::to_string(from);
        struct stat st;
        if (stat(fromDir.c_str(), &st) == 0) {
            rename(fromDir.c_str(), staged.c_str());
        }
        if (from < to) {
            // Shift layer_{from+1..to} down by one.
            for (size_t j = from; j < to; ++j) {
                std::string oldDir = pageDir + "/layer_" + std::to_string(j + 1);
                std::string newDir = pageDir + "/layer_" + std::to_string(j);
                if (stat(oldDir.c_str(), &st) == 0) {
                    rename(oldDir.c_str(), newDir.c_str());
                }
            }
        } else {
            // Shift layer_{to..from-1} up by one. Iterate top-down so we
            // don't clobber a target slot that's still occupied.
            for (size_t j = from; j > to; --j) {
                std::string oldDir = pageDir + "/layer_" + std::to_string(j - 1);
                std::string newDir = pageDir + "/layer_" + std::to_string(j);
                if (stat(oldDir.c_str(), &st) == 0) {
                    rename(oldDir.c_str(), newDir.c_str());
                }
            }
        }
        std::string toDir = pageDir + "/layer_" + std::to_string(to);
        if (stat(staged.c_str(), &st) == 0) {
            rename(staged.c_str(), toDir.c_str());
        }
    }

    size_t& active = g_pages[g_activePageIdx]->activeLayer;
    if (active == from) {
        active = to;
    } else if (from < to && active > from && active <= to) {
        --active;
    } else if (to < from && active >= to && active < from) {
        ++active;
    }

    LOGI("layer moved %zu->%zu (count=%zu, active=%zu)",
         from, to, ls.size(), active);
}

// ---- Rasterization ------------------------------------------------------

// Render a list of vector shapes into a target raster layer's tiles. Used
// by both "rasterize entire vector layer" and "rasterize selected shape
// to layer below". Uses a page-sized off-screen FBO so shapes only need
// to be drawn once; per-tile glReadPixels chunks the result and CPU
// blends "src over dst" against existing tile bytes (premultiplied).
//
// Notes:
//   - We disable page-clip in the line shader: the shapes were already
//     authored within the page, and we want them to bake into whichever
//     tile they touch without per-fragment clipping artifacts.
//   - Tile/fbo orientation matches the bake: doc-y=0 lands at GL bottom,
//     so glReadPixels rows from temp FBO can be uploaded directly to
//     tiles via uploadTileBytesAndSave (same convention).
void rasterizeShapesIntoTiles(size_t targetLayerIdx,
                              const std::vector<Line>&    lines,
                              const std::vector<Rect>&    rects,
                              const std::vector<Ellipse>& ellipses,
                              const std::vector<Circle>&  circles) {
    if (lines.empty() && rects.empty() && ellipses.empty() && circles.empty()) {
        return;
    }
    if (targetLayerIdx >= layers().size() || !layers()[targetLayerIdx]) return;
    Layer& target = *layers()[targetLayerIdx];
    if (target.type != LayerType::Raster) return;

    PageClip page = readPageClip();
    if (!page.active) return;
    int pageMinX = static_cast<int>(std::floor(page.minX));
    int pageMinY = static_cast<int>(std::floor(page.minY));
    int pageMaxX = static_cast<int>(std::ceil (page.maxX));
    int pageMaxY = static_cast<int>(std::ceil (page.maxY));
    int pageW = pageMaxX - pageMinX;
    int pageH = pageMaxY - pageMinY;
    if (pageW <= 0 || pageH <= 0) return;
    constexpr int kMaxRasterizeDim = 4096;
    if (pageW > kMaxRasterizeDim || pageH > kMaxRasterizeDim) {
        LOGE("rasterize: page %dx%d exceeds cap %d", pageW, pageH, kMaxRasterizeDim);
        return;
    }

    // Allocate a one-shot temp FBO at page size and draw the shapes.
    GLint prevDrawFbo = 0, prevReadFbo = 0;
    GLint prevViewport[4] = {0};
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &prevDrawFbo);
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &prevReadFbo);
    glGetIntegerv(GL_VIEWPORT, prevViewport);

    GLuint tempTex = 0, tempFbo = 0;
    glGenTextures(1, &tempTex);
    glBindTexture(GL_TEXTURE_2D, tempTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, pageW, pageH, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glGenFramebuffers(1, &tempFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, tempFbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, tempTex, 0);

    glViewport(0, 0, pageW, pageH);
    glDisable(GL_BLEND);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

    // doc → tempFbo transform: shift so doc(page.minX, page.minY) → (0,0).
    float t[16] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        static_cast<float>(-pageMinX), static_cast<float>(-pageMinY), 0, 1
    };

    glUseProgram(g_lineProg.program);
    glBindVertexArray(g_quadVao);
    glUniformMatrix4fv(g_lineProg.uTransform, 1, GL_FALSE, t);
    glUniform2f(g_lineProg.uScreen, static_cast<float>(pageW),
                                    static_cast<float>(pageH));
    uploadPageClip(g_lineProg.uPageMin, g_lineProg.uPageMax,
                   g_lineProg.uPageActive, PageClip{false, 0, 0, 0, 0});
    glUniform1f(g_lineProg.uOpacity, 1.0f);

    for (const auto& l : lines) {
        drawLineSegment(l.x0, l.y0, l.x1, l.y1, l.color, l.width, 1.0f);
    }
    for (const auto& r : rects) {
        drawRectangleAsLines(r.x0, r.y0, r.x1, r.y1, r.rotation,
                             r.color, r.width, 1.0f);
    }
    for (const auto& e : ellipses) {
        drawEllipseAsLines(e.cx, e.cy, e.rx, e.ry, e.rotation,
                           e.color, e.width, 1.0f);
    }
    for (const auto& c : circles) {
        drawEllipseAsLines(c.cx, c.cy, c.radius, c.radius, /*rotation*/ 0.0f,
                           c.color, c.width, 1.0f);
    }
    glBindVertexArray(0);

    // Iterate every tile in the page bbox; chunk pixels from tempFbo
    // and CPU-blend into existing tile bytes ("src over dst" with
    // premultiplied alpha).
    auto floorDiv = [](int a, int b) {
        // C++ integer division truncates toward zero; floor-divide is
        // safer for negative tile coords (page anchored away from doc 0).
        int q = a / b;
        if ((a ^ b) < 0 && q * b != a) --q;
        return q;
    };
    int tx0 = floorDiv(pageMinX,        kTileSize);
    int tx1 = floorDiv(pageMaxX - 1,    kTileSize);
    int ty0 = floorDiv(pageMinY,        kTileSize);
    int ty1 = floorDiv(pageMaxY - 1,    kTileSize);

    glBindFramebuffer(GL_READ_FRAMEBUFFER, tempFbo);
    std::vector<uint8_t> tileBytes(kTileBytes);
    std::vector<uint8_t> srcChunk; // sized per tile

    for (int ty = ty0; ty <= ty1; ++ty) {
        for (int tx = tx0; tx <= tx1; ++tx) {
            int tileDocX = tx * kTileSize;
            int tileDocY = ty * kTileSize;
            // Source rect in tempFbo (doc origin at (pageMinX, pageMinY)).
            int srcX = tileDocX - pageMinX;
            int srcY = tileDocY - pageMinY;
            int srcW = kTileSize, srcH = kTileSize;
            int dstX = 0, dstY = 0;
            if (srcX < 0)         { dstX = -srcX; srcW -= dstX; srcX = 0; }
            if (srcY < 0)         { dstY = -srcY; srcH -= dstY; srcY = 0; }
            if (srcX + srcW > pageW) srcW = pageW - srcX;
            if (srcY + srcH > pageH) srcH = pageH - srcY;
            if (srcW <= 0 || srcH <= 0) continue;

            // Read tile-shaped chunk from tempFbo.
            srcChunk.assign(static_cast<size_t>(srcW) * srcH * 4, 0);
            glReadPixels(srcX, srcY, srcW, srcH,
                         GL_RGBA, GL_UNSIGNED_BYTE, srcChunk.data());

            // Skip fully-transparent tiles — saves an empty
            // upload + neighbor apron invalidation chain.
            bool anyOpaque = false;
            for (size_t i = 3; i < srcChunk.size(); i += 4) {
                if (srcChunk[i] != 0) { anyOpaque = true; break; }
            }
            if (!anyOpaque) continue;

            // Read existing tile bytes (zeros if no tile yet).
            auto it = target.tiles.find(tileKey(tx, ty));
            if (it != target.tiles.end()) {
                glBindFramebuffer(GL_READ_FRAMEBUFFER, it->second.fbo);
                glReadPixels(kApron, kApron, kTileSize, kTileSize,
                             GL_RGBA, GL_UNSIGNED_BYTE, tileBytes.data());
                glBindFramebuffer(GL_READ_FRAMEBUFFER, tempFbo);
            } else {
                std::fill(tileBytes.begin(), tileBytes.end(), 0);
            }

            // Premultiplied "src over dst" — same blend the GPU does
            // during composite.
            for (int row = 0; row < srcH; ++row) {
                for (int col = 0; col < srcW; ++col) {
                    int dstIdx = ((dstY + row) * kTileSize + (dstX + col)) * 4;
                    int srcIdx = (row * srcW + col) * 4;
                    uint8_t sr = srcChunk[srcIdx + 0];
                    uint8_t sg = srcChunk[srcIdx + 1];
                    uint8_t sb = srcChunk[srcIdx + 2];
                    uint8_t sa = srcChunk[srcIdx + 3];
                    if (sa == 0) continue;
                    if (sa == 255) {
                        tileBytes[dstIdx + 0] = sr;
                        tileBytes[dstIdx + 1] = sg;
                        tileBytes[dstIdx + 2] = sb;
                        tileBytes[dstIdx + 3] = sa;
                    } else {
                        uint32_t inv = 255u - sa;
                        uint8_t dr = tileBytes[dstIdx + 0];
                        uint8_t dg = tileBytes[dstIdx + 1];
                        uint8_t db = tileBytes[dstIdx + 2];
                        uint8_t da = tileBytes[dstIdx + 3];
                        tileBytes[dstIdx + 0] =
                            static_cast<uint8_t>(sr + (dr * inv + 127u) / 255u);
                        tileBytes[dstIdx + 1] =
                            static_cast<uint8_t>(sg + (dg * inv + 127u) / 255u);
                        tileBytes[dstIdx + 2] =
                            static_cast<uint8_t>(sb + (db * inv + 127u) / 255u);
                        tileBytes[dstIdx + 3] =
                            static_cast<uint8_t>(sa + (da * inv + 127u) / 255u);
                    }
                }
            }

            uploadTileBytesAndSave(targetLayerIdx, tx, ty, tileBytes.data());
        }
    }

    // Tear down the temp FBO + restore prior bindings.
    glBindFramebuffer(GL_READ_FRAMEBUFFER, prevReadFbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, prevDrawFbo);
    glDeleteFramebuffers(1, &tempFbo);
    glDeleteTextures(1, &tempTex);
    glViewport(prevViewport[0], prevViewport[1],
               prevViewport[2], prevViewport[3]);
}

// Convert a vector layer to a raster layer in place. Renders all of its
// shapes into freshly-baked tiles, drops the shape arrays, flips
// layer.type, and removes the on-disk shapes.bin so the type is
// canonical after restart. Pushes a RasterizeLayer undo entry that
// can flip the layer back to vector.
void rasterizeVectorLayerImpl(size_t layerIdx) {
    if (layerIdx >= layers().size() || !layers()[layerIdx]) return;
    Layer& src = *layers()[layerIdx];
    if (src.type != LayerType::Vector) return;
    if (src.lines.empty() && src.rects.empty()
        && src.ellipses.empty() && src.circles.empty()) {
        // Empty vector layer — just flip the type and drop shapes.bin.
        // No undo entry: nothing user-visible changed.
        src.type = LayerType::Raster;
        if (!g_docDir.empty()) {
            std::string p = activeLayerDir(layerIdx) + "/shapes.bin";
            unlink(p.c_str());
        }
        return;
    }

    // Capture pre-rasterize shape lists for undo. Then move them into
    // typed locals so the rasterize call can consume them after the
    // layer's shape vectors are cleared.
    UndoEntry entry;
    entry.op              = UndoOp::RasterizeLayer;
    entry.layerIdx        = layerIdx;
    entry.layerTypeBefore = LayerType::Vector;
    entry.layerTypeAfter  = LayerType::Raster;
    entry.beforeLines     = src.lines;
    entry.beforeRects     = src.rects;
    entry.beforeEllipses  = src.ellipses;
    entry.beforeCircles   = src.circles;

    // Promote the layer type up front so rasterizeShapesIntoTiles
    // accepts it as a raster target. If we promoted after the call,
    // the early type-check would refuse and we'd silently no-op.
    src.type = LayerType::Raster;

    // Snapshot shape lists so they survive being cleared mid-call.
    std::vector<Line>    ls = std::move(src.lines);    src.lines.clear();
    std::vector<Rect>    rs = std::move(src.rects);    src.rects.clear();
    std::vector<Ellipse> es = std::move(src.ellipses); src.ellipses.clear();
    std::vector<Circle>  cs = std::move(src.circles);  src.circles.clear();

    rasterizeShapesIntoTiles(layerIdx, ls, rs, es, cs);

    // Capture the resulting tiles so redo can re-apply them without
    // re-running the GPU rasterize path.
    snapshotAllTiles(layerIdx, entry.afterTiles);

    // Disk side: shapes.bin is no longer the source of truth — a vector
    // layer's presence-of-shapes.bin is how loadAllLayersFromDisk
    // detects the type, so removing it commits the conversion.
    if (!g_docDir.empty()) {
        std::string p = activeLayerDir(layerIdx) + "/shapes.bin";
        unlink(p.c_str());
    }

    // The vector layer's selection (if it pointed at one of these
    // shapes) is now stale.
    {
        std::lock_guard<std::mutex> lock(g_selectionMutex);
        if (g_selection.kind != ShapeKind::None
            && g_selection.layerIdx == layerIdx) {
            g_selection = Selection{};
        }
    }
    // Pre-existing undo entries on this layer reference shape indices
    // that no longer exist (the layer is now raster). Drop just those;
    // entries on OTHER layers stay intact.
    {
        std::lock_guard<std::mutex> lock(g_undoMutex);
        auto purge = [&](std::deque<UndoEntry>& stack, size_t& total) {
            for (auto it = stack.begin(); it != stack.end(); ) {
                bool stale = (it->layerIdx == layerIdx)
                    && (it->op == UndoOp::VectorAdd
                     || it->op == UndoOp::VectorDelete
                     || it->op == UndoOp::VectorMutate
                     || it->op == UndoOp::LayerClear);
                if (stale) {
                    total -= it->bytes;
                    it = stack.erase(it);
                } else {
                    ++it;
                }
            }
        };
        purge(g_undoStack, g_undoTotalBytes);
        purge(g_redoStack, g_redoTotalBytes);
    }

    pushUndoEntry(std::move(entry));

    LOGI("rasterized vector layer %zu (%zu lines, %zu rects, "
         "%zu ellipses, %zu circles)",
         layerIdx, ls.size(), rs.size(), es.size(), cs.size());
}

// Doc-space AABB of a single vector shape, padded by half its stroke
// width plus 1 doc-px so anti-aliased edges land safely inside the
// snapshot region. Used by rasterizeShapeBelowImpl to bound the set of
// target tiles whose pixels need to be captured for undo.
struct DocBbox { float minX, minY, maxX, maxY; };
DocBbox shapeAabb(const ShapeData& s) {
    DocBbox b{0, 0, 0, 0};
    float pad = 1.0f;
    switch (s.kind) {
        case ShapeKind::Line: {
            const Line& l = s.line;
            b.minX = std::min(l.x0, l.x1); b.maxX = std::max(l.x0, l.x1);
            b.minY = std::min(l.y0, l.y1); b.maxY = std::max(l.y0, l.y1);
            pad += l.width * 0.5f;
            break;
        }
        case ShapeKind::Rect: {
            const Rect& r = s.rect;
            float cx = (r.x0 + r.x1) * 0.5f, cy = (r.y0 + r.y1) * 0.5f;
            float hw = std::fabs(r.x1 - r.x0) * 0.5f;
            float hh = std::fabs(r.y1 - r.y0) * 0.5f;
            float c = std::cos(r.rotation), si = std::sin(r.rotation);
            float ax = std::fabs(hw * c) + std::fabs(hh * si);
            float ay = std::fabs(hw * si) + std::fabs(hh * c);
            b.minX = cx - ax; b.maxX = cx + ax;
            b.minY = cy - ay; b.maxY = cy + ay;
            pad += r.width * 0.5f;
            break;
        }
        case ShapeKind::Ellipse: {
            const Ellipse& e = s.ellipse;
            float c = std::cos(e.rotation), si = std::sin(e.rotation);
            float ax = std::fabs(e.rx * c) + std::fabs(e.ry * si);
            float ay = std::fabs(e.rx * si) + std::fabs(e.ry * c);
            b.minX = e.cx - ax; b.maxX = e.cx + ax;
            b.minY = e.cy - ay; b.maxY = e.cy + ay;
            pad += e.width * 0.5f;
            break;
        }
        case ShapeKind::Circle: {
            const Circle& c = s.circle;
            b.minX = c.cx - c.radius; b.maxX = c.cx + c.radius;
            b.minY = c.cy - c.radius; b.maxY = c.cy + c.radius;
            pad += c.width * 0.5f;
            break;
        }
        default: return b;
    }
    b.minX -= pad; b.minY -= pad;
    b.maxX += pad; b.maxY += pad;
    return b;
}

// Floor-divide that handles negative numerators correctly (C++ /
// truncates toward zero; we want toward -inf for tile coords on a
// page anchored away from doc 0).
inline int tileFloorDiv(int a, int b) {
    int q = a / b;
    if ((a ^ b) < 0 && q * b != a) --q;
    return q;
}

// Rasterize the currently-selected vector shape onto the raster layer
// directly below the source vector layer. No-op if there's no
// selection, the layer below is missing or non-raster, or the source
// shape index is out of range. Pushes a single RasterizeShapeBelow
// undo entry covering both the shape's removal and the target's tile
// changes.
void rasterizeShapeBelowImpl() {
    Selection sel;
    {
        std::lock_guard<std::mutex> lock(g_selectionMutex);
        sel = g_selection;
    }
    if (sel.kind == ShapeKind::None) return;
    if (sel.layerIdx == 0) return;                   // no layer below
    if (sel.layerIdx >= layers().size()) return;
    if (!layers()[sel.layerIdx]) return;
    Layer& src = *layers()[sel.layerIdx];
    if (src.type != LayerType::Vector) return;
    size_t targetIdx = sel.layerIdx - 1;
    if (!layers()[targetIdx]) return;
    if (layers()[targetIdx]->type != LayerType::Raster) return;

    // Capture the shape into both a typed one-element list (for the
    // rasterize call) and a ShapeData (for the undo entry).
    ShapeData beforeShape;
    beforeShape.kind = sel.kind;
    std::vector<Line>    ls;
    std::vector<Rect>    rs;
    std::vector<Ellipse> es;
    std::vector<Circle>  cs;
    switch (sel.kind) {
        case ShapeKind::Line:
            if (sel.shapeIdx >= src.lines.size())    return;
            beforeShape.line = src.lines[sel.shapeIdx];
            ls.push_back(beforeShape.line);
            src.lines.erase(src.lines.begin() + sel.shapeIdx);
            break;
        case ShapeKind::Rect:
            if (sel.shapeIdx >= src.rects.size())    return;
            beforeShape.rect = src.rects[sel.shapeIdx];
            rs.push_back(beforeShape.rect);
            src.rects.erase(src.rects.begin() + sel.shapeIdx);
            break;
        case ShapeKind::Ellipse:
            if (sel.shapeIdx >= src.ellipses.size()) return;
            beforeShape.ellipse = src.ellipses[sel.shapeIdx];
            es.push_back(beforeShape.ellipse);
            src.ellipses.erase(src.ellipses.begin() + sel.shapeIdx);
            break;
        case ShapeKind::Circle:
            if (sel.shapeIdx >= src.circles.size())  return;
            beforeShape.circle = src.circles[sel.shapeIdx];
            cs.push_back(beforeShape.circle);
            src.circles.erase(src.circles.begin() + sel.shapeIdx);
            break;
        default: return;
    }

    // Snapshot the target tiles within the shape's bbox BEFORE the
    // bake; same range AFTER. Both go on the undo entry so reverse
    // restores the target to its pre-bake pixels and forward restores
    // the post-bake pixels. Clamp the bbox to the page rect (rasterize
    // skips anything outside it anyway).
    DocBbox bb = shapeAabb(beforeShape);
    PageClip page = readPageClip();
    if (page.active) {
        bb.minX = std::max(bb.minX, page.minX);
        bb.minY = std::max(bb.minY, page.minY);
        bb.maxX = std::min(bb.maxX, page.maxX);
        bb.maxY = std::min(bb.maxY, page.maxY);
    }
    std::vector<TileSnap> beforeTiles, afterTiles;
    if (bb.maxX > bb.minX && bb.maxY > bb.minY) {
        int tx0 = tileFloorDiv(static_cast<int>(std::floor(bb.minX)), kTileSize);
        int tx1 = tileFloorDiv(
            static_cast<int>(std::ceil(bb.maxX) - 1), kTileSize);
        int ty0 = tileFloorDiv(static_cast<int>(std::floor(bb.minY)), kTileSize);
        int ty1 = tileFloorDiv(
            static_cast<int>(std::ceil(bb.maxY) - 1), kTileSize);
        snapshotTilesInBbox(targetIdx, tx0, tx1, ty0, ty1, beforeTiles);
        rasterizeShapesIntoTiles(targetIdx, ls, rs, es, cs);
        snapshotTilesInBbox(targetIdx, tx0, tx1, ty0, ty1, afterTiles);
    } else {
        rasterizeShapesIntoTiles(targetIdx, ls, rs, es, cs);
    }

    // Persist the source layer's new (one-shape-shorter) state.
    saveVectorLayer(sel.layerIdx, src);

    // The selected shape is gone — clear the selection.
    {
        std::lock_guard<std::mutex> lock(g_selectionMutex);
        g_selection = Selection{};
    }

    // Push the reversible record. layerIdx points at the source vector
    // layer (where the shape gets re-inserted on undo); targetLayerIdx
    // points at the raster layer (whose tiles get restored).
    UndoEntry entry;
    entry.op             = UndoOp::RasterizeShapeBelow;
    entry.layerIdx       = sel.layerIdx;
    entry.targetLayerIdx = targetIdx;
    entry.shapeIdx       = sel.shapeIdx;
    entry.beforeShape    = beforeShape;
    entry.beforeTiles    = std::move(beforeTiles);
    entry.afterTiles     = std::move(afterTiles);
    pushUndoEntry(std::move(entry));

    LOGI("rasterized shape from layer %zu onto layer %zu",
         sel.layerIdx, targetIdx);
}

// Merge the raster layer at [idx] onto the raster layer at [idx-1]
// using premultiplied "src over dst" so the top layer's pixels stay on
// top. Then delete the source. Refuses if either layer is vector or
// the source is the bottom layer (nothing to merge with). Pushes a
// MergeLayerDown undo entry capturing the source layer's full state
// plus the target's tile diff.
void mergeRasterLayerDownImpl(size_t idx) {
    if (g_pages.empty() || g_activePageIdx >= g_pages.size()) return;
    auto& ls = g_pages[g_activePageIdx]->layers;
    if (idx == 0 || idx >= ls.size() || !ls[idx] || !ls[idx - 1]) {
        LOGI("merge layer down %zu refused — bad index", idx);
        return;
    }
    Layer& src = *ls[idx];
    Layer& tgt = *ls[idx - 1];
    if (src.type != LayerType::Raster || tgt.type != LayerType::Raster) {
        LOGI("merge layer down %zu refused — both must be raster "
             "(src=%d tgt=%d)", idx,
             static_cast<int>(src.type), static_cast<int>(tgt.type));
        return;
    }

    // Floating selection's lifted pixels live on its source layer; bake
    // back to keep the merge operating on a consistent picture.
    cancelRasterSelectionImpl();

    // Capture the source layer's full state BEFORE we modify anything,
    // so undo can re-create it. captureLayerSnapshot reads tile FBOs.
    UndoEntry entry;
    entry.op             = UndoOp::MergeLayerDown;
    entry.layerIdx       = idx;
    entry.targetLayerIdx = idx - 1;
    captureLayerSnapshot(idx, entry.srcSnapshot);

    // Snapshot the target tiles that source would touch BEFORE the
    // composite. Source's tile keys are exactly the set of target
    // tiles that may change (anything outside source's grid is
    // untouched). Some target keys won't exist yet (new tile creation
    // during merge) — TileSnap with existed=false handles that.
    GLint prevReadFbo = 0;
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &prevReadFbo);

    std::vector<int64_t> tileKeys;
    tileKeys.reserve(src.tiles.size());
    for (const auto& kv : src.tiles) tileKeys.push_back(kv.first);

    auto readTileSnap = [&](Layer& layer, int64_t k, TileSnap& snap) {
        unpackTileKey(k, snap.tx, snap.ty);
        auto it = layer.tiles.find(k);
        if (it != layer.tiles.end()) {
            snap.existed = true;
            if (it->second.cachedBytes) {
                snap.bytes = it->second.cachedBytes;
            } else {
                auto fresh = tilePool().acquire();
                glBindFramebuffer(GL_READ_FRAMEBUFFER, it->second.fbo);
                glReadPixels(kApron, kApron, kTileSize, kTileSize,
                             GL_RGBA, GL_UNSIGNED_BYTE, fresh->data());
                it->second.cachedBytes = fresh;
                snap.bytes = fresh;
            }
        }
    };

    entry.beforeTiles.reserve(tileKeys.size());
    for (int64_t k : tileKeys) {
        TileSnap snap;
        readTileSnap(tgt, k, snap);
        entry.beforeTiles.push_back(std::move(snap));
    }

    // Composite source over target, tile by tile.
    std::vector<uint8_t> srcBytes(kTileBytes);
    std::vector<uint8_t> tgtBytes(kTileBytes);
    for (auto& kv : src.tiles) {
        int tx, ty;
        unpackTileKey(kv.first, tx, ty);

        glBindFramebuffer(GL_READ_FRAMEBUFFER, kv.second.fbo);
        glReadPixels(kApron, kApron, kTileSize, kTileSize,
                     GL_RGBA, GL_UNSIGNED_BYTE, srcBytes.data());

        bool anyOpaque = false;
        for (size_t i = 3; i < srcBytes.size(); i += 4) {
            if (srcBytes[i] != 0) { anyOpaque = true; break; }
        }
        if (!anyOpaque) continue;

        auto it = tgt.tiles.find(kv.first);
        if (it != tgt.tiles.end()) {
            glBindFramebuffer(GL_READ_FRAMEBUFFER, it->second.fbo);
            glReadPixels(kApron, kApron, kTileSize, kTileSize,
                         GL_RGBA, GL_UNSIGNED_BYTE, tgtBytes.data());
        } else {
            std::fill(tgtBytes.begin(), tgtBytes.end(), 0);
        }

        // Premultiplied "src over dst" — same blend the GPU does during
        // composite, so the visual after merge matches what the user
        // saw with both layers visible.
        for (size_t i = 0; i < kTileBytes; i += 4) {
            uint8_t sa = srcBytes[i + 3];
            if (sa == 0) continue;
            if (sa == 255) {
                tgtBytes[i + 0] = srcBytes[i + 0];
                tgtBytes[i + 1] = srcBytes[i + 1];
                tgtBytes[i + 2] = srcBytes[i + 2];
                tgtBytes[i + 3] = 255;
            } else {
                uint32_t inv = 255u - sa;
                uint8_t dr = tgtBytes[i + 0];
                uint8_t dg = tgtBytes[i + 1];
                uint8_t db = tgtBytes[i + 2];
                uint8_t da = tgtBytes[i + 3];
                tgtBytes[i + 0] = static_cast<uint8_t>(
                    srcBytes[i + 0] + (dr * inv + 127u) / 255u);
                tgtBytes[i + 1] = static_cast<uint8_t>(
                    srcBytes[i + 1] + (dg * inv + 127u) / 255u);
                tgtBytes[i + 2] = static_cast<uint8_t>(
                    srcBytes[i + 2] + (db * inv + 127u) / 255u);
                tgtBytes[i + 3] = static_cast<uint8_t>(
                    sa + (da * inv + 127u) / 255u);
            }
        }

        uploadTileBytesAndSave(idx - 1, tx, ty, tgtBytes.data());
    }

    // Snapshot the post-composite target tiles (same key set) so redo
    // can restore the merged pixels without re-running the blend.
    entry.afterTiles.reserve(tileKeys.size());
    for (int64_t k : tileKeys) {
        TileSnap snap;
        readTileSnap(tgt, k, snap);
        entry.afterTiles.push_back(std::move(snap));
    }

    glBindFramebuffer(GL_READ_FRAMEBUFFER, prevReadFbo);

    // Extract any undo entries pinned to the source layer BEFORE the
    // delete drops them. Stored on the merge entry so undo of the
    // merge can re-push them — preserving the layer's pre-merge edit
    // history (e.g. a rasterize-layer that happened just before the
    // merge stays undoable after the merge is undone).
    {
        std::lock_guard<std::mutex> lock(g_undoMutex);
        auto extract = [&](std::deque<UndoEntry>& stack, size_t& total) {
            for (auto it = stack.begin(); it != stack.end(); ) {
                if (it->layerIdx == idx) {
                    total -= it->bytes;
                    entry.srcLayerUndoEntries.push_back(std::move(*it));
                    it = stack.erase(it);
                } else {
                    ++it;
                }
            }
        };
        extract(g_undoStack, g_undoTotalBytes);
        // Redo stack will be cleared by pushUndoEntry below — no need
        // to extract from it; those entries are about to vanish anyway.
    }

    // Delete the source layer. deleteLayerImpl handles GL teardown,
    // dir removal + renumber, active-layer adjustment, and remaps
    // surviving undo entries to the new layer-idx scheme. (The
    // source-layer entries we just extracted aren't in the stack, so
    // the remap pass leaves them untouched.)
    deleteLayerImpl(idx);

    // Push our entry AFTER the delete remap so the freshly-pushed
    // entry's layerIdx (= original source idx) is correct in the
    // post-delete index space — undo's insertLayerWithSnapshot will
    // re-introduce that slot.
    pushUndoEntry(std::move(entry));

    LOGI("merged raster layer %zu down into %zu", idx, idx - 1);
}

// Recursively wipe a page dir (`<docDir>/page_<idx>/`). Walks two levels
// because the layout is shallow: page_<idx>/layer_<m>/<files>. No-op if
// the dir doesn't exist.
void deletePageDirIfExists(size_t pageIdx) {
    if (g_docDir.empty()) return;
    std::string pageDir = pageDirOf(pageIdx);
    DIR* d = opendir(pageDir.c_str());
    if (!d) return;
    struct dirent* e;
    while ((e = readdir(d)) != nullptr) {
        const char* n = e->d_name;
        if (n[0] == '.') continue;
        std::string sub = pageDir + "/" + n;
        // page_*/ contents are layer_*/ dirs (plus possibly stray files).
        DIR* d2 = opendir(sub.c_str());
        if (d2) {
            struct dirent* e2;
            while ((e2 = readdir(d2)) != nullptr) {
                const char* n2 = e2->d_name;
                if (n2[0] == '.') continue;
                std::string p = sub + "/" + n2;
                unlink(p.c_str());
            }
            closedir(d2);
            rmdir(sub.c_str());
        } else {
            unlink(sub.c_str());
        }
    }
    closedir(d);
    rmdir(pageDir.c_str());
}

// Free GL resources owned by every layer on a page. Called before the
// Page is removed from g_pages (delete) or never — moves preserve
// resources by transferring the unique_ptr.
void freePageGLResources(Page& page) {
    for (auto& layer : page.layers) {
        if (!layer) continue;
        freeLayerGLResources(*layer);
    }
}

// Delete the page at `idx`. No-op if there's only one page left (the
// document needs at least one). Frees GL resources, wipes the on-disk
// dir, renumbers trailing page dirs to keep them 0..N-1 contiguous, and
// adjusts g_activePageIdx.
//
// Undo entries are scoped to the active page (no pageIdx field). If the
// deleted page IS the active page, the entries reference layers on a
// page that's about to be freed — they have to go. If the deleted page
// is some other page, the active page's layer indices are unchanged
// and the entries stay valid.
void deletePageImpl(size_t idx) {
    if (idx >= g_pages.size() || !g_pages[idx]) return;
    if (g_pages.size() <= 1) {
        LOGI("delete page %zu refused — last page", idx);
        return;
    }

    cancelRasterSelectionImpl();
    if (idx == g_activePageIdx) {
        {
            std::lock_guard<std::mutex> lock(g_selectionMutex);
            g_selection = Selection{};
        }
        {
            std::lock_guard<std::mutex> lock(g_undoMutex);
            g_undoStack.clear(); g_undoTotalBytes = 0;
            g_redoStack.clear(); g_redoTotalBytes = 0;
        }
    }

    freePageGLResources(*g_pages[idx]);
    deletePageDirIfExists(idx);

    g_pages.erase(g_pages.begin() + static_cast<ptrdiff_t>(idx));

    // Renumber trailing page dirs.
    if (!g_docDir.empty()) {
        for (size_t j = idx; j < g_pages.size(); ++j) {
            std::string oldDir = g_docDir + "/page_" + std::to_string(j + 1);
            std::string newDir = g_docDir + "/page_" + std::to_string(j);
            struct stat st;
            if (stat(oldDir.c_str(), &st) == 0) {
                rename(oldDir.c_str(), newDir.c_str());
            }
        }
    }

    if (g_activePageIdx == idx) {
        g_activePageIdx = std::min(idx, g_pages.size() - 1);
    } else if (g_activePageIdx > idx) {
        --g_activePageIdx;
    }

    // Deleting shifts the active page; the lazily-loaded target may not
    // be resident yet.
    ensurePageContentLoaded(g_activePageIdx);
    saveActivePageToDisk();

    LOGI("page %zu deleted (count=%zu, active=%zu)",
         idx, g_pages.size(), g_activePageIdx);
}

// Move the page at `from` to position `to`. Both the in-memory Page
// vector and on-disk page_<n>/ dirs are reordered; per-page contents
// are preserved.
//
// Undo / redo and the vector selection are kept untouched. They're
// implicitly scoped to the active page (no page index in UndoEntry),
// and g_activePageIdx is adjusted below to track the same Page object
// across the move — so existing entries still apply to the right
// layers after the reorder.
void movePageImpl(size_t from, size_t to) {
    if (from >= g_pages.size() || to >= g_pages.size() || from == to) return;
    if (!g_pages[from]) return;

    cancelRasterSelectionImpl();

    auto holder = std::move(g_pages[from]);
    g_pages.erase(g_pages.begin() + static_cast<ptrdiff_t>(from));
    g_pages.insert(g_pages.begin() + static_cast<ptrdiff_t>(to), std::move(holder));

    if (!g_docDir.empty()) {
        std::string staged = g_docDir + "/page_tmp_move";
        std::string fromDir = g_docDir + "/page_" + std::to_string(from);
        struct stat st;
        if (stat(fromDir.c_str(), &st) == 0) {
            rename(fromDir.c_str(), staged.c_str());
        }
        if (from < to) {
            for (size_t j = from; j < to; ++j) {
                std::string oldDir = g_docDir + "/page_" + std::to_string(j + 1);
                std::string newDir = g_docDir + "/page_" + std::to_string(j);
                if (stat(oldDir.c_str(), &st) == 0) {
                    rename(oldDir.c_str(), newDir.c_str());
                }
            }
        } else {
            for (size_t j = from; j > to; --j) {
                std::string oldDir = g_docDir + "/page_" + std::to_string(j - 1);
                std::string newDir = g_docDir + "/page_" + std::to_string(j);
                if (stat(oldDir.c_str(), &st) == 0) {
                    rename(oldDir.c_str(), newDir.c_str());
                }
            }
        }
        std::string toDir = g_docDir + "/page_" + std::to_string(to);
        if (stat(staged.c_str(), &st) == 0) {
            rename(staged.c_str(), toDir.c_str());
        }
    }

    if (g_activePageIdx == from) {
        g_activePageIdx = to;
    } else if (from < to && g_activePageIdx > from && g_activePageIdx <= to) {
        --g_activePageIdx;
    } else if (to < from && g_activePageIdx >= to && g_activePageIdx < from) {
        ++g_activePageIdx;
    }

    // The Page objects moved with their dirs, so residency and lazy-load
    // state track correctly across the reorder — only the persisted
    // index needs rewriting.
    saveActivePageToDisk();

    LOGI("page moved %zu->%zu (count=%zu, active=%zu)",
         from, to, g_pages.size(), g_activePageIdx);
}

// ---- Persistence ---------------------------------------------------------

// (Path helpers moved up near g_docDir so they're visible to earlier
// users in applyPendingLayerActions / clearLayerDirOnDisk / etc.)

// Move any tile_*_*.bin files at <docDir> root into <docDir>/layer_0/.
// One-time migration for documents written before the per-layer refactor.
void migrateLegacyTilesToLayer0IfNeeded() {
    if (g_docDir.empty()) return;

    DIR* d = opendir(g_docDir.c_str());
    if (!d) return;

    std::vector<std::string> legacy;
    struct dirent* e;
    while ((e = readdir(d)) != nullptr) {
        int tx, ty;
        if (sscanf(e->d_name, "tile_%d_%d.bin", &tx, &ty) == 2) {
            legacy.emplace_back(e->d_name);
        }
    }
    closedir(d);

    if (legacy.empty()) return;

    std::string layer0Dir = g_docDir + "/layer_0";
    if (mkdir(layer0Dir.c_str(), 0755) != 0 && errno != EEXIST) {
        LOGE("can't create %s (errno=%d)", layer0Dir.c_str(), errno);
        return;
    }

    int moved = 0;
    for (const auto& name : legacy) {
        std::string from = g_docDir + "/" + name;
        std::string to   = layer0Dir + "/" + name;
        if (rename(from.c_str(), to.c_str()) == 0) ++moved;
    }
    LOGI("migrated %d legacy tiles to layer_0", moved);
}

// Documents from before the multi-page refactor have layer_* dirs at the
// root of g_docDir. Move them into page_0/ so the current loader (which
// expects page_*) finds them. No-op when the doc is already laid out by
// page or when there's nothing on disk yet.
void migrateLegacyLayersToPage0IfNeeded() {
    if (g_docDir.empty()) return;

    DIR* d = opendir(g_docDir.c_str());
    if (!d) return;

    std::vector<std::string> rootLayers;
    bool hasPageDir = false;
    struct dirent* e;
    while ((e = readdir(d)) != nullptr) {
        int idx;
        if (sscanf(e->d_name, "page_%d", &idx) == 1) { hasPageDir = true; continue; }
        if (sscanf(e->d_name, "layer_%d", &idx) == 1) {
            rootLayers.emplace_back(e->d_name);
        }
    }
    closedir(d);

    if (rootLayers.empty()) return;
    if (hasPageDir) {
        // Mixed state — pages and stray root layers. Don't auto-merge to
        // avoid clobbering. Log and leave the strays where they are.
        LOGE("doc has both page_* and root layer_* dirs; skipping migration");
        return;
    }

    std::string page0Dir = pageDirOf(0);
    if (mkdir(page0Dir.c_str(), 0755) != 0 && errno != EEXIST) {
        LOGE("can't create %s (errno=%d)", page0Dir.c_str(), errno);
        return;
    }
    int moved = 0;
    for (const auto& name : rootLayers) {
        std::string from = g_docDir + "/" + name;
        std::string to   = page0Dir + "/" + name;
        if (rename(from.c_str(), to.c_str()) == 0) ++moved;
    }
    LOGI("migrated %d root layer dirs into page_0/", moved);
}

void loadTilesIntoLayer(Layer& layer, const std::string& dir) {
    DIR* d = opendir(dir.c_str());
    if (!d) return;

    std::vector<uint8_t> buf(kTileBytes);
    int loaded = 0;
    struct dirent* e;
    while ((e = readdir(d)) != nullptr) {
        int tx, ty;
        if (sscanf(e->d_name, "tile_%d_%d.bin", &tx, &ty) != 2) continue;
        std::string path = dir + "/" + e->d_name;
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) {
            LOGE("can't open %s", path.c_str());
            continue;
        }
        size_t n = fread(buf.data(), 1, buf.size(), f);
        fclose(f);
        if (n != buf.size()) {
            LOGE("tile %s short read: %zu bytes", e->d_name, n);
            continue;
        }
        getOrCreateTile(layer, tx, ty, buf.data());
        ++loaded;
    }
    closedir(d);
    LOGI("loaded %d tiles from %s", loaded, dir.c_str());
}

// Load every layer of a single page from <docDir>/page_<pageIdx>/ into
// the supplied Page. Detects each layer's type by file presence
// (shapes.bin → Vector; tile_*.bin → Raster).
//
// loadContent=false skips the GL-touching parts (raster tile upload,
// vector shapes table) — see loadAllPageMetadataFromDisk's comment for
// why we want a metadata-only mode reachable from the UI thread.
void loadPageLayersFromDisk(size_t pageIdx, Page& page, bool loadContent = true) {
    if (g_docDir.empty()) return;
    // Latch before the early-outs below: a page whose dir is missing or
    // holds no layers has no content to load, and we don't want
    // ensurePageContentLoaded retrying the directory scan every frame.
    if (loadContent) page.contentLoaded = true;
    std::string root = pageDirOf(pageIdx);
    DIR* d = opendir(root.c_str());
    if (!d) return;

    std::vector<int> indices;
    struct dirent* e;
    while ((e = readdir(d)) != nullptr) {
        int idx;
        if (sscanf(e->d_name, "layer_%d", &idx) == 1 && idx >= 0) {
            indices.push_back(idx);
        }
    }
    closedir(d);
    if (indices.empty()) return;

    std::sort(indices.begin(), indices.end());
    int maxIdx = indices.back();
    page.layers.resize(static_cast<size_t>(maxIdx + 1));
    for (auto& slot : page.layers) {
        if (!slot) slot = std::make_unique<Layer>();
    }

    for (int idx : indices) {
        std::string dirPath = root + "/layer_" + std::to_string(idx);
        Layer& layer = *page.layers[idx];
        struct stat st;
        std::string shapesPath = dirPath + "/shapes.bin";
        if (stat(shapesPath.c_str(), &st) == 0) {
            layer.type = LayerType::Vector;
            if (loadContent) loadVectorLayerShapes(layer, dirPath);
        } else {
            layer.type = LayerType::Raster;
            if (loadContent) loadTilesIntoLayer(layer, dirPath);
        }
        // Optional layer-hidden flag. Presence = layer is hidden in
        // the composite. Absence (the default) = visible. Atomic
        // because the compositor reads it on the GL thread while the
        // UI thread can flip it from setLayerVisible.
        std::string hiddenPath = dirPath + "/hidden.flag";
        struct stat hst;
        layer.visible.store(stat(hiddenPath.c_str(), &hst) != 0,
                            std::memory_order_relaxed);

        // Optional opacity. Plain text "0.0".."1.0"; absence = 1.0.
        std::string opacityPath = dirPath + "/opacity.txt";
        if (FILE* f = std::fopen(opacityPath.c_str(), "rb")) {
            char buf[32] = {};
            size_t n = std::fread(buf, 1, sizeof(buf) - 1, f);
            std::fclose(f);
            if (n > 0) {
                float o = std::strtof(buf, nullptr);
                if (o >= 0.0f && o <= 1.0f) {
                    layer.opacity.store(o, std::memory_order_relaxed);
                }
            }
        }

        // Optional user-set name. Absence = no custom name (UI shows
        // a default like "layer N" / "vector N").
        std::string namePath = dirPath + "/name.txt";
        if (FILE* f = std::fopen(namePath.c_str(), "rb")) {
            std::fseek(f, 0, SEEK_END);
            long sz = std::ftell(f);
            std::fseek(f, 0, SEEK_SET);
            if (sz > 0 && sz < 1024) {
                std::string buf(static_cast<size_t>(sz), '\0');
                if (std::fread(buf.data(), 1, sz, f) == static_cast<size_t>(sz)) {
                    // Trim trailing whitespace (newlines from manual edits).
                    while (!buf.empty() && (buf.back() == '\n' || buf.back() == '\r'
                                         || buf.back() == ' '  || buf.back() == '\t')) {
                        buf.pop_back();
                    }
                    std::lock_guard<std::mutex> lock(g_layerNameMutex);
                    layer.name = std::move(buf);
                }
            }
            std::fclose(f);
        }
    }
}

// Pull one page's content (raster tiles, vector shapes) off disk if it
// hasn't been already. GL thread only — creates textures and FBOs.
//
// Restores the draw-framebuffer binding on the way out: getOrCreateTile
// leaves the last tile's FBO bound, and this runs from inside
// compositeAllLayers and the pending-action drain, where clobbering the
// multi-buffer FBO would send the frame's clear and composite into a
// tile instead of the screen.
void ensurePageContentLoaded(size_t pageIdx) {
    if (pageIdx >= g_pages.size() || !g_pages[pageIdx]) return;
    if (g_pages[pageIdx]->contentLoaded) return;
    GLint prevFbo = 0;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &prevFbo);
    loadPageLayersFromDisk(pageIdx, *g_pages[pageIdx], /*loadContent=*/true);
    glBindFramebuffer(GL_FRAMEBUFFER, prevFbo);
}

void loadAllLayersFromDisk(bool loadContent = true) {
    if (g_docDir.empty()) return;

    // Is this a document coming into memory for the first time, or a
    // metadata refresh over a doc that's already resident? Only the
    // former should adopt the persisted active page. renameCurrentDocument
    // re-points g_docDir at the renamed directory and calls back through
    // here precisely BECAUSE it wants in-memory state left alone — it
    // must not yank the user off the page they're on.
    bool freshLoad = g_pages.empty();

    // Two-step legacy migration: bare tiles → layer_0/, root layer_* →
    // page_0/. Both are no-ops on already-migrated docs.
    migrateLegacyTilesToLayer0IfNeeded();
    migrateLegacyLayersToPage0IfNeeded();

    DIR* d = opendir(g_docDir.c_str());
    if (!d) return;

    std::vector<int> pageIndices;
    struct dirent* e;
    while ((e = readdir(d)) != nullptr) {
        int idx;
        if (sscanf(e->d_name, "page_%d", &idx) == 1 && idx >= 0) {
            pageIndices.push_back(idx);
        }
    }
    closedir(d);

    if (pageIndices.empty()) return;

    std::sort(pageIndices.begin(), pageIndices.end());
    int maxIdx = pageIndices.back();
    g_pages.resize(static_cast<size_t>(maxIdx + 1));
    for (auto& slot : g_pages) {
        if (!slot) slot = std::make_unique<Page>();
    }
    // Metadata for every page. Cheap: a directory scan plus a handful of
    // sub-kilobyte files per layer, no GL work, no tile reads.
    for (int idx : pageIndices) {
        loadPageLayersFromDisk(static_cast<size_t>(idx), *g_pages[idx],
                               /*loadContent=*/false);
    }

    // Restore the page the user left this document on before deciding
    // whose content to pull in.
    if (freshLoad) {
        g_activePageIdx = readActivePageFromDisk(g_pages.size());
    } else if (g_activePageIdx >= g_pages.size()) {
        g_activePageIdx = g_pages.size() - 1;
    }

    // Content for the active page only — every other page waits until
    // something actually needs its pixels (a page switch, a thumbnail,
    // an export). See ensurePageContentLoaded.
    if (loadContent) {
        ensurePageContentLoaded(g_activePageIdx);
        LOGI("loaded %zu pages (content: page %zu)",
             g_pages.size(), g_activePageIdx);
    }
}

// Metadata-only load — reads layer types, names, visibility, opacity
// from disk into g_pages without touching any GL state. Safe to call
// from the UI thread, which we do from setDocumentDir so the layer
// panel can render correct state immediately, instead of waiting for
// the GL thread's lazy ensureLoaded (which races a fixed-delay or
// polled UI sync).
void loadAllPageMetadataFromDisk() {
    loadAllLayersFromDisk(/*loadContent=*/false);
}

void ensureLoaded() {
    if (g_loaded.load(std::memory_order_acquire)) return;

    GLint prevFbo = 0;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &prevFbo);
    loadAllLayersFromDisk();
    glBindFramebuffer(GL_FRAMEBUFFER, prevFbo);
    // Guarantee at least one page exists post-load so subsequent code can
    // freely call layers() / activeLayer() (which deref g_pages).
    ensureAtLeastOnePage();
    // Publish *after* the load is complete — see comment on g_loaded.
    g_loaded.store(true, std::memory_order_release);
}

// ---- Vector-layer persistence --------------------------------------------
//
// Each vector layer is one file: <docDir>/layer_<idx>/shapes.bin. The
// presence of this file is also how loadAllLayersFromDisk recognizes a
// layer as Vector vs Raster, so we always write at least the header
// (4-byte magic + 4-byte shape count) — even for an empty vector layer.

constexpr uint32_t kShapesMagicV0   = 0x30434556u;   // "VEC0" — pre-rotation
constexpr uint32_t kShapesMagicV1   = 0x31434556u;   // "VEC1" — Rect/Ellipse have rotation
constexpr uint8_t  kShapeTypeLine    = 1;
constexpr uint8_t  kShapeTypeRect    = 2;
constexpr uint8_t  kShapeTypeEllipse = 3;
constexpr uint8_t  kShapeTypeCircle  = 4;

// Pre-rotation struct layouts, used only for migrating V0 files. Keep
// these byte-for-byte identical to what V0 was writing.
struct RectV0 {
    float    x0, y0;
    float    x1, y1;
    uint32_t color;
    float    width;
};
struct EllipseV0 {
    float    cx, cy;
    float    rx, ry;
    uint32_t color;
    float    width;
};

void saveVectorLayer(size_t layerIdx, const Layer& layer) {
    if (g_docDir.empty()) return;

    // Ensure the page dir exists too — pageIdx-only mkdirs(2) doesn't
    // create parent dirs.
    std::string pageDir = pageDirOf(g_activePageIdx);
    if (mkdir(pageDir.c_str(), 0755) != 0 && errno != EEXIST) {
        LOGE("save vector layer %zu: mkdir page (errno=%d)", layerIdx, errno);
        return;
    }
    std::string layerDir = activeLayerDir(layerIdx);
    if (mkdir(layerDir.c_str(), 0755) != 0 && errno != EEXIST) {
        LOGE("save vector layer %zu: mkdir failed (errno=%d)", layerIdx, errno);
        return;
    }

    std::string path    = layerDir + "/shapes.bin";
    std::string tmpPath = path + ".tmp";

    FILE* f = fopen(tmpPath.c_str(), "wb");
    if (!f) {
        LOGE("save vector layer %zu: fopen failed", layerIdx);
        return;
    }
    uint32_t magic = kShapesMagicV1;
    uint32_t count = static_cast<uint32_t>(
        layer.lines.size() + layer.rects.size()
        + layer.ellipses.size() + layer.circles.size());
    fwrite(&magic, sizeof(magic), 1, f);
    fwrite(&count, sizeof(count), 1, f);
    for (const auto& l : layer.lines) {
        uint8_t t = kShapeTypeLine;    fwrite(&t, 1, 1, f); fwrite(&l, sizeof(Line),    1, f);
    }
    for (const auto& r : layer.rects) {
        uint8_t t = kShapeTypeRect;    fwrite(&t, 1, 1, f); fwrite(&r, sizeof(Rect),    1, f);
    }
    for (const auto& e : layer.ellipses) {
        uint8_t t = kShapeTypeEllipse; fwrite(&t, 1, 1, f); fwrite(&e, sizeof(Ellipse), 1, f);
    }
    for (const auto& c : layer.circles) {
        uint8_t t = kShapeTypeCircle;  fwrite(&t, 1, 1, f); fwrite(&c, sizeof(Circle),  1, f);
    }
    fclose(f);
    if (rename(tmpPath.c_str(), path.c_str()) != 0) {
        LOGE("save vector layer %zu: rename failed", layerIdx);
    }
}

void loadVectorLayerShapes(Layer& layer, const std::string& dir) {
    std::string path = dir + "/shapes.bin";
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return;

    uint32_t magic = 0, count = 0;
    if (fread(&magic, sizeof(magic), 1, f) != 1
        || (magic != kShapesMagicV0 && magic != kShapesMagicV1)) {
        LOGE("vector layer at %s: bad magic 0x%x", dir.c_str(), magic);
        fclose(f);
        return;
    }
    bool migrate = (magic == kShapesMagicV0);
    if (fread(&count, sizeof(count), 1, f) != 1) {
        fclose(f);
        return;
    }
    for (uint32_t i = 0; i < count; ++i) {
        uint8_t type = 0;
        if (fread(&type, sizeof(type), 1, f) != 1) break;
        if (type == kShapeTypeLine) {
            Line l;    if (fread(&l, sizeof(Line),    1, f) != 1) break; layer.lines.push_back(l);
        } else if (type == kShapeTypeRect) {
            if (migrate) {
                RectV0 v0; if (fread(&v0, sizeof(RectV0), 1, f) != 1) break;
                Rect r{ v0.x0, v0.y0, v0.x1, v0.y1, /*rotation*/ 0.0f, v0.color, v0.width };
                layer.rects.push_back(r);
            } else {
                Rect r;    if (fread(&r, sizeof(Rect),    1, f) != 1) break; layer.rects.push_back(r);
            }
        } else if (type == kShapeTypeEllipse) {
            if (migrate) {
                EllipseV0 v0; if (fread(&v0, sizeof(EllipseV0), 1, f) != 1) break;
                Ellipse e{ v0.cx, v0.cy, v0.rx, v0.ry, /*rotation*/ 0.0f, v0.color, v0.width };
                layer.ellipses.push_back(e);
            } else {
                Ellipse e; if (fread(&e, sizeof(Ellipse), 1, f) != 1) break; layer.ellipses.push_back(e);
            }
        } else if (type == kShapeTypeCircle) {
            Circle c;  if (fread(&c, sizeof(Circle),  1, f) != 1) break; layer.circles.push_back(c);
        } else {
            LOGE("vector layer at %s: unknown shape type %d", dir.c_str(), type);
            break;
        }
    }
    fclose(f);
    // After migrating, rewrite in V1 format so subsequent loads are clean.
    // Caller will trigger save naturally on next stroke; but write here too.
    // (No-op for V1 files since we'd just overwrite identical content.)
    LOGI("loaded %zu lines + %zu rects + %zu ellipses + %zu circles from %s",
         layer.lines.size(), layer.rects.size(),
         layer.ellipses.size(), layer.circles.size(), dir.c_str());
}

// ---- Pending vector-layer shape additions --------------------------------

void applyPendingShapes() {
    std::vector<Line>    lines;
    std::vector<Rect>    rects;
    std::vector<Ellipse> ellipses;
    std::vector<Circle>  circles;
    {
        std::lock_guard<std::mutex> lock(g_pendingShapesMutex);
        lines.swap(g_pendingLines);
        rects.swap(g_pendingRects);
        ellipses.swap(g_pendingEllipses);
        circles.swap(g_pendingCircles);
    }
    if (lines.empty() && rects.empty() && ellipses.empty() && circles.empty())
        return;
    // Any new vector shape changes what compositeAllLayers will draw,
    // so the partial-recomposite cache must be considered stale.
    g_mbCacheValid = false;

    ensureAtLeastOnePage();

    // Shapes append to whichever layer is active when applied. If the
    // active layer isn't vector, drop them (the shape tools shouldn't
    // have committed in the first place; UI can prevent this).
    if (activeLayer() >= layers().size() || !layers()[activeLayer()]) return;
    Layer& layer = *layers()[activeLayer()];
    if (layer.type != LayerType::Vector) {
        LOGE("dropping queued shapes: active layer %zu is not vector",
             activeLayer());
        return;
    }
    for (const auto& s : lines) {
        layer.lines.push_back(s);
        UndoEntry e;
        e.op = UndoOp::VectorAdd;
        e.layerIdx = activeLayer();
        e.shapeIdx = layer.lines.size() - 1;
        e.afterShape.kind = ShapeKind::Line;
        e.afterShape.line = s;
        pushUndoEntry(std::move(e));
    }
    for (const auto& s : rects) {
        layer.rects.push_back(s);
        UndoEntry e;
        e.op = UndoOp::VectorAdd;
        e.layerIdx = activeLayer();
        e.shapeIdx = layer.rects.size() - 1;
        e.afterShape.kind = ShapeKind::Rect;
        e.afterShape.rect = s;
        pushUndoEntry(std::move(e));
    }
    for (const auto& s : ellipses) {
        layer.ellipses.push_back(s);
        UndoEntry e;
        e.op = UndoOp::VectorAdd;
        e.layerIdx = activeLayer();
        e.shapeIdx = layer.ellipses.size() - 1;
        e.afterShape.kind = ShapeKind::Ellipse;
        e.afterShape.ellipse = s;
        pushUndoEntry(std::move(e));
    }
    for (const auto& s : circles) {
        layer.circles.push_back(s);
        UndoEntry e;
        e.op = UndoOp::VectorAdd;
        e.layerIdx = activeLayer();
        e.shapeIdx = layer.circles.size() - 1;
        e.afterShape.kind = ShapeKind::Circle;
        e.afterShape.circle = s;
        pushUndoEntry(std::move(e));
    }
    saveVectorLayer(activeLayer(), layer);
}

// Write tile pixel bytes to disk via tmp+rename. Callable when the
// caller already has the bytes in hand (e.g. commitStroke's combined
// after-snapshot + disk save pass). Bytes must be exactly kTileBytes.
// The fopen/fwrite/fclose/rename happens on a background thread; the
// caller pays only one extra memcpy of kTileBytes. mkdir is kept on
// the calling thread — cheap when the dir already exists, and racing
// it across threads would require the writer to know the doc/page
// layout.
void writeTileBytesToDisk(size_t layerIdx, int tx, int ty,
                          const uint8_t* bytes) {
    if (g_docDir.empty()) return;
    // Make sure the page dir exists; mkdir doesn't create intermediates.
    // Kept sync on the calling thread — mkdir is cheap when the dir
    // already exists (a stat-like check) and racing it across threads
    // would require the writer to know the doc/page layout.
    std::string pageDir = pageDirOf(g_activePageIdx);
    mkdir(pageDir.c_str(), 0755);   // ignore EEXIST
    std::string layerDir = activeLayerDir(layerIdx);
    if (mkdir(layerDir.c_str(), 0755) != 0 && errno != EEXIST) {
        LOGE("can't create %s (errno=%d)", layerDir.c_str(), errno);
        return;
    }
    DiskTask task;
    task.op = DiskTask::kWrite;
    task.path    = layerDir + "/tile_" + std::to_string(tx)
                            + "_"      + std::to_string(ty) + ".bin";
    task.tmpPath = task.path + ".tmp";
    task.bytes.assign(bytes, bytes + kTileBytes);
    enqueueDiskTask(std::move(task));
}

void saveTileToDisk(size_t layerIdx, int64_t tileK) {
    if (g_docDir.empty()) return;
    if (layerIdx >= layers().size() || !layers()[layerIdx]) return;
    auto& tiles = layers()[layerIdx]->tiles;
    auto it = tiles.find(tileK);
    if (it == tiles.end()) return;

    int tx, ty;
    unpackTileKey(tileK, tx, ty);

    // Read pixels via the tile's FBO. Restore the previous binding before
    // returning so callers (notably undo/redo from inside the pending-
    // action drain at the start of compositeAllLayers) don't end up with
    // a tile FBO leaked into the multi-buffer composite path.
    GLint prevFbo = 0;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &prevFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, it->second.fbo);
    auto fresh = tilePool().acquire();
    glReadPixels(kApron, kApron, kTileSize, kTileSize, GL_RGBA, GL_UNSIGNED_BYTE,
                 fresh->data());
    glBindFramebuffer(GL_FRAMEBUFFER, prevFbo);

    // Cache the freshly-read bytes for the BEFORE-snapshot fast path —
    // saveTileToDisk is called by every mutation path that doesn't go
    // through uploadTileBytesAndSave (selection lift's glClear, bucket
    // fill, etc.), so this catches the post-mutation state.
    it->second.cachedBytes = fresh;

    writeTileBytesToDisk(layerIdx, tx, ty, fresh->data());
}

// ---- Deferred saveTile queue ---------------------------------------------
//
// saveTileToDisk's glReadPixels is the dominant cost in commitStroke
// (~3 ms per tile on TBDR mobile GPUs — same tile-store flush issue
// as apronSync's old glCopyTexSubImage2D path). For typical fast
// handwriting commits with 3 dirty tiles that's ~10 ms of
// synchronous stall, enough to push commit past one vsync.
//
// We defer the call: commitStroke enqueues `(layerIdx, tileK)` here,
// nulls the tile's cachedBytes (so any subsequent BEFORE snapshot
// knows to refresh from the GPU rather than use stale data), and
// returns fast. The queue is drained from:
//   - Trailing-edge idle commits (Kotlin scheduler, ~250 ms after the
//     last stroke), via the JNI flushPendingSaveTiles().
//   - The next stroke's BEFORE-snapshot for tiles that overlap its
//     bbox (drainPendingSaveTilesForBbox below) — those tiles need
//     fresh cachedBytes anyway.
//   - applyPendingLayerActions when any action is queued, since
//     layer/page mutations would invalidate layerIdx-relative keys.
//   - Best-effort on app pause (MainActivity.onPause calls
//     forceRedraw() which triggers an idle MB pass and drains).
//
// GL-thread only — saveTileToDisk uses GL. No mutex required.
std::deque<std::pair<size_t, int64_t>> g_pendingSaveTiles;

void enqueueDeferredSave(size_t layerIdx, int64_t tileK) {
    // Null out cachedBytes — the in-memory mirror is now stale
    // (FBO has post-bake content, cache still has pre-bake). The
    // BEFORE-snapshot fast path checks cachedBytes; nulling it
    // routes to the synchronous-readback fallback for any tile
    // that's still pending when its next snapshot needs it.
    if (layerIdx < layers().size() && layers()[layerIdx]) {
        auto& tiles = layers()[layerIdx]->tiles;
        auto it = tiles.find(tileK);
        if (it != tiles.end()) {
            it->second.cachedBytes.reset();
        }
    }
    g_pendingSaveTiles.emplace_back(layerIdx, tileK);
}

void drainPendingSaveTiles() {
    if (g_pendingSaveTiles.empty()) return;
    ATRACE_SCOPE("DrawingApp.drainPendingSaveTiles");
    // saveTileToDisk does its own FBO save/restore, so iteration is
    // safe even if the GL state changes between entries.
    while (!g_pendingSaveTiles.empty()) {
        auto [layerIdx, k] = g_pendingSaveTiles.front();
        g_pendingSaveTiles.pop_front();
        saveTileToDisk(layerIdx, k);
    }
}

void drainPendingSaveTilesForBbox(size_t layerIdx,
                                  int tx0, int tx1, int ty0, int ty1) {
    if (g_pendingSaveTiles.empty()) return;
    ATRACE_SCOPE("DrawingApp.drainPendingSaveTilesForBbox");
    auto it = g_pendingSaveTiles.begin();
    while (it != g_pendingSaveTiles.end()) {
        if (it->first == layerIdx) {
            int tx, ty;
            unpackTileKey(it->second, tx, ty);
            if (tx >= tx0 && tx <= tx1 && ty >= ty0 && ty <= ty1) {
                size_t li = it->first;
                int64_t k = it->second;
                it = g_pendingSaveTiles.erase(it);
                saveTileToDisk(li, k);
                continue;
            }
        }
        ++it;
    }
}

// ---- Shade tool: closed-path fill ----------------------------------------

// Compute a buffer-px AABB scissor for a doc-px bbox, given the
// column-major doc→buffer mat4. uTransform's output is buffer-px in
// the same convention glScissor expects (origin bottom-left, Y up), so
// no Y flip is needed. The result is padded and clamped to the
// drawable size; if the bbox lies entirely off-screen the rect's w or
// h come back zero.
struct PartialScissor { int x, y, w, h; };
PartialScissor dirtyBboxToScissor(const float* m,
                                  float x0, float y0,
                                  float x1, float y1,
                                  int width, int height,
                                  int pad) {
    const float cornersX[4] = { x0, x1, x1, x0 };
    const float cornersY[4] = { y0, y0, y1, y1 };
    float minBX = std::numeric_limits<float>::infinity();
    float minBY = std::numeric_limits<float>::infinity();
    float maxBX = -std::numeric_limits<float>::infinity();
    float maxBY = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < 4; ++i) {
        float bx = m[0] * cornersX[i] + m[4] * cornersY[i] + m[12];
        float by = m[1] * cornersX[i] + m[5] * cornersY[i] + m[13];
        if (bx < minBX) minBX = bx;
        if (by < minBY) minBY = by;
        if (bx > maxBX) maxBX = bx;
        if (by > maxBY) maxBY = by;
    }
    int ix0 = static_cast<int>(std::floor(minBX)) - pad;
    int iy0 = static_cast<int>(std::floor(minBY)) - pad;
    int ix1 = static_cast<int>(std::ceil (maxBX)) + pad;
    int iy1 = static_cast<int>(std::ceil (maxBY)) + pad;
    if (ix0 < 0)      ix0 = 0;
    if (iy0 < 0)      iy0 = 0;
    if (ix1 > width)  ix1 = width;
    if (iy1 > height) iy1 = height;
    int w = ix1 - ix0; if (w < 0) w = 0;
    int h = iy1 - iy0; if (h < 0) h = 0;
    return { ix0, iy0, w, h };
}

// Nonzero-winding point-in-polygon over the closed path xy[0..n) (the
// closing edge from the last vertex back to the first is implicit).
// Matches what the stencil fill draws, and is used by the bake to
// decide whether a tile with no samples near it is nonetheless in the
// filled interior.
bool pointInPolygonNonzero(const float* xy, size_t n, float px, float py) {
    int wind = 0;
    for (size_t i = 0; i < n; ++i) {
        size_t k = (i + 1) % n;
        float ax = xy[2 * i],     ay = xy[2 * i + 1];
        float bx = xy[2 * k],     by = xy[2 * k + 1];
        // Signed area of (a, b, p): >0 means p is on one side of a→b.
        float side = (bx - ax) * (py - ay) - (px - ax) * (by - ay);
        if (ay <= py) {
            if (by > py && side > 0.0f) ++wind;        // upward crossing
        } else {
            if (by <= py && side < 0.0f) --wind;       // downward crossing
        }
    }
    return wind != 0;
}

// Liang-Barsky segment-vs-AABB overlap test.
bool segmentIntersectsRect(float x0, float y0, float x1, float y1,
                           float rx0, float ry0, float rx1, float ry1) {
    if ((x0 >= rx0 && x0 <= rx1 && y0 >= ry0 && y0 <= ry1) ||
        (x1 >= rx0 && x1 <= rx1 && y1 >= ry0 && y1 <= ry1)) {
        return true;
    }
    float dx = x1 - x0, dy = y1 - y0;
    float t0 = 0.0f, t1 = 1.0f;
    const float p[4] = { -dx, dx, -dy, dy };
    const float q[4] = { x0 - rx0, rx1 - x0, y0 - ry0, ry1 - y0 };
    for (int i = 0; i < 4; ++i) {
        if (std::fabs(p[i]) < 1e-12f) {
            if (q[i] < 0.0f) return false;   // parallel and outside
            continue;
        }
        float r = q[i] / p[i];
        if (p[i] < 0.0f) {
            if (r > t1) return false;
            if (r > t0) t0 = r;
        } else {
            if (r < t0) return false;
            if (r < t1) t1 = r;
        }
    }
    return true;
}

// A shade path uploaded to g_polyVbo and ready to fill.
struct ShadePoly {
    bool    valid = false;
    GLsizei count = 0;                        // path vertices
    float   minX = 0, minY = 0, maxX = 0, maxY = 0;
    // The implicit closing chord, last path point → first.
    float   chordX0 = 0, chordY0 = 0;
    float   chordX1 = 0, chordY1 = 0;
};

// How the fill's boundary behaves at the closing chord.
struct ChordFill {
    // How far the fill's edge sits outside the chord, doc-px. Constant
    // along the chord, so the bridge stays a straight edge, and set to
    // the furthest any dab reaches past the chord — i.e. tangent to the
    // stroke's outermost dab, with nothing left protruding through.
    float offset = 0.0f;
    // Soft-edge width, doc-px.
    float band   = 0.0f;
};

// Width of the soft band the fill fades across at the chord, matching
// what the brush lays down along the rest of the boundary: widest dab
// radius in the stroke x (1 - hardness). A hard brush gives 0 — a crisp
// edge, same as its dabs.
//
// Deliberately NOT the dab radius at the chord's own endpoints: those
// are the pen-down and pen-lift samples, whose pressure (and hence
// radius, with kMinRadius = 0) is ~0, which zeroes the band and makes
// the feather silently vanish.
inline float chordFeatherBand(float maxStrokeRadius) {
    return maxStrokeRadius * std::max(0.0f, 1.0f - g_strokeBrushHardness);
}

// Unit normal of the closing chord pointing AWAY from the filled
// interior. Signed area picks the side; the point test confirms it,
// which also covers self-intersecting paths where the area is
// misleading. Returns false for a degenerate (zero-length) chord.
bool chordOutwardNormal(const float* xy, size_t n, float& nx, float& ny) {
    if (n < 3) return false;
    float c0x = xy[2 * (n - 1)], c0y = xy[2 * (n - 1) + 1];
    float c1x = xy[0],           c1y = xy[1];
    float dx = c1x - c0x, dy = c1y - c0y;
    float len = std::sqrt(dx * dx + dy * dy);
    if (len < 1e-6f) return false;
    float ux = dx / len, uy = dy / len;

    double area2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        size_t k = (i + 1) % n;
        area2 += static_cast<double>(xy[2 * i])     * xy[2 * k + 1]
               - static_cast<double>(xy[2 * k])     * xy[2 * i + 1];
    }
    nx = (area2 > 0.0) ?  uy : -uy;
    ny = (area2 > 0.0) ? -ux :  ux;

    constexpr float kProbe = 0.5f;
    if (pointInPolygonNonzero(xy, n,
                              (c0x + c1x) * 0.5f + nx * kProbe,
                              (c0y + c1y) * 0.5f + ny * kProbe)) {
        nx = -nx; ny = -ny;
    }
    return true;
}

// How far past the chord the stroke's dabs actually reach: for each
// sample, its outward distance from the chord line plus its own dab
// radius. The largest such reach is where the fill's edge has to sit
// for the bridge to be tangent to the outermost dab instead of letting
// it protrude.
//
// Note this is a MAX, not an average of the endpoint samples: pressure
// falls off as the pen lifts (and ramps up as it lands), so the biggest
// dab near either end is several samples in from the tip, and any
// averaged or endpoint-only reading sits inside it. Sampling reach
// rather than radius keeps dabs that are set back inside the shape from
// inflating the offset.
//
// Capped at the stroke's widest dab so the fill can never spill further
// past the chord than the brush itself spills past its centreline —
// that bounds the damage when a path swings outside the chord line.
// `xy` and `samples` are the same points in the same order.
float chordOutwardOffset(const float* xy, size_t n,
                         const std::vector<Sample>& samples,
                         float maxDabRadius) {
    float nx, ny;
    if (!chordOutwardNormal(xy, n, nx, ny)) return 0.0f;
    float c0x = xy[2 * (n - 1)], c0y = xy[2 * (n - 1) + 1];

    float best = 0.0f;
    for (const auto& s : samples) {
        float outward = (s.x - c0x) * nx + (s.y - c0y) * ny;
        best = std::max(best, outward + radiusOf(s.p));
    }
    return std::min(best, maxDabRadius);
}

// Scratch upload buffer — file-scope so a long stroke doesn't
// reallocate on every batch.
std::vector<float> g_polyUpload;

// Upload `n` interleaved doc-px path vertices, then the 4 corners of
// their AABB (triangle strip; the stencil cover quad), then 4 corners
// of an oriented box around the chord (triangle strip; covers the band
// the chord's outward offset adds outside the polygon). Oriented rather
// than axis-aligned so a long diagonal chord doesn't drag a huge
// mostly-discarded quad through the fragment stage every batch.
ShadePoly uploadShadePolygon(const float* xy, size_t n,
                             const ChordFill& chord) {
    ShadePoly out;
    if (n < 3) return out;
    float minX = xy[0], maxX = xy[0];
    float minY = xy[1], maxY = xy[1];
    for (size_t i = 1; i < n; ++i) {
        minX = std::min(minX, xy[2 * i]);
        maxX = std::max(maxX, xy[2 * i]);
        minY = std::min(minY, xy[2 * i + 1]);
        maxY = std::max(maxY, xy[2 * i + 1]);
    }
    // A path with no area encloses nothing.
    if (maxX - minX < 1e-3f || maxY - minY < 1e-3f) return out;

    g_polyUpload.clear();
    g_polyUpload.reserve(n * 2 + 16);
    g_polyUpload.insert(g_polyUpload.end(), xy, xy + n * 2);
    const float quad[8] = {
        minX, minY,  maxX, minY,
        minX, maxY,  maxX, maxY,
    };
    g_polyUpload.insert(g_polyUpload.end(), quad, quad + 8);

    // Oriented box around the chord, padded by the offset it reaches
    // (coverage is zero past that, so the pad needs no feather slack)
    // plus a pixel of slack.
    {
        float cx0 = xy[2 * (n - 1)], cy0 = xy[2 * (n - 1) + 1];
        float cx1 = xy[0],           cy1 = xy[1];
        float dx = cx1 - cx0, dy = cy1 - cy0;
        float len = std::sqrt(dx * dx + dy * dy);
        float ux = (len > 1e-6f) ? dx / len : 1.0f;
        float uy = (len > 1e-6f) ? dy / len : 0.0f;
        float px = -uy, py = ux;
        float w = chord.offset + 1.0f;
        // Extend past both ends so the rounded ends fit inside the box.
        float ax = cx0 - ux * w, ay = cy0 - uy * w;
        float bx = cx1 + ux * w, by = cy1 + uy * w;
        const float band[8] = {
            ax - px * w, ay - py * w,   bx - px * w, by - py * w,
            ax + px * w, ay + py * w,   bx + px * w, by + py * w,
        };
        g_polyUpload.insert(g_polyUpload.end(), band, band + 8);
    }

    glBindBuffer(GL_ARRAY_BUFFER, g_polyVbo);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(g_polyUpload.size() * sizeof(float)),
                 g_polyUpload.data(), GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    out.valid = true;
    out.count = static_cast<GLsizei>(n);
    out.minX = minX; out.minY = minY;
    out.maxX = maxX; out.maxY = maxY;
    out.chordX0 = xy[2 * (n - 1)]; out.chordY0 = xy[2 * (n - 1) + 1];
    out.chordX1 = xy[0];           out.chordY1 = xy[1];
    return out;
}

// Fill the uploaded path's interior into the bound FBO as flat coverage
// (rgb 0, alpha `alpha`), using the nonzero winding rule. The FBO MUST
// have a stencil attachment — without one the stencil test always
// passes and the whole AABB would fill. `mat4` maps the path's doc-px
// coordinates into the target (buffer-px for the live preview, a
// tile-local translation for the bake); `clip` is in doc-px either way.
//
// Caller owns the blend state: both the live preview and the bake run
// this with GL_MAX so the fill and the dabbed outline merge into one
// flat coverage instead of double-darkening where they overlap.
void drawShadeFill(const ShadePoly& poly, const float* mat4,
                   float screenW, float screenH,
                   const PageClip& clip, float alpha,
                   const ChordFill& chord) {
    if (!poly.valid) return;

    glBindVertexArray(g_polyVao);
    glUseProgram(g_poly.program);
    glUniformMatrix4fv(g_poly.uTransform, 1, GL_FALSE, mat4);
    glUniform2f(g_poly.uScreen, screenW, screenH);
    glUniform2f(g_poly.uChord0, poly.chordX0, poly.chordY0);
    glUniform2f(g_poly.uChord1, poly.chordX1, poly.chordY1);
    glUniform1f(g_poly.uChordOffset, chord.offset);
    glUniform1f(g_poly.uFeatherBand, chord.band);
    uploadPageClip(g_poly.uPageMin, g_poly.uPageMax,
                   g_poly.uPageActive, clip);

    // Pass 1: accumulate a winding number per pixel. Drawing the path as
    // a triangle fan makes each edge contribute exactly one signed
    // crossing (INCR_WRAP on front faces, DECR_WRAP on back), so the
    // stencil ends up holding the winding number — self-intersecting
    // scribbles come out solid rather than punched through.
    glEnable(GL_STENCIL_TEST);
    glStencilMask(0xFF);
    glClearStencil(0);
    glClear(GL_STENCIL_BUFFER_BIT);
    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
    glDisable(GL_CULL_FACE);
    glStencilFunc(GL_ALWAYS, 0, 0xFF);
    glStencilOpSeparate(GL_FRONT, GL_KEEP, GL_KEEP, GL_INCR_WRAP);
    glStencilOpSeparate(GL_BACK,  GL_KEEP, GL_KEEP, GL_DECR_WRAP);
    // Opaque + featherless for the winding pass: color writes are
    // masked off anyway, and the fragment shader discards at zero
    // alpha — a transparent or feathered pass 1 would silently drop
    // stencil writes and punch holes in the fill.
    glUniform4f(g_poly.uColor, 0.0f, 0.0f, 0.0f, 1.0f);
    glUniform1i(g_poly.uChordMode, 0);
    glDrawArrays(GL_TRIANGLE_FAN, 0, poly.count);

    // Pass 2: cover the AABB, keeping only nonzero-winding pixels.
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glStencilFunc(GL_NOTEQUAL, 0, 0xFF);
    glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
    glUniform4f(g_poly.uColor, 0.0f, 0.0f, 0.0f, alpha);
    glUniform1i(g_poly.uChordMode, 1);
    glDrawArrays(GL_TRIANGLE_STRIP, poly.count, 4);

    // Pass 3: the sliver the chord's outward offset adds beyond the
    // polygon. Stencil off — this is the one part of the fill that is
    // deliberately outside the path. It overlaps the interior too, but
    // its own falloff there is never above what pass 2 wrote, and the
    // caller's GL_MAX keeps whichever is larger.
    if (chord.offset > 1e-4f) {
        glDisable(GL_STENCIL_TEST);
        glUniform1i(g_poly.uChordMode, 2);
        glDrawArrays(GL_TRIANGLE_STRIP, poly.count + 4, 4);
    }

    glDisable(GL_STENCIL_TEST);
    glBindVertexArray(0);
}

// ---- Bake -----------------------------------------------------------------

void bakeCurrentStrokeIntoTiles(std::vector<int64_t>* dirtyOut,
                                size_t layerIdx) {
    if (g_current.samples.empty()) return;
    if (layerIdx >= layers().size()) return;
    Layer& layer = *layers()[layerIdx];

    // Brush/eraser are raster operations; they can't bake into a vector
    // layer. Drop the samples so they don't get re-tried on next render
    // and so stale tiles don't end up in the layer's tile map.
    if (layer.type != LayerType::Raster) {
        g_current.samples.clear();
        LOGI("brush/eraser stroke dropped: active layer %zu is vector",
             layerIdx);
        return;
    }

    // The dab radius is kMaxRadius * brush-size scale. At higher brush
    // sizes the bbox/touched checks below have to widen by the same
    // factor or they'll skip tiles the stroke actually paints into,
    // producing notches at tile corners after bake.
    float maxR = kMaxRadius * g_strokeBrushSizeScale;
    float pad  = maxR;
    float minX = g_current.samples.front().x, maxX = minX;
    float minY = g_current.samples.front().y, maxY = minY;
    for (const auto& s : g_current.samples) {
        if (s.x < minX) minX = s.x;
        if (s.x > maxX) maxX = s.x;
        if (s.y < minY) minY = s.y;
        if (s.y > maxY) maxY = s.y;
    }
    minX -= pad; maxX += pad;
    minY -= pad; maxY += pad;

    // Shade tool: the stroke's samples double as a closed path whose
    // interior is filled. Build the doc-px path once for the whole bake
    // — every tile draws the same buffer, only the tile-local
    // translation in uTransform differs.
    ShadePoly shadePoly;
    ChordFill shadeChord;
    std::vector<float> shadePath;
    if (g_strokeFillClosed) {
        shadePath.reserve(g_current.samples.size() * 2);
        float maxP = 0.0f;
        for (const auto& s : g_current.samples) {
            shadePath.push_back(s.x);
            shadePath.push_back(s.y);
            maxP = std::max(maxP, s.p);
        }
        float maxDabR     = radiusOf(maxP);
        shadeChord.offset = chordOutwardOffset(shadePath.data(),
                                               g_current.samples.size(),
                                               g_current.samples, maxDabR);
        shadeChord.band   = chordFeatherBand(maxDabR);
        shadePoly = uploadShadePolygon(shadePath.data(),
                                       g_current.samples.size(), shadeChord);
    }
    // The fill rides the uniform-alpha coverage path (that's what keeps
    // outline and fill one flat tone); beginStroke guarantees the pair
    // move together, and the guard keeps a mismatched snapshot from
    // dumping raw coverage into a tile.
    const bool shading = shadePoly.valid && g_strokeUniformAlpha;

    int tx0 = static_cast<int>(std::floor(minX / kTileSizeF));
    int tx1 = static_cast<int>(std::floor(maxX / kTileSizeF));
    int ty0 = static_cast<int>(std::floor(minY / kTileSizeF));
    int ty1 = static_cast<int>(std::floor(maxY / kTileSizeF));

    glUseProgram(g_dab.program);
    glBindVertexArray(g_quadVao);
    glUniformMatrix4fv(g_dab.uTransform, 1, GL_FALSE, kIdentity);
    glUniform2f(g_dab.uScreen, kTileSizeF, kTileSizeF);
    glUniform1f(g_dab.uHardness, g_strokeBrushHardness);

    // Uniform-alpha mode: dabs MAX-blend into a per-tile coverage texture
    // (only the alpha channel matters), and a single stamp pass composites
    // the brush color + coverage onto the tile. Stacking mode: existing
    // path — dabs blend directly into the tile and accumulate where they
    // overlap. Both paths share the outer dab-program setup; only the
    // per-tile inner work differs.
    if (g_strokeUniformAlpha) {
        // Stencil attachment is for the shade tool's fill; harmless (and
        // one-time) for plain uniform-alpha strokes.
        ensureViewFbo(g_strokeCoverageTile, kTileSize, kTileSize,
                      /*withStencil=*/true);
        // Coverage dab output: zero RGB, brush alpha. The stamp shader
        // supplies the brush color so the dab pass only needs to track
        // the max alpha. Same setup for brush and eraser; eraser is
        // distinguished at stamp time by uMode + blend func.
        float coverageColor[4] = { 0.0f, 0.0f, 0.0f, g_strokeBrushAlpha };
        glUniform4fv(g_dab.uColor, 1, coverageColor);
        glBlendEquation(GL_MAX);
    } else if (!strokeToolIsEraser()) {
        // Brush: additive premultiplied.
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
        glUniform4fv(g_dab.uColor, 1, g_strokeBrushColor);
    } else {
        // Eraser: subtract coverage. Tile alpha (and rgb) get scaled by
        // (1 - srcAlpha), so painted pixels go transparent and the
        // multi-buffer's paper-white shows through during composite.
        // Reuse g_strokeBrushAlpha for the per-dab strength so the
        // opacity slider drives "how much to erase" the same way it
        // drives brush opacity (the dab-accumulation curve is the
        // same shape for building up paint and tearing it down).
        glBlendFunc(GL_ZERO, GL_ONE_MINUS_SRC_ALPHA);
        float eraserColor[4] = { 0.0f, 0.0f, 0.0f, g_strokeBrushAlpha };
        glUniform4fv(g_dab.uColor, 1, eraserColor);
    }

    // Page bounds (doc-px) treat the canvas as a fixed-size sheet:
    //   - Tiles fully outside the page aren't created or baked at all.
    //   - Boundary tiles only emit dabs within the page rectangle.
    //   - The dab fragment shader also discards fragments outside the
    //     page in those boundary tiles, giving clean truncation right at
    //     the edge (the emitter-level clip alone leaves half-dab blobs).
    // When the bounds are inactive (zero-size), no clipping happens and
    // the doc behaves as the original infinite plane.
    PageClip pageClip = readPageClip();
    bool  pageActive = pageClip.active;
    float pageX0 = pageClip.minX, pageY0 = pageClip.minY;
    float pageX1 = pageClip.maxX, pageY1 = pageClip.maxY;

    // Walk the cells in the AABB but skip ones the stroke can't reach.
    // Without this, a long diagonal stroke (especially at low zoom-out)
    // would create thousands of empty 256x256 RGBA tile FBOs purely
    // because the AABB spans them — quickly exhausting GPU memory.
    // Pessimistic check uses kMaxRadius regardless of per-sample pressure.
    for (int ty = ty0; ty <= ty1; ++ty) {
        float tileY0 = ty * kTileSizeF;
        float tileY1 = tileY0 + kTileSizeF;
        for (int tx = tx0; tx <= tx1; ++tx) {
            float tileX0 = tx * kTileSizeF;
            float tileX1 = tileX0 + kTileSizeF;

            // Skip tiles entirely outside the page.
            if (pageActive
                && (tileX1 <= pageX0 || tileX0 >= pageX1
                 || tileY1 <= pageY0 || tileY0 >= pageY1)) {
                continue;
            }

            bool touched = false;
            for (const auto& s : g_current.samples) {
                if (s.x + maxR >= tileX0 && s.x - maxR <= tileX1
                 && s.y + maxR >= tileY0 && s.y - maxR <= tileY1) {
                    touched = true;
                    break;
                }
            }
            // A shaded region's interior tiles can sit far from any
            // sample, so the outline test above misses them. Add the
            // fill's own coverage: the tile is in the interior (corner
            // or centre inside the path), or the closing chord — the
            // one stretch of the fill boundary with no dabs on it —
            // crosses the tile.
            if (!touched && shading) {
                const float* p = shadePath.data();
                size_t n = g_current.samples.size();
                const float testX[5] = { tileX0, tileX1, tileX1, tileX0,
                                         (tileX0 + tileX1) * 0.5f };
                const float testY[5] = { tileY0, tileY0, tileY1, tileY1,
                                         (tileY0 + tileY1) * 0.5f };
                for (int i = 0; i < 5 && !touched; ++i) {
                    if (pointInPolygonNonzero(p, n, testX[i], testY[i])) {
                        touched = true;
                    }
                }
                // Grow the tile by the chord's outward offset — the fill
                // reaches that far past the chord, into tiles the chord
                // itself never enters.
                float co = shadeChord.offset;
                if (!touched
                    && segmentIntersectsRect(p[2 * (n - 1)], p[2 * (n - 1) + 1],
                                             p[0], p[1],
                                             tileX0 - co, tileY0 - co,
                                             tileX1 + co, tileY1 + co)) {
                    touched = true;
                }
            }
            if (!touched) continue;

            Tile& tile = getOrCreateTile(layer, tx, ty);
            float ox = tx * kTileSizeF;
            float oy = ty * kTileSizeF;

            // Per-tile clip bounds in tile-local coords. Start with the
            // full tile, then narrow to the intersection with the page
            // for boundary tiles so dabs that cross the edge stop at it.
            float clipMinX = 0.0f, clipMinY = 0.0f;
            float clipMaxX = kTileSizeF, clipMaxY = kTileSizeF;
            if (pageActive) {
                clipMinX = std::max(0.0f,        pageX0 - ox);
                clipMinY = std::max(0.0f,        pageY0 - oy);
                clipMaxX = std::min(kTileSizeF,  pageX1 - ox);
                clipMaxY = std::min(kTileSizeF,  pageY1 - oy);
            }

            // Set the dab fragment shader's page-clip uniforms in
            // tile-local coords. Together with the emitter-level clip
            // above, this gives clean fragment-level truncation at the
            // page edge (no half-dab blobs leaking past it).
            PageClip tilePage{pageActive, clipMinX, clipMinY, clipMaxX, clipMaxY};
            uploadPageClip(g_dab.uPageMin, g_dab.uPageMax,
                           g_dab.uPageActive, tilePage);

            if (g_strokeUniformAlpha) {
                // Coverage build: draw dabs into the scratch coverage
                // FBO with MAX blending. No apron — the coverage is
                // 256x256, aligned 1:1 with the tile's central region.
                glBindFramebuffer(GL_FRAMEBUFFER, g_strokeCoverageTile.fbo);
                glViewport(0, 0, kTileSize, kTileSize);
                glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
                glClear(GL_COLOR_BUFFER_BIT);
            } else {
                glBindFramebuffer(GL_FRAMEBUFFER, tile.fbo);
                glViewport(kApron, kApron, kTileSize, kTileSize);
            }

            DabEmitter e;
            // Clip to the tile (and page, where applicable) so the path's
            // interpolated dabs outside this region don't issue throwaway
            // GL draw calls and don't paint outside the canvas.
            e.setClipBounds(clipMinX, clipMinY, clipMaxX, clipMaxY);
            for (const auto& s : g_current.samples) {
                e.extend(s.x - ox, s.y - oy, s.p);
            }

            if (shading) {
                // Merge the enclosed region into the coverage the dabs
                // just built (still GL_MAX, so the shared border stays
                // one flat tone). The path stays in doc-px and the
                // tile's origin goes into the transform, so the page
                // clip here is the plain doc-space rect — not the
                // tile-local one the dab program needs.
                const float tileMat[16] = {
                    1.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 1.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 1.0f, 0.0f,
                     -ox,  -oy, 0.0f, 1.0f,
                };
                drawShadeFill(shadePoly, tileMat, kTileSizeF, kTileSizeF,
                              pageClip, g_strokeBrushAlpha, shadeChord);
                // Restore what the rest of the loop expects — the fill
                // swaps in its own program and vertex array.
                glUseProgram(g_dab.program);
                glBindVertexArray(g_quadVao);
            }

            if (g_strokeUniformAlpha) {
                // Stamp coverage onto the tile in one OVER (brush) or
                // DESTOUT (eraser) pass. The dab program's MAX state
                // is restored at the bottom of this iteration so the
                // next tile's coverage build picks up where it left off.
                glBindFramebuffer(GL_FRAMEBUFFER, tile.fbo);
                glViewport(kApron, kApron, kTileSize, kTileSize);
                glBlendEquation(GL_FUNC_ADD);
                if (!strokeToolIsEraser()) {
                    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
                } else {
                    glBlendFunc(GL_ZERO, GL_ONE_MINUS_SRC_ALPHA);
                }
                glUseProgram(g_stamp.program);
                glUniform1i(g_stamp.uCoverage, 0);
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, g_strokeCoverageTile.texture);
                glUniform4fv(g_stamp.uColor, 1, g_strokeBrushColor);
                glUniform1i(g_stamp.uMode, strokeToolIsEraser() ? 1 : 0);
                glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
                // Back to the dab program + MAX blend for the next
                // tile's coverage build.
                glUseProgram(g_dab.program);
                glBlendEquation(GL_MAX);
            }

            if (dirtyOut) dirtyOut->push_back(tileKey(tx, ty));
            // Tile content changed; flag this tile + its 8 neighbors
            // for apron resync before the next composite.
            markApronStaleAround(layer, tx, ty);
        }
    }

    g_current.samples.clear();
    glBindVertexArray(0);

    // Restore the default (additive premultiplied) blend so subsequent
    // paths don't inherit uniform-alpha (GL_MAX) or eraser state.
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    if (g_strokeUniformAlpha) {
        glBindTexture(GL_TEXTURE_2D, 0);
    }
}

// Replay a raster stroke's bake using inputs captured in the undo
// entry. Called from applyEntryForward (redo) on RasterStroke entries
// in place of the older "memcpy afterTiles back" path. The bake is
// deterministic given these inputs, so the result is bit-identical to
// the original commit (this is the property the BakeFidelityTest
// suite verifies). Saves the per-stroke brush + page-clip state,
// installs the entry's snapshot, bakes, persists dirty tiles to disk,
// then restores the prior state so subsequent strokes / live preview
// resume with the user's current brush.
static void rebakeStroke(const UndoEntry& e) {
    if (e.rebakeSamples.empty()) return;
    if (e.layerIdx >= layers().size() || !layers()[e.layerIdx]) return;

    // Snapshot brush + sample + page-clip state we're about to
    // overwrite. These are all GL-thread globals so no locking
    // required beyond the page-bounds mutex.
    float saveColor[4] = {
        g_strokeBrushColor[0], g_strokeBrushColor[1],
        g_strokeBrushColor[2], g_strokeBrushColor[3]
    };
    float saveAlpha = g_strokeBrushAlpha;
    float saveSize  = g_strokeBrushSizeScale;
    float saveHard  = g_strokeBrushHardness;
    int   saveTool  = g_strokeTool;
    bool  saveUniformAlpha = g_strokeUniformAlpha;
    bool  saveFillClosed   = g_strokeFillClosed;
    std::vector<Sample> saveSamples = std::move(g_current.samples);
    g_current.samples.clear();
    float savePageX0, savePageY0, savePageX1, savePageY1;
    {
        std::lock_guard<std::mutex> lock(g_pageBoundsMutex);
        savePageX0 = g_pageX0; savePageY0 = g_pageY0;
        savePageX1 = g_pageX1; savePageY1 = g_pageY1;
    }

    // Install the entry's snapshot.
    g_strokeBrushColor[0] = e.rebakeBrushColor[0];
    g_strokeBrushColor[1] = e.rebakeBrushColor[1];
    g_strokeBrushColor[2] = e.rebakeBrushColor[2];
    g_strokeBrushColor[3] = e.rebakeBrushColor[3];
    g_strokeBrushAlpha     = e.rebakeBrushAlpha;
    g_strokeBrushSizeScale = e.rebakeBrushSize;
    g_strokeBrushHardness  = e.rebakeBrushHardness;
    g_strokeTool           = e.rebakeTool;
    g_strokeUniformAlpha   = e.rebakeUniformAlpha;
    g_strokeFillClosed     = e.rebakeFillClosed;
    g_current.samples      = e.rebakeSamples;     // copy (re-bakeable)
    {
        std::lock_guard<std::mutex> lock(g_pageBoundsMutex);
        if (e.rebakePageActive) {
            g_pageX0 = e.rebakePageX0; g_pageY0 = e.rebakePageY0;
            g_pageX1 = e.rebakePageX1; g_pageY1 = e.rebakePageY1;
        } else {
            // Inactive snapshot — leave bounds as a zero-size rect so
            // readPageClip returns inactive.
            g_pageX0 = 0; g_pageY0 = 0; g_pageX1 = 0; g_pageY1 = 0;
        }
    }

    // Bake + persist. bakeCurrentStrokeIntoTiles binds tile FBOs in a
    // loop and exits with the last one bound — fine for commitStroke
    // (which wraps the call in a prevFbo save/restore) but fatal here:
    // rebakeStroke runs inside applyPendingLayerActions, which is
    // drained before compositeAllLayers. If we exit with a tile FBO
    // bound, the subsequent clear/grid/composite all go to the tile
    // instead of MB.back and the canvas turns black. Save + restore the
    // binding around the bake to keep the multi-buffer FBO active.
    GLint prevFbo = 0;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &prevFbo);

    std::vector<int64_t> dirty;
    bakeCurrentStrokeIntoTiles(&dirty, e.layerIdx);
    for (int64_t k : dirty) {
        saveTileToDisk(e.layerIdx, k);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, prevFbo);

    // Restore.
    g_strokeBrushColor[0] = saveColor[0];
    g_strokeBrushColor[1] = saveColor[1];
    g_strokeBrushColor[2] = saveColor[2];
    g_strokeBrushColor[3] = saveColor[3];
    g_strokeBrushAlpha     = saveAlpha;
    g_strokeBrushSizeScale = saveSize;
    g_strokeBrushHardness  = saveHard;
    g_strokeTool           = saveTool;
    g_strokeUniformAlpha   = saveUniformAlpha;
    g_strokeFillClosed     = saveFillClosed;
    g_current.samples      = std::move(saveSamples);
    {
        std::lock_guard<std::mutex> lock(g_pageBoundsMutex);
        g_pageX0 = savePageX0; g_pageY0 = savePageY0;
        g_pageX1 = savePageX1; g_pageY1 = savePageY1;
    }
}

// ---- Compose --------------------------------------------------------------

// Helper: bind the raster compositing pipeline (program + uniforms).
// Reused both at the top of compositeAllLayers and after we temporarily
// switch to the line program for a vector layer.
void bindRasterCompositePipeline(JNIEnv* env, jint width, jint height,
                                 jfloatArray transform) {
    glUseProgram(g_comp.program);
    glBindVertexArray(g_quadVao);
    uploadMat4(env, g_comp.uTransform, transform);
    glUniform2f(g_comp.uScreen, (float)width, (float)height);
    glUniform1f(g_comp.uTileHalf, kTileHalfF);
    glUniform1i(g_comp.uTileTex, 0);
    glActiveTexture(GL_TEXTURE0);
    // Default to fully opaque; per-layer compositors override before
    // their draw calls and we don't bother resetting on the way out
    // because the composite program isn't used by any non-layer pass.
    glUniform1f(g_comp.uOpacity, 1.0f);
}

void compositeRasterLayer(const Layer& layer, float opacityOverride) {
    ATRACE_SCOPE("DrawingApp.compositeRasterLayer");
    // Caller must have bound the raster pipeline first. opacityOverride
    // < 0 (the default) means "use the layer's own opacity"; passing
    // 1.0 forces opaque compositing (used by bucket fill so a partly-
    // transparent active layer still produces a clean boundary image).
    float effective = (opacityOverride >= 0.0f)
        ? opacityOverride
        : layer.opacity.load(std::memory_order_relaxed);
    glUniform1f(g_comp.uOpacity, effective);

    {
        ATRACE_SCOPE("DrawingApp.compositeRasterLayer.apronSync");
        // Lazily refresh any tile aprons that have been flagged stale since
        // the last composite. This guarantees LINEAR sampling at the tile
        // edges blends into up-to-date neighbor data instead of leaving a
        // visible seam. Apron sync mutates tile state (apronStale flag and
        // tile texture), so we cast away const; conceptually the
        // composite is read-only and the apron is just a derived cache.
        // syncTileApron saves/restores FBO + scissor state on its own.
        Layer& mut = const_cast<Layer&>(layer);
        for (auto& kv : mut.tiles) {
            if (!kv.second.apronStale) continue;
            int tx, ty;
            unpackTileKey(kv.first, tx, ty);
            syncTileApron(mut, tx, ty);
        }
        glActiveTexture(GL_TEXTURE0);
    }

    {
        ATRACE_SCOPE("DrawingApp.compositeRasterLayer.drawTiles");
        for (const auto& kv : layer.tiles) {
            int tx, ty;
            unpackTileKey(kv.first, tx, ty);
            float cx = tx * kTileSizeF + kTileHalfF;
            float cy = ty * kTileSizeF + kTileHalfF;
            glBindTexture(GL_TEXTURE_2D, kv.second.texture);
            glUniform2f(g_comp.uTileCenter, cx, cy);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        }
    }
}

// Draws one line segment using the line program (caller has already
// bound the program, VAO, and set uTransform / uScreen).
void drawLineSegment(float x0, float y0, float x1, float y1,
                     uint32_t rgb, float width, float alpha) {
    glUniform2f(g_lineProg.uP0, x0, y0);
    glUniform2f(g_lineProg.uP1, x1, y1);
    glUniform1f(g_lineProg.uHalfWidth, width * 0.5f);
    float r = ((rgb >> 16) & 0xFFu) / 255.0f;
    float g = ((rgb >>  8) & 0xFFu) / 255.0f;
    float b = ( rgb        & 0xFFu) / 255.0f;
    float c[4] = { r, g, b, alpha };
    glUniform4fv(g_lineProg.uColor, 1, c);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}

// Single-quad ellipse outline via SDF. Caller must have bound g_ellipseProg
// and set the per-frame uniforms (uTransform, uScreen, uOpacity, page clip).
// rx == ry produces an exact circle. Used in place of drawEllipseAsLines
// when the alpha may be < 1 (the polyline version double-covers at every
// segment junction, making sub-opaque strokes look darker than requested).
void drawEllipseOutline(float cx, float cy, float rx, float ry,
                        float rotation, uint32_t rgb, float width,
                        float alpha) {
    glUniform2f(g_ellipseProg.uCenter, cx, cy);
    glUniform2f(g_ellipseProg.uRadii, rx, ry);
    glUniform1f(g_ellipseProg.uRotation, rotation);
    glUniform1f(g_ellipseProg.uHalfWidth, width * 0.5f);
    float r = ((rgb >> 16) & 0xFFu) / 255.0f;
    float g = ((rgb >>  8) & 0xFFu) / 255.0f;
    float b = ( rgb        & 0xFFu) / 255.0f;
    float c[4] = { r, g, b, alpha };
    glUniform4fv(g_ellipseProg.uColor, 1, c);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}

// Single-quad rectangle outline via box SDF. Caller must have bound
// g_rectProg and set per-frame uniforms. Same motivation as
// drawEllipseOutline: drawing the outline as 4 line segments leaves the
// rounded caps overlapping at every corner.
void drawRectOutline(float x0, float y0, float x1, float y1, float rotation,
                     uint32_t rgb, float width, float alpha) {
    float cx = (x0 + x1) * 0.5f;
    float cy = (y0 + y1) * 0.5f;
    float hw = std::fabs(x1 - x0) * 0.5f;
    float hh = std::fabs(y1 - y0) * 0.5f;
    glUniform2f(g_rectProg.uCenter, cx, cy);
    glUniform2f(g_rectProg.uHalfExt, hw, hh);
    glUniform1f(g_rectProg.uRotation, rotation);
    glUniform1f(g_rectProg.uHalfWidth, width * 0.5f);
    float r = ((rgb >> 16) & 0xFFu) / 255.0f;
    float g = ((rgb >>  8) & 0xFFu) / 255.0f;
    float b = ( rgb        & 0xFFu) / 255.0f;
    float c[4] = { r, g, b, alpha };
    glUniform4fv(g_rectProg.uColor, 1, c);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}

constexpr uint32_t kHandleColor       = 0x4080FFu;   // blue
// Visual / hit-test sizes for selection UI, expressed in view pixels.
// Wrapped through vpxToDoc() at use sites so the on-screen size stays
// constant regardless of view scale. (Drawn shape widths remain in
// doc-px since they're a property of the doc, not the viewport.)
constexpr float    kHandleSizeViewPx        = 14.0f;
constexpr float    kHandleHitRadiusViewPx   = 18.0f;
constexpr float    kRotateHandleOffsetViewPx = 28.0f;
constexpr float    kSelectionOutlineWidthViewPx = 1.5f;
inline float vpxToDoc(float vpx) { return vpx / currentViewScale(); }
// UI-thread variant. Hit-tests run on the UI thread, where
// currentViewScale() is unsafe: renderPageThumbnail and the bucket-fill
// composite both temporarily overwrite g_viewScaleBits on the GL thread
// (see the comment on g_userViewScaleBits). A tap dispatched inside a
// thumbnail render's window would otherwise read the thumbnail's scale
// — ~0.05 — and inflate an 18 view-px handle radius to ~350 doc-px, so
// a tap well outside the selection registers as a corner grab.
inline float vpxToDocUi(float vpx) { return vpx / userViewScale(); }

// Oriented bounding box for a shape, in doc-pixel space. Defined before
// the handle-rendering helpers (which take it by reference) and before
// hit-testing helpers; the obbFor* dispatchers are defined alongside.
struct Obb {
    float cx, cy;     // center
    float hw, hh;     // half-extents (in shape-local frame)
    float rotation;   // radians
};

inline void rotateLocalToWorld(const Obb& o, float lx, float ly,
                               float& wx, float& wy) {
    float c = std::cos(o.rotation), s = std::sin(o.rotation);
    wx = o.cx + lx * c - ly * s;
    wy = o.cy + lx * s + ly * c;
}

inline void rotateWorldToLocal(const Obb& o, float wx, float wy,
                               float& lx, float& ly) {
    float c = std::cos(-o.rotation), s = std::sin(-o.rotation);
    float dx = wx - o.cx, dy = wy - o.cy;
    lx = dx * c - dy * s;
    ly = dx * s + dy * c;
}

Obb obbForLine(const Line& l) {
    float cx = (l.x0 + l.x1) * 0.5f;
    float cy = (l.y0 + l.y1) * 0.5f;
    float dx = l.x1 - l.x0;
    float dy = l.y1 - l.y0;
    float len = std::sqrt(dx*dx + dy*dy);
    float angle = (len > 1e-6f) ? std::atan2(dy, dx) : 0.0f;
    float hh = std::max(l.width * 0.5f, 6.0f);
    return { cx, cy, len * 0.5f, hh, angle };
}

Obb obbForRect(const Rect& r) {
    float cx = (r.x0 + r.x1) * 0.5f;
    float cy = (r.y0 + r.y1) * 0.5f;
    float hw = std::fabs(r.x1 - r.x0) * 0.5f;
    float hh = std::fabs(r.y1 - r.y0) * 0.5f;
    return { cx, cy, hw, hh, r.rotation };
}

Obb obbForEllipse(const Ellipse& e) {
    return { e.cx, e.cy, e.rx, e.ry, e.rotation };
}

Obb obbForCircle(const Circle& c) {
    return { c.cx, c.cy, c.radius, c.radius, 0.0f };
}

// OBB for the floating raster selection (its placement, not the source
// lift bbox — those diverge once the user scales/rotates). Caller must
// hold g_rasterSelMutex; helper is just the field-shuffle.
Obb obbForRasterSelection(const RasterSelection& s) {
    return { s.centerX, s.centerY, s.halfW, s.halfH, s.rotation };
}

bool obbForSelection(const Selection& sel, Obb& out) {
    if (sel.kind == ShapeKind::None) return false;
    if (sel.layerIdx >= layers().size() || !layers()[sel.layerIdx]) return false;
    const Layer& layer = *layers()[sel.layerIdx];
    // VALUE COPIES — see compositeVectorLayer's note. UI-thread
    // applyMoveTo can mutate shape fields without locking; reading
    // through a reference here would let obbForLine etc. see torn
    // values (x0 from the new state, x1 from the old) and produce a
    // bogus OBB.
    switch (sel.kind) {
        case ShapeKind::Line:
            if (sel.shapeIdx < layer.lines.size()) {
                Line snap = layer.lines[sel.shapeIdx];
                out = obbForLine(snap); return true;
            }
            break;
        case ShapeKind::Rect:
            if (sel.shapeIdx < layer.rects.size()) {
                Rect snap = layer.rects[sel.shapeIdx];
                out = obbForRect(snap); return true;
            }
            break;
        case ShapeKind::Ellipse:
            if (sel.shapeIdx < layer.ellipses.size()) {
                Ellipse snap = layer.ellipses[sel.shapeIdx];
                out = obbForEllipse(snap); return true;
            }
            break;
        case ShapeKind::Circle:
            if (sel.shapeIdx < layer.circles.size()) {
                Circle snap = layer.circles[sel.shapeIdx];
                out = obbForCircle(snap); return true;
            }
            break;
        case ShapeKind::None:
            break;
    }
    return false;
}

// ---- Snapping ------------------------------------------------------------

// Visit every snap-target point in every vector layer with the given
// callback. Each shape contributes its natural anchor points:
//   Line:    p0, p1, midpoint
//   Rect:    4 (rotated) corners + center
//   Ellipse: center + 4 cardinal points (rotated)
//   Circle:  center + 4 cardinal points
//
// `exclude`, if non-null, skips that exact (layer, kind, shapeIdx) so a
// shape can't snap to its own vertices during a transform drag.
template <typename F>
void forEachShapeSnapTarget(const F& cb, const Selection* exclude = nullptr) {
    for (size_t li = 0; li < layers().size(); ++li) {
        const auto& layer = layers()[li];
        if (!layer || layer->type != LayerType::Vector) continue;
        auto skip = [&](ShapeKind k, size_t i) {
            return exclude && exclude->kind == k
                && exclude->layerIdx == li && exclude->shapeIdx == i;
        };
        for (size_t i = 0; i < layer->lines.size(); ++i) {
            if (skip(ShapeKind::Line, i)) continue;
            const auto& l = layer->lines[i];
            cb(l.x0, l.y0);
            cb(l.x1, l.y1);
            cb((l.x0 + l.x1) * 0.5f, (l.y0 + l.y1) * 0.5f);
        }
        for (size_t i = 0; i < layer->rects.size(); ++i) {
            if (skip(ShapeKind::Rect, i)) continue;
            Obb o = obbForRect(layer->rects[i]);
            float cx, cy;
            // Corners.
            rotateLocalToWorld(o, -o.hw, -o.hh, cx, cy); cb(cx, cy);
            rotateLocalToWorld(o, +o.hw, -o.hh, cx, cy); cb(cx, cy);
            rotateLocalToWorld(o, +o.hw, +o.hh, cx, cy); cb(cx, cy);
            rotateLocalToWorld(o, -o.hw, +o.hh, cx, cy); cb(cx, cy);
            // Edge midpoints (rotated).
            rotateLocalToWorld(o, 0,     -o.hh, cx, cy); cb(cx, cy);
            rotateLocalToWorld(o, +o.hw, 0,     cx, cy); cb(cx, cy);
            rotateLocalToWorld(o, 0,     +o.hh, cx, cy); cb(cx, cy);
            rotateLocalToWorld(o, -o.hw, 0,     cx, cy); cb(cx, cy);
            cb(o.cx, o.cy);
        }
        for (size_t i = 0; i < layer->ellipses.size(); ++i) {
            if (skip(ShapeKind::Ellipse, i)) continue;
            Obb o = obbForEllipse(layer->ellipses[i]);
            cb(o.cx, o.cy);
            float cx, cy;
            rotateLocalToWorld(o, +o.hw, 0,      cx, cy); cb(cx, cy);
            rotateLocalToWorld(o, -o.hw, 0,      cx, cy); cb(cx, cy);
            rotateLocalToWorld(o, 0,     +o.hh,  cx, cy); cb(cx, cy);
            rotateLocalToWorld(o, 0,     -o.hh,  cx, cy); cb(cx, cy);
        }
        for (size_t i = 0; i < layer->circles.size(); ++i) {
            if (skip(ShapeKind::Circle, i)) continue;
            const auto& c = layer->circles[i];
            cb(c.cx, c.cy);
            cb(c.cx + c.radius, c.cy);
            cb(c.cx - c.radius, c.cy);
            cb(c.cx, c.cy + c.radius);
            cb(c.cx, c.cy - c.radius);
        }
    }
}

// Returns the nearest snap target to (x, y) within the screen-relative
// snap radius. Falls back to the nearest grid intersection when the grid
// is on and no shape target qualifies.
struct SnapHit { float x, y; bool found; };

SnapHit findSnap(float x, float y, const Selection* exclude = nullptr,
                 const SnapHit* prev = nullptr,
                 float releaseRViewPx = kSnapReleaseFastViewPx) {
    if (g_snapEnabled.load() == 0) return { x, y, false };

    // Convert the view-pixel target radius into doc-px at the current zoom
    // so snap stays the same on-screen distance regardless of view scale.
    // Use userViewScale() — currentViewScale() can be transiently clobbered
    // by GL-thread thumbnail/bucket-fill renders.
    float radiusDoc  = kSnapRadiusViewPx / userViewScale();
    float bestDist2  = radiusDoc * radiusDoc;
    SnapHit best     = { x, y, false };

    forEachShapeSnapTarget([&](float tx, float ty) {
        float dx = tx - x, dy = ty - y;
        float d2 = dx * dx + dy * dy;
        if (d2 < bestDist2) {
            bestDist2 = d2;
            best = { tx, ty, true };
        }
    }, exclude);

    // Grid intersections compete on the same closest-wins basis as
    // vector vertices; previously the grid was only tested when no
    // shape target was in range, which let a far-away line endpoint
    // win over a nearby cell corner — visible to the user as the
    // selection snapping to vector geometry many grid cells away
    // when the pen sat near a grid cell center.
    if (g_gridEnabled.load() != 0) {
        const float spacing = g_gridSpacing.load();
        float gx = std::round(x / spacing) * spacing;
        float gy = std::round(y / spacing) * spacing;
        float dx = gx - x, dy = gy - y;
        float gd2 = dx * dx + dy * dy;
        if (gd2 < bestDist2) {
            best     = { gx, gy, true };
            bestDist2 = gd2;
        }
    }

    // Hysteresis: when the previous frame's target is provided, keep
    // snapping to it unless a different candidate is meaningfully
    // closer. Absorbs both the EMR sensor's micro-jitter on a
    // "stationary" pen and slow hand drift when zoomed in.
    //   - Release radius is independent of entry — set in view-px so
    //     a locked snap stays locked through the user's normal hand
    //     drift, regardless of zoom.
    //   - Stickiness ratio: a new candidate has to be ≥30% closer
    //     (squared = ≥51%) than the previous target to win.
    if (prev != nullptr && prev->found) {
        float pdx = prev->x - x, pdy = prev->y - y;
        float pd2 = pdx * pdx + pdy * pdy;
        float releaseR = releaseRViewPx / userViewScale();
        if (pd2 < releaseR * releaseR) {
            if (best.found) {
                if (bestDist2 > pd2 * 0.49f) {
                    best = *prev;
                }
            } else {
                best = *prev;
            }
        }
    }
    return best;
}

// Forward decl; defined below in the shape-rendering helpers section.
void drawEllipseAsLines(float cx, float cy, float rx, float ry, float rotation,
                        uint32_t rgb, float width, float alpha);

// Draw a small ring + crosshair at (x, y) — the visual indicator for an
// active snap. Caller has already bound the line program.
void drawSnapMarker(float x, float y) {
    float r = vpxToDoc(kSnapMarkerRViewPx);
    float w = vpxToDoc(kSelectionOutlineWidthViewPx);
    drawEllipseAsLines(x, y, r, r, /*rotation*/ 0.0f,
                       kSnapMarkerColor, w, 1.0f);
    drawLineSegment(x - r, y, x + r, y, kSnapMarkerColor, w, 1.0f);
    drawLineSegment(x, y - r, x, y + r, kSnapMarkerColor, w, 1.0f);
}

// Draw a small square handle (4 line segments) centered at (x, y).
void drawHandle(float x, float y, float size, uint32_t color) {
    float h = size * 0.5f;
    float w = vpxToDoc(kSelectionOutlineWidthViewPx);
    drawLineSegment(x - h, y - h, x + h, y - h, color, w, 1.0f);
    drawLineSegment(x + h, y - h, x + h, y + h, color, w, 1.0f);
    drawLineSegment(x + h, y + h, x - h, y + h, color, w, 1.0f);
    drawLineSegment(x - h, y + h, x - h, y - h, color, w, 1.0f);
}

// Compute the world position of one of the four scale handles. handleIdx:
// 0 = top-left, 1 = top-right, 2 = bottom-right, 3 = bottom-left.
// For a Line (kind==Line) we emit only handles 0 (left endpoint) and 1
// (right endpoint), at the midpoints of the OBB's left/right sides.
void scaleHandlePosition(const Obb& o, ShapeKind kind, int handleIdx,
                         float& outX, float& outY) {
    float lx = 0.0f, ly = 0.0f;
    if (kind == ShapeKind::Line) {
        lx = (handleIdx == 0) ? -o.hw : +o.hw;
        ly = 0.0f;
    } else {
        lx = (handleIdx == 0 || handleIdx == 3) ? -o.hw : +o.hw;
        ly = (handleIdx == 0 || handleIdx == 1) ? -o.hh : +o.hh;
    }
    rotateLocalToWorld(o, lx, ly, outX, outY);
}

// Rotate handle is offset above the OBB top-center. `uiThread` selects
// which view-scale mirror the offset is derived from: hit-tests run on
// the UI thread and must use userViewScale() (see vpxToDocUi), render
// runs on the GL thread and uses currentViewScale(). Getting this wrong
// puts the hit-test's notion of the handle hundreds of doc-px away from
// where it was drawn.
void rotateHandlePosition(const Obb& o, float& outX, float& outY,
                          bool uiThread = false) {
    float off = uiThread ? vpxToDocUi(kRotateHandleOffsetViewPx)
                         : vpxToDoc(kRotateHandleOffsetViewPx);
    rotateLocalToWorld(o, 0.0f, -o.hh - off, outX, outY);
}

// Number of scale handles for the shape (2 for Line, 4 for others).
int scaleHandleCount(ShapeKind kind) {
    return (kind == ShapeKind::Line) ? 2 : 4;
}

// Render the OBB outline + 4 scale handles + rotate handle for an
// arbitrary Obb (used by the floating raster selection, which has no
// owning shape). Caller must have the line program + page-clip-off
// uniforms bound.
void renderObbWithHandles(const Obb& obb) {
    float outlineW = vpxToDoc(kSelectionOutlineWidthViewPx);
    float handleSz = vpxToDoc(kHandleSizeViewPx);

    float c0x, c0y, c1x, c1y, c2x, c2y, c3x, c3y;
    rotateLocalToWorld(obb, -obb.hw, -obb.hh, c0x, c0y);
    rotateLocalToWorld(obb, +obb.hw, -obb.hh, c1x, c1y);
    rotateLocalToWorld(obb, +obb.hw, +obb.hh, c2x, c2y);
    rotateLocalToWorld(obb, -obb.hw, +obb.hh, c3x, c3y);
    drawLineSegment(c0x, c0y, c1x, c1y, kHandleColor, outlineW, 1.0f);
    drawLineSegment(c1x, c1y, c2x, c2y, kHandleColor, outlineW, 1.0f);
    drawLineSegment(c2x, c2y, c3x, c3y, kHandleColor, outlineW, 1.0f);
    drawLineSegment(c3x, c3y, c0x, c0y, kHandleColor, outlineW, 1.0f);

    // 4 corner scale handles. Reuse scaleHandlePosition with kind=Rect.
    for (int i = 0; i < 4; ++i) {
        float hx, hy;
        scaleHandlePosition(obb, ShapeKind::Rect, i, hx, hy);
        drawHandle(hx, hy, handleSz, kHandleColor);
    }

    // Rotate handle: tether line from top-edge midpoint, then handle.
    float anchorX, anchorY;
    rotateLocalToWorld(obb, 0.0f, -obb.hh, anchorX, anchorY);
    float rhX, rhY;
    rotateHandlePosition(obb, rhX, rhY);
    drawLineSegment(anchorX, anchorY, rhX, rhY,
                    kHandleColor, outlineW, 1.0f);
    drawHandle(rhX, rhY, handleSz, kHandleColor);
}

// Renders the OBB outline + scale handles + rotate handle for the given
// selection. Caller must already have the line program / VAO bound.
void renderSelectionOverlay(const Selection& sel) {
    Obb obb;
    if (!obbForSelection(sel, obb)) return;

    float outlineW = vpxToDoc(kSelectionOutlineWidthViewPx);
    float handleSz = vpxToDoc(kHandleSizeViewPx);

    // OBB outline — 4 corners of the box (rotated).
    float c0x, c0y, c1x, c1y, c2x, c2y, c3x, c3y;
    rotateLocalToWorld(obb, -obb.hw, -obb.hh, c0x, c0y);
    rotateLocalToWorld(obb, +obb.hw, -obb.hh, c1x, c1y);
    rotateLocalToWorld(obb, +obb.hw, +obb.hh, c2x, c2y);
    rotateLocalToWorld(obb, -obb.hw, +obb.hh, c3x, c3y);
    drawLineSegment(c0x, c0y, c1x, c1y, kHandleColor, outlineW, 1.0f);
    drawLineSegment(c1x, c1y, c2x, c2y, kHandleColor, outlineW, 1.0f);
    drawLineSegment(c2x, c2y, c3x, c3y, kHandleColor, outlineW, 1.0f);
    drawLineSegment(c3x, c3y, c0x, c0y, kHandleColor, outlineW, 1.0f);

    // Scale handles.
    int n = scaleHandleCount(sel.kind);
    for (int i = 0; i < n; ++i) {
        float hx, hy;
        scaleHandlePosition(obb, sel.kind, i, hx, hy);
        drawHandle(hx, hy, handleSz, kHandleColor);
    }

    // Rotate handle (skip for circles — rotation is invariant).
    if (sel.kind != ShapeKind::Circle) {
        float anchorX, anchorY;
        rotateLocalToWorld(obb, 0.0f, -obb.hh, anchorX, anchorY);
        float rhX, rhY;
        rotateHandlePosition(obb, rhX, rhY);
        drawLineSegment(anchorX, anchorY, rhX, rhY,
                        kHandleColor, outlineW, 1.0f);
        drawHandle(rhX, rhY, handleSz, kHandleColor);
    }
}

// Helper: rotate (lx, ly) around (cx, cy) by `rotation` radians.
inline void rotateAround(float cx, float cy, float lx, float ly, float rotation,
                         float& outX, float& outY) {
    float c = std::cos(rotation), s = std::sin(rotation);
    float dx = lx - cx, dy = ly - cy;
    outX = cx + dx * c - dy * s;
    outY = cy + dx * s + dy * c;
}

void drawRectangleAsLines(float x0, float y0, float x1, float y1, float rotation,
                          uint32_t rgb, float width, float alpha) {
    float cx = (x0 + x1) * 0.5f;
    float cy = (y0 + y1) * 0.5f;
    float ax, ay, bx, by, cx_, cy_, dx, dy;
    rotateAround(cx, cy, x0, y0, rotation, ax, ay);
    rotateAround(cx, cy, x1, y0, rotation, bx, by);
    rotateAround(cx, cy, x1, y1, rotation, cx_, cy_);
    rotateAround(cx, cy, x0, y1, rotation, dx, dy);
    drawLineSegment(ax, ay, bx, by, rgb, width, alpha);
    drawLineSegment(bx, by, cx_, cy_, rgb, width, alpha);
    drawLineSegment(cx_, cy_, dx, dy, rgb, width, alpha);
    drawLineSegment(dx, dy, ax, ay, rgb, width, alpha);
}

void drawEllipseAsLines(float cx, float cy, float rx, float ry, float rotation,
                        uint32_t rgb, float width, float alpha) {
    constexpr float kTau = 6.283185307179586f;
    // Adaptive segment count: target ~3 view-pixels per chord so the
    // polygonization stays imperceptible at any zoom. Use the larger
    // semi-axis as a perimeter proxy (upper-bounds the true Ramanujan
    // perimeter and is good enough for picking a segment count). Use
    // currentViewScale so this scales correctly for thumbnail renders
    // too — those use a transient render scale.
    constexpr float kTargetChordViewPx = 3.0f;
    constexpr int   kMinSegments       = 24;
    constexpr int   kMaxSegments       = 256;
    float maxRadiusView = std::max(std::fabs(rx), std::fabs(ry))
                          * currentViewScale();
    float perimViewUpper = kTau * maxRadiusView;
    int segments = static_cast<int>(perimViewUpper / kTargetChordViewPx);
    if (segments < kMinSegments) segments = kMinSegments;
    if (segments > kMaxSegments) segments = kMaxSegments;

    float c = std::cos(rotation), s = std::sin(rotation);
    auto pointAt = [&](float a, float& x, float& y) {
        float lx = std::cos(a) * rx;
        float ly = std::sin(a) * ry;
        x = cx + lx * c - ly * s;
        y = cy + lx * s + ly * c;
    };
    float prevX, prevY;
    pointAt(0.0f, prevX, prevY);
    for (int i = 1; i <= segments; ++i) {
        float a = (float)i / (float)segments * kTau;
        float x, y;
        pointAt(a, x, y);
        drawLineSegment(prevX, prevY, x, y, rgb, width, alpha);
        prevX = x;
        prevY = y;
    }
}

// ---- Selection helpers ---------------------------------------------------

constexpr float kSelectionHaloPad = 3.0f;            // extra half-width
constexpr uint32_t kSelectionHaloColor = 0xFF8030u;  // orange
constexpr float kSelectionHaloAlpha = 0.55f;
constexpr float kHitThresholdPadViewPx = 6.0f;       // tap tolerance, view px

// Convert the view-pixel hit-test pad to a doc-pixel pad at the current
// zoom level, so tapping near a shape outline keeps the same on-screen
// tolerance regardless of how zoomed-in the user is.
inline float hitThresholdPadDoc() {
    return kHitThresholdPadViewPx / userViewScale();
}

inline bool isShapeSelected(const Selection& sel, size_t layerIdx,
                            ShapeKind kind, size_t shapeIdx) {
    return sel.kind == kind && sel.layerIdx == layerIdx
        && sel.shapeIdx == shapeIdx;
}

float distToSegment(float px, float py,
                    float ax, float ay, float bx, float by) {
    float abx = bx - ax, aby = by - ay;
    float apx = px - ax, apy = py - ay;
    float ab2 = abx*abx + aby*aby;
    float t = (ab2 > 1e-6f) ? (apx*abx + apy*aby) / ab2 : 0.0f;
    if (t < 0.0f) t = 0.0f; else if (t > 1.0f) t = 1.0f;
    float cx = ax + abx * t, cy = ay + aby * t;
    float dx = px - cx, dy = py - cy;
    return std::sqrt(dx*dx + dy*dy);
}

float distToRectOutline(float px, float py,
                        float x0, float y0, float x1, float y1) {
    float d1 = distToSegment(px, py, x0, y0, x1, y0);
    float d2 = distToSegment(px, py, x1, y0, x1, y1);
    float d3 = distToSegment(px, py, x1, y1, x0, y1);
    float d4 = distToSegment(px, py, x0, y1, x0, y0);
    return std::min(std::min(d1, d2), std::min(d3, d4));
}

float distToCircleOutline(float px, float py,
                          float cx, float cy, float r) {
    float dx = px - cx, dy = py - cy;
    return std::fabs(std::sqrt(dx*dx + dy*dy) - r);
}

// Approximate distance from point to ellipse outline. Scales to a unit
// circle, computes distance there, then scales back by the average
// radius. Good enough for tap-to-select.
float distToEllipseOutline(float px, float py,
                           float cx, float cy, float rx, float ry) {
    if (rx < 1e-3f || ry < 1e-3f) return 1e9f;
    float dx = (px - cx) / rx, dy = (py - cy) / ry;
    float lenScaled = std::sqrt(dx*dx + dy*dy);
    float avg = (rx + ry) * 0.5f;
    return std::fabs(lenScaled - 1.0f) * avg;
}

// ---- Marquee outline-intersection helpers --------------------------------
//
// Used by endInteraction's Marquee branch to decide which vector shapes a
// drag-rectangle picks up. Semantics: "touch outline only" — a marquee
// fully inside a large rectangle does NOT select it; only marquees that
// actually overlap painted strokes do.

// Liang-Barsky: does segment (ax,ay)->(bx,by) overlap AABB [x0,x1]x[y0,y1]?
bool segmentIntersectsAabb(float ax, float ay, float bx, float by,
                           float x0, float y0, float x1, float y1) {
    if (ax >= x0 && ax <= x1 && ay >= y0 && ay <= y1) return true;
    if (bx >= x0 && bx <= x1 && by >= y0 && by <= y1) return true;
    float dx = bx - ax, dy = by - ay;
    float t0 = 0.0f, t1 = 1.0f;
    float p[4] = { -dx,  dx, -dy,  dy };
    float q[4] = { ax - x0, x1 - ax, ay - y0, y1 - ay };
    for (int i = 0; i < 4; ++i) {
        if (std::fabs(p[i]) < 1e-9f) {
            if (q[i] < 0.0f) return false;
        } else {
            float r = q[i] / p[i];
            if (p[i] < 0.0f) {
                if (r > t1) return false;
                if (r > t0) t0 = r;
            } else {
                if (r < t0) return false;
                if (r < t1) t1 = r;
            }
        }
    }
    return t0 <= t1;
}

// Fat segment (capsule with square caps) vs AABB: expand the AABB by the
// stroke's half-width and run the segment test. Slight over-acceptance at
// the rounded endpoints, well below what users can perceive.
inline bool fatSegmentIntersectsAabb(float ax, float ay, float bx, float by,
                                     float halfWidth,
                                     float x0, float y0,
                                     float x1, float y1) {
    return segmentIntersectsAabb(ax, ay, bx, by,
                                 x0 - halfWidth, y0 - halfWidth,
                                 x1 + halfWidth, y1 + halfWidth);
}

// Rotated-rect outline (4 fat edges of the OBB) vs AABB.
bool rectOutlineIntersectsAabb(const Rect& r,
                               float x0, float y0, float x1, float y1) {
    float cx = (r.x0 + r.x1) * 0.5f, cy = (r.y0 + r.y1) * 0.5f;
    float hw = std::fabs(r.x1 - r.x0) * 0.5f;
    float hh = std::fabs(r.y1 - r.y0) * 0.5f;
    float c = std::cos(r.rotation), si = std::sin(r.rotation);
    // Corners in traversal order: TL, TR, BR, BL.
    auto corner = [&](float sx, float sy, float& ox, float& oy) {
        float lx = sx * hw, ly = sy * hh;
        ox = cx + lx * c - ly * si;
        oy = cy + lx * si + ly * c;
    };
    float qx[4], qy[4];
    corner(-1.0f, -1.0f, qx[0], qy[0]);
    corner( 1.0f, -1.0f, qx[1], qy[1]);
    corner( 1.0f,  1.0f, qx[2], qy[2]);
    corner(-1.0f,  1.0f, qx[3], qy[3]);
    float halfW = r.width * 0.5f;
    for (int i = 0; i < 4; ++i) {
        int j = (i + 1) & 3;
        if (fatSegmentIntersectsAabb(qx[i], qy[i], qx[j], qy[j],
                                     halfW, x0, y0, x1, y1)) return true;
    }
    return false;
}

// Ellipse outline (rotated) sampled as a 64-segment polyline vs AABB.
// 64 keeps the worst-case chord error well under a stroke width at all
// realistic sizes (single-shape, single-marquee-end-of-drag — perf is fine).
bool ellipseOutlineIntersectsAabb(float cx, float cy, float rx, float ry,
                                  float rotation, float halfWidth,
                                  float x0, float y0,
                                  float x1, float y1) {
    constexpr int N = 64;
    float c = std::cos(rotation), si = std::sin(rotation);
    float prevX = cx + rx * c;
    float prevY = cy + rx * si;
    for (int i = 1; i <= N; ++i) {
        float ang = (float)i * (2.0f * 3.14159265358979323846f / N);
        float lx = rx * std::cos(ang), ly = ry * std::sin(ang);
        float px = cx + lx * c - ly * si;
        float py = cy + lx * si + ly * c;
        if (fatSegmentIntersectsAabb(prevX, prevY, px, py, halfWidth,
                                     x0, y0, x1, y1)) return true;
        prevX = px; prevY = py;
    }
    return false;
}

// Circle outline vs AABB, closed form. The outline crosses or touches the
// AABB iff the nearest point on the AABB to the center is within
// (radius + halfWidth) AND the farthest point on the AABB is at least
// (radius - halfWidth) away — i.e. the AABB straddles the ring.
bool circleOutlineIntersectsAabb(float cx, float cy, float radius,
                                 float halfWidth,
                                 float x0, float y0, float x1, float y1) {
    float ccx = cx < x0 ? x0 : (cx > x1 ? x1 : cx);
    float ccy = cy < y0 ? y0 : (cy > y1 ? y1 : cy);
    float ndx = cx - ccx, ndy = cy - ccy;
    float minD2 = ndx*ndx + ndy*ndy;
    float outer = radius + halfWidth;
    if (minD2 > outer * outer) return false;
    float fdx = std::fmax(std::fabs(cx - x0), std::fabs(cx - x1));
    float fdy = std::fmax(std::fabs(cy - y0), std::fabs(cy - y1));
    float maxD2 = fdx*fdx + fdy*fdy;
    float inner = radius - halfWidth;
    if (inner > 0.0f && maxD2 < inner * inner) return false;
    return true;
}

// Per-shape outline intersection test for the marquee. AABB pre-reject
// keeps the loop fast when there are lots of shapes far from the drag.
bool shapeOutlineIntersectsMarquee(const ShapeData& s,
                                   float mx0, float my0,
                                   float mx1, float my1) {
    DocBbox bb = shapeAabb(s);
    if (bb.maxX < mx0 || bb.minX > mx1
     || bb.maxY < my0 || bb.minY > my1) return false;
    switch (s.kind) {
        case ShapeKind::Line: {
            const Line& l = s.line;
            return fatSegmentIntersectsAabb(l.x0, l.y0, l.x1, l.y1,
                                            l.width * 0.5f,
                                            mx0, my0, mx1, my1);
        }
        case ShapeKind::Rect:
            return rectOutlineIntersectsAabb(s.rect, mx0, my0, mx1, my1);
        case ShapeKind::Ellipse: {
            const Ellipse& e = s.ellipse;
            return ellipseOutlineIntersectsAabb(e.cx, e.cy, e.rx, e.ry,
                                                e.rotation, e.width * 0.5f,
                                                mx0, my0, mx1, my1);
        }
        case ShapeKind::Circle: {
            const Circle& c = s.circle;
            return circleOutlineIntersectsAabb(c.cx, c.cy, c.radius,
                                               c.width * 0.5f,
                                               mx0, my0, mx1, my1);
        }
        default:
            return false;
    }
}

// Hit-test the active layer at (x, y); on hit, set g_selection. Returns
// true if a shape was selected, false if no shape was hit (and clears
// any prior selection). Searches in render order so the topmost shape
// wins on overlap.
bool hitTestActiveVectorLayer(float x, float y) {
    if (activeLayer() >= layers().size() || !layers()[activeLayer()]) return false;
    Layer& layer = *layers()[activeLayer()];
    if (layer.type != LayerType::Vector) return false;

    ShapeKind hitKind = ShapeKind::None;
    size_t    hitIdx  = 0;
    float     pad     = hitThresholdPadDoc();

    for (size_t i = 0; i < layer.lines.size(); ++i) {
        const auto& l = layer.lines[i];
        float thr = l.width * 0.5f + pad;
        if (distToSegment(x, y, l.x0, l.y0, l.x1, l.y1) <= thr) {
            hitKind = ShapeKind::Line; hitIdx = i;
        }
    }
    for (size_t i = 0; i < layer.rects.size(); ++i) {
        const auto& r = layer.rects[i];
        float thr = r.width * 0.5f + pad;
        if (distToRectOutline(x, y, r.x0, r.y0, r.x1, r.y1) <= thr) {
            hitKind = ShapeKind::Rect; hitIdx = i;
        }
    }
    for (size_t i = 0; i < layer.ellipses.size(); ++i) {
        const auto& e = layer.ellipses[i];
        float thr = e.width * 0.5f + pad;
        if (distToEllipseOutline(x, y, e.cx, e.cy, e.rx, e.ry) <= thr) {
            hitKind = ShapeKind::Ellipse; hitIdx = i;
        }
    }
    for (size_t i = 0; i < layer.circles.size(); ++i) {
        const auto& c = layer.circles[i];
        float thr = c.width * 0.5f + pad;
        if (distToCircleOutline(x, y, c.cx, c.cy, c.radius) <= thr) {
            hitKind = ShapeKind::Circle; hitIdx = i;
        }
    }

    std::lock_guard<std::mutex> lock(g_selectionMutex);
    // Single-tap selection always replaces any prior multi-selection;
    // marquee is the dedicated path for picking up multiple shapes.
    g_extraSelections.clear();
    if (hitKind == ShapeKind::None) {
        g_selection = Selection{};
        return false;
    }
    g_selection.kind     = hitKind;
    g_selection.layerIdx = activeLayer();
    g_selection.shapeIdx = hitIdx;
    return true;
}

void compositeVectorLayer(JNIEnv* env, const Layer& layer, size_t layerIdx,
                          jint width, jint height, jfloatArray transform) {
    if (layer.lines.empty() && layer.rects.empty()
        && layer.ellipses.empty() && layer.circles.empty()) return;

    // Snapshot selection for this composite pass. Includes both the
    // primary single-select and any extra (marquee'd) selections so
    // every selected shape gets a halo.
    Selection sel;
    std::vector<Selection> extras;
    {
        std::lock_guard<std::mutex> lock(g_selectionMutex);
        sel    = g_selection;
        extras = g_extraSelections;
    }
    auto inAnySel = [&](ShapeKind kind, size_t i) -> bool {
        if (isShapeSelected(sel, layerIdx, kind, i)) return true;
        for (const auto& e : extras) {
            if (isShapeSelected(e, layerIdx, kind, i)) return true;
        }
        return false;
    };

    glUseProgram(g_lineProg.program);
    glBindVertexArray(g_quadVao);
    uploadMat4(env, g_lineProg.uTransform, transform);
    glUniform2f(g_lineProg.uScreen, (float)width, (float)height);
    // Vector content gets the same page-clip treatment as raster strokes.
    PageClip pageClip = readPageClip();
    uploadPageClip(g_lineProg.uPageMin, g_lineProg.uPageMax,
                   g_lineProg.uPageActive, pageClip);
    glUniform1f(g_lineProg.uOpacity,
                layer.opacity.load(std::memory_order_relaxed));

    // VALUE COPIES (not references) — applyMoveTo / applyScaleTo /
    // applyRotateTo run on the UI thread and mutate shape coordinates
    // without locking. If we kept references, the halo and the line
    // could read different values for x0/y0/x1/y1 within a single
    // render frame, producing a visible offset between the halo and
    // the stroke (the user reported this as "vibrating snap"). A
    // local copy snapshots the shape's geometry per iteration so both
    // draws come from the same data.
    for (size_t i = 0; i < layer.lines.size(); ++i) {
        const Line l = layer.lines[i];
        if (inAnySel(ShapeKind::Line, i)) {
            drawLineSegment(l.x0, l.y0, l.x1, l.y1,
                            kSelectionHaloColor, l.width + kSelectionHaloPad * 2.0f,
                            kSelectionHaloAlpha);
        }
        drawLineSegment(l.x0, l.y0, l.x1, l.y1, l.color, l.width, 1.0f);
    }
    // Rectangles use the box-SDF program — single quad, no segment-cap
    // overlap at the corners.
    if (!layer.rects.empty()) {
        glUseProgram(g_rectProg.program);
        uploadMat4(env, g_rectProg.uTransform, transform);
        glUniform2f(g_rectProg.uScreen, (float)width, (float)height);
        uploadPageClip(g_rectProg.uPageMin, g_rectProg.uPageMax,
                       g_rectProg.uPageActive, pageClip);
        glUniform1f(g_rectProg.uOpacity,
                    layer.opacity.load(std::memory_order_relaxed));
        for (size_t i = 0; i < layer.rects.size(); ++i) {
            const Rect r = layer.rects[i];
            if (inAnySel(ShapeKind::Rect, i)) {
                drawRectOutline(r.x0, r.y0, r.x1, r.y1, r.rotation,
                                kSelectionHaloColor,
                                r.width + kSelectionHaloPad * 2.0f,
                                kSelectionHaloAlpha);
            }
            drawRectOutline(r.x0, r.y0, r.x1, r.y1, r.rotation,
                            r.color, r.width, 1.0f);
        }
        glUniform1f(g_rectProg.uOpacity, 1.0f);
    }
    // Ellipses and circles use the SDF program — one quad per shape with
    // no segment junctions, so the rendered alpha is uniform regardless
    // of the layer's opacity.
    if (!layer.ellipses.empty() || !layer.circles.empty()) {
        glUseProgram(g_ellipseProg.program);
        uploadMat4(env, g_ellipseProg.uTransform, transform);
        glUniform2f(g_ellipseProg.uScreen, (float)width, (float)height);
        uploadPageClip(g_ellipseProg.uPageMin, g_ellipseProg.uPageMax,
                       g_ellipseProg.uPageActive, pageClip);
        glUniform1f(g_ellipseProg.uOpacity,
                    layer.opacity.load(std::memory_order_relaxed));

        for (size_t i = 0; i < layer.ellipses.size(); ++i) {
            const Ellipse e = layer.ellipses[i];
            if (inAnySel(ShapeKind::Ellipse, i)) {
                drawEllipseOutline(e.cx, e.cy, e.rx, e.ry, e.rotation,
                                   kSelectionHaloColor,
                                   e.width + kSelectionHaloPad * 2.0f,
                                   kSelectionHaloAlpha);
            }
            drawEllipseOutline(e.cx, e.cy, e.rx, e.ry, e.rotation,
                               e.color, e.width, 1.0f);
        }
        for (size_t i = 0; i < layer.circles.size(); ++i) {
            const Circle c = layer.circles[i];
            if (inAnySel(ShapeKind::Circle, i)) {
                drawEllipseOutline(c.cx, c.cy, c.radius, c.radius,
                                   /*rotation*/ 0.0f, kSelectionHaloColor,
                                   c.width + kSelectionHaloPad * 2.0f,
                                   kSelectionHaloAlpha);
            }
            drawEllipseOutline(c.cx, c.cy, c.radius, c.radius,
                               /*rotation*/ 0.0f, c.color, c.width, 1.0f);
        }

        // Same uOpacity reset as for the line program: keep the default
        // 1.0 for non-layer callers (snap marker, shape preview).
        glUniform1f(g_ellipseProg.uOpacity, 1.0f);
    }

    // Reset opacity so the next non-layer use of the line program
    // (handles, snap markers, page outline, shape preview) doesn't
    // inherit this layer's value.
    glUseProgram(g_lineProg.program);
    glUniform1f(g_lineProg.uOpacity, 1.0f);
}

void compositeAllLayers(JNIEnv* env, jint width, jint height,
                        jfloatArray transform) {
    ATRACE_SCOPE("DrawingApp.compositeAllLayers");
    glViewport(0, 0, width, height);

    PageClip pageClip = readPageClip();

    {
        ATRACE_SCOPE("DrawingApp.compositeAllLayers.clearAndGrid");
    if (pageClip.active) {
        // Off-canvas background: light gray clear, then paint paper-white
        // over the page rectangle. With page disabled, fall back to a
        // single paper-white clear (the original behavior).
        glDisable(GL_BLEND);
        // Warm beige, a touch lighter than the menus' paper (#F5F0E6).
        // Matches the eyedropper's off-page fallback below.
        glClearColor(0xFB / 255.0f, 0xF7 / 255.0f, 0xEE / 255.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(g_fill.program);
        glBindVertexArray(g_quadVao);
        uploadMat4(env, g_fill.uTransform, transform);
        glUniform2f(g_fill.uScreen, (float)width, (float)height);
        glUniform2f(g_fill.uMin, pageClip.minX, pageClip.minY);
        glUniform2f(g_fill.uMax, pageClip.maxX, pageClip.maxY);
        glUniform4f(g_fill.uFillColor, 1.0f, 1.0f, 1.0f, 1.0f);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    } else {
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
    }

    // Grid is part of the page background — between the paper-white clear
    // and the layer tiles, so user strokes naturally occlude it. Grid
    // shader discards fragments outside the page when active.
    {
        glUseProgram(g_grid.program);
        uploadPageClip(g_grid.uPageMin, g_grid.uPageMax,
                       g_grid.uPageActive, pageClip);
    }
    renderGridOverlay(env, width, height, transform);
    renderMarginRules(env, width, height, transform, pageClip);

    // Page-boundary rectangle (anchor for the user when zoomed/rotated).
    // Drawn under the layers so strokes that cross the boundary occlude
    // it locally while the rest stays visible. The outline itself must
    // NOT be page-clipped — that would invisibly trim its own edges.
    // Skipped by the export path (renderPageThumbnail with drawChrome=
    // false) so the saved image isn't bordered.
    if (pageClip.active && !g_skipPageOutline) {
        glUseProgram(g_lineProg.program);
        glBindVertexArray(g_quadVao);
        uploadMat4(env, g_lineProg.uTransform, transform);
        glUniform2f(g_lineProg.uScreen, (float)width, (float)height);
        uploadPageClip(g_lineProg.uPageMin, g_lineProg.uPageMax,
                       g_lineProg.uPageActive,
                       PageClip{false, 0, 0, 0, 0});
        constexpr uint32_t kPageColor = 0x808890u;     // medium gray
        constexpr float    kPageAlpha = 0.85f;
        constexpr float    kPageWidthViewPx = 1.5f;
        float w = vpxToDoc(kPageWidthViewPx);
        drawLineSegment(pageClip.minX, pageClip.minY,
                        pageClip.maxX, pageClip.minY, kPageColor, w, kPageAlpha);
        drawLineSegment(pageClip.maxX, pageClip.minY,
                        pageClip.maxX, pageClip.maxY, kPageColor, w, kPageAlpha);
        drawLineSegment(pageClip.maxX, pageClip.maxY,
                        pageClip.minX, pageClip.maxY, kPageColor, w, kPageAlpha);
        drawLineSegment(pageClip.minX, pageClip.maxY,
                        pageClip.minX, pageClip.minY, kPageColor, w, kPageAlpha);
        glBindVertexArray(0);
    }
    }  // close clearAndGrid ATRACE_SCOPE

    if (layers().empty()) return;

    // Premultiplied blend is the global default; both raster tiles and
    // vector lines composite correctly under it.
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

    bindRasterCompositePipeline(env, width, height, transform);

    {
        ATRACE_SCOPE("DrawingApp.compositeAllLayers.layerLoop");
        for (size_t i = 0; i < layers().size(); ++i) {
            const auto& layer = layers()[i];
            if (!layer) continue;
            if (!layer->visible.load(std::memory_order_relaxed)) continue;
            if (layer->type == LayerType::Raster) {
                compositeRasterLayer(*layer);
            } else { // Vector
                compositeVectorLayer(env, *layer, i, width, height, transform);
                // Switch back to raster pipeline for the next raster layer.
                bindRasterCompositePipeline(env, width, height, transform);
            }
        }
    }

    // Pixel-grid overlay sits on top of the layers so the user can see
    // individual pixel boundaries when zoomed in. Internally gated on
    // view scale, so it's a no-op at low zoom.
    renderPixelGridOverlay(env, width, height, transform);

    ATRACE_SCOPE("DrawingApp.compositeAllLayers.selAndChrome");
    // Floating raster selection (if any) — drawn over its owner layer's
    // composite so the lifted pixels appear at their currently-translated
    // position. The owner layer's tiles already have a hole where the
    // selection was lifted, so this overlay completes the visible image.
    {
        bool   selActive = false;
        size_t selLayerIdx = 0;
        Obb    selObb{};
        GLuint contentTex = 0;
        {
            std::lock_guard<std::mutex> lock(g_rasterSelMutex);
            if (g_rasterSel.active) {
                selActive   = true;
                selLayerIdx = g_rasterSel.layerIdx;
                selObb      = obbForRasterSelection(g_rasterSel);
                contentTex  = g_rasterSel.contentTex;
            }
        }
        if (selActive && contentTex != 0) {
            // Inherit the source layer's opacity so the floating
            // selection visually matches the rest of its layer (the
            // tiles around the lift-hole composite at the layer's
            // opacity, so a 100%-opacity overlay would pop). Bake
            // (commit drop) sets uOpacity = 1.0 explicitly below.
            float layerOpacity = 1.0f;
            if (selLayerIdx < layers().size() && layers()[selLayerIdx]) {
                layerOpacity = layers()[selLayerIdx]->opacity
                    .load(std::memory_order_relaxed);
            }
            // Compute the four placement corners in doc-coords.
            float c0x, c0y, c1x, c1y, c2x, c2y, c3x, c3y;
            rotateLocalToWorld(selObb, -selObb.hw, -selObb.hh, c0x, c0y);
            rotateLocalToWorld(selObb, +selObb.hw, -selObb.hh, c1x, c1y);
            rotateLocalToWorld(selObb, +selObb.hw, +selObb.hh, c2x, c2y);
            rotateLocalToWorld(selObb, -selObb.hw, +selObb.hh, c3x, c3y);
            glUseProgram(g_sel.program);
            glBindVertexArray(g_quadVao);
            uploadMat4(env, g_sel.uTransform, transform);
            glUniform2f(g_sel.uScreen, (float)width, (float)height);
            glUniform2f(g_sel.uC0, c0x, c0y);
            glUniform2f(g_sel.uC1, c1x, c1y);
            glUniform2f(g_sel.uC2, c2x, c2y);
            glUniform2f(g_sel.uC3, c3x, c3y);
            glUniform1f(g_sel.uOpacity, layerOpacity);
            // Page-clip in doc-coords: matches the dab/line/grid pattern,
            // so the off-page portion is fragment-discarded rather than
            // squashing the geometry.
            uploadPageClip(g_sel.uPageMin, g_sel.uPageMax,
                           g_sel.uPageActive, readPageClip());
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, contentTex);
            glUniform1i(g_sel.uContent, 0);
            glEnable(GL_BLEND);
            glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
            glBindTexture(GL_TEXTURE_2D, 0);
        }
    }

    // Selection overlay (OBB + handles) — drawn on top of everything so
    // it's visible regardless of which layer the selected shape lives on.
    Selection sel;
    bool  hasExtras  = false;
    bool  snapActive = false;
    bool  marqueeActive = false;
    float snapMx = 0.0f, snapMy = 0.0f;
    float mqx0 = 0, mqy0 = 0, mqx1 = 0, mqy1 = 0;
    {
        std::lock_guard<std::mutex> lock(g_selectionMutex);
        sel = g_selection;
        hasExtras = !g_extraSelections.empty();
        snapActive = g_drag.snapActive;
        snapMx = g_drag.snapX;
        snapMy = g_drag.snapY;
        marqueeActive = g_marqueeActive;
        mqx0 = g_marqueeX0; mqy0 = g_marqueeY0;
        mqx1 = g_marqueeX1; mqy1 = g_marqueeY1;
    }
    bool rasterSelActive = false;
    Obb  rasterSelObb{};
    {
        std::lock_guard<std::mutex> lock(g_rasterSelMutex);
        if (g_rasterSel.active) {
            rasterSelActive = true;
            rasterSelObb = obbForRasterSelection(g_rasterSel);
        }
    }
    if (sel.kind != ShapeKind::None || rasterSelActive || marqueeActive) {
        glUseProgram(g_lineProg.program);
        glBindVertexArray(g_quadVao);
        uploadMat4(env, g_lineProg.uTransform, transform);
        glUniform2f(g_lineProg.uScreen, (float)width, (float)height);
        // Selection handles & snap marker are UI affordances — must stay
        // visible even when the selected shape is right at the page edge.
        uploadPageClip(g_lineProg.uPageMin, g_lineProg.uPageMax,
                       g_lineProg.uPageActive,
                       PageClip{false, 0, 0, 0, 0});
        // Show transform handles only when a SINGLE shape is selected;
        // multi-select v1 supports move + delete only, so per-shape
        // OBB handles wouldn't make sense for the group.
        if (sel.kind != ShapeKind::None && !hasExtras) {
            renderSelectionOverlay(sel);
        }
        if (rasterSelActive) {
            renderObbWithHandles(rasterSelObb);
        }
        if (snapActive) {
            drawSnapMarker(snapMx, snapMy);
        }
        if (marqueeActive) {
            // Marquee rect — orange dashed-ish outline. Use the
            // selection-handle color so it reads as a UI affordance.
            float w = vpxToDoc(kSelectionOutlineWidthViewPx);
            float lx = std::min(mqx0, mqx1), rx = std::max(mqx0, mqx1);
            float ty = std::min(mqy0, mqy1), by = std::max(mqy0, mqy1);
            drawLineSegment(lx, ty, rx, ty, kHandleColor, w, 1.0f);
            drawLineSegment(rx, ty, rx, by, kHandleColor, w, 1.0f);
            drawLineSegment(rx, by, lx, by, kHandleColor, w, 1.0f);
            drawLineSegment(lx, by, lx, ty, kHandleColor, w, 1.0f);
        }
    }

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindVertexArray(0);

    // Eyedropper sample. Reads tile FBOs directly (source of truth) and
    // composites raster layers per-pixel. This avoids glReadPixels on
    // the framework's multi-buffer, which on some Android GPU drivers
    // returns stale or empty pixels even when the visible composite is
    // correct.
    if (g_pendingSampleHasReq.exchange(0)) {
        int xBits = g_pendingSampleDocXBits.load();
        int yBits = g_pendingSampleDocYBits.load();
        float dx, dy;
        std::memcpy(&dx, &xBits, sizeof(float));
        std::memcpy(&dy, &yBits, sizeof(float));
        int idx = static_cast<int>(std::floor(dx));
        int idy = static_cast<int>(std::floor(dy));
        // Floor-divide to handle negative doc coordinates correctly.
        int tx = (idx >= 0) ? (idx / kTileSize) : ((idx - (kTileSize - 1)) / kTileSize);
        int ty = (idy >= 0) ? (idy / kTileSize) : ((idy - (kTileSize - 1)) / kTileSize);
        int subX = idx - tx * kTileSize;          // [0, 256)
        int subY = idy - ty * kTileSize;          // [0, 256), top-down doc

        // Page background: paper-white inside the page rect, off-canvas
        // gray outside (matches compositeAllLayers' clear / paper-fill).
        PageClip page = readPageClip();
        bool insidePage = !page.active
            || (dx >= page.minX && dx < page.maxX
                && dy >= page.minY && dy < page.maxY);
        // Off-page beige matches compositeAllLayers' clear color above.
        float accR = insidePage ? 1.0f : (0xF8 / 255.0f);
        float accG = insidePage ? 1.0f : (0xF3 / 255.0f);
        float accB = insidePage ? 1.0f : (0xE8 / 255.0f);

        // Save FBO bindings so reading tile FBOs doesn't disturb the
        // multi-buffer state for whatever runs after this.
        GLint prevDrawFbo = 0, prevReadFbo = 0;
        glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &prevDrawFbo);
        glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &prevReadFbo);

        for (size_t i = 0; i < layers().size(); ++i) {
            const auto& layer = layers()[i];
            if (!layer) continue;
            if (!layer->visible.load(std::memory_order_relaxed)) continue;
            if (layer->type != LayerType::Raster) continue;
            auto it = layer->tiles.find(tileKey(tx, ty));
            if (it == layer->tiles.end()) continue;

            // Bind the tile FBO and read one pixel. Tile interior occupies
            // texels [kApron..kApron+kTileSize). NOTE: doc-y maps directly
            // to framebuffer-y in tile FBOs — the bake's vertex shader
            // (kDabVS) maps doc-y=0 to NDC.y=-1, which the GL viewport
            // places at framebuffer-y=kApron (the bottom-most row of the
            // interior in GL bottom-left convention). So doc-y=subY
            // lives at framebuffer-y=kApron+subY. No Y-flip.
            glBindFramebuffer(GL_READ_FRAMEBUFFER, it->second.fbo);
            int glX = kApron + subX;
            int glY = kApron + subY;
            unsigned char tile[4] = {0, 0, 0, 0};
            glReadPixels(glX, glY, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, tile);

            // Tile pixels are premultiplied. Apply layer opacity, then
            // composite Porter-Duff "over" onto the accumulator.
            float lo = layer->opacity.load(std::memory_order_relaxed);
            float sR = (tile[0] / 255.0f) * lo;
            float sG = (tile[1] / 255.0f) * lo;
            float sB = (tile[2] / 255.0f) * lo;
            float sA = (tile[3] / 255.0f) * lo;
            float inv = 1.0f - sA;
            accR = sR + accR * inv;
            accG = sG + accG * inv;
            accB = sB + accB * inv;
        }
        glBindFramebuffer(GL_READ_FRAMEBUFFER, prevReadFbo);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, prevDrawFbo);

        auto byte = [](float v) -> int {
            int b = static_cast<int>(std::round(v * 255.0f));
            if (b < 0) b = 0; if (b > 255) b = 255;
            return b;
        };
        int rgb = (byte(accR) << 16) | (byte(accG) << 8) | byte(accB);
        g_lastSampledRgb.store(rgb);
        LOGI("eyedropper sample doc(%.1f,%.1f) tile(%d,%d) sub(%d,%d) "
             "-> #%06X", static_cast<double>(dx), static_cast<double>(dy),
             tx, ty, subX, subY, rgb);
    }

}

// Renders a shape preview into the currently-bound framebuffer (the
// front-buffered layer when called from the framework callback). Clears
// the buffer first so successive previews replace rather than accumulate.
//
// Shape types and their (x0, y0, x1, y1) interpretation:
//   0  Line:       endpoints
//   1  Rectangle:  bbox corners (any two opposite corners)
//   2  Circle:     p0 = center, p1 = point on circle (radius = distance)
//   3  Ellipse:    bbox corners (oval inscribed in the bbox)
void renderShapePreviewToFront(JNIEnv* env, jint width, jint height,
                               jfloatArray transform,
                               int shapeType,
                               float x0, float y0, float x1, float y1,
                               uint32_t rgb, float lineWidth, float alpha,
                               bool snapped) {
    glViewport(0, 0, width, height);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(g_lineProg.program);
    glBindVertexArray(g_quadVao);
    uploadMat4(env, g_lineProg.uTransform, transform);
    glUniform2f(g_lineProg.uScreen, (float)width, (float)height);
    {
        PageClip pageClip = readPageClip();
        uploadPageClip(g_lineProg.uPageMin, g_lineProg.uPageMax,
                       g_lineProg.uPageActive, pageClip);
    }

    switch (shapeType) {
        case 0:
            drawLineSegment(x0, y0, x1, y1, rgb, lineWidth, alpha);
            break;
        case 1: {
            // Swap to the box-SDF program for the preview rect, same
            // approach as the ellipse/circle case below.
            glUseProgram(g_rectProg.program);
            uploadMat4(env, g_rectProg.uTransform, transform);
            glUniform2f(g_rectProg.uScreen, (float)width, (float)height);
            {
                PageClip pc = readPageClip();
                uploadPageClip(g_rectProg.uPageMin, g_rectProg.uPageMax,
                               g_rectProg.uPageActive, pc);
            }
            drawRectOutline(x0, y0, x1, y1, /*rotation*/ 0.0f,
                            rgb, lineWidth, alpha);
            glUseProgram(g_lineProg.program);
            break;
        }
        case 2:
        case 3: {
            float cx, cy, rx, ry;
            if (shapeType == 2) {
                float dx = x1 - x0, dy = y1 - y0;
                cx = x0; cy = y0;
                rx = ry = std::sqrt(dx * dx + dy * dy);
            } else {
                cx = (x0 + x1) * 0.5f;
                cy = (y0 + y1) * 0.5f;
                rx = std::fabs(x1 - x0) * 0.5f;
                ry = std::fabs(y1 - y0) * 0.5f;
            }
            // Swap to the SDF ellipse program for the duration of this
            // draw, then restore the line program so the snap marker
            // below (if any) renders correctly.
            glUseProgram(g_ellipseProg.program);
            uploadMat4(env, g_ellipseProg.uTransform, transform);
            glUniform2f(g_ellipseProg.uScreen, (float)width, (float)height);
            {
                PageClip pc = readPageClip();
                uploadPageClip(g_ellipseProg.uPageMin, g_ellipseProg.uPageMax,
                               g_ellipseProg.uPageActive, pc);
            }
            drawEllipseOutline(cx, cy, rx, ry, /*rotation*/ 0.0f,
                               rgb, lineWidth, alpha);
            glUseProgram(g_lineProg.program);
            break;
        }
    }
    // If the dragged endpoint is currently snapped, draw the indicator
    // at that endpoint so the user can see what they snapped to.
    if (snapped) {
        drawSnapMarker(x1, y1);
    }
    glBindVertexArray(0);
}

// Render a polyline (the in-progress lasso path) into the bound
// framebuffer. Clears first so successive previews replace rather than
// accumulate, mirroring renderShapePreviewToFront. Closing edge from
// points[n-1] → points[0] is drawn iff `closed` is true.
void renderLassoPathToFront(JNIEnv* env, jint width, jint height,
                            jfloatArray transform,
                            const float* points, size_t nPoints,
                            uint32_t rgb, float lineWidth, float alpha,
                            bool closed) {
    glViewport(0, 0, width, height);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    if (nPoints < 2) {
        return;
    }
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(g_lineProg.program);
    glBindVertexArray(g_quadVao);
    uploadMat4(env, g_lineProg.uTransform, transform);
    glUniform2f(g_lineProg.uScreen, (float)width, (float)height);
    {
        PageClip pageClip = readPageClip();
        uploadPageClip(g_lineProg.uPageMin, g_lineProg.uPageMax,
                       g_lineProg.uPageActive, pageClip);
    }
    for (size_t i = 0; i + 1 < nPoints; ++i) {
        drawLineSegment(points[2*i],   points[2*i+1],
                        points[2*i+2], points[2*i+3],
                        rgb, lineWidth, alpha);
    }
    if (closed && nPoints >= 3) {
        drawLineSegment(points[2*(nPoints-1)],   points[2*(nPoints-1)+1],
                        points[0],               points[1],
                        rgb, lineWidth, alpha);
    }
    glBindVertexArray(0);
}

// ---- Undo / redo apply ---------------------------------------------------

void insertShapeAt(Layer& layer, ShapeKind kind, size_t idx, const ShapeData& sd) {
    switch (kind) {
        case ShapeKind::Line:
            layer.lines.insert(layer.lines.begin()
                + std::min(idx, layer.lines.size()), sd.line);
            break;
        case ShapeKind::Rect:
            layer.rects.insert(layer.rects.begin()
                + std::min(idx, layer.rects.size()), sd.rect);
            break;
        case ShapeKind::Ellipse:
            layer.ellipses.insert(layer.ellipses.begin()
                + std::min(idx, layer.ellipses.size()), sd.ellipse);
            break;
        case ShapeKind::Circle:
            layer.circles.insert(layer.circles.begin()
                + std::min(idx, layer.circles.size()), sd.circle);
            break;
        case ShapeKind::None:
            break;
    }
}

void eraseShapeAt(Layer& layer, ShapeKind kind, size_t idx) {
    switch (kind) {
        case ShapeKind::Line:
            if (idx < layer.lines.size())
                layer.lines.erase(layer.lines.begin() + idx);
            break;
        case ShapeKind::Rect:
            if (idx < layer.rects.size())
                layer.rects.erase(layer.rects.begin() + idx);
            break;
        case ShapeKind::Ellipse:
            if (idx < layer.ellipses.size())
                layer.ellipses.erase(layer.ellipses.begin() + idx);
            break;
        case ShapeKind::Circle:
            if (idx < layer.circles.size())
                layer.circles.erase(layer.circles.begin() + idx);
            break;
        case ShapeKind::None:
            break;
    }
}

void assignShapeAt(Layer& layer, ShapeKind kind, size_t idx, const ShapeData& sd) {
    switch (kind) {
        case ShapeKind::Line:
            if (idx < layer.lines.size())    layer.lines[idx]    = sd.line;
            break;
        case ShapeKind::Rect:
            if (idx < layer.rects.size())    layer.rects[idx]    = sd.rect;
            break;
        case ShapeKind::Ellipse:
            if (idx < layer.ellipses.size()) layer.ellipses[idx] = sd.ellipse;
            break;
        case ShapeKind::Circle:
            if (idx < layer.circles.size())  layer.circles[idx]  = sd.circle;
            break;
        case ShapeKind::None:
            break;
    }
}

// Drop GL resources for every tile in a layer, clear the tile map. Used
// by both LayerClear redo and LayerAdd undo (defensive — added layers
// should be empty already, but safe to call regardless).
void dropAllTilesGl(Layer& layer) {
    for (auto& kv : layer.tiles) {
        if (kv.second.fbo)     glDeleteFramebuffers(1, &kv.second.fbo);
        if (kv.second.texture) glDeleteTextures(1, &kv.second.texture);
    }
    layer.tiles.clear();
}

// Wipe the on-disk contents of a layer dir (but leave the dir itself).
// Used by LayerClear redo to mirror the live clearActiveLayer path.
// Metadata files (name.txt, opacity.txt, hidden.flag) are preserved —
// the layer's identity survives a clear, only its content is dropped.
void clearLayerDirOnDisk(size_t layerIdx) {
    if (g_docDir.empty()) return;
    std::string layerDir = activeLayerDir(layerIdx);
    DIR* d = opendir(layerDir.c_str());
    if (!d) return;
    struct dirent* dirEnt;
    while ((dirEnt = readdir(d)) != nullptr) {
        const char* n = dirEnt->d_name;
        if (n[0] == '.') continue;
        if (std::strcmp(n, "name.txt") == 0
            || std::strcmp(n, "opacity.txt") == 0
            || std::strcmp(n, "hidden.flag") == 0) continue;
        std::string p = layerDir + "/" + n;
        unlink(p.c_str());
    }
    closedir(d);
}

// Reverse a previously-recorded action.
void applyEntryReverse(UndoEntry& e) {
    switch (e.op) {
        case UndoOp::RasterStroke: {
            for (const auto& s : e.beforeTiles) applyTileSnap(e.layerIdx, s);
            break;
        }
        case UndoOp::VectorAdd: {
            if (e.layerIdx >= layers().size() || !layers()[e.layerIdx]) break;
            Layer& layer = *layers()[e.layerIdx];
            eraseShapeAt(layer, e.afterShape.kind, e.shapeIdx);
            {
                std::lock_guard<std::mutex> lock(g_selectionMutex);
                if (g_selection.kind == e.afterShape.kind
                    && g_selection.layerIdx == e.layerIdx
                    && g_selection.shapeIdx == e.shapeIdx) {
                    g_selection = Selection{};
                }
            }
            saveVectorLayer(e.layerIdx, layer);
            break;
        }
        case UndoOp::VectorDelete: {
            if (e.layerIdx >= layers().size() || !layers()[e.layerIdx]) break;
            Layer& layer = *layers()[e.layerIdx];
            insertShapeAt(layer, e.beforeShape.kind, e.shapeIdx, e.beforeShape);
            saveVectorLayer(e.layerIdx, layer);
            break;
        }
        case UndoOp::VectorMutate: {
            if (e.layerIdx >= layers().size() || !layers()[e.layerIdx]) break;
            Layer& layer = *layers()[e.layerIdx];
            assignShapeAt(layer, e.beforeShape.kind, e.shapeIdx, e.beforeShape);
            saveVectorLayer(e.layerIdx, layer);
            break;
        }
        case UndoOp::VectorMutateGroup: {
            // Restore each shape's pre-drag state. Save once per
            // affected layer at the end.
            std::vector<size_t> affected;
            for (size_t i = 0; i < e.mutateGroupSels.size()
                            && i < e.mutateGroupBefore.size(); ++i) {
                const Selection& s = e.mutateGroupSels[i];
                if (s.layerIdx >= layers().size() || !layers()[s.layerIdx])
                    continue;
                Layer& layer = *layers()[s.layerIdx];
                assignShapeAt(layer, e.mutateGroupBefore[i].kind,
                              s.shapeIdx, e.mutateGroupBefore[i]);
                if (std::find(affected.begin(), affected.end(),
                              s.layerIdx) == affected.end()) {
                    affected.push_back(s.layerIdx);
                }
            }
            for (size_t idx : affected) {
                if (idx < layers().size() && layers()[idx]) {
                    saveVectorLayer(idx, *layers()[idx]);
                }
            }
            break;
        }
        case UndoOp::LayerClear: {
            if (e.layerIdx >= layers().size() || !layers()[e.layerIdx]) break;
            Layer& layer = *layers()[e.layerIdx];
            for (const auto& s : e.beforeTiles) applyTileSnap(e.layerIdx, s);
            layer.lines    = e.beforeLines;
            layer.rects    = e.beforeRects;
            layer.ellipses = e.beforeEllipses;
            layer.circles  = e.beforeCircles;
            if (layer.type == LayerType::Vector) {
                saveVectorLayer(e.layerIdx, layer);
            }
            break;
        }
        case UndoOp::LayerAdd: {
            // Layer should be the topmost (other entries were undone in
            // reverse). If not, something's gone wrong; bail safely.
            if (e.layerIdx >= layers().size()
                || e.layerIdx + 1 != layers().size()) {
                LOGE("undo LayerAdd: layer %zu isn't topmost (size=%zu)",
                     e.layerIdx, layers().size());
                break;
            }
            if (layers()[e.layerIdx]) {
                dropAllTilesGl(*layers()[e.layerIdx]);
            }
            layers().pop_back();
            deleteLayerDirIfExists(e.layerIdx);
            {
                std::lock_guard<std::mutex> lock(g_selectionMutex);
                if (g_selection.layerIdx == e.layerIdx) {
                    g_selection = Selection{};
                }
            }
            if (e.prevActiveLayer < layers().size()) {
                activeLayer() = e.prevActiveLayer;
            } else if (!layers().empty()) {
                activeLayer() = layers().size() - 1;
            } else {
                activeLayer() = 0;
            }
            break;
        }
        case UndoOp::RasterizeShapeBelow: {
            // Restore target tiles to their pre-bake pixels.
            if (e.targetLayerIdx < layers().size() && layers()[e.targetLayerIdx]) {
                for (const auto& s : e.beforeTiles) {
                    applyTileSnap(e.targetLayerIdx, s);
                }
            }
            // Re-insert the shape at its original index in the source
            // vector layer. Skip if the source has since been
            // converted to raster — restoring tiles alone is still
            // useful.
            if (e.layerIdx < layers().size() && layers()[e.layerIdx]
                && layers()[e.layerIdx]->type == LayerType::Vector) {
                Layer& src = *layers()[e.layerIdx];
                insertShapeAt(src, e.beforeShape.kind, e.shapeIdx, e.beforeShape);
                saveVectorLayer(e.layerIdx, src);
            }
            break;
        }
        case UndoOp::RasterizeLayer: {
            // Flip the layer back to vector: drop GL tile resources +
            // disk tiles, restore shape lists, regenerate shapes.bin.
            if (e.layerIdx >= layers().size() || !layers()[e.layerIdx]) break;
            Layer& layer = *layers()[e.layerIdx];
            dropAllTilesGl(layer);
            // Wipes tile_*.bin (and shapes.bin if any), preserves
            // metadata (name/visibility/opacity).
            clearLayerDirOnDisk(e.layerIdx);
            layer.type     = LayerType::Vector;
            layer.lines    = e.beforeLines;
            layer.rects    = e.beforeRects;
            layer.ellipses = e.beforeEllipses;
            layer.circles  = e.beforeCircles;
            saveVectorLayer(e.layerIdx, layer);
            // Selection on this layer was cleared on the forward path;
            // it stays cleared on reverse.
            break;
        }
        case UndoOp::MergeLayerDown: {
            // Restore target tiles to their pre-merge state. Do this
            // BEFORE re-inserting the source layer so the target's
            // index in `layers()` is still e.targetLayerIdx (insertion
            // doesn't shift layers below the insert point, but doing
            // tiles first keeps the order obvious).
            if (e.targetLayerIdx < layers().size() && layers()[e.targetLayerIdx]) {
                for (const auto& s : e.beforeTiles) {
                    applyTileSnap(e.targetLayerIdx, s);
                }
            }
            // Re-create the source layer at its original idx using the
            // captured snapshot. This shifts trailing layers up + their
            // on-disk dirs.
            insertLayerWithSnapshot(e.layerIdx, e.srcSnapshot);
            // Pre-existing entries' layerIdx pointed into the post-
            // delete index space; the insert just shifted everything
            // above e.layerIdx up by one. Mirror that shift on the
            // remaining stack so future undos still target the right
            // layers. (This entry is being popped now so it skips
            // self-remap.)
            {
                std::lock_guard<std::mutex> lock(g_undoMutex);
                auto shiftUp = [&](UndoEntry& other) {
                    if (other.layerIdx >= e.layerIdx) ++other.layerIdx;
                    if ((other.op == UndoOp::RasterizeShapeBelow
                      || other.op == UndoOp::MergeLayerDown)
                        && other.targetLayerIdx >= e.layerIdx) {
                        ++other.targetLayerIdx;
                    }
                    if (other.op == UndoOp::LayerAdd
                        && other.prevActiveLayer >= e.layerIdx) {
                        ++other.prevActiveLayer;
                    }
                };
                std::function<void(UndoEntry&)> shiftRecursive =
                    [&](UndoEntry& other) {
                        shiftUp(other);
                        if (other.op == UndoOp::MergeLayerDown) {
                            for (auto& nested : other.srcLayerUndoEntries) {
                                shiftRecursive(nested);
                            }
                        }
                    };
                for (auto& other : g_undoStack) shiftRecursive(other);
                for (auto& other : g_redoStack) shiftRecursive(other);

                // Re-push the source-layer's pre-merge undo history
                // onto the bottom of the undo stack. Their layerIdx
                // values were captured PRE-delete, then tracked through
                // any subsequent ops via the recursive remap helpers.
                // After insertLayerWithSnapshot above, the source
                // layer is back at e.layerIdx, so they're addressable.
                for (auto& nested : e.srcLayerUndoEntries) {
                    nested.bytes = computeEntrySize(nested);
                    g_undoTotalBytes += nested.bytes;
                    g_undoStack.push_back(std::move(nested));
                }
                e.srcLayerUndoEntries.clear();
                // Bytes shrunk (nested entries moved out); recompute so
                // applyUndo's redoTotalBytes += e.bytes accounting is
                // correct.
                e.bytes = computeEntrySize(e);
            }
            break;
        }
    }
}

// Re-apply a previously-undone action.
void applyEntryForward(UndoEntry& e) {
    switch (e.op) {
        case UndoOp::RasterStroke: {
            // Replay the bake from stored samples + brush state rather
            // than memcpy'ing afterTiles back. Deterministic given the
            // pre-stroke tile state that reverse() restored — see
            // rebakeStroke + BakeFidelityTest.
            rebakeStroke(e);
            break;
        }
        case UndoOp::VectorAdd: {
            if (e.layerIdx >= layers().size() || !layers()[e.layerIdx]) break;
            Layer& layer = *layers()[e.layerIdx];
            insertShapeAt(layer, e.afterShape.kind, e.shapeIdx, e.afterShape);
            saveVectorLayer(e.layerIdx, layer);
            break;
        }
        case UndoOp::VectorDelete: {
            if (e.layerIdx >= layers().size() || !layers()[e.layerIdx]) break;
            Layer& layer = *layers()[e.layerIdx];
            eraseShapeAt(layer, e.beforeShape.kind, e.shapeIdx);
            {
                std::lock_guard<std::mutex> lock(g_selectionMutex);
                if (g_selection.kind == e.beforeShape.kind
                    && g_selection.layerIdx == e.layerIdx
                    && g_selection.shapeIdx == e.shapeIdx) {
                    g_selection = Selection{};
                }
            }
            saveVectorLayer(e.layerIdx, layer);
            break;
        }
        case UndoOp::VectorMutate: {
            if (e.layerIdx >= layers().size() || !layers()[e.layerIdx]) break;
            Layer& layer = *layers()[e.layerIdx];
            assignShapeAt(layer, e.afterShape.kind, e.shapeIdx, e.afterShape);
            saveVectorLayer(e.layerIdx, layer);
            break;
        }
        case UndoOp::VectorMutateGroup: {
            std::vector<size_t> affected;
            for (size_t i = 0; i < e.mutateGroupSels.size()
                            && i < e.mutateGroupAfter.size(); ++i) {
                const Selection& s = e.mutateGroupSels[i];
                if (s.layerIdx >= layers().size() || !layers()[s.layerIdx])
                    continue;
                Layer& layer = *layers()[s.layerIdx];
                assignShapeAt(layer, e.mutateGroupAfter[i].kind,
                              s.shapeIdx, e.mutateGroupAfter[i]);
                if (std::find(affected.begin(), affected.end(),
                              s.layerIdx) == affected.end()) {
                    affected.push_back(s.layerIdx);
                }
            }
            for (size_t idx : affected) {
                if (idx < layers().size() && layers()[idx]) {
                    saveVectorLayer(idx, *layers()[idx]);
                }
            }
            break;
        }
        case UndoOp::LayerClear: {
            if (e.layerIdx >= layers().size() || !layers()[e.layerIdx]) break;
            Layer& layer = *layers()[e.layerIdx];
            dropAllTilesGl(layer);
            layer.lines.clear();
            layer.rects.clear();
            layer.ellipses.clear();
            layer.circles.clear();
            clearLayerDirOnDisk(e.layerIdx);
            if (layer.type == LayerType::Vector) {
                saveVectorLayer(e.layerIdx, layer);
            }
            break;
        }
        case UndoOp::LayerAdd: {
            auto layer = std::make_unique<Layer>();
            layer->type = e.addedLayerType;
            layers().push_back(std::move(layer));
            activeLayer() = layers().size() - 1;
            if (e.addedLayerType == LayerType::Vector) {
                saveVectorLayer(activeLayer(), *layers()[activeLayer()]);
            }
            break;
        }
        case UndoOp::RasterizeShapeBelow: {
            // Re-erase the shape from the source vector layer.
            if (e.layerIdx < layers().size() && layers()[e.layerIdx]
                && layers()[e.layerIdx]->type == LayerType::Vector) {
                Layer& src = *layers()[e.layerIdx];
                eraseShapeAt(src, e.beforeShape.kind, e.shapeIdx);
                saveVectorLayer(e.layerIdx, src);
                {
                    std::lock_guard<std::mutex> lock(g_selectionMutex);
                    if (g_selection.kind == e.beforeShape.kind
                        && g_selection.layerIdx == e.layerIdx
                        && g_selection.shapeIdx == e.shapeIdx) {
                        g_selection = Selection{};
                    }
                }
            }
            // Restore the post-bake tiles on the target raster layer.
            if (e.targetLayerIdx < layers().size() && layers()[e.targetLayerIdx]) {
                for (const auto& s : e.afterTiles) {
                    applyTileSnap(e.targetLayerIdx, s);
                }
            }
            break;
        }
        case UndoOp::RasterizeLayer: {
            // Re-do the rasterize: drop the shapes, flip type, restore
            // the post-bake tiles. The shapes.bin file gets removed by
            // the type flip + saveVectorLayer-skip; clearLayerDirOnDisk
            // would also drop tile files we're about to recreate, so
            // we just unlink shapes.bin directly.
            if (e.layerIdx >= layers().size() || !layers()[e.layerIdx]) break;
            Layer& layer = *layers()[e.layerIdx];
            layer.lines.clear();
            layer.rects.clear();
            layer.ellipses.clear();
            layer.circles.clear();
            layer.type = LayerType::Raster;
            for (const auto& s : e.afterTiles) {
                applyTileSnap(e.layerIdx, s);
            }
            if (!g_docDir.empty()) {
                std::string p = activeLayerDir(e.layerIdx) + "/shapes.bin";
                unlink(p.c_str());
            }
            // Selection on this layer is gone — clear if it pointed here.
            {
                std::lock_guard<std::mutex> lock(g_selectionMutex);
                if (g_selection.kind != ShapeKind::None
                    && g_selection.layerIdx == e.layerIdx) {
                    g_selection = Selection{};
                }
            }
            break;
        }
        case UndoOp::MergeLayerDown: {
            // Pull the source-layer's history back out of the undo
            // stack INTO this entry, so the next undo of this merge
            // can re-restore them. (Mirror image of applyEntryReverse,
            // which moved them onto the stack.)
            {
                std::lock_guard<std::mutex> lock(g_undoMutex);
                for (auto it = g_undoStack.begin(); it != g_undoStack.end(); ) {
                    if (it->layerIdx == e.layerIdx) {
                        g_undoTotalBytes -= it->bytes;
                        e.srcLayerUndoEntries.push_back(std::move(*it));
                        it = g_undoStack.erase(it);
                    } else {
                        ++it;
                    }
                }
            }
            // Re-apply post-merge tiles to the target.
            if (e.targetLayerIdx < layers().size() && layers()[e.targetLayerIdx]) {
                for (const auto& s : e.afterTiles) {
                    applyTileSnap(e.targetLayerIdx, s);
                }
            }
            // Delete the source layer that undo re-inserted. Use
            // deleteLayerImpl so trailing layers + their dirs renumber
            // correctly. Its remap pass leaves our extracted source-
            // layer entries alone (they're inside e, not in the stack).
            if (e.layerIdx < layers().size() && layers()[e.layerIdx]) {
                deleteLayerImpl(e.layerIdx);
            }
            // Recompute bytes — we just absorbed the nested entries.
            e.bytes = computeEntrySize(e);
            break;
        }
    }
}

// Pop the top of the undo stack, reverse it, push onto the redo stack.
// Caller is responsible for any subsequent re-render.
void applyUndo() {
    UndoEntry e;
    {
        std::lock_guard<std::mutex> lock(g_undoMutex);
        if (g_undoStack.empty()) return;
        e = std::move(g_undoStack.back());
        g_undoStack.pop_back();
        g_undoTotalBytes -= e.bytes;
    }
    applyEntryReverse(e);
    {
        std::lock_guard<std::mutex> lock(g_undoMutex);
        g_redoTotalBytes += e.bytes;
        g_redoStack.push_back(std::move(e));
    }
}

void applyRedo() {
    UndoEntry e;
    {
        std::lock_guard<std::mutex> lock(g_undoMutex);
        if (g_redoStack.empty()) return;
        e = std::move(g_redoStack.back());
        g_redoStack.pop_back();
        g_redoTotalBytes -= e.bytes;
    }
    applyEntryForward(e);
    {
        std::lock_guard<std::mutex> lock(g_undoMutex);
        g_undoTotalBytes += e.bytes;
        g_undoStack.push_back(std::move(e));
    }
}

// ---- Bucket fill ----------------------------------------------------------
//
// Active-layer-source / active-layer-target flood fill bounded by the
// page rect. ONLY the active raster layer's content acts as a boundary
// — vector shapes on sibling layers, strokes on other raster layers,
// and the page grid are ignored. Run on the GL thread inside the
// bucketFillAt JNI.
//
// Cost is dominated by (1) rendering the active layer at page
// resolution, (2) reading it back to CPU, (3) the iterative flood. For
// typical page sizes this is a few hundred ms — the user perceives a
// brief freeze. Acceptable for v1.

// ---- Raster selection (Phase 1: rectangular, translate-only) -----------

// Free GPU resources held by an active raster selection. Idempotent.
void disposeRasterSelectionGl() {
    if (g_rasterSel.contentTex) {
        glDeleteTextures(1, &g_rasterSel.contentTex);
        g_rasterSel.contentTex = 0;
    }
    g_rasterSel.contentW = 0;
    g_rasterSel.contentH = 0;
    g_rasterSel.liftedTiles.clear();
    g_rasterSel.fixedAspect = false;
}

// CPU scanline polygon fill into a single-channel R8 buffer. Output
// pixel (i, j) covers doc-coords (offsetX + i, offsetY + j); inside-
// polygon pixels are 255, outside are 0. Uses the even-odd rule so
// self-intersecting lasso paths fill the way users expect (the bounded
// regions are filled). `points` is interleaved [x0,y0,x1,y1,...].
//
// Cost is O(bufH * nPoints + bufW * filled-pixels). For typical lasso
// sizes (a few hundred px / a few hundred points) this is well under a
// frame on the device.
void rasterizePolygonMask(const float* points, size_t nPoints,
                          int offsetX, int offsetY, int bufW, int bufH,
                          uint8_t* outBuf) {
    std::memset(outBuf, 0, static_cast<size_t>(bufW) * bufH);
    if (nPoints < 3) return;

    std::vector<float> xs;
    xs.reserve(16);
    for (int j = 0; j < bufH; ++j) {
        float y = static_cast<float>(offsetY + j) + 0.5f;  // pixel-center
        xs.clear();
        for (size_t i = 0; i < nPoints; ++i) {
            size_t k = (i + 1) % nPoints;
            float ax = points[2*i],   ay = points[2*i+1];
            float bx = points[2*k],   by = points[2*k+1];
            // Half-open interval rule: each vertex counts in exactly one
            // adjacent edge, so vertices and horizontal edges don't
            // double-count. (ay <= y < by) OR (by <= y < ay).
            bool crosses = (ay <= y && by >  y) || (by <= y && ay >  y);
            if (!crosses) continue;
            float t = (y - ay) / (by - ay);
            xs.push_back(ax + t * (bx - ax));
        }
        if (xs.size() < 2) continue;
        std::sort(xs.begin(), xs.end());
        for (size_t pi = 0; pi + 1 < xs.size(); pi += 2) {
            int x0i = static_cast<int>(std::ceil (xs[pi]   - offsetX));
            int x1i = static_cast<int>(std::floor(xs[pi+1] - offsetX));
            x0i = std::max(0, x0i);
            x1i = std::min(bufW, x1i);
            if (x1i > x0i) {
                std::memset(outBuf + static_cast<size_t>(j) * bufW + x0i,
                            255, static_cast<size_t>(x1i - x0i));
            }
        }
    }
}

// "Lift" a polygonal region of the active raster layer into a floating
// selection. Same overall flow as liftRasterSelectionRect but the lift
// shape is the polygon-filled area (even-odd rule) of the supplied
// closed polyline. Pixels outside the polygon (and outside the page
// rect) are not lifted; both contentTex and the source-tile clear are
// masked accordingly. Caller is responsible for commit or cancel.
//
// `points` is interleaved [x0,y0,x1,y1,...] of `nPoints` doc-coord
// vertices, implicit closing edge from points[n-1] → points[0].
bool liftRasterSelectionPolygon(const float* points, size_t nPoints) {
    {
        std::lock_guard<std::mutex> lock(g_rasterSelMutex);
        if (g_rasterSel.active) return false;
    }
    if (nPoints < 3) return false;
    // Lifting clears the lifted region in source tiles AND adds a
    // floating-selection overlay to compositeAllLayers — both
    // change MB content outside any stroke's bbox. Invalidate cache.
    g_mbCacheValid = false;

    // Polygon AABB.
    float pMinX = points[0], pMinY = points[1];
    float pMaxX = pMinX,     pMaxY = pMinY;
    for (size_t i = 1; i < nPoints; ++i) {
        pMinX = std::min(pMinX, points[2*i]);
        pMinY = std::min(pMinY, points[2*i+1]);
        pMaxX = std::max(pMaxX, points[2*i]);
        pMaxY = std::max(pMaxY, points[2*i+1]);
    }

    // Clamp the bbox to the page rect — pixels outside the page are
    // always blank in the doc model and lifting them is meaningless.
    PageClip page = readPageClip();
    if (page.active) {
        pMinX = std::max(pMinX, page.minX);
        pMinY = std::max(pMinY, page.minY);
        pMaxX = std::min(pMaxX, page.maxX);
        pMaxY = std::min(pMaxY, page.maxY);
    }
    int rectIX0 = static_cast<int>(std::floor(pMinX));
    int rectIY0 = static_cast<int>(std::floor(pMinY));
    int rectIX1 = static_cast<int>(std::ceil (pMaxX));
    int rectIY1 = static_cast<int>(std::ceil (pMaxY));
    int rectW = rectIX1 - rectIX0;
    int rectH = rectIY1 - rectIY0;
    if (rectW <= 0 || rectH <= 0) return false;

    ensureAtLeastOneLayer();
    if (activeLayer() >= layers().size() || !layers()[activeLayer()]) return false;
    Layer& layer = *layers()[activeLayer()];
    if (layer.type != LayerType::Raster) return false;

    // Rasterize the polygon mask once for the whole bbox.
    std::vector<uint8_t> mask(static_cast<size_t>(rectW) * rectH);
    rasterizePolygonMask(points, nPoints,
                         rectIX0, rectIY0, rectW, rectH, mask.data());

    // Allocate the content texture (premultiplied RGBA, sized to bbox)
    // and an accumulator buffer we'll glTexSubImage2D from once at the
    // end. Skipping per-tile sub-image uploads avoids a glReadPixels +
    // glTexSubImage2D round-trip per tile.
    GLuint contentTex = 0;
    glGenTextures(1, &contentTex);
    glBindTexture(GL_TEXTURE_2D, contentTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, rectW, rectH, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    std::vector<uint8_t> contentBytes(
        static_cast<size_t>(rectW) * rectH * 4, 0);

    int tx0 = rectIX0 / kTileSize;
    int ty0 = rectIY0 / kTileSize;
    int tx1 = (rectIX1 - 1) / kTileSize;
    int ty1 = (rectIY1 - 1) / kTileSize;

    GLint prevFbo = 0;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &prevFbo);

    std::vector<TileSnap> liftedTiles;
    liftedTiles.reserve(static_cast<size_t>(tx1 - tx0 + 1)
                      * static_cast<size_t>(ty1 - ty0 + 1));

    std::vector<uint8_t> tileBytes(kTileBytes);
    for (int ty = ty0; ty <= ty1; ++ty) {
        for (int tx = tx0; tx <= tx1; ++tx) {
            int tileX0 = tx * kTileSize;
            int tileY0 = ty * kTileSize;
            int ix0 = std::max(rectIX0, tileX0);
            int iy0 = std::max(rectIY0, tileY0);
            int ix1 = std::min(rectIX1, tileX0 + kTileSize);
            int iy1 = std::min(rectIY1, tileY0 + kTileSize);
            if (ix1 <= ix0 || iy1 <= iy0) continue;

            auto it = layer.tiles.find(tileKey(tx, ty));
            if (it == layer.tiles.end()) continue;

            // Read the entire tile; cheap (one glReadPixels per tile)
            // and avoids partial-row/column glReadPixels arithmetic.
            glBindFramebuffer(GL_FRAMEBUFFER, it->second.fbo);
            glReadPixels(kApron, kApron, kTileSize, kTileSize,
                         GL_RGBA, GL_UNSIGNED_BYTE, tileBytes.data());

            // Pre-lift snapshot for undo + cancel restoration. Copy
            // the freshly-read bytes into a shared buffer; this is
            // the BEFORE state for the lift's undo.
            TileSnap snap;
            snap.tx = tx; snap.ty = ty;
            snap.existed = true;
            snap.bytes = acquireTileBytesFrom(tileBytes.data());
            liftedTiles.push_back(std::move(snap));

            // For each pixel inside the tile-bbox intersection that the
            // mask marks as inside-polygon, copy into contentBytes and
            // zero out in the modified tile bytes.
            bool tileTouched = false;
            for (int y = iy0; y < iy1; ++y) {
                int tileLocalY = y - tileY0;
                int bboxLocalY = y - rectIY0;
                const uint8_t* maskRow =
                    mask.data() + static_cast<size_t>(bboxLocalY) * rectW;
                for (int x = ix0; x < ix1; ++x) {
                    int bboxLocalX = x - rectIX0;
                    if (maskRow[bboxLocalX] == 0) continue;
                    int tileLocalX = x - tileX0;
                    size_t tilePxIdx =
                        (static_cast<size_t>(tileLocalY) * kTileSize
                         + tileLocalX) * 4;
                    size_t bboxPxIdx =
                        (static_cast<size_t>(bboxLocalY) * rectW
                         + bboxLocalX) * 4;
                    contentBytes[bboxPxIdx + 0] = tileBytes[tilePxIdx + 0];
                    contentBytes[bboxPxIdx + 1] = tileBytes[tilePxIdx + 1];
                    contentBytes[bboxPxIdx + 2] = tileBytes[tilePxIdx + 2];
                    contentBytes[bboxPxIdx + 3] = tileBytes[tilePxIdx + 3];
                    tileBytes[tilePxIdx + 0] = 0;
                    tileBytes[tilePxIdx + 1] = 0;
                    tileBytes[tilePxIdx + 2] = 0;
                    tileBytes[tilePxIdx + 3] = 0;
                    tileTouched = true;
                }
            }
            if (tileTouched) {
                uploadTileBytesAndSave(activeLayer(), tx, ty,
                                       tileBytes.data());
            } else {
                // Tile didn't actually intersect the polygon — drop the
                // pre-lift snapshot we pushed above so we don't pollute
                // the lift's undo set with no-op tiles.
                liftedTiles.pop_back();
            }
        }
    }

    glBindTexture(GL_TEXTURE_2D, contentTex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, rectW, rectH,
                    GL_RGBA, GL_UNSIGNED_BYTE, contentBytes.data());
    glBindFramebuffer(GL_FRAMEBUFFER, prevFbo);

    {
        std::lock_guard<std::mutex> lock(g_rasterSelMutex);
        g_rasterSel.active = true;
        g_rasterSel.layerIdx = activeLayer();
        g_rasterSel.bboxMinX = static_cast<float>(rectIX0);
        g_rasterSel.bboxMinY = static_cast<float>(rectIY0);
        g_rasterSel.bboxMaxX = static_cast<float>(rectIX1);
        g_rasterSel.bboxMaxY = static_cast<float>(rectIY1);
        g_rasterSel.centerX  = (rectIX0 + rectIX1) * 0.5f;
        g_rasterSel.centerY  = (rectIY0 + rectIY1) * 0.5f;
        g_rasterSel.halfW    = rectW * 0.5f;
        g_rasterSel.halfH    = rectH * 0.5f;
        g_rasterSel.rotation = 0.0f;
        g_rasterSel.contentTex = contentTex;
        g_rasterSel.contentW = rectW;
        g_rasterSel.contentH = rectH;
        g_rasterSel.liftedTiles = std::move(liftedTiles);
    }
    return true;
}

// "Lift" a rectangular region of the active raster layer into a floating
// selection: copy its pixels into a content texture and clear them from
// the source tiles. Returns true if a selection was created. On success
// the caller is responsible for eventually calling commit or cancel.
bool liftRasterSelectionRect(float x0, float y0, float x1, float y1) {
    // If a selection is already active, refuse — caller should commit
    // or cancel first. (Kotlin enforces this in the touch handler.)
    {
        std::lock_guard<std::mutex> lock(g_rasterSelMutex);
        if (g_rasterSel.active) return false;
    }
    // Same rationale as liftRasterSelectionPolygon — invalidate the
    // partial-recomposite cache.
    g_mbCacheValid = false;

    // Normalize the rect.
    if (x1 < x0) std::swap(x0, x1);
    if (y1 < y0) std::swap(y0, y1);

    // Clamp to the page rect — selecting outside the page never makes
    // sense (the fill / draw paths already exclude pixels there too).
    PageClip page = readPageClip();
    if (page.active) {
        x0 = std::max(x0, page.minX);
        y0 = std::max(y0, page.minY);
        x1 = std::min(x1, page.maxX);
        y1 = std::min(y1, page.maxY);
    }

    int rectW = static_cast<int>(std::floor(x1) - std::floor(x0));
    int rectH = static_cast<int>(std::floor(y1) - std::floor(y0));
    if (rectW <= 0 || rectH <= 0) return false;

    ensureAtLeastOneLayer();
    if (activeLayer() >= layers().size() || !layers()[activeLayer()]) return false;
    Layer& layer = *layers()[activeLayer()];
    if (layer.type != LayerType::Raster) return false;

    int rectIX0 = static_cast<int>(std::floor(x0));
    int rectIY0 = static_cast<int>(std::floor(y0));
    int rectIX1 = rectIX0 + rectW;
    int rectIY1 = rectIY0 + rectH;

    // Allocate the content texture (premultiplied RGBA, sized to bbox).
    GLuint contentTex = 0;
    glGenTextures(1, &contentTex);
    glBindTexture(GL_TEXTURE_2D, contentTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, rectW, rectH, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    // Initialize transparent so any region of the bbox not covered by
    // an existing tile (no source content) reads as zeros.
    {
        std::vector<uint8_t> zeros(static_cast<size_t>(rectW) * rectH * 4, 0);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, rectW, rectH,
                        GL_RGBA, GL_UNSIGNED_BYTE, zeros.data());
    }

    // For each tile that overlaps the rect:
    //  1. Snapshot pre-lift bytes (for undo + cancel restoration).
    //  2. glCopyTexSubImage2D from the tile FBO into the content texture
    //     (orientation matches because both use bottom-up byte layout
    //     consistent with our doc-top = byte-row-0 convention).
    //  3. Scissor-clear the lifted region in the tile FBO.
    int tx0 = rectIX0 / kTileSize;
    int ty0 = rectIY0 / kTileSize;
    int tx1 = (rectIX1 - 1) / kTileSize;
    int ty1 = (rectIY1 - 1) / kTileSize;

    GLint prevFbo = 0;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &prevFbo);

    std::vector<TileSnap> liftedTiles;
    liftedTiles.reserve(static_cast<size_t>(tx1 - tx0 + 1) * (ty1 - ty0 + 1));

    for (int ty = ty0; ty <= ty1; ++ty) {
        for (int tx = tx0; tx <= tx1; ++tx) {
            // Doc-coord intersection of tile and selection rect.
            int tileX0 = tx * kTileSize;
            int tileY0 = ty * kTileSize;
            int ix0 = std::max(rectIX0, tileX0);
            int iy0 = std::max(rectIY0, tileY0);
            int ix1 = std::min(rectIX1, tileX0 + kTileSize);
            int iy1 = std::min(rectIY1, tileY0 + kTileSize);
            int iw = ix1 - ix0;
            int ih = iy1 - iy0;
            if (iw <= 0 || ih <= 0) continue;

            // Skip tiles that don't exist (no content to lift; the
            // content texture's zero init covers that region).
            auto it = layer.tiles.find(tileKey(tx, ty));
            if (it == layer.tiles.end()) continue;

            // (a) Snapshot for undo / restore. Share the cache if
            // available, fall back to glReadPixels otherwise.
            TileSnap snap;
            snap.tx = tx; snap.ty = ty;
            snap.existed = true;
            if (it->second.cachedBytes) {
                snap.bytes = it->second.cachedBytes;
            } else {
                auto fresh = tilePool().acquire();
                glBindFramebuffer(GL_FRAMEBUFFER, it->second.fbo);
                glReadPixels(kApron, kApron, kTileSize, kTileSize,
                             GL_RGBA, GL_UNSIGNED_BYTE, fresh->data());
                it->second.cachedBytes = fresh;
                snap.bytes = fresh;
            }
            liftedTiles.push_back(std::move(snap));

            // (b) Copy the tile's intersection into the content texture.
            // glCopyTexSubImage2D reads from GL_READ_FRAMEBUFFER, so
            // bind the tile FBO explicitly. The cache-hit path in
            // step (a) skips the bind that the old (always-readback)
            // code relied on, so without this we'd copy from
            // whatever framebuffer was bound before — usually the
            // previous iteration's tile, which produces the "lift
            // shows a shifted region of the canvas" bug.
            glBindFramebuffer(GL_READ_FRAMEBUFFER, it->second.fbo);
            int srcX = ix0 - tileX0;
            int srcY = iy0 - tileY0;
            int dstX = ix0 - rectIX0;
            int dstY = iy0 - rectIY0;
            glBindTexture(GL_TEXTURE_2D, contentTex);
            glCopyTexSubImage2D(GL_TEXTURE_2D, 0,
                                dstX, dstY,
                                srcX + kApron, srcY + kApron, iw, ih);

            // (c) Clear the lifted pixels in the source tile. Scissor
            // is in FBO coords, so the apron offset has to be applied.
            glBindFramebuffer(GL_FRAMEBUFFER, it->second.fbo);
            glViewport(kApron, kApron, kTileSize, kTileSize);
            glEnable(GL_SCISSOR_TEST);
            glScissor(srcX + kApron, srcY + kApron, iw, ih);
            glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            glDisable(GL_SCISSOR_TEST);

            // (d) Persist the tile's new state to disk so a crash mid-
            // selection doesn't leave the tile and the on-disk file out
            // of sync. The disk save is required by saveTileToDisk's API.
            saveTileToDisk(activeLayer(), tileKey(tx, ty));
            // Tile content changed (lift cleared the lifted region);
            // its 8 neighbors' aprons need resync.
            markApronStaleAround(layer, tx, ty);
        }
    }
    glBindFramebuffer(GL_FRAMEBUFFER, prevFbo);

    {
        std::lock_guard<std::mutex> lock(g_rasterSelMutex);
        g_rasterSel.active = true;
        g_rasterSel.layerIdx = activeLayer();
        g_rasterSel.bboxMinX = static_cast<float>(rectIX0);
        g_rasterSel.bboxMinY = static_cast<float>(rectIY0);
        g_rasterSel.bboxMaxX = static_cast<float>(rectIX1);
        g_rasterSel.bboxMaxY = static_cast<float>(rectIY1);
        g_rasterSel.centerX  = (rectIX0 + rectIX1) * 0.5f;
        g_rasterSel.centerY  = (rectIY0 + rectIY1) * 0.5f;
        g_rasterSel.halfW    = rectW * 0.5f;
        g_rasterSel.halfH    = rectH * 0.5f;
        g_rasterSel.rotation = 0.0f;
        g_rasterSel.contentTex = contentTex;
        g_rasterSel.contentW = rectW;
        g_rasterSel.contentH = rectH;
        g_rasterSel.liftedTiles = std::move(liftedTiles);
    }
    return true;
}

// Bake the floating selection at its current transformed position back
// into the active layer's tiles. Pushes a single undo entry covering
// both the original lift and the drop so undo restores the pre-lift
// state in one step. Releases the selection.
void commitRasterSelectionImpl() {
    RasterSelection sel;
    {
        std::lock_guard<std::mutex> lock(g_rasterSelMutex);
        if (!g_rasterSel.active) return;
        sel = g_rasterSel;            // shallow copy; we'll move liftedTiles below
        // Empty the global so further calls don't re-process; the GL
        // resources are still owned via the local copy and will be
        // freed by disposeRasterSelectionGl after we apply the drop.
        g_rasterSel.liftedTiles.clear();
        g_rasterSel.active = false;
    }
    // Dropping the floating selection's pixels back into the layer
    // changes MB content outside any stroke's bbox. Invalidate cache.
    g_mbCacheValid = false;
    if (sel.layerIdx >= layers().size() || !layers()[sel.layerIdx]) {
        if (sel.contentTex) glDeleteTextures(1, &sel.contentTex);
        return;
    }
    Layer& layer = *layers()[sel.layerIdx];
    if (layer.type != LayerType::Raster) {
        if (sel.contentTex) glDeleteTextures(1, &sel.contentTex);
        return;
    }

    // Compute placement corners in doc-coords from the floating
    // selection's OBB. Layout matches kSelVS (uC0=top-left UV, etc.).
    Obb selObb = obbForRasterSelection(sel);
    float c0x, c0y, c1x, c1y, c2x, c2y, c3x, c3y;
    rotateLocalToWorld(selObb, -selObb.hw, -selObb.hh, c0x, c0y);
    rotateLocalToWorld(selObb, +selObb.hw, -selObb.hh, c1x, c1y);
    rotateLocalToWorld(selObb, +selObb.hw, +selObb.hh, c2x, c2y);
    rotateLocalToWorld(selObb, -selObb.hw, +selObb.hh, c3x, c3y);

    // Snap an unrotated placement to the doc-pixel grid before baking.
    //
    // A move drag leaves centerX/centerY at an arbitrary float, and the
    // bake samples contentTex with GL_LINEAR — so a fractional doc-px
    // offset makes every destination pixel a 2x2 blend of its
    // neighbors. That is a permanent one-pixel box blur baked into the
    // tiles, and it compounds on every lift/drop cycle. This is the
    // "content goes soft after I move it" bug.
    //
    // With the corners on integers, each fragment samples a texel dead
    // center and the bake is lossless. Rotated placements resample no
    // matter what, so they're left alone. Worst-case visible effect is
    // the selection settling by up to half a doc-pixel on pen-up.
    bool pixelExact = false;      // unrotated AND unscaled => 1:1 copy
    if (std::fabs(selObb.rotation) < 1e-4f) {
        float minX = std::round(selObb.cx - selObb.hw);
        float minY = std::round(selObb.cy - selObb.hh);
        float maxX, maxY;
        // Unscaled? Carry the source dimensions across exactly instead
        // of rounding each edge independently — that could shave a
        // pixel off the extent and reintroduce the resample we're
        // trying to avoid.
        if (std::fabs(selObb.hw * 2.0f - static_cast<float>(sel.contentW)) < 1e-3f
         && std::fabs(selObb.hh * 2.0f - static_cast<float>(sel.contentH)) < 1e-3f) {
            maxX = minX + static_cast<float>(sel.contentW);
            maxY = minY + static_cast<float>(sel.contentH);
            pixelExact = true;
        } else {
            maxX = std::round(selObb.cx + selObb.hw);
            maxY = std::round(selObb.cy + selObb.hh);
            if (maxX <= minX) maxX = minX + 1.0f;
            if (maxY <= minY) maxY = minY + 1.0f;
        }
        c0x = minX; c0y = minY;
        c1x = maxX; c1y = minY;
        c2x = maxX; c2y = maxY;
        c3x = minX; c3y = maxY;
    }

    // Axis-aligned bounding box of the (possibly rotated) placement.
    float aabbMinX = std::min(std::min(c0x, c1x), std::min(c2x, c3x));
    float aabbMinY = std::min(std::min(c0y, c1y), std::min(c2y, c3y));
    float aabbMaxX = std::max(std::max(c0x, c1x), std::max(c2x, c3x));
    float aabbMaxY = std::max(std::max(c0y, c1y), std::max(c2y, c3y));

    // Tile iteration covers the on-page portion of the AABB only, so we
    // don't bake into tiles that would be entirely fragment-discarded.
    // The corners themselves stay UNCLAMPED so aspect ratio (and any
    // rotation) is preserved; the FS discards off-page fragments via
    // uPageMin/uPageMax.
    PageClip page = readPageClip();
    int iterX0 = static_cast<int>(std::floor(aabbMinX));
    int iterY0 = static_cast<int>(std::floor(aabbMinY));
    int iterX1 = static_cast<int>(std::ceil(aabbMaxX));
    int iterY1 = static_cast<int>(std::ceil(aabbMaxY));
    if (page.active) {
        iterX0 = std::max(iterX0, static_cast<int>(page.minX));
        iterY0 = std::max(iterY0, static_cast<int>(page.minY));
        iterX1 = std::min(iterX1, static_cast<int>(page.maxX));
        iterY1 = std::min(iterY1, static_cast<int>(page.maxY));
    }
    bool willDrop = (iterX1 > iterX0 && iterY1 > iterY0);

    GLint prevFbo = 0;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &prevFbo);

    // For the drop, render the content texture into each affected tile
    // via the selection shader (premultiplied over). Build a fresh undo
    // entry whose beforeTiles = the pre-lift snapshots and afterTiles =
    // post-drop tile state.
    UndoEntry entry;
    entry.op = UndoOp::RasterStroke;
    entry.layerIdx = sel.layerIdx;
    entry.beforeTiles = std::move(sel.liftedTiles);

    // Set of tiles touched by either lift OR drop, so undo restores both.
    auto tileKeyForCoord = [](int tx, int ty) { return tileKey(tx, ty); };
    std::unordered_map<int64_t, std::pair<int, int>> touchedTiles;
    for (const auto& s : entry.beforeTiles) {
        touchedTiles[tileKeyForCoord(s.tx, s.ty)] = {s.tx, s.ty};
    }
    if (willDrop) {
        int tx0 = iterX0 / kTileSize;
        int ty0 = iterY0 / kTileSize;
        int tx1 = (iterX1 - 1) / kTileSize;
        int ty1 = (iterY1 - 1) / kTileSize;
        for (int ty = ty0; ty <= ty1; ++ty)
            for (int tx = tx0; tx <= tx1; ++tx)
                touchedTiles[tileKey(tx, ty)] = {tx, ty};
    }

    // Add before-snapshots for any drop-only tiles (not in the lift set).
    {
        std::unordered_map<int64_t, bool> haveBefore;
        for (const auto& s : entry.beforeTiles)
            haveBefore[tileKey(s.tx, s.ty)] = true;
        for (auto& kv : touchedTiles) {
            if (haveBefore[kv.first]) continue;
            int tx = kv.second.first, ty = kv.second.second;
            TileSnap snap;
            snap.tx = tx; snap.ty = ty;
            auto it = layer.tiles.find(kv.first);
            if (it != layer.tiles.end()) {
                snap.existed = true;
                if (it->second.cachedBytes) {
                    snap.bytes = it->second.cachedBytes;
                } else {
                    auto fresh = tilePool().acquire();
                    glBindFramebuffer(GL_FRAMEBUFFER, it->second.fbo);
                    glReadPixels(kApron, kApron, kTileSize, kTileSize,
                                 GL_RGBA, GL_UNSIGNED_BYTE, fresh->data());
                    it->second.cachedBytes = fresh;
                    snap.bytes = fresh;
                }
            }
            entry.beforeTiles.push_back(std::move(snap));
        }
    }

    // Drop: render the content texture into each affected drop-tile.
    if (willDrop && sel.contentTex != 0) {
        glUseProgram(g_sel.program);
        glBindVertexArray(g_quadVao);
        glActiveTexture(GL_TEXTURE0);
        glUniform1i(g_sel.uContent, 0);
        // Bake at full opacity — the layer's opacity is applied at
        // composite time, so scaling here would double up.
        glUniform1f(g_sel.uOpacity, 1.0f);
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

        // Integer-aligned 1:1 placement: sample NEAREST so the bake is
        // bit-exact. LINEAR would give the same bytes in principle (the
        // samples land on texel centers) but only up to float precision
        // in the corner math at large doc coordinates — NEAREST removes
        // the question. The texture is deleted below, so no restore.
        if (pixelExact) {
            glBindTexture(GL_TEXTURE_2D, sel.contentTex);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        }

        // Per-tile bake. Each tile uses its own viewport and the four
        // placement corners translated into the tile's local coordinate
        // frame (uTransform=identity, uScreen=(kTileSize,kTileSize)).
        // Geometry uses unclamped corners so rotation/scale is preserved;
        // FS-level page-clip discards off-page fragments per tile.
        glUniformMatrix4fv(g_sel.uTransform, 1, GL_FALSE, kIdentity);
        glUniform2f(g_sel.uScreen, kTileSizeF, kTileSizeF);

        int tx0 = iterX0 / kTileSize;
        int ty0 = iterY0 / kTileSize;
        int tx1 = (iterX1 - 1) / kTileSize;
        int ty1 = (iterY1 - 1) / kTileSize;
        for (int ty = ty0; ty <= ty1; ++ty) {
            for (int tx = tx0; tx <= tx1; ++tx) {
                Tile& tile = getOrCreateTile(layer, tx, ty);
                // getOrCreateTile silently rebinds GL_TEXTURE_2D to the
                // new tile's texture when it has to create one (no-op
                // for existing tiles). Re-bind the source contentTex
                // inside the loop so the draw samples from the
                // selection content — not from the destination tile,
                // which would be a same-texture-as-FBO read whose
                // result is undefined per spec (and was producing
                // intermittent all-white tiles on this device).
                glBindTexture(GL_TEXTURE_2D, sel.contentTex);
                glBindFramebuffer(GL_FRAMEBUFFER, tile.fbo);
                glViewport(kApron, kApron, kTileSize, kTileSize);
                float tileX0 = static_cast<float>(tx * kTileSize);
                float tileY0 = static_cast<float>(ty * kTileSize);
                glUniform2f(g_sel.uC0, c0x - tileX0, c0y - tileY0);
                glUniform2f(g_sel.uC1, c1x - tileX0, c1y - tileY0);
                glUniform2f(g_sel.uC2, c2x - tileX0, c2y - tileY0);
                glUniform2f(g_sel.uC3, c3x - tileX0, c3y - tileY0);
                PageClip tilePage = page;
                if (tilePage.active) {
                    tilePage.minX -= tileX0;
                    tilePage.minY -= tileY0;
                    tilePage.maxX -= tileX0;
                    tilePage.maxY -= tileY0;
                }
                uploadPageClip(g_sel.uPageMin, g_sel.uPageMax,
                               g_sel.uPageActive, tilePage);
                glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
                saveTileToDisk(sel.layerIdx, tileKey(tx, ty));
                // Drop changed this tile's content; mark it + 8
                // neighbors' aprons stale for resync at next composite.
                markApronStaleAround(layer, tx, ty);
            }
        }
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    // Build afterTiles snapshots from the post-drop state. saveTileToDisk
    // above already updated each touched tile's cachedBytes from the
    // post-drop FBO, so we share those refcounted buffers here.
    entry.afterTiles.reserve(touchedTiles.size());
    for (auto& kv : touchedTiles) {
        int tx = kv.second.first, ty = kv.second.second;
        TileSnap snap;
        snap.tx = tx; snap.ty = ty;
        auto it = layer.tiles.find(kv.first);
        if (it != layer.tiles.end()) {
            snap.existed = true;
            if (it->second.cachedBytes) {
                snap.bytes = it->second.cachedBytes;
            } else {
                auto fresh = tilePool().acquire();
                glBindFramebuffer(GL_FRAMEBUFFER, it->second.fbo);
                glReadPixels(kApron, kApron, kTileSize, kTileSize,
                             GL_RGBA, GL_UNSIGNED_BYTE, fresh->data());
                it->second.cachedBytes = fresh;
                snap.bytes = fresh;
            }
        }
        entry.afterTiles.push_back(std::move(snap));
    }

    glBindFramebuffer(GL_FRAMEBUFFER, prevFbo);

    if (sel.contentTex) glDeleteTextures(1, &sel.contentTex);

    // Push undo entry only if anything actually differs.
    bool diff = false;
    // Build a key→idx map for fast pairing.
    std::unordered_map<int64_t, size_t> beforeIdx;
    for (size_t i = 0; i < entry.beforeTiles.size(); ++i) {
        beforeIdx[tileKey(entry.beforeTiles[i].tx, entry.beforeTiles[i].ty)] = i;
    }
    for (const auto& a : entry.afterTiles) {
        auto it = beforeIdx.find(tileKey(a.tx, a.ty));
        if (it == beforeIdx.end()) { diff = true; break; }
        const auto& b = entry.beforeTiles[it->second];
        if (b.existed != a.existed
         || (b.existed && b.bytes != a.bytes
             && (!b.bytes || !a.bytes
                 || std::memcmp(b.bytes->data(), a.bytes->data(),
                                b.bytes->size()) != 0))) {
            diff = true; break;
        }
    }
    if (diff) pushUndoEntry(std::move(entry));
}

// Discard the floating selection WITHOUT restoring the lifted tiles —
// the source layer keeps the hole that was punched out at lift time.
// Used by cut: the clipboard already holds the pixels, so the source
// loses them. Pushes a RasterStroke undo entry whose beforeTiles are
// the pre-lift snapshots (held on the floating selection since the
// lift itself never pushed an entry — the assumption used to be that
// cancel/commit would each balance the books) and whose afterTiles
// are the current hole-punched state. Undo restores the original
// pixels; redo re-applies the hole.
void discardRasterSelectionImpl() {
    GLuint contentTex = 0;
    size_t layerIdx = 0;
    std::vector<TileSnap> beforeTiles;
    {
        std::lock_guard<std::mutex> lock(g_rasterSelMutex);
        if (!g_rasterSel.active) return;
        contentTex  = g_rasterSel.contentTex;
        layerIdx    = g_rasterSel.layerIdx;
        beforeTiles = std::move(g_rasterSel.liftedTiles);
        g_rasterSel.contentTex = 0;
        g_rasterSel.active = false;
    }
    if (contentTex) glDeleteTextures(1, &contentTex);

    if (beforeTiles.empty()) return;
    if (layerIdx >= layers().size() || !layers()[layerIdx]) return;
    Layer& layer = *layers()[layerIdx];

    // Snapshot the same tile keys' current (hole) state for redo.
    std::vector<TileSnap> afterTiles;
    afterTiles.reserve(beforeTiles.size());
    GLint prevFbo = 0;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &prevFbo);
    for (const auto& b : beforeTiles) {
        TileSnap a;
        a.tx = b.tx;
        a.ty = b.ty;
        auto it = layer.tiles.find(tileKey(b.tx, b.ty));
        if (it != layer.tiles.end()) {
            a.existed = true;
            if (it->second.cachedBytes) {
                a.bytes = it->second.cachedBytes;
            } else {
                auto fresh = tilePool().acquire();
                glBindFramebuffer(GL_READ_FRAMEBUFFER, it->second.fbo);
                glReadPixels(kApron, kApron, kTileSize, kTileSize,
                             GL_RGBA, GL_UNSIGNED_BYTE, fresh->data());
                it->second.cachedBytes = fresh;
                a.bytes = fresh;
            }
        }
        afterTiles.push_back(std::move(a));
    }
    glBindFramebuffer(GL_FRAMEBUFFER, prevFbo);

    UndoEntry entry;
    entry.op          = UndoOp::RasterStroke;
    entry.layerIdx    = layerIdx;
    entry.beforeTiles = std::move(beforeTiles);
    entry.afterTiles  = std::move(afterTiles);
    pushUndoEntry(std::move(entry));
}

// Discard the floating selection by restoring each lifted tile's
// pre-lift bytes. No undo entry is pushed (net zero change).
void cancelRasterSelectionImpl() {
    // Restoring lifted tiles writes back outside any stroke's bbox
    // and the floating-selection overlay disappears from the
    // compositor — both invalidate the partial-recomposite cache.
    g_mbCacheValid = false;
    std::vector<TileSnap> snaps;
    GLuint contentTex = 0;
    size_t layerIdx = 0;
    {
        std::lock_guard<std::mutex> lock(g_rasterSelMutex);
        if (!g_rasterSel.active) return;
        snaps = std::move(g_rasterSel.liftedTiles);
        contentTex = g_rasterSel.contentTex;
        layerIdx = g_rasterSel.layerIdx;
        g_rasterSel.contentTex = 0;
        g_rasterSel.active = false;
    }
    if (layerIdx < layers().size() && layers()[layerIdx]) {
        for (const auto& s : snaps) {
            applyTileSnap(layerIdx, s);
        }
    }
    if (contentTex) glDeleteTextures(1, &contentTex);
}

// Copy the active floating raster selection's pixels (and current OBB)
// into the global clipboard. Selection state is left unchanged. No-op
// if no selection is active. Runs on the GL thread (uses a temporary
// FBO + glReadPixels to read the contentTex back to CPU).
void copyRasterSelectionImpl() {
    GLuint contentTex = 0;
    int    w = 0, h = 0;
    float  cx = 0, cy = 0, hw = 0, hh = 0, rot = 0;
    {
        std::lock_guard<std::mutex> lock(g_rasterSelMutex);
        if (!g_rasterSel.active) return;
        contentTex = g_rasterSel.contentTex;
        w  = g_rasterSel.contentW;
        h  = g_rasterSel.contentH;
        cx = g_rasterSel.centerX;
        cy = g_rasterSel.centerY;
        hw = g_rasterSel.halfW;
        hh = g_rasterSel.halfH;
        rot = g_rasterSel.rotation;
    }
    if (contentTex == 0 || w <= 0 || h <= 0) return;

    GLint prevFbo = 0;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &prevFbo);
    GLuint tmpFbo = 0;
    glGenFramebuffers(1, &tmpFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, tmpFbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, contentTex, 0);
    std::vector<uint8_t> bytes(static_cast<size_t>(w) * h * 4);
    glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, bytes.data());
    glBindFramebuffer(GL_FRAMEBUFFER, prevFbo);
    glDeleteFramebuffers(1, &tmpFbo);

    {
        std::lock_guard<std::mutex> lock(g_rasterClipboardMutex);
        g_rasterClipboard.bytes    = std::move(bytes);
        g_rasterClipboard.w        = w;
        g_rasterClipboard.h        = h;
        g_rasterClipboard.centerX  = cx;
        g_rasterClipboard.centerY  = cy;
        g_rasterClipboard.halfW    = hw;
        g_rasterClipboard.halfH    = hh;
        g_rasterClipboard.rotation = rot;
        g_rasterClipboard.present  = true;
    }
    g_clipboardKind.store(1, std::memory_order_release);   // 1 = Raster
}

// Create a fresh floating raster selection from the global clipboard at
// the same doc-coord OBB as the original copy (so Copy → Paste behaves
// like duplicate-in-place; the user can drag the new floating sel off
// to reveal the source). Auto-commits any existing floating selection
// first. Returns false if the clipboard is empty or the active layer
// isn't raster.
bool pasteRasterSelectionImpl() {
    RasterClipboard cb;
    {
        std::lock_guard<std::mutex> lock(g_rasterClipboardMutex);
        if (!g_rasterClipboard.present) return false;
        cb = g_rasterClipboard;     // copies bytes — clipboard stays valid
    }

    // If a floating selection is already active, commit it so the paste
    // becomes the new active one. (commit may push an undo entry; the
    // paste's own commit later will push a separate entry, so undo
    // lands on the paste first, then the prior commit.)
    bool active = false;
    {
        std::lock_guard<std::mutex> lock(g_rasterSelMutex);
        active = g_rasterSel.active;
    }
    if (active) commitRasterSelectionImpl();

    ensureAtLeastOneLayer();
    if (activeLayer() >= layers().size() || !layers()[activeLayer()]) {
        return false;
    }
    Layer& layer = *layers()[activeLayer()];
    if (layer.type != LayerType::Raster) return false;

    GLuint contentTex = 0;
    glGenTextures(1, &contentTex);
    glBindTexture(GL_TEXTURE_2D, contentTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, cb.w, cb.h, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, cb.bytes.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);

    std::lock_guard<std::mutex> lock(g_rasterSelMutex);
    // bbox fields aren't actually consumed after the lift step; set
    // them to the implicit identity bbox at the paste OBB's center so
    // they describe a sensible source rect for any future reader.
    g_rasterSel.bboxMinX = cb.centerX - cb.w * 0.5f;
    g_rasterSel.bboxMinY = cb.centerY - cb.h * 0.5f;
    g_rasterSel.bboxMaxX = cb.centerX + cb.w * 0.5f;
    g_rasterSel.bboxMaxY = cb.centerY + cb.h * 0.5f;
    g_rasterSel.centerX  = cb.centerX;
    g_rasterSel.centerY  = cb.centerY;
    g_rasterSel.halfW    = cb.halfW;
    g_rasterSel.halfH    = cb.halfH;
    g_rasterSel.rotation = cb.rotation;
    g_rasterSel.contentTex = contentTex;
    g_rasterSel.contentW   = cb.w;
    g_rasterSel.contentH   = cb.h;
    // Paste has no source tiles to restore on cancel — cancel just
    // drops the floating selection.
    g_rasterSel.liftedTiles.clear();
    g_rasterSel.layerIdx = activeLayer();
    g_rasterSel.active   = true;
    return true;
}

void applyBucketFill(JNIEnv* env, float seedDocX, float seedDocY,
                     uint32_t fillRgb) {
    PageClip page = readPageClip();
    if (!page.active) return;
    if (seedDocX < page.minX || seedDocX >= page.maxX ||
        seedDocY < page.minY || seedDocY >= page.maxY) return;

    int pageW = static_cast<int>(page.maxX - page.minX);
    int pageH = static_cast<int>(page.maxY - page.minY);
    if (pageW <= 0 || pageH <= 0) return;

    // Active layer must be raster — vector layers don't have pixel tiles
    // for the fill to land in.
    ensureAtLeastOneLayer();
    if (activeLayer() >= layers().size() || !layers()[activeLayer()]) return;
    Layer& layer = *layers()[activeLayer()];
    if (layer.type != LayerType::Raster) {
        LOGI("bucket fill: active layer is vector, ignoring");
        return;
    }

    using Clock = std::chrono::steady_clock;
    auto t0 = Clock::now();
    auto millis = [](Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    GLint prevFbo = 0;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &prevFbo);

    // (1) One-shot composite FBO at page resolution. Doc-to-buffer
    // transform is identity-scale + translate (page.min → 0); positive
    // y-slope so glReadPixels' bottom-up byte order lands doc-top in
    // byte-row-0 (matching tile-byte orientation).
    // Drain any pre-existing GL error so the fresh diagnostics below
    // accurately reflect THIS function's calls.
    while (glGetError() != GL_NO_ERROR) {}

    // One-shot logging of GPU limits so we can spot a "page is bigger
    // than the device's max renderbuffer" case if it ever bites us.
    {
        static bool logged = false;
        if (!logged) {
            GLint maxTex = 0, maxRb = 0;
            glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxTex);
            glGetIntegerv(GL_MAX_RENDERBUFFER_SIZE, &maxRb);
            const char* ver = reinterpret_cast<const char*>(glGetString(GL_VERSION));
            const char* ren = reinterpret_cast<const char*>(glGetString(GL_RENDERER));
            LOGI("bucket fill: GL_VERSION=%s RENDERER=%s "
                 "MAX_TEXTURE_SIZE=%d MAX_RENDERBUFFER_SIZE=%d",
                 ver ? ver : "?", ren ? ren : "?", maxTex, maxRb);
            logged = true;
        }
    }

    // Use a renderbuffer for the color attachment (more conventional for
    // offscreen FBOs than a texture, and avoids texture-completeness
    // quirks some drivers apply to NPOT FBO color textures). We still
    // glReadPixels off the bound FBO afterward, just like before.
    GLuint compFbo = 0, compRb = 0;
    glGenRenderbuffers(1, &compRb);
    glBindRenderbuffer(GL_RENDERBUFFER, compRb);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, pageW, pageH);
    {
        GLenum err = glGetError();
        if (err != GL_NO_ERROR) {
            LOGE("bucket fill: glRenderbufferStorage %dx%d failed (0x%x)",
                 pageW, pageH, err);
            glDeleteRenderbuffers(1, &compRb);
            return;
        }
    }

    glGenFramebuffers(1, &compFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, compFbo);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                              GL_RENDERBUFFER, compRb);
    {
        GLenum err = glGetError();
        if (err != GL_NO_ERROR) {
            LOGE("bucket fill: glFramebufferRenderbuffer failed (0x%x)", err);
        }
        GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if (status != GL_FRAMEBUFFER_COMPLETE) {
            LOGE("bucket fill: composite FBO incomplete: 0x%x (size=%dx%d, "
                 "post-check err=0x%x)",
                 status, pageW, pageH, glGetError());
            glBindFramebuffer(GL_FRAMEBUFFER, prevFbo);
            glDeleteFramebuffers(1, &compFbo);
            glDeleteRenderbuffers(1, &compRb);
            return;
        }
    }

    float t[16] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        -page.minX, -page.minY, 0.0f, 1.0f
    };

    // Force view scale = 1 so screen-relative widths render at their
    // "natural" doc-px width (1:1 in this transform). Restore after.
    uint32_t savedScaleBits = g_viewScaleBits.load();
    {
        float one = 1.0f;
        uint32_t bits;
        std::memcpy(&bits, &one, sizeof(bits));
        g_viewScaleBits.store(bits);
    }

    jfloatArray jtransform = env->NewFloatArray(16);
    env->SetFloatArrayRegion(jtransform, 0, 16, t);
    auto tCompositeStart = Clock::now();
    // Bucket fill source = paper-white background + the active raster
    // layer's tiles ONLY. Vector lines on other layers, raster strokes
    // on hidden/visible siblings, and the page-grid don't act as fill
    // boundaries — the user expects the bucket to follow the active
    // layer's content. Force opacity = 1.0 so a partly-transparent
    // active layer still produces a clean boundary image.
    glViewport(0, 0, pageW, pageH);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    bindRasterCompositePipeline(env, pageW, pageH, jtransform);
    compositeRasterLayer(layer, /*opacityOverride=*/ 1.0f);
    glFinish();   // ensure render is done before timing the readback
    auto tComposite = Clock::now();
    env->DeleteLocalRef(jtransform);

    g_viewScaleBits.store(savedScaleBits);

    // (2) Read pixels and free the FBO.
    std::vector<uint8_t> pixels(static_cast<size_t>(pageW) * pageH * 4);
    glReadPixels(0, 0, pageW, pageH, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    auto tReadback = Clock::now();
    glBindFramebuffer(GL_FRAMEBUFFER, prevFbo);
    glDeleteFramebuffers(1, &compFbo);
    glDeleteRenderbuffers(1, &compRb);

    // (3) Iterative 4-way flood fill. RGBA Chebyshev tolerance of 16
    // catches anti-aliased boundary pixels without over-flooding.
    int sx = static_cast<int>(seedDocX - page.minX);
    int sy = static_cast<int>(seedDocY - page.minY);
    if (sx < 0 || sx >= pageW || sy < 0 || sy >= pageH) return;

    constexpr int kTolerance = 16;
    std::vector<uint8_t> mask(static_cast<size_t>(pageW) * pageH, 0);
    const uint8_t* p0 = pixels.data() + (static_cast<size_t>(sy) * pageW + sx) * 4;
    int seedR = p0[0], seedG = p0[1], seedB = p0[2], seedA = p0[3];

    // Track the mask's bbox as we flood so the dilation scan only has
    // to revisit the area we actually painted into, not the whole page.
    int maskMinX = pageW, maskMaxX = -1;
    int maskMinY = pageH, maskMaxY = -1;

    std::vector<std::pair<int, int>> stack;
    stack.reserve(1024);
    stack.push_back({sx, sy});
    while (!stack.empty()) {
        auto [x, y] = stack.back();
        stack.pop_back();
        if (x < 0 || x >= pageW || y < 0 || y >= pageH) continue;
        size_t off = static_cast<size_t>(y) * pageW + x;
        if (mask[off]) continue;
        const uint8_t* px = pixels.data() + off * 4;
        if (std::abs(static_cast<int>(px[0]) - seedR) > kTolerance ||
            std::abs(static_cast<int>(px[1]) - seedG) > kTolerance ||
            std::abs(static_cast<int>(px[2]) - seedB) > kTolerance ||
            std::abs(static_cast<int>(px[3]) - seedA) > kTolerance) continue;
        mask[off] = 1;
        if (x < maskMinX) maskMinX = x;
        if (x > maskMaxX) maskMaxX = x;
        if (y < maskMinY) maskMinY = y;
        if (y > maskMaxY) maskMaxY = y;
        stack.push_back({x + 1, y});
        stack.push_back({x - 1, y});
        stack.push_back({x, y + 1});
        stack.push_back({x, y - 1});
    }
    auto tFlood = Clock::now();

    // Dilate the mask outward by a few pixels so the fill bleeds into
    // the boundary's anti-aliased gradient. Without this, even with a
    // generous tolerance, a thin halo of un-filled "almost-but-not-quite
    // similar to seed" pixels remains between the fill and the stroke.
    //
    // Implementation: frontier propagation. The naive "iterate every
    // mask pixel each pass" approach is O(W*H) per iteration; for a
    // ~3-megapixel mask with two iterations that was the dominant cost
    // of the entire fill. The frontier version does ONE linear scan
    // (which we can't avoid: we need the initial outer ring of the
    // mask) and then per-iteration work is proportional to the mask
    // PERIMETER, not its area — typically a 50-100x speedup.
    int kDilatePx = g_bucketBleedPx.load(std::memory_order_relaxed);
    if (kDilatePx < 0) kDilatePx = 0;
    if (maskMaxX >= maskMinX) {     // skip when flood marked nothing
        // Restrict the initial-scan area to the mask's bbox plus the
        // dilation distance — any pixel outside that range can't be on
        // the dilation wavefront. Single biggest speedup of the whole
        // dilation step on small fills in big pages.
        int sx0 = std::max(0,     maskMinX - kDilatePx - 1);
        int sx1 = std::min(pageW, maskMaxX + kDilatePx + 2);
        int sy0 = std::max(0,     maskMinY - kDilatePx - 1);
        int sy1 = std::min(pageH, maskMaxY + kDilatePx + 2);

        std::vector<std::pair<int, int>> frontier;
        frontier.reserve(8192);
        for (int y = sy0; y < sy1; ++y) {
            size_t row = static_cast<size_t>(y) * pageW;
            for (int x = sx0; x < sx1; ++x) {
                if (mask[row + x]) continue;
                bool nb =
                    (x > 0          && mask[row + x - 1])     ||
                    (x + 1 < pageW  && mask[row + x + 1])     ||
                    (y > 0          && mask[row - pageW + x]) ||
                    (y + 1 < pageH  && mask[row + pageW + x]);
                if (nb) frontier.emplace_back(x, y);
            }
        }
        for (int iter = 0; iter < kDilatePx; ++iter) {
            // Mark current frontier (it's now part of the dilated mask).
            for (auto [x, y] : frontier) {
                mask[static_cast<size_t>(y) * pageW + x] = 1;
            }
            // Collect next frontier from current frontier's neighbors.
            // Mark provisionally as we go to avoid duplicates from
            // sibling frontier pixels touching the same neighbor.
            std::vector<std::pair<int, int>> nextFront;
            nextFront.reserve(frontier.size() * 2);
            for (auto [x, y] : frontier) {
                auto consider = [&](int nx, int ny) {
                    if (nx < 0 || nx >= pageW
                     || ny < 0 || ny >= pageH) return;
                    size_t off = static_cast<size_t>(ny) * pageW + nx;
                    if (mask[off]) return;
                    mask[off] = 1;
                    nextFront.emplace_back(nx, ny);
                };
                consider(x + 1, y);
                consider(x - 1, y);
                consider(x, y + 1);
                consider(x, y - 1);
            }
            frontier = std::move(nextFront);
        }
    }
    auto tDilate = Clock::now();

    // (4) Apply mask to active layer's tiles. Build the affected-tile
    // list first so undo only snapshots tiles that actually changed.
    int tx0 = static_cast<int>(std::floor(page.minX / kTileSizeF));
    int tx1 = static_cast<int>(std::floor((page.maxX - 1) / kTileSizeF));
    int ty0 = static_cast<int>(std::floor(page.minY / kTileSizeF));
    int ty1 = static_cast<int>(std::floor((page.maxY - 1) / kTileSizeF));

    std::vector<std::pair<int, int>> affected;
    affected.reserve(static_cast<size_t>(tx1 - tx0 + 1) * (ty1 - ty0 + 1));
    for (int ty = ty0; ty <= ty1; ++ty) {
        int tileDocY0 = ty * kTileSize;
        for (int tx = tx0; tx <= tx1; ++tx) {
            int tileDocX0 = tx * kTileSize;
            // Quick scan: any masked pixel in the tile's intersection with
            // the page rect?
            bool any = false;
            for (int ly = 0; ly < kTileSize && !any; ++ly) {
                int doc_y = tileDocY0 + ly;
                if (doc_y < static_cast<int>(page.minY)
                 || doc_y >= static_cast<int>(page.maxY)) continue;
                size_t maskRow = static_cast<size_t>(doc_y - static_cast<int>(page.minY)) * pageW;
                for (int lx = 0; lx < kTileSize; ++lx) {
                    int doc_x = tileDocX0 + lx;
                    if (doc_x < static_cast<int>(page.minX)
                     || doc_x >= static_cast<int>(page.maxX)) continue;
                    if (mask[maskRow + (doc_x - static_cast<int>(page.minX))]) {
                        any = true; break;
                    }
                }
            }
            if (any) affected.emplace_back(tx, ty);
        }
    }
    auto tAffected = Clock::now();
    if (affected.empty()) return;

    UndoEntry entry;
    entry.op = UndoOp::RasterStroke;
    entry.layerIdx = activeLayer();
    entry.beforeTiles.reserve(affected.size());
    entry.afterTiles.reserve(affected.size());

    uint8_t fr = (fillRgb >> 16) & 0xFF;
    uint8_t fg = (fillRgb >>  8) & 0xFF;
    uint8_t fb =  fillRgb        & 0xFF;

    // The opacity slider's user-facing value is "target stroke opacity"
    // — the native side stores the per-dab α derived via the curve
    // α = 1 - (1 - target)^(1/N). Bucket fill is a single one-shot
    // application (not a stack of overlapping dabs), so we want the
    // user's target opacity directly. Invert the curve to recover it.
    constexpr int kAlphaDabsPerOverlap = 6;
    float perDabAlpha = currentBrushAlpha();
    float fillAlpha = 1.0f - std::pow(1.0f - perDabAlpha, kAlphaDabsPerOverlap);
    if (fillAlpha < 0.0f) fillAlpha = 0.0f;
    if (fillAlpha > 1.0f) fillAlpha = 1.0f;

    GLint prevFbo2 = 0;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &prevFbo2);
    std::vector<uint8_t> tilePixels(kTileBytes);
    for (auto [tx, ty] : affected) {
        int tileDocX0 = tx * kTileSize;
        int tileDocY0 = ty * kTileSize;

        // (a) Capture before-state. Share the cache if present, fall
        // back to glReadPixels for tiles whose cache hasn't been
        // populated yet. Tiles that don't yet exist are conceptually
        // all-transparent.
        TileSnap before;
        before.tx = tx; before.ty = ty;
        auto existing = layer.tiles.find(tileKey(tx, ty));
        if (existing != layer.tiles.end()) {
            before.existed = true;
            if (existing->second.cachedBytes) {
                before.bytes = existing->second.cachedBytes;
            } else {
                auto fresh = tilePool().acquire();
                glBindFramebuffer(GL_FRAMEBUFFER, existing->second.fbo);
                glReadPixels(kApron, kApron, kTileSize, kTileSize,
                             GL_RGBA, GL_UNSIGNED_BYTE, fresh->data());
                existing->second.cachedBytes = fresh;
                before.bytes = fresh;
            }
            std::memcpy(tilePixels.data(), before.bytes->data(), kTileBytes);
        } else {
            std::memset(tilePixels.data(), 0, kTileBytes);
        }

        // (b) Apply mask in CPU.
        for (int ly = 0; ly < kTileSize; ++ly) {
            int doc_y = tileDocY0 + ly;
            if (doc_y < static_cast<int>(page.minY)
             || doc_y >= static_cast<int>(page.maxY)) continue;
            size_t maskRow = static_cast<size_t>(doc_y - static_cast<int>(page.minY)) * pageW;
            for (int lx = 0; lx < kTileSize; ++lx) {
                int doc_x = tileDocX0 + lx;
                if (doc_x < static_cast<int>(page.minX)
                 || doc_x >= static_cast<int>(page.maxX)) continue;
                if (!mask[maskRow + (doc_x - static_cast<int>(page.minX))]) continue;
                size_t off = (static_cast<size_t>(ly) * kTileSize + lx) * 4;
                // Premultiplied "src over dst" using the user's chosen
                // fill opacity. With α=1 this collapses to a pure
                // replace (matching the original full-opacity behavior),
                // and at α<1 the pre-existing tile pixel shows through.
                float invA = 1.0f - fillAlpha;
                float sr = fr * fillAlpha;
                float sg = fg * fillAlpha;
                float sb = fb * fillAlpha;
                float sa = 255.0f * fillAlpha;
                float dr = static_cast<float>(tilePixels[off + 0]);
                float dg = static_cast<float>(tilePixels[off + 1]);
                float db = static_cast<float>(tilePixels[off + 2]);
                float da = static_cast<float>(tilePixels[off + 3]);
                tilePixels[off + 0] =
                    static_cast<uint8_t>(std::lround(sr + dr * invA));
                tilePixels[off + 1] =
                    static_cast<uint8_t>(std::lround(sg + dg * invA));
                tilePixels[off + 2] =
                    static_cast<uint8_t>(std::lround(sb + db * invA));
                tilePixels[off + 3] =
                    static_cast<uint8_t>(std::lround(sa + da * invA));
            }
        }

        // (c) Upload + persist + record undo. tilePixels is now the new
        // tile state, so it doubles as the after-snapshot bytes.
        Tile& tile = getOrCreateTile(layer, tx, ty);
        glBindTexture(GL_TEXTURE_2D, tile.texture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, kApron, kApron, kTileSize, kTileSize,
                        GL_RGBA, GL_UNSIGNED_BYTE, tilePixels.data());
        writeTileBytesToDisk(activeLayer(), tx, ty, tilePixels.data());
        // Bucket fill changed this tile; flag for apron resync.
        markApronStaleAround(layer, tx, ty);

        // Build the AFTER snap into a fresh shared buffer and reuse
        // it as the new cache, so the next BEFORE snapshot for this
        // tile is free.
        auto fresh = acquireTileBytesFrom(tilePixels.data());
        tile.cachedBytes = fresh;
        TileSnap after;
        after.tx = tx; after.ty = ty;
        after.existed = true;
        after.bytes = fresh;
        entry.beforeTiles.push_back(std::move(before));
        entry.afterTiles.push_back(std::move(after));
    }
    glBindFramebuffer(GL_FRAMEBUFFER, prevFbo2);

    auto tApply = Clock::now();

    // Push undo entry if any tile actually changed (a fill that hit only
    // already-fill-color pixels would produce identical before/after).
    bool diff = false;
    for (size_t i = 0; i < entry.beforeTiles.size() && !diff; ++i) {
        const auto& b = entry.beforeTiles[i];
        const auto& a = entry.afterTiles[i];
        if (b.existed != a.existed
         || (b.existed && b.bytes != a.bytes
             && (!b.bytes || !a.bytes
                 || std::memcmp(b.bytes->data(), a.bytes->data(),
                                b.bytes->size()) != 0))) {
            diff = true;
        }
    }
    if (diff) pushUndoEntry(std::move(entry));
    auto tEnd = Clock::now();

    LOGI("bucket fill timings (ms): composite=%.1f readback=%.1f flood=%.1f "
         "dilate=%.1f detect=%.1f apply=%.1f undo=%.1f total=%.1f "
         "tiles=%zu pageW=%d pageH=%d",
         millis(tCompositeStart, tComposite),
         millis(tComposite, tReadback),
         millis(tReadback, tFlood),
         millis(tFlood, tDilate),
         millis(tDilate, tAffected),
         millis(tAffected, tApply),
         millis(tApply, tEnd),
         millis(t0, tEnd),
         affected.size(), pageW, pageH);
}

// ---- Vector clipboard helpers --------------------------------------------
//
// Forward-declared near the raster equivalents at the top of the
// anonymous namespace; defined here so they sit alongside the
// matching JNI dispatchers below.

// Snapshot every selected vector shape (primary + extras) into the
// vector clipboard. No-op if nothing is selected or the layer/shape
// pointers are stale. Selection state is left unchanged. Sets
// g_clipboardKind so a subsequent paste knows to drop vector shapes.
void copyVectorSelectionImpl() {
    Selection primary;
    std::vector<Selection> extras;
    {
        std::lock_guard<std::mutex> lock(g_selectionMutex);
        primary = g_selection;
        extras  = g_extraSelections;
    }
    if (primary.kind == ShapeKind::None) return;

    auto snapshot = [&](const Selection& s, ShapeData& out) -> bool {
        if (s.layerIdx >= layers().size() || !layers()[s.layerIdx]) return false;
        Layer& layer = *layers()[s.layerIdx];
        if (layer.type != LayerType::Vector) return false;
        out.kind = s.kind;
        switch (s.kind) {
            case ShapeKind::Line:
                if (s.shapeIdx >= layer.lines.size())    return false;
                out.line    = layer.lines[s.shapeIdx];    return true;
            case ShapeKind::Rect:
                if (s.shapeIdx >= layer.rects.size())    return false;
                out.rect    = layer.rects[s.shapeIdx];    return true;
            case ShapeKind::Ellipse:
                if (s.shapeIdx >= layer.ellipses.size()) return false;
                out.ellipse = layer.ellipses[s.shapeIdx]; return true;
            case ShapeKind::Circle:
                if (s.shapeIdx >= layer.circles.size())  return false;
                out.circle  = layer.circles[s.shapeIdx];  return true;
            default: return false;
        }
    };

    std::vector<ShapeData> shapes;
    ShapeData primarySD;
    if (!snapshot(primary, primarySD)) return;
    shapes.push_back(primarySD);
    for (const auto& e : extras) {
        ShapeData sd;
        if (snapshot(e, sd)) shapes.push_back(sd);
    }

    {
        std::lock_guard<std::mutex> lock(g_vectorClipboardMutex);
        g_vectorClipboard.present = true;
        g_vectorClipboard.shapes  = std::move(shapes);
    }
    g_clipboardKind.store(2, std::memory_order_release);   // 2 = Vector
}

// Forward decl to share delete logic with deleteSelection JNI.
void deleteAllSelectionsImpl();

// Cut for vectors: copy every selected shape then erase them all,
// pushing a VectorDelete undo entry per shape (descending index order
// per layer/kind so erases don't invalidate later targets).
void cutVectorSelectionImpl() {
    copyVectorSelectionImpl();
    if (g_clipboardKind.load() != 2) return;   // copy didn't take
    deleteAllSelectionsImpl();
}

// Erase every currently-selected vector shape (primary + extras) and
// push one VectorDelete entry per shape. Shared between the
// deleteSelection JNI and cutVectorSelectionImpl.
void deleteAllSelectionsImpl() {
    std::vector<Selection> all;
    {
        std::lock_guard<std::mutex> lock(g_selectionMutex);
        if (g_selection.kind != ShapeKind::None) all.push_back(g_selection);
        for (const auto& e : g_extraSelections) all.push_back(e);
        g_selection = Selection{};
        g_extraSelections.clear();
    }
    if (all.empty()) return;

    std::sort(all.begin(), all.end(),
              [](const Selection& a, const Selection& b) {
                  if (a.layerIdx != b.layerIdx) return a.layerIdx > b.layerIdx;
                  if (a.kind != b.kind)
                      return static_cast<int>(a.kind) > static_cast<int>(b.kind);
                  return a.shapeIdx > b.shapeIdx;
              });

    std::vector<size_t> affectedLayers;
    for (const auto& s : all) {
        if (s.layerIdx >= layers().size() || !layers()[s.layerIdx]) continue;
        Layer& layer = *layers()[s.layerIdx];
        if (layer.type != LayerType::Vector) continue;

        UndoEntry entry;
        entry.op       = UndoOp::VectorDelete;
        entry.layerIdx = s.layerIdx;
        entry.shapeIdx = s.shapeIdx;
        entry.beforeShape.kind = s.kind;
        bool captured = false;
        switch (s.kind) {
            case ShapeKind::Line:
                if (s.shapeIdx < layer.lines.size()) {
                    entry.beforeShape.line = layer.lines[s.shapeIdx];
                    layer.lines.erase(layer.lines.begin() + s.shapeIdx);
                    captured = true;
                }
                break;
            case ShapeKind::Rect:
                if (s.shapeIdx < layer.rects.size()) {
                    entry.beforeShape.rect = layer.rects[s.shapeIdx];
                    layer.rects.erase(layer.rects.begin() + s.shapeIdx);
                    captured = true;
                }
                break;
            case ShapeKind::Ellipse:
                if (s.shapeIdx < layer.ellipses.size()) {
                    entry.beforeShape.ellipse = layer.ellipses[s.shapeIdx];
                    layer.ellipses.erase(layer.ellipses.begin() + s.shapeIdx);
                    captured = true;
                }
                break;
            case ShapeKind::Circle:
                if (s.shapeIdx < layer.circles.size()) {
                    entry.beforeShape.circle = layer.circles[s.shapeIdx];
                    layer.circles.erase(layer.circles.begin() + s.shapeIdx);
                    captured = true;
                }
                break;
            case ShapeKind::None: break;
        }
        if (captured) {
            pushUndoEntry(std::move(entry));
            if (std::find(affectedLayers.begin(), affectedLayers.end(),
                          s.layerIdx) == affectedLayers.end()) {
                affectedLayers.push_back(s.layerIdx);
            }
        }
    }
    for (size_t idx : affectedLayers) {
        if (idx < layers().size() && layers()[idx]) {
            saveVectorLayer(idx, *layers()[idx]);
        }
    }
}

// Paste from the vector clipboard: append every clipboard shape to
// the active layer (which must be vector). Returns false if the
// clipboard is empty or the active layer isn't vector. Pushes one
// VectorAdd entry per shape. Sets the selection to the pasted shapes
// (primary = first, extras = rest) so the user can immediately drag
// the whole group.
bool pasteVectorSelectionImpl() {
    std::vector<ShapeData> shapes;
    {
        std::lock_guard<std::mutex> lock(g_vectorClipboardMutex);
        if (!g_vectorClipboard.present) return false;
        shapes = g_vectorClipboard.shapes;
    }
    if (shapes.empty()) return false;
    if (activeLayer() >= layers().size() || !layers()[activeLayer()]) return false;
    Layer& layer = *layers()[activeLayer()];
    if (layer.type != LayerType::Vector) return false;

    std::vector<Selection> pasted;
    pasted.reserve(shapes.size());
    for (const auto& sd : shapes) {
        size_t idx = 0;
        bool ok = true;
        switch (sd.kind) {
            case ShapeKind::Line:
                idx = layer.lines.size();    layer.lines.push_back(sd.line);       break;
            case ShapeKind::Rect:
                idx = layer.rects.size();    layer.rects.push_back(sd.rect);       break;
            case ShapeKind::Ellipse:
                idx = layer.ellipses.size(); layer.ellipses.push_back(sd.ellipse); break;
            case ShapeKind::Circle:
                idx = layer.circles.size();  layer.circles.push_back(sd.circle);   break;
            default: ok = false; break;
        }
        if (!ok) continue;

        UndoEntry entry;
        entry.op          = UndoOp::VectorAdd;
        entry.layerIdx    = activeLayer();
        entry.shapeIdx    = idx;
        entry.afterShape  = sd;
        pushUndoEntry(std::move(entry));

        Selection s;
        s.kind     = sd.kind;
        s.layerIdx = activeLayer();
        s.shapeIdx = idx;
        pasted.push_back(s);
    }
    if (pasted.empty()) return false;
    saveVectorLayer(activeLayer(), layer);

    {
        std::lock_guard<std::mutex> lock(g_selectionMutex);
        g_selection = pasted.front();
        g_extraSelections.assign(pasted.begin() + 1, pasted.end());
    }
    return true;
}

}  // namespace

// ---- JNI ------------------------------------------------------------------

extern "C" {

// Block until every queued tile write/delete has been drained.
// Called from Kotlin on app pause and from inside loadDocument so the
// user can't navigate away with pending writes in memory.
JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_flushTileWrites(JNIEnv*, jobject) {
    flushDiskWriter();
}

// Drain deferred saveTileToDisk work queued by commitStroke. Must be
// called from the GL thread (uses GL functions). Called from Kotlin's
// no-stroke onDrawMultiBufferedLayer path (the trailing-edge idle
// frame ~250 ms after the last stroke commit) — see DrawingSurfaceView.
JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_flushPendingSaveTiles(JNIEnv*, jobject) {
    drainPendingSaveTiles();
}

JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_setDocumentDir(JNIEnv* env, jobject, jstring jpath) {
    const char* str = env->GetStringUTFChars(jpath, nullptr);
    g_docDir = str;
    env->ReleaseStringUTFChars(jpath, str);
    LOGI("document dir = %s", g_docDir.c_str());
    // Synchronously pull layer metadata (names, types, visibility,
    // opacity) so the UI thread can read correct state immediately.
    // The GL-touching parts (raster tile uploads, vector shape tables)
    // still happen lazily via ensureLoaded on the next render pass.
    loadAllPageMetadataFromDisk();
}

// Switch the active document at runtime. Frees the current doc's GL
// state, clears selection / undo / pending writes, and points g_docDir
// at the new path. Lazy-loads the new doc on the next render entrypoint.
// Queued through the action drain so all GL work happens on the GL thread.
JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_loadDocument(
        JNIEnv* env, jobject, jstring jpath) {
    const char* str = env->GetStringUTFChars(jpath, nullptr);
    {
        std::lock_guard<std::mutex> lock(g_pendingDocPathMutex);
        g_pendingDocPath = str;
    }
    env->ReleaseStringUTFChars(jpath, str);
    enqueuePendingAction(kActionLoadDocument);
}

JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_setTool(JNIEnv*, jobject, jint tool) {
    g_currentTool.store(tool);
}

JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_setBrushColor(JNIEnv*, jobject, jint rgb) {
    // Mask to 24 bits in case caller passes a packed ARGB; we ignore alpha.
    g_currentBrushColor.store(static_cast<uint32_t>(rgb) & 0x00FFFFFFu);
}

// Eyedropper. UI thread submits a sample request in DOC pixels (matches
// the same coordinate space tile FBOs use internally). The GL thread
// drains at the end of compositeAllLayers, walks visible raster layers,
// reads each layer's tile FBO directly, composites Porter-Duff "over",
// and stores the result. UI polls getLastSampledColor a frame later.
JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_requestColorSample(JNIEnv*, jobject,
                                                      jfloat docX, jfloat docY) {
    int xBits, yBits;
    std::memcpy(&xBits, &docX, sizeof(int));
    std::memcpy(&yBits, &docY, sizeof(int));
    g_pendingSampleDocXBits.store(xBits);
    g_pendingSampleDocYBits.store(yBits);
    // Stale prior result must not be served as the new answer.
    g_lastSampledRgb.store(-1);
    g_pendingSampleHasReq.store(1);
}

// Returns the most recently sampled color as 0xRRGGBB, or -1 if no
// sample has been completed since the last requestColorSample. Single-
// shot: subsequent calls return -1 until another request is submitted.
JNIEXPORT jint JNICALL
Java_com_bk_drawing_NativeRenderer_getLastSampledColor(JNIEnv*, jobject) {
    return g_lastSampledRgb.exchange(-1);
}

// Brush size scale — multiplier on the per-pressure dab radius. Snapshot
// at beginStroke so a slider change mid-stroke doesn't visibly split a
// stroke. Pass any positive float; ignored if non-finite or <= 0.
JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_setBrushSize(JNIEnv*, jobject, jfloat scale) {
    if (!(scale > 0.0f) || !std::isfinite(scale)) return;
    uint32_t bits;
    std::memcpy(&bits, &scale, sizeof(bits));
    g_brushSizeScaleBits.store(bits);
}

// Brush opacity in [0, 1]. Snapshotted at the next beginStroke so a
// mid-stroke slider change doesn't visibly split a stroke. Affects
// brush only; the eraser keeps its historical 0.85 strength.
JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_setBrushAlpha(JNIEnv*, jobject, jfloat alpha) {
    if (!std::isfinite(alpha)) return;
    float clamped = alpha < 0.0f ? 0.0f : (alpha > 1.0f ? 1.0f : alpha);
    uint32_t bits;
    std::memcpy(&bits, &clamped, sizeof(bits));
    g_brushAlphaBits.store(bits);
}

// Stroke alpha mode. When enabled, dabs MAX-blend within a single stroke
// so overlapping dabs cap at the brush alpha instead of accumulating;
// disabled (default) keeps the historical "opacity stacks within a
// stroke" behavior. Snapshotted at beginStroke so a mid-stroke toggle
// doesn't split a stroke between the two modes.
JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_setStrokeUniformAlpha(JNIEnv*, jobject,
                                                        jboolean enabled) {
    g_strokeUniformAlphaEnabled.store(enabled == JNI_TRUE);
}

// Brush "hardness" in [0, 1]. 0 = full radial gradient (soft), 1 =
// solid disc (hard). Snapshotted at beginStroke so mid-stroke changes
// don't split a stroke; affects both brush and eraser dabs.
JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_setBrushHardness(JNIEnv*, jobject, jfloat h) {
    if (!std::isfinite(h)) return;
    float clamped = h < 0.0f ? 0.0f : (h > 1.0f ? 1.0f : h);
    uint32_t bits;
    std::memcpy(&bits, &clamped, sizeof(bits));
    g_brushHardnessBits.store(bits);
}

// Bucket-fill bleed in pixels — how much the filled region grows
// outward past the flood-fill tolerance match. Read at fill time, so
// the next bucket tap picks up changes immediately.
JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_setBucketBleed(JNIEnv*, jobject, jint px) {
    int clamped = px < 0 ? 0 : (px > 64 ? 64 : px);
    g_bucketBleedPx.store(clamped);
}

// Vector tool line width (doc-px). Read at addLine/addRectangle/etc time;
// the captured width travels with the shape forever after. Existing
// shapes are unaffected by later changes.
JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_setVectorLineWidth(JNIEnv*, jobject, jfloat width) {
    if (!(width > 0.0f) || !std::isfinite(width)) return;
    uint32_t bits;
    std::memcpy(&bits, &width, sizeof(bits));
    g_vectorLineWidthBits.store(bits);
}

// Bucket fill at (x, y) in doc-pixels using the current brush color.
// Uses the page composite as the boundary source (so vector outlines on
// other layers can act as boundaries) and writes pixels into the active
// raster layer's tiles. No-op if the active layer is vector or the seed
// is outside the page bounds. Synchronous on the GL thread; commits
// directly to disk and pushes a single undo entry covering the change.
JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_bucketFillAt(
        JNIEnv* env, jobject, jfloat x, jfloat y) {
    ensureInited();
    ensureLoaded();
    applyPendingLayerActions();
    applyPendingShapes();
    ensureAtLeastOneLayer();
    uint32_t fillRgb = g_currentBrushColor.load();
    applyBucketFill(env, x, y, fillRgb);
    // Bucket fill changes pixels across many tiles; the partial-
    // recomposite cache is no longer trustworthy. Force a full
    // re-render at the next renderDocument.
    g_mbCacheValid = false;
}

// ---- Raster selection JNIs ----------------------------------------------

JNIEXPORT jboolean JNICALL
Java_com_bk_drawing_NativeRenderer_beginRasterSelection(
        JNIEnv*, jobject, jfloat x0, jfloat y0, jfloat x1, jfloat y1) {
    ensureInited();
    ensureLoaded();
    applyPendingLayerActions();
    applyPendingShapes();
    ensureAtLeastOneLayer();
    return liftRasterSelectionRect(x0, y0, x1, y1) ? JNI_TRUE : JNI_FALSE;
}

// Lasso (freeform polygon) lift. `points` is a flat [x0,y0,x1,y1,...]
// FloatArray of the doc-coord polyline; the closing edge is implicit
// from the last point to the first. Same lifecycle as
// beginRasterSelection — Kotlin queues this into the multi-buffer pass
// since the lift needs a current GL context.
JNIEXPORT jboolean JNICALL
Java_com_bk_drawing_NativeRenderer_beginLassoSelection(
        JNIEnv* env, jobject, jfloatArray jpoints) {
    ensureInited();
    ensureLoaded();
    applyPendingLayerActions();
    applyPendingShapes();
    ensureAtLeastOneLayer();
    if (!jpoints) return JNI_FALSE;
    jsize len = env->GetArrayLength(jpoints);
    if (len < 6 || (len & 1) != 0) return JNI_FALSE;     // need ≥3 (x,y) pairs
    std::vector<float> pts(static_cast<size_t>(len));
    env->GetFloatArrayRegion(jpoints, 0, len, pts.data());
    bool ok = liftRasterSelectionPolygon(pts.data(),
                                         static_cast<size_t>(len) / 2);
    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_translateRasterSelection(
        JNIEnv*, jobject, jfloat dx, jfloat dy) {
    std::lock_guard<std::mutex> lock(g_rasterSelMutex);
    if (!g_rasterSel.active) return;
    g_rasterSel.centerX += dx;
    g_rasterSel.centerY += dy;
    // The floating selection's screen position changed; the cached
    // frame holds it at the previous location.
    g_mbCacheValid = false;
}

JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_commitRasterSelection(JNIEnv*, jobject) {
    ensureInited();
    ensureLoaded();
    commitRasterSelectionImpl();
}

JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_cancelRasterSelection(JNIEnv*, jobject) {
    ensureInited();
    cancelRasterSelectionImpl();
}

JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_copySelection(JNIEnv*, jobject) {
    ensureInited();
    bool rasterActive = false;
    {
        std::lock_guard<std::mutex> lock(g_rasterSelMutex);
        rasterActive = g_rasterSel.active;
    }
    if (rasterActive) {
        copyRasterSelectionImpl();
    } else {
        copyVectorSelectionImpl();
    }
}

// Cut = copy + discard the floating raster selection without
// restoring its lifted pixels. For vector selections, defers to
// cutVectorSelectionImpl. Source layer keeps the hole / shape
// removed; the clipboard holds the content for a subsequent paste.
JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_cutSelection(JNIEnv*, jobject) {
    ensureInited();
    // Cut removes either pixels or a vector shape from the doc;
    // either way MB content changes outside any stroke's bbox.
    g_mbCacheValid = false;
    bool rasterActive = false;
    {
        std::lock_guard<std::mutex> lock(g_rasterSelMutex);
        rasterActive = g_rasterSel.active;
    }
    if (rasterActive) {
        copyRasterSelectionImpl();
        discardRasterSelectionImpl();
        return;
    }
    cutVectorSelectionImpl();
}

JNIEXPORT jboolean JNICALL
Java_com_bk_drawing_NativeRenderer_pasteSelection(JNIEnv*, jobject) {
    ensureInited();
    ensureLoaded();
    applyPendingLayerActions();
    applyPendingShapes();
    ensureAtLeastOneLayer();
    // Paste creates a new floating selection or vector shape —
    // invalidates the partial-recomposite cache.
    g_mbCacheValid = false;
    int kind = g_clipboardKind.load(std::memory_order_acquire);
    if (kind == 2) {
        return pasteVectorSelectionImpl() ? JNI_TRUE : JNI_FALSE;
    }
    return pasteRasterSelectionImpl() ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_bk_drawing_NativeRenderer_hasClipboardContent(JNIEnv*, jobject) {
    int kind = g_clipboardKind.load(std::memory_order_acquire);
    if (kind == 2) {
        std::lock_guard<std::mutex> lock(g_vectorClipboardMutex);
        return g_vectorClipboard.present ? JNI_TRUE : JNI_FALSE;
    }
    std::lock_guard<std::mutex> lock(g_rasterClipboardMutex);
    return g_rasterClipboard.present ? JNI_TRUE : JNI_FALSE;
}

// 0 = empty, 1 = raster, 2 = vector. Used by the UI to pick the right
// post-paste select tool (raster floating sel vs vector shape sel).
JNIEXPORT jint JNICALL
Java_com_bk_drawing_NativeRenderer_getClipboardKind(JNIEnv*, jobject) {
    return static_cast<jint>(
        g_clipboardKind.load(std::memory_order_acquire));
}

JNIEXPORT jboolean JNICALL
Java_com_bk_drawing_NativeRenderer_hasRasterSelection(JNIEnv*, jobject) {
    std::lock_guard<std::mutex> lock(g_rasterSelMutex);
    return g_rasterSel.active ? JNI_TRUE : JNI_FALSE;
}

// Hit-test: is the doc-coord point currently inside the floating
// selection's transformed bbox? Used by the touch handler to decide
// between "drag to translate" and "tap outside = commit".
JNIEXPORT jboolean JNICALL
Java_com_bk_drawing_NativeRenderer_rasterSelectionContains(
        JNIEnv*, jobject, jfloat x, jfloat y) {
    std::lock_guard<std::mutex> lock(g_rasterSelMutex);
    if (!g_rasterSel.active) return JNI_FALSE;
    // Inlined OBB local-frame test (isPointInsideObb is defined further
    // down in the file; pulling that helper above the forward-decl block
    // would require also moving the Obb struct, which is invasive).
    Obb o = obbForRasterSelection(g_rasterSel);
    float lx, ly;
    rotateWorldToLocal(o, x, y, lx, ly);
    return (std::fabs(lx) <= o.hw && std::fabs(ly) <= o.hh)
           ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_setGridEnabled(JNIEnv*, jobject, jboolean enabled) {
    g_gridEnabled.store(enabled ? 1 : 0);
    g_mbCacheValid = false;
}

JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_setGridStyle(JNIEnv*, jobject, jint style) {
    // Only 1 (lines) or 2 (dots) are valid; clamp.
    int s = (style == 2) ? 2 : 1;
    g_gridStyle.store(s);
    g_mbCacheValid = false;
}

JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_setGridSpacing(JNIEnv*, jobject, jfloat spacing) {
    // Floor at a few doc px: the shader's mod() and the snap rounding
    // both divide by it, and a sub-pixel grid is just a solid wash.
    g_gridSpacing.store(std::max(4.0f, (float)spacing));
    g_mbCacheValid = false;
}

JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_setMarginsEnabled(JNIEnv*, jobject, jboolean enabled) {
    g_marginEnabled.store(enabled ? 1 : 0);
    g_mbCacheValid = false;
}

JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_setMarginSizes(JNIEnv*, jobject,
                                                 jfloat top, jfloat side) {
    g_marginTop.store(std::max(0.0f, (float)top));
    g_marginSide.store(std::max(0.0f, (float)side));
    g_mbCacheValid = false;
}

// Pixel grid (1-doc-pixel boundaries) overlay. Renders only when the
// view scale is high enough to make each doc-pixel several buffer-
// pixels — see kPixelGridMinViewScale.
JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_setPixelGridEnabled(JNIEnv*, jobject,
                                                      jboolean enabled) {
    g_pixelGridEnabled.store(enabled ? 1 : 0);
    g_mbCacheValid = false;
}

JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_addLayer(JNIEnv*, jobject) {
    enqueuePendingAction(kActionAddLayer);
}

JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_cycleActiveLayer(JNIEnv*, jobject) {
    enqueuePendingAction(kActionCycleActive);
}

JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_clearActiveLayer(JNIEnv*, jobject) {
    enqueuePendingAction(kActionClearActive);
}

JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_addVectorLayer(JNIEnv*, jobject) {
    enqueuePendingAction(kActionAddVectorLayer);
}

// Delete the layer at `idx` on the active page. Queued through the
// pending-action drain so the GL-resource frees and disk I/O run on the
// GL thread. The drain refuses if only one layer remains.
JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_deleteLayer(JNIEnv*, jobject, jint idx) {
    if (idx < 0) return;
    g_pendingDeleteLayerIdx.store(idx);
    enqueuePendingAction(kActionDeleteLayer);
}

// Move the layer at `fromIdx` to position `toIdx` on the active page.
// Same queueing rationale as deleteLayer. Last-write-wins on the side
// channel — caller should serialize moves on its end if it cares.
JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_moveLayer(JNIEnv*, jobject,
                                             jint fromIdx, jint toIdx) {
    if (fromIdx < 0 || toIdx < 0) return;
    {
        std::lock_guard<std::mutex> lock(g_pendingMoveLayerMutex);
        g_pendingMoveLayerFrom = fromIdx;
        g_pendingMoveLayerTo   = toIdx;
    }
    enqueuePendingAction(kActionMoveLayer);
}

// Undo / redo are queued through the same pending-action path so the
// inverse mutation (which can touch GL state and disk) runs on the GL
// thread. Caller should trigger a redraw afterward; the change won't be
// visible until the next composite.
JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_undo(JNIEnv*, jobject) {
    enqueuePendingAction(kActionUndo);
}

JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_redo(JNIEnv*, jobject) {
    enqueuePendingAction(kActionRedo);
}

JNIEXPORT jboolean JNICALL
Java_com_bk_drawing_NativeRenderer_canUndo(JNIEnv*, jobject) {
    std::lock_guard<std::mutex> lock(g_undoMutex);
    return g_undoStack.empty() ? JNI_FALSE : JNI_TRUE;
}

JNIEXPORT jboolean JNICALL
Java_com_bk_drawing_NativeRenderer_canRedo(JNIEnv*, jobject) {
    std::lock_guard<std::mutex> lock(g_undoMutex);
    return g_redoStack.empty() ? JNI_FALSE : JNI_TRUE;
}

// Each addXxx packs the user's drag into a shape struct, picks up the
// current brush color and a default line width, and queues it for the
// GL thread to apply on the next render.

JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_addLine(
        JNIEnv*, jobject,
        jfloat x0, jfloat y0, jfloat x1, jfloat y1) {
    Line l;
    l.x0 = x0; l.y0 = y0;
    l.x1 = x1; l.y1 = y1;
    l.color = g_currentBrushColor.load();
    l.width = currentVectorLineWidth();
    std::lock_guard<std::mutex> lock(g_pendingShapesMutex);
    g_pendingLines.push_back(l);
}

JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_addRectangle(
        JNIEnv*, jobject,
        jfloat x0, jfloat y0, jfloat x1, jfloat y1) {
    Rect r;
    r.x0 = x0; r.y0 = y0;
    r.x1 = x1; r.y1 = y1;
    r.rotation = 0.0f;
    r.color = g_currentBrushColor.load();
    r.width = currentVectorLineWidth();
    std::lock_guard<std::mutex> lock(g_pendingShapesMutex);
    g_pendingRects.push_back(r);
}

// Ellipse from bbox: center at midpoint, semi-axes from half-extents.
JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_addEllipse(
        JNIEnv*, jobject,
        jfloat x0, jfloat y0, jfloat x1, jfloat y1) {
    Ellipse e;
    e.cx = (x0 + x1) * 0.5f;
    e.cy = (y0 + y1) * 0.5f;
    e.rx = std::fabs(x1 - x0) * 0.5f;
    e.ry = std::fabs(y1 - y0) * 0.5f;
    e.rotation = 0.0f;
    e.color = g_currentBrushColor.load();
    e.width = currentVectorLineWidth();
    std::lock_guard<std::mutex> lock(g_pendingShapesMutex);
    g_pendingEllipses.push_back(e);
}

// Circle from center+radius: p0 is the center, p1 is a point on the
// circle. Radius = distance(p0, p1). Different gesture from ellipse/rect
// because dragging a circle from its center reads more naturally.
JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_addCircle(
        JNIEnv*, jobject,
        jfloat x0, jfloat y0, jfloat x1, jfloat y1) {
    Circle c;
    c.cx = x0;
    c.cy = y0;
    float dx = x1 - x0;
    float dy = y1 - y0;
    c.radius = std::sqrt(dx * dx + dy * dy);
    c.color = g_currentBrushColor.load();
    c.width = currentVectorLineWidth();
    std::lock_guard<std::mutex> lock(g_pendingShapesMutex);
    g_pendingCircles.push_back(c);
}

// Hit-test the active vector layer at (x, y); on hit, set g_selection
// and return true. On miss, clear any prior selection and return false.
JNIEXPORT jboolean JNICALL
Java_com_bk_drawing_NativeRenderer_selectShapeAt(
        JNIEnv*, jobject, jfloat x, jfloat y) {
    // Selection chrome (handles, OBB) renders in compositeAllLayers, so
    // adding / clearing a selection changes MB content outside any
    // stroke bbox.
    g_mbCacheValid = false;
    return hitTestActiveVectorLayer(x, y) ? JNI_TRUE : JNI_FALSE;
}

// Test if (x, y) hits any of the current selection's handles. Returns:
//   -2 = rotate handle
//   0..3 = scale handle (corner index)
//   -1 = no handle hit
int hitTestSelectionHandle(float x, float y) {
    Selection sel;
    {
        std::lock_guard<std::mutex> lock(g_selectionMutex);
        sel = g_selection;
    }
    if (sel.kind == ShapeKind::None) return -1;
    Obb obb;
    if (!obbForSelection(sel, obb)) return -1;

    // Radius via the UI-thread scale mirror — see vpxToDocUi.
    float r2 = vpxToDocUi(kHandleHitRadiusViewPx)
             * vpxToDocUi(kHandleHitRadiusViewPx);

    // Rotate handle (skip for circles).
    if (sel.kind != ShapeKind::Circle) {
        float rhX, rhY;
        rotateHandlePosition(obb, rhX, rhY, /*uiThread=*/true);
        float dx = x - rhX, dy = y - rhY;
        if (dx*dx + dy*dy <= r2) {
            return -2;
        }
    }

    // Scale handles.
    int n = scaleHandleCount(sel.kind);
    for (int i = 0; i < n; ++i) {
        float hx, hy;
        scaleHandlePosition(obb, sel.kind, i, hx, hy);
        float dx = x - hx, dy = y - hy;
        if (dx*dx + dy*dy <= r2) {
            return i;
        }
    }
    return -1;
}

bool isPointInsideObb(const Obb& o, float x, float y) {
    float lx, ly;
    rotateWorldToLocal(o, x, y, lx, ly);
    return std::fabs(lx) <= o.hw && std::fabs(ly) <= o.hh;
}

// Hit-test the floating raster selection's handles. Returns -2 for the
// rotate handle, 0..3 for a corner, -1 for no hit / no active selection.
int hitTestRasterSelectionHandle(float x, float y) {
    Obb obb;
    {
        std::lock_guard<std::mutex> lock(g_rasterSelMutex);
        if (!g_rasterSel.active) return -1;
        obb = obbForRasterSelection(g_rasterSel);
    }
    // Radius via the UI-thread scale mirror — see vpxToDocUi.
    float r2 = vpxToDocUi(kHandleHitRadiusViewPx)
             * vpxToDocUi(kHandleHitRadiusViewPx);

    // Rotate handle.
    float rhX, rhY;
    rotateHandlePosition(obb, rhX, rhY, /*uiThread=*/true);
    {
        float dx = x - rhX, dy = y - rhY;
        if (dx*dx + dy*dy <= r2) return -2;
    }
    // 4 corner scale handles.
    for (int i = 0; i < 4; ++i) {
        float hx, hy;
        scaleHandlePosition(obb, ShapeKind::Rect, i, hx, hy);
        float dx = x - hx, dy = y - hy;
        if (dx*dx + dy*dy <= r2) return i;
    }
    return -1;
}

// Drag state for the floating raster selection. Independent of g_drag
// (which is used by the vector selection tool); the two are dispatched
// via separate JNIs so they never clash.
DragState g_rasterDrag;

// Snapshot the OBB's rotation and the drag-corner's anchor at the start
// of a scale drag, so subsequent moves recompute against the initial
// state instead of accumulating float drift.
void applyRasterScaleTo(float x, float y) {
    DragState d;
    {
        std::lock_guard<std::mutex> lock(g_rasterSelMutex);
        d = g_rasterDrag;
    }
    if (d.mode != DragMode::Scale) return;

    // Work in handle space, not pen space: the corner sits wherever the
    // pen is minus where the pen was relative to the corner at grab
    // time. Applied before snap/velocity so everything downstream sees
    // the corner's actual position.
    x -= d.grabOffsetX;
    y -= d.grabOffsetY;

    float vView = 0.0f;
    bool fast = false;
    if (d.hasPrevQuery) {
        float qdx = x - d.prevQueryX;
        float qdy = y - d.prevQueryY;
        vView = std::sqrt(qdx * qdx + qdy * qdy) * userViewScale();
        fast = vView > kSnapVelocityViewPx;
    }
    SnapHit prev = { d.snapX, d.snapY, d.snapActive };
    float releaseR = fast ? kSnapReleaseFastViewPx : kSnapReleaseSlowViewPx;
    SnapHit snap = findSnap(x, y, /*exclude*/ nullptr, &prev, releaseR);
    if (fast && snap.found && !prev.found) snap.found = false;
    float origQueryX = x, origQueryY = y;
    if (snap.found) { x = snap.x; y = snap.y; }

    // New center = midpoint of fixed anchor and dragged pen.
    float newCx = (d.anchorX + x) * 0.5f;
    float newCy = (d.anchorY + y) * 0.5f;
    // Diagonal vector rotated into the OBB-local frame to recover
    // independent half-extents along the OBB's axes.
    float c = std::cos(-d.initialRotation), s = std::sin(-d.initialRotation);
    float wdx = x - d.anchorX;
    float wdy = y - d.anchorY;
    float ldx = wdx * c - wdy * s;
    float ldy = wdx * s + wdy * c;
    float newHw = std::fabs(ldx) * 0.5f;
    float newHh = std::fabs(ldy) * 0.5f;
    if (newHw < 0.5f) newHw = 0.5f;
    if (newHh < 0.5f) newHh = 0.5f;

    // Fixed-aspect override: pick whichever axis the user has scaled
    // furthest from its initial extent and apply that factor to both
    // axes. The center is recomputed so the anchor (opposite corner)
    // stays put, otherwise the OBB would slide as the constraint
    // adjusts the dimensions.
    bool aspectLocked = false;
    {
        std::lock_guard<std::mutex> lock(g_rasterSelMutex);
        aspectLocked = g_rasterSel.active && g_rasterSel.fixedAspect;
    }
    if (aspectLocked && d.initialHalfW > 0.5f && d.initialHalfH > 0.5f) {
        float fw = newHw / d.initialHalfW;
        float fh = newHh / d.initialHalfH;
        float f  = std::max(fw, fh);
        if (f < 0.05f) f = 0.05f;
        newHw = d.initialHalfW * f;
        newHh = d.initialHalfH * f;
        // anchor → center is (sign(ldx)*newHw, sign(ldy)*newHh) in local
        // frame; rotate by the OBB's rotation and add to anchor.
        float lcx = (ldx >= 0 ? 1.0f : -1.0f) * newHw;
        float lcy = (ldy >= 0 ? 1.0f : -1.0f) * newHh;
        float cw = std::cos(d.initialRotation);
        float sw = std::sin(d.initialRotation);
        newCx = d.anchorX + (lcx * cw - lcy * sw);
        newCy = d.anchorY + (lcx * sw + lcy * cw);
    }

    std::lock_guard<std::mutex> lock(g_rasterSelMutex);
    if (!g_rasterSel.active) return;
    g_rasterSel.centerX  = newCx;
    g_rasterSel.centerY  = newCy;
    g_rasterSel.halfW    = newHw;
    g_rasterSel.halfH    = newHh;
    g_rasterSel.rotation = d.initialRotation;
    g_rasterDrag.snapActive = snap.found;
    g_rasterDrag.snapX = snap.x;
    g_rasterDrag.snapY = snap.y;
    g_rasterDrag.prevQueryX = origQueryX;
    g_rasterDrag.prevQueryY = origQueryY;
    g_rasterDrag.hasPrevQuery = true;
}

void applyRasterRotateTo(float x, float y) {
    DragState d;
    {
        std::lock_guard<std::mutex> lock(g_rasterSelMutex);
        d = g_rasterDrag;
    }
    if (d.mode != DragMode::Rotate) return;

    float curAngle = std::atan2(y - d.centerY, x - d.centerX);
    float delta = curAngle - d.initialPenAngle;
    float newRotation = d.initialRotation + delta;

    if (g_snapEnabled.load() != 0) {
        constexpr float kStep = 3.14159265358979323846f / 12.0f;  // 15 deg
        constexpr float kTol  = 5.0f * 3.14159265358979323846f / 180.0f;
        float k = std::round(newRotation / kStep);
        float snapped = k * kStep;
        if (std::fabs(newRotation - snapped) <= kTol) {
            newRotation = snapped;
        }
    }

    std::lock_guard<std::mutex> lock(g_rasterSelMutex);
    if (!g_rasterSel.active) return;
    g_rasterSel.rotation = newRotation;
    g_rasterDrag.snapActive = false;
}

void applyRasterMoveTo(float x, float y) {
    DragState d;
    {
        std::lock_guard<std::mutex> lock(g_rasterSelMutex);
        d = g_rasterDrag;
    }
    if (d.mode != DragMode::Move) return;

    float targetCx = x - d.moveOffsetX;
    float targetCy = y - d.moveOffsetY;

    float vView = 0.0f;
    bool fast = false;
    if (d.hasPrevQuery) {
        float qdx = targetCx - d.prevQueryX;
        float qdy = targetCy - d.prevQueryY;
        vView = std::sqrt(qdx * qdx + qdy * qdy) * userViewScale();
        fast = vView > kSnapVelocityViewPx;
    }

    SnapHit prev = { d.snapX, d.snapY, d.snapActive };
    float releaseR = fast ? kSnapReleaseFastViewPx : kSnapReleaseSlowViewPx;
    SnapHit snap = findSnap(targetCx, targetCy, /*exclude*/ nullptr, &prev, releaseR);
    if (fast && snap.found && !prev.found) snap.found = false;
    float origQueryX = targetCx, origQueryY = targetCy;
    if (snap.found) { targetCx = snap.x; targetCy = snap.y; }

    std::lock_guard<std::mutex> lock(g_rasterSelMutex);
    if (!g_rasterSel.active) return;
    g_rasterSel.centerX = targetCx;
    g_rasterSel.centerY = targetCy;
    g_rasterDrag.snapActive = snap.found;
    g_rasterDrag.snapX = snap.x;
    g_rasterDrag.snapY = snap.y;
    g_rasterDrag.prevQueryX = origQueryX;
    g_rasterDrag.prevQueryY = origQueryY;
    g_rasterDrag.hasPrevQuery = true;
}

// Begin an interaction at (x, y). Tries handles first, then shape body
// (re-hit-test if no current selection or tap is outside selected OBB).
// Returns drag mode: 0=none, 1=move, 2=scale, 3=rotate.
JNIEXPORT jint JNICALL
Java_com_bk_drawing_NativeRenderer_beginInteractionAt(
        JNIEnv*, jobject, jfloat x, jfloat y) {
    // Reset stale snap state from a prior drag so the hysteresis in
    // findSnap doesn't bias the first frame of this new interaction.
    {
        std::lock_guard<std::mutex> lock(g_selectionMutex);
        g_drag.snapActive = false;
        g_drag.hasPrevQuery = false;
    }
    int handleHit = hitTestSelectionHandle(x, y);

    if (handleHit == -2) {
        // Rotate handle. Capture initial state.
        Selection sel;
        Obb obb;
        {
            std::lock_guard<std::mutex> lock(g_selectionMutex);
            sel = g_selection;
        }
        if (!obbForSelection(sel, obb)) return 0;
        std::lock_guard<std::mutex> lock(g_selectionMutex);
        g_drag.mode = DragMode::Rotate;
        g_drag.centerX = obb.cx;
        g_drag.centerY = obb.cy;
        g_drag.initialPenAngle = std::atan2(y - obb.cy, x - obb.cx);
        g_drag.initialRotation = obb.rotation;
        // Lines don't have a rotation field; snapshot initial endpoints
        // so we can recompute from initial + delta on each update.
        if (sel.kind == ShapeKind::Line
            && sel.layerIdx < layers().size()
            && layers()[sel.layerIdx]
            && sel.shapeIdx < layers()[sel.layerIdx]->lines.size()) {
            g_drag.initialLine = layers()[sel.layerIdx]->lines[sel.shapeIdx];
        }
        g_transformBeforeSel = sel;
        snapshotSelectionShape(sel, g_transformBeforeShape);
        return 3;
    }

    if (handleHit >= 0) {
        // Scale handle. Anchor = opposite handle's world position.
        Selection sel;
        Obb obb;
        {
            std::lock_guard<std::mutex> lock(g_selectionMutex);
            sel = g_selection;
        }
        if (!obbForSelection(sel, obb)) return 0;
        int anchorIdx;
        if (sel.kind == ShapeKind::Line) {
            anchorIdx = (handleHit == 0) ? 1 : 0;
        } else {
            anchorIdx = (handleHit + 2) % 4;   // diagonally opposite
        }
        float ax, ay;
        scaleHandlePosition(obb, sel.kind, anchorIdx, ax, ay);
        // Grab offset: how far the pen landed from the handle it hit.
        float gx, gy;
        scaleHandlePosition(obb, sel.kind, handleHit, gx, gy);
        std::lock_guard<std::mutex> lock(g_selectionMutex);
        g_drag.mode = DragMode::Scale;
        g_drag.handleIdx = handleHit;
        g_drag.anchorX = ax;
        g_drag.anchorY = ay;
        g_drag.grabOffsetX = x - gx;
        g_drag.grabOffsetY = y - gy;
        g_drag.initialRotation = obb.rotation;
        g_transformBeforeSel = sel;
        snapshotSelectionShape(sel, g_transformBeforeShape);
        return 2;
    }

    // No handle hit. Decide between "move existing selection if tap is
    // inside ANY selected shape's OBB" vs "re-hit-test for a new shape
    // selection". For multi-select, tapping any selected shape grabs
    // the whole group; the move uses the primary's center as the
    // offset reference (extras follow by the same delta).
    {
        Selection sel;
        std::vector<Selection> extras;
        {
            std::lock_guard<std::mutex> lock(g_selectionMutex);
            sel    = g_selection;
            extras = g_extraSelections;
        }
        Obb primaryObb;
        bool tapInsidePrimary = (sel.kind != ShapeKind::None)
            && obbForSelection(sel, primaryObb)
            && isPointInsideObb(primaryObb, x, y);
        bool tapInsideAnyExtra = false;
        if (!tapInsidePrimary) {
            for (const auto& e : extras) {
                Obb eobb;
                if (obbForSelection(e, eobb) && isPointInsideObb(eobb, x, y)) {
                    tapInsideAnyExtra = true;
                    break;
                }
            }
        }
        if (tapInsidePrimary || tapInsideAnyExtra) {
            // Capture the primary's pre-move state for VectorMutate
            // undo, plus each extra's so a multi-select drag is fully
            // reversible.
            std::vector<ShapeData> extraBefore;
            extraBefore.reserve(extras.size());
            for (const auto& e : extras) {
                ShapeData sd;
                snapshotSelectionShape(e, sd);
                extraBefore.push_back(sd);
            }
            std::lock_guard<std::mutex> lock(g_selectionMutex);
            g_drag.mode = DragMode::Move;
            g_drag.moveOffsetX = x - primaryObb.cx;
            g_drag.moveOffsetY = y - primaryObb.cy;
            g_transformBeforeSel = sel;
            snapshotSelectionShape(sel, g_transformBeforeShape);
            g_transformBeforeExtraSels   = extras;
            g_transformBeforeExtraShapes = std::move(extraBefore);
            return 1;
        }
    }
    if (hitTestActiveVectorLayer(x, y)) {
        // First tap on a not-yet-selected shape: select it but DON'T
        // start a move drag. The user has to tap-and-drag inside the
        // (now selected) OBB on a second gesture to translate. This
        // prevents accidental nudges from a stylus that slides
        // slightly during the initial selection tap. The earlier
        // "tap inside existing selection's OBB" path above handles
        // the move on the second gesture.
        std::lock_guard<std::mutex> lock(g_selectionMutex);
        g_drag.mode = DragMode::None;
        return 0;
    }
    // Empty tap on the canvas. hitTestActiveVectorLayer already cleared
    // the prior selection; start a marquee at this point so a
    // subsequent drag picks up multiple shapes. End-up on the same
    // point (no drag) just lands as "deselect" — endInteraction sees
    // the marquee rect with zero area and produces no selection.
    std::lock_guard<std::mutex> lock(g_selectionMutex);
    g_drag.mode = DragMode::Marquee;
    g_marqueeActive = true;
    g_marqueeX0 = x; g_marqueeY0 = y;
    g_marqueeX1 = x; g_marqueeY1 = y;
    return 4;
}

// Translate the given selection's shape by (dx, dy). Caller must hold
// any necessary locks and have validated layer/index. Used by both
// translateSelection (incremental) and applyMoveTo (snap-aware).
void translateShape(Layer& layer, const Selection& sel, float dx, float dy) {
    switch (sel.kind) {
        case ShapeKind::Line:
            if (sel.shapeIdx < layer.lines.size()) {
                auto& l = layer.lines[sel.shapeIdx];
                l.x0 += dx; l.y0 += dy;
                l.x1 += dx; l.y1 += dy;
            }
            break;
        case ShapeKind::Rect:
            if (sel.shapeIdx < layer.rects.size()) {
                auto& r = layer.rects[sel.shapeIdx];
                r.x0 += dx; r.y0 += dy;
                r.x1 += dx; r.y1 += dy;
            }
            break;
        case ShapeKind::Ellipse:
            if (sel.shapeIdx < layer.ellipses.size()) {
                auto& e = layer.ellipses[sel.shapeIdx];
                e.cx += dx; e.cy += dy;
            }
            break;
        case ShapeKind::Circle:
            if (sel.shapeIdx < layer.circles.size()) {
                auto& c = layer.circles[sel.shapeIdx];
                c.cx += dx; c.cy += dy;
            }
            break;
        case ShapeKind::None:
            break;
    }
}

// Apply a move interaction: snap-aware absolute drag. Computes the
// would-be center (penX − captured offset), snaps that against other
// shapes' targets (excluding self), then translates so the shape's
// center matches the snapped position.
void applyMoveTo(float x, float y) {
    Selection sel;
    DragState d;
    {
        std::lock_guard<std::mutex> lock(g_selectionMutex);
        sel = g_selection;
        d   = g_drag;
    }
    if (d.mode != DragMode::Move || sel.kind == ShapeKind::None) return;
    if (sel.layerIdx >= layers().size() || !layers()[sel.layerIdx]) return;
    Layer& layer = *layers()[sel.layerIdx];
    if (layer.type != LayerType::Vector) return;

    Obb obb;
    if (!obbForSelection(sel, obb)) return;

    float targetCx = x - d.moveOffsetX;
    float targetCy = y - d.moveOffsetY;

    // Velocity gate: don't engage a fresh snap while the pen is
    // sweeping through the canvas. Existing locks still hold via the
    // hysteresis in findSnap.
    float vView = 0.0f;
    bool fast = false;
    if (d.hasPrevQuery) {
        float qdx = targetCx - d.prevQueryX;
        float qdy = targetCy - d.prevQueryY;
        vView = std::sqrt(qdx * qdx + qdy * qdy) * userViewScale();
        fast = vView > kSnapVelocityViewPx;
    }

    SnapHit prev = { d.snapX, d.snapY, d.snapActive };
    float releaseR = fast ? kSnapReleaseFastViewPx : kSnapReleaseSlowViewPx;
    SnapHit snap = findSnap(targetCx, targetCy, &sel, &prev, releaseR);
    if (fast && snap.found && !prev.found) {
        // Fresh engagement while moving fast — suppress.
        snap.found = false;
    }
    float origQueryX = targetCx, origQueryY = targetCy;
    if (snap.found) {
        targetCx = snap.x;
        targetCy = snap.y;
    }

    float dx = targetCx - obb.cx;
    float dy = targetCy - obb.cy;
    translateShape(layer, sel, dx, dy);

    // Multi-select: every other selected shape moves by the same
    // delta. Snapshot extras under the lock.
    std::vector<Selection> extras;
    {
        std::lock_guard<std::mutex> lock(g_selectionMutex);
        extras = g_extraSelections;
    }
    for (const auto& e : extras) {
        if (e.layerIdx >= layers().size() || !layers()[e.layerIdx]) continue;
        translateShape(*layers()[e.layerIdx], e, dx, dy);
    }

    {
        std::lock_guard<std::mutex> lock(g_selectionMutex);
        g_drag.snapActive = snap.found;
        g_drag.snapX = snap.x;
        g_drag.snapY = snap.y;
        // Track the unsnapped query position so the next call can
        // measure pen velocity (not snap-distorted velocity).
        g_drag.prevQueryX = origQueryX;
        g_drag.prevQueryY = origQueryY;
        g_drag.hasPrevQuery = true;
    }
}

// Apply a scale interaction: drag of handle to (x, y). Anchor (opposite
// corner) stays fixed in world space. Computes new local extents from
// the (anchor → pen) vector rotated into shape-local frame.
void applyScaleTo(float x, float y) {
    Selection sel;
    DragState d;
    {
        std::lock_guard<std::mutex> lock(g_selectionMutex);
        sel = g_selection;
        d   = g_drag;
    }
    if (d.mode != DragMode::Scale || sel.kind == ShapeKind::None) return;
    if (sel.layerIdx >= layers().size() || !layers()[sel.layerIdx]) return;
    Layer& layer = *layers()[sel.layerIdx];

    // Work in handle space, not pen space — see applyRasterScaleTo.
    x -= d.grabOffsetX;
    y -= d.grabOffsetY;

    // Snap the dragged handle's pen position against other shapes'
    // targets (excluding self) before recomputing extents.
    float vView = 0.0f;
    bool fast = false;
    if (d.hasPrevQuery) {
        float qdx = x - d.prevQueryX;
        float qdy = y - d.prevQueryY;
        vView = std::sqrt(qdx * qdx + qdy * qdy) * userViewScale();
        fast = vView > kSnapVelocityViewPx;
    }
    SnapHit prev = { d.snapX, d.snapY, d.snapActive };
    float releaseR = fast ? kSnapReleaseFastViewPx : kSnapReleaseSlowViewPx;
    SnapHit snap = findSnap(x, y, &sel, &prev, releaseR);
    if (fast && snap.found && !prev.found) snap.found = false;
    float origQueryX = x, origQueryY = y;
    if (snap.found) {
        x = snap.x;
        y = snap.y;
    }
    // Line endpoint drag + angle snap → constrain the dragged endpoint
    // to lie on a 15° ray from the anchor (other endpoint), preserving
    // the cursor distance. Only applies when point-snap didn't already
    // lock the endpoint (a vertex is a deliberate target). Other shape
    // kinds use rotation-snap via applyRotateTo for orientation
    // constraints; their scale stays free.
    if (sel.kind == ShapeKind::Line
        && g_angleSnapEnabled.load() != 0
        && !snap.found) {
        float dx = x - d.anchorX;
        float dy = y - d.anchorY;
        float dist = std::sqrt(dx * dx + dy * dy);
        if (dist >= 1.0f) {
            constexpr float kStep = 3.14159265358979323846f / 12.0f; // 15°
            float ang        = std::atan2(dy, dx);
            float snappedAng = std::round(ang / kStep) * kStep;
            x = d.anchorX + std::cos(snappedAng) * dist;
            y = d.anchorY + std::sin(snappedAng) * dist;
        }
    }
    {
        std::lock_guard<std::mutex> lock(g_selectionMutex);
        g_drag.snapActive = snap.found;
        g_drag.snapX = snap.x;
        g_drag.snapY = snap.y;
        g_drag.prevQueryX = origQueryX;
        g_drag.prevQueryY = origQueryY;
        g_drag.hasPrevQuery = true;
    }

    // New center is midpoint of anchor and pen in world.
    float newCx = (d.anchorX + x) * 0.5f;
    float newCy = (d.anchorY + y) * 0.5f;
    // Diagonal vector in world, rotated into shape-local frame to recover
    // (full-)width and -height.
    float c = std::cos(-d.initialRotation), s = std::sin(-d.initialRotation);
    float wdx = x - d.anchorX;
    float wdy = y - d.anchorY;
    float ldx = wdx * c - wdy * s;
    float ldy = wdx * s + wdy * c;
    float newHw = std::fabs(ldx) * 0.5f;
    float newHh = std::fabs(ldy) * 0.5f;
    if (newHw < 0.5f) newHw = 0.5f;
    if (newHh < 0.5f) newHh = 0.5f;

    switch (sel.kind) {
        case ShapeKind::Rect:
            if (sel.shapeIdx < layer.rects.size()) {
                auto& r = layer.rects[sel.shapeIdx];
                r.x0 = newCx - newHw; r.y0 = newCy - newHh;
                r.x1 = newCx + newHw; r.y1 = newCy + newHh;
                // rotation unchanged
            }
            break;
        case ShapeKind::Ellipse:
            if (sel.shapeIdx < layer.ellipses.size()) {
                auto& e = layer.ellipses[sel.shapeIdx];
                e.cx = newCx; e.cy = newCy;
                e.rx = newHw; e.ry = newHh;
            }
            break;
        case ShapeKind::Circle:
            if (sel.shapeIdx < layer.circles.size()) {
                auto& cir = layer.circles[sel.shapeIdx];
                // For a circle, drag any handle = scale uniformly. Use
                // distance from anchor to pen as the diameter.
                float dx = x - d.anchorX, dy = y - d.anchorY;
                float diam = std::sqrt(dx*dx + dy*dy);
                cir.cx = newCx; cir.cy = newCy;
                cir.radius = std::max(diam * 0.5f, 0.5f);
            }
            break;
        case ShapeKind::Line:
            if (sel.shapeIdx < layer.lines.size()) {
                auto& l = layer.lines[sel.shapeIdx];
                // For Line, the dragged handle is one endpoint, the
                // anchor is the other. Just move that endpoint.
                if (d.handleIdx == 0) {
                    l.x0 = x; l.y0 = y;
                } else {
                    l.x1 = x; l.y1 = y;
                }
            }
            break;
        case ShapeKind::None:
            break;
    }
}

// Apply a rotate interaction: drag of rotate handle to (x, y). Computes
// new rotation = initialRotation + (current pen angle − initial pen angle).
void applyRotateTo(float x, float y) {
    Selection sel;
    DragState d;
    {
        std::lock_guard<std::mutex> lock(g_selectionMutex);
        sel = g_selection;
        d   = g_drag;
    }
    if (d.mode != DragMode::Rotate || sel.kind == ShapeKind::None) return;
    if (sel.layerIdx >= layers().size() || !layers()[sel.layerIdx]) return;
    Layer& layer = *layers()[sel.layerIdx];

    float curAngle = std::atan2(y - d.centerY, x - d.centerX);
    float delta = curAngle - d.initialPenAngle;
    float newRotation = d.initialRotation + delta;

    // Angle-snap: when on, force rotation to the nearest 15° step.
    // This is the same lock used by the LINE tool's drawing path, so
    // toggling angle snap globally constrains both new lines and any
    // rotation drag of an existing shape. Geometry snap (g_snapEnabled,
    // a separate toggle) used to also fire a soft 5° tolerance snap
    // here; angle snap subsumes that — the user can flip it on for the
    // duration of the drag.
    if (g_angleSnapEnabled.load() != 0) {
        constexpr float kStep = 3.14159265358979323846f / 12.0f;  // 15°
        float k = std::round(newRotation / kStep);
        float snapped = k * kStep;
        // Re-derive delta so Line endpoints (which compute from
        // initialLine + delta) match the snapped rotation exactly.
        delta = snapped - d.initialRotation;
        newRotation = snapped;
    }
    // Rotate has no on-canvas marker target, so no snapActive flag here.
    {
        std::lock_guard<std::mutex> lock(g_selectionMutex);
        g_drag.snapActive = false;
    }

    switch (sel.kind) {
        case ShapeKind::Rect:
            if (sel.shapeIdx < layer.rects.size())
                layer.rects[sel.shapeIdx].rotation = newRotation;
            break;
        case ShapeKind::Ellipse:
            if (sel.shapeIdx < layer.ellipses.size())
                layer.ellipses[sel.shapeIdx].rotation = newRotation;
            break;
        case ShapeKind::Line:
            if (sel.shapeIdx < layer.lines.size()) {
                // Rotate p0, p1 around the line's midpoint by `delta`
                // (relative to initial endpoints, so we don't accumulate
                // float drift across many move events).
                auto& l = layer.lines[sel.shapeIdx];
                float cx = (d.initialLine.x0 + d.initialLine.x1) * 0.5f;
                float cy = (d.initialLine.y0 + d.initialLine.y1) * 0.5f;
                float c = std::cos(delta), s = std::sin(delta);
                auto rot = [&](float px, float py, float& ox, float& oy) {
                    float dx = px - cx, dy = py - cy;
                    ox = cx + dx * c - dy * s;
                    oy = cy + dx * s + dy * c;
                };
                rot(d.initialLine.x0, d.initialLine.y0, l.x0, l.y0);
                rot(d.initialLine.x1, d.initialLine.y1, l.x1, l.y1);
            }
            break;
        case ShapeKind::Circle:
        case ShapeKind::None:
            break;
    }
}

JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_updateInteractionAt(
        JNIEnv*, jobject, jfloat x, jfloat y) {
    DragMode mode;
    {
        std::lock_guard<std::mutex> lock(g_selectionMutex);
        mode = g_drag.mode;
    }
    if (mode == DragMode::Scale) {
        applyScaleTo(x, y);
    } else if (mode == DragMode::Rotate) {
        applyRotateTo(x, y);
    } else if (mode == DragMode::Move) {
        applyMoveTo(x, y);
    } else if (mode == DragMode::Marquee) {
        std::lock_guard<std::mutex> lock(g_selectionMutex);
        g_marqueeX1 = x;
        g_marqueeY1 = y;
    }
    // Any of these drag modes moves selection chrome / shape geometry.
    g_mbCacheValid = false;
}

JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_endInteraction(JNIEnv*, jobject) {
    DragMode  wasMode;
    Selection beforeSel;
    ShapeData beforeShape;
    std::vector<Selection> beforeExtraSels;
    std::vector<ShapeData> beforeExtraShapes;
    float mx0 = 0, my0 = 0, mx1 = 0, my1 = 0;
    {
        std::lock_guard<std::mutex> lock(g_selectionMutex);
        wasMode     = g_drag.mode;
        beforeSel   = g_transformBeforeSel;
        beforeShape = g_transformBeforeShape;
        beforeExtraSels   = std::move(g_transformBeforeExtraSels);
        beforeExtraShapes = std::move(g_transformBeforeExtraShapes);
        mx0 = std::min(g_marqueeX0, g_marqueeX1);
        my0 = std::min(g_marqueeY0, g_marqueeY1);
        mx1 = std::max(g_marqueeX0, g_marqueeX1);
        my1 = std::max(g_marqueeY0, g_marqueeY1);
        g_drag.mode = DragMode::None;
        g_drag.snapActive = false;
        g_marqueeActive = false;
        g_transformBeforeSel   = Selection{};
        g_transformBeforeShape = ShapeData{};
        g_transformBeforeExtraSels.clear();
        g_transformBeforeExtraShapes.clear();
    }
    if (wasMode == DragMode::Marquee) {
        // Finalize marquee selection: every shape on the active vector
        // layer whose actual outline overlaps the rectangle joins the
        // selection. AABB-only tests over-select badly for diagonal
        // lines (their AABB covers the whole bounding square) and
        // rotated rects/ellipses (AABB > OBB). See
        // shapeOutlineIntersectsMarquee for the per-kind geometry.
        // Tap-with-no-drag (zero-area rect) leaves the selection empty
        // — hitTestActiveVectorLayer in the begin path already cleared
        // the prior selection.
        if (mx1 - mx0 < 1.0f && my1 - my0 < 1.0f) return;
        if (activeLayer() >= layers().size() || !layers()[activeLayer()]) return;
        Layer& layer = *layers()[activeLayer()];
        if (layer.type != LayerType::Vector) return;

        std::vector<Selection> hits;
        auto add = [&](ShapeKind kind, size_t idx) {
            Selection s;
            s.kind     = kind;
            s.layerIdx = activeLayer();
            s.shapeIdx = idx;
            hits.push_back(s);
        };
        for (size_t i = 0; i < layer.lines.size(); ++i) {
            ShapeData sd; sd.kind = ShapeKind::Line; sd.line = layer.lines[i];
            if (shapeOutlineIntersectsMarquee(sd, mx0, my0, mx1, my1))
                add(ShapeKind::Line, i);
        }
        for (size_t i = 0; i < layer.rects.size(); ++i) {
            ShapeData sd; sd.kind = ShapeKind::Rect; sd.rect = layer.rects[i];
            if (shapeOutlineIntersectsMarquee(sd, mx0, my0, mx1, my1))
                add(ShapeKind::Rect, i);
        }
        for (size_t i = 0; i < layer.ellipses.size(); ++i) {
            ShapeData sd; sd.kind = ShapeKind::Ellipse; sd.ellipse = layer.ellipses[i];
            if (shapeOutlineIntersectsMarquee(sd, mx0, my0, mx1, my1))
                add(ShapeKind::Ellipse, i);
        }
        for (size_t i = 0; i < layer.circles.size(); ++i) {
            ShapeData sd; sd.kind = ShapeKind::Circle; sd.circle = layer.circles[i];
            if (shapeOutlineIntersectsMarquee(sd, mx0, my0, mx1, my1))
                add(ShapeKind::Circle, i);
        }
        std::lock_guard<std::mutex> lock(g_selectionMutex);
        if (hits.empty()) {
            g_selection = Selection{};
            g_extraSelections.clear();
        } else {
            g_selection = hits.front();
            g_extraSelections.assign(hits.begin() + 1, hits.end());
        }
        return;
    }
    if (wasMode == DragMode::None || beforeSel.kind == ShapeKind::None) return;
    if (beforeShape.kind == ShapeKind::None) return;

    // Single-shape drag (no extras captured) → VectorMutate, same as
    // before. Multi-shape drag → bundle every changed shape into a
    // single VectorMutateGroup so one undo step reverses the whole
    // group transform.
    if (beforeExtraSels.empty()) {
        ShapeData afterShape;
        if (snapshotSelectionShape(beforeSel, afterShape)
            && !shapeDataEqual(beforeShape, afterShape)) {
            UndoEntry entry;
            entry.op          = UndoOp::VectorMutate;
            entry.layerIdx    = beforeSel.layerIdx;
            entry.shapeIdx    = beforeSel.shapeIdx;
            entry.beforeShape = beforeShape;
            entry.afterShape  = afterShape;
            pushUndoEntry(std::move(entry));
        }
        return;
    }

    UndoEntry entry;
    entry.op       = UndoOp::VectorMutateGroup;
    entry.layerIdx = beforeSel.layerIdx;   // representative; not load-bearing

    auto addToGroup = [&](const Selection& s, const ShapeData& before) {
        if (s.kind == ShapeKind::None) return;
        ShapeData after;
        if (!snapshotSelectionShape(s, after)) return;
        if (shapeDataEqual(before, after)) return;
        entry.mutateGroupSels.push_back(s);
        entry.mutateGroupBefore.push_back(before);
        entry.mutateGroupAfter.push_back(after);
    };
    addToGroup(beforeSel, beforeShape);
    for (size_t i = 0; i < beforeExtraSels.size()
                    && i < beforeExtraShapes.size(); ++i) {
        addToGroup(beforeExtraSels[i], beforeExtraShapes[i]);
    }
    if (!entry.mutateGroupSels.empty()) {
        pushUndoEntry(std::move(entry));
    }
}

// Raster floating selection: handle hit-test → mode dispatch. Returns:
//   0 = no hit (caller commits + starts a new define gesture)
//   1 = move      (drag inside body)
//   2 = scale     (corner handle)
//   3 = rotate    (rotate handle)
// All transform state is recorded into g_rasterDrag for the subsequent
// updateRasterInteractionAt calls. No undo entry is recorded here or
// in endRasterInteraction — the eventual commitRasterSelection bakes
// one combined RasterStroke entry covering lift + final placement.
JNIEXPORT jint JNICALL
Java_com_bk_drawing_NativeRenderer_beginRasterInteractionAt(
        JNIEnv*, jobject, jfloat x, jfloat y) {
    // Reset stale snap state from a prior drag — see beginInteractionAt.
    {
        std::lock_guard<std::mutex> lock(g_rasterSelMutex);
        g_rasterDrag.snapActive = false;
        g_rasterDrag.hasPrevQuery = false;
    }
    int hit = hitTestRasterSelectionHandle(x, y);
    if (hit == -2) {
        std::lock_guard<std::mutex> lock(g_rasterSelMutex);
        if (!g_rasterSel.active) return 0;
        Obb obb = obbForRasterSelection(g_rasterSel);
        g_rasterDrag.mode = DragMode::Rotate;
        g_rasterDrag.centerX = obb.cx;
        g_rasterDrag.centerY = obb.cy;
        g_rasterDrag.initialPenAngle = std::atan2(y - obb.cy, x - obb.cx);
        g_rasterDrag.initialRotation = obb.rotation;
        return 3;
    }
    if (hit >= 0) {
        std::lock_guard<std::mutex> lock(g_rasterSelMutex);
        if (!g_rasterSel.active) return 0;
        Obb obb = obbForRasterSelection(g_rasterSel);
        // Anchor = diagonally opposite corner.
        int anchorIdx = (hit + 2) % 4;
        float ax, ay;
        scaleHandlePosition(obb, ShapeKind::Rect, anchorIdx, ax, ay);
        // Grab offset: how far the pen landed from the handle it hit.
        float gx, gy;
        scaleHandlePosition(obb, ShapeKind::Rect, hit, gx, gy);
        g_rasterDrag.mode = DragMode::Scale;
        g_rasterDrag.handleIdx = hit;
        g_rasterDrag.anchorX = ax;
        g_rasterDrag.anchorY = ay;
        g_rasterDrag.grabOffsetX = x - gx;
        g_rasterDrag.grabOffsetY = y - gy;
        g_rasterDrag.initialRotation = obb.rotation;
        // Snapshot the initial half-extents so applyRasterScaleTo can
        // enforce a uniform scale when fixedAspect is set on the
        // selection (imported images).
        g_rasterDrag.initialHalfW = g_rasterSel.halfW;
        g_rasterDrag.initialHalfH = g_rasterSel.halfH;
        return 2;
    }
    // Body hit → move.
    {
        std::lock_guard<std::mutex> lock(g_rasterSelMutex);
        if (!g_rasterSel.active) return 0;
        Obb obb = obbForRasterSelection(g_rasterSel);
        if (!isPointInsideObb(obb, x, y)) {
            g_rasterDrag.mode = DragMode::None;
            return 0;
        }
        g_rasterDrag.mode = DragMode::Move;
        g_rasterDrag.moveOffsetX = x - obb.cx;
        g_rasterDrag.moveOffsetY = y - obb.cy;
        return 1;
    }
}

JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_updateRasterInteractionAt(
        JNIEnv*, jobject, jfloat x, jfloat y) {
    DragMode mode;
    {
        std::lock_guard<std::mutex> lock(g_rasterSelMutex);
        mode = g_rasterDrag.mode;
    }
    if (mode == DragMode::Scale)       applyRasterScaleTo(x, y);
    else if (mode == DragMode::Rotate) applyRasterRotateTo(x, y);
    else if (mode == DragMode::Move)   applyRasterMoveTo(x, y);
    // The floating raster selection's screen footprint changed.
    g_mbCacheValid = false;
}

JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_endRasterInteraction(JNIEnv*, jobject) {
    std::lock_guard<std::mutex> lock(g_rasterSelMutex);
    g_rasterDrag.mode = DragMode::None;
    g_rasterDrag.snapActive = false;
}

JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_clearSelection(JNIEnv*, jobject) {
    {
        std::lock_guard<std::mutex> lock(g_selectionMutex);
        g_selection = Selection{};
        g_extraSelections.clear();
    }
    // Selection chrome went away — cached frame still has it drawn.
    g_mbCacheValid = false;
}

JNIEXPORT jboolean JNICALL
Java_com_bk_drawing_NativeRenderer_hasSelection(JNIEnv*, jobject) {
    std::lock_guard<std::mutex> lock(g_selectionMutex);
    return g_selection.kind != ShapeKind::None ? JNI_TRUE : JNI_FALSE;
}

// Translate the currently selected shape by (dx, dy) doc-pixels. No-op
// if there's no selection or the layer is gone.
JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_translateSelection(
        JNIEnv*, jobject, jfloat dx, jfloat dy) {
    Selection sel;
    {
        std::lock_guard<std::mutex> lock(g_selectionMutex);
        sel = g_selection;
    }
    if (sel.kind == ShapeKind::None) return;
    if (sel.layerIdx >= layers().size() || !layers()[sel.layerIdx]) return;
    Layer& layer = *layers()[sel.layerIdx];
    if (layer.type != LayerType::Vector) return;
    translateShape(layer, sel, dx, dy);
    g_mbCacheValid = false;
}

// Snap-aware absolute move: drives an in-progress Move drag with the
// pen's current position. Uses the captured moveOffset (snapshotted in
// beginInteractionAt) so the user's grab-point follows the pen, and
// snaps the would-be center against other shapes' targets / grid.
JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_moveSelectionTo(
        JNIEnv*, jobject, jfloat x, jfloat y) {
    applyMoveTo(x, y);
    g_mbCacheValid = false;
}

// Remove the currently selected content. For vector selections, erases
// each selected shape from its layer (one VectorDelete undo entry each).
// For a floating raster selection, discards the lifted pixels — the
// source layer keeps the hole punched out at lift time — and pushes a
// RasterStroke undo entry for it. Both paths clear the selection.
//
// Raster path uses GL (texture delete + glReadPixels on the hole-state
// tiles for the redo afterTiles snapshot), so callers must invoke this
// on the GL thread whenever a raster selection is active. The Kotlin
// layer routes through DrawingSurfaceView.queueDeleteSelection().
JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_deleteSelection(JNIEnv*, jobject) {
    ensureInited();
    bool rasterActive = false;
    {
        std::lock_guard<std::mutex> lock(g_rasterSelMutex);
        rasterActive = g_rasterSel.active;
    }
    if (rasterActive) {
        discardRasterSelectionImpl();
    } else {
        deleteAllSelectionsImpl();
    }
    g_mbCacheValid = false;
}

// Persist the active vector layer to disk. Used after a transform-drag
// completes so the move/scale/rotate is durable.
JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_persistActiveVectorLayer(
        JNIEnv*, jobject) {
    if (activeLayer() >= layers().size() || !layers()[activeLayer()]) return;
    if (layers()[activeLayer()]->type != LayerType::Vector) return;
    saveVectorLayer(activeLayer(), *layers()[activeLayer()]);
}

// Live preview during shape-tool drag. Color follows the current brush
// color; width follows the current vector-line-width slider. shapeType
// matches the addXxx-call shape semantics in renderShapePreviewToFront.
// If `snapped` is true, also draws a snap-marker overlay at (x1, y1).
JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_renderShapePreview(
        JNIEnv* env, jobject,
        jint width, jint height,
        jfloatArray transform,
        jint shapeType,
        jfloat x0, jfloat y0, jfloat x1, jfloat y1,
        jboolean snapped) {
    ensureInited();
    uint32_t rgb = g_currentBrushColor.load();
    renderShapePreviewToFront(env, width, height, transform,
                              shapeType, x0, y0, x1, y1,
                              rgb, currentVectorLineWidth(), /*alpha=*/0.7f,
                              snapped == JNI_TRUE);
}

// Live preview for the lasso (freeform) selection during drag. Renders
// the in-progress polyline into the bound (front) framebuffer.
JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_renderLassoPathPreview(
        JNIEnv* env, jobject,
        jint width, jint height,
        jfloatArray transform,
        jfloatArray jpoints,
        jboolean closed) {
    ensureInited();
    if (!jpoints) return;
    jsize len = env->GetArrayLength(jpoints);
    if (len < 2 || (len & 1) != 0) {
        // Still clear the front buffer so a stale preview doesn't linger.
        glViewport(0, 0, width, height);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        return;
    }
    std::vector<float> pts(static_cast<size_t>(len));
    env->GetFloatArrayRegion(jpoints, 0, len, pts.data());
    // Marquee/lasso preview is selection chrome, not user content —
    // render as a thin black hairline regardless of brush color or
    // vector line-width. Width is taken in view-pixels (then converted
    // back to doc-px) so the outline stays the same visual thickness
    // across pan/zoom.
    constexpr uint32_t kMarqueeColor = 0x000000u;
    constexpr float    kMarqueeWidthVpx = 1.5f;
    float widthDoc = kMarqueeWidthVpx / currentViewScale();
    renderLassoPathToFront(env, width, height, transform,
                           pts.data(), static_cast<size_t>(len) / 2,
                           kMarqueeColor, widthDoc, /*alpha=*/0.85f,
                           closed == JNI_TRUE);
}

// Find nearest snap target for (x, y); fills output[0..2] with
// [snapX, snapY, didSnap (1.0/0.0)]. If no target within range, output
// is (x, y, 0). Output array must have length >= 3.
JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_snapPoint(
        JNIEnv* env, jobject,
        jfloat x, jfloat y, jfloatArray output) {
    SnapHit h = findSnap(x, y);
    jfloat* arr = env->GetFloatArrayElements(output, nullptr);
    arr[0] = h.x;
    arr[1] = h.y;
    arr[2] = h.found ? 1.0f : 0.0f;
    env->ReleaseFloatArrayElements(output, arr, 0);
}

JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_setSnapEnabled(
        JNIEnv*, jobject, jboolean enabled) {
    g_snapEnabled.store(enabled == JNI_TRUE ? 1 : 0);
}

// Mirror of DrawingSurfaceView.angleSnapEnabled. When on, applyRotateTo
// locks rotation to 15° increments and applyScaleTo on a Line locks
// the dragged endpoint's direction from the anchor to 15° increments.
JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_setAngleSnapEnabled(
        JNIEnv*, jobject, jboolean enabled) {
    g_angleSnapEnabled.store(enabled == JNI_TRUE ? 1 : 0);
}

// Runtime toggle for motion prediction. The Kotlin side reads this
// (via isPredictionEnabled) to decide whether to dispatch predicted
// samples; native code only consults g_predictionInFlight, which is
// implicitly off if the Kotlin side never sends predicted batches.
// Exposing both directions makes A/B'ing trivial without rebuilding.
JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_setPredictionEnabled(
        JNIEnv*, jobject, jboolean enabled) {
    g_predictionEnabled.store(enabled == JNI_TRUE);
}

JNIEXPORT jboolean JNICALL
Java_com_bk_drawing_NativeRenderer_isPredictionEnabled(JNIEnv*, jobject) {
    return g_predictionEnabled.load() ? JNI_TRUE : JNI_FALSE;
}

// Set the page bounds (in doc-pixels). Drawn during composite as a thin
// outlined rectangle so the user can see where the page edges are when
// zoomed/rotated. Pass any zero-size rect to disable the outline.
JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_setPageBounds(
        JNIEnv*, jobject,
        jfloat x0, jfloat y0, jfloat x1, jfloat y1) {
    {
        std::lock_guard<std::mutex> lock(g_pageBoundsMutex);
        g_pageX0 = x0; g_pageY0 = y0;
        g_pageX1 = x1; g_pageY1 = y1;
    }
    // Page bg / outline / discard regions all repaint on resize.
    g_mbCacheValid = false;
}

// Update the cached view scale (doc-px per view-px). Read by snap/hit-
// test code and selection-overlay rendering to keep their effective
// radii / sizes constant in screen space across pan/zoom/rotate.
JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_setViewScale(
        JNIEnv*, jobject, jfloat scale) {
    if (!(scale > 1e-6f)) return;   // also rejects NaN
    uint32_t bits;
    std::memcpy(&bits, &scale, sizeof(bits));
    g_viewScaleBits.store(bits);
    g_userViewScaleBits.store(bits);
}

// Throw away an in-progress brush/eraser stroke without baking it into
// tiles. Called when a 2-finger gesture interrupts a stroke so the
// partial preview doesn't end up as an unwanted permanent mark.
JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_discardStroke(JNIEnv*, jobject) {
    g_current.samples.clear();
    g_liveEmitter.reset();
    g_shadePolyDoc.clear();
    g_shadeBboxValid = false;
}

// Read accessors for UI display. Called from the UI thread; the values may
// momentarily be stale relative to queued addLayer/cycleActiveLayer
// actions (which apply on the GL thread) but the discrepancy resolves on
// the next stroke or render and is fine for status text. They tolerate
// the not-yet-loaded state (g_pages empty) by returning 0.
JNIEXPORT jint JNICALL
Java_com_bk_drawing_NativeRenderer_getLayerCount(JNIEnv*, jobject) {
    if (g_pages.empty() || g_activePageIdx >= g_pages.size()) return 0;
    return static_cast<jint>(g_pages[g_activePageIdx]->layers.size());
}

// True once loadAllLayersFromDisk has fully populated all layer
// metadata. UI uses this to defer the first layer-panel render past
// the GL thread's lazy load — otherwise the panel paints with
// default-constructed Layer slots that aren't yet filled in.
JNIEXPORT jboolean JNICALL
Java_com_bk_drawing_NativeRenderer_isFullyLoaded(JNIEnv*, jobject) {
    return g_loaded.load(std::memory_order_acquire) ? JNI_TRUE : JNI_FALSE;
}

// Page-rect dimensions in doc-px — used by the exporter to allocate
// bitmaps at the canvas's natural resolution. Returns 0 if no page
// bounds are active (the doc behaves as an infinite plane in that case
// and the caller should fall back to surface dims).
JNIEXPORT jint JNICALL
Java_com_bk_drawing_NativeRenderer_getPageWidth(JNIEnv*, jobject) {
    PageClip page = readPageClip();
    if (!page.active) return 0;
    return static_cast<jint>(std::max(0.0f, page.maxX - page.minX));
}

JNIEXPORT jint JNICALL
Java_com_bk_drawing_NativeRenderer_getPageHeight(JNIEnv*, jobject) {
    PageClip page = readPageClip();
    if (!page.active) return 0;
    return static_cast<jint>(std::max(0.0f, page.maxY - page.minY));
}

JNIEXPORT jint JNICALL
Java_com_bk_drawing_NativeRenderer_getActiveLayer(JNIEnv*, jobject) {
    if (g_pages.empty() || g_activePageIdx >= g_pages.size()) return 0;
    return static_cast<jint>(g_pages[g_activePageIdx]->activeLayer);
}

// Layer type at the given index on the active page. 0 = raster, 1 =
// vector. Returns 0 if the index is out of range.
JNIEXPORT jint JNICALL
Java_com_bk_drawing_NativeRenderer_getLayerType(JNIEnv*, jobject, jint idx) {
    if (g_pages.empty() || g_activePageIdx >= g_pages.size()) return 0;
    auto& ls = g_pages[g_activePageIdx]->layers;
    if (idx < 0 || static_cast<size_t>(idx) >= ls.size() || !ls[idx]) return 0;
    return ls[idx]->type == LayerType::Vector ? 1 : 0;
}

// User-set display name for the layer. Empty string = no custom name
// (caller renders a default). Read-only; writes go through setLayerName.
JNIEXPORT jstring JNICALL
Java_com_bk_drawing_NativeRenderer_getLayerName(JNIEnv* env, jobject, jint idx) {
    std::string name;
    {
        std::lock_guard<std::mutex> lock(g_layerNameMutex);
        if (!g_pages.empty() && g_activePageIdx < g_pages.size()) {
            auto& ls = g_pages[g_activePageIdx]->layers;
            if (idx >= 0 && static_cast<size_t>(idx) < ls.size() && ls[idx]) {
                name = ls[idx]->name;
            }
        }
    }
    return env->NewStringUTF(name.c_str());
}

// Set the user-defined name for the layer at the given index on the
// active page. Persists immediately to <layerDir>/name.txt; absence of
// the file means "no custom name". Pass an empty string to clear.
JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_setLayerName(JNIEnv* env, jobject,
                                                jint idx, jstring jname) {
    if (g_pages.empty() || g_activePageIdx >= g_pages.size()) return;
    auto& ls = g_pages[g_activePageIdx]->layers;
    if (idx < 0 || static_cast<size_t>(idx) >= ls.size() || !ls[idx]) return;

    const char* utf = env->GetStringUTFChars(jname, nullptr);
    std::string name = utf ? utf : "";
    if (utf) env->ReleaseStringUTFChars(jname, utf);

    {
        std::lock_guard<std::mutex> lock(g_layerNameMutex);
        ls[idx]->name = name;
    }

    // Persist. Empty name removes the file so absence is canonical.
    std::string dir = activeLayerDir(static_cast<size_t>(idx));
    std::string path = dir + "/name.txt";
    if (name.empty()) {
        std::remove(path.c_str());
        return;
    }
    // A freshly-added raster layer (and the page containing it) may have
    // no on-disk presence yet — the layer dir is created lazily by the
    // first stroke bake. Without these mkdirs the fopen below silently
    // fails and the rename is lost on next launch.
    mkdir(pageDirOf(g_activePageIdx).c_str(), 0755);
    mkdir(dir.c_str(), 0755);
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        LOGE("setLayerName: fopen %s failed (errno=%d)", path.c_str(), errno);
        return;
    }
    std::fwrite(name.data(), 1, name.size(), f);
    std::fclose(f);
}

// Per-layer visibility. Returns true if visible (default), false if
// hidden. The compositor reads this every frame so the change takes
// effect on the next forceRedraw.
JNIEXPORT jboolean JNICALL
Java_com_bk_drawing_NativeRenderer_getLayerVisible(JNIEnv*, jobject, jint idx) {
    if (g_pages.empty() || g_activePageIdx >= g_pages.size()) return JNI_TRUE;
    auto& ls = g_pages[g_activePageIdx]->layers;
    if (idx < 0 || static_cast<size_t>(idx) >= ls.size() || !ls[idx]) return JNI_TRUE;
    return ls[idx]->visible.load(std::memory_order_relaxed) ? JNI_TRUE : JNI_FALSE;
}

// Toggle the visibility flag and persist it. Hidden state is encoded
// as the *presence* of <layerDir>/hidden.flag (an empty file). Visible
// removes the file. Caller should forceRedraw after.
JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_setLayerVisible(JNIEnv*, jobject,
                                                   jint idx, jboolean visible) {
    if (g_pages.empty() || g_activePageIdx >= g_pages.size()) return;
    auto& ls = g_pages[g_activePageIdx]->layers;
    if (idx < 0 || static_cast<size_t>(idx) >= ls.size() || !ls[idx]) return;
    bool vis = (visible == JNI_TRUE);
    ls[idx]->visible.store(vis, std::memory_order_relaxed);
    g_mbCacheValid = false;

    std::string dir = activeLayerDir(static_cast<size_t>(idx));
    std::string flagPath = dir + "/hidden.flag";
    if (vis) {
        std::remove(flagPath.c_str());
    } else {
        // Make sure both the page dir and the layer dir exist — a
        // freshly-added empty raster layer (and a page that hasn't been
        // saved to disk yet) has no on-disk presence until something
        // gets baked into it.
        mkdir(pageDirOf(g_activePageIdx).c_str(), 0755);
        mkdir(dir.c_str(), 0755);
        FILE* f = std::fopen(flagPath.c_str(), "wb");
        if (!f) {
            LOGE("setLayerVisible: fopen %s failed (errno=%d)",
                 flagPath.c_str(), errno);
            return;
        }
        std::fclose(f);
    }
}

// Per-layer opacity in [0, 1]. 1.0 is the default for layers without
// a saved override. Caller should forceRedraw after a setLayerOpacity.
JNIEXPORT jfloat JNICALL
Java_com_bk_drawing_NativeRenderer_getLayerOpacity(JNIEnv*, jobject, jint idx) {
    if (g_pages.empty() || g_activePageIdx >= g_pages.size()) return 1.0f;
    auto& ls = g_pages[g_activePageIdx]->layers;
    if (idx < 0 || static_cast<size_t>(idx) >= ls.size() || !ls[idx]) return 1.0f;
    return ls[idx]->opacity.load(std::memory_order_relaxed);
}

JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_setLayerOpacity(JNIEnv*, jobject,
                                                   jint idx, jfloat opacity) {
    if (g_pages.empty() || g_activePageIdx >= g_pages.size()) return;
    auto& ls = g_pages[g_activePageIdx]->layers;
    if (idx < 0 || static_cast<size_t>(idx) >= ls.size() || !ls[idx]) return;
    float clamped = opacity < 0.0f ? 0.0f : (opacity > 1.0f ? 1.0f : opacity);
    ls[idx]->opacity.store(clamped, std::memory_order_relaxed);
    g_mbCacheValid = false;

    std::string dir = activeLayerDir(static_cast<size_t>(idx));
    std::string path = dir + "/opacity.txt";
    if (clamped >= 0.999f) {
        // Default — drop the file so the layer dir stays clean.
        std::remove(path.c_str());
        return;
    }
    // Both the page dir and the layer dir may not exist yet (empty layer
    // on a never-saved page). Create both before the fopen.
    mkdir(pageDirOf(g_activePageIdx).c_str(), 0755);
    mkdir(dir.c_str(), 0755);
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        LOGE("setLayerOpacity: fopen %s failed (errno=%d)",
             path.c_str(), errno);
        return;
    }
    char buf[16];
    int n = std::snprintf(buf, sizeof(buf), "%.4f", clamped);
    if (n > 0) std::fwrite(buf, 1, static_cast<size_t>(n), f);
    std::fclose(f);
}

JNIEXPORT jint JNICALL
Java_com_bk_drawing_NativeRenderer_getPageCount(JNIEnv*, jobject) {
    return static_cast<jint>(g_pages.size());
}

JNIEXPORT jint JNICALL
Java_com_bk_drawing_NativeRenderer_getActivePage(JNIEnv*, jobject) {
    return static_cast<jint>(g_activePageIdx);
}

// Page navigation. Both queue through the pending-action drain so the
// mutation runs on the GL thread alongside other state changes (and
// shapes pending in g_pendingShapes get applied to the OLD page first).
JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_addPage(JNIEnv*, jobject) {
    enqueuePendingAction(kActionAddPage);
}

// Delete the page at `idx`. Queued through the pending-action drain so
// the GL-resource frees and disk recursion run on the GL thread. Refused
// by the drain if only one page remains.
JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_deletePage(JNIEnv*, jobject, jint idx) {
    if (idx < 0) return;
    g_pendingDeletePageIdx.store(idx);
    enqueuePendingAction(kActionDeletePage);
}

// Import a bitmap as a floating raster selection on a fresh layer. The
// pixel array is ARGB ints in Android Color order (alpha << 24 | R<<16 |
// G<<8 | B); we premultiply and reorder to RGBA8 here so the GL thread
// can upload the bytes directly without another conversion pass.
JNIEXPORT jboolean JNICALL
Java_com_bk_drawing_NativeRenderer_importImageAsSelection(
        JNIEnv* env, jobject, jint width, jint height, jintArray argbArr) {
    if (width  <= 0 || height <= 0) return JNI_FALSE;
    if (argbArr == nullptr) return JNI_FALSE;
    jsize n = env->GetArrayLength(argbArr);
    if (n != width * height) return JNI_FALSE;

    jint* src = env->GetIntArrayElements(argbArr, nullptr);
    if (src == nullptr) return JNI_FALSE;

    std::vector<uint8_t> rgba;
    rgba.resize(static_cast<size_t>(width) * height * 4);
    for (jsize i = 0; i < n; ++i) {
        uint32_t c = static_cast<uint32_t>(src[i]);
        uint8_t  a = static_cast<uint8_t>((c >> 24) & 0xFFu);
        uint8_t  r = static_cast<uint8_t>((c >> 16) & 0xFFu);
        uint8_t  g = static_cast<uint8_t>((c >>  8) & 0xFFu);
        uint8_t  b = static_cast<uint8_t>( c        & 0xFFu);
        // Premultiply to match the rest of the pipeline (tiles, content
        // textures, blend equation are all premultiplied).
        rgba[i * 4 + 0] = static_cast<uint8_t>((static_cast<uint32_t>(r) * a + 127u) / 255u);
        rgba[i * 4 + 1] = static_cast<uint8_t>((static_cast<uint32_t>(g) * a + 127u) / 255u);
        rgba[i * 4 + 2] = static_cast<uint8_t>((static_cast<uint32_t>(b) * a + 127u) / 255u);
        rgba[i * 4 + 3] = a;
    }
    env->ReleaseIntArrayElements(argbArr, src, JNI_ABORT);

    {
        std::lock_guard<std::mutex> lock(g_pendingImportMutex);
        g_pendingImport.active = true;
        g_pendingImport.width  = width;
        g_pendingImport.height = height;
        g_pendingImport.rgba   = std::move(rgba);
    }
    enqueuePendingAction(kActionImportImage);
    return JNI_TRUE;
}

// Rasterize a vector layer in place — its shapes get baked into fresh
// raster tiles and the layer's type flips to Raster. Queued through the
// pending-action drain so the GL work runs on the GL thread. No-op if
// the layer is already raster or out of range.
JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_rasterizeLayer(JNIEnv*, jobject, jint idx) {
    if (idx < 0) return;
    g_pendingRasterizeLayerIdx.store(idx);
    enqueuePendingAction(kActionRasterizeLayer);
}

// Merge the raster layer at [idx] down onto the raster layer at [idx-1]
// using premultiplied "src over dst" so the top layer's pixels stay on
// top. The source layer is then deleted and trailing layer dirs are
// renumbered. No-op (logged) when idx == 0 or either layer isn't
// raster. Queued through the pending-action drain; clears the undo
// stack since target tile pixels change in a way prior entries can't
// reverse. Caller should forceRedraw + resync layer state afterward.
JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_mergeLayerWithBelow(JNIEnv*, jobject, jint idx) {
    if (idx <= 0) return;
    g_pendingMergeLayerIdx.store(idx);
    enqueuePendingAction(kActionMergeLayerDown);
}

// Rasterize the currently-selected vector shape onto the raster layer
// directly below the source. Reads g_selection on the GL thread, so the
// caller doesn't need to pass parameters.
JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_rasterizeSelectionToLayerBelow(
        JNIEnv*, jobject) {
    enqueuePendingAction(kActionRasterizeShapeBelow);
}

// Move the page at `fromIdx` to `toIdx`. Same queueing rationale as
// deletePage. Last-write-wins on the side channel.
JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_movePage(JNIEnv*, jobject,
                                            jint fromIdx, jint toIdx) {
    if (fromIdx < 0 || toIdx < 0) return;
    {
        std::lock_guard<std::mutex> lock(g_pendingMovePageMutex);
        g_pendingMovePageFrom = fromIdx;
        g_pendingMovePageTo   = toIdx;
    }
    enqueuePendingAction(kActionMovePage);
}

JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_switchPage(JNIEnv*, jobject, jint idx) {
    if (idx < 0) return;
    g_pendingSwitchPage.store(idx);
    enqueuePendingAction(kActionSwitchPage);
}

// Render a given page's composite into an Android Bitmap (must be
// ARGB_8888 / RGBA_8888 in NDK terms). Used by the sidebar to draw page
// thumbnails. Synchronous on the GL thread — callers should invoke this
// from a callback that already runs there (e.g. via forceRedraw of a
// hidden offscreen surface) OR via the framework's renderer callback.
//
// In practice we accept calling from any GL-thread context, including
// inside an onDrawMultiBufferedLayer callback after the main composite
// has finished. The swapping of g_activePageIdx is restored before
// return so the caller's state is untouched.
JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_renderPageThumbnail(
        JNIEnv* env, jobject,
        jint pageIdx, jobject bitmap, jboolean drawChrome) {
    ensureInited();
    ensureLoaded();
    if (pageIdx < 0 || static_cast<size_t>(pageIdx) >= g_pages.size()) return;

    AndroidBitmapInfo info;
    if (AndroidBitmap_getInfo(env, bitmap, &info) < 0) return;
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        LOGE("thumbnail: bitmap format %d not RGBA_8888", info.format);
        return;
    }
    int w = static_cast<int>(info.width);
    int h = static_cast<int>(info.height);
    if (w <= 0 || h <= 0) return;

    // Save state we're about to clobber.
    size_t   savedPage      = g_activePageIdx;
    uint32_t savedScaleBits = g_viewScaleBits.load();
    GLint    prevFbo = 0;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &prevFbo);

    g_activePageIdx = static_cast<size_t>(pageIdx);
    // Pages load lazily, and a thumbnail is a full composite of one —
    // so this is a load trigger. Note the cost: refreshing every
    // thumbnail in a document pulls in every page. The Kotlin side
    // spreads a full refresh over several frames for exactly that
    // reason (see thumbnailRefreshQueue in DrawingSurfaceView).
    ensurePageContentLoaded(g_activePageIdx);

    // Fit the page rect into (w, h) with letterboxing. The transform is
    // doc → buffer with the same y direction in both spaces; combined with
    // glReadPixels' bottom-up byte order, this lands doc-top at the top
    // row of the bitmap (no row flip needed during memcpy).
    PageClip page = readPageClip();
    float pageW = page.active ? (page.maxX - page.minX) : static_cast<float>(w);
    float pageH = page.active ? (page.maxY - page.minY) : static_cast<float>(h);
    float minX  = page.active ? page.minX : 0.0f;
    float minY  = page.active ? page.minY : 0.0f;
    float s = std::min(static_cast<float>(w) / pageW,
                       static_cast<float>(h) / pageH);
    float offsetX = (static_cast<float>(w) - s * pageW) * 0.5f;
    float offsetY = (static_cast<float>(h) - s * pageH) * 0.5f;
    float t[16] = {
        s,    0.0f, 0.0f, 0.0f,
        0.0f, s,    0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        offsetX - minX * s,
        offsetY - minY * s,
        0.0f, 1.0f
    };

    // Mirror the thumbnail scale into g_viewScaleBits so screen-relative
    // widths (page outline etc.) come out as a consistent number of
    // thumbnail pixels regardless of doc/thumbnail size.
    {
        uint32_t bits;
        std::memcpy(&bits, &s, sizeof(bits));
        g_viewScaleBits.store(bits);
    }

    // Cached thumbnail FBO + color attachment (resized when dimensions
    // change). Static is fine — only used from the GL thread.
    static GLuint thumbFbo = 0, thumbTex = 0;
    static int thumbW = 0, thumbH = 0;
    if (thumbW != w || thumbH != h) {
        if (thumbFbo) { glDeleteFramebuffers(1, &thumbFbo); thumbFbo = 0; }
        if (thumbTex) { glDeleteTextures(1, &thumbTex);     thumbTex = 0; }
        glGenTextures(1, &thumbTex);
        glBindTexture(GL_TEXTURE_2D, thumbTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glGenFramebuffers(1, &thumbFbo);
        glBindFramebuffer(GL_FRAMEBUFFER, thumbFbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, thumbTex, 0);
        thumbW = w; thumbH = h;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, thumbFbo);

    // compositeAllLayers takes a jfloatArray; wrap our local matrix.
    jfloatArray transformArr = env->NewFloatArray(16);
    env->SetFloatArrayRegion(transformArr, 0, 16, t);
    // The page-edge outline is on by default for sidebar thumbnails;
    // disable it for export so the saved PNG / PDF has no border.
    bool savedSkipOutline = g_skipPageOutline;
    g_skipPageOutline = (drawChrome == JNI_FALSE);
    compositeAllLayers(env, w, h, transformArr);
    g_skipPageOutline = savedSkipOutline;
    env->DeleteLocalRef(transformArr);

    // Read pixels back. The bitmap may have stride > w*4 (rare for
    // ARGB_8888, but possible). Copy row-by-row to honor stride; the
    // y-flip in our transform already orients doc-top at bitmap-top, so
    // this is a straight memcpy per row.
    std::vector<uint8_t> buf(static_cast<size_t>(w) * h * 4);
    glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, buf.data());

    void* pixels = nullptr;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) == 0 && pixels) {
        uint8_t* dst = static_cast<uint8_t*>(pixels);
        for (int y = 0; y < h; ++y) {
            std::memcpy(dst + y * info.stride,
                        buf.data() + static_cast<size_t>(y) * w * 4,
                        static_cast<size_t>(w) * 4);
        }
        AndroidBitmap_unlockPixels(env, bitmap);
    }

    // Restore caller state.
    glBindFramebuffer(GL_FRAMEBUFFER, prevFbo);
    g_viewScaleBits.store(savedScaleBits);
    g_activePageIdx = savedPage;
}

// Render the preview-overlay shader pass against whatever FBO is
// currently bound. Caller is responsible for binding the target FBO
// and (for the front-buffer case) clearing it first. Used in two
// places: the live front-buffer preview inside extendStrokeBatch, and
// the multi-buffer commit "mask" pass that overlays the just-baked
// stroke into MB so the user can't see the brief gap between the
// front buffer being hidden by GLFrontBufferedRenderer's commit
// transition and the new multi-buffer state actually appearing.
//
// Uses g_coverage / g_belowFbo / g_aboveFbo as populated during the
// most recent extendStrokeBatch; g_strokeBrushColor + g_strokeTool
// pick the brush/eraser branch. Doesn't touch transform — the shader
// is a fullscreen quad and the per-pixel logic comes from the
// pre-rendered below/above/coverage textures.
static void renderPreviewOverlayToBoundFbo(int width, int height) {
    glViewport(0, 0, width, height);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(g_preview.program);
    glBindVertexArray(g_quadVao);
    glUniform1i(g_preview.uBelow,    0);
    glUniform1i(g_preview.uAbove,    1);
    glUniform1i(g_preview.uCoverage, 2);
    glUniform1i(g_preview.uMode,     strokeToolIsEraser() ? 1 : 0);
    glUniform3fv(g_preview.uBrushRgb, 1, g_strokeBrushColor);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, g_belowFbo.texture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, g_aboveFbo.texture);
    glActiveTexture(GL_TEXTURE2);
    // Shade strokes preview from the merged (dabs ∪ fill) coverage that
    // extendStrokeBatch rebuilds each batch; every other tool previews
    // from the accumulated dab coverage directly.
    glBindTexture(GL_TEXTURE_2D, g_strokeFillClosed ? g_shadeCoverage.texture
                                                    : g_coverage.texture);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, 0);
    glBindVertexArray(0);
}

JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_beginStroke(JNIEnv*, jobject) {
    ensureInited();
    ensureLoaded();
    applyPendingLayerActions();
    applyPendingShapes();
    ensureAtLeastOneLayer();

    g_strokeTarget = activeLayer();
    g_strokeTool   = g_currentTool.load();
    // Snapshot brush RGB + alpha so mid-stroke color/opacity changes
    // don't split a stroke.
    {
        uint32_t rgb = g_currentBrushColor.load();
        g_strokeBrushColor[0] = ((rgb >> 16) & 0xFFu) / 255.0f;
        g_strokeBrushColor[1] = ((rgb >>  8) & 0xFFu) / 255.0f;
        g_strokeBrushColor[2] = ( rgb        & 0xFFu) / 255.0f;
        g_strokeBrushAlpha    = currentBrushAlpha();
        g_strokeBrushColor[3] = g_strokeBrushAlpha;
    }
    // Same idea for the brush size + hardness — fixed for the duration
    // of the stroke so a mid-stroke slider drag doesn't split it.
    g_strokeBrushSizeScale = currentBrushSizeScale();
    g_strokeBrushHardness  = currentBrushHardness();
    // Both brush and eraser strokes use the WYSIWYG preview path so that
    // strokes appear under layers-above-active correctly. Defer the
    // setup to the first extendStrokeBatch (we don't have width/height/
    // transform here yet).
    g_needsPreviewPrep = true;
    g_current.samples.clear();
    g_liveEmitter.reset();
    // Shade tool: the stroke doubles as a closed path to fill. Snapshot
    // it per-stroke like the rest of the brush state, and reset the
    // fill's own scratch state.
    g_strokeFillClosed = (g_strokeTool == kToolShade);
    g_shadePolyDoc.clear();
    g_shadeBboxValid = false;
    g_shadeMaxPressure = 0.0f;
    // Lock prediction's active state for the duration of this stroke.
    // Mid-stroke toggles don't take effect until the next stroke so the
    // coverage mirror can't desync (a toggle on→off mid-stroke would
    // strand a predictionInFlight; off→on would force the next revert
    // to read a stale g_coverageReal).
    g_strokePredictionActive = g_predictionEnabled.load();
    // Snapshot uniform-alpha mode for the life of the stroke. The preview
    // dab pass and the bake both branch on this so any mid-stroke toggle
    // would otherwise render the head of the stroke with one mode and the
    // tail with the other.
    g_strokeUniformAlpha = g_strokeUniformAlphaEnabled.load();
    // Shade always runs uniform: outline and fill are one coverage mask
    // stamped once, so the shared border can't darken and the opacity
    // slider means exactly "tone of the shaded region".
    if (g_strokeFillClosed) g_strokeUniformAlpha = true;

    // In stacking mode the OVER-blend asymptote at the stroke spine is
    // ~1 - (1 - α)^N where N is the typical per-pixel dab overlap (4-10
    // depending on hardness/spacing/speed). With the slider passed
    // directly as the per-dab α, α = 0.25 already saturates the spine
    // near 75% and α = 0.5 saturates near 95% — the slider behaves as a
    // narrow lower-range curve, and uniform mode (where slider = stroke
    // alpha exactly) looks dramatically lighter at the same value.
    //
    // Apply the inverse so the slider means "stroke alpha at the spine"
    // in stacking mode too: per_dab = 1 - (1 - α)^(1/N). After N
    // overlapping dabs the cumulative spine returns to α. Self-crossings
    // beyond N still darken (preserving the "stacking" feel that
    // distinguishes the mode from uniform), but the baseline matches
    // the slider. Uniform mode leaves the slider as-is.
    if (!g_strokeUniformAlpha) {
        constexpr float kStackingDabOverlap = 6.0f;
        float compensated = 1.0f
            - std::pow(1.0f - g_strokeBrushAlpha,
                       1.0f / kStackingDabOverlap);
        g_strokeBrushAlpha    = compensated;
        g_strokeBrushColor[3] = compensated;
    }
}

// Batched stroke extension. xyp is [x,y,p, x,y,p, ...]; the first
// realCount triples are real samples (push to g_current.samples and
// mirror coverage afterwards); the rest are predicted (live preview
// only, reverted at the next batch). The batch does its dab work
// inline but the expensive coverage mirror and front-buffer overlay
// run ONCE at the appropriate boundaries, regardless of sample count
// — the per-sample versions of these would otherwise dominate GL
// thread time inside a single MotionEvent and build queue
// backpressure on long strokes.
JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_extendStrokeBatch(
        JNIEnv* env, jobject,
        jint width, jint height,
        jfloatArray transform,
        jfloatArray xypArr, jint realCount) {
    ATRACE_SCOPE("DrawingApp.extendStrokeBatch");
    ensureInited();

    jsize len = env->GetArrayLength(xypArr);
    if (len <= 0 || (len % 3) != 0) return;
    int total = len / 3;
    if (realCount < 0)     realCount = 0;
    if (realCount > total) realCount = total;
    int predCount = total - realCount;

    // Copy out once. Dab loops below want random access without per-
    // element JNI overhead; the array is small (typically <30 floats).
    std::vector<float> xyp(static_cast<size_t>(len));
    env->GetFloatArrayRegion(xypArr, 0, len, xyp.data());

    // Persist real samples for the eventual bake.
    for (int i = 0; i < realCount; ++i) {
        g_current.samples.push_back(
            { xyp[i * 3], xyp[i * 3 + 1], xyp[i * 3 + 2] });
    }
    // Mirror them into the shade path (same points, packed for the fill's
    // vertex buffer). Appending here keeps the per-batch cost proportional
    // to the new samples rather than to the whole stroke.
    if (g_strokeFillClosed) {
        for (int i = 0; i < realCount; ++i) {
            shadePathPush(xyp[i * 3], xyp[i * 3 + 1]);
            g_shadeMaxPressure = std::max(g_shadeMaxPressure, xyp[i * 3 + 2]);
        }
    }

    // Drop predicted dabs entirely if the stroke didn't begin with
    // prediction active — keeps coverage mirror coherent across
    // mid-stroke toggles (the snapshot taken at beginStroke wins).
    if (!g_strokePredictionActive) predCount = 0;

    GLint frontFbo = 0;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &frontFbo);

    // Capture before clearing — used below to decide whether to clear
    // the front buffer for the overlay render.
    bool firstBatchOfStroke = g_needsPreviewPrep;
    if (g_needsPreviewPrep) {
        preparePreviewBuffers(env, width, height, transform);
        g_needsPreviewPrep = false;
    }

    // Revert any predictions left over from the previous batch before
    // applying new real dabs. Only reaches here if g_strokePredictionActive
    // (the only path that sets predictionInFlight).
    if (realCount > 0 && g_predictionInFlight) {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, g_coverageReal.fbo);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, g_coverage.fbo);
        glBlitFramebuffer(0, 0, width, height,
                          0, 0, width, height,
                          GL_COLOR_BUFFER_BIT, GL_NEAREST);
        g_liveEmitter = g_liveEmitterReal;
        g_predictionInFlight = false;
    }

    // Bind coverage + dab program ONCE for the whole batch.
    float coverageRgba[4] = { 0.0f, 0.0f, 0.0f, g_strokeBrushAlpha };
    glBindFramebuffer(GL_FRAMEBUFFER, g_coverage.fbo);
    glViewport(0, 0, width, height);
    // Uniform-alpha mode: dabs MAX-blend into the coverage so overlaps
    // within a stroke clamp to the brush alpha rather than accumulating.
    // Stacking mode: standard OVER blend (the historical behavior).
    if (g_strokeUniformAlpha) {
        glBlendEquation(GL_MAX);
    } else {
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    }
    glUseProgram(g_dab.program);
    glBindVertexArray(g_quadVao);
    uploadMat4(env, g_dab.uTransform, transform);
    glUniform2f(g_dab.uScreen, (float)width, (float)height);
    glUniform4fv(g_dab.uColor, 1, coverageRgba);
    glUniform1f(g_dab.uHardness, g_strokeBrushHardness);
    {
        PageClip pageClip = readPageClip();
        uploadPageClip(g_dab.uPageMin, g_dab.uPageMax,
                       g_dab.uPageActive, pageClip);
    }

    // Real dabs first.
    for (int i = 0; i < realCount; ++i) {
        g_liveEmitter.extend(
            xyp[i * 3], xyp[i * 3 + 1], xyp[i * 3 + 2]);
    }

    // Single mirror at the end of the real region — captures the
    // post-real coverage and emitter state for the next revert.
    if (realCount > 0 && g_strokePredictionActive) {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, g_coverage.fbo);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, g_coverageReal.fbo);
        glBlitFramebuffer(0, 0, width, height,
                          0, 0, width, height,
                          GL_COLOR_BUFFER_BIT, GL_NEAREST);
        g_liveEmitterReal = g_liveEmitter;
        // Re-bind coverage as the draw target for predicted dabs.
        glBindFramebuffer(GL_FRAMEBUFFER, g_coverage.fbo);
    }

    // Predicted dabs.
    if (predCount > 0) {
        for (int i = realCount; i < total; ++i) {
            g_liveEmitter.extend(
                xyp[i * 3], xyp[i * 3 + 1], xyp[i * 3 + 2]);
        }
        g_predictionInFlight = true;
    }

    glBindVertexArray(0);
    // Restore blend equation in case we switched to GL_MAX for uniform
    // alpha — the front-buffer overlay below + all subsequent renders
    // assume the default GL_FUNC_ADD.
    if (g_strokeUniformAlpha) {
        glBlendEquation(GL_FUNC_ADD);
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    }

    // Shade tool: rebuild the merged coverage the overlay renders from.
    // The dab half lives in g_coverage and accumulates as usual; the
    // fill half has to be rebuilt every batch because the closing chord
    // sweeps as the pen moves, so the enclosed region can shrink.
    if (g_strokeFillClosed && g_shadeBboxValid) {
        // Real samples only — the predicted tail is deliberately left
        // out of the fill. Prediction buys tip latency, which the dabs
        // still get, but the chord is a long edge anchored at the far
        // end of the shape: feeding it the predictor's overshoot makes
        // the whole bridge (and, through the endpoint pressure, its
        // offset width) shimmer every frame for no perceptual gain.
        // It also keeps the preview's chord identical to the one the
        // bake commits, since the bake never sees predicted samples.
        ChordFill chord;
        float maxDabR = radiusOf(g_shadeMaxPressure);
        chord.offset  = chordOutwardOffset(g_shadePolyDoc.data(),
                                           g_shadePolyDoc.size() / 2,
                                           g_current.samples, maxDabR);
        chord.band    = chordFeatherBand(maxDabR);

        ShadePoly poly = uploadShadePolygon(g_shadePolyDoc.data(),
                                            g_shadePolyDoc.size() / 2, chord);

        float mat[16];
        env->GetFloatArrayRegion(transform, 0, 16, mat);
        float maxR = kMaxRadius * g_strokeBrushSizeScale;
        PartialScissor sc = dirtyBboxToScissor(
            mat, g_shadeMinX - maxR, g_shadeMinY - maxR,
                 g_shadeMaxX + maxR, g_shadeMaxY + maxR,
            width, height, /*pad=*/2);

        if (sc.w > 0 && sc.h > 0) {
            ensureViewFbo(g_shadeCoverage, width, height,
                          /*withStencil=*/true);
            glBindFramebuffer(GL_FRAMEBUFFER, g_shadeCoverage.fbo);
            glViewport(0, 0, width, height);
            glEnable(GL_SCISSOR_TEST);
            glScissor(sc.x, sc.y, sc.w, sc.h);
            glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
            glClear(GL_COLOR_BUFFER_BIT);

            // MAX for both halves: outline and fill share a border, and
            // OVER there would show as a darker rim at low opacity.
            glBlendEquation(GL_MAX);
            PageClip pageClip = readPageClip();
            drawShadeFill(poly, mat, (float)width, (float)height,
                          pageClip, g_strokeBrushAlpha, chord);

            // Merge in the accumulated dab coverage. The stamp shader
            // with a black color and brush mode emits (0, 0, 0, c) —
            // exactly the coverage format the dab pass writes.
            const float kZeroRgb[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
            glUseProgram(g_stamp.program);
            glBindVertexArray(g_quadVao);
            glUniform1i(g_stamp.uCoverage, 0);
            glUniform1i(g_stamp.uMode, 0);
            glUniform4fv(g_stamp.uColor, 1, kZeroRgb);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, g_coverage.texture);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
            glBindTexture(GL_TEXTURE_2D, 0);
            glBindVertexArray(0);

            glBlendEquation(GL_FUNC_ADD);
            glDisable(GL_SCISSOR_TEST);
        }
    }

    // Single front-buffer overlay render. Normally we clear FB first
    // and then run the preview overlay shader, so the FB pixels are
    // exactly the current stroke's coverage. EXCEPTION: on the first
    // batch of a new stroke (right after beginStroke), skip the
    // clear so the previous stroke's preview overlay survives in FB
    // for one frame. That bridges the GLFrontBufferedRenderer commit
    // transition: if the framework hides FB before the new
    // multi-buffer state lands, the user sees the still-visible old
    // preview instead of paper-white where the just-finished stroke
    // should be. The overlay shader's premultiplied output blends
    // additively on the surviving content, and the NEXT batch (which
    // does clear) returns FB to the correct state.
    glBindFramebuffer(GL_FRAMEBUFFER, frontFbo);
    glViewport(0, 0, width, height);
    if (!firstBatchOfStroke) {
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT);
    }
    renderPreviewOverlayToBoundFbo(width, height);
}

JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_commitStroke(JNIEnv*, jobject) {
    ATRACE_SCOPE("DrawingApp.commitStroke");
    ensureInited();

    GLint prevFbo = 0;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &prevFbo);

    {
        ATRACE_SCOPE("DrawingApp.commitStroke.prep");
        ensureLoaded();
        applyPendingLayerActions();
        applyPendingShapes();
        ensureAtLeastOneLayer();
    }

    size_t layerIdx = g_strokeTarget < layers().size() ? g_strokeTarget
                                                       : layers().size() - 1;

    // Snapshot tiles in the stroke's bbox BEFORE bake (only meaningful
    // for raster layers; bake is a no-op for vector). Skip snapshot if
    // there are no samples — the bake is a no-op too.
    bool   snapForUndo = (layerIdx < layers().size()
                          && layers()[layerIdx]
                          && layers()[layerIdx]->type == LayerType::Raster);
    int    tx0 = 0, tx1 = 0, ty0 = 0, ty1 = 0;
    UndoEntry entry;
    if (snapForUndo && currentStrokeTileBbox(tx0, tx1, ty0, ty1)) {
        ATRACE_SCOPE("DrawingApp.commitStroke.beforeSnapshot");
        // If a prior stroke deferred its saveTileToDisk for tiles in
        // this bbox, their cachedBytes is nulled and snapshotTilesInBbox
        // would fall through to a per-tile glReadPixels anyway. Drain
        // them first — saveTileToDisk does the same readback but ALSO
        // populates cachedBytes, so the snapshot below hits the fast
        // path. Net cost is the same as the fallback; the upside is
        // future strokes don't keep paying.
        drainPendingSaveTilesForBbox(layerIdx, tx0, tx1, ty0, ty1);
        entry.op = UndoOp::RasterStroke;
        entry.layerIdx = layerIdx;
        snapshotTilesInBbox(layerIdx, tx0, tx1, ty0, ty1, entry.beforeTiles);

        // Capture the rebake inputs while g_current.samples is still
        // populated and the brush snapshot is still the one the bake
        // is about to run with. Copy (not move) so the bake below can
        // still walk g_current.samples.
        entry.rebakeSamples       = g_current.samples;
        entry.rebakeBrushColor[0] = g_strokeBrushColor[0];
        entry.rebakeBrushColor[1] = g_strokeBrushColor[1];
        entry.rebakeBrushColor[2] = g_strokeBrushColor[2];
        entry.rebakeBrushColor[3] = g_strokeBrushColor[3];
        entry.rebakeBrushAlpha    = g_strokeBrushAlpha;
        entry.rebakeBrushSize     = g_strokeBrushSizeScale;
        entry.rebakeBrushHardness = g_strokeBrushHardness;
        entry.rebakeTool          = g_strokeTool;
        entry.rebakeUniformAlpha  = g_strokeUniformAlpha;
        entry.rebakeFillClosed    = g_strokeFillClosed;
        PageClip pc = readPageClip();
        entry.rebakePageActive    = pc.active;
        entry.rebakePageX0        = pc.minX;
        entry.rebakePageY0        = pc.minY;
        entry.rebakePageX1        = pc.maxX;
        entry.rebakePageY1        = pc.maxY;
    } else {
        snapForUndo = false;
    }

    std::vector<int64_t> dirty;
    {
        ATRACE_SCOPE("DrawingApp.commitStroke.bake");
        bakeCurrentStrokeIntoTiles(&dirty, layerIdx);
    }

    // Defer per-tile glReadPixels off the commit critical path. See
    // the enqueueDeferredSave / drainPendingSaveTiles block above for
    // the full rationale; trade-off is that any future code path
    // reading these tiles' cachedBytes will hit the readback instead
    // of the cache.
    {
        ATRACE_SCOPE("DrawingApp.commitStroke.deferSaveTiles");
        for (int64_t k : dirty) {
            enqueueDeferredSave(layerIdx, k);
        }
    }

    // Compute doc-px bbox of the dirty tiles and stash it so the
    // following renderDocument can take the partial-recomposite
    // path (restore prior MB content from cache, scissor to this
    // bbox, re-composite only the changed region).
    if (!dirty.empty()) {
        int tdx0 = INT_MAX, tdy0 = INT_MAX;
        int tdx1 = INT_MIN, tdy1 = INT_MIN;
        for (int64_t k : dirty) {
            int tx, ty;
            unpackTileKey(k, tx, ty);
            tdx0 = std::min(tdx0, tx);
            tdy0 = std::min(tdy0, ty);
            tdx1 = std::max(tdx1, tx);
            tdy1 = std::max(tdy1, ty);
        }
        g_pendingDirtyBbox.populated = true;
        g_pendingDirtyBbox.minX = static_cast<float>(tdx0 * kTileSize);
        g_pendingDirtyBbox.minY = static_cast<float>(tdy0 * kTileSize);
        g_pendingDirtyBbox.maxX = static_cast<float>((tdx1 + 1) * kTileSize);
        g_pendingDirtyBbox.maxY = static_cast<float>((tdy1 + 1) * kTileSize);
    }

    // Push the undo entry if anything was actually drawn. We don't
    // need a memcmp diff vs an after-snapshot anymore — a non-empty
    // dirty list means the bake touched real pixels, and redo will
    // reproduce the same result by re-baking from the captured
    // samples.
    if (snapForUndo && !dirty.empty()) {
        pushUndoEntry(std::move(entry));
    }

    glBindFramebuffer(GL_FRAMEBUFFER, prevFbo);
    g_liveEmitter.reset();
}

JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_renderDocument(
        JNIEnv* env, jobject,
        jint width, jint height,
        jfloatArray transform) {
    ATRACE_SCOPE("DrawingApp.renderDocument");
    ensureInited();
    ensureLoaded();
    // Defensive: covers pages created in-memory (addPage, the
    // ensureAtLeastOnePage fallback) reaching the compositor before
    // anything else triggered their load. Cheap once loaded — a bool.
    // Deliberately NOT inside compositeAllLayers: the partial-recomposite
    // path calls that with the scissor test enabled, and tile creation
    // does a full-texture glClear that a live scissor would truncate.
    ensurePageContentLoaded(g_activePageIdx);
    {
        ATRACE_SCOPE("DrawingApp.renderDocument.applyPending");
        // applyPending* will flip g_mbCacheValid to false if anything
        // beyond a plain stroke commit drained — those ops can't be
        // partial-recomposited and force the full path below.
        applyPendingLayerActions();
        applyPendingShapes();
    }

    // The framework binds MB.back before calling onDrawMultiBufferedLayer;
    // capture it so we can blit to/from g_mbCache and restore at the end.
    GLint mbBackFbo = 0;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &mbBackFbo);

    float xform[16];
    env->GetFloatArrayRegion(transform, 0, 16, xform);

    // Consume the pending dirty bbox (single-shot per renderDocument).
    PendingDirtyBbox bbox = g_pendingDirtyBbox;
    g_pendingDirtyBbox.populated = false;

    bool sameSize = (g_mbCacheWidth == width && g_mbCacheHeight == height);
    bool sameXform = std::memcmp(xform, g_mbCacheTransform, sizeof(xform)) == 0;
    bool canPartial = g_mbCacheValid && sameSize && sameXform && bbox.populated;

    if (canPartial) {
        ATRACE_SCOPE("DrawingApp.renderDocument.partial");
        {
            ATRACE_SCOPE("DrawingApp.renderDocument.partial.cacheRestore");
            // 1) Restore previous frame's pixels from the cache. NEAREST is
            //    fine — sizes match exactly.
            glBindFramebuffer(GL_READ_FRAMEBUFFER, g_mbCache.fbo);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, mbBackFbo);
            glDisable(GL_SCISSOR_TEST);
            glBlitFramebuffer(0, 0, width, height,
                              0, 0, width, height,
                              GL_COLOR_BUFFER_BIT, GL_NEAREST);
        }

        // 2) Confine the re-composite to the dirty region. Pad by a
        //    few buffer-px for antialiasing slop. compositeAllLayers
        //    does an unconditional glClear at start — scissor confines
        //    the clear, too, so the rest of the cached frame survives.
        PartialScissor s = dirtyBboxToScissor(
            xform, bbox.minX, bbox.minY, bbox.maxX, bbox.maxY,
            width, height, /*pad=*/8);
        if (s.w > 0 && s.h > 0) {
            ATRACE_SCOPE("DrawingApp.renderDocument.partial.composite");
            glEnable(GL_SCISSOR_TEST);
            glScissor(s.x, s.y, s.w, s.h);
            compositeAllLayers(env, width, height, transform);
            glDisable(GL_SCISSOR_TEST);
        }
        // If the dirty bbox is fully offscreen, the cached pixels are
        // already correct — nothing to redraw this frame.
    } else {
        ATRACE_SCOPE("DrawingApp.renderDocument.full");
        glDisable(GL_SCISSOR_TEST);
        compositeAllLayers(env, width, height, transform);
    }

    // 3) Snapshot the freshly-rendered MB.back into g_mbCache for the
    //    next frame's partial path. ensureViewFbo will rebind GL state;
    //    rebind MB.back afterwards so the framework gets the right
    //    surface to present.
    {
        ATRACE_SCOPE("DrawingApp.renderDocument.cacheBlit");
        ensureViewFbo(g_mbCache, width, height);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, mbBackFbo);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, g_mbCache.fbo);
        glDisable(GL_SCISSOR_TEST);
        glBlitFramebuffer(0, 0, width, height,
                          0, 0, width, height,
                          GL_COLOR_BUFFER_BIT, GL_NEAREST);
        glBindFramebuffer(GL_FRAMEBUFFER, mbBackFbo);
    }

    std::memcpy(g_mbCacheTransform, xform, sizeof(xform));
    g_mbCacheWidth  = width;
    g_mbCacheHeight = height;
    g_mbCacheValid  = true;
}

// ---- Test-only JNI access ------------------------------------------------
//
// Used by the androidTest fidelity suite to read tile state and apply
// pending undo/redo without going through compositeAllLayers (which
// would require an output FBO). Not called from production code.

JNIEXPORT void JNICALL
Java_com_bk_drawing_NativeRenderer_flushPendingActions(JNIEnv*, jobject) {
    ensureInited();
    ensureLoaded();
    applyPendingLayerActions();
    applyPendingShapes();
}

JNIEXPORT jint JNICALL
Java_com_bk_drawing_NativeRenderer_getLayerTileCount(
        JNIEnv*, jobject, jint layerIdx) {
    ensureInited();
    ensureLoaded();
    if (layerIdx < 0 || static_cast<size_t>(layerIdx) >= layers().size()
        || !layers()[layerIdx]) return 0;
    return static_cast<jint>(layers()[layerIdx]->tiles.size());
}

// Flat [tx, ty, tx, ty, …] of every existing tile in the layer.
// Order is insertion-order from the underlying map; tests should sort
// before comparing if they care.
JNIEXPORT jintArray JNICALL
Java_com_bk_drawing_NativeRenderer_getLayerTileCoords(
        JNIEnv* env, jobject, jint layerIdx) {
    ensureInited();
    ensureLoaded();
    if (layerIdx < 0 || static_cast<size_t>(layerIdx) >= layers().size()
        || !layers()[layerIdx]) return env->NewIntArray(0);
    const auto& tiles = layers()[layerIdx]->tiles;
    std::vector<jint> coords;
    coords.reserve(tiles.size() * 2);
    for (const auto& kv : tiles) {
        int tx, ty;
        unpackTileKey(kv.first, tx, ty);
        coords.push_back(tx);
        coords.push_back(ty);
    }
    jintArray out = env->NewIntArray(static_cast<jsize>(coords.size()));
    env->SetIntArrayRegion(out, 0, static_cast<jsize>(coords.size()),
                           coords.data());
    return out;
}

// Returns the kTileBytes interior pixels of a tile, or null if the
// tile doesn't exist. Reads from the CPU-side cache when populated
// (zero-cost) and falls back to glReadPixels otherwise.
JNIEXPORT jbyteArray JNICALL
Java_com_bk_drawing_NativeRenderer_readTileBytes(
        JNIEnv* env, jobject, jint layerIdx, jint tx, jint ty) {
    ensureInited();
    ensureLoaded();
    if (layerIdx < 0 || static_cast<size_t>(layerIdx) >= layers().size()
        || !layers()[layerIdx]) return nullptr;
    auto& tiles = layers()[layerIdx]->tiles;
    auto it = tiles.find(tileKey(tx, ty));
    if (it == tiles.end()) return nullptr;
    jbyteArray out = env->NewByteArray(static_cast<jsize>(kTileBytes));
    if (!out) return nullptr;
    if (it->second.cachedBytes
        && it->second.cachedBytes->size() == kTileBytes) {
        env->SetByteArrayRegion(
            out, 0, static_cast<jsize>(kTileBytes),
            reinterpret_cast<const jbyte*>(it->second.cachedBytes->data()));
    } else {
        std::vector<uint8_t> buf(kTileBytes);
        GLint prev = 0;
        glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &prev);
        glBindFramebuffer(GL_FRAMEBUFFER, it->second.fbo);
        glReadPixels(kApron, kApron, kTileSize, kTileSize,
                     GL_RGBA, GL_UNSIGNED_BYTE, buf.data());
        glBindFramebuffer(GL_FRAMEBUFFER, prev);
        env->SetByteArrayRegion(
            out, 0, static_cast<jsize>(kTileBytes),
            reinterpret_cast<const jbyte*>(buf.data()));
    }
    return out;
}

}  // extern "C"

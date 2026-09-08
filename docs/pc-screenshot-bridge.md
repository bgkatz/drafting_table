# PC screenshot bridge — design notes

Status: **not started**. Notes from a design review (Sept 2026), kept so the
feature can be picked up later without re-deriving the options.

## Goal

Annotate CAD screenshots (or any image from the PC) on the tablet without
the slide-deck / Drive round trip:

1. On the PC, capture a region of the screen (a hotkey).
2. It appears on the **current page of the open document** on the tablet,
   ready to sketch over.
3. A second hotkey pulls the annotated result back to the PC — ideally onto
   the PC clipboard, so it can be pasted into a deck or an email.

Both directions should be one keystroke with no file browsing.

## Is it possible?

Yes. Android has no shared clipboard with a PC, so it can't literally be
Ctrl+V on the tablet reading the PC clipboard. The equivalent is the PC
*pushing* the image to the app and the app dropping it on the page, and the
reverse for pulling. Everything else already exists in the app:

- `importImageAsSelection` lands pixels on the active page as a positionable
  floating raster selection (the "Import image…" flow).
- The export pipeline (`runExport` / `renderPageThumbnail`) renders a page
  to a bitmap at a chosen scale, with or without paper and guides, with text
  boxes rasterized at the export scale.

The missing piece is a transport layer plus a little glue on each side.

## Transport options

| Route | How | Pros | Cons |
|---|---|---|---|
| **Wi-Fi, app runs a tiny HTTP server** (recommended) | PC `POST /paste` with a PNG body; `GET /page.png` to pull. mDNS/NSD for discovery, pairing token header. | No cable, sub-second for a 3 MB PNG, works from any machine on the LAN, nothing touches the drawing path | Both devices must see each other on the network (corporate client isolation breaks it) |
| **USB via `adb forward`** | Same HTTP server; `adb forward tcp:PORT tcp:PORT` maps it to `localhost` on the PC | Works with no network; USB debugging is already enabled for development | Needs adb on the PC and the cable plugged in |
| **USB tethering (RNDIS)** | Tablet shares its network over USB; same HTTP server over the USB NIC | No adb | Must toggle tethering each session; fiddly |
| **Folder sync, no app changes** | Syncthing on both, ShareX saves captures into the synced folder, app watches its inbox folder with `FileObserver` | Zero transport code | Multi-second round trip; reverse direction ends as a file, not the PC clipboard |
| Quick Share for Windows / KDE Connect | Send file to tablet; app watches Downloads | Zero code | Needs an accept tap; KDE Connect doesn't sync image clipboards on Android |
| Windows Phone Link cross-device clipboard | — | — | Only supports a few phone brands; MovinkPad not among them |

**Recommendation:** Wi-Fi HTTP server as the primary path, USB `adb forward`
as the fallback using the identical protocol. Folder sync is the zero-code
plan B if the app-side work never happens.

## App-side design

### Server

- Plain `ServerSocket` on a background thread, no framework. Two or three
  routes; hand-parse the request line and headers.
- Runs only while `MainActivity` is resumed — rendering needs the GL surface,
  so a foreground service that survives screen-off buys nothing. Treat the
  feature as "tablet awake with the app open".
- Menu toggle "PC bridge: on/off" (persisted). Status-bar chip shows the
  address and port while on, e.g. `pc: 192.168.1.23:8642`.
- Advertise `_draftingtable._tcp` via `NsdManager` so the PC helper finds
  the tablet by name. Also fine to type the address once.

### Routes

| Route | Body / params | Behaviour |
|---|---|---|
| `POST /paste` | `image/png` (or JPEG) body; header `X-Token` | Decode on the worker thread, then on the UI thread go through the same path as "Import image…" (downsample to `kImportMaxDim`, `importImageAsSelection`). Remember the placed rectangle as the *last paste rect*. Respond 200 with the page index. |
| `GET /page.png` | `scale=1..3`, `bg=paper|transparent`, `guides=0|1`, `region=page|paste` | Render through the export pipeline. `region=paste` crops to the last paste rect so the result comes back at the screenshot's own pixel size plus annotations — the main use case. Blocks the HTTP thread on a latch until the GL pass completes. |
| `GET /status` | — | Doc name, active page, page count, last paste rect. Used by the helper to show a tray tooltip. |

### Placement of a pasted image

Make it a setting with two modes:

- **Floating** (like Import today): the image arrives as a floating raster
  selection, user positions it, tap elsewhere commits.
- **Auto-place**: fit-to-page (or at page centre at 1:1 if it fits) and
  commit immediately, so the screenshot is simply there when you look up.
  This is the mode the workflow wants; floating is the safe default.

Either way it becomes a new raster layer on the active page, same as import.

### Pull crop

`region=paste` needs the paste rect in doc px. Store it per page in memory
(and optionally in `page_setup.txt`-style metadata) when the import commits.
Render the page at `scale` × the ratio that maps the rect back to the
original screenshot's pixel size, then crop. Transparent background makes the
result composite cleanly onto whatever the PC pastes it into.

### Security

- Random pairing token generated on first enable, shown in the menu; the
  helper sends it in `X-Token`. Reject anything without it.
- LAN only; never bind to anything but the Wi-Fi interface (and `127.0.0.1`
  for the adb case).
- No route can write documents other than adding a layer to the open page.

### Threading and latency

Server and decode live on their own thread. The only main-thread work is the
existing import call, and the export render is the same GL pass the Export
dialog uses. Nothing is added to the stroke path, so idle pen latency is
unaffected; a paste landing mid-stroke would cost one import's worth of
GL-thread time, same as tapping Import today.

## PC-side design

Two pieces that combine well:

1. **ShareX for capture (push direction, zero code).** ShareX has region
   capture on a hotkey and a "custom uploader" that POSTs the capture to a
   URL. Point it at `http://<tablet>:8642/paste` with the token header and
   the push direction is done. Bonus: ShareX's own crop / arrow / blur tools
   before it leaves the PC.

2. **A small tray helper (pull direction, and push without ShareX).**
   ~150 lines of Python (`pystray`, `keyboard`, `Pillow.ImageGrab`,
   `requests`, `zeroconf`), packaged with PyInstaller:
   - Hotkey A: grab the PC clipboard image (after the user's own
     Win+Shift+S), or take a region shot, and POST it.
   - Hotkey B: `GET /page.png?region=paste&bg=transparent` and put the result
     on the PC clipboard; optionally also save to a folder.
   - Finds the tablet via mDNS; remembers the address and token; runs
     `adb forward` automatically when a tablet is on USB and Wi-Fi fails.

An AutoHotkey script calling `curl` is an even smaller alternative for the
pull hotkey.

## Effort estimate

| Piece | Rough size |
|---|---|
| App: server thread + routes + token + NSD | ~200 lines, half a day |
| App: paste placement modes + last-paste-rect + pull crop | half a day |
| App: menu toggle + status chip + settings | an hour |
| PC: ShareX uploader config | minutes |
| PC: tray helper | a couple of hours |

## Caveats to remember

- Tablet must be awake with the app in front; no background pastes.
- Same-network requirement for Wi-Fi; USB `adb forward` is the workaround.
- Pasted images are raster layers and follow the import size cap
  (`kImportMaxDim`, 2048 today). A 4K screenshot gets downsampled on the way
  in; the pull crop can still render at 2x to get a clean result back.
- Paste lands on whichever page is active; the helper's tray tooltip
  (from `/status`) is how you confirm which page that is.

## Suggested order when picking this up

1. Server + `POST /paste` + floating placement + token. Test with `curl`.
2. ShareX uploader config → the push direction is usable.
3. `GET /page.png` with `region=paste` + the pull hotkey helper.
4. Auto-place mode, NSD discovery, USB fallback, status chip polish.

# Changelog

Notable changes to the engine, in the format of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions
follow semantic versioning as described in [CONTRIBUTING.md](CONTRIBUTING.md).

## [Unreleased]

### Changed

- Auto on a display that is not a whole multiple of the source rate runs on the exact grid: every tick of the display
  (or every nth) shows a picture at its own media time. Factors put in-betweens at k / n between two frames, which is
  even on screen only at whole multiples; 24 fps on a 60 Hz display as x2 was 48 pictures on 60 ticks. Pictures off
  pace in Chrome (a 240 Hz panel presenting on every 4th or 2nd tick for 60 and 120 Hz): 24 fps on 60 Hz 25.5 % to
  2.5 %, 25 fps on 60 Hz 20 % to 1.9 %, 50 fps on 120 Hz 20 % to 2.5 %, 25 and 50 fps on 240 Hz 17 to 40 % to 2 to
  4 % (what remains is scene cuts). The same holds for 60 fps on 144 or 165 Hz. A display at a whole multiple keeps
  the factors and the source frames with them; a host with a present limit below the refresh rate keeps the factors
  too; under twice the source rate nothing is generated, as before.
- Display Hz mode and a target at the display's rate use the whole rate, on the measured tick period: the 97 %
  headroom left a tick without a new frame every 33 ticks, a held picture twice a second on a 60 Hz display (a 24 fps
  pan on 240 Hz: 1.5 to 5 % of the pictures off pace to 0.5 %). `display_headroom` remains for Auto's factors, a
  capped Auto and a host present limit.

### Fixed

- A source that skips frames is paced by its timestamps. A browser playing 60 fps at 1.5x delivers two of every three
  frames; Auto read the rate as more than the display has and generated nothing, and the exact modes gave every pair
  the average length (50 to 75 % of the pictures off pace). Auto takes the exact grid for such a source and each
  pair is as long as its timestamps say: 0.5 % off pace at 120 pictures a second.
- The display's rate was measured half a percent low on a clock with 0.1 ms steps (240 Hz as 238.6), and the
  presentation target of a factor went by the arrival average (x4 of 60 fps as 241 to 247 fps): at the display's own
  rate the queue dropped frames it could have shown a tick later, the drops drew a penalty, and fast motion lost its
  in-betweens for six seconds in about one run in three. The rate goes by the mean tick period and the target by the
  decoded cadence.
- An exact mode started generating a second after every start and seek (the first plan waited 16 frames like a
  change of plan): at once now, 0.5 s to 0.1 s at 24 fps.
- A seek or a change of speed threw the decoded cadence away, eight frames without in-betweens while it was
  relearned (1.2 s after a change of speed). It is kept.
- Video mode (the lookahead, cadence re-timing): the in-betweens of a pair were spread over the clock's step of the
  frame after it, 7.2 to 9.5 ms apart at x5 and off their frames; they take their own pair's span (a 24 fps pan,
  x5 on 240 Hz: 18.7 % of the pictures off pace to 2.2 %).
- The browser `CostSampler` timed every 30th generate, and in video mode, where an interval's requests start with the
  re-timing of a real frame, a stride over 2 or 5 requests only ever met those, which have no cost: the frame cost
  was never measured and Auto stayed on the seed. A sample that did not land (nothing written, or refused in a settle
  window) is taken again on the next call.
- `settle_active` in the report was false before the first tick, so a host measuring its GPU costs once measured
  inside the settle window, had the samples refused and stayed on the 10 ms seed (the reference player: x4 instead
  of x5 in about one run in four). The report is right from the first event, and the player measures until taken.

## [0.3.5] - 2026-11-09

### Fixed

- Target and Display Hz modes put the tick nearest each source frame onto that frame wherever it stood, up to half a
  step off its time: 23.976 fps at 60 showed phases 0, .41, .81, 1.0, 1.61 instead of 0, .4, .8, 1.2, 1.6, a shake
  on slow pans. A tick shows the source frame only where it lands on it; Target 60 on a 24 fps pan: 60 % of the
  pictures off pace to under 6 %.
- Static protection broke thin lines on camera pans. In a panned flat fill (anime line art) the fill between where a
  line is in the two frames reads unchanged in both, so the age pass held it as an overlay and the in-between's line
  was erased there: a dotted, broken outline over the whole pan (on a user's 720p anime pan, about 2500 pixels per
  in-between). A pixel flat in both frames now counts as changed; crosshairs, text and panel borders are held as
  before (crosshair error 1.8 against 7.0 unprotected, unchanged).
- The tick lock. A frame was presented on the first tick past its due time, and due times carry a few milliseconds
  of jitter (a browser delivers a 24 fps video up to 20 ms early or late frame by frame): every crossing of a tick
  boundary left one picture up three ticks and the next one. A frame due a whole number of ticks after the one before
  it is now held to that many ticks, and slips one only when it runs a tick and the jitter's reach late. x5 of
  23.976 fps on 240 Hz, a camera pan in Chrome: 4 to 7 % of the pictures off pace to 0 to 0.6 % (5 runs each); 60 fps
  game footage 0 to 0.7 %. The due-time shift of 0.3.3 is gone with it.
- A cold start ran Auto on the bare 10 ms cost default until the first timestamp query, a second at x4 of 24 fps
  (uneven on 240 Hz). Auto sizes by the host's cost seeds until then.
- The drop penalty stepped x4 down to x3 at 60 fps on 240 Hz, the uneven rate the even cadence avoids, and a run
  that had drawn a penalty showed it for six seconds (9 to 16 % of the pictures off pace). It now steps along the even
  factors: x4, x2, x1.

## [0.3.4] - 2026-11-09

### Fixed

- Auto stopped generating where the even-cadence rule (0.3.3) found no even factor but x1: 24 fps on a 120 Hz display
  (5x, a prime) before the GPU could afford x5, or on a GPU that never can. x1 is never the even choice now; such a
  GPU keeps x4. Caught by the C, C++ and Python binding checks.

### Documentation

- ARCHITECTURE.md describes the 0.3.3 pacing (even cadence, the schedule clock, the delay drift, the due-time shift,
  the missed-vsync exemption) and the scene-cut rule; the README lists OpenNOW among the projects built on the engine.

## [0.3.3] - 2026-11-09

### Fixed

- Even pacing. On game footage (60 fps on a 240 Hz display, Chrome, Auto) a quarter of the pictures (37 a second)
  reached the screen a display tick early or late; now 0 to 2 % do, the rest being frames the browser's decoder
  skipped (pace error p95 2.85 to 0.13 ms). 24 fps anime on the same display: 38.6 % to 1.4 %. A fixed x2 went from
  13 % to none, Display Hz mode from 9.1 % to 2.3 % (repeats of one picture counted as one showing).
  The causes, each measured:
  - Auto ran rates that land on the display's ticks unevenly, x3 of 60 fps on 240 Hz showing pictures 4.2, 4.2 and
    8.3 ms apart. When the display runs at a whole multiple of the source rate, the motion ceiling now rounds up to
    the next factor that divides it (anime's x3 and x4 on 240 Hz become x5, a flick's x3 at 60 fps becomes x4) and
    Auto takes the largest dividing factor the GPU allows; the display's own rate (x4 there, x5 for 24 fps on
    120 Hz) needs no headroom, since every tick then shows one frame. Display Hz mode does the same. Low latency
    keeps its headroom: its frames follow the arrival jitter.
  - The motion ceiling changed the factor three to five times a second, each change a hitch. A change the factor in
    use does not force now waits until Auto has asked for it 250 ms and the last change is a second old.
  - The presentation delay followed a floor that depended on the factor of the moment and the lateness, moving by
    18 ms several times a second at 2 ms per frame (12 % faster or slower for a few frames). The floor now depends on
    the cost and the factor Auto could run at, and the delay drifts at 0.25 ms per frame unless frames run late.
  - The schedule clock stepped by the arrival average and ran about 1 % slow when the frame callbacks land on the
    display's ticks, and restarted on every frame the decoder skipped. It now steps by the frames' timestamps, scaled
    by how fast they really come, follows the arrivals only slowly, and carries on over a skipped frame; a delivery
    that wanders further than a tick is followed as before. The in-betweens are spread over the clock's own step.
  - Due times sitting on a tick were shown on it or the next one by a fraction of a millisecond, so a steady x2
    alternated 4 and 12 ms gaps. A constant shift keeps them halfway between ticks when the output rate divides the
    display's.
- Fast camera moves are no longer taken for scene cuts. The classifier compares two frames, and a flick in a game
  differs from one frame to the next as much as a cut does, so a flick got no in-betweens: a 16.7 ms step among 4 ms
  ones, at the moment the motion is fastest. A cut now also needs a calm pair before it (a moving pair under half the
  threshold): on 22 s of 60 fps game footage 23 "cuts" became 3 and the output rate went from 143 to about 175 fps,
  while the anime fixture keeps its real cuts (its fast action, a run of large differences, is interpolated now).
- Auto matched the display against the fastest recent source interval, which millisecond timestamps (41 and 42 ms at
  23.976 fps) read as 24.39 fps: no whole multiple of 240 Hz, so anime fell back to x3 for moments. The cadence rule
  uses the decoded rate.
- At the display's own rate a missed vsync, or a source running a little fast, drops one frame with nowhere to go.
  Such a drop is now the compositor's (within the stall budget), not a step down for Auto: it had held Auto at x1
  for six seconds.

## [0.3.2] - 2026-11-09

### Added

- `engine.kernelSource(name)` (browser module; `Engine::kernel_wgsl` natively): the WGSL and entry point of a kernel
  exactly as the device compiles it. A host hands them all to `createComputePipelineAsync` before opening a stream:
  the browser compiles them in parallel, off the thread that composites the page, and the engine's own creation of
  the same pipelines then hits the browser's cache. On a site whose shader cache is cold (Chrome, RTX 4060 Ti) the
  start of a stream drops from 1.1-1.3 s to 0.4 s, with no frozen frame.

### Fixed

- Nearest-source regions (0.3.1) no longer drag still things: a pixel the two source frames agree on keeps the
  render. A still character over a panned background (anime) had its outlines pulled along with the pan where the
  network lost the motion; still outlines on such a scene: 30.6 to 44.7 dB against the source, the same as without
  the regions, while fast turns keep the regions' sharpness (local contrast 0.911 against 0.909 before, 0.867 without
  them). Exact render, realtime only (the fast render never had the regions).

## [0.3.1] - 2026-11-04

### Changed

- Realtime sessions and streams (games) finish the in-between with two per-pixel decisions on the exact render:
  - Still test: where the two source frames agree in place better than the frames warped by the network's flow
    agree with each other, the pixel stayed (HUD text, a crosshair, icons over a moving world) and keeps the still
    pixels. The flow lives on an 8 x 8 px grid and dragged thin overlays along with the world. HUD region on held-out
    CS2 with a synthetic HUD: 27.5 to 30.1 dB (Lossless Scaling's LSFG 3.1: 31.1), ahead of it on any motion over
    5 px.
  - Nearest-source regions: where the warped frames disagree over a region (a flick the network cannot follow), the
    in-between there is the nearer source frame alone, warped by the network's flow, decided once per frame. The
    blend of two misplaced copies read as a milky fog with double outlines; now a flick shows sharp frames and a short
    hold. Held-out CS2 flicks (over 60 px per 60 fps frame) keep 0.885 of the real frames' local contrast instead of
    0.60 (LSFG 0.89); PSNR and LPIPS, which reward the blend, drop.
  Together 0.47 ms more per generated 1080p frame on an RTX 4060 Ti at worst (the nearest-source passes act only where
  the warped frames disagree). Video mode keeps the plain render. `FH_STILL=0` / `FH_NEAR=0` in the environment turn
  them off for comparisons; `FH_BLEND=decisive` tries a per-pixel choice between the warped frames (not the default:
  its patches read no better than the fog).

### Fixed

- Browser: pushing a cross-origin `<video>` or canvas the page cannot read (a "tainted" source, for example a video
  from another domain without CORS) no longer kills the module. WebGPU's copy throws a SecurityError there, which
  wgpu turned into a panic (`RuntimeError: unreachable`) after which every call failed; `pushVideo` / `pushCanvas`
  now throw that SecurityError as an ordinary exception and the engine keeps working, so a host can reload the video
  in CORS mode and go on.

## [0.3.0] - 2026-11-03

### Added

- Camera warp (`Stream::mouse`, `set_mouse_captured`, `warp`; C `fh_stream_mouse`, `fh_stream_set_mouse_captured`,
  `fh_stream_warp`; JavaScript `mouse`, `setMouseCaptured`, `warp`; Python `mouse`, `set_mouse_captured`, `warp`):
  for games, the frame about to be presented is turned by the camera motion the mouse made since the moment it shows,
  so the camera answers the mouse at once however late the picture is. The host passes the mouse counts and whether
  the mouse is captured; the Stream measures each source frame's turn and learns how many pixels a count turns the
  picture and how late the picture shows it (on synthetic play: the lag to the millisecond, the gain within 0.1 %).
  It holds the warp off until the fit is strong, in menus and for games whose mouse drives a cursor. A weapon in
  hand and the HUD are detected and keep their pixels. 0.07 ms per source frame and 0.06 ms per presented frame at
  1080p; a host that never passes the mouse runs none of it. `warp::CameraWarp` and `warp::InputCalibrator` are
  public for hosts with their own textures.
- `Readback::try_finish`: a readback checked once per frame instead of awaited.

### Changed

- Output textures are also `COPY_DST` (the warp writes a turned frame back into its slot).
- `StreamReport` has `camera_warp` (the calibration, once started) and `counters.warped`.

## [0.2.0] - 2026-11-02

### Added

- `low_latency` (Pacer option, C `FhPacerOptions.low_latency`, JavaScript `lowLatency`): for games, the presentation
  delay is what the next in-between needs, (1 - 1/N) of a source interval plus the measured pair, frame and present
  cost and the arrival jitter, instead of the video default with its 60 ms floor. A 60 fps stream to a 240 Hz display:
  a source frame reaches the screen 18 ms after its arrival instead of 61, without drops in Auto (heavy network jitter
  22 ms, a slow GPU 23 ms, 30 fps sources 31 ms). Off by default; video keeps its behaviour.
- Trace replays report how long source frames take from arrival to the screen, and can charge GPU time per pair and
  per generated frame (`flyhigh-dev replay --pair-ms --mid-ms`), so a shorter delay that the work does not fit shows
  as late frames.

### Changed

- `FhPacerOptions` grew a field at its end (`low_latency`): C hosts rebuild against the new header.

## [0.1.4] - 2026-11-02

### Fixed

- Static protection no longer holds parts of a moving scene as overlays. After the camera stood still every pixel
  had a high age, and in the first pairs of a fast turn the pixels that matched by coincidence (more of them with a
  looser tolerance) were rendered as the unwarped average of the two frames: walls, palms and the edges of the weapon
  appeared twice. A pixel is now protected only after it stayed unchanged while the picture around it moved, in two
  pairs: the HUD and the crosshair qualify from the second in-between of a turn on. On a still-then-turn sequence
  the first in-between of the turn rises from 32.6 to 34.9 dB (30.1 to 34.9 with frames 4 and tolerance 8), the same
  as without protection, and the later ones keep the protected HUD (36.6 to 36.8 dB, against 35.8 without). The age
  pass costs 0.34 ms more per source frame at 1080p.

## [0.1.3] - 2026-11-02

### Added

- `compiled_kernels` and `precompile` (Rust `Engine`, the browser `FhEngine` as `compiledKernels` / `precompile`): a
  host records the kernels its configurations compile and compiles them a few per frame before the next stream opens.
  Chrome compiles in the process that composites the page and caches per site, so opening a stream on a new site froze
  a playing video for about a second (Framegen: three to four stalls of up to 1.1 s; one of 0.1 to 0.3 s with the
  kernels precompiled while the pointer rests on the video).

## [0.1.2] - 2026-11-02

### Changed

- Cadence re-timing keeps the real drawing where the transition pair's model could not match the frames (an arm
  appearing from behind the body): the re-timed render there was a guess and showed as a translucent or doubled limb.
  Still shots go through the same mask. On a clip with such a disocclusion the re-timed real slots are about half as far
  from the source frames, with the same 2 % of stopped steps; ripples and ghost limbs on crowd and walk shots are gone.
  No measurable cost: the same frame times natively, and in Chrome the same presents, delay and zero drops.
- The cadence option describes its price: presented frames are generated in-betweens, so big drawing steps show the
  generator's softness more often than plain interpolation does. A choice for the viewer, not a default.

## [0.1.1] - 2026-11-02

### Fixed

- The model loader checks a model completely before anything reads it: integer dimensions of at least 1, offsets and
  sizes computed without overflow, every tensor inside the blob, every weight the graph names present (the FiLM
  weights included), operands that are strings, repeat bodies that are lists. A malformed model directory is an
  error with the tensor's name; it used to pass the loader and panic in the builder.
- The browser classes refuse an option number that does not fit (`internal`, `staticProtection.frames` and
  `tolerance`) instead of turning it into a nearby one.
- Every C ABI function turns a panic into `fh_last_error` and its error return (null or -1); a panic that reached the
  boundary aborted the host's process.

## [0.1.0] - 2026-11-02

The first release: the engine as a library with a Rust API, a C ABI and a browser module, the reference hosts, and the
Framegen 2.0 extension built on it.

### Added

- The engine as the whole product in layers: the Core (`Engine`, `Model`, `Session`: two frames in, the frame at any
  phase out), the Stream (timestamped frames in, intervals out, with scene cut and held frame classification and the
  anime cadence), the Pacer (what to present on a display tick and what to generate, replayable from `trace v1`
  recordings). Reference hosts: a web page (`examples/web`) and a native player (`examples/player`).
- Surfaces: the Rust API, the C ABI (`flyhigh.h`, `libflyhigh`) and the JavaScript classes (`FhEngine`, `FhModel`,
  `FhSession`, `FhStream`, `FhPacer`); Python (ctypes) and C++ (header only) bindings over the C ABI; a C example.
- Options declared once with the `Options` derive; `docs/options.md` generated by `flyhigh-dev options-doc` and checked
  in CI.
- Static protection for HUDs and subtitles, and the host's protection mask (`protection_mask`, `write_mask`).
- D3D11 shared textures into the Vulkan device without a copy (`interop::d3d11_shared_texture`).
- DX12 through the DXC compiler: the DLLs beside the library or `EngineOptions::dxc_path`, `DeviceInfo::shader_compiler`
  in the report; the Windows release archive carries them.
- The bench suite: per-GPU baselines and `tools/baseline.py --check` with a 3% rule on times.
- GPU costs from the frames themselves: `Stream::push_timed` / `generate_timed` (`pushTimed` / `generateTimed` in
  the browser) run the real work under timestamp queries; the reference web host feeds them to the Pacer.
- The Pacer records its own `trace v1` (`record`, `trace`, `stop_recording`; `fh_pacer_record` / `fh_pacer_trace_json`
  in C, `record` / `trace` / `stopRecording` in the browser), with live option changes as events a replay applies.
- `adapters.js` next to the browser module: `VideoSource`, `CanvasPresenter`, `TickSource`, `CostSampler`,
  `ReadbackNudge`, `warmUp`, `pacerDefaults`; the reference web host is built from them.
- Stream and Pacer options and inputs in the browser accept camelCase keys beside snake_case.
- A release workflow on `v*` tags: the C library, header, bindings and model per platform, the browser module.

### Changed

- The browser module is built with a size-first profile: 1.02 MB, with a 1.3 MB budget in CI.
- `generate_timed` measures the mid in video mode too (a cadence composite stays unmeasured), so a host in video mode
  gets its mid cost samples instead of the seed.
- `VideoSource` (adapters.js) draws a frame whose size differs from the slot's through `importExternalTexture`, so a
  host can run the stream at the presentation size without a copy of its own; the 1:1 copy stays for matching sizes.
- Pipelines compile on first use; `warm_up` compiles a session's kernels before the first frame.
- Firefox hosts present at least 7 ms apart (the Pacer's `min_present_interval_ms`), the adapter's default.

### Removed

- The `fh_*` function families of the browser module (`fh_create`, `fh_vid_*`, `fh_ext_*`, `fh_prepare*`,
  `fh_autotune`, `fh_tune_*`, `fh_set_*`): the classes replace them; `engine.autotune` and the session's
  `pairTimed` / `midTimed` carry over what only they had.

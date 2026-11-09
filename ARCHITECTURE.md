# Architecture

How the engine is put together: the layers and the rules between them, the module map, what each layer does and does
not do, the network, the static plans, the kernels, the content-aware stages, and the three surfaces. The API itself
is in [docs/api-rust.md](docs/api-rust.md), [docs/api-c.md](docs/api-c.md) and [docs/api-js.md](docs/api-js.md); the
options in [docs/options.md](docs/options.md).

## The layers

A host gives the engine frames and the time; the engine gives back the frames to show and when. Everything between
those two lines (motion estimation, synthesis, scheduling, adapting to the GPU and the display) lives in the engine,
as options with defaults. A host is a thin shell: capture or decode, hand over, present.

| Layer | Modules | Takes | Gives | Enter here when |
|---|---|---|---|---|
| Core (`Engine`, `Model`, `Session`) | `engine.rs`, `model/`, `session.rs` | a device, a model, two frames and a phase `t` | the frame at `t`: `pair` once per interval, `mid` once per generated frame | you schedule frames yourself and only want the synthesis |
| Stream | `stream/` | source frames with timestamps | generatable intervals, their kind (motion, scene cut, held frame), the cadence statistics | you have your own display loop but want the engine to decide what each interval contains |
| Pacer | `pacer/` | display ticks, source frames, measured GPU costs | what to present on this tick and what to generate for the coming ones; no clock of its own, replayable from a trace | you want frame generation that just works on a display |
| Host adapters | `examples/web`, `examples/player`, `interop.rs` | the platform's frames and ticks | calls into the three layers above | you are a browser page, a native app, a capture plugin |

Each layer uses only the public API of the one below: the Pacer knows nothing about GPU internals, the Stream nothing
about displays, the Core nothing about time. The engine's own hosts (the reference web page, the native player, the
Framegen extension, the desktop overlay) use only these public layers, so a gap in the API shows up on us first. The
CPU convenience `api.rs` (frames as byte buffers, the display order resolved for the host) sits on the Core and the
Stream.

Rules every layer follows:

1. **The engine never reads a clock.** Hosts pass `now` (milliseconds of their monotonic clock) and the source
   timestamps into every call that needs time, so scheduling is a pure function of its inputs and a recorded trace
   replays bit for bit in a test.
2. **Every heuristic is an option** with a default, a doc comment and a field in the report; the options reference is
   generated from the declarations. Auto is a preset of options, not hidden magic.
3. **The engine reports what it did**: each layer's `report()` says what was chosen, measured and dropped, and why, as
   a plain structure that serialises to JSON.
4. **No panics on input.** A bad option, slot, model or frame is an error in every surface; a panic is an engine bug.
5. **One semantics, three surfaces.** Rust is the source of truth; the C ABI and the JavaScript classes carry every
   layer, and the Pacer's parity across them is tested on recorded traces.
6. **Behaviour stays measurable.** Golden plans, parity, quality baselines and trace tests gate every change.

### Vocabulary

| Word | Meaning here |
|---|---|
| source frame | a frame the host decoded or captured, with its presentation timestamp (pts: seconds of media or capture time) |
| interval | the time between two consecutive source frames, [pts_a, pts_b) |
| phase `t` | position inside an interval, 0 = the first source frame's time, 1 = the second's |
| pair | the motion estimate for one interval (the expensive, t-independent half of the network) |
| mid | one synthesised frame at phase `t` of a pair (the cheap, t-dependent half plus the render) |
| factor | output frames per source frame (2 = one mid per interval) |
| slot | an engine-owned input or output texture a host writes into or presents from |
| plan | the fixed list of GPU dispatches for a session (static; built once per size) |
| display tick | the host's vsync or `requestAnimationFrame` callback, the Pacer's heartbeat |

## One engine crate, every target

The engine is one Rust library crate (`crates/flyhigh-engine`); the C ABI crate, the wasm module crate and the
developer CLI wrap it. Natively it runs on wgpu's Vulkan, DX12 or Metal backend; compiled to `wasm32-unknown-unknown`
the same code runs on the browser's WebGPU through wgpu's web backend. The GPU work is in the WGSL files of `shaders/`,
included at compile time, so there is one copy of every kernel for all targets.

| Module (`crates/flyhigh-engine/src/`) | Role |
|---|---|
| `engine.rs` | `Engine`: the device (own, or the host's through `with_device`), the compiled kernels, the autotune table, `DeviceInfo` and the report |
| `model/` | `Model`: the weight blob and manifest (`blob`), the loader checks, helpers over the operation graph (`graph`) |
| `session.rs` | `Session` and `SessionOptions`: the plans for one frame size, own slots or the host's views, `pair` / `mid`, the protection mask, `Readback` results |
| `stream/` | `Stream` and `StreamOptions` over `gen_tex`: timestamps and intervals, the pair classifier (`classify`), the report |
| `pacer/` | `Pacer` and `PacerOptions`: display and source estimators (`display`, `source`), the output rate policy (`rate`), the exact-grid planner (`plan`), the presentation queue (`queue`), the AIMD controller (`control`), the cost model (`costs`), `trace v1` recording and replay (`trace`) |
| `options.rs` | `EngineOptions`, `NetOpts`, the defaults; `OptionInfo` and `groups()` behind the `Options` derive (the reference is generated from them) |
| `gpu/` | `Gpu`: device setup and adapter choice (`device`), the DX12 shader compiler discovery (`dxc`), shader compilation and the f32 rewrite (`pipeline`), buffers and the per-plan arena (`alloc`), recording and submitting a plan (`exec`), GPU timing (`timing`), readback (`readback`) |
| `ops/` | the kernels as plan steps: `step`, `tensor`, `params`, `elementwise`, `interp`, `network`, `render`, `texture` |
| `ops/conv/` | convolutions: the kernels (`kernels`), the candidates (`candidates`), per-shape autotune (`autotune`), the persisted table (`tune_cache`) |
| `net/` | graph to plan (`build`), the plan sessions underneath the Core (`session` on byte frames, `session_ext` on textures with the fast render, `session_tex` ping-pong) |
| `gen_tex.rs` | the texture-path generator with modes (realtime, video with lookahead, cadence) that the Stream wraps |
| `gmhs.rs` | the repair stage and the cadence statistics and composite |
| `interop.rs` | Windows: a D3D11 shared texture into the Vulkan device without a copy |
| `api.rs` | the CPU convenience: `api::Engine`, `api::Generator`, frames as bytes |

| Other crates and directories | Role |
|---|---|
| `crates/flyhigh-options` | the `Options` derive (proc macro): doc comment, default and `#[option]` attributes become `OptionInfo` |
| `crates/flyhigh-capi` | the C ABI: `fh_engine_*`, `fh_model_*`, `fh_session_*`, `fh_stream_*`, `fh_pacer_*`, `fh_generator_*`; the header `include/flyhigh.h` |
| `crates/flyhigh-web` | the wasm-bindgen module: the classes `FhEngine`, `FhModel`, `FhSession`, `FhStream`, `FhPacer` and `fh_pacer_replay`; `js/adapters.js` shipped beside it |
| `crates/flyhigh-cli` | `flyhigh-dev`: self-tests, parity, quality regression, timings, trace replay, the options reference; one module per subcommand |
| `crates/flyhigh-engine/tests` | the engine's tests, the golden plans (`golden/`) and the recorded host traces (`traces/`) |
| `examples/player`, `examples/web`, `examples/c` | the reference native host, the reference web host (with the browser build in `examples/web/pkg`, not in git), a C host |
| `bindings/` | Python (ctypes) and C++ (header only) over the C ABI |
| `shaders/` | the WGSL kernels |
| `bench/` | per-GPU baselines and the regression rule |
| `models/` | the shipped model and the catalogue |

The kernel fixtures and the full-graph golden dumps the self-tests compare against (`fixtures/`) are not in git; see
[docs/development.md](docs/development.md).

## The Core

`Engine` is the device: its own (adapter selection, features and limits per `EngineOptions`) or one the host already
has (`with_device`: a game's or a player's wgpu device, a page's WebGPU device), checked for the features the engine
needs (shader-f16, or the f32 kernels without it; timestamps and cooperative matrices optional) and never
reconfigured. `Model` is CPU data: the weight blob, its manifest and the operation graph, checked completely when it
loads ([docs/model-format.md](docs/model-format.md)); several models may live beside one engine and a model may be
reloaded without touching the device. `Session` holds the plans of one model for one frame size and the slots the
host writes into and presents from; the two halves of the network are `pair(a, b)` for the interval's motion and
`mid(t, out)` for the frame at phase `t`, with timed variants that run the stage under timestamp queries and hand back
its GPU milliseconds (the Pacer's cost samples).

The Core is deterministic: the same slots and the same `t` give the same pixels. It knows nothing about timestamps,
clocks, which interval or phase to generate, scene cuts, held frames or displays. A new frame size is a new session;
the engine keeps the compiled kernels and the autotune table, so a rebuild is tens of milliseconds. The first version
keeps engine-owned slots, which is what every host uses today; the device holds its caches in `Rc`, so an engine stays
on one thread ([decision 0005](docs/decisions.md)).

## The Stream

A Stream turns source frames with timestamps into intervals that can be generated. It owns the input ring and the
output slots, remembers which slot holds which frame, runs the pair stage at the right time, classifies each pair,
keeps the per-pair cadence statistics and answers one question: for the interval I can show now, what does phase `t`
look like.

- **Realtime.** `push` runs the pair (previous, new) and returns the interval at once; `generate(t)` renders phase t.
  For games, capture and anything interactive.
- **Video.** One source frame of lookahead: `push` runs the pair in the "next" context and the interval returned is
  the one that started two frames back. Cadence re-timing needs it; the statistics of the newest pair come back
  asynchronously (`stats_pending`). Three rotating pair contexts.
- **Classification.** The two frames are sampled on a 48 x 27 grid (`classify.wgsl`); a mean per-channel difference
  above `scene_cut.threshold` is a cut when the moving pair before it differed by less than half the threshold (a fast
  camera move is a run of large differences, a cut a lone spike after a calm pair), a mean below `held_frames.mean` with
  every cell below `held_frames.max` is a hold, anything else motion. The readback is asynchronous (three may be in
  flight; beyond that the pair counts as motion) and lands before the interval's first mid is due. The mean cell
  difference in 0..765 units is the motion statistic the Pacer's motion-adaptive ceiling uses.
- **Order with the Pacer.** Classify the new frame against the previous one, then `pacer.source_frame` with the kind,
  generate the requests it returns, then `push`: the pair must not change before the previous interval's mids are out.
- **Held frames are reported, never dropped silently**: the Pacer shows the frame and generates nothing for the
  interval, and the report counts it.
- **Costs.** `push_timed` and `generate_timed` run the real pair and mid under timestamp queries: no second pass, the
  numbers are the frames' own. A cadence composite (two mids and a mask) is the one generated frame without a cost.
- **Readbacks** (`gpu::readback::Readback<T>`) are one-shot futures over a mapped staging buffer; native hosts block,
  the browser awaits. Firefox resolves them only on its GPU polls unless work is submitted; the browser adapter's
  nudge handles that.

- **Camera warp.** Started by the host's first `mouse` call; each `push` then measures the frame's turn, and
  `warp(frame, out, now)` writes the frame about to be presented, turned by the mouse motion since, into an output
  slot (false: the warp is off, present the frame as it is). See [Camera warp](#camera-warp).

Decisions: the scene-cut threshold defaults to 0.08 (20 / 255), what the CLI uses; a host may set its own preset.
No extrapolation (intervals that look ahead, `t` > 1): predicted frames fail exactly when the player acts. What the
player feels of the delay is the camera, and the camera warp brings it up to date from the input instead.

## The Pacer

A pure scheduler. The host feeds events with its own clock; the Pacer answers what to present on this tick and what
to generate for the coming ones. It never calls the GPU and never reads a clock, so a recorded trace replays exactly.
Frames are the host's own ids (slot indices); the Pacer never touches a texture.

Per display tick: `tick(now)`; present `tick.present` (or nothing); for every request in `tick.generate`, generate
phase `t` of the pair through the Stream and call `generated(now, id, output)`, or `skipped(id)` when the Stream
answers that the source frame is enough. Per source frame, once its kind is known: `source_frame(now, frame)`,
generate the requests it returns (mids of the previous pair that must go out before the pair changes), then push the
frame. Costs come from the Stream's measurements through `cost_sample`; `host_stall` reports a loop that stood still,
`mode_change` a fullscreen change or a resize, `reset` a seek. The reference web host (`examples/web/main.js`) is the
executable version of this paragraph.

Inside, in the order a tick runs:

1. **Display estimator** (`display.rs`). From tick intervals: the refresh floor (service interval, not the fastest
   sample), the nominal rate (snapped to 360/240/180/165/144/120/100/90/75/72/60/50 within 3 %), stability
   (coefficient of variation <= 0.12 over 32 samples), transitions in either direction (10 agreeing samples, a faster
   candidate survives 3 misses). Capacity = min(raw, nominal, the host's present limit). A tick gap above
   max(25 ms, 4 x floor) is a host stall.
2. **Source estimator** (`source.rs`). From source pts: the interval (trimmed mean over 32 samples, snapped to common
   video rates within 1.5 %, held through 3 % noise once nominal, a transition after 4 agreeing samples outside 12 %),
   the arrival-interval EMA, the unique-frame interval (holds and cuts excluded) and the PLL schedule clock that places
   presentations. While frames arrive within 1.5 display ticks of it, the clock steps by each frame's pts step scaled
   by how fast frames really came against their pts over the last 2 s, and follows the arrivals at 2 %; without pts
   or with a delivery that wanders further it steps by the arrival EMA and follows at 8 %; it resyncs hard beyond
   80 ms and carries on over a frame the decoder skipped.
3. **Output rate policy** (`rate.rs`). Modes: `Factor(n)`, `Auto` (as many as the display and GPU allow, capped by
   motion), `DisplayHz`, `Target(fps)`; Auto with `fps_limit` is a strict-ceiling target. Even cadence: when the display
   runs at a whole multiple n of the decoded source rate (within 1 %), Auto rounds the motion ceiling up to the next
   factor dividing n and takes the largest dividing factor the GPU allows (never x1 for it; a GPU that cannot reach an
   even rate keeps an uneven one below half the display's rate), and `DisplayHz` outputs n x source exactly; the
   display's own rate needs no headroom, every tick showing one frame (not with `low_latency`). A factor change the
   factor in use does not force waits until Auto has asked for it 250 ms and the last change is 1 s old. The 2x floor,
   the display headroom (0.97), compute capacity from the pair / mid / present costs within `compute_budget` of the
   source interval, the queue capacity, the clamp reason (display / gpu / runtime / minimum) and a warning text. Exact
   modes hold their plan until a different one has been proposed 16 frames in a row. When the GPU looks too slow for 2x,
   one real pair with tiny assumed costs every 2.5 s measures it (`probe`).
4. **Presentation planner** (`plan.rs`). Exact modes: target ticks on the exact output grid in media phase, carried
   across intervals; each tick becomes one presentation, an anchor (`previous` / `current`) or an `interpolate` at phase
   t; cuts and holds disable interpolation but keep the grid; a tick that lands on a source frame (within 5 % of a step)
   shows that frame, every other tick interpolates at its own phase. Factor modes: mids at k / n of the interval and the
   source frame at its arrival plus the delay. The delay target (pair lookahead in exact modes + pair cost + 2 x mid
   cost + 25 ms + burst pad, floor 60 ms or 42 ms when a frame is cheap and the steady factor is 3 or less, max 180 ms)
   sets how far ahead frames are due; it drifts 0.25 ms per frame and rises 2 ms per frame only while frames run late
   (exact modes: a whole interval). Factor-mode mids are spread over the clock's own step. With `low_latency` (games)
   the target is what the first in-between needs and no more: it is due 1/N of an interval after its source frame and
   needs the next one, so (1 - 1/N) x interval + pair + frame + present cost + the arrival jitter (a decaying envelope
   of how late frames come against the schedule clock) + burst pad, with no floor; it rises 2 ms per frame and falls
   0.25, so the jitter it follows does not squeeze the timeline.
5. **Presentation queue** (`queue.rs`). Bounded (24), ordered by due time; outside the exact modes, when the output
   rate divides the display's and is below it, a shift keeps the due times halfway between ticks, following the
   phase a little on every due (source and display clocks drift apart, 23.976 fps on 240 Hz by about 1 ms a second)
   and wrapping by a whole tick when the drift has used one up; one presentation per tick; the oldest entry
   inside the lateness contract (3 output intervals) is presented when the display can service the target, else the
   newest due. Entries before it are dropped and counted; entries due before the last presented one are stale. A
   requested mid that is not there in time shows its anchor instead (`fallback`).
6. **Auto controller** (`control.rs`, AIMD). Drop pressure is a leaky integrator (tau 300 ms); several frames due in
   one tick count as one event; drops within 20 ms after a host stall are not charged while stalls are rarer than 4
   per 2 s, and a missed vsync (a tick gap over 1.5 floors) or a single drop at the display's own rate counts as
   such a stall; pressure > 1.2 lowers the factor by one (by two above 3), not more often than every 500 ms, along
   the even factors when the even cadence holds (x4, x2, x1 at 60 fps on 240 Hz); after 6 s with
   pressure < 0.15 the factor recovers one step. Tick strain (ticks 1.7 x slower than the floor) adds a provisional
   one-step penalty that lifts 500 ms after the strain ends. A settle window (1200 ms, the extension's measured
   fullscreen reflow) after a mode change exempts drops. The motion-adaptive ceiling (> 45 motion units: 2, > 28: 3,
   > 16: 4, else 6) is an option. Auto that cannot run by the budget probes one 2x pair at 2.5 / 5 / 10 / 20 s.
7. **Just-in-time requests.** A mid is requested `pair + 2 x (mid + present) + 8` ms before its slot, so present
   blits interleave with computes instead of the first mid waiting for the whole batch.
8. **Host present limit.** No two presents closer than `min_present_interval_ms` (Firefox: 7 ms); the display
   capacity is capped accordingly and a frame that would violate it waits for the next tick.

**Traces.** `trace v1` (`pacer/trace.rs`): `{ "version": 1, "host": "...", "options": {...}, "events": [...] }` with
events `tick {t}`, `source {t, id, pts, kind_of, motion, skipped}`, `cost {t, stage, ms}`, `stall {t, ms}`,
`mode_change {t}`, `reset {t}`, `options {t, options}`. The Pacer records its own session (`record`, `trace`,
`stop_recording`); `trace::replay` runs a trace through a Pacer standing in for the host and returns the report plus a
per-second table; `flyhigh-dev replay <trace.json>` prints it. Floats round-trip exactly, so a replay is bit for bit.
`tests/pacer.rs` holds the synthetic regression set (a steady 120 Hz Chrome at 24 fps, Firefox with its periodic
stalls and the 7 ms limit, a saturated loop, a fullscreen change, Display Hz from 24 on 60, a fixed factor through a
cut, the estimators and the planner on their own) and replays the recorded sessions of `tests/traces/` (its README
lists them: the extension in Chrome and Firefox, the native player, the reference web host). Each new host finding
becomes a trace and an assertion.

Open: the game mode and the insert mode the desktop's vblank pacer needs (real frames shown natively, predicted
in-betweens on other refreshes, vblank-phase locked) become a Pacer mode of their own with extrapolation intervals
from the Stream.

## Host adapters

Optional, thin, replaceable: each turns a platform's way of delivering frames and ticks into the calls of the three
layers, and the engine's outputs back into something the platform shows.

Web (`crates/flyhigh-web/js/adapters.js`, shipped next to the wasm module):

| Piece | Does |
|---|---|
| `FhEngine.create(options)` | creates the `GPUDevice`; `engine.device()` is the one the host presents on |
| `VideoSource` | writes a `<video>` frame into a stream slot: `pushVideo` (`copyExternalImageToTexture`) where the browser takes it and the frame has the slot's size, an `importExternalTexture` render pass into the slot otherwise (Firefox, or a frame of another size) |
| `CanvasPresenter` | configures the canvas and blits the slot the Pacer names on each tick |
| `TickSource` | `requestAnimationFrame` as `pacer.tick(now)`; a tick later than 50 ms after the previous one is a host stall |
| `warmUp` | the session's or stream's kernels compiled and awaited through `onSubmittedWorkDone` before playback |
| `ReadbackNudge` | while a readback is pending in Firefox, `queue.submit([])` every millisecond so its promise resolves in 5 ms instead of 70 to 100 |
| `CostSampler` | `push` and `generate` with a timed run every thirtieth call and on every probe, the GPU milliseconds fed to `costSample` |
| `pacerDefaults()` | the host quirks below as Pacer options for this browser |

The reference web host (`examples/web/`) is these pieces plus a page's own UI over `FhStream` and `FhPacer`; it is
the template for a browser host and the bench for browser findings.

Native (`examples/player`, the `interop` module):

| Piece | Does |
|---|---|
| ticks | Windows: DWM vblank timing with a PLL that keeps the phase; elsewhere the window system's frame callbacks |
| `interop::d3d11_shared_texture(engine, handle, &desc)` | a D3D11 shared texture (screen capture) into the Vulkan device without a copy, then `add_input_texture` on a session or a stream; C: `fh_session_import_d3d11` / `fh_stream_import_d3d11`. DX12 import, Vulkan external memory on Linux and IOSurface on macOS are open |
| presentation | the host's swapchain; the engine hands back the output texture, the host blits or composites it |

A plugin for OBS or a media framework is a native host with frames arriving as textures or buffers: the same import
functions, `Stream` and `Pacer`, with the plugin's own clock.

Host quirks, each with the measurement behind it, applied as option defaults by the adapter, never silently inside a
layer:

| Host | Quirk | Default set by the adapter |
|---|---|---|
| Firefox (WebGPU) | the compositor stalls 40 to 90 ms when two canvas presents are closer than about 7 ms (5 ms still stalls, 7 ms is clean) | `min_present_interval_ms = 7` |
| Firefox | pipelines compile serially in the GPU process, 0.1 to 2 s each | `warmUp()` before playback |
| Firefox | `mapAsync` resolves only on GPU polls (70 to 100 ms) unless work is submitted | the readback nudge |
| Firefox | `copyExternalImageToTexture` rejects `<video>` and `VideoFrame` | the `importExternalTexture` pass |
| Firefox content scripts | DOM prototypes are Xray wrappers (no patching, no spread of `device.features`) | `engine.device()` and the texture getters |
| Chrome MV3 extension | the page's CSP governs WebAssembly in a content script: a `script-src` without `'wasm-unsafe-eval'` refuses the module, and frames cannot be handed to an extension iframe | documented; the extension says so on such a page |
| Windows Optimus laptops | capture lands on the iGPU, the engine on the dGPU | the capture adapter option (desktop) |
| Windows DX12 (wgpu) | wgpu compiles HLSL through FXC unless it finds `dxcompiler.dll`: 139 s for the kernel set, no shader-f16 | the engine looks for the DLL beside its own module; `EngineOptions::dxc_path`; `DeviceInfo::shader_compiler` says what loaded |

## Options and reports

An option is declared once, as a field of its layer's options struct (`EngineOptions`, `SessionOptions`,
`StreamOptions`, `PacerOptions`): the doc comment is its description, the default instance gives its default, and
`#[option(default, values, c)]` adds what the type cannot say. The `Options` derive (`crates/flyhigh-options`) turns
that into `OptionInfo`; a field without a doc comment does not compile. `flyhigh-dev options-doc` writes
[docs/options.md](docs/options.md) from it and CI fails when the file is stale; rustdoc shows the same text.
`validate()` names the field and the reason; dependencies between options (cadence needs repair at least 1, linear t
needs the fast render, tensor cores need f16) are resolved there and the resolution is reported, not silent. Pacer
options may change live (`set_options`); Session and Stream options rebuild the object.

Each layer reports what it did: `engine.report()` (kernels compiled, autotune entries, tensor bytes),
`session.report()` and `stream.report()` (sizes, slots, counters), `pacer.report()` (the display and source
estimates, the plan with its clamp reason and warning, the factor, penalties and pressure, delay and lateness, the
measured costs, the queue, the counters of presents and drops by cause). Counters are monotonic since creation; a
host diffs them for rates. The `FH_*` developer variables ([docs/development.md](docs/development.md)) are for the
CLI and the kernel builders only; library hosts set options through the API.

Open: named presets across the layers, and a single report tree with `to_json()` instead of one per layer.

## Errors

Every public function returns a result: `anyhow::Result` in Rust with the field, the slot or the file named in the
message; a status code (null or -1) with `fh_last_error()` in C, where every exported function also turns a panic
into that error instead of unwinding into the host; an exception with the same message in JavaScript (a rejected
promise for the asynchronous calls). A bad option, slot, model or frame is an error, never a crash; device loss
surfaces from the next call, and the host drops the engine and creates a new one. Open: typed error variants with
codes across the three surfaces.

## The surfaces

| Concept | Rust | C | JavaScript |
|---|---|---|---|
| device | `Engine::new(&EngineOptions)` / `with_device` | `fh_engine_create(&FhEngineOptions)` | `await FhEngine.create(opts)` |
| model | `Model::from_dir` / `from_bytes` | `fh_model_load(dir)` / `fh_model_from_bytes(...)` | `engine.loadModel(graph, manifest, weights)` |
| session | `Session::new(&engine, &model, &SessionOptions)` | `fh_session_create(engine, model, &FhSessionOptions)` | `engine.session(model, opts)` |
| slots | `add_input` / `add_output`, `input_texture`, `add_input_texture` | `fh_session_add_input` ..., `fh_session_import_d3d11` | `session.addInput()` returns the index, `inputTexture(slot)` the `GPUTexture` |
| core calls | `pair`, `mid`, `generate`, `write_rgba`, `write_mask`, `read_rgba` | the same names with `fh_session_`; `read_rgba` fills a caller buffer | the same names in camelCase; `readRgba()` returns a promise |
| stream | `Stream::open(&engine, &model, &SessionOptions, &StreamOptions)`, `push`, `generate`, `reset` | `fh_stream_*` | `engine.stream(model, sessionOpts, streamOpts)` |
| pacer | `Pacer::new(&PacerOptions)`, `display_tick`, ... | `fh_pacer_*`, `Tick` as a struct out-parameter | `new FhPacer(opts)`, `pacer.tick(now)` returns `{ present, generate, factor }` |
| report | `report()` on each object | `fh_*_report_json(buf, cap)` | `.report()` returns the object |
| options | structs with `Default` | `#[repr(C)]` structs with `fh_*_options_default()` | plain objects, every field optional, camelCase or snake_case, unknown keys rejected with the valid list |
| errors | `Result` | status codes + `fh_last_error` | exceptions |

The Pacer's parity is tested on the recorded traces: the C test gets the Rust report byte for byte, the browser the
same numbers as `flyhigh-dev replay --json`. The header `include/flyhigh.h` is written by hand; the Python (ctypes)
and C++ (header only) bindings follow it. Versioning is in [CONTRIBUTING.md](CONTRIBUTING.md).

## The network

FlyHighFG (776 k parameters, f16) takes two frames A and B and a phase t and predicts the frame at t. The graph is
data (`models/graph.json`, see [docs/model-format.md](docs/model-format.md)); the engine walks it. Its stages:

1. **Encoders** on the frames downscaled 4x: features at 1/8 and 1/16 of the input.
2. **Coarse flow** at 1/16 from the features and a luma cost volume (radius 3): bidirectional flow and visibility.
3. **Fine iterations** at 1/8, `k` times with shared weights: warp the features and images with the current flow,
   a cost volume (radius 2), refine flow, visibility and a hidden state.
4. *(pair / mid boundary: everything above depends only on A and B, everything below on t)*
5. **FiLM(t)**: a tiny MLP turns t into a modulation of the hidden state; flow and visibility are moved to time t.
6. **Convex upsample** to 1/4, a **refine** head at 1/4 that predicts a residual.
7. **Render** at full resolution: warp A and B with the upsampled flow (bicubic or bilinear sampling), blend by
   visibility, add the residual.

Frames enter in BGR order (the training order), padded to multiples of 32 by edge replication.

**Internal size.** Steps 1 to 6 run on the frames downscaled to the internal size; only the render runs at the frame
size, with the flow rescaled. The cost of the network is then nearly independent of the output resolution.

## Static plans

When a session is built, `net::build` walks the graph once and records a plan: a list of compute dispatches with every
buffer, uniform and bind group created up front. A frame replays the plan into one command buffer, in one compute
pass, with one submit. Building also simplifies the graph:

- slice / scale / neg are views (a channel offset and a scale carried by the reader), with no dispatch;
- conv + add becomes a conv with a fused residual; blend weights + 2 multiplies + add is one `blendmix`; concat is
  one `chcopy` dispatch;
- outputs nobody reads (oracle-only heads) are not computed;
- tensors share buffers through an arena: a buffer is reused once the last reader of its tensor is recorded (about
  half the memory of a plan without reuse). Kernels therefore never assume zeroed memory.

**Pair and mid plans.** The plan splits at the pair / mid boundary. The pair plan runs once per source frame; the mid
plan runs per generated frame. t lives in a few uniform buffers (the `mul_t` scales and the FiLM vector) that are
rewritten before a mid, so any t works without a rebuild. With `fast_render` and `linear_t`, the pair also runs the
t-dependent stage at t = 0.5 and each mid is one render step whose motion is scaled linearly. Pipelines compile on
first use; `warm_up` compiles a session's kernels before the first real frame.

**Golden plans.** `tests/golden_plans.rs` dumps the plans of a fixed set of sessions (steps, bindings, parameters) and
compares them with the files in `tests/golden/`; a deliberate change re-records them. The plan structure is therefore
a regression guard of its own, beside the pixel parity.

## Conv kernels

Convs are 3x3 implicit GEMMs, f16 in and out, f32 accumulation, bias + PReLU (+ residual) fused. Candidates:

- tensor-core kernels (WGSL cooperative matrix, native Vulkan on NVIDIA): `direct` (input pre-shifted by `xshift3`,
  no gather) and `tc` (gathering);
- portable FMA kernels, everywhere including the browser: `tiled` (4x4 register block, several tile shapes),
  `tiled8` (8x4), `tiled88` (8x8), each optionally split along K (`splitk_reduce`) when a layer has too few output
  tiles to fill the GPU.

With autotune, each conv shape of the plan is timed with every applicable candidate (GPU timestamps, min of 15) and
the fastest is kept; the choices persist through the tune cache. Without it, a fixed routing applies. A plan at k = 1
is about 80 dispatches, about 100 with repair. GPUs without `shader-f16` run the same kernels rewritten to f32
(`gpu::pipeline`), the `portable-f32` class. The classes are `tc-f16`, `portable-f16` and `portable-f32`; the
autotune table is per class and persists through `EngineOptions::tune_cache` natively or as text the host stores in
the browser (`engine.autotune`, the `tune` session option).

## Repair (GMHS)

The network matches at 1/8 to 1/16 of its input, where a fence or a window grid under a pan is ambiguous: the true
motion and the motion plus or minus one period look the same. At the end of the pair plan the repair stage fits a
robust affine motion to the flow, scores the network's flow and the affine hypothesis photometrically on a coarse
grid, and replaces the flow where the hypothesis explains the frames clearly better, jumps by a real distance and
the texture is not flat. Level 2 drops the jump requirement (better on drawn content, worse on natural video).

## Static protection and the mask

Pixels unchanged for `frames` source pairs (a HUD, chat, burnt-in subtitles) are taken from the source frames instead of
being warped with the motion behind them: an age pass in the pair stage counts unchanged pairs per pixel, the render
avoids warp sources on protected pixels, and an apply pass after the render mixes them toward the sources' value with a
two-pair ramp. Unchanged is not enough on its own: after the camera stood still every pixel carries a high age, and in
the first moving pair pixels of the moving picture that match by coincidence would be held as overlays, rendered as the
unwarped average of the two frames (a half-step ghost of the whole scene, worst with a loose tolerance). The age pass
therefore also counts the pairs in which a pixel stayed unchanged while something within 32 px of it changed, and
protects only from the second such pair: an overlay over a moving scene does that every pair, a coincidence does not
repeat. A pixel whose 3 x 3 neighbourhood is one flat colour in both frames counts as changed: it carries no evidence of
standing still, and in a panned flat fill (anime) the fill between where a thin line is in the two frames reads
unchanged in both, so holding it erased the in-between's line there. An overlay has an edge of its own (a crosshair,
text, a panel's border). A flat wall whose surroundings never change is never held; the network renders it as it is. The
host's protection mask (`protection_mask`, `write_mask`) enters the same age pass as age 255, so it works with or
without the automatic detection (`shaders/keep_static*.wgsl`).

## Camera warp

Interpolation shows a frame at least part of a source interval after it arrived, and the picture of a streamed or
captured game is already late when it arrives (network, encoder, the game's own frame). A player feels that delay as
a mouse that drags. The camera warp answers the mouse at once: the frame about to be presented is turned by the
camera motion the mouse made since the moment the frame shows. Only the camera is brought up to date; what moves in
the world (players, projectiles) stays where the frame had it.

The host passes the mouse counts it sends to the game (`Stream::mouse`, with times on the clock of `push`'s
timestamps, so those must be arrival times in host milliseconds) and whether the mouse is captured
(`set_mouse_captured`: released means a cursor, a menu). Nothing else: whether the mouse turns a camera at all, how
far and how late is measured.

- **Measuring the picture.** Each pushed source frame's global shift against the previous one is searched on a
  three-level pyramid of cell grids (`warp_cells`, `warp_shift`, `warp_refine`, `warp_best`): every candidate on the
  coarsest (+-40 degrees of yaw, +-20 of pitch), a few cells around the result on each finer one. The grids are even
  in angle, not in pixels: at a 106 degree view the edges of the frame move nearly three times as far as the centre
  when the camera turns, so a flat grid mistakes a turn for a larger one; on the angular grid a turn is the same
  shift everywhere (yaw exactly, pitch as cos(azimuth) of it, which the refinement models). The result is in pixels
  at the centre, with a sub-cell parabola; on synthetic turns it is within 0.1 px for yaw and 0.2 px for pitch. A
  flat or changing picture has no clear minimum and gives no shift. 0.07 ms per source frame at 1080p (RTX 4060 Ti),
  read back 16 bytes a frame or so later; nothing waits on it.
- **What is locked to the screen.** A weapon in hand, the HUD and a cockpit do not turn with the camera. Per 8 x 8
  block, after each pair that turned by 4 px or more, a block that standing still explains better than the turn is
  screen-locked (`warp_lock`); the search leaves those blocks out (otherwise they pull small turns towards zero) and
  the render keeps their pixels.
- **Calibration** (`warp::InputCalibrator`, pure logic). The shift between two frames is explained by the mouse
  counts of the same interval moved back by the picture's lag. Over the last two seconds the calibrator searches the
  lag (0 to 200 ms in 2 ms steps) whose counts explain the shifts best and fits pixels per count through the origin,
  refitted without outliers (a cut, a mismeasured pair), every 100 ms. The warp turns on when the correlation is above
  0.9 and the mouse is captured, and off when it falls below 0.6, at once when the host releases the mouse, and after
  two frames in which the mouse moved and the picture did not (a menu the host did not report: up to about the lag
  plus three frames of a wrong turn, then off).
- **The render** (`warp_render`). Each output pixel's ray is turned back by the yaw and pitch of the turn, `shift / f`
  radians (the gain is linear in counts, as a game's camera is), and the frame sampled there. The strip the camera
  turned towards has no picture: it takes a blur of the frame's edge, which on recorded play looked better than a
  stretched or a stale edge. Screen-locked blocks keep their own pixels, softened over a block. 0.06 ms at 1080p.

On recorded CS2 play at 60 fps, while the camera turned more than a degree per frame, a frame turned by the true
camera motion of one interval is 3.2 dB closer to the next frame than the frame itself (18.5 to 21.7 dB), with the
locked mask and the blurred edge 3.3 dB over the whole frame.

Limits. Rotation only: strafing and moving forward are not turned (they change the picture by depth). The field of
view is assumed (`WarpOptions::hfov_deg`, 106 degrees, the default of most shooters); another view gets the centre
right and the edges a little off. A game with mouse acceleration fits a single gain worse and stays off when the fit
does.

## Where the network fails: the render's last decisions

The network describes the in-between as two flows and a blend; where its motion is wrong, that blend of two misplaced
copies is what the viewer sees: a milky fog with double outlines on a flick, a HUD smeared with the world moving
under it. Realtime sessions and streams therefore end the exact render with two decisions the network cannot make on
its 8 x 8 px flow grid:

- **Still test** (render.wgsl, render_tex.wgsl): per pixel, "it stayed" (the two source frames agree in place, over
  3 x 3) against "it moved with the flow" (the two warped frames agree). A still overlay wins by far and keeps the
  still pixels; only nearly identical pixels count as still, so a world the network got wrong is not frozen into a
  blend.
- **Nearest-source regions** (near_d, near_k, near_t, near_apply): where the warped frames disagree over a region
  (about 28 px, so regions decide together), the in-between there is the nearer source frame alone, warped by the
  network's own flow. Which frame is nearer is decided once per frame from the flows (near_t), so neighbouring regions
  never come from different frames. On a flick the viewer sees sharp frames and a short hold instead of fog; a
  per-pixel choice between the frames was tried and read as patches.

Both are measured with the local contrast an in-between keeps against the real frame, not PSNR or LPIPS, which reward
the blend that reads as fog. Video mode keeps the plain render. `FH_STILL=0` and `FH_NEAR=0` turn them off for
comparisons.

## Cadence (video mode)

Anime often holds each drawing for two or more frames while the camera pans on every frame. Interpolating such a
pair moves the background smoothly but makes the character jump. In video mode the generator keeps three pair
contexts (previous, current, next) and per-pair statistics from the repair maps: how much of the frame moves against
the global motion (an element jump), how much changed at all, and whether the pair is a scene cut. From the recent
hold lengths it places each drawing on a uniform path and renders the element from the pair that contains its
transition, at a re-timed phase, composited over the background by a mask (or the whole frame on a still shot). The
real slot of an interval may then be re-synthesised too. Where the transition pair's model could not match the frames
(a disocclusion: an arm appearing from behind the body) its render is a guess, so those cells, dilated and softened,
keep the plain interpolation of the current interval, which shows the real drawing (`gm_trust.wgsl`). The price that
remains: elsewhere every presented frame is a generated in-between, so big drawing steps show the generator's softness
more often than plain interpolation does. Hosts offer it as a choice.

## Browser specifics

- WebGPU exposes no tensor cores and no build-time autotune (timing is asynchronous): the autotune runs explicitly
  and its table is persisted by the host.
- Per-frame objects are cached per slot: in the browser every uniform buffer and bind group is a garbage-collected JS
  object, and churning them every frame costs frame-time stability. The cache is keyed by slot id, so one id must
  always stand for one texture view.
- The engine creates the `GPUDevice` (or adopts the page's); hosts present, sample and copy on the same device and use
  the slot textures directly.
- Kernels compile on first use, in the browser process that also composites the page, with a shader cache per site:
  a host that opens a stream while a video plays precompiles the kernels it recorded before (`compiledKernels`,
  `precompile`) a few per frame, or the video stalls for about a second on every new site.
- Firefox needs three host rules, all in the adapter: presents at least 7 ms apart, kernels warmed up before
  playback, a submit every millisecond while a readback is pending ([docs/platforms.md](docs/platforms.md)).

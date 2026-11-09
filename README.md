<div align="center">

![FlyHigh](assets/banner.svg)

**One frame-generation engine, one set of WGSL shaders.<br>Native (Vulkan, DX12, Metal) and browser (WebGPU), on `wgpu`.**

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-3b82f6?style=flat-square)](LICENSE)
[![Weights: CC BY-NC 4.0](https://img.shields.io/badge/weights-CC%20BY--NC%204.0-7c3aed?style=flat-square)](models/WEIGHTS_LICENSE.md)
[![Releases](https://img.shields.io/badge/releases-page-444?style=flat-square)](https://github.com/getflyhigh/flyhigh-engine/releases)
[![Uses FlyHigh](https://img.shields.io/badge/uses-FlyHigh-19c37d?style=flat-square)](docs/projects.md)
[![Ko-fi](https://img.shields.io/badge/support-ko--fi-ff5e5b?style=flat-square)](https://ko-fi.com/monzikxd)
[![Compute by CloudRift](assets/cloudrift.svg)](https://cloudrift.ai)

[**Documentation**](docs/README.md) / [Integration guide](docs/integration.md) /
[Options](docs/options.md) / [Architecture](ARCHITECTURE.md) /
[Contributing](CONTRIBUTING.md)

</div>

---

FlyHigh Engine generates new frames between the frames of a game, a video or a capture, on the GPU, in real time.
It runs the FlyHighFG network and is written in Rust and WGSL on top of `wgpu`, so the same engine runs in a native
application and in a web page.

<div align="center">

![Source frames go into the FlyHigh Engine, which inserts generated frames between them](assets/pipeline.svg)

</div>

## What you get

- Real time: about 1.6 ms of GPU time per source frame plus 0.8 ms per generated frame at 1080p on an RTX 4060 Ti,
  from 720p to 4K.
- Small: 776 k parameters, 1.5 MB of weights, a 1 MB wasm module. Everything a browser downloads is under 3 MB.
- One Rust crate and one set of WGSL kernels: Vulkan, DX12 and Metal natively, WebGPU in Chrome, Edge and Firefox.
- A GPU-only path: frames go from capture or decode to display without leaving the GPU.
- The whole product in the engine, as layers: the Core generates a frame between two frames, the Stream turns
  timestamped frames into intervals (scene cuts, held frames, anime cadence), the Pacer decides what to present on
  each display tick and what to generate for the next ones. Hosts stay thin.
- Every heuristic is an option with a documented default and a field in the engine's report
  ([options](docs/options.md), generated from the code).
- Keeps fences and window grids steady during camera pans, protects HUDs and subtitles, re-times anime drawn on twos.
- Rust, C and JavaScript APIs, with Python and C++ bindings over the C ABI.

## Quick start

Every API has the same shape: an **engine** (the GPU device), a **model**, and a **session** for one frame size that
takes two frames and gives the frame at any phase between them. The Stream and the Pacer sit on top for real playback.

<details open>
<summary><b>Rust</b></summary>

```toml
[dependencies]
flyhigh-engine = { git = "https://github.com/getflyhigh/flyhigh-engine" }
```

```rust
use flyhigh_engine::{Engine, EngineOptions, Model, Session, SessionOptions};

let engine = Engine::new(&EngineOptions::default())?;              // the device, once
let model = Model::from_dir(std::path::Path::new("models"))?;      // weights and graph
let mut s = Session::new(&engine, &model, &SessionOptions::new(1920, 1080))?;

let (a, b) = (s.add_input(), s.add_input());                      // rgba8 textures the session owns
let out = s.add_output();
s.write_rgba(a, &frame0)?;                                        // or write into s.input_texture(a) yourself
s.write_rgba(b, &frame1)?;
s.pair(a, b)?;                                                    // the motion of the interval, once per pair
s.mid(0.5, out)?;                                                 // the frame at phase 0.5, once per generated frame
let rgba = s.read_rgba(out)?.block(engine.gpu())?;                // or present s.output_texture(out)
```

[Rust guide](docs/api-rust.md): the Stream, the Pacer, the reference player in `examples/player`.

</details>

<details>
<summary><b>C, C++, Python</b></summary>

`cargo build --release` builds `flyhigh.dll` / `libflyhigh.so` / `libflyhigh.dylib` (crate `flyhigh-capi`); the header
is `crates/flyhigh-capi/include/flyhigh.h`. C++ and Python call the same functions through `bindings/`.

```c
#include "flyhigh.h"

FhEngineOptions eo = fh_engine_options_default();
FhEngine* engine = fh_engine_create(&eo);                        // the device, once
FhModel* model = fh_model_load("models");                        // graph.json, flyhighfg.json, flyhighfg.bin
FhSessionOptions so = fh_session_options_default(1920, 1080);
FhSession* s = fh_session_create(engine, model, &so);

int a = fh_session_add_input(s), b = fh_session_add_input(s), out = fh_session_add_output(s);
fh_session_write_rgba(s, a, frame0, 1920 * 1080 * 4);
fh_session_write_rgba(s, b, frame1, 1920 * 1080 * 4);
fh_session_pair(s, a, b);
fh_session_mid(s, 0.5f, out);
fh_session_read_rgba(s, out, rgba, 1920 * 1080 * 4);
```

[C guide](docs/api-c.md), [bindings](bindings/README.md), the C example in `examples/c`.

</details>

<details>
<summary><b>Browser (WebGPU)</b></summary>

Serve `examples/web/pkg/` (built by `sh tools/build_web.sh`) and `models/` with your page.

```js
import init, * as fh from './pkg/flyhigh_web.js';

await init();
const engine = await fh.FhEngine.create({});                      // the WebGPU device
const model = engine.loadModel(graphText, manifestText, weightBytes);
const s = engine.session(model, { width: 1920, height: 1080 });

const [a, b] = [s.addInput(), s.addInput()], out = s.addOutput();  // rgba8 textures the session owns
s.pushVideo(a, videoA); s.pushVideo(b, videoB);                    // or write into s.inputTexture(a) yourself
s.pair(a, b);
s.mid(0.5, out);
present(s.outputTexture(out));                                     // or: const rgba = await s.readRgba(out);
```

[JavaScript guide](docs/api-js.md): `FhStream` and `FhPacer` for a video element, the reference web host in
`examples/web`.

</details>

Then read the [integration guide](docs/integration.md): presenting and pacing, seeks, scene cuts, resizes.

## Platforms

| | Platform | Backend | Status |
|---|---|---|---|
| Browser | Chrome, Edge 120+ | WebGPU | Supported |
| | Firefox 141+ (Windows) | WebGPU | Supported |
| | Safari | WebGPU | Untested |
| Native | Windows | Vulkan | Experimental |
| | Windows | DX12 | Experimental (the DXC compiler ships in the Windows archive) |
| | Linux | Vulkan | Experimental |
| | macOS | Metal | Experimental (CI only) |

The browser is the primary target today: Chrome and Firefox are tested on every change through the reference web
host. Native hosts work and are measured, but the native pacing for games is still being designed, so they stay
experimental until 1.0. The verified matrix with dated numbers: [platforms](docs/platforms.md).

Runs the f16 kernels on GPUs with 16-bit float shaders (`shader-f16`) and the f32 kernels elsewhere; uses tensor
cores on NVIDIA GPUs where the driver exposes them.

## Performance

GPU time with the default settings on an RTX 4060 Ti (Vulkan, tensor-core kernels), measured with GPU timestamps:

| Output | Motion, per source frame | Synthesis, per generated frame | 2x, per generated frame |
|---|---|---|---|
| 1280x720 | 1.49 ms | 0.59 ms | 2.08 ms |
| 1920x1080 | 1.61 ms | 0.81 ms | 2.41 ms |
| 2560x1440 | 1.81 ms | 1.09 ms | 2.90 ms |
| 3840x2160 | 2.43 ms | 1.91 ms | 4.34 ms |

The network runs at about 480 lines whatever the output size, so the motion part grows little with resolution, and
it runs once per source frame: 3x and 4x cost less per generated frame than 2x. Every setting, other GPUs and how to
measure: [performance](docs/performance.md). The numbers are guarded by a baseline check ([bench](bench/README.md)).

## Build from source

```
cargo build --release                                        # engine, C library, CLI, example player
target/release/flyhigh-dev                                   # kernel self-tests and parity against the PyTorch golden
target/release/flyhigh-dev frames A.png B.png out.png        # one generated frame between two images
target/release/flyhigh-dev play in.mp4 out.mp4 --factor 2    # any video through the API, via ffmpeg
target/release/flyhigh-player in.mp4 2                       # windowed playback with frame generation
sh tools/build_web.sh                                        # browser module into examples/web/pkg (wasm-bindgen 0.2.129)
```

Tests, CLI tools, developer settings: [development](docs/development.md). How the engine is put together:
[ARCHITECTURE.md](ARCHITECTURE.md). How to contribute: [CONTRIBUTING.md](CONTRIBUTING.md).

## Repository

| Path | What is there |
|------|---------------|
| `crates/flyhigh-engine/` | the engine library: Core, Stream, Pacer, kernels, plans |
| `crates/flyhigh-options/` | the `Options` derive: one declaration per option, the reference generated from it |
| `crates/flyhigh-capi/` | the C ABI (`include/flyhigh.h`, `libflyhigh`) |
| `crates/flyhigh-web/` | the wasm-bindgen module for browsers |
| `crates/flyhigh-cli/` | developer tools (`flyhigh-dev`): self-tests, quality regression, timings, the options reference |
| `examples/player/` | the reference native host: a window playing any ffmpeg input through the Stream and the Pacer |
| `examples/web/` | the reference web host: a video element to a canvas through the Stream and the Pacer |
| `examples/c/` | a C host of the C ABI, with a `CMakeLists.txt` |
| `bindings/` | tier-2 bindings over the C ABI: Python (ctypes) and C++ (header only) |
| `shaders/` | all WGSL kernels, one copy for every target |
| `models/` | the model: `graph.json`, the `flyhighfg.json` manifest, the f16 weights `flyhighfg.bin` |
| `bench/` | per-GPU performance and quality baselines |
| `docs/` | [documentation](docs/README.md) and the [decision records](docs/decisions.md) |

Status: 0.x. The Rust API follows [ARCHITECTURE.md](ARCHITECTURE.md) and may still change; the C ABI and the
JavaScript classes change additively. What changed is in the [changelog](CHANGELOG.md).

## Projects using FlyHigh

- **[Framegen](https://github.com/MONZikWasTaken/Framegen)**: a browser extension that generates frames for any
  `<video>` on a page (Chrome, Firefox); the first host written on the engine's classes alone.
- **[OpenNOW](https://github.com/OpenCloudGaming/OpenNOW)**: an open-source cloud gaming client whose frame
  generation runs on the engine: in-betweens for the game stream in the native client.

[See all projects](docs/projects.md)

## Weights

The weights ship in this repository: `models/flyhighfg.bin` (little-endian f16, 62 tensors, 776 k parameters) with
its `models/flyhighfg.json` manifest and `models/graph.json`. They are licensed separately under CC BY-NC 4.0 (see
`models/WEIGHTS_LICENSE.md`), not under the code's Apache 2.0 licence. Another checkpoint in the same format loads
the same way ([model format](docs/model-format.md)).

## Acknowledgments

<div align="center">

[![Compute provided by CloudRift](assets/cloudrift-card.svg)](https://cloudrift.ai)

The FlyHighFG models this engine runs were trained and evaluated on GPU time generously provided by
**[CloudRift](https://cloudrift.ai)**, and so were the measurements in this README. Thank you.

</div>

Built on **[wgpu](https://wgpu.rs)** and **[wasm-bindgen](https://github.com/rustwasm/wasm-bindgen)**.

## Support

The engine is free; the models behind it are trained on rented GPUs. If it is useful to you:
[ko-fi.com/monzikxd](https://ko-fi.com/monzikxd).

## License

- **Code** (the Rust crates, WGSL shaders, bindings and examples in this repository): Apache License 2.0, see
  [`LICENSE`](LICENSE). Free to use, modify and ship, commercially too.
- **Model weights** (`models/flyhighfg.bin` with its manifest and graph): dual licensed. Creative Commons
  Attribution-NonCommercial 4.0 for noncommercial use at no cost, and a commercial licence by agreement with the
  maintainers for products that ship or run the weights commercially. See
  [`models/WEIGHTS_LICENSE.md`](models/WEIGHTS_LICENSE.md). A model you train yourself is yours.
- [`NOTICE`](NOTICE) has the attribution text, the third-party dependency licences and the licences of the DirectX
  Shader Compiler binaries in the Windows archive.

# FlyHigh Engine documentation

Start with the guide for your platform, then [options.md](options.md) when you want to change a default.

| Document | For |
|---|---|
| [integration.md](integration.md) | putting frame generation into an app: which API, presenting and pacing, seeks, resizes, device loss, performance, troubleshooting |
| [options.md](options.md) | every option, its default, what it costs and when to change it |
| [api-rust.md](api-rust.md) | Rust hosts (`api::Engine`, `api::Generator`, the texture-path `GenTex`) |
| [api-c.md](api-c.md) | C, C++, C#, Python and other FFI hosts (`flyhigh.h`) |
| [api-js.md](api-js.md) | browser hosts (WebGPU, the wasm module in `examples/web/pkg/`) |
| [platforms.md](platforms.md) | the verified matrix: backends and browsers with parity, kernel class, timings and the dated notes |
| [performance.md](performance.md) | GPU time per frame at each resolution and setting, the motion / synthesis split, how to measure |
| [ARCHITECTURE.md](../ARCHITECTURE.md) | how the engine is put together: the layers and their rules, the module map, what the Core, the Stream and the Pacer do, the adapters and the host quirks, options and reports, the surfaces, the network, plans, kernels, repair, protection, cadence |
| [model-format.md](model-format.md) | the model files (`graph.json`, manifest, weight blob) and how to ship another checkpoint |
| [development.md](development.md) | building, the CLI tools, tests and parity fixtures, developer environment variables |
| [decisions.md](decisions.md) | the decision records: what was decided, when and why |
| [projects.md](projects.md) | the apps that use the engine, and how to add yours |
| [CONTRIBUTING.md](../CONTRIBUTING.md), [SECURITY.md](../SECURITY.md), [CHANGELOG.md](../CHANGELOG.md) | how to build, check and contribute; how to report a vulnerability; what changed |

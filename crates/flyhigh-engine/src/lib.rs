//! FlyHigh Engine: frame generation on wgpu and WGSL. The public layers are the Core (`Engine`, `Model`, `Session`),
//! the Stream (`stream`) and the Pacer (`pacer`); `api` is the CPU convenience over them. Underneath: the kernels as
//! plan steps (`ops`), graph to plan (`net`), the GPU context (`gpu`) and the model files (`model`). A plan is built
//! once per frame size with every buffer and bind group ready, and a frame replays it in one submit.

// the `Options` derive names this crate by its package name, so the engine's own option structs can use it
extern crate self as flyhigh_engine;

pub mod engine;
pub mod gpu;
#[cfg(all(windows, not(target_arch = "wasm32")))]
pub mod interop;
pub mod model;
pub mod net;
pub mod ops;
pub mod options;
pub mod session;
#[cfg(all(any(windows, target_os = "linux"), not(target_arch = "wasm32")))]
pub mod vulkan_interop;

// Every type and function keeps its pre-split path at the crate root.
#[cfg(not(target_arch = "wasm32"))]
pub use gpu::device::list_adapters;
pub use gpu::pipeline::{make_pipe, shader_f32};
pub use gpu::readback::{map_read, queue_done, MapWait, QueueWait};
pub use gpu::timing::TimedRun;
pub use gpu::Gpu;
pub use model::blob::Blob;
pub use model::graph::{mid_start, names};
pub use net::build::{build, build_network, build_network_ex, set_t, BuildCtx, PhaseDep};
pub use net::session::Session as PlanSession;
pub use net::session_ext::SessionExt;
pub use net::session_tex::SessionTex;
pub use ops::conv::autotune::{autotune, ConvOp, ConvShape, TuneRequest};
pub use ops::conv::candidates::{
    cand_list, cand_name, CAND_DIRECT, CAND_DIRECT64, CAND_NAMES, CAND_TC, CAND_TILED, CAND_TILED16, CAND_TILED8,
    CAND_TILED88, CAND_TILED8P, CAND_TILEDM32, CAND_TILEDN8,
};
pub use ops::conv::kernels::{ConvArgs, TiledOpts};
pub use ops::conv::tune_cache::{format_tune, parse_tune, tune_table_from, TuneKey, TuneTable};
pub use ops::params::{
    BinaryParams, ChannelCopyParams, ConvParams, EpilogueParams, ImplicitGemmParams, InterpParams, LumaParams,
    NearKParams, NearParams, PoolParams, RenderParams, RgbaParams, SadParams, Shape3Params, ShiftParams, SplitKParams,
    StaticParams, UnaryParams, WarpParams,
};
pub use ops::render::{RenderInputs, RenderOpts};
pub use ops::step::{clone_step, plan_dump, workgroups_2d, Binding, OpOutput, Step, StepMeta};
pub use ops::tensor::Tensor;
pub use options::{
    default_internal, EngineOptions, NetOpts, OptionsError, Precision, DEFAULT_REPAIR, DEFAULT_STATIC_FRAMES,
    DEFAULT_STATIC_TOLERANCE,
};

// The Core: a device, a model, the plans for one size.
pub use engine::{DeviceInfo, Engine, EngineReport, KernelClass};
pub use gpu::dxc::ShaderCompiler;
pub use model::{Model, ModelInfo};
pub use session::{Internal, RenderSampling, Session, SessionOptions, SessionReport, StaticProtection};

#[cfg(not(target_arch = "wasm32"))]
pub mod api;
pub mod gen_tex;
pub mod gmhs;
pub mod pacer;
mod pan;
pub mod stream;
pub mod warp;

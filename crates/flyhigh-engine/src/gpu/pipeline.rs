//! Shader sources to pipelines: the f32 rewrite of the f16 sources and the static-protection render variant.

use crate::Gpu;
use std::rc::Rc;

// render.wgsl with static protection (keep_static.wgsl): a warp source within 1 px of a protected pixel (age >= min_age) is
// an overlay, not the moving world, so it is not used: the other frame's source is taken alone, and when both land on
// protected pixels the pixel keeps the unwarped average of the two frames; the residual is dropped there too. The rule
// goes in at render.wgsl's "static hook", after the (soft or decisive) blend weights.
pub(crate) fn render_static_src() -> String {
    let src = include_str!("../../../../shaders/render.wgsl");
    let helper = "@group(0) @binding(7) var<storage, read> age: array<f16>;\n\
fn on_static(gx: f32, gy: f32) -> bool {\n\
    let cx = i32(round(gx)); let cy = i32(round(gy));\n\
    for (var dy = -1; dy <= 1; dy++) { for (var dx = -1; dx <= 1; dx++) {\n\
        if (f32(age[clampy(cy + dy) * pp.W + clampx(cx + dx)]) >= f32(pp.min_age)) { return true; }\n\
    } }\n\
    return false;\n\
}\n";
    let hook = "    // static hook\n";
    let rule = "    let sa = on_static(gxa, gya); let sb = on_static(gxb, gyb);\n\
    var raw = 0.0;\n\
    stillw = stillw * select(0.0, 1.0, abs(f32(age[y * pp.W + x])) >= STILL_RUN);\n\
    if (sa && sb) { rk = 0.0; raw = 1.0; } else if (sa) { rk = 0.0; bwa = 0.0; bwb = 1.0; } else if (sb) { rk = 0.0; bwa = 1.0; bwb = 0.0; }\n";
    let out_old = "        outp[(c * pp.H + y) * pp.W + x] = f16(o[c]);";
    let out_new = "        let i = (c * pp.H + y) * pp.W + x;\n\
        outp[i] = f16(clamp(mix(o[c], 0.5 * (f32(A[i]) + f32(B[i])), raw), 0.0, 1.0));";
    let anchor = "@group(0) @binding(6) var<uniform> pp: Par;\n";
    assert!(
        src.contains(anchor) && src.contains(hook) && src.contains(out_old),
        "render.wgsl changed: update render_static_src"
    );
    src.replace(anchor, &format!("{anchor}{helper}")).replace(hook, rule).replace(out_old, out_new)
}

/// The f32 variant of a shader: tensors are f16 everywhere in the sources, so the type is swapped as a whole word and
/// `enable f16` dropped; chcopy moves f16 pairs as packed u32 and becomes a `vec2<f32>` copy.
pub fn shader_f32(src: &str) -> String {
    let src = src.replace("enable f16;", "");
    let (b, mut out, mut i) = (src.as_bytes(), String::with_capacity(src.len()), 0);
    let word = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    while i < b.len() {
        if b[i..].starts_with(b"f16") && (i == 0 || !word(b[i - 1])) && (i + 3 >= b.len() || !word(b[i + 3])) {
            out.push_str("f32");
            i += 3;
        } else {
            let c = src[i..].chars().next().unwrap();
            out.push(c);
            i += c.len_utf8();
        }
    }
    if out.contains("pack2x16float(unpack2x16float(v) * s)") {
        out = out
            .replace("array<u32>", "array<vec2<f32>>")
            .replace("var v: u32;", "var v: vec2<f32>;")
            .replace("fn scl(v: u32, s: f32) -> u32 {", "fn scl(v: vec2<f32>, s: f32) -> vec2<f32> {")
            .replace("return pack2x16float(unpack2x16float(v) * s);", "return v * s;");
    }
    out
}

/// Compile one compute pipeline from WGSL; `f32` = the device runs the f32 path, so the f16 source is rewritten first.
pub fn make_pipe(device: &wgpu::Device, f32: bool, label: &str, src: &str, ep: &str) -> wgpu::ComputePipeline {
    let f32_src;
    let src = if f32 {
        f32_src = shader_f32(src);
        f32_src.as_str()
    } else {
        src
    };
    let m = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &m,
        entry_point: Some(ep),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// Every kernel name `kernel_source` answers, besides the cadence kernels in `gmhs::PIPES`.
const KERNELS: &[&str] = &[
    "conv",
    "interp",
    "avgpool",
    "warp",
    "binary",
    "unary",
    "luma",
    "grad",
    "sad",
    "convex",
    "film",
    "blend",
    "render",
    "rgba_in",
    "rgba_out",
    "chcopy",
    "blendmix",
    "keep_static",
    "render_tex",
    "tex_down",
    "keep_static_age_tex",
    "keep_static_age",
    "tex_in",
    "tex_out",
    "classify",
    "implicit2",
    "epilogue",
    "fused",
    "fused2",
    "fused4",
    "xshift",
    "fdirect",
    "fdirect32",
    "tconv",
    "tconv8",
    "tconv88",
    "tconv8p",
    "fdirect64",
    "skreduce",
    "tconv32",
    "tconv16",
    "tconvn8",
    "tconvm32",
    "render_static",
    "tex_out_bgra",
    "warp_cells",
    "warp_shift",
    "warp_refine",
    "warp_dilate",
    "warp_best",
    "warp_lock",
    "warp_render",
    "near_d",
    "near_k",
    "near_t",
    "near_apply",
    "pan_search",
    "pan_select",
    "pan_apply",
];

/// The kernel's name as the cache holds it, or None for a name that is not a kernel.
fn kernel_name(name: &str) -> Option<&'static str> {
    KERNELS.iter().copied().chain(crate::gmhs::PIPES.iter().map(|(k, _, _)| *k)).find(|k| *k == name)
}

/// The WGSL source and entry point of a kernel by its pipeline name. The narrow tiled-conv variants rewrite the tile
/// constants of `conv_tiled.wgsl`, `render_static` is generated from `render.wgsl`, `tex_out_bgra` from `tex_out.wgsl`.
/// None for a name that is not a kernel.
pub(crate) fn kernel_source(name: &str) -> Option<(String, &'static str)> {
    fn plain(src: &str, ep: &'static str) -> Option<(String, &'static str)> {
        Some((src.to_string(), ep))
    }
    const TILED: &str = include_str!("../../../../shaders/conv_tiled.wgsl");
    match name {
        "conv" => plain(include_str!("../../../../shaders/conv.wgsl"), "main"),
        "interp" => plain(include_str!("../../../../shaders/interp.wgsl"), "main"),
        "avgpool" => plain(include_str!("../../../../shaders/avgpool.wgsl"), "main"),
        "warp" => plain(include_str!("../../../../shaders/warp.wgsl"), "main"),
        "binary" => plain(include_str!("../../../../shaders/binary.wgsl"), "main"),
        "unary" => plain(include_str!("../../../../shaders/unary.wgsl"), "main"),
        "luma" => plain(include_str!("../../../../shaders/luma.wgsl"), "luma"),
        "grad" => plain(include_str!("../../../../shaders/luma.wgsl"), "grad"),
        "sad" => plain(include_str!("../../../../shaders/sad.wgsl"), "main"),
        "convex" => plain(include_str!("../../../../shaders/convex.wgsl"), "main"),
        "film" => plain(include_str!("../../../../shaders/film.wgsl"), "main"),
        "blend" => plain(include_str!("../../../../shaders/blend.wgsl"), "main"),
        "render" => plain(include_str!("../../../../shaders/render.wgsl"), "main"),
        "rgba_in" => plain(include_str!("../../../../shaders/rgba_in.wgsl"), "main"),
        "rgba_out" => plain(include_str!("../../../../shaders/rgba_out.wgsl"), "main"),
        "chcopy" => plain(include_str!("../../../../shaders/chcopy.wgsl"), "main"),
        "blendmix" => plain(include_str!("../../../../shaders/blendmix.wgsl"), "main"),
        "keep_static" => plain(include_str!("../../../../shaders/keep_static.wgsl"), "apply"),
        "render_tex" => plain(include_str!("../../../../shaders/render_tex.wgsl"), "main"),
        "tex_down" => plain(include_str!("../../../../shaders/tex_down.wgsl"), "main"),
        "keep_static_age_tex" => plain(include_str!("../../../../shaders/keep_static_age_tex.wgsl"), "main"),
        "keep_static_age" => plain(include_str!("../../../../shaders/keep_static_age.wgsl"), "age"),
        "tex_in" => plain(include_str!("../../../../shaders/tex_in.wgsl"), "main"),
        "tex_out" => plain(include_str!("../../../../shaders/tex_out.wgsl"), "main"),
        "classify" => plain(include_str!("../../../../shaders/classify.wgsl"), "main"),
        "pan_search" => plain(include_str!("../../../../shaders/pan_search.wgsl"), "main"),
        "pan_select" => plain(include_str!("../../../../shaders/pan_select.wgsl"), "main"),
        "pan_apply" => plain(include_str!("../../../../shaders/pan_apply.wgsl"), "main"),
        "implicit2" => plain(include_str!("../../../../shaders/conv_implicit2.wgsl"), "main"),
        "epilogue" => plain(include_str!("../../../../shaders/conv_epilogue.wgsl"), "main"),
        "fused" => plain(include_str!("../../../../shaders/conv_fused.wgsl"), "main"),
        "fused2" => plain(include_str!("../../../../shaders/conv_fused2.wgsl"), "main"),
        "fused4" => plain(include_str!("../../../../shaders/conv_fused4.wgsl"), "main"),
        "xshift" => plain(include_str!("../../../../shaders/xshift3.wgsl"), "main"),
        "fdirect" => plain(include_str!("../../../../shaders/conv_fused_direct.wgsl"), "main"),
        "fdirect32" => plain(include_str!("../../../../shaders/conv_fused_direct32.wgsl"), "main"),
        "tconv" => plain(include_str!("../../../../shaders/conv_tiled.wgsl"), "main"),
        "tconv8" => plain(include_str!("../../../../shaders/conv_tiled8.wgsl"), "main"),
        "tconv88" => plain(include_str!("../../../../shaders/conv_tiled88.wgsl"), "main"),
        "tconv8p" => plain(include_str!("../../../../shaders/conv_tiled8p.wgsl"), "main"),
        "fdirect64" => plain(include_str!("../../../../shaders/conv_fused_direct64.wgsl"), "main"),
        "skreduce" => plain(include_str!("../../../../shaders/splitk_reduce.wgsl"), "main"),
        "warp_cells" => plain(include_str!("../../../../shaders/warp_cells.wgsl"), "main"),
        "warp_shift" => plain(include_str!("../../../../shaders/warp_shift.wgsl"), "main"),
        "warp_refine" => plain(include_str!("../../../../shaders/warp_refine.wgsl"), "main"),
        "warp_dilate" => plain(include_str!("../../../../shaders/warp_dilate.wgsl"), "main"),
        "warp_best" => plain(include_str!("../../../../shaders/warp_best.wgsl"), "main"),
        "warp_lock" => plain(include_str!("../../../../shaders/warp_lock.wgsl"), "main"),
        "near_d" => plain(include_str!("../../../../shaders/near_d.wgsl"), "main"),
        "near_k" => plain(include_str!("../../../../shaders/near_k.wgsl"), "main"),
        "near_t" => plain(include_str!("../../../../shaders/near_t.wgsl"), "main"),
        "near_apply" => plain(include_str!("../../../../shaders/near_apply.wgsl"), "main"),
        "warp_render" => plain(include_str!("../../../../shaders/warp_render.wgsl"), "main"),
        // portable tiled conv, narrow variants for small cout: same source, different tile constants
        // (128 px x 32 ch / 256 thr, 128 x 16 / 128 thr, 128 x 8 / 64 thr) + 32 px x 64 ch / 128 thr for mid-size
        // convs that would otherwise need split-K
        "tconv32" | "tconv16" | "tconvn8" | "tconvm32" => {
            let (tx, ty) = match name {
                "tconv32" => (32u32, 8u32),
                "tconv16" => (32, 4),
                "tconvn8" => (32, 2),
                _ => (8, 16),
            };
            let anchor = "const TX: u32 = 16u; const TY: u32 = 16u;";
            assert!(TILED.contains(anchor), "conv_tiled TX/TY line not found");
            Some((TILED.replace(anchor, &format!("const TX: u32 = {tx}u; const TY: u32 = {ty}u;")), "main"))
        }
        "render_static" => Some((render_static_src(), "main")),
        "tex_out_bgra" => Some((
            include_str!("../../../../shaders/tex_out.wgsl")
                .replace("texture_storage_2d<rgba8unorm, write>", "texture_storage_2d<bgra8unorm, write>"),
            "main",
        )),
        _ => crate::gmhs::PIPES.iter().find(|(k, _, _)| *k == name).map(|(_, src, ep)| (src.to_string(), *ep)),
    }
}

impl Gpu {
    /// The compute pipeline of a kernel, compiled on first use: a session only pays for the kernels it runs. (Firefox
    /// compiles pipelines serially in its GPU process at 0.1-2 s each, so compiling all of them up front cost 14 s
    /// before the first frame.) FH_PIPE_TIMES=1 prints the compile time of each (native).
    pub fn pipeline(&self, name: &'static str) -> Rc<wgpu::ComputePipeline> {
        if let Some(p) = self.p.borrow().get(name) {
            return p.clone();
        }
        let (src, ep) = kernel_source(name).unwrap_or_else(|| panic!("unknown kernel {name}"));
        #[cfg(not(target_arch = "wasm32"))]
        let t0 = std::time::Instant::now();
        let p = Rc::new(make_pipe(&self.device, self.f32_mode, name, &src, ep));
        #[cfg(not(target_arch = "wasm32"))]
        if std::env::var_os("FH_PIPE_TIMES").is_some() {
            eprintln!("pipeline {name}: {:.1} ms", t0.elapsed().as_secs_f64() * 1000.0);
        }
        self.p.borrow_mut().insert(name, p.clone());
        p
    }

    /// The WGSL and entry point `pipeline` compiles for a kernel on this device (the f32 rewrite applied on the f32
    /// path), or None for a name that is not a kernel. A browser host can hand these to
    /// `createComputePipelineAsync` all at once: the browser compiles them in parallel off the thread that composites
    /// the page, and the engine's own creation of the same pipelines then finds them in its cache.
    pub fn kernel_wgsl(&self, name: &str) -> Option<(String, &'static str)> {
        let (src, ep) = kernel_source(kernel_name(name)?)?;
        Some((if self.f32_mode { shader_f32(&src) } else { src }, ep))
    }

    /// The kernels compiled so far on this device (`pipeline`), in no particular order.
    pub fn compiled_kernels(&self) -> Vec<&'static str> {
        self.p.borrow().keys().copied().collect()
    }

    /// Compiles the named kernels that are not compiled yet and returns how many it compiled; a name that is not a
    /// kernel is skipped, so a list recorded with another version of the engine stays usable. A browser compiles
    /// pipelines in the process that also composites the page, and a whole stream's worth at once (about a second on
    /// a site whose shader cache is cold) freezes the video playing under it: a host passes a few names per frame
    /// instead, ahead of `Stream::open`, which then finds them compiled.
    pub fn precompile(&self, names: &[&str]) -> usize {
        let mut compiled = 0;
        for k in names.iter().filter_map(|n| kernel_name(n)) {
            if !self.p.borrow().contains_key(k) {
                self.pipeline(k);
                compiled += 1;
            }
        }
        compiled
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn every_listed_kernel_has_a_source() {
        for k in super::KERNELS.iter().chain(crate::gmhs::PIPES.iter().map(|(k, _, _)| k)) {
            assert!(super::kernel_source(k).is_some(), "{k} has no source");
        }
        assert_eq!(super::kernel_name("not a kernel"), None);
        assert_eq!(super::kernel_name("classify"), Some("classify"));
    }
}

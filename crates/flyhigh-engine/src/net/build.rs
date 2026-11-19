//! Graph to plan: the build context (caches, liveness, pinning), `build`, `build_network` and the pair / mid split.

use crate::gmhs;
use crate::gpu::alloc::Workspace;
use crate::gpu::memory::Buffer;
use crate::model::graph::count_uses;
use crate::model::graph::count_uses_unrolled;
use crate::{
    mid_start, names, workgroups_2d, Blob, ConvArgs, ConvOp, ConvShape, Gpu, NetOpts, OpOutput, RenderInputs,
    RenderOpts, StaticParams, Step, Tensor, TuneTable, UnaryParams,
};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

// a buffer whose contents depend on the phase t (a `mul_t` scale uniform, the FiLM MLP output): `f(t)` gives the bytes
pub type PhaseDep = (Rc<Buffer>, Box<dyn Fn(f32) -> Vec<u8>>);
pub fn set_t(gpu: &Gpu, deps: &[PhaseDep], t: f32) {
    for (b, f) in deps {
        gpu.queue.write_buffer(b, 0, &f(t));
    }
}

type PackKey = (String, u32, u32);

#[derive(Default)]
pub(crate) struct PlanResources {
    weights: Weights,
    pub(crate) render: Option<Rc<Buffer>>,
    pub(crate) workspace: Option<Workspace>,
    pub(crate) layout_mismatch: bool,
}

#[derive(Default)]
struct Weights {
    raw: HashMap<String, Rc<Buffer>>,
    gemm: HashMap<PackKey, (Rc<Buffer>, u32, u32)>,
    tap: HashMap<PackKey, (Rc<Buffer>, u32, u32, u32)>,
}

pub struct BuildCtx<'a> {
    pub gpu: &'a Gpu,
    pub blob: &'a Blob,
    weights: Weights,
    render_target: Option<Rc<Buffer>>,
    pub k: u32,
    pub t: f32,
    pub hs: u32,
    pub ws: u32,
    pub tune: TuneTable,                   // (cin, cout, h, w, stride) -> chosen conv candidate
    pub uses: HashMap<String, usize>,      // tensor name -> number of ops reading it (filled by build)
    pub remaining: HashMap<String, usize>, // reads still to come in the UNROLLED plan (buffer liveness)
    // buffers never released to the arena (inputs, outputs, stage exports), by address; the Weak reserves the
    // address (see `Gpu::arena_ids`), otherwise a freed pinned buffer's address could pin an unrelated new one
    pub pinned: HashMap<usize, std::rc::Weak<Buffer>>,
    pub t_deps: Vec<PhaseDep>,                 // t-dependent buffers of the plan (see `set_t`)
    pub keep_all: bool,                        // FH_ALL_OUTPUTS=1: also compute oracle-only outputs
    pub cubic: bool,                           // render_pix source sampler (NetOpts::cubic)
    pub render_flags: u32,                     // render_pix mode word (NetOpts::render_flags)
    pub static_age: Option<(Rc<Buffer>, u32)>, // static protection age and protecting age (NetOpts::keep_static)
    pub discover: Option<Vec<ConvShape>>,
} // Some: record conv shapes, pick fixed kernels (autotune discovery)

impl<'a> BuildCtx<'a> {
    pub fn new(gpu: &'a Gpu, blob: &'a Blob, k: u32, t: f32, hs: u32, ws: u32) -> BuildCtx<'a> {
        BuildCtx {
            gpu,
            blob,
            weights: Weights::default(),
            render_target: None,
            k,
            t,
            hs,
            ws,
            tune: HashMap::new(),
            uses: HashMap::new(),
            remaining: HashMap::new(),
            pinned: HashMap::new(),
            t_deps: Vec::new(),
            keep_all: std::env::var_os("FH_ALL_OUTPUTS").is_some(),
            cubic: blob.render_cubic,
            render_flags: 0,
            static_age: None,
            discover: None,
        }
    }
    pub fn pin(&mut self, t: &Tensor) {
        self.pinned.insert(Rc::as_ptr(&t.buf) as usize, Rc::downgrade(&t.buf));
    }
    pub fn is_pinned(&self, b: &Rc<Buffer>) -> bool {
        self.pinned.get(&(Rc::as_ptr(b) as usize)).is_some_and(|w| w.as_ptr() == Rc::as_ptr(b))
    }
    // one op has been built: its inputs were read. Drop names whose last read this was and hand dead buffers to the arena.
    pub fn consume(&mut self, env: &mut HashMap<String, Tensor>, read: Vec<(String, Option<Tensor>)>) {
        for (name, t) in read {
            let left = self
                .remaining
                .get_mut(&name)
                .map(|r| {
                    *r = r.saturating_sub(1);
                    *r
                })
                .unwrap_or(0);
            let Some(t) = t else { continue };
            let pinned = self.is_pinned(&t.buf);
            if left == 0 && !pinned && env.get(&name).is_some_and(|e| Rc::ptr_eq(&e.buf, &t.buf)) {
                env.remove(&name);
            }
            if !pinned {
                self.gpu.release(t.buf);
            }
        }
    }
}

impl<'a> BuildCtx<'a> {
    pub fn wb(&mut self, name: &str) -> Rc<Buffer> {
        if let Some(b) = self.weights.raw.get(name) {
            return b.clone();
        }
        let b = self.gpu.buf(&self.blob.weight(name));
        self.weights.raw.insert(name.to_string(), b.clone());
        b
    }
    pub fn bbuf(&mut self, name: &str, cin: u32, cout: u32) -> (Rc<Buffer>, u32, u32) {
        let key = (name.to_owned(), cin, cout);
        if let Some(b) = self.weights.gemm.get(&key) {
            return b.clone();
        }
        let b = self.gpu.buildb(&self.blob.weight(name), cin, cout);
        self.weights.gemm.insert(key, b.clone());
        b
    }
    pub fn bbuf_tap(&mut self, name: &str, cin: u32, cout: u32) -> (Rc<Buffer>, u32, u32, u32) {
        let key = (name.to_owned(), cin, cout);
        if let Some(b) = self.weights.tap.get(&key) {
            return b.clone();
        }
        let b = self.gpu.buildb_tap(&self.blob.weight(name), cin, cout);
        self.weights.tap.insert(key, b.clone());
        b
    }
}

// walk graph.json, allocate buffers and append compute steps to `plan`.
// Graph-level simplifications applied here (all parity-neutral): slice / scale / neg / mul_t become views (no dispatch);
// ops whose outputs nobody reads are skipped (oracle-only outputs such as conf/gate/sal/state_out);
// blend_weights + 2 mul + add fuse into one `blendmix` dispatch.
pub fn build(ops: &[Value], env: &mut HashMap<String, Tensor>, plan: &mut Vec<Step>, cx: &mut BuildCtx) {
    if cx.uses.is_empty() {
        count_uses(ops, &mut cx.uses);
    }
    if cx.remaining.is_empty() {
        // top-level call: the caller's tensors (graph inputs) are never released
        count_uses_unrolled(ops, cx.k, &mut cx.remaining);
        for v in env.values() {
            cx.pin(v);
        }
    }
    let mut i = 0usize;
    while i < ops.len() {
        cx.gpu.release_scratch(); // the previous op's kernel scratch is dead once its steps are recorded
        let op = &ops[i];
        i += 1;
        let name = op["op"].as_str().unwrap();
        let ins = names(op, "in");
        let outs = names(op, "out");
        let g = cx.gpu;
        // what this op reads, captured before it runs (a carry op may rebind one of these names to its own output)
        let read: Vec<(String, Option<Tensor>)> = ins.iter().map(|n| (n.clone(), env.get(n).cloned())).collect();
        if name != "repeat"
            && name != "render_pix"
            && !cx.keep_all
            && !outs.is_empty()
            && outs.iter().all(|o| cx.uses.get(o).copied().unwrap_or(0) == 0)
        {
            cx.consume(env, read);
            continue;
        } // nobody reads it
        let t = |env: &HashMap<String, Tensor>, i: usize| env[&ins[i]].clone();
        // kernels that take plain inputs get views materialized first (one chcopy dispatch)
        let mat = |x: Tensor, plan: &mut Vec<Step>| -> Tensor {
            let (y, s) = g.mat(&x);
            plan.extend(s);
            y
        };
        // oracle outputs stay readable after the run; constants (zeros, the FiLM vector) are written once and must never
        // become somebody's scratch — a replayed plan would read the previous frame's garbage as "zeros"
        let pin_out =
            cx.keep_all || op["output"].as_bool() == Some(true) || matches!(name, "zeros" | "zeros_like" | "film_mlp");
        let (res, steps): OpOutput = match name {
            "repeat" => {
                let body: Vec<Value> = op["body"].as_array().unwrap().clone();
                for _ in 0..crate::model::graph::repeat_times(op, cx.k) {
                    build(&body, env, plan, cx);
                }
                continue;
            }
            "interpolate" => {
                let x = t(env, 0);
                let (oh, ow) = if op["size"].as_str() == Some("HW") {
                    (cx.hs, cx.ws)
                } else {
                    let s = op["scale"].as_f64().unwrap() as f32;
                    ((x.h as f32 * s).floor() as u32, (x.w as f32 * s).floor() as u32)
                };
                g.interp(&x, oh, ow, op["antialias"].as_bool().unwrap_or(false))
            }
            "conv" | "conv2d" => {
                let w = names(op, "w");
                let act = (name == "conv") as u32;
                let sh = cx.blob.shape(&w[0]);
                let cout = sh[0] as u32;
                let cin = sh[1] as u32;
                let stride = op["stride"].as_u64().unwrap_or(1) as u32;
                // every conv goes through the Tensor-Core implicit GEMM: stride and small cin are handled by the
                // kernel's gather + K padding; no-activation convs pass a slope of 1 (identity epilogue). The naive
                // `conv` kernel measured ~1% of peak and cost 45% of the frame.
                let bias = cx.wb(&w[1]);
                let slope = if act == 1 { cx.wb(&w[2]) } else { g.ones.clone() };
                let x = mat(t(env, 0), plan);
                // peephole: conv -> add(conv_out, Y) with conv_out used once and Y a plain same-shape tensor => fused residual
                let (hout, wout) = ((x.h + 2 - 3) / stride + 1, (x.w + 2 - 3) / stride + 1);
                let mut resid: Option<Tensor> = None;
                let mut fused_name: Option<String> = None;
                if i < ops.len() && ops[i]["op"] == "add" && cx.uses.get(&outs[0]).copied().unwrap_or(0) == 1 {
                    let ai = names(&ops[i], "in");
                    if ai.len() == 2 && (ai[0] == outs[0] || ai[1] == outs[0]) {
                        let other = if ai[0] == outs[0] { &ai[1] } else { &ai[0] };
                        if let Some(o) = env.get(other) {
                            if o.is_plain() && o.c == cout && o.h == hout && o.w == wout {
                                resid = Some(o.clone());
                                fused_name = Some(names(&ops[i], "out")[0].clone());
                            }
                        }
                    }
                }
                if let Some(fname) = fused_name {
                    let op = ConvOp { x: &x, weight: &w[0], bias: &bias, slope: &slope, cin, cout, stride };
                    let (y, s) = cx.conv_pick(&op, resid.as_ref());
                    let add_read: Vec<(String, Option<Tensor>)> =
                        names(&ops[i], "in").iter().map(|n| (n.clone(), env.get(n).cloned())).collect();
                    plan.extend(s);
                    if pin_out || ops[i]["output"].as_bool() == Some(true) {
                        cx.pin(&y);
                    }
                    env.insert(fname, y);
                    cx.consume(env, read);
                    cx.consume(env, add_read);
                    i += 1;
                    continue;
                }
                // stride-1 convs on 16-aligned rows: shifted-input kernel, A tile is a direct coopLoad (no gather).
                // Everything else (stride 2, odd widths) keeps the gathering kernel.
                // cout >= 64: below that the xshift3 prep (3x input writes) outweighs the conv itself
                // cout > 32 (64-wide N tile anyway) and 16-aligned rows: the 32-pixel register-blocked direct kernel
                // (two halves may be in different rows) repays its xshift3 prep; smaller heads stay on the gathering kernel
                // FH_NAIVE=1 keeps the reference kernel (no tensor cores only); otherwise per-shape autotune among the
                // applicable candidates (direct / direct64 / tc on coop devices, tiled / tiled8 everywhere)
                if !g.coop && std::env::var_os("FH_NAIVE").is_some() {
                    let wt = cx.wb(&w[0]);
                    let raw = ConvArgs {
                        x: &x,
                        weights: &wt,
                        bias: &bias,
                        slope: &slope,
                        cin,
                        cout,
                        kp: 0,
                        np: 0,
                        cinp: 0,
                        stride,
                    };
                    g.conv(&raw, act)
                } else {
                    cx.conv_pick(&ConvOp { x: &x, weight: &w[0], bias: &bias, slope: &slope, cin, cout, stride }, None)
                }
            }
            "avgpool" => {
                let x = mat(t(env, 0), plan);
                g.avgpool(&x, op["k"].as_u64().unwrap() as u32)
            }
            "warp" => g.warp(&t(env, 0), &t(env, 1)), // image and flow may be views
            "concat" => {
                let ts: Vec<Tensor> = ins.iter().map(|k| env[k].clone()).collect();
                g.concat(&ts)
            } // inputs may be views
            "slice" => g.slice(&t(env, 0), op["a"].as_u64().unwrap() as u32, op["b"].as_u64().unwrap() as u32), // view
            "scale" => (t(env, 0).scaled(op["factor"].as_f64().unwrap() as f32), vec![]), // view
            "mul_t" => {
                // materialized (not a view) so that t lives in one rewritable uniform: `set_t` retargets the plan
                let fwd = op["mode"].as_str() == Some("fwd");
                let f = if fwd { 2.0 * cx.t } else { 2.0 * (1.0 - cx.t) };
                let x = t(env, 0);
                let r = g.unary(&x, 0, f, 0.0);
                let Step::Compute { meta, .. } = &r.1[0] else { unreachable!("unary step") };
                let pb = meta.params_buffer.as_ref().expect("unary bind").clone();
                let p0 = UnaryParams { c: x.c, h: x.h, w: x.w, op: 0, a: f, b: 0.0, c0: x.c0, scale: x.scale };
                cx.t_deps.push((
                    pb,
                    Box::new(move |t: f32| {
                        let mut p = p0;
                        p.a = if fwd { 2.0 * t } else { 2.0 * (1.0 - t) };
                        bytemuck::bytes_of(&p).to_vec()
                    }),
                ));
                r
            }
            "neg" => (t(env, 0).scaled(-1.0), vec![]),
            "add" => g.binary(&t(env, 0), &t(env, 1), 0),
            "mul" => g.binary(&t(env, 0), &t(env, 1), 1),
            "zeros" => {
                let r = t(env, 0);
                g.zeros(op["ch"].as_u64().unwrap() as u32, r.h, r.w)
            }
            "zeros_like" => {
                let r = t(env, 0);
                g.zeros(r.c, r.h, r.w)
            }
            "copy" => (t(env, 0), vec![]),
            "luma" => g.luma(&t(env, 0), false),
            "luma_grad" => {
                let x = mat(t(env, 0), plan);
                g.luma(&x, true)
            }
            "sad_volume" => {
                let (a, b) = (mat(t(env, 0), plan), mat(t(env, 1), plan));
                g.sad(&a, &b, op["radius"].as_u64().unwrap() as u32)
            }
            "affine_out" => g.unary(&t(env, 0), 2, 0.0, 0.0),
            "clamp" => g.unary(&t(env, 0), 1, op["lo"].as_f64().unwrap() as f32, op["hi"].as_f64().unwrap() as f32),
            "film_mlp" => {
                let (w1, b1) = (cx.blob.weight("film.film.0.weight"), cx.blob.weight("film.film.0.bias"));
                let (w2, b2) = (cx.blob.weight("film.film.2.weight"), cx.blob.weight("film.film.2.bias"));
                let mlp = move |t: f32| -> Vec<f32> {
                    let hv: Vec<f32> = (0..64).map(|i| (w1[i] * t + b1[i]).max(0.0)).collect();
                    (0..b2.len()).map(|o| b2[o] + (0..64).map(|i| w2[o * 64 + i] * hv[i]).sum::<f32>()).collect()
                };
                let gvec = mlp(cx.t);
                let n = gvec.len() as u32;
                let buf = g.buf(&gvec);
                let f32m = g.f32_mode;
                cx.t_deps.push((
                    buf.clone(),
                    Box::new(move |t: f32| {
                        mlp(t)
                            .iter()
                            .flat_map(|&x| {
                                if f32m {
                                    x.to_le_bytes().to_vec()
                                } else {
                                    half::f16::from_f32(x).to_bits().to_le_bytes().to_vec()
                                }
                            })
                            .collect()
                    }),
                ));
                (Tensor { buf, c: n, h: 1, w: 1, c0: 0, scale: 1.0 }, vec![])
            }
            "film_modulate" => {
                let (h, gv) = (mat(t(env, 0), plan), mat(t(env, 1), plan));
                g.film(&h, &gv)
            }
            "render_pix" => {
                let (a, b, rt) = (mat(t(env, 0), plan), mat(t(env, 1), plan), mat(t(env, 4), plan));
                let (ftq, vtq) = (t(env, 2), t(env, 3));
                g.render_ex_to(
                    &RenderInputs { a: &a, b: &b, flow: &ftq, vis: &vtq, residual: &rt },
                    &RenderOpts {
                        fsx: ftq.scale,
                        fsy: ftq.scale,
                        cubic: cx.cubic,
                        flags: cx.render_flags,
                        static_age: cx.static_age.as_ref().map(|(b, n)| (&**b, *n)),
                    },
                    cx.render_target.as_ref().filter(|_| outs[0] == "render"),
                )
            } // flow_tq / vis_tq may be views
            "convex_upsample" => {
                let (h, x) = (mat(t(env, 0), plan), mat(t(env, 1), plan));
                let ub = cx.wb("up.w.bias"); // no activation -> ones slope
                                             // cout 36: the gathering TC kernel wins here (the direct kernel's xshift3 prep would dominate)
                let (wt36, mut s1) = if !g.coop && std::env::var_os("FH_NAIVE").is_some() {
                    let uw = cx.wb("up.w.weight");
                    let raw = ConvArgs {
                        x: &h,
                        weights: &uw,
                        bias: &ub,
                        slope: &ub,
                        cin: h.c,
                        cout: 36,
                        kp: 0,
                        np: 0,
                        cinp: 0,
                        stride: 1,
                    };
                    g.conv(&raw, 0)
                } else {
                    let ones = g.ones.clone();
                    cx.conv_pick(
                        &ConvOp {
                            x: &h,
                            weight: "up.w.weight",
                            bias: &ub,
                            slope: &ones,
                            cin: h.c,
                            cout: 36,
                            stride: 1,
                        },
                        None,
                    )
                };
                let (o, s2) = g.convex(&wt36, &x);
                s1.extend(s2);
                (o, s1)
            }
            "blend_weights" => {
                // peephole: blend_weights -> (bwA, bwB); mul X,bwA -> t1; mul Y,bwB -> t2; add t1,t2 -> out  ==> one blendmix
                let fused = if i + 3 <= ops.len() {
                    let (m1, m2, ad) = (&ops[i], &ops[i + 1], &ops[i + 2]);
                    let (i1, i2, ia) = (names(m1, "in"), names(m2, "in"), names(ad, "in"));
                    let (o1, o2) = (names(m1, "out"), names(m2, "out"));
                    let only = |n: &str| cx.uses.get(n).copied().unwrap_or(0) == 1;
                    if m1["op"] == "mul"
                        && m2["op"] == "mul"
                        && ad["op"] == "add"
                        && i1.len() == 2
                        && i2.len() == 2
                        && ia.len() == 2
                        && i1[1] == outs[0]
                        && i2[1] == outs[1]
                        && ia[0] == o1[0]
                        && ia[1] == o2[0]
                        && only(&outs[0])
                        && only(&outs[1])
                        && only(&o1[0])
                        && only(&o2[0])
                    {
                        let fused_read: Vec<(String, Option<Tensor>)> =
                            [&i1[0], &i2[0]].iter().map(|n| ((*n).clone(), env.get(*n).cloned())).collect();
                        let (wa, wb) = (mat(env[&i1[0]].clone(), plan), mat(env[&i2[0]].clone(), plan));
                        let (o, s) = g.blendmix(&wa, &wb, &t(env, 0));
                        plan.extend(s);
                        if ad["output"].as_bool() == Some(true) {
                            cx.pin(&o);
                        }
                        env.insert(names(ad, "out")[0].clone(), o);
                        cx.consume(env, fused_read);
                        i += 3;
                        true
                    } else {
                        false
                    }
                } else {
                    false
                };
                if fused {
                    cx.consume(env, read);
                    continue;
                }
                let (bw, s) = g.blend(&t(env, 0));
                if pin_out {
                    cx.pin(&bw);
                }
                env.insert(outs[0].clone(), bw.view(0, 1));
                env.insert(outs[1].clone(), bw.view(1, 2));
                plan.extend(s);
                cx.consume(env, read);
                continue;
            }
            other => {
                eprintln!("unhandled op: {}", other);
                continue;
            }
        };
        plan.extend(steps);
        if pin_out {
            cx.pin(&res);
        }
        env.insert(outs[0].clone(), res);
        cx.consume(env, read);
    }
    cx.gpu.release_scratch();
}

// The network between two padded input tensors (a, b: [3][hp][wp]) and the rendered frame, with the optional internal
// resolution (iw x ih; the frame size = full resolution). Appends to `plan`, returns the render tensor and the env.
// `tune`: a pre-computed kernel choice table (browser: from `autotune`), otherwise the autotuner runs during the build.
// `net`: repair level, kept maps and render sampler of this session.
#[allow(clippy::too_many_arguments)] // the Core `Session` is the public constructor; the tools and the plan tests call this
pub fn build_network(
    gpu: &Gpu,
    blob: &Blob,
    ops: &[Value],
    a: &Tensor,
    b: &Tensor,
    k: u32,
    t: f32,
    iw: u32,
    ih: u32,
    plan: &mut Vec<Step>,
    tune: Option<&TuneTable>,
    net: &NetOpts,
) -> (Tensor, HashMap<String, Tensor>) {
    let (r, env, _deps) = build_network_ex(gpu, blob, ops, a, b, k, t, iw, ih, plan, None, tune, net);
    (r, env)
}
// Same, optionally split into two plans: `plan` gets the t-independent pair stage (encoders, coarse flow, K fine
// iterations), `plan_mid` the per-generated-frame stage (FiLM(t), convex upsample, refine, render). The mid plan is
// re-run for every t: the returned t-dependent buffers are rewritten by `set_t` (no rebuild). Buffers come from the arena
// (released after their last reader is built); the pair stage's live tensors are pinned so mids never overwrite them.
#[allow(clippy::too_many_arguments)] // the Core `Session` is the public constructor; the tools and the plan tests call this
pub fn build_network_ex(
    gpu: &Gpu,
    blob: &Blob,
    ops: &[Value],
    a: &Tensor,
    b: &Tensor,
    k: u32,
    t: f32,
    iw: u32,
    ih: u32,
    plan: &mut Vec<Step>,
    plan_mid: Option<&mut Vec<Step>>,
    tune: Option<&TuneTable>,
    net: &NetOpts,
) -> (Tensor, HashMap<String, Tensor>, Vec<PhaseDep>) {
    build_network_with(gpu, blob, ops, a, b, k, t, iw, ih, plan, plan_mid, tune, net, &mut PlanResources::default())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_network_with(
    gpu: &Gpu,
    blob: &Blob,
    ops: &[Value],
    a: &Tensor,
    b: &Tensor,
    k: u32,
    t: f32,
    iw: u32,
    ih: u32,
    plan: &mut Vec<Step>,
    plan_mid: Option<&mut Vec<Step>>,
    tune: Option<&TuneTable>,
    net: &NetOpts,
    resources: &mut PlanResources,
) -> (Tensor, HashMap<String, Tensor>, Vec<PhaseDep>) {
    let (hp, wp) = (a.h, a.w);
    let (ihp, iwp) = (ih.div_ceil(32) * 32, iw.div_ceil(32) * 32);
    let internal = (iwp, ihp) != (wp, hp);
    // cascade: as many of the requested levels as the network size divides into (the coarsest level needs /16)
    let mut levels = if net.fast_render { 1 } else { net.cascade.max(1) };
    while levels > 1 && (ihp % (16 << (levels - 1)) != 0 || iwp % (16 << (levels - 1)) != 0) {
        levels -= 1;
    }
    let cascaded = (levels > 1).then(|| crate::model::graph::cascade(ops, levels, &net.cascade_iters)).flatten();
    let ops: &[Value] = cascaded.as_deref().unwrap_or(ops);
    let arena_was = if std::env::var_os("FH_NO_ARENA").is_some() {
        gpu.arena.borrow_mut().take()
    } else {
        gpu.arena.borrow_mut().replace(Vec::new())
    };
    // the ids are the addresses of this build's arena buffers; a dropped buffer's address is reused by a later session's
    // buffers, so stale ids let a buffer that never came from the arena enter the free list (which also made the plan's
    // buffer assignment depend on allocator luck)
    gpu.arena_ids.borrow_mut().clear();
    let workspace_was = gpu.workspace.replace(resources.workspace.take());
    let mut env: HashMap<String, Tensor> = HashMap::new();
    let mut cx = BuildCtx::new(gpu, blob, k, t, ihp, iwp);
    cx.weights = std::mem::take(&mut resources.weights);
    cx.render_target = resources.render.clone();
    cx.cubic = net.cubic;
    cx.render_flags = net.render_flags();
    cx.pin(a);
    cx.pin(b);
    // static protection: the per-pixel age of static pairs, a fresh zeroed buffer outside the arena (it persists)
    // fast render: the session renders from its source textures; at an internal size below the frame it also fills the
    // network inputs straight from them (tex_down), so the full-resolution tensors a / b are never used
    let fast = net.fast_render && internal;
    let static_frames = net.protecting_age();
    let static_p = StaticParams::new(net, wp, hp, wp, hp);
    let static_age = net.protects().then(|| gpu.buf(&vec![0.0; (wp * hp) as usize]));
    // the age pass's own counters (keep_static_age.wgsl): unchanged pairs, and those with the surroundings moving
    let static_state = static_age.as_ref().map(|_| gpu.buf(&vec![0.0; (3 * wp * hp) as usize]));
    if let Some(st) = &static_state {
        env.insert("static:state".into(), Tensor { buf: st.clone(), c: 3, h: hp, w: wp, c0: 0, scale: 1.0 });
    }
    // the host's mask (`SessionExt::write_mask`); the zero buffer fills the binding when the session has none
    let mask = net.protection_mask.then(|| gpu.buf(&vec![0.0; (wp * hp) as usize]));
    if let Some(m) = &mask {
        env.insert("mask".into(), Tensor { buf: m.clone(), c: 1, h: hp, w: wp, c0: 0, scale: 1.0 });
    }
    cx.static_age = static_age.clone().map(|b| (b, static_frames));
    if let (Some(age), false) = (&static_age, net.fast_render) {
        plan.push(gpu.compute_step_ex(
            "keep_static_age",
            &[&a.buf, &b.buf],
            bytemuck::bytes_of(&static_p),
            age,
            &[mask.as_deref().unwrap_or(&gpu.zero), static_state.as_deref().expect("state beside the age")],
            workgroups_2d(wp, hp, 1),
        ));
    }
    // fast render: the session fills the network inputs straight from its source textures (tex_down.wgsl) and renders
    // from them too, so the full-resolution tensors a / b are never used
    if fast {
        let ai = Tensor { buf: gpu.alloc((3 * ihp * iwp) as usize), c: 3, h: ihp, w: iwp, c0: 0, scale: 1.0 };
        let bi = Tensor { buf: gpu.alloc((3 * ihp * iwp) as usize), c: 3, h: ihp, w: iwp, c0: 0, scale: 1.0 };
        cx.pin(&ai);
        cx.pin(&bi);
        env.insert("fast:ai".into(), ai.clone());
        env.insert("fast:bi".into(), bi.clone());
        env.insert("A".into(), ai);
        env.insert("B".into(), bi);
    } else if internal {
        let (ai, sa) = gpu.interp(a, ihp, iwp, true);
        let (bi, sb) = gpu.interp(b, ihp, iwp, true);
        plan.extend(sa);
        plan.extend(sb);
        env.insert("A".into(), ai);
        env.insert("B".into(), bi);
    } else {
        env.insert("A".into(), a.clone());
        env.insert("B".into(), b.clone());
    }
    match tune {
        Some(tn) => cx.tune = tn.clone(),
        None => cx.tune = gpu.tune.borrow().clone(),
    }
    count_uses(ops, &mut cx.uses); // over the FULL graph, so render_pix's inputs are not treated as dead outputs
    count_uses_unrolled(ops, k, &mut cx.remaining); // liveness over the unrolled plan
                                                    // the graph's own render_pix is the last op; in internal mode it is replaced by a full-resolution render
    let rend = ops.iter().rposition(|o| o["op"] == "render_pix");
    let (body, rop) = match (internal || net.fast_render, rend) {
        (true, Some(i)) => (&ops[..i], Some(&ops[i])),
        _ => (ops, None),
    };
    let ms = mid_start(body); // first t-dependent op: the pair / mid boundary (also where the single-plan case inserts GMHS)
    build(&body[..ms], &mut env, plan, &mut cx);
    if net.repair > 0 {
        // optional periodic-structure repair of the pair flow (see gmhs.rs); the mid ops read the rewritten buffer
        // `flow` may already be gone from env (its pair-stage readers are built); its buffer lives on in the `flowf` view
        let f = env.get("flow").cloned().or_else(|| {
            env.get("flowf").map(|v| Tensor { buf: v.buf.clone(), c: 4, h: v.h, w: v.w, c0: 0, scale: 1.0 })
        });
        match (f, env.get("A"), env.get("B")) {
            (Some(f), Some(ai), Some(bi)) if f.is_plain() => {
                let (st, kept) = gmhs::gm_steps(gpu, ai, bi, &f, wp, net.repair, net.keep_maps);
                plan.extend(st);
                if let Some(m) = kept {
                    cx.pin(&m);
                    env.insert("gm_maps".into(), m);
                } // cadence smoothing reads the maps after the pair
            }
            _ => eprintln!("gmhs: pair flow not found in the graph env (keys: {:?}); stage skipped", {
                let mut k: Vec<&String> = env.keys().collect();
                k.sort();
                k
            }),
        }
    }
    let (pair_live, pair_used): (HashSet<_>, HashSet<_>) = if gpu.workspace.borrow().is_some() {
        (
            env.values().map(|t| Rc::as_ptr(&t.buf.charge)).collect(),
            plan.iter().flat_map(Step::allocations).map(Rc::as_ptr).collect(),
        )
    } else {
        Default::default()
    };
    #[cfg(test)]
    let (retained, pair_buffers) = if net.keep_maps && std::env::var_os("FH_CADENCE_DIAGNOSTICS").is_some() {
        let retained = env
            .values()
            .map(|t| &t.buf)
            .chain([&a.buf, &b.buf])
            .chain(static_age.iter())
            .map(|b| Rc::as_ptr(&b.charge))
            .collect::<std::collections::HashSet<_>>();
        let pair = plan.iter().flat_map(Step::allocations).map(|a| (Rc::as_ptr(a), a.bytes)).collect::<HashMap<_, _>>();
        (retained, pair)
    } else {
        Default::default()
    };
    let tail: &mut Vec<Step> = match plan_mid {
        Some(pm) => {
            for v in env.values() {
                cx.pin(v);
            }
            pm
        }
        None => plan,
    }; // stage exports stay put
    build(&body[ms..], &mut env, tail, &mut cx);
    if let (Some(op), true) = (rop, net.fast_render) {
        // fast render: the session samples the source textures itself (render_tex.wgsl); export the maps it reads
        let ins = names(op, "in");
        let (rt, s) = gpu.mat(&env[&ins[4]]);
        tail.extend(s);
        for (k, t) in [("fast:ftq", env[&ins[2]].clone()), ("fast:vtq", env[&ins[3]].clone()), ("fast:rt", rt)] {
            cx.pin(&t);
            env.insert(k.into(), t);
        }
        if let Some(age) = &static_age {
            env.insert(
                "fast:age".into(),
                Tensor { buf: age.clone(), c: 1, h: hp, w: wp, c0: 0, scale: static_frames as f32 },
            );
        }
        env.insert("render".into(), a.clone()); // no full-resolution render tensor on this path
        gpu.remember_tune(&cx.tune);
        *gpu.arena.borrow_mut() = arena_was;
        gpu.arena_ids.borrow_mut().clear();
        let r = env["render"].clone();
        resources.workspace = gpu.workspace.replace(workspace_was);
        resources.weights = cx.weights;
        return (r, env, cx.t_deps);
    }
    if let Some(op) = rop {
        let ins = names(op, "in"); // A, B, flow_tq, vis_tq, R_t
        let (ftq, vtq) = (env[&ins[2]].clone(), env[&ins[3]].clone());
        let (rt, s) = gpu.mat(&env[&ins[4]]);
        tail.extend(s);
        let (fsx, fsy) = (ftq.scale * wp as f32 / iwp as f32, ftq.scale * hp as f32 / ihp as f32);
        let (o, s) = gpu.render_ex_to(
            &RenderInputs { a, b, flow: &ftq, vis: &vtq, residual: &rt },
            &RenderOpts {
                fsx,
                fsy,
                cubic: net.cubic,
                flags: net.render_flags(),
                static_age: static_age.as_deref().map(|b| (b, static_frames)),
            },
            cx.render_target.as_ref(),
        );
        tail.extend(s);
        cx.pin(&o);
        env.insert(names(op, "out")[0].clone(), o);
    }
    if let Some(age) = &static_age {
        if net.repair == 2 {
            env.insert(
                "static:age".into(),
                Tensor { buf: age.clone(), c: 1, h: hp, w: wp, c0: 0, scale: static_frames as f32 },
            );
        }
        // the whole padded plane: the padding replicates the frame's edge, so the neighbourhood test stays valid there
        let r = &env["render"];
        assert!(r.c0 == 0 && (r.h, r.w) == (hp, wp), "keep_static: render is not a full-resolution plain tensor");
        tail.push(gpu.compute_step(
            "keep_static",
            &[age, &a.buf, &b.buf],
            bytemuck::bytes_of(&static_p),
            &r.buf,
            workgroups_2d(wp, hp, 1),
        ));
    }
    resources.workspace = gpu.workspace.replace(workspace_was);
    if let Some(workspace) = resources.workspace.as_mut() {
        // Only data that crosses the pair/mid boundary or feeds the GMHS stats/composite must remain pair-local.
        let mid: HashSet<_> = tail.iter().flat_map(Step::allocations).map(Rc::as_ptr).collect();
        let mut retained: HashSet<_> = pair_live.intersection(&mid).copied().collect();
        retained.extend(env.get("gm_maps").map(|m| Rc::as_ptr(&m.buf.charge)));
        let used = mid.into_iter().chain(pair_used).collect();
        resources.layout_mismatch = !workspace.finish(&retained, &used);
    }
    #[cfg(test)]
    if !retained.is_empty() {
        let weights = cx
            .weights
            .raw
            .values()
            .chain(cx.weights.gemm.values().map(|b| &b.0))
            .chain(cx.weights.tap.values().map(|b| &b.0))
            .map(|b| Rc::as_ptr(&b.charge))
            .collect::<std::collections::HashSet<_>>();
        let mut bytes = pair_buffers;
        bytes.extend(tail.iter().flat_map(Step::allocations).map(|a| (Rc::as_ptr(a), a.bytes)));
        bytes.remove(&Rc::as_ptr(&gpu.zero.charge));
        bytes.remove(&Rc::as_ptr(&gpu.ones.charge));
        let (mut w, mut r, mut tmp) = (0, 0, 0);
        for (id, size) in bytes {
            if weights.contains(&id) {
                w += size;
            } else if retained.contains(&id) {
                r += size;
            } else {
                tmp += size;
            }
        }
        eprintln!("plan {wp}x{hp}: weights={w}, retained={r}, workspace={tmp}");
    }
    gpu.remember_tune(&cx.tune);
    *gpu.arena.borrow_mut() = arena_was; // released scratch that nobody took is dropped here (VRAM freed)
    gpu.arena_ids.borrow_mut().clear();
    let r = env["render"].clone();
    resources.weights = cx.weights;
    (r, env, cx.t_deps)
}

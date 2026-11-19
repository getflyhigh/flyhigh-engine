//! Texture-path generator (native and browser): the host keeps its frames in GPU textures (a ring the runtime owns and
//! the host writes into), the runtime runs the pair stage on any two slots and synthesises output frames into storage
//! texture views — no CPU bytes anywhere. Same contract as `api::Generator`:
//! - Realtime: `push_slot(s)` runs the pair (previous slot, s); `generate(t, out)` synthesises phase t of that pair.
//! - Video: one frame of lookahead and three rotating pair contexts (prev / cur / next). `push_slot(s)` runs the pair
//!   (previous, s) in the "next" context; the interval that can now be generated is [f_{n-2}, f_{n-1}) = the "cur"
//!   context. With `cadence`, each pair's jump / hold statistics (GMHS maps) re-time elements animated on 2s: the real
//!   slot of the interval is then synthesised too (`generate(0.0, out)` returns true) and the host must present that
//!   output instead of its source frame.
//!
//! Statistics come back through a tagged async readback. Hosts resolve it before the interval's first `generate`.
use crate::gmhs::{CadenceScratch, PairStats, StatsRun};
use crate::gpu::memory::Charge;
use crate::net::build::PlanResources;
use crate::pan::PanRepair;
use crate::{gmhs, Blob, Gpu, NetOpts, SessionExt, Step, TimedRun};
use anyhow::{Context, Result};
use serde_json::Value;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "gen_tex_tests.rs"]
mod tests;

/// Still shot: some pair of the window changed less than this share of cells (held drawing, maybe with twinkling
/// effects); a camera move changes most of the frame in every pair.
const STILL_MAX: f32 = 0.01;
/// Still-shot cadence: a new drawing changes at least this share of cells (and a quarter of a typical step).
const STEP_MIN: f32 = 0.0005;
/// Pair statistics kept for the cadence decision (hold-length estimate).
const HIST: usize = 16;

/// How an output frame of the current interval is re-timed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Retime {
    /// pair context (session index) that renders the element
    pub sess: usize,
    /// phase inside that pair
    pub te: f32,
    /// still shot: render the whole frame from the element pair (no mask)
    pub full: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TexMode {
    Realtime,
    Video,
}

/// Frame size, fine iterations, network size, mode, repair level (0..2; cadence needs >= 1 and raises 0 to 1),
/// cadence re-timing (video mode only), render sampler (None = the model's), static protection (`NetOpts::keep_static`),
/// fast render (`NetOpts::fast_render`, realtime mode), linear_t (with fast
/// render: the t-dependent network runs once per pair at t = 0.5, other phases scale its motion linearly).
#[derive(Clone, Debug)]
pub struct TexOptions {
    pub w: u32,
    pub h: u32,
    pub k: u32,
    pub iw: u32,
    pub ih: u32,
    pub mode: TexMode,
    pub repair: u32,
    pub cadence: bool,
    pub cubic: Option<bool>,
    pub keep_static: bool,
    pub static_frames: u32,
    pub static_tolerance: u32,
    pub fast_render: bool,
    pub linear_t: bool,
    pub protection_mask: bool,
    /// nearest-source regions (`NetOpts::near`, realtime mode)
    pub near: bool,
    /// precise motion (`SessionOptions::precise_motion`): the two-level pair stage; `iw` x `ih` is its fine level
    pub precise: bool,
}
impl TexOptions {
    /// defaults: k 1, internal size `default_internal`, realtime, repair `DEFAULT_REPAIR`, no cadence, model sampler, no static protection
    pub fn new(w: u32, h: u32) -> TexOptions {
        let precise = crate::options::precise_default() && crate::options::precise_applies(w, h);
        let (iw, ih) = if precise { crate::options::precise_internal(w, h) } else { crate::default_internal(w, h) };
        TexOptions {
            w,
            h,
            k: 1,
            iw,
            ih,
            mode: TexMode::Realtime,
            repair: crate::DEFAULT_REPAIR,
            cadence: false,
            cubic: None,
            keep_static: true,
            static_frames: crate::DEFAULT_STATIC_FRAMES,
            static_tolerance: crate::DEFAULT_STATIC_TOLERANCE,
            fast_render: false,
            linear_t: false,
            protection_mask: false,
            near: crate::options::near_default(),
            precise,
        }
    }
}

pub struct GenTex {
    pub opts: TexOptions,
    pub sess: Vec<SessionExt>,
    pub ins: Vec<(wgpu::Texture, wgpu::TextureView)>,
    pub outs: Vec<(wgpu::Texture, wgpu::TextureView)>,
    owned_slots: Vec<Charge>,
    order: [usize; 3],
    stats: [PairStats; 3],
    scratch: Option<CadenceScratch>,
    pan: Option<RefCell<PanRepair>>,
    slots: Vec<usize>, // last pushed input slots, newest last (<= 3)
    n_pushed: u64,
    pending: Option<PendingPairStats>,
    generation: Rc<()>,
    hist: VecDeque<PairStats>, // per-pair statistics, oldest first; back = the "next" pair (lookahead)
    half_done: bool,           // linear_t: the t = 0.5 maps of the current pair are computed
}

struct PairKey {
    generation: Rc<()>,
    sequence: u64,
}

/// In-flight readback for one cadence pair. Finish it before passing it back to the stream.
pub struct PendingPairStats {
    key: PairKey,
    run: StatsRun,
}

/// Completed statistics bound to their original stream generation and pair.
pub struct FinishedPairStats {
    key: PairKey,
    stats: PairStats,
}

impl PendingPairStats {
    pub async fn finish(self) -> Result<FinishedPairStats> {
        Ok(FinishedPairStats { key: self.key, stats: self.run.finish().await? })
    }
}

/// What the host can do after a push.
#[derive(Clone, Copy, Debug, Default)]
pub struct PushInfo {
    /// an interval can be generated now
    pub ready: bool,
    /// input slot of the source frame that starts the generatable interval (Realtime: the newest slot; Video: f_{n-2})
    pub base_slot: usize,
    /// Video + cadence: the base real frame will be re-synthesised by `generate(0.0, ..)`; present that instead
    pub real_replaced: bool,
}

/// The cadence decision for phase `t` of the current interval [k, k+1] from the per-pair statistics `hist` (oldest
/// first; the last entry is the lookahead pair (k+1, k+2), the one before it the current pair) and the session order
/// `order` (prev, cur, next pair contexts): which pair renders the element, at what phase. Pure: the host-independent
/// part of `GenTex::retime`, so it is tested on synthetic histories.
///
/// Animation drawn on 2s / 3s / 4s holds every drawing for several video frames. The drawings are placed on a
/// uniform path: drawing j (first shown at frame f_j, held until f_{j+1}) is reached at the middle of its hold,
/// c_j = f_j + (f_{j+1} - f_j - 1) / 2, and between c_j and c_{j+1} the element is rendered from the pair whose
/// second frame is f_{j+1} (the transition into the next drawing) at phase (tau - c_j) / (c_{j+1} - c_j).
/// Only the prev / cur / next pairs exist, which covers holds of up to 3 frames exactly (one frame of lookahead);
/// unknown future holds use the recent hold length. Drawing changes ("steps") come from the plain frame change on a
/// still shot (any new drawing, however slow, while a held one is an exact copy) and from the element-vs-global
/// motion rule during camera moves. A step between two steps (hold 1) is continuous motion and is left alone;
/// nothing is re-timed across a scene cut.
pub fn cadence_retime(hist: &VecDeque<PairStats>, order: &[usize; 3], t: f32) -> Option<Retime> {
    let n = hist.len();
    if n < 2 {
        return None;
    }
    let cur = n - 2; // hist index of the current pair (k, k+1)
    let lo = hist.iter().rposition(|s| s.cut).map_or(0, |i| i + 1);
    if lo > cur {
        return None;
    } // the current or the next pair is a cut
      // still shot: some pair in the window barely changed (a camera move changes most of the frame every pair)
    let still = hist.iter().skip(lo).any(|s| s.changed < STILL_MAX);
    // a still-shot step changes at least a quarter as much as a typical step of the window (the top third of the
    // changes): twinkling effects / grain animated on 1s stay below it, slow movers with small steps stay above
    let mut ch: Vec<f32> = hist.iter().skip(lo).map(|s| s.changed).collect();
    ch.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let top = &ch[..(ch.len() / 3).max(1)];
    let step_thr = STEP_MIN.max(0.25 * top[top.len() / 2]);
    let step = |s: &PairStats| !s.cut && if still { s.changed > step_thr } else { s.jump };
    // drawing starts as frame offsets from k: pair i spans frames (i - cur, i - cur + 1); a step starts a drawing
    // at its second frame
    let rel = |i: usize| i as i32 - cur as i32;
    let known: Vec<i32> = (lo..n).filter(|&i| step(&hist[i])).map(|i| rel(i) + 1).collect();
    if known.is_empty() {
        return None;
    } // nothing changed in the window: a hold, nothing to re-time
    let mut d: Vec<i32> = known.windows(2).map(|w| w[1] - w[0]).collect();
    d.sort_unstable();
    let l_est = if d.is_empty() { 2 } else { d[d.len() / 2].clamp(1, 4) };
    // virtual neighbours: the drawing before the first known start began at least l_est earlier and no later than
    // the first frame we know about; the next unknown start lies beyond the lookahead (frame k + 2)
    let mut starts = Vec::with_capacity(known.len() + 3);
    starts.push((known[0] - l_est).min(rel(lo)));
    starts.extend_from_slice(&known);
    let last = *known.last().unwrap();
    let s1 = (last + l_est).max(3);
    starts.push(s1);
    starts.push(s1 + l_est);
    let centre = |m: usize| starts[m] as f32 + (starts[m + 1] - starts[m] - 1) as f32 / 2.0;
    let tau = t;
    let m = (0..starts.len() - 2).find(|&m| centre(m) <= tau && tau < centre(m + 1))?;
    let (l0, l1) = (starts[m + 1] - starts[m], starts[m + 2] - starts[m + 1]);
    if l0 == 1 && l1 == 1 {
        return None;
    } // continuous motion: plain interpolation already is the path
    let q = starts[m + 1] - 1; // transition pair, as an offset from the current pair
    let qi = cur as i32 + q;
    if !(-1..=1).contains(&q) || qi < lo as i32 || qi >= n as i32 || !step(&hist[qi as usize]) {
        return None;
    }
    let sess = order[(q + 1) as usize];
    let te = ((tau - centre(m)) / (centre(m + 1) - centre(m))).clamp(0.0, 1.0);
    Some(Retime { sess, te, full: still })
}

impl GenTex {
    pub fn new(
        gpu: &Gpu,
        blob: &Blob,
        ops: &[Value],
        opts: TexOptions,
        tune: Option<&crate::TuneTable>,
    ) -> Result<GenTex> {
        // fast render has no cascade: precise motion set by the environment default falls back to the plain rung
        let mut opts = opts;
        opts.precise &= crate::options::precise_applies(opts.w, opts.h);
        if opts.precise && opts.fast_render {
            opts.precise = false;
            if (opts.iw, opts.ih) == crate::options::precise_internal(opts.w, opts.h) {
                (opts.iw, opts.ih) = crate::default_internal(opts.w, opts.h);
            }
        }
        anyhow::ensure!((1..=3).contains(&opts.k), "k must be 1..3");
        anyhow::ensure!(opts.repair <= 2, "repair must be 0..2");
        anyhow::ensure!(opts.w >= 64 && opts.h >= 64, "frame size too small (min 64 x 64)");
        anyhow::ensure!(
            opts.iw >= 32 && opts.ih >= 32 && opts.iw <= opts.w && opts.ih <= opts.h,
            "internal size must be within 32 x 32 .. the frame size"
        );
        let n_sess = if opts.mode == TexMode::Video { 3 } else { 1 };
        let cadence = opts.mode == TexMode::Video && opts.cadence;
        let net = NetOpts {
            repair: if cadence { opts.repair.max(1) } else { opts.repair },
            keep_maps: cadence,
            cubic: opts.cubic.unwrap_or(blob.render_cubic),
            keep_static: opts.keep_static,
            static_frames: opts.static_frames,
            static_tolerance: opts.static_tolerance,
            fast_render: opts.fast_render && opts.mode == TexMode::Realtime,
            protection_mask: opts.protection_mask,
            // games (realtime); video and anime keep the soft blend until it is measured there
            decisive: crate::options::decisive_default() && opts.mode == TexMode::Realtime,
            still: crate::options::still_default() && opts.mode == TexMode::Realtime,
            near: opts.near && opts.mode == TexMode::Realtime,
            cascade: crate::options::cascade_for(opts.precise).0,
            cascade_iters: crate::options::cascade_for(opts.precise).1,
        };
        let mut sess = Vec::new();
        let mut resources = PlanResources::default();
        // Every mid consumes the render before the next one, including the compositor's saved background.
        if cadence {
            resources.render = Some(gpu.alloc((3 * (opts.w.div_ceil(32) * 32) * (opts.h.div_ceil(32) * 32)) as usize));
        }
        let can_share_workspace = cadence && std::env::var_os("FH_NO_ARENA").is_none();
        let mut warmed_tune = None;
        for attempt in 0..3 {
            resources.workspace = (can_share_workspace && attempt < 2).then(Default::default);
            resources.layout_mismatch = false;
            for _ in 0..n_sess {
                sess.push(SessionExt::new_with(
                    gpu,
                    blob,
                    ops,
                    opts.w,
                    opts.h,
                    opts.k,
                    opts.iw,
                    opts.ih,
                    warmed_tune.as_ref().or(tune),
                    &net,
                    &mut resources,
                ));
                if resources.layout_mismatch {
                    break;
                }
            }
            if !resources.layout_mismatch {
                break;
            }
            // Cold native autotune can change the allocation trace. Retry with those choices, otherwise keep private arenas.
            sess.clear();
            if attempt == 0 {
                let mut choices = gpu.tune.borrow().clone();
                choices.extend(tune.into_iter().flatten().map(|(k, v)| (*k, *v)));
                warmed_tune = Some(choices);
            }
        }
        let pan = (opts.repair == 2 && !opts.protection_mask)
            .then(|| RefCell::new(PanRepair::new(gpu, opts.w, opts.h, &sess)));
        let scratch = if cadence {
            Some(CadenceScratch::new(gpu, &sess[0]).context("cadence: GMHS maps missing")?)
        } else {
            None
        };
        Ok(GenTex {
            opts,
            sess,
            ins: Vec::new(),
            outs: Vec::new(),
            owned_slots: Vec::new(),
            order: [0, 1, 2],
            stats: [PairStats::default(); 3],
            scratch,
            pan,
            slots: Vec::new(),
            n_pushed: 0,
            pending: None,
            generation: Rc::new(()),
            hist: VecDeque::new(),
            half_done: false,
        })
    }
    pub fn add_input(&mut self, gpu: &Gpu) -> usize {
        let t = gpu.input_texture_l(self.opts.w, self.opts.h, &format!("fh_vin_{}", self.ins.len()));
        self.owned_slots.push(gpu.memory.rgba8_slot(&t));
        let v = t.create_view(&wgpu::TextureViewDescriptor::default());
        self.ins.push((t, v));
        self.ins.len() - 1
    }
    /// A texture the host owns (imported from another API, or its own) as an input slot: the frame size, rgba8 or
    /// bgra8, sampleable.
    pub fn add_input_texture(&mut self, tex: wgpu::Texture) -> Result<usize> {
        anyhow::ensure!(
            tex.width() == self.opts.w && tex.height() == self.opts.h,
            "input texture is {}x{}, the generator takes {}x{}",
            tex.width(),
            tex.height(),
            self.opts.w,
            self.opts.h
        );
        anyhow::ensure!(
            matches!(tex.format(), wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Bgra8Unorm),
            "input texture format {:?}: rgba8unorm or bgra8unorm",
            tex.format()
        );
        anyhow::ensure!(tex.usage().contains(wgpu::TextureUsages::TEXTURE_BINDING), "input texture is not sampleable");
        let v = tex.create_view(&wgpu::TextureViewDescriptor::default());
        self.ins.push((tex, v));
        Ok(self.ins.len() - 1)
    }
    pub fn add_output(&mut self, gpu: &Gpu) -> usize {
        let t = gpu.output_texture_l(self.opts.w, self.opts.h, &format!("fh_vout_{}", self.outs.len()));
        self.owned_slots.push(gpu.memory.rgba8_slot(&t));
        let v = t.create_view(&wgpu::TextureViewDescriptor::default());
        self.outs.push((t, v));
        self.outs.len() - 1
    }
    /// The protection mask (`TexOptions::protection_mask`) into every session of the generator (video mode rotates three).
    pub fn write_mask(&self, gpu: &Gpu, mask: &[u8]) -> Result<()> {
        self.sess.iter().try_for_each(|s| s.write_mask(gpu, mask))
    }
    pub fn warmup(&self) -> usize {
        if self.opts.mode == TexMode::Video {
            3
        } else {
            2
        }
    }
    /// Forget the pushed frames (seek, decoder gap): the next pushes warm up again; the pair contexts stay built.
    pub fn reset_history(&mut self) {
        self.slots.clear();
        self.n_pushed = 0;
        self.stats = [PairStats::default(); 3];
        self.pending = None;
        self.generation = Rc::new(());
        self.hist.clear();
    }
    pub fn stats(&self) -> [PairStats; 3] {
        self.stats
    }
    pub fn cadence(&self) -> bool {
        self.scratch.is_some()
    }

    fn pair_steps(&self, gpu: &Gpu, pair: usize, a: usize, b: usize) -> Vec<Step> {
        let (a, b) = ((a, &self.ins[a].1), (b, &self.ins[b].1));
        let mut steps = self.sess[pair].pair_steps_views(gpu, a, b);
        if let Some(pan) = &self.pan {
            pan.borrow_mut().search(gpu, pair, a, b, &mut steps);
        }
        steps
    }

    fn repair_pan(&self, gpu: &Gpu, steps: &mut Vec<Step>, pair: usize, slots: (usize, usize), out: usize, t: f32) {
        if let Some(pan) = &self.pan {
            let (a, b) = slots;
            steps.push(pan.borrow_mut().render(
                gpu,
                pair,
                (a, &self.ins[a].1),
                (b, &self.ins[b].1),
                (out, &self.outs[out].1),
                t,
            ));
        }
    }

    /// The host wrote the newest source frame into input slot `slot`: run its pair. Video + cadence: the pair's
    /// statistics readback is started; call `take_stats()` (native) or await it (browser) before generating.
    pub fn push_slot(&mut self, gpu: &Gpu, slot: usize) -> Result<PushInfo> {
        self.push_slot_ex(gpu, slot, false).map(|(info, _)| info)
    }

    /// `push_slot` with the pair submitted under timestamp queries: the run to await for its GPU milliseconds (None
    /// without timestamps, or when no pair ran).
    pub fn push_slot_timed(&mut self, gpu: &Gpu, slot: usize) -> Result<(PushInfo, Option<TimedRun>)> {
        self.push_slot_ex(gpu, slot, true)
    }

    fn push_slot_ex(&mut self, gpu: &Gpu, slot: usize, timed: bool) -> Result<(PushInfo, Option<TimedRun>)> {
        anyhow::ensure!(slot < self.ins.len(), "bad input slot");
        let prev = self.slots.last().copied();
        self.slots.push(slot);
        if self.slots.len() > 3 {
            self.slots.remove(0);
        }
        self.n_pushed += 1;
        self.half_done = false;
        let Some(prev) = prev else {
            return Ok((PushInfo::default(), None));
        };
        match self.opts.mode {
            TexMode::Realtime => {
                let steps = self.pair_steps(gpu, 0, prev, slot);
                let run = run(gpu, &steps, timed);
                Ok((PushInfo { ready: true, base_slot: prev, real_replaced: false }, run))
            }
            TexMode::Video => {
                self.order.rotate_left(1);
                let sn = self.order[2];
                let steps = self.pair_steps(gpu, sn, prev, slot);
                let run = run(gpu, &steps, timed);
                self.stats.rotate_left(1);
                self.stats[2] = PairStats::default();
                self.hist.push_back(PairStats::default());
                while self.hist.len() > HIST {
                    self.hist.pop_front();
                }
                if let Some(sc) = &self.scratch {
                    let key = PairKey { generation: self.generation.clone(), sequence: self.n_pushed };
                    self.pending =
                        Some(PendingPairStats { key, run: gmhs::pair_stats_begin(gpu, &self.sess[sn], sc)? });
                }
                let ready = self.slots.len() >= 3;
                Ok((PushInfo { ready, base_slot: if ready { self.slots[0] } else { slot }, real_replaced: false }, run))
            }
        }
    }
    /// Take the pending statistics readback (Video + cadence); the caller awaits `finish` and passes the result to `set_stats`.
    pub fn take_stats(&mut self) -> Option<PendingPairStats> {
        self.pending.take()
    }
    pub fn has_pending_stats(&self) -> bool {
        self.pending.is_some()
    }
    /// Ignore completions from another generation or a pair that has left history.
    pub fn set_stats(&mut self, st: FinishedPairStats) {
        if !Rc::ptr_eq(&st.key.generation, &self.generation) {
            return;
        }
        let Some(age) = self.n_pushed.checked_sub(st.key.sequence) else { return };
        if age >= self.hist.len() as u64 {
            return;
        }
        let i = self.hist.len() - 1 - age as usize;
        self.hist[i] = st.stats;
        if age < self.stats.len() as u64 {
            let i = self.stats.len() - 1 - age as usize;
            self.stats[i] = st.stats;
        }
    }
    /// Native convenience: block for the pending statistics.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn resolve_stats(&mut self, gpu: &Gpu) -> Result<()> {
        if let Some(run) = self.pending.take() {
            gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
            let st = pollster::block_on(run.finish())?;
            self.set_stats(st);
        }
        Ok(())
    }
    /// Cadence decision for phase `t` of the current interval [k, k+1]: which pair renders the element, at what phase.
    ///
    /// Animation drawn on 2s / 3s / 4s holds every drawing for several video frames. The drawings are placed on a
    /// uniform path: drawing j (first shown at frame f_j, held until f_{j+1}) is reached at the middle of its hold,
    /// c_j = f_j + (f_{j+1} - f_j - 1) / 2, and between c_j and c_{j+1} the element is rendered from the pair whose
    /// second frame is f_{j+1} (the transition into the next drawing) at phase (tau - c_j) / (c_{j+1} - c_j).
    /// Only the prev / cur / next pairs exist, which covers holds of up to 3 frames exactly (one frame of lookahead);
    /// unknown future holds use the recent hold length. Drawing changes ("steps") come from the plain frame change on a
    /// still shot (any new drawing, however slow, while a held one is an exact copy) and from the element-vs-global
    /// motion rule during camera moves. A step between two steps (hold 1) is continuous motion and is left alone;
    /// nothing is re-timed across a scene cut.
    fn retime_for(&self, t: f32) -> Option<Retime> {
        if self.opts.mode != TexMode::Video || self.scratch.is_none() || self.slots.len() < 3 {
            return None;
        }
        cadence_retime(&self.hist, &self.order, t)
    }
    /// Whether the real slot of the current interval is replaced by a synthesised frame (Video + cadence).
    pub fn real_replaced(&self) -> bool {
        self.retime_for(0.0).is_some()
    }
    /// The cadence decision for phase `t` (diagnostics).
    pub fn retime(&self, t: f32) -> Option<Retime> {
        self.retime_for(t)
    }
    /// GPU time (ms, timestamp queries) of one more generated frame of the current pair in this configuration, written to
    /// output slot `out`: the render alone with linear_t, else the t-dependent network plus the render. None before a
    /// pair exists or in video mode.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn frame_cost_ms(&self, gpu: &Gpu, out: usize) -> Option<f64> {
        if self.opts.mode != TexMode::Realtime || self.slots.len() < 2 || out >= self.outs.len() {
            return None;
        }
        let s = &self.sess[0];
        let (sa, sb) = (self.slots[self.slots.len() - 2], self.slots[self.slots.len() - 1]);
        let (a, b, o) = ((sa, &self.ins[sa].1), (sb, &self.ins[sb].1), (out, &self.outs[out].1));
        let mut steps = if s.is_fast() && self.opts.linear_t {
            vec![s.render_tex_step(gpu, a, b, o, Some(0.5))]
        } else if s.is_fast() {
            s.mid_steps_fast(gpu, a, b, o)
        } else {
            s.mid_steps(gpu, out, o.1)
        };
        self.repair_pan(gpu, &mut steps, 0, (sa, sb), out, 0.5);
        Some(gpu.gpu_time_ms_n(&steps, 5))
    }
    /// GPU time (ms) of the pair stage for the current pair (realtime mode), measured by running it again.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn pair_cost_ms(&self, gpu: &Gpu) -> Option<f64> {
        if self.opts.mode != TexMode::Realtime || self.slots.len() < 2 {
            return None;
        }
        let (sa, sb) = (self.slots[self.slots.len() - 2], self.slots[self.slots.len() - 1]);
        let steps = self.pair_steps(gpu, 0, sa, sb);
        Some(gpu.gpu_time_ms_n(&steps, 3))
    }
    /// Synthesise phase `t` of the current interval into output slot `out`. Returns false when nothing was written
    /// (t == 0 without cadence re-timing: the host presents its own source frame).
    pub fn generate(&mut self, gpu: &Gpu, t: f32, out: usize) -> Result<bool> {
        self.generate_ex(gpu, t, out, false).map(|(written, _)| written)
    }

    /// `generate` with the work submitted under timestamp queries: the run to await for its GPU milliseconds; None
    /// without timestamps, when nothing was written, or for a cadence composite (two mids and a mask: not one cost).
    pub fn generate_timed(&mut self, gpu: &Gpu, t: f32, out: usize) -> Result<(bool, Option<TimedRun>)> {
        self.generate_ex(gpu, t, out, true)
    }

    fn generate_ex(&mut self, gpu: &Gpu, t: f32, out: usize, timed: bool) -> Result<(bool, Option<TimedRun>)> {
        anyhow::ensure!(out < self.outs.len(), "bad output slot");
        let view = &self.outs[out].1;
        match self.opts.mode {
            TexMode::Realtime => {
                if t <= 0.0 {
                    return Ok((false, None));
                }
                anyhow::ensure!(self.slots.len() >= 2, "realtime mode: push two frames first");
                let s = &self.sess[0];
                let (sa, sb) = (self.slots[self.slots.len() - 2], self.slots[self.slots.len() - 1]);
                let (setup, mut steps) = if s.is_fast() {
                    let (a, b) = ((sa, &self.ins[sa].1), (sb, &self.ins[sb].1));
                    if self.opts.linear_t {
                        // the network's t-dependent stage once per pair; each phase is then only a render
                        let setup = if self.half_done {
                            Vec::new()
                        } else {
                            let setup = s.mid_map_steps_half(gpu);
                            self.half_done = true;
                            setup
                        };
                        (setup, vec![s.render_tex_step(gpu, a, b, (out, view), Some(t))])
                    } else {
                        s.set_t(gpu, t);
                        (Vec::new(), s.mid_steps_fast(gpu, a, b, (out, view)))
                    }
                } else {
                    s.set_t(gpu, t);
                    (Vec::new(), s.mid_steps(gpu, out, view))
                };
                self.repair_pan(gpu, &mut steps, 0, (sa, sb), out, t);
                Ok((true, gpu.submit_optional_timing(&setup, &steps, timed)))
            }
            TexMode::Video => {
                anyhow::ensure!(self.slots.len() >= 3, "video mode: push three frames first");
                let sc_ = self.order[1];
                match (self.retime_for(t), &self.scratch) {
                    (Some(r), Some(scr)) => {
                        gmhs::composite_frame(
                            gpu,
                            scr,
                            (&self.sess[sc_], t),
                            (&self.sess[r.sess], r.te),
                            (&self.sess[r.sess], r.full),
                            out,
                            view,
                        )?;
                        Ok((true, None))
                    }
                    _ => {
                        if t <= 0.0 {
                            return Ok((false, None));
                        }
                        let s = &self.sess[sc_];
                        s.set_t(gpu, t);
                        let mut steps = s.mid_steps(gpu, out, view);
                        self.repair_pan(gpu, &mut steps, sc_, (self.slots[0], self.slots[1]), out, t);
                        Ok((true, run(gpu, &steps, timed)))
                    }
                }
            }
        }
    }
}

/// Runs `steps`; with `timed` and timestamp queries they go out under a timestamp pair instead and the run is
/// returned to await for the GPU milliseconds.
fn run(gpu: &Gpu, steps: &[Step], timed: bool) -> Option<TimedRun> {
    gpu.submit_optional_timing(&[], steps, timed)
}

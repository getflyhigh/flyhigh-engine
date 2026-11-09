//! The Stream (ARCHITECTURE.md): source frames with timestamps in, generatable intervals out. It owns the input ring and
//! the output slots, runs the pair stage when a frame arrives (at once in realtime mode, one frame behind in video
//! mode), classifies each pair (scene cut, held frame, motion), keeps the cadence statistics and synthesises any
//! phase of the current interval on request.
//!
//! It wraps `GenTex`, the texture-path generator, and adds timestamps and intervals, the classifier, the camera warp
//! (once the host passes mouse input) and the report.
pub mod classify;

pub use classify::{Classification, Classifier, Thresholds};

use crate::gen_tex::{GenTex, Retime, TexMode, TexOptions};
use crate::gmhs::{PairStats, StatsRun};
use crate::gpu::readback::Readback;
use crate::options::Options;
use crate::warp::{Calibration, CameraWarp, WarpOptions};
use crate::{Blob, Engine, Gpu, Model, SessionOptions, TimedRun, TuneTable};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, VecDeque};

/// What the interval between the previous source frame and this one holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntervalKind {
    /// Ordinary motion: interpolated.
    #[default]
    Motion,
    /// A scene cut: the source frames are shown, nothing in between.
    Cut,
    /// A repeated frame (anime held frames, a paused game): shown as is, counted for the unique-frame rate.
    Hold,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    /// No lookahead: `push` runs the pair (previous, new) and the interval can be generated at once.
    Realtime,
    /// One frame of lookahead: the interval that can be generated starts two frames back; cadence re-timing
    /// (`cadence`) needs this mode.
    Video,
}

/// A pair whose mean per-channel difference (0..1) is above `threshold` is a scene cut.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SceneCut {
    pub threshold: f32,
}

impl Default for SceneCut {
    fn default() -> Self {
        SceneCut { threshold: 0.08 }
    }
}

/// A pair whose mean difference is below `mean` and whose most different cell is below `max` is a held frame.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct HeldFrames {
    pub mean: f32,
    pub max: f32,
}

impl Default for HeldFrames {
    fn default() -> Self {
        // the extension's measured thresholds (2.5 / 765 and 45 / 765): codec noise on a re-encoded hold stays below
        HeldFrames { mean: 0.0033, max: 0.06 }
    }
}

/// How source frames with timestamps become generatable intervals (ARCHITECTURE.md); fixed when the stream opens. The
/// session options travel beside them in `Stream::open`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Options)]
#[serde(default, deny_unknown_fields)]
pub struct StreamOptions {
    /// Realtime: no lookahead, the in-betweens of the previous and the newest frame; for games and capture. Video: one
    /// source frame of lookahead (one frame more latency), the interval that can be generated starts two frames back;
    /// cadence re-timing needs it.
    #[option(default = "realtime", values = "realtime, video", c = "mode (0, 1)")]
    pub mode: Mode,
    /// Video mode only. Anime is often drawn on twos (each drawing held for two frames) over a camera pan on every
    /// frame; with cadence on, held elements are re-timed onto a smooth path. Presented frames are then generated
    /// in-betweens, the real drawings included, except where the model could not match the frames (there the real
    /// drawing stays): big drawing steps show the generator's softness more often. A choice for the viewer, not a
    /// default. Needs repair at least 1 (0 is raised to 1).
    #[option(values = "on, off", c = "cadence (0, 1)")]
    pub cadence: bool,
    /// A pair whose mean per-channel difference (0 to 1, measured on a 48 x 27 grid) is above the threshold is a scene
    /// cut: the interval is not interpolated, the source frames are shown. Off = every pair is motion.
    #[option(values = "off, or a threshold 0 to 1", c = "scene_cut (threshold, 0 = off)")]
    pub scene_cut: Option<SceneCut>,
    /// A pair whose mean difference is below `mean` and whose most different cell is below `max` is a held frame: shown
    /// as is and counted for the unique-frame rate, instead of a still in-between. The first change after a held frame is
    /// then shown as drawn too (animation on twos and threes is not motion), unless cadence re-timing spreads it. On for
    /// anime; the defaults 0.0033 and 0.06 stay above codec noise on a re-encoded hold.
    #[option(values = "off, or mean and max thresholds 0 to 1", c = "held_frames (0, 1), held_mean, held_max")]
    pub held_frames: Option<HeldFrames>,
    /// Input slots created up front (more with `add_input`); the Pacer's history says how many a host needs.
    pub inputs: u32,
    /// Output slots created up front (more with `add_output`).
    pub outputs: u32,
}

impl Default for StreamOptions {
    fn default() -> Self {
        StreamOptions {
            mode: Mode::Realtime,
            cadence: false,
            scene_cut: Some(SceneCut::default()),
            held_frames: None,
            inputs: 4,
            outputs: 4,
        }
    }
}

/// An interval the Stream can generate now.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Interval {
    /// Timestamps of the source frames that bound it (the host's units, as given to `push`).
    pub start_pts: f64,
    pub end_pts: f64,
    /// The input slot of the real frame that starts it; present it unless `real_replaced()` says the cadence
    /// re-timed it.
    pub base: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Push {
    /// None while warming up (realtime: the first frame; video: the first two).
    pub interval: Option<Interval>,
    /// Video + cadence: the pair's statistics are still in flight; take them with `take_stats` (browser) or
    /// `resolve_stats` (native) before the first `generate` of the interval.
    pub stats_pending: bool,
}

/// A frame the host is about to present, for `Stream::warp`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shown {
    /// A source frame in its input slot (`Generated::SourceIsEnough`, or a held frame or a cut).
    Input(usize),
    /// A generated frame in its output slot.
    Output(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Generated {
    /// The output slot holds the frame.
    Written,
    /// Nothing was written: `t` is 0 and the cadence did not re-time the real slot, so the host presents its own
    /// source frame.
    SourceIsEnough,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamCounters {
    pub pushes: u64,
    pub pairs: u64,
    pub mids: u64,
    pub retimed: u64,
    pub source_enough: u64,
    pub classified: u64,
    /// Classifications skipped because every readback was still in flight.
    pub classify_busy: u64,
    pub resets: u64,
    /// Frames the camera warp turned.
    #[serde(default)]
    pub warped: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StreamReport {
    pub mode: Mode,
    pub cadence: bool,
    pub width: u32,
    pub height: u32,
    pub internal_width: u32,
    pub internal_height: u32,
    pub k: u32,
    pub repair: u32,
    pub inputs: usize,
    pub outputs: usize,
    pub pushed: usize,
    pub counters: StreamCounters,
    /// The camera warp's calibration; None until the host passes mouse input.
    #[serde(default)]
    pub camera_warp: Option<Calibration>,
}

pub struct Stream {
    gen: GenTex,
    opts: StreamOptions,
    classifier: Option<Classifier>,
    /// (slot, pts) of the frames the generator holds, newest last.
    pushed: VecDeque<(usize, f64)>,
    next_in: usize,
    counters: StreamCounters,
    /// Created by the first mouse input: a host without one pays nothing.
    warp: Option<CameraWarp>,
    /// The moment each slot shows, in push timestamps: inputs by their own, outputs by their interval and phase.
    in_pts: HashMap<usize, f64>,
    out_pts: HashMap<usize, f64>,
    interval: Option<Interval>,
}

impl Stream {
    /// `tex` carries the frame size and the network options; `opts.mode` and `opts.cadence` override its mode fields.
    pub fn new(
        gpu: &Gpu,
        blob: &Blob,
        ops: &[Value],
        tex: TexOptions,
        opts: &StreamOptions,
        tune: Option<&TuneTable>,
    ) -> Result<Stream> {
        let mut tex = tex;
        tex.mode = match opts.mode {
            Mode::Realtime => TexMode::Realtime,
            Mode::Video => TexMode::Video,
        };
        tex.cadence = opts.cadence;
        let mut gen = GenTex::new(gpu, blob, ops, tex, tune)?;
        for _ in 0..opts.inputs {
            gen.add_input(gpu);
        }
        for _ in 0..opts.outputs {
            gen.add_output(gpu);
        }
        let classifier = (opts.scene_cut.is_some() || opts.held_frames.is_some()).then(|| {
            Classifier::new(
                gpu,
                Thresholds {
                    cut_mean: opts.scene_cut.map(|c| c.threshold),
                    hold_mean: opts.held_frames.map(|h| h.mean),
                    hold_max: opts.held_frames.map_or(0.0, |h| h.max),
                    // cadence re-timing spreads such a change over the frames it was held for, and needs it as motion
                    step_after_hold: opts.held_frames.is_some() && !opts.cadence,
                },
            )
        });
        Ok(Stream {
            gen,
            opts: opts.clone(),
            classifier,
            pushed: VecDeque::new(),
            next_in: 0,
            counters: StreamCounters::default(),
            warp: None,
            in_pts: HashMap::new(),
            out_pts: HashMap::new(),
            interval: None,
        })
    }

    /// The Stream of `model` on `engine` with the session options (ARCHITECTURE.md) (the mode fields come from `opts`).
    pub fn open(engine: &Engine, model: &Model, session: &SessionOptions, opts: &StreamOptions) -> Result<Stream> {
        session.validate()?;
        Stream::new(engine.gpu(), model.blob(), model.ops(), session.tex_options(), opts, session.tune.as_ref())
    }

    pub fn options(&self) -> &StreamOptions {
        &self.opts
    }

    pub fn tex_options(&self) -> &TexOptions {
        &self.gen.opts
    }

    /// The generator underneath, for what this API does not expose yet.
    pub fn gen(&self) -> &GenTex {
        &self.gen
    }

    pub fn gen_mut(&mut self) -> &mut GenTex {
        &mut self.gen
    }

    pub fn add_input(&mut self, gpu: &Gpu) -> usize {
        self.gen.add_input(gpu)
    }

    pub fn add_output(&mut self, gpu: &Gpu) -> usize {
        self.gen.add_output(gpu)
    }

    /// A texture the host owns (its own, or one imported with `interop`) as an input slot of the ring.
    pub fn add_input_texture(&mut self, tex: wgpu::Texture) -> Result<usize> {
        if let Some(c) = &mut self.classifier {
            c.forget_textures();
        }
        if let Some(w) = &mut self.warp {
            w.forget_textures();
        }
        self.gen.add_input_texture(tex)
    }

    pub fn inputs(&self) -> usize {
        self.gen.ins.len()
    }

    pub fn outputs(&self) -> usize {
        self.gen.outs.len()
    }

    pub fn input_texture(&self, slot: usize) -> &wgpu::Texture {
        &self.gen.ins[slot].0
    }

    pub fn input_view(&self, slot: usize) -> &wgpu::TextureView {
        &self.gen.ins[slot].1
    }

    pub fn output_texture(&self, slot: usize) -> &wgpu::Texture {
        &self.gen.outs[slot].0
    }

    pub fn output_view(&self, slot: usize) -> &wgpu::TextureView {
        &self.gen.outs[slot].1
    }

    /// The protection mask (`SessionOptions::protection_mask`): one byte per pixel, 0 interpolates, anything else is
    /// taken from the source frames; applies from the next pair.
    pub fn write_mask(&self, gpu: &Gpu, mask: &[u8]) -> Result<()> {
        self.gen.write_mask(gpu, mask)
    }

    /// Frames to push before the first interval can be generated (realtime 2, video 3).
    pub fn warmup(&self) -> usize {
        self.gen.warmup()
    }

    /// The newest pushed frame's slot.
    pub fn last_pushed(&self) -> Option<usize> {
        self.pushed.back().map(|(s, _)| *s)
    }

    /// The slot to write the next source frame into: round robin over the ring, never one the generator still
    /// reads. `in_use` says which slots the host still needs (queued for presentation); with a ring sized by the
    /// Pacer's `history_frames` there is always a free one.
    pub fn next_input(&mut self, in_use: impl Fn(usize) -> bool) -> Option<usize> {
        let n = self.gen.ins.len();
        for i in 0..n {
            let s = (self.next_in + i) % n;
            if !self.pushed.iter().any(|(p, _)| *p == s) && !in_use(s) {
                self.next_in = (s + 1) % n;
                return Some(s);
            }
        }
        None
    }

    /// Classifies the pair (`previous`, `slot`) before it is pushed: the kind the Pacer wants with the frame. None
    /// when the Stream has no classifier or every readback is in flight; the host then takes the pair as motion.
    pub fn classify(&mut self, gpu: &Gpu, previous: usize, slot: usize) -> Option<Readback<Classification>> {
        let c = self.classifier.as_mut()?;
        let (a, b) = (&self.gen.ins[previous].1, &self.gen.ins[slot].1);
        match c.begin(gpu, (previous, a), (slot, b)) {
            Some(r) => {
                self.counters.classified += 1;
                Some(r)
            }
            None => {
                self.counters.classify_busy += 1;
                None
            }
        }
    }

    /// The host wrote the newest source frame into `slot` with timestamp `pts` (its own units, usually media
    /// milliseconds): runs the pair stage.
    pub fn push(&mut self, gpu: &Gpu, slot: usize, pts: f64) -> Result<Push> {
        self.push_ex(gpu, slot, pts, false).map(|(p, _)| p)
    }

    /// `push` with the pair submitted under timestamp queries: the run to await for its GPU milliseconds (the
    /// Pacer's `cost_sample`); None without timestamps, or when no pair ran.
    pub fn push_timed(&mut self, gpu: &Gpu, slot: usize, pts: f64) -> Result<(Push, Option<TimedRun>)> {
        self.push_ex(gpu, slot, pts, true)
    }

    fn push_ex(&mut self, gpu: &Gpu, slot: usize, pts: f64, timed: bool) -> Result<(Push, Option<TimedRun>)> {
        let (info, run) =
            if timed { self.gen.push_slot_timed(gpu, slot)? } else { (self.gen.push_slot(gpu, slot)?, None) };
        self.counters.pushes += 1;
        if self.pushed.back().is_some() {
            self.counters.pairs += 1;
        }
        self.pushed.push_back((slot, pts));
        while self.pushed.len() > 3 {
            self.pushed.pop_front();
        }
        self.in_pts.insert(slot, pts);
        if let Some(w) = &mut self.warp {
            w.frame(gpu, (in_key(slot), &self.gen.ins[slot].1), pts);
        }
        let interval = info.ready.then(|| {
            let i = self.pushed.iter().position(|(s, _)| *s == info.base_slot).unwrap_or(0);
            let (base, start_pts) = self.pushed[i];
            let end_pts = self.pushed.get(i + 1).map_or(pts, |(_, p)| *p);
            Interval { start_pts, end_pts, base }
        });
        self.interval = interval;
        Ok((Push { interval, stats_pending: self.gen.has_pending_stats() }, run))
    }

    /// Video + cadence: the pending statistics readback; await `finish` and pass the result to `set_stats`.
    pub fn take_stats(&mut self) -> Option<StatsRun> {
        self.gen.take_stats()
    }

    pub fn set_stats(&mut self, st: PairStats) {
        self.gen.set_stats(st);
    }

    /// Native: block for the pending statistics.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn resolve_stats(&mut self, gpu: &Gpu) -> Result<()> {
        self.gen.resolve_stats(gpu)
    }

    /// Whether the real slot of the current interval is replaced by a synthesised frame (video + cadence).
    pub fn real_replaced(&self) -> bool {
        self.gen.real_replaced()
    }

    /// The cadence decision for phase `t` (diagnostics).
    pub fn retime(&self, t: f32) -> Option<Retime> {
        self.gen.retime(t)
    }

    /// Phase `t` of the current interval into output slot `out`.
    pub fn generate(&mut self, gpu: &Gpu, t: f32, out: usize) -> Result<Generated> {
        self.generate_ex(gpu, t, out, false).map(|(g, _)| g)
    }

    /// `generate` with the work submitted under timestamp queries: the run to await for its GPU milliseconds; None
    /// without timestamps, when nothing was written, or for a cadence composite (two mids and a mask).
    pub fn generate_timed(&mut self, gpu: &Gpu, t: f32, out: usize) -> Result<(Generated, Option<TimedRun>)> {
        self.generate_ex(gpu, t, out, true)
    }

    fn generate_ex(&mut self, gpu: &Gpu, t: f32, out: usize, timed: bool) -> Result<(Generated, Option<TimedRun>)> {
        let (written, run) =
            if timed { self.gen.generate_timed(gpu, t, out)? } else { (self.gen.generate(gpu, t, out)?, None) };
        if written {
            if let Some(i) = self.interval {
                self.out_pts.insert(out, i.start_pts + t as f64 * (i.end_pts - i.start_pts));
            }
            if t <= 0.0 {
                self.counters.retimed += 1;
            } else {
                self.counters.mids += 1;
            }
            Ok((Generated::Written, run))
        } else {
            self.counters.source_enough += 1;
            Ok((Generated::SourceIsEnough, run))
        }
    }

    /// GPU time of one more generated frame of the current pair (realtime, timestamp queries).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn frame_cost_ms(&self, gpu: &Gpu, out: usize) -> Option<f64> {
        self.gen.frame_cost_ms(gpu, out)
    }

    /// GPU time of the pair stage (realtime, timestamp queries).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn pair_cost_ms(&self, gpu: &Gpu) -> Option<f64> {
        self.gen.pair_cost_ms(gpu)
    }

    /// Compiles every kernel the stream runs before the first real frame: the warm-up pushes and one mid on its own
    /// slots (added when missing), then `reset`. Browsers compile pipelines on first use (Firefox serially, 0.1 to 2 s
    /// each); a host calls this and awaits the queue before playback.
    pub fn warm_up(&mut self, gpu: &Gpu) -> Result<()> {
        let need = self.warmup();
        while self.gen.ins.len() < need {
            self.gen.add_input(gpu);
        }
        if self.gen.outs.is_empty() {
            self.gen.add_output(gpu);
        }
        for i in 0..need {
            self.gen.push_slot(gpu, i)?;
        }
        self.gen.take_stats();
        self.gen.generate(gpu, 0.5, 0)?;
        self.reset();
        Ok(())
    }

    /// A seek, a decoder gap, a new source: forget the frames, keep the plans (and the camera warp's calibration).
    pub fn reset(&mut self) {
        self.counters.resets += 1;
        self.gen.reset_history();
        self.pushed.clear();
        self.interval = None;
        if let Some(c) = &self.classifier {
            c.forget_history();
        }
        if let Some(w) = &mut self.warp {
            w.break_pair();
        }
    }

    /// Mouse counts at time `t`, as the game receives them (relative input). `t` is on the clock of `push`'s
    /// timestamps, so those must be the frames' arrival times on the host's clock in milliseconds: the camera warp
    /// learns from the two together how many pixels a count turns the picture and how late the picture shows it. The
    /// first call starts the warp; a host that never calls it runs none of it.
    pub fn mouse(&mut self, gpu: &Gpu, t: f64, dx: f64, dy: f64) {
        let (w, h) = (self.gen.opts.w, self.gen.opts.h);
        self.warp.get_or_insert_with(|| CameraWarp::new(gpu, w, h, WarpOptions::default())).mouse(t, dx, dy);
    }

    /// Whether the host has the mouse captured (relative input turning a game camera) or released (a cursor on
    /// screen, a menu): released holds the camera warp off at once.
    pub fn set_mouse_captured(&mut self, captured: bool) {
        if let Some(w) = &mut self.warp {
            w.set_captured(captured);
        }
    }

    /// The camera warp, once started by `mouse` (its options, measurements and calibration).
    pub fn camera_warp(&mut self) -> Option<&mut CameraWarp> {
        self.warp.as_mut()
    }

    /// The frame about to be presented at `now` (the clock of `mouse`), turned by the camera motion the mouse made
    /// since the moment it shows, into output slot `out`; present `out` then. False when nothing was written (the warp
    /// is off: no mouse input, a released mouse, a picture that does not follow the mouse, a turn under a quarter
    /// pixel): present the frame as it is. `out` may be the frame's own output slot; once written it is not turned
    /// again (a frame presented twice is turned again from where it came from).
    pub fn warp(&mut self, gpu: &Gpu, frame: Shown, out: usize, now: f64) -> Result<bool> {
        anyhow::ensure!(out < self.gen.outs.len(), "output slot {out} does not exist");
        let Some(w) = self.warp.as_mut() else { return Ok(false) };
        let t_frame = match frame {
            Shown::Input(slot) => self.in_pts.get(&slot),
            Shown::Output(slot) => self.out_pts.get(&slot),
        };
        let Some(&t_frame) = t_frame else { return Ok(false) };
        let Some(shift) = w.decide(gpu, t_frame, now) else { return Ok(false) };
        let dst = (out_key(out), &self.gen.outs[out].1);
        match frame {
            Shown::Input(slot) => {
                anyhow::ensure!(slot < self.gen.ins.len(), "input slot {slot} does not exist");
                w.render_into(gpu, (in_key(slot), &self.gen.ins[slot].1), dst, shift);
            }
            Shown::Output(slot) if slot != out => {
                anyhow::ensure!(slot < self.gen.outs.len(), "output slot {slot} does not exist");
                w.render_into(gpu, (out_key(slot), &self.gen.outs[slot].1), dst, shift);
            }
            Shown::Output(slot) => {
                let turned = w.render_shift(gpu, (out_key(slot), &self.gen.outs[slot].1), shift);
                let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                enc.copy_texture_to_texture(
                    turned.as_image_copy(),
                    self.gen.outs[out].0.as_image_copy(),
                    turned.size(),
                );
                gpu.queue.submit(Some(enc.finish()));
            }
        }
        // `out` now holds a turned frame: turning it again would turn it twice
        self.out_pts.remove(&out);
        self.counters.warped += 1;
        Ok(true)
    }

    pub fn counters(&self) -> &StreamCounters {
        &self.counters
    }

    pub fn report(&self) -> StreamReport {
        let o = &self.gen.opts;
        StreamReport {
            mode: self.opts.mode,
            cadence: self.gen.cadence(),
            width: o.w,
            height: o.h,
            internal_width: o.iw,
            internal_height: o.ih,
            k: o.k,
            repair: o.repair,
            inputs: self.gen.ins.len(),
            outputs: self.gen.outs.len(),
            pushed: self.pushed.len(),
            counters: self.counters.clone(),
            camera_warp: self.warp.as_ref().map(|w| w.report()),
        }
    }
}

// Bind-group ids of the camera warp: inputs and outputs in one space.
fn in_key(slot: usize) -> usize {
    2 * slot
}

fn out_key(slot: usize) -> usize {
    2 * slot + 1
}

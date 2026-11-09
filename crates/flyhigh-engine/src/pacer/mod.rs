//! The Pacer: a pure scheduler between source frames with timestamps and a display with ticks. The host feeds
//! events with its own clock (milliseconds, monotonic); the Pacer answers what to present on this tick and which
//! frames to generate for the coming ones. It never calls the GPU and never reads a clock, so a recorded trace
//! replays exactly ([`trace`]).
//!
//! Per display tick the host calls [`Pacer::display_tick`], presents `tick.present` (or nothing), generates
//! `tick.generate` through the Stream and reports each with [`Pacer::generated`]. Per source frame, once its kind is
//! known, the host calls [`Pacer::source_frame`], generates the requests it returns (mids of the previous pair that
//! must go out before the pair changes) and then pushes the frame into the Stream. Costs come from the Stream's
//! measurements through [`Pacer::cost_sample`].
pub mod control;
pub mod costs;
pub mod display;
pub mod plan;
pub mod queue;
pub mod rate;
pub mod source;
pub mod trace;

use crate::options::Options;
use control::{AutoController, ControlOptions, DropCharge};
use costs::RollingCost;
use display::DisplayEstimator;
pub use display::{DisplayMeasure, DISPLAY_RATES};
use plan::{plan_source_cadence, Kind, Slot};
pub use queue::Frame;
use queue::{Entry, PresentationQueue};
use rate::{
    auto_factor, fixed_factor, pair_gpu_cost, presentation_delay_ms, resolve_output_rate, AutoInputs, RateInputs,
    RateStabilizer,
};
pub use rate::{ClampReason, OutputPlan, RateMode, RateState};
use serde::{Deserialize, Serialize};
pub use source::COMMON_VIDEO_RATES;
use source::{target_needs_interpolation, SourceEstimator};
use std::collections::HashMap;
use trace::{Trace, TraceEvent};

pub use crate::stream::IntervalKind;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    Pair,
    Mid,
    Present,
}

/// A source frame as the host hands it over.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SourceFrame {
    /// The host's id for the frame (a slot index); comes back in [`Frame::Source`].
    pub id: u32,
    /// Media or capture time in milliseconds.
    pub pts_ms: f64,
    #[serde(default)]
    pub kind: IntervalKind,
    /// The pair's motion statistic (mean absolute difference, 0..255) for the motion-adaptive factor.
    #[serde(default)]
    pub motion: Option<f64>,
    /// Frames the host knows it missed before this one (a coalesced callback): no interpolation across the gap.
    #[serde(default)]
    pub skipped: u32,
}

/// A frame to generate: phase `t` of the pair (`previous`, `current`), wanted on screen at `due_ms`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Request {
    pub id: u64,
    pub previous: u32,
    pub current: u32,
    pub t: f32,
    pub due_ms: f64,
    /// The interval's real slot re-timed (cadence): `t` is 0 and the result replaces the queued source frame. The
    /// Stream may answer that the source is enough; the host then calls [`Pacer::skipped`].
    pub retime: bool,
    /// A probe: run so the cost sample teaches the controller, whatever the budget says.
    pub probe: bool,
}

/// What to show on this tick.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Present {
    pub frame: Frame,
    pub due_ms: f64,
    pub late_ms: f64,
    pub mid: bool,
    /// The generated frame was not there in time: this is its anchor instead.
    pub fallback: bool,
}

#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct Tick {
    pub present: Option<Present>,
    pub generate: Vec<Request>,
    /// The factor in force (for the host's UI).
    pub factor: u32,
    /// Frames skipped on this tick to show the newest due one.
    pub dropped: u32,
}

/// Every heuristic of the Pacer (ARCHITECTURE.md) with its default: what to present on a display tick and what to generate
/// for the coming ones. Times in milliseconds, rates in Hz. They may change live (`set_options`); the defaults are
/// tuned on the recorded sessions in `crates/flyhigh-engine/tests/traces/`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Options)]
#[serde(default, deny_unknown_fields)]
pub struct PacerOptions {
    /// How many frames to generate per source frame: a fixed factor (benchmarks and players that want exactly 2x),
    /// Auto (as many as the display and the GPU allow, capped by motion: whole factors where the display runs at a
    /// whole multiple of the source rate, else every tick or every nth tick of the display at its own media time,
    /// so 24 fps on a 60 Hz display is 60 even pictures and not 48 uneven ones), the display's rate, or a target
    /// rate.
    #[option(
        values = "Auto, Factor(n), DisplayHz, Target(fps)",
        c = "mode (0 auto, 1 factor, 2 display Hz, 3 target), factor, target_fps"
    )]
    pub mode: RateMode,
    /// The ceiling of Auto.
    #[option(values = "1 to 6")]
    pub max_factor: u32,
    /// Auto capped at this many presentations per second: a strict ceiling, never raised to the 2x floor. Off = the
    /// display's rate.
    #[option(values = "off, or a rate", c = "fps_limit (0 = off)")]
    pub fps_limit: Option<f64>,
    /// The generated interval is one source frame back: the Stream's video mode with its lookahead, where the real
    /// slot may still be re-timed.
    #[option(values = "on, off", c = "lookahead (0, 1)")]
    pub lookahead: bool,
    /// The measured refresh rate is a ceiling on the output rate. Off lets the GPU generate above it; the extra frames
    /// are superseded, not dropped.
    #[option(values = "on, off", c = "limit_to_display (0, 1)")]
    pub limit_to_display: bool,
    /// The exact modes use the whole rate of a host that cannot present on every tick, instead of leaving the headroom
    /// (a display's own rate is used whole either way).
    #[option(values = "on, off", c = "fill_display (0, 1)")]
    pub fill_display: bool,
    /// Fraction of the presentation rate Auto's factors, a capped Auto and a host with a present limit may use; the
    /// rest absorbs hitches. A display's own rate and its whole fractions are used exactly (every tick one frame).
    #[option(values = "0.9 to 1")]
    pub display_headroom: f64,
    /// Fraction of the source interval the GPU may spend per interval; the factor is capped so the measured costs fit.
    #[option(values = "0 to 1")]
    pub compute_budget: f64,
    /// The least presentation delay: how far ahead of their arrival source frames are shown, so that their in-betweens
    /// have time to be generated.
    pub delay_floor_ms: f64,
    /// For games and anything driven by a hand on a mouse: the presentation delay is the least that still lets the
    /// in-betweens be generated, (1 - 1/N) of a source interval at factor N plus the measured pair and frame cost,
    /// and the arrival jitter, instead of the video window (`delay_floor_ms`) with its 25 ms reserve. At
    /// 60 fps that is 10 to 15 ms instead of 60. Auto and a fixed factor; the exact modes keep their own delay.
    #[option(values = "on, off", c = "low_latency (0, 1)")]
    pub low_latency: bool,
    /// The most presentation delay the controller may grow to (the exact modes may grow further, to 2500).
    pub delay_max_ms: f64,
    /// How late a frame may still be shown, in output intervals; later ones are dropped for the newest due one.
    pub lateness_intervals: f64,
    /// Pending presentations the queue holds.
    pub queue_capacity: usize,
    /// Memory of the drop pressure integrator: how long a drop keeps weighing.
    pub pressure_tau_ms: f64,
    /// Pressure that costs Auto one step of factor.
    pub penalty_up: f64,
    /// Pressure that costs two steps at once.
    pub penalty_burst: f64,
    /// The least time between two penalties.
    pub penalty_min_gap_ms: f64,
    /// The most steps Auto may be penalised below its ceiling.
    pub penalty_max: u32,
    /// Clean time (pressure below 0.15) before Auto recovers one step.
    pub recovery_after_ms: f64,
    /// Drops within this window after a host stall are not charged to the GPU.
    pub stall_window_ms: f64,
    /// Stalls per 2 s above which they stop being exempt: that often, the GPU is the problem after all.
    pub stall_rate_limit: f64,
    /// Drops exempt after a mode change (fullscreen, resize): the reflow stall is not a GPU limit (the extension's
    /// measured fullscreen reflow).
    pub settle_ms: f64,
    /// Fewer in-betweens on fast scenes, where they are worth less and cost the same. On a display that runs at a whole
    /// multiple of the source rate the limit rounds up to a factor that lands evenly on its ticks.
    #[option(values = "on, off", c = "motion_adaptive (0, 1)")]
    pub motion_adaptive: bool,
    /// The host's present spacing limit: presentations closer than this are held to it. The Firefox adapter sets 7
    /// (its compositor stalls 40 to 90 ms below that); 0 = none.
    pub min_present_interval_ms: f64,
    /// Media seconds per wall second (a player's speed).
    pub playback_rate: f64,
    /// The GPU cost assumed per generated frame until the first measurement (10 ms at 848 x 480; scale with the
    /// output resolution).
    pub mid_cost_seed_ms: f64,
    /// The pair cost assumed until the first measurement.
    pub pair_cost_seed_ms: f64,
    /// The host measures GPU time with timestamp queries: cost probes are only useful when their cost comes back. Off =
    /// the seeds stay.
    #[option(values = "on, off", c = "gpu_timestamps (0, 1)")]
    pub gpu_timestamps: bool,
}

impl Default for PacerOptions {
    fn default() -> Self {
        PacerOptions {
            mode: RateMode::Auto,
            max_factor: 6,
            fps_limit: None,
            lookahead: false,
            limit_to_display: true,
            fill_display: false,
            display_headroom: 0.97,
            compute_budget: 0.9,
            delay_floor_ms: 60.0,
            low_latency: false,
            delay_max_ms: 180.0,
            lateness_intervals: 3.0,
            queue_capacity: 24,
            pressure_tau_ms: 300.0,
            penalty_up: 1.2,
            penalty_burst: 3.0,
            penalty_min_gap_ms: 500.0,
            penalty_max: 3,
            recovery_after_ms: 6000.0,
            stall_window_ms: 20.0,
            stall_rate_limit: 4.0,
            settle_ms: 1200.0,
            motion_adaptive: true,
            min_present_interval_ms: 0.0,
            playback_rate: 1.0,
            mid_cost_seed_ms: 10.0,
            pair_cost_seed_ms: 4.0,
            gpu_timestamps: true,
        }
    }
}

pub use crate::options::OptionsError;

impl PacerOptions {
    /// The presentation delay a reset starts from: the video floor, or with `low_latency` at most 30 ms (its working
    /// point at 30 to 60 fps sources lies below; the controller settles there within a few frames).
    pub fn start_delay_ms(&self) -> f64 {
        if self.low_latency {
            self.delay_floor_ms.min(30.0)
        } else {
            self.delay_floor_ms
        }
    }

    pub fn validate(&self) -> Result<(), OptionsError> {
        let err = |field, reason: String| Err(OptionsError { field, reason });
        match self.mode {
            RateMode::Factor(f) if !(1..=6).contains(&f) => return err("mode", format!("factor {f} is not 1..=6")),
            RateMode::Target(fps) if rate::target_fps(fps).is_none() => {
                return err("mode", format!("target {fps} is not a positive rate"))
            }
            _ => {}
        }
        if !(1..=6).contains(&self.max_factor) {
            return err("max_factor", format!("{} is not 1..=6", self.max_factor));
        }
        if let Some(l) = self.fps_limit {
            if rate::target_fps(l).is_none() {
                return err("fps_limit", format!("{l} is not a positive rate"));
            }
        }
        if !(self.display_headroom > 0.0 && self.display_headroom <= 1.0) {
            return err("display_headroom", format!("{} is not in (0, 1]", self.display_headroom));
        }
        if !(self.compute_budget > 0.0 && self.compute_budget <= 1.0) {
            return err("compute_budget", format!("{} is not in (0, 1]", self.compute_budget));
        }
        let positive = |v: f64| v.is_finite() && v > 0.0;
        let non_negative = |v: f64| v.is_finite() && v >= 0.0;
        if !non_negative(self.delay_floor_ms) || !non_negative(self.delay_max_ms - self.delay_floor_ms) {
            return err("delay_max_ms", "must be at least delay_floor_ms, both non-negative".into());
        }
        if !non_negative(self.lateness_intervals) {
            return err("lateness_intervals", "must be non-negative".into());
        }
        if self.queue_capacity < 2 {
            return err("queue_capacity", "must hold at least 2 presentations".into());
        }
        for (field, v) in [
            ("pressure_tau_ms", self.pressure_tau_ms),
            ("penalty_up", self.penalty_up),
            ("penalty_burst", self.penalty_burst),
            ("recovery_after_ms", self.recovery_after_ms),
            ("playback_rate", self.playback_rate),
            ("mid_cost_seed_ms", self.mid_cost_seed_ms),
        ] {
            if !positive(v) {
                return err(field, format!("{v} must be positive"));
            }
        }
        for (field, v) in [
            ("penalty_min_gap_ms", self.penalty_min_gap_ms),
            ("stall_window_ms", self.stall_window_ms),
            ("stall_rate_limit", self.stall_rate_limit),
            ("settle_ms", self.settle_ms),
            ("min_present_interval_ms", self.min_present_interval_ms),
            ("pair_cost_seed_ms", self.pair_cost_seed_ms),
        ] {
            if !non_negative(v) {
                return err(field, format!("{v} must be non-negative"));
            }
        }
        Ok(())
    }

    fn control(&self) -> ControlOptions {
        ControlOptions {
            pressure_tau_ms: self.pressure_tau_ms,
            penalty_up: self.penalty_up,
            penalty_burst: self.penalty_burst,
            penalty_min_gap_ms: self.penalty_min_gap_ms,
            penalty_max: self.penalty_max,
            recovery_after_ms: self.recovery_after_ms,
            recovery_below: 0.15,
            stall_window_ms: self.stall_window_ms,
            stall_rate_limit: self.stall_rate_limit,
            strain_threshold: 1.7,
            strain_release_ms: 500.0,
            probe_initial_ms: 2500.0,
            probe_max_ms: 20000.0,
            probe_attempts: 4,
        }
    }

    /// Exact modes place presentations on an output grid; a capped Auto is one too.
    fn exact_cadence(&self) -> bool {
        self.mode.exact() || (self.mode == RateMode::Auto && self.fps_limit.is_some())
    }
}

/// Exact modes let the delay grow to a whole lookahead interval.
const EXACT_DELAY_MAX_MS: f64 = 2500.0;
/// Every change of Auto's factor re-times the in-betweens, and the motion ceiling alone used to change it three to five
/// times a second over game footage, each change a visible hitch (measured: 60 % of the off-pace ticks lay next to
/// one). A change the factor in use does not force waits until Auto has asked for it `AUTO_ASK_MS` and the last change
/// is `AUTO_HOLD_MS` old; only a factor that no longer fits gives way at once.
const AUTO_ASK_MS: f64 = 250.0;
/// Source frames in a row that must agree before Auto moves between factors and the exact grid (see `auto_grid_hz`).
const AUTO_GRID_VOTES: u32 = 4;
/// The least rate of Auto's grid as a multiple of the source rate. A factor needs twice the source rate; the grid does
/// not: 60 fps on a 75 Hz display shows every fourth frame twice, and 75 pictures at their own media time are even.
const AUTO_GRID_MIN_RATIO: f64 = 1.15;
const AUTO_HOLD_MS: f64 = 1000.0;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Counters {
    pub ticks: u64,
    pub sources: u64,
    pub presents: u64,
    pub presented_mids: u64,
    pub presented_sources: u64,
    pub requests: u64,
    /// Frames skipped to show a newer due one, with the display as the ceiling.
    pub drops: u64,
    pub drops_stall_exempt: u64,
    pub drops_settle_exempt: u64,
    /// Frames due before the last presented one (a resynced grid).
    pub stale: u64,
    /// Frames the compositor could not show while generating above the refresh rate.
    pub superseded: u64,
    /// Entries pushed out of a full queue.
    pub evicted: u64,
    /// Generated frames that were not there in time (the anchor was shown).
    pub late_fallbacks: u64,
    /// Presents held back by the host's present spacing limit.
    pub held_by_host_limit: u64,
    pub cuts: u64,
    pub holds: u64,
    pub stalls: u64,
    pub host_stalls: u64,
    pub probes: u64,
    pub rate_transitions: u64,
    pub rate_plan_resets: u64,
    pub history_breaks: u64,
    pub pairs_skipped_gpu: u64,
    pub resets: u64,
}

/// Everything the Pacer decided and measured.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PacerReport {
    pub display: DisplayMeasure,
    pub display_floor_ms: f64,
    pub display_ready: bool,
    pub source_interval_ms: f64,
    pub source_hz: Option<f64>,
    pub unique_interval_ms: f64,
    pub source_ready: bool,
    pub mode: RateMode,
    pub plan: Option<OutputPlan>,
    pub factor: u32,
    pub penalty: u32,
    pub strain_penalty: u32,
    pub pressure: f64,
    pub strain_pressure: f64,
    pub stall_load: f64,
    pub settle_active: bool,
    pub delay_ms: f64,
    pub late_avg_ms: f64,
    pub late_tail_ms: f64,
    pub motion_avg: f64,
    pub pair_cost_ms: f64,
    pub mid_cost_ms: f64,
    pub present_cost_ms: f64,
    pub queued: usize,
    /// Source frames a host must keep to serve the presentation delay.
    pub history_frames: u32,
    pub counters: Counters,
}

impl PacerReport {
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }
}

#[derive(Clone, Copy, Debug)]
struct Prev {
    id: u32,
    /// Whether the interval ending at this frame may be interpolated.
    interpolate: bool,
    /// How long the interval ending at this frame is on the schedule clock.
    span_ms: f64,
}

#[derive(Debug, Clone)]
struct Job {
    requests: Vec<Request>,
    next: usize,
}

#[derive(Debug, Clone, Copy)]
enum Pending {
    Mid,
    Retime { source: u32 },
}

#[derive(Debug, Clone)]
pub struct Pacer {
    opts: PacerOptions,
    control_opts: ControlOptions,
    display: DisplayEstimator,
    source: SourceEstimator,
    pair_cost: RollingCost,
    mid_cost: RollingCost,
    present_cost: RollingCost,
    control: AutoController,
    queue: PresentationQueue,
    rate: RateStabilizer,
    last_plan: Option<OutputPlan>,
    gpu_probe_started: Option<f64>,
    gpu_probe_last: f64,
    recording: Option<Trace>,
    grid_next_at: f64,
    grid_phase_ms: f64,
    bridge: bool,
    delay_ms: f64,
    late_avg: f64,
    late_tail_ms: f64,
    late_tail_t: f64,
    motion_avg: f64,
    factor: u32,
    /// Since when Auto has asked for a higher (true) or a lower factor than the one in use.
    auto_candidate: Option<(bool, f64)>,
    auto_changed_at: f64,
    /// Auto runs on the exact grid (see `auto_grid_hz`).
    auto_exact: bool,
    auto_exact_votes: u32,
    /// The latest time any event carried: what the report's `settle_active` is judged at before the first tick.
    seen: f64,
    last_tick: Option<f64>,
    last_present_wall: f64,
    last_pts: Option<f64>,
    prev: Option<Prev>,
    prev2: Option<Prev>,
    job: Option<Job>,
    next_request: u64,
    pending: HashMap<u64, Pending>,
    counters: Counters,
}

impl Pacer {
    pub fn new(opts: &PacerOptions) -> Result<Pacer, OptionsError> {
        opts.validate()?;
        let control_opts = opts.control();
        let mut source = SourceEstimator::new();
        source.set_playback_rate(opts.playback_rate);
        Ok(Pacer {
            opts: opts.clone(),
            control_opts,
            display: DisplayEstimator::new(opts.min_present_interval_ms),
            source,
            pair_cost: RollingCost::default(),
            mid_cost: RollingCost::default(),
            present_cost: RollingCost::default(),
            control: AutoController::new(0.0, &control_opts),
            queue: PresentationQueue::new(opts.queue_capacity),
            rate: RateStabilizer::default(),
            last_plan: None,
            gpu_probe_started: None,
            gpu_probe_last: f64::NEG_INFINITY,
            recording: None,
            grid_next_at: 0.0,
            grid_phase_ms: 0.0,
            bridge: false,
            delay_ms: opts.start_delay_ms(),
            late_avg: 0.0,
            late_tail_ms: 0.0,
            late_tail_t: 0.0,
            motion_avg: 0.0,
            factor: 2,
            auto_candidate: None,
            auto_changed_at: f64::NEG_INFINITY,
            auto_exact: false,
            auto_exact_votes: 0,
            seen: f64::NEG_INFINITY,
            last_tick: None,
            last_present_wall: f64::NEG_INFINITY,
            last_pts: None,
            prev: None,
            prev2: None,
            job: None,
            next_request: 1,
            pending: HashMap::new(),
            counters: Counters::default(),
        })
    }

    pub fn options(&self) -> &PacerOptions {
        &self.opts
    }

    /// Changes options live. A new mode restarts the controller and the output grid (queued frames drain).
    pub fn set_options(&mut self, now: f64, opts: &PacerOptions) -> Result<(), OptionsError> {
        self.log(TraceEvent::Options { t: now, options: Box::new(opts.clone()) });
        opts.validate()?;
        let mode_changed = opts.mode != self.opts.mode || opts.fps_limit != self.opts.fps_limit;
        self.opts = opts.clone();
        self.control_opts = opts.control();
        self.display.set_host_min_present(opts.min_present_interval_ms);
        self.source.set_playback_rate(opts.playback_rate);
        self.queue.set_capacity(opts.queue_capacity);
        if mode_changed {
            self.control.reset(now, &self.control_opts);
            self.delay_ms = opts.start_delay_ms();
            self.reset_output_cadence(true);
        }
        Ok(())
    }

    /// A seek, a decoder gap, a new source: forget the history, keep the estimators' knowledge of the display.
    pub fn reset(&mut self, now: f64) {
        self.log(TraceEvent::Reset { t: now });
        self.counters.resets += 1;
        self.source.reset();
        self.control.reset(now, &self.control_opts);
        self.reset_output_cadence(true);
        self.queue.reset();
        self.delay_ms = self.opts.start_delay_ms();
        self.late_avg = 0.0;
        self.late_tail_ms = 0.0;
        self.motion_avg = 0.0;
        self.auto_candidate = None;
        self.auto_changed_at = f64::NEG_INFINITY;
        self.last_pts = None;
        self.prev = None;
        self.prev2 = None;
        self.last_present_wall = f64::NEG_INFINITY;
    }

    /// A fullscreen change, a resize: the reflow stalls ticks and delivery, which is not a GPU limit. Drops are
    /// exempt for `settle_ms`, the controller and the cost estimates restart.
    pub fn mode_change(&mut self, now: f64) {
        self.log(TraceEvent::ModeChange { t: now });
        self.seen = self.seen.max(now);
        self.control.settle(now, self.opts.settle_ms);
        self.control.reset(now, &self.control_opts);
        self.mid_cost.reset();
        self.pair_cost.reset();
    }

    /// The host's loop stood still for `ms` (the Pacer also infers stalls from tick gaps).
    pub fn host_stall(&mut self, now: f64, ms: f64) {
        self.log(TraceEvent::Stall { t: now, ms });
        self.counters.host_stalls += 1;
        self.control.stall(now);
    }

    /// A measured GPU cost (timestamp queries) of one stage. Returns whether it was taken: samples inside the settle
    /// window after a mode change are not (the reflow inflates them), and a host measuring once should measure again.
    pub fn cost_sample(&mut self, now: f64, stage: Stage, ms: f64) -> bool {
        self.log(TraceEvent::Cost { t: now, stage, ms });
        if !self.control.settled(now) {
            return false;
        }
        match stage {
            Stage::Pair => self.pair_cost.push(ms),
            Stage::Mid => {
                self.mid_cost.push(ms);
                self.gpu_probe_started = None;
            }
            Stage::Present => self.present_cost.push(ms),
        }
        true
    }

    /// The host generated (submitted) the frame of a request.
    pub fn generated(&mut self, _now: f64, request: u64, frame: Frame) {
        match self.pending.remove(&request) {
            Some(Pending::Mid) => {
                self.queue.fill(request, frame);
            }
            Some(Pending::Retime { source }) => {
                // the source entry may be gone already (presented late): then it stays what it was
                self.queue.replace_source(source, frame);
            }
            None => {}
        }
    }

    /// The Stream answered that the source frame is enough for a request (a re-time that was not needed).
    pub fn skipped(&mut self, request: u64) {
        if let Some(Pending::Mid) = self.pending.remove(&request) {
            self.queue.resolve_fallback(request);
        }
    }

    pub fn report(&self) -> PacerReport {
        let wall_pair = self.source.wall_pair_ms();
        PacerReport {
            display: self.display.measure(),
            display_floor_ms: self.display.floor_ms(),
            display_ready: self.display.ready(),
            source_interval_ms: self.source.decoded_ms(),
            source_hz: self.source.source_hz(),
            unique_interval_ms: self.source.unique_ms(),
            source_ready: self.source.ready(),
            mode: self.opts.mode,
            plan: self.last_plan.clone(),
            factor: self.factor,
            penalty: self.control.durable_penalty(),
            strain_penalty: self.control.strain_penalty(),
            pressure: self.control.pressure(),
            strain_pressure: self.control.strain_pressure(),
            stall_load: self.control.stall_load(),
            settle_active: !self.control.settled(self.seen),
            delay_ms: self.delay_ms,
            late_avg_ms: self.late_avg,
            late_tail_ms: self.late_tail_ms,
            motion_avg: self.motion_avg,
            pair_cost_ms: self.pair_cost.value(),
            mid_cost_ms: self.mid_cost.value(),
            present_cost_ms: self.present_cost.value(),
            queued: self.queue.len(),
            history_frames: (self.delay_ms / wall_pair.max(1.0)).ceil() as u32 + 3,
            counters: self.counters.clone(),
        }
    }

    /// Records every call from now on into a `trace v1` named after `host`, with the options in force: the session a
    /// bug report needs, replayable bit for bit (`trace::replay`, `flyhigh-dev replay`).
    pub fn record(&mut self, host: &str) {
        let mut t = Trace::new(host);
        t.options = Some(self.opts.clone());
        self.recording = Some(t);
    }

    /// The recording so far (None when `record` was not called).
    pub fn trace(&self) -> Option<&Trace> {
        self.recording.as_ref()
    }

    /// Ends the recording and hands it over.
    pub fn stop_recording(&mut self) -> Option<Trace> {
        self.recording.take()
    }

    fn log(&mut self, e: TraceEvent) {
        if let Some(t) = &mut self.recording {
            t.events.push(e);
        }
    }

    pub fn counters(&self) -> &Counters {
        &self.counters
    }

    pub fn factor(&self) -> u32 {
        self.factor
    }

    pub fn penalty(&self) -> u32 {
        self.control.penalty()
    }

    pub fn delay_ms(&self) -> f64 {
        self.delay_ms
    }

    fn mid_cost(&self) -> f64 {
        self.mid_cost.or(self.opts.mid_cost_seed_ms)
    }

    fn pair_cost(&self) -> f64 {
        self.pair_cost.or(self.opts.pair_cost_seed_ms)
    }

    fn enqueue(&mut self, e: Entry) {
        if self.queue.enqueue(e).is_some() {
            self.counters.evicted += 1;
            self.counters.drops += 1;
        }
    }

    fn enqueue_source(&mut self, id: u32, at: f64) {
        self.enqueue(Entry {
            due: at,
            frame: Some(Frame::Source(id)),
            fallback: Frame::Source(id),
            request: None,
            mid: false,
        });
    }

    fn reset_output_cadence(&mut self, clear_queue: bool) {
        self.grid_next_at = 0.0;
        self.grid_phase_ms = 0.0;
        self.rate.reset();
        self.last_plan = None;
        self.job = None;
        if clear_queue {
            self.queue.clear();
            self.pending.clear();
        }
    }

    fn presentation_target_hz(&self) -> f64 {
        if self.exact() {
            return self.rate.snapshot().and_then(|p| p.output_hz).unwrap_or(0.0);
        }
        // fixed factors and uncapped Auto place mids at k / n across the interval: target 0 disables recovery and
        // drops a one-vsync backlog even when the display keeps up with that rate
        // by the decoded cadence once it is known: the arrival average wanders (16.3 to 16.6 ms on a 60 fps video in a
        // browser), which read x4 as 241 to 247 fps, more than a 240 Hz display has, and the backlog was dropped
        let iv = if self.source.ready() { self.source.wall_pair_ms() } else { self.source.wall_ms() };
        if iv > 0.0 {
            self.factor.max(1) as f64 * 1000.0 / iv
        } else {
            0.0
        }
    }

    /// The tick lock. A frame due a whole number n of ticks after the one before it is not shown sooner than n ticks
    /// after that one was shown. Due times carry a few milliseconds of jitter (a browser delivers a 24 fps video up to
    /// 20 ms early or late frame by frame, and the schedule clock passes part of that on); on the plain rule, the
    /// first tick past the due time, every crossing of a tick boundary left one picture up for three ticks and the
    /// next for one: x5 of 24 fps on 240 Hz, 5 % of the pictures off pace on a pan. Held to its n ticks, a frame that
    /// came a tick late is followed in step, the lateness sits a tick higher, and jitter no longer reaches a boundary.
    /// The due time stays the long-term reference: past a tick and the jitter's reach late, the frame goes out at
    /// once, the one slip two drifting clocks force. Spacings that are no whole number of ticks (72 fps on 240 Hz),
    /// the display's own rate and a backlog keep the plain rule.
    fn tick_lock(&self, now: f64, due: Option<usize>) -> Option<usize> {
        if due != Some(0) || !self.display.ready() {
            return due;
        }
        let head = &self.queue.entries()[0];
        let tick = self.display.floor_ms();
        let spacing = head.due - self.queue.last_presented_due();
        let n = (spacing / tick).round();
        if !(spacing.is_finite() && (2.0..=8.0).contains(&n) && (spacing - n * tick).abs() <= 0.25 * tick) {
            return due;
        }
        let elapsed = ((now - self.last_present_wall) / tick).round();
        let reach = tick + tick.clamp(4.0, 8.0);
        due.filter(|_| elapsed >= n || now - head.due > reach)
    }

    /// The display tick: the heartbeat.
    pub fn display_tick(&mut self, now: f64) -> Tick {
        self.log(TraceEvent::Tick { t: now });
        self.seen = self.seen.max(now);
        self.counters.ticks += 1;
        let mut sample = None;
        if let Some(last) = self.last_tick {
            let d = now - last;
            sample = self.display.sample(d);
            if d > 25f64.max(self.display.floor_ms() * 4.0) {
                self.counters.stalls += 1;
                self.control.stall(now);
            } else if self.display.ready() && d > self.display.floor_ms() * 1.5 {
                // a missed vsync: at the display's own rate the frame due on the lost tick has nowhere to go, and
                // that is the compositor's, not the GPU's (the stall budget still charges a run of them)
                self.control.stall(now);
            }
        }
        self.last_tick = Some(now);

        let stale = self.queue.remove_stale();
        self.counters.stale += stale as u64;
        if self.opts.limit_to_display {
            self.counters.drops += stale as u64;
        }
        let target_hz = self.presentation_target_hz();
        let capacity_hz = self.display.measure().capacity_hz;
        let selection = self.queue.select_due(now, target_hz, capacity_hz, self.opts.lateness_intervals);
        let mut due = self.tick_lock(now, selection.index);
        if due.is_some() && now - self.last_present_wall < self.opts.min_present_interval_ms {
            due = None;
            self.counters.held_by_host_limit += 1;
        }
        self.control.decay(now, self.opts.pressure_tau_ms);

        let mut present = None;
        let mut dropped = 0;
        if let Some(index) = due {
            dropped = index as u32;
            if self.opts.limit_to_display {
                self.counters.drops += index as u64;
                if index == 1 && self.at_display_rate() {
                    // one frame too many at the display's own rate is the source running a little fast or a clock
                    // slip: the picture skips one tick either way, and it is no sign of the GPU (the stall budget still
                    // charges a run of them)
                    self.control.stall(now);
                }
                match self.control.charge_drops(now, index, &self.control_opts) {
                    DropCharge::Charged => {}
                    DropCharge::StallExempt => self.counters.drops_stall_exempt += index as u64,
                    DropCharge::SettleExempt => self.counters.drops_settle_exempt += index as u64,
                }
            } else {
                self.counters.superseded += index as u64;
            }
            let entry = self.queue.present(index);
            let late = now - entry.due;
            if self.control.settled(now) {
                // lateness: learn fast, forget slowly; the tail attacks instantly and decays over ~700 ms so the
                // delay target can pre-buffer delivery bursts
                self.late_avg = if late > self.late_avg {
                    self.late_avg * 0.7 + late * 0.3
                } else {
                    self.late_avg * 0.985 + late * 0.015
                };
                self.late_tail_ms = late.max(self.late_tail_ms * libm::exp((self.late_tail_t - now) / 700.0));
                self.late_tail_t = now;
            }
            let (frame, fallback) = match entry.frame {
                Some(f) => (f, false),
                None => {
                    self.counters.late_fallbacks += 1;
                    if let Some(r) = entry.request {
                        self.pending.remove(&r);
                    }
                    (entry.fallback, true)
                }
            };
            let mid = entry.mid && !fallback;
            self.last_present_wall = now;
            self.counters.presents += 1;
            if mid {
                self.counters.presented_mids += 1;
            } else {
                self.counters.presented_sources += 1;
            }
            present = Some(Present { frame, due_ms: entry.due, late_ms: late, mid, fallback });
        }

        let generate = self.drive_job(now);
        if self.opts.mode == RateMode::Auto && self.control.settled(now) {
            if let Some(d) = sample {
                if self.display.floor_ms() < 90.0 {
                    self.control.charge_strain(now, d, self.display.floor_ms(), &self.control_opts);
                }
            }
            self.control.step(now, &self.control_opts);
        }
        Tick { present, generate, factor: self.factor, dropped }
    }

    /// Just-in-time requests: a mid is asked for `lead` before its slot, so present blits interleave with computes
    /// instead of the first mid waiting for the whole batch.
    fn drive_job(&mut self, now: f64) -> Vec<Request> {
        let lead = self.pair_cost() + 2.0 * (self.mid_cost() + self.present_cost.value()) + 8.0;
        let mut out = Vec::new();
        let Some(job) = &mut self.job else { return out };
        while job.next < job.requests.len() && job.requests[job.next].due_ms - now <= lead {
            let r = job.requests[job.next];
            job.next += 1;
            out.push(r);
        }
        if job.next >= job.requests.len() {
            self.job = None;
        }
        for r in &out {
            self.issue(*r);
        }
        out
    }

    fn issue(&mut self, r: Request) {
        self.counters.requests += 1;
        if r.retime {
            self.pending.insert(r.id, Pending::Retime { source: r.previous });
            return;
        }
        let anchor = if r.t < 0.5 { r.previous } else { r.current };
        self.pending.insert(r.id, Pending::Mid);
        self.enqueue(Entry {
            due: r.due_ms,
            frame: None,
            fallback: Frame::Source(anchor),
            request: Some(r.id),
            mid: true,
        });
    }

    /// Leftovers of the previous pair go out before the pair changes.
    fn flush_job(&mut self) -> Vec<Request> {
        let Some(job) = self.job.take() else { return Vec::new() };
        let rest: Vec<Request> = job.requests[job.next..].to_vec();
        for r in &rest {
            self.issue(*r);
        }
        rest
    }

    fn request(&mut self, previous: u32, current: u32, t: f64, due_ms: f64, retime: bool, probe: bool) -> Request {
        let id = self.next_request;
        self.next_request += 1;
        Request { id, previous, current, t: t as f32, due_ms, retime, probe }
    }

    fn rate_plan(&mut self, now: f64) -> OutputPlan {
        let source_hz = self.source.source_hz();
        let (mid, pair, present) = (self.mid_cost(), self.pair_cost(), self.present_cost.value());
        let capped = self.opts.mode == RateMode::Auto && self.opts.fps_limit.is_some();
        let mode = match (self.opts.mode, self.opts.fps_limit) {
            (RateMode::Auto, None) if self.auto_exact => {
                let display_hz = self.display.measure().display_hz;
                RateMode::Target(self.auto_grid_hz().unwrap_or(display_hz))
            }
            (RateMode::Auto, Some(limit)) => {
                // capped Auto: the Auto policy's presentation rate, no higher than the limit
                let requested = match source_hz {
                    Some(hz) => {
                        let policy = self.auto_policy(hz, self.opts.motion_adaptive);
                        if policy.runnable {
                            limit.min(policy.presentation_hz)
                        } else {
                            limit.min(hz)
                        }
                    }
                    None => limit,
                };
                RateMode::Target(requested)
            }
            (m, _) => m,
        };
        let queue_capacity_hz =
            if self.opts.limit_to_display { f64::INFINITY } else { self.queue.capacity_hz(self.delay_ms) };
        let inputs = RateInputs {
            mode,
            strict_ceiling: capped,
            display: self.display.measure(),
            display_ready: self.display.ready(),
            source_hz,
            mid_cost_ms: mid,
            pair_cost_ms: pair,
            present_cost_ms: present,
            fill_display: self.opts.fill_display,
            limit_to_display: self.opts.limit_to_display,
            queue_capacity_hz,
            headroom: self.opts.display_headroom,
            compute_budget: self.opts.compute_budget,
            max_mids: self.opts.queue_capacity - 1,
            min_ratio: if self.auto_exact { AUTO_GRID_MIN_RATIO } else { 2.0 },
        };
        let mut plan = resolve_output_rate(&inputs);
        // the GPU looks too slow for 2x: one real pair with tiny assumed costs every 2.5 s measures it, while a
        // probe is pending for at most two source intervals
        let probe_timeout = source_hz.map_or(1500.0, |hz| (2.0 * 1000.0 / hz + 250.0).max(1500.0));
        if let Some(started) = self.gpu_probe_started {
            if !self.opts.gpu_timestamps || now - started > probe_timeout {
                self.gpu_probe_started = None;
            }
        }
        if self.gpu_probe_started.is_none()
            && self.opts.gpu_timestamps
            && plan.state == RateState::NoTwoXGpu
            && now - self.gpu_probe_last >= 2500.0
        {
            self.gpu_probe_started = Some(now);
            self.gpu_probe_last = now;
            self.counters.probes += 1;
        }
        if self.gpu_probe_started.is_some() {
            let minimum = plan.minimum_hz.unwrap_or(0.0);
            let mut probe = resolve_output_rate(&RateInputs {
                mode: RateMode::Target(minimum),
                strict_ceiling: false,
                mid_cost_ms: 0.1,
                pair_cost_ms: 0.1,
                ..inputs
            });
            probe.requested_hz = plan.requested_hz;
            probe.state = RateState::ProbingGpu;
            probe.clamp = Some(ClampReason::GpuProbe);
            probe.warning = Some("Checking GPU capacity at 2x".into());
            probe.probe = true;
            plan = probe;
        }
        let (plan, transitioned) = self.rate.sync(plan);
        if transitioned {
            // scheduled frames stay valid under the old grid and drain; future planning switches to the new clock,
            // and the just-arrived frame gets an arrival-time slot
            self.counters.rate_plan_resets += 1;
            self.bridge = true;
            self.grid_next_at = 0.0;
            self.grid_phase_ms = 0.0;
            self.job = None;
        }
        self.last_plan = Some(plan.clone());
        plan
    }

    /// Whether every display tick shows a frame of its own: the factor times the source rate is the display's rate.
    fn at_display_rate(&self) -> bool {
        let display_hz = self.display.measure().display_hz;
        self.source
            .source_hz()
            .is_some_and(|hz| (self.factor as f64 * hz - display_hz).abs() <= display_hz * rate::CADENCE_TOLERANCE)
    }

    fn exact(&self) -> bool {
        self.opts.exact_cadence() || self.auto_exact
    }

    /// An output rate that is the display's or a whole fraction of it, as the measured tick period has it: a grid one
    /// part in a thousand off the ticks slips one every four seconds at 240 Hz. A rate that is a whole multiple of the
    /// source stays as it is, so its ticks keep landing on the source frames.
    fn on_ticks(&self, output_hz: f64) -> f64 {
        let tick_hz = 1000.0 / self.display.period_ms();
        let k = (tick_hz / output_hz).round();
        let of_source = self
            .source
            .source_hz()
            .is_some_and(|s| ((output_hz / s).round() * s - output_hz).abs() <= 0.005 * output_hz);
        if !self.display.ready() || of_source || k < 1.0 || (tick_hz / k - output_hz).abs() > 0.015 * output_hz {
            output_hz
        } else {
            tick_hz / k
        }
    }

    /// The exact output rate Auto runs at, or None where factors serve. Factors put in-betweens at k / n between two
    /// frames, even on screen only when the display runs at a whole multiple of a regular source; 24 fps on a 60 Hz
    /// display as x2 is 48 pictures a second on 60 ticks, a quarter of them held twice. There Auto takes the grid in
    /// media time instead: the display's rate, or the whole fraction of it that the GPU, the motion and the drop
    /// penalty leave, down to 1.15 times the source. A host that cannot present on every tick keeps the factors.
    fn auto_grid_hz(&self) -> Option<f64> {
        let source_hz = self.source.source_hz()?;
        let d = self.display.measure();
        if !d.measured || d.capacity_hz < d.display_hz * 0.97 {
            return None;
        }
        if self.source.regular() && rate::display_multiple(source_hz, d.display_hz).is_some() {
            return None;
        }
        let interval = 1000.0 / source_hz;
        let by_gpu = 1.0 + ((0.85 * interval - self.pair_cost()) / self.mid_cost().max(0.01)).max(0.0);
        let by_motion = if !self.opts.motion_adaptive {
            f64::INFINITY
        } else if self.motion_avg > 45.0 {
            2.5
        } else if self.motion_avg > 28.0 {
            3.75
        } else if self.motion_avg > 16.0 {
            5.0
        } else {
            f64::INFINITY
        };
        let limit = by_gpu.min(by_motion).min(self.opts.max_factor as f64 + 1.0);
        let least = AUTO_GRID_MIN_RATIO * source_hz;
        let rates: Vec<f64> = (1..=8).map(|k| d.display_hz / k as f64).filter(|r| *r >= least).collect();
        let first = rates.iter().position(|r| r / source_hz <= limit).unwrap_or(rates.len().checked_sub(1)?);
        rates.get((first + self.control.penalty() as usize).min(rates.len() - 1)).copied()
    }

    fn vote_auto_grid(&mut self) {
        let eligible = self.opts.mode == RateMode::Auto
            && self.opts.fps_limit.is_none()
            && self.opts.limit_to_display
            && !self.opts.low_latency;
        let want = eligible && self.display.ready() && self.auto_grid_hz().is_some();
        if want == self.auto_exact {
            self.auto_exact_votes = 0;
            return;
        }
        self.auto_exact_votes += 1;
        if self.auto_exact_votes >= AUTO_GRID_VOTES {
            self.auto_exact = want;
            self.auto_exact_votes = 0;
            self.reset_output_cadence(false);
        }
    }

    /// The factor the mode settles at regardless of motion: what the delay floor is sized for.
    fn steady_factor(&self) -> u32 {
        match self.opts.mode {
            RateMode::Auto if !self.exact() => self.auto_policy(self.source.auto_source_hz(), false).factor,
            RateMode::Factor(f) => f,
            _ => self.factor,
        }
    }

    /// Auto's factor for this interval and whether it runs (see `AUTO_HOLD_MS`).
    fn auto_settled_factor(&mut self, now: f64, source_hz: f64) -> (u32, bool) {
        let policy = self.auto_policy(source_hz, self.opts.motion_adaptive);
        let current = self.factor;
        if self.auto_changed_at == f64::NEG_INFINITY {
            // the start counts as a change: the queue fills and the estimates settle before the first rise
            self.auto_changed_at = now;
        }
        if policy.factor == current {
            self.auto_candidate = None;
            return (current, policy.runnable);
        }
        let free = self.auto_policy(source_hz, false);
        let up = policy.factor > current;
        let since = match self.auto_candidate {
            Some((dir, since)) if dir == up => since,
            _ => {
                self.auto_candidate = Some((up, now));
                now
            }
        };
        if current > free.factor || (now - since >= AUTO_ASK_MS && now - self.auto_changed_at >= AUTO_HOLD_MS) {
            self.auto_candidate = None;
            self.auto_changed_at = now;
            return (policy.factor, policy.runnable);
        }
        // the factor in use fits wherever the motion-free factor does
        (current, free.runnable)
    }

    fn auto_policy(&self, source_hz: f64, motion_adaptive: bool) -> rate::AutoPolicy {
        let display = display::measure_display_hz(self.display.policy_sample_ms(), self.opts.min_present_interval_ms);
        auto_factor(&AutoInputs {
            max_factor: self.opts.max_factor,
            // the host's seeds until the costs are measured: on the bare default (10 ms a frame) a cold start ran a
            // second at x4 of 24 fps before the first timestamp query let it reach the even x5
            mid_cost_ms: self.mid_cost(),
            pair_cost_ms: self.pair_cost(),
            source_hz,
            unique_interval_ms: self.source.unique_ms(),
            wall_interval_ms: self.source.wall_ms(),
            display_capacity_hz: display.capacity_hz,
            display_hz: display.display_hz,
            even_cadence: self.display.ready() && !self.opts.low_latency,
            cadence_hz: self.source.source_hz().unwrap_or(source_hz),
            limit_to_display: self.opts.limit_to_display,
            queue_capacity_hz: self.queue.capacity_hz(self.delay_ms),
            headroom: self.opts.display_headroom,
            motion_avg: self.motion_avg,
            motion_adaptive,
            penalty: self.control.penalty(),
        })
    }

    fn history_break(&mut self) {
        self.counters.history_breaks += 1;
        self.job = None;
        self.prev = None;
        self.prev2 = None;
        self.grid_next_at = 0.0;
        self.grid_phase_ms = 0.0;
    }

    /// A source frame arrived, its interval kind known. Returns the requests that must be generated before the pair
    /// changes (leftovers of the previous pair).
    pub fn source_frame(&mut self, now: f64, frame: SourceFrame) -> Vec<Request> {
        self.seen = self.seen.max(now);
        self.log(TraceEvent::Source {
            t: now,
            id: frame.id,
            pts: frame.pts_ms,
            kind: frame.kind,
            motion: frame.motion,
            skipped: frame.skipped,
        });
        self.counters.sources += 1;
        let dt = self.source.wall_sample(now);
        let pts_step = self.last_pts.map(|p| (frame.pts_ms - p) / self.source.playback_rate());
        let decoded = match self.last_pts {
            Some(p) => (frame.pts_ms - p) / (frame.skipped + 1) as f64,
            None => dt,
        };
        self.last_pts = Some(frame.pts_ms);
        if frame.skipped > 0 {
            // a coalesced callback: never interpolate across the missing frame as though this one followed at the
            // normal cadence. The clock carries on where the timestamp step spans the missing frame: restarting it
            // on the late arrival shifted every later picture (a browser decoder under load skips a frame a second)
            self.history_break();
            if !pts_step.is_some_and(|d| d > 0.5 && d < 500.0) {
                self.source.reset_clock();
            }
        }
        if self.source.push_decoded(decoded) {
            self.counters.rate_transitions += 1;
            self.reset_output_cadence(true);
            self.delay_ms = self.opts.start_delay_ms();
            self.prev = None;
            self.prev2 = None;
            self.source.reset_clock();
        }
        let tick_ms = if self.display.ready() { self.display.floor_ms() } else { 0.0 };
        self.source.advance_clock(now, pts_step, tick_ms);

        // the presentation delay must cover the batch compute time (own plus the previous batch draining), or high
        // factors drop their early mids as stale; it follows its target at 2 ms per frame (a whole interval in
        // exact modes, where it also carries the pair lookahead)
        let wall_pair = self.source.wall_pair_ms();
        self.vote_auto_grid();
        let exact = self.exact();
        let burst_pad = ((self.late_avg - 4.0) * 2.0).max((self.late_tail_ms - 8.0) * 2.0).clamp(0.0, 60.0);
        let lookahead = if self.opts.lookahead { wall_pair.min(120.0) } else { 0.0 };
        let measured_mid = self.mid_cost.value();
        // decided on what holds still (the cost, the factor Auto could run at), never on the factor of the moment or
        // the lateness: either moved the floor by 18 ms several times a second, and the picture's timeline with it
        let fast = measured_mid > 0.0 && measured_mid < 6.0 && self.steady_factor() <= 3;
        let floor = if fast { self.opts.delay_floor_ms * 0.7 } else { self.opts.delay_floor_ms } + lookahead;
        let max = if exact { EXACT_DELAY_MAX_MS } else { self.opts.delay_max_ms } + lookahead;
        let target = if self.opts.low_latency && !exact {
            // the first in-between after a source frame is due 1/N of an interval after it, and it needs the next
            // frame: that frame's arrival, its pair and the in-between itself must fit before then (a frame ready a
            // little after its due time still goes out on the next tick, within the lateness window)
            let n = self.factor.max(2) as f64;
            ((1.0 - 1.0 / n) * self.source.wall_ms()
                + self.pair_cost()
                + self.mid_cost()
                + self.present_cost.value()
                + self.source.arrival_late_ms()
                + burst_pad)
                .min(max)
        } else {
            presentation_delay_ms(
                exact,
                wall_pair,
                self.mid_cost() + self.present_cost.value(),
                self.pair_cost(),
                burst_pad,
                floor,
                max,
            )
        };
        // every millisecond the delay moves stretches or squeezes the picture's timeline by as much: 2 ms per frame is
        // 12 % faster or slower for a few frames, a visible hitch each time the target moves (a factor change moved it
        // between 42 and 60 ms). Video drifts at 0.25 ms per frame (1.5 %) and rises fast only while frames run late;
        // low latency sits close to its target, which follows the arrival jitter, and only falls slowly
        let (rise, fall) = if exact {
            (wall_pair.max(2.0), 2.0)
        } else if self.opts.low_latency {
            (2.0, 0.25)
        } else {
            (if burst_pad > 0.0 { 2.0 } else { 0.25 }, 0.25)
        };
        self.delay_ms += (target - self.delay_ms).clamp(-fall, rise);

        let sched_t = self.source.sched_t();
        let src_at = sched_t + self.delay_ms;
        // in-betweens are spread over the step the clock took, the spacing of the source frames around them: the
        // arrival average differs from it by up to a millisecond, and at x4 on 240 Hz that piled the frames up against
        // the ticks until one was dropped every second or two
        // on the exact grid a pair is as long as its timestamps say: a source that skips frames (a browser at 1.5x
        // delivers steps of one and two) gets in-betweens by media time, not as though every pair were the average
        let interval = if exact {
            pts_step.filter(|d| *d > 0.5 && *d < 500.0).unwrap_or(wall_pair)
        } else {
            self.source.sched_step_ms().unwrap_or(self.source.wall_ms())
        };
        let start_at = sched_t - interval + self.delay_ms;
        let plan = if exact { Some(self.rate_plan(now)) } else { None };
        let cadence_mode = plan.as_ref().is_some_and(|p| p.interpolation_allowed);
        if !cadence_mode || self.bridge {
            self.enqueue_source(frame.id, src_at);
        }
        self.bridge = false;

        if let Some(m) = frame.motion {
            if m < 90.0 {
                self.motion_avg = self.motion_avg * 0.7 + m * 0.3;
            }
        }
        match frame.kind {
            IntervalKind::Cut => {
                self.counters.cuts += 1;
                self.source.mark_unique_at(now);
            }
            IntervalKind::Hold => self.counters.holds += 1,
            IntervalKind::Motion => self.source.note_unique(now),
        }
        let cur = Prev { id: frame.id, interpolate: frame.kind == IntervalKind::Motion, span_ms: interval };
        let Some(prev) = self.prev else {
            if cadence_mode {
                self.enqueue_source(frame.id, src_at);
            }
            self.prev = Some(cur);
            return Vec::new();
        };
        let flush = self.flush_job();
        let (mid, pair, present) = (self.mid_cost(), self.pair_cost(), self.present_cost.value());

        if exact {
            let Some(plan) = plan.filter(|_| cadence_mode) else {
                self.factor = 1;
                self.advance(prev, cur);
                return flush;
            };
            let output_hz = self.on_ticks(plan.output_hz.unwrap_or(0.0));
            let interpolate = cur.interpolate && target_needs_interpolation(1000.0 / wall_pair, output_hz);
            let grid =
                plan_source_cadence(self.grid_next_at, self.grid_phase_ms, start_at, interval, output_hz, interpolate);
            self.grid_next_at = grid.next_at;
            self.grid_phase_ms = grid.next_phase_ms;
            self.schedule_anchors(&grid.slots, prev.id, cur.id);
            self.factor = grid.slots.len().max(1) as u32;
            if grid.overflowed {
                self.enqueue_source(cur.id, src_at);
                self.advance(prev, cur);
                return flush;
            }
            let mids: Vec<Slot> = grid.slots.iter().copied().filter(|s| s.kind == Kind::Interpolate).collect();
            if !interpolate || mids.is_empty() {
                self.advance(prev, cur);
                return flush;
            }
            let fits = pair_gpu_cost(mids.len() as u32, grid.slots.len() as f64, mid, present, pair)
                <= wall_pair * self.opts.compute_budget;
            if !fits {
                self.counters.pairs_skipped_gpu += 1;
                let mut anchors = mids.clone();
                plan::fallback_to_anchors(&mut anchors);
                self.schedule_anchors(&anchors, prev.id, cur.id);
                self.advance(prev, cur);
                return flush;
            }
            let requests = mids.iter().map(|s| self.request(prev.id, cur.id, s.t, s.at, false, plan.probe)).collect();
            self.job = Some(Job { requests, next: 0 });
            self.advance(prev, cur);
            return flush;
        }

        let source_hz = self.source.auto_source_hz();
        let (n, run, probe) = match self.opts.mode {
            RateMode::Auto => {
                let (mut n, runnable) = self.auto_settled_factor(now, source_hz);
                let mut run = runnable || (n > 1 && self.control.gpu_keeping_up(&self.control_opts));
                let mut probe = false;
                if run {
                    self.control.probe_reset(now, &self.control_opts);
                } else if n > 1 && self.control.probe_due(now, &self.control_opts) {
                    n = 2;
                    run = true;
                    probe = true;
                    self.counters.probes += 1;
                    self.control.probe_fire(now, &self.control_opts);
                }
                (n, run, probe)
            }
            RateMode::Factor(f) => {
                let (n, run) = fixed_factor(f, mid, present, pair, source_hz, self.source.unique_ms(), wall_pair);
                (n, run, false)
            }
            RateMode::DisplayHz | RateMode::Target(_) => unreachable!("exact modes planned above"),
        };
        self.factor = n;
        if run && n > 1 {
            if self.opts.lookahead {
                // the generatable interval is the one before: (prev2, prev). Its real slot (t 0) may be re-timed by
                // the Stream's cadence logic and then replaces the queued source frame.
                if let Some(p2) = self.prev2 {
                    if prev.interpolate {
                        // the pair is as long as the clock's step was when its second frame came, not as the step now:
                        // spread over the current one, its in-betweens sat 7.2 to 9.5 ms apart at x5 and off their
                        // frames (8 % of the pictures off pace on a pan)
                        let at = sched_t - interval - prev.span_ms + self.delay_ms;
                        let mut requests = vec![self.request(p2.id, prev.id, 0.0, at, true, probe)];
                        for k in 1..n {
                            let t = k as f64 / n as f64;
                            requests.push(self.request(p2.id, prev.id, t, at + t * prev.span_ms, false, probe));
                        }
                        self.job = Some(Job { requests, next: 0 });
                    }
                }
            } else if cur.interpolate {
                let requests = (1..n)
                    .map(|k| {
                        let t = k as f64 / n as f64;
                        self.request(prev.id, cur.id, t, start_at + t * interval, false, probe)
                    })
                    .collect();
                self.job = Some(Job { requests, next: 0 });
            }
        }
        self.advance(prev, cur);
        flush
    }

    fn advance(&mut self, prev: Prev, cur: Prev) {
        self.prev2 = Some(prev);
        self.prev = Some(cur);
    }

    fn schedule_anchors(&mut self, slots: &[Slot], previous: u32, current: u32) {
        for s in slots {
            match s.kind {
                Kind::Previous => self.enqueue_source(previous, s.at),
                Kind::Current => self.enqueue_source(current, s.at),
                Kind::Interpolate => {}
            }
        }
    }
}

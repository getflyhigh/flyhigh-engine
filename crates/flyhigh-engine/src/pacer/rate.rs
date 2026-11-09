//! The output rate policy: how many frames to make and show per second, and why not more.
use super::display::{DisplayMeasure, NOMINAL_RATE_TOLERANCE};
use super::source::{
    auto_display_budget_hz, cap_factor_for_display, mixed_presentation_hz, VIDEO_RATE_MATCH_TOLERANCE,
};
use serde::{Deserialize, Serialize};

/// The most target ticks one source interval may hold.
pub const MAX_TICKS_PER_INTERVAL: usize = 64;
pub(crate) const MIN_TARGET_FPS: f64 = 2.0;
pub(crate) const MAX_TARGET_FPS: f64 = 1000.0;

/// How many frames to show per source frame.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RateMode {
    /// `n` presentations per source frame: the source frame and `n - 1` mids at k / n.
    Factor(u32),
    /// As many as the display, the GPU and the motion allow.
    Auto,
    /// Presentations on the display's own grid.
    DisplayHz,
    /// Presentations on a grid of this many per second.
    Target(f64),
}

impl RateMode {
    /// Modes that place presentations on an exact output grid instead of at k / n of each interval.
    pub fn exact(self) -> bool {
        matches!(self, RateMode::DisplayHz | RateMode::Target(_))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RateState {
    /// A factor mode: no output grid.
    Factor,
    InvalidTarget,
    Measuring,
    /// The display cannot show two frames per source frame.
    NoTwoXDisplay,
    /// The GPU cannot make one mid per source frame at this quality.
    NoTwoXGpu,
    Active,
    ProbingGpu,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClampReason {
    Display,
    Gpu,
    Runtime,
    Minimum,
    GpuProbe,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OutputPlan {
    pub state: RateState,
    pub display: DisplayMeasure,
    pub requested_hz: Option<f64>,
    /// Twice the source rate: below it there is no frame generation.
    pub minimum_hz: Option<f64>,
    pub compute_capacity_hz: Option<f64>,
    pub runtime_capacity_hz: Option<f64>,
    pub output_hz: Option<f64>,
    pub interpolation_allowed: bool,
    pub clamp: Option<ClampReason>,
    pub warning: Option<String>,
    /// A probe plan: tiny assumed costs so one real pair measures the GPU.
    pub probe: bool,
}

pub struct RateInputs {
    pub mode: RateMode,
    /// A capped Auto: the requested rate is a strict ceiling, never raised to the 2x floor.
    pub strict_ceiling: bool,
    pub display: DisplayMeasure,
    pub display_ready: bool,
    pub source_hz: Option<f64>,
    pub mid_cost_ms: f64,
    pub pair_cost_ms: f64,
    pub present_cost_ms: f64,
    pub fill_display: bool,
    pub limit_to_display: bool,
    pub queue_capacity_hz: f64,
    pub headroom: f64,
    pub compute_budget: f64,
    pub max_mids: usize,
}

pub fn target_fps(v: f64) -> Option<f64> {
    (v.is_finite() && v > 0.0).then(|| ((v * 100.0).round() / 100.0).clamp(MIN_TARGET_FPS, MAX_TARGET_FPS))
}

fn rate(v: f64) -> String {
    format!("{}", (v * 100.0).round() / 100.0)
}

pub fn resolve_output_rate(i: &RateInputs) -> OutputPlan {
    let d = i.display;
    let mut plan = OutputPlan {
        state: RateState::Factor,
        display: d,
        requested_hz: None,
        minimum_hz: None,
        compute_capacity_hz: None,
        runtime_capacity_hz: None,
        output_hz: None,
        interpolation_allowed: false,
        clamp: None,
        warning: None,
        probe: false,
    };
    let requested = match i.mode {
        RateMode::Factor(_) | RateMode::Auto => return plan,
        RateMode::DisplayHz => Some(d.capacity_hz),
        RateMode::Target(fps) => target_fps(fps),
    };
    let minimum = i.source_hz.map(|s| s * 2.0);
    plan.requested_hz = requested;
    plan.minimum_hz = minimum;
    let Some(requested) = requested else {
        plan.state = RateState::InvalidTarget;
        plan.warning = Some("Target FPS must be a positive number".into());
        return plan;
    };
    let (Some(source_hz), Some(minimum)) = (i.source_hz, minimum) else {
        plan.state = RateState::Measuring;
        plan.warning = Some("Measuring source FPS and display refresh rate".into());
        return plan;
    };
    if !i.display_ready || !d.measured {
        plan.state = RateState::Measuring;
        plan.warning = Some("Measuring source FPS and display refresh rate".into());
        return plan;
    }
    let queue_cap =
        if i.queue_capacity_hz.is_finite() && i.queue_capacity_hz > 0.0 { i.queue_capacity_hz } else { f64::INFINITY };
    let runtime_capacity = (source_hz * (i.max_mids + 1) as f64).min(queue_cap);
    let (mid, pair, present) = (i.mid_cost_ms.max(0.0), i.pair_cost_ms.max(0.0), i.present_cost_ms.max(0.0));
    let mut compute_capacity = f64::INFINITY;
    if mid > 0.0 || pair > 0.0 || present > 0.0 {
        let budget = 1000.0 / source_hz * i.compute_budget;
        let mut max_mids = 0usize;
        for mids in 1..MAX_TICKS_PER_INTERVAL {
            if pair + mids as f64 * mid + (mids + 1) as f64 * present > budget {
                break;
            }
            max_mids = mids;
        }
        compute_capacity = source_hz * (max_mids + 1) as f64;
        if present > 0.0 {
            // a strict ceiling can decimate below the source rate without any mids: bounded by the output frame
            // cost itself, not by the decoded cadence the pair model above uses
            compute_capacity = compute_capacity.min(900.0 / present);
        }
    }
    plan.compute_capacity_hz = Some(compute_capacity);
    plan.runtime_capacity_hz = Some(runtime_capacity);
    let tolerance = (minimum * VIDEO_RATE_MATCH_TOLERANCE).max(0.01);
    // a 120 Hz panel often reports ~118 Hz: the nominal snap accepts that as 120 and the 2x gate uses the same
    // snap; scheduling stays on the capacity, so a 110 Hz reading still cannot be promoted to 120
    let snapped_nominal = d.raw_hz > 0.0
        && d.display_hz != d.capacity_hz
        && ((d.raw_hz - d.display_hz) / d.display_hz).abs() <= NOMINAL_RATE_TOLERANCE;
    let floor_display = if snapped_nominal { d.display_hz } else { d.capacity_hz };
    if i.limit_to_display && !i.strict_ceiling && floor_display + tolerance < minimum {
        plan.state = RateState::NoTwoXDisplay;
        plan.warning = Some(format!("Needs at least {} Hz; display is ~{} Hz", rate(minimum), rate(floor_display)));
        return plan;
    }
    if !i.strict_ceiling && compute_capacity + tolerance < minimum {
        plan.state = RateState::NoTwoXGpu;
        plan.warning = Some(format!("GPU cannot sustain the {} FPS minimum at this quality", rate(minimum)));
        return plan;
    }
    let desired = if i.strict_ceiling {
        requested
    } else if matches!(i.mode, RateMode::Target(_)) {
        requested.max(minimum)
    } else {
        d.capacity_hz
    };
    // a request at the ceiling of a host that cannot present on every tick (a present limit below the refresh rate) or
    // of a strict cap reserves a little presentation service for catch-up; the reserve never violates the strict 2x
    // floor. On the display's own rate there is nothing to reserve: 97 % of it left a tick without a new frame every
    // 33 ticks, a held picture twice a second on a 60 Hz display, where the full rate (the Pacer puts the grid on the
    // measured tick period) holds one only when the clocks have drifted a whole tick apart
    let headroom_display = d.capacity_hz * i.headroom;
    let targets_ceiling = requested >= d.capacity_hz;
    let fill_panel = i.fill_display && !i.strict_ceiling;
    let host_capped = d.capacity_hz < d.display_hz * 0.97;
    let use_headroom = targets_ceiling
        && !fill_panel
        && (i.strict_ceiling || (host_capped && headroom_display + tolerance >= minimum));
    let display_limit = if !i.limit_to_display {
        f64::INFINITY
    } else if use_headroom {
        headroom_display
    } else {
        d.capacity_hz
    };
    let adjusted =
        if !i.strict_ceiling && i.mode == RateMode::DisplayHz && i.limit_to_display { display_limit } else { desired };
    let maximum = display_limit.min(compute_capacity).min(runtime_capacity);
    let mut output = adjusted.min(maximum);
    // the display's own rate as a whole multiple of the source needs no headroom: every tick shows exactly one frame,
    // due on it, where 97 % of it put three in-betweens in some intervals and four in others (see `auto_factor`)
    if i.mode == RateMode::DisplayHz && i.limit_to_display && !i.strict_ceiling && !fill_panel {
        if let Some(n) = display_multiple(source_hz, d.display_hz) {
            let even = n as f64 * source_hz;
            if even <= d.capacity_hz * (1.0 + CADENCE_TOLERANCE) && even <= compute_capacity.min(runtime_capacity) {
                output = even;
            }
        }
    }
    let clamp = if adjusted > maximum + tolerance {
        Some(if maximum == display_limit {
            ClampReason::Display
        } else if maximum == compute_capacity {
            ClampReason::Gpu
        } else {
            ClampReason::Runtime
        })
    } else if !i.strict_ceiling && matches!(i.mode, RateMode::Target(_)) && requested + tolerance < minimum {
        Some(ClampReason::Minimum)
    } else {
        None
    };
    plan.warning = clamp.map(|c| match c {
        ClampReason::Minimum => {
            format!("{} FPS raised to {} FPS: minimum is 2x the source", rate(requested), rate(output))
        }
        ClampReason::Display => format!("{} FPS capped to {} FPS by the display", rate(requested), rate(output)),
        ClampReason::Gpu => format!("{} FPS capped to {} FPS at this quality", rate(requested), rate(output)),
        ClampReason::Runtime => format!("{} FPS capped to {} FPS by the scheduler", rate(requested), rate(output)),
        ClampReason::GpuProbe => String::new(),
    });
    plan.state = RateState::Active;
    plan.output_hz = Some(output);
    plan.interpolation_allowed = true;
    plan.clamp = clamp;
    plan
}

/// GPU milliseconds for one pair with `mids` generated frames and `presentations` frames shown.
pub fn pair_gpu_cost(mids: u32, presentations: f64, mid_ms: f64, present_ms: f64, pair_ms: f64) -> f64 {
    (if mids > 0 { pair_ms.max(0.0) } else { 0.0 })
        + mids as f64 * mid_ms.max(0.0)
        + presentations.max(0.0) * present_ms.max(0.0)
}

/// The cost of a factor over one unique interval: the mids plus every decoded anchor shown in it.
pub fn unique_interval_cost(
    factor: u32,
    mid_ms: f64,
    present_ms: f64,
    source_hz: f64,
    pair_ms: f64,
    unique_interval_ms: f64,
) -> f64 {
    let mids = factor.saturating_sub(1);
    let anchors = (unique_interval_ms * source_hz.max(0.0) / 1000.0).max(1.0);
    pair_gpu_cost(mids, anchors + mids as f64, mid_ms, present_ms, pair_ms)
}

/// The cost of a factor over one source interval that has motion.
pub fn active_interval_cost(factor: u32, mid_ms: f64, present_ms: f64, pair_ms: f64) -> f64 {
    let mids = factor.saturating_sub(1);
    pair_gpu_cost(mids, 1.0 + mids as f64, mid_ms, present_ms, pair_ms)
}

pub struct AutoInputs {
    pub max_factor: u32,
    pub mid_cost_ms: f64,
    pub pair_cost_ms: f64,
    pub source_hz: f64,
    pub unique_interval_ms: f64,
    pub wall_interval_ms: f64,
    pub display_capacity_hz: f64,
    /// The display's nominal refresh: the tick grid the frames land on, which a host present limit does not change.
    pub display_hz: f64,
    /// Whether to keep to the display's cadence (see `auto_factor`).
    pub even_cadence: bool,
    /// The decoded cadence the cadence rule matches against the display. `source_hz` is the fastest recent interval,
    /// a GPU budget's pessimism: millisecond timestamps (41 and 42 ms at 23.976 fps) read 24.39 there.
    pub cadence_hz: f64,
    pub limit_to_display: bool,
    pub queue_capacity_hz: f64,
    pub headroom: f64,
    pub motion_avg: f64,
    pub motion_adaptive: bool,
    pub penalty: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AutoPolicy {
    pub factor: u32,
    pub unique_hz: f64,
    pub presentation_hz: f64,
    /// Whether the factor fits the GPU budget (within 10 %).
    pub runnable: bool,
}

/// How near a whole multiple of the source rate the display rate must be for Auto to keep to the display's cadence
/// (23.976 fps on 240 Hz is 0.1 % off 10x).
pub const CADENCE_TOLERANCE: f64 = 0.01;

/// n when the display runs at n times the source rate (n >= 2, within `CADENCE_TOLERANCE`), else None (60 fps on
/// 144 Hz: no factor lands evenly).
pub fn display_multiple(source_hz: f64, display_hz: f64) -> Option<u32> {
    if !(source_hz > 0.0 && display_hz > 0.0) {
        return None;
    }
    let r = display_hz / source_hz;
    let n = r.round();
    (n >= 2.0 && (r - n).abs() / r <= CADENCE_TOLERANCE).then_some(n as u32)
}

/// Auto: as much as fits the unique-frame budget, the display service, the drop feedback and the current motion.
pub fn auto_factor(i: &AutoInputs) -> AutoPolicy {
    let mid = if i.mid_cost_ms > 0.0 { i.mid_cost_ms } else { 10.0 };
    let pair = i.pair_cost_ms.max(0.0);
    let active_interval = if i.source_hz > 0.0 { 1000.0 / i.source_hz } else { i.wall_interval_ms };
    let unique = i.unique_interval_ms;
    let mut factor = i.max_factor.clamp(1, 6);
    while factor > 2
        && (unique_interval_cost(factor, mid, 0.0, i.source_hz, pair, unique) > unique * 0.85
            || active_interval_cost(factor, mid, 0.0, pair) > active_interval * 0.85)
    {
        factor -= 1;
    }
    let gpu_factor = factor;
    let unique_hz = if unique > 1.0 { i.source_hz.min(1000.0 / unique) } else { i.source_hz };
    if i.limit_to_display {
        let display_hz = auto_display_budget_hz(i.source_hz, i.display_capacity_hz, i.headroom);
        // every mid lands inside the current decoded interval: cap the burst at the decoded cadence, since
        // duplicate-heavy averages alone can hide a one-interval overload when motion returns
        factor = cap_factor_for_display(factor, i.source_hz, i.source_hz, display_hz);
    } else {
        // no display ceiling: only the presentation queue bounds how many frames may be in flight
        let by_queue = (i.queue_capacity_hz / i.source_hz.max(1.0)).floor().max(1.0) as u32;
        factor = factor.min(by_queue).max(1);
    }
    // fewer mids on fast scenes: interpolation artefacts show most there (the rate swing it causes, x6 to x2 within
    // a second on anime action, is itself visible, hence the switch)
    let motion_ceiling = if !i.motion_adaptive {
        6
    } else if i.motion_avg > 45.0 {
        2
    } else if i.motion_avg > 28.0 {
        3
    } else if i.motion_avg > 16.0 {
        4
    } else {
        6
    };
    factor = factor.min(motion_ceiling);
    // An output rate that does not divide the display's lands on the ticks unevenly: x3 of a 60 fps video on a 240 Hz
    // display shows pictures 4.2, 4.2 and 8.3 ms apart, every third one held twice as long, which reads as judder
    // however high the rate (measured on game footage: picture pace error 1.36 ms p50 at x2/x3, 0.04 ms at x4; 24 fps
    // anime at x3/x4 on 240 Hz: 30 % of the pictures off pace). When the display runs at a whole multiple n of the
    // source rate, the motion ceiling rounds up to the next factor that divides n (anime's x3 and x4 become x5, a
    // flick's x3 at 60 fps becomes x4) and Auto takes the largest dividing factor the GPU allows; the display's own
    // rate (factor n) needs no headroom, since every tick then shows exactly one frame, due on it. Only a GPU that
    // cannot reach the next even rate keeps an uneven one below half the display's rate: 96 fps on 240 Hz varies far
    // less than it would lose by falling to 48. The drop penalty still steps down from there.
    let mut even_of = None;
    if i.even_cadence && i.limit_to_display {
        if let Some(n) = display_multiple(i.cadence_hz, i.display_hz) {
            let motion_even = (motion_ceiling..=6).find(|f| n % f == 0).unwrap_or(motion_ceiling);
            let ceiling = gpu_factor.min(motion_even).min(6);
            let fits = |f: u32| f as f64 * i.cadence_hz <= i.display_capacity_hz * (1.0 + CADENCE_TOLERANCE);
            // x1 is never the even choice: no in-betweens at all is worse than any uneven rate (24 fps on 120 Hz before
            // the GPU can afford x5 would otherwise stop generating)
            if let Some(even) = (2..=ceiling).rev().find(|&f| n % f == 0 && fits(f)) {
                if even >= factor || factor as f64 * i.cadence_hz > i.display_hz * 0.5 {
                    factor = even;
                    even_of = Some(n);
                }
            }
        }
    }
    // the drop penalty steps down through the even factors (x4 to x2 at 60 fps on 240 Hz): x3 there was the judder
    // the even cadence removes, and a run that had drawn a penalty showed it for six seconds (17 % of the pictures
    // off pace); with no even factor left above x1 it steps down one at a time
    let evens: Vec<u32> = even_of.map_or(Vec::new(), |n| (2..=factor).rev().filter(|f| n % f == 0).collect());
    factor = match (evens.get(i.penalty as usize), evens.last()) {
        (Some(&f), _) => f,
        (None, Some(&last)) => last.saturating_sub(i.penalty + 1 - evens.len() as u32).max(1),
        (None, None) => factor.saturating_sub(i.penalty).max(1),
    };
    let runnable = factor > 1
        && unique_interval_cost(factor, mid, 0.0, i.source_hz, pair, unique) <= unique * 1.1
        && active_interval_cost(factor, mid, 0.0, pair) <= active_interval * 1.1;
    AutoPolicy { factor, unique_hz, presentation_hz: mixed_presentation_hz(i.source_hz, unique_hz, factor), runnable }
}

/// A fixed factor is a ceiling: under sustained overload it steps down to what fits the frame budget, because
/// piling up the queue looks far worse than a lower factor. Returns the factor and whether even that runs.
pub fn fixed_factor(
    requested: u32,
    mid_ms: f64,
    present_ms: f64,
    pair_ms: f64,
    source_hz: f64,
    unique_interval_ms: f64,
    source_interval_ms: f64,
) -> (u32, bool) {
    let mut n = requested.max(1);
    while n > 2
        && (unique_interval_cost(n, mid_ms, present_ms, source_hz, pair_ms, unique_interval_ms)
            > unique_interval_ms * 0.9
            || active_interval_cost(n, mid_ms, present_ms, pair_ms) > source_interval_ms * 0.9)
    {
        n -= 1;
    }
    let run = unique_interval_cost(n, mid_ms, present_ms, source_hz, pair_ms, unique_interval_ms)
        <= unique_interval_ms * 1.15
        && active_interval_cost(n, mid_ms, present_ms, pair_ms) <= source_interval_ms * 1.15;
    (n, run)
}

/// How far ahead of their arrival source frames are shown: the pair lookahead (exact modes), the pair and two
/// mid costs, 25 ms of slack and the burst pad, within [floor, max].
pub fn presentation_delay_ms(
    lookahead: bool,
    source_interval_ms: f64,
    mid_cost_ms: f64,
    pair_cost_ms: f64,
    burst_pad_ms: f64,
    floor_ms: f64,
    max_ms: f64,
) -> f64 {
    let pair_lookahead = if lookahead { source_interval_ms } else { 0.0 };
    let required = pair_lookahead + pair_cost_ms + 2.0 * mid_cost_ms + 25.0 + burst_pad_ms;
    required.max(floor_ms).min(max_ms.max(floor_ms))
}

#[derive(Clone, PartialEq, Debug)]
struct Identity {
    state: RateState,
    clamp: Option<ClampReason>,
    output_hz: f64,
}

impl Identity {
    fn of(p: &OutputPlan) -> Identity {
        Identity { state: p.state, clamp: p.clamp, output_hz: p.output_hz.unwrap_or(0.0) }
    }

    fn differs(&self, o: &Identity) -> bool {
        let positive = |v: f64| v.is_finite() && v > 0.0;
        let rel = |a: f64, b: f64| {
            if !positive(a) || !positive(b) {
                if a == b {
                    0.0
                } else {
                    f64::INFINITY
                }
            } else {
                (a - b).abs() / a.max(b)
            }
        };
        self.state != o.state || self.clamp != o.clamp || rel(self.output_hz, o.output_hz) > 0.02
    }
}

/// Holds the output grid on one plan until a different one has been proposed 16 source frames in a row, so cost
/// jitter does not restart the output clock.
#[derive(Default, Debug, Clone)]
pub struct RateStabilizer {
    identity: Option<Identity>,
    snapshot: Option<OutputPlan>,
    candidate: Option<Identity>,
    candidate_samples: u32,
}

impl RateStabilizer {
    pub const TRANSITION_SAMPLES: u32 = 16;

    fn stable(&self, plan: OutputPlan) -> OutputPlan {
        match &self.snapshot {
            // the source cadence is stabilised on its own: the exact 2x floor stays current without a clock reset
            Some(s) => OutputPlan {
                state: s.state,
                clamp: s.clamp,
                output_hz: s.output_hz,
                interpolation_allowed: s.interpolation_allowed,
                warning: s.warning.clone(),
                probe: s.probe,
                ..plan
            },
            None => plan,
        }
    }

    /// The plan to act on, and whether the grid must restart on it.
    pub fn sync(&mut self, plan: OutputPlan) -> (OutputPlan, bool) {
        let next = Identity::of(&plan);
        let Some(previous) = &self.identity else {
            self.identity = Some(next);
            self.snapshot = Some(plan.clone());
            self.candidate = None;
            self.candidate_samples = 0;
            return (plan, false);
        };
        if !previous.differs(&next) {
            self.candidate = None;
            self.candidate_samples = 0;
            return (self.stable(plan), false);
        }
        // nothing runs on a grid yet (measuring, or no 2x): there is no output clock to protect, and holding the first
        // real plan back 16 frames kept an exact mode at the source rate for a second after every start and seek
        let idle = !matches!(previous.state, RateState::Active | RateState::ProbingGpu);
        if idle && next.state == RateState::Active {
            self.identity = Some(next);
            self.snapshot = Some(plan.clone());
            self.candidate = None;
            self.candidate_samples = 0;
            return (plan, true);
        }
        match &self.candidate {
            Some(c) if !c.differs(&next) => self.candidate_samples += 1,
            _ => {
                self.candidate = Some(next.clone());
                self.candidate_samples = 1;
            }
        }
        if self.candidate_samples < Self::TRANSITION_SAMPLES {
            return (self.stable(plan), false);
        }
        self.identity = Some(next);
        self.snapshot = Some(plan.clone());
        self.candidate = None;
        self.candidate_samples = 0;
        (plan, true)
    }

    pub fn snapshot(&self) -> Option<&OutputPlan> {
        self.snapshot.as_ref()
    }

    pub fn reset(&mut self) {
        *self = RateStabilizer::default();
    }
}

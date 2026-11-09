//! The source as seen through its timestamps: the decoded cadence, the unique-frame rate and the smoothed clock
//! that places presentations.
use super::display::nearest_rate;

/// Decoded cadences a measured rate snaps to.
pub const COMMON_VIDEO_RATES: [f64; 16] = [
    24000.0 / 1001.0,
    24.0,
    25.0,
    30000.0 / 1001.0,
    30.0,
    48000.0 / 1001.0,
    48.0,
    50.0,
    60000.0 / 1001.0,
    60.0,
    72.0,
    75.0,
    90.0,
    100.0,
    120000.0 / 1001.0,
    120.0,
];
pub const VIDEO_RATE_NORMALIZE_TOLERANCE: f64 = 0.015;
const VIDEO_RATE_HOLD_TOLERANCE: f64 = 0.03;
pub const VIDEO_RATE_MATCH_TOLERANCE: f64 = 0.005;
const WINDOW: usize = 32;
const TRANSITION_TOLERANCE: f64 = 0.12;
const TRANSITION_SAMPLES: u32 = 4;
const FAST_SIGNAL_WINDOW: usize = 4;

pub fn normalize_video_rate(hz: f64, tolerance: f64) -> f64 {
    let nearest = nearest_rate(&COMMON_VIDEO_RATES, hz);
    if ((hz - nearest) / nearest).abs() <= tolerance {
        nearest
    } else {
        hz
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CadenceEstimate {
    pub interval_ms: f64,
    pub source_hz: f64,
    pub raw_hz: f64,
    pub sample_count: usize,
}

/// Media timestamps jitter in quantised short / long pairs: a median follows whichever side holds the middle slot,
/// a symmetric trimmed mean cancels the pairs and rejects isolated doubled frames or stalls. Once a common decoded
/// cadence is established it is held through 3 % noise; a sustained real change still moves the window outside
/// that band and acquires the new cadence.
pub fn estimate_source_cadence(samples: &[f64], fallback_interval_ms: f64) -> CadenceEstimate {
    let mut valid: Vec<f64> = samples.iter().copied().filter(|v| *v > 0.5 && *v < 2000.0).collect();
    valid.sort_by(f64::total_cmp);
    if valid.is_empty() {
        let hz = normalize_video_rate(1000.0 / fallback_interval_ms, VIDEO_RATE_NORMALIZE_TOLERANCE);
        return CadenceEstimate {
            interval_ms: 1000.0 / hz,
            source_hz: hz,
            raw_hz: 1000.0 / fallback_interval_ms,
            sample_count: 0,
        };
    }
    let trim = if valid.len() >= 8 { (valid.len() / 10).max(1) } else { 0 };
    let robust = if trim > 0 && trim * 2 < valid.len() { &valid[trim..valid.len() - trim] } else { &valid[..] };
    let robust_ms = robust.iter().sum::<f64>() / robust.len() as f64;
    let raw_hz = 1000.0 / robust_ms;
    let measured_hz = normalize_video_rate(raw_hz, VIDEO_RATE_NORMALIZE_TOLERANCE);
    let previous_hz = 1000.0 / fallback_interval_ms;
    let previous_nominal = nearest_rate(&COMMON_VIDEO_RATES, previous_hz);
    let previous_is_nominal = ((previous_hz - previous_nominal) / previous_nominal).abs() < 1e-9;
    let hold = valid.len() >= 8
        && previous_is_nominal
        && ((raw_hz - previous_nominal) / previous_nominal).abs() <= VIDEO_RATE_HOLD_TOLERANCE;
    let source_hz = if hold { previous_nominal } else { measured_hz };
    CadenceEstimate { interval_ms: 1000.0 / source_hz, source_hz, raw_hz, sample_count: valid.len() }
}

/// A target at or below the decoded rate is a decimation job, not frame generation.
pub fn target_needs_interpolation(source_hz: f64, target_hz: f64) -> bool {
    let s = normalize_video_rate(source_hz, VIDEO_RATE_NORMALIZE_TOLERANCE);
    let t = normalize_video_rate(target_hz, VIDEO_RATE_NORMALIZE_TOLERANCE);
    t > s && (t - s) / s > VIDEO_RATE_MATCH_TOLERANCE
}

/// The source rate the Auto policy budgets against. Before the decoded cadence is established the first real sample
/// replaces the 24 fps start-up fallback; afterwards two recent fast samples are required, so one timestamp outlier
/// cannot suppress Auto for the whole 32-sample window.
pub fn estimate_auto_source_hz(
    decoded_interval_ms: f64,
    decoded_samples: &[f64],
    wall_interval_ms: f64,
    playback_rate: f64,
) -> f64 {
    let valid: Vec<f64> = decoded_samples.iter().copied().filter(|v| *v > 0.5 && *v < 2000.0).collect();
    let mut recent: Vec<f64> = valid[valid.len().saturating_sub(FAST_SIGNAL_WINDOW)..].to_vec();
    recent.sort_by(f64::total_cmp);
    let recent_fast = if valid.len() < 8 {
        recent.first().copied()
    } else if recent.len() >= 2 {
        Some(recent[1])
    } else {
        None
    };
    let wall_media =
        if valid.len() < 2 && wall_interval_ms.is_finite() { Some(wall_interval_ms * playback_rate) } else { None };
    let fastest = [Some(decoded_interval_ms), recent_fast, wall_media]
        .into_iter()
        .flatten()
        .filter(|v| *v > 0.5 && *v < 2000.0)
        .fold(f64::INFINITY, f64::min);
    if fastest.is_finite() {
        playback_rate * 1000.0 / fastest
    } else {
        24.0 * playback_rate
    }
}

/// Presentations per second when `factor - 1` mids are made for every unique frame and every source frame is shown.
pub fn mixed_presentation_hz(source_hz: f64, unique_hz: f64, factor: u32) -> f64 {
    source_hz + (factor.max(1) - 1) as f64 * unique_hz.min(source_hz)
}

pub fn cap_factor_for_display(max_factor: u32, source_hz: f64, unique_hz: f64, display_hz: f64) -> u32 {
    let mut factor = max_factor.max(1);
    while factor > 1
        && mixed_presentation_hz(source_hz, unique_hz, factor) > display_hz * (1.0 + VIDEO_RATE_MATCH_TOLERANCE)
    {
        factor -= 1;
    }
    factor
}

/// The display rate Auto may fill: `headroom` of the capacity, or the whole capacity when only that fits the 2x
/// floor.
pub fn auto_display_budget_hz(source_hz: f64, display_hz: f64, headroom: f64) -> f64 {
    let reserved = display_hz * headroom;
    let minimum = source_hz * 2.0;
    let reserved_fits = minimum <= reserved * (1.0 + VIDEO_RATE_MATCH_TOLERANCE);
    let full_fits = minimum <= display_hz * (1.0 + VIDEO_RATE_MATCH_TOLERANCE);
    if !reserved_fits && full_fits {
        display_hz
    } else {
        reserved
    }
}

/// More frequent unique frames raise the load, so react at once; longer duplicate-heavy intervals unlock work
/// gradually.
pub fn update_unique_interval(interval_ms: f64, sample_ms: f64, minimum_ms: f64) -> f64 {
    if !(interval_ms.is_finite() && interval_ms > 0.0) {
        return sample_ms;
    }
    let bounded = sample_ms.max(minimum_ms);
    if bounded < interval_ms {
        bounded
    } else {
        interval_ms * 0.85 + bounded * 0.15
    }
}

#[derive(Default, Debug, Clone, Copy)]
struct Transition {
    interval_ms: f64,
    samples: u32,
    direction: i8,
}

#[derive(Debug, Clone)]
pub struct SourceEstimator {
    arrival_late_ms: f64,
    decoded_ms: f64,
    samples: Vec<f64>,
    transition: Transition,
    wall_ms: f64,
    unique_ms: f64,
    last_unique_at: Option<f64>,
    last_arrival: Option<f64>,
    sched_t: Option<f64>,
    /// How far the schedule clock moved on the last frame: the interval its in-betweens are spread over.
    sched_step_ms: Option<f64>,
    /// The last decoded intervals as they came, for `regular`.
    raw: std::collections::VecDeque<f64>,
    /// The last two seconds of (arrival, timestamp step): how fast frames really come against their timestamps.
    steps: std::collections::VecDeque<(f64, f64)>,
    playback_rate: f64,
}

impl Default for SourceEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl SourceEstimator {
    pub fn new() -> Self {
        SourceEstimator {
            arrival_late_ms: 0.0,
            decoded_ms: 42.0,
            samples: Vec::new(),
            transition: Transition::default(),
            wall_ms: 42.0,
            unique_ms: 42.0,
            last_unique_at: None,
            last_arrival: None,
            sched_t: None,
            sched_step_ms: None,
            raw: std::collections::VecDeque::new(),
            steps: std::collections::VecDeque::new(),
            playback_rate: 1.0,
        }
    }

    pub fn set_playback_rate(&mut self, rate: f64) {
        self.playback_rate = rate.abs().max(0.01);
    }

    pub fn playback_rate(&self) -> f64 {
        self.playback_rate
    }

    /// One decoded interval (media time between consecutive frames, per frame when frames were skipped). Returns
    /// true when a sustained rate change was accepted: the window is restarted on the new cadence.
    pub fn push_decoded(&mut self, sample_ms: f64) -> bool {
        self.raw.push_back(sample_ms);
        if self.raw.len() > 16 {
            self.raw.pop_front();
        }
        if !(sample_ms > 0.5 && sample_ms < 2000.0) {
            self.transition = Transition::default();
            return false;
        }
        let mut accepted = sample_ms;
        let mut transitioned = false;
        // the window stays unbiased while deciding whether an abrupt change is a transition: dropping the first
        // outliers made alternating short / long timestamps converge on the long half and understate the rate
        let outside =
            self.samples.len() >= 8 && ((sample_ms - self.decoded_ms) / self.decoded_ms).abs() > TRANSITION_TOLERANCE;
        if outside {
            let direction = if sample_ms > self.decoded_ms { 1 } else { -1 };
            let t = &mut self.transition;
            if t.samples > 0 && t.direction == direction {
                t.samples += 1;
                t.interval_ms += (sample_ms - t.interval_ms) / t.samples as f64;
            } else {
                *t = Transition { interval_ms: sample_ms, samples: 1, direction };
            }
            if t.samples >= TRANSITION_SAMPLES {
                accepted = t.interval_ms;
                self.samples.clear();
                self.decoded_ms = accepted;
                self.transition = Transition::default();
                transitioned = true;
            }
        } else {
            self.transition = Transition::default();
        }
        self.samples.push(accepted);
        if self.samples.len() > WINDOW {
            self.samples.remove(0);
        }
        self.decoded_ms = estimate_source_cadence(&self.samples, self.decoded_ms).interval_ms;
        if transitioned {
            self.wall_ms = self.decoded_ms / self.playback_rate;
            self.unique_ms = self.wall_ms;
        }
        transitioned
    }

    /// The arrival of a frame on the host clock: the wall interval EMA. Returns the gap to the previous arrival.
    pub fn wall_sample(&mut self, now: f64) -> f64 {
        let dt = self.last_arrival.map_or(f64::INFINITY, |l| now - l);
        if dt > 0.5 && dt < 500.0 {
            self.wall_ms = self.wall_ms * 0.9 + dt * 0.1;
        }
        self.last_arrival = Some(now);
        dt
    }

    /// How much later than the schedule clock frames arrive: a slowly decaying envelope of the recent worst case.
    pub fn arrival_late_ms(&self) -> f64 {
        self.arrival_late_ms
    }

    /// The PLL-smoothed schedule clock: decode jitter must not shake presentation. While frames arrive within one and
    /// a half display ticks (`tick_ms`) of it, it steps by each frame's own timestamp step (`step_ms`, wall time),
    /// scaled by how fast frames have really come against their timestamps over the last two seconds, and follows the
    /// arrivals only slowly. A source without timestamps or with a delivery that wanders further steps by the
    /// arrival EMA and follows the arrivals at 8 %. Both resync hard on seeks and stalls (more than 80 ms off).
    /// Stepping by the arrival EMA alone left the clock about 1 % slow on a 60 fps video whose callbacks land on a
    /// 240 Hz display's ticks, so the due times slid across the ticks.
    pub fn advance_clock(&mut self, now: f64, step_ms: Option<f64>, tick_ms: f64) -> f64 {
        let step = step_ms.filter(|d| *d > 0.5 && *d < 500.0);
        match step {
            Some(d) => {
                self.steps.push_back((now, d));
                while self.steps.front().is_some_and(|(t, _)| now - t > 2000.0) {
                    self.steps.pop_front();
                }
            }
            None => self.steps.clear(),
        }
        let pace = match (self.steps.front(), self.steps.len()) {
            (Some(&(first, _)), n) if n >= 8 && now - first >= 500.0 => {
                let stamped: f64 = self.steps.iter().skip(1).map(|(_, d)| d).sum();
                ((now - first) / stamped).clamp(0.5, 2.0)
            }
            _ => 1.0,
        };
        let t = match self.sched_t {
            Some(s) => {
                let stamped = step.map(|d| s + d * pace);
                match stamped {
                    Some(expected) if (now - expected).abs() <= 80.0 && now - expected <= 1.5 * tick_ms => {
                        // timestamps space the frames exactly, so the arrivals only correct a slow drift: following
                        // them harder passed their bursts (frames a tick late, or a few delivered at once) on to the
                        // spacing of the in-betweens
                        let gain = if now - expected < -1.5 * tick_ms { 0.05 } else { 0.02 };
                        expected + gain * (now - expected)
                    }
                    _ => {
                        // no timestamps, or a delivery that wanders further than a tick: follow the arrival rhythm
                        let expected = s + self.wall_ms;
                        if (now - expected).abs() > 80.0 {
                            self.steps.clear();
                            now
                        } else {
                            expected + 0.08 * (now - expected)
                        }
                    }
                }
            }
            None => now,
        };
        self.sched_step_ms = self.sched_t.map(|s| t - s).filter(|d| *d > 0.5 && *d < 3.0 * self.wall_ms);
        self.sched_t = Some(t);
        self.arrival_late_ms = (self.arrival_late_ms * 0.99).max((now - t).clamp(0.0, 80.0));
        t
    }

    pub fn sched_step_ms(&self) -> Option<f64> {
        self.sched_step_ms
    }

    pub fn sched_t(&self) -> f64 {
        self.sched_t.unwrap_or(0.0)
    }

    pub fn reset_clock(&mut self) {
        self.sched_t = None;
        self.sched_step_ms = None;
        self.steps.clear();
        self.last_unique_at = None;
    }

    pub fn ready(&self) -> bool {
        self.samples.len() >= 8
    }

    /// The decoded cadence at the playback rate, once eight samples are in.
    pub fn source_hz(&self) -> Option<f64> {
        self.ready().then(|| self.playback_rate * 1000.0 / self.decoded_ms)
    }

    pub fn auto_source_hz(&self) -> f64 {
        estimate_auto_source_hz(self.decoded_ms, &self.samples, self.wall_ms, self.playback_rate)
    }

    /// Whether the frames come one decoded interval apart. A browser playing 60 fps at 1.5x delivers two of every
    /// three frames (steps of one and two intervals), a variable-rate recording wanders: in-betweens at k / n between
    /// such frames move at an uneven pace, so they belong on a grid in media time.
    pub fn regular(&self) -> bool {
        let off = self.raw.iter().filter(|d| (**d - self.decoded_ms).abs() > 0.15 * self.decoded_ms).count();
        self.raw.len() < 8 || off * 4 < self.raw.len()
    }

    pub fn decoded_ms(&self) -> f64 {
        self.decoded_ms
    }

    /// The decoded interval in wall time (at the playback rate).
    pub fn wall_pair_ms(&self) -> f64 {
        self.decoded_ms / self.playback_rate
    }

    /// The arrival interval EMA.
    pub fn wall_ms(&self) -> f64 {
        self.wall_ms
    }

    pub fn unique_ms(&self) -> f64 {
        self.unique_ms
    }

    /// A unique (not held) frame arrived: the unique-interval estimate. The first unique frame after a long
    /// frozen run has no usable sample and is admitted at the decoded cadence until evidence grows.
    pub fn note_unique(&mut self, now: f64) {
        let source_ms = self.wall_pair_ms();
        match self.last_unique_at.map(|l| now - l) {
            Some(du) if du > 5.0 && du < 500.0 => {
                self.unique_ms = update_unique_interval(self.unique_ms, du, source_ms);
            }
            _ => self.unique_ms = self.unique_ms.min(source_ms),
        }
        self.last_unique_at = Some(now);
    }

    /// A cut: the unique clock restarts here without a sample.
    pub fn mark_unique_at(&mut self, now: f64) {
        self.last_unique_at = Some(now);
    }

    /// A seek or a decoder gap: the clock and the arrival history go, the decoded cadence stays. A video keeps its
    /// frame rate across a seek, and relearning it (eight frames, a third of a second at 24 fps) left every seek and
    /// every change of speed without in-betweens that long; another source with another rate is picked up as any
    /// rate change is.
    pub fn reset(&mut self) {
        let (rate, decoded, samples) = (self.playback_rate, self.decoded_ms, std::mem::take(&mut self.samples));
        let known = samples.len() >= 8;
        *self = SourceEstimator::new();
        self.playback_rate = rate;
        if known {
            self.decoded_ms = decoded;
            self.samples = samples;
            self.wall_ms = decoded / rate;
            self.unique_ms = self.wall_ms;
        }
    }
}

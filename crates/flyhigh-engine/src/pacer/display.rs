//! The display as seen through the host's tick callbacks: the refresh service interval, its nominal rate and the
//! presentation capacity the planner may use.
use serde::{Deserialize, Serialize};

/// Refresh rates a measured interval snaps to when it is within 3 %.
pub const DISPLAY_RATES: [f64; 12] = [360.0, 240.0, 180.0, 165.0, 144.0, 120.0, 100.0, 90.0, 75.0, 72.0, 60.0, 50.0];
pub(crate) const NOMINAL_RATE_TOLERANCE: f64 = 0.03;
const SAMPLE_RELATIVE_TOLERANCE: f64 = 0.08;
const SAMPLE_QUANTIZATION_MS: f64 = 1.05;
const STABILITY_CV: f64 = 0.12;
const PERIOD_WINDOW: usize = 720;
const SERVICE_WINDOW: usize = 32;
const TRANSITION_SAMPLES: usize = 10;
const FASTER_MISS_BUDGET: u32 = 3;

/// What the display can do, in three numbers that are deliberately kept apart: a 110 Hz reading labelled 120 Hz
/// must never authorise 120 presentations per second.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DisplayMeasure {
    pub measured: bool,
    /// 1000 / service interval.
    pub raw_hz: f64,
    /// The nominal rate for people (snapped to [`DISPLAY_RATES`] within 3 %).
    pub display_hz: f64,
    /// What the planner may schedule: min(raw, nominal, the host's present limit).
    pub capacity_hz: f64,
}

pub(crate) fn nearest_rate(rates: &[f64], v: f64) -> f64 {
    let mut nearest = rates[0];
    for &r in rates {
        if ((v - r) / r).abs() < ((v - nearest) / nearest).abs() {
            nearest = r;
        }
    }
    nearest
}

/// `host_min_present_ms` is the host's own limit on present spacing (Firefox: 7 ms), 0 for none.
pub fn measure_display_hz(floor_ms: f64, host_min_present_ms: f64) -> DisplayMeasure {
    let host_cap = if host_min_present_ms > 0.0 { 1000.0 / host_min_present_ms } else { f64::INFINITY };
    if !(floor_ms > 2.0 && floor_ms < 90.0) {
        return DisplayMeasure { measured: false, raw_hz: 0.0, display_hz: 60.0, capacity_hz: 60f64.min(host_cap) };
    }
    let raw_hz = 1000.0 / floor_ms;
    let nearest = nearest_rate(&DISPLAY_RATES, raw_hz);
    let display_hz = if ((raw_hz - nearest) / nearest).abs() <= NOMINAL_RATE_TOLERANCE {
        nearest
    } else {
        ((raw_hz * 100.0).round() / 100.0).max(1.0)
    };
    let capacity_hz = ((raw_hz.min(display_hz).min(host_cap) * 100.0).round() / 100.0).max(1.0);
    DisplayMeasure { measured: true, raw_hz, display_hz, capacity_hz }
}

struct Summary {
    mean: f64,
    /// A finite quantised window can hold one fewer long interval than the service pattern (4 / 4 / 5 ms); one
    /// range-per-window unit keeps the estimate conservative rather than faster than the service observed.
    service: f64,
    cv: f64,
}

fn summary(samples: &[f64]) -> Summary {
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let mut var = 0.0;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &v in samples {
        var += (v - mean) * (v - mean);
        min = min.min(v);
        max = max.max(v);
    }
    var /= n;
    let cv = if mean > 0.0 { var.sqrt() / mean } else { f64::INFINITY };
    Summary { mean, service: mean + (max - min) / n, cv }
}

fn sample_tolerance(ms: f64) -> f64 {
    SAMPLE_QUANTIZATION_MS.max(ms.abs() * SAMPLE_RELATIVE_TOLERANCE)
}

fn samples_agree(a: f64, b: f64) -> bool {
    (a - b).abs() <= sample_tolerance(a).max(sample_tolerance(b))
}

/// Tracks a sustained move in either direction. A single short callback must not authorise a faster target and a
/// single long callback must not relabel the monitor; quantised callback patterns are summarised as service time,
/// never as their fastest sample.
#[derive(Debug, Clone)]
pub struct DisplayEstimator {
    floor_ms: f64,
    confirmed: bool,
    ready: bool,
    stable_samples: usize,
    service: Vec<f64>,
    faster: Vec<f64>,
    faster_misses: u32,
    slower: Vec<f64>,
    tick_ms: f64,
    host_min_present_ms: f64,
    /// Recent tick gaps with the whole number of refresh periods each spans, for `period_ms`.
    spans: std::collections::VecDeque<(f64, f64)>,
}

impl DisplayEstimator {
    pub fn new(host_min_present_ms: f64) -> Self {
        DisplayEstimator {
            floor_ms: 100.0,
            confirmed: false,
            ready: false,
            stable_samples: 0,
            service: Vec::new(),
            faster: Vec::new(),
            faster_misses: 0,
            slower: Vec::new(),
            tick_ms: 0.0,
            host_min_present_ms,
            spans: std::collections::VecDeque::new(),
        }
    }

    pub fn set_host_min_present(&mut self, ms: f64) {
        self.host_min_present_ms = ms.max(0.0);
    }

    /// One gap between consecutive ticks. Returns the gap when it is a usable sample (1 to 100 ms).
    pub fn sample(&mut self, d: f64) -> Option<f64> {
        if d > 1.0 && d < 100.0 {
            // pessimist: believe slowdowns fast (40 %), speedups slowly (3 %), so Auto cannot re-inflate on a lull
            self.tick_ms = if self.tick_ms == 0.0 {
                d
            } else if d > self.tick_ms {
                self.tick_ms * 0.6 + d * 0.4
            } else {
                self.tick_ms * 0.97 + d * 0.03
            };
            self.update_interval(d);
            self.span(d);
            let agrees = ((d - self.floor_ms) / self.floor_ms).abs() <= 0.06;
            self.stable_samples = if agrees { (self.stable_samples + 1).min(SERVICE_WINDOW) } else { 0 };
            Some(d)
        } else {
            if d >= 100.0 {
                self.stable_samples = 0;
            }
            None
        }
    }

    fn span(&mut self, d: f64) {
        let n = (d / self.floor_ms).round();
        if !self.confirmed || n < 1.0 || (d - n * self.floor_ms).abs() > 0.25 * self.floor_ms {
            return;
        }
        if self.spans.back().is_some_and(|_| (self.period_ms() - self.floor_ms).abs() > 0.03 * self.floor_ms) {
            self.spans.clear();
        }
        self.spans.push_back((d, n));
        if self.spans.len() > PERIOD_WINDOW {
            self.spans.pop_front();
        }
    }

    /// The refresh period as the mean over the last ticks (missed ticks counted as the periods they span): the
    /// service floor is a conservative single-gap figure, half a percent long on a clock with 0.1 ms steps, while a
    /// grid that must stay on the ticks needs the period itself. The floor until enough ticks are in.
    pub fn period_ms(&self) -> f64 {
        let (time, periods) = self.spans.iter().fold((0.0, 0.0), |a, s| (a.0 + s.0, a.1 + s.1));
        if periods >= 30.0 {
            time / periods
        } else {
            self.floor_ms
        }
    }

    fn age_faster(&mut self) {
        if self.faster.is_empty() {
            return;
        }
        self.faster_misses += 1;
        if self.faster_misses > FASTER_MISS_BUDGET {
            self.faster.clear();
            self.faster_misses = 0;
        }
    }

    fn update_interval(&mut self, d: f64) -> bool {
        if self.confirmed && samples_agree(d, self.floor_ms) {
            self.age_faster();
            self.slower.clear();
            self.service.push(d);
            if self.service.len() > SERVICE_WINDOW {
                self.service.remove(0);
            }
            let previous = self.floor_ms;
            let s = summary(&self.service);
            if s.cv <= STABILITY_CV {
                self.floor_ms = s.service;
            }
            self.stable_samples = (self.stable_samples + 1).clamp(TRANSITION_SAMPLES, SERVICE_WINDOW);
            self.ready = true;
            return (self.floor_ms - previous).abs() > 1e-9;
        }
        // Delayed callbacks cannot disprove faster service: up to ten agreeing faster observations survive three
        // intervening misses, so a missed callback at a harmonic interval cannot latch a slower rate for good, and
        // the bounded miss budget keeps isolated fast outliers from accumulating into a false transition.
        let faster = d < self.floor_ms;
        if faster {
            self.slower.clear();
        } else {
            self.age_faster();
        }
        let candidates = if faster { &mut self.faster } else { &mut self.slower };
        if !candidates.is_empty() && !samples_agree(d, summary(candidates).mean) {
            candidates.clear();
            if faster {
                self.faster_misses = 0;
            }
        }
        candidates.push(d);
        if candidates.len() > TRANSITION_SAMPLES {
            candidates.remove(0);
        }
        let count = candidates.len();
        // One or two long callbacks are ordinary scheduling stalls: a confirmed display stays usable through them,
        // but three agreeing slow samples are a real refresh-rate transition.
        let confirmed_slowdown = self.confirmed && !faster && count >= 3;
        if !self.confirmed || confirmed_slowdown {
            self.ready = false;
            self.stable_samples = 0;
        }
        if count < TRANSITION_SAMPLES {
            return false;
        }
        let s = summary(candidates);
        if s.cv > STABILITY_CV {
            return false;
        }
        let confirmed = candidates.clone();
        let previous = self.floor_ms;
        self.floor_ms = s.service;
        self.confirmed = true;
        self.service = confirmed;
        self.faster.clear();
        self.faster_misses = 0;
        self.slower.clear();
        self.stable_samples = TRANSITION_SAMPLES.max(self.service.len());
        self.ready = true;
        (self.floor_ms - previous).abs() > 1e-9
    }

    /// The service interval: 100 ms until measured.
    pub fn floor_ms(&self) -> f64 {
        self.floor_ms
    }

    /// The pessimist tick EMA (0 until the first gap).
    pub fn tick_ms(&self) -> f64 {
        self.tick_ms
    }

    pub fn ready(&self) -> bool {
        self.ready || self.stable_samples >= 10
    }

    pub fn stable_samples(&self) -> usize {
        self.stable_samples
    }

    /// By the tick period once it is known: the floor reads a 240 Hz display on a 0.1 ms clock as 238.6 Hz, under
    /// what x4 of 60 fps needs, and the queue then dropped the frames a display at its own rate would have shown a
    /// tick later (the drops drew a penalty: no in-betweens for six seconds of fast motion).
    pub fn measure(&self) -> DisplayMeasure {
        measure_display_hz(self.period_ms(), self.host_min_present_ms)
    }

    /// The interval the Auto policy sizes against: the service floor once confirmed, else the tick EMA.
    pub fn policy_sample_ms(&self) -> f64 {
        if self.ready() {
            self.period_ms()
        } else if self.tick_ms > 1.0 {
            self.tick_ms
        } else {
            self.floor_ms
        }
    }

    pub fn reset(&mut self) {
        let host = self.host_min_present_ms;
        *self = DisplayEstimator::new(host);
    }
}

//! The Auto controller: AIMD on drop pressure, with the exemptions that keep host hiccups from pinning the factor.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ControlOptions {
    pub pressure_tau_ms: f64,
    pub penalty_up: f64,
    pub penalty_burst: f64,
    pub penalty_min_gap_ms: f64,
    pub penalty_max: u32,
    pub recovery_after_ms: f64,
    pub recovery_below: f64,
    pub stall_window_ms: f64,
    pub stall_rate_limit: f64,
    pub strain_threshold: f64,
    pub strain_release_ms: f64,
    pub probe_initial_ms: f64,
    pub probe_max_ms: f64,
    pub probe_attempts: u32,
}

const HOST_BOUND_CHECK_MS: f64 = 2000.0;
const HOST_BOUND_MS: f64 = 30_000.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DropCharge {
    Charged,
    /// Inside the settle window after a mode change: the reflow stall is not a GPU limit.
    SettleExempt,
    /// Right after an occasional host stall: fewer mids would not have prevented these.
    StallExempt,
}

#[derive(Debug, Clone)]
pub struct AutoController {
    penalty: u32,
    penalty_t: f64,
    pressure: f64,
    last_pressure_t: Option<f64>,
    strain_penalty: u32,
    strain_pressure: f64,
    strain_last_t: f64,
    stall_t: f64,
    stall_load: f64,
    settle_until: f64,
    probe_attempts: u32,
    probe_next_at: f64,
    /// The stall load when the penalty last rose, and until when stalls count as the host's own (see `step`).
    stall_load_at_penalty: f64,
    host_bound_until: f64,
}

impl AutoController {
    pub fn new(now: f64, o: &ControlOptions) -> Self {
        AutoController {
            penalty: 0,
            penalty_t: now,
            pressure: 0.0,
            last_pressure_t: None,
            strain_penalty: 0,
            strain_pressure: 0.0,
            strain_last_t: now,
            stall_t: f64::NEG_INFINITY,
            stall_load: 0.0,
            settle_until: f64::NEG_INFINITY,
            probe_attempts: 0,
            probe_next_at: now + o.probe_initial_ms,
            stall_load_at_penalty: 0.0,
            host_bound_until: f64::NEG_INFINITY,
        }
    }

    pub fn reset(&mut self, now: f64, o: &ControlOptions) {
        let settle = self.settle_until;
        *self = AutoController::new(now, o);
        self.settle_until = settle;
    }

    pub fn settle(&mut self, now: f64, ms: f64) {
        self.settle_until = now + ms;
    }

    pub fn settled(&self, now: f64) -> bool {
        now >= self.settle_until
    }

    /// The host's loop stood still (GC, layout, another script, a compositor stall): load ~ stalls per 2 s.
    pub fn stall(&mut self, now: f64) {
        self.stall_load = self.stall_load * libm::exp((self.stall_t - now) / 2000.0) + 1.0;
        self.stall_t = now;
    }

    /// Leaky integrators (tau 300 ms): a burst of drops is visible in milliseconds instead of averaging out over
    /// seconds.
    pub fn decay(&mut self, now: f64, tau_ms: f64) {
        if let Some(last) = self.last_pressure_t {
            let k = libm::exp((last - now) / tau_ms);
            self.pressure *= k;
            self.strain_pressure *= k;
        }
        self.last_pressure_t = Some(now);
    }

    /// `due` frames were skipped to show the newest one. Several frames due in one tick are one event: charging
    /// each pinned Auto at x1 for seconds.
    pub fn charge_drops(&mut self, now: f64, due: usize, o: &ControlOptions) -> DropCharge {
        if !self.settled(now) {
            return DropCharge::SettleExempt;
        }
        let after_stall = now - self.stall_t < o.stall_window_ms;
        if after_stall && (self.stall_load < o.stall_rate_limit || now < self.host_bound_until) {
            return DropCharge::StallExempt;
        }
        self.pressure += (due as f64).min(1.0);
        DropCharge::Charged
    }

    /// Ticks slower than `strain_threshold` times the floor: the compositor is saturated (frames late by a vsync,
    /// not yet dropped). Charged by elapsed service time so sustained strain converges on the same pressure at
    /// every refresh rate while one long callback counts once.
    pub fn charge_strain(&mut self, now: f64, d: f64, floor_ms: f64, o: &ControlOptions) {
        if d <= floor_ms * o.strain_threshold {
            return;
        }
        self.strain_pressure += 2.0 * d / o.pressure_tau_ms;
        self.strain_last_t = now;
    }

    /// The AIMD evaluation, every tick: aggressive decrease on pressure, additive recovery after a clean stretch.
    /// Strain gives a provisional one-step penalty that lifts 500 ms after the strain ends; durable penalties are
    /// reserved for real drops, so harmonic 4 / 8 ms service cannot escalate.
    pub fn step(&mut self, now: f64, o: &ControlOptions) {
        if self.strain_pressure > o.penalty_up {
            self.strain_penalty = 1;
        }
        if self.strain_penalty > 0 && now - self.strain_last_t >= o.strain_release_ms {
            self.strain_penalty = 0;
            self.strain_pressure = 0.0;
        }
        // Stalls that go on as before at the lower factor are not the in-betweens' doing (Firefox stalls its display
        // loop 50 to 100 ms a few times a second while any page element lies over the canvas, whatever is generated):
        // the penalty only took frames away, 21 pictures a second at x1 where x5 showed 44 to 90 between the same
        // stalls. Two seconds after a step down with the stall load no lower, the penalty is lifted and the drops that
        // follow a stall are the host's for half a minute; drops without a stall are charged as ever.
        if self.penalty > 0
            && now - self.penalty_t > HOST_BOUND_CHECK_MS
            && self.stall_load >= (0.7 * self.stall_load_at_penalty).max(0.5 * o.stall_rate_limit)
        {
            self.penalty = 0;
            self.pressure = 0.0;
            self.penalty_t = now;
            self.host_bound_until = now + HOST_BOUND_MS;
        }
        if self.pressure > o.penalty_up && self.penalty < o.penalty_max && now - self.penalty_t > o.penalty_min_gap_ms {
            let step = if self.pressure > o.penalty_burst { 2 } else { 1 };
            self.penalty = (self.penalty + step).min(o.penalty_max);
            self.penalty_t = now;
            self.pressure = 0.0;
            self.stall_load_at_penalty = self.stall_load;
        } else if self.penalty > 0 && self.pressure < o.recovery_below && now - self.penalty_t > o.recovery_after_ms {
            self.penalty -= 1;
            self.penalty_t = now;
        }
    }

    pub fn penalty(&self) -> u32 {
        self.penalty + self.strain_penalty
    }

    pub fn durable_penalty(&self) -> u32 {
        self.penalty
    }

    pub fn strain_penalty(&self) -> u32 {
        self.strain_penalty
    }

    pub fn pressure(&self) -> f64 {
        self.pressure
    }

    pub fn strain_pressure(&self) -> f64 {
        self.strain_pressure
    }

    pub fn stall_load(&self) -> f64 {
        self.stall_load
    }

    pub fn gpu_keeping_up(&self, o: &ControlOptions) -> bool {
        self.pressure < o.recovery_below && self.strain_penalty == 0
    }

    pub fn probe_reset(&mut self, now: f64, o: &ControlOptions) {
        self.probe_attempts = 0;
        self.probe_next_at = now + o.probe_initial_ms;
    }

    /// Auto that cannot run by the budget tries one 2x pair at exponentially growing intervals, at most
    /// `probe_attempts` times, to learn the real cost instead of trusting a stale estimate.
    pub fn probe_due(&self, now: f64, o: &ControlOptions) -> bool {
        self.probe_attempts < o.probe_attempts && now >= self.probe_next_at
    }

    pub fn probe_fire(&mut self, now: f64, o: &ControlOptions) {
        self.probe_attempts += 1;
        self.probe_next_at = if self.probe_attempts >= o.probe_attempts {
            f64::INFINITY
        } else {
            now + (o.probe_initial_ms * 2f64.powi(self.probe_attempts as i32)).min(o.probe_max_ms)
        };
    }
}

//! Presentations on an exact output grid: every target tick becomes one presentation, an anchor on a source frame
//! or an interpolation at phase t.
use super::rate::MAX_TICKS_PER_INTERVAL;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Previous,
    Current,
    Interpolate,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Slot {
    pub at: f64,
    pub t: f64,
    pub kind: Kind,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GridPlan {
    pub next_at: f64,
    pub next_phase_ms: f64,
    pub slots: Vec<Slot>,
    pub step_ms: f64,
    pub resynced: bool,
    pub overflowed: bool,
}

/// Interpolations everywhere except a tick that lands on a source frame (within 5 % of a step), which shows that frame.
/// Snapping the nearest tick to its source frame wherever it stood put the picture up to half a step off its time:
/// 23.976 fps at 60 showed phases 0, .41, .81, 1.0, 1.61 instead of 0, .4, .8, 1.2, 1.6, a shake on slow pans.
pub fn assign_kinds(slots: &mut [Slot], interpolate: bool, source_interval_ms: f64, step_ms: f64) {
    let on_frame = |t: f64| t.min(1.0 - t).abs() * source_interval_ms <= 0.05 * step_ms;
    for s in slots.iter_mut() {
        s.kind = if interpolate && !on_frame(s.t) {
            Kind::Interpolate
        } else if s.t < 0.5 {
            Kind::Previous
        } else {
            Kind::Current
        };
    }
}

/// The GPU could not afford the interpolations: keep the cadence and fail safe to the nearest anchor instead of an
/// off-grid frame.
pub fn fallback_to_anchors(slots: &mut [Slot]) {
    for s in slots.iter_mut() {
        if s.kind == Kind::Interpolate {
            s.kind = if s.t < 0.5 { Kind::Previous } else { Kind::Current };
        }
    }
}

/// Decoded media time decides how many target ticks belong to a source pair; wall time only anchors the absolute
/// output grid. Keeping the phases apart stops callback jitter from turning nominal 60 to 120 into third-of-a-tick
/// bursts while every deadline stays exactly one step apart. `next_at` 0 starts a new grid.
pub fn plan_source_cadence(
    next_at: f64,
    phase_ms: f64,
    start_at: f64,
    source_interval_ms: f64,
    output_hz: f64,
    interpolate: bool,
) -> GridPlan {
    let step = 1000.0 / output_hz;
    let mut phase = phase_ms;
    let mut at = next_at;
    let mut resynced = false;
    if at <= 0.0 {
        phase = step;
        at = start_at + step;
        resynced = true;
    } else {
        // keep the absolute clock near the wall-time place of this media phase: a stall may move start_at by many
        // periods; move the cursor by whole periods only, so the grid and the phase survive without a backlog of
        // stale deadlines
        let desired = start_at + phase;
        let epsilon = (step * 1e-9).max(1e-7);
        if desired - at > step + epsilon {
            let shift = ((desired - at) / step - 1e-10).ceil();
            at += shift * step;
            resynced = true;
        } else if at - desired > step + epsilon {
            // never rewind the clock (that would re-emit a scheduled deadline): carry its lead as media phase
            phase = (at - start_at).max(0.0);
            resynced = true;
        }
    }
    let mut slots = Vec::new();
    let stop_before = source_interval_ms - (step * 1e-9).max(1e-7);
    while phase < stop_before && slots.len() < MAX_TICKS_PER_INTERVAL {
        slots.push(Slot { at, t: phase / source_interval_ms, kind: Kind::Interpolate });
        phase += step;
        at += step;
    }
    let overflowed = phase < stop_before;
    if overflowed {
        return GridPlan {
            next_at: start_at + source_interval_ms + step,
            next_phase_ms: step,
            slots: Vec::new(),
            step_ms: step,
            resynced,
            overflowed,
        };
    }
    assign_kinds(&mut slots, interpolate, source_interval_ms, step);
    GridPlan {
        next_at: at,
        next_phase_ms: (phase - source_interval_ms).max(0.0),
        slots,
        step_ms: step,
        resynced,
        overflowed,
    }
}

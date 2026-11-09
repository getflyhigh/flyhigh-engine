// The Pacer without a GPU: the estimators and planners on their own, then whole sessions replayed from synthetic
// traces shaped like the hosts' recordings (steady Chrome, Firefox with its periodic stalls, a saturated display
// loop, a fullscreen change).
use flyhigh_engine::pacer::costs::RollingCost;
use flyhigh_engine::pacer::display::measure_display_hz;
use flyhigh_engine::pacer::plan::{plan_source_cadence, Kind};
use flyhigh_engine::pacer::queue::{Entry, PresentationQueue};
use flyhigh_engine::pacer::rate::{presentation_delay_ms, target_fps};
use flyhigh_engine::pacer::source::estimate_source_cadence;
use flyhigh_engine::pacer::trace::{replay, Replay, Trace, TraceEvent};
use flyhigh_engine::pacer::{
    Frame, IntervalKind, Pacer, PacerOptions, RateMode, RateState, Request, SourceFrame, Stage,
};

#[test]
fn a_recording_pacer_writes_the_trace_its_replay_reproduces() {
    let opts = PacerOptions::default();
    let mut live = Pacer::new(&opts).unwrap();
    live.record("test");
    let mut outputs = 0u32;
    let mut drive = |p: &mut Pacer, now: f64, reqs: Vec<Request>| {
        for r in reqs {
            outputs += 1;
            p.generated(now, r.id, Frame::Output(outputs));
        }
    };
    // 2.5 s of a 24 fps source on a 240 Hz display, with a cost sample, a host stall and a mode change in the middle
    let mut t = 0.0;
    for i in 0..600u32 {
        t += 1000.0 / 240.0;
        if i % 10 == 0 {
            let frame = SourceFrame {
                id: (i / 10) % 8,
                pts_ms: (i / 10) as f64 * 1000.0 / 24.0,
                kind: IntervalKind::Motion,
                motion: Some(10.0),
                skipped: 0,
            };
            let reqs = live.source_frame(t, frame);
            drive(&mut live, t, reqs);
        }
        if i == 300 {
            live.cost_sample(t, Stage::Pair, 3.0);
            live.cost_sample(t, Stage::Mid, 1.5);
        }
        if i == 400 {
            live.host_stall(t, 60.0);
        }
        if i == 450 {
            live.mode_change(t);
        }
        let tick = live.display_tick(t);
        drive(&mut live, t, tick.generate);
    }
    let trace = live.stop_recording().expect("recording");
    assert_eq!(trace.options.as_ref(), Some(&opts));
    assert!(trace.events.iter().any(|e| matches!(e, TraceEvent::Stall { .. })));
    assert!(trace.events.iter().any(|e| matches!(e, TraceEvent::ModeChange { .. })));
    assert_eq!(trace.events.iter().filter(|e| matches!(e, TraceEvent::Tick { .. })).count(), 600);
    let again = replay(&trace, &opts).unwrap();
    assert_eq!(again.report, live.report(), "the replay of what the pacer recorded is the pacer");
    assert!(live.trace().is_none(), "stopped");
}

#[test]
fn display_rates_snap_but_capacity_does_not() {
    let m = measure_display_hz(8.4, 0.0);
    assert_eq!(m.display_hz, 120.0, "119 Hz is labelled 120");
    assert!(m.capacity_hz < 119.1 && m.capacity_hz > 119.0, "scheduling stays on the measured {}", m.capacity_hz);
    let firefox = measure_display_hz(4.2, 7.0);
    assert_eq!(firefox.display_hz, 240.0);
    assert!((firefox.capacity_hz - 142.86).abs() < 0.01, "the host's 7 ms present limit caps the capacity");
    assert!(!measure_display_hz(100.0, 0.0).measured, "unmeasured displays are assumed 60 Hz");
}

#[test]
fn source_cadence_trims_jitter_and_snaps() {
    let (short, long) = (1000.0 / 24.0 - 8.0, 1000.0 / 24.0 + 8.0);
    let jittered: Vec<f64> = (0..32).map(|i| if i % 2 == 0 { short } else { long }).collect();
    let e = estimate_source_cadence(&jittered, 42.0);
    assert!((e.source_hz - 24.0).abs() < 1e-9, "alternating short / long timestamps are 24 fps: {}", e.source_hz);
    let film: Vec<f64> = vec![41.708; 16];
    assert!((estimate_source_cadence(&film, 42.0).source_hz - 24000.0 / 1001.0).abs() < 1e-9);
    assert_eq!(estimate_source_cadence(&[], 1000.0 / 30.0).source_hz, 30.0, "no samples: the fallback, snapped");
}

#[test]
fn exact_grid_shows_a_source_frame_only_where_a_tick_lands_on_it() {
    // 24 fps (and 23.976) into 60 Hz: 2 or 3 ticks per interval, each at its own phase; a tick shows the source frame
    // only where it lands on it, never the nearest tick snapped onto it (that put the picture up to half a step off)
    for source_hz in [24.0, 24000.0 / 1001.0] {
        let interval = 1000.0 / source_hz;
        let mut next_at = 0.0;
        let mut phase = 0.0;
        let mut start = 1000.0;
        let mut counts = Vec::new();
        for _ in 0..12 {
            let g = plan_source_cadence(next_at, phase, start, interval, 60.0, true);
            assert!(!g.overflowed);
            for s in &g.slots {
                let on_frame = s.t.min(1.0 - s.t) * interval <= 0.05 * g.step_ms;
                assert_eq!(s.kind != Kind::Interpolate, on_frame, "{:?}", g.slots);
            }
            counts.push(g.slots.len());
            for w in g.slots.windows(2) {
                assert!((w[1].at - w[0].at - g.step_ms).abs() < 1e-6, "deadlines are one step apart");
            }
            next_at = g.next_at;
            phase = g.next_phase_ms;
            start += interval;
        }
        assert!(counts.iter().all(|c| *c == 2 || *c == 3), "{counts:?}");
    }
    let cut = plan_source_cadence(0.0, 0.0, 0.0, 1000.0 / 24.0, 60.0, false);
    assert!(cut.slots.iter().all(|s| s.kind != Kind::Interpolate), "a cut keeps the grid on source frames");
}

fn entry(due: f64, id: u32) -> Entry {
    Entry { due, frame: Some(Frame::Source(id)), fallback: Frame::Source(id), request: None, mid: false }
}

#[test]
fn queue_recovers_inside_the_lateness_contract_and_drops_beyond() {
    let mut q = PresentationQueue::new(24);
    for (i, due) in [100.0, 120.0, 140.0, 160.0].iter().enumerate() {
        q.enqueue(entry(*due, i as u32));
    }
    // the display matches the target: show the oldest frame still inside 3 output intervals
    let s = q.select_due(165.0, 50.0, 60.0, 3.0);
    assert_eq!((s.index, s.due_count, s.drop_count), (Some(1), 4, 1), "100 is 65 ms late, past 60 ms: dropped");
    // a display slower than the target cannot recover: newest due, the rest dropped
    let s = q.select_due(165.0, 50.0, 30.0, 3.0);
    assert_eq!((s.index, s.drop_count), (Some(3), 3));
    assert_eq!(q.select_due(50.0, 50.0, 60.0, 3.0).index, None, "nothing due yet");
    let shown = q.present(1);
    assert_eq!((shown.frame, q.len()), (Some(Frame::Source(1)), 2));
    q.enqueue(entry(110.0, 9));
    assert_eq!(q.remove_stale(), 1, "a deadline before the last presented one would step time backwards");
    let mut small = PresentationQueue::new(2);
    small.enqueue(entry(1.0, 0));
    small.enqueue(entry(2.0, 1));
    assert_eq!(small.enqueue(entry(3.0, 2)).map(|e| e.due), Some(1.0), "over capacity the oldest gives way");
}

#[test]
fn costs_attack_fast_and_relax_slowly() {
    let mut c = RollingCost::default();
    c.push(2.0);
    assert_eq!(c.value(), 2.0);
    c.push(6.0);
    assert_eq!(c.value(), 4.0, "a slower sample is real work: halfway at once");
    for _ in 0..20 {
        c.push(2.0);
    }
    assert!(c.value() > 2.0 && c.value() < 2.5, "relaxation is gradual: {}", c.value());
    c.push(500.0);
    assert!(c.value() < 2.5, "samples above 80 ms are artefacts");
}

#[test]
fn delay_and_target_sanitising() {
    assert_eq!(presentation_delay_ms(false, 41.7, 10.0, 4.0, 0.0, 60.0, 180.0), 60.0, "the floor covers small costs");
    assert_eq!(presentation_delay_ms(true, 41.7, 10.0, 4.0, 0.0, 60.0, 180.0), 41.7 + 4.0 + 20.0 + 25.0);
    assert_eq!(presentation_delay_ms(false, 41.7, 60.0, 40.0, 60.0, 60.0, 180.0), 180.0, "capped at the maximum");
    assert_eq!(target_fps(59.996), Some(60.0));
    assert_eq!(target_fps(0.5), Some(2.0));
    assert_eq!(target_fps(-1.0), None);
}

#[test]
fn options_validate_and_round_trip() {
    let mut o = PacerOptions::default();
    assert!(o.validate().is_ok());
    o.display_headroom = 1.5;
    assert_eq!(o.validate().unwrap_err().field, "display_headroom");
    o = PacerOptions { mode: RateMode::Factor(9), ..PacerOptions::default() };
    assert_eq!(o.validate().unwrap_err().field, "mode");
    let text = serde_json::to_string(&PacerOptions::default()).unwrap();
    let back: PacerOptions = serde_json::from_str(&text).unwrap();
    assert_eq!(back, PacerOptions::default());
    let partial: PacerOptions = serde_json::from_str(r#"{ "mode": { "factor": 3 }, "settle_ms": 500 }"#).unwrap();
    assert_eq!((partial.mode, partial.settle_ms, partial.max_factor), (RateMode::Factor(3), 500.0, 6));
}

/// A session: ticks spaced by `gap(t)`, a source at `source_fps` with exact timestamps, GPU costs every 250 ms.
fn synth(duration_ms: f64, source_fps: f64, mid_ms: f64, pair_ms: f64, gap: impl Fn(f64) -> f64) -> Trace {
    let mut events = Vec::new();
    let mut t = 0.0;
    while t < duration_ms {
        events.push(TraceEvent::Tick { t });
        t += gap(t);
    }
    let interval = 1000.0 / source_fps;
    let mut t = 5.0;
    let mut n = 0u32;
    while t < duration_ms {
        events.push(TraceEvent::Source {
            t,
            id: n % 16,
            pts: n as f64 * interval,
            kind: IntervalKind::Motion,
            motion: Some(10.0),
            skipped: 0,
        });
        n += 1;
        t += interval;
    }
    let mut t = 50.0;
    while t < duration_ms {
        events.push(TraceEvent::Cost { t, stage: Stage::Mid, ms: mid_ms });
        events.push(TraceEvent::Cost { t, stage: Stage::Pair, ms: pair_ms });
        t += 250.0;
    }
    events.sort_by(|a, b| a.t().total_cmp(&b.t()));
    Trace { version: 1, host: "synthetic".into(), notes: String::new(), options: None, events }
}

fn run(trace: &Trace, opts: &PacerOptions) -> Replay {
    replay(trace, opts).expect("valid options")
}

fn presents_per_second(r: &Replay, skip: usize) -> Vec<u64> {
    r.steady(skip).iter().map(|s| s.presents).collect()
}

#[test]
fn steady_chrome_fills_a_120hz_display_from_24fps_without_drops() {
    let trace = synth(12_000.0, 24.0, 2.0, 3.0, |_| 1000.0 / 120.0);
    let r = run(&trace, &PacerOptions::default());
    let steady = r.steady(3);
    assert!(steady.iter().all(|s| s.factor == 5), "x5 is the display's own 120 Hz, one frame per tick: {steady:?}");
    assert!(steady.iter().all(|s| s.drops == 0), "{steady:?}");
    let presents = presents_per_second(&r, 3);
    assert!(presents.iter().all(|p| (118..=122).contains(p)), "120 presentations per second: {presents:?}");
    assert_eq!(r.penalty_max, 0);
    assert_eq!(r.report.counters.late_fallbacks, 0, "every requested mid was there in time");
    assert!(r.report.history_frames >= 4);
}

#[test]
fn auto_keeps_to_the_display_cadence_above_half_its_rate() {
    // 60 fps on 240 Hz: x3 (180 per second) would show pictures 4.2, 4.2 and 8.3 ms apart; x4 is one per tick
    let trace = synth(12_000.0, 60.0, 0.3, 1.5, |_| 1000.0 / 240.0);
    let r = run(&trace, &PacerOptions::default());
    let steady = r.steady(3);
    assert!(steady.iter().all(|s| s.factor == 4 && s.drops == 0), "{steady:?}");
    assert!(presents_per_second(&r, 3).iter().all(|p| (238..=242).contains(p)));
    // 24 fps on 144 Hz with the factor capped at 4: 4 does not divide 6 (ticks 1, 2, 1, 2), 3 does
    let trace = synth(12_000.0, 24.0, 0.3, 1.5, |_| 1000.0 / 144.0);
    let r = run(&trace, &PacerOptions { motion_adaptive: false, max_factor: 4, ..PacerOptions::default() });
    assert!(r.steady(3).iter().all(|s| s.factor == 3), "{:?}", r.steady(3));
    // 24 fps on 120 Hz is 5x, and 5 has no divisor but 1 below it: a GPU that cannot afford x5 keeps x4, uneven,
    // rather than stopping generation
    let trace = synth(12_000.0, 24.0, 9.0, 3.0, |_| 1000.0 / 120.0);
    let r = run(&trace, &PacerOptions::default());
    assert!(r.steady(3).iter().all(|s| s.factor >= 2 && s.mids > 0), "{:?}", r.steady(3));
}

#[test]
fn the_drop_penalty_steps_along_the_even_factors() {
    use flyhigh_engine::pacer::rate::{auto_factor, AutoInputs};
    let inputs = |source_hz: f64, display_hz: f64, penalty: u32| AutoInputs {
        max_factor: 6,
        mid_cost_ms: 0.3,
        pair_cost_ms: 1.5,
        source_hz,
        unique_interval_ms: 1000.0 / source_hz,
        wall_interval_ms: 1000.0 / source_hz,
        display_capacity_hz: display_hz,
        display_hz,
        even_cadence: true,
        cadence_hz: source_hz,
        limit_to_display: true,
        queue_capacity_hz: f64::INFINITY,
        headroom: 0.97,
        motion_avg: 10.0,
        motion_adaptive: true,
        penalty,
    };
    let f = |s, d, p| auto_factor(&inputs(s, d, p)).factor;
    // 60 fps on 240 Hz: x4, then x2 (x3 would judder), then x1
    assert_eq!((f(60.0, 240.0, 0), f(60.0, 240.0, 1), f(60.0, 240.0, 2)), (4, 2, 1));
    // 24 fps on 120 Hz has no even factor between x5 and x1: one step at a time
    assert_eq!((f(24.0, 120.0, 0), f(24.0, 120.0, 1)), (5, 4));
}

#[test]
fn the_tick_lock_keeps_the_cadence_through_jittered_arrivals() {
    // 24 fps at x5 on 240 Hz, every picture two ticks; the frames arrive up to 12 ms early or late (a browser's
    // decode), which moves the due times across tick boundaries
    let opts = PacerOptions { mode: RateMode::Factor(5), mid_cost_seed_ms: 0.3, ..PacerOptions::default() };
    let mut pacer = Pacer::new(&opts).unwrap();
    let (tick, interval) = (1000.0 / 240.0, 1000.0 / 24.0);
    let mut seed = 7u64;
    let mut rnd = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let arrivals: Vec<f64> = (0..240).map(|n| 20.0 + n as f64 * interval + (rnd() - 0.5) * 24.0).collect();
    let (mut next, mut outputs, mut last, mut gaps) = (0usize, 0u32, None::<i64>, Vec::new());
    for i in 0..(10_000.0 / tick) as i64 {
        let now = i as f64 * tick;
        while next < arrivals.len() && arrivals[next] <= now {
            let f = SourceFrame {
                id: next as u32,
                pts_ms: next as f64 * interval,
                kind: IntervalKind::Motion,
                motion: Some(8.0),
                skipped: 0,
            };
            for r in pacer.source_frame(arrivals[next], f) {
                outputs += 1;
                pacer.generated(arrivals[next], r.id, Frame::Output(outputs));
            }
            next += 1;
        }
        let t = pacer.display_tick(now);
        for r in t.generate {
            outputs += 1;
            pacer.generated(now, r.id, Frame::Output(outputs));
        }
        if t.present.is_some() {
            if let (Some(l), true) = (last, now > 3000.0) {
                gaps.push(i - l);
            }
            last = Some(i);
        }
    }
    let off = gaps.iter().filter(|g| **g != 2).count();
    assert!(gaps.len() > 700, "x5 runs: {} presents", gaps.len());
    assert!(off * 100 <= gaps.len(), "{off} of {} pictures were not up for two ticks", gaps.len());
}

// Drives a Pacer the way a host does (ticks, source frames, every request generated at once) and returns what each
// presenting tick showed after the first three seconds: (tick index, media time in wall milliseconds, factor).
fn shown(opts: &PacerOptions, tick_ms: f64, frames: &[(f64, f64)], rate: f64) -> Vec<(i64, f64, u32)> {
    let mut pacer = Pacer::new(opts).unwrap();
    let mut content: std::collections::HashMap<u32, f64> = std::collections::HashMap::new();
    let (mut next, mut outputs, mut out) = (0usize, 0u32, Vec::new());
    let mut issue =
        |pacer: &mut Pacer, now: f64, reqs: Vec<Request>, content: &mut std::collections::HashMap<u32, f64>| {
            for r in reqs {
                outputs += 1;
                let (a, b) = (frames[r.previous as usize].1, frames[r.current as usize].1);
                content.insert(outputs, (a + r.t as f64 * (b - a)) / rate);
                pacer.generated(now, r.id, Frame::Output(outputs));
            }
        };
    let end = frames.last().unwrap().0;
    for i in 0..(end / tick_ms) as i64 {
        let now = i as f64 * tick_ms;
        while next < frames.len() && frames[next].0 <= now {
            let f = SourceFrame {
                id: next as u32,
                pts_ms: frames[next].1,
                kind: IntervalKind::Motion,
                motion: Some(8.0),
                skipped: 0,
            };
            let flush = pacer.source_frame(frames[next].0, f);
            issue(&mut pacer, frames[next].0, flush, &mut content);
            next += 1;
        }
        let t = pacer.display_tick(now);
        issue(&mut pacer, now, t.generate, &mut content);
        if let (Some(p), true) = (t.present, now > 3000.0) {
            let media = match p.frame {
                Frame::Source(id) => frames[id as usize].1 / rate,
                Frame::Output(id) => content[&id],
            };
            out.push((i, media, t.factor));
        }
    }
    out
}

// The share of pictures that stayed up longer or shorter than the media time they covered, by more than half a tick
// (the same picture presented again counts as staying up).
fn off_pace(shown: &[(i64, f64, u32)], tick_ms: f64) -> f64 {
    let mut pictures: Vec<(i64, f64)> = Vec::new();
    for &(i, media, _) in shown {
        if pictures.last().is_none_or(|l| (l.1 - media).abs() > 0.01) {
            pictures.push((i, media));
        }
    }
    let steps: Vec<f64> = pictures.windows(2).map(|w| (w[1].1 - w[0].1) - (w[1].0 - w[0].0) as f64 * tick_ms).collect();
    steps.iter().filter(|e| e.abs() > tick_ms / 2.0).count() as f64 / steps.len().max(1) as f64
}

fn regular(fps: f64, seconds: f64) -> Vec<(f64, f64)> {
    (0..(seconds * fps) as usize).map(|n| (20.0 + n as f64 * 1000.0 / fps, n as f64 * 1000.0 / fps)).collect()
}

#[test]
fn auto_takes_the_exact_grid_where_no_factor_is_even() {
    let auto = PacerOptions { mid_cost_seed_ms: 0.3, ..PacerOptions::default() };
    // 24 fps on a 60 Hz display: x2 would be 48 pictures on 60 ticks, every fourth held twice. On the grid every tick
    // has its own picture at its own media time
    let s = shown(&auto, 1000.0 / 60.0, &regular(24.0, 12.0), 1.0);
    assert!(s.len() as f64 > 8.9 * 59.0, "a picture on every tick: {}", s.len());
    assert!(off_pace(&s, 1000.0 / 60.0) < 0.01, "24 fps on 60 Hz: {:.3} off pace", off_pace(&s, 1000.0 / 60.0));
    // 25 fps on 60 Hz and 50 fps on 120 Hz, the same
    let s = shown(&auto, 1000.0 / 60.0, &regular(25.0, 12.0), 1.0);
    assert!(off_pace(&s, 1000.0 / 60.0) < 0.01, "25 fps on 60 Hz: {:.3}", off_pace(&s, 1000.0 / 60.0));
    let s = shown(&auto, 1000.0 / 120.0, &regular(50.0, 12.0), 1.0);
    assert!(s.len() as f64 > 8.9 * 118.0 && off_pace(&s, 1000.0 / 120.0) < 0.01, "50 fps on 120 Hz");
    // a display at a whole multiple of the source keeps the factors, and the source frames with them
    let s = shown(&auto, 1000.0 / 240.0, &regular(60.0, 12.0), 1.0);
    assert!(s.iter().all(|x| x.2 == 4), "60 fps on 240 Hz is x4");
    assert!(off_pace(&s, 1000.0 / 240.0) < 0.01);
    // the grid needs no doubling: 60 fps on 75 Hz, every fourth frame shown twice as it is, becomes 75 even pictures,
    // and 50 fps on 60 Hz 60
    let s = shown(&auto, 1000.0 / 75.0, &regular(60.0, 12.0), 1.0);
    assert!(s.len() as f64 > 8.9 * 74.0 && off_pace(&s, 1000.0 / 75.0) < 0.01, "60 fps on 75 Hz: {}", s.len());
    let s = shown(&auto, 1000.0 / 60.0, &regular(50.0, 12.0), 1.0);
    assert!(s.len() as f64 > 8.9 * 59.0 && off_pace(&s, 1000.0 / 60.0) < 0.01, "50 fps on 60 Hz: {}", s.len());
    // a display at the source's own rate has nothing to add
    let s = shown(&auto, 1000.0 / 60.0, &regular(60.0, 12.0), 1.0);
    assert!(s.iter().all(|x| x.2 == 1));
}

#[test]
fn a_source_that_skips_frames_is_paced_by_its_timestamps() {
    // a browser playing 60 fps at 1.5x: 60 frames a second arrive, two of every three, so the media steps go one
    // frame, two frames, one, two. Factors gave no in-betweens at all (the rate read as more than the display has)
    let auto = PacerOptions { mid_cost_seed_ms: 0.3, playback_rate: 1.5, ..PacerOptions::default() };
    let frames: Vec<(f64, f64)> =
        (0..720).map(|n| (20.0 + n as f64 * 1000.0 / 60.0, ((n / 2) * 3 + n % 2) as f64 * 1000.0 / 60.0)).collect();
    let s = shown(&auto, 1000.0 / 240.0, &frames, 1.5);
    assert!(s.len() as f64 > 8.9 * 115.0, "generates: {} presents in 9 s", s.len());
    assert!(off_pace(&s, 1000.0 / 240.0) < 0.02, "{:.3} off pace", off_pace(&s, 1000.0 / 240.0));
}

#[test]
fn firefox_stalls_are_not_charged_to_auto() {
    // 60 Hz ticks with a 70 ms compositor stall every 1.5 s, the host's 7 ms present limit in force
    let trace = synth(15_000.0, 24.0, 2.0, 3.0, |t| if t.rem_euclid(1500.0) < 16.7 { 70.0 } else { 1000.0 / 60.0 });
    let opts = PacerOptions { min_present_interval_ms: 7.0, ..PacerOptions::default() };
    let r = run(&trace, &opts);
    assert_eq!(r.penalty_max, 0, "an isolated stall every 1.5 s never pins the factor");
    assert!(r.steady(3).iter().all(|s| s.factor >= 2), "24 fps on the 60 Hz grid: {:?}", r.steady(3));
    let presents = presents_per_second(&r, 3);
    assert!(presents.iter().all(|p| *p >= 54), "60 per second minus the stalled ticks: {presents:?}");
    assert!(r.report.counters.drops_stall_exempt > 0, "the drops after the stalls exist and are exempt");
}

#[test]
fn stalls_that_a_lower_factor_does_not_cure_stop_costing_frames() {
    // a display loop that stalls 70 ms five times every two seconds whatever is generated (Firefox with a page
    // element over the canvas): the drops after the stalls first draw a penalty, the stalls go on at the lower factor,
    // and the penalty is lifted instead of sitting at x1
    let trace = synth(24_000.0, 24.0, 2.0, 3.0, |t| if t.rem_euclid(400.0) < 8.4 { 70.0 } else { 1000.0 / 120.0 });
    let r = run(&trace, &PacerOptions::default());
    let late = &r.seconds[8..23];
    let full = late.iter().filter(|s| s.factor >= 4).count();
    assert!(full >= 12, "back at the full factor for most of the time: {late:?}");
    assert!(late.iter().map(|s| s.presents).sum::<u64>() / late.len() as u64 >= 70, "{late:?}");
}

#[test]
fn a_saturated_loop_steps_auto_down_and_a_clean_stretch_recovers_it() {
    // ticks slow to 30 ms (3.6 x the floor) for three seconds: frames pile up, several due per tick
    let trace = synth(28_000.0, 24.0, 2.0, 3.0, |t| if (3000.0..6000.0).contains(&t) { 30.0 } else { 1000.0 / 120.0 });
    let r = run(&trace, &PacerOptions::default());
    assert!(r.seconds[2].penalty == 0 && r.seconds[2].factor == 5, "{:?}", r.seconds[2]);
    assert!(r.seconds[3].penalty >= 1, "the first step lands within the first overloaded second: {:?}", r.seconds[3]);
    assert!(r.penalty_max >= 2, "sustained overload keeps stepping: {}", r.penalty_max);
    assert!(r.seconds[5].factor < 5, "{:?}", r.seconds[5]);
    assert_eq!(r.report.penalty, 0, "6 s clean per step: recovered by 28 s ({:?})", r.seconds.last());
    assert_eq!(r.seconds.last().map(|s| s.factor), Some(5));
}

#[test]
fn a_mode_change_settles_without_penalty() {
    let trace = synth(9_000.0, 24.0, 2.0, 3.0, |t| if (5000.0..5300.0).contains(&t) { 300.0 } else { 1000.0 / 60.0 });
    let mut events = trace.events.clone();
    events.push(TraceEvent::ModeChange { t: 4999.0 });
    events.sort_by(|a, b| a.t().total_cmp(&b.t()));
    let trace = Trace { events, ..trace };
    let r = run(&trace, &PacerOptions::default());
    assert_eq!(r.penalty_max, 0, "the reflow stall is not a GPU limit");
    assert!(r.report.counters.drops_settle_exempt + r.report.counters.drops_stall_exempt > 0);
    assert!(
        r.seconds[7].presents >= 56 && r.seconds[7].factor >= 2,
        "24 fps on the 60 Hz grid again: {:?}",
        r.seconds[7]
    );
}

#[test]
fn display_hz_mode_places_frames_on_the_display_grid() {
    let trace = synth(10_000.0, 24.0, 2.0, 3.0, |_| 1000.0 / 60.0);
    let opts = PacerOptions { mode: RateMode::DisplayHz, ..PacerOptions::default() };
    let r = run(&trace, &opts);
    let plan = r.report.plan.clone().expect("an exact mode has a plan");
    assert_eq!(plan.state, RateState::Active, "{plan:?}");
    let out = plan.output_hz.unwrap();
    assert!((out - 60.0).abs() < 0.3, "the display's own rate, no headroom: {out}");
    let presents = presents_per_second(&r, 3);
    assert!(presents.iter().all(|p| (59..=61).contains(p)), "a frame on every tick: {presents:?}");
    assert!(r.steady(3).iter().all(|s| s.drops == 0), "{:?}", r.steady(3));
    assert!(r.report.counters.presented_mids > r.report.counters.presented_sources, "most grid ticks are mids");
}

#[test]
fn a_fixed_factor_and_a_cut_keep_the_source_frames() {
    let mut trace = synth(6_000.0, 30.0, 2.0, 3.0, |_| 1000.0 / 120.0);
    for e in &mut trace.events {
        if let TraceEvent::Source { t, kind, .. } = e {
            if (2000.0..2100.0).contains(t) {
                *kind = IntervalKind::Cut;
            }
        }
    }
    let opts = PacerOptions { mode: RateMode::Factor(3), ..PacerOptions::default() };
    let r = run(&trace, &opts);
    assert!(r.steady(1).iter().all(|s| s.factor == 3), "{:?}", r.steady(1));
    assert_eq!(r.report.counters.cuts, 3);
    let presents = presents_per_second(&r, 1);
    assert!(presents.iter().all(|p| (84..=92).contains(p)), "90 per second, 6 fewer around the cut: {presents:?}");
    assert_eq!(r.report.counters.drops, 0);
}

#[test]
fn trace_json_round_trips() {
    let trace = synth(1000.0, 24.0, 2.0, 3.0, |_| 1000.0 / 60.0);
    let text = trace.to_json();
    let back = Trace::from_json(&text).unwrap();
    assert_eq!(back, trace);
    assert!(text.contains(r#""kind":"tick""#) && text.contains(r#""kind_of":"motion""#));
    assert!(Trace::from_json(r#"{ "version": 2, "events": [] }"#).is_err());
}

fn recorded(name: &str) -> Trace {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/traces/");
    let text = std::fs::read_to_string(format!("{path}{name}.json")).expect("the recorded trace");
    Trace::from_json(&text).expect("trace v1")
}

#[test]
fn recorded_chrome_session_replays_without_drops_or_penalties() {
    // 30 s of the extension on a 240 Hz display, 24 fps anime, Auto: the extension saw 101.7 presents/s and no drops
    let trace = recorded("chrome-anime-auto");
    let opts = trace.options.clone().expect("the host recorded its options");
    let r = replay(&trace, &opts).unwrap();
    assert_eq!((r.report.counters.drops, r.penalty_max, r.report.counters.late_fallbacks), (0, 0, 0));
    let total: u64 = r.seconds.iter().map(|s| s.presents).sum();
    assert!((3200..=3450).contains(&total), "about 110 presents per second over 30 s: {total}");
    let tail = &r.seconds[23..29];
    assert!(
        tail.iter().all(|s| s.factor == 5 && s.presents >= 110),
        "x5 is 120 Hz, every second tick, where x6 would hold every third picture twice as long: {tail:?}"
    );
}

#[test]
fn recorded_firefox_session_keeps_auto_through_its_stalls() {
    // the same session in Firefox: the 7 ms present limit caps the capacity at 142.86 Hz (x5 at 24 fps) and the
    // compositor's occasional stalls must not be charged to the GPU
    let trace = recorded("firefox-anime-auto");
    let opts = trace.options.clone().expect("the host recorded its options");
    assert_eq!(opts.min_present_interval_ms, 7.0);
    let r = replay(&trace, &opts).unwrap();
    assert_eq!(r.penalty_max, 0, "an isolated stall never pins the factor");
    assert_eq!(r.factor_max, 5, "the host's present limit caps the factor");
    let c = &r.report.counters;
    assert!(c.drops <= 6 && c.drops == c.drops_stall_exempt, "every drop follows a stall: {c:?}");
    let total: u64 = r.seconds.iter().map(|s| s.presents).sum();
    assert!((3000..=3250).contains(&total), "about 105 presents per second over 30 s: {total}");
}

#[test]
fn recorded_firefox_reference_host_runs_a_minute_without_stalls() {
    // the reference web host (examples/web) in Firefox: the same display and video, nothing but the engine's Stream and
    // Pacer on the page. A minute at 240 Hz ticks with the 7 ms present limit: no compositor stalls, one drop. Firefox
    // has no timestamp queries, so Auto sizes by the host's cost seed throughout (x5; on the bare 10 ms default, x4).
    let trace = recorded("firefox-refhost-auto");
    let opts = trace.options.clone().expect("the host recorded its options");
    assert_eq!((opts.min_present_interval_ms, opts.gpu_timestamps), (7.0, false));
    let r = replay(&trace, &opts).unwrap();
    let c = &r.report.counters;
    assert_eq!((c.stalls, c.host_stalls, r.penalty_max), (0, 0, 0), "{c:?}");
    assert!(c.drops <= 2, "{c:?}");
    assert!(r.factor_max <= 5, "the 7 ms limit caps the factor at 142.86 Hz: {}", r.factor_max);
    let total: u64 = r.seconds.iter().map(|s| s.presents).sum();
    assert!((5700..=6100).contains(&total), "about 107 presents per second over 55 s: {total}");
    assert!(c.cuts > 100, "the anime fixture's cuts are classified: {}", c.cuts);
}

#[test]
fn recorded_player_session_replays_to_the_report_the_host_wrote() {
    // the reference native host (examples/player) wrote its trace and its own report side by side: the replay must
    // reproduce the report byte for byte. Timer ticks at the monitor's 240 Hz, 24 fps anime, Auto, measured GPU costs.
    let trace = recorded("player-anime-auto");
    let opts = trace.options.clone().expect("the host recorded its options");
    let r = replay(&trace, &opts).unwrap();
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/traces/player-anime-auto.report.json");
    let written = std::fs::read_to_string(path).expect("the report the player wrote");
    assert_eq!(r.report.to_json(), written);
    assert!(r.report.pair_cost_ms > 0.0 && r.report.mid_cost_ms > 0.0, "costs came from timestamp queries");
    assert_eq!((r.report.penalty, r.factor_max), (0, 5), "x5 lands on every second tick of 240 Hz, x6 would not");
}

// A game streamed at 60 fps to a 240 Hz display: arrivals with 1 ms of jitter and an occasional 3 to 5 ms late frame,
// costs as the 1080p texture path measures them. Deterministic (a fixed LCG instead of a random source).
fn game_trace() -> flyhigh_engine::pacer::trace::Trace {
    use flyhigh_engine::pacer::trace::{Trace, TraceEvent};
    use flyhigh_engine::pacer::{IntervalKind, Stage};
    let mut seed = 12345u64;
    let mut rnd = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut ev = Vec::new();
    let mut tick = 1000.0;
    while tick < 13000.0 {
        ev.push(TraceEvent::Tick { t: tick });
        tick += 1000.0 / 240.0;
    }
    for n in 0..720u32 {
        let jitter = rnd() * 1.5 + if rnd() < 0.03 { 3.0 + 2.0 * rnd() } else { 0.0 };
        let t = 1005.0 + n as f64 * 1000.0 / 60.0 + jitter;
        ev.push(TraceEvent::Source {
            t,
            id: n,
            pts: n as f64 * 1000.0 / 60.0,
            kind: IntervalKind::Motion,
            motion: Some(20.0),
            skipped: 0,
        });
        if n % 10 == 0 {
            ev.push(TraceEvent::Cost { t: t + 2.0, stage: Stage::Pair, ms: 1.5 });
            ev.push(TraceEvent::Cost { t: t + 2.3, stage: Stage::Mid, ms: 0.3 });
        }
    }
    ev.sort_by(|a, b| a.t().partial_cmp(&b.t()).unwrap());
    let mut tr = Trace::new("synthetic game, 60 fps to 240 Hz");
    tr.events = ev;
    tr
}

#[test]
fn low_latency_shows_a_game_frame_within_a_source_interval_or_so() {
    use flyhigh_engine::pacer::trace::{replay_with, Compute};
    use flyhigh_engine::pacer::PacerOptions;
    let tr = game_trace();
    let compute = Some(Compute { pair_ms: 1.5, mid_ms: 0.3 });
    let video = replay_with(&tr, &PacerOptions::default(), compute).unwrap();
    let game = replay_with(&tr, &PacerOptions { low_latency: true, ..PacerOptions::default() }, compute).unwrap();
    eprintln!(
        "source frame to screen: video {:.1} ms (p95 {:.1}), low latency {:.1} ms (p95 {:.1}); drops {} / {}",
        video.latency.mean_ms,
        video.latency.p95_ms,
        game.latency.mean_ms,
        game.latency.p95_ms,
        video.report.counters.drops,
        game.report.counters.drops
    );
    assert!(video.latency.mean_ms > 55.0, "the video default keeps its 60 ms floor");
    assert!(game.latency.mean_ms < 20.0, "low latency: (1 - 1/N) of an interval plus the work and the jitter");
    assert!(game.latency.p95_ms < 24.0);
    assert_eq!(game.report.counters.drops, 0, "Auto keeps headroom: no drops for the lower delay");
    assert!(game.report.counters.late_fallbacks <= 2, "the in-betweens are still ready in time");
    assert!(game.seconds.iter().skip(2).all(|s| s.mids > 0), "and it still generates");
}

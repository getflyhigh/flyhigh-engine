// The Stream: the classifier's decisions without a GPU, then the push / interval / generate contract and the
// classifier kernel on a real GPU (ignored by default, like gpu_smoke: `cargo test --release -- --ignored`).
use flyhigh_engine::api::Engine;
use flyhigh_engine::gen_tex::TexOptions;
use flyhigh_engine::gpu::readback::Readback;
use flyhigh_engine::stream::classify::{Classification, Thresholds, GRID};
use flyhigh_engine::stream::{Generated, HeldFrames, IntervalKind, Mode, Stream, StreamOptions};
use std::path::PathBuf;

const W: u32 = 256;
const H: u32 = 192;

#[test]
fn classification_thresholds_are_pure() {
    let th = Thresholds { cut_mean: Some(0.08), hold_mean: Some(0.0033), hold_max: 0.06, step_after_hold: true };
    let cells = GRID.0 * GRID.1;
    let identical = Classification::from_stats(0, 0, cells, &th);
    assert_eq!((identical.kind, identical.black), (IntervalKind::Hold, true));
    let cut = Classification::from_stats(cells * 765 / 5, 765, cells, &th);
    assert_eq!(cut.kind, IntervalKind::Cut, "a mean difference of 0.2 is a cut");
    let motion = Classification::from_stats(cells * 20, 400, cells, &th);
    assert_eq!(motion.kind, IntervalKind::Motion);
    assert!((motion.motion - 20.0).abs() < 0.01, "the Pacer's motion units are the mean cell difference");
    let noise = Classification::from_stats(cells, 30, cells, &th);
    assert_eq!(noise.kind, IntervalKind::Hold, "codec noise on a held frame stays a hold");
    let off = Thresholds { cut_mean: None, hold_mean: None, hold_max: 0.0, step_after_hold: false };
    assert_eq!(Classification::from_stats(cells * 765, 765, cells, &off).kind, IntervalKind::Motion);
    // a cut is a lone spike after a calm pair; after a moving pair it is the middle of a fast camera move
    assert_eq!(cut.after(Some(0.02), &th).kind, IntervalKind::Cut);
    assert_eq!(cut.after(None, &th).kind, IntervalKind::Cut);
    assert_eq!(cut.after(Some(0.07), &th).kind, IntervalKind::Motion);
    assert_eq!(motion.after(Some(0.2), &th).kind, IntervalKind::Motion);
    // a change after a held frame is animation on twos or threes: shown as drawn, not interpolated
    assert_eq!(motion.after_hold(true, &th).kind, IntervalKind::Cut);
    assert_eq!(motion.after_hold(false, &th).kind, IntervalKind::Motion);
    assert_eq!(noise.after_hold(true, &th).kind, IntervalKind::Hold);
    assert_eq!(motion.after_hold(true, &off).kind, IntervalKind::Motion);
}

fn engine() -> Option<Engine> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..").join("models");
    match Engine::load(&dir) {
        Ok(e) => Some(e),
        Err(e) if format!("{e:#}").contains("shader-f16") || format!("{e:#}").contains("adapter") => {
            eprintln!("skipped: {e:#}");
            None
        }
        Err(e) => panic!("engine: {e:#}"),
    }
}

// a soft pattern shifted `dx` px to the right
fn frame(dx: f32) -> Vec<u8> {
    let mut v = vec![0u8; (W * H * 4) as usize];
    for y in 0..H {
        for x in 0..W {
            let u = (x as f32 - dx) * 0.07;
            let w = y as f32 * 0.05;
            let o = ((y * W + x) * 4) as usize;
            v[o] = (128.0 + 100.0 * u.sin()) as u8;
            v[o + 1] = (128.0 + 100.0 * (u * 0.7 + w).cos()) as u8;
            v[o + 2] = (128.0 + 90.0 * (u + w).sin()) as u8;
            v[o + 3] = 255;
        }
    }
    v
}

fn write(e: &Engine, s: &Stream, slot: usize, rgba: &[u8]) {
    e.gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: s.input_texture(slot),
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        rgba,
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(W * 4), rows_per_image: Some(H) },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
}

fn read_output(e: &Engine, s: &Stream, slot: usize) -> Vec<u8> {
    let stage = e.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (W * H * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = e.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: s.output_texture(slot),
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &stage,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(W * 4), rows_per_image: Some(H) },
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
    e.gpu.queue.submit(Some(enc.finish()));
    Readback::new(stage, Box::new(|d: &[u8]| d.to_vec()), None).block(&e.gpu).unwrap()
}

fn mae(a: &[u8], b: &[u8]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (*x as f64 - *y as f64).abs()).sum::<f64>() / a.len() as f64
}

#[test]
#[ignore]
fn the_classifier_tells_cuts_holds_and_motion_apart() {
    let Some(e) = engine() else { return };
    let opts = StreamOptions { held_frames: Some(HeldFrames::default()), ..Default::default() };
    let mut s = Stream::new(&e.gpu, &e.blob, &e.ops, TexOptions::new(W, H), &opts, None).unwrap();
    let flat = vec![30u8; (W * H * 4) as usize];
    write(&e, &s, 0, &frame(0.0));
    write(&e, &s, 1, &frame(2.0));
    write(&e, &s, 2, &frame(0.0));
    write(&e, &s, 3, &flat);
    let motion = s.classify(&e.gpu, 0, 1).unwrap().block(&e.gpu).unwrap();
    let hold = s.classify(&e.gpu, 0, 2).unwrap().block(&e.gpu).unwrap();
    let cut = s.classify(&e.gpu, 0, 3).unwrap().block(&e.gpu).unwrap();
    eprintln!("motion {motion:?}\nhold {hold:?}\ncut {cut:?}");
    assert_eq!(motion.kind, IntervalKind::Motion);
    assert!(motion.mean > 0.0033 && motion.mean < 0.08 && !motion.black);
    assert_eq!((hold.kind, hold.black), (IntervalKind::Hold, true), "identical frames");
    assert_eq!(cut.kind, IntervalKind::Cut);
    assert_eq!(s.counters().classified, 3);
}

#[test]
#[ignore]
fn realtime_pushes_give_timestamped_intervals_and_mids() {
    let Some(e) = engine() else { return };
    let mut s = Stream::new(&e.gpu, &e.blob, &e.ops, TexOptions::new(W, H), &StreamOptions::default(), None).unwrap();
    assert_eq!(s.warmup(), 2);
    let s0 = s.next_input(|_| false).unwrap();
    write(&e, &s, s0, &frame(0.0));
    assert!(s.push(&e.gpu, s0, 0.0).unwrap().interval.is_none(), "one frame is no interval");
    let s1 = s.next_input(|_| false).unwrap();
    assert_ne!(s0, s1);
    write(&e, &s, s1, &frame(8.0));
    let p = s.push(&e.gpu, s1, 41.7).unwrap();
    let iv = p.interval.expect("an interval after two frames");
    assert_eq!((iv.base, iv.start_pts, iv.end_pts, p.stats_pending), (s0, 0.0, 41.7, false));
    assert_eq!(s.generate(&e.gpu, 0.0, 0).unwrap(), Generated::SourceIsEnough);
    assert_eq!(s.generate(&e.gpu, 0.5, 0).unwrap(), Generated::Written);
    let (f0, f1, mid) = (frame(0.0), frame(8.0), frame(4.0));
    let out = read_output(&e, &s, 0);
    let (to_mid, to_f0, to_f1) = (mae(&out, &mid), mae(&out, &f0), mae(&out, &f1));
    eprintln!("MAE to the true middle {to_mid:.2}, to f0 {to_f0:.2}, to f1 {to_f1:.2}");
    assert!(to_mid < to_f0 && to_mid < to_f1 && to_mid < 6.0, "the mid lies between its neighbours");
    let c = &s.report().counters;
    assert_eq!((c.pushes, c.pairs, c.mids, c.source_enough), (2, 1, 1, 1));
    let s2 = s.next_input(|_| false).unwrap();
    assert!(s2 != s0 && s2 != s1, "the ring never hands out a slot the generator still reads");
    assert_eq!(s.next_input(|slot| slot == s2), Some((s2 + 1) % 4), "nor one the host still shows");
    s.reset();
    assert_eq!(s.report().pushed, 0);
}

#[test]
#[ignore]
fn timed_push_and_generate_report_gpu_time() {
    let Some(e) = engine() else { return };
    let mut s = Stream::new(&e.gpu, &e.blob, &e.ops, TexOptions::new(W, H), &StreamOptions::default(), None).unwrap();
    let s0 = s.next_input(|_| false).unwrap();
    write(&e, &s, s0, &frame(0.0));
    let (_, run) = s.push_timed(&e.gpu, s0, 0.0).unwrap();
    assert!(run.is_none(), "one frame runs no pair");
    let s1 = s.next_input(|_| false).unwrap();
    write(&e, &s, s1, &frame(8.0));
    let (p, run) = s.push_timed(&e.gpu, s1, 41.7).unwrap();
    assert!(p.interval.is_some());
    let pair_ms = run.map(|r| r.block(&e.gpu));
    let (g, run) = s.generate_timed(&e.gpu, 0.5, 0).unwrap();
    assert_eq!(g, Generated::Written);
    let mid_ms = run.map(|r| r.block(&e.gpu));
    eprintln!("pair {pair_ms:?} ms, mid {mid_ms:?} ms");
    if e.gpu.timestamps {
        let (pair_ms, mid_ms) = (pair_ms.unwrap(), mid_ms.unwrap());
        assert!(pair_ms > 0.0 && mid_ms > 0.0 && pair_ms > mid_ms, "the pair is the expensive half");
    } else {
        assert!(pair_ms.is_none() && mid_ms.is_none(), "no timestamp queries, no times");
    }
    assert_eq!(s.report().counters.mids, 1, "a timed generate counts like any other");
}

#[test]
#[ignore]
fn two_streams_share_one_engine() {
    let Some(e) = engine() else { return };
    let opts = StreamOptions::default();
    let mut a = Stream::new(&e.gpu, &e.blob, &e.ops, TexOptions::new(W, H), &opts, None).unwrap();
    let mut b = Stream::new(&e.gpu, &e.blob, &e.ops, TexOptions::new(W, H), &opts, None).unwrap();
    let run = |s: &mut Stream, dx0: f32, dx1: f32| {
        let s0 = s.next_input(|_| false).unwrap();
        write(&e, s, s0, &frame(dx0));
        s.push(&e.gpu, s0, 0.0).unwrap();
        let s1 = s.next_input(|_| false).unwrap();
        write(&e, s, s1, &frame(dx1));
        s.push(&e.gpu, s1, 41.7).unwrap();
        assert_eq!(s.generate(&e.gpu, 0.5, 0).unwrap(), Generated::Written);
    };
    run(&mut a, 0.0, 8.0);
    let kernels = e.gpu.p.borrow().len();
    run(&mut b, 4.0, 8.0);
    assert_eq!(e.gpu.p.borrow().len(), kernels, "the second stream runs the kernels the first one compiled");
    let (out_a, out_b) = (read_output(&e, &a, 0), read_output(&e, &b, 0));
    let (to_a, to_b) = (mae(&out_a, &frame(4.0)), mae(&out_b, &frame(6.0)));
    eprintln!("MAE to the true middles {to_a:.2} and {to_b:.2}");
    assert!(to_a < 6.0 && to_b < 6.0, "each stream keeps its own frames and intervals");
    assert_eq!((a.report().counters.mids, b.report().counters.mids), (1, 1));
}

#[test]
#[ignore]
fn video_mode_has_one_frame_of_lookahead_and_cadence_statistics() {
    let Some(e) = engine() else { return };
    let opts = StreamOptions { mode: Mode::Video, cadence: true, inputs: 5, ..Default::default() };
    let mut s = Stream::new(&e.gpu, &e.blob, &e.ops, TexOptions::new(W, H), &opts, None).unwrap();
    assert_eq!(s.warmup(), 3);
    let mut slots = Vec::new();
    for (i, (dx, pts)) in [(0.0, 0.0), (8.0, 41.7), (16.0, 83.4)].into_iter().enumerate() {
        let slot = s.next_input(|_| false).unwrap();
        write(&e, &s, slot, &frame(dx));
        let p = s.push(&e.gpu, slot, pts).unwrap();
        slots.push(slot);
        if i < 2 {
            assert!(p.interval.is_none(), "video mode waits for the lookahead frame");
        } else {
            let iv = p.interval.expect("the interval [f0, f1)");
            assert_eq!((iv.base, iv.start_pts, iv.end_pts), (slots[0], 0.0, 41.7));
            assert!(p.stats_pending, "cadence statistics are read back asynchronously");
            s.resolve_stats(&e.gpu).unwrap();
        }
    }
    assert_eq!(s.generate(&e.gpu, 0.5, 0).unwrap(), Generated::Written);
    assert!(s.report().cadence);
}

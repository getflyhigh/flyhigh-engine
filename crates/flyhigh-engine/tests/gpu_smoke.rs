// End-to-end on a real GPU: load the shipped model, push moving frames through the generator, check the in-betweens.
// Ignored by default (needs a GPU with shader-f16): `cargo test --release -- --ignored`. On a device without f16
// the tests print why and pass, so a CI runner without a usable GPU does not fail the build.
use flyhigh_engine::api::{Engine, Generator, Mode, Options};
use std::path::PathBuf;

const W: u32 = 256;
const H: u32 = 192;

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

// a soft pattern shifted `dx` px to the right: the in-between of two shifts should sit between them
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
fn mae(a: &[u8], b: &[u8]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (*x as f64 - *y as f64).abs()).sum::<f64>() / a.len() as f64
}

#[test]
#[ignore]
fn realtime_inbetween_lies_between_its_neighbours() {
    let Some(e) = engine() else { return };
    let mut g = Generator::new(&e, Options::new(W, H)).unwrap();
    let (f0, f1, mid) = (frame(0.0), frame(8.0), frame(4.0));
    let out0 = g.push(&e, &f0, 2).unwrap();
    assert_eq!(out0.len(), 1, "first push returns the source frame only");
    assert!(out0[0].real);
    let out1 = g.push(&e, &f1, 2).unwrap();
    assert_eq!(out1.len(), 2, "then the in-between and the source frame");
    assert!(!out1[0].real && out1[1].real && (out1[0].phase - 0.5).abs() < 1e-6);
    let gen = &out1[0].rgba;
    assert!(gen.as_chunks::<4>().0.iter().all(|p| p[3] == 255), "alpha is opaque");
    let (to_mid, to_f0, to_f1) = (mae(gen, &mid), mae(gen, &f0), mae(gen, &f1));
    eprintln!("MAE to the true middle {to_mid:.2}, to f0 {to_f0:.2}, to f1 {to_f1:.2}");
    assert!(
        to_mid < to_f0 && to_mid < to_f1,
        "the in-between is closer to the true middle than to either source frame"
    );
    assert!(to_mid < 6.0, "and close in absolute terms");
}

#[test]
#[ignore]
fn video_mode_has_one_frame_of_lookahead() {
    let Some(e) = engine() else { return };
    let mut o = Options::new(W, H);
    o.mode = Mode::Video;
    o.cadence = true;
    let mut g = Generator::new(&e, o).unwrap();
    assert_eq!(g.warmup(), 3);
    assert!(g.push(&e, &frame(0.0), 2).unwrap().is_empty());
    assert!(g.push(&e, &frame(8.0), 2).unwrap().is_empty());
    let out = g.push(&e, &frame(16.0), 2).unwrap();
    assert_eq!(out.len(), 2, "the interval [f0, f1): its real slot and one in-between");
}

#[test]
#[ignore]
fn odd_frame_widths_work() {
    let Some(e) = engine() else { return };
    let mut g = Generator::new(&e, Options::new(266, 150)).unwrap(); // 266 * 4 bytes: not a 256-byte multiple
    let f = vec![200u8; 266 * 150 * 4];
    g.push(&e, &f, 2).unwrap();
    let out = g.push(&e, &f, 2).unwrap();
    assert_eq!(out[0].rgba.len(), 266 * 150 * 4);
}

// a static box (a HUD panel with text-like stripes) over the moving pattern
fn with_hud(mut v: Vec<u8>) -> Vec<u8> {
    for y in 40..100u32 {
        for x in 30..140u32 {
            let o = ((y * W + x) * 4) as usize;
            let s = if (x / 3 + y / 5) % 4 == 0 { 230 } else { 40 };
            v[o] = s;
            v[o + 1] = s;
            v[o + 2] = s / 2;
            v[o + 3] = 255;
        }
    }
    // and a 1 px crosshair
    for i in 0..21u32 {
        for (x, y) in [(118 + i, 140), (128, 130 + i)] {
            let o = ((y * W + x) * 4) as usize;
            v[o] = 255;
            v[o + 1] = 255;
            v[o + 2] = 255;
        }
    }
    v
}
// mean error in a rectangle (x0..x1, y0..y1)
fn rect_mae(a: &[u8], b: &[u8], r: (u32, u32, u32, u32)) -> f64 {
    let (mut s, mut n) = (0f64, 0u64);
    for y in r.2..r.3 {
        for x in r.0..r.1 {
            let o = ((y * W + x) * 4) as usize;
            for c in 0..3 {
                s += (a[o + c] as f64 - b[o + c] as f64).abs();
                n += 1;
            }
        }
    }
    s / n as f64
}
fn hud_mae(a: &[u8], b: &[u8]) -> f64 {
    let (mut s, mut n) = (0f64, 0u64);
    for y in 40..100u32 {
        for x in 30..140u32 {
            let o = ((y * W + x) * 4) as usize;
            for c in 0..3 {
                s += (a[o + c] as f64 - b[o + c] as f64).abs();
                n += 1;
            }
        }
    }
    s / n as f64
}

#[test]
#[ignore]
fn keep_static_holds_a_hud_still_over_motion() {
    let Some(e) = engine() else { return };
    // the HUD has to be static for a while before it is protected: feed a pan of 12 frames, check the last interval
    let mid = with_hud(frame(11.0 * 12.0 + 6.0));
    let run = |keep: bool| {
        let mut o = Options::new(W, H);
        o.keep_static = keep;
        let mut g = Generator::new(&e, o).unwrap();
        let mut last = Vec::new();
        for i in 0..13 {
            let out = g.push(&e, &with_hud(frame(i as f32 * 12.0)), 2).unwrap();
            if out.len() == 2 {
                last = out[0].rgba.clone();
            }
        }
        last
    };
    let young = {
        let mut o = Options::new(W, H);
        o.keep_static = true;
        let mut g = Generator::new(&e, o).unwrap();
        g.push(&e, &with_hud(frame(0.0)), 2).unwrap();
        g.push(&e, &with_hud(frame(12.0)), 2).unwrap().remove(0).rgba
    };
    let young_off = {
        let mut g = Generator::new(&e, Options::new(W, H)).unwrap();
        g.push(&e, &with_hud(frame(0.0)), 2).unwrap();
        g.push(&e, &with_hud(frame(12.0)), 2).unwrap().remove(0).rgba
    };
    // (kernel choices may differ between two builds: compare loosely)
    eprintln!("one static pair: protection on vs off MAE {:.3}", mae(&young, &young_off));
    assert!(mae(&young, &young_off) < 0.05, "one static pair is not enough to protect anything");
    let (off, on) = (run(false), run(true));
    // the crosshair and the world around the HUD (where a ghost of the HUD would be dragged by the motion)
    let (cross, ring) = ((112, 145, 124, 157), (140, 175, 40, 100));
    eprintln!(
        "crosshair MAE: off {:.2}, on {:.2} | right of the HUD: off {:.2}, on {:.2}",
        rect_mae(&off, &mid, cross),
        rect_mae(&on, &mid, cross),
        rect_mae(&off, &mid, ring),
        rect_mae(&on, &mid, ring)
    );
    assert!(
        rect_mae(&on, &mid, cross) < 0.5 * rect_mae(&off, &mid, cross),
        "the crosshair is not smeared along the motion"
    );
    eprintln!(
        "HUD MAE: protection off {:.2}, on {:.2} | whole frame vs true middle: off {:.2}, on {:.2}",
        hud_mae(&off, &mid),
        hud_mae(&on, &mid),
        mae(&off, &mid),
        mae(&on, &mid)
    );
    assert!(hud_mae(&on, &mid) < 0.5, "the HUD is reproduced exactly");
    assert!(hud_mae(&on, &mid) <= hud_mae(&off, &mid));
    assert!(mae(&on, &mid) <= mae(&off, &mid) + 0.05, "and the rest of the frame is not worse");
}

// line art panned `dy` px down: thin dark lines over flat fills, the way anime draws a character
fn line_art(dy: f32) -> Vec<u8> {
    let mut v = vec![0u8; (W * H * 4) as usize];
    for y in 0..H {
        for x in 0..W {
            let (fx, fy) = (x as f32, y as f32 - dy);
            let fill = if fx + 0.3 * fy < 150.0 { [225u8, 205, 170] } else { [170, 140, 100] };
            let on_line = [(fy - 0.25 * fx - 40.0).abs(), (fy + 0.1 * fx - 120.0).abs(), (fx + 0.3 * fy - 150.0).abs()]
                .iter()
                .any(|d| *d < 1.0);
            let c = if on_line { [60u8, 45, 35] } else { fill };
            let o = ((y * W + x) * 4) as usize;
            v[o..o + 3].copy_from_slice(&c);
            v[o + 3] = 255;
        }
    }
    v
}

#[test]
#[ignore]
fn keep_static_leaves_line_art_on_a_pan_alone() {
    let Some(e) = engine() else { return };
    // a fill pixel between where a thin line is in two frames reads unchanged in both: it is not an overlay, and
    // protecting it erased the in-between's line there (a dotted outline on anime pans)
    let run = |keep: bool| {
        let mut o = Options::new(W, H);
        o.keep_static = keep;
        let mut g = Generator::new(&e, o).unwrap();
        let mut last = Vec::new();
        for i in 0..14 {
            let out = g.push(&e, &line_art(i as f32 * 6.0), 2).unwrap();
            if out.len() == 2 {
                last = out[0].rgba.clone();
            }
        }
        last
    };
    let (off, on) = (run(false), run(true));
    let differ = off.chunks(4).zip(on.chunks(4)).filter(|(a, b)| (0..3).any(|c| a[c].abs_diff(b[c]) > 30)).count();
    eprintln!("line art pan: protection on vs off, {differ} pixels differ by more than 30");
    assert!(differ < 20, "protection changed {differ} pixels of moving line art");
}

#[test]
#[ignore]
fn keep_static_after_a_still_camera_has_no_coincidence_ghost() {
    let Some(e) = engine() else { return };
    // the camera stands still for 10 frames (every pixel ages), then pans 12 px a frame under the HUD. Pixels of the
    // moving pattern that match by coincidence must not be held as overlays in the first moving pair (they rendered as
    // a speckled half-step ghost), and the HUD must be held again once it has stayed put over the motion twice.
    let run = |keep: bool| {
        let mut o = Options::new(W, H);
        o.keep_static = keep;
        let mut g = Generator::new(&e, o).unwrap();
        let mut mids = Vec::new();
        for i in 0..14 {
            let dx = if i < 10 { 0.0 } else { (i - 9) as f32 * 12.0 };
            let out = g.push(&e, &with_hud(frame(dx)), 2).unwrap();
            if out.len() == 2 {
                mids.push(out[0].rgba.clone());
            }
        }
        mids
    };
    let (off, on) = (run(false), run(true));
    // mids[k] is the middle of the pair (k, k + 1): the first moving pair is k = 9, its truth the pattern at 6 px
    let truth = |k: usize| with_hud(frame(if k < 9 { 0.0 } else { (k as f32 - 9.0) * 12.0 + 6.0 }));
    let first = (mae(&on[9], &truth(9)), mae(&off[9], &truth(9)));
    eprintln!("first moving pair vs truth: protection on {:.2}, off {:.2}", first.0, first.1);
    assert!(first.0 <= first.1 + 0.05, "the first moving pair is no worse with protection than without");
    let k = on.len() - 1;
    eprintln!("HUD after the pan: on {:.2}, off {:.2}", hud_mae(&on[k], &truth(k)), hud_mae(&off[k], &truth(k)));
    assert!(hud_mae(&on[k], &truth(k)) < 0.5, "the HUD is held again once verified over the motion");
}

#[test]
#[ignore]
fn fast_render_matches_the_exact_render() {
    let Some(e) = engine() else { return };
    // above 960 px, so the network runs at the 480-line rung and the fast render applies
    let (w, h) = (1280u32, 720u32);
    let frame = |dx: f32| -> Vec<u8> {
        let mut v = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let o = ((y * w + x) * 4) as usize;
                let u = (x as f32 - dx) * 0.05;
                v[o] = (128.0 + 100.0 * u.sin()) as u8;
                v[o + 1] = (128.0 + 90.0 * (u * 0.7 + y as f32 * 0.03).cos()) as u8;
                v[o + 2] = (128.0 + 80.0 * (y as f32 * 0.04).sin()) as u8;
                v[o + 3] = 255;
            }
        }
        v
    };
    let run = |fast: bool, linear: bool| {
        let mut o = Options::new(w, h);
        o.fast_render = fast;
        o.linear_t = linear;
        let mut g = Generator::new(&e, o).unwrap();
        g.push(&e, &frame(0.0), 4).unwrap();
        g.push(&e, &frame(16.0), 4).unwrap().into_iter().filter(|f| !f.real).map(|f| f.rgba).collect::<Vec<_>>()
    };
    let (exact, fast, lin) = (run(false, false), run(true, false), run(true, true));
    for i in 0..3 {
        let truth = frame(4.0 * (i + 1) as f32);
        eprintln!(
            "phase {}: exact {:.2}, fast {:.2}, fast+linear {:.2} (MAE to the true frame); fast vs exact {:.3}",
            i + 1,
            mae(&exact[i], &truth),
            mae(&fast[i], &truth),
            mae(&lin[i], &truth),
            mae(&fast[i], &exact[i])
        );
        assert!(mae(&fast[i], &exact[i]) < 0.5, "the fast render reproduces the exact one");
        assert!(mae(&lin[i], &truth) < mae(&exact[i], &truth) + 1.0, "linear t stays close on linear motion");
    }
}

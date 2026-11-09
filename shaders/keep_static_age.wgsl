enable f16;
// Static protection, age pass: see keep_static.wgsl.
struct P { w: u32, h: u32, wp: u32, hp: u32, thr: f32, min_age: f32, use_mask: u32, _b: u32 };
@group(0) @binding(0) var<storage, read> a: array<f16>;
@group(0) @binding(1) var<storage, read> b: array<f16>;
@group(0) @binding(2) var<storage, read_write> outp: array<f16>;
@group(0) @binding(3) var<uniform> p: P;
// the host's protection mask (option protection_mask): 1 where never to interpolate, read as age 255; the zero
// buffer stands in when the session has none (use_mask 0)
@group(0) @binding(4) var<storage, read> mask: array<f16>;
@group(0) @binding(5) var<storage, read_write> state: array<f16>;   // [2][hp][wp]

// state: [0] consecutive unchanged pairs, [1] of those, pairs in which something changed around the pixel (the 3 x 3
// neighbourhood, or 8 directions at 4, 8, 16 and 32 px, so the inside of a large HUD panel counts too). A pixel is
// protected only after it stayed unchanged while its surroundings moved, twice: that is what an overlay over moving
// content looks like, while a pixel of a moving texture that matches by coincidence (most likely in the first pair after
// the camera stood still, when every pixel carries a high age) does not repeat it.
const R: i32 = 1;
const DIRS = array<vec2<i32>, 8>(vec2<i32>(1, 0), vec2<i32>(-1, 0), vec2<i32>(0, 1), vec2<i32>(0, -1),
                                 vec2<i32>(1, 1), vec2<i32>(-1, 1), vec2<i32>(1, -1), vec2<i32>(-1, -1));
const FAR = array<i32, 4>(4, 8, 16, 32);
const VERIFIED: f32 = 2.0;
// a pixel flat in both frames is no evidence of standing still (see keep_static_age_tex.wgsl)
const FLAT: f32 = 0.03;
fn decide(o: u32, plane: u32, still: bool, moved: bool) -> f16 {
    let raw = select(0.0, min(f32(state[o]) + 1.0, 255.0), still);
    let seen = select(0.0, min(f32(state[plane + o]) + select(0.0, 1.0, moved), 255.0), still);
    state[o] = f16(raw);
    state[plane + o] = f16(seen);
    return f16(select(0.0, raw, seen >= VERIFIED));
}

fn spread(x: u32, y: u32, plane: u32) -> f32 {
    let o = y * p.wp + x;
    var m = 0.0;
    for (var dy = -1; dy <= 1; dy++) { for (var dx = -1; dx <= 1; dx++) {
        let xx = u32(clamp(i32(x) + dx, 0, i32(p.w) - 1)); let yy = u32(clamp(i32(y) + dy, 0, i32(p.h) - 1));
        let q = yy * p.wp + xx;
        for (var c = 0u; c < 3u; c++) {
            m = max(m, abs(f32(a[c * plane + q]) - f32(a[c * plane + o])));
            m = max(m, abs(f32(b[c * plane + q]) - f32(b[c * plane + o])));
        }
    } }
    return m;
}

fn diff(o: u32, plane: u32) -> f32 {
    var d = 0.0;
    for (var c = 0u; c < 3u; c++) { d = max(d, abs(f32(a[c * plane + o]) - f32(b[c * plane + o]))); }
    return d;
}

// outp = the age the render and the apply pass protect by
@compute @workgroup_size(8, 8, 1)
fn age(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.w || gid.y >= p.h) { return; }
    let plane = p.hp * p.wp;
    let o = gid.y * p.wp + gid.x;
    var moved = false;
    for (var dy = -R; dy <= R; dy++) { for (var dx = -R; dx <= R; dx++) {
        let xx = u32(clamp(i32(gid.x) + dx, 0, i32(p.w) - 1)); let yy = u32(clamp(i32(gid.y) + dy, 0, i32(p.h) - 1));
        moved = moved || diff(yy * p.wp + xx, plane) >= p.thr;
    } }
    for (var i = 0; i < 8; i++) { for (var j = 0; j < 4; j++) {
        let q = vec2<i32>(i32(gid.x), i32(gid.y)) + DIRS[i] * FAR[j];
        let xx = u32(clamp(q.x, 0, i32(p.w) - 1)); let yy = u32(clamp(q.y, 0, i32(p.h) - 1));
        moved = moved || diff(yy * p.wp + xx, plane) >= p.thr;
    } }
    let aged = decide(o, plane, diff(o, plane) < p.thr && spread(gid.x, gid.y, plane) >= FLAT, moved);
    outp[o] = select(aged, f16(255.0), p.use_mask != 0u && f32(mask[o]) > 0.5);
}

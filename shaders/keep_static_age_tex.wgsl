enable f16;
// Static protection, age pass on the fast render path: as keep_static_age.wgsl, reading the pair's two source textures
// (replicate-padded to the working size) instead of full-resolution tensors.
struct P { w: u32, h: u32, wp: u32, hp: u32, thr: f32, min_age: f32, use_mask: u32, _b: u32 };
@group(0) @binding(0) var tex_a: texture_2d<f32>;
@group(0) @binding(1) var tex_b: texture_2d<f32>;
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
// A pixel whose 3 x 3 neighbourhood is one flat colour in both frames carries no evidence of standing still: in a
// panned flat fill it reads unchanged while the content moves, and between where a thin line is in the two frames it
// is fill in both, so protecting it erased the in-between's line there (a dotted, broken outline on anime pans). An
// overlay has an edge of its own (a crosshair line, text, a panel's border); a flat panel inside needs no protection.
const FLAT: f32 = 0.03;
fn decide(o: u32, plane: u32, still: bool, moved: bool) -> f16 {
    let raw = select(0.0, min(f32(state[o]) + 1.0, 255.0), still);
    let seen = select(0.0, min(f32(state[plane + o]) + select(0.0, 1.0, moved), 255.0), still);
    state[o] = f16(raw);
    state[plane + o] = f16(seen);
    return f16(select(0.0, raw, seen >= VERIFIED));
}
fn spread(t: vec2<i32>, hi: vec2<i32>) -> f32 {
    let ca = textureLoad(tex_a, t, 0).rgb;
    let cb = textureLoad(tex_b, t, 0).rgb;
    var m = 0.0;
    for (var dy = -1; dy <= 1; dy++) { for (var dx = -1; dx <= 1; dx++) {
        let q = clamp(t + vec2<i32>(dx, dy), vec2<i32>(0), hi);
        let da = abs(textureLoad(tex_a, q, 0).rgb - ca);
        let db = abs(textureLoad(tex_b, q, 0).rgb - cb);
        m = max(m, max(max(da.r, max(da.g, da.b)), max(db.r, max(db.g, db.b))));
    } }
    return m;
}
fn change(t: vec2<i32>) -> f32 {
    let d = abs(textureLoad(tex_a, t, 0).rgb - textureLoad(tex_b, t, 0).rgb);
    return max(d.r, max(d.g, d.b));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.wp || gid.y >= p.hp) { return; }
    let t = vec2<i32>(i32(min(gid.x, p.w - 1u)), i32(min(gid.y, p.h - 1u)));
    let hi = vec2<i32>(i32(p.w) - 1, i32(p.h) - 1);
    var moved = false;
    for (var dy = -R; dy <= R; dy++) { for (var dx = -R; dx <= R; dx++) {
        moved = moved || change(clamp(t + vec2<i32>(dx, dy), vec2<i32>(0), hi)) >= p.thr;
    } }
    for (var i = 0; i < 8; i++) { for (var j = 0; j < 4; j++) {
        moved = moved || change(clamp(t + DIRS[i] * FAR[j], vec2<i32>(0), hi)) >= p.thr;
    } }
    let o = gid.y * p.wp + gid.x;
    let aged = decide(o, p.hp * p.wp, change(t) < p.thr && spread(t, hi) >= FLAT, moved);
    outp[o] = select(aged, f16(255.0), p.use_mask != 0u && f32(mask[o]) > 0.5);
}

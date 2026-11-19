enable f16;
// Use the translation where both warped originals still agree. Occlusions, independently moving elements and the
// revealed frame border retain the network result already in dst.
@group(0) @binding(0) var a: texture_2d<f32>;
@group(0) @binding(1) var b: texture_2d<f32>;
@group(0) @binding(2) var linear_sampler: sampler;
@group(0) @binding(3) var<storage, read> fit: vec4<f32>;
@group(0) @binding(4) var dst: texture_storage_2d<rgba8unorm, write>;
struct Phase { t: f32, min_age: f32, _b: f32, _c: f32 }
@group(0) @binding(5) var<uniform> phase: Phase;
@group(0) @binding(6) var<storage, read> ages: array<f16>;

fn age_at(p: vec2<u32>, stride: u32) -> f32 {
    return f32(ages[p.y * stride + p.x]);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(dst);
    if (fit.z == 0.0 || any(id.xy >= size)) { return; }
    let p = vec2<f32>(id.xy) + 0.5;
    let pa = p - phase.t * fit.xy;
    let pb = p + (1.0 - phase.t) * fit.xy;
    let lo = vec2<f32>(0.5);
    let hi = vec2<f32>(size) - lo;
    if (any(pa < lo) || any(pa > hi) || any(pb < lo) || any(pb > hi)) { return; }
    if (phase.min_age > 0.0) {
        let stride = ((size.x + 31u) / 32u) * 32u;
        if (age_at(id.xy, stride) > phase.min_age - 2.0
            || age_at(vec2<u32>(pa), stride) >= phase.min_age
            || age_at(vec2<u32>(pb), stride) >= phase.min_age) { return; }
    }
    let ca = textureSampleLevel(a, linear_sampler, pa / vec2<f32>(size), 0.0).rgb;
    let cb = textureSampleLevel(b, linear_sampler, pb / vec2<f32>(size), 0.0).rgb;
    if (any(abs(ca - cb) > vec3<f32>(0.06))) { return; }
    textureStore(dst, id.xy, vec4<f32>(mix(ca, cb, phase.t), 1.0));
}

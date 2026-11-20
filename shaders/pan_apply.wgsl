enable f16;
// Use the translation where the originals agree on the background motion.
// Independently moving elements retain the network result already in dst.
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

fn rgb_peak(ca: vec3<f32>, cb: vec3<f32>) -> f32 {
    let d = abs(ca - cb);
    return max(max(d.r, d.g), d.b);
}

struct Patch { mean_error: f32, max_error: f32, same_detail: bool }

fn compare_patch(p: vec2<f32>, dims: vec2<f32>, shift_a: vec2<f32>, shift_b: vec2<f32>,
                 center_a: vec3<f32>, center_b: vec3<f32>, compare_detail: bool) -> Patch {
    let neighbours = array<vec2<f32>, 4>(
        vec2<f32>(-2.0, 0.0), vec2<f32>(2.0, 0.0), vec2<f32>(0.0, -2.0), vec2<f32>(0.0, 2.0)
    );
    var sum = 0.0;
    var peak = 0.0;
    var detail_a = vec4<f32>(0.0);
    var detail_b = vec4<f32>(0.0);
    let luma = vec3<f32>(0.2126, 0.7152, 0.0722);
    for (var i = 0u; i < 4u; i++) {
        let q = p + neighbours[i];
        let ca = textureSampleLevel(a, linear_sampler, (q + shift_a) / dims, 0.0).rgb;
        let cb = textureSampleLevel(b, linear_sampler, (q + shift_b) / dims, 0.0).rgb;
        let error = rgb_peak(ca, cb);
        sum += error;
        peak = max(peak, error);
        if (compare_detail) {
            detail_a[i] = dot(center_a - ca, luma);
            detail_b[i] = dot(center_b - cb, luma);
        }
    }
    // A thin edge sampled at different subpixel positions can have different brightness in each
    // source. Compare the signed local contrast pattern so it survives without accepting unrelated pixels.
    let aa = dot(detail_a, detail_a);
    let bb = dot(detail_b, detail_b);
    let ab = dot(detail_a, detail_b);
    let same_detail = min(aa, bb) > 0.02 && min(aa, bb) > 0.42 * max(aa, bb)
        && ab > 0.0 && ab * ab > 0.94 * aa * bb;
    return Patch(sum * 0.25, peak, same_detail);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(dst);
    if (fit.z <= 0.0 || any(id.xy >= size)) { return; }
    let dims = vec2<f32>(size);
    let p = vec2<f32>(id.xy) + 0.5;
    let shift_a = -phase.t * fit.xy;
    let shift_b = (1.0 - phase.t) * fit.xy;
    let pa = p + shift_a;
    let pb = p + shift_b;
    let lo = vec2<f32>(0.5);
    let hi = dims - lo;
    let has_a = all(pa >= lo) && all(pa <= hi);
    let has_b = all(pb >= lo) && all(pb <= hi);
    let ca = textureSampleLevel(a, linear_sampler, pa / dims, 0.0).rgb;
    let cb = textureSampleLevel(b, linear_sampler, pb / dims, 0.0).rgb;
    if (has_a && has_b) {
        // Compression and subpixel sampling can leave isolated mismatches on an otherwise aligned line.
        let peak = rgb_peak(ca, cb);
        if (peak > 0.06) {
            if (peak > 0.16) { return; }
            let local = compare_patch(p, dims, shift_a, shift_b, ca, cb, true);
            let colour_match = local.mean_error <= 0.045 && local.max_error <= 0.10;
            if (!colour_match && !(local.same_detail && local.max_error <= 0.13)) { return; }
        }
    } else {
        // The pan reveals or hides a strip at the frame edge. Only one source sees those pixels.
        // Check the adjacent overlap before using that source in place of a two-sided warp.
        if (!has_a && !has_b) { return; }
        let overlap_lo = lo - min(shift_a, shift_b) + 2.0;
        let overlap_hi = hi - max(shift_a, shift_b) - 2.0;
        if (any(overlap_lo > overlap_hi)
            || compare_patch(clamp(p, overlap_lo, overlap_hi), dims, shift_a, shift_b, ca, cb, false).max_error > 0.045) {
            return;
        }
    }
    if (phase.min_age > 0.0) {
        let stride = ((size.x + 31u) / 32u) * 32u;
        let ia = vec2<u32>(clamp(pa, lo, hi));
        let ib = vec2<u32>(clamp(pb, lo, hi));
        if (age_at(id.xy, stride) > phase.min_age - 2.0
            || (has_a && age_at(ia, stride) >= phase.min_age)
            || (has_b && age_at(ib, stride) >= phase.min_age)) { return; }
    }
    let weight = select(select(1.0, 0.0, has_a), phase.t, has_a && has_b);
    textureStore(dst, id.xy, vec4<f32>(mix(ca, cb, weight), 1.0));
}

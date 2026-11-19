// Pair classification for the Stream (stream/classify.rs): the two source frames sampled on a gw x gh grid (one
// bilinear sample at each cell centre), the sum over cells of the rgb absolute difference in 1/255 steps (0..765 per
// cell), its maximum, and the strongest coherent brightness change in an 8 x 9 tile.
struct Par { gw: u32, gh: u32, _a: u32, _b: u32 };
@group(0) @binding(0) var tex_a: texture_2d<f32>;
@group(0) @binding(1) var tex_b: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<storage, read_write> stats: array<atomic<u32>, 3>;
@group(0) @binding(4) var<uniform> pp: Par;

var<workgroup> luma_change: array<f32, 72>;

@compute @workgroup_size(8, 9, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) i: u32) {
    var luma = 0.0;
    if (gid.x < pp.gw && gid.y < pp.gh) {
        let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(f32(pp.gw), f32(pp.gh));
        let a = textureSampleLevel(tex_a, samp, uv, 0.0).rgb;
        let b = textureSampleLevel(tex_b, samp, uv, 0.0).rgb;
        let delta = b - a;
        let d = u32(dot(abs(delta), vec3<f32>(255.0)));
        atomicAdd(&stats[0], d);
        atomicMax(&stats[1], d);
        luma = dot(delta, vec3<f32>(0.2126, 0.7152, 0.0722));
    }
    luma_change[i] = luma;
    workgroupBarrier();
    if (i == 0u) {
        var signed = 0.0;
        var total = 0.0;
        for (var j = 0u; j < 72u; j++) {
            let change = luma_change[j];
            signed += change;
            total += abs(change);
        }
        // Motion normally makes both brighter and darker samples in a tile. A flash changes them in one direction.
        if (abs(signed) >= 0.74 * total) {
            atomicMax(&stats[2], u32(round(abs(signed) * 255.0 / 72.0)));
        }
    }
}

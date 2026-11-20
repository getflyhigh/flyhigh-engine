// Score a translation on the original textures. Each workgroup tests one candidate.
struct Search { w: u32, h: u32, width: u32, divisor: u32, initial: u32, finish: u32, _a: u32, _b: u32 }
@group(0) @binding(0) var a: texture_2d<f32>;
@group(0) @binding(1) var b: texture_2d<f32>;
@group(0) @binding(2) var linear_sampler: sampler;
@group(0) @binding(3) var<storage, read> fit: vec4<f32>;
@group(0) @binding(4) var<storage, read_write> errors: array<f32>;
@group(0) @binding(5) var<uniform> search: Search;

const N = 192u;
var<workgroup> sums: array<f32, 64>;
var<workgroup> skip: u32;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) candidate: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    // A negative flag means the initial zero-shift comparison already ruled out a pan.
    if (search.initial == 0u) {
        if (lane == 0u) { skip = u32(fit.z < 0.0); }
        if (workgroupUniformLoad(&skip) != 0u) { return; }
    }
    let size = vec2<f32>(f32(search.w), f32(search.h));
    let spacing = max(1.0, size.y / 144.0) / f32(search.divisor);
    let offset = (vec2<f32>(f32(candidate.x % search.width), f32(candidate.x / search.width))
        - f32(search.width / 2u)) * spacing;
    let base = select(fit.xy, vec2<f32>(0.0), search.initial != 0u);
    let delta = (base + offset) / size;
    var sum = 0.0;
    for (var i = lane; i < N; i += 64u) {
        // Jitter the sparse grid so a periodic edge can't always land between sample rows.
        let jitter = vec2<f32>(f32((i * 13u) % 17u), f32((i * 11u) % 13u)) / vec2<f32>(17.0, 13.0);
        let uv = 0.12 + 0.76 * (vec2<f32>(f32(i % 16u), f32(i / 16u)) + jitter) / vec2<f32>(16.0, 12.0);
        let ca = textureSampleLevel(a, linear_sampler, uv, 0.0).rgb;
        let cb = textureSampleLevel(b, linear_sampler, uv + delta, 0.0).rgb;
        sum += dot(abs(ca - cb), vec3<f32>(1.0 / 3.0));
    }
    sums[lane] = sum;
    workgroupBarrier();
    for (var step = 32u; step > 0u; step >>= 1u) {
        if (lane < step) { sums[lane] += sums[lane + step]; }
        workgroupBarrier();
    }
    if (lane == 0u) { errors[candidate.x] = sums[0] / f32(N); }
}

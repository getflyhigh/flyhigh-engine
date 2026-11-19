// Retain the best candidate for the next refinement. Only a close match with a large gain over zero shift
// can replace the network's motion in the renderer.
struct Search { w: u32, h: u32, width: u32, divisor: u32, initial: u32, finish: u32, _a: u32, _b: u32 }
@group(0) @binding(0) var<storage, read> errors: array<f32>;
@group(0) @binding(1) var<storage, read_write> fit: vec4<f32>;
@group(0) @binding(2) var<uniform> search: Search;

@compute @workgroup_size(1)
fn main() {
    let size = vec2<f32>(f32(search.w), f32(search.h));
    let count = search.width * search.width;
    let center = count / 2u;
    var best = center;
    for (var j = 0u; j < count; j++) {
        if (errors[j] < errors[best]) { best = j; }
    }
    let spacing = max(1.0, size.y / 144.0) / f32(search.divisor);
    let offset = (vec2<f32>(f32(best % search.width), f32(best / search.width))
        - f32(search.width / 2u)) * spacing;
    let shift = select(fit.xy, vec2<f32>(0.0), search.initial != 0u) + offset;
    let still_error = select(fit.w, errors[center], search.initial != 0u);
    let trusted = search.finish != 0u && errors[best] < 0.010 && errors[best] < 0.30 * still_error
        && still_error > 0.016 && length(shift) > 1.0 && all(abs(shift) < size * 0.10);
    fit = vec4<f32>(shift, select(0.0, 1.0, trusted), still_error);
}

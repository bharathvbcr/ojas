// Layout moves. Values travel as u32 words, never through f32 arithmetic, so
// the output bits are the input bits: NaN payloads, -0.0 and subnormals
// included. The words are f32 values, and a non-finite one (exponent all
// ones) raises the launching op's fault bit like every other op's output;
// it is still moved unchanged.
//
// permute: one lane per output element. Words: 0 element count, 1 rank.
// geom holds `rank` output extents, then `rank` input strides (in elements)
// listed in output-axis order. Output element i is input element
// sum_a coord_a(i) * geom[rank + a], with coord_a the row-major digits of i.

@group(0) @binding(2) var<storage, read> src: array<u32>;
@group(0) @binding(3) var<storage, read> geom: array<u32>;
@group(0) @binding(4) var<storage, read_write> dst: array<u32>;
@group(0) @binding(5) var<storage, read> status_in: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn permute(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = flat_group(wg, nwg) * 256u + lid.x;
    if (i >= pw(0u)) { return; }
    let rank = pw(1u);
    var rem = i;
    var at = 0u;
    for (var a = rank; a > 0u; a = a - 1u) {
        let extent = geom[a - 1u];
        at = at + (rem % extent) * geom[rank + a - 1u];
        rem = rem / extent;
    }
    let w = src[at];
    if ((w & 0x7f800000u) == 0x7f800000u) { raise(); }
    dst[i] = w;
}

// kv_cache_write: src [B, Tn, row] into dst [B, Tcap, row] at time `at`, one
// lane per source word. Words: 0 element count (B * Tn * row), 1 Tn * row,
// 2 Tcap * row, 3 at * row. status_in is this call's fault word, set by a
// finiteness check of src: if it is set nothing is written and the op's bit
// is raised in the context's word, so the cache is never half-written.
@compute @workgroup_size(256, 1, 1)
fn kv_write(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = flat_group(wg, nwg) * 256u + lid.x;
    if (i >= pw(0u)) { return; }
    // Both mask words: the op's bit is in word 1 when its index is 32 or more.
    if ((status_in[0] | status_in[1]) != 0u) {
        if (i == 0u) { raise(); }
        return;
    }
    let per_src = pw(1u);
    dst[(i / per_src) * pw(2u) + pw(3u) + i % per_src] = src[i];
}

// Multi-tensor global L2 norm and in-place scale, for clip_grad_norm and
// Muon's normalization. One dispatch binds up to {{K}} gradients (g0.. for
// the norm, w0.. for the scale); `tbl` holds every gradient of the call as
// (element count, index of its first CHUNK-element chunk among all the
// call's chunks). Words: 0 the dispatch's first gradient, 1 gradients bound,
// 2 the dispatch's first chunk. Workgroup g takes chunk word2 + g, finds its
// slot (uniform across the workgroup) and reads its chunk once.
//
// The norm is amax * sqrt(sum((g / amax)^2)), so the f32 sum of squares
// cannot overflow while the norm is finite: each chunk keeps its own max m
// and its sum of (g / m)^2, and the finish rescales every chunk to the
// global max. `status` is four words per call: 0 norm bits, 1 non-finite
// flag, 2 amax bits.

const CHUNK: u32 = 4096u;
const PER: u32 = 16u; // CHUNK / 256

@group(0) @binding(2) var<storage, read_write> part: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> tbl: array<u32>;
@group(0) @binding(4) var<storage, read_write> status: array<atomic<u32>>;
{{DECLS}}

fn load(s: u32, i: u32) -> f32 {
    switch s {
{{LOAD}}
        default: { return 0.0; }
    }
}

fn scale_at(s: u32, i: u32, k: f32) {
    switch s {
{{STORE}}
        default: {}
    }
}

// The slot of the dispatch chunk `c` falls in: the last bound gradient
// whose first chunk is at or before it.
fn slot_of(c: u32) -> u32 {
    let t0 = pw(0u);
    var s = 0u;
    for (var k = 1u; k < pw(1u); k = k + 1u) {
        if (c >= tbl[2u * (t0 + k) + 1u]) {
            s = k;
        }
    }
    return s;
}

// part[c] = (max |g|, sum (g / max)^2) of chunk c. A non-finite element sets
// status[1] and is left out of both.
@compute @workgroup_size(256, 1, 1)
fn norm_partial(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let c = pw(2u) + flat_group(wg, nwg);
    let s = slot_of(c);
    let t = pw(0u) + s;
    let n = tbl[2u * t];
    let lo = (c - tbl[2u * t + 1u]) * CHUNK;
    var v: array<f32, 16>;
    var peak = 0.0;
    var bad = false;
    for (var j = 0u; j < PER; j = j + 1u) {
        let i = lo + j * 256u + lid.x;
        var a = 0.0;
        if (i < n) {
            a = load(s, i);
        }
        if (nonfinite(a)) {
            bad = true;
            a = 0.0;
        }
        v[j] = a;
        peak = max(peak, abs(a));
    }
    if (bad) {
        atomicOr(&status[1], 1u);
    }
    let m = tree_max(lid.x, peak);
    var acc = 0.0;
    if (m > 0.0) {
        for (var j = 0u; j < PER; j = j + 1u) {
            let r = v[j] / m;
            acc = acc + r * r;
        }
    }
    let total = tree_sum(lid.x, acc);
    if (lid.x == 0u) {
        part[c] = vec2<f32>(m, total);
    }
}

// Words: 0 chunk count. One workgroup: the global max, then every chunk's
// sum rescaled to it. status[0] = norm bits, status[2] = amax bits.
@compute @workgroup_size(256, 1, 1)
fn norm_finish(@builtin(local_invocation_id) lid: vec3<u32>) {
    let count = pw(0u);
    var peak = 0.0;
    for (var i = lid.x; i < count; i = i + 256u) {
        peak = max(peak, part[i].x);
    }
    let big = tree_max(lid.x, peak);
    var acc = 0.0;
    if (big > 0.0) {
        for (var i = lid.x; i < count; i = i + 256u) {
            let r = part[i].x / big;
            acc = acc + part[i].y * (r * r);
        }
    }
    let total = tree_sum(lid.x, acc);
    if (lid.x == 0u) {
        atomicStore(&status[2], bitcast<u32>(big));
        atomicStore(&status[0], bitcast<u32>(big * sqrt(total)));
    }
}

// Word 3: the scale. Every bound gradient times it, in place; a non-finite
// product raises the op's fault bit.
@compute @workgroup_size(256, 1, 1)
fn scale_multi(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let c = pw(2u) + flat_group(wg, nwg);
    let s = slot_of(c);
    let t = pw(0u) + s;
    let n = tbl[2u * t];
    let lo = (c - tbl[2u * t + 1u]) * CHUNK;
    let k = pf(3u);
    for (var j = 0u; j < PER; j = j + 1u) {
        let i = lo + j * 256u + lid.x;
        if (i < n) {
            scale_at(s, i, k);
        }
    }
}

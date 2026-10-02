// Causal attention of Tq new queries against the first kv_len positions of a
// time-major KV cache, split over the keys so a single decode query still
// fills the GPU.
//
// q is [B, Tq, H, D]; k and v are [B, Tcap, Hkv, D]. Query i sits at position
// kv_len - Tq + i and reads keys 0..=that position; head h reads KV head
// h / (H / Hkv). Nothing at or past a row's own position is read, so values
// there (stale or never written) cannot reach the output.
//
// cattn_split: one workgroup per (split, query row * head, batch) computes,
// over its key range, the running max m, the sum l of exp(score - m) and the
// unnormalised output acc[d] = sum exp(score - m) * v[d], in ascending key
// order. cattn_merge: one workgroup per (query row * head, batch) combines
// the splits in ascending order, so results are identical run to run. Every
// score is reported, so a -inf key (whose exp is a clean 0) still faults.
//
// Words: 0 B, 1 Tq, 2 H, 3 Hkv, 4 D, 5 Tcap, 6 kv_len, 7 split length S
// (multiple of 64, at most MAX_SPLIT), 8 split count, 9 scale bits.
// Partial layout per (b, i, h, split): [m, l, acc[0..D]], D + 2 words.

@group(0) @binding(2) var<storage, read> q: array<f32>;
@group(0) @binding(3) var<storage, read> kc: array<f32>;
@group(0) @binding(4) var<storage, read> vc: array<f32>;
@group(0) @binding(5) var<storage, read_write> part: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(7) var<storage, read> part_in: array<f32>;

const LANES: u32 = 64u;
const MAX_SPLIT: u32 = 1024u;
const MAX_D: u32 = 128u;
const NEG: f32 = -3.0e38;

var<workgroup> q_s: array<f32, MAX_D>;
var<workgroup> p_s: array<f32, MAX_SPLIT>;
var<workgroup> red: array<f32, LANES>;

fn wg_max(lane: u32, x: f32) -> f32 {
    red[lane] = x;
    workgroupBarrier();
    for (var w = LANES / 2u; w > 0u; w = w / 2u) {
        if (lane < w) { red[lane] = max(red[lane], red[lane + w]); }
        workgroupBarrier();
    }
    let r = red[0];
    workgroupBarrier();
    return r;
}

fn wg_sum(lane: u32, x: f32) -> f32 {
    red[lane] = x;
    workgroupBarrier();
    for (var w = LANES / 2u; w > 0u; w = w / 2u) {
        if (lane < w) { red[lane] = red[lane] + red[lane + w]; }
        workgroupBarrier();
    }
    let r = red[0];
    workgroupBarrier();
    return r;
}

@compute @workgroup_size(64, 1, 1)
fn cattn_split(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tq = pw(1u);
    let heads = pw(2u);
    let kv_heads = pw(3u);
    let dim = pw(4u);
    let cap = pw(5u);
    let kv_len = pw(6u);
    let split = pw(7u);
    let splits = pw(8u);
    let scale = pf(9u);
    let s = wg.x;
    let i = wg.y / heads;
    let h = wg.y % heads;
    let b = wg.z;
    let lane = lid.x;
    let pos = kv_len - tq + i;
    let start = s * split;
    let end = min(start + split, pos + 1u);
    let base = (((b * tq + i) * heads + h) * splits + s) * (dim + 2u);
    if (start >= end) {
        // Past this row's position: an empty split.
        if (lane == 0u) {
            part[base] = NEG;
            part[base + 1u] = 0.0;
        }
        for (var d = lane; d < dim; d = d + LANES) {
            part[base + 2u + d] = 0.0;
        }
        return;
    }
    let qbase = ((b * tq + i) * heads + h) * dim;
    for (var d = lane; d < dim; d = d + LANES) {
        q_s[d] = q[qbase + d];
    }
    workgroupBarrier();
    let kvh = h / (heads / kv_heads);
    let row = kv_heads * dim;
    let kvbase = (b * cap) * row + kvh * dim;
    var peak = NEG;
    for (var j = start + lane; j < end; j = j + LANES) {
        let kb = kvbase + j * row;
        var dot = 0.0;
        for (var d = 0u; d < dim; d = d + 1u) {
            dot = dot + q_s[d] * kc[kb + d];
        }
        let sc = dot * scale;
        report(sc);
        p_s[j - start] = sc;
        peak = max(peak, sc);
    }
    let m = wg_max(lane, peak);
    var acc_l = 0.0;
    for (var j = start + lane; j < end; j = j + LANES) {
        let p = exp(p_s[j - start] - m);
        p_s[j - start] = p;
        acc_l = acc_l + p;
    }
    let l = wg_sum(lane, acc_l);
    for (var d = lane; d < dim; d = d + LANES) {
        var acc = 0.0;
        for (var j = start; j < end; j = j + 1u) {
            acc = acc + p_s[j - start] * vc[kvbase + j * row + d];
        }
        part[base + 2u + d] = acc;
    }
    if (lane == 0u) {
        part[base] = m;
        part[base + 1u] = l;
    }
}

@compute @workgroup_size(128, 1, 1)
fn cattn_merge(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tq = pw(1u);
    let heads = pw(2u);
    let dim = pw(4u);
    let splits = pw(8u);
    let d = lid.x;
    if (d >= dim) { return; }
    let i = wg.x / heads;
    let h = wg.x % heads;
    let b = wg.y;
    let rowpart = ((b * tq + i) * heads + h) * splits;
    var m = NEG;
    for (var s = 0u; s < splits; s = s + 1u) {
        m = max(m, part_in[(rowpart + s) * (dim + 2u)]);
    }
    var l = 0.0;
    var acc = 0.0;
    for (var s = 0u; s < splits; s = s + 1u) {
        let at = (rowpart + s) * (dim + 2u);
        let w = exp(part_in[at] - m);
        l = l + w * part_in[at + 1u];
        acc = acc + w * part_in[at + 2u + d];
    }
    let o = acc / l;
    report(o);
    out[((b * tq + i) * heads + h) * dim + d] = o;
}

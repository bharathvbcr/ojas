// Causal flash attention, [batch * heads, time, D] contiguous, D fixed per
// pipeline. A workgroup of 64 lanes owns 64 rows of one head and streams the
// other operand through shared tiles of TILE rows. Scores live in registers;
// no time x time matrix is written anywhere. Each row's loop stops at its own
// position, so a future key never enters a result, not even as zero times a
// value, and a NaN there cannot leak backwards. The forward and prep kernels
// run an online softmax; dq and dkv rebuild probabilities from the saved
// log-sum-exp.
//
// Words: 0 time, 1 scale bits, 2 batch * heads * time.
// Grid (ceil(time / 64), batch * heads).
//
// stats holds the row log-sum-exp at [row] and delta = dO . O at
// [bht + row]; the backward prep kernel writes it, dq and dkv read it.

const D: u32 = {{D}}u;
const TILE: u32 = {{TILE}}u;
const ROWS: u32 = 64u;

@group(0) @binding(2) var<storage, read> aq: array<f32>;
@group(0) @binding(3) var<storage, read> ak: array<f32>;
@group(0) @binding(4) var<storage, read> av: array<f32>;
@group(0) @binding(5) var<storage, read> ado: array<f32>;
@group(0) @binding(6) var<storage, read> astats: array<f32>;
@group(0) @binding(7) var<storage, read_write> out0: array<f32>;
@group(0) @binding(8) var<storage, read_write> out1: array<f32>;

var<workgroup> tile0: array<f32, {{TILE_D}}>;
var<workgroup> tile1: array<f32, {{TILE_D}}>;
var<workgroup> tile_s: array<f32, {{TILE2}}>;

fn load_kv(lane: u32, base: u32, kb: u32, time: u32) {
    for (var e = lane; e < TILE * D; e = e + ROWS) {
        let j = kb + e / D;
        var kv = 0.0;
        var vv = 0.0;
        if (j < time) {
            kv = ak[base + j * D + e % D];
            vv = av[base + j * D + e % D];
        }
        tile0[e] = kv;
        tile1[e] = vv;
    }
}

@compute @workgroup_size(64, 1, 1)
fn attn_fwd(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let time = pw(0u);
    let scale = pf(1u);
    let row0 = wg.x * ROWS;
    let t = row0 + lid.x;
    let valid = t < time;
    let base = wg.y * time * D;
    var q: array<f32, {{D}}>;
    var acc: array<f32, {{D}}>;
    for (var d = 0u; d < D; d = d + 1u) {
        acc[d] = 0.0;
        q[d] = 0.0;
        if (valid) {
            q[d] = aq[base + t * D + d];
        }
    }
    var m = -3.0e38;
    var l = 0.0;
    let kend = min(time, row0 + ROWS);
    for (var kb = 0u; kb < kend; kb = kb + TILE) {
        load_kv(lid.x, base, kb, time);
        workgroupBarrier();
        if (valid && t >= kb) {
            let lim = min(TILE, t - kb + 1u);
            for (var jj = 0u; jj < lim; jj = jj + 1u) {
                var dot = 0.0;
                for (var d = 0u; d < D; d = d + 1u) {
                    dot = dot + q[d] * tile0[jj * D + d];
                }
                let s = dot * scale;
                report(s);
                if (s > m) {
                    let c = exp(m - s);
                    l = l * c;
                    for (var d = 0u; d < D; d = d + 1u) {
                        acc[d] = acc[d] * c;
                    }
                    m = s;
                }
                let p = exp(s - m);
                l = l + p;
                for (var d = 0u; d < D; d = d + 1u) {
                    acc[d] = acc[d] + p * tile1[jj * D + d];
                }
            }
        }
        workgroupBarrier();
    }
    if (valid) {
        for (var d = 0u; d < D; d = d + 1u) {
            let o = acc[d] / l;
            report(o);
            out0[base + t * D + d] = o;
        }
    }
}

// Recomputes the forward row to get the log-sum-exp and delta = dO . O.
// out0 = stats (2 * bht).
@compute @workgroup_size(64, 1, 1)
fn attn_bwd_prep(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let time = pw(0u);
    let scale = pf(1u);
    let bht = pw(2u);
    let row0 = wg.x * ROWS;
    let t = row0 + lid.x;
    let valid = t < time;
    let base = wg.y * time * D;
    var q: array<f32, {{D}}>;
    var acc: array<f32, {{D}}>;
    for (var d = 0u; d < D; d = d + 1u) {
        acc[d] = 0.0;
        q[d] = 0.0;
        if (valid) {
            q[d] = aq[base + t * D + d];
        }
    }
    var m = -3.0e38;
    var l = 0.0;
    let kend = min(time, row0 + ROWS);
    for (var kb = 0u; kb < kend; kb = kb + TILE) {
        load_kv(lid.x, base, kb, time);
        workgroupBarrier();
        if (valid && t >= kb) {
            let lim = min(TILE, t - kb + 1u);
            for (var jj = 0u; jj < lim; jj = jj + 1u) {
                var dot = 0.0;
                for (var d = 0u; d < D; d = d + 1u) {
                    dot = dot + q[d] * tile0[jj * D + d];
                }
                let s = dot * scale;
                report(s);
                if (s > m) {
                    let c = exp(m - s);
                    l = l * c;
                    for (var d = 0u; d < D; d = d + 1u) {
                        acc[d] = acc[d] * c;
                    }
                    m = s;
                }
                let p = exp(s - m);
                l = l + p;
                for (var d = 0u; d < D; d = d + 1u) {
                    acc[d] = acc[d] + p * tile1[jj * D + d];
                }
            }
        }
        workgroupBarrier();
    }
    if (valid) {
        var delta = 0.0;
        for (var d = 0u; d < D; d = d + 1u) {
            delta = delta + ado[base + t * D + d] * (acc[d] / l);
        }
        let row = wg.y * time + t;
        let lse = m + log(l);
        report(lse);
        report(delta);
        out0[row] = lse;
        out0[bht + row] = delta;
    }
}

// One lane per query row. out0 = grad_q.
@compute @workgroup_size(64, 1, 1)
fn attn_bwd_dq(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let time = pw(0u);
    let scale = pf(1u);
    let bht = pw(2u);
    let row0 = wg.x * ROWS;
    let t = row0 + lid.x;
    let valid = t < time;
    let base = wg.y * time * D;
    var q: array<f32, {{D}}>;
    var g: array<f32, {{D}}>;
    var dq: array<f32, {{D}}>;
    var lse = 0.0;
    var delta = 0.0;
    if (valid) {
        lse = astats[wg.y * time + t];
        delta = astats[bht + wg.y * time + t];
    }
    for (var d = 0u; d < D; d = d + 1u) {
        dq[d] = 0.0;
        q[d] = 0.0;
        g[d] = 0.0;
        if (valid) {
            q[d] = aq[base + t * D + d];
            g[d] = ado[base + t * D + d];
        }
    }
    let kend = min(time, row0 + ROWS);
    for (var kb = 0u; kb < kend; kb = kb + TILE) {
        load_kv(lid.x, base, kb, time);
        workgroupBarrier();
        if (valid && t >= kb) {
            let lim = min(TILE, t - kb + 1u);
            for (var jj = 0u; jj < lim; jj = jj + 1u) {
                var dot = 0.0;
                var dp = 0.0;
                for (var d = 0u; d < D; d = d + 1u) {
                    dot = dot + q[d] * tile0[jj * D + d];
                    dp = dp + g[d] * tile1[jj * D + d];
                }
                let p = exp(dot * scale - lse);
                let coef = scale * (p * (dp - delta));
                for (var d = 0u; d < D; d = d + 1u) {
                    dq[d] = dq[d] + coef * tile0[jj * D + d];
                }
            }
        }
        workgroupBarrier();
    }
    if (valid) {
        for (var d = 0u; d < D; d = d + 1u) {
            report(dq[d]);
            out0[base + t * D + d] = dq[d];
        }
    }
}

// One lane per key row j; queries i >= j stream through shared tiles.
// out0 = grad_k, out1 = grad_v.
@compute @workgroup_size(64, 1, 1)
fn attn_bwd_dkv(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let time = pw(0u);
    let scale = pf(1u);
    let bht = pw(2u);
    let col0 = wg.x * ROWS;
    let j = col0 + lid.x;
    let valid = j < time;
    let base = wg.y * time * D;
    let srow = wg.y * time;
    var kj: array<f32, {{D}}>;
    var vj: array<f32, {{D}}>;
    var dk: array<f32, {{D}}>;
    var dv: array<f32, {{D}}>;
    for (var d = 0u; d < D; d = d + 1u) {
        dk[d] = 0.0;
        dv[d] = 0.0;
        kj[d] = 0.0;
        vj[d] = 0.0;
        if (valid) {
            kj[d] = ak[base + j * D + d];
            vj[d] = av[base + j * D + d];
        }
    }
    for (var qb = col0; qb < time; qb = qb + TILE) {
        for (var e = lid.x; e < TILE * D; e = e + ROWS) {
            let i = qb + e / D;
            var qv = 0.0;
            var gv = 0.0;
            if (i < time) {
                qv = aq[base + i * D + e % D];
                gv = ado[base + i * D + e % D];
            }
            tile0[e] = qv;
            tile1[e] = gv;
        }
        for (var e = lid.x; e < TILE; e = e + ROWS) {
            let i = qb + e;
            var lse = 0.0;
            var delta = 0.0;
            if (i < time) {
                lse = astats[srow + i];
                delta = astats[bht + srow + i];
            }
            tile_s[e] = lse;
            tile_s[TILE + e] = delta;
        }
        workgroupBarrier();
        if (valid) {
            for (var ii = 0u; ii < TILE; ii = ii + 1u) {
                let i = qb + ii;
                if (i >= time) {
                    break;
                }
                if (i < j) {
                    continue;
                }
                var dot = 0.0;
                var dp = 0.0;
                for (var d = 0u; d < D; d = d + 1u) {
                    dot = dot + tile0[ii * D + d] * kj[d];
                    dp = dp + tile1[ii * D + d] * vj[d];
                }
                let p = exp(dot * scale - tile_s[ii]);
                let coef = scale * (p * (dp - tile_s[TILE + ii]));
                for (var d = 0u; d < D; d = d + 1u) {
                    dv[d] = dv[d] + p * tile1[ii * D + d];
                    dk[d] = dk[d] + coef * tile0[ii * D + d];
                }
            }
        }
        workgroupBarrier();
    }
    if (valid) {
        for (var d = 0u; d < D; d = d + 1u) {
            report(dk[d]);
            report(dv[d]);
            out0[base + j * D + d] = dk[d];
            out1[base + j * D + d] = dv[d];
        }
    }
}

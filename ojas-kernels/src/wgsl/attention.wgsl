// Causal attention, FlashAttention-2 style, in plain WGSL (no matrix units,
// no subgroups). Planes are [batch * heads, time, D] contiguous; D (word 3)
// is the real head dimension and DP4 * 4 >= D its padded width, so one
// pipeline serves every D that pads to the same width.
//
//   S   = scale * Q K^T              (query t sees keys 0..=t)
//   O   = softmax(S) V, online over key blocks                    attn_fwd
//   lse = m + log l, Dr = dO . O, per query row                attn_bwd_prep
//   P   = exp(S - lse), dP = dO V^T, dS = scale * P * (dP - Dr)
//   dQ  = dS K                                                  attn_bwd_dq
//   dK  = dS^T Q, dV = P^T dO                                   attn_bwd_dkv
//
// A workgroup owns a block of rows; PARTS threads share a row. In each block
// step a thread forms the scores of its row against its share of the keys
// (a register dot over shared Q and K tiles), the row's max and sum are
// combined through shared memory, and P (or dS) goes to a shared tile that
// every thread of the row reads back against its own share of the D columns.
// Nothing time x time is stored. Each output row is written once by the
// workgroup that owns it, in a fixed order, with no atomics: results repeat
// bit for bit.
//
// Causality is exact: masked entries are never used, a row's accumulation
// loops stop at its own position (so a future key or value never enters a
// result, not even as 0 * NaN), and only live scores can raise the fault
// bit. A live score that is not finite raises the bit and counts as masked.
// Shared tiles use a row stride of DP4 + 1 vec4 (and BK + 1 floats) so the
// rows that neighbouring threads read fall in different banks.
//
// Words: 0 time, 1 scale bits, 2 batch * heads * time, 3 D.
// Grids: (ceil(time / FQ), batch * heads) for attn_fwd and attn_bwd_prep;
// (ceil(time / BB), batch * heads) for attn_bwd_dq and attn_bwd_dkv.
// stats holds lse at [row] and Dr at [bht + row]; prep writes it.

const DP4: u32 = {{DP4}}u;
const SP: u32 = {{SP}}u;
const PARTS: u32 = 4u;
const CV: u32 = {{CV}}u;
const FQ: u32 = {{FQ}}u;
const FK: u32 = {{FK}}u;
const FKP: u32 = {{FKP}}u;
const FKPT: u32 = {{FKPT}}u;
const FWG: u32 = {{FWG}}u;
const BB: u32 = {{BB}}u;
const BBP: u32 = {{BBP}}u;
const BKPT: u32 = {{BKPT}}u;
const BWG: u32 = {{BWG}}u;
const NEG: f32 = -3.0e38;

@group(0) @binding(2) var<storage, read> aq: array<f32>;
@group(0) @binding(3) var<storage, read> ak: array<f32>;
@group(0) @binding(4) var<storage, read> av: array<f32>;
@group(0) @binding(5) var<storage, read> ado: array<f32>;
@group(0) @binding(6) var<storage, read> astats: array<f32>;
@group(0) @binding(7) var<storage, read_write> out0: array<f32>;
@group(0) @binding(8) var<storage, read_write> out1: array<f32>;

// Forward tiles: Q block, K then V block, P, and two per-row reductions.
var<workgroup> f_q: array<vec4<f32>, {{FQ_SP}}>;
var<workgroup> f_kv: array<vec4<f32>, {{FK_SP}}>;
var<workgroup> f_p: array<f32, {{FQ_FKP}}>;
var<workgroup> f_red: array<f32, {{FQ_PARTS}}>;
var<workgroup> f_red2: array<f32, {{FQ_PARTS}}>;

// Backward tiles: three row blocks, P^T and dS tiles, and row statistics.
var<workgroup> b_a: array<vec4<f32>, {{BB_SP}}>;
var<workgroup> b_b: array<vec4<f32>, {{BB_SP}}>;
var<workgroup> b_c: array<vec4<f32>, {{BB_SP}}>;
var<workgroup> b_p: array<f32, {{BB_BBP}}>;
var<workgroup> b_s: array<f32, {{BB_BBP}}>;
var<workgroup> b_st: array<f32, {{BB2}}>;

// Four consecutive elements of row `row`, column `c`, zero past `time` or D.
fn ld4_q(base: u32, row: u32, c: u32) -> vec4<f32> {
    var v = vec4<f32>(0.0);
    let d = pw(3u);
    if (row < pw(0u)) {
        let i = base + row * d + c;
        for (var k = 0u; k < 4u; k = k + 1u) {
            if (c + k < d) { v[k] = aq[i + k]; }
        }
    }
    return v;
}

fn ld4_k(base: u32, row: u32, c: u32) -> vec4<f32> {
    var v = vec4<f32>(0.0);
    let d = pw(3u);
    if (row < pw(0u)) {
        let i = base + row * d + c;
        for (var k = 0u; k < 4u; k = k + 1u) {
            if (c + k < d) { v[k] = ak[i + k]; }
        }
    }
    return v;
}

fn ld4_v(base: u32, row: u32, c: u32) -> vec4<f32> {
    var v = vec4<f32>(0.0);
    let d = pw(3u);
    if (row < pw(0u)) {
        let i = base + row * d + c;
        for (var k = 0u; k < 4u; k = k + 1u) {
            if (c + k < d) { v[k] = av[i + k]; }
        }
    }
    return v;
}

fn ld4_do(base: u32, row: u32, c: u32) -> vec4<f32> {
    var v = vec4<f32>(0.0);
    let d = pw(3u);
    if (row < pw(0u)) {
        let i = base + row * d + c;
        for (var k = 0u; k < 4u; k = k + 1u) {
            if (c + k < d) { v[k] = ado[i + k]; }
        }
    }
    return v;
}

fn hsum(v: vec4<f32>) -> f32 {
    return (v.x + v.y) + (v.z + v.w);
}

// Store the four columns starting at `c` of row `row` of out0 (or out1),
// reporting each stored value.
fn st4_0(base: u32, row: u32, c: u32, v: vec4<f32>) {
    let d = pw(3u);
    let i = base + row * d + c;
    for (var k = 0u; k < 4u; k = k + 1u) {
        if (c + k < d) {
            report(v[k]);
            out0[i + k] = v[k];
        }
    }
}

fn st4_1(base: u32, row: u32, c: u32, v: vec4<f32>) {
    let d = pw(3u);
    let i = base + row * d + c;
    for (var k = 0u; k < 4u; k = k + 1u) {
        if (c + k < d) {
            report(v[k]);
            out1[i + k] = v[k];
        }
    }
}

struct Fwd {
    m: f32,
    l: f32,
    acc: array<vec4<f32>, {{CV}}>,
}

// The online-softmax forward of one query block. Thread `lane` owns row
// lane / PARTS and, of each key block, keys part * FKPT .. + FKPT for the
// scores and vec4 columns part * CV .. + CV for the output.
fn fwd_core(qblock: u32, bh: u32, lane: u32) -> Fwd {
    let time = pw(0u);
    let scale = pf(1u);
    let r = lane / PARTS;
    let part = lane % PARTS;
    let q0 = qblock * FQ;
    let qi = q0 + r;
    let live_row = qi < time;
    let base = bh * time * pw(3u);
    for (var e = lane; e < FQ * DP4; e = e + FWG) {
        f_q[(e / DP4) * SP + e % DP4] = ld4_q(base, q0 + e / DP4, (e % DP4) * 4u);
    }
    var st: Fwd;
    st.m = NEG;
    st.l = 0.0;
    for (var cv = 0u; cv < CV; cv = cv + 1u) {
        st.acc[cv] = vec4<f32>(0.0);
    }
    let kend = min(time, q0 + FQ);
    for (var kb = 0u; kb < kend; kb = kb + FK) {
        workgroupBarrier();
        for (var e = lane; e < FK * DP4; e = e + FWG) {
            f_kv[(e / DP4) * SP + e % DP4] = ld4_k(base, kb + e / DP4, (e % DP4) * 4u);
        }
        workgroupBarrier();
        var s: array<f32, {{FKPT}}>;
        var bmax = NEG;
        for (var c = 0u; c < FKPT; c = c + 1u) {
            let jj = part * FKPT + c;
            var dot = vec4<f32>(0.0);
            for (var d4 = 0u; d4 < DP4; d4 = d4 + 1u) {
                dot = dot + f_q[r * SP + d4] * f_kv[jj * SP + d4];
            }
            var sv = hsum(dot) * scale;
            if (live_row && kb + jj <= qi) {
                if (nonfinite(sv)) {
                    raise();
                    sv = NEG;
                }
            } else {
                sv = NEG;
            }
            s[c] = sv;
            bmax = max(bmax, sv);
        }
        f_red[r * PARTS + part] = bmax;
        workgroupBarrier();
        var m_new = st.m;
        for (var p = 0u; p < PARTS; p = p + 1u) {
            m_new = max(m_new, f_red[r * PARTS + p]);
        }
        // A row that has seen nothing has a zero accumulator; its rescale is
        // exactly 0, never exp(NEG - NEG).
        var alpha = 0.0;
        if (st.m != NEG) {
            alpha = exp(st.m - m_new);
        }
        var psum = 0.0;
        for (var c = 0u; c < FKPT; c = c + 1u) {
            var p = 0.0;
            if (s[c] != NEG) {
                p = exp(s[c] - m_new);
            }
            f_p[r * FKP + part * FKPT + c] = p;
            psum = psum + p;
        }
        f_red2[r * PARTS + part] = psum;
        workgroupBarrier();
        for (var e = lane; e < FK * DP4; e = e + FWG) {
            f_kv[(e / DP4) * SP + e % DP4] = ld4_v(base, kb + e / DP4, (e % DP4) * 4u);
        }
        workgroupBarrier();
        var bsum = 0.0;
        for (var p = 0u; p < PARTS; p = p + 1u) {
            bsum = bsum + f_red2[r * PARTS + p];
        }
        st.l = st.l * alpha + bsum;
        st.m = m_new;
        for (var cv = 0u; cv < CV; cv = cv + 1u) {
            st.acc[cv] = st.acc[cv] * alpha;
        }
        if (live_row && qi >= kb) {
            let jlim = min(FK, qi - kb + 1u);
            for (var jj = 0u; jj < jlim; jj = jj + 1u) {
                let p = f_p[r * FKP + jj];
                for (var cv = 0u; cv < CV; cv = cv + 1u) {
                    st.acc[cv] = st.acc[cv] + p * f_kv[jj * SP + part * CV + cv];
                }
            }
        }
    }
    return st;
}

@compute @workgroup_size({{FWG}}, 1, 1)
fn attn_fwd(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let st = fwd_core(wg.x, wg.y, lid.x);
    let time = pw(0u);
    let r = lid.x / PARTS;
    let part = lid.x % PARTS;
    let qi = wg.x * FQ + r;
    if (qi < time) {
        var inv = 0.0;
        if (st.l > 0.0) {
            inv = 1.0 / st.l;
        }
        let base = wg.y * time * pw(3u);
        for (var cv = 0u; cv < CV; cv = cv + 1u) {
            st4_0(base, qi, (part * CV + cv) * 4u, st.acc[cv] * inv);
        }
    }
}

// The forward again, for lse = m + log l and Dr = dO . O. out0 = stats.
@compute @workgroup_size({{FWG}}, 1, 1)
fn attn_bwd_prep(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let st = fwd_core(wg.x, wg.y, lid.x);
    let time = pw(0u);
    let bht = pw(2u);
    let r = lid.x / PARTS;
    let part = lid.x % PARTS;
    let qi = wg.x * FQ + r;
    let live = qi < time;
    var partial = 0.0;
    if (live) {
        var inv = 0.0;
        if (st.l > 0.0) {
            inv = 1.0 / st.l;
        }
        let base = wg.y * time * pw(3u);
        for (var cv = 0u; cv < CV; cv = cv + 1u) {
            let g = ld4_do(base, qi, (part * CV + cv) * 4u);
            partial = partial + hsum(g * (st.acc[cv] * inv));
        }
    }
    // fwd_core's last reads of f_red were before its final two barriers.
    f_red[r * PARTS + part] = partial;
    workgroupBarrier();
    if (live && part == 0u) {
        var delta = 0.0;
        for (var p = 0u; p < PARTS; p = p + 1u) {
            delta = delta + f_red[r * PARTS + p];
        }
        let lse = st.m + log(st.l);
        report(lse);
        report(delta);
        let row = wg.y * time + qi;
        out0[row] = lse;
        out0[bht + row] = delta;
    }
}

// dQ for one block of BB query rows. b_a = Q, b_b = dO, b_c = V then K of
// the current key block, b_s = dS. out0 = grad_q.
@compute @workgroup_size({{BWG}}, 1, 1)
fn attn_bwd_dq(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let time = pw(0u);
    let scale = pf(1u);
    let bht = pw(2u);
    let lane = lid.x;
    let r = lane / PARTS;
    let part = lane % PARTS;
    let q0 = wg.x * BB;
    let qi = q0 + r;
    let live_row = qi < time;
    let base = wg.y * time * pw(3u);
    for (var e = lane; e < BB * DP4; e = e + BWG) {
        let at = (e / DP4) * SP + e % DP4;
        b_a[at] = ld4_q(base, q0 + e / DP4, (e % DP4) * 4u);
        b_b[at] = ld4_do(base, q0 + e / DP4, (e % DP4) * 4u);
    }
    var lse = 0.0;
    var delta = 0.0;
    if (live_row) {
        lse = astats[wg.y * time + qi];
        delta = astats[bht + wg.y * time + qi];
    }
    var acc: array<vec4<f32>, {{CV}}>;
    for (var cv = 0u; cv < CV; cv = cv + 1u) {
        acc[cv] = vec4<f32>(0.0);
    }
    let kend = min(time, q0 + BB);
    for (var kb = 0u; kb < kend; kb = kb + BB) {
        workgroupBarrier();
        for (var e = lane; e < BB * DP4; e = e + BWG) {
            b_c[(e / DP4) * SP + e % DP4] = ld4_v(base, kb + e / DP4, (e % DP4) * 4u);
        }
        workgroupBarrier();
        var dp: array<f32, {{BKPT}}>;
        for (var c = 0u; c < BKPT; c = c + 1u) {
            let jj = part * BKPT + c;
            var dot = vec4<f32>(0.0);
            for (var d4 = 0u; d4 < DP4; d4 = d4 + 1u) {
                dot = dot + b_b[r * SP + d4] * b_c[jj * SP + d4];
            }
            dp[c] = hsum(dot);
        }
        workgroupBarrier();
        for (var e = lane; e < BB * DP4; e = e + BWG) {
            b_c[(e / DP4) * SP + e % DP4] = ld4_k(base, kb + e / DP4, (e % DP4) * 4u);
        }
        workgroupBarrier();
        for (var c = 0u; c < BKPT; c = c + 1u) {
            let jj = part * BKPT + c;
            var ds = 0.0;
            if (live_row && kb + jj <= qi) {
                var dot = vec4<f32>(0.0);
                for (var d4 = 0u; d4 < DP4; d4 = d4 + 1u) {
                    dot = dot + b_a[r * SP + d4] * b_c[jj * SP + d4];
                }
                let s = hsum(dot) * scale;
                report(s);
                let p = exp(s - lse);
                ds = scale * (p * (dp[c] - delta));
            }
            b_s[r * BBP + jj] = ds;
        }
        workgroupBarrier();
        if (live_row && qi >= kb) {
            let jlim = min(BB, qi - kb + 1u);
            for (var jj = 0u; jj < jlim; jj = jj + 1u) {
                let ds = b_s[r * BBP + jj];
                for (var cv = 0u; cv < CV; cv = cv + 1u) {
                    acc[cv] = acc[cv] + ds * b_c[jj * SP + part * CV + cv];
                }
            }
        }
    }
    if (live_row) {
        for (var cv = 0u; cv < CV; cv = cv + 1u) {
            st4_0(base, qi, (part * CV + cv) * 4u, acc[cv]);
        }
    }
}

// dK and dV for one block of BB key rows, walking the query blocks from the
// one holding the first key. b_b = K, b_c = V (fixed), b_a = Q, then dO,
// then Q of the current query block; b_p = P^T, b_s = dS^T, b_st = the
// block's lse and Dr. out0 = grad_k, out1 = grad_v.
@compute @workgroup_size({{BWG}}, 1, 1)
fn attn_bwd_dkv(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let time = pw(0u);
    let scale = pf(1u);
    let bht = pw(2u);
    let lane = lid.x;
    let r = lane / PARTS;
    let part = lane % PARTS;
    let k0 = wg.x * BB;
    let kj = k0 + r;
    let live_key = kj < time;
    let base = wg.y * time * pw(3u);
    let srow = wg.y * time;
    for (var e = lane; e < BB * DP4; e = e + BWG) {
        let at = (e / DP4) * SP + e % DP4;
        b_b[at] = ld4_k(base, k0 + e / DP4, (e % DP4) * 4u);
        b_c[at] = ld4_v(base, k0 + e / DP4, (e % DP4) * 4u);
    }
    var dk: array<vec4<f32>, {{CV}}>;
    var dv: array<vec4<f32>, {{CV}}>;
    for (var cv = 0u; cv < CV; cv = cv + 1u) {
        dk[cv] = vec4<f32>(0.0);
        dv[cv] = vec4<f32>(0.0);
    }
    for (var qb = k0; qb < time; qb = qb + BB) {
        workgroupBarrier();
        for (var e = lane; e < BB * DP4; e = e + BWG) {
            b_a[(e / DP4) * SP + e % DP4] = ld4_q(base, qb + e / DP4, (e % DP4) * 4u);
        }
        if (lane < BB) {
            let i = qb + lane;
            var lse = 0.0;
            var dr = 0.0;
            if (i < time) {
                lse = astats[srow + i];
                dr = astats[bht + srow + i];
            }
            b_st[lane] = lse;
            b_st[BB + lane] = dr;
        }
        workgroupBarrier();
        var p: array<f32, {{BKPT}}>;
        for (var c = 0u; c < BKPT; c = c + 1u) {
            let ii = part * BKPT + c;
            let i = qb + ii;
            var pv = 0.0;
            if (live_key && i < time && i >= kj) {
                var dot = vec4<f32>(0.0);
                for (var d4 = 0u; d4 < DP4; d4 = d4 + 1u) {
                    dot = dot + b_a[ii * SP + d4] * b_b[r * SP + d4];
                }
                let s = hsum(dot) * scale;
                report(s);
                pv = exp(s - b_st[ii]);
            }
            p[c] = pv;
        }
        workgroupBarrier();
        for (var e = lane; e < BB * DP4; e = e + BWG) {
            b_a[(e / DP4) * SP + e % DP4] = ld4_do(base, qb + e / DP4, (e % DP4) * 4u);
        }
        workgroupBarrier();
        for (var c = 0u; c < BKPT; c = c + 1u) {
            let ii = part * BKPT + c;
            let i = qb + ii;
            var ds = 0.0;
            if (live_key && i < time && i >= kj) {
                var dot = vec4<f32>(0.0);
                for (var d4 = 0u; d4 < DP4; d4 = d4 + 1u) {
                    dot = dot + b_a[ii * SP + d4] * b_c[r * SP + d4];
                }
                ds = scale * (p[c] * (hsum(dot) - b_st[BB + ii]));
            }
            b_p[r * BBP + ii] = p[c];
            b_s[r * BBP + ii] = ds;
        }
        workgroupBarrier();
        var istart = 0u;
        if (kj > qb) {
            istart = kj - qb;
        }
        let iend = min(BB, time - qb);
        if (live_key) {
            for (var ii = istart; ii < iend; ii = ii + 1u) {
                let pv = b_p[r * BBP + ii];
                for (var cv = 0u; cv < CV; cv = cv + 1u) {
                    dv[cv] = dv[cv] + pv * b_a[ii * SP + part * CV + cv];
                }
            }
        }
        workgroupBarrier();
        for (var e = lane; e < BB * DP4; e = e + BWG) {
            b_a[(e / DP4) * SP + e % DP4] = ld4_q(base, qb + e / DP4, (e % DP4) * 4u);
        }
        workgroupBarrier();
        if (live_key) {
            for (var ii = istart; ii < iend; ii = ii + 1u) {
                let ds = b_s[r * BBP + ii];
                for (var cv = 0u; cv < CV; cv = cv + 1u) {
                    dk[cv] = dk[cv] + ds * b_a[ii * SP + part * CV + cv];
                }
            }
        }
    }
    if (live_key) {
        for (var cv = 0u; cv < CV; cv = cv + 1u) {
            st4_0(base, kj, (part * CV + cv) * 4u, dk[cv]);
            st4_1(base, kj, (part * CV + cv) * 4u, dv[cv]);
        }
    }
}

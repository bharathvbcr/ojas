// Two-stage deterministic sums. Stage one writes one partial per fixed chunk;
// stage two adds the partials in index order with one workgroup. The chunk
// size is a constant, so the association depends on the length, not the GPU.

@group(0) @binding(2) var<storage, read> x0: array<f32>;
@group(0) @binding(3) var<storage, read> x1: array<f32>;
@group(0) @binding(4) var<storage, read> x2: array<f32>;
@group(0) @binding(5) var<storage, read_write> y0: array<f32>;
@group(0) @binding(6) var<storage, read_write> yu: array<u32>;

const CHUNK: u32 = 4096u;

fn term(mode: u32, i: u32) -> f32 {
    if (mode == 1u) {
        // value residual: (value0 - value) * grad_output
        return (x1[i] - x0[i]) * x2[i];
    }
    return x0[i];
}

// Words: 0 n, 1 mode. y0[group] = sum of the group's chunk.
@compute @workgroup_size(256, 1, 1)
fn sum_partial(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let group = flat_group(wg, nwg);
    let n = pw(0u);
    let mode = pw(1u);
    let start = group * CHUNK;
    var acc = 0.0;
    for (var j = 0u; j < CHUNK / 256u; j = j + 1u) {
        let i = start + j * 256u + lid.x;
        if (i < n) {
            acc = acc + term(mode, i);
        }
    }
    let total = tree_sum(lid.x, acc);
    if (lid.x == 0u) {
        y0[group] = total;
    }
}

// Words: 0 partial count, 1 mode, 2 divisor bits.
// mode 0: y0[0] = total / divisor. mode 1: y0[0] = total * s * (1 - s) with
// s = sigmoid(x1[0]), the value-residual lambda gradient.
@compute @workgroup_size(256, 1, 1)
fn sum_finish(@builtin(local_invocation_id) lid: vec3<u32>) {
    let count = pw(0u);
    var acc = 0.0;
    for (var i = lid.x; i < count; i = i + 256u) {
        acc = acc + x0[i];
    }
    let total = tree_sum(lid.x, acc);
    if (lid.x == 0u) {
        var out = total / pf(2u);
        if (pw(1u) == 1u) {
            let s = sigmoid(x1[0]);
            out = total * s * (1.0 - s);
        }
        report(out);
        y0[0] = out;
    }
}

// Column sums of a [rows, cols] matrix, stage one. Grid (cols / 256, chunks).
// Words: 0 rows, 1 cols, 2 rows per chunk, 3 mode.
// mode 0: x0[r, c]. mode 1: x0[r, c] * (x1[r, c] * x2[r]) (RMSNorm grad_weight).
@compute @workgroup_size(256, 1, 1)
fn col_partial(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let rows = pw(0u);
    let cols = pw(1u);
    let per = pw(2u);
    let c = wg.x * 256u + lid.x;
    if (c >= cols) { return; }
    let r0 = wg.y * per;
    let r1 = min(rows, r0 + per);
    var acc = 0.0;
    for (var r = r0; r < r1; r = r + 1u) {
        let i = r * cols + c;
        if (pw(3u) == 1u) {
            acc = acc + x0[i] * (x1[i] * x2[r]);
        } else {
            acc = acc + x0[i];
        }
    }
    y0[wg.y * cols + c] = acc;
}

// Stage two. Words: 0 chunks, 1 cols. One lane per column.
@compute @workgroup_size(256, 1, 1)
fn col_finish(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let c = flat_group(wg, nwg) * 256u + lid.x;
    let chunks = pw(0u);
    let cols = pw(1u);
    if (c >= cols) { return; }
    var acc = 0.0;
    for (var ch = 0u; ch < chunks; ch = ch + 1u) {
        acc = acc + x0[ch * cols + c];
    }
    report(acc);
    y0[c] = acc;
}

// argmax_rows: one workgroup per row. Each lane keeps its best over a strided
// walk in rising column order (a strict > keeps the lowest tie), then a tree
// over the lanes keeps the larger value, or the lower column on a tie. A
// non-finite value raises the op's fault and takes no part.
var<workgroup> arg_v: array<f32, 256>;
var<workgroup> arg_i: array<u32, 256>;

const ARG_NONE: u32 = 0xffffffffu;

// Words: 0 rows, 1 cols. yu[row] = the column of the row's largest value.
@compute @workgroup_size(256, 1, 1)
fn argmax_rows(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = flat_group(wg, nwg);
    let rows = pw(0u);
    let cols = pw(1u);
    // `row` is the same for every lane of the workgroup, so this return
    // leaves the barriers below in uniform control flow.
    if (row >= rows) {
        return;
    }
    var best = 0.0;
    var idx = ARG_NONE;
    var bad = false;
    for (var c = lid.x; c < cols; c = c + 256u) {
        let v = x0[row * cols + c];
        if (nonfinite(v)) {
            bad = true;
        } else if (idx == ARG_NONE || v > best) {
            best = v;
            idx = c;
        }
    }
    if (bad) {
        raise();
    }
    arg_v[lid.x] = best;
    arg_i[lid.x] = idx;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid.x < s) {
            let ov = arg_v[lid.x + s];
            let oi = arg_i[lid.x + s];
            let mi = arg_i[lid.x];
            if (oi != ARG_NONE && (mi == ARG_NONE || ov > arg_v[lid.x] || (ov == arg_v[lid.x] && oi < mi))) {
                arg_v[lid.x] = ov;
                arg_i[lid.x] = oi;
            }
        }
        workgroupBarrier();
    }
    if (lid.x == 0u) {
        // A row with no finite value has raised already; 0 keeps the id a
        // valid column.
        var r = arg_i[0];
        if (r == ARG_NONE) {
            r = 0u;
        }
        yu[row] = r;
    }
}

// topk_rows: one workgroup per row and k rounds, ranked as
// `ojas_infer::sample_token` ranks candidates: value descending under
// `total_cmp`, then column ascending. Round j takes, over the columns ranked
// strictly after round j - 1's pick (a smaller key, or the same key at a
// higher column), the largest key, then the lowest column: a strided walk per
// lane (rising columns, so a strict > keeps the lowest column of equal keys),
// then a tree over the lanes. The first round's walk raises the op's fault for
// a NaN or +inf; -inf is a value. Distinct (key, column) pairs leave cols - j
// candidates in round j, so every round picks.
var<workgroup> top_k: array<u32, 256>;
var<workgroup> top_i: array<u32, 256>;

// An f32's rank key: unsigned order of keys is `f32::total_cmp` order, so
// +0.0 outranks -0.0 and -inf ranks below every finite value.
fn rank_key(v: f32) -> u32 {
    let b = bitcast<u32>(v);
    return b ^ select(0x80000000u, 0xffffffffu, (b >> 31u) != 0u);
}

// Words: 0 rows, 1 cols, 2 k. y0[row * k + j] and yu[row * k + j] are the
// row's j-th leader's value and column.
@compute @workgroup_size(256, 1, 1)
fn topk_rows(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = flat_group(wg, nwg);
    let rows = pw(0u);
    let cols = pw(1u);
    let k = pw(2u);
    // `row` and `k` are the same for every lane of the workgroup, so this
    // return and the round loop leave the barriers in uniform control flow.
    if (row >= rows) {
        return;
    }
    let base = row * cols;
    var pk = 0u;
    var pi = ARG_NONE;
    for (var j = 0u; j < k; j = j + 1u) {
        var best_k = 0u;
        var best_i = ARG_NONE;
        var bad = false;
        for (var c = lid.x; c < cols; c = c + 256u) {
            let v = x0[base + c];
            let key = rank_key(v);
            if (j == 0u && nonfinite(v) && bitcast<u32>(v) != 0xff800000u) {
                bad = true;
            }
            let after = j == 0u || key < pk || (key == pk && c > pi);
            if (after && (best_i == ARG_NONE || key > best_k)) {
                best_k = key;
                best_i = c;
            }
        }
        if (bad) {
            raise();
        }
        top_k[lid.x] = best_k;
        top_i[lid.x] = best_i;
        workgroupBarrier();
        for (var s = 128u; s > 0u; s = s >> 1u) {
            if (lid.x < s) {
                let ck = top_k[lid.x + s];
                let oi = top_i[lid.x + s];
                let mi = top_i[lid.x];
                if (oi != ARG_NONE && (mi == ARG_NONE || ck > top_k[lid.x] || (ck == top_k[lid.x] && oi < mi))) {
                    top_k[lid.x] = ck;
                    top_i[lid.x] = oi;
                }
            }
            workgroupBarrier();
        }
        pk = top_k[0];
        pi = top_i[0];
        if (lid.x == 0u) {
            // Unreachable for 1 <= k <= cols; 0 keeps the id a valid column.
            var col = pi;
            if (col == ARG_NONE) {
                col = 0u;
            }
            y0[row * k + j] = x0[base + col];
            yu[row * k + j] = col;
        }
        // Every lane has read the pick before the next round overwrites it.
        workgroupBarrier();
    }
}

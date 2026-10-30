// Embedding and mean cross-entropy. Token ids and targets arrive already
// range-checked by the host, which keeps a copy of every uploaded U32 tensor.

@group(0) @binding(2) var<storage, read> x0: array<f32>;
@group(0) @binding(3) var<storage, read> u0: array<u32>;
@group(0) @binding(4) var<storage, read> u1: array<u32>;
@group(0) @binding(5) var<storage, read> u2: array<u32>;
@group(0) @binding(6) var<storage, read_write> y0: array<f32>;

// One lane per output element. Words: 0 tokens*dim, 1 dim.
// x0 = table, u0 = ids.
@compute @workgroup_size(256, 1, 1)
fn embed_fwd(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = flat_group(wg, nwg) * 256u + lid.x;
    if (i >= pw(0u)) { return; }
    let dim = pw(1u);
    let v = x0[u0[i / dim] * dim + i % dim];
    report(v);
    y0[i] = v;
}

// One lane per (distinct id, column). The host sorts token positions by id,
// keeping position order inside an id, so each row adds its tokens in the
// same order as the CPU loop. Words: 0 distinct*dim, 1 dim.
// x0 = grad_output, u0 = positions sorted by id, u1 = segment starts
// (distinct + 1), u2 = distinct ids. y0 is zeroed first.
@compute @workgroup_size(256, 1, 1)
fn embed_bwd(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = flat_group(wg, nwg) * 256u + lid.x;
    if (i >= pw(0u)) { return; }
    let dim = pw(1u);
    let u = i / dim;
    let c = i % dim;
    var acc = 0.0;
    for (var j = u1[u]; j < u1[u + 1u]; j = j + 1u) {
        acc = acc + x0[u0[j] * dim + c];
    }
    report(acc);
    y0[u2[u] * dim + c] = acc;
}

// Words: 0 rows, 1 vocab, 2 has_ignore, 3 ignore id, 4 valid count bits (f32).
fn row_target(row: u32) -> u32 { return u0[row]; }
fn ignored(tgt: u32) -> bool { return pw(2u) == 1u && tgt == pw(3u); }

fn row_peak(lane: u32, base: u32, vocab: u32) -> f32 {
    var peak = -3.40282347e+38;
    for (var c = lane; c < vocab; c = c + 256u) {
        peak = max(peak, x0[base + c]);
    }
    return tree_max(lane, peak);
}

fn row_exp_sum(lane: u32, base: u32, vocab: u32, peak: f32) -> f32 {
    var acc = 0.0;
    for (var c = lane; c < vocab; c = c + 256u) {
        acc = acc + exp(x0[base + c] - peak);
    }
    let sum = tree_sum(lane, acc);
    if (nonfinite(sum) || !(sum > 0.0)) {
        if (lane == 0u) { raise(); }
    }
    return sum;
}

// Softmax Collapse. A row whose target logit is its maximum and whose f32 sum
// is exactly 1.0 has a standard target gradient `p - 1` of exactly 0 whatever
// the other classes hold. Such a row that has another class takes the
// target-excluded form, with S the sum over the other classes of
// exp(x - peak): loss log1p(S), target gradient -(S / (1 + S)) / valid, each
// other class (exp(x - peak) / (1 + S)) / valid. Every other row is the
// standard form, bit for bit.

// ln(1 + x) for the small x >= 0 of a collapsed row, as 2 atanh(x / (2 + x)):
// the odd series to z^15, exact to f32 rounding for z <= 1/3. WGSL has no
// log1p, and its `log` carries an absolute error of 2^-21 near 1, which would
// swamp a loss of 1e-8.
fn log1p_small(x: f32) -> f32 {
    let z = x / (2.0 + x);
    let z2 = z * z;
    var p = 1.0 / 15.0;
    p = p * z2 + 1.0 / 13.0;
    p = p * z2 + 1.0 / 11.0;
    p = p * z2 + 1.0 / 9.0;
    p = p * z2 + 1.0 / 7.0;
    p = p * z2 + 1.0 / 5.0;
    p = p * z2 + 1.0 / 3.0;
    p = p * z2 + 1.0;
    return 2.0 * z * p;
}

// The row's target logit, or the peak for an ignored row (whose target may
// lie outside the vocabulary and is never read).
fn row_picked(base: u32, tgt: u32, peak: f32) -> f32 {
    if (ignored(tgt)) { return peak; }
    return x0[base + tgt];
}

// Whether the row needs its target-excluded sum, as one value every lane holds.
// Lane 0 writes the whole condition to a workgroup variable and every lane
// reads it with `workgroupUniformLoad`, which the compiler knows is uniform:
// the tree reduction behind `if (row_gate(..))` is then legal, and a row that
// does not need it pays this one load (two barriers) instead of a tree. The
// condition must not be combined with a value derived from a reduction
// (`peak`, `sum`) at the branch: that would not be uniform to the compiler.
var<workgroup> gate_flag: u32;
fn row_gate(lane: u32, need: bool) -> bool {
    if (lane == 0u) { gate_flag = select(0u, 1u, need); }
    return workgroupUniformLoad(&gate_flag) != 0u;
}

// The sum over the other classes of exp(x - peak) of a row, in the tree's fixed
// order. Call it only behind `row_gate`: every lane must reach its barriers.
fn row_other_sum(lane: u32, base: u32, vocab: u32, peak: f32, tgt: u32) -> f32 {
    var acc = 0.0;
    for (var c = lane; c < vocab; c = c + 256u) {
        if (c != tgt) {
            acc = acc + exp(x0[base + c] - peak);
        }
    }
    return tree_sum(lane, acc);
}

// One workgroup per row. x0 = logits, u0 = targets. y0[row] = row loss, or
// 0 for an ignored row.
@compute @workgroup_size(256, 1, 1)
fn ce_fwd(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = flat_group(wg, nwg);
    if (row >= pw(0u)) { return; }
    let vocab = pw(1u);
    let base = row * vocab;
    let peak = row_peak(lid.x, base, vocab);
    let sum = row_exp_sum(lid.x, base, vocab, peak);
    let tgt = row_target(row);
    let collapsed = row_gate(lid.x, !ignored(tgt) && row_picked(base, tgt, peak) == peak && sum == 1.0);
    var other = 0.0;
    if (collapsed) {
        other = row_other_sum(lid.x, base, vocab, peak, tgt);
    }
    if (lid.x == 0u) {
        var loss = 0.0;
        if (!ignored(tgt)) {
            if (collapsed && other > 0.0) {
                loss = log1p_small(other);
            } else {
                loss = peak + log(sum) - x0[base + tgt];
            }
        }
        report(loss);
        y0[row] = loss;
    }
}

// One workgroup per row. grad = softmax / valid, minus 1 / valid at the
// target; an ignored row is 0. A collapsed row with another class is the
// target-excluded form above.
@compute @workgroup_size(256, 1, 1)
fn ce_bwd(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = flat_group(wg, nwg);
    if (row >= pw(0u)) { return; }
    let vocab = pw(1u);
    let denom = pf(4u);
    let base = row * vocab;
    let peak = row_peak(lid.x, base, vocab);
    let sum = row_exp_sum(lid.x, base, vocab, peak);
    let tgt = row_target(row);
    let skip = ignored(tgt);
    let collapsed = row_gate(lid.x, !skip && row_picked(base, tgt, peak) == peak && sum == 1.0);
    var other = 0.0;
    if (collapsed) {
        other = row_other_sum(lid.x, base, vocab, peak, tgt);
    }
    let switched = collapsed && other > 0.0;
    let one_plus = 1.0 + other;
    for (var c = lid.x; c < vocab; c = c + 256u) {
        var g = 0.0;
        if (!skip) {
            if (switched) {
                if (c == tgt) {
                    g = -(other / one_plus) / denom;
                } else {
                    g = exp(x0[base + c] - peak) / one_plus / denom;
                }
            } else {
                g = exp(x0[base + c] - peak) / sum / denom;
                if (c == tgt) {
                    g = g - 1.0 / denom;
                }
            }
        }
        report(g);
        y0[base + c] = g;
    }
}

// linear_cross_entropy_mean: the logits exist only as an R' x C' tile of
// rows r0.. and vocabulary columns c0... `stats` is four arrays of N rows:
// the running max m, the running sum s of exp(logit - m), the target's logit
// t, and the running sum so of exp(logit - m) over every class but the target
// (the backend allocates and charges `4 * N`). Words 2 and 3 are the ignore
// flag and id, as above. A row whose target logit is its maximum with a sum of
// exactly 1.0 and another class takes ce_bwd's target-excluded form, with
// S = so (Softmax Collapse, ft-f40eeed8); every other row is the standard form.

// One workgroup per tile row. Words: 0 R', 1 C', 4 c0, 5 r0, 6 first tile,
// 7 N. x0 = tile, u0 = targets (all N), y0 = stats. The tile's own max and
// sum are merged into the row's: m' = max(m, tm), s' = s exp(m - m') +
// ts exp(tm - m'), and so' likewise from the tile's sum over every class but
// the target.
@compute @workgroup_size(256, 1, 1)
fn lce_stats(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = flat_group(wg, nwg);
    if (row >= pw(0u)) { return; }
    let cols = pw(1u);
    let base = row * cols;
    let peak = row_peak(lid.x, base, cols);
    let sum = row_exp_sum(lid.x, base, cols, peak);
    let n = pw(7u);
    let c0 = pw(4u);
    let g = pw(5u) + row;
    let tgt = u0[g];
    let in_tile = tgt >= c0 && tgt < c0 + cols;
    // A tile without the row's target has no target to leave out: its
    // contribution to the excluded sum is exactly its `sum`. The target's tile
    // runs the second reduction only when the target is the tile's peak and the
    // tile's sum is not above 1.0 (`!(sum > 1.0)`): a row can collapse only
    // then, since its final sum is at least this tile's. Otherwise the row
    // cannot collapse, lce_finish and lce_grad never read the slot, and `sum`
    // goes in it.
    let again = row_gate(lid.x, in_tile && x0[base + tgt - c0] == peak && !(sum > 1.0));
    var other = sum;
    if (again) {
        other = row_other_sum(lid.x, base, cols, peak, tgt - c0);
    }
    if (lid.x == 0u) {
        if (pw(6u) == 1u) {
            y0[g] = peak;
            y0[n + g] = sum;
            y0[3u * n + g] = other;
        } else {
            let m = y0[g];
            let m2 = max(m, peak);
            y0[n + g] = y0[n + g] * exp(m - m2) + sum * exp(peak - m2);
            y0[3u * n + g] = y0[3u * n + g] * exp(m - m2) + other * exp(peak - m2);
            y0[g] = m2;
        }
        if (in_tile) {
            y0[2u * n + g] = x0[base + tgt - c0];
        }
    }
}

// One lane per row. Words: 0 N. x0 = stats, u0 = targets, y0 = row loss
// (m + log s - t, log1p of so for a collapsed row, or 0 for an ignored row),
// summed by the reduce module.
@compute @workgroup_size(256, 1, 1)
fn lce_finish(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let g = flat_group(wg, nwg) * 256u + lid.x;
    let n = pw(0u);
    if (g >= n) { return; }
    var loss = 0.0;
    if (!ignored(u0[g])) {
        // The standard expression first and unconditionally, then replaced for a
        // collapsed row: a compiler with fast math may reassociate
        // `m + log(s) - t`, and keeping it out of the branch keeps the bits of
        // every other row what they were.
        loss = x0[g] + log(x0[n + g]) - x0[2u * n + g];
        if (x0[2u * n + g] == x0[g] && x0[n + g] == 1.0 && x0[3u * n + g] > 0.0) {
            loss = log1p_small(x0[3u * n + g]);
        }
    }
    report(loss);
    y0[g] = loss;
}

// One lane per tile element, in place: the tile's logits become the seed-1
// gradient, ce_bwd's expression with the row's final m, s and so. Words: 0
// R' * C', 1 C', 4 c0, 5 r0, 6 valid count bits (f32), 7 N. x0 = stats,
// u0 = targets, y0 = tile.
@compute @workgroup_size(256, 1, 1)
fn lce_grad(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = flat_group(wg, nwg) * 256u + lid.x;
    if (i >= pw(0u)) { return; }
    let cols = pw(1u);
    let n = pw(7u);
    let denom = pf(6u);
    let g = pw(5u) + i / cols;
    let tgt = u0[g];
    var grad = 0.0;
    if (!ignored(tgt)) {
        let is_target = pw(4u) + i % cols == tgt;
        let other = x0[3u * n + g];
        if (x0[2u * n + g] == x0[g] && x0[n + g] == 1.0 && other > 0.0) {
            let one_plus = 1.0 + other;
            if (is_target) {
                grad = -(other / one_plus) / denom;
            } else {
                grad = exp(y0[i] - x0[g]) / one_plus / denom;
            }
        } else {
            grad = exp(y0[i] - x0[g]) / x0[n + g] / denom;
            if (is_target) {
                grad = grad - 1.0 / denom;
            }
        }
    }
    report(grad);
    y0[i] = grad;
}

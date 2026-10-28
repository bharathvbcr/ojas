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

// Empty-max sentinel. A lane that skips every column keeps this, and
// tree_max of an empty other-set stays here. It is finite, so a sum of 0
// is what distinguishes "no other class" from a non-finite logit.
const CE_EMPTY: f32 = -3.40282347e+38;

fn ce_nan() -> f32 { return bitcast<f32>(0x7fc00000u); }

// softplus(-(max_other - target)). sum == 0 is 0, including when max_other
// is CE_EMPTY. When 1 + q rounds to 1 the loss is q, not log(1 + q).
fn ce_loss(max_o: f32, sum_o: f32, picked: f32) -> f32 {
    if (nonfinite(sum_o) || nonfinite(picked) || sum_o < 0.0) { return ce_nan(); }
    if (!(sum_o > 0.0)) { return 0.0; }
    if (nonfinite(max_o)) { return ce_nan(); }
    let delta = max_o - picked;
    if (delta <= 0.0) {
        let q = sum_o * exp(delta);
        let u = 1.0 + q;
        if (u == 1.0) { return q; }
        return log(u) * q / (u - 1.0);
    }
    let e = exp(-delta);
    return delta + log(sum_o + e);
}

// Unnormalized scale. The target gradient is -sum * scale; each other class
// is exp(z - max) * scale. sum == 0 returns 0 and the caller does not exp.
fn ce_scale(max_o: f32, sum_o: f32, picked: f32) -> f32 {
    if (nonfinite(sum_o) || nonfinite(picked) || sum_o < 0.0) { return ce_nan(); }
    if (!(sum_o > 0.0)) { return 0.0; }
    if (nonfinite(max_o)) { return ce_nan(); }
    let delta = max_o - picked;
    if (delta <= 0.0) {
        let e = exp(delta);
        let q = sum_o * e;
        return e / (1.0 + q);
    }
    let e = exp(-delta);
    return 1.0 / (e + sum_o);
}

// skip_col < 0 excludes nothing. Every column is reported, including one
// that is left out of the max, so a NaN target cannot disappear.
fn row_peak(lane: u32, base: u32, n: u32, skip_col: i32) -> f32 {
    var peak = CE_EMPTY;
    for (var c = lane; c < n; c = c + 256u) {
        let x = x0[base + c];
        report(x);
        if (i32(c) != skip_col) {
            peak = max(peak, x);
        }
    }
    return tree_max(lane, peak);
}

// A skipped column is not exponentiated. An empty other-set (every column
// skipped, or a target-only tile) sums to 0 and does not raise: that row's
// loss is 0. A non-finite sum still raises.
fn row_exp_sum(lane: u32, base: u32, n: u32, peak: f32, skip_col: i32) -> f32 {
    var acc = 0.0;
    for (var c = lane; c < n; c = c + 256u) {
        if (i32(c) != skip_col) {
            acc = acc + exp(x0[base + c] - peak);
        }
    }
    let sum = tree_sum(lane, acc);
    if (nonfinite(sum)) {
        if (lane == 0u) { raise(); }
    }
    return sum;
}

// One workgroup per row. x0 = logits, u0 = targets. y0[row] = row loss, or
// 0 for an ignored row.
@compute @workgroup_size(256, 1, 1)
fn ce_fwd(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = flat_group(wg, nwg);
    if (row >= pw(0u)) { return; }
    let vocab = pw(1u);
    let base = row * vocab;
    let tgt = row_target(row);
    let skip = ignored(tgt);
    var skip_col = -1;
    if (!skip) { skip_col = i32(tgt); }
    let peak = row_peak(lid.x, base, vocab, skip_col);
    let sum = row_exp_sum(lid.x, base, vocab, peak, skip_col);
    if (lid.x == 0u) {
        var loss = 0.0;
        if (!skip) {
            loss = ce_loss(peak, sum, x0[base + tgt]);
        }
        report(loss);
        y0[row] = loss;
    }
}

// One workgroup per row. Other classes are exp(z - max) * scale / valid and
// the target is the negation of their sum; an ignored row, and a one-class
// row, is 0.
@compute @workgroup_size(256, 1, 1)
fn ce_bwd(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = flat_group(wg, nwg);
    if (row >= pw(0u)) { return; }
    let vocab = pw(1u);
    let denom = pf(4u);
    let base = row * vocab;
    let tgt = row_target(row);
    let skip = ignored(tgt);
    var skip_col = -1;
    if (!skip) { skip_col = i32(tgt); }
    let peak = row_peak(lid.x, base, vocab, skip_col);
    let sum = row_exp_sum(lid.x, base, vocab, peak, skip_col);
    var picked = 0.0;
    if (!skip) { picked = x0[base + tgt]; }
    let scale = ce_scale(peak, sum, picked) / denom;
    for (var c = lid.x; c < vocab; c = c + 256u) {
        var g = 0.0;
        if (!skip && sum > 0.0) {
            if (c == tgt) {
                g = -sum * scale;
            } else {
                g = exp(x0[base + c] - peak) * scale;
            }
        }
        report(g);
        y0[base + c] = g;
    }
}

// linear_cross_entropy_mean: the logits exist only as an R' x C' tile of
// rows r0.. and vocabulary columns c0... `stats` is three arrays of N rows:
// the running max m, the running sum s of exp(logit - m), and the target's
// logit t. Words 2 and 3 are the ignore flag and id, as above.

// One workgroup per tile row. Words: 0 R', 1 C', 4 c0, 5 r0, 6 first tile,
// 7 N. x0 = tile, u0 = targets (all N), y0 = stats. The tile's own max and
// sum are merged into the row's: m' = max(m, tm), s' = s exp(m - m') +
// ts exp(tm - m').
@compute @workgroup_size(256, 1, 1)
fn lce_stats(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = flat_group(wg, nwg);
    if (row >= pw(0u)) { return; }
    let cols = pw(1u);
    let base = row * cols;
    let n = pw(7u);
    let c0 = pw(4u);
    let g = pw(5u) + row;
    let tgt = u0[g];
    // An ignored id may lie outside the vocabulary. Skip the target column
    // only when this tile holds it and the row is counted.
    var skip_col = -1;
    if (!ignored(tgt) && tgt >= c0 && tgt < c0 + cols) {
        skip_col = i32(tgt - c0);
    }
    let peak = row_peak(lid.x, base, cols, skip_col);
    let sum = row_exp_sum(lid.x, base, cols, peak, skip_col);
    if (lid.x == 0u) {
        if (pw(6u) == 1u) {
            y0[g] = peak;
            y0[n + g] = sum;
        } else {
            let m = y0[g];
            let m2 = max(m, peak);
            // Both exponents are <= 0. An empty tile (peak == CE_EMPTY,
            // sum == 0) contributes 0 and leaves the other side unchanged.
            y0[n + g] = y0[n + g] * exp(m - m2) + sum * exp(peak - m2);
            y0[g] = m2;
        }
        if (tgt >= c0 && tgt < c0 + cols) {
            y0[2u * n + g] = x0[base + tgt - c0];
        }
    }
}

// One lane per row. Words: 0 N. x0 = stats, u0 = targets, y0 = row loss
// (softplus of the negated margin, or 0 for an ignored row or a one-class
// row), summed by the reduce module.
@compute @workgroup_size(256, 1, 1)
fn lce_finish(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let g = flat_group(wg, nwg) * 256u + lid.x;
    let n = pw(0u);
    if (g >= n) { return; }
    var loss = 0.0;
    if (!ignored(u0[g])) {
        loss = ce_loss(x0[g], x0[n + g], x0[2u * n + g]);
    }
    report(loss);
    y0[g] = loss;
}

// One lane per tile element, in place: the tile's logits become the seed-1
// gradient, ce_bwd's excluded-sum expression with the row's final m and s.
// Words: 0
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
        let mx = x0[g];
        let sum = x0[n + g];
        let scale = ce_scale(mx, sum, x0[2u * n + g]) / denom;
        if (sum > 0.0) {
            if (pw(4u) + i % cols == tgt) {
                grad = -sum * scale;
            } else {
                grad = exp(y0[i] - mx) * scale;
            }
        }
    }
    report(grad);
    y0[i] = grad;
}

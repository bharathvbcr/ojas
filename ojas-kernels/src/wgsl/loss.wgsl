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
    if (lid.x == 0u) {
        let tgt = row_target(row);
        var loss = 0.0;
        if (!ignored(tgt)) {
            loss = peak + log(sum) - x0[base + tgt];
        }
        report(loss);
        y0[row] = loss;
    }
}

// One workgroup per row. grad = softmax / valid, minus 1 / valid at the
// target; an ignored row is 0.
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
    for (var c = lid.x; c < vocab; c = c + 256u) {
        var g = 0.0;
        if (!skip) {
            g = exp(x0[base + c] - peak) / sum / denom;
            if (c == tgt) {
                g = g - 1.0 / denom;
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
    let peak = row_peak(lid.x, base, cols);
    let sum = row_exp_sum(lid.x, base, cols, peak);
    if (lid.x == 0u) {
        let n = pw(7u);
        let c0 = pw(4u);
        let g = pw(5u) + row;
        if (pw(6u) == 1u) {
            y0[g] = peak;
            y0[n + g] = sum;
        } else {
            let m = y0[g];
            let m2 = max(m, peak);
            y0[n + g] = y0[n + g] * exp(m - m2) + sum * exp(peak - m2);
            y0[g] = m2;
        }
        let tgt = u0[g];
        if (tgt >= c0 && tgt < c0 + cols) {
            y0[2u * n + g] = x0[base + tgt - c0];
        }
    }
}

// One lane per row. Words: 0 N. x0 = stats, u0 = targets, y0 = row loss
// (m + log s - t, or 0 for an ignored row), summed by the reduce module.
@compute @workgroup_size(256, 1, 1)
fn lce_finish(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let g = flat_group(wg, nwg) * 256u + lid.x;
    let n = pw(0u);
    if (g >= n) { return; }
    var loss = 0.0;
    if (!ignored(u0[g])) {
        loss = x0[g] + log(x0[n + g]) - x0[2u * n + g];
    }
    report(loss);
    y0[g] = loss;
}

// One lane per tile element, in place: the tile's logits become the seed-1
// gradient, ce_bwd's expression with the row's final m and s. Words: 0
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
        grad = exp(y0[i] - x0[g]) / x0[n + g] / denom;
        if (pw(4u) + i % cols == tgt) {
            grad = grad - 1.0 / denom;
        }
    }
    report(grad);
    y0[i] = grad;
}

// RMSNorm, one 256-lane workgroup per row. Words: 0 rows, 1 dim, 2 eps bits.
// rstd = 1 / sqrt(mean(x^2) + eps), as the CPU reference forms it (not
// inverseSqrt). A non-positive or non-finite denominator raises the fault bit.

@group(0) @binding(2) var<storage, read> x0: array<f32>;
@group(0) @binding(3) var<storage, read> x1: array<f32>;
@group(0) @binding(4) var<storage, read> x2: array<f32>;
@group(0) @binding(5) var<storage, read_write> y0: array<f32>;
@group(0) @binding(6) var<storage, read_write> y1: array<f32>;

fn row_rstd(lane: u32, base: u32, dim: u32) -> f32 {
    var acc = 0.0;
    for (var c = lane; c < dim; c = c + 256u) {
        let v = x0[base + c];
        acc = acc + v * v;
    }
    let sum_sq = tree_sum(lane, acc);
    let denom = sum_sq / f32(dim) + pf(2u);
    if (nonfinite(denom) || !(denom > 0.0)) {
        if (lane == 0u) { raise(); }
    }
    return 1.0 / sqrt(denom);
}

// x0 = x, x1 = weight. y0 = y.
@compute @workgroup_size(256, 1, 1)
fn rms_fwd(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = flat_group(wg, nwg);
    if (row >= pw(0u)) { return; }
    let dim = pw(1u);
    let base = row * dim;
    let rstd = row_rstd(lid.x, base, dim);
    for (var c = lid.x; c < dim; c = c + 256u) {
        let v = x0[base + c] * rstd * x1[c];
        report(v);
        y0[base + c] = v;
    }
}

// x0 = x, x1 = weight, x2 = grad_output. y0 = grad_x, y1 = rstd per row.
@compute @workgroup_size(256, 1, 1)
fn rms_bwd(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = flat_group(wg, nwg);
    if (row >= pw(0u)) { return; }
    let dim = pw(1u);
    let base = row * dim;
    let rstd = row_rstd(lid.x, base, dim);
    var acc = 0.0;
    for (var c = lid.x; c < dim; c = c + 256u) {
        let dxhat = x2[base + c] * x1[c];
        let xhat = x0[base + c] * rstd;
        acc = acc + dxhat * xhat;
    }
    let dot = tree_sum(lid.x, acc);
    let mean = dot * (1.0 / f32(dim));
    for (var c = lid.x; c < dim; c = c + 256u) {
        let dxhat = x2[base + c] * x1[c];
        let xhat = x0[base + c] * rstd;
        let v = (dxhat - xhat * mean) * rstd;
        report(v);
        y0[base + c] = v;
    }
    if (lid.x == 0u) {
        y1[row] = rstd;
    }
}

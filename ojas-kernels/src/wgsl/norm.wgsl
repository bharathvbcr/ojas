// RMSNorm. A 256-lane workgroup holds 256 / width rows, `width` lanes each,
// where width is the row length rounded up to a power of two and capped at
// 256: a 64-wide row uses 64 lanes, so four rows share a workgroup instead of
// one row leaving 192 lanes idle. Words: 0 rows, 1 dim, 2 eps bits, 3 width.
// Each lane sums the squares of its strided columns and the segment is
// reduced with seg_sum, which pairs lanes as tree_sum does; the lanes a row
// does not reach contribute zeros, so the result is the bits one row per
// workgroup would give.
//
// rstd = 1 / sqrt(mean(x^2) + eps), as the CPU reference forms it (not
// inverseSqrt). A non-positive or non-finite denominator raises the fault
// bit. Lanes past the last row run the reductions with zeros (every lane
// must reach the barriers) and write nothing.

@group(0) @binding(2) var<storage, read> x0: array<f32>;
@group(0) @binding(3) var<storage, read> x1: array<f32>;
@group(0) @binding(4) var<storage, read> x2: array<f32>;
@group(0) @binding(5) var<storage, read_write> y0: array<f32>;
@group(0) @binding(6) var<storage, read_write> y1: array<f32>;

struct Seg {
    row: u32,
    at: u32,
    live: bool,
}

fn seg_of(wg: vec3<u32>, nwg: vec3<u32>, lane: u32) -> Seg {
    let width = pw(3u);
    let per = 256u / width;
    let row = flat_group(wg, nwg) * per + lane / width;
    return Seg(row, lane & (width - 1u), row < pw(0u));
}

fn row_rstd(lane: u32, s: Seg, dim: u32) -> f32 {
    let width = pw(3u);
    var acc = 0.0;
    if (s.live) {
        let base = s.row * dim;
        for (var c = s.at; c < dim; c = c + width) {
            let v = x0[base + c];
            acc = acc + v * v;
        }
    }
    let sum_sq = seg_sum(lane, width, acc);
    let denom = sum_sq / f32(dim) + pf(2u);
    if (s.live && s.at == 0u && (nonfinite(denom) || !(denom > 0.0))) {
        raise();
    }
    return 1.0 / sqrt(denom);
}

// x0 = x, x1 = weight. y0 = y.
@compute @workgroup_size(256, 1, 1)
fn rms_fwd(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    if (flat_group(wg, nwg) * (256u / pw(3u)) >= pw(0u)) { return; }
    let dim = pw(1u);
    let s = seg_of(wg, nwg, lid.x);
    let rstd = row_rstd(lid.x, s, dim);
    if (!s.live) { return; }
    let base = s.row * dim;
    for (var c = s.at; c < dim; c = c + pw(3u)) {
        let v = x0[base + c] * rstd * x1[c];
        report(v);
        y0[base + c] = v;
    }
}

// x0 = x, x1 = weight, x2 = grad_output. y0 = grad_x, y1 = rstd per row.
@compute @workgroup_size(256, 1, 1)
fn rms_bwd(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    if (flat_group(wg, nwg) * (256u / pw(3u)) >= pw(0u)) { return; }
    let dim = pw(1u);
    let width = pw(3u);
    let s = seg_of(wg, nwg, lid.x);
    let rstd = row_rstd(lid.x, s, dim);
    let base = s.row * dim;
    var acc = 0.0;
    if (s.live) {
        for (var c = s.at; c < dim; c = c + width) {
            let dxhat = x2[base + c] * x1[c];
            let xhat = x0[base + c] * rstd;
            acc = acc + dxhat * xhat;
        }
    }
    let dot = seg_sum(lid.x, width, acc);
    if (!s.live) { return; }
    let mean = dot * (1.0 / f32(dim));
    for (var c = s.at; c < dim; c = c + width) {
        let dxhat = x2[base + c] * x1[c];
        let xhat = x0[base + c] * rstd;
        let v = (dxhat - xhat * mean) * rstd;
        report(v);
        y0[base + c] = v;
    }
    if (s.at == 0u) {
        y1[s.row] = rstd;
    }
}

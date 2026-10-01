// Elementwise and per-row-of-heads kernels. One lane per element, 256 lanes a
// workgroup, groups folded into (x, y) so a large tensor stays inside the
// per-dimension dispatch limit. Word 0 is the lane count.
//
// x0..x3 are inputs and y0, y1 outputs. Each entry point's pipeline layout
// lists only the bindings that entry point uses.

@group(0) @binding(2) var<storage, read> x0: array<f32>;
@group(0) @binding(3) var<storage, read> x1: array<f32>;
@group(0) @binding(4) var<storage, read> x2: array<f32>;
@group(0) @binding(5) var<storage, read> x3: array<f32>;
@group(0) @binding(6) var<storage, read_write> y0: array<f32>;
@group(0) @binding(7) var<storage, read_write> y1: array<f32>;

fn lane_index(wg: vec3<u32>, nwg: vec3<u32>, lid: vec3<u32>) -> u32 {
    return flat_group(wg, nwg) * 256u + lid.x;
}

fn put0(i: u32, v: f32) {
    report(v);
    y0[i] = v;
}

fn put1(i: u32, v: f32) {
    report(v);
    y1[i] = v;
}

@compute @workgroup_size(256, 1, 1)
fn silu_fwd(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = lane_index(wg, nwg, lid);
    if (i >= pw(0u)) { return; }
    let v = x0[i];
    put0(i, v * sigmoid(v));
}

// x0 = input, x1 = grad_output.
@compute @workgroup_size(256, 1, 1)
fn silu_bwd(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = lane_index(wg, nwg, lid);
    if (i >= pw(0u)) { return; }
    let v = x0[i];
    let s = sigmoid(v);
    put0(i, x1[i] * s * (1.0 + v * (1.0 - s)));
}

@compute @workgroup_size(256, 1, 1)
fn mul_fwd(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = lane_index(wg, nwg, lid);
    if (i >= pw(0u)) { return; }
    put0(i, x0[i] * x1[i]);
}

// x0 = a, x1 = b, x2 = grad_output. y0 = grad_a, y1 = grad_b.
@compute @workgroup_size(256, 1, 1)
fn mul_bwd(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = lane_index(wg, nwg, lid);
    if (i >= pw(0u)) { return; }
    let g = x2[i];
    put0(i, g * x1[i]);
    put1(i, g * x0[i]);
}

@compute @workgroup_size(256, 1, 1)
fn add_fwd(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = lane_index(wg, nwg, lid);
    if (i >= pw(0u)) { return; }
    put0(i, x0[i] + x1[i]);
}

// Reads x0 and raises the fault bit on a non-finite value. Writes nothing.
@compute @workgroup_size(256, 1, 1)
fn check_finite(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = lane_index(wg, nwg, lid);
    if (i >= pw(0u)) { return; }
    report(x0[i]);
}

// y0 = x0 * word 1.
@compute @workgroup_size(256, 1, 1)
fn scale(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = lane_index(wg, nwg, lid);
    if (i >= pw(0u)) { return; }
    put0(i, x0[i] * pf(1u));
}

// y0 *= word 1, in place. Binding the same buffer as x0 and y0 is a wgpu
// usage conflict, so the in-place form has its own entry point.
@compute @workgroup_size(256, 1, 1)
fn scale_inplace(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = lane_index(wg, nwg, lid);
    if (i >= pw(0u)) { return; }
    put0(i, y0[i] * pf(1u));
}

// x0 = value, x1 = value0, x2 = [lambda].
@compute @workgroup_size(256, 1, 1)
fn vr_fwd(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = lane_index(wg, nwg, lid);
    if (i >= pw(0u)) { return; }
    let s = sigmoid(x2[0]);
    put0(i, (1.0 - s) * x0[i] + s * x1[i]);
}

// x0 = grad_output, x2 = [lambda]. y0 = grad_value, y1 = grad_value0.
@compute @workgroup_size(256, 1, 1)
fn vr_bwd(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = lane_index(wg, nwg, lid);
    if (i >= pw(0u)) { return; }
    let s = sigmoid(x2[0]);
    let g = x0[i];
    put0(i, (1.0 - s) * g);
    put1(i, s * g);
}

// Words: 0 rows*heads*head_dim, 1 heads, 2 head_dim.
// x0 = z [rows, heads] (x @ W^T), x1 = bias [heads], x2 = attn.
@compute @workgroup_size(256, 1, 1)
fn gate_fwd(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = lane_index(wg, nwg, lid);
    if (i >= pw(0u)) { return; }
    let rh = i / pw(2u);
    let z = x0[rh] + x1[rh % pw(1u)];
    report(z);
    put0(i, x2[i] * sigmoid(z));
}

// One lane per (row, head). Words: 0 rows*heads, 1 heads, 2 head_dim.
// x0 = z, x1 = bias, x2 = attn, x3 = grad_output.
// y0 = grad_attn (rows*heads*head_dim), y1 = grad_z (rows*heads).
@compute @workgroup_size(256, 1, 1)
fn gate_bwd(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let rh = lane_index(wg, nwg, lid);
    if (rh >= pw(0u)) { return; }
    let dh = pw(2u);
    let z = x0[rh] + x1[rh % pw(1u)];
    report(z);
    let g = sigmoid(z);
    var grad_g = 0.0;
    let base = rh * dh;
    for (var d = 0u; d < dh; d = d + 1u) {
        let gy = x3[base + d];
        put0(base + d, gy * g);
        grad_g = grad_g + gy * x2[base + d];
    }
    put1(rh, grad_g * g * (1.0 - g));
}

// Half-split RoPE. One lane per (row, column < half).
// Words: 0 rows*half, 1 dim, 2 layout (0 same shape, 1 [time, dim] tables),
// 3 heads, 4 time, 5 direction (0 forward, 1 backward).
// x0 = x or grad_output, x1 = cos, x2 = sin.
@compute @workgroup_size(256, 1, 1)
fn rope(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = lane_index(wg, nwg, lid);
    if (i >= pw(0u)) { return; }
    let dim = pw(1u);
    let half = dim / 2u;
    let row = i / half;
    let col = i % half;
    var coeff = row * dim;
    if (pw(2u) == 1u) {
        coeff = ((row / pw(3u)) % pw(4u)) * dim;
    }
    let base = row * dim;
    let a = x0[base + col];
    let b = x0[base + col + half];
    let c1 = x1[coeff + col];
    let s1 = x2[coeff + col];
    let c2 = x1[coeff + col + half];
    let s2 = x2[coeff + col + half];
    if (pw(5u) == 0u) {
        put0(base + col, a * c1 + (-b) * s1);
        put0(base + col + half, b * c2 + a * s2);
    } else {
        put0(base + col, a * c1 + b * s2);
        put0(base + col + half, -a * s1 + b * c2);
    }
}

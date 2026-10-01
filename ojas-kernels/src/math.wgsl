struct Params { a: u32, b: u32, c: u32, d: u32 }

@group(0) @binding(0) var<storage, read> in0: array<f32>;
@group(0) @binding(1) var<storage, read> in1: array<f32>;
@group(0) @binding(2) var<storage, read_write> out0: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> tile_a: array<f32, 256>;
var<workgroup> tile_b: array<f32, 256>;
var<workgroup> red: array<f32, 64>;

@compute @workgroup_size(16, 16, 1)
fn gemm_tiled(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let m = params.a;
    let k = params.b;
    let n = params.c;
    let row = gid.y;
    let col = gid.x;
    let local_row = lid.y;
    let local_col = lid.x;
    var acc = 0.0;
    let tiles = (k + 15u) / 16u;
    for (var t = 0u; t < tiles; t = t + 1u) {
        let a_col = t * 16u + local_col;
        let b_row = t * 16u + local_row;
        var av = 0.0;
        var bv = 0.0;
        if (row < m && a_col < k) {
            av = in0[row * k + a_col];
        }
        if (col < n && b_row < k) {
            bv = in1[b_row * n + col];
        }
        tile_a[local_row * 16u + local_col] = av;
        tile_b[local_row * 16u + local_col] = bv;
        workgroupBarrier();
        for (var i = 0u; i < 16u; i = i + 1u) {
            let kk = t * 16u + i;
            if (kk < k) {
                acc = acc + tile_a[local_row * 16u + i] * tile_b[i * 16u + local_col];
            }
        }
        workgroupBarrier();
    }
    if (row < m && col < n) {
        out0[row * n + col] = acc;
    }
}

@compute @workgroup_size(64, 1, 1)
fn row_sum(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let cols = params.b;
    let row = wgid.x;
    let _keep = in1[0];
    var acc = 0.0;
    var c = lid.x;
    loop {
        if (c >= cols) { break; }
        acc = acc + in0[row * cols + c];
        c = c + 64u;
    }
    red[lid.x] = acc;
    workgroupBarrier();
    var offset = 32u;
    loop {
        if (offset == 0u) { break; }
        if (lid.x < offset) {
            red[lid.x] = red[lid.x] + red[lid.x + offset];
        }
        workgroupBarrier();
        offset = offset / 2u;
    }
    if (lid.x == 0u) {
        out0[row] = red[0];
    }
}

@compute @workgroup_size(64, 1, 1)
fn softmax_row(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let cols = params.b;
    let row = wgid.x;
    let _keep = in1[0];
    var mx = -3.40282347e+38;
    var c = lid.x;
    loop {
        if (c >= cols) { break; }
        let v = in0[row * cols + c];
        if (v > mx) { mx = v; }
        c = c + 64u;
    }
    red[lid.x] = mx;
    workgroupBarrier();
    var offset = 32u;
    loop {
        if (offset == 0u) { break; }
        if (lid.x < offset) {
            let other = red[lid.x + offset];
            if (other > red[lid.x]) { red[lid.x] = other; }
        }
        workgroupBarrier();
        offset = offset / 2u;
    }
    let peak = red[0];
    workgroupBarrier();
    var sum = 0.0;
    c = lid.x;
    loop {
        if (c >= cols) { break; }
        sum = sum + exp(in0[row * cols + c] - peak);
        c = c + 64u;
    }
    red[lid.x] = sum;
    workgroupBarrier();
    offset = 32u;
    loop {
        if (offset == 0u) { break; }
        if (lid.x < offset) {
            red[lid.x] = red[lid.x] + red[lid.x + offset];
        }
        workgroupBarrier();
        offset = offset / 2u;
    }
    let denom = red[0];
    workgroupBarrier();
    c = lid.x;
    loop {
        if (c >= cols) { break; }
        let slot = row * cols + c;
        out0[slot] = exp(in0[slot] - peak) / denom;
        c = c + 64u;
    }
}

@compute @workgroup_size(64, 1, 1)
fn rms_norm(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cols = params.b;
    let row = gid.x;
    if (row >= params.a) { return; }
    var sum = 0.0;
    for (var c = 0u; c < cols; c = c + 1u) {
        let v = in0[row * cols + c];
        sum = sum + v * v;
    }
    let inv = inverseSqrt(sum / f32(cols) + bitcast<f32>(params.c));
    for (var c = 0u; c < cols; c = c + 1u) {
        let slot = row * cols + c;
        out0[slot] = in0[slot] * inv * in1[c];
    }
}

@compute @workgroup_size(64, 1, 1)
fn silu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.a) { return; }
    let x = in0[i] + in1[i] * 0.0;
    let e = exp(-abs(x));
    let r = 1.0 / (1.0 + e);
    let s = select(e * r, r, x >= 0.0);
    out0[i] = x * s;
}

@compute @workgroup_size(64, 1, 1)
fn mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.a) { return; }
    out0[i] = in0[i] * in1[i];
}

@compute @workgroup_size(64, 1, 1)
fn residual(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.a) { return; }
    out0[i] = in0[i] + in1[i];
}

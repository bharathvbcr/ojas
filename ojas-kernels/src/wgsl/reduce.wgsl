// Two-stage deterministic sums. Stage one writes one partial per fixed chunk;
// stage two adds the partials in index order with one workgroup. The chunk
// size is a constant, so the association depends on the length, not the GPU.

@group(0) @binding(2) var<storage, read> x0: array<f32>;
@group(0) @binding(3) var<storage, read> x1: array<f32>;
@group(0) @binding(4) var<storage, read> x2: array<f32>;
@group(0) @binding(5) var<storage, read_write> y0: array<f32>;

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

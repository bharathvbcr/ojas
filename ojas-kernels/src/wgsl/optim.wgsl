// Global-norm clipping and AdamW.
//
// The norm is amax * sqrt(sum((g / amax)^2)), so the f32 sum of squares cannot
// overflow while the norm itself is finite. `result` is four words per call:
// 0 norm bits, 1 non-finite flag, 2 amax bits.
//
// AdamW runs in f32; WGSL has no f64. The host forms the step scalars in f64
// and rounds each once. adam_update writes the new state in place into fresh
// copies; adam_commit restores the old values into those copies if any lane
// saw a non-finite value, so a bad step is never applied in part.

@group(0) @binding(2) var<storage, read> x0: array<f32>;
@group(0) @binding(3) var<storage, read> x1: array<f32>;
@group(0) @binding(4) var<storage, read> x2: array<f32>;
@group(0) @binding(5) var<storage, read_write> y0: array<f32>;
@group(0) @binding(6) var<storage, read_write> y1: array<f32>;
@group(0) @binding(7) var<storage, read_write> y2: array<f32>;
@group(0) @binding(8) var<storage, read_write> status: array<atomic<u32>>;
@group(0) @binding(9) var<storage, read> status_in: array<u32>;

const CHUNK: u32 = 4096u;

// Words: 0 n, 1 partial base. x0 = grad. y0[base + group] = max |g| over the
// chunk; a non-finite element sets status[1].
@compute @workgroup_size(256, 1, 1)
fn absmax_partial(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let group = flat_group(wg, nwg);
    let n = pw(0u);
    let start = group * CHUNK;
    var peak = 0.0;
    for (var j = 0u; j < CHUNK / 256u; j = j + 1u) {
        let i = start + j * 256u + lid.x;
        if (i < n) {
            let v = x0[i];
            if (nonfinite(v)) {
                atomicOr(&status[1], 1u);
            } else {
                peak = max(peak, abs(v));
            }
        }
    }
    let m = tree_max(lid.x, peak);
    if (lid.x == 0u) {
        y0[pw(1u) + group] = m;
    }
}

// Words: 0 partial count. x0 = partials. status[2] = amax bits.
@compute @workgroup_size(256, 1, 1)
fn absmax_finish(@builtin(local_invocation_id) lid: vec3<u32>) {
    var peak = 0.0;
    for (var i = lid.x; i < pw(0u); i = i + 256u) {
        peak = max(peak, x0[i]);
    }
    let m = tree_max(lid.x, peak);
    if (lid.x == 0u) {
        atomicStore(&status[2], bitcast<u32>(m));
    }
}

// Words: 0 n, 1 partial base. x0 = grad, status_in[2] = amax bits.
@compute @workgroup_size(256, 1, 1)
fn sumsq_partial(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let group = flat_group(wg, nwg);
    let n = pw(0u);
    let amax = bitcast<f32>(status_in[2]);
    var inv = 0.0;
    if (amax > 0.0) {
        inv = 1.0 / amax;
    }
    let start = group * CHUNK;
    var acc = 0.0;
    for (var j = 0u; j < CHUNK / 256u; j = j + 1u) {
        let i = start + j * 256u + lid.x;
        if (i < n) {
            let v = x0[i] * inv;
            acc = acc + v * v;
        }
    }
    let total = tree_sum(lid.x, acc);
    if (lid.x == 0u) {
        y0[pw(1u) + group] = total;
    }
}

// Words: 0 partial count. x0 = partials. status[0] = norm bits.
@compute @workgroup_size(256, 1, 1)
fn clip_finish(@builtin(local_invocation_id) lid: vec3<u32>) {
    var acc = 0.0;
    for (var i = lid.x; i < pw(0u); i = i + 256u) {
        acc = acc + x0[i];
    }
    let total = tree_sum(lid.x, acc);
    if (lid.x == 0u) {
        let amax = bitcast<f32>(atomicLoad(&status[2]));
        atomicStore(&status[0], bitcast<u32>(amax * sqrt(total)));
    }
}

// Words: 0 n, 1 step_size, 2 sqrt(bias_correction2), 3 eps, 4 beta1,
// 5 1 - beta1, 6 beta2, 7 1 - beta2, 8 decay (1 - lr * wd), 9 flags
// (bit 0 weight decay is non-zero, bit 1 lerp uses the low-weight form).
// x0 = grad. y0, y1, y2 = param, moment1, moment2, updated in place.
@compute @workgroup_size(256, 1, 1)
fn adam_update(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = flat_group(wg, nwg) * 256u + lid.x;
    if (i >= pw(0u)) { return; }
    let flags = pw(9u);
    let decays = (flags & 1u) == 1u;
    let g = x0[i];
    let p0 = y0[i];
    var p = p0;
    var m = y1[i];
    var v = y2[i];
    if (decays) {
        p = p * pf(8u);
    }
    if ((flags & 2u) == 2u) {
        m = m + pf(5u) * (g - m);
    } else {
        m = g - (g - m) * pf(4u);
    }
    v = pf(6u) * v + pf(7u) * g * g;
    let denom = sqrt(v) / pf(2u) + pf(3u);
    let delta = (-pf(1u)) * m / denom;
    if (nonfinite(g) || nonfinite(m) || nonfinite(v) || nonfinite(denom) || nonfinite(delta)) {
        atomicOr(&status[0], 1u);
    }
    if (!decays && delta == 0.0) {
        p = p0;
    } else {
        p = p + delta;
        if (nonfinite(p)) {
            atomicOr(&status[0], 1u);
        }
    }
    y0[i] = p;
    y1[i] = m;
    y2[i] = v;
}

// Words: 0 n. x0, x1, x2 = old param, moment1, moment2. y0, y1, y2 = new.
// status_in[0] != 0 restores the old state and raises the fault bit.
@compute @workgroup_size(256, 1, 1)
fn adam_commit(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = flat_group(wg, nwg) * 256u + lid.x;
    if (i >= pw(0u)) { return; }
    if (status_in[0] == 0u) { return; }
    if (i == 0u) { raise(); }
    y0[i] = x0[i];
    y1[i] = x1[i];
    y2[i] = x2[i];
}

// Muon NS5 in f32, the CPU reference's order of operations. The host runs
// every kernel before muon_commit with a per-call fault word bound at
// binding 1, so `raise` there marks only this call; muon_commit then reads
// that word as status_in and runs against the context's fault word.

// Words: 0 n, 1 momentum, 2 nesterov. x0 = grad, x1 = momentum.
// y0 = buf = momentum * m + g; y1 = update (g + momentum * buf, or buf).
@compute @workgroup_size(256, 1, 1)
fn muon_momentum(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = flat_group(wg, nwg) * 256u + lid.x;
    if (i >= pw(0u)) { return; }
    let mom = pf(1u);
    let g = x0[i];
    let b = mom * x1[i] + g;
    report(b);
    var upd = b;
    if (pw(2u) != 0u) {
        upd = g + mom * b;
        report(upd);
    }
    y0[i] = b;
    y1[i] = upd;
}

// Words: 0 n, 1 eps. x0 = update, status_in = the norm words of
// clip_finish (0 norm, 1 non-finite flag). y0 = x0 / (norm + eps).
@compute @workgroup_size(256, 1, 1)
fn muon_normalize(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = flat_group(wg, nwg) * 256u + lid.x;
    if (i >= pw(0u)) { return; }
    let denom = bitcast<f32>(status_in[0]) + pf(1u);
    if (status_in[1] != 0u || nonfinite(denom) || denom == 0.0) {
        raise();
        return;
    }
    let v = x0[i] / denom;
    report(v);
    y0[i] = v;
}

// Words: 0 n, 1 s0, 2 s1. y0 = s0 * x0 + s1 * x1.
@compute @workgroup_size(256, 1, 1)
fn muon_lincomb(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = flat_group(wg, nwg) * 256u + lid.x;
    if (i >= pw(0u)) { return; }
    let v = pf(1u) * x0[i] + pf(2u) * x1[i];
    report(v);
    y0[i] = v;
}

// Words: 0 n, 1 alpha (-lr * scale), 2 decay (1 - lr * wd), 3 decays.
// x0 = param, x1 = orthogonalized update. y0 = new param.
@compute @workgroup_size(256, 1, 1)
fn muon_apply(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = flat_group(wg, nwg) * 256u + lid.x;
    if (i >= pw(0u)) { return; }
    var p = x0[i];
    if (pw(3u) != 0u) {
        p = p * pf(2u);
    }
    let v = p + pf(1u) * x1[i];
    report(v);
    y0[i] = v;
}

// Words: 0 n. x0 = new param, x1 = new momentum, status_in = this call's
// fault word. Clean: y0 = param, y1 = momentum are overwritten. Otherwise
// nothing is written and the op's bit is raised in the context's word.
@compute @workgroup_size(256, 1, 1)
fn muon_commit(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = flat_group(wg, nwg) * 256u + lid.x;
    if (i >= pw(0u)) { return; }
    // Both mask words: the op's bit is in word 1 when its index is 32 or more.
    if ((status_in[0] | status_in[1]) != 0u) {
        if (i == 0u) { raise(); }
        return;
    }
    y0[i] = x0[i];
    y1[i] = x1[i];
}

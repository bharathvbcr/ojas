// AdamW and Muon. The global norm both use (clip_grad_norm, Muon's
// normalization) is the multi-tensor pass in clip.wgsl.
//
// AdamW runs in f32; WGSL has no f64. The host forms the step scalars in f64
// and rounds each once. Transactional without copies, as Metal's
// ojas_adamw_check / ojas_adamw_apply: adam_check computes every element and
// only flags status[0]; adam_apply, recorded after it, recomputes with the
// same function and writes in place only if no lane flagged, so a bad step
// is never applied in part and what is written is what was checked.

@group(0) @binding(2) var<storage, read> x0: array<f32>;
@group(0) @binding(3) var<storage, read> x1: array<f32>;
@group(0) @binding(4) var<storage, read> x2: array<f32>;
@group(0) @binding(5) var<storage, read_write> y0: array<f32>;
@group(0) @binding(6) var<storage, read_write> y1: array<f32>;
@group(0) @binding(7) var<storage, read_write> y2: array<f32>;
@group(0) @binding(8) var<storage, read_write> status: array<atomic<u32>>;
@group(0) @binding(9) var<storage, read> status_in: array<u32>;

// Words: 0 n, 1 step_size, 2 sqrt(bias_correction2), 3 eps, 4 beta1,
// 5 1 - beta1, 6 beta2, 7 1 - beta2, 8 decay (1 - lr * wd), 9 flags
// (bit 0 weight decay is non-zero, bit 1 lerp uses the low-weight form).
// x0 = grad. y0, y1, y2 = param, moment1, moment2.
struct AdamOut { p: f32, m: f32, v: f32, bad: bool, }

// Element i's new state, and whether the step is non-finite there.
fn adam_elem(i: u32) -> AdamOut {
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
    var bad = nonfinite(g) || nonfinite(m) || nonfinite(v) || nonfinite(denom) || nonfinite(delta);
    if (!decays && delta == 0.0) {
        p = p0;
    } else {
        p = p + delta;
        bad = bad || nonfinite(p);
    }
    return AdamOut(p, m, v, bad);
}

// Writes nothing but status[0], set when any element's step is non-finite.
@compute @workgroup_size(256, 1, 1)
fn adam_check(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = flat_group(wg, nwg) * 256u + lid.x;
    if (i >= pw(0u)) { return; }
    if (adam_elem(i).bad) {
        atomicOr(&status[0], 1u);
    }
}

// status_in[0] != 0 (adam_check flagged): nothing is written and the op's
// fault bit is raised. Otherwise every element is updated in place.
@compute @workgroup_size(256, 1, 1)
fn adam_apply(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = flat_group(wg, nwg) * 256u + lid.x;
    if (i >= pw(0u)) { return; }
    if (status_in[0] != 0u) {
        if (i == 0u) { raise(); }
        return;
    }
    let o = adam_elem(i);
    y0[i] = o.p;
    y1[i] = o.m;
    y2[i] = o.v;
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

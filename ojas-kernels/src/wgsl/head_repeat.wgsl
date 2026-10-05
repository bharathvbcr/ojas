// Expand or reduce a head axis. One lane per destination element.
//
// Layout is head-major planes of `plane` floats (one `[T, D]` head). Query
// plane `p` reads KV plane `p / rep`, because a batch of `Hq` query heads
// with `Hq` divisible by `rep` satisfies
// `(b * Hq + h) / rep = b * (Hq / rep) + h / rep`. The sum walks `r` in
// `0 .. rep` so the reduction order is increasing query head.
//
// Words: 0 element count of `dst`, 1 plane, 2 rep. `plane == 0` or
// `rep == 0` is a host bug; lane 0 raises and nothing is written.

@group(0) @binding(2) var<storage, read> src: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn head_repeat(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = flat_group(wg, nwg) * 256u + lid.x;
    let n = pw(0u);
    if (i >= n) { return; }
    let plane = pw(1u);
    let rep = pw(2u);
    if (plane == 0u || rep == 0u) {
        if (i == 0u) { raise(); }
        return;
    }
    let within = i % plane;
    let src_p = (i / plane) / rep;
    dst[i] = src[src_p * plane + within];
}

@compute @workgroup_size(256, 1, 1)
fn head_sum(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = flat_group(wg, nwg) * 256u + lid.x;
    let n = pw(0u);
    if (i >= n) { return; }
    let plane = pw(1u);
    let rep = pw(2u);
    if (plane == 0u || rep == 0u) {
        if (i == 0u) { raise(); }
        return;
    }
    let within = i % plane;
    let dst_p = i / plane;
    var acc = 0.0;
    for (var r = 0u; r < rep; r = r + 1u) {
        acc = acc + src[(dst_p * rep + r) * plane + within];
    }
    report(acc);
    dst[i] = acc;
}

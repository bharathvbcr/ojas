// Shared by every ojas-wgpu module.
//
// Binding 0 is sixteen u32 parameter words. Word 15 is the fault bit of the
// op that launched the kernel. Binding 1 is the backend's fault word: a
// kernel that produces a non-finite value ORs its bit into word 0 and keeps
// going, and the host reports it at the next synchronization. Word 1 holds
// the first faulting op (bit index + 1, 0 = none): only the lane whose OR
// newly set the bit tries to claim it, so the clean path costs nothing and a
// fault costs one compare-exchange per op. Dispatches all write binding 1,
// so wgpu orders them, and "first" is first in recording order.
//
// Non-finite tests use the exponent bits, not `x != x`, so a compiler that
// assumes finite math cannot fold them away.

struct Params { w: array<vec4<u32>, 4>, }

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> fault: array<atomic<u32>>;

fn pw(i: u32) -> u32 { return P.w[i >> 2u][i & 3u]; }
fn pf(i: u32) -> f32 { return bitcast<f32>(pw(i)); }
fn nonfinite(x: f32) -> bool { return (bitcast<u32>(x) & 0x7f800000u) == 0x7f800000u; }
fn claim_first(id: u32) {
    // The weak form may fail spuriously; retry until claimed or taken.
    loop {
        let r = atomicCompareExchangeWeak(&fault[1], 0u, id);
        if (r.exchanged || r.old_value != 0u) { break; }
    }
}
fn raise() {
    let bit = pw(15u);
    let old = atomicOr(&fault[0], bit);
    if (bit != 0u && (old & bit) == 0u) { claim_first(firstTrailingBit(bit) + 1u); }
}
fn report(x: f32) { if (nonfinite(x)) { raise(); } }
fn flat_group(wg: vec3<u32>, nwg: vec3<u32>) -> u32 { return wg.y * nwg.x + wg.x; }

// Same branch split as the CPU reference, so neither side overflows exp().
fn sigmoid(x: f32) -> f32 {
    if (x >= 0.0) {
        return 1.0 / (1.0 + exp(-x));
    }
    let z = exp(x);
    return z / (1.0 + z);
}

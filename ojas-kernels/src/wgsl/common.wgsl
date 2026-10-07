// Shared by every ojas-wgpu module.
//
// Binding 0 is sixteen u32 parameter words. Word 15 is the fault id of the
// op that launched the kernel: its op index + 1, or 0 for a launch that
// reports nothing. Binding 1 is the backend's fault words. Words 0 and 1
// are a 64-bit op mask, ops 0..31 in word 0 and ops 32..63 in word 1: a
// kernel that produces a non-finite value ORs its op's bit in and keeps
// going, and the host reports it at the next synchronization. Word 2 holds
// the first faulting op (index + 1, 0 = none): only the lane whose OR newly
// set the bit tries to claim it, so the clean path costs nothing and a
// fault costs one compare-exchange per op. Dispatches all write binding 1,
// so wgpu orders them, and "first" is first in recording order. Word 3 is
// unused.
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
        let r = atomicCompareExchangeWeak(&fault[2], 0u, id);
        if (r.exchanged || r.old_value != 0u) { break; }
    }
}
fn raise() {
    let id = pw(15u);
    // The host refuses an id above 64, so this never drops a real fault; it
    // keeps a bad id from writing into the first-op word.
    if (id == 0u || id > 64u) { return; }
    let index = id - 1u;
    let bit = 1u << (index & 31u);
    let old = atomicOr(&fault[index >> 5u], bit);
    if ((old & bit) == 0u) { claim_first(id); }
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

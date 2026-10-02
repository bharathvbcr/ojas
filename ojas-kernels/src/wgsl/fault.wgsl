// Fault hand-off to the host: a module of its own so the first read on a
// context compiles two one-lane kernels, not a whole op family.
//
// `held` is the context's held word pair. fault_hold moves the live fault
// words into it and clears the live ones; each read records it just before
// copying `held` out, so a read whose copy is never observed (a failed wait
// or map) leaves the bits held for the next read. An older first-op entry
// wins over a newer one.

@group(0) @binding(2) var<storage, read_write> held: array<atomic<u32>>;

@compute @workgroup_size(1, 1, 1)
fn fault_hold() {
    atomicOr(&held[0], atomicExchange(&fault[0], 0u));
    let first = atomicExchange(&fault[1], 0u);
    if (first != 0u) {
        loop {
            let r = atomicCompareExchangeWeak(&held[1], 0u, first);
            if (r.exchanged || r.old_value != 0u) { break; }
        }
    }
}

// Words: 0 bits the host observed, 1 first-op entry it observed. Clears
// exactly those from `held`, so bits a later hold added stay held.
@compute @workgroup_size(1, 1, 1)
fn fault_release() {
    atomicAnd(&held[0], ~pw(0u));
    let first = pw(1u);
    if (first != 0u) {
        loop {
            let r = atomicCompareExchangeWeak(&held[1], first, 0u);
            if (r.exchanged || r.old_value != first) { break; }
        }
    }
}

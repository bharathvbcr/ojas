// Fault hand-off to the host: a module of its own so the first read on a
// context compiles one one-lane kernel, not a whole op family.
//
// `held` is an array of hold slots of four words each: 0 and 1 the 64-bit op
// mask, 2 the first-op entry, 3 unused. Every read owns one slot for the
// span between its hold and the host seeing the copy (ft-84e2b74e), so no two
// reads ever hold the same words: the hold writes its slot (a store, not a
// merge), the read copies exactly that slot out, and nothing needs clearing
// afterwards. A pair is moved out of the live words by one single-lane
// dispatch, so it is consistent when it is copied, and it is delivered to
// exactly one read.
//
// A read whose copy is never observed (a failed wait or map) leaves its slot
// holding the pair; the host names such slots (orphans, oldest first) in the
// next hold's parameter words and that hold folds them into its own slot,
// older entries first, so the older first-op entry wins and no fault is
// lost or reported twice.
//
// Parameter words: 0 the slot this hold fills, 1 the number of orphan slots
// to fold in (at most HOLD_ORPHANS), 2.. their slot indices. A slot the host
// also lists as an orphan may be the hold's own.

const HOLD_ORPHANS: u32 = 13u;

@group(0) @binding(2) var<storage, read_write> held: array<atomic<u32>>;

@compute @workgroup_size(1, 1, 1)
fn fault_hold() {
    var lo = 0u;
    var hi = 0u;
    var first = 0u;
    let n = min(pw(1u), HOLD_ORPHANS);
    for (var i = 0u; i < n; i = i + 1u) {
        let o = pw(2u + i) * 4u;
        lo = lo | atomicExchange(&held[o], 0u);
        hi = hi | atomicExchange(&held[o + 1u], 0u);
        let f = atomicExchange(&held[o + 2u], 0u);
        if (first == 0u) { first = f; }
    }
    lo = lo | atomicExchange(&fault[0], 0u);
    hi = hi | atomicExchange(&fault[1], 0u);
    let live_first = atomicExchange(&fault[2], 0u);
    if (first == 0u) { first = live_first; }
    let s = pw(0u) * 4u;
    atomicStore(&held[s], lo);
    atomicStore(&held[s + 1u], hi);
    atomicStore(&held[s + 2u], first);
    atomicStore(&held[s + 3u], 0u);
}

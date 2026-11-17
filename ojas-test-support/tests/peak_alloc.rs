//! `PeakAlloc` sees allocations, in-place and moving reallocations, and frees.
//! One test only: the counters are process-wide, so nothing else may allocate
//! while it measures.

use ojas_test_support::PeakAlloc;

#[global_allocator]
static ALLOC: PeakAlloc = PeakAlloc::new();

const MIB: usize = 1 << 20;

#[test]
fn the_peak_tracks_live_bytes_and_realloc_counts_the_net_change() {
    let base = ALLOC.live();
    let (buf, grew) = ALLOC.measure(|| vec![1u8; 4 * MIB]);
    assert!(grew >= 4 * MIB && grew < 4 * MIB + 64 * 1024, "grew {grew}");
    assert!(ALLOC.live() >= base + 4 * MIB);
    drop(buf);
    assert!(
        ALLOC.live() < base + MIB,
        "the buffer's bytes were returned"
    );
    assert!(ALLOC.peak() >= base + 4 * MIB, "the peak survives the free");

    // `reserve_exact` reallocates one buffer. In place, the peak is the net
    // growth. Moved, both blocks were live during the copy, so the peak is
    // old plus new. Which one happens is the system allocator's choice, so
    // the branch is read from the pointer.
    for (old, new) in [(2 * MIB, 5 * MIB), (64, 64 * 1024)] {
        let mut v: Vec<u8> = Vec::with_capacity(old);
        let before = v.as_ptr();
        ALLOC.reset_peak();
        let start = ALLOC.live();
        v.reserve_exact(new);
        let moved = v.as_ptr() != before;
        let peak = ALLOC.peak() - start;
        let cap = v.capacity();
        if moved {
            assert!(
                peak >= cap,
                "a moved realloc charges the new block beside the old one: \
                 {old} -> {cap}, peak {peak}"
            );
        } else {
            assert!(
                peak >= cap - old && peak < cap,
                "an in-place realloc charges only the growth: {old} -> {cap}, peak {peak}"
            );
        }
        assert!(
            ALLOC.live() - start == cap - old,
            "after a realloc only the net change stays live"
        );
        drop(v);
    }

    let (z, zeroed) = ALLOC.measure(|| vec![0u8; 2 * MIB]);
    assert!(zeroed >= 2 * MIB, "alloc_zeroed is counted, got {zeroed}");
    drop(z);
}

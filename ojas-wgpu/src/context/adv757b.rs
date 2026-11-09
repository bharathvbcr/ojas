//! Adversarial tests, second pass, for ft-757cb6b2 (lane 757wadv2). Not for merge.
use super::*;

impl WgpuContext {
    /// Test seam: write raw words into the held fault buffer (as a torn copy
    /// would leave them) and let the next read observe them.
    pub(crate) fn adv_set_held(&self, bits: u64, first: u32) {
        let words = [bits as u32, (bits >> 32) as u32, first, 0u32];
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        self.inner
            .queue
            .as_ref()
            .expect("queue")
            .write_buffer(&self.inner.held, 0, &bytes);
    }
}

// ---- attack 5: torn words ----

#[test]
fn adv2_a5_normalize_never_panics_and_fails_closed() {
    let edge_bits = [0u64, 1, 1 << 2, 1 << 63, u64::MAX, 0xdead_beef_0000_0001];
    let edge_first = [0u32, 1, 2, 38, 39, 63, 64, 65, 66, 1 << 31, u32::MAX - 1, u32::MAX];
    for &b in &edge_bits {
        for &f in &edge_first {
            let (nb, nf) = normalize_fault_words(b, f);
            if b == 0 && f == 0 {
                assert_eq!((nb, nf), (0, 0));
                continue;
            }
            assert!(nb != 0 || nf != 0, "({b:#x},{f}) normalized to clean");
            assert_eq!(nb & b, b, "normalization dropped a bit");
            if (1..=64).contains(&nf) {
                assert_ne!(nb & (1u64 << (nf - 1)), 0, "first {nf} not in {nb:#x}");
            }
            if nb != 0 {
                assert_ne!(nf, 0, "({b:#x},{f}) -> ({nb:#x}, 0): no first op");
            }
        }
    }
}

/// Torn raw words really observed on the device through the real read path:
/// no panic, the normalized pair, and the release leaves `held` clean.
#[test]
fn adv2_a5_torn_words_on_the_device_are_observed_normalized_and_released() {
    let ctx = WgpuContext::open().expect("wgpu adapter");
    ctx.sync().unwrap();
    type Pair = (u64, u32);
    let cases: &[(Pair, Pair)] = &[
        ((0, 5), (1 << 4, 5)),
        ((1 << 2, 0), (1 << 2, 3)),
        ((0b1100, 0), (0b1100, 3)),
        ((1 << 2, 9), ((1 << 2) | (1 << 8), 9)),
        ((0, 64), (1 << 63, 64)),
        ((0, 65), (0, 65)),
        ((1, 65), (1, 65)),
        ((0, u32::MAX), (0, u32::MAX)),
        ((u64::MAX, 0), (u64::MAX, 1)),
        ((u64::MAX, u32::MAX), (u64::MAX, u32::MAX)),
    ];
    for &(raw, want) in cases {
        ctx.adv_set_held(raw.0, raw.1);
        let got = ctx.sync_observed().unwrap();
        assert_eq!(got, want, "raw {raw:?}");
        let again = ctx.sync_observed().unwrap();
        assert_eq!(again, (0, 0), "raw {raw:?}: held not clean after the release");
        assert_eq!(ctx.take_faults(), (0, 0), "raw {raw:?}: also recorded");
    }
}

/// Garbage first word with live bits that arrive after the read: the release
/// of the torn pair must not clear a LATER real fault's bits or first op.
#[test]
fn adv2_a5_release_of_a_torn_pair_keeps_a_later_fault() {
    let ctx = WgpuContext::open().expect("wgpu adapter");
    ctx.sync().unwrap();
    // The observed pair says first=9 (not in bits); the device later holds a
    // real fault (bit 2, first 3).
    ctx.adv_set_held(1 << 2, 9);
    let got = ctx.sync_observed().unwrap();
    assert_eq!(got, ((1 << 2) | (1 << 8), 9));
    ctx.adv_set_held(1 << 20, 21);
    assert_eq!(ctx.sync_observed().unwrap(), (1 << 20, 21));
}

// ---- attack 6: ErrorQueue ----

#[test]
fn adv2_a6_truncation_lands_on_a_char_boundary_for_every_straddle() {
    for (ch, w) in [("\u{e9}", 2usize), ("\u{4e16}", 3), ("\u{1f642}", 4)] {
        for pad in (UNCAPTURED_MSG_BYTES - w - 1)..=(UNCAPTURED_MSG_BYTES + 1) {
            let msg = format!("{}{}tail", "a".repeat(pad), ch);
            let mut q = ErrorQueue::default();
            q.push(msg.clone());
            let kept = &q.msgs[0];
            assert!(std::str::from_utf8(kept.as_bytes()).is_ok());
            let marker = kept.find("... [truncated").expect("marked");
            assert!(marker <= UNCAPTURED_MSG_BYTES, "{marker} > cap");
            assert!(msg.starts_with(&kept[..marker]), "prefix preserved");
            // The cut is as late as a boundary allows.
            assert!(UNCAPTURED_MSG_BYTES - marker < w, "pad {pad} w {w}: cut at {marker}");
        }
    }
}

#[test]
fn adv2_a6_exactly_at_the_cap_is_untouched_and_one_over_is_cut() {
    let mut q = ErrorQueue::default();
    q.push("x".repeat(UNCAPTURED_MSG_BYTES));
    q.push("x".repeat(UNCAPTURED_MSG_BYTES + 1));
    assert_eq!(q.msgs[0].len(), UNCAPTURED_MSG_BYTES);
    assert!(q.msgs[1].contains("truncated, 4097 bytes"));
    // An all-4-byte message cuts cleanly too.
    let mut q = ErrorQueue::default();
    q.push("\u{1f642}".repeat(5000));
    assert!(q.msgs[0].starts_with("\u{1f642}"));
}

/// The doc claims "at most 4096 bytes each are kept" (backend.rs module doc,
/// ErrorQueue doc). A cut message carries a marker on top of the 4096 bytes.
#[test]
fn adv2_a6_doc_claim_at_most_4096_bytes_each() {
    let mut q = ErrorQueue::default();
    q.push("x".repeat(100_000));
    let n = q.msgs[0].len();
    eprintln!("ADV2 A6 a truncated message is {n} bytes (cap {UNCAPTURED_MSG_BYTES})");
    assert!(n <= UNCAPTURED_MSG_BYTES, "kept message is {n} bytes");
}

#[test]
fn adv2_a6_dropped_count_is_exact_under_many_threads() {
    let ctx = WgpuContext::open().expect("wgpu adapter");
    let (threads, per) = (16usize, 200usize);
    std::thread::scope(|s| {
        for t in 0..threads {
            let ctx = &ctx;
            s.spawn(move || {
                for i in 0..per {
                    // multi-byte, long enough to need a cut in some
                    let body = if (t + i) % 3 == 0 {
                        "\u{4e16}".repeat(2000)
                    } else {
                        "e".to_string()
                    };
                    ctx.push_uncaptured(&body);
                }
            });
        }
    });
    let (kept, dropped, bytes) = {
        let q = lock(&ctx.inner.errors);
        (q.msgs.len(), q.dropped, q.msgs.iter().map(String::len).max().unwrap())
    };
    assert_eq!(kept, UNCAPTURED_CAP);
    assert_eq!(kept as u64 + dropped, (threads * per) as u64);
    eprintln!("ADV2 A6 kept={kept} dropped={dropped} max_msg_bytes={bytes}");
    let text = format!("{:?}", ctx.check_errors().unwrap_err());
    assert!(text.contains(&format!("and {dropped} more uncaptured errors dropped")));
}

// ---- attack 1/3: overlapping reads on one context ----

/// Many threads read at once while `held` has a fault: documented to be able
/// to report it to several. Measures how many; asserts at-least-once and a
/// clean device afterwards.
#[test]
fn adv2_a3_overlapping_reads_of_one_held_fault() {
    let ctx = WgpuContext::open().expect("wgpu adapter");
    ctx.sync().unwrap();
    let threads = 8;
    let rounds = 200;
    let mut max_dup = 0usize;
    let mut total_dup = 0usize;
    let mut zero = 0usize;
    for _ in 0..rounds {
        ctx.adv_set_held(1 << 3, 4);
        let barrier = std::sync::Barrier::new(threads);
        let got: usize = std::thread::scope(|s| {
            let hs: Vec<_> = (0..threads)
                .map(|_| {
                    let (ctx, barrier) = (&ctx, &barrier);
                    s.spawn(move || {
                        barrier.wait();
                        usize::from(ctx.sync_observed().unwrap() != (0, 0))
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).sum()
        });
        if got == 0 {
            zero += 1;
        }
        max_dup = max_dup.max(got);
        total_dup += got.saturating_sub(1);
        assert_eq!(ctx.sync_observed().unwrap(), (0, 0), "device not clean");
    }
    eprintln!(
        "ADV2 A3 overlap: rounds={rounds} threads={threads} max_deliveries_of_one_fault={max_dup} extra_deliveries={total_dup} lost={zero}"
    );
    assert_eq!(zero, 0, "a held fault was lost");
    assert_eq!(total_dup, 0, "one held fault was delivered to several readers");
}

//! The dequant fault ring (`DEQUANT_FAULT_WORDS`): its exhaustion is a named wait, a word is held by
//! exactly one pending dequant, and it returns to the ring, zeroed, on every path that ends the
//! dequant: a scan, and each failure after the word was taken.
//!
//! These tests drive the device thread's own `Worker`, so they read the ring's state directly. The
//! invariant behind every one of them ([`assert_ring_whole`]): each of the 64 words is either free
//! or held by exactly one pending entry, never both and never neither, and every free word reads
//! zero. The error paths that cannot be reached from outside are armed with `Cmd::FailNextDequant`.

use super::*;
use ojas_core::{AffineBits, Tensor};

const OP: &str = "dequant_affine_bf16";

fn worker() -> Worker {
    Worker::open(Arc::new(Waits::default()), Budget::new(1 << 36)).expect("Metal device")
}

fn host() -> Budget {
    Budget::new(1 << 24)
}

fn upload(w: &mut Worker, src: Tensor, n: usize) -> Arg {
    match w.command(Cmd::Upload { src }) {
        Ok(Reply::Bufs(b)) if b.len() == 1 => Arg {
            id: b[0].id,
            off: 0,
            n,
        },
        other => panic!("upload: {other:?}"),
    }
}

/// One q8 group on the device.
struct Triple {
    packed: Arg,
    scales: Arg,
    biases: Arg,
}

fn triple(w: &mut Worker, word: u32, scale: u16, bias: u16) -> Triple {
    let h = host();
    Triple {
        packed: upload(w, Tensor::from_u32(&[word; 16], &[16], &h).unwrap(), 16),
        scales: upload(w, Tensor::from_bf16_bits(&[scale], &[1], &h).unwrap(), 1),
        biases: upload(w, Tensor::from_bf16_bits(&[bias], &[1], &h).unwrap(), 1),
    }
}

/// Codes 0, scale 1.0, bias 0.5: every value is 0.5.
fn good(w: &mut Worker) -> Triple {
    triple(w, 0, 0x3F80, 0x3F00)
}

/// Codes 255, scale and bias `bf16::MAX`: finite operands, an overflowing result.
fn overflowing(w: &mut Worker) -> Triple {
    triple(w, u32::MAX, 0x7F7F, 0x7F7F)
}

fn dequant(w: &mut Worker, t: &Triple) -> Res<Reply> {
    w.command(Cmd::DequantAffine {
        packed: t.packed,
        scales: t.scales,
        biases: t.biases,
        bits: AffineBits::Q8,
        rows: 1,
        cols: 64,
    })
}

fn arm(w: &mut Worker, at: DequantFail) {
    assert!(matches!(
        w.command(Cmd::FailNextDequant { at }),
        Ok(Reply::Done)
    ));
}

fn sync(w: &mut Worker) -> Res<Reply> {
    w.command(Cmd::Sync)
}

fn fault_ring_waits(w: &Worker) -> u64 {
    w.waits.snapshot().fault_ring
}

/// Words held by pending entries.
fn held(w: &Worker) -> Vec<usize> {
    w.book
        .borrow()
        .pending
        .iter()
        .filter_map(|e| e.fault_word)
        .collect()
}

fn free(w: &Worker) -> usize {
    w.book.borrow().free_words.len()
}

/// Every word is free xor held once, and every free word is zero.
#[track_caller]
fn assert_ring_whole(w: &Worker, tag: &str) {
    let book = w.book.borrow();
    let mut seen = vec![0u8; DEQUANT_FAULT_WORDS];
    for &i in &book.free_words {
        seen[i] += 1;
        let word = w.fault_words[i].try_contents_u32().unwrap();
        assert_eq!(word[0], 0, "{tag}: free word {i} is not zero");
    }
    for e in &book.pending {
        if let Some(i) = e.fault_word {
            seen[i] += 1;
        }
    }
    for (i, &n) in seen.iter().enumerate() {
        assert_eq!(
            n, 1,
            "{tag}: ring word {i} is held {n} times (want exactly once)"
        );
    }
}

fn values_of(w: &mut Worker, reply: Reply) -> Vec<u16> {
    let Reply::Bufs(b) = reply else {
        panic!("not a buffer: {reply:?}")
    };
    match w.command(Cmd::Read {
        id: b[0].id,
        off: 0,
        len: 64 * 2,
    }) {
        Ok(Reply::Bytes(bytes)) => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_ne_bytes(*c))
            .collect(),
        other => panic!("read: {other:?}"),
    }
}

#[track_caller]
fn assert_reports_once(w: &mut Worker, tag: &str) {
    match sync(w) {
        Err(OjasError::NonFinite { op }) => assert_eq!(op, OP, "{tag}"),
        other => panic!("{tag}: expected NonFinite {{ {OP} }}, got {other:?}"),
    }
    assert!(sync(w).is_ok(), "{tag}: a fault is reported once");
}

#[test]
fn ring_exhaustion_is_a_named_wait_and_a_sync_frees_every_word() {
    let mut w = worker();
    let t = good(&mut w);
    sync(&mut w).unwrap();
    assert_eq!(free(&w), DEQUANT_FAULT_WORDS);
    let before = fault_ring_waits(&w);
    for i in 0..DEQUANT_FAULT_WORDS {
        dequant(&mut w, &t).unwrap_or_else(|e| panic!("dequant {i}: {e:?}"));
    }
    assert_eq!(fault_ring_waits(&w), before, "64 dequants fit the ring");
    assert_eq!((free(&w), held(&w).len()), (0, DEQUANT_FAULT_WORDS));
    assert_ring_whole(&w, "full");

    let last = dequant(&mut w, &t).unwrap();
    assert_eq!(fault_ring_waits(&w), before + 1, "the 65th forces one wait");
    assert_eq!((free(&w), held(&w).len()), (DEQUANT_FAULT_WORDS - 1, 1));
    assert_ring_whole(&w, "after the forced settle");
    sync(&mut w).unwrap();
    assert_eq!(free(&w), DEQUANT_FAULT_WORDS, "a sync scans every entry");
    assert_ring_whole(&w, "after sync");
    assert!(values_of(&mut w, last).iter().all(|&v| v == 0x3F00));
}

#[test]
fn a_word_comes_back_when_the_ring_settle_fails() {
    let mut w = worker();
    let t = good(&mut w);
    sync(&mut w).unwrap();
    for _ in 0..DEQUANT_FAULT_WORDS {
        dequant(&mut w, &t).unwrap();
    }
    arm(&mut w, DequantFail::RingSettle);
    let err = dequant(&mut w, &t).expect_err("the ring settle was armed to fail");
    assert!(format!("{err:?}").contains("injected"), "{err:?}");
    // It took no word and settled nothing.
    assert_eq!((free(&w), held(&w).len()), (0, DEQUANT_FAULT_WORDS));
    assert_ring_whole(&w, "after a failed ring settle");
    // Unarmed, the same call forces the settle and succeeds.
    dequant(&mut w, &t).unwrap();
    assert_eq!(held(&w).len(), 1);
    assert_ring_whole(&w, "retried");
    sync(&mut w).unwrap();
    assert_eq!(free(&w), DEQUANT_FAULT_WORDS);
}

#[test]
fn a_word_comes_back_when_the_status_entry_fails() {
    let mut w = worker();
    let t = good(&mut w);
    sync(&mut w).unwrap();
    arm(&mut w, DequantFail::Status);
    let err = dequant(&mut w, &t).expect_err("the status entry was armed to fail");
    assert!(format!("{err:?}").contains("injected"), "{err:?}");
    assert_eq!((free(&w), held(&w).len()), (DEQUANT_FAULT_WORDS, 0));
    assert_ring_whole(&w, "after a failed status entry");
    // And the ring serves a full 64 afterwards.
    for _ in 0..DEQUANT_FAULT_WORDS {
        dequant(&mut w, &t).unwrap();
    }
    assert_eq!(fault_ring_waits(&w), 0);
}

#[test]
fn a_word_comes_back_when_the_dispatch_fails_and_nothing_is_reported() {
    let mut w = worker();
    let t = good(&mut w);
    sync(&mut w).unwrap();
    arm(&mut w, DequantFail::Dispatch);
    let err = dequant(&mut w, &t).expect_err("the dispatch was armed to fail");
    assert!(format!("{err:?}").contains("injected"), "{err:?}");
    // `command` dropped the failed command's entry and gave its word back.
    assert_eq!((free(&w), held(&w).len()), (DEQUANT_FAULT_WORDS, 0));
    assert_ring_whole(&w, "after a failed dispatch");
    sync(&mut w).expect("the word was never raised, so nothing is reported");
}

#[test]
fn a_raised_word_is_reported_zeroed_and_freed() {
    let mut w = worker();
    let (ok, bad) = (good(&mut w), overflowing(&mut w));
    sync(&mut w).unwrap();
    dequant(&mut w, &ok).unwrap();
    dequant(&mut w, &bad).unwrap();
    dequant(&mut w, &ok).unwrap();
    assert_reports_once(&mut w, "one bad dequant of three");
    assert_eq!(free(&w), DEQUANT_FAULT_WORDS);
    assert_ring_whole(&w, "after the report");
    // The raised word was zeroed on the way back: every later dequant of 64 reports nothing.
    for _ in 0..DEQUANT_FAULT_WORDS {
        dequant(&mut w, &ok).unwrap();
    }
    sync(&mut w).expect("a stale fault word reported a clean dequant");
}

#[test]
fn the_dequant_that_forces_the_settle_can_itself_fault() {
    let mut w = worker();
    let (ok, bad) = (good(&mut w), overflowing(&mut w));
    sync(&mut w).unwrap();
    for _ in 0..DEQUANT_FAULT_WORDS {
        dequant(&mut w, &ok).unwrap();
    }
    dequant(&mut w, &bad).unwrap();
    assert_eq!(fault_ring_waits(&w), 1);
    assert_eq!(held(&w).len(), 1, "the settle drained the 64, not the 65th");
    assert_reports_once(&mut w, "the 65th of 65");
}

#[test]
fn a_failed_dequant_on_a_full_ring_neither_leaks_nor_hides_an_earlier_fault() {
    let mut w = worker();
    let (ok, bad) = (good(&mut w), overflowing(&mut w));
    sync(&mut w).unwrap();
    dequant(&mut w, &bad).unwrap();
    for _ in 1..DEQUANT_FAULT_WORDS {
        dequant(&mut w, &ok).unwrap();
    }
    arm(&mut w, DequantFail::Dispatch);
    dequant(&mut w, &ok).expect_err("the dispatch was armed to fail");
    // Its forced settle drained the 64 first: the fault is held, the ring is whole and empty.
    assert_eq!((free(&w), held(&w).len()), (DEQUANT_FAULT_WORDS, 0));
    assert_ring_whole(&w, "after the failure");
    assert_reports_once(&mut w, "the earlier fault");
}

/// A recycle (`retrying` after a refused allocation) scans only entries before the current command;
/// a dequant whose output allocation is refused once still completes and leaves the ring whole.
#[test]
fn a_refused_output_allocation_recycles_without_losing_a_word() {
    let mut w = worker();
    let t = good(&mut w);
    sync(&mut w).unwrap();
    for _ in 0..8 {
        dequant(&mut w, &t).unwrap();
    }
    assert!(matches!(
        w.command(Cmd::Tune {
            overlap: None,
            ws_limit: None,
            fail_allocs: (0, 1),
        }),
        Ok(Reply::Done)
    ));
    let out = dequant(&mut w, &t).expect("the retry after the recycle succeeds");
    assert_ring_whole(&w, "after a recycled allocation");
    sync(&mut w).unwrap();
    assert_eq!(free(&w), DEQUANT_FAULT_WORDS);
    assert!(values_of(&mut w, out).iter().all(|&v| v == 0x3F00));
}

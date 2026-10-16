//! Adversarial tests for card ft-cab0fdf1 on Metal: `classify_alloc` against
//! seeded variants of tessl's real allocation failures, and
//! `Worker::retrying` under partial and interleaved failures.
//!
//! Tests marked `#[ignore = "BREAK: ..."]` assert the property the card or
//! the doc comments promise and currently fail.

use super::*;
use crate::link::Waits;

/// xorshift64*, seeded: the same sequence on every run.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

const LIMIT_A: &str = "exceeds host/device allocation limit";
const LIMIT_B: &str = "exceeds device limit";

fn edge(rng: &mut Rng) -> u64 {
    match rng.below(8) {
        0 => 0,
        1 => 1,
        2 => u64::from(u32::MAX),
        3 => u64::from(u32::MAX) + 1,
        4 => isize::MAX as u64,
        5 => u64::MAX,
        6 => 256,
        _ => rng.next(),
    }
}

/// Every message shape tessl builds on an allocation failure
/// (runtime.rs `BufferPool::alloc`), with random sizes, classifies as
/// tessl means it: the two permanent texts `Limit`, `newBuffer(key) failed`
/// `Exhausted`.
#[test]
fn tessls_real_message_shapes_classify_for_every_size() {
    let mut rng = Rng(0x0A5_5EED);
    for _ in 0..20_000 {
        let n = edge(&mut rng);
        assert_eq!(
            classify_alloc(&format!(
                "buffer request {n} exceeds host/device allocation limit"
            )),
            AllocFailure::Limit
        );
        assert_eq!(
            classify_alloc(&format!("newBuffer({n}) failed")),
            AllocFailure::Exhausted
        );
    }
    assert_eq!(
        classify_alloc("rounded buffer size exceeds device limit"),
        AllocFailure::Limit
    );
}

/// The permanent class must win whenever its text is present, wherever it
/// sits, so a Limit message can never take the retry path; and a string with
/// both texts is not Exhausted even when it has the exhausted shape.
#[test]
fn a_message_with_the_limit_text_is_never_retried() {
    let mut rng = Rng(0xB0_07);
    let noise = [
        "",
        " ",
        "\n",
        "x",
        "é",
        "日本語",
        "newBuffer(",
        ") failed",
        "\0",
        "🙂",
    ];
    for _ in 0..20_000 {
        let limit = if rng.below(2) == 0 { LIMIT_A } else { LIMIT_B };
        let pick = |rng: &mut Rng| noise[rng.below(noise.len() as u64) as usize];
        let (a, b) = (pick(&mut rng), pick(&mut rng));
        let n = edge(&mut rng);
        let shapes = [
            format!("{a}{limit}{b}"),
            format!("newBuffer({n}) failed {limit}"),
            format!("newBuffer({n}) {limit}) failed"),
            format!("newBuffer({limit}) failed"),
            format!("{limit} newBuffer({n}) failed"),
            format!("newBuffer({n}) failed{a}{limit}) failed"),
        ];
        for s in &shapes {
            assert_eq!(classify_alloc(s), AllocFailure::Limit, "{s:?}");
        }
    }
}

/// `Exhausted` must be reached by every exact `newBuffer(<digits>) failed`
/// and by nothing that lacks the shape. Mutations of the real text that
/// tessl never emits (case, whitespace, a different verb, a missing paren)
/// are not allocation failures and must stay `Other`.
#[test]
fn near_misses_of_the_exhausted_text_are_other() {
    let near = [
        "newbuffer(4096) failed",
        "NewBuffer(4096) failed",
        "newBuffer (4096) failed",
        " newBuffer(4096) failed",
        "newBuffer(4096) failed ",
        "newBuffer(4096) failed.",
        "newBuffer(4096 failed",
        "newBuffer 4096 failed",
        "newBuffer(4096) Failed",
        "newBuffer(4096) fail",
        "xnewBuffer(4096) failed",
        "newBuffer failed",
        "newBuffer: failed",
        "newBuffer(4096)failed",
        "",
        "\0",
        "failed",
        ") failed",
        "newBuffer(",
        "runtime is poisoned after encode/submit failure (newBuffer(1) failed); recreate it",
        "bump arena exhausted: need 4 more bytes (cursor=0, cap=1)",
        "params buffer exhausted (cap 16 bytes)",
        "tensor byte size overflow",
    ];
    for s in near {
        assert_eq!(classify_alloc(s), AllocFailure::Other, "{s:?}");
    }
}

/// A key is a number. A message whose "key" is not one is not tessl's text.
#[test]
#[ignore = "BREAK: classify_alloc takes any text between 'newBuffer(' and ') failed' as Exhausted, e.g. 'newBuffer(not a size) failed'"]
fn the_key_must_be_a_number() {
    for s in [
        "newBuffer(not a size) failed",
        "newBuffer() failed",
        "newBuffer(\u{663}\u{664}) failed",
        "newBuffer(12) x (34) failed",
        "newBuffer(-1) failed",
        "newBuffer(4096)) failed",
    ] {
        assert_eq!(classify_alloc(s), AllocFailure::Other, "{s:?}");
    }
}

/// Seeded random strings never panic the classifier (char boundaries,
/// NULs, lone surrogates cannot occur in `str`, but multi-byte tails can).
#[test]
fn random_text_never_panics_and_only_known_shapes_match() {
    let mut rng = Rng(0xF00D);
    let alphabet: Vec<&str> = vec![
        "n",
        "e",
        "w",
        "B",
        "u",
        "f",
        "r",
        "(",
        ")",
        " ",
        "failed",
        "newBuffer(",
        ") failed",
        "exceeds",
        "device",
        "limit",
        "host/",
        "allocation",
        "é",
        "日",
        "🙂",
        "0",
        "9",
        "\n",
    ];
    for _ in 0..50_000 {
        let len = rng.below(12) as usize;
        let s: String = (0..len)
            .map(|_| alphabet[rng.below(alphabet.len() as u64) as usize])
            .collect();
        let got = classify_alloc(&s);
        let limit = s.contains(LIMIT_A) || s.contains(LIMIT_B);
        let shape = s.starts_with("newBuffer(") && s.ends_with(") failed");
        let want = if limit {
            AllocFailure::Limit
        } else if shape {
            AllocFailure::Exhausted
        } else {
            AllocFailure::Other
        };
        assert_eq!(got, want, "{s:?}");
    }
}

fn worker() -> (Worker, Arc<Waits>) {
    let waits = Arc::new(Waits::default());
    let w = Worker::open(Arc::clone(&waits), 1 << 44).expect("Metal device");
    (w, waits)
}

/// A refusal that runs out of retries leaves the books as it found them:
/// the pool cap restored, nothing counted as allocated, exactly one waited
/// commit (the recycle), and the next allocation works and is counted.
#[test]
fn a_failed_retry_restores_the_pool_cap_and_counters() {
    let (w, waits) = worker();
    let cap = w.rt.memory_info().pool_cache_cap;
    assert!(cap > 0, "this test needs a non-zero pool cap");
    let before = waits.snapshot();
    let mut calls = 0;
    let r: Res<()> = w.retrying(4096, || {
        calls += 1;
        Err("newBuffer(4096) failed".to_string())
    });
    assert!(
        matches!(
            r,
            Err(OjasError::CapacityExceeded {
                requested: 4096,
                ..
            })
        ),
        "{r:?}"
    );
    assert_eq!(calls, 3);
    assert_eq!(w.rt.memory_info().pool_cache_cap, cap, "cap restored");
    assert_eq!(w.book.borrow().alloc_since_wait, 0);
    let after = waits.snapshot();
    assert_eq!(after.recycle, before.recycle + 1, "one recycle commit");
    let ok = w.alloc(4096).expect("a later allocation");
    drop(ok);
    assert_eq!(w.book.borrow().alloc_since_wait, 4096);
}

/// The class can change between attempts. Exhausted then Limit stops at once
/// (no pool trim); Exhausted then another error is that error as `Backend`;
/// the third attempt's failure of another class is `Backend`, not capacity.
#[test]
fn a_different_error_between_attempts_decides_the_outcome() {
    let (w, _waits) = worker();
    let cap = w.rt.memory_info().pool_cache_cap;
    let mut calls = 0;
    let r: Res<()> = w.retrying(8, || {
        calls += 1;
        Err(if calls == 1 {
            "newBuffer(8) failed".to_string()
        } else {
            "rounded buffer size exceeds device limit".to_string()
        })
    });
    assert!(
        matches!(r, Err(OjasError::CapacityExceeded { .. })),
        "{r:?}"
    );
    assert_eq!(calls, 2, "a permanent refusal on the retry ends it");

    calls = 0;
    let r: Res<()> = w.retrying(8, || {
        calls += 1;
        Err(if calls == 1 {
            "newBuffer(8) failed".to_string()
        } else {
            "runtime is poisoned after encode/submit failure; recreate it".to_string()
        })
    });
    assert!(matches!(r, Err(OjasError::Backend { .. })), "{r:?}");
    assert_eq!(calls, 2);

    calls = 0;
    let r: Res<()> = w.retrying(8, || {
        calls += 1;
        Err(if calls < 3 {
            "newBuffer(8) failed".to_string()
        } else {
            "pipeline 'x' not found".to_string()
        })
    });
    assert!(matches!(r, Err(OjasError::Backend { .. })), "{r:?}");
    assert_eq!(calls, 3);
    assert_eq!(w.rt.memory_info().pool_cache_cap, cap);
}

/// Partial failure, as `GdnTrainWorkspace::new` makes it (run alone, with
/// `--test-threads=1`: the device's allocated figure is process-wide, so a
/// sibling test's buffers would move it): buffers allocated
/// before the refused one are dropped. They go back to the pool at the next
/// waited commit, so after a recycle and a pool trim the device holds what
/// it held before.
#[test]
fn partial_allocations_are_released_after_the_final_refusal() {
    let (w, _waits) = worker();
    w.recycle().unwrap();
    w.drop_pool();
    let base = w.rt.current_allocated_bytes();
    let r: Res<()> = w.retrying(3 << 20, || {
        let _a = w.rt.alloc_buffer(1 << 20)?;
        let _b = w.rt.alloc_buffer(2 << 20)?;
        Err("newBuffer(1048576) failed".to_string())
    });
    assert!(
        matches!(r, Err(OjasError::CapacityExceeded { .. })),
        "{r:?}"
    );
    w.recycle().unwrap();
    w.drop_pool();
    assert_eq!(
        w.rt.current_allocated_bytes(),
        base,
        "partial buffers leaked on the device"
    );
}

/// The device's documented bound for what the budget does not charge counts
/// "dropped buffers awaiting a commit" in two buckets: bytes freed
/// (`freed_since_wait`) and bytes allocated since the last wait
/// (`alloc_since_wait`). A partial allocation that is then dropped on error
/// sits in neither: it is device memory awaiting a commit that no counter
/// sees, so the trigger that waits "before a command that would start past
/// the memory cap" undercounts it.
#[test]
#[ignore = "BREAK: buffers dropped by a failed multi-buffer allocation are counted in neither alloc_since_wait nor freed_since_wait (device.rs retrying/gdn_workspace add only on success)"]
fn dropped_partial_buffers_count_toward_a_wait_trigger() {
    let (w, _waits) = worker();
    w.recycle().unwrap();
    let before = {
        let b = w.book.borrow();
        b.alloc_since_wait + b.freed_since_wait
    };
    let _: Res<()> = w.retrying(3 << 20, || {
        let _a = w.rt.alloc_buffer(1 << 20)?;
        let _b = w.rt.alloc_buffer(2 << 20)?;
        Err("newBuffer(1048576) failed".to_string())
    });
    let after = {
        let b = w.book.borrow();
        b.alloc_since_wait + b.freed_since_wait
    };
    assert!(
        after >= before + (3 << 20),
        "3 MiB sat on the device unaccounted: {before} -> {after}"
    );
}

/// Tessl's real permanent refusals through `Worker::alloc` and
/// `fresh_from`, at sizes around every threshold the allocator checks: the
/// result is `CapacityExceeded` whose `requested` is the caller's figure,
/// with no waited commit and nothing counted. (Sizes at or under the
/// device's own buffer limit are not tried: they could succeed and hold
/// real memory.)
#[test]
fn permanent_refusals_report_the_callers_size_and_cost_nothing() {
    let (w, waits) = worker();
    let before = waits.snapshot();
    let big = [
        (isize::MAX as usize) + 1,
        usize::MAX - 1,
        usize::MAX,
        1 << 62,
        1 << 48,
    ];
    for bytes in big {
        let r = w.alloc(bytes);
        assert!(
            matches!(&r, Err(OjasError::CapacityExceeded { requested, cap, .. })
                if *requested == bytes as u64 && *cap > 0),
            "{bytes}: {:?}",
            r.map(|_| ())
        );
    }
    assert_eq!(waits.snapshot(), before);
    assert_eq!(w.book.borrow().alloc_since_wait, 0);
}

/// A poisoned runtime must not look like capacity: tessl's allocator
/// refuses with its poison text, which is neither an allocation failure
/// nor retried.
#[test]
fn a_poisoned_runtime_is_backend_not_capacity() {
    let (w, waits) = worker();
    let before = waits.snapshot();
    let r: Res<()> = w.retrying(8, || {
        Err(
            "runtime is poisoned after encode/submit failure (newBuffer(8) failed); recreate it"
                .to_string(),
        )
    });
    assert!(matches!(r, Err(OjasError::Backend { .. })), "{r:?}");
    assert_eq!(waits.snapshot(), before, "no commit for a non-allocation");
}

//! Resume holds one host copy of each tensor, measured on the heap itself.
//!
//! The budget bound in `tests/checkpoint.rs` sees only budget-charged
//! allocations; a plain `Vec` beside each tensor would not show there. This
//! binary counts every heap byte through its global allocator, so it holds a
//! single test: nothing else allocates while it measures.

mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use common::{token_bin, TempDir};
use ojas_core::{Budget, Numerics};
use ojas_cpu::{CosineSchedule, CpuBackend, LrSchedule};
use ojas_data::TokenBin;
use ojas_model::{init_params, ModelSpec, TrainConfig, Trainer};

/// The system allocator, counting live bytes and their peak.
struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn grew(by: usize) {
    let now = LIVE.fetch_add(by, Ordering::SeqCst) + by;
    PEAK.fetch_max(now, Ordering::SeqCst);
}

// SAFETY: every call forwards to `System` with the caller's arguments; the
// counters do not affect what is allocated.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarded unchanged.
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            grew(layout.size());
        }
        p
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarded unchanged.
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            grew(layout.size());
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarded unchanged.
        unsafe { System.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size(), Ordering::SeqCst);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: forwarded unchanged.
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            LIVE.fetch_sub(layout.size(), Ordering::SeqCst);
            grew(new_size);
        }
        p
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

fn exact() -> CpuBackend {
    CpuBackend::new(Budget::new(1 << 30)).with_numerics(Numerics::Exact)
}

fn config() -> TrainConfig {
    let schedule = LrSchedule::Cosine(CosineSchedule::new(2, 40).unwrap());
    TrainConfig::nanolab(2, 32, 2, 1337, schedule)
}

#[test]
fn resume_peaks_below_one_extra_embedding() {
    // Vocab 1024 makes the embedding 256 KiB, four of the loader's
    // 64 KiB read pieces (`ojas_core::LE_READ_CHUNK_BYTES`). With the tiny
    // spec's 64 KiB embedding, one piece and a whole second copy are the
    // same size and the bound below could not tell them apart.
    let spec = ModelSpec {
        vocab: 1024,
        ..ModelSpec::tiny()
    };
    assert!(spec.vocab * spec.n_embd * 4 >= 4 * ojas_core::LE_READ_CHUNK_BYTES);
    let dir = TempDir::new();
    let (tmp, bin) = token_bin(20_000, 256);
    let params = init_params(&spec, 5, &Budget::new(1 << 30)).unwrap();
    let mut t = Trainer::new(exact(), spec, &params, bin, config()).unwrap();
    t.step().unwrap();
    t.save(&dir.ckpt()).unwrap();
    drop(t);
    drop(params);

    // The largest tensors (the embedding and its two AdamW moments).
    let embedding = spec.vocab * spec.n_embd * 4;
    let backend = exact();
    let bin = TokenBin::open_headerless(&tmp.path).unwrap();
    let cfg = config();
    let base = LIVE.load(Ordering::SeqCst);
    PEAK.store(base, Ordering::SeqCst);
    let resumed = Trainer::resume_from(backend, &dir.ckpt(), bin, cfg).unwrap();
    let peak = PEAK.load(Ordering::SeqCst) - base;
    let kept = LIVE.load(Ordering::SeqCst) - base;
    let transient = peak - kept;
    eprintln!(
        "resume: kept {kept} bytes, peak {peak}, transient {transient}, embedding {embedding}"
    );
    // A second host copy of any embedding-sized tensor would add at least
    // `embedding` bytes; everything else resume allocates and frees (file
    // headers, the state file, names) is far smaller.
    assert!(
        transient < embedding / 2,
        "transient {transient} bytes against an embedding of {embedding}"
    );
    assert_eq!(resumed.step_count(), 1);
}

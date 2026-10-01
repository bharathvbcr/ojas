//! Red-team gate: the bytes a linear op really allocates on the Rust heap
//! stay within what it charges to its `Budget`.
//!
//! A counting global allocator (this test binary only) records the heap
//! high-water mark while one op runs; the charged peak is measured the same
//! way as in `redteam_linear_budget.rs` (smallest room the op accepts). A
//! new buffer that a copy-cutting change forgets to charge shows up here even
//! when every bit is right.
//!
//! Fixed gap, asserted by the strict test: `linear_forward` used to drop its
//! `room_for` hold when it returned, then `alloc_f32` (backend.rs) copied the
//! still-live output `Vec` into a new tensor, so for a moment the output was
//! on the heap twice while the budget held it once. The output's charge now
//! travels with it (validate.rs `F32Out`) and is released by `alloc_out`
//! only after the tensor exists and the buffer is freed. The looser gate
//! below still allows one forward output of slack and is kept as a floor.

mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use common::SplitMix64;
use ojas_core::{Backend, Budget, Numerics, OjasError, Tensor};
use ojas_cpu::CpuBackend;

struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn grow(bytes: usize) {
    let now = LIVE.fetch_add(bytes, Ordering::SeqCst) + bytes;
    PEAK.fetch_max(now, Ordering::SeqCst);
}

fn shrink(bytes: usize) {
    LIVE.fetch_sub(bytes, Ordering::SeqCst);
}

// SAFETY: every method forwards to `System` with the caller's arguments
// unchanged and only updates two counters on success.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            grow(layout.size());
        }
        p
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            grow(layout.size());
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        shrink(layout.size());
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            if new_size >= layout.size() {
                grow(new_size - layout.size());
            } else {
                shrink(layout.size() - new_size);
            }
        }
        p
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// Heap measurements need the whole process to themselves.
static SERIAL: Mutex<()> = Mutex::new(());

const HUGE: u64 = 1 << 40;
/// Pool batch bookkeeping, result slots, shape vectors, error strings.
const METADATA_SLACK: usize = 64 << 10;

type Op<'a> = &'a dyn Fn(&CpuBackend) -> Result<Vec<Tensor>, OjasError>;

fn accepts(be: &CpuBackend, room: u64, op: Op<'_>) -> bool {
    let filler = be.budget().try_reserve(HUGE - room).unwrap();
    let ok = match op(be) {
        Ok(_) => true,
        Err(OjasError::CapacityExceeded { .. }) => false,
        Err(err) => panic!("unexpected {err:?}"),
    };
    drop(filler);
    assert_eq!(be.budget().live_bytes().unwrap(), 0);
    ok
}

/// Smallest room `op` accepts.
fn charged_peak(be: &CpuBackend, op: Op<'_>) -> u64 {
    let mut hi = 1u64 << 20;
    while !accepts(be, hi, op) {
        hi *= 2;
        assert!(hi < HUGE, "op refused with {hi} bytes of room");
    }
    let mut lo = 0u64;
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if accepts(be, mid, op) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    hi
}

/// Heap high-water growth while `op` runs, outputs included.
fn heap_peak(be: &CpuBackend, op: Op<'_>) -> usize {
    let base = LIVE.load(Ordering::SeqCst);
    PEAK.store(base, Ordering::SeqCst);
    let out = op(be).unwrap();
    let peak = PEAK.load(Ordering::SeqCst) - base;
    drop(out);
    peak
}

struct Measured {
    what: String,
    heap: usize,
    charged: u64,
    outputs: u64,
    /// Heap bytes the default gate tolerates above the charged peak: one
    /// output for forward (the known double-hold), nothing for backward.
    known_gap: u64,
}

fn measure() -> Vec<Measured> {
    let shapes = [
        (1024usize, 1usize, 1024usize), // output far larger than inputs + scratch
        (64, 96, 128),
        (73, 257, 257),
        (1, 768, 768),
        (512, 768, 768),
        (128, 128, 128),
    ];
    let mut rng = SplitMix64(0x4ea9);
    let mut out = Vec::new();
    for &(rows, kin, nout) in &shapes {
        let ib = Budget::new(u64::MAX);
        let x = Tensor::from_f32(&rng.vec(rows * kin, 1.0), &[rows, kin], &ib).unwrap();
        let w = Tensor::from_f32(&rng.vec(nout * kin, 1.0), &[nout, kin], &ib).unwrap();
        let g = Tensor::from_f32(&rng.vec(rows * nout, 1.0), &[rows, nout], &ib).unwrap();
        for threads in [1usize, 7] {
            for numerics in [Numerics::Exact, Numerics::Fast] {
                let be = CpuBackend::with_threads(Budget::new(HUGE), threads)
                    .unwrap()
                    .with_numerics(numerics);
                let fwd = |be: &CpuBackend| be.linear_forward(&x, &w).map(|y| vec![y]);
                let bwd = |be: &CpuBackend| be.linear_backward(&x, &w, &g).map(|(a, b)| vec![a, b]);
                for (dir, op, outputs) in [
                    ("forward", &fwd as Op<'_>, rows * nout),
                    ("backward", &bwd as Op<'_>, rows * kin + nout * kin),
                ] {
                    // Warm the pool so worker spawns are not counted.
                    drop(op(&be).unwrap());
                    let charged = charged_peak(&be, op);
                    let heap = heap_peak(&be, op);
                    let outputs = 4 * outputs as u64;
                    out.push(Measured {
                        what: format!("{dir} {rows}x{kin}x{nout} threads {threads} {numerics:?}"),
                        heap,
                        charged,
                        outputs,
                        known_gap: if dir == "forward" { outputs } else { 0 },
                    });
                }
            }
        }
    }
    out
}

/// Heap peak <= charged peak + slack, plus one output for forward only (the
/// known double-hold). Backward has no known gap and gets no extra room.
#[test]
fn heap_peak_stays_within_the_charged_peak_plus_the_known_output_double_hold() {
    let _serial = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    let mut failures = Vec::new();
    for m in measure() {
        println!(
            "{}: heap peak {} bytes, charged peak {} bytes, outputs {} bytes",
            m.what, m.heap, m.charged, m.outputs
        );
        let allowed = m.charged as usize + m.known_gap as usize + METADATA_SLACK;
        if m.heap > allowed {
            failures.push(format!(
                "{}: heap {} > charged {} + known gap {} + slack {METADATA_SLACK}",
                m.what, m.heap, m.charged, m.known_gap
            ));
        }
    }
    assert!(failures.is_empty(), "uncharged heap growth:\n{}", failures.join("\n"));
}

/// Heap peak <= charged peak + slack. Failed on forward shapes whose output
/// exceeds inputs + scratch until the output's charge was held through the
/// tensor copy (see the module comment).
#[test]
fn heap_peak_stays_within_the_charged_peak() {
    let _serial = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    let mut failures = Vec::new();
    for m in measure() {
        if m.heap > m.charged as usize + METADATA_SLACK {
            failures.push(format!(
                "{}: heap {} > charged {} (+{} over)",
                m.what,
                m.heap,
                m.charged,
                m.heap - m.charged as usize
            ));
        }
    }
    assert!(failures.is_empty(), "uncharged heap growth:\n{}", failures.join("\n"));
}

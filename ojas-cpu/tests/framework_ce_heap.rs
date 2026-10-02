//! Heap gate for `linear_cross_entropy_mean`: the op never allocates the
//! `[N, V]` logits, and every byte it allocates is charged.
//!
//! A counting global allocator (this test binary only) records the Rust heap
//! high-water mark while one call runs, as in `redteam_linear_heap.rs`.
//! Accelerate's own buffers are not Rust heap and are not counted.

mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use common::SplitMix64;
use ojas_core::{Backend, Budget, CeChunk, Numerics, OjasError, Tensor};
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

struct Inputs {
    x: Tensor,
    w: Tensor,
    t: Tensor,
}

fn inputs(n: usize, d: usize, v: usize, seed: u64) -> Inputs {
    let mut rng = SplitMix64(seed);
    let ib = Budget::new(u64::MAX);
    let targets: Vec<u32> = (0..n)
        .map(|i| {
            if i % 5 == 2 {
                u32::MAX
            } else {
                rng.below(v) as u32
            }
        })
        .collect();
    Inputs {
        x: Tensor::from_f32(&rng.vec(n * d, 1.0), &[n, d], &ib).unwrap(),
        w: Tensor::from_f32(&rng.vec(v * d, 0.5), &[v, d], &ib).unwrap(),
        t: Tensor::from_u32(&targets, &[n], &ib).unwrap(),
    }
}

fn run(
    be: &CpuBackend,
    i: &Inputs,
    chunk: CeChunk,
    want_grad: bool,
) -> Result<Vec<Tensor>, OjasError> {
    let out = be.linear_cross_entropy_mean(&i.x, &i.w, &i.t, Some(u32::MAX), chunk, want_grad)?;
    let mut all = vec![out.loss];
    all.extend(out.grad_input);
    all.extend(out.grad_weight);
    Ok(all)
}

/// Heap high-water growth while one call runs, outputs included.
fn heap_peak(be: &CpuBackend, i: &Inputs, chunk: CeChunk, want_grad: bool) -> usize {
    let base = LIVE.load(Ordering::SeqCst);
    PEAK.store(base, Ordering::SeqCst);
    let out = run(be, i, chunk, want_grad).unwrap();
    let peak = PEAK.load(Ordering::SeqCst) - base;
    drop(out);
    peak
}

fn accepts(be: &CpuBackend, room: u64, i: &Inputs, chunk: CeChunk, want_grad: bool) -> bool {
    let filler = be.budget().try_reserve(HUGE - room).unwrap();
    let ok = match run(be, i, chunk, want_grad) {
        Ok(_) => true,
        Err(OjasError::CapacityExceeded { .. }) => false,
        Err(err) => panic!("unexpected {err:?}"),
    };
    drop(filler);
    assert_eq!(be.budget().live_bytes().unwrap(), 0);
    ok
}

/// Smallest room the call accepts.
fn charged_peak(be: &CpuBackend, i: &Inputs, chunk: CeChunk, want_grad: bool) -> u64 {
    let mut hi = 1u64 << 16;
    while !accepts(be, hi, i, chunk, want_grad) {
        hi *= 2;
        assert!(hi < HUGE, "refused with {hi} bytes of room");
    }
    let mut lo = 0u64;
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if accepts(be, mid, i, chunk, want_grad) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    hi
}

/// At N=4096, V=50304 the logits alone are 824 MB. The fused call's whole
/// heap peak (input copies, both gradients and their tensors, one tile) stays
/// a small fraction of that. `d` is 64 so the unavoidable `(N + V) * d`
/// terms do not swamp the comparison.
#[test]
fn heap_peak_is_far_below_the_logits() {
    let _serial = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    let (n, d, v) = (4096usize, 64usize, 50304usize);
    let logits = n * v * 4;
    let i = inputs(n, d, v, 0x4ea1);
    let chunk = CeChunk {
        rows: 512,
        cols: 8192,
    };
    let be = CpuBackend::with_threads(Budget::new(HUGE), 4).unwrap();
    drop(run(&be, &i, chunk, true).unwrap());
    let heap = heap_peak(&be, &i, chunk, true);
    let tile = 512 * 8192 * 4;
    let unavoidable = 2 * (n + v) * d * 4;
    println!(
        "N {n} V {v} d {d}: heap peak {heap} bytes; logits N*V*4 {logits} bytes ({:.1}x the peak); \
         one tile {tile} bytes; input copies + gradients {unavoidable} bytes",
        logits as f64 / heap as f64
    );
    assert!(
        heap * 8 < logits,
        "heap peak {heap} is not far below {logits}"
    );
}

/// Heap peak <= charged peak + slack, over shapes whose chunks do not divide
/// N or V, both numerics, one and seven threads, with and without gradients.
#[test]
fn heap_peak_stays_within_the_charged_peak() {
    let _serial = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    let mut failures = Vec::new();
    for &(n, d, v, rows, cols) in &[
        (300usize, 64usize, 5000usize, 128usize, 1024usize),
        (97, 33, 1031, 40, 300),
        (512, 128, 4096, 512, 4096),
        (1, 16, 777, 1, 100),
    ] {
        let i = inputs(n, d, v, 0x4ea2 ^ n as u64);
        let chunk = CeChunk { rows, cols };
        for threads in [1usize, 7] {
            for numerics in [Numerics::Exact, Numerics::Fast] {
                for want_grad in [false, true] {
                    let be = CpuBackend::with_threads(Budget::new(HUGE), threads)
                        .unwrap()
                        .with_numerics(numerics);
                    drop(run(&be, &i, chunk, want_grad).unwrap());
                    let charged = charged_peak(&be, &i, chunk, want_grad);
                    let heap = heap_peak(&be, &i, chunk, want_grad);
                    let what = format!(
                        "n{n} d{d} v{v} chunk {rows}x{cols} threads {threads} {numerics:?} grad {want_grad}"
                    );
                    println!("{what}: heap peak {heap} bytes, charged peak {charged} bytes");
                    if heap > charged as usize + METADATA_SLACK {
                        failures.push(format!("{what}: heap {heap} > charged {charged}"));
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "uncharged heap growth:\n{}",
        failures.join("\n")
    );
}

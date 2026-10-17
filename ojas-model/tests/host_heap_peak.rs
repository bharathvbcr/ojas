//! Host heap peak of `load_hf` and `LoraLinear::merge`, against what each
//! charges to its `Budget`.
//!
//! Each test holds `SERIAL`, so no other test allocates while a peak is measured. The
//! counting allocator forwards to `System`; it is test-only.

mod qwen35_common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use ojas_core::{Budget, DType, Numerics, Tensor, LE_READ_CHUNK_BYTES};
use ojas_cpu::CpuBackend;
use ojas_io::SafeTensors;
use ojas_model::lora::{LoraConfig, LoraLinear};
use ojas_model::qwen35::load_hf;
use qwen35_common::{dir, spec, PREFIX};

struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every call forwards to `System` with the caller's arguments, so the
// `GlobalAlloc` contract is `System`'s. The counters are plain atomics.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            let now = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(now, Ordering::Relaxed);
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Heap above the starting point that `f` reached at its peak.
fn heap_peak<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let base = LIVE.load(Ordering::Relaxed);
    PEAK.store(base, Ordering::Relaxed);
    let out = f();
    (out, PEAK.load(Ordering::Relaxed) - base)
}

const SLACK: usize = 128 << 10;

/// The tiny fixture (vocab 8320) loads within its charged bytes, one read
/// chunk and the largest split part of host heap.
fn load_hf_peak() {
    let s = spec();
    let model = SafeTensors::open(&dir().join("model.safetensors")).unwrap();
    for (linears, table) in [(DType::F32, DType::F32), (DType::Bf16, DType::Bf16)] {
        let budget = Budget::new(1 << 30);
        let (params, heap) = heap_peak(|| load_hf(&s, &model, PREFIX, linears, table, &budget));
        let params = params.unwrap();
        let charged = budget.live_bytes().unwrap() as usize;
        let largest_part = params
            .values()
            .into_iter()
            .skip(1)
            .map(|t| t.num_elements().unwrap() * t.dtype().size())
            .max()
            .unwrap();
        let bound = charged + LE_READ_CHUNK_BYTES + largest_part;
        eprintln!(
            "load_hf {linears:?}/{table:?}: heap peak {heap}, charged {charged}, bound {bound}"
        );
        assert!(
            heap <= bound,
            "heap peak {heap} > bound {bound} (charged {charged})"
        );
        assert!(
            heap >= charged,
            "heap peak {heap} below the charged {charged}"
        );
        drop(params);
    }
}

/// A merge holds the delta and the merged weight, both charged: its heap
/// peak is what the budget saw, not three more weights.
fn merge_peak() {
    let budget = Budget::new(1 << 30);
    let backend = CpuBackend::new(budget.clone()).with_numerics(Numerics::Exact);
    let (out, inn, rank) = (768usize, 512usize, 4usize);
    let fill =
        |n: usize, k: f32| -> Vec<f32> { (0..n).map(|i| ((i % 97) as f32 - 48.0) * k).collect() };
    let w = Tensor::from_f32(&fill(out * inn, 0.01), &[out, inn], &budget).unwrap();
    let a = Tensor::from_f32(&fill(rank * inn, 0.02), &[rank, inn], &budget).unwrap();
    let b = Tensor::from_f32(&fill(out * rank, 0.03), &[out, rank], &budget).unwrap();
    let mut lin = LoraLinear::new(LoraConfig::new(rank, 8.0, 0.0).unwrap(), w, a, b).unwrap();
    let before = budget.live_bytes().unwrap();
    budget.reset_peak();
    let (merged, heap) = heap_peak(|| lin.merge(&backend));
    merged.unwrap();
    let charged = (budget.peak_bytes() - before) as usize;
    let weight = out * inn * 4;
    eprintln!("merge: heap peak {heap}, charged peak {charged}, one weight {weight}");
    assert!(
        heap <= charged + SLACK,
        "heap peak {heap} > charged peak {charged} + {SLACK}: the merge holds uncharged weight-sized copies"
    );
}

static SERIAL: Mutex<()> = Mutex::new(());

#[test]
fn a_merge_holds_only_charged_weight_sized_copies() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    merge_peak();
}

#[test]
fn load_hf_stays_within_its_charge_one_chunk_and_the_largest_part() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    load_hf_peak();
}

//! An upload never holds a second full host copy of the tensor. Before,
//! `upload` encoded the whole window with `Tensor::to_ne_bytes` (outside
//! the budget) and then copied it into the device mapping; now the window
//! is encoded into the mapping a bounded piece at a time. A counting global
//! allocator in this test binary measures the host heap's peak.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use ojas_core::{Backend, Budget, Tensor};

struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every call forwards to the system allocator unchanged; the
// counters are bookkeeping only.
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

#[test]
fn a_large_upload_makes_no_second_host_copy() {
    let backend = ojas_metal::MetalBackend::new(Budget::new(1 << 30)).expect("Metal device");
    let n = 16 << 20; // 64 MiB of f32
    let host = Tensor::from_f32(&vec![0.25f32; n], &[n], &Budget::new(1 << 30)).unwrap();
    // Warm the device path (pipelines, pools) on a small upload first.
    drop(backend.upload(&Tensor::from_f32(&[1.0; 64], &[64], &Budget::new(1 << 20)).unwrap()));
    backend.sync().unwrap();
    let base = LIVE.load(Ordering::Relaxed);
    PEAK.store(base, Ordering::Relaxed);
    let dev = backend.upload(&host).unwrap();
    backend.sync().unwrap();
    let grew = PEAK.load(Ordering::Relaxed) - base;
    assert!(
        grew < 8 << 20,
        "the upload peaked {grew} bytes over the heap it started with (a copy is {} bytes)",
        n * 4
    );
    let back = backend.download(&dev).unwrap();
    assert_eq!(back.to_f32_vec().unwrap()[n - 1], 0.25);
}

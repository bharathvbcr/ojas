//! Peak heap while streaming a large tensor through `SafeTensorsWriter`.
//!
//! This file holds one test so that no other test allocates while the peak is
//! measured. The counting allocator forwards to `System`; it is test-only,
//! and the library itself stays `forbid(unsafe_code)`.

use ojas_io::{SafeTensorsWriter, StDtype, TensorSpec};
use std::alloc::{GlobalAlloc, Layout, System};
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};

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

/// Discards bytes, keeping a count.
struct Sink(u64);

impl Write for Sink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0 += buf.len() as u64;
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn streaming_a_gibibyte_keeps_peak_heap_under_a_mebibyte() {
    const TOTAL: u64 = 1 << 30;
    const CHUNK: usize = 64 << 10;
    let shape = [TOTAL / 2];
    let specs = [TensorSpec {
        name: "embed",
        dtype: StDtype::BF16,
        shape: &shape,
    }];
    let chunk = vec![0x5Au8; CHUNK];
    let base = LIVE.load(Ordering::Relaxed);
    PEAK.store(base, Ordering::Relaxed);
    let mut w = SafeTensorsWriter::new(Sink(0), &specs, &[("ojas.step", "7")]).unwrap();
    let mut sent = 0u64;
    while sent < TOTAL {
        w.write("embed", &chunk).unwrap();
        sent += CHUNK as u64;
    }
    let file_len = w.file_len();
    let sink = w.finish().unwrap();
    let peak = PEAK.load(Ordering::Relaxed) - base;
    assert_eq!(sink.0, file_len);
    assert!(sink.0 > TOTAL);
    assert!(
        peak < 1 << 20,
        "peak heap above the starting point was {peak} bytes for a {TOTAL}-byte tensor"
    );
}

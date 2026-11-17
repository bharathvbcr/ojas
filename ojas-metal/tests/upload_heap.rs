//! An upload never holds a second full host copy of the tensor. Before,
//! `upload` encoded the whole window with `Tensor::to_ne_bytes` (outside
//! the budget) and then copied it into the device mapping; now the window
//! is encoded into the mapping a bounded piece at a time. A counting global
//! allocator in this test binary measures the host heap's peak.
#![cfg(target_os = "macos")]

use ojas_core::{BackendCore, Budget, Tensor};
use ojas_test_support::PeakAlloc;

#[global_allocator]
static ALLOC: PeakAlloc = PeakAlloc::new();

#[test]
fn a_large_upload_makes_no_second_host_copy() {
    let backend = ojas_metal::MetalBackend::new(Budget::new(1 << 30)).expect("Metal device");
    let n = 16 << 20; // 64 MiB of f32
    let host = Tensor::from_f32(&vec![0.25f32; n], &[n], &Budget::new(1 << 30)).unwrap();
    // Warm the device path (pipelines, pools) on a small upload first.
    drop(backend.upload(&Tensor::from_f32(&[1.0; 64], &[64], &Budget::new(1 << 20)).unwrap()));
    backend.sync().unwrap();
    let (dev, grew) = ALLOC.measure(|| {
        let dev = backend.upload(&host).unwrap();
        backend.sync().unwrap();
        dev
    });
    assert!(
        grew < 8 << 20,
        "the upload peaked {grew} bytes over the heap it started with (a copy is {} bytes)",
        n * 4
    );
    let back = backend.download(&dev).unwrap();
    assert_eq!(back.to_f32_vec().unwrap()[n - 1], 0.25);
}

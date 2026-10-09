//! `CudaBackend` on the device: one budget for every device byte, and
//! uploads that land exactly. Every test here needs an sm_90 NVIDIA GPU, so
//! each is `#[ignore]`: on the Mac they compile and are reported NOT RUN. On
//! the box (README, "Device tests on the box"):
//! `LD_LIBRARY_PATH=<cublas>:<nvrtc> ./device_backend-<hash> --ignored --test-threads=1`.
#![cfg(feature = "cuda")]

use ojas_core::{Backend, Budget, DType, OjasError, Tensor};
use ojas_cuda::runtime::CUBLAS_WORKSPACE_BYTES;
use ojas_cuda::{CudaBackend, CudaDeviceBuffer, CudaError};

const MIB: u64 = 1 << 20;

fn open(budget: &Budget) -> CudaBackend {
    CudaBackend::open(budget.clone()).unwrap_or_else(|e| panic!("CudaBackend::open: {e}"))
}

/// Before the fix `upload` charged the backend's `Budget` while the
/// runtime's kernel buffers charged a separate `AllocBudget`, both sized to
/// the same cap, so together they could hold twice it: with a 96 MiB cap, a
/// 40 MiB tensor and then a 40 MiB kernel buffer both succeeded (the runtime
/// had counted only its 32 MiB workspace). Now the kernel buffer is refused,
/// and in the other order the upload is, as `CapacityExceeded`.
#[test]
#[ignore = "needs an sm_90 NVIDIA GPU; run on the box with --ignored"]
fn backend_tensors_and_kernel_buffers_share_one_cap() {
    // Uses only API the pre-fix crate also has, so the same file shows the
    // failure there (the kernel buffer below is granted) and the fix here.
    let budget = Budget::new(96 * MIB);
    let backend = open(&budget);
    let rt = backend.runtime();
    let workspace = CUBLAS_WORKSPACE_BYTES as u64;

    let host = Budget::new(64 * MIB);
    let x = Tensor::zeros(&[10 << 20], DType::F32, &host).unwrap();
    let on_device = backend.upload(&x).expect("40 MiB fits 96 MiB holding 32");
    let err = rt
        .alloc_zeros::<f32>(10 << 20, "kernel buffer")
        .expect_err("40 MiB more must not fit a 96 MiB cap holding 72 MiB");
    assert!(matches!(err, CudaError::Capacity { .. }), "{err}");
    assert_eq!(
        budget.live_bytes().unwrap(),
        workspace + 40 * MIB,
        "the caller's budget holds the cuBLAS workspace and the tensor"
    );

    drop(on_device);
    assert_eq!(budget.live_bytes().unwrap(), workspace);
    let kernel = rt
        .alloc_zeros::<f32>(10 << 20, "kernel buffer")
        .expect("40 MiB fits once the tensor is gone");
    assert_eq!(budget.live_bytes().unwrap(), workspace + 40 * MIB);
    match backend.upload(&x) {
        Err(OjasError::CapacityExceeded {
            requested,
            cap,
            live,
        }) => assert_eq!(
            (requested, cap, live),
            (40 * MIB, 96 * MIB, workspace + 40 * MIB)
        ),
        other => panic!("upload past the shared cap: {other:?}"),
    }
    drop(kernel);
    assert_eq!(budget.live_bytes().unwrap(), workspace);
}

/// A session budget is a child of the process ceiling: the backend's device
/// bytes charge the parent too.
#[test]
#[ignore = "needs an sm_90 NVIDIA GPU; run on the box with --ignored"]
fn a_child_budget_charges_its_parent() {
    let root = Budget::new(256 * MIB);
    let backend = open(&root.child(128 * MIB));
    let host = Budget::new(8 * MIB);
    let x = Tensor::zeros(&[1 << 20], DType::F32, &host).unwrap();
    let _on_device = backend.upload(&x).unwrap();
    assert_eq!(
        root.live_bytes().unwrap(),
        CUBLAS_WORKSPACE_BYTES as u64 + 4 * MIB
    );
}

/// Uploads no longer zero the allocation before the copy, so every byte of
/// the result must come from the source. Non-zero patterns of every dtype
/// read back exactly; only a U32 upload keeps a host shadow.
#[test]
#[ignore = "needs an sm_90 NVIDIA GPU; run on the box with --ignored"]
fn uploads_write_every_byte_and_only_u32_keeps_a_shadow() {
    let budget = Budget::new(256 * MIB);
    let backend = open(&budget);
    let host = Budget::new(64 * MIB);
    // Odd lengths, so no allocation is a multiple of any tile.
    let n = 100_003usize;
    let f: Vec<f32> = (0..n).map(|i| (i as f32).mul_add(0.5, -7.25)).collect();
    let u: Vec<u32> = (0..n as u32)
        .map(|i| i.wrapping_mul(2_654_435_761))
        .collect();
    let b: Vec<u16> = (0..n).map(|i| (i as u16) | 0x8001).collect();
    let tensors = [
        Tensor::from_f32(&f, &[n], &host).unwrap(),
        Tensor::from_u32(&u, &[n], &host).unwrap(),
        Tensor::from_bf16_bits(&b, &[n], &host).unwrap(),
        // F16 takes the staged path (no borrowed accessor).
        Tensor::zeros(&[n], DType::F16, &host).unwrap(),
    ];
    for t in &tensors {
        let on_device = backend.upload(t).unwrap();
        let buf = on_device
            .device_buffer()
            .and_then(|b| b.as_any().downcast_ref::<CudaDeviceBuffer>())
            .expect("a CUDA device buffer");
        match t.dtype() {
            DType::U32 => assert_eq!(buf.shadow_u32().map(|s| s.to_vec()), Some(u.clone())),
            other => assert!(buf.shadow_u32().is_none(), "{other:?} kept a shadow"),
        }
        let back = backend.download(&on_device).unwrap();
        assert_eq!(
            back.to_ne_bytes().unwrap(),
            t.to_ne_bytes().unwrap(),
            "{:?}",
            t.dtype()
        );
    }
}

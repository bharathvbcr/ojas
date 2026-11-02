//! Adversarial Verification Test for Milestone M3: Zero-Copy UMA Adoption & Budget Ledgers
#![cfg(all(target_os = "macos", feature = "metal"))]
#![allow(clippy::needless_range_loop)]

use ojas_core::{
    Backend, BackendId, Budget, DType, DeviceBuffer, MappedBuffer, OjasError, Tensor, Timeline,
};
use std::sync::Arc;
use tessl::shared_buf::SharedAlloc;

#[test]
fn upload_mapped_shared_buf_is_strictly_zero_copy_and_matches_bits() {
    let budget = Budget::new(64 << 20);
    let backend = ojas_metal::MetalBackend::new(budget.clone()).expect("Metal device");

    let n = 256 * 1024; // 1 MiB of f32 (262,144 elements)
    let nbytes = n * 4;

    // 1. Create a tessl SharedBuf directly from the backend's runtime via with_runtime
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    backend
        .with_runtime(move |rt| {
            let alloc = SharedAlloc::new(rt, nbytes).map_err(|e| OjasError::Backend {
                id: BackendId::Metal,
                detail: e,
            })?;
            tx.send(alloc.into_shared_buf()).unwrap();
            Ok(())
        })
        .expect("with_runtime allocation");
    let shared = rx.recv().expect("receive shared");

    // Fill with known deterministic pattern via host write lease
    {
        let mut w = shared.write().expect("exclusive write lease");
        let floats = w.as_f32_mut().expect("f32 slice");
        for (i, val) in floats.iter_mut().enumerate() {
            *val = (i as f32) * 1.5 + 0.25;
        }
    }

    // 2. Wrap into MappedBuffer
    let mapped_tensor = Tensor::from_mapped(Arc::new(shared), &[256, 1024], DType::F32, &budget)
        .expect("Tensor::from_mapped");

    // 3. Measure memory allocated by backend before upload
    let mem_before = backend.memory().unwrap().allocated;

    // 4. Upload the mapped tensor to backend
    let dev_tensor = backend
        .upload(&mapped_tensor)
        .expect("upload mapped tensor");
    backend.sync().expect("sync");

    // 5. Measure memory allocated by backend after upload
    let mem_after = backend.memory().unwrap().allocated;

    // VERIFICATION: Worker::upload adopted the GpuBuffer directly!
    // It must NOT have called alloc(len), so mem_after == mem_before.
    assert_eq!(
        mem_after, mem_before,
        "Zero-Copy Invariant Violated: backend allocated new device memory ({mem_after} > {mem_before}) instead of adopting existing GpuBuffer!"
    );

    // 6. Verify bits read back from the device tensor
    let downloaded = backend.download(&dev_tensor).expect("download");
    let dl_floats = downloaded.to_f32_vec().expect("f32 vec");
    assert_eq!(dl_floats.len(), n);
    for i in 0..n {
        let expected = (i as f32) * 1.5 + 0.25;
        assert_eq!(
            dl_floats[i].to_bits(),
            expected.to_bits(),
            "Bit mismatch at index {i}: downloaded {} != expected {}",
            dl_floats[i],
            expected
        );
    }

    // 7. Execute a compute op on the adopted tensor (e.g., scale)
    let scaled = backend
        .scale_forward(&dev_tensor, 2.0)
        .expect("scale_forward");
    backend.sync().expect("sync");
    let scaled_dl = backend
        .download(&scaled)
        .expect("download scaled")
        .to_f32_vec()
        .expect("f32 vec");
    for i in 0..100 {
        let expected = ((i as f32) * 1.5 + 0.25) * 2.0;
        assert_eq!(scaled_dl[i].to_bits(), expected.to_bits());
    }
}

// Mock non-Metal MappedBuffer to verify fail-closed placement check
#[derive(Debug)]
struct MockCpuMappedBuffer {
    data: Vec<u8>,
}

impl DeviceBuffer for MockCpuMappedBuffer {
    fn backend(&self) -> BackendId {
        BackendId::Cpu
    }
    fn byte_len(&self) -> usize {
        self.data.len()
    }
    fn read_bytes(&self, offset: usize, len: usize) -> Result<Vec<u8>, OjasError> {
        Ok(self.data[offset..offset + len].to_vec())
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Debug)]
struct MockTimeline;
impl Timeline for MockTimeline {
    fn current_value(&self) -> u64 {
        0
    }
    fn wait(&self, _val: u64) -> Result<(), ojas_core::FenceError> {
        Ok(())
    }
}

struct MockHostLease(Vec<u8>);
impl ojas_core::HostLease for MockHostLease {
    fn as_bytes(&self) -> &[u8] {
        &self.0
    }
    fn as_f32(&self) -> Result<&[f32], OjasError> {
        let (head, mid, tail) = unsafe { self.0.align_to::<f32>() };
        if head.is_empty() && tail.is_empty() {
            Ok(mid)
        } else {
            Err(OjasError::Shape {
                op: "as_f32",
                detail: "unaligned".into(),
            })
        }
    }
    fn as_u32(&self) -> Result<&[u32], OjasError> {
        let (head, mid, tail) = unsafe { self.0.align_to::<u32>() };
        if head.is_empty() && tail.is_empty() {
            Ok(mid)
        } else {
            Err(OjasError::Shape {
                op: "as_u32",
                detail: "unaligned".into(),
            })
        }
    }
}

impl MappedBuffer for MockCpuMappedBuffer {
    fn host_ptr(&self) -> *mut u8 {
        self.data.as_ptr() as *mut u8
    }
    fn gpu_address(&self) -> u64 {
        0
    }
    fn timeline(&self) -> &dyn Timeline {
        &MockTimeline
    }
    fn host_read(&self) -> Result<Box<dyn ojas_core::HostLease>, ojas_core::FenceError> {
        Ok(Box::new(MockHostLease(self.data.clone())))
    }
    fn host_write(&self) -> Result<Box<dyn ojas_core::HostLeaseMut>, ojas_core::FenceError> {
        Err(ojas_core::FenceError::Busy {
            detail: "unsupported".into(),
        })
    }
}

#[test]
fn upload_non_metal_mapped_buffer_refuses_fail_closed() {
    let budget = Budget::new(64 << 20);
    let backend = ojas_metal::MetalBackend::new(budget.clone()).expect("Metal device");

    let mock = MockCpuMappedBuffer {
        data: vec![0u8; 1024],
    };
    let tensor = Tensor::from_mapped(Arc::new(mock), &[256], DType::F32, &budget).unwrap();

    let res = backend.upload(&tensor);
    assert!(
        matches!(
            res,
            Err(OjasError::Placement {
                expected: Some(BackendId::Metal),
                found: Some(BackendId::Cpu),
                ..
            })
        ),
        "upload of non-Metal MappedBuffer must fail closed with OjasError::Placement: got {res:?}"
    );
}

#[test]
fn budget_capacity_exhaustion_is_intercepted_and_recycles_cleanly() {
    // Open a backend with a 1 MiB budget cap
    let tight_budget = Budget::new(1 << 20);
    let backend = ojas_metal::MetalBackend::new(tight_budget.clone()).expect("Metal device");

    // Attempt to allocate 8 MiB tensor
    let oversized = Tensor::from_f32(
        &vec![1.0f32; 2 * 1024 * 1024],
        &[2 * 1024 * 1024],
        &Budget::new(64 << 20),
    )
    .unwrap();
    let res = backend.upload(&oversized);

    assert!(
        matches!(res, Err(OjasError::CapacityExceeded { .. })),
        "oversized allocation must return OjasError::CapacityExceeded: got {res:?}"
    );

    // Verify backend is still functional after refusal
    let small = Tensor::from_f32(&[42.0f32; 16], &[16], &tight_budget).unwrap();
    let dev = backend
        .upload(&small)
        .expect("subsequent small upload must succeed");
    backend.sync().expect("sync");
    let back = backend
        .download(&dev)
        .expect("download")
        .to_f32_vec()
        .unwrap();
    assert_eq!(back[0], 42.0);
}

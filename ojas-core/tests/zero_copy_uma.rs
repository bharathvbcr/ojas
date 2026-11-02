//! Programmatic Validation of Unified Zero-Copy Memory Execution in Ojas.
//!
//! Validates:
//! 1. UMA pointer equivalence (`host_ptr == device_ptr`).
//! 2. `Tensor::to_host` on `Payload::Mapped` over `MappedBuffer` (as implemented
//!    by `tessl::SharedBuf`) returns `Ok(self.clone())` without copying any bytes.
//! 3. Copy tracking confirms 0 calls to `DeviceBuffer::read_bytes` and 0 bytes transferred.
//! 4. Memory budget reservations are preserved without redundant host-device allocations.
//! 5. Fail-closed placement: direct CPU access is refused with `OjasError::Placement`.
//! 6. Host access is safely mediated via timeline-fenced zero-copy leases (`Tensor::host_view`).
//! 7. GPU timeline synchronization drains in-flight device writes before host observation.
//! 8. Strided and sub-tensor views preserve zero-copy sharing and budget tracking.

use ojas_core::{
    BackendId, Budget, DType, DeviceBuffer, FenceError, HostLease, HostLeaseMut, MappedBuffer,
    OjasError, Tensor, Timeline,
};
use std::any::Any;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

#[derive(Debug)]
struct MockTimeline {
    submitted: AtomicU64,
    completed: AtomicU64,
}

impl MockTimeline {
    fn new(submitted: u64, completed: u64) -> Self {
        Self {
            submitted: AtomicU64::new(submitted),
            completed: AtomicU64::new(completed),
        }
    }
}

impl Timeline for MockTimeline {
    fn submitted(&self) -> Result<u64, FenceError> {
        Ok(self.submitted.load(Ordering::SeqCst))
    }

    fn completed(&self) -> Result<u64, FenceError> {
        Ok(self.completed.load(Ordering::SeqCst))
    }

    fn wait(&self, value: u64) -> Result<(), FenceError> {
        let sub = self.submitted.load(Ordering::SeqCst);
        let target = value.min(sub);
        self.completed.fetch_max(target, Ordering::SeqCst);
        Ok(())
    }
}

struct UmaLease {
    bytes: Vec<u8>,
    f32s: Option<Vec<f32>>,
}

impl HostLease for UmaLease {
    fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    fn as_f32(&self) -> Result<&[f32], OjasError> {
        self.f32s.as_deref().ok_or_else(|| OjasError::Shape {
            op: "UmaLease::as_f32",
            detail: "unaligned or non-word length".to_string(),
        })
    }
}

struct UmaLeaseMut {
    data: Arc<RwLock<Vec<u8>>>,
    bytes: Vec<u8>,
    f32s: Option<Vec<f32>>,
}

impl HostLease for UmaLeaseMut {
    fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    fn as_f32(&self) -> Result<&[f32], OjasError> {
        self.f32s.as_deref().ok_or_else(|| OjasError::Shape {
            op: "UmaLeaseMut::as_f32",
            detail: "unaligned or non-word length".to_string(),
        })
    }
}

impl HostLeaseMut for UmaLeaseMut {
    fn as_bytes_mut(&mut self) -> &mut [u8] {
        &mut self.bytes
    }

    fn as_f32_mut(&mut self) -> Result<&mut [f32], OjasError> {
        self.f32s.as_deref_mut().ok_or_else(|| OjasError::Shape {
            op: "UmaLeaseMut::as_f32_mut",
            detail: "unaligned or non-word length".to_string(),
        })
    }
}

impl Drop for UmaLeaseMut {
    fn drop(&mut self) {
        if let Some(ref f32s) = self.f32s {
            let bytes: Vec<u8> = f32s.iter().flat_map(|f| f.to_ne_bytes()).collect();
            let mut guard = self.data.write().unwrap();
            *guard = bytes;
        } else {
            let mut guard = self.data.write().unwrap();
            *guard = self.bytes.clone();
        }
    }
}

/// Simulated Apple Silicon UMA MappedBuffer that monitors all data transfers.
#[derive(Debug)]
struct UmaMappedBuffer {
    data: Arc<RwLock<Vec<u8>>>,
    virtual_address: usize,
    timeline: Arc<MockTimeline>,
    pub read_bytes_calls: AtomicUsize,
    pub bytes_copied: AtomicUsize,
    pub host_read_leases: AtomicUsize,
    pub host_write_leases: AtomicUsize,
}

impl UmaMappedBuffer {
    fn new(initial_bytes: Vec<u8>) -> Self {
        // Allocate a dedicated address representation simulating Apple Silicon UMA
        // where host virtual address and GPU virtual address point to identical DRAM pages.
        let addr = initial_bytes.as_ptr() as usize;
        Self {
            data: Arc::new(RwLock::new(initial_bytes)),
            virtual_address: addr,
            timeline: Arc::new(MockTimeline::new(1, 1)),
            read_bytes_calls: AtomicUsize::new(0),
            bytes_copied: AtomicUsize::new(0),
            host_read_leases: AtomicUsize::new(0),
            host_write_leases: AtomicUsize::new(0),
        }
    }

    fn with_timeline(initial_bytes: Vec<u8>, submitted: u64, completed: u64) -> Self {
        let addr = initial_bytes.as_ptr() as usize;
        Self {
            data: Arc::new(RwLock::new(initial_bytes)),
            virtual_address: addr,
            timeline: Arc::new(MockTimeline::new(submitted, completed)),
            read_bytes_calls: AtomicUsize::new(0),
            bytes_copied: AtomicUsize::new(0),
            host_read_leases: AtomicUsize::new(0),
            host_write_leases: AtomicUsize::new(0),
        }
    }
}

impl DeviceBuffer for UmaMappedBuffer {
    fn backend(&self) -> BackendId {
        BackendId::Metal
    }

    fn byte_len(&self) -> usize {
        self.data.read().unwrap().len()
    }

    fn read_bytes(&self, offset: usize, len: usize) -> Result<Vec<u8>, OjasError> {
        self.read_bytes_calls.fetch_add(1, Ordering::SeqCst);
        self.bytes_copied.fetch_add(len, Ordering::SeqCst);
        let guard = self.data.read().unwrap();
        if offset + len > guard.len() {
            return Err(OjasError::OutOfRange {
                op: "DeviceBuffer::read_bytes",
                detail: format!("offset {offset} + len {len} > {}", guard.len()),
            });
        }
        Ok(guard[offset..offset + len].to_vec())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl MappedBuffer for UmaMappedBuffer {
    fn host_ptr(&self) -> *mut u8 {
        self.virtual_address as *mut u8
    }

    fn gpu_address(&self) -> u64 {
        self.virtual_address as u64
    }

    fn timeline(&self) -> &dyn Timeline {
        self.timeline.as_ref()
    }

    fn host_read(&self) -> Result<Box<dyn HostLease>, FenceError> {
        self.timeline.wait(self.timeline.submitted()?)?;
        self.host_read_leases.fetch_add(1, Ordering::SeqCst);
        let guard = self.data.read().unwrap();
        let bytes = guard.clone();
        let f32s = if bytes.len().is_multiple_of(4) {
            Some(
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_ne_bytes(*b))
                    .collect(),
            )
        } else {
            None
        };
        Ok(Box::new(UmaLease { bytes, f32s }))
    }

    fn host_write(&self) -> Result<Box<dyn HostLeaseMut>, FenceError> {
        self.timeline.wait(self.timeline.submitted()?)?;
        self.host_write_leases.fetch_add(1, Ordering::SeqCst);
        let guard = self.data.read().unwrap();
        let bytes = guard.clone();
        let f32s = if bytes.len().is_multiple_of(4) {
            Some(
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_ne_bytes(*b))
                    .collect(),
            )
        } else {
            None
        };
        drop(guard);
        Ok(Box::new(UmaLeaseMut {
            data: Arc::clone(&self.data),
            bytes,
            f32s,
        }))
    }
}

/// A standard discrete (non-mapped) DeviceBuffer that requires explicit PCIe copies.
#[derive(Debug)]
struct DiscreteDeviceBuffer {
    data: Vec<u8>,
    pub read_bytes_calls: AtomicUsize,
    pub bytes_copied: AtomicUsize,
}

impl DiscreteDeviceBuffer {
    fn new(data: Vec<u8>) -> Self {
        Self {
            data,
            read_bytes_calls: AtomicUsize::new(0),
            bytes_copied: AtomicUsize::new(0),
        }
    }
}

impl DeviceBuffer for DiscreteDeviceBuffer {
    fn backend(&self) -> BackendId {
        BackendId::Cuda
    }

    fn byte_len(&self) -> usize {
        self.data.len()
    }

    fn read_bytes(&self, offset: usize, len: usize) -> Result<Vec<u8>, OjasError> {
        self.read_bytes_calls.fetch_add(1, Ordering::SeqCst);
        self.bytes_copied.fetch_add(len, Ordering::SeqCst);
        if offset + len > self.data.len() {
            return Err(OjasError::OutOfRange {
                op: "DiscreteDeviceBuffer::read_bytes",
                detail: format!("offset {offset} + len {len} > {}", self.data.len()),
            });
        }
        Ok(self.data[offset..offset + len].to_vec())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[test]
fn test_uma_pointer_equivalence_and_zero_copy_elimination() {
    let budget = Budget::new(64 * 1024);
    let values: Vec<f32> = (0..64).map(|i| i as f32 * 2.5).collect();
    let raw_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();

    let uma_buf = Arc::new(UmaMappedBuffer::new(raw_bytes));

    // 1. UMA Pointer Equivalence Verification
    // On unified memory architectures, the CPU virtual address (host_ptr)
    // and GPU virtual address (gpu_address) address the identical DRAM page.
    let host_ptr = uma_buf.host_ptr();
    let gpu_addr = uma_buf.gpu_address();
    assert_eq!(
        host_ptr as u64, gpu_addr,
        "UMA pointer equivalence violated: host_ptr ({host_ptr:?}) != gpu_addr ({gpu_addr:#x})"
    );

    // 2. Wrap buffer into Tensor::from_mapped
    let tensor = Tensor::from_mapped(uma_buf.clone(), &[64], DType::F32, &budget)
        .expect("from_mapped tensor");

    assert_eq!(tensor.device(), Some(BackendId::Metal));
    assert!(tensor.mapped_buffer().is_some());
    assert!(tensor.device_buffer().is_none());
    assert_eq!(
        budget.live_bytes().unwrap(),
        256,
        "Initial budget must reflect mapped buffer length exactly"
    );

    // 3. Programmatic to_host verification
    // Calling to_host on a Mapped tensor MUST NOT copy any bytes or charge extra budget!
    let host_tensor = tensor.to_host(&budget).expect("to_host on mapped tensor");

    // Assert zero buffer copying
    assert_eq!(
        uma_buf.read_bytes_calls.load(Ordering::SeqCst),
        0,
        "Tensor::to_host on Payload::Mapped must not invoke DeviceBuffer::read_bytes"
    );
    assert_eq!(
        uma_buf.bytes_copied.load(Ordering::SeqCst),
        0,
        "Tensor::to_host on Payload::Mapped must copy exactly 0 bytes"
    );

    // Assert memory footprint elimination (no double allocation)
    assert_eq!(
        budget.live_bytes().unwrap(),
        256,
        "Budget must remain exactly 256 bytes without allocating duplicate host copy"
    );

    // Assert underlying storage identity (shared Arc<dyn MappedBuffer>)
    assert!(
        Arc::ptr_eq(
            tensor.mapped_buffer().unwrap(),
            host_tensor.mapped_buffer().unwrap()
        ),
        "Returned tensor must share identical MappedBuffer storage"
    );

    // 4. Safe data access via host_view
    let lease = host_tensor.host_view().expect("host_view");
    let f32_slice = lease.as_f32().expect("view as f32");
    assert_eq!(f32_slice, &values[..]);

    drop(lease);
    drop(host_tensor);
    drop(tensor);

    // 5. Clean teardown
    assert_eq!(
        budget.live_bytes().unwrap(),
        0,
        "Budget must cleanly return to 0 upon dropping all references"
    );
}

#[test]
fn test_comparison_discrete_vs_uma_zero_copy() {
    let budget = Budget::new(64 * 1024);
    let values: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let raw_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();

    // Discrete Device Tensor
    let discrete_buf = Arc::new(DiscreteDeviceBuffer::new(raw_bytes.clone()));
    let res = budget.try_reserve(256).unwrap();
    let discrete_tensor =
        Tensor::from_device_reserved(discrete_buf.clone(), &[64], DType::F32, res).unwrap();

    let discrete_host = discrete_tensor.to_host(&budget).expect("discrete to_host");

    // Discrete path MUST perform read_bytes and copy memory
    assert_eq!(
        discrete_buf.read_bytes_calls.load(Ordering::SeqCst),
        1,
        "Discrete device buffer must invoke read_bytes"
    );
    assert_eq!(
        discrete_buf.bytes_copied.load(Ordering::SeqCst),
        256,
        "Discrete device buffer must copy all 256 bytes"
    );
    // Discrete to_host produces a Host payload
    assert_eq!(discrete_host.device(), None);

    // In contrast, UMA Mapped Tensor performs ZERO read_bytes calls and copies ZERO bytes!
    let uma_buf = Arc::new(UmaMappedBuffer::new(raw_bytes));
    let uma_tensor = Tensor::from_mapped(uma_buf.clone(), &[64], DType::F32, &budget).unwrap();
    let uma_host = uma_tensor.to_host(&budget).expect("uma to_host");

    assert_eq!(
        uma_buf.read_bytes_calls.load(Ordering::SeqCst),
        0,
        "UMA MappedBuffer must eliminate read_bytes completely"
    );
    assert_eq!(
        uma_buf.bytes_copied.load(Ordering::SeqCst),
        0,
        "UMA MappedBuffer must eliminate memory copying completely"
    );
    assert_eq!(
        uma_host.device(),
        Some(BackendId::Metal),
        "UMA host tensor remains resident on Metal with zero-copy access"
    );
}

#[test]
fn test_fail_closed_placement_on_uma_tensor() {
    let budget = Budget::new(1024);
    let values = [1.0f32, 2.0, 3.0, 4.0];
    let raw_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();

    let uma_buf = Arc::new(UmaMappedBuffer::new(raw_bytes));
    let tensor = Tensor::from_mapped(uma_buf, &[4], DType::F32, &budget).unwrap();

    // Even though physically resident in unified DRAM, direct host CPU methods
    // must fail closed to prevent unsynchronized concurrent access.
    assert!(matches!(
        tensor.f32_slice(),
        Err(OjasError::Placement {
            expected: None,
            found: Some(BackendId::Metal),
            ..
        })
    ));

    assert!(matches!(
        tensor.to_f32_vec(),
        Err(OjasError::Placement {
            expected: None,
            found: Some(BackendId::Metal),
            ..
        })
    ));

    let mut mut_tensor = tensor.clone();
    assert!(matches!(
        mut_tensor.write_f32(&values),
        Err(OjasError::Placement {
            expected: None,
            found: Some(BackendId::Metal),
            ..
        })
    ));

    // Access must be explicitly leased via host_view
    let lease = tensor.host_view().unwrap();
    assert_eq!(lease.as_f32().unwrap(), &values);
}

#[test]
fn test_timeline_drain_on_to_host() {
    let budget = Budget::new(1024);
    let values = [10.0f32, 20.0, 30.0, 40.0];
    let raw_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();

    // Timeline condition: a GPU write was submitted at 8, but only completed to 3.
    let uma_buf = Arc::new(UmaMappedBuffer::with_timeline(raw_bytes, 8, 3));
    assert_eq!(uma_buf.timeline.completed().unwrap(), 3);
    assert_eq!(uma_buf.timeline.submitted().unwrap(), 8);

    let tensor = Tensor::from_mapped(uma_buf.clone(), &[4], DType::F32, &budget).unwrap();

    // to_host must drain the timeline to 8 before returning to prevent read races.
    let host_tensor = tensor.to_host(&budget).expect("to_host");

    assert_eq!(
        uma_buf.timeline.completed().unwrap(),
        8,
        "to_host must drain pending GPU writes to match submitted completion stamp"
    );
    assert_eq!(uma_buf.read_bytes_calls.load(Ordering::SeqCst), 0);
    assert_eq!(uma_buf.bytes_copied.load(Ordering::SeqCst), 0);

    let lease = host_tensor.host_view().unwrap();
    assert_eq!(lease.as_f32().unwrap(), &values);
}

#[test]
fn test_strided_and_narrow_views_preserve_zero_copy() {
    let budget = Budget::new(4096);
    let values: Vec<f32> = (0..32).map(|i| i as f32 * 1.0).collect();
    let raw_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();

    let uma_buf = Arc::new(UmaMappedBuffer::new(raw_bytes));
    let tensor = Tensor::from_mapped(uma_buf.clone(), &[32], DType::F32, &budget).unwrap();

    // Narrow tensor view from index 8 to 24 (16 elements = 64 bytes)
    // byte_offset = 8 * 4 = 32 bytes, shape = [16], strides = [1]
    let narrowed = tensor.narrow(32, &[16], &[1]).expect("narrow view");
    assert_eq!(narrowed.shape(), &[16]);

    // to_host on the narrowed view
    let host_narrowed = narrowed.to_host(&budget).expect("narrowed to_host");

    // Zero-copy invariant preserved
    assert_eq!(uma_buf.read_bytes_calls.load(Ordering::SeqCst), 0);
    assert_eq!(uma_buf.bytes_copied.load(Ordering::SeqCst), 0);

    // Host view reflects the backing shared buffer and window offset
    let lease = host_narrowed.host_view().expect("narrowed host_view");
    let f32s = lease.as_f32().expect("slice as f32");
    assert_eq!(f32s.len(), 32);
    let window_start = host_narrowed.byte_offset() / std::mem::size_of::<f32>();
    let window_len = host_narrowed.num_elements().unwrap();
    assert_eq!(window_start, 8);
    assert_eq!(window_len, 16);
    assert_eq!(
        &f32s[window_start..window_start + window_len],
        &values[8..24]
    );

    drop(lease);
    drop(host_narrowed);
    drop(narrowed);
    drop(tensor);

    assert_eq!(budget.live_bytes().unwrap(), 0);
}

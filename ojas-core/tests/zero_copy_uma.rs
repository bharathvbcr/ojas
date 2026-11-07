//! Programmatic Validation of Unified Zero-Copy Memory Execution in Ojas.
//!
//! Validates:
//! 1. UMA pointer equivalence (`host_ptr == device_ptr`).
//! 2. `Tensor::host_view` on `Payload::Mapped` is the zero-copy path: a lease over
//!    the view's window, no budget charge. `Tensor::to_host` is a host snapshot:
//!    it charges the window, drops the lease, and is unchanged by later device writes.
//! 3. Copy tracking confirms 0 calls to `DeviceBuffer::read_bytes` (the snapshot reads
//!    through the fenced lease, not `read_bytes`).
//! 4. Memory budget reservations are exact: the view is free, the snapshot is charged and released.
//! 5. Fail-closed placement: direct CPU access is refused with `OjasError::Placement`.
//! 6. Host access is safely mediated via timeline-fenced zero-copy leases (`Tensor::host_view`).
//! 7. GPU timeline synchronization drains in-flight device writes before host observation.
//! 8. Narrow and strided views: the lease is windowed, a strided view is refused, a snapshot keeps strides.

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

    // 3. to_host is a host snapshot: a budget-charged copy of the window that
    // reads through the fenced lease, not DeviceBuffer::read_bytes.
    let host_tensor = tensor.to_host(&budget).expect("to_host on mapped tensor");
    assert_eq!(
        uma_buf.read_bytes_calls.load(Ordering::SeqCst),
        0,
        "the snapshot reads through the fenced host_read lease"
    );
    assert_eq!(uma_buf.host_read_leases.load(Ordering::SeqCst), 1);
    assert_eq!(host_tensor.device(), None, "a snapshot is a host tensor");
    assert!(host_tensor.mapped_buffer().is_none());
    assert_eq!(
        budget.live_bytes().unwrap(),
        512,
        "the snapshot charges exactly the window's 256 bytes on top of the mapping"
    );
    assert_eq!(budget.device_readbacks(), (1, 256));
    assert_eq!(
        host_tensor.to_f32_vec().expect("host snapshot reads"),
        values
    );

    // 4. host_view is the zero-copy path: leasing it charges nothing.
    let lease = tensor.host_view().expect("host_view");
    assert_eq!(lease.as_f32().expect("view as f32"), &values[..]);
    assert_eq!(budget.live_bytes().unwrap(), 512, "a lease is free");
    assert_eq!(
        budget.device_readbacks(),
        (1, 256),
        "a lease is no readback"
    );

    drop(lease);
    drop(host_tensor);
    assert_eq!(
        budget.live_bytes().unwrap(),
        256,
        "the snapshot is released"
    );
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

    // In contrast, a UMA mapped tensor snapshots through its fenced lease: ZERO read_bytes calls.
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
        None,
        "a mapped to_host is a host snapshot, not a mapped clone"
    );
    assert_eq!(uma_host.to_f32_vec().expect("snapshot reads"), values);
    assert_eq!(
        uma_buf.host_read_leases.load(Ordering::SeqCst),
        1,
        "the snapshot is taken through one fenced lease"
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
fn test_narrow_view_snapshot_is_the_window_and_host_view_is_windowed() {
    let budget = Budget::new(4096);
    let values: Vec<f32> = (0..32).map(|i| i as f32 * 1.0).collect();
    let raw_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();

    let uma_buf = Arc::new(UmaMappedBuffer::new(raw_bytes));
    let tensor = Tensor::from_mapped(uma_buf.clone(), &[32], DType::F32, &budget).unwrap();

    // Narrow tensor view from index 8 to 24 (16 elements = 64 bytes)
    // byte_offset = 8 * 4 = 32 bytes, shape = [16], strides = [1]
    let narrowed = tensor.narrow(32, &[16], &[1]).expect("narrow view");
    assert_eq!(narrowed.shape(), &[16]);

    // to_host copies exactly the window into a host tensor at offset 0.
    let host_narrowed = narrowed.to_host(&budget).expect("narrowed to_host");
    assert_eq!(uma_buf.read_bytes_calls.load(Ordering::SeqCst), 0);
    assert_eq!(host_narrowed.device(), None);
    assert_eq!(host_narrowed.byte_offset(), 0);
    assert_eq!(host_narrowed.to_f32_vec().unwrap(), &values[8..24]);
    assert_eq!(budget.live_bytes().unwrap(), 128 + 64);
    assert_eq!(budget.device_readbacks(), (1, 64));

    // host_view of the mapped view is exactly the window, not the allocation.
    let lease = narrowed.host_view().expect("narrowed host_view");
    assert_eq!(lease.as_bytes().len(), 64);
    assert_eq!(lease.as_f32().expect("window as f32"), &values[8..24]);
    assert_eq!(budget.live_bytes().unwrap(), 128 + 64, "a lease is free");

    drop(lease);
    drop(host_narrowed);
    drop(narrowed);
    drop(tensor);

    assert_eq!(budget.live_bytes().unwrap(), 0);
}

#[test]
fn test_strided_mapped_view_is_refused_by_host_view_and_snapshotted_by_to_host() {
    let budget = Budget::new(4096);
    let values: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let raw_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let uma_buf = Arc::new(UmaMappedBuffer::new(raw_bytes));
    let tensor = Tensor::from_mapped(uma_buf.clone(), &[4, 4], DType::F32, &budget).unwrap();

    // Transposed view: strides [1, 4] elements, not one run of bytes.
    let strided = tensor.view(&[4, 4], &[1, 4], 0).expect("strided view");
    assert!(!strided.is_contiguous().unwrap());

    let err = strided.host_view().map(|_| ()).unwrap_err();
    assert!(
        matches!(
            err,
            OjasError::Shape {
                op: "Tensor::host_view",
                ..
            }
        ),
        "a non-contiguous mapped view must be refused, got {err:?}"
    );
    assert_eq!(
        uma_buf.host_read_leases.load(Ordering::SeqCst),
        0,
        "the refusal happens before any lease is taken"
    );

    // to_host mirrors the Device branch: the whole allocation, strides kept.
    let snap = strided.to_host(&budget).expect("strided to_host");
    assert_eq!(snap.device(), None);
    assert!(!snap.is_contiguous().unwrap());
    assert_eq!(snap.strides(), strided.strides());
    assert_eq!(budget.live_bytes().unwrap(), 64 + 64);
    assert_eq!(budget.device_readbacks(), (1, 64));
    drop((snap, strided, tensor));
    assert_eq!(budget.live_bytes().unwrap(), 0);
}

/// A mapped buffer whose lease borrows the stored bytes (an `Arc`, so the
/// address is checkable), counts live leases, and refuses a device write while
/// one is live, as a real fence does.
#[derive(Debug)]
struct ArcMapped {
    snap: RwLock<Arc<Snap>>,
    live_leases: Arc<AtomicUsize>,
    timeline: MockTimeline,
    /// When set, `host_read` hands back a lease this many bytes short.
    short_by: usize,
}

#[derive(Debug)]
struct Snap {
    bytes: Vec<u8>,
    f32s: Vec<f32>,
}

impl Snap {
    fn of(values: &[f32]) -> Arc<Self> {
        Arc::new(Self {
            bytes: values.iter().flat_map(|v| v.to_ne_bytes()).collect(),
            f32s: values.to_vec(),
        })
    }
}

struct ArcLease {
    snap: Arc<Snap>,
    live: Arc<AtomicUsize>,
    short_by: usize,
}

impl HostLease for ArcLease {
    fn as_bytes(&self) -> &[u8] {
        &self.snap.bytes[..self.snap.bytes.len() - self.short_by]
    }
    fn as_f32(&self) -> Result<&[f32], OjasError> {
        Ok(&self.snap.f32s)
    }
}

impl Drop for ArcLease {
    fn drop(&mut self) {
        self.live.fetch_sub(1, Ordering::SeqCst);
    }
}

impl ArcMapped {
    fn new(values: &[f32]) -> Arc<Self> {
        Arc::new(Self {
            snap: RwLock::new(Snap::of(values)),
            live_leases: Arc::new(AtomicUsize::new(0)),
            timeline: MockTimeline::new(1, 1),
            short_by: 0,
        })
    }

    fn short(values: &[f32], short_by: usize) -> Arc<Self> {
        Arc::new(Self {
            snap: RwLock::new(Snap::of(values)),
            live_leases: Arc::new(AtomicUsize::new(0)),
            timeline: MockTimeline::new(1, 1),
            short_by,
        })
    }

    fn base_ptr(&self) -> *const u8 {
        self.snap.read().unwrap().bytes.as_ptr()
    }

    /// A GPU write (or bind): refused while a host lease is live.
    fn device_write(&self, values: &[f32]) -> Result<(), FenceError> {
        if self.live_leases.load(Ordering::SeqCst) != 0 {
            return Err(FenceError::Busy {
                detail: "a host lease is live".to_string(),
            });
        }
        *self.snap.write().unwrap() = Snap::of(values);
        Ok(())
    }
}

impl DeviceBuffer for ArcMapped {
    fn backend(&self) -> BackendId {
        BackendId::Metal
    }
    fn byte_len(&self) -> usize {
        self.snap.read().unwrap().bytes.len()
    }
    fn read_bytes(&self, _offset: usize, _len: usize) -> Result<Vec<u8>, OjasError> {
        Err(OjasError::Unsupported {
            op: "ArcMapped::read_bytes",
            detail: "this mock is read only through its lease".to_string(),
        })
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl MappedBuffer for ArcMapped {
    fn timeline(&self) -> &dyn Timeline {
        &self.timeline
    }
    fn host_read(&self) -> Result<Box<dyn HostLease>, FenceError> {
        self.timeline.wait(self.timeline.submitted()?)?;
        self.live_leases.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(ArcLease {
            snap: Arc::clone(&self.snap.read().unwrap()),
            live: Arc::clone(&self.live_leases),
            short_by: self.short_by,
        }))
    }
}

#[test]
fn host_view_is_zero_copy_over_the_view_window_and_holds_the_fence() {
    let budget = Budget::new(4096);
    let values: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let buf = ArcMapped::new(&values);
    let tensor = Tensor::from_mapped(buf.clone(), &[4, 8], DType::F32, &budget).unwrap();
    assert_eq!(budget.live_bytes().unwrap(), 128);

    // Row 2: byte_offset 64, 8 elements. The lease is that row, at that address.
    let row = tensor.narrow(64, &[8], &[1]).expect("row view");
    let lease = row.host_view().expect("row host_view");
    assert_eq!(
        lease.as_bytes().len(),
        32,
        "exactly the window, not the allocation"
    );
    assert_eq!(lease.as_f32().unwrap(), &values[16..24]);
    assert_eq!(
        lease.as_bytes().as_ptr(),
        buf.base_ptr().wrapping_add(64),
        "zero copy: the window starts inside the base allocation"
    );
    assert_eq!(budget.live_bytes().unwrap(), 128, "a lease charges nothing");
    assert_eq!(budget.device_readbacks(), (0, 0));

    // The lease holds the buffer's fence: a device write is refused until it drops.
    assert!(matches!(
        buf.device_write(&values),
        Err(FenceError::Busy { .. })
    ));
    drop(lease);
    buf.device_write(&values)
        .expect("write after the lease dropped");
}

#[test]
fn to_host_is_a_snapshot_unchanged_by_a_later_device_write_and_released_on_drop() {
    let budget = Budget::new(4096);
    let values: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let buf = ArcMapped::new(&values);
    let tensor = Tensor::from_mapped(buf.clone(), &[4, 8], DType::F32, &budget).unwrap();
    let row = tensor.narrow(64, &[8], &[1]).expect("row view");

    let snap = row.to_host(&budget).expect("snapshot");
    assert_eq!(snap.device(), None);
    assert_eq!(snap.to_f32_vec().unwrap(), &values[16..24]);
    assert_eq!(
        budget.live_bytes().unwrap(),
        128 + 32,
        "window only is charged"
    );
    assert_eq!(budget.device_readbacks(), (1, 32));
    assert_eq!(
        buf.live_leases.load(Ordering::SeqCst),
        0,
        "the snapshot drops its lease before returning"
    );

    // A device write after the snapshot must not show through it.
    let rewritten: Vec<f32> = values.iter().map(|v| v + 100.0).collect();
    buf.device_write(&rewritten).expect("device write");
    assert_eq!(snap.to_f32_vec().unwrap(), &values[16..24]);
    assert_eq!(
        row.host_view().unwrap().as_f32().unwrap(),
        &rewritten[16..24],
        "the leased view does see the new bytes"
    );

    drop(snap);
    assert_eq!(
        budget.live_bytes().unwrap(),
        128,
        "the snapshot's charge is released"
    );
    drop((row, tensor));
    assert_eq!(budget.live_bytes().unwrap(), 0);
}

#[test]
fn a_lease_shorter_than_the_buffer_is_a_backend_error_and_charges_nothing() {
    let budget = Budget::new(4096);
    let values: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let buf = ArcMapped::short(&values, 4);
    let tensor = Tensor::from_mapped(buf, &[8], DType::F32, &budget).unwrap();

    let err = tensor.to_host(&budget).unwrap_err();
    assert!(matches!(err, OjasError::Backend { .. }), "got {err:?}");
    assert_eq!(budget.live_bytes().unwrap(), 32, "no snapshot kept");
    assert_eq!(budget.device_readbacks(), (0, 0));
    let err = tensor.host_view().map(|_| ()).unwrap_err();
    assert!(matches!(err, OjasError::OutOfRange { .. }), "got {err:?}");
}

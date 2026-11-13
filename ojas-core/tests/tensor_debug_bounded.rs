//! `Tensor`'s `Debug` is bounded: it names dtype, shape and placement and
//! never prints (or reads) the contents.

use std::any::Any;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use ojas_core::{BackendId, Budget, DType, DeviceBuffer, OjasError, Tensor};

const N: usize = 1 << 20;

#[test]
fn host_tensor_debug_is_bounded_and_names_dtype_shape_placement() {
    let budget = Budget::new(64 << 20);
    let data: Vec<f32> = (0..N).map(|i| i as f32 + 0.123_456).collect();
    let t = Tensor::from_f32(&data, &[N], &budget).unwrap();
    let s = format!("{t:?}");
    assert!(s.len() < 256, "Debug output is {} bytes: {s}", s.len());
    assert!(s.contains("F32"), "dtype missing: {s}");
    assert!(s.contains(&N.to_string()), "shape missing: {s}");
    assert!(s.contains("host"), "placement missing: {s}");
    // No element text: element 1 is 1.123456.
    assert!(!s.contains("1.123"), "contents leaked: {s}");
}

#[test]
fn host_tensor_debug_names_each_dtype_and_a_2d_shape() {
    let budget = Budget::new(1 << 20);
    for dtype in [DType::F32, DType::U32, DType::Bf16, DType::F16] {
        let t = Tensor::zeros(&[3, 5], dtype, &budget).unwrap();
        let s = format!("{t:?}");
        assert!(s.contains(&format!("{dtype:?}")), "{s}");
        assert!(s.contains("[3, 5]"), "{s}");
    }
}

struct CountingDevice {
    bytes: usize,
    reads: AtomicUsize,
}

impl std::fmt::Debug for CountingDevice {
    // Deliberately huge: a Debug that delegated to the buffer would blow the bound.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for _ in 0..1000 {
            f.write_str("DEVICE-BUFFER-DEBUG-LEAK ")?;
        }
        Ok(())
    }
}

impl DeviceBuffer for CountingDevice {
    fn backend(&self) -> BackendId {
        BackendId::Metal
    }

    fn byte_len(&self) -> usize {
        self.bytes
    }

    fn read_bytes(&self, _offset: usize, _len: usize) -> Result<Vec<u8>, OjasError> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        Ok(vec![0; self.bytes])
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[test]
fn device_tensor_debug_reads_nothing_and_charges_nothing() {
    let budget = Budget::new(64 << 20);
    let buf = Arc::new(CountingDevice {
        bytes: N * 4,
        reads: AtomicUsize::new(0),
    });
    let t = Tensor::from_device(buf.clone(), &[N], DType::F32, &budget).unwrap();
    let live_before = budget.live_bytes().unwrap();
    let s = format!("{t:?}");
    assert!(s.len() < 256, "Debug output is {} bytes: {s}", s.len());
    assert!(s.contains("F32"), "{s}");
    assert!(s.contains(&N.to_string()), "{s}");
    assert!(s.contains("device"), "{s}");
    assert!(s.contains("Metal"), "{s}");
    assert!(!s.contains("LEAK"), "buffer Debug leaked: {s}");
    assert_eq!(buf.reads.load(Ordering::SeqCst), 0, "Debug read the device");
    assert_eq!(
        budget.device_readbacks(),
        (0, 0),
        "Debug counted a readback"
    );
    assert_eq!(
        budget.live_bytes().unwrap(),
        live_before,
        "Debug charged the budget"
    );
}

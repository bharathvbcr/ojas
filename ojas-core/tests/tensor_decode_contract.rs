//! Contract of `Tensor::to_f32_vec` and `Tensor::to_u32_vec`.
//!
//! These pin the observable behaviour of the host decode path: bit-exact
//! output for every 32-bit pattern (NaN payloads, signalling NaNs, -0.0,
//! subnormals, infinities), the dtype check, the error each refused view
//! reports, and how `byte_offset`, `narrow` and `view` select the window.
//! They were written against the push-loop decoder and must keep passing
//! for any rewrite of it.

use ojas_core::{BackendId, Budget, DType, DeviceBuffer, OjasError, Scratch, Tensor};
use std::any::Any;
use std::sync::Arc;

struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Bit patterns a decoder that goes through float arithmetic, a canonicalising
/// move, or a lossy cast would change.
const SPECIAL_BITS: [u32; 16] = [
    0x0000_0000, // +0.0
    0x8000_0000, // -0.0
    0x0000_0001, // smallest positive subnormal
    0x8000_0001, // smallest negative subnormal
    0x007F_FFFF, // largest subnormal
    0x0080_0000, // smallest normal
    0x7F80_0000, // +inf
    0xFF80_0000, // -inf
    0x7FC0_0000, // canonical quiet NaN
    0x7FC0_0001, // quiet NaN with payload
    0xFFC0_1234, // negative quiet NaN with payload
    0x7F80_0001, // signalling NaN, smallest payload
    0x7FBF_FFFF, // signalling NaN, largest payload
    0xFF80_0001, // negative signalling NaN
    0x7F7F_FFFF, // f32::MAX
    0xDEAD_BEEF, // arbitrary
];

/// Host tensor whose storage is exactly `bytes`, built without going through
/// any f32 value, so the decoder sees the raw pattern.
fn raw_tensor(bytes: &[u8], shape: &[usize], dtype: DType, budget: &Budget) -> Tensor {
    let mut scratch = Scratch::<u8>::try_alloc(bytes.len(), budget).unwrap();
    scratch.as_mut_slice().copy_from_slice(bytes);
    Tensor::from_scratch(scratch, shape, dtype).unwrap()
}

/// Independent reference: index arithmetic, no chunking iterator.
fn reference_words(bytes: &[u8]) -> Vec<u32> {
    assert_eq!(bytes.len() % 4, 0);
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        out.push(u32::from_ne_bytes([
            bytes[i],
            bytes[i + 1],
            bytes[i + 2],
            bytes[i + 3],
        ]));
        i += 4;
    }
    out
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn special_bytes() -> Vec<u8> {
    SPECIAL_BITS.iter().flat_map(|w| w.to_ne_bytes()).collect()
}

#[test]
fn special_f32_patterns_decode_bit_exact_from_raw_bytes() {
    let budget = Budget::new(1 << 20);
    let bytes = special_bytes();
    let t = raw_tensor(&bytes, &[SPECIAL_BITS.len()], DType::F32, &budget);
    assert_eq!(bits(&t.to_f32_vec().unwrap()), SPECIAL_BITS);
    let u = raw_tensor(&bytes, &[SPECIAL_BITS.len()], DType::U32, &budget);
    assert_eq!(u.to_u32_vec().unwrap(), SPECIAL_BITS);
}

#[test]
fn special_f32_patterns_round_trip_through_from_f32() {
    let budget = Budget::new(1 << 20);
    let values: Vec<f32> = SPECIAL_BITS.iter().map(|&b| f32::from_bits(b)).collect();
    let t = Tensor::from_f32(&values, &[4, 4], &budget).unwrap();
    assert_eq!(bits(&t.to_f32_vec().unwrap()), SPECIAL_BITS);
    let u = Tensor::from_u32(&SPECIAL_BITS, &[2, 8], &budget).unwrap();
    assert_eq!(u.to_u32_vec().unwrap(), SPECIAL_BITS);
}

#[test]
fn random_bytes_decode_bit_exact_at_many_lengths() {
    let budget = Budget::new(1 << 24);
    let mut rng = SplitMix64(0xD3C0_DE00);
    // Lengths straddle every vector width and unroll factor a decoder might use.
    let counts = [
        0usize, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1000,
        4099, 65_537,
    ];
    for &n in &counts {
        let mut bytes: Vec<u8> = (0..n * 4).map(|_| rng.next() as u8).collect();
        // Splice in the special patterns so they also land at odd positions.
        for (k, w) in SPECIAL_BITS.iter().enumerate() {
            let at = (k * 7) % n.max(1);
            if at < n {
                bytes[at * 4..at * 4 + 4].copy_from_slice(&w.to_ne_bytes());
            }
        }
        let expect = reference_words(&bytes);
        let f = raw_tensor(&bytes, &[n], DType::F32, &budget);
        let got = f.to_f32_vec().unwrap();
        assert_eq!(got.len(), n);
        assert_eq!(bits(&got), expect, "f32 n={n}");
        let u = raw_tensor(&bytes, &[n], DType::U32, &budget);
        assert_eq!(u.to_u32_vec().unwrap(), expect, "u32 n={n}");
    }
}

#[test]
fn dtype_mismatch_is_refused_with_op_expected_and_got() {
    let budget = Budget::new(1 << 10);
    let f = Tensor::from_f32(&[1.0, 2.0], &[2], &budget).unwrap();
    let u = Tensor::from_u32(&[1, 2], &[2], &budget).unwrap();
    let bf = Tensor::zeros(&[2], DType::Bf16, &budget).unwrap();
    let hf = Tensor::zeros(&[2], DType::F16, &budget).unwrap();

    let err = f.to_u32_vec().unwrap_err();
    assert!(
        matches!(
            err,
            OjasError::Dtype {
                op: "Tensor::to_u32_vec",
                expected: DType::U32,
                got: DType::F32
            }
        ),
        "{err}"
    );
    assert_eq!(
        err.to_string(),
        "Tensor::to_u32_vec: dtype: expected U32, got F32"
    );

    let err = u.to_f32_vec().unwrap_err();
    assert!(
        matches!(
            err,
            OjasError::Dtype {
                op: "Tensor::to_f32_vec",
                expected: DType::F32,
                got: DType::U32
            }
        ),
        "{err}"
    );
    assert_eq!(
        err.to_string(),
        "Tensor::to_f32_vec: dtype: expected F32, got U32"
    );

    for (t, got) in [(&bf, DType::Bf16), (&hf, DType::F16)] {
        let err = t.to_f32_vec().unwrap_err();
        assert!(
            matches!(err, OjasError::Dtype { op: "Tensor::to_f32_vec", expected: DType::F32, got: g } if g == got),
            "{err}"
        );
        let err = t.to_u32_vec().unwrap_err();
        assert!(
            matches!(err, OjasError::Dtype { op: "Tensor::to_u32_vec", expected: DType::U32, got: g } if g == got),
            "{err}"
        );
    }
}

#[test]
fn narrow_and_view_select_the_window_at_byte_offset() {
    let budget = Budget::new(1 << 10);
    let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let t = Tensor::from_f32(&data, &[2, 4], &budget).unwrap();

    let row1 = t.narrow(16, &[4], &[1]).unwrap();
    assert_eq!(row1.byte_offset(), 16);
    assert_eq!(row1.to_f32_vec().unwrap(), [4.0, 5.0, 6.0, 7.0]);

    // narrow adds to the current offset; view takes it as absolute.
    let inner = row1.narrow(4, &[2], &[1]).unwrap();
    assert_eq!(inner.byte_offset(), 20);
    assert_eq!(inner.to_f32_vec().unwrap(), [5.0, 6.0]);
    let absolute = row1.view(&[2], &[1], 4).unwrap();
    assert_eq!(absolute.to_f32_vec().unwrap(), [1.0, 2.0]);

    // A contiguous 2-D window in the middle of the allocation.
    let mid = t.view(&[2, 3], &[3, 1], 4).unwrap();
    assert_eq!(mid.to_f32_vec().unwrap(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // Same for u32, through the same accessor.
    let ids: Vec<u32> = (100..108).collect();
    let u = Tensor::from_u32(&ids, &[8], &budget).unwrap();
    assert_eq!(
        u.narrow(28, &[1], &[1]).unwrap().to_u32_vec().unwrap(),
        [107]
    );
    assert_eq!(
        u.view(&[3], &[1], 8).unwrap().to_u32_vec().unwrap(),
        [102, 103, 104]
    );
}

fn not_contiguous(err: &OjasError) -> bool {
    matches!(
        err,
        OjasError::Shape { op: "Tensor::contiguous_bytes", detail } if detail == "view is not contiguous"
    )
}

#[test]
fn non_contiguous_views_are_refused_not_gathered() {
    let budget = Budget::new(1 << 10);
    let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let t = Tensor::from_f32(&data, &[2, 3], &budget).unwrap();
    let col = t.view(&[2], &[3], 4).unwrap();
    let transposed = t.view(&[3, 2], &[1, 3], 0).unwrap();
    let broadcast = t.view(&[4, 3], &[0, 1], 0).unwrap();
    for v in [&col, &transposed, &broadcast] {
        let err = v.to_f32_vec().unwrap_err();
        assert!(not_contiguous(&err), "{err}");
        assert_eq!(
            err.to_string(),
            "Tensor::contiguous_bytes: shape: view is not contiguous"
        );
    }
    let ids = Tensor::from_u32(&[1, 2, 3, 4], &[2, 2], &budget).unwrap();
    let err = ids.view(&[2], &[2], 0).unwrap().to_u32_vec().unwrap_err();
    assert!(not_contiguous(&err), "{err}");
}

#[test]
fn empty_and_scalar_shapes() {
    let budget = Budget::new(1 << 10);
    let empty = Tensor::from_f32(&[], &[0, 5], &budget).unwrap();
    assert_eq!(empty.to_f32_vec().unwrap(), Vec::<f32>::new());
    let empty_u = Tensor::from_u32(&[], &[3, 0], &budget).unwrap();
    assert_eq!(empty_u.to_u32_vec().unwrap(), Vec::<u32>::new());
    let scalar = Tensor::from_f32(&[f32::from_bits(0x7F80_0001)], &[], &budget).unwrap();
    assert_eq!(bits(&scalar.to_f32_vec().unwrap()), [0x7F80_0001]);
    let scalar_u = Tensor::from_u32(&[u32::MAX], &[], &budget).unwrap();
    assert_eq!(scalar_u.to_u32_vec().unwrap(), [u32::MAX]);
    // An empty view at the very end of the allocation.
    let t = Tensor::from_f32(&[1.0, 2.0], &[2], &budget).unwrap();
    assert_eq!(
        t.view(&[0], &[1], 8).unwrap().to_f32_vec().unwrap(),
        Vec::<f32>::new()
    );
}

#[test]
fn decoding_does_not_charge_the_budget_or_alias_storage() {
    let budget = Budget::new(1 << 10);
    let t = Tensor::from_f32(&[1.0, 2.0, 3.0], &[3], &budget).unwrap();
    let live = budget.live_bytes().unwrap();
    let mut a = t.to_f32_vec().unwrap();
    assert_eq!(budget.live_bytes().unwrap(), live);
    a[0] = 99.0;
    assert_eq!(t.to_f32_vec().unwrap(), [1.0, 2.0, 3.0]);
}

#[derive(Debug)]
struct FakeDevice(Vec<u8>);

impl DeviceBuffer for FakeDevice {
    fn backend(&self) -> BackendId {
        BackendId::Metal
    }
    fn byte_len(&self) -> usize {
        self.0.len()
    }
    fn read_bytes(&self, offset: usize, len: usize) -> Result<Vec<u8>, OjasError> {
        Ok(self.0[offset..offset + len].to_vec())
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[test]
fn device_tensor_is_placement_and_dtype_is_checked_first() {
    let budget = Budget::new(1 << 10);
    let buf: Arc<dyn DeviceBuffer> = Arc::new(FakeDevice(special_bytes()));
    let f = Tensor::from_device(Arc::clone(&buf), &[16], DType::F32, &budget).unwrap();
    let err = f.to_f32_vec().unwrap_err();
    assert!(
        matches!(
            err,
            OjasError::Placement {
                op: "Tensor::contiguous_bytes",
                expected: None,
                found: Some(BackendId::Metal)
            }
        ),
        "{err}"
    );
    assert_eq!(
        err.to_string(),
        "Tensor::contiguous_bytes: tensor is on Metal device, expected host"
    );
    // The dtype is checked before placement.
    let err = f.to_u32_vec().unwrap_err();
    assert!(
        matches!(
            err,
            OjasError::Dtype {
                op: "Tensor::to_u32_vec",
                ..
            }
        ),
        "{err}"
    );
    // After a readback the bytes decode bit-exact.
    let host = f.to_host(&budget).unwrap();
    assert_eq!(bits(&host.to_f32_vec().unwrap()), SPECIAL_BITS);
    let u = Tensor::from_device(buf, &[16], DType::U32, &budget).unwrap();
    assert_eq!(
        u.narrow(8, &[2], &[1])
            .unwrap()
            .to_host(&budget)
            .unwrap()
            .to_u32_vec()
            .unwrap(),
        SPECIAL_BITS[2..4]
    );
}

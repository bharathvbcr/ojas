//! The operand contract of `BackendModule::dequant_affine_bf16`: `dequant_affine_bf16_dims`, the
//! one public owner of the checked size arithmetic (`AffineSpec::geometry`), and the host
//! implementation `dequantize_affine_bf16_tensors` that the CPU backend calls.

use ojas_core::{
    dequant_affine_bf16_dims, dequantize_affine_bf16, dequantize_affine_bf16_tensors, AffineDims,
    AffineGeometry, AffineQuantized, AffineSpec, Budget, DType, DeviceBuffer, OjasError, Tensor,
};
use std::any::Any;
use std::sync::Arc;

const OP: &str = "dequant_affine_bf16";

fn big() -> Budget {
    Budget::new(1 << 30)
}

fn words_per_row(spec: AffineSpec, groups: usize) -> usize {
    groups * spec.words_per_group()
}

/// A valid `(packed, scales, biases)` triple: `rows` rows of `groups` groups, scale 1, bias 0.
fn triple(spec: AffineSpec, rows: usize, groups: usize, b: &Budget) -> (Tensor, Tensor, Tensor) {
    (
        Tensor::zeros(&[rows, words_per_row(spec, groups)], DType::U32, b).unwrap(),
        Tensor::from_bf16_bits(&vec![0x3F80; rows * groups], &[rows, groups], b).unwrap(),
        Tensor::zeros(&[rows, groups], DType::Bf16, b).unwrap(),
    )
}

fn shape_err(r: Result<AffineDims, OjasError>, needle: &str) {
    match r {
        Err(OjasError::Shape { op, detail }) => {
            assert_eq!(op, OP);
            assert!(detail.contains(needle), "{detail:?} lacks {needle:?}");
        }
        other => panic!("expected Shape containing {needle:?}, got {other:?}"),
    }
}

#[test]
fn dims_accepts_q8_and_q4_triples_and_returns_rows_cols() {
    let b = big();
    for spec in [AffineSpec::q8(), AffineSpec::q4()] {
        let (p, s, z) = triple(spec, 3, 5, &b);
        let dims = dequant_affine_bf16_dims(&p, &s, &z, spec).unwrap();
        assert_eq!(
            dims,
            AffineDims {
                rows: 3,
                cols: 5 * 64,
                groups: 5,
                words: words_per_row(spec, 5),
            }
        );
    }
    assert_eq!(AffineSpec::q8().words_per_group(), 16);
    assert_eq!(AffineSpec::q4().words_per_group(), 8);
}

#[test]
fn dims_refuses_each_malformed_triple() {
    let b = big();
    let spec = AffineSpec::q8();
    let (p, s, z) = triple(spec, 2, 2, &b);
    let t = |shape: &[usize], dtype| Tensor::zeros(shape, dtype, &b).unwrap();

    // dtype, in argument order, naming the expected one
    for (bad_p, bad_s, bad_z, expected, got) in [
        (
            t(&[2, 32], DType::F32),
            s.clone(),
            z.clone(),
            DType::U32,
            DType::F32,
        ),
        (
            t(&[2, 32], DType::Bf16),
            s.clone(),
            z.clone(),
            DType::U32,
            DType::Bf16,
        ),
        (
            p.clone(),
            t(&[2, 2], DType::F32),
            z.clone(),
            DType::Bf16,
            DType::F32,
        ),
        (
            p.clone(),
            s.clone(),
            t(&[2, 2], DType::F16),
            DType::Bf16,
            DType::F16,
        ),
    ] {
        match dequant_affine_bf16_dims(&bad_p, &bad_s, &bad_z, spec) {
            Err(OjasError::Dtype {
                op,
                expected: e,
                got: g,
            }) => {
                assert_eq!((op, e, g), (OP, expected, got));
            }
            other => panic!("expected Dtype, got {other:?}"),
        }
    }

    // rank
    shape_err(
        dequant_affine_bf16_dims(&t(&[64], DType::U32), &s, &z, spec),
        "rank 1 != 2",
    );
    shape_err(
        dequant_affine_bf16_dims(&p, &t(&[2, 2, 1], DType::Bf16), &z, spec),
        "rank 3 != 2",
    );
    shape_err(
        dequant_affine_bf16_dims(&p, &s, &t(&[4], DType::Bf16), spec),
        "rank 1 != 2",
    );
    // a zero extent in any operand
    shape_err(
        dequant_affine_bf16_dims(&t(&[0, 32], DType::U32), &s, &z, spec),
        "empty tensor",
    );
    shape_err(
        dequant_affine_bf16_dims(&p, &t(&[2, 0], DType::Bf16), &z, spec),
        "empty tensor",
    );
    // rows disagree between the parts
    shape_err(
        dequant_affine_bf16_dims(&p, &t(&[3, 2], DType::Bf16), &z, spec),
        "rows disagree",
    );
    shape_err(
        dequant_affine_bf16_dims(&p, &s, &t(&[1, 2], DType::Bf16), spec),
        "rows disagree",
    );
    // scales and biases differ
    shape_err(
        dequant_affine_bf16_dims(&p, &s, &t(&[2, 3], DType::Bf16), spec),
        "differ",
    );
    // the packed width is not groups * words_per_group
    shape_err(
        dequant_affine_bf16_dims(&t(&[2, 31], DType::U32), &s, &z, spec),
        "need 32",
    );
    shape_err(
        dequant_affine_bf16_dims(&t(&[2, 33], DType::U32), &s, &z, spec),
        "need 32",
    );
}

/// A q4 triple presented as q8 is refused, and the refusal names the width its words imply.
#[test]
fn q4_triple_read_as_q8_is_refused_naming_four_bits() {
    let b = big();
    let (p, s, z) = triple(AffineSpec::q4(), 2, 3, &b);
    dequant_affine_bf16_dims(&p, &s, &z, AffineSpec::q4()).unwrap();
    shape_err(
        dequant_affine_bf16_dims(&p, &s, &z, AffineSpec::q8()),
        "implies 4-bit codes",
    );
    let (p, s, z) = triple(AffineSpec::q8(), 2, 3, &b);
    shape_err(
        dequant_affine_bf16_dims(&p, &s, &z, AffineSpec::q4()),
        "implies 8-bit codes",
    );
    // A width that is neither: the detail says so rather than inventing one.
    let odd = Tensor::zeros(&[2, 3 * 5], DType::U32, &b).unwrap();
    shape_err(
        dequant_affine_bf16_dims(&odd, &s, &z, AffineSpec::q8()),
        "no supported code width",
    );
}

#[test]
fn geometry_refuses_size_overflow_without_allocating() {
    let spec = AffineSpec::q8();
    let before = big();
    for (rows, cols) in [
        (usize::MAX, 64),
        (usize::MAX / 2 + 1, 64),
        (usize::MAX / 64, 64 * 64),
        (2, usize::MAX / 2 + 1 - (usize::MAX / 2 + 1) % 64),
    ] {
        match spec.geometry(OP, rows, cols) {
            Err(OjasError::OutOfRange { op, detail }) => {
                assert_eq!(op, OP);
                assert!(detail.contains("overflows usize"), "{detail}");
            }
            other => panic!("[{rows}, {cols}]: expected OutOfRange, got {other:?}"),
        }
    }
    // bf16 output bytes (rows * cols * 2) overflow while rows * cols does not.
    let half = usize::MAX / 2 + 1;
    assert!(matches!(
        spec.geometry(OP, 1, half - half % 64),
        Err(OjasError::OutOfRange { .. })
    ));
    assert_eq!(before.live_bytes().unwrap(), 0, "geometry charged a budget");
    // A valid one.
    assert_eq!(
        spec.geometry(OP, 4, 128).unwrap(),
        AffineGeometry {
            groups: 2,
            words: 32
        }
    );
    assert_eq!(
        AffineSpec::q4().geometry(OP, 4, 128).unwrap(),
        AffineGeometry {
            groups: 2,
            words: 16
        }
    );
    // Shape refusals precede the overflow checks.
    assert!(matches!(
        spec.geometry(OP, 0, 64),
        Err(OjasError::Shape { .. })
    ));
    assert!(matches!(
        spec.geometry(OP, usize::MAX, 65),
        Err(OjasError::Shape { .. })
    ));
}

#[test]
fn from_parts_and_dims_agree_on_malformed_geometry() {
    let b = big();
    let spec = AffineSpec::q8();
    let (p, s, z) = triple(spec, 1, 1, &b);
    // from_parts reports `geometry`'s own error, verbatim, under its own op name.
    for (rows, cols) in [
        (0, 64),
        (1, 0),
        (1, 65),
        (usize::MAX, 64),
        (1, usize::MAX - 63),
    ] {
        let want = spec
            .geometry("AffineQuantized::from_parts", rows, cols)
            .unwrap_err();
        let got = AffineQuantized::from_parts(spec, rows, cols, p.clone(), s.clone(), z.clone())
            .unwrap_err();
        assert_eq!(format!("{got:?}"), format!("{want:?}"), "[{rows}, {cols}]");
        // And the validator reports the same variant for the same geometry under its name.
        let dims_err = spec.geometry(OP, rows, cols).unwrap_err();
        assert_eq!(
            std::mem::discriminant(&dims_err),
            std::mem::discriminant(&want)
        );
    }
}

// --- the host implementation --------------------------------------------------------------

#[test]
fn tensors_path_equals_the_quantized_path_and_is_charged_before_it_writes() {
    let b = big();
    let spec = AffineSpec::q4();
    let rows = 2;
    let groups = 2;
    let words: Vec<u32> = (0..rows * words_per_row(spec, groups))
        .map(|i| (i as u32).wrapping_mul(0x9E37_79B1))
        .collect();
    let p = Tensor::from_u32(&words, &[rows, words_per_row(spec, groups)], &b).unwrap();
    let s = Tensor::from_bf16_bits(&[0x3F80, 0x3E80, 0x4000, 0x3F00], &[rows, groups], &b).unwrap();
    let z = Tensor::from_bf16_bits(&[0, 0x3F80, 0xBF80, 0], &[rows, groups], &b).unwrap();
    let live = b.live_bytes().unwrap();
    let got = dequantize_affine_bf16_tensors(&p, &s, &z, spec, &b).unwrap();
    assert_eq!(got.shape(), [rows, groups * 64]);
    assert_eq!(got.dtype(), DType::Bf16);
    assert_eq!(
        b.live_bytes().unwrap(),
        live + (rows * groups * 64 * 2) as u64
    );
    let q = AffineQuantized::from_parts(spec, rows, groups * 64, p, s, z).unwrap();
    let want = dequantize_affine_bf16(&q, &b).unwrap();
    assert_eq!(got.bf16_slice().unwrap(), want.bf16_slice().unwrap());
}

/// F1 amendment: the refusal is on the output, so finite scale and bias whose affine map
/// overflows are `NonFinite` under the op's own name, and nothing stays charged.
#[test]
fn finite_inputs_whose_output_overflows_are_non_finite_and_leave_nothing_charged() {
    let b = big();
    let spec = AffineSpec::q8();
    // scale bf16::MAX, code 255, bias bf16::MAX: finite parts, the result overflows bf16.
    let p = Tensor::from_u32(&[u32::MAX; 16], &[1, 16], &b).unwrap();
    let s = Tensor::from_bf16_bits(&[0x7F7F], &[1, 1], &b).unwrap();
    let z = Tensor::from_bf16_bits(&[0x7F7F], &[1, 1], &b).unwrap();
    let live = b.live_bytes().unwrap();
    match dequantize_affine_bf16_tensors(&p, &s, &z, spec, &b) {
        Err(OjasError::NonFinite { op }) => assert_eq!(op, OP),
        other => panic!("expected NonFinite, got {other:?}"),
    }
    assert_eq!(b.live_bytes().unwrap(), live);
    // NaN and infinite scales, and an infinite bias, with a zero code (inf * 0 is NaN).
    let zero = Tensor::zeros(&[1, 16], DType::U32, &b).unwrap();
    let plain = Tensor::from_bf16_bits(&[0x3F80], &[1, 1], &b).unwrap();
    let none = Tensor::zeros(&[1, 1], DType::Bf16, &b).unwrap();
    for bad in [0x7FC0u16, 0x7F80, 0xFF80] {
        let bad_t = Tensor::from_bf16_bits(&[bad], &[1, 1], &b).unwrap();
        for (s, z) in [(&bad_t, &none), (&plain, &bad_t)] {
            match dequantize_affine_bf16_tensors(&zero, s, z, spec, &b) {
                Err(OjasError::NonFinite { op }) => assert_eq!(op, OP, "{bad:#x}"),
                other => panic!("{bad:#x}: expected NonFinite, got {other:?}"),
            }
        }
    }
    assert_eq!(b.live_bytes().unwrap(), live);
}

#[test]
fn a_budget_one_byte_short_refuses_with_nothing_charged() {
    let spec = AffineSpec::q8();
    let probe = big();
    let _operands = triple(spec, 2, 2, &probe);
    let held = probe.live_bytes().unwrap();
    let need = (2 * 2 * 64 * 2) as u64;

    let short = Budget::new(held + need - 1);
    let (p, s, z) = triple(spec, 2, 2, &short);
    assert_eq!(short.live_bytes().unwrap(), held);
    assert!(matches!(
        dequantize_affine_bf16_tensors(&p, &s, &z, spec, &short),
        Err(OjasError::CapacityExceeded { .. })
    ));
    assert_eq!(short.live_bytes().unwrap(), held, "a refusal charged");

    let tight = Budget::new(held + need);
    let (p, s, z) = triple(spec, 2, 2, &tight);
    let out = dequantize_affine_bf16_tensors(&p, &s, &z, spec, &tight).unwrap();
    assert_eq!(tight.live_bytes().unwrap(), held + need);
    drop(out);
    assert_eq!(tight.live_bytes().unwrap(), held);
}

#[derive(Debug)]
struct Resident;

impl DeviceBuffer for Resident {
    fn backend(&self) -> ojas_core::BackendId {
        ojas_core::BackendId::Metal
    }
    fn byte_len(&self) -> usize {
        128
    }
    fn read_bytes(&self, _offset: usize, len: usize) -> Result<Vec<u8>, OjasError> {
        Ok(vec![0; len])
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[test]
fn non_contiguous_and_device_operands_are_refused_before_anything_is_charged() {
    let b = big();
    let spec = AffineSpec::q8();
    let (p, s, z) = triple(spec, 2, 2, &b);
    let live = b.live_bytes().unwrap();
    // Permuted strides over a bf16 `[2, 2]` allocation: same shape, not one run of bytes.
    let strided = s.view(&[2, 2], &[1, 2], 0).unwrap();
    let r = dequantize_affine_bf16_tensors(&p, &strided, &z, spec, &b);
    assert!(r.is_err(), "a permuted view was read as contiguous");
    // A device-resident operand is a Placement refusal, never a readback.
    let device = Tensor::from_device(Arc::new(Resident), &[2, 2], DType::Bf16, &b).unwrap();
    let live_with_device = b.live_bytes().unwrap();
    let reads = b.device_readbacks();
    assert!(matches!(
        dequantize_affine_bf16_tensors(&p, &device, &z, spec, &b),
        Err(OjasError::Placement { .. })
    ));
    assert_eq!(b.device_readbacks(), reads);
    assert_eq!(b.live_bytes().unwrap(), live_with_device);
    drop(device);
    assert_eq!(b.live_bytes().unwrap(), live);
}

#[test]
fn rejects_the_malformed_triples_with_the_validators_error_and_nothing_charged() {
    let b = big();
    let spec = AffineSpec::q8();
    let (p, s, z) = triple(spec, 2, 2, &b);
    let wrong = Tensor::zeros(&[2, 31], DType::U32, &b).unwrap();
    let live = b.live_bytes().unwrap();
    let want = dequant_affine_bf16_dims(&wrong, &s, &z, spec).unwrap_err();
    let got = dequantize_affine_bf16_tensors(&wrong, &s, &z, spec, &b).unwrap_err();
    assert_eq!(format!("{got:?}"), format!("{want:?}"));
    // Even under a zero budget the shape error wins (shape first).
    let zero = Budget::new(0);
    let got = dequantize_affine_bf16_tensors(&wrong, &s, &z, spec, &zero).unwrap_err();
    assert_eq!(format!("{got:?}"), format!("{want:?}"));
    assert_eq!(b.live_bytes().unwrap(), live);
    drop(p);
}

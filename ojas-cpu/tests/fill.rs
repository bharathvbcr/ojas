//! `Backend::fill` on the CPU: torch's `full`, made on the backend with no
//! host upload.
//!
//! - An `F32` fill stores the value's exact bits (both zeros, subnormals and
//!   the extremes); a `U32` fill an integer in `0..2^32`; a `Bf16` fill only
//!   a value a bf16 holds, as its high half. Anything else is refused, never
//!   rounded or truncated.
//! - A NaN or infinity is refused (no op takes or stores one), after the
//!   shape rules.
//! - The output is the only charge, and a refused call leaves the budget as
//!   it was.
//! - A filled tensor feeds the other ops like any other, and under bf16
//!   autocast nothing is rounded.

use ojas_core::{Autocast, AutocastMode, Backend, Budget, DType, Numerics, OjasError, Tensor};
use ojas_cpu::CpuBackend;

mod common;
use common::{assert_capacity, assert_nonfinite, assert_range, assert_shape, bits, SplitMix64};

fn backend() -> CpuBackend {
    CpuBackend::with_threads(Budget::new(1 << 32), 2)
        .unwrap()
        .with_numerics(Numerics::Exact)
}

fn v(t: &Tensor) -> Vec<f32> {
    t.to_f32_vec().unwrap()
}

#[test]
fn an_f32_fill_stores_the_exact_bits_of_every_finite_value() {
    let cpu = backend();
    let values = [
        0.0f32,
        -0.0,
        1.0,
        -1.5,
        0.1,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        1e-45,
        -1e-45,
        f32::MAX,
        f32::MIN,
        f32::from_bits(0x3f80_0001),
    ];
    for shape in [&[][..], &[1], &[5], &[2, 3, 4], &[1, 2, 1, 3]] {
        let n: usize = shape.iter().product();
        for value in values {
            let out = cpu.fill(shape, DType::F32, value).unwrap();
            assert_eq!(out.shape(), shape);
            assert_eq!(out.dtype(), DType::F32);
            let got = v(&out);
            assert_eq!(got.len(), n);
            assert!(
                got.iter().all(|g| g.to_bits() == value.to_bits()),
                "{shape:?} {value:e}"
            );
        }
    }
    // Random bit patterns, so every exponent and mantissa shape is covered.
    let mut rng = SplitMix64(1);
    for _ in 0..500 {
        let value = f32::from_bits(rng.next_u64() as u32);
        if !value.is_finite() {
            continue;
        }
        let got = v(&cpu.fill(&[3, 2], DType::F32, value).unwrap());
        assert!(
            got.iter().all(|g| g.to_bits() == value.to_bits()),
            "{value:e}"
        );
    }
}

#[test]
fn a_u32_fill_stores_the_integer() {
    let cpu = backend();
    for value in [
        0.0f32,
        1.0,
        255.0,
        65_536.0,
        16_777_216.0,
        2_147_483_648.0,
        4_294_967_040.0,
    ] {
        let out = cpu.fill(&[2, 3], DType::U32, value).unwrap();
        assert_eq!((out.dtype(), out.shape()), (DType::U32, &[2usize, 3][..]));
        assert_eq!(out.u32_slice().unwrap(), [value as u32; 6], "{value}");
    }
    // -0.0 is the integer 0.
    let out = cpu.fill(&[2], DType::U32, -0.0).unwrap();
    assert_eq!(out.u32_slice().unwrap(), [0, 0]);
    // Not an integer, negative, or past 2^32: refused, never truncated.
    for value in [
        0.5f32,
        -0.5,
        -1.0,
        1e-45,
        4_294_967_296.0,
        1e30,
        f32::MAX,
        f32::MIN,
    ] {
        assert_range(cpu.fill(&[2], DType::U32, value));
    }
}

#[test]
fn a_bf16_fill_stores_the_high_half_of_a_value_a_bf16_holds() {
    let cpu = backend();
    for value in [
        1.0f32,
        -2.5,
        0.15625,
        0.0,
        -0.0,
        f32::from_bits(0x7f7f_0000),
        f32::from_bits(0x0080_0000),
    ] {
        let out = cpu.fill(&[2, 2], DType::Bf16, value).unwrap();
        assert_eq!((out.dtype(), out.shape()), (DType::Bf16, &[2usize, 2][..]));
        let want = (value.to_bits() >> 16) as u16;
        assert_eq!(out.bf16_slice().unwrap(), [want; 4], "{value:e}");
        // The widened value is the one asked for.
        assert_eq!(
            f32::from_bits(u32::from(want) << 16).to_bits(),
            value.to_bits()
        );
    }
    // A value a bf16 cannot hold is refused, not rounded.
    for value in [
        0.1f32,
        1.0 + f32::EPSILON,
        f32::from_bits(0x3f80_0001),
        f32::from_bits(0x3f80_8000),
        f32::MAX,
        1e-45,
    ] {
        assert_range(cpu.fill(&[2], DType::Bf16, value));
    }
}

#[test]
fn a_non_finite_value_is_refused_for_every_dtype_after_the_shape() {
    let cpu = backend();
    for dtype in [DType::F32, DType::Bf16, DType::U32] {
        for value in [
            f32::NAN,
            f32::from_bits(0x7f80_0001),
            f32::from_bits(0xffc0_1234),
            f32::INFINITY,
            f32::NEG_INFINITY,
        ] {
            assert_nonfinite(cpu.fill(&[2, 2], dtype, value));
            assert_nonfinite(cpu.fill(&[], dtype, value));
            // The shape is judged first.
            assert_shape(cpu.fill(&[2, 0], dtype, value));
        }
    }
}

#[test]
fn shapes_dtypes_and_budgets_are_refused_with_the_neighbours_variants() {
    let cpu = backend();
    let before = cpu.budget().live_bytes().unwrap();
    // An empty extent anywhere.
    for shape in [&[0][..], &[2, 0], &[0, 3], &[2, 0, 4]] {
        for dtype in [DType::F32, DType::Bf16, DType::U32] {
            assert_shape(cpu.fill(shape, dtype, 1.0));
        }
    }
    // An element or byte count that overflows.
    for shape in [
        &[1usize << 40, 1 << 40][..],
        &[usize::MAX],
        &[1 << 62],
        &[usize::MAX, 2],
    ] {
        assert_range(cpu.fill(shape, DType::F32, 1.0));
        assert_range(cpu.fill(shape, DType::U32, 1.0));
    }
    // Two-byte elements: a bf16 fill of 2^62 elements has a byte count that
    // fits, so it reaches the allocator and the budget.
    assert_capacity(cpu.fill(&[1 << 62], DType::Bf16, 1.0));
    // A zero extent is a shape error for every dtype, F16 included.
    assert_shape(cpu.fill(&[2, 0], DType::F16, 1.0));
    // F16 has no host storage here.
    match cpu.fill(&[2], DType::F16, 1.0) {
        Err(OjasError::Dtype { op, expected, got }) => {
            assert_eq!((op, expected, got), ("fill", DType::F32, DType::F16))
        }
        other => panic!("expected Dtype, got {other:?}"),
    }
    // Over budget: the charge is refused and nothing stays charged.
    let tight = CpuBackend::with_threads(Budget::new(64), 1).unwrap();
    for dtype in [DType::F32, DType::U32, DType::Bf16] {
        assert_capacity(tight.fill(&[33], dtype, 1.0));
    }
    assert_eq!(tight.budget().live_bytes().unwrap(), 0);
    assert_eq!(cpu.budget().live_bytes().unwrap(), before);
}

#[test]
fn the_output_is_the_only_charge_and_is_released_when_dropped() {
    let cpu = backend();
    let before = cpu.budget().live_bytes().unwrap();
    let f = cpu.fill(&[4, 4], DType::F32, 2.0).unwrap();
    let u = cpu.fill(&[4, 4], DType::U32, 2.0).unwrap();
    let b = cpu.fill(&[4, 4], DType::Bf16, 2.0).unwrap();
    assert_eq!(cpu.budget().live_bytes().unwrap(), before + 64 + 64 + 32);
    drop((f, u, b));
    assert_eq!(cpu.budget().live_bytes().unwrap(), before);
}

#[test]
fn a_filled_tensor_feeds_other_ops_like_a_host_one() {
    // The group_norm weight and bias SpikeTrack uploads from host Vecs.
    let cpu = backend();
    let mut rng = SplitMix64(2);
    let shape = [2usize, 8, 5, 6];
    let x_data = rng.vec(shape.iter().product(), 2.0);
    let host = |value: f32| Tensor::from_f32(&[value; 8], &[8], cpu.budget()).unwrap();
    let x = Tensor::from_f32(&x_data, &shape, cpu.budget()).unwrap();
    let want = cpu
        .group_norm_forward(&x, &host(1.0), &host(0.0), 4, 1e-5)
        .unwrap();
    let ones = cpu.fill(&[8], DType::F32, 1.0).unwrap();
    let zeros = cpu.fill(&[8], DType::F32, 0.0).unwrap();
    let got = cpu.group_norm_forward(&x, &ones, &zeros, 4, 1e-5).unwrap();
    assert_eq!(bits(&v(&got)), bits(&v(&want)));
    // And as a zero block joined to data.
    let pad = cpu.fill(&[2, 3, 4], DType::F32, 0.0).unwrap();
    let data = Tensor::from_f32(&[0.5; 16], &[2, 2, 4], cpu.budget()).unwrap();
    let joined = cpu.concat(&[&data, &pad], 1).unwrap();
    assert_eq!(joined.shape(), [2, 5, 4]);
    assert_eq!(v(&cpu.slice(&joined, 1, 2, 3).unwrap()), [0.0; 24]);
}

#[test]
fn a_large_fill_is_every_element() {
    let cpu = backend();
    let out = cpu.fill(&[1, 256, 288, 288], DType::F32, -0.0).unwrap();
    assert!(v(&out).iter().all(|g| g.to_bits() == (-0.0f32).to_bits()));
}

#[test]
fn a_bf16_region_adds_no_rounding_and_forwards_refusals() {
    let wrapped = Autocast::new(backend());
    let _region = wrapped.autocast_region(AutocastMode::Bf16).unwrap();
    // A value no bf16 holds is stored as it is, and the output is not tagged.
    let fine = f32::from_bits(0x3f80_0001);
    let out = wrapped.fill(&[3], DType::F32, fine).unwrap();
    assert_eq!(bits(&v(&out)), bits(&[fine; 3]));
    assert!(!out.bf16_exact());
    // The compute dtype does not make a fill round: a bf16 fill still
    // refuses a value a bf16 cannot hold.
    assert_range(wrapped.fill(&[3], DType::Bf16, fine));
    assert!(wrapped.fill(&[3], DType::Bf16, 1.0).is_ok());
    assert_nonfinite(wrapped.fill(&[3], DType::F32, f32::NAN));
    assert_shape(wrapped.fill(&[0], DType::F32, 1.0));
}

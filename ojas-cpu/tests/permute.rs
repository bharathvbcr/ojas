//! `CpuBackend::permute`: a contiguous copy with axes reordered, moved bit
//! for bit, validated by `permute_output_shape` and charged to the budget.

use ojas_core::{inverse_permutation, Backend, Budget, DType, OjasError, Tensor, MAX_PERMUTE_RANK};
use ojas_cpu::CpuBackend;

mod common;
use common::{assert_capacity, assert_nonfinite, assert_shape, bits, f32t, u32t, SplitMix64};

fn wide() -> CpuBackend {
    CpuBackend::new(Budget::new(1 << 22))
}

/// Row-major strides of `shape`.
fn strides(shape: &[usize]) -> Vec<usize> {
    let mut out = vec![1usize; shape.len()];
    for axis in (0..shape.len().saturating_sub(1)).rev() {
        out[axis] = out[axis + 1] * shape[axis + 1];
    }
    out
}

/// Scalar reference: walk the output index in row-major order and read the
/// input at the permuted index.
fn reference(data: &[f32], shape: &[usize], dims: &[usize]) -> Vec<f32> {
    let out_shape: Vec<usize> = dims.iter().map(|&d| shape[d]).collect();
    let n: usize = out_shape.iter().product();
    let in_strides = strides(shape);
    let out_strides = strides(&out_shape);
    (0..n)
        .map(|flat| {
            let mut src = 0;
            for (axis, &stride) in out_strides.iter().enumerate() {
                let idx = (flat / stride) % out_shape[axis];
                src += idx * in_strides[dims[axis]];
            }
            data[src]
        })
        .collect()
}

fn all_permutations(n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return vec![Vec::new()];
    }
    let mut out = Vec::new();
    for rest in all_permutations(n - 1) {
        for slot in 0..=rest.len() {
            let mut p = rest.clone();
            p.insert(slot, n - 1);
            out.push(p);
        }
    }
    out
}

/// Distinct values plus the bit patterns a lossy move would disturb:
/// negative zero, a subnormal and the extremes.
fn tricky(n: usize) -> Vec<f32> {
    let mut rng = SplitMix64(0x5eed_0001);
    let mut data = rng.vec(n, 3.0);
    let specials = [
        -0.0f32,
        f32::from_bits(1),
        f32::MIN_POSITIVE,
        f32::MAX,
        f32::MIN,
    ];
    for (slot, value) in data.iter_mut().zip(specials) {
        *slot = value;
    }
    data
}

#[test]
fn every_rank_four_permutation_matches_the_scalar_reference_bit_for_bit() {
    let cpu = wide();
    let shape = [2usize, 3, 4, 5];
    let data = tricky(shape.iter().product());
    let x = f32t(&cpu, &data, &shape);
    let perms = all_permutations(4);
    assert_eq!(perms.len(), 24);
    for dims in &perms {
        let y = cpu.permute(&x, dims).unwrap();
        let want_shape: Vec<usize> = dims.iter().map(|&d| shape[d]).collect();
        assert_eq!(y.shape(), want_shape.as_slice(), "dims {dims:?}");
        assert!(y.is_contiguous().unwrap(), "dims {dims:?}");
        assert_eq!(y.dtype(), DType::F32);
        assert_eq!(y.device(), None);
        let got = y.to_f32_vec().unwrap();
        assert_eq!(
            bits(&got),
            bits(&reference(&data, &shape, dims)),
            "dims {dims:?}"
        );
    }
}

#[test]
fn inverse_permutation_round_trips_bit_for_bit() {
    let cpu = wide();
    for (shape, seed) in [
        (vec![2usize, 3, 4, 5], 1u64),
        (vec![3, 1, 2], 2),
        (vec![2, 2, 1, 3, 2], 3),
    ] {
        let n: usize = shape.iter().product();
        let data = SplitMix64(seed).vec(n, 10.0);
        let x = f32t(&cpu, &data, &shape);
        for dims in all_permutations(shape.len()) {
            let y = cpu.permute(&x, &dims).unwrap();
            let back = cpu.permute(&y, &inverse_permutation(&dims)).unwrap();
            assert_eq!(back.shape(), shape.as_slice());
            assert_eq!(bits(&back.to_f32_vec().unwrap()), bits(&data), "{dims:?}");
        }
    }
}

#[test]
fn rank_zero_and_rank_one_are_copies() {
    let cpu = wide();
    let scalar = f32t(&cpu, &[-0.0], &[]);
    let y = cpu.permute(&scalar, &[]).unwrap();
    assert!(y.shape().is_empty());
    assert_eq!(bits(&y.to_f32_vec().unwrap()), bits(&[-0.0]));

    let row = f32t(&cpu, &[1.0, 2.5, -3.0], &[3]);
    let y = cpu.permute(&row, &[0]).unwrap();
    assert_eq!(y.shape(), &[3]);
    assert_eq!(y.to_f32_vec().unwrap(), vec![1.0, 2.5, -3.0]);
}

#[test]
fn zero_size_axes_permute_to_an_empty_tensor() {
    let cpu = wide();
    let empty = Tensor::zeros(&[2, 0, 3], DType::F32, cpu.budget()).unwrap();
    let y = cpu.permute(&empty, &[2, 0, 1]).unwrap();
    assert_eq!(y.shape(), &[3, 2, 0]);
    assert!(y.to_f32_vec().unwrap().is_empty());
    let back = cpu.permute(&y, &inverse_permutation(&[2, 0, 1])).unwrap();
    assert_eq!(back.shape(), &[2, 0, 3]);
}

#[test]
fn a_contiguous_window_with_an_offset_is_read_from_its_offset() {
    let cpu = wide();
    let base: Vec<f32> = (0..30).map(|v| v as f32).collect();
    let parent = f32t(&cpu, &base, &[30]);
    // Elements 6..30 viewed as [2, 3, 4], contiguous, starting 24 bytes in.
    let window = parent.narrow(6 * 4, &[2, 3, 4], &[12, 4, 1]).unwrap();
    assert!(window.is_contiguous().unwrap());
    let y = cpu.permute(&window, &[2, 0, 1]).unwrap();
    let want = reference(&base[6..], &[2, 3, 4], &[2, 0, 1]);
    assert_eq!(bits(&y.to_f32_vec().unwrap()), bits(&want));
}

#[test]
fn invalid_axes_strided_views_and_wrong_dtypes_are_refused() {
    let cpu = wide();
    let x = f32t(&cpu, &[0.5; 24], &[2, 3, 4]);
    assert_shape(cpu.permute(&x, &[0, 1]));
    assert_shape(cpu.permute(&x, &[0, 1, 2, 3]));
    assert_shape(cpu.permute(&x, &[0, 0, 1]));
    assert_shape(cpu.permute(&x, &[0, 1, 3]));
    let deep = vec![1usize; MAX_PERMUTE_RANK + 1];
    let too_deep = f32t(&cpu, &[1.0], &deep);
    let dims: Vec<usize> = (0..deep.len()).collect();
    assert_shape(cpu.permute(&too_deep, &dims));

    // Every other column of a [2, 4] matrix: legal metadata, not contiguous.
    let base = f32t(&cpu, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]);
    let strided = base.view(&[2, 2], &[4, 2], 0).unwrap();
    assert!(!strided.is_contiguous().unwrap());
    assert_shape(cpu.permute(&strided, &[1, 0]));

    let ids = u32t(&cpu, &[1, 2, 3, 4], &[2, 2]);
    match cpu.permute(&ids, &[1, 0]) {
        Err(OjasError::Dtype {
            expected: DType::F32,
            got: DType::U32,
            ..
        }) => {}
        other => panic!("expected Dtype, got {other:?}"),
    }
}

/// Every CPU op refuses a non-finite operand; permute keeps that contract
/// rather than passing NaN through to the next op.
#[test]
fn non_finite_input_is_refused() {
    let cpu = wide();
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let x = f32t(&cpu, &[1.0, bad, 3.0, 4.0], &[2, 2]);
        assert_nonfinite(cpu.permute(&x, &[1, 0]));
    }
}

#[test]
fn the_output_is_charged_and_a_full_budget_refuses_it() {
    let inputs = Budget::new(1 << 20);
    let x = Tensor::from_f32(&[0.25; 64], &[4, 4, 4], &inputs).unwrap();
    // Room for 63 of the 64 output values: refused, nothing left charged.
    let tight = CpuBackend::new(Budget::new(63 * 4));
    assert_capacity(tight.permute(&x, &[2, 1, 0]));
    assert_eq!(tight.budget().live_bytes().unwrap(), 0);
    // Exactly the output: accepted, and the output holds the charge.
    let exact = CpuBackend::new(Budget::new(64 * 4));
    let y = exact.permute(&x, &[2, 1, 0]).unwrap();
    assert_eq!(exact.budget().live_bytes().unwrap(), 64 * 4);
    drop(y);
    assert_eq!(exact.budget().live_bytes().unwrap(), 0);
}

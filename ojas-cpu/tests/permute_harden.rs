//! Gather correctness for `CpuBackend::permute`.
//!
//! The reference below is a scalar index walk. It does not call the op, so a
//! gather that drops or reorders a run fails here even if it is internally
//! consistent. Thread count must not change bits.

use std::thread;

use ojas_core::{Backend, Budget, DType, Tensor};
use ojas_cpu::CpuBackend;

mod common;
use common::{assert_capacity, assert_nonfinite, assert_shape, bits, f32t, SplitMix64};

fn cpu(threads: usize) -> CpuBackend {
    CpuBackend::with_threads(Budget::new(1 << 28), threads).unwrap()
}

fn strides(shape: &[usize]) -> Vec<usize> {
    let mut out = vec![1usize; shape.len()];
    for axis in (0..shape.len().saturating_sub(1)).rev() {
        out[axis] = out[axis + 1] * shape[axis + 1];
    }
    out
}

/// Output row-major, one source index per element. Independent of `permute`.
fn reference(data: &[f32], shape: &[usize], dims: &[usize]) -> Vec<f32> {
    let out_shape: Vec<usize> = dims.iter().map(|&d| shape[d]).collect();
    let n: usize = out_shape.iter().copied().product();
    if n == 0 {
        return Vec::new();
    }
    let in_strides = strides(shape);
    let out_strides = strides(&out_shape);
    let mut out = Vec::with_capacity(n);
    for flat in 0..n {
        let mut src = 0usize;
        for (axis, &stride) in out_strides.iter().enumerate() {
            let idx = (flat / stride) % out_shape[axis];
            src += idx * in_strides[dims[axis]];
        }
        out.push(data[src]);
    }
    out
}

fn check(threads: usize, shape: &[usize], dims: &[usize], data: &[f32]) {
    let backend = cpu(threads);
    if threads > 1 {
        backend.start_workers().unwrap();
    }
    let x = f32t(&backend, data, shape);
    let y = backend
        .permute(&x, dims)
        .unwrap_or_else(|err| panic!("threads {threads} {shape:?} {dims:?}: {err}"));
    let want_shape: Vec<usize> = dims.iter().map(|&d| shape[d]).collect();
    assert_eq!(y.shape(), want_shape.as_slice(), "{shape:?} {dims:?}");
    assert!(y.is_contiguous().unwrap());
    assert_eq!(y.dtype(), DType::F32);
    assert_eq!(
        bits(&y.to_f32_vec().unwrap()),
        bits(&reference(data, shape, dims)),
        "threads {threads} {shape:?} {dims:?}"
    );
}

#[test]
fn gather_matches_scalar_reference_at_one_and_six_threads() {
    let shapes: &[&[usize]] = &[&[1, 1024, 12, 64], &[2, 3, 5, 7], &[4, 1, 1, 1]];
    let perms: &[&[usize]] = &[&[0, 1, 2, 3], &[0, 2, 1, 3], &[3, 2, 1, 0], &[0, 1, 3, 2]];
    for threads in [1usize, 6] {
        for shape in shapes {
            let n: usize = shape.iter().product();
            let mut data = SplitMix64(0x0a5_0000 + n as u64).vec(n, 3.0);
            data[0] = -0.0;
            if n > 64 {
                data[64] = -0.0;
            }
            if n > 3 {
                data[n / 2] = f32::from_bits(1);
                data[n - 1] = f32::MIN;
            }
            for dims in perms {
                check(threads, shape, dims, &data);
            }
        }
        check(threads, &[17], &[0], &SplitMix64(9).vec(17, 1.0));
        check(threads, &[], &[], &[-0.0]);
    }
}

/// Input `[1, 1024, 12, 64]` index 64 is head 1 of token 0. After
/// `(0, 2, 1, 3)` that lane is output index `1 * 1024 * 64`.
#[test]
fn negative_zero_at_input_64_lands_at_output_65536() {
    let shape = [1usize, 1024, 12, 64];
    let mut data = vec![1.0f32; 1024 * 768];
    data[64] = -0.0;
    let dims = [0usize, 2, 1, 3];
    for threads in [1usize, 6] {
        let backend = cpu(threads);
        if threads > 1 {
            backend.start_workers().unwrap();
        }
        let x = f32t(&backend, &data, &shape);
        let y = backend.permute(&x, &dims).unwrap();
        let out = y.to_f32_vec().unwrap();
        assert_eq!(
            out[65536].to_bits(),
            (-0.0f32).to_bits(),
            "threads {threads}"
        );
        assert_eq!(bits(&out), bits(&reference(&data, &shape, &dims)));
    }
}

#[test]
fn empty_axis_is_an_empty_tensor() {
    for threads in [1usize, 6] {
        let backend = cpu(threads);
        let empty = Tensor::zeros(&[2, 0, 4], DType::F32, backend.budget()).unwrap();
        let y = backend.permute(&empty, &[2, 0, 1]).unwrap();
        assert_eq!(y.shape(), &[4, 2, 0]);
        assert!(y.to_f32_vec().unwrap().is_empty());
        let back = backend.permute(&y, &[1, 2, 0]).unwrap();
        assert_eq!(back.shape(), &[2, 0, 4]);
        assert!(back.to_f32_vec().unwrap().is_empty());
    }
}

#[test]
fn nan_and_pos_inf_are_refused_without_a_budget_charge() {
    for threads in [1usize, 6] {
        for bad in [f32::NAN, f32::INFINITY] {
            let backend = cpu(threads);
            let mut data = SplitMix64(4).vec(24, 1.0);
            data[17] = bad;
            let x = f32t(&backend, &data, &[2, 3, 4]);
            let live = backend.budget().live_bytes().unwrap();
            assert_nonfinite(backend.permute(&x, &[0, 2, 1]));
            assert_eq!(backend.budget().live_bytes().unwrap(), live);
        }
    }
}

#[test]
fn non_contiguous_input_is_refused() {
    let backend = cpu(1);
    let base = f32t(&backend, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]);
    let strided = base.view(&[2, 2], &[4, 2], 0).unwrap();
    assert!(!strided.is_contiguous().unwrap());
    let live = backend.budget().live_bytes().unwrap();
    assert_shape(backend.permute(&strided, &[1, 0]));
    assert_eq!(backend.budget().live_bytes().unwrap(), live);
}

#[test]
fn budget_too_small_for_the_output_is_capacity_exceeded() {
    let inputs = Budget::new(1 << 20);
    let data = [0.25f32; 64];
    let x = Tensor::from_f32(&data, &[4, 4, 4], &inputs).unwrap();
    let tight = CpuBackend::new(Budget::new(63 * 4));
    let live = tight.budget().live_bytes().unwrap();
    assert_eq!(live, 0);
    assert_capacity(tight.permute(&x, &[2, 1, 0]));
    assert_eq!(tight.budget().live_bytes().unwrap(), live);

    // The input lives on the same budget the output would charge.
    let shared = Budget::new((64 + 63) * 4);
    let held = Tensor::from_f32(&data, &[4, 4, 4], &shared).unwrap();
    let backend = CpuBackend::new(shared.clone());
    let live = backend.budget().live_bytes().unwrap();
    assert_capacity(backend.permute(&held, &[2, 1, 0]));
    assert_eq!(backend.budget().live_bytes().unwrap(), live);
    drop(held);
}

#[test]
fn negative_zero_bits_survive_an_identity_permute() {
    for threads in [1usize, 6] {
        let backend = cpu(threads);
        let data = [-0.0f32, 1.0, -0.0, 2.5, f32::from_bits((-0.0f32).to_bits())];
        let x = f32t(&backend, &data, &[5]);
        let y = backend.permute(&x, &[0]).unwrap();
        assert_eq!(bits(&y.to_f32_vec().unwrap()), bits(&data));

        let scalar = f32t(&backend, &[-0.0], &[]);
        let y = backend.permute(&scalar, &[]).unwrap();
        assert_eq!(y.to_f32_vec().unwrap()[0].to_bits(), (-0.0f32).to_bits());
    }
}

#[test]
fn two_concurrent_permutes_match_the_reference() {
    let backend = CpuBackend::with_threads(Budget::new(1 << 30), 4).unwrap();
    let shape = [1usize, 1024, 12, 64];
    let n: usize = shape.iter().product();
    let data = SplitMix64(0xC0FFEE).vec(n, 2.0);
    let dims = [0usize, 2, 1, 3];
    let want = reference(&data, &shape, &dims);
    let x = f32t(&backend, &data, &shape);
    let left = thread::spawn({
        let backend = backend.clone();
        let x = x.clone();
        move || backend.permute(&x, &dims).unwrap().to_f32_vec().unwrap()
    });
    let right = thread::spawn({
        let backend = backend.clone();
        let x = x.clone();
        move || backend.permute(&x, &dims).unwrap().to_f32_vec().unwrap()
    });
    let a = left.join().expect("left permute");
    let b = right.join().expect("right permute");
    assert_eq!(bits(&a), bits(&want));
    assert_eq!(bits(&b), bits(&want));
}

/// Above `PARALLEL_MIN_FLOATS` in `layout.rs`, six threads take the scoped
/// path and one thread does not. Both must match the scalar reference.
#[test]
fn parallel_gather_matches_the_scalar_reference() {
    let shape = [8usize, 1024, 12, 64];
    let n: usize = shape.iter().product();
    let data = SplitMix64(0x91A1).vec(n, 1.0);
    let dims = [0usize, 2, 1, 3];
    for threads in [1usize, 6] {
        check(threads, &shape, &dims, &data);
    }
}

/// Head count above the precomputed-base table, so offsets are unraveled.
#[test]
fn large_head_unravel_matches_the_scalar_reference() {
    let shape = [2usize, 2, 2, 2, 16, 16, 16, 32];
    let n: usize = shape.iter().product();
    let dims: Vec<usize> = (0..shape.len()).rev().collect();
    let data = SplitMix64(0x11EA0).vec(n, 1.0);
    for threads in [1usize, 6] {
        check(threads, &shape, &dims, &data);
    }
}

/// Unravel and the scoped path at once. The serial large-head case is under
/// the parallel cutoff, and the parallel nanolab case precomputes its few
/// heads. This shape is at the cutoff and has more than 65,536 heads.
#[test]
fn parallel_unravel_matches_the_scalar_reference() {
    let shape = [2usize, 2, 32, 32, 32, 32];
    let n: usize = shape.iter().product();
    assert!(n >= 1 << 22, "this case must cross the parallel cutoff");
    let dims: Vec<usize> = (0..shape.len()).rev().collect();
    let data = SplitMix64(0x0A11).vec(n, 1.0);
    for threads in [1usize, 6] {
        check(threads, &shape, &dims, &data);
    }
}

fn time_ms(rounds: u32, mut body: impl FnMut()) -> f64 {
    for _ in 0..4 {
        body();
    }
    let start = std::time::Instant::now();
    for _ in 0..rounds {
        body();
    }
    start.elapsed().as_secs_f64() * 1e3 / f64::from(rounds)
}

/// Wall time of `(0, 2, 1, 3)` on `[1, 1024, 12, 64]`. Not part of the suite.
#[test]
#[ignore]
fn bench_nanolab_permute() {
    let shape = [1usize, 1024, 12, 64];
    let n: usize = shape.iter().product();
    let data = SplitMix64(1).vec(n, 1.0);
    let dims = [0usize, 2, 1, 3];
    let rounds = 80u32;
    for threads in [1usize, 6] {
        let backend = cpu(threads);
        let x = f32t(&backend, &data, &shape);
        let per_ms = time_ms(rounds, || drop(backend.permute(&x, &dims).unwrap()));
        let bytes = (n * 2 * 4) as f64;
        let gbps = bytes / (per_ms * 1e-3) / 1e9;
        eprintln!("nanolab threads {threads}: {per_ms:.4} ms/call  {gbps:.1} GB/s (read+write)");
    }
    let big = [8usize, 1024, 12, 64];
    let big_n: usize = big.iter().product();
    let big_data = SplitMix64(2).vec(big_n, 1.0);
    for threads in [1usize, 6] {
        let backend = CpuBackend::with_threads(Budget::new(1 << 30), threads).unwrap();
        let x = f32t(&backend, &big_data, &big);
        let per_ms = time_ms(20, || drop(backend.permute(&x, &dims).unwrap()));
        eprintln!("[8,1024,12,64] threads {threads}: {per_ms:.4} ms/call");
    }
}

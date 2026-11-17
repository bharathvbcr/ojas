//! Helpers shared by the CPU integration tests. Each test binary uses a subset.
#![allow(dead_code)]

use ojas_core::{OjasError, Tensor};
use ojas_cpu::CpuBackend;

/// Deterministic splitmix64 so randomized tests need no dependency.
// Shared by `mod common;`; binaries that never name it must not warn.
#[allow(unused_imports)]
pub use ojas_test_support::SplitMix64;

pub fn f32t(cpu: &CpuBackend, data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_f32(data, shape, cpu.budget()).unwrap()
}

pub fn u32t(cpu: &CpuBackend, data: &[u32], shape: &[usize]) -> Tensor {
    Tensor::from_u32(data, shape, cpu.budget()).unwrap()
}

pub fn assert_shape<T: std::fmt::Debug>(result: Result<T, OjasError>) {
    match result {
        Err(OjasError::Shape { .. }) => {}
        other => panic!("expected Shape, got {other:?}"),
    }
}

pub fn assert_nonfinite<T: std::fmt::Debug>(result: Result<T, OjasError>) {
    match result {
        Err(OjasError::NonFinite { .. }) => {}
        other => panic!("expected NonFinite, got {other:?}"),
    }
}

pub fn assert_capacity<T: std::fmt::Debug>(result: Result<T, OjasError>) {
    match result {
        Err(OjasError::CapacityExceeded { .. }) => {}
        other => panic!("expected CapacityExceeded, got {other:?}"),
    }
}

pub fn assert_range<T: std::fmt::Debug>(result: Result<T, OjasError>) {
    match result {
        Err(OjasError::OutOfRange { .. }) => {}
        other => panic!("expected OutOfRange, got {other:?}"),
    }
}

pub fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().copied().map(f32::to_bits).collect()
}

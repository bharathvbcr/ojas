//! Helpers shared by the CPU integration tests. Each test binary uses a subset.
#![allow(dead_code)]

use ojas_core::{OjasError, Tensor};
use ojas_cpu::CpuBackend;

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

/// Deterministic splitmix64 so randomized tests need no dependency.
pub struct SplitMix64(pub u64);

impl SplitMix64 {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `[-1, 1)`.
    pub fn unit(&mut self) -> f32 {
        let mantissa = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        2.0 * mantissa - 1.0
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    pub fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| scale * self.unit()).collect()
    }
}

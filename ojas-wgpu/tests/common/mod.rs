//! Shared by the wgpu integration tests.
#![allow(dead_code)]

use std::sync::OnceLock;

use ojas_core::{Backend, BackendId, Budget, Numerics, Tensor};
use ojas_cpu::CpuBackend;
use ojas_wgpu::WgpuBackend;

/// Relative to the largest reference magnitude in the tensor.
pub const TOL: f64 = 1.0e-4;

pub fn gpu() -> &'static WgpuBackend {
    static GPU: OnceLock<WgpuBackend> = OnceLock::new();
    GPU.get_or_init(|| {
        WgpuBackend::open(Budget::new(8 << 30)).expect("wgpu adapter; a missing GPU fails the test")
    })
}

/// A backend of its own, for tests that inject faults: the fault word is
/// shared by every caller of one backend.
pub fn fresh() -> WgpuBackend {
    WgpuBackend::open(Budget::new(1 << 30)).expect("wgpu adapter")
}

/// A backend on the shared device with a budget of its own. Readbacks are
/// counted per budget tree (`Budget::device_readbacks`), so a "no readback"
/// assertion on this backend cannot see another test's downloads, and a
/// download made through it on any thread is still counted.
pub fn own() -> WgpuBackend {
    WgpuBackend::with_context(gpu().context().clone(), Budget::new(8 << 30))
}

/// `(calls, bytes)` of device-to-host copies charged to `g`'s budget.
pub fn readbacks(g: &WgpuBackend) -> (u64, u64) {
    g.budget().device_readbacks()
}

/// Parity reference. CpuBackend defaults to `Numerics::Fast`; the references
/// here are the exact path.
pub fn cpu() -> CpuBackend {
    CpuBackend::new(Budget::new(8 << 30)).with_numerics(Numerics::Exact)
}

pub fn host_budget() -> &'static Budget {
    static B: OnceLock<Budget> = OnceLock::new();
    B.get_or_init(|| Budget::new(8 << 30))
}

pub fn data(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^= z >> 31;
            ((z >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

pub fn host(seed: u64, shape: &[usize]) -> Tensor {
    let n = shape.iter().product();
    Tensor::from_f32(&data(seed, n), shape, host_budget()).unwrap()
}

pub fn host_u32(values: &[u32], shape: &[usize]) -> Tensor {
    Tensor::from_u32(values, shape, host_budget()).unwrap()
}

pub fn up(t: &Tensor) -> Tensor {
    let d = gpu().upload(t).unwrap();
    assert_eq!(d.device(), Some(BackendId::Wgpu));
    d
}

pub fn down(t: &Tensor) -> Vec<f32> {
    assert_eq!(
        t.device(),
        Some(BackendId::Wgpu),
        "op returned a host tensor"
    );
    gpu().download(t).unwrap().to_f32_vec().unwrap()
}

pub fn close(name: &str, got: &Tensor, want: &Tensor) {
    assert_eq!(got.shape(), want.shape(), "{name}: shape");
    let got = down(got);
    let want = want.to_f32_vec().unwrap();
    close_vec(name, &got, &want);
}

pub fn close_vec(name: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    let scale = want.iter().fold(1.0f64, |m, v| m.max(f64::from(v.abs())));
    let mut worst = (0.0f64, 0usize);
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(g.is_finite(), "{name}[{i}] = {g}, cpu {w}");
        let err = (f64::from(*g) - f64::from(*w)).abs();
        if err > worst.0 {
            worst = (err, i);
        }
    }
    assert!(
        worst.0 <= TOL * scale,
        "{name}: |err| {:.3e} at {} (gpu {} cpu {}) > {:.1e} * {scale:.3}",
        worst.0,
        worst.1,
        got[worst.1],
        want[worst.1],
        TOL
    );
}

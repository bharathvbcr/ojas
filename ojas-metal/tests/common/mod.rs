#![allow(dead_code)]

use ojas_core::{Backend, Budget, Numerics, OjasError, Tensor};
use ojas_cpu::CpuBackend;
use ojas_metal::MetalBackend;

pub const GIB: u64 = 1 << 30;

pub fn metal() -> MetalBackend {
    match MetalBackend::new(Budget::new(8 * GIB)) {
        Ok(m) => m,
        Err(e) => panic!("Metal device unavailable: {e:?}"),
    }
}

/// The CPU reference under its reference contract. `CpuBackend::new`
/// defaults to `Numerics::Fast`; parity is judged against `Exact`.
pub fn cpu() -> CpuBackend {
    CpuBackend::new(Budget::new(8 * GIB)).with_numerics(Numerics::Exact)
}

/// Deterministic values in `[-scale, scale)`.
pub fn values(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = (s >> 40) as f32 / (1u64 << 24) as f32;
            (2.0 * u - 1.0) * scale
        })
        .collect()
}

pub fn ids(n: usize, seed: u64, limit: u32) -> Vec<u32> {
    let mut s = seed.wrapping_mul(0x2545_F491_4F6C_DD1D).wrapping_add(7);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % u64::from(limit)) as u32
        })
        .collect()
}

pub fn host(data: &[f32], shape: &[usize]) -> Tensor {
    match Tensor::from_f32(data, shape, &Budget::new(8 * GIB)) {
        Ok(t) => t,
        Err(e) => panic!("host tensor: {e:?}"),
    }
}

pub fn host_u32(data: &[u32], shape: &[usize]) -> Tensor {
    match Tensor::from_u32(data, shape, &Budget::new(8 * GIB)) {
        Ok(t) => t,
        Err(e) => panic!("host tensor: {e:?}"),
    }
}

pub fn rand(shape: &[usize], seed: u64, scale: f32) -> Tensor {
    host(&values(shape.iter().product(), seed, scale), shape)
}

pub fn up(m: &MetalBackend, t: &Tensor) -> Tensor {
    match m.upload(t) {
        Ok(d) => d,
        Err(e) => panic!("upload: {e:?}"),
    }
}

pub fn down(t: &Tensor) -> Vec<f32> {
    let h = match t.to_host(&Budget::new(8 * GIB)) {
        Ok(h) => h,
        Err(e) => panic!("download: {e:?}"),
    };
    match h.to_f32_vec() {
        Ok(v) => v,
        Err(e) => panic!("decode: {e:?}"),
    }
}

pub fn ok<T>(what: &str, r: Result<T, OjasError>) -> T {
    match r {
        Ok(v) => v,
        Err(e) => panic!("{what}: {e:?}"),
    }
}

/// Elementwise `|a - b| <= atol + rtol * |b|`, with the worst index in the
/// failure message.
pub fn close(what: &str, got: &[f32], want: &[f32], atol: f32, rtol: f32) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mut worst = (0usize, 0.0f32);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(g.is_finite(), "{what}: non-finite at {i}: {g}");
        let err = (g - w).abs() - (atol + rtol * w.abs());
        if err > worst.1 {
            worst = (i, err);
        }
    }
    let i = worst.0;
    assert!(
        worst.1 <= 0.0,
        "{what}: index {i} got {} want {} (atol {atol}, rtol {rtol})",
        got[i],
        want[i]
    );
}

/// Run the same op on CPU (host tensors) and Metal (uploaded copies) and
/// compare every f32 output.
pub fn same_tensor(what: &str, got: &Tensor, want: &Tensor, atol: f32, rtol: f32) {
    assert_eq!(got.shape(), want.shape(), "{what}: shape");
    assert_eq!(
        got.device(),
        Some(ojas_core::BackendId::Metal),
        "{what}: residency"
    );
    close(what, &down(got), &ok(what, want.to_f32_vec()), atol, rtol);
}

/// A device-detected fault under the deferred-fault contract
/// (`docs/metal-deferred-faults.md` §6): the op returns `Ok`, the next
/// `sync` reports `NonFinite` naming `op`, and the sync after that is `Ok`.
/// Returns the op's value so callers can check in-place state afterwards.
pub fn deferred<T: std::fmt::Debug>(
    m: &MetalBackend,
    what: &str,
    r: Result<T, OjasError>,
    op: &str,
) -> T {
    let v = match r {
        Ok(v) => v,
        Err(e) => panic!("{what}: the op must record, not refuse: {e:?}"),
    };
    match m.sync() {
        Err(OjasError::NonFinite { op: got }) => assert_eq!(got, op, "{what}: sync named another op"),
        other => panic!("{what}: sync must report NonFinite {{ {op} }}: {other:?}"),
    }
    let again = m.sync();
    assert!(again.is_ok(), "{what}: a fault is reported once: {again:?}");
    v
}

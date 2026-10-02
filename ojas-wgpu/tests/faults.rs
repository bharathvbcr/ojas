//! The deferred non-finite contract (module docs of `ojas_wgpu::backend`):
//! an op that produces a non-finite value returns `Ok`, and the very next
//! `sync`, `download` or `clip_grad_norm` returns `NonFinite` naming the
//! first op, in recording order, that faulted. Each test opens its own
//! backend because the fault word belongs to the context.

mod common;

use common::*;
use ojas_core::{AdamWConfig, Backend, DType, OjasError, Tensor};
use ojas_wgpu::WgpuBackend;

fn nonfinite_op(r: Result<impl std::fmt::Debug, OjasError>) -> &'static str {
    match r {
        Err(OjasError::NonFinite { op }) => op,
        other => panic!("expected NonFinite, got {other:?}"),
    }
}

fn with_nan(g: &WgpuBackend, seed: u64, shape: &[usize]) -> Tensor {
    let n = shape.iter().product();
    let mut v = data(seed, n);
    v[n / 2] = f32::NAN;
    g.upload(&Tensor::from_f32(&v, shape, host_budget()).unwrap())
        .unwrap()
}

#[test]
fn the_first_faulting_op_is_named_not_the_lowest_bit() {
    // Pre-fix the report named the lowest set bit: linear_forward (bit 2)
    // even though mul_forward (bit 18) faulted first and fed it.
    let g = &fresh();
    g.sync().unwrap();
    let a = with_nan(g, 4000, &[4, 8]);
    let b = g.upload(&host(4001, &[4, 8])).unwrap();
    let w = g.upload(&host(4002, &[3, 8])).unwrap();
    let y = g.mul_forward(&a, &b).unwrap();
    let _z = g.linear_forward(&y, &w).unwrap();
    assert_eq!(nonfinite_op(g.sync()), "mul_forward");
    g.sync().expect("reported once");

    // A later fault is named after itself, not after the cleared one.
    let inf = g
        .upload(&Tensor::from_f32(&[f32::INFINITY, 1.0], &[2], host_budget()).unwrap())
        .unwrap();
    let _ = g.silu_forward(&inf).unwrap();
    let _ = g.mul_forward(&a, &b).unwrap();
    assert_eq!(nonfinite_op(g.sync()), "silu_forward");
    g.sync().unwrap();
}

#[test]
fn adamw_fault_is_named_at_every_kind_of_sync_point() {
    let g = &fresh();
    let cfg = AdamWConfig::nanolab(1e-2, 0.1);
    let state = |seed| {
        (
            g.upload(&host(seed, &[64])).unwrap(),
            g.upload(&Tensor::zeros(&[64], DType::F32, host_budget()).unwrap())
                .unwrap(),
            g.upload(&Tensor::zeros(&[64], DType::F32, host_budget()).unwrap())
                .unwrap(),
        )
    };
    let bad = with_nan(g, 4100, &[64]);

    // sync
    let (mut p, mut m, mut v) = state(4101);
    g.adamw_step(&mut p, &bad, &mut m, &mut v, 0, cfg)
        .expect("deferred: the step itself returns Ok");
    assert_eq!(nonfinite_op(g.sync()), "adamw_step");

    // download of an unrelated tensor
    let other = g.upload(&host(4102, &[3])).unwrap();
    g.adamw_step(&mut p, &bad, &mut m, &mut v, 0, cfg).unwrap();
    assert_eq!(nonfinite_op(g.download(&other)), "adamw_step");
    g.download(&other).expect("reported once");

    // clip_grad_norm refuses before scaling anything
    g.adamw_step(&mut p, &bad, &mut m, &mut v, 0, cfg).unwrap();
    let clean = host(4103, &[16]);
    let mut grads = vec![g.upload(&clean).unwrap()];
    assert_eq!(
        nonfinite_op(g.clip_grad_norm(&mut grads, 1e-3)),
        "adamw_step"
    );
    close_vec(
        "clip left the gradient alone",
        &g.download(&grads[0]).unwrap().to_f32_vec().unwrap(),
        &clean.to_f32_vec().unwrap(),
    );

    // The parameter never moved: every step was rolled back on the device.
    let p0 = host(4101, &[64]).to_f32_vec().unwrap();
    let p_now = g.download(&p).unwrap().to_f32_vec().unwrap();
    assert_eq!(
        p_now.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        p0.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
    g.sync().unwrap();
}

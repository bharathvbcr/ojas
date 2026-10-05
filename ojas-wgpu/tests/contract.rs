//! Contract checks that only use the `Backend` trait, so they compile against
//! any revision of `WgpuBackend`. Each one failed on the round-trip backend:
//! it panicked on rank-0 input, returned host tensors, and refused backward ops.

use std::panic::{catch_unwind, AssertUnwindSafe};

use ojas_core::{Backend, BackendId, Budget, OjasError, Tensor, RMS_NORM_EPS};
use ojas_wgpu::WgpuBackend;

fn gpu() -> WgpuBackend {
    WgpuBackend::open(Budget::new(1 << 30)).expect("wgpu adapter; a missing GPU fails the test")
}

fn up(gpu: &WgpuBackend, data: &[f32], shape: &[usize]) -> Tensor {
    let host = Tensor::from_f32(data, shape, gpu.budget()).unwrap();
    gpu.upload(&host).unwrap()
}

type ShapeCall<'a> = (&'a str, Box<dyn Fn() -> Result<(), OjasError> + 'a>);
type SyncCall<'a> = (&'a str, &'a dyn Fn() -> Result<(), OjasError>);

#[test]
fn malformed_shapes_are_errors_not_panics() {
    let g = gpu();
    let scalar = up(&g, &[1.0], &[]);
    let w = up(&g, &[1.0; 6], &[2, 3]);
    let x = up(&g, &[1.0; 8], &[2, 4]);
    let gy = up(&g, &[1.0; 4], &[2, 2]);
    let calls: Vec<ShapeCall> = vec![
        (
            "linear_backward rank 0",
            Box::new(|| g.linear_backward(&scalar, &w, &scalar).map(|_| ())),
        ),
        (
            "linear_backward mismatched weight",
            Box::new(|| g.linear_backward(&x, &w, &gy).map(|_| ())),
        ),
        (
            "linear_forward mismatched weight",
            Box::new(|| g.linear_forward(&x, &w).map(|_| ())),
        ),
        (
            "rms_norm_forward rank 0",
            Box::new(|| {
                g.rms_norm_forward(&scalar, &scalar, RMS_NORM_EPS)
                    .map(|_| ())
            }),
        ),
        (
            "rms_norm_forward mismatched weight",
            Box::new(|| g.rms_norm_forward(&x, &w, RMS_NORM_EPS).map(|_| ())),
        ),
    ];
    for (name, call) in calls {
        let got = catch_unwind(AssertUnwindSafe(call));
        match got {
            Ok(Err(OjasError::Shape { .. })) => {}
            Ok(other) => panic!("{name}: expected a Shape error, got {other:?}"),
            Err(_) => panic!("{name}: panicked"),
        }
    }
}

#[test]
fn outputs_stay_on_the_device_until_download() {
    let g = gpu();
    let x = up(&g, &[0.5, -0.25, 1.0, 0.0, 0.2, -0.7], &[2, 3]);
    let w = up(&g, &[0.1, 0.2, -0.3, 0.4, 0.0, 0.5], &[2, 3]);
    let before = g.budget().device_readbacks();
    let y = g.linear_forward(&x, &w).unwrap();
    let s = g.silu_forward(&y).unwrap();
    assert_eq!(
        g.budget().device_readbacks(),
        before,
        "an op read back to the host"
    );
    assert_eq!(y.device(), Some(BackendId::Wgpu));
    assert_eq!(s.device(), Some(BackendId::Wgpu));
    let host = g.download(&s).unwrap();
    assert_eq!(g.budget().device_readbacks().0, before.0 + 1);
    assert_eq!(host.to_f32_vec().unwrap().len(), 4);
}

#[test]
fn backward_ops_are_implemented() {
    let g = gpu();
    let x = up(&g, &[0.5, -0.25, 1.0, 0.75], &[2, 2]);
    let w = up(&g, &[1.0, 0.5], &[2]);
    let gy = up(&g, &[0.1, 0.2, 0.3, 0.4], &[2, 2]);
    g.rms_norm_backward(&x, &w, &gy, RMS_NORM_EPS).unwrap();
    g.silu_backward(&x, &gy).unwrap();
    g.mul_backward(&x, &gy, &gy).unwrap();
    g.residual_add_backward(&x, &gy, &gy).unwrap();
}

/// `sync` through the trait, the way a generic training loop (or an
/// `Arc<WgpuBackend>` / `&dyn Backend` caller) reaches it. Before wgpu
/// overrode `Backend::sync`, every one of these returned the trait default
/// `Ok(())` and the deferred fault was dropped.
#[test]
fn the_trait_sync_reports_the_deferred_fault() {
    fn trait_sync<B: Backend + ?Sized>(b: &B) -> Result<(), OjasError> {
        b.sync()
    }
    let g = std::sync::Arc::new(gpu());
    let callers: [SyncCall; 3] = [
        ("generic", &|| trait_sync(&*g)),
        ("dyn", &|| (&*g as &dyn Backend).sync()),
        ("arc", &|| g.sync()),
    ];
    for (name, sync) in callers {
        let x = up(&g, &[1.0, f32::NAN, 0.5, 2.0], &[2, 2]);
        let _s = g.silu_forward(&x).unwrap();
        match sync() {
            Err(OjasError::NonFinite { op }) => assert!(op.contains("silu"), "{name}: {op}"),
            other => panic!("{name}: the deferred NaN was dropped: {other:?}"),
        }
        sync().unwrap_or_else(|e| panic!("{name}: reported once, then clear: {e:?}"));
    }
}

/// Not only waits and maps: a call that fails creating or mapping a buffer
/// on the destroyed device must also say the device is gone. On CI's
/// llvmpipe a sync after the loss failed at "upload map" without naming it.
#[test]
fn every_call_after_a_loss_names_it() {
    let g = gpu();
    let x = up(&g, &[1.0; 4], &[2, 2]);
    g.sync().unwrap();
    g.context().device().destroy();
    let host = Tensor::from_f32(&[2.0; 4], &[2, 2], g.budget()).unwrap();
    let results = [
        ("upload", g.upload(&host).map(drop)),
        ("silu", g.silu_forward(&x).map(drop)),
        ("sync", g.sync()),
    ];
    for (name, result) in results {
        match result {
            Err(OjasError::Backend { detail, .. }) => {
                assert!(
                    detail.contains("device lost"),
                    "{name}: loss not named: {detail}"
                )
            }
            other => panic!("{name} on a destroyed device: {other:?}"),
        }
    }
}

#[test]
fn the_trait_sync_reports_a_lost_device() {
    let g = std::sync::Arc::new(gpu());
    let x = up(&g, &[1.0; 4], &[2, 2]);
    let _y = g.silu_forward(&x).unwrap();
    g.sync().unwrap();
    g.context().device().destroy();
    // Twice: the first call is the one whose wait wgpu refuses before it
    // delivers the loss; the error must still say the device is gone.
    for _ in 0..2 {
        match g.sync() {
            Err(OjasError::Backend { detail, .. }) => {
                assert!(detail.contains("device lost"), "loss not named: {detail}")
            }
            other => panic!("a destroyed device synced clean: {other:?}"),
        }
    }
}

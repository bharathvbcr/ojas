//! `Probe<B>` is `B` with counters and faults, not a different backend: a
//! trainer test that wraps `CpuBackend` in it must run the CPU's overrides,
//! not the trait defaults.

mod common;
#[path = "reach/mod.rs"]
mod reach;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use common::Probe;
use ojas_core::{forward_backend_methods, Backend, BackendId, Budget, OjasError, Tensor};
use ojas_cpu::CpuBackend;
use reach::{call, op_names, Reach};

/// Forwards every method to `inner` except
/// `causal_sdpa_backward_recompute`, which `Spy` overrides.
macro_rules! all_but_recompute {
    (causal_sdpa_backward_recompute, { $($item:tt)* }) => {};
    ($name:ident, { $($item:tt)* }) => {
        $($item)*
    };
}

/// A backend whose `causal_sdpa_backward_recompute` is an override: it
/// counts the call and answers with its own marker error.
struct Spy {
    inner: CpuBackend,
    recomputes: AtomicUsize,
}

impl Backend for Spy {
    ojas_core::with_backend_methods!(
        forward_backend_methods,
        { self, self.inner, none, all_but_recompute }
    );
    fn causal_sdpa_backward_recompute(
        &self,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
        _: Option<usize>,
    ) -> Result<(Tensor, Tensor, Tensor), OjasError> {
        self.recomputes.fetch_add(1, Ordering::SeqCst);
        Err(OjasError::Backend {
            id: BackendId::Cpu,
            detail: "spy:recompute".into(),
        })
    }
}

#[test]
fn the_probe_reaches_the_inner_recompute_override() {
    let spy = Spy {
        inner: CpuBackend::new(Budget::new(1 << 20)),
        recomputes: AtomicUsize::new(0),
    };
    let probe = Probe::new(&spy);
    let t = Tensor::from_f32(&[1.0], &[1], &Budget::new(1 << 10)).unwrap();
    let err = probe
        .causal_sdpa_backward_recompute(&t, &t, &t, &t, None)
        .unwrap_err();
    assert!(err.to_string().contains("spy:recompute"), "{err}");
    assert_eq!(spy.recomputes.load(Ordering::SeqCst), 1);
    assert_eq!(probe.count("causal_sdpa_backward_recompute"), 1);
}

/// The CPU scales a uniquely owned gradient in place with no charge; the
/// trait default builds and charges a broadcast operand and a new product.
#[test]
fn the_probe_runs_the_cpu_scale_grad_in_place() {
    let probe = Probe::new(CpuBackend::new(Budget::new(1 << 20)));
    let mut grad = Tensor::from_f32(&[1.0, -2.0, 4.0, 0.5], &[2, 2], probe.budget()).unwrap();
    let live = probe.budget().live_bytes().unwrap();
    probe.budget().reset_peak();
    probe.scale_grad(&mut grad, 0.25).unwrap();
    assert_eq!(grad.to_f32_vec().unwrap(), vec![0.25, -0.5, 1.0, 0.125]);
    assert_eq!(
        probe.budget().peak_bytes(),
        live,
        "scaled in place, no charge"
    );
    assert_eq!(probe.count("scale_grad"), 1);
}

/// `bf16_operands` is the backend's, not the trait default's `false`.
#[test]
fn the_probe_reports_the_inner_bf16_operands() {
    let probe = Probe::new(CpuBackend::new(Budget::new(1 << 20)));
    assert!(probe.inner().bf16_operands());
    assert!(probe.bf16_operands());
}

/// Adversarial: every method `Probe` leaves to the trait default is a way a
/// probe-based test silently exercises something other than the backend.
/// `CpuBackend` implements the hybrid sigmoid; the probe must give what the
/// CPU gives, not the trait's Unsupported default.
#[test]
fn the_probe_runs_the_cpu_sigmoid_forward() {
    let probe = Probe::new(CpuBackend::new(Budget::new(1 << 20)));
    let x = Tensor::from_f32(&[0.0, 1.0, -2.0], &[3], probe.budget()).unwrap();
    let want = probe
        .inner()
        .sigmoid_forward(&x)
        .unwrap()
        .to_f32_vec()
        .unwrap();
    let got = probe
        .sigmoid_forward(&x)
        .expect("Probe<Cpu>::sigmoid_forward fell to the Unsupported trait default")
        .to_f32_vec()
        .unwrap();
    assert_eq!(got, want);
}

/// Card ft-fe7dcee6 criterion 4: every op reaches the wrapped backend
/// through `Probe` exactly once, and is counted by the probe. `Reach` answers
/// each op with an error naming it, so a method that is compiled out, answered
/// locally or forwarded to another op fails here, whatever the source looks
/// like (a source scan accepts `#[cfg(any())]` and a call inside a comment).
#[test]
fn probe_reaches_the_wrapped_backend_once_per_op() {
    let reach = Arc::new(Reach::new());
    let probe = Probe::new(&reach);
    for name in op_names() {
        let before = (reach.count(name), reach.total());
        let err = call(&probe, name)
            .unwrap()
            .expect_err("Reach answers every op with an error");
        assert!(
            err.to_string().contains(&format!("reached:{name}")),
            "Probe/{name} did not reach the inner op: {err}"
        );
        assert_eq!(
            (reach.count(name), reach.total()),
            (before.0 + 1, before.1 + 1),
            "Probe/{name}: wrong inner call count"
        );
        assert_eq!(probe.count(name), 1, "Probe did not count {name}");
    }
}

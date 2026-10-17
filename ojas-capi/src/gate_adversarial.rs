//! Adversarial probes for the `Gated` wrapper (card ft-6fb212ce): a dropped
//! forward of `linear_backward_input` would silently run the trait default
//! (the full backward with dW computed and dropped) on the wrapper. The spy
//! overrides both entry points, so the count tells which one ran.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use ojas_core::{forward_backend_methods, Autocast, Backend, BackendId, Budget, OjasError, Tensor};
use ojas_cpu::CpuBackend;

use crate::gate::{CancelSlot, Gated};

macro_rules! all_but_linear_backward {
    (linear_backward_input, { $($item:tt)* }) => {};
    (linear_backward, { $($item:tt)* }) => {};
    ($name:ident, { $($item:tt)* }) => {
        $($item)*
    };
}

struct Spy {
    inner: CpuBackend,
    dx_only: AtomicUsize,
    full: AtomicUsize,
}

impl Spy {
    fn new() -> Self {
        Self {
            inner: CpuBackend::new(Budget::new(1 << 24)),
            dx_only: AtomicUsize::new(0),
            full: AtomicUsize::new(0),
        }
    }
}

impl Backend for Spy {
    ojas_core::with_backend_methods!(
        forward_backend_methods,
        { self, self.inner, none, all_but_linear_backward }
    );
    fn linear_backward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        self.full.fetch_add(1, Ordering::SeqCst);
        self.inner.linear_backward(input, weight, grad_output)
    }
    fn linear_backward_input(
        &self,
        input: &Tensor,
        weight: &Tensor,
        grad_output: &Tensor,
    ) -> Result<Tensor, OjasError> {
        self.dx_only.fetch_add(1, Ordering::SeqCst);
        self.inner.linear_backward_input(input, weight, grad_output)
    }
}

fn operands(b: &Budget) -> (Tensor, Tensor, Tensor) {
    let f = |n: usize, s: f32| {
        Tensor::from_f32(
            &(0..n).map(|i| s + i as f32 * 0.01).collect::<Vec<_>>(),
            &[n],
            b,
        )
    };
    let x = f(6, 0.1).unwrap().reshape(&[2, 3]).unwrap();
    let w = f(12, 0.2).unwrap().reshape(&[4, 3]).unwrap();
    let g = f(8, 0.3).unwrap().reshape(&[2, 4]).unwrap();
    (x, w, g)
}

#[test]
fn the_gated_dx_only_backward_reaches_the_inner_override_and_polls() {
    let slot = Arc::new(CancelSlot::default());
    let gated = Gated::new(Spy::new(), Arc::clone(&slot));
    let b = Budget::new(1 << 20);
    let (x, w, g) = operands(&b);
    gated.linear_backward_input(&x, &w, &g).unwrap();
    assert_eq!(gated.inner().dx_only.load(Ordering::SeqCst), 1);
    assert_eq!(
        gated.inner().full.load(Ordering::SeqCst),
        0,
        "the trait default ran on the wrapper instead of the inner override"
    );
    let armed = slot.arm(Box::new(|| Err("cancelled: Explicit".to_string())));
    assert!(gated.linear_backward_input(&x, &w, &g).is_err());
    assert_eq!(armed.tripped().as_deref(), Some("cancelled: Explicit"));
    assert_eq!(gated.inner().dx_only.load(Ordering::SeqCst), 1);
}

/// Forwards every method to `inner` except
/// `causal_sdpa_backward_recompute`, which `Recompute` overrides.
macro_rules! all_but_recompute {
    (causal_sdpa_backward_recompute, { $($item:tt)* }) => {};
    ($name:ident, { $($item:tt)* }) => {
        $($item)*
    };
}

/// A backend whose `causal_sdpa_backward_recompute` is an override (the
/// trait default composes the forward and the backward): it counts the call
/// and answers with its own marker error.
struct Recompute {
    inner: CpuBackend,
    calls: AtomicUsize,
}

impl Backend for Recompute {
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
        self.calls.fetch_add(1, Ordering::SeqCst);
        Err(OjasError::Backend {
            id: BackendId::Cpu,
            detail: "spy:recompute".into(),
        })
    }
}

/// Card ft-fe7dcee6 criterion 1: the override is reached through `&B`,
/// `Arc<B>`, `Autocast<B>`, `Gated<B>` and a stack of all four, each call
/// counted exactly once (the trait default on a wrapper would count none).
#[test]
fn the_recompute_override_is_reached_through_every_wrapper() {
    fn reach<W: Backend>(label: &str, wrapper: &W, spy: &Recompute) {
        let before = spy.calls.load(Ordering::SeqCst);
        let t = Tensor::from_f32(&[1.0], &[1], &Budget::new(1 << 10)).unwrap();
        let err = wrapper
            .causal_sdpa_backward_recompute(&t, &t, &t, &t, None)
            .unwrap_err();
        assert!(err.to_string().contains("spy:recompute"), "{label}: {err}");
        assert_eq!(
            spy.calls.load(Ordering::SeqCst),
            before + 1,
            "{label}: the override did not run exactly once"
        );
    }
    let spy = Arc::new(Recompute {
        inner: CpuBackend::new(Budget::new(1 << 20)),
        calls: AtomicUsize::new(0),
    });
    let slot = Arc::new(CancelSlot::default());
    reach("&B", &&*spy, &spy);
    reach("Arc<B>", &Arc::clone(&spy), &spy);
    reach("Autocast<B>", &Autocast::new(&*spy), &spy);
    reach("Gated<B>", &Gated::new(&*spy, Arc::clone(&slot)), &spy);
    let stack = Gated::new(Autocast::new(Arc::new(&*spy)), Arc::clone(&slot));
    reach("Gated<Autocast<Arc<&B>>>", &stack, &spy);
}

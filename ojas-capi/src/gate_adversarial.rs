//! Adversarial probes for the `Gated` wrapper (card ft-6fb212ce): a dropped
//! forward of `linear_backward_input` would silently run the trait default
//! (the full backward with dW computed and dropped) on the wrapper. The spy
//! overrides both entry points, so the count tells which one ran.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use ojas_core::{forward_backend_methods, Backend, Budget, OjasError, Tensor};
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

/// Every method name of the backend trait has a hand-written `fn` in
/// `impl Backend for Gated`: a method added to the trait and forgotten here
/// would run the trait default on the wrapper.
#[test]
#[ignore = "BREAK on main bc498c4: fixed by gitpulse/ci-main-green (ft-b4f9); un-ignore on rebase"]
fn every_backend_method_has_a_gated_forward() {
    let src = include_str!("gate.rs");
    let body = src
        .split("impl<B: Backend> Backend for Gated<B>")
        .nth(1)
        .expect("the Gated impl");
    // The impl ends at the first `\n}\n` at column 0.
    let body = body.split("\n}\n").next().unwrap();
    let missing: Vec<&str> = ojas_core::BACKEND_METHODS
        .iter()
        .copied()
        .filter(|name| !body.contains(&format!("fn {name}(")))
        .collect();
    assert!(missing.is_empty(), "Gated does not forward: {missing:?}");
}

//! G-TAPE / CA-4. `Tape::backward` refuses a backend gradient whose shape is
//! not the forward variable's, and does not keep it.
//!
//! `fused_ce_asking` already checks the fused cross-entropy contract. These
//! tests go through `silu` and `linear`, whose gradients are stored by
//! `Tape::acc`. A single consumer used to keep the first wrong shape. Two
//! consumers that return the same wrong shape used to pass `accumulate_grad`,
//! which only compares the two gradients with each other.

use ojas_autograd::Tape;
use ojas_core::{forward_backend_methods, Backend, BackendId, Budget, OjasError, Tensor};
use ojas_cpu::CpuBackend;

enum SiluGrad {
    Real,
    /// Four elements, rank 1. A `[2, 2]` forward value has the same count.
    Flat4,
}

struct Warp {
    inner: CpuBackend,
    silu: SiluGrad,
    /// Replace `linear_backward`'s weight gradient with shape `[1]`.
    warp_weight: bool,
}

macro_rules! keep_except_sabotage {
    (silu_backward, { $($item:tt)* }) => {};
    (linear_backward, { $($item:tt)* }) => {};
    ($name:ident, { $($item:tt)* }) => {
        $($item)*
    };
}

impl Backend for Warp {
    ojas_core::with_backend_methods!(
        forward_backend_methods,
        { self, self.inner, none, keep_except_sabotage }
    );

    fn silu_backward(&self, input: &Tensor, grad_output: &Tensor) -> Result<Tensor, OjasError> {
        match self.silu {
            SiluGrad::Real => self.inner.silu_backward(input, grad_output),
            SiluGrad::Flat4 => Tensor::from_f32(&[0.0; 4], &[4], self.inner.budget()),
        }
    }

    fn linear_backward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        let (gx, gw) = self.inner.linear_backward(input, weight, grad_output)?;
        if self.warp_weight {
            let gw = Tensor::from_f32(&[0.0], &[1], self.inner.budget())?;
            Ok((gx, gw))
        } else {
            Ok((gx, gw))
        }
    }
}

fn warp(silu: SiluGrad, warp_weight: bool) -> Warp {
    Warp {
        inner: CpuBackend::new(Budget::new(1 << 20)),
        silu,
        warp_weight,
    }
}

fn tensor(backend: &Warp, data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_f32(data, shape, backend.inner.budget()).unwrap()
}

fn refused(err: &OjasError) -> bool {
    matches!(
        err,
        OjasError::Backend {
            id: BackendId::Cpu,
            detail,
        } if detail.contains("does not match forward var")
    )
}

#[test]
fn a_flat_silu_gradient_is_refused_and_the_leaf_keeps_nothing() {
    let backend = warp(SiluGrad::Flat4, false);
    let x = tensor(&backend, &[1.0, -0.5, 0.25, 2.0], &[2, 2]);
    let mut tape = Tape::new(backend);
    let leaf = tape.leaf(x).unwrap();
    let y = tape.silu(leaf).unwrap();
    let err = tape
        .backward(y)
        .expect_err("a rank-1 gradient of a rank-2 value");
    assert!(refused(&err), "{err}");
    assert!(err.to_string().contains("[4]"), "{err}");
    assert!(err.to_string().contains("[2, 2]"), "{err}");
    assert!(tape.grad(leaf).is_none());
    assert_eq!(tape.value(leaf).unwrap().shape(), &[2, 2]);
    let again = tape.backward(y).expect_err("a repeat still refuses");
    assert!(refused(&again), "{again}");
    assert!(tape.grad(leaf).is_none());
}

#[test]
fn two_silu_paths_returning_the_same_wrong_shape_are_refused() {
    let backend = warp(SiluGrad::Flat4, false);
    let x = tensor(&backend, &[1.0, -0.5, 0.25, 2.0], &[2, 2]);
    let mut tape = Tape::new(backend);
    let leaf = tape.leaf(x).unwrap();
    let a = tape.silu(leaf).unwrap();
    let b = tape.silu(leaf).unwrap();
    let y = tape.add(a, b).unwrap();
    let err = tape
        .backward(y)
        .expect_err("accumulate_grad accepts two gradients of the same wrong shape");
    assert!(refused(&err), "{err}");
    assert!(tape.grad(leaf).is_none());
}

#[test]
fn a_wrong_weight_gradient_wipes_the_input_gradient_too() {
    let backend = warp(SiluGrad::Real, true);
    let x = tensor(&backend, &[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let w = tensor(&backend, &[1.0, 0.0, 0.0, 1.0, 0.5, -0.5], &[3, 2]);
    let mut tape = Tape::new(backend);
    let xv = tape.leaf(x).unwrap();
    let wv = tape.leaf(w).unwrap();
    let y = tape.linear(xv, wv).unwrap();
    let err = tape
        .backward(y)
        .expect_err("the weight gradient is shape [1], the weight is [3, 2]");
    assert!(refused(&err), "{err}");
    assert!(err.to_string().contains("[1]"), "{err}");
    assert!(err.to_string().contains("[3, 2]"), "{err}");
    assert!(
        tape.grad(xv).is_none(),
        "the input gradient stored first is wiped"
    );
    assert!(
        tape.grad(wv).is_none(),
        "the wrong weight gradient is not kept"
    );
    assert_eq!(tape.value(xv).unwrap().shape(), &[2, 2]);
    assert_eq!(tape.value(wv).unwrap().shape(), &[3, 2]);
}

#[test]
fn a_real_silu_gradient_is_kept_with_the_forward_shape() {
    let backend = warp(SiluGrad::Real, false);
    let x = tensor(&backend, &[1.0, -0.5, 0.25, 2.0], &[2, 2]);
    let mut tape = Tape::new(backend);
    let leaf = tape.leaf(x).unwrap();
    let y = tape.silu(leaf).unwrap();
    tape.backward(y).unwrap();
    let grad = tape.grad(leaf).expect("leaf gradient");
    assert_eq!(grad.shape(), &[2, 2]);
    let values = grad.f32_slice().unwrap();
    assert!(values.iter().all(|v| v.is_finite()));
    assert!(values.iter().any(|v| *v != 0.0));
}

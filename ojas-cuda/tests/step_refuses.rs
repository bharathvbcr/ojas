//! `Qwen35Step` has no kernels behind it yet (ft-7162), so every compute
//! method must refuse rather than return fabricated values.

use ojas_core::OjasError;
use ojas_cuda::{
    AdamWHyper, BankState, GemmOperands, Numerics, Qwen35Step, Sequence, StepProvider, Supervise,
};

fn fresh() -> Qwen35Step {
    Qwen35Step::new(Numerics::ExactF32)
}

fn hyper() -> AdamWHyper {
    AdamWHyper {
        lr: 1e-4,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        grad_scale: 1.0,
    }
}

fn assert_not_wired<T: std::fmt::Debug>(r: Result<T, OjasError>, want_op: &str) {
    match r {
        Err(OjasError::Unsupported { op, detail }) => {
            assert_eq!(op, want_op);
            assert!(detail.contains("ft-7162"), "{detail}");
        }
        other => panic!("{want_op}: expected Unsupported, got {other:?}"),
    }
}

fn assert_untouched(s: &Qwen35Step) {
    assert_eq!(s.step_count(), 0);
    assert_eq!(*s.bank_state(), BankState::Empty);
}

#[test]
fn forward_refuses_on_a_fresh_provider() {
    let s = fresh();
    let seq = Sequence {
        ids: &[1, 2, 3],
        letter_rows: &[0, 1],
        letter_targets: &[2, 3],
        letter_scale: 1.0,
        span_positions: &[1],
    };
    assert_not_wired(s.forward(&seq), "Qwen35Step::forward");
    assert_untouched(&s);
}

#[test]
fn train_forward_refuses_on_a_fresh_provider() {
    let s = fresh();
    let sup = Supervise::Rows {
        positions: &[0],
        targets: &[2],
        scale: 1.0,
    };
    assert_not_wired(
        s.train_forward(&[1, 2, 3], GemmOperands::ExactF32, sup),
        "Qwen35Step::train_forward",
    );
    assert_untouched(&s);
}

#[test]
fn grad_sq_norm_refuses() {
    let s = fresh();
    assert_not_wired(s.grad_sq_norm(), "Qwen35Step::grad_sq_norm");
    let bank = <Qwen35Step as StepProvider>::Bank::default();
    assert_not_wired(
        StepProvider::grad_sq_norm(&s, &bank),
        "Qwen35Step::grad_sq_norm",
    );
}

#[test]
fn adamw_step_without_a_real_gradient_refuses_and_does_not_advance() {
    let mut s = fresh();
    assert_not_wired(
        s.adamw_step(&hyper(), &[0.01], &[1.0]),
        "Qwen35Step::adamw_step",
    );
    assert_untouched(&s);
}

#[test]
fn provider_adamw_step_on_an_empty_bank_refuses_and_does_not_advance() {
    let mut s = fresh();
    let bank = <Qwen35Step as StepProvider>::Bank::default();
    assert_not_wired(
        StepProvider::adamw_step(&mut s, &bank, hyper(), &[0.01], &[1.0]),
        "Qwen35Step::adamw_step",
    );
    assert_untouched(&s);
}

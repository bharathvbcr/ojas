//! `save_state` refuses while the gradient bank holds a partial accumulation.
//! The checkpoint stores masters, both moments and the step count, and
//! `load_state` empties the bank, so a checkpoint cut mid-accumulation would
//! resume without those gradients and say nothing. Opens a Metal runtime:
//! `#[ignore]`, run in the agreed GPU window.
//!
//! ```text
//! cargo test -p ojas-qwen35 --release --test gpu_save_state_bank -- --ignored --test-threads=1
//! ```

#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};

use ojas_core::Budget;
use ojas_qwen35::{
    AdamWHyper, BankState, GroupSpec, LrRule, Numerics, OptimizerPlan, Precision, Qwen35Error,
    Qwen35Step, Select, Sequence, Snapshot, WdRule,
};

fn tiny_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tessl/tests/fixtures/qwen35_train")
}

fn scratch(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("ojas-qwen35-bank-{name}-{}", std::process::id()));
    if d.exists() {
        std::fs::remove_dir_all(&d).unwrap();
    }
    d
}

fn open_tiny() -> Qwen35Step {
    let snap = Snapshot::from_dir(&tiny_dir()).unwrap();
    Qwen35Step::open(
        &snap,
        Numerics::ExactF32,
        Precision::F32,
        Budget::new(1 << 30),
    )
    .unwrap()
}

fn backward_one(s: &mut Qwen35Step) {
    let ids: Vec<u32> = (0..12u32).map(|i| (i * 7 + 3) % 64).collect();
    let rows: Vec<u32> = (0..11u32).collect();
    let pending = s
        .forward(&Sequence {
            ids: &ids,
            letter_rows: &rows,
            letter_targets: &ids[1..],
            letter_scale: 1.0 / 11.0,
            span_positions: &[],
        })
        .unwrap();
    s.backward(pending, None).unwrap();
}

fn plan(s: &Qwen35Step) -> OptimizerPlan {
    OptimizerPlan::build(
        s.parameter_table(),
        &GroupSpec {
            lr: vec![LrRule {
                label: "all".into(),
                select: Select::All,
                lr_scale: 1.0,
            }],
            weight_decay: vec![WdRule {
                label: "all".into(),
                select: Select::All,
                weight_decay: 0.01,
            }],
        },
    )
    .unwrap()
}

#[test]
#[ignore]
fn gpu_save_state_refuses_a_half_filled_gradient_bank() {
    let mut s = open_tiny();
    // Empty bank: allowed.
    let ok = scratch("empty");
    s.save_state(&ok).unwrap();

    backward_one(&mut s);
    assert_eq!(*s.bank_state(), BankState::Holds(1));
    backward_one(&mut s);
    assert_eq!(*s.bank_state(), BankState::Holds(2));

    let dir = scratch("holds");
    let err = s.save_state(&dir).unwrap_err();
    match &err {
        Qwen35Error::Invalid { op, detail } => {
            assert_eq!(*op, "Qwen35Step::save_state");
            assert!(detail.contains("Holds(2)"), "{detail}");
        }
        other => panic!("expected a typed Invalid refusal, got {other:?}"),
    }
    assert!(!dir.exists(), "a refused save must leave nothing behind");
    assert_eq!(
        *s.bank_state(),
        BankState::Holds(2),
        "refusal keeps the bank"
    );

    // The two ways out: discard, or take the step.
    s.discard_gradients();
    s.save_state(&scratch("discarded")).unwrap();
    backward_one(&mut s);
    let hyper = AdamWHyper {
        lr: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        grad_scale: 1.0,
    };
    let p = plan(&s);
    s.adamw_step(&hyper, &p).unwrap();
    s.save_state(&scratch("stepped")).unwrap();
}

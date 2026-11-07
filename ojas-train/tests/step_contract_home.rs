//! The `StepProvider` contract has one definition, in `ojas-core`; `ojas_train::lappi::step`
//! re-exports it. A second copy in this crate would be a different type, and these checks (which
//! compile only when the paths name the same items) would fail.

use std::any::TypeId;

use ojas_core::step as core_step;
use ojas_train::lappi::mock::ToyProvider;
use ojas_train::lappi::step as train_step;

fn is_core_provider<P: core_step::StepProvider>() {}

#[test]
fn the_trainers_step_module_is_the_core_contract() {
    // The mock implements the trainer-path trait; it satisfies the core-path bound only when
    // both paths name one trait.
    is_core_provider::<ToyProvider>();

    assert_eq!(
        TypeId::of::<train_step::StepError>(),
        TypeId::of::<core_step::StepError>()
    );
    assert_eq!(
        TypeId::of::<train_step::ParamSpec>(),
        TypeId::of::<core_step::ParamSpec>()
    );
    assert_eq!(
        TypeId::of::<train_step::AdamWHyper>(),
        TypeId::of::<core_step::AdamWHyper>()
    );
    assert_eq!(
        TypeId::of::<train_step::BankMode>(),
        TypeId::of::<core_step::BankMode>()
    );
    assert_eq!(
        TypeId::of::<train_step::StorageDtype>(),
        TypeId::of::<core_step::StorageDtype>()
    );
    assert_eq!(
        TypeId::of::<train_step::CheckpointContext>(),
        TypeId::of::<core_step::CheckpointContext>()
    );
}

#[test]
fn the_shared_checks_refuse_with_the_core_error() {
    let err: core_step::StepError =
        train_step::check_per_entry(2, &[1.0], &[0.0, 0.0]).unwrap_err();
    assert!(matches!(err, core_step::StepError::Contract(_)));
    let err: core_step::StepError = train_step::refuse_lr_scale(&[0.5], "x").unwrap_err();
    assert!(matches!(err, core_step::StepError::Unsupported(_)));
}

/// `SequenceJob::validate` refuses a row position scored twice, and the mock's two callers
/// (`accumulate` and `row_loss`) go through it; scale 0 with rows stays legal.
#[test]
fn the_mock_refuses_a_repeated_row_position_through_the_shared_validate() {
    use ojas_train::lappi::step::{BankMode, RowTargets, SequenceJob, StepProvider};

    let mut p = ToyProvider::new(8, 4, 1, 7).unwrap();
    let tokens = [1u32, 2, 3, 4];
    let job = |positions: &'static [u32], scale: f32| SequenceJob {
        tokens: &tokens,
        rows: RowTargets {
            positions,
            targets: &[5, 6, 7][..positions.len()],
            scale,
        },
        hidden_at: &[],
    };
    let dup = job(&[1, 2, 1], 1.0);
    let err = p.accumulate(&dup, BankMode::Overwrite, None).unwrap_err();
    assert!(
        matches!(err, core_step::StepError::Contract(_)) && err.to_string().contains("position 1"),
        "{err}"
    );
    assert!(p.row_loss(&dup).is_err());
    let distinct = job(&[0, 1, 2], 1.0);
    assert!(p.row_loss(&distinct).is_ok());
    assert!(
        p.accumulate(&job(&[0, 1, 2], 0.0), BankMode::Overwrite, None)
            .is_ok()
    );
}

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

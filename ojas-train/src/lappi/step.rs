//! The step-provider contract, re-exported from [`ojas_core::step`].
//!
//! The contract has one definition, in `ojas-core`, so a backend crate can implement it without
//! depending on this trainer. This module keeps
//! the paths the trainer, the Tape provider and Lappi already import
//! (`ojas_train::lappi::step::StepProvider` and the rest). Read the contract itself, and its
//! seven rules, in `ojas-core/src/step.rs`.

pub use ojas_core::step::{
    AdamWHyper, BankMode, CheckpointContext, HiddenGrad, ParamSpec, RowTargets, SequenceJob,
    StepError, StepProvider, StorageDtype, check_per_entry, refuse_lr_scale,
};

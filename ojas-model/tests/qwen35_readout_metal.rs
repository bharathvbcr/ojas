//! The readout seam of `qwen35_readout.rs` on the Metal tape: `gather_rows`
//! forward and backward on a computed table, the composed answer head against
//! torch, a right-padded batch against each example alone, and the
//! padding being unreadable. A missing device fails unless
//! `OJAS_ALLOW_NO_GPU=1`.
//!
//! Metal's gated delta rule runs within 2e-4 of the CPU's, so a gradient
//! behind it is held to that against torch (as in `qwen35_metal_tape.rs`);
//! row-wise values of a padded batch against each example alone are held to
//! [`METAL_VALUES`], where a kernel's tiling may change with the row count.

#![cfg(target_os = "macos")]

mod qwen35_common;

use ojas_core::{Backend, Budget, Tensor};
use ojas_metal::MetalBackend;
use qwen35_common::lora::{fixture_sites, lora_file};
use qwen35_common::readout::*;
use qwen35_common::*;

const METAL_GRAD_BOUND: f32 = 2e-4;
/// Row-wise values of a padded batch against an example alone.
const METAL_VALUES: f32 = 1e-5;
const METAL: Tol = Tol {
    values: METAL_VALUES,
    sums: 2e-4,
};

fn metal() -> Option<MetalBackend> {
    match MetalBackend::new(Budget::new(8 << 30)) {
        Ok(m) => Some(m),
        Err(e) if std::env::var_os("OJAS_ALLOW_NO_GPU").is_some() => {
            eprintln!("no Metal device: {e:?}");
            None
        }
        Err(e) => panic!("no Metal device (set OJAS_ALLOW_NO_GPU=1 to skip): {e:?}"),
    }
}

#[test]
fn gather_rows_is_index_select_and_its_scatter_on_metal() {
    let Some(m) = metal() else { return };
    check_gather(m.clone(), &|t: &Tensor| m.upload(t).unwrap());
}

#[test]
fn two_examples_at_the_same_position_do_not_alias_on_metal() {
    let Some(m) = metal() else { return };
    check_same_position(m.clone(), &|t: &Tensor| m.upload(t).unwrap());
}

#[test]
fn the_composed_answer_head_matches_torch_on_metal() {
    let Some(m) = metal() else { return };
    check_against_torch(
        m.clone(),
        &|t: &Tensor| m.upload(t).unwrap(),
        LOSS_BOUND,
        METAL_GRAD_BOUND,
    );
}

#[test]
fn the_full_vocabulary_value_adds_no_gradient_on_metal() {
    let Some(m) = metal() else { return };
    check_full_vocab_is_inert(m.clone(), &|t: &Tensor| m.upload(t).unwrap());
}

#[test]
fn a_right_padded_batch_gives_each_example_its_own_readout_on_metal() {
    let Some(m) = metal() else { return };
    let up = |t: &Tensor| m.upload(t).unwrap();
    check_batch_against_alone(m.clone(), &up, &Base::Trainable, METAL);
    let sites = fixture_sites(&lora_file(), true);
    check_batch_against_alone(m.clone(), &up, &Base::Lora(&sites), METAL);
}

#[test]
fn right_padding_cannot_reach_the_readout_on_metal() {
    let Some(m) = metal() else { return };
    let up = |t: &Tensor| m.upload(t).unwrap();
    check_padding_is_inert(m.clone(), &up, &Base::Frozen, METAL);
    check_padding_is_inert(m.clone(), &up, &Base::Trainable, METAL);
    let sites = fixture_sites(&lora_file(), true);
    check_padding_is_inert(m.clone(), &up, &Base::Lora(&sites), METAL);
}

#[test]
fn checkpointed_layers_give_the_readout_the_same_bits_on_metal() {
    use ojas_model::ActivationCheckpoint::{Blocks, Off};
    let Some(m) = metal() else { return };
    let up = |t: &Tensor| m.upload(t).unwrap();
    let c = case(true);
    let sites = fixture_sites(&lora_file(), true);
    for base in [Base::Frozen, Base::Lora(&sites)] {
        let a = run_on(m.clone(), &up, &c, &base, Off);
        let b = run_on(m.clone(), &up, &c, &base, Blocks);
        assert_eq!(a.loss.to_bits(), b.loss.to_bits());
        assert_close(&a.hidden, &b.hidden, 0.0, "hidden");
        assert_close(&a.dh, &b.dh, 0.0, "dH");
        assert_close(&a.d17_grad, &b.d17_grad, 0.0, "dD17");
        for (x, y) in a.param_grads.iter().zip(&b.param_grads) {
            assert_close(x, y, 0.0, "adapter gradient");
        }
    }
}

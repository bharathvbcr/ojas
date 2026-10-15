//! LoRA on the Qwen3.5 tape at scaling 1, CPU: the pre-registered LoRA-2B
//! recipe is r = 16, alpha = 16, where `scaled_b` returns `B` itself and
//! `LoraForward` records no scale constant, a graph the scaling-2 fixture
//! never builds. Bounds (a'), (b'), (c') of the fine-tune brief, the same
//! clauses as `qwen35_lora.rs`, against `tests/fixtures/qwen35_tiny_lora_s1`
//! (`qwen35_lora_fixture.py --alpha 4`: rank 4, alpha 4). The two fixtures
//! share the seed, so they hold the same `A`, `B` and batch and differ only
//! by the scale.

mod qwen35_common;

use ojas_autograd::Tape;
use ojas_core::{Budget, OjasError};
use ojas_data::CounterRng;
use ojas_model::lora::scaled_b;
use ojas_model::qwen35::LoraForward;
use qwen35_common::lora::*;
use qwen35_common::*;

/// The fixture is the scaling-1 twin of the scaling-2 one: same adapters
/// and data, a different scale, so a different loss.
#[test]
fn the_fixture_is_the_scaling_2_fixture_at_scale_1() {
    let (one, two) = (SCALING_1.file(), SCALING_2.file());
    assert_eq!(SCALING_1.config().scaling(), 1.0);
    assert_eq!(SCALING_2.config().scaling(), 2.0);
    let mut shared = 0;
    for name in two
        .names()
        .filter(|n| n.starts_with("init.") || n.starts_with("bnz."))
    {
        assert_eq!(read(&one, name), read(&two, name), "{name}");
        shared += 1;
    }
    assert_eq!(shared, 2 * targets().len());
    assert_eq!(read(&one, "c.loss"), read(&two, "c.loss"));
    assert!(rel(read(&one, "a.loss")[0], read(&two, "a.loss")[0]) > 1e-4);
}

/// The short-circuit is what runs: at scaling 1 `LoraForward` records no
/// scale constant (its budget is never charged), at scaling 2 it does; and
/// `scaled_b` at 1 is `B` itself with no constant, at 2 it needs one.
#[test]
fn scaling_one_records_no_scale_constant() {
    let d = data();
    let sites = fixture_sites(&SCALING_1.file(), true);
    let charged = |fx: Fixture| {
        let mut tape = Tape::new(cpu());
        let bound = sites
            .try_map(&mut |site| site.try_map(&mut |t| tape.leaf(t.clone())))
            .unwrap();
        let own = Budget::new(1 << 30);
        let [b, t] = [d.ids.shape()[0], d.ids.shape()[1]];
        let mut rng = CounterRng::new(0);
        let fwd = LoraForward::new(&mut tape, bound, &fx.config(), b, t, &mut rng, &own).unwrap();
        drop(fwd);
        own.peak_bytes()
    };
    assert_eq!(charged(SCALING_1), 0);
    assert!(charged(SCALING_2) > 0);

    let mut tape = Tape::new(cpu());
    let leaf = tape
        .leaf(fixture_sites(&SCALING_1.file(), true).sites()[0].a.clone())
        .unwrap();
    assert_eq!(scaled_b(&mut tape, &leaf, 1.0, None).unwrap(), leaf);
    assert!(matches!(
        scaled_b(&mut tape, &leaf, 2.0, None),
        Err(OjasError::OutOfRange { .. })
    ));
}

#[test]
fn a_prime_loss_and_adapter_gradients_match_torch_at_scaling_one() {
    check_a_prime(SCALING_1);
}

#[test]
fn c_prime_zero_b_is_the_identity_bit_for_bit_at_scaling_one() {
    check_c_prime(SCALING_1);
}

#[test]
fn b_prime_twenty_step_losses_match_torch_and_repeat_at_scaling_one() {
    check_b_prime_losses(SCALING_1);
}

#[test]
fn b_prime_final_adapters_are_within_the_oracles_noise_floor_at_scaling_one() {
    check_b_prime_finals(SCALING_1);
}

#[test]
fn a_lora_linear_matches_torch_on_dx_da_db_at_scaling_one() {
    check_lora_linear(SCALING_1);
}

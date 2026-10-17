//! LoRA on the Qwen3.5 tape, CPU, over the fake-quant q8 base: the (a'), (b'),
//! (c') clauses of `qwen35_lora.rs`, with the same assertions and bounds, on
//! `qwen35_tiny_fq8`, the tiny base whose ten wrapped linear kinds are
//! `dequantize(quantize(W))` from mlx (q8, group 64, bf16 scales and biases;
//! `qwen35_fakequant_base.py`, `manifest.json` records the version and the
//! call). The references in `qwen35_tiny_lora_fq8` are the plain-torch fp32
//! one-step and 20-step runs over that base (`qwen35_lora_fixture.py --base`).
//! The Metal twin is `qwen35_lora_fq8_metal.rs`.

mod qwen35_common;

use ojas_model::qwen35::{lora_targets, LORA_KINDS};
use qwen35_common::lora::*;
use qwen35_common::*;

/// The fixture is what it says: the wrapped linears differ from the bf16
/// base (quantisation took effect), every other tensor is the same bits, and
/// the base loss moved, so the clauses below are not the bf16 ones again.
#[test]
fn the_fq8_base_changes_exactly_the_wrapped_linears() {
    let (fq, bf) = (FQ8.base_file(), base_file());
    let wrapped: Vec<String> = lora_targets(&spec(), &fq, PREFIX, &LORA_KINDS)
        .unwrap()
        .into_iter()
        .map(|t| t.name)
        .collect();
    assert_eq!(wrapped.len(), 3 + 4 + 2 * 3);
    let names: Vec<String> = fq.names().map(str::to_string).collect();
    let mut expected: Vec<String> = bf.names().map(str::to_string).collect();
    let (mut a, mut b) = (names.clone(), expected.clone());
    a.sort();
    b.sort();
    assert_eq!(a, b, "same tensors in both bases");
    expected.retain(|n| !wrapped.contains(n));
    for n in &wrapped {
        let (x, y) = (read(&fq, n), read(&bf, n));
        assert!(x != y, "{n} is unchanged by quantisation");
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        // Quantised values are bf16: the low 16 bits of the f32 are zero.
        assert!(bits(&x).iter().all(|b| b & 0xffff == 0), "{n} is not bf16");
    }
    for n in &expected {
        let bits = |f: &ojas_io::SafeTensors<'_>| {
            read(f, n).iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        };
        assert_eq!(bits(&fq), bits(&bf), "{n} is not a byte copy");
    }
    let (q, p) = (
        read(&FQ8.file(), "c.base_loss")[0],
        read(&SCALING_2.file(), "c.base_loss")[0],
    );
    assert!(
        rel(q, p) > 1e-6,
        "fq8 base loss {q} equals the bf16 base's {p}"
    );
}

#[test]
fn targets_over_the_fq8_base_cover_every_linear() {
    let t = lora_targets(&spec(), &FQ8.base_file(), PREFIX, &LORA_KINDS).unwrap();
    assert_eq!(t.len(), 3 + 4 + 2 * 3);
    for kind in LORA_KINDS {
        assert!(t.iter().any(|x| x.kind == kind), "{kind:?} has no target");
    }
}

/// (a') loss and every adapter gradient against plain torch; no gradient
/// for any frozen leaf.
#[test]
fn a_prime_loss_and_adapter_gradients_match_torch() {
    check_a_prime(FQ8);
}

/// (c') at B = 0 the LoRA loss is the base loss bit for bit.
#[test]
fn c_prime_zero_b_is_the_identity_bit_for_bit() {
    check_c_prime(FQ8);
}

/// (b') the loss clauses, and the run is bit-repeatable.
#[test]
fn b_prime_twenty_step_losses_match_torch_and_repeat_bit_for_bit() {
    check_b_prime_losses(FQ8);
}

/// (b') the final adapters, within the oracle's own noise floor.
#[test]
fn b_prime_final_adapters_are_within_the_oracles_own_noise_floor() {
    check_b_prime_finals(FQ8);
}

/// A plain LoRA linear (the fixture's own x/W/A/B, not the base).
#[test]
fn a_prime_a_lora_linear_matches_torch_on_dx_da_db() {
    check_lora_linear(FQ8);
}

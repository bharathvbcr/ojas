//! The resident q8 base on the Tape, on the CPU backend (card ft-5f743c19, qv2 W3).
//!
//! The base is `tests/fixtures/qwen35_tiny_q8` (mlx 0.31.2 q8, group 64) loaded by
//! `load_hf_q8_pinned` with its ten adapted linears packed, and its fake-quant twin
//! `qwen35_tiny_fq8` (`mx.dequantize` of the same quantisation, held dense in bf16). A linear over
//! a packed weight dequantises it to a transient per use and runs the frozen dense backward over
//! it, so the two bases give the same loss and the same adapter gradients bit for bit. Each test
//! fails without the change (the API does not exist), and a few carry a negative control that
//! proves the check can fire.

mod common;
mod qwen35_common;
mod qwen35_q8_common;

use ojas_autograd::{Tape, Var};
use ojas_core::{Autocast, DType, Tensor};
use ojas_data::CounterRng;
use ojas_model::qwen35::{
    bind, bind_frozen, bind_frozen_base, forward_hidden, forward_hidden_lora_base,
    forward_loss_lora, forward_loss_lora_base, load_hf_q8_pinned, FrozenWeight, LoraForward,
    Qwen35Params,
};
use ojas_model::{ActivationCheckpoint, Eval, DEFAULT_CE_CHUNK};
use qwen35_common::lora::*;
use qwen35_common::*;
use qwen35_q8_common::*;

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn fixed(run: &LoraRun) -> Vec<Vec<u32>> {
    run.grads
        .iter()
        .flat_map(|(_, a, b)| [bits(a), bits(b)])
        .collect()
}

/// The fq8 base over the dense frozen path against the q8 base over the packed path, both over
/// fresh backends `mk` makes, with non-zero `B` so every layer's dX reaches an adapter. Returns
/// the two runs after asserting the loss and every adapter gradient bit-equal.
fn assert_q8_equals_fq8<B: ojas_core::Backend>(
    mk: impl Fn() -> B,
    activations: ActivationCheckpoint,
    bf16: bool,
) -> (LoraRun, LoraRun) {
    let fx = FQ8;
    let (d, sites) = (fx.data(), fixture_sites(&fx.file(), true));
    let opts = || Opts {
        activations,
        bf16,
        ..fx.opts()
    };
    let dense = lora_step(mk(), &fq8_host(), &sites, &d, &opts());
    let backend = mk();
    let packed_base = load_q8(&backend);
    let packed = lora_step_base(backend, &packed_base, &sites, &d, &opts());
    assert_eq!(
        packed.loss.to_bits(),
        dense.loss.to_bits(),
        "loss {} against {}",
        packed.loss,
        dense.loss
    );
    assert_eq!(packed.grads.len(), dense.grads.len());
    for ((module, ..), (other, ..)) in packed.grads.iter().zip(&dense.grads) {
        assert_eq!(module, other);
    }
    assert_eq!(fixed(&packed), fixed(&dense), "adapter gradients differ");
    // The comparison is not vacuous: the gradients exist and are not all zero.
    assert!(packed
        .grads
        .iter()
        .any(|(_, a, b)| a.iter().chain(b).any(|&x| x != 0.0)));
    (dense, packed)
}

#[test]
fn q8_base_loss_and_adapter_grads_equal_fq8_bitwise_off() {
    assert_q8_equals_fq8(cpu, ActivationCheckpoint::Off, false);
}

#[test]
fn q8_base_loss_and_adapter_grads_equal_fq8_bitwise_blocks() {
    assert_q8_equals_fq8(cpu, ActivationCheckpoint::Blocks, false);
}

/// Inside a bf16 autocast region the dequantisation passes through (a bf16 weight is already
/// bf16), so the packed path equals the dense one there too.
#[test]
fn q8_base_equals_fq8_bitwise_under_bf16_autocast_off_and_blocks() {
    for activations in [ActivationCheckpoint::Off, ActivationCheckpoint::Blocks] {
        assert_q8_equals_fq8(|| Autocast::new(cpu()), activations, true);
    }
}

/// Checkpointing changes memory, not values: Blocks equals Off over the packed base, bit for bit.
#[test]
fn the_packed_base_gives_the_same_bits_under_off_and_blocks() {
    let fx = FQ8;
    let (d, sites) = (fx.data(), fixture_sites(&fx.file(), true));
    let run = |activations| {
        let backend = cpu();
        let base = load_q8(&backend);
        let o = Opts {
            activations,
            ..fx.opts()
        };
        lora_step_base(backend, &base, &sites, &d, &o)
    };
    let (off, blocks) = (
        run(ActivationCheckpoint::Off),
        run(ActivationCheckpoint::Blocks),
    );
    assert_eq!(off.loss.to_bits(), blocks.loss.to_bits());
    assert_eq!(fixed(&off), fixed(&blocks));
}

/// An eager forward over the packed base equals the eager forward over the fake-quant base.
#[test]
fn eval_over_the_q8_base_equals_eval_over_the_fq8_base_bitwise() {
    let (s, d) = (spec(), FQ8.data());
    let mut e = Eval::new(cpu());
    let dense = bind(&mut e, &fq8_host()).unwrap();
    let want = forward_hidden(
        &mut e,
        &s,
        &dense,
        &d.ids,
        &d.tables,
        ActivationCheckpoint::Off,
    )
    .unwrap();
    let base = load_q8(e.backend());
    let packed = bind_frozen_base(&mut e, &base).unwrap();
    let got = forward_hidden_lora_base(
        &mut e,
        &s,
        &packed,
        None,
        &d.ids,
        &d.tables,
        ActivationCheckpoint::Off,
    )
    .unwrap();
    assert_eq!(got.shape(), want.shape());
    let (got, want) = (got.to_f32_vec().unwrap(), want.to_f32_vec().unwrap());
    assert!(want.iter().any(|&x| x != 0.0));
    assert_eq!(bits(&got), bits(&want));
}

/// A base that holds some wrapped weights dense and the rest packed is the same model.
#[test]
fn a_base_mixing_dense_and_packed_weights_gives_the_same_bits() {
    let fx = FQ8;
    let (d, sites) = (fx.data(), fixture_sites(&fx.file(), true));
    let backend = cpu();
    let packed_base = load_q8(&backend);
    // Layer 0's eight wrapped weights dense, layer 1's eight packed.
    let mut seen = 0usize;
    let mixed = map_wrapped(&packed_base, &mut |w| {
        seen += 1;
        match w {
            FrozenWeight::Quant(q) if seen <= WRAPPED_PARTS / 2 => {
                FrozenWeight::Dense(dequant(&backend, q))
            }
            other => other.clone(),
        }
    });
    assert_eq!(seen, WRAPPED_PARTS);
    let fields = wrapped_fields(&mixed);
    let dense = fields
        .iter()
        .filter(|(_, w)| matches!(w, FrozenWeight::Dense(_)))
        .count();
    assert_eq!((dense, fields.len() - dense), (8, 8));
    let all_packed = lora_step_base(cpu(), &packed_base, &sites, &d, &fx.opts());
    let some = lora_step_base(cpu(), &mixed, &sites, &d, &fx.opts());
    assert_eq!(some.loss.to_bits(), all_packed.loss.to_bits());
    assert_eq!(fixed(&some), fixed(&all_packed));
}

/// The dense entry points are the base-generic ones over a `frozen_view`: the same bits.
#[test]
fn the_dense_forward_is_the_base_form_over_a_frozen_view() {
    let fx = FQ8;
    let (s, d, sites) = (spec(), fx.data(), fixture_sites(&fx.file(), true));
    let host = fq8_host();
    let loss = |packed_form: bool, activations| {
        let mut tape = Tape::new(cpu());
        let bound = sites
            .try_map(&mut |site| site.try_map(&mut |t| tape.leaf(t.clone())))
            .unwrap();
        let mut rng = CounterRng::new(0);
        let [b, t] = [d.ids.shape()[0], d.ids.shape()[1]];
        let fwd =
            LoraForward::new(&mut tape, bound, &fx.config(), b, t, &mut rng, &budget()).unwrap();
        let var = if packed_form {
            let params = bind_frozen_base(&mut tape, &host.frozen_view()).unwrap();
            forward_loss_lora_base(
                &mut tape,
                &s,
                &params,
                Some(&fwd),
                &d.ids,
                &d.targets,
                &d.tables,
                Some(IGNORE),
                DEFAULT_CE_CHUNK,
                activations,
            )
        } else {
            let params = bind_frozen(&mut tape, &host).unwrap();
            forward_loss_lora(
                &mut tape,
                &s,
                &params,
                Some(&fwd),
                &d.ids,
                &d.targets,
                &d.tables,
                Some(IGNORE),
                DEFAULT_CE_CHUNK,
                activations,
            )
        }
        .unwrap();
        to_vec(tape.value(var).unwrap())[0].to_bits()
    };
    for activations in [ActivationCheckpoint::Off, ActivationCheckpoint::Blocks] {
        assert_eq!(loss(true, activations), loss(false, activations));
    }
}

// ---- C2: every packed part dequantises to the fq8 field ---------------------------------------

/// All 16 parts, compared per field (layer 0's `wq/wk/wv` are row blocks of `in_proj_qkv`, layer
/// 1's `wq/wgate` the query and gate rows of `q_proj`): a swapped pair fails here, which a
/// comparison of the union would not see.
#[test]
fn every_frozen_weight_dequantises_to_the_fq8_field_cpu() {
    let be = cpu();
    let (q8, fq8) = (load_q8(&be), fq8_host());
    let (packed, dense) = (wrapped_fields(&q8), wrapped_fields(&fq8));
    assert_eq!((packed.len(), dense.len()), (WRAPPED_PARTS, WRAPPED_PARTS));
    for ((name, w), (other, want)) in packed.into_iter().zip(dense) {
        assert_eq!(name, other);
        let FrozenWeight::Quant(q) = w else {
            panic!("{name} is not packed");
        };
        assert_eq!([q.rows(), q.cols()], want.shape(), "{name}");
        assert_eq!(
            host_bits(&be, &dequant(&be, q)),
            want.bf16_slice().unwrap(),
            "{name}"
        );
    }
}

/// The negative control: attention's query and gate rows differ, so the per-field comparison
/// tells them apart.
#[test]
fn the_per_field_comparison_can_tell_the_query_rows_from_the_gate_rows() {
    let be = cpu();
    let (q8, fq8) = (load_q8(&be), fq8_host());
    let (packed, dense) = (wrapped_fields(&q8), wrapped_fields(&fq8));
    let find = |name: &str| packed.iter().position(|(n, _)| n == name).unwrap();
    let (wq, wgate) = (find("layer1.attn.wq"), find("layer1.attn.wgate"));
    let FrozenWeight::Quant(q) = packed[wq].1 else {
        panic!("not packed");
    };
    let got = host_bits(&be, &dequant(&be, q));
    assert_eq!(got, dense[wq].1.bf16_slice().unwrap());
    assert_ne!(got, dense[wgate].1.bf16_slice().unwrap());
}

// ---- C3: the dequantisations a step makes ----------------------------------------------------

/// Off: the bind dequantises nothing, the forward once per packed part, the backward once more.
/// The tiny tower has 16 parts, so 16 after the forward and 32 after the backward.
#[test]
fn a_step_dequantises_two_times_per_linear_without_checkpointing() {
    let base = load_q8(&cpu());
    let c = counted_step(cpu(), &base, ActivationCheckpoint::Off, false);
    assert_eq!(
        (c.after_bind, c.after_forward, c.after_backward),
        (0, WRAPPED_PARTS, 2 * WRAPPED_PARTS)
    );
}

/// Blocks: the replay of each layer dequantises again, so three times per linear: 16, then
/// 16 + 32 = 48.
#[test]
fn a_step_dequantises_three_times_per_linear_under_blocks() {
    let base = load_q8(&cpu());
    let c = counted_step(cpu(), &base, ActivationCheckpoint::Blocks, false);
    assert_eq!(
        (c.after_bind, c.after_forward, c.after_backward),
        (0, WRAPPED_PARTS, 3 * WRAPPED_PARTS)
    );
}

/// The control: dequantising at bind counts 16 at the bind and none per step, so it fails the
/// two tests above.
#[test]
fn the_o1_control_dequantises_at_bind_and_never_again() {
    let base = load_q8(&cpu());
    for activations in [ActivationCheckpoint::Off, ActivationCheckpoint::Blocks] {
        let c = counted_step(cpu(), &base, activations, true);
        assert_eq!(
            (c.after_bind, c.after_forward, c.after_backward),
            (WRAPPED_PARTS, WRAPPED_PARTS, WRAPPED_PARTS)
        );
    }
}

// ---- C4: no dense wrapped-shape weight in the tape's values ----------------------------------

/// Every value on `tape` with the `[out, in]` of a wrapped part, in bf16 or f32.
fn dense_weight_values(tape: &Tape, shapes: &[[usize; 2]]) -> usize {
    let mut found = 0;
    let mut i = 0;
    while let Ok(v) = tape.value(Var(i)) {
        if matches!(v.dtype(), DType::Bf16 | DType::F32) && shapes.iter().any(|s| v.shape() == s) {
            found += 1;
        }
        i += 1;
    }
    found
}

fn part_shapes(base: &Qwen35Params<Tensor, FrozenWeight<Tensor>>) -> Vec<[usize; 2]> {
    wrapped_fields(base)
        .into_iter()
        .map(|(name, w)| match w {
            FrozenWeight::Quant(q) => [q.rows(), q.cols()],
            FrozenWeight::Dense(_) => panic!("{name} is not packed"),
        })
        .collect()
}

/// A forward over `base` (no backward): the tape holds whatever the forward retains.
fn forward_tape(
    base: &Qwen35Params<Tensor, FrozenWeight<Tensor>>,
    activations: ActivationCheckpoint,
) -> Tape {
    let fx = FQ8;
    let (d, sites) = (fx.data(), fixture_sites(&fx.file(), true));
    let mut tape = Tape::new(cpu());
    let params = bind_frozen_base(&mut tape, base).unwrap();
    let bound = sites
        .try_map(&mut |site| site.try_map(&mut |t| tape.leaf(t.clone())))
        .unwrap();
    let mut rng = CounterRng::new(0);
    let [b, t] = [d.ids.shape()[0], d.ids.shape()[1]];
    let fwd = LoraForward::new(&mut tape, bound, &fx.config(), b, t, &mut rng, &budget()).unwrap();
    forward_loss_lora_base(
        &mut tape,
        &spec(),
        &params,
        Some(&fwd),
        &d.ids,
        &d.targets,
        &d.tables,
        Some(IGNORE),
        DEFAULT_CE_CHUNK,
        activations,
    )
    .unwrap();
    tape
}

#[test]
fn no_dense_wrapped_weight_is_on_the_tape_over_the_q8_base() {
    let base = load_q8(&cpu());
    let shapes = part_shapes(&base);
    assert_eq!(shapes.len(), WRAPPED_PARTS);
    for activations in [ActivationCheckpoint::Off, ActivationCheckpoint::Blocks] {
        let tape = forward_tape(&base, activations);
        assert_eq!(
            dense_weight_values(&tape, &shapes),
            0,
            "{activations:?}: a dense [out, in] weight is a value of the tape"
        );
    }
}

/// The negative control: the same scan over the dense base finds a frozen weight per part, so
/// the scan above can fire.
#[test]
fn the_dense_fq8_base_does_have_wrapped_weights_on_the_tape() {
    let shapes = part_shapes(&load_q8(&cpu()));
    let dense = fq8_host().frozen_view();
    for activations in [ActivationCheckpoint::Off, ActivationCheckpoint::Blocks] {
        let mut tape = Tape::new(cpu());
        let fx = FQ8;
        let (d, sites) = (fx.data(), fixture_sites(&fx.file(), true));
        let params = bind_frozen_base(&mut tape, &dense).unwrap();
        let bound = sites
            .try_map(&mut |site| site.try_map(&mut |t| tape.leaf(t.clone())))
            .unwrap();
        let mut rng = CounterRng::new(0);
        let [b, t] = [d.ids.shape()[0], d.ids.shape()[1]];
        let fwd =
            LoraForward::new(&mut tape, bound, &fx.config(), b, t, &mut rng, &budget()).unwrap();
        forward_loss_lora_base(
            &mut tape,
            &spec(),
            &params,
            Some(&fwd),
            &d.ids,
            &d.targets,
            &d.tables,
            Some(IGNORE),
            DEFAULT_CE_CHUNK,
            activations,
        )
        .unwrap();
        let found = dense_weight_values(&tape, &shapes);
        assert!(
            found >= WRAPPED_PARTS,
            "{activations:?}: found {found} dense weights, expected at least {WRAPPED_PARTS}"
        );
    }
}

/// On the CPU the packed weights stay host-resident and are charged as such: the budget holds the
/// dense tensors and the packed parts and nothing else, so the fused reads and the row-cut copies
/// the loader makes on the way are all gone.
#[test]
fn the_cpu_base_is_charged_its_dense_tensors_plus_its_packed_parts_and_nothing_else() {
    let budget = budget();
    let base = load_hf_q8_pinned(
        &spec(),
        &q8_path(),
        &q8_identity(),
        &q8_config(),
        PREFIX,
        DType::Bf16,
        DType::F32,
        &budget,
        &cpu(),
    )
    .unwrap();
    assert_eq!(
        budget.live_bytes().unwrap(),
        dense_host_bytes(&base) + packed_bytes(&base)
    );
    assert!(packed_bytes(&base) > 0 && dense_host_bytes(&base) > 0);
}

/// What stays resident is the packed parts: 1 byte a code and 4 bytes a group of 64, so about
/// 53% of the bf16 bytes.
#[test]
fn the_packed_base_holds_about_half_the_bytes_of_the_bf16_one() {
    let base = load_q8(&cpu());
    let (mut packed, mut dense) = (0usize, 0usize);
    for (name, w) in wrapped_fields(&base) {
        let FrozenWeight::Quant(q) = w else {
            panic!("{name} is not packed");
        };
        let (rows, cols) = (q.rows(), q.cols());
        assert_eq!(
            q.nbytes().unwrap(),
            rows * cols + rows * (cols / 64) * 4,
            "{name}"
        );
        assert_eq!(q.dequantised_bytes().unwrap(), rows * cols * 2, "{name}");
        packed += q.nbytes().unwrap();
        dense += rows * cols * 2;
    }
    assert!(packed * 100 < dense * 54, "{packed} bytes against {dense}");
    assert!(packed * 100 > dense * 53, "{packed} bytes against {dense}");
}

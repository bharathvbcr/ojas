//! The Qwen3.5 readout seam on the CPU tape: `forward_hidden` (the tower to
//! the final norm), `gather_rows` (one hidden row per example, with a
//! gradient) and `RightPadded` (the only padding the readout accepts), and
//! the composition a host-side answer head makes of them, pinned against
//! torch (`tests/fixtures/qwen35_tiny_readout`, see `qwen35_common::readout`).
//!
//! Bounds are the tessl/transformers ones of `qwen35_fixture.rs` (loss within
//! 1e-5 relative, a gradient within 1e-4 of its own peak). The padded batch
//! against each example alone is held to bit equality wherever the ops are
//! row independent and to `SUM_ORDER_BOUND` (in `qwen35_common::readout`)
//! where a sum over the batch changes order.

mod qwen35_common;

use ojas_autograd::Tape;
use ojas_core::{DType, OjasError, Tensor};
use ojas_io::SafeTensors;
use ojas_model::qwen35::{
    bind, forward_hidden, forward_hidden_lora, forward_loss, gather_rows, load_hf, LoraForward,
    Qwen35Params, Qwen35Tables, RightPadded,
};
use ojas_model::{ActivationCheckpoint, Eval, Graph, DEFAULT_CE_CHUNK};
use qwen35_common::lora::{self as lora_common, fixture_sites, lora_file, to_vec};
use qwen35_common::readout::*;
use qwen35_common::*;

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn run_case(c: &Case, base: &Base<'_>, act: ActivationCheckpoint) -> Step {
    run_on(cpu(), &Tensor::clone, c, base, act)
}

// ---- forward_hidden is forward_loss's tower ---------------------------------

/// Loss bits and an FNV-1a hash of every gradient's bits of `forward_loss` on
/// the `qwen35_tiny` batch, recorded by `qwen35_loss_bench` (its `PIN cpu`
/// line) on the tree before `forward_loss` was split onto `forward_hidden`.
const PIN_LOSS: u32 = 0x412d_d22b;
const PIN_GRADS: u64 = 0xa78b_729e_f5d1_c13a;

fn fnv(grads: &Qwen35Params<Tensor>) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for t in grads.values() {
        for x in to_vec(t) {
            for byte in x.to_bits().to_le_bytes() {
                h = (h ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
            }
        }
    }
    h
}

#[test]
fn forward_loss_keeps_the_bits_the_unmodified_code_gave() {
    let r = run(cpu(), Tensor::clone, ActivationCheckpoint::Off);
    assert_eq!(
        (r.loss.to_bits(), fnv(&r.grads)),
        (PIN_LOSS, PIN_GRADS),
        "loss {:#010x}, gradients {:#018x}",
        r.loss.to_bits(),
        fnv(&r.grads)
    );
}

#[test]
fn forward_hidden_then_the_tied_head_is_forward_loss() {
    let s = spec();
    let model = SafeTensors::open(&dir().join("model.safetensors")).unwrap();
    let reference = SafeTensors::open(&dir().join("reference.safetensors")).unwrap();
    let params = load_hf(&s, &model, PREFIX, DType::F32, DType::F32, &budget()).unwrap();
    let (ids, targets) = ids_and_targets(&reference);
    let tables = Qwen35Tables::new(&s, ids.shape()[1], &budget()).unwrap();
    let mut eval = Eval::new(cpu());
    let bound = bind(&mut eval, &params).unwrap();
    let h = forward_hidden(
        &mut eval,
        &s,
        &bound,
        &ids,
        &tables,
        ActivationCheckpoint::Off,
    )
    .unwrap();
    assert_eq!(h.shape(), &[2, ids.shape()[1], s.hidden]);
    let flat = eval.reshape(&h, &[2 * ids.shape()[1], s.hidden]).unwrap();
    let flat_targets = targets.reshape(&[2 * ids.shape()[1]]).unwrap();
    let split = eval
        .lin_ce(
            &flat,
            &bound.embed,
            &flat_targets,
            Some(IGNORE),
            DEFAULT_CE_CHUNK,
        )
        .unwrap();
    let whole = forward_loss(
        &mut eval,
        &s,
        &bound,
        &ids,
        &targets,
        &tables,
        Some(IGNORE),
        DEFAULT_CE_CHUNK,
        ActivationCheckpoint::Off,
    )
    .unwrap();
    assert_eq!(
        split.to_f32_vec().unwrap()[0].to_bits(),
        whole.to_f32_vec().unwrap()[0].to_bits()
    );
}

#[test]
fn forward_hidden_refuses_what_forward_loss_refused() {
    let s = spec();
    let params = host();
    let tables = Qwen35Tables::new(&s, 8, &budget()).unwrap();
    let mut eval = Eval::new(cpu());
    let bound = bind(&mut eval, &params).unwrap();
    let run = |eval: &mut Eval<_>, ids: &Tensor, tables: &Qwen35Tables, bound: &_| {
        forward_hidden(eval, &s, bound, ids, tables, ActivationCheckpoint::Off)
    };
    let rank1 = Tensor::from_u32(&[1, 2, 3], &[3], &budget()).unwrap();
    assert!(matches!(
        run(&mut eval, &rank1, &tables, &bound),
        Err(OjasError::Shape { .. })
    ));
    let long = Tensor::from_u32(&[1; 9], &[1, 9], &budget()).unwrap();
    assert!(matches!(
        run(&mut eval, &long, &tables, &bound),
        Err(OjasError::Shape { .. })
    ));
    let mut short = bound;
    short.layers.pop();
    let ok = Tensor::from_u32(&[1; 4], &[1, 4], &budget()).unwrap();
    assert!(matches!(
        run(&mut eval, &ok, &tables, &short),
        Err(OjasError::Shape { .. })
    ));
}

#[test]
fn forward_loss_still_refuses_mismatched_targets() {
    let s = spec();
    let tables = Qwen35Tables::new(&s, 8, &budget()).unwrap();
    let mut eval = Eval::new(cpu());
    let bound = bind(&mut eval, &host()).unwrap();
    let ids = Tensor::from_u32(&[1; 4], &[1, 4], &budget()).unwrap();
    let targets = Tensor::from_u32(&[1; 3], &[1, 3], &budget()).unwrap();
    let err = forward_loss(
        &mut eval,
        &s,
        &bound,
        &ids,
        &targets,
        &tables,
        None,
        DEFAULT_CE_CHUNK,
        ActivationCheckpoint::Off,
    )
    .unwrap_err();
    assert!(format!("{err:?}").contains("targets"), "{err:?}");
}

#[test]
fn a_bf16_tied_table_gives_forward_hidden_the_f32_tables_bits() {
    let s = spec();
    let model = SafeTensors::open(&dir().join("model.safetensors")).unwrap();
    let f32_params = load_hf(&s, &model, PREFIX, DType::F32, DType::F32, &budget()).unwrap();
    let bf16_params = load_hf(&s, &model, PREFIX, DType::F32, DType::Bf16, &budget()).unwrap();
    assert_eq!(bf16_params.embed.dtype(), DType::Bf16);
    let c = case(true);
    let hidden = |p: &Qwen35Params<Tensor>| {
        let tables = Qwen35Tables::new(&s, c.rows.seq(), &budget()).unwrap();
        let mut eval = Eval::new(cpu());
        let bound = bind(&mut eval, p).unwrap();
        forward_hidden(
            &mut eval,
            &s,
            &bound,
            &c.ids,
            &tables,
            ActivationCheckpoint::Off,
        )
        .unwrap()
        .to_f32_vec()
        .unwrap()
    };
    assert_eq!(bits(&hidden(&f32_params)), bits(&hidden(&bf16_params)));
    // And on a tape, frozen, through the gathered head: the bits agree there too.
    let go = |p: &Qwen35Params<Tensor>| {
        step(
            cpu(),
            &Tensor::clone,
            p,
            &Base::Frozen,
            &c.ids,
            &c.rows,
            &c.positions,
            &c.head,
            ActivationCheckpoint::Off,
            true,
        )
    };
    let (a, b) = (go(&f32_params), go(&bf16_params));
    assert_eq!(a.loss.to_bits(), b.loss.to_bits());
    assert_eq!(bits(&a.dh), bits(&b.dh));
    assert_eq!(bits(&a.d17_grad), bits(&b.d17_grad));
    assert_eq!(a.full_loss.to_bits(), b.full_loss.to_bits());
}

// ---- gather_rows: forward and backward --------------------------------------

#[test]
fn gather_rows_is_index_select_and_its_scatter_on_the_cpu() {
    check_gather(cpu(), &Tensor::clone);
}

#[test]
fn two_examples_at_the_same_position_do_not_alias() {
    check_same_position(cpu(), &Tensor::clone);
}

#[test]
fn gather_rows_refuses_what_it_cannot_read() {
    let mut tape = Tape::new(cpu());
    let bud = budget();
    let h = Tensor::from_f32(&vec![0.5; 3 * 4 * 8], &[3, 4, 8], &bud).unwrap();
    let h = tape.leaf(h).unwrap();
    let rows = RightPadded::new(vec![4, 2, 1], 4).unwrap();
    let go = |tape: &mut Tape<_>, rows: &RightPadded, pos: &[usize]| {
        gather_rows(tape, &h, rows, pos, &bud)
    };
    assert!(go(&mut tape, &rows, &[3, 1, 0]).is_ok());
    for bad in [
        &[3usize, 1][..],
        &[3, 1, 0, 0],
        &[],
        &[4, 1, 0],
        &[3, 2, 0],
        &[3, 1, 1],
    ] {
        assert!(
            matches!(go(&mut tape, &rows, bad), Err(OjasError::Shape { .. })),
            "{bad:?}"
        );
    }
    // `h` of another batch or length than `rows` describes.
    for other in [
        RightPadded::new(vec![4, 2], 4).unwrap(),
        RightPadded::new(vec![3, 2, 1], 3).unwrap(),
    ] {
        assert!(
            matches!(go(&mut tape, &other, &[0, 0]), Err(OjasError::Shape { .. }))
                || matches!(
                    go(&mut tape, &other, &[0, 0, 0]),
                    Err(OjasError::Shape { .. })
                )
        );
    }
    let flat = tape
        .leaf(Tensor::from_f32(&[0.0; 12 * 8], &[12, 8], &bud).unwrap())
        .unwrap();
    assert!(matches!(
        gather_rows(&mut tape, &flat, &rows, &[0, 0, 0], &bud),
        Err(OjasError::Shape { .. })
    ));
}

// ---- RightPadded ------------------------------------------------------------

#[test]
fn an_attention_mask_gives_right_padded_lengths() {
    let mask = [1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1];
    let r = RightPadded::from_attention_mask(&mask, 3, 4).unwrap();
    assert_eq!((r.lengths(), r.batch(), r.seq()), (&[3, 1, 4][..], 3, 4));
    assert_eq!(RightPadded::full(2, 5).unwrap().lengths(), &[5, 5]);
}

#[test]
fn an_attention_mask_that_is_not_right_padding_is_refused() {
    let refused = |mask: &[i64], batch: usize, seq: usize, why: &str| {
        let err = RightPadded::from_attention_mask(mask, batch, seq).unwrap_err();
        assert!(matches!(err, OjasError::Shape { .. }), "{why}: {err:?}");
        assert!(format!("{err:?}").contains(why), "{why}: {err:?}");
    };
    refused(&[0, 1, 1, 1], 1, 4, "left padded");
    refused(&[0, 0, 1, 1], 1, 4, "left padded");
    refused(&[1, 0, 1, 1], 1, 4, "after padding");
    refused(&[1, 1, 0, 1], 1, 4, "after padding");
    refused(&[1, 1, 1, 1, 0, 0, 0, 0], 2, 4, "all padding");
    refused(&[1, 1, 2, 0], 1, 4, "not 0 or 1");
    refused(&[1, 1, -1, 0], 1, 4, "not 0 or 1");
    refused(&[1, 1, 1], 1, 4, "mask values");
    refused(&[], 0, 4, "0 rows");
}

#[test]
fn lengths_and_positions_are_bounded() {
    for bad in [vec![], vec![0], vec![5], vec![3, 0]] {
        assert!(RightPadded::new(bad.clone(), 4).is_err(), "{bad:?}");
    }
    assert!(RightPadded::new(vec![1], 0).is_err());
    let r = RightPadded::new(vec![4, 2, 1], 4).unwrap();
    assert_eq!(r.flat_ids(&[3, 1, 0]).unwrap(), vec![3, 4 + 1, 8]);
    assert!(
        r.flat_ids(&[3, 2, 0]).is_err(),
        "a position at its row's length"
    );
    assert!(r.flat_ids(&[4, 1, 0]).is_err());
    assert!(r.flat_ids(&[]).is_err(), "no positions");
    let one = RightPadded::new(vec![1], 1).unwrap();
    assert_eq!(one.flat_ids(&[0]).unwrap(), vec![0], "B = 1, T = 1");
}

#[cfg(target_pointer_width = "64")]
#[test]
fn a_flat_index_past_u32_is_refused_before_anything_is_allocated() {
    let max = u32::MAX as usize;
    // B * T = 2^32 - 1 is the last count refused-free; one more is refused.
    assert!(RightPadded::new(vec![1], max).is_ok());
    let r = RightPadded::new(vec![1, 1, 1], max / 3).unwrap();
    assert_eq!(r.batch() * r.seq(), max);
    assert!(RightPadded::new(vec![1, 1], 1 << 31).is_err());
    assert!(RightPadded::new(vec![1; 3], max / 3 + 1).is_err());
    // Far past it, and past usize: no overflow, no allocation of that many rows.
    assert!(RightPadded::new(vec![1, 1], usize::MAX).is_err());
    let last = RightPadded::new(vec![max], max).unwrap();
    assert_eq!(last.flat_ids(&[max - 1]).unwrap(), vec![u32::MAX - 1]);
}

// ---- right padding is harmless ----------------------------------------------

#[test]
fn right_padding_cannot_reach_the_readout() {
    check_padding_is_inert(cpu(), &Tensor::clone, &Base::Trainable, EXACT);
}

#[test]
fn a_right_padded_batch_gives_each_example_its_own_readout_on_the_cpu() {
    check_batch_against_alone(cpu(), &Tensor::clone, &Base::Trainable, EXACT);
}

#[test]
fn the_same_holds_with_lora_adapters() {
    let sites = fixture_sites(&lora_file(), true);
    check_batch_against_alone(cpu(), &Tensor::clone, &Base::Lora(&sites), EXACT);
    check_padding_is_inert(cpu(), &Tensor::clone, &Base::Lora(&sites), EXACT);
}

// ---- LoRA and checkpointing -------------------------------------------------

#[test]
fn zero_b_adapters_give_the_base_hidden_state_and_none_is_the_base() {
    let s = spec();
    let c = case(true);
    let tables = Qwen35Tables::new(&s, c.rows.seq(), &budget()).unwrap();
    let params = host();
    let sites = fixture_sites(&lora_file(), false);
    let mut eval = Eval::new(cpu());
    let bound = bind(&mut eval, &params).unwrap();
    let base = forward_hidden(
        &mut eval,
        &s,
        &bound,
        &c.ids,
        &tables,
        ActivationCheckpoint::Off,
    )
    .unwrap();
    let none = forward_hidden_lora(
        &mut eval,
        &s,
        &bound,
        None,
        &c.ids,
        &tables,
        ActivationCheckpoint::Off,
    )
    .unwrap();
    assert_eq!(
        bits(&base.to_f32_vec().unwrap()),
        bits(&none.to_f32_vec().unwrap())
    );
    let bound_sites = sites
        .try_map(&mut |site| site.try_map(&mut |t| eval.param(t)))
        .unwrap();
    let mut rng = ojas_data::CounterRng::new(0);
    let fwd = LoraForward::new(
        &mut eval,
        bound_sites,
        &lora_common::config(),
        c.rows.batch(),
        c.rows.seq(),
        &mut rng,
        &budget(),
    )
    .unwrap();
    let with = forward_hidden_lora(
        &mut eval,
        &s,
        &bound,
        Some(&fwd),
        &c.ids,
        &tables,
        ActivationCheckpoint::Off,
    )
    .unwrap();
    assert_eq!(
        bits(&base.to_f32_vec().unwrap()),
        bits(&with.to_f32_vec().unwrap())
    );
}

#[test]
fn checkpointed_layers_give_the_readout_the_same_bits() {
    let c = case(true);
    for base in [Base::Trainable, Base::Frozen] {
        let (off, blocks) = (
            run_case(&c, &base, ActivationCheckpoint::Off),
            run_case(&c, &base, ActivationCheckpoint::Blocks),
        );
        assert_eq!(off.loss.to_bits(), blocks.loss.to_bits());
        assert_eq!(bits(&off.hidden), bits(&blocks.hidden));
        assert_eq!(bits(&off.dh), bits(&blocks.dh));
        assert_eq!(bits(&off.d17_grad), bits(&blocks.d17_grad));
        for (a, b) in off.param_grads.iter().zip(&blocks.param_grads) {
            assert_eq!(bits(a), bits(b));
        }
    }
    let sites = fixture_sites(&lora_file(), true);
    let lora = Base::Lora(&sites);
    let (off, blocks) = (
        run_case(&c, &lora, ActivationCheckpoint::Off),
        run_case(&c, &lora, ActivationCheckpoint::Blocks),
    );
    assert_eq!(bits(&off.hidden), bits(&blocks.hidden));
    for (a, b) in off.param_grads.iter().zip(&blocks.param_grads) {
        assert_eq!(bits(a), bits(b));
    }
}

// ---- the answer head against torch ------------------------------------------

#[test]
fn the_composed_answer_head_matches_torch() {
    check_against_torch(cpu(), &Tensor::clone, LOSS_BOUND, GRAD_BOUND);
}

#[test]
fn the_full_vocabulary_value_adds_no_gradient() {
    check_full_vocab_is_inert(cpu(), &Tensor::clone);
}

#[test]
fn recording_the_full_vocabulary_value_changes_no_bit_of_the_training_step() {
    let c = case(true);
    let with = run_case(&c, &Base::Frozen, ActivationCheckpoint::Off);
    let without = step(
        cpu(),
        &Tensor::clone,
        &host(),
        &Base::Frozen,
        &c.ids,
        &c.rows,
        &c.positions,
        &c.head,
        ActivationCheckpoint::Off,
        false,
    );
    assert!(without.full_loss.is_nan() && with.full_loss.is_finite());
    assert_eq!(with.loss.to_bits(), without.loss.to_bits());
    assert_eq!(bits(&with.dh), bits(&without.dh));
    assert_eq!(bits(&with.d17_grad), bits(&without.d17_grad));
}

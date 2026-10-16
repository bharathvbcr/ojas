//! The Qwen3.5 hybrid tower on the CPU tape against transformers' autograd.
//!
//! The fixture is `tests/fixtures/qwen35_tiny` (see `qwen35_common`): a random
//! `Qwen3_5ForCausalLM` of the 2B's shape family (head_dim 256, two GDN heads
//! of 128, grouped KV heads, tied embeddings, vocab 8320), one gated delta net
//! layer and one gated attention layer, as a bf16 checkpoint, run at batch 2
//! with `-100` labels on a prompt and on right padding, with transformers'
//! float32 loss and every parameter's gradient. Every parameter is non-trivial
//! in the fixture, so a weight applied in the wrong place (a `1 + w` norm, a
//! fused row block) shows in that tensor's gradient.
//!
//! Bounds are tessl's for the same fixture against the same reference
//! (`tests/qwen35_train.rs`): loss within 1e-5 relative, and each gradient
//! within 1e-4 of its own peak.

mod qwen35_common;

use ojas_core::{bf16_to_f32, Budget, DType, OjasError, Tensor};
use ojas_io::SafeTensors;
use ojas_model::qwen35::{
    bind, forward_loss, hf_tensors, load_hf, MixerParams, Qwen35Mixer, Qwen35Params, Qwen35Tables,
};
use ojas_model::{ActivationCheckpoint, Eval, DEFAULT_CE_CHUNK};
use qwen35_common::*;

#[test]
fn the_name_table_is_the_checkpoints_tower() {
    let model = SafeTensors::open(&dir().join("model.safetensors")).unwrap();
    let mut want: Vec<String> = model.names().map(str::to_string).collect();
    let mut got: Vec<String> = hf_tensors(&spec(), PREFIX)
        .into_iter()
        .map(|t| t.name)
        .collect();
    want.sort();
    got.sort();
    assert_eq!(got, want);
}

#[test]
fn the_cpu_tape_matches_transformers() {
    let r = run(cpu(), Tensor::clone, ActivationCheckpoint::Off);
    let want = reference_loss();
    let rel = (f64::from(r.loss) - want).abs() / want.abs();
    eprintln!("loss {} vs transformers {want} (rel {rel:.2e})", r.loss);
    assert!(rel <= LOSS_BOUND, "loss {} vs {want}", r.loss);
    let worst = compare_to_reference(&r.grads, GRAD_BOUND);
    eprintln!("worst gradient {worst:.3e} of its peak");
}

fn bits(p: &Qwen35Params<Tensor>) -> Vec<Vec<u32>> {
    p.values()
        .into_iter()
        .map(|t| {
            t.to_f32_vec()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect()
        })
        .collect()
}

#[test]
fn checkpointed_layers_give_the_same_bits_on_the_cpu() {
    let direct = run(cpu(), Tensor::clone, ActivationCheckpoint::Off);
    let ckpt = run(cpu(), Tensor::clone, ActivationCheckpoint::Blocks);
    assert_eq!(direct.loss.to_bits(), ckpt.loss.to_bits());
    assert_eq!(bits(&direct.grads), bits(&ckpt.grads));
}

#[test]
fn eval_gives_the_tapes_loss_bits() {
    let s = spec();
    let model = SafeTensors::open(&dir().join("model.safetensors")).unwrap();
    let reference = SafeTensors::open(&dir().join("reference.safetensors")).unwrap();
    let params = load_hf(&s, &model, PREFIX, DType::F32, DType::F32, &budget()).unwrap();
    let (ids, targets) = ids_and_targets(&reference);
    let tables = Qwen35Tables::new(&s, ids.shape()[1], &budget()).unwrap();
    let mut eval = Eval::new(cpu());
    let bound = bind(&mut eval, &params).unwrap();
    let loss = forward_loss(
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
    let tape = run(cpu(), Tensor::clone, ActivationCheckpoint::Off);
    assert_eq!(loss.to_f32_vec().unwrap()[0].to_bits(), tape.loss.to_bits());
}

#[test]
fn a_mismatched_checkpoint_is_refused_by_name() {
    let model = SafeTensors::open(&dir().join("model.safetensors")).unwrap();
    let mut s = spec();
    s.layers = vec![Qwen35Mixer::Attention, Qwen35Mixer::GatedDeltaNet];
    let err = load_hf(&s, &model, PREFIX, DType::F32, DType::F32, &budget()).unwrap_err();
    assert!(
        format!("{err:?}").contains("model.layers.0.self_attn.q_proj.weight"),
        "{err:?}"
    );
    let mut s = spec();
    s.head_dim = 128;
    s.rotary_dim = 32;
    let err = load_hf(&s, &model, PREFIX, DType::F32, DType::F32, &budget()).unwrap_err();
    assert!(format!("{err:?}").contains("shape"), "{err:?}");
}

/// Every weight a `linear` reads, in one fixed order.
fn linears(p: &Qwen35Params<Tensor>) -> Vec<&Tensor> {
    let mut out = Vec::new();
    for l in &p.layers {
        match &l.mixer {
            MixerParams::GatedDeltaNet(m) => {
                out.extend([&m.wq, &m.wk, &m.wv, &m.wz, &m.wa, &m.wb, &m.out])
            }
            MixerParams::Attention(m) => out.extend([&m.wq, &m.wgate, &m.wk, &m.wv, &m.wo]),
        }
        out.extend([&l.gate, &l.up, &l.down]);
    }
    out
}

fn elems(t: &Tensor) -> u64 {
    t.num_elements().unwrap() as u64
}

#[test]
fn a_bf16_frozen_base_holds_its_linears_at_two_bytes_and_keeps_the_rest_f32() {
    let s = spec();
    let model = SafeTensors::open(&dir().join("model.safetensors")).unwrap();
    let (fb, hb) = (Budget::new(1 << 30), Budget::new(1 << 30));
    let full = load_hf(&s, &model, PREFIX, DType::F32, DType::F32, &fb).unwrap();
    let half = load_hf(&s, &model, PREFIX, DType::Bf16, DType::F32, &hb).unwrap();

    let lin_half = linears(&half);
    let lin_elems: u64 = lin_half.iter().map(|t| elems(t)).sum();
    let all_elems: u64 = half.values().iter().map(|t| elems(t)).sum();
    assert_eq!(lin_half.len(), 7 + 5 + 2 * 3, "every linear of both layers");
    for t in &lin_half {
        assert_eq!(t.dtype(), DType::Bf16);
    }
    // The embedding, the norms, the convolutions, A_log and dt_bias: f32.
    let lin_ptrs: Vec<*const Tensor> = lin_half.iter().map(|t| *t as *const Tensor).collect();
    let rest: Vec<&Tensor> = half
        .values()
        .into_iter()
        .filter(|t| !lin_ptrs.contains(&(*t as *const Tensor)))
        .collect();
    assert_eq!(rest.len(), half.values().len() - lin_half.len());
    for t in &rest {
        assert_eq!(t.dtype(), DType::F32, "shape {:?}", t.shape());
    }
    assert_eq!(half.embed.dtype(), DType::F32);

    // Resident bytes: 2 per linear element, 4 for the rest.
    assert_eq!(fb.live_bytes().unwrap(), 4 * all_elems);
    assert_eq!(
        hb.live_bytes().unwrap(),
        2 * lin_elems + 4 * (all_elems - lin_elems)
    );

    // The checkpoint is bf16: its bits are kept, and widen to the f32 load.
    for (h, f) in lin_half.iter().zip(linears(&full)) {
        let widened: Vec<u32> = h
            .bf16_slice()
            .unwrap()
            .iter()
            .map(|&b| bf16_to_f32(b).to_bits())
            .collect();
        let want: Vec<u32> = f
            .to_f32_vec()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(widened, want);
    }

    let err = load_hf(&s, &model, PREFIX, DType::U32, DType::F32, &budget()).unwrap_err();
    assert!(
        matches!(
            err,
            ojas_core::OjasError::Dtype {
                got: DType::U32,
                ..
            }
        ),
        "{err:?}"
    );
}

#[test]
fn a_bf16_frozen_base_gives_the_f32_loss_and_gradient_bits_on_the_cpu() {
    let full = run(cpu(), Tensor::clone, ActivationCheckpoint::Blocks);
    let half = run_with(
        cpu(),
        Tensor::clone,
        ActivationCheckpoint::Blocks,
        DType::Bf16,
        DType::F32,
    );
    assert_eq!(half.loss.to_bits(), full.loss.to_bits());
    assert_eq!(bits(&half.grads), bits(&full.grads));
}

#[test]
fn a_bf16_tied_table_holds_two_bytes_per_element_and_keeps_its_bits() {
    let s = spec();
    let model = SafeTensors::open(&dir().join("model.safetensors")).unwrap();
    let (fb, hb) = (Budget::new(1 << 30), Budget::new(1 << 30));
    let full = load_hf(&s, &model, PREFIX, DType::F32, DType::F32, &fb).unwrap();
    let half = load_hf(&s, &model, PREFIX, DType::Bf16, DType::Bf16, &hb).unwrap();

    assert_eq!(half.embed.dtype(), DType::Bf16);
    assert_eq!(half.embed.shape(), full.embed.shape());
    let table = elems(&half.embed);
    let lin: u64 = linears(&half).iter().map(|t| elems(t)).sum();
    let all: u64 = half.values().iter().map(|t| elems(t)).sum();
    // Resident bytes: 2 per table and linear element, 4 for the rest.
    assert_eq!(fb.live_bytes().unwrap(), 4 * all);
    assert_eq!(
        hb.live_bytes().unwrap(),
        2 * (table + lin) + 4 * (all - table - lin)
    );

    // The checkpoint is bf16: the table's bits are kept, and widen to the
    // f32 load.
    let widened: Vec<u32> = half
        .embed
        .bf16_slice()
        .unwrap()
        .iter()
        .map(|&b| bf16_to_f32(b).to_bits())
        .collect();
    let want: Vec<u32> = full
        .embed
        .to_f32_vec()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect();
    assert_eq!(widened, want);

    // The table alone may be Bf16 too; any other table storage is refused.
    let only = load_hf(&s, &model, PREFIX, DType::F32, DType::Bf16, &budget()).unwrap();
    assert_eq!(only.embed.dtype(), DType::Bf16);
    assert!(linears(&only).iter().all(|t| t.dtype() == DType::F32));
    for bad in [DType::U32, DType::F16] {
        let err = load_hf(&s, &model, PREFIX, DType::F32, bad, &budget()).unwrap_err();
        assert!(
            matches!(err, OjasError::Dtype { got, .. } if got == bad),
            "{err:?}"
        );
    }
}

#[test]
fn a_bf16_tied_table_gives_the_f32_loss_and_gradient_bits_on_the_cpu() {
    for activations in [ActivationCheckpoint::Off, ActivationCheckpoint::Blocks] {
        let full = run(cpu(), Tensor::clone, activations);
        let half = run_with(cpu(), Tensor::clone, activations, DType::Bf16, DType::Bf16);
        assert_eq!(half.loss.to_bits(), full.loss.to_bits());
        assert_eq!(bits(&half.grads), bits(&full.grads));
        assert_eq!(half.grads.embed.dtype(), DType::F32);
    }
}

fn loss_with(
    ignore: Option<u32>,
    targets_of: impl Fn(&Tensor) -> Tensor,
) -> Result<f32, OjasError> {
    let s = spec();
    let model = SafeTensors::open(&dir().join("model.safetensors")).unwrap();
    let reference = SafeTensors::open(&dir().join("reference.safetensors")).unwrap();
    let params = load_hf(&s, &model, PREFIX, DType::F32, DType::F32, &budget()).unwrap();
    let (ids, targets) = ids_and_targets(&reference);
    let targets = targets_of(&targets);
    let tables = Qwen35Tables::new(&s, ids.shape()[1], &budget()).unwrap();
    let mut eval = Eval::new(cpu());
    let bound = bind(&mut eval, &params).unwrap();
    let loss = forward_loss(
        &mut eval,
        &s,
        &bound,
        &ids,
        &targets,
        &tables,
        ignore,
        DEFAULT_CE_CHUNK,
        ActivationCheckpoint::Off,
    )?;
    Ok(loss.to_f32_vec().unwrap()[0])
}

/// Without an ignore index the `-100` labels are targets outside the
/// vocabulary: refused, never trained on or silently dropped.
#[test]
fn masked_labels_without_an_ignore_index_are_refused() {
    let err = loss_with(None, Tensor::clone).unwrap_err();
    assert!(matches!(err, OjasError::OutOfRange { .. }), "{err:?}");
}

/// The ignore index is whatever the caller says, and only it masks: the same
/// positions under another ignore value give the same bits, and scoring them
/// as token 5 instead moves the loss.
#[test]
fn only_the_ignore_index_masks_a_position() {
    let masked = loss_with(Some(IGNORE), Tensor::clone).unwrap();
    let remap = |to: u32| {
        move |t: &Tensor| {
            let v: Vec<u32> = t
                .to_u32_vec()
                .unwrap()
                .into_iter()
                .map(|x| if x == IGNORE { to } else { x })
                .collect();
            Tensor::from_u32(&v, t.shape(), &budget()).unwrap()
        }
    };
    let other = loss_with(Some(8319), remap(8319)).unwrap();
    assert_eq!(masked.to_bits(), other.to_bits());
    let scored = loss_with(None, remap(5)).unwrap();
    assert!((scored - masked).abs() > 1e-3, "{scored} vs {masked}");
}

/// A batch whose every target is ignored has no mean: refused, not zero.
#[test]
fn a_batch_with_every_target_ignored_is_refused() {
    let err = loss_with(Some(IGNORE), |t| {
        let n = t.num_elements().unwrap();
        Tensor::from_u32(&vec![IGNORE; n], t.shape(), &budget()).unwrap()
    })
    .unwrap_err();
    assert!(matches!(err, OjasError::NonFinite { .. }), "{err:?}");
}

//! The Qwen3.5 hybrid tower on the CPU tape against transformers' autograd.
//!
//! The fixture is tessl's committed `tests/fixtures/qwen35_train` (tessl
//! 56f3058, `make_train_fixture.py tiny`): a random `Qwen3_5ForCausalLM` of
//! the 2B's shape family (head_dim 256, GDN heads of 128, grouped KV heads,
//! tied embeddings), one gated delta net layer and one gated attention
//! layer, as a bf16 checkpoint, with transformers' float32 loss over
//! `labels=ids` and every parameter's gradient. `reference.safetensors`
//! packs `ids.npy`, `loss.npy` (rounded to f32) and the 27 `grad.*.npy` under
//! their Hugging Face names (`qwen35_common`). Every parameter is non-trivial in the fixture,
//! so a weight applied in the wrong place (a `1 + w` norm, a fused row
//! block) shows in that tensor's gradient.
//!
//! Bounds are tessl's for the same fixture against the same reference
//! (`tests/qwen35_train.rs`): loss within 1e-5 relative, and each gradient
//! within 1e-4 of its own peak.

mod qwen35_common;

use ojas_core::Tensor;
use ojas_io::SafeTensors;
use ojas_model::qwen35::{
    bind, forward_loss, hf_tensors, load_hf, Qwen35Mixer, Qwen35Params, Qwen35Tables,
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
    let params = load_hf(&s, &model, PREFIX, &budget()).unwrap();
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
    let err = load_hf(&s, &model, PREFIX, &budget()).unwrap_err();
    assert!(
        format!("{err:?}").contains("model.layers.0.self_attn.q_proj.weight"),
        "{err:?}"
    );
    let mut s = spec();
    s.head_dim = 128;
    s.rotary_dim = 32;
    let err = load_hf(&s, &model, PREFIX, &budget()).unwrap_err();
    assert!(format!("{err:?}").contains("shape"), "{err:?}");
}

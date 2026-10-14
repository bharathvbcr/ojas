//! The Qwen3.5 fixture shared by `qwen35_fixture.rs` (CPU) and
//! `qwen35_metal_tape.rs`: `tests/fixtures/qwen35_tiny`, made by
//! `ojas-oracle/python/qwen35_wide_fixture.py` (transformers 5.15, torch 2.13).
//! A random `Qwen3_5ForCausalLM` of the 2B's shape family, one gated delta net
//! layer and one gated attention layer, widened to what a fine-tune uses:
//! batch 2, two GDN heads, vocab 8320 (past one 8192-column fused-CE tile) and
//! `-100` labels (a masked prompt in row 0, masked padding in row 1).
//! `reference.safetensors` packs `ids` and `labels` (int64 `[2, 70]`), `loss`
//! (transformers' `cross_entropy(ignore_index=-100)`, rounded to f32) and the
//! 27 gradients under their Hugging Face names.
//!
//! Each test binary compiles this module and uses part of it.
#![allow(dead_code)]

pub mod lora;

use std::path::PathBuf;

use ojas_autograd::Tape;
use ojas_core::{Backend, Budget, DType, Numerics, Tensor};
use ojas_cpu::CpuBackend;
use ojas_io::SafeTensors;
use ojas_model::qwen35::{
    bind, forward_loss, fuse_grads, load_hf, Qwen35Mixer, Qwen35Params, Qwen35Spec, Qwen35Tables,
};
use ojas_model::{ActivationCheckpoint, DEFAULT_CE_CHUNK};

pub const LOSS_BOUND: f64 = 1e-5;
pub const GRAD_BOUND: f32 = 1e-4;
pub const PREFIX: &str = "model.";

pub fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen35_tiny")
}

/// The fixture's `config.json`, as `Qwen35TextConfig::tape_spec` gives it:
/// hidden 64, intermediate 128, vocab 8320, `[linear_attention,
/// full_attention]`, 2 query heads over 1 KV head of 256, partial rotary
/// 0.25, rope theta 1e4, two GDN heads of 128 by 128, conv width 4.
pub fn spec() -> Qwen35Spec {
    Qwen35Spec {
        vocab: 8320,
        hidden: 64,
        intermediate: 128,
        layers: vec![Qwen35Mixer::GatedDeltaNet, Qwen35Mixer::Attention],
        q_heads: 2,
        kv_heads: 1,
        head_dim: 256,
        rotary_dim: 64,
        rope_theta: 1e4,
        gdn_heads: 2,
        gdn_key_dim: 128,
        gdn_value_dim: 128,
        conv_width: 4,
        eps: 1e-6,
    }
}

pub fn budget() -> Budget {
    Budget::new(1 << 30)
}

/// transformers' ignore index, `-100`, as the `u32` the tape's targets carry.
pub const IGNORE: u32 = -100i32 as u32;

/// `(ids, targets)` `[2, T - 1]`: transformers' shift of `labels` (`ids` are
/// the first `T - 1` columns, `targets` the labels from column 1), with
/// `-100` kept as [`IGNORE`].
pub fn ids_and_targets(reference: &SafeTensors<'_>) -> (Tensor, Tensor) {
    let (shape, ids) = reference.read_i64("ids").unwrap();
    let (lshape, labels) = reference.read_i64("labels").unwrap();
    assert_eq!(shape, lshape);
    let (rows, t) = (shape[0] as usize, shape[1] as usize);
    let (mut inp, mut tgt) = (Vec::new(), Vec::new());
    for b in 0..rows {
        for i in 0..t - 1 {
            inp.push(u32::try_from(ids[b * t + i]).unwrap());
            let l = labels[b * t + i + 1];
            tgt.push(if l == -100 {
                IGNORE
            } else {
                u32::try_from(l).unwrap()
            });
        }
    }
    let bud = budget();
    (
        Tensor::from_u32(&inp, &[rows, t - 1], &bud).unwrap(),
        Tensor::from_u32(&tgt, &[rows, t - 1], &bud).unwrap(),
    )
}

pub struct Run {
    pub loss: f32,
    pub grads: Qwen35Params<Tensor>,
}

/// One forward and backward on `backend`'s tape; parameters and tables are
/// uploaded first (`up`).
pub fn run<B: Backend>(
    backend: B,
    up: impl Fn(&Tensor) -> Tensor,
    activations: ActivationCheckpoint,
) -> Run {
    run_with(backend, up, activations, DType::F32)
}

/// [`run`] with the linear weights loaded as `linears` (`load_hf`).
pub fn run_with<B: Backend>(
    backend: B,
    up: impl Fn(&Tensor) -> Tensor,
    activations: ActivationCheckpoint,
    linears: DType,
) -> Run {
    let s = spec();
    let model = SafeTensors::open(&dir().join("model.safetensors")).unwrap();
    let reference = SafeTensors::open(&dir().join("reference.safetensors")).unwrap();
    let host = load_hf(&s, &model, PREFIX, linears, &budget()).unwrap();
    let (ids, targets) = ids_and_targets(&reference);
    let tables = Qwen35Tables::new(&s, ids.shape()[1], &budget()).unwrap();
    let tables = Qwen35Tables {
        cos: up(&tables.cos),
        sin: up(&tables.sin),
        ones_hidden: up(&tables.ones_hidden),
        ones_head: up(&tables.ones_head),
    };
    let mut tape = Tape::new(backend);
    let device = host.try_map(&mut |t| Ok::<_, ()>(up(t))).unwrap();
    let vars = bind(&mut tape, &device).unwrap();
    let loss = forward_loss(
        &mut tape,
        &s,
        &vars,
        &ids,
        &targets,
        &tables,
        Some(IGNORE),
        DEFAULT_CE_CHUNK,
        activations,
    )
    .unwrap();
    tape.backward(loss).unwrap();
    let loss = tape
        .value(loss)
        .unwrap()
        .to_host(&budget())
        .unwrap()
        .to_f32_vec()
        .unwrap()[0];
    let grads = vars
        .try_map(&mut |v| tape.grad(*v).cloned().ok_or("no gradient"))
        .unwrap();
    Run { loss, grads }
}

pub fn cpu() -> CpuBackend {
    CpuBackend::new(Budget::new(4 << 30)).with_numerics(Numerics::Exact)
}

/// Every gradient of `got` against the fixture, by Hugging Face name; each
/// within `bound` of its own peak. Returns the worst ratio seen.
pub fn compare_to_reference(got: &Qwen35Params<Tensor>, bound: f32) -> f32 {
    let reference = SafeTensors::open(&dir().join("reference.safetensors")).unwrap();
    let fused = fuse_grads(&spec(), got, PREFIX, &budget()).unwrap();
    let mut worst = 0.0f32;
    let mut seen = Vec::new();
    for g in &fused {
        let (shape, want) = reference.read_f32(&g.name).unwrap();
        let shape: Vec<usize> = shape.iter().map(|&n| n as usize).collect();
        assert_eq!(shape, g.shape, "{}", g.name);
        let peak = want.iter().fold(0.0f32, |p, x| p.max(x.abs()));
        assert!(
            peak > 0.0,
            "{}: an all-zero reference checks nothing",
            g.name
        );
        let err = g
            .values
            .iter()
            .zip(&want)
            .map(|(a, b)| {
                assert!(a.is_finite(), "{}: non-finite", g.name);
                (a - b).abs()
            })
            .fold(0.0f32, f32::max);
        let ratio = err / peak;
        eprintln!("{}: {ratio:.3e} of peak {peak:.3e}", g.name);
        assert!(ratio <= bound, "{}: {ratio:.3e} of its peak", g.name);
        worst = worst.max(ratio);
        seen.push(g.name.clone());
    }
    let mut files: Vec<String> = reference
        .names()
        .filter(|n| n.starts_with(PREFIX))
        .map(str::to_string)
        .collect();
    files.sort();
    seen.sort();
    assert_eq!(seen, files, "every reference gradient is compared, once");
    assert_eq!(seen.len(), 27);
    worst
}

pub fn reference_loss() -> f64 {
    let reference = SafeTensors::open(&dir().join("reference.safetensors")).unwrap();
    f64::from(reference.read_f32("loss").unwrap().1[0])
}

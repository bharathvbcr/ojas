//! The Qwen3.5 fixture shared by `qwen35_fixture.rs` (CPU) and
//! `qwen35_metal_tape.rs`: tessl's committed `tests/fixtures/qwen35_train`
//! (tessl 56f3058, `make_train_fixture.py tiny`), copied to
//! `tests/fixtures/qwen35_tiny`. `reference.safetensors` packs its
//! `ids.npy`, `loss.npy` (rounded to f32) and the 27 `grad.*.npy` under their
//! Hugging Face names.
//!
//! Each test binary compiles this module and uses part of it.
#![allow(dead_code)]

use std::path::PathBuf;

use ojas_autograd::Tape;
use ojas_core::{Backend, Budget, Numerics, Tensor};
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
/// hidden 64, intermediate 128, vocab 64, `[linear_attention,
/// full_attention]`, 2 query heads over 1 KV head of 256, partial rotary
/// 0.25, rope theta 1e4, one GDN head of 128 by 128, conv width 4.
pub fn spec() -> Qwen35Spec {
    Qwen35Spec {
        vocab: 64,
        hidden: 64,
        intermediate: 128,
        layers: vec![Qwen35Mixer::GatedDeltaNet, Qwen35Mixer::Attention],
        q_heads: 2,
        kv_heads: 1,
        head_dim: 256,
        rotary_dim: 64,
        rope_theta: 1e4,
        gdn_heads: 1,
        gdn_key_dim: 128,
        gdn_value_dim: 128,
        conv_width: 4,
        eps: 1e-6,
    }
}

pub fn budget() -> Budget {
    Budget::new(1 << 30)
}

/// `(ids, targets)` `[1, T - 1]`: transformers' `labels=ids` shift.
pub fn ids_and_targets(reference: &SafeTensors<'_>) -> (Tensor, Tensor) {
    let (_, ids) = reference.read_i64("ids").unwrap();
    let ids: Vec<u32> = ids.iter().map(|&i| u32::try_from(i).unwrap()).collect();
    let n = ids.len() - 1;
    let b = budget();
    (
        Tensor::from_u32(&ids[..n], &[1, n], &b).unwrap(),
        Tensor::from_u32(&ids[1..], &[1, n], &b).unwrap(),
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
    let s = spec();
    let model = SafeTensors::open(&dir().join("model.safetensors")).unwrap();
    let reference = SafeTensors::open(&dir().join("reference.safetensors")).unwrap();
    let host = load_hf(&s, &model, PREFIX, &budget()).unwrap();
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

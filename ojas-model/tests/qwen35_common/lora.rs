//! Shared LoRA helpers for `qwen35_lora.rs` (CPU) and `qwen35_lora_metal.rs`:
//! the torch LoRA fixture (`tests/fixtures/qwen35_tiny_lora`), the host
//! adapters built from it, and one forward and backward over a frozen base.
#![allow(dead_code)]

use std::path::PathBuf;

use ojas_autograd::Tape;
use ojas_core::{Backend, Tensor};
use ojas_data::CounterRng;
use ojas_io::SafeTensors;
use ojas_model::lora::{LoraConfig, LoraSite};
use ojas_model::qwen35::{
    bind_frozen, forward_loss_lora, load_hf, lora_targets, LoraForward, LoraTarget, Qwen35Lora,
    Qwen35Params, Qwen35Spec, Qwen35Tables, LORA_KINDS,
};
use ojas_model::{ActivationCheckpoint, DEFAULT_CE_CHUNK};

use super::*;

pub const RANK: usize = 4;
pub const ALPHA: f32 = 8.0;
pub const LORA_LOSS_BOUND: f64 = 1e-5;
pub const LORA_GRAD_BOUND: f32 = 1e-5;
pub const STEPS: usize = 20;
pub const LR: f64 = 5e-3;
pub const WD: f64 = 0.01;

pub fn lora_file() -> SafeTensors<'static> {
    let path: PathBuf = dir()
        .parent()
        .unwrap()
        .join("qwen35_tiny_lora/lora.safetensors");
    SafeTensors::open(&path).unwrap()
}

pub fn config() -> LoraConfig {
    LoraConfig::new(RANK, ALPHA, 0.0).unwrap()
}

pub fn base_file() -> SafeTensors<'static> {
    SafeTensors::open(&dir().join("model.safetensors")).unwrap()
}

pub fn targets() -> Vec<LoraTarget> {
    lora_targets(&spec(), &base_file(), PREFIX, &LORA_KINDS).unwrap()
}

pub fn read(file: &SafeTensors<'_>, name: &str) -> Vec<f32> {
    file.read_f32(name).unwrap().1
}

/// Host adapters from fused `A`/`B` values by module name.
pub fn sites_from(
    spec: &Qwen35Spec,
    targets: &[LoraTarget],
    a_of: impl Fn(&str) -> Vec<f32>,
    b_of: impl Fn(&str) -> Vec<f32>,
) -> Qwen35Lora<LoraSite<Tensor>> {
    let bud = budget();
    let mut lora = Qwen35Lora::empty(spec);
    for t in targets {
        let a = Tensor::from_f32(&a_of(t.module()), &[RANK, t.in_features], &bud).unwrap();
        let fused_b = b_of(t.module());
        let b = t
            .split_b(spec, &fused_b, RANK)
            .into_iter()
            .zip(&t.out_parts)
            .map(|(v, &rows)| Tensor::from_f32(&v, &[rows, RANK], &bud).unwrap())
            .collect();
        *lora.slot_mut(t.layer, t.kind).unwrap() = Some(LoraSite { a, b });
    }
    lora
}

pub fn zero_b(t: &str, targets: &[LoraTarget]) -> Vec<f32> {
    let t = targets.iter().find(|x| x.module() == t).unwrap();
    vec![0.0; t.out_parts.iter().sum::<usize>() * RANK]
}

pub fn fixture_sites(file: &SafeTensors<'_>, nonzero_b: bool) -> Qwen35Lora<LoraSite<Tensor>> {
    let (s, t) = (spec(), targets());
    sites_from(
        &s,
        &t,
        |m| read(file, &format!("init.{m}.lora_A.weight")),
        |m| {
            if nonzero_b {
                read(file, &format!("bnz.{m}.lora_B.weight"))
            } else {
                zero_b(m, &t)
            }
        },
    )
}

pub struct Data {
    pub ids: Tensor,
    pub targets: Tensor,
    pub tables: Qwen35Tables,
}

pub fn data() -> Data {
    let reference = SafeTensors::open(&dir().join("reference.safetensors")).unwrap();
    let (ids, tg) = ids_and_targets(&reference);
    let tables = Qwen35Tables::new(&spec(), ids.shape()[1], &budget()).unwrap();
    Data {
        ids,
        targets: tg,
        tables,
    }
}

pub struct LoraRun {
    pub loss: f32,
    /// `(module, A grad, fused B grad)` per target, in target order.
    pub grads: Vec<(String, Vec<f32>, Vec<f32>)>,
}

pub struct Opts {
    pub config: LoraConfig,
    pub dropout_seed: u64,
    pub activations: ActivationCheckpoint,
    /// Run the whole step inside an `AutocastMode::Bf16` region (the
    /// backend must be an `Autocast`).
    pub bf16: bool,
}

impl Opts {
    pub fn new() -> Self {
        Self {
            config: config(),
            dropout_seed: 0,
            activations: ActivationCheckpoint::Off,
            bf16: false,
        }
    }
}

pub fn to_vec(t: &Tensor) -> Vec<f32> {
    t.to_host(&budget()).unwrap().to_f32_vec().unwrap()
}

/// One forward and backward with the adapters `sites` over the frozen base.
pub fn lora_step<B: Backend>(
    backend: B,
    host: &Qwen35Params<Tensor>,
    sites: &Qwen35Lora<LoraSite<Tensor>>,
    d: &Data,
    o: &Opts,
) -> LoraRun {
    let s = spec();
    let bud = budget();
    let mut tape = Tape::new(backend);
    let _region = o.bf16.then(|| {
        tape.backend()
            .autocast_region(ojas_core::AutocastMode::Bf16)
            .unwrap()
    });
    let params = bind_frozen(&mut tape, host).unwrap();
    let bound = sites
        .try_map(&mut |site| site.try_map(&mut |t| tape.leaf(t.clone())))
        .unwrap();
    let mut rng = CounterRng::new(o.dropout_seed);
    let [b, t] = [d.ids.shape()[0], d.ids.shape()[1]];
    let fwd = LoraForward::new(&mut tape, bound, &o.config, b, t, &mut rng, &bud).unwrap();
    let loss = forward_loss_lora(
        &mut tape,
        &s,
        &params,
        Some(&fwd),
        &d.ids,
        &d.targets,
        &d.tables,
        Some(IGNORE),
        DEFAULT_CE_CHUNK,
        o.activations,
    )
    .unwrap();
    tape.backward(loss).unwrap();
    let loss = to_vec(tape.value(loss).unwrap())[0];
    // Frozen leaves have no gradient: absent, not zero.
    for v in params.values() {
        assert!(tape.grad(*v).is_none(), "a frozen leaf has a gradient");
    }
    let tg = targets();
    let grads = tg
        .iter()
        .zip(fwd.sites().sites())
        .map(|(t, site)| {
            let a = to_vec(tape.grad(site.a).expect("A has a gradient"));
            let parts: Vec<Vec<f32>> = site
                .b
                .iter()
                .map(|v| to_vec(tape.grad(*v).expect("B has a gradient")))
                .collect();
            (t.module().to_string(), a, t.fuse_b(&s, &parts, RANK))
        })
        .collect();
    LoraRun { loss, grads }
}

pub fn host_params() -> Qwen35Params<Tensor> {
    let model = base_file();
    load_hf(&spec(), &model, PREFIX, ojas_core::DType::F32, &budget()).unwrap()
}

pub fn rel(got: f32, want: f32) -> f64 {
    f64::from((got - want).abs()) / f64::from(want.abs())
}

pub fn peak_ratio(got: &[f32], want: &[f32], what: &str) -> f32 {
    assert_eq!(got.len(), want.len(), "{what}");
    let peak = want.iter().fold(0.0f32, |p, x| p.max(x.abs()));
    assert!(peak > 0.0, "{what}: an all-zero reference checks nothing");
    got.iter()
        .zip(want)
        .map(|(a, b)| {
            assert!(a.is_finite(), "{what}: non-finite");
            (a - b).abs()
        })
        .fold(0.0f32, f32::max)
        / peak
}

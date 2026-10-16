//! Shared LoRA helpers for `qwen35_lora.rs`, `qwen35_lora_scale1.rs` (CPU),
//! `qwen35_lora_adversarial.rs` and `qwen35_lora_metal.rs`: the torch LoRA
//! fixtures ([`SCALING_2`], [`SCALING_1`]), the host adapters built from
//! them, one forward and backward over a frozen base, the (b') AdamW
//! trajectory, and the (a'), (b'), (c') clauses each fixture is held to.
#![allow(dead_code)]

use std::path::PathBuf;

use ojas_autograd::Tape;
use ojas_core::{AdamWConfig, Backend, Tensor};
use ojas_data::CounterRng;
use ojas_io::SafeTensors;
use ojas_model::lora::{LoraConfig, LoraSite};
use ojas_model::qwen35::{
    bind, bind_frozen, forward_loss, forward_loss_lora, load_hf, lora_targets, LoraForward,
    LoraTarget, MixerLora, Qwen35Lora, Qwen35Params, Qwen35Spec, Qwen35Tables, LORA_KINDS,
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

/// A torch LoRA fixture written by `ojas-oracle/python/qwen35_lora_fixture.py`:
/// its directory beside `qwen35_tiny` and the alpha it was written at. Rank
/// [`RANK`]; the seed, data and optimizer are shared, so two fixtures hold
/// the same `A`, `B` and batch and differ only by the scale.
#[derive(Clone, Copy, Debug)]
pub struct Fixture {
    pub dir: &'static str,
    pub alpha: f32,
}

/// Rank 4, alpha 8: scaling 2, so a dropped scaling cannot pass.
pub const SCALING_2: Fixture = Fixture {
    dir: "qwen35_tiny_lora",
    alpha: ALPHA,
};

/// Rank 4, alpha 4: scaling 1, the pre-registered run's alpha == r, where
/// `scaled_b` returns `B` itself and `LoraForward` records no scale constant.
pub const SCALING_1: Fixture = Fixture {
    dir: "qwen35_tiny_lora_s1",
    alpha: 4.0,
};

impl Fixture {
    pub fn file(self) -> SafeTensors<'static> {
        let path: PathBuf = dir()
            .parent()
            .unwrap()
            .join(self.dir)
            .join("lora.safetensors");
        SafeTensors::open(&path).unwrap()
    }

    pub fn config(self) -> LoraConfig {
        LoraConfig::new(RANK, self.alpha, 0.0).unwrap()
    }

    /// [`Opts::new`] at this fixture's scaling.
    pub fn opts(self) -> Opts {
        Opts {
            config: self.config(),
            ..Opts::new()
        }
    }
}

pub fn lora_file() -> SafeTensors<'static> {
    SCALING_2.file()
}

pub fn config() -> LoraConfig {
    SCALING_2.config()
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
    load_hf(
        &spec(),
        &model,
        PREFIX,
        ojas_core::DType::F32,
        ojas_core::DType::F32,
        &budget(),
    )
    .unwrap()
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

// ---- (b'): the AdamW trajectory --------------------------------------------

/// The fixture's optimizer: AdamW lr 5e-3, betas (0.9, 0.999), eps 1e-8,
/// weight decay 0.01 on every adapter.
pub fn adamw() -> AdamWConfig {
    AdamWConfig {
        lr: LR,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: WD,
    }
}

pub struct Trajectory {
    pub losses: Vec<f32>,
    /// `(module, A, fused B)` after the last update.
    pub finals: Vec<(String, Vec<f32>, Vec<f32>)>,
}

/// (b') `STEPS` AdamW steps from `fx`'s init (A random, B = 0) on the same
/// batch every step, under `lora` and `cfg` (the fixture's own, or a
/// deliberately wrong one for the sensitivity tests).
pub fn trajectory(fx: Fixture, lora: &LoraConfig, cfg: AdamWConfig) -> Trajectory {
    let (s, tg, d, host) = (spec(), targets(), data(), host_params());
    let backend = cpu();
    let mut sites = fixture_sites(&fx.file(), false);
    // Moments follow the leaves: A, then each B part, per site.
    let zero = |t: &Tensor| {
        Tensor::from_f32(&vec![0.0; t.num_elements().unwrap()], t.shape(), &budget()).unwrap()
    };
    let mut moments: Vec<(Tensor, Tensor)> = sites
        .sites()
        .iter()
        .flat_map(|site| std::iter::once(&site.a).chain(&site.b))
        .map(|t| (zero(t), zero(t)))
        .collect();
    let mut losses = Vec::new();
    for step in 0..STEPS {
        let grads: Vec<Tensor> = {
            let mut tape = Tape::new(backend.clone());
            let params = bind_frozen(&mut tape, &host).unwrap();
            let bound = sites
                .try_map(&mut |site| site.try_map(&mut |t| tape.leaf(t.clone())))
                .unwrap();
            let mut rng = CounterRng::new(0);
            let [b, t] = [d.ids.shape()[0], d.ids.shape()[1]];
            let fwd = LoraForward::new(&mut tape, bound, lora, b, t, &mut rng, &budget()).unwrap();
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
                ActivationCheckpoint::Off,
            )
            .unwrap();
            tape.backward(loss).unwrap();
            losses.push(to_vec(tape.value(loss).unwrap())[0]);
            fwd.sites()
                .sites()
                .iter()
                .flat_map(|site| std::iter::once(&site.a).chain(&site.b))
                .map(|v| tape.take_grad(*v).expect("an adapter leaf has a gradient"))
                .collect()
        };
        // The tape is gone, so each parameter is uniquely owned again.
        let mut grads = grads.into_iter();
        let mut moment = moments.iter_mut();
        let mut update = |p: &mut Tensor| {
            let g = grads.next().unwrap();
            let (m1, m2) = moment.next().unwrap();
            backend.adamw_step(p, &g, m1, m2, step as u64, cfg).unwrap();
        };
        for l in &mut sites.layers {
            let mixer: Vec<&mut Option<LoraSite<Tensor>>> = match &mut l.mixer {
                MixerLora::GatedDeltaNet { qkv, z, out } => vec![qkv, z, out],
                MixerLora::Attention { q, k, v, o } => vec![q, k, v, o],
            };
            for site in mixer
                .into_iter()
                .chain([&mut l.gate, &mut l.up, &mut l.down])
                .flatten()
            {
                update(&mut site.a);
                for b in &mut site.b {
                    update(b);
                }
            }
        }
    }
    let finals = tg
        .iter()
        .zip(sites.sites())
        .map(|(t, site)| {
            let parts: Vec<Vec<f32>> = site.b.iter().map(to_vec).collect();
            (
                t.module().to_string(),
                to_vec(&site.a),
                t.fuse_b(&s, &parts, RANK),
            )
        })
        .collect();
    Trajectory { losses, finals }
}

/// The card's bound on a final adapter: within 1e-5 of the tensor's peak.
pub const FINAL_BOUND: f32 = 1e-5;

/// Per final adapter tensor: `(name, ojas vs torch fp32, torch fp32 vs fp64
/// (the oracle's own noise floor), ojas vs fp64)`, each of the reference's
/// peak.
pub fn final_ratios(t: &Trajectory, file: &SafeTensors<'_>) -> Vec<(String, f32, f32, f32)> {
    t.finals
        .iter()
        .flat_map(|(module, a, b)| {
            [("A", a), ("B", b)].into_iter().map(move |(w, v)| {
                let name = format!("{module}.lora_{w}.weight");
                let torch = read(file, &format!("b.final.{name}"));
                let exact = read(file, &format!("b.final64.{name}"));
                (
                    name.clone(),
                    peak_ratio(v, &torch, &name),
                    peak_ratio(&torch, &exact, &name),
                    peak_ratio(v, &exact, &name),
                )
            })
        })
        .collect()
}

/// The ratified (b') final-adapter bound on one tensor: the larger of the
/// card's 1e-5 and three times the oracle's own noise floor on it (two
/// independent fp32 runs each sit about one floor from the exact answer).
pub fn final_bound(floor: f32) -> f32 {
    FINAL_BOUND.max(3.0 * floor)
}

// ---- the (a'), (b'), (c') clauses, per fixture ------------------------------

/// (a') loss and every adapter gradient against plain torch; no gradient
/// for any frozen leaf (asserted inside `lora_step`).
pub fn check_a_prime(fx: Fixture) {
    let file = fx.file();
    let run = lora_step(
        cpu(),
        &host_params(),
        &fixture_sites(&file, true),
        &data(),
        &fx.opts(),
    );
    let want = read(&file, "a.loss")[0];
    let r = rel(run.loss, want);
    eprintln!(
        "{}: a' loss {} vs torch {want} (rel {r:.2e})",
        fx.dir, run.loss
    );
    assert!(r <= LORA_LOSS_BOUND, "loss rel {r:.3e}");
    let mut worst = 0.0f32;
    for (module, a, b) in &run.grads {
        let ra = peak_ratio(
            a,
            &read(&file, &format!("a.grad.{module}.lora_A.weight")),
            module,
        );
        let rb = peak_ratio(
            b,
            &read(&file, &format!("a.grad.{module}.lora_B.weight")),
            module,
        );
        eprintln!("{module}: dA {ra:.2e} dB {rb:.2e} of peak");
        assert!(
            ra <= LORA_GRAD_BOUND && rb <= LORA_GRAD_BOUND,
            "{module}: dA {ra:.3e} dB {rb:.3e}"
        );
        worst = worst.max(ra).max(rb);
    }
    eprintln!("worst adapter gradient: {worst:.3e} of its peak");
}

/// (c') at B = 0 the LoRA loss is the base loss bit for bit; dB is non-zero
/// and dA exactly zero there.
pub fn check_c_prime(fx: Fixture) {
    let file = fx.file();
    let d = data();
    let host = host_params();
    let run = lora_step(cpu(), &host, &fixture_sites(&file, false), &d, &fx.opts());
    let mut tape = Tape::new(cpu());
    let vars = bind(&mut tape, &host).unwrap();
    let loss = forward_loss(
        &mut tape,
        &spec(),
        &vars,
        &d.ids,
        &d.targets,
        &d.tables,
        Some(IGNORE),
        DEFAULT_CE_CHUNK,
        ActivationCheckpoint::Off,
    )
    .unwrap();
    let plain = to_vec(tape.value(loss).unwrap())[0];
    assert_eq!(
        run.loss.to_bits(),
        plain.to_bits(),
        "{} vs {plain}",
        run.loss
    );
    let torch = read(&file, "c.loss")[0];
    assert!(rel(run.loss, torch) <= LORA_LOSS_BOUND);
    // The adapter still learns from the identity: dB is non-zero at B = 0.
    assert!(run
        .grads
        .iter()
        .any(|(_, _, b)| b.iter().any(|&x| x != 0.0)));
    // And dA is exactly zero there (dA flows through B = 0).
    assert!(run
        .grads
        .iter()
        .all(|(_, a, _)| a.iter().all(|&x| x == 0.0)));
}

/// (b') the loss clauses (steps 0-5 within 1e-5, to 20 within 1e-4), and the
/// run is bit-repeatable.
pub fn check_b_prime_losses(fx: Fixture) {
    let file = fx.file();
    let first = trajectory(fx, &fx.config(), adamw());
    let want = read(&file, "b.losses");
    assert_eq!(first.losses.len(), STEPS);
    for (i, (&got, &w)) in first.losses.iter().zip(&want).enumerate() {
        let r = rel(got, w);
        eprintln!("step {i}: {got} vs torch {w} (rel {r:.2e})");
        let bound = if i <= 5 { 1e-5 } else { 1e-4 };
        assert!(r <= bound, "step {i}: loss rel {r:.3e} > {bound:e}");
    }
    let again = trajectory(fx, &fx.config(), adamw());
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&first.losses), bits(&again.losses));
    for (x, y) in first.finals.iter().zip(&again.finals) {
        assert_eq!(
            (bits(&x.1), bits(&x.2)),
            (bits(&y.1), bits(&y.2)),
            "{}",
            x.0
        );
    }
}

/// (b') the final adapters, under the ratified bound (gp-ft-frozen-params-
/// and-lora criterion 6, 2026-10-15): the card asks for 1e-5 of each
/// tensor's peak against torch fp32, but torch fp32 is itself further than
/// that from the exact (fp64) trajectory on some tensors (AdamW's step is
/// scale-free, so a gradient error of 1e-6 of a tensor's peak becomes a
/// parameter error of several 1e-5 within 20 steps). Each tensor is held to
/// [`final_bound`] of its floor against torch and against fp64, and ojas
/// must be closer to the exact trajectory than torch fp32 is on most
/// tensors.
pub fn check_b_prime_finals(fx: Fixture) {
    let file = fx.file();
    let rows = final_ratios(&trajectory(fx, &fx.config(), adamw()), &file);
    let mut worst_vs_torch: f32 = 0.0;
    let mut closer = 0;
    for (name, vs_torch, floor, vs_exact) in &rows {
        let bound = final_bound(*floor);
        eprintln!(
            "{name}: ojas vs torch {vs_torch:.2e}, torch vs fp64 {floor:.2e}, ojas vs fp64 {vs_exact:.2e}"
        );
        assert!(
            *vs_torch <= bound,
            "{name}: ojas vs torch {vs_torch:.3e} > {bound:.3e}"
        );
        assert!(
            *vs_exact <= bound,
            "{name}: ojas vs fp64 {vs_exact:.3e} > {bound:.3e}"
        );
        closer += usize::from(vs_exact <= floor);
        worst_vs_torch = worst_vs_torch.max(*vs_torch);
    }
    eprintln!(
        "{}: worst ojas vs torch final adapter: {worst_vs_torch:.3e} of peak",
        fx.dir
    );
    // ojas is no noisier than torch fp32 itself: closer to the exact
    // trajectory than torch is on most tensors.
    assert!(
        closer * 2 > rows.len(),
        "ojas is closer to fp64 than torch fp32 on only {closer} of {} tensors",
        rows.len()
    );
}

/// A plain LoRA linear: y, dX, dA, dB against torch, and the frozen weight
/// has no gradient. The scale constant is built only at a scaling other than
/// 1, as `LoraForward` does.
pub fn check_lora_linear(fx: Fixture) {
    use ojas_model::lora::{lora_linear, scale_constant, scaled_b};
    use ojas_model::Graph;
    let file = fx.file();
    let bud = budget();
    let (rows, inn, out) = (12, 40, 24);
    let t =
        |name: &str, shape: &[usize]| Tensor::from_f32(&read(&file, name), shape, &bud).unwrap();
    let scale = fx.config().scaling();
    let mut tape = Tape::new(cpu());
    let x = tape.leaf(t("lin.x", &[rows, inn])).unwrap();
    let w = tape.frozen_leaf(t("lin.w", &[out, inn])).unwrap();
    let a = tape.leaf(t("lin.a", &[RANK, inn])).unwrap();
    let b = tape.leaf(t("lin.b", &[out, RANK])).unwrap();
    let c = (scale != 1.0).then(|| scale_constant(&mut tape, &[out, RANK], scale, &bud).unwrap());
    let b_eff = scaled_b(&mut tape, &b, scale, c.as_ref()).unwrap();
    let y = lora_linear(&mut tape, &x, &w, &a, &b_eff, None).unwrap();
    let gy = tape.frozen(&t("lin.gy", &[rows, out])).unwrap();
    let weighted = Graph::mul(&mut tape, &y, &gy).unwrap();
    tape.backward(weighted).unwrap();
    let yv = to_vec(tape.value(y).unwrap());
    assert!(peak_ratio(&yv, &read(&file, "lin.y"), "y") <= LORA_GRAD_BOUND);
    for (var, name) in [(x, "lin.dx"), (a, "lin.da"), (b, "lin.db")] {
        let got = to_vec(tape.grad(var).unwrap());
        let r = peak_ratio(&got, &read(&file, name), name);
        eprintln!("{name}: {r:.2e} of peak");
        assert!(r <= LORA_GRAD_BOUND, "{name}: {r:.3e}");
    }
    assert!(tape.grad(w).is_none() && tape.grad(gy).is_none());
}

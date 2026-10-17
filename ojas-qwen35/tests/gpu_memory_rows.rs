//! Peak device bytes of one Qwen3.5 training step on Metal, one arm per
//! process, for `bench/results/`. `#[ignore]` and named `gpu_*`: run it alone,
//! under the machine's heavy-job lock, one process per arm so each peak is
//! that arm's alone (the device figures are process-wide).
//!
//! ```text
//! QWEN35_MEM_ARM=tape-bf16 QWEN35_MEM_TOKENS=2048 \
//! cargo test -p ojas-qwen35 --release --test gpu_memory_rows \
//!     -- --ignored --test-threads=1 --nocapture gpu_memory_row
//! ```
//!
//! Arms (`QWEN35_MEM_ARM`):
//! - `provider-f32`: this crate's provider at `Precision::F32`,
//!   `Numerics::ExactF32`; `provider-bf16`: `Precision::Bf16`,
//!   `Numerics::Bf16Operands`. One forward, backward and AdamW step.
//! - `tape-f32` / `tape-bf16`: ojas's Tape on `MetalBackend`, linear weights
//!   loaded `F32` or `Bf16` (`load_hf`) and the tied table `F32`, every layer
//!   a checkpointed segment, one forward and backward. `tape-bf16-table`:
//!   `tape-bf16` with the tied table `Bf16` too, read in place by the gather
//!   and the fused head. Every parameter takes a gradient (no leaf is
//!   frozen): this is a full-gradient step, not the LoRA step, so its peak
//!   over-counts a LoRA step's by the weight gradients.
//!
//! - `tape-lora`: the LoRA step of the fine-tune track, composed from the
//!   merged primitives: the base bound frozen (`bind_frozen`, bf16 linears
//!   and a bf16 table), rank 16 / alpha 16 adapters (f32, `A` random and `B`
//!   zero) on the ten `LoraKind` linears of every layer, `forward_loss_lora`
//!   (the fused tied head over every position), `backward`, and one
//!   `Backend::adamw_step` per adapter tensor with both moments resident.
//!   `QWEN35_MEM_CKPT` is `on` (every layer a checkpointed segment, the
//!   default) or `off`. One warmup step, then three timed steps: the row
//!   carries the first step's wall, every step's wall, tokens/s as the mean
//!   of the three, and the device peak since the weights were resident. The
//!   device is opened under a 28 GiB budget, below memguard's 32 GiB, so a
//!   misfit is a typed refusal row. Host `phys_footprint` is recorded beside
//!   the device figures. `gpu_lora_step_trains_only_the_adapters` is the
//!   tiny-fixture check that this step moves the adapters and nothing else.
//!
//! Every position is scored (the causal loss as letter rows), micro-batch 1.
//! The snapshot is `QWEN35_MEM_SNAPSHOT`, or the one snapshot of
//! `Qwen/Qwen3.5-2B-Base` in the Hugging Face cache. The row is one line
//! starting `ROW ` (JSON). A refusal at any phase is a row too, naming the
//! phase and the error, and the test then fails.

#![cfg(target_os = "macos")]

mod common;

use std::time::Instant;

use ojas_core::{AdamWConfig, OjasError};
use ojas_model::lora::{Entropy, LoraConfig, LoraSite};
use ojas_model::qwen35::{
    bind_frozen, forward_loss_lora, lora_targets, LoraForward, MixerLora, Qwen35Lora, Qwen35Params,
    Qwen35Spec, LORA_KINDS,
};

use ojas_autograd::Tape;
use ojas_core::{Backend, Budget, DType, Tensor};
use ojas_io::SafeTensors;
use ojas_metal::MetalBackend;
use ojas_model::qwen35::{bind, forward_loss, load_hf, Qwen35Tables};
use ojas_model::{ActivationCheckpoint, DEFAULT_CE_CHUNK};
use ojas_qwen35::{
    AdamWHyper, GroupSpec, LrRule, Numerics, OptimizerPlan, Precision, Qwen35Step, Select,
    Sequence, Snapshot, WdRule,
};

use common::{footprint_or_refuse, refused, single_weights_file, snapshot_dir, Row};

const GIB: u64 = 1 << 30;

/// `n` deterministic ids below 150000 (ordinary text tokens), skipping the
/// vision special tokens the config declares.
fn ids(n: usize, reserved: &[u32]) -> Vec<u32> {
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let id = (s % 150_000) as u32;
        if !reserved.contains(&id) {
            out.push(id);
        }
    }
    out
}

#[test]
#[ignore]
fn gpu_memory_row() {
    let arm = std::env::var("QWEN35_MEM_ARM").expect("QWEN35_MEM_ARM");
    let tokens: usize = std::env::var("QWEN35_MEM_TOKENS")
        .expect("QWEN35_MEM_TOKENS")
        .parse()
        .expect("QWEN35_MEM_TOKENS");
    let dir = snapshot_dir("QWEN35_MEM_SNAPSHOT");
    let snap = Snapshot::from_dir(&dir).unwrap();
    let cfg = snap.config.clone();
    let reserved: Vec<u32> = cfg.reserved_token_ids.iter().map(|(_, id)| *id).collect();
    let ids = ids(tokens, &reserved);
    let mut row = Row::new();
    row.text("arm", &arm);
    row.text("snapshot", &dir.display().to_string());
    row.num("tokens", tokens as u64);
    let started = Instant::now();
    let mut row = match arm.as_str() {
        "provider-f32" => provider(&snap, &ids, Precision::F32, Numerics::ExactF32, row),
        "provider-bf16" => provider(&snap, &ids, Precision::Bf16, Numerics::Bf16Operands, row),
        "tape-f32" => tape(&snap, &ids, DType::F32, DType::F32, row),
        "tape-bf16" => tape(&snap, &ids, DType::Bf16, DType::F32, row),
        "tape-bf16-table" => tape(&snap, &ids, DType::Bf16, DType::Bf16, row),
        "tape-lora" => tape_lora(&snap, &ids, row),
        other => panic!("unknown QWEN35_MEM_ARM {other}"),
    };
    row.num("wall_ms", started.elapsed().as_millis() as u64);
    row.text("status", "ok");
    row.print();
}

fn provider(snap: &Snapshot, ids: &[u32], p: Precision, n: Numerics, mut r: Row) -> Row {
    let mut s = match Qwen35Step::open(snap, n, p, Budget::new(16 * GIB)) {
        Ok(s) => s,
        Err(e) => refused(r, "open", e),
    };
    let open = s.device_bytes();
    r.num("resident_bytes", open.allocated);
    r.num("open_peak_bytes", open.peak);
    r.phase("open");
    s.reset_peak_device_bytes();
    let rows: Vec<u32> = (0..ids.len() as u32 - 1).collect();
    let k = rows.len();
    let pending = match s.forward(&Sequence {
        ids,
        letter_rows: &rows,
        letter_targets: &ids[1..],
        letter_scale: 1.0 / k as f32,
        span_positions: &[],
    }) {
        Ok(p) => p,
        Err(e) => refused(r, "forward", e),
    };
    let loss = pending.letter_ce_sum() / k as f64;
    let after_forward = s.device_bytes();
    r.num("after_forward_bytes", after_forward.allocated);
    r.num("forward_peak_bytes", after_forward.peak);
    r.phase("forward");
    if let Err(e) = s.backward(pending, None) {
        refused(r, "backward", e);
    }
    let plan = OptimizerPlan::build(
        s.parameter_table(),
        &GroupSpec {
            lr: vec![LrRule {
                label: "all".into(),
                select: Select::All,
                lr_scale: 1.0,
            }],
            weight_decay: vec![WdRule {
                label: "all".into(),
                select: Select::All,
                weight_decay: 0.0,
            }],
        },
    )
    .unwrap();
    let hyper = AdamWHyper {
        lr: 1e-5,
        beta1: 0.9,
        beta2: 0.95,
        eps: 1e-8,
        grad_scale: 1.0,
    };
    if let Err(e) = s.adamw_step(&hyper, &plan) {
        refused(r, "adamw_step", e);
    }
    let end = s.device_bytes();
    r.num("step_peak_bytes", end.peak);
    r.text("loss", &format!("{loss:.6}"));
    r
}

fn tape(snap: &Snapshot, ids: &[u32], linears: DType, table: DType, mut r: Row) -> Row {
    let cfg = snap.config.clone();
    let spec = cfg.tape_spec();
    let n = ids.len() - 1;
    let m = match MetalBackend::new(Budget::new(56 * GIB)) {
        Ok(m) => m,
        Err(e) => refused(r, "MetalBackend::new", e),
    };
    let host_budget = Budget::new(24 * GIB);
    let device = {
        let file = SafeTensors::open(&single_weights_file(&snap.weights_path)).unwrap();
        let host = match load_hf(
            &spec,
            &file,
            &cfg.tower_prefix,
            linears,
            table,
            &host_budget,
        ) {
            Ok(h) => h,
            Err(e) => refused(r, "load_hf", e),
        };
        let bytes = |want: DType| -> u64 {
            host.values()
                .iter()
                .filter(|t| t.dtype() == want)
                .map(|t| (t.num_elements().unwrap() * want.size()) as u64)
                .sum()
        };
        r.num("bf16_param_bytes", bytes(DType::Bf16));
        r.num("f32_param_bytes", bytes(DType::F32));
        r.num(
            "tied_embed_f32_bytes",
            (host.embed.num_elements().unwrap() * 4) as u64,
        );
        r.num(
            "tied_embed_bytes",
            (host.embed.num_elements().unwrap() * host.embed.dtype().size()) as u64,
        );
        match host.try_map(&mut |t| m.upload(t)) {
            Ok(d) => d,
            Err(e) => refused(r, "upload", e),
        }
    };
    if let Err(e) = m.sync() {
        refused(r, "sync weights", e);
    }
    let resident = m.memory_and_reset_peak().unwrap();
    r.num("resident_bytes", resident.allocated);
    r.num("budget_resident_bytes", m.budget().live_bytes().unwrap());
    let tables = Qwen35Tables::new(&spec, n, &host_budget)
        .unwrap()
        .upload(&m)
        .unwrap();
    let ids_t = Tensor::from_u32(&ids[..n], &[1, n], &host_budget).unwrap();
    let targets = Tensor::from_u32(&ids[1..], &[1, n], &host_budget).unwrap();
    let mut tape = Tape::new(m.clone());
    let vars = bind(&mut tape, &device).unwrap();
    drop(device);
    let loss = match forward_loss(
        &mut tape,
        &spec,
        &vars,
        &ids_t,
        &targets,
        &tables,
        None,
        DEFAULT_CE_CHUNK,
        ActivationCheckpoint::Blocks,
    ) {
        Ok(l) => l,
        Err(e) => refused(r, "forward", e),
    };
    if let Err(e) = m.sync() {
        refused(r, "sync forward", e);
    }
    let after_forward = m.memory().unwrap();
    if let Err(e) = tape.backward(loss) {
        refused(r, "backward", e);
    }
    if let Err(e) = m.sync() {
        refused(r, "sync backward", e);
    }
    let end = m.memory().unwrap();
    let got = tape
        .value(loss)
        .unwrap()
        .to_host(&host_budget)
        .unwrap()
        .to_f32_vec()
        .unwrap()[0];
    r.num("after_forward_bytes", after_forward.allocated);
    r.num("step_peak_bytes", end.peak_allocated);
    r.num("working_set_bytes", resident.recommended_working_set);
    r.text("loss", &format!("{got:.6}"));
    drop(tape);
    r
}

const LORA_RANK: usize = 16;
const LORA_ALPHA: f32 = 16.0;
/// The adapters' seed and the learning rate of the measured steps. The rate
/// does not change the step's cost; it is the pre-registered order of
/// magnitude.
const LORA_SEED: u64 = 0x10_4a;
const LORA_LR: f64 = 1e-4;
/// Steps after the warmup step that the tokens/s mean covers.
const TIMED_STEPS: usize = 3;
/// Below memguard's 32 GiB `phys_footprint` kill, so a misfit is a typed
/// refusal, not a SIGTERM.
const LORA_DEVICE_BUDGET: u64 = 28 * GIB;

/// Where a LoRA step stopped and why: a typed refusal row, not a panic with
/// no row.
#[derive(Debug)]
struct StepFault {
    phase: &'static str,
    error: String,
}

fn at(phase: &'static str) -> impl FnOnce(OjasError) -> StepFault {
    move |e| StepFault {
        phase,
        error: format!("{e:?}"),
    }
}

/// Dropout is 0, so the stream is never read; the type satisfies
/// `LoraForward::new`.
struct NoDropout;

impl Entropy for NoDropout {
    fn next_u64(&mut self) -> u64 {
        0
    }
}

/// Every adapter tensor in the order of `Qwen35Lora::sites`: `A`, then each
/// `B` part, per site.
fn adapter_tensors(lora: &mut Qwen35Lora<LoraSite<Tensor>>) -> Vec<&mut Tensor> {
    let mut out = Vec::new();
    for l in &mut lora.layers {
        let mixer: Vec<&mut Option<LoraSite<Tensor>>> = match &mut l.mixer {
            MixerLora::GatedDeltaNet { qkv, z, out } => vec![qkv, z, out],
            MixerLora::Attention { q, k, v, o } => vec![q, k, v, o],
        };
        for site in mixer
            .into_iter()
            .chain([&mut l.gate, &mut l.up, &mut l.down])
            .flatten()
        {
            out.push(&mut site.a);
            out.extend(site.b.iter_mut());
        }
    }
    out
}

/// One LoRA training step over a frozen, device-resident base: the loop the
/// fine-tune run composes from the merged primitives.
struct LoraTrainer {
    backend: MetalBackend,
    spec: Qwen35Spec,
    base: Qwen35Params<Tensor>,
    tables: Qwen35Tables,
    adapters: Qwen35Lora<LoraSite<Tensor>>,
    moments: Vec<(Tensor, Tensor)>,
    config: LoraConfig,
    adamw: AdamWConfig,
    checkpoint: ActivationCheckpoint,
    steps: u64,
}

impl LoraTrainer {
    /// `base` and `adapters` resident on `backend`; every adapter tensor
    /// gets zeroed first and second moments.
    #[allow(clippy::too_many_arguments)]
    fn new(
        backend: MetalBackend,
        spec: Qwen35Spec,
        base: Qwen35Params<Tensor>,
        tables: Qwen35Tables,
        mut adapters: Qwen35Lora<LoraSite<Tensor>>,
        config: LoraConfig,
        checkpoint: ActivationCheckpoint,
        host_budget: &Budget,
    ) -> Result<Self, StepFault> {
        let zeros = |t: &Tensor| -> Result<Tensor, StepFault> {
            let n = t.num_elements().map_err(at("moments"))?;
            let host =
                Tensor::from_f32(&vec![0.0; n], t.shape(), host_budget).map_err(at("moments"))?;
            backend.upload(&host).map_err(at("moments"))
        };
        let moments = adapter_tensors(&mut adapters)
            .into_iter()
            .map(|t| Ok((zeros(t)?, zeros(t)?)))
            .collect::<Result<Vec<_>, StepFault>>()?;
        Ok(Self {
            backend,
            spec,
            base,
            tables,
            adapters,
            moments,
            config,
            adamw: AdamWConfig {
                lr: LORA_LR,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.0,
            },
            checkpoint,
            steps: 0,
        })
    }

    /// Forward, backward and one AdamW update of every adapter tensor on
    /// `ids` against `targets`; the loss of the step, read after the update
    /// has finished on the device. A frozen base parameter that holds a
    /// gradient is a fault, not a slow step.
    fn step(
        &mut self,
        ids: &Tensor,
        targets: &Tensor,
        host_budget: &Budget,
    ) -> Result<f32, StepFault> {
        let grads: Vec<Tensor>;
        let loss;
        {
            let mut tape = Tape::new(self.backend.clone());
            let params = bind_frozen(&mut tape, &self.base).map_err(at("bind"))?;
            let bound = self
                .adapters
                .try_map(&mut |site| site.try_map(&mut |t| tape.leaf(t.clone())))
                .map_err(at("bind adapters"))?;
            let [b, t] = [ids.shape()[0], ids.shape()[1]];
            let fwd = LoraForward::new(
                &mut tape,
                bound,
                &self.config,
                b,
                t,
                &mut NoDropout,
                host_budget,
            )
            .map_err(at("lora forward"))?;
            let l = forward_loss_lora(
                &mut tape,
                &self.spec,
                &params,
                Some(&fwd),
                ids,
                targets,
                &self.tables,
                None,
                DEFAULT_CE_CHUNK,
                self.checkpoint,
            )
            .map_err(at("forward"))?;
            tape.backward(l).map_err(at("backward"))?;
            self.backend.sync().map_err(at("sync backward"))?;
            if let Some(i) = params
                .values()
                .iter()
                .position(|v| tape.grad(**v).is_some())
            {
                return Err(StepFault {
                    phase: "frozen gradient",
                    error: format!("base parameter {i} holds a gradient"),
                });
            }
            loss = tape
                .value(l)
                .and_then(|v| v.to_host(host_budget))
                .and_then(|v| v.to_f32_vec())
                .map_err(at("loss"))?[0];
            grads = fwd
                .sites()
                .sites()
                .iter()
                .flat_map(|site| std::iter::once(&site.a).chain(&site.b))
                .map(|v| {
                    tape.take_grad(*v).ok_or_else(|| StepFault {
                        phase: "adapter gradient",
                        error: "an adapter leaf has no gradient".to_string(),
                    })
                })
                .collect::<Result<_, _>>()?;
        }
        // The tape is gone, so each adapter is uniquely owned again.
        let params = adapter_tensors(&mut self.adapters);
        if params.len() != grads.len() || params.len() != self.moments.len() {
            return Err(StepFault {
                phase: "adamw",
                error: format!(
                    "{} adapter tensors, {} gradients, {} moment pairs",
                    params.len(),
                    grads.len(),
                    self.moments.len()
                ),
            });
        }
        for ((p, g), (m1, m2)) in params.into_iter().zip(&grads).zip(&mut self.moments) {
            self.backend
                .adamw_step(p, g, m1, m2, self.steps, self.adamw)
                .map_err(at("adamw_step"))?;
        }
        self.backend.sync().map_err(at("sync adamw"))?;
        self.steps += 1;
        Ok(loss)
    }
}

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn tape_lora(snap: &Snapshot, ids: &[u32], mut r: Row) -> Row {
    let cfg = snap.config.clone();
    let spec = cfg.tape_spec();
    let n = ids.len() - 1;
    let checkpoint = match std::env::var("QWEN35_MEM_CKPT").as_deref() {
        Ok("on") | Err(std::env::VarError::NotPresent) => ActivationCheckpoint::Blocks,
        Ok("off") => ActivationCheckpoint::Off,
        other => panic!("QWEN35_MEM_CKPT is on or off, got {other:?}"),
    };
    r.text(
        "checkpointing",
        if checkpoint == ActivationCheckpoint::Blocks {
            "on"
        } else {
            "off"
        },
    );
    r.num("positions", n as u64);
    r.num("lora_rank", LORA_RANK as u64);
    r.num("device_budget_bytes", LORA_DEVICE_BUDGET);
    let m = match MetalBackend::new(Budget::new(LORA_DEVICE_BUDGET)) {
        Ok(m) => m,
        Err(e) => refused(r, "MetalBackend::new", e),
    };
    let host_budget = Budget::new(24 * GIB);
    let config = LoraConfig::new(LORA_RANK, LORA_ALPHA, 0.0).unwrap();
    let (base, adapters) = {
        let file = SafeTensors::open(&single_weights_file(&snap.weights_path)).unwrap();
        let host = match load_hf(
            &spec,
            &file,
            &cfg.tower_prefix,
            DType::Bf16,
            DType::Bf16,
            &host_budget,
        ) {
            Ok(h) => h,
            Err(e) => refused(r, "load_hf", e),
        };
        let targets = match lora_targets(&spec, &file, &cfg.tower_prefix, &LORA_KINDS) {
            Ok(t) => t,
            Err(e) => refused(r, "lora_targets", e),
        };
        let host_adapters =
            match Qwen35Lora::init(&spec, &targets, &config, LORA_SEED, &host_budget) {
                Ok(a) => a,
                Err(e) => refused(r, "lora init", e),
            };
        r.num("adapter_sites", targets.len() as u64);
        let base = match host.try_map(&mut |t| m.upload(t)) {
            Ok(d) => d,
            Err(e) => refused(r, "upload", e),
        };
        let adapters = match host_adapters.try_map(&mut |s| s.try_map(&mut |t| m.upload(t))) {
            Ok(d) => d,
            Err(e) => refused(r, "upload adapters", e),
        };
        (base, adapters)
    };
    if let Err(e) = m.sync() {
        refused(r, "sync weights", e);
    }
    let resident = m.memory_and_reset_peak().unwrap();
    r.num("resident_bytes", resident.allocated);
    r.num("budget_resident_bytes", m.budget().live_bytes().unwrap());
    r.num("working_set_bytes", resident.recommended_working_set);
    let fp = footprint_or_refuse(&r);
    r.num("resident_footprint_bytes", fp.now);
    r.num("resident_footprint_max_bytes", fp.lifetime_max);
    r.phase("resident");
    let tables = Qwen35Tables::new(&spec, n, &host_budget)
        .unwrap()
        .upload(&m)
        .unwrap();
    let ids_t = Tensor::from_u32(&ids[..n], &[1, n], &host_budget).unwrap();
    let targets = Tensor::from_u32(&ids[1..], &[1, n], &host_budget).unwrap();
    let mut trainer = match LoraTrainer::new(
        m.clone(),
        spec,
        base,
        tables,
        adapters,
        config,
        checkpoint,
        &host_budget,
    ) {
        Ok(t) => t,
        Err(f) => refused(r, f.phase, f.error),
    };
    let step = |trainer: &mut LoraTrainer, r: &Row| -> (f32, f64) {
        let t0 = Instant::now();
        match trainer.step(&ids_t, &targets, &host_budget) {
            Ok(loss) => (loss, ms(t0.elapsed())),
            Err(f) => refused(r.clone(), f.phase, f.error),
        }
    };
    // The warmup step is the first step: its wall includes shader compilation
    // and first-use allocation, which is what a run's first step costs.
    let (first_loss, first_ms) = step(&mut trainer, &r);
    let first = m.memory().unwrap();
    r.real("first_step_ms", first_ms);
    r.num("first_step_peak_bytes", first.peak_allocated);
    r.text("first_step_loss", &format!("{first_loss:.6}"));
    let fp = footprint_or_refuse(&r);
    r.num("first_step_footprint_max_bytes", fp.lifetime_max);
    r.phase("first step");
    let mut walls = Vec::with_capacity(TIMED_STEPS);
    let mut last_loss = first_loss;
    for _ in 0..TIMED_STEPS {
        let (loss, wall) = step(&mut trainer, &r);
        walls.push(wall);
        last_loss = loss;
    }
    let end = m.memory().unwrap();
    let mean_ms = walls.iter().sum::<f64>() / walls.len() as f64;
    for (k, w) in ["step1_ms", "step2_ms", "step3_ms"].into_iter().zip(&walls) {
        r.real(k, *w);
    }
    r.real("mean_step_ms", mean_ms);
    r.real("tokens_per_s", n as f64 / (mean_ms / 1e3));
    r.num("step_peak_bytes", end.peak_allocated);
    r.num("end_allocated_bytes", end.allocated);
    r.text("last_loss", &format!("{last_loss:.6}"));
    let fp = footprint_or_refuse(&r);
    r.num("end_footprint_bytes", fp.now);
    r.num("footprint_max_bytes", fp.lifetime_max);
    drop(trainer);
    r
}

/// The step of `tape-lora` on the tiny two-layer fixture (one gated delta
/// net layer, one attention layer, vocab 8,320): after two steps every
/// adapter tensor has moved (`B` at the first, `A` at the second, once `B`
/// is not zero), no base tensor has changed a bit, the loss is finite and
/// changes, and `LoraTrainer::step` itself refuses a gradient on a frozen
/// base parameter. Run it like the rows:
///
/// ```text
/// cargo test -p ojas-qwen35 --release --test gpu_memory_rows \
///     -- --ignored --test-threads=1 gpu_lora_step_trains_only_the_adapters
/// ```
#[test]
#[ignore]
fn gpu_lora_step_trains_only_the_adapters() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../ojas-model/tests/fixtures/qwen35_tiny");
    let snap = Snapshot::from_dir(&dir).unwrap();
    let cfg = snap.config.clone();
    let spec = cfg.tape_spec();
    let n = 69;
    let ids: Vec<u32> = (0..=n as u32).map(|i| (i * 7919 + 13) % 8000).collect();
    let m = MetalBackend::new(Budget::new(2 * GIB)).unwrap();
    let host_budget = Budget::new(GIB);
    let file = SafeTensors::open(&single_weights_file(&snap.weights_path)).unwrap();
    let host = load_hf(
        &spec,
        &file,
        &cfg.tower_prefix,
        DType::Bf16,
        DType::Bf16,
        &host_budget,
    )
    .unwrap();
    let targets = lora_targets(&spec, &file, &cfg.tower_prefix, &LORA_KINDS).unwrap();
    assert_eq!(targets.len(), 3 + 4 + 2 * 3, "ten kinds over two layers");
    let config = LoraConfig::new(LORA_RANK.min(4), LORA_ALPHA.min(4.0), 0.0).unwrap();
    let host_adapters =
        Qwen35Lora::init(&spec, &targets, &config, LORA_SEED, &host_budget).unwrap();
    let base = host.try_map(&mut |t| m.upload(t)).unwrap();
    let adapters = host_adapters
        .try_map(&mut |s| s.try_map(&mut |t| m.upload(t)))
        .unwrap();
    let tables = Qwen35Tables::new(&spec, n, &host_budget)
        .unwrap()
        .upload(&m)
        .unwrap();
    let ids_t = Tensor::from_u32(&ids[..n], &[1, n], &host_budget).unwrap();
    let targets_t = Tensor::from_u32(&ids[1..], &[1, n], &host_budget).unwrap();
    let mut trainer = LoraTrainer::new(
        m.clone(),
        spec,
        base,
        tables,
        adapters,
        config,
        ActivationCheckpoint::Blocks,
        &host_budget,
    )
    .unwrap();
    // bf16 widens to f32 exactly, so equal widened values are equal bits.
    let read = |t: &Tensor| -> Vec<f32> {
        let wide = if t.dtype() == DType::Bf16 {
            m.to_f32(t).unwrap()
        } else {
            t.clone()
        };
        m.download(&wide).unwrap().to_f32_vec().unwrap()
    };
    let snapshot_base = |trainer: &LoraTrainer| -> Vec<Vec<f32>> {
        trainer.base.values().into_iter().map(&read).collect()
    };
    let snapshot_adapters = |trainer: &mut LoraTrainer| -> Vec<Vec<f32>> {
        adapter_tensors(&mut trainer.adapters)
            .into_iter()
            .map(|t| read(t))
            .collect()
    };
    let base_before = snapshot_base(&trainer);
    let a0 = snapshot_adapters(&mut trainer);
    let l1 = trainer.step(&ids_t, &targets_t, &host_budget).unwrap();
    let a1 = snapshot_adapters(&mut trainer);
    let l2 = trainer.step(&ids_t, &targets_t, &host_budget).unwrap();
    let a2 = snapshot_adapters(&mut trainer);
    assert!(l1.is_finite() && l2.is_finite(), "losses {l1} {l2}");
    assert_ne!(
        l1.to_bits(),
        l2.to_bits(),
        "the first update left the loss as it was"
    );
    // Adapter tensors come as (A, B part, ...) per site; B starts at zero.
    for (i, (before, after)) in a0.iter().zip(&a1).enumerate() {
        let is_b = before.iter().all(|&v| v == 0.0);
        if is_b {
            assert_ne!(before, after, "B tensor {i} did not move at step 1");
        }
    }
    for (i, (before, after)) in a1.iter().zip(&a2).enumerate() {
        assert_ne!(before, after, "adapter tensor {i} did not move at step 2");
    }
    assert!(
        a0.iter().zip(&a2).all(|(x, y)| x != y),
        "an adapter tensor is unchanged after two steps"
    );
    assert_eq!(
        base_before,
        snapshot_base(&trainer),
        "a frozen base tensor changed"
    );
}

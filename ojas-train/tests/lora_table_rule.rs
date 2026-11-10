//! The optimizer table a LoRA run is driven with.
//!
//! Lappi's LoRA recipe (`lora.lora_param_groups`, `recipe::lora_optimizer_table`) trains the
//! adapters at 1.0x the schedule and D17 and the span head at `HEAD_LR_SCALE` (0.5x). The loop's
//! `train` derives every per-entry number from `recipe::optimizer_table`, the full fine-tune
//! table, which puts the host heads at 1.0x. Pinned here:
//!
//! - `train` refuses a provider of LoRA adapter leaves next to host entries, instead of training
//!   the heads at twice the recipe's rate under a row that names the LoRA recipe;
//! - `train_with_table(TableRule::Lora)` steps the host heads at `HEAD_LR_SCALE` times the
//!   schedule and hands the provider the adapters' 1.0 and the recipe's weight decay;
//! - `train_with_table(TableRule::Full)` is `train`, unchanged for a provider with no adapter.

use std::path::{Path, PathBuf};

use ojas_train::lappi::mock::ToyProvider;
use ojas_train::lappi::recipe::{HEAD_LR_SCALE, LORA_WEIGHT_DECAY, LoraHostNames, OptimizerRecipe};
use ojas_train::lappi::run_control::{ConsumedPrefix, MonotonicClock, WallClockCap};
use ojas_train::lappi::schedule::LrSchedule;
use ojas_train::lappi::step::{
    AdamWHyper, BankMode, HiddenGrad, ParamSpec, SequenceJob, StepError, StepProvider,
};
use ojas_train::lappi::trainer::{
    BatchCounts, ConsumedBatch, Hooks, HostParams, MicroBatchPlan, MicroLoss, Objective, Progress,
    SequenceWork, TableRule, TrainConfig, TrainError, train, train_with_table,
};

const VOCAB: usize = 8;
const LR: f64 = 1e-2;
const BETA2: f64 = 0.999;
const D17: &str = "d17.delta";
const HEAD: [&str; 1] = ["span_head.start"];
const A: &str = "layers.0.self_attn.q_proj.lora_A.weight";
const B: &str = "layers.0.self_attn.q_proj.lora_B.weight";

/// Two adapter leaves and nothing else trainable: what `TapeStepProvider::open_lora` lists.
/// Every gradient is 0.1; the per-entry vectors `adamw_step` is handed are recorded.
struct AdapterStub {
    specs: Vec<ParamSpec>,
    values: Vec<Vec<f32>>,
    step: u64,
    seen: Vec<(Vec<f64>, Vec<f64>)>,
}

impl AdapterStub {
    fn new() -> Self {
        Self {
            specs: vec![ParamSpec::new(A, &[1, 2]), ParamSpec::new(B, &[2, 1])],
            values: vec![vec![0.5, -0.5], vec![0.0, 0.0]],
            step: 0,
            seen: Vec::new(),
        }
    }
}

impl StepProvider for AdapterStub {
    fn parameters(&self) -> &[ParamSpec] {
        &self.specs
    }

    fn hidden_size(&self) -> usize {
        2
    }

    fn vocab_size(&self) -> usize {
        VOCAB
    }

    fn accumulate(
        &mut self,
        job: &SequenceJob<'_>,
        _mode: BankMode,
        _external: Option<&mut HiddenGrad<'_>>,
    ) -> Result<f64, StepError> {
        job.validate(VOCAB)?;
        Ok(1.0)
    }

    fn grad_sq_norm(&mut self) -> Result<f64, StepError> {
        Ok(self.values.iter().map(|v| v.len() as f64 * 0.01).sum())
    }

    fn supports_lr_scale(&self) -> bool {
        true
    }

    fn adamw_step(
        &mut self,
        _hyper: &AdamWHyper,
        step: u64,
        lr_scale: &[f64],
        weight_decay: &[f64],
    ) -> Result<(), StepError> {
        self.seen.push((lr_scale.to_vec(), weight_decay.to_vec()));
        self.step = step;
        Ok(())
    }

    fn step_count(&self) -> u64 {
        self.step
    }

    fn read_parameters(&mut self) -> Result<Vec<Vec<f32>>, StepError> {
        Ok(self.values.clone())
    }

    fn describe(&self) -> String {
        "adapter stub".into()
    }

    fn save_state(&mut self, _dir: &Path) -> Result<Vec<PathBuf>, StepError> {
        Err(StepError::Unsupported("the stub keeps no state".into()))
    }

    fn load_state(&mut self, _dir: &Path) -> Result<(), StepError> {
        Err(StepError::Unsupported("the stub keeps no state".into()))
    }
}

struct Batch(u64);

impl ConsumedBatch for Batch {
    fn index(&self) -> u64 {
        self.0
    }

    fn fold_into(&self, prefix: &mut ConsumedPrefix) -> Result<(), TrainError> {
        prefix.fold(&[&self.0.to_be_bytes()]);
        Ok(())
    }
}

/// D17 and one span-head tensor as host entries, each with gradient 0.05 everywhere.
struct Heads {
    names: Vec<String>,
    values: Vec<Vec<f32>>,
    grads: Vec<Option<Vec<f32>>>,
}

impl Heads {
    fn new() -> Self {
        Self {
            names: vec![D17.into(), HEAD[0].into()],
            values: vec![vec![0.0; 3], vec![0.25; 2]],
            grads: vec![None, None],
        }
    }
}

impl HostParams for Heads {
    fn entries(&self) -> Vec<ParamSpec> {
        self.names
            .iter()
            .zip(&self.values)
            .map(|(n, v)| ParamSpec::new(n.clone(), &[v.len()]))
            .collect()
    }

    fn zero_grads(&mut self) {
        self.grads.iter_mut().for_each(|g| *g = None);
    }

    fn grads(&self) -> Vec<Option<&[f32]>> {
        self.grads.iter().map(|g| g.as_deref()).collect()
    }

    fn values(&self) -> Vec<&[f32]> {
        self.values.iter().map(Vec::as_slice).collect()
    }

    fn values_and_grads(&mut self) -> Vec<(&mut [f32], Option<&[f32]>)> {
        self.values
            .iter_mut()
            .zip(self.grads.iter())
            .map(|(v, g)| (v.as_mut_slice(), g.as_deref()))
            .collect()
    }
}

impl Objective for Heads {
    type Batch = Batch;

    fn run_kind(&self) -> &'static str {
        "ft"
    }

    fn plan(&mut self, batch: &Batch) -> Result<MicroBatchPlan, TrainError> {
        let tokens: Vec<u32> = (0..5)
            .map(|k| ((batch.0 * 3 + k * 2) % VOCAB as u64) as u32)
            .collect();
        Ok(MicroBatchPlan {
            sequences: vec![SequenceWork {
                positions: (0..4).collect(),
                targets: tokens[1..].to_vec(),
                tokens,
                hidden_at: Vec::new(),
            }],
            row_scale: 0.25,
            counts: BatchCounts::default(),
        })
    }

    fn hidden_grad(
        &mut self,
        _seq: usize,
        _positions: &[u32],
        _hidden: &[f32],
        _width: usize,
    ) -> Result<Vec<f32>, TrainError> {
        Err(TrainError::Refused("no outside loss".into()))
    }

    fn close(&mut self, row_loss_sums: &[f64]) -> Result<MicroLoss, TrainError> {
        for (g, v) in self.grads.iter_mut().zip(&self.values) {
            *g = Some(vec![0.05; v.len()]);
        }
        Ok(MicroLoss {
            total: row_loss_sums.iter().sum::<f64>() * 0.25,
            channels: Vec::new(),
        })
    }

    fn host(&mut self) -> Option<&mut dyn HostParams> {
        Some(self)
    }

    fn host_ref(&self) -> Option<&dyn HostParams> {
        Some(self)
    }
}

fn config() -> TrainConfig {
    TrainConfig {
        // The schedule refuses a warmup that is not shorter than the run, so the run has two
        // steps; step 0's rate is the peak either way, and one batch ends it (DataExhausted).
        schedule: LrSchedule::new(LR, 2, 1, 0.0).expect("schedule"),
        cap: WallClockCap::new(3600.0).expect("cap"),
        grad_accum: 1,
        optimizer: OptimizerRecipe {
            lr: LR,
            beta2: BETA2,
            lower_layers: None,
        },
        max_grad_norm: 1e30,
        epoch: 0,
        epochs: 1,
        seed: 7,
        checkpoint: None,
        eta: None,
    }
}

fn names() -> LoraHostNames<'static> {
    LoraHostNames {
        d17_entry: D17,
        span_head: &HEAD,
    }
}

/// One step under `rule`; the provider, the heads and the step's learning rate.
fn one_step(rule: Option<TableRule<'_>>) -> (Result<(), TrainError>, AdapterStub, Heads, f64) {
    let mut provider = AdapterStub::new();
    let mut heads = Heads::new();
    let mut lr = f64::NAN;
    let mut on_progress = |p: &Progress| lr = p.record.lr;
    let mut hooks = Hooks {
        on_progress: Some(&mut on_progress),
    };
    let batches = (0..1).map(|i| Ok::<_, TrainError>(Batch(i)));
    let clock = MonotonicClock::new();
    let r = match rule {
        None => train(
            &mut provider,
            &mut heads,
            batches,
            &config(),
            &clock,
            None,
            &mut hooks,
        ),
        Some(rule) => train_with_table(
            &mut provider,
            &mut heads,
            batches,
            &config(),
            rule,
            &clock,
            None,
            &mut hooks,
        ),
    }
    .map(|_| ());
    (r, provider, heads, lr)
}

#[test]
fn train_refuses_lora_adapters_next_to_host_heads() {
    let (r, provider, heads, _) = one_step(None);
    let e = r.expect_err("the full table would train the heads at the adapters' rate");
    let text = e.to_string();
    assert!(
        text.contains(A) && text.contains("TableRule::Lora"),
        "{text}"
    );
    assert_eq!(provider.step_count(), 0, "refused before any step");
    assert_eq!(heads.values, Heads::new().values, "no head moved");
}

#[test]
fn the_full_rule_refuses_the_same_run() {
    let (r, provider, _, _) = one_step(Some(TableRule::Full));
    r.expect_err("TableRule::Full is train's rule");
    assert_eq!(provider.step_count(), 0);
}

#[test]
fn the_lora_rule_steps_the_heads_at_the_head_scale() {
    let (r, provider, heads, lr) = one_step(Some(TableRule::Lora(names())));
    r.expect("one LoRA step");
    assert_eq!(provider.step_count(), 1);
    assert!(lr.is_finite() && lr > 0.0, "the step's lr {lr}");
    // The adapters at 1.0 and the recipe's decay, for both entries.
    assert_eq!(
        provider.seen,
        vec![(vec![1.0, 1.0], vec![LORA_WEIGHT_DECAY, LORA_WEIGHT_DECAY])]
    );
    // AdamW's first step moves each element by lr_eff * (g / (|g| + eps)) plus the decoupled
    // decay lr_eff * wd * w: with g = 0.05 the first term is 1 - 2e-7. The rate the heads moved
    // at, divided by the schedule's, is the scale they were stepped with.
    let start = Heads::new().values;
    for (entry, (now, was)) in heads.values.iter().zip(&start).enumerate() {
        for (&x, &x0) in now.iter().zip(was) {
            let per_unit = 1.0 + LORA_WEIGHT_DECAY * f64::from(x0);
            let scale = f64::from(x0 - x) / (lr * per_unit);
            assert!(
                (scale - HEAD_LR_SCALE).abs() < 1e-3,
                "host entry {entry}: stepped at {scale}x the schedule, not {HEAD_LR_SCALE}x"
            );
        }
    }
}

#[test]
fn the_full_rule_is_train_for_a_provider_with_no_adapter() {
    // The toy's entries are all base parameters and no host entry is involved: both entry points
    // take the same step to the bit.
    let run = |rule: Option<TableRule<'_>>| {
        let mut p = ToyProvider::new(VOCAB, 4, 2, 42).expect("toy");
        let mut obj = NoHost;
        let batches = (0..2).map(|i| Ok::<_, TrainError>(Batch(i)));
        let clock = MonotonicClock::new();
        let mut hooks = Hooks::default();
        let cfg = TrainConfig {
            schedule: LrSchedule::new(LR, 2, 1, 0.0).expect("schedule"),
            ..config()
        };
        match rule {
            None => train(&mut p, &mut obj, batches, &cfg, &clock, None, &mut hooks),
            Some(r) => {
                train_with_table(&mut p, &mut obj, batches, &cfg, r, &clock, None, &mut hooks)
            }
        }
        .expect("two steps");
        p.read_parameters()
            .expect("values")
            .iter()
            .map(|v| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>())
            .collect::<Vec<_>>()
    };
    assert_eq!(run(None), run(Some(TableRule::Full)));
}

/// Next-token loss with no host entries.
struct NoHost;

impl Objective for NoHost {
    type Batch = Batch;

    fn run_kind(&self) -> &'static str {
        "ft"
    }

    fn plan(&mut self, batch: &Batch) -> Result<MicroBatchPlan, TrainError> {
        Heads::new().plan(batch)
    }

    fn hidden_grad(
        &mut self,
        _seq: usize,
        _positions: &[u32],
        _hidden: &[f32],
        _width: usize,
    ) -> Result<Vec<f32>, TrainError> {
        Err(TrainError::Refused("no outside loss".into()))
    }

    fn close(&mut self, row_loss_sums: &[f64]) -> Result<MicroLoss, TrainError> {
        Ok(MicroLoss {
            total: row_loss_sums.iter().sum::<f64>() * 0.25,
            channels: Vec::new(),
        })
    }

    fn host(&mut self) -> Option<&mut dyn HostParams> {
        None
    }

    fn host_ref(&self) -> Option<&dyn HostParams> {
        None
    }
}

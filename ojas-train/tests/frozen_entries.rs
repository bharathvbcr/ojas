//! Frozen parameter-table entries (`ParamSpec::trainable == false`) and the checkpoint context.
//!
//! The contract: the per-entry vectors, `grad_sq_norm`, `read_parameters` and
//! `parameters_digest` cover the TRAINABLE subsequence only, and a provider whose entries are all
//! trainable (the toy; Lappi's tessl provider) behaves exactly as before. These tests drive the
//! real loop over `ToyProvider` with and without a frozen entry.

use std::path::PathBuf;

use ojas_train::lappi::mock::ToyProvider;
use ojas_train::lappi::recipe::{self, BETA1, EPS, OptimizerRecipe, WEIGHT_DECAY};
use ojas_train::lappi::run_control::{ConsumedPrefix, MonotonicClock, WallClockCap};
use ojas_train::lappi::schedule::LrSchedule;
use ojas_train::lappi::step::{
    AdamWHyper, BankMode, CheckpointContext, ParamSpec, RowTargets, SequenceJob, StepError,
    StepProvider, StorageDtype,
};
use ojas_train::lappi::trainer::{
    BatchCounts, CheckpointPolicy, ConsumedBatch, HostParams, Hooks, MicroBatchPlan, MicroLoss,
    Objective, ResumeState, SequenceWork, TrainConfig, TrainError, TrainResult, parameters_digest,
    train,
};

const VOCAB: usize = 8;
const HIDDEN: usize = 4;
const LAYERS: usize = 2;
const SEED: u64 = 7;
const LR: f64 = 1e-2;
const BETA2: f64 = 0.999;

struct Batch {
    index: u64,
    tokens: Vec<u32>,
}

fn batches(n: u64) -> Vec<Batch> {
    (0..n)
        .map(|i| Batch {
            index: i,
            tokens: (0..5).map(|k| ((i * 3 + k * 2) % VOCAB as u64) as u32).collect(),
        })
        .collect()
}

impl ConsumedBatch for Batch {
    fn index(&self) -> u64 {
        self.index
    }

    fn fold_into(&self, prefix: &mut ConsumedPrefix) -> Result<(), TrainError> {
        let tokens: Vec<u8> = self.tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
        prefix.fold(&[&self.index.to_be_bytes(), &tokens]);
        Ok(())
    }
}

/// Next-token cross-entropy over every position but the last; no outside loss, no host entries.
struct NextToken {
    scale: f32,
}

impl Objective for NextToken {
    type Batch = Batch;

    fn run_kind(&self) -> &'static str {
        "ft"
    }

    fn plan(&mut self, batch: &Batch) -> Result<MicroBatchPlan, TrainError> {
        let t = batch.tokens.len();
        self.scale = 1.0 / (t - 1) as f32;
        Ok(MicroBatchPlan {
            sequences: vec![SequenceWork {
                tokens: batch.tokens.clone(),
                positions: (0..t as u32 - 1).collect(),
                targets: batch.tokens[1..].to_vec(),
                hidden_at: Vec::new(),
            }],
            row_scale: self.scale,
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
        Err(TrainError::Refused("this objective has no outside loss".into()))
    }

    fn close(&mut self, row_loss_sums: &[f64]) -> Result<MicroLoss, TrainError> {
        Ok(MicroLoss {
            total: row_loss_sums.iter().sum::<f64>() * f64::from(self.scale),
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

fn plain() -> ToyProvider {
    ToyProvider::new(VOCAB, HIDDEN, LAYERS, 42).expect("toy provider")
}

fn with_frozen() -> ToyProvider {
    plain()
        .with_frozen_entry("base.frozen.weight", &[3, 2], StorageDtype::Bf16)
        .expect("frozen entry")
}

fn config(steps: u64, checkpoint: Option<CheckpointPolicy>) -> TrainConfig {
    TrainConfig {
        schedule: LrSchedule::new(LR, steps, 1, LR / 10.0).expect("schedule"),
        cap: WallClockCap::new(3600.0).expect("cap"),
        grad_accum: 1,
        optimizer: OptimizerRecipe {
            lr: LR,
            beta2: BETA2,
            lower_layers: None,
        },
        max_grad_norm: 1.0,
        epoch: 0,
        epochs: 1,
        seed: SEED,
        checkpoint,
        eta: None,
    }
}

fn run<P: StepProvider>(
    provider: &mut P,
    steps: u64,
    n_batches: u64,
    checkpoint: Option<CheckpointPolicy>,
    resume: Option<ResumeState>,
) -> TrainResult {
    train(
        provider,
        &mut NextToken { scale: 0.0 },
        batches(n_batches).into_iter().map(Ok::<_, TrainError>),
        &config(steps, checkpoint),
        &MonotonicClock::new(),
        resume,
        &mut Hooks::default(),
    )
    .expect("train")
}

fn job(tokens: &[u32], scale: f32) -> (Vec<u32>, Vec<u32>, f32) {
    let t = tokens.len() as u32;
    ((0..t - 1).collect(), tokens[1..].to_vec(), scale)
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ojas-frozen-{name}-{}", std::process::id()));
    if dir.exists() {
        std::fs::remove_dir_all(&dir).expect("clear scratch");
    }
    dir
}

#[test]
fn a_frozen_entry_is_listed_but_absent_from_every_trainable_view() {
    let mut a = plain();
    let mut b = with_frozen();
    assert_eq!(b.parameters().len(), a.parameters().len() + 1);
    let frozen: Vec<&ParamSpec> = b.parameters().iter().filter(|s| !s.trainable).collect();
    assert_eq!(frozen.len(), 1);
    assert_eq!(frozen[0].storage, StorageDtype::Bf16);
    assert!(a.parameters().iter().all(|s| s.trainable && s.storage == StorageDtype::F32));

    assert_eq!(b.read_parameters().unwrap(), a.read_parameters().unwrap());
    assert_eq!(b.grads().len(), a.grads().len());
    assert_eq!(b.frozen_values().len(), 1);

    let tokens = [1u32, 2, 3, 4, 5];
    let (positions, targets, scale) = job(&tokens, 0.25);
    let j = SequenceJob {
        tokens: &tokens,
        rows: RowTargets {
            positions: &positions,
            targets: &targets,
            scale,
        },
        hidden_at: &[],
    };
    let la = a.accumulate(&j, BankMode::Overwrite, None).unwrap();
    let lb = b.accumulate(&j, BankMode::Overwrite, None).unwrap();
    assert_eq!(la.to_bits(), lb.to_bits());
    assert_eq!(
        a.grad_sq_norm().unwrap().to_bits(),
        b.grad_sq_norm().unwrap().to_bits()
    );

    let hyper = AdamWHyper {
        lr: LR,
        beta1: BETA1,
        beta2: BETA2,
        eps: EPS,
        grad_scale: 1.0,
    };
    let n = a.parameters().len();
    let long = vec![1.0; n + 1];
    let wd_long = vec![WEIGHT_DECAY; n + 1];
    assert!(
        matches!(
            b.adamw_step(&hyper, 1, &long, &wd_long),
            Err(StepError::Contract(_))
        ),
        "a vector with a slot for the frozen entry is the old contract and is refused"
    );
    assert_eq!(b.step_count(), 0, "a refusal moves nothing");
    b.adamw_step(&hyper, 1, &vec![1.0; n], &vec![WEIGHT_DECAY; n])
        .unwrap();
    assert_eq!(b.frozen_values()[0], with_frozen().frozen_values()[0]);

    assert_eq!(
        parameters_digest(&mut plain(), None).unwrap(),
        parameters_digest(&mut with_frozen(), None).unwrap(),
        "the digest covers the trainable subsequence only"
    );
}

#[test]
fn training_with_a_frozen_entry_is_the_all_trainable_trajectory() {
    let mut a = plain();
    let mut b = with_frozen();
    let ra = run(&mut a, 4, 4, None, None);
    let rb = run(&mut b, 4, 4, None, None);
    assert_eq!(ra.steps, rb.steps);
    assert_eq!(ra.loss_log_digest().unwrap(), rb.loss_log_digest().unwrap());
    assert_eq!(ra.consumed_digest, rb.consumed_digest);
    for (x, y) in a.values().iter().zip(b.values()) {
        assert_eq!(x.len(), y.len());
        assert!(x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()));
    }
    assert_eq!(
        parameters_digest(&mut a, None).unwrap(),
        parameters_digest(&mut b, None).unwrap()
    );
    assert_eq!(b.frozen_values()[0], with_frozen().frozen_values()[0]);
}

#[test]
fn an_all_trainable_loop_step_is_the_direct_provider_step() {
    let mut looped = plain();
    // The schedule refuses a warmup that is not shorter than the run, so the run has two steps;
    // each loop step is compared with the direct provider step on the same batch.
    let r = run(&mut looped, 2, 2, None, None);
    assert_eq!(r.steps.len(), 2);

    let mut direct = plain();
    let all = batches(2);
    for (i, b) in all.iter().enumerate() {
        let rec = r.steps[i];
        let (positions, targets, scale) = job(&b.tokens, 1.0 / (b.tokens.len() - 1) as f32);
        let j = SequenceJob {
            tokens: &b.tokens,
            rows: RowTargets {
                positions: &positions,
                targets: &targets,
                scale,
            },
            hidden_at: &[],
        };
        direct.accumulate(&j, BankMode::Overwrite, None).unwrap();
        let sq = direct.grad_sq_norm().unwrap();
        assert_eq!(rec.grad_norm.to_bits(), sq.sqrt().to_bits(), "step {}", i + 1);
        let coef = recipe::clip_coefficient(sq, 1.0).unwrap();
        assert_eq!(rec.clip_coefficient.to_bits(), coef.to_bits(), "step {}", i + 1);
        let hyper = AdamWHyper {
            lr: rec.lr,
            beta1: BETA1,
            beta2: BETA2,
            eps: EPS,
            grad_scale: coef,
        };
        let n = direct.parameters().len();
        direct
            .adamw_step(&hyper, i as u64 + 1, &vec![1.0; n], &vec![WEIGHT_DECAY; n])
            .unwrap();
    }
    for (x, y) in looped.values().iter().zip(direct.values()) {
        assert!(x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()));
    }
}

#[test]
fn a_checkpoint_of_a_frozen_provider_resumes_the_exact_trajectory() {
    let dir = scratch("resume");
    let policy = || CheckpointPolicy {
        dir: dir.clone(),
        every: 2,
    };

    let mut straight = with_frozen();
    run(&mut straight, 4, 4, None, None);

    let mut first = with_frozen();
    let r1 = run(&mut first, 4, 2, Some(policy()), None);
    let ckpt = r1.final_checkpoint.expect("a checkpoint at the end");
    let state = ResumeState::read(&ckpt).expect("read checkpoint");
    assert_eq!(state.optimizer_step, 2);

    let mut resumed = with_frozen();
    run(&mut resumed, 4, 4, Some(policy()), Some(state));
    assert_eq!(
        parameters_digest(&mut straight, None).unwrap(),
        parameters_digest(&mut resumed, None).unwrap()
    );
    std::fs::remove_dir_all(&dir).expect("remove scratch");
}

/// A toy provider that records the context the loop hands its checkpoint methods.
struct Recording {
    inner: ToyProvider,
    saved: Vec<CheckpointContext>,
    loaded: Vec<CheckpointContext>,
}

impl StepProvider for Recording {
    fn parameters(&self) -> &[ParamSpec] {
        self.inner.parameters()
    }
    fn hidden_size(&self) -> usize {
        self.inner.hidden_size()
    }
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
    fn accumulate(
        &mut self,
        job: &SequenceJob<'_>,
        mode: BankMode,
        external: Option<&mut ojas_train::lappi::step::HiddenGrad<'_>>,
    ) -> Result<f64, StepError> {
        self.inner.accumulate(job, mode, external)
    }
    fn grad_sq_norm(&mut self) -> Result<f64, StepError> {
        self.inner.grad_sq_norm()
    }
    fn supports_lr_scale(&self) -> bool {
        self.inner.supports_lr_scale()
    }
    fn adamw_step(
        &mut self,
        hyper: &AdamWHyper,
        step: u64,
        lr_scale: &[f64],
        weight_decay: &[f64],
    ) -> Result<(), StepError> {
        self.inner.adamw_step(hyper, step, lr_scale, weight_decay)
    }
    fn step_count(&self) -> u64 {
        self.inner.step_count()
    }
    fn read_parameters(&mut self) -> Result<Vec<Vec<f32>>, StepError> {
        self.inner.read_parameters()
    }
    fn describe(&self) -> String {
        self.inner.describe()
    }
    fn save_state(&mut self, dir: &std::path::Path) -> Result<Vec<PathBuf>, StepError> {
        self.inner.save_state(dir)
    }
    fn load_state(&mut self, dir: &std::path::Path) -> Result<(), StepError> {
        self.inner.load_state(dir)
    }
    fn save_state_with(
        &mut self,
        dir: &std::path::Path,
        context: &CheckpointContext,
    ) -> Result<Vec<PathBuf>, StepError> {
        self.saved.push(context.clone());
        self.inner.save_state(dir)
    }
    fn load_state_with(
        &mut self,
        dir: &std::path::Path,
        context: &CheckpointContext,
    ) -> Result<(), StepError> {
        self.loaded.push(context.clone());
        self.inner.load_state(dir)
    }
}

#[test]
fn the_loop_hands_the_checkpoint_context_to_save_and_resume() {
    let dir = scratch("context");
    let policy = || CheckpointPolicy {
        dir: dir.clone(),
        every: 2,
    };
    let mut first = Recording {
        inner: plain(),
        saved: Vec::new(),
        loaded: Vec::new(),
    };
    let r1 = run(&mut first, 4, 2, Some(policy()), None);
    assert_eq!(first.saved.len(), 1);
    let ctx = &first.saved[0];
    assert_eq!(ctx.optimizer_step, 2);
    assert_eq!(ctx.seed, SEED);
    assert_eq!(ctx.epoch, 0);
    assert_eq!(ctx.consumed_digest, r1.consumed_digest);

    let state = ResumeState::read(&r1.final_checkpoint.expect("checkpoint")).unwrap();
    let mut second = Recording {
        inner: plain(),
        saved: Vec::new(),
        loaded: Vec::new(),
    };
    run(&mut second, 4, 4, Some(policy()), Some(state));
    assert_eq!(second.loaded, first.saved);
    std::fs::remove_dir_all(&dir).expect("remove scratch");
}

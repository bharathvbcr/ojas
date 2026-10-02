//! The training loop: `trainer._train_loop` (`python/qd_train/trainer.py:730-952`) and
//! `QwenDecisionStep.apply` (`python/qd_train/backbone.py:1187-1195`), over any
//! [`StepProvider`].
//!
//! **One optimizer step**, as the PyTorch trainer takes it (`grad_accum` micro-batches, 1 in
//! every `tools/real_ft_run.py` run):
//!
//! 1. Each micro-batch's sequences go through [`StepProvider::accumulate`] one at a time into
//!    the bank (the first sequence of the step overwrites, the rest add), with the batch-mean
//!    scale its [`Objective`] chose; a loss outside the model (Lappi's span head) reads hidden
//!    states and hands its gradient back through the callback, and accumulates its own
//!    parameters' gradients on the host ([`HostParams`]).
//! 2. The micro-batch's loss is the objective's; a **non-finite loss stops the run here**,
//!    before the clip, before AdamW, so no moment ever sees it (`AccumulationGroup.add`, rule 3).
//! 3. When the group is full: the group's loss is the plain mean of its micro-batches' (rule 4;
//!    the gradients are summed, not averaged -- torch's `backward()` per micro-batch does the
//!    same). The clip coefficient is `max_norm / (norm + 1e-6)`, at most 1, from the provider's
//!    squared norm plus the host parameters' (`clip_grad_norm_` over tower and head together);
//!    a non-finite norm is refused before AdamW -- stricter than torch, which would write NaN
//!    parameters, and the reason this loop can promise the moments never move on a bad step.
//! 4. AdamW: the provider with per-entry learning-rate scales and weight decays from Lappi's
//!    recipe ([`crate::recipe`]), the host parameters with the same scalars at scale 1.0 (the
//!    span head joins the base group), the same `t`, the same clip coefficient.
//! 5. The cap is polled only between groups (rule 2), so no half-accumulated gradient is
//!    discarded; overshoot is at most one step.
//!
//! **Bookkeeping**, each a port: the loss log and its digest, the consumed-batch digest, strictly
//! increasing batch indices, refusal of a partial group at the end of the data (rule 1), a
//! checkpoint that resumes the exact trajectory (the consumed digest of the skipped prefix must
//! match), and the three termination reasons.

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::adamw::{adamw_update, Moments};
use serde_json::Value;

use crate::pyjson::{dumps, float_fromhex, float_hex, obj, PyJsonError, CANONICAL};
use crate::recipe::{self, OptimizerRecipe};
use crate::run_control::{hex, Clock, LossLog, LossPoint, RunClock, RunControlError, WallClockCap};
use crate::schedule::{LrSchedule, ScheduleError};
use crate::shards::ConsumedPrefix;
use crate::step::{AdamWHyper, BankMode, ParamSpec, RowTargets, SequenceJob, StepError, StepProvider};

/// `run_control.MAX_GRAD_ACCUM`.
pub const MAX_GRAD_ACCUM: u32 = 4096;

/// The checkpoint directory format this module writes and reads.
pub const CHECKPOINT_FORMAT: &str = "qd-train-checkpoint-v1";

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum TrainError {
    /// `TrainerContractViolation`: the run was asked to do something its contract forbids.
    #[error("refused: {0}")]
    Refused(String),
    /// A non-finite loss or gradient norm. The step it belongs to was not applied.
    #[error("non-finite: {0}")]
    NonFinite(String),
    #[error(transparent)]
    Step(#[from] StepError),
    #[error(transparent)]
    Schedule(#[from] ScheduleError),
    #[error(transparent)]
    RunControl(#[from] RunControlError),
    #[error(transparent)]
    Json(#[from] PyJsonError),
    #[error("io: {0}")]
    Io(String),
}

fn io(path: &Path, e: impl std::fmt::Display) -> TrainError {
    TrainError::Io(format!("{}: {e}", path.display()))
}

/// A batch the loop can account for: its index in the epoch's order, and the bytes that say
/// what it trains on.
pub trait ConsumedBatch {
    /// `Batch.index`: strictly increasing within an epoch.
    fn index(&self) -> u64;

    /// Fold this batch into `prefix` exactly as `trainer._fold` does, so the digest is the one
    /// Python computes over the same batch: for a real [`crate::shards::Batch`],
    /// [`ConsumedPrefix::fold_batch`]; for anything else, one call to [`ConsumedPrefix::fold`]
    /// with the parts `[index (8 bytes BE), bucket (8 bytes BE), tokens, lengths, slot_kind,
    /// target_index, span_target, line_starts]`, each array as its numpy buffer's bytes (C
    /// order, its own dtype) and an absent optional array as no bytes.
    fn fold_into(&self, prefix: &mut ConsumedPrefix) -> Result<(), TrainError>;
}

/// Parameters trained on the host beside the provider's (Lappi's span head).
pub trait HostParams {
    /// One entry per tensor, in a fixed order.
    fn entries(&self) -> Vec<ParamSpec>;
    /// Zero every gradient. Called at the start of each optimizer step.
    fn zero_grads(&mut self);
    /// Every gradient, in [`HostParams::entries`] order.
    fn grads(&self) -> Vec<&[f32]>;
    /// Every value, in [`HostParams::entries`] order.
    fn values(&self) -> Vec<&[f32]>;
    /// `(value, gradient)` of every tensor, in [`HostParams::entries`] order.
    fn values_and_grads(&mut self) -> Vec<(&mut [f32], &[f32])>;
}

/// One sequence of a micro-batch, as the objective plans it.
#[derive(Debug, Clone, PartialEq)]
pub struct SequenceWork {
    pub tokens: Vec<u32>,
    pub positions: Vec<u32>,
    pub targets: Vec<u32>,
    pub hidden_at: Vec<u32>,
}

/// What a micro-batch contributed to the run's counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BatchCounts {
    pub supervised_tokens: u64,
    pub span_rows: u64,
    pub padded_positions: u64,
    pub total_positions: u64,
}

impl BatchCounts {
    fn add(&mut self, o: &BatchCounts) {
        self.supervised_tokens += o.supervised_tokens;
        self.span_rows += o.span_rows;
        self.padded_positions += o.padded_positions;
        self.total_positions += o.total_positions;
    }
}

/// A micro-batch's plan: its sequences, the scale of the provider's own row loss, its counts.
#[derive(Debug, Clone, PartialEq)]
pub struct MicroBatchPlan {
    pub sequences: Vec<SequenceWork>,
    pub row_scale: f32,
    pub counts: BatchCounts,
}

/// A micro-batch's loss and its named channels (Lappi: `letter`, `span`).
#[derive(Debug, Clone, PartialEq)]
pub struct MicroLoss {
    pub total: f64,
    pub channels: Vec<(String, f64)>,
}

/// What the loop trains toward: how a batch becomes sequence jobs, the loss outside the model,
/// and how the micro-batch's loss is formed. Lappi's letter + span objective is
/// [`crate::objective::LetterSpanObjective`].
pub trait Objective {
    type Batch: ConsumedBatch;

    /// The ledger's run kind (`ft`).
    fn run_kind(&self) -> &'static str;

    /// Plan one micro-batch. Called once per batch, before any of its sequences runs.
    fn plan(&mut self, batch: &Self::Batch) -> Result<MicroBatchPlan, TrainError>;

    /// The outside loss's gradient at sequence `seq` of the batch last planned: `hidden` is
    /// `[positions.len(), width]`, row-major; returns the same shape. Accumulates the host
    /// parameters' own gradients as a side effect.
    fn hidden_grad(&mut self, seq: usize, positions: &[u32], hidden: &[f32], width: usize) -> Result<Vec<f32>, TrainError>;

    /// Close the micro-batch last planned, given the provider's unscaled row-loss sum per
    /// sequence.
    fn close(&mut self, row_loss_sums: &[f64]) -> Result<MicroLoss, TrainError>;

    /// The host parameters this objective trains, if any.
    fn host(&mut self) -> Option<&mut dyn HostParams>;

    /// The host parameters, read-only.
    fn host_ref(&self) -> Option<&dyn HostParams>;
}

/// Why a loop stopped (`TerminationReason`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Termination {
    DataExhausted,
    StepsExhausted,
    WallClockCap,
    /// [`EtaRule`]: the projected run would not finish inside the cap, so it stopped early
    /// rather than be cut off by it.
    EtaRule,
}

impl Termination {
    pub fn as_str(&self) -> &'static str {
        match self {
            Termination::DataExhausted => "data_exhausted",
            Termination::StepsExhausted => "steps_exhausted",
            Termination::WallClockCap => "wall_clock_cap",
            Termination::EtaRule => "eta_rule",
        }
    }

    /// Whether the run ended because it ran out of work, not because it ran out of time.
    pub fn finished(&self) -> bool {
        matches!(self, Termination::DataExhausted | Termination::StepsExhausted)
    }
}

/// Rung (d)'s early stop (`AUDIT/ojas-training-2026-10-01/fable-rung-d-resize.md`): once
/// optimizer step `at_step` has completed, project this process's elapsed time to the end of
/// the schedule and stop with [`Termination::EtaRule`] when the projection exceeds the cap
/// less `margin_s`. For a fresh run that is `elapsed / at_step * total_steps > cap - margin_s`
/// (rung (d): `elapsed / 10 * 200 > 21600 - 900`). A resumed run's clock and cap are its own
/// process's, so the projection is over the steps this process took and has left
/// (`at_step - first_step` of `total_steps - first_step`); a run resumed at or past `at_step`
/// was projected by the process that took that step, and is not projected again.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EtaRule {
    pub at_step: u64,
    pub margin_s: f64,
}

impl EtaRule {
    /// Refuse a rule that could not fire or could not stop anything; `train` calls this before
    /// the clock starts, and a command line can call it before anything is read.
    pub fn validate(&self, total_steps: u64, cap_s: f64) -> Result<(), TrainError> {
        if self.at_step == 0 || self.at_step >= total_steps {
            return Err(TrainError::Refused(format!(
                "the ETA rule's step {} must be in [1, {total_steps}): at step 0 nothing has been timed, \
                 and at the last step there is nothing left to stop",
                self.at_step
            )));
        }
        if !(self.margin_s.is_finite() && self.margin_s >= 0.0 && self.margin_s < cap_s) {
            return Err(TrainError::Refused(format!(
                "the ETA rule's margin {} s must be finite, non-negative and under the {cap_s} s cap",
                self.margin_s
            )));
        }
        Ok(())
    }

    /// The projected seconds for this process's whole schedule, when the rule is evaluated
    /// after `optimizer_step` steps (only at `at_step`, and only by the process that took it).
    fn projection(&self, optimizer_step: u64, first_step: u64, total_steps: u64, elapsed_s: f64) -> Option<f64> {
        if optimizer_step != self.at_step || first_step >= self.at_step {
            return None;
        }
        let taken = (self.at_step - first_step) as f64;
        Some(elapsed_s / taken * (total_steps - first_step) as f64)
    }

    /// Whether a projection stops the run: it exceeds the cap less the margin.
    pub fn stops(&self, projected_s: f64, cap_s: f64) -> bool {
        projected_s > cap_s - self.margin_s
    }
}

/// Write a checkpoint every `every` optimizer steps (and at the end) into `dir`.
#[derive(Debug, Clone, PartialEq)]
pub struct CheckpointPolicy {
    pub dir: PathBuf,
    pub every: u64,
}

/// Everything about a run that is recipe rather than data.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainConfig {
    pub schedule: LrSchedule,
    pub cap: WallClockCap,
    pub grad_accum: u32,
    pub optimizer: OptimizerRecipe,
    pub max_grad_norm: f64,
    pub epoch: u64,
    pub seed: u64,
    pub checkpoint: Option<CheckpointPolicy>,
    pub eta: Option<EtaRule>,
}

/// One optimizer step, as it was taken.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepRecord {
    pub optimizer_step: u64,
    pub lr: f64,
    pub loss: f64,
    /// The global gradient norm before clipping (tower and host together).
    pub grad_norm: f64,
    pub clip_coefficient: f64,
}

/// Handed to `on_progress` after every optimizer step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Progress {
    pub record: StepRecord,
    pub total_steps: u64,
    pub micro_batches: u64,
    pub elapsed_s: f64,
}

/// What a loop did, and why it stopped.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainResult {
    pub termination: Termination,
    pub optimizer_steps: u64,
    pub micro_batches: u64,
    pub counts: BatchCounts,
    pub loss_log: LossLog,
    pub steps: Vec<StepRecord>,
    /// Per micro-batch, per channel, in consumption order.
    pub channel_log: Vec<(String, Vec<f64>)>,
    pub consumed_digest: String,
    pub consumed_n: u64,
    /// The next batch index (`Position.index`).
    pub next_index: u64,
    pub wall_clock_s: f64,
    /// The checkpoint written at the end, if a policy was given.
    pub final_checkpoint: Option<PathBuf>,
    /// The [`EtaRule`]'s projection of this process's schedule, when this process evaluated it.
    pub eta_projected_s: Option<f64>,
}

impl TrainResult {
    pub fn loss_log_digest(&self) -> Result<String, PyJsonError> {
        self.loss_log.digest()
    }
}

/// A checkpoint read back: where the run was and what it had done.
#[derive(Debug, Clone, PartialEq)]
pub struct ResumeState {
    pub dir: PathBuf,
    pub epoch: u64,
    pub next_index: u64,
    pub optimizer_step: u64,
    pub seed: u64,
    pub schedule: LrSchedule,
    pub grad_accum: u32,
    pub loss_log: LossLog,
    pub steps: Vec<StepRecord>,
    pub channel_log: Vec<(String, Vec<f64>)>,
    pub consumed_digest: String,
    pub host_entries: Vec<ParamSpec>,
    pub host_values: Vec<Vec<f32>>,
    pub host_moments: Vec<Moments>,
}

/// Optional callbacks.
#[derive(Default)]
pub struct Hooks<'h> {
    pub on_progress: Option<&'h mut dyn FnMut(&Progress)>,
}

struct Group {
    losses: Vec<f64>,
    channels: Vec<Vec<(String, f64)>>,
}

impl Group {
    fn open(&self) -> bool {
        !self.losses.is_empty()
    }
}

/// Run the loop. `batches` yields this epoch's batches in order; a resume skips (and hashes)
/// those before the checkpoint's position.
pub fn train<P, O, I>(
    provider: &mut P,
    objective: &mut O,
    batches: I,
    cfg: &TrainConfig,
    clock: &dyn Clock,
    resume: Option<ResumeState>,
    hooks: &mut Hooks<'_>,
) -> Result<TrainResult, TrainError>
where
    P: StepProvider,
    O: Objective,
    I: IntoIterator<Item = Result<O::Batch, TrainError>>,
{
    // ---- everything refusable is refused before the clock starts -------------------------
    if !(1..=MAX_GRAD_ACCUM).contains(&cfg.grad_accum) {
        return Err(TrainError::Refused(format!(
            "grad_accum must be in [1, {MAX_GRAD_ACCUM}], got {}",
            cfg.grad_accum
        )));
    }
    if !(cfg.max_grad_norm.is_finite() && cfg.max_grad_norm > 0.0) {
        return Err(TrainError::Refused(format!("max_grad_norm must be positive, got {}", cfg.max_grad_norm)));
    }
    cfg.optimizer.validate()?;
    if let Some(eta) = &cfg.eta {
        eta.validate(cfg.schedule.total_steps(), cfg.cap.cap_s())?;
    }
    if cfg.schedule.peak_lr() != cfg.optimizer.lr {
        return Err(TrainError::Refused(format!(
            "the schedule peaks at {} but the recipe's lr is {}: two learning rates for one run",
            cfg.schedule.peak_lr(),
            cfg.optimizer.lr
        )));
    }
    let entries = provider.parameters().to_vec();
    let host_entries = objective.host_ref().map(|h| h.entries()).unwrap_or_default();
    // Every per-entry number the optimizer uses comes out of this one table (Amendment 2 (ii)).
    let table = recipe::optimizer_table(&entries, &host_entries, &cfg.optimizer)?;
    let (provider_rows, host_rows) = table.split_at(entries.len());
    let lr_scale: Vec<f64> = provider_rows.iter().map(|e| e.lr_scale).collect();
    let weight_decay: Vec<f64> = provider_rows.iter().map(|e| e.weight_decay).collect();
    let host_rows = host_rows.to_vec();
    // One AdamWHyper serves every entry, so eps and the betas must be one value across the table.
    let Some(first) = table.first() else {
        return Err(TrainError::Refused("the model has no parameters to train".into()));
    };
    let (beta1, beta2, eps) = (first.beta1, first.beta2, first.eps);
    if let Some(e) = table.iter().find(|e| e.beta1 != beta1 || e.beta2 != beta2 || e.eps != eps) {
        return Err(TrainError::Refused(format!(
            "{} has betas ({}, {}) and eps {} but the step applies ({beta1}, {beta2}) and {eps} to every entry",
            e.name, e.beta1, e.beta2, e.eps
        )));
    }
    if lr_scale.iter().any(|s| *s != 1.0) && !provider.supports_lr_scale() {
        return Err(TrainError::Refused(format!(
            "the recipe asks for a layer-wise learning rate ({:?}) and this provider ({}) has no \
             per-entry learning rate. Folding the scale into one rate or running two updates would \
             change the optimizer under test, so the run is refused before it starts",
            cfg.optimizer.lower_layers,
            provider.describe()
        )));
    }
    let mut host_moments: Vec<Moments> = host_entries
        .iter()
        .map(|e| e.numel().map(Moments::zeros))
        .collect::<Result<_, _>>()?;

    let mut optimizer_step = 0u64;
    let mut log = LossLog::new();
    let mut steps: Vec<StepRecord> = Vec::new();
    let mut channel_log: Vec<(String, Vec<f64>)> = Vec::new();
    let mut start_index = 0u64;
    let mut resume_digest: Option<String> = None;
    if let Some(r) = resume {
        if r.seed != cfg.seed {
            return Err(TrainError::Refused(format!(
                "checkpoint was taken at seed {} but this run's seed is {}; the batch order is a \
                 function of the seed",
                r.seed, cfg.seed
            )));
        }
        if r.epoch != cfg.epoch {
            return Err(TrainError::Refused(format!(
                "checkpoint is at epoch {} but this call was given epoch {}",
                r.epoch, cfg.epoch
            )));
        }
        if r.schedule != cfg.schedule || r.grad_accum != cfg.grad_accum {
            return Err(TrainError::Refused(
                "the checkpoint's schedule or grad_accum differs from this run's: the resumed run \
                 would use rates the original never would have"
                    .into(),
            ));
        }
        if r.host_entries != host_entries {
            return Err(TrainError::Refused("the checkpoint's host parameters describe a different head".into()));
        }
        provider.load_state(&r.dir.join("provider"))?;
        if provider.step_count() != r.optimizer_step {
            return Err(TrainError::Refused(format!(
                "the provider's restored step count is {} but the checkpoint is at step {}",
                provider.step_count(),
                r.optimizer_step
            )));
        }
        if let Some(h) = objective.host() {
            for ((value, _), saved) in h.values_and_grads().into_iter().zip(&r.host_values) {
                if value.len() != saved.len() {
                    return Err(TrainError::Refused("a host tensor changed size since the checkpoint".into()));
                }
                value.copy_from_slice(saved);
            }
        }
        host_moments = r.host_moments;
        optimizer_step = r.optimizer_step;
        log = r.loss_log;
        steps = r.steps;
        channel_log = r.channel_log;
        start_index = r.next_index;
        resume_digest = Some(r.consumed_digest);
    } else if provider.step_count() != 0 {
        return Err(TrainError::Refused(format!(
            "a fresh run was handed a provider that has already taken {} step(s)",
            provider.step_count()
        )));
    }
    let first_step = optimizer_step;
    let total_steps = cfg.schedule.total_steps();
    let max_micro = total_steps
        .checked_mul(u64::from(cfg.grad_accum))
        .ok_or_else(|| TrainError::Refused("total_steps * grad_accum overflows".into()))?;

    let mut clock = RunClock::start(clock, cfg.cap)?;
    let mut consumed = ConsumedPrefix::new();
    let mut source = batches.into_iter();
    let mut pending: Option<O::Batch> = None;

    // ---- a resume rebuilds the data side and proves it is the same data ------------------
    if start_index > 0 {
        let mut skipped = 0u64;
        loop {
            let Some(b) = source.next() else {
                return Err(TrainError::Refused(format!(
                    "resume wanted batch index {start_index} but the source ended first; this is not \
                     the source the checkpoint was taken from"
                )));
            };
            let b = b?;
            if skipped >= max_micro {
                return Err(TrainError::Refused(format!(
                    "skipped {skipped} batches without reaching index {start_index}"
                )));
            }
            if b.index() < start_index {
                b.fold_into(&mut consumed)?;
                skipped += 1;
                continue;
            }
            if b.index() != start_index {
                return Err(TrainError::Refused(format!(
                    "resume wanted batch index {start_index} but the source jumped to {}",
                    b.index()
                )));
            }
            pending = Some(b);
            break;
        }
    }
    if let Some(want) = &resume_digest
        && &consumed.hexdigest() != want
    {
        return Err(TrainError::Refused(format!(
            "the {start_index} batch(es) before the resume point hash to {} but the checkpoint \
             recorded {want}: same seed and epoch, different batches",
            consumed.hexdigest()
        )));
    }

    let mut group = Group {
        losses: Vec::new(),
        channels: Vec::new(),
    };
    let mut micro_batches = 0u64;
    let mut counts = BatchCounts::default();
    let mut last_index: Option<u64> = start_index.checked_sub(1);
    let mut first_in_group = true;
    let mut last_checkpoint: Option<PathBuf> = None;
    let mut eta_projected_s: Option<f64> = None;
    let termination;

    loop {
        if optimizer_step >= total_steps {
            termination = Termination::StepsExhausted;
            break;
        }
        if !group.open() && clock.expired()? {
            termination = Termination::WallClockCap;
            break;
        }
        if micro_batches >= max_micro {
            return Err(TrainError::Refused(format!(
                "consumed {micro_batches} batches without completing the schedule's {total_steps} steps"
            )));
        }
        let batch = match pending.take() {
            Some(b) => b,
            None => match source.next() {
                Some(b) => b?,
                None => {
                    if group.open() {
                        return Err(TrainError::Refused(format!(
                            "the source ended {} micro-batch(es) into an accumulation group of {}: \
                             applying a partial group would take a step at a different effective \
                             batch size than every other step",
                            group.losses.len(),
                            cfg.grad_accum
                        )));
                    }
                    termination = Termination::DataExhausted;
                    break;
                }
            },
        };
        let index = batch.index();
        if let Some(last) = last_index
            && index <= last
        {
            return Err(TrainError::Refused(format!(
                "batch index {index} follows {last}; indices within an epoch are strictly increasing"
            )));
        }
        last_index = Some(index);
        batch.fold_into(&mut consumed)?;

        let plan = objective.plan(&batch)?;
        if plan.sequences.is_empty() {
            return Err(TrainError::Refused(format!("batch {index} planned no sequence")));
        }
        if first_in_group && let Some(h) = objective.host() {
            h.zero_grads();
        }
        let width = provider.hidden_size();
        let mut row_sums = Vec::with_capacity(plan.sequences.len());
        for (i, seq) in plan.sequences.iter().enumerate() {
            let mode = if first_in_group { BankMode::Overwrite } else { BankMode::Add };
            first_in_group = false;
            let job = SequenceJob {
                tokens: &seq.tokens,
                rows: RowTargets {
                    positions: &seq.positions,
                    targets: &seq.targets,
                    scale: plan.row_scale,
                },
                hidden_at: &seq.hidden_at,
            };
            // The objective's own error is kept and preferred over the provider's report of it.
            let mut objective_error: Option<TrainError> = None;
            let result = if seq.hidden_at.is_empty() {
                provider.accumulate(&job, mode, None)
            } else {
                let obj = &mut *objective;
                let err = &mut objective_error;
                let mut ext = |pos: &[u32], hidden: &[f32]| -> Result<Vec<f32>, StepError> {
                    obj.hidden_grad(i, pos, hidden, width).map_err(|e| {
                        let msg = e.to_string();
                        *err = Some(e);
                        StepError::Contract(format!("the objective's hidden-state loss failed: {msg}"))
                    })
                };
                provider.accumulate(&job, mode, Some(&mut ext))
            };
            match (result, objective_error) {
                (Ok(sum), None) => row_sums.push(sum),
                (_, Some(e)) => return Err(e),
                (Err(e), None) => return Err(e.into()),
            }
        }
        let loss = objective.close(&row_sums)?;
        let where_ = format!("epoch {}, batch {index}", cfg.epoch);
        if !loss.total.is_finite() {
            return Err(TrainError::NonFinite(format!(
                "the step returned a non-finite loss {} at {where_}. A run that keeps going trains on \
                 NaN gradients; this step was not applied and no moment moved",
                loss.total
            )));
        }
        if let Some((name, v)) = loss.channels.iter().find(|(_, v)| !v.is_finite()) {
            return Err(TrainError::NonFinite(format!("non-finite {name} loss {v} at {where_}")));
        }
        group.losses.push(loss.total);
        group.channels.push(loss.channels);
        micro_batches += 1;
        counts.add(&plan.counts);
        if group.losses.len() < cfg.grad_accum as usize {
            continue;
        }

        // ---- the group is full: clip, AdamW, log --------------------------------------
        let group_loss = group.losses.iter().sum::<f64>() / group.losses.len() as f64;
        for ch in group.channels.drain(..) {
            for (name, v) in ch {
                match channel_log.iter_mut().find(|(n, _)| *n == name) {
                    Some((_, vs)) => vs.push(v),
                    None => channel_log.push((name, vec![v])),
                }
            }
        }
        group.losses.clear();
        first_in_group = true;

        let lr = cfg.schedule.lr_at(optimizer_step)?;
        let tower_sq = provider.grad_sq_norm()?;
        let host_sq: f64 = objective
            .host_ref()
            .map(|h| {
                h.grads()
                    .iter()
                    .map(|g| g.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>())
                    .sum()
            })
            .unwrap_or(0.0);
        let sq = tower_sq + host_sq;
        let coef = recipe::clip_coefficient(sq, cfg.max_grad_norm)
            .map_err(|e| TrainError::NonFinite(format!("at {where_}: {e}; this step was not applied")))?;
        let hyper = AdamWHyper {
            lr,
            beta1,
            beta2,
            eps,
            grad_scale: coef,
        };
        let t = optimizer_step + 1;
        // Host shapes are checked before the provider moves, so the two halves move together.
        if let Some(h) = objective.host_ref() {
            if h.values().len() != host_rows.len() || host_moments.len() != host_rows.len() {
                return Err(TrainError::Refused("the host parameters no longer match the optimizer table".into()));
            }
            for ((g, v), m) in h.grads().iter().zip(h.values()).zip(&host_moments) {
                if g.len() != v.len() || m.m.len() != v.len() {
                    return Err(TrainError::Refused("a host tensor's gradient or moments do not fit it".into()));
                }
            }
        }
        provider.adamw_step(&hyper, t, &lr_scale, &weight_decay)?;
        if let Some(h) = objective.host() {
            for (((value, grad), m), row) in h.values_and_grads().into_iter().zip(host_moments.iter_mut()).zip(&host_rows) {
                adamw_update(value, grad, m, &hyper, lr * row.lr_scale, row.weight_decay, t)?;
            }
        }
        log.append(LossPoint {
            optimizer_step,
            epoch: cfg.epoch,
            batch_index: index,
            loss: group_loss,
        })?;
        let record = StepRecord {
            optimizer_step,
            lr,
            loss: group_loss,
            grad_norm: sq.sqrt(),
            clip_coefficient: coef,
        };
        steps.push(record);
        optimizer_step += 1;
        if let Some(cb) = hooks.on_progress.as_mut() {
            cb(&Progress {
                record,
                total_steps,
                micro_batches,
                elapsed_s: clock.elapsed_s()?,
            });
        }
        if let Some(policy) = &cfg.checkpoint {
            let done = optimizer_step - first_step;
            if policy.every > 0 && done.is_multiple_of(policy.every) {
                let state = CheckpointBody {
                    epoch: cfg.epoch,
                    next_index: index + 1,
                    optimizer_step,
                    seed: cfg.seed,
                    schedule: &cfg.schedule,
                    grad_accum: cfg.grad_accum,
                    loss_log: &log,
                    steps: &steps,
                    channel_log: &channel_log,
                    consumed: &consumed,
                };
                last_checkpoint = Some(write_checkpoint(&policy.dir, &state, provider, objective, &host_moments, last_checkpoint.as_deref())?);
            }
        }
        if let Some(eta) = &cfg.eta
            && let Some(projected) = eta.projection(optimizer_step, first_step, total_steps, clock.elapsed_s()?)
        {
            eta_projected_s = Some(projected);
            if eta.stops(projected, cfg.cap.cap_s()) {
                termination = Termination::EtaRule;
                break;
            }
        }
    }

    let next_index = last_index.map_or(start_index, |i| i + 1);
    let wall = clock.elapsed_s()?;
    if let Some(policy) = &cfg.checkpoint {
        let already = last_checkpoint
            .as_ref()
            .is_some_and(|p| p.file_name().and_then(|n| n.to_str()) == Some(checkpoint_name(optimizer_step).as_str()));
        if !already {
            let state = CheckpointBody {
                epoch: cfg.epoch,
                next_index,
                optimizer_step,
                seed: cfg.seed,
                schedule: &cfg.schedule,
                grad_accum: cfg.grad_accum,
                loss_log: &log,
                steps: &steps,
                channel_log: &channel_log,
                consumed: &consumed,
            };
            last_checkpoint = Some(write_checkpoint(&policy.dir, &state, provider, objective, &host_moments, last_checkpoint.as_deref())?);
        }
    }
    Ok(TrainResult {
        termination,
        optimizer_steps: optimizer_step,
        micro_batches,
        counts,
        loss_log: log,
        steps,
        channel_log,
        consumed_digest: consumed.hexdigest(),
        consumed_n: consumed.n_folded(),
        next_index,
        wall_clock_s: wall,
        final_checkpoint: last_checkpoint,
        eta_projected_s,
    })
}

/// sha256 over every parameter's f32 bytes (provider entries, then host entries, each with its
/// name and shape), the identity of a trained model for a repeat-run comparison.
pub fn parameters_digest<P: StepProvider>(provider: &mut P, host: Option<&dyn HostParams>) -> Result<String, TrainError> {
    let mut h = Sha256::new();
    let values = provider.read_parameters()?;
    let entries = provider.parameters().to_vec();
    if values.len() != entries.len() {
        return Err(TrainError::Refused("the provider read a different number of tensors than it lists".into()));
    }
    let mut feed = |e: &ParamSpec, v: &[f32]| {
        h.update((e.name.len() as u64).to_le_bytes());
        h.update(e.name.as_bytes());
        h.update((e.shape.len() as u64).to_le_bytes());
        for d in &e.shape {
            h.update((*d as u64).to_le_bytes());
        }
        for x in v {
            h.update(x.to_le_bytes());
        }
    };
    for (e, v) in entries.iter().zip(&values) {
        feed(e, v);
    }
    if let Some(host) = host {
        for (e, v) in host.entries().iter().zip(host.values()) {
            feed(e, v);
        }
    }
    Ok(hex(&h.finalize()))
}

// ---- checkpoints ------------------------------------------------------------------------

struct CheckpointBody<'a> {
    epoch: u64,
    next_index: u64,
    optimizer_step: u64,
    seed: u64,
    schedule: &'a LrSchedule,
    grad_accum: u32,
    loss_log: &'a LossLog,
    steps: &'a [StepRecord],
    channel_log: &'a [(String, Vec<f64>)],
    consumed: &'a ConsumedPrefix,
}

fn checkpoint_name(step: u64) -> String {
    format!("ckpt-{step:08}")
}

fn hex_list(xs: &[f64]) -> Result<Value, PyJsonError> {
    Ok(Value::Array(xs.iter().map(|x| float_hex(*x).map(Value::from)).collect::<Result<_, _>>()?))
}

fn write_synced(path: &Path, bytes: &[u8]) -> Result<(), TrainError> {
    let mut f = fs::File::options().write(true).create_new(true).open(path).map_err(|e| io(path, e))?;
    f.write_all(bytes).map_err(|e| io(path, e))?;
    f.sync_all().map_err(|e| io(path, e))
}

fn sync_dir(dir: &Path) -> Result<(), TrainError> {
    fs::File::open(dir).and_then(|f| f.sync_all()).map_err(|e| io(dir, e))
}

/// Write `<dir>/ckpt-<step>/` through a staging directory and a rename, then remove the
/// previous checkpoint: one resume point, replaced atomically (`CheckpointSink`'s policy).
fn write_checkpoint<P: StepProvider, O: Objective>(
    dir: &Path,
    body: &CheckpointBody<'_>,
    provider: &mut P,
    objective: &O,
    host_moments: &[Moments],
    previous: Option<&Path>,
) -> Result<PathBuf, TrainError> {
    fs::create_dir_all(dir).map_err(|e| io(dir, e))?;
    let name = checkpoint_name(body.optimizer_step);
    let final_dir = dir.join(&name);
    let staging = dir.join(format!(".{name}.partial"));
    if final_dir.exists() || staging.exists() {
        return Err(TrainError::Refused(format!("{} or its staging directory already exists", final_dir.display())));
    }
    fs::create_dir(&staging).map_err(|e| io(&staging, e))?;
    let provider_dir = staging.join("provider");
    fs::create_dir(&provider_dir).map_err(|e| io(&provider_dir, e))?;
    let written = provider.save_state(&provider_dir)?;
    let mut provider_files = Vec::new();
    for path in &written {
        let bytes = fs::read(path).map_err(|e| io(path, e))?;
        let rel = path
            .strip_prefix(&provider_dir)
            .map_err(|_| TrainError::Refused(format!("{} was written outside the provider directory", path.display())))?
            .to_string_lossy()
            .into_owned();
        provider_files.push((rel, Value::from(hex(&Sha256::digest(&bytes)))));
    }

    // Host values and moments: raw little-endian f32, value then m then v per entry.
    let mut host_bytes = Vec::new();
    let host_entries = objective.host_ref().map(|h| h.entries()).unwrap_or_default();
    if let Some(h) = objective.host_ref() {
        for (v, m) in h.values().iter().zip(host_moments) {
            for x in v.iter().chain(&m.m).chain(&m.v) {
                host_bytes.extend_from_slice(&x.to_le_bytes());
            }
        }
    }
    let host_path = staging.join("host.f32");
    write_synced(&host_path, &host_bytes)?;

    let body_json = obj([
        ("format", Value::from(CHECKPOINT_FORMAT)),
        ("epoch", Value::from(body.epoch)),
        ("next_index", Value::from(body.next_index)),
        ("optimizer_step", Value::from(body.optimizer_step)),
        ("seed", Value::from(body.seed)),
        ("schedule", body.schedule.to_json()?),
        ("grad_accum", Value::from(body.grad_accum)),
        ("loss_log", body.loss_log.to_json()?),
        (
            "steps",
            Value::Array(
                body.steps
                    .iter()
                    .map(|s| {
                        obj([
                            ("step", Value::from(s.optimizer_step)),
                            ("lr_hex", Value::from(float_hex(s.lr)?)),
                            ("loss_hex", Value::from(float_hex(s.loss)?)),
                            ("grad_norm_hex", Value::from(float_hex(s.grad_norm)?)),
                            ("clip_hex", Value::from(float_hex(s.clip_coefficient)?)),
                        ])
                    })
                    .collect::<Result<_, _>>()?,
            ),
        ),
        (
            "channel_log",
            obj(body.channel_log
                .iter()
                .map(|(n, vs)| Ok((n.clone(), hex_list(vs)?)))
                .collect::<Result<Vec<_>, PyJsonError>>()?)?,
        ),
        ("consumed_digest", Value::from(body.consumed.hexdigest())),
        ("consumed_n", Value::from(body.consumed.n_folded())),
        ("provider_files", obj(provider_files)?),
        (
            "host_entries",
            Value::Array(
                host_entries
                    .iter()
                    .map(|e| {
                        obj([
                            ("name", Value::from(e.name.clone())),
                            ("shape", Value::from(e.shape.clone())),
                        ])
                    })
                    .collect::<Result<_, PyJsonError>>()?,
            ),
        ),
        ("host_sha256", Value::from(hex(&Sha256::digest(&host_bytes)))),
    ])?;
    write_synced(&staging.join("trainer.json"), dumps(&body_json, CANONICAL)?.as_bytes())?;
    sync_dir(&staging)?;
    fs::rename(&staging, &final_dir).map_err(|e| io(&final_dir, e))?;
    sync_dir(dir)?;
    if let Some(prev) = previous
        && prev != final_dir
    {
        fs::remove_dir_all(prev).map_err(|e| io(prev, e))?;
    }
    Ok(final_dir)
}

fn field<'a>(v: &'a serde_json::Value, key: &str, path: &Path) -> Result<&'a serde_json::Value, TrainError> {
    v.get(key)
        .ok_or_else(|| TrainError::Refused(format!("{}: checkpoint has no {key:?}", path.display())))
}

fn as_u64(v: &serde_json::Value, key: &str, path: &Path) -> Result<u64, TrainError> {
    field(v, key, path)?
        .as_u64()
        .ok_or_else(|| TrainError::Refused(format!("{}: {key:?} is not a non-negative integer", path.display())))
}

fn as_hex(v: &serde_json::Value, path: &Path) -> Result<f64, TrainError> {
    let s = v
        .as_str()
        .ok_or_else(|| TrainError::Refused(format!("{}: expected a float.hex string", path.display())))?;
    Ok(float_fromhex(s)?)
}

impl ResumeState {
    /// Read `ckpt_dir` (one `ckpt-<step>` directory), checking every recorded digest.
    pub fn read(ckpt_dir: &Path) -> Result<Self, TrainError> {
        let body_path = ckpt_dir.join("trainer.json");
        let text = fs::read_to_string(&body_path).map_err(|e| io(&body_path, e))?;
        let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| io(&body_path, e))?;
        let p = body_path.as_path();
        if field(&v, "format", p)?.as_str() != Some(CHECKPOINT_FORMAT) {
            return Err(TrainError::Refused(format!("{}: not a {CHECKPOINT_FORMAT} checkpoint", p.display())));
        }
        let sched = field(&v, "schedule", p)?;
        let schedule = LrSchedule::new(
            field(sched, "peak_lr", p)?.as_f64().ok_or_else(|| TrainError::Refused("peak_lr".into()))?,
            as_u64(sched, "total_steps", p)?,
            as_u64(sched, "warmup_steps", p)?,
            field(sched, "min_lr", p)?.as_f64().ok_or_else(|| TrainError::Refused("min_lr".into()))?,
        )?;
        let mut loss_log = LossLog::new();
        for pt in field(&v, "loss_log", p)?.as_array().ok_or_else(|| TrainError::Refused("loss_log".into()))? {
            loss_log.append(LossPoint {
                optimizer_step: as_u64(pt, "step", p)?,
                epoch: as_u64(pt, "epoch", p)?,
                batch_index: as_u64(pt, "index", p)?,
                loss: as_hex(field(pt, "loss_hex", p)?, p)?,
            })?;
        }
        let mut steps = Vec::new();
        for s in field(&v, "steps", p)?.as_array().ok_or_else(|| TrainError::Refused("steps".into()))? {
            steps.push(StepRecord {
                optimizer_step: as_u64(s, "step", p)?,
                lr: as_hex(field(s, "lr_hex", p)?, p)?,
                loss: as_hex(field(s, "loss_hex", p)?, p)?,
                grad_norm: as_hex(field(s, "grad_norm_hex", p)?, p)?,
                clip_coefficient: as_hex(field(s, "clip_hex", p)?, p)?,
            });
        }
        let mut channel_log = Vec::new();
        for (name, vals) in field(&v, "channel_log", p)?.as_object().ok_or_else(|| TrainError::Refused("channel_log".into()))? {
            let xs = vals
                .as_array()
                .ok_or_else(|| TrainError::Refused("channel_log values".into()))?
                .iter()
                .map(|x| as_hex(x, p))
                .collect::<Result<Vec<_>, _>>()?;
            channel_log.push((name.clone(), xs));
        }
        let provider_dir = ckpt_dir.join("provider");
        for (rel, digest) in field(&v, "provider_files", p)?.as_object().ok_or_else(|| TrainError::Refused("provider_files".into()))? {
            let path = provider_dir.join(rel);
            let bytes = fs::read(&path).map_err(|e| io(&path, e))?;
            if Some(hex(&Sha256::digest(&bytes)).as_str()) != digest.as_str() {
                return Err(TrainError::Refused(format!("{} does not match its recorded sha256", path.display())));
            }
        }
        let mut host_entries = Vec::new();
        for e in field(&v, "host_entries", p)?.as_array().ok_or_else(|| TrainError::Refused("host_entries".into()))? {
            let name = field(e, "name", p)?.as_str().ok_or_else(|| TrainError::Refused("host entry name".into()))?;
            let shape = field(e, "shape", p)?
                .as_array()
                .ok_or_else(|| TrainError::Refused("host entry shape".into()))?
                .iter()
                .map(|d| d.as_u64().and_then(|d| usize::try_from(d).ok()).ok_or_else(|| TrainError::Refused("host dim".into())))
                .collect::<Result<Vec<_>, _>>()?;
            host_entries.push(ParamSpec::new(name, &shape));
        }
        let host_path = ckpt_dir.join("host.f32");
        let host_bytes = fs::read(&host_path).map_err(|e| io(&host_path, e))?;
        if Some(hex(&Sha256::digest(&host_bytes)).as_str()) != field(&v, "host_sha256", p)?.as_str() {
            return Err(TrainError::Refused(format!("{} does not match its recorded sha256", host_path.display())));
        }
        let floats: Vec<f32> = host_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        let mut at = 0usize;
        let mut host_values = Vec::new();
        let mut host_moments = Vec::new();
        for e in &host_entries {
            let n = e.numel()?;
            let take = |at: &mut usize| -> Result<Vec<f32>, TrainError> {
                let end = *at + n;
                let s = floats
                    .get(*at..end)
                    .ok_or_else(|| TrainError::Refused(format!("{} is shorter than its entries", host_path.display())))?;
                *at = end;
                Ok(s.to_vec())
            };
            host_values.push(take(&mut at)?);
            host_moments.push(Moments {
                m: take(&mut at)?,
                v: take(&mut at)?,
            });
        }
        if at != floats.len() {
            return Err(TrainError::Refused(format!("{} is longer than its entries", host_path.display())));
        }
        let grad_accum = u32::try_from(as_u64(&v, "grad_accum", p)?).map_err(|e| TrainError::Refused(e.to_string()))?;
        let state = Self {
            dir: ckpt_dir.to_path_buf(),
            epoch: as_u64(&v, "epoch", p)?,
            next_index: as_u64(&v, "next_index", p)?,
            optimizer_step: as_u64(&v, "optimizer_step", p)?,
            seed: as_u64(&v, "seed", p)?,
            schedule,
            grad_accum,
            loss_log,
            steps,
            channel_log,
            consumed_digest: field(&v, "consumed_digest", p)?
                .as_str()
                .ok_or_else(|| TrainError::Refused("consumed_digest".into()))?
                .to_string(),
            host_entries,
            host_values,
            host_moments,
        };
        if state.loss_log.len() as u64 != state.optimizer_step || state.steps.len() as u64 != state.optimizer_step {
            return Err(TrainError::Refused(format!(
                "{}: {} loss points and {} step records for optimizer step {}",
                p.display(),
                state.loss_log.len(),
                state.steps.len(),
                state.optimizer_step
            )));
        }
        Ok(state)
    }
}

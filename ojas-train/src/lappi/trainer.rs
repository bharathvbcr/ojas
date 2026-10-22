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
//!    recipe ([`super::recipe`]), the host parameters with the same scalars at scale 1.0 (the
//!    span head joins the base group), the same `t`, the same clip coefficient.
//! 5. The cap is polled only between groups (rule 2), so no half-accumulated gradient is
//!    discarded; overshoot is at most one step.
//!
//! **Bookkeeping**, each a port: the loss log and its digest, the consumed-batch digest, strictly
//! increasing batch indices, refusal of a partial group at the end of the data (rule 1), a
//! checkpoint that resumes the exact trajectory (the consumed digest of the skipped prefix must
//! match), and the three termination reasons.

use std::fmt;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use ojas_io::json::{JsonValue as Value, parse_json};
use ojas_io::sha256::{Sha256, sha256};

use super::adamw::{EntryState, Moments, adamw_entry_step};
use super::pyjson::{CANONICAL, PyJsonError, dumps, float_fromhex, float_hex, obj};
use super::recipe::{self, OptimizerRecipe};
use super::run_control::{
    Clock, ConsumedPrefix, LossLog, LossPoint, RunClock, RunControlError, WallClockCap, hex,
};
use super::schedule::{Decay, LrSchedule, ScheduleError};
use super::step::{
    AdamWHyper, BankMode, CheckpointContext, ParamSpec, RowTargets, SequenceJob, StepError,
    StepProvider,
};

/// `run_control.MAX_GRAD_ACCUM`.
pub const MAX_GRAD_ACCUM: u32 = 4096;

/// The checkpoint directory format this module writes and reads. v2 adds each host entry's own
/// AdamW step count (`host_steps`). A v1 checkpoint has no counts and is refused: resuming it
/// would have to guess them.
pub const CHECKPOINT_FORMAT: &str = "qd-train-checkpoint-v2";

#[derive(Debug, Clone, PartialEq)]
pub enum TrainError {
    /// `TrainerContractViolation`: the run was asked to do something its contract forbids.
    Refused(String),
    /// A non-finite loss or gradient norm. The step it belongs to was not applied.
    NonFinite(String),
    Step(StepError),
    Schedule(ScheduleError),
    RunControl(RunControlError),
    Json(PyJsonError),
    Io(String),
}

impl fmt::Display for TrainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Refused(m) => write!(f, "refused: {m}"),
            Self::NonFinite(m) => write!(f, "non-finite: {m}"),
            Self::Step(e) => e.fmt(f),
            Self::Schedule(e) => e.fmt(f),
            Self::RunControl(e) => e.fmt(f),
            Self::Json(e) => e.fmt(f),
            Self::Io(m) => write!(f, "io: {m}"),
        }
    }
}

/// The wrapped errors were `#[error(transparent)]`: `source()` is the inner error's own
/// source, not the inner error.
impl std::error::Error for TrainError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Step(e) => e.source(),
            Self::Schedule(e) => e.source(),
            Self::RunControl(e) => e.source(),
            Self::Json(e) => e.source(),
            Self::Refused(_) | Self::NonFinite(_) | Self::Io(_) => None,
        }
    }
}

impl From<StepError> for TrainError {
    fn from(e: StepError) -> Self {
        Self::Step(e)
    }
}

impl From<ScheduleError> for TrainError {
    fn from(e: ScheduleError) -> Self {
        Self::Schedule(e)
    }
}

impl From<RunControlError> for TrainError {
    fn from(e: RunControlError) -> Self {
        Self::RunControl(e)
    }
}

impl From<PyJsonError> for TrainError {
    fn from(e: PyJsonError) -> Self {
        Self::Json(e)
    }
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
    /// Python computes over the same batch: for a real shard `Batch`,
    /// the shard `Batch::fold_into`; for anything else, one call to [`ConsumedPrefix::fold`]
    /// with the parts `[index (8 bytes BE), bucket (8 bytes BE), tokens, lengths, slot_kind,
    /// target_index, span_target, line_starts]`, each array as its numpy buffer's bytes (C
    /// order, its own dtype) and an absent optional array as no bytes.
    fn fold_into(&self, prefix: &mut ConsumedPrefix) -> Result<(), TrainError>;
}

/// Parameters trained on the host beside the provider's (Lappi's span head).
///
/// A gradient is `None` until a loss reaches the tensor, as torch's `p.grad` is after
/// `zero_grad(set_to_none=True)`. The loop skips a `None` entry's AdamW step entirely
/// ([`super::adamw::adamw_entry_step`]). This matters for the span head: a letter-only batch
/// never reaches it.
pub trait HostParams {
    /// One entry per tensor, in a fixed order.
    fn entries(&self) -> Vec<ParamSpec>;
    /// Reset every gradient to `None`. Called at the start of each optimizer step.
    fn zero_grads(&mut self);
    /// Every gradient, in [`HostParams::entries`] order: `None` where no loss reached the tensor
    /// since [`HostParams::zero_grads`].
    fn grads(&self) -> Vec<Option<&[f32]>>;
    /// Every value, in [`HostParams::entries`] order.
    fn values(&self) -> Vec<&[f32]>;
    /// `(value, gradient)` of every tensor, in [`HostParams::entries`] order.
    fn values_and_grads(&mut self) -> Vec<(&mut [f32], Option<&[f32]>)>;
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
/// `LetterSpanObjective` (the `objective` module).
pub trait Objective {
    type Batch: ConsumedBatch;

    /// The ledger's run kind (`ft`).
    fn run_kind(&self) -> &'static str;

    /// Plan one micro-batch. Called once per batch, before any of its sequences runs.
    fn plan(&mut self, batch: &Self::Batch) -> Result<MicroBatchPlan, TrainError>;

    /// The outside loss's gradient at sequence `seq` of the batch last planned: `hidden` is
    /// `[positions.len(), width]`, row-major; returns the same shape. Accumulates the host
    /// parameters' own gradients as a side effect.
    fn hidden_grad(
        &mut self,
        seq: usize,
        positions: &[u32],
        hidden: &[f32],
        width: usize,
    ) -> Result<Vec<f32>, TrainError>;

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
        matches!(
            self,
            Termination::DataExhausted | Termination::StepsExhausted
        )
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
    fn projection(
        &self,
        optimizer_step: u64,
        first_step: u64,
        total_steps: u64,
        elapsed_s: f64,
    ) -> Option<f64> {
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
    /// The epoch this call trains. [`train`] runs exactly one; [`train_epochs`] sets it itself.
    pub epoch: u64,
    /// How many epochs the whole run has (at least 1; `epoch < epochs`). The schedule's
    /// `total_steps` covers all of them. A checkpoint of a run of more than one epoch records it,
    /// and a resume of another count is refused; a one-epoch checkpoint records nothing.
    pub epochs: u64,
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

/// Where the consumed-batch digest stood when an epoch began: the earlier epochs' digest and
/// how many batches they folded. The epoch's own digest is chained onto it
/// ([`ConsumedPrefix::continuing`]), so the digest is never reset between epochs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpochLink {
    pub digest: String,
    pub n: u64,
}

impl EpochLink {
    fn prefix(&self) -> ConsumedPrefix {
        ConsumedPrefix::continuing(&self.digest, self.n)
    }
}

/// A checkpoint read back: where the run was and what it had done.
#[derive(Debug, Clone, PartialEq)]
pub struct ResumeState {
    pub dir: PathBuf,
    pub epoch: u64,
    /// How many epochs the run this checkpoint belongs to has (1 when it recorded none).
    pub epochs: u64,
    /// `None` in epoch 0; in a later epoch, what the earlier epochs consumed.
    pub epoch_link: Option<EpochLink>,
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
    /// Each host entry's moments and its own AdamW step count.
    pub host_state: Vec<EntryState>,
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
        return Err(TrainError::Refused(format!(
            "max_grad_norm must be positive, got {}",
            cfg.max_grad_norm
        )));
    }
    if cfg.epochs == 0 || cfg.epoch >= cfg.epochs {
        return Err(TrainError::Refused(format!(
            "epoch {} of {} epoch(s): a run has at least one epoch and trains one inside it",
            cfg.epoch, cfg.epochs
        )));
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
    // Frozen entries have no gradient, moments or step: the optimizer table, the per-entry
    // vectors and the provider's bank are all the trainable subsequence.
    let entries: Vec<ParamSpec> = provider
        .parameters()
        .iter()
        .filter(|e| e.trainable)
        .cloned()
        .collect();
    let host_entries = objective
        .host_ref()
        .map(|h| h.entries())
        .unwrap_or_default();
    // Every per-entry number the optimizer uses comes out of this one table (Amendment 2 (ii)).
    let table = recipe::optimizer_table(&entries, &host_entries, &cfg.optimizer)?;
    let (provider_rows, host_rows) = table.split_at(entries.len());
    let lr_scale: Vec<f64> = provider_rows.iter().map(|e| e.lr_scale).collect();
    let weight_decay: Vec<f64> = provider_rows.iter().map(|e| e.weight_decay).collect();
    let host_rows = host_rows.to_vec();
    // One AdamWHyper serves every entry, so eps and the betas must be one value across the table.
    let Some(first) = table.first() else {
        return Err(TrainError::Refused(
            "the model has no parameters to train".into(),
        ));
    };
    let (beta1, beta2, eps) = (first.beta1, first.beta2, first.eps);
    if let Some(e) = table
        .iter()
        .find(|e| e.beta1 != beta1 || e.beta2 != beta2 || e.eps != eps)
    {
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
    let mut host_state: Vec<EntryState> = host_entries
        .iter()
        .map(|e| e.numel().map(EntryState::zeros))
        .collect::<Result<_, _>>()?;

    let mut optimizer_step = 0u64;
    let mut log = LossLog::new();
    let mut steps: Vec<StepRecord> = Vec::new();
    let mut channel_log: Vec<(String, Vec<f64>)> = Vec::new();
    let mut start_index = 0u64;
    let mut resume_digest: Option<String> = None;
    let mut epoch_link: Option<EpochLink> = None;
    let mut resumed_from: Option<PathBuf> = None;
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
                "checkpoint is at epoch {} but this call was given epoch {}; a checkpoint's position \
                 is in its own epoch's batch order, so the only way to the next epoch is \
                 ResumeState::advance_to_epoch",
                r.epoch, cfg.epoch
            )));
        }
        if r.epochs != cfg.epochs {
            return Err(TrainError::Refused(format!(
                "checkpoint belongs to a run of {} epoch(s) but this run has {}; the schedule and the \
                 batch plan of a run are fixed by its epoch count",
                r.epochs, cfg.epochs
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
            return Err(TrainError::Refused(
                "the checkpoint's host parameters describe a different head".into(),
            ));
        }
        provider.load_state_with(
            &r.dir.join("provider"),
            &CheckpointContext {
                optimizer_step: r.optimizer_step,
                seed: r.seed,
                epoch: r.epoch,
                consumed_digest: r.consumed_digest.clone(),
            },
        )?;
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
                    return Err(TrainError::Refused(
                        "a host tensor changed size since the checkpoint".into(),
                    ));
                }
                value.copy_from_slice(saved);
            }
        }
        if let Some(s) = r.host_state.iter().find(|s| s.steps > r.optimizer_step) {
            return Err(TrainError::Refused(format!(
                "a host entry has taken {} AdamW steps in a checkpoint at optimizer step {}",
                s.steps, r.optimizer_step
            )));
        }
        resumed_from = Some(r.dir.clone());
        host_state = r.host_state;
        optimizer_step = r.optimizer_step;
        log = r.loss_log;
        steps = r.steps;
        channel_log = r.channel_log;
        start_index = r.next_index;
        resume_digest = Some(r.consumed_digest);
        epoch_link = r.epoch_link;
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
    let mut consumed = epoch_link
        .as_ref()
        .map_or_else(ConsumedPrefix::new, EpochLink::prefix);
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
            return Err(TrainError::Refused(format!(
                "batch {index} planned no sequence"
            )));
        }
        if first_in_group && let Some(h) = objective.host() {
            h.zero_grads();
        }
        let width = provider.hidden_size();
        let mut row_sums = Vec::with_capacity(plan.sequences.len());
        for (i, seq) in plan.sequences.iter().enumerate() {
            let mode = if first_in_group {
                BankMode::Overwrite
            } else {
                BankMode::Add
            };
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
                        StepError::Contract(format!(
                            "the objective's hidden-state loss failed: {msg}"
                        ))
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
            return Err(TrainError::NonFinite(format!(
                "non-finite {name} loss {v} at {where_}"
            )));
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
        // `clip_grad_norm_` reads only parameters whose grad is not None.
        let host_sq: f64 = objective
            .host_ref()
            .map(|h| {
                h.grads()
                    .into_iter()
                    .flatten()
                    .map(|g| g.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>())
                    .sum()
            })
            .unwrap_or(0.0);
        let sq = tower_sq + host_sq;
        let coef = recipe::clip_coefficient(sq, cfg.max_grad_norm).map_err(|e| {
            TrainError::NonFinite(format!("at {where_}: {e}; this step was not applied"))
        })?;
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
            if h.values().len() != host_rows.len() || host_state.len() != host_rows.len() {
                return Err(TrainError::Refused(
                    "the host parameters no longer match the optimizer table".into(),
                ));
            }
            for ((g, v), s) in h.grads().into_iter().zip(h.values()).zip(&host_state) {
                if g.is_some_and(|g| g.len() != v.len()) || s.moments.m.len() != v.len() {
                    return Err(TrainError::Refused(
                        "a host tensor's gradient or moments do not fit it".into(),
                    ));
                }
            }
        }
        provider.adamw_step(&hyper, t, &lr_scale, &weight_decay)?;
        if let Some(h) = objective.host() {
            // Each host entry steps on its own count, and not at all without a gradient.
            for (((value, grad), s), row) in h
                .values_and_grads()
                .into_iter()
                .zip(host_state.iter_mut())
                .zip(&host_rows)
            {
                adamw_entry_step(value, grad, s, row, lr, coef)?;
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
                    epochs: cfg.epochs,
                    next_index: index + 1,
                    optimizer_step,
                    seed: cfg.seed,
                    schedule: &cfg.schedule,
                    grad_accum: cfg.grad_accum,
                    loss_log: &log,
                    steps: &steps,
                    channel_log: &channel_log,
                    consumed: &consumed,
                    epoch_link: epoch_link.as_ref(),
                };
                last_checkpoint = Some(write_checkpoint(
                    &policy.dir,
                    &state,
                    provider,
                    objective,
                    &host_state,
                    last_checkpoint.as_deref(),
                )?);
            }
        }
        if let Some(eta) = &cfg.eta
            && let Some(projected) =
                eta.projection(optimizer_step, first_step, total_steps, clock.elapsed_s()?)
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
        // A call that took no step (resumed at the cap, at the end of the schedule, or right
        // after an epoch advance that the cap then stopped) has nothing to add to the checkpoint
        // it resumed from, when that one is already this call's step in this call's directory:
        // that checkpoint is its resume point, not a name to refuse.
        if optimizer_step == first_step
            && last_checkpoint.is_none()
            && let Some(from) = &resumed_from
            && from.parent() == Some(policy.dir.as_path())
            && from.file_name().and_then(|n| n.to_str())
                == Some(checkpoint_name(optimizer_step).as_str())
        {
            last_checkpoint = Some(from.clone());
        }
        let already = last_checkpoint.as_ref().is_some_and(|p| {
            p.file_name().and_then(|n| n.to_str()) == Some(checkpoint_name(optimizer_step).as_str())
        });
        if !already {
            let state = CheckpointBody {
                epoch: cfg.epoch,
                epochs: cfg.epochs,
                next_index,
                optimizer_step,
                seed: cfg.seed,
                schedule: &cfg.schedule,
                grad_accum: cfg.grad_accum,
                loss_log: &log,
                steps: &steps,
                channel_log: &channel_log,
                consumed: &consumed,
                epoch_link: epoch_link.as_ref(),
            };
            last_checkpoint = Some(write_checkpoint(
                &policy.dir,
                &state,
                provider,
                objective,
                &host_state,
                last_checkpoint.as_deref(),
            )?);
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

/// One epoch's data as [`train_epochs`] asks for it: the batches in order and how many there are.
/// The count is the caller's plan (the loop cannot know it: a truncated source also ends with
/// `DataExhausted`); it is what proves an epoch finished before the next one starts.
pub struct EpochSource<I> {
    pub batches: I,
    pub count: u64,
}

/// Run every epoch of `cfg.epochs`, one [`train`] call each, from `resume` (or the start) to the
/// last. Epoch `e`'s data is `source(e)`, drawn by the caller from (seed, `e`); between epochs the
/// epoch-end checkpoint is read back and moved on with [`ResumeState::advance_to_epoch`], so the
/// step count, the moments, the loss log and the consumed digest continue and the schedule (whose
/// `total_steps` covers all epochs) is one cosine over the run, never restarted. A resume
/// checkpoint from the middle of epoch `k` continues into epoch `k` and on to the end; one cut at
/// the very end of an epoch moves straight on to the next.
///
/// The wall-clock cap is the run's: each epoch gets what the earlier ones left. The ETA rule is
/// refused for more than one epoch (its projection is over one process's steps and one cap), and
/// so is a run with no checkpoint policy (the epoch-end checkpoint is how the model crosses an
/// epoch). A run that stops early (`WallClockCap`, `EtaRule`) returns that epoch's result, whose
/// checkpoint resumes it. The result is the last epoch's, its micro-batch and position counters
/// and wall clock summed over the epochs this call ran. An epoch-end checkpoint the next epoch
/// has superseded is removed.
pub fn train_epochs<P, O, I, F>(
    provider: &mut P,
    objective: &mut O,
    mut source: F,
    cfg: &TrainConfig,
    clock: &dyn Clock,
    resume: Option<ResumeState>,
    hooks: &mut Hooks<'_>,
) -> Result<TrainResult, TrainError>
where
    P: StepProvider,
    O: Objective,
    I: IntoIterator<Item = Result<O::Batch, TrainError>>,
    F: FnMut(u64) -> Result<EpochSource<I>, TrainError>,
{
    if cfg.epochs == 0 {
        return Err(TrainError::Refused(
            "a run has at least one epoch, got 0".into(),
        ));
    }
    if cfg.epoch != 0 {
        return Err(TrainError::Refused(format!(
            "train_epochs owns the epoch and starts a fresh run at 0 (a resume carries its own), \
             got epoch {}",
            cfg.epoch
        )));
    }
    if cfg.epochs > 1 {
        if cfg.eta.is_some() {
            return Err(TrainError::Refused(
                "the ETA rule projects one process's steps against one cap and is not defined \
                 across epochs; give no rule for a run of more than one epoch"
                    .into(),
            ));
        }
        if cfg.checkpoint.is_none() {
            return Err(TrainError::Refused(
                "a run of more than one epoch needs a checkpoint policy: the epoch-end checkpoint \
                 is how the model and optimizer cross into the next epoch"
                    .into(),
            ));
        }
    }
    let mut run_clock = RunClock::start(clock, cfg.cap)?;
    let mut state = resume;
    let mut epoch = state.as_ref().map_or(0, |r| r.epoch);
    if epoch >= cfg.epochs {
        return Err(TrainError::Refused(format!(
            "the checkpoint is at epoch {epoch} of a run of {} epoch(s)",
            cfg.epochs
        )));
    }
    let mut micro_batches = 0u64;
    let mut counts = BatchCounts::default();
    // The epoch-end checkpoint the epoch now running started from; removed once it has a successor.
    let mut crossed: Option<PathBuf> = None;
    loop {
        let EpochSource { batches, count } = source(epoch)?;
        if let Some(s) = &state
            && s.next_index == count
            && epoch + 1 < cfg.epochs
        {
            // Cut exactly at an epoch's end: nothing of this epoch is left to train.
            let dir = s.dir.clone();
            state = state
                .take()
                .map(|s| s.advance_to_epoch(epoch + 1, count))
                .transpose()?;
            crossed = Some(dir);
            epoch += 1;
            continue;
        }
        let mut epoch_cfg = cfg.clone();
        epoch_cfg.epoch = epoch;
        let elapsed = run_clock.elapsed_s()?;
        if elapsed > 0.0 {
            epoch_cfg.cap = WallClockCap::new(cfg.cap.cap_s() - elapsed)?;
        }
        let mut result = train(
            provider,
            objective,
            batches,
            &epoch_cfg,
            clock,
            state.take(),
            hooks,
        )?;
        micro_batches += result.micro_batches;
        counts.add(&result.counts);
        result.micro_batches = micro_batches;
        result.counts = counts;
        result.wall_clock_s = run_clock.elapsed_s()?;
        if let (Some(old), Some(new), Some(policy)) =
            (&crossed, &result.final_checkpoint, &cfg.checkpoint)
            && old != new
            && old.parent() == Some(policy.dir.as_path())
        {
            fs::remove_dir_all(old).map_err(|e| io(old, e))?;
        }
        if epoch + 1 == cfg.epochs || result.termination != Termination::DataExhausted {
            if result.termination == Termination::StepsExhausted && epoch + 1 < cfg.epochs {
                return Err(TrainError::Refused(format!(
                    "the schedule's {} steps ended inside epoch {epoch} of {}: its total_steps \
                     must cover every epoch",
                    cfg.schedule.total_steps(),
                    cfg.epochs
                )));
            }
            return Ok(result);
        }
        if run_clock.expired()? {
            result.termination = Termination::WallClockCap;
            return Ok(result);
        }
        let Some(ckpt) = result.final_checkpoint.clone() else {
            return Err(TrainError::Refused(format!(
                "epoch {epoch} wrote no checkpoint to carry into epoch {}",
                epoch + 1
            )));
        };
        state = Some(ResumeState::read(&ckpt)?.advance_to_epoch(epoch + 1, count)?);
        crossed = Some(ckpt);
        epoch += 1;
    }
}

/// sha256 over every trainable parameter's f32 bytes (provider entries, then host entries, each with its
/// name and shape), the identity of a trained model for a repeat-run comparison.
pub fn parameters_digest<P: StepProvider>(
    provider: &mut P,
    host: Option<&dyn HostParams>,
) -> Result<String, TrainError> {
    let mut h = Sha256::new();
    let values = provider.read_parameters()?;
    let entries: Vec<ParamSpec> = provider
        .parameters()
        .iter()
        .filter(|e| e.trainable)
        .cloned()
        .collect();
    if values.len() != entries.len() {
        return Err(TrainError::Refused(
            "the provider read a different number of tensors than it lists as trainable".into(),
        ));
    }
    let mut feed = |e: &ParamSpec, v: &[f32]| {
        h.update(&(e.name.len() as u64).to_le_bytes());
        h.update(e.name.as_bytes());
        h.update(&(e.shape.len() as u64).to_le_bytes());
        for d in &e.shape {
            h.update(&(*d as u64).to_le_bytes());
        }
        for x in v {
            h.update(&x.to_le_bytes());
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
    epochs: u64,
    next_index: u64,
    optimizer_step: u64,
    seed: u64,
    schedule: &'a LrSchedule,
    grad_accum: u32,
    loss_log: &'a LossLog,
    steps: &'a [StepRecord],
    channel_log: &'a [(String, Vec<f64>)],
    consumed: &'a ConsumedPrefix,
    epoch_link: Option<&'a EpochLink>,
}

fn checkpoint_name(step: u64) -> String {
    format!("ckpt-{step:08}")
}

fn hex_list(xs: &[f64]) -> Result<Value, PyJsonError> {
    Ok(Value::Array(
        xs.iter()
            .map(|x| float_hex(*x).map(Value::from))
            .collect::<Result<_, _>>()?,
    ))
}

fn write_synced(path: &Path, bytes: &[u8]) -> Result<(), TrainError> {
    let mut f = fs::File::options()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| io(path, e))?;
    f.write_all(bytes).map_err(|e| io(path, e))?;
    f.sync_all().map_err(|e| io(path, e))
}

fn sync_dir(dir: &Path) -> Result<(), TrainError> {
    fs::File::open(dir)
        .and_then(|f| f.sync_all())
        .map_err(|e| io(dir, e))
}

/// Write `<dir>/ckpt-<step>/` through a staging directory and a rename, then remove the
/// previous checkpoint: one resume point, replaced atomically (`CheckpointSink`'s policy).
fn write_checkpoint<P: StepProvider, O: Objective>(
    dir: &Path,
    body: &CheckpointBody<'_>,
    provider: &mut P,
    objective: &O,
    host_state: &[EntryState],
    previous: Option<&Path>,
) -> Result<PathBuf, TrainError> {
    fs::create_dir_all(dir).map_err(|e| io(dir, e))?;
    let name = checkpoint_name(body.optimizer_step);
    let final_dir = dir.join(&name);
    let staging = dir.join(format!(".{name}.partial"));
    if final_dir.exists() || staging.exists() {
        return Err(TrainError::Refused(format!(
            "{} or its staging directory already exists",
            final_dir.display()
        )));
    }
    fs::create_dir(&staging).map_err(|e| io(&staging, e))?;
    let provider_dir = staging.join("provider");
    fs::create_dir(&provider_dir).map_err(|e| io(&provider_dir, e))?;
    let written = provider.save_state_with(
        &provider_dir,
        &CheckpointContext {
            optimizer_step: body.optimizer_step,
            seed: body.seed,
            epoch: body.epoch,
            consumed_digest: body.consumed.hexdigest(),
        },
    )?;
    let mut provider_files = Vec::new();
    for path in &written {
        let bytes = fs::read(path).map_err(|e| io(path, e))?;
        let rel = path
            .strip_prefix(&provider_dir)
            .map_err(|_| {
                TrainError::Refused(format!(
                    "{} was written outside the provider directory",
                    path.display()
                ))
            })?
            .to_string_lossy()
            .into_owned();
        provider_files.push((rel, Value::from(hex(&sha256(&bytes)))));
    }

    // Host values and moments: raw little-endian f32, value then m then v per entry. Each
    // entry's own AdamW step count goes in trainer.json beside its name.
    let mut host_bytes = Vec::new();
    let host_entries = objective
        .host_ref()
        .map(|h| h.entries())
        .unwrap_or_default();
    if host_entries.len() != host_state.len() {
        return Err(TrainError::Refused(format!(
            "{} host entries and {} optimizer states",
            host_entries.len(),
            host_state.len()
        )));
    }
    if let Some(h) = objective.host_ref() {
        for (v, s) in h.values().iter().zip(host_state) {
            for x in v.iter().chain(&s.moments.m).chain(&s.moments.v) {
                host_bytes.extend_from_slice(&x.to_le_bytes());
            }
        }
    }
    let host_path = staging.join("host.f32");
    write_synced(&host_path, &host_bytes)?;

    // Only a checkpoint past epoch 0 has a link; epoch 0's body is byte-for-byte what it was.
    let epoch_link_json = body
        .epoch_link
        .map(|l| {
            Ok::<_, PyJsonError>((
                "epoch_link",
                obj([
                    ("digest", Value::from(l.digest.clone())),
                    ("n", Value::from(l.n)),
                ])?,
            ))
        })
        .transpose()?;
    // Only a run of several epochs records its count: a one-epoch body is byte-for-byte what it
    // was before epochs existed.
    let epochs_json = (body.epochs > 1).then(|| ("epochs", Value::from(body.epochs)));
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
            obj(body
                .channel_log
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
                    .zip(host_state)
                    .map(|(e, s)| {
                        obj([
                            ("name", Value::from(e.name.clone())),
                            ("shape", Value::from(e.shape.clone())),
                            ("adamw_steps", Value::from(s.steps)),
                        ])
                    })
                    .collect::<Result<_, PyJsonError>>()?,
            ),
        ),
        (
            "host_sha256",
            Value::from(hex(&sha256(&host_bytes))),
        ),
    ]
    .into_iter()
    .chain(epoch_link_json)
    .chain(epochs_json))?;
    write_synced(
        &staging.join("trainer.json"),
        dumps(&body_json, CANONICAL)?.as_bytes(),
    )?;
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

fn field<'a>(v: &'a Value, key: &str, path: &Path) -> Result<&'a Value, TrainError> {
    match v.field_opt(key) {
        Ok(Some(found)) => Ok(found),
        Ok(None) | Err(_) => Err(TrainError::Refused(format!(
            "{}: checkpoint has no {key:?}",
            path.display()
        ))),
    }
}

fn as_u64(v: &Value, key: &str, path: &Path) -> Result<u64, TrainError> {
    field(v, key, path)?.as_u64().map_err(|_| {
        TrainError::Refused(format!(
            "{}: {key:?} is not a non-negative integer",
            path.display()
        ))
    })
}

fn as_hex(v: &Value, path: &Path) -> Result<f64, TrainError> {
    let s = v.as_str().map_err(|_| {
        TrainError::Refused(format!("{}: expected a float.hex string", path.display()))
    })?;
    Ok(float_fromhex(s)?)
}

fn as_array<'a>(v: &'a Value, what: &str) -> Result<&'a [Value], TrainError> {
    v.as_array()
        .map_err(|_| TrainError::Refused(what.to_owned()))
}

fn as_object<'a>(v: &'a Value, what: &str) -> Result<&'a [(String, Value)], TrainError> {
    v.as_object()
        .map_err(|_| TrainError::Refused(what.to_owned()))
}

/// The schedule's `"decay"` member: absent is cosine (every checkpoint written before decay
/// kinds existed, and Python's own); anything unrecognised or ill-typed is refused.
fn read_decay(sched: &Value, path: &Path) -> Result<Decay, TrainError> {
    Decay::from_json(sched)
        .map_err(|e| TrainError::Refused(format!("{}: schedule decay: {e}", path.display())))
}

impl ResumeState {
    /// Move a checkpoint taken at the end of epoch `self.epoch` to the start of `epoch`, the
    /// next one: position 0, the consumed digest chained onto the finished epoch's. Everything
    /// else (step count, loss log, host moments, the provider directory) is carried as it is, so
    /// the schedule and the optimizer continue where the last epoch stopped.
    ///
    /// `epoch_batches` is how many batches epoch `self.epoch` has, from the caller's plan; the
    /// loop cannot know it (a truncated source also ends with `DataExhausted`). It is refused
    /// unless the checkpoint consumed exactly that many, and unless `epoch` is the immediate
    /// successor (no skipped epoch, no going back) and the schedule still has steps to take.
    pub fn advance_to_epoch(mut self, epoch: u64, epoch_batches: u64) -> Result<Self, TrainError> {
        if self.epoch.checked_add(1) != Some(epoch) {
            return Err(TrainError::Refused(format!(
                "a checkpoint at epoch {} can only advance to epoch {}, not {epoch}",
                self.epoch,
                self.epoch.saturating_add(1)
            )));
        }
        if epoch_batches == 0 {
            return Err(TrainError::Refused(
                "an epoch has at least one batch; advancing past a zero-length epoch proves nothing"
                    .into(),
            ));
        }
        if self.next_index != epoch_batches {
            return Err(TrainError::Refused(format!(
                "epoch {} is not finished: the checkpoint consumed {} of its {epoch_batches} batches",
                self.epoch, self.next_index
            )));
        }
        let total = self.schedule.total_steps();
        if self.optimizer_step >= total {
            return Err(TrainError::Refused(format!(
                "the schedule has no steps left ({} of {total} taken); epoch {epoch} would train \
                 nothing",
                self.optimizer_step
            )));
        }
        let link = EpochLink {
            digest: self.consumed_digest.clone(),
            n: self.epoch_link.as_ref().map_or(0, |l| l.n) + self.next_index,
        };
        if epoch >= self.epochs {
            return Err(TrainError::Refused(format!(
                "the run has {} epoch(s); epoch {epoch} is past its last",
                self.epochs
            )));
        }
        self.consumed_digest = link.prefix().hexdigest();
        self.epoch_link = Some(link);
        self.epoch = epoch;
        self.next_index = 0;
        Ok(self)
    }

    /// Read `ckpt_dir` (one `ckpt-<step>` directory), checking every recorded digest.
    pub fn read(ckpt_dir: &Path) -> Result<Self, TrainError> {
        let body_path = ckpt_dir.join("trainer.json");
        let text = fs::read_to_string(&body_path).map_err(|e| io(&body_path, e))?;
        let v = parse_json(&text).map_err(|e| io(&body_path, e))?;
        let p = body_path.as_path();
        if field(&v, "format", p)?.as_str().ok() != Some(CHECKPOINT_FORMAT) {
            return Err(TrainError::Refused(format!(
                "{}: not a {CHECKPOINT_FORMAT} checkpoint",
                p.display()
            )));
        }
        let sched = field(&v, "schedule", p)?;
        let schedule = LrSchedule::with_decay(
            field(sched, "peak_lr", p)?
                .as_f64()
                .map_err(|_| TrainError::Refused("peak_lr".into()))?,
            as_u64(sched, "total_steps", p)?,
            as_u64(sched, "warmup_steps", p)?,
            field(sched, "min_lr", p)?
                .as_f64()
                .map_err(|_| TrainError::Refused("min_lr".into()))?,
            read_decay(sched, p)?,
        )?;
        let mut loss_log = LossLog::new();
        for pt in as_array(field(&v, "loss_log", p)?, "loss_log")? {
            loss_log.append(LossPoint {
                optimizer_step: as_u64(pt, "step", p)?,
                epoch: as_u64(pt, "epoch", p)?,
                batch_index: as_u64(pt, "index", p)?,
                loss: as_hex(field(pt, "loss_hex", p)?, p)?,
            })?;
        }
        let mut steps = Vec::new();
        for s in as_array(field(&v, "steps", p)?, "steps")? {
            steps.push(StepRecord {
                optimizer_step: as_u64(s, "step", p)?,
                lr: as_hex(field(s, "lr_hex", p)?, p)?,
                loss: as_hex(field(s, "loss_hex", p)?, p)?,
                grad_norm: as_hex(field(s, "grad_norm_hex", p)?, p)?,
                clip_coefficient: as_hex(field(s, "clip_hex", p)?, p)?,
            });
        }
        let mut channel_log = Vec::new();
        for (name, vals) in as_object(field(&v, "channel_log", p)?, "channel_log")? {
            let xs = as_array(vals, "channel_log values")?
                .iter()
                .map(|x| as_hex(x, p))
                .collect::<Result<Vec<_>, _>>()?;
            channel_log.push((name.clone(), xs));
        }
        let provider_dir = ckpt_dir.join("provider");
        for (rel, digest) in as_object(field(&v, "provider_files", p)?, "provider_files")? {
            let path = provider_dir.join(rel);
            let bytes = fs::read(&path).map_err(|e| io(&path, e))?;
            if Some(hex(&sha256(&bytes)).as_str()) != digest.as_str().ok() {
                return Err(TrainError::Refused(format!(
                    "{} does not match its recorded sha256",
                    path.display()
                )));
            }
        }
        let mut host_entries = Vec::new();
        let mut host_steps = Vec::new();
        for e in as_array(field(&v, "host_entries", p)?, "host_entries")? {
            let name = field(e, "name", p)?
                .as_str()
                .map_err(|_| TrainError::Refused("host entry name".into()))?;
            let shape = as_array(field(e, "shape", p)?, "host entry shape")?
                .iter()
                .map(|d| {
                    d.as_u64()
                        .ok()
                        .and_then(|d| usize::try_from(d).ok())
                        .ok_or_else(|| TrainError::Refused("host dim".into()))
                })
                .collect::<Result<Vec<_>, _>>()?;
            host_steps.push(as_u64(e, "adamw_steps", p)?);
            host_entries.push(ParamSpec::new(name, &shape));
        }
        let host_path = ckpt_dir.join("host.f32");
        let host_bytes = fs::read(&host_path).map_err(|e| io(&host_path, e))?;
        if Some(hex(&sha256(&host_bytes)).as_str()) != field(&v, "host_sha256", p)?.as_str().ok()
        {
            return Err(TrainError::Refused(format!(
                "{} does not match its recorded sha256",
                host_path.display()
            )));
        }
        let floats: Vec<f32> = host_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        let mut at = 0usize;
        let mut host_values = Vec::new();
        let mut host_state = Vec::new();
        for (e, &steps) in host_entries.iter().zip(&host_steps) {
            let n = e.numel()?;
            let take = |at: &mut usize| -> Result<Vec<f32>, TrainError> {
                let end = *at + n;
                let s = floats.get(*at..end).ok_or_else(|| {
                    TrainError::Refused(format!(
                        "{} is shorter than its entries",
                        host_path.display()
                    ))
                })?;
                *at = end;
                Ok(s.to_vec())
            };
            host_values.push(take(&mut at)?);
            host_state.push(EntryState {
                moments: Moments {
                    m: take(&mut at)?,
                    v: take(&mut at)?,
                },
                steps,
            });
        }
        if at != floats.len() {
            return Err(TrainError::Refused(format!(
                "{} is longer than its entries",
                host_path.display()
            )));
        }
        let grad_accum = u32::try_from(as_u64(&v, "grad_accum", p)?)
            .map_err(|e| TrainError::Refused(e.to_string()))?;
        let epoch = as_u64(&v, "epoch", p)?;
        let epoch_link = match v.field_opt("epoch_link").map_err(|e| io(p, e))? {
            None => None,
            Some(l) => {
                let digest = field(l, "digest", p)?
                    .as_str()
                    .ok()
                    .filter(|d| d.len() == 64 && d.bytes().all(|b| b.is_ascii_hexdigit()))
                    .ok_or_else(|| {
                        TrainError::Refused(format!(
                            "{}: epoch_link.digest is not a sha256 hex digest",
                            p.display()
                        ))
                    })?
                    .to_string();
                Some(EpochLink {
                    digest,
                    n: as_u64(l, "n", p)?,
                })
            }
        };
        // advance_to_epoch is the only way past epoch 0 and always sets the link; epoch 0 never
        // has one. The other two shapes are bodies no run wrote.
        if (epoch == 0) != epoch_link.is_none() {
            return Err(TrainError::Refused(format!(
                "{}: epoch {epoch} with {} epoch_link; a checkpoint past epoch 0 carries the link \
                 to the epochs before it and an epoch-0 checkpoint carries none",
                p.display(),
                if epoch_link.is_some() { "an" } else { "no" }
            )));
        }
        let epochs = match v.field_opt("epochs").map_err(|e| io(p, e))? {
            None => 1,
            Some(e) => e.as_u64().ok().filter(|&n| n > 1).ok_or_else(|| {
                TrainError::Refused(format!(
                    "{}: \"epochs\" is {e:?}; a checkpoint records it only for a run of more than one",
                    p.display()
                ))
            })?,
        };
        if epoch >= epochs {
            return Err(TrainError::Refused(format!(
                "{}: epoch {epoch} of a run of {epochs} epoch(s)",
                p.display()
            )));
        }
        let state = Self {
            dir: ckpt_dir.to_path_buf(),
            epoch,
            epochs,
            epoch_link,
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
                .map_err(|_| TrainError::Refused("consumed_digest".into()))?
                .to_string(),
            host_entries,
            host_values,
            host_state,
        };
        if state.loss_log.len() as u64 != state.optimizer_step
            || state.steps.len() as u64 != state.optimizer_step
        {
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

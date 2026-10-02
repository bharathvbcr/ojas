//! The step-provider trait: what a model must offer a training loop that is not its own.
//!
//! **Model-generic on purpose.** The trainer starts in Lappi's `crates/qd-train` and moves to an
//! `ojas-train` crate later (the user's decision, 2026-10-01, `HANDOFF/ojas-training-2026-10-01.md`).
//! So nothing in this file names a Lappi type: no batch, no shard, no span head, no ledger.
//! It speaks only of token sequences, parameter-table entries, a gradient bank, gradients at
//! hidden states and AdamW. Its first implementor is canonical tessl's Qwen3.5 step (through
//! ojas's `ojas-qwen35`, adapted in `crates/qd-train-metal`); its second will be ojas's Tape
//! model. [`crate::mock::ToyProvider`] implements it on the CPU so the loop is tested end to end
//! without a GPU.
//!
//! **The contract, as the loop relies on it:**
//!
//! 1. A step is a sequence of [`StepProvider::accumulate`] calls, one per sequence. The first
//!    call of a step passes [`BankMode::Overwrite`] and every later one [`BankMode::Add`], so the
//!    bank holds the SUM of the sequences' gradients (tessl's `train_step_into(.., accumulate)`).
//! 2. Each sequence's own loss is a cross-entropy over the full vocabulary at chosen positions
//!    ([`RowTargets`]): the provider returns the *unscaled* sum of those cross-entropies and adds
//!    `scale` times its gradient to the bank, so a batch mean over `N` rows spread across many
//!    calls is `scale = 1 / N` in each (tessl's `Supervise::Rows`).
//! 3. A loss outside the model reads the final hidden state (after the final norm) at
//!    [`SequenceJob::hidden_at`] and returns its gradient there through the callback. The
//!    positions are **distinct** and below the sequence length: a caller whose loss reads one
//!    position twice pre-sums the two gradient rows before handing them back. The callback's
//!    gradient is added to the provider's own before the backward, and is NOT scaled by the
//!    provider: whatever weight the outside loss carries, the callback has already applied.
//! 4. [`StepProvider::grad_sq_norm`] is the sum of squares of the whole bank (f64), the square of
//!    the global L2 norm `torch.nn.utils.clip_grad_norm_` takes over the model's parameters.
//! 5. [`StepProvider::adamw_step`] is `torch.optim.AdamW`'s single-tensor update with
//!    `amsgrad`/`maximize` off, every gradient first multiplied by `hyper.grad_scale` (the clip
//!    coefficient). Learning rate and decoupled weight decay are per parameter-table entry:
//!    entry `i` uses `lr = hyper.lr * lr_scale[i]` and `weight_decay[i]`. Both vectors come
//!    from the caller's recipe -- a provider never substitutes defaults of its own. `step` is the
//!    1-based count this update makes; a provider whose own count disagrees refuses, because a
//!    bias correction at the wrong `t` is a silently different optimizer.
//! 6. A provider that cannot honour a per-entry learning rate says so through
//!    [`StepProvider::supports_lr_scale`] and refuses any `lr_scale` entry other than 1.0 in
//!    `adamw_step`. It never folds the scale into a single rate or runs two updates.
//! 7. Every method fails closed: a refusal is an `Err`, and nothing moves after an `Err` from
//!    `adamw_step` is returned before its first write.

use std::path::{Path, PathBuf};

/// One parameter-table entry: its name and its shape, as the model's reference implementation
/// (transformers, for Qwen3.5) names and shapes it. Values read through
/// [`StepProvider::read_parameters`] are row-major in exactly this shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamSpec {
    pub name: String,
    pub shape: Vec<usize>,
}

impl ParamSpec {
    pub fn new(name: impl Into<String>, shape: &[usize]) -> Self {
        Self {
            name: name.into(),
            shape: shape.to_vec(),
        }
    }

    /// Element count, refusing a shape whose product overflows.
    pub fn numel(&self) -> Result<usize, StepError> {
        self.shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| StepError::Contract(format!("{}: shape {:?} overflows usize", self.name, self.shape)))
    }
}

/// Whether a sequence's gradients replace the bank or add to it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BankMode {
    Overwrite,
    Add,
}

/// The positions a sequence's own cross-entropy scores, the token each predicts, and the
/// gradient's weight. `positions[i]` predicts `targets[i]`.
#[derive(Debug, Clone, Copy)]
pub struct RowTargets<'a> {
    pub positions: &'a [u32],
    pub targets: &'a [u32],
    pub scale: f32,
}

/// One sequence's work.
#[derive(Debug, Clone, Copy)]
pub struct SequenceJob<'a> {
    /// The unpadded token sequence, positions from 0.
    pub tokens: &'a [u32],
    pub rows: RowTargets<'a>,
    /// Distinct positions whose final hidden state a loss outside the model reads.
    pub hidden_at: &'a [u32],
}

impl SequenceJob<'_> {
    /// The checks every provider needs before it runs anything: lengths agree, positions are in
    /// range, the outside loss's positions are distinct, the scale is finite and non-negative.
    pub fn validate(&self, vocab: usize) -> Result<(), StepError> {
        let t = self.tokens.len();
        if t == 0 {
            return Err(StepError::Contract("a sequence needs at least one token".into()));
        }
        if let Some(bad) = self.tokens.iter().find(|&&id| id as usize >= vocab) {
            return Err(StepError::Contract(format!("token {bad} >= vocabulary {vocab}")));
        }
        let rows = &self.rows;
        if rows.positions.len() != rows.targets.len() {
            return Err(StepError::Contract(format!(
                "{} row positions for {} targets",
                rows.positions.len(),
                rows.targets.len()
            )));
        }
        if !(rows.scale.is_finite() && rows.scale >= 0.0) {
            return Err(StepError::Contract(format!("row scale {} must be finite and >= 0", rows.scale)));
        }
        if let Some(bad) = rows.positions.iter().find(|&&p| p as usize >= t) {
            return Err(StepError::Contract(format!("row position {bad} >= {t} tokens")));
        }
        if let Some(bad) = rows.targets.iter().find(|&&id| id as usize >= vocab) {
            return Err(StepError::Contract(format!("row target {bad} >= vocabulary {vocab}")));
        }
        let mut seen = self.hidden_at.to_vec();
        seen.sort_unstable();
        if let Some(w) = seen.windows(2).find(|w| w[0] == w[1]) {
            return Err(StepError::Contract(format!(
                "hidden position {} is requested twice; positions are distinct and the caller pre-sums",
                w[0]
            )));
        }
        if let Some(bad) = self.hidden_at.iter().find(|&&p| p as usize >= t) {
            return Err(StepError::Contract(format!("hidden position {bad} >= {t} tokens")));
        }
        Ok(())
    }
}

/// AdamW's scalars other than the per-entry vectors, as torch names them, plus the step's
/// gradient scale (`clip_grad_norm_`'s coefficient, or 1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdamWHyper {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub grad_scale: f64,
}

impl AdamWHyper {
    pub fn validate(&self) -> Result<(), StepError> {
        let ok = self.lr.is_finite()
            && self.lr > 0.0
            && (0.0..1.0).contains(&self.beta1)
            && (0.0..1.0).contains(&self.beta2)
            && self.eps.is_finite()
            && self.eps > 0.0
            && self.grad_scale.is_finite()
            && self.grad_scale > 0.0;
        if ok {
            Ok(())
        } else {
            Err(StepError::Contract(format!("AdamW hyperparameters out of range: {self:?}")))
        }
    }
}

/// The gradient of a loss outside the model at the hidden states it read: called with the
/// positions and a row-major `[positions.len(), hidden]` block of final hidden states, it
/// returns a block of the same shape.
pub type HiddenGrad<'a> = dyn FnMut(&[u32], &[f32]) -> Result<Vec<f32>, StepError> + 'a;

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum StepError {
    /// The caller asked for something the contract above does not allow.
    #[error("refused: {0}")]
    Contract(String),
    /// The provider cannot do this at all (for example a per-entry learning rate).
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// The provider's backend failed.
    #[error("backend: {0}")]
    Backend(String),
    #[error("io: {0}")]
    Io(String),
}

/// A model a training loop can drive one sequence at a time. See the module docs for the
/// contract every method is held to.
pub trait StepProvider {
    /// Every trainable parameter, in the provider's fixed order. `lr_scale` and
    /// `weight_decay` vectors are indexed by this order.
    fn parameters(&self) -> &[ParamSpec];

    /// Width of the hidden state [`HiddenGrad`] blocks carry.
    fn hidden_size(&self) -> usize;

    /// Vocabulary size: tokens and row targets are below it.
    fn vocab_size(&self) -> usize;

    /// Forward and backward one sequence into the bank. Returns the unscaled sum of the row
    /// cross-entropies (0.0 for no rows). `external` is called once, with `job.hidden_at`,
    /// when that is non-empty, and must be `Some` then.
    fn accumulate(
        &mut self,
        job: &SequenceJob<'_>,
        mode: BankMode,
        external: Option<&mut HiddenGrad<'_>>,
    ) -> Result<f64, StepError>;

    /// Sum of squares of every gradient in the bank.
    fn grad_sq_norm(&mut self) -> Result<f64, StepError>;

    /// Whether `adamw_step` honours an `lr_scale` other than 1.0.
    fn supports_lr_scale(&self) -> bool;

    /// One AdamW update of every parameter from the bank; `step` is the 1-based count it makes.
    fn adamw_step(&mut self, hyper: &AdamWHyper, step: u64, lr_scale: &[f64], weight_decay: &[f64]) -> Result<(), StepError>;

    /// Optimizer steps taken so far (torch's `state["step"]`).
    fn step_count(&self) -> u64;

    /// Every parameter's current value, f32, row-major in its [`ParamSpec::shape`], in
    /// [`StepProvider::parameters`] order.
    fn read_parameters(&mut self) -> Result<Vec<Vec<f32>>, StepError>;

    /// A short description of what runs (backend, kernels, operand precision), for the ledger.
    fn describe(&self) -> String;

    /// Write the parameters, both moments and the step count under `dir` (which exists and
    /// is empty); returns the files written. Enough for [`StepProvider::load_state`] to resume
    /// the exact trajectory.
    fn save_state(&mut self, dir: &Path) -> Result<Vec<PathBuf>, StepError>;

    /// Restore what [`StepProvider::save_state`] wrote.
    fn load_state(&mut self, dir: &Path) -> Result<(), StepError>;
}

/// `lr_scale` and `weight_decay` fit `n` entries and hold finite values in range. A shared
/// check so every provider refuses the same malformed vectors with the same words.
pub fn check_per_entry(n: usize, lr_scale: &[f64], weight_decay: &[f64]) -> Result<(), StepError> {
    if lr_scale.len() != n || weight_decay.len() != n {
        return Err(StepError::Contract(format!(
            "{} lr scales and {} weight decays for {n} parameters",
            lr_scale.len(),
            weight_decay.len()
        )));
    }
    if let Some((i, s)) = lr_scale.iter().enumerate().find(|(_, s)| !(s.is_finite() && **s > 0.0)) {
        return Err(StepError::Contract(format!("entry {i}: lr scale {s} must be finite and > 0")));
    }
    if let Some((i, w)) = weight_decay.iter().enumerate().find(|(_, w)| !(w.is_finite() && **w >= 0.0)) {
        return Err(StepError::Contract(format!("entry {i}: weight decay {w} must be finite and >= 0")));
    }
    Ok(())
}

/// The refusal a provider without per-entry learning rates makes (contract item 6).
pub fn refuse_lr_scale(lr_scale: &[f64], who: &str) -> Result<(), StepError> {
    match lr_scale.iter().enumerate().find(|(_, s)| **s != 1.0) {
        None => Ok(()),
        Some((i, s)) => Err(StepError::Unsupported(format!(
            "{who} has no per-entry learning rate, and entry {i} asks for {s}x. Folding it into one \
             rate or running two updates would change the optimizer under test, so it is refused"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job<'a>(tokens: &'a [u32], pos: &'a [u32], tgt: &'a [u32], hidden: &'a [u32]) -> SequenceJob<'a> {
        SequenceJob {
            tokens,
            rows: RowTargets {
                positions: pos,
                targets: tgt,
                scale: 0.5,
            },
            hidden_at: hidden,
        }
    }

    #[test]
    fn a_job_is_checked_before_anything_runs() {
        let toks = [1u32, 2, 3];
        assert!(job(&toks, &[1], &[3], &[0, 2]).validate(8).is_ok());
        assert!(job(&[], &[], &[], &[]).validate(8).is_err(), "empty sequence");
        assert!(job(&toks, &[3], &[3], &[]).validate(8).is_err(), "row past the end");
        assert!(job(&toks, &[1], &[9], &[]).validate(8).is_err(), "target past the vocabulary");
        assert!(job(&toks, &[1, 2], &[3], &[]).validate(8).is_err(), "lengths differ");
        assert!(job(&toks, &[], &[], &[2, 2]).validate(8).is_err(), "a repeated hidden position");
        assert!(job(&toks, &[], &[], &[3]).validate(8).is_err(), "hidden past the end");
        assert!(job(&[1, 9], &[], &[], &[]).validate(8).is_err(), "token past the vocabulary");
        let mut nan = job(&toks, &[1], &[3], &[]);
        nan.rows.scale = f32::NAN;
        assert!(nan.validate(8).is_err());
    }

    #[test]
    fn per_entry_vectors_are_checked_and_an_unsupported_scale_is_refused() {
        assert!(check_per_entry(2, &[1.0, 0.1], &[0.01, 0.01]).is_ok());
        assert!(check_per_entry(2, &[1.0], &[0.01, 0.01]).is_err());
        assert!(check_per_entry(1, &[0.0], &[0.01]).is_err());
        assert!(check_per_entry(1, &[1.0], &[-0.01]).is_err());
        assert!(refuse_lr_scale(&[1.0, 1.0], "x").is_ok());
        assert!(matches!(refuse_lr_scale(&[1.0, 0.1], "x"), Err(StepError::Unsupported(_))));
    }
}

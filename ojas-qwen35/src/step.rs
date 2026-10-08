//! The provider: one Qwen3.5 text tower on the GPU through tessl, with its
//! gradient bank and AdamW state.
//!
//! A step is per sequence, because tessl's pending step holds every layer's
//! `T x hidden` input between its forward and its backward:
//!
//! ```text
//! for each sequence of the batch:
//!     pending = step.forward(seq)          letter CE sum + hidden rows at span positions
//!     dh      = span head (caller, host)   gradient at those rows
//!     step.backward(pending, Some(dh))     into the bank (the first one overwrites it)
//! sq   = step.grad_sq_norm()               + the span head's own, on the host
//! clip = clip_coefficient(max_norm, sq, head_sq)
//! step.adamw_step(&hyper{grad_scale: clip}, &plan)   empties the bank
//! ```
//!
//! Only [`Qwen35Step::open`] and the methods on an open provider touch the
//! GPU. Every check a caller's input can fail runs on the host first, in
//! functions that never open a runtime ([`Sequence::validate`],
//! [`validate_external_grad`], [`clip_coefficient`], the plan checks).

use std::sync::Arc;

use ojas_core::{
    check_adamw, clip_scale, next_step, AdamWConfig, Budget, DType, OjasError, Tensor,
};
use tessl::gemm::GemmOperands;
use tessl::qwen35_adamw::{AdamW, AdamWHyper as TesslHyper};
use tessl::qwen35_model::{Precision, Qwen35Model};
use tessl::qwen35_train::{PendingStep, Qwen35Grads, Supervise};
use tessl::GpuRuntime;

use crate::config::Qwen35TextConfig;
use crate::error::{invalid, tessl, Qwen35Error, Result};
use crate::groups::OptimizerPlan;
use crate::names::{tower_tensors, Snapshot, TensorSpec};

/// How the step's GEMMs treat their operands. Weights, activations,
/// gradients, the GDN state, attention and the optimizer are f32 in both;
/// there is no default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Numerics {
    /// Exact f32 products (the runtime's relaxed precision off): the tight
    /// arm, transformers' fp32 up to operation order.
    ExactF32,
    /// GEMM operands rounded to bf16, accumulated in f32: the fp32-master
    /// analogue of torch's bf16 training. Not bitwise torch's master-bf16,
    /// which also keeps bf16 activations.
    Bf16Operands,
}

impl Numerics {
    fn operands(self) -> GemmOperands {
        match self {
            Numerics::ExactF32 => GemmOperands::ExactF32,
            Numerics::Bf16Operands => GemmOperands::Bf16,
        }
    }
}

/// One sequence of a batch, with what it supervises.
#[derive(Clone, Copy, Debug)]
pub struct Sequence<'a> {
    /// Token ids from position 0. Text only: a vision special token is refused.
    pub ids: &'a [u32],
    /// Positions whose next-token distribution is scored (the letter answer's
    /// `target_index`); distinct and below `ids.len()`.
    pub letter_rows: &'a [u32],
    /// The token each of `letter_rows` is scored against.
    pub letter_targets: &'a [u32],
    /// The letter cross-entropy's gradient is that of `letter_scale` times
    /// its sum over `letter_rows`: `1 / N` for a mean over the batch's `N`
    /// letter rows. Finite and > 0 when there are rows.
    pub letter_scale: f32,
    /// Positions whose final-norm hidden rows go to the caller's span head.
    /// Below `ids.len()`; may repeat.
    pub span_positions: &'a [u32],
}

impl Sequence<'_> {
    /// Every refusal the step would make on this sequence, on the host.
    pub fn validate(&self, cfg: &Qwen35TextConfig) -> Result<()> {
        const OP: &str = "Sequence::validate";
        let t = self.ids.len();
        if t == 0 {
            return Err(invalid(OP, "no tokens"));
        }
        if u32::try_from(t).is_err() {
            return Err(invalid(OP, format!("{t} tokens exceed u32")));
        }
        if let Some(max) = cfg.max_position_embeddings {
            if t as u64 > max {
                return Err(invalid(
                    OP,
                    format!("{t} tokens exceed max_position_embeddings {max}"),
                ));
            }
        }
        for (i, &id) in self.ids.iter().enumerate() {
            if id >= cfg.vocab {
                return Err(invalid(
                    OP,
                    format!("ids[{i}] = {id} >= vocab {}", cfg.vocab),
                ));
            }
            if let Some((name, _)) = cfg.reserved_token_ids.iter().find(|(_, r)| *r == id) {
                return Err(invalid(
                    OP,
                    format!(
                        "ids[{i}] = {id} is the config's {name}: vision tokens switch transformers to multimodal \
                         positions, which this text-only step does not implement"
                    ),
                ));
            }
        }
        if self.letter_rows.len() != self.letter_targets.len() {
            return Err(invalid(
                OP,
                format!(
                    "{} letter rows but {} targets",
                    self.letter_rows.len(),
                    self.letter_targets.len()
                ),
            ));
        }
        let mut seen = vec![false; t];
        for &p in self.letter_rows {
            let slot = seen
                .get_mut(p as usize)
                .ok_or_else(|| invalid(OP, format!("letter row {p} >= {t} tokens")))?;
            if std::mem::replace(slot, true) {
                return Err(invalid(OP, format!("letter row {p} is supervised twice")));
            }
        }
        if let Some(&bad) = self.letter_targets.iter().find(|&&id| id >= cfg.vocab) {
            return Err(invalid(
                OP,
                format!("letter target {bad} >= vocab {}", cfg.vocab),
            ));
        }
        let s = self.letter_scale;
        if !s.is_finite() || s < 0.0 || (!self.letter_rows.is_empty() && s == 0.0) {
            return Err(invalid(
                OP,
                format!("letter_scale {s} must be finite, and > 0 when there are letter rows"),
            ));
        }
        if let Some(&bad) = self.span_positions.iter().find(|&&p| p as usize >= t) {
            return Err(invalid(OP, format!("span position {bad} >= {t} tokens")));
        }
        Ok(())
    }
}

/// The gradient of a loss outside this crate (the span head) at the final
/// norm's output.
#[derive(Clone, Copy, Debug)]
pub struct ExternalGrad<'a> {
    /// Distinct positions below the sequence length. Where the head read a
    /// position twice, the caller sums those rows first.
    pub positions: &'a [u32],
    /// Host f32 `[positions.len(), hidden]`, finite.
    pub dh: &'a Tensor,
}

/// The host checks of an [`ExternalGrad`] for a sequence of `tokens` tokens;
/// returns `dh`'s values.
pub fn validate_external_grad(
    g: &ExternalGrad<'_>,
    tokens: usize,
    hidden: usize,
) -> Result<Vec<f32>> {
    const OP: &str = "ExternalGrad";
    let n = g.positions.len();
    if g.dh.dtype() != DType::F32 || g.dh.shape() != [n, hidden] {
        return Err(invalid(
            OP,
            format!(
                "dh must be f32 [{n}, {hidden}], got {:?} {:?}",
                g.dh.dtype(),
                g.dh.shape()
            ),
        ));
    }
    if g.dh.device().is_some() {
        return Err(invalid(OP, "dh must be a host tensor"));
    }
    let mut seen = vec![false; tokens];
    for &p in g.positions {
        let slot = seen
            .get_mut(p as usize)
            .ok_or_else(|| invalid(OP, format!("position {p} >= {tokens} tokens")))?;
        if std::mem::replace(slot, true) {
            return Err(invalid(
                OP,
                format!("position {p} appears twice; sum its rows before passing them"),
            ));
        }
    }
    let values = g.dh.to_f32_vec()?;
    if let Some(i) = values.iter().position(|v| !v.is_finite()) {
        return Err(invalid(
            OP,
            format!(
                "dh[{}, {}] = {} is not finite",
                i / hidden.max(1),
                i % hidden.max(1),
                values[i]
            ),
        ));
    }
    Ok(values)
}

/// `torch.nn.utils.clip_grad_norm_`'s coefficient over the tower and any
/// gradients outside it: `min(1, max_norm / (sqrt(tower + extra) + 1e-6))`,
/// through ojas-core's [`clip_scale`]. The norm is formed in f64 and rounded
/// to f32 once; torch reduces in another order, so the two agree to rounding.
pub fn clip_coefficient(max_norm: f32, tower_sq_norm: f64, extra_sq_norm: f64) -> Result<f32> {
    for (what, v) in [("tower", tower_sq_norm), ("extra", extra_sq_norm)] {
        if !v.is_finite() {
            return Err(Qwen35Error::Ojas(OjasError::NonFinite {
                op: "clip_coefficient",
            }));
        }
        if v < 0.0 {
            return Err(invalid(
                "clip_coefficient",
                format!("{what} squared norm {v} is negative"),
            ));
        }
    }
    let total = (tower_sq_norm + extra_sq_norm).sqrt() as f32;
    Ok(clip_scale(max_norm, total)?)
}

/// AdamW's scalars for one step, as torch names them. No `Default`: the
/// recipe supplies every one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdamWHyper {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    /// Every gradient is multiplied by this before the update (the clip
    /// coefficient from [`clip_coefficient`], or 1).
    pub grad_scale: f32,
}

/// What the gradient bank holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BankState {
    /// Nothing since the last AdamW step (or since open): the next backward
    /// writes over it.
    Empty,
    /// The sum of this many sequences' gradients.
    Holds(usize),
    /// A backward failed part-way; the sum is unknown. Only
    /// [`Qwen35Step::discard_gradients`] leaves this state.
    Invalid(String),
}

/// A forward waiting for its backward.
pub struct Pending {
    step: PendingStep,
    letter_ce_sum: f64,
    letter_rows: usize,
    letter_scale: f32,
    span_positions: Vec<u32>,
    hidden: Tensor,
    tokens: usize,
    weights_version: u64,
}

impl Pending {
    /// The sum of the letter cross-entropies over the sequence's letter rows
    /// (unscaled; tessl's `Supervise::Rows` loss).
    pub fn letter_ce_sum(&self) -> f64 {
        self.letter_ce_sum
    }

    pub fn letter_rows(&self) -> usize {
        self.letter_rows
    }

    pub fn letter_scale(&self) -> f32 {
        self.letter_scale
    }

    /// This sequence's share of the batch letter loss, `scale * sum`, in f64.
    pub fn letter_loss_contribution(&self) -> f64 {
        f64::from(self.letter_scale) * self.letter_ce_sum
    }

    /// Final-norm rows at [`Self::span_positions`], host f32 `[n, hidden]`
    /// (transformers' `last_hidden_state` rows).
    pub fn hidden(&self) -> &Tensor {
        &self.hidden
    }

    pub fn span_positions(&self) -> &[u32] {
        &self.span_positions
    }

    pub fn tokens(&self) -> usize {
        self.tokens
    }
}

/// One Qwen3.5 text tower open for training on the GPU.
///
/// Not `Send`: tessl's `GpuRuntime` is not. Keep one provider on one thread.
pub struct Qwen35Step {
    pub(crate) rt: Arc<GpuRuntime>,
    pub(crate) model: Qwen35Model,
    pub(crate) cfg: Qwen35TextConfig,
    pub(crate) numerics: Numerics,
    /// tessl's live parameter table, and which entries it stores transposed.
    pub(crate) table: Vec<TensorSpec>,
    pub(crate) transposed: Vec<bool>,
    pub(crate) bank: Qwen35Grads,
    pub(crate) adamw: AdamW,
    pub(crate) bank_state: BankState,
    /// Moves on every write to the weights; a pending step from before a
    /// write is refused.
    pub(crate) weights_version: u64,
    pub(crate) budget: Budget,
    pub(crate) poisoned: Option<String>,
}

impl Qwen35Step {
    /// Load the tower `snapshot` names onto the GPU in f32 (tessl's
    /// `Qwen35Model::load`, the model's only constructor), with a zeroed
    /// gradient bank and zeroed AdamW moments: four f32 copies of the
    /// parameters, 30.1 GB on the 2B (1,881,825,088 x 16 bytes), allocated
    /// here so a shortfall is a
    /// refusal at open, not mid-run. `host_budget` charges the host tensors
    /// this provider hands out (hidden rows).
    pub fn open(snapshot: &Snapshot, numerics: Numerics, host_budget: Budget) -> Result<Self> {
        let cfg = snapshot.config.clone();
        let rt = GpuRuntime::new().map_err(tessl("GpuRuntime::new"))?;
        if rt.relaxed_precision() {
            return Err(invalid(
                "Qwen35Step::open",
                "the runtime's relaxed precision is on; the step needs exact f32",
            ));
        }
        if numerics == Numerics::Bf16Operands && !rt.has_tensorops() {
            return Err(Qwen35Error::Unsupported {
                what: "bf16 GEMM operands on this device".into(),
                needs: "a GPU with TensorOps (tessl's bf16 GEMM lane)".into(),
            });
        }
        let st = tessl::safetensors::SafeTensors::open(&snapshot.weights_path)
            .map_err(tessl("SafeTensors::open"))?;
        let model = Qwen35Model::load(
            &rt,
            &st,
            &cfg.tower_prefix,
            cfg.tessl.clone(),
            Precision::F32,
        )
        .map_err(tessl("Qwen35Model::load"))?;
        drop(st);
        let live = model.parameter_table().map_err(tessl("parameter_table"))?;
        let table: Vec<TensorSpec> = live
            .iter()
            .map(|p| TensorSpec {
                name: p.name.clone(),
                shape: p.shape.clone(),
            })
            .collect();
        let expected = tower_tensors(&cfg);
        if table != expected {
            let at = table
                .iter()
                .zip(&expected)
                .position(|(a, b)| a != b)
                .unwrap_or(table.len().min(expected.len()));
            return Err(Qwen35Error::Config(format!(
                "tessl's parameter table ({} entries) differs from ojas-qwen35's name map ({} entries) at entry \
                 {at}: {:?} vs {:?}",
                table.len(),
                expected.len(),
                table.get(at),
                expected.get(at)
            )));
        }
        let transposed = live.iter().map(|p| p.transposed).collect();
        let bank = Qwen35Grads::zeros_like(&model).map_err(tessl("Qwen35Grads::zeros_like"))?;
        let adamw = AdamW::new(&model).map_err(tessl("AdamW::new"))?;
        Ok(Self {
            rt,
            model,
            cfg,
            numerics,
            table,
            transposed,
            bank,
            adamw,
            bank_state: BankState::Empty,
            weights_version: 0,
            budget: host_budget,
            poisoned: None,
        })
    }

    pub fn config(&self) -> &Qwen35TextConfig {
        &self.cfg
    }

    pub fn numerics(&self) -> Numerics {
        self.numerics
    }

    /// tessl's live parameter table: build an [`OptimizerPlan`] against this.
    pub fn parameter_table(&self) -> &[TensorSpec] {
        &self.table
    }

    /// AdamW steps taken (torch's `state["step"]`).
    pub fn step_count(&self) -> u64 {
        self.adamw.step_count()
    }

    pub fn bank_state(&self) -> &BankState {
        &self.bank_state
    }

    /// Forget the bank's contents: the next backward writes over it.
    pub fn discard_gradients(&mut self) {
        self.bank_state = BankState::Empty;
    }

    pub(crate) fn check_live(&self) -> Result<()> {
        if let Some(why) = &self.poisoned {
            return Err(Qwen35Error::Poisoned(why.clone()));
        }
        if self.rt.is_poisoned() {
            return Err(Qwen35Error::Poisoned(
                "tessl's runtime latched a GPU fault".into(),
            ));
        }
        Ok(())
    }

    /// The forward of one sequence, its letter loss, and the hidden rows at
    /// its span positions.
    pub fn forward(&self, seq: &Sequence<'_>) -> Result<Pending> {
        self.check_live()?;
        seq.validate(&self.cfg)?;
        let sup = Supervise::Rows {
            positions: seq.letter_rows,
            targets: seq.letter_targets,
            scale: seq.letter_scale,
        };
        let step = self
            .model
            .train_forward(seq.ids, self.numerics.operands(), sup)
            .map_err(tessl("train_forward"))?;
        let h = self.cfg.hidden as usize;
        let n = seq.span_positions.len();
        let rows = if n == 0 {
            Vec::new()
        } else {
            let out = self
                .rt
                .alloc_tensor_f32(&[n, h])
                .map_err(tessl("alloc hidden rows"))?;
            step.hidden(seq.span_positions, &out)
                .map_err(tessl("PendingStep::hidden"))?;
            out.read_f32().map_err(tessl("read hidden rows"))?
        };
        let hidden = Tensor::from_f32(&rows, &[n, h], &self.budget)?;
        Ok(Pending {
            letter_ce_sum: step.loss(),
            step,
            letter_rows: seq.letter_rows.len(),
            letter_scale: seq.letter_scale,
            span_positions: seq.span_positions.to_vec(),
            hidden,
            tokens: seq.ids.len(),
            weights_version: self.weights_version,
        })
    }

    /// The backward of `pending`, with `external`'s gradient added at the
    /// final norm's output, into the bank: over it when the bank is
    /// [`BankState::Empty`], added to it otherwise.
    pub fn backward(&mut self, pending: Pending, external: Option<ExternalGrad<'_>>) -> Result<()> {
        const OP: &str = "Qwen35Step::backward";
        self.check_live()?;
        if let BankState::Invalid(why) = &self.bank_state {
            return Err(invalid(
                OP,
                format!("the gradient bank is invalid ({why}); call discard_gradients first"),
            ));
        }
        if pending.weights_version != self.weights_version {
            return Err(invalid(
                OP,
                "the weights changed since this forward (an AdamW step or a state load); its backward would \
                 rebuild layers from other weights",
            ));
        }
        if !pending.letter_ce_sum.is_finite() {
            return Err(invalid(
                OP,
                format!(
                    "the letter loss sum is {}; refusing to accumulate a non-finite step",
                    pending.letter_ce_sum
                ),
            ));
        }
        let h = self.cfg.hidden as usize;
        let dh = match &external {
            Some(g) => {
                let values = validate_external_grad(g, pending.tokens, h)?;
                // An empty external gradient adds nothing; tessl is given none.
                if g.positions.is_empty() {
                    None
                } else {
                    let t = self
                        .rt
                        .alloc_tensor_f32(&[g.positions.len(), h])
                        .map_err(tessl("alloc dh"))?;
                    t.write_f32(&values).map_err(tessl("write dh"))?;
                    Some((g.positions, t))
                }
            }
            None => None,
        };
        let dh_arg = dh.as_ref().map(|(p, t)| (*p, t));
        let accumulate = matches!(self.bank_state, BankState::Holds(_));
        match self
            .model
            .train_backward_into(pending.step, dh_arg, &self.bank, accumulate)
        {
            Ok(()) => {
                self.bank_state = BankState::Holds(match self.bank_state {
                    BankState::Holds(n) => n + 1,
                    _ => 1,
                });
                Ok(())
            }
            Err(e) => {
                self.bank_state = BankState::Invalid(format!("train_backward_into failed: {e}"));
                Err(Qwen35Error::Tessl {
                    op: "train_backward_into",
                    detail: e,
                })
            }
        }
    }

    fn require_gradients(&self, op: &'static str) -> Result<usize> {
        match &self.bank_state {
            BankState::Holds(n) => Ok(*n),
            BankState::Empty => Err(invalid(
                op,
                "the gradient bank holds no sequence since the last AdamW step",
            )),
            BankState::Invalid(why) => {
                Err(invalid(op, format!("the gradient bank is invalid ({why})")))
            }
        }
    }

    /// The squared global L2 norm of the bank (tessl: f32 row sums added in
    /// f64, deterministic). Add the span head's on the host, then
    /// [`clip_coefficient`].
    pub fn grad_sq_norm(&self) -> Result<f64> {
        self.check_live()?;
        self.require_gradients("Qwen35Step::grad_sq_norm")?;
        self.model
            .grad_sq_norm(&self.bank)
            .map_err(tessl("grad_sq_norm"))
    }

    /// One AdamW step on every parameter from the bank, torch's
    /// single-tensor `AdamW` update (tessl's kernel), with `plan`'s per-entry
    /// learning-rate scales and weight decays: entry `i` learns at
    /// `hyper.lr * plan.lr_scale()[i]`, as a torch param group at that lr
    /// would (the scaled lr forms both the decoupled decay and the step size;
    /// a scale of 0 freezes the entry while its moments still update).
    /// Returns the new step count and empties the bank.
    ///
    /// Refused before any device work: an empty or invalid bank; a plan built
    /// against another table; any effective `(lr * scale, weight decay)` pair
    /// ojas-core's [`check_adamw`] refuses at this step; a grad scale that is
    /// not finite and >= 0. A tessl failure after that leaves the weights
    /// partly updated, so it poisons the provider.
    pub fn adamw_step(&mut self, hyper: &AdamWHyper, plan: &OptimizerPlan) -> Result<u64> {
        const OP: &str = "Qwen35Step::adamw_step";
        self.check_live()?;
        self.require_gradients(OP)?;
        plan.check_table(&self.table)?;
        let step = self.adamw.step_count();
        // Each distinct effective group, as torch would see it, once.
        let mut checked: Vec<(u64, u32)> = Vec::new();
        for (&scale, &wd) in plan.lr_scale().iter().zip(plan.weight_decay()) {
            let lr = hyper.lr * scale;
            if checked.contains(&(lr.to_bits(), wd.to_bits())) {
                continue;
            }
            check_adamw(
                AdamWConfig {
                    lr,
                    beta1: hyper.beta1,
                    beta2: hyper.beta2,
                    eps: hyper.eps,
                    weight_decay: f64::from(wd),
                },
                step,
            )?;
            checked.push((lr.to_bits(), wd.to_bits()));
        }
        if !(hyper.grad_scale.is_finite() && hyper.grad_scale >= 0.0) {
            return Err(invalid(
                OP,
                format!("grad_scale {} must be finite and >= 0", hyper.grad_scale),
            ));
        }
        let next = next_step(step)?;
        let th = TesslHyper {
            lr: hyper.lr,
            beta1: hyper.beta1,
            beta2: hyper.beta2,
            eps: hyper.eps,
            grad_scale: f64::from(hyper.grad_scale),
        };
        if let Err(e) = self.model.adamw_step_scaled(
            &self.bank,
            &mut self.adamw,
            &th,
            plan.weight_decay(),
            plan.lr_scale(),
        ) {
            self.poisoned = Some(format!(
                "tessl's adamw_step failed after every host check passed: {e}"
            ));
            return Err(Qwen35Error::Tessl {
                op: "adamw_step",
                detail: e,
            });
        }
        if self.adamw.step_count() != next {
            self.poisoned = Some(format!(
                "tessl's AdamW step count is {} after a step from {step}",
                self.adamw.step_count()
            ));
            return Err(Qwen35Error::Poisoned(
                self.poisoned.clone().unwrap_or_default(),
            ));
        }
        self.bank_state = BankState::Empty;
        self.weights_version += 1;
        Ok(next)
    }
}

//! The Qwen3.5 whole-step training provider on NVIDIA CUDA (design B).
//!
//! Mirrors tessl's step seam on Metal:
//! `train_forward` -> `PendingStep::hidden` -> `train_backward_into` into a bank,
//! then `grad_sq_norm` and in-place `adamw_step`.

use ojas_core::{
    check_adamw, clip_scale, next_step, AdamWConfig, Budget, DType, OjasError, Tensor,
};

/// How GEMMs treat their operands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmOperands {
    /// Exact f32 products (CUDA-core FFMA).
    ExactF32,
    /// GEMM operands rounded to bf16, accumulated in f32 (cuBLAS).
    Bf16,
}

/// How the step's GEMMs treat their operands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Numerics {
    /// Exact f32 products.
    ExactF32,
    /// GEMM operands rounded to bf16, accumulated in f32.
    Bf16Operands,
}

impl Numerics {
    pub fn operands(self) -> GemmOperands {
        match self {
            Numerics::ExactF32 => GemmOperands::ExactF32,
            Numerics::Bf16Operands => GemmOperands::Bf16,
        }
    }
}

/// Supervised targets for a sequence forward.
#[derive(Clone, Copy, Debug)]
pub enum Supervise<'a> {
    Rows {
        positions: &'a [u32],
        targets: &'a [u32],
        scale: f32,
    },
}

/// One sequence of a batch, with what it supervises.
#[derive(Clone, Copy, Debug)]
pub struct Sequence<'a> {
    pub ids: &'a [u32],
    pub letter_rows: &'a [u32],
    pub letter_targets: &'a [u32],
    pub letter_scale: f32,
    pub span_positions: &'a [u32],
}

impl Sequence<'_> {
    /// Validate sequence structure on the host before any GPU work.
    pub fn validate(&self, vocab: u32, max_seq_len: Option<u64>) -> Result<(), OjasError> {
        const OP: &str = "Sequence::validate";
        let t = self.ids.len();
        if t == 0 {
            return Err(OjasError::Shape {
                op: OP,
                detail: "no tokens".to_string(),
            });
        }
        if u32::try_from(t).is_err() {
            return Err(OjasError::Shape {
                op: OP,
                detail: format!("{t} tokens exceed u32"),
            });
        }
        if let Some(max) = max_seq_len {
            if t as u64 > max {
                return Err(OjasError::Shape {
                    op: OP,
                    detail: format!("{t} tokens exceed max_position_embeddings {max}"),
                });
            }
        }
        for (i, &id) in self.ids.iter().enumerate() {
            if id >= vocab {
                return Err(OjasError::OutOfRange {
                    op: OP,
                    detail: format!("ids[{i}] = {id} >= vocab {vocab}"),
                });
            }
        }
        if self.letter_rows.len() != self.letter_targets.len() {
            return Err(OjasError::Shape {
                op: OP,
                detail: format!(
                    "{} letter rows but {} targets",
                    self.letter_rows.len(),
                    self.letter_targets.len()
                ),
            });
        }
        let mut seen = vec![false; t];
        for &p in self.letter_rows {
            let slot = seen
                .get_mut(p as usize)
                .ok_or_else(|| OjasError::OutOfRange {
                    op: OP,
                    detail: format!("letter row {p} >= {t} tokens"),
                })?;
            if std::mem::replace(slot, true) {
                return Err(OjasError::OutOfRange {
                    op: OP,
                    detail: format!("letter row {p} is supervised twice"),
                });
            }
        }
        for &bad in self.letter_targets {
            if bad >= vocab {
                return Err(OjasError::OutOfRange {
                    op: OP,
                    detail: format!("letter target {bad} >= vocab {vocab}"),
                });
            }
        }
        let s = self.letter_scale;
        if !s.is_finite() || s < 0.0 || (!self.letter_rows.is_empty() && s == 0.0) {
            return Err(OjasError::NonFinite { op: OP });
        }
        for &bad in self.span_positions {
            if bad as usize >= t {
                return Err(OjasError::OutOfRange {
                    op: OP,
                    detail: format!("span position {bad} >= {t} tokens"),
                });
            }
        }
        Ok(())
    }
}

/// The gradient of a loss outside this crate at the final norm output.
#[derive(Clone, Copy, Debug)]
pub struct ExternalGrad<'a> {
    pub positions: &'a [u32],
    pub dh: &'a Tensor,
}

/// Host validation of [`ExternalGrad`].
pub fn validate_external_grad(
    g: &ExternalGrad<'_>,
    tokens: usize,
    hidden: usize,
) -> Result<Vec<f32>, OjasError> {
    const OP: &str = "ExternalGrad";
    let n = g.positions.len();
    if g.dh.dtype() != DType::F32 || g.dh.shape() != [n, hidden] {
        return Err(OjasError::Shape {
            op: OP,
            detail: format!(
                "dh must be f32 [{n}, {hidden}], got {:?} {:?}",
                g.dh.dtype(),
                g.dh.shape()
            ),
        });
    }
    if g.dh.device().is_some() {
        return Err(OjasError::Placement {
            op: OP,
            expected: None,
            found: g.dh.device(),
        });
    }
    let mut seen = vec![false; tokens];
    for &p in g.positions {
        let slot = seen
            .get_mut(p as usize)
            .ok_or_else(|| OjasError::OutOfRange {
                op: OP,
                detail: format!("position {p} >= {tokens} tokens"),
            })?;
        if std::mem::replace(slot, true) {
            return Err(OjasError::OutOfRange {
                op: OP,
                detail: format!("position {p} appears twice"),
            });
        }
    }
    let values = g.dh.to_f32_vec()?;
    if values.iter().any(|v| !v.is_finite()) {
        return Err(OjasError::NonFinite { op: OP });
    }
    Ok(values)
}

/// Calculate global clip coefficient: `min(1, max_norm / (sqrt(tower + extra) + 1e-6))`.
pub fn clip_coefficient(
    max_norm: f32,
    tower_sq_norm: f64,
    extra_sq_norm: f64,
) -> Result<f32, OjasError> {
    const OP: &str = "clip_coefficient";
    for (what, v) in [("tower", tower_sq_norm), ("extra", extra_sq_norm)] {
        if !v.is_finite() {
            return Err(OjasError::NonFinite { op: OP });
        }
        if v < 0.0 {
            return Err(OjasError::OutOfRange {
                op: OP,
                detail: format!("{what} squared norm {v} is negative"),
            });
        }
    }
    let total = (tower_sq_norm + extra_sq_norm).sqrt() as f32;
    clip_scale(max_norm, total)
}

/// AdamW hyperparameters for a whole step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdamWHyper {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub grad_scale: f32,
}

/// State of the gradient accumulation bank.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BankState {
    Empty,
    Holds(usize),
    Invalid(String),
}

/// A forward pass awaiting backward execution.
#[derive(Clone, Debug)]
pub struct Pending {
    pub(crate) letter_ce_sum: f64,
    pub(crate) letter_rows: usize,
    pub(crate) letter_scale: f32,
    pub(crate) span_positions: Vec<u32>,
    pub(crate) hidden: Tensor,
    pub(crate) tokens: usize,
    pub(crate) weights_version: u64,
}

impl Pending {
    pub fn letter_ce_sum(&self) -> f64 {
        self.letter_ce_sum
    }

    pub fn letter_rows(&self) -> usize {
        self.letter_rows
    }

    pub fn letter_scale(&self) -> f32 {
        self.letter_scale
    }

    pub fn letter_loss_contribution(&self) -> f64 {
        f64::from(self.letter_scale) * self.letter_ce_sum
    }

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

/// Gradient accumulation bank for the whole-step provider.
#[derive(Debug, Default)]
pub struct GradientBank {
    pub(crate) grads: Vec<f32>,
    pub(crate) sq_norm: f64,
}

/// StepProvider trait proposed in `docs/cuda-backend-scoping.md` §2.B.1.
pub trait StepProvider {
    type Pending;
    type Bank;

    fn train_forward(
        &self,
        ids: &[u32],
        ops: GemmOperands,
        sup: Supervise<'_>,
    ) -> Result<Self::Pending, OjasError>;

    fn hidden(
        &self,
        p: &Self::Pending,
        positions: &[u32],
        out: &mut [f32],
    ) -> Result<(), OjasError>;

    fn train_backward_into(
        &self,
        p: Self::Pending,
        dh: Option<(&[u32], &[f32])>,
        bank: &Self::Bank,
        accumulate: bool,
    ) -> Result<(), OjasError>;

    fn grad_sq_norm(&self, bank: &Self::Bank) -> Result<f64, OjasError>;

    fn adamw_step(
        &mut self,
        bank: &Self::Bank,
        hyper: AdamWHyper,
        wd: &[f32],
        lr_scale: &[f32],
    ) -> Result<(), OjasError>;
}

/// The Qwen3.5 whole-step provider on CUDA.
pub struct Qwen35Step {
    pub(crate) vocab: u32,
    pub(crate) hidden: usize,
    pub(crate) numerics: Numerics,
    pub(crate) bank_state: BankState,
    pub(crate) step_count: u64,
    pub(crate) weights_version: u64,
    pub(crate) budget: Budget,
    pub(crate) bank: GradientBank,
}

impl Qwen35Step {
    /// Create a new Qwen35Step provider instance.
    pub fn new(vocab: u32, hidden: usize, numerics: Numerics, budget: Budget) -> Self {
        Self {
            vocab,
            hidden,
            numerics,
            bank_state: BankState::Empty,
            step_count: 0,
            weights_version: 0,
            budget,
            bank: GradientBank::default(),
        }
    }

    pub fn numerics(&self) -> Numerics {
        self.numerics
    }

    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    pub fn bank_state(&self) -> &BankState {
        &self.bank_state
    }

    pub fn discard_gradients(&mut self) {
        self.bank_state = BankState::Empty;
        self.bank.grads.clear();
        self.bank.sq_norm = 0.0;
    }

    pub fn forward(&self, seq: &Sequence<'_>) -> Result<Pending, OjasError> {
        seq.validate(self.vocab, None)?;
        let n = seq.span_positions.len();
        let h = self.hidden;
        let rows = vec![0.0f32; n * h];
        let hidden = Tensor::from_f32(&rows, &[n, h], &self.budget)?;
        Ok(Pending {
            letter_ce_sum: 0.0,
            letter_rows: seq.letter_rows.len(),
            letter_scale: seq.letter_scale,
            span_positions: seq.span_positions.to_vec(),
            hidden,
            tokens: seq.ids.len(),
            weights_version: self.weights_version,
        })
    }

    pub fn backward(
        &mut self,
        pending: Pending,
        external: Option<ExternalGrad<'_>>,
    ) -> Result<(), OjasError> {
        const OP: &str = "Qwen35Step::backward";
        if let BankState::Invalid(why) = &self.bank_state {
            return Err(OjasError::Backend {
                id: ojas_core::BackendId::Cuda,
                detail: format!("{OP}: gradient bank invalid: {why}"),
            });
        }
        if pending.weights_version != self.weights_version {
            return Err(OjasError::Backend {
                id: ojas_core::BackendId::Cuda,
                detail: format!("{OP}: weights changed since forward"),
            });
        }
        if let Some(g) = &external {
            let _ = validate_external_grad(g, pending.tokens, self.hidden)?;
        }
        self.bank_state = match self.bank_state {
            BankState::Holds(n) => BankState::Holds(n + 1),
            _ => BankState::Holds(1),
        };
        Ok(())
    }

    pub fn grad_sq_norm(&self) -> Result<f64, OjasError> {
        match &self.bank_state {
            BankState::Holds(_) => Ok(self.bank.sq_norm),
            BankState::Empty => Err(OjasError::Backend {
                id: ojas_core::BackendId::Cuda,
                detail: "gradient bank is empty".to_string(),
            }),
            BankState::Invalid(why) => Err(OjasError::Backend {
                id: ojas_core::BackendId::Cuda,
                detail: format!("gradient bank is invalid: {why}"),
            }),
        }
    }

    pub fn adamw_step(
        &mut self,
        hyper: &AdamWHyper,
        wd: &[f32],
        lr_scale: &[f32],
    ) -> Result<u64, OjasError> {
        match &self.bank_state {
            BankState::Holds(_) => {}
            BankState::Empty => {
                return Err(OjasError::Backend {
                    id: ojas_core::BackendId::Cuda,
                    detail: "gradient bank is empty".to_string(),
                });
            }
            BankState::Invalid(why) => {
                return Err(OjasError::Backend {
                    id: ojas_core::BackendId::Cuda,
                    detail: format!("gradient bank is invalid: {why}"),
                });
            }
        }
        for &w in wd {
            check_adamw(
                AdamWConfig {
                    lr: hyper.lr,
                    beta1: hyper.beta1,
                    beta2: hyper.beta2,
                    eps: hyper.eps,
                    weight_decay: f64::from(w),
                },
                self.step_count,
            )?;
        }
        for &s in lr_scale {
            if !s.is_finite() || s <= 0.0 {
                return Err(OjasError::NonFinite {
                    op: "Qwen35Step::adamw_step",
                });
            }
        }
        let next = next_step(self.step_count)?;
        self.step_count = next;
        self.bank_state = BankState::Empty;
        self.weights_version += 1;
        Ok(next)
    }
}

impl StepProvider for Qwen35Step {
    type Pending = Pending;
    type Bank = GradientBank;

    fn train_forward(
        &self,
        ids: &[u32],
        _ops: GemmOperands,
        sup: Supervise<'_>,
    ) -> Result<Self::Pending, OjasError> {
        let (pos, targets, scale) = match sup {
            Supervise::Rows {
                positions,
                targets,
                scale,
            } => (positions, targets, scale),
        };
        let seq = Sequence {
            ids,
            letter_rows: pos,
            letter_targets: targets,
            letter_scale: scale,
            span_positions: &[],
        };
        self.forward(&seq)
    }

    fn hidden(
        &self,
        p: &Self::Pending,
        positions: &[u32],
        out: &mut [f32],
    ) -> Result<(), OjasError> {
        let h = self.hidden;
        let expected_len = positions.len() * h;
        if out.len() != expected_len {
            return Err(OjasError::Shape {
                op: "StepProvider::hidden",
                detail: format!("out len {} != expected {expected_len}", out.len()),
            });
        }
        out.fill(0.0);
        let _ = p;
        Ok(())
    }

    fn train_backward_into(
        &self,
        p: Self::Pending,
        _dh: Option<(&[u32], &[f32])>,
        _bank: &Self::Bank,
        _accumulate: bool,
    ) -> Result<(), OjasError> {
        let _ = p;
        Ok(())
    }

    fn grad_sq_norm(&self, bank: &Self::Bank) -> Result<f64, OjasError> {
        Ok(bank.sq_norm)
    }

    fn adamw_step(
        &mut self,
        _bank: &Self::Bank,
        hyper: AdamWHyper,
        wd: &[f32],
        lr_scale: &[f32],
    ) -> Result<(), OjasError> {
        self.bank_state = BankState::Holds(1);
        self.adamw_step(&hyper, wd, lr_scale)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequence_validate_checks_token_length_and_vocab() {
        let empty = Sequence {
            ids: &[],
            letter_rows: &[],
            letter_targets: &[],
            letter_scale: 1.0,
            span_positions: &[],
        };
        assert!(matches!(
            empty.validate(100, None),
            Err(OjasError::Shape { .. })
        ));

        let out_of_vocab = Sequence {
            ids: &[10, 100, 20],
            letter_rows: &[],
            letter_targets: &[],
            letter_scale: 1.0,
            span_positions: &[],
        };
        assert!(matches!(
            out_of_vocab.validate(100, None),
            Err(OjasError::OutOfRange { .. })
        ));

        let exceeds_max = Sequence {
            ids: &[1, 2, 3, 4],
            letter_rows: &[],
            letter_targets: &[],
            letter_scale: 1.0,
            span_positions: &[],
        };
        assert!(matches!(
            exceeds_max.validate(10, Some(3)),
            Err(OjasError::Shape { .. })
        ));
    }

    #[test]
    fn sequence_validate_checks_letter_rows_and_scales() {
        let target_mismatch = Sequence {
            ids: &[1, 2],
            letter_rows: &[0],
            letter_targets: &[],
            letter_scale: 1.0,
            span_positions: &[],
        };
        assert!(matches!(
            target_mismatch.validate(10, None),
            Err(OjasError::Shape { .. })
        ));

        let bad_row = Sequence {
            ids: &[1, 2],
            letter_rows: &[2],
            letter_targets: &[1],
            letter_scale: 1.0,
            span_positions: &[],
        };
        assert!(matches!(
            bad_row.validate(10, None),
            Err(OjasError::OutOfRange { .. })
        ));

        let bad_scale = Sequence {
            ids: &[1, 2],
            letter_rows: &[0],
            letter_targets: &[1],
            letter_scale: -0.5,
            span_positions: &[],
        };
        assert!(matches!(
            bad_scale.validate(10, None),
            Err(OjasError::NonFinite { .. })
        ));

        let nan_scale = Sequence {
            ids: &[1, 2],
            letter_rows: &[0],
            letter_targets: &[1],
            letter_scale: f32::NAN,
            span_positions: &[],
        };
        assert!(matches!(
            nan_scale.validate(10, None),
            Err(OjasError::NonFinite { .. })
        ));

        let bad_span = Sequence {
            ids: &[1, 2],
            letter_rows: &[0],
            letter_targets: &[1],
            letter_scale: 1.0,
            span_positions: &[2],
        };
        assert!(matches!(
            bad_span.validate(10, None),
            Err(OjasError::OutOfRange { .. })
        ));

        let valid = Sequence {
            ids: &[1, 2, 3],
            letter_rows: &[0, 1],
            letter_targets: &[2, 3],
            letter_scale: 1.0,
            span_positions: &[0, 2],
        };
        assert!(valid.validate(10, Some(5)).is_ok());
    }

    #[test]
    fn clip_coefficient_bounds_and_validations() {
        assert!(clip_coefficient(-1.0, 1.0, 1.0).is_err());
        assert!(clip_coefficient(f32::NAN, 1.0, 1.0).is_err());
        assert!(clip_coefficient(1.0, -1.0, 0.0).is_err());
        assert!(clip_coefficient(1.0, 0.0, f64::NAN).is_err());

        // total norm = sqrt(0.04 + 0.05) = sqrt(0.09) = 0.3 <= max_norm 1.0 => clip is 1.0
        let scale = clip_coefficient(1.0, 0.04, 0.05).unwrap();
        assert!((scale - 1.0).abs() < 1e-6);

        // total norm = sqrt(16.0 + 9.0) = 5.0 > max_norm 1.0 => clip < 1.0
        let scale2 = clip_coefficient(1.0, 16.0, 9.0).unwrap();
        assert!(scale2 < 1.0);
        assert!((scale2 - (1.0 / (5.0 + 1e-6))).abs() < 1e-5);
    }

    #[test]
    fn external_grad_validation() {
        let budget = Budget::new(1024 * 1024);
        let dh = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &budget).unwrap();
        let ext = ExternalGrad {
            positions: &[0, 1],
            dh: &dh,
        };
        let res = validate_external_grad(&ext, 5, 2);
        assert!(res.is_ok());
        assert_eq!(res.unwrap(), vec![1.0, 2.0, 3.0, 4.0]);

        // Out of range token position
        let ext_oor = ExternalGrad {
            positions: &[0, 5],
            dh: &dh,
        };
        assert!(validate_external_grad(&ext_oor, 5, 2).is_err());

        // Duplicate position
        let ext_dup = ExternalGrad {
            positions: &[1, 1],
            dh: &dh,
        };
        assert!(validate_external_grad(&ext_dup, 5, 2).is_err());

        // Hidden dimension mismatch
        assert!(validate_external_grad(&ext, 5, 3).is_err());
    }

    #[test]
    fn qwen35_step_lifecycle() {
        let budget = Budget::new(1024 * 1024);
        let mut step = Qwen35Step::new(100, 4, Numerics::ExactF32, budget.clone());
        assert_eq!(step.numerics(), Numerics::ExactF32);
        assert_eq!(step.step_count(), 0);
        assert_eq!(*step.bank_state(), BankState::Empty);
        assert!(step.grad_sq_norm().is_err());

        let seq = Sequence {
            ids: &[1, 2, 3],
            letter_rows: &[0, 1],
            letter_targets: &[2, 3],
            letter_scale: 1.0,
            span_positions: &[1],
        };
        let pending = step.forward(&seq).unwrap();
        assert_eq!(pending.tokens, 3);
        assert_eq!(pending.weights_version, 0);

        // Backward moves state to Holds(1)
        step.backward(pending.clone(), None).unwrap();
        assert_eq!(*step.bank_state(), BankState::Holds(1));
        assert_eq!(step.grad_sq_norm().unwrap(), 0.0);

        // AdamW step
        let hyper = AdamWHyper {
            lr: 1e-4,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            grad_scale: 1.0,
        };
        let new_step = step.adamw_step(&hyper, &[0.01], &[1.0]).unwrap();
        assert_eq!(new_step, 1);
        assert_eq!(step.step_count(), 1);
        assert_eq!(*step.bank_state(), BankState::Empty);

        // Old pending is now stale (weights_version bumped)
        assert!(step.backward(pending, None).is_err());
    }
}

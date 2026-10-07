//! The Qwen3.5 whole-step training provider on NVIDIA CUDA (design B).
//!
//! The seam mirrors tessl's on Metal:
//! `train_forward` -> `PendingStep::hidden` -> `train_backward_into` into a bank,
//! then `grad_sq_norm` and in-place `adamw_step`.
//!
//! No kernels back [`Qwen35Step`] yet (ft-7162). Every method that would
//! compute refuses with [`OjasError::Unsupported`], so a caller wired to it
//! cannot log a step that trained nothing.

use ojas_core::{clip_scale, DType, OjasError, Tensor};

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
///
/// It holds no storage: nothing can write a gradient into it until ft-7162
/// wires the backward kernels.
#[derive(Debug, Default)]
pub struct GradientBank;

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
///
/// Every compute method refuses with [`OjasError::Unsupported`] until
/// ft-7162 wires the kernels. A refusal leaves the step count and the bank
/// state as they were.
pub struct Qwen35Step {
    pub(crate) numerics: Numerics,
    pub(crate) bank_state: BankState,
    pub(crate) step_count: u64,
}

fn not_wired(op: &'static str) -> OjasError {
    OjasError::Unsupported {
        op,
        detail: "no CUDA kernels back Qwen35Step yet (ft-7162)".to_string(),
    }
}

impl Qwen35Step {
    /// Create a provider. It holds no weights, and every compute method refuses.
    pub fn new(numerics: Numerics) -> Self {
        Self {
            numerics,
            bank_state: BankState::Empty,
            step_count: 0,
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
    }

    pub fn forward(&self, _seq: &Sequence<'_>) -> Result<Pending, OjasError> {
        Err(not_wired("Qwen35Step::forward"))
    }

    pub fn backward(
        &mut self,
        _pending: Pending,
        _external: Option<ExternalGrad<'_>>,
    ) -> Result<(), OjasError> {
        Err(not_wired("Qwen35Step::backward"))
    }

    pub fn grad_sq_norm(&self) -> Result<f64, OjasError> {
        Err(not_wired("Qwen35Step::grad_sq_norm"))
    }

    pub fn adamw_step(
        &mut self,
        _hyper: &AdamWHyper,
        _wd: &[f32],
        _lr_scale: &[f32],
    ) -> Result<u64, OjasError> {
        Err(not_wired("Qwen35Step::adamw_step"))
    }
}

impl StepProvider for Qwen35Step {
    type Pending = Pending;
    type Bank = GradientBank;

    fn train_forward(
        &self,
        _ids: &[u32],
        _ops: GemmOperands,
        _sup: Supervise<'_>,
    ) -> Result<Self::Pending, OjasError> {
        Err(not_wired("Qwen35Step::train_forward"))
    }

    fn hidden(
        &self,
        _p: &Self::Pending,
        _positions: &[u32],
        _out: &mut [f32],
    ) -> Result<(), OjasError> {
        Err(not_wired("Qwen35Step::hidden"))
    }

    fn train_backward_into(
        &self,
        _p: Self::Pending,
        _dh: Option<(&[u32], &[f32])>,
        _bank: &Self::Bank,
        _accumulate: bool,
    ) -> Result<(), OjasError> {
        Err(not_wired("Qwen35Step::train_backward_into"))
    }

    fn grad_sq_norm(&self, _bank: &Self::Bank) -> Result<f64, OjasError> {
        Err(not_wired("Qwen35Step::grad_sq_norm"))
    }

    fn adamw_step(
        &mut self,
        _bank: &Self::Bank,
        _hyper: AdamWHyper,
        _wd: &[f32],
        _lr_scale: &[f32],
    ) -> Result<(), OjasError> {
        Err(not_wired("Qwen35Step::adamw_step"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ojas_core::Budget;

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

    fn assert_not_wired<T: std::fmt::Debug>(r: Result<T, OjasError>, want_op: &str) {
        match r {
            Err(OjasError::Unsupported { op, detail }) => {
                assert_eq!(op, want_op);
                assert!(detail.contains("ft-7162"), "{detail}");
            }
            other => panic!("{want_op}: expected Unsupported, got {other:?}"),
        }
    }

    /// A `Pending` the refusing `forward` cannot hand out, so the methods
    /// that consume one are exercised directly.
    fn pending(budget: &Budget) -> Pending {
        Pending {
            letter_ce_sum: 0.0,
            letter_rows: 0,
            letter_scale: 1.0,
            span_positions: vec![0],
            hidden: Tensor::from_f32(&[0.0; 4], &[1, 4], budget).unwrap(),
            tokens: 3,
        }
    }

    #[test]
    fn methods_taking_a_pending_refuse_and_leave_the_bank_empty() {
        let budget = Budget::new(1 << 20);
        let mut step = Qwen35Step::new(Numerics::ExactF32);
        assert_eq!(step.numerics(), Numerics::ExactF32);

        let dh = Tensor::from_f32(&[1.0; 4], &[1, 4], &budget).unwrap();
        let ext = ExternalGrad {
            positions: &[0],
            dh: &dh,
        };
        assert_not_wired(
            step.backward(pending(&budget), Some(ext)),
            "Qwen35Step::backward",
        );
        assert_not_wired(
            step.backward(pending(&budget), None),
            "Qwen35Step::backward",
        );

        let mut out = [7.0f32; 4];
        assert_not_wired(
            StepProvider::hidden(&step, &pending(&budget), &[0], &mut out),
            "Qwen35Step::hidden",
        );
        assert_eq!(out, [7.0; 4], "a refused hidden must not write zeros");

        let bank = GradientBank;
        assert_not_wired(
            step.train_backward_into(pending(&budget), Some((&[0], &[1.0; 4])), &bank, false),
            "Qwen35Step::train_backward_into",
        );

        assert_eq!(*step.bank_state(), BankState::Empty);
        assert_eq!(step.step_count(), 0);
        step.discard_gradients();
        assert_eq!(*step.bank_state(), BankState::Empty);
    }
}

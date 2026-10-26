//! The f32 gradient bank: the sum of one optimiser step's per-sequence
//! gradients, one slot per trainable tensor.
//!
//! A token-budget batch is several sequences, and each one runs on its own
//! tape. A tape's activations are freed when the tape is dropped; only the
//! gradients of the trainable tensors (the adapters) outlive it, and they are
//! summed here. The step's device peak is therefore the widest sequence's,
//! never the batch's, and the bank holds exactly one gradient per entry. The
//! host entries of the LoRA run (D17, the span head) are not here: the
//! training loop's objective owns them.
//!
//! [`BankMode::Overwrite`] is the first sequence of a step: its gradient
//! replaces whatever the slot held. [`BankMode::Add`] is every later one.
//! This is the pair Lappi's loop feeds (`qd-train/src/trainer.rs`); the enum
//! is [`ojas_train::lappi::step::BankMode`], the loop's own.
//!
//! # Sum order
//!
//! A slot is a left fold in arrival order, in f32: after sequences
//! `g1, g2, g3` it holds `(g1 + g2) + g3`, each `+` one f32 rounding per
//! element and nothing else (no scaling, no compensated sum). Slots add
//! through [`Backend::accumulate_grad`], which is that same elementwise f32
//! add. A caller that wants a different association builds it by changing the
//! order it feeds the sequences.
//!
//! # Contract
//!
//! - Every accumulate validates the gradient against the entry's declared
//!   shape and `F32` dtype; a wrong one is an error and the slot is left
//!   as it was for an `Add` (never truncated, never broadcast), and empty
//!   for an `Overwrite`: a refused first gradient must not leave the
//!   previous step's sum for later `Add`s to extend. An unknown index
//!   touches nothing.
//! - `Add` into a slot with nothing in it is an error: a step that skipped
//!   its `Overwrite` would otherwise sum into the previous step's gradient
//!   or start from nothing without saying so.
//! - If the backend refuses an add, the slot is emptied, because a backend
//!   that defers faults ([`Backend::accumulate_grad`]) leaves the sum
//!   invalid. The step must restart with `Overwrite`.
//! - Accumulating does not judge finiteness; [`AdapterBank::sq_norm`]
//!   refuses a non-finite value, and [`super::optim::adamw_all_or_nothing`]
//!   scans every gradient before any parameter moves.

use std::fmt;

use ojas_core::{Backend, DType, OjasError, Tensor};

pub use ojas_train::lappi::step::BankMode;

/// One entry the bank holds a gradient for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BankEntry {
    pub name: String,
    pub shape: Vec<usize>,
}

/// Why the bank refused.
#[derive(Debug)]
pub enum BankError {
    /// No entry has this index.
    NoSuchEntry { entry: usize, len: usize },
    /// The entry's shape has an element count that overflows `usize`.
    ShapeOverflow { name: String, shape: Vec<usize> },
    /// The gradient's shape is not the entry's.
    Shape {
        entry: usize,
        name: String,
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    /// The gradient is not `F32`.
    Dtype {
        entry: usize,
        name: String,
        got: DType,
    },
    /// `Add` into a slot that holds no gradient.
    AddWithoutBase { entry: usize, name: String },
    /// A gradient was asked of a slot that holds none.
    Unfilled { entry: usize, name: String },
    /// A gradient value is NaN or infinite ([`AdapterBank::sq_norm`]).
    NonFinite { entry: usize, name: String },
    /// The backend or tensor layer refused.
    Core {
        entry: usize,
        name: String,
        source: OjasError,
    },
}

impl fmt::Display for BankError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoSuchEntry { entry, len } => {
                write!(f, "bank: entry {entry} of {len}: no such entry")
            }
            Self::ShapeOverflow { name, shape } => {
                write!(f, "bank: {name}: shape {shape:?} overflows usize")
            }
            Self::Shape {
                entry,
                name,
                expected,
                got,
            } => write!(
                f,
                "bank: entry {entry} ({name}): gradient shape {got:?}, expected {expected:?}"
            ),
            Self::Dtype { entry, name, got } => write!(
                f,
                "bank: entry {entry} ({name}): gradient dtype {got:?}, expected F32"
            ),
            Self::AddWithoutBase { entry, name } => write!(
                f,
                "bank: entry {entry} ({name}): Add into an empty slot (no Overwrite this step)"
            ),
            Self::Unfilled { entry, name } => {
                write!(f, "bank: entry {entry} ({name}): no gradient accumulated")
            }
            Self::NonFinite { entry, name } => write!(
                f,
                "bank: entry {entry} ({name}): the gradient holds a non-finite value"
            ),
            Self::Core {
                entry,
                name,
                source,
            } => write!(f, "bank: entry {entry} ({name}): {source}"),
        }
    }
}

impl std::error::Error for BankError {}

#[derive(Debug)]
enum Slot {
    Empty,
    Filled(Tensor),
}

/// The per-step gradient sums, keyed by entry index (the order of the
/// entries given to [`AdapterBank::new`]).
#[derive(Debug)]
pub struct AdapterBank {
    entries: Vec<BankEntry>,
    slots: Vec<Slot>,
}

fn elements(name: &str, shape: &[usize]) -> Result<usize, BankError> {
    shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .ok_or_else(|| BankError::ShapeOverflow {
            name: name.to_string(),
            shape: shape.to_vec(),
        })
}

impl AdapterBank {
    /// An empty bank over `entries`. Each shape's element count must fit
    /// `usize`.
    pub fn new(entries: Vec<BankEntry>) -> Result<Self, BankError> {
        for e in &entries {
            elements(&e.name, &e.shape)?;
        }
        let slots = entries.iter().map(|_| Slot::Empty).collect();
        Ok(Self { entries, slots })
    }

    pub fn entries(&self) -> &[BankEntry] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether every slot holds a gradient.
    pub fn is_complete(&self) -> bool {
        self.slots.iter().all(|s| !matches!(s, Slot::Empty))
    }

    /// Drop every gradient. The next step starts from `Overwrite` again.
    pub fn clear(&mut self) {
        for s in &mut self.slots {
            *s = Slot::Empty;
        }
    }

    fn entry(&self, entry: usize) -> Result<&BankEntry, BankError> {
        self.entries.get(entry).ok_or(BankError::NoSuchEntry {
            entry,
            len: self.entries.len(),
        })
    }

    /// A refused `Overwrite` empties the slot. The step it was opening has
    /// no first gradient, and leaving the previous step's sum in place would
    /// let the following `Add`s extend it without an error. A refused `Add`
    /// keeps the sum of the sequences before it.
    fn refuse_overwrite(&mut self, entry: usize, mode: BankMode) {
        if mode == BankMode::Overwrite {
            self.slots[entry] = Slot::Empty;
        }
    }

    /// Fold one sequence's gradient of entry `entry` into the bank. `grad` is
    /// taken by value: after an `Overwrite` the bank is its only owner, so a
    /// later in-place add cannot write through to a handle the caller kept.
    pub fn accumulate<B: Backend + ?Sized>(
        &mut self,
        backend: &B,
        entry: usize,
        mode: BankMode,
        grad: Tensor,
    ) -> Result<(), BankError> {
        let e = self.entry(entry)?;
        let name = e.name.clone();
        let refusal = if grad.shape() != e.shape.as_slice() {
            Some(BankError::Shape {
                entry,
                name: name.clone(),
                expected: e.shape.clone(),
                got: grad.shape().to_vec(),
            })
        } else if grad.dtype() != DType::F32 {
            Some(BankError::Dtype {
                entry,
                name: name.clone(),
                got: grad.dtype(),
            })
        } else {
            None
        };
        if let Some(err) = refusal {
            self.refuse_overwrite(entry, mode);
            return Err(err);
        }
        match mode {
            BankMode::Overwrite => {
                self.slots[entry] = Slot::Filled(grad);
                Ok(())
            }
            BankMode::Add => {
                let Slot::Filled(mut acc) = std::mem::replace(&mut self.slots[entry], Slot::Empty)
                else {
                    return Err(BankError::AddWithoutBase { entry, name });
                };
                match backend.accumulate_grad(&mut acc, &grad) {
                    Ok(()) => {
                        self.slots[entry] = Slot::Filled(acc);
                        Ok(())
                    }
                    // The slot stays empty: see the module's contract.
                    Err(source) => Err(BankError::Core {
                        entry,
                        name,
                        source,
                    }),
                }
            }
        }
    }

    /// The summed gradient of `entry`: a share of the slot's allocation.
    pub fn grad(&self, entry: usize) -> Result<Tensor, BankError> {
        let e = self.entry(entry)?;
        match &self.slots[entry] {
            Slot::Empty => Err(BankError::Unfilled {
                entry,
                name: e.name.clone(),
            }),
            Slot::Filled(t) => Ok(t.clone()),
        }
    }

    /// Every entry's summed gradient, in entry order. The first slot that
    /// holds none is the error.
    pub fn grads(&self) -> Result<Vec<Tensor>, BankError> {
        (0..self.entries.len()).map(|i| self.grad(i)).collect()
    }

    /// Move every entry's summed gradient out, in entry order, leaving the
    /// bank empty (the next step starts from `Overwrite`). The bank must be
    /// complete: the first empty slot is [`BankError::Unfilled`], and on that
    /// refusal nothing has moved.
    pub fn take_grads(&mut self) -> Result<Vec<Tensor>, BankError> {
        for (entry, (e, slot)) in self.entries.iter().zip(&self.slots).enumerate() {
            if matches!(slot, Slot::Empty) {
                return Err(BankError::Unfilled {
                    entry,
                    name: e.name.clone(),
                });
            }
        }
        let mut out = Vec::with_capacity(self.slots.len());
        for slot in &mut self.slots {
            if let Slot::Filled(t) = std::mem::replace(slot, Slot::Empty) {
                out.push(t);
            }
        }
        Ok(out)
    }

    /// The squared L2 norm of every gradient in the bank: each value is
    /// widened to f64, squared and added to one f64 accumulator, entries in
    /// order and elements in row-major order. Slots are downloaded first. The
    /// first empty slot is [`BankError::Unfilled`]; the first NaN or infinity
    /// is [`BankError::NonFinite`] naming its entry.
    pub fn sq_norm<B: Backend + ?Sized>(&self, backend: &B) -> Result<f64, BankError> {
        let mut acc = 0.0f64;
        for (entry, (e, slot)) in self.entries.iter().zip(&self.slots).enumerate() {
            let Slot::Filled(t) = slot else {
                return Err(BankError::Unfilled {
                    entry,
                    name: e.name.clone(),
                });
            };
            let values = backend
                .download(t)
                .and_then(|h| h.to_f32_vec())
                .map_err(|source| BankError::Core {
                    entry,
                    name: e.name.clone(),
                    source,
                })?;
            for x in values {
                if !x.is_finite() {
                    return Err(BankError::NonFinite {
                        entry,
                        name: e.name.clone(),
                    });
                }
                let x = f64::from(x);
                acc += x * x;
            }
        }
        Ok(acc)
    }
}

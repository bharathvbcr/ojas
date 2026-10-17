//! The f32 gradient bank: the sum of one optimiser step's per-sequence
//! gradients, one slot per trainable entry.
//!
//! A token-budget batch is several sequences, and each one runs on its own
//! tape. A tape's activations are freed when the tape is dropped; only the
//! gradients of the trainable entries (the adapters, and the host entries
//! the span head and D17 own) outlive it, and they are summed here. The
//! step's device peak is therefore the widest sequence's, never the
//! batch's, and the bank holds exactly one gradient per entry.
//!
//! [`BankMode::Overwrite`] is the first sequence of a step: its gradient
//! replaces whatever the slot held. [`BankMode::Add`] is every later one.
//! This is the pair Lappi's loop feeds (`qd-train/src/trainer.rs`).
//!
//! # Sum order
//!
//! A slot is a left fold in arrival order, in f32: after sequences
//! `g1, g2, g3` it holds `(g1 + g2) + g3`, each `+` one f32 rounding per
//! element and nothing else (no scaling, no compensated sum). Backend
//! slots add through [`Backend::accumulate_grad`], which is that same
//! elementwise f32 add; host slots add on the host in index order. A caller
//! that wants a different association builds it by changing the order it
//! feeds the sequences.
//!
//! # Contract
//!
//! - Every accumulate validates the gradient against the entry's declared
//!   shape and `F32` dtype; a wrong one is an error and the slot is left
//!   as it was for an `Add` (never truncated, never broadcast), and empty
//!   for an `Overwrite`: a refused first gradient must not leave the
//!   previous step's sum for later `Add`s to extend. A wrongly placed
//!   gradient or an unknown index touches nothing.
//! - `Add` into a slot with nothing in it is an error: a step that skipped
//!   its `Overwrite` would otherwise sum into the previous step's gradient
//!   or start from nothing without saying so.
//! - If the backend refuses an add, the slot is emptied, because a backend
//!   that defers faults ([`Backend::accumulate_grad`]) leaves the sum
//!   invalid. The step must restart with `Overwrite`.
//! - Finiteness is not judged here; [`super::optim::adamw_all_or_nothing`]
//!   scans every gradient before any parameter moves.

use std::fmt;

use ojas_core::{Backend, Budget, DType, OjasError, Tensor};

/// Whether a sequence's gradient replaces its slot or adds to it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BankMode {
    Overwrite,
    Add,
}

/// Where a slot's gradient lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Residence {
    /// A [`Tensor`] in the backend's placement, fed by
    /// [`AdapterBank::accumulate`]. The adapters.
    Backend,
    /// A host `Vec<f32>`, fed by [`AdapterBank::accumulate_host`]. The
    /// host entries (D17, span head).
    Host,
}

/// One entry the bank holds a gradient for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BankEntry {
    pub name: String,
    pub shape: Vec<usize>,
    pub residence: Residence,
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
    /// A backend gradient was fed to a host slot, or the reverse.
    Residence {
        entry: usize,
        name: String,
        slot: Residence,
    },
    /// `Add` into a slot that holds no gradient.
    AddWithoutBase { entry: usize, name: String },
    /// A gradient was asked of a slot that holds none.
    Unfilled { entry: usize, name: String },
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
            Self::Residence { entry, name, slot } => write!(
                f,
                "bank: entry {entry} ({name}): the slot is {slot:?}-resident, \
                 fed through the other accumulate"
            ),
            Self::AddWithoutBase { entry, name } => write!(
                f,
                "bank: entry {entry} ({name}): Add into an empty slot (no Overwrite this step)"
            ),
            Self::Unfilled { entry, name } => {
                write!(f, "bank: entry {entry} ({name}): no gradient accumulated")
            }
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
    Backend(Tensor),
    Host(Vec<f32>),
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

    /// Fold one sequence's gradient of backend-resident entry `entry` into
    /// the bank. `grad` is taken by value: after an `Overwrite` the bank is
    /// its only owner, so a later in-place add cannot write through to a
    /// handle the caller kept.
    pub fn accumulate<B: Backend + ?Sized>(
        &mut self,
        backend: &B,
        entry: usize,
        mode: BankMode,
        grad: Tensor,
    ) -> Result<(), BankError> {
        let e = self.entry(entry)?;
        let name = e.name.clone();
        if e.residence != Residence::Backend {
            return Err(BankError::Residence {
                entry,
                name,
                slot: e.residence,
            });
        }
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
                self.slots[entry] = Slot::Backend(grad);
                Ok(())
            }
            BankMode::Add => {
                let Slot::Backend(mut acc) = std::mem::replace(&mut self.slots[entry], Slot::Empty)
                else {
                    return Err(BankError::AddWithoutBase { entry, name });
                };
                match backend.accumulate_grad(&mut acc, &grad) {
                    Ok(()) => {
                        self.slots[entry] = Slot::Backend(acc);
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

    /// [`AdapterBank::accumulate`] for a host entry: `grad` is row-major
    /// with the entry's element count.
    pub fn accumulate_host(
        &mut self,
        entry: usize,
        mode: BankMode,
        grad: &[f32],
    ) -> Result<(), BankError> {
        let e = self.entry(entry)?;
        let name = e.name.clone();
        if e.residence != Residence::Host {
            return Err(BankError::Residence {
                entry,
                name,
                slot: e.residence,
            });
        }
        let want = elements(&e.name, &e.shape)?;
        if grad.len() != want {
            let err = BankError::Shape {
                entry,
                name,
                expected: e.shape.clone(),
                got: vec![grad.len()],
            };
            self.refuse_overwrite(entry, mode);
            return Err(err);
        }
        match mode {
            BankMode::Overwrite => {
                self.slots[entry] = Slot::Host(grad.to_vec());
                Ok(())
            }
            BankMode::Add => match &mut self.slots[entry] {
                Slot::Host(acc) => {
                    for (a, g) in acc.iter_mut().zip(grad) {
                        *a += *g;
                    }
                    Ok(())
                }
                _ => Err(BankError::AddWithoutBase { entry, name }),
            },
        }
    }

    /// The summed gradient of `entry` as a tensor: a share of the backend
    /// slot's allocation, or a new host tensor charged to `budget` for a
    /// host slot.
    pub fn grad(&self, entry: usize, budget: &Budget) -> Result<Tensor, BankError> {
        let e = self.entry(entry)?;
        match &self.slots[entry] {
            Slot::Empty => Err(BankError::Unfilled {
                entry,
                name: e.name.clone(),
            }),
            Slot::Backend(t) => Ok(t.clone()),
            Slot::Host(v) => {
                Tensor::from_f32(v, &e.shape, budget).map_err(|source| BankError::Core {
                    entry,
                    name: e.name.clone(),
                    source,
                })
            }
        }
    }

    /// Every entry's summed gradient, in entry order. The first slot that
    /// holds none is the error.
    pub fn grads(&self, budget: &Budget) -> Result<Vec<Tensor>, BankError> {
        (0..self.entries.len())
            .map(|i| self.grad(i, budget))
            .collect()
    }

    /// The host slot's values, or `None` for an empty or backend slot.
    pub fn host_values(&self, entry: usize) -> Option<&[f32]> {
        match self.slots.get(entry)? {
            Slot::Host(v) => Some(v),
            _ => None,
        }
    }
}

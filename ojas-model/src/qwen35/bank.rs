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
//!
//! # Zero marker
//!
//! [`AdapterBank::accumulate_zero`] folds an all-zero gradient without
//! allocating one: an `Overwrite` leaves the slot holding a marker (a step
//! whose rows are all at scale 0 has no gradient, and a full fine-tune's `[V, d]`
//! embedding gradient would otherwise be materialised to hold zeros), an `Add`
//! changes nothing. An `Add` of a real gradient into a marker slot makes the
//! slot that gradient. A marker counts as filled. The `_on` readers
//! ([`AdapterBank::grad_on`], [`AdapterBank::grads_on`]) materialise a marker
//! as a zero tensor on the backend they are given, transiently and not stored;
//! the backend-free readers ([`AdapterBank::grad`], [`AdapterBank::grads`],
//! [`AdapterBank::take_grads`]) refuse a marker ([`BankError::ZeroMarker`]).
//!
//! # One read per gradient per step
//!
//! [`AdapterBank::scan`] reads each stored gradient to the host once and keeps
//! only a summary of it (the running squared norm, the largest finite
//! magnitude, whether a non-finite value was seen). The summary is dropped by
//! every operation that changes a slot, so it never describes a sum the bank
//! no longer holds. The squared norm
//! ([`AdapterBank::sq_norm_scanned`]) and the optimiser's finiteness check
//! ([`super::optim::check_all`]) both read it, so a step that asks for the norm
//! and then updates reads each gradient once, not twice.

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
    /// A backend-free reader met a zero marker ([`AdapterBank::accumulate_zero`]);
    /// [`AdapterBank::grad_on`] materialises it.
    ZeroMarker { entry: usize, name: String },
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
            Self::ZeroMarker { entry, name } => write!(
                f,
                "bank: entry {entry} ({name}): the slot holds a zero marker, not a tensor (read it \
                 with grad_on)"
            ),
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
    /// An all-zero gradient, held without an allocation.
    Zero,
    Filled(Tensor),
}

/// What [`AdapterBank::scan`] read from one stored gradient.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradScan {
    /// The squared L2 norm of every gradient up to and including this one
    /// (the running f64 accumulator of [`AdapterBank::sq_norm`]); a non-finite
    /// value adds nothing.
    pub running_sq: f64,
    /// The largest `|g|` among this gradient's finite values.
    pub max_abs: f32,
    /// A NaN or infinity was seen.
    pub non_finite: bool,
}

/// Fold `values` into the running accumulator `running`, widening to f64,
/// elements in row-major order.
fn scan_values(values: &[f32], running: f64) -> GradScan {
    let mut acc = running;
    let mut max_abs = 0.0f32;
    let mut non_finite = false;
    for &x in values {
        if !x.is_finite() {
            non_finite = true;
            continue;
        }
        let w = f64::from(x);
        acc += w * w;
        max_abs = max_abs.max(x.abs());
    }
    GradScan {
        running_sq: acc,
        max_abs,
        non_finite,
    }
}

/// The per-step gradient sums, keyed by entry index (the order of the
/// entries given to [`AdapterBank::new`]).
#[derive(Debug)]
pub struct AdapterBank {
    entries: Vec<BankEntry>,
    slots: Vec<Slot>,
    /// [`AdapterBank::scan`]'s result for the slots as they are now; every
    /// change to a slot drops it.
    scans: Option<Vec<GradScan>>,
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
        Ok(Self {
            entries,
            slots,
            scans: None,
        })
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
        self.scans = None;
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
        self.scans = None;
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
                let mut acc = match std::mem::replace(&mut self.slots[entry], Slot::Empty) {
                    Slot::Filled(acc) => acc,
                    // 0 + g is g.
                    Slot::Zero => {
                        self.slots[entry] = Slot::Filled(grad);
                        return Ok(());
                    }
                    Slot::Empty => return Err(BankError::AddWithoutBase { entry, name }),
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

    /// Fold an all-zero gradient into entry `entry` without allocating one,
    /// under the same rules as [`AdapterBank::accumulate`] (the gradient is
    /// the entry's shape and `F32` by construction): `Overwrite` leaves a zero
    /// marker, `Add` changes nothing and still refuses an empty slot.
    pub fn accumulate_zero(&mut self, entry: usize, mode: BankMode) -> Result<(), BankError> {
        self.scans = None;
        let name = self.entry(entry)?.name.clone();
        match mode {
            BankMode::Overwrite => {
                self.slots[entry] = Slot::Zero;
                Ok(())
            }
            BankMode::Add => match self.slots[entry] {
                Slot::Empty => Err(BankError::AddWithoutBase { entry, name }),
                Slot::Zero | Slot::Filled(_) => Ok(()),
            },
        }
    }

    /// Whether `entry` holds a zero marker rather than a tensor.
    pub fn is_zero(&self, entry: usize) -> Result<bool, BankError> {
        self.entry(entry)?;
        Ok(matches!(self.slots[entry], Slot::Zero))
    }

    /// Like [`AdapterBank::grad`], with a zero marker materialised as a zero
    /// tensor placed by `backend` (not stored: the slot stays a marker).
    pub fn grad_on<B: Backend + ?Sized>(
        &self,
        backend: &B,
        entry: usize,
    ) -> Result<Tensor, BankError> {
        let e = self.entry(entry)?;
        match &self.slots[entry] {
            Slot::Empty => Err(BankError::Unfilled {
                entry,
                name: e.name.clone(),
            }),
            Slot::Filled(t) => Ok(t.clone()),
            Slot::Zero => Tensor::zeros(&e.shape, DType::F32, backend.budget())
                .and_then(|z| backend.upload(&z))
                .map_err(|source| BankError::Core {
                    entry,
                    name: e.name.clone(),
                    source,
                }),
        }
    }

    /// [`AdapterBank::grads`] with markers materialised ([`AdapterBank::grad_on`]).
    pub fn grads_on<B: Backend + ?Sized>(&self, backend: &B) -> Result<Vec<Tensor>, BankError> {
        (0..self.entries.len())
            .map(|i| self.grad_on(backend, i))
            .collect()
    }

    /// Read every stored gradient to the host once and summarise it
    /// ([`GradScan`], entries in order). A marker is read as zeros without a
    /// readback. The result is kept until a slot changes, so a second call
    /// (the optimiser's, after the norm's) reads nothing. The first empty slot
    /// is [`BankError::Unfilled`]; a non-finite value is reported in the
    /// summary, not as an error.
    pub fn scan<B: Backend + ?Sized>(&mut self, backend: &B) -> Result<&[GradScan], BankError> {
        if self.scans.is_none() {
            let mut out = Vec::with_capacity(self.slots.len());
            let mut running = 0.0f64;
            for (entry, (e, slot)) in self.entries.iter().zip(&self.slots).enumerate() {
                let one =
                    match slot {
                        Slot::Empty => {
                            return Err(BankError::Unfilled {
                                entry,
                                name: e.name.clone(),
                            })
                        }
                        Slot::Zero => scan_values(&[], running),
                        Slot::Filled(t) => {
                            let values = backend.download(t).and_then(|h| h.to_f32_vec()).map_err(
                                |source| BankError::Core {
                                    entry,
                                    name: e.name.clone(),
                                    source,
                                },
                            )?;
                            scan_values(&values, running)
                        }
                    };
                running = one.running_sq;
                out.push(one);
            }
            self.scans = Some(out);
        }
        Ok(self.scans.as_deref().unwrap_or(&[]))
    }

    /// [`AdapterBank::sq_norm`] through [`AdapterBank::scan`]: the same value
    /// and the same refusals, and the readback is shared with the optimiser's
    /// finiteness check.
    pub fn sq_norm_scanned<B: Backend + ?Sized>(&mut self, backend: &B) -> Result<f64, BankError> {
        let scans = self.scan(backend)?;
        if let Some(entry) = scans.iter().position(|s| s.non_finite) {
            return Err(BankError::NonFinite {
                entry,
                name: self.entries[entry].name.clone(),
            });
        }
        Ok(scans.last().map_or(0.0, |s| s.running_sq))
    }

    /// The summed gradient of `entry`: a share of the slot's allocation.
    pub fn grad(&self, entry: usize) -> Result<Tensor, BankError> {
        let e = self.entry(entry)?;
        match &self.slots[entry] {
            Slot::Empty => Err(BankError::Unfilled {
                entry,
                name: e.name.clone(),
            }),
            Slot::Zero => Err(BankError::ZeroMarker {
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
    /// complete: the first empty slot is [`BankError::Unfilled`], a zero marker
    /// is [`BankError::ZeroMarker`] (use [`AdapterBank::grads_on`] then
    /// [`AdapterBank::clear`]), and on either refusal nothing has moved.
    pub fn take_grads(&mut self) -> Result<Vec<Tensor>, BankError> {
        for (entry, (e, slot)) in self.entries.iter().zip(&self.slots).enumerate() {
            match slot {
                Slot::Empty => {
                    return Err(BankError::Unfilled {
                        entry,
                        name: e.name.clone(),
                    })
                }
                Slot::Zero => {
                    return Err(BankError::ZeroMarker {
                        entry,
                        name: e.name.clone(),
                    })
                }
                Slot::Filled(_) => {}
            }
        }
        self.scans = None;
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
            let values = match slot {
                Slot::Empty => {
                    return Err(BankError::Unfilled {
                        entry,
                        name: e.name.clone(),
                    })
                }
                Slot::Zero => continue,
                Slot::Filled(t) => {
                    backend
                        .download(t)
                        .and_then(|h| h.to_f32_vec())
                        .map_err(|source| BankError::Core {
                            entry,
                            name: e.name.clone(),
                            source,
                        })?
                }
            };
            let one = scan_values(&values, acc);
            if one.non_finite {
                return Err(BankError::NonFinite {
                    entry,
                    name: e.name.clone(),
                });
            }
            acc = one.running_sq;
        }
        Ok(acc)
    }
}

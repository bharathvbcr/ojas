//! The Qwen3.5 Tape tower as a training loop's [`StepProvider`].
//!
//! [`TapeStepProvider`] owns everything one LoRA (or, to compare against the
//! tessl provider, full) fine-tune step touches: the backend, the frozen base
//! loaded through [`load_hf_pinned`] (so its sha256 is checked against the
//! run's [`BaseIdentity`]), the trainable leaves with their AdamW moments, the
//! f32 gradient [`AdapterBank`], the step count and the RoPE tables uploaded
//! once at the run's widest bucket (each forward narrows them to its own
//! sequence, [`crate::qwen35::forward_hidden_lora`]).
//!
//! # One sequence per call
//!
//! [`StepProvider::accumulate`] runs one sequence on its own tape: bind the
//! base frozen, the adapters as tape leaves, run the tower to the final norm,
//! gather rows, back-propagate into the trainable leaves, and drop the tape.
//! Each leaf's gradient goes to the bank under the loop's [`BankMode`], so a
//! step's device peak is its widest sequence's, never the batch's. The
//! returned value is the unscaled sum of the row cross-entropies (the fused
//! cross-entropy's mean times the row count).
//!
//! The backward has one root. Rows alone: the mean cross-entropy seeded with
//! `scale * rows` ([`ojas_autograd::Tape::backward_seeded`]), so the bank gets
//! `scale` times the gradient of the sum. An outside loss alone (the span
//! head): `linear(reshape(H, [1, n d]), frozen(reshape(dh, [1, n d])))`, a dot
//! product whose gradient at `H` is exactly `dh`, which the provider does not
//! scale. Both: `scale_constant * ce_mean + that dot product`, one backward,
//! no change to the autograd core; the combined arm still refuses every scale
//! the rows-only seed would (a subnormal one included), by running the tape's
//! own seed check before its backward. Scale 0 takes the value from
//! [`ojas_autograd::Tape::linear_cross_entropy_value`] (no node, no `[V, d]`
//! gradient), and with no outside loss runs no backward at all: the bank gets
//! zero markers ([`AdapterBank::accumulate_zero`]: no tensor is allocated, so a
//! full fine-tune's `[V, d]` embedding gradient is never materialised for a
//! step that has none), and an `Overwrite` still replaces the previous step's
//! sums. The update materialises a marker as zeros for its own duration only.
//!
//! D17 and the span head are not here. The loop's objective owns them and
//! checkpoints them itself; this provider's entries are the adapters.
//!
//! # Entries and leaves
//!
//! The loop's table has one entry per Hugging Face tensor of a LoRA site: `A`
//! `[r, in]` and the fused `B` `[out, r]` ([`LoraTarget::fuse_b`], PEFT's
//! shape). The model holds a fused `B` as parts (one per output block), each a
//! separate tape leaf. AdamW is elementwise, so the update, the gradient bank
//! and the squared norm run per leaf with the entry's learning rate and
//! weight decay, which is the fused entry's update exactly. Values and moments
//! are fused only to be read ([`StepProvider::read_parameters`]) or saved.
//!
//! # Checkpoints
//!
//! [`StepProvider::save_state_with`] writes the adapter checkpoint of
//! `docs/adapter-checkpoint.md` (values and moments of every LoRA entry, the
//! step, the data-order state from the loop's context, the base's identity)
//! into `<dir>/provider`; [`StepProvider::load_state_with`] reads it back
//! under the same identity, rank and alpha, and refuses a checkpoint whose
//! step, seed, epoch or consumed digest is not the loop's. The plain
//! `save_state` and `load_state` refuse: the context is required. A LoRA
//! dropout other than 0 is refused at open, because no dropout stream is
//! checkpointed.
//!
//! # The device
//!
//! On a backend that is not the CPU, [`StepProvider::accumulate`] refuses until
//! [`TapeStepProvider::preflight`] has passed: it plans the widest bucket
//! against the calibration, then runs one real step at that width under a child
//! budget (through [`DeviceHooks::rebind`]) and refuses with the shape if it
//! does not fit.
//!
//! The tower is generic over [`ojas_core::Backend`], and the trait has no
//! method for the two things a device-resident training step needs beyond
//! it: marking a frozen weight so the device refuses any write to it, and
//! rebinding the backend to a smaller [`Budget`] once a plan is known. Both
//! are inherent methods of the concrete backend (`MetalBackend::freeze_checked`
//! and `MetalBackend::with_budget`). A trait method would have to be
//! forwarded by every wrapper backend (autocast, tracing, test doubles), and
//! a wrapper that forgets one silently drops the guarantee (bug class
//! ft-fe7dcee6); a pair of closures the caller owns cannot be forgotten by a
//! wrapper, because no wrapper sits between them and the concrete backend.
//!
//! A Metal caller passes
//!
//! ```ignore
//! DeviceHooks {
//!     freeze: Some(Box::new(|m, t| m.freeze_checked(t))),
//!     rebind: Some(Box::new(|m, b| m.with_budget(b))),
//! }
//! ```
//!
//! and a CPU caller uses [`DeviceHooks::default`], which has neither: a
//! missing hook means the backend has no such notion, not that the step
//! skipped it. `freeze` is applied once to each frozen base tensor when the
//! LoRA provider opens, then the backend is synchronised.
//!
//! The pre-flight's dry run measures only what the backend `rebind` returns
//! charges to the budget it was handed. An allocation made on any other budget
//! (the original backend's, a hook that ignores its argument) is invisible to
//! the measured peak, and the pre-flight still passes: the hook is trusted to
//! rebind, and cannot be checked for it here.
//!
//! # The update
//!
//! [`StepProvider::adamw_step`] judges the whole batch before the bank gives
//! anything up: [`crate::qwen35::optim::check_all`] (the one owner of the
//! refusal rule) runs on the bank's gradients, scaled as the clip scale will
//! scale them, and only an accepted batch empties the bank and is applied. A
//! refused update therefore leaves every gradient sum, parameter and moment as
//! it was and the step count unchanged. The norm
//! ([`StepProvider::grad_sq_norm`]) and that check share one host readback of
//! each gradient ([`AdapterBank::scan`]), so a step costs one download per
//! gradient, not two. A refusal after the update has begun (the backend
//! refusing one entry, [`OptimError::Applied`]) has already consumed the
//! sums: those are not recoverable.

use std::collections::BTreeSet;
use std::ops::Range;
use std::path::{Path, PathBuf};

use ojas_autograd::{Tape, Var};
use ojas_core::{AdamWConfig, Backend, BackendId, Budget, CeChunk, DType, OjasError, Tensor};
use ojas_data::CounterRng;
use ojas_train::lappi::step::{
    check_per_entry, AdamWHyper, BankMode, CheckpointContext, HiddenGrad, ParamSpec, RowTargets,
    SequenceJob, StepError, StepProvider, StorageDtype,
};

use crate::block::ActivationCheckpoint;
use crate::lora::{LoraConfig, LoraSite};
use crate::preflight::{self, Calibration, PreflightError};
use crate::qwen35::adapters::{
    lora_a_name, lora_b_name, read_adapter_checkpoint, write_adapter_checkpoint, AdapterCheckpoint,
    AdapterEntry, AdapterError, BaseIdentity, DataState, Expected, ExpectedEntry, ADAPTERS_FILE,
    MOMENTS_FILE,
};
use crate::qwen35::bank::{AdapterBank, BankEntry, BankError};
use crate::qwen35::optim::{apply_all, check_all, AdamWEntry, GradKnown, OptimError};
use crate::qwen35::{
    bind_frozen, forward_hidden_lora, fuse_grads, hf_tensors, load_hf_pinned, HfKind, LoraForward,
    LoraTarget, Qwen35Lora, Qwen35Mixer, Qwen35Params, Qwen35Spec, Qwen35Tables,
};

/// Freeze a weight on the device.
pub type FreezeHook<B> = Box<dyn Fn(&B, &Tensor) -> Result<(), OjasError>>;

/// Both AdamW moments of every trainable entry, in table order: `(first, second)`.
pub type HostMoments = Vec<(Vec<f32>, Vec<f32>)>;

/// Another handle on the backend that charges the given budget.
pub type RebindHook<B> = Box<dyn Fn(&B, Budget) -> B>;

/// The optional device operations a step provider needs from a concrete
/// backend `B`; see the module docs.
pub struct DeviceHooks<B> {
    /// Mark `tensor` frozen on `backend` so a write to it is refused.
    pub freeze: Option<FreezeHook<B>>,
    /// A handle on `backend`'s device charging a different budget.
    pub rebind: Option<RebindHook<B>>,
}

impl<B> Default for DeviceHooks<B> {
    fn default() -> Self {
        Self {
            freeze: None,
            rebind: None,
        }
    }
}

/// What both constructors need: the base, the shape of a run and its memory
/// policy.
#[derive(Clone, Debug)]
pub struct OpenConfig {
    pub spec: Qwen35Spec,
    /// The Hugging Face tensor prefix (`"model."`, or `"model.language_model."`).
    pub prefix: String,
    /// The single-file base checkpoint.
    pub base_path: PathBuf,
    /// The base the run is pinned to: [`load_hf_pinned`] refuses any other.
    pub base_id: BaseIdentity,
    /// Storage of the linear weights, [`DType::F32`] or [`DType::Bf16`].
    pub linears: DType,
    /// Storage of the tied embedding table, [`DType::F32`] or [`DType::Bf16`].
    pub table: DType,
    /// The widest sequence in tokens: the tables are built for it.
    pub widest: usize,
    pub chunk: CeChunk,
    pub activations: ActivationCheckpoint,
}

/// The adapters of a LoRA run.
#[derive(Clone, Debug)]
pub struct LoraInit {
    /// Rank, alpha and dropout; the dropout must be 0.
    pub config: LoraConfig,
    /// Every adapted linear, in [`Qwen35Lora::sites`] order
    /// ([`crate::qwen35::lora_targets`]).
    pub targets: Vec<LoraTarget>,
    /// One site per target and none other, host or device tensors, all `F32`.
    pub adapters: Qwen35Lora<LoraSite<Tensor>>,
}

enum Trainable {
    Lora {
        base: Qwen35Params<Tensor>,
        config: LoraConfig,
        targets: Vec<LoraTarget>,
    },
    /// Every base parameter is trained. Only the structure is kept here: the
    /// live values are the provider's leaves.
    Full { template: Qwen35Params<()> },
}

#[derive(Clone, Copy, Debug)]
enum Role {
    LoraA,
    LoraB { target: usize },
    Hf,
}

/// A trainable table entry and the consecutive leaves that make it up.
struct EntryMap {
    leaves: Range<usize>,
    role: Role,
}

/// Why a sequence's run failed: the backend or the caller's callback.
enum Fail {
    Core(OjasError),
    Step(StepError),
}

impl From<OjasError> for Fail {
    fn from(e: OjasError) -> Self {
        Self::Core(e)
    }
}

impl From<StepError> for Fail {
    fn from(e: StepError) -> Self {
        Self::Step(e)
    }
}

impl Fail {
    fn into_step(self) -> StepError {
        match self {
            Self::Core(e) => core_err(e),
            Self::Step(e) => e,
        }
    }

    /// For a caller that must keep an [`OjasError`] (the pre-flight's dry
    /// run reads [`OjasError::CapacityExceeded`]).
    fn into_ojas(self) -> OjasError {
        match self {
            Self::Core(e) => e,
            Self::Step(e) => OjasError::OutOfRange {
                op: "TapeStepProvider::preflight",
                detail: e.to_string(),
            },
        }
    }
}

pub(crate) fn core_err(e: OjasError) -> StepError {
    StepError::Backend(e.to_string())
}

fn contract(detail: impl Into<String>) -> StepError {
    StepError::Contract(detail.into())
}

/// The refusals that depend on the model's config (`Qwen35Spec::reserved_token_ids` and
/// `max_positions`), the one owner both [`TapeStepProvider::accumulate`]'s path and
/// [`TapeStepProvider::preflight`]'s dry run go through, before any gather, loss or tape
/// write: a token id in the reserved set, as an input or as a row target (a text-only run
/// must neither embed a vision special token nor train toward one), and a sequence longer
/// than `max_positions` when the config sets one.
fn check_config_limits(spec: &Qwen35Spec, job: &SequenceJob<'_>) -> Result<(), StepError> {
    if let Some(id) = job
        .tokens
        .iter()
        .find(|id| spec.reserved_token_ids.contains(id))
    {
        return Err(contract(format!(
            "token {id} is reserved (a vision special token); a text-only run refuses it"
        )));
    }
    if let Some(id) = job
        .rows
        .targets
        .iter()
        .find(|id| spec.reserved_token_ids.contains(id))
    {
        return Err(contract(format!(
            "row target {id} is reserved (a vision special token); a text-only run cannot train \
             toward it"
        )));
    }
    if let Some(max) = spec.max_positions {
        if job.tokens.len() > max {
            return Err(contract(format!(
                "a sequence of {} tokens is longer than max_position_embeddings {max}",
                job.tokens.len()
            )));
        }
    }
    Ok(())
}

/// The pre-flight dry run's tokens: `bucket` of them, cycling through the ids the config does
/// not reserve, so the measured step is one a real run could take.
fn dry_run_tokens(spec: &Qwen35Spec, bucket: usize) -> Result<Vec<u32>, StepError> {
    let vocab = u32::try_from(spec.vocab)
        .map_err(|_| contract(format!("vocabulary {} does not fit u32", spec.vocab)))?;
    let allowed: Vec<u32> = (0..vocab)
        .filter(|id| !spec.reserved_token_ids.contains(id))
        .collect();
    if allowed.is_empty() {
        return Err(contract(
            "every token id in the vocabulary is reserved; there is nothing to dry-run with",
        ));
    }
    Ok((0..bucket).map(|i| allowed[i % allowed.len()]).collect())
}

fn adapter_err(e: AdapterError) -> StepError {
    match e {
        AdapterError::Unreadable { .. } => StepError::Io(e.to_string()),
        AdapterError::Core(_) => StepError::Backend(e.to_string()),
        other => StepError::Contract(other.to_string()),
    }
}

pub(crate) fn bank_err(e: BankError) -> StepError {
    match e {
        BankError::Core { .. } | BankError::NonFinite { .. } => StepError::Backend(e.to_string()),
        other => StepError::Contract(other.to_string()),
    }
}

fn bank_to_ojas(e: BankError) -> OjasError {
    match e {
        BankError::Core { source, .. } => source,
        other => OjasError::OutOfRange {
            op: "TapeStepProvider::preflight",
            detail: other.to_string(),
        },
    }
}

fn preflight_err(e: PreflightError) -> StepError {
    match e {
        PreflightError::Step(source) => core_err(source),
        other => StepError::Contract(other.to_string()),
    }
}

fn storage_of(dtype: DType) -> Result<StorageDtype, StepError> {
    match dtype {
        DType::F32 => Ok(StorageDtype::F32),
        DType::Bf16 => Ok(StorageDtype::Bf16),
        other => Err(contract(format!(
            "a base stored as {other:?} has no storage dtype in the parameter table"
        ))),
    }
}

fn encode_rng(seed: u64, epoch: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(16);
    out.extend_from_slice(&seed.to_le_bytes());
    out.extend_from_slice(&epoch.to_le_bytes());
    out
}

/// What one sequence's run reads, borrowed from the provider.
struct Seq<'a> {
    spec: &'a Qwen35Spec,
    tables: &'a Qwen35Tables,
    trainable: &'a Trainable,
    leaves: &'a [Tensor],
    activations: ActivationCheckpoint,
    chunk: CeChunk,
}

/// One leaf's gradient out of a sequence.
enum LeafGrad {
    /// `F32`, placed on the backend.
    Dense(Tensor),
    /// All zeros, not allocated ([`AdapterBank::accumulate_zero`]).
    Zero,
}

/// Fold one leaf's gradient into `bank`.
fn fold_grad<K: Backend>(
    bank: &mut AdapterBank,
    backend: &K,
    entry: usize,
    mode: BankMode,
    grad: LeafGrad,
) -> Result<(), BankError> {
    match grad {
        LeafGrad::Dense(g) => bank.accumulate(backend, entry, mode, g),
        LeafGrad::Zero => bank.accumulate_zero(entry, mode),
    }
}

struct SeqOut {
    /// The unscaled sum of the row cross-entropies.
    value: f64,
    /// One gradient per leaf, in leaf order.
    grads: Vec<LeafGrad>,
}

fn leaf_at(leaves: &[Tensor], at: usize) -> Result<&Tensor, OjasError> {
    leaves.get(at).ok_or_else(|| OjasError::OutOfRange {
        op: "TapeStepProvider",
        detail: format!("leaf {at} of {}", leaves.len()),
    })
}

/// The loss tensor's first element, after a synchronisation so a deferred
/// device fault surfaces here rather than as a silent NaN.
fn scalar<K: Backend>(backend: &K, t: &Tensor) -> Result<f32, OjasError> {
    backend.sync()?;
    let host = backend.download(t)?;
    host.to_f32_vec()?
        .first()
        .copied()
        .ok_or_else(|| OjasError::OutOfRange {
            op: "TapeStepProvider",
            detail: "the loss tensor is empty".into(),
        })
}

/// Rows `positions` of `table` `[T, d]`, `[n, d]`, with a scatter-add backward.
fn gather<B: Backend>(
    tape: &mut Tape<B>,
    table: Var,
    positions: &[u32],
    budget: &Budget,
) -> Result<Var, OjasError> {
    let ids = Tensor::from_u32(positions, &[positions.len()], budget)?;
    tape.embedding(table, ids)
}

/// Refuse a row seed that [`Tape::backward_seeded`] would refuse.
///
/// The rows-only arm seeds its backward with `seed`, so the tape's own seed
/// check applies. The combined arm multiplies `seed` into the CE term and
/// seeds the root with 1, which would let a subnormal scale through. This
/// runs the tape's check, not a copy of it, on a one-leaf scratch tape, so
/// both arms refuse the same scales with the same error. It runs before the
/// combined backward and writes no gradient anywhere real.
fn refuse_unseedable<K: Backend>(backend: &K, seed: f32, budget: &Budget) -> Result<(), OjasError> {
    let mut probe = Tape::new(backend);
    let leaf = probe.leaf(Tensor::from_f32(&[0.0], &[1], budget)?)?;
    probe.backward_seeded(leaf, seed)
}

/// One sequence forward and backward on a fresh tape over `backend`, which is
/// dropped before this returns. See the module docs for the backward's root.
fn run_sequence<K: Backend>(
    backend: &K,
    seq: &Seq<'_>,
    job: &SequenceJob<'_>,
    external: Option<&mut HiddenGrad<'_>>,
) -> Result<SeqOut, Fail> {
    let spec = seq.spec;
    let (t, d) = (job.tokens.len(), spec.hidden);
    let budget = backend.budget().clone();
    let ids = Tensor::from_u32(job.tokens, &[1, t], &budget)?;
    let mut tape = Tape::new(backend);

    let (params, lora_fwd, leaf_vars) = match seq.trainable {
        Trainable::Lora {
            base,
            config,
            targets,
        } => {
            let params = bind_frozen(&mut tape, base)?;
            let mut sites = Qwen35Lora::<LoraSite<Var>>::empty(spec);
            let mut vars = Vec::with_capacity(seq.leaves.len());
            let mut at = 0usize;
            for target in targets {
                let a = tape.leaf(leaf_at(seq.leaves, at)?.clone())?;
                at += 1;
                vars.push(a);
                let mut b = Vec::with_capacity(target.out_parts.len());
                for _ in &target.out_parts {
                    let part = tape.leaf(leaf_at(seq.leaves, at)?.clone())?;
                    at += 1;
                    vars.push(part);
                    b.push(part);
                }
                let slot = sites
                    .slot_mut(target.layer, target.kind)
                    .ok_or_else(|| contract(format!("{}: no such LoRA slot", target.name)))?;
                *slot = Some(LoraSite { a, b });
            }
            let mut rng = CounterRng::new(0);
            let fwd = LoraForward::new(&mut tape, sites, config, 1, t, &mut rng, &budget)?;
            (params, Some(fwd), vars)
        }
        Trainable::Full { template } => {
            let mut leaves = seq.leaves.iter();
            let bound = template.try_map(&mut |_| {
                let value = leaves.next().ok_or_else(|| OjasError::OutOfRange {
                    op: "TapeStepProvider",
                    detail: "fewer leaves than parameters".into(),
                })?;
                tape.leaf(value.clone())
            })?;
            let vars: Vec<Var> = bound.values().into_iter().copied().collect();
            (bound, None, vars)
        }
    };

    let h = forward_hidden_lora(
        &mut tape,
        spec,
        &params,
        lora_fwd.as_ref(),
        &ids,
        seq.tables,
        seq.activations,
    )?;
    let table = tape.reshape(h, &[t, d])?;

    let n = job.rows.positions.len();
    let scale = job.rows.scale;
    let mut ce: Option<Var> = None;
    let mut value = 0.0f64;
    if n > 0 {
        let x = gather(&mut tape, table, job.rows.positions, &budget)?;
        let targets = Tensor::from_u32(job.rows.targets, &[n], &budget)?;
        if scale > 0.0 {
            let loss = tape.linear_cross_entropy(x, params.embed, targets, None, seq.chunk)?;
            value = f64::from(scalar(backend, tape.value(loss)?)?) * n as f64;
            ce = Some(loss);
        } else {
            let loss =
                tape.linear_cross_entropy_value(x, params.embed, &targets, None, seq.chunk)?;
            value = f64::from(scalar(backend, &loss)?) * n as f64;
        }
    }

    let ext: Option<Var> = if job.hidden_at.is_empty() {
        None
    } else {
        let callback = external.ok_or_else(|| {
            contract("hidden positions were requested without a gradient callback")
        })?;
        let m = job.hidden_at.len();
        let hv = gather(&mut tape, table, job.hidden_at, &budget)?;
        backend.sync()?;
        let host = backend.download(tape.value(hv)?)?.to_f32_vec()?;
        let dh = callback(job.hidden_at, &host)?;
        if dh.len() != m * d {
            return Err(contract(format!(
                "the outside loss returned {} values for {m} positions of width {d}",
                dh.len()
            ))
            .into());
        }
        if let Some(i) = dh.iter().position(|x| !x.is_finite()) {
            return Err(contract(format!(
                "the outside loss's gradient is {} at element {i}",
                dh[i]
            ))
            .into());
        }
        let dh = Tensor::from_f32(&dh, &[1, m * d], &budget)?;
        let w = tape.frozen_leaf(dh)?;
        let x = tape.reshape(hv, &[1, m * d])?;
        Some(tape.linear(x, w)?)
    };

    let seed = scale * n as f32;
    let backward = match (ce, ext) {
        (None, None) => false,
        (Some(c), None) => {
            tape.backward_seeded(c, seed)?;
            true
        }
        (None, Some(e)) => {
            tape.backward(e)?;
            true
        }
        (Some(c), Some(e)) => {
            refuse_unseedable(backend, seed, &budget)?;
            let k = tape.frozen_leaf(Tensor::from_f32(&[seed], &[1, 1], &budget)?)?;
            let c = tape.reshape(c, &[1, 1])?;
            let scaled = tape.mul(k, c)?;
            let root = tape.add(scaled, e)?;
            tape.backward(root)?;
            true
        }
    };

    let grads = if backward {
        leaf_vars
            .iter()
            .map(|v| {
                tape.take_grad(*v)
                    .map(LeafGrad::Dense)
                    .ok_or_else(|| OjasError::OutOfRange {
                        op: "TapeStepProvider",
                        detail: format!("leaf {} has no gradient after the backward", v.0),
                    })
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        seq.leaves.iter().map(|_| LeafGrad::Zero).collect()
    };
    backend.sync()?;
    drop(tape);
    Ok(SeqOut { value, grads })
}

/// The Qwen3.5 Tape tower, driven one sequence at a time. See the module docs.
pub struct TapeStepProvider<B: Backend> {
    backend: B,
    hooks: DeviceHooks<B>,
    spec: Qwen35Spec,
    prefix: String,
    tables: Qwen35Tables,
    widest: usize,
    trainable: Trainable,
    /// Every trainable tensor, in leaf order.
    leaves: Vec<Tensor>,
    /// AdamW's first and second moments of each leaf.
    moments: Vec<(Tensor, Tensor)>,
    /// The entry each leaf belongs to.
    leaf_entry: Vec<usize>,
    leaf_names: Vec<String>,
    /// The trainable entries and their leaves, in [`StepProvider::parameters`]
    /// order.
    entries: Vec<EntryMap>,
    /// Frozen base entries first, then the trainable entries.
    table: Vec<ParamSpec>,
    bank: AdapterBank,
    step: u64,
    activations: ActivationCheckpoint,
    chunk: CeChunk,
    base_id: BaseIdentity,
    preflighted: bool,
}

/// How many leaves each Hugging Face tensor of [`hf_tensors`] is made of:
/// `in_proj_qkv` and `conv1d` are three parts and `q_proj` two.
fn full_leaf_counts(spec: &Qwen35Spec) -> Vec<usize> {
    let mut counts = vec![1];
    for kind in &spec.layers {
        counts.push(1);
        match kind {
            Qwen35Mixer::GatedDeltaNet => counts.extend([3, 3, 1, 1, 1, 1, 1, 1, 1]),
            Qwen35Mixer::Attention => counts.extend([2, 1, 1, 1, 1, 1]),
        }
        counts.extend([1, 1, 1, 1]);
    }
    counts.push(1);
    counts
}

impl<B: Backend> TapeStepProvider<B> {
    /// A LoRA provider: the base frozen, `lora`'s adapters trained.
    ///
    /// Refused: a LoRA dropout other than 0, an adapter whose shape or dtype
    /// is not its target's, a target without a site and a site without a
    /// target, and any base [`load_hf_pinned`] refuses.
    pub fn open_lora(
        backend: B,
        hooks: DeviceHooks<B>,
        cfg: OpenConfig,
        lora: LoraInit,
    ) -> Result<Self, StepError> {
        let LoraInit {
            config,
            targets,
            adapters,
        } = lora;
        if config.dropout() != 0.0 {
            return Err(contract(format!(
                "LoRA dropout {} is not 0: no dropout stream is checkpointed, so the run could not \
                 resume",
                config.dropout()
            )));
        }
        if targets.is_empty() {
            return Err(contract("a LoRA provider needs at least one target"));
        }
        if adapters.sites().len() != targets.len() {
            return Err(contract(format!(
                "{} adapters for {} targets: every site needs a target and every target a site",
                adapters.sites().len(),
                targets.len()
            )));
        }
        let rank = config.rank();
        let mut seen = BTreeSet::new();
        let mut leaves = Vec::new();
        let mut entries = Vec::new();
        let mut trainable_specs = Vec::new();
        let mut leaf_names = Vec::new();
        let mut leaf_entry = Vec::new();
        for (index, target) in targets.iter().enumerate() {
            if !seen.insert(target.name.clone()) {
                return Err(contract(format!("{}: a duplicate target", target.name)));
            }
            let site = adapters
                .slot(target.layer, target.kind)
                .and_then(Option::as_ref)
                .ok_or_else(|| contract(format!("{}: the target has no adapter", target.name)))?;
            let expect_a = [rank, target.in_features];
            if site.a.shape() != expect_a || site.a.dtype() != DType::F32 {
                return Err(contract(format!(
                    "{}: A is {:?} {:?}, expected F32 {expect_a:?}",
                    target.name,
                    site.a.dtype(),
                    site.a.shape()
                )));
            }
            if site.b.len() != target.out_parts.len() {
                return Err(contract(format!(
                    "{}: {} B parts for {} output parts",
                    target.name,
                    site.b.len(),
                    target.out_parts.len()
                )));
            }
            for (b, &rows) in site.b.iter().zip(&target.out_parts) {
                if b.shape() != [rows, rank] || b.dtype() != DType::F32 {
                    return Err(contract(format!(
                        "{}: a B part is {:?} {:?}, expected F32 {:?}",
                        target.name,
                        b.dtype(),
                        b.shape(),
                        [rows, rank]
                    )));
                }
            }
            let module = target.module();
            let out_total: usize = target.out_parts.iter().sum();
            let a_name = lora_a_name(module);
            let b_name = lora_b_name(module);
            trainable_specs.push(ParamSpec::new(a_name.clone(), &expect_a));
            trainable_specs.push(ParamSpec::new(b_name.clone(), &[out_total, rank]));

            let start = leaves.len();
            leaves.push(backend.upload(&site.a).map_err(core_err)?);
            leaf_names.push(a_name);
            leaf_entry.push(entries.len());
            entries.push(EntryMap {
                leaves: start..start + 1,
                role: Role::LoraA,
            });
            let b_start = leaves.len();
            for (part, b) in site.b.iter().enumerate() {
                leaves.push(backend.upload(b).map_err(core_err)?);
                leaf_names.push(if site.b.len() == 1 {
                    b_name.clone()
                } else {
                    format!("{b_name}[{part}]")
                });
                leaf_entry.push(entries.len());
            }
            entries.push(EntryMap {
                leaves: b_start..leaves.len(),
                role: Role::LoraB { target: index },
            });
        }
        // The caller's handles must not keep an allocation shared: AdamW
        // writes the leaves in place.
        drop(adapters);

        let base = load_base(&backend, &cfg)?;
        if let Some(freeze) = &hooks.freeze {
            for t in base.values() {
                freeze(&backend, t).map_err(core_err)?;
            }
            backend.sync().map_err(core_err)?;
        }
        let mut table = Vec::new();
        for t in hf_tensors(&cfg.spec, &cfg.prefix) {
            let storage = match t.kind {
                HfKind::Table => storage_of(cfg.table)?,
                HfKind::Linear => storage_of(cfg.linears)?,
                HfKind::Plain => StorageDtype::F32,
            };
            table.push(ParamSpec::frozen(t.name, &t.shape, storage));
        }
        table.extend(trainable_specs);
        Self::assemble(
            backend,
            hooks,
            cfg,
            Trainable::Lora {
                base,
                config,
                targets,
            },
            leaves,
            leaf_entry,
            leaf_names,
            entries,
            table,
        )
    }

    /// A provider that trains every base parameter, to compare against the
    /// tessl provider entry for entry. Its table is [`hf_tensors`]; it has no
    /// checkpoint (the adapter layout names LoRA entries) and needs an `F32`
    /// base, which AdamW writes in place.
    pub fn open_full(
        backend: B,
        hooks: DeviceHooks<B>,
        cfg: OpenConfig,
    ) -> Result<Self, StepError> {
        if cfg.linears != DType::F32 || cfg.table != DType::F32 {
            return Err(contract(
                "a full fine-tune updates the base in place and needs an F32 base",
            ));
        }
        let base = load_base(&backend, &cfg)?;
        let template = base
            .try_map(&mut |_| Ok::<(), OjasError>(()))
            .map_err(core_err)?;
        let leaves: Vec<Tensor> = base.values().into_iter().cloned().collect();
        // The loaded tensors are the leaves now: drop the originals so each
        // allocation has one owner.
        drop(base);
        let hf = hf_tensors(&cfg.spec, &cfg.prefix);
        let counts = full_leaf_counts(&cfg.spec);
        if counts.len() != hf.len() || counts.iter().sum::<usize>() != leaves.len() {
            return Err(contract(format!(
                "{} leaves in {} entries do not match {} Hugging Face tensors",
                leaves.len(),
                counts.len(),
                hf.len()
            )));
        }
        let mut entries = Vec::with_capacity(hf.len());
        let mut table = Vec::with_capacity(hf.len());
        let mut leaf_names = Vec::with_capacity(leaves.len());
        let mut leaf_entry = Vec::with_capacity(leaves.len());
        let mut at = 0usize;
        for (t, &count) in hf.iter().zip(&counts) {
            for part in 0..count {
                leaf_names.push(if count == 1 {
                    t.name.clone()
                } else {
                    format!("{}[{part}]", t.name)
                });
                leaf_entry.push(entries.len());
            }
            entries.push(EntryMap {
                leaves: at..at + count,
                role: Role::Hf,
            });
            table.push(ParamSpec::new(t.name.clone(), &t.shape));
            at += count;
        }
        Self::assemble(
            backend,
            hooks,
            cfg,
            Trainable::Full { template },
            leaves,
            leaf_entry,
            leaf_names,
            entries,
            table,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn assemble(
        backend: B,
        hooks: DeviceHooks<B>,
        cfg: OpenConfig,
        trainable: Trainable,
        leaves: Vec<Tensor>,
        leaf_entry: Vec<usize>,
        leaf_names: Vec<String>,
        entries: Vec<EntryMap>,
        table: Vec<ParamSpec>,
    ) -> Result<Self, StepError> {
        if cfg.widest == 0 {
            return Err(contract("the widest sequence is 0 tokens"));
        }
        let tables = Qwen35Tables::new(&cfg.spec, cfg.widest, backend.budget())
            .and_then(|t| t.upload(&backend))
            .map_err(core_err)?;
        let mut moments = Vec::with_capacity(leaves.len());
        for leaf in &leaves {
            let zeros = || -> Result<Tensor, OjasError> {
                let z = Tensor::zeros(leaf.shape(), DType::F32, backend.budget())?;
                backend.upload(&z)
            };
            moments.push((zeros().map_err(core_err)?, zeros().map_err(core_err)?));
        }
        let bank_entries = leaves
            .iter()
            .zip(&leaf_names)
            .map(|(leaf, name)| BankEntry {
                name: name.clone(),
                shape: leaf.shape().to_vec(),
            })
            .collect();
        let bank = AdapterBank::new(bank_entries).map_err(bank_err)?;
        Ok(Self {
            backend,
            hooks,
            spec: cfg.spec,
            prefix: cfg.prefix,
            tables,
            widest: cfg.widest,
            trainable,
            leaves,
            moments,
            leaf_entry,
            leaf_names,
            entries,
            table,
            bank,
            step: 0,
            activations: cfg.activations,
            chunk: cfg.chunk,
            base_id: cfg.base_id,
            preflighted: false,
        })
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// The gradient bank, per trainable leaf: the seam for a caller that adds
    /// a gradient the tape did not produce, and for tests that poison one.
    pub fn bank(&self) -> &AdapterBank {
        &self.bank
    }

    pub fn bank_mut(&mut self) -> &mut AdapterBank {
        &mut self.bank
    }

    /// Number of trainable tensors (a fused `B` counts once per part).
    pub fn leaf_count(&self) -> usize {
        self.leaves.len()
    }

    fn accumulate_into_bank(
        &mut self,
        job: &SequenceJob<'_>,
        mode: BankMode,
        external: Option<&mut HiddenGrad<'_>>,
    ) -> Result<f64, StepError> {
        self.require_ready()?;
        job.validate(self.spec.vocab)?;
        check_config_limits(&self.spec, job)?;
        if job.tokens.len() > self.widest {
            return Err(contract(format!(
                "a sequence of {} tokens is wider than the {} the tables were built for",
                job.tokens.len(),
                self.widest
            )));
        }
        let out =
            run_sequence(&self.backend, &self.seq(), job, external).map_err(Fail::into_step)?;
        for (i, grad) in out.grads.into_iter().enumerate() {
            if let Err(e) = fold_grad(&mut self.bank, &self.backend, i, mode, grad) {
                // A half-written bank must not look complete.
                self.bank.clear();
                return Err(bank_err(e));
            }
        }
        Ok(out.value)
    }

    fn seq(&self) -> Seq<'_> {
        Seq {
            spec: &self.spec,
            tables: &self.tables,
            trainable: &self.trainable,
            leaves: &self.leaves,
            activations: self.activations,
            chunk: self.chunk,
        }
    }

    fn require_ready(&self) -> Result<(), StepError> {
        if self.backend.id() != BackendId::Cpu && !self.preflighted {
            return Err(contract(format!(
                "the {:?} backend has not passed the memory pre-flight: call preflight() before the \
                 first step",
                self.backend.id()
            )));
        }
        Ok(())
    }

    fn host(&self, t: &Tensor) -> Result<Vec<f32>, StepError> {
        self.backend
            .download(t)
            .and_then(|h| h.to_f32_vec())
            .map_err(core_err)
    }

    fn device_tensor(&self, values: &[f32], shape: &[usize]) -> Result<Tensor, StepError> {
        let host = Tensor::from_f32(values, shape, self.backend.budget()).map_err(core_err)?;
        self.backend.upload(&host).map_err(core_err)
    }

    /// One entry's values (or moments) from its leaves' host copies: a fused
    /// `B` from its parts, everything else concatenated.
    fn fuse(&self, role: Role, parts: Vec<Vec<f32>>) -> Vec<f32> {
        match (role, &self.trainable) {
            (
                Role::LoraB { target },
                Trainable::Lora {
                    config, targets, ..
                },
            ) => targets[target].fuse_b(&self.spec, &parts, config.rank()),
            _ => parts.concat(),
        }
    }

    fn trainable_specs(&self) -> impl Iterator<Item = &ParamSpec> {
        self.table.iter().filter(|p| p.trainable)
    }

    /// Every LoRA entry with its values and both moments, `B` fused.
    fn host_entries(&self) -> Result<Vec<AdapterEntry>, StepError> {
        if !matches!(self.trainable, Trainable::Lora { .. }) {
            return Err(StepError::Unsupported(
                "only the LoRA provider has adapter entries".into(),
            ));
        }
        self.entries
            .iter()
            .zip(self.trainable_specs())
            .map(|(e, spec)| {
                let (mut values, mut m1, mut m2) = (Vec::new(), Vec::new(), Vec::new());
                for i in e.leaves.clone() {
                    values.push(self.host(&self.leaves[i])?);
                    m1.push(self.host(&self.moments[i].0)?);
                    m2.push(self.host(&self.moments[i].1)?);
                }
                Ok(AdapterEntry {
                    name: spec.name.clone(),
                    shape: spec.shape.clone(),
                    values: self.fuse(e.role, values),
                    moment1: self.fuse(e.role, m1),
                    moment2: self.fuse(e.role, m2),
                })
            })
            .collect()
    }

    /// Both AdamW moments of every trainable entry, in table order, `B`
    /// fused (LoRA provider).
    pub fn read_moments(&mut self) -> Result<HostMoments, StepError> {
        Ok(self
            .host_entries()?
            .into_iter()
            .map(|e| (e.moment1, e.moment2))
            .collect())
    }

    /// The bank's gradient of every trainable entry, in table order, `B`
    /// fused (LoRA provider): what the next `adamw_step` would apply before
    /// the clip scale.
    pub fn read_gradients(&self) -> Result<Vec<Vec<f32>>, StepError> {
        if !matches!(self.trainable, Trainable::Lora { .. }) {
            return Err(StepError::Unsupported(
                "only the LoRA provider reads fused gradients".into(),
            ));
        }
        self.entries
            .iter()
            .map(|e| {
                let mut parts = Vec::new();
                for i in e.leaves.clone() {
                    parts.push(self.host(&self.bank.grad_on(&self.backend, i).map_err(bank_err)?)?);
                }
                Ok(self.fuse(e.role, parts))
            })
            .collect()
    }

    /// The memory pre-flight: plan the run's widest bucket against
    /// `calibration` and `cap_bytes` ([`preflight::plan`]), then run one real
    /// step at that bucket's width under a child budget of the backend's
    /// ([`preflight::dry_run`], through [`DeviceHooks::rebind`]) into a scratch
    /// bank, and refuse with the shape if either does not fit. A pass is what
    /// lets [`StepProvider::accumulate`] run on a device backend.
    ///
    /// `widest_tokens` is the widest sequence of the run; it is rounded up to
    /// `buckets` and must not exceed the width the provider was opened for.
    /// `external_live_bytes` is device memory held by someone else that the
    /// calibrated peak does not contain. The dry run scores one row and reads
    /// one hidden position, with a zero outside gradient, so it exercises
    /// every part of a step but allocates what the widest sequence does.
    /// Without a rebind hook the dry run could measure nothing, so the
    /// pre-flight refuses rather than pass unmeasured. The dry run sees only
    /// what the rebound backend charges to the budget the hook is handed
    /// ([`preflight::dry_run`]): an allocation made through the original
    /// backend, or by a hook that ignores its budget, is invisible to the
    /// measured peak and the pre-flight still passes.
    pub fn preflight(
        &mut self,
        calibration: &Calibration,
        widest_tokens: usize,
        buckets: &[u64],
        cap_bytes: u64,
        external_live_bytes: u64,
    ) -> Result<preflight::Plan, StepError> {
        // A pass is for one plan: a later pre-flight that refuses must not
        // leave the gate open on an earlier pass.
        self.preflighted = false;
        let tokens = i64::try_from(widest_tokens)
            .map_err(|_| contract(format!("{widest_tokens} tokens do not fit i64")))?;
        let plan = preflight::plan(calibration, tokens, buckets, cap_bytes, external_live_bytes)
            .map_err(preflight_err)?;
        let bucket = usize::try_from(plan.bucket)
            .map_err(|_| contract(format!("bucket {} does not fit usize", plan.bucket)))?;
        if bucket > self.widest {
            return Err(contract(format!(
                "the planned bucket of {bucket} tokens is wider than the {} tokens the tables were \
                 built for",
                self.widest
            )));
        }
        if let Some(max) = self.spec.max_positions {
            if bucket > max {
                return Err(contract(format!(
                    "the planned bucket of {bucket} tokens is longer than max_position_embeddings \
                     {max}"
                )));
            }
        }
        let rebind = self.hooks.rebind.as_ref().ok_or_else(|| {
            StepError::Unsupported(
                "the pre-flight's dry run needs DeviceHooks::rebind to charge a child budget; \
                 without it nothing would be measured"
                    .into(),
            )
        })?;
        let tokens = dry_run_tokens(&self.spec, bucket)?;
        let last = [(bucket - 1) as u32];
        let target = [tokens[0]];
        let seq = self.seq();
        let job = SequenceJob {
            tokens: &tokens,
            rows: RowTargets {
                positions: &last,
                targets: &target,
                scale: 1.0,
            },
            hidden_at: &last,
        };
        check_config_limits(&self.spec, &job)?;
        preflight::dry_run(
            &self.backend,
            i64::try_from(bucket).unwrap_or(i64::MAX),
            |budget| {
                let scratch = rebind(&self.backend, budget.clone());
                let mut zero = |_: &[u32], h: &[f32]| -> Result<Vec<f32>, StepError> {
                    Ok(vec![0.0; h.len()])
                };
                let out =
                    run_sequence(&scratch, &seq, &job, Some(&mut zero)).map_err(Fail::into_ojas)?;
                let mut bank =
                    AdapterBank::new(self.bank.entries().to_vec()).map_err(bank_to_ojas)?;
                for (i, g) in out.grads.into_iter().enumerate() {
                    fold_grad(&mut bank, &scratch, i, BankMode::Overwrite, g)
                        .map_err(bank_to_ojas)?;
                }
                Ok(out.value)
            },
        )
        .map_err(preflight_err)?;
        self.preflighted = true;
        Ok(plan)
    }
}

/// One AdamW entry per leaf, each with its entry's learning rate and weight
/// decay. Takes the provider's fields separately so the caller can keep
/// borrowing its backend.
#[allow(clippy::too_many_arguments)]
fn adamw_batch<'a>(
    leaves: &'a mut [Tensor],
    moments: &'a mut [(Tensor, Tensor)],
    grads: &'a [Tensor],
    names: &'a [String],
    leaf_entry: &'a [usize],
    hyper: &AdamWHyper,
    lr_scale: &[f64],
    weight_decay: &[f64],
) -> Vec<AdamWEntry<'a>> {
    leaves
        .iter_mut()
        .zip(moments.iter_mut())
        .zip(grads)
        .zip(names)
        .zip(leaf_entry)
        .map(|((((param, (m1, m2)), grad), name), &entry)| AdamWEntry {
            name: name.as_str(),
            param,
            grad,
            moment1: m1,
            moment2: m2,
            config: AdamWConfig {
                lr: hyper.lr * lr_scale[entry],
                beta1: hyper.beta1,
                beta2: hyper.beta2,
                eps: hyper.eps,
                weight_decay: weight_decay[entry],
            },
        })
        .collect()
}

fn load_base<B: Backend>(backend: &B, cfg: &OpenConfig) -> Result<Qwen35Params<Tensor>, StepError> {
    let host = load_hf_pinned(
        &cfg.spec,
        &cfg.base_path,
        &cfg.base_id,
        &cfg.prefix,
        cfg.linears,
        cfg.table,
        backend.budget(),
    )
    .map_err(adapter_err)?;
    host.try_map(&mut |t| backend.upload(t)).map_err(core_err)
}

impl<B: Backend> StepProvider for TapeStepProvider<B> {
    fn parameters(&self) -> &[ParamSpec] {
        &self.table
    }

    fn hidden_size(&self) -> usize {
        self.spec.hidden
    }

    fn vocab_size(&self) -> usize {
        self.spec.vocab
    }

    fn accumulate(
        &mut self,
        job: &SequenceJob<'_>,
        mode: BankMode,
        external: Option<&mut HiddenGrad<'_>>,
    ) -> Result<f64, StepError> {
        let result = self.accumulate_into_bank(job, mode, external);
        if result.is_err() && mode == BankMode::Overwrite {
            // A refused first sequence of a step must not leave the previous
            // sum for the following `Add`s to extend (the bank's own rule for
            // a refused `Overwrite`, held here for every refusal).
            self.bank.clear();
        }
        result
    }

    fn grad_sq_norm(&mut self) -> Result<f64, StepError> {
        self.bank.sq_norm_scanned(&self.backend).map_err(bank_err)
    }

    fn supports_lr_scale(&self) -> bool {
        true
    }

    fn adamw_step(
        &mut self,
        hyper: &AdamWHyper,
        step: u64,
        lr_scale: &[f64],
        weight_decay: &[f64],
    ) -> Result<(), StepError> {
        hyper.validate()?;
        check_per_entry(self.entries.len(), lr_scale, weight_decay)?;
        if step != self.step + 1 {
            return Err(contract(format!(
                "AdamW step {step} asked of a provider that has taken {} steps; the next is {}",
                self.step,
                self.step + 1
            )));
        }
        let scale = hyper.grad_scale as f32;
        // Everything that can refuse runs while the bank still holds the sums:
        // the finiteness and shape rule is `check_all`, on the gradients as the
        // clip scale will leave them, and the readback is the norm's (or the
        // first, if no norm was asked for).
        let known: Vec<GradKnown> = self
            .bank
            .scan(&self.backend)
            .map_err(bank_err)?
            .iter()
            .map(|s| GradKnown {
                max_abs: s.max_abs,
                non_finite: s.non_finite,
            })
            .collect();
        let mut grads = self.bank.grads_on(&self.backend).map_err(bank_err)?;
        {
            let mut batch = adamw_batch(
                &mut self.leaves,
                &mut self.moments,
                &grads,
                &self.leaf_names,
                &self.leaf_entry,
                hyper,
                lr_scale,
                weight_decay,
            );
            check_all(
                &self.backend,
                &mut batch,
                self.step,
                self.backend.budget(),
                scale,
                Some(&known),
            )
            .map_err(|e: OptimError| StepError::Backend(e.to_string()))?;
        }
        // Accepted: the update begins, and the bank gives its sums up.
        self.bank.clear();
        if scale != 1.0 {
            for g in &mut grads {
                self.backend.scale_grad(g, scale).map_err(core_err)?;
            }
        }
        let mut batch = adamw_batch(
            &mut self.leaves,
            &mut self.moments,
            &grads,
            &self.leaf_names,
            &self.leaf_entry,
            hyper,
            lr_scale,
            weight_decay,
        );
        apply_all(&self.backend, &mut batch, self.step)
            .map_err(|e: OptimError| StepError::Backend(e.to_string()))?;
        self.step += 1;
        Ok(())
    }

    fn step_count(&self) -> u64 {
        self.step
    }

    fn read_parameters(&mut self) -> Result<Vec<Vec<f32>>, StepError> {
        match &self.trainable {
            Trainable::Lora { .. } => {
                Ok(self.host_entries()?.into_iter().map(|e| e.values).collect())
            }
            Trainable::Full { template } => {
                let mut leaves = self.leaves.iter();
                let params = template
                    .try_map(&mut |_| {
                        leaves.next().cloned().ok_or_else(|| OjasError::OutOfRange {
                            op: "TapeStepProvider",
                            detail: "fewer leaves than parameters".into(),
                        })
                    })
                    .map_err(core_err)?;
                let fused = fuse_grads(&self.spec, &params, &self.prefix, self.backend.budget())
                    .map_err(core_err)?;
                Ok(fused.into_iter().map(|h| h.values).collect())
            }
        }
    }

    fn describe(&self) -> String {
        let kind = match self.trainable {
            Trainable::Lora { .. } => "LoRA",
            Trainable::Full { .. } => "full",
        };
        format!(
            "ojas Qwen3.5 Tape tower ({kind}) on the {:?} backend: {} trainable entries in {} \
             tensors, activations {:?}, base {} ({})",
            self.backend.id(),
            self.entries.len(),
            self.leaves.len(),
            self.activations,
            self.base_id.file_name,
            self.base_id.sha256,
        )
    }

    fn save_state(&mut self, _dir: &Path) -> Result<Vec<PathBuf>, StepError> {
        Err(StepError::Unsupported(
            "the Tape provider's checkpoint records the loop's seed, epoch and consumed digest: \
             a CheckpointContext is required, use save_state_with"
                .into(),
        ))
    }

    fn load_state(&mut self, _dir: &Path) -> Result<(), StepError> {
        Err(StepError::Unsupported(
            "the Tape provider's resume checks the loop's seed, epoch and consumed digest: a \
             CheckpointContext is required, use load_state_with"
                .into(),
        ))
    }

    fn save_state_with(
        &mut self,
        dir: &Path,
        context: &CheckpointContext,
    ) -> Result<Vec<PathBuf>, StepError> {
        let Trainable::Lora { config, .. } = &self.trainable else {
            return Err(StepError::Unsupported(
                "the adapter checkpoint holds LoRA entries; the full provider has none".into(),
            ));
        };
        if self.step != context.optimizer_step {
            return Err(contract(format!(
                "the provider is at step {} but the checkpoint context says {}",
                self.step, context.optimizer_step
            )));
        }
        let ckpt = AdapterCheckpoint {
            lora: *config,
            step: self.step,
            data: DataState {
                rng_state: encode_rng(context.seed, context.epoch),
                consumed_digest: context.consumed_digest.clone(),
            },
            base: self.base_id.clone(),
            entries: self.host_entries()?,
        };
        let target = dir.join("provider");
        write_adapter_checkpoint(&target, &ckpt).map_err(adapter_err)?;
        Ok(vec![target.join(ADAPTERS_FILE), target.join(MOMENTS_FILE)])
    }

    fn load_state_with(
        &mut self,
        dir: &Path,
        context: &CheckpointContext,
    ) -> Result<(), StepError> {
        let Trainable::Lora {
            config, targets, ..
        } = &self.trainable
        else {
            return Err(StepError::Unsupported(
                "the adapter checkpoint holds LoRA entries; the full provider has none".into(),
            ));
        };
        let expected = Expected {
            base: self.base_id.clone(),
            lora: *config,
            entries: self
                .trainable_specs()
                .map(|p| ExpectedEntry {
                    name: p.name.clone(),
                    shape: p.shape.clone(),
                })
                .collect(),
        };
        let ckpt =
            read_adapter_checkpoint(&dir.join("provider"), &expected).map_err(adapter_err)?;
        if ckpt.step != context.optimizer_step {
            return Err(contract(format!(
                "the adapter checkpoint is at step {} but the loop resumes at step {}",
                ckpt.step, context.optimizer_step
            )));
        }
        if ckpt.data.rng_state != encode_rng(context.seed, context.epoch) {
            return Err(contract(
                "the adapter checkpoint's data-order state is not the loop's seed and epoch",
            ));
        }
        if ckpt.data.consumed_digest != context.consumed_digest {
            return Err(contract(
                "the adapter checkpoint's consumed-data digest is not the loop's",
            ));
        }
        let rank = config.rank();
        let mut leaves = self.leaves.clone();
        let mut moments: Vec<(Tensor, Tensor)> = Vec::with_capacity(self.leaves.len());
        moments.extend(self.moments.iter().cloned());
        for (e, entry) in self.entries.iter().zip(&ckpt.entries) {
            let split = |v: &[f32]| -> Vec<Vec<f32>> {
                match e.role {
                    Role::LoraB { target } => targets[target].split_b(&self.spec, v, rank),
                    _ => vec![v.to_vec()],
                }
            };
            let (values, m1, m2) = (
                split(&entry.values),
                split(&entry.moment1),
                split(&entry.moment2),
            );
            if values.len() != e.leaves.len() {
                return Err(contract(format!(
                    "{}: {} parts for {} tensors",
                    entry.name,
                    values.len(),
                    e.leaves.len()
                )));
            }
            for (k, i) in e.leaves.clone().enumerate() {
                let shape = self.leaves[i].shape().to_vec();
                leaves[i] = self.device_tensor(&values[k], &shape)?;
                moments[i] = (
                    self.device_tensor(&m1[k], &shape)?,
                    self.device_tensor(&m2[k], &shape)?,
                );
            }
        }
        self.leaves = leaves;
        self.moments = moments;
        self.step = ckpt.step;
        self.bank.clear();
        Ok(())
    }
}

//! Device-resident training step (`docs/framework-design.md` §3).
//!
//! [`Trainer`] owns the parameters (uniquely), their optimizer moments, the
//! step counter, the schedule, the token bin with its sampler cursor, and a
//! [`TrainState`]. One [`Trainer::step`]:
//! 1. `mult = schedule.multiplier(step)` at the pre-increment step.
//! 2. For each of K micro-batches: clear the tape, bind every parameter as a
//!    leaf, forward through every block and the fused head,
//!    `backward_seeded(loss, 1/K)`, then `take_grad` and
//!    [`Backend::accumulate_grad`] into trainer-owned buffers. Losses are
//!    summed on the device.
//! 3. Clear the tape, so the in-place optimizer owns each parameter alone.
//! 4. `clip_grad_norm(grads, grad_clip)`.
//! 5. Muon NS5 or AdamW per parameter, with `ojas_cpu::optim_group`'s
//!    constants, exactly as `ojas_cpu::HybridOptimizer::step` forms them.
//! 6. `backend.sync()`, then the one readback of the step: the loss sum.
//! 7. Commit: the step counter and the data cursor advance together.
//!
//! An error in steps 1-4 leaves parameters, moments, step and cursor as
//! they were. An error in steps 5-6 marks the trainer
//! [`TrainState::Poisoned`]; every later step is refused with
//! [`OjasError::Poisoned`] (resume from a checkpoint instead).
//!
//! The data cursor is the sampler's ([`ojas_data::BatchSampler::cursor`]):
//! `shard` is the epoch and `token_index` the ordinal of the next window
//! within it, not a token offset.
//!
//! Checkpoints are `crate::checkpoint`'s ([`Trainer::save`],
//! [`Trainer::resume_from`]). A trainer carries a run id: a hash of the
//! spec, the train config and the initial weights, fixed by
//! [`Trainer::new`] and kept across resumes, so files of two runs are
//! never mixed.

use ojas_autograd::Tape;
use ojas_core::{
    next_step, AdamWConfig, Backend, BackendId, CeChunk, DType, DataCursor, MuonNs5Config,
    OjasError, OptimizerKind, Tensor,
};
use ojas_cpu::{
    scaled_lr, LrSchedule, OptimGroup, ADAM_HYBRID_WEIGHT_DECAY, MUON_MOMENTUM, MUON_WEIGHT_DECAY,
};
use ojas_data::{Batch, BatchSampler, DataError, SamplerConfig, TokenBin};

use crate::block::{bind, forward_loss, Rope};
use crate::init::{fnv1a, fnv1a_extend};
use crate::json::quote;
use crate::names::{param_table, ParamInfo};
use crate::spec::ModelSpec;

/// nanolab `matrix_lr`: the Muon peak learning rate.
pub const NANOLAB_MATRIX_LR: f64 = 0.025;
/// nanolab `lr`: the AdamW peak learning rate for the embedding and vectors.
pub const NANOLAB_ADAM_LR: f64 = 6e-4;
/// nanolab `grad_clip`.
pub const NANOLAB_GRAD_CLIP: f32 = 1.0;
/// Default fused-CE tile: 1024 rows by 8192 vocabulary columns, 32 MiB of
/// f32 logits. Backends clamp a tile larger than the problem.
pub const DEFAULT_CE_CHUNK: CeChunk = CeChunk {
    rows: 1024,
    cols: 8192,
};

/// Largest accumulation count: `1 / K` must be an exact, normal `f32`
/// scale of a power of two up to here, and the count exact in `f32`.
const MAX_ACCUM: usize = 1 << 24;

/// What a non-finite loss or gradient does to the data cursor (§3 step 4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NonFinitePolicy {
    /// Keep the cursor: the next step retries the same batches.
    Abort,
    /// Advance the cursor past the batches, still returning the error.
    SkipBatch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrainState {
    Ready,
    /// An optimizer or sync error left parameters partly updated.
    Poisoned,
}

/// Training hyperparameters (the capi `TrainConfig`).
#[derive(Clone, Copy, Debug)]
pub struct TrainConfig {
    /// `B`, rows per micro-batch.
    pub batch: usize,
    /// `T`, tokens per row.
    pub seq_len: usize,
    /// `K`, micro-batches per step.
    pub accum: usize,
    /// Sampler seed (the Feistel key).
    pub data_seed: u64,
    pub schedule: LrSchedule,
    /// Muon peak LR for hidden matrices.
    pub matrix_lr: f64,
    /// AdamW peak LR for the embedding and every vector.
    pub adam_lr: f64,
    /// Global-norm clip.
    pub grad_clip: f32,
    pub on_nonfinite: NonFinitePolicy,
    /// Target id that does not count toward the loss (nanolab's `-1`).
    pub ignore_index: Option<u32>,
    pub chunk: CeChunk,
    /// Hash of the tokenizer that made the token bin. Saved with every
    /// checkpoint; a resume under a different one is refused.
    pub tokenizer_hash: [u8; 32],
    /// Raw SHA-1 of the source that runs the trainer. Saved with every
    /// checkpoint and not compared on resume.
    pub git_sha: [u8; 20],
}

impl TrainConfig {
    /// nanolab's optimizer defaults: Muon 0.025, AdamW 6e-4, clip 1.0,
    /// abort on a non-finite value, no ignored target, the default CE tile.
    pub fn nanolab(
        batch: usize,
        seq_len: usize,
        accum: usize,
        data_seed: u64,
        schedule: LrSchedule,
    ) -> Self {
        Self {
            batch,
            seq_len,
            accum,
            data_seed,
            schedule,
            matrix_lr: NANOLAB_MATRIX_LR,
            adam_lr: NANOLAB_ADAM_LR,
            grad_clip: NANOLAB_GRAD_CLIP,
            on_nonfinite: NonFinitePolicy::Abort,
            ignore_index: None,
            chunk: DEFAULT_CE_CHUNK,
            tokenizer_hash: [0; 32],
            git_sha: [0; 20],
        }
    }

    /// Every field that shapes the run, as one flat JSON object with
    /// sorted keys. Floats print in Rust's shortest round-trip form, so two
    /// configs give the same text exactly when these fields are equal.
    /// `tokenizer_hash` and `git_sha` are not included; the checkpoint
    /// holds them in their own fields.
    pub fn to_json(&self) -> String {
        let (schedule, decay_frac) = match self.schedule {
            LrSchedule::Cosine(_) => ("cosine", 0.0),
            LrSchedule::Wsd(s) => ("wsd", s.decay_frac()),
        };
        let on_nonfinite = match self.on_nonfinite {
            NonFinitePolicy::Abort => "abort",
            NonFinitePolicy::SkipBatch => "skip_batch",
        };
        format!(
            "{{\"accum\":{},\"adam_lr\":{:?},\"batch\":{},\"chunk_cols\":{},\
             \"chunk_rows\":{},\"data_seed\":{},\"decay_frac\":{:?},\"grad_clip\":{:?},\
             \"ignore\":{},\"ignore_index\":{},\"matrix_lr\":{:?},\"on_nonfinite\":{},\
             \"schedule\":{},\"seq_len\":{},\"total_steps\":{},\"warmup_steps\":{}}}",
            self.accum,
            self.adam_lr,
            self.batch,
            self.chunk.cols,
            self.chunk.rows,
            self.data_seed,
            decay_frac,
            self.grad_clip,
            self.ignore_index.is_some(),
            self.ignore_index.unwrap_or(0),
            self.matrix_lr,
            quote(on_nonfinite),
            quote(schedule),
            self.seq_len,
            self.schedule.total_steps(),
            self.schedule.warmup_steps(),
        )
    }

    pub fn validate(&self) -> Result<(), OjasError> {
        const OP: &str = "TrainConfig::validate";
        let range = |detail: String| OjasError::OutOfRange { op: OP, detail };
        if self.batch == 0 || self.seq_len == 0 || self.accum == 0 {
            return Err(range(format!(
                "batch {}, seq_len {} and accum {} must be non-zero",
                self.batch, self.seq_len, self.accum
            )));
        }
        if self.accum > MAX_ACCUM {
            return Err(range(format!("accum {} exceeds {MAX_ACCUM}", self.accum)));
        }
        for (name, lr) in [("matrix_lr", self.matrix_lr), ("adam_lr", self.adam_lr)] {
            if !(lr.is_finite() && lr >= 0.0) {
                return Err(range(format!("{name} {lr} is not finite and non-negative")));
            }
        }
        if !(self.grad_clip.is_finite() && self.grad_clip >= 0.0) {
            return Err(range(format!(
                "grad_clip {} is not finite and non-negative",
                self.grad_clip
            )));
        }
        if self.chunk.rows == 0 || self.chunk.cols == 0 {
            return Err(range(format!("CE chunk {:?} has a zero side", self.chunk)));
        }
        Ok(())
    }

    fn initial_lr(&self, group: OptimGroup) -> f64 {
        match group {
            OptimGroup::MuonMatrix => self.matrix_lr,
            OptimGroup::AdamEmbedding | OptimGroup::AdamVector => self.adam_lr,
        }
    }
}

/// What one step did.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepReport {
    /// Mean of the K micro-batch losses.
    pub loss: f32,
    /// Global gradient norm before clipping.
    pub grad_norm: f32,
    /// Schedule multiplier applied to every group's peak LR.
    pub lr_multiplier: f64,
    /// Completed steps after this one.
    pub step: u64,
    /// Tokens in this step's micro-batches.
    pub tokens: u64,
}

/// A parameter's optimizer state.
pub(crate) enum Moments {
    Muon {
        momentum: Tensor,
    },
    AdamW {
        m: Tensor,
        v: Tensor,
    },
    /// Never receives a gradient (layer 0's `vr_lambda`); never stepped.
    Frozen,
}

/// Borrowed view of a parameter's optimizer state.
#[derive(Clone, Copy, Debug)]
pub enum MomentsRef<'a> {
    Muon { momentum: &'a Tensor },
    AdamW { m: &'a Tensor, v: &'a Tensor },
    Frozen,
}

pub(crate) struct Slot {
    pub(crate) info: ParamInfo,
    pub(crate) value: Tensor,
    pub(crate) moments: Moments,
    pub(crate) initial_lr: f64,
}

/// The nanolab trainer on any [`Backend`].
pub struct Trainer<B: Backend> {
    pub(crate) spec: ModelSpec,
    pub(crate) cfg: TrainConfig,
    pub(crate) tape: Tape<B>,
    pub(crate) slots: Vec<Slot>,
    pub(crate) rope: Rope,
    pub(crate) step: u64,
    pub(crate) bin: TokenBin,
    pub(crate) sampler: SamplerConfig,
    pub(crate) cursor: DataCursor,
    pub(crate) state: TrainState,
    /// 16 lowercase hex digits; see the module docs.
    pub(crate) run: String,
    /// The most scratch any one optimizer call of this trainer charges
    /// ([`Backend::optimizer_scratch_bytes`]); 0 where the backend does not
    /// say.
    pub(crate) optimizer_scratch: u64,
    /// The budget a completed step took above its starting live bytes, and
    /// the micro-batch shape that took it.
    pub(crate) step_peak: Option<StepPeak>,
}

/// What one completed step charged at its peak, above the live bytes it
/// started from, for micro-batches of at most `rows` rows, `k` of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StepPeak {
    pub(crate) rows: usize,
    pub(crate) k: usize,
    pub(crate) bytes: u64,
}

/// Bytes a trainer allocates for its own state beyond the caller's
/// tensors: a copy of every parameter, its moments (Muon one, AdamW two,
/// frozen none) and the RoPE tables for `seq_len`. Checked arithmetic; no
/// allocation.
pub(crate) fn state_bytes(
    table: &[ParamInfo],
    spec: &ModelSpec,
    seq_len: usize,
) -> Result<u64, OjasError> {
    let overflow = || OjasError::OutOfRange {
        op: "Trainer::state_bytes",
        detail: "trainer state size overflows u64".to_string(),
    };
    let mut total = 0u64;
    for info in table {
        let elems = ojas_core::shape_product(&info.shape)? as u64;
        let copies = match (info.trains, info.group) {
            (false, _) => 1,
            (true, OptimGroup::MuonMatrix) => 2,
            (true, OptimGroup::AdamEmbedding | OptimGroup::AdamVector) => 3,
        };
        total = elems
            .checked_mul(4 * copies)
            .and_then(|b| total.checked_add(b))
            .ok_or_else(overflow)?;
    }
    // cos and sin, `[seq_len, head_dim]` f32 each.
    let rope = (seq_len as u64)
        .checked_mul(spec.head_dim as u64)
        .and_then(|n| n.checked_mul(8))
        .ok_or_else(overflow)?;
    total.checked_add(rope).ok_or_else(overflow)
}

/// The largest scratch one optimizer call on any trainable parameter of
/// `table` charges on `backend`. A Muon matrix is sized as `[rows, cols]`,
/// an AdamW tensor of `n` values as `[1, n]`. A backend that does not
/// report a figure counts 0 for that call: the check before the optimizer
/// then cannot cover it.
pub(crate) fn optimizer_scratch<'a, B: Backend + ?Sized>(
    backend: &B,
    table: impl IntoIterator<Item = &'a ParamInfo>,
) -> Result<u64, OjasError> {
    let mut most = 0u64;
    for info in table.into_iter().filter(|info| info.trains) {
        let (kind, rows, cols) = match (info.group, info.shape.as_slice()) {
            (OptimGroup::MuonMatrix, &[rows, cols]) => (OptimizerKind::MuonNs5, rows, cols),
            (OptimGroup::MuonMatrix, other) => {
                return Err(OjasError::Shape {
                    op: "Trainer::optimizer_scratch",
                    detail: format!("{}: Muon needs a matrix, got {other:?}", info.name),
                })
            }
            _ => (
                OptimizerKind::AdamW,
                1,
                ojas_core::shape_product(&info.shape)?,
            ),
        };
        if let Some(bytes) = backend.optimizer_scratch_bytes(kind, rows, cols)? {
            most = most.max(bytes);
        }
    }
    Ok(most)
}

fn data_error(err: DataError) -> OjasError {
    OjasError::OutOfRange {
        op: "Trainer::data",
        detail: err.detail().to_string(),
    }
}

/// Domain tag of the run-id hash.
const RUN_TAG: &[u8] = b"ojas-run-v1\0";

/// The run id of a trainer built by [`Trainer::new`] from `params`.
fn compute_run_id(
    spec: &ModelSpec,
    cfg: &TrainConfig,
    params: &[Tensor],
) -> Result<String, OjasError> {
    let mut hash = fnv1a(RUN_TAG);
    for text in [spec.to_json()?, cfg.to_json()] {
        hash = fnv1a_extend(hash, text.as_bytes());
        hash = fnv1a_extend(hash, &[0]);
    }
    for p in params {
        for x in p.f32_slice()? {
            hash = fnv1a_extend(hash, &x.to_ne_bytes());
        }
    }
    Ok(format!("{hash:016x}"))
}

impl Slot {
    /// A slot of `info` holding `value` and `moments`, at `cfg`'s peak LR.
    pub(crate) fn new(info: ParamInfo, value: Tensor, moments: Moments, cfg: &TrainConfig) -> Self {
        let initial_lr = cfg.initial_lr(info.group);
        Self {
            info,
            value,
            moments,
            initial_lr,
        }
    }
}

impl<B: Backend> Trainer<B> {
    /// A trainer at step 0 and cursor `(0, 0)`.
    ///
    /// `params` are host `F32` tensors in [`param_table`] order (from
    /// [`crate::init_params`] or [`crate::load_params`]). Each is copied
    /// into a new allocation on `backend`, so the trainer owns every
    /// parameter and moment alone and the caller's tensors are never
    /// written. Moments start at zero. The run id is fixed here.
    pub fn new(
        backend: B,
        spec: ModelSpec,
        params: &[Tensor],
        bin: TokenBin,
        cfg: TrainConfig,
    ) -> Result<Self, OjasError> {
        const OP: &str = "Trainer::new";
        let cursor = DataCursor::default();
        let sampler = Self::check_setup(&spec, &cfg, &bin, cursor)?;
        let table = param_table(&spec)?;
        if params.len() != table.len() {
            return Err(OjasError::Shape {
                op: OP,
                detail: format!("{} tensors for {} parameters", params.len(), table.len()),
            });
        }
        for (info, host) in table.iter().zip(params) {
            if host.dtype() != DType::F32 || host.shape() != info.shape.as_slice() {
                return Err(OjasError::Shape {
                    op: OP,
                    detail: format!(
                        "{}: {:?} {:?}, expected F32 {:?}",
                        info.name,
                        host.dtype(),
                        host.shape(),
                        info.shape
                    ),
                });
            }
        }
        let run = compute_run_id(&spec, &cfg, params)?;
        // Everything below charges the budget; refuse before the first
        // allocation when the whole state cannot fit.
        backend
            .budget()
            .check_room(state_bytes(&table, &spec, cfg.seq_len)?)?;
        let mut slots = Vec::with_capacity(table.len());
        for (info, host) in table.into_iter().zip(params) {
            let value = fresh(&backend, host)?;
            let zeros = || {
                fresh(
                    &backend,
                    &Tensor::zeros(&info.shape, DType::F32, backend.budget())?,
                )
            };
            let moments = match (info.trains, info.group) {
                (false, _) => Moments::Frozen,
                (true, OptimGroup::MuonMatrix) => Moments::Muon { momentum: zeros()? },
                (true, OptimGroup::AdamEmbedding | OptimGroup::AdamVector) => Moments::AdamW {
                    m: zeros()?,
                    v: zeros()?,
                },
            };
            slots.push(Slot::new(info, value, moments, &cfg));
        }
        Self::assemble(backend, spec, cfg, bin, sampler, slots, 0, cursor, run)
    }

    /// Every refusal of a trainer setup that needs no parameter: the spec,
    /// the config, `seq_len` against `max_seq`, and `cursor` against the
    /// sampler over `bin`. Returns the sampler config.
    pub(crate) fn check_setup(
        spec: &ModelSpec,
        cfg: &TrainConfig,
        bin: &TokenBin,
        cursor: DataCursor,
    ) -> Result<SamplerConfig, OjasError> {
        spec.validate_for_training()?;
        cfg.validate()?;
        if cfg.seq_len > spec.max_seq {
            return Err(OjasError::Shape {
                op: "Trainer::new",
                detail: format!("seq_len {} exceeds max_seq {}", cfg.seq_len, spec.max_seq),
            });
        }
        let sampler = SamplerConfig {
            seq_len: cfg.seq_len,
            batch: cfg.batch,
            seed: cfg.data_seed,
        };
        BatchSampler::resume(bin, sampler.clone(), cursor).map_err(data_error)?;
        Ok(sampler)
    }

    /// The one constructor: `slots` follow [`param_table`] and are owned by
    /// the trainer alone. Every check is the caller's, done before.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn assemble(
        backend: B,
        spec: ModelSpec,
        cfg: TrainConfig,
        bin: TokenBin,
        sampler: SamplerConfig,
        slots: Vec<Slot>,
        step: u64,
        cursor: DataCursor,
        run: String,
    ) -> Result<Self, OjasError> {
        let rope = Rope::new(&spec, cfg.seq_len, backend.budget())?.upload(&backend)?;
        let optimizer_scratch = optimizer_scratch(&backend, slots.iter().map(|s| &s.info))?;
        Ok(Self {
            spec,
            cfg,
            tape: Tape::new(backend),
            slots,
            rope,
            step,
            bin,
            sampler,
            cursor,
            state: TrainState::Ready,
            run,
            optimizer_scratch,
            step_peak: None,
        })
    }

    pub fn backend(&self) -> &B {
        self.tape.backend()
    }

    pub fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    pub fn config(&self) -> &TrainConfig {
        &self.cfg
    }

    /// Completed optimizer steps.
    pub fn step_count(&self) -> u64 {
        self.step
    }

    /// Where the next sampled batch starts: `shard` is the sampler epoch,
    /// `token_index` the ordinal of the next window within it.
    pub fn cursor(&self) -> DataCursor {
        self.cursor
    }

    /// The run id (16 lowercase hex digits), fixed by [`Self::new`] and
    /// kept by [`Self::resume_from`].
    pub fn run_id(&self) -> &str {
        &self.run
    }

    pub fn state(&self) -> TrainState {
        self.state
    }

    /// Every parameter with its table row, in [`param_table`] order. The
    /// tensors live on the backend.
    pub fn params(&self) -> impl Iterator<Item = (&ParamInfo, &Tensor)> {
        self.slots.iter().map(|s| (&s.info, &s.value))
    }

    /// The parameter called `name`.
    pub fn param(&self, name: &str) -> Option<&Tensor> {
        self.slot(name).map(|s| &s.value)
    }

    /// The optimizer state of the parameter called `name`.
    pub fn moments(&self, name: &str) -> Option<MomentsRef<'_>> {
        self.slot(name).map(|s| match &s.moments {
            Moments::Muon { momentum } => MomentsRef::Muon { momentum },
            Moments::AdamW { m, v } => MomentsRef::AdamW { m, v },
            Moments::Frozen => MomentsRef::Frozen,
        })
    }

    fn slot(&self, name: &str) -> Option<&Slot> {
        self.slots.iter().find(|s| s.info.name == name)
    }

    /// One step on the next K sampled micro-batches.
    ///
    /// On success the cursor moves past them. On an error before the
    /// optimizer the cursor stays, unless the error is
    /// [`OjasError::NonFinite`] and the policy is
    /// [`NonFinitePolicy::SkipBatch`], which advances it and still returns
    /// the error. Parameters, moments and the step counter do not change.
    ///
    /// Memory is checked before the work and again before the first update
    /// (a `CapacityExceeded` from either leaves the trainer Ready), so
    /// capacity never poisons. To measure each step, this resets
    /// `backend().budget()`'s [`ojas_core::Budget::peak_bytes`] at its
    /// start; a caller watching that peak reads it per step, not per run.
    pub fn step(&mut self) -> Result<StepReport, OjasError> {
        self.ready()?;
        let mut sampler = BatchSampler::resume(&self.bin, self.sampler.clone(), self.cursor)
            .map_err(data_error)?;
        let mut batches = Vec::with_capacity(self.cfg.accum);
        for _ in 0..self.cfg.accum {
            batches.push(sampler.next_batch().map_err(data_error)?);
        }
        let after = sampler.cursor();
        match self.run(&batches) {
            Ok(report) => {
                self.cursor = after;
                Ok(report)
            }
            Err(err) => {
                let skip = self.state == TrainState::Ready
                    && self.cfg.on_nonfinite == NonFinitePolicy::SkipBatch
                    && matches!(err, OjasError::NonFinite { .. });
                if skip {
                    self.cursor = after;
                }
                Err(err)
            }
        }
    }

    /// One step on caller-supplied micro-batches (K = `batches.len()`).
    /// The sampler cursor does not move. Each batch must have `seq_len`
    /// columns; its row count may differ from the configured batch. Memory
    /// checks and the budget peak reset are as in [`Self::step`].
    pub fn step_tokens(&mut self, batches: &[Batch]) -> Result<StepReport, OjasError> {
        self.run(batches)
    }

    /// The room a step of `k` micro-batches of at most `rows` rows needs
    /// before it starts: the optimizer's scratch, or the measured peak of an
    /// earlier step no larger in either dimension, whichever is more. A
    /// larger step allocates at least what the smaller one did, so a step
    /// refused here would have been refused part-way.
    pub(crate) fn preflight_bytes(&self, rows: usize, k: usize) -> u64 {
        let measured = match self.step_peak {
            Some(p) if rows >= p.rows && k >= p.k => p.bytes,
            _ => 0,
        };
        measured.max(self.optimizer_scratch)
    }

    /// Keep the peak of the largest step shape seen: a step that is at
    /// least as large in both dimensions replaces the record (the larger
    /// bytes win at an equal shape); an incomparable one leaves it.
    fn record_peak(&mut self, rows: usize, k: usize, bytes: u64) {
        let replace = match self.step_peak {
            None => true,
            Some(p) if rows == p.rows && k == p.k => bytes > p.bytes,
            Some(p) => rows >= p.rows && k >= p.k,
        };
        if replace {
            self.step_peak = Some(StepPeak { rows, k, bytes });
        }
    }

    pub(crate) fn ready(&self) -> Result<(), OjasError> {
        match self.state {
            TrainState::Ready => Ok(()),
            TrainState::Poisoned => Err(OjasError::Poisoned),
        }
    }

    fn run(&mut self, batches: &[Batch]) -> Result<StepReport, OjasError> {
        const OP: &str = "Trainer::step";
        self.ready()?;
        // Step 1, and every refusal that does not depend on the data,
        // before any work.
        let mult = self.cfg.schedule.multiplier(self.step)?;
        let next = next_step(self.step)?;
        let k = batches.len();
        if k == 0 || k > MAX_ACCUM {
            return Err(OjasError::OutOfRange {
                op: OP,
                detail: format!("{k} micro-batches; expected 1..={MAX_ACCUM}"),
            });
        }
        let mut tokens = 0u64;
        for b in batches {
            let n = b.batch.checked_mul(b.seq_len);
            if b.seq_len != self.cfg.seq_len
                || b.batch == 0
                || n != Some(b.x.len())
                || n != Some(b.y.len())
            {
                return Err(OjasError::Shape {
                    op: OP,
                    detail: format!(
                        "micro-batch [{}, {}] with {} ids and {} targets; seq_len is {}",
                        b.batch,
                        b.seq_len,
                        b.x.len(),
                        b.y.len(),
                        self.cfg.seq_len
                    ),
                });
            }
            tokens = tokens.saturating_add(b.x.len() as u64);
        }
        let rows = batches.iter().map(|b| b.batch).max().unwrap_or(0);
        let configs = self.optimizer_configs(mult)?;
        let budget = self.tape.backend().budget().clone();
        // Before any work: room for the optimizer's scratch, and for the
        // whole step once a step of at least this shape has been measured.
        // The refusal leaves everything unchanged and wastes no compute.
        budget.check_room(self.preflight_bytes(rows, k))?;
        let live_before = budget.live_bytes()?;
        budget.reset_peak();
        // Steps 2-4: nothing the trainer owns changes.
        let result = self.gradients(batches, 1.0 / k as f32);
        // Step 3: the tape holds a clone of every parameter until cleared.
        self.tape.clear();
        let (mut grads, loss_sum) = result?;
        let grad_norm = self
            .tape
            .backend()
            .clip_grad_norm(&mut grads, self.cfg.grad_clip)?;
        // The last refusal that leaves the parameters whole: the optimizer
        // charges its scratch inside the update, after which a refusal
        // poisons. Checked with the gradients alive, as they are then.
        budget.check_room(self.optimizer_scratch)?;
        // Steps 5-6: an error here leaves parameters partly updated.
        let loss = match self.apply(&grads, &configs) {
            Ok(()) => self.finish(&loss_sum),
            Err(err) => Err(err),
        };
        let loss = match loss {
            Ok(loss) => loss,
            Err(err) => {
                self.state = TrainState::Poisoned;
                return Err(err);
            }
        };
        // Step 7.
        self.step = next;
        drop(grads);
        self.record_peak(rows, k, budget.peak_bytes().saturating_sub(live_before));
        Ok(StepReport {
            loss: loss / k as f32,
            grad_norm,
            lr_multiplier: mult,
            step: next,
            tokens,
        })
    }

    /// Per trainable slot, the Muon or AdamW config of this step, formed as
    /// `HybridOptimizer::step` forms it.
    fn optimizer_configs(&self, mult: f64) -> Result<Vec<StepConfig>, OjasError> {
        self.slots
            .iter()
            .filter(|s| !matches!(s.moments, Moments::Frozen))
            .map(|s| {
                let lr = scaled_lr(s.initial_lr, mult)?;
                Ok(match s.moments {
                    Moments::Muon { .. } => StepConfig::Muon(MuonNs5Config {
                        lr,
                        momentum: MUON_MOMENTUM,
                        weight_decay: MUON_WEIGHT_DECAY,
                        nesterov: true,
                    }),
                    _ => StepConfig::AdamW(AdamWConfig::nanolab(lr, ADAM_HYBRID_WEIGHT_DECAY)),
                })
            })
            .collect()
    }

    /// Steps 2: the summed gradient of every trainable slot, in slot order,
    /// and the device-side sum of the K losses.
    fn gradients(
        &mut self,
        batches: &[Batch],
        seed: f32,
    ) -> Result<(Vec<Tensor>, Tensor), OjasError> {
        const OP: &str = "Trainer::step";
        let Self {
            spec,
            cfg,
            tape,
            slots,
            rope,
            ..
        } = self;
        let mut accs: Vec<Option<Tensor>> = slots
            .iter()
            .filter(|s| !matches!(s.moments, Moments::Frozen))
            .map(|_| None)
            .collect();
        let mut loss_sum: Option<Tensor> = None;
        let values: Vec<Tensor> = slots.iter().map(|s| s.value.clone()).collect();
        for batch in batches {
            tape.clear();
            let budget = tape.backend().budget().clone();
            let ids = Tensor::from_u32(&batch.x, &[batch.batch, batch.seq_len], &budget)?;
            let targets = Tensor::from_u32(&batch.y, &[batch.x.len()], &budget)?;
            let params = bind(tape, spec, &values)?;
            let loss = forward_loss(
                tape,
                spec,
                &params,
                &ids,
                &targets,
                rope,
                cfg.ignore_index,
                cfg.chunk,
            )?;
            let loss_value = tape.value(loss)?.clone();
            tape.backward_seeded(loss, seed)?;
            let mut acc_slots = accs.iter_mut();
            for (slot, var) in slots.iter().zip(params.into_flat()) {
                if matches!(slot.moments, Moments::Frozen) {
                    continue;
                }
                let grad = tape.take_grad(var).ok_or_else(|| OjasError::Shape {
                    op: OP,
                    detail: format!("{} received no gradient", slot.info.name),
                })?;
                let acc = acc_slots.next().ok_or_else(|| OjasError::Shape {
                    op: OP,
                    detail: "accumulator count disagrees with the trainable slots".to_string(),
                })?;
                match acc {
                    Some(sum) => tape.backend().accumulate_grad(sum, &grad)?,
                    None => *acc = Some(grad),
                }
            }
            loss_sum = Some(match loss_sum {
                None => loss_value,
                Some(sum) => tape.backend().residual_add_forward(&sum, &loss_value)?,
            });
        }
        drop(values);
        let grads = accs
            .into_iter()
            .collect::<Option<Vec<Tensor>>>()
            .ok_or_else(|| OjasError::Shape {
                op: OP,
                detail: "a trainable parameter has no accumulated gradient".to_string(),
            })?;
        let loss_sum = loss_sum.ok_or_else(|| OjasError::OutOfRange {
            op: OP,
            detail: "no micro-batch".to_string(),
        })?;
        Ok((grads, loss_sum))
    }

    /// Step 5. `grads` and `configs` follow the trainable slots in order.
    fn apply(&mut self, grads: &[Tensor], configs: &[StepConfig]) -> Result<(), OjasError> {
        let backend = self.tape.backend();
        let step = self.step;
        let trainable = self
            .slots
            .iter_mut()
            .filter(|s| !matches!(s.moments, Moments::Frozen));
        for ((slot, grad), config) in trainable.zip(grads).zip(configs) {
            match (&mut slot.moments, config) {
                (Moments::Muon { momentum }, StepConfig::Muon(c)) => {
                    backend.muon_ns5_step(&mut slot.value, grad, momentum, *c)?
                }
                (Moments::AdamW { m, v }, StepConfig::AdamW(c)) => {
                    backend.adamw_step(&mut slot.value, grad, m, v, step, *c)?
                }
                _ => {
                    return Err(OjasError::Shape {
                        op: "Trainer::step",
                        detail: format!("{}: optimizer state and config disagree", slot.info.name),
                    })
                }
            }
        }
        Ok(())
    }

    /// Step 6: surface deferred faults, then read the loss sum back once.
    fn finish(&self, loss_sum: &Tensor) -> Result<f32, OjasError> {
        let backend = self.tape.backend();
        backend.sync()?;
        let host = backend.download(loss_sum)?;
        let values = host.to_f32_vec()?;
        match values.as_slice() {
            [loss] if loss.is_finite() => Ok(*loss),
            [_] => Err(OjasError::NonFinite {
                op: "Trainer::step",
            }),
            other => Err(OjasError::Shape {
                op: "Trainer::step",
                detail: format!("loss has {} elements", other.len()),
            }),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum StepConfig {
    Muon(MuonNs5Config),
    AdamW(AdamWConfig),
}

/// A new allocation on `backend` holding `host`'s values, owned by the
/// caller alone. A host tensor is required: on the CPU, `Backend::upload`
/// of a host tensor shares it, so the values are copied instead.
fn fresh<B: Backend + ?Sized>(backend: &B, host: &Tensor) -> Result<Tensor, OjasError> {
    if host.device().is_some() {
        return Err(OjasError::Placement {
            op: "Trainer::new",
            expected: None,
            found: host.device(),
        });
    }
    if backend.id() == BackendId::Cpu {
        Tensor::from_f32(&host.to_f32_vec()?, host.shape(), backend.budget())
    } else {
        backend.upload(host)
    }
}

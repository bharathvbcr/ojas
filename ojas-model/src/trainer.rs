//! Device-resident training step (`docs/framework-design.md` §3).
//!
//! [`Trainer`] owns the parameters (uniquely), their optimizer moments, the
//! step counter, the schedule, the token bin with its sampler cursor, and a
//! [`TrainState`]. One [`Trainer::step`]:
//! 1. `mult = schedule.multiplier(step)` at the pre-increment step.
//! 2. Weight each micro-batch by its share of the step's valid targets,
//!    `w_i = n_i / N` (`n_i` counts the targets that are not
//!    `ignore_index`), which is `1/K` when every count is equal. For each
//!    micro-batch with `w_i > 0`: clear the tape, bind every parameter as a
//!    leaf, forward through every block and the fused head,
//!    `backward_seeded(loss, w_i)`, then `take_grad` and
//!    [`ojas_core::BackendFloat::accumulate_grad`] into trainer-owned buffers. Losses are
//!    weighted and summed on the device. A micro-batch whose targets are all
//!    ignored has `w_i = 0` and is not run (its ids are still checked); a
//!    step whose targets are all ignored is [`OjasError::NonFinite`].
//! 3. Clear the tape, so the in-place optimizer owns each parameter alone.
//! 4. `clip_grad_norm(grads, grad_clip)`.
//! 5. Muon NS5 or AdamW per parameter, with `ojas_cpu::optim_group`'s
//!    constants, exactly as `ojas_cpu::HybridOptimizer::step` forms them.
//! 6. `backend.sync()`, then the one readback of the step: the weighted loss.
//! 7. Commit: the step counter and the data cursor advance together.
//!
//! An error in steps 1-4 leaves parameters, moments, step and cursor as
//! they were. An error in steps 5-6 marks the trainer
//! [`TrainState::Poisoned`]; every later step is refused with
//! [`OjasError::Poisoned`] (resume from a checkpoint instead).
//! An [`OjasError::Poisoned`] from the autocast wrapper during the forward
//! or backward is a step 2 error: [`TrainState`] stays `Ready` because the
//! wrapper itself stays poisoned, and the next step fails the same way
//! until a new trainer is built.
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
    next_step, AdamWConfig, AutocastGuard, AutocastMode, Backend, BackendId, CeChunk, DType,
    DataCursor, MuonNs5Config, Ns5Precision, OjasError, OptimizerKind, Tensor,
};
use ojas_cpu::{
    scaled_lr, LrSchedule, OptimGroup, ADAM_HYBRID_WEIGHT_DECAY, MUON_MOMENTUM, MUON_WEIGHT_DECAY,
};
use ojas_data::{Batch, BatchSampler, DataError, SamplerConfig, TokenBin};
use ojas_io::quote;

use crate::block::{bind, forward_loss, ActivationCheckpoint, Rope};
use crate::init::{fnv1a, fnv1a_extend};
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

/// Largest accumulation count: the count must be exact in `f32`, and `1 / K`
/// a normal `f32` seed ([`ojas_autograd::Tape::backward_seeded`] refuses a
/// subnormal one).
const MAX_ACCUM: usize = 1 << 24;

/// Consecutive [`OjasError::NonFinite`] steps [`Trainer::step`] skips under
/// [`NonFinitePolicy::SkipBatch`] before it stops skipping and returns an
/// [`OjasError::OutOfRange`] naming the run of failures (ft-9c13f472 D9).
/// The count lives in memory; a resumed trainer starts it at 0.
pub(crate) const MAX_CONSECUTIVE_SKIPS: u32 = 8;

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
    /// Global-norm clip: finite and positive. 0 is refused by
    /// [`Self::validate`], since torch's and [`ojas_core::clip_scale`]'s 0
    /// zeroes every gradient; there is no "off" value.
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
    /// Matmul-class bf16 region for the forward and backward. Off omits the
    /// key from [`Self::to_json`], so a run id hashed from that text stays
    /// the same as a config written before the field existed.
    pub autocast: AutocastMode,
    /// Precision of Muon's Newton-Schulz iterate. F32 omits the key from
    /// [`Self::to_json`], as `autocast` Off does. [`Ns5Precision::Bf16`] is
    /// stock nanolab's.
    pub muon_ns5: Ns5Precision,
    /// Whether each block's activations are recomputed in the backward
    /// instead of kept ([`ActivationCheckpoint`]). It trades compute for
    /// memory and leaves every value the same, so it is not part of
    /// [`Self::to_json`] or the run id, and a resume may change it.
    pub activations: ActivationCheckpoint,
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
            autocast: AutocastMode::Off,
            muon_ns5: Ns5Precision::F32,
            activations: ActivationCheckpoint::Off,
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
        // Absent when Off, so the sorted-key text of an Off config is the
        // text it had before this field existed.
        let autocast = match self.autocast {
            AutocastMode::Off => "",
            AutocastMode::Bf16 => "\"autocast\":\"bf16\",",
        };
        // Absent when F32, for the same reason.
        let muon_ns5 = match self.muon_ns5 {
            Ns5Precision::F32 => "",
            Ns5Precision::Bf16 => "\"muon_ns5\":\"bf16\",",
        };
        format!(
            "{{\"accum\":{},\"adam_lr\":{:?},{autocast}\"batch\":{},\"chunk_cols\":{},\
             \"chunk_rows\":{},\"data_seed\":{},\"decay_frac\":{:?},\"grad_clip\":{:?},\
             \"ignore\":{},\"ignore_index\":{},\"matrix_lr\":{:?},{muon_ns5}\"on_nonfinite\":{},\
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
        // 0 is refused, not read as "no clipping": torch's
        // `clip_grad_norm_(max_norm=0)`, which nanolab calls unconditionally
        // (`train.py:347`), multiplies every gradient by 0, and so does
        // `clip_scale`, so a 0 run would never learn.
        //
        // `+inf` is refused too, though `ojas_core::clip_scale` accepts it
        // as torch does (measure only, coefficient 1; ft-9c13f472 D3b):
        // [`Self::to_json`] prints floats with `{:?}`, which writes `inf`,
        // not a JSON number, and the config is saved in every checkpoint.
        // `f32::MAX` clips nothing either, and round-trips.
        if self.grad_clip == f32::INFINITY {
            return Err(range(
                "grad_clip inf is refused: torch's clip_grad_norm_(max_norm=inf) measures the \
                 norm without clipping, but this config is serialized to JSON, which cannot \
                 carry inf; use f32::MAX to clip nothing"
                    .to_string(),
            ));
        }
        if !(self.grad_clip.is_finite() && self.grad_clip > 0.0) {
            return Err(range(format!(
                "grad_clip {} is not finite and positive (torch's clip_grad_norm_ with \
                 max_norm 0 zeroes every gradient)",
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
    /// Token-weighted mean of the micro-batch losses: each is weighted by
    /// its share of the step's valid targets, which is the plain mean of
    /// the K losses when every micro-batch has the same count. nanolab
    /// divides each micro-batch loss by `grad_accum` (`train.py:342`), so
    /// the two differ only when `ignore_index` leaves unequal counts
    /// (ft-9c13f472 D2).
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
    /// ([`ojas_core::BackendCore::optimizer_scratch_bytes`]); 0 where the backend does not
    /// say.
    pub(crate) optimizer_scratch: u64,
    /// The budget a completed step took above its starting live bytes, and
    /// the micro-batch shape that took it.
    pub(crate) step_peak: Option<StepPeak>,
    /// Steps in a row that [`Trainer::step`] skipped for a non-finite value;
    /// see [`MAX_CONSECUTIVE_SKIPS`]. Cleared by a committed step.
    pub(crate) consecutive_skips: u32,
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
            consecutive_skips: 0,
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
    /// After 8 such skips in a row (`MAX_CONSECUTIVE_SKIPS`), the next
    /// non-finite step returns [`OjasError::OutOfRange`] instead and keeps
    /// the cursor; any committed step clears the count. Only this method
    /// counts: [`Self::step_tokens`] has no cursor to skip.
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
                self.consecutive_skips = 0;
                Ok(report)
            }
            Err(err) => {
                let skippable = self.state == TrainState::Ready
                    && self.cfg.on_nonfinite == NonFinitePolicy::SkipBatch
                    && matches!(err, OjasError::NonFinite { .. });
                if !skippable {
                    return Err(err);
                }
                if self.consecutive_skips >= MAX_CONSECUTIVE_SKIPS {
                    // Skipping forever would hide a run that no longer
                    // trains: stop, keep the cursor, and say why.
                    return Err(OjasError::OutOfRange {
                        op: "Trainer::step",
                        detail: format!(
                            "{} consecutive non-finite steps under SkipBatch (last: {err}); \
                             the run is not recovering",
                            self.consecutive_skips + 1
                        ),
                    });
                }
                self.cursor = after;
                self.consecutive_skips += 1;
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
        // The weights need only the host targets: a step with no valid
        // target is `NonFinite` before any work, as the all-ignored batch
        // always was.
        let counts = batches
            .iter()
            .map(|b| valid_targets(&b.y, self.cfg.ignore_index))
            .collect::<Vec<u64>>();
        let weights = MicroWeights::new(&counts)?;
        let configs = self.optimizer_configs(mult)?;
        let budget = self.tape.backend().budget().clone();
        // Before any work: room for the optimizer's scratch, and for the
        // whole step once a step of at least this shape has been measured.
        // The refusal leaves everything unchanged and wastes no compute.
        budget.check_room(self.preflight_bytes(rows, k))?;
        let live_before = budget.live_bytes()?;
        budget.reset_peak();
        // Steps 2-4: nothing the trainer owns changes.
        let result = self.gradients(batches, &weights);
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
            loss: weights.report(loss),
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
                        ns5: self.cfg.muon_ns5,
                    }),
                    _ => StepConfig::AdamW(AdamWConfig::nanolab(lr, ADAM_HYBRID_WEIGHT_DECAY)),
                })
            })
            .collect()
    }

    /// Bf16 enters a region. Off does not call [`ojas_core::BackendCore::autocast_region`].
    fn enter_autocast(backend: &B, mode: AutocastMode) -> Result<Option<AutocastGuard>, OjasError> {
        match mode {
            AutocastMode::Off => Ok(None),
            AutocastMode::Bf16 => Ok(Some(backend.autocast_region(mode)?)),
        }
    }

    /// Steps 2: the summed gradient of every trainable slot, in slot order,
    /// and the device-side sum of the micro-batch losses. Micro-batch `i`
    /// seeds its backward with `weights.weights[i]`; a zero weight skips the
    /// batch after its ids are checked. The losses are summed unscaled when
    /// `weights.uniform` ([`MicroWeights::report`] divides), and each scaled
    /// by its weight on the device otherwise.
    fn gradients(
        &mut self,
        batches: &[Batch],
        weights: &MicroWeights,
    ) -> Result<(Vec<Tensor>, Tensor), OjasError> {
        const OP: &str = "Trainer::step";
        if weights.weights.len() != batches.len() {
            return Err(OjasError::Shape {
                op: OP,
                detail: format!(
                    "{} weights for {} micro-batches",
                    weights.weights.len(),
                    batches.len()
                ),
            });
        }
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
        for (batch, &weight) in batches.iter().zip(&weights.weights) {
            if weight == 0.0 {
                // No valid target: the batch contributes nothing and is not
                // run, but the forward would have refused a bad id, so the
                // ids are checked here. Every target is `ignore_index`.
                check_ids(&batch.x, spec.vocab)?;
                continue;
            }
            tape.clear();
            let budget = tape.backend().budget().clone();
            let ids = Tensor::from_u32(&batch.x, &[batch.batch, batch.seq_len], &budget)?;
            let targets = Tensor::from_u32(&batch.y, &[batch.x.len()], &budget)?;
            // The region covers the forward, the backward and take_grad.
            // It is dropped before accumulate_grad and before the loss-sum
            // residual add, both of which stay in f32. Off does not enter.
            let (loss_value, taken) = {
                let _region = Self::enter_autocast(tape.backend(), cfg.autocast)?;
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
                    cfg.activations,
                )?;
                let loss_value = tape.value(loss)?.clone();
                tape.backward_seeded(loss, weight)?;
                let mut taken = Vec::with_capacity(accs.len());
                for (slot, var) in slots.iter().zip(params.into_flat()) {
                    if matches!(slot.moments, Moments::Frozen) {
                        continue;
                    }
                    let grad = tape.take_grad(var).ok_or_else(|| OjasError::Shape {
                        op: OP,
                        detail: format!("{} received no gradient", slot.info.name),
                    })?;
                    taken.push(grad);
                }
                (loss_value, taken)
            };
            if taken.len() != accs.len() {
                return Err(OjasError::Shape {
                    op: OP,
                    detail: "accumulator count disagrees with the trainable slots".to_string(),
                });
            }
            for (acc, grad) in accs.iter_mut().zip(taken) {
                match acc {
                    Some(sum) => tape.backend().accumulate_grad(sum, &grad)?,
                    None => *acc = Some(grad),
                }
            }
            let mut loss_value = loss_value;
            if !weights.uniform && weight != 1.0 {
                tape.backend().scale_grad(&mut loss_value, weight)?;
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

/// Targets that count toward the loss: every one that is not `ignore`.
/// An out-of-vocabulary target counts; the forward refuses it.
fn valid_targets(targets: &[u32], ignore: Option<u32>) -> u64 {
    targets.iter().filter(|&&t| ignore != Some(t)).count() as u64
}

/// Refuse a token id outside the vocabulary, as the embedding does.
fn check_ids(ids: &[u32], vocab: usize) -> Result<(), OjasError> {
    for (n, &id) in ids.iter().enumerate() {
        if u64::from(id) >= vocab as u64 {
            return Err(OjasError::OutOfRange {
                op: "Trainer::step",
                detail: format!("token id {id} at {n} is outside vocab {vocab}"),
            });
        }
    }
    Ok(())
}

/// How a step weights its micro-batches (ft-9c13f472 D2, D4).
///
/// Micro-batch `i` has `n_i` valid targets, `N = sum n_i`, and weight
/// `n_i / N`: its share of the step's tokens, so one step over uneven
/// micro-batches equals one batch of all their rows (HF's
/// `num_items_in_batch`). nanolab divides every micro-batch by `grad_accum`
/// (`train.py:342`) and so weights by batch; the two agree when every count
/// is equal, which is the only case the oracle fixtures reach.
///
/// When every non-empty micro-batch has the same count, the weight is
/// `1.0 / run as f32` (`run` = non-empty micro-batches), the seed the
/// trainer used for every step before, and `uniform` is set so the losses
/// are summed unscaled and divided by `run` on the host, as before: equal
/// counts keep their bits. A micro-batch with `n_i = 0` has weight 0.
#[derive(Clone, Debug, PartialEq)]
struct MicroWeights {
    weights: Vec<f32>,
    uniform: bool,
    run: usize,
}

impl MicroWeights {
    /// `counts[i]` is micro-batch `i`'s valid targets. `N = 0` is
    /// [`OjasError::NonFinite`]: the mean has no denominator.
    fn new(counts: &[u64]) -> Result<Self, OjasError> {
        const OP: &str = "Trainer::step";
        let mut total = 0u64;
        for &n in counts {
            total = total.checked_add(n).ok_or_else(|| OjasError::OutOfRange {
                op: OP,
                detail: "valid target count overflows u64".to_string(),
            })?;
        }
        let Some(first) = counts.iter().copied().find(|&n| n > 0) else {
            return Err(OjasError::NonFinite { op: OP });
        };
        let run = counts.iter().filter(|&&n| n > 0).count();
        let uniform = counts.iter().all(|&n| n == 0 || n == first);
        let weights = counts
            .iter()
            .map(|&n| match n {
                0 => 0.0,
                _ if uniform => 1.0 / run as f32,
                _ => (n as f64 / total as f64) as f32,
            })
            .collect();
        Ok(Self {
            weights,
            uniform,
            run,
        })
    }

    /// The step's loss from the device's loss sum.
    fn report(&self, loss_sum: f32) -> f32 {
        if self.uniform {
            loss_sum / self.run as f32
        } else {
            loss_sum
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum StepConfig {
    Muon(MuonNs5Config),
    AdamW(AdamWConfig),
}

/// A new allocation on `backend` holding `host`'s values, owned by the
/// caller alone. A host tensor is required: on the CPU, `ojas_core::BackendCore::upload`
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
        // Straight from the host window into the charged copy: one
        // allocation, and that one fallible.
        Tensor::from_f32(host.f32_slice()?, host.shape(), backend.budget())
    } else {
        backend.upload(host)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ojas_core::{Budget, Numerics};
    use ojas_cpu::{CosineSchedule, CpuBackend};

    /// Equal counts keep the pre-ft-9c13f472 seed bit for bit: the uniform
    /// weight is `1.0 / k as f32`, and the general `n / N` formula agrees
    /// with it for every `k` up to [`MAX_ACCUM`] (so a count that is equal
    /// by accident cannot move bits between the two forms).
    #[test]
    fn equal_counts_keep_the_one_over_k_seed() {
        for k in [1usize, 2, 3, 5, 6, 7, 10, 100, 1000, 4096] {
            for n in [1u64, 3, 64, 4096] {
                let w = MicroWeights::new(&vec![n; k]).unwrap();
                assert!(w.uniform && w.run == k);
                let seed = (1.0 / k as f32).to_bits();
                assert!(w.weights.iter().all(|x| x.to_bits() == seed), "k {k} n {n}");
            }
        }
        for k in 1..=MAX_ACCUM as u64 {
            let general = (1.0f64 / k as f64) as f32;
            let seed = 1.0f32 / k as f32;
            assert_eq!(general.to_bits(), seed.to_bits(), "k {k}");
            let by_share = (7.0f64 / (7 * k) as f64) as f32;
            assert_eq!(by_share.to_bits(), seed.to_bits(), "k {k} share");
        }
    }

    #[test]
    fn uneven_counts_weigh_by_share_and_an_empty_batch_by_zero() {
        let w = MicroWeights::new(&[8, 1]).unwrap();
        assert!(!w.uniform);
        assert_eq!(
            w.weights,
            vec![(8.0f64 / 9.0) as f32, (1.0f64 / 9.0) as f32]
        );
        let w = MicroWeights::new(&[0, 5, 0, 5]).unwrap();
        assert!(w.uniform && w.run == 2);
        assert_eq!(w.weights, vec![0.0, 0.5, 0.0, 0.5]);
        assert_eq!(w.report(3.0), 1.5);
        let w = MicroWeights::new(&[3, 0, 1]).unwrap();
        assert_eq!(w.weights, vec![0.75, 0.0, 0.25]);
        assert_eq!(w.report(3.0), 3.0);
        assert!(matches!(
            MicroWeights::new(&[0, 0]),
            Err(OjasError::NonFinite { .. })
        ));
        assert!(matches!(
            MicroWeights::new(&[u64::MAX, 1]),
            Err(OjasError::OutOfRange { .. })
        ));
    }

    #[test]
    fn ids_outside_the_vocabulary_are_refused() {
        assert!(check_ids(&[0, 255], 256).is_ok());
        assert!(matches!(
            check_ids(&[1, 256], 256),
            Err(OjasError::OutOfRange { .. })
        ));
        assert_eq!(valid_targets(&[7, 1, 7, 2], Some(7)), 2);
        assert_eq!(valid_targets(&[7, 1, 7, 2], None), 4);
    }

    /// A trainable slot that the backward pass leaves without a gradient
    /// fails the step closed, before any optimizer call: nothing it holds
    /// is skipped silently. Layer 0's `vr_lambda` is the one parameter the
    /// model never differentiates (`ParamInfo::trains`), so it is marked
    /// trainable here to make one.
    #[test]
    fn a_trainable_slot_without_a_gradient_fails_the_step_and_changes_nothing() {
        let spec = ModelSpec::tiny();
        let budget = Budget::new(1 << 30);
        let params = crate::init_params(&spec, 5, &budget).unwrap();
        let path = std::env::temp_dir().join(format!(
            "ojas-model-trainer-unit-{}.bin",
            std::process::id()
        ));
        let tokens: Vec<u8> = (0..20_000u32)
            .flat_map(|i| (((i * 5 + 3) % 256) as u16).to_le_bytes())
            .collect();
        std::fs::write(&path, &tokens).unwrap();
        let bin = TokenBin::open_headerless(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        let schedule = LrSchedule::Cosine(CosineSchedule::new(2, 40).unwrap());
        let cfg = TrainConfig::nanolab(2, 32, 1, 1337, schedule);
        let backend = CpuBackend::new(Budget::new(1 << 30)).with_numerics(Numerics::Exact);
        let mut t = Trainer::new(backend, spec, &params, bin, cfg).unwrap();
        let unused = t.slots.iter().position(|s| !s.info.trains).unwrap();
        let zeros = || Tensor::zeros(&[1], DType::F32, t.backend().budget()).unwrap();
        let moments = Moments::AdamW {
            m: zeros(),
            v: zeros(),
        };
        let name = t.slots[unused].info.name.clone();
        t.slots[unused].moments = moments;
        let before: Vec<Vec<f32>> = t
            .slots
            .iter()
            .map(|s| s.value.to_f32_vec().unwrap())
            .collect();
        let x: Vec<u32> = (0..64).map(|i| (i * 29 + 1) % 256).collect();
        let y: Vec<u32> = x.iter().map(|v| (v * 5 + 3) % 256).collect();
        let batch = Batch {
            x,
            y,
            batch: 2,
            seq_len: 32,
        };
        let err = t.step_tokens(&[batch]).unwrap_err();
        assert!(
            matches!(&err, OjasError::Shape { detail, .. }
                if *detail == format!("{name} received no gradient")),
            "{err:?}"
        );
        assert_eq!(t.step_count(), 0);
        for (slot, want) in t.slots.iter().zip(&before) {
            assert_eq!(
                &slot.value.to_f32_vec().unwrap(),
                want,
                "{}",
                slot.info.name
            );
        }
    }
}

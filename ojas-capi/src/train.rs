//! Training: TRAIN_OPEN, TRAIN_STEP (sampled or caller tokens), SAVE and
//! RESUME, over `ojas_model::Trainer` on the session's device.
//!
//! Every call holds the session's model for its whole run and ends with
//! [`crate::settled`], so a fault a device deferred is this call's error.
//! A cancelled step commits nothing: the session's backend refuses ops once
//! the call is cancelled, but only ops before the trainer's optimizer
//! (`crate::gate`). A step that fails in its optimizer or finish leaves the
//! trainer poisoned (`ojas:E_POISONED:`); Step, Save and Sample then refuse
//! until a Resume.

use std::path::PathBuf;

use ojas_core::{Backend, OjasError, Tensor};
use ojas_cpu::{scaled_lr, CosineSchedule, LrSchedule, WsdSchedule};
use ojas_data::{Batch, TokenBin};
use ojas_model::{NonFinitePolicy, TrainConfig, Trainer};

use crate::gate::Check;
use crate::load::{self, Verified, PLACEMENT_FIELDS};
use crate::model::{on_model, Build, Model, Weights};
use crate::session;
use crate::wire::{required, tag, Fields, Kind, Reader};

pub const BIN_HEADERLESS: u32 = 0;
pub const BIN_FINEWEB: u32 = 1;

pub const SCHEDULE_COSINE: u32 = 0;
pub const SCHEDULE_WSD: u32 = 1;

pub const STEP_SAMPLED: u32 = 0;
pub const STEP_TOKENS: u32 = 1;

/// Bytes of a TRAIN_STEP result: loss f32, grad norm f32, Muon LR f64,
/// AdamW LR f64, step u64, tokens u64.
pub const STEP_RESULT_BYTES: usize = 40;

const TRAIN_FIELDS: [(u32, &str, Kind); 15] = [
    (tag::TOKEN_BIN, "token_bin", Kind::Str),
    (tag::BIN_FORMAT, "bin_format", Kind::U32),
    (tag::BATCH, "batch", Kind::U32),
    (tag::SEQ, "seq", Kind::U32),
    (tag::ACCUM, "accum", Kind::U32),
    (tag::DATA_SEED, "data_seed", Kind::U64),
    (tag::SCHEDULE, "schedule", Kind::U32),
    (tag::WARMUP, "warmup", Kind::U64),
    (tag::TOTAL, "total", Kind::U64),
    (tag::DECAY_FRAC, "decay_frac", Kind::F64),
    (tag::MATRIX_LR, "matrix_lr", Kind::F64),
    (tag::ADAM_LR, "adam_lr", Kind::F64),
    (tag::GRAD_CLIP, "grad_clip", Kind::F32),
    (tag::ON_NONFINITE, "on_nonfinite", Kind::U32),
    (tag::TOKENIZER_HASH, "tokenizer_hash", Kind::Bytes32),
];

/// A token bin and the config to train on it.
struct Setup {
    bin: TokenBin,
    cfg: TrainConfig,
}

fn usize_of(f: &Fields<'_>, t: u32) -> Result<usize, String> {
    usize::try_from(required(f.u32(t), t)?).map_err(|_| "train: size exceeds usize".to_string())
}

/// Every field is required except `decay_frac` (WSD only, and refused for
/// cosine) and `tokenizer_hash` (zeros). The bin is opened under the root.
fn setup(f: &Fields<'_>) -> Result<Setup, String> {
    let warmup = required(f.u64(tag::WARMUP), tag::WARMUP)?;
    let total = required(f.u64(tag::TOTAL), tag::TOTAL)?;
    let show = |e: OjasError| crate::ojas_error("train", &e);
    let schedule = match required(f.u32(tag::SCHEDULE), tag::SCHEDULE)? {
        SCHEDULE_COSINE => {
            if f.f64(tag::DECAY_FRAC).is_some() {
                return Err(
                    "train: decay_frac is a WSD field; the cosine schedule refuses it".into(),
                );
            }
            LrSchedule::Cosine(CosineSchedule::new(warmup, total).map_err(show)?)
        }
        SCHEDULE_WSD => {
            let decay = required(f.f64(tag::DECAY_FRAC), tag::DECAY_FRAC)?;
            LrSchedule::Wsd(WsdSchedule::new(warmup, total, decay).map_err(show)?)
        }
        other => return Err(format!("train: unknown schedule {other}")),
    };
    let mut cfg = TrainConfig::nanolab(
        usize_of(f, tag::BATCH)?,
        usize_of(f, tag::SEQ)?,
        usize_of(f, tag::ACCUM)?,
        required(f.u64(tag::DATA_SEED), tag::DATA_SEED)?,
        schedule,
    );
    cfg.matrix_lr = required(f.f64(tag::MATRIX_LR), tag::MATRIX_LR)?;
    cfg.adam_lr = required(f.f64(tag::ADAM_LR), tag::ADAM_LR)?;
    cfg.grad_clip = required(f.f32(tag::GRAD_CLIP), tag::GRAD_CLIP)?;
    cfg.on_nonfinite = match required(f.u32(tag::ON_NONFINITE), tag::ON_NONFINITE)? {
        0 => NonFinitePolicy::Abort,
        1 => NonFinitePolicy::SkipBatch,
        other => return Err(format!("train: unknown on_nonfinite {other}")),
    };
    if let Some(hash) = f.bytes32(tag::TOKENIZER_HASH) {
        cfg.tokenizer_hash = hash;
    }
    cfg.validate().map_err(show)?;
    let raw = required(f.str(tag::TOKEN_BIN), tag::TOKEN_BIN)?;
    let format = required(f.u32(tag::BIN_FORMAT), tag::BIN_FORMAT)?;
    let bin = open_bin(raw, format)?;
    Ok(Setup { bin, cfg })
}

/// The token bin `raw` under the root. `ojas_data::TokenBin` opens by path,
/// so it is handed the descriptor of the file opened here with
/// `O_NOFOLLOW` ([`Verified::fd_path`]).
fn open_bin(raw: &str, format: u32) -> Result<TokenBin, String> {
    let file = Verified::open(raw)?;
    let opened = match format {
        BIN_HEADERLESS => TokenBin::open_headerless(&file.fd_path()),
        BIN_FINEWEB => TokenBin::open_fineweb(&file.fd_path()),
        other => return Err(format!("train: unknown bin_format {other}")),
    };
    opened.map_err(|e| format!("token bin: {}", file.explain(e.detail())))
}

/// TRAIN_OPEN: `id: u64`, then the train fields. The session's resident
/// weights become the trainer's (`Trainer::new` copies them; the resident
/// copy is dropped only once the trainer exists). A session that already
/// trains is refused.
pub fn open_request(bytes: &[u8], mut check: Check) -> Result<(), String> {
    check()?;
    let mut r = Reader::new(bytes);
    let id = r.u64()?;
    let f = Fields::read(&mut r, &TRAIN_FIELDS)?;
    r.finish()?;
    let session = session::require(id)?;
    let setup = setup(&f)?;
    let mut guard = session.lock_state()?;
    let state = &mut *guard;
    let armed = state.slot.arm(check);
    on_model!(&mut state.engine, m => {
        let opened = open_on(m, setup).map_err(|e| armed.report("train_open", &e));
        crate::settled("train_open", &m.backend, opened)
    })
}

fn open_on<B: Backend + Clone>(m: &mut Model<B>, setup: Setup) -> Result<(), OjasError> {
    let Weights::Resident(weights) = &m.weights else {
        return Err(OjasError::Unsupported {
            op: "train_open",
            detail: "this model already has a trainer".to_string(),
        });
    };
    // `Trainer::new` copies host tensors; a device tensor comes back first.
    let host = weights
        .iter()
        .map(|t| match t.device() {
            Some(_) => m.backend.download(t),
            None => Ok(t.clone()),
        })
        .collect::<Result<Vec<Tensor>, OjasError>>()?;
    let trainer = Trainer::new(m.backend.clone(), m.spec, &host, setup.bin, setup.cfg)?;
    m.weights = Weights::Training(Box::new(trainer));
    Ok(())
}

/// What one step returns.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepOut {
    pub loss: f32,
    pub grad_norm: f32,
    /// The Muon learning rate this step applied (`matrix_lr` x schedule).
    pub matrix_lr: f64,
    /// The AdamW learning rate this step applied (`adam_lr` x schedule).
    pub adam_lr: f64,
    /// Completed steps after this one.
    pub step: u64,
    pub tokens: u64,
}

impl StepOut {
    pub fn to_bytes(self) -> Vec<u8> {
        let mut out = Vec::with_capacity(STEP_RESULT_BYTES);
        out.extend_from_slice(&self.loss.to_le_bytes());
        out.extend_from_slice(&self.grad_norm.to_le_bytes());
        out.extend_from_slice(&self.matrix_lr.to_le_bytes());
        out.extend_from_slice(&self.adam_lr.to_le_bytes());
        out.extend_from_slice(&self.step.to_le_bytes());
        out.extend_from_slice(&self.tokens.to_le_bytes());
        out
    }
}

/// TRAIN_STEP: `id: u64, mode: u32`. Mode [`STEP_SAMPLED`] takes nothing
/// more and steps on the trainer's next K sampled micro-batches. Mode
/// [`STEP_TOKENS`] takes `k: u32, seq: u32`, then per micro-batch
/// `rows: u32, x: [u32; rows*seq], y: [u32; rows*seq]`; the cursor does not
/// move. `seq` must be the trainer's.
pub fn step_request(bytes: &[u8], mut check: Check) -> Result<StepOut, String> {
    check()?;
    let mut r = Reader::new(bytes);
    let id = r.u64()?;
    let batches = match r.u32()? {
        STEP_SAMPLED => None,
        STEP_TOKENS => Some(read_batches(&mut r)?),
        other => return Err(format!("train_step: shape: unknown mode {other}")),
    };
    r.finish()?;
    let session = session::require(id)?;
    let mut guard = session.lock_state()?;
    let state = &mut *guard;
    let armed = state.slot.arm(check);
    on_model!(&mut state.engine, m => {
        let stepped = step_on(m, batches.as_deref()).map_err(|e| armed.report("train_step", &e));
        crate::settled("train_step", &m.backend, stepped)
    })
}

fn read_batches(r: &mut Reader<'_>) -> Result<Vec<Batch>, String> {
    let k = r.u32()?;
    if k == 0 {
        return Err("train_step: shape: no micro-batches".to_string());
    }
    let seq = usize::try_from(r.u32()?).map_err(|_| "train_step: shape: seq".to_string())?;
    // Not reserved from `k`: each batch is read, and length-checked, in turn.
    let mut batches = Vec::new();
    for _ in 0..k {
        let rows = usize::try_from(r.u32()?).map_err(|_| "train_step: shape: rows".to_string())?;
        let n = rows
            .checked_mul(seq)
            .ok_or_else(|| "train_step: shape: rows * seq overflows".to_string())?;
        let x = r.values(n, u32::from_le_bytes)?;
        let y = r.values(n, u32::from_le_bytes)?;
        batches.push(Batch {
            x,
            y,
            batch: rows,
            seq_len: seq,
        });
    }
    Ok(batches)
}

fn trainer_mut<B: Backend>(m: &mut Model<B>) -> Result<&mut Trainer<B>, OjasError> {
    match &mut m.weights {
        Weights::Training(t) => Ok(t),
        Weights::Resident(_) => Err(OjasError::Unsupported {
            op: "train",
            detail: "no trainer is open on this model; call TrainOpen or Resume".to_string(),
        }),
    }
}

fn step_on<B: Backend>(m: &mut Model<B>, batches: Option<&[Batch]>) -> Result<StepOut, OjasError> {
    let trainer = trainer_mut(m)?;
    let report = match batches {
        None => trainer.step()?,
        Some(batches) => trainer.step_tokens(batches)?,
    };
    let cfg = trainer.config();
    Ok(StepOut {
        loss: report.loss,
        grad_norm: report.grad_norm,
        matrix_lr: scaled_lr(cfg.matrix_lr, report.lr_multiplier)?,
        adam_lr: scaled_lr(cfg.adam_lr, report.lr_multiplier)?,
        step: report.step,
        tokens: report.tokens,
    })
}

/// SAVE: `id: u64`, then the checkpoint directory (the rest of the payload),
/// relative to the root. It may not exist yet; its parent must.
/// `Trainer::save` writes a staging directory and swaps it in whole.
pub fn save_request(bytes: &[u8], mut check: Check) -> Result<(), String> {
    check()?;
    let mut r = Reader::new(bytes);
    let id = r.u64()?;
    let raw = r.rest_str()?;
    let dir = load::resolve_dir_under_root(&session::root()?, raw, false)?;
    let session = session::require(id)?;
    let mut guard = session.lock_state()?;
    let state = &mut *guard;
    let armed = state.slot.arm(check);
    on_model!(&mut state.engine, m => {
        let saved = trainer_mut(m)
            .and_then(|t| t.save(&dir))
            .map_err(|e| armed.report("save", &e));
        crate::settled("save", &m.backend, saved)
    })
}

/// A trainer restored from a checkpoint directory.
struct FromCheckpoint {
    dir: PathBuf,
    setup: Setup,
}

impl Build for FromCheckpoint {
    fn build<B: Backend + Clone>(self, backend: B) -> Result<Model<B>, OjasError> {
        let trainer =
            Trainer::resume_from(backend.clone(), &self.dir, self.setup.bin, self.setup.cfg)?;
        Ok(Model {
            backend,
            spec: *trainer.spec(),
            weights: Weights::Training(Box::new(trainer)),
        })
    }
}

/// RESUME: `{path (the checkpoint directory), placement fields, train
/// fields}`. The train config must equal the saved one field for field and
/// the tokenizer hash the saved hash (`Trainer::resume_from`). A new session
/// is returned; nothing is added on a refusal.
pub fn resume_request(bytes: &[u8], mut check: Check) -> Result<session::Session, String> {
    check()?;
    let mut r = Reader::new(bytes);
    let mut allowed = PLACEMENT_FIELDS.to_vec();
    allowed.extend(TRAIN_FIELDS);
    allowed.push((tag::PATH, "path", Kind::Str));
    let f = Fields::read(&mut r, &allowed)?;
    r.finish()?;
    let placement = load::placement(&f)?;
    let raw = required(f.str(tag::PATH), tag::PATH)?;
    let dir = load::resolve_dir_under_root(&session::root()?, raw, true)?;
    session::check_room()?;
    let setup = setup(&f)?;
    load::create(
        Some(dir.clone()),
        &placement,
        check,
        FromCheckpoint { dir, setup },
    )
}

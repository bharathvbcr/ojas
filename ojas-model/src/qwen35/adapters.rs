//! The adapter checkpoint of the Qwen3.5 Tape LoRA track: the trainable
//! entries by name, their AdamW moments, and what a resume needs to
//! continue the same run or refuse. `docs/adapter-checkpoint.md` is the
//! layout contract Lappi's converter reads; this module is its writer and
//! reader and the two must agree (the doc is the contract, the tests pin it).
//!
//! A checkpoint is a directory of two safetensors files, written to a
//! staging sibling and swapped in whole with [`ojas_io::replace_dir_with_if`]
//! (an existing target is replaced only if it is empty or an earlier adapter
//! checkpoint), so a kill at any instant leaves the old directory or the new
//! one, never a mix:
//!
//! - [`ADAPTERS_FILE`]: one `F32` tensor per entry under its own name, and
//!   every scalar of the checkpoint as safetensors metadata.
//! - [`MOMENTS_FILE`]: `adam_m.<name>` and `adam_v.<name>` per entry (the
//!   names `crate::checkpoint` uses for the nanolab optimiser file).
//!
//! # Names
//!
//! A LoRA entry is `<module path>.lora_A.weight` (`A`, `[r, in]`) and
//! `<module path>.lora_B.weight` (`B`, `[out, r]`), the PEFT convention
//! (`crate::lora` follows PEFT's init and shapes). The module path is the
//! Hugging Face tensor name without `.weight`
//! ([`super::LoraTarget::module`]), e.g.
//! `model.language_model.layers.3.self_attn.q_proj.lora_A.weight`; a fused
//! tensor's `B` is stored fused ([`super::LoraTarget::fuse_b`]), as PEFT
//! shapes it, not as the model's parts. The host entries (D17, the span
//! head) have the fixed names [`HOST_ENTRY_NAMES`], Lappi's own; any other
//! name is refused on save and on read. [`lora_a_name`] and [`lora_b_name`]
//! build the two LoRA names. The base is named by one file's name and
//! sha256 ([`BaseIdentity`]); a sharded base is refused.
//!
//! # Refusals
//!
//! Save refuses a LoRA dropout other than 0 ([`AdapterError::Dropout`]):
//! no dropout key is stored, so a resume could not reproduce a dropout
//! stream. It also refuses duplicate names, a shape that disagrees with its
//! values or moments, non-finite values or moments, a negative second
//! moment, and malformed identity fields.
//!
//! Read refuses, in this order, each with its own [`AdapterError`]: an
//! unreadable or truncated file, a layout version that is not
//! [`ADAPTER_LAYOUT_VERSION`], missing or malformed metadata, a different
//! frozen base (sha256 or quantizer record), a different LoRA rank or alpha,
//! moments that disagree with the adapters' step, a missing or extra entry
//! in either file, a dtype or shape that is not the expected one, any
//! non-finite value, a payload whose FNV-1a does not match the one recorded,
//! and a negative second moment.
//! The payload hash is an integrity check against truncation and bit rot,
//! not an authenticity one.

use std::cell::RefCell;
use std::fmt;
use std::fs::OpenOptions;
use std::io::BufWriter;
use std::path::Path;

use ojas_core::{Budget, OjasError, Tensor};
use ojas_io::{
    recover_replaced_dir, replace_dir_with_if, IoError, SafeTensors, SafeTensorsWriter, StDtype,
    TensorSpec,
};

use crate::checkpoint::check_target_dir;
use crate::init::{fnv1a, fnv1a_extend};
use crate::lora::LoraConfig;

/// The layout version this module writes and the only one it reads.
pub const ADAPTER_LAYOUT_VERSION: u32 = 1;
/// Entries by name, plus the checkpoint's scalars as metadata.
pub const ADAPTERS_FILE: &str = "adapters.safetensors";
/// `adam_m.<name>` and `adam_v.<name>` per entry.
pub const MOMENTS_FILE: &str = "moments.safetensors";

/// Layout version, decimal. Both files.
pub const KEY_LAYOUT: &str = "ojas.adapter.layout";
/// Completed optimiser steps, decimal: the schedule step, and the `step`
/// argument of the next `adamw_step`. Both files.
pub const KEY_STEP: &str = "ojas.adapter.step";
/// FNV-1a 64 of the file's tensor bytes, 16 lowercase hex digits. Both files.
pub const KEY_PAYLOAD: &str = "ojas.adapter.payload_fnv1a64";
/// `moments.safetensors` only: the adapters file's [`KEY_PAYLOAD`], binding
/// the two files together.
pub const KEY_PAIR: &str = "ojas.adapter.pair";
/// Data-order RNG state, lowercase hex of the caller's bytes.
pub const KEY_RNG: &str = "ojas.adapter.rng_state";
/// Consumed-data digest, the caller's lowercase hex string.
pub const KEY_DIGEST: &str = "ojas.adapter.consumed_digest";
/// sha256 of the frozen base file, 64 lowercase hex digits.
pub const KEY_BASE: &str = "ojas.adapter.base_sha256";
/// Name of the (single) base file `KEY_BASE` is the sha256 of.
pub const KEY_BASE_FILE: &str = "ojas.adapter.base_file";
/// LoRA rank, decimal.
pub const KEY_RANK: &str = "ojas.adapter.lora_rank";
/// LoRA alpha, Rust's shortest round-trip decimal for the f32.
pub const KEY_ALPHA: &str = "ojas.adapter.lora_alpha";
/// Quantizer record: the three keys are all present or all absent.
pub const KEY_QUANT_FORMAT: &str = "ojas.adapter.quant.format";
pub const KEY_QUANT_GROUP: &str = "ojas.adapter.quant.group_size";
pub const KEY_QUANT_MLX: &str = "ojas.adapter.quant.mlx_version";

const M_PREFIX: &str = "adam_m.";
const V_PREFIX: &str = "adam_v.";
/// Longest entry name, field string or hex digest accepted, in bytes.
const MAX_FIELD_BYTES: usize = 4096;
/// Longest RNG state accepted, in bytes.
const MAX_RNG_BYTES: usize = 4096;
/// Values written per `write` call.
const WRITE_PIECE: usize = 16 * 1024;

/// The host entries layout 1 may hold, by their names in Lappi's own
/// checkpoint (read from Lappi: `qd-export/src/layout.rs` `D17_PREFIX` +
/// `D17_TENSOR` = `AnswerRowDelta.delta`, and `SPAN_PREFIX` + the torch
/// state-dict keys of `qd-train/src/span_head.rs` `PARAMETER_NAMES`). Any
/// other name that is not a LoRA `A` or `B` is refused on save and read.
pub const HOST_ENTRY_NAMES: [&str; 5] = [
    "d17.delta",
    "span_head.start_proj.weight",
    "span_head.end_proj.weight",
    "span_head.abstain_start",
    "span_head.abstain_end",
];

/// `<module>.lora_A.weight`.
pub fn lora_a_name(module: &str) -> String {
    format!("{module}.lora_A.weight")
}

/// `<module>.lora_B.weight`.
pub fn lora_b_name(module: &str) -> String {
    format!("{module}.lora_B.weight")
}

/// Why a save or a resume was refused.
#[derive(Debug)]
pub enum AdapterError {
    /// Save: the LoRA dropout is not 0.
    Dropout(f32),
    /// An entry that cannot be written or was read inconsistently.
    InvalidEntry { name: String, detail: String },
    /// An identity or state field that is malformed.
    InvalidField { field: &'static str, detail: String },
    /// Save: the target exists and is not an earlier adapter checkpoint.
    TargetRefused(String),
    /// A filesystem or safetensors failure, including a missing or
    /// truncated file.
    Unreadable { file: String, detail: String },
    /// The layout version is not [`ADAPTER_LAYOUT_VERSION`].
    UnknownVersion { found: String, supported: u32 },
    /// Metadata that is missing or does not parse, or a payload that does
    /// not match its recorded hash.
    Corrupt { file: String, detail: String },
    /// The frozen base's sha256 differs.
    BaseMismatch { expected: String, found: String },
    /// The base file's recorded name differs.
    BaseFileMismatch { expected: String, found: String },
    /// The base is one shard (or the index) of a multi-file checkpoint,
    /// which layout 1 does not cover.
    ShardedBase { file_name: String },
    /// An entry name that is neither a LoRA `A`/`B` nor one of
    /// [`HOST_ENTRY_NAMES`].
    UnknownEntryName(String),
    /// The quantizer record differs, or is present on one side only.
    QuantizerMismatch {
        expected: Option<QuantizerRecord>,
        found: Option<QuantizerRecord>,
    },
    /// The LoRA rank or alpha differs.
    LoraMismatch { detail: String },
    /// An expected entry the checkpoint does not hold.
    MissingEntry(String),
    /// An entry the checkpoint holds that nothing expected.
    ExtraEntry(String),
    /// An entry's dtype or shape is not the expected one.
    LayoutMismatch { name: String, detail: String },
    /// The two files were written at different steps.
    StepMismatch { adapters: u64, moments: u64 },
    /// The moments file was written beside a different adapters file than
    /// this one (another run, or another save at the same step).
    PairMismatch {
        adapters: String,
        moments_pair: String,
    },
    /// A tensor layer failure while moving values to or from host.
    Core(OjasError),
}

impl fmt::Display for AdapterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dropout(p) => write!(
                f,
                "adapter checkpoint: LoRA dropout {p} is not 0; no dropout key is stored, \
                 so the run could not resume"
            ),
            Self::InvalidEntry { name, detail } => {
                write!(f, "adapter checkpoint: entry {name:?}: {detail}")
            }
            Self::InvalidField { field, detail } => {
                write!(f, "adapter checkpoint: {field}: {detail}")
            }
            Self::TargetRefused(detail) => f.write_str(detail),
            Self::Unreadable { file, detail } => {
                write!(f, "adapter checkpoint: {file}: {detail}")
            }
            Self::UnknownVersion { found, supported } => write!(
                f,
                "adapter checkpoint: layout version {found:?}, this reader supports {supported}"
            ),
            Self::Corrupt { file, detail } => {
                write!(f, "adapter checkpoint: {file} is corrupt: {detail}")
            }
            Self::BaseMismatch { expected, found } => write!(
                f,
                "adapter checkpoint: frozen base sha256 {found}, the run's base is {expected}"
            ),
            Self::QuantizerMismatch { expected, found } => write!(
                f,
                "adapter checkpoint: quantizer record {found:?}, the run's is {expected:?}"
            ),
            Self::LoraMismatch { detail } => write!(f, "adapter checkpoint: LoRA {detail}"),
            Self::MissingEntry(n) => write!(f, "adapter checkpoint: entry {n:?} is missing"),
            Self::ExtraEntry(n) => write!(f, "adapter checkpoint: unexpected entry {n:?}"),
            Self::LayoutMismatch { name, detail } => {
                write!(f, "adapter checkpoint: entry {name:?}: {detail}")
            }
            Self::BaseFileMismatch { expected, found } => write!(
                f,
                "adapter checkpoint: frozen base file {found:?}, the run's base is {expected:?}"
            ),
            Self::ShardedBase { file_name } => write!(
                f,
                "adapter checkpoint: base {file_name:?} is part of a sharded checkpoint; \
                 layout 1 names a single base file"
            ),
            Self::UnknownEntryName(n) => write!(
                f,
                "adapter checkpoint: entry name {n:?} is not a LoRA lora_A/lora_B weight or \
                 one of the fixed host entries"
            ),
            Self::PairMismatch {
                adapters,
                moments_pair,
            } => write!(
                f,
                "adapter checkpoint: {MOMENTS_FILE} was written beside adapters file \
                 {moments_pair}, but {ADAPTERS_FILE} is {adapters}"
            ),
            Self::StepMismatch { adapters, moments } => write!(
                f,
                "adapter checkpoint: {ADAPTERS_FILE} is at step {adapters}, \
                 {MOMENTS_FILE} at step {moments}"
            ),
            Self::Core(e) => write!(f, "adapter checkpoint: {e}"),
        }
    }
}

impl std::error::Error for AdapterError {}

impl From<OjasError> for AdapterError {
    fn from(e: OjasError) -> Self {
        Self::Core(e)
    }
}

/// The quantizer that produced a fake-quant base file, copied by the
/// caller from the arm's pre-registration. Strings, compared exactly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuantizerRecord {
    pub format: String,
    pub group_size: String,
    pub mlx_version: String,
}

/// The frozen base the adapters were trained against: sha256 of the base
/// file (lowercase hex, 64 digits) and, for the quantised arm, the
/// quantizer record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BaseIdentity {
    /// The base file's own name (no directory), recorded beside its hash.
    /// Layout 1 covers a single-file base only: a shard name
    /// (`...-00001-of-00004.safetensors`) or an index
    /// (`...safetensors.index.json`) is refused with
    /// [`AdapterError::ShardedBase`], since one sha256 would then name an
    /// ambiguous set of files.
    pub file_name: String,
    pub sha256: String,
    pub quantizer: Option<QuantizerRecord>,
}

/// The data order's state: the caller's RNG bytes and consumed-data digest
/// (lowercase hex). Neither is interpreted here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DataState {
    pub rng_state: Vec<u8>,
    pub consumed_digest: String,
}

/// One trainable entry on the host: values and both AdamW moments, f32,
/// row-major, all of `shape`'s element count.
#[derive(Clone, Debug, PartialEq)]
pub struct AdapterEntry {
    pub name: String,
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
    pub moment1: Vec<f32>,
    pub moment2: Vec<f32>,
}

impl AdapterEntry {
    /// An entry read back from `param` and its moments (any placement; a
    /// device tensor is downloaded, charged to `budget`, one at a time).
    pub fn from_tensors(
        name: &str,
        param: &Tensor,
        moment1: &Tensor,
        moment2: &Tensor,
        budget: &Budget,
    ) -> Result<Self, AdapterError> {
        let host = |t: &Tensor| -> Result<Vec<f32>, AdapterError> {
            if t.shape() != param.shape() {
                return Err(AdapterError::InvalidEntry {
                    name: name.to_string(),
                    detail: format!(
                        "shape {:?} differs from the parameter's {:?}",
                        t.shape(),
                        param.shape()
                    ),
                });
            }
            Ok(t.to_host(budget)?.to_f32_vec()?)
        };
        Ok(Self {
            name: name.to_string(),
            shape: param.shape().to_vec(),
            values: host(param)?,
            moment1: host(moment1)?,
            moment2: host(moment2)?,
        })
    }

    /// Host tensors of the values and the two moments, charged to `budget`.
    pub fn to_tensors(&self, budget: &Budget) -> Result<(Tensor, Tensor, Tensor), AdapterError> {
        Ok((
            Tensor::from_f32(&self.values, &self.shape, budget)?,
            Tensor::from_f32(&self.moment1, &self.shape, budget)?,
            Tensor::from_f32(&self.moment2, &self.shape, budget)?,
        ))
    }
}

/// A whole adapter checkpoint in memory.
#[derive(Clone, Debug, PartialEq)]
pub struct AdapterCheckpoint {
    /// Rank and alpha are stored; the dropout is never stored and must be 0
    /// to save. A read returns it as 0.
    pub lora: LoraConfig,
    /// Completed optimiser steps: the schedule step and the next
    /// `adamw_step`'s `step` argument.
    pub step: u64,
    pub data: DataState,
    pub base: BaseIdentity,
    pub entries: Vec<AdapterEntry>,
}

/// One entry a resume expects.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpectedEntry {
    pub name: String,
    pub shape: Vec<usize>,
}

/// What a resume demands of the checkpoint before it reads a value.
#[derive(Clone, Debug)]
pub struct Expected {
    pub base: BaseIdentity,
    /// Rank and alpha must equal the saved ones; the dropout is not
    /// compared (it is never stored).
    pub lora: LoraConfig,
    pub entries: Vec<ExpectedEntry>,
}

fn invalid_field(field: &'static str, detail: impl Into<String>) -> AdapterError {
    AdapterError::InvalidField {
        field,
        detail: detail.into(),
    }
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(char::from(DIGITS[usize::from(b >> 4)]));
        s.push(char::from(DIGITS[usize::from(b & 15)]));
    }
    s
}

fn is_lower_hex(s: &str) -> bool {
    s.bytes()
        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn unhex(s: &str) -> Option<Vec<u8>> {
    if s.len() % 2 != 0 || !is_lower_hex(s) {
        return None;
    }
    let nibble = |b: u8| {
        if b.is_ascii_digit() {
            b - b'0'
        } else {
            b - b'a' + 10
        }
    };
    Some(
        s.as_bytes()
            .chunks_exact(2)
            .map(|p| nibble(p[0]) << 4 | nibble(p[1]))
            .collect(),
    )
}

fn check_string(field: &'static str, s: &str) -> Result<(), AdapterError> {
    if s.is_empty() || s.len() > MAX_FIELD_BYTES {
        return Err(invalid_field(
            field,
            format!("{} bytes, must be 1..={MAX_FIELD_BYTES}", s.len()),
        ));
    }
    Ok(())
}

/// Whether `name` is a shard of a multi-file Hugging Face checkpoint
/// (`model-00001-of-00004.safetensors`) or its index
/// (`model.safetensors.index.json`).
fn is_sharded(name: &str) -> bool {
    if name.ends_with(".index.json") {
        return true;
    }
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    // `-00001-of-00001` is the whole checkpoint in one file (how the local
    // Qwen3.5 2B-Base and 0.8B-Base snapshots are named), so one sha256 names
    // it; every other `-N-of-M` is one shard of several.
    let one = |s: &str| s.trim_start_matches('0') == "1";
    name.match_indices("-of-").any(|(at, _)| {
        let left = name[..at].rsplit('-').next().unwrap_or("");
        let right = name[at + 4..].split('.').next().unwrap_or("");
        digits(left) && digits(right) && !(one(left) && one(right))
    })
}

/// Whether `name` is an entry layout 1 may hold: a LoRA `A` or `B` of some
/// module, or one of [`HOST_ENTRY_NAMES`].
fn is_known_entry_name(name: &str) -> bool {
    HOST_ENTRY_NAMES.contains(&name)
        || [".lora_A.weight", ".lora_B.weight"]
            .iter()
            .any(|suffix| name.strip_suffix(suffix).is_some_and(|m| !m.is_empty()))
}

fn check_base(base: &BaseIdentity) -> Result<(), AdapterError> {
    let file = &base.file_name;
    if file.is_empty()
        || file.len() > 255
        || file == "."
        || file == ".."
        || file.contains(['/', '\\', '\0'])
    {
        return Err(invalid_field(
            "base file_name",
            "must be one plain file name of 1 to 255 bytes",
        ));
    }
    if is_sharded(file) {
        return Err(AdapterError::ShardedBase {
            file_name: file.clone(),
        });
    }
    if base.sha256.len() != 64 || !is_lower_hex(&base.sha256) {
        return Err(invalid_field(
            "base sha256",
            "must be 64 lowercase hex digits",
        ));
    }
    if let Some(q) = &base.quantizer {
        check_string("quantizer format", &q.format)?;
        check_string("quantizer group_size", &q.group_size)?;
        check_string("quantizer mlx_version", &q.mlx_version)?;
    }
    Ok(())
}

fn numel(name: &str, shape: &[usize]) -> Result<usize, AdapterError> {
    let n = shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .filter(|&n| n > 0)
        .ok_or_else(|| AdapterError::InvalidEntry {
            name: name.to_string(),
            detail: format!("shape {shape:?} is empty or overflows"),
        })?;
    Ok(n)
}

fn check_entry(e: &AdapterEntry) -> Result<(), AdapterError> {
    let bad = |detail: String| AdapterError::InvalidEntry {
        name: e.name.clone(),
        detail,
    };
    if e.name.is_empty() || e.name.len() > MAX_FIELD_BYTES {
        return Err(bad(format!(
            "name of {} bytes, must be 1..={MAX_FIELD_BYTES}",
            e.name.len()
        )));
    }
    if e.name == "__metadata__" {
        return Err(bad("reserved name".to_string()));
    }
    if !is_known_entry_name(&e.name) {
        return Err(AdapterError::UnknownEntryName(e.name.clone()));
    }
    let n = numel(&e.name, &e.shape)?;
    for (what, v) in [
        ("values", &e.values),
        ("first moment", &e.moment1),
        ("second moment", &e.moment2),
    ] {
        if v.len() != n {
            return Err(bad(format!(
                "{what} hold {} elements, shape {:?} has {n}",
                v.len(),
                e.shape
            )));
        }
        if let Some(i) = v.iter().position(|x| !x.is_finite()) {
            return Err(bad(format!("{what}[{i}] is {}", v[i])));
        }
    }
    if let Some(i) = e.moment2.iter().position(|x| *x < 0.0) {
        return Err(bad(format!("second moment[{i}] is negative")));
    }
    Ok(())
}

/// The file's integrity hash: FNV-1a 64 over the little-endian bytes of
/// `tensors` in the order given, then over every metadata pair except
/// [`KEY_PAYLOAD`] itself, sorted by key, each as `len(key)` (u64 LE), key,
/// `len(value)` (u64 LE), value. A flipped rng state or digest that is
/// still valid hex changes it.
fn integrity_hash<'a>(
    tensors: impl Iterator<Item = &'a [f32]>,
    metadata: impl Iterator<Item = (&'a str, &'a str)>,
) -> u64 {
    let mut h = fnv1a(&[]);
    for t in tensors {
        for x in t {
            h = fnv1a_extend(h, &x.to_le_bytes());
        }
    }
    let mut pairs: Vec<(&str, &str)> = metadata.filter(|(k, _)| *k != KEY_PAYLOAD).collect();
    pairs.sort_unstable();
    for (k, v) in pairs {
        for s in [k, v] {
            h = fnv1a_extend(h, &(s.len() as u64).to_le_bytes());
            h = fnv1a_extend(h, s.as_bytes());
        }
    }
    h
}

fn shape_u64(shape: &[usize]) -> Vec<u64> {
    shape.iter().map(|&d| d as u64).collect()
}

/// One file's tensors, sorted by name, ready to write.
struct Staged<'a> {
    name: String,
    shape: Vec<u64>,
    values: &'a [f32],
}

fn write_file(
    path: &Path,
    mut tensors: Vec<Staged<'_>>,
    metadata: &mut Vec<(String, String)>,
) -> Result<String, IoError> {
    tensors.sort_by(|a, b| a.name.cmp(&b.name));
    let hash = format!(
        "{:016x}",
        integrity_hash(
            tensors.iter().map(|t| t.values),
            metadata.iter().map(|(k, v)| (k.as_str(), v.as_str()))
        )
    );
    metadata.push((KEY_PAYLOAD.to_string(), hash.clone()));
    let meta: Vec<(&str, &str)> = metadata
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    let specs: Vec<TensorSpec<'_>> = tensors
        .iter()
        .map(|t| TensorSpec {
            name: &t.name,
            dtype: StDtype::F32,
            shape: &t.shape,
        })
        .collect();
    let what = path.display();
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| IoError::new(format!("{what}: {e}")))?;
    let mut writer = SafeTensorsWriter::new(BufWriter::new(file), &specs, &meta)
        .map_err(|e| IoError::new(format!("{what}: {}", e.detail())))?;
    let mut le = Vec::with_capacity(WRITE_PIECE * 4);
    for t in &tensors {
        for piece in t.values.chunks(WRITE_PIECE) {
            le.clear();
            for x in piece {
                le.extend_from_slice(&x.to_le_bytes());
            }
            writer
                .write(&t.name, &le)
                .map_err(|e| IoError::new(format!("{what}: {}", e.detail())))?;
        }
    }
    writer
        .finish()
        .map_err(|e| IoError::new(format!("{what}: {}", e.detail())))?;
    Ok(hash)
}

/// Write `ckpt` to the directory `dir`, replacing an earlier adapter
/// checkpoint (or an empty directory, or nothing) whole. Every refusal in
/// the module docs happens before the directory is touched except a
/// filesystem failure, which leaves the old directory in place.
pub fn write_adapter_checkpoint(dir: &Path, ckpt: &AdapterCheckpoint) -> Result<(), AdapterError> {
    if ckpt.lora.dropout() != 0.0 {
        return Err(AdapterError::Dropout(ckpt.lora.dropout()));
    }
    check_base(&ckpt.base)?;
    if ckpt.data.rng_state.is_empty() || ckpt.data.rng_state.len() > MAX_RNG_BYTES {
        return Err(invalid_field(
            "rng_state",
            format!(
                "{} bytes, must be 1..={MAX_RNG_BYTES}",
                ckpt.data.rng_state.len()
            ),
        ));
    }
    let digest = &ckpt.data.consumed_digest;
    if digest.is_empty()
        || digest.len() > MAX_FIELD_BYTES
        || digest.len() % 2 != 0
        || !is_lower_hex(digest)
    {
        return Err(invalid_field(
            "consumed_digest",
            "must be non-empty lowercase hex of even length",
        ));
    }
    if ckpt.entries.is_empty() {
        return Err(invalid_field("entries", "a checkpoint with no entries"));
    }
    let mut seen = std::collections::BTreeSet::new();
    for e in &ckpt.entries {
        check_entry(e)?;
        if !seen.insert(e.name.as_str()) {
            return Err(AdapterError::InvalidEntry {
                name: e.name.clone(),
                detail: "duplicate name".to_string(),
            });
        }
    }

    let step = ckpt.step.to_string();
    let common = |extra: &[(&str, String)]| -> Vec<(String, String)> {
        let mut m = vec![
            (KEY_LAYOUT.to_string(), ADAPTER_LAYOUT_VERSION.to_string()),
            (KEY_STEP.to_string(), step.clone()),
        ];
        m.extend(extra.iter().map(|(k, v)| (k.to_string(), v.clone())));
        m
    };
    let mut adapter_meta = vec![
        (KEY_RNG, hex(&ckpt.data.rng_state)),
        (KEY_DIGEST, ckpt.data.consumed_digest.clone()),
        (KEY_BASE, ckpt.base.sha256.clone()),
        (KEY_BASE_FILE, ckpt.base.file_name.clone()),
        (KEY_RANK, ckpt.lora.rank().to_string()),
        (KEY_ALPHA, ckpt.lora.alpha().to_string()),
    ];
    if let Some(q) = &ckpt.base.quantizer {
        adapter_meta.push((KEY_QUANT_FORMAT, q.format.clone()));
        adapter_meta.push((KEY_QUANT_GROUP, q.group_size.clone()));
        adapter_meta.push((KEY_QUANT_MLX, q.mlx_version.clone()));
    }
    let mut adapter_meta = common(&adapter_meta);
    let mut moment_meta = common(&[]);

    let adapters: Vec<Staged<'_>> = ckpt
        .entries
        .iter()
        .map(|e| Staged {
            name: e.name.clone(),
            shape: shape_u64(&e.shape),
            values: &e.values,
        })
        .collect();
    let mut moments: Vec<Staged<'_>> = Vec::with_capacity(ckpt.entries.len() * 2);
    for e in &ckpt.entries {
        moments.push(Staged {
            name: format!("{M_PREFIX}{}", e.name),
            shape: shape_u64(&e.shape),
            values: &e.moment1,
        });
        moments.push(Staged {
            name: format!("{V_PREFIX}{}", e.name),
            shape: shape_u64(&e.shape),
            values: &e.moment2,
        });
    }

    let refused: RefCell<Option<String>> = RefCell::new(None);
    let swapped = replace_dir_with_if(
        dir,
        |existing| {
            check_target_dir(existing, &[ADAPTERS_FILE, MOMENTS_FILE], ADAPTERS_FILE).map_err(|d| {
                *refused.borrow_mut() = Some(d.clone());
                IoError::new(d)
            })
        },
        |stage| {
            let pair = write_file(&stage.join(ADAPTERS_FILE), adapters, &mut adapter_meta)?;
            moment_meta.push((KEY_PAIR.to_string(), pair));
            write_file(&stage.join(MOMENTS_FILE), moments, &mut moment_meta).map(drop)
        },
    );
    match (swapped, refused.into_inner()) {
        (Ok(()), _) => Ok(()),
        (Err(_), Some(detail)) => Err(AdapterError::TargetRefused(detail)),
        (Err(e), None) => Err(AdapterError::Unreadable {
            file: dir.display().to_string(),
            detail: e.detail().to_string(),
        }),
    }
}

/// An opened file with its metadata reader.
struct Opened {
    file: &'static str,
    st: SafeTensors<'static>,
}

impl Opened {
    fn open(dir: &Path, file: &'static str) -> Result<Self, AdapterError> {
        let st = SafeTensors::open(&dir.join(file)).map_err(|e| AdapterError::Unreadable {
            file: file.to_string(),
            detail: e.detail().to_string(),
        })?;
        Ok(Self { file, st })
    }

    fn corrupt(&self, detail: impl Into<String>) -> AdapterError {
        AdapterError::Corrupt {
            file: self.file.to_string(),
            detail: detail.into(),
        }
    }

    fn meta(&self, key: &str) -> Result<&str, AdapterError> {
        self.st
            .metadata()
            .get(key)
            .map(String::as_str)
            .ok_or_else(|| self.corrupt(format!("metadata key {key} is missing")))
    }

    fn optional(&self, key: &str) -> Option<&str> {
        self.st.metadata().get(key).map(String::as_str)
    }

    /// Layout version (checked first) and step.
    fn version_and_step(&self) -> Result<u64, AdapterError> {
        let found = self.meta(KEY_LAYOUT)?;
        if found != ADAPTER_LAYOUT_VERSION.to_string() {
            return Err(AdapterError::UnknownVersion {
                found: found.to_string(),
                supported: ADAPTER_LAYOUT_VERSION,
            });
        }
        self.canonical_u64(KEY_STEP)
    }

    /// `key` as a decimal u64 in the one spelling the writer produces: no
    /// sign, no leading zeros, no whitespace.
    fn canonical_u64(&self, key: &str) -> Result<u64, AdapterError> {
        let text = self.meta(key)?;
        let n = text
            .parse::<u64>()
            .map_err(|e| self.corrupt(format!("{key}: {e}")))?;
        if n.to_string() != text {
            return Err(self.corrupt(format!("{key}: {text:?} is not canonical decimal")));
        }
        Ok(n)
    }

    /// Every tensor in name order, validated against `want` (name, shape),
    /// finite, with the payload hash checked. Returns values in `want` order.
    fn read_all(&self, want: &[(String, Vec<usize>)]) -> Result<Vec<Vec<f32>>, AdapterError> {
        let wanted: std::collections::BTreeMap<&str, &[usize]> = want
            .iter()
            .map(|(n, s)| (n.as_str(), s.as_slice()))
            .collect();
        for (name, _) in want {
            if self.st.info(name).is_err() {
                return Err(AdapterError::MissingEntry(name.clone()));
            }
        }
        for name in self.st.names() {
            if !wanted.contains_key(name) {
                return Err(AdapterError::ExtraEntry(name.to_string()));
            }
        }
        for (name, shape) in want {
            let info = self
                .st
                .info(name)
                .map_err(|_| AdapterError::MissingEntry(name.clone()))?;
            if info.dtype != StDtype::F32 {
                return Err(AdapterError::LayoutMismatch {
                    name: name.clone(),
                    detail: format!("dtype {:?}, expected F32", info.dtype),
                });
            }
            if info.shape != shape_u64(shape) {
                return Err(AdapterError::LayoutMismatch {
                    name: name.clone(),
                    detail: format!("shape {:?}, expected {shape:?}", info.shape),
                });
            }
        }
        let mut by_name = std::collections::BTreeMap::new();
        for name in self.st.names() {
            let (_, values) = self
                .st
                .read_f32(name)
                .map_err(|e| self.corrupt(format!("{name}: {}", e.detail())))?;
            if let Some(i) = values.iter().position(|x| !x.is_finite()) {
                return Err(self.corrupt(format!("{name}[{i}] is {}", values[i])));
            }
            by_name.insert(name.to_string(), values);
        }
        let hash = integrity_hash(
            by_name.values().map(|v| v.as_slice()),
            self.st
                .metadata()
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_str())),
        );
        let recorded = self.meta(KEY_PAYLOAD)?;
        let actual = format!("{hash:016x}");
        if recorded != actual {
            return Err(self.corrupt(format!(
                "payload hash {actual} differs from the recorded {recorded}"
            )));
        }
        want.iter()
            .map(|(n, _)| {
                by_name
                    .remove(n)
                    .ok_or_else(|| AdapterError::MissingEntry(n.clone()))
            })
            .collect()
    }
}

/// Read the checkpoint in `dir` for a resume under `expected`. Every check
/// in the module docs runs before a value is returned; the result holds the
/// entries in `expected.entries` order.
pub fn read_adapter_checkpoint(
    dir: &Path,
    expected: &Expected,
) -> Result<AdapterCheckpoint, AdapterError> {
    check_base(&expected.base)?;
    recover_replaced_dir(dir).map_err(|e| AdapterError::Unreadable {
        file: dir.display().to_string(),
        detail: e.detail().to_string(),
    })?;
    let adapters = Opened::open(dir, ADAPTERS_FILE)?;
    let step = adapters.version_and_step()?;
    // The version of both files is judged before anything else is
    // interpreted (the document's order).
    let moments = Opened::open(dir, MOMENTS_FILE)?;
    let moment_step = moments.version_and_step()?;

    let sha256 = adapters.meta(KEY_BASE)?.to_string();
    let base_file = adapters.meta(KEY_BASE_FILE)?.to_string();
    let quantizer = match (
        adapters.optional(KEY_QUANT_FORMAT),
        adapters.optional(KEY_QUANT_GROUP),
        adapters.optional(KEY_QUANT_MLX),
    ) {
        (None, None, None) => None,
        (Some(format), Some(group_size), Some(mlx_version)) => Some(QuantizerRecord {
            format: format.to_string(),
            group_size: group_size.to_string(),
            mlx_version: mlx_version.to_string(),
        }),
        _ => return Err(adapters.corrupt("the quantizer record is partly present")),
    };
    if sha256 != expected.base.sha256 {
        return Err(AdapterError::BaseMismatch {
            expected: expected.base.sha256.clone(),
            found: sha256,
        });
    }
    if base_file != expected.base.file_name {
        return Err(AdapterError::BaseFileMismatch {
            expected: expected.base.file_name.clone(),
            found: base_file,
        });
    }
    if quantizer != expected.base.quantizer {
        return Err(AdapterError::QuantizerMismatch {
            expected: expected.base.quantizer.clone(),
            found: quantizer,
        });
    }
    let rank = usize::try_from(adapters.canonical_u64(KEY_RANK)?)
        .map_err(|e| adapters.corrupt(format!("{KEY_RANK}: {e}")))?;
    let alpha = adapters
        .meta(KEY_ALPHA)?
        .parse::<f32>()
        .map_err(|e| adapters.corrupt(format!("{KEY_ALPHA}: {e}")))?;
    if rank != expected.lora.rank() || alpha.to_bits() != expected.lora.alpha().to_bits() {
        return Err(AdapterError::LoraMismatch {
            detail: format!(
                "rank {rank} alpha {alpha}, the run's are rank {} alpha {}",
                expected.lora.rank(),
                expected.lora.alpha()
            ),
        });
    }
    let lora = LoraConfig::new(rank, alpha, 0.0)
        .map_err(|e| adapters.corrupt(format!("lora config: {e}")))?;
    let rng_hex = adapters.meta(KEY_RNG)?;
    let rng_state = unhex(rng_hex)
        .filter(|b| !b.is_empty() && b.len() <= MAX_RNG_BYTES)
        .ok_or_else(|| adapters.corrupt(format!("{KEY_RNG} is not lowercase hex")))?;
    let consumed_digest = adapters.meta(KEY_DIGEST)?.to_string();
    if consumed_digest.is_empty()
        || consumed_digest.len() > MAX_FIELD_BYTES
        || consumed_digest.len() % 2 != 0
        || !is_lower_hex(&consumed_digest)
    {
        return Err(adapters.corrupt(format!("{KEY_DIGEST} is not lowercase hex")));
    }

    if moment_step != step {
        return Err(AdapterError::StepMismatch {
            adapters: step,
            moments: moment_step,
        });
    }

    if expected.entries.is_empty() {
        return Err(invalid_field("entries", "a resume that expects no entries"));
    }
    let mut seen = std::collections::BTreeSet::new();
    for e in &expected.entries {
        if !seen.insert(e.name.as_str()) {
            return Err(AdapterError::InvalidEntry {
                name: e.name.clone(),
                detail: "duplicate expected name".to_string(),
            });
        }
        if !is_known_entry_name(&e.name) {
            return Err(AdapterError::UnknownEntryName(e.name.clone()));
        }
        numel(&e.name, &e.shape)?;
    }
    let want: Vec<(String, Vec<usize>)> = expected
        .entries
        .iter()
        .map(|e| (e.name.clone(), e.shape.clone()))
        .collect();
    let want_moments: Vec<(String, Vec<usize>)> = want
        .iter()
        .flat_map(|(n, s)| {
            [
                (format!("{M_PREFIX}{n}"), s.clone()),
                (format!("{V_PREFIX}{n}"), s.clone()),
            ]
        })
        .collect();
    let values = adapters.read_all(&want)?;
    let mut moment_values = moments.read_all(&want_moments)?.into_iter();
    // Both files verified against their own hash; now that they belong to
    // each other: the moments name the adapters file they were written with.
    let (adapters_hash, paired_with) = (adapters.meta(KEY_PAYLOAD)?, moments.meta(KEY_PAIR)?);
    if adapters_hash != paired_with {
        return Err(AdapterError::PairMismatch {
            adapters: adapters_hash.to_string(),
            moments_pair: paired_with.to_string(),
        });
    }
    let mut entries = Vec::with_capacity(want.len());
    for ((name, shape), values) in want.into_iter().zip(values) {
        let (Some(moment1), Some(moment2)) = (moment_values.next(), moment_values.next()) else {
            return Err(AdapterError::MissingEntry(format!("{M_PREFIX}{name}")));
        };
        if let Some(i) = moment2.iter().position(|x| *x < 0.0) {
            return Err(moments.corrupt(format!("{V_PREFIX}{name}[{i}] is negative")));
        }
        entries.push(AdapterEntry {
            name,
            shape,
            values,
            moment1,
            moment2,
        });
    }
    Ok(AdapterCheckpoint {
        lora,
        step,
        data: DataState {
            rng_state,
            consumed_digest,
        },
        base: BaseIdentity {
            file_name: expected.base.file_name.clone(),
            sha256: expected.base.sha256.clone(),
            quantizer: expected.base.quantizer.clone(),
        },
        entries,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_round_trips_and_refuses_the_rest() {
        let bytes = [0u8, 1, 0x7f, 0x80, 0xff];
        assert_eq!(hex(&bytes), "00017f80ff");
        assert_eq!(unhex("00017f80ff").as_deref(), Some(&bytes[..]));
        assert_eq!(unhex("0"), None);
        assert_eq!(unhex("0G"), None);
        assert_eq!(unhex("0A"), None);
        assert_eq!(unhex(""), Some(Vec::new()));
    }

    #[test]
    fn names_follow_peft() {
        assert_eq!(
            lora_a_name("model.layers.0.self_attn.q_proj"),
            "model.layers.0.self_attn.q_proj.lora_A.weight"
        );
        assert_eq!(lora_b_name("m"), "m.lora_B.weight");
    }
}

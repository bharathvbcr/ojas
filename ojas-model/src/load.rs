//! Parameters from a safetensors file with nanolab `state_dict` names.
//!
//! Rules (`docs/framework-design.md` §2):
//! - A leading `_orig_mod.` (from `torch.compile`) is stripped.
//! - Every name of [`param_table`] must be present exactly once, `F32`,
//!   with exactly the table's shape.
//! - `lm_head.weight` is accepted only when it is bit-equal to
//!   `tok_emb.weight` (the tie); any other extra tensor is refused.
//!
//!
//! Every name, dtype and shape, and the total bytes against the budget, are
//! checked from the header before any tensor data is read. Each tensor is
//! then decoded into charged storage in bounded chunks. The whole load is
//! checked before a tensor is returned, so a refusal hands back nothing.
//!
//! Values are not scanned: a non-finite weight is refused by the first op
//! that reads it (every backend checks its inputs), and the engine's tests
//! load such files on purpose to drive the deferred-fault paths.

use std::collections::BTreeMap;

use ojas_core::{Budget, DType, OjasError, Tensor};
use ojas_io::{SafeTensors, StDtype};

use crate::names::param_table;
use crate::spec::{ModelSpec, SPEC_METADATA_KEY};

/// The `torch.compile` wrapper prefix nanolab checkpoints may carry.
pub const COMPILED_PREFIX: &str = "_orig_mod.";

/// The untied head name. Accepted only as a bit-equal copy of the embedding.
pub const LM_HEAD: &str = "lm_head.weight";

/// f32 values of `lm_head.weight` compared per read (256 KiB).
const LM_HEAD_CHUNK_VALUES: usize = 64 << 10;

fn refuse(detail: String) -> OjasError {
    OjasError::Shape {
        op: "load_params",
        detail,
    }
}

/// The spec in `file`'s `__metadata__["ojas.spec"]` ([`ModelSpec::from_json`]).
/// A file without one is refused: the shape is never guessed from tensors.
pub fn load_spec(file: &SafeTensors<'_>) -> Result<ModelSpec, OjasError> {
    let text = file
        .metadata()
        .get(SPEC_METADATA_KEY)
        .ok_or_else(|| refuse(format!("no {SPEC_METADATA_KEY:?} in the file's metadata")))?;
    ModelSpec::from_json(text)
}

/// [`load_spec`], then [`load_params`] under that spec.
pub fn load_model(
    file: &SafeTensors<'_>,
    budget: &Budget,
) -> Result<(ModelSpec, Vec<Tensor>), OjasError> {
    let spec = load_spec(file)?;
    let params = load_params(&spec, file, budget)?;
    Ok((spec, params))
}

/// Every parameter of `spec` from `file`, in [`param_table`] order, as host
/// `F32` tensors charged to `budget`.
pub fn load_params(
    spec: &ModelSpec,
    file: &SafeTensors<'_>,
    budget: &Budget,
) -> Result<Vec<Tensor>, OjasError> {
    let table = param_table(spec)?;
    // Stored name for each stripped name; a collision is refused.
    let mut stored: BTreeMap<&str, &str> = BTreeMap::new();
    for raw in file.names() {
        let name = raw.strip_prefix(COMPILED_PREFIX).unwrap_or(raw);
        if let Some(prev) = stored.insert(name, raw) {
            return Err(refuse(format!(
                "{prev:?} and {raw:?} name the same parameter"
            )));
        }
    }
    // The header alone: every name, dtype and shape, and the total bytes
    // against the budget, before one byte of tensor data is read.
    let header = |name: &str, shape: &[usize]| -> Result<&str, OjasError> {
        let raw = *stored
            .get(name)
            .ok_or_else(|| refuse(format!("missing tensor {name:?}")))?;
        let info = file
            .info(raw)
            .map_err(|e| refuse(format!("{name}: {}", e.detail())))?;
        if info.dtype != StDtype::F32 {
            return Err(refuse(format!("{name}: {:?}, expected F32", info.dtype)));
        }
        if !info
            .shape
            .iter()
            .map(|&d| d as usize)
            .eq(shape.iter().copied())
        {
            return Err(refuse(format!(
                "{name}: shape {:?} != {shape:?}",
                info.shape
            )));
        }
        Ok(raw)
    };
    let mut raws = Vec::with_capacity(table.len());
    let mut total = 0u64;
    for info in &table {
        raws.push(header(&info.name, &info.shape)?);
        let bytes = (ojas_core::shape_product(&info.shape)? as u64)
            .checked_mul(4)
            .ok_or_else(|| refuse(format!("{}: byte size overflows", info.name)))?;
        total = total
            .checked_add(bytes)
            .ok_or_else(|| refuse("total parameter bytes overflow".to_string()))?;
    }
    let mut head = None;
    for name in stored.keys() {
        if *name == LM_HEAD {
            // A head of another dtype or shape cannot be the tie.
            head = Some(header(LM_HEAD, &table[0].shape).map_err(|_| untied())?);
        } else if !table.iter().any(|info| info.name == *name) {
            return Err(refuse(format!("unexpected tensor {name:?}")));
        }
    }
    budget.check_room(total)?;
    // Each tensor is decoded straight into charged storage in bounded
    // chunks, then checked finite.
    let mut values = Vec::with_capacity(table.len());
    for (info, raw) in table.iter().zip(raws) {
        let tensor = Tensor::from_le_reader(&info.shape, DType::F32, budget, |offset, chunk| {
            file.read_into(raw, offset, chunk)
                .map_err(|e| refuse(format!("{}: {}", info.name, e.detail())))
        })?;
        values.push(tensor);
    }
    if let Some(raw) = head {
        same_bits_as(file, raw, &values[0])?;
    }
    Ok(values)
}

fn untied() -> OjasError {
    refuse(format!(
        "{LM_HEAD} is not bit-equal to tok_emb.weight; an untied head is not supported"
    ))
}

/// `raw` in `file` holds exactly `emb`'s bits, compared in bounded chunks.
fn same_bits_as(file: &SafeTensors<'_>, raw: &str, emb: &Tensor) -> Result<(), OjasError> {
    let emb = emb.f32_slice()?;
    let mut buf = vec![0u8; LM_HEAD_CHUNK_VALUES * 4];
    for (index, piece) in emb.chunks(LM_HEAD_CHUNK_VALUES).enumerate() {
        let bytes = &mut buf[..piece.len() * 4];
        let offset = (index * LM_HEAD_CHUNK_VALUES * 4) as u64;
        file.read_into(raw, offset, bytes)
            .map_err(|e| refuse(format!("{LM_HEAD}: {}", e.detail())))?;
        let equal = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .zip(piece)
            .all(|(b, v)| u32::from_le_bytes(*b) == v.to_bits());
        if !equal {
            return Err(untied());
        }
    }
    Ok(())
}

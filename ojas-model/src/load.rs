//! Parameters from a safetensors file with nanolab `state_dict` names.
//!
//! Rules (`docs/framework-design.md` §2):
//! - A leading `_orig_mod.` (from `torch.compile`) is stripped.
//! - Every name of [`param_table`] must be present exactly once, `F32`,
//!   with exactly the table's shape.
//! - `lm_head.weight` is accepted only when it is bit-equal to
//!   `tok_emb.weight` (the tie); any other extra tensor is refused.
//!
//! The whole load is checked before a tensor is returned, so a refusal
//! hands back nothing.

use std::collections::BTreeMap;

use ojas_core::{Budget, OjasError, Tensor};
use ojas_io::SafeTensors;

use crate::names::param_table;
use crate::spec::{ModelSpec, SPEC_METADATA_KEY};

/// The `torch.compile` wrapper prefix nanolab checkpoints may carry.
pub const COMPILED_PREFIX: &str = "_orig_mod.";

/// The untied head name. Accepted only as a bit-equal copy of the embedding.
pub const LM_HEAD: &str = "lm_head.weight";

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
    let read = |name: &str| -> Result<(Vec<u64>, Vec<f32>), OjasError> {
        let raw = stored
            .get(name)
            .ok_or_else(|| refuse(format!("missing tensor {name:?}")))?;
        file.read_f32(raw)
            .map_err(|e| refuse(format!("{name}: {}", e.detail())))
    };
    let mut values = Vec::with_capacity(table.len());
    for info in &table {
        let (shape, data) = read(&info.name)?;
        let want: Vec<u64> = info.shape.iter().map(|&d| d as u64).collect();
        if shape != want {
            return Err(refuse(format!(
                "{}: shape {shape:?} != {:?}",
                info.name, info.shape
            )));
        }
        values.push(data);
    }
    for name in stored.keys() {
        if *name == LM_HEAD {
            let (shape, head) = read(LM_HEAD)?;
            let emb = &values[0];
            let same_shape = shape
                .iter()
                .map(|&d| d as usize)
                .eq(table[0].shape.iter().copied());
            let same_bits = same_shape
                && head.len() == emb.len()
                && head
                    .iter()
                    .zip(emb)
                    .all(|(a, b)| a.to_bits() == b.to_bits());
            if !same_bits {
                return Err(refuse(format!(
                    "{LM_HEAD} is not bit-equal to tok_emb.weight; an untied head is not supported"
                )));
            }
        } else if !table.iter().any(|info| info.name == *name) {
            return Err(refuse(format!("unexpected tensor {name:?}")));
        }
    }
    table
        .iter()
        .zip(values)
        .map(|(info, data)| Tensor::from_f32(&data, &info.shape, budget))
        .collect()
}

//! Reading an MLX affine-quantised linear (`mlx_lm`'s q8 and q4 layout) from safetensors.
//!
//! A quantised linear named `<p>` is three tensors, `<p>.weight` (`U32 [out, in * bits / 32]`),
//! `<p>.scales` and `<p>.biases` (`BF16 [out, in / group_size]`), and the file's `config.json`
//! carries `"quantization": {"group_size": 64, "bits": 8}`. [`affine_spec_from_config`] reads that
//! block strictly, and [`read_affine_quantized`] reads one linear into an
//! [`ojas_core::AffineQuantized`].
//!
//! **The caller supplies the expected `[out, in]`.** The three shapes alone do not say which
//! width and group size wrote them: a 4-bit, group-128 file has the same words-to-groups ratio (16)
//! as an 8-bit, group-64 one, so a triple read without its true `in` would be taken as q8 with half
//! the columns. With `[out, in]` and a spec, every shape is exact and a mismatch is a typed
//! [`AffineLoadError::Layout`] naming the width and group size the stored shapes imply.
//!
//! The size arithmetic is [`ojas_core::AffineSpec::geometry`], the one owner; this module does none
//! of its own.

use std::fmt;

use ojas_core::{
    AffineGeometry, AffineQuantized, AffineSpec, Budget, DType, OjasError, Scratch, Tensor,
};

use crate::{IoError, JsonValue, SafeTensors, StDtype, TensorInfo};

const OP: &str = "read_affine_quantized";

/// Why an MLX affine linear could not be read.
///
/// Not exhaustive: a new cause is a new variant, not a breaking change.
#[derive(Debug)]
#[non_exhaustive]
pub enum AffineLoadError {
    /// A tensor of the linear (`<prefix>.weight`, `.scales` or `.biases`) is not in the file.
    Missing { name: String },
    /// A tensor has the wrong dtype.
    Dtype {
        name: String,
        expected: StDtype,
        got: StDtype,
    },
    /// A tensor's shape is not the one the caller's `[out, in]` and spec require. `detail` names
    /// the width and group size the stored shapes imply, where they imply one.
    Layout {
        name: String,
        expected: Vec<usize>,
        got: Vec<u64>,
        detail: String,
    },
    /// The `quantization` block of the config is absent where required, malformed, unknown in a
    /// key, or names a mode, width or group size this reader does not support.
    Config { detail: String },
    /// Reading the file failed, or it is not what its header says. Keeps the source and the
    /// refusal flag of the [`IoError`].
    Io(IoError),
    /// `CapacityExceeded`, `NonFinite`, `OutOfRange`, `Shape` (zero or ragged geometry) or any
    /// other error of the checked arithmetic, the budget and the tensor constructors.
    Core(OjasError),
}

impl fmt::Display for AffineLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing { name } => write!(f, "affine linear: tensor {name:?} is missing"),
            Self::Dtype {
                name,
                expected,
                got,
            } => write!(
                f,
                "affine linear: {name:?} is {got:?}, expected {expected:?}"
            ),
            Self::Layout {
                name,
                expected,
                got,
                detail,
            } => write!(
                f,
                "affine linear: {name:?} has shape {got:?}, expected {expected:?}: {detail}"
            ),
            Self::Config { detail } => write!(f, "affine linear: quantization config: {detail}"),
            Self::Io(e) => write!(f, "affine linear: {e}"),
            Self::Core(e) => write!(f, "affine linear: {e}"),
        }
    }
}

impl std::error::Error for AffineLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Core(e) => Some(e),
            _ => None,
        }
    }
}

impl From<OjasError> for AffineLoadError {
    fn from(e: OjasError) -> Self {
        Self::Core(e)
    }
}

impl From<IoError> for AffineLoadError {
    fn from(e: IoError) -> Self {
        Self::Io(e)
    }
}

fn config_err(detail: impl Into<String>) -> AffineLoadError {
    AffineLoadError::Config {
        detail: detail.into(),
    }
}

/// The affine spec a model config declares, or `None` when it declares no quantisation.
///
/// Reads `quantization`; `quantization_config`, if present, must equal it (and may stand alone).
/// The block may hold only `group_size`, `bits` and, optionally, `mode` equal to `"affine"`: any
/// other key, including the object-valued per-layer overrides `mlx_lm` can write, is refused, so a
/// checkpoint whose layers are not all one width is not read as if they were. An unsupported width
/// or group size is the [`OjasError::OutOfRange`] of [`AffineSpec::new`], as
/// [`AffineLoadError::Core`].
pub fn affine_spec_from_config(config: &JsonValue) -> Result<Option<AffineSpec>, AffineLoadError> {
    let field = |key: &str| config.field_opt(key).map_err(|e| config_err(e.to_string()));
    let (quantization, quantization_config) =
        (field("quantization")?, field("quantization_config")?);
    let spec = match (quantization, quantization_config) {
        (None, None) => return Ok(None),
        (Some(block), None) | (None, Some(block)) => block_spec(block)?,
        (Some(q), Some(c)) => {
            // Compared by meaning, not by text: key order is not part of a JSON object.
            let (q, c) = (block_spec(q)?, block_spec(c)?);
            if q != c {
                return Err(config_err(format!(
                    "quantization ({q:?}) and quantization_config ({c:?}) are both present and differ"
                )));
            }
            q
        }
    };
    Ok(Some(spec))
}

/// One `quantization` block: `group_size`, `bits` and an optional `mode` of `"affine"`, nothing
/// else.
fn block_spec(block: &JsonValue) -> Result<AffineSpec, AffineLoadError> {
    block
        .deny_unknown_keys_at("quantization", &["group_size", "bits", "mode"])
        .map_err(|e| config_err(e.to_string()))?;
    if let Some(mode) = block.get("mode") {
        match mode.as_str() {
            Ok("affine") => {}
            Ok(other) => {
                return Err(config_err(format!(
                    "mode {other:?} is not supported (only \"affine\")"
                )))
            }
            Err(e) => return Err(config_err(format!("mode: {e}"))),
        }
    }
    let bits = block
        .get_u64("bits")
        .map_err(|e| config_err(e.to_string()))?;
    let group = block
        .get_u64("group_size")
        .map_err(|e| config_err(e.to_string()))?;
    let bits =
        u32::try_from(bits).map_err(|_| config_err(format!("bits {bits} is out of range")))?;
    let group = usize::try_from(group)
        .map_err(|_| config_err(format!("group_size {group} is out of range")))?;
    Ok(AffineSpec::new(bits, group)?)
}

/// One of the three tensors of a linear, checked against the header before any byte is read.
fn check_part(
    file: &SafeTensors<'_>,
    name: &str,
    dtype: StDtype,
    expected: [usize; 2],
    note: &dyn Fn(&TensorInfo) -> String,
) -> Result<(), AffineLoadError> {
    if !file.names().any(|n| n == name) {
        return Err(AffineLoadError::Missing {
            name: name.to_string(),
        });
    }
    let info = file.info(name)?;
    if info.dtype != dtype {
        return Err(AffineLoadError::Dtype {
            name: name.to_string(),
            expected: dtype,
            got: info.dtype,
        });
    }
    let matches =
        info.shape.len() == 2 && info.shape.iter().zip(expected).all(|(&a, b)| a == b as u64);
    if !matches {
        return Err(AffineLoadError::Layout {
            name: name.to_string(),
            expected: expected.to_vec(),
            got: info.shape.clone(),
            detail: note(info),
        });
    }
    Ok(())
}

/// Read the linear `prefix` (`<prefix>.weight`, `.scales`, `.biases`) of `file` as `spec`, for a
/// weight of `rows = out` by `cols = in`.
///
/// In order: the checked geometry ([`AffineSpec::geometry`]); each tensor's presence, dtype and
/// shape from the header; the room for the three parts ([`Budget::check_room`], so a short budget
/// refuses before anything is charged); the storage, charged as it is allocated; the reads; then
/// [`AffineQuantized::from_parts`], which refuses a non-finite scale or bias. On any error the
/// charge is released. The parts total exactly [`AffineQuantized::nbytes`] bytes.
pub fn read_affine_quantized(
    file: &SafeTensors<'_>,
    prefix: &str,
    spec: AffineSpec,
    rows: usize,
    cols: usize,
    budget: &Budget,
) -> Result<AffineQuantized, AffineLoadError> {
    let AffineGeometry { groups, words } = spec.geometry(OP, rows, cols)?;
    let (packed_name, scales_name, biases_name) = (
        format!("{prefix}.weight"),
        format!("{prefix}.scales"),
        format!("{prefix}.biases"),
    );
    let bits = spec.bits().bits();
    let group = spec.group_size();
    let cols_u64 = cols as u64;
    let packed_note = |info: &TensorInfo| {
        let stored = match info.shape.as_slice() {
            [_, w] => w
                .checked_mul(32)
                .filter(|stored_bits| stored_bits.is_multiple_of(cols_u64))
                .map(|stored_bits| format!("{}-bit codes", stored_bits / cols_u64)),
            _ => None,
        }
        .unwrap_or_else(|| "no whole code width".to_string());
        format!(
            "the caller asked for {bits}-bit codes over [{rows}, {cols}]; the stored words per \
             row imply {stored}"
        )
    };
    let params_note = |info: &TensorInfo| {
        let stored = match info.shape.as_slice() {
            [_, g] if *g > 0 && cols_u64.is_multiple_of(*g) => {
                Some(format!("group size {}", cols_u64 / g))
            }
            _ => None,
        }
        .unwrap_or_else(|| "no whole group size".to_string());
        format!(
            "the caller asked for group size {group} over [{rows}, {cols}]; the stored groups \
             per row imply {stored}"
        )
    };
    check_part(
        file,
        &packed_name,
        StDtype::U32,
        [rows, words],
        &packed_note,
    )?;
    check_part(
        file,
        &scales_name,
        StDtype::BF16,
        [rows, groups],
        &params_note,
    )?;
    check_part(
        file,
        &biases_name,
        StDtype::BF16,
        [rows, groups],
        &params_note,
    )?;

    // `geometry` bounded each part's element and byte counts; the sum is checked here.
    let part_bytes = |elems: Option<usize>, size: usize| elems?.checked_mul(size);
    let total = part_bytes(rows.checked_mul(words), DType::U32.size())
        .zip(part_bytes(rows.checked_mul(groups), 2 * DType::Bf16.size()))
        .and_then(|(p, sb)| p.checked_add(sb))
        .ok_or_else(|| OjasError::OutOfRange {
            op: OP,
            detail: "byte total overflows".to_string(),
        })? as u64;
    budget.check_room(total)?;

    let mut packed = Scratch::<u32>::try_alloc(rows * words, budget)?;
    let mut scales = Tensor::zeros(&[rows, groups], DType::Bf16, budget)?;
    let mut biases = Tensor::zeros(&[rows, groups], DType::Bf16, budget)?;
    file.read_u32_into(&packed_name, &[rows, words], packed.as_mut_slice())?;
    file.read_bf16_bits_into(&scales_name, &[rows, groups], scales.bf16_slice_mut()?)?;
    file.read_bf16_bits_into(&biases_name, &[rows, groups], biases.bf16_slice_mut()?)?;
    let packed = Tensor::from_scratch(packed, &[rows, words])?;
    Ok(AffineQuantized::from_parts(
        spec, rows, cols, packed, scales, biases,
    )?)
}

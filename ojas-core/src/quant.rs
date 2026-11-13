//! MLX-exact affine group quantisation on the host.
//!
//! The one owner of the packed-affine format a resident quantised base is stored in, and the
//! oracle every device kernel is checked against. [`quantize_affine`] reproduces
//! `mx.quantize(w, group_size=64, bits=B, stream=mx.gpu)` bit for bit on a bf16 weight, and
//! [`dequantize_affine_bf16`] reproduces `mx.dequantize(..., stream=mx.gpu)`: both are pinned by
//! golden arrays from mlx 0.31.2 (`ojas-core/tests/fixtures/mlx_affine/`, made by
//! `ojas-oracle/python/mlx_affine_goldens.py`). The target is mlx's **GPU** path: `mx.cpu`
//! disagrees with it on roughly 0.2 to 0.4 percent of the codes, so "MLX-exact" names a device.
//!
//! # The format
//!
//! A weight `[rows, cols]` is cut into groups of 64 consecutive values along a row (`cols` is a
//! multiple of 64). Each group stores one bf16 `scale` and one bf16 `bias`, and one code per value
//! in `0..=2^bits - 1`. [`AffineQuantized`] holds three tensors, the layout mlx and `mlx_lm` use
//! for a quantised linear (`<prefix>.weight`, `.scales`, `.biases`):
//!
//! - `packed`: `U32` `[rows, cols * bits / 32]`. Code `i` of a word sits at bits
//!   `[bits * i, bits * (i + 1))`, little-endian lanes: 8 codes per word for q4, 4 for q8.
//! - `scales`, `biases`: `Bf16` `[rows, cols / 64]`.
//!
//! There is no `DType` variant: the packed codes are an ordinary `U32` tensor.
//!
//! # The quantiser (all arithmetic in IEEE f32)
//!
//! Per group `w` (bf16 widened to f32), `n_bins = 2^bits - 1`:
//!
//! ```text
//! w_min  = min(bf16::MAX, min(w))      w_max = max(0, max(w))       (w_max starts at 0, not -inf)
//! scale  = max((w_max - w_min) / n_bins, 1e-7)
//! side   = |w_min| > |w_max|           scale = side ? scale : -scale
//! edge   = side ? w_min : w_max
//! q0     = round(edge / scale)         (round half away from zero)
//! scale  = q0 == 0 ? scale : edge / q0
//! bias   = q0 == 0 ? 0 : edge
//! code   = min(round((w - bias) / scale), n_bins)     (the unrounded f32 scale and bias)
//! stored scale, bias = bf16 round-to-nearest-even of the f32 values
//! ```
//!
//! # Decisions the goldens made
//!
//! - **Subnormals are flushed.** Apple GPU f32 arithmetic treats a subnormal operand as zero
//!   (`ojas` card ft-8f8e3fac). The goldens show it in the `side` test: a group of zeros with one
//!   `-3e-40` has `|w_min| == |w_max| == 0`, so the scale comes out negative. The quantiser
//!   flushes every input value (`flush`). Result flushes cannot change a bit here: `scale` is
//!   floored at `1e-7`, so every later operand is either a flushed input or at least `5e-8` in
//!   magnitude, and a subnormal `(w - bias) / scale` rounds to code 0 either way. The dequantiser
//!   flushes its operands (a subnormal scale or bias) and the fused `scale * code + bias` result,
//!   which the `deq_params_w64` golden shows matters (the quantiser never writes such a scale or bias,
//!   a loaded checkpoint could carry one).
//! - **`min` ignores a NaN.** A constant group of `bf16::MAX` has `scale = -1e-7`, `q0 = -inf`,
//!   `scale = edge / -inf = -0.0`, and `(w - bias) / scale = 0 / -0 = NaN`. mlx's `min(NaN, n_bins)`
//!   returns `n_bins`; so does `f32::min`. The stored scale is `-0.0`, the bias `bf16::MAX`, and
//!   the dequantised value `-0.0 * 255 + bias`, the weight again.
//! - **A negative code cannot happen except as `-0.0`.** With the edge the larger-magnitude end,
//!   `(w - bias) / scale` is non-negative; when `q0 == 0` the bias is 0 and every `|w|` is under
//!   half the scale, so each code is `+-0.0`, which maps to 0. A strictly negative rounded code
//!   would be a logic error (Metal's float to uint8 conversion of a negative is undefined), so it
//!   is refused with [`OjasError::OutOfRange`], never clamped.
//! - **A group whose range overflows f32** (a `+bf16::MAX` beside a `-bf16::MAX`) has no finite
//!   scale; mlx returns `-inf` there. It is refused with [`OjasError::OutOfRange`].
//! - **Rounding.** `f32::round` (exact, half away from zero) is used. The goldens cannot separate
//!   it from `floor(x + 0.5)`, which differs only at `x = 0.49999997` (and its negative): no group
//!   in 335 million random bf16 weights produced that quotient. Unverified on the GPU.

use crate::{
    bf16_to_f32, f32_to_bf16, Backend, BackendId, Budget, DType, OjasError, Scratch, Tensor,
};

/// The group size every [`AffineSpec`] carries: the size mlx's goldens cover.
pub const AFFINE_GROUP_SIZE: usize = 64;

/// `bf16::MAX` as f32 (`0x7F7F0000`): the initial `w_min` of the quantiser.
const BF16_MAX: f32 = 3.389_531_4e38;
/// The smallest scale the quantiser writes before the `edge / q0` refinement (mlx `eps`).
const SCALE_FLOOR: f32 = 1e-7;

/// Code width of an affine quantisation. Only 4 and 8 bits have goldens, so only they exist.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AffineBits {
    Q4,
    Q8,
}

impl AffineBits {
    /// `4` or `8`; any other width is [`OjasError::OutOfRange`].
    pub fn from_bits(bits: u32) -> Result<Self, OjasError> {
        match bits {
            4 => Ok(AffineBits::Q4),
            8 => Ok(AffineBits::Q8),
            _ => Err(OjasError::OutOfRange {
                op: "AffineBits::from_bits",
                detail: format!("{bits}-bit affine quantisation is not supported (4 or 8)"),
            }),
        }
    }

    /// Width of one code in bits.
    pub fn bits(self) -> u32 {
        match self {
            AffineBits::Q4 => 4,
            AffineBits::Q8 => 8,
        }
    }

    /// Largest code, `2^bits - 1`.
    pub fn max_code(self) -> u32 {
        (1u32 << self.bits()) - 1
    }

    /// Codes in one packed `u32` word: 8 for q4, 4 for q8.
    pub fn codes_per_word(self) -> usize {
        (32 / self.bits()) as usize
    }
}

/// Width and group size of an affine quantisation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AffineSpec {
    bits: AffineBits,
    group_size: usize,
}

impl AffineSpec {
    /// A spec for `bits` (4 or 8) and `group_size`. Only 64 is accepted: it is the only size the
    /// mlx goldens cover. Zero is refused like any other size.
    pub fn new(bits: u32, group_size: usize) -> Result<Self, OjasError> {
        let bits = AffineBits::from_bits(bits)?;
        if group_size != AFFINE_GROUP_SIZE {
            return Err(OjasError::OutOfRange {
                op: "AffineSpec::new",
                detail: format!(
                    "group size {group_size} is not supported (only {AFFINE_GROUP_SIZE})"
                ),
            });
        }
        Ok(Self { bits, group_size })
    }

    /// 8-bit codes, group size 64.
    pub fn q8() -> Self {
        Self {
            bits: AffineBits::Q8,
            group_size: AFFINE_GROUP_SIZE,
        }
    }

    /// 4-bit codes, group size 64.
    pub fn q4() -> Self {
        Self {
            bits: AffineBits::Q4,
            group_size: AFFINE_GROUP_SIZE,
        }
    }

    pub fn bits(&self) -> AffineBits {
        self.bits
    }

    pub fn group_size(&self) -> usize {
        self.group_size
    }

    /// `u32` words holding one group's codes: 8 for q4, 16 for q8 at group size 64.
    pub fn words_per_group(&self) -> usize {
        self.group_size / self.bits.codes_per_word()
    }

    /// Checked `[rows, cols]` geometry of a quantised matrix: groups per row and packed words per
    /// row. The one owner of the size arithmetic: [`quantize_affine`],
    /// [`AffineQuantized::from_parts`], [`dequant_affine_bf16_dims`](crate::dequant_affine_bf16_dims)
    /// and the loaders and row helpers in other crates all call it, so no copy can drift.
    ///
    /// `rows` and `cols` must be non-zero ([`OjasError::Shape`]) and `cols` a multiple of the group
    /// size ([`OjasError::Shape`]). Every count the three parts and the bf16 result need must fit
    /// `usize`, in elements and in bytes ([`OjasError::OutOfRange`], named by `op`): `rows * cols`
    /// and its bf16 byte count, `rows * words` and its byte count, `rows * groups` and its byte
    /// count. Nothing is allocated.
    pub fn geometry(
        &self,
        op: &'static str,
        rows: usize,
        cols: usize,
    ) -> Result<AffineGeometry, OjasError> {
        if rows == 0 || cols == 0 {
            return Err(OjasError::Shape {
                op,
                detail: format!("an empty matrix [{rows}, {cols}] cannot be quantised"),
            });
        }
        if !cols.is_multiple_of(self.group_size) {
            return Err(OjasError::Shape {
                op,
                detail: format!(
                    "cols {cols} is not a multiple of the group size {}",
                    self.group_size
                ),
            });
        }
        let groups = cols / self.group_size;
        let words = groups
            .checked_mul(self.words_per_group())
            .ok_or_else(|| overflow(op, "packed words per row"))?;
        let elems = rows
            .checked_mul(cols)
            .ok_or_else(|| overflow(op, "rows * cols"))?;
        elems
            .checked_mul(DType::Bf16.size())
            .ok_or_else(|| overflow(op, "rows * cols * 2 bytes"))?;
        rows.checked_mul(words)
            .and_then(|n| n.checked_mul(DType::U32.size()))
            .ok_or_else(|| overflow(op, "rows * packed words * 4 bytes"))?;
        rows.checked_mul(groups)
            .and_then(|n| n.checked_mul(DType::Bf16.size()))
            .ok_or_else(|| overflow(op, "rows * groups * 2 bytes"))?;
        Ok(AffineGeometry { groups, words })
    }
}

/// Per-row counts of a checked quantised matrix, from [`AffineSpec::geometry`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AffineGeometry {
    /// Groups per row, `cols / group_size`: the width of `scales` and `biases`.
    pub groups: usize,
    /// Packed `u32` words per row: the width of `packed`.
    pub words: usize,
}

fn overflow(op: &'static str, what: &str) -> OjasError {
    OjasError::OutOfRange {
        op,
        detail: format!("{what} overflows usize"),
    }
}

/// An affine-quantised `[rows, cols]` matrix: packed codes plus bf16 scales and biases.
///
/// Built by [`quantize_affine`] or, from stored parts, by [`AffineQuantized::from_parts`], which
/// checks every part so a value of this type always has the shapes, dtypes and finite scales and
/// biases [`dequantize_affine_bf16`] relies on.
#[derive(Clone, Debug)]
pub struct AffineQuantized {
    spec: AffineSpec,
    rows: usize,
    cols: usize,
    packed: Tensor,
    scales: Tensor,
    biases: Tensor,
}

impl AffineQuantized {
    /// Validate stored parts. `packed` is `U32 [rows, cols * bits / 32]`, `scales` and `biases`
    /// are `Bf16 [rows, cols / group_size]`, all host tensors with a readable window, and every
    /// scale and bias is finite (`+-0.0` is allowed: a constant `bf16::MAX` group stores `-0.0`).
    /// A refusal returns the parts' own errors: [`OjasError::Shape`] for rows or cols that are
    /// zero, not a multiple of the group size, or tensors whose shape differs,
    /// [`OjasError::Dtype`] for a wrong dtype, [`OjasError::NonFinite`] for a NaN or infinite
    /// scale or bias, and the tensor accessors' errors for a device or non-contiguous tensor.
    /// Parts are read through the host slice accessors, so device-resident tensors are refused
    /// today (a validated host-to-device path is a later wave).
    pub fn from_parts(
        spec: AffineSpec,
        rows: usize,
        cols: usize,
        packed: Tensor,
        scales: Tensor,
        biases: Tensor,
    ) -> Result<Self, OjasError> {
        const OP: &str = "AffineQuantized::from_parts";
        let AffineGeometry { groups, words } = spec.geometry(OP, rows, cols)?;
        check_part(OP, "packed", &packed, DType::U32, &[rows, words])?;
        check_part(OP, "scales", &scales, DType::Bf16, &[rows, groups])?;
        check_part(OP, "biases", &biases, DType::Bf16, &[rows, groups])?;
        packed.u32_slice()?;
        for part in [&scales, &biases] {
            if part.bf16_slice()?.iter().any(|&bits| !bf16_is_finite(bits)) {
                return Err(OjasError::NonFinite { op: OP });
            }
        }
        Ok(Self {
            spec,
            rows,
            cols,
            packed,
            scales,
            biases,
        })
    }

    pub fn spec(&self) -> AffineSpec {
        self.spec
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    /// `U32 [rows, cols * bits / 32]`.
    pub fn packed(&self) -> &Tensor {
        &self.packed
    }

    /// `Bf16 [rows, cols / group_size]`.
    pub fn scales(&self) -> &Tensor {
        &self.scales
    }

    /// `Bf16 [rows, cols / group_size]`.
    pub fn biases(&self) -> &Tensor {
        &self.biases
    }

    /// Bytes the three tensors hold.
    pub fn nbytes(&self) -> Result<usize, OjasError> {
        let mut total = 0usize;
        for t in [&self.packed, &self.scales, &self.biases] {
            total = t
                .num_elements()?
                .checked_mul(t.dtype().size())
                .and_then(|bytes| total.checked_add(bytes))
                .ok_or_else(|| overflow("AffineQuantized::nbytes", "byte count"))?;
        }
        Ok(total)
    }

    /// The three tensors, `(packed, scales, biases)`.
    pub fn into_parts(self) -> (Tensor, Tensor, Tensor) {
        (self.packed, self.scales, self.biases)
    }

    /// Make the weight resident where `backend` computes, once: the owner of the placement a
    /// resident quantised base needs ([`PlacedAffine`]).
    ///
    /// On the CPU backend the host tensors are kept as they are. On any other backend each
    /// part is uploaded and `self`'s host copies are dropped, so no host copy of a placed
    /// weight survives. The room for all three parts is checked on `backend`'s budget
    /// ([`Budget::check_room`]) before the first upload, so a refusal leaves nothing uploaded.
    /// A part the backend returns that is not resident on it is [`OjasError::Placement`].
    /// The parts were validated by [`AffineQuantized::from_parts`] on the host, so a placed
    /// weight has the shapes, dtypes and finite scales and biases the dequantiser relies on.
    pub fn place<B: Backend + ?Sized>(self, backend: &B) -> Result<PlacedAffine, OjasError> {
        const OP: &str = "AffineQuantized::place";
        let id = backend.id();
        let (packed, scales, biases) = if id == BackendId::Cpu {
            (self.packed, self.scales, self.biases)
        } else {
            let bytes = self.nbytes()?;
            backend.budget().check_room(bytes as u64)?;
            let up = |t: &Tensor| -> Result<Tensor, OjasError> {
                let placed = backend.upload(t)?;
                if placed.device() != Some(id) {
                    return Err(OjasError::Placement {
                        op: OP,
                        expected: Some(id),
                        found: placed.device(),
                    });
                }
                Ok(placed)
            };
            (up(&self.packed)?, up(&self.scales)?, up(&self.biases)?)
        };
        Ok(PlacedAffine {
            backend: id,
            spec: self.spec,
            rows: self.rows,
            cols: self.cols,
            packed,
            scales,
            biases,
        })
    }
}

/// An affine-quantised `[rows, cols]` weight resident where one backend computes: the packed
/// codes and the bf16 scales and biases, each a tensor on that backend (host tensors for the
/// CPU backend).
///
/// The only constructor is [`AffineQuantized::place`], so every value was validated on the host
/// first and uploaded exactly once. [`AffineQuantized::from_parts`] refuses device tensors, so a
/// placed weight needs a type of its own. Cloning is cheap: the tensors are shared. The weight is
/// never held dequantised; a `linear_q` over it dequantises to a bf16 `[rows, cols]` transient
/// each time it runs (`ojas_autograd::Tape::linear_q`).
#[derive(Clone, Debug)]
pub struct PlacedAffine {
    backend: BackendId,
    spec: AffineSpec,
    rows: usize,
    cols: usize,
    packed: Tensor,
    scales: Tensor,
    biases: Tensor,
}

impl PlacedAffine {
    /// The backend whose memory holds the parts.
    pub fn backend(&self) -> BackendId {
        self.backend
    }

    pub fn spec(&self) -> AffineSpec {
        self.spec
    }

    /// Output features of a linear over this weight.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Input features of a linear over this weight.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// `U32 [rows, cols * bits / 32]`.
    pub fn packed(&self) -> &Tensor {
        &self.packed
    }

    /// `Bf16 [rows, cols / group_size]`.
    pub fn scales(&self) -> &Tensor {
        &self.scales
    }

    /// `Bf16 [rows, cols / group_size]`.
    pub fn biases(&self) -> &Tensor {
        &self.biases
    }

    /// Bytes the three parts hold: what stays resident, never the dequantised weight.
    pub fn nbytes(&self) -> Result<usize, OjasError> {
        let mut total = 0usize;
        for t in [&self.packed, &self.scales, &self.biases] {
            total = t
                .num_elements()?
                .checked_mul(t.dtype().size())
                .and_then(|bytes| total.checked_add(bytes))
                .ok_or_else(|| overflow("PlacedAffine::nbytes", "byte count"))?;
        }
        Ok(total)
    }

    /// The bytes of the bf16 `[rows, cols]` transient a dequantisation of this weight makes.
    pub fn dequantised_bytes(&self) -> Result<usize, OjasError> {
        self.rows
            .checked_mul(self.cols)
            .and_then(|n| n.checked_mul(DType::Bf16.size()))
            .ok_or_else(|| overflow("PlacedAffine::dequantised_bytes", "rows * cols * 2 bytes"))
    }

    /// The shape check every `linear_q` runs before any dispatch: `x` (the input's shape) is
    /// rank 2 or more (a rank-1 input is refused, so the check never depends on what a backend's
    /// linear does with one) and its last extent is this weight's input width, `cols`. A refusal
    /// is [`OjasError::Shape`] naming `op`.
    pub fn check_input(&self, op: &'static str, x: &[usize]) -> Result<(), OjasError> {
        match x.last() {
            Some(&last) if x.len() >= 2 && last == self.cols => Ok(()),
            _ => Err(OjasError::Shape {
                op,
                detail: format!(
                    "x {x:?} is not rank 2 or more ending in the quantised weight's {} input features \
                     ([{}, {}])",
                    self.cols, self.rows, self.cols
                ),
            }),
        }
    }

    /// Refuse a weight that was placed on another backend than `computes_on`, with
    /// [`OjasError::Placement`] naming `op`. Nothing is copied implicitly: a host weight handed
    /// to a GPU tape would otherwise be uploaded again for every tape (1.46 GB per sequence at
    /// 2B), and a device weight handed to a CPU tape would be read back. A weight from another
    /// device of the same backend is refused by that backend's own operand check at the first
    /// dequantisation.
    pub fn require_on(&self, op: &'static str, computes_on: BackendId) -> Result<(), OjasError> {
        if self.backend == computes_on {
            return Ok(());
        }
        Err(OjasError::Placement {
            op,
            // Host memory is `None` on both sides, as the CPU backend reports it.
            expected: (computes_on != BackendId::Cpu).then_some(computes_on),
            found: (self.backend != BackendId::Cpu).then_some(self.backend),
        })
    }
}

fn check_part(
    op: &'static str,
    name: &str,
    t: &Tensor,
    dtype: DType,
    shape: &[usize],
) -> Result<(), OjasError> {
    if t.dtype() != dtype {
        return Err(OjasError::Dtype {
            op,
            expected: dtype,
            got: t.dtype(),
        });
    }
    if t.shape() != shape {
        return Err(OjasError::Shape {
            op,
            detail: format!("{name} has shape {:?}, expected {shape:?}", t.shape()),
        });
    }
    Ok(())
}

fn bf16_is_finite(bits: u16) -> bool {
    bits & 0x7F80 != 0x7F80
}

/// Apple-GPU float behaviour: a subnormal f32 is treated as a zero of the same sign.
fn flush(x: f32) -> f32 {
    if x.is_subnormal() {
        0.0f32.copysign(x)
    } else {
        x
    }
}

/// Why one group could not be quantised.
enum GroupRefusal {
    /// `w_max - w_min` is not finite.
    RangeOverflow,
    /// A rounded code below zero (cannot happen for finite input; refused, not clamped).
    NegativeCode,
}

/// The code of one value: `min(x, n_bins)` as an integer. `x` is the rounded
/// `(w - bias) / scale`. A NaN takes `n_bins` (`f32::min` ignores it, as Metal's `min` does), and
/// `-0.0` is code 0. A strictly negative value is [`GroupRefusal::NegativeCode`].
fn code_of(x: f32, n_bins: f32) -> Result<u32, GroupRefusal> {
    let c = x.min(n_bins);
    if c < 0.0 {
        return Err(GroupRefusal::NegativeCode);
    }
    Ok(c as u32)
}

/// Quantise one group of already-flushed values; returns the unrounded f32 `(scale, bias)` and
/// writes the codes.
fn quantize_group(
    w: &[f32; AFFINE_GROUP_SIZE],
    n_bins: f32,
    codes: &mut [u32; AFFINE_GROUP_SIZE],
) -> Result<(f32, f32), GroupRefusal> {
    let mut w_min = BF16_MAX;
    let mut w_max = 0.0f32;
    for &x in w {
        w_min = w_min.min(x);
        w_max = w_max.max(x);
    }
    let range = w_max - w_min;
    if !range.is_finite() {
        return Err(GroupRefusal::RangeOverflow);
    }
    let mut scale = (range / n_bins).max(SCALE_FLOOR);
    let side = w_min.abs() > w_max.abs();
    if !side {
        scale = -scale;
    }
    let edge = if side { w_min } else { w_max };
    let q0 = (edge / scale).round();
    let at_zero = q0 == 0.0;
    if !at_zero {
        scale = edge / q0;
    }
    let bias = if at_zero { 0.0 } else { edge };
    for (code, &x) in codes.iter_mut().zip(w) {
        *code = code_of(((x - bias) / scale).round(), n_bins)?;
    }
    Ok((scale, bias))
}

/// Quantise a bf16 weight `[rows, cols]` exactly as `mx.quantize(w, group_size=64, bits,
/// stream=mx.gpu)` does.
///
/// `weight` is `Bf16`, or `F32` with every value exactly representable in bf16 (the definition of
/// "MLX-exact" is for bf16 input; any other `F32` is [`OjasError::OutOfRange`]). Refusals, all
/// before anything is allocated except the range overflow, which is found while quantising and
/// releases what was charged: a dtype other than `Bf16`/`F32` is [`OjasError::Dtype`]; rank not 2,
/// zero rows or cols, or cols not a multiple of the group size is [`OjasError::Shape`]; a NaN or
/// infinity is [`OjasError::NonFinite`]; a group whose `w_max - w_min` overflows f32 is
/// [`OjasError::OutOfRange`]. The three outputs are charged to `budget` before they are
/// allocated, and a charge that does not fit is [`OjasError::CapacityExceeded`] with nothing left
/// charged.
pub fn quantize_affine(
    weight: &Tensor,
    spec: AffineSpec,
    budget: &Budget,
) -> Result<AffineQuantized, OjasError> {
    const OP: &str = "quantize_affine";
    let shape = weight.shape();
    let [rows, cols] = *shape else {
        return Err(OjasError::Shape {
            op: OP,
            detail: format!("weight must be 2-D, got shape {shape:?}"),
        });
    };
    let src = match weight.dtype() {
        DType::Bf16 => Source::Bf16(weight.bf16_slice()?),
        DType::F32 => Source::F32(weight.f32_slice()?),
        got => {
            return Err(OjasError::Dtype {
                op: OP,
                expected: DType::Bf16,
                got,
            })
        }
    };
    let AffineGeometry { groups, words } = spec.geometry(OP, rows, cols)?;
    src.validate()?;

    let packed_len = rows * words;
    let mut packed = Scratch::<u32>::try_alloc(packed_len, budget)?;
    let mut scales = Tensor::zeros(&[rows, groups], DType::Bf16, budget)?;
    let mut biases = Tensor::zeros(&[rows, groups], DType::Bf16, budget)?;
    {
        let scale_bits = scales.bf16_slice_mut()?;
        let bias_bits = biases.bf16_slice_mut()?;
        let n_bins = spec.bits.max_code() as f32;
        let per_word = spec.bits.codes_per_word();
        let lane_bits = spec.bits.bits();
        let wpg = spec.words_per_group();
        let mut w = [0.0f32; AFFINE_GROUP_SIZE];
        let mut codes = [0u32; AFFINE_GROUP_SIZE];
        for (g, out) in packed.as_mut_slice().chunks_exact_mut(wpg).enumerate() {
            src.load_group(g * AFFINE_GROUP_SIZE, &mut w);
            let (scale, bias) = match quantize_group(&w, n_bins, &mut codes) {
                Ok(sb) => sb,
                Err(GroupRefusal::RangeOverflow) => {
                    return Err(OjasError::OutOfRange {
                        op: OP,
                        detail: format!(
                        "row {}, group {}: the value range overflows f32, there is no finite scale",
                        g / groups,
                        g % groups
                    ),
                    })
                }
                Err(GroupRefusal::NegativeCode) => {
                    return Err(OjasError::OutOfRange {
                        op: OP,
                        detail: format!(
                            "row {}, group {}: a rounded code is negative",
                            g / groups,
                            g % groups
                        ),
                    })
                }
            };
            scale_bits[g] = f32_to_bf16(scale);
            bias_bits[g] = f32_to_bf16(bias);
            for (word, lanes) in out.iter_mut().zip(codes.chunks_exact(per_word)) {
                *word = lanes
                    .iter()
                    .enumerate()
                    .fold(0u32, |acc, (i, &c)| acc | (c << (lane_bits * i as u32)));
            }
        }
    }
    let packed = Tensor::from_scratch(packed, &[rows, words])?;
    AffineQuantized::from_parts(spec, rows, cols, packed, scales, biases)
}

/// The weight values of one `quantize_affine` call, read as f32.
enum Source<'a> {
    Bf16(&'a [u16]),
    F32(&'a [f32]),
}

impl Source<'_> {
    /// Refuse a non-finite value, then an `F32` value that is not bf16-exact.
    fn validate(&self) -> Result<(), OjasError> {
        const OP: &str = "quantize_affine";
        match self {
            Source::Bf16(bits) => {
                if bits.iter().any(|&b| !bf16_is_finite(b)) {
                    return Err(OjasError::NonFinite { op: OP });
                }
            }
            Source::F32(values) => {
                if values.iter().any(|v| !v.is_finite()) {
                    return Err(OjasError::NonFinite { op: OP });
                }
                if let Some(i) = values.iter().position(|v| v.to_bits() & 0xFFFF != 0) {
                    return Err(OjasError::OutOfRange {
                        op: OP,
                        detail: format!(
                            "F32 value {} at index {i} is not exactly representable in bf16; \
                             MLX-exact quantisation is defined for bf16 input",
                            values[i]
                        ),
                    });
                }
            }
        }
        Ok(())
    }

    /// Copy `out.len()` values starting at element `start` into `out`, flushing subnormals.
    fn load_group(&self, start: usize, out: &mut [f32; AFFINE_GROUP_SIZE]) {
        match self {
            Source::Bf16(bits) => {
                for (o, &b) in out.iter_mut().zip(&bits[start..start + AFFINE_GROUP_SIZE]) {
                    *o = flush(bf16_to_f32(b));
                }
            }
            Source::F32(values) => {
                for (o, &v) in out
                    .iter_mut()
                    .zip(&values[start..start + AFFINE_GROUP_SIZE])
                {
                    *o = flush(v);
                }
            }
        }
    }
}

/// Dequantise to bf16 `[rows, cols]` exactly as `mx.dequantize(..., stream=mx.gpu)` does:
/// `bf16(flush(fma(flush(scale), code, flush(bias))))`, one rounding.
///
/// The multiply-add is fused (`f32::mul_add`, a correctly rounded fma on every target). The
/// product `scale * code` is exact in f32, so fused and unfused agree except where the unfused
/// product overflows: scale `2^125`, code 8, bias `-bf16::MAX` is `inf` unfused and finite fused,
/// and mlx returns the finite value (checked against `mx.dequantize(stream=mx.gpu)`). The
/// two-rounding form `bf16(bf16(scale * code) + bias)` is not what mlx computes. The output is charged to `budget`
/// before it is allocated. A weight that would round to a bf16 infinity (stored parts whose
/// affine map overflows) is [`OjasError::NonFinite`]: no valid weight is infinite, and the
/// refusal is loud rather than a poisoned base.
pub fn dequantize_affine_bf16(q: &AffineQuantized, budget: &Budget) -> Result<Tensor, OjasError> {
    const OP: &str = "dequantize_affine_bf16";
    let packed = q.packed.u32_slice()?;
    let scales = q.scales.bf16_slice()?;
    let biases = q.biases.bf16_slice()?;
    let mut out = Tensor::zeros(&[q.rows, q.cols], DType::Bf16, budget)?;
    expand(OP, q.spec, packed, scales, biases, out.bf16_slice_mut()?)?;
    Ok(out)
}

/// [`dequantize_affine_bf16`] on three loose tensors: the host implementation of
/// [`BackendModule::dequant_affine_bf16`](crate::BackendModule::dequant_affine_bf16), under the
/// op name `"dequant_affine_bf16"` every backend reports.
///
/// The operands are checked by [`dequant_affine_bf16_dims`](crate::dequant_affine_bf16_dims)
/// first, then read through the host slice accessors (a device-resident or non-contiguous operand
/// is refused by them, before anything is charged). The `[rows, cols]` bf16 output is charged to
/// `budget` before it is written. The inputs are not scanned: a non-finite scale or bias makes
/// every output of its group non-finite, and so does a finite pair whose affine map overflows, so
/// the check stays on the output, as the device's fault word does.
pub fn dequantize_affine_bf16_tensors(
    packed: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
    spec: AffineSpec,
    budget: &Budget,
) -> Result<Tensor, OjasError> {
    const OP: &str = "dequant_affine_bf16";
    let dims = crate::dequant_affine_bf16_dims(packed, scales, biases, spec)?;
    let (packed, scales, biases) = (
        packed.u32_slice()?,
        scales.bf16_slice()?,
        biases.bf16_slice()?,
    );
    let mut out = Tensor::zeros(&[dims.rows, dims.cols], DType::Bf16, budget)?;
    expand(OP, spec, packed, scales, biases, out.bf16_slice_mut()?)?;
    Ok(out)
}

/// The one dequantisation loop. `packed` is `rows * words` words, `scales` and `biases` are
/// `rows * groups` values, and `dst` is `rows * cols` values; the callers validated the geometry,
/// so a group of `words_per_group` words always pairs with a whole group of `dst`. Any output that
/// is not a finite bf16 is [`OjasError::NonFinite`] under `op`.
fn expand(
    op: &'static str,
    spec: AffineSpec,
    packed: &[u32],
    scales: &[u16],
    biases: &[u16],
    dst: &mut [u16],
) -> Result<(), OjasError> {
    let per_word = spec.bits.codes_per_word();
    let lane_bits = spec.bits.bits();
    let mask = spec.bits.max_code();
    let wpg = spec.words_per_group();
    for (g, (words, values)) in packed
        .chunks_exact(wpg)
        .zip(dst.as_chunks_mut::<AFFINE_GROUP_SIZE>().0)
        .enumerate()
    {
        let scale = flush(bf16_to_f32(scales[g]));
        let bias = flush(bf16_to_f32(biases[g]));
        for (word, lanes) in words.iter().zip(values.chunks_exact_mut(per_word)) {
            for (i, v) in lanes.iter_mut().enumerate() {
                let code = ((word >> (lane_bits * i as u32)) & mask) as f32;
                let r = flush(scale.mul_add(code, bias));
                if !r.is_finite() {
                    return Err(OjasError::NonFinite { op });
                }
                let bits = f32_to_bf16(r);
                if !bf16_is_finite(bits) {
                    return Err(OjasError::NonFinite { op });
                }
                *v = bits;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_max_constant_is_the_bf16_maximum() {
        assert_eq!(BF16_MAX.to_bits(), 0x7F7F_0000);
    }

    #[test]
    fn code_of_maps_signed_zero_and_nan_and_refuses_a_negative() {
        assert!(matches!(code_of(-0.0, 255.0), Ok(0)));
        assert!(matches!(code_of(0.0, 255.0), Ok(0)));
        assert!(matches!(code_of(f32::NAN, 255.0), Ok(255)));
        assert!(matches!(code_of(254.0, 255.0), Ok(254)));
        assert!(matches!(code_of(300.0, 255.0), Ok(255)));
        assert!(matches!(code_of(f32::INFINITY, 15.0), Ok(15)));
        assert!(matches!(
            code_of(-1.0, 255.0),
            Err(GroupRefusal::NegativeCode)
        ));
        assert!(matches!(
            code_of(f32::NEG_INFINITY, 255.0),
            Err(GroupRefusal::NegativeCode)
        ));
    }

    #[test]
    fn flush_zeroes_subnormals_keeping_the_sign() {
        assert_eq!(flush(1e-39).to_bits(), 0);
        assert_eq!(flush(-3e-40).to_bits(), 0x8000_0000);
        assert_eq!(flush(f32::MIN_POSITIVE), f32::MIN_POSITIVE);
        assert_eq!(flush(-1.5), -1.5);
    }
}

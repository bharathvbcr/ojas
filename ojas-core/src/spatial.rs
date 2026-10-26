//! Spatial (image-shaped) ops for SAM-style vision models: the shape contract
//! of 2D convolution, transposed convolution, max pooling, bilinear
//! resampling and interleaved RoPE; the axial 2D RoPE tables; and the layout
//! ops (window split and merge, pixel shuffle) that are views and
//! [`Backend::permute`] rather than kernels.
//!
//! Image tensors are `NCHW` and convolution weights are PyTorch's
//! (`[out, in / groups, kh, kw]`; transposed `[in, out, kh, kw]`), so a
//! checkpoint's tensors load without a transpose. The windowed attention
//! layout is `NHWC`, as SAM's ViT blocks hold it.
//!
//! Like [`crate::shapes`], every `*_dims` function is pure: it reads tensor
//! metadata, never values, placement or a [`crate::Budget`], and a backend
//! calls it before it reserves budget or dispatches. A malformed call is
//! refused with the same variant, op name and detail on every backend.

use crate::shapes::{f32_operand, f32_operands, f32_product, refuse, same, too_large};
use crate::{Backend, Budget, DType, OjasError, Tensor};

const CONV2D: &str = "conv2d_forward";
const CONV_TRANSPOSE2D: &str = "conv_transpose2d_forward";
const MAX_POOL2D: &str = "max_pool2d_forward";
const BILINEAR: &str = "interpolate_bilinear_forward";
const ROPE_INTERLEAVED_FORWARD: &str = "rope_interleaved_forward";
const ROPE_INTERLEAVED_BACKWARD: &str = "rope_interleaved_backward";
const AXIAL_TABLES: &str = "axial_rope_2d_tables";
const WINDOW_PARTITION: &str = "window_partition";
const WINDOW_MERGE: &str = "window_merge";
const PIXEL_SHUFFLE: &str = "pixel_shuffle";
const PIXEL_UNSHUFFLE: &str = "pixel_unshuffle";

/// Hyper-parameters of [`Backend::conv2d_forward`], each pair `[h, w]`.
/// `groups == in_channels` is a depthwise convolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conv2dParams {
    pub stride: [usize; 2],
    /// Zero padding on each side of each spatial axis.
    pub padding: [usize; 2],
    pub dilation: [usize; 2],
    pub groups: usize,
}

impl Conv2dParams {
    /// Stride 1, no padding, dilation 1, one group.
    pub const fn new() -> Self {
        Self {
            stride: [1, 1],
            padding: [0, 0],
            dilation: [1, 1],
            groups: 1,
        }
    }
}

impl Default for Conv2dParams {
    fn default() -> Self {
        Self::new()
    }
}

/// Validated dimensions of [`Backend::conv2d_forward`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conv2dDims {
    pub batch: usize,
    pub in_channels: usize,
    pub height: usize,
    pub width: usize,
    pub out_channels: usize,
    pub kernel: [usize; 2],
    pub params: Conv2dParams,
    pub out_height: usize,
    pub out_width: usize,
}

impl Conv2dDims {
    /// Input channels of one group (the weight's second axis).
    pub fn in_per_group(&self) -> usize {
        self.in_channels / self.params.groups
    }

    /// Output channels of one group.
    pub fn out_per_group(&self) -> usize {
        self.out_channels / self.params.groups
    }

    /// `[batch, out_channels, out_height, out_width]`.
    pub fn out_shape(&self) -> [usize; 4] {
        [
            self.batch,
            self.out_channels,
            self.out_height,
            self.out_width,
        ]
    }
}

/// Hyper-parameters of [`Backend::conv_transpose2d_forward`], each pair
/// `[h, w]`. One group, dilation 1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConvTranspose2dParams {
    pub stride: [usize; 2],
    /// Rows and columns trimmed from each side of the full output.
    pub padding: [usize; 2],
    /// Rows and columns added to the bottom and right of the output; below
    /// the stride on each axis.
    pub output_padding: [usize; 2],
}

impl ConvTranspose2dParams {
    /// Stride 1, no padding, no output padding.
    pub const fn new() -> Self {
        Self {
            stride: [1, 1],
            padding: [0, 0],
            output_padding: [0, 0],
        }
    }
}

impl Default for ConvTranspose2dParams {
    fn default() -> Self {
        Self::new()
    }
}

/// Validated dimensions of [`Backend::conv_transpose2d_forward`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConvTranspose2dDims {
    pub batch: usize,
    pub in_channels: usize,
    pub height: usize,
    pub width: usize,
    pub out_channels: usize,
    pub kernel: [usize; 2],
    pub params: ConvTranspose2dParams,
    pub out_height: usize,
    pub out_width: usize,
}

impl ConvTranspose2dDims {
    /// `[batch, out_channels, out_height, out_width]`.
    pub fn out_shape(&self) -> [usize; 4] {
        [
            self.batch,
            self.out_channels,
            self.out_height,
            self.out_width,
        ]
    }
}

/// Validated dimensions of [`Backend::max_pool2d_forward`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaxPool2dDims {
    pub batch: usize,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
    pub kernel: [usize; 2],
    pub stride: [usize; 2],
    pub out_height: usize,
    pub out_width: usize,
}

impl MaxPool2dDims {
    /// `[batch, channels, out_height, out_width]`.
    pub fn out_shape(&self) -> [usize; 4] {
        [self.batch, self.channels, self.out_height, self.out_width]
    }
}

/// Validated dimensions of [`Backend::interpolate_bilinear_forward`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BilinearDims {
    pub batch: usize,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
    pub out_height: usize,
    pub out_width: usize,
}

impl BilinearDims {
    /// `[batch, channels, out_height, out_width]`.
    pub fn out_shape(&self) -> [usize; 4] {
        [self.batch, self.channels, self.out_height, self.out_width]
    }
}

/// Dimensions of an interleaved RoPE over `[batch, time, heads, dim]` with
/// `[time, dim / 2]` tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InterleavedRopeDims {
    /// `batch * time * heads`.
    pub rows: usize,
    pub time: usize,
    pub heads: usize,
    pub dim: usize,
}

fn rank4(op: &'static str, what: &str, t: &Tensor) -> Result<[usize; 4], OjasError> {
    match *t.shape() {
        [a, b, c, d] => Ok([a, b, c, d]),
        ref other => Err(refuse(
            op,
            format!("{what} rank {} != 4, shape {other:?}", other.len()),
        )),
    }
}

fn positive_pair(op: &'static str, what: &str, v: [usize; 2]) -> Result<(), OjasError> {
    if v.contains(&0) {
        Err(refuse(op, format!("{what} {v:?} must be at least 1")))
    } else {
        Ok(())
    }
}

/// `(size + 2 * pad - span) / stride + 1` where `span` is the extent one
/// window covers, or a refusal when the window does not fit the padded
/// axis.
fn window_count(
    op: &'static str,
    size: usize,
    pad: usize,
    span: usize,
    stride: usize,
) -> Result<usize, OjasError> {
    let padded = pad
        .checked_mul(2)
        .and_then(|p| p.checked_add(size))
        .ok_or_else(|| too_large(op, "padded extent overflows"))?;
    let room = padded.checked_sub(span).ok_or_else(|| {
        refuse(
            op,
            format!("window extent {span} exceeds the padded input extent {padded}"),
        )
    })?;
    Ok(room / stride + 1)
}

fn bias_vector(op: &'static str, bias: Option<&Tensor>, len: usize) -> Result<(), OjasError> {
    let Some(bias) = bias else { return Ok(()) };
    f32_operand(op, bias)?;
    if bias.shape() != [len] {
        return Err(refuse(
            op,
            format!("bias shape {:?} != [{len}] (out channels)", bias.shape()),
        ));
    }
    Ok(())
}

/// [`Backend::conv2d_forward`]: `input` `[N, C, H, W]`, `weight`
/// `[O, C / groups, kh, kw]` and an optional `bias` `[O]`, all `F32`.
/// `stride`, `dilation` and `groups` are at least 1, `groups` divides both
/// `C` and `O`, and the dilated kernel fits the padded input. The output is
/// `[N, O, Ho, Wo]` with `Ho = (H + 2 pad - dil (kh - 1) - 1) / stride + 1`.
pub fn conv2d_forward_dims(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    params: Conv2dParams,
) -> Result<Conv2dDims, OjasError> {
    const OP: &str = CONV2D;
    f32_operands(OP, &[input, weight])?;
    let [batch, in_channels, height, width] = rank4(OP, "input", input)?;
    let [out_channels, per_group, kh, kw] = rank4(OP, "weight", weight)?;
    positive_pair(OP, "stride", params.stride)?;
    positive_pair(OP, "dilation", params.dilation)?;
    let groups = params.groups;
    if groups == 0 || in_channels % groups != 0 || out_channels % groups != 0 {
        return Err(refuse(
            OP,
            format!(
                "groups {groups} must be at least 1 and divide the input channels \
                 {in_channels} and the output channels {out_channels}"
            ),
        ));
    }
    if per_group != in_channels / groups {
        return Err(refuse(
            OP,
            format!(
                "weight in-channels {per_group} != input channels {in_channels} / groups {groups}"
            ),
        ));
    }
    bias_vector(OP, bias, out_channels)?;
    let span = |k: usize, d: usize| d.checked_mul(k - 1).and_then(|s| s.checked_add(1));
    let (span_h, span_w) = span(kh, params.dilation[0])
        .zip(span(kw, params.dilation[1]))
        .ok_or_else(|| too_large(OP, "dilated kernel extent overflows"))?;
    let out_height = window_count(OP, height, params.padding[0], span_h, params.stride[0])?;
    let out_width = window_count(OP, width, params.padding[1], span_w, params.stride[1])?;
    let dims = Conv2dDims {
        batch,
        in_channels,
        height,
        width,
        out_channels,
        kernel: [kh, kw],
        params,
        out_height,
        out_width,
    };
    f32_product(OP, &dims.out_shape())?;
    // The im2col matrix of one group, `[in / groups * kh * kw, Ho * Wo]`.
    f32_product(OP, &[per_group, kh, kw, out_height, out_width])?;
    Ok(dims)
}

/// [`Backend::conv_transpose2d_forward`]: `input` `[N, C, H, W]`, `weight`
/// `[C, O, kh, kw]` and an optional `bias` `[O]`, all `F32`. `stride` is at
/// least 1, `output_padding` is below `stride`, and `padding` leaves a
/// non-empty output. The output is `[N, O, Ho, Wo]` with `Ho = (H - 1)
/// stride - 2 pad + kh + output_padding`.
pub fn conv_transpose2d_forward_dims(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    params: ConvTranspose2dParams,
) -> Result<ConvTranspose2dDims, OjasError> {
    const OP: &str = CONV_TRANSPOSE2D;
    f32_operands(OP, &[input, weight])?;
    let [batch, in_channels, height, width] = rank4(OP, "input", input)?;
    let [w_in, out_channels, kh, kw] = rank4(OP, "weight", weight)?;
    positive_pair(OP, "stride", params.stride)?;
    if w_in != in_channels {
        return Err(refuse(
            OP,
            format!("weight in-channels {w_in} != input channels {in_channels}"),
        ));
    }
    for axis in 0..2 {
        if params.output_padding[axis] >= params.stride[axis] {
            return Err(refuse(
                OP,
                format!(
                    "output_padding {:?} must be below stride {:?}",
                    params.output_padding, params.stride
                ),
            ));
        }
    }
    bias_vector(OP, bias, out_channels)?;
    let extent = |size: usize, k: usize, axis: usize| {
        (size - 1)
            .checked_mul(params.stride[axis])
            .and_then(|v| v.checked_add(k))
            .and_then(|v| v.checked_add(params.output_padding[axis]))
            .ok_or_else(|| too_large(OP, "output extent overflows"))
            .and_then(|full| {
                params.padding[axis]
                    .checked_mul(2)
                    .and_then(|trim| full.checked_sub(trim))
                    .filter(|&n| n > 0)
                    .ok_or_else(|| {
                        refuse(
                            OP,
                            format!(
                                "padding {} leaves no output on axis {axis} (full extent {full})",
                                params.padding[axis]
                            ),
                        )
                    })
            })
    };
    let out_height = extent(height, kh, 0)?;
    let out_width = extent(width, kw, 1)?;
    let dims = ConvTranspose2dDims {
        batch,
        in_channels,
        height,
        width,
        out_channels,
        kernel: [kh, kw],
        params,
        out_height,
        out_width,
    };
    f32_product(OP, &dims.out_shape())?;
    // The column matrix, `[O * kh * kw, H * W]`.
    f32_product(OP, &[out_channels, kh, kw, height, width])?;
    Ok(dims)
}

/// [`Backend::max_pool2d_forward`]: `input` `[N, C, H, W]` `F32`, `kernel`
/// and `stride` at least 1, the kernel no larger than the input. The output
/// is `[N, C, (H - kh) / sh + 1, (W - kw) / sw + 1]`; there is no padding
/// and a window never leaves the input.
pub fn max_pool2d_forward_dims(
    input: &Tensor,
    kernel: [usize; 2],
    stride: [usize; 2],
) -> Result<MaxPool2dDims, OjasError> {
    const OP: &str = MAX_POOL2D;
    f32_operand(OP, input)?;
    let [batch, channels, height, width] = rank4(OP, "input", input)?;
    positive_pair(OP, "kernel", kernel)?;
    positive_pair(OP, "stride", stride)?;
    let out_height = window_count(OP, height, 0, kernel[0], stride[0])?;
    let out_width = window_count(OP, width, 0, kernel[1], stride[1])?;
    let dims = MaxPool2dDims {
        batch,
        channels,
        height,
        width,
        kernel,
        stride,
        out_height,
        out_width,
    };
    f32_product(OP, &dims.out_shape())?;
    Ok(dims)
}

/// [`Backend::interpolate_bilinear_forward`]: `input` `[N, C, H, W]` `F32`
/// and a target size of at least 1 on each axis. The output is
/// `[N, C, out_h, out_w]`.
pub fn interpolate_bilinear_forward_dims(
    input: &Tensor,
    out_size: [usize; 2],
) -> Result<BilinearDims, OjasError> {
    const OP: &str = BILINEAR;
    f32_operand(OP, input)?;
    let [batch, channels, height, width] = rank4(OP, "input", input)?;
    positive_pair(OP, "out_size", out_size)?;
    let dims = BilinearDims {
        batch,
        channels,
        height,
        width,
        out_height: out_size[0],
        out_width: out_size[1],
    };
    f32_product(OP, &dims.out_shape())?;
    Ok(dims)
}

fn rope_interleaved_layout(
    op: &'static str,
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<InterleavedRopeDims, OjasError> {
    f32_operands(op, &[x, cos, sin])?;
    let [batch, time, heads, dim] = rank4(op, "x", x)?;
    if dim % 2 != 0 {
        return Err(refuse(op, format!("head dim {dim} must be even")));
    }
    same(op, cos.shape(), sin.shape())?;
    if cos.shape() != [time, dim / 2] {
        return Err(refuse(
            op,
            format!(
                "tables {:?} != [time {time}, dim / 2 = {}]",
                cos.shape(),
                dim / 2
            ),
        ));
    }
    let rows = f32_product(op, &[batch, time, heads])?;
    Ok(InterleavedRopeDims {
        rows,
        time,
        heads,
        dim,
    })
}

/// [`Backend::rope_interleaved_forward`]: `x` `[batch, time, heads, dim]`
/// with `dim` even and `cos`, `sin` `[time, dim / 2]`, all `F32`.
pub fn rope_interleaved_forward_dims(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<InterleavedRopeDims, OjasError> {
    rope_interleaved_layout(ROPE_INTERLEAVED_FORWARD, x, cos, sin)
}

/// [`Backend::rope_interleaved_backward`]: `grad_output` shaped like `x`.
pub fn rope_interleaved_backward_dims(
    grad_output: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<InterleavedRopeDims, OjasError> {
    rope_interleaved_layout(ROPE_INTERLEAVED_BACKWARD, grad_output, cos, sin)
}

/// `cos` and `sin`, each `[grid_h * grid_w, head_dim / 2]`, of SAM2 and SAM3's
/// 2D axial RoPE (`compute_axial_cis`), for [`Backend::rope_interleaved_forward`].
///
/// Token `t` of the row-major grid sits at column `t % grid_w` and row `t /
/// grid_w`. The first `head_dim / 4` pairs rotate by `column * scale_pos *
/// f_j` and the last `head_dim / 4` by `row * scale_pos * f_j`, with `f_j =
/// 1 / theta^(4 j / head_dim)`. Frequencies and angles are formed in f32 as
/// torch forms them (the power comes from the platform's `powf`); only the
/// cosine and sine are taken in f64 and rounded once. Nothing is normalised:
/// this is the rotation alone, with no QK-norm.
///
/// Refused: `head_dim` not a positive multiple of 4, an empty grid, `theta`
/// not finite and above 1 as an f32 (the power is formed in f32, so a theta
/// that rounds to infinity or to 1 there is refused, not silently
/// degenerate), `scale_pos` not finite and positive, or one that makes an
/// angle overflow f32. The two `[tokens, head_dim / 2]` tables and the
/// `head_dim / 4` frequencies are checked against `budget` before anything is
/// allocated, and the frequencies are charged while they exist.
pub fn axial_rope_2d_tables(
    head_dim: usize,
    grid_h: usize,
    grid_w: usize,
    theta: f64,
    scale_pos: f32,
    budget: &Budget,
) -> Result<(Tensor, Tensor), OjasError> {
    const OP: &str = AXIAL_TABLES;
    let range = |detail: String| OjasError::OutOfRange { op: OP, detail };
    if head_dim == 0 || !head_dim.is_multiple_of(4) {
        return Err(range(format!(
            "head_dim {head_dim} must be a positive multiple of 4"
        )));
    }
    if grid_h == 0 || grid_w == 0 {
        return Err(refuse(
            OP,
            format!("grid {grid_h} x {grid_w} must be non-empty"),
        ));
    }
    let theta32 = theta as f32;
    if !(theta.is_finite() && theta > 1.0 && theta32.is_finite() && theta32 > 1.0) {
        return Err(range(format!(
            "theta {theta} must be finite and above 1 as an f32"
        )));
    }
    if !(scale_pos.is_finite() && scale_pos > 0.0) {
        return Err(range(format!(
            "scale_pos {scale_pos} must be finite and positive"
        )));
    }
    let tokens = grid_h
        .checked_mul(grid_w)
        .ok_or_else(|| too_large(OP, "grid size overflows"))?;
    let (half, quarter) = (head_dim / 2, head_dim / 4);
    // Everything that will exist at once, in bytes: both tables and the
    // frequencies. Refused before the first allocation, so a huge `head_dim`
    // is an error and not an allocator abort.
    let f32_bytes = DType::F32.size();
    let bytes = |elements: Option<usize>| {
        elements
            .and_then(|n| n.checked_mul(f32_bytes))
            .ok_or_else(|| too_large(OP, "table size overflows"))
    };
    let table_bytes = bytes(tokens.checked_mul(half))?;
    let freq_bytes = bytes(Some(quarter))?;
    let total = table_bytes
        .checked_mul(2)
        .and_then(|t| t.checked_add(freq_bytes))
        .and_then(|t| u64::try_from(t).ok())
        .ok_or_else(|| too_large(OP, "table size overflows"))?;
    budget.check_room(total)?;
    let _freqs_hold = budget.try_reserve(freq_bytes as u64)?;
    let mut freqs: Vec<f32> = Vec::new();
    if freqs.try_reserve_exact(quarter).is_err() {
        return Err(OjasError::CapacityExceeded {
            requested: freq_bytes as u64,
            cap: budget.cap_bytes(),
            live: budget.live_bytes()?,
        });
    }
    freqs.extend((0..quarter).map(|j| 1.0 / theta32.powf((4 * j) as f32 / head_dim as f32)));
    let mut cos = Tensor::zeros(&[tokens, half], DType::F32, budget)?;
    let mut sin = Tensor::zeros(&[tokens, half], DType::F32, budget)?;
    {
        let (c, s) = (cos.f32_slice_mut()?, sin.f32_slice_mut()?);
        let rows = c.chunks_exact_mut(half).zip(s.chunks_exact_mut(half));
        for (t, (c_row, s_row)) in rows.enumerate() {
            let (col, row) = ((t % grid_w) as f32, (t / grid_w) as f32);
            let (col, row) = (col * scale_pos, row * scale_pos);
            for (j, &f) in freqs.iter().enumerate() {
                for (pair, pos) in [(j, col), (quarter + j, row)] {
                    let angle = pos * f;
                    if !angle.is_finite() {
                        return Err(range(format!(
                            "scale_pos {scale_pos} overflows the angle at token {t}"
                        )));
                    }
                    let (sv, cv) = f64::from(angle).sin_cos();
                    c_row[pair] = cv as f32;
                    s_row[pair] = sv as f32;
                }
            }
        }
    }
    Ok((cos, sin))
}

fn divides(op: &'static str, what: &str, size: usize, by: usize) -> Result<usize, OjasError> {
    if by == 0 || !size.is_multiple_of(by) {
        Err(refuse(
            op,
            format!(
                "{what} {size} is not a multiple of {by}; padding is not implemented, \
                 so the caller pads"
            ),
        ))
    } else {
        Ok(size / by)
    }
}

/// Splits `x` `[B, H, W, C]` into non-overlapping `window x window` windows:
/// `[B * (H / window) * (W / window), window, window, C]`, windows in
/// row-major grid order within each batch element. A reshape (a view), one
/// [`Backend::permute`] and a reshape. `H` and `W` must be multiples of
/// `window`; there is no padding. Inverse of [`window_merge`].
pub fn window_partition<B: Backend + ?Sized>(
    backend: &B,
    x: &Tensor,
    window: usize,
) -> Result<Tensor, OjasError> {
    const OP: &str = WINDOW_PARTITION;
    f32_operand(OP, x)?;
    let &[b, h, w, c] = x.shape() else {
        return Err(refuse(
            OP,
            format!(
                "input rank {} != 4 [batch, height, width, channels]",
                x.shape().len()
            ),
        ));
    };
    let (nh, nw) = (
        divides(OP, "height", h, window)?,
        divides(OP, "width", w, window)?,
    );
    let split = x.reshape(&[b, nh, window, nw, window, c])?;
    let grouped = backend.permute(&split, &[0, 1, 3, 2, 4, 5])?;
    grouped.reshape(&[b * nh * nw, window, window, c])
}

/// Merges `windows` `[B * (H / window) * (W / window), window, window, C]`
/// back into `[B, H, W, C]`. Inverse of [`window_partition`].
pub fn window_merge<B: Backend + ?Sized>(
    backend: &B,
    windows: &Tensor,
    height: usize,
    width: usize,
) -> Result<Tensor, OjasError> {
    const OP: &str = WINDOW_MERGE;
    f32_operand(OP, windows)?;
    let &[count, window, w2, c] = windows.shape() else {
        return Err(refuse(
            OP,
            format!(
                "windows rank {} != 4 [windows, window, window, channels]",
                windows.shape().len()
            ),
        ));
    };
    if window != w2 {
        return Err(refuse(
            OP,
            format!("windows {window} x {w2} are not square"),
        ));
    }
    let (nh, nw) = (
        divides(OP, "height", height, window)?,
        divides(OP, "width", width, window)?,
    );
    let per_image = nh * nw;
    let b = divides(OP, "window count", count, per_image)?;
    let split = windows.reshape(&[b, nh, nw, window, window, c])?;
    let grouped = backend.permute(&split, &[0, 1, 3, 2, 4, 5])?;
    grouped.reshape(&[b, height, width, c])
}

/// `torch.nn.functional.pixel_shuffle`: `x` `[N, C * r * r, H, W]` to
/// `[N, C, H * r, W * r]`, channel `c * r * r + i * r + j` landing at row
/// `h * r + i`, column `w * r + j`. A view and one [`Backend::permute`].
pub fn pixel_shuffle<B: Backend + ?Sized>(
    backend: &B,
    x: &Tensor,
    factor: usize,
) -> Result<Tensor, OjasError> {
    const OP: &str = PIXEL_SHUFFLE;
    f32_operand(OP, x)?;
    let &[n, ch, h, w] = x.shape() else {
        return Err(refuse(
            OP,
            format!("input rank {} != 4 [N, C, H, W]", x.shape().len()),
        ));
    };
    let r2 = factor
        .checked_mul(factor)
        .ok_or_else(|| too_large(OP, "factor squared overflows"))?;
    let c = divides(OP, "channels", ch, r2)?;
    let split = x.reshape(&[n, c, factor, factor, h, w])?;
    let moved = backend.permute(&split, &[0, 1, 4, 2, 5, 3])?;
    moved.reshape(&[n, c, h * factor, w * factor])
}

/// `torch.nn.functional.pixel_unshuffle`, the inverse of [`pixel_shuffle`]:
/// `x` `[N, C, H * r, W * r]` to `[N, C * r * r, H, W]`.
pub fn pixel_unshuffle<B: Backend + ?Sized>(
    backend: &B,
    x: &Tensor,
    factor: usize,
) -> Result<Tensor, OjasError> {
    const OP: &str = PIXEL_UNSHUFFLE;
    f32_operand(OP, x)?;
    let &[n, c, hr, wr] = x.shape() else {
        return Err(refuse(
            OP,
            format!("input rank {} != 4 [N, C, H, W]", x.shape().len()),
        ));
    };
    let (h, w) = (
        divides(OP, "height", hr, factor)?,
        divides(OP, "width", wr, factor)?,
    );
    let split = x.reshape(&[n, c, h, factor, w, factor])?;
    let moved = backend.permute(&split, &[0, 1, 3, 5, 2, 4])?;
    moved.reshape(&[n, c * factor * factor, h, w])
}

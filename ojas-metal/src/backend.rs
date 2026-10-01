//! [`MetalBackend`]: `ojas_core::Backend` on device-resident Metal tensors.
//!
//! Every input must already live on this backend's device (see
//! [`Backend::upload`]); a host tensor or another device's tensor is
//! [`OjasError::Placement`]. Outputs are fresh device buffers wrapped with
//! [`Tensor::from_device_reserved`], so nothing comes back to the host until
//! [`Tensor::to_host`] or [`Backend::download`].
//!
//! Each op reserves its output and scratch bytes in the [`Budget`] before the
//! device allocates anything, and refuses with
//! [`OjasError::CapacityExceeded`] when they do not fit. Device memory is
//! tessl's pool, which rounds each allocation up to a power of two (at least
//! 256 bytes); the budget charges the logical byte count, so resident device
//! memory can be up to twice the budget's live bytes.
//!
//! Numerics are [`Numerics::Fast`]: GEMMs are tessl TensorOps with their own
//! reduction order, and reductions are threadgroup trees, not ascending f32.
//!
//! Validation order differs from the CPU reference in one way: shape, dtype
//! and placement are checked on the host first, and non-finite values are
//! found by a device kernel, so an input that is both misshapen and non-finite
//! reports the shape error.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use ojas_core::{
    check_adamw, clip_scale, permute_output_shape, refuse_unsupported_metal_head_dim, AdamWConfig,
    Backend, BackendId, Budget, DType, DeviceBuffer, MuonNs5Config, Numerics, OjasError,
    PerHeadGateGrad, Reservation, Tensor, ValueResidualGrad, MAX_PERMUTE_RANK,
};

use crate::link::{metal_err, rms_w_chunks, Arg, Cmd, Link, Reply, Res, RopeMode};

/// A Metal allocation owned by the device thread, named by id.
pub struct MetalBuffer {
    id: u64,
    len: usize,
    link: Arc<Link>,
}

impl fmt::Debug for MetalBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MetalBuffer")
            .field("id", &self.id)
            .field("len", &self.len)
            .finish()
    }
}

impl Drop for MetalBuffer {
    fn drop(&mut self) {
        self.link.post(Cmd::Free { id: self.id });
    }
}

impl DeviceBuffer for MetalBuffer {
    fn backend(&self) -> BackendId {
        BackendId::Metal
    }

    fn byte_len(&self) -> usize {
        self.len
    }

    fn read_bytes(&self, offset: usize, len: usize) -> Result<Vec<u8>, OjasError> {
        match self.link.call(Cmd::Read {
            id: self.id,
            off: offset,
            len,
        })? {
            Reply::Bytes(bytes) => Ok(bytes),
            other => Err(metal_err(format!("read returned {other:?}"))),
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Device-resident Metal backend. Clones share one device thread; any
/// number of threads may issue ops on one backend, and the device runs them
/// one at a time.
#[derive(Clone)]
pub struct MetalBackend {
    link: Arc<Link>,
    budget: Budget,
}

impl fmt::Debug for MetalBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MetalBackend")
            .field("device", &self.link.device_name())
            .finish()
    }
}

impl MetalBackend {
    /// Open the default Metal device on its own thread. Off macOS, or without
    /// the `metal` feature, this is [`OjasError::Unsupported`].
    pub fn new(budget: Budget) -> Result<Self, OjasError> {
        let (tx, name) = crate::device::spawn()?;
        Ok(Self {
            link: Arc::new(Link::new(tx, name)),
            budget,
        })
    }

    pub fn device_name(&self) -> &str {
        self.link.device_name()
    }

    fn arg(&self, op: &'static str, t: &Tensor, dtype: DType) -> Res<Arg> {
        let buf = t.device_buffer().ok_or(OjasError::Placement {
            op,
            expected: Some(BackendId::Metal),
            found: None,
        })?;
        if buf.backend() != BackendId::Metal {
            return Err(OjasError::Placement {
                op,
                expected: Some(BackendId::Metal),
                found: Some(buf.backend()),
            });
        }
        let mb = buf
            .as_any()
            .downcast_ref::<MetalBuffer>()
            .ok_or_else(|| metal_err(format!("{op}: Metal tensor is not an ojas-metal buffer")))?;
        if !Arc::ptr_eq(&mb.link, &self.link) {
            return Err(metal_err(format!(
                "{op}: tensor belongs to a different MetalBackend device"
            )));
        }
        if t.dtype() != dtype {
            return Err(OjasError::Dtype {
                op,
                expected: dtype,
                got: t.dtype(),
            });
        }
        if !t.is_contiguous()? {
            return Err(shape(op, "non-contiguous view is not supported"));
        }
        let n = t.num_elements()?;
        if n == 0 || t.shape().contains(&0) {
            return Err(shape(op, "empty tensor"));
        }
        if n > u32::MAX as usize {
            return Err(OjasError::Unsupported {
                op,
                detail: format!("{n} elements exceed the kernels' 32-bit indexing"),
            });
        }
        Ok(Arg {
            id: mb.id,
            off: t.byte_offset(),
            n,
        })
    }

    fn f32(&self, op: &'static str, t: &Tensor) -> Res<Arg> {
        self.arg(op, t, DType::F32)
    }

    fn reserve(&self, op: &'static str, elems: usize) -> Res<Reservation> {
        let bytes = elems.checked_mul(4).ok_or_else(|| overflow(op))?;
        self.budget.try_reserve(bytes as u64)
    }

    /// Reserve every output, hold `scratch` elements for the call, run `cmd`,
    /// and wrap the outputs in the order given.
    fn outputs(
        &self,
        op: &'static str,
        shapes: &[&[usize]],
        scratch: usize,
        cmd: Cmd,
    ) -> Res<Vec<Tensor>> {
        let mut reservations = Vec::with_capacity(shapes.len());
        for s in shapes {
            reservations.push(self.reserve(op, product(op, s)?)?);
        }
        let _scratch = self.reserve(op, scratch)?;
        let bufs = match self.link.call(cmd)? {
            Reply::Bufs(bufs) => bufs,
            other => return Err(metal_err(format!("{op}: device returned {other:?}"))),
        };
        let owned: Vec<Arc<MetalBuffer>> = bufs
            .into_iter()
            .map(|b| {
                Arc::new(MetalBuffer {
                    id: b.id,
                    len: b.bytes,
                    link: Arc::clone(&self.link),
                })
            })
            .collect();
        if owned.len() != shapes.len() {
            return Err(metal_err(format!(
                "{op}: device returned {} outputs, expected {}",
                owned.len(),
                shapes.len()
            )));
        }
        let mut out = Vec::with_capacity(owned.len());
        for ((buf, s), r) in owned.into_iter().zip(shapes).zip(reservations) {
            let dynbuf: Arc<dyn DeviceBuffer> = buf;
            out.push(Tensor::from_device_reserved(dynbuf, s, DType::F32, r)?);
        }
        Ok(out)
    }

    fn one(&self, op: &'static str, s: &[usize], scratch: usize, cmd: Cmd) -> Res<Tensor> {
        let mut v = self.outputs(op, &[s], scratch, cmd)?;
        v.pop().ok_or_else(|| metal_err(format!("{op}: no output")))
    }

    fn two(&self, op: &'static str, a: &[usize], b: &[usize], cmd: Cmd) -> Res<(Tensor, Tensor)> {
        let mut v = self.outputs(op, &[a, b], 0, cmd)?.into_iter();
        match (v.next(), v.next()) {
            (Some(x), Some(y)) => Ok((x, y)),
            _ => Err(metal_err(format!("{op}: missing outputs"))),
        }
    }

    /// The device id of a tensor this call may write in place. The tensor
    /// must solely own its buffer, as [`Tensor::write_f32`] requires on the
    /// host; a shared one is [`OjasError::Shape`].
    fn unique(&self, op: &'static str, t: &mut Tensor) -> Res<()> {
        let buf = t.device_buffer_mut()?;
        if buf.as_any().downcast_ref::<MetalBuffer>().is_none() {
            return Err(metal_err(format!("{op}: not an ojas-metal buffer")));
        }
        Ok(())
    }
}

fn shape(op: &'static str, detail: impl Into<String>) -> OjasError {
    OjasError::Shape {
        op,
        detail: detail.into(),
    }
}

fn overflow(op: &'static str) -> OjasError {
    OjasError::OutOfRange {
        op,
        detail: "size overflows".to_string(),
    }
}

fn product(op: &'static str, dims: &[usize]) -> Res<usize> {
    dims.iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| overflow(op))
}

/// The per-step f32 scalars of `ojas_adamw_*`, formed in f64 from the bias
/// corrections [`check_adamw`] returns (so `1 - beta^step` has the CPU
/// reference's bits), in `OjasAdamW` field order. A scalar that does not
/// survive the f32 rounding is [`OjasError::NonFinite`] before the device
/// sees it.
fn adamw_scalars(config: AdamWConfig, bc1: f64, bc2: f64) -> Res<[f32; 7]> {
    let s = [
        (1.0 - config.lr * config.weight_decay) as f32,
        (1.0 - config.beta1) as f32,
        config.beta2 as f32,
        (1.0 - config.beta2) as f32,
        (config.lr / bc1) as f32,
        bc2.sqrt() as f32,
        config.eps as f32,
    ];
    if s.iter().any(|v| !v.is_finite()) {
        return Err(OjasError::NonFinite { op: "adamw_step" });
    }
    Ok(s)
}

/// Row-major element strides of `shape`. The input has already passed
/// [`MetalBackend::arg`], so the product fits u32 and cannot overflow.
fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

fn u32_dim(op: &'static str, n: usize) -> Res<u32> {
    u32::try_from(n).map_err(|_| OjasError::Unsupported {
        op,
        detail: format!("dimension {n} exceeds the kernels' 32-bit indexing"),
    })
}

fn same(op: &'static str, a: &[usize], b: &[usize]) -> Res<()> {
    if a == b {
        Ok(())
    } else {
        Err(shape(op, format!("shape {a:?} does not match {b:?}")))
    }
}

/// `(rows, last)` of `[..., last]`.
fn rows_last(op: &'static str, s: &[usize]) -> Res<(usize, usize)> {
    let last = *s.last().ok_or_else(|| shape(op, "rank 0 input"))?;
    Ok((product(op, &s[..s.len() - 1])?, last))
}

fn linear_dims(
    op: &'static str,
    x: &[usize],
    w: &[usize],
) -> Res<(usize, usize, usize, Vec<usize>)> {
    if w.len() != 2 {
        return Err(shape(
            op,
            format!("weight rank {} != 2 ([out, in])", w.len()),
        ));
    }
    let (rows, kin) = rows_last(op, x)?;
    if kin != w[1] {
        return Err(shape(
            op,
            format!("input in-features {kin} != weight in-features {}", w[1]),
        ));
    }
    let mut y = x[..x.len() - 1].to_vec();
    y.push(w[0]);
    Ok((rows, kin, w[0], y))
}

fn rms_dims(op: &'static str, x: &[usize], w: &[usize], eps: f32) -> Res<(u32, u32)> {
    if !eps.is_finite() {
        return Err(OjasError::NonFinite { op });
    }
    let (rows, dim) = rows_last(op, x)?;
    if w.len() != 1 || w[0] != dim {
        return Err(shape(op, format!("rms weight {w:?} != last dim {dim}")));
    }
    Ok((u32_dim(op, rows)?, u32_dim(op, dim)?))
}

fn rope_dims(op: &'static str, x: &[usize], c: &[usize], s: &[usize]) -> Res<(u32, u32, RopeMode)> {
    if c != s {
        return Err(shape(op, format!("cos shape {c:?} != sin shape {s:?}")));
    }
    let (rows, dim) = rows_last(op, x)?;
    if dim % 2 != 0 {
        return Err(shape(op, format!("rope last dim {dim} is odd")));
    }
    let mode = if c == x {
        RopeMode::Same
    } else if x.len() == 4 && c.len() == 2 && c[0] == x[1] && c[1] == dim {
        RopeMode::TimeDim {
            time: u32_dim(op, x[1])?,
            heads: u32_dim(op, x[2])?,
        }
    } else {
        return Err(shape(
            op,
            format!("cos/sin shape {c:?} does not broadcast onto {x:?}"),
        ));
    };
    Ok((u32_dim(op, rows)?, u32_dim(op, dim)?, mode))
}

/// `(bh, t, d)` of `[B, H, T, D]`, with the Metal head-dim refusal.
fn sdpa_dims(op: &'static str, q: &[usize], k: &[usize], v: &[usize]) -> Res<(u32, u32, u32)> {
    if q.len() != 4 {
        return Err(shape(
            op,
            format!("sdpa query rank {} != 4 [B, H, T, D]", q.len()),
        ));
    }
    if k != q || v != q {
        return Err(shape(
            op,
            format!("sdpa shapes q {q:?} k {k:?} v {v:?} differ"),
        ));
    }
    let d = u32_dim(op, q[3])?;
    refuse_unsupported_metal_head_dim(BackendId::Metal, d)?;
    Ok((u32_dim(op, q[0] * q[1])?, u32_dim(op, q[2])?, d))
}

struct GateDims {
    rows: u32,
    din: u32,
    heads: u32,
    dh: u32,
}

fn gate_dims(
    op: &'static str,
    x: &[usize],
    w: &[usize],
    b: &[usize],
    a: &[usize],
) -> Res<GateDims> {
    if w.len() != 2 {
        return Err(shape(op, "gate weight must be [n_head, d_model]"));
    }
    let (rows, din) = rows_last(op, x)?;
    if w[1] != din {
        return Err(shape(
            op,
            format!("gate weight in {} != input dim {din}", w[1]),
        ));
    }
    let heads = w[0];
    if b.len() != 1 || b[0] != heads {
        return Err(shape(op, format!("gate bias {b:?} != [{heads}]")));
    }
    if a.len() != x.len() + 1 {
        return Err(shape(op, "gate attn rank must be input rank + 1"));
    }
    if a[a.len() - 2] != heads {
        return Err(shape(op, "gate attn head axis != weight rows"));
    }
    if a[..a.len() - 2] != x[..x.len() - 1] {
        return Err(shape(op, "gate attn prefix does not match input prefix"));
    }
    Ok(GateDims {
        rows: u32_dim(op, rows)?,
        din: u32_dim(op, din)?,
        heads: u32_dim(op, heads)?,
        dh: u32_dim(op, a[a.len() - 1])?,
    })
}

fn embed_dims(op: &'static str, table: &[usize]) -> Res<(u32, u32)> {
    if table.len() != 2 {
        return Err(shape(
            op,
            format!("embedding table rank {} != 2 [vocab, dim]", table.len()),
        ));
    }
    Ok((u32_dim(op, table[0])?, u32_dim(op, table[1])?))
}

fn ce_dims(op: &'static str, logits: &[usize], targets: &[usize]) -> Res<(u32, u32)> {
    let (rows, vocab) = rows_last(op, logits)?;
    let prefix = &logits[..logits.len() - 1];
    if targets != prefix {
        return Err(shape(
            op,
            format!("targets {targets:?} != logits prefix {prefix:?}"),
        ));
    }
    Ok((u32_dim(op, rows)?, u32_dim(op, vocab)?))
}

impl Backend for MetalBackend {
    fn id(&self) -> BackendId {
        BackendId::Metal
    }

    fn budget(&self) -> &Budget {
        &self.budget
    }

    fn numerics(&self) -> Numerics {
        Numerics::Fast
    }

    /// Host bytes are copied to a new device buffer. A tensor already on this
    /// device is a shared clone; another device's tensor is
    /// [`OjasError::Placement`].
    fn upload(&self, tensor: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "MetalBackend::upload";
        match tensor.device_buffer() {
            Some(buf) => {
                let ours = buf
                    .as_any()
                    .downcast_ref::<MetalBuffer>()
                    .is_some_and(|mb| Arc::ptr_eq(&mb.link, &self.link));
                if ours {
                    Ok(tensor.clone())
                } else {
                    Err(OjasError::Placement {
                        op: OP,
                        expected: Some(BackendId::Metal),
                        found: Some(buf.backend()),
                    })
                }
            }
            None => {
                let bytes = tensor.contiguous_bytes()?;
                if bytes.is_empty() {
                    return Err(shape(OP, "empty tensor"));
                }
                let reservation = self.budget.try_reserve(bytes.len() as u64)?;
                let nb = match self.link.call(Cmd::Upload {
                    bytes: bytes.to_vec(),
                })? {
                    Reply::Bufs(mut v) if v.len() == 1 => v.remove(0),
                    other => return Err(metal_err(format!("{OP}: device returned {other:?}"))),
                };
                let buf: Arc<dyn DeviceBuffer> = Arc::new(MetalBuffer {
                    id: nb.id,
                    len: nb.bytes,
                    link: Arc::clone(&self.link),
                });
                Tensor::from_device_reserved(buf, tensor.shape(), tensor.dtype(), reservation)
            }
        }
    }

    /// A bit-exact device copy. As on the CPU reference, a NaN or infinity
    /// in the input is [`OjasError::NonFinite`]. Unlike the CPU, an input
    /// with a zero-length axis is [`OjasError::Shape`], as for every Metal
    /// op: a Metal tensor is never empty.
    fn permute(&self, input: &Tensor, dims: &[usize]) -> Result<Tensor, OjasError> {
        const OP: &str = "permute";
        let x = self.f32(OP, input)?;
        let out = permute_output_shape(OP, input.shape(), dims)?;
        let in_strides = row_major_strides(input.shape());
        let mut oshape = [1u32; MAX_PERMUTE_RANK];
        let mut istride = [0u32; MAX_PERMUTE_RANK];
        for (a, &axis) in dims.iter().enumerate() {
            oshape[a] = u32_dim(OP, out[a])?;
            istride[a] = u32_dim(OP, in_strides[axis])?;
        }
        self.one(
            OP,
            &out,
            0,
            Cmd::Permute {
                x,
                rank: u32_dim(OP, dims.len())?,
                oshape,
                istride,
            },
        )
    }

    fn embedding_forward(&self, table: &Tensor, token_ids: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "embedding_forward";
        let t = self.f32(OP, table)?;
        let ids = self.arg(OP, token_ids, DType::U32)?;
        let (vocab, dim) = embed_dims(OP, table.shape())?;
        let mut out = token_ids.shape().to_vec();
        out.push(dim as usize);
        u32_dim(OP, product(OP, &out)?)?;
        self.one(
            OP,
            &out,
            0,
            Cmd::Embed {
                table: t,
                ids,
                vocab,
                dim,
            },
        )
    }

    fn embedding_backward(
        &self,
        table: &Tensor,
        token_ids: &Tensor,
        grad_output: &Tensor,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "embedding_backward";
        let t = self.f32(OP, table)?;
        let ids = self.arg(OP, token_ids, DType::U32)?;
        let g = self.f32(OP, grad_output)?;
        let (vocab, dim) = embed_dims(OP, table.shape())?;
        let mut expect = token_ids.shape().to_vec();
        expect.push(dim as usize);
        if grad_output.shape() != expect.as_slice() {
            return Err(shape(
                OP,
                format!(
                    "embedding grad shape {:?} != {expect:?}",
                    grad_output.shape()
                ),
            ));
        }
        let scratch = 2 * vocab as usize + ids.n;
        self.one(
            OP,
            table.shape(),
            scratch,
            Cmd::EmbedBwd {
                table: t,
                ids,
                grad: g,
                vocab,
                dim,
            },
        )
    }

    fn linear_forward(&self, input: &Tensor, weight: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "linear_forward";
        let x = self.f32(OP, input)?;
        let w = self.f32(OP, weight)?;
        let (rows, kin, nout, y) = linear_dims(OP, input.shape(), weight.shape())?;
        u32_dim(OP, product(OP, &y)?)?;
        self.one(
            OP,
            &y,
            0,
            Cmd::Linear {
                x,
                w,
                rows,
                kin,
                nout,
            },
        )
    }

    fn linear_backward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        const OP: &str = "linear_backward";
        let x = self.f32(OP, input)?;
        let w = self.f32(OP, weight)?;
        let gy = self.f32(OP, grad_output)?;
        let (rows, kin, nout, y) = linear_dims(OP, input.shape(), weight.shape())?;
        same(OP, grad_output.shape(), &y)?;
        self.two(
            OP,
            input.shape(),
            weight.shape(),
            Cmd::LinearBwd {
                x,
                w,
                gy,
                rows,
                kin,
                nout,
            },
        )
    }

    fn rms_norm_forward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        eps: f32,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "rms_norm_forward";
        let x = self.f32(OP, input)?;
        let w = self.f32(OP, weight)?;
        let (rows, dim) = rms_dims(OP, input.shape(), weight.shape(), eps)?;
        self.one(
            OP,
            input.shape(),
            0,
            Cmd::Rms {
                x,
                w,
                rows,
                dim,
                eps,
            },
        )
    }

    fn rms_norm_backward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        grad_output: &Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor), OjasError> {
        const OP: &str = "rms_norm_backward";
        let x = self.f32(OP, input)?;
        let w = self.f32(OP, weight)?;
        let gy = self.f32(OP, grad_output)?;
        same(OP, input.shape(), grad_output.shape())?;
        let (rows, dim) = rms_dims(OP, input.shape(), weight.shape(), eps)?;
        let mut v = self
            .outputs(
                OP,
                &[input.shape(), weight.shape()],
                rows as usize + rms_w_chunks(rows) as usize * dim as usize,
                Cmd::RmsBwd {
                    x,
                    w,
                    gy,
                    rows,
                    dim,
                    eps,
                },
            )?
            .into_iter();
        match (v.next(), v.next()) {
            (Some(a), Some(b)) => Ok((a, b)),
            _ => Err(metal_err(format!("{OP}: missing outputs"))),
        }
    }

    fn rope_half_split_forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "rope_half_split_forward";
        rope(self, OP, x, cos, sin, false)
    }

    fn rope_half_split_backward(
        &self,
        grad_output: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "rope_half_split_backward";
        rope(self, OP, grad_output, cos, sin, true)
    }

    fn rms_qk_norm_forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        q_weight: &Tensor,
        k_weight: &Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor), OjasError> {
        let qn = self.rms_norm_forward(q, q_weight, eps)?;
        let kn = self.rms_norm_forward(k, k_weight, eps)?;
        Ok((qn, kn))
    }

    fn rms_qk_norm_backward(
        &self,
        q: &Tensor,
        k: &Tensor,
        q_weight: &Tensor,
        k_weight: &Tensor,
        grad_q: &Tensor,
        grad_k: &Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor), OjasError> {
        let (gq, gqw) = self.rms_norm_backward(q, q_weight, grad_q, eps)?;
        let (gk, gkw) = self.rms_norm_backward(k, k_weight, grad_k, eps)?;
        Ok((gq, gk, gqw, gkw))
    }

    fn causal_sdpa_forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "causal_sdpa_forward";
        let (qa, ka, va) = (self.f32(OP, q)?, self.f32(OP, k)?, self.f32(OP, v)?);
        let (bh, t, d) = sdpa_dims(OP, q.shape(), k.shape(), v.shape())?;
        self.one(
            OP,
            q.shape(),
            0,
            Cmd::Sdpa {
                q: qa,
                k: ka,
                v: va,
                bh,
                t,
                d,
            },
        )
    }

    fn causal_sdpa_backward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor), OjasError> {
        const OP: &str = "causal_sdpa_backward";
        let (qa, ka, va) = (self.f32(OP, q)?, self.f32(OP, k)?, self.f32(OP, v)?);
        let gy = self.f32(OP, grad_output)?;
        let (bh, t, d) = sdpa_dims(OP, q.shape(), k.shape(), v.shape())?;
        same(OP, grad_output.shape(), q.shape())?;
        let s = q.shape();
        let mut out = self
            .outputs(
                OP,
                &[s, s, s],
                2 * bh as usize * t as usize,
                Cmd::SdpaBwd {
                    q: qa,
                    k: ka,
                    v: va,
                    gy,
                    bh,
                    t,
                    d,
                },
            )?
            .into_iter();
        match (out.next(), out.next(), out.next()) {
            (Some(a), Some(b), Some(c)) => Ok((a, b, c)),
            _ => Err(metal_err(format!("{OP}: missing outputs"))),
        }
    }

    fn per_head_sigmoid_gate_forward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        attn_out: &Tensor,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "per_head_sigmoid_gate_forward";
        let (x, w, b, a) = (
            self.f32(OP, input)?,
            self.f32(OP, weight)?,
            self.f32(OP, bias)?,
            self.f32(OP, attn_out)?,
        );
        let g = gate_dims(
            OP,
            input.shape(),
            weight.shape(),
            bias.shape(),
            attn_out.shape(),
        )?;
        self.one(
            OP,
            attn_out.shape(),
            g.rows as usize * g.heads as usize,
            Cmd::Gate {
                x,
                w,
                b,
                attn: a,
                rows: g.rows,
                din: g.din,
                heads: g.heads,
                dh: g.dh,
            },
        )
    }

    fn per_head_sigmoid_gate_backward(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        attn_out: &Tensor,
        grad_output: &Tensor,
    ) -> Result<PerHeadGateGrad, OjasError> {
        const OP: &str = "per_head_sigmoid_gate_backward";
        let (x, w, b, a) = (
            self.f32(OP, input)?,
            self.f32(OP, weight)?,
            self.f32(OP, bias)?,
            self.f32(OP, attn_out)?,
        );
        let gy = self.f32(OP, grad_output)?;
        same(OP, attn_out.shape(), grad_output.shape())?;
        let g = gate_dims(
            OP,
            input.shape(),
            weight.shape(),
            bias.shape(),
            attn_out.shape(),
        )?;
        let mut out = self
            .outputs(
                OP,
                &[
                    input.shape(),
                    weight.shape(),
                    bias.shape(),
                    attn_out.shape(),
                ],
                2 * g.rows as usize * g.heads as usize,
                Cmd::GateBwd {
                    x,
                    w,
                    b,
                    attn: a,
                    gy,
                    rows: g.rows,
                    din: g.din,
                    heads: g.heads,
                    dh: g.dh,
                },
            )?
            .into_iter();
        match (out.next(), out.next(), out.next(), out.next()) {
            (Some(input), Some(weight), Some(bias), Some(attn_out)) => Ok(PerHeadGateGrad {
                input,
                weight,
                bias,
                attn_out,
            }),
            _ => Err(metal_err(format!("{OP}: missing outputs"))),
        }
    }

    fn value_residual_blend_forward(
        &self,
        value: &Tensor,
        value0: &Tensor,
        lambda: &Tensor,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "value_residual_blend_forward";
        let (v, v0, lam) = (
            self.f32(OP, value)?,
            self.f32(OP, value0)?,
            self.f32(OP, lambda)?,
        );
        if lam.n != 1 {
            return Err(shape(OP, "expected a scalar tensor"));
        }
        same(OP, value.shape(), value0.shape())?;
        self.one(OP, value.shape(), 0, Cmd::Vres { v, v0, lam })
    }

    fn value_residual_blend_backward(
        &self,
        value: &Tensor,
        value0: &Tensor,
        lambda: &Tensor,
        grad_output: &Tensor,
    ) -> Result<ValueResidualGrad, OjasError> {
        const OP: &str = "value_residual_blend_backward";
        let (v, v0, lam) = (
            self.f32(OP, value)?,
            self.f32(OP, value0)?,
            self.f32(OP, lambda)?,
        );
        let gy = self.f32(OP, grad_output)?;
        if lam.n != 1 {
            return Err(shape(OP, "value residual lambda must be a scalar"));
        }
        same(OP, value.shape(), value0.shape())?;
        same(OP, value.shape(), grad_output.shape())?;
        let mut out = self
            .outputs(
                OP,
                &[value.shape(), value0.shape(), lambda.shape()],
                v.n + 2048,
                Cmd::VresBwd { v, v0, lam, gy },
            )?
            .into_iter();
        match (out.next(), out.next(), out.next()) {
            (Some(value), Some(value0), Some(lambda)) => Ok(ValueResidualGrad {
                value,
                value0,
                lambda,
            }),
            _ => Err(metal_err(format!("{OP}: missing outputs"))),
        }
    }

    fn silu_forward(&self, input: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "silu_forward";
        let x = self.f32(OP, input)?;
        self.one(OP, input.shape(), 0, Cmd::Silu { x })
    }

    fn silu_backward(&self, input: &Tensor, grad_output: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "silu_backward";
        let x = self.f32(OP, input)?;
        let gy = self.f32(OP, grad_output)?;
        same(OP, input.shape(), grad_output.shape())?;
        self.one(OP, input.shape(), 0, Cmd::SiluBwd { x, gy })
    }

    fn mul_forward(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "mul_forward";
        let (aa, ba) = (self.f32(OP, a)?, self.f32(OP, b)?);
        same(OP, a.shape(), b.shape())?;
        self.one(OP, a.shape(), 0, Cmd::Mul { a: aa, b: ba })
    }

    fn mul_backward(
        &self,
        a: &Tensor,
        b: &Tensor,
        grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        const OP: &str = "mul_backward";
        let (aa, ba) = (self.f32(OP, a)?, self.f32(OP, b)?);
        let gy = self.f32(OP, grad_output)?;
        same(OP, a.shape(), b.shape())?;
        same(OP, a.shape(), grad_output.shape())?;
        self.two(OP, a.shape(), b.shape(), Cmd::MulBwd { a: aa, b: ba, gy })
    }

    fn residual_add_forward(&self, x: &Tensor, y: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "residual_add_forward";
        let (xa, ya) = (self.f32(OP, x)?, self.f32(OP, y)?);
        same(OP, x.shape(), y.shape())?;
        self.one(OP, x.shape(), 0, Cmd::Add { x: xa, y: ya })
    }

    fn residual_add_backward(
        &self,
        x: &Tensor,
        y: &Tensor,
        grad_output: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        const OP: &str = "residual_add_backward";
        let (xa, ya) = (self.f32(OP, x)?, self.f32(OP, y)?);
        let gy = self.f32(OP, grad_output)?;
        same(OP, x.shape(), y.shape())?;
        same(OP, x.shape(), grad_output.shape())?;
        self.two(OP, x.shape(), y.shape(), Cmd::AddBwd { x: xa, y: ya, gy })
    }

    fn cross_entropy_mean_forward(
        &self,
        logits: &Tensor,
        targets: &Tensor,
        ignore_index: Option<u32>,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "cross_entropy_mean_forward";
        ce(self, OP, logits, targets, ignore_index, false)
    }

    fn cross_entropy_mean_backward(
        &self,
        logits: &Tensor,
        targets: &Tensor,
        ignore_index: Option<u32>,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "cross_entropy_mean_backward";
        ce(self, OP, logits, targets, ignore_index, true)
    }

    fn clip_grad_norm(&self, grads: &mut [Tensor], max_norm: f32) -> Result<f32, OjasError> {
        const OP: &str = "clip_grad_norm";
        if grads.is_empty() {
            return Err(shape(OP, "empty tensor"));
        }
        let args = grads
            .iter()
            .map(|g| self.f32(OP, g))
            .collect::<Res<Vec<_>>>()?;
        let _stats = self.reserve(OP, 2 * args.len() + 2048)?;
        let norm = match self.link.call(Cmd::ClipNorm {
            grads: args.clone(),
        })? {
            Reply::Norm(n) => n,
            other => return Err(metal_err(format!("{OP}: device returned {other:?}"))),
        };
        if !norm.is_finite() {
            return Err(OjasError::NonFinite { op: OP });
        }
        let scale = clip_scale(max_norm, norm)?;
        if scale < 1.0 {
            for g in grads.iter_mut() {
                self.unique(OP, g)?;
            }
            match self.link.call(Cmd::Scale { grads: args, scale })? {
                Reply::Done => {}
                other => return Err(metal_err(format!("{OP}: device returned {other:?}"))),
            }
        }
        Ok(norm)
    }

    fn adamw_step(
        &self,
        param: &mut Tensor,
        grad: &Tensor,
        moment1: &mut Tensor,
        moment2: &mut Tensor,
        step: u64,
        config: AdamWConfig,
    ) -> Result<(), OjasError> {
        const OP: &str = "adamw_step";
        let (p, g, m, v) = (
            self.f32(OP, param)?,
            self.f32(OP, grad)?,
            self.f32(OP, moment1)?,
            self.f32(OP, moment2)?,
        );
        same(OP, param.shape(), grad.shape())?;
        same(OP, param.shape(), moment1.shape())?;
        same(OP, param.shape(), moment2.shape())?;
        let (_, bc1, bc2) = check_adamw(config, step)?;
        let scalars = adamw_scalars(config, bc1, bc2)?;
        self.unique(OP, param)?;
        self.unique(OP, moment1)?;
        self.unique(OP, moment2)?;
        // In place with no scratch: the device checks the whole step, then
        // writes only if it is finite (see `ojas_adamw_check`).
        match self.link.call(Cmd::AdamW {
            p,
            g,
            m,
            v,
            scalars,
        })? {
            Reply::Done => Ok(()),
            other => Err(metal_err(format!("{OP}: device returned {other:?}"))),
        }
    }

    fn muon_ns5_step(
        &self,
        param: &mut Tensor,
        grad: &Tensor,
        momentum: &mut Tensor,
        config: MuonNs5Config,
    ) -> Result<(), OjasError> {
        const OP: &str = "muon_ns5_step";
        let (p, g, m) = (
            self.f32(OP, param)?,
            self.f32(OP, grad)?,
            self.f32(OP, momentum)?,
        );
        if param.shape().len() != 2 {
            return Err(shape(OP, "muon parameter must be a matrix"));
        }
        same(OP, param.shape(), grad.shape())?;
        same(OP, param.shape(), momentum.shape())?;
        ojas_core::require_ns5(5)?;
        if !config.lr.is_finite()
            || !config.momentum.is_finite()
            || !config.weight_decay.is_finite()
        {
            return Err(OjasError::NonFinite { op: OP });
        }
        if config.lr < 0.0 || config.momentum < 0.0 || config.weight_decay < 0.0 {
            return Err(OjasError::OutOfRange {
                op: OP,
                detail: "muon lr, momentum and weight_decay must be >= 0".to_string(),
            });
        }
        let (rows, cols) = (param.shape()[0], param.shape()[1]);
        let r = rows.min(cols);
        let scale = (rows as f64 / cols as f64).max(1.0).sqrt();
        let alpha = (-config.lr * scale) as f32;
        let decay = if config.weight_decay == 0.0 {
            1.0
        } else {
            (1.0 - config.lr * config.weight_decay) as f32
        };
        self.unique(OP, param)?;
        self.unique(OP, momentum)?;
        let scratch = 6 * p.n + 3 * r * r + 2048;
        let _scratch = self.reserve(OP, scratch)?;
        match self.link.call(Cmd::Muon {
            p,
            g,
            m,
            rows,
            cols,
            momentum: config.momentum as f32,
            nesterov: config.nesterov,
            decay,
            alpha,
        })? {
            Reply::Done => Ok(()),
            other => Err(metal_err(format!("{OP}: device returned {other:?}"))),
        }
    }
}

fn rope(
    be: &MetalBackend,
    op: &'static str,
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    backward: bool,
) -> Res<Tensor> {
    let (xa, ca, sa) = (be.f32(op, x)?, be.f32(op, cos)?, be.f32(op, sin)?);
    let (rows, dim, mode) = rope_dims(op, x.shape(), cos.shape(), sin.shape())?;
    be.one(
        op,
        x.shape(),
        0,
        Cmd::Rope {
            x: xa,
            cos: ca,
            sin: sa,
            rows,
            dim,
            mode,
            backward,
        },
    )
}

fn ce(
    be: &MetalBackend,
    op: &'static str,
    logits: &Tensor,
    targets: &Tensor,
    ignore: Option<u32>,
    grad: bool,
) -> Res<Tensor> {
    let la = be.f32(op, logits)?;
    let ta = be.arg(op, targets, DType::U32)?;
    let (rows, vocab) = ce_dims(op, logits.shape(), targets.shape())?;
    let out: &[usize] = if grad { logits.shape() } else { &[] };
    be.one(
        op,
        out,
        rows as usize + 2048,
        Cmd::Ce {
            logits: la,
            targets: ta,
            rows,
            vocab,
            ignore,
            grad,
        },
    )
}

#[cfg(all(test, target_os = "macos", feature = "metal"))]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    use super::*;

    /// Run `f` on its own thread; panic if it neither returns nor panics in
    /// time, so a hang is a failure rather than a stuck test run.
    fn watchdog<T: Send + 'static>(secs: u64, f: impl FnOnce() -> T + Send + 'static) -> T {
        let (tx, rx) = mpsc::channel();
        let h = thread::spawn(move || {
            let out = f();
            let _ = tx.send(());
            out
        });
        match rx.recv_timeout(Duration::from_secs(secs)) {
            Ok(()) => h.join().expect("worker panicked"),
            Err(mpsc::RecvTimeoutError::Disconnected) => match h.join() {
                Ok(_) => unreachable!("sender dropped without sending"),
                Err(p) => std::panic::resume_unwind(p),
            },
            Err(mpsc::RecvTimeoutError::Timeout) => {
                panic!("watchdog: no result after {secs} s; a call hung")
            }
        }
    }

    /// A panic on the device thread in the middle of a stream of ops from
    /// several threads poisons the backend: the panicking call and every
    /// later call fail promptly with an error (none hangs, none returns a
    /// result), and dropping the tensors and the backend completes.
    #[test]
    fn a_device_thread_panic_mid_stream_fails_later_calls_cleanly() {
        watchdog(120, || {
            let host = Budget::new(1 << 30);
            let m = MetalBackend::new(Budget::new(1 << 30)).expect("Metal device");
            let vals: Vec<f32> = (0..64 * 64).map(|i| (i % 17) as f32 / 17.0).collect();
            let x = m
                .upload(&Tensor::from_f32(&vals, &[64, 64], &host).expect("host"))
                .expect("upload");
            let done = Arc::new(AtomicUsize::new(0));
            let workers: Vec<_> = (0..6)
                .map(|i| {
                    let (m, x, done) = (m.clone(), x.clone(), Arc::clone(&done));
                    thread::spawn(move || {
                        let mut ok_before = 0usize;
                        loop {
                            let r = if i % 2 == 0 {
                                m.linear_forward(&x, &x).map(drop)
                            } else {
                                m.rms_norm_forward(&x, &x.view(&[64], &[1], 0).expect("w"), 1e-6)
                                    .map(drop)
                            };
                            match r {
                                Ok(()) => {
                                    ok_before += 1;
                                    done.fetch_add(1, Ordering::AcqRel);
                                }
                                Err(e) => return (ok_before, e),
                            }
                        }
                    })
                })
                .collect();
            while done.load(Ordering::Acquire) < 60 {
                thread::yield_now();
            }
            let injected = m.link.call(Cmd::InjectPanic);
            assert!(
                matches!(&injected, Err(OjasError::Backend { detail, .. }) if detail.contains("panicked")),
                "{injected:?}"
            );
            for (i, w) in workers.into_iter().enumerate() {
                let (ok_before, err) = w.join().expect("worker thread");
                assert!(ok_before > 0, "worker {i} never ran before the panic");
                assert!(matches!(err, OjasError::Poisoned), "worker {i}: {err:?}");
            }
            assert!(matches!(m.silu_forward(&x), Err(OjasError::Poisoned)));
            assert!(matches!(m.download(&x), Err(OjasError::Poisoned)));
            let fresh = Tensor::from_f32(&[1.0], &[1], &host).expect("host");
            assert!(matches!(m.upload(&fresh), Err(OjasError::Poisoned)));
            drop(x);
            drop(m);
        });
    }
}

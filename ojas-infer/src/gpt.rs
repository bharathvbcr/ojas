use crate::kernels::{attend_one, linear_bytes};
use crate::sample::{sample_token, GenerateConfig, SplitMix64};
use ojas_core::{Backend, Budget, DType, Numerics, OjasError, Scratch, Tensor, RMS_NORM_EPS};
use ojas_cpu::CpuBackend;

/// Nanolab's default attention block, for inference.
///
/// Per block, as `nanolab/mixers.py` `Attention.forward` and
/// `nanolab/model.py` `Block` / `SwiGLU` compute it:
///
/// 1. `h = RMSNorm(x)` (eps [`RMS_NORM_EPS`]).
/// 2. `q, k, v = h Wq^T, h Wk^T, h Wv^T`, no bias, split into heads.
/// 3. RMS QK-norm over `head_dim` (one weight shared by every head), then
///    half-split RoPE at the token's absolute position. QK-norm comes first.
/// 4. Value residual: layer 0 keeps its `v` and publishes it as `v0`; every
///    later layer uses `(1 - s) v + s v0`, `s = sigmoid(lambda)`. The cache
///    holds the blended values.
/// 5. Causal attention, scale `1 / sqrt(head_dim)`. Query head `i` reads KV
///    head `i / (n_head / n_kv_head)` (GQA).
/// 6. Per-head gate `sigmoid(h W_g^T + b_g)` multiplied onto each head's
///    output. The gate reads the normed `h`, not the residual `x`.
/// 7. `x += y Wo^T`, then `x += down(silu(h2 W_gate^T) * (h2 W_up^T))` with
///    `h2 = RMSNorm(x)`.
///
/// The head is the tied embedding after a final RMSNorm.
///
/// [`CpuGpt::forward_token`] runs one token against a [`KvCache`] with
/// local one-row linears and single-query attention: the trait linear
/// copies and repacks the whole weight per call. RMSNorm, QK-norm, RoPE,
/// the gate, the value residual, SiLU and the product run on
/// [`CpuBackend`]. [`CpuGpt::forward_sequence`] runs a whole sequence
/// through the [`Backend`] ops the trainer uses, including causal SDPA, so
/// cached decode can be checked against the training kernels.
pub struct CpuGpt {
    vocab: usize,
    n_embd: usize,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    hidden: usize,
    max_seq: usize,
    rope_base: f64,
    eps: f32,
    cpu: CpuBackend,
    /// `[vocab, n_embd]`. A clone of the caller's tensor: the bytes stay
    /// charged on the budget that allocated them, including the tied head.
    tok_emb: Tensor,
    blocks: Vec<BlockWeights>,
    ln_f: Tensor,
}

/// Checks host f32 layout and returns a view of the same allocation.
fn host_f32(tensor: &Tensor, shape: &[usize]) -> Result<Tensor, OjasError> {
    if tensor.dtype() != DType::F32 {
        return Err(OjasError::Dtype {
            op: "host_f32",
            expected: DType::F32,
            got: tensor.dtype(),
        });
    }
    if tensor.shape() != shape {
        return Err(OjasError::Shape {
            op: "host_f32",
            detail: format!("shape {:?} != {:?}", tensor.shape(), shape),
        });
    }
    let bytes = tensor.contiguous_bytes()?;
    if bytes.len() % 4 != 0 {
        return Err(OjasError::Shape {
            op: "host_f32",
            detail: "f32 storage is not a multiple of 4".into(),
        });
    }
    Ok(tensor.clone())
}

/// Shape of the decoder. `hidden` is the SwiGLU width.
///
/// `head_dim` is its own field, as in nanolab's `Config`: `n_head *
/// head_dim` need not equal `n_embd`. `n_kv_head == n_head` is plain
/// multi-head attention. `rope_base` is nanolab's `rope_base` (10000.0 by
/// default).
#[derive(Clone, Debug)]
pub struct GptConfig {
    pub vocab: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub n_layer: usize,
    pub hidden: usize,
    pub max_seq: usize,
    pub rope_base: f64,
}

/// One block's f32 weights. Linear matrices are `[out, in]`, `y = x @ W^T`.
///
/// With `q = n_head * head_dim` and `kv = n_kv_head * head_dim`:
/// `wq [q, d]`, `wk [kv, d]`, `wv [kv, d]`, `q_norm [head_dim]`,
/// `k_norm [head_dim]`, `gate_w [n_head, d]`, `gate_b [n_head]`,
/// `vr_lambda [1]`, `wo [d, q]`, `w_gate [hidden, d]`, `w_up [hidden, d]`,
/// `w_down [d, hidden]`. Layer 0's `vr_lambda` is present, as in nanolab,
/// and unused: layer 0 has no earlier values to blend.
#[derive(Clone)]
pub struct BlockWeights {
    pub ln1: Tensor,
    pub wq: Tensor,
    pub wk: Tensor,
    pub wv: Tensor,
    pub q_norm: Tensor,
    pub k_norm: Tensor,
    pub gate_w: Tensor,
    pub gate_b: Tensor,
    pub vr_lambda: Tensor,
    pub wo: Tensor,
    pub ln2: Tensor,
    pub w_gate: Tensor,
    pub w_up: Tensor,
    pub w_down: Tensor,
}

pub struct GptWeights {
    pub tok_emb: Tensor,
    pub blocks: Vec<BlockWeights>,
    pub ln_f: Tensor,
}

/// Per-layer keys and values. `len` is how many positions are stored, and
/// the next token goes at absolute position `len`.
///
/// Storage is one allocation charged to the [`Budget`] for this value's
/// lifetime: `2 * n_layer * width * max_len` f32s, where `width` is the
/// model's `n_kv_head * head_dim`. A refusal reports that request against
/// the budget's cap and live bytes. Appending when `len == max_len` returns
/// [`OjasError::CapacityExceeded`] and does not clamp or overwrite the last
/// position.
pub struct KvCache {
    width: usize,
    n_layer: usize,
    max_len: usize,
    len: usize,
    /// Keys, then values. Each side is `n_layer` rows of `max_len * width`.
    storage: Scratch<f32>,
}

impl KvCache {
    /// `width` is the per-position key width, `n_kv_head * head_dim`.
    pub fn new(
        n_layer: usize,
        width: usize,
        max_len: usize,
        budget: &Budget,
    ) -> Result<Self, OjasError> {
        let overflow = || OjasError::OutOfRange {
            op: "KvCache::new",
            detail: "cache length overflows".into(),
        };
        let cells = width.checked_mul(max_len).ok_or_else(overflow)?;
        let side = cells.checked_mul(n_layer).ok_or_else(overflow)?;
        let elems = side.checked_mul(2).ok_or_else(overflow)?;
        let storage = Scratch::try_alloc(elems, budget)?;
        Ok(Self {
            width,
            n_layer,
            max_len,
            len: 0,
            storage,
        })
    }

    /// A cache shaped for `model`, holding up to `max_len` positions.
    pub fn for_model(model: &CpuGpt, max_len: usize, budget: &Budget) -> Result<Self, OjasError> {
        Self::new(model.blocks.len(), model.kv_width(), max_len, budget)
    }

    /// Key and value rows for one layer, each `[max_len, width]`.
    fn layer_kv(&mut self, layer: usize) -> (&mut [f32], &mut [f32]) {
        let cells = self.width * self.max_len;
        let side = self.n_layer * cells;
        let start = layer * cells;
        let end = start + cells;
        let (keys, values) = self.storage.as_mut_slice().split_at_mut(side);
        (&mut keys[start..end], &mut values[start..end])
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn max_len(&self) -> usize {
        self.max_len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Positions still free.
    pub fn remaining(&self) -> usize {
        self.max_len - self.len
    }

    fn push_slot(&mut self) -> Result<usize, OjasError> {
        if self.len >= self.max_len {
            return Err(self.refusal(1));
        }
        Ok(self.len)
    }

    /// The error for a request of `extra` more positions than fit.
    pub(crate) fn refusal(&self, extra: usize) -> OjasError {
        let width = u64::try_from(self.width)
            .ok()
            .and_then(|n| n.checked_mul(4))
            .unwrap_or(u64::MAX);
        let live = (self.len as u64).saturating_mul(width);
        OjasError::CapacityExceeded {
            requested: live.saturating_add((extra as u64).saturating_mul(width)),
            cap: (self.max_len as u64).saturating_mul(width),
            live,
        }
    }
}

/// `[B, T, H, D]` row-major to `[B, H, T, D]` for `B == 1`, repeating each
/// head `rep` times (torch `repeat_interleave(rep, dim=1)`).
fn to_heads(x: &[f32], time: usize, heads: usize, dim: usize, rep: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(time * heads * rep * dim);
    for head in 0..heads {
        for _ in 0..rep {
            for t in 0..time {
                let at = (t * heads + head) * dim;
                out.extend_from_slice(&x[at..at + dim]);
            }
        }
    }
    out
}

/// `[1, H, T, D]` row-major back to `[1, T, H, D]`.
fn from_heads(y: &[f32], time: usize, heads: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; time * heads * dim];
    for head in 0..heads {
        for t in 0..time {
            let src = (head * time + t) * dim;
            let dst = (t * heads + head) * dim;
            out[dst..dst + dim].copy_from_slice(&y[src..src + dim]);
        }
    }
    out
}

/// The same contiguous allocation under another shape.
fn reshaped(tensor: &Tensor, shape: &[usize]) -> Result<Tensor, OjasError> {
    let want = shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| OjasError::OutOfRange {
            op: "reshaped",
            detail: "shape product overflows".into(),
        })?;
    if !tensor.is_contiguous()? || tensor.num_elements()? != want {
        return Err(OjasError::Shape {
            op: "reshaped",
            detail: format!("cannot view {:?} as {shape:?}", tensor.shape()),
        });
    }
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    tensor.view(shape, &strides, tensor.byte_offset())
}

fn add_into(x: &mut [f32], y: &[f32]) -> Result<(), OjasError> {
    if x.len() != y.len() {
        return Err(OjasError::Shape {
            op: "residual_add",
            detail: format!("{} != {}", x.len(), y.len()),
        });
    }
    for (a, b) in x.iter_mut().zip(y) {
        *a += b;
        if !a.is_finite() {
            return Err(OjasError::NonFinite { op: "residual_add" });
        }
    }
    Ok(())
}

impl CpuGpt {
    pub fn new(cfg: &GptConfig, weights: &GptWeights) -> Result<Self, OjasError> {
        if cfg.vocab == 0
            || cfg.n_embd == 0
            || cfg.n_head == 0
            || cfg.n_kv_head == 0
            || cfg.head_dim == 0
            || cfg.hidden == 0
            || cfg.max_seq == 0
        {
            return Err(OjasError::OutOfRange {
                op: "CpuGpt::new",
                detail: "vocab, n_embd, n_head, n_kv_head, head_dim, hidden, and max_seq \
                         must be non-zero"
                    .into(),
            });
        }
        if u32::try_from(cfg.vocab).is_err() || u32::try_from(cfg.head_dim).is_err() {
            return Err(OjasError::OutOfRange {
                op: "CpuGpt::new",
                detail: "vocab and head_dim must fit u32".into(),
            });
        }
        if !cfg.n_head.is_multiple_of(cfg.n_kv_head) {
            return Err(OjasError::Shape {
                op: "CpuGpt::new",
                detail: format!(
                    "n_head {} is not a multiple of n_kv_head {}",
                    cfg.n_head, cfg.n_kv_head
                ),
            });
        }
        if !cfg.head_dim.is_multiple_of(2) {
            return Err(OjasError::Shape {
                op: "CpuGpt::new",
                detail: format!("half-split RoPE needs an even head_dim, got {}", cfg.head_dim),
            });
        }
        if !(cfg.rope_base.is_finite() && cfg.rope_base > 0.0) {
            return Err(OjasError::OutOfRange {
                op: "CpuGpt::new",
                detail: format!("rope_base {} is not a positive finite number", cfg.rope_base),
            });
        }
        if weights.blocks.len() != cfg.n_layer {
            return Err(OjasError::Shape {
                op: "CpuGpt::new",
                detail: format!(
                    "block count {} != n_layer {}",
                    weights.blocks.len(),
                    cfg.n_layer
                ),
            });
        }
        let overflow = || OjasError::OutOfRange {
            op: "CpuGpt::new",
            detail: "head width overflows".into(),
        };
        let qw = cfg.n_head.checked_mul(cfg.head_dim).ok_or_else(overflow)?;
        let kvw = cfg
            .n_kv_head
            .checked_mul(cfg.head_dim)
            .ok_or_else(overflow)?;
        let (d, h, dh, nh) = (cfg.n_embd, cfg.hidden, cfg.head_dim, cfg.n_head);
        let tok_emb = host_f32(&weights.tok_emb, &[cfg.vocab, d])?;
        let ln_f = host_f32(&weights.ln_f, &[d])?;
        let mut blocks = Vec::with_capacity(cfg.n_layer);
        for (i, b) in weights.blocks.iter().enumerate() {
            let tag = |e| tag_block(i, e);
            blocks.push(BlockWeights {
                ln1: host_f32(&b.ln1, &[d]).map_err(tag)?,
                wq: host_f32(&b.wq, &[qw, d]).map_err(tag)?,
                wk: host_f32(&b.wk, &[kvw, d]).map_err(tag)?,
                wv: host_f32(&b.wv, &[kvw, d]).map_err(tag)?,
                q_norm: host_f32(&b.q_norm, &[dh]).map_err(tag)?,
                k_norm: host_f32(&b.k_norm, &[dh]).map_err(tag)?,
                gate_w: host_f32(&b.gate_w, &[nh, d]).map_err(tag)?,
                gate_b: host_f32(&b.gate_b, &[nh]).map_err(tag)?,
                vr_lambda: host_f32(&b.vr_lambda, &[1]).map_err(tag)?,
                wo: host_f32(&b.wo, &[d, qw]).map_err(tag)?,
                ln2: host_f32(&b.ln2, &[d]).map_err(tag)?,
                w_gate: host_f32(&b.w_gate, &[h, d]).map_err(tag)?,
                w_up: host_f32(&b.w_up, &[h, d]).map_err(tag)?,
                w_down: host_f32(&b.w_down, &[d, h]).map_err(tag)?,
            });
        }
        // Backend outputs per token are a few rows of the widest activation.
        let widest = d.max(h).max(qw);
        let scratch = u64::try_from(widest)
            .ok()
            .and_then(|w| w.checked_mul(4 * 16))
            .ok_or_else(|| OjasError::OutOfRange {
                op: "CpuGpt::new",
                detail: "activation scratch overflows".into(),
            })?
            .max(1 << 20);
        Ok(Self {
            vocab: cfg.vocab,
            n_embd: d,
            n_head: nh,
            n_kv_head: cfg.n_kv_head,
            head_dim: dh,
            hidden: h,
            max_seq: cfg.max_seq,
            rope_base: cfg.rope_base,
            eps: RMS_NORM_EPS,
            cpu: CpuBackend::new(Budget::new(scratch)),
            tok_emb,
            blocks,
            ln_f,
        })
    }

    /// Same model, with every [`CpuBackend`] op it runs under `numerics`.
    /// The default is [`CpuBackend`]'s, [`Numerics::Fast`]. The one-row
    /// linears and single-query attention of [`CpuGpt::forward_token`] are
    /// this crate's own ascending-order kernels under either setting, so
    /// cached decode and [`CpuGpt::forward_sequence`] agree to a tolerance,
    /// never bit for bit.
    pub fn with_numerics(mut self, numerics: Numerics) -> Self {
        self.cpu = self.cpu.with_numerics(numerics);
        self
    }

    /// Arithmetic contract of the [`CpuBackend`] ops this model runs.
    pub fn numerics(&self) -> Numerics {
        self.cpu.numerics()
    }

    fn q_width(&self) -> usize {
        self.n_head * self.head_dim
    }

    /// Per-position key (and value) width, `n_kv_head * head_dim`.
    pub fn kv_width(&self) -> usize {
        self.n_kv_head * self.head_dim
    }

    pub fn n_embd(&self) -> usize {
        self.n_embd
    }

    pub fn n_layer(&self) -> usize {
        self.blocks.len()
    }

    pub fn vocab(&self) -> usize {
        self.vocab
    }

    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    /// RoPE `cos` and `sin` rows for absolute positions `start..start+len`,
    /// each `[len, head_dim]`: `inv_i = base^(-2i / head_dim)`, angle
    /// `pos * inv_i`, then `cat(freqs, freqs)` as nanolab's
    /// `build_rope_cache`. Angles are computed in f64 and rounded once, as
    /// `ojas-autograd`'s `rope_cache` does. nanolab builds the table in f32,
    /// so its values differ in the last bits and the gap grows with `pos`.
    fn rope_rows(&self, start: usize, len: usize) -> (Vec<f32>, Vec<f32>) {
        let dim = self.head_dim;
        let half = dim / 2;
        let mut cos = vec![0.0f32; len * dim];
        let mut sin = vec![0.0f32; len * dim];
        for row in 0..len {
            let pos = (start + row) as f64;
            for i in 0..half {
                let inv = self.rope_base.powf(-((2 * i) as f64) / dim as f64);
                let angle = pos * inv;
                let (s, c) = angle.sin_cos();
                let at = row * dim;
                cos[at + i] = c as f32;
                cos[at + half + i] = c as f32;
                sin[at + i] = s as f32;
                sin[at + half + i] = s as f32;
            }
        }
        (cos, sin)
    }

    fn apply_linear(
        x: &[f32],
        weight: &Tensor,
        in_dim: usize,
        out_dim: usize,
        y: &mut [f32],
    ) -> Result<(), OjasError> {
        linear_bytes(x, weight.contiguous_bytes()?, in_dim, out_dim, y)
    }

    fn row(&self, data: &[f32], shape: &[usize]) -> Result<Tensor, OjasError> {
        Tensor::from_f32(data, shape, Backend::budget(&self.cpu))
    }

    /// `embedding_forward` copies its whole table on every call, so it is
    /// given the one-row view for `token` and looks up row 0.
    fn embed_token(&self, token: u32) -> Result<Vec<f32>, OjasError> {
        let row = self.check_token("CpuGpt::embed_token", token)?;
        let offset = row
            .checked_mul(self.n_embd)
            .and_then(|elems| elems.checked_mul(4))
            .ok_or_else(|| OjasError::OutOfRange {
                op: "CpuGpt::embed_token",
                detail: "row offset overflows".into(),
            })?;
        let table = self
            .tok_emb
            .narrow(offset, &[1, self.n_embd], &[self.n_embd, 1])?;
        let ids = Tensor::from_u32(&[0], &[1], Backend::budget(&self.cpu))?;
        self.cpu.embedding_forward(&table, &ids)?.to_f32_vec()
    }

    fn check_token(&self, op: &'static str, token: u32) -> Result<usize, OjasError> {
        usize::try_from(token)
            .ok()
            .filter(|&row| row < self.vocab)
            .ok_or_else(|| OjasError::OutOfRange {
                op,
                detail: format!("token id {token} >= vocab {}", self.vocab),
            })
    }

    fn rms_norm(&self, x: &[f32], weight: &Tensor) -> Result<Vec<f32>, OjasError> {
        let row = self.row(x, &[1, self.n_embd])?;
        self.cpu
            .rms_norm_forward(&row, weight, self.eps)?
            .to_f32_vec()
    }

    fn check_cache(&self, cache: &KvCache) -> Result<(), OjasError> {
        if cache.width != self.kv_width()
            || cache.n_layer != self.blocks.len()
            || cache.max_len > self.max_seq
        {
            return Err(OjasError::Shape {
                op: "CpuGpt::forward_token",
                detail: format!(
                    "cache (width {}, layers {}, max_len {}) does not match this model \
                     (kv width {}, layers {}, max_seq {})",
                    cache.width,
                    cache.n_layer,
                    cache.max_len,
                    self.kv_width(),
                    self.blocks.len(),
                    self.max_seq
                ),
            });
        }
        Ok(())
    }

    /// Forward one token at absolute position `cache.len()`, appending its
    /// keys and values. Linear outputs and residual sums are checked
    /// finite, so the returned logits are finite. A cache longer than the
    /// model's `max_seq` is refused before any position is written; a
    /// failure part-way leaves `cache.len()` unchanged.
    pub fn forward_token(&self, token: u32, cache: &mut KvCache) -> Result<Vec<f32>, OjasError> {
        self.check_cache(cache)?;
        let pos = cache.push_slot()?;
        let (d, qw, kvw, dh) = (self.n_embd, self.q_width(), self.kv_width(), self.head_dim);
        let mut x = self.embed_token(token)?;
        let (cos, sin) = self.rope_rows(pos, 1);
        let cos = self.row(&cos, &[1, dh])?;
        let sin = self.row(&sin, &[1, dh])?;
        let mut q = vec![0.0f32; qw];
        let mut k = vec![0.0f32; kvw];
        let mut v = vec![0.0f32; kvw];
        let mut proj = vec![0.0f32; d];
        let mut gate = vec![0.0f32; self.hidden];
        let mut up = vec![0.0f32; self.hidden];
        let mut v0: Option<Vec<f32>> = None;
        for (layer, b) in self.blocks.iter().enumerate() {
            let h = self.rms_norm(&x, &b.ln1)?;
            Self::apply_linear(&h, &b.wq, d, qw, &mut q)?;
            Self::apply_linear(&h, &b.wk, d, kvw, &mut k)?;
            Self::apply_linear(&h, &b.wv, d, kvw, &mut v)?;
            let (qn, kn) = self.cpu.rms_qk_norm_forward(
                &self.row(&q, &[1, 1, self.n_head, dh])?,
                &self.row(&k, &[1, 1, self.n_kv_head, dh])?,
                &b.q_norm,
                &b.k_norm,
                self.eps,
            )?;
            let qr = self
                .cpu
                .rope_half_split_forward(&qn, &cos, &sin)?
                .to_f32_vec()?;
            let kr = self
                .cpu
                .rope_half_split_forward(&kn, &cos, &sin)?
                .to_f32_vec()?;
            let vb = match &v0 {
                None => {
                    v0 = Some(v.clone());
                    v.clone()
                }
                Some(first) => self
                    .cpu
                    .value_residual_blend_forward(
                        &self.row(&v, &[1, kvw])?,
                        &self.row(first, &[1, kvw])?,
                        &b.vr_lambda,
                    )?
                    .to_f32_vec()?,
            };
            let base = pos * kvw;
            let seen = (pos + 1) * kvw;
            let (k_layer, v_layer) = cache.layer_kv(layer);
            k_layer[base..base + kvw].copy_from_slice(&kr);
            v_layer[base..base + kvw].copy_from_slice(&vb);
            let mixed = attend_one(
                &qr,
                &k_layer[..seen],
                &v_layer[..seen],
                self.n_head,
                self.n_kv_head,
                dh,
            )?;
            let gated = self
                .cpu
                .per_head_sigmoid_gate_forward(
                    &self.row(&h, &[1, d])?,
                    &b.gate_w,
                    &b.gate_b,
                    &self.row(&mixed, &[1, self.n_head, dh])?,
                )?
                .to_f32_vec()?;
            Self::apply_linear(&gated, &b.wo, qw, d, &mut proj)?;
            add_into(&mut x, &proj)?;
            let h2 = self.rms_norm(&x, &b.ln2)?;
            Self::apply_linear(&h2, &b.w_gate, d, self.hidden, &mut gate)?;
            Self::apply_linear(&h2, &b.w_up, d, self.hidden, &mut up)?;
            let act = self
                .cpu
                .silu_forward(&self.row(&gate, &[1, self.hidden])?)?;
            let hidden = self
                .cpu
                .mul_forward(&act, &self.row(&up, &[1, self.hidden])?)?
                .to_f32_vec()?;
            Self::apply_linear(&hidden, &b.w_down, self.hidden, d, &mut proj)?;
            add_into(&mut x, &proj)?;
        }
        cache.len = pos + 1;
        let norm = self.rms_norm(&x, &self.ln_f)?;
        let mut logits = vec![0.0f32; self.vocab];
        Self::apply_linear(&norm, &self.tok_emb, d, self.vocab, &mut logits)?;
        Ok(logits)
    }

    /// Logits for every position of `tokens`, `[len, vocab]` row-major,
    /// through the [`Backend`] ops the trainer uses: batched linears, causal
    /// SDPA over `[1, H, T, D]`, and the same RoPE, QK-norm, gate and value
    /// residual ops. Positions start at 0. Every output and intermediate is
    /// charged to `budget`. `tokens` must hold `1..=max_seq` ids.
    ///
    /// The `[1, T, H, D]` to `[1, H, T, D]` reorder and GQA's head repeat
    /// happen on the host. `CpuBackend::permute` landed while this was
    /// written and could take over the reorder; the repeat has no trait op.
    pub fn forward_sequence(&self, tokens: &[u32], budget: &Budget) -> Result<Vec<f32>, OjasError> {
        const OP: &str = "CpuGpt::forward_sequence";
        if tokens.is_empty() || tokens.len() > self.max_seq {
            return Err(OjasError::Shape {
                op: OP,
                detail: format!(
                    "sequence length {} outside 1..={}",
                    tokens.len(),
                    self.max_seq
                ),
            });
        }
        for &id in tokens {
            self.check_token(OP, id)?;
        }
        let cpu = CpuBackend::new(budget.clone()).with_numerics(self.numerics());
        let t = tokens.len();
        let (d, qw, dh) = (self.n_embd, self.q_width(), self.head_dim);
        let (nh, nkv) = (self.n_head, self.n_kv_head);
        let rep = nh / nkv;
        let ids = Tensor::from_u32(tokens, &[1, t], budget)?;
        let mut x = cpu.embedding_forward(&self.tok_emb, &ids)?.to_f32_vec()?;
        let (cos, sin) = self.rope_rows(0, t);
        let cos = Tensor::from_f32(&cos, &[t, dh], budget)?;
        let sin = Tensor::from_f32(&sin, &[t, dh], budget)?;
        let mut v0: Option<Tensor> = None;
        for b in &self.blocks {
            let xt = Tensor::from_f32(&x, &[1, t, d], budget)?;
            let h = cpu.rms_norm_forward(&xt, &b.ln1, self.eps)?;
            let q = reshaped(&cpu.linear_forward(&h, &b.wq)?, &[1, t, nh, dh])?;
            let k = reshaped(&cpu.linear_forward(&h, &b.wk)?, &[1, t, nkv, dh])?;
            let v = cpu.linear_forward(&h, &b.wv)?;
            let (qn, kn) = cpu.rms_qk_norm_forward(&q, &k, &b.q_norm, &b.k_norm, self.eps)?;
            let qr = cpu.rope_half_split_forward(&qn, &cos, &sin)?.to_f32_vec()?;
            let kr = cpu.rope_half_split_forward(&kn, &cos, &sin)?.to_f32_vec()?;
            let vb = match &v0 {
                None => {
                    v0 = Some(v.clone());
                    v
                }
                Some(first) => cpu.value_residual_blend_forward(&v, first, &b.vr_lambda)?,
            }
            .to_f32_vec()?;
            let heads = |data: &[f32], n: usize, r: usize| {
                Tensor::from_f32(&to_heads(data, t, n, dh, r), &[1, n * r, t, dh], budget)
            };
            let y = cpu.causal_sdpa_forward(
                &heads(&qr, nh, 1)?,
                &heads(&kr, nkv, rep)?,
                &heads(&vb, nkv, rep)?,
            )?;
            let y = Tensor::from_f32(
                &from_heads(&y.to_f32_vec()?, t, nh, dh),
                &[1, t, nh, dh],
                budget,
            )?;
            let gated = cpu.per_head_sigmoid_gate_forward(&h, &b.gate_w, &b.gate_b, &y)?;
            let proj = cpu.linear_forward(&reshaped(&gated, &[1, t, qw])?, &b.wo)?;
            add_into(&mut x, &proj.to_f32_vec()?)?;
            let xt = Tensor::from_f32(&x, &[1, t, d], budget)?;
            let h2 = cpu.rms_norm_forward(&xt, &b.ln2, self.eps)?;
            let act = cpu.silu_forward(&cpu.linear_forward(&h2, &b.w_gate)?)?;
            let hidden = cpu.mul_forward(&act, &cpu.linear_forward(&h2, &b.w_up)?)?;
            let down = cpu.linear_forward(&hidden, &b.w_down)?;
            add_into(&mut x, &down.to_f32_vec()?)?;
        }
        let xt = Tensor::from_f32(&x, &[1, t, d], budget)?;
        let norm = cpu.rms_norm_forward(&xt, &self.ln_f, self.eps)?;
        cpu.linear_forward(&norm, &self.tok_emb)?.to_f32_vec()
    }

    /// Greedy continuation through [`argmax_token`]: a non-finite logit is
    /// [`OjasError::NonFinite`], never token 0. Same cache contract as
    /// [`Self::generate`].
    pub fn greedy_decode(
        &self,
        prompt: &[u32],
        cache: &mut KvCache,
        new_tokens: usize,
    ) -> Result<Vec<u32>, OjasError> {
        self.decode(prompt, cache, new_tokens, &[], argmax_token)
    }

    /// Sampled continuation of `prompt` ([`sample_token`] with a
    /// [`SplitMix64`] seeded from `cfg.seed`).
    ///
    /// The prompt is forwarded from position `cache.len()`. Each emitted
    /// token except the last is forwarded, so the cache must have room for
    /// `prompt.len() + max_new_tokens - 1` more positions (`prompt.len()`
    /// when `max_new_tokens` is 0). That is checked before any forward: a
    /// request that does not fit is [`OjasError::CapacityExceeded`] and the
    /// cache is untouched. To continue later, pass the last returned token
    /// as the first token of the next prompt.
    pub fn generate(
        &self,
        prompt: &[u32],
        cache: &mut KvCache,
        cfg: &GenerateConfig,
    ) -> Result<Vec<u32>, OjasError> {
        cfg.sampling.validate()?;
        let mut rng = SplitMix64::new(cfg.seed);
        self.decode(prompt, cache, cfg.max_new_tokens, &cfg.stop_tokens, |logits| {
            sample_token(logits, &cfg.sampling, &mut rng)
        })
    }

    /// The one decode loop. `select` picks the next id from a logit row.
    fn decode(
        &self,
        prompt: &[u32],
        cache: &mut KvCache,
        new_tokens: usize,
        stop_tokens: &[u32],
        mut select: impl FnMut(&[f32]) -> Result<u32, OjasError>,
    ) -> Result<Vec<u32>, OjasError> {
        const OP: &str = "CpuGpt::decode";
        if prompt.is_empty() {
            return Err(OjasError::Shape {
                op: OP,
                detail: "prompt is empty".into(),
            });
        }
        self.check_cache(cache)?;
        for &id in prompt.iter().chain(stop_tokens) {
            self.check_token(OP, id)?;
        }
        let needed = prompt
            .len()
            .checked_add(new_tokens.saturating_sub(1))
            .ok_or_else(|| OjasError::OutOfRange {
                op: OP,
                detail: "prompt plus new tokens overflows".into(),
            })?;
        if needed > cache.remaining() {
            return Err(cache.refusal(needed));
        }
        let mut out = Vec::new();
        out.try_reserve_exact(new_tokens)
            .map_err(|_| OjasError::OutOfRange {
                op: OP,
                detail: format!("cannot allocate {new_tokens} output ids"),
            })?;
        let mut logits = Vec::new();
        for &id in prompt {
            logits = self.forward_token(id, cache)?;
        }
        for step in 0..new_tokens {
            let next = select(&logits)?;
            out.push(next);
            if step + 1 == new_tokens || stop_tokens.contains(&next) {
                break;
            }
            logits = self.forward_token(next, cache)?;
        }
        Ok(out)
    }
}

/// Index of the largest finite logit.
///
/// Any NaN or infinity, or an empty row, is an error. An all-NaN row does not
/// return token 0.
pub fn argmax_token(logits: &[f32]) -> Result<u32, OjasError> {
    if logits.is_empty() {
        return Err(OjasError::Shape {
            op: "argmax_token",
            detail: "empty logits".into(),
        });
    }
    if logits.iter().any(|v| !v.is_finite()) {
        return Err(OjasError::NonFinite { op: "argmax_token" });
    }
    let mut best_i = 0usize;
    let mut best_v = logits[0];
    for (i, &v) in logits.iter().enumerate().skip(1) {
        if v > best_v {
            best_i = i;
            best_v = v;
        }
    }
    u32::try_from(best_i).map_err(|_| OjasError::OutOfRange {
        op: "argmax_token",
        detail: "logit index exceeds u32".into(),
    })
}

fn tag_block(index: usize, err: OjasError) -> OjasError {
    match err {
        OjasError::Shape { op, detail } => OjasError::Shape {
            op,
            detail: format!("block {index}: {detail}"),
        },
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ojas_core::Budget;

    fn tensor(data: &[f32], shape: &[usize], budget: &Budget) -> Tensor {
        Tensor::from_f32(data, shape, budget).unwrap()
    }

    fn tiny_cfg() -> GptConfig {
        GptConfig {
            vocab: 2,
            n_embd: 2,
            n_head: 1,
            n_kv_head: 1,
            head_dim: 2,
            n_layer: 1,
            hidden: 1,
            max_seq: 4,
            rope_base: 10000.0,
        }
    }

    /// d = 2, one head of width 2. `wo` and the MLP are zero, so the logits
    /// are `emb . RMSNorm(emb[token])` and the larger embedding row wins.
    fn tiny_weights(budget: &Budget, emb: &[f32]) -> GptWeights {
        let eye = [1.0, 0.0, 0.0, 1.0];
        GptWeights {
            tok_emb: tensor(emb, &[2, 2], budget),
            ln_f: tensor(&[1.0, 1.0], &[2], budget),
            blocks: vec![BlockWeights {
                ln1: tensor(&[1.0, 1.0], &[2], budget),
                wq: tensor(&eye, &[2, 2], budget),
                wk: tensor(&eye, &[2, 2], budget),
                wv: tensor(&eye, &[2, 2], budget),
                q_norm: tensor(&[1.0, 1.0], &[2], budget),
                k_norm: tensor(&[1.0, 1.0], &[2], budget),
                gate_w: tensor(&[0.0, 0.0], &[1, 2], budget),
                gate_b: tensor(&[0.0], &[1], budget),
                vr_lambda: tensor(&[0.0], &[1], budget),
                wo: tensor(&[0.0; 4], &[2, 2], budget),
                ln2: tensor(&[1.0, 1.0], &[2], budget),
                w_gate: tensor(&[0.0, 0.0], &[1, 2], budget),
                w_up: tensor(&[0.0, 0.0], &[1, 2], budget),
                w_down: tensor(&[0.0, 0.0], &[2, 1], budget),
            }],
        }
    }

    const BIG_FIRST: [f32; 4] = [1.0, 0.0, 0.25, 0.0];

    fn tiny(budget: &Budget, emb: &[f32]) -> (CpuGpt, GptConfig) {
        let cfg = tiny_cfg();
        let model = CpuGpt::new(&cfg, &tiny_weights(budget, emb)).unwrap();
        (model, cfg)
    }

    #[test]
    fn all_nan_logits_are_an_error_not_token_zero() {
        let err = argmax_token(&[f32::NAN, f32::NAN]).unwrap_err();
        assert!(matches!(err, OjasError::NonFinite { .. }), "{err}");
        assert!(argmax_token(&[f32::NAN, 1.0]).is_err());
        assert!(argmax_token(&[f32::INFINITY, 1.0]).is_err());
        assert!(argmax_token(&[f32::NEG_INFINITY, f32::NEG_INFINITY]).is_err());
        assert_eq!(argmax_token(&[0.25, 0.5, 0.5]).unwrap(), 1);

        let budget = Budget::new(1 << 20);
        let (model, cfg) = tiny(&budget, &[f32::NAN, 0.0, 0.0, 0.0]);
        let mut cache = KvCache::for_model(&model, cfg.max_seq, &budget).unwrap();
        let decoded = model.greedy_decode(&[0], &mut cache, 1);
        assert!(decoded.is_err(), "NaN model returned {decoded:?}");
        assert!(!matches!(decoded, Ok(ref ids) if ids.first() == Some(&0)));
    }

    #[test]
    fn greedy_picks_the_larger_tied_embedding_and_cache_refuses_the_next_slot() {
        let budget = Budget::new(1 << 20);
        let (model, cfg) = tiny(&budget, &BIG_FIRST);
        let mut cache = KvCache::for_model(&model, cfg.max_seq, &budget).unwrap();
        let logits = model.forward_token(0, &mut cache).unwrap();
        assert!(logits.iter().all(|v| v.is_finite()), "{logits:?}");
        assert!(logits[0] > logits[1], "{logits:?}");
        assert_eq!(argmax_token(&logits).unwrap(), 0);
        assert_eq!(cache.len(), 1);

        // One slot holds a one-token prompt plus one new token (F17: the
        // emitted token is not forwarded). Two new tokens need a second
        // slot and are refused before any forward.
        let mut cache = KvCache::for_model(&model, 1, &budget).unwrap();
        assert_eq!(model.greedy_decode(&[0], &mut cache, 1).unwrap(), vec![0]);
        assert_eq!(cache.len(), 1);
        let mut cache = KvCache::for_model(&model, 1, &budget).unwrap();
        let err = model.greedy_decode(&[0], &mut cache, 2).unwrap_err();
        assert!(matches!(err, OjasError::CapacityExceeded { .. }), "{err}");
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn kv_cache_refuses_an_unallocatable_length_instead_of_panicking() {
        for (n_layer, width, max_len) in [
            (1, 1, usize::MAX / 2),
            (1, 1 << 20, 1 << 40),
            (usize::MAX, 1, 1),
        ] {
            let got = std::panic::catch_unwind(|| {
                let budget = Budget::new(u64::MAX);
                KvCache::new(n_layer, width, max_len, &budget)
            });
            let err = got
                .unwrap_or_else(|_| panic!("KvCache::new({n_layer}, {width}, {max_len}) panicked"))
                .err()
                .unwrap_or_else(|| panic!("KvCache::new({n_layer}, {width}, {max_len}) allocated"));
            assert!(
                matches!(
                    err,
                    OjasError::CapacityExceeded { .. } | OjasError::OutOfRange { .. }
                ),
                "{err}"
            );
        }
    }

    #[test]
    fn positions_past_max_seq_are_refused_and_max_seq_zero_is_invalid() {
        let budget = Budget::new(1 << 20);
        let (model, mut cfg) = tiny(&budget, &BIG_FIRST);
        let mut cache = KvCache::for_model(&model, cfg.max_seq, &budget).unwrap();
        for _ in 0..cfg.max_seq {
            model.forward_token(0, &mut cache).unwrap();
        }
        let mut long = KvCache::for_model(&model, cfg.max_seq + 1, &budget).unwrap();
        let err = model.forward_token(0, &mut long).unwrap_err();
        assert!(matches!(err, OjasError::Shape { .. }), "{err}");
        assert_eq!(long.len(), 0);

        cfg.max_seq = 0;
        let w = GptWeights {
            tok_emb: tensor(&BIG_FIRST, &[2, 2], &budget),
            ln_f: tensor(&[1.0, 1.0], &[2], &budget),
            blocks: Vec::new(),
        };
        cfg.n_layer = 0;
        assert!(CpuGpt::new(&cfg, &w).is_err());
    }

    /// Two layers, four heads, deterministic weights from a xorshift stream.
    fn seeded(budget: &Budget) -> (CpuGpt, GptConfig) {
        let cfg = GptConfig {
            vocab: 37,
            n_embd: 16,
            n_head: 4,
            n_kv_head: 4,
            head_dim: 4,
            n_layer: 2,
            hidden: 24,
            max_seq: 32,
            rope_base: 10000.0,
        };
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut fill = |n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    ((state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * scale
                })
                .collect()
        };
        let (d, h, dh) = (cfg.n_embd, cfg.hidden, cfg.head_dim);
        let ones = |v: Vec<f32>| v.iter().map(|v| v + 1.0).collect::<Vec<_>>();
        let tok_emb = tensor(&fill(cfg.vocab * d, 0.5), &[cfg.vocab, d], budget);
        let blocks = (0..cfg.n_layer)
            .map(|_| BlockWeights {
                ln1: tensor(&ones(fill(d, 0.5)), &[d], budget),
                wq: tensor(&fill(d * d, 1.6), &[d, d], budget),
                wk: tensor(&fill(d * d, 1.6), &[d, d], budget),
                wv: tensor(&fill(d * d, 1.6), &[d, d], budget),
                q_norm: tensor(&ones(fill(dh, 0.5)), &[dh], budget),
                k_norm: tensor(&ones(fill(dh, 0.5)), &[dh], budget),
                gate_w: tensor(&fill(cfg.n_head * d, 1.0), &[cfg.n_head, d], budget),
                gate_b: tensor(&fill(cfg.n_head, 1.0), &[cfg.n_head], budget),
                vr_lambda: tensor(&fill(1, 2.0), &[1], budget),
                wo: tensor(&fill(d * d, 1.6), &[d, d], budget),
                ln2: tensor(&ones(fill(d, 0.5)), &[d], budget),
                w_gate: tensor(&fill(h * d, 1.6), &[h, d], budget),
                w_up: tensor(&fill(h * d, 1.6), &[h, d], budget),
                w_down: tensor(&fill(d * h, 1.6), &[d, h], budget),
            })
            .collect();
        let ln_f = tensor(&ones(fill(d, 0.5)), &[d], budget);
        let w = GptWeights {
            tok_emb,
            blocks,
            ln_f,
        };
        (CpuGpt::new(&cfg, &w).unwrap(), cfg)
    }

    #[test]
    fn backend_rms_norm_refuses_non_finite_and_f32_overflowing_rows() {
        let budget = Budget::new(1 << 24);
        let (model, _) = seeded(&budget);
        let weight = model.blocks[0].ln1.clone();
        let ok = model.rms_norm(&[0.5; 16], &weight).unwrap();
        assert!(ok.iter().all(|v| v.is_finite()));
        let mut nan = [1.0f32; 16];
        nan[3] = f32::NAN;
        assert!(matches!(
            model.rms_norm(&nan, &weight),
            Err(OjasError::NonFinite { .. })
        ));
        // The f64 kernel this replaced normalized rows whose f32 sum of
        // squares overflows. ojas-cpu accumulates in f32 and refuses them.
        assert!(matches!(
            model.rms_norm(&[1.0e19; 16], &weight),
            Err(OjasError::NonFinite { .. })
        ));
        assert!(model.rms_norm(&[1.0; 15], &weight).is_err());
    }

    #[test]
    fn greedy_decode_leaves_the_last_token_out_of_the_cache() {
        let budget = Budget::new(1 << 20);
        let (model, cfg) = tiny(&budget, &BIG_FIRST);
        let prompt = [0u32];
        let new_tokens = 2usize;
        let slots = prompt.len() + new_tokens;
        assert!(slots <= cfg.max_seq);
        let mut cache = KvCache::for_model(&model, slots, &budget).unwrap();
        let out = model
            .greedy_decode(&prompt, &mut cache, new_tokens)
            .unwrap();
        assert_eq!(out.len(), new_tokens);
        assert_eq!(cache.len(), slots - 1);
    }

    /// F17: the last emitted token is returned, not forwarded, so `prompt +
    /// N - 1` positions are enough for `N` new tokens.
    #[test]
    fn greedy_decode_fits_prompt_plus_n_minus_one_slots() {
        let budget = Budget::new(1 << 20);
        let (model, _) = tiny(&budget, &BIG_FIRST);
        let prompt = [0u32, 1];
        let n = 3usize;
        let mut cache = KvCache::for_model(&model, prompt.len() + n - 1, &budget).unwrap();
        let out = model.greedy_decode(&prompt, &mut cache, n).unwrap();
        assert_eq!(out.len(), n);
        assert_eq!(cache.len(), prompt.len() + n - 1);
        assert_eq!(cache.remaining(), 0);
    }

    #[test]
    fn token_id_out_of_range_is_an_error() {
        let budget = Budget::new(1 << 20);
        let (model, _) = tiny(&budget, &BIG_FIRST);
        let mut cache = KvCache::for_model(&model, 4, &budget).unwrap();
        let err = model.forward_token(7, &mut cache).unwrap_err();
        assert!(matches!(err, OjasError::OutOfRange { .. }), "{err}");
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn a_cache_shaped_for_another_model_is_refused() {
        let budget = Budget::new(1 << 24);
        let (model, cfg) = tiny(&budget, &BIG_FIRST);
        let mut wrong = KvCache::new(cfg.n_layer, 1, cfg.max_seq, &budget).unwrap();
        let err = model.forward_token(0, &mut wrong).unwrap_err();
        assert!(matches!(err, OjasError::Shape { .. }), "{err}");
        assert_eq!(wrong.len(), 0);
    }

    #[test]
    fn kv_cache_refusal_reports_budget_bytes_and_releases_on_drop() {
        let budget = Budget::new(100);
        let held = budget.try_reserve(40).unwrap();
        // 2 * 1 layer * 4 * 4 f32s = 128 bytes, above the cap.
        let err = match KvCache::new(1, 4, 4, &budget) {
            Err(err) => err,
            Ok(_) => panic!("cache allocated under a budget that cannot hold it"),
        };
        assert!(
            matches!(
                err,
                OjasError::CapacityExceeded {
                    requested: 128,
                    cap: 100,
                    live: 40
                }
            ),
            "{err}"
        );
        assert_eq!(budget.live_bytes().unwrap(), 40);
        drop(held);

        let budget = Budget::new(128);
        let cache = KvCache::new(1, 4, 4, &budget).unwrap();
        assert_eq!(budget.live_bytes().unwrap(), 128);
        drop(cache);
        assert_eq!(budget.live_bytes().unwrap(), 0);
    }

    #[test]
    fn gpt_holds_views_of_weights_the_caller_already_charged() {
        // Measure what the weights cost, then build them again under a
        // budget of exactly that size.
        let probe = Budget::new(1 << 20);
        let sized = tiny_weights(&probe, &BIG_FIRST);
        let exact = probe.live_bytes().unwrap();
        drop(sized);
        let budget = Budget::new(exact);
        let cfg = tiny_cfg();
        let w = tiny_weights(&budget, &BIG_FIRST);
        let charged = budget.live_bytes().unwrap();
        assert_eq!(charged, budget.cap_bytes());
        let model = CpuGpt::new(&cfg, &w).unwrap();
        assert_eq!(budget.live_bytes().unwrap(), charged);
        drop(w);
        assert_eq!(budget.live_bytes().unwrap(), charged);
        let cache_budget = Budget::new(1 << 20);
        let mut cache = KvCache::for_model(&model, cfg.max_seq, &cache_budget).unwrap();
        let logits = model.forward_token(0, &mut cache).unwrap();
        assert!(logits.iter().all(|v| v.is_finite()), "{logits:?}");
        drop(model);
        assert_eq!(budget.live_bytes().unwrap(), 0);
    }
}

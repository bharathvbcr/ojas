//! The nanolab block, written once over [`Graph`], and the forward to the
//! fused head loss.
//!
//! [`block`] follows nanolab `Attention.forward` (`mixers.py:280-345`) and
//! `Block.forward` (`model.py:189-193`), the same order as `CpuGpt` in
//! `ojas-infer`:
//! 1. `h = norm1(x)`, then the q, k and v projections, viewed `[B, T, H, D]`.
//! 2. Per-head QK RMSNorm, then half-split RoPE, both in `[B, T, H, D]`.
//! 3. Value residual in `[B, T, H, D]`: every layer after the first blends
//!    its `v` with layer 0's raw `v` (`v0`, not detached).
//! 4. Attention, time-major `[B, T, H, D]` in and out: [`block_with`] takes
//!    it from the caller; [`block`] passes [`causal_attention`] (permute to
//!    `[B, H, T, D]`, causal SDPA, permute back; grouped-query through
//!    cached attention on `Eval`).
//! 5. Per-head gate (it reads `h`, not `x`), `o_proj`, residual.
//! 6. `norm2`, SwiGLU `down(silu(gate(h2)) * up(h2))`, residual.

use ojas_core::{Backend, Budget, CeChunk, DType, OjasError, Tensor};

use crate::graph::Graph;
use crate::names::{BlockParams, ModelParams};
use crate::spec::ModelSpec;

/// `[B, T, H, D]` to `[B, H, T, D]`; it is its own inverse.
const SWAP_TIME_HEADS: [usize; 4] = [0, 2, 1, 3];

/// RoPE `cos` and `sin` rows, each `[len, head_dim]`, for absolute positions
/// `start..start + len`.
#[derive(Clone, Debug)]
pub struct Rope {
    pub cos: Tensor,
    pub sin: Tensor,
    start: usize,
    len: usize,
}

impl Rope {
    /// Positions `0..seq_len`.
    pub fn new(spec: &ModelSpec, seq_len: usize, budget: &Budget) -> Result<Self, OjasError> {
        Self::rows(spec, 0, seq_len, budget)
    }

    /// Positions `start..start + len`: `inv_i = base^(-2i / head_dim)`,
    /// angle `pos * inv_i`, then `cat(freqs, freqs)` as nanolab's
    /// `build_rope_cache`. Angles are formed in f64 and rounded once, with
    /// the same expression `ojas-infer`'s `CpuGpt` uses, so the two tables
    /// are equal bit for bit. nanolab builds its table in f32, so its values
    /// differ in the last bits and the gap grows with the position.
    pub fn rows(
        spec: &ModelSpec,
        start: usize,
        len: usize,
        budget: &Budget,
    ) -> Result<Self, OjasError> {
        spec.validate()?;
        let dim = spec.head_dim;
        let half = dim / 2;
        let n = len.checked_mul(dim).ok_or_else(|| OjasError::OutOfRange {
            op: "Rope::rows",
            detail: "table size overflows".to_string(),
        })?;
        start
            .checked_add(len)
            .ok_or_else(|| OjasError::OutOfRange {
                op: "Rope::rows",
                detail: "last position overflows".to_string(),
            })?;
        // Both tables are charged before they are allocated and filled in
        // place: no uncharged staging copy.
        let mut cos_t = Tensor::zeros(&[len, dim], DType::F32, budget)?;
        let mut sin_t = Tensor::zeros(&[len, dim], DType::F32, budget)?;
        {
            let cos = cos_t.f32_slice_mut()?;
            let sin = sin_t.f32_slice_mut()?;
            debug_assert_eq!(cos.len(), n);
            for row in 0..len {
                let pos = (start + row) as f64;
                for i in 0..half {
                    let inv = spec.rope_base.powf(-((2 * i) as f64) / dim as f64);
                    let (s, c) = (pos * inv).sin_cos();
                    let at = row * dim;
                    cos[at + i] = c as f32;
                    cos[at + half + i] = c as f32;
                    sin[at + i] = s as f32;
                    sin[at + half + i] = s as f32;
                }
            }
        }
        Ok(Self {
            cos: cos_t,
            sin: sin_t,
            start,
            len,
        })
    }

    /// The same table resident on `backend`, uploaded once so every later
    /// op shares it.
    pub fn upload<B: Backend + ?Sized>(&self, backend: &B) -> Result<Self, OjasError> {
        Ok(Self {
            cos: backend.upload(&self.cos)?,
            sin: backend.upload(&self.sin)?,
            start: self.start,
            len: self.len,
        })
    }

    pub fn start(&self) -> usize {
        self.start
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// A block's output: the new residual stream and this layer's raw `v`
/// (before any blend), which layer 0 publishes as `v0`.
pub struct BlockOut<V> {
    pub x: V,
    pub raw_v: V,
}

/// One nanolab block over `x` `[batch, rope.len(), n_embd]` with causal
/// attention: [`block_with`] and [`causal_attention`]. `v0` is layer 0's raw
/// `v`; pass `None` for layer 0, which then does not blend and does not read
/// its `vr_lambda`.
pub fn block<G: Graph>(
    g: &mut G,
    spec: &ModelSpec,
    p: &BlockParams<G::V>,
    x: &G::V,
    v0: Option<&G::V>,
    rope: &Rope,
    batch: usize,
) -> Result<BlockOut<G::V>, OjasError> {
    block_with(g, spec, p, x, v0, rope, batch, causal_attention)
}

/// Causal attention of a whole sequence, the attention step of [`block`].
/// `q` is `[B, T, H, D]`, `k` and `v` `[B, T, Hkv, D]`; returns
/// `[B, T, H, D]`.
///
/// Multi-head (`H == Hkv`): permute to `[B, H, T, D]`, [`Graph::sdpa`],
/// permute back, which the tape can differentiate. Grouped-query: the
/// time-major [`Graph::cached_attn`] with `k` and `v` as a cache of exactly
/// `T` positions and `kv_len = T`, in which head `h` reads KV head
/// `h / (H / Hkv)`; only [`crate::Eval`] runs it (a tape refuses grouped
/// query in [`Graph::check_spec`] and cached attention in `cached_attn`).
pub fn causal_attention<G: Graph>(
    g: &mut G,
    q: &G::V,
    k: &G::V,
    v: &G::V,
) -> Result<G::V, OjasError> {
    let heads = |g: &G, t: &G::V| g.tensor(t).map(|t| t.shape().get(2).copied());
    let (h, hkv) = (heads(g, q)?, heads(g, k)?);
    if h == hkv {
        let qh = g.permute(q, &SWAP_TIME_HEADS)?;
        let kh = g.permute(k, &SWAP_TIME_HEADS)?;
        let vh = g.permute(v, &SWAP_TIME_HEADS)?;
        let y = g.sdpa(&qh, &kh, &vh)?;
        g.permute(&y, &SWAP_TIME_HEADS)
    } else {
        let t = g.tensor(q)?.shape().get(1).copied().unwrap_or(0);
        g.cached_attn(q, k, v, t)
    }
}

/// [`block`] with its attention step supplied by the caller, so one
/// function owns the block for every executor (the training and eval
/// forward through [`causal_attention`], a KV-cache decoder through
/// `kv_cache_write` and [`Graph::cached_attn`]).
///
/// `attn(g, q, k, v)` receives, in time-major `[B, T, heads, D]` layout:
/// - `q` `[B, T, n_head, head_dim]` after QK-norm and RoPE;
/// - `k` `[B, T, n_kv_head, head_dim]` after QK-norm and RoPE, the layout
///   `Backend::kv_cache_write` takes;
/// - `v` `[B, T, n_kv_head, head_dim]` after the value residual (the blended
///   `v`, which is what a KV cache holds; the raw `v` is in [`BlockOut`]).
///
/// It returns the attention output `[B, T, n_head, head_dim]`, the layout
/// `Graph::cached_attn` produces. Everything before and after it is the
/// same `Graph` calls in the same order for every caller.
#[allow(clippy::too_many_arguments)]
pub fn block_with<G: Graph>(
    g: &mut G,
    spec: &ModelSpec,
    p: &BlockParams<G::V>,
    x: &G::V,
    v0: Option<&G::V>,
    rope: &Rope,
    batch: usize,
    attn: impl FnOnce(&mut G, &G::V, &G::V, &G::V) -> Result<G::V, OjasError>,
) -> Result<BlockOut<G::V>, OjasError> {
    let (t, dh) = (rope.len(), spec.head_dim);
    let (nh, nkv) = (spec.n_head, spec.n_kv_head);
    // 1. norm1, projections.
    let h = g.rms_norm(x, &p.norm1, spec.eps())?;
    let q = g.linear(&h, &p.q_proj)?;
    let q = g.reshape(&q, &[batch, t, nh, dh])?;
    let k = g.linear(&h, &p.k_proj)?;
    let k = g.reshape(&k, &[batch, t, nkv, dh])?;
    let v = g.linear(&h, &p.v_proj)?;
    let raw_v = g.reshape(&v, &[batch, t, nkv, dh])?;
    // 2. QK-norm, then RoPE.
    let q = g.rms_norm(&q, &p.q_norm, spec.eps())?;
    let k = g.rms_norm(&k, &p.k_norm, spec.eps())?;
    let q = g.rope(&q, &rope.cos, &rope.sin)?;
    let k = g.rope(&k, &rope.cos, &rope.sin)?;
    // 3. Value residual.
    let v = match v0 {
        Some(v0) => g.vres(&raw_v, v0, &p.vr_lambda)?,
        None => raw_v.clone(),
    };
    // 4. Attention, time-major in and out.
    let y = attn(g, &q, &k, &v)?;
    // 5. Gate on h, o_proj, residual.
    let y = g.gate(&h, &p.gate_w, &p.gate_b, &y)?;
    let y = g.reshape(&y, &[batch, t, spec.q_width()])?;
    let o = g.linear(&y, &p.o_proj)?;
    let x = g.add(x, &o)?;
    // 6. norm2, SwiGLU, residual.
    let h2 = g.rms_norm(&x, &p.norm2, spec.eps())?;
    let a = g.linear(&h2, &p.ffn_gate)?;
    let a = g.silu(&a)?;
    let u = g.linear(&h2, &p.ffn_up)?;
    let m = g.mul(&a, &u)?;
    let down = g.linear(&m, &p.ffn_down)?;
    let x = g.add(&x, &down)?;
    Ok(BlockOut { x, raw_v })
}

/// Bring every parameter (in [`crate::param_table`] order) into `g`.
pub fn bind<G: Graph>(
    g: &mut G,
    spec: &ModelSpec,
    params: &[Tensor],
) -> Result<ModelParams<G::V>, OjasError> {
    let flat = params
        .iter()
        .map(|t| g.param(t))
        .collect::<Result<Vec<_>, _>>()?;
    ModelParams::from_flat(spec, flat)
}

/// `(batch, seq)` of `ids`, checked: `U32` `[B, T]` with `T == rope.len()`,
/// on a spec `g` can run ([`Graph::check_spec`]).
fn batch_dims<G: Graph>(
    g: &G,
    spec: &ModelSpec,
    ids: &Tensor,
    rope: &Rope,
) -> Result<(usize, usize), OjasError> {
    g.check_spec(spec)?;
    let &[batch, seq] = ids.shape() else {
        return Err(OjasError::Shape {
            op: "forward",
            detail: format!("token ids {:?} are not [B, T]", ids.shape()),
        });
    };
    if seq > spec.max_seq {
        return Err(OjasError::Shape {
            op: "forward",
            detail: format!("sequence length {seq} exceeds max_seq {}", spec.max_seq),
        });
    }
    if seq != rope.len() || batch == 0 || seq == 0 {
        return Err(OjasError::Shape {
            op: "forward",
            detail: format!(
                "token ids [{batch}, {seq}] against a RoPE table of {} positions",
                rope.len()
            ),
        });
    }
    if rope.start() != 0 {
        return Err(OjasError::Shape {
            op: "forward",
            detail: format!(
                "a full-sequence forward starts at position 0, the table starts at {}",
                rope.start()
            ),
        });
    }
    Ok((batch, seq))
}

/// Embedding, every block, then `norm_f`: `[B, T, n_embd]`.
pub fn forward_hidden<G: Graph>(
    g: &mut G,
    spec: &ModelSpec,
    params: &ModelParams<G::V>,
    ids: &Tensor,
    rope: &Rope,
) -> Result<G::V, OjasError> {
    let (batch, _) = batch_dims(g, spec, ids, rope)?;
    if params.blocks.len() != spec.n_layer {
        return Err(OjasError::Shape {
            op: "forward",
            detail: format!(
                "{} blocks for n_layer {}",
                params.blocks.len(),
                spec.n_layer
            ),
        });
    }
    let mut x = g.embedding(&params.tok_emb, ids)?;
    let mut v0: Option<G::V> = None;
    for p in &params.blocks {
        let out = block(g, spec, p, &x, v0.as_ref(), rope, batch)?;
        if v0.is_none() {
            v0 = Some(out.raw_v);
        }
        x = out.x;
    }
    g.rms_norm(&x, &params.norm_f, spec.eps())
}

/// Logits `[B, T, vocab]` through the tied head.
pub fn forward_logits<G: Graph>(
    g: &mut G,
    spec: &ModelSpec,
    params: &ModelParams<G::V>,
    ids: &Tensor,
    rope: &Rope,
) -> Result<G::V, OjasError> {
    let h = forward_hidden(g, spec, params, ids, rope)?;
    g.linear(&h, &params.tok_emb)
}

/// Mean cross-entropy of the next-token `targets` through the fused tied
/// head (§5): a rank-0 loss. `targets` is `U32` `[B, T]` or `[B * T]`.
#[allow(clippy::too_many_arguments)]
pub fn forward_loss<G: Graph>(
    g: &mut G,
    spec: &ModelSpec,
    params: &ModelParams<G::V>,
    ids: &Tensor,
    targets: &Tensor,
    rope: &Rope,
    ignore: Option<u32>,
    chunk: CeChunk,
) -> Result<G::V, OjasError> {
    let (batch, seq) = batch_dims(g, spec, ids, rope)?;
    let rows = batch * seq;
    let flat_ok = targets.shape() == [rows] || targets.shape() == [batch, seq];
    if !flat_ok {
        return Err(OjasError::Shape {
            op: "forward_loss",
            detail: format!("targets {:?} for ids [{batch}, {seq}]", targets.shape()),
        });
    }
    let targets = targets.reshape(&[rows])?;
    let h = forward_hidden(g, spec, params, ids, rope)?;
    let h = g.reshape(&h, &[rows, spec.n_embd])?;
    g.lin_ce(&h, &params.tok_emb, &targets, ignore, chunk)
}

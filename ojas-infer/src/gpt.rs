use crate::decode::{self, Forward};
use crate::kernels::{attend_one, linear};
use crate::sample::GenerateConfig;
use ojas_core::{Backend, Budget, DType, Numerics, OjasError, Scratch, Tensor};
use ojas_cpu::CpuBackend;
use ojas_model::{param_table, BlockParams, ModelParams, ModelSpec, Rope};

/// The decoder's shape: `ojas_model`'s [`ModelSpec`], the one definition
/// the trainer, the checkpoint and both decoders share.
pub type GptConfig = ModelSpec;

/// Every weight of the model, in `ojas_model`'s nanolab names.
pub type GptWeights = ModelParams<Tensor>;

/// One block's weights, in `ojas_model`'s nanolab names.
pub type BlockWeights = BlockParams<Tensor>;

/// Nanolab's default attention block, for inference on the host.
///
/// Per block, as `nanolab/mixers.py` `Attention.forward` and
/// `nanolab/model.py` `Block` / `SwiGLU` compute it (and as
/// [`ojas_model::block`] writes it once for every executor):
///
/// 1. `h = RMSNorm(x)` (eps [`ModelSpec::eps`]).
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
/// [`CpuGpt::forward_token`] is the fast host path: one token against a
/// [`KvCache`] with local one-row linears and single-query attention (the
/// trait linear copies and repacks the whole weight per call). RMSNorm,
/// QK-norm, RoPE, the gate, the value residual, SiLU and the product run on
/// [`CpuBackend`]. The whole-sequence forward is `ojas_model::Eval` through
/// [`ojas_model::forward_logits`]; [`crate::DeviceDecoder`] decodes on any
/// [`Backend`].
pub struct CpuGpt {
    spec: ModelSpec,
    cpu: CpuBackend,
    /// Clones of the caller's tensors: the bytes stay charged on the budget
    /// that allocated them, including the tied head.
    params: ModelParams<Tensor>,
}

/// Check `params` against `spec`'s [`param_table`]: `n_layer` blocks, and
/// one `F32` tensor of exactly the table's shape per name. Errors name the
/// parameter.
pub(crate) fn check_params(
    op: &'static str,
    spec: &ModelSpec,
    params: &ModelParams<Tensor>,
) -> Result<(), OjasError> {
    spec.validate()?;
    if params.blocks.len() != spec.n_layer {
        return Err(OjasError::Shape {
            op,
            detail: format!(
                "block count {} != n_layer {}",
                params.blocks.len(),
                spec.n_layer
            ),
        });
    }
    let table = param_table(spec)?;
    let flat = params.clone().into_flat();
    for (info, tensor) in table.iter().zip(&flat) {
        if tensor.dtype() != DType::F32 {
            return Err(OjasError::Dtype {
                op,
                expected: DType::F32,
                got: tensor.dtype(),
            });
        }
        if tensor.shape() != info.shape.as_slice() {
            return Err(OjasError::Shape {
                op,
                detail: format!(
                    "{}: shape {:?} != {:?}",
                    info.name,
                    tensor.shape(),
                    info.shape
                ),
            });
        }
    }
    Ok(())
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
        Self::new(model.n_layer(), model.kv_width(), max_len, budget)
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

    /// Forget every position. The storage stays allocated and charged; the
    /// next token goes at position 0.
    pub fn reset(&mut self) {
        self.len = 0;
    }

    /// Keep positions `0..len` and forget the rest, so the next token goes
    /// at `len` (roll back to a prefix to regenerate from it). A `len`
    /// above [`Self::len`] is [`OjasError::OutOfRange`] and changes nothing:
    /// the slots past the current length hold nothing valid.
    pub fn truncate(&mut self, len: usize) -> Result<(), OjasError> {
        truncate_len(&mut self.len, len, "KvCache::truncate")
    }

    fn push_slot(&mut self) -> Result<usize, OjasError> {
        if self.len >= self.max_len {
            return Err(self.refusal(1));
        }
        Ok(self.len)
    }

    /// The error for a request of `extra` more positions than fit.
    pub(crate) fn refusal(&self, extra: usize) -> OjasError {
        capacity_refusal(self.width, self.len, self.max_len, extra)
    }
}

/// Shorten a cache length to `to`, refusing to lengthen it.
pub(crate) fn truncate_len(len: &mut usize, to: usize, op: &'static str) -> Result<(), OjasError> {
    if to > *len {
        return Err(OjasError::OutOfRange {
            op,
            detail: format!("cannot truncate {} filled positions to {to}", *len),
        });
    }
    *len = to;
    Ok(())
}

/// [`OjasError::CapacityExceeded`] for `extra` more positions of `width`
/// f32s onto a cache holding `len` of `max_len`, in bytes.
pub(crate) fn capacity_refusal(
    width: usize,
    len: usize,
    max_len: usize,
    extra: usize,
) -> OjasError {
    let width = u64::try_from(width)
        .ok()
        .and_then(|n| n.checked_mul(4))
        .unwrap_or(u64::MAX);
    let live = (len as u64).saturating_mul(width);
    OjasError::CapacityExceeded {
        requested: live.saturating_add((extra as u64).saturating_mul(width)),
        cap: (max_len as u64).saturating_mul(width),
        live,
    }
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
    /// Check `weights` against `cfg` ([`ModelSpec::validate`], then every
    /// parameter's dtype and shape from [`param_table`]) and keep clones.
    /// Every weight must be a host tensor: the one-row linears read its
    /// bytes directly.
    pub fn new(cfg: &GptConfig, weights: &GptWeights) -> Result<Self, OjasError> {
        const OP: &str = "CpuGpt::new";
        check_params(OP, cfg, weights)?;
        for tensor in weights.clone().into_flat() {
            tensor.f32_slice()?;
        }
        // Backend outputs per token are a few rows of the widest activation.
        let widest = cfg.n_embd.max(cfg.hidden).max(cfg.q_width());
        let scratch = u64::try_from(widest)
            .ok()
            .and_then(|w| w.checked_mul(4 * 16))
            .ok_or_else(|| OjasError::OutOfRange {
                op: OP,
                detail: "activation scratch overflows".into(),
            })?
            .max(1 << 20);
        Ok(Self {
            spec: *cfg,
            cpu: CpuBackend::new(Budget::new(scratch)),
            params: weights.clone(),
        })
    }

    /// Same model, with every [`CpuBackend`] op it runs under `numerics`.
    /// The default is [`CpuBackend`]'s, [`Numerics::Fast`]. The one-row
    /// linears and single-query attention of [`CpuGpt::forward_token`] are
    /// this crate's own ascending-order kernels under either setting.
    pub fn with_numerics(mut self, numerics: Numerics) -> Self {
        self.cpu = self.cpu.with_numerics(numerics);
        self
    }

    /// Arithmetic contract of the [`CpuBackend`] ops this model runs.
    pub fn numerics(&self) -> Numerics {
        self.cpu.numerics()
    }

    /// The model's shape.
    pub fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    /// Per-position key (and value) width, `n_kv_head * head_dim`.
    pub fn kv_width(&self) -> usize {
        self.spec.kv_width()
    }

    pub fn n_embd(&self) -> usize {
        self.spec.n_embd
    }

    pub fn n_layer(&self) -> usize {
        self.spec.n_layer
    }

    pub fn vocab(&self) -> usize {
        self.spec.vocab
    }

    pub fn max_seq(&self) -> usize {
        self.spec.max_seq
    }

    fn apply_linear(
        x: &[f32],
        weight: &Tensor,
        in_dim: usize,
        out_dim: usize,
        y: &mut [f32],
    ) -> Result<(), OjasError> {
        linear(x, weight.f32_slice()?, in_dim, out_dim, y)
    }

    fn row(&self, data: &[f32], shape: &[usize]) -> Result<Tensor, OjasError> {
        Tensor::from_f32(data, shape, Backend::budget(&self.cpu))
    }

    /// `embedding_forward` copies its whole table on every call, so it is
    /// given the one-row view for `token` and looks up row 0.
    fn embed_token(&self, token: u32) -> Result<Vec<f32>, OjasError> {
        let row = self.check_token("CpuGpt::embed_token", token)?;
        let d = self.spec.n_embd;
        let offset = row
            .checked_mul(d)
            .and_then(|elems| elems.checked_mul(4))
            .ok_or_else(|| OjasError::OutOfRange {
                op: "CpuGpt::embed_token",
                detail: "row offset overflows".into(),
            })?;
        let table = self.params.tok_emb.narrow(offset, &[1, d], &[d, 1])?;
        let ids = Tensor::from_u32(&[0], &[1], Backend::budget(&self.cpu))?;
        self.cpu.embedding_forward(&table, &ids)?.to_f32_vec()
    }

    fn check_token(&self, op: &'static str, token: u32) -> Result<usize, OjasError> {
        usize::try_from(token)
            .ok()
            .filter(|&row| row < self.spec.vocab)
            .ok_or_else(|| OjasError::OutOfRange {
                op,
                detail: format!("token id {token} >= vocab {}", self.spec.vocab),
            })
    }

    fn rms_norm(&self, x: &[f32], weight: &Tensor) -> Result<Vec<f32>, OjasError> {
        let row = self.row(x, &[1, self.spec.n_embd])?;
        self.cpu
            .rms_norm_forward(&row, weight, self.spec.eps())?
            .to_f32_vec()
    }

    fn check_cache(&self, cache: &KvCache) -> Result<(), OjasError> {
        if cache.width != self.kv_width()
            || cache.n_layer != self.spec.n_layer
            || cache.max_len > self.spec.max_seq
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
                    self.spec.n_layer,
                    self.spec.max_seq
                ),
            });
        }
        Ok(())
    }

    /// Forward one token at absolute position `cache.len()`, appending its
    /// keys and values. Linear outputs and residual sums are checked
    /// finite, so the returned logits are finite. A cache longer than the
    /// model's `max_seq` is refused before any position is written; a
    /// failure part-way, the final norm and the head included, leaves
    /// `cache.len()` unchanged.
    pub fn forward_token(&self, token: u32, cache: &mut KvCache) -> Result<Vec<f32>, OjasError> {
        self.forward_tokens(&[token], cache)
    }

    /// Forward `tokens` at positions `cache.len()..` (a prompt prefill when
    /// there are several), appending their keys and values, and return the
    /// last position's logits. Only the last position runs the final norm
    /// and the vocabulary head. All or nothing, as
    /// [`crate::DeviceDecoder::forward`]: a request past the cache's room is
    /// [`OjasError::CapacityExceeded`] before anything runs, and a failure
    /// at any token leaves `cache.len()` where it was. Slots at or past
    /// `len` are never read, so what a failed call wrote there is never
    /// seen.
    pub fn forward_tokens(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
    ) -> Result<Vec<f32>, OjasError> {
        self.check_cache(cache)?;
        let Some((&last, head)) = tokens.split_last() else {
            return Err(OjasError::Shape {
                op: "CpuGpt::forward_tokens",
                detail: "no tokens".into(),
            });
        };
        if tokens.len() > cache.remaining() {
            return Err(cache.refusal(tokens.len()));
        }
        let start = cache.len;
        let run = |cache: &mut KvCache| -> Result<Vec<f32>, OjasError> {
            for &id in head {
                let pos = cache.push_slot()?;
                self.trunk(id, pos, cache)?;
                cache.len = pos + 1;
            }
            let pos = cache.push_slot()?;
            let x = self.trunk(last, pos, cache)?;
            let norm = self.rms_norm(&x, &self.params.norm_f)?;
            let mut logits = vec![0.0f32; self.spec.vocab];
            Self::apply_linear(
                &norm,
                &self.params.tok_emb,
                self.spec.n_embd,
                self.spec.vocab,
                &mut logits,
            )?;
            cache.len = pos + 1;
            Ok(logits)
        };
        let out = run(cache);
        if out.is_err() {
            cache.len = start;
        }
        out
    }

    /// Every block for `token` at position `pos`: writes the token's keys
    /// and values into slot `pos`, attends over `0..=pos`, and returns the
    /// residual stream. `cache.len` is the caller's to advance.
    fn trunk(&self, token: u32, pos: usize, cache: &mut KvCache) -> Result<Vec<f32>, OjasError> {
        let s = &self.spec;
        let (d, qw, kvw, dh, hidden) = (s.n_embd, s.q_width(), s.kv_width(), s.head_dim, s.hidden);
        let eps = s.eps();
        let mut x = self.embed_token(token)?;
        let rope = Rope::rows(s, pos, 1, Backend::budget(&self.cpu))?;
        let mut q = vec![0.0f32; qw];
        let mut k = vec![0.0f32; kvw];
        let mut v = vec![0.0f32; kvw];
        let mut proj = vec![0.0f32; d];
        let mut gate = vec![0.0f32; hidden];
        let mut up = vec![0.0f32; hidden];
        let mut v0: Option<Vec<f32>> = None;
        for (layer, b) in self.params.blocks.iter().enumerate() {
            let h = self.rms_norm(&x, &b.norm1)?;
            Self::apply_linear(&h, &b.q_proj, d, qw, &mut q)?;
            Self::apply_linear(&h, &b.k_proj, d, kvw, &mut k)?;
            Self::apply_linear(&h, &b.v_proj, d, kvw, &mut v)?;
            let (qn, kn) = self.cpu.rms_qk_norm_forward(
                &self.row(&q, &[1, 1, s.n_head, dh])?,
                &self.row(&k, &[1, 1, s.n_kv_head, dh])?,
                &b.q_norm,
                &b.k_norm,
                eps,
            )?;
            let qr = self
                .cpu
                .rope_half_split_forward(&qn, &rope.cos, &rope.sin)?
                .to_f32_vec()?;
            let kr = self
                .cpu
                .rope_half_split_forward(&kn, &rope.cos, &rope.sin)?
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
                s.n_head,
                s.n_kv_head,
                dh,
            )?;
            let gated = self
                .cpu
                .per_head_sigmoid_gate_forward(
                    &self.row(&h, &[1, d])?,
                    &b.gate_w,
                    &b.gate_b,
                    &self.row(&mixed, &[1, s.n_head, dh])?,
                )?
                .to_f32_vec()?;
            Self::apply_linear(&gated, &b.o_proj, qw, d, &mut proj)?;
            add_into(&mut x, &proj)?;
            let h2 = self.rms_norm(&x, &b.norm2)?;
            Self::apply_linear(&h2, &b.ffn_gate, d, hidden, &mut gate)?;
            Self::apply_linear(&h2, &b.ffn_up, d, hidden, &mut up)?;
            let act = self.cpu.silu_forward(&self.row(&gate, &[1, hidden])?)?;
            let mixed = self
                .cpu
                .mul_forward(&act, &self.row(&up, &[1, hidden])?)?
                .to_f32_vec()?;
            Self::apply_linear(&mixed, &b.ffn_down, hidden, d, &mut proj)?;
            add_into(&mut x, &proj)?;
        }
        Ok(x)
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
        self.check_cache(cache)?;
        let mut step = HostStep { model: self, cache };
        decode::decode(&mut step, DECODE_OP, prompt, new_tokens, &[], None)
    }

    /// Sampled continuation of `prompt` ([`crate::sample_token`] with a
    /// [`crate::SplitMix64`] seeded from `cfg.seed`).
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
        self.check_cache(cache)?;
        let mut step = HostStep { model: self, cache };
        decode::generate(&mut step, DECODE_OP, prompt, cfg)
    }
}

const DECODE_OP: &str = "CpuGpt::decode";

/// [`CpuGpt`] and its host cache, one [`CpuGpt::forward_token`] per token.
struct HostStep<'a> {
    model: &'a CpuGpt,
    cache: &'a mut KvCache,
}

impl Forward for HostStep<'_> {
    fn room(&self) -> usize {
        self.cache.remaining()
    }

    fn refusal(&self, needed: usize) -> OjasError {
        self.cache.refusal(needed)
    }

    fn check_token(&self, op: &'static str, token: u32) -> Result<(), OjasError> {
        self.model.check_token(op, token).map(|_| ())
    }

    fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>, OjasError> {
        self.model.forward_tokens(tokens, self.cache)
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
            rms_eps: 1e-6,
            tie_embeddings: true,
        }
    }

    /// d = 2, one head of width 2. `o_proj` and the MLP are zero, so the
    /// logits are `emb . RMSNorm(emb[token])` and the larger embedding row
    /// wins.
    fn tiny_weights(budget: &Budget, emb: &[f32]) -> GptWeights {
        let eye = [1.0, 0.0, 0.0, 1.0];
        GptWeights {
            tok_emb: tensor(emb, &[2, 2], budget),
            norm_f: tensor(&[1.0, 1.0], &[2], budget),
            blocks: vec![BlockWeights {
                norm1: tensor(&[1.0, 1.0], &[2], budget),
                q_proj: tensor(&eye, &[2, 2], budget),
                k_proj: tensor(&eye, &[2, 2], budget),
                v_proj: tensor(&eye, &[2, 2], budget),
                q_norm: tensor(&[1.0, 1.0], &[2], budget),
                k_norm: tensor(&[1.0, 1.0], &[2], budget),
                gate_w: tensor(&[0.0, 0.0], &[1, 2], budget),
                gate_b: tensor(&[0.0], &[1], budget),
                vr_lambda: tensor(&[0.0], &[1], budget),
                o_proj: tensor(&[0.0; 4], &[2, 2], budget),
                norm2: tensor(&[1.0, 1.0], &[2], budget),
                ffn_gate: tensor(&[0.0, 0.0], &[1, 2], budget),
                ffn_up: tensor(&[0.0, 0.0], &[1, 2], budget),
                ffn_down: tensor(&[0.0, 0.0], &[2, 1], budget),
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

    /// `max_seq` 0 is refused by [`ModelSpec::validate`]. The spec keeps one
    /// layer with valid weights, so the refusal is `max_seq`'s: the same
    /// weights under the unchanged spec build.
    #[test]
    fn positions_past_max_seq_are_refused_and_max_seq_zero_is_invalid() {
        let budget = Budget::new(1 << 20);
        let (model, cfg) = tiny(&budget, &BIG_FIRST);
        let mut cache = KvCache::for_model(&model, cfg.max_seq, &budget).unwrap();
        for _ in 0..cfg.max_seq {
            model.forward_token(0, &mut cache).unwrap();
        }
        let mut long = KvCache::for_model(&model, cfg.max_seq + 1, &budget).unwrap();
        let err = model.forward_token(0, &mut long).unwrap_err();
        assert!(matches!(err, OjasError::Shape { .. }), "{err}");
        assert_eq!(long.len(), 0);

        let w = tiny_weights(&budget, &BIG_FIRST);
        assert!(CpuGpt::new(&cfg, &w).is_ok());
        let zero = GptConfig { max_seq: 0, ..cfg };
        let err = match CpuGpt::new(&zero, &w) {
            Err(err) => err,
            Ok(_) => panic!("max_seq 0 accepted"),
        };
        assert!(matches!(err, OjasError::OutOfRange { .. }), "{err}");
        assert!(format!("{err}").contains("max_seq"), "{err}");
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
            rms_eps: 1e-6,
            tie_embeddings: true,
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
                norm1: tensor(&ones(fill(d, 0.5)), &[d], budget),
                q_proj: tensor(&fill(d * d, 1.6), &[d, d], budget),
                k_proj: tensor(&fill(d * d, 1.6), &[d, d], budget),
                v_proj: tensor(&fill(d * d, 1.6), &[d, d], budget),
                q_norm: tensor(&ones(fill(dh, 0.5)), &[dh], budget),
                k_norm: tensor(&ones(fill(dh, 0.5)), &[dh], budget),
                gate_w: tensor(&fill(cfg.n_head * d, 1.0), &[cfg.n_head, d], budget),
                gate_b: tensor(&fill(cfg.n_head, 1.0), &[cfg.n_head], budget),
                vr_lambda: tensor(&fill(1, 2.0), &[1], budget),
                o_proj: tensor(&fill(d * d, 1.6), &[d, d], budget),
                norm2: tensor(&ones(fill(d, 0.5)), &[d], budget),
                ffn_gate: tensor(&fill(h * d, 1.6), &[h, d], budget),
                ffn_up: tensor(&fill(h * d, 1.6), &[h, d], budget),
                ffn_down: tensor(&fill(d * h, 1.6), &[d, h], budget),
            })
            .collect();
        let norm_f = tensor(&ones(fill(d, 0.5)), &[d], budget);
        let w = GptWeights {
            tok_emb,
            blocks,
            norm_f,
        };
        (CpuGpt::new(&cfg, &w).unwrap(), cfg)
    }

    #[test]
    fn backend_rms_norm_refuses_non_finite_and_f32_overflowing_rows() {
        let budget = Budget::new(1 << 24);
        let (model, _) = seeded(&budget);
        let weight = model.params.blocks[0].norm1.clone();
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

    /// The final norm and the head run after every block has written slot
    /// 0. In the tiny model `x` stays `emb[token]` (`o_proj` and the MLP are
    /// zero), so a vocabulary row of `f32::MAX` overflows the head's dot
    /// product, and a `norm_f` of `f32::MAX` overflows the final norm.
    #[test]
    fn a_failure_in_the_final_norm_or_head_leaves_the_cache_length_unchanged() {
        let budget = Budget::new(1 << 20);
        let (head, cfg) = tiny(&budget, &[1.0, 0.0, f32::MAX, f32::MAX]);
        let mut w = tiny_weights(&budget, &BIG_FIRST);
        w.norm_f = tensor(&[f32::MAX, f32::MAX], &[2], &budget);
        let norm = CpuGpt::new(&cfg, &w).unwrap();
        for model in [&head, &norm] {
            let mut cache = KvCache::for_model(model, cfg.max_seq, &budget).unwrap();
            let err = model.forward_token(0, &mut cache).unwrap_err();
            assert!(matches!(err, OjasError::NonFinite { .. }), "{err}");
            assert_eq!(cache.len(), 0);
        }
    }

    /// Token 0 runs clean (its logits are finite); token 1's embedding has a
    /// sum of squares past `f32::MAX`, so `norm1` refuses it after token 0
    /// filled slot 0. The prompt is all or nothing.
    #[test]
    fn a_failure_at_any_prompt_token_leaves_the_cache_length_unchanged() {
        let budget = Budget::new(1 << 20);
        let (model, cfg) = tiny(&budget, &[1.0, 0.0, 3.0e19, 0.0]);
        let mut cache = KvCache::for_model(&model, cfg.max_seq, &budget).unwrap();
        assert!(model.forward_token(0, &mut cache).is_ok());
        assert_eq!(cache.len(), 1);
        let mut cache = KvCache::for_model(&model, cfg.max_seq, &budget).unwrap();
        let err = model.greedy_decode(&[0, 1], &mut cache, 1).unwrap_err();
        assert!(matches!(err, OjasError::NonFinite { .. }), "{err}");
        assert_eq!(cache.len(), 0);
        let cfg_gen = GenerateConfig {
            sampling: crate::SamplingConfig::greedy(),
            seed: 0,
            max_new_tokens: 1,
            stop_tokens: Vec::new(),
        };
        let err = model
            .generate(&[0, 0, 1], &mut cache, &cfg_gen)
            .unwrap_err();
        assert!(matches!(err, OjasError::NonFinite { .. }), "{err}");
        assert_eq!(cache.len(), 0);
        // The cache still decodes from where it was: row 1 (3e19) leads the
        // head for token 0.
        assert_eq!(model.greedy_decode(&[0], &mut cache, 1).unwrap(), vec![1]);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn kv_cache_reset_and_truncate_roll_back_to_a_prefix() {
        let budget = Budget::new(1 << 24);
        let (model, cfg) = seeded(&budget);
        let prompt = [3u32, 9, 14, 2, 30];
        let mut cache = KvCache::for_model(&model, cfg.max_seq, &budget).unwrap();
        let full = model.forward_tokens(&prompt, &mut cache).unwrap();
        assert_eq!(cache.len(), prompt.len());

        // Regenerate the last two positions from the 3-token prefix.
        cache.truncate(3).unwrap();
        assert_eq!(cache.len(), 3);
        assert_eq!(
            model.forward_tokens(&prompt[3..], &mut cache).unwrap(),
            full
        );

        let err = cache.truncate(prompt.len() + 1).unwrap_err();
        assert!(matches!(err, OjasError::OutOfRange { .. }), "{err}");
        assert_eq!(cache.len(), prompt.len());
        cache.truncate(prompt.len()).unwrap();
        assert_eq!(cache.len(), prompt.len());

        cache.reset();
        assert!(cache.is_empty());
        assert_eq!(cache.remaining(), cfg.max_seq);
        assert_eq!(model.forward_tokens(&prompt, &mut cache).unwrap(), full);
    }

    /// Prefill computes the vocabulary head for the last prompt token only,
    /// and its logits equal the token-by-token forward's bit for bit.
    #[test]
    fn prompt_prefill_equals_token_by_token_and_refuses_up_front() {
        let budget = Budget::new(1 << 24);
        let (model, cfg) = seeded(&budget);
        let prompt = [5u32, 0, 36, 17, 8, 8, 21];
        let mut one = KvCache::for_model(&model, cfg.max_seq, &budget).unwrap();
        let mut want = Vec::new();
        for &id in &prompt {
            want = model.forward_token(id, &mut one).unwrap();
        }
        let mut all = KvCache::for_model(&model, cfg.max_seq, &budget).unwrap();
        assert_eq!(model.forward_tokens(&prompt, &mut all).unwrap(), want);
        assert_eq!(all.len(), prompt.len());

        let mut small = KvCache::for_model(&model, prompt.len() - 1, &budget).unwrap();
        let err = model.forward_tokens(&prompt, &mut small).unwrap_err();
        assert!(matches!(err, OjasError::CapacityExceeded { .. }), "{err}");
        assert_eq!(small.len(), 0);
        assert!(matches!(
            model.forward_tokens(&[], &mut small),
            Err(OjasError::Shape { .. })
        ));
    }

    /// A weight of the wrong shape or dtype, or a missing block, is refused
    /// with the parameter's nanolab name.
    #[test]
    fn weights_off_the_param_table_are_refused_by_name() {
        let budget = Budget::new(1 << 20);
        let cfg = tiny_cfg();
        let mut w = tiny_weights(&budget, &BIG_FIRST);
        w.blocks[0].k_norm = tensor(&[1.0, 1.0, 1.0], &[3], &budget);
        let err = CpuGpt::new(&cfg, &w).err().expect("wrong shape accepted");
        assert!(
            format!("{err}").contains("blocks.0.mixer.k_norm.weight"),
            "{err}"
        );
        let mut w = tiny_weights(&budget, &BIG_FIRST);
        w.blocks[0].gate_b = Tensor::from_u32(&[0], &[1], &budget).unwrap();
        assert!(matches!(
            CpuGpt::new(&cfg, &w),
            Err(OjasError::Dtype { .. })
        ));
        let mut w = tiny_weights(&budget, &BIG_FIRST);
        w.blocks.clear();
        assert!(matches!(
            CpuGpt::new(&cfg, &w),
            Err(OjasError::Shape { .. })
        ));
    }
}

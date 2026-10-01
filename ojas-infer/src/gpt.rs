use crate::kernels::{attend_one, embed, linear, rms_norm};
use ojas_core::{OjasError, Tensor, RMS_NORM_EPS};

/// Greedy decode over a small pre-norm GPT.
pub struct CpuGpt {
    vocab: usize,
    n_embd: usize,
    n_head: usize,
    head_dim: usize,
    hidden: usize,
    max_seq: usize,
    eps: f32,
    tok_emb: Vec<f32>,
    blocks: Vec<Block>,
    ln_f: Vec<f32>,
}

struct Block {
    ln1: Vec<f32>,
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    ln2: Vec<f32>,
    w_gate: Vec<f32>,
    w_up: Vec<f32>,
    w_down: Vec<f32>,
}

/// Shape of the small decoder. `hidden` is the SwiGLU width.
#[derive(Clone, Debug)]
pub struct GptConfig {
    pub vocab: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub hidden: usize,
    pub max_seq: usize,
}

/// One block's f32 weights. Linear matrices are `[out, in]`, `y = x @ W^T`.
pub struct BlockWeights {
    pub ln1: Tensor,
    pub wq: Tensor,
    pub wk: Tensor,
    pub wv: Tensor,
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

/// Per-layer keys and values. `len` is how many positions are stored.
/// Appending when `len == max_len` returns [`OjasError::CapacityExceeded`]
/// and does not clamp or overwrite the last position.
pub struct KvCache {
    n_embd: usize,
    max_len: usize,
    len: usize,
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl KvCache {
    pub fn new(n_layer: usize, n_embd: usize, max_len: usize) -> Result<Self, OjasError> {
        let cells = n_embd
            .checked_mul(max_len)
            .ok_or_else(|| OjasError::OutOfRange {
                op: "KvCache::new",
                detail: "cache length overflows".into(),
            })?;
        let refused = || OjasError::CapacityExceeded {
            requested: (cells as u64).saturating_mul(4),
            cap: 0,
            live: 0,
        };
        let mut k = Vec::new();
        let mut v = Vec::new();
        k.try_reserve_exact(n_layer).map_err(|_| refused())?;
        v.try_reserve_exact(n_layer).map_err(|_| refused())?;
        for _ in 0..n_layer {
            for side in [&mut k, &mut v] {
                let mut layer: Vec<f32> = Vec::new();
                layer.try_reserve_exact(cells).map_err(|_| refused())?;
                layer.resize(cells, 0.0);
                side.push(layer);
            }
        }
        Ok(Self {
            n_embd,
            max_len,
            len: 0,
            k,
            v,
        })
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

    fn push_slot(&mut self) -> Result<usize, OjasError> {
        if self.len >= self.max_len {
            let width = u64::try_from(self.n_embd)
                .ok()
                .and_then(|n| n.checked_mul(4))
                .unwrap_or(u64::MAX);
            let live = (self.len as u64).saturating_mul(width);
            return Err(OjasError::CapacityExceeded {
                requested: live.saturating_add(width),
                cap: (self.max_len as u64).saturating_mul(width),
                live,
            });
        }
        Ok(self.len)
    }
}

impl CpuGpt {
    pub fn new(cfg: &GptConfig, weights: &GptWeights) -> Result<Self, OjasError> {
        if cfg.vocab == 0
            || cfg.n_embd == 0
            || cfg.n_head == 0
            || cfg.hidden == 0
            || cfg.max_seq == 0
        {
            return Err(OjasError::OutOfRange {
                op: "CpuGpt::new",
                detail: "vocab, n_embd, n_head, hidden, and max_seq must be non-zero".into(),
            });
        }
        if cfg.n_embd % cfg.n_head != 0 {
            return Err(OjasError::Shape {
                op: "CpuGpt::new",
                detail: format!(
                    "n_embd {} is not divisible by n_head {}",
                    cfg.n_embd, cfg.n_head
                ),
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
        let d = cfg.n_embd;
        let h = cfg.hidden;
        let tok_emb = crate::kernels::copy_f32(&weights.tok_emb, &[cfg.vocab, d])?;
        let ln_f = crate::kernels::copy_f32(&weights.ln_f, &[d])?;
        let mut blocks = Vec::with_capacity(cfg.n_layer);
        for (i, b) in weights.blocks.iter().enumerate() {
            blocks.push(Block {
                ln1: crate::kernels::copy_f32(&b.ln1, &[d]).map_err(|e| tag_block(i, e))?,
                wq: crate::kernels::copy_f32(&b.wq, &[d, d]).map_err(|e| tag_block(i, e))?,
                wk: crate::kernels::copy_f32(&b.wk, &[d, d]).map_err(|e| tag_block(i, e))?,
                wv: crate::kernels::copy_f32(&b.wv, &[d, d]).map_err(|e| tag_block(i, e))?,
                wo: crate::kernels::copy_f32(&b.wo, &[d, d]).map_err(|e| tag_block(i, e))?,
                ln2: crate::kernels::copy_f32(&b.ln2, &[d]).map_err(|e| tag_block(i, e))?,
                w_gate: crate::kernels::copy_f32(&b.w_gate, &[h, d])
                    .map_err(|e| tag_block(i, e))?,
                w_up: crate::kernels::copy_f32(&b.w_up, &[h, d]).map_err(|e| tag_block(i, e))?,
                w_down: crate::kernels::copy_f32(&b.w_down, &[d, h])
                    .map_err(|e| tag_block(i, e))?,
            });
        }
        Ok(Self {
            vocab: cfg.vocab,
            n_embd: d,
            n_head: cfg.n_head,
            head_dim: d / cfg.n_head,
            hidden: h,
            max_seq: cfg.max_seq,
            eps: RMS_NORM_EPS,
            tok_emb,
            blocks,
            ln_f,
        })
    }

    pub fn n_embd(&self) -> usize {
        self.n_embd
    }

    pub fn n_layer(&self) -> usize {
        self.blocks.len()
    }

    /// Forward one token, appending its keys and values. The returned logits
    /// may contain NaN; [`argmax_token`] refuses those. A cache longer than
    /// the model's `max_seq` is refused before any position is written.
    pub fn forward_token(&self, token: u32, cache: &mut KvCache) -> Result<Vec<f32>, OjasError> {
        if cache.n_embd != self.n_embd
            || cache.k.len() != self.blocks.len()
            || cache.max_len > self.max_seq
        {
            return Err(OjasError::Shape {
                op: "CpuGpt::forward_token",
                detail: format!(
                    "cache (n_embd {}, layers {}, max_len {}) does not match this model \
                     (n_embd {}, layers {}, max_seq {})",
                    cache.n_embd,
                    cache.k.len(),
                    cache.max_len,
                    self.n_embd,
                    self.blocks.len(),
                    self.max_seq
                ),
            });
        }
        let pos = cache.push_slot()?;
        let mut x = vec![0.0f32; self.n_embd];
        embed(&self.tok_emb, self.n_embd, &[token], &mut x)?;
        let mut norm = vec![0.0f32; self.n_embd];
        let mut q = vec![0.0f32; self.n_embd];
        let mut k = vec![0.0f32; self.n_embd];
        let mut v = vec![0.0f32; self.n_embd];
        let mut proj = vec![0.0f32; self.n_embd];
        let mut hidden = vec![0.0f32; self.hidden];
        let mut gate = vec![0.0f32; self.hidden];
        let mut up = vec![0.0f32; self.hidden];
        for (layer, block) in self.blocks.iter().enumerate() {
            rms_norm(&x, &block.ln1, self.eps, &mut norm)?;
            linear(&norm, &block.wq, self.n_embd, self.n_embd, &mut q)?;
            linear(&norm, &block.wk, self.n_embd, self.n_embd, &mut k)?;
            linear(&norm, &block.wv, self.n_embd, self.n_embd, &mut v)?;
            let base = pos * self.n_embd;
            cache.k[layer][base..base + self.n_embd].copy_from_slice(&k);
            cache.v[layer][base..base + self.n_embd].copy_from_slice(&v);
            let seen = (pos + 1) * self.n_embd;
            let mixed = attend_one(
                &q,
                &cache.k[layer][..seen],
                &cache.v[layer][..seen],
                self.n_head,
                self.head_dim,
            )?;
            linear(&mixed, &block.wo, self.n_embd, self.n_embd, &mut proj)?;
            for i in 0..self.n_embd {
                x[i] += proj[i];
            }
            rms_norm(&x, &block.ln2, self.eps, &mut norm)?;
            linear(&norm, &block.w_gate, self.n_embd, self.hidden, &mut gate)?;
            linear(&norm, &block.w_up, self.n_embd, self.hidden, &mut up)?;
            for i in 0..self.hidden {
                hidden[i] = silu(gate[i]) * up[i];
            }
            linear(&hidden, &block.w_down, self.hidden, self.n_embd, &mut proj)?;
            for i in 0..self.n_embd {
                x[i] += proj[i];
            }
        }
        cache.len = pos + 1;
        rms_norm(&x, &self.ln_f, self.eps, &mut norm)?;
        let logits = self
            .tok_emb
            .chunks_exact(self.n_embd)
            .take(self.vocab)
            .map(|row| norm.iter().zip(row).map(|(a, b)| a * b).sum())
            .collect();
        Ok(logits)
    }

    /// Greedy continuation. A non-finite logit is [`OjasError::NonFinite`],
    /// never token 0. The cache grows by one position per forwarded token
    /// and refuses to grow past its max length.
    pub fn greedy_decode(
        &self,
        prompt: &[u32],
        cache: &mut KvCache,
        new_tokens: usize,
    ) -> Result<Vec<u32>, OjasError> {
        if prompt.is_empty() {
            return Err(OjasError::Shape {
                op: "CpuGpt::greedy_decode",
                detail: "prompt is empty".into(),
            });
        }
        let mut logits = Vec::new();
        for &id in prompt {
            logits = self.forward_token(id, cache)?;
        }
        let mut out = Vec::new();
        out.try_reserve_exact(new_tokens)
            .map_err(|_| OjasError::CapacityExceeded {
                requested: new_tokens as u64,
                cap: cache.max_len as u64,
                live: cache.len as u64,
            })?;
        for _ in 0..new_tokens {
            let next = argmax_token(&logits)?;
            out.push(next);
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

fn silu(x: f32) -> f32 {
    if x >= 0.0 {
        x / (1.0 + (-x).exp())
    } else {
        let z = x.exp();
        x * z / (1.0 + z)
    }
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

    fn tiny(budget: &Budget, emb: &[f32]) -> (CpuGpt, GptConfig) {
        let cfg = GptConfig {
            vocab: 2,
            n_embd: 1,
            n_head: 1,
            n_layer: 1,
            hidden: 1,
            max_seq: 4,
        };
        let w = GptWeights {
            tok_emb: tensor(emb, &[2, 1], budget),
            ln_f: tensor(&[1.0], &[1], budget),
            blocks: vec![BlockWeights {
                ln1: tensor(&[1.0], &[1], budget),
                wq: tensor(&[1.0], &[1, 1], budget),
                wk: tensor(&[1.0], &[1, 1], budget),
                wv: tensor(&[1.0], &[1, 1], budget),
                wo: tensor(&[1.0], &[1, 1], budget),
                ln2: tensor(&[1.0], &[1], budget),
                w_gate: tensor(&[0.0], &[1, 1], budget),
                w_up: tensor(&[0.0], &[1, 1], budget),
                w_down: tensor(&[0.0], &[1, 1], budget),
            }],
        };
        let model = CpuGpt::new(&cfg, &w).unwrap();
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
        let (model, cfg) = tiny(&budget, &[f32::NAN, 0.0]);
        let mut cache = KvCache::new(cfg.n_layer, cfg.n_embd, cfg.max_seq).unwrap();
        let decoded = model.greedy_decode(&[0], &mut cache, 1);
        assert!(decoded.is_err(), "NaN model returned {decoded:?}");
        assert!(!matches!(decoded, Ok(ref ids) if ids.first() == Some(&0)));
    }

    #[test]
    fn greedy_picks_the_larger_tied_embedding_and_cache_refuses_the_next_slot() {
        let budget = Budget::new(1 << 20);
        let (model, cfg) = tiny(&budget, &[1.0, 0.25]);
        let mut cache = KvCache::new(cfg.n_layer, cfg.n_embd, cfg.max_seq).unwrap();
        let logits = model.forward_token(0, &mut cache).unwrap();
        assert!(logits.iter().all(|v| v.is_finite()), "{logits:?}");
        assert!(logits[0] > logits[1], "{logits:?}");
        assert_eq!(argmax_token(&logits).unwrap(), 0);
        assert_eq!(cache.len(), 1);

        let mut cache = KvCache::new(1, 1, 1).unwrap();
        let err = model.greedy_decode(&[0], &mut cache, 1).unwrap_err();
        assert!(matches!(err, OjasError::CapacityExceeded { .. }), "{err}");
        assert_eq!(cache.len(), cache.max_len());
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn kv_cache_refuses_an_unallocatable_length_instead_of_panicking() {
        for (n_layer, n_embd, max_len) in [
            (1, 1, usize::MAX / 2),
            (1, 1 << 20, 1 << 40),
            (usize::MAX, 1, 1),
        ] {
            let got = std::panic::catch_unwind(|| KvCache::new(n_layer, n_embd, max_len));
            let err = got
                .unwrap_or_else(|_| panic!("KvCache::new({n_layer}, {n_embd}, {max_len}) panicked"))
                .err()
                .unwrap_or_else(|| panic!("KvCache::new({n_layer}, {n_embd}, {max_len}) allocated"));
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
        let (model, mut cfg) = tiny(&budget, &[1.0, 0.25]);
        let mut cache = KvCache::new(cfg.n_layer, cfg.n_embd, cfg.max_seq).unwrap();
        for _ in 0..cfg.max_seq {
            model.forward_token(0, &mut cache).unwrap();
        }
        let mut long = KvCache::new(cfg.n_layer, cfg.n_embd, cfg.max_seq + 1).unwrap();
        let err = model.forward_token(0, &mut long).unwrap_err();
        assert!(matches!(err, OjasError::Shape { .. }), "{err}");
        assert_eq!(long.len(), 0);

        cfg.max_seq = 0;
        let w = GptWeights {
            tok_emb: tensor(&[1.0, 0.25], &[2, 1], &budget),
            ln_f: tensor(&[1.0], &[1], &budget),
            blocks: Vec::new(),
        };
        cfg.n_layer = 0;
        assert!(CpuGpt::new(&cfg, &w).is_err());
    }

    #[test]
    fn token_id_out_of_range_is_an_error() {
        let budget = Budget::new(1 << 20);
        let (model, _) = tiny(&budget, &[1.0, 0.25]);
        let mut cache = KvCache::new(1, 1, 4).unwrap();
        let err = model.forward_token(7, &mut cache).unwrap_err();
        assert!(matches!(err, OjasError::OutOfRange { .. }), "{err}");
        assert_eq!(cache.len(), 0);
    }
}

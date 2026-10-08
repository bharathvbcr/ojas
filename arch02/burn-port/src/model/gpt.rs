//! GPT backbone (arch_02): tied embedding + bigram hash embedding →
//! RMSNorm → SmearGate → U-net encoder/decoder with learned skip weights →
//! final RMSNorm → tied logit head with softcap 30.
//!
//! Bigram hash indices are computed CPU-side in the data loader (integer
//! XOR hashing is cheap and exact there); the model consumes them as a
//! second Int tensor. Hash spec (must match data::bigram_hash):
//!   idx[0]  = bigram_vocab_size - 1                      (= 2047)
//!   idx[t]  = (36313*tok[t] ^ 27191*tok[t-1]) mod 2047   (wrapping i32 mul)

use burn::module::{Module, Param};
use burn::prelude::*;
use burn::tensor::activation::log_softmax;

use super::attention::{build_causal_mask, CausalSelfAttention};
use super::block::{Block, BlockOutput};
use super::mlp::Mlp;
use super::norm::rms_norm;
use super::rope::Rotary;
use crate::config::ModelConfig;
use crate::optim::init::{ns_orthogonal, normal_init};

#[derive(Module, Debug)]
pub struct Gpt<B: Backend> {
    pub tok_emb: Param<Tensor<B, 2>>, // [1024, 512], tied with logit head
    // SmearGate
    pub smear_gate: Param<Tensor<B, 1>>, // [512], init 0 (sigmoid → 0.5)
    // BigramHashEmbedding
    pub bigram_embed: Param<Tensor<B, 2>>, // [2048, 48], zero-init
    pub bigram_proj: Param<Tensor<B, 2>>,  // [48, 512], zero-init
    pub bigram_scale: Param<Tensor<B, 1>>, // [1], init 0.05
    // ValueEmbedding (shared table, layers 9 & 10)
    pub ve_embed: Param<Tensor<B, 2>>,  // [1024, 24], normal(0, 0.01)
    pub ve_proj: Param<Tensor<B, 2>>,   // [24, 256], zero-init
    pub ve_scale: Param<Tensor<B, 1>>,  // [1], init 0.1
    pub ve_layer_scales: Param<Tensor<B, 1>>, // [2], init 1.0 (layers 9, 10)
    // U-net skip weights: decoder blocks 5..9 get skips from encoder 4..0
    pub skip_weights: Param<Tensor<B, 2>>, // [5, 512], init 1
    pub blocks: Vec<Block<B>>,
    pub rotary_base: f64,
    pub rotary_dims: usize,
    pub rotary_train_seq_len: usize,
    pub num_encoder_layers: usize,
    pub logit_softcap: f64,
    pub ve_layers: [usize; 2],
}

impl<B: Backend> Gpt<B> {
    pub fn new(cfg: &ModelConfig, device: &B::Device) -> Self {
        let c = cfg.model_dim;
        let kvd = cfg.kv_dim();
        let blocks = (0..cfg.num_layers)
            .map(|i| Block {
                attn: CausalSelfAttention {
                    q_w: Param::from_tensor(ns_orthogonal::<B>([c, c], device)),
                    k_w: Param::from_tensor(ns_orthogonal::<B>([c, kvd], device)),
                    v_w: Param::from_tensor(ns_orthogonal::<B>([c, kvd], device)),
                    out_w: Param::from_tensor(Tensor::zeros([c, c], device)),
                    q_gain: Param::from_tensor(
                        Tensor::ones([cfg.num_heads], device) * cfg.qk_gain_init,
                    ),
                    vr_lambda: Param::from_tensor(
                        Tensor::from_floats([0.5, 0.5], device),
                    ),
                    num_heads: cfg.num_heads,
                    num_kv_heads: cfg.num_kv_heads,
                    head_dim: cfg.head_dim(),
                    rope_dims: cfg.rope_dims,
                    xsa: cfg.xsa_active(i),
                },
                mlp: Mlp {
                    up_w: Param::from_tensor(ns_orthogonal::<B>([c, cfg.mlp_dim], device)),
                    down_w: Param::from_tensor(Tensor::zeros([cfg.mlp_dim, c], device)),
                },
                resid_mix: Param::from_tensor(Tensor::cat(
                    vec![
                        Tensor::<B, 2>::ones([1, c], device),
                        Tensor::<B, 2>::zeros([1, c], device),
                    ],
                    0,
                )),
                attn_scale: Param::from_tensor(Tensor::ones([c], device)),
                mlp_scale: Param::from_tensor(Tensor::ones([c], device)),
                layer_idx: i,
            })
            .collect();

        Self {
            tok_emb: Param::from_tensor(normal_init::<B>(
                [cfg.vocab_size, c],
                cfg.tied_embed_init_std,
                device,
            )),
            smear_gate: Param::from_tensor(Tensor::zeros([c], device)),
            bigram_embed: Param::from_tensor(Tensor::zeros(
                [cfg.bigram_vocab_size, cfg.bigram_dim],
                device,
            )),
            bigram_proj: Param::from_tensor(Tensor::zeros([cfg.bigram_dim, c], device)),
            bigram_scale: Param::from_tensor(
                Tensor::ones([1], device) * cfg.bigram_scale_init,
            ),
            ve_embed: Param::from_tensor(normal_init::<B>(
                [cfg.vocab_size, cfg.ve_dim],
                0.01,
                device,
            )),
            ve_proj: Param::from_tensor(Tensor::zeros([cfg.ve_dim, kvd], device)),
            ve_scale: Param::from_tensor(Tensor::ones([1], device) * cfg.ve_scale_init),
            ve_layer_scales: Param::from_tensor(Tensor::ones([2], device)),
            skip_weights: Param::from_tensor(Tensor::ones(
                [cfg.num_encoder_layers(), c],
                device,
            )),
            blocks,
            rotary_base: cfg.rope_base,
            rotary_dims: cfg.rope_dims,
            rotary_train_seq_len: cfg.rope_train_seq_len,
            num_encoder_layers: cfg.num_encoder_layers(),
            logit_softcap: cfg.logit_softcap,
            ve_layers: cfg.ve_layers,
        }
    }

    /// Returns logits [B, T, V] (softcapped).
    /// `tokens`, `bigram_idx`: [B, T] Int.
    pub fn forward(&self, tokens: Tensor<B, 2, Int>, bigram_idx: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let device = tokens.device();
        let [b, t] = tokens.dims();
        let [v_sz, c] = self.tok_emb.dims();
        let _ = v_sz;
        let rotary = Rotary::new(self.rotary_base, self.rotary_dims, self.rotary_train_seq_len);
        // Build RoPE cos/sin + causal mask ONCE per forward (was rebuilt inside
        // every attention call — 11 layers x micro-steps of redundant CPU trig
        // + host->device uploads). Cloned handles below share device buffers.
        let tables = rotary.tables::<B>(t, &device);
        let mask = build_causal_mask::<B>(t, &device);

        // token + bigram embeddings
        let flat = tokens.clone().reshape([b * t]);
        let mut x = self.tok_emb.val().select(0, flat.clone()).reshape([b, t, c]);
        x = x + self.bigram(bigram_idx, b, t, c);
        x = rms_norm(x);
        x = self.smear(x);
        let x0 = x.clone();

        // value embedding base (shared): scale * proj(embed(tokens))
        let ve_base = {
            let e = self.ve_embed.val().select(0, flat); // [B*T, ve_dim]
            let p = e.matmul(self.ve_proj.val()); // [B*T, kv_dim]
            let kvd = self.ve_proj.dims()[1];
            p.reshape([b, t, kvd]) * self.ve_scale.val().reshape([1, 1, 1])
        };

        let mut v0: Option<Tensor<B, 4>> = None;
        let mut skips: Vec<Tensor<B, 3>> = Vec::with_capacity(self.num_encoder_layers);

        for (i, block) in self.blocks.iter().enumerate() {
            // decoder skip connections (LIFO)
            if i >= self.num_encoder_layers {
                if let Some(skip) = skips.pop() {
                    // python: x = x + skip_weights[decoder_i] * skips.pop()
                    // (LIFO: decoder 0 ↔ encoder 4, decoder 1 ↔ encoder 3, ...)
                    let di = i - self.num_encoder_layers; // decoder index 0..4
                    let w = self.skip_weights.val().narrow(0, di, 1).reshape([1, 1, c]);
                    x = x + skip * w;
                }
            }

            // value embedding for this layer?
            let ve = if i == self.ve_layers[0] {
                Some(ve_base.clone() * self.ve_layer_scales.val().narrow(0, 0, 1).reshape([1, 1, 1]))
            } else if i == self.ve_layers[1] {
                Some(ve_base.clone() * self.ve_layer_scales.val().narrow(0, 1, 1).reshape([1, 1, 1]))
            } else {
                None
            };

            let BlockOutput { x: xb, raw_v } =
                block.forward(x, x0.clone(), v0.clone(), ve, &tables, &mask);
            x = xb;

            if i == 0 {
                v0 = Some(raw_v); // layer 0's RAW value projection
            }
            if i < self.num_encoder_layers {
                skips.push(x.clone());
            }
        }

        let x = rms_norm(x);
        // tied logit head + softcap
        let logits = x
            .reshape([b * t, c])
            .matmul(self.tok_emb.val().transpose())
            .reshape([b, t, self.tok_emb.dims()[0]]);
        (logits / self.logit_softcap).tanh() * self.logit_softcap
    }

    /// Mean cross-entropy (nats) over all positions.
    pub fn loss(
        &self,
        tokens: Tensor<B, 2, Int>,
        bigram_idx: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
    ) -> Tensor<B, 1> {
        let logits = self.forward(tokens, bigram_idx);
        let [b, t, v] = logits.dims();
        let logp = log_softmax(logits.reshape([b * t, v]), 1);
        let tgt = targets.reshape([b * t, 1]);
        let nll = -logp.gather(1, tgt); // [B*T, 1]
        nll.mean()
    }

    /// Per-token negative log likelihood (nats), [B, T] — for BPB eval.
    pub fn nll_per_token(
        &self,
        tokens: Tensor<B, 2, Int>,
        bigram_idx: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
    ) -> Tensor<B, 2> {
        let logits = self.forward(tokens, bigram_idx);
        let [b, t, v] = logits.dims();
        let logp = log_softmax(logits.reshape([b * t, v]), 1);
        let tgt = targets.reshape([b * t, 1]);
        let nll = -logp.gather(1, tgt);
        nll.reshape([b, t])
    }

    fn bigram(&self, idx: Tensor<B, 2, Int>, b: usize, t: usize, c: usize) -> Tensor<B, 3> {
        let flat = idx.reshape([b * t]);
        let e = self.bigram_embed.val().select(0, flat); // [B*T, 48]
        let h = e.matmul(self.bigram_proj.val()).reshape([b, t, c]);
        h * self.bigram_scale.val().reshape([1, 1, 1])
    }

    /// SmearGate: per-channel sigmoid gate mixing each position with its
    /// predecessor (position 0 mixes with zeros).
    fn smear(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, t, c] = x.dims();
        let g = burn::tensor::activation::sigmoid(self.smear_gate.val()).reshape([1, 1, c]);
        let zeros = Tensor::zeros([b, 1, c], &x.device());
        let x_prev = Tensor::cat(vec![zeros, x.clone().narrow(1, 0, t - 1)], 1);
        x.clone() * (Tensor::ones_like(&g) - g.clone()) + x_prev * g
    }
}

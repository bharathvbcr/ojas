//! Phase 5: Multi-Token Prediction (MTP) — assistant config, cross-KV, verify.
//!
//! Real E4B assistant weights (`google/gemma-4-E4B-it-assistant`) load into
//! [`MtpSession::from_assistant_dir`] (centroids + embeds + pre/post + 4-layer
//! Q/MLP stack). Draft attendsover target **shared sliding + shared global** KV
//! (no K/V on the assistant). Synthetic path remains for unit tests.

use crate::config::{Gemma4AssistantConfig, Gemma4Config, Gemma4TextConfig, LayerType};
use crate::diag;
use crate::error::{Error, Result};
use crate::forward::{apply_rope, gemv, rms_norm, softcap_f32};
use crate::kv::{KvLayout, SharedKvBuffer};
use crate::quant::bf16_bits_to_f32;
use safetensors::tensor::SafeTensors;
use std::fs;
use std::path::Path;
use std::time::Instant;

/// Adaptive draft length policy (challenge / LiteRT-class knob).
#[derive(Clone, Debug)]
pub struct AdaptiveDraftPolicy {
    pub min_draft: usize,
    pub max_draft: usize,
    /// Rolling accept rate EMA (0..=1).
    pub accept_ema: f32,
    pub ema_alpha: f32,
}

impl Default for AdaptiveDraftPolicy {
    fn default() -> Self {
        Self {
            min_draft: 1,
            max_draft: 5,
            accept_ema: 0.7,
            ema_alpha: 0.2,
        }
    }
}

impl AdaptiveDraftPolicy {
    /// Map accept EMA → draft length in `[min, max]`.
    pub fn draft_len(&self) -> usize {
        let t = self.accept_ema.clamp(0.0, 1.0);
        let span = self.max_draft.saturating_sub(self.min_draft) as f32;
        let n = self.min_draft + (t * span).round() as usize;
        n.clamp(self.min_draft, self.max_draft)
    }

    pub fn observe_accept_rate(&mut self, accepted: usize, drafted: usize) {
        if drafted == 0 {
            return;
        }
        let rate = accepted as f32 / drafted as f32;
        self.accept_ema =
            (1.0 - self.ema_alpha) * self.accept_ema + self.ema_alpha * rate;
    }
}

/// E4B-style clustered LM head (centroids → top-k → vocab).
#[derive(Clone, Debug)]
pub struct ClusteredLmHead {
    pub hidden: usize,
    pub num_centroids: usize,
    pub top_k: usize,
    /// `[num_centroids, hidden]`
    pub centroids: Vec<f32>,
    /// HF `masked_embedding.token_ordering`: cluster `c` holds the tokens
    /// `cluster_tokens[c * per .. (c + 1) * per]`, `per = vocab / num_centroids`.
    pub cluster_tokens: Vec<u32>,
    /// Optional dense `[vocab, hidden]` rows for selected clusters (synthetic: full).
    pub cluster_rows: Vec<f32>,
    pub vocab: usize,
}

impl ClusteredLmHead {
    /// Tiny synthetic clustered head for unit tests.
    pub fn synthetic_mini(hidden: usize, vocab: usize, centroids: usize, top_k: usize) -> Self {
        let mut seed = 42u64;
        let mut rand = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32 / (u32::MAX as f32)) * 0.1 - 0.05
        };
        let centroids_w: Vec<f32> = (0..centroids * hidden).map(|_| rand()).collect();
        let cluster_tokens: Vec<u32> = (0..vocab as u32).collect();
        let cluster_rows: Vec<f32> = (0..vocab * hidden).map(|_| rand()).collect();
        Self {
            hidden,
            num_centroids: centroids,
            top_k,
            centroids: centroids_w,
            cluster_tokens,
            cluster_rows,
            vocab,
        }
    }

    /// HF `Gemma4AssistantMaskedEmbedder`: score centroids, take the top-k clusters,
    /// and score only their tokens; every other logit is `-inf` (HF fills a value below
    /// the selected minimum, so the argmax is the same).
    pub fn logits(&self, h: &[f32]) -> Result<Vec<f32>> {
        if h.len() != self.hidden {
            return Err(Error::Config(format!(
                "clustered head: h len {} != {}",
                h.len(),
                self.hidden
            )));
        }
        let mut scores = vec![0f32; self.num_centroids];
        for (c, score) in scores.iter_mut().enumerate() {
            *score = dot_fast(&self.centroids[c * self.hidden..(c + 1) * self.hidden], h);
        }
        let mut order: Vec<usize> = (0..self.num_centroids).collect();
        order.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        order.truncate(self.top_k.min(self.num_centroids));
        let per = self.vocab / self.num_centroids.max(1);
        let mut logits = vec![f32::NEG_INFINITY; self.vocab];
        for c in order {
            for &tok in &self.cluster_tokens[c * per..(c + 1) * per] {
                let v = tok as usize;
                if v >= self.vocab {
                    return Err(Error::Weights(format!("token_ordering entry {v} >= vocab")));
                }
                logits[v] = dot_fast(&self.cluster_rows[v * self.hidden..(v + 1) * self.hidden], h);
            }
        }
        Ok(logits)
    }

    pub fn greedy(&self, h: &[f32], softcap: Option<f32>) -> Result<u32> {
        let mut logits = self.logits(h)?;
        if let Some(sc) = softcap {
            for v in &mut logits {
                if v.is_finite() {
                    *v = softcap_f32(*v, sc);
                }
            }
        }
        let mut best_i = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best_i = i;
            }
        }
        Ok(best_i as u32)
    }
}

/// Activation bridge: project backbone hidden → assistant hidden.
#[derive(Clone, Debug)]
pub struct ActivationBridge {
    pub w: Vec<f32>, // [asst_hidden, backbone_hidden]
    pub asst_hidden: usize,
    pub backbone_hidden: usize,
}

impl ActivationBridge {
    pub fn identity_pad(backbone: usize, asst: usize) -> Self {
        let mut w = vec![0f32; asst * backbone];
        let n = backbone.min(asst);
        for i in 0..n {
            w[i * backbone + i] = 1.0;
        }
        Self {
            w,
            asst_hidden: asst,
            backbone_hidden: backbone,
        }
    }

    pub fn project(&self, h: &[f32]) -> Result<Vec<f32>> {
        if h.len() != self.backbone_hidden {
            return Err(Error::Config("bridge: backbone dim mismatch".into()));
        }
        Ok(gemv(
            &self.w,
            h,
            self.asst_hidden,
            self.backbone_hidden,
        ))
    }
}

/// Cross-KV: draft assistant consumes target's last-sliding / last-global shared slots.
#[derive(Debug)]
pub struct CrossKvBridge {
    pub target_sliding: SharedKvBuffer,
    pub target_global: SharedKvBuffer,
    /// True after [`Self::replace_from_densified`] from live backbone shared buffers.
    pub synced_from_target: bool,
}

impl CrossKvBridge {
    pub fn from_target_layout(layout: &KvLayout) -> Self {
        Self {
            target_sliding: SharedKvBuffer::new(
                layout.max_seq,
                layout.local_kv_heads,
                layout.local_head_dim,
            ),
            target_global: SharedKvBuffer::new(
                layout.max_seq,
                layout.global_kv_heads,
                layout.global_head_dim,
            ),
            synced_from_target: false,
        }
    }

    /// Replace host mirrors with densified shared K/V from the target decode session.
    pub fn replace_from_densified(
        &mut self,
        sliding_k: &[f32],
        sliding_v: &[f32],
        sliding_t: usize,
        global_k: &[f32],
        global_v: &[f32],
        global_t: usize,
    ) -> Result<()> {
        let sn = self.target_sliding.heads * self.target_sliding.dim;
        let gn = self.target_global.heads * self.target_global.dim;
        if sliding_t > 0 {
            if sliding_k.len() < sliding_t * sn || sliding_v.len() < sliding_t * sn {
                return Err(Error::Kv("sliding densify len mismatch".into()));
            }
            if sliding_t > self.target_sliding.max_seq {
                return Err(Error::Kv("sliding densify exceeds max_seq".into()));
            }
            self.target_sliding.k[..sliding_t * sn]
                .copy_from_slice(&sliding_k[..sliding_t * sn]);
            self.target_sliding.v[..sliding_t * sn]
                .copy_from_slice(&sliding_v[..sliding_t * sn]);
            self.target_sliding.seq_len = sliding_t;
        } else {
            self.target_sliding.seq_len = 0;
        }
        if global_t > 0 {
            if global_k.len() < global_t * gn || global_v.len() < global_t * gn {
                return Err(Error::Kv("global densify len mismatch".into()));
            }
            if global_t > self.target_global.max_seq {
                return Err(Error::Kv("global densify exceeds max_seq".into()));
            }
            self.target_global.k[..global_t * gn]
                .copy_from_slice(&global_k[..global_t * gn]);
            self.target_global.v[..global_t * gn]
                .copy_from_slice(&global_v[..global_t * gn]);
            self.target_global.seq_len = global_t;
        } else {
            self.target_global.seq_len = 0;
        }
        self.synced_from_target = sliding_t > 0 || global_t > 0;
        Ok(())
    }

    /// Append one draft KV timestep into the matching target shared buffer.
    pub fn append_draft(
        &mut self,
        layer_type: LayerType,
        k: &[f32],
        v: &[f32],
    ) -> Result<()> {
        match layer_type {
            LayerType::SlidingAttention => self.target_sliding.append(k, v),
            LayerType::FullAttention => self.target_global.append(k, v),
        }
    }
}

/// One assistant draft layer (Q-only consumer — no K/V weights).
#[derive(Clone, Debug)]
pub struct AssistantDraftLayer {
    pub layer_type: LayerType,
    pub input_norm: Vec<f32>,
    pub post_attn_norm: Vec<f32>,
    pub pre_ff_norm: Vec<f32>,
    pub post_ff_norm: Vec<f32>,
    pub q_proj: Vec<f32>,
    pub o_proj: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub gate_proj: Vec<f32>,
    pub up_proj: Vec<f32>,
    pub down_proj: Vec<f32>,
    pub hq: usize,
    pub head_dim: usize,
    pub layer_scalar: f32,
}

/// Result of verifying a draft against the target model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifyResult {
    pub accepted: usize,
    pub drafted: usize,
    /// First rejected position in draft (None if all accepted).
    pub reject_at: Option<usize>,
    pub bonus_token: Option<u32>,
}

/// Greedy verify: accept while `draft[i] == target_greedy[i]`.
pub fn verify_draft(draft: &[u32], target_greedy: &[u32]) -> VerifyResult {
    let n = draft.len().min(target_greedy.len());
    let mut accepted = 0usize;
    let mut reject_at = None;
    for i in 0..n {
        if draft[i] == target_greedy[i] {
            accepted += 1;
        } else {
            reject_at = Some(i);
            break;
        }
    }
    let bonus = if reject_at.is_none() && target_greedy.len() > draft.len() {
        Some(target_greedy[draft.len()])
    } else if let Some(r) = reject_at {
        Some(target_greedy[r])
    } else {
        None
    };
    let r = VerifyResult {
        accepted,
        drafted: draft.len(),
        reject_at,
        bonus_token: bonus,
    };
    diag::log(
        "mtp",
        format_args!(
            "verify accepted={}/{} reject_at={:?} bonus={:?}",
            r.accepted, r.drafted, r.reject_at, r.bonus_token
        ),
    );
    r
}

/// Canonical E4B assistant shapes (plan: ~77M, hidden 256, clustered head).
pub fn e4b_assistant_preset() -> Gemma4AssistantConfig {
    let mut text = Gemma4TextConfig::e4b_preset();
    // Drafter is a small stack — override text dims for assistant.
    text.hidden_size = 256;
    text.intermediate_size = 2048;
    text.num_hidden_layers = 4;
    text.num_attention_heads = 4;
    text.num_key_value_heads = 2;
    text.num_kv_shared_layers = 4; // all consumer relative to target KV
    text.hidden_size_per_layer_input = 0;
    text.layer_types = vec![
        LayerType::SlidingAttention,
        LayerType::SlidingAttention,
        LayerType::SlidingAttention,
        LayerType::FullAttention,
    ];
    Gemma4AssistantConfig {
        backbone_hidden_size: 2560,
        num_centroids: 2048,
        centroid_intermediate_top_k: 32,
        use_ordered_embeddings: true,
        tie_word_embeddings: false,
        text_config: text,
    }
}

/// Canonical 31B assistant (~500M, hidden 1024, dense vocab — no centroids required).
pub fn b31_assistant_preset() -> Gemma4AssistantConfig {
    let mut text = Gemma4TextConfig::b31_preset();
    text.hidden_size = 1024;
    text.intermediate_size = 4096;
    text.num_hidden_layers = 6;
    text.num_attention_heads = 8;
    text.num_key_value_heads = 4;
    text.num_kv_shared_layers = 0;
    text.hidden_size_per_layer_input = 0;
    text.attention_k_eq_v = false;
    text.layer_types = vec![
        LayerType::SlidingAttention,
        LayerType::SlidingAttention,
        LayerType::SlidingAttention,
        LayerType::SlidingAttention,
        LayerType::SlidingAttention,
        LayerType::FullAttention,
    ];
    Gemma4AssistantConfig {
        backbone_hidden_size: 5376,
        num_centroids: 0, // dense vocab path
        centroid_intermediate_top_k: 0,
        use_ordered_embeddings: false,
        tie_word_embeddings: false,
        text_config: text,
    }
}

/// MTP session: bridge + clustered head + adaptive draft + verify.
pub struct MtpSession {
    pub assistant: Gemma4AssistantConfig,
    pub bridge: ActivationBridge,
    pub clustered: Option<ClusteredLmHead>,
    pub policy: AdaptiveDraftPolicy,
    pub cross_kv: CrossKvBridge,
    /// HF `pre_projection.weight` [asst_h, 2*backbone] when real weights loaded.
    pub pre_projection: Option<Vec<f32>>,
    /// HF `post_projection.weight` [backbone, asst_h] (reserved for activation bridge).
    pub post_projection: Option<Vec<f32>>,
    /// 4-layer Q-consumer stack (real assistant).
    pub draft_layers: Vec<AssistantDraftLayer>,
    pub final_norm: Option<Vec<f32>>,
    pub real_weights: bool,
    /// Last synced shared KV lengths (diagnostic).
    pub last_shared_sliding_t: usize,
    pub last_shared_global_t: usize,
}

impl MtpSession {
    pub fn e4b_synthetic(target_layout: &KvLayout) -> Result<Self> {
        let assistant = e4b_assistant_preset();
        let bridge = ActivationBridge::identity_pad(
            assistant.backbone_hidden_size,
            assistant.text_config.hidden_size,
        );
        let clustered = Some(ClusteredLmHead::synthetic_mini(
            assistant.text_config.hidden_size,
            64, // tiny vocab for tests
            8,
            4,
        ));
        Ok(Self {
            assistant,
            bridge,
            clustered,
            policy: AdaptiveDraftPolicy::default(),
            cross_kv: CrossKvBridge::from_target_layout(target_layout),
            pre_projection: None,
            post_projection: None,
            draft_layers: Vec::new(),
            final_norm: None,
            real_weights: false,
            last_shared_sliding_t: 0,
            last_shared_global_t: 0,
        })
    }

    /// Mini-graph MTP session sized to a synthetic backbone `hidden`.
    pub fn mini_synthetic(target_layout: &KvLayout, backbone_hidden: usize) -> Result<Self> {
        let mut assistant = e4b_assistant_preset();
        assistant.backbone_hidden_size = backbone_hidden;
        assistant.text_config.hidden_size = backbone_hidden.min(64).max(16);
        let draft_h = assistant.text_config.hidden_size;
        let bridge = ActivationBridge::identity_pad(backbone_hidden, draft_h);
        let clustered = Some(ClusteredLmHead::synthetic_mini(draft_h, 64, 8, 4));
        Ok(Self {
            assistant,
            bridge,
            clustered,
            policy: AdaptiveDraftPolicy {
                min_draft: 1,
                max_draft: 3,
                accept_ema: 0.7,
                ema_alpha: 0.2,
            },
            cross_kv: CrossKvBridge::from_target_layout(target_layout),
            pre_projection: None,
            post_projection: None,
            draft_layers: Vec::new(),
            final_norm: None,
            real_weights: false,
            last_shared_sliding_t: 0,
            last_shared_global_t: 0,
        })
    }

    /// One assistant step (HF `Gemma4AssistantForCausalLM.forward` for one query):
    /// `pre_projection(concat(target_embed, target_hidden))`, the four KV-shared
    /// layers attending over the synced target sliding / global KV at position `pos`,
    /// the final norm, the clustered head's greedy token, and
    /// `post_projection(normed)` as the next step's backbone-sized hidden.
    pub fn assistant_step(
        &self,
        target_embed: &[f32],
        target_hidden: &[f32],
        pos: usize,
    ) -> Result<(u32, Vec<f32>)> {
        let bb = self.assistant.backbone_hidden_size;
        let asst_h = self.assistant.text_config.hidden_size;
        let (pre, post) = match (&self.pre_projection, &self.post_projection) {
            (Some(pre), Some(post)) => (pre, post),
            _ => {
                return Err(Error::Config(
                    "assistant_step needs the real pre/post projections".into(),
                ))
            }
        };
        if target_embed.len() != bb || target_hidden.len() != bb {
            return Err(Error::Config(format!(
                "assistant_step: embed {} / hidden {} != backbone {bb}",
                target_embed.len(),
                target_hidden.len()
            )));
        }
        if !self.cross_kv.synced_from_target {
            return Err(Error::Kv("assistant_step: target KV not synced".into()));
        }
        let head = self
            .clustered
            .as_ref()
            .ok_or_else(|| Error::Config("assistant_step needs the clustered head".into()))?;
        let mut cat = Vec::with_capacity(2 * bb);
        cat.extend_from_slice(target_embed);
        cat.extend_from_slice(target_hidden);
        let x = gemv_fast(pre, &cat, asst_h, 2 * bb);
        let mut h = self.forward_draft_layers(&x, pos)?;
        if let Some(ref fnorm) = self.final_norm {
            h = rms_norm(&h, fnorm, self.assistant.text_config.rms_norm_eps as f32);
        }
        let tok = head.greedy(&h, None)?;
        Ok((tok, gemv_fast(post, &h, bb, asst_h)))
    }

    /// Draft `n` tokens after `last_token` (HF `get_candidates` loop). `last_hidden`
    /// is the target's final-normed hidden state at the position before
    /// `last_token` (the one whose logits chose it), `pos` is `last_token`'s
    /// position (held fixed across the draft, as HF does), and `embed` returns the
    /// target's scaled input embedding of a token.
    pub fn draft(
        &self,
        last_token: u32,
        last_hidden: &[f32],
        pos: usize,
        n: usize,
        embed: &mut dyn FnMut(u32) -> Result<Vec<f32>>,
    ) -> Result<Vec<u32>> {
        let mut tok = last_token;
        let mut hidden = last_hidden.to_vec();
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let e = embed(tok)?;
            let (next, h) = self.assistant_step(&e, &hidden, pos)?;
            out.push(next);
            tok = next;
            hidden = h;
        }
        Ok(out)
    }

    /// The four assistant layers (HF `Gemma4TextDecoderLayer`, all KV-shared): Q from
    /// the assistant, K/V from the target's last sliding / full layer. Queries see
    /// every synced key (bidirectional mask) within the sliding window for sliding
    /// layers. Attention scale is 1.0 (Q/K are RMS-normed).
    fn forward_draft_layers(&self, x: &[f32], pos: usize) -> Result<Vec<f32>> {
        let tc = &self.assistant.text_config;
        let eps = tc.rms_norm_eps as f32;
        let asst_h = tc.hidden_size;
        let window = tc.sliding_window.unwrap_or(usize::MAX);
        let mut h = x.to_vec();
        for layer in &self.draft_layers {
            let (s, theta, rotary_dim) = match layer.layer_type {
                LayerType::SlidingAttention => {
                    let fam = tc.rope_parameters.as_ref().map(|r| &r.sliding_attention);
                    let theta = fam.map(|f| f.rope_theta).unwrap_or(10_000.0);
                    (&self.cross_kv.target_sliding, theta, layer.head_dim)
                }
                LayerType::FullAttention => {
                    let fam = tc.rope_parameters.as_ref().map(|r| &r.full_attention);
                    let theta = fam.map(|f| f.rope_theta).unwrap_or(1_000_000.0);
                    let partial = fam.and_then(|f| f.partial_rotary_factor).unwrap_or(1.0);
                    let rd = crate::forward::proportional_rope_dim(layer.head_dim, partial);
                    (&self.cross_kv.target_global, theta, rd)
                }
            };
            if s.dim != layer.head_dim || s.seq_len == 0 {
                return Err(Error::Kv(format!(
                    "assistant layer {:?}: target KV dim {} len {} (head_dim {})",
                    layer.layer_type, s.dim, s.seq_len, layer.head_dim
                )));
            }
            let residual = h.clone();
            let normed = rms_norm(&h, &layer.input_norm, eps);
            let mut q = gemv_fast(&layer.q_proj, &normed, layer.hq * layer.head_dim, asst_h);
            for hi in 0..layer.hq {
                let slice = &mut q[hi * layer.head_dim..(hi + 1) * layer.head_dim];
                let row = rms_norm(slice, &layer.q_norm, eps);
                slice.copy_from_slice(&row);
                apply_rope(slice, layer.head_dim, rotary_dim, pos, theta as f32);
            }
            let n = s.heads * s.dim;
            let t0 = match layer.layer_type {
                LayerType::SlidingAttention => s.seq_len.saturating_sub(window),
                LayerType::FullAttention => 0,
            };
            let attn = gqa_attend_one(
                &q,
                &s.k[t0 * n..s.seq_len * n],
                &s.v[t0 * n..s.seq_len * n],
                layer.hq,
                s.heads,
                s.dim,
                s.seq_len - t0,
            )?;
            let o = gemv_fast(&layer.o_proj, &attn, asst_h, layer.hq * layer.head_dim);
            let o = rms_norm(&o, &layer.post_attn_norm, eps);
            for ((a, r), b) in h.iter_mut().zip(residual.iter()).zip(o.iter()) {
                *a = r + b;
            }

            let residual = h.clone();
            let n3 = rms_norm(&h, &layer.pre_ff_norm, eps);
            let inter = layer.gate_proj.len() / asst_h;
            let gate = gemv_fast(&layer.gate_proj, &n3, inter, asst_h);
            let up = gemv_fast(&layer.up_proj, &n3, inter, asst_h);
            let mid: Vec<f32> = gate
                .iter()
                .zip(up.iter())
                .map(|(&g, &u)| crate::forward::gelu_pytorch_tanh(g) * u)
                .collect();
            let down = gemv_fast(&layer.down_proj, &mid, asst_h, inter);
            let down = rms_norm(&down, &layer.post_ff_norm, eps);
            for ((a, r), b) in h.iter_mut().zip(residual.iter()).zip(down.iter()) {
                *a = (r + b) * layer.layer_scalar;
            }
        }
        Ok(h)
    }

    pub fn verify_and_adapt(&mut self, draft: &[u32], target: &[u32]) -> VerifyResult {
        let r = verify_draft(draft, target);
        self.policy
            .observe_accept_rate(r.accepted, r.drafted.max(1));
        diag::log(
            "mtp",
            format_args!(
                "adapt accept_ema→{:.3} draft_len→{}",
                self.policy.accept_ema,
                self.policy.draft_len()
            ),
        );
        r
    }

    /// Load real HF assistant (`google/gemma-4-E4B-it-assistant`) into MTP session.
    ///
    /// Loads centroids + ordered embeds + pre/post + 4-layer Q-consumer stack.
    /// Cross-KV attention runs after [`CrossKvBridge::replace_from_densified`]
    /// from the target's shared sliding / global buffers.
    pub fn from_assistant_dir(dir: impl AsRef<Path>, target_layout: &KvLayout) -> Result<Self> {
        let dir = dir.as_ref();
        let t0 = Instant::now();
        diag::log(
            "mtp",
            format_args!("▶ from_assistant_dir {}", dir.display()),
        );
        let cfg = Gemma4Config::from_path(dir.join("config.json")).map_err(|e| {
            diag::err("mtp", "assistant config.json", &e);
            e
        })?;
        let assistant = match cfg {
            Gemma4Config::Assistant(a) => a,
            _ => {
                let e = Error::Config(format!(
                    "{}: expected gemma4_assistant config",
                    dir.display()
                ));
                diag::err("mtp", "config kind", &e);
                return Err(e);
            }
        };
        let path = dir.join("model.safetensors");
        let bytes = fs::read(&path).map_err(|e| {
            let err = Error::Io(format!("{}: {e}", path.display()));
            diag::err("mtp", "assistant weight read", &err);
            err
        })?;
        diag::log(
            "mtp",
            format_args!(
                "read assistant weights {} in {:.1} ms",
                diag::fmt_bytes(bytes.len() as u64),
                t0.elapsed().as_secs_f64() * 1e3
            ),
        );
        let st = SafeTensors::deserialize(&bytes).map_err(|e| {
            let err = Error::Safetensors(format!("{}: {e}", path.display()));
            diag::err("mtp", "assistant safetensors", &err);
            err
        })?;
        diag::log(
            "mtp",
            format_args!("assistant safetensors keys={}", st.names().len()),
        );
        let centroids = load_bf16_matrix(&st, "masked_embedding.centroids.weight")?;
        let embeds = load_bf16_matrix(&st, "model.embed_tokens.weight")?;
        let pre = load_bf16_matrix(&st, "pre_projection.weight").ok();
        let post = load_bf16_matrix(&st, "post_projection.weight").ok();
        let ordering = load_i64_u32(&st, "masked_embedding.token_ordering").ok();
        let final_norm = load_bf16_matrix(&st, "model.norm.weight").ok();

        let asst_h = assistant.text_config.hidden_size;
        let vocab = assistant.text_config.vocab_size;
        let n_cent = assistant.num_centroids;
        if centroids.len() != n_cent * asst_h {
            return Err(Error::Weights(format!(
                "centroids len {} != {}×{}",
                centroids.len(),
                n_cent,
                asst_h
            )));
        }
        if embeds.len() != vocab * asst_h {
            return Err(Error::Weights(format!(
                "embed len {} != vocab×hidden",
                embeds.len()
            )));
        }

        // HF `token_ordering.view(num_centroids, vocab / num_centroids)`: row c lists
        // cluster c's tokens. Without ordered embeddings the clusters are contiguous.
        let cluster_tokens: Vec<u32> = match ordering {
            Some(ord) if ord.len() == vocab && vocab % n_cent == 0 => ord,
            Some(ord) => {
                return Err(Error::Weights(format!(
                    "token_ordering len {} (vocab {vocab}, {n_cent} centroids)",
                    ord.len()
                )))
            }
            None => (0..vocab as u32).collect(),
        };

        let clustered = ClusteredLmHead {
            hidden: asst_h,
            num_centroids: n_cent,
            top_k: assistant.centroid_intermediate_top_k.max(1),
            centroids,
            cluster_tokens,
            cluster_rows: embeds,
            vocab,
        };

        let bridge = ActivationBridge::identity_pad(
            assistant.backbone_hidden_size,
            asst_h,
        );

        let n_layers = assistant.text_config.num_hidden_layers;
        let mut draft_layers = Vec::with_capacity(n_layers);
        for li in 0..n_layers {
            let lt = assistant
                .text_config
                .layer_types
                .get(li)
                .copied()
                .unwrap_or(LayerType::SlidingAttention);
            let head_dim = match lt {
                LayerType::SlidingAttention => assistant.text_config.head_dim,
                LayerType::FullAttention => assistant.text_config.global_head_dim,
            };
            let q_proj = load_bf16_matrix(&st, &format!("model.layers.{li}.self_attn.q_proj.weight"))?;
            let hq = (q_proj.len() / asst_h).max(1) / head_dim.max(1);
            let layer_scalar = load_bf16_matrix(&st, &format!("model.layers.{li}.layer_scalar"))
                .ok()
                .and_then(|v| v.first().copied())
                .unwrap_or(1.0);
            draft_layers.push(AssistantDraftLayer {
                layer_type: lt,
                input_norm: load_bf16_matrix(
                    &st,
                    &format!("model.layers.{li}.input_layernorm.weight"),
                )?,
                post_attn_norm: load_bf16_matrix(
                    &st,
                    &format!("model.layers.{li}.post_attention_layernorm.weight"),
                )?,
                pre_ff_norm: load_bf16_matrix(
                    &st,
                    &format!("model.layers.{li}.pre_feedforward_layernorm.weight"),
                )?,
                post_ff_norm: load_bf16_matrix(
                    &st,
                    &format!("model.layers.{li}.post_feedforward_layernorm.weight"),
                )?,
                q_proj,
                o_proj: load_bf16_matrix(
                    &st,
                    &format!("model.layers.{li}.self_attn.o_proj.weight"),
                )?,
                q_norm: load_bf16_matrix(
                    &st,
                    &format!("model.layers.{li}.self_attn.q_norm.weight"),
                )?,
                gate_proj: load_bf16_matrix(
                    &st,
                    &format!("model.layers.{li}.mlp.gate_proj.weight"),
                )?,
                up_proj: load_bf16_matrix(
                    &st,
                    &format!("model.layers.{li}.mlp.up_proj.weight"),
                )?,
                down_proj: load_bf16_matrix(
                    &st,
                    &format!("model.layers.{li}.mlp.down_proj.weight"),
                )?,
                hq: hq.max(1),
                head_dim,
                layer_scalar,
            });
        }

        eprintln!(
            "gemma-metal: loaded E4B assistant from {} — centroids={} asst_h={} vocab={} layers={} pre={} post={}",
            dir.display(),
            n_cent,
            asst_h,
            vocab,
            draft_layers.len(),
            pre.as_ref().map(|p| p.len()).unwrap_or(0),
            post.as_ref().map(|p| p.len()).unwrap_or(0)
        );
        diag::log(
            "mtp",
            format_args!(
                "✔ assistant loaded centroids={n_cent} asst_h={asst_h} vocab={vocab} layers={} \
                 pre={} post={} in {:.1}s",
                draft_layers.len(),
                pre.is_some(),
                post.is_some(),
                t0.elapsed().as_secs_f64()
            ),
        );

        Ok(Self {
            assistant,
            bridge,
            clustered: Some(clustered),
            policy: AdaptiveDraftPolicy::default(),
            cross_kv: CrossKvBridge::from_target_layout(target_layout),
            pre_projection: pre,
            post_projection: post,
            draft_layers,
            final_norm,
            real_weights: true,
            last_shared_sliding_t: 0,
            last_shared_global_t: 0,
        })
    }
}

/// `w · x` with eight independent partial sums, so the loop vectorizes (one running sum
/// cannot: float addition does not reassociate). The drafter's numbers only steer which
/// tokens are proposed; verification decides what is emitted.
fn dot_fast(w: &[f32], x: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let wc = w.chunks_exact(8);
    let xc = x.chunks_exact(8);
    let (wr, xr) = (wc.remainder(), xc.remainder());
    for (a, b) in wc.zip(xc) {
        for i in 0..8 {
            acc[i] += a[i] * b[i];
        }
    }
    let mut tail = 0f32;
    for (a, b) in wr.iter().zip(xr) {
        tail += a * b;
    }
    ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7])) + tail
}

/// `W [rows, cols] · x` through [`dot_fast`].
fn gemv_fast(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    w[..rows * cols].chunks_exact(cols).map(|row| dot_fast(row, &x[..cols])).collect()
}

/// Decode-step GQA: q [hq*d], k/v [T, hkv, d] → [hq*d].
fn gqa_attend_one(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    hq: usize,
    hkv: usize,
    d: usize,
    tkv: usize,
) -> Result<Vec<f32>> {
    if hq == 0 || hkv == 0 || d == 0 || tkv == 0 {
        return Ok(vec![0f32; hq * d]);
    }
    if q.len() < hq * d {
        return Err(Error::Config("gqa q len".into()));
    }
    let mut out = vec![0f32; hq * d];
    let reps = (hq / hkv).max(1);
    for h in 0..hq {
        let kv_h = h / reps;
        let qh = &q[h * d..(h + 1) * d];
        // Softmax scores over T.
        let mut scores = vec![0f32; tkv];
        let mut mx = f32::NEG_INFINITY;
        for t in 0..tkv {
            let mut s = 0f32;
            let koff = (t * hkv + kv_h) * d;
            for i in 0..d {
                s += qh[i] * k[koff + i];
            }
            // Gemma 4: scale 1.0 after QK-norm.
            scores[t] = s;
            mx = mx.max(s);
        }
        let mut sum = 0f32;
        for t in 0..tkv {
            scores[t] = (scores[t] - mx).exp();
            sum += scores[t];
        }
        let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        let ooff = h * d;
        for t in 0..tkv {
            let w = scores[t] * inv;
            let voff = (t * hkv + kv_h) * d;
            for i in 0..d {
                out[ooff + i] += w * v[voff + i];
            }
        }
    }
    Ok(out)
}

fn load_bf16_matrix(st: &SafeTensors<'_>, name: &str) -> Result<Vec<f32>> {
    let t = st
        .tensor(name)
        .map_err(|_| Error::Weights(format!("missing {name}")))?;
    if t.dtype() != safetensors::Dtype::BF16 {
        return Err(Error::Weights(format!("{name}: expected BF16")));
    }
    let n: usize = t.shape().iter().product();
    let data = t.data();
    let mut out = Vec::with_capacity(n);
    for chunk in data.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        out.push(bf16_bits_to_f32(bits));
    }
    Ok(out)
}

fn load_i64_u32(st: &SafeTensors<'_>, name: &str) -> Result<Vec<u32>> {
    let t = st
        .tensor(name)
        .map_err(|_| Error::Weights(format!("missing {name}")))?;
    let data = t.data();
    let mut out = Vec::with_capacity(data.len() / 8);
    for chunk in data.chunks_exact(8) {
        let v = i64::from_le_bytes(chunk.try_into().unwrap());
        out.push(v.max(0) as u32);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Gemma4TextConfig;
    use crate::forward::SyntheticE4bGraph;

    #[test]
    fn adaptive_draft_len_bounds() {
        let mut p = AdaptiveDraftPolicy {
            min_draft: 1,
            max_draft: 5,
            accept_ema: 0.0,
            ema_alpha: 0.5,
        };
        assert_eq!(p.draft_len(), 1);
        p.accept_ema = 1.0;
        assert_eq!(p.draft_len(), 5);
        p.observe_accept_rate(0, 5);
        assert!(p.accept_ema < 1.0);
    }

    #[test]
    fn verify_partial_accept() {
        let r = verify_draft(&[1, 2, 3, 4], &[1, 2, 9, 4, 5]);
        assert_eq!(r.accepted, 2);
        assert_eq!(r.reject_at, Some(2));
        assert_eq!(r.bonus_token, Some(9));
    }

    #[test]
    fn clustered_head_greedy_finite() {
        let h = ClusteredLmHead::synthetic_mini(16, 32, 4, 2);
        let x = vec![0.01f32; 16];
        let tok = h.greedy(&x, Some(30.0)).unwrap();
        assert!(tok < 32);
    }


    #[test]
    fn bridge_projects() {
        let b = ActivationBridge::identity_pad(8, 4);
        let y = b.project(&vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .unwrap();
        assert_eq!(y, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn assistant_step_refuses_without_projections_or_synced_kv() {
        let host = SyntheticE4bGraph::mini_parity().unwrap();
        let mtp = MtpSession::e4b_synthetic(&host.kv).unwrap();
        let h = vec![0.02f32; mtp.assistant.backbone_hidden_size];
        assert!(mtp.assistant_step(&h, &h, 0).is_err());
        let mut tok_calls = 0;
        let r = mtp.draft(3, &h, 0, 2, &mut |_| {
            tok_calls += 1;
            Ok(h.clone())
        });
        assert!(r.is_err());
        assert_eq!(tok_calls, 1);
    }

    #[test]
    fn cross_kv_append() {
        let cfg = Gemma4TextConfig::e4b_preset();
        let layout = KvLayout::from_config(&cfg, 128).unwrap();
        let mut x = CrossKvBridge::from_target_layout(&layout);
        let n = layout.local_kv_heads * layout.local_head_dim;
        let k = vec![0.1f32; n];
        let v = vec![0.2f32; n];
        x.append_draft(LayerType::SlidingAttention, &k, &v).unwrap();
        assert_eq!(x.target_sliding.seq_len, 1);
    }

    #[test]
    fn load_real_assistant_if_cached() {
        let Some(dir) = crate::weights::resolve_default_e4b_assistant_cache() else {
            eprintln!("skip: E4B assistant not in HF cache");
            return;
        };
        let layout = KvLayout::from_config(&Gemma4TextConfig::e4b_preset(), 64).unwrap();
        let mtp = MtpSession::from_assistant_dir(&dir, &layout).unwrap();
        assert!(mtp.real_weights);
        assert!(mtp.pre_projection.is_some());
        assert!(mtp.clustered.is_some());
        // token_ordering is a permutation of the vocab.
        let head = mtp.clustered.as_ref().unwrap();
        let mut seen = vec![false; head.vocab];
        for &t in &head.cluster_tokens {
            assert!(!std::mem::replace(&mut seen[t as usize], true), "token {t} twice");
        }
    }

    #[test]
    fn presets_parse() {
        let e = e4b_assistant_preset();
        assert_eq!(e.backbone_hidden_size, 2560);
        assert_eq!(e.num_centroids, 2048);
        let b = b31_assistant_preset();
        assert_eq!(b.backbone_hidden_size, 5376);
        assert_eq!(b.num_centroids, 0);
    }
}

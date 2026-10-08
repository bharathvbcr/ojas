//! Hyperparameters mirroring train_gpt_sprint_native.py defaults
//! (VALUE_RESIDUAL=1, GATED_ATTENTION=0).

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub vocab_size: usize,     // 1024
    pub num_layers: usize,     // 11
    pub num_heads: usize,      // 8
    pub num_kv_heads: usize,   // 4  (GQA group = 2)
    pub model_dim: usize,      // 512
    pub mlp_dim: usize,        // 1536 (3.0 * 512)
    pub rope_base: f64,        // 10000.0 (NTK-extended at runtime, see rope.rs)
    pub rope_dims: usize,      // 16 (partial RoPE)
    pub rope_train_seq_len: usize, // 1024 — rotary's internal reference length.
    // Training used seq_len=2048 > 1024, so the NTK extension branch FIRES:
    // effective base ≈ 10000 * 2^(16/14) ≈ 22082. Keep this at 1024 to match.
    pub logit_softcap: f64,    // 30.0
    pub qk_gain_init: f64,     // 1.5
    pub tied_embed_init_std: f64, // 0.005
    pub bigram_vocab_size: usize, // 2048 (hash mod 2047; index 2047 = position 0)
    pub bigram_dim: usize,     // 48
    pub bigram_scale_init: f64, // 0.05
    pub ve_dim: usize,         // 24
    pub ve_layers: [usize; 2], // [9, 10]
    pub ve_scale_init: f64,    // 0.1
    pub xsa_last_n: usize,     // 4 → XSA on layers 7,8,9,10
}

impl ModelConfig {
    pub fn head_dim(&self) -> usize {
        self.model_dim / self.num_heads // 64
    }
    pub fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim() // 256
    }
    pub fn num_encoder_layers(&self) -> usize {
        self.num_layers / 2 // 5
    }
    pub fn xsa_active(&self, layer: usize) -> bool {
        layer >= self.num_layers - self.xsa_last_n
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 1024,
            num_layers: 11,
            num_heads: 8,
            num_kv_heads: 4,
            model_dim: 512,
            mlp_dim: 1536,
            rope_base: 10000.0,
            rope_dims: 16,
            rope_train_seq_len: 1024,
            logit_softcap: 30.0,
            qk_gain_init: 1.5,
            tied_embed_init_std: 0.005,
            bigram_vocab_size: 2048,
            bigram_dim: 48,
            bigram_scale_init: 0.05,
            ve_dim: 24,
            ve_layers: [9, 10],
            ve_scale_init: 0.1,
            xsa_last_n: 4,
        }
    }
}

impl ModelConfig {
    /// The `sota` preset from `run_toy_3070ti.py` — **this is the config the
    /// recorded arch_02 numbers were measured on** (calibrated BPB 1.9875,
    /// sliding 1.9902, seeds 1337+42, 3000 steps on the 3070 Ti). The
    /// `Default` config above is the full sprint model; use this one when
    /// reproducing the ablation ladder on the M5.
    ///
    /// From the original run log header:
    ///   num_layers:4 model_dim:128 heads:4 kv:2 mlp_mult:3 rope_dims:8
    ///   bigram 512/48, VE 24 @ layers 2,3, XSA last 2, model_params:780188
    pub fn sota_toy() -> Self {
        Self {
            vocab_size: 1024,
            num_layers: 4,
            num_heads: 4,
            num_kv_heads: 2,
            model_dim: 128,
            mlp_dim: 384, // 3.0 * 128
            rope_base: 10000.0,
            rope_dims: 8,
            rope_train_seq_len: 1024, // seq 256 < 1024 → NTK branch does NOT fire
            logit_softcap: 30.0,
            qk_gain_init: 1.5,
            tied_embed_init_std: 0.005,
            bigram_vocab_size: 512,
            bigram_dim: 48,
            bigram_scale_init: 0.05,
            ve_dim: 24,
            ve_layers: [2, 3],
            ve_scale_init: 0.1,
            xsa_last_n: 2,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub seq_len: usize,          // 2048
    pub iterations: usize,       // 20000
    pub warmdown_iters: usize,   // 3500 (linear LR decay to 0 at the end)
    pub micro_batch: usize,      // sequences per micro-step (memory-bound; naive
    // SDPA scores are [B,8,T,T] so keep small; python used 48 w/ flash-attn)
    pub grad_accum: usize,       // micro-steps per optimizer step.
    // python: 786432 tokens/step = 384 seqs of 2048. micro_batch*grad_accum
    // should equal 384 for parity (e.g. 4 * 96).
    pub matrix_lr: f64,          // 0.025 (Muon)
    pub tied_embed_lr: f64,      // 0.035 (AdamW, embeddings)
    pub scalar_lr: f64,          // 0.025 (AdamW, scalars/controls)
    pub weight_decay: f64,       // 0.04 (all groups)
    pub adam_beta1: f64,         // 0.9
    pub adam_beta2: f64,         // 0.95
    pub adam_eps: f64,           // 1e-8
    pub muon_momentum_start: f64, // 0.92
    pub muon_momentum_end: f64,  // 0.99
    pub muon_momentum_warmup: usize, // 1500 steps
    pub grad_clip: f64,          // 0.3 (global L2 norm)
    pub ema_decay: f64,          // 0.997
    pub eval_every: usize,       // steps between validation BPB evals
    pub eval_stride: usize,      // 64 (sliding-window eval stride)
    pub eval_batch: usize,       // windows per batched eval forward
    pub val_cap: usize,          // max val tokens used for sliding eval (0 = all)
    pub log_every: usize,        // steps between console/JSONL metric lines
    pub prefetch_depth: usize,   // prefetched micro-batches held in the channel
    pub profile_every: usize,    // 0 = off; else sync-gated phase timing every N steps
    pub profile_first: usize,    // always profile the first N steps
    pub seed: u64,               // 1337
}

impl TrainConfig {
    /// Python's lr_mul (step-based mode): 1.0 until the warmdown window,
    /// then linear decay to 0. There is NO warmup ramp.
    pub fn lr_mul(&self, step: usize) -> f64 {
        if step < self.iterations.saturating_sub(self.warmdown_iters) {
            1.0
        } else {
            (self.iterations - step) as f64 / self.warmdown_iters as f64
        }
    }

    /// Muon momentum warmup: (1-frac)*0.92 + frac*0.99, frac = min(step/1500, 1).
    pub fn muon_momentum(&self, step: usize) -> f64 {
        let frac = (step as f64 / self.muon_momentum_warmup as f64).min(1.0);
        (1.0 - frac) * self.muon_momentum_start + frac * self.muon_momentum_end
    }
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            seq_len: 2048,
            iterations: 20000,
            warmdown_iters: 3500,
            micro_batch: 4,
            grad_accum: 96,
            matrix_lr: 0.025,
            tied_embed_lr: 0.035,
            scalar_lr: 0.025,
            weight_decay: 0.04,
            adam_beta1: 0.9,
            adam_beta2: 0.95,
            adam_eps: 1e-8,
            muon_momentum_start: 0.92,
            muon_momentum_end: 0.99,
            muon_momentum_warmup: 1500,
            grad_clip: 0.3,
            ema_decay: 0.997,
            eval_every: 500,
            eval_stride: 64,
            eval_batch: 16,
            val_cap: 65_536,
            log_every: 20,
            prefetch_depth: 4,
            profile_every: 0,
            profile_first: 3,
            seed: 1337,
        }
    }
}

impl TrainConfig {
    /// Training-side counterpart of [`ModelConfig::sota_toy`] — the env of the
    /// 3070 Ti ablation long stage (`run_toy_3070ti.py` sota preset +
    /// conductor stage overrides): 3000 steps, seq 256, **4096 tok/step**,
    /// no warmup, no warmdown (`WARMDOWN_ITERS=0` → lr_mul ≡ 1), Muon momentum
    /// warmup 0.92→**0.95**, sliding eval on the first 16384 val tokens.
    ///
    /// Batch shape: the 3070 Ti ran 2 micro x 8 accum purely for 8 GB VRAM.
    /// On the M5 that accumulation is pure kernel-dispatch overhead (~8x the
    /// launches for identical math), so the default here is 16 micro x 1 accum
    /// — the same 4096 tokens and the same summed gradient per step. Use
    /// `--micro-batch 2 --grad-accum 8` to replicate the original shape.
    pub fn sota_toy() -> Self {
        Self {
            seq_len: 256,
            iterations: 3000,
            warmdown_iters: 0,
            micro_batch: 16,
            grad_accum: 1,
            muon_momentum_end: 0.95,
            val_cap: 16_384,
            eval_every: 500,
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lr_schedule_shape() {
        let c = TrainConfig::default();
        assert_eq!(c.lr_mul(0), 1.0);
        assert_eq!(c.lr_mul(16499), 1.0);
        assert!((c.lr_mul(16500) - 1.0).abs() < 1e-9);
        assert!((c.lr_mul(18250) - 0.5).abs() < 1e-9);
        assert!(c.lr_mul(19999) > 0.0);
    }

    #[test]
    fn momentum_warmup() {
        let c = TrainConfig::default();
        assert!((c.muon_momentum(0) - 0.92).abs() < 1e-12);
        assert!((c.muon_momentum(750) - 0.955).abs() < 1e-12);
        assert!((c.muon_momentum(1500) - 0.99).abs() < 1e-12);
        assert!((c.muon_momentum(9999) - 0.99).abs() < 1e-12);
    }

    #[test]
    fn derived_dims() {
        let m = ModelConfig::default();
        assert_eq!(m.head_dim(), 64);
        assert_eq!(m.kv_dim(), 256);
        assert_eq!(m.num_encoder_layers(), 5);
        assert!(!m.xsa_active(6));
        assert!(m.xsa_active(7));
        assert!(m.xsa_active(10));
    }
}

//! Fixed-architecture activation stash for hand-written backward.

use crate::tensor::Tensor;

/// Sota-toy dims mirrored for tape allocation docs.
pub const SOTA_L: usize = 4;
pub const SOTA_C: usize = 128;
pub const SOTA_H: usize = 4;
pub const SOTA_HKV: usize = 2;
pub const SOTA_D: usize = 32;

/// Per-step activation tape (stem → blocks → head). Owns GPU tensors when filled.
#[derive(Default)]
pub struct Tape {
    pub step: u64,
    pub input_ids: Option<Tensor>,
    pub target_ids: Option<Tensor>,
    /// Token+bigram pre-RMSNorm.
    pub stem_pre_norm: Option<Tensor>,
    /// Post-RMSNorm, pre-smear.
    pub stem_post_norm: Option<Tensor>,
    /// Post-smear residual stream (= x0).
    pub stem: Option<Tensor>,
    pub x0: Option<Tensor>,
    /// Layer-0 raw V before λ mix `[B,T,Hkv,D]`.
    pub v0: Option<Tensor>,
    pub layer: Vec<LayerTape>,
    /// Encoder skip tensors in push order (layer 0, then 1).
    pub skips: Vec<Tensor>,
    /// Final RMSNorm input (= last layer x_out, post skip).
    pub pre_final_norm: Option<Tensor>,
    pub final_norm: Option<Tensor>,
    pub logits_pre: Option<Tensor>,
    pub logits_post: Option<Tensor>,
    pub loss: Option<Tensor>,
}

#[derive(Default)]
pub struct LayerTape {
    /// Stream entering the block (post skip-add if decoder).
    pub x_stream: Option<Tensor>,
    pub x_in: Option<Tensor>,
    pub attn_in: Option<Tensor>, // RMS(x_in)*α
    /// Post-GEMM, pre-qkv_post.
    pub q_pre: Option<Tensor>,
    pub k_pre: Option<Tensor>,
    pub v_pre: Option<Tensor>,
    /// VE contribution `[BT, kv]` when layer uses VE.
    pub ve: Option<Tensor>,
    pub q: Option<Tensor>,
    pub k: Option<Tensor>,
    pub v_mixed: Option<Tensor>,
    pub raw_v: Option<Tensor>,
    /// Flash attention output (pre-XSA).
    pub attn_y_flash: Option<Tensor>,
    /// Flash attention logsumexp L `[B, H, T]` (FA-2 backward).
    pub attn_lse: Option<Tensor>,
    /// Post-flash (+ XSA) heads.
    pub attn_y: Option<Tensor>,
    pub attn_out: Option<Tensor>,
    // --- Mamba-2 tape ---
    pub mamba_z: Option<Tensor>,
    pub mamba_z_pre_silu: Option<Tensor>,
    pub mamba_xbc_pre: Option<Tensor>,
    pub mamba_xbc_post: Option<Tensor>,
    pub mamba_xbc: Option<Tensor>,
    pub mamba_xs: Option<Tensor>,
    pub mamba_bm: Option<Tensor>,
    pub mamba_cm: Option<Tensor>,
    pub mamba_dt_raw: Option<Tensor>,
    pub mamba_dt: Option<Tensor>,
    pub mamba_x_scaled: Option<Tensor>,
    pub mamba_log_da: Option<Tensor>,
    pub mamba_ssd_y: Option<Tensor>,
    pub mamba_h_states: Option<Tensor>,
    pub mamba_y_pre_out: Option<Tensor>,
    pub mamba_y_flat: Option<Tensor>,
    pub mamba_y_norm: Option<Tensor>,
    // --- MinGRU tape ---
    pub mingru_z_raw: Option<Tensor>,
    pub mingru_h_raw: Option<Tensor>,
    pub mingru_h_pre: Option<Tensor>,
    pub mingru_v0_up: Option<Tensor>,
    pub mingru_h_out: Option<Tensor>,
    /// After attn residual, before MLP.
    pub x_mid: Option<Tensor>,
    pub mlp_in: Option<Tensor>,
    /// Pre-activation MLP hidden `[BT, mlp]`.
    pub mlp_pre_act: Option<Tensor>,
    /// Post-activation hidden (input to down proj).
    pub mlp_hidden: Option<Tensor>,
    pub mlp_out: Option<Tensor>,
    pub x_out: Option<Tensor>,
    pub after_skip: Option<Tensor>, // decoder only
}

impl Tape {
    pub fn new(num_layers: usize) -> Self {
        Self {
            layer: (0..num_layers).map(|_| LayerTape::default()).collect(),
            ..Default::default()
        }
    }

    pub fn new_sota() -> Self {
        Self::new(SOTA_L)
    }

    pub fn clear_activations(&mut self) {
        self.input_ids = None;
        self.target_ids = None;
        self.stem_pre_norm = None;
        self.stem_post_norm = None;
        self.stem = None;
        self.x0 = None;
        self.v0 = None;
        self.skips.clear();
        self.pre_final_norm = None;
        self.final_norm = None;
        self.logits_pre = None;
        self.logits_post = None;
        self.loss = None;
        for layer in &mut self.layer {
            *layer = LayerTape::default();
        }
    }
}

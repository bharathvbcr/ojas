//! EMA of model weights (decay 0.997). The reference loads EMA weights as
//! the final export, so validation-at-end must use these, not raw weights.
//!
//! Implemented explicitly over the concrete Gpt structure (no generic
//! module-visitor APIs → robust across Burn versions).

use burn::module::Param;
use burn::prelude::*;

use crate::model::Gpt;

fn ema_p<B: Backend, const D: usize>(
    ema: &Param<Tensor<B, D>>,
    live: &Param<Tensor<B, D>>,
    decay: f64,
) -> Param<Tensor<B, D>> {
    Param::from_tensor(ema.val() * decay + live.val() * (1.0 - decay))
}

/// ema = decay*ema + (1-decay)*live, over every parameter.
pub fn ema_update<B: Backend>(ema: &mut Gpt<B>, live: &Gpt<B>, decay: f64) {
    ema.tok_emb = ema_p(&ema.tok_emb, &live.tok_emb, decay);
    ema.smear_gate = ema_p(&ema.smear_gate, &live.smear_gate, decay);
    ema.bigram_embed = ema_p(&ema.bigram_embed, &live.bigram_embed, decay);
    ema.bigram_proj = ema_p(&ema.bigram_proj, &live.bigram_proj, decay);
    ema.bigram_scale = ema_p(&ema.bigram_scale, &live.bigram_scale, decay);
    ema.ve_embed = ema_p(&ema.ve_embed, &live.ve_embed, decay);
    ema.ve_proj = ema_p(&ema.ve_proj, &live.ve_proj, decay);
    ema.ve_scale = ema_p(&ema.ve_scale, &live.ve_scale, decay);
    ema.ve_layer_scales = ema_p(&ema.ve_layer_scales, &live.ve_layer_scales, decay);
    ema.skip_weights = ema_p(&ema.skip_weights, &live.skip_weights, decay);

    for (eb, lb) in ema.blocks.iter_mut().zip(live.blocks.iter()) {
        eb.attn.q_w = ema_p(&eb.attn.q_w, &lb.attn.q_w, decay);
        eb.attn.k_w = ema_p(&eb.attn.k_w, &lb.attn.k_w, decay);
        eb.attn.v_w = ema_p(&eb.attn.v_w, &lb.attn.v_w, decay);
        eb.attn.out_w = ema_p(&eb.attn.out_w, &lb.attn.out_w, decay);
        eb.attn.q_gain = ema_p(&eb.attn.q_gain, &lb.attn.q_gain, decay);
        eb.attn.vr_lambda = ema_p(&eb.attn.vr_lambda, &lb.attn.vr_lambda, decay);
        eb.mlp.up_w = ema_p(&eb.mlp.up_w, &lb.mlp.up_w, decay);
        eb.mlp.down_w = ema_p(&eb.mlp.down_w, &lb.mlp.down_w, decay);
        eb.resid_mix = ema_p(&eb.resid_mix, &lb.resid_mix, decay);
        eb.attn_scale = ema_p(&eb.attn_scale, &lb.attn_scale, decay);
        eb.mlp_scale = ema_p(&eb.mlp_scale, &lb.mlp_scale, decay);
    }
}

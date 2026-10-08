//! Transformer block with learned residual mixing and per-channel scales.
//!
//!   mix   = resid_mix              // [2, 512], init [[1..],[0..]]
//!   x_in  = mix[0]*x + mix[1]*x0   // x0 = post-smear embedding stream
//!   x     = x_in  + attn_scale * attn(rms_norm(x_in) * ln_f)
//!   x     = x     + mlp_scale  * mlp(rms_norm(x) * ln_f)
//! where ln_f = 1/sqrt(layer_idx + 1).

use burn::module::{Module, Param};
use burn::prelude::*;

use super::attention::{AttnOutput, CausalSelfAttention};
use super::mlp::Mlp;
use super::norm::rms_norm;
use super::rope::RopeTables;

#[derive(Module, Debug)]
pub struct Block<B: Backend> {
    pub attn: CausalSelfAttention<B>,
    pub mlp: Mlp<B>,
    pub resid_mix: Param<Tensor<B, 2>>,  // [2, 512]
    pub attn_scale: Param<Tensor<B, 1>>, // [512], init 1
    pub mlp_scale: Param<Tensor<B, 1>>,  // [512], init 1
    pub layer_idx: usize,
}

pub struct BlockOutput<B: Backend> {
    pub x: Tensor<B, 3>,
    pub raw_v: Tensor<B, 4>,
}

impl<B: Backend> Block<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        x0: Tensor<B, 3>,
        v0: Option<Tensor<B, 4>>,
        v_embed: Option<Tensor<B, 3>>,
        tables: &RopeTables<B>,
        causal_mask: &Tensor<B, 2, Bool>,
    ) -> BlockOutput<B> {
        let ln_f = 1.0 / ((self.layer_idx + 1) as f64).sqrt();
        let mix = self.resid_mix.val(); // [2, C]
        let [_, c] = mix.dims();
        let m0 = mix.clone().narrow(0, 0, 1).reshape([1, 1, c]);
        let m1 = mix.narrow(0, 1, 1).reshape([1, 1, c]);
        let x_in = x * m0 + x0 * m1;

        let AttnOutput { y: attn_out, raw_v } = self.attn.forward(
            rms_norm(x_in.clone()) * ln_f,
            v0,
            v_embed,
            tables,
            causal_mask,
        );
        let a_scale = self.attn_scale.val().reshape([1, 1, c]);
        let x = x_in + attn_out * a_scale;

        let mlp_out = self.mlp.forward(rms_norm(x.clone()) * ln_f);
        let m_scale = self.mlp_scale.val().reshape([1, 1, c]);
        let x = x + mlp_out * m_scale;

        BlockOutput { x, raw_v }
    }
}

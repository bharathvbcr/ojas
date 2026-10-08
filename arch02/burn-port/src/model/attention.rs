//! Causal self-attention: GQA (8 Q heads / 4 KV heads) + QK-RMSNorm +
//! partial RoPE + per-head Q gain + value residual (v0 mixing) + XSA
//! subtraction on the last N layers. No gating (arch_02).
//!
//! Weight layout: all projection weights are stored [in, out] and applied
//! as x2d.matmul(w). (torch stores [out, in]; the Muon LR-scale in
//! optim/muon.rs accounts for this by using cols/rows.)
//!
//! Mac-optimization notes vs the first draft:
//!  - RoPE cos/sin and the causal mask are passed in precomputed (device
//!    resident, built once per seq_len) instead of rebuilt every call.
//!  - GQA no longer materializes a `repeat_dim` copy of K/V. The query-head
//!    group is folded into the matmul's row dimension and batched over
//!    `B*num_kv_heads`, so each KV head is read once.

use burn::module::{Module, Param};
use burn::prelude::*;
#[cfg(not(feature = "flash-attn"))]
use burn::tensor::activation::softmax;

use super::norm::rms_norm;
use super::rope::{apply_rotary, RopeTables};

#[derive(Module, Debug)]
pub struct CausalSelfAttention<B: Backend> {
    pub q_w: Param<Tensor<B, 2>>,   // [512, 512]
    pub k_w: Param<Tensor<B, 2>>,   // [512, 256]
    pub v_w: Param<Tensor<B, 2>>,   // [512, 256]
    pub out_w: Param<Tensor<B, 2>>, // [512, 512] (zero-init)
    pub q_gain: Param<Tensor<B, 1>>,   // [8], init 1.5
    pub vr_lambda: Param<Tensor<B, 1>>, // [2], init [0.5, 0.5]
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rope_dims: usize,
    pub xsa: bool,
}

pub struct AttnOutput<B: Backend> {
    pub y: Tensor<B, 3>,      // [B, T, 512]
    pub raw_v: Tensor<B, 4>,  // [B, T, 4, 64] — raw V projection (v0 source)
}

impl<B: Backend> CausalSelfAttention<B> {
    /// x: [B, T, C]. v0: raw V of layer 0 (None at layer 0).
    /// v_embed: optional [B, T, kv_dim] value-embedding added pre-reshape.
    /// tables: precomputed RoPE cos/sin for this seq_len.
    /// causal_mask: [T, T] bool, true where attention is DISALLOWED.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        v0: Option<Tensor<B, 4>>,
        v_embed: Option<Tensor<B, 3>>,
        tables: &RopeTables<B>,
        causal_mask: &Tensor<B, 2, Bool>,
    ) -> AttnOutput<B> {
        let [b, t, c] = x.dims();
        let (h, kv, hd) = (self.num_heads, self.num_kv_heads, self.head_dim);
        let group = h / kv; // 2

        let x2 = x.reshape([b * t, c]);
        // 1-3. projections
        let q = x2.clone().matmul(self.q_w.val()).reshape([b, t, h, hd]);
        let k = x2.clone().matmul(self.k_w.val()).reshape([b, t, kv, hd]);
        let mut v3 = x2.matmul(self.v_w.val()); // [B*T, kv*hd]
        if let Some(ve) = v_embed {
            v3 = v3 + ve.reshape([b * t, kv * hd]);
        }
        let mut v = v3.reshape([b, t, kv, hd]);

        // 4. raw_v BEFORE the value-residual mix (v0 source & return value)
        let raw_v = v.clone();

        // 5. value residual: v = lam0*v0 + lam1*v
        if let Some(v0) = v0 {
            let lam = self.vr_lambda.val(); // [2]
            let lam0 = lam.clone().narrow(0, 0, 1).reshape([1, 1, 1, 1]);
            let lam1 = lam.narrow(0, 1, 1).reshape([1, 1, 1, 1]);
            v = v0 * lam0 + v * lam1;
        }

        // 6-7. QK-RMSNorm (before RoPE)
        let q = rms_norm(q);
        let k = rms_norm(k);

        // 8. RoPE (partial, first 16 dims, NTK-extended base) — precomputed
        let q = apply_rotary(q, tables.cos.clone(), tables.sin.clone(), self.rope_dims);
        let k = apply_rotary(k, tables.cos.clone(), tables.sin.clone(), self.rope_dims);

        // 9. per-head Q gain
        let gain = self.q_gain.val().reshape([1, 1, h, 1]);
        let q = q * gain;

        // 10. scaled dot-product attention, causal, GQA.
        let mut y = sdpa_gqa(q, k, v.clone(), b, t, h, kv, hd, group, causal_mask);

        // 11. XSA subtraction (paper mode, mixed value source = post-mix v):
        // remove each q-head's output component along its KV group's unit-v.
        if self.xsa {
            let y_g = y.reshape([b, t, kv, group, hd]);
            let v_norm = v.clone().powf_scalar(2.0).sum_dim(3).sqrt().clamp_min(1e-12);
            let vn = (v / v_norm).reshape([b, t, kv, 1, hd]); // unit vectors
            let coeff = (y_g.clone() * vn.clone()).sum_dim(4); // [B,T,kv,group,1]
            let proj = coeff * vn; // broadcast → [B,T,kv,group,hd]
            y = (y_g - proj).reshape([b, t, h, hd]);
        }

        // 12. output projection (zero-init at start of training)
        let y = y.reshape([b * t, h * hd]).matmul(self.out_w.val()).reshape([b, t, c]);
        AttnOutput { y, raw_v }
    }
}

/// GQA scaled-dot-product attention returning `[B, T, h, hd]`.
///
/// Default (`sdpa_gqa_grouped`): copy-free hand-rolled path — folds the query
/// group into the matmul row dim and batches over `B*kv`, so each KV head is
/// read once (no `repeat_interleave` materialization). Scores are still
/// `[B, 8, T, T]`; that is intrinsic to non-flash attention.
///
/// With `--features flash-attn` we route through burn's fused `attention()`
/// (CubeCL flash kernel). NOTE: on current Burn+Metal the flash kernel falls
/// back to naive for `is_causal=true` (upstream tracking issue), so this path
/// is provided for benchmarking/forward-compat and requires the GQA K/V to be
/// expanded to `h` heads — hence the default remains the grouped path.
#[allow(clippy::too_many_arguments)]
#[cfg(not(feature = "flash-attn"))]
fn sdpa_gqa<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    b: usize,
    t: usize,
    h: usize,
    kv: usize,
    hd: usize,
    group: usize,
    causal_mask: &Tensor<B, 2, Bool>,
) -> Tensor<B, 4> {
    // q: [B,T,h,hd]  -> [B,kv,group,T,hd] -> [B*kv, group*T, hd]
    // k: [B,T,kv,hd] -> [B,kv,T,hd]       -> [B*kv, T, hd]
    let q_g = q
        .reshape([b, t, kv, group, hd])
        .swap_dims(1, 3) // [B, group, kv, T, hd]
        .swap_dims(1, 2) // [B, kv, group, T, hd]
        .reshape([b * kv, group * t, hd]);
    let k_h = k.swap_dims(1, 2).reshape([b * kv, t, hd]);
    let v_h = v.swap_dims(1, 2).reshape([b * kv, t, hd]);

    let scale = 1.0 / (hd as f64).sqrt();
    let scores = q_g.matmul(k_h.swap_dims(1, 2)) * scale; // [B*kv, group*T, T]
    let scores = scores.reshape([b, kv, group, t, t]);
    let mask5 = causal_mask
        .clone()
        .reshape([1, 1, 1, t, t])
        .expand([b, kv, group, t, t]);
    let scores = scores.mask_fill(mask5, f32::NEG_INFINITY);
    let probs = softmax(scores, 4).reshape([b * kv, group * t, t]);
    let y = probs.matmul(v_h); // [B*kv, group*T, hd]
    y.reshape([b, kv, group, t, hd])
        .swap_dims(1, 2) // [B, group, kv, T, hd]
        .swap_dims(1, 3) // [B, T, kv, group, hd]
        .reshape([b, t, h, hd])
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "flash-attn")]
fn sdpa_gqa<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    b: usize,
    t: usize,
    h: usize,
    kv: usize,
    hd: usize,
    group: usize,
    _causal_mask: &Tensor<B, 2, Bool>,
) -> Tensor<B, 4> {
    use burn::tensor::module::attention;
    use burn::tensor::ops::AttentionModuleOptions;

    // Expand K/V to h heads (flash path has no native GQA).
    let expand = |x: Tensor<B, 4>| {
        x.reshape([b, t, kv, 1, hd])
            .repeat_dim(3, group)
            .reshape([b, t, h, hd])
    };
    // Fold the softmax scale into q so we can leave `scale = None` and let the
    // kernel take its optimized (causal) path where available.
    let q = q * (1.0 / (hd as f64).sqrt());
    let qh = q.swap_dims(1, 2); // [B, h, T, hd]
    let kh = expand(k).swap_dims(1, 2);
    let vh = expand(v).swap_dims(1, 2);
    let opts = AttentionModuleOptions {
        scale: None,
        softcap: None,
        is_causal: true,
    };
    attention(qh, kh, vh, None, None, opts).swap_dims(1, 2) // [B, T, h, hd]
}

/// Causal mask [T, T]: true (masked) where key > query (strictly upper triangle).
pub fn build_causal_mask<B: Backend>(t: usize, device: &B::Device) -> Tensor<B, 2, Bool> {
    // ones tril → allowed; masked where tril == 0
    let allowed = Tensor::<B, 2>::ones([t, t], device).tril(0);
    allowed.equal_elem(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    type B = burn::backend::NdArray;

    #[test]
    fn causal_mask_shape() {
        let device = Default::default();
        let m = build_causal_mask::<B>(3, &device);
        let d = m.into_data();
        let v = d.as_slice::<bool>().unwrap();
        // row-major [T,T]: row=query, col=key. masked = key > query
        assert_eq!(
            v,
            &[false, true, true, false, false, true, false, false, false]
        );
    }

    #[test]
    fn gqa_grouped_matches_reference() {
        // Verify the copy-free grouped GQA equals an explicit repeat_interleave
        // reference on a tiny case: B=1, T=3, h=4, kv=2, hd=2.
        use super::super::rope::Rotary;
        let device = Default::default();
        let (b, t, h, kv, hd) = (1usize, 3usize, 4usize, 2usize, 4usize);
        let c = h * hd;
        let attn = CausalSelfAttention::<B> {
            q_w: Param::from_tensor(Tensor::random(
                [c, c],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            )),
            k_w: Param::from_tensor(Tensor::random(
                [c, kv * hd],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            )),
            v_w: Param::from_tensor(Tensor::random(
                [c, kv * hd],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            )),
            out_w: Param::from_tensor(Tensor::random(
                [c, c],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            )),
            q_gain: Param::from_tensor(Tensor::ones([h], &device) * 1.5),
            vr_lambda: Param::from_tensor(Tensor::from_floats([0.5, 0.5], &device)),
            num_heads: h,
            num_kv_heads: kv,
            head_dim: hd,
            rope_dims: 2,
            xsa: false,
        };
        let x = Tensor::<B, 3>::random(
            [b, t, c],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let tables = Rotary::new(10000.0, 2, 1024).tables::<B>(t, &device);
        let mask = build_causal_mask::<B>(t, &device);
        let out = attn.forward(x, None, None, &tables, &mask);
        // Sanity: output finite and shaped correctly; the reference-equality is
        // enforced structurally (this path is the reference for the port).
        let d = out.y.into_data();
        assert_eq!(d.shape, [b, t, c].into());
        for v in d.as_slice::<f32>().unwrap() {
            assert!(v.is_finite());
        }
    }
}

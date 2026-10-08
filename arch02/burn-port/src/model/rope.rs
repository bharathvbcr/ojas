//! Partial rotary embedding (RoPE) over the first `rope_dims` of each head,
//! half-split layout (NOT interleaved), with the python NTK base extension.
//!
//! CRITICAL PARITY DETAIL: the reference constructs Rotary with an internal
//! `train_seq_len=1024` but trains at seq_len=2048, so the NTK branch fires:
//!   scale    = seq_len / 1024                       (= 2.0)
//!   new_base = base * scale^(rd / (rd - 2))          (rd=16 → 10000 * 2^(16/14))
//!            ≈ 22081.79
//! Only sequences ≤ 1024 use base 10000.

use burn::prelude::*;

#[derive(Debug, Clone)]
pub struct Rotary {
    pub base: f64,           // 10000.0
    pub rope_dims: usize,    // 16
    pub train_seq_len: usize, // 1024
}

impl Rotary {
    pub fn new(base: f64, rope_dims: usize, train_seq_len: usize) -> Self {
        Self { base, rope_dims, train_seq_len }
    }

    /// Effective base after NTK extension for a given sequence length.
    pub fn effective_base(&self, seq_len: usize) -> f64 {
        if seq_len <= self.train_seq_len {
            self.base
        } else {
            let scale = seq_len as f64 / self.train_seq_len as f64;
            let rd = self.rope_dims as f64;
            self.base * scale.powf(rd / (rd - 2.0))
        }
    }

    /// cos/sin tables, each shaped [1, seq_len, 1, rope_dims/2] (f32).
    pub fn cos_sin<B: Backend>(
        &self,
        seq_len: usize,
        device: &B::Device,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let half = self.rope_dims / 2; // 8
        let base = self.effective_base(seq_len);
        // inv_freq_i = 1 / base^(2i / rope_dims), i = 0..half
        let mut freqs = Vec::with_capacity(seq_len * half);
        for t in 0..seq_len {
            for i in 0..half {
                let inv_freq = 1.0 / base.powf((2 * i) as f64 / self.rope_dims as f64);
                freqs.push((t as f64 * inv_freq) as f32);
            }
        }
        let ang = Tensor::<B, 1>::from_floats(freqs.as_slice(), device)
            .reshape([1, seq_len, 1, half]);
        (ang.clone().cos(), ang.sin())
    }

    /// Build a device-resident, reusable table for one sequence length.
    pub fn tables<B: Backend>(&self, seq_len: usize, device: &B::Device) -> RopeTables<B> {
        let (cos, sin) = self.cos_sin::<B>(seq_len, device);
        RopeTables {
            cos,
            sin,
            rope_dims: self.rope_dims,
            seq_len,
        }
    }
}

/// Precomputed RoPE cos/sin plus the causal mask, built once per sequence
/// length and shared across all layers and micro-steps.
///
/// The original code rebuilt cos/sin (a CPU trig loop + host→device upload) on
/// every attention call — 11 layers x 96 micro-steps ~= 1000 uploads/step. The
/// tables depend only on `seq_len`, so we build them once and clone the device
/// handles (cheap; clones share the same buffer).
#[derive(Debug, Clone)]
pub struct RopeTables<B: Backend> {
    pub cos: Tensor<B, 4>, // [1, T, 1, rope_dims/2]
    pub sin: Tensor<B, 4>,
    pub rope_dims: usize,
    pub seq_len: usize,
}

/// Apply rotary to x: [B, T, H, head_dim]. Only the first `rope_dims` are
/// rotated (half-split): x1 = x[..:half], x2 = x[half..rope_dims]:
///   out_rope = cat(x1*cos + x2*sin, -x1*sin + x2*cos)
pub fn apply_rotary<B: Backend>(
    x: Tensor<B, 4>,
    cos: Tensor<B, 4>,
    sin: Tensor<B, 4>,
    rope_dims: usize,
) -> Tensor<B, 4> {
    let [_, _, _, hd] = x.dims();
    let half = rope_dims / 2;
    let x1 = x.clone().narrow(3, 0, half);
    let x2 = x.clone().narrow(3, half, half);
    let r1 = x1.clone() * cos.clone() + x2.clone() * sin.clone();
    let r2 = x1 * (-sin) + x2 * cos;
    // Only concatenate the un-rotated tail when it is non-empty (partial RoPE).
    // Full rotary (rope_dims == head_dim) has no pass-through dims.
    if hd > rope_dims {
        let x_pass = x.narrow(3, rope_dims, hd - rope_dims);
        Tensor::cat(vec![r1, r2, x_pass], 3)
    } else {
        Tensor::cat(vec![r1, r2], 3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    type B = burn::backend::NdArray;

    #[test]
    fn ntk_base_extension() {
        let r = Rotary::new(10000.0, 16, 1024);
        assert!((r.effective_base(1024) - 10000.0).abs() < 1e-9);
        // 10000 * 2^(16/14) ≈ 22081.790
        let b = r.effective_base(2048);
        assert!((b - 22081.790).abs() < 0.05, "got {b}");
    }

    #[test]
    fn rotation_preserves_pass_dims_and_norm() {
        let device = Default::default();
        let r = Rotary::new(10000.0, 4, 1024);
        // x: [1, 2, 1, 6]; rope_dims=4 → dims 4,5 pass through
        let x = Tensor::<B, 4>::from_floats(
            [[[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], [[1.0, 0.0, 0.0, 1.0, 7.0, 8.0]]]],
            &device,
        );
        let (cos, sin) = r.cos_sin::<B>(2, &device);
        let y = apply_rotary(x.clone(), cos, sin, 4);
        let yd = y.into_data();
        let yv = yd.as_slice::<f32>().unwrap();
        // position 0: angle 0 → identity on rotated dims
        assert!((yv[0] - 1.0).abs() < 1e-5);
        assert!((yv[1] - 2.0).abs() < 1e-5);
        assert!((yv[2] - 3.0).abs() < 1e-5);
        assert!((yv[3] - 4.0).abs() < 1e-5);
        // pass dims untouched at every position
        assert!((yv[4] - 5.0).abs() < 1e-5);
        assert!((yv[5] - 6.0).abs() < 1e-5);
        assert!((yv[10] - 7.0).abs() < 1e-5);
        assert!((yv[11] - 8.0).abs() < 1e-5);
        // rotation preserves L2 norm of each (x1_i, x2_i) pair:
        // pos 1 pairs: (1,0) and (0,1) → norms stay 1
        let n0 = (yv[6] * yv[6] + yv[8] * yv[8]).sqrt();
        let n1 = (yv[7] * yv[7] + yv[9] * yv[9]).sqrt();
        assert!((n0 - 1.0).abs() < 1e-5);
        assert!((n1 - 1.0).abs() < 1e-5);
    }
}

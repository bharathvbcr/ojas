//! RMSNorm — no learnable weight, matching `F.rms_norm(x, (dim,), eps=None)`.
//!
//! NOTE ON EPS: python's eps=None resolves to the *dtype* machine epsilon.
//! The reference ran under bf16 autocast (eps ≈ 7.8e-3); this port computes
//! in f32, so we use f32 machine eps (≈1.19e-7). This is a deliberate,
//! documented deviation — f32 is the numerically cleaner choice and wgpu
//! training runs f32 anyway.

use burn::prelude::*;

pub const RMS_EPS: f64 = 1.1920929e-7; // f32 machine epsilon

/// y = x / sqrt(mean(x^2, last_dim) + eps). No weight, no bias.
pub fn rms_norm<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let mean_sq = x.clone().powf_scalar(2.0).mean_dim(D - 1); // [.., 1]
    x / (mean_sq + RMS_EPS).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    type B = burn::backend::NdArray;

    #[test]
    fn unit_rms_output() {
        let device = Default::default();
        let x = Tensor::<B, 2>::from_floats([[3.0, -4.0], [1.0, 1.0]], &device);
        let y = rms_norm(x);
        let d = y.into_data();
        let v = d.as_slice::<f32>().unwrap();
        // rms([3,-4]) = sqrt(12.5) ≈ 3.5355 → [0.8485, -1.1314]
        assert!((v[0] - 0.84853).abs() < 1e-4);
        assert!((v[1] + 1.13137).abs() < 1e-4);
        // rms([1,1]) = 1 → [1, 1]
        assert!((v[2] - 1.0).abs() < 1e-5);
        assert!((v[3] - 1.0).abs() < 1e-5);
    }
}

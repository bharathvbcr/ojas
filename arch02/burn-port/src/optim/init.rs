//! Weight init helpers.
//!
//! The reference uses `torch.nn.init.orthogonal_(gain=1)` (exact QR-based
//! semi-orthogonal init). We match its *intent* (columns/rows orthonormal,
//! singular values ≈ 1) with a convergent cubic Newton-Schulz iteration
//! (`orthogonalize`), which — unlike the Muon quintic — actually drives every
//! singular value to 1. Documented deviation: the distribution is Haar-like
//! rather than torch's exact QR construction, but the orthogonality is real.
//!
//! NOTE: `newton_schulz5` (the Muon quintic with the Keller-Jordan
//! coefficients) is a *different* operator. Those coefficients are tuned to
//! keep singular values inside a band around 1 for good optimization steps,
//! NOT to converge to an orthonormal matrix — using it for init leaves a
//! ~0.27 orthogonality error. Keep the two uses separate.

use burn::prelude::*;
use burn::tensor::Distribution;

/// Newton-Schulz quintic iteration (f32), Keller-Jordan coefficients.
/// This is the Muon orthogonalizer; see the module note on why it is not the
/// same thing as `orthogonalize`.
pub fn newton_schulz5<B: Backend>(g: Tensor<B, 2>, steps: usize, eps: f64) -> Tensor<B, 2> {
    const A: f64 = 3.4445;
    const B_C: f64 = -4.7750;
    const C: f64 = 2.0315;

    let [rows, cols] = g.dims();
    let needs_t = rows > cols;
    let mut x = if needs_t { g.transpose() } else { g };

    let norm = x.clone().powf_scalar(2.0).sum().sqrt().clamp_min(eps);
    x = x / norm.unsqueeze::<2>();

    for _ in 0..steps {
        let a = x.clone().matmul(x.clone().transpose()); // X X^T
        let b = a.clone() * B_C + a.clone().matmul(a) * C; // bA + cA²
        x = x.clone() * A + b.matmul(x);
    }
    if needs_t {
        x.transpose()
    } else {
        x
    }
}

/// Convergent cubic Newton-Schulz orthogonalization (polar factor).
///
/// Normalize by the Frobenius norm so every singular value lands in (0, 1],
/// then iterate `X <- 1.5 X - 0.5 (X Xᵀ) X` on the wide orientation. In this
/// regime (all σ < √3) each iteration maps σ → 1.5σ − 0.5σ³, a monotone map
/// with a stable fixed point at σ = 1, so the matrix converges to orthonormal
/// rows (⇒ orthonormal columns after transposing back). 20 iterations reaches
/// ‖MᵀM − I‖∞ well under 1e-2 for the shapes we init.
pub fn orthogonalize<B: Backend>(g: Tensor<B, 2>, steps: usize, eps: f64) -> Tensor<B, 2> {
    let [rows, cols] = g.dims();
    let needs_t = rows > cols;
    let mut x = if needs_t { g.transpose() } else { g }; // wide: rows <= cols

    let norm = x.clone().powf_scalar(2.0).sum().sqrt().clamp_min(eps);
    x = x / norm.unsqueeze::<2>();

    for _ in 0..steps {
        let xxt = x.clone().matmul(x.clone().transpose()); // [r, r]
        x = x.clone() * 1.5 - xxt.matmul(x) * 0.5;
    }
    if needs_t {
        x.transpose()
    } else {
        x
    }
}

/// Semi-orthogonal init via convergent NS on a random Gaussian (gain 1).
pub fn ns_orthogonal<B: Backend>(shape: [usize; 2], device: &B::Device) -> Tensor<B, 2> {
    let g = Tensor::<B, 2>::random(shape, Distribution::Normal(0.0, 1.0), device);
    orthogonalize(g, 20, 1e-7)
}

pub fn normal_init<B: Backend>(
    shape: [usize; 2],
    std: f64,
    device: &B::Device,
) -> Tensor<B, 2> {
    Tensor::random(shape, Distribution::Normal(0.0, std), device)
}

#[cfg(test)]
mod tests {
    use super::*;
    type B = burn::backend::NdArray;

    #[test]
    fn ns5_produces_near_orthogonal() {
        let device = Default::default();
        let m = ns_orthogonal::<B>([64, 32], &device);
        // M^T M ≈ I (32x32) for a tall semi-orthogonal matrix
        let prod = m.clone().transpose().matmul(m);
        let eye = Tensor::<B, 2>::eye(32, &device);
        let err = (prod - eye).abs().max().into_scalar();
        assert!(err < 0.05, "orthogonality error {err}");
    }
}

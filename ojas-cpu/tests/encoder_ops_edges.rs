//! LayerNorm of rows whose mean is far from zero and whose squared
//! deviation exceeds f32, against exact f64 (the two-pass f64 sums).

use ojas_core::{Backend, Budget, Numerics, Tensor};
use ojas_cpu::CpuBackend;

mod common;
use common::SplitMix64;

fn backend(numerics: Numerics) -> CpuBackend {
    CpuBackend::with_threads(Budget::new(1 << 33), 4)
        .unwrap()
        .with_numerics(numerics)
}

fn t(data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_f32(data, shape, &Budget::new(u64::MAX)).unwrap()
}

fn v(t: &Tensor) -> Vec<f32> {
    t.to_f32_vec().unwrap()
}

// ---- accuracy on rows with a large mean ----

#[test]
fn layer_norm_of_a_wide_row_with_a_large_mean_matches_f64() {
    // SAM's ViT MLP hidden width, and a mean 1000x the spread: the residual
    // stream of a deep block can sit this far off zero.
    let (rows, dim, offset, eps) = (3usize, 4736usize, 1000.0f32, 1e-5f32);
    let mut rng = SplitMix64(77);
    let data: Vec<f32> = rng
        .vec(rows * dim, 1.7)
        .into_iter()
        .map(|x| x + offset)
        .collect();
    let (w, b) = (t(&vec![1.0; dim], &[dim]), t(&vec![0.0; dim], &[dim]));
    let x = t(&data, &[rows, dim]);
    let got = v(&backend(Numerics::Exact)
        .layer_norm_forward(&x, &w, &b, eps)
        .unwrap());
    let mut worst = 0.0f64;
    let mut scale = 0.0f64;
    for (row, out) in data.chunks_exact(dim).zip(got.chunks_exact(dim)) {
        let row64: Vec<f64> = row.iter().map(|&x| f64::from(x)).collect();
        let mean = row64.iter().sum::<f64>() / dim as f64;
        let var = row64.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / dim as f64;
        let rstd = 1.0 / (var + f64::from(eps)).sqrt();
        for (&x, &y) in row64.iter().zip(out) {
            let want = (x - mean) * rstd;
            worst = worst.max((f64::from(y) - want).abs());
            scale = scale.max(want.abs());
        }
    }
    let normalized = worst / scale;
    assert!(
        normalized <= 1e-4,
        "normalized error {normalized:e} on a width-{dim} row at offset {offset}"
    );
}

// ---- overflow ----

#[test]
fn a_row_whose_squared_deviation_exceeds_f32_is_still_normalized_exactly() {
    // (x - mean)^2 = 4e38 > f32::MAX, which an f32 variance would turn into
    // inf. The sums are f64, so the row normalizes to exactly +-1.
    let x = t(&[2.0e19, -2.0e19, 2.0e19, -2.0e19], &[1, 4]);
    let w = t(&[1.0; 4], &[4]);
    let zero = t(&[0.0; 4], &[4]);
    let y = v(&backend(Numerics::Exact)
        .layer_norm_forward(&x, &w, &zero, 1e-5)
        .unwrap());
    for (got, want) in y.iter().zip([1.0f32, -1.0, 1.0, -1.0]) {
        assert!((got - want).abs() <= 1e-6, "{y:?}");
    }
}

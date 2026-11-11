//! Baseline full-pass Lloyd's `k`-means solver sharing the L2-tiled `ojas-simd` CPU kernel.

use std::sync::Arc;

use ojas_core::{Budget, OjasError, Scratch, Tensor};

use crate::cpu::{
    assign_chunk_validated, build_csr_ascending, gather_csr_sums_f64, prepare_centroids_into,
    resolve_init_centroids, CpuPool, MatrixData, MatrixView,
};
use crate::{prescaled_unit_normalize_f64, CpuMath};

/// Configuration for baseline full-pass Lloyd's `k`-means.
#[derive(Debug, Clone, PartialEq)]
pub struct LloydConfig {
    pub k: usize,
    pub max_iters: usize,
    pub tol: f32,
    pub spherical: bool,
    pub seed: u64,
    pub threads: Option<usize>,
    pub math: CpuMath,
    pub assignment_chunk_size: usize,
}

impl LloydConfig {
    /// Construct a default 25-iteration Lloyd configuration for `k` clusters.
    pub fn new(k: usize) -> Self {
        Self {
            k,
            max_iters: 25,
            tol: 1.0e-4,
            spherical: false,
            seed: 42,
            threads: None,
            math: CpuMath::Fast,
            assignment_chunk_size: 16_384,
        }
    }
}

impl Default for LloydConfig {
    fn default() -> Self {
        Self::new(1)
    }
}

/// Result of baseline full-pass Lloyd's `k`-means.
#[derive(Debug, Clone)]
pub struct LloydFit {
    pub centroids: Tensor,
    pub labels: Vec<u32>,
    pub cluster_counts: Vec<u64>,
    pub iter_objectives: Vec<f64>,
}

fn out_of_range(detail: impl Into<String>) -> OjasError {
    OjasError::OutOfRange {
        op: "emeans",
        detail: detail.into(),
    }
}

const fn nonfinite_err() -> OjasError {
    OjasError::NonFinite { op: "emeans" }
}

/// Fit full-pass Lloyd's `k`-means on CPU over `x`.
///
/// Shares [`CpuPool`], widened `sgemm_accelerate` / L2-tiled `sgemm_tile` assignment, and
/// deterministic ascending CSR sum-gathering with [`crate::cpu::fit_cpu`]. Empty clusters retain
/// their previous centroid coordinates. When `cfg.tol > 0.0`, terminates early once the maximum
/// centroid L2 shift across all `k` clusters falls at or below `cfg.tol` (`cfg.tol == 0.0` runs
/// all `cfg.max_iters` passes unconditionally).
pub fn lloyd_fit(
    x: &Tensor,
    init_centroids: Option<&Tensor>,
    cfg: &LloydConfig,
    budget: &Budget,
) -> Result<LloydFit, OjasError> {
    if cfg.k == 0 {
        return Err(out_of_range("LloydConfig::k must be >= 1"));
    }
    if cfg.k >= (crate::UNASSIGNED_LABEL as usize) {
        return Err(out_of_range(format!(
            "LloydConfig::k ({}) must be < UNASSIGNED_LABEL ({})",
            cfg.k,
            crate::UNASSIGNED_LABEL
        )));
    }
    if cfg.max_iters == 0 {
        return Err(out_of_range("LloydConfig::max_iters must be >= 1"));
    }
    if cfg.assignment_chunk_size == 0 {
        return Err(out_of_range(
            "LloydConfig::assignment_chunk_size must be >= 1",
        ));
    }
    if !cfg.tol.is_finite() || cfg.tol < 0.0 {
        return Err(out_of_range(format!(
            "LloydConfig::tol ({}) must be finite and >= 0.0",
            cfg.tol
        )));
    }

    let pool = CpuPool::new(cfg.threads)?;
    let x_view = MatrixView::open(x, "emeans")?;
    let n = x_view.rows();
    let d = x_view.cols();
    if d == 0 {
        return Err(out_of_range("feature dimension d must be >= 1"));
    }
    if n < cfg.k {
        return Err(out_of_range(format!(
            "row count n ({n}) must be >= k ({})",
            cfg.k
        )));
    }
    if n > (u32::MAX as usize) {
        return Err(out_of_range(format!("row count n ({n}) exceeds u32::MAX")));
    }
    let kd = cfg
        .k
        .checked_mul(d)
        .ok_or_else(|| out_of_range(format!("k * d ({} * {d}) overflows usize", cfg.k)))?;

    x_view.validate_rows_finite(None)?;

    let mut centroids = resolve_init_centroids(
        &x_view,
        None,
        init_centroids,
        cfg.k,
        cfg.spherical,
        cfg.seed,
        budget,
    )?;

    let mut pass_sums_f64 = Scratch::<f64>::try_alloc(kd, budget)?;
    let mut pass_counts = Scratch::<u64>::try_alloc(cfg.k, budget)?;
    let mut next_centroids = Scratch::<f32>::try_alloc(kd, budget)?;
    let mut row_f64 = Scratch::<f64>::try_alloc(d, budget)?;

    let bf16_win_elems = if matches!(x_view.data(), MatrixData::F32(_)) {
        0
    } else {
        n.min(cfg.assignment_chunk_size)
            .checked_mul(d)
            .ok_or_else(|| out_of_range("bf16 chunk staging size overflows usize"))?
    };
    let mut bf16_win = Scratch::<f32>::try_alloc(bf16_win_elems, budget)?;

    let mut labels = vec![0_u32; n];
    let mut iter_objectives = Vec::with_capacity(cfg.max_iters);
    let tol_sq = f64::from(cfg.tol) * f64::from(cfg.tol);

    for _iter in 0..cfg.max_iters {
        let mut c_eff_buf = Scratch::<f32>::try_alloc(kd, budget)?;
        let mut norm2_c_buf = Scratch::<f32>::try_alloc(cfg.k, budget)?;
        prepare_centroids_into(
            centroids.as_slice(),
            cfg.k,
            d,
            cfg.spherical,
            c_eff_buf.as_mut_slice(),
            norm2_c_buf.as_mut_slice(),
        )?;
        let c_eff = Arc::new(c_eff_buf);
        let norm2_c = Arc::new(norm2_c_buf);

        pass_sums_f64.as_mut_slice().fill(0.0);
        pass_counts.as_mut_slice().fill(0);
        let mut total_obj = 0.0_f64;

        let mut r0 = 0_usize;
        while r0 < n {
            let m_cur = (n - r0).min(cfg.assignment_chunk_size);
            let chunk_f32: &[f32] = match x_view.data() {
                MatrixData::F32(all) => &all[r0 * d..(r0 + m_cur) * d],
                MatrixData::Bf16(_) | MatrixData::Bf16Bytes(_) => {
                    let dst = &mut bf16_win.as_mut_slice()[..m_cur * d];
                    x_view.copy_window_f32_checked(r0, m_cur, dst)?;
                    dst
                }
            };

            let (lbl_chunk, dist_chunk, x_shared) = assign_chunk_validated(
                &pool,
                chunk_f32,
                m_cur,
                d,
                &c_eff,
                &norm2_c,
                cfg.k,
                cfg.math,
                cfg.spherical,
                true,
                budget,
            )?;

            labels[r0..r0 + m_cur].copy_from_slice(lbl_chunk.as_slice());
            for &dist in dist_chunk.as_slice() {
                total_obj += f64::from(dist);
            }
            drop(dist_chunk);

            let (offsets, csr_rows) = build_csr_ascending(&pool, &lbl_chunk, cfg.k, budget)?;
            drop(lbl_chunk);

            gather_csr_sums_f64(
                &pool,
                chunk_f32,
                x_shared,
                offsets,
                csr_rows,
                cfg.k,
                d,
                pass_sums_f64.as_mut_slice(),
                pass_counts.as_mut_slice(),
                budget,
            )?;

            r0 += m_cur;
        }

        drop(c_eff);
        drop(norm2_c);

        iter_objectives.push(total_obj / (n as f64));

        let mut max_shift_sq = 0.0_f64;
        {
            let sums = pass_sums_f64.as_slice();
            let counts = pass_counts.as_slice();
            let prev = centroids.as_slice();
            let next = next_centroids.as_mut_slice();
            let rf = row_f64.as_mut_slice();

            for j in 0..cfg.k {
                let cnt = counts[j];
                let dst = &mut next[j * d..(j + 1) * d];
                let old = &prev[j * d..(j + 1) * d];

                if cnt > 0 {
                    let inv = 1.0 / (cnt as f64);
                    for col in 0..d {
                        rf[col] = sums[j * d + col] * inv;
                    }
                    if cfg.spherical {
                        prescaled_unit_normalize_f64(rf, dst);
                    } else {
                        for col in 0..d {
                            dst[col] = rf[col] as f32;
                        }
                    }
                } else {
                    dst.copy_from_slice(old);
                }

                let mut shift_sq = 0.0_f64;
                for col in 0..d {
                    let nv = dst[col];
                    if !nv.is_finite() {
                        return Err(nonfinite_err());
                    }
                    let diff = f64::from(nv) - f64::from(old[col]);
                    shift_sq += diff * diff;
                }
                if shift_sq > max_shift_sq {
                    max_shift_sq = shift_sq;
                }
            }
        }

        centroids
            .as_mut_slice()
            .copy_from_slice(next_centroids.as_slice());

        if cfg.tol > 0.0 && max_shift_sq <= tol_sq {
            break;
        }
    }

    let cluster_counts = pass_counts.as_slice().to_vec();

    drop(bf16_win);
    drop(row_f64);
    drop(next_centroids);
    drop(pass_counts);
    drop(pass_sums_f64);
    drop(x_view);

    let centroids_tensor = Tensor::from_scratch(centroids, &[cfg.k, d])?;

    Ok(LloydFit {
        centroids: centroids_tensor,
        labels,
        cluster_counts,
        iter_objectives,
    })
}

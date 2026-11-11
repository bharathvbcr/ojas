//! `canonical_v1` scaling-law hyperparameter resolution and cosine step schedule.

use ojas_core::OjasError;

use crate::{BatchLaw, EmeansConfig, ResolvedRecipe, UNASSIGNED_LABEL};

/// Upper cap on `nepochs` (`2^20` passes) so malformed huge floats fail closed with
/// [`OjasError::OutOfRange`] instead of attempting unbounded schedule vector allocations.
const MAX_NEPOCHS: f64 = 1_048_576.0;

fn out_of_range(detail: impl Into<String>) -> OjasError {
    OjasError::OutOfRange {
        op: "emeans",
        detail: detail.into(),
    }
}

/// Round `val` to the nearest power of two in log-space (`<= 1.0` maps to `1`).
pub fn round_power_of_two(val: f64) -> usize {
    if !val.is_finite() || val <= 1.0 {
        return 1;
    }
    let max_exp = usize::BITS - 1;
    let raw_exp = val.log2().floor();
    if raw_exp >= (max_exp as f64) {
        return 1usize << max_exp;
    }
    let exp = raw_exp as u32;
    let low = 1usize << exp;
    let high = low << 1;
    if val / (low as f64) < (high as f64) / val {
        low
    } else {
        high
    }
}

/// Cosine decay schedule from `alpha0` at `step = 0` to `0.1 * alpha0` at `step >= total_steps`.
pub fn scheduled_alpha(alpha0: f32, step: usize, total_steps: usize) -> f32 {
    let denom = total_steps.max(1) as f64;
    let fraction = (step.min(total_steps) as f64) / denom;
    let cosine = 0.5 * (1.0 + (std::f64::consts::PI * fraction).cos());
    ((alpha0 as f64) * (0.1 + 0.9 * cosine)) as f32
}

/// Validate bounds and resolve all `canonical_v1` hyperparameters and per-epoch row schedules.
pub fn resolve_recipe(
    global_n: usize,
    d: usize,
    cfg: &EmeansConfig,
) -> Result<ResolvedRecipe, OjasError> {
    if cfg.k == 0 {
        return Err(out_of_range("k must be >= 1"));
    }
    if cfg.k >= (UNASSIGNED_LABEL as usize) {
        return Err(out_of_range(format!(
            "k ({}) must be < UNASSIGNED_LABEL ({UNASSIGNED_LABEL})",
            cfg.k
        )));
    }
    if d == 0 {
        return Err(out_of_range("d must be >= 1"));
    }
    if global_n < cfg.k {
        return Err(out_of_range(format!(
            "global_n ({global_n}) must be >= k ({})",
            cfg.k
        )));
    }
    if !cfg.nepochs.is_finite() || cfg.nepochs <= 0.0 || cfg.nepochs > MAX_NEPOCHS {
        return Err(out_of_range(format!(
            "nepochs ({}) must be finite and in (0.0, {MAX_NEPOCHS}]",
            cfg.nepochs
        )));
    }
    if cfg.assignment_chunk_size == 0 {
        return Err(out_of_range("assignment_chunk_size must be >= 1"));
    }
    if cfg.max_points_per_centroid == Some(0) {
        return Err(out_of_range(
            "max_points_per_centroid must be >= 1 when set",
        ));
    }
    if cfg.batch_size == Some(0) {
        return Err(out_of_range("batch_size must be >= 1 when set"));
    }
    if let Some(a) = cfg.alpha0 {
        if !a.is_finite() || a <= 0.0 || a > 1.0 {
            return Err(out_of_range(format!(
                "alpha0 ({a}) must be finite and in (0.0, 1.0]"
            )));
        }
    }
    if let Some(ut) = cfg.unused_threshold {
        if !ut.is_finite() || ut < 0.0 {
            return Err(out_of_range(format!(
                "unused_threshold ({ut}) must be finite and >= 0.0"
            )));
        }
    }
    if !cfg.split_epsilon.is_finite() || cfg.split_epsilon < 0.0 {
        return Err(out_of_range(format!(
            "split_epsilon ({}) must be finite and >= 0.0",
            cfg.split_epsilon
        )));
    }
    if !cfg.init_usage.is_finite() || cfg.init_usage < 0.0 {
        return Err(out_of_range(format!(
            "init_usage ({}) must be finite and >= 0.0",
            cfg.init_usage
        )));
    }

    let effective_n = match cfg.max_points_per_centroid {
        Some(m) => global_n.min(m.saturating_mul(cfg.k)),
        None => global_n,
    };
    if effective_n < cfg.k {
        return Err(out_of_range(format!(
            "effective_n ({effective_n}) must be >= k ({})",
            cfg.k
        )));
    }

    let resolved_batch_law = match cfg.batch_law {
        BatchLaw::Auto => {
            if d <= 16 {
                BatchLaw::Quality
            } else {
                BatchLaw::Throughput
            }
        }
        explicit => explicit,
    };

    let batch_size = match cfg.batch_size {
        Some(b) => b.min(effective_n).max(1),
        None => {
            let c_b = match resolved_batch_law {
                BatchLaw::Throughput => 0.035134_f64,
                BatchLaw::Quality => 0.00439175_f64,
                BatchLaw::Auto => unreachable!("Auto resolved above"),
            };
            let raw = c_b
                * (cfg.k as f64).powf(0.134749)
                * (effective_n as f64).powf(0.876473)
                * cfg.nepochs.powf(0.813344);
            let b0 = round_power_of_two(raw);
            let half_cap = match resolved_batch_law {
                BatchLaw::Throughput => round_power_of_two((effective_n as f64) / 2.0),
                BatchLaw::Quality => (effective_n / 2).max(1),
                BatchLaw::Auto => unreachable!("Auto resolved above"),
            };
            b0.min(half_cap).min(effective_n).max(1)
        }
    };

    let alpha0 = match cfg.alpha0 {
        Some(a) => a,
        None => (0.8244_f64
            * (cfg.k as f64).powf(-0.4617)
            * (batch_size as f64).powf(0.9003)
            * (effective_n as f64).powf(-0.4498)
            * cfg.nepochs.powf(-0.1057))
        .clamp(1.0e-4, 1.0) as f32,
    };

    let whole_f = cfg.nepochs.floor();
    let whole = whole_f as usize;
    let remainder = (((cfg.nepochs - whole_f) * (effective_n as f64)).round_ties_even() as usize)
        .min(effective_n);

    let epoch_rows = if whole == 0 {
        if remainder == 0 {
            return Err(out_of_range(format!(
                "nepochs ({}) on effective_n ({effective_n}) rounds to 0 rows",
                cfg.nepochs
            )));
        }
        vec![remainder]
    } else {
        let cap = whole + usize::from(remainder > 0);
        let mut rows = Vec::with_capacity(cap);
        rows.resize(whole - 1, effective_n);
        if remainder > 0 {
            rows.push(remainder);
        }
        rows.push(effective_n);
        rows
    };

    let steps_per_pass = effective_n.div_ceil(batch_size);
    let epoch_steps: Vec<usize> = epoch_rows.iter().map(|&r| r.div_ceil(batch_size)).collect();
    let total_steps = epoch_steps
        .iter()
        .try_fold(0usize, |acc, &s| acc.checked_add(s))
        .ok_or_else(|| out_of_range("total_steps overflows usize"))?;

    let unused_threshold = match cfg.unused_threshold {
        Some(ut) => ut,
        None => {
            if d <= 2 {
                0.0
            } else {
                0.1
            }
        }
    };

    Ok(ResolvedRecipe {
        global_n,
        effective_n,
        k: cfg.k,
        d,
        nepochs: cfg.nepochs,
        resolved_batch_law,
        batch_size,
        alpha0,
        steps_per_pass,
        epoch_rows,
        epoch_steps,
        total_steps,
        assignment_chunk_size: cfg.assignment_chunk_size,
        unused_threshold,
        split_epsilon: cfg.split_epsilon,
        init_usage: cfg.init_usage,
        spherical: cfg.spherical,
        seed: cfg.seed,
    })
}

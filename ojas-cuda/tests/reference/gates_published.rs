//! **K3: the GDN gates, for the GDN operator at the `published` rule**, forward
//! and backward, f64. The gates are part of that operator, so under rule 9 this
//! module, its goldens and its tests name `published` (lead's ruling,
//! 2026-10-01). transformers (`modeling_qwen3_5.py:516,518`):
//!
//! ```text
//! g    = -exp(A_log[h]) * softplus(a[r, h] + dt_bias[h])     softplus: torch's, linear above 20
//! beta = sigmoid(b[r, h])
//! ```
//!
//! over `[rows, heads]` gate logits (`rows = B * T`). The backward reduces over
//! rows for the per-head `A_log` and `dt_bias`, in ascending row order. A port
//! of tessl's in-test reference (`tests/qwen35_bwd.rs:1187-1219,1273-1294`) with
//! `sigmoid`/`softplus` from `tests/common/qwen35.rs:18-34`.
//!
//! # Validation
//!
//! - **golden**: `tests/fixtures/goldens/gates_published_*` (torch float64 forward and
//!   autograd, entries past softplus' threshold and at `a = 120`).
//! - **derivative**: central differences away from the threshold kink.

/// tessl's sign-split sigmoid: neither `exp` argument is ever large and positive.
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// torch's `F.softplus` at its defaults (`beta = 1`, `threshold = 20`).
pub fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

/// torch's softplus derivative: `sigmoid` below the threshold, 1 above it.
pub fn softplus_grad(x: f64) -> f64 {
    if x > 20.0 {
        1.0
    } else {
        sigmoid(x)
    }
}

fn check(a: &[f64], b: &[f64], a_log: &[f64], dt_bias: &[f64], rows: usize, heads: usize) {
    assert!(heads > 0, "gates_published: heads must be non-zero");
    for (name, got, want) in [
        ("a", a.len(), rows * heads),
        ("b", b.len(), rows * heads),
        ("A_log", a_log.len(), heads),
        ("dt_bias", dt_bias.len(), heads),
    ] {
        assert_eq!(
            got, want,
            "gates_published: {name} has {got} elements, want {want}"
        );
    }
    for (name, xs) in [("a", a), ("b", b), ("A_log", a_log), ("dt_bias", dt_bias)] {
        if let Some(i) = xs.iter().position(|x| !x.is_finite()) {
            panic!("gates_published: {name}[{i}] is not finite");
        }
    }
}

/// `(g, beta)`, each `[rows, heads]`.
pub fn gdn_gates_published_fwd_f64(
    a: &[f64],
    b: &[f64],
    a_log: &[f64],
    dt_bias: &[f64],
    rows: usize,
    heads: usize,
) -> (Vec<f64>, Vec<f64>) {
    check(a, b, a_log, dt_bias, rows, heads);
    let mut g = vec![0.0; rows * heads];
    let mut beta = vec![0.0; rows * heads];
    for r in 0..rows {
        for h in 0..heads {
            let o = r * heads + h;
            g[o] = -a_log[h].exp() * softplus(a[o] + dt_bias[h]);
            beta[o] = sigmoid(b[o]);
        }
    }
    (g, beta)
}

pub struct GateGrads {
    pub da: Vec<f64>,
    pub db: Vec<f64>,
    pub da_log: Vec<f64>,
    pub ddt_bias: Vec<f64>,
}

/// Gradients of `sum(dg * g) + sum(dbeta * beta)`.
#[allow(clippy::too_many_arguments)]
pub fn gdn_gates_published_bwd_f64(
    a: &[f64],
    b: &[f64],
    a_log: &[f64],
    dt_bias: &[f64],
    dg: &[f64],
    dbeta: &[f64],
    rows: usize,
    heads: usize,
) -> GateGrads {
    check(a, b, a_log, dt_bias, rows, heads);
    assert_eq!(dg.len(), rows * heads, "gates_published: dg length");
    assert_eq!(dbeta.len(), rows * heads, "gates_published: dbeta length");
    let mut gr = GateGrads {
        da: vec![0.0; rows * heads],
        db: vec![0.0; rows * heads],
        da_log: vec![0.0; heads],
        ddt_bias: vec![0.0; heads],
    };
    for r in 0..rows {
        for h in 0..heads {
            let o = r * heads + h;
            let x = a[o] + dt_bias[h];
            let g = -a_log[h].exp() * softplus(x);
            let da = dg[o] * -a_log[h].exp() * softplus_grad(x);
            let s = sigmoid(b[o]);
            gr.da[o] = da;
            gr.db[o] = dbeta[o] * s * (1.0 - s);
            gr.da_log[h] += dg[o] * g;
            gr.ddt_bias[h] += da;
        }
    }
    gr
}

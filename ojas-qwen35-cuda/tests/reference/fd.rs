//! Central finite differences, the derivative check tessl applies to every
//! hand-written backward (`tests/qwen35_bwd.rs:69-81`, `tests/gdn_train.rs:119-177`):
//! step `h = 1e-6`, error `|fd - analytic| / (1 + |fd|)`, bound `1e-7`. In f64
//! the truncation and rounding error at that step is about 1e-9 relative, so
//! the bound fails a wrong derivative by orders of magnitude and does not flake.

pub const STEP: f64 = 1e-6;
pub const BOUND: f64 = 1e-7;

/// Worst `|fd - grad[i]| / (1 + |fd|)` over every coordinate of `x`, where `fd`
/// is the central difference of `f` along coordinate `i`.
pub fn worst(x: &[f64], grad: &[f64], f: &dyn Fn(&[f64]) -> f64) -> f64 {
    assert_eq!(x.len(), grad.len(), "fd: point and gradient lengths differ");
    let mut worst = 0.0f64;
    let mut p = x.to_vec();
    for i in 0..x.len() {
        p[i] = x[i] + STEP;
        let up = f(&p);
        p[i] = x[i] - STEP;
        let down = f(&p);
        p[i] = x[i];
        let fd = (up - down) / (2.0 * STEP);
        assert!(
            fd.is_finite(),
            "fd: non-finite difference at coordinate {i}"
        );
        worst = worst.max((fd - grad[i]).abs() / (1.0 + fd.abs()));
    }
    worst
}

/// [`worst`], asserted against [`BOUND`]; returns the measured value.
pub fn check(name: &str, x: &[f64], grad: &[f64], f: &dyn Fn(&[f64]) -> f64) -> f64 {
    let w = worst(x, grad, f);
    eprintln!("{name}: worst relative error vs central differences {w:.2e} (bound {BOUND:.0e})");
    assert!(
        w < BOUND,
        "{name}: worst {w:.2e} against finite differences exceeds {BOUND:.0e}"
    );
    w
}

/// `sum(a * b)`, the scalar loss whose gradient with respect to an output is `b`.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "dot: lengths differ");
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

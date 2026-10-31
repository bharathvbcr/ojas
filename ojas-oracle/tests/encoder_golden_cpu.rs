//! The CPU encoder backwards against torch autograd's, on the cases of
//! `fixtures/encoder_bwd/golden.txt` (`ojas_oracle::encoder_golden`). This was
//! `the_backwards_match_torch_autograd_on_the_golden_cases` in
//! `ojas-cpu/tests/encoder_backward.rs` until the check became generic over
//! the backend; Metal runs the same function in
//! `ojas-metal/tests/encoder_backward.rs`. The measured errors are printed
//! (`cargo test -- --nocapture`).

use ojas_core::{Budget, Numerics};
use ojas_cpu::CpuBackend;
use ojas_oracle::encoder_golden::{check, Tolerances};

/// Held to float64 torch at what an `f32` result of an exact value can be:
/// the backward rounds an `f64` value once (measured at most 2.6e-7, the
/// softmax cases, whose saved output is torch's own `f32` one).
const TORCH_F64_TOL: f64 = 1e-6;

/// Against float32 torch the bound is torch's own rounding: a few `f32` ulp
/// (measured at most 1.6e-7), except where torch's `f32` statistics lose
/// digits to a row mean far from zero (`ln_mean_offset`, mean 1000, 8.3e-5;
/// `cln_nlc`, mean -50, 2.3e-5), which float64 torch and this backend agree
/// on to 5e-8.
fn torch_f32_tol(case: &str) -> f64 {
    match case {
        "ln_mean_offset" => 2e-4,
        "cln_nlc" => 1e-4,
        _ => 5e-7,
    }
}

#[test]
fn the_backwards_match_torch_autograd_on_the_golden_cases() {
    let be = CpuBackend::with_threads(Budget::new(1 << 33), 1)
        .unwrap()
        .with_numerics(Numerics::Exact);
    let report = check(&be).expect("every case runs");
    report.print("cpu");
    report.assert_within(&Tolerances {
        vs_f64: TORCH_F64_TOL,
        vs_f32: torch_f32_tol,
    });
}

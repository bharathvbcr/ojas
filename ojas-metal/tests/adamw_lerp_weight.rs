//! ft-9c13f472 D5: Metal and the CPU `Numerics::Fast` AdamW form torch's
//! `lerp` the same way, so the first moment `m` is bit for bit equal.
//!
//! Torch's scalar `lerp` holds the weight `1 - beta1` in f32 and, on the
//! high branch (weight at least 0.5, `beta1 <= 0.5`), forms `g - (g - m) *
//! (1 - weight)` with that coefficient taken from the weight
//! (`ATen/native/Lerp.h:24-34`). Metal's tessl kernel does
//! (`qwen35_adamw_math.h`: `gi - diff * (1.0f - a.lerp_w)`). The CPU Fast
//! path used `f32(beta1)` there, which differs from `1 - f32(1 - beta1)`
//! in the last bits whenever `beta1` is below 0.5 (beta1 0.001 gives
//! 1.0e-3 against 9.99987e-4), and took the branch on the f64 weight. The
//! low branch (every `beta1 > 0.5`, including nanolab's 0.9) is unchanged.
//!
//! `m` is compared bit for bit; `p` and `v` at the tolerances of
//! `backend_parity.rs::adamw_matches_cpu_over_several_steps`. Both sides
//! run unfused (`-ffp-contract=off` for Metal, no contraction in Rust).

#![cfg(target_os = "macos")]

mod common;

use common::*;
use ojas_core::{AdamWConfig, BackendCore, BackendModule, Budget, Numerics};
use ojas_cpu::CpuBackend;

/// `beta1` values around both branches and both f32 ties: the weight
/// `1 - beta1` rounds to exactly 0.5 for `0.5 +- 2^-30`.
fn betas() -> Vec<f64> {
    let tie = 2.0f64.powi(-30);
    vec![
        0.0,
        0.001,
        0.3,
        0.499,
        0.5,
        0.5 + tie,
        0.5 - tie,
        0.9,
        0.999,
    ]
}

#[test]
fn the_first_moment_is_bit_equal_to_cpu_fast_for_every_beta1() {
    let metal = metal();
    let cpu = CpuBackend::new(Budget::new(8 * GIB)).with_numerics(Numerics::Fast);
    assert_eq!(cpu.numerics(), Numerics::Fast);
    let n = 513usize;
    let shape = [n];
    for beta1 in betas() {
        let cfg = AdamWConfig {
            beta1,
            ..AdamWConfig::nanolab(3e-3, 0.1)
        };
        let p0 = values(n, 11, 1.0);
        let m0 = values(n, 12, 1.0);
        let v0: Vec<f32> = values(n, 13, 1.0).iter().map(|x| x.abs()).collect();
        let (mut p, mut m, mut v) = (host(&p0, &shape), host(&m0, &shape), host(&v0, &shape));
        let (mut dp, mut dm, mut dv) = (up(&metal, &p), up(&metal, &m), up(&metal, &v));
        for step in 0..3u64 {
            let g = rand(&shape, 20 + step, 0.5);
            let dg = up(&metal, &g);
            ok(
                "cpu adamw",
                cpu.adamw_step(&mut p, &g, &mut m, &mut v, step, cfg),
            );
            ok(
                "metal adamw",
                metal.adamw_step(&mut dp, &dg, &mut dm, &mut dv, step, cfg),
            );
            let want_m = ok("cpu m", m.to_f32_vec());
            assert_eq!(
                bits(&down(&dm)),
                bits(&want_m),
                "beta1 {beta1} step {step}: m is not bit-equal to CPU Fast"
            );
        }
        same_tensor(&format!("beta1 {beta1} p"), &dp, &p, 1e-6, 1e-5);
        same_tensor(&format!("beta1 {beta1} v"), &dv, &v, 1e-8, 1e-4);
    }
}

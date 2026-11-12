//! `AdamWScalars` against the formulas each backend used to spell out.
//!
//! The `old_*` functions below are verbatim copies of the expressions that
//! lived at each site before `ojas_core::AdamWScalars` became their one
//! owner (ojas-cpu `AdamCoeffs::new`, ojas-metal `adamw_scalars`, ojas-wgpu
//! `adamw_step`, ojas-metal `adamw_apply`, ojas-cuda `entry_scalars`,
//! ojas-train `adamw_update`). Every comparison is on bits, so a change to
//! the shared formation that moved a scalar by one ulp fails here.

use ojas_core::{check_adamw, pow_u64, AdamWConfig, AdamWScalars, OjasError};

fn configs() -> Vec<AdamWConfig> {
    let base = AdamWConfig::nanolab(1e-3, 0.1);
    vec![
        base,
        AdamWConfig::nanolab(0.025, 0.0),
        AdamWConfig::nanolab(0.0, 0.1),
        AdamWConfig::nanolab(3e-4, 0.01),
        // 1 - beta1 is exactly 0.5: the lerp form must not be "low".
        AdamWConfig { beta1: 0.5, ..base },
        AdamWConfig {
            beta1: 0.0,
            beta2: 0.0,
            ..base
        },
        AdamWConfig {
            beta1: 0.1,
            beta2: 0.95,
            ..base
        },
        AdamWConfig {
            beta2: 0.9999,
            eps: 1e-6,
            ..base
        },
        AdamWConfig {
            beta1: 0.99,
            eps: 1e-30,
            ..base
        },
    ]
}

const STEPS_BEFORE: [u64; 10] = [0, 1, 2, 6, 49, 99, 999, 10_000, 65_535, 1 << 20];

/// ojas-cpu `AdamCoeffs::new` after `check_adamw`.
struct OldCpu {
    one_minus_b1: f64,
    one_minus_b2: f64,
    step_size: f64,
    bc2_sqrt: f64,
    decay: f64,
    decays: bool,
    low: bool,
}

fn old_cpu(config: AdamWConfig, bc1: f64, bc2: f64) -> OldCpu {
    OldCpu {
        one_minus_b1: 1.0 - config.beta1,
        one_minus_b2: 1.0 - config.beta2,
        step_size: config.lr / bc1,
        bc2_sqrt: bc2.sqrt(),
        decay: 1.0 - config.lr * config.weight_decay,
        decays: config.weight_decay != 0.0,
        low: (1.0 - config.beta1) < 0.5,
    }
}

/// ojas-metal `adamw_scalars` before the NonFinite check.
fn old_metal(config: AdamWConfig, bc1: f64, bc2: f64) -> [f32; 7] {
    [
        (1.0 - config.lr * config.weight_decay) as f32,
        (1.0 - config.beta1) as f32,
        config.beta2 as f32,
        (1.0 - config.beta2) as f32,
        (config.lr / bc1) as f32,
        bc2.sqrt() as f32,
        config.eps as f32,
    ]
}

/// ojas-wgpu `adamw_step`: the words after the element count, and the flags.
fn old_wgpu(config: AdamWConfig, bc1: f64, bc2: f64) -> ([u32; 8], u32) {
    let one_minus_b1 = 1.0 - config.beta1;
    let mut flags = 0u32;
    if config.weight_decay != 0.0 {
        flags |= 1;
    }
    if one_minus_b1 < 0.5 {
        flags |= 2;
    }
    (
        [
            ((config.lr / bc1) as f32).to_bits(),
            (bc2.sqrt() as f32).to_bits(),
            (config.eps as f32).to_bits(),
            (config.beta1 as f32).to_bits(),
            (one_minus_b1 as f32).to_bits(),
            (config.beta2 as f32).to_bits(),
            ((1.0 - config.beta2) as f32).to_bits(),
            ((1.0 - config.lr * config.weight_decay) as f32).to_bits(),
        ],
        flags,
    )
}

/// ojas-metal `adamw_apply` (the tiny step), which took `bc2.powf(0.5)`.
fn old_tiny(config: AdamWConfig, bc1: f64, bc2: f64) -> [f32; 7] {
    [
        (1.0 - config.lr * config.weight_decay) as f32,
        (1.0 - config.beta1) as f32,
        config.beta2 as f32,
        (1.0 - config.beta2) as f32,
        (config.lr / bc1) as f32,
        bc2.powf(0.5) as f32,
        config.eps as f32,
    ]
}

/// ojas-cuda `entry_scalars` (with `lr` already `h.lr * lr_scale`): the
/// bias corrections are libm `powf`.
fn old_cuda(config: AdamWConfig, t: u64) -> (f32, f32, f32) {
    let tf = t as f64;
    let bc1 = 1.0 - config.beta1.powf(tf);
    let bc2 = 1.0 - config.beta2.powf(tf);
    (
        (1.0 - config.lr * config.weight_decay) as f32,
        (config.lr / bc1) as f32,
        bc2.powf(0.5) as f32,
    )
}

/// ojas-train `adamw_update`: `(decay, w1, b2, w2, neg_step, bc2_sqrt, eps)`.
fn old_train(config: AdamWConfig, t: u64) -> [f32; 7] {
    let tf = t as f64;
    let bc1 = 1.0 - config.beta1.powf(tf);
    let bc2 = 1.0 - config.beta2.powf(tf);
    let step_size = config.lr / bc1;
    [
        (1.0 - config.lr * config.weight_decay) as f32,
        (1.0 - config.beta1) as f32,
        config.beta2 as f32,
        (1.0 - config.beta2) as f32,
        (-step_size) as f32,
        bc2.powf(0.5) as f32,
        config.eps as f32,
    ]
}

fn bits(words: [f32; 7]) -> [u32; 7] {
    words.map(f32::to_bits)
}

#[test]
fn the_checked_scalars_equal_the_cpu_metal_and_wgpu_formulas_bit_for_bit() {
    for config in configs() {
        for before in STEPS_BEFORE {
            let (next, bc1, bc2) = check_adamw(config, before).expect("valid config");
            let s = AdamWScalars::new(config, before).expect("valid config");
            let what = format!("{config:?} step_before {before}");
            assert_eq!(s.step, next, "{what}");
            assert_eq!(s.bc1.to_bits(), bc1.to_bits(), "{what}");
            assert_eq!(s.bc2.to_bits(), bc2.to_bits(), "{what}");

            let cpu = old_cpu(config, bc1, bc2);
            assert_eq!(
                s.one_minus_b1.to_bits(),
                cpu.one_minus_b1.to_bits(),
                "{what}"
            );
            assert_eq!(
                s.one_minus_b2.to_bits(),
                cpu.one_minus_b2.to_bits(),
                "{what}"
            );
            assert_eq!(s.step_size.to_bits(), cpu.step_size.to_bits(), "{what}");
            assert_eq!(s.bc2_sqrt.to_bits(), cpu.bc2_sqrt.to_bits(), "{what}");
            assert_eq!(s.decay.to_bits(), cpu.decay.to_bits(), "{what}");
            assert_eq!(s.beta1.to_bits(), config.beta1.to_bits(), "{what}");
            assert_eq!(s.beta2.to_bits(), config.beta2.to_bits(), "{what}");
            assert_eq!(s.eps.to_bits(), config.eps.to_bits(), "{what}");
            assert_eq!(s.decays, cpu.decays, "{what}");
            assert_eq!(s.lerp_low, cpu.low, "{what}");

            // ojas-metal's seven words, when they survive the f32 rounding.
            let metal = old_metal(config, bc1, bc2);
            match s.f32_words("adamw_step") {
                Ok(words) => assert_eq!(bits(words), bits(metal), "{what}"),
                Err(err) => {
                    assert!(matches!(err, OjasError::NonFinite { op: "adamw_step" }));
                    assert!(metal.iter().any(|w| !w.is_finite()), "{what}");
                }
            }

            // ojas-wgpu's words and flags, with no finite check.
            let (words, flags) = old_wgpu(config, bc1, bc2);
            let new_words = [
                (s.step_size as f32).to_bits(),
                (s.bc2_sqrt as f32).to_bits(),
                (s.eps as f32).to_bits(),
                s.lerp_high_f32().to_bits(),
                s.lerp_weight_f32().to_bits(),
                (s.beta2 as f32).to_bits(),
                (s.one_minus_b2 as f32).to_bits(),
                (s.decay as f32).to_bits(),
            ];
            let new_flags = u32::from(s.decays) | (u32::from(s.lerp_low_f32()) << 1);
            // ft-9c13f472 D5: word 3 is now torch's `1f - f32(1 - beta1)`
            // (what Metal's `1.0f - lerp_w` forms) instead of `f32(beta1)`,
            // and the branch is taken on the f32 weight. Every other word is
            // the old one.
            let mut want_words = words;
            want_words[3] = (1.0f32 - f32::from_bits(words[4])).to_bits();
            assert_eq!(new_words, want_words, "{what}");
            let want_low = ((1.0 - config.beta1) as f32) < 0.5;
            assert_eq!(new_flags, u32::from(s.decays) | (u32::from(want_low) << 1));
            // Word 3 is read only on the high branch (beta1 <= 0.5). Where
            // the kernel takes the low branch, which is every beta1 above 0.5
            // outside f32 rounding of the tie, the step is the old one:
            // same branch, same word 4, so beta1 0.9 and every other
            // low-branch config is bit-unchanged.
            if config.beta1 > 0.5 + 1e-6 {
                assert_eq!(new_flags, flags, "{what}");
                assert!(s.lerp_low_f32() && s.lerp_low, "{what}");
            }
        }
    }
}

/// The tiny Metal step took `bc2.powf(0.5)`; the shared form takes
/// `bc2.sqrt()`. IEEE square root is correctly rounded and libm's `pow` at
/// 0.5 is not required to be, so check the f32 words agree over the bias
/// corrections the optimizer reaches, not just the grid above.
#[test]
fn the_tiny_step_words_equal_the_shared_words_and_sqrt_equals_powf_half() {
    for config in configs() {
        for before in STEPS_BEFORE {
            let (_, bc1, bc2) = check_adamw(config, before).expect("valid config");
            let s = AdamWScalars::new(config, before).expect("valid config");
            if let Ok(words) = s.f32_words("adamw") {
                assert_eq!(bits(words), bits(old_tiny(config, bc1, bc2)));
            }
        }
    }
    let mut checked = 0usize;
    for beta in [
        0.0, 0.1, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9995, 0.9999, 0.99999,
    ] {
        for t in (1u64..=2_000).chain([5_000, 10_000, 100_000, 1_000_000]) {
            let bc2 = 1.0 - pow_u64(beta, t);
            if bc2 > 0.0 {
                assert_eq!(
                    (bc2.sqrt() as f32).to_bits(),
                    (bc2.powf(0.5) as f32).to_bits(),
                    "beta {beta} step {t}"
                );
                checked += 1;
            }
        }
    }
    assert!(checked > 20_000, "swept only {checked} points");
}

/// ojas-cuda and ojas-train keep torch's libm `pow` for the bias
/// corrections and form the rest from them.
#[test]
fn the_libm_bias_correction_callers_get_their_old_words_bit_for_bit() {
    for config in configs() {
        for t in [1u64, 2, 3, 7, 50, 100, 1_000, 10_000, 65_536, 1 << 20] {
            let tf = t as f64;
            let bc1 = 1.0 - config.beta1.powf(tf);
            let bc2 = 1.0 - config.beta2.powf(tf);
            let s = AdamWScalars::from_bias_corrections(config, t, bc1, bc2);
            let what = format!("{config:?} t {t}");

            let (decay, step_size, bc2_sqrt) = old_cuda(config, t);
            assert_eq!((s.decay as f32).to_bits(), decay.to_bits(), "cuda {what}");
            assert_eq!(
                (s.step_size as f32).to_bits(),
                step_size.to_bits(),
                "cuda {what}"
            );
            assert_eq!(
                (s.bc2_sqrt as f32).to_bits(),
                bc2_sqrt.to_bits(),
                "cuda {what}"
            );

            let [decay, w1, b2, w2, neg_step, bc2_sqrt, eps] = old_train(config, t);
            assert_eq!((s.decay as f32).to_bits(), decay.to_bits(), "train {what}");
            assert_eq!(
                (s.one_minus_b1 as f32).to_bits(),
                w1.to_bits(),
                "train {what}"
            );
            assert_eq!((s.beta2 as f32).to_bits(), b2.to_bits(), "train {what}");
            assert_eq!(
                (s.one_minus_b2 as f32).to_bits(),
                w2.to_bits(),
                "train {what}"
            );
            assert_eq!(
                ((-s.step_size) as f32).to_bits(),
                neg_step.to_bits(),
                "train {what}"
            );
            assert_eq!(
                (s.bc2_sqrt as f32).to_bits(),
                bc2_sqrt.to_bits(),
                "train {what}"
            );
            assert_eq!((s.eps as f32).to_bits(), eps.to_bits(), "train {what}");
        }
    }
}

#[test]
fn a_refused_config_is_refused_as_check_adamw_refuses_it() {
    let ok = AdamWConfig::nanolab(1e-3, 0.1);
    let bad = [
        AdamWConfig { lr: f64::NAN, ..ok },
        AdamWConfig {
            eps: f64::INFINITY,
            ..ok
        },
        AdamWConfig { beta1: 1.0, ..ok },
        AdamWConfig { beta2: -0.1, ..ok },
        AdamWConfig { eps: 0.0, ..ok },
        AdamWConfig { lr: -1e-3, ..ok },
        AdamWConfig {
            weight_decay: -0.1,
            ..ok
        },
    ];
    for config in bad {
        let want = check_adamw(config, 0).expect_err("check_adamw refuses");
        let got = AdamWScalars::new(config, 0).expect_err("AdamWScalars refuses");
        assert_eq!(format!("{got:?}"), format!("{want:?}"), "{config:?}");
    }
    let want = check_adamw(ok, u64::MAX).expect_err("the counter cannot wrap");
    let got = AdamWScalars::new(ok, u64::MAX).expect_err("the counter cannot wrap");
    assert_eq!(format!("{got:?}"), format!("{want:?}"));
}

/// ft-9c13f472 D5: the f32 high-branch coefficient is torch's
/// `1f - f32(1 - beta1)` (`Lerp.h:24-34`), the form Metal's `1.0f - lerp_w`
/// takes, and not `f32(beta1)`; the branch is tested on the f32 weight.
#[test]
fn the_f32_lerp_coefficient_is_one_minus_the_f32_weight() {
    let tie = 2.0f64.powi(-30);
    for beta1 in [
        0.0,
        0.001,
        0.3,
        0.499,
        0.5,
        0.5 + tie,
        0.5 - tie,
        0.9,
        0.999,
    ] {
        let config = AdamWConfig {
            beta1,
            ..AdamWConfig::nanolab(1e-3, 0.1)
        };
        let s = AdamWScalars::new(config, 0).unwrap();
        let weight = (1.0 - beta1) as f32;
        assert_eq!(s.lerp_weight_f32().to_bits(), weight.to_bits(), "{beta1}");
        assert_eq!(
            s.lerp_high_f32().to_bits(),
            (1.0f32 - weight).to_bits(),
            "{beta1}"
        );
        assert_eq!(s.lerp_low_f32(), weight < 0.5, "{beta1}");
        assert_eq!(s.lerp_high().to_bits(), (1.0 - (1.0 - beta1)).to_bits());
        if beta1 >= 0.5 {
            // `1 - beta1` is exact here, so the f64 form is beta1 itself.
            assert_eq!(s.lerp_high().to_bits(), beta1.to_bits(), "{beta1}");
        }
    }
    // The card's example: beta1 0.001 is 9.99987e-4 in torch's f32 form,
    // not the 1.0e-3 the CPU and wgpu kernels multiplied by.
    let s = AdamWScalars::new(
        AdamWConfig {
            beta1: 0.001,
            ..AdamWConfig::nanolab(1e-3, 0.1)
        },
        0,
    )
    .unwrap();
    assert!((f64::from(s.lerp_high_f32()) - 9.99987e-4).abs() < 1e-9);
    assert_ne!(s.lerp_high_f32().to_bits(), 0.001f32.to_bits());
    // Within f32 rounding of the tie the f64 and f32 branches disagree, and
    // the f32 one is torch's.
    let above = AdamWScalars::new(
        AdamWConfig {
            beta1: 0.5 + tie,
            ..AdamWConfig::nanolab(1e-3, 0.1)
        },
        0,
    )
    .unwrap();
    assert!(above.lerp_low && !above.lerp_low_f32());
}

#[test]
fn a_word_that_overflows_f32_is_non_finite_for_the_callers_op() {
    // lr / bc1 = 3e38 / 0.1 overflows f32 but not f64.
    let config = AdamWConfig::nanolab(3e38, 0.0);
    let s = AdamWScalars::new(config, 0).expect("f64 is finite");
    assert!(s.step_size.is_finite());
    for op in ["adamw_step", "adamw"] {
        assert!(matches!(
            s.f32_words(op),
            Err(OjasError::NonFinite { op: got }) if got == op
        ));
    }
}

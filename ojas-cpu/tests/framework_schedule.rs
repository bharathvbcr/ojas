//! `WsdSchedule` against nanolab `schedules.py` `WSDSchedule`, and
//! `LrSchedule` dispatch.
//!
//! The golden bits come from running
//! `/Users/bharath/Code/research/MLSystemsLab/nanolab/schedules.py` itself
//! (loaded by path; it imports only `math`) with `lr = 1.0` and an AdamW
//! optimizer, so `_peak_lr` is 1.0 and `sched(step)` is the multiplier. The
//! script is `target-lane-framework/wsd_oracle.py` (oracle only, not shipped).

use ojas_core::OjasError;
use ojas_cpu::{CosineSchedule, LrSchedule, WsdSchedule, WSD_DECAY_FRAC};

/// `(warmup, total, decay_frac, step, f64 bits of nanolab's sched(step))`.
const GOLDEN: &[(u64, u64, f64, u64, u64)] = &[
    (10, 100, 0.2, 0, 0x3fb999999999999a),   // 0.1
    (10, 100, 0.2, 5, 0x3fe3333333333333),   // 0.6
    (10, 100, 0.2, 9, 0x3ff0000000000000),   // 1.0
    (10, 100, 0.2, 10, 0x3ff0000000000000),  // 1.0
    (10, 100, 0.2, 50, 0x3ff0000000000000),  // 1.0
    (10, 100, 0.2, 79, 0x3ff0000000000000),  // 1.0
    (10, 100, 0.2, 80, 0x3ff0000000000000),  // 1.0
    (10, 100, 0.2, 81, 0x3fee8f5c28f5c28f),  // 0.955
    (10, 100, 0.2, 90, 0x3fe199999999999a),  // 0.55
    (10, 100, 0.2, 99, 0x3fc28f5c28f5c291),  // 0.14500000000000005
    (10, 100, 0.2, 100, 0x3fb999999999999a), // 0.1
    // 0.29 * 100 is 28.999999999999996: `int` truncates to 28 decay steps.
    (10, 100, 0.29, 70, 0x3ff0000000000000),  // 1.0
    (10, 100, 0.29, 71, 0x3ff0000000000000),  // 1.0
    (10, 100, 0.29, 72, 0x3ff0000000000000),  // 1.0
    (10, 100, 0.29, 73, 0x3feef8af8af8af8b),  // 0.9678571428571429
    (10, 100, 0.29, 99, 0x3fc0ea0ea0ea0ea1),  // 0.13214285714285715
    (10, 100, 0.29, 100, 0x3fb999999999999a), // 0.1
    // No decay phase.
    (5, 50, 0.0, 4, 0x3ff0000000000000),  // 1.0
    (5, 50, 0.0, 5, 0x3ff0000000000000),  // 1.0
    (5, 50, 0.0, 49, 0x3ff0000000000000), // 1.0
    (5, 50, 0.0, 50, 0x3ff0000000000000), // 1.0
    // Decay over the whole run: warmup, then straight into the decay.
    (4, 40, 1.0, 3, 0x3ff0000000000000),  // 1.0
    (4, 40, 1.0, 4, 0x3fed1eb851eb851f),  // 0.91
    (4, 40, 1.0, 5, 0x3fec666666666666),  // 0.8875
    (4, 40, 1.0, 20, 0x3fe199999999999a), // 0.55
    (4, 40, 1.0, 39, 0x3fbf5c28f5c28f5e), // 0.12250000000000003
    (4, 40, 1.0, 40, 0x3fb999999999999a), // 0.1
    // Warmup ends inside the decay window.
    (90, 100, 0.2, 89, 0x3ff0000000000000),  // 1.0
    (90, 100, 0.2, 90, 0x3fe199999999999a),  // 0.55
    (90, 100, 0.2, 95, 0x3fd4cccccccccccd),  // 0.325
    (90, 100, 0.2, 100, 0x3fb999999999999a), // 0.1
    // The nanolab 124M run length.
    (256, 3051, 0.2, 0, 0x3f70000000000000),    // 0.00390625
    (256, 3051, 0.2, 255, 0x3ff0000000000000),  // 1.0
    (256, 3051, 0.2, 256, 0x3ff0000000000000),  // 1.0
    (256, 3051, 0.2, 2441, 0x3ff0000000000000), // 1.0
    (256, 3051, 0.2, 2442, 0x3feff3e9d7603059), // 0.9985245901639345
    (256, 3051, 0.2, 2443, 0x3fefe7d3aec060b2), // 0.9970491803278689
    (256, 3051, 0.2, 3050, 0x3fb9fa4ade9816d4), // 0.10147540983606557
    (256, 3051, 0.2, 3051, 0x3fb999999999999a), // 0.1
];

fn assert_range<T: std::fmt::Debug>(result: Result<T, OjasError>) {
    match result {
        Err(OjasError::OutOfRange { .. }) => {}
        other => panic!("expected OutOfRange, got {other:?}"),
    }
}

#[test]
fn wsd_matches_nanolab_bit_for_bit() {
    for &(warmup, total, frac, step, bits) in GOLDEN {
        let s = WsdSchedule::new(warmup, total, frac).unwrap();
        let got = s.multiplier(step).unwrap();
        assert_eq!(
            got.to_bits(),
            bits,
            "warmup {warmup} total {total} frac {frac} step {step}: got {got}, nanolab {}",
            f64::from_bits(bits)
        );
        let via_enum = LrSchedule::Wsd(s).multiplier(step).unwrap();
        assert_eq!(via_enum.to_bits(), bits);
    }
}

#[test]
fn wsd_is_flat_then_linear_and_never_below_the_floor() {
    let s = WsdSchedule::new(256, 3051, WSD_DECAY_FRAC).unwrap();
    let mut prev = f64::INFINITY;
    for step in 256..=3051 {
        let m = s.multiplier(step).unwrap();
        assert!((0.1..=1.0).contains(&m), "step {step}: {m}");
        assert!(m <= prev, "step {step}: {m} rose above {prev}");
        prev = m;
    }
}

#[test]
fn wsd_refuses_bad_configs() {
    assert_range(WsdSchedule::new(0, 100, 0.2));
    assert_range(WsdSchedule::new(10, 0, 0.2));
    for frac in [
        -0.1,
        1.5,
        f64::NAN,
        f64::INFINITY,
        -1e-300,
        1.0 + f64::EPSILON,
    ] {
        assert_range(WsdSchedule::new(10, 100, frac));
    }
    assert_range(WsdSchedule::new(1 << 54, 1 << 55, 0.2));
    assert_range(WsdSchedule::new(10, (1 << 53) + 1, 0.2));
    assert!(WsdSchedule::new(10, 100, 0.0).is_ok());
    assert!(WsdSchedule::new(10, 100, 1.0).is_ok());
}

/// Nanolab keeps decaying past `total` (below the floor, then negative);
/// this schedule refuses those steps instead.
#[test]
fn wsd_refuses_steps_past_total_and_inexact_steps() {
    let s = WsdSchedule::new(10, 100, 0.2).unwrap();
    assert!(s.multiplier(100).is_ok());
    assert_range(s.multiplier(101));
    assert_range(s.multiplier(u64::MAX));
    let long = WsdSchedule::new(10, 1 << 53, 0.2).unwrap();
    assert!(long.multiplier((1 << 53) - 1).is_ok());
    assert_range(long.multiplier(1 << 53));
}

#[test]
fn lr_schedule_dispatches_to_the_wrapped_schedule() {
    let cosine = CosineSchedule::new(10, 100).unwrap();
    let wsd = WsdSchedule::new(10, 100, 0.2).unwrap();
    for step in [0u64, 9, 10, 50, 80, 99, 100] {
        assert_eq!(
            LrSchedule::Cosine(cosine)
                .multiplier(step)
                .unwrap()
                .to_bits(),
            cosine.multiplier(step).unwrap().to_bits()
        );
        assert_eq!(
            LrSchedule::Wsd(wsd).multiplier(step).unwrap().to_bits(),
            wsd.multiplier(step).unwrap().to_bits()
        );
    }
    // The cosine schedule clamps past `total`; WSD refuses.
    assert!(LrSchedule::Cosine(cosine).multiplier(150).is_ok());
    assert_range(LrSchedule::Wsd(wsd).multiplier(150));
    assert_eq!(LrSchedule::Wsd(wsd).warmup_steps(), 10);
    assert_eq!(LrSchedule::Wsd(wsd).total_steps(), 100);
    assert_eq!(LrSchedule::Cosine(cosine).total_steps(), 100);
}

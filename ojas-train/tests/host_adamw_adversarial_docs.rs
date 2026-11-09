//! ft-1185100b adversarial lane, attack 5: the doc comment on the overflow threshold
//! (5.8e20 at beta2 = 0.999, 1.8e20 at 0.99; it once said 1.8e20 at 0.999). This measures the real
//! threshold of `adamw_check` (and so of `adamw_update`) and prints it.

use ojas_train::lappi::adamw::{Moments, adamw_check};
use ojas_train::lappi::step::AdamWHyper;

fn refused(g: f32, beta2: f64) -> bool {
    let h = AdamWHyper {
        lr: 1e-3,
        beta1: 0.9,
        beta2,
        eps: 1e-8,
        grad_scale: 1.0,
    };
    adamw_check(&mut [0.5], &[g], &mut Moments::zeros(1), &h, 1e-3, 0.01, 1).is_err()
}

/// Bisect (on f32 bit patterns) the smallest positive gradient that is refused.
fn threshold(beta2: f64) -> f32 {
    let (mut lo, mut hi) = (1.0f32.to_bits(), 3e38f32.to_bits());
    assert!(!refused(1.0, beta2) && refused(3e38, beta2));
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if refused(f32::from_bits(mid), beta2) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    f32::from_bits(hi)
}

#[test]
fn attack5_the_documented_overflow_threshold_at_beta2_0_999() {
    let t = threshold(0.999);
    eprintln!("ATTACK5 beta2=0.999 smallest refused |g| = {t:e}");
    eprintln!(
        "ATTACK5 beta2=0.99  smallest refused |g| = {:e}",
        threshold(0.99)
    );
    // The doc comment says ~5.8e20 at 0.999 and ~1.8e20 at 0.99.
    assert!(
        (t / 5.8e20 - 1.0).abs() < 0.1,
        "DOC CLAIM FALSE: the comment says ~5.8e20, the measured threshold is {t:e}"
    );
    let t99 = threshold(0.99);
    assert!(
        (t99 / 1.8e20 - 1.0).abs() < 0.1,
        "DOC CLAIM FALSE: the comment says ~1.8e20 at beta2 0.99, measured {t99:e}"
    );
}

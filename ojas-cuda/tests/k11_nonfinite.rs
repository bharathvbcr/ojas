//! K11 host emulation: a step that would store a non-finite first moment,
//! second moment or parameter is refused with nothing written (ft-608f764f),
//! and a finite update is bit-for-bit what 7680d51's `adamw_update_f32` wrote.
//!
//! The reference below is that function's loop, copied verbatim, with every
//! scalar passed through `black_box` so the compiler cannot constant-fold or
//! fuse differently from the library.

use std::hint::black_box;

use ojas_cuda::k11_host::{
    adamw_update_f32, AdamwBank, AdamwEntry, AdamwHyper, AdamwTable, EntryScalars, StepScalars,
};
use ojas_cuda::CudaError;

fn f(bits: u32) -> f32 {
    f32::from_bits(bits)
}

fn entry() -> EntryScalars {
    EntryScalars {
        decay_mul: f(0x3f7ff972),
        step_size: f(0x3c23d70a),
        bc2_sqrt: f(0x3d0186e2),
    }
}

/// `low` takes the `m + w * d` lerp branch (w = 0.1), else `gi - d * (1 - w)`
/// (w = 0.9).
fn shared(low: bool) -> StepScalars {
    StepScalars {
        lerp_w: if low { f(0x3dcccccd) } else { f(0x3f666666) },
        beta2: f(0x3f7fbe77),
        one_minus_beta2: f(0x3a83126f),
        eps: f(0x322bcc77),
        grad_scale: 1.0,
    }
}

/// 7680d51 `adamw_update_f32`'s loop, verbatim, scalars opaque.
fn reference(
    p: &mut [f32],
    g: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    e: EntryScalars,
    s: StepScalars,
) {
    let e = black_box(e);
    let s = black_box(s);
    for k in 0..p.len() {
        let w = p[k] * e.decay_mul;
        let gi = g[k] * s.grad_scale;
        let d = gi - m[k];
        let mi = if s.lerp_w < 0.5 {
            m[k] + s.lerp_w * d
        } else {
            gi - d * (1.0 - s.lerp_w)
        };
        let vi = v[k] * s.beta2 + (s.one_minus_beta2 * gi) * gi;
        let denom = vi.sqrt() / e.bc2_sqrt + s.eps;
        p[k] = w + (-e.step_size) * (mi / denom);
        m[k] = mi;
        v[k] = vi;
    }
}

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

struct Win {
    p: Vec<f32>,
    g: Vec<f32>,
    m: Vec<f32>,
    v: Vec<f32>,
}

impl Win {
    /// Four finite elements (recorded below).
    fn finite() -> Win {
        Win {
            p: [0x3f400000, 0xbfc00000, 0x3b03126f, 0x40400000]
                .map(f)
                .to_vec(),
            g: [0x3f000000, 0xbe800000, 0x40400000, 0x3a83126f]
                .map(f)
                .to_vec(),
            m: [0x3dcccccd, 0xbe4ccccd, 0x0, 0x3f000000].map(f).to_vec(),
            v: [0x3c23d70a, 0x3ca3d70a, 0x0, 0x0].map(f).to_vec(),
        }
    }

    fn snapshot(&self) -> [Vec<u32>; 4] {
        [bits(&self.p), bits(&self.g), bits(&self.m), bits(&self.v)]
    }

    fn update(&mut self, e: EntryScalars, s: StepScalars) -> Result<(), CudaError> {
        adamw_update_f32(&mut self.p, &self.g, &mut self.m, &mut self.v, e, s)
    }
}

fn detail(err: &CudaError) -> &str {
    let CudaError::Invalid { detail, .. } = err else {
        panic!("expected an Invalid refusal, got {err:?}");
    };
    detail
}

/// Run the update on `w` and demand a refusal that names `element` and every
/// word of `names`, with all four windows byte-identical afterwards.
fn assert_refused(mut w: Win, low: bool, element: usize, names: &[&str]) {
    let before = w.snapshot();
    let err = w.update(entry(), shared(low)).unwrap_err();
    let d = detail(&err);
    assert!(d.contains(&format!("element {element}:")), "{d}");
    for n in names {
        assert!(d.contains(n), "{d} should name {n}");
    }
    assert!(d.contains("nothing was written"), "{d}");
    assert_eq!(w.snapshot(), before, "a refusal wrote into a window");
}

#[test]
fn a_finite_update_is_bit_equal_to_the_reference_on_both_lerp_branches() {
    for low in [true, false] {
        let (e, s) = (entry(), shared(low));
        let mut got = Win::finite();
        got.update(e, s).unwrap();
        let mut want = Win::finite();
        reference(&mut want.p, &want.g, &mut want.m, &mut want.v, e, s);
        assert_eq!(bits(&got.p), bits(&want.p), "p, low={low}");
        assert_eq!(bits(&got.m), bits(&want.m), "m, low={low}");
        assert_eq!(bits(&got.v), bits(&want.v), "v, low={low}");
    }
}

#[test]
fn a_finite_update_matches_recorded_bits_on_both_lerp_branches() {
    // Computed outside this crate: Python floats rounded to f32 after every
    // operation, on the inputs and scalars above (`double rounding` is exact
    // for + - * / sqrt from f64 to f32).
    let low = (
        [0x3f3fde6a, 0xbfbfec15, 0x3a830bb9, 0xbfc0291c],
        [0x3e0f5c29, 0xbe51eb85, 0x3e99999a, 0x3ee67382],
    );
    let high = (
        [0x3f3f9ce1, 0xbfbfe927, 0xbbe561ec, 0x401f67b6],
        [0x3eeb851e, 0xbe7ae148, 0x402ccccc, 0x3d507c88],
    );
    let v_after = [0x3c27c5ac, 0x3ca4302b, 0x3c1374bc, 0x30897061];
    for (is_low, (p_bits, m_bits)) in [(true, low), (false, high)] {
        let mut w = Win::finite();
        w.update(entry(), shared(is_low)).unwrap();
        assert_eq!(bits(&w.p), p_bits, "p, low={is_low}");
        assert_eq!(bits(&w.m), m_bits, "m, low={is_low}");
        assert_eq!(bits(&w.v), v_after, "v, low={is_low}");
    }
}

#[test]
fn a_nan_gradient_is_refused() {
    let mut w = Win::finite();
    w.g[1] = f32::NAN;
    assert_refused(w, true, 1, &["first moment", "second moment", "parameter"]);
}

#[test]
fn an_infinite_gradient_is_refused() {
    for g in [f32::INFINITY, f32::NEG_INFINITY] {
        let mut w = Win::finite();
        w.g[2] = g;
        assert_refused(w, true, 2, &["first moment", "second moment"]);
    }
}

#[test]
fn a_corrupt_second_moment_is_refused() {
    for v0 in [f32::INFINITY, f32::NAN] {
        let mut w = Win::finite();
        w.v[0] = v0;
        assert_refused(w, true, 0, &["second moment"]);
    }
}

#[test]
fn a_corrupt_first_moment_is_refused() {
    for m0 in [f32::INFINITY, f32::NAN] {
        for low in [true, false] {
            let mut w = Win::finite();
            w.m[3] = m0;
            assert_refused(w, low, 3, &["first moment"]);
        }
    }
}

#[test]
fn an_overflow_of_the_stored_second_moment_is_refused() {
    let mut w = Win::finite();
    w.g[1] = 1e21;
    assert_refused(w, true, 1, &["second moment"]);
}

#[test]
fn an_infinite_parameter_from_a_huge_moment_and_a_zero_v_is_refused() {
    // m0 = 3e38, v0 = 0, g = 0: m and v stay finite, eps in the denominator
    // makes m / denom overflow, so only the parameter would be non-finite.
    let mut w = Win::finite();
    w.g[2] = 0.0;
    w.m[2] = 3.0e38;
    w.v[2] = 0.0;
    assert_refused(w, true, 2, &["parameter"]);
}

#[test]
fn a_gradient_whose_square_overflows_but_whose_stored_term_does_not_is_ok() {
    // g * g = 9e38 is +inf, (one_minus_beta2 * g) * g = 9e35 is finite.
    let g = 3.0e19f32;
    assert!((g * g).is_infinite());
    for low in [true, false] {
        let (e, s) = (entry(), shared(low));
        let mut got = Win::finite();
        got.g[1] = g;
        got.update(e, s)
            .expect("a finite stored update must not be refused");
        let mut want = Win::finite();
        want.g[1] = g;
        reference(&mut want.p, &want.g, &mut want.m, &mut want.v, e, s);
        assert_eq!(bits(&got.p), bits(&want.p), "p, low={low}");
        assert_eq!(bits(&got.m), bits(&want.m), "m, low={low}");
        assert_eq!(bits(&got.v), bits(&want.v), "v, low={low}");
        assert!(got.v[1].is_finite());
    }
}

#[test]
fn a_bad_element_at_the_first_middle_and_last_position_leaves_every_window_untouched() {
    for at in [0usize, 2, 3] {
        for bad in [f32::NAN, f32::INFINITY] {
            let mut w = Win::finite();
            w.g[at] = bad;
            // The elements before and after `at` are finite and would have
            // been updated by a write-as-you-go loop.
            assert_refused(w, true, at, &["first moment"]);
        }
    }
}

#[test]
fn the_first_bad_element_is_the_one_named() {
    let mut w = Win::finite();
    w.g[1] = f32::NAN;
    w.g[3] = f32::INFINITY;
    assert_refused(w, true, 1, &["first moment"]);
}

#[test]
fn a_length_mismatch_is_still_the_length_error() {
    let mut w = Win::finite();
    w.g.pop();
    let before = w.snapshot();
    let err = w.update(entry(), shared(true)).unwrap_err();
    assert!(detail(&err).contains("lengths p 4, g 3, m 4, v 4"));
    assert_eq!(w.snapshot(), before);
}

fn bank() -> AdamwBank {
    let entries = vec![
        AdamwEntry {
            name: "tower".into(),
            offset: 0,
            len: 3,
            lr_scale: 1.0,
            weight_decay: 0.01,
        },
        AdamwEntry {
            name: "head".into(),
            offset: 3,
            len: 2,
            lr_scale: 1.0,
            weight_decay: 0.01,
        },
    ];
    AdamwBank::new(AdamwTable::new(entries, 5).unwrap())
}

fn hyper() -> AdamwHyper {
    AdamwHyper {
        lr: 1e-2,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        grad_scale: 1.0,
    }
}

#[test]
fn a_refused_step_writes_no_window_commits_no_count_and_kills_the_bank() {
    let mut bank = bank();
    let h = hyper();
    let mut p = vec![0.5f32, -0.25, 1.0, 0.75, -0.5];
    let mut m = vec![0.01f32; 5];
    let mut v = vec![0.02f32; 5];
    // The first entry is finite and would be updated; the second holds the NaN.
    let mut g = vec![0.1f32, 0.2, -0.3, 0.4, -0.5];
    g[4] = f32::NAN;
    let before = (bits(&p), bits(&g), bits(&m), bits(&v));
    let err = bank
        .step_f32(&mut p, &g, &mut m, &mut v, &[true, true], &h)
        .unwrap_err();
    let d = detail(&err);
    assert!(d.contains("entry head: element 1:"), "{d}");
    assert!(d.contains("nothing was written"), "{d}");
    assert_eq!(
        (bits(&p), bits(&g), bits(&m), bits(&v)),
        before,
        "the finite window before the bad one was written"
    );
    assert_eq!(bank.steps(), &[0, 0], "a refused step committed counts");
    assert!(bank.is_dead(), "the refusal is the failed-launch rule");

    // A dead bank refuses a later, perfectly finite step before any write.
    let clean = vec![0.1f32; 5];
    let err = bank
        .step_f32(&mut p, &clean, &mut m, &mut v, &[true, true], &h)
        .unwrap_err();
    assert!(detail(&err).contains("bank is dead"), "{}", detail(&err));
    assert_eq!(
        (bits(&p), bits(&m), bits(&v)),
        (before.0, before.2, before.3)
    );
    assert_eq!(bank.steps(), &[0, 0]);
}

#[test]
fn a_non_finite_value_in_an_inactive_entry_is_not_a_refusal() {
    let mut bank = bank();
    let mut p = vec![0.5f32, -0.25, 1.0, 0.75, -0.5];
    let mut m = vec![0.0f32; 5];
    let mut v = vec![0.0f32; 5];
    let mut g = vec![0.1f32, 0.2, -0.3, 0.4, -0.5];
    g[3] = f32::NAN;
    let head = (bits(&p[3..]), bits(&m[3..]), bits(&v[3..]));
    bank.step_f32(&mut p, &g, &mut m, &mut v, &[true, false], &hyper())
        .expect("an entry with no gradient is never read");
    assert_eq!((bits(&p[3..]), bits(&m[3..]), bits(&v[3..])), head);
    assert_eq!(bank.steps(), &[1, 0]);
    assert!(!bank.is_dead());
}

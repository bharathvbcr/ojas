//! vForce `vvexpf` against libm `exp` on ordinary values, and against itself
//! when the same elements are split or evaluated in place.
//! Built only with `--features accelerate` on macOS.
#![cfg(all(feature = "accelerate", target_os = "macos"))]

use ojas_simd::{vvexpf, vvexpf_inplace, NegAbsExp, SimdError};

fn bits(xs: &[f32]) -> Vec<u32> {
    xs.iter().map(|v| v.to_bits()).collect()
}

/// Positive finite results: bit distance is the ulp distance.
fn ulp(got: f32, want: f32) -> u32 {
    assert!(got.is_finite() && want.is_finite() && got >= 0.0 && want >= 0.0);
    got.to_bits().abs_diff(want.to_bits())
}

fn series(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = (i as f32) / (n.max(2) - 1) as f32;
            -80.0 + 160.0 * t
        })
        .collect()
}

#[test]
fn ordinary_values_stay_near_libm_including_the_tail_and_an_offset() {
    // vForce's header allows the exact value to change across OS versions
    // and may flush denormals. On this SDK, normals in [-80, 80] stay within
    // a few ulps of libm. Inputs whose libm result is subnormal are only
    // required to be finite and non-negative.
    const MAX_ULP: u32 = 4;
    let mut worst = 0u32;
    for &n in &[1usize, 2, 3, 7, 8, 15, 16, 17, 31, 1024, 4097] {
        let x = series(n);
        let mut y = vec![7.0; n];
        vvexpf(&mut y, &x).unwrap();
        let mut inplace = x.clone();
        vvexpf_inplace(&mut inplace).unwrap();
        assert_eq!(bits(&y), bits(&inplace), "in place len {n}");
        for (&xi, &yi) in x.iter().zip(&y) {
            let want = (f64::from(xi).exp()) as f32;
            if want < f32::MIN_POSITIVE {
                assert!(yi.is_finite() && yi >= 0.0, "exp({xi:e}) = {yi:e}");
            } else {
                let u = ulp(yi, want);
                worst = worst.max(u);
                assert!(u <= MAX_ULP, "exp({xi:e}) = {yi:e}, libm {want:e}: {u} ulp");
            }
        }
        if n > 1 {
            let x = &x[1..];
            let mut y = vec![7.0; x.len()];
            vvexpf(&mut y, x).unwrap();
            for (&xi, &yi) in x.iter().zip(&y) {
                let want = (f64::from(xi).exp()) as f32;
                if want >= f32::MIN_POSITIVE {
                    let u = ulp(yi, want);
                    worst = worst.max(u);
                    assert!(u <= MAX_ULP, "offset exp({xi:e}) = {yi:e}: {u} ulp");
                }
            }
        }
    }
    assert!(worst <= MAX_ULP, "worst {worst}");
    let mut one = [0.0f32];
    vvexpf_inplace(&mut one).unwrap();
    assert_eq!(one[0].to_bits(), 1.0f32.to_bits());
}

#[test]
fn an_element_does_not_depend_on_the_length_or_its_neighbors() {
    let mut x = series(4096);
    x[10] = 0.0;
    x[11] = -0.0;
    x[12] = f32::MIN_POSITIVE;
    x[13] = -f32::MIN_POSITIVE;
    x[14] = f32::from_bits(1);
    x[15] = 88.0;
    x[16] = -88.0;
    x[17] = 100.0;
    x[18] = -100.0;
    let mut full = x.clone();
    vvexpf_inplace(&mut full).unwrap();
    for &n in &[1usize, 3, 7, 13, 16, 17, 1024] {
        let mut got = x.clone();
        for chunk in got.chunks_mut(n) {
            vvexpf_inplace(chunk).unwrap();
        }
        assert_eq!(bits(&got), bits(&full), "chunks of {n}");
    }
    let mut got = x.clone();
    vvexpf_inplace(&mut got[..1]).unwrap();
    for chunk in got[1..].chunks_mut(13) {
        vvexpf_inplace(chunk).unwrap();
    }
    assert_eq!(bits(&got), bits(&full), "offset chunks");
}

#[test]
fn nonfinite_inputs_follow_exp() {
    let x = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0];
    let mut y = [3.0; 4];
    vvexpf(&mut y, &x).unwrap();
    assert!(y[0].is_nan(), "{}", y[0]);
    assert_eq!(y[1].to_bits(), f32::INFINITY.to_bits());
    assert_eq!(y[2].to_bits(), 0.0f32.to_bits());
    assert_eq!(y[3].to_bits(), 1.0f32.to_bits());
}

#[test]
fn empty_does_not_write() {
    let mut y = [0.0f32; 0];
    vvexpf(&mut y, &[]).unwrap();
    vvexpf_inplace(&mut y).unwrap();
}

#[test]
fn a_length_mismatch_does_not_write() {
    let mut y = [7.0f32, 7.0];
    let err = vvexpf(&mut y[..1], &[1.0, 2.0]).unwrap_err();
    assert_eq!(
        err,
        SimdError::OutputLength {
            output: 1,
            expected: 2
        }
    );
    assert_eq!(y, [7.0, 7.0]);
}

#[test]
fn neg_abs_exp_stores_every_lane_and_a_miss_does_not_return_it() {
    let src = [0.0f32, -0.0, 1.0, -2.5, 40.0, -40.0];
    let mut expect: Vec<f32> = src.iter().map(|v| -v.abs()).collect();
    vvexpf_inplace(&mut expect).unwrap();
    let buf = NegAbsExp::try_new(src.len(), &[(0, 4), (4, 2)]).unwrap();
    let mut saw = false;
    buf.write_chunk(0, &src[..4], |dst| {
        saw = true;
        assert_eq!(bits(dst), bits(&expect[..4]));
        true
    })
    .unwrap();
    assert!(saw);
    buf.write_chunk(1, &src[4..], |_| true).unwrap();
    assert_eq!(bits(&buf.into_vec().unwrap()), bits(&expect));

    assert!(NegAbsExp::try_new(4, &[(0, 2)]).is_err());
    let buf = NegAbsExp::try_new(4, &[(0, 4)]).unwrap();
    let mut called = false;
    assert!(buf
        .write_chunk(0, &[1.0], |_| {
            called = true;
            true
        })
        .is_err());
    assert!(!called, "a short chunk is not presented as initialized");
    assert!(buf.into_vec().is_err());

    let buf = NegAbsExp::try_new(2, &[(0, 2)]).unwrap();
    buf.write_chunk(0, &[1.0, -1.0], |_| true).unwrap();
    assert!(buf.write_chunk(0, &[3.0, 4.0], |_| true).is_err());
    let got = buf.into_vec().unwrap();
    let mut once = [-1.0f32, -1.0];
    vvexpf_inplace(&mut once).unwrap();
    assert_eq!(bits(&got), bits(&once));

    let buf = NegAbsExp::try_new(0, &[]).unwrap();
    assert!(buf.into_vec().unwrap().is_empty());
}

#[test]
fn a_partial_overlap_is_refused_without_a_write() {
    let mut buf = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let ptr = buf.as_mut_ptr();
    let err = unsafe {
        let x = std::slice::from_raw_parts(ptr, 4);
        let y = std::slice::from_raw_parts_mut(ptr.add(2), 4);
        vvexpf(y, x).unwrap_err()
    };
    assert_eq!(err, SimdError::OverlappingOutput);
    assert_eq!(buf, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

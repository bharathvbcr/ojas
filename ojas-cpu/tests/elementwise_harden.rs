//! Adversarial checks for mul and residual add.
//!
//! The lane loops and the add-backward copy must match a scalar reference
//! at every length and thread count, refuse a non-finite value without
//! keeping a charge, and hand back two gradients that do not share storage.

mod common;

use std::sync::{Arc, Barrier};
use std::thread;

use common::f32t;
use ojas_core::{Backend, Budget, Numerics, OjasError, Tensor};
use ojas_cpu::CpuBackend;

fn cpu(threads: usize) -> CpuBackend {
    CpuBackend::with_threads(Budget::new(1 << 28), threads).unwrap()
}

fn series(n: usize, salt: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let bits = (i as u32).wrapping_mul(0x9E37_79B1).wrapping_add(salt);
            let mag = (bits >> 8) as f32 * (2.0 / 16_777_216.0) - 1.0;
            mag * (1.0 + (i % 5) as f32)
        })
        .collect()
}

fn bits(xs: &[f32]) -> Vec<u32> {
    xs.iter().map(|v| v.to_bits()).collect()
}

fn live(cpu: &CpuBackend) -> u64 {
    cpu.budget().live_bytes().unwrap()
}

#[test]
fn lanes_match_the_scalar_formula_at_every_length_and_thread_count() {
    let lengths = [
        1usize, 7, 8, 9, 15, 16, 17, 31, 32, 33, 1023, 1024, 1025, 4095, 4096, 4097, 32768, 32771,
    ];
    for numerics in [Numerics::Exact, Numerics::Fast] {
        for threads in [1usize, 3, 6] {
            let be = cpu(threads).with_numerics(numerics);
            for n in lengths {
                let a = series(n, 1);
                let b = series(n, 2);
                let g = series(n, 3);
                let at = f32t(&be, &a, &[n]);
                let bt = f32t(&be, &b, &[n]);
                let gt = f32t(&be, &g, &[n]);
                let mul: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x * y).collect();
                let add: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
                let ga: Vec<f32> = g.iter().zip(&b).map(|(dy, y)| dy * y).collect();
                let gb: Vec<f32> = g.iter().zip(&a).map(|(dy, x)| dy * x).collect();
                assert_eq!(
                    bits(&be.mul_forward(&at, &bt).unwrap().to_f32_vec().unwrap()),
                    bits(&mul),
                    "mul fwd n={n} threads={threads} {numerics:?}"
                );
                assert_eq!(
                    bits(
                        &be.residual_add_forward(&at, &bt)
                            .unwrap()
                            .to_f32_vec()
                            .unwrap()
                    ),
                    bits(&add),
                    "add fwd n={n} threads={threads} {numerics:?}"
                );
                let (mga, mgb) = be.mul_backward(&at, &bt, &gt).unwrap();
                assert_eq!(bits(&mga.to_f32_vec().unwrap()), bits(&ga), "mul ga");
                assert_eq!(bits(&mgb.to_f32_vec().unwrap()), bits(&gb), "mul gb");
                let (agx, agy) = be.residual_add_backward(&at, &bt, &gt).unwrap();
                assert_eq!(bits(&agx.to_f32_vec().unwrap()), bits(&g), "add gx");
                assert_eq!(bits(&agy.to_f32_vec().unwrap()), bits(&g), "add gy");
            }
        }
    }
}

#[test]
fn rank_and_sign_bits_survive() {
    let a = vec![-0.0f32, 0.0, -2.5, 4.0, f32::MIN_POSITIVE, -1.0];
    let b = vec![1.0f32, -0.0, 3.0, -0.5, 2.0, -0.0];
    let want_sum: Vec<u32> = a.iter().zip(&b).map(|(x, y)| (x + y).to_bits()).collect();
    let want_prod: Vec<u32> = a.iter().zip(&b).map(|(x, y)| (x * y).to_bits()).collect();
    for numerics in [Numerics::Exact, Numerics::Fast] {
        let be = cpu(4).with_numerics(numerics);
        let at = f32t(&be, &a, &[2, 3]);
        let bt = f32t(&be, &b, &[2, 3]);
        let sum = be.residual_add_forward(&at, &bt).unwrap();
        assert_eq!(sum.shape(), &[2, 3]);
        assert_eq!(
            bits(&sum.to_f32_vec().unwrap()),
            want_sum,
            "add {numerics:?}"
        );
        assert_eq!(
            bits(&be.mul_forward(&at, &bt).unwrap().to_f32_vec().unwrap()),
            want_prod,
            "mul {numerics:?}"
        );
        let gy = f32t(&be, &[-0.0, 1.0, -0.0, 2.0, 3.0, -4.0], &[2, 3]);
        let (gx, gy2) = be.residual_add_backward(&at, &bt, &gy).unwrap();
        let src = bits(&gy.to_f32_vec().unwrap());
        assert_eq!(bits(&gx.to_f32_vec().unwrap()), src);
        assert_eq!(bits(&gy2.to_f32_vec().unwrap()), src);
        assert_eq!(gx.to_f32_vec().unwrap()[0].to_bits(), (-0.0f32).to_bits());
    }
}

#[test]
fn add_backward_grads_are_independent_allocations() {
    let be = cpu(1);
    let a = f32t(&be, &[1.0, 2.0, 3.0, 4.0], &[4]);
    let b = f32t(&be, &[5.0, 6.0, 7.0, 8.0], &[4]);
    let g = f32t(&be, &[-0.0, 1.0, 2.0, 3.0], &[4]);
    let src = bits(&g.to_f32_vec().unwrap());
    let (mut gx, mut gy) = be.residual_add_backward(&a, &b, &g).unwrap();
    gx.f32_slice_mut().unwrap()[0] = 42.0;
    assert_eq!(bits(&gy.to_f32_vec().unwrap()), src);
    assert_eq!(bits(&g.to_f32_vec().unwrap()), src);
    assert!(
        gy.f32_slice_mut().is_ok(),
        "the other gradient is also sole owner"
    );
}

#[test]
fn a_nonfinite_operand_or_overflow_refuses_and_releases_the_charge() {
    for numerics in [Numerics::Exact, Numerics::Fast] {
        let be = cpu(6).with_numerics(numerics);
        let n = 64usize;
        let a = series(n, 9);
        let b = series(n, 10);
        let before = live(&be);
        for (name, poison_a, poison_b) in [
            ("nan-a-first", Some((0, f32::NAN)), None),
            ("nan-b-last", None, Some((n - 1, f32::NAN))),
            ("inf-a-mid", Some((n / 2, f32::INFINITY)), None),
            ("neg-inf-b", None, Some((3, f32::NEG_INFINITY))),
        ] {
            let mut av = a.clone();
            let mut bv = b.clone();
            if let Some((i, v)) = poison_a {
                av[i] = v;
            }
            if let Some((i, v)) = poison_b {
                bv[i] = v;
            }
            let at = f32t(&be, &av, &[n]);
            let bt = f32t(&be, &bv, &[n]);
            let held = live(&be);
            assert!(
                matches!(be.mul_forward(&at, &bt), Err(OjasError::NonFinite { .. })),
                "{name} mul"
            );
            assert!(
                matches!(
                    be.residual_add_forward(&at, &bt),
                    Err(OjasError::NonFinite { .. })
                ),
                "{name} add"
            );
            let g = f32t(&be, &series(n, 11), &[n]);
            assert!(
                matches!(
                    be.residual_add_backward(&at, &bt, &g),
                    Err(OjasError::NonFinite { .. })
                ),
                "{name} add bwd"
            );
            drop((at, bt, g));
            assert_eq!(
                live(&be),
                held - (n as u64) * 4 * 2,
                "{name} leaked an input"
            );
            assert_eq!(live(&be), before, "{name} leaked");
        }

        let big = vec![1.0e30f32; 8];
        let at = f32t(&be, &big, &[8]);
        let bt = f32t(&be, &big, &[8]);
        let held = live(&be);
        assert!(
            matches!(be.mul_forward(&at, &bt), Err(OjasError::NonFinite { .. })),
            "overflow product must be refused"
        );
        assert_eq!(live(&be), held, "overflow kept a charge");

        let huge = vec![2.0e38f32; 8];
        let hat = f32t(&be, &huge, &[8]);
        let hbt = f32t(&be, &huge, &[8]);
        let held = live(&be);
        assert!(
            matches!(
                be.residual_add_forward(&hat, &hbt),
                Err(OjasError::NonFinite { .. })
            ),
            "overflow sum must be refused"
        );
        assert_eq!(live(&be), held, "add overflow kept a charge");
    }
}

/// Fast and exact add match scalar `f32` addition, including `-0` and
/// subnormals. A non-finite sum is not returned: NaN, infinity, and
/// `2e38 + 2e38` in the first lane, a 16-wide lane, a one-element tail, and
/// the last lane of a multi-chunk buffer are `NonFinite` and leave the
/// budget where it was.
#[test]
fn add_forward_matches_scalar_and_drops_a_nonfinite_sum() {
    let quiet_nan = f32::from_bits(0x7fc0_0000);
    let pairs = [
        (-0.0f32, -0.0f32),
        (-0.0, 0.0),
        (0.0, -0.0),
        (1.0, -1.0),
        (f32::from_bits(1), f32::from_bits(1)),
        (f32::from_bits(0x007f_ffff), -0.0),
        (f32::from_bits(0x8000_0001), f32::MIN_POSITIVE),
        (-f32::MIN_POSITIVE, -0.0),
        (-2.0e38, 1.0),
        (2.0e38, -1.0),
        (3.25, -0.5),
        (-7.5, 0.25),
    ];
    let lengths = [1usize, 4, 15, 16, 17, 31, 32, 33, 65536, 65539];
    for numerics in [Numerics::Exact, Numerics::Fast] {
        for threads in [1usize, 6] {
            let be = cpu(threads).with_numerics(numerics);
            for &n in &lengths {
                let a: Vec<f32> = (0..n).map(|i| pairs[i % pairs.len()].0).collect();
                let b: Vec<f32> = (0..n).map(|i| pairs[i % pairs.len()].1).collect();
                let want: Vec<u32> = a.iter().zip(&b).map(|(x, y)| (x + y).to_bits()).collect();
                let at = f32t(&be, &a, &[n]);
                let bt = f32t(&be, &b, &[n]);
                let got = be.residual_add_forward(&at, &bt).unwrap_or_else(|err| {
                    panic!("finite add {numerics:?} threads {threads} n={n}: {err}")
                });
                assert_eq!(
                    bits(&got.to_f32_vec().unwrap()),
                    want,
                    "finite add {numerics:?} threads {threads} n={n}"
                );
                drop(got);

                let spots = [0usize, 15, 16, n / 2, n - 1];
                let poisons = [
                    ("nan", f32::NAN, 1.0f32),
                    ("inf", 1.0, f32::INFINITY),
                    ("neg-inf", f32::NEG_INFINITY, 0.0),
                    ("quiet-nan", quiet_nan, 0.0),
                    ("overflow", 2.0e38, 2.0e38),
                ];
                for &index in &spots {
                    if index >= n {
                        continue;
                    }
                    for (name, va, vb) in poisons {
                        let mut av = a.clone();
                        let mut bv = b.clone();
                        av[index] = va;
                        bv[index] = vb;
                        let at = f32t(&be, &av, &[n]);
                        let bt = f32t(&be, &bv, &[n]);
                        let held = live(&be);
                        let got = be.residual_add_forward(&at, &bt);
                        assert!(
                            matches!(got, Err(OjasError::NonFinite { .. })),
                            "{name} at {index} n={n} threads={threads} {numerics:?}: {got:?}"
                        );
                        assert_eq!(
                            live(&be),
                            held,
                            "{name} at {index} n={n} threads={threads} {numerics:?} kept a buffer"
                        );
                        drop((at, bt));
                    }
                }
            }
        }
    }
}

/// Mul backward publishes both gradients or neither. A second charge that
/// does not fit allocates no gradient. A one-element `-0` in both products
/// is stored, including the last lane of a 17-element pair. A non-finite
/// product in either gradient, including the last lane of a multi-chunk pair,
/// releases both charges. Length 0 is covered on the adopt path directly;
/// the backend refuses an empty operand before that path runs.
#[test]
fn mul_backward_publishes_both_gradients_or_neither() {
    for threads in [1usize, 6] {
        let n = 1usize;
        let bytes = 4u64;
        let tight = CpuBackend::new(Budget::new(bytes * 3));
        let a = f32t(&tight, &[1.0], &[n]);
        let b = f32t(&tight, &[2.0], &[n]);
        let g = f32t(&tight, &[3.0], &[n]);
        assert_eq!(live(&tight), bytes * 3);
        let err = tight.mul_backward(&a, &b, &g).unwrap_err();
        assert!(
            matches!(err, OjasError::CapacityExceeded { .. }),
            "threads {threads}: {err:?}"
        );
        assert_eq!(live(&tight), bytes * 3, "a failed first charge leaked");
        drop((a, b, g));

        let tight = CpuBackend::new(Budget::new(bytes * 4));
        let a = f32t(&tight, &[1.0], &[n]);
        let b = f32t(&tight, &[2.0], &[n]);
        let g = f32t(&tight, &[3.0], &[n]);
        assert_eq!(live(&tight), bytes * 3);
        let err = tight.mul_backward(&a, &b, &g).unwrap_err();
        assert!(
            matches!(err, OjasError::CapacityExceeded { .. }),
            "threads {threads}: {err:?}"
        );
        assert_eq!(live(&tight), bytes * 3, "a failed second charge leaked");

        let be = cpu(threads);
        let a = f32t(&be, &[-0.0], &[1]);
        let b = f32t(&be, &[-0.0], &[1]);
        let g = f32t(&be, &[1.0], &[1]);
        let (ga, gb) = be.mul_backward(&a, &b, &g).unwrap();
        assert_eq!(ga.to_f32_vec().unwrap()[0].to_bits(), (-0.0f32).to_bits());
        assert_eq!(gb.to_f32_vec().unwrap()[0].to_bits(), (-0.0f32).to_bits());
        drop((ga, gb, a, b, g));

        let mut tail_a = vec![1.0f32; 17];
        let mut tail_b = vec![2.0f32; 17];
        let tail_g = vec![1.0f32; 17];
        tail_a[16] = -0.0;
        tail_b[16] = -0.0;
        let at = f32t(&be, &tail_a, &[17]);
        let bt = f32t(&be, &tail_b, &[17]);
        let gt = f32t(&be, &tail_g, &[17]);
        let (ga, gb) = be.mul_backward(&at, &bt, &gt).unwrap();
        let got_a = ga.to_f32_vec().unwrap();
        let got_b = gb.to_f32_vec().unwrap();
        assert_eq!(got_a[16].to_bits(), (-0.0f32).to_bits());
        assert_eq!(got_b[16].to_bits(), (-0.0f32).to_bits());
        assert_eq!(got_a[0].to_bits(), (1.0f32 * 2.0).to_bits());
        drop((ga, gb, at, bt, gt));

        let wide_n = (1 << 15) * 2 + 1;
        let base = series(wide_n, 4);
        let held = live(&be);
        for (name, poison_a, poison_b, poison_g) in [
            ("nan-a-last", Some(wide_n - 1), None, None),
            ("inf-b-first", None, Some(0), None),
            ("nan-g-mid", None, None, Some(wide_n / 2)),
        ] {
            let mut av = base.clone();
            let mut bv = base.clone();
            let mut gv = base.clone();
            if let Some(i) = poison_a {
                av[i] = f32::NAN;
            }
            if let Some(i) = poison_b {
                bv[i] = f32::INFINITY;
            }
            if let Some(i) = poison_g {
                gv[i] = f32::NAN;
            }
            let at = f32t(&be, &av, &[wide_n]);
            let bt = f32t(&be, &bv, &[wide_n]);
            let gt = f32t(&be, &gv, &[wide_n]);
            let charged = live(&be);
            assert!(
                matches!(
                    be.mul_backward(&at, &bt, &gt),
                    Err(OjasError::NonFinite { .. })
                ),
                "{name} threads {threads}"
            );
            assert_eq!(live(&be), charged, "{name} kept a gradient charge");
            drop((at, bt, gt));
            assert_eq!(live(&be), held, "{name} leaked an input");
        }
    }
}

#[test]
fn add_backward_second_copy_refusal_releases_the_first() {
    let n = 4usize;
    let bytes = (n * 4) as u64;
    let budget = Budget::new(bytes * 4);
    let be = CpuBackend::new(budget);
    let a = f32t(&be, &[1.0, 2.0, 3.0, 4.0], &[n]);
    let b = f32t(&be, &[1.0, 1.0, 1.0, 1.0], &[n]);
    let g = f32t(&be, &[9.0, 8.0, 7.0, 6.0], &[n]);
    assert_eq!(live(&be), bytes * 3);
    let err = be.residual_add_backward(&a, &b, &g).unwrap_err();
    assert!(
        matches!(err, OjasError::CapacityExceeded { .. }),
        "got {err:?}"
    );
    assert_eq!(live(&be), bytes * 3, "a failed second copy kept the first");
}

#[test]
fn empty_and_strided_inputs_are_refused() {
    let be = cpu(2);
    let empty = Tensor::zeros(&[0, 2], ojas_core::DType::F32, be.budget()).unwrap();
    let row = f32t(&be, &[1.0, 2.0], &[2]);
    assert!(matches!(
        be.residual_add_forward(&empty, &empty),
        Err(OjasError::Shape { .. })
    ));
    assert!(matches!(
        be.mul_forward(&row, &empty),
        Err(OjasError::Shape { .. })
    ));
    let base = f32t(&be, &[1.0, 2.0, 3.0, 4.0], &[4]);
    let strided = base.view(&[2], &[2], 0).unwrap();
    assert!(matches!(
        be.mul_forward(&strided, &row),
        Err(OjasError::Shape { .. })
    ));
    assert!(matches!(
        be.residual_add_backward(&strided, &row, &row),
        Err(OjasError::Shape { .. })
    ));
}

#[test]
fn concurrent_muls_on_one_backend_match_the_scalar_product() {
    let be = Arc::new(cpu(4));
    let n = 10_000usize;
    let a = series(n, 21);
    let b = series(n, 22);
    let at = f32t(&be, &a, &[n]);
    let bt = f32t(&be, &b, &[n]);
    let want: Vec<u32> = a.iter().zip(&b).map(|(x, y)| (x * y).to_bits()).collect();
    let barrier = Arc::new(Barrier::new(4));
    let mut joins = Vec::new();
    for _ in 0..4 {
        let (be, at, bt, barrier, want) = (
            Arc::clone(&be),
            at.clone(),
            bt.clone(),
            Arc::clone(&barrier),
            want.clone(),
        );
        joins.push(thread::spawn(move || {
            barrier.wait();
            let got = be.mul_forward(&at, &bt).unwrap().to_f32_vec().unwrap();
            assert_eq!(bits(&got), want);
        }));
    }
    for join in joins {
        join.join().unwrap();
    }
}

/// Wall time of the nanolab shapes the CPU scorecard uses. Ignored; run with
/// `--ignored` in release. Prints the minimum of 20 calls after two warmups.
#[test]
#[ignore]
fn time_nanolab_elementwise() {
    let be = cpu(6);
    let add_n = 1024 * 768;
    let mul_n = 1024 * 2048;
    let a = series(add_n, 1);
    let b = series(add_n, 2);
    let g = series(add_n, 3);
    let ma = series(mul_n, 4);
    let mb = series(mul_n, 5);
    let mg = series(mul_n, 6);
    let at = f32t(&be, &a, &[1024, 768]);
    let bt = f32t(&be, &b, &[1024, 768]);
    let gt = f32t(&be, &g, &[1024, 768]);
    let mat = f32t(&be, &ma, &[1024, 2048]);
    let mbt = f32t(&be, &mb, &[1024, 2048]);
    let mgt = f32t(&be, &mg, &[1024, 2048]);
    let time = |f: &dyn Fn()| {
        f();
        f();
        let mut best = std::time::Duration::from_secs(60);
        for _ in 0..20 {
            let t0 = std::time::Instant::now();
            f();
            best = best.min(t0.elapsed());
        }
        best
    };
    let add_b = time(&|| {
        be.residual_add_backward(&at, &bt, &gt).unwrap();
    });
    let add_f = time(&|| {
        be.residual_add_forward(&at, &bt).unwrap();
    });
    let mul_f = time(&|| {
        be.mul_forward(&mat, &mbt).unwrap();
    });
    let mul_b = time(&|| {
        be.mul_backward(&mat, &mbt, &mgt).unwrap();
    });
    eprintln!(
        "add_bwd {:.3} ms  add_fwd {:.3} ms  mul_fwd {:.3} ms  mul_bwd {:.3} ms",
        add_b.as_secs_f64() * 1e3,
        add_f.as_secs_f64() * 1e3,
        mul_f.as_secs_f64() * 1e3,
        mul_b.as_secs_f64() * 1e3,
    );
}

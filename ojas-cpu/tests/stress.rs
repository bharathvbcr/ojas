//! Randomized and extreme-magnitude checks for the CPU reference.
//! Policy under test (ojas-core `Backend` docs): a non-finite input, gradient,
//! or result is `NonFinite` and nothing is written; a finite result is returned.

use std::sync::mpsc;
use std::time::Duration;

use ojas_core::{AdamWConfig, Backend, Budget, MuonNs5Config, OjasError, Tensor, RMS_NORM_EPS};
use ojas_cpu::CpuBackend;

mod common;
use common::{assert_capacity, assert_nonfinite, assert_range, bits, f32t, u32t, SplitMix64};

fn wide() -> CpuBackend {
    CpuBackend::new(Budget::new(1 << 24))
}

fn vals(t: &Tensor) -> Vec<f32> {
    t.to_f32_vec().unwrap()
}

fn finite(t: &Tensor) -> bool {
    vals(t).iter().all(|v| v.is_finite())
}

#[test]
fn clip_norm_of_large_finite_gradients_is_finite() {
    let cpu = wide();
    // 3e19^2 overflows f32, but the norm 5e19 is finite.
    let mut g = f32t(&cpu, &[3e19, 4e19], &[2]);
    let norm = cpu
        .clip_grad_norm(std::slice::from_mut(&mut g), 1.0)
        .unwrap();
    assert!((f64::from(norm) / 5e19 - 1.0).abs() < 1e-6, "norm {norm}");
    let got = vals(&g);
    assert!(
        (got[0] - 0.6).abs() < 1e-6 && (got[1] - 0.8).abs() < 1e-6,
        "{got:?}"
    );

    let mut parts = vec![
        f32t(&cpu, &[1e20; 3], &[3]),
        f32t(&cpu, &[-1e20; 6], &[2, 3]),
    ];
    let norm = cpu.clip_grad_norm(&mut parts, 1.0).unwrap();
    assert!((f64::from(norm) / 3e20 - 1.0).abs() < 1e-6, "norm {norm}");
    assert!(parts.iter().all(finite));

    // A norm that does not fit in f32 is refused and nothing is scaled.
    let mut g = f32t(&cpu, &[3e38, 3e38], &[2]);
    let before = bits(&vals(&g));
    assert_nonfinite(cpu.clip_grad_norm(std::slice::from_mut(&mut g), 1.0));
    assert_eq!(bits(&vals(&g)), before);

    // Subnormal gradients: the norm is far below max_norm, bits are unchanged.
    let tiny = f32::from_bits(1);
    let mut g = f32t(&cpu, &[tiny, -tiny, 1e-30], &[3]);
    let before = bits(&vals(&g));
    let norm = cpu
        .clip_grad_norm(std::slice::from_mut(&mut g), 1.0)
        .unwrap();
    assert!(norm.is_finite() && norm < 1e-29);
    assert_eq!(bits(&vals(&g)), before);
}

/// Runs `f` on a worker and fails if it does not return within `secs`.
fn within<T: Send + 'static>(secs: u64, f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(Duration::from_secs(secs))
        .expect("op did not return; the output was sized before the budget check")
}

#[test]
fn oversized_outputs_are_refused_before_they_are_computed() {
    // Inputs are 512 KiB each; outputs would be 2^34 f32 values (64 GiB).
    let n = 1usize << 17;
    let result = within(30, move || {
        let cpu = CpuBackend::new(Budget::new(1 << 21));
        let x = f32t(&cpu, &vec![1.0; n], &[n, 1]);
        let w = f32t(&cpu, &vec![1.0; n], &[n, 1]);
        cpu.linear_forward(&x, &w).map(|_| ())
    });
    assert_capacity(result);

    let result = within(30, move || {
        let cpu = CpuBackend::new(Budget::new(1 << 21));
        let table = f32t(&cpu, &vec![1.0; n], &[1, n]);
        let ids = u32t(&cpu, &vec![0; n], &[n]);
        cpu.embedding_forward(&table, &ids).map(|_| ())
    });
    assert_capacity(result);
}

#[test]
fn optimizer_configs_outside_torch_ranges_are_refused_without_writes() {
    let cpu = wide();
    let base = AdamWConfig::nanolab(1e-3, 0.0);
    for cfg in [
        AdamWConfig { lr: -1e-3, ..base },
        AdamWConfig {
            weight_decay: -0.1,
            ..base
        },
    ] {
        let mut p = f32t(&cpu, &[1.0, -2.0], &[2]);
        let g = f32t(&cpu, &[0.5, 0.25], &[2]);
        let mut m1 = f32t(&cpu, &[0.1, 0.2], &[2]);
        let mut m2 = f32t(&cpu, &[0.3, 0.4], &[2]);
        let snap = (bits(&vals(&p)), bits(&vals(&m1)), bits(&vals(&m2)));
        assert_range(cpu.adamw_step(&mut p, &g, &mut m1, &mut m2, 0, cfg));
        assert_eq!((bits(&vals(&p)), bits(&vals(&m1)), bits(&vals(&m2))), snap);
    }

    let base = MuonNs5Config::nanolab_default();
    for cfg in [
        MuonNs5Config { lr: -0.025, ..base },
        MuonNs5Config {
            weight_decay: -0.1,
            ..base
        },
        MuonNs5Config {
            momentum: -0.5,
            ..base
        },
    ] {
        let mut p = f32t(&cpu, &[1.0, -2.0], &[1, 2]);
        let g = f32t(&cpu, &[0.5, 0.25], &[1, 2]);
        let mut m = f32t(&cpu, &[0.1, 0.2], &[1, 2]);
        let snap = (bits(&vals(&p)), bits(&vals(&m)));
        assert_range(cpu.muon_ns5_step(&mut p, &g, &mut m, cfg));
        assert_eq!((bits(&vals(&p)), bits(&vals(&m))), snap);
    }
}

#[test]
fn adamw_many_steps_converge_and_step_counter_edges() {
    let cpu = wide();
    let mut rng = SplitMix64(7);
    let n = 37;
    let target = rng.vec(n, 3.0);
    let mut p = f32t(&cpu, &rng.vec(n, 3.0), &[n]);
    let mut m1 = f32t(&cpu, &vec![0.0; n], &[n]);
    let mut m2 = f32t(&cpu, &vec![0.0; n], &[n]);
    let cfg = AdamWConfig::nanolab(2e-2, 0.0);
    for step in 0..3000u64 {
        let grad: Vec<f32> = vals(&p).iter().zip(&target).map(|(a, b)| a - b).collect();
        let g = f32t(&cpu, &grad, &[n]);
        cpu.adamw_step(&mut p, &g, &mut m1, &mut m2, step, cfg)
            .unwrap();
        assert!(finite(&p) && finite(&m1) && finite(&m2), "step {step}");
    }
    for (got, want) in vals(&p).iter().zip(&target) {
        assert!((got - want).abs() < 1e-2, "{got} vs {want}");
    }

    // The last representable step: step_before = u64::MAX - 1 runs at step u64::MAX.
    let mut p = f32t(&cpu, &[1.0], &[1]);
    let g = f32t(&cpu, &[0.5], &[1]);
    let mut m1 = f32t(&cpu, &[0.0], &[1]);
    let mut m2 = f32t(&cpu, &[0.0], &[1]);
    cpu.adamw_step(&mut p, &g, &mut m1, &mut m2, u64::MAX - 1, cfg)
        .unwrap();
    // Both bias corrections are 1: m = 0.05, v = 0.0125, p -= lr * m / (sqrt(v) + eps).
    let expect = 1.0 - 2e-2 * 0.05 / (0.0125f64.sqrt() + 1e-8);
    assert!((f64::from(vals(&p)[0]) - expect).abs() < 1e-7);
    let snap = bits(&vals(&p));
    assert_range(cpu.adamw_step(&mut p, &g, &mut m1, &mut m2, u64::MAX, cfg));
    assert_eq!(bits(&vals(&p)), snap);

    // Second moment that overflows the f32 store is refused, not stored as inf.
    let mut p = f32t(&cpu, &[1.0], &[1]);
    let huge = f32t(&cpu, &[1e20], &[1]);
    let mut m1 = f32t(&cpu, &[0.0], &[1]);
    let mut m2 = f32t(&cpu, &[0.0], &[1]);
    assert_nonfinite(cpu.adamw_step(&mut p, &huge, &mut m1, &mut m2, 0, cfg));
    assert_eq!(
        (vals(&p), vals(&m1), vals(&m2)),
        (vec![1.0], vec![0.0], vec![0.0])
    );

    // Subnormal gradient: finite, tiny update.
    let sub = f32t(&cpu, &[f32::from_bits(3)], &[1]);
    cpu.adamw_step(&mut p, &sub, &mut m1, &mut m2, 0, cfg)
        .unwrap();
    assert!(finite(&p) && finite(&m1) && finite(&m2));
    assert!((vals(&p)[0] - 1.0).abs() < 1e-6);
}

fn sym2_eigs(a: f64, b: f64, d: f64) -> (f64, f64) {
    let mid = 0.5 * (a + d);
    let rad = (0.25 * (a - d) * (a - d) + b * b).sqrt();
    (mid - rad, mid + rad)
}

/// Orthogonalized update for `g` (lr 1, no momentum, no decay, from zero).
fn ortho(cpu: &CpuBackend, g: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let cfg = MuonNs5Config {
        lr: 1.0,
        momentum: 0.0,
        weight_decay: 0.0,
        nesterov: false,
        ns5: ojas_core::Ns5Precision::F32,
    };
    let mut p = f32t(cpu, &vec![0.0; rows * cols], &[rows, cols]);
    let grad = f32t(cpu, g, &[rows, cols]);
    let mut m = f32t(cpu, &vec![0.0; rows * cols], &[rows, cols]);
    cpu.muon_ns5_step(&mut p, &grad, &mut m, cfg).unwrap();
    let scale = (rows as f32 / cols as f32).max(1.0).sqrt();
    vals(&p).iter().map(|v| -v / scale).collect()
}

#[test]
fn muon_newton_schulz_pushes_singular_values_toward_one() {
    let cpu = wide();
    let mut rng = SplitMix64(11);
    for trial in 0..50 {
        // Well-conditioned 2x3: [I | 0] plus noise.
        let mut g = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
        for v in &mut g {
            *v += 0.3 * rng.unit();
        }
        let o = ortho(&cpu, &g, 2, 3);
        let row = |r: usize, s: usize| {
            (0..3)
                .map(|c| f64::from(o[r * 3 + c] * o[s * 3 + c]))
                .sum::<f64>()
        };
        let (lo, hi) = sym2_eigs(row(0, 0), row(0, 1), row(1, 1));
        // NS5 with (3.4445, -4.7750, 2.0315) lands singular values in about [0.68, 1.13].
        assert!(
            lo.sqrt() > 0.6 && hi.sqrt() < 1.2,
            "trial {trial}: {lo} {hi}"
        );

        // The tall transpose path gives the transposed answer.
        let mut gt = vec![0.0f32; 6];
        for r in 0..2 {
            for c in 0..3 {
                gt[c * 2 + r] = g[r * 3 + c];
            }
        }
        let ot = ortho(&cpu, &gt, 3, 2);
        for r in 0..2 {
            for c in 0..3 {
                assert!((o[r * 3 + c] - ot[c * 2 + r]).abs() < 1e-6, "trial {trial}");
            }
        }
    }
}

#[test]
fn muon_many_steps_stay_finite() {
    let cpu = wide();
    let mut rng = SplitMix64(13);
    for (rows, cols) in [(1, 1), (1, 7), (7, 1), (4, 4), (5, 3), (3, 5)] {
        let n = rows * cols;
        let mut p = f32t(&cpu, &rng.vec(n, 1.0), &[rows, cols]);
        let mut m = f32t(&cpu, &vec![0.0; n], &[rows, cols]);
        let cfg = MuonNs5Config::nanolab_default();
        for step in 0..300 {
            let g = f32t(&cpu, &rng.vec(n, 1.0), &[rows, cols]);
            cpu.muon_ns5_step(&mut p, &g, &mut m, cfg).unwrap();
            assert!(finite(&p) && finite(&m), "{rows}x{cols} step {step}");
        }
        // Zero gradient after momentum has built up still runs.
        let zero = f32t(&cpu, &vec![0.0; n], &[rows, cols]);
        cpu.muon_ns5_step(&mut p, &zero, &mut m, cfg).unwrap();
        assert!(finite(&p) && finite(&m));
    }
}

#[test]
fn extreme_magnitudes_are_finite_or_nonfinite_errors() {
    let cpu = wide();
    let sub = f32::from_bits(5);

    let x = f32t(
        &cpu,
        &[1e30, -1e30, 1e-30, -1e-30, sub, -sub, 0.0, 88.0, -88.0],
        &[9],
    );
    let y = cpu.silu_forward(&x).unwrap();
    let yv = vals(&y);
    assert_eq!(yv[0], 1e30);
    assert_eq!(yv[1], 0.0);
    assert!(yv.iter().all(|v| v.is_finite()));
    let gx = cpu.silu_backward(&x, &f32t(&cpu, &[1.0; 9], &[9])).unwrap();
    let gv = vals(&gx);
    assert_eq!(gv[0], 1.0);
    assert_eq!(gv[1], 0.0);
    assert!((gv[6] - 0.5).abs() < 1e-7);

    let big = f32t(&cpu, &[1e30, 1e30], &[1, 2]);
    assert_nonfinite(cpu.linear_forward(&big, &big));
    let small = f32t(&cpu, &[1e-30, sub], &[1, 2]);
    assert_eq!(
        vals(&cpu.linear_forward(&small, &small).unwrap()),
        vec![0.0]
    );

    let w = f32t(&cpu, &[1.0, 1.0], &[2]);
    assert_nonfinite(cpu.rms_norm_forward(&big, &w, RMS_NORM_EPS));
    assert_nonfinite(cpu.rms_norm_backward(
        &big,
        &w,
        &f32t(&cpu, &[1.0, 1.0], &[1, 2]),
        RMS_NORM_EPS,
    ));
    // Subnormal row: mean square underflows to 0, rstd is 1/sqrt(eps).
    let y = cpu.rms_norm_forward(&small, &w, RMS_NORM_EPS).unwrap();
    assert!(finite(&y));
    let (gx, gw) = cpu
        .rms_norm_backward(&small, &w, &f32t(&cpu, &[1.0, -1.0], &[1, 2]), RMS_NORM_EPS)
        .unwrap();
    assert!(finite(&gx) && finite(&gw));

    // Cross-entropy of a 2e30 margin is finite; the softmax saturates exactly.
    let logits = f32t(&cpu, &[1e30, -1e30], &[1, 2]);
    let t = u32t(&cpu, &[1], &[1]);
    assert_eq!(
        vals(&cpu.cross_entropy_mean_forward(&logits, &t, None).unwrap()),
        vec![2e30]
    );
    assert_eq!(
        vals(&cpu.cross_entropy_mean_backward(&logits, &t, None).unwrap()),
        vec![1.0, -1.0]
    );
    let logits = f32t(&cpu, &[3e38, -3e38], &[1, 2]);
    assert_nonfinite(cpu.cross_entropy_mean_forward(&logits, &t, None));
    assert_nonfinite(cpu.cross_entropy_mean_backward(&logits, &t, None));

    // Large finite scores saturate the causal softmax without NaN.
    let q = f32t(&cpu, &[100.0, 100.0], &[1, 1, 2, 1]);
    let k = f32t(&cpu, &[100.0, -100.0], &[1, 1, 2, 1]);
    let v = f32t(&cpu, &[3.0, -7.0], &[1, 1, 2, 1]);
    assert_eq!(
        vals(
            &cpu.causal_sdpa_forward(&q, &k, &v, None)
                .map(|(y, _)| y)
                .unwrap()
        ),
        vec![3.0, 3.0]
    );
    let (gq, gk, gv) = cpu
        .causal_sdpa_backward_recompute(&q, &k, &v, &f32t(&cpu, &[1.0, 1.0], &[1, 1, 2, 1]), None)
        .unwrap();
    assert!(finite(&gq) && finite(&gk) && finite(&gv));
    let huge = f32t(&cpu, &[1e20, 1e20], &[1, 1, 2, 1]);
    assert_nonfinite(
        cpu.causal_sdpa_forward(&huge, &huge, &v, None)
            .map(|(y, _)| y),
    );

    // Saturated gate and blend are finite in both directions.
    let x = f32t(&cpu, &[1.0], &[1, 1]);
    let gw = f32t(&cpu, &[1e30, -1e30], &[2, 1]);
    let gb = f32t(&cpu, &[0.0, 0.0], &[2]);
    let attn = f32t(&cpu, &[2.0, 5.0], &[1, 2, 1]);
    assert_eq!(
        vals(
            &cpu.per_head_sigmoid_gate_forward(&x, &gw, &gb, &attn)
                .unwrap()
        ),
        vec![2.0, 0.0]
    );
    let g = cpu
        .per_head_sigmoid_gate_backward(&x, &gw, &gb, &attn, &f32t(&cpu, &[1.0, 1.0], &[1, 2, 1]))
        .unwrap();
    assert!([g.input, g.weight, g.bias, g.attn_out].iter().all(finite));
    // A gate logit that overflows to inf is refused, not saturated to 1.
    let x = f32t(&cpu, &[1e30], &[1, 1]);
    let gw = f32t(&cpu, &[1e30, 1.0], &[2, 1]);
    assert_nonfinite(cpu.per_head_sigmoid_gate_forward(&x, &gw, &gb, &attn));
    assert_nonfinite(cpu.per_head_sigmoid_gate_backward(
        &x,
        &gw,
        &gb,
        &attn,
        &f32t(&cpu, &[1.0, 1.0], &[1, 2, 1]),
    ));
    for lam in [1e30f32, -1e30] {
        let l = f32t(&cpu, &[lam], &[1]);
        let a = f32t(&cpu, &[1.0, 2.0], &[2]);
        let b = f32t(&cpu, &[3.0, 4.0], &[2]);
        let y = cpu.value_residual_blend_forward(&a, &b, &l).unwrap();
        assert_eq!(
            vals(&y),
            if lam > 0.0 {
                vec![3.0, 4.0]
            } else {
                vec![1.0, 2.0]
            }
        );
        let g = cpu
            .value_residual_blend_backward(&a, &b, &l, &f32t(&cpu, &[1.0, 1.0], &[2]))
            .unwrap();
        assert_eq!(vals(&g.lambda), vec![0.0]);
    }
}

#[test]
fn nan_and_inf_in_any_operand_are_nonfinite() {
    let cpu = wide();
    type Op = fn(&CpuBackend, &[Tensor]) -> Result<(), OjasError>;
    // Each case: the operand shapes and an op over them.
    let cases: Vec<(Vec<Vec<usize>>, Op)> = vec![
        (vec![vec![2, 2], vec![3, 2]], |c, t| {
            c.linear_forward(&t[0], &t[1]).map(|_| ())
        }),
        (vec![vec![2, 2], vec![3, 2], vec![2, 3]], |c, t| {
            c.linear_backward(&t[0], &t[1], &t[2]).map(|_| ())
        }),
        (vec![vec![2, 4], vec![4]], |c, t| {
            c.rms_norm_forward(&t[0], &t[1], RMS_NORM_EPS).map(|_| ())
        }),
        (vec![vec![2, 4], vec![4], vec![2, 4]], |c, t| {
            c.rms_norm_backward(&t[0], &t[1], &t[2], RMS_NORM_EPS)
                .map(|_| ())
        }),
        (vec![vec![2, 4], vec![2, 4], vec![2, 4]], |c, t| {
            c.rope_half_split_forward(&t[0], &t[1], &t[2]).map(|_| ())
        }),
        (vec![vec![2, 4], vec![2, 4], vec![2, 4]], |c, t| {
            c.rope_half_split_backward(&t[0], &t[1], &t[2]).map(|_| ())
        }),
        (vec![vec![1, 1, 3, 2]; 3], |c, t| {
            c.causal_sdpa_forward(&t[0], &t[1], &t[2], None)
                .map(|(y, _)| y)
                .map(|_| ())
        }),
        (vec![vec![1, 1, 3, 2]; 4], |c, t| {
            c.causal_sdpa_backward_recompute(&t[0], &t[1], &t[2], &t[3], None)
                .map(|_| ())
        }),
        (
            vec![vec![2, 3], vec![2, 3], vec![2], vec![2, 2, 2]],
            |c, t| {
                c.per_head_sigmoid_gate_forward(&t[0], &t[1], &t[2], &t[3])
                    .map(|_| ())
            },
        ),
        (
            vec![
                vec![2, 3],
                vec![2, 3],
                vec![2],
                vec![2, 2, 2],
                vec![2, 2, 2],
            ],
            |c, t| {
                c.per_head_sigmoid_gate_backward(&t[0], &t[1], &t[2], &t[3], &t[4])
                    .map(|_| ())
            },
        ),
        (vec![vec![3], vec![3], vec![1]], |c, t| {
            c.value_residual_blend_forward(&t[0], &t[1], &t[2])
                .map(|_| ())
        }),
        (vec![vec![3], vec![3], vec![1], vec![3]], |c, t| {
            c.value_residual_blend_backward(&t[0], &t[1], &t[2], &t[3])
                .map(|_| ())
        }),
        (vec![vec![3]], |c, t| c.silu_forward(&t[0]).map(|_| ())),
        (vec![vec![3], vec![3]], |c, t| {
            c.silu_backward(&t[0], &t[1]).map(|_| ())
        }),
        (vec![vec![3], vec![3]], |c, t| {
            c.mul_forward(&t[0], &t[1]).map(|_| ())
        }),
        (vec![vec![3], vec![3], vec![3]], |c, t| {
            c.mul_backward(&t[0], &t[1], &t[2]).map(|_| ())
        }),
        (vec![vec![3], vec![3]], |c, t| {
            c.residual_add_forward(&t[0], &t[1]).map(|_| ())
        }),
        (vec![vec![3], vec![3], vec![3]], |c, t| {
            c.residual_add_backward(&t[0], &t[1], &t[2]).map(|_| ())
        }),
        (vec![vec![2, 4], vec![2, 4], vec![4], vec![4]], |c, t| {
            c.rms_qk_norm_forward(&t[0], &t[1], &t[2], &t[3], RMS_NORM_EPS)
                .map(|_| ())
        }),
        (
            vec![
                vec![2, 4],
                vec![2, 4],
                vec![4],
                vec![4],
                vec![2, 4],
                vec![2, 4],
            ],
            |c, t| {
                c.rms_qk_norm_backward(&t[0], &t[1], &t[2], &t[3], &t[4], &t[5], RMS_NORM_EPS)
                    .map(|_| ())
            },
        ),
    ];
    let mut rng = SplitMix64(17);
    for (index, (shapes, op)) in cases.iter().enumerate() {
        let make = |rng: &mut SplitMix64| -> Vec<Tensor> {
            shapes
                .iter()
                .map(|s| f32t(&cpu, &rng.vec(s.iter().product(), 0.5), s))
                .collect()
        };
        op(&cpu, &make(&mut rng)).unwrap_or_else(|e| panic!("case {index} clean input: {e:?}"));
        for operand in 0..shapes.len() {
            for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut ts = make(&mut rng);
                let mut data = vals(&ts[operand]);
                let at = rng.below(data.len());
                data[at] = bad;
                ts[operand] = f32t(&cpu, &data, &shapes[operand]);
                assert_nonfinite(op(&cpu, &ts));
            }
        }
    }

    let table = f32t(&cpu, &[1.0, f32::INFINITY], &[1, 2]);
    let ids = u32t(&cpu, &[0], &[1]);
    assert_nonfinite(cpu.embedding_forward(&table, &ids));
    let table = f32t(&cpu, &[1.0, 2.0], &[1, 2]);
    assert_nonfinite(cpu.embedding_backward(&table, &ids, &f32t(&cpu, &[f32::NAN, 0.0], &[1, 2])));
    let logits = f32t(&cpu, &[0.0, f32::NEG_INFINITY], &[1, 2]);
    assert_nonfinite(cpu.cross_entropy_mean_forward(&logits, &ids, None));
    assert_nonfinite(cpu.cross_entropy_mean_backward(&logits, &ids, None));
}

#[test]
fn randomized_ops_are_bit_deterministic_and_finite() {
    let cpu = wide();
    let mut rng = SplitMix64(23);
    for _ in 0..40 {
        let b = 1 + rng.below(2);
        let h = 1 + rng.below(3);
        let t = 1 + rng.below(5);
        let d = [1usize, 2, 3, 7, 65][rng.below(5)];
        let shape = [b, h, t, d];
        let n = b * h * t * d;
        let q = f32t(&cpu, &rng.vec(n, 2.0), &shape);
        let k = f32t(&cpu, &rng.vec(n, 2.0), &shape);
        let v = f32t(&cpu, &rng.vec(n, 2.0), &shape);
        let gy = f32t(&cpu, &rng.vec(n, 1.0), &shape);
        let y1 = vals(
            &cpu.causal_sdpa_forward(&q, &k, &v, None)
                .map(|(y, _)| y)
                .unwrap(),
        );
        let y2 = vals(
            &cpu.causal_sdpa_forward(&q, &k, &v, None)
                .map(|(y, _)| y)
                .unwrap(),
        );
        assert_eq!(bits(&y1), bits(&y2));
        let (a1, b1, c1) = cpu
            .causal_sdpa_backward_recompute(&q, &k, &v, &gy, None)
            .unwrap();
        let (a2, b2, c2) = cpu
            .causal_sdpa_backward_recompute(&q, &k, &v, &gy, None)
            .unwrap();
        for (x, y) in [(a1, a2), (b1, b2), (c1, c2)] {
            assert!(finite(&x));
            assert_eq!(bits(&vals(&x)), bits(&vals(&y)));
        }
        // Position 0 attends only to itself.
        for bh in 0..b * h {
            let base = bh * t * d;
            assert_eq!(bits(&y1[base..base + d]), bits(&vals(&v)[base..base + d]));
        }

        let rows = 1 + rng.below(4);
        let (kin, nout) = (1 + rng.below(6), 1 + rng.below(6));
        let x = f32t(&cpu, &rng.vec(rows * kin, 3.0), &[rows, kin]);
        let w = f32t(&cpu, &rng.vec(nout * kin, 3.0), &[nout, kin]);
        let y1 = vals(&cpu.linear_forward(&x, &w).unwrap());
        assert_eq!(bits(&y1), bits(&vals(&cpu.linear_forward(&x, &w).unwrap())));
        let rw = f32t(&cpu, &rng.vec(kin, 2.0), &[kin]);
        let r1 = vals(&cpu.rms_norm_forward(&x, &rw, RMS_NORM_EPS).unwrap());
        assert_eq!(
            bits(&r1),
            bits(&vals(&cpu.rms_norm_forward(&x, &rw, RMS_NORM_EPS).unwrap()))
        );
    }
}

#[test]
fn newton_schulz_f64_accumulation_prevents_false_rejection() {
    // A matrix whose f32 sum-of-squares overflows but whose norm is a valid f32.
    // Before the f64 fix, this would have returned NonFinite.
    let cpu = wide();
    let n = 128;
    // Each element ~1e19: sum_sq ~128 * 1e38 overflows f32 but not f64.
    let data: Vec<f32> = (0..n)
        .map(|i| if i % 2 == 0 { 1e19 } else { -1e19 })
        .collect();
    let cfg = MuonNs5Config {
        lr: 0.025,
        momentum: 0.0,
        weight_decay: 0.0,
        nesterov: false,
        ns5: ojas_core::Ns5Precision::F32,
    };
    let mut p = f32t(&cpu, &vec![0.0; n], &[8, 16]);
    let g = f32t(&cpu, &data, &[8, 16]);
    let mut m = f32t(&cpu, &vec![0.0; n], &[8, 16]);
    // With f64 accumulation, the norm is ~sqrt(128) * 1e19 which is finite in f32.
    let result = cpu.muon_ns5_step(&mut p, &g, &mut m, cfg);
    // The norm sqrt(128 * 1e38) ≈ 1.13e20, which is finite in f32.
    // The orthogonalized update should succeed.
    result.unwrap();
    assert!(vals(&p).iter().all(|v| v.is_finite()));
}

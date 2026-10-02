//! Adversarial checks against the CPU reference. Each assertion is the
//! behavior the op must have; a failure here is a logic bug.

use ojas_core::{
    next_step, AdamWConfig, Backend, Budget, DType, MuonNs5Config, OjasError, Tensor,
    CLIP_GRAD_NORM_EPS, MUON_NS5_A, MUON_NS5_B, MUON_NS5_C, MUON_NS_EPS,
};
use ojas_cpu::CpuBackend;

mod common;
use common::{assert_nonfinite, assert_shape, bits, f32t, u32t};

fn wide() -> CpuBackend {
    CpuBackend::new(Budget::new(1 << 22)).with_numerics(ojas_core::Numerics::Exact)
}

/// Mean log-softmax loss over rows whose target is not `ignore`.
/// An empty valid set is NaN, matching torch.
fn hand_ce(logits: &[f32], targets: &[u32], vocab: usize, ignore: Option<u32>) -> f64 {
    let mut n_valid = 0u32;
    let mut total = 0.0f64;
    for (row, &target) in targets.iter().enumerate() {
        if ignore == Some(target) {
            continue;
        }
        n_valid += 1;
        let start = row * vocab;
        let slice = &logits[start..start + vocab];
        let max = slice
            .iter()
            .fold(f64::NEG_INFINITY, |m, v| m.max(f64::from(*v)));
        let sum: f64 = slice.iter().map(|v| (f64::from(*v) - max).exp()).sum();
        total += max + sum.ln() - f64::from(slice[target as usize]);
    }
    if n_valid == 0 {
        f64::NAN
    } else {
        total / f64::from(n_valid)
    }
}

#[test]
fn all_ignored_cross_entropy_is_nonfinite_not_a_zero_loss() {
    let cpu = wide();
    let logits = f32t(&cpu, &[0.2, -0.4, 1.5, 0.0, -1.0, 0.3], &[3, 2]);

    // ignore_index is a valid class, and every row holds that class.
    let targets = u32t(&cpu, &[1, 1, 1], &[3]);
    let forward = cpu.cross_entropy_mean_forward(&logits, &targets, Some(1));
    assert_nonfinite(forward);
    let backward = cpu.cross_entropy_mean_backward(&logits, &targets, Some(1));
    assert_nonfinite(backward);

    // ignore_index sits outside the vocab and still covers every row.
    let outside = u32t(&cpu, &[9, 9], &[2]);
    let row = f32t(&cpu, &[0.0, 0.0, 2.0, 0.0], &[2, 2]);
    assert_nonfinite(cpu.cross_entropy_mean_forward(&row, &outside, Some(9)));
    assert_nonfinite(cpu.cross_entropy_mean_backward(&row, &outside, Some(9)));

    // A valid class that ignores only some rows still reduces over the rest.
    let mixed_targets = u32t(&cpu, &[1, 0, 1], &[3]);
    let loss = cpu
        .cross_entropy_mean_forward(&logits, &mixed_targets, Some(1))
        .unwrap()
        .to_f32_vec()
        .unwrap();
    let expect = hand_ce(&logits.to_f32_vec().unwrap(), &[1, 0, 1], 2, Some(1));
    assert!(expect.is_finite());
    assert!(
        (f64::from(loss[0]) - expect).abs() < 1e-5,
        "{loss:?} vs {expect}"
    );
    let grad = cpu
        .cross_entropy_mean_backward(&logits, &mixed_targets, Some(1))
        .unwrap()
        .to_f32_vec()
        .unwrap();
    // Row 0 and row 2 are class 1, the ignored class. Their gradients stay zero.
    assert_eq!(&grad[0..2], &[0.0, 0.0]);
    assert_eq!(&grad[4..6], &[0.0, 0.0]);
    assert!(grad[2].is_finite() && grad[3].is_finite());
    assert!(grad[2].abs() + grad[3].abs() > 0.0);
}

#[test]
fn adamw_refuses_a_nonfinite_f32_store_without_touching_moments() {
    let cpu = wide();
    let cfg = AdamWConfig {
        lr: 1e39,
        beta1: 0.9,
        beta2: 0.95,
        eps: 1e-8,
        weight_decay: 0.0,
    };
    let mut param = f32t(&cpu, &[1.0], &[1]);
    let pbits = param.to_f32_vec().unwrap()[0].to_bits();
    let grad = f32t(&cpu, &[1.0], &[1]);
    let mut m1 = f32t(&cpu, &[0.0], &[1]);
    let mut m2 = f32t(&cpu, &[0.0], &[1]);
    let m1bits = m1.to_f32_vec().unwrap()[0].to_bits();
    let m2bits = m2.to_f32_vec().unwrap()[0].to_bits();
    assert_nonfinite(cpu.adamw_step(&mut param, &grad, &mut m1, &mut m2, 0, cfg));
    assert_eq!(param.to_f32_vec().unwrap()[0].to_bits(), pbits);
    assert_eq!(m1.to_f32_vec().unwrap()[0].to_bits(), m1bits);
    assert_eq!(m2.to_f32_vec().unwrap()[0].to_bits(), m2bits);

    // Inf in the gradient, the parameter, and a moment each refuse before a write.
    for (bad_param, bad_grad, bad_m) in [
        (1.0, f32::INFINITY, 0.0),
        (f32::NAN, 1.0, 0.0),
        (1.0, 1.0, f32::NEG_INFINITY),
    ] {
        let mut param = f32t(&cpu, &[bad_param], &[1]);
        let grad = f32t(&cpu, &[bad_grad], &[1]);
        let mut m1 = f32t(&cpu, &[bad_m], &[1]);
        let mut m2 = f32t(&cpu, &[0.0], &[1]);
        let before_p = bits(&param.to_f32_vec().unwrap());
        let before_m = bits(&m1.to_f32_vec().unwrap());
        let before_v = bits(&m2.to_f32_vec().unwrap());
        let small = AdamWConfig::nanolab(1e-3, 0.0);
        assert_nonfinite(cpu.adamw_step(&mut param, &grad, &mut m1, &mut m2, 0, small));
        assert_eq!(bits(&param.to_f32_vec().unwrap()), before_p);
        assert_eq!(bits(&m1.to_f32_vec().unwrap()), before_m);
        assert_eq!(bits(&m2.to_f32_vec().unwrap()), before_v);
    }

    // Weight decay exactly 0 does not multiply the parameter before the update.
    let cfg = AdamWConfig::nanolab(0.0, 0.0);
    let odd = f32::from_bits(0x3f000001);
    let mut param = f32t(&cpu, &[odd], &[1]);
    let grad = f32t(&cpu, &[0.0], &[1]);
    let mut m1 = f32t(&cpu, &[0.0], &[1]);
    let mut m2 = f32t(&cpu, &[0.0], &[1]);
    cpu.adamw_step(&mut param, &grad, &mut m1, &mut m2, 0, cfg)
        .unwrap();
    assert_eq!(param.to_f32_vec().unwrap()[0].to_bits(), odd.to_bits());

    assert!(matches!(
        next_step(u64::MAX),
        Err(OjasError::OutOfRange { .. })
    ));
}

#[test]
fn muon_1x1_and_tall_matrix_match_hand_newton_schulz() {
    let cpu = wide();
    let cfg = MuonNs5Config {
        lr: 0.025,
        momentum: 0.99,
        weight_decay: 0.1,
        nesterov: true,
    };

    let g = 0.37f32;
    let mut param = f32t(&cpu, &[0.5], &[1, 1]);
    let mut mom = f32t(&cpu, &[0.11], &[1, 1]);
    let grad = f32t(&cpu, &[g], &[1, 1]);
    cpu.muon_ns5_step(&mut param, &grad, &mut mom, cfg).unwrap();
    let (expect_p, expect_m) = hand_muon(&[0.5], &[g], &[0.11], 1, 1, cfg);
    assert_eq!(param.to_f32_vec().unwrap(), expect_p);
    assert_eq!(mom.to_f32_vec().unwrap(), expect_m);

    // rows > cols takes the transpose path.
    let p0 = [0.2f32, -0.4, 0.15, 0.8, -0.3, 0.05];
    let g0 = [0.3f32, -0.1, 0.2, -0.5, 0.4, 0.05];
    let m0 = [0.0f32, 0.01, -0.02, 0.0, 0.03, -0.01];
    let mut param = f32t(&cpu, &p0, &[3, 2]);
    let grad = f32t(&cpu, &g0, &[3, 2]);
    let mut mom = f32t(&cpu, &m0, &[3, 2]);
    cpu.muon_ns5_step(&mut param, &grad, &mut mom, cfg).unwrap();
    let (expect_p, expect_m) = hand_muon(&p0, &g0, &m0, 3, 2, cfg);
    let got_p = param.to_f32_vec().unwrap();
    let got_m = mom.to_f32_vec().unwrap();
    for (a, b) in got_p.iter().zip(&expect_p) {
        assert!(
            (a - b).abs() <= 1e-6,
            "tall muon param {got_p:?} vs {expect_p:?}"
        );
    }
    assert_eq!(got_m, expect_m);
    assert!(got_p.iter().all(|v| v.is_finite()));

    let again = {
        let mut param = f32t(&cpu, &p0, &[3, 2]);
        let mut mom = f32t(&cpu, &m0, &[3, 2]);
        cpu.muon_ns5_step(&mut param, &grad, &mut mom, cfg).unwrap();
        (param.to_f32_vec().unwrap(), mom.to_f32_vec().unwrap())
    };
    assert_eq!(got_p, again.0);
    assert_eq!(got_m, again.1);
}

fn hand_muon(
    param: &[f32],
    grad: &[f32],
    momentum: &[f32],
    rows: usize,
    cols: usize,
    cfg: MuonNs5Config,
) -> (Vec<f32>, Vec<f32>) {
    let mom = cfg.momentum as f32;
    let buf: Vec<f32> = momentum
        .iter()
        .zip(grad)
        .map(|(m, g)| mom * m + g)
        .collect();
    let update: Vec<f32> = if cfg.nesterov {
        grad.iter().zip(&buf).map(|(g, b)| g + mom * b).collect()
    } else {
        buf.clone()
    };
    let ortho = hand_ns(&update, rows, cols);
    // nanolab forms both scalars as Python floats; torch rounds each to f32 once.
    let scale = (rows as f64 / cols as f64).max(1.0).sqrt();
    let alpha = (-cfg.lr * scale) as f32;
    let decay = (1.0 - cfg.lr * cfg.weight_decay) as f32;
    let mut next = param.to_vec();
    for p in &mut next {
        *p *= decay;
    }
    for (p, o) in next.iter_mut().zip(&ortho) {
        *p += alpha * o;
    }
    (next, buf)
}

#[test]
fn muon_step_scalars_round_once_from_f64_like_nanolab() {
    let cpu = wide();
    // rows/cols = 1.5: f32(-lr * sqrt(1.5)) differs from f32(-lr) * f32(sqrt(1.5)).
    let cfg = MuonNs5Config::nanolab_default();
    let p0 = [0.2f32, -0.4, 0.15, 0.8, -0.3, 0.05];
    let g0 = [0.3f32, -0.1, 0.2, -0.5, 0.4, 0.05];
    let m0 = [0.0f32, 0.01, -0.02, 0.0, 0.03, -0.01];
    let mut param = f32t(&cpu, &p0, &[3, 2]);
    let grad = f32t(&cpu, &g0, &[3, 2]);
    let mut mom = f32t(&cpu, &m0, &[3, 2]);
    cpu.muon_ns5_step(&mut param, &grad, &mut mom, cfg).unwrap();
    let (expect_p, expect_m) = hand_muon(&p0, &g0, &m0, 3, 2, cfg);
    assert_eq!(bits(&param.to_f32_vec().unwrap()), bits(&expect_p));
    assert_eq!(bits(&mom.to_f32_vec().unwrap()), bits(&expect_m));

    // Decay alone (zero update): p * f32(1 - lr * wd).
    for (lr, wd) in [(0.025, 0.1), (0.3, 0.3), (0.3, 0.7), (0.7, 0.3)] {
        let cfg = MuonNs5Config {
            lr,
            momentum: 0.0,
            weight_decay: wd,
            nesterov: false,
        };
        let mut param = f32t(&cpu, &[0.9, -1.7], &[1, 2]);
        let zero = f32t(&cpu, &[0.0, 0.0], &[1, 2]);
        let mut mom = f32t(&cpu, &[0.0, 0.0], &[1, 2]);
        cpu.muon_ns5_step(&mut param, &zero, &mut mom, cfg).unwrap();
        let decay = (1.0 - lr * wd) as f32;
        assert_eq!(
            bits(&param.to_f32_vec().unwrap()),
            bits(&[0.9f32 * decay, -1.7f32 * decay]),
            "lr {lr} wd {wd}"
        );
    }
}

fn hand_ns(update: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let transposed = rows > cols;
    let (mut x, r, c) = if transposed {
        (hand_transpose(update, rows, cols), cols, rows)
    } else {
        (update.to_vec(), rows, cols)
    };
    let mut sum_sq = 0.0f32;
    for value in &x {
        sum_sq += value * value;
    }
    let denom = sum_sq.sqrt() + (MUON_NS_EPS as f32);
    for value in &mut x {
        *value /= denom;
    }
    let a = MUON_NS5_A as f32;
    let b = MUON_NS5_B as f32;
    let c_coef = MUON_NS5_C as f32;
    for _ in 0..5 {
        let xt = hand_transpose(&x, r, c);
        let a_mat = hand_matmul(&x, &xt, r, c, r);
        let a2 = hand_matmul(&a_mat, &a_mat, r, r, r);
        let mut b_mat = vec![0.0f32; r * r];
        for i in 0..b_mat.len() {
            b_mat[i] = b * a_mat[i] + c_coef * a2[i];
        }
        let bx = hand_matmul(&b_mat, &x, r, r, c);
        for i in 0..x.len() {
            x[i] = a * x[i] + bx[i];
        }
    }
    if transposed {
        x = hand_transpose(&x, r, c);
    }
    x
}

fn hand_transpose(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            out[col * rows + row] = a[row * cols + col];
        }
    }
    out
}

fn hand_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for inner in 0..k {
                acc += a[row * k + inner] * b[inner * n + col];
            }
            out[row * n + col] = acc;
        }
    }
    out
}

#[test]
fn causal_sdpa_t1_t2_and_head_dim_65_match_hand_softmax() {
    let cpu = wide();

    // T = 1: the single position attends only to itself.
    let q = f32t(&cpu, &[0.4, -0.2], &[1, 1, 1, 2]);
    let k = f32t(&cpu, &[0.1, 0.7], &[1, 1, 1, 2]);
    let v = f32t(&cpu, &[3.0, -4.0], &[1, 1, 1, 2]);
    let y = cpu
        .causal_sdpa_forward(&q, &k, &v)
        .unwrap()
        .to_f32_vec()
        .unwrap();
    assert_eq!(y, vec![3.0, -4.0]);

    // T = 2, D = 1. Position 0 attends only to key 0, so its output is v[0].
    let q = f32t(&cpu, &[1.0, 1.0], &[1, 1, 2, 1]);
    let k = f32t(&cpu, &[1.0, 0.0], &[1, 1, 2, 1]);
    let v = f32t(&cpu, &[2.0, 100.0], &[1, 1, 2, 1]);
    let y = cpu
        .causal_sdpa_forward(&q, &k, &v)
        .unwrap()
        .to_f32_vec()
        .unwrap();
    assert!(
        (y[0] - 2.0).abs() < 1e-6,
        "position 0 saw the future: {y:?}"
    );
    let scale = 1.0f64;
    let scores = [scale * 1.0, scale * 0.0];
    let max_s = scores[0].max(scores[1]);
    let e0 = (scores[0] - max_s).exp();
    let e1 = (scores[1] - max_s).exp();
    let expect = (e0 * 2.0 + e1 * 100.0) / (e0 + e1);
    assert!(
        (f64::from(y[1]) - expect).abs() < 1e-5,
        "t1 {} vs {expect}",
        y[1]
    );

    // Head dim 65 is legal on CPU and follows 1/sqrt(65), not a clamp to 64.
    let dim = 65usize;
    let mut qv = vec![0.0f32; dim * 2];
    let mut kv = vec![0.0f32; dim * 2];
    let mut vv = vec![0.0f32; dim * 2];
    qv[0] = 1.0;
    qv[dim] = 1.0;
    kv[0] = 1.0;
    kv[dim + 1] = 1.0;
    for d in 0..dim {
        vv[d] = 0.25;
        vv[dim + d] = -0.5;
    }
    let q = f32t(&cpu, &qv, &[1, 1, 2, dim]);
    let k = f32t(&cpu, &kv, &[1, 1, 2, dim]);
    let v = f32t(&cpu, &vv, &[1, 1, 2, dim]);
    let y = cpu.causal_sdpa_forward(&q, &k, &v).unwrap();
    let got = y.to_f32_vec().unwrap();
    assert_eq!(y.shape(), &[1, 1, 2, dim]);
    assert!(got.iter().all(|value| value.is_finite()));
    for (d, value) in got.iter().take(dim).enumerate() {
        assert!((value - 0.25).abs() < 1e-5, "t0 d{d} {value}");
    }
    let scale = 1.0 / (dim as f64).sqrt();
    let scores = [scale * 1.0, 0.0];
    let max_s = scores[0].max(scores[1]);
    let e0 = (scores[0] - max_s).exp();
    let e1 = (scores[1] - max_s).exp();
    let p0 = e0 / (e0 + e1);
    let p1 = e1 / (e0 + e1);
    for d in 0..dim {
        let expect = p0 * 0.25 + p1 * -0.5;
        let value = f64::from(got[dim + d]);
        assert!((value - expect).abs() < 1e-5, "t1 d{d} {value} vs {expect}");
    }
    let y2 = cpu
        .causal_sdpa_forward(&q, &k, &v)
        .unwrap()
        .to_f32_vec()
        .unwrap();
    assert_eq!(got, y2);
}

#[test]
fn zero_extent_inf_offset_and_budget_do_not_corrupt_inputs() {
    let cpu = wide();
    let w = f32t(&cpu, &[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    for shape in [
        [0usize, 2].as_slice(),
        [2, 0].as_slice(),
        [0, 0].as_slice(),
        [1, 0, 4].as_slice(),
    ] {
        let empty = Tensor::zeros(shape, DType::F32, cpu.budget()).unwrap();
        assert_shape(cpu.linear_forward(&empty, &w));
        assert_shape(cpu.silu_forward(&empty));
        assert_shape(cpu.rms_norm_forward(&empty, &f32t(&cpu, &[1.0, 1.0], &[2]), 1e-6));
    }
    let empty_q = Tensor::zeros(&[0, 1, 2, 4], DType::F32, cpu.budget()).unwrap();
    let k = f32t(&cpu, &[0.0; 8], &[1, 1, 2, 4]);
    assert_shape(cpu.causal_sdpa_forward(&empty_q, &k, &k));
    let zero_time = Tensor::zeros(&[1, 1, 0, 4], DType::F32, cpu.budget()).unwrap();
    assert_shape(cpu.causal_sdpa_forward(&zero_time, &zero_time, &zero_time));

    // Contiguous view with a non-zero byte offset must be read at that offset.
    let parent = f32t(&cpu, &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[6]);
    let window = parent.narrow(8, &[4], &[1]).unwrap();
    assert_eq!(window.byte_offset(), 8);
    assert_eq!(window.to_f32_vec().unwrap(), vec![30.0, 40.0, 50.0, 60.0]);
    let y = cpu.silu_forward(&window).unwrap().to_f32_vec().unwrap();
    let direct = cpu
        .silu_forward(&f32t(&cpu, &[30.0, 40.0, 50.0, 60.0], &[4]))
        .unwrap()
        .to_f32_vec()
        .unwrap();
    assert_eq!(y, direct);
    assert_eq!(
        parent.to_f32_vec().unwrap(),
        vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    );

    let table_parent = f32t(&cpu, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let table = table_parent.narrow(8, &[2, 2], &[2, 1]).unwrap();
    let ids = u32t(&cpu, &[0, 1], &[2]);
    let looked = cpu
        .embedding_forward(&table, &ids)
        .unwrap()
        .to_f32_vec()
        .unwrap();
    assert_eq!(looked, vec![3.0, 4.0, 5.0, 6.0]);

    // In-place step on a uniquely owned offset view must not write the prefix.
    let budget = cpu.budget();
    let mut view = {
        let parent = Tensor::from_f32(&[9.0, 8.0, 1.25, 0.5], &[4], budget).unwrap();
        parent.narrow(8, &[2], &[1]).unwrap()
    };
    let grad = f32t(&cpu, &[0.0, 0.0], &[2]);
    let mut m1 = f32t(&cpu, &[0.0, 0.0], &[2]);
    let mut m2 = f32t(&cpu, &[0.0, 0.0], &[2]);
    let cfg = AdamWConfig::nanolab(0.1, 0.0);
    cpu.adamw_step(&mut view, &grad, &mut m1, &mut m2, 0, cfg)
        .unwrap();
    let whole = view.view(&[4], &[1], 0).unwrap();
    assert_eq!(whole.to_f32_vec().unwrap()[0].to_bits(), 9.0f32.to_bits());
    assert_eq!(whole.to_f32_vec().unwrap()[1].to_bits(), 8.0f32.to_bits());

    let mut grads = {
        let parent = Tensor::from_f32(&[7.0, 6.0, 3.0, 4.0], &[4], budget).unwrap();
        parent.narrow(8, &[2], &[1]).unwrap()
    };
    let before_prefix_budget = cpu.budget().live_bytes().unwrap();
    cpu.clip_grad_norm(std::slice::from_mut(&mut grads), 1.0)
        .unwrap();
    let whole = grads.view(&[4], &[1], 0).unwrap();
    let got = whole.to_f32_vec().unwrap();
    assert_eq!(got[0].to_bits(), 7.0f32.to_bits());
    assert_eq!(got[1].to_bits(), 6.0f32.to_bits());
    let scale = 1.0 / (5.0 + f64::from(CLIP_GRAD_NORM_EPS));
    assert!((f64::from(got[2]) - 3.0 * scale).abs() < 1e-5);
    assert!((f64::from(got[3]) - 4.0 * scale).abs() < 1e-5);
    let _ = before_prefix_budget;

    // A full budget refuses the step and leaves the stored values alone.
    let budget = Budget::new(4 * 4);
    let tight = CpuBackend::new(budget.clone());
    let mut param = Tensor::from_f32(&[1.0], &[1], &budget).unwrap();
    let grad = Tensor::from_f32(&[f32::INFINITY], &[1], &budget).unwrap();
    let mut m1 = Tensor::from_f32(&[0.25], &[1], &budget).unwrap();
    let mut m2 = Tensor::from_f32(&[0.5], &[1], &budget).unwrap();
    let snap_p = param.to_f32_vec().unwrap();
    let snap_m = m1.to_f32_vec().unwrap();
    let snap_v = m2.to_f32_vec().unwrap();
    assert_nonfinite(tight.adamw_step(
        &mut param,
        &grad,
        &mut m1,
        &mut m2,
        3,
        AdamWConfig::nanolab(0.01, 0.0),
    ));
    assert_eq!(param.to_f32_vec().unwrap(), snap_p);
    assert_eq!(m1.to_f32_vec().unwrap(), snap_m);
    assert_eq!(m2.to_f32_vec().unwrap(), snap_v);

    // A full budget no longer refuses: the step is in place and charges
    // nothing. The values match a roomy backend's bit for bit.
    let budget = Budget::new(16);
    let tight = CpuBackend::new(budget.clone());
    let roomy = CpuBackend::new(Budget::new(1 << 20));
    let run = |be: &CpuBackend, b: &Budget| {
        let mut param = Tensor::from_f32(&[1.5], &[1], b).unwrap();
        let grad = Tensor::from_f32(&[0.2], &[1], b).unwrap();
        let mut m1 = Tensor::from_f32(&[0.0], &[1], b).unwrap();
        let mut m2 = Tensor::from_f32(&[0.0], &[1], b).unwrap();
        be.adamw_step(
            &mut param,
            &grad,
            &mut m1,
            &mut m2,
            0,
            AdamWConfig::nanolab(0.01, 0.1),
        )
        .unwrap();
        // Only the four 4-byte tensors are charged; the step adds nothing.
        assert_eq!(b.live_bytes().unwrap(), 16);
        [param, m1, m2].map(|t| t.to_f32_vec().unwrap()[0].to_bits())
    };
    let got = run(&tight, &budget);
    assert_eq!(got, run(&roomy, roomy.budget()));
    assert_ne!(f32::from_bits(got[0]), 1.5);
}

#[test]
fn mismatched_shapes_and_noncontiguous_are_errors() {
    let cpu = wide();
    let x = f32t(&cpu, &[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let w = f32t(&cpu, &[1.0, 0.0, 0.0], &[1, 3]);
    assert_shape(cpu.linear_forward(&x, &w));
    let logits = f32t(&cpu, &[0.0, 1.0, 0.0, 1.0], &[2, 2]);
    assert_shape(cpu.cross_entropy_mean_forward(&logits, &u32t(&cpu, &[0, 1, 0], &[3]), None));
    let base = f32t(&cpu, &[1.0, 2.0, 3.0, 4.0], &[4]);
    let skipped = base.view(&[2], &[2], 0).unwrap();
    assert_shape(cpu.silu_forward(&skipped));
    assert_shape(cpu.rms_norm_forward(&skipped, &f32t(&cpu, &[1.0, 1.0], &[2]), 1e-6));
}

/// Ops that run on the pool share their operands with its tasks by cloning
/// the tensor (one more owner of its storage). Every clone is gone when the
/// op returns, on success, after a task is refused (the cancel hook) and
/// after a NaN refusal, so the caller's next in-place write to the same
/// tensor is not refused as shared and an optimizer step on it runs.
#[test]
fn operands_shared_with_pool_tasks_are_released_when_the_op_returns() {
    let be = CpuBackend::with_threads(Budget::new(1 << 30), 4).unwrap();
    let shape = [1usize, 8, 128, 64];
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();
    let inputs = Budget::new(1 << 30);
    let mut q = Tensor::from_f32(&data, &shape, &inputs).unwrap();
    let k = Tensor::from_f32(&data, &shape, &inputs).unwrap();

    be.causal_sdpa_forward(&q, &k, &k).unwrap();
    be.silu_forward(&q).unwrap();
    q.f32_slice_mut().unwrap()[0] = 0.5;

    be.set_cancel(|| {
        Err(OjasError::Unsupported {
            op: "test-cancel",
            detail: "stop".to_string(),
        })
    });
    assert!(matches!(
        be.causal_sdpa_forward(&q, &k, &k),
        Err(OjasError::Unsupported {
            op: "test-cancel",
            ..
        })
    ));
    be.set_cancel(|| Ok(()));
    q.f32_slice_mut().unwrap()[0] = 0.25;

    let mut nan = data.clone();
    nan[n - 1] = f32::NAN;
    let bad = Tensor::from_f32(&nan, &shape, &inputs).unwrap();
    assert_nonfinite(be.causal_sdpa_forward(&q, &k, &bad));
    let flat = [n];
    let mut p = q.reshape(&flat).unwrap();
    drop(q);
    let (g, mut m1, mut m2) = (
        Tensor::from_f32(&data, &flat, &inputs).unwrap(),
        Tensor::from_f32(&vec![0.0; n], &flat, &inputs).unwrap(),
        Tensor::from_f32(&vec![0.0; n], &flat, &inputs).unwrap(),
    );
    be.silu_forward(&p).unwrap();
    be.adamw_step(
        &mut p,
        &g,
        &mut m1,
        &mut m2,
        0,
        AdamWConfig::nanolab(1e-3, 0.0),
    )
    .unwrap();
}

/// An operand found finite once is not scanned again until it is written,
/// so a NaN written in place after a successful op is refused by the next
/// one, through every entry check: one operand (`check_f32`), several
/// (`check_f32s`), permute's own scan, and an op output, which `fill_out`
/// records as finite when it is made.
#[test]
fn a_nan_written_in_place_after_a_successful_op_is_refused_by_the_next() {
    let cpu = wide();
    let poison = |t: &mut Tensor| t.f32_slice_mut().unwrap()[1] = f32::NAN;

    let mut x = f32t(&cpu, &[0.5, -1.0, 2.0, 0.25], &[2, 2]);
    let w = f32t(&cpu, &[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    cpu.silu_forward(&x).unwrap();
    cpu.residual_add_forward(&x, &w).unwrap();
    cpu.linear_forward(&x, &w).unwrap();
    cpu.permute(&x, &[1, 0]).unwrap();
    let mut acc = f32t(&cpu, &[0.0; 4], &[2, 2]);
    cpu.accumulate_grad(&mut acc, &x).unwrap();
    poison(&mut x);
    assert_nonfinite(cpu.silu_forward(&x));
    assert_nonfinite(cpu.residual_add_forward(&w, &x));
    assert_nonfinite(cpu.linear_forward(&x, &w));
    assert_nonfinite(cpu.permute(&x, &[1, 0]));
    assert_nonfinite(cpu.accumulate_grad(&mut acc, &x));

    let mut y = cpu.silu_forward(&w).unwrap();
    cpu.residual_add_forward(&y, &w).unwrap();
    poison(&mut y);
    assert_nonfinite(cpu.residual_add_forward(&y, &w));
    assert_nonfinite(cpu.silu_forward(&y));
}

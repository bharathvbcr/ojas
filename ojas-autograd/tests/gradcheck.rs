//! f64 central differences against the f32 CPU backward.
//! Loss is the sum of the forward outputs, so the analytic seed is ones.

use ojas_autograd::{central_diff, gradients_match, Tape};
use ojas_core::{Backend, Budget, OjasError, Tensor, RMS_NORM_EPS};
use ojas_cpu::CpuBackend;

const H: f64 = 1e-3;
const ATOL: f64 = 2e-3;
const RTOL: f64 = 2e-2;

fn cpu() -> CpuBackend {
    CpuBackend::new(Budget::new(1 << 20))
}

fn tensor(cpu: &CpuBackend, data: &[f64], shape: &[usize]) -> Tensor {
    let f: Vec<f32> = data.iter().copied().map(|v| v as f32).collect();
    Tensor::from_f32(&f, shape, cpu.budget()).unwrap()
}

fn ones(cpu: &CpuBackend, shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::from_f32(&vec![1.0f32; n], shape, cpu.budget()).unwrap()
}

fn check(analytic: &[f32], numeric: &[f64]) {
    gradients_match(analytic, numeric, ATOL, RTOL).unwrap();
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

#[test]
fn gradcheck_linear_rms_silu_mul_add() {
    let cpu = cpu();
    let x0 = vec![0.5, -1.25, 0.75];
    let w0 = vec![1.5, -0.5, 0.25, 0.5, 1.0, -1.5];
    let x = tensor(&cpu, &x0, &[1, 3]);
    let w = tensor(&cpu, &w0, &[2, 3]);
    let y = cpu.linear_forward(&x, &w).unwrap();
    let gy = ones(&cpu, y.shape());
    let (gx, gw) = cpu.linear_backward(&x, &w, &gy).unwrap();
    let nx = central_diff(&x0, H, |p| {
        let (rows, kin, nout) = (1usize, 3usize, 2usize);
        let mut sum = 0.0;
        for c in 0..nout {
            let mut acc = 0.0;
            for i in 0..kin {
                acc += p[i] * w0[c * kin + i];
            }
            sum += acc;
        }
        let _ = rows;
        Ok(sum)
    })
    .unwrap();
    check(&gx.to_f32_vec().unwrap(), &nx);
    let nw = central_diff(&w0, H, |p| {
        let mut sum = 0.0;
        for c in 0..2 {
            for i in 0..3 {
                sum += x0[i] * p[c * 3 + i];
            }
        }
        Ok(sum)
    })
    .unwrap();
    check(&gw.to_f32_vec().unwrap(), &nw);

    let rx = vec![0.5, -1.25, 0.75, 2.0];
    let rw = vec![1.0, 0.5, -1.0, 1.5];
    let xt = tensor(&cpu, &rx, &[4]);
    let wt = tensor(&cpu, &rw, &[4]);
    let gy = ones(&cpu, &[4]);
    let (gx, gw) = cpu.rms_norm_backward(&xt, &wt, &gy, RMS_NORM_EPS).unwrap();
    let nx = central_diff(&rx, H, |p| Ok(rms_sum(p, &rw))).unwrap();
    let nw = central_diff(&rw, H, |p| Ok(rms_sum(&rx, p))).unwrap();
    check(&gx.to_f32_vec().unwrap(), &nx);
    check(&gw.to_f32_vec().unwrap(), &nw);

    let s0 = vec![0.3, -0.7, 1.1];
    let st = tensor(&cpu, &s0, &[3]);
    let gs = cpu.silu_backward(&st, &ones(&cpu, &[3])).unwrap();
    let ns = central_diff(&s0, H, |p| Ok(p.iter().map(|v| v * sigmoid(*v)).sum())).unwrap();
    check(&gs.to_f32_vec().unwrap(), &ns);

    let a0 = vec![0.4, -1.2];
    let b0 = vec![-0.3, 0.8];
    let at = tensor(&cpu, &a0, &[2]);
    let bt = tensor(&cpu, &b0, &[2]);
    let gy = ones(&cpu, &[2]);
    let (ga, gb) = cpu.mul_backward(&at, &bt, &gy).unwrap();
    check(
        &ga.to_f32_vec().unwrap(),
        &central_diff(&a0, H, |p| Ok(p.iter().zip(&b0).map(|(a, b)| a * b).sum())).unwrap(),
    );
    check(
        &gb.to_f32_vec().unwrap(),
        &central_diff(&b0, H, |p| Ok(a0.iter().zip(p).map(|(a, b)| a * b).sum())).unwrap(),
    );
    let (gx, gy) = cpu.residual_add_backward(&at, &bt, &ones(&cpu, &[2])).unwrap();
    check(&gx.to_f32_vec().unwrap(), &[1.0, 1.0]);
    check(&gy.to_f32_vec().unwrap(), &[1.0, 1.0]);
}

fn rms_sum(x: &[f64], w: &[f64]) -> f64 {
    let dim = x.len() as f64;
    let ms = x.iter().map(|v| v * v).sum::<f64>() / dim;
    let rstd = 1.0 / (ms + 1e-6).sqrt();
    x.iter().zip(w).map(|(v, weight)| v * rstd * weight).sum()
}

#[test]
fn gradcheck_rope_embedding_and_cross_entropy() {
    let cpu = cpu();
    let x0 = vec![0.2, -0.4, 0.7, 1.1];
    let cos = vec![0.6, -0.3, 0.6, -0.3];
    let sin = vec![0.8, 0.9, 0.8, 0.9];
    let xt = tensor(&cpu, &x0, &[4]);
    let ct = tensor(&cpu, &cos, &[4]);
    let st = tensor(&cpu, &sin, &[4]);
    let y = cpu.rope_half_split_forward(&xt, &ct, &st).unwrap();
    let gx = cpu
        .rope_half_split_backward(&ones(&cpu, y.shape()), &ct, &st)
        .unwrap();
    let numeric = central_diff(&x0, H, |p| Ok(rope_sum(p, &cos, &sin))).unwrap();
    check(&gx.to_f32_vec().unwrap(), &numeric);

    let table0 = vec![0.2, -0.5, 0.7, 1.2, -0.1, 0.4];
    let ids = Tensor::from_u32(&[2, 0, 2], &[3], cpu.budget()).unwrap();
    let table = tensor(&cpu, &table0, &[3, 2]);
    let gathered = cpu.embedding_forward(&table, &ids).unwrap();
    let gt = cpu
        .embedding_backward(&table, &ids, &ones(&cpu, gathered.shape()))
        .unwrap();
    let numeric = central_diff(&table0, H, |p| {
        let rows = [[p[4], p[5]], [p[0], p[1]], [p[4], p[5]]];
        Ok(rows.into_iter().flatten().sum())
    })
    .unwrap();
    check(&gt.to_f32_vec().unwrap(), &numeric);

    let logits0 = vec![0.2, -0.4, 0.5, 0.1, -0.2, 0.3];
    let targets = Tensor::from_u32(&[1, 9, 0], &[3], cpu.budget()).unwrap();
    let logits = tensor(&cpu, &logits0, &[3, 2]);
    let gl = cpu
        .cross_entropy_mean_backward(&logits, &targets, Some(9))
        .unwrap();
    let numeric = central_diff(&logits0, H, |p| Ok(ce_mean(p, &[1, 9, 0], Some(9)))).unwrap();
    check(&gl.to_f32_vec().unwrap(), &numeric);
}

fn rope_sum(x: &[f64], cos: &[f64], sin: &[f64]) -> f64 {
    let half = x.len() / 2;
    let mut sum = 0.0;
    for i in 0..half {
        let y1 = x[i] * cos[i] + (-x[i + half]) * sin[i];
        let y2 = x[i + half] * cos[i + half] + x[i] * sin[i + half];
        sum += y1 + y2;
    }
    sum
}

fn ce_mean(logits: &[f64], targets: &[u32], ignore: Option<u32>) -> f64 {
    let vocab = 2;
    let mut n_valid = 0u32;
    let mut total = 0.0;
    for (n, &target) in targets.iter().enumerate() {
        if ignore == Some(target) {
            continue;
        }
        n_valid += 1;
        let row = &logits[n * vocab..(n + 1) * vocab];
        let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = row.iter().map(|v| (v - max).exp()).sum();
        total += max + sum.ln() - row[target as usize];
    }
    if n_valid == 0 {
        f64::NAN
    } else {
        total / f64::from(n_valid)
    }
}

#[test]
fn gradcheck_attention_gate_and_value_residual() {
    let cpu = cpu();
    let q0 = vec![0.2, -0.1, 0.4, 0.3];
    let k0 = vec![0.1, 0.2, -0.3, 0.05];
    let v0 = vec![0.3, -0.2, 0.1, 0.4];
    let shape = [1usize, 1, 2, 2];
    let q = tensor(&cpu, &q0, &shape);
    let k = tensor(&cpu, &k0, &shape);
    let v = tensor(&cpu, &v0, &shape);
    let y = cpu.causal_sdpa_forward(&q, &k, &v).unwrap();
    let gy = ones(&cpu, y.shape());
    let (gq, gk, gv) = cpu.causal_sdpa_backward(&q, &k, &v, &gy).unwrap();
    check(
        &gq.to_f32_vec().unwrap(),
        &central_diff(&q0, H, |p| Ok(sdpa_sum(p, &k0, &v0))).unwrap(),
    );
    check(
        &gk.to_f32_vec().unwrap(),
        &central_diff(&k0, H, |p| Ok(sdpa_sum(&q0, p, &v0))).unwrap(),
    );
    check(
        &gv.to_f32_vec().unwrap(),
        &central_diff(&v0, H, |p| Ok(sdpa_sum(&q0, &k0, p))).unwrap(),
    );

    let x0 = vec![0.4, -0.2];
    let w0 = vec![0.3, 0.1, -0.4, 0.2];
    let b0 = vec![0.05, -0.1];
    let a0 = vec![0.7, -0.3, 0.2, 0.5];
    let x = tensor(&cpu, &x0, &[1, 2]);
    let w = tensor(&cpu, &w0, &[2, 2]);
    let b = tensor(&cpu, &b0, &[2]);
    let attn = tensor(&cpu, &a0, &[1, 2, 2]);
    let y = cpu.per_head_sigmoid_gate_forward(&x, &w, &b, &attn).unwrap();
    let g = cpu
        .per_head_sigmoid_gate_backward(&x, &w, &b, &attn, &ones(&cpu, y.shape()))
        .unwrap();
    check(
        &g.input.to_f32_vec().unwrap(),
        &central_diff(&x0, H, |p| Ok(gate_sum(p, &w0, &b0, &a0))).unwrap(),
    );
    check(
        &g.weight.to_f32_vec().unwrap(),
        &central_diff(&w0, H, |p| Ok(gate_sum(&x0, p, &b0, &a0))).unwrap(),
    );
    check(
        &g.bias.to_f32_vec().unwrap(),
        &central_diff(&b0, H, |p| Ok(gate_sum(&x0, &w0, p, &a0))).unwrap(),
    );
    check(
        &g.attn_out.to_f32_vec().unwrap(),
        &central_diff(&a0, H, |p| Ok(gate_sum(&x0, &w0, &b0, p))).unwrap(),
    );

    let vv = vec![0.2, -0.5];
    let v0s = vec![0.8, 0.1];
    let lam = vec![0.3];
    let vt = tensor(&cpu, &vv, &[2]);
    let v0t = tensor(&cpu, &v0s, &[2]);
    let lt = tensor(&cpu, &lam, &[1]);
    let y = cpu.value_residual_blend_forward(&vt, &v0t, &lt).unwrap();
    let g = cpu
        .value_residual_blend_backward(&vt, &v0t, &lt, &ones(&cpu, y.shape()))
        .unwrap();
    let blend = |v: &[f64], v0: &[f64], lambda: f64| {
        let s = sigmoid(lambda);
        v.iter().zip(v0).map(|(a, b)| (1.0 - s) * a + s * b).sum::<f64>()
    };
    check(
        &g.value.to_f32_vec().unwrap(),
        &central_diff(&vv, H, |p| Ok(blend(p, &v0s, lam[0]))).unwrap(),
    );
    check(
        &g.value0.to_f32_vec().unwrap(),
        &central_diff(&v0s, H, |p| Ok(blend(&vv, p, lam[0]))).unwrap(),
    );
    check(
        &g.lambda.to_f32_vec().unwrap(),
        &central_diff(&lam, H, |p| Ok(blend(&vv, &v0s, p[0]))).unwrap(),
    );
}

fn sdpa_sum(q: &[f64], k: &[f64], v: &[f64]) -> f64 {
    let dim = 2usize;
    let scale = 1.0 / (dim as f64).sqrt();
    let mut sum = 0.0;
    for t in 0..2 {
        let mut scores = vec![0.0; t + 1];
        let mut max_s = f64::NEG_INFINITY;
        for j in 0..=t {
            let mut dot = 0.0;
            for d in 0..dim {
                dot += q[t * dim + d] * k[j * dim + d];
            }
            scores[j] = dot * scale;
            max_s = max_s.max(scores[j]);
        }
        let mut z = 0.0;
        let mut p = vec![0.0; t + 1];
        for j in 0..=t {
            p[j] = (scores[j] - max_s).exp();
            z += p[j];
        }
        for d in 0..dim {
            let mut acc = 0.0;
            for j in 0..=t {
                acc += (p[j] / z) * v[j * dim + d];
            }
            sum += acc;
        }
    }
    sum
}

fn gate_sum(x: &[f64], w: &[f64], bias: &[f64], attn: &[f64]) -> f64 {
    let din = 2usize;
    let heads = 2usize;
    let dh = 2usize;
    let mut sum = 0.0;
    for head in 0..heads {
        let mut z = bias[head];
        for i in 0..din {
            z += x[i] * w[head * din + i];
        }
        let g = sigmoid(z);
        for d in 0..dh {
            sum += attn[head * dh + d] * g;
        }
    }
    sum
}

#[test]
fn tape_backward_matches_direct_cpu() {
    let cpu = cpu();
    let mut tape = Tape::new(cpu.clone());
    let x = tape.leaf(tensor(tape.backend(), &[0.5, -1.25, 0.75], &[1, 3]));
    let w = tape.leaf(tensor(
        tape.backend(),
        &[1.5, -0.5, 0.25, 0.5, 1.0, -1.5],
        &[2, 3],
    ));
    let y = tape.linear(x, w).unwrap();
    let direct = tape.backend().clone();
    let gy = ones(&direct, tape.value(y).unwrap().shape());
    let (gx, gw) = direct
        .linear_backward(tape.value(x).unwrap(), tape.value(w).unwrap(), &gy)
        .unwrap();
    tape.backward(y).unwrap();
    assert_eq!(
        tape.grad(x).unwrap().to_f32_vec().unwrap(),
        gx.to_f32_vec().unwrap()
    );
    assert_eq!(
        tape.grad(w).unwrap().to_f32_vec().unwrap(),
        gw.to_f32_vec().unwrap()
    );

    let mut tape = Tape::new(cpu.clone());
    let hidden = tape.leaf(tensor(tape.backend(), &[0.2, -0.4], &[2]));
    let activated = tape.silu(hidden).unwrap();
    let other = tape.leaf(tensor(tape.backend(), &[0.5, 0.5], &[2]));
    let mixed = tape.mul(activated, other).unwrap();
    tape.backward(mixed).unwrap();
    assert!(tape.grad(hidden).unwrap().to_f32_vec().unwrap().iter().all(|v| v.is_finite()));
    assert!(tape
        .grad(mixed)
        .is_none()
        || tape.grad(mixed).unwrap().to_f32_vec().unwrap().iter().all(|v| v.is_finite()));

    match tape.value(ojas_autograd::Var(50)) {
        Err(ojas_core::OjasError::OutOfRange { .. }) => {}
        other => panic!("expected OutOfRange, got {other:?}"),
    }
}

#[test]
fn all_ignored_cross_entropy_on_the_tape_is_nonfinite() {
    let cpu = cpu();
    let mut tape = Tape::new(cpu.clone());
    let logits = tape.leaf(tensor(tape.backend(), &[0.2, -0.4, 1.0, 0.0], &[2, 2]));
    let targets = Tensor::from_u32(&[0, 0], &[2], cpu.budget()).unwrap();
    match tape.cross_entropy(logits, targets, Some(0)) {
        Err(OjasError::NonFinite { .. }) => {}
        other => panic!("expected NonFinite, got {other:?}"),
    }
}

#[test]
fn rms_qk_norm_matches_central_diff() {
    let cpu = cpu();
    let mut tape = Tape::new(cpu.clone());

    let q0 = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
    let k0 = vec![-0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8];
    let qw0 = vec![1.1, 1.2, 1.3, 1.4];
    let kw0 = vec![0.9, 0.8, 0.7, 0.6];

    let q = tape.leaf(tensor(tape.backend(), &q0, &[2, 4]));
    let k = tape.leaf(tensor(tape.backend(), &k0, &[2, 4]));
    let qw = tape.leaf(tensor(tape.backend(), &qw0, &[4]));
    let kw = tape.leaf(tensor(tape.backend(), &kw0, &[4]));

    let (qn, kn) = tape.rms_qk_norm(q, k, qw, kw, 1e-6).unwrap();
    let sum = tape.add(qn, kn).unwrap();
    tape.backward(sum).unwrap();

    let nq = central_diff(&q0, H, |p| {
        Ok(rms_sum(&p[0..4], &qw0) + rms_sum(&p[4..8], &qw0))
    }).unwrap();
    let nqw = central_diff(&qw0, H, |p| {
        Ok(rms_sum(&q0[0..4], p) + rms_sum(&q0[4..8], p))
    }).unwrap();
    let nk = central_diff(&k0, H, |p| {
        Ok(rms_sum(&p[0..4], &kw0) + rms_sum(&p[4..8], &kw0))
    }).unwrap();
    let nkw = central_diff(&kw0, H, |p| {
        Ok(rms_sum(&k0[0..4], p) + rms_sum(&k0[4..8], p))
    }).unwrap();

    check(&tape.grad(q).unwrap().to_f32_vec().unwrap(), &nq);
    check(&tape.grad(k).unwrap().to_f32_vec().unwrap(), &nk);
    check(&tape.grad(qw).unwrap().to_f32_vec().unwrap(), &nqw);
    check(&tape.grad(kw).unwrap().to_f32_vec().unwrap(), &nkw);
}

#[test]
fn rms_qk_norm_refuses_invalid_inputs() {
    let cpu = cpu();
    let mut tape = Tape::new(cpu.clone());

    let q = tape.leaf(tensor(tape.backend(), &[0.1; 8], &[2, 4]));
    let k = tape.leaf(tensor(tape.backend(), &[0.1; 8], &[2, 4]));
    let qw_bad = tape.leaf(tensor(tape.backend(), &[1.0; 3], &[3])); // bad dim: 3 != 4
    let kw = tape.leaf(tensor(tape.backend(), &[1.0; 4], &[4]));

    // Mismatched q weight
    assert!(tape.rms_qk_norm(q, k, qw_bad, kw, 1e-6).is_err());

    // Non-finite eps
    let qw_good = tape.leaf(tensor(tape.backend(), &[1.0; 4], &[4]));
    assert!(tape.rms_qk_norm(q, k, qw_good, kw, f32::NAN).is_err());
}

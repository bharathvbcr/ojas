//! Randomized f64 central differences against every CPU backward.
//! The loss is `sum(r * y)` with a random `r`, so a swapped or mis-signed
//! term cannot hide behind an all-ones seed. Shapes include head_dim 1, odd
//! dims, seq 1, and head_dim 65.

use ojas_autograd::{central_diff, gradients_match, Tape, Var};
use ojas_core::{Backend, Budget, OjasError, Tensor, RMS_NORM_EPS};
use ojas_cpu::CpuBackend;

const H: f64 = 1e-4;
const ATOL: f64 = 1e-4;
const RTOL: f64 = 1e-3;
const EPS: f64 = RMS_NORM_EPS as f64;

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// f32-representable value in `[-scale, scale)`, returned as f64.
    fn unit(&mut self, scale: f32) -> f64 {
        let u = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        f64::from(scale * (2.0 * u - 1.0))
    }

    /// Bounded away from zero, so RMSNorm of a short row stays smooth.
    fn away(&mut self) -> f64 {
        let v = self.unit(1.0);
        f64::from((v.signum() * (0.25 + 0.75 * v.abs())) as f32)
    }

    fn vec(&mut self, n: usize, scale: f32) -> Vec<f64> {
        (0..n).map(|_| self.unit(scale)).collect()
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

fn cpu() -> CpuBackend {
    CpuBackend::new(Budget::new(1 << 26))
}

fn t(cpu: &CpuBackend, data: &[f64], shape: &[usize]) -> Tensor {
    let f: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    Tensor::from_f32(&f, shape, cpu.budget()).unwrap()
}

fn v(t: &Tensor) -> Vec<f32> {
    t.to_f32_vec().unwrap()
}

fn dot(r: &[f64], y: &[f64]) -> f64 {
    assert_eq!(r.len(), y.len());
    r.iter().zip(y).map(|(a, b)| a * b).sum()
}

fn check(what: &str, analytic: &Tensor, numeric: &[f64]) {
    gradients_match(&v(analytic), numeric, ATOL, RTOL).unwrap_or_else(|e| panic!("{what}: {e}"));
}

fn fd(x: &[f64], f: impl Fn(&[f64]) -> f64) -> Vec<f64> {
    central_diff(x, H, |p| Ok(f(p))).unwrap()
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn lin(x: &[f64], w: &[f64], kin: usize, nout: usize) -> Vec<f64> {
    let rows = x.len() / kin;
    let mut y = vec![0.0; rows * nout];
    for r in 0..rows {
        for c in 0..nout {
            y[r * nout + c] = (0..kin).map(|i| x[r * kin + i] * w[c * kin + i]).sum();
        }
    }
    y
}

fn rms(x: &[f64], w: &[f64]) -> Vec<f64> {
    let dim = w.len();
    let mut y = vec![0.0; x.len()];
    for (row, out) in x.chunks(dim).zip(y.chunks_mut(dim)) {
        let ms = row.iter().map(|a| a * a).sum::<f64>() / dim as f64;
        let rstd = 1.0 / (ms + EPS).sqrt();
        for i in 0..dim {
            out[i] = row[i] * rstd * w[i];
        }
    }
    y
}

/// `cos`/`sin` are either `x`-shaped or `[T, D]` against `x` = `[B, T, H, D]`.
fn rope(x: &[f64], x_shape: &[usize], cos: &[f64], sin: &[f64]) -> Vec<f64> {
    let d = *x_shape.last().unwrap();
    let half = d / 2;
    let mut y = vec![0.0; x.len()];
    for row in 0..x.len() / d {
        let crow = if cos.len() == x.len() {
            row
        } else {
            (row / x_shape[2]) % x_shape[1]
        };
        for i in 0..half {
            let (x1, x2) = (x[row * d + i], x[row * d + i + half]);
            y[row * d + i] = x1 * cos[crow * d + i] - x2 * sin[crow * d + i];
            y[row * d + i + half] = x2 * cos[crow * d + i + half] + x1 * sin[crow * d + i + half];
        }
    }
    y
}

fn sdpa(q: &[f64], k: &[f64], val: &[f64], shape: [usize; 4]) -> Vec<f64> {
    let [b, h, tt, d] = shape;
    let scale = 1.0 / (d as f64).sqrt();
    let mut y = vec![0.0; q.len()];
    for bh in 0..b * h {
        let base = bh * tt * d;
        for t in 0..tt {
            let s: Vec<f64> = (0..=t)
                .map(|j| {
                    scale
                        * (0..d)
                            .map(|e| q[base + t * d + e] * k[base + j * d + e])
                            .sum::<f64>()
                })
                .collect();
            let m = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let p: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
            let z: f64 = p.iter().sum();
            for e in 0..d {
                y[base + t * d + e] = (0..=t).map(|j| p[j] / z * val[base + j * d + e]).sum();
            }
        }
    }
    y
}

fn gate(x: &[f64], w: &[f64], bias: &[f64], attn: &[f64], din: usize, dh: usize) -> Vec<f64> {
    let heads = bias.len();
    let mut y = vec![0.0; attn.len()];
    for row in 0..x.len() / din {
        for hd in 0..heads {
            let z = bias[hd]
                + (0..din)
                    .map(|i| x[row * din + i] * w[hd * din + i])
                    .sum::<f64>();
            let g = sigmoid(z);
            for e in 0..dh {
                let at = (row * heads + hd) * dh + e;
                y[at] = attn[at] * g;
            }
        }
    }
    y
}

fn ce(logits: &[f64], targets: &[u32], vocab: usize, ignore: Option<u32>) -> f64 {
    let mut total = 0.0;
    let mut n = 0.0;
    for (row, &tg) in targets.iter().enumerate() {
        if ignore == Some(tg) {
            continue;
        }
        let l = &logits[row * vocab..(row + 1) * vocab];
        let m = l.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lse = m + l.iter().map(|x| (x - m).exp()).sum::<f64>().ln();
        total += lse - l[tg as usize];
        n += 1.0;
    }
    total / n
}

fn silu(x: &[f64]) -> Vec<f64> {
    x.iter().map(|a| a * sigmoid(*a)).collect()
}

#[test]
fn gradcheck_linear_random_shapes() {
    let cpu = cpu();
    let mut rng = Rng(1);
    for _ in 0..20 {
        let prefix = [1 + rng.below(3), 1 + rng.below(3)];
        let (kin, nout) = ([1, 2, 3, 5, 65][rng.below(5)], 1 + rng.below(7));
        let rows = prefix[0] * prefix[1];
        let x0 = rng.vec(rows * kin, 1.0);
        let w0 = rng.vec(nout * kin, 1.0);
        let r = rng.vec(rows * nout, 1.0);
        let (gx, gw) = cpu
            .linear_backward(
                &t(&cpu, &x0, &[prefix[0], prefix[1], kin]),
                &t(&cpu, &w0, &[nout, kin]),
                &t(&cpu, &r, &[prefix[0], prefix[1], nout]),
            )
            .unwrap();
        check(
            "linear x",
            &gx,
            &fd(&x0, |p| dot(&r, &lin(p, &w0, kin, nout))),
        );
        check(
            "linear w",
            &gw,
            &fd(&w0, |p| dot(&r, &lin(&x0, p, kin, nout))),
        );
    }
}

#[test]
fn gradcheck_rms_norm_and_qk_norm_random_shapes() {
    let cpu = cpu();
    let mut rng = Rng(2);
    for _ in 0..20 {
        let rows = 1 + rng.below(4);
        let dim = [1, 2, 3, 7, 65][rng.below(5)];
        let x0: Vec<f64> = (0..rows * dim).map(|_| rng.away()).collect();
        let w0 = rng.vec(dim, 2.0);
        let r = rng.vec(rows * dim, 1.0);
        let shape = [rows, dim];
        let (gx, gw) = cpu
            .rms_norm_backward(
                &t(&cpu, &x0, &shape),
                &t(&cpu, &w0, &[dim]),
                &t(&cpu, &r, &shape),
                RMS_NORM_EPS,
            )
            .unwrap();
        check("rms x", &gx, &fd(&x0, |p| dot(&r, &rms(p, &w0))));
        check("rms w", &gw, &fd(&w0, |p| dot(&r, &rms(&x0, p))));
    }

    for _ in 0..10 {
        let (b, tt, h) = (1 + rng.below(2), 1 + rng.below(3), 1 + rng.below(3));
        let d = [1, 2, 5, 65][rng.below(4)];
        let shape = [b, tt, h, d];
        let n = b * tt * h * d;
        let q0: Vec<f64> = (0..n).map(|_| rng.away()).collect();
        let k0: Vec<f64> = (0..n).map(|_| rng.away()).collect();
        let qw = rng.vec(d, 2.0);
        let kw = rng.vec(d, 2.0);
        let rq = rng.vec(n, 1.0);
        let rk = rng.vec(n, 1.0);
        let (gq, gk, gqw, gkw) = cpu
            .rms_qk_norm_backward(
                &t(&cpu, &q0, &shape),
                &t(&cpu, &k0, &shape),
                &t(&cpu, &qw, &[d]),
                &t(&cpu, &kw, &[d]),
                &t(&cpu, &rq, &shape),
                &t(&cpu, &rk, &shape),
                RMS_NORM_EPS,
            )
            .unwrap();
        check("qk q", &gq, &fd(&q0, |p| dot(&rq, &rms(p, &qw))));
        check("qk k", &gk, &fd(&k0, |p| dot(&rk, &rms(p, &kw))));
        check("qk qw", &gqw, &fd(&qw, |p| dot(&rq, &rms(&q0, p))));
        check("qk kw", &gkw, &fd(&kw, |p| dot(&rk, &rms(&k0, p))));
    }
}

#[test]
fn gradcheck_rope_both_layouts_random_shapes() {
    let cpu = cpu();
    let mut rng = Rng(3);
    for trial in 0..20 {
        let (b, tt, h) = (1 + rng.below(2), 1 + rng.below(4), 1 + rng.below(3));
        let d = [2, 4, 6, 66][rng.below(4)];
        let shape = [b, tt, h, d];
        let n = b * tt * h * d;
        let x0 = rng.vec(n, 1.0);
        let (cos, sin, cs_shape) = if trial % 2 == 0 {
            (rng.vec(n, 1.0), rng.vec(n, 1.0), shape.to_vec())
        } else {
            (rng.vec(tt * d, 1.0), rng.vec(tt * d, 1.0), vec![tt, d])
        };
        let r = rng.vec(n, 1.0);
        let ct = t(&cpu, &cos, &cs_shape);
        let st = t(&cpu, &sin, &cs_shape);
        let y = cpu
            .rope_half_split_forward(&t(&cpu, &x0, &shape), &ct, &st)
            .unwrap();
        gradients_match(&v(&y), &rope(&x0, &shape, &cos, &sin), 1e-6, 1e-5).unwrap();
        let gx = cpu
            .rope_half_split_backward(&t(&cpu, &r, &shape), &ct, &st)
            .unwrap();
        check(
            "rope x",
            &gx,
            &fd(&x0, |p| dot(&r, &rope(p, &shape, &cos, &sin))),
        );
    }
}

#[test]
fn gradcheck_causal_sdpa_random_shapes() {
    let cpu = cpu();
    let mut rng = Rng(4);
    let mut dims = vec![1usize, 1, 2, 3, 7, 65, 65];
    dims.extend((0..5).map(|_| 1 + rng.below(9)));
    for (trial, &d) in dims.iter().enumerate() {
        let tt = if trial < 2 { 1 } else { 1 + rng.below(4) };
        let shape = [1 + rng.below(2), 1 + rng.below(2), tt, d];
        let n: usize = shape.iter().product();
        let q0 = rng.vec(n, 1.5);
        let k0 = rng.vec(n, 1.5);
        let v0 = rng.vec(n, 1.5);
        let r = rng.vec(n, 1.0);
        let y = cpu
            .causal_sdpa_forward(
                &t(&cpu, &q0, &shape),
                &t(&cpu, &k0, &shape),
                &t(&cpu, &v0, &shape),
                None,
            )
            .map(|(y, _)| y)
            .unwrap();
        gradients_match(&v(&y), &sdpa(&q0, &k0, &v0, shape), 1e-5, 1e-4).unwrap();
        let (gq, gk, gv) = cpu
            .causal_sdpa_backward_recompute(
                &t(&cpu, &q0, &shape),
                &t(&cpu, &k0, &shape),
                &t(&cpu, &v0, &shape),
                &t(&cpu, &r, &shape),
                None,
            )
            .unwrap();
        check(
            "sdpa q",
            &gq,
            &fd(&q0, |p| dot(&r, &sdpa(p, &k0, &v0, shape))),
        );
        check(
            "sdpa k",
            &gk,
            &fd(&k0, |p| dot(&r, &sdpa(&q0, p, &v0, shape))),
        );
        check(
            "sdpa v",
            &gv,
            &fd(&v0, |p| dot(&r, &sdpa(&q0, &k0, p, shape))),
        );
    }
}

#[test]
fn gradcheck_gate_value_residual_pointwise_random_shapes() {
    let cpu = cpu();
    let mut rng = Rng(5);
    for _ in 0..15 {
        let (b, tt) = (1 + rng.below(2), 1 + rng.below(3));
        let din = [1, 3, 8][rng.below(3)];
        let heads = 1 + rng.below(3);
        let dh = [1, 2, 65][rng.below(3)];
        let rows = b * tt;
        let x0 = rng.vec(rows * din, 1.0);
        let w0 = rng.vec(heads * din, 1.0);
        let b0 = rng.vec(heads, 1.0);
        let a0 = rng.vec(rows * heads * dh, 1.0);
        let r = rng.vec(a0.len(), 1.0);
        let a_shape = [b, tt, heads, dh];
        let g = cpu
            .per_head_sigmoid_gate_backward(
                &t(&cpu, &x0, &[b, tt, din]),
                &t(&cpu, &w0, &[heads, din]),
                &t(&cpu, &b0, &[heads]),
                &t(&cpu, &a0, &a_shape),
                &t(&cpu, &r, &a_shape),
            )
            .unwrap();
        check(
            "gate x",
            &g.input,
            &fd(&x0, |p| dot(&r, &gate(p, &w0, &b0, &a0, din, dh))),
        );
        check(
            "gate w",
            &g.weight,
            &fd(&w0, |p| dot(&r, &gate(&x0, p, &b0, &a0, din, dh))),
        );
        check(
            "gate b",
            &g.bias,
            &fd(&b0, |p| dot(&r, &gate(&x0, &w0, p, &a0, din, dh))),
        );
        check(
            "gate attn",
            &g.attn_out,
            &fd(&a0, |p| dot(&r, &gate(&x0, &w0, &b0, p, din, dh))),
        );
    }

    for _ in 0..15 {
        let n = 1 + rng.below(9);
        let va = rng.vec(n, 2.0);
        let vb = rng.vec(n, 2.0);
        let lam = vec![rng.unit(4.0)];
        let r = rng.vec(n, 1.0);
        let blend = |a: &[f64], b: &[f64], l: f64| -> Vec<f64> {
            let s = sigmoid(l);
            a.iter()
                .zip(b)
                .map(|(x, y)| (1.0 - s) * x + s * y)
                .collect()
        };
        let g = cpu
            .value_residual_blend_backward(
                &t(&cpu, &va, &[n]),
                &t(&cpu, &vb, &[n]),
                &t(&cpu, &lam, &[]),
                &t(&cpu, &r, &[n]),
            )
            .unwrap();
        check(
            "vres v",
            &g.value,
            &fd(&va, |p| dot(&r, &blend(p, &vb, lam[0]))),
        );
        check(
            "vres v0",
            &g.value0,
            &fd(&vb, |p| dot(&r, &blend(&va, p, lam[0]))),
        );
        check(
            "vres lambda",
            &g.lambda,
            &fd(&lam, |p| dot(&r, &blend(&va, &vb, p[0]))),
        );

        let x0 = rng.vec(n, 6.0);
        let gx = cpu
            .silu_backward(&t(&cpu, &x0, &[n]), &t(&cpu, &r, &[n]))
            .unwrap();
        check("silu", &gx, &fd(&x0, |p| dot(&r, &silu(p))));

        let prod =
            |a: &[f64], b: &[f64]| -> Vec<f64> { a.iter().zip(b).map(|(x, y)| x * y).collect() };
        let (ga, gb) = cpu
            .mul_backward(&t(&cpu, &va, &[n]), &t(&cpu, &vb, &[n]), &t(&cpu, &r, &[n]))
            .unwrap();
        check("mul a", &ga, &fd(&va, |p| dot(&r, &prod(p, &vb))));
        check("mul b", &gb, &fd(&vb, |p| dot(&r, &prod(&va, p))));
        let (gx, gy) = cpu
            .residual_add_backward(&t(&cpu, &va, &[n]), &t(&cpu, &vb, &[n]), &t(&cpu, &r, &[n]))
            .unwrap();
        check("add x", &gx, &r);
        check("add y", &gy, &r);
    }
}

#[test]
fn gradcheck_embedding_and_cross_entropy_random_shapes() {
    let cpu = cpu();
    let mut rng = Rng(6);
    for _ in 0..15 {
        let (vocab, dim) = (1 + rng.below(6), [1, 3, 65][rng.below(3)]);
        let n = 1 + rng.below(6);
        let ids: Vec<u32> = (0..n).map(|_| rng.below(vocab) as u32).collect();
        let table0 = rng.vec(vocab * dim, 1.0);
        let r = rng.vec(n * dim, 1.0);
        let idt = Tensor::from_u32(&ids, &[n], cpu.budget()).unwrap();
        let gt = cpu
            .embedding_backward(
                &t(&cpu, &table0, &[vocab, dim]),
                &idt,
                &t(&cpu, &r, &[n, dim]),
            )
            .unwrap();
        let gather = |p: &[f64]| -> Vec<f64> {
            ids.iter()
                .flat_map(|&id| p[id as usize * dim..(id as usize + 1) * dim].to_vec())
                .collect()
        };
        check("embedding", &gt, &fd(&table0, |p| dot(&r, &gather(p))));

        let ignore = if rng.below(2) == 0 {
            None
        } else {
            Some(vocab as u32 + 3)
        };
        let mut targets: Vec<u32> = (0..n).map(|_| rng.below(vocab) as u32).collect();
        if let (Some(sentinel), true) = (ignore, n > 1) {
            targets[rng.below(n)] = sentinel;
        }
        let logits0 = rng.vec(n * vocab, 4.0);
        let tt = Tensor::from_u32(&targets, &[n], cpu.budget()).unwrap();
        let lt = t(&cpu, &logits0, &[n, vocab]);
        let loss = cpu.cross_entropy_mean_forward(&lt, &tt, ignore).unwrap();
        let want = ce(&logits0, &targets, vocab, ignore);
        assert!((f64::from(v(&loss)[0]) - want).abs() < 1e-5 * (1.0 + want.abs()));
        let gl = cpu.cross_entropy_mean_backward(&lt, &tt, ignore).unwrap();
        check(
            "cross_entropy",
            &gl,
            &fd(&logits0, |p| ce(p, &targets, vocab, ignore)),
        );
    }
}

#[test]
fn tape_accumulates_reused_nodes() {
    let cpu = cpu();
    let x0 = [0.3, -1.2, 2.5];
    let mut tape = Tape::new(cpu.clone());
    let x = tape.leaf(t(&cpu, &x0, &[3])).unwrap();
    let sq = tape.mul(x, x).unwrap();
    let twice = tape.add(sq, x).unwrap();
    let out = tape.add(twice, twice).unwrap();
    tape.backward(out).unwrap();
    // out = 2 * (x^2 + x), d/dx = 4x + 2.
    let want: Vec<f64> = x0.iter().map(|a| 4.0 * a + 2.0).collect();
    check("reused", tape.grad(x).unwrap(), &want);
}

#[test]
fn tape_backward_twice_starts_from_fresh_gradients() {
    let cpu = cpu();
    let x0 = [0.4, -0.9];
    let mut tape = Tape::new(cpu.clone());
    let x = tape.leaf(t(&cpu, &x0, &[2])).unwrap();
    let y = tape.silu(x).unwrap();
    let z = tape.silu(y).unwrap();
    tape.backward(z).unwrap();
    let first = v(tape.grad(x).unwrap());
    tape.backward(z).unwrap();
    assert_eq!(
        v(tape.grad(x).unwrap()),
        first,
        "repeat backward must not double-count"
    );

    tape.backward(y).unwrap();
    let want = fd(&x0, |p| silu(p).iter().sum());
    check(
        "backward(y) after backward(z)",
        tape.grad(x).unwrap(),
        &want,
    );
    assert!(
        tape.grad(z).is_none(),
        "a node above the seed has no gradient"
    );
}

#[test]
fn tape_backward_failure_leaves_no_partial_gradients() {
    let cpu = cpu();
    let mut tape = Tape::new(cpu.clone());
    let x = tape.leaf(t(&cpu, &[1e-20], &[1])).unwrap();
    let big = tape.leaf(t(&cpu, &[1e30], &[1])).unwrap();
    let c = tape.leaf(t(&cpu, &[1e10], &[1])).unwrap();
    let a = tape.mul(x, big).unwrap();
    let b = tape.mul(a, c).unwrap();
    // d b / d a = 1e10 is finite; d b / d x = 1e10 * 1e30 overflows.
    match tape.backward(b) {
        Err(OjasError::NonFinite { .. }) => {}
        other => panic!("expected NonFinite, got {other:?}"),
    }
    for var in [x, big, c, a, b] {
        assert!(tape.grad(var).is_none(), "{var:?} kept a partial gradient");
    }
}

/// Small nanolab block with tied embeddings, on the tape and in f64.
struct Block {
    ids: Vec<u32>,
    targets: Vec<u32>,
    ignore: Option<u32>,
    cos: Vec<f64>,
    sin: Vec<f64>,
    t: usize,
    c: usize,
    vocab: usize,
}

const LEAVES: [&str; 11] = [
    "table", "norm", "wq", "wk", "wv", "qn", "kn", "wv0", "lam", "wu", "wg",
];

impl Block {
    fn leaf_shapes(&self) -> Vec<Vec<usize>> {
        let (c, v) = (self.c, self.vocab);
        vec![
            vec![v, c],
            vec![c],
            vec![c, c],
            vec![c, c],
            vec![c, c],
            vec![c],
            vec![c],
            vec![c, c],
            vec![1],
            vec![c, c],
            vec![c, c],
        ]
    }

    fn tape_loss(&self, tape: &mut Tape, p: &[Var]) -> Result<Var, OjasError> {
        let cpu = tape.backend().clone();
        let shape = [1, 1, self.t, self.c];
        let ids = Tensor::from_u32(&self.ids, &[1, 1, self.t], cpu.budget())?;
        let cos = t(&cpu, &self.cos, &shape);
        let sin = t(&cpu, &self.sin, &shape);
        let x = tape.embedding(p[0], ids)?;
        let h = tape.rms_norm(x, p[1], RMS_NORM_EPS)?;
        let q = tape.linear(h, p[2])?;
        let k = tape.linear(h, p[3])?;
        let val = tape.linear(h, p[4])?;
        let q = tape.rms_norm(q, p[5], RMS_NORM_EPS)?;
        let k = tape.rms_norm(k, p[6], RMS_NORM_EPS)?;
        let q = tape.rope(q, cos.clone(), sin.clone())?;
        let k = tape.rope(k, cos, sin)?;
        let v0 = tape.linear(x, p[7])?;
        let val = tape.value_residual(val, v0, p[8])?;
        let a = tape.causal_sdpa(q, k, val, None)?;
        let up = tape.linear(a, p[9])?;
        let gl = tape.linear(a, p[10])?;
        let act = tape.silu(gl)?;
        let m = tape.mul(act, up)?;
        let o = tape.add(x, m)?;
        let logits = tape.linear(o, p[0])?;
        let targets = Tensor::from_u32(&self.targets, &[1, 1, self.t], cpu.budget())?;
        tape.cross_entropy(logits, targets, self.ignore)
    }

    fn f64_loss(&self, p: &[Vec<f64>]) -> f64 {
        let (c, t) = (self.c, self.t);
        let shape = [1, 1, t, c];
        let x: Vec<f64> = self
            .ids
            .iter()
            .flat_map(|&id| p[0][id as usize * c..(id as usize + 1) * c].to_vec())
            .collect();
        let h = rms(&x, &p[1]);
        let q = rope(
            &rms(&lin(&h, &p[2], c, c), &p[5]),
            &shape,
            &self.cos,
            &self.sin,
        );
        let k = rope(
            &rms(&lin(&h, &p[3], c, c), &p[6]),
            &shape,
            &self.cos,
            &self.sin,
        );
        let s = sigmoid(p[8][0]);
        let val: Vec<f64> = lin(&h, &p[4], c, c)
            .iter()
            .zip(lin(&x, &p[7], c, c))
            .map(|(a, b)| (1.0 - s) * a + s * b)
            .collect();
        let a = sdpa(&q, &k, &val, shape);
        let up = lin(&a, &p[9], c, c);
        let act = silu(&lin(&a, &p[10], c, c));
        let o: Vec<f64> = x
            .iter()
            .zip(act.iter().zip(&up))
            .map(|(xi, (g, u))| xi + g * u)
            .collect();
        ce(
            &lin(&o, &p[0], c, self.vocab),
            &self.targets,
            self.vocab,
            self.ignore,
        )
    }
}

#[test]
fn tape_gradcheck_tied_embedding_block() {
    let cpu = cpu();
    let mut rng = Rng(8);
    for (t_len, c) in [(1usize, 2usize), (3, 4), (4, 6)] {
        let vocab = 5;
        let mut targets: Vec<u32> = (0..t_len).map(|_| rng.below(vocab) as u32).collect();
        if t_len > 1 {
            targets[0] = 99;
        }
        let n = t_len * c;
        let block = Block {
            ids: (0..t_len).map(|_| rng.below(vocab) as u32).collect(),
            targets,
            ignore: Some(99),
            cos: rng.vec(n, 1.0),
            sin: rng.vec(n, 1.0),
            t: t_len,
            c,
            vocab,
        };
        let params: Vec<Vec<f64>> = block
            .leaf_shapes()
            .iter()
            .map(|s| rng.vec(s.iter().product(), 0.8))
            .collect();
        let mut tape = Tape::new(cpu.clone());
        let vars: Vec<Var> = params
            .iter()
            .zip(block.leaf_shapes())
            .map(|(p, s)| tape.leaf(t(&cpu, p, &s)).unwrap())
            .collect();
        let loss = block.tape_loss(&mut tape, &vars).unwrap();
        let got = f64::from(v(tape.value(loss).unwrap())[0]);
        let want = block.f64_loss(&params);
        assert!(
            (got - want).abs() < 1e-5 * (1.0 + want.abs()),
            "loss {got} vs {want}"
        );
        tape.backward(loss).unwrap();
        for (i, name) in LEAVES.iter().enumerate() {
            let numeric = fd(&params[i], |pi| {
                let mut all = params.clone();
                all[i] = pi.to_vec();
                block.f64_loss(&all)
            });
            let grad = tape
                .grad(vars[i])
                .unwrap_or_else(|| panic!("{name} has no gradient"));
            check(&format!("block T{t_len} C{c} {name}"), grad, &numeric);
        }
    }
}

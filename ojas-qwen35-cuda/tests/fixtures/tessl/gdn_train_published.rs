//! The gated delta rule at transformers' op seam, forward and backward, in f64.
//!
//! `torch_chunk_gated_delta_rule(q, k, v, g, beta, initial_state,
//! use_qk_l2norm_in_kernel=True)` with the gates already computed: per head,
//!
//! ```text
//! q^ = l2norm(q) / sqrt(Dk),  k^ = l2norm(k)       l2norm(x) = x * rsqrt(|x|^2 + 1e-6)
//! S^ = exp(g_t) S_{t-1}
//! u  = v_t - S^T k^                                 (the correction, before beta)
//! S_t = S^ + k^ (beta_t u)^T
//! o_t = S_t^T q^
//! ```
//!
//! with `S` `[Dk, Dv]`. The backward is the reverse-mode derivative of exactly
//! that recurrence, written out by hand; `gdn_train_bwd_f64` is checked against
//! central finite differences of `gdn_train_f64` (in f64, where they are
//! clean) before it is used to judge the kernels.

pub const DK: usize = 128;
pub const L2_EPS: f64 = 1e-6;

#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub b: usize,
    pub t: usize,
    pub h: usize,
    pub dk: usize,
    pub dv: usize,
}

/// Operands in the seam's layout: `q`, `k` `[B, T, H, Dk]`, `v` `[B, T, H, Dv]`,
/// `g`, `beta` `[B, T, H]`, `s0` `[B, H, Dk, Dv]`.
pub struct Inputs<'a> {
    pub s: Shape,
    pub q: &'a [f64],
    pub k: &'a [f64],
    pub v: &'a [f64],
    pub g: &'a [f64],
    pub beta: &'a [f64],
    pub s0: Option<&'a [f64]>,
}

/// Gradients of every input.
pub struct Grads {
    pub dq: Vec<f64>,
    pub dk: Vec<f64>,
    pub dv: Vec<f64>,
    pub dg: Vec<f64>,
    pub dbeta: Vec<f64>,
    pub ds0: Vec<f64>,
}

fn l2norm(x: &[f64]) -> (Vec<f64>, f64) {
    let r = 1.0 / (x.iter().map(|v| v * v).sum::<f64>() + L2_EPS).sqrt();
    (x.iter().map(|v| v * r).collect(), r)
}

/// `(o [B,T,H,Dv], final state [B,H,Dk,Dv])`.
pub fn gdn_train_f64(p: &Inputs<'_>) -> (Vec<f64>, Vec<f64>) {
    let Shape { b, t, h, dk, dv } = p.s;
    let scale = (dk as f64).powf(-0.5);
    let mut o = vec![0.0; b * t * h * dv];
    let mut fin = vec![0.0; b * h * dk * dv];
    for bi in 0..b {
        for hi in 0..h {
            let base = (bi * h + hi) * dk * dv;
            let mut s: Vec<f64> = match p.s0 {
                Some(s0) => s0[base..base + dk * dv].to_vec(),
                None => vec![0.0; dk * dv],
            };
            for ti in 0..t {
                let row = (bi * t + ti) * h + hi;
                let (qh, _) = l2norm(&p.q[row * dk..(row + 1) * dk]);
                let (kh, _) = l2norm(&p.k[row * dk..(row + 1) * dk]);
                let a = p.g[row].exp();
                for x in s.iter_mut() {
                    *x *= a;
                }
                for j in 0..dv {
                    let kv: f64 = (0..dk).map(|i| s[i * dv + j] * kh[i]).sum();
                    let delta = p.beta[row] * (p.v[row * dv + j] - kv);
                    for i in 0..dk {
                        s[i * dv + j] += kh[i] * delta;
                    }
                    o[row * dv + j] = (0..dk).map(|i| s[i * dv + j] * qh[i] * scale).sum();
                }
            }
            fin[base..base + dk * dv].copy_from_slice(&s);
        }
    }
    (o, fin)
}

/// Gradients of `sum(do * o) + sum(dfin * final_state)`.
pub fn gdn_train_bwd_f64(p: &Inputs<'_>, d_o: &[f64], dfin: Option<&[f64]>) -> Grads {
    let Shape { b, t, h, dk, dv } = p.s;
    let scale = (dk as f64).powf(-0.5);
    let mut gr = Grads {
        dq: vec![0.0; b * t * h * dk],
        dk: vec![0.0; b * t * h * dk],
        dv: vec![0.0; b * t * h * dv],
        dg: vec![0.0; b * t * h],
        dbeta: vec![0.0; b * t * h],
        ds0: vec![0.0; b * h * dk * dv],
    };
    for bi in 0..b {
        for hi in 0..h {
            let base = (bi * h + hi) * dk * dv;
            // Forward, keeping every state S_{t-1} (the reference can afford it).
            let mut states = Vec::with_capacity(t + 1);
            let mut s: Vec<f64> = match p.s0 {
                Some(s0) => s0[base..base + dk * dv].to_vec(),
                None => vec![0.0; dk * dv],
            };
            for ti in 0..t {
                states.push(s.clone());
                let row = (bi * t + ti) * h + hi;
                let (kh, _) = l2norm(&p.k[row * dk..(row + 1) * dk]);
                let a = p.g[row].exp();
                for x in s.iter_mut() {
                    *x *= a;
                }
                for j in 0..dv {
                    let kv: f64 = (0..dk).map(|i| s[i * dv + j] * kh[i]).sum();
                    let delta = p.beta[row] * (p.v[row * dv + j] - kv);
                    for i in 0..dk {
                        s[i * dv + j] += kh[i] * delta;
                    }
                }
            }
            let mut ds: Vec<f64> = match dfin {
                Some(d) => d[base..base + dk * dv].to_vec(),
                None => vec![0.0; dk * dv],
            };
            for ti in (0..t).rev() {
                let row = (bi * t + ti) * h + hi;
                let qr = &p.q[row * dk..(row + 1) * dk];
                let kr = &p.k[row * dk..(row + 1) * dk];
                let (qn, rq) = l2norm(qr);
                let (kh, rk) = l2norm(kr);
                let qh: Vec<f64> = qn.iter().map(|x| x * scale).collect();
                let a = p.g[row].exp();
                let beta = p.beta[row];
                let sh: Vec<f64> = states[ti].iter().map(|x| x * a).collect();
                let u: Vec<f64> = (0..dv)
                    .map(|j| p.v[row * dv + j] - (0..dk).map(|i| sh[i * dv + j] * kh[i]).sum::<f64>())
                    .collect();
                let delta: Vec<f64> = u.iter().map(|x| beta * x).collect();
                let st: Vec<f64> = (0..dk * dv).map(|x| sh[x] + kh[x / dv] * delta[x % dv]).collect();
                let dor = &d_o[row * dv..(row + 1) * dv];
                // o = S_t^T q^
                let mut dqh = vec![0.0; dk];
                for i in 0..dk {
                    dqh[i] = (0..dv).map(|j| st[i * dv + j] * dor[j]).sum();
                    for j in 0..dv {
                        ds[i * dv + j] += qh[i] * dor[j];
                    }
                }
                // S_t = S^ + k^ delta^T
                let ddelta: Vec<f64> = (0..dv).map(|j| (0..dk).map(|i| ds[i * dv + j] * kh[i]).sum()).collect();
                let mut dkh: Vec<f64> = (0..dk)
                    .map(|i| (0..dv).map(|j| ds[i * dv + j] * delta[j]).sum())
                    .collect();
                // delta = beta (v - S^T k^)
                for (d, dd) in gr.dv[row * dv..(row + 1) * dv].iter_mut().zip(&ddelta) {
                    *d = beta * dd;
                }
                gr.dbeta[row] = (0..dv).map(|j| ddelta[j] * u[j]).sum();
                // dS^ = dS - beta k^ ddelta^T ; dk^ += -beta S^ ddelta
                let mut dsh = ds.clone();
                for i in 0..dk {
                    for j in 0..dv {
                        dsh[i * dv + j] -= beta * kh[i] * ddelta[j];
                        dkh[i] -= beta * sh[i * dv + j] * ddelta[j];
                    }
                }
                // S^ = a S_{t-1}: dg = <dS^, S^>, dS_{t-1} = a dS^
                gr.dg[row] = (0..dk * dv).map(|x| dsh[x] * sh[x]).sum();
                for x in 0..dk * dv {
                    ds[x] = a * dsh[x];
                }
                // l2norm: y = x r  =>  dx = r (dy - y (y . dy)); q^ = scale * y.
                let dyq: Vec<f64> = dqh.iter().map(|x| x * scale).collect();
                let yq_dot: f64 = (0..dk).map(|i| qn[i] * dyq[i]).sum();
                let yk_dot: f64 = (0..dk).map(|i| kh[i] * dkh[i]).sum();
                for i in 0..dk {
                    gr.dq[row * dk + i] = rq * (dyq[i] - qn[i] * yq_dot);
                    gr.dk[row * dk + i] = rk * (dkh[i] - kh[i] * yk_dot);
                }
            }
            gr.ds0[base..base + dk * dv].copy_from_slice(&ds);
        }
    }
    gr
}

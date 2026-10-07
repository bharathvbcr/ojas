//! **K2: the gated delta rule at the `published` rule**, forward and backward, in
//! f64, at transformers' training seam:
//! `torch_chunk_gated_delta_rule(q, k, v, g, beta, initial_state,
//! use_qk_l2norm_in_kernel=True)` with the gates already computed. Per head,
//! with the state `S` `[Dk, Dv]`:
//!
//! ```text
//! q^ = l2norm(q) / sqrt(Dk),  k^ = l2norm(k)       l2norm(x) = x * rsqrt(|x|^2 + 1e-6)
//! S^ = exp(g_t) S_{t-1}
//! u  = v_t - S^^T k^                                (the published read: the DECAYED state)
//! S_t = S^ + k^ (beta_t u)^T
//! o_t = S_t^T q^
//! ```
//!
//! This is a port of tessl `tests/common/gdn_train.rs` (sha256
//! `dc455ac993c29e8b34f4300c427aca55ff33ce437f9b67f78a454930eca98656` when
//! ported): the same `[B, T, H, D]` / `[B, H, Dk, Dv]` layouts, the same gating
//! (`exp(g)` applied to the whole state before the read), the same l2norm and
//! `1/sqrt(Dk)` placement, and the same operation order, so the two agree bit
//! for bit (`tests/reference_gdn_published.rs` asserts it). Two things are
//! added for the CUDA kernels, both from tessl's Metal kernel
//! (`kernels/gdn_train.metal:29-31,96-101,197-206`):
//!
//! - the forward also returns the **checkpoints**, `[B, H, NC, Dk, Dv]` with
//!   `NC = ceil(T / 64)`: checkpoint `c` is the state entering token `64c`,
//!   before that token's decay (chunk 0's is `s0`, or zeros);
//! - the backward **consumes** those checkpoints, recomputing each 64-token
//!   chunk's states from its checkpoint before running the reverse-mode
//!   recurrence over it, as the kernel does. It is therefore a reference for
//!   the kernel's seam, not only for the operator.
//!
//! The recurrence is one function ([`recurrence_f64`]) over pre-normalized
//! operands, used by the seam ([`gdn_train_published_fwd_f64`]) and by the
//! check against tessl's published fixture corpus, so the code the corpus
//! validates is the code the seam runs. [`Rule::Repo`] (nanolab's default, the
//! undecayed read) exists only for the test that shows it does not reproduce
//! the published golden.
//!
//! # Validation
//!
//! - **golden** (forward recurrence): tessl's 13-case `gdn_published_*` corpus
//!   (numpy f64 sequential, nanolab `mixers.py`, `rule: published` in every
//!   manifest), including T = 1, 63, 64, 65 and 127/129, within `1e-12` of the
//!   golden's magnitude (tessl `tests/gdn_fixtures.rs:63`).
//! - **golden** (forward and backward at the seam, l2norm and scale included):
//!   `tests/fixtures/goldens/gdn_published_train_*`, transformers'
//!   `torch_recurrent_gated_delta_rule` transcribed at float64 with torch
//!   autograd for the backward, T = 1, 63, 64, 65, 130.
//! - **tessl in process**: tessl's `gdn_train_f64` / `gdn_train_bwd_f64`,
//!   compiled in unmodified, at T = 1, 63, 64, 65, 130 (tessl's own case list,
//!   `tests/gdn_train.rs:365-371`).
//! - **derivative**: central differences of the forward at T = 1, 63, 64, 65,
//!   130 through the checkpointed backward (tessl's own FD check runs T = 7
//!   only, `tests/gdn_train.rs:124-130`, so it never crosses a checkpoint).

/// Key head dim the CUDA kernels are compiled for (tessl `GDN_TRAIN_DK`,
/// `src/gdn_train.rs:39`). The reference itself takes any `dk`.
pub const KERNEL_DK: usize = 128;
/// Tokens between saved states (tessl `GDN_TRAIN_CKPT`, `src/gdn_train.rs:43`;
/// `kernels/gdn_train.metal:37`).
pub const CKPT: usize = 64;
/// transformers' `l2norm` epsilon (`modeling_qwen3_5.py:240-243`).
pub const L2_EPS: f64 = 1e-6;

/// Which state the delta correction reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rule {
    /// `arXiv:2412.06464` eq. 8 and transformers' recurrence: the correction
    /// reads the state after this token's decay. The operator this project
    /// targets.
    Published,
    /// nanolab's default `rule="repo"`: the correction reads the undecayed
    /// state. A different operator. Present for contrast only.
    Repo,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub b: usize,
    pub t: usize,
    pub h: usize,
    pub dk: usize,
    pub dv: usize,
}

impl Shape {
    /// `B * T * H`: the `(b, t, h)` rows of `q`, `k`, `v`, `g`, `beta`.
    pub fn rows(&self) -> usize {
        self.b * self.t * self.h
    }

    /// `ceil(T / CKPT)`: the checkpoints the forward saves per head.
    pub fn checkpoints(&self) -> usize {
        self.t.div_ceil(CKPT)
    }

    /// Elements of one state set, `[B, H, Dk, Dv]`.
    pub fn state_len(&self) -> usize {
        self.b * self.h * self.dk * self.dv
    }

    /// Elements of the checkpoint tensor, `[B, H, NC, Dk, Dv]`.
    pub fn checkpoint_len(&self) -> usize {
        self.b * self.h * self.checkpoints() * self.dk * self.dv
    }

    /// The kernel refuses a zero batch, sequence or head count
    /// (`src/gdn_train.rs:57-59`); so does the reference, and a zero head dim.
    fn validate(&self) {
        assert!(
            self.b > 0 && self.t > 0 && self.h > 0 && self.dk > 0 && self.dv > 0,
            "gdn_published: every dimension must be non-zero, got {self:?}"
        );
    }
}

/// Operands in the seam's layout: `q`, `k` `[B, T, H, Dk]`, `v` `[B, T, H, Dv]`,
/// `g`, `beta` `[B, T, H]`, `s0` `[B, H, Dk, Dv]`.
#[derive(Clone, Copy, Debug)]
pub struct Inputs<'a> {
    pub s: Shape,
    pub q: &'a [f64],
    pub k: &'a [f64],
    pub v: &'a [f64],
    pub g: &'a [f64],
    pub beta: &'a [f64],
    pub s0: Option<&'a [f64]>,
}

impl Inputs<'_> {
    /// Panics on a length that disagrees with the shape or a non-finite
    /// operand: a reference that quietly consumed either would return
    /// plausible numbers for a different problem.
    fn validate(&self) {
        let s = self.s;
        s.validate();
        let n = s.rows();
        for (name, got, want) in [
            ("q", self.q.len(), n * s.dk),
            ("k", self.k.len(), n * s.dk),
            ("v", self.v.len(), n * s.dv),
            ("g", self.g.len(), n),
            ("beta", self.beta.len(), n),
        ] {
            assert_eq!(
                got, want,
                "gdn_published: {name} has {got} elements, want {want} for {s:?}"
            );
        }
        if let Some(s0) = self.s0 {
            assert_eq!(
                s0.len(),
                s.state_len(),
                "gdn_published: s0 length for {s:?}"
            );
        }
        for (name, xs) in [
            ("q", self.q),
            ("k", self.k),
            ("v", self.v),
            ("g", self.g),
            ("beta", self.beta),
        ] {
            assert_finite(name, xs);
        }
        if let Some(s0) = self.s0 {
            assert_finite("s0", s0);
        }
    }
}

/// What the forward produces: `o` `[B, T, H, Dv]`, the final state
/// `[B, H, Dk, Dv]`, and the checkpoints `[B, H, NC, Dk, Dv]`.
#[derive(Clone, Debug)]
pub struct Forward {
    pub o: Vec<f64>,
    pub fin: Vec<f64>,
    pub ckpt: Vec<f64>,
}

/// Gradients of `sum(d_o * o) + sum(dfin * final_state)` with respect to every
/// input, in the inputs' layouts. `ds0` is the gradient with respect to the
/// initial state, zeros or given.
#[derive(Clone, Debug)]
pub struct Grads {
    pub dq: Vec<f64>,
    pub dk: Vec<f64>,
    pub dv: Vec<f64>,
    pub dg: Vec<f64>,
    pub dbeta: Vec<f64>,
    pub ds0: Vec<f64>,
}

fn assert_finite(name: &str, xs: &[f64]) {
    if let Some(i) = xs.iter().position(|x| !x.is_finite()) {
        panic!("gdn_published: {name}[{i}] is not finite: {}", xs[i]);
    }
}

/// `usize` to `f64`, checked: every count here is far below 2^53, and a value
/// that is not would make `1/sqrt(Dk)` silently wrong.
pub fn exact_f64(n: usize) -> f64 {
    f64::from(u32::try_from(n).expect("a dimension fits u32"))
}

/// transformers' `l2norm`: `(x * r, r)` with `r = 1 / sqrt(|x|^2 + 1e-6)`.
pub fn l2norm(x: &[f64]) -> (Vec<f64>, f64) {
    let r = 1.0 / (x.iter().map(|v| v * v).sum::<f64>() + L2_EPS).sqrt();
    (x.iter().map(|v| v * r).collect(), r)
}

/// Each `d`-wide row of `x` through [`l2norm`]: the normalized rows and each
/// row's `r`.
fn l2norm_rows(x: &[f64], d: usize) -> (Vec<f64>, Vec<f64>) {
    let mut out = Vec::with_capacity(x.len());
    let mut rs = Vec::with_capacity(x.len() / d);
    for row in x.chunks_exact(d) {
        let (n, r) = l2norm(row);
        out.extend(n);
        rs.push(r);
    }
    (out, rs)
}

/// One token of the recurrence on one head's state `s` `[dk, dv]`, in place.
/// `kh` is the key as the recurrence reads it (normalized at the seam), `a`
/// the decay `exp(g)`. Writes `u = v - read(S)^T k^`, the correction before
/// `beta`, into `u`.
fn token(s: &mut [f64], kh: &[f64], v: &[f64], a: f64, beta: f64, rule: Rule, u: &mut [f64]) {
    let (dk, dv) = (kh.len(), v.len());
    match rule {
        Rule::Published => {
            for x in s.iter_mut() {
                *x *= a;
            }
            for j in 0..dv {
                let kv: f64 = (0..dk).map(|i| s[i * dv + j] * kh[i]).sum();
                u[j] = v[j] - kv;
                let delta = beta * u[j];
                for i in 0..dk {
                    s[i * dv + j] += kh[i] * delta;
                }
            }
        }
        Rule::Repo => {
            for j in 0..dv {
                let kv: f64 = (0..dk).map(|i| s[i * dv + j] * kh[i]).sum();
                u[j] = v[j] - kv;
            }
            for x in s.iter_mut() {
                *x *= a;
            }
            for j in 0..dv {
                let delta = beta * u[j];
                for i in 0..dk {
                    s[i * dv + j] += kh[i] * delta;
                }
            }
        }
    }
}

/// The recurrence over operands already in the form it reads them: `qn`
/// `[B, T, H, Dk]` (the output is `scale * S_t^T qn`, the scale applied per
/// term as tessl does), `kh` `[B, T, H, Dk]` read as given, `v`, `g` (log
/// decay), `beta`, optional `s0`. The seam passes `l2norm` rows and
/// `scale = Dk^-1/2`; the published fixture corpus passes its stored `q` and
/// `k` (the corpus's `k` is pre-normalized and its output unscaled) with
/// `scale = 1` and `g = ln(alpha)`.
#[allow(clippy::too_many_arguments)]
pub fn recurrence_f64(
    s: Shape,
    qn: &[f64],
    kh: &[f64],
    v: &[f64],
    g: &[f64],
    beta: &[f64],
    s0: Option<&[f64]>,
    scale: f64,
    rule: Rule,
) -> Forward {
    Inputs {
        s,
        q: qn,
        k: kh,
        v,
        g,
        beta,
        s0,
    }
    .validate();
    let Shape { b, t, h, dk, dv } = s;
    let nc = s.checkpoints();
    let per = dk * dv;
    let mut o = vec![0.0; s.rows() * dv];
    let mut fin = vec![0.0; s.state_len()];
    let mut ckpt = vec![0.0; s.checkpoint_len()];
    let mut u = vec![0.0; dv];
    for bi in 0..b {
        for hi in 0..h {
            let bh = bi * h + hi;
            let mut st: Vec<f64> = match s0 {
                Some(s0) => s0[bh * per..(bh + 1) * per].to_vec(),
                None => vec![0.0; per],
            };
            for ti in 0..t {
                if ti % CKPT == 0 {
                    let c = (bh * nc + ti / CKPT) * per;
                    ckpt[c..c + per].copy_from_slice(&st);
                }
                let row = (bi * t + ti) * h + hi;
                token(
                    &mut st,
                    &kh[row * dk..(row + 1) * dk],
                    &v[row * dv..(row + 1) * dv],
                    g[row].exp(),
                    beta[row],
                    rule,
                    &mut u,
                );
                let qr = &qn[row * dk..(row + 1) * dk];
                for j in 0..dv {
                    o[row * dv + j] = (0..dk).map(|i| st[i * dv + j] * qr[i] * scale).sum();
                }
            }
            fin[bh * per..(bh + 1) * per].copy_from_slice(&st);
        }
    }
    Forward { o, fin, ckpt }
}

/// The forward at the seam, published rule: `l2norm` on `q` and `k`, the
/// `Dk^-1/2` scale, and the checkpoints the backward consumes.
pub fn gdn_train_published_fwd_f64(p: &Inputs<'_>) -> Forward {
    p.validate();
    let (qn, _) = l2norm_rows(p.q, p.s.dk);
    let (kh, _) = l2norm_rows(p.k, p.s.dk);
    let scale = exact_f64(p.s.dk).powf(-0.5);
    recurrence_f64(
        p.s,
        &qn,
        &kh,
        p.v,
        p.g,
        p.beta,
        p.s0,
        scale,
        Rule::Published,
    )
}

/// The backward at the seam, published rule. Consumes the forward's
/// checkpoints (`[B, H, NC, Dk, Dv]`) and recomputes each chunk's states from
/// them, as `kernels/gdn_train.metal:197-206` does; then the reverse-mode
/// recurrence of tessl's `gdn_train_bwd_f64`, term for term.
pub fn gdn_train_published_bwd_f64(
    p: &Inputs<'_>,
    ckpt: &[f64],
    d_o: &[f64],
    dfin: Option<&[f64]>,
) -> Grads {
    p.validate();
    let s = p.s;
    let Shape { b, t, h, dk, dv } = s;
    assert_eq!(
        ckpt.len(),
        s.checkpoint_len(),
        "gdn_published bwd: ckpt length for {s:?}"
    );
    assert_eq!(
        d_o.len(),
        s.rows() * dv,
        "gdn_published bwd: d_o length for {s:?}"
    );
    if let Some(d) = dfin {
        assert_eq!(
            d.len(),
            s.state_len(),
            "gdn_published bwd: dfin length for {s:?}"
        );
        assert_finite("dfin", d);
    }
    assert_finite("ckpt", ckpt);
    assert_finite("d_o", d_o);

    let scale = exact_f64(dk).powf(-0.5);
    let (qn_all, rq_all) = l2norm_rows(p.q, dk);
    let (kh_all, rk_all) = l2norm_rows(p.k, dk);
    let nc = s.checkpoints();
    let per = dk * dv;
    let mut gr = Grads {
        dq: vec![0.0; s.rows() * dk],
        dk: vec![0.0; s.rows() * dk],
        dv: vec![0.0; s.rows() * dv],
        dg: vec![0.0; s.rows()],
        dbeta: vec![0.0; s.rows()],
        ds0: vec![0.0; s.state_len()],
    };
    let mut u_scratch = vec![0.0; dv];
    for bi in 0..b {
        for hi in 0..h {
            let bh = bi * h + hi;
            let mut ds: Vec<f64> = match dfin {
                Some(d) => d[bh * per..(bh + 1) * per].to_vec(),
                None => vec![0.0; per],
            };
            for cc in (0..nc).rev() {
                let (t0, t1) = (cc * CKPT, t.min(cc * CKPT + CKPT));
                // Recompute S_{t-1} for every token of the chunk from its checkpoint.
                let mut st = ckpt[(bh * nc + cc) * per..(bh * nc + cc + 1) * per].to_vec();
                let mut states = Vec::with_capacity(t1 - t0);
                for ti in t0..t1 {
                    states.push(st.clone());
                    let row = (bi * t + ti) * h + hi;
                    token(
                        &mut st,
                        &kh_all[row * dk..(row + 1) * dk],
                        &p.v[row * dv..(row + 1) * dv],
                        p.g[row].exp(),
                        p.beta[row],
                        Rule::Published,
                        &mut u_scratch,
                    );
                }
                for ti in (t0..t1).rev() {
                    let row = (bi * t + ti) * h + hi;
                    let qn = &qn_all[row * dk..(row + 1) * dk];
                    let kh = &kh_all[row * dk..(row + 1) * dk];
                    let (rq, rk) = (rq_all[row], rk_all[row]);
                    let qh: Vec<f64> = qn.iter().map(|x| x * scale).collect();
                    let a = p.g[row].exp();
                    let beta = p.beta[row];
                    let sh: Vec<f64> = states[ti - t0].iter().map(|x| x * a).collect();
                    let u: Vec<f64> = (0..dv)
                        .map(|j| {
                            p.v[row * dv + j] - (0..dk).map(|i| sh[i * dv + j] * kh[i]).sum::<f64>()
                        })
                        .collect();
                    let delta: Vec<f64> = u.iter().map(|x| beta * x).collect();
                    let stn: Vec<f64> = (0..per)
                        .map(|x| sh[x] + kh[x / dv] * delta[x % dv])
                        .collect();
                    let dor = &d_o[row * dv..(row + 1) * dv];
                    // o = S_t^T q^
                    let mut dqh = vec![0.0; dk];
                    for i in 0..dk {
                        dqh[i] = (0..dv).map(|j| stn[i * dv + j] * dor[j]).sum();
                        for j in 0..dv {
                            ds[i * dv + j] += qh[i] * dor[j];
                        }
                    }
                    // S_t = S^ + k^ delta^T
                    let ddelta: Vec<f64> = (0..dv)
                        .map(|j| (0..dk).map(|i| ds[i * dv + j] * kh[i]).sum())
                        .collect();
                    let mut dkh: Vec<f64> = (0..dk)
                        .map(|i| (0..dv).map(|j| ds[i * dv + j] * delta[j]).sum())
                        .collect();
                    // delta = beta (v - S^^T k^)
                    for (d, dd) in gr.dv[row * dv..(row + 1) * dv].iter_mut().zip(&ddelta) {
                        *d = beta * dd;
                    }
                    gr.dbeta[row] = (0..dv).map(|j| ddelta[j] * u[j]).sum();
                    // dS^ = dS - beta k^ ddelta^T ; dk^ -= beta S^ ddelta
                    let mut dsh = ds.clone();
                    for i in 0..dk {
                        for j in 0..dv {
                            dsh[i * dv + j] -= beta * kh[i] * ddelta[j];
                            dkh[i] -= beta * sh[i * dv + j] * ddelta[j];
                        }
                    }
                    // S^ = a S_{t-1}: dg = <dS^, S^>, dS_{t-1} = a dS^
                    gr.dg[row] = (0..per).map(|x| dsh[x] * sh[x]).sum();
                    for x in 0..per {
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
            }
            gr.ds0[bh * per..(bh + 1) * per].copy_from_slice(&ds);
        }
    }
    gr
}

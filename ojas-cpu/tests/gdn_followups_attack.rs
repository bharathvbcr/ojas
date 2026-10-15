//! Hostile cases for the GDN forward scratch cut, mul's 4-piece band, and the
//! 2 MiB `chunks_into_n` serial cut.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use ojas_core::{
    exp_exact, sdpa_scale, Backend, Budget, GdnInputs, Numerics, OjasError, Tensor,
    GDN_CHECKPOINT_TOKENS, GDN_L2NORM_EPS, RMS_NORM_EPS,
};
use ojas_cpu::CpuBackend;

fn bits(xs: &[f32]) -> Vec<u32> {
    xs.iter().map(|x| x.to_bits()).collect()
}

fn cpu(threads: usize, numerics: Numerics) -> CpuBackend {
    let budget = Budget::new(1 << 30);
    let cpu = if threads == 1 {
        CpuBackend::new(budget)
    } else {
        CpuBackend::with_threads(budget, threads).unwrap()
    };
    cpu.with_numerics(numerics)
}

fn assert_sync(cpu: &CpuBackend) {
    fn is_sync<T: Sync>(_: &T) {}
    is_sync(cpu);
}

fn pattern(n: usize, scale: f32, bias: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i % 17) as f32 - 8.0) * scale + bias)
        .collect()
}

#[derive(Clone, Copy)]
struct Shape {
    b: usize,
    t: usize,
    h: usize,
    dk: usize,
    dv: usize,
}

struct Raw {
    shape: Shape,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    g: Vec<f32>,
    beta: Vec<f32>,
    s0: Option<Vec<f32>>,
}

fn rows(s: &Shape) -> usize {
    s.b * s.t * s.h
}

fn normal_raw(s: Shape, with_state: bool) -> Raw {
    let qn = rows(&s) * s.dk;
    let vn = rows(&s) * s.dv;
    let gn = rows(&s);
    let s0 = if with_state {
        let sn = s.b * s.h * s.dk * s.dv;
        let mut state = pattern(sn, 0.01, 0.0);
        if sn > 0 {
            state[0] = -0.0;
        }
        Some(state)
    } else {
        None
    };
    Raw {
        q: pattern(qn, 0.05, 0.1),
        k: pattern(qn, 0.04, -0.2),
        v: pattern(vn, 0.03, 0.05),
        g: pattern(gn, 0.01, -0.05),
        beta: pattern(gn, 0.02, 0.4),
        s0,
        shape: s,
    }
}

/// One 1e20 row (f32 sum of squares overflows), one subnormal row, one
/// negative-zero row, then a mixed row. `v` stays O(1) so a correct f64
/// l2norm cannot collapse the output to zeros.
fn extreme_raw(with_state: bool) -> Raw {
    let s = Shape {
        b: 2,
        t: 4,
        h: 1,
        dk: 4,
        dv: 3,
    };
    let qn = rows(&s) * s.dk;
    let mut q = pattern(qn, 0.05, 0.1);
    let mut k = pattern(qn, 0.04, -0.2);
    let v = pattern(rows(&s) * s.dv, 0.03, 0.5);
    let mut g = vec![0.0f32; rows(&s)];
    let beta = vec![0.5f32; rows(&s)];
    for row in 0..rows(&s) {
        let qrow = &mut q[row * s.dk..(row + 1) * s.dk];
        let krow = &mut k[row * s.dk..(row + 1) * s.dk];
        match row % 4 {
            0 => {
                qrow.fill(1e20);
                krow.fill(1e20);
            }
            1 => {
                qrow.fill(f32::from_bits(1));
                krow.fill(f32::from_bits(1));
            }
            2 => {
                qrow.fill(-0.0);
                krow.fill(-0.0);
            }
            _ => {
                qrow[0] = -0.0;
                qrow[1] = 1e20;
                qrow[2] = f32::from_bits(1);
                krow[0] = 1e20;
                krow[1] = -0.0;
                krow[3] = f32::from_bits(0x8000_0001);
            }
        }
    }
    g[3] = -0.0;
    let s0 = if with_state {
        let mut state = pattern(s.b * s.h * s.dk * s.dv, 0.02, 0.0);
        state[0] = -0.0;
        state[1] = f32::from_bits(1);
        Some(state)
    } else {
        None
    };
    Raw {
        shape: s,
        q,
        k,
        v,
        g,
        beta,
        s0,
    }
}

fn stack_tokens(h1: &[f32], other: &[f32], tokens: usize, width: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(tokens * 2 * width);
    for i in 0..tokens {
        out.extend_from_slice(&h1[i * width..(i + 1) * width]);
        out.extend_from_slice(&other[i * width..(i + 1) * width]);
    }
    out
}

fn stack_states(h1: &[f32], other: &[f32], b: usize, per: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(b * 2 * per);
    for bi in 0..b {
        out.extend_from_slice(&h1[bi * per..(bi + 1) * per]);
        out.extend_from_slice(&other[bi * per..(bi + 1) * per]);
    }
    out
}

/// `H = 2` tensor whose head 0 is `raw` (itself `H = 1`) and whose head 1 is
/// a different finite pattern, so a stride mix-up cannot hide.
fn stack_h2(raw: &Raw) -> Raw {
    assert_eq!(raw.shape.h, 1);
    let s = &raw.shape;
    let other_q = pattern(s.b * s.t * s.dk, 0.07, 1.7);
    let other_k = pattern(s.b * s.t * s.dk, -0.06, 0.3);
    let other_v = pattern(s.b * s.t * s.dv, 0.08, -0.4);
    let other_g = pattern(s.b * s.t, 0.03, 0.2);
    let other_beta = pattern(s.b * s.t, 0.01, 0.1);
    let s0 = raw.s0.as_ref().map(|state| {
        let per = s.dk * s.dv;
        let other = pattern(s.b * per, 0.05, -0.2);
        stack_states(state, &other, s.b, per)
    });
    Raw {
        q: stack_tokens(&raw.q, &other_q, s.b * s.t, s.dk),
        k: stack_tokens(&raw.k, &other_k, s.b * s.t, s.dk),
        v: stack_tokens(&raw.v, &other_v, s.b * s.t, s.dv),
        g: stack_tokens(&raw.g, &other_g, s.b * s.t, 1),
        beta: stack_tokens(&raw.beta, &other_beta, s.b * s.t, 1),
        s0,
        shape: Shape { h: 2, ..*s },
    }
}

struct Case {
    raw: Raw,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    s0: Option<Tensor>,
}

fn load(budget: &Budget, raw: &Raw) -> Case {
    let s = &raw.shape;
    let q = Tensor::from_f32(&raw.q, &[s.b, s.t, s.h, s.dk], budget).unwrap();
    let k = Tensor::from_f32(&raw.k, &[s.b, s.t, s.h, s.dk], budget).unwrap();
    let v = Tensor::from_f32(&raw.v, &[s.b, s.t, s.h, s.dv], budget).unwrap();
    let g = Tensor::from_f32(&raw.g, &[s.b, s.t, s.h], budget).unwrap();
    let beta = Tensor::from_f32(&raw.beta, &[s.b, s.t, s.h], budget).unwrap();
    let s0 = raw
        .s0
        .as_ref()
        .map(|state| Tensor::from_f32(state, &[s.b, s.h, s.dk, s.dv], budget).unwrap());
    Case {
        raw: Raw {
            shape: Shape { ..*s },
            q: raw.q.clone(),
            k: raw.k.clone(),
            v: raw.v.clone(),
            g: raw.g.clone(),
            beta: raw.beta.clone(),
            s0: raw.s0.clone(),
        },
        q,
        k,
        v,
        g,
        beta,
        s0,
    }
}

fn inputs(c: &Case) -> GdnInputs<'_> {
    GdnInputs {
        q: &c.q,
        k: &c.k,
        v: &c.v,
        g: &c.g,
        beta: &c.beta,
        initial_state: c.s0.as_ref(),
    }
}

fn head0(full: &[f32], groups: usize, heads: usize, width: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(groups * width);
    for g in 0..groups {
        let start = (g * heads) * width;
        out.extend_from_slice(&full[start..start + width]);
    }
    out
}

struct FwdBits {
    out: Vec<u32>,
    state: Vec<u32>,
    ckpt: Vec<u32>,
}

struct BwdBits {
    q: Vec<u32>,
    k: Vec<u32>,
    v: Vec<u32>,
    g: Vec<u32>,
    beta: Vec<u32>,
    s0: Option<Vec<u32>>,
}

fn forward_head0(cpu: &CpuBackend, c: &Case) -> (FwdBits, ojas_core::GdnForward) {
    let y = cpu.chunked_gdn_forward(inputs(c)).unwrap();
    let s = &c.raw.shape;
    let out = y.output.to_f32_vec().unwrap();
    let state = y.final_state.to_f32_vec().unwrap();
    let ckpt = y.checkpoints.to_f32_vec().unwrap();
    let nc = s.t.div_ceil(GDN_CHECKPOINT_TOKENS);
    (
        FwdBits {
            out: bits(&head0(&out, s.b * s.t, s.h, s.dv)),
            state: bits(&head0(&state, s.b, s.h, s.dk * s.dv)),
            ckpt: bits(&head0(&ckpt, s.b, s.h, nc * s.dk * s.dv)),
        },
        y,
    )
}

/// Upstream gradient whose head 0 does not depend on `H`, so an `H = 1`
/// backward is comparable to head 0 of `H > 1`. A flat `pattern` is not:
/// head 1 shifts every later index.
fn head_major_pattern(groups: usize, heads: usize, width: usize, head0_bias: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; groups * heads * width];
    for g in 0..groups {
        for h in 0..heads {
            for d in 0..width {
                let bias = if h == 0 { head0_bias } else { 0.8 };
                out[(g * heads + h) * width + d] = bias + d as f32 * 0.01 + g as f32 * 0.001;
            }
        }
    }
    out
}

fn backward_head0(
    cpu: &CpuBackend,
    c: &Case,
    y: &ojas_core::GdnForward,
    grad_state: bool,
) -> BwdBits {
    let s = &c.raw.shape;
    let gy_full = head_major_pattern(s.b * s.t, s.h, s.dv, 0.02);
    let gy = Tensor::from_f32(&gy_full, y.output.shape(), cpu.budget()).unwrap();
    let gs = if grad_state {
        let gs_full = head_major_pattern(s.b, s.h, s.dk * s.dv, 0.01);
        Some(Tensor::from_f32(&gs_full, y.final_state.shape(), cpu.budget()).unwrap())
    } else {
        None
    };
    let g = cpu
        .chunked_gdn_backward(inputs(c), &y.checkpoints, &gy, gs.as_ref())
        .unwrap();
    let dq = g.q.to_f32_vec().unwrap();
    let dk = g.k.to_f32_vec().unwrap();
    let dv = g.v.to_f32_vec().unwrap();
    let dg = g.g.to_f32_vec().unwrap();
    let db = g.beta.to_f32_vec().unwrap();
    BwdBits {
        q: bits(&head0(&dq, s.b * s.t, s.h, s.dk)),
        k: bits(&head0(&dk, s.b * s.t, s.h, s.dk)),
        v: bits(&head0(&dv, s.b * s.t, s.h, s.dv)),
        g: bits(&head0(&dg, s.b * s.t, s.h, 1)),
        beta: bits(&head0(&db, s.b * s.t, s.h, 1)),
        s0: g.initial_state.map(|ds| {
            let v = ds.to_f32_vec().unwrap();
            bits(&head0(&v, s.b, s.h, s.dk * s.dv))
        }),
    }
}

fn ckpt0_matches_initial(raw: &Raw, ckpt: &[f32]) {
    let s = &raw.shape;
    let per = s.dk * s.dv;
    let nc = s.t.div_ceil(GDN_CHECKPOINT_TOKENS);
    let zeros = vec![0.0f32; per];
    for head in 0..(s.b * s.h) {
        let start = head * nc * per;
        let expect = raw
            .s0
            .as_ref()
            .map(|state| &state[head * per..(head + 1) * per])
            .unwrap_or(&zeros);
        assert_eq!(
            &ckpt[start..start + per],
            expect,
            "checkpoint 0 head {head} is not the state entering token 0"
        );
    }
}

fn oracle_forward(raw: &Raw) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let s = &raw.shape;
    let scale = sdpa_scale(s.dk as u32).unwrap();
    let per = s.dk * s.dv;
    let nc = s.t.div_ceil(GDN_CHECKPOINT_TOKENS);
    let mut out = vec![0.0f32; s.b * s.t * s.h * s.dv];
    let mut state = vec![0.0f32; s.b * s.h * per];
    let mut ckpt = vec![0.0f32; s.b * s.h * nc * per];
    let l2 = |row: &[f32]| -> Vec<f32> {
        let mut ss = 0.0f64;
        for &v in row {
            let v = f64::from(v);
            ss += v * v;
        }
        let r = 1.0 / (ss + f64::from(GDN_L2NORM_EPS)).sqrt();
        row.iter().map(|&v| (f64::from(v) * r) as f32).collect()
    };
    for b in 0..s.b {
        for h in 0..s.h {
            let head = b * s.h + h;
            let mut st = raw
                .s0
                .as_ref()
                .map(|s0| s0[head * per..(head + 1) * per].to_vec())
                .unwrap_or_else(|| vec![0.0; per]);
            for ti in 0..s.t {
                if ti % GDN_CHECKPOINT_TOKENS == 0 {
                    let c = (head * nc + ti / GDN_CHECKPOINT_TOKENS) * per;
                    ckpt[c..c + per].copy_from_slice(&st);
                }
                let row = (b * s.t + ti) * s.h + h;
                let mut q = l2(&raw.q[row * s.dk..(row + 1) * s.dk]);
                for qv in &mut q {
                    *qv *= scale;
                }
                let k = l2(&raw.k[row * s.dk..(row + 1) * s.dk]);
                let v = &raw.v[row * s.dv..(row + 1) * s.dv];
                let a = exp_exact(raw.g[row]);
                let beta = raw.beta[row];
                for slot in &mut st {
                    *slot *= a;
                }
                let mut u = vec![0.0f32; s.dv];
                for (srow, &ki) in st.chunks_exact(s.dv).zip(&k) {
                    for (uj, &sv) in u.iter_mut().zip(srow) {
                        *uj += sv * ki;
                    }
                }
                let mut delta = vec![0.0f32; s.dv];
                for j in 0..s.dv {
                    u[j] = v[j] - u[j];
                    delta[j] = beta * u[j];
                }
                for (srow, &ki) in st.chunks_exact_mut(s.dv).zip(&k) {
                    for (slot, &dj) in srow.iter_mut().zip(&delta) {
                        *slot += ki * dj;
                    }
                }
                let mut o = vec![0.0f32; s.dv];
                for (srow, &qi) in st.chunks_exact(s.dv).zip(&q) {
                    for (oj, &sv) in o.iter_mut().zip(srow) {
                        *oj += sv * qi;
                    }
                }
                let at = row * s.dv;
                out[at..at + s.dv].copy_from_slice(&o);
            }
            state[head * per..(head + 1) * per].copy_from_slice(&st);
        }
    }
    (out, state, ckpt)
}

#[test]
fn h1_matches_h2_head0_threads_numerics_extremes_and_oracle() {
    let base = cpu(1, Numerics::Fast);
    assert_sync(&base);
    let shapes = [
        Shape {
            b: 2,
            t: 1,
            h: 1,
            dk: 1,
            dv: 1,
        },
        Shape {
            b: 2,
            t: 1,
            h: 1,
            dk: 1,
            dv: 4,
        },
        Shape {
            b: 1,
            t: GDN_CHECKPOINT_TOKENS,
            h: 1,
            dk: 3,
            dv: 5,
        },
        Shape {
            b: 2,
            t: GDN_CHECKPOINT_TOKENS + 1,
            h: 1,
            dk: 4,
            dv: 1,
        },
        Shape {
            b: 1,
            t: 2 * GDN_CHECKPOINT_TOKENS,
            h: 1,
            dk: 8,
            dv: 2,
        },
        Shape {
            b: 2,
            t: 3,
            h: 1,
            dk: 5,
            dv: 7,
        },
    ];
    let mut raws = Vec::new();
    for shape in shapes {
        raws.push(normal_raw(shape, false));
        raws.push(normal_raw(shape, true));
    }
    raws.push(extreme_raw(false));
    raws.push(extreme_raw(true));

    let configs = [
        (1usize, Numerics::Fast),
        (6, Numerics::Fast),
        (8, Numerics::Fast),
        (6, Numerics::Exact),
        (1, Numerics::Exact),
    ];
    for (ri, raw) in raws.iter().enumerate() {
        let wide = stack_h2(raw);
        let mut fwd_ref: Option<FwdBits> = None;
        let mut bwd_ref: Option<BwdBits> = None;
        for (threads, numerics) in configs {
            for (tag, spec) in [("h1", raw), ("h2", &wide)] {
                let backend = cpu(threads, numerics);
                let case = load(backend.budget(), spec);
                let (fwd, y) = forward_head0(&backend, &case);
                let out = y.output.to_f32_vec().unwrap();
                let state = y.final_state.to_f32_vec().unwrap();
                let ckpt = y.checkpoints.to_f32_vec().unwrap();
                assert!(
                    out.iter().chain(&state).chain(&ckpt).all(|v| v.is_finite()),
                    "case {ri} {tag} threads {threads} {numerics:?} produced a non-finite"
                );
                ckpt0_matches_initial(spec, &ckpt);
                let bwd = backward_head0(&backend, &case, &y, spec.s0.is_some());
                if tag == "h1" && threads == 1 && numerics == Numerics::Fast {
                    let (o, st, c) = oracle_forward(spec);
                    assert_eq!(bits(&o), bits(&out), "oracle output case {ri}");
                    assert_eq!(bits(&st), bits(&state), "oracle state case {ri}");
                    assert_eq!(bits(&c), bits(&ckpt), "oracle checkpoints case {ri}");
                    fwd_ref = Some(FwdBits {
                        out: fwd.out.clone(),
                        state: fwd.state.clone(),
                        ckpt: fwd.ckpt.clone(),
                    });
                    bwd_ref = Some(BwdBits {
                        q: bwd.q.clone(),
                        k: bwd.k.clone(),
                        v: bwd.v.clone(),
                        g: bwd.g.clone(),
                        beta: bwd.beta.clone(),
                        s0: bwd.s0.clone(),
                    });
                }
                let fwd_ref = fwd_ref.as_ref().unwrap();
                let bwd_ref = bwd_ref.as_ref().unwrap();
                assert_eq!(
                    fwd.out, fwd_ref.out,
                    "out case {ri} {tag} {threads} {numerics:?}"
                );
                assert_eq!(
                    fwd.state, fwd_ref.state,
                    "state case {ri} {tag} {threads} {numerics:?}"
                );
                assert_eq!(
                    fwd.ckpt, fwd_ref.ckpt,
                    "ckpt case {ri} {tag} {threads} {numerics:?}"
                );
                assert_eq!(
                    bwd.q, bwd_ref.q,
                    "dq case {ri} {tag} {threads} {numerics:?}"
                );
                assert_eq!(
                    bwd.k, bwd_ref.k,
                    "dk case {ri} {tag} {threads} {numerics:?}"
                );
                assert_eq!(
                    bwd.v, bwd_ref.v,
                    "dv case {ri} {tag} {threads} {numerics:?}"
                );
                assert_eq!(
                    bwd.g, bwd_ref.g,
                    "dg case {ri} {tag} {threads} {numerics:?}"
                );
                assert_eq!(
                    bwd.beta, bwd_ref.beta,
                    "dbeta case {ri} {tag} {threads} {numerics:?}"
                );
                assert_eq!(
                    bwd.s0, bwd_ref.s0,
                    "ds0 case {ri} {tag} {threads} {numerics:?}"
                );
            }
        }
        if raw.q.iter().any(|v| v.abs() > 1e19) {
            let backend = cpu(6, Numerics::Fast);
            let case = load(backend.budget(), raw);
            let y = backend.chunked_gdn_forward(inputs(&case)).unwrap();
            let out = y.output.to_f32_vec().unwrap();
            assert!(
                out.iter().any(|v| v.is_finite() && v.abs() > 1e-3),
                "1e20 row normalized to a zero output: {out:?}"
            );
        }
    }
}

#[test]
fn checkpoint_after_one_chunk_matches_the_shorter_forward() {
    let long_t = GDN_CHECKPOINT_TOKENS + 3;
    for h in [1usize, 3] {
        let shape = Shape {
            b: 2,
            t: long_t,
            h,
            dk: 4,
            dv: 3,
        };
        let mut raw = normal_raw(shape, true);
        let prefix = truncate_tokens(&raw, GDN_CHECKPOINT_TOKENS);
        // Tokens at and after the checkpoint boundary must not be required
        // to form that checkpoint. Poison only those tokens.
        let row_q = shape.h * shape.dk;
        for bi in 0..shape.b {
            let from = (bi * shape.t + GDN_CHECKPOINT_TOKENS) * row_q;
            let to = (bi + 1) * shape.t * row_q;
            for q in &mut raw.q[from..to] {
                *q = 1e20;
            }
        }
        let backend = cpu(8, Numerics::Exact);
        let long = load(backend.budget(), &raw);
        let short_case = load(backend.budget(), &prefix);
        let y_long = backend.chunked_gdn_forward(inputs(&long)).unwrap();
        let y_short = backend.chunked_gdn_forward(inputs(&short_case)).unwrap();
        let ckpt = y_long.checkpoints.to_f32_vec().unwrap();
        let state = y_short.final_state.to_f32_vec().unwrap();
        let per = shape.dk * shape.dv;
        let nc = long_t.div_ceil(GDN_CHECKPOINT_TOKENS);
        let heads = shape.b * shape.h;
        for head in 0..heads {
            let from = (head * nc + 1) * per;
            let expect = &state[head * per..(head + 1) * per];
            assert_eq!(
                &ckpt[from..from + per],
                expect,
                "H={h} checkpoint 1 != state after {GDN_CHECKPOINT_TOKENS} tokens"
            );
        }
        let one = truncate_tokens(&raw, 1);
        let y1 = {
            let case = load(backend.budget(), &one);
            backend.chunked_gdn_forward(inputs(&case)).unwrap()
        };
        let first = y1.output.to_f32_vec().unwrap();
        let long_out = y_long.output.to_f32_vec().unwrap();
        let width = shape.h * shape.dv;
        for b in 0..shape.b {
            let at = b * long_t * width;
            assert_eq!(
                &long_out[at..at + width],
                &first[b * width..(b + 1) * width],
                "H={h} token 0 depends on later tokens"
            );
        }
    }
}

/// Per-batch token prefix. Flat `[B, T, H, width]` is batch-major, so the
/// first `B * T_short * …` elements are not that prefix when `B > 1`.
fn copy_token_prefix(
    dst: &mut [f32],
    src: &[f32],
    b: usize,
    t_dst: usize,
    t_src: usize,
    row: usize,
) {
    for bi in 0..b {
        let from = bi * t_src * row;
        let to = bi * t_dst * row;
        dst[to..to + t_dst * row].copy_from_slice(&src[from..from + t_dst * row]);
    }
}

fn truncate_tokens(raw: &Raw, t: usize) -> Raw {
    let s = raw.shape;
    let mut out = normal_raw(Shape { t, ..s }, raw.s0.is_some());
    copy_token_prefix(&mut out.q, &raw.q, s.b, t, s.t, s.h * s.dk);
    copy_token_prefix(&mut out.k, &raw.k, s.b, t, s.t, s.h * s.dk);
    copy_token_prefix(&mut out.v, &raw.v, s.b, t, s.t, s.h * s.dv);
    copy_token_prefix(&mut out.g, &raw.g, s.b, t, s.t, s.h);
    copy_token_prefix(&mut out.beta, &raw.beta, s.b, t, s.t, s.h);
    out.s0 = raw.s0.clone();
    out
}

#[test]
fn nonfinite_inputs_are_typed_errors_and_do_not_leak() {
    let shape_h = 4usize;
    let specs: &[(&str, usize)] = &[
        ("q", 1),
        ("k", 1),
        ("v", 2),
        ("g", 3),
        ("beta", 3),
        ("s0", 4),
    ];
    for h in [1usize, shape_h] {
        for &(name, _rank) in specs {
            for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let b = 2usize;
                let t = 3usize;
                let dk = 3usize;
                let dv = 2usize;
                let rows = b * t * h;
                let mut q = pattern(rows * dk, 0.05, 0.1);
                let mut k = pattern(rows * dk, 0.04, -0.2);
                let mut v = pattern(rows * dv, 0.03, 0.05);
                let mut g = pattern(rows, 0.01, -0.05);
                let mut beta = pattern(rows, 0.02, 0.4);
                let mut s0 = pattern(b * h * dk * dv, 0.01, 0.0);
                let poke = |xs: &mut [f32]| {
                    xs[xs.len() - 1] = bad;
                };
                match name {
                    "q" => poke(&mut q),
                    "k" => poke(&mut k),
                    "v" => poke(&mut v),
                    "g" => poke(&mut g),
                    "beta" => poke(&mut beta),
                    "s0" => poke(&mut s0),
                    _ => unreachable!(),
                }
                // A NaN only in the last head: every earlier head is finite.
                if name == "q" && bad.is_nan() {
                    q.fill(0.1);
                    let last = ((b * t - 1) * h + (h - 1)) * dk + (dk - 1);
                    q[last] = f32::NAN;
                }
                let budget = Budget::new(1 << 26);
                let backend = CpuBackend::with_threads(budget.clone(), 6).unwrap();
                let raw = Raw {
                    shape: Shape { b, t, h, dk, dv },
                    q,
                    k,
                    v,
                    g,
                    beta,
                    s0: Some(s0),
                };
                let case = load(&budget, &raw);
                let held = budget.live_bytes().unwrap();
                let got = catch_unwind(AssertUnwindSafe(|| {
                    backend.chunked_gdn_forward(inputs(&case))
                }));
                let got = got.unwrap_or_else(|_| panic!("panic H={h} {name} {bad:?}"));
                let err = got.expect_err(&format!("Ok H={h} {name} {bad:?}"));
                assert!(
                    matches!(err, OjasError::NonFinite { .. }),
                    "H={h} {name} {bad:?}: {err:?}"
                );
                assert_eq!(
                    budget.live_bytes().unwrap(),
                    held,
                    "leak H={h} {name} {bad:?}"
                );
            }
        }
    }
}

#[test]
fn finite_g_that_overflows_exp_is_nonfinite_not_a_panic() {
    for h in [1usize, 2] {
        let backend = cpu(4, Numerics::Fast);
        let s = Shape {
            b: 1,
            t: 2,
            h,
            dk: 2,
            dv: 2,
        };
        let mut raw = normal_raw(s, true);
        raw.g.fill(100.0);
        let case = load(backend.budget(), &raw);
        let held = backend.budget().live_bytes().unwrap();
        let got = catch_unwind(AssertUnwindSafe(|| {
            backend.chunked_gdn_forward(inputs(&case))
        }));
        let got = got.unwrap_or_else(|_| panic!("panic on overflowing exp H={h}"));
        let err = got.expect_err("overflowing exp returned Ok");
        assert!(matches!(err, OjasError::NonFinite { .. }), "{err:?}");
        assert_eq!(backend.budget().live_bytes().unwrap(), held);
        let _ = case;
    }
}

fn fail_on(n: usize) -> (Arc<AtomicUsize>, impl Fn() -> Result<(), OjasError>) {
    let hits = Arc::new(AtomicUsize::new(0));
    let hits2 = Arc::clone(&hits);
    let hook = move || {
        let i = hits2.fetch_add(1, Ordering::SeqCst) + 1;
        if i == n {
            Err(OjasError::Unsupported {
                op: "cancel",
                detail: format!("stop at {n}"),
            })
        } else {
            Ok(())
        }
    };
    (hits, hook)
}

#[test]
fn cancellation_restores_the_budget_and_a_later_call_matches() {
    let shape_fwd = [
        (4usize, 1usize, 3usize), // B, H, fail at call. H=1 direct write, mid-head.
        (1, 4, 3),                // H>1 scratch, mid-head.
    ];
    for (b, h, stop) in shape_fwd {
        for threads in [1usize, 6] {
            let backend = cpu(threads, Numerics::Fast);
            let clean = cpu(threads, Numerics::Fast);
            let raw = normal_raw(
                Shape {
                    b,
                    t: 5,
                    h,
                    dk: 4,
                    dv: 3,
                },
                true,
            );
            let case = load(backend.budget(), &raw);
            let reference = load(clean.budget(), &raw);
            let (hits, hook) = fail_on(stop);
            backend.set_cancel(hook);
            let held = backend.budget().live_bytes().unwrap();
            let err = backend
                .chunked_gdn_forward(inputs(&case))
                .expect_err("cancel returned Ok");
            assert!(
                matches!(err, OjasError::Unsupported { op: "cancel", .. }),
                "fwd B={b} H={h} threads {threads}: {err:?} hits {}",
                hits.load(Ordering::SeqCst)
            );
            assert_eq!(backend.budget().live_bytes().unwrap(), held);
            backend.set_cancel(|| Ok(()));
            let y = backend.chunked_gdn_forward(inputs(&case)).unwrap();
            let y2 = backend.chunked_gdn_forward(inputs(&case)).unwrap();
            let expect = clean.chunked_gdn_forward(inputs(&reference)).unwrap();
            assert!(y.output.to_f32_vec().unwrap().iter().all(|v| v.is_finite()));
            assert_eq!(
                bits(&y.output.to_f32_vec().unwrap()),
                bits(&expect.output.to_f32_vec().unwrap())
            );
            assert_eq!(
                bits(&y.final_state.to_f32_vec().unwrap()),
                bits(&expect.final_state.to_f32_vec().unwrap())
            );
            assert_eq!(
                bits(&y.checkpoints.to_f32_vec().unwrap()),
                bits(&expect.checkpoints.to_f32_vec().unwrap())
            );
            assert_eq!(
                bits(&y.output.to_f32_vec().unwrap()),
                bits(&y2.output.to_f32_vec().unwrap())
            );
        }
    }

    // Backward: one cancel per head, then one per token. T=8 => 9 calls on H=1.
    for h in [1usize, 3] {
        let backend = cpu(1, Numerics::Exact);
        let clean = cpu(1, Numerics::Exact);
        let raw = normal_raw(
            Shape {
                b: 1,
                t: 8,
                h,
                dk: 3,
                dv: 2,
            },
            true,
        );
        let case = load(backend.budget(), &raw);
        let reference = load(clean.budget(), &raw);
        let y = backend.chunked_gdn_forward(inputs(&case)).unwrap();
        let y_ref = clean.chunked_gdn_forward(inputs(&reference)).unwrap();
        let gy = Tensor::from_f32(
            &vec![0.02f32; y.output.num_elements().unwrap()],
            y.output.shape(),
            backend.budget(),
        )
        .unwrap();
        let gs = Tensor::from_f32(
            &vec![0.01f32; y.final_state.num_elements().unwrap()],
            y.final_state.shape(),
            backend.budget(),
        )
        .unwrap();
        let gy_ref = Tensor::from_f32(
            &vec![0.02f32; gy.num_elements().unwrap()],
            gy.shape(),
            clean.budget(),
        )
        .unwrap();
        let gs_ref = Tensor::from_f32(
            &vec![0.01f32; gs.num_elements().unwrap()],
            gs.shape(),
            clean.budget(),
        )
        .unwrap();
        for stop in [1usize, 5] {
            let (hits, hook) = fail_on(stop);
            backend.set_cancel(hook);
            let held = backend.budget().live_bytes().unwrap();
            let err = backend
                .chunked_gdn_backward(inputs(&case), &y.checkpoints, &gy, Some(&gs))
                .expect_err("backward cancel returned Ok");
            assert!(
                matches!(err, OjasError::Unsupported { op: "cancel", .. }),
                "bwd H={h} stop {stop}: {err:?} hits {}",
                hits.load(Ordering::SeqCst)
            );
            assert_eq!(backend.budget().live_bytes().unwrap(), held);
        }
        backend.set_cancel(|| Ok(()));
        let g = backend
            .chunked_gdn_backward(inputs(&case), &y.checkpoints, &gy, Some(&gs))
            .unwrap();
        let expect = clean
            .chunked_gdn_backward(
                inputs(&reference),
                &y_ref.checkpoints,
                &gy_ref,
                Some(&gs_ref),
            )
            .unwrap();
        assert_eq!(
            bits(&g.q.to_f32_vec().unwrap()),
            bits(&expect.q.to_f32_vec().unwrap())
        );
        assert!(g.q.to_f32_vec().unwrap().iter().all(|v| v.is_finite()));
    }
}

fn shape_err(run: impl FnOnce() -> Result<(), OjasError>) {
    let got = catch_unwind(AssertUnwindSafe(run));
    let got = got.expect("shape case panicked");
    let err = got.expect_err("shape case returned Ok");
    assert!(
        matches!(err, OjasError::Shape { .. } | OjasError::OutOfRange { .. }),
        "{err:?}"
    );
}

#[test]
fn empty_and_mismatched_shapes_do_not_panic() {
    let budget = Budget::new(1 << 20);
    let backend = CpuBackend::new(budget.clone());
    let q = Tensor::from_f32(&pattern(8, 0.1, 0.0), &[1, 2, 1, 4], &budget).unwrap();
    let k = Tensor::from_f32(&pattern(8, 0.1, 0.0), &[1, 2, 1, 4], &budget).unwrap();
    let v = Tensor::from_f32(&pattern(8, 0.1, 0.0), &[1, 2, 1, 4], &budget).unwrap();
    let g = Tensor::from_f32(&pattern(2, 0.01, 0.0), &[1, 2, 1], &budget).unwrap();
    let beta = Tensor::from_f32(&pattern(2, 0.01, 0.4), &[1, 2, 1], &budget).unwrap();

    let bad_k = Tensor::from_f32(&pattern(16, 0.1, 0.0), &[1, 2, 1, 8], &budget).unwrap();
    shape_err(|| {
        backend
            .chunked_gdn_forward(GdnInputs {
                q: &q,
                k: &bad_k,
                v: &v,
                g: &g,
                beta: &beta,
                initial_state: None,
            })
            .map(|_| ())
    });

    let rank3 = Tensor::from_f32(&pattern(8, 0.1, 0.0), &[2, 1, 4], &budget).unwrap();
    shape_err(|| {
        backend
            .chunked_gdn_forward(GdnInputs {
                q: &rank3,
                k: &k,
                v: &v,
                g: &g,
                beta: &beta,
                initial_state: None,
            })
            .map(|_| ())
    });

    let zeros = [
        Tensor::from_f32(&[], &[1, 0, 1, 4], &budget).unwrap(),
        Tensor::from_f32(&[], &[1, 2, 0, 4], &budget).unwrap(),
        Tensor::from_f32(&[], &[1, 2, 1, 0], &budget).unwrap(),
    ];
    // T=0, H=0, Dk=0 on q.
    for zq in &zeros {
        let got = catch_unwind(AssertUnwindSafe(|| {
            backend.chunked_gdn_forward(GdnInputs {
                q: zq,
                k: &k,
                v: &v,
                g: &g,
                beta: &beta,
                initial_state: None,
            })
        }));
        let err = got
            .expect("zero extent panicked")
            .expect_err("zero extent Ok");
        assert!(
            matches!(err, OjasError::Shape { .. } | OjasError::OutOfRange { .. }),
            "{err:?}"
        );
    }
    let zv = Tensor::from_f32(&[], &[1, 2, 1, 0], &budget).unwrap();
    let got = catch_unwind(AssertUnwindSafe(|| {
        backend.chunked_gdn_forward(GdnInputs {
            q: &q,
            k: &k,
            v: &zv,
            g: &g,
            beta: &beta,
            initial_state: None,
        })
    }));
    let err = got.expect("Dv=0 panicked").expect_err("Dv=0 Ok");
    assert!(
        matches!(err, OjasError::Shape { .. } | OjasError::OutOfRange { .. }),
        "{err:?}"
    );

    let len_err = catch_unwind(AssertUnwindSafe(|| {
        Tensor::from_f32(&[1.0, 2.0], &[1, 1, 1, 4], &budget)
    }));
    let err = len_err
        .expect("from_f32 length panicked")
        .expect_err("length mismatch Ok");
    assert!(
        matches!(err, OjasError::Shape { .. } | OjasError::OutOfRange { .. }),
        "{err:?}"
    );
}

fn gdn_charge(
    b: usize,
    t: usize,
    h: usize,
    dk: usize,
    dv: usize,
    with_state: bool,
    threads: usize,
) -> u64 {
    let rows = b * t * h;
    let per = dk * dv;
    let nc = t.div_ceil(GDN_CHECKPOINT_TOKENS);
    let mut inputs = rows * (2 * dk + dv + 2);
    if with_state {
        inputs += b * h * per;
    }
    let outputs = rows * dv + b * h * per + b * h * nc * per;
    let heads = b * h;
    let work = heads
        .saturating_mul(t)
        .saturating_mul(per)
        .saturating_mul(3);
    let inflight = if work >= 2 * (1 << 20) {
        heads.min(threads).max(1)
    } else {
        1
    };
    let head_major = if h == 1 { 0 } else { rows * dv };
    let scratch = rows * 2 * dk + head_major + inflight * (per + 2 * dv);
    let old_scratch = rows * (2 * dk + 2 + dv) + inflight * (per + 2 * dv);
    assert!(
        old_scratch > scratch,
        "new charge is not tighter than the old one"
    );
    ((inputs + outputs + scratch) * 4) as u64
}

#[test]
fn forward_cap_is_the_new_scratch_one_byte_under_refuses() {
    // The committed serial numbers, plus the old (looser) charge, which must
    // still succeed, and a parallel shape whose inflight matches the threads.
    let cases = [
        (
            1usize, 2usize, 1usize, 4usize, 4usize, false, 1usize, 432u64,
        ),
        (1, 2, 2, 4, 4, false, 1, 832),
        (1, 2, 1, 4, 4, true, 1, 0),
        (8, 128, 1, 32, 32, false, 8, 0),
        (1, 128, 8, 32, 32, false, 8, 0),
        (1, 128, 8, 32, 32, true, 6, 0),
    ];
    for (b, t, h, dk, dv, with_state, threads, pinned) in cases {
        let cap = gdn_charge(b, t, h, dk, dv, with_state, threads);
        if pinned != 0 {
            assert_eq!(cap, pinned, "formula drifted for H={h} state {with_state}");
        }
        for (use_cap, expect_ok) in [(cap, true), (cap - 1, false)] {
            let budget = Budget::new(use_cap);
            let backend = if threads == 1 {
                CpuBackend::new(budget.clone())
            } else {
                CpuBackend::with_threads(budget.clone(), threads).unwrap()
            };
            let raw = normal_raw(Shape { b, t, h, dk, dv }, with_state);
            let built = catch_unwind(AssertUnwindSafe(|| load(&budget, &raw)));
            let case = built.unwrap_or_else(|_| panic!("load panicked cap {use_cap}"));
            let held = budget.live_bytes().unwrap();
            let got = catch_unwind(AssertUnwindSafe(|| {
                backend.chunked_gdn_forward(inputs(&case))
            }));
            let got = got.unwrap_or_else(|_| panic!("forward panicked cap {use_cap} H={h}"));
            if expect_ok {
                let y = got.expect("tighter cap refused a charge that matches the allocation");
                assert!(y.output.to_f32_vec().unwrap().iter().all(|v| v.is_finite()));
                let outs = (y.output.num_elements().unwrap()
                    + y.final_state.num_elements().unwrap()
                    + y.checkpoints.num_elements().unwrap()) as u64
                    * 4;
                assert_eq!(budget.live_bytes().unwrap(), held + outs);
            } else {
                let err = got.expect_err("one byte under the peak succeeded");
                assert!(matches!(err, OjasError::CapacityExceeded { .. }), "{err:?}");
                assert_eq!(budget.live_bytes().unwrap(), held);
                let again = backend.chunked_gdn_forward(inputs(&case));
                assert!(matches!(again, Err(OjasError::CapacityExceeded { .. })));
                assert_eq!(budget.live_bytes().unwrap(), held);
            }
        }
    }
}

fn mul_vec(n: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            if i == 0 {
                -0.0
            } else if i == n / 2 {
                f32::from_bits(1)
            } else {
                ((i % 13) as f32 - 6.0) * scale
            }
        })
        .collect()
}

#[test]
fn mul_forward_bits_match_across_the_piece_band_and_nan_does_not_leak() {
    let ns = [
        0usize,
        1,
        1024 * 1024 - 1,
        1024 * 1024,
        1024 * 2048,
        1024 * 2048 + 1,
        1024 * 4096,
    ];
    for n in ns {
        if n == 0 {
            let budget = Budget::new(1 << 20);
            let backend = CpuBackend::with_threads(budget.clone(), 6).unwrap();
            let got = catch_unwind(AssertUnwindSafe(|| {
                let a = Tensor::from_f32(&[], &[0], &budget)?;
                let b = Tensor::from_f32(&[], &[0], &budget)?;
                backend.mul_forward(&a, &b).map(|_| ())
            }));
            let err = got.expect("n=0 panicked").expect_err("n=0 Ok");
            assert!(
                matches!(err, OjasError::Shape { .. } | OjasError::OutOfRange { .. }),
                "{err:?}"
            );
            continue;
        }
        let mut reference: Option<Vec<u32>> = None;
        for threads in [1usize, 4, 6, 7, 8] {
            let backend = cpu(
                threads,
                if threads == 8 {
                    Numerics::Exact
                } else {
                    Numerics::Fast
                },
            );
            let a = Tensor::from_f32(&mul_vec(n, 0.1), &[n], backend.budget()).unwrap();
            let b = Tensor::from_f32(&mul_vec(n, -0.2), &[n], backend.budget()).unwrap();
            let y = catch_unwind(AssertUnwindSafe(|| backend.mul_forward(&a, &b)));
            let y = y
                .unwrap_or_else(|_| panic!("mul panicked n={n} threads {threads}"))
                .unwrap_or_else(|err| panic!("mul refused n={n} threads {threads}: {err:?}"));
            let got = bits(&y.to_f32_vec().unwrap());
            assert!(
                got.iter().all(|w| w & 0x7fff_ffff < 0x7f80_0000),
                "non-finite product"
            );
            // -0 * (negative pattern at i=0: scale -0.2 => bias of pattern).
            // Element 0 of `a` is -0 and element 0 of `b` is -0, product +0.
            assert_eq!(got[0], 0.0f32.to_bits(), "n={n} threads {threads}");
            if let Some(reference) = &reference {
                assert_eq!(got, *reference, "mul bits n={n} threads {threads}");
            } else {
                reference = Some(got);
            }
        }
        for bad_at in [0usize, n / 2, n - 1] {
            for threads in [1usize, 6, 8] {
                for bad in [f32::NAN, f32::INFINITY] {
                    let budget = Budget::new(1 << 30);
                    let backend = CpuBackend::with_threads(budget.clone(), threads).unwrap();
                    let mut av = mul_vec(n, 0.1);
                    av[bad_at] = bad;
                    let a = Tensor::from_f32(&av, &[n], &budget).unwrap();
                    let b = Tensor::from_f32(&mul_vec(n, 0.2), &[n], &budget).unwrap();
                    let held = budget.live_bytes().unwrap();
                    let err = backend.mul_forward(&a, &b).expect_err("NaN mul Ok");
                    assert!(matches!(err, OjasError::NonFinite { .. }), "n={n} {err:?}");
                    assert_eq!(budget.live_bytes().unwrap(), held, "mul leaked n={n}");
                }
            }
        }
    }
}

fn rms_pair(threads: usize, rows: usize, dim: usize) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let backend = cpu(threads, Numerics::Fast);
    let mut x = pattern(rows * dim, 0.05, 0.2);
    if !x.is_empty() {
        let last = x.len() - 1;
        x[0] = -0.0;
        x[last] = f32::from_bits(1);
    }
    let mut w = pattern(dim, 0.01, 1.0);
    w[0] = 0.5;
    let input = Tensor::from_f32(&x, &[rows, dim], backend.budget()).unwrap();
    let weight = Tensor::from_f32(&w, &[dim], backend.budget()).unwrap();
    let y = backend
        .rms_norm_forward(&input, &weight, RMS_NORM_EPS)
        .unwrap();
    let gy = Tensor::from_f32(
        &pattern(rows * dim, 0.02, 0.1),
        &[rows, dim],
        backend.budget(),
    )
    .unwrap();
    let (gx, gw) = backend
        .rms_norm_backward(&input, &weight, &gy, RMS_NORM_EPS)
        .unwrap();
    (
        bits(&y.to_f32_vec().unwrap()),
        bits(&gx.to_f32_vec().unwrap()),
        bits(&gw.to_f32_vec().unwrap()),
    )
}

#[test]
fn rms_bits_match_under_and_over_the_two_mib_cut() {
    // 512 * 1024 f32 = exactly 2 MiB. 513 rows is just over. 768 * 1024 = 3 MiB.
    // dim = 1 is the other edge of the cut (one value per row).
    let shapes = [
        (512usize, 1024usize),
        (513, 1024),
        (768, 1024),
        (524_288, 1),
        (524_289, 1),
    ];
    for (rows, dim) in shapes {
        let one = rms_pair(1, rows, dim);
        let six = rms_pair(6, rows, dim);
        assert_eq!(one.0, six.0, "rms forward {rows}x{dim}");
        assert_eq!(one.1, six.1, "rms grad_x {rows}x{dim}");
        assert_eq!(one.2, six.2, "rms grad_w {rows}x{dim}");
        assert!(one.0.iter().all(|w| w & 0x7fff_ffff < 0x7f80_0000));
    }
}

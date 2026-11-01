//! `Backend::attention_forward`, `attention_forward_saving` and
//! `attention_backward`: bidirectional attention, a query length that
//! differs from the key length, grouped-query heads, key lengths, an
//! additive bias (with its gradient) and the causal / window masks, against
//! an f64 oracle.
//!
//! Also: the causal case is `causal_sdpa_*` bit for bit, results do not
//! depend on the thread count, a query that sees no key is zeros, and every
//! malformed call is refused.

use std::ops::Range;

use ojas_core::{AttentionOpts, Backend, Budget, Numerics, OjasError, Tensor};
use ojas_cpu::CpuBackend;

fn backend(threads: usize, numerics: Numerics) -> CpuBackend {
    CpuBackend::with_threads(Budget::new(1 << 40), threads)
        .unwrap()
        .with_numerics(numerics)
}

fn tensor(values: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_f32(values, shape, &Budget::new(u64::MAX)).unwrap()
}

fn lens_tensor(values: &[u32]) -> Tensor {
    Tensor::from_u32(values, &[values.len()], &Budget::new(u64::MAX)).unwrap()
}

/// Deterministic values in `[-amp, amp)`.
fn values(n: usize, seed: u64, amp: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * amp
        })
        .collect()
}

/// One attention call: shapes and options.
#[derive(Clone, Debug)]
struct Case {
    b: usize,
    h: usize,
    hkv: usize,
    tq: usize,
    tk: usize,
    d: usize,
    causal: bool,
    window: Option<usize>,
    /// `[Bb, Hb, Tqb]`; the last axis is `Tk`.
    bias: Option<[usize; 3]>,
    lens: Option<Vec<u32>>,
    /// `AttentionOpts::scale`; `None` is `1 / sqrt(D)`.
    scale: Option<f32>,
    /// `AttentionOpts::radius`: key `j` is visible to query `i` when
    /// `|i - j| <= radius`.
    radius: Option<usize>,
}

impl Case {
    fn new(b: usize, h: usize, hkv: usize, tq: usize, tk: usize, d: usize) -> Self {
        Case {
            b,
            h,
            hkv,
            tq,
            tk,
            d,
            causal: false,
            window: None,
            bias: None,
            lens: None,
            scale: None,
            radius: None,
        }
    }
    fn causal(mut self, window: Option<usize>) -> Self {
        self.causal = true;
        self.window = window;
        self
    }
    fn scale(mut self, scale: f32) -> Self {
        self.scale = Some(scale);
        self
    }
    fn radius(mut self, radius: usize) -> Self {
        self.radius = Some(radius);
        self
    }
    fn bias(mut self, shape: [usize; 3]) -> Self {
        self.bias = Some(shape);
        self
    }
    fn lens(mut self, lens: &[u32]) -> Self {
        self.lens = Some(lens.to_vec());
        self
    }
    fn q_shape(&self) -> [usize; 4] {
        [self.b, self.h, self.tq, self.d]
    }
    fn kv_shape(&self) -> [usize; 4] {
        [self.b, self.hkv, self.tk, self.d]
    }
    fn bias_shape(&self) -> Option<[usize; 4]> {
        self.bias.map(|[b, h, q]| [b, h, q, self.tk])
    }
    /// Key `j` is visible to query `i` of batch row `b`. With `Tq == Tk`, a
    /// query at or past the key length is a padding row and sees nothing.
    fn visible(&self, b: usize, i: usize, j: usize) -> bool {
        let len = self
            .lens
            .as_ref()
            .map_or(self.tk, |l| (l[b] as usize).min(self.tk));
        if self.tq == self.tk && i >= len {
            return false;
        }
        if j >= len {
            return false;
        }
        if self.radius.is_some_and(|r| i.abs_diff(j) > r) {
            return false;
        }
        if self.causal {
            let pos = (i + self.tk) as isize - self.tq as isize;
            if (j as isize) > pos {
                return false;
            }
            if let Some(w) = self.window {
                if (j + w) as isize <= pos {
                    return false;
                }
            }
        }
        true
    }
}

struct Inputs {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    gy: Vec<f32>,
    bias: Option<Vec<f32>>,
}

fn inputs(c: &Case, seed: u64) -> Inputs {
    let n = |s: &[usize]| s.iter().product::<usize>();
    Inputs {
        q: values(n(&c.q_shape()), seed, 1.0),
        k: values(n(&c.kv_shape()), seed + 1, 1.0),
        v: values(n(&c.kv_shape()), seed + 2, 1.0),
        gy: values(n(&c.q_shape()), seed + 3, 1.0),
        bias: c.bias_shape().map(|s| values(n(&s), seed + 4, 2.0)),
    }
}

/// `[y, lse, gq, gk, gv, gbias]` in f64.
type Exact = [Vec<f64>; 6];

fn oracle(c: &Case, x: &Inputs) -> Exact {
    let (b_n, h, hkv, tq, tk, d) = (c.b, c.h, c.hkv, c.tq, c.tk, c.d);
    let rep = h / hkv;
    let scale = c.scale.map_or(1.0 / (d as f64).sqrt(), f64::from);
    let q = |plane: usize, i: usize, c: usize| f64::from(x.q[(plane * tq + i) * d + c]);
    let gy = |plane: usize, i: usize, c: usize| f64::from(x.gy[(plane * tq + i) * d + c]);
    let k = |plane: usize, j: usize, c: usize| f64::from(x.k[(plane * tk + j) * d + c]);
    let v = |plane: usize, j: usize, c: usize| f64::from(x.v[(plane * tk + j) * d + c]);
    let mut out = [
        vec![0.0; b_n * h * tq * d],
        vec![0.0; b_n * h * tq],
        vec![0.0; b_n * h * tq * d],
        vec![0.0; b_n * hkv * tk * d],
        vec![0.0; b_n * hkv * tk * d],
        vec![0.0; c.bias_shape().map_or(0, |s| s.iter().product())],
    ];
    for plane in 0..b_n * h {
        let (b, head) = (plane / h, plane % h);
        let kvp = b * hkv + head / rep;
        for i in 0..tq {
            let keys: Vec<usize> = (0..tk).filter(|&j| c.visible(b, i, j)).collect();
            if keys.is_empty() {
                continue;
            }
            let bias_at = |j: usize| -> Option<usize> {
                let [bb, bh, bq] = c.bias?;
                let pick = |extent: usize, at: usize| if extent == 1 { 0 } else { at };
                Some(((pick(bb, b) * bh + pick(bh, head)) * bq + pick(bq, i)) * tk + j)
            };
            let s: Vec<f64> = keys
                .iter()
                .map(|&j| {
                    let dot: f64 = (0..d).map(|c| q(plane, i, c) * k(kvp, j, c)).sum();
                    dot * scale
                        + bias_at(j).map_or(0.0, |at| f64::from(x.bias.as_ref().unwrap()[at]))
                })
                .collect();
            let max = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let z: f64 = s.iter().map(|v| (v - max).exp()).sum();
            out[1][plane * tq + i] = max + z.ln();
            let p: Vec<f64> = s.iter().map(|v| (v - max).exp() / z).collect();
            let dp: Vec<f64> = keys
                .iter()
                .map(|&j| (0..d).map(|c| gy(plane, i, c) * v(kvp, j, c)).sum())
                .collect();
            let delta: f64 = p.iter().zip(&dp).map(|(a, b)| a * b).sum();
            for (n, &j) in keys.iter().enumerate() {
                let ds = p[n] * (dp[n] - delta);
                if let Some(at) = bias_at(j) {
                    out[5][at] += ds;
                }
                for c in 0..d {
                    out[0][(plane * tq + i) * d + c] += p[n] * v(kvp, j, c);
                    out[2][(plane * tq + i) * d + c] += ds * scale * k(kvp, j, c);
                    out[3][(kvp * tk + j) * d + c] += ds * scale * q(plane, i, c);
                    out[4][(kvp * tk + j) * d + c] += p[n] * gy(plane, i, c);
                }
            }
        }
    }
    out
}

/// `max |got - want| / max(1, max |want|)`.
fn err(got: &Tensor, want: &[f64]) -> f64 {
    let got = got.to_f32_vec().unwrap();
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(1.0f64, |m, v| m.max(v.abs()));
    got.iter()
        .zip(want)
        .fold(0.0f64, |m, (&a, &b)| m.max((f64::from(a) - b).abs()))
        / scale
}

fn bits(t: &Tensor) -> Vec<u32> {
    t.to_f32_vec()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// Every tensor of a case, built once.
struct Built {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    gy: Tensor,
    bias: Option<Tensor>,
    lens: Option<Tensor>,
}

impl Built {
    fn new(c: &Case, x: &Inputs) -> Self {
        Built {
            q: tensor(&x.q, &c.q_shape()),
            k: tensor(&x.k, &c.kv_shape()),
            v: tensor(&x.v, &c.kv_shape()),
            gy: tensor(&x.gy, &c.q_shape()),
            bias: x.bias.as_ref().map(|b| tensor(b, &c.bias_shape().unwrap())),
            lens: c.lens.as_ref().map(|l| lens_tensor(l)),
        }
    }
    fn opts<'a>(&'a self, c: &Case) -> AttentionOpts<'a> {
        AttentionOpts {
            causal: c.causal,
            window: c.window,
            bias: self.bias.as_ref(),
            key_lens: self.lens.as_ref(),
            scale: c.scale,
            radius: c.radius,
        }
    }
}

/// Forward, saving forward and backward of `c` on `be`: bits of
/// `[y, lse, gq, gk, gv, gbias]`, after each is checked against the oracle.
fn check(be: &CpuBackend, c: &Case, seed: u64, tol: f64) -> Vec<Vec<u32>> {
    let x = inputs(c, seed);
    let want = oracle(c, &x);
    let t = Built::new(c, &x);
    let opts = t.opts(c);
    let y = be.attention_forward(&t.q, &t.k, &t.v, opts).unwrap();
    let (y2, lse) = be.attention_forward_saving(&t.q, &t.k, &t.v, opts).unwrap();
    assert_eq!(bits(&y), bits(&y2), "{c:?}: forward forms differ");
    assert_eq!(y.shape(), t.q.shape());
    assert_eq!(lse.shape(), [c.b, c.h, c.tq]);
    let e = err(&y, &want[0]);
    assert!(e <= tol, "{c:?}: output error {e}");
    let e = err(&lse, &want[1]);
    assert!(e <= tol, "{c:?}: lse error {e}");
    let g = be
        .attention_backward(&t.q, &t.k, &t.v, &y, &lse, &t.gy, opts, c.bias.is_some())
        .unwrap();
    for (name, got, want) in [
        ("gq", &g.q, &want[2]),
        ("gk", &g.k, &want[3]),
        ("gv", &g.v, &want[4]),
    ] {
        let e = err(got, want);
        assert!(e <= tol, "{c:?}: {name} error {e}");
    }
    let mut all = vec![bits(&y), bits(&lse), bits(&g.q), bits(&g.k), bits(&g.v)];
    match (&g.bias, c.bias) {
        (Some(gb), Some(_)) => {
            assert_eq!(gb.shape(), t.bias.as_ref().unwrap().shape());
            let e = err(gb, &want[5]);
            assert!(e <= tol, "{c:?}: bias gradient error {e}");
            all.push(bits(gb));
        }
        (None, None) => {}
        (got, want) => panic!("{c:?}: bias gradient {} but bias {want:?}", got.is_some()),
    }
    // Without the ask, no bias gradient and the same other gradients.
    let g2 = be
        .attention_backward(&t.q, &t.k, &t.v, &y, &lse, &t.gy, opts, false)
        .unwrap();
    assert!(g2.bias.is_none());
    assert_eq!(bits(&g2.q), bits(&g.q));
    assert_eq!(bits(&g2.k), bits(&g.k));
    assert_eq!(bits(&g2.v), bits(&g.v));
    all
}

fn cases() -> Vec<Case> {
    vec![
        // Bidirectional self-attention.
        Case::new(2, 3, 3, 9, 9, 8),
        // Cross-attention, Tq != Tk both ways.
        Case::new(2, 2, 2, 5, 11, 8),
        Case::new(1, 2, 2, 12, 4, 16),
        // Grouped-query heads.
        Case::new(2, 4, 2, 7, 10, 8),
        // Head dim not a multiple of anything.
        Case::new(1, 2, 1, 6, 17, 5),
        // Causal, square.
        Case::new(1, 2, 2, 10, 10, 8).causal(None),
        Case::new(1, 2, 2, 10, 10, 8).causal(Some(3)),
        // Causal, queries are the last positions (prefill onto a cache).
        Case::new(1, 2, 2, 4, 9, 8).causal(None),
        Case::new(1, 2, 2, 4, 9, 8).causal(Some(2)),
        // Causal, more queries than keys: the first ones see nothing.
        Case::new(1, 2, 2, 8, 5, 8).causal(None),
        // Key lengths: padded prompts, an empty one, one over Tk.
        Case::new(3, 2, 2, 6, 12, 8).lens(&[12, 5, 1]),
        Case::new(3, 2, 2, 6, 12, 8).lens(&[0, 7, 99]),
        Case::new(2, 2, 1, 5, 9, 8).causal(None).lens(&[4, 9]),
        // Bias: full, and each broadcast axis alone and together.
        Case::new(2, 3, 3, 5, 7, 8).bias([2, 3, 5]),
        Case::new(2, 3, 3, 5, 7, 8).bias([1, 3, 5]),
        Case::new(2, 3, 3, 5, 7, 8).bias([2, 1, 5]),
        Case::new(2, 3, 3, 5, 7, 8).bias([2, 3, 1]),
        Case::new(2, 3, 3, 5, 7, 8).bias([1, 1, 1]),
        Case::new(2, 4, 2, 6, 7, 8).bias([1, 4, 6]),
        // Bias with the masks.
        Case::new(2, 2, 2, 6, 10, 8)
            .causal(Some(4))
            .bias([2, 2, 6])
            .lens(&[10, 6]),
        Case::new(3, 2, 2, 4, 9, 8).bias([1, 1, 4]).lens(&[2, 0, 9]),
        // Padding rows: `Tq == Tk` with a length below `Tq` (queries at or
        // past it are zeros), alone, empty, one over, with the masks, a bias
        // and grouped heads.
        Case::new(2, 3, 3, 9, 9, 8).lens(&[9, 4]),
        Case::new(2, 2, 2, 7, 7, 8).lens(&[0, 99]),
        Case::new(3, 4, 2, 8, 8, 8).lens(&[8, 1, 5]),
        Case::new(2, 2, 2, 8, 8, 8).causal(None).lens(&[8, 3]),
        Case::new(2, 2, 2, 9, 9, 8).causal(Some(3)).lens(&[9, 5]),
        Case::new(2, 2, 2, 7, 7, 8).bias([2, 2, 7]).lens(&[7, 2]),
        Case::new(2, 2, 2, 7, 7, 8).bias([1, 1, 1]).lens(&[3, 7]),
        // A symmetric band: alone, one wide, grouped heads, with a bias, one
        // that hides nothing, and a single position. Then the band with
        // padding rows (the EmbeddingGemma 2 encoder's call: scale 1, grouped
        // heads, a band, a ragged batch).
        Case::new(2, 3, 3, 9, 9, 8).radius(2),
        Case::new(1, 2, 2, 10, 10, 8).radius(1),
        Case::new(2, 4, 2, 12, 12, 8).radius(3),
        Case::new(2, 2, 2, 11, 11, 16).radius(4).bias([2, 2, 11]),
        Case::new(1, 2, 2, 7, 7, 8).radius(6),
        Case::new(1, 1, 1, 1, 1, 8).radius(1),
        Case::new(2, 2, 2, 10, 10, 8).radius(2).lens(&[10, 4]),
        Case::new(2, 4, 2, 12, 12, 16)
            .radius(3)
            .scale(1.0)
            .lens(&[12, 5]),
        Case::new(2, 4, 1, 11, 11, 32)
            .radius(4)
            .scale(1.0)
            .bias([2, 4, 11])
            .lens(&[11, 0]),
        // An explicit score scale: the usual, larger, smaller, negative, zero.
        Case::new(2, 3, 3, 9, 9, 8).scale(1.0),
        Case::new(2, 3, 3, 9, 9, 8).scale(0.3),
        Case::new(2, 3, 3, 9, 9, 8).scale(-0.7),
        Case::new(2, 3, 3, 9, 9, 8).scale(0.0),
        Case::new(2, 4, 2, 7, 10, 8)
            .scale(1.0)
            .bias([1, 4, 7])
            .lens(&[10, 4]),
        Case::new(1, 2, 2, 10, 10, 8).causal(Some(3)).scale(2.0),
        Case::new(1, 2, 2, 6, 6, 8).radius(2).scale(1.0),
    ]
}

#[test]
fn matches_the_f64_oracle_for_every_mask_and_bias_layout() {
    for numerics in [Numerics::Exact, Numerics::Fast] {
        let be = backend(1, numerics);
        for (n, c) in cases().iter().enumerate() {
            check(&be, c, 100 + n as u64, 2e-5);
        }
    }
}

#[test]
fn results_do_not_depend_on_the_thread_count() {
    // Big enough that the passes split into tasks.
    // Work above two pool tasks' worth (`TASK_WORK`), so the passes run on
    // scoped threads; few planes, so each plane's rows split into blocks.
    let mut big = vec![
        Case::new(2, 4, 2, 140, 180, 32)
            .bias([1, 4, 140])
            .lens(&[180, 41]),
        Case::new(1, 3, 3, 128, 128, 64)
            .causal(Some(20))
            .bias([1, 1, 128]),
        Case::new(2, 2, 2, 66, 300, 16).bias([2, 2, 66]),
        Case::new(1, 2, 2, 300, 300, 32)
            .causal(None)
            .bias([1, 2, 300]),
        Case::new(1, 2, 2, 260, 340, 32).lens(&[200]),
        Case::new(1, 3, 3, 200, 200, 32).radius(40),
        Case::new(2, 2, 2, 130, 130, 16).radius(9).scale(1.0),
    ];
    big.extend(cases().into_iter().take(6));
    for (n, c) in big.iter().enumerate() {
        let one = check(&backend(1, Numerics::Exact), c, 300 + n as u64, 5e-5);
        for threads in [2, 5] {
            let many = check(&backend(threads, Numerics::Exact), c, 300 + n as u64, 5e-5);
            assert_eq!(one, many, "{c:?}: {threads} threads changed the bits");
        }
        // Fast shares the row kernels: same bits as Exact.
        let fast = check(&backend(3, Numerics::Fast), c, 300 + n as u64, 5e-5);
        assert_eq!(one, fast, "{c:?}: Fast changed the bits");
    }
}

#[test]
fn causal_attention_is_causal_sdpa_bit_for_bit() {
    for window in [None, Some(1), Some(5), Some(100)] {
        for (h, hkv) in [(2, 2), (4, 2)] {
            let c = Case::new(2, h, hkv, 13, 13, 8).causal(window);
            let x = inputs(&c, 7);
            let t = Built::new(&c, &x);
            let be = backend(2, Numerics::Exact);
            let (y, lse) = be
                .attention_forward_saving(&t.q, &t.k, &t.v, t.opts(&c))
                .unwrap();
            let (ry, rlse) = be.causal_sdpa_forward(&t.q, &t.k, &t.v, window).unwrap();
            assert_eq!(bits(&y), bits(&ry), "{window:?} {h}/{hkv}: output");
            assert_eq!(bits(&lse), bits(&rlse), "{window:?} {h}/{hkv}: lse");
            let g = be
                .attention_backward(&t.q, &t.k, &t.v, &y, &lse, &t.gy, t.opts(&c), false)
                .unwrap();
            let (gq, gk, gv) = be
                .causal_sdpa_backward(&t.q, &t.k, &t.v, &ry, &rlse, &t.gy, window)
                .unwrap();
            assert_eq!(bits(&g.q), bits(&gq), "{window:?} {h}/{hkv}: gq");
            assert_eq!(bits(&g.k), bits(&gk), "{window:?} {h}/{hkv}: gk");
            assert_eq!(bits(&g.v), bits(&gv), "{window:?} {h}/{hkv}: gv");
        }
    }
}

#[test]
fn a_query_that_sees_no_key_is_zeros_with_no_gradient() {
    let be = backend(1, Numerics::Exact);
    // Batch row 1 has no keys; causal Tq > Tk hides the first 3 queries.
    for c in [
        Case::new(2, 2, 2, 4, 6, 8).lens(&[6, 0]),
        Case::new(1, 2, 2, 8, 5, 8).causal(None),
        // Padding rows: batch row 1 keeps two of its six queries.
        Case::new(2, 2, 2, 6, 6, 8).lens(&[6, 2]),
    ] {
        let x = inputs(&c, 11);
        let t = Built::new(&c, &x);
        let (y, lse) = be
            .attention_forward_saving(&t.q, &t.k, &t.v, t.opts(&c))
            .unwrap();
        let g = be
            .attention_backward(&t.q, &t.k, &t.v, &y, &lse, &t.gy, t.opts(&c), false)
            .unwrap();
        let (y, lse, gq) = (
            y.to_f32_vec().unwrap(),
            lse.to_f32_vec().unwrap(),
            g.q.to_f32_vec().unwrap(),
        );
        let (b, h, tq, d) = (c.b, c.h, c.tq, c.d);
        let mut hidden = 0;
        for plane in 0..b * h {
            for i in 0..tq {
                if (0..c.tk).any(|j| c.visible(plane / h, i, j)) {
                    continue;
                }
                hidden += 1;
                let row = (plane * tq + i) * d;
                assert_eq!(lse[plane * tq + i], 0.0);
                assert!(y[row..row + d].iter().all(|&v| v == 0.0));
                assert!(gq[row..row + d].iter().all(|&v| v == 0.0));
            }
        }
        assert!(hidden > 0, "{c:?}: no query was hidden");
    }
    // A batch row with no keys gets no K or V gradient either.
    let c = Case::new(2, 2, 2, 4, 6, 8).lens(&[6, 0]);
    let x = inputs(&c, 12);
    let t = Built::new(&c, &x);
    let (y, lse) = be
        .attention_forward_saving(&t.q, &t.k, &t.v, t.opts(&c))
        .unwrap();
    let g = be
        .attention_backward(&t.q, &t.k, &t.v, &y, &lse, &t.gy, t.opts(&c), false)
        .unwrap();
    let per_batch = c.hkv * c.tk * c.d;
    for grad in [&g.k, &g.v] {
        let grad = grad.to_f32_vec().unwrap();
        assert!(grad[per_batch..].iter().all(|&v| v == 0.0));
        assert!(grad[..per_batch].iter().any(|&v| v != 0.0));
    }
}

#[test]
fn a_window_over_every_key_is_no_window() {
    let be = backend(1, Numerics::Exact);
    let base = Case::new(1, 2, 2, 4, 9, 8).causal(None);
    let wide = Case::new(1, 2, 2, 4, 9, 8).causal(Some(9));
    let x = inputs(&base, 5);
    let (tb, tw) = (Built::new(&base, &x), Built::new(&wide, &x));
    let a = be
        .attention_forward(&tb.q, &tb.k, &tb.v, tb.opts(&base))
        .unwrap();
    let b = be
        .attention_forward(&tw.q, &tw.k, &tw.v, tw.opts(&wide))
        .unwrap();
    assert_eq!(bits(&a), bits(&b));
}

/// Rows `rows` of every `[plane, T, D]` plane of `data`.
fn plane_rows(data: &[f32], planes: usize, t: usize, d: usize, rows: Range<usize>) -> Vec<u32> {
    (0..planes)
        .flat_map(|p| data[(p * t + rows.start) * d..(p * t + rows.end) * d].iter())
        .map(|v| v.to_bits())
        .collect()
}

#[test]
fn padding_rows_are_inert_and_cross_attention_is_not_padded() {
    // Self-attention, batch row 1 has 3 of 8 rows. Whatever queries (and
    // their output gradients) sit in rows 3.. cannot change one bit of the
    // live rows, the key and value gradients or the bias gradient, and the
    // padding rows themselves are zeros with a log-sum-exp of 0.
    let be = backend(2, Numerics::Exact);
    for c in [
        Case::new(2, 4, 2, 8, 8, 16).lens(&[8, 3]),
        Case::new(2, 4, 2, 8, 8, 16)
            .radius(2)
            .scale(1.0)
            .lens(&[8, 3]),
        Case::new(2, 4, 2, 8, 8, 16).causal(None).lens(&[8, 3]),
        Case::new(2, 4, 2, 8, 8, 16).bias([2, 4, 8]).lens(&[8, 3]),
    ] {
        let x = inputs(&c, 61);
        let mut moved = Inputs {
            q: x.q.clone(),
            k: x.k.clone(),
            v: x.v.clone(),
            gy: x.gy.clone(),
            bias: x.bias.clone(),
        };
        // Batch row 1 is planes `h..2h` of q and gy: rewrite rows 3.. there.
        for plane in c.h..2 * c.h {
            for i in 3..c.tq {
                for col in 0..c.d {
                    let at = (plane * c.tq + i) * c.d + col;
                    moved.q[at] = 5.0 - moved.q[at];
                    moved.gy[at] = -2.0 * moved.gy[at] + 1.0;
                }
            }
        }
        let run = |x: &Inputs| {
            let t = Built::new(&c, x);
            let (y, lse) = be
                .attention_forward_saving(&t.q, &t.k, &t.v, t.opts(&c))
                .unwrap();
            let g = be
                .attention_backward(
                    &t.q,
                    &t.k,
                    &t.v,
                    &y,
                    &lse,
                    &t.gy,
                    t.opts(&c),
                    c.bias.is_some(),
                )
                .unwrap();
            (y, lse, g)
        };
        let (y0, lse0, g0) = run(&x);
        let (y1, lse1, g1) = run(&moved);
        for (name, a, b) in [
            ("y", &y0, &y1),
            ("lse", &lse0, &lse1),
            ("gq", &g0.q, &g1.q),
            ("gk", &g0.k, &g1.k),
            ("gv", &g0.v, &g1.v),
        ] {
            let before = a.to_f32_vec().unwrap();
            let after = b.to_f32_vec().unwrap();
            if name == "gk" || name == "gv" {
                assert_eq!(before, after, "{c:?}: {name} depends on a padding row");
            } else {
                // Live rows are equal; padding rows are zeros in both.
                let width = if name == "lse" { 1 } else { c.d };
                // Batch row 0 keeps all 8 rows, so compare it whole.
                let whole = |v: &[f32]| plane_rows(v, c.h, c.tq, width, 0..c.tq);
                assert_eq!(whole(&before), whole(&after), "{c:?}: {name}, batch row 0");
                let live = |v: &[f32]| plane_rows(v, c.b * c.h, c.tq, width, 0..3);
                assert_eq!(live(&before), live(&after), "{c:?}: {name}, live rows");
                for v in [&before, &after] {
                    for plane in c.h..2 * c.h {
                        let lo = (plane * c.tq + 3) * width;
                        let hi = (plane + 1) * c.tq * width;
                        assert!(
                            v[lo..hi].iter().all(|&e| e == 0.0),
                            "{c:?}: {name}: a padding row of plane {plane} is not zeros"
                        );
                    }
                }
            }
        }
        if let (Some(a), Some(b)) = (&g0.bias, &g1.bias) {
            assert_eq!(
                bits(a),
                bits(b),
                "{c:?}: bias gradient depends on a padding row"
            );
        }
    }
    // Cross-attention with the same lengths is not padded: every query still
    // sees the keys below its length.
    let c = Case::new(2, 2, 2, 8, 9, 8).lens(&[9, 3]);
    let x = inputs(&c, 62);
    let t = Built::new(&c, &x);
    let y = be
        .attention_forward(&t.q, &t.k, &t.v, t.opts(&c))
        .unwrap()
        .to_f32_vec()
        .unwrap();
    for plane in c.h..2 * c.h {
        for i in 0..c.tq {
            let row = &y[(plane * c.tq + i) * c.d..(plane * c.tq + i + 1) * c.d];
            assert!(row.iter().any(|&e| e != 0.0), "plane {plane} query {i}");
        }
    }
}

/// Forward, saving forward and backward of `c` on `be`: the bits of
/// `[y, lse, gq, gk, gv]`.
fn run_bits(be: &CpuBackend, c: &Case, seed: u64) -> Vec<Vec<u32>> {
    let x = inputs(c, seed);
    let t = Built::new(c, &x);
    let opts = t.opts(c);
    let y = be.attention_forward(&t.q, &t.k, &t.v, opts).unwrap();
    let (_, lse) = be.attention_forward_saving(&t.q, &t.k, &t.v, opts).unwrap();
    let g = be
        .attention_backward(&t.q, &t.k, &t.v, &y, &lse, &t.gy, opts, false)
        .unwrap();
    vec![bits(&y), bits(&lse), bits(&g.q), bits(&g.k), bits(&g.v)]
}

#[test]
fn a_radius_that_hides_nothing_is_no_radius_bit_for_bit() {
    let be = backend(2, Numerics::Exact);
    let base = Case::new(2, 4, 2, 9, 9, 8).bias([1, 4, 9]).lens(&[9, 5]);
    let want = run_bits(&be, &base, 21);
    for radius in [8, 9, 1000, usize::MAX] {
        assert_eq!(
            run_bits(&be, &base.clone().radius(radius), 21),
            want,
            "radius {radius}"
        );
    }
    // One short of that cuts something: the bits change.
    assert_ne!(run_bits(&be, &base.clone().radius(7), 21), want);
}

#[test]
fn an_explicit_scale_equal_to_the_default_is_the_default_bit_for_bit() {
    let be = backend(1, Numerics::Exact);
    for d in [8usize, 16, 5] {
        let base = Case::new(1, 2, 2, 6, 7, d).lens(&[7]);
        let same = base.clone().scale(ojas_core::sdpa_scale(d as u32).unwrap());
        assert_eq!(run_bits(&be, &same, 31), run_bits(&be, &base, 31), "d {d}");
    }
}

#[test]
fn keys_outside_the_band_cannot_change_a_row() {
    // Perturb the key and value at one position: a query farther than the
    // radius from it must come out with the same bits, a nearer one must not.
    let be = backend(1, Numerics::Exact);
    let (t_len, radius, poked) = (13usize, 2usize, 6usize);
    let c = Case::new(1, 2, 2, t_len, t_len, 8).radius(radius);
    let x = inputs(&c, 41);
    let run = |x: &Inputs| {
        let t = Built::new(&c, x);
        be.attention_forward(&t.q, &t.k, &t.v, t.opts(&c))
            .unwrap()
            .to_f32_vec()
            .unwrap()
    };
    let before = run(&x);
    let mut moved = Inputs {
        q: x.q.clone(),
        k: x.k.clone(),
        v: x.v.clone(),
        gy: x.gy.clone(),
        bias: None,
    };
    for plane in 0..c.hkv {
        for col in 0..c.d {
            let at = (plane * t_len + poked) * c.d + col;
            moved.k[at] += 3.0;
            moved.v[at] -= 2.0;
        }
    }
    let after = run(&moved);
    for plane in 0..c.h {
        for i in 0..t_len {
            let row = (plane * t_len + i) * c.d..(plane * t_len + i + 1) * c.d;
            let same = before[row.clone()]
                .iter()
                .zip(&after[row])
                .all(|(a, b)| a.to_bits() == b.to_bits());
            assert_eq!(same, i.abs_diff(poked) > radius, "head {plane} query {i}");
        }
    }
}

#[test]
fn a_band_and_a_scale_refuse_what_they_cannot_mean() {
    let be = backend(1, Numerics::Exact);
    let c = Case::new(2, 4, 2, 5, 5, 8);
    let x = inputs(&c, 1);
    let t = Built::new(&c, &x);
    let none = AttentionOpts::default();
    let go = |opts: AttentionOpts<'_>| be.attention_forward(&t.q, &t.k, &t.v, opts);
    go(AttentionOpts {
        radius: Some(2),
        ..none
    })
    .unwrap();
    // Zero would read as "no band" in some kernels and "no key" in others.
    refused(
        go(AttentionOpts {
            radius: Some(0),
            ..none
        }),
        "radius 0",
    );
    refused(
        go(AttentionOpts {
            radius: Some(2),
            causal: true,
            ..none
        }),
        "radius with causal",
    );
    refused(
        go(AttentionOpts {
            radius: Some(2),
            window: Some(3),
            ..none
        }),
        "radius with a window",
    );
    // Cross-attention has no diagonal to centre the band on.
    let cross = Case::new(2, 4, 2, 5, 7, 8);
    let xc = inputs(&cross, 2);
    let tc = Built::new(&cross, &xc);
    refused(
        be.attention_forward(
            &tc.q,
            &tc.k,
            &tc.v,
            AttentionOpts {
                radius: Some(2),
                ..none
            },
        ),
        "radius with Tq != Tk",
    );
    for scale in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for result in [
            go(AttentionOpts {
                scale: Some(scale),
                ..none
            })
            .map(drop),
            be.attention_forward_saving(
                &t.q,
                &t.k,
                &t.v,
                AttentionOpts {
                    scale: Some(scale),
                    ..none
                },
            )
            .map(drop),
        ] {
            assert!(
                matches!(result, Err(OjasError::NonFinite { .. })),
                "{scale}: {result:?}"
            );
        }
    }
    // A zero or negative scale is legal.
    for scale in [0.0f32, -1.0] {
        go(AttentionOpts {
            scale: Some(scale),
            ..none
        })
        .unwrap();
    }
}

fn refused<T: std::fmt::Debug>(result: Result<T, OjasError>, what: &str) -> OjasError {
    match result {
        Err(e @ OjasError::Shape { .. }) | Err(e @ OjasError::Dtype { .. }) => e,
        other => panic!("{what}: expected Shape or Dtype, got {other:?}"),
    }
}

#[test]
fn malformed_calls_are_refused() {
    let be = backend(1, Numerics::Exact);
    let c = Case::new(2, 4, 2, 5, 7, 8);
    let x = inputs(&c, 1);
    let t = Built::new(&c, &x);
    let go = |opts: AttentionOpts<'_>| be.attention_forward(&t.q, &t.k, &t.v, opts);
    let none = AttentionOpts::default();
    go(none).unwrap();

    // Window: needs causal, and at least 1.
    refused(
        go(AttentionOpts {
            window: Some(3),
            ..none
        }),
        "window without causal",
    );
    refused(
        go(AttentionOpts {
            causal: true,
            window: Some(0),
            ..none
        }),
        "window 0",
    );

    // Bias shapes.
    let bias = |shape: &[usize]| tensor(&vec![0.5; shape.iter().product()], shape);
    for shape in [
        &[2usize, 4, 5][..],
        &[3, 1, 1, 7],
        &[1, 3, 1, 7],
        &[1, 1, 4, 7],
        &[1, 1, 1, 6],
        &[2, 4, 5, 7, 1],
    ] {
        let b = bias(shape);
        refused(
            go(AttentionOpts {
                bias: Some(&b),
                ..none
            }),
            &format!("bias {shape:?}"),
        );
    }
    let ok = bias(&[1, 4, 1, 7]);
    go(AttentionOpts {
        bias: Some(&ok),
        ..none
    })
    .unwrap();

    // Key lengths: dtype and shape.
    let f32_lens = tensor(&[1.0, 2.0], &[2]);
    refused(
        go(AttentionOpts {
            key_lens: Some(&f32_lens),
            ..none
        }),
        "f32 key_lens",
    );
    let short = lens_tensor(&[1]);
    refused(
        go(AttentionOpts {
            key_lens: Some(&short),
            ..none
        }),
        "key_lens [1] for batch 2",
    );

    // Operands.
    let bad_k = tensor(&vec![0.0; 2 * 2 * 7 * 4], &[2, 2, 7, 4]);
    refused(
        be.attention_forward(&t.q, &bad_k, &bad_k, none),
        "head dim mismatch",
    );
    let three = tensor(&vec![0.0; 2 * 3 * 7 * 8], &[2, 3, 7, 8]);
    refused(
        be.attention_forward(&t.q, &three, &three, none),
        "4 heads over 3 kv heads",
    );
    refused(
        be.attention_forward(&t.q, &t.k, &t.q, none),
        "k and v differ",
    );
    let rank3 = tensor(&vec![0.0; 2 * 5 * 8], &[2, 5, 8]);
    refused(be.attention_forward(&rank3, &t.k, &t.v, none), "q rank 3");

    // Backward: bias gradient needs a bias, and the saved tensors' shapes.
    let (y, lse) = be.attention_forward_saving(&t.q, &t.k, &t.v, none).unwrap();
    refused(
        be.attention_backward(&t.q, &t.k, &t.v, &y, &lse, &t.gy, none, true),
        "bias gradient without a bias",
    );
    refused(
        be.attention_backward(&t.q, &t.k, &t.v, &y, &t.gy, &t.gy, none, false),
        "lse shaped like q",
    );
    refused(
        be.attention_backward(&t.q, &t.k, &t.v, &t.k, &lse, &t.gy, none, false),
        "output shaped like k",
    );
    refused(
        be.attention_backward(&t.q, &t.k, &t.v, &y, &lse, &t.k, none, false),
        "grad shaped like k",
    );
}

#[test]
fn a_non_finite_operand_is_refused() {
    let be = backend(1, Numerics::Exact);
    let c = Case::new(1, 2, 2, 3, 4, 8).bias([1, 1, 1]);
    let x = inputs(&c, 2);
    let t = Built::new(&c, &x);
    let nan_bias = tensor(&[f32::NAN, 0.0, 0.0, 0.0], &[1, 1, 1, 4]);
    let inf_bias = tensor(&[0.0, f32::INFINITY, 0.0, 0.0], &[1, 1, 1, 4]);
    for b in [&nan_bias, &inf_bias] {
        let opts = AttentionOpts {
            bias: Some(b),
            ..t.opts(&c)
        };
        match be.attention_forward(&t.q, &t.k, &t.v, opts) {
            Err(OjasError::NonFinite { .. }) => {}
            other => panic!("expected NonFinite, got {other:?}"),
        }
    }
    let mut q = x.q.clone();
    q[0] = f32::NAN;
    let q = tensor(&q, &c.q_shape());
    match be.attention_forward(&q, &t.k, &t.v, t.opts(&c)) {
        Err(OjasError::NonFinite { .. }) => {}
        other => panic!("expected NonFinite, got {other:?}"),
    }
}

/// A pass too small to fan out runs on the calling thread, so it needs one
/// task's scratch whatever the pool size: many heads of tiny work must fit
/// a budget that only holds the outputs and a single task's buffers.
#[test]
fn a_serial_pass_reserves_one_tasks_scratch_not_one_per_worker() {
    let c = Case::new(1, 32, 32, 2, 16, 8).causal(None);
    let x = inputs(&c, 77);
    let built = Built::new(&c, &x);
    let tight = CpuBackend::with_threads(Budget::new(4096), 8)
        .unwrap()
        .with_numerics(Numerics::Exact);
    let (out, lse) = tight
        .attention_forward_saving(&built.q, &built.k, &built.v, built.opts(&c))
        .unwrap();
    let roomy = backend(8, Numerics::Exact);
    let (want, want_lse) = roomy
        .attention_forward_saving(&built.q, &built.k, &built.v, built.opts(&c))
        .unwrap();
    assert_eq!(bits(&out), bits(&want));
    assert_eq!(bits(&lse), bits(&want_lse));
    // Backward: the output, its lse, the three gradients and one task's two
    // length-Tk buffers fit; eight tasks' worth (1 KiB) would not.
    let (kv_bytes, q_bytes) = (32 * 16 * 8 * 4, 32 * 2 * 8 * 4);
    let backward = CpuBackend::with_threads(
        Budget::new((q_bytes + 32 * 2 * 4 + q_bytes + 2 * kv_bytes + 512) as u64),
        8,
    )
    .unwrap()
    .with_numerics(Numerics::Exact);
    let (out, lse) = backward
        .attention_forward_saving(&built.q, &built.k, &built.v, built.opts(&c))
        .unwrap();
    let grad = backward.attention_backward(
        &built.q,
        &built.k,
        &built.v,
        &out,
        &lse,
        &out,
        built.opts(&c),
        false,
    );
    assert!(grad.is_ok(), "{grad:?}");
}

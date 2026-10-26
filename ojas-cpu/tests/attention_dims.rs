//! General attention at the SAM 3.1 shapes `attention_general.rs` does not
//! cover: head dim 256 (memory attention, 1 head), head dim 64 at a ViT
//! window's length (576), a global-length key set (5184), `Tq = 1` /
//! `Tk = 1`, and a `Tk = 1` batch row whose key length is 0.
//! The reference is an f64 forward written in this file (output and
//! log-sum-exp); the gradients at these dims are not re-derived here.

use ojas_core::{AttentionOpts, Backend, Budget, Numerics, Tensor};
use ojas_cpu::CpuBackend;

fn backend(threads: usize) -> CpuBackend {
    CpuBackend::with_threads(Budget::new(1 << 40), threads)
        .unwrap()
        .with_numerics(Numerics::Exact)
}

fn f32t(values: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_f32(values, shape, &Budget::new(u64::MAX)).unwrap()
}

fn lens_t(values: &[u32]) -> Tensor {
    Tensor::from_u32(values, &[values.len()], &Budget::new(u64::MAX)).unwrap()
}

/// Deterministic values in `[-amp, amp)` (the generator of attention_general.rs).
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

#[derive(Clone, Debug)]
struct Case {
    b: usize,
    h: usize,
    hkv: usize,
    tq: usize,
    tk: usize,
    d: usize,
    amp: f32,
    /// `[Bb, Hb, Tqb]`; the last axis is `Tk`.
    bias: Option<[usize; 3]>,
    lens: Option<Vec<u32>>,
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
            amp: 1.0,
            bias: None,
            lens: None,
        }
    }
    fn bias(mut self, s: [usize; 3]) -> Self {
        self.bias = Some(s);
        self
    }
    fn lens(mut self, l: &[u32]) -> Self {
        self.lens = Some(l.to_vec());
        self
    }
    fn amp(mut self, a: f32) -> Self {
        self.amp = a;
        self
    }
    fn len_of(&self, b: usize) -> usize {
        self.lens
            .as_ref()
            .map_or(self.tk, |l| (l[b] as usize).min(self.tk))
    }
}

/// `(output, lse)` in f64 for a bidirectional case; a batch row with no keys
/// is zeros and lse 0, as `AttentionOpts` documents.
fn oracle(c: &Case, q: &[f32], k: &[f32], v: &[f32], bias: Option<&[f32]>) -> (Vec<f64>, Vec<f64>) {
    let rep = c.h / c.hkv;
    let scale = 1.0 / (c.d as f64).sqrt();
    let mut out = vec![0.0f64; c.b * c.h * c.tq * c.d];
    let mut lse = vec![0.0f64; c.b * c.h * c.tq];
    for b in 0..c.b {
        let len = c.len_of(b);
        for h in 0..c.h {
            let plane = b * c.h + h;
            let kvp = b * c.hkv + h / rep;
            for i in 0..c.tq {
                if len == 0 {
                    continue;
                }
                let s: Vec<f64> = (0..len)
                    .map(|j| {
                        let dot: f64 = (0..c.d)
                            .map(|x| {
                                f64::from(q[(plane * c.tq + i) * c.d + x])
                                    * f64::from(k[(kvp * c.tk + j) * c.d + x])
                            })
                            .sum();
                        let extra = c.bias.map_or(0.0, |[bb, bh, bq]| {
                            let pick = |e: usize, at: usize| if e == 1 { 0 } else { at };
                            let at =
                                ((pick(bb, b) * bh + pick(bh, h)) * bq + pick(bq, i)) * c.tk + j;
                            f64::from(bias.unwrap()[at])
                        });
                        dot * scale + extra
                    })
                    .collect();
                let max = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let z: f64 = s.iter().map(|x| (x - max).exp()).sum();
                lse[plane * c.tq + i] = max + z.ln();
                for (j, x) in s.iter().enumerate() {
                    let p = (x - max).exp() / z;
                    for e in 0..c.d {
                        out[(plane * c.tq + i) * c.d + e] +=
                            p * f64::from(v[(kvp * c.tk + j) * c.d + e]);
                    }
                }
            }
        }
    }
    (out, lse)
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

fn run(c: &Case, seed: u64, tol: f64) {
    let be = backend(1);
    let n = |s: &[usize]| s.iter().product::<usize>();
    let (qs, ks) = ([c.b, c.h, c.tq, c.d], [c.b, c.hkv, c.tk, c.d]);
    let q = values(n(&qs), seed, c.amp);
    let k = values(n(&ks), seed + 1, c.amp);
    let v = values(n(&ks), seed + 2, 1.0);
    let bias = c
        .bias
        .map(|[bb, bh, bq]| values(bb * bh * bq * c.tk, seed + 3, 2.0));
    let (want, want_lse) = oracle(c, &q, &k, &v, bias.as_deref());
    let (qt, kt, vt) = (f32t(&q, &qs), f32t(&k, &ks), f32t(&v, &ks));
    let bias_t = c
        .bias
        .map(|[bb, bh, bq]| f32t(bias.as_ref().unwrap(), &[bb, bh, bq, c.tk]));
    let lens_tensor = c.lens.as_ref().map(|l| lens_t(l));
    let opts = AttentionOpts {
        causal: false,
        window: None,
        bias: bias_t.as_ref(),
        key_lens: lens_tensor.as_ref(),
    };
    let (y, lse) = be.attention_forward_saving(&qt, &kt, &vt, opts).unwrap();
    let e = err(&y, &want);
    assert!(e <= tol, "{c:?}: output error {e}");
    let e = err(&lse, &want_lse);
    assert!(e <= tol, "{c:?}: lse error {e}");
    let plain = be.attention_forward(&qt, &kt, &vt, opts).unwrap();
    assert_eq!(
        plain.to_f32_vec().unwrap(),
        y.to_f32_vec().unwrap(),
        "{c:?}: forward forms differ"
    );
}

#[test]
fn head_dim_256_matches_the_f64_reference() {
    // Memory attention: one head, 256 wide, a bias and a padded key set.
    run(
        &Case::new(1, 1, 1, 40, 70, 256).bias([1, 1, 40]).lens(&[50]),
        1,
        5e-5,
    );
    // Grouped heads, a per-batch bias and one batch row with 5 live keys.
    run(
        &Case::new(2, 4, 2, 9, 33, 256)
            .bias([2, 4, 1])
            .lens(&[33, 5]),
        2,
        5e-5,
    );
}

#[test]
fn head_dim_64_at_vit_window_and_prompt_lengths() {
    // One window of the windowed ViT: 576 tokens, no mask.
    run(&Case::new(1, 2, 2, 576, 576, 64), 3, 5e-5);
    // The text prompt: 32 padded keys, 7 live in batch row 0.
    run(&Case::new(2, 8, 8, 100, 32, 64).lens(&[7, 32]), 4, 5e-5);
}

#[test]
fn head_dim_32_detector_bias_per_head() {
    run(&Case::new(1, 8, 8, 50, 144, 32).bias([1, 8, 50]), 5, 5e-5);
}

#[test]
fn a_sharper_softmax_stays_finite_and_matches() {
    // Scores of magnitude tens: the row max is subtracted before exp.
    run(&Case::new(1, 2, 2, 20, 90, 64).amp(4.0), 6, 5e-5);
}

#[test]
fn one_query_over_a_global_length_key_set() {
    // Tq = 1 over 5184 keys, 5000 live (the global ViT layer's row length).
    run(&Case::new(1, 2, 2, 1, 5184, 64).lens(&[5000]), 7, 5e-5);
}

#[test]
fn single_key_and_single_query_extremes() {
    // Tk = 1: the softmax is 1, so the output is that key's value row.
    let be = backend(1);
    let (q, k, v) = (
        values(2 * 3 * 4 * 64, 1, 1.0),
        values(2 * 3 * 64, 2, 1.0),
        values(2 * 3 * 64, 3, 1.0),
    );
    let (qt, kt, vt) = (
        f32t(&q, &[2, 3, 4, 64]),
        f32t(&k, &[2, 3, 1, 64]),
        f32t(&v, &[2, 3, 1, 64]),
    );
    let y = be
        .attention_forward(&qt, &kt, &vt, AttentionOpts::default())
        .unwrap()
        .to_f32_vec()
        .unwrap();
    for plane in 0..6 {
        for i in 0..4 {
            for e in 0..64 {
                let got = y[(plane * 4 + i) * 64 + e];
                let want = v[plane * 64 + e];
                assert!((got - want).abs() <= 1e-6, "plane {plane} row {i} dim {e}");
            }
        }
    }
    // Tk = 1 with key length 0: that batch row is zeros with lse 0, not NaN.
    let lens = lens_t(&[1, 0]);
    let opts = AttentionOpts {
        key_lens: Some(&lens),
        ..AttentionOpts::default()
    };
    let (y, lse) = be.attention_forward_saving(&qt, &kt, &vt, opts).unwrap();
    let (y, lse) = (y.to_f32_vec().unwrap(), lse.to_f32_vec().unwrap());
    let half = 3 * 4 * 64;
    assert!(
        y[half..].iter().all(|&x| x == 0.0),
        "empty batch row output"
    );
    assert!(
        lse[3 * 4..].iter().all(|&x| x == 0.0),
        "empty batch row lse"
    );
    assert!(y[..half].iter().all(|x| x.is_finite()));
    // Tq = 1, Tk = 1, D = 64 bidirectional.
    run(&Case::new(1, 1, 1, 1, 1, 64), 8, 1e-6);
}

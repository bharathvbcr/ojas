//! `Tape::attention`: general attention (bidirectional, cross, grouped,
//! biased, key-length masked) against f64 central differences, with the
//! bias a differentiable input; a frozen bias takes no gradient; and a
//! checkpointed segment replays it bit for bit.

use ojas_autograd::{central_diff, gradients_match, Tape, Var};
use ojas_core::{Budget, Numerics, OjasError, Tensor};
use ojas_cpu::CpuBackend;

const H: f64 = 1e-3;
const ATOL: f64 = 2e-3;
const RTOL: f64 = 2e-2;

fn cpu(numerics: Numerics) -> CpuBackend {
    CpuBackend::new(Budget::new(1 << 24)).with_numerics(numerics)
}

fn f32t(budget: &Budget, data: &[f64], shape: &[usize]) -> Tensor {
    let f: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    Tensor::from_f32(&f, shape, budget).unwrap()
}

/// Deterministic values in `[-amp, amp)`.
fn vals(seed: u64, n: usize, amp: f64) -> Vec<f64> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f64 / (1u64 << 24) as f64 * 2.0 - 1.0) * amp
        })
        .collect()
}

#[derive(Clone, Copy)]
struct Shape {
    b: usize,
    h: usize,
    hkv: usize,
    tq: usize,
    tk: usize,
    d: usize,
    /// `[Bb, Hb, Tqb]`.
    bias: [usize; 3],
    causal: bool,
    window: Option<usize>,
}

const CASES: [Shape; 3] = [
    // Cross-attention, grouped heads, a bias broadcast over the batch.
    Shape {
        b: 2,
        h: 2,
        hkv: 1,
        tq: 3,
        tk: 4,
        d: 2,
        bias: [1, 2, 3],
        causal: false,
        window: None,
    },
    // Bidirectional, a bias that broadcasts over heads and queries.
    Shape {
        b: 2,
        h: 2,
        hkv: 2,
        tq: 3,
        tk: 3,
        d: 2,
        bias: [2, 1, 1],
        causal: false,
        window: None,
    },
    // Causal prefill with a window.
    Shape {
        b: 1,
        h: 2,
        hkv: 2,
        tq: 3,
        tk: 5,
        d: 2,
        bias: [1, 1, 3],
        causal: true,
        window: Some(2),
    },
];

const LENS: [u32; 2] = [2, 4];

fn lens_for(s: &Shape) -> Vec<u32> {
    // Key lengths, kept inside Tk, and never empty for a causal case.
    LENS.iter().take(s.b).map(|&l| l.min(s.tk as u32)).collect()
}

fn visible(s: &Shape, lens: &[u32], b: usize, i: usize, j: usize) -> bool {
    if j >= (lens[b] as usize).min(s.tk) {
        return false;
    }
    if s.causal {
        let pos = (i + s.tk) as isize - s.tq as isize;
        if j as isize > pos || s.window.is_some_and(|w| (j + w) as isize <= pos) {
            return false;
        }
    }
    true
}

/// `sum(out * w)` of the f64 forward.
#[allow(clippy::too_many_arguments)]
fn loss(s: &Shape, lens: &[u32], q: &[f64], k: &[f64], v: &[f64], bias: &[f64], w: &[f64]) -> f64 {
    let rep = s.h / s.hkv;
    let scale = 1.0 / (s.d as f64).sqrt();
    let mut sum = 0.0;
    for b in 0..s.b {
        for h in 0..s.h {
            let plane = b * s.h + h;
            let kvp = b * s.hkv + h / rep;
            for i in 0..s.tq {
                let keys: Vec<usize> = (0..s.tk).filter(|&j| visible(s, lens, b, i, j)).collect();
                if keys.is_empty() {
                    continue;
                }
                let pick = |extent: usize, at: usize| if extent == 1 { 0 } else { at };
                let scores: Vec<f64> = keys
                    .iter()
                    .map(|&j| {
                        let dot: f64 = (0..s.d)
                            .map(|c| {
                                q[(plane * s.tq + i) * s.d + c] * k[(kvp * s.tk + j) * s.d + c]
                            })
                            .sum();
                        let at = ((pick(s.bias[0], b) * s.bias[1] + pick(s.bias[1], h))
                            * s.bias[2]
                            + pick(s.bias[2], i))
                            * s.tk
                            + j;
                        dot * scale + bias[at]
                    })
                    .collect();
                let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let z: f64 = scores.iter().map(|x| (x - max).exp()).sum();
                for c in 0..s.d {
                    let out: f64 = keys
                        .iter()
                        .zip(&scores)
                        .map(|(&j, x)| (x - max).exp() / z * v[(kvp * s.tk + j) * s.d + c])
                        .sum();
                    sum += out * w[(plane * s.tq + i) * s.d + c];
                }
            }
        }
    }
    sum
}

struct Parts {
    q: Vec<f64>,
    k: Vec<f64>,
    v: Vec<f64>,
    bias: Vec<f64>,
    w: Vec<f64>,
}

fn parts(s: &Shape, seed: u64) -> Parts {
    let q_n = s.b * s.h * s.tq * s.d;
    let kv_n = s.b * s.hkv * s.tk * s.d;
    Parts {
        q: vals(seed, q_n, 0.8),
        k: vals(seed + 1, kv_n, 0.8),
        v: vals(seed + 2, kv_n, 0.8),
        bias: vals(seed + 3, s.bias.iter().product::<usize>() * s.tk, 1.0),
        w: vals(seed + 4, q_n, 1.0),
    }
}

/// Tape variables of a case, in order `q, k, v, bias, w`.
fn record(
    t: &mut Tape,
    s: &Shape,
    p: &Parts,
    frozen_bias: bool,
) -> Result<([Var; 5], Var), OjasError> {
    let budget = t.backend().budget().clone();
    let q = t.leaf(f32t(&budget, &p.q, &[s.b, s.h, s.tq, s.d]))?;
    let k = t.leaf(f32t(&budget, &p.k, &[s.b, s.hkv, s.tk, s.d]))?;
    let v = t.leaf(f32t(&budget, &p.v, &[s.b, s.hkv, s.tk, s.d]))?;
    let bias_t = f32t(&budget, &p.bias, &[s.bias[0], s.bias[1], s.bias[2], s.tk]);
    let bias = if frozen_bias {
        t.frozen_leaf(bias_t)?
    } else {
        t.leaf(bias_t)?
    };
    let w = t.frozen_leaf(f32t(&budget, &p.w, &[s.b, s.h, s.tq, s.d]))?;
    let lens = Tensor::from_u32(&lens_for(s), &[s.b], &budget)?;
    let y = t.attention(q, k, v, Some(bias), Some(lens), s.causal, s.window)?;
    let z = t.mul(y, w)?;
    Ok(([q, k, v, bias, w], z))
}

fn check(analytic: &[f32], numeric: &[f64], what: &str) {
    gradients_match(analytic, numeric, ATOL, RTOL).unwrap_or_else(|e| panic!("{what}: {e:?}"));
}

#[test]
fn gradcheck_attention_q_k_v_and_bias() {
    for (n, s) in CASES.iter().enumerate() {
        let p = parts(s, 10 + n as u64);
        let lens = lens_for(s);
        let mut tape = Tape::new(cpu(Numerics::Exact));
        let ([q, k, v, bias, _], z) = record(&mut tape, s, &p, false).unwrap();
        tape.backward(z).unwrap();
        let l = |q: &[f64], k: &[f64], v: &[f64], b: &[f64]| loss(s, &lens, q, k, v, b, &p.w);
        check(
            &tape.grad(q).unwrap().to_f32_vec().unwrap(),
            &central_diff(&p.q, H, |x| Ok(l(x, &p.k, &p.v, &p.bias))).unwrap(),
            &format!("case {n} q"),
        );
        check(
            &tape.grad(k).unwrap().to_f32_vec().unwrap(),
            &central_diff(&p.k, H, |x| Ok(l(&p.q, x, &p.v, &p.bias))).unwrap(),
            &format!("case {n} k"),
        );
        check(
            &tape.grad(v).unwrap().to_f32_vec().unwrap(),
            &central_diff(&p.v, H, |x| Ok(l(&p.q, &p.k, x, &p.bias))).unwrap(),
            &format!("case {n} v"),
        );
        check(
            &tape.grad(bias).unwrap().to_f32_vec().unwrap(),
            &central_diff(&p.bias, H, |x| Ok(l(&p.q, &p.k, &p.v, x))).unwrap(),
            &format!("case {n} bias"),
        );
    }
}

#[test]
fn a_frozen_bias_takes_no_gradient_and_the_others_are_unchanged() {
    for (n, s) in CASES.iter().enumerate() {
        let p = parts(s, 20 + n as u64);
        let mut live = Tape::new(cpu(Numerics::Exact));
        let (vars, z) = record(&mut live, s, &p, false).unwrap();
        live.backward(z).unwrap();
        let mut frozen = Tape::new(cpu(Numerics::Exact));
        let (fvars, fz) = record(&mut frozen, s, &p, true).unwrap();
        frozen.backward(fz).unwrap();
        assert!(
            frozen.grad(fvars[3]).is_none(),
            "case {n}: frozen bias grad"
        );
        for (i, name) in ["q", "k", "v"].iter().enumerate() {
            let a = live.grad(vars[i]).unwrap().to_f32_vec().unwrap();
            let b = frozen.grad(fvars[i]).unwrap().to_f32_vec().unwrap();
            assert_eq!(a, b, "case {n}: {name}");
        }
    }
}

#[test]
fn a_checkpointed_segment_replays_attention_bit_for_bit() {
    let s = CASES[0];
    let p = parts(&s, 30);
    for numerics in [Numerics::Exact, Numerics::Fast] {
        let run = |checkpoint: bool| {
            let mut t = Tape::new(cpu(numerics));
            let budget = t.backend().budget().clone();
            let q = t.leaf(f32t(&budget, &p.q, &[s.b, s.h, s.tq, s.d])).unwrap();
            let k = t
                .leaf(f32t(&budget, &p.k, &[s.b, s.hkv, s.tk, s.d]))
                .unwrap();
            let v = t
                .leaf(f32t(&budget, &p.v, &[s.b, s.hkv, s.tk, s.d]))
                .unwrap();
            let bias = t
                .leaf(f32t(
                    &budget,
                    &p.bias,
                    &[s.bias[0], s.bias[1], s.bias[2], s.tk],
                ))
                .unwrap();
            let lens = Tensor::from_u32(&lens_for(&s), &[s.b], &budget).unwrap();
            let body = |t: &mut Tape| -> Result<Vec<Var>, OjasError> {
                let y = t.attention(q, k, v, Some(bias), Some(lens.clone()), false, None)?;
                Ok(vec![y])
            };
            let out = if checkpoint {
                t.checkpoint(body).unwrap()
            } else {
                body(&mut t).unwrap()
            };
            let root = t.mul(out[0], out[0]).unwrap();
            t.backward(root).unwrap();
            [q, k, v, bias].map(|x| t.grad(x).unwrap().to_f32_vec().unwrap())
        };
        assert_eq!(run(true), run(false), "{numerics:?}");
    }
}

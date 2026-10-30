//! The encoder ops on the tape: `layer_norm`, `channel_layer_norm`,
//! `gelu_erf`, `linear_bias`, `broadcast_add`, `broadcast_mul`, `scale` and
//! `softmax` (ft-96baeabc, criteria 1, 3, 4, 5 and 6).
//!
//! - A small encoder block recorded on the tape must give the gradients of the
//!   backend's backward calls chained by hand, bit for bit, recorded
//!   directly or inside a checkpointed segment, and with a frozen weight. The
//!   values of those backwards are checked against `f64` differences and
//!   torch in `ojas-cpu/tests/encoder_backward.rs`.
//! - The same block's gradients are held to central differences of an `f64`
//!   forward, so a gradient the tape routes to the wrong input fails here.
//! - A node used twice (through two broadcast shapes) sums its gradients; a
//!   record the backend refuses pushes nothing.

use ojas_autograd::{central_diff, gradients_match, Tape, Var};
use ojas_core::{erf_f64, Backend, Budget, Numerics, OjasError, Tensor};
use ojas_cpu::CpuBackend;

const EPS: f32 = 1e-5;
const SCALE: f32 = 0.7;
const TOKENS: usize = 3;
const IN: usize = 6;
const OUT: usize = 5;

fn cpu() -> CpuBackend {
    CpuBackend::with_threads(Budget::new(1 << 24), 2)
        .unwrap()
        .with_numerics(Numerics::Exact)
}

/// Deterministic values in `[-1, 1)`.
fn vals(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            -1.0 + 2.0 * ((s >> 40) as f32 / (1u64 << 24) as f32)
        })
        .collect()
}

fn tensor(be: &CpuBackend, seed: u64, shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::from_f32(&vals(seed, n), shape, be.budget()).unwrap()
}

/// A scale near 1, so a norm's weight is not degenerate.
fn near_one(be: &CpuBackend, seed: u64, shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = vals(seed, n).iter().map(|v| 1.0 + 0.4 * v).collect();
    Tensor::from_f32(&data, shape, be.budget()).unwrap()
}

fn bits(t: &Tensor) -> Vec<u32> {
    t.to_f32_vec()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn ones_like(be: &CpuBackend, t: &Tensor) -> Tensor {
    let n: usize = t.shape().iter().product();
    Tensor::from_f32(&vec![1.0; n], t.shape(), be.budget()).unwrap()
}

/// The block's leaves, in the order of [`NAMES`].
struct Block {
    x: Tensor,
    w: Tensor,
    b: Tensor,
    g: Tensor,
    beta: Tensor,
    s: Tensor,
    p: Tensor,
    r: Tensor,
}

const NAMES: [&str; 8] = ["x", "w", "b", "g", "beta", "s", "p", "r"];

fn block(be: &CpuBackend) -> Block {
    Block {
        x: tensor(be, 1, &[2, TOKENS, IN]),
        w: tensor(be, 2, &[OUT, IN]),
        b: tensor(be, 3, &[OUT]),
        g: near_one(be, 4, &[OUT]),
        beta: tensor(be, 5, &[OUT]),
        s: near_one(be, 6, &[TOKENS, 1]),
        p: tensor(be, 7, &[OUT]),
        r: tensor(be, 8, &[2, TOKENS, OUT]),
    }
}

impl Block {
    fn all(&self) -> [&Tensor; 8] {
        [
            &self.x, &self.w, &self.b, &self.g, &self.beta, &self.s, &self.p, &self.r,
        ]
    }
}

/// linear with bias -> exact GELU -> LayerNorm -> times a per-token scale
/// (a `[TOKENS, 1]` broadcast) -> plus a bias row -> scalar scale -> softmax
/// -> times a fixed weighting `r`, so the sum of the output is not a constant.
fn net(tape: &mut Tape<CpuBackend>, v: &[Var; 8]) -> Var {
    let [x, w, b, g, beta, s, p, r] = *v;
    let h = tape.linear_bias(x, w, b).unwrap();
    let h = tape.gelu_erf(h).unwrap();
    let h = tape.layer_norm(h, g, beta, EPS).unwrap();
    let h = tape.broadcast_mul(h, s).unwrap();
    let h = tape.broadcast_add(h, p).unwrap();
    let h = tape.scale(h, SCALE).unwrap();
    let y = tape.softmax(h).unwrap();
    tape.broadcast_mul(y, r).unwrap()
}

/// The same chain through the backend: the gradient of every leaf, for a seed
/// of ones.
fn by_hand(be: &CpuBackend, k: &Block) -> [Tensor; 8] {
    let h1 = be.linear_bias_forward(&k.x, &k.w, &k.b).unwrap();
    let h2 = be.gelu_erf_forward(&h1).unwrap();
    let h3 = be.layer_norm_forward(&h2, &k.g, &k.beta, EPS).unwrap();
    let h4 = be.broadcast_mul_forward(&h3, &k.s).unwrap();
    let h5 = be.broadcast_add_forward(&h4, &k.p).unwrap();
    let h6 = be.scale_forward(&h5, SCALE).unwrap();
    let y = be.softmax_forward(&h6).unwrap();
    let z = be.broadcast_mul_forward(&y, &k.r).unwrap();
    let (gy, gr) = be
        .broadcast_mul_backward(&y, &k.r, &ones_like(be, &z))
        .unwrap();
    let g6 = be.softmax_backward(&y, &gy).unwrap();
    let g5 = be.scale_forward(&g6, SCALE).unwrap();
    let (g4, gp) = be.broadcast_add_backward(&h4, &k.p, &g5).unwrap();
    let (g3, gs) = be.broadcast_mul_backward(&h3, &k.s, &g4).unwrap();
    let ln = be
        .layer_norm_backward(&h2, &k.g, &k.beta, &g3, EPS)
        .unwrap();
    let g1 = be.gelu_erf_backward(&h1, &ln.input).unwrap();
    let lin = be.linear_bias_backward(&k.x, &k.w, &k.b, &g1).unwrap();
    [
        lin.input, lin.weight, lin.bias, ln.weight, ln.bias, gs, gp, gr,
    ]
}

fn leaves(tape: &mut Tape<CpuBackend>, k: &Block) -> [Var; 8] {
    k.all().map(|t| tape.leaf(t.clone()).unwrap())
}

#[test]
fn the_tape_chains_the_encoder_backwards() {
    let be = cpu();
    let k = block(&be);
    let want = by_hand(&be, &k);
    let mut tape = Tape::new(be.clone());
    let v = leaves(&mut tape, &k);
    let z = net(&mut tape, &v);
    assert_eq!(tape.value(z).unwrap().shape(), &[2, TOKENS, OUT]);
    tape.backward(z).unwrap();
    for ((var, want), name) in v.iter().zip(&want).zip(NAMES) {
        let got = tape
            .grad(*var)
            .unwrap_or_else(|| panic!("{name} takes a gradient"));
        assert_eq!(got.shape(), want.shape(), "{name}");
        assert_eq!(bits(got), bits(want), "{name}");
    }
}

#[test]
fn a_checkpointed_encoder_segment_gives_the_same_bits() {
    let be = cpu();
    let k = block(&be);
    let want = by_hand(&be, &k);
    let mut tape = Tape::new(be.clone());
    let v = leaves(&mut tape, &k);
    let out = tape.checkpoint(|t| Ok(vec![net(t, &v)])).unwrap();
    tape.backward(out[0]).unwrap();
    for ((var, want), name) in v.iter().zip(&want).zip(NAMES) {
        assert_eq!(bits(tape.grad(*var).unwrap()), bits(want), "{name}");
    }
}

#[test]
fn a_frozen_weight_takes_no_gradient_and_the_rest_do() {
    let be = cpu();
    let k = block(&be);
    let want = by_hand(&be, &k);
    let mut tape = Tape::new(be.clone());
    let frozen = tape.frozen_leaf(k.w.clone()).unwrap();
    let mut v = leaves(&mut tape, &k);
    v[1] = frozen;
    let z = net(&mut tape, &v);
    tape.backward(z).unwrap();
    assert!(tape.grad(frozen).is_none());
    for (i, name) in NAMES.iter().enumerate() {
        if i != 1 {
            assert_eq!(bits(tape.grad(v[i]).unwrap()), bits(&want[i]), "{name}");
        }
    }
}

/// The channel branch of a neck: LayerNorm2d -> GELU -> plus a `[C, 1, 1]`
/// bias.
fn neck(tape: &mut Tape<CpuBackend>, v: &[Var; 4]) -> Var {
    let [x, g, b, q] = *v;
    let h = tape.channel_layer_norm(x, g, b, 1e-6).unwrap();
    let h = tape.gelu_erf(h).unwrap();
    tape.broadcast_add(h, q).unwrap()
}

#[test]
fn the_tape_chains_the_channel_layer_norm_backward() {
    let be = cpu();
    let x = tensor(&be, 11, &[2, 4, 3, 3]);
    let g = near_one(&be, 12, &[4]);
    let b = tensor(&be, 13, &[4]);
    let q = tensor(&be, 14, &[4, 1, 1]);
    let h1 = be.channel_layer_norm_forward(&x, &g, &b, 1e-6).unwrap();
    let h2 = be.gelu_erf_forward(&h1).unwrap();
    let out = be.broadcast_add_forward(&h2, &q).unwrap();
    let (g2, gq) = be
        .broadcast_add_backward(&h2, &q, &ones_like(&be, &out))
        .unwrap();
    let g1 = be.gelu_erf_backward(&h1, &g2).unwrap();
    let ln = be
        .channel_layer_norm_backward(&x, &g, &b, &g1, 1e-6)
        .unwrap();
    let want = [ln.input, ln.weight, ln.bias, gq];
    let mut tape = Tape::new(be.clone());
    let v = [&x, &g, &b, &q].map(|t| tape.leaf(t.clone()).unwrap());
    let z = neck(&mut tape, &v);
    tape.backward(z).unwrap();
    for (var, want) in v.iter().zip(&want) {
        assert_eq!(bits(tape.grad(*var).unwrap()), bits(want));
    }
    let mut tape = Tape::new(be.clone());
    let v = [&x, &g, &b, &q].map(|t| tape.leaf(t.clone()).unwrap());
    let out = tape.checkpoint(|t| Ok(vec![neck(t, &v)])).unwrap();
    tape.backward(out[0]).unwrap();
    for (var, want) in v.iter().zip(&want) {
        assert_eq!(bits(tape.grad(*var).unwrap()), bits(want), "checkpointed");
    }
}

#[test]
fn a_node_used_through_two_broadcast_shapes_sums_its_gradients() {
    let be = cpu();
    let (x, p, q) = (
        tensor(&be, 21, &[3, 5]),
        tensor(&be, 22, &[5]),
        tensor(&be, 23, &[3, 1]),
    );
    let m1 = be.broadcast_mul_forward(&x, &p).unwrap();
    let ones = ones_like(&be, &m1);
    let (gx1, gp) = be.broadcast_mul_backward(&x, &p, &ones).unwrap();
    let (gx2, gq) = be.broadcast_mul_backward(&x, &q, &ones).unwrap();
    let gx = be.residual_add_forward(&gx1, &gx2).unwrap();
    let mut tape = Tape::new(be.clone());
    let [vx, vp, vq] = [&x, &p, &q].map(|t| tape.leaf(t.clone()).unwrap());
    let a = tape.broadcast_mul(vx, vp).unwrap();
    let c = tape.broadcast_mul(vx, vq).unwrap();
    let z = tape.broadcast_add(a, c).unwrap();
    tape.backward(z).unwrap();
    assert_eq!(bits(tape.grad(vx).unwrap()), bits(&gx));
    assert_eq!(bits(tape.grad(vp).unwrap()), bits(&gp));
    assert_eq!(bits(tape.grad(vq).unwrap()), bits(&gq));
}

#[test]
fn a_scale_passes_its_gradient_through_times_the_scale() {
    let be = cpu();
    let x = tensor(&be, 31, &[4, 3]);
    let mut tape = Tape::new(be.clone());
    let vx = tape.leaf(x.clone()).unwrap();
    let y = tape.scale(vx, 0.37).unwrap();
    let z = tape.scale(y, -2.5).unwrap();
    tape.backward(z).unwrap();
    let ones = ones_like(&be, &x);
    let want = be
        .scale_forward(&be.scale_forward(&ones, -2.5).unwrap(), 0.37)
        .unwrap();
    assert_eq!(bits(tape.grad(vx).unwrap()), bits(&want));
}

// ---- gradcheck against an f64 forward ----

/// The block in `f64`, for the flat vector of every leaf in [`NAMES`] order.
fn block_f64(p: &[f64]) -> f64 {
    let sizes = [
        2 * TOKENS * IN,
        OUT * IN,
        OUT,
        OUT,
        OUT,
        TOKENS,
        OUT,
        2 * TOKENS * OUT,
    ];
    let mut at = 0;
    let [x, w, b, g, beta, s, pb, r]: [&[f64]; 8] = std::array::from_fn(|i| {
        let part = &p[at..at + sizes[i]];
        at += sizes[i];
        part
    });
    let rows = 2 * TOKENS;
    let mut total = 0.0;
    for row in 0..rows {
        let token = row % TOKENS;
        // linear with bias, then the exact GELU.
        let h: Vec<f64> = (0..OUT)
            .map(|o| {
                let dot: f64 = (0..IN).map(|i| x[row * IN + i] * w[o * IN + i]).sum();
                let z = dot + b[o];
                0.5 * z * (1.0 + erf_f64(z * std::f64::consts::FRAC_1_SQRT_2))
            })
            .collect();
        // LayerNorm over the row.
        let mean = h.iter().sum::<f64>() / OUT as f64;
        let var = h.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / OUT as f64;
        let rstd = 1.0 / (var + f64::from(EPS)).sqrt();
        // Scale by the token, add the bias row, scale; softmax; weight by r.
        let logits: Vec<f64> = (0..OUT)
            .map(|o| ((h[o] - mean) * rstd * g[o] + beta[o]) * s[token] + pb[o])
            .map(|v| v * f64::from(SCALE))
            .collect();
        let max = logits.iter().fold(f64::NEG_INFINITY, |m, v| m.max(*v));
        let e: Vec<f64> = logits.iter().map(|v| (v - max).exp()).collect();
        let sum: f64 = e.iter().sum();
        total += (0..OUT).map(|o| e[o] / sum * r[row * OUT + o]).sum::<f64>();
    }
    total
}

#[test]
fn the_tapes_encoder_gradients_match_central_differences() {
    let be = cpu();
    let k = block(&be);
    let mut tape = Tape::new(be.clone());
    let v = leaves(&mut tape, &k);
    let z = net(&mut tape, &v);
    tape.backward(z).unwrap();
    let flat: Vec<f64> = k
        .all()
        .iter()
        .flat_map(|t| t.to_f32_vec().unwrap())
        .map(f64::from)
        .collect();
    let numeric = central_diff(&flat, 1e-6, |p| Ok(block_f64(p))).unwrap();
    let mut at = 0;
    for (var, name) in v.iter().zip(NAMES) {
        let got = tape.grad(*var).unwrap().to_f32_vec().unwrap();
        gradients_match(&got, &numeric[at..at + got.len()], 2e-4, 2e-3)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        at += got.len();
    }
}

// ---- refusals ----

#[test]
fn a_record_the_backend_refuses_pushes_nothing() {
    let be = cpu();
    let mut tape = Tape::new(be.clone());
    let a = tape.leaf(tensor(&be, 41, &[2, 3])).unwrap();
    let b = tape.leaf(tensor(&be, 42, &[4])).unwrap();
    let w = tape.leaf(tensor(&be, 43, &[7])).unwrap();
    let before = |tape: &Tape<CpuBackend>| tape.value(Var(3)).is_err();
    assert!(before(&tape));
    let shape = |r: Result<Var, OjasError>, op: &str| match r {
        Err(OjasError::Shape { op: named, .. }) => assert_eq!(named, op),
        other => panic!("{op}: expected Shape, got {other:?}"),
    };
    shape(tape.broadcast_add(a, b), "broadcast_add_forward");
    shape(tape.broadcast_mul(a, b), "broadcast_mul_forward");
    shape(tape.layer_norm(a, w, w, EPS), "layer_norm_forward");
    shape(
        tape.channel_layer_norm(b, b, b, EPS),
        "channel_layer_norm_forward",
    );
    shape(tape.linear_bias(a, a, w), "linear_bias_forward");
    match tape.scale(a, f32::NAN) {
        Err(OjasError::NonFinite { op }) => assert_eq!(op, "scale_forward"),
        other => panic!("expected NonFinite, got {other:?}"),
    }
    match tape.layer_norm(a, b, b, f32::INFINITY) {
        Err(OjasError::NonFinite { op }) => assert_eq!(op, "layer_norm_forward"),
        other => panic!("expected NonFinite, got {other:?}"),
    }
    assert!(before(&tape), "a refused record pushed a node");
    // A variable that is not on this tape.
    let ghost = Var(99);
    for r in [
        tape.gelu_erf(ghost),
        tape.softmax(ghost),
        tape.scale(ghost, 2.0),
        tape.broadcast_add(a, ghost),
        tape.broadcast_mul(ghost, a),
        tape.layer_norm(a, ghost, b, EPS),
        tape.channel_layer_norm(a, b, ghost, EPS),
        tape.linear_bias(a, b, ghost),
    ] {
        match r {
            Err(OjasError::OutOfRange { op, .. }) => assert!(op.starts_with("Tape::"), "{op}"),
            other => panic!("expected OutOfRange, got {other:?}"),
        }
    }
    assert!(before(&tape));
}

//! `Tape::checkpoint`: a segment keeps its outputs and records through the
//! forward and is recomputed from those records by the backward walk.
//! Gradients must equal the direct recording bit for bit, the forward must
//! hold less, and every refusal must leave the tape as it was.

use ojas_autograd::{Tape, Var};
use ojas_core::{Backend, Budget, CeChunk, Numerics, OjasError, Tensor};
use ojas_cpu::CpuBackend;

const ROWS: usize = 24;
const DIM: usize = 8;
const HIDDEN: usize = 16;
const LAYERS: usize = 4;
const EPS: f32 = 1e-5;

fn cpu(numerics: Numerics) -> CpuBackend {
    CpuBackend::new(Budget::new(1 << 26)).with_numerics(numerics)
}

/// Deterministic values in `[lo, hi)`.
fn vals(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            lo + (hi - lo) * ((s >> 40) as f32 / (1u64 << 24) as f32)
        })
        .collect()
}

fn bits(t: &Tensor) -> Vec<u32> {
    t.to_f32_vec()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn leaf<B: Backend>(t: &mut Tape<B>, seed: u64, shape: &[usize], lo: f32, hi: f32) -> Var {
    let n = shape.iter().product();
    let budget = t.backend().budget().clone();
    t.leaf(Tensor::from_f32(&vals(seed, n, lo, hi), shape, &budget).unwrap())
        .unwrap()
}

/// A pre-norm SwiGLU residual layer over `x` with `p = [norm, gate, up,
/// down]`. Returns the new stream and, as a second output, the gate
/// projection.
fn layer<B: Backend>(t: &mut Tape<B>, x: Var, p: [Var; 4]) -> Result<Vec<Var>, OjasError> {
    let h = t.rms_norm(x, p[0], EPS)?;
    let a = t.linear(h, p[1])?;
    let s = t.silu(a)?;
    let u = t.linear(h, p[2])?;
    let m = t.mul(s, u)?;
    let d = t.linear(m, p[3])?;
    let y = t.add(x, d)?;
    Ok(vec![y, a])
}

struct Model {
    x: Var,
    params: Vec<[Var; 4]>,
}

impl Model {
    fn new<B: Backend>(t: &mut Tape<B>) -> Self {
        let x = leaf(t, 1, &[ROWS, DIM], -1.0, 1.0);
        let params = (0..LAYERS as u64)
            .map(|l| {
                [
                    leaf(t, 10 + 4 * l, &[DIM], 0.5, 1.5),
                    leaf(t, 11 + 4 * l, &[HIDDEN, DIM], -0.4, 0.4),
                    leaf(t, 12 + 4 * l, &[HIDDEN, DIM], -0.4, 0.4),
                    leaf(t, 13 + 4 * l, &[DIM, HIDDEN], -0.3, 0.3),
                ]
            })
            .collect();
        Self { x, params }
    }

    fn leaves(&self) -> Vec<Var> {
        let mut v = vec![self.x];
        for p in &self.params {
            v.extend_from_slice(p);
        }
        v
    }

    /// Every layer, then `stream @ gate0_w^T * gate0`: the loss reads layer
    /// 0's second output too, so a two-output segment gets a gradient on
    /// both, and layer 0's gate weight is used inside and outside its
    /// segment.
    fn forward<B: Backend>(&self, t: &mut Tape<B>, checkpoint: bool) -> Var {
        let mut x = self.x;
        let mut first_gate = None;
        for (l, &p) in self.params.iter().enumerate() {
            let outs = if checkpoint {
                t.checkpoint(|t| layer(t, x, p)).unwrap()
            } else {
                layer(t, x, p).unwrap()
            };
            x = outs[0];
            if l == 0 {
                first_gate = Some(outs[1]);
            }
        }
        let up = t.linear(x, self.params[0][1]).unwrap();
        t.mul(up, first_gate.unwrap()).unwrap()
    }
}

fn grads(backend: CpuBackend, checkpoint: bool, seed: f32) -> Vec<Vec<u32>> {
    let mut t = Tape::new(backend);
    let m = Model::new(&mut t);
    let root = m.forward(&mut t, checkpoint);
    t.backward_seeded(root, seed).unwrap();
    m.leaves()
        .into_iter()
        .map(|v| bits(t.grad(v).expect("every leaf gets a gradient")))
        .collect()
}

#[test]
fn checkpointed_gradients_equal_the_direct_recording_bit_for_bit() {
    for numerics in [Numerics::Exact, Numerics::Fast] {
        for seed in [1.0, 0.25] {
            let direct = grads(cpu(numerics), false, seed);
            let checkpointed = grads(cpu(numerics), true, seed);
            assert_eq!(direct.len(), 1 + 4 * LAYERS);
            for (i, (d, c)) in direct.iter().zip(&checkpointed).enumerate() {
                assert_eq!(d, c, "{numerics:?}, seed {seed}: leaf {i}");
            }
        }
    }
}

#[test]
fn a_second_backward_gives_the_same_bits() {
    let mut t = Tape::new(cpu(Numerics::Fast));
    let m = Model::new(&mut t);
    let root = m.forward(&mut t, true);
    let read = |t: &Tape| -> Vec<Vec<u32>> {
        m.leaves()
            .iter()
            .map(|&v| bits(t.grad(v).unwrap()))
            .collect()
    };
    t.backward(root).unwrap();
    let first = read(&t);
    t.backward(root).unwrap();
    assert_eq!(first, read(&t));
}

/// The forward keeps each segment's outputs only, so it holds less than the
/// direct recording, and the backward peak is lower too: one segment's
/// values are live at a time.
#[test]
fn the_forward_releases_the_segment_and_the_peak_drops() {
    let measure = |checkpoint: bool| {
        let budget = Budget::new(1 << 26);
        let mut t = Tape::new(CpuBackend::new(budget.clone()));
        let m = Model::new(&mut t);
        let before = budget.live_bytes().unwrap();
        budget.reset_peak();
        let root = m.forward(&mut t, checkpoint);
        let after_forward = budget.live_bytes().unwrap() - before;
        t.backward(root).unwrap();
        (after_forward, budget.peak_bytes() - before)
    };
    let (direct_live, direct_peak) = measure(false);
    let (ckpt_live, ckpt_peak) = measure(true);
    // A layer recorded directly keeps h, d and y ([ROWS, DIM]) and a, s, u
    // and m ([ROWS, HIDDEN]). Checkpointed it keeps its two outputs, y and a.
    let rows = |n: usize| (ROWS * n * 4) as u64;
    let dropped = (rows(DIM) * 2 + rows(HIDDEN) * 3) * LAYERS as u64;
    assert_eq!(
        direct_live - ckpt_live,
        dropped,
        "forward: direct {direct_live} B, checkpointed {ckpt_live} B"
    );
    assert!(
        ckpt_peak < direct_peak,
        "peak: direct {direct_peak} B, checkpointed {ckpt_peak} B"
    );
}

/// An output with no consumer is fine: only the outputs with a gradient are
/// seeded. A segment none of whose outputs reaches the root is not
/// recomputed, and its inputs get nothing from it.
#[test]
fn unused_outputs_and_unused_segments_match_the_direct_recording() {
    let run = |checkpoint: bool| {
        let mut t = Tape::new(cpu(Numerics::Exact));
        let m = Model::new(&mut t);
        let (p, q) = (m.params[0], m.params[1]);
        let seg = |t: &mut Tape, p| {
            if checkpoint {
                t.checkpoint(|t| layer(t, m.x, p)).unwrap()
            } else {
                layer(t, m.x, p).unwrap()
            }
        };
        let used = seg(&mut t, p);
        seg(&mut t, q);
        // Only the second output (the gate projection) reaches the root.
        let root = t.silu(used[1]).unwrap();
        t.backward(root).unwrap();
        for v in q {
            assert!(
                t.grad(v).is_none(),
                "an unused segment's input got a gradient"
            );
        }
        assert!(
            t.grad(p[3]).is_none(),
            "the down projection does not reach the gate"
        );
        [m.x, p[0], p[1]].map(|v| bits(t.grad(v).unwrap()))
    };
    assert_eq!(run(true), run(false));
}

/// Every op kind whose forward returns something beside its value (GDN
/// state checkpoints, the fused cross-entropy's gradients) has it dropped
/// with the segment and recomputed by the replay, with the same bits.
#[test]
fn gdn_and_the_fused_loss_inside_a_segment_match_the_direct_recording() {
    const B: usize = 1;
    const T: usize = 70;
    const H: usize = 2;
    const DK: usize = 4;
    const DV: usize = 3;
    let run = |checkpoint: bool| {
        let mut t = Tape::new(cpu(Numerics::Exact));
        let q = leaf(&mut t, 1, &[B, T, H, DK], -1.0, 1.0);
        let k = leaf(&mut t, 2, &[B, T, H, DK], -1.0, 1.0);
        let v = leaf(&mut t, 3, &[B, T, H, DV], -1.0, 1.0);
        let g = leaf(&mut t, 4, &[B, T, H], -1.0, -0.05);
        let beta = leaf(&mut t, 5, &[B, T, H], 0.1, 0.9);
        let w = leaf(&mut t, 6, &[11, H * DV], -0.5, 0.5);
        let budget = t.backend().budget().clone();
        let targets: Vec<u32> = (0..(B * T) as u32).map(|i| (i * 7) % 11).collect();
        let targets = Tensor::from_u32(&targets, &[B * T], &budget).unwrap();
        let chunk = CeChunk { rows: 16, cols: 4 };
        let body = |t: &mut Tape| -> Result<Vec<Var>, OjasError> {
            let o = t.chunked_gdn(q, k, v, g, beta)?;
            let o = t.reshape(o, &[B * T, H * DV])?;
            let loss = t.linear_cross_entropy(o, w, targets.clone(), None, chunk)?;
            Ok(vec![loss])
        };
        let loss = if checkpoint {
            t.checkpoint(body).unwrap()[0]
        } else {
            body(&mut t).unwrap()[0]
        };
        t.backward_seeded(loss, 0.5).unwrap();
        [q, k, v, g, beta, w].map(|x| bits(t.grad(x).unwrap()))
    };
    assert_eq!(run(true), run(false));
}

/// Attention, the gate, value residual, RoPE and permute inside a segment,
/// the ops of the nanolab block, give the direct recording's bits.
#[test]
fn attention_ops_inside_a_segment_match_the_direct_recording() {
    const B: usize = 2;
    const T: usize = 6;
    const H: usize = 2;
    const D: usize = 4;
    let run = |checkpoint: bool, numerics: Numerics| {
        let mut t = Tape::new(cpu(numerics));
        let q = leaf(&mut t, 1, &[B, T, H, D], -1.0, 1.0);
        let k = leaf(&mut t, 2, &[B, T, H, D], -1.0, 1.0);
        let v = leaf(&mut t, 3, &[B, T, H, D], -1.0, 1.0);
        let v0 = leaf(&mut t, 4, &[B, T, H, D], -1.0, 1.0);
        let lambda = leaf(&mut t, 5, &[1], -1.0, 1.0);
        let x = leaf(&mut t, 6, &[B, T, H * D], -1.0, 1.0);
        let gw = leaf(&mut t, 7, &[H, H * D], -0.5, 0.5);
        let gb = leaf(&mut t, 8, &[H], -0.5, 0.5);
        let budget = t.backend().budget().clone();
        let table = |seed| Tensor::from_f32(&vals(seed, T * D, -1.0, 1.0), &[T, D], &budget);
        let (cos, sin) = (table(9).unwrap(), table(10).unwrap());
        let swap = [0usize, 2, 1, 3];
        let body = |t: &mut Tape| -> Result<Vec<Var>, OjasError> {
            let qr = t.rope(q, cos.clone(), sin.clone())?;
            let kr = t.rope(k, cos.clone(), sin.clone())?;
            let vb = t.value_residual(v, v0, lambda)?;
            let (qh, kh, vh) = (
                t.permute(qr, &swap)?,
                t.permute(kr, &swap)?,
                t.permute(vb, &swap)?,
            );
            let y = t.causal_sdpa(qh, kh, vh, None)?;
            let y = t.permute(y, &swap)?;
            let y = t.per_head_gate(x, gw, gb, y)?;
            Ok(vec![y, vb])
        };
        let outs = if checkpoint {
            t.checkpoint(body).unwrap()
        } else {
            body(&mut t).unwrap()
        };
        let root = t.mul(outs[0], outs[0]).unwrap();
        let other = t.reshape(outs[1], &[B, T, H, D]).unwrap();
        let root = t.mul(root, other).unwrap();
        t.backward(root).unwrap();
        [q, k, v, v0, lambda, x, gw, gb].map(|x| bits(t.grad(x).unwrap()))
    };
    for numerics in [Numerics::Exact, Numerics::Fast] {
        assert_eq!(run(true, numerics), run(false, numerics), "{numerics:?}");
    }
}

#[test]
fn refusals_leave_the_tape_as_it_was() {
    let budget = Budget::new(1 << 26);
    let mut t = Tape::new(CpuBackend::new(budget.clone()));
    let m = Model::new(&mut t);
    let (x, p) = (m.x, m.params[0]);
    let live = budget.live_bytes().unwrap();
    let next = Var(1 + 4 * LAYERS);

    type Body = Box<dyn FnOnce(&mut Tape) -> Result<Vec<Var>, OjasError>>;
    let cases: Vec<(&str, Body)> = vec![
        (
            "no outputs",
            Box::new(move |t| layer(t, x, p).map(|_| Vec::new())),
        ),
        (
            "an input as output",
            Box::new(move |t| layer(t, x, p).map(|_| vec![x])),
        ),
        (
            "an output twice",
            Box::new(move |t| layer(t, x, p).map(|o| vec![o[0], o[0]])),
        ),
        (
            "a leaf inside",
            Box::new(move |t| {
                let w = leaf(t, 99, &[DIM], 0.5, 1.5);
                Ok(vec![t.rms_norm(x, w, EPS)?])
            }),
        ),
        (
            "an error from the body",
            Box::new(move |t| {
                layer(t, x, p)?;
                // A rank-1 weight is refused by linear_forward.
                Ok(vec![t.linear(x, p[0])?])
            }),
        ),
        (
            "a nested checkpoint",
            Box::new(move |t| {
                let inner = t.checkpoint(|t| layer(t, x, p))?;
                Ok(vec![t.silu(inner[0])?])
            }),
        ),
        ("a var off the tape", Box::new(|_| Ok(vec![Var(10_000)]))),
    ];
    for (name, body) in cases {
        let r = t.checkpoint(body);
        assert!(r.is_err(), "{name}: accepted");
        assert_eq!(budget.live_bytes().unwrap(), live, "{name}: values kept");
        let probe = leaf(&mut t, 0, &[1], 0.0, 1.0);
        assert_eq!(probe, next, "{name}: nodes kept");
        t.clear();
        assert_eq!(Model::new(&mut t).x, x);
    }
}

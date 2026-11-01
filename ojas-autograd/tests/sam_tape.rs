//! `Tape::group_norm` and `Tape::interpolate_bilinear`: a small SAM-style
//! neck (GroupNorm, an upsample by bilinear resampling, a downsample with the
//! other `align_corners`) recorded on the tape must give the gradients of the
//! backend's backward calls chained by hand, bit for bit, recorded directly,
//! inside a checkpointed segment and with a frozen weight. The values of those
//! backwards are checked against `f64` references and torch in
//! `ojas-cpu/tests/spatial_backward.rs` and `sam_backward_golden.rs`.
//!
//! A node read by two bilinear records sums their gradients, and a record the
//! backend refuses pushes nothing.

use ojas_autograd::{Tape, Var};
use ojas_core::{Backend, Budget, Numerics, OjasError, Tensor};
use ojas_cpu::CpuBackend;

const N: usize = 2;
const C: usize = 8;
const GROUPS: usize = 2;
const EPS: f32 = 1e-5;
const SIZE: usize = 5;
const UP: [usize; 2] = [8, 7];
const DOWN: [usize; 2] = [4, 3];

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

struct Neck {
    x: Tensor,
    w: Tensor,
    b: Tensor,
}

const NAMES: [&str; 3] = ["x", "w", "b"];

fn neck(be: &CpuBackend) -> Neck {
    Neck {
        x: tensor(be, 1, &[N, C, SIZE, SIZE]),
        w: near_one(be, 2, &[C]),
        b: tensor(be, 3, &[C]),
    }
}

/// GroupNorm -> bilinear up (`align_corners` off) -> bilinear down (on).
fn net(tape: &mut Tape<CpuBackend>, v: [Var; 3]) -> Var {
    let [x, w, b] = v;
    let h = tape.group_norm(x, w, b, GROUPS, EPS).unwrap();
    let h = tape.interpolate_bilinear(h, UP, false).unwrap();
    tape.interpolate_bilinear(h, DOWN, true).unwrap()
}

/// The same chain through the backend: `(dx, dw, db)` for a seed of ones.
fn by_hand(be: &CpuBackend, k: &Neck) -> [Tensor; 3] {
    let h1 = be
        .group_norm_forward(&k.x, &k.w, &k.b, GROUPS, EPS)
        .unwrap();
    let h2 = be.interpolate_bilinear_forward(&h1, UP, false).unwrap();
    let y = be.interpolate_bilinear_forward(&h2, DOWN, true).unwrap();
    let g2 = be
        .interpolate_bilinear_backward(&h2, &ones_like(be, &y), true)
        .unwrap();
    let g1 = be.interpolate_bilinear_backward(&h1, &g2, false).unwrap();
    let gn = be
        .group_norm_backward(&k.x, &k.w, &k.b, &g1, GROUPS, EPS)
        .unwrap();
    [gn.input, gn.weight, gn.bias]
}

fn leaves(tape: &mut Tape<CpuBackend>, k: &Neck) -> [Var; 3] {
    [&k.x, &k.w, &k.b].map(|t| tape.leaf(t.clone()).unwrap())
}

#[test]
fn the_tape_chains_group_norm_and_the_bilinear_backwards() {
    let be = cpu();
    let k = neck(&be);
    let want = by_hand(&be, &k);
    let mut tape = Tape::new(be.clone());
    let v = leaves(&mut tape, &k);
    let y = net(&mut tape, v);
    assert_eq!(tape.value(y).unwrap().shape(), &[N, C, DOWN[0], DOWN[1]]);
    tape.backward(y).unwrap();
    for ((var, want), name) in v.iter().zip(&want).zip(NAMES) {
        let got = tape
            .grad(*var)
            .unwrap_or_else(|| panic!("{name} takes a gradient"));
        assert_eq!(got.shape(), want.shape(), "{name}");
        assert_eq!(bits(got), bits(want), "{name}");
    }
}

#[test]
fn a_checkpointed_group_norm_and_resample_segment_gives_the_same_bits() {
    let be = cpu();
    let k = neck(&be);
    let want = by_hand(&be, &k);
    let mut tape = Tape::new(be.clone());
    let v = leaves(&mut tape, &k);
    let out = tape.checkpoint(|t| Ok(vec![net(t, v)])).unwrap();
    tape.backward(out[0]).unwrap();
    for ((var, want), name) in v.iter().zip(&want).zip(NAMES) {
        assert_eq!(bits(tape.grad(*var).unwrap()), bits(want), "{name}");
    }
}

#[test]
fn a_frozen_image_and_a_frozen_weight_take_no_gradient_and_the_rest_do() {
    let be = cpu();
    let k = neck(&be);
    let want = by_hand(&be, &k);
    // The image frozen: the weight and bias still take theirs.
    let mut tape = Tape::new(be.clone());
    let x = tape.frozen_leaf(k.x.clone()).unwrap();
    let w = tape.leaf(k.w.clone()).unwrap();
    let b = tape.leaf(k.b.clone()).unwrap();
    let y = net(&mut tape, [x, w, b]);
    tape.backward(y).unwrap();
    assert!(tape.grad(x).is_none());
    assert_eq!(bits(tape.grad(w).unwrap()), bits(&want[1]));
    assert_eq!(bits(tape.grad(b).unwrap()), bits(&want[2]));
    // The weight frozen: the image and bias still take theirs.
    let mut tape = Tape::new(be.clone());
    let x = tape.leaf(k.x.clone()).unwrap();
    let w = tape.frozen_leaf(k.w.clone()).unwrap();
    let b = tape.leaf(k.b.clone()).unwrap();
    let y = net(&mut tape, [x, w, b]);
    tape.backward(y).unwrap();
    assert!(tape.grad(w).is_none());
    assert_eq!(bits(tape.grad(x).unwrap()), bits(&want[0]));
    assert_eq!(bits(tape.grad(b).unwrap()), bits(&want[2]));
}

#[test]
fn a_node_read_by_two_resamples_sums_their_gradients() {
    let be = cpu();
    let x = tensor(&be, 7, &[1, 3, 4, 5]);
    let mut tape = Tape::new(be.clone());
    let v = tape.leaf(x.clone()).unwrap();
    let a = tape.interpolate_bilinear(v, [6, 7], false).unwrap();
    let b = tape.interpolate_bilinear(v, [6, 7], true).unwrap();
    let y = tape.add(a, b).unwrap();
    tape.backward(y).unwrap();
    let ones = ones_like(&be, tape.value(y).unwrap());
    let ga = be.interpolate_bilinear_backward(&x, &ones, false).unwrap();
    let gb = be.interpolate_bilinear_backward(&x, &ones, true).unwrap();
    let want: Vec<u32> = ga
        .to_f32_vec()
        .unwrap()
        .iter()
        .zip(gb.to_f32_vec().unwrap())
        .map(|(p, q)| (p + q).to_bits())
        .collect();
    assert_eq!(bits(tape.grad(v).unwrap()), want);
}

#[test]
fn a_record_the_backend_refuses_pushes_nothing() {
    let be = cpu();
    let mut tape = Tape::new(be.clone());
    let x = tape.leaf(tensor(&be, 41, &[1, 6, 3, 3])).unwrap();
    let w = tape.leaf(tensor(&be, 42, &[6])).unwrap();
    let b = tape.leaf(tensor(&be, 43, &[6])).unwrap();
    let flat = tape.leaf(tensor(&be, 44, &[2, 3])).unwrap();
    let not_pushed = |tape: &Tape<CpuBackend>| tape.value(Var(4)).is_err();
    assert!(not_pushed(&tape));
    // 4 groups do not divide 6 channels.
    match tape.group_norm(x, w, b, 4, EPS) {
        Err(OjasError::Shape { op, .. }) => assert_eq!(op, "group_norm_forward"),
        other => panic!("expected Shape, got {other:?}"),
    }
    match tape.group_norm(x, w, b, 0, EPS) {
        Err(OjasError::OutOfRange { op, .. }) => assert_eq!(op, "group_norm_forward"),
        other => panic!("expected OutOfRange, got {other:?}"),
    }
    match tape.group_norm(x, w, b, 2, f32::NAN) {
        Err(OjasError::NonFinite { op }) => assert_eq!(op, "group_norm_forward"),
        other => panic!("expected NonFinite, got {other:?}"),
    }
    // A rank-2 value is not an image; an empty output size is no size.
    for r in [
        tape.interpolate_bilinear(flat, [4, 4], false),
        tape.interpolate_bilinear(x, [0, 4], false),
    ] {
        match r {
            Err(OjasError::Shape { op, .. }) => assert_eq!(op, "interpolate_bilinear_forward"),
            other => panic!("expected Shape, got {other:?}"),
        }
    }
    assert!(not_pushed(&tape), "a refused record pushed a node");
    // A variable that is not on this tape.
    let ghost = Var(99);
    for (r, name) in [
        (tape.group_norm(x, ghost, b, 2, EPS), "Tape::group_norm"),
        (tape.group_norm(ghost, w, b, 2, EPS), "Tape::group_norm"),
        (
            tape.interpolate_bilinear(ghost, [4, 4], true),
            "Tape::interpolate_bilinear",
        ),
        (tape.relu(ghost), "Tape::relu"),
        (tape.slice(ghost, 1, 0, 1), "Tape::slice"),
        (tape.concat(&[ghost], 0), "Tape::concat"),
    ] {
        match r {
            Err(OjasError::OutOfRange { op, .. }) => assert_eq!(op, name),
            other => panic!("expected OutOfRange, got {other:?}"),
        }
    }
    assert!(not_pushed(&tape));
}

#[test]
fn the_tape_chains_relu_backward() {
    let be = cpu();
    let x = tensor(&be, 101, &[2, 4, 3]);
    let mut tape = Tape::new(be.clone());
    let vx = tape.leaf(x.clone()).unwrap();
    let vy = tape.relu(vx).unwrap();
    tape.backward(vy).unwrap();
    let seed = ones_like(&be, &x);
    let want = be.relu_backward(&x, &seed).unwrap();
    assert_eq!(bits(tape.grad(vx).unwrap()), bits(&want));
}

#[test]
fn the_tape_chains_slice_and_concat_backwards() {
    let be = cpu();
    let a = tensor(&be, 201, &[2, 3, 4]);
    let b = tensor(&be, 202, &[2, 5, 4]);
    let mut tape = Tape::new(be.clone());
    let va = tape.leaf(a.clone()).unwrap();
    let vb = tape.leaf(b.clone()).unwrap();
    let cat = tape.concat(&[va, vb], 1).unwrap();
    assert_eq!(tape.value(cat).unwrap().shape(), &[2, 8, 4]);
    let sl = tape.slice(cat, 1, 1, 4).unwrap();
    assert_eq!(tape.value(sl).unwrap().shape(), &[2, 4, 4]);
    tape.backward(sl).unwrap();

    let ones_sl = ones_like(&be, tape.value(sl).unwrap());
    let g_cat = be.slice_backward(&ones_sl, &[2, 8, 4], 1, 1).unwrap();
    let want_a = be.slice(&g_cat, 1, 0, 3).unwrap();
    let want_b = be.slice(&g_cat, 1, 3, 5).unwrap();

    assert_eq!(bits(tape.grad(va).unwrap()), bits(&want_a));
    assert_eq!(bits(tape.grad(vb).unwrap()), bits(&want_b));
}

#[test]
fn the_tape_chains_rope_interleaved_backward() {
    let be = cpu();
    let x = tensor(&be, 301, &[2, 4, 3, 6]);
    let cos = tensor(&be, 302, &[4, 3]);
    let sin = tensor(&be, 303, &[4, 3]);
    let mut tape = Tape::new(be.clone());
    let vx = tape.leaf(x.clone()).unwrap();
    let vy = tape.rope_interleaved(vx, cos.clone(), sin.clone()).unwrap();
    tape.backward(vy).unwrap();

    let seed = ones_like(&be, tape.value(vy).unwrap());
    let want = be.rope_interleaved_backward(&seed, &cos, &sin).unwrap();
    assert_eq!(bits(tape.grad(vx).unwrap()), bits(&want));
}

#[test]
fn adversarial_vision_tape_all_ops_chaining_and_checkpointing() {
    let be = cpu();
    let x = tensor(&be, 501, &[1, 4, 2, 4]);
    let cos = tensor(&be, 502, &[4, 2]);
    let sin = tensor(&be, 503, &[4, 2]);
    let gn_w = near_one(&be, 504, &[4]);
    let gn_b = tensor(&be, 505, &[4]);
    let side = tensor(&be, 506, &[1, 2, 2, 4]);

    let run_net = |tape: &mut Tape<CpuBackend>, vx: Var, vw: Var, vb: Var, vside: Var| -> Var {
        let r = tape.rope_interleaved(vx, cos.clone(), sin.clone()).unwrap();
        let gn = tape.group_norm(r, vw, vb, 2, 1e-5).unwrap();
        let act = tape.relu(gn).unwrap();
        let sl = tape.slice(act, 1, 0, 2).unwrap();
        let cat = tape.concat(&[sl, vside], 1).unwrap();
        tape.interpolate_bilinear(cat, [4, 6], false).unwrap()
    };

    // By hand backward:
    let r_fwd = be.rope_interleaved_forward(&x, &cos, &sin).unwrap();
    let gn_fwd = be
        .group_norm_forward(&r_fwd, &gn_w, &gn_b, 2, 1e-5)
        .unwrap();
    let act_fwd = be.relu_forward(&gn_fwd).unwrap();
    let sl_fwd = be.slice(&act_fwd, 1, 0, 2).unwrap();
    let cat_fwd = be.concat(&[&sl_fwd, &side], 1).unwrap();
    let out_fwd = be
        .interpolate_bilinear_forward(&cat_fwd, [4, 6], false)
        .unwrap();

    let seed = ones_like(&be, &out_fwd);
    let g_cat = be
        .interpolate_bilinear_backward(&cat_fwd, &seed, false)
        .unwrap();
    let g_sl = be.slice(&g_cat, 1, 0, 2).unwrap();
    let g_side_want = be.slice(&g_cat, 1, 2, 2).unwrap();
    let g_act = be.slice_backward(&g_sl, &[1, 4, 2, 4], 1, 0).unwrap();
    let g_gn = be.relu_backward(&gn_fwd, &g_act).unwrap();
    let gn_grads = be
        .group_norm_backward(&r_fwd, &gn_w, &gn_b, &g_gn, 2, 1e-5)
        .unwrap();
    let g_x_want = be
        .rope_interleaved_backward(&gn_grads.input, &cos, &sin)
        .unwrap();
    let g_w_want = gn_grads.weight;
    let g_b_want = gn_grads.bias;

    // Normal tape run:
    let mut tape = Tape::new(be.clone());
    let vx = tape.leaf(x.clone()).unwrap();
    let vw = tape.leaf(gn_w.clone()).unwrap();
    let vb = tape.leaf(gn_b.clone()).unwrap();
    let vside = tape.leaf(side.clone()).unwrap();
    let out = run_net(&mut tape, vx, vw, vb, vside);
    tape.backward(out).unwrap();

    assert_eq!(bits(tape.grad(vx).unwrap()), bits(&g_x_want));
    assert_eq!(bits(tape.grad(vw).unwrap()), bits(&g_w_want));
    assert_eq!(bits(tape.grad(vb).unwrap()), bits(&g_b_want));
    assert_eq!(bits(tape.grad(vside).unwrap()), bits(&g_side_want));

    // Checkpointed segment run:
    let mut tape_cp = Tape::new(be.clone());
    let vx = tape_cp.leaf(x.clone()).unwrap();
    let vw = tape_cp.leaf(gn_w.clone()).unwrap();
    let vb = tape_cp.leaf(gn_b.clone()).unwrap();
    let vside = tape_cp.leaf(side.clone()).unwrap();
    let out_cp = tape_cp
        .checkpoint(|t| Ok(vec![run_net(t, vx, vw, vb, vside)]))
        .unwrap();
    tape_cp.backward(out_cp[0]).unwrap();

    assert_eq!(bits(tape_cp.grad(vx).unwrap()), bits(&g_x_want));
    assert_eq!(bits(tape_cp.grad(vw).unwrap()), bits(&g_w_want));
    assert_eq!(bits(tape_cp.grad(vb).unwrap()), bits(&g_b_want));
    assert_eq!(bits(tape_cp.grad(vside).unwrap()), bits(&g_side_want));

    // Frozen leaf run (e.g. frozen weight and frozen side):
    let mut tape_frz = Tape::new(be.clone());
    let vx = tape_frz.leaf(x.clone()).unwrap();
    let vw = tape_frz.frozen_leaf(gn_w.clone()).unwrap();
    let vb = tape_frz.leaf(gn_b.clone()).unwrap();
    let vside = tape_frz.frozen_leaf(side.clone()).unwrap();
    let out_frz = run_net(&mut tape_frz, vx, vw, vb, vside);
    tape_frz.backward(out_frz).unwrap();

    assert!(tape_frz.grad(vw).is_none());
    assert!(tape_frz.grad(vside).is_none());
    assert_eq!(bits(tape_frz.grad(vx).unwrap()), bits(&g_x_want));
    assert_eq!(bits(tape_frz.grad(vb).unwrap()), bits(&g_b_want));
}

#[test]
fn adversarial_multi_branch_fanin_summation() {
    let be = cpu();
    let x = tensor(&be, 601, &[1, 4, 4, 4]);
    let mut tape = Tape::new(be.clone());
    let vx = tape.leaf(x.clone()).unwrap();

    // Branch 1: relu
    let y1 = tape.relu(vx).unwrap();
    // Branch 2: interpolate_bilinear (align_corners = false)
    let y2 = tape.interpolate_bilinear(vx, [4, 4], false).unwrap();
    // Branch 3: interpolate_bilinear (align_corners = true)
    let y3 = tape.interpolate_bilinear(vx, [4, 4], true).unwrap();

    // Sum branches on the tape to test fan-in accumulation into vx
    let y12 = tape.add(y1, y2).unwrap();
    let y = tape.add(y12, y3).unwrap();

    // Run backwards independently through backend to compute exact expected gradients:
    let ones = ones_like(&be, &x);
    let g1 = be.relu_backward(&x, &ones).unwrap();
    let g2 = be.interpolate_bilinear_backward(&x, &ones, false).unwrap();
    let g3 = be.interpolate_bilinear_backward(&x, &ones, true).unwrap();

    let g_total: Vec<u32> = g1
        .to_f32_vec()
        .unwrap()
        .iter()
        .zip(g2.to_f32_vec().unwrap())
        .zip(g3.to_f32_vec().unwrap())
        .map(|((a, b), c)| (a + b + c).to_bits())
        .collect();

    tape.backward(y).unwrap();

    assert_eq!(bits(tape.grad(vx).unwrap()), g_total);
}

#[test]
fn adversarial_invalid_shape_rejection() {
    let be = cpu();
    let mut tape = Tape::new(be.clone());
    let x = tape.leaf(tensor(&be, 701, &[2, 4, 6, 8])).unwrap();

    // Slice: start + len > dim (start 3, len 6 on axis 1 of size 4)
    assert!(tape.slice(x, 1, 3, 6).is_err());
    // Concat: mismatched shapes along non-concat axis
    let bad_cat = tape.leaf(tensor(&be, 702, &[2, 3, 5, 8])).unwrap();
    assert!(tape.concat(&[x, bad_cat], 1).is_err());
    // RopeInterleaved: mismatched cos/sin
    let bad_cos = tensor(&be, 703, &[5, 5]);
    let bad_sin = tensor(&be, 704, &[5, 5]);
    assert!(tape.rope_interleaved(x, bad_cos, bad_sin).is_err());
}

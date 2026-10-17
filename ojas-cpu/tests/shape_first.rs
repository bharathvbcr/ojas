//! Shape first: every op runs its `ojas_core::shapes` validator before it
//! charges the budget, copies an input or scans one for NaN
//! (docs/shape-contract.md, D2 and D3).
//!
//! Each malformed call below is made three ways, and each must return
//! exactly what the validator returns when called directly (variant, op name
//! and detail), with nothing left charged:
//!
//! - (a) under `Budget::new(0)`;
//! - (b) under a cap that holds the input copies but not an output as well;
//! - (c) with a NaN in a well-formed operand, under an ample budget.

use ojas_core::{
    accumulate_grad_dims, cached_attention_dims, causal_conv1d_silu_backward_dims,
    causal_conv1d_silu_forward_dims, causal_sdpa_backward_dims, causal_sdpa_forward_dims,
    chunked_gdn_backward_dims, chunked_gdn_forward_dims, gated_rms_norm_backward_dims,
    gated_rms_norm_forward_dims, gdn_log_decay_backward_dims, gdn_log_decay_forward_dims,
    kv_cache_write_dims, linear_backward_dims, linear_ce_dims, linear_forward_dims,
    mul_backward_dims, mul_forward_dims, per_head_sigmoid_gate_backward_dims,
    per_head_sigmoid_gate_forward_dims, permute_dims, residual_add_backward_dims,
    residual_add_forward_dims, rms_norm_backward_dims, rms_norm_forward_dims,
    rms_qk_norm_backward_dims, rms_qk_norm_forward_dims, rope_half_split_backward_dims,
    rope_half_split_forward_dims, rope_partial_backward_dims, rope_partial_forward_dims,
    sigmoid_backward_dims, sigmoid_forward_dims, silu_backward_dims, silu_forward_dims,
    value_residual_blend_backward_dims, value_residual_blend_forward_dims, Backend, Budget,
    CeChunk, CeGrads, DType, GdnInputs, OjasError, Tensor,
};
use ojas_cpu::CpuBackend;

type Check = Box<dyn Fn(&[Tensor]) -> Result<(), OjasError>>;
type Run = Box<dyn Fn(&CpuBackend, &[Tensor]) -> Result<(), OjasError>>;

struct Case {
    name: String,
    operands: Vec<Tensor>,
    /// A well-formed f32 operand that (c) replaces with NaNs; `None` when
    /// the op has no such operand (one input, or a malformed f32 operand).
    nan_at: Option<usize>,
    check: Check,
    run: Run,
}

fn pool() -> Budget {
    Budget::new(1 << 24)
}

/// An f32 tensor of `shape` holding small finite values.
fn f(shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| 0.25 + (i % 7) as f32 * 0.125).collect();
    Tensor::from_f32(&data, shape, &pool()).unwrap()
}

/// A u32 tensor of `shape` holding zeros (valid token ids and targets).
fn u(shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::from_u32(&vec![0; n], shape, &pool()).unwrap()
}

/// `t`'s shape, every value NaN.
fn nan_like(t: &Tensor) -> Tensor {
    let n: usize = t.shape().iter().product();
    Tensor::from_f32(&vec![f32::NAN; n], t.shape(), &pool()).unwrap()
}

fn case<C, R>(name: &str, operands: Vec<Tensor>, nan_at: Option<usize>, check: C, run: R) -> Case
where
    C: Fn(&[Tensor]) -> Result<(), OjasError> + 'static,
    R: Fn(&CpuBackend, &[Tensor]) -> Result<(), OjasError> + 'static,
{
    if let Some(i) = nan_at {
        assert_eq!(operands[i].dtype(), DType::F32, "{name}: NaN operand");
    }
    Case {
        name: name.to_string(),
        operands,
        nan_at,
        check: Box::new(check),
        run: Box::new(run),
    }
}

const EPS: f32 = 1e-6;
const CHUNK: CeChunk = CeChunk { rows: 2, cols: 2 };

/// The gated delta rule's operands `q, k, v, g, beta, initial_state` as the
/// backends take them; the backward's `checkpoints`, `grad_output` and
/// optional final-state gradient follow at 6, 7 and 8.
fn gdn_inputs(o: &[Tensor]) -> GdnInputs<'_> {
    GdnInputs {
        q: &o[0],
        k: &o[1],
        v: &o[2],
        g: &o[3],
        beta: &o[4],
        initial_state: o.get(5),
    }
}

/// Well-formed gated-delta-rule operands (B 1, T 3, H 2, Dk 4, Dv 4) with
/// the slot `which` replaced by `bad`.
fn gdn_with(which: usize, bad: Tensor) -> Vec<Tensor> {
    let mut ops = vec![
        f(&[1, 3, 2, 4]),
        f(&[1, 3, 2, 4]),
        f(&[1, 3, 2, 4]),
        f(&[1, 3, 2]),
        f(&[1, 3, 2]),
        f(&[1, 2, 4, 4]),
    ];
    ops[which] = bad;
    ops
}

#[allow(clippy::too_many_lines)]
fn cases() -> Vec<Case> {
    let mut v = Vec::new();

    // ---- linear ---------------------------------------------------------
    for (name, x, w, nan) in [
        ("weight rank 1", f(&[3, 4]), f(&[4]), 0),
        ("weight rank 3", f(&[3, 4]), f(&[1, 2, 4]), 0),
        ("input rank 0", f(&[]), f(&[2, 4]), 1),
        ("in-features", f(&[3, 4]), f(&[2, 5]), 0),
    ] {
        v.push(case(
            &format!("linear_forward {name}"),
            vec![x, w],
            Some(nan),
            |o| linear_forward_dims(&o[0], &o[1]).map(drop),
            |b, o| b.linear_forward(&o[0], &o[1]).map(drop),
        ));
    }
    v.push(case(
        "linear_forward u32 weight after a NaN input",
        vec![f(&[3, 4]), u(&[2, 4])],
        Some(0),
        |o| linear_forward_dims(&o[0], &o[1]).map(drop),
        |b, o| b.linear_forward(&o[0], &o[1]).map(drop),
    ));
    for (name, x, w, g, nan) in [
        ("in-features", f(&[3, 4]), f(&[2, 5]), f(&[3, 2]), 2),
        ("weight rank 1", f(&[3, 4]), f(&[4]), f(&[3, 2]), 0),
        ("grad out", f(&[3, 4]), f(&[2, 4]), f(&[3, 3]), 0),
        ("grad transposed", f(&[3, 4]), f(&[2, 4]), f(&[2, 3]), 1),
    ] {
        v.push(case(
            &format!("linear_backward {name}"),
            vec![x, w, g],
            Some(nan),
            |o| linear_backward_dims(&o[0], &o[1], &o[2]).map(drop),
            |b, o| b.linear_backward(&o[0], &o[1], &o[2]).map(drop),
        ));
    }

    // ---- rms_norm and rms_qk_norm ---------------------------------------
    for (name, x, w, nan) in [
        ("input rank 0", f(&[]), f(&[1]), 1),
        ("weight dim", f(&[2, 4]), f(&[3]), 0),
        ("weight rank 2", f(&[2, 4]), f(&[1, 4]), 0),
    ] {
        v.push(case(
            &format!("rms_norm_forward {name}"),
            vec![x, w],
            Some(nan),
            |o| rms_norm_forward_dims(&o[0], &o[1], EPS).map(drop),
            |b, o| b.rms_norm_forward(&o[0], &o[1], EPS).map(drop),
        ));
    }
    for (name, x, w, g, eps, nan) in [
        ("grad shape", f(&[2, 4]), f(&[4]), f(&[2, 3]), EPS, 0),
        ("weight dim", f(&[2, 4]), f(&[3]), f(&[2, 4]), EPS, 2),
        // D8: the grad check runs before eps and the weight.
        (
            "grad before weight",
            f(&[2, 4]),
            f(&[3]),
            f(&[4, 2]),
            EPS,
            0,
        ),
        (
            "grad before eps",
            f(&[2, 4]),
            f(&[4]),
            f(&[4, 2]),
            f32::NAN,
            1,
        ),
    ] {
        v.push(case(
            &format!("rms_norm_backward {name}"),
            vec![x, w, g],
            Some(nan),
            move |o| rms_norm_backward_dims(&o[0], &o[1], &o[2], eps).map(drop),
            move |b, o| b.rms_norm_backward(&o[0], &o[1], &o[2], eps).map(drop),
        ));
    }
    for (name, q, k, qw, kw, nan) in [
        ("k weight", f(&[2, 4]), f(&[2, 4]), f(&[4]), f(&[3]), 0),
        ("q weight", f(&[2, 4]), f(&[2, 4]), f(&[5]), f(&[4]), 1),
    ] {
        v.push(case(
            &format!("rms_qk_norm_forward {name}"),
            vec![q, k, qw, kw],
            Some(nan),
            |o| rms_qk_norm_forward_dims(&o[0], &o[1], &o[2], &o[3], EPS).map(drop),
            |b, o| {
                b.rms_qk_norm_forward(&o[0], &o[1], &o[2], &o[3], EPS)
                    .map(drop)
            },
        ));
    }
    for (name, gk, kw, nan) in [
        ("k grad", f(&[4, 2]), f(&[4]), 0),
        ("k weight", f(&[2, 4]), f(&[3]), 4),
    ] {
        v.push(case(
            &format!("rms_qk_norm_backward {name}"),
            vec![f(&[2, 4]), f(&[2, 4]), f(&[4]), kw, f(&[2, 4]), gk],
            Some(nan),
            |o| rms_qk_norm_backward_dims(&o[0], &o[1], &o[2], &o[3], &o[4], &o[5], EPS).map(drop),
            |b, o| {
                b.rms_qk_norm_backward(&o[0], &o[1], &o[2], &o[3], &o[4], &o[5], EPS)
                    .map(drop)
            },
        ));
    }

    // ---- rope -----------------------------------------------------------
    for (name, x, c, s, nan) in [
        ("cos vs sin", f(&[2, 4]), f(&[2, 4]), f(&[4, 2]), 0),
        ("input rank 0", f(&[]), f(&[2]), f(&[2]), 1),
        ("odd dim", f(&[2, 3]), f(&[2, 3]), f(&[2, 3]), 1),
        ("no broadcast", f(&[2, 4]), f(&[3, 4]), f(&[3, 4]), 0),
        ("time axis", f(&[1, 2, 2, 4]), f(&[3, 4]), f(&[3, 4]), 0),
    ] {
        v.push(case(
            &format!("rope_half_split_forward {name}"),
            vec![x.clone(), c.clone(), s.clone()],
            Some(nan),
            |o| rope_half_split_forward_dims(&o[0], &o[1], &o[2]).map(drop),
            |b, o| b.rope_half_split_forward(&o[0], &o[1], &o[2]).map(drop),
        ));
        v.push(case(
            &format!("rope_half_split_backward {name}"),
            vec![x, c, s],
            Some(nan),
            |o| rope_half_split_backward_dims(&o[0], &o[1], &o[2]).map(drop),
            |b, o| b.rope_half_split_backward(&o[0], &o[1], &o[2]).map(drop),
        ));
    }

    // ---- causal sdpa ----------------------------------------------------
    let qkv = [1usize, 2, 3, 4];
    for (name, q, k, vv, nan) in [
        ("rank 3", f(&[2, 3, 4]), f(&[2, 3, 4]), f(&[2, 3, 4]), 0),
        ("k differs", f(&qkv), f(&[1, 2, 4, 4]), f(&qkv), 0),
        ("v differs", f(&qkv), f(&qkv), f(&[1, 1, 3, 4]), 0),
    ] {
        v.push(case(
            &format!("causal_sdpa_forward {name}"),
            vec![q, k, vv],
            Some(nan),
            |o| causal_sdpa_forward_dims(&o[0], &o[1], &o[2], None).map(drop),
            |b, o| b.causal_sdpa_forward(&o[0], &o[1], &o[2], None).map(drop),
        ));
    }
    // Operands: q, k, v, output, lse, grad_output.
    let rows = [1usize, 2, 3];
    for (name, ops, nan) in [
        (
            "grad shape",
            [
                f(&qkv),
                f(&qkv),
                f(&qkv),
                f(&qkv),
                f(&rows),
                f(&[1, 2, 3, 5]),
            ],
            0,
        ),
        (
            "k differs",
            [
                f(&qkv),
                f(&[1, 2, 4, 4]),
                f(&qkv),
                f(&qkv),
                f(&rows),
                f(&qkv),
            ],
            5,
        ),
        (
            "output shape",
            [
                f(&qkv),
                f(&qkv),
                f(&qkv),
                f(&[1, 2, 3, 5]),
                f(&rows),
                f(&qkv),
            ],
            5,
        ),
        (
            "lse shape",
            [f(&qkv), f(&qkv), f(&qkv), f(&qkv), f(&[1, 2, 4]), f(&qkv)],
            3,
        ),
        // D9: the grad check runs before the rank.
        (
            "grad before rank",
            [
                f(&[2, 3, 4]),
                f(&[2, 3, 4]),
                f(&[2, 3, 4]),
                f(&[2, 3, 4]),
                f(&[2, 3]),
                f(&[2, 3, 5]),
            ],
            1,
        ),
    ] {
        v.push(case(
            &format!("causal_sdpa_backward {name}"),
            ops.to_vec(),
            Some(nan),
            |o| causal_sdpa_backward_dims(&o[0], &o[1], &o[2], &o[3], &o[4], &o[5], None).map(drop),
            |b, o| {
                b.causal_sdpa_backward(&o[0], &o[1], &o[2], &o[3], &o[4], &o[5], None)
                    .map(drop)
            },
        ));
    }

    // ---- per-head sigmoid gate: input [2, 4], weight [2, 4], bias [2],
    // attn [2, 2, 3] is well formed.
    for (name, x, w, bias, attn, nan) in [
        (
            "weight rank 1",
            f(&[2, 4]),
            f(&[8]),
            f(&[2]),
            f(&[2, 2, 3]),
            0,
        ),
        ("input rank 0", f(&[]), f(&[2, 1]), f(&[2]), f(&[2, 3]), 2),
        (
            "weight in",
            f(&[2, 4]),
            f(&[2, 5]),
            f(&[2]),
            f(&[2, 2, 3]),
            0,
        ),
        ("bias", f(&[2, 4]), f(&[2, 4]), f(&[3]), f(&[2, 2, 3]), 0),
        ("attn rank", f(&[2, 4]), f(&[2, 4]), f(&[2]), f(&[2, 6]), 0),
        (
            "attn heads",
            f(&[2, 4]),
            f(&[2, 4]),
            f(&[2]),
            f(&[2, 3, 3]),
            0,
        ),
        (
            "attn prefix",
            f(&[2, 4]),
            f(&[2, 4]),
            f(&[2]),
            f(&[3, 2, 3]),
            0,
        ),
    ] {
        v.push(case(
            &format!("per_head_sigmoid_gate_forward {name}"),
            vec![x.clone(), w.clone(), bias.clone(), attn.clone()],
            Some(nan),
            |o| per_head_sigmoid_gate_forward_dims(&o[0], &o[1], &o[2], &o[3]).map(drop),
            |b, o| {
                b.per_head_sigmoid_gate_forward(&o[0], &o[1], &o[2], &o[3])
                    .map(drop)
            },
        ));
        let g = f(attn.shape());
        v.push(case(
            &format!("per_head_sigmoid_gate_backward {name}"),
            vec![x, w, bias, attn, g],
            Some(nan),
            |o| per_head_sigmoid_gate_backward_dims(&o[0], &o[1], &o[2], &o[3], &o[4]).map(drop),
            |b, o| {
                b.per_head_sigmoid_gate_backward(&o[0], &o[1], &o[2], &o[3], &o[4])
                    .map(drop)
            },
        ));
    }
    // D10: the grad check runs before the layout.
    v.push(case(
        "per_head_sigmoid_gate_backward grad before layout",
        vec![
            f(&[2, 4]),
            f(&[2, 5]),
            f(&[2]),
            f(&[2, 2, 3]),
            f(&[2, 2, 4]),
        ],
        Some(0),
        |o| per_head_sigmoid_gate_backward_dims(&o[0], &o[1], &o[2], &o[3], &o[4]).map(drop),
        |b, o| {
            b.per_head_sigmoid_gate_backward(&o[0], &o[1], &o[2], &o[3], &o[4])
                .map(drop)
        },
    ));

    // ---- value residual blend -------------------------------------------
    for (name, val, v0, lam, nan) in [
        ("lambda", f(&[2, 3]), f(&[2, 3]), f(&[2]), 0),
        ("value0", f(&[2, 3]), f(&[3, 2]), f(&[1]), 0),
    ] {
        v.push(case(
            &format!("value_residual_blend_forward {name}"),
            vec![val, v0, lam],
            Some(nan),
            |o| value_residual_blend_forward_dims(&o[0], &o[1], &o[2]).map(drop),
            |b, o| {
                b.value_residual_blend_forward(&o[0], &o[1], &o[2])
                    .map(drop)
            },
        ));
    }
    for (name, val, v0, lam, g, nan) in [
        ("lambda", f(&[2, 3]), f(&[2, 3]), f(&[1, 2]), f(&[2, 3]), 0),
        ("value0", f(&[2, 3]), f(&[6]), f(&[]), f(&[2, 3]), 3),
        ("grad", f(&[2, 3]), f(&[2, 3]), f(&[1, 1]), f(&[3, 2]), 1),
    ] {
        v.push(case(
            &format!("value_residual_blend_backward {name}"),
            vec![val, v0, lam, g],
            Some(nan),
            |o| value_residual_blend_backward_dims(&o[0], &o[1], &o[2], &o[3]).map(drop),
            |b, o| {
                b.value_residual_blend_backward(&o[0], &o[1], &o[2], &o[3])
                    .map(drop)
            },
        ));
    }

    // ---- silu, mul, residual add ----------------------------------------
    // silu_forward has one operand and so no cross-operand rule; its
    // operand rules (dtype, empty) are swept instead.
    for (name, x) in [("u32 input", u(&[2, 3])), ("empty", f(&[2, 0]))] {
        v.push(case(
            &format!("silu_forward {name}"),
            vec![x],
            None,
            |o| silu_forward_dims(&o[0]).map(drop),
            |b, o| b.silu_forward(&o[0]).map(drop),
        ));
    }
    v.push(case(
        "silu_backward grad",
        vec![f(&[2, 3]), f(&[3, 2])],
        Some(0),
        |o| silu_backward_dims(&o[0], &o[1]).map(drop),
        |b, o| b.silu_backward(&o[0], &o[1]).map(drop),
    ));
    v.push(case(
        "mul_forward b",
        vec![f(&[2, 3]), f(&[6])],
        Some(0),
        |o| mul_forward_dims(&o[0], &o[1]).map(drop),
        |b, o| b.mul_forward(&o[0], &o[1]).map(drop),
    ));
    for (name, a, bb, g, nan) in [
        ("b", f(&[2, 3]), f(&[3, 2]), f(&[2, 3]), 2),
        ("grad", f(&[2, 3]), f(&[2, 3]), f(&[2, 3, 1]), 1),
    ] {
        v.push(case(
            &format!("mul_backward {name}"),
            vec![a, bb, g],
            Some(nan),
            |o| mul_backward_dims(&o[0], &o[1], &o[2]).map(drop),
            |b, o| b.mul_backward(&o[0], &o[1], &o[2]).map(drop),
        ));
    }
    v.push(case(
        "residual_add_forward y",
        vec![f(&[2, 3]), f(&[2, 2])],
        Some(1),
        |o| residual_add_forward_dims(&o[0], &o[1]).map(drop),
        |b, o| b.residual_add_forward(&o[0], &o[1]).map(drop),
    ));
    for (name, x, y, g, nan) in [
        ("y", f(&[2, 3]), f(&[3, 2]), f(&[2, 3]), 2),
        ("grad", f(&[2, 3]), f(&[2, 3]), f(&[6]), 0),
    ] {
        v.push(case(
            &format!("residual_add_backward {name}"),
            vec![x, y, g],
            Some(nan),
            |o| residual_add_backward_dims(&o[0], &o[1], &o[2]).map(drop),
            |b, o| b.residual_add_backward(&o[0], &o[1], &o[2]).map(drop),
        ));
    }
    for (name, acc, g, nan) in [
        ("shape", f(&[2, 3]), f(&[3, 2]), 1),
        ("u32 grad after a NaN acc", f(&[2, 3]), u(&[2, 3]), 0),
    ] {
        v.push(case(
            &format!("accumulate_grad {name}"),
            vec![acc, g],
            Some(nan),
            |o| accumulate_grad_dims(&o[0], &o[1]).map(drop),
            |b, o| {
                let mut acc = o[0].clone();
                b.accumulate_grad(&mut acc, &o[1])
            },
        ));
    }

    // ---- permute: its validator is `permute_dims` -----------------------
    for (name, x, dims, nan) in [
        ("axis count", f(&[2, 3]), vec![1usize, 0, 2], Some(0)),
        ("repeated axis", f(&[2, 3]), vec![0, 0], Some(0)),
        ("axis out of range", f(&[2, 3]), vec![0, 2], Some(0)),
        ("u32 input", u(&[2, 3]), vec![1, 0], None),
        ("zero axis", f(&[2, 0, 3]), vec![2, 0, 1], None),
        ("u32 input with a zero axis", u(&[0, 3]), vec![1, 0], None),
    ] {
        let check_dims = dims.clone();
        v.push(case(
            &format!("permute {name}"),
            vec![x],
            nan,
            move |o| permute_dims(&o[0], &check_dims).map(drop),
            move |b, o| b.permute(&o[0], &dims).map(drop),
        ));
    }

    // ---- fused linear cross-entropy -------------------------------------
    for (name, x, w, t, chunk, nan) in [
        ("input rank 3", f(&[1, 3, 4]), f(&[5, 4]), u(&[3]), CHUNK, 1),
        ("model dim", f(&[3, 4]), f(&[5, 3]), u(&[3]), CHUNK, 0),
        ("target rows", f(&[3, 4]), f(&[5, 4]), u(&[2]), CHUNK, 0),
        (
            "zero chunk",
            f(&[3, 4]),
            f(&[5, 4]),
            u(&[3]),
            CeChunk { rows: 0, cols: 2 },
            0,
        ),
    ] {
        v.push(case(
            &format!("linear_cross_entropy_mean {name}"),
            vec![x, w, t],
            Some(nan),
            move |o| linear_ce_dims(&o[0], &o[1], &o[2], chunk).map(drop),
            move |b, o| {
                b.linear_cross_entropy_mean(&o[0], &o[1], &o[2], None, chunk, CeGrads::BOTH)
                    .map(drop)
            },
        ));
    }

    // ---- KV cache: q [1, 2, 4, 3], caches [1, 5, 2, 3] -------------------
    for (name, q, k, vv, kv_len, nan) in [
        (
            "q rank 3",
            f(&[2, 4, 3]),
            f(&[1, 5, 2, 3]),
            f(&[1, 5, 2, 3]),
            3,
            1,
        ),
        (
            "k vs v",
            f(&[1, 2, 4, 3]),
            f(&[1, 5, 2, 3]),
            f(&[1, 4, 2, 3]),
            3,
            0,
        ),
        (
            "head dim",
            f(&[1, 2, 4, 2]),
            f(&[1, 5, 2, 3]),
            f(&[1, 5, 2, 3]),
            3,
            1,
        ),
        (
            "kv heads",
            f(&[1, 2, 3, 3]),
            f(&[1, 5, 2, 3]),
            f(&[1, 5, 2, 3]),
            3,
            0,
        ),
        (
            "kv_len",
            f(&[1, 2, 4, 3]),
            f(&[1, 5, 2, 3]),
            f(&[1, 5, 2, 3]),
            6,
            0,
        ),
    ] {
        v.push(case(
            &format!("cached_attention_forward {name}"),
            vec![q, k, vv],
            Some(nan),
            move |o| cached_attention_dims(&o[0], &o[1], &o[2], kv_len, None).map(drop),
            move |b, o| {
                b.cached_attention_forward(&o[0], &o[1], &o[2], kv_len, None)
                    .map(drop)
            },
        ));
    }
    for (name, cache, src, at, nan) in [
        ("src heads", f(&[1, 5, 2, 3]), f(&[1, 2, 3, 3]), 0, 0),
        ("cache rank 3", f(&[5, 2, 3]), f(&[1, 2, 2, 3]), 0, 1),
        (
            "more positions than slots",
            f(&[1, 5, 2, 3]),
            f(&[1, 6, 2, 3]),
            0,
            1,
        ),
    ] {
        v.push(case(
            &format!("kv_cache_write {name}"),
            vec![cache, src],
            Some(nan),
            move |o| kv_cache_write_dims(&o[0], &o[1], at).map(drop),
            move |b, o| {
                let mut cache = o[0].clone();
                b.kv_cache_write(&mut cache, &o[1], at)
            },
        ));
    }

    // ---- causal conv1d + SiLU: input [B, T, C], weight [C, K] ------------
    for (name, x, w, nan) in [
        ("input rank 2", f(&[3, 4]), f(&[4, 2]), Some(1)),
        ("weight rank 1", f(&[1, 3, 4]), f(&[4]), Some(0)),
        ("weight rank 3", f(&[1, 3, 4]), f(&[4, 2, 1]), Some(0)),
        ("channels", f(&[1, 3, 4]), f(&[5, 2]), Some(0)),
        (
            "u32 input after a NaN weight",
            u(&[1, 3, 4]),
            f(&[4, 2]),
            Some(1),
        ),
        ("empty time", f(&[1, 0, 4]), f(&[4, 2]), Some(1)),
    ] {
        v.push(case(
            &format!("causal_conv1d_silu_forward {name}"),
            vec![x, w],
            nan,
            |o| causal_conv1d_silu_forward_dims(&o[0], &o[1]).map(drop),
            |b, o| b.causal_conv1d_silu_forward(&o[0], &o[1]).map(drop),
        ));
    }
    for (name, x, w, g, nan) in [
        ("grad shape", f(&[1, 3, 4]), f(&[4, 2]), f(&[1, 2, 4]), 0),
        ("grad rank", f(&[1, 3, 4]), f(&[4, 2]), f(&[3, 4]), 1),
        ("channels", f(&[1, 3, 4]), f(&[5, 2]), f(&[1, 3, 4]), 0),
        // The grad-shape check runs before the weight's layout.
        (
            "grad before weight",
            f(&[1, 3, 4]),
            f(&[8]),
            f(&[2, 3, 4]),
            0,
        ),
    ] {
        v.push(case(
            &format!("causal_conv1d_silu_backward {name}"),
            vec![x, w, g],
            Some(nan),
            |o| causal_conv1d_silu_backward_dims(&o[0], &o[1], &o[2]).map(drop),
            |b, o| b.causal_conv1d_silu_backward(&o[0], &o[1], &o[2]).map(drop),
        ));
    }

    // ---- gated RMSNorm: input and gate [.., D], weight [D] ---------------
    for (name, x, z, w, eps, nan) in [
        ("gate shape", f(&[2, 4]), f(&[2, 3]), f(&[4]), EPS, 0),
        ("weight dim", f(&[2, 4]), f(&[2, 4]), f(&[3]), EPS, 0),
        ("weight rank 2", f(&[2, 4]), f(&[2, 4]), f(&[1, 4]), EPS, 0),
        ("input rank 0", f(&[]), f(&[]), f(&[1]), EPS, 2),
        ("eps NaN", f(&[2, 4]), f(&[2, 4]), f(&[4]), f32::NAN, 0),
    ] {
        v.push(case(
            &format!("gated_rms_norm_forward {name}"),
            vec![x, z, w],
            Some(nan),
            move |o| gated_rms_norm_forward_dims(&o[0], &o[1], &o[2], eps).map(drop),
            move |b, o| b.gated_rms_norm_forward(&o[0], &o[1], &o[2], eps).map(drop),
        ));
    }
    for (name, x, z, w, g, nan) in [
        ("grad shape", f(&[2, 4]), f(&[2, 4]), f(&[4]), f(&[2, 3]), 0),
        ("gate shape", f(&[2, 4]), f(&[3, 4]), f(&[4]), f(&[2, 4]), 3),
        ("weight dim", f(&[2, 4]), f(&[2, 4]), f(&[3]), f(&[2, 4]), 0),
        // The grad-shape check runs before the weight.
        (
            "grad before weight",
            f(&[2, 4]),
            f(&[2, 4]),
            f(&[3]),
            f(&[4, 2]),
            0,
        ),
    ] {
        v.push(case(
            &format!("gated_rms_norm_backward {name}"),
            vec![x, z, w, g],
            Some(nan),
            |o| gated_rms_norm_backward_dims(&o[0], &o[1], &o[2], &o[3], EPS).map(drop),
            |b, o| {
                b.gated_rms_norm_backward(&o[0], &o[1], &o[2], &o[3], EPS)
                    .map(drop)
            },
        ));
    }

    // ---- partial RoPE: x [B, T, H, D], tables [T, rotary] ----------------
    for (name, x, c, s) in [
        ("x rank 3", f(&[1, 2, 8]), f(&[2, 4]), f(&[2, 4])),
        ("cos vs sin", f(&[1, 2, 1, 8]), f(&[2, 4]), f(&[4, 2])),
        ("time axis", f(&[1, 2, 1, 8]), f(&[3, 4]), f(&[3, 4])),
        ("odd rotary dim", f(&[1, 2, 1, 8]), f(&[2, 3]), f(&[2, 3])),
        (
            "rotary above head dim",
            f(&[1, 2, 1, 8]),
            f(&[2, 10]),
            f(&[2, 10]),
        ),
        (
            "table rank 3",
            f(&[1, 2, 1, 8]),
            f(&[1, 2, 4]),
            f(&[1, 2, 4]),
        ),
    ] {
        v.push(case(
            &format!("rope_partial_forward {name}"),
            vec![x.clone(), c.clone(), s.clone()],
            Some(0),
            |o| rope_partial_forward_dims(&o[0], &o[1], &o[2]).map(drop),
            |b, o| b.rope_partial_forward(&o[0], &o[1], &o[2]).map(drop),
        ));
        v.push(case(
            &format!("rope_partial_backward {name}"),
            vec![x, c, s],
            Some(0),
            |o| rope_partial_backward_dims(&o[0], &o[1], &o[2]).map(drop),
            |b, o| b.rope_partial_backward(&o[0], &o[1], &o[2]).map(drop),
        ));
    }

    // ---- sigmoid ---------------------------------------------------------
    for (name, x) in [("u32 input", u(&[2, 3])), ("empty", f(&[2, 0]))] {
        v.push(case(
            &format!("sigmoid_forward {name}"),
            vec![x],
            None,
            |o| sigmoid_forward_dims(&o[0]).map(drop),
            |b, o| b.sigmoid_forward(&o[0]).map(drop),
        ));
    }
    for (name, x, g, nan) in [
        ("grad", f(&[2, 3]), f(&[3, 2]), 0),
        ("u32 grad after a NaN input", f(&[2, 3]), u(&[2, 3]), 0),
    ] {
        v.push(case(
            &format!("sigmoid_backward {name}"),
            vec![x, g],
            Some(nan),
            |o| sigmoid_backward_dims(&o[0], &o[1]).map(drop),
            |b, o| b.sigmoid_backward(&o[0], &o[1]).map(drop),
        ));
    }

    // ---- gdn_log_decay: a [.., H], a_log and dt_bias [H] -----------------
    for (name, a, al, dt, nan) in [
        ("input rank 0", f(&[]), f(&[1]), f(&[1]), 1),
        ("a_log", f(&[2, 3]), f(&[2]), f(&[3]), 0),
        ("dt_bias", f(&[2, 3]), f(&[3]), f(&[4]), 0),
        ("a_log rank 2", f(&[2, 3]), f(&[1, 3]), f(&[3]), 0),
    ] {
        v.push(case(
            &format!("gdn_log_decay_forward {name}"),
            vec![a, al, dt],
            Some(nan),
            |o| gdn_log_decay_forward_dims(&o[0], &o[1], &o[2]).map(drop),
            |b, o| b.gdn_log_decay_forward(&o[0], &o[1], &o[2]).map(drop),
        ));
    }
    for (name, a, al, dt, g, nan) in [
        ("grad", f(&[2, 3]), f(&[3]), f(&[3]), f(&[3, 2]), 0),
        ("a_log", f(&[2, 3]), f(&[2]), f(&[3]), f(&[2, 3]), 3),
        ("dt_bias", f(&[2, 3]), f(&[3]), f(&[2]), f(&[2, 3]), 0),
        // The grad's dtype is checked before the layout.
        (
            "u32 grad before layout",
            f(&[2, 3]),
            f(&[2]),
            f(&[3]),
            u(&[2, 3]),
            0,
        ),
    ] {
        v.push(case(
            &format!("gdn_log_decay_backward {name}"),
            vec![a, al, dt, g],
            Some(nan),
            |o| gdn_log_decay_backward_dims(&o[0], &o[1], &o[2], &o[3]).map(drop),
            |b, o| {
                b.gdn_log_decay_backward(&o[0], &o[1], &o[2], &o[3])
                    .map(drop)
            },
        ));
    }

    // ---- chunked gated delta rule ----------------------------------------
    for (name, ops, nan) in [
        ("q rank 3", gdn_with(0, f(&[3, 2, 4])), 1),
        ("k vs q", gdn_with(1, f(&[1, 3, 2, 5])), 0),
        ("v tokens", gdn_with(2, f(&[1, 2, 2, 4])), 0),
        ("g shape", gdn_with(3, f(&[1, 3, 3])), 0),
        ("beta shape", gdn_with(4, f(&[1, 2, 2])), 0),
        ("state shape", gdn_with(5, f(&[1, 2, 4, 5])), 0),
        ("u32 beta after a NaN q", gdn_with(4, u(&[1, 3, 2])), 0),
    ] {
        v.push(case(
            &format!("chunked_gdn_forward {name}"),
            ops,
            Some(nan),
            |o| chunked_gdn_forward_dims(gdn_inputs(o)).map(drop),
            |b, o| b.chunked_gdn_forward(gdn_inputs(o)).map(drop),
        ));
    }
    // Backward operands: the six above, then checkpoints [1, 2, 1, 4, 4]
    // (one 64-token chunk), grad_output shaped like v, and optionally the
    // final-state gradient [1, 2, 4, 4].
    let gdn_back = |which: usize, bad: Tensor| -> Vec<Tensor> {
        let mut ops = gdn_with(0, f(&[1, 3, 2, 4]));
        ops.extend([f(&[1, 2, 1, 4, 4]), f(&[1, 3, 2, 4]), f(&[1, 2, 4, 4])]);
        ops[which] = bad;
        ops
    };
    for (name, ops, nan) in [
        ("checkpoints", gdn_back(6, f(&[1, 2, 2, 4, 4])), 0),
        ("grad shape", gdn_back(7, f(&[1, 3, 2, 5])), 0),
        ("final state grad", gdn_back(8, f(&[1, 2, 4, 5])), 0),
        ("state", gdn_back(5, f(&[1, 2, 4, 5])), 7),
        (
            "u32 checkpoints after a NaN q",
            gdn_back(6, u(&[1, 2, 1, 4, 4])),
            0,
        ),
    ] {
        v.push(case(
            &format!("chunked_gdn_backward {name}"),
            ops,
            Some(nan),
            |o| chunked_gdn_backward_dims(gdn_inputs(o), &o[6], &o[7], o.get(8)).map(drop),
            |b, o| {
                b.chunked_gdn_backward(gdn_inputs(o), &o[6], &o[7], o.get(8))
                    .map(drop)
            },
        ));
    }

    v
}

/// The bytes of every operand: what the input copies charge.
fn input_bytes(operands: &[Tensor]) -> u64 {
    operands
        .iter()
        .map(|t| (t.shape().iter().product::<usize>() * t.dtype().size()) as u64)
        .sum()
}

/// One row of the sweep's report: `case | mode | got | validator`.
fn sweep() -> (usize, Vec<String>) {
    let mut runs = 0;
    let mut failures = Vec::new();
    for c in cases() {
        let want = match (c.check)(&c.operands) {
            Err(err) => format!("{err:?}"),
            Ok(()) => panic!("{}: the validator accepts this call", c.name),
        };
        let mut modes: Vec<(&str, u64, Vec<Tensor>)> = vec![
            ("a budget 0", 0, c.operands.clone()),
            ("b inputs fit", input_bytes(&c.operands), c.operands.clone()),
        ];
        if let Some(i) = c.nan_at {
            let mut poisoned = c.operands.clone();
            poisoned[i] = nan_like(&poisoned[i]);
            modes.push(("c NaN operand", 1 << 24, poisoned));
        }
        for (mode, cap, operands) in modes {
            runs += 1;
            let cpu = CpuBackend::new(Budget::new(cap));
            let got = match (c.run)(&cpu, &operands) {
                Err(err) => format!("{err:?}"),
                Ok(()) => "Ok".to_string(),
            };
            let live = cpu.budget().live_bytes().unwrap();
            if got != want || live != 0 {
                failures.push(format!(
                    "{} | {mode} | {got} | {want} | live {live}",
                    c.name
                ));
            }
        }
    }
    (runs, failures)
}

#[test]
fn every_malformed_call_returns_the_validator_error_whatever_the_budget_or_values() {
    let (runs, failures) = sweep();
    assert!(
        failures.is_empty(),
        "{} of {runs} malformed calls disagree with their validator \
         (case | mode | got | validator):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

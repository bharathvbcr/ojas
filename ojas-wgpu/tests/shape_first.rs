//! Shape first (docs/shape-contract.md): every wgpu op runs its
//! `ojas_core::shapes` validator before it checks placement, looks up ids,
//! checks values or device limits, or charges the budget.
//!
//! Each malformed call below must return exactly what the validator returns
//! when called directly (variant, op name and detail), leave nothing charged
//! and record nothing (the next `sync` is clean), in four modes:
//!
//! - (a) device operands, the op's backend under `Budget::new(0)`;
//! - (b) device operands, a cap that holds the inputs' bytes but no output;
//! - (c) device operands with a NaN in a well-formed f32 operand;
//! - (d) host operands (decision 2: the validator runs before placement).
//!
//! An operand with a zero axis cannot be uploaded, so its case runs (d) only.

use std::sync::OnceLock;

use ojas_core::{
    accumulate_grad_dims, adamw_step_dims, cached_attention_dims, causal_sdpa_backward_dims,
    causal_sdpa_forward_dims, clip_grad_norm_dims, cross_entropy_mean_backward_dims,
    cross_entropy_mean_forward_dims, embedding_backward_dims, embedding_forward_dims,
    kv_cache_write_dims, linear_backward_dims, linear_ce_dims, linear_forward_dims,
    mul_backward_dims, mul_forward_dims, muon_ns5_step_dims, per_head_sigmoid_gate_backward_dims,
    per_head_sigmoid_gate_forward_dims, permute_dims, residual_add_backward_dims,
    residual_add_forward_dims, rms_norm_backward_dims, rms_norm_forward_dims,
    rms_qk_norm_backward_dims, rms_qk_norm_forward_dims, rope_half_split_backward_dims,
    rope_half_split_forward_dims, silu_backward_dims, silu_forward_dims,
    value_residual_blend_backward_dims, value_residual_blend_forward_dims, AdamWConfig, Backend,
    Budget, CeChunk, DType, MuonNs5Config, OjasError, Tensor,
};
use ojas_wgpu::WgpuBackend;

type Check = Box<dyn Fn(&[Tensor]) -> Result<(), OjasError>>;
type Run = Box<dyn Fn(&WgpuBackend, &[Tensor]) -> Result<(), OjasError>>;

struct Case {
    name: String,
    operands: Vec<Tensor>,
    /// A well-formed f32 operand that (c) fills with NaN; `None` when there
    /// is none.
    nan_at: Option<usize>,
    check: Check,
    run: Run,
}

fn host_budget() -> &'static Budget {
    static B: OnceLock<Budget> = OnceLock::new();
    B.get_or_init(|| Budget::new(1 << 26))
}

/// An f32 tensor of `shape` holding small finite values.
fn f(shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| 0.25 + (i % 7) as f32 * 0.125).collect();
    Tensor::from_f32(&data, shape, host_budget()).unwrap()
}

/// A u32 tensor of `shape` holding `value` (0 is a valid id and target).
fn uv(shape: &[usize], value: u32) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::from_u32(&vec![value; n], shape, host_budget()).unwrap()
}

fn u(shape: &[usize]) -> Tensor {
    uv(shape, 0)
}

fn nan_like(t: &Tensor) -> Tensor {
    let n: usize = t.shape().iter().product();
    Tensor::from_f32(&vec![f32::NAN; n], t.shape(), host_budget()).unwrap()
}

fn case<C, R>(name: &str, operands: Vec<Tensor>, nan_at: Option<usize>, check: C, run: R) -> Case
where
    C: Fn(&[Tensor]) -> Result<(), OjasError> + 'static,
    R: Fn(&WgpuBackend, &[Tensor]) -> Result<(), OjasError> + 'static,
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

fn adamw() -> AdamWConfig {
    AdamWConfig::nanolab(1e-3, 0.1)
}

#[allow(clippy::too_many_lines)]
fn cases() -> Vec<Case> {
    let mut v = Vec::new();

    // ---- embedding: the id lookup must not run before the table rank (D12)
    for (name, table, ids, nan) in [
        ("table rank 1", f(&[4]), u(&[2]), Some(0)),
        ("table rank 3", f(&[2, 2, 2]), uv(&[2], 9), Some(0)),
        ("f32 ids", f(&[4, 2]), f(&[2]), Some(0)),
        ("u32 table", u(&[4, 2]), u(&[2]), None),
    ] {
        v.push(case(
            &format!("embedding_forward {name}"),
            vec![table, ids],
            nan,
            |o| embedding_forward_dims(&o[0], &o[1]).map(drop),
            |b, o| b.embedding_forward(&o[0], &o[1]).map(drop),
        ));
    }
    for (name, table, ids, g, nan) in [
        ("grad shape", f(&[4, 2]), u(&[3]), f(&[3, 3]), 0),
        (
            "grad shape, id out of range",
            f(&[4, 2]),
            uv(&[3], 9),
            f(&[2, 2]),
            0,
        ),
        ("table rank 1", f(&[4]), u(&[3]), f(&[3, 4]), 2),
        ("u32 grad", f(&[4, 2]), u(&[3]), u(&[3, 2]), 0),
    ] {
        v.push(case(
            &format!("embedding_backward {name}"),
            vec![table, ids, g],
            Some(nan),
            |o| embedding_backward_dims(&o[0], &o[1], &o[2]).map(drop),
            |b, o| b.embedding_backward(&o[0], &o[1], &o[2]).map(drop),
        ));
    }

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
        "linear_forward u32 weight",
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
    for (name, x, w, eps, nan) in [
        ("input rank 0", f(&[]), f(&[1]), EPS, 1),
        ("weight dim", f(&[2, 4]), f(&[3]), EPS, 0),
        ("weight rank 2", f(&[2, 4]), f(&[1, 4]), EPS, 0),
        ("eps before the weight", f(&[2, 4]), f(&[3]), f32::NAN, 0),
    ] {
        v.push(case(
            &format!("rms_norm_forward {name}"),
            vec![x, w],
            Some(nan),
            move |o| rms_norm_forward_dims(&o[0], &o[1], eps).map(drop),
            move |b, o| b.rms_norm_forward(&o[0], &o[1], eps).map(drop),
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
        // The head-dim cap is a device limit, after the validator.
        (
            "shape before the head-dim cap",
            f(&[1, 1, 2, 130]),
            f(&[1, 1, 3, 130]),
            f(&[1, 1, 2, 130]),
            0,
        ),
    ] {
        v.push(case(
            &format!("causal_sdpa_forward {name}"),
            vec![q, k, vv],
            Some(nan),
            |o| causal_sdpa_forward_dims(&o[0], &o[1], &o[2], None).map(drop),
            |b, o| b.causal_sdpa_forward(&o[0], &o[1], &o[2], None).map(drop),
        ));
    }
    for (name, q, k, vv, g, nan) in [
        ("grad shape", f(&qkv), f(&qkv), f(&qkv), f(&[1, 2, 3, 5]), 0),
        ("k differs", f(&qkv), f(&[1, 2, 4, 4]), f(&qkv), f(&qkv), 3),
        // D9: the grad check runs before the rank.
        (
            "grad before rank",
            f(&[2, 3, 4]),
            f(&[2, 3, 4]),
            f(&[2, 3, 4]),
            f(&[2, 3, 5]),
            1,
        ),
    ] {
        // The output and lse are well formed for `q`; index 3 of the old
        // four operands (the gradient) is index 5 of q, k, v, output, lse,
        // grad_output.
        let out = f(q.shape());
        let lse = f(&q.shape()[..q.shape().len().saturating_sub(1)]);
        let nan = if nan == 3 { 5 } else { nan };
        v.push(case(
            &format!("causal_sdpa_backward {name}"),
            vec![q, k, vv, out, lse, g],
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

    // ---- silu, mul, residual add, accumulate_grad -----------------------
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
        ("u32 grad", f(&[2, 3]), u(&[2, 3]), 0),
        ("u32 acc", u(&[2, 3]), f(&[2, 3]), 1),
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

    // ---- cross-entropy: logits are checked before targets (D6) ----------
    for (name, logits, targets, nan) in [
        ("targets prefix", f(&[2, 5]), u(&[3]), Some(0)),
        (
            "targets prefix, target out of range",
            f(&[2, 5]),
            uv(&[3], 99),
            Some(0),
        ),
        ("logits rank 0", f(&[]), u(&[]), Some(0)),
        ("f32 targets", f(&[2, 5]), f(&[2]), Some(0)),
        (
            "u32 logits before f32 targets",
            u(&[2, 5]),
            f(&[2]),
            Some(1),
        ),
    ] {
        v.push(case(
            &format!("cross_entropy_mean_forward {name}"),
            vec![logits.clone(), targets.clone()],
            nan,
            |o| cross_entropy_mean_forward_dims(&o[0], &o[1]).map(drop),
            |b, o| b.cross_entropy_mean_forward(&o[0], &o[1], None).map(drop),
        ));
        v.push(case(
            &format!("cross_entropy_mean_backward {name}"),
            vec![logits, targets],
            nan,
            |o| cross_entropy_mean_backward_dims(&o[0], &o[1]).map(drop),
            |b, o| b.cross_entropy_mean_backward(&o[0], &o[1], None).map(drop),
        ));
    }

    // ---- clip_grad_norm -------------------------------------------------
    v.push(case(
        "clip_grad_norm no gradients",
        vec![],
        None,
        |o| clip_grad_norm_dims(o).map(drop),
        |b, o| {
            let mut grads = o.to_vec();
            b.clip_grad_norm(&mut grads, 1.0).map(drop)
        },
    ));
    v.push(case(
        "clip_grad_norm u32 gradient",
        vec![f(&[2, 3]), u(&[4])],
        Some(0),
        |o| clip_grad_norm_dims(o).map(drop),
        |b, o| {
            let mut grads = o.to_vec();
            b.clip_grad_norm(&mut grads, 1.0).map(drop)
        },
    ));
    v.push(case(
        "clip_grad_norm u32 gradient before a bad max_norm",
        vec![u(&[4])],
        None,
        |o| clip_grad_norm_dims(o).map(drop),
        |b, o| {
            let mut grads = o.to_vec();
            b.clip_grad_norm(&mut grads, f32::NAN).map(drop)
        },
    ));

    // ---- AdamW and Muon: param first (D7), shapes before the config -----
    for (name, p, g, m1, m2, nan) in [
        (
            "grad shape",
            f(&[2, 3]),
            f(&[3, 2]),
            f(&[2, 3]),
            f(&[2, 3]),
            0,
        ),
        (
            "moment2 shape",
            f(&[2, 3]),
            f(&[2, 3]),
            f(&[2, 3]),
            f(&[6]),
            1,
        ),
        (
            "u32 param",
            u(&[2, 3]),
            f(&[2, 3]),
            f(&[2, 3]),
            f(&[2, 3]),
            1,
        ),
        (
            "u32 moment1",
            f(&[2, 3]),
            f(&[2, 3]),
            u(&[2, 3]),
            f(&[2, 3]),
            1,
        ),
    ] {
        v.push(case(
            &format!("adamw_step {name}"),
            vec![p, g, m1, m2],
            Some(nan),
            |o| adamw_step_dims(&o[0], &o[1], &o[2], &o[3]).map(drop),
            |b, o| {
                let (mut p, mut m1, mut m2) = (o[0].clone(), o[2].clone(), o[3].clone());
                b.adamw_step(&mut p, &o[1], &mut m1, &mut m2, 0, adamw())
            },
        ));
    }
    v.push(case(
        "adamw_step shape before a bad config",
        vec![f(&[2, 3]), f(&[3, 2]), f(&[2, 3]), f(&[2, 3])],
        Some(0),
        |o| adamw_step_dims(&o[0], &o[1], &o[2], &o[3]).map(drop),
        |b, o| {
            let (mut p, mut m1, mut m2) = (o[0].clone(), o[2].clone(), o[3].clone());
            b.adamw_step(
                &mut p,
                &o[1],
                &mut m1,
                &mut m2,
                0,
                AdamWConfig::nanolab(f64::NAN, 0.1),
            )
        },
    ));
    for (name, p, g, m, nan) in [
        ("param rank 1", f(&[4]), f(&[4]), f(&[4]), 1),
        ("grad shape", f(&[2, 3]), f(&[3, 2]), f(&[2, 3]), 0),
        ("momentum shape", f(&[2, 3]), f(&[2, 3]), f(&[2, 2]), 1),
        ("u32 param", u(&[2, 3]), f(&[2, 3]), f(&[2, 3]), 1),
    ] {
        v.push(case(
            &format!("muon_ns5_step {name}"),
            vec![p, g, m],
            Some(nan),
            |o| muon_ns5_step_dims(&o[0], &o[1], &o[2]).map(drop),
            |b, o| {
                let (mut p, mut m) = (o[0].clone(), o[2].clone());
                b.muon_ns5_step(&mut p, &o[1], &mut m, MuonNs5Config::nanolab_default())
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
                b.linear_cross_entropy_mean(&o[0], &o[1], &o[2], None, chunk, true)
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
            move |o| cached_attention_dims(&o[0], &o[1], &o[2], kv_len).map(drop),
            move |b, o| {
                b.cached_attention_forward(&o[0], &o[1], &o[2], kv_len)
                    .map(drop)
            },
        ));
    }
    for (name, cache, src, at, nan) in [
        ("src heads", f(&[1, 5, 2, 3]), f(&[1, 2, 3, 3]), 0, 0),
        ("cache rank 3", f(&[5, 2, 3]), f(&[1, 2, 2, 3]), 0, 1),
        ("past capacity", f(&[1, 5, 2, 3]), f(&[1, 2, 2, 3]), 4, 1),
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

    v
}

fn input_bytes(operands: &[Tensor]) -> u64 {
    operands
        .iter()
        .map(|t| (t.shape().iter().product::<usize>() * t.dtype().size()) as u64)
        .sum()
}

/// One row of the report per disagreement: `case | mode | got | validator`.
fn sweep() -> (usize, Vec<String>) {
    let pool = WgpuBackend::open(Budget::new(1 << 28)).expect("wgpu adapter");
    pool.sync().unwrap();
    let mut runs = 0;
    let mut failures = Vec::new();
    for c in cases() {
        let want = match (c.check)(&c.operands) {
            Err(err) => format!("{err:?}"),
            Ok(()) => panic!("{}: the validator accepts this call", c.name),
        };
        let uploadable = c.operands.iter().all(|t| !t.shape().contains(&0));
        let up = |ops: &[Tensor]| -> Vec<Tensor> {
            ops.iter().map(|t| pool.upload(t).unwrap()).collect()
        };
        let mut modes: Vec<(&str, u64, Vec<Tensor>)> = Vec::new();
        if uploadable {
            modes.push(("a budget 0", 0, up(&c.operands)));
            modes.push(("b inputs fit", input_bytes(&c.operands), up(&c.operands)));
            if let Some(i) = c.nan_at {
                let mut poisoned = c.operands.clone();
                poisoned[i] = nan_like(&poisoned[i]);
                modes.push(("c NaN operand", 1 << 26, up(&poisoned)));
            }
        }
        modes.push(("d host operands", 1 << 26, c.operands.clone()));
        for (mode, cap, operands) in modes {
            runs += 1;
            let g = WgpuBackend::with_context(pool.context().clone(), Budget::new(cap));
            let got = match (c.run)(&g, &operands) {
                Err(err) => format!("{err:?}"),
                Ok(()) => "Ok".to_string(),
            };
            let live = g.budget().live_bytes().unwrap();
            let synced = g.sync();
            if got != want || live != 0 || synced.is_err() {
                failures.push(format!(
                    "{} | {mode} | {got} | {want} | live {live} | sync {synced:?}",
                    c.name
                ));
            }
        }
    }
    (runs, failures)
}

/// Decision 4: contiguity is the backend's and comes after the validator,
/// so a non-contiguous device view in a malformed call reports the shape.
#[test]
fn a_non_contiguous_view_in_a_malformed_call_reports_the_validator_error() {
    let g = WgpuBackend::open(Budget::new(1 << 24)).expect("wgpu adapter");
    let x = g.upload(&f(&[2, 3])).unwrap();
    let transposed = x.view(&[3, 2], &[1, 3], 0).unwrap();
    assert!(!transposed.is_contiguous().unwrap());
    let y = g.upload(&f(&[2, 3])).unwrap();
    let want = format!("{:?}", mul_forward_dims(&transposed, &y).unwrap_err());
    let got = format!("{:?}", g.mul_forward(&transposed, &y).unwrap_err());
    assert_eq!(got, want);
    // Well formed, it is refused for the layout, as before.
    let z = g.upload(&f(&[3, 2])).unwrap();
    match g.mul_forward(&transposed, &z) {
        Err(OjasError::Shape { detail, .. }) => {
            assert_eq!(detail, "non-contiguous view is not supported")
        }
        other => panic!("{other:?}"),
    }
    g.sync().unwrap();
}

/// D17: `ojas_core::permute_dims` refuses a zero axis, so an empty device
/// view gets the validator's error before placement or dispatch.
#[test]
fn permute_refuses_an_empty_device_view() {
    let g = WgpuBackend::open(Budget::new(1 << 24)).expect("wgpu adapter");
    let x = g.upload(&f(&[2, 3])).unwrap();
    let empty = x.view(&[0, 3], &[3, 1], 0).unwrap();
    match g.permute(&empty, &[1, 0]) {
        Err(OjasError::Shape {
            op: "permute",
            detail,
        }) => assert_eq!(detail, "empty tensor"),
        other => panic!("expected Shape(empty tensor), got {other:?}"),
    }
    g.sync().unwrap();
}

/// D13: `max_norm` is checked after the norm, as the CPU checks it, so
/// non-finite gradients outrank a negative `max_norm`. Each case must give
/// the CPU's exact error.
#[test]
fn clip_checks_max_norm_after_the_norm_as_the_cpu_does() {
    let g = WgpuBackend::open(Budget::new(1 << 24)).expect("wgpu adapter");
    g.sync().unwrap();
    let cpu = ojas_cpu::CpuBackend::new(Budget::new(1 << 24));
    let clean = f(&[3, 5]);
    let mut nan = (0..15).map(|i| i as f32 * 0.1).collect::<Vec<_>>();
    nan[4] = f32::NAN;
    let nan = Tensor::from_f32(&nan, &[3, 5], host_budget()).unwrap();
    for (name, grad, max_norm) in [
        ("NaN grad, negative max_norm", &nan, -1.0f32),
        ("NaN grad, NaN max_norm", &nan, f32::NAN),
        ("clean grad, NaN max_norm", &clean, f32::NAN),
        ("clean grad, negative max_norm", &clean, -1.0),
    ] {
        let want = format!(
            "{:?}",
            cpu.clip_grad_norm(&mut [grad.clone()], max_norm)
                .unwrap_err()
        );
        let mut grads = [g.upload(grad).unwrap()];
        let got = format!("{:?}", g.clip_grad_norm(&mut grads, max_norm).unwrap_err());
        assert_eq!(got, want, "{name}");
        // Nothing was scaled: the gradient holds its uploaded bits.
        let after = g.download(&grads[0]).unwrap().to_f32_vec().unwrap();
        let before = grad.to_f32_vec().unwrap();
        assert!(
            after
                .iter()
                .zip(&before)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{name}: the gradient changed"
        );
        g.sync().unwrap();
    }
}

#[test]
fn every_malformed_call_returns_the_validator_error_whatever_the_budget_values_or_placement() {
    let (runs, failures) = sweep();
    assert!(
        failures.is_empty(),
        "{} of {runs} malformed calls disagree with their validator \
         (case | mode | got | validator):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

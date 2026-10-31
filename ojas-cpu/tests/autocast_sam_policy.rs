//! The bf16 autocast policy of the SAM ops, enforced op by op.
//!
//! `ojas_core::SAM_AUTOCAST_POLICY` assigns each op added for the SAM
//! encoder (attention, encoder ops, spatial ops) an `AutocastClass`; the
//! classes are defined, once, in `ojas-core/src/autocast.rs`. This test runs
//! every op in that table through `Autocast<CpuBackend>` inside a bf16
//! region and checks the observable behaviour of its class: which values are
//! rounded, which operands round before the op, and what the output's
//! storage tag (`Tensor::bf16_exact`) says.
//!
//! It fails when an op is added to the `Backend` trait without a class (or
//! an entry in `OUTSIDE_THE_SAM_SET`), when a class in the table is wrong
//! for what `Autocast` does, and when `Autocast` changes an op's rounding
//! without the table following.

use ojas_core::{
    round_f32_to_bf16, AffineGrad, AttentionOpts, Autocast, AutocastClass, AutocastMode, Backend,
    Budget, Conv2dParams, ConvTranspose2dParams, DType, Numerics, OjasError, Tensor,
    BACKEND_METHODS, SAM_AUTOCAST_POLICY,
};
use ojas_cpu::CpuBackend;

/// Policy entries whose `CpuBackend` implementation has not landed: the
/// trait default refuses them (and `Autocast<CpuBackend>` passes the refusal
/// through), so the class cannot be observed yet. The ops stay in the policy
/// and have `arguments`/`try_call` arms, so the lane that implements one
/// deletes its name here and the class check below then runs it. Both halves
/// are enforced: a name listed here must still refuse, so the list cannot
/// outlive an implementation, and a name not listed must run.
const REFUSED_ON_CPU_UNTIL_IMPLEMENTED: &[&str] = &[
    "gelu_tanh_forward",
    "gelu_tanh_backward",
    "seq_mean_forward",
    "l2_normalize_forward",
];

/// Every other `Backend` method. Their policy predates the SAM ops and is
/// not classified here.
const OUTSIDE_THE_SAM_SET: &[&str] = &[
    "accumulate_grad",
    "adamw_step",
    "argmax_rows",
    "autocast_region",
    "bf16_operands",
    "budget",
    "cached_attention_forward",
    "cast_bf16",
    "causal_conv1d_silu_backward",
    "causal_conv1d_silu_forward",
    "causal_sdpa_backward",
    "causal_sdpa_backward_recompute",
    "causal_sdpa_forward",
    "chunked_gdn_backward",
    "chunked_gdn_forward",
    "clip_grad_norm",
    "cross_entropy_mean_backward",
    "cross_entropy_mean_forward",
    "download",
    "embedding_backward",
    "embedding_forward",
    "gated_rms_norm_backward",
    "gated_rms_norm_forward",
    "gdn_log_decay_backward",
    "gdn_log_decay_forward",
    "id",
    "kv_cache_write",
    "linear_backward",
    "linear_backward_input",
    "linear_cross_entropy_mean",
    "linear_forward",
    "mul_backward",
    "mul_forward",
    "muon_ns5_step",
    "numerics",
    "optimizer_scratch_bytes",
    "per_head_sigmoid_gate_backward",
    "per_head_sigmoid_gate_backward_saved",
    "per_head_sigmoid_gate_forward",
    "per_head_sigmoid_gate_forward_saving",
    "permute",
    "residual_add_backward",
    "residual_add_forward",
    "rms_norm_backward",
    "rms_norm_forward",
    "rms_qk_norm_backward",
    "rms_qk_norm_forward",
    "rope_half_split_backward",
    "rope_half_split_forward",
    "rope_partial_backward",
    "rope_partial_forward",
    "scale_grad",
    "sigmoid_backward",
    "sigmoid_forward",
    "silu_backward",
    "silu_forward",
    "sync",
    "to_bf16",
    "to_f32",
    "topk_rows",
    "upload",
    "value_residual_blend_backward",
    "value_residual_blend_forward",
];

fn budget() -> Budget {
    Budget::new(1 << 30)
}

fn raw() -> CpuBackend {
    CpuBackend::with_threads(budget(), 1)
        .unwrap()
        .with_numerics(Numerics::Exact)
}

/// Deterministic values in `[-amp, amp)`, almost none of them bf16-exact.
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

fn tensor(shape: &[usize], seed: u64, amp: f32) -> Tensor {
    let n = shape.iter().product();
    Tensor::from_f32(&values(n, seed, amp), shape, &budget()).unwrap()
}

/// Values near 1, for a norm's scale.
fn positive(shape: &[usize], seed: u64) -> Tensor {
    let n = shape.iter().product();
    let v: Vec<f32> = values(n, seed, 0.4).iter().map(|x| 1.0 + x).collect();
    Tensor::from_f32(&v, shape, &budget()).unwrap()
}

fn bits(t: &Tensor) -> Vec<u32> {
    t.to_f32_vec()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn round_all(t: &Tensor) -> Tensor {
    let v: Vec<f32> = t
        .to_f32_vec()
        .unwrap()
        .into_iter()
        .map(round_f32_to_bf16)
        .collect();
    Tensor::from_f32(&v, t.shape(), &budget()).unwrap()
}

fn all_bf16_values(t: &Tensor) -> bool {
    t.to_f32_vec()
        .unwrap()
        .iter()
        .all(|v| round_f32_to_bf16(*v).to_bits() == v.to_bits())
}

/// The tensor arguments of `op`, in the order of its `Backend` method
/// (`AttentionOpts::bias` last), and for `attention_backward` the forward's
/// `output` and `lse` computed on the raw backend.
fn arguments(op: &str) -> Vec<Tensor> {
    let attn = || {
        [
            tensor(&[1, 2, 3, 8], 1, 1.0),
            tensor(&[1, 1, 5, 8], 2, 1.0),
            tensor(&[1, 1, 5, 8], 3, 1.0),
        ]
    };
    match op {
        "layer_norm_forward" => vec![
            tensor(&[3, 8], 10, 2.0),
            positive(&[8], 11),
            tensor(&[8], 12, 0.5),
        ],
        "channel_layer_norm_forward" | "group_norm_forward" => vec![
            tensor(&[2, 4, 3, 3], 13, 2.0),
            positive(&[4], 14),
            tensor(&[4], 15, 0.5),
        ],
        "gelu_erf_forward"
        | "gelu_tanh_forward"
        | "relu_forward"
        | "softmax_forward"
        | "scale_forward"
        | "l2_normalize_forward" => vec![tensor(&[3, 8], 16, 2.0)],
        // `try_call` supplies the lengths `[5, 3]`: `U32` cannot be rounded.
        "seq_mean_forward" => vec![tensor(&[2, 5, 8], 46, 2.0)],
        "group_norm_backward" => vec![
            tensor(&[2, 4, 3, 3], 13, 2.0),
            positive(&[4], 14),
            tensor(&[4], 15, 0.5),
            tensor(&[2, 4, 3, 3], 47, 1.0),
        ],
        "gelu_tanh_backward" => vec![tensor(&[3, 8], 16, 2.0), tensor(&[3, 8], 41, 1.0)],
        "interpolate_bilinear_backward" => vec![
            tensor(&[1, 2, 4, 4], 29, 2.0),
            tensor(&[1, 2, 6, 6], 48, 2.0),
        ],
        "broadcast_add_forward" | "broadcast_mul_forward" => {
            vec![tensor(&[3, 8], 17, 2.0), tensor(&[8], 18, 2.0)]
        }
        "linear_bias_forward" => vec![
            tensor(&[3, 8], 19, 1.0),
            tensor(&[5, 8], 20, 1.0),
            tensor(&[5], 21, 1.0),
        ],
        "conv2d_forward" => vec![
            tensor(&[1, 4, 6, 6], 22, 1.0),
            tensor(&[4, 2, 3, 3], 23, 1.0),
            tensor(&[4], 24, 1.0),
        ],
        "conv_transpose2d_forward" => vec![
            tensor(&[1, 4, 4, 4], 25, 1.0),
            tensor(&[4, 3, 3, 3], 26, 1.0),
            tensor(&[3], 27, 1.0),
        ],
        "max_pool2d_forward" => vec![tensor(&[1, 2, 6, 6], 28, 2.0)],
        "layer_norm_backward" => vec![
            tensor(&[3, 8], 10, 2.0),
            positive(&[8], 11),
            tensor(&[8], 12, 0.5),
            tensor(&[3, 8], 39, 1.0),
        ],
        "channel_layer_norm_backward" => vec![
            tensor(&[2, 4, 3, 3], 13, 2.0),
            positive(&[4], 14),
            tensor(&[4], 15, 0.5),
            tensor(&[2, 4, 3, 3], 40, 1.0),
        ],
        "gelu_erf_backward" => vec![tensor(&[3, 8], 16, 2.0), tensor(&[3, 8], 41, 1.0)],
        "softmax_backward" => vec![tensor(&[3, 8], 42, 1.0), tensor(&[3, 8], 43, 1.0)],
        "broadcast_add_backward" | "broadcast_mul_backward" => vec![
            tensor(&[3, 8], 17, 2.0),
            tensor(&[8], 18, 2.0),
            tensor(&[3, 8], 44, 1.0),
        ],
        "linear_bias_backward" => vec![
            tensor(&[3, 8], 19, 1.0),
            tensor(&[5, 8], 20, 1.0),
            tensor(&[5], 21, 1.0),
            tensor(&[3, 5], 45, 1.0),
        ],
        "conv2d_backward" => vec![
            tensor(&[1, 4, 6, 6], 22, 1.0),
            tensor(&[4, 2, 3, 3], 23, 1.0),
            tensor(&[4], 24, 1.0),
            tensor(&[1, 4, 4, 4], 36, 1.0),
        ],
        "conv_transpose2d_backward" => vec![
            tensor(&[1, 4, 4, 4], 25, 1.0),
            tensor(&[4, 3, 3, 3], 26, 1.0),
            tensor(&[3], 27, 1.0),
            tensor(&[1, 3, 6, 6], 37, 1.0),
        ],
        "max_pool2d_backward" => vec![
            tensor(&[1, 2, 6, 6], 28, 2.0),
            tensor(&[1, 2, 3, 3], 38, 2.0),
        ],
        "interpolate_bilinear_forward" => vec![tensor(&[1, 2, 4, 4], 29, 2.0)],
        "rope_interleaved_forward" | "rope_interleaved_backward" => vec![
            tensor(&[1, 3, 2, 8], 30, 1.0),
            tensor(&[3, 4], 31, 1.0),
            tensor(&[3, 4], 32, 1.0),
        ],
        "concat" => vec![tensor(&[2, 3], 33, 2.0), tensor(&[2, 3], 34, 2.0)],
        "slice" => vec![tensor(&[2, 6], 35, 2.0)],
        "fill" => vec![],
        "attention_forward" | "attention_forward_saving" => {
            let [q, k, v] = attn();
            vec![q, k, v, tensor(&[1, 2, 3, 5], 4, 2.0)]
        }
        "attention_backward" => {
            let [q, k, v] = attn();
            let bias = tensor(&[1, 2, 3, 5], 4, 2.0);
            let lens = Tensor::from_u32(&[4], &[1], &budget()).unwrap();
            let opts = AttentionOpts {
                bias: Some(&bias),
                key_lens: Some(&lens),
                ..AttentionOpts::default()
            };
            let (out, lse) = raw().attention_forward_saving(&q, &k, &v, opts).unwrap();
            vec![q, k, v, out, lse, tensor(&[1, 2, 3, 8], 5, 1.0), bias]
        }
        other => panic!("{other} is in SAM_AUTOCAST_POLICY but this test has no call for it"),
    }
}

/// One call of `op` on `backend`; every tensor it returns, in order.
fn call<B: Backend>(op: &str, backend: &B, a: &[Tensor]) -> Vec<Tensor> {
    try_call(op, backend, a).unwrap_or_else(|e| panic!("{op}: {e}"))
}

/// [`call`], handing the backend's refusal back.
fn try_call<B: Backend>(op: &str, backend: &B, a: &[Tensor]) -> Result<Vec<Tensor>, OjasError> {
    let lens = Tensor::from_u32(&[4], &[1], &budget()).unwrap();
    // `seq_mean_forward`'s lengths: one per sequence of its `[2, 5, 8]` input.
    let seq_lens = Tensor::from_u32(&[5, 3], &[2], &budget()).unwrap();
    let opts = |bias| AttentionOpts {
        bias,
        key_lens: Some(&lens),
        ..AttentionOpts::default()
    };
    let one = |r: Result<Tensor, OjasError>| r.map(|t| vec![t]);
    let affine = |r: Result<AffineGrad, OjasError>| r.map(|g| vec![g.input, g.weight, g.bias]);
    match op {
        "layer_norm_forward" => one(backend.layer_norm_forward(&a[0], &a[1], &a[2], 1e-5)),
        "channel_layer_norm_forward" => {
            one(backend.channel_layer_norm_forward(&a[0], &a[1], &a[2], 1e-5))
        }
        "group_norm_forward" => one(backend.group_norm_forward(&a[0], &a[1], &a[2], 2, 1e-5)),
        "gelu_erf_forward" => one(backend.gelu_erf_forward(&a[0])),
        "relu_forward" => one(backend.relu_forward(&a[0])),
        "softmax_forward" => one(backend.softmax_forward(&a[0])),
        "scale_forward" => one(backend.scale_forward(&a[0], 0.37)),
        "broadcast_add_forward" => one(backend.broadcast_add_forward(&a[0], &a[1])),
        "broadcast_mul_forward" => one(backend.broadcast_mul_forward(&a[0], &a[1])),
        "linear_bias_forward" => one(backend.linear_bias_forward(&a[0], &a[1], &a[2])),
        "conv2d_forward" => {
            let params = Conv2dParams {
                groups: 2,
                ..Conv2dParams::new()
            };
            one(backend.conv2d_forward(&a[0], &a[1], Some(&a[2]), params))
        }
        "conv_transpose2d_forward" => one(backend.conv_transpose2d_forward(
            &a[0],
            &a[1],
            Some(&a[2]),
            ConvTranspose2dParams::new(),
        )),
        "max_pool2d_forward" => one(backend.max_pool2d_forward(&a[0], [2, 2], [2, 2])),
        "group_norm_backward" => {
            affine(backend.group_norm_backward(&a[0], &a[1], &a[2], &a[3], 2, 1e-5))
        }
        "layer_norm_backward" => {
            affine(backend.layer_norm_backward(&a[0], &a[1], &a[2], &a[3], 1e-5))
        }
        "channel_layer_norm_backward" => {
            affine(backend.channel_layer_norm_backward(&a[0], &a[1], &a[2], &a[3], 1e-5))
        }
        "gelu_erf_backward" => one(backend.gelu_erf_backward(&a[0], &a[1])),
        "gelu_tanh_forward" => one(backend.gelu_tanh_forward(&a[0])),
        "gelu_tanh_backward" => one(backend.gelu_tanh_backward(&a[0], &a[1])),
        "seq_mean_forward" => one(backend.seq_mean_forward(&a[0], &seq_lens)),
        "l2_normalize_forward" => one(backend.l2_normalize_forward(&a[0], 1e-12)),
        "softmax_backward" => one(backend.softmax_backward(&a[0], &a[1])),
        "broadcast_add_backward" => backend
            .broadcast_add_backward(&a[0], &a[1], &a[2])
            .map(|(ga, gb)| vec![ga, gb]),
        "broadcast_mul_backward" => backend
            .broadcast_mul_backward(&a[0], &a[1], &a[2])
            .map(|(ga, gb)| vec![ga, gb]),
        "linear_bias_backward" => affine(backend.linear_bias_backward(&a[0], &a[1], &a[2], &a[3])),
        "conv2d_backward" => {
            let params = Conv2dParams {
                groups: 2,
                ..Conv2dParams::new()
            };
            backend
                .conv2d_backward(&a[0], &a[1], Some(&a[2]), &a[3], params)
                .map(|g| vec![g.input, g.weight, g.bias.unwrap()])
        }
        "conv_transpose2d_backward" => backend
            .conv_transpose2d_backward(
                &a[0],
                &a[1],
                Some(&a[2]),
                &a[3],
                ConvTranspose2dParams::new(),
            )
            .map(|g| vec![g.input, g.weight, g.bias.unwrap()]),
        "max_pool2d_backward" => one(backend.max_pool2d_backward(&a[0], &a[1], [2, 2], [2, 2])),
        "interpolate_bilinear_forward" => {
            one(backend.interpolate_bilinear_forward(&a[0], [6, 6], false))
        }
        "interpolate_bilinear_backward" => {
            one(backend.interpolate_bilinear_backward(&a[0], &a[1], false))
        }
        "rope_interleaved_forward" => one(backend.rope_interleaved_forward(&a[0], &a[1], &a[2])),
        "rope_interleaved_backward" => one(backend.rope_interleaved_backward(&a[0], &a[1], &a[2])),
        "concat" => one(backend.concat(&[&a[0], &a[1]], 1)),
        "slice" => one(backend.slice(&a[0], 1, 1, 3)),
        "fill" => one(backend.fill(&[2, 3], DType::F32, f32::from_bits(0x3f80_0001))),
        "attention_forward" => {
            one(backend.attention_forward(&a[0], &a[1], &a[2], opts(Some(&a[3]))))
        }
        "attention_forward_saving" => backend
            .attention_forward_saving(&a[0], &a[1], &a[2], opts(Some(&a[3])))
            .map(|(out, lse)| vec![out, lse]),
        "attention_backward" => backend
            .attention_backward(
                &a[0],
                &a[1],
                &a[2],
                &a[3],
                &a[4],
                &a[5],
                opts(Some(&a[6])),
                true,
            )
            .map(|g| vec![g.q, g.k, g.v, g.bias.unwrap()]),
        other => panic!("{other} is in SAM_AUTOCAST_POLICY but this test has no call for it"),
    }
}

/// `args` with the operands at `at` replaced by their bf16 rounding.
fn rounded_at(args: &[Tensor], at: impl Fn(usize) -> bool) -> Vec<Tensor> {
    args.iter()
        .enumerate()
        .map(|(i, t)| if at(i) { round_all(t) } else { t.clone() })
        .collect()
}

/// `args` as a bf16 region tags them: rounded, with `bf16_exact` set, at the
/// operands for which `at` holds; the rest untouched.
fn tagged_at<B: Backend>(
    auto: &Autocast<B>,
    args: &[Tensor],
    at: impl Fn(usize) -> bool,
) -> Vec<Tensor> {
    args.iter()
        .enumerate()
        .map(|(i, t)| {
            if at(i) {
                auto.cast_bf16(t).unwrap()
            } else {
                t.clone()
            }
        })
        .collect()
}

fn assert_same(op: &str, what: &str, got: &[Tensor], want: &[Tensor]) {
    assert_eq!(got.len(), want.len(), "{op}: {what}: output count");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.shape(), w.shape(), "{op}: {what}: output {i} shape");
        assert_eq!(bits(g), bits(w), "{op}: {what}: output {i} bits");
    }
}

fn assert_tags(op: &str, what: &str, out: &[Tensor], tagged: impl Fn(usize) -> bool) {
    for (i, t) in out.iter().enumerate() {
        assert_eq!(
            t.bf16_exact(),
            tagged(i),
            "{op}: {what}: output {i} bf16_exact"
        );
    }
}

fn check(op: &str, class: AutocastClass) {
    let plain = raw();
    let auto = Autocast::new(&plain);
    let args = arguments(op);
    let reference = call(op, &plain, &args);
    // The fixtures must hold values rounding would change, or a class that
    // rounds could not be told from one that does not.
    for t in args.iter().chain(&reference) {
        assert!(!all_bf16_values(t), "{op}: a fixture tensor is bf16-exact");
    }

    // Outside a region every class leaves the call alone, whatever the tags.
    let region_off = call(op, &auto, &args);
    assert_same(op, "off, untagged", &region_off, &reference);
    assert_tags(op, "off, untagged", &region_off, |_| false);

    let _region = auto.autocast_region(AutocastMode::Bf16).unwrap();
    let n = args.len();
    match class {
        AutocastClass::Matmul {
            f32_operands,
            f32_outputs,
        } => {
            let want_in = rounded_at(&args, |i| !f32_operands.contains(&i));
            let want = call(op, &plain, &want_in);
            let want: Vec<Tensor> = want
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    if f32_outputs.contains(&i) {
                        t.clone()
                    } else {
                        round_all(t)
                    }
                })
                .collect();
            let emitted = |i: usize| !f32_outputs.contains(&i);
            let untagged = call(op, &auto, &args);
            assert_same(op, "untagged operands", &untagged, &want);
            assert_tags(op, "untagged operands", &untagged, emitted);
            // Operands the region already rounded are not rounded again.
            let tagged_args = tagged_at(&auto, &args, |i| !f32_operands.contains(&i));
            let tagged = call(op, &auto, &tagged_args);
            assert_same(op, "tagged operands", &tagged, &want);
            assert_tags(op, "tagged operands", &tagged, emitted);
            // Rounding must be visible, or the check proves nothing.
            assert_ne!(
                bits(&untagged[0]),
                bits(&reference[0]),
                "{op}: no rounding seen"
            );
        }
        AutocastClass::Pointwise { tag_from } => {
            let gate = |i: usize| tag_from.contains(&i);
            let untagged = call(op, &auto, &args);
            assert_same(op, "untagged operands", &untagged, &reference);
            assert_tags(op, "untagged operands", &untagged, |_| false);
            // Tagged exactly where the policy says the tag gates rounding:
            // the output is that call's result rounded, and tagged.
            let gated_args = tagged_at(&auto, &args, gate);
            let want: Vec<Tensor> = call(op, &plain, &gated_args)
                .iter()
                .map(round_all)
                .collect();
            let gated = call(op, &auto, &gated_args);
            assert_same(op, "gating operands tagged", &gated, &want);
            assert_tags(op, "gating operands tagged", &gated, |_| true);
            // Tagged everywhere gives the same.
            let all_args = tagged_at(&auto, &args, |_| true);
            let want_all: Vec<Tensor> = call(op, &plain, &all_args).iter().map(round_all).collect();
            let all = call(op, &auto, &all_args);
            assert_same(op, "every operand tagged", &all, &want_all);
            assert_tags(op, "every operand tagged", &all, |_| true);
            // Every gating operand must be tagged: leave one out.
            for &left_out in tag_from {
                let part = tagged_at(&auto, &args, |i| gate(i) && i != left_out);
                let out = call(op, &auto, &part);
                let want = call(op, &plain, &part);
                assert_same(op, "a gating operand untagged", &out, &want);
                assert_tags(op, "a gating operand untagged", &out, |_| false);
            }
            assert_ne!(
                bits(&gated[0]),
                bits(&reference[0]),
                "{op}: no rounding seen"
            );
        }
        AutocastClass::Reduction => {
            let untagged = call(op, &auto, &args);
            assert_same(op, "untagged operands", &untagged, &reference);
            assert_tags(op, "untagged operands", &untagged, |_| false);
            // Tagged operands do not make the output rounded or tagged.
            let all_args = tagged_at(&auto, &args, |_| true);
            let want = call(op, &plain, &all_args);
            let tagged = call(op, &auto, &all_args);
            assert_same(op, "tagged operands", &tagged, &want);
            assert_tags(op, "tagged operands", &tagged, |_| false);
            assert!(!all_bf16_values(&tagged[0]), "{op}: output was rounded");
        }
        AutocastClass::Move => {
            let untagged = call(op, &auto, &args);
            assert_same(op, "untagged operands", &untagged, &reference);
            assert_tags(op, "untagged operands", &untagged, |_| false);
            // Values move unchanged; the tag follows only when every input
            // carries it.
            let all_args = tagged_at(&auto, &args, |_| true);
            let want = call(op, &plain, &all_args);
            let tagged = call(op, &auto, &all_args);
            assert_same(op, "tagged operands", &tagged, &want);
            assert_tags(op, "tagged operands", &tagged, |_| true);
            for left_out in 0..n {
                let part = tagged_at(&auto, &args, |i| i != left_out);
                let out = call(op, &auto, &part);
                assert_same(op, "an operand untagged", &out, &call(op, &plain, &part));
                assert_tags(op, "an operand untagged", &out, |_| false);
            }
        }
        AutocastClass::Source => {
            assert_eq!(n, 0, "{op}: a source takes no tensor");
            let out = call(op, &auto, &args);
            assert_same(op, "in a region", &out, &reference);
            assert_tags(op, "in a region", &out, |_| false);
        }
    }
}

#[test]
fn every_backend_method_has_one_autocast_policy_and_every_sam_op_follows_it() {
    // Completeness: each Backend method is classified here exactly once, in
    // the table or in the list of the ops outside the SAM set.
    for &method in BACKEND_METHODS {
        let policy = SAM_AUTOCAST_POLICY
            .iter()
            .filter(|(op, _)| *op == method)
            .count();
        let outside = OUTSIDE_THE_SAM_SET
            .iter()
            .filter(|op| **op == method)
            .count();
        assert_eq!(
            policy + outside,
            1,
            "Backend::{method} must appear exactly once in SAM_AUTOCAST_POLICY \
             (ojas-core/src/autocast.rs) or OUTSIDE_THE_SAM_SET (this file); \
             found {policy} and {outside}"
        );
    }
    for op in SAM_AUTOCAST_POLICY
        .iter()
        .map(|(op, _)| op)
        .chain(OUTSIDE_THE_SAM_SET)
    {
        assert!(
            BACKEND_METHODS.contains(op),
            "{op} is named in the policy but is not a Backend method"
        );
    }
    // Behaviour: every op follows its class.
    for &(op, class) in SAM_AUTOCAST_POLICY {
        if REFUSED_ON_CPU_UNTIL_IMPLEMENTED.contains(&op) {
            continue;
        }
        check(op, class);
    }
}

/// The ops [`REFUSED_ON_CPU_UNTIL_IMPLEMENTED`] names are in the policy, and
/// each is still the trait default's refusal, directly and through
/// `Autocast` in a bf16 region, under its own name. Their class check is the
/// loop above; this keeps a skipped check from also hiding an implementation
/// that landed without removing the name.
#[test]
fn ops_without_a_cpu_implementation_refuse_and_have_a_policy_entry() {
    for &op in REFUSED_ON_CPU_UNTIL_IMPLEMENTED {
        assert!(
            SAM_AUTOCAST_POLICY.iter().any(|(name, _)| *name == op),
            "{op} is listed as not yet on the CPU but has no policy entry"
        );
        let plain = raw();
        let auto = Autocast::new(&plain);
        let args = arguments(op);
        let refused = |got: Result<Vec<Tensor>, OjasError>, how: &str| match got {
            Err(OjasError::Unsupported { op: name, .. }) => {
                assert_eq!(name, op, "{how}: refused under another op's name");
            }
            Ok(_) => panic!(
                "{op}: CpuBackend implements it ({how}): remove the name from \
                 REFUSED_ON_CPU_UNTIL_IMPLEMENTED so its class is checked"
            ),
            Err(other) => panic!("{op}: {how}: expected Unsupported, got {other:?}"),
        };
        refused(try_call(op, &plain, &args), "direct");
        refused(try_call(op, &auto, &args), "Autocast, region off");
        let _region = auto.autocast_region(AutocastMode::Bf16).unwrap();
        refused(try_call(op, &auto, &args), "Autocast, bf16 region");
    }
}

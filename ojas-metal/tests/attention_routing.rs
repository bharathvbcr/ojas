//! `attention_forward` on Metal routes plain self-attention at head dim 64
//! (no bias, key lengths, window or causal mask, `Tq == Tk`, equal head
//! counts) to tessl's tiled `encoder_attn`; every other call stays on
//! `ojas_attn_gen_fwd`. These tests pin the route, the parity of the routed
//! path against the CPU reference and against the general kernel at the SAM
//! vision shapes, and the fault contract (a non-finite operand is reported
//! under `attention_forward`).
//!
//! The route is observed through [`MetalBackend::dispatches`]: the routed
//! path encodes three permutes in, the kernel and one permute out, where the
//! general kernel is a single dispatch.

#![cfg(target_os = "macos")]

mod common;

use std::time::Instant;

use common::*;
use ojas_core::{AttentionOpts, Backend, OjasError, Tensor};
use ojas_metal::MetalBackend;

fn tol(tk: usize) -> f32 {
    4e-5 * (tk as f32).sqrt().max(1.0)
}

fn qkv(b: usize, h: usize, t: usize, d: usize, seed: u64) -> [Tensor; 3] {
    [
        rand(&[b, h, t, d], seed, 1.0),
        rand(&[b, h, t, d], seed + 1, 1.0),
        rand(&[b, h, t, d], seed + 2, 1.0),
    ]
}

/// Dispatches one `attention_forward` encodes, after the backend is idle.
fn dispatches_of(
    m: &MetalBackend,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    opts: AttentionOpts<'_>,
) -> u64 {
    ok("sync", m.sync());
    let before = m.dispatches();
    let out = ok("attention_forward", m.attention_forward(q, k, v, opts));
    ok("sync", m.sync());
    drop(out);
    m.dispatches() - before
}

#[test]
fn routed_self_attention_matches_cpu_on_small_and_vision_head_slices() {
    routed_matches_cpu(64);
}

#[test]
fn routed_self_attention_at_head_dim_32_matches_cpu() {
    routed_matches_cpu(32);
}

fn routed_matches_cpu(d: usize) {
    let (m, c) = (metal(), cpu());
    for (i, (b, h, t)) in [
        (1usize, 1usize, 1usize),
        (2, 3, 33),
        (1, 4, 64),
        (9, 2, 576),
        // One head of the global layers' 72 x 72 grid.
        (1, 1, 5184),
    ]
    .into_iter()
    .enumerate()
    {
        let seed = 10 * i as u64;
        let [q, k, v] = qkv(b, h, t, d, seed + d as u64);
        let want = ok(
            "cpu",
            c.attention_forward(&q, &k, &v, AttentionOpts::default()),
        );
        let got = ok(
            "metal",
            m.attention_forward(
                &up(&m, &q),
                &up(&m, &k),
                &up(&m, &v),
                AttentionOpts::default(),
            ),
        );
        ok("sync", m.sync());
        same_tensor(&format!("[{b},{h},{t},{d}]"), &got, &want, tol(t), 1e-4);
    }
}

#[test]
fn routed_self_attention_matches_the_general_kernel_at_the_sam_shapes() {
    routed_matches_general(64, &[(9, 16, 576), (1, 16, 5184)]);
}

/// The detector's head dim: 8 heads of 32 over the 72 x 72 grid, and a
/// windowed shape.
#[test]
fn routed_self_attention_at_head_dim_32_matches_the_general_kernel() {
    routed_matches_general(32, &[(9, 8, 576), (1, 8, 5184)]);
}

fn routed_matches_general(d: usize, shapes: &[(usize, usize, usize)]) {
    let m = metal();
    // `attention_forward_saving` never routes, so it is the general kernel.
    for &(b, h, t) in shapes {
        let [q, k, v] = qkv(b, h, t, d, 100 + t as u64);
        let (dq, dk, dv) = (up(&m, &q), up(&m, &k), up(&m, &v));
        let (want, _lse) = ok(
            "general",
            m.attention_forward_saving(&dq, &dk, &dv, AttentionOpts::default()),
        );
        let got = ok(
            "routed",
            m.attention_forward(&dq, &dk, &dv, AttentionOpts::default()),
        );
        ok("sync", m.sync());
        let want_host = ok("down", want.to_host(&ojas_core::Budget::new(8 * GIB)));
        same_tensor(
            &format!("[{b},{h},{t},{d}]"),
            &got,
            &want_host,
            tol(t),
            1e-4,
        );
    }
}

#[test]
fn offset_device_views_take_the_routed_path_correctly() {
    let (m, c) = (metal(), cpu());
    let [q, k, v] = qkv(2, 2, 40, 64, 7);
    let want = ok(
        "cpu",
        c.attention_forward(&q, &k, &v, AttentionOpts::default()),
    );
    let got = ok(
        "metal",
        m.attention_forward(
            &offset_view(&m, &q, 3, 20),
            &offset_view(&m, &k, 5, 21),
            &offset_view(&m, &v, 7, 22),
            AttentionOpts::default(),
        ),
    );
    ok("sync", m.sync());
    same_tensor("views", &got, &want, tol(40), 1e-4);
}

#[test]
fn the_route_is_taken_for_plain_self_attention_and_for_nothing_else() {
    let m = metal();
    let [q, k, v] = qkv(1, 2, 64, 64, 1);
    let (dq, dk, dv) = (up(&m, &q), up(&m, &k), up(&m, &v));
    let plain = dispatches_of(&m, &dq, &dk, &dv, AttentionOpts::default());

    // Grouped heads (2 query heads over 1 KV head) keep the general kernel.
    let kg = up(&m, &rand(&[1, 1, 64, 64], 2, 1.0));
    let vg = up(&m, &rand(&[1, 1, 64, 64], 3, 1.0));
    let general = dispatches_of(&m, &dq, &kg, &vg, AttentionOpts::default());

    if m.has_tensorops() {
        // Routed: three permutes in, the kernel, one out; the general kernel
        // is one dispatch with its two output checks.
        assert!(plain > general, "routed {plain} vs general {general}");
    } else {
        assert_eq!(plain, general, "no TensorOps: nothing may route");
    }

    // Head dim 32 is routed too: more dispatches than the general kernel.
    let (q32, k32, v32) = (
        up(&m, &rand(&[1, 2, 64, 32], 12, 1.0)),
        up(&m, &rand(&[1, 2, 64, 32], 13, 1.0)),
        up(&m, &rand(&[1, 2, 64, 32], 14, 1.0)),
    );
    let routed32 = dispatches_of(&m, &q32, &k32, &v32, AttentionOpts::default());
    let kg32 = up(&m, &rand(&[1, 1, 64, 32], 15, 1.0));
    let vg32 = up(&m, &rand(&[1, 1, 64, 32], 16, 1.0));
    let general32 = dispatches_of(&m, &q32, &kg32, &vg32, AttentionOpts::default());
    if m.has_tensorops() {
        assert!(
            routed32 > general32,
            "head dim 32: routed {routed32} vs general {general32}"
        );
    } else {
        assert_eq!(routed32, general32, "no TensorOps: nothing may route");
    }

    let bias = up(&m, &rand(&[1, 2, 64, 64], 4, 1.0));
    let lens = up(&m, &host_u32(&[64], &[1]));
    // Tq < Tk is routed up to Tk = 2.25 Tq (Q is padded); Tq > Tk and a Tk past
    // the bound are not.
    let q_short = up(&m, &rand(&[1, 2, 40, 64], 5, 1.0));
    let q_long = up(&m, &rand(&[1, 2, 80, 64], 5, 1.0));
    let q_tiny = up(&m, &rand(&[1, 2, 4, 64], 5, 1.0));
    let routed_cross = dispatches_of(&m, &q_short, &dk, &dv, AttentionOpts::default());
    if m.has_tensorops() {
        assert!(
            routed_cross > general,
            "Tq 40 over Tk 64 routes: {routed_cross} vs general {general}"
        );
    }
    let (q128, k128, v128) = (
        up(&m, &rand(&[1, 2, 64, 128], 6, 1.0)),
        up(&m, &rand(&[1, 2, 64, 128], 7, 1.0)),
        up(&m, &rand(&[1, 2, 64, 128], 8, 1.0)),
    );
    let (q16, k16, v16) = (
        up(&m, &rand(&[1, 2, 64, 16], 9, 1.0)),
        up(&m, &rand(&[1, 2, 64, 16], 10, 1.0)),
        up(&m, &rand(&[1, 2, 64, 16], 11, 1.0)),
    );
    let stays: Vec<(&str, u64)> = vec![
        (
            "bias",
            dispatches_of(
                &m,
                &dq,
                &dk,
                &dv,
                AttentionOpts {
                    bias: Some(&bias),
                    ..AttentionOpts::default()
                },
            ),
        ),
        (
            "key lengths",
            dispatches_of(
                &m,
                &dq,
                &dk,
                &dv,
                AttentionOpts {
                    key_lens: Some(&lens),
                    ..AttentionOpts::default()
                },
            ),
        ),
        (
            "causal",
            dispatches_of(
                &m,
                &dq,
                &dk,
                &dv,
                AttentionOpts {
                    causal: true,
                    ..AttentionOpts::default()
                },
            ),
        ),
        (
            "causal window",
            dispatches_of(
                &m,
                &dq,
                &dk,
                &dv,
                AttentionOpts {
                    causal: true,
                    window: Some(8),
                    ..AttentionOpts::default()
                },
            ),
        ),
        (
            "Tq > Tk",
            dispatches_of(&m, &q_long, &dk, &dv, AttentionOpts::default()),
        ),
        (
            "Tk past the pad bound",
            dispatches_of(&m, &q_tiny, &dk, &dv, AttentionOpts::default()),
        ),
        (
            "head dim 128",
            dispatches_of(&m, &q128, &k128, &v128, AttentionOpts::default()),
        ),
        (
            "head dim 16",
            dispatches_of(&m, &q16, &k16, &v16, AttentionOpts::default()),
        ),
    ];
    for (what, n) in stays {
        // The general kernel, with at most a bias or key-length scan more.
        assert!(
            n <= general + 1,
            "{what}: {n} dispatches, general kernel {general}, routed {plain}"
        );
    }
}

#[test]
fn the_calls_that_stay_on_the_general_kernel_still_match_cpu() {
    let (m, c) = (metal(), cpu());
    let [q, k, v] = qkv(2, 2, 40, 64, 31);
    let bias = rand(&[2, 2, 40, 40], 34, 1.0);
    let lens = host_u32(&[40, 9], &[2]);
    let cases = [
        AttentionOpts {
            bias: Some(&bias),
            ..AttentionOpts::default()
        },
        AttentionOpts {
            key_lens: Some(&lens),
            ..AttentionOpts::default()
        },
        AttentionOpts {
            causal: true,
            ..AttentionOpts::default()
        },
    ];
    for (i, opts) in cases.iter().enumerate() {
        let want = ok("cpu", c.attention_forward(&q, &k, &v, *opts));
        let dev_bias = opts.bias.map(|b| up(&m, b));
        let dev_lens = opts.key_lens.map(|l| up(&m, l));
        let dev = AttentionOpts {
            causal: opts.causal,
            window: opts.window,
            bias: dev_bias.as_ref(),
            key_lens: dev_lens.as_ref(),
        };
        let got = ok(
            "metal",
            m.attention_forward(&up(&m, &q), &up(&m, &k), &up(&m, &v), dev),
        );
        ok("sync", m.sync());
        same_tensor(&format!("case {i}"), &got, &want, tol(40), 1e-4);
    }
}

#[test]
fn a_non_finite_operand_faults_under_the_name_that_was_called_on_the_routed_path() {
    let m = metal();
    for d in [64usize, 32] {
        non_finite_faults(&m, d);
    }
}

fn non_finite_faults(m: &MetalBackend, d: usize) {
    // Tq == Tk, then Tq < Tk (the padded route: `ojas_pad_q` runs under the
    // fault); the general kernel's contract is the same for both.
    for (tq, tk) in [(64usize, 64usize), (24, 64)] {
        let q = rand(&[1, 2, tq, d], 41, 1.0);
        let k = rand(&[1, 2, tk, d], 42, 1.0);
        let v = rand(&[1, 2, tk, d], 43, 1.0);
        for (what, which) in [("q", 0usize), ("k", 1), ("v", 2)] {
            for bad in [f32::NAN, f32::INFINITY] {
                let mut ops = [q.clone(), k.clone(), v.clone()];
                let mut data = ok("host", ops[which].to_f32_vec());
                data[5] = bad;
                ops[which] = host(&data, ops[which].shape());
                let r = m.attention_forward(
                    &up(m, &ops[0]),
                    &up(m, &ops[1]),
                    &up(m, &ops[2]),
                    AttentionOpts::default(),
                );
                deferred(
                    m,
                    &format!("{bad} in {what}, D{d} {tq}x{tk}"),
                    r,
                    "attention_forward",
                );
            }
        }
    }
}

/// tessl's `encoder_attn` is compiled with fast math. With q and k scaled so
/// the scores reach the hundreds (`exp` of them overflows f32 unless the row
/// maximum is subtracted first) the routed output must still match the CPU,
/// which is finite throughout, and stay finite.
#[test]
fn routed_attention_with_scores_past_the_exp_overflow_matches_cpu() {
    for d in [64usize, 32] {
        large_scores_match_cpu(d);
    }
}

fn large_scores_match_cpu(d: usize) {
    let (m, c) = (metal(), cpu());
    let (b, h, t) = (2usize, 2usize, 128usize);
    let q = rand(&[b, h, t, d], 91, 15.0);
    let k = rand(&[b, h, t, d], 92, 15.0);
    let v = rand(&[b, h, t, d], 93, 1.0);
    let want = ok(
        "cpu",
        c.attention_forward(&q, &k, &v, AttentionOpts::default()),
    );
    let scores = ok("q", q.to_f32_vec());
    assert!(
        scores.iter().any(|x| x.abs() > 14.0),
        "the inputs must be large for this test to mean anything"
    );
    // The general kernel's dispatches, on fresh uploads, as the yardstick.
    let (gq, gk, gv) = (up(&m, &q), up(&m, &k), up(&m, &v));
    ok("sync", m.sync());
    let general_before = m.dispatches();
    let _saved = ok(
        "saving",
        m.attention_forward_saving(&gq, &gk, &gv, AttentionOpts::default()),
    );
    ok("sync", m.sync());
    let general = m.dispatches() - general_before;
    let (rq, rk, rv) = (up(&m, &q), up(&m, &k), up(&m, &v));
    let routed_before = m.dispatches();
    let got = ok(
        "metal",
        m.attention_forward(&rq, &rk, &rv, AttentionOpts::default()),
    );
    ok("sync", m.sync());
    if m.has_tensorops() {
        assert!(
            m.dispatches() - routed_before > general,
            "the routed path ran"
        );
    }
    same_tensor("large scores", &got, &want, 1e-3, 1e-3);
}

/// The routed path holds, besides the output, four permuted copies and a
/// length per batch row. A budget with room for the general kernel's output
/// and log-sum-exp but not for that scratch must run the general kernel, not
/// fail with `CapacityExceeded`, and leave nothing charged behind.
#[test]
fn a_budget_that_holds_the_output_but_not_the_routed_scratch_runs_the_general_kernel() {
    for d in [64usize, 32] {
        budget_fallback(d);
    }
}

fn budget_fallback(d: usize) {
    let (b, h, t) = (1usize, 2usize, 256usize);
    let n = b * h * t * d;
    let [q, k, v] = qkv(b, h, t, d, 77);
    let want = ok(
        "cpu",
        cpu().attention_forward(&q, &k, &v, AttentionOpts::default()),
    );

    // Inputs, the output and the log-sum-exp, and no more than 1 KiB over:
    // the routed scratch (4n + B values) needs 512 KiB more.
    let room = ((3 * n + n + b * h * t) * 4 + 1024) as u64;
    let tight = MetalBackend::new(ojas_core::Budget::new(room))
        .unwrap_or_else(|e| panic!("Metal device unavailable: {e:?}"));
    let (dq, dk, dv) = (up(&tight, &q), up(&tight, &k), up(&tight, &v));
    let baseline = ok("live", tight.budget().live_bytes());

    let general = dispatches_of(&tight, &dq, &dk, &dv, AttentionOpts::default());
    let saving_before = tight.dispatches();
    let (saved, lse) = ok(
        "saving",
        tight.attention_forward_saving(&dq, &dk, &dv, AttentionOpts::default()),
    );
    ok("sync", tight.sync());
    let saving = tight.dispatches() - saving_before;
    drop((saved, lse));
    assert_eq!(
        general, saving,
        "a call that did not fit the routed scratch ran the saving form's dispatches"
    );

    let got = ok(
        "attention_forward under a tight budget",
        tight.attention_forward(&dq, &dk, &dv, AttentionOpts::default()),
    );
    ok("sync", tight.sync());
    same_tensor("tight budget", &got, &want, tol(t), 1e-4);
    drop(got);
    assert_eq!(
        ok("live", tight.budget().live_bytes()),
        baseline,
        "the refused reservation left nothing charged"
    );

    // With room the same call routes: more dispatches than the general kernel.
    if tight.has_tensorops() {
        let roomy = metal();
        let (rq, rk, rv) = (up(&roomy, &q), up(&roomy, &k), up(&roomy, &v));
        let routed = dispatches_of(&roomy, &rq, &rk, &rv, AttentionOpts::default());
        assert!(
            routed > general,
            "routed {routed} dispatches vs general {general}"
        );
    }
}

/// Cross-attention (`Tq < Tk`) routes by zero-padding Q to `Tk` rows and
/// keeping the first `Tq` output rows. Parity with the CPU on small shapes
/// and on one head of the memory encoder's `[5184] x [5200, 10400]`, at both
/// routed head dims.
#[test]
fn routed_cross_attention_matches_cpu() {
    let (m, c) = (metal(), cpu());
    for d in [32usize, 64] {
        for (b, h, tq, tk) in [
            (2usize, 3usize, 17usize, 40usize),
            (1, 4, 33, 64),
            (2, 2, 100, 101),
            (1, 1, 5184, 5200),
            (1, 1, 5184, 10400),
        ] {
            let q = rand(&[b, h, tq, d], 200 + d as u64, 1.0);
            let k = rand(&[b, h, tk, d], 201 + d as u64, 1.0);
            let v = rand(&[b, h, tk, d], 202 + d as u64, 1.0);
            let want = ok(
                "cpu",
                c.attention_forward(&q, &k, &v, AttentionOpts::default()),
            );
            let (dq, dk, dv) = (up(&m, &q), up(&m, &k), up(&m, &v));
            let got = ok(
                "metal",
                m.attention_forward(&dq, &dk, &dv, AttentionOpts::default()),
            );
            ok("sync", m.sync());
            same_tensor(
                &format!("[{b},{h},{tq}x{tk},{d}]"),
                &got,
                &want,
                tol(tk),
                1e-4,
            );
        }
    }
}

/// The memory encoder's cross-attention at its real shapes, 8 heads of 32,
/// against the general kernel (`attention_forward_saving` never routes), and
/// the route taken: more dispatches than the general kernel, which fails if
/// the predicate rejects D32 cross-attention.
#[test]
fn routed_cross_attention_matches_the_general_kernel_at_the_memory_shapes() {
    let m = metal();
    for tk in [5200usize, 10400] {
        let q = rand(&[1, 8, 5184, 32], 210, 1.0);
        let k = rand(&[1, 8, tk, 32], 211, 1.0);
        let v = rand(&[1, 8, tk, 32], 212, 1.0);
        let (dq, dk, dv) = (up(&m, &q), up(&m, &k), up(&m, &v));
        ok("sync", m.sync());
        let before = m.dispatches();
        let (want, _lse) = ok(
            "general",
            m.attention_forward_saving(&dq, &dk, &dv, AttentionOpts::default()),
        );
        ok("sync", m.sync());
        let general = m.dispatches() - before;
        let before = m.dispatches();
        let got = ok(
            "routed",
            m.attention_forward(&dq, &dk, &dv, AttentionOpts::default()),
        );
        ok("sync", m.sync());
        let routed = m.dispatches() - before;
        if m.has_tensorops() {
            assert!(
                routed > general,
                "5184x{tk}: the D32 cross-attention route was not taken ({routed} vs {general})"
            );
        }
        let want_host = ok("down", want.to_host(&ojas_core::Budget::new(8 * GIB)));
        same_tensor(&format!("5184x{tk}"), &got, &want_host, tol(tk), 1e-4);
    }
}

fn opts<'a>(causal: bool, bias: Option<&'a Tensor>, lens: Option<&'a Tensor>) -> AttentionOpts<'a> {
    AttentionOpts {
        causal,
        window: None,
        bias,
        key_lens: lens,
    }
}

/// One dispatcher per shape class, pinned on both sides of each bound.
/// `Tk / Tq <= 2.25` at head dim 32 or 64 with no bias or lengths takes the
/// pad route; beyond it, and at any ratio with a bias, key lengths or head
/// dim 16, the inference form takes the split-K tiled kernel at head dim 16
/// and 32 (one dispatch more than the scalar kernel; at head dim 32 with no
/// bias or lengths, from `flash_min_groups` threadgroups on, the flash kernel,
/// one dispatch, which the `Tiled` class stands for there) and the scalar
/// kernel at head dim 64. Which one ran is read off the dispatches of the call against
/// the same call made `causal` (always scalar) and against the saving form
/// (never routed), and every call is checked against the CPU.
#[test]
fn the_pad_ratio_bound_and_the_tiled_kernel_split_the_shape_classes() {
    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Ran {
        Pad,
        Tiled,
        Scalar,
    }
    let (m, c) = (metal(), cpu());
    // (head dim, Tq, Tk, bias, lens, expected). 2.25 Tq is exactly 9 for Tq 4.
    let cases = [
        (32usize, 4usize, 9usize, false, false, Ran::Pad),
        (32, 4, 10, false, false, Ran::Tiled),
        (64, 4, 9, false, false, Ran::Pad),
        (64, 4, 10, false, false, Ran::Scalar),
        (32, 8, 18, false, false, Ran::Pad),
        (32, 8, 19, false, false, Ran::Tiled),
        (32, 40, 64, false, false, Ran::Pad),
        (32, 40, 64, true, false, Ran::Tiled),
        (32, 40, 64, false, true, Ran::Tiled),
        (16, 40, 64, false, false, Ran::Tiled),
        (16, 40, 64, true, false, Ran::Tiled),
        (64, 40, 64, true, false, Ran::Scalar),
    ];
    for (i, (d, tq, tk, with_bias, with_lens, expected)) in cases.into_iter().enumerate() {
        let tag = format!("D{d} {tq}x{tk} bias {with_bias} lens {with_lens}");
        let seed = 230 + 5 * i as u64;
        let q = rand(&[2, 2, tq, d], seed, 1.0);
        let k = rand(&[2, 2, tk, d], seed + 1, 1.0);
        let v = rand(&[2, 2, tk, d], seed + 2, 1.0);
        let bias = with_bias.then(|| rand(&[2, 1, tq, tk], seed + 3, 1.0));
        let lens = with_lens.then(|| host_u32(&[tk as u32, (tk / 2) as u32], &[2]));
        let want = ok(
            "cpu",
            c.attention_forward(&q, &k, &v, opts(false, bias.as_ref(), lens.as_ref())),
        );
        let (dq, dk, dv) = (up(&m, &q), up(&m, &k), up(&m, &v));
        let dbias = bias.as_ref().map(|t| up(&m, t));
        let dlens = lens.as_ref().map(|t| up(&m, t));
        // Dispatches of a call and the flash and scalar kernels it encoded
        // (both are one dispatch, so only the kernel names tell them apart).
        let count = |f: &dyn Fn()| {
            ok("sync", m.sync());
            let before = m.dispatches();
            let kernels = (
                m.kernel_dispatches("ojas_attn_fwd_d32"),
                m.kernel_dispatches("ojas_attn_gen_fwd"),
            );
            f();
            ok("sync", m.sync());
            (
                m.dispatches() - before,
                m.kernel_dispatches("ojas_attn_fwd_d32") - kernels.0,
                m.kernel_dispatches("ojas_attn_gen_fwd") - kernels.1,
            )
        };
        let (plain, plain_flash, plain_scalar) = count(&|| {
            drop(ok(
                "inference",
                m.attention_forward(&dq, &dk, &dv, opts(false, dbias.as_ref(), dlens.as_ref())),
            ));
        });
        let out = ok(
            "inference",
            m.attention_forward(&dq, &dk, &dv, opts(false, dbias.as_ref(), dlens.as_ref())),
        );
        let (scalar, _, _) = count(&|| {
            drop(ok(
                "scalar",
                m.attention_forward_saving(
                    &dq,
                    &dk,
                    &dv,
                    opts(true, dbias.as_ref(), dlens.as_ref()),
                ),
            ));
        });
        let (saving, saving_flash, saving_scalar) = count(&|| {
            drop(ok(
                "saving",
                m.attention_forward_saving(
                    &dq,
                    &dk,
                    &dv,
                    opts(false, dbias.as_ref(), dlens.as_ref()),
                ),
            ));
        });
        // Head dim 32 with no bias or lengths takes the flash kernel (one
        // dispatch, as the scalar kernel) from `flash_min_groups` threadgroups
        // of 32 query rows; these are B = 2, H = 2.
        let flash = d == 32
            && !with_bias
            && !with_lens
            && (tq.div_ceil(32) * 4) as u64 >= MetalBackend::flash_min_groups();
        if m.has_tensorops() {
            let ran = if plain == scalar {
                if flash {
                    Ran::Tiled
                } else {
                    Ran::Scalar
                }
            } else if plain == scalar + 1 {
                Ran::Tiled
            } else {
                Ran::Pad
            };
            assert_eq!(
                ran, expected,
                "{tag}: {plain} dispatches; scalar {scalar}, saving form {saving}"
            );
            // The saving form never routes: tiled at head dim 16 and 32,
            // scalar at 64.
            assert_eq!(
                saving,
                if d == 64 || flash { scalar } else { scalar + 1 },
                "{tag}"
            );
            // The kernel names settle what the dispatch counts cannot: flash
            // and scalar are both one dispatch, so a scalar misroute at a
            // lowered floor would pass the counts alone.
            assert_eq!(
                (saving_flash, saving_scalar),
                (u64::from(flash), u64::from(d == 64)),
                "{tag}: the saving form's kernels (flash, scalar)"
            );
            if expected != Ran::Pad {
                assert_eq!(
                    (plain_flash, plain_scalar),
                    (u64::from(flash), u64::from(expected == Ran::Scalar)),
                    "{tag}: the inference form's kernels (flash, scalar)"
                );
            }
        }
        same_tensor(&tag, &out, &want, tol(tk), 1e-4);
    }
}

/// Metal refuses every empty operand before it routes (backend.rs `arg`:
/// "empty tensor", `OjasError::Shape` under the called op), where the CPU
/// returns an empty output; the difference is tracked on a card and not
/// changed here. A device tensor cannot be empty by upload, so the operands
/// are zero-row views of a real buffer. Both forms refuse alike, at both
/// routed head dims, whichever operand is empty.
#[test]
fn empty_operands_are_refused_with_a_shape_error_before_routing() {
    let m = metal();
    for d in [32usize, 64] {
        let base = up(&m, &rand(&[1, 2, 4, d], 240, 1.0));
        // Row-major strides of [1, 2, 0, d].
        let empty = ok("view", base.view(&[1, 2, 0, d], &[0, 0, d, 1], 0));
        let real = up(&m, &rand(&[1, 2, 64, d], 241, 1.0));
        for (what, q, k, v) in [
            ("0x0", &empty, &empty, &empty),
            ("0x64", &empty, &real, &real),
        ] {
            let inference = m.attention_forward(q, k, v, AttentionOpts::default());
            assert!(
                matches!(
                    inference,
                    Err(OjasError::Shape {
                        op: "attention_forward",
                        ..
                    })
                ),
                "D{d} {what}: {inference:?}"
            );
            let saving = m.attention_forward_saving(q, k, v, AttentionOpts::default());
            assert!(
                matches!(
                    saving,
                    Err(OjasError::Shape {
                        op: "attention_forward_saving",
                        ..
                    })
                ),
                "D{d} {what} (saving): {saving:?}"
            );
            ok("sync", m.sync());
        }
    }
}

/// The padded route holds Q, K, V and the `Tk`-row output beside the
/// result, so it asks for more than the general kernel; a budget that holds
/// only the general kernel's output and log-sum-exp runs the general kernel.
#[test]
fn a_cross_attention_budget_without_room_for_the_padded_scratch_runs_the_general_kernel() {
    let (b, h, tq, tk, d) = (1usize, 2usize, 200usize, 400usize, 32usize);
    let (nq, nk) = (b * h * tq * d, b * h * tk * d);
    let q = rand(&[b, h, tq, d], 220, 1.0);
    let k = rand(&[b, h, tk, d], 221, 1.0);
    let v = rand(&[b, h, tk, d], 222, 1.0);
    let want = ok(
        "cpu",
        cpu().attention_forward(&q, &k, &v, AttentionOpts::default()),
    );
    let room = ((nq + 2 * nk + nq + b * h * tq) * 4 + 1024) as u64;
    let tight = MetalBackend::new(ojas_core::Budget::new(room))
        .unwrap_or_else(|e| panic!("Metal device unavailable: {e:?}"));
    let (dq, dk, dv) = (up(&tight, &q), up(&tight, &k), up(&tight, &v));
    let baseline = ok("live", tight.budget().live_bytes());
    let general = dispatches_of(&tight, &dq, &dk, &dv, AttentionOpts::default());
    let saving_before = tight.dispatches();
    drop(ok(
        "saving",
        tight.attention_forward_saving(&dq, &dk, &dv, AttentionOpts::default()),
    ));
    ok("sync", tight.sync());
    assert_eq!(general, tight.dispatches() - saving_before);
    let got = ok(
        "tight cross",
        tight.attention_forward(&dq, &dk, &dv, AttentionOpts::default()),
    );
    ok("sync", tight.sync());
    same_tensor("tight cross", &got, &want, tol(tk), 1e-4);
    drop(got);
    assert_eq!(ok("live", tight.budget().live_bytes()), baseline);
}

/// Wall clock of the padded route against the general kernel for the memory
/// cross-attention, `q [1, 8, 5184, 32]` over `k, v [1, 8, m, 32]`:
/// minimum of 3 after 1 warm-up. Run with `--ignored --nocapture`.
#[test]
#[ignore = "timing, not a correctness check"]
fn cross_attention_padded_route_against_general_wall_clock() {
    let m = metal();
    let q = up(&m, &rand(&[1, 8, 5184, 32], 1, 1.0));
    for tk in [5200usize, 10400, 36544] {
        let (k, v) = (
            up(&m, &rand(&[1, 8, tk, 32], 2, 1.0)),
            up(&m, &rand(&[1, 8, tk, 32], 3, 1.0)),
        );
        let time = |routed: bool| {
            let mut best = f64::MAX;
            for i in 0..4 {
                ok("sync", m.sync());
                let start = Instant::now();
                if routed {
                    drop(ok(
                        "routed",
                        m.attention_forward(&q, &k, &v, AttentionOpts::default()),
                    ));
                } else {
                    drop(ok(
                        "general",
                        m.attention_forward_saving(&q, &k, &v, AttentionOpts::default()),
                    ));
                }
                ok("sync", m.sync());
                if i >= 1 {
                    best = best.min(start.elapsed().as_secs_f64() * 1e3);
                }
            }
            best
        };
        let (general, routed) = (time(false), time(true));
        eprintln!("CROSS_ATTENTION m={tk}: general {general:.1} ms, padded route {routed:.1} ms");
    }
}

/// The detector's cross-attention on the general kernel (`Tq != Tk`, a bias):
/// wall clock, minimum of 5 after 2 warm-up calls. Run with
/// `--ignored --nocapture`.
#[test]
#[ignore = "timing, not a correctness check"]
fn detector_cross_attention_wall_clock() {
    let m = metal();
    let q = up(&m, &rand(&[1, 8, 201, 32], 1, 1.0));
    let k = up(&m, &rand(&[1, 8, 5184, 32], 2, 1.0));
    let v = up(&m, &rand(&[1, 8, 5184, 32], 3, 1.0));
    let bias = up(&m, &rand(&[1, 8, 201, 5184], 4, 1.0));
    let opts = AttentionOpts {
        bias: Some(&bias),
        ..AttentionOpts::default()
    };
    let mut best = f64::MAX;
    for i in 0..7 {
        ok("sync", m.sync());
        let start = Instant::now();
        let out = ok("attention_forward", m.attention_forward(&q, &k, &v, opts));
        ok("sync", m.sync());
        let ms = start.elapsed().as_secs_f64() * 1e3;
        drop(out);
        eprintln!("detector cross-attention call {i}: {ms:.3} ms");
        if i >= 2 {
            best = best.min(ms);
        }
    }
    eprintln!("DETECTOR_CROSS_ATTENTION min of 5: {best:.3} ms");
}

/// Wall clock of the routed path against the general kernel at the
/// detector's head dim, `[1, 8, 5184, 32]` (the 72 x 72 grid, 8 heads):
/// minimum of 5 after 2 warm-up calls, each ended by a sync. The saving form
/// never routes, so it is the general kernel. Run with `--ignored
/// --nocapture`.
#[test]
#[ignore = "timing, not a correctness check"]
fn head_dim_32_routed_against_general_wall_clock() {
    let m = metal();
    let (q, k, v) = (
        up(&m, &rand(&[1, 8, 5184, 32], 1, 1.0)),
        up(&m, &rand(&[1, 8, 5184, 32], 2, 1.0)),
        up(&m, &rand(&[1, 8, 5184, 32], 3, 1.0)),
    );
    let time = |routed: bool| {
        let mut best = f64::MAX;
        for i in 0..7 {
            ok("sync", m.sync());
            let start = Instant::now();
            if routed {
                drop(ok(
                    "routed",
                    m.attention_forward(&q, &k, &v, AttentionOpts::default()),
                ));
            } else {
                drop(ok(
                    "general",
                    m.attention_forward_saving(&q, &k, &v, AttentionOpts::default()),
                ));
            }
            ok("sync", m.sync());
            if i >= 2 {
                best = best.min(start.elapsed().as_secs_f64() * 1e3);
            }
        }
        best
    };
    let (general, routed) = (time(false), time(true));
    eprintln!("D32_ROUTING [1,8,5184,32]: general {general:.2} ms, routed {routed:.2} ms");
}

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
use ojas_core::{AttentionOpts, Backend, Tensor};
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
        let [q, k, v] = qkv(b, h, t, 64, seed);
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
        same_tensor(&format!("[{b},{h},{t},64]"), &got, &want, tol(t), 1e-4);
    }
}

#[test]
fn routed_self_attention_matches_the_general_kernel_at_the_sam_shapes() {
    let m = metal();
    // `attention_forward_saving` never routes, so it is the general kernel.
    for (b, h, t) in [(9usize, 16usize, 576usize), (1, 16, 5184)] {
        let [q, k, v] = qkv(b, h, t, 64, 100 + t as u64);
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
        same_tensor(&format!("[{b},{h},{t},64]"), &got, &want_host, tol(t), 1e-4);
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

    let bias = up(&m, &rand(&[1, 2, 64, 64], 4, 1.0));
    let lens = up(&m, &host_u32(&[64], &[1]));
    let q_short = up(&m, &rand(&[1, 2, 17, 64], 5, 1.0));
    let (q32, k32, v32) = (
        up(&m, &rand(&[1, 2, 64, 32], 6, 1.0)),
        up(&m, &rand(&[1, 2, 64, 32], 7, 1.0)),
        up(&m, &rand(&[1, 2, 64, 32], 8, 1.0)),
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
            "Tq != Tk",
            dispatches_of(&m, &q_short, &dk, &dv, AttentionOpts::default()),
        ),
        (
            "head dim 32",
            dispatches_of(&m, &q32, &k32, &v32, AttentionOpts::default()),
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
    let [q, k, v] = qkv(1, 2, 64, 64, 41);
    for (what, which) in [("q", 0usize), ("k", 1), ("v", 2)] {
        for bad in [f32::NAN, f32::INFINITY] {
            let mut ops = [q.clone(), k.clone(), v.clone()];
            let mut data = ok("host", ops[which].to_f32_vec());
            data[5] = bad;
            ops[which] = host(&data, &[1, 2, 64, 64]);
            let r = m.attention_forward(
                &up(&m, &ops[0]),
                &up(&m, &ops[1]),
                &up(&m, &ops[2]),
                AttentionOpts::default(),
            );
            deferred(&m, &format!("{bad} in {what}"), r, "attention_forward");
        }
    }
}

/// tessl's `encoder_attn` is compiled with fast math. With q and k scaled so
/// the scores reach the hundreds (`exp` of them overflows f32 unless the row
/// maximum is subtracted first) the routed output must still match the CPU,
/// which is finite throughout, and stay finite.
#[test]
fn routed_attention_with_scores_past_the_exp_overflow_matches_cpu() {
    let (m, c) = (metal(), cpu());
    let (b, h, t, d) = (2usize, 2usize, 128usize, 64usize);
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
    let (b, h, t, d) = (1usize, 2usize, 256usize, 64usize);
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

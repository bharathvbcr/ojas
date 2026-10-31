//! The general attention backward on Metal (`attention_backward`, kernels in
//! `attn_gen_bwd.metal` over `attn_bwd_tiled.h`) against the CPU reference:
//! bidirectional and cross-attention (`Tq != Tk`), grouped-query heads, a bias
//! in every broadcast layout with its gradient, key lengths, and the causal /
//! window masks, at every compiled head-dim width and at widths it pads, at
//! the SAM shapes; plus determinism, the dispatch count, and the error
//! contract.
//!
//! Tolerance, set before measuring: a gradient element is within `2e-5 *
//! sqrt(max(Tq, Tk))` absolute plus `1e-3` relative of the CPU `Exact`
//! backward, the precedent of `attention_backward.rs` for the causal
//! kernels. The CPU backward is fed the CPU forward's output and
//! log-sum-exp; the Metal backward is fed the same pair, and (a second run)
//! Metal's own forward, which agrees with it to rounding.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use ojas_core::{AttentionGrad, AttentionOpts, Backend, BackendId, Budget, OjasError, Tensor};
use ojas_metal::MetalBackend;

#[derive(Clone, Debug)]
struct Case {
    b: usize,
    h: usize,
    hkv: usize,
    tq: usize,
    tk: usize,
    d: usize,
    causal: bool,
    window: Option<usize>,
    /// `[Bb, Hb, Tqb]`.
    bias: Option<[usize; 3]>,
    lens: Option<Vec<u32>>,
    bias_grad: bool,
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
            causal: false,
            window: None,
            bias: None,
            lens: None,
            bias_grad: false,
        }
    }
    fn causal(mut self, window: Option<usize>) -> Self {
        self.causal = true;
        self.window = window;
        self
    }
    /// A bias of these extents, with its gradient.
    fn bias(mut self, shape: [usize; 3]) -> Self {
        self.bias = Some(shape);
        self.bias_grad = true;
        self
    }
    fn lens(mut self, lens: &[u32]) -> Self {
        self.lens = Some(lens.to_vec());
        self
    }
}

/// Host operands of a case, or the same on a device.
struct Operands {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    gy: Tensor,
    bias: Option<Tensor>,
    lens: Option<Tensor>,
}

impl Operands {
    fn host(c: &Case, seed: u64) -> Self {
        Operands {
            q: rand(&[c.b, c.h, c.tq, c.d], seed, 1.0),
            k: rand(&[c.b, c.hkv, c.tk, c.d], seed + 1, 1.0),
            v: rand(&[c.b, c.hkv, c.tk, c.d], seed + 2, 1.0),
            gy: rand(&[c.b, c.h, c.tq, c.d], seed + 4, 1.0),
            bias: c
                .bias
                .map(|[b, h, q]| rand(&[b, h, q, c.tk], seed + 3, 2.0)),
            lens: c.lens.as_ref().map(|l| host_u32(l, &[l.len()])),
        }
    }
    fn up(&self, m: &MetalBackend) -> Self {
        Operands {
            q: up(m, &self.q),
            k: up(m, &self.k),
            v: up(m, &self.v),
            gy: up(m, &self.gy),
            bias: self.bias.as_ref().map(|t| up(m, t)),
            lens: self.lens.as_ref().map(|t| up(m, t)),
        }
    }
    fn opts<'a>(&'a self, c: &Case) -> AttentionOpts<'a> {
        // Two struct updates, so the literals list only some of the fields
        // and survive new `AttentionOpts` fields without a needless update.
        let operands = AttentionOpts {
            bias: self.bias.as_ref(),
            key_lens: self.lens.as_ref(),
            ..Default::default()
        };
        AttentionOpts {
            causal: c.causal,
            window: c.window,
            ..operands
        }
    }
}

/// Gradient tolerance: absolute part, widened with the longest sequence.
fn tol(c: &Case) -> f32 {
    2e-5 * (c.tq.max(c.tk) as f32).sqrt().max(1.0)
}
const RTOL: f32 = 1e-3;

/// The compiled width a head dim runs at.
fn width(d: usize) -> usize {
    match d {
        0..=16 => 16,
        17..=32 => 32,
        33..=64 => 64,
        65..=128 => 128,
        _ => 256,
    }
}

fn compare(tag: &str, c: &Case, got: &AttentionGrad, want: &AttentionGrad) {
    let (atol, rtol) = (tol(c), RTOL);
    same_tensor(&format!("{tag} dq"), &got.q, &want.q, atol, rtol);
    same_tensor(&format!("{tag} dk"), &got.k, &want.k, atol, rtol);
    same_tensor(&format!("{tag} dv"), &got.v, &want.v, atol, rtol);
    match (&got.bias, &want.bias) {
        (Some(g), Some(w)) => same_tensor(&format!("{tag} dbias"), g, w, atol, rtol),
        (None, None) => {}
        (g, w) => panic!(
            "{tag}: bias gradient present on Metal {}, on the CPU {}",
            g.is_some(),
            w.is_some()
        ),
    }
}

fn check(m: &MetalBackend, c: &Case, seed: u64) {
    let cpu = cpu();
    let host = Operands::host(c, seed);
    let tag = format!("{c:?}");
    let (y, lse) = ok(
        &tag,
        cpu.attention_forward_saving(&host.q, &host.k, &host.v, host.opts(c)),
    );
    let want = ok(
        &tag,
        cpu.attention_backward(
            &host.q,
            &host.k,
            &host.v,
            &y,
            &lse,
            &host.gy,
            host.opts(c),
            c.bias_grad,
        ),
    );
    let dev = host.up(m);
    let (dy, dlse) = (up(m, &y), up(m, &lse));
    let names = ["dq", "dkv", "dbias"].map(|n| format!("ojas_attn_gen_bwd_{n}_d{}", width(c.d)));
    let before = names.each_ref().map(|n| m.kernel_dispatches(n));
    let got = m.attention_backward(
        &dev.q,
        &dev.k,
        &dev.v,
        &dy,
        &dlse,
        &dev.gy,
        dev.opts(c),
        c.bias_grad,
    );
    if !m.has_tensorops() {
        // No matrix units: an explicit refusal, never a host computation.
        assert!(
            matches!(
                got,
                Err(OjasError::Unsupported {
                    op: "attention_backward",
                    ..
                })
            ),
            "{tag}: {got:?}"
        );
        return;
    }
    let got = ok(&tag, got);
    ok("sync", m.sync());
    let ran = names.each_ref().map(|n| m.kernel_dispatches(n));
    assert_eq!(
        ran[0] - before[0],
        1,
        "{tag}: {} did not run once",
        names[0]
    );
    assert_eq!(
        ran[1] - before[1],
        1,
        "{tag}: {} did not run once",
        names[1]
    );
    assert_eq!(
        ran[2] - before[2],
        u64::from(c.bias_grad),
        "{tag}: {} ran {} times for bias_grad {}",
        names[2],
        ran[2] - before[2],
        c.bias_grad
    );
    compare(&tag, c, &got, &want);

    // Metal's own forward feeds the backward too: its output and lse agree
    // with the CPU's to rounding.
    let (my, mlse) = ok(
        &tag,
        m.attention_forward_saving(&dev.q, &dev.k, &dev.v, dev.opts(c)),
    );
    let own = ok(
        &tag,
        m.attention_backward(
            &dev.q,
            &dev.k,
            &dev.v,
            &my,
            &mlse,
            &dev.gy,
            dev.opts(c),
            c.bias_grad,
        ),
    );
    compare(&format!("{tag} (own forward)"), c, &own, &want);
}

#[test]
fn bidirectional_and_cross_attention_match_cpu() {
    let m = metal();
    let mut n = 0;
    for &d in &[16usize, 32, 64] {
        for &(tq, tk) in &[
            (1usize, 1usize),
            (2, 2),
            (31, 31),
            (32, 32),
            (33, 33),
            (65, 65),
            (5, 40),
            (40, 5),
            (33, 70),
            (70, 33),
            (1, 97),
            (97, 1),
        ] {
            check(
                &m,
                &Case::new(2, 3, 3, tq, tk, d),
                (d * 1000 + tq * 7 + tk) as u64,
            );
            n += 1;
        }
    }
    // The widest compiled width, at the same boundaries.
    for &(tq, tk) in &[(1usize, 1usize), (33, 33), (5, 40), (40, 5), (65, 33)] {
        check(&m, &Case::new(1, 2, 2, tq, tk, 256), (tq * 13 + tk) as u64);
        n += 1;
    }
    assert_eq!(n, 41);
}

#[test]
fn grouped_query_heads_sum_onto_their_kv_head() {
    let m = metal();
    for &(h, hkv) in &[(4usize, 2usize), (4, 1), (6, 2), (8, 1)] {
        for &(tq, tk, d) in &[(33usize, 33usize, 32usize), (7, 70, 64), (70, 7, 16)] {
            check(
                &m,
                &Case::new(2, h, hkv, tq, tk, d),
                (h * 100 + tq + d) as u64,
            );
        }
    }
}

#[test]
fn every_bias_layout_and_its_gradient_match_cpu() {
    let m = metal();
    let (b, h) = (2usize, 4usize);
    for &(tq, tk, d) in &[
        (33usize, 70usize, 32usize),
        (70, 33, 64),
        (32, 32, 16),
        (5, 5, 32),
        (1, 40, 32),
    ] {
        let layouts = [
            [1, 1, 1],
            [b, 1, 1],
            [1, h, 1],
            [1, 1, tq],
            [b, 1, tq],
            [1, h, tq],
            [b, h, tq],
            [b, h, 1],
        ];
        for (i, layout) in layouts.into_iter().enumerate() {
            // Two query heads share each KV head, so a gradient that
            // broadcasts over heads sums across a KV group as well.
            check(
                &m,
                &Case::new(b, h, h / 2, tq, tk, d).bias(layout),
                (tq * 31 + tk * 7 + i) as u64,
            );
        }
    }
    // A bias whose gradient is not asked for (a mask) costs no bias kernel.
    let mut c = Case::new(2, 4, 2, 33, 70, 32).bias([1, 1, 33]);
    c.bias_grad = false;
    check(&m, &c, 99);
}

#[test]
fn key_lengths_match_cpu() {
    let m = metal();
    // Partial, full, longer than Tk (clamped), and a batch row with none.
    for lens in [[5u32, 70], [70, 70], [200, 33], [0, 40], [1, 1]] {
        for &(tq, tk, d) in &[(33usize, 70usize, 32usize), (70, 70, 64), (7, 40, 16)] {
            check(
                &m,
                &Case::new(2, 4, 2, tq, tk, d).lens(&lens),
                (lens[0] as usize * 3 + tq) as u64,
            );
        }
    }
    // With a bias and its gradient.
    check(
        &m,
        &Case::new(2, 4, 2, 40, 70, 32)
            .lens(&[20, 70])
            .bias([1, 4, 40]),
        7,
    );
    check(
        &m,
        &Case::new(2, 4, 2, 40, 70, 64)
            .lens(&[0, 33])
            .bias([2, 1, 1]),
        8,
    );
}

#[test]
fn causal_and_window_masks_match_cpu() {
    let m = metal();
    for &(tq, tk, d) in &[
        // Self-attention, prefill (Tq < Tk) and Tq > Tk, where the first
        // Tq - Tk queries see nothing.
        (33usize, 33usize, 32usize),
        (65, 65, 64),
        (17, 70, 32),
        (70, 17, 16),
        (1, 40, 64),
        (40, 1, 32),
    ] {
        for window in [None, Some(1), Some(8), Some(33), Some(1000)] {
            check(
                &m,
                &Case::new(2, 4, 2, tq, tk, d).causal(window),
                (tq * 11 + tk + window.unwrap_or(0)) as u64,
            );
        }
    }
    // Causal with a bias, key lengths and the bias gradient.
    check(
        &m,
        &Case::new(2, 4, 2, 40, 40, 32)
            .causal(Some(9))
            .lens(&[25, 40])
            .bias([2, 1, 40]),
        5,
    );
    check(
        &m,
        &Case::new(1, 2, 2, 33, 70, 64).causal(None).bias([1, 2, 1]),
        6,
    );
}

/// The shapes of `ojas-cpu/tests/attention_general.rs`, which hold the CPU
/// forward and backward to an f64 oracle: the same cases, run on Metal.
#[test]
fn the_cpu_general_attention_cases_match_on_metal() {
    let m = metal();
    let cases = [
        Case::new(2, 3, 3, 9, 9, 8),
        Case::new(2, 2, 2, 5, 11, 8),
        Case::new(1, 2, 2, 12, 4, 16),
        Case::new(2, 4, 2, 7, 10, 8),
        Case::new(1, 2, 1, 6, 17, 5),
        Case::new(1, 2, 2, 10, 10, 8).causal(None),
        Case::new(1, 2, 2, 10, 10, 8).causal(Some(3)),
        Case::new(1, 2, 2, 4, 9, 8).causal(None),
        Case::new(1, 2, 2, 4, 9, 8).causal(Some(2)),
        Case::new(1, 2, 2, 8, 5, 8).causal(None),
        Case::new(3, 2, 2, 6, 12, 8).lens(&[12, 5, 1]),
        Case::new(3, 2, 2, 6, 12, 8).lens(&[0, 7, 99]),
        Case::new(2, 2, 1, 5, 9, 8).causal(None).lens(&[4, 9]),
        Case::new(2, 3, 3, 5, 7, 8).bias([2, 3, 5]),
        Case::new(2, 3, 3, 5, 7, 8).bias([1, 3, 5]),
        Case::new(2, 3, 3, 5, 7, 8).bias([2, 1, 5]),
        Case::new(2, 3, 3, 5, 7, 8).bias([2, 3, 1]),
        Case::new(2, 3, 3, 5, 7, 8).bias([1, 1, 1]),
        Case::new(2, 4, 2, 6, 7, 8).bias([1, 4, 6]),
        Case::new(2, 2, 2, 6, 10, 8)
            .causal(Some(4))
            .bias([2, 2, 6])
            .lens(&[10, 6]),
        Case::new(3, 2, 2, 4, 9, 8).bias([1, 1, 4]).lens(&[2, 0, 9]),
        Case::new(2, 4, 2, 140, 180, 32)
            .bias([1, 4, 140])
            .lens(&[180, 41]),
        Case::new(1, 3, 3, 128, 128, 64)
            .causal(Some(20))
            .bias([1, 1, 128]),
        Case::new(2, 2, 2, 66, 300, 16).bias([2, 2, 66]),
        Case::new(1, 2, 2, 300, 300, 32)
            .causal(None)
            .bias([1, 2, 300]),
        Case::new(1, 2, 2, 260, 340, 32).lens(&[200]),
    ];
    for (n, c) in cases.iter().enumerate() {
        check(&m, c, 4000 + n as u64);
    }
}

/// Head dims that are not a compiled width run zero-padded to the next one,
/// through the limit.
#[test]
fn padded_head_dims_match_cpu() {
    let m = metal();
    for &d in &[1usize, 5, 20, 33, 63, 80, 100, 128, 129, 200, 256] {
        check(
            &m,
            &Case::new(1, 2, 1, 37, 41, d).bias([1, 2, 37]),
            d as u64,
        );
        check(
            &m,
            &Case::new(1, 2, 2, 40, 40, d).causal(Some(7)),
            d as u64 + 1,
        );
    }
}

/// One window of the SAM windowed ViT layers: 9 windows of 576 tokens, 16
/// heads of 64, no mask.
#[test]
fn sam_windowed_attention_matches_cpu() {
    let m = metal();
    check(&m, &Case::new(9, 16, 16, 576, 576, 64), 576);
}

/// The SAM detector's cross-attention: 201 queries over 5184 keys at head
/// dim 32, 8 heads, with a full-extent bias and its gradient.
#[test]
fn sam_detector_cross_attention_with_bias_matches_cpu() {
    let m = metal();
    check(
        &m,
        &Case::new(1, 8, 8, 201, 5184, 32).bias([1, 8, 201]),
        5184,
    );
}

/// No atomics, a fixed order: the same call repeats bit for bit, with grouped
/// heads and a bias that broadcasts over the batch and the heads.
#[test]
fn the_backward_is_deterministic() {
    let m = metal();
    for c in [
        Case::new(2, 4, 2, 70, 97, 32).bias([1, 1, 1]),
        Case::new(2, 4, 1, 97, 70, 64)
            .bias([1, 1, 97])
            .lens(&[50, 70]),
        Case::new(3, 4, 2, 65, 65, 16)
            .causal(Some(20))
            .bias([3, 4, 65]),
    ] {
        let cpu = cpu();
        let host = Operands::host(&c, 31);
        let (y, lse) = ok(
            "forward",
            cpu.attention_forward_saving(&host.q, &host.k, &host.v, host.opts(&c)),
        );
        let dev = host.up(&m);
        let (dy, dlse) = (up(&m, &y), up(&m, &lse));
        let run = || {
            let g = ok(
                "backward",
                m.attention_backward(
                    &dev.q,
                    &dev.k,
                    &dev.v,
                    &dy,
                    &dlse,
                    &dev.gy,
                    dev.opts(&c),
                    true,
                ),
            );
            let mut all = vec![down(&g.q), down(&g.k), down(&g.v)];
            all.extend(g.bias.as_ref().map(down));
            all.iter().map(|v| bits(v)).collect::<Vec<_>>()
        };
        let first = run();
        for rep in 0..3 {
            assert_eq!(run(), first, "{c:?}: run {rep} differs");
        }
    }
}

/// A causal self-attention call runs the same bodies as
/// `causal_sdpa_backward`, so it gives the same bits.
#[test]
fn causal_attention_backward_equals_causal_sdpa_backward() {
    let m = metal();
    for &(t, d, window) in &[
        (70usize, 32usize, None),
        (33, 64, Some(9usize)),
        (65, 16, None),
    ] {
        let shape = [2, 4, t, d];
        let kv_shape = [2, 2, t, d];
        let (q, k, v, gy) = (
            up(&m, &rand(&shape, 1, 1.0)),
            up(&m, &rand(&kv_shape, 2, 1.0)),
            up(&m, &rand(&kv_shape, 3, 1.0)),
            up(&m, &rand(&shape, 4, 1.0)),
        );
        let (y, lse) = ok("fwd", m.causal_sdpa_forward(&q, &k, &v, window));
        let (wq, wk, wv) = ok(
            "causal",
            m.causal_sdpa_backward(&q, &k, &v, &y, &lse, &gy, window),
        );
        let opts = AttentionOpts {
            causal: true,
            window,
            ..Default::default()
        };
        let got = ok(
            "general",
            m.attention_backward(&q, &k, &v, &y, &lse, &gy, opts, false),
        );
        let tag = format!("t{t} d{d} {window:?}");
        let down_host = |t: &Tensor| host(&down(t), t.shape());
        same_bits(&format!("{tag} dq"), &got.q, &down_host(&wq));
        same_bits(&format!("{tag} dk"), &got.k, &down_host(&wk));
        same_bits(&format!("{tag} dv"), &got.v, &down_host(&wv));
    }
}

#[test]
fn refusals_match_the_validator_and_the_device_limits() {
    let m = metal();
    let c = Case::new(1, 2, 2, 5, 6, 32);
    let host = Operands::host(&c, 3);
    let (y, lse) = ok(
        "forward",
        cpu().attention_forward_saving(&host.q, &host.k, &host.v, host.opts(&c)),
    );
    let dev = host.up(&m);
    let (dy, dlse) = (up(&m, &y), up(&m, &lse));

    // A bias gradient without a bias is the CPU's refusal, word for word.
    let cpu_err = cpu()
        .attention_backward(
            &host.q,
            &host.k,
            &host.v,
            &y,
            &lse,
            &host.gy,
            host.opts(&c),
            true,
        )
        .expect_err("cpu refuses");
    let err = m
        .attention_backward(
            &dev.q,
            &dev.k,
            &dev.v,
            &dy,
            &dlse,
            &dev.gy,
            dev.opts(&c),
            true,
        )
        .expect_err("metal refuses");
    assert_eq!(format!("{err:?}"), format!("{cpu_err:?}"));

    // A head dim past the limit names it.
    let wide = Case::new(1, 1, 1, 3, 3, 257);
    let w = Operands::host(&wide, 4);
    let dw = w.up(&m);
    let wy = up(&m, &rand(&[1, 1, 3, 257], 5, 1.0));
    let wl = up(&m, &rand(&[1, 1, 3], 6, 1.0));
    let err = m
        .attention_backward(
            &dw.q,
            &dw.k,
            &dw.v,
            &wy,
            &wl,
            &dw.gy,
            AttentionOpts::default(),
            false,
        )
        .expect_err("past the limit");
    assert!(
        matches!(
            err,
            OjasError::UnsupportedHeadDim {
                head_dim: 257,
                limit: 256
            }
        ),
        "{err:?}"
    );

    // A host tensor is a placement error, not a silent upload.
    let err = m
        .attention_backward(
            &host.q,
            &dev.k,
            &dev.v,
            &dy,
            &dlse,
            &dev.gy,
            AttentionOpts::default(),
            false,
        )
        .expect_err("host operand");
    assert!(
        matches!(
            err,
            OjasError::Placement {
                expected: Some(BackendId::Metal),
                ..
            }
        ),
        "{err:?}"
    );
}

/// Every operand is scanned, not only the entries a mask lets a query read,
/// and the fault surfaces at the next sync under the op's name.
#[test]
fn a_non_finite_operand_is_a_deferred_fault() {
    let m = metal();
    let c = Case::new(1, 2, 2, 4, 6, 32);
    let mut host = Operands::host(&c, 9);
    let (y, lse) = ok(
        "forward",
        cpu().attention_forward_saving(&host.q, &host.k, &host.v, host.opts(&c)),
    );
    let mut k = ok("keys", host.k.to_f32_vec());
    k[0] = f32::NAN;
    host.k = self::host(&k, &[1, 2, 6, 32]);
    let dev = host.up(&m);
    let (dy, dlse) = (up(&m, &y), up(&m, &lse));
    let r = m.attention_backward(
        &dev.q,
        &dev.k,
        &dev.v,
        &dy,
        &dlse,
        &dev.gy,
        AttentionOpts::default(),
        false,
    );
    if m.has_tensorops() {
        deferred(&m, "nan key", r, "attention_backward");
    }
}

#[test]
fn the_backward_charges_its_outputs_and_scratch_to_the_budget() {
    // Outputs: dQ, dK, dV; scratch: Dr (B * H * Tq). A budget one float short
    // of that refuses, and exactly that fits.
    let m = metal();
    let c = Case::new(1, 2, 2, 8, 8, 16);
    let host = Operands::host(&c, 12);
    let (y, lse) = ok(
        "forward",
        cpu().attention_forward_saving(&host.q, &host.k, &host.v, host.opts(&c)),
    );
    let dev = host.up(&m);
    let (dy, dlse) = (up(&m, &y), up(&m, &lse));
    let need = (3 * 2 * 8 * 16 + 2 * 8) as u64 * 4;
    let tight = m.with_budget(Budget::new(need - 4));
    let err = tight
        .attention_backward(
            &dev.q,
            &dev.k,
            &dev.v,
            &dy,
            &dlse,
            &dev.gy,
            AttentionOpts::default(),
            false,
        )
        .expect_err("one float short");
    assert!(matches!(err, OjasError::CapacityExceeded { .. }), "{err:?}");
    // The refusal left nothing charged: the exact amount fits.
    let exact = m.with_budget(Budget::new(need));
    let got = ok(
        "exact budget",
        exact.attention_backward(
            &dev.q,
            &dev.k,
            &dev.v,
            &dy,
            &dlse,
            &dev.gy,
            AttentionOpts::default(),
            false,
        ),
    );
    assert_eq!(got.q.shape(), dev.q.shape());
}

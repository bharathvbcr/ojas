//! General attention (`attention_forward`, `attention_forward_saving`) on
//! Metal against the CPU reference: bidirectional, cross-attention with
//! `Tq != Tk`, grouped-query heads, a bias in every broadcast layout, key
//! lengths and the causal / window masks, at head dims the kernel pads and
//! at its limit; plus its error contract.
//!
//! The kernel is scalar f32 (`ojas_attn_gen_fwd`), so unlike the tiled causal
//! kernels it needs no matrix units. There is no Metal backward yet:
//! `attention_backward` is `Unsupported`.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use ojas_core::{AttentionOpts, Backend, OjasError, Tensor, METAL_MAX_HEAD_DIM};
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
        }
    }
    fn causal(mut self, window: Option<usize>) -> Self {
        self.causal = true;
        self.window = window;
        self
    }
    fn bias(mut self, shape: [usize; 3]) -> Self {
        self.bias = Some(shape);
        self
    }
    fn lens(mut self, lens: &[u32]) -> Self {
        self.lens = Some(lens.to_vec());
        self
    }
}

/// Host operands of a case, then the same on `m`.
struct Operands {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    bias: Option<Tensor>,
    lens: Option<Tensor>,
}

impl Operands {
    fn host(c: &Case, seed: u64) -> Self {
        Operands {
            q: rand(&[c.b, c.h, c.tq, c.d], seed, 1.0),
            k: rand(&[c.b, c.hkv, c.tk, c.d], seed + 1, 1.0),
            v: rand(&[c.b, c.hkv, c.tk, c.d], seed + 2, 1.0),
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
            bias: self.bias.as_ref().map(|t| up(m, t)),
            lens: self.lens.as_ref().map(|t| up(m, t)),
        }
    }
    fn opts<'a>(&'a self, c: &Case) -> AttentionOpts<'a> {
        AttentionOpts {
            causal: c.causal,
            window: c.window,
            bias: self.bias.as_ref(),
            key_lens: self.lens.as_ref(),
        }
    }
}

/// The forward tolerance of the causal tests, widened with sqrt(Tk).
fn tol(tk: usize) -> f32 {
    4e-5 * (tk as f32).sqrt().max(1.0)
}

fn check(m: &MetalBackend, c: &Case, seed: u64) {
    let cpu = cpu();
    let host = Operands::host(c, seed);
    let dev = host.up(m);
    let tag = format!("{c:?}");
    let (want, want_lse) = ok(
        &tag,
        cpu.attention_forward_saving(&host.q, &host.k, &host.v, host.opts(c)),
    );
    ok("sync", m.sync());
    let before = m.dispatches();
    let (got, got_lse) = ok(
        &tag,
        m.attention_forward_saving(&dev.q, &dev.k, &dev.v, dev.opts(c)),
    );
    ok("sync", m.sync());
    let saving_dispatches = m.dispatches() - before;
    same_tensor(&format!("{tag} output"), &got, &want, tol(c.tk), 1e-4);
    same_tensor(&format!("{tag} lse"), &got_lse, &want_lse, tol(c.tk), 1e-4);
    // Fresh uploads, so no scan of the first call's operands is skipped.
    let dev = host.up(m);
    let before = m.dispatches();
    let plain = ok(
        &tag,
        m.attention_forward(&dev.q, &dev.k, &dev.v, dev.opts(c)),
    );
    ok("sync", m.sync());
    let plain_dispatches = m.dispatches() - before;
    // The inference form is the saving form's output, bit for bit on the
    // general kernel. Unmasked self-attention at head dim 32 or 64 on a TensorOps
    // device is the exception: `attention_forward` runs tessl's tiled
    // `encoder_attn` (26x faster at [1, 16, 5184, 64]), whose sums are in
    // another order, so the two forms agree to the forward tolerance (they
    // measured 1.5e-7 apart) and no closer.
    if routes_to_encoder_attn(m, c) {
        assert!(
            plain_dispatches > saving_dispatches,
            "{tag}: routed {plain_dispatches} dispatches vs the general kernel's {saving_dispatches}"
        );
        same_tensor(&format!("{tag} routed"), &plain, &want, tol(c.tk), 1e-4);
        same_tensor(
            &format!("{tag} routed vs saving"),
            &plain,
            &down_host(&got),
            tol(c.tk),
            1e-4,
        );
    } else {
        assert_eq!(
            plain_dispatches, saving_dispatches,
            "{tag}: a form that does not route dispatches as the saving form does"
        );
        assert_eq!(down(&plain), down(&got), "{tag}: forward forms differ");
    }
}

/// A device tensor's values as a host tensor, for [`same_tensor`].
fn down_host(t: &Tensor) -> Tensor {
    ok("to host", t.to_host(&ojas_core::Budget::new(8 * GIB)))
}

/// Whether `attention_forward` takes tessl's `encoder_attn` for this case:
/// plain attention at head dim 32 or 64, `Tq <= Tk <= 8 Tq`, on a device with
/// TensorOps.
fn routes_to_encoder_attn(m: &MetalBackend, c: &Case) -> bool {
    m.has_tensorops()
        && !c.causal
        && c.window.is_none()
        && c.bias.is_none()
        && c.lens.is_none()
        && c.tq <= c.tk
        && c.tk <= 8 * c.tq
        && c.h == c.hkv
        && (c.d == 32 || c.d == 64)
}

#[test]
fn bidirectional_and_cross_attention_match_cpu() {
    let m = metal();
    for (n, c) in [
        Case::new(2, 3, 3, 9, 9, 64),
        Case::new(2, 2, 2, 5, 11, 64),
        Case::new(1, 2, 2, 12, 4, 32),
        Case::new(2, 4, 2, 7, 10, 64),
        // Head dims that are not a multiple of 32, and the limit.
        Case::new(1, 2, 2, 20, 33, 80),
        Case::new(1, 2, 1, 6, 17, 5),
        Case::new(1, 2, 2, 31, 31, 96),
        Case::new(1, 1, 1, 33, 65, 256),
        // Longer than a tile of anything.
        Case::new(1, 2, 2, 130, 700, 64),
    ]
    .iter()
    .enumerate()
    {
        check(&m, c, 100 + n as u64);
    }
}

#[test]
fn causal_window_and_prefill_alignment_match_cpu() {
    let m = metal();
    for (n, c) in [
        Case::new(1, 2, 2, 10, 10, 64).causal(None),
        Case::new(1, 2, 2, 10, 10, 64).causal(Some(3)),
        // The queries are the last positions of the keys.
        Case::new(1, 2, 2, 4, 9, 64).causal(None),
        Case::new(1, 2, 2, 4, 9, 64).causal(Some(2)),
        // More queries than keys: the first ones see nothing.
        Case::new(1, 2, 2, 8, 5, 64).causal(None),
        Case::new(2, 4, 2, 70, 70, 64).causal(Some(33)),
    ]
    .iter()
    .enumerate()
    {
        check(&m, c, 200 + n as u64);
    }
}

#[test]
fn key_lengths_match_cpu_including_empty_and_oversized() {
    let m = metal();
    for (n, c) in [
        Case::new(3, 2, 2, 6, 12, 64).lens(&[12, 5, 1]),
        Case::new(3, 2, 2, 6, 12, 64).lens(&[0, 7, 99]),
        Case::new(2, 2, 1, 5, 9, 64).causal(None).lens(&[4, 9]),
        // A padded text prompt: 32 keys, a few live.
        Case::new(2, 8, 8, 100, 32, 64).lens(&[7, 32]),
    ]
    .iter()
    .enumerate()
    {
        check(&m, c, 300 + n as u64);
    }
}

#[test]
fn every_bias_layout_matches_cpu() {
    let m = metal();
    for (n, c) in [
        Case::new(2, 3, 3, 5, 7, 64).bias([2, 3, 5]),
        Case::new(2, 3, 3, 5, 7, 64).bias([1, 3, 5]),
        Case::new(2, 3, 3, 5, 7, 64).bias([2, 1, 5]),
        Case::new(2, 3, 3, 5, 7, 64).bias([2, 3, 1]),
        Case::new(2, 3, 3, 5, 7, 64).bias([1, 1, 1]),
        Case::new(2, 4, 2, 6, 7, 64).bias([1, 4, 6]),
        Case::new(2, 2, 2, 6, 10, 64)
            .causal(Some(4))
            .bias([2, 2, 6])
            .lens(&[10, 6]),
        // The detector's shape: queries over image tokens, a bias per head.
        Case::new(1, 8, 8, 200, 144, 32).bias([1, 8, 200]),
    ]
    .iter()
    .enumerate()
    {
        check(&m, c, 400 + n as u64);
    }
}

#[test]
fn offset_views_are_read_in_place() {
    let m = metal();
    let c = Case::new(2, 2, 2, 5, 9, 64).bias([1, 2, 5]).lens(&[9, 4]);
    let host = Operands::host(&c, 500);
    let cpu = cpu();
    let want = ok(
        "cpu",
        cpu.attention_forward(&host.q, &host.k, &host.v, host.opts(&c)),
    );
    let q = offset_view(&m, &host.q, 3, 1);
    let k = offset_view(&m, &host.k, 5, 2);
    let v = offset_view(&m, &host.v, 7, 3);
    let bias = offset_view(&m, host.bias.as_ref().unwrap(), 1, 4);
    let lens = up(&m, host.lens.as_ref().unwrap());
    let opts = AttentionOpts {
        causal: false,
        window: None,
        bias: Some(&bias),
        key_lens: Some(&lens),
    };
    let got = ok("metal", m.attention_forward(&q, &k, &v, opts));
    same_tensor("offset views", &got, &want, tol(9), 1e-4);
}

#[test]
fn a_sharp_softmax_rescales_instead_of_overflowing() {
    let (m, c) = (metal(), cpu());
    let (q, k, v) = (
        rand(&[1, 2, 40, 32], 1, 8.0),
        rand(&[1, 2, 100, 32], 2, 8.0),
        rand(&[1, 2, 100, 32], 3, 1.0),
    );
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
    same_tensor("sharp", &got, &want, tol(100), 1e-4);
}

#[test]
fn head_dims_follow_the_core_limit() {
    let m = metal();
    let limit = METAL_MAX_HEAD_DIM as usize;
    for d in [limit, limit + 1] {
        let (q, k) = (rand(&[1, 1, 3, d], 1, 1.0), rand(&[1, 1, 4, d], 2, 1.0));
        let r = m.attention_forward(
            &up(&m, &q),
            &up(&m, &k),
            &up(&m, &k),
            AttentionOpts::default(),
        );
        if d > limit {
            assert!(
                matches!(r, Err(OjasError::UnsupportedHeadDim { .. })),
                "d{d}: {r:?}"
            );
        } else {
            ok(&format!("d{d}"), r);
        }
    }
}

#[test]
fn malformed_calls_are_refused_as_the_cpu_refuses_them() {
    let (m, c) = (metal(), cpu());
    let case = Case::new(2, 4, 2, 5, 7, 64);
    let host = Operands::host(&case, 9);
    let dev = host.up(&m);
    let none = AttentionOpts::default();
    let bad_bias = rand(&[1, 1, 4, 7], 1, 1.0);
    let bad_lens = host_u32(&[1], &[1]);
    let bad_window = AttentionOpts {
        window: Some(3),
        ..none
    };
    let calls: Vec<(&str, AttentionOpts<'_>, AttentionOpts<'_>)> = vec![
        ("window without causal", bad_window, bad_window),
        (
            "bias queries",
            AttentionOpts {
                bias: Some(&bad_bias),
                ..none
            },
            AttentionOpts {
                bias: Some(&bad_bias),
                ..none
            },
        ),
        (
            "key_lens batch",
            AttentionOpts {
                key_lens: Some(&bad_lens),
                ..none
            },
            AttentionOpts {
                key_lens: Some(&bad_lens),
                ..none
            },
        ),
    ];
    for (what, host_opts, dev_opts) in calls {
        let want = c.attention_forward(&host.q, &host.k, &host.v, host_opts);
        let got = m.attention_forward(&dev.q, &dev.k, &dev.v, dev_opts);
        match (want, got) {
            (Err(w), Err(g)) => assert_eq!(format!("{w:?}"), format!("{g:?}"), "{what}"),
            (w, g) => panic!("{what}: cpu {w:?} metal {g:?}"),
        }
    }
}

#[test]
fn a_non_finite_operand_is_reported_at_the_next_sync() {
    let m = metal();
    let c = Case::new(1, 2, 2, 3, 4, 64);
    let mut host = Operands::host(&c, 11);
    let mut k = ok("keys", host.k.to_f32_vec());
    // Every operand is scanned, not only the keys a mask lets a query read.
    k[0] = f32::NAN;
    host.k = self::host(&k, &[1, 2, 4, 64]);
    let dev = host.up(&m);
    let r = m.attention_forward_saving(&dev.q, &dev.k, &dev.v, AttentionOpts::default());
    deferred(&m, "nan key", r, "attention_forward_saving");
}

#[test]
fn there_is_no_metal_backward_yet() {
    let m = metal();
    let c = Case::new(1, 2, 2, 3, 4, 64);
    let host = Operands::host(&c, 13);
    let dev = host.up(&m);
    let (y, lse) = ok(
        "forward",
        m.attention_forward_saving(&dev.q, &dev.k, &dev.v, AttentionOpts::default()),
    );
    let r = m.attention_backward(
        &dev.q,
        &dev.k,
        &dev.v,
        &y,
        &lse,
        &dev.q,
        AttentionOpts::default(),
        false,
    );
    assert!(
        matches!(
            r,
            Err(OjasError::Unsupported {
                op: "attention_backward",
                ..
            })
        ),
        "{r:?}"
    );
}

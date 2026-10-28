//! The CPU softmax row kernel's bits, pinned across its callers.
//!
//! `softmax_forward` (encoder softmax, both `Numerics`), the general
//! attention rows behind `attention_forward_saving`, `causal_sdpa_forward`
//! and `cached_attention_forward` all normalize a score row with one kernel
//! (`ojas-cpu/src/attn.rs::softmax_row`). The attention softmax is the
//! oracle the Metal parity tests compare against, so each digest here is
//! FNV-1a over the output bits of one call on a fixed seeded fixture,
//! recorded from the two separate kernels (`encoder_ops::softmax_row` and
//! `attn::softmax_range`) before they were merged. A change to the shared
//! kernel's max, exponential, sum order or division moves them.
//!
//! The fixtures include the rows that could tell a different row maximum
//! apart: all-equal rows and rows whose maximum is `+0.0` or `-0.0`.

use ojas_core::{AttentionOpts, Backend, Budget, Numerics, Tensor};
use ojas_cpu::CpuBackend;

mod common;
use common::{f32t, u32t, SplitMix64};

fn backend(numerics: Numerics) -> CpuBackend {
    CpuBackend::with_threads(Budget::new(1 << 32), 2)
        .unwrap()
        .with_numerics(numerics)
}

fn fnv(tensors: &[&Tensor]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for t in tensors {
        for v in t.to_f32_vec().unwrap() {
            for byte in v.to_bits().to_le_bytes() {
                h ^= u64::from(byte);
                h = h.wrapping_mul(0x0100_0000_01b3);
            }
        }
    }
    h
}

/// Six rows of width `dim`: constant, alternating signed zeros, a signed
/// zero at the maximum with negatives after it, then three random rows of
/// spread `amp`.
fn softmax_rows(dim: usize, amp: f32, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix64(seed);
    let mut data = vec![0.7f32; dim];
    data.extend((0..dim).map(|i| if i % 2 == 0 { 0.0 } else { -0.0 }));
    data.extend((0..dim).map(|i| match i {
        0 => -0.0,
        1 => 0.0,
        _ => -amp.abs() * rng.unit().abs(),
    }));
    data.extend(rng.vec(3 * dim, amp));
    data
}

struct Attn {
    b: usize,
    h: usize,
    hkv: usize,
    tq: usize,
    tk: usize,
    d: usize,
    causal: bool,
    window: Option<usize>,
    bias: Option<[usize; 3]>,
    lens: Option<&'static [u32]>,
}

const ATTN: [Attn; 5] = [
    Attn {
        b: 2,
        h: 4,
        hkv: 2,
        tq: 5,
        tk: 7,
        d: 8,
        causal: false,
        window: None,
        bias: Some([1, 4, 5]),
        lens: Some(&[7, 3]),
    },
    Attn {
        b: 1,
        h: 2,
        hkv: 2,
        tq: 9,
        tk: 9,
        d: 16,
        causal: true,
        window: None,
        bias: None,
        lens: None,
    },
    Attn {
        b: 1,
        h: 2,
        hkv: 1,
        tq: 6,
        tk: 20,
        d: 8,
        causal: true,
        window: Some(4),
        bias: None,
        lens: None,
    },
    Attn {
        b: 1,
        h: 3,
        hkv: 3,
        tq: 1,
        tk: 33,
        d: 32,
        causal: false,
        window: None,
        bias: None,
        lens: None,
    },
    Attn {
        b: 1,
        h: 2,
        hkv: 2,
        tq: 40,
        tk: 40,
        d: 8,
        causal: false,
        window: None,
        bias: Some([1, 1, 1]),
        lens: None,
    },
];

fn digests(numerics: Numerics) -> Vec<(String, u64)> {
    let cpu = backend(numerics);
    let mut out = Vec::new();
    for (i, &dim) in [1usize, 2, 3, 7, 8, 9, 16, 17, 31, 64, 100, 1000, 4099]
        .iter()
        .enumerate()
    {
        for (j, amp) in [1.0f32, 40.0, 3.0e5].into_iter().enumerate() {
            let data = softmax_rows(dim, amp, 7000 + (i * 3 + j) as u64);
            let y = cpu.softmax_forward(&f32t(&cpu, &data, &[6, dim])).unwrap();
            out.push((format!("softmax dim={dim} amp={amp}"), fnv(&[&y])));
        }
    }
    for (i, c) in ATTN.iter().enumerate() {
        let mut rng = SplitMix64(900 + i as u64);
        let q = f32t(
            &cpu,
            &rng.vec(c.b * c.h * c.tq * c.d, 1.0),
            &[c.b, c.h, c.tq, c.d],
        );
        let k = f32t(
            &cpu,
            &rng.vec(c.b * c.hkv * c.tk * c.d, 1.0),
            &[c.b, c.hkv, c.tk, c.d],
        );
        let v = f32t(
            &cpu,
            &rng.vec(c.b * c.hkv * c.tk * c.d, 1.0),
            &[c.b, c.hkv, c.tk, c.d],
        );
        let bias = c.bias.map(|[bb, bh, bq]| {
            f32t(
                &cpu,
                &rng.vec(bb * bh * bq * c.tk, 2.0),
                &[bb, bh, bq, c.tk],
            )
        });
        let lens = c.lens.map(|l| u32t(&cpu, l, &[l.len()]));
        let opts = AttentionOpts {
            causal: c.causal,
            window: c.window,
            bias: bias.as_ref(),
            key_lens: lens.as_ref(),
        };
        let (y, lse) = cpu.attention_forward_saving(&q, &k, &v, opts).unwrap();
        out.push((format!("attention case {i}"), fnv(&[&y, &lse])));
        if c.tq == c.tk && c.bias.is_none() && c.lens.is_none() && c.causal {
            let (y, lse) = cpu.causal_sdpa_forward(&q, &k, &v, c.window).unwrap();
            out.push((format!("causal_sdpa case {i}"), fnv(&[&y, &lse])));
        }
    }
    for (i, &(b, tq, h, hkv, d, tcap, kv_len, window)) in [
        (
            1usize, 3usize, 4usize, 2usize, 16usize, 12usize, 11usize, None,
        ),
        (2, 1, 4, 4, 8, 6, 6, None),
        (1, 4, 2, 1, 8, 16, 14, Some(5usize)),
    ]
    .iter()
    .enumerate()
    {
        let mut rng = SplitMix64(1700 + i as u64);
        let q = f32t(&cpu, &rng.vec(b * tq * h * d, 1.0), &[b, tq, h, d]);
        let kc = f32t(&cpu, &rng.vec(b * tcap * hkv * d, 1.0), &[b, tcap, hkv, d]);
        let vc = f32t(&cpu, &rng.vec(b * tcap * hkv * d, 1.0), &[b, tcap, hkv, d]);
        let y = cpu
            .cached_attention_forward(&q, &kc, &vc, kv_len, window)
            .unwrap();
        out.push((format!("cached_attention case {i}"), fnv(&[&y])));
    }
    out
}

/// Digests recorded on the unmodified two-kernel code (2026-10-27), one
/// table per `Numerics`.
const EXACT: &[(&str, u64)] = &[
    ("softmax dim=1 amp=1", 0x616b7bfc18e4b415),
    ("softmax dim=1 amp=40", 0x616b7bfc18e4b415),
    ("softmax dim=1 amp=300000", 0x616b7bfc18e4b415),
    ("softmax dim=2 amp=1", 0x9ddb68be768f8242),
    ("softmax dim=2 amp=40", 0x5d225b77a01c3be7),
    ("softmax dim=2 amp=300000", 0xd267a95254231308),
    ("softmax dim=3 amp=1", 0xb6548e4b064e6d56),
    ("softmax dim=3 amp=40", 0x5d0580b25afb1cac),
    ("softmax dim=3 amp=300000", 0x3a5a4b0c777c8110),
    ("softmax dim=7 amp=1", 0xf9b271cc125df745),
    ("softmax dim=7 amp=40", 0x378c0501bae9ba91),
    ("softmax dim=7 amp=300000", 0x3cfde49817347b94),
    ("softmax dim=8 amp=1", 0x91d7d41a4fb73a53),
    ("softmax dim=8 amp=40", 0x0f560a2c16546322),
    ("softmax dim=8 amp=300000", 0x84f7b91c0a04dcc8),
    ("softmax dim=9 amp=1", 0x71c8c1b3c16df2d6),
    ("softmax dim=9 amp=40", 0x454927c53b772e08),
    ("softmax dim=9 amp=300000", 0x90517d80e366b90c),
    ("softmax dim=16 amp=1", 0x44b6e67343db0837),
    ("softmax dim=16 amp=40", 0xc0bbe71bec157dd0),
    ("softmax dim=16 amp=300000", 0x9aabb257fb0f9788),
    ("softmax dim=17 amp=1", 0xec5b548f1563605f),
    ("softmax dim=17 amp=40", 0x13a2494fd6d6dc63),
    ("softmax dim=17 amp=300000", 0xf7ba79c804b9421c),
    ("softmax dim=31 amp=1", 0x993bc99ed9e72d28),
    ("softmax dim=31 amp=40", 0x7b62431b399f3565),
    ("softmax dim=31 amp=300000", 0x0db914b5740bf308),
    ("softmax dim=64 amp=1", 0x0c00dcc6aa4e1252),
    ("softmax dim=64 amp=40", 0xae0a44713c2e9ba3),
    ("softmax dim=64 amp=300000", 0x26073b11418f1de8),
    ("softmax dim=100 amp=1", 0x274a29c6a3347d19),
    ("softmax dim=100 amp=40", 0xb9d6d0a9fbd20d5a),
    ("softmax dim=100 amp=300000", 0xdb43f11209ce49e8),
    ("softmax dim=1000 amp=1", 0xab0a39372efce27d),
    ("softmax dim=1000 amp=40", 0xde593d1215eec01b),
    ("softmax dim=1000 amp=300000", 0x04952866584f0a48),
    ("softmax dim=4099 amp=1", 0xe203fe325751fa3a),
    ("softmax dim=4099 amp=40", 0xfa6971386011ef1c),
    ("softmax dim=4099 amp=300000", 0xaf65a832ebb998e6),
    ("attention case 0", 0x1d372bc64a5f026c),
    ("attention case 1", 0x5f213cdf1c9e935d),
    ("causal_sdpa case 1", 0x5f213cdf1c9e935d),
    ("attention case 2", 0x5f60921deac29a46),
    ("attention case 3", 0x2c1bbffb774678a6),
    ("attention case 4", 0x20a22c0c765a836a),
    ("cached_attention case 0", 0x99115a722d05f72d),
    ("cached_attention case 1", 0x95f322394cd2abbb),
    ("cached_attention case 2", 0xf1032bc2b967d861),
];
const FAST: &[(&str, u64)] = &[
    ("softmax dim=1 amp=1", 0x616b7bfc18e4b415),
    ("softmax dim=1 amp=40", 0x616b7bfc18e4b415),
    ("softmax dim=1 amp=300000", 0x616b7bfc18e4b415),
    ("softmax dim=2 amp=1", 0x9ddb68be768f8242),
    ("softmax dim=2 amp=40", 0x5d225b77a01c3be7),
    ("softmax dim=2 amp=300000", 0xd267a95254231308),
    ("softmax dim=3 amp=1", 0xb6548e4b064e6d56),
    ("softmax dim=3 amp=40", 0x5d0580b25afb1cac),
    ("softmax dim=3 amp=300000", 0x3a5a4b0c777c8110),
    ("softmax dim=7 amp=1", 0xf9b271cc125df745),
    ("softmax dim=7 amp=40", 0x378c0501bae9ba91),
    ("softmax dim=7 amp=300000", 0x3cfde49817347b94),
    ("softmax dim=8 amp=1", 0x5616ef6c5a159406),
    ("softmax dim=8 amp=40", 0x7d532e089bb3d9fe),
    ("softmax dim=8 amp=300000", 0x84f7b91c0a04dcc8),
    ("softmax dim=9 amp=1", 0xf9d27749305155b2),
    ("softmax dim=9 amp=40", 0x454927c53b772e08),
    ("softmax dim=9 amp=300000", 0x90517d80e366b90c),
    ("softmax dim=16 amp=1", 0x23a97555c8f798e0),
    ("softmax dim=16 amp=40", 0x0a34438b40febf0d),
    ("softmax dim=16 amp=300000", 0x9aabb257fb0f9788),
    ("softmax dim=17 amp=1", 0xd15b43658c72117b),
    ("softmax dim=17 amp=40", 0x785a506b6b473004),
    ("softmax dim=17 amp=300000", 0xf7ba79c804b9421c),
    ("softmax dim=31 amp=1", 0x29e03fcb8e9b2109),
    ("softmax dim=31 amp=40", 0x672304b61f2730dd),
    ("softmax dim=31 amp=300000", 0x0db914b5740bf308),
    ("softmax dim=64 amp=1", 0xeafede7192252595),
    ("softmax dim=64 amp=40", 0xbea66f0d3b28ee1e),
    ("softmax dim=64 amp=300000", 0x26073b11418f1de8),
    ("softmax dim=100 amp=1", 0x231480482009c4f5),
    ("softmax dim=100 amp=40", 0x300d736e120fce82),
    ("softmax dim=100 amp=300000", 0xdb43f11209ce49e8),
    ("softmax dim=1000 amp=1", 0x433b4cd8928e7f6b),
    ("softmax dim=1000 amp=40", 0xe84d5a5bec19a28a),
    ("softmax dim=1000 amp=300000", 0x04952866584f0a48),
    ("softmax dim=4099 amp=1", 0x0f3da79cae004a3d),
    ("softmax dim=4099 amp=40", 0xa5d92078494c16d4),
    ("softmax dim=4099 amp=300000", 0xaf65a832ebb998e6),
    ("attention case 0", 0x1d372bc64a5f026c),
    ("attention case 1", 0x5f213cdf1c9e935d),
    ("causal_sdpa case 1", 0x5f213cdf1c9e935d),
    ("attention case 2", 0x5f60921deac29a46),
    ("attention case 3", 0x2c1bbffb774678a6),
    ("attention case 4", 0x20a22c0c765a836a),
    ("cached_attention case 0", 0x99115a722d05f72d),
    ("cached_attention case 1", 0x95f322394cd2abbb),
    ("cached_attention case 2", 0xf1032bc2b967d861),
];

fn check(numerics: Numerics, want: &[(&str, u64)]) {
    let got = digests(numerics);
    let table: String = got
        .iter()
        .map(|(name, h)| format!("    (\"{name}\", {h:#018x}),\n"))
        .collect();
    assert_eq!(got.len(), want.len(), "{numerics:?} digests:\n{table}");
    for ((name, h), (want_name, want_h)) in got.iter().zip(want) {
        assert_eq!(name, want_name, "{numerics:?} digests:\n{table}");
        assert_eq!(
            h, want_h,
            "{numerics:?} {name} moved; current digests:\n{table}"
        );
    }
}

#[test]
fn exact_softmax_and_attention_bits_are_pinned() {
    check(Numerics::Exact, EXACT);
}

#[test]
fn fast_softmax_and_attention_bits_are_pinned() {
    check(Numerics::Fast, FAST);
}

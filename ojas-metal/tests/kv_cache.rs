//! KV-cache ops on Metal: `kv_cache_write` (T5) and `cached_attention_forward`
//! (T4), gate G4. The cache is time-major `[B, Tcap, Hkv, D]`.
#![cfg(all(target_os = "macos", feature = "metal"))]

mod common;

use std::sync::Arc;

use common::*;
use ojas_core::{sdpa_scale, Backend, OjasError, Tensor};

fn bits(t: &Tensor) -> Vec<u32> {
    down(t).iter().map(|x| x.to_bits()).collect()
}

fn buffer_ptr(t: &Tensor) -> *const () {
    match t.device_buffer() {
        Some(b) => Arc::as_ptr(b) as *const (),
        None => panic!("not a device tensor"),
    }
}

/// Dimensions of one cached-attention call.
#[derive(Clone, Copy, Debug)]
struct Dims {
    b: usize,
    tq: usize,
    h: usize,
    hkv: usize,
    d: usize,
    cap: usize,
    kv_len: usize,
}

/// Query `i` at position `kv_len - tq + i` attends to keys `0..=pos` of KV
/// head `h / (H / Hkv)`, in f64.
fn naive(q: &[f32], k: &[f32], v: &[f32], s: Dims) -> Vec<f32> {
    let group = s.h / s.hkv;
    let scale = f64::from(ok("scale", sdpa_scale(s.d as u32)));
    let mut out = vec![0.0f32; s.b * s.tq * s.h * s.d];
    for b in 0..s.b {
        for i in 0..s.tq {
            let pos = s.kv_len - s.tq + i;
            for hh in 0..s.h {
                let kh = hh / group;
                let qrow = &q[((b * s.tq + i) * s.h + hh) * s.d..][..s.d];
                let scores: Vec<f64> = (0..=pos)
                    .map(|j| {
                        let krow = &k[((b * s.cap + j) * s.hkv + kh) * s.d..][..s.d];
                        let dot: f64 = qrow
                            .iter()
                            .zip(krow)
                            .map(|(a, c)| f64::from(*a) * f64::from(*c))
                            .sum();
                        dot * scale
                    })
                    .collect();
                let mx = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let w: Vec<f64> = scores.iter().map(|x| (x - mx).exp()).collect();
                let total: f64 = w.iter().sum();
                for dd in 0..s.d {
                    let acc: f64 = (0..=pos)
                        .map(|j| w[j] * f64::from(v[((b * s.cap + j) * s.hkv + kh) * s.d + dd]))
                        .sum();
                    out[((b * s.tq + i) * s.h + hh) * s.d + dd] = (acc / total) as f32;
                }
            }
        }
    }
    out
}

fn attend(s: Dims, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let m = metal();
    let q = rand(&[s.b, s.tq, s.h, s.d], seed, 1.0);
    let k = rand(&[s.b, s.cap, s.hkv, s.d], seed + 1, 1.0);
    let v = rand(&[s.b, s.cap, s.hkv, s.d], seed + 2, 1.0);
    let got = ok(
        &format!("{s:?}"),
        m.cached_attention_forward(&up(&m, &q), &up(&m, &k), &up(&m, &v), s.kv_len),
    );
    assert_eq!(got.shape(), &[s.b, s.tq, s.h, s.d]);
    let want = naive(
        &ok("q", q.to_f32_vec()),
        &ok("k", k.to_f32_vec()),
        &ok("v", v.to_f32_vec()),
        s,
    );
    (down(&got), want)
}

#[test]
fn full_cache_without_grouping_equals_causal_sdpa_after_the_permute() {
    let m = metal();
    for (i, &(b, t, h, d, extra)) in [
        (1usize, 1usize, 1usize, 16usize, 0usize),
        (2, 5, 3, 32, 4),
        (1, 33, 2, 64, 0),
        (2, 70, 2, 64, 9),
        (1, 40, 2, 128, 3),
        (1, 17, 3, 24, 0),
    ]
    .iter()
    .enumerate()
    {
        let seed = 100 * i as u64;
        let q = rand(&[b, t, h, d], seed, 1.0);
        // Positions past kv_len hold unrelated values the op must not read.
        let k = rand(&[b, t + extra, h, d], seed + 1, 1.0);
        let v = rand(&[b, t + extra, h, d], seed + 2, 1.0);
        let (qd, kd, vd) = (up(&m, &q), up(&m, &k), up(&m, &v));
        let cached = ok("cached", m.cached_attention_forward(&qd, &kd, &vd, t));
        // SDPA wants [B, H, T, D] over exactly T positions.
        let head_major = |x: &Tensor| {
            let full = down(x);
            let rows = &full[..];
            let cap = x.shape()[1];
            let mut out = Vec::with_capacity(b * h * t * d);
            for bb in 0..b {
                for hh in 0..h {
                    for tt in 0..t {
                        let base = ((bb * cap + tt) * h + hh) * d;
                        out.extend_from_slice(&rows[base..base + d]);
                    }
                }
            }
            up(&m, &host(&out, &[b, h, t, d]))
        };
        let o = ok(
            "sdpa",
            m.causal_sdpa_forward(&head_major(&qd), &head_major(&kd), &head_major(&vd)),
        );
        let want = ok("permute", m.permute(&o, &[0, 2, 1, 3]));
        close(
            &format!("b{b} t{t} h{h} d{d}"),
            &down(&cached),
            &down(&want),
            2e-6,
            1e-5,
        );
    }
}

#[test]
fn grouped_query_attention_matches_a_naive_f64_reference() {
    let mut cases = Vec::new();
    for (hkv, h) in [(1usize, 6usize), (2, 6), (3, 6), (6, 6), (1, 1)] {
        for (tq, kv_len, cap) in [
            (1usize, 1usize, 9usize),
            (1, 9, 9),
            (3, 7, 12),
            (5, 5, 5),
            (4, 64, 80),
        ] {
            for d in [8usize, 64, 128] {
                cases.push(Dims {
                    b: 2,
                    tq,
                    h,
                    hkv,
                    d,
                    cap,
                    kv_len,
                });
            }
        }
    }
    for (i, s) in cases.into_iter().enumerate() {
        let (got, want) = attend(s, 7 + i as u64);
        close(&format!("{s:?}"), &got, &want, 1e-5, 1e-5);
    }
}

#[test]
fn decode_after_each_cache_write_matches_the_reference() {
    let m = metal();
    let (b, cap, h, hkv, d) = (2usize, 24usize, 4usize, 2usize, 32usize);
    let mut cache_k = up(&m, &host(&vec![0.0; b * cap * hkv * d], &[b, cap, hkv, d]));
    let mut cache_v = up(&m, &host(&vec![0.0; b * cap * hkv * d], &[b, cap, hkv, d]));
    let mut host_k = vec![0.0f32; b * cap * hkv * d];
    let mut host_v = vec![0.0f32; b * cap * hkv * d];
    // A 5-token prefill, then single-token decode steps.
    let mut at = 0usize;
    for (step, tn) in [5usize, 1, 1, 1, 3, 1].into_iter().enumerate() {
        let seed = 1000 + 10 * step as u64;
        let kn = values(b * tn * hkv * d, seed, 1.0);
        let vn = values(b * tn * hkv * d, seed + 1, 1.0);
        ok(
            "write k",
            m.kv_cache_write(&mut cache_k, &up(&m, &host(&kn, &[b, tn, hkv, d])), at),
        );
        ok(
            "write v",
            m.kv_cache_write(&mut cache_v, &up(&m, &host(&vn, &[b, tn, hkv, d])), at),
        );
        for bb in 0..b {
            for t in 0..tn {
                let src = (bb * tn + t) * hkv * d;
                let dst = (bb * cap + at + t) * hkv * d;
                host_k[dst..dst + hkv * d].copy_from_slice(&kn[src..src + hkv * d]);
                host_v[dst..dst + hkv * d].copy_from_slice(&vn[src..src + hkv * d]);
            }
        }
        at += tn;
        assert_eq!(down(&cache_k), host_k, "step {step}: k cache");
        assert_eq!(down(&cache_v), host_v, "step {step}: v cache");
        let q = values(b * tn * h * d, seed + 2, 1.0);
        let s = Dims {
            b,
            tq: tn,
            h,
            hkv,
            d,
            cap,
            kv_len: at,
        };
        let got = ok(
            "attend",
            m.cached_attention_forward(&up(&m, &host(&q, &[b, tn, h, d])), &cache_k, &cache_v, at),
        );
        close(
            &format!("step {step}"),
            &down(&got),
            &naive(&q, &host_k, &host_v, s),
            1e-5,
            1e-5,
        );
    }
}

#[test]
fn cached_attention_is_deterministic() {
    let s = Dims {
        b: 1,
        tq: 1,
        h: 12,
        hkv: 12,
        d: 64,
        cap: 1024,
        kv_len: 1024,
    };
    let m = metal();
    let q = up(&m, &rand(&[s.b, s.tq, s.h, s.d], 1, 1.0));
    let k = up(&m, &rand(&[s.b, s.cap, s.hkv, s.d], 2, 1.0));
    let v = up(&m, &rand(&[s.b, s.cap, s.hkv, s.d], 3, 1.0));
    let first = bits(&ok("a", m.cached_attention_forward(&q, &k, &v, s.kv_len)));
    for _ in 0..5 {
        assert_eq!(
            bits(&ok("b", m.cached_attention_forward(&q, &k, &v, s.kv_len))),
            first
        );
    }
}

#[test]
fn cached_attention_refusals() {
    let m = metal();
    let (b, tq, h, hkv, d, cap) = (1usize, 2usize, 4usize, 2usize, 16usize, 8usize);
    let q = up(&m, &rand(&[b, tq, h, d], 1, 1.0));
    let k = up(&m, &rand(&[b, cap, hkv, d], 2, 1.0));
    let v = up(&m, &rand(&[b, cap, hkv, d], 3, 1.0));
    for kv_len in [1usize, cap + 1] {
        let r = m.cached_attention_forward(&q, &k, &v, kv_len);
        assert!(
            matches!(r, Err(OjasError::OutOfRange { .. })),
            "kv_len {kv_len}: {r:?}"
        );
    }
    let q3 = up(&m, &rand(&[b, tq, 3, d], 4, 1.0));
    let r = m.cached_attention_forward(&q3, &k, &v, 4);
    assert!(matches!(r, Err(OjasError::Shape { .. })), "{r:?}");
    let wide = |s: &[usize], seed| up(&m, &rand(s, seed, 1.0));
    let r = m.cached_attention_forward(
        &wide(&[1, 1, 1, 192], 5),
        &wide(&[1, 4, 1, 192], 6),
        &wide(&[1, 4, 1, 192], 7),
        2,
    );
    assert!(
        matches!(r, Err(OjasError::UnsupportedHeadDim { head_dim: 192, .. })),
        "{r:?}"
    );
    let r = m.cached_attention_forward(&rand(&[b, tq, h, d], 1, 1.0), &k, &v, 4);
    assert!(matches!(r, Err(OjasError::Placement { .. })), "{r:?}");

    // Non-finite values in q or in the read window of either cache.
    let kv_len = 5usize;
    let poison_at = |shape: &[usize], idx: usize, seed: u64, val: f32| {
        let mut x = values(shape.iter().product(), seed, 1.0);
        x[idx] = val;
        up(&m, &host(&x, shape))
    };
    let qs = [b, tq, h, d];
    let cs = [b, cap, hkv, d];
    // The last element of the window: position kv_len - 1, last head, last dim.
    let last_in_window = ((kv_len - 1) * hkv + hkv - 1) * d + d - 1;
    for val in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let cases = [
            (
                "q",
                poison_at(&qs, b * tq * h * d - 1, 8, val),
                k.clone(),
                v.clone(),
            ),
            (
                "k",
                q.clone(),
                poison_at(&cs, last_in_window, 9, val),
                v.clone(),
            ),
            (
                "v",
                q.clone(),
                k.clone(),
                poison_at(&cs, last_in_window, 10, val),
            ),
            ("k first", q.clone(), poison_at(&cs, 0, 11, val), v.clone()),
        ];
        for (what, qq, kk, vv) in cases {
            let r = m.cached_attention_forward(&qq, &kk, &vv, kv_len);
            deferred(&m, &format!("{what} {val}"), r, "cached_attention_forward");
        }
    }
    // Positions at or past kv_len are not read, so they are not checked.
    let tail = poison_at(&cs, kv_len * hkv * d, 12, f32::NAN);
    ok("tail", m.cached_attention_forward(&q, &tail, &tail, kv_len));
    ok("tail leaves nothing pending", m.sync());
    // Scores that overflow f32 are a non-finite intermediate.
    let huge = |shape: &[usize]| up(&m, &host(&vec![1e30; shape.iter().product()], shape));
    let r = m.cached_attention_forward(&huge(&qs), &huge(&cs), &v, kv_len);
    deferred(&m, "overflowing scores", r, "cached_attention_forward");
}

/// Gate G4's device-vs-CPU row, against `CpuBackend::cached_attention_forward`.
#[test]
fn cached_attention_matches_the_cpu_reference() {
    let m = metal();
    let c = cpu();
    for (i, (tq, kv_len, h, hkv, d)) in [
        (1usize, 1024usize, 12usize, 12usize, 64usize),
        (4, 37, 6, 2, 128),
    ]
    .into_iter()
    .enumerate()
    {
        let cap = kv_len + 3;
        let q = rand(&[1, tq, h, d], 50 + i as u64, 1.0);
        let k = rand(&[1, cap, hkv, d], 51 + i as u64, 1.0);
        let v = rand(&[1, cap, hkv, d], 52 + i as u64, 1.0);
        let want = ok("cpu", c.cached_attention_forward(&q, &k, &v, kv_len));
        let got = ok(
            "metal",
            m.cached_attention_forward(&up(&m, &q), &up(&m, &k), &up(&m, &v), kv_len),
        );
        same_tensor("cached attention", &got, &want, 1e-5, 1e-5);
    }
}

#[test]
fn kv_cache_write_places_src_and_nothing_else() {
    let m = metal();
    let (b, cap, hkv, d) = (2usize, 7usize, 3usize, 5usize);
    for (tn, at) in [(3usize, 2usize), (1, 0), (1, 6), (7, 0), (2, 5)] {
        let base = values(b * cap * hkv * d, 20 + at as u64, 1.0);
        let src = values(b * tn * hkv * d, 30 + tn as u64, 1.0);
        let mut cache = up(&m, &host(&base, &[b, cap, hkv, d]));
        let ptr = buffer_ptr(&cache);
        ok(
            "write",
            m.kv_cache_write(&mut cache, &up(&m, &host(&src, &[b, tn, hkv, d])), at),
        );
        assert_eq!(buffer_ptr(&cache), ptr, "not written in place");
        let mut want = base.clone();
        let row = hkv * d;
        for bb in 0..b {
            for t in 0..tn {
                let s = (bb * tn + t) * row;
                let dst = (bb * cap + at + t) * row;
                want[dst..dst + row].copy_from_slice(&src[s..s + row]);
            }
        }
        let got: Vec<u32> = bits(&cache);
        let want: Vec<u32> = want.iter().map(|x| x.to_bits()).collect();
        assert_eq!(got, want, "tn {tn} at {at}");
    }
}

#[test]
fn kv_cache_write_refusals_leave_the_cache_unchanged() {
    let m = metal();
    let (b, cap, hkv, d) = (2usize, 6usize, 2usize, 4usize);
    let mut cache = up(&m, &rand(&[b, cap, hkv, d], 1, 1.0));
    let before = bits(&cache);
    let src = |tn: usize, seed| up(&m, &rand(&[b, tn, hkv, d], seed, 1.0));
    for (tn, at) in [(2usize, 5usize), (7, 0), (1, 6), (1, usize::MAX)] {
        let r = m.kv_cache_write(&mut cache, &src(tn, 2), at);
        assert!(
            matches!(r, Err(OjasError::OutOfRange { .. })),
            "tn {tn} at {at}: {r:?}"
        );
    }
    let r = m.kv_cache_write(&mut cache, &up(&m, &rand(&[b, 1, hkv, d + 1], 3, 1.0)), 0);
    assert!(matches!(r, Err(OjasError::Shape { .. })), "{r:?}");
    let r = m.kv_cache_write(&mut cache, &rand(&[b, 1, hkv, d], 4, 1.0), 0);
    assert!(matches!(r, Err(OjasError::Placement { .. })), "{r:?}");
    for val in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let n = b * 3 * hkv * d;
        for idx in [0, n - 1] {
            let mut x = values(n, 5, 1.0);
            x[idx] = val;
            let r = m.kv_cache_write(&mut cache, &up(&m, &host(&x, &[b, 3, hkv, d])), 1);
            deferred(&m, &format!("{val} at {idx}"), r, "kv_cache_write");
            assert_eq!(
                bits(&cache),
                before,
                "{val} at {idx}: a faulted write changed the cache"
            );
        }
    }
    // A cache another handle shares is not written through.
    let other = cache.clone();
    let r = m.kv_cache_write(&mut cache, &src(1, 6), 0);
    assert!(matches!(r, Err(OjasError::Shape { .. })), "{r:?}");
    drop(other);
    assert_eq!(bits(&cache), before, "a refused write changed the cache");
    ok("clean", m.kv_cache_write(&mut cache, &src(2, 7), 4));
}

//! KV-cache ops on wgpu (T4 `cached_attention_forward`, T5 `kv_cache_write`),
//! gate G4. The cache is time-major `[B, Tcap, Hkv, D]`.
//!
//! `kv_cache_write` refuses shape, range, placement and sharing problems
//! before anything is recorded, so the cache is unchanged. A non-finite
//! source is checked on the device first and the copy runs only if that
//! check was clean, so the cache is unchanged then too; the fault surfaces
//! at the next `sync` naming `kv_cache_write`.

mod common;

use std::sync::Arc;

use common::*;
use ojas_core::{Backend, OjasError, Tensor};
use ojas_wgpu::WgpuBackend;

fn bits(g: &WgpuBackend, t: &Tensor) -> Vec<u32> {
    g.download(t)
        .unwrap()
        .to_f32_vec()
        .unwrap()
        .iter()
        .map(|x| x.to_bits())
        .collect()
}

fn buffer_ptr(t: &Tensor) -> *const () {
    match t.device_buffer() {
        Some(b) => Arc::as_ptr(b) as *const (),
        None => panic!("not a device tensor"),
    }
}

fn tensor(v: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_f32(v, shape, host_budget()).unwrap()
}

#[test]
fn kv_cache_write_places_src_and_nothing_else() {
    let g = fresh();
    let (b, cap, hkv, d) = (2usize, 7usize, 3usize, 5usize);
    for (tn, at) in [(3usize, 2usize), (1, 0), (1, 6), (7, 0), (2, 5)] {
        let base = data(20 + at as u64, b * cap * hkv * d);
        let src = data(30 + tn as u64, b * tn * hkv * d);
        let mut cache = g.upload(&tensor(&base, &[b, cap, hkv, d])).unwrap();
        let ptr = buffer_ptr(&cache);
        g.kv_cache_write(
            &mut cache,
            &g.upload(&tensor(&src, &[b, tn, hkv, d])).unwrap(),
            at,
        )
        .unwrap();
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
        g.sync().unwrap();
        let want: Vec<u32> = want.iter().map(|x| x.to_bits()).collect();
        assert_eq!(bits(&g, &cache), want, "tn {tn} at {at}");
    }
}

#[test]
fn kv_cache_write_moves_bits_unchanged() {
    // -0.0, subnormals and the largest finite values travel bit-exact.
    let g = fresh();
    let vals = [-0.0f32, f32::MIN_POSITIVE / 8.0, -f32::MAX, f32::MAX, 1.5];
    let src: Vec<f32> = (0..2 * 5).map(|i| vals[i % vals.len()]).collect();
    let mut cache = g.upload(&tensor(&[0.0; 4 * 2 * 5], &[1, 4, 2, 5])).unwrap();
    g.kv_cache_write(
        &mut cache,
        &g.upload(&tensor(&src, &[1, 1, 2, 5])).unwrap(),
        2,
    )
    .unwrap();
    g.sync().unwrap();
    let got = bits(&g, &cache);
    let want: Vec<u32> = src.iter().map(|x| x.to_bits()).collect();
    assert_eq!(&got[2 * 10..3 * 10], &want[..]);
}

#[test]
fn kv_cache_write_from_a_view_with_a_byte_offset() {
    let g = fresh();
    let (b, cap, hkv, d) = (1usize, 4usize, 2usize, 3usize);
    let whole = data(41, 2 * hkv * d);
    let src_all = g.upload(&tensor(&whole, &[b, 2, hkv, d])).unwrap();
    let second = src_all
        .view(&[b, 1, hkv, d], &[hkv * d, hkv * d, d, 1], hkv * d * 4)
        .unwrap();
    let mut cache = g.upload(&tensor(&[0.0; 24], &[b, cap, hkv, d])).unwrap();
    g.kv_cache_write(&mut cache, &second, 3).unwrap();
    g.sync().unwrap();
    let got = g.download(&cache).unwrap().to_f32_vec().unwrap();
    assert_eq!(&got[3 * 6..], &whole[6..]);
    assert!(got[..18].iter().all(|v| *v == 0.0));
}

#[test]
fn kv_cache_write_refusals_leave_the_cache_unchanged() {
    let g = fresh();
    let (b, cap, hkv, d) = (2usize, 6usize, 2usize, 4usize);
    let mut cache = g.upload(&host(1, &[b, cap, hkv, d])).unwrap();
    let before = bits(&g, &cache);
    let src = |tn: usize, seed| g.upload(&host(seed, &[b, tn, hkv, d])).unwrap();
    for (tn, at) in [(2usize, 5usize), (7, 0), (1, 6), (1, usize::MAX)] {
        let r = g.kv_cache_write(&mut cache, &src(tn, 2), at);
        assert!(
            matches!(r, Err(OjasError::OutOfRange { .. })),
            "tn {tn} at {at}: {r:?}"
        );
    }
    let r = g.kv_cache_write(
        &mut cache,
        &g.upload(&host(3, &[b, 1, hkv, d + 1])).unwrap(),
        0,
    );
    assert!(matches!(r, Err(OjasError::Shape { .. })), "{r:?}");
    let r = g.kv_cache_write(&mut cache, &host(4, &[b, 1, hkv, d]), 0);
    assert!(matches!(r, Err(OjasError::Placement { .. })), "{r:?}");
    // A cache another handle shares is not written through.
    let other = cache.clone();
    let r = g.kv_cache_write(&mut cache, &src(1, 6), 0);
    assert!(matches!(r, Err(OjasError::Shape { .. })), "{r:?}");
    drop(other);
    g.sync().unwrap();
    assert_eq!(
        bits(&g, &cache),
        before,
        "a refused write changed the cache"
    );
    g.kv_cache_write(&mut cache, &src(2, 7), 4).unwrap();
    g.sync().unwrap();
}

#[test]
fn a_non_finite_source_is_reported_and_the_cache_is_unchanged() {
    let g = fresh();
    let (b, cap, hkv, d) = (2usize, 6usize, 2usize, 4usize);
    let mut cache = g.upload(&host(1, &[b, cap, hkv, d])).unwrap();
    let before = bits(&g, &cache);
    for val in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let n = b * 3 * hkv * d;
        for idx in [0, n / 2, n - 1] {
            let mut x = data(5, n);
            x[idx] = val;
            let r = g.kv_cache_write(
                &mut cache,
                &g.upload(&tensor(&x, &[b, 3, hkv, d])).unwrap(),
                1,
            );
            assert!(r.is_ok(), "the fault is deferred: {r:?}");
            match g.sync() {
                Err(OjasError::NonFinite { op }) => assert_eq!(op, "kv_cache_write"),
                other => panic!("{val} at {idx}: {other:?}"),
            }
            assert_eq!(
                bits(&g, &cache),
                before,
                "{val} at {idx}: the cache changed"
            );
        }
    }
}

// ---- cached_attention_forward ----------------------------------------------

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

/// f64 reference from the trait's definition: query i at position
/// kv_len - tq + i reads keys 0..=that position of KV head h / (H / Hkv).
fn naive(q: &[f32], k: &[f32], v: &[f32], s: Dims) -> Vec<f32> {
    let group = s.h / s.hkv;
    let scale = 1.0 / (s.d as f64).sqrt();
    let mut out = vec![0.0f32; s.b * s.tq * s.h * s.d];
    for b in 0..s.b {
        for i in 0..s.tq {
            let pos = s.kv_len - s.tq + i;
            for h in 0..s.h {
                let kvh = h / group;
                let qo = ((b * s.tq + i) * s.h + h) * s.d;
                let kat = |j: usize| ((b * s.cap + j) * s.hkv + kvh) * s.d;
                let scores: Vec<f64> = (0..=pos)
                    .map(|j| {
                        (0..s.d)
                            .map(|d| f64::from(q[qo + d]) * f64::from(k[kat(j) + d]))
                            .sum::<f64>()
                            * scale
                    })
                    .collect();
                let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let w: Vec<f64> = scores.iter().map(|x| (x - m).exp()).collect();
                let l: f64 = w.iter().sum();
                for d in 0..s.d {
                    let acc: f64 = (0..=pos).map(|j| w[j] * f64::from(v[kat(j) + d])).sum();
                    out[qo + d] = (acc / l) as f32;
                }
            }
        }
    }
    out
}

fn attend(g: &WgpuBackend, s: Dims, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let q = data(seed, s.b * s.tq * s.h * s.d);
    let mut k = data(seed + 1, s.b * s.cap * s.hkv * s.d);
    let mut v = data(seed + 2, s.b * s.cap * s.hkv * s.d);
    // Positions at or past kv_len are never read: poison them.
    for b in 0..s.b {
        for j in s.kv_len..s.cap {
            let at = (b * s.cap + j) * s.hkv * s.d;
            k[at..at + s.hkv * s.d].fill(f32::NAN);
            v[at..at + s.hkv * s.d].fill(f32::NAN);
        }
    }
    let want = naive(&q, &k, &v, s);
    let got = g
        .cached_attention_forward(
            &g.upload(&tensor(&q, &[s.b, s.tq, s.h, s.d])).unwrap(),
            &g.upload(&tensor(&k, &[s.b, s.cap, s.hkv, s.d])).unwrap(),
            &g.upload(&tensor(&v, &[s.b, s.cap, s.hkv, s.d])).unwrap(),
            s.kv_len,
        )
        .unwrap();
    assert_eq!(got.shape(), &[s.b, s.tq, s.h, s.d]);
    g.sync().unwrap_or_else(|e| panic!("{s:?}: {e:?}"));
    (g.download(&got).unwrap().to_f32_vec().unwrap(), want)
}

fn within(tag: &str, got: &[f32], want: &[f32], tol: f64) {
    assert_eq!(got.len(), want.len(), "{tag}");
    let mut worst = (0.0f64, 0usize);
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert!(a.is_finite(), "{tag}[{i}] = {a}");
        let e = (f64::from(*a) - f64::from(*b)).abs();
        if e > worst.0 {
            worst = (e, i);
        }
    }
    assert!(
        worst.0 <= tol,
        "{tag}: |err| {:.3e} at {} > {tol:.0e}",
        worst.0,
        worst.1
    );
}

#[test]
fn full_cache_without_grouping_equals_causal_sdpa_after_the_permute() {
    let g = fresh();
    let c = cpu();
    for (i, (b, t, h, d)) in [
        (1usize, 1usize, 1usize, 16usize),
        (2, 17, 3, 64),
        (1, 200, 2, 128),
        (1, 65, 4, 13),
    ]
    .into_iter()
    .enumerate()
    {
        let seed = 100 + 10 * i as u64;
        let cap = t + 5;
        let q = host(seed, &[b, t, h, d]);
        let k_full = host(seed + 1, &[b, cap, h, d]);
        let v_full = host(seed + 2, &[b, cap, h, d]);
        let got = g
            .cached_attention_forward(
                &g.upload(&q).unwrap(),
                &g.upload(&k_full).unwrap(),
                &g.upload(&v_full).unwrap(),
                t,
            )
            .unwrap();
        // The first t positions of the cache, permuted to [B, H, T, D].
        let first = |x: &Tensor| {
            let vals = x.to_f32_vec().unwrap();
            let mut out = Vec::with_capacity(b * t * h * d);
            for bb in 0..b {
                out.extend_from_slice(&vals[bb * cap * h * d..(bb * cap + t) * h * d]);
            }
            c.permute(&tensor(&out, &[b, t, h, d]), &[0, 2, 1, 3])
                .unwrap()
        };
        let qp = c.permute(&q, &[0, 2, 1, 3]).unwrap();
        let (kp, vp) = (first(&k_full), first(&v_full));
        let cpu_y = c
            .permute(
                &c.causal_sdpa_forward(&qp, &kp, &vp).unwrap(),
                &[0, 2, 1, 3],
            )
            .unwrap();
        let gpu_y = g
            .permute(
                &g.causal_sdpa_forward(
                    &g.upload(&qp).unwrap(),
                    &g.upload(&kp).unwrap(),
                    &g.upload(&vp).unwrap(),
                )
                .unwrap(),
                &[0, 2, 1, 3],
            )
            .unwrap();
        g.sync().unwrap();
        let got = g.download(&got).unwrap().to_f32_vec().unwrap();
        let tag = format!("[{b},{t},{h},{d}]");
        within(
            &format!("{tag} vs cpu sdpa"),
            &got,
            &cpu_y.to_f32_vec().unwrap(),
            1e-5,
        );
        within(
            &format!("{tag} vs wgpu sdpa"),
            &got,
            &g.download(&gpu_y).unwrap().to_f32_vec().unwrap(),
            1e-5,
        );
    }
}

#[test]
fn grouped_query_attention_matches_a_naive_f64_reference() {
    let g = fresh();
    let cases = [
        Dims {
            b: 1,
            tq: 1,
            h: 6,
            hkv: 2,
            d: 64,
            cap: 40,
            kv_len: 37,
        },
        Dims {
            b: 2,
            tq: 4,
            h: 6,
            hkv: 3,
            d: 32,
            cap: 50,
            kv_len: 50,
        },
        Dims {
            b: 1,
            tq: 5,
            h: 8,
            hkv: 1,
            d: 128,
            cap: 300,
            kv_len: 260,
        },
        Dims {
            b: 1,
            tq: 3,
            h: 4,
            hkv: 4,
            d: 7,
            cap: 9,
            kv_len: 3,
        },
        Dims {
            b: 1,
            tq: 1,
            h: 12,
            hkv: 12,
            d: 64,
            cap: 1024,
            kv_len: 1024,
        },
        Dims {
            b: 1,
            tq: 2,
            h: 4,
            hkv: 2,
            d: 256,
            cap: 16,
            kv_len: 9,
        },
        Dims {
            b: 1,
            tq: 1,
            h: 2,
            hkv: 1,
            d: 200,
            cap: 8,
            kv_len: 5,
        },
        Dims {
            b: 1,
            tq: 3,
            h: 6,
            hkv: 3,
            d: 129,
            cap: 12,
            kv_len: 7,
        },
    ];
    for (i, s) in cases.into_iter().enumerate() {
        let (got, want) = attend(&g, s, 200 + 10 * i as u64);
        within(&format!("{s:?}"), &got, &want, 1e-5);
    }
}

#[test]
fn split_boundaries_match_the_reference() {
    let g = fresh();
    for (i, kv_len) in [1usize, 63, 64, 65, 127, 128, 129, 1023, 1024, 1025, 2049]
        .into_iter()
        .enumerate()
    {
        for tq in [1usize, 3] {
            if tq > kv_len {
                continue;
            }
            let s = Dims {
                b: 1,
                tq,
                h: 2,
                hkv: 1,
                d: 16,
                cap: kv_len + 2,
                kv_len,
            };
            let (got, want) = attend(&g, s, 300 + 10 * i as u64 + tq as u64);
            within(&format!("{s:?}"), &got, &want, 1e-5);
        }
    }
}

#[test]
fn decode_after_each_cache_write_matches_the_reference() {
    let g = fresh();
    let (h, hkv, d, cap, steps) = (4usize, 2usize, 32usize, 24usize, 20usize);
    let ks = data(400, steps * hkv * d);
    let vs = data(401, steps * hkv * d);
    let mut kc = g
        .upload(&tensor(&vec![f32::NAN; cap * hkv * d], &[1, cap, hkv, d]))
        .unwrap();
    let mut vc = g
        .upload(&tensor(&vec![f32::NAN; cap * hkv * d], &[1, cap, hkv, d]))
        .unwrap();
    let row = hkv * d;
    let mut k_host = vec![f32::NAN; cap * row];
    let mut v_host = vec![f32::NAN; cap * row];
    for t in 0..steps {
        let kt = &ks[t * row..(t + 1) * row];
        let vt = &vs[t * row..(t + 1) * row];
        g.kv_cache_write(&mut kc, &g.upload(&tensor(kt, &[1, 1, hkv, d])).unwrap(), t)
            .unwrap();
        g.kv_cache_write(&mut vc, &g.upload(&tensor(vt, &[1, 1, hkv, d])).unwrap(), t)
            .unwrap();
        k_host[t * row..(t + 1) * row].copy_from_slice(kt);
        v_host[t * row..(t + 1) * row].copy_from_slice(vt);
        let q = data(500 + t as u64, h * d);
        let y = g
            .cached_attention_forward(
                &g.upload(&tensor(&q, &[1, 1, h, d])).unwrap(),
                &kc,
                &vc,
                t + 1,
            )
            .unwrap();
        g.sync().unwrap_or_else(|e| panic!("step {t}: {e:?}"));
        let s = Dims {
            b: 1,
            tq: 1,
            h,
            hkv,
            d,
            cap,
            kv_len: t + 1,
        };
        let want = naive(&q, &k_host, &v_host, s);
        within(
            &format!("step {t}"),
            &g.download(&y).unwrap().to_f32_vec().unwrap(),
            &want,
            1e-5,
        );
    }
}

#[test]
fn cached_attention_is_deterministic() {
    let g = fresh();
    let s = Dims {
        b: 2,
        tq: 3,
        h: 6,
        hkv: 2,
        d: 64,
        cap: 700,
        kv_len: 650,
    };
    let q = g.upload(&host(1, &[s.b, s.tq, s.h, s.d])).unwrap();
    let k = g.upload(&host(2, &[s.b, s.cap, s.hkv, s.d])).unwrap();
    let v = g.upload(&host(3, &[s.b, s.cap, s.hkv, s.d])).unwrap();
    let first = bits(
        &g,
        &g.cached_attention_forward(&q, &k, &v, s.kv_len).unwrap(),
    );
    for _ in 0..3 {
        assert_eq!(
            bits(
                &g,
                &g.cached_attention_forward(&q, &k, &v, s.kv_len).unwrap()
            ),
            first
        );
    }
}

#[test]
fn cached_attention_refusals() {
    let g = fresh();
    let (b, tq, h, hkv, d, cap) = (1usize, 2usize, 4usize, 2usize, 16usize, 8usize);
    let q = g.upload(&host(1, &[b, tq, h, d])).unwrap();
    let k = g.upload(&host(2, &[b, cap, hkv, d])).unwrap();
    let v = g.upload(&host(3, &[b, cap, hkv, d])).unwrap();
    for kv_len in [1usize, cap + 1] {
        let r = g.cached_attention_forward(&q, &k, &v, kv_len);
        assert!(
            matches!(r, Err(OjasError::OutOfRange { .. })),
            "kv_len {kv_len}: {r:?}"
        );
    }
    let q3 = g.upload(&host(4, &[b, tq, 3, d])).unwrap();
    let r = g.cached_attention_forward(&q3, &k, &v, 4);
    assert!(matches!(r, Err(OjasError::Shape { .. })), "{r:?}");
    let wide = |s: &[usize], seed| g.upload(&host(seed, s)).unwrap();
    let r = g.cached_attention_forward(
        &wide(&[1, 1, 1, 257], 5),
        &wide(&[1, 4, 1, 257], 6),
        &wide(&[1, 4, 1, 257], 7),
        2,
    );
    assert!(
        matches!(
            r,
            Err(OjasError::UnsupportedHeadDim {
                head_dim: 257,
                limit: 256
            })
        ),
        "{r:?}"
    );
    let r = g.cached_attention_forward(&host(1, &[b, tq, h, d]), &k, &v, 4);
    assert!(matches!(r, Err(OjasError::Placement { .. })), "{r:?}");
    g.sync().unwrap();
}

#[test]
fn non_finite_values_in_the_read_window_are_reported() {
    let g = fresh();
    let (b, tq, h, hkv, d, cap) = (1usize, 2usize, 4usize, 2usize, 16usize, 8usize);
    let kv_len = 5usize;
    let qs = [b, tq, h, d];
    let cs = [b, cap, hkv, d];
    let poison = |shape: &[usize], idx: usize, seed: u64, val: f32| {
        let mut x = data(seed, shape.iter().product());
        x[idx] = val;
        g.upload(&tensor(&x, shape)).unwrap()
    };
    let q = g.upload(&host(1, &qs)).unwrap();
    let k = g.upload(&host(2, &cs)).unwrap();
    let v = g.upload(&host(3, &cs)).unwrap();
    let last_in_window = ((kv_len - 1) * hkv + hkv - 1) * d + d - 1;
    for val in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let cases = [
            (
                "q",
                poison(&qs, b * tq * h * d - 1, 8, val),
                k.clone(),
                v.clone(),
            ),
            (
                "k",
                q.clone(),
                poison(&cs, last_in_window, 9, val),
                v.clone(),
            ),
            (
                "v",
                q.clone(),
                k.clone(),
                poison(&cs, last_in_window, 10, val),
            ),
            ("k first", q.clone(), poison(&cs, 0, 11, val), v.clone()),
        ];
        for (what, qq, kk, vv) in cases {
            assert!(g.cached_attention_forward(&qq, &kk, &vv, kv_len).is_ok());
            match g.sync() {
                Err(OjasError::NonFinite { op }) => {
                    assert_eq!(op, "cached_attention_forward", "{what} {val}")
                }
                other => panic!("{what} {val}: {other:?}"),
            }
        }
    }
    // Positions at or past kv_len are not read.
    let tail = poison(&cs, kv_len * hkv * d, 12, f32::NAN);
    g.cached_attention_forward(&q, &tail, &tail, kv_len)
        .unwrap();
    g.sync().unwrap();
    // A key only the last query reads: the earlier row stays clean only if
    // it never reads it, which the reference comparison checks; here the
    // poisoned key must still fault through the last row.
    let late = poison(&cs, (kv_len - 1) * hkv * d, 13, f32::NAN);
    g.cached_attention_forward(&q, &late, &v, kv_len).unwrap();
    assert!(matches!(g.sync(), Err(OjasError::NonFinite { .. })));
    // Scores that overflow f32.
    let huge = |shape: &[usize]| {
        g.upload(&tensor(&vec![1e30; shape.iter().product()], shape))
            .unwrap()
    };
    g.cached_attention_forward(&huge(&qs), &huge(&cs), &v, kv_len)
        .unwrap();
    assert!(matches!(
        g.sync(),
        Err(OjasError::NonFinite {
            op: "cached_attention_forward"
        })
    ));
}

#[test]
fn a_minus_infinity_key_faults_even_when_the_output_stays_finite() {
    // With q all positive, a -inf in one key makes that key's every score
    // -inf: exp gives a clean 0 and the output is finite, so only a check of
    // the score itself can see it.
    let g = fresh();
    let (h, hkv, d, cap, kv_len) = (2usize, 1usize, 16usize, 8usize, 6usize);
    let q: Vec<f32> = data(1, h * d).iter().map(|x| x.abs() + 0.1).collect();
    let mut k = data(2, cap * hkv * d);
    k[2 * hkv * d] = f32::NEG_INFINITY;
    let v = data(3, cap * hkv * d);
    let y = g
        .cached_attention_forward(
            &g.upload(&tensor(&q, &[1, 1, h, d])).unwrap(),
            &g.upload(&tensor(&k, &[1, cap, hkv, d])).unwrap(),
            &g.upload(&tensor(&v, &[1, cap, hkv, d])).unwrap(),
            kv_len,
        )
        .unwrap();
    match g.sync() {
        Err(OjasError::NonFinite { op }) => assert_eq!(op, "cached_attention_forward"),
        other => panic!("{other:?}"),
    }
    drop(y);
}

#[test]
fn an_earlier_query_never_reads_a_later_position() {
    // Row 0 of Tq = 2 sits at kv_len - 2; a NaN at kv_len - 1 is read only
    // by row 1, so row 0's output must match the reference computed without
    // that position at all.
    let g = fresh();
    let s = Dims {
        b: 1,
        tq: 2,
        h: 2,
        hkv: 2,
        d: 16,
        cap: 70,
        kv_len: 66,
    };
    let q = data(1, s.tq * s.h * s.d);
    let mut k = data(2, s.cap * s.hkv * s.d);
    let v = data(3, s.cap * s.hkv * s.d);
    let late = (s.kv_len - 1) * s.hkv * s.d;
    k[late..late + s.hkv * s.d].fill(1e4);
    let y = g
        .cached_attention_forward(
            &g.upload(&tensor(&q, &[1, s.tq, s.h, s.d])).unwrap(),
            &g.upload(&tensor(&k, &[1, s.cap, s.hkv, s.d])).unwrap(),
            &g.upload(&tensor(&v, &[1, s.cap, s.hkv, s.d])).unwrap(),
            s.kv_len,
        )
        .unwrap();
    g.sync().unwrap();
    let got = g.download(&y).unwrap().to_f32_vec().unwrap();
    let row0 = Dims {
        tq: 1,
        kv_len: s.kv_len - 1,
        ..s
    };
    let want0 = naive(&q[..s.h * s.d], &k, &v, row0);
    within("row 0", &got[..s.h * s.d], &want0, 1e-5);
}

/// Gate G4's device-vs-CPU row at 1e-5. It panics rather than passing if
/// the CPU method is unimplemented.
#[test]
fn cached_attention_matches_the_cpu_reference() {
    let g = fresh();
    let c = cpu();
    for (i, (tq, kv_len, h, hkv, d)) in [
        (1usize, 1024usize, 12usize, 12usize, 64usize),
        (4, 37, 6, 2, 128),
    ]
    .into_iter()
    .enumerate()
    {
        let cap = kv_len + 3;
        let q = host(50 + i as u64, &[1, tq, h, d]);
        let k = host(51 + i as u64, &[1, cap, hkv, d]);
        let v = host(52 + i as u64, &[1, cap, hkv, d]);
        let want = match c.cached_attention_forward(&q, &k, &v, kv_len) {
            Err(OjasError::Unsupported { .. }) => {
                panic!(
                    "CpuBackend::cached_attention_forward is not implemented; nothing to compare"
                )
            }
            other => other.unwrap(),
        };
        let got = g
            .cached_attention_forward(
                &g.upload(&q).unwrap(),
                &g.upload(&k).unwrap(),
                &g.upload(&v).unwrap(),
                kv_len,
            )
            .unwrap();
        g.sync().unwrap();
        within(
            &format!("cpu {tq} {kv_len}"),
            &g.download(&got).unwrap().to_f32_vec().unwrap(),
            &want.to_f32_vec().unwrap(),
            1e-5,
        );
    }
}

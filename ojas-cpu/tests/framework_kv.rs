//! `cached_attention_forward` (T4) and `kv_cache_write` (T5).
//!
//! Attention: bitwise equal to `permute → causal_sdpa_forward → permute`
//! under Exact when `kv_len == Tq` and `H == Hkv`; GQA against a naive f64
//! reference; unused cache capacity filled with NaN is never read.
//! Cache write: all-or-nothing, proved by snapshotting the cache's bits
//! around every refused call.

mod common;

use common::{assert_capacity, assert_nonfinite, assert_range, assert_shape, bits, SplitMix64};
use ojas_core::{Backend, Budget, DType, Numerics, OjasError, Tensor};
use ojas_cpu::CpuBackend;

fn t(data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_f32(data, shape, &Budget::new(u64::MAX)).unwrap()
}

fn f(x: &Tensor) -> Vec<f32> {
    x.to_f32_vec().unwrap()
}

/// `[B, Tcap, Hkv, D]` with time slots `0..kv_len` from `rng` and the rest
/// set to `pad`.
fn cache(rng: &mut SplitMix64, dims: [usize; 4], kv_len: usize, pad: f32) -> Vec<f32> {
    let [b, tcap, hkv, d] = dims;
    let mut out = Vec::with_capacity(b * tcap * hkv * d);
    for _ in 0..b {
        for slot in 0..tcap {
            for _ in 0..hkv * d {
                out.push(if slot < kv_len { rng.unit() } else { pad });
            }
        }
    }
    out
}

/// Naive causal GQA attention in f64, `[B, Tq, H, D]`.
#[allow(clippy::too_many_arguments)]
fn reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    [b, tq, h, d]: [usize; 4],
    tcap: usize,
    hkv: usize,
    kv_len: usize,
) -> Vec<f64> {
    let scale = 1.0 / (d as f64).sqrt();
    let group = h / hkv;
    let mut out = vec![0.0f64; b * tq * h * d];
    for bb in 0..b {
        for i in 0..tq {
            let pos = kv_len - tq + i;
            for hh in 0..h {
                let hk = hh / group;
                let qs = ((bb * tq + i) * h + hh) * d;
                let kv = |j: usize| ((bb * tcap + j) * hkv + hk) * d;
                let scores: Vec<f64> = (0..=pos)
                    .map(|j| {
                        (0..d)
                            .map(|x| q[qs + x] as f64 * k[kv(j) + x] as f64)
                            .sum::<f64>()
                            * scale
                    })
                    .collect();
                let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                let sum: f64 = exps.iter().sum();
                for (j, e) in exps.iter().enumerate() {
                    for x in 0..d {
                        out[qs + x] += e / sum * v[kv(j) + x] as f64;
                    }
                }
            }
        }
    }
    out
}

fn assert_close(what: &str, got: &[f32], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let err = (g as f64 - w).abs();
        assert!(
            err <= 1e-5 * (1.0 + w.abs()),
            "{what} [{i}]: {g} vs {w} (err {err:e})"
        );
    }
}

#[test]
fn full_cache_mha_equals_causal_sdpa_bit_for_bit_under_exact() {
    let mut rng = SplitMix64(0x4b01);
    for threads in [1usize, 4] {
        let be = CpuBackend::with_threads(Budget::new(u64::MAX), threads)
            .unwrap()
            .with_numerics(Numerics::Exact);
        for &(b, tq, h, d, extra) in &[
            (1usize, 1usize, 1usize, 1usize, 0usize),
            (2, 5, 3, 4, 0),
            (1, 17, 2, 8, 3),
            (2, 40, 4, 16, 0),
            (1, 300, 2, 8, 5),
        ] {
            let q = t(&rng.vec(b * tq * h * d, 1.0), &[b, tq, h, d]);
            let tcap = tq + extra;
            let kc = t(
                &cache(&mut rng, [b, tcap, h, d], tq, f32::NAN),
                &[b, tcap, h, d],
            );
            let vc = t(
                &cache(&mut rng, [b, tcap, h, d], tq, f32::NAN),
                &[b, tcap, h, d],
            );
            let got = be.cached_attention_forward(&q, &kc, &vc, tq).unwrap();
            assert_eq!(got.shape(), &[b, tq, h, d]);
            // The composed reference reads a `[B, T, H, D]` K and V of exactly
            // `tq` slots, permuted to `[B, H, T, D]`.
            let used = |c: &Tensor| {
                let all = f(c);
                let mut out = Vec::new();
                for bb in 0..b {
                    let start = bb * tcap * h * d;
                    out.extend_from_slice(&all[start..start + tq * h * d]);
                }
                t(&out, &[b, tq, h, d])
            };
            let perm = [0, 2, 1, 3];
            let qp = be.permute(&q, &perm).unwrap();
            let kp = be.permute(&used(&kc), &perm).unwrap();
            let vp = be.permute(&used(&vc), &perm).unwrap();
            let y = be.causal_sdpa_forward(&qp, &kp, &vp).unwrap();
            let want = be.permute(&y, &perm).unwrap();
            assert_eq!(
                bits(&f(&got)),
                bits(&f(&want)),
                "threads {threads} b{b} t{tq} h{h} d{d} tcap {tcap}"
            );
        }
    }
}

#[test]
fn gqa_and_partial_caches_match_an_f64_reference() {
    let mut rng = SplitMix64(0x4b02);
    for threads in [1usize, 3] {
        for numerics in [Numerics::Exact, Numerics::Fast] {
            let be = CpuBackend::with_threads(Budget::new(u64::MAX), threads)
                .unwrap()
                .with_numerics(numerics);
            for &(b, tq, h, hkv, d, kv_len, tcap) in &[
                (2usize, 3usize, 6usize, 2usize, 8usize, 11usize, 16usize),
                (1, 1, 4, 1, 16, 1, 4),
                (1, 1, 12, 4, 64, 37, 64),
                (3, 4, 4, 4, 5, 4, 9),
                (1, 9, 8, 2, 32, 300, 301),
            ] {
                let qd = rng.vec(b * tq * h * d, 1.0);
                let kd = cache(&mut rng, [b, tcap, hkv, d], kv_len, f32::NAN);
                let vd = cache(&mut rng, [b, tcap, hkv, d], kv_len, f32::NAN);
                let got = be
                    .cached_attention_forward(
                        &t(&qd, &[b, tq, h, d]),
                        &t(&kd, &[b, tcap, hkv, d]),
                        &t(&vd, &[b, tcap, hkv, d]),
                        kv_len,
                    )
                    .unwrap();
                let want = reference(&qd, &kd, &vd, [b, tq, h, d], tcap, hkv, kv_len);
                assert_close(
                    &format!(
                        "{numerics:?} t{threads} b{b} tq{tq} h{h}/{hkv} d{d} kv {kv_len}/{tcap}"
                    ),
                    &f(&got),
                    &want,
                );
            }
        }
    }
}

/// Slots past `kv_len` are never read: NaN there gives a finite result with
/// the same bits as zero padding and as a cache with no spare capacity.
#[test]
fn padding_past_kv_len_is_never_read() {
    let be = CpuBackend::new(Budget::new(u64::MAX)).with_numerics(Numerics::Exact);
    let (b, tq, h, hkv, d, kv_len, tcap) =
        (2usize, 2usize, 4usize, 2usize, 8usize, 7usize, 12usize);
    let mut rng = SplitMix64(0x4b03);
    let q = t(&rng.vec(b * tq * h * d, 1.0), &[b, tq, h, d]);
    let mut seeded = SplitMix64(0x4b04);
    let k_nan = cache(&mut seeded, [b, tcap, hkv, d], kv_len, f32::NAN);
    let v_nan = cache(&mut seeded, [b, tcap, hkv, d], kv_len, f32::INFINITY);
    let mut seeded = SplitMix64(0x4b04);
    let k_zero = cache(&mut seeded, [b, tcap, hkv, d], kv_len, 0.0);
    let v_zero = cache(&mut seeded, [b, tcap, hkv, d], kv_len, 0.0);
    let shape = [b, tcap, hkv, d];
    let nan = f(&be
        .cached_attention_forward(&q, &t(&k_nan, &shape), &t(&v_nan, &shape), kv_len)
        .unwrap());
    assert!(nan.iter().all(|x| x.is_finite()));
    let zero = f(&be
        .cached_attention_forward(&q, &t(&k_zero, &shape), &t(&v_zero, &shape), kv_len)
        .unwrap());
    assert_eq!(bits(&nan), bits(&zero));
    let trim = |c: &[f32]| {
        let mut out = Vec::new();
        for bb in 0..b {
            let start = bb * tcap * hkv * d;
            out.extend_from_slice(&c[start..start + kv_len * hkv * d]);
        }
        t(&out, &[b, kv_len, hkv, d])
    };
    let tight = f(&be
        .cached_attention_forward(&q, &trim(&k_nan), &trim(&v_nan), kv_len)
        .unwrap());
    assert_eq!(bits(&nan), bits(&tight));
}

#[test]
fn cached_attention_refusals() {
    let be = CpuBackend::new(Budget::new(u64::MAX));
    let q = t(&[0.5; 2 * 3 * 4 * 2], &[2, 3, 4, 2]);
    let kc = t(&[0.25; 2 * 8 * 2 * 2], &[2, 8, 2, 2]);
    let vc = t(&[0.75; 2 * 8 * 2 * 2], &[2, 8, 2, 2]);
    assert!(be.cached_attention_forward(&q, &kc, &vc, 3).is_ok());
    assert!(be.cached_attention_forward(&q, &kc, &vc, 8).is_ok());
    assert_range(be.cached_attention_forward(&q, &kc, &vc, 2));
    assert_range(be.cached_attention_forward(&q, &kc, &vc, 9));
    assert_shape(be.cached_attention_forward(&t(&[0.5; 24], &[2, 3, 4]), &kc, &vc, 3));
    assert_shape(be.cached_attention_forward(
        &q,
        &kc,
        &t(&[0.75; 2 * 7 * 2 * 2], &[2, 7, 2, 2]),
        3,
    ));
    assert_shape(be.cached_attention_forward(
        &q,
        &t(&[0.25; 8 * 2 * 2], &[1, 8, 2, 2]),
        &t(&[0.25; 8 * 2 * 2], &[1, 8, 2, 2]),
        3,
    ));
    assert_shape(be.cached_attention_forward(
        &q,
        &t(&[0.25; 2 * 8 * 2 * 3], &[2, 8, 2, 3]),
        &t(&[0.25; 2 * 8 * 2 * 3], &[2, 8, 2, 3]),
        3,
    ));
    // Four query heads over three KV heads.
    assert_shape(be.cached_attention_forward(
        &q,
        &t(&[0.25; 2 * 8 * 3 * 2], &[2, 8, 3, 2]),
        &t(&[0.25; 2 * 8 * 3 * 2], &[2, 8, 3, 2]),
        3,
    ));
    let qu = Tensor::from_u32(&[0; 24], &[2, 3, 4, 1], &Budget::new(u64::MAX)).unwrap();
    assert!(matches!(
        be.cached_attention_forward(&qu, &kc, &vc, 3),
        Err(OjasError::Dtype {
            expected: DType::F32,
            ..
        })
    ));
    // NaN in q or in the used prefix is refused before any charge.
    let none = CpuBackend::new(Budget::new(0));
    let mut qd = vec![0.5f32; 48];
    qd[40] = f32::NAN;
    assert_nonfinite(none.cached_attention_forward(&t(&qd, &[2, 3, 4, 2]), &kc, &vc, 3));
    let mut kd = vec![0.25f32; 64];
    // Batch 1, time slot 2: inside kv_len 3, outside nothing else.
    kd[32 + 2 * 4] = f32::INFINITY;
    assert_nonfinite(none.cached_attention_forward(&q, &t(&kd, &[2, 8, 2, 2]), &vc, 3));
    // The same infinity is past kv_len 2 for a single query: not read.
    let q1 = t(&[0.5; 2 * 4 * 2], &[2, 1, 4, 2]);
    assert!(be
        .cached_attention_forward(&q1, &t(&kd, &[2, 8, 2, 2]), &vc, 2)
        .is_ok());
    assert_capacity(none.cached_attention_forward(&q, &kc, &vc, 3));
    assert_eq!(none.budget().live_bytes().unwrap(), 0);
    assert_eq!(be.budget().live_bytes().unwrap(), 0);
}

/// A decode loop: prefill writes 5 positions, then one token at a time is
/// written and attended to. Every step matches the f64 reference over the
/// whole sequence so far.
#[test]
fn write_then_attend_decode_loop_matches_the_reference() {
    let be = CpuBackend::with_threads(Budget::new(u64::MAX), 2).unwrap();
    let (b, h, hkv, d, tcap, total) = (2usize, 4usize, 2usize, 8usize, 16usize, 12usize);
    let mut rng = SplitMix64(0x4b05);
    let kseq = rng.vec(b * total * hkv * d, 1.0);
    let vseq = rng.vec(b * total * hkv * d, 1.0);
    let qseq = rng.vec(b * total * h * d, 1.0);
    let slice = |seq: &[f32], heads: usize, from: usize, to: usize| {
        let mut out = Vec::new();
        for bb in 0..b {
            let row = heads * d;
            out.extend_from_slice(&seq[(bb * total + from) * row..(bb * total + to) * row]);
        }
        t(&out, &[b, to - from, heads, d])
    };
    let budget = Budget::new(u64::MAX);
    let nan = vec![f32::NAN; b * tcap * hkv * d];
    let mut kc = Tensor::from_f32(&nan, &[b, tcap, hkv, d], &budget).unwrap();
    let mut vc = Tensor::from_f32(&nan, &[b, tcap, hkv, d], &budget).unwrap();
    let mut pos = 0;
    for step in [5usize, 1, 1, 1, 1, 1, 1, 1] {
        be.kv_cache_write(&mut kc, &slice(&kseq, hkv, pos, pos + step), pos)
            .unwrap();
        be.kv_cache_write(&mut vc, &slice(&vseq, hkv, pos, pos + step), pos)
            .unwrap();
        let q = slice(&qseq, h, pos, pos + step);
        let got = be
            .cached_attention_forward(&q, &kc, &vc, pos + step)
            .unwrap();
        pos += step;
        let want = reference(&f(&q), &f(&kc), &f(&vc), [b, step, h, d], tcap, hkv, pos);
        assert_close(&format!("decode to {pos}"), &f(&got), &want);
    }
    // Slots never written still hold NaN.
    let k = f(&kc);
    for bb in 0..b {
        for slot in pos..tcap {
            let at = ((bb * tcap + slot) * hkv) * d;
            assert!(k[at..at + hkv * d].iter().all(|x| x.is_nan()));
        }
    }
}

#[test]
fn cache_write_places_src_and_keeps_every_other_bit() {
    let be = CpuBackend::new(Budget::new(u64::MAX));
    let (b, tcap, hkv, d) = (2usize, 6usize, 3usize, 2usize);
    let mut rng = SplitMix64(0x4b06);
    let mut before = rng.vec(b * tcap * hkv * d, 1.0);
    before[5] = f32::NAN;
    before[b * tcap * hkv * d - 1] = -0.0;
    let mut c = t(&before, &[b, tcap, hkv, d]);
    let src = rng.vec(b * 2 * hkv * d, 1.0);
    be.kv_cache_write(&mut c, &t(&src, &[b, 2, hkv, d]), 3)
        .unwrap();
    let after = f(&c);
    let row = hkv * d;
    for bb in 0..b {
        for slot in 0..tcap {
            let at = (bb * tcap + slot) * row;
            let want: Vec<f32> = if (3..5).contains(&slot) {
                let s = (bb * 2 + slot - 3) * row;
                src[s..s + row].to_vec()
            } else {
                before[at..at + row].to_vec()
            };
            assert_eq!(bits(&after[at..at + row]), bits(&want), "b{bb} slot {slot}");
        }
    }
}

/// Every refused write leaves the cache's bits exactly as they were.
#[test]
fn cache_write_is_all_or_nothing() {
    let (b, tcap, hkv, d) = (2usize, 6usize, 3usize, 2usize);
    let mut rng = SplitMix64(0x4b07);
    let mut init = rng.vec(b * tcap * hkv * d, 1.0);
    init[7] = f32::NAN;
    let shape = [b, tcap, hkv, d];
    let ok_src = t(&rng.vec(b * 2 * hkv * d, 1.0), &[b, 2, hkv, d]);
    let be = CpuBackend::new(Budget::new(u64::MAX));

    // The refusal, after proving the cache's bits did not move.
    let unchanged = |cache: &Tensor, result: Result<(), OjasError>, what: &str| {
        assert_eq!(bits(&f(cache)), bits(&init), "{what}: cache changed");
        result
    };

    let mut c = t(&init, &shape);
    let r = be.kv_cache_write(&mut c, &ok_src, 5);
    assert_range(unchanged(&c, r, "past capacity"));
    let r = be.kv_cache_write(&mut c, &ok_src, usize::MAX);
    assert_range(unchanged(&c, r, "position overflow"));
    let wrong_heads = t(&[0.5; 2 * 2 * 2 * 2], &[2, 2, 2, 2]);
    let r = be.kv_cache_write(&mut c, &wrong_heads, 0);
    assert_shape(unchanged(&c, r, "head mismatch"));
    let wrong_batch = t(&vec![0.5; 2 * hkv * d], &[1, 2, hkv, d]);
    let r = be.kv_cache_write(&mut c, &wrong_batch, 0);
    assert_shape(unchanged(&c, r, "batch mismatch"));
    let mut bad = f(&ok_src);
    bad[13] = f32::NAN;
    let r = be.kv_cache_write(&mut c, &t(&bad, &[b, 2, hkv, d]), 1);
    assert_nonfinite(unchanged(&c, r, "NaN in src"));
    let src_u32 = Tensor::from_u32(&[0; 24], &[b, 2, hkv, d], &Budget::new(u64::MAX)).unwrap();
    let r = be.kv_cache_write(&mut c, &src_u32, 1);
    assert!(matches!(
        unchanged(&c, r, "u32 src"),
        Err(OjasError::Dtype {
            expected: DType::F32,
            ..
        })
    ));

    // A cache whose storage is shared cannot be written; neither handle
    // changes.
    let other = c.clone();
    let r = be.kv_cache_write(&mut c, &ok_src, 1);
    assert_shape(unchanged(&c, r, "shared cache"));
    assert_eq!(bits(&f(&other)), bits(&init));
    drop(other);

    // No room for the scratch: refused, the cache unchanged, nothing left
    // charged.
    let tight = CpuBackend::new(Budget::new(16));
    let r = tight.kv_cache_write(&mut c, &ok_src, 1);
    assert_capacity(unchanged(&c, r, "budget"));
    assert_eq!(tight.budget().live_bytes().unwrap(), 0);

    // And the same write then succeeds.
    be.kv_cache_write(&mut c, &ok_src, 1).unwrap();
    assert_ne!(bits(&f(&c)), bits(&init));
    assert_eq!(be.budget().live_bytes().unwrap(), 0);
}

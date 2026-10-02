//! Causal SDPA on wgpu against `CpuBackend` (`Numerics::Exact`): the lengths
//! that straddle query/key blocks, B and H above one, head dims that are not
//! a supported width (padded inside the kernel), views with a byte offset,
//! a non-finite value in each input, overflowing scores, future keys that
//! must not leak, and run-to-run determinism.
//!
//! Tolerance is the shared `TOL * max(1, max |cpu|)` per output tensor.

mod common;

use common::*;
use ojas_core::{Backend, OjasError, Tensor};
use ojas_wgpu::WgpuBackend;

fn qkvg(seed: u64, shape: &[usize]) -> [Tensor; 4] {
    [
        host(seed, shape),
        host(seed + 1, shape),
        host(seed + 2, shape),
        host(seed + 3, shape),
    ]
}

fn parity(tag: &str, shape: [usize; 4], seed: u64) {
    let c = cpu();
    let g = gpu();
    let [q, k, v, gy] = qkvg(seed, &shape);
    let (dq, dk, dv) = (up(&q), up(&k), up(&v));
    let tag = format!("{tag} {shape:?}");
    close(
        &format!("{tag} y"),
        &g.causal_sdpa_forward(&dq, &dk, &dv).unwrap(),
        &c.causal_sdpa_forward(&q, &k, &v).unwrap(),
    );
    let got = g.causal_sdpa_backward(&dq, &dk, &dv, &up(&gy)).unwrap();
    let want = c.causal_sdpa_backward(&q, &k, &v, &gy).unwrap();
    close(&format!("{tag} dq"), &got.0, &want.0);
    close(&format!("{tag} dk"), &got.1, &want.1);
    close(&format!("{tag} dv"), &got.2, &want.2);
}

#[test]
fn lengths_that_straddle_blocks_match_cpu() {
    for (i, &t) in [1usize, 2, 17, 31, 32, 33, 64, 255].iter().enumerate() {
        parity("len", [2, 3, t, 64], 6000 + 10 * i as u64);
    }
}

#[test]
fn long_sequences_match_cpu() {
    parity("long", [1, 2, 1024, 64], 6100);
    parity("long", [1, 1, 2048, 64], 6110);
    parity("long d128", [1, 1, 1024, 128], 6120);
}

#[test]
fn every_head_dim_up_to_the_limit_matches_cpu() {
    // Supported widths 16/32/64/128 and dims padded up to them.
    for (i, &d) in [1usize, 3, 13, 16, 24, 32, 48, 63, 64, 80, 100, 127, 128]
        .iter()
        .enumerate()
    {
        parity("dim", [1, 2, 33, d], 6200 + 10 * i as u64);
    }
}

#[test]
fn views_with_a_byte_offset_match_cpu() {
    let c = cpu();
    let g = gpu();
    let shape = [2usize, 2, 37, 24];
    let n: usize = shape.iter().product();
    let strides = [2 * 37 * 24, 37 * 24, 24, 1];
    let mut views = Vec::new();
    let mut hosts = Vec::new();
    for (i, pad) in [5usize, 1, 12, 7].into_iter().enumerate() {
        let whole = host(6300 + i as u64, &[n + pad]);
        let vals = whole.to_f32_vec().unwrap();
        hosts.push(Tensor::from_f32(&vals[pad..], &shape, host_budget()).unwrap());
        views.push(up(&whole).view(&shape, &strides, pad * 4).unwrap());
    }
    close(
        "offset y",
        &g.causal_sdpa_forward(&views[0], &views[1], &views[2])
            .unwrap(),
        &c.causal_sdpa_forward(&hosts[0], &hosts[1], &hosts[2])
            .unwrap(),
    );
    let got = g
        .causal_sdpa_backward(&views[0], &views[1], &views[2], &views[3])
        .unwrap();
    let want = c
        .causal_sdpa_backward(&hosts[0], &hosts[1], &hosts[2], &hosts[3])
        .unwrap();
    close("offset dq", &got.0, &want.0);
    close("offset dk", &got.1, &want.1);
    close("offset dv", &got.2, &want.2);
}

fn nonfinite_op(r: Result<(), OjasError>) -> &'static str {
    match r {
        Err(OjasError::NonFinite { op }) => op,
        other => panic!("expected NonFinite, got {other:?}"),
    }
}

#[test]
fn a_non_finite_value_in_any_input_is_reported() {
    let shape = [1usize, 2, 41, 32];
    // Row 40 is the last query, so every key and value row is live for it.
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for which in 0..4 {
            let g = &fresh();
            g.sync().unwrap();
            let mut ins = qkvg(6400, &shape);
            let mut vals = ins[which].to_f32_vec().unwrap();
            vals[(41 + 23) * 32 + 5] = bad;
            ins[which] = Tensor::from_f32(&vals, &shape, host_budget()).unwrap();
            let d: Vec<Tensor> = ins.iter().map(|t| g.upload(t).unwrap()).collect();
            if which < 3 {
                let _ = g.causal_sdpa_forward(&d[0], &d[1], &d[2]).unwrap();
                assert_eq!(
                    nonfinite_op(g.sync()),
                    "causal_sdpa_forward",
                    "{bad} in input {which}"
                );
            }
            let _ = g.causal_sdpa_backward(&d[0], &d[1], &d[2], &d[3]).unwrap();
            assert_eq!(
                nonfinite_op(g.sync()),
                "causal_sdpa_backward",
                "{bad} in input {which}"
            );
        }
    }
}

#[test]
fn overflowing_scores_are_reported() {
    let g = &fresh();
    g.sync().unwrap();
    let shape = [1usize, 1, 9, 64];
    let big = Tensor::from_f32(&vec![1.0e20; 9 * 64], &shape, host_budget()).unwrap();
    let v = g.upload(&host(6500, &shape)).unwrap();
    let qk = g.upload(&big).unwrap();
    let _ = g.causal_sdpa_forward(&qk, &qk, &v).unwrap();
    assert_eq!(nonfinite_op(g.sync()), "causal_sdpa_forward");
    let gy = g.upload(&host(6501, &shape)).unwrap();
    let _ = g.causal_sdpa_backward(&qk, &qk, &v, &gy).unwrap();
    assert_eq!(nonfinite_op(g.sync()), "causal_sdpa_backward");
}

fn bits(g: &WgpuBackend, t: &Tensor) -> Vec<u32> {
    g.download(t)
        .unwrap()
        .to_f32_vec()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

#[test]
fn future_keys_never_reach_an_earlier_row_across_blocks() {
    // Rows 0..=cut must be bit-identical whatever the later keys and values
    // hold, NaN included, and no fault may be raised for them.
    let g = &fresh();
    g.sync().unwrap();
    let (h, t, d) = (2usize, 100usize, 48usize);
    let shape = [1, h, t, d];
    let [q, k, v, gy] = qkvg(6600, &shape);
    for cut in [0usize, 15, 31, 32, 40, 63, 64, 98] {
        let mut k2 = k.to_f32_vec().unwrap();
        let mut v2 = v.to_f32_vec().unwrap();
        for head in 0..h {
            for pos in cut + 1..t {
                for j in 0..d {
                    let idx = (head * t + pos) * d + j;
                    k2[idx] = if j % 2 == 0 { f32::NAN } else { 1.0e30 };
                    v2[idx] = f32::NAN;
                }
            }
        }
        let k2 = Tensor::from_f32(&k2, &shape, host_budget()).unwrap();
        let v2 = Tensor::from_f32(&v2, &shape, host_budget()).unwrap();
        let (dq, dg) = (g.upload(&q).unwrap(), g.upload(&gy).unwrap());
        let (dk, dv) = (g.upload(&k).unwrap(), g.upload(&v).unwrap());
        let (dk2, dv2) = (g.upload(&k2).unwrap(), g.upload(&v2).unwrap());
        let a = bits(g, &g.causal_sdpa_forward(&dq, &dk, &dv).unwrap());
        let ga = bits(g, &g.causal_sdpa_backward(&dq, &dk, &dv, &dg).unwrap().0);
        // The poisoned rows are live for the later queries, so a fault is
        // expected (unless nothing follows the cut); only rows 0..=cut are
        // compared.
        let z = g.causal_sdpa_forward(&dq, &dk2, &dv2).unwrap();
        let gz = g.causal_sdpa_backward(&dq, &dk2, &dv2, &dg).unwrap().0;
        let fault = g.sync();
        assert_eq!(fault.is_err(), cut + 1 < t, "cut {cut}: {fault:?}");
        let (z, gz) = (bits(g, &z), bits(g, &gz));
        for head in 0..h {
            for pos in 0..=cut {
                for j in 0..d {
                    let idx = (head * t + pos) * d + j;
                    assert_eq!(a[idx], z[idx], "cut {cut} head {head} row {pos} y");
                    assert_eq!(ga[idx], gz[idx], "cut {cut} head {head} row {pos} dq");
                }
            }
        }
    }
}

#[test]
fn results_repeat_bit_for_bit() {
    let g = gpu();
    let shape = [2usize, 3, 255, 64];
    let [q, k, v, gy] = qkvg(6700, &shape);
    let (q, k, v, gy) = (up(&q), up(&k), up(&v), up(&gy));
    let run = || {
        let y = g.causal_sdpa_forward(&q, &k, &v).unwrap();
        let (a, b, c) = g.causal_sdpa_backward(&q, &k, &v, &gy).unwrap();
        [bits(g, &y), bits(g, &a), bits(g, &b), bits(g, &c)]
    };
    let first = run();
    for i in 0..3 {
        assert!(run() == first, "run {i} differs");
    }
}

#[test]
fn an_earlier_query_never_reaches_a_later_key_gradient() {
    // dK and dV of key j sum over queries i >= j only. Poisoning Q and dO of
    // query `row` must leave dK/dV of every key after it bit-identical, even
    // though the poisoned row is live for the keys up to it.
    let g = &fresh();
    g.sync().unwrap();
    let (h, t, d) = (2usize, 70usize, 32usize);
    let shape = [1, h, t, d];
    let [q, k, v, gy] = qkvg(6800, &shape);
    let (dk, dv) = (g.upload(&k).unwrap(), g.upload(&v).unwrap());
    let clean = g
        .causal_sdpa_backward(&g.upload(&q).unwrap(), &dk, &dv, &g.upload(&gy).unwrap())
        .unwrap();
    let (ck, cv) = (bits(g, &clean.1), bits(g, &clean.2));
    for row in [0usize, 14, 15, 16, 33, 63, 64, 68] {
        let mut q2 = q.to_f32_vec().unwrap();
        let mut g2 = gy.to_f32_vec().unwrap();
        for head in 0..h {
            for j in 0..d {
                q2[(head * t + row) * d + j] = f32::NAN;
                g2[(head * t + row) * d + j] = f32::NAN;
            }
        }
        let q2 = g
            .upload(&Tensor::from_f32(&q2, &shape, host_budget()).unwrap())
            .unwrap();
        let g2 = g
            .upload(&Tensor::from_f32(&g2, &shape, host_budget()).unwrap())
            .unwrap();
        let got = g.causal_sdpa_backward(&q2, &dk, &dv, &g2).unwrap();
        assert!(g.sync().is_err(), "row {row}: the poisoned row is live");
        let (gk, gv) = (bits(g, &got.1), bits(g, &got.2));
        for head in 0..h {
            for key in row + 1..t {
                for j in 0..d {
                    let idx = (head * t + key) * d + j;
                    assert_eq!(gk[idx], ck[idx], "row {row} key {key} dk");
                    assert_eq!(gv[idx], cv[idx], "row {row} key {key} dv");
                }
            }
        }
    }
}

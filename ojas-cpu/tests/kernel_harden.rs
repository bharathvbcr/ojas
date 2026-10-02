//! Break attempts against the tiled linear and causal kernels.
//! A failure is a real mismatch with the scalar reduction order, the causal
//! mask, or the existing non-finite / empty-tensor policy.

use ojas_core::{AdamWConfig, Backend, Budget, DType, Numerics, Tensor, RMS_NORM_EPS};
use ojas_cpu::{scaled_lr, CosineSchedule, CpuBackend, GradAccumulator};

mod common;
use common::{assert_nonfinite, assert_range, assert_shape, bits, f32t, SplitMix64};

fn wide() -> CpuBackend {
    CpuBackend::new(Budget::new(64 << 20)).with_numerics(Numerics::Exact)
}

fn ref_forward(x: &[f32], w: &[f32], rows: usize, kin: usize, nout: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; rows * nout];
    for row in 0..rows {
        for col in 0..nout {
            let mut acc = 0.0f32;
            for inner in 0..kin {
                acc += x[row * kin + inner] * w[col * kin + inner];
            }
            y[row * nout + col] = acc;
        }
    }
    y
}

fn ref_backward(
    x: &[f32],
    w: &[f32],
    gy: &[f32],
    rows: usize,
    kin: usize,
    nout: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut gx = vec![0.0f32; rows * kin];
    let mut gw = vec![0.0f32; nout * kin];
    for row in 0..rows {
        for inner in 0..kin {
            let mut acc = 0.0f32;
            for col in 0..nout {
                acc += gy[row * nout + col] * w[col * kin + inner];
            }
            gx[row * kin + inner] = acc;
        }
    }
    for col in 0..nout {
        for inner in 0..kin {
            let mut acc = 0.0f32;
            for row in 0..rows {
                acc += gy[row * nout + col] * x[row * kin + inner];
            }
            gw[col * kin + inner] = acc;
        }
    }
    (gx, gw)
}

fn assert_bits(got: &[f32], expect: &[f32]) {
    assert_eq!(got.len(), expect.len());
    for (i, (a, b)) in got.iter().zip(expect).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "index {i}: {a} vs {b}");
    }
}

fn check_linear(cpu: &CpuBackend, rows: usize, kin: usize, nout: usize, seed: u64) {
    let mut rng = SplitMix64(seed);
    let x = rng.vec(rows * kin, 0.5);
    let w = rng.vec(nout * kin, 0.5);
    let gy = rng.vec(rows * nout, 0.5);
    let xt = f32t(cpu, &x, &[rows, kin]);
    let wt = f32t(cpu, &w, &[nout, kin]);
    let y = cpu.linear_forward(&xt, &wt).unwrap().to_f32_vec().unwrap();
    assert_bits(&y, &ref_forward(&x, &w, rows, kin, nout));
    let y2 = cpu.linear_forward(&xt, &wt).unwrap().to_f32_vec().unwrap();
    assert_eq!(bits(&y), bits(&y2));

    let gt = f32t(cpu, &gy, &[rows, nout]);
    let (gx, gw) = cpu.linear_backward(&xt, &wt, &gt).unwrap();
    let (rx, rw) = ref_backward(&x, &w, &gy, rows, kin, nout);
    assert_bits(&gx.to_f32_vec().unwrap(), &rx);
    assert_bits(&gw.to_f32_vec().unwrap(), &rw);

    for threads in [1usize, 3, 8] {
        let parallel = CpuBackend::with_threads(Budget::new(64 << 20), threads)
            .unwrap()
            .with_numerics(Numerics::Exact);
        let yp = parallel
            .linear_forward(
                &f32t(&parallel, &x, &[rows, kin]),
                &f32t(&parallel, &w, &[nout, kin]),
            )
            .unwrap()
            .to_f32_vec()
            .unwrap();
        assert_eq!(bits(&y), bits(&yp), "forward threads {threads}");
        let (gxp, gwp) = parallel
            .linear_backward(
                &f32t(&parallel, &x, &[rows, kin]),
                &f32t(&parallel, &w, &[nout, kin]),
                &f32t(&parallel, &gy, &[rows, nout]),
            )
            .unwrap();
        assert_eq!(
            bits(&gx.to_f32_vec().unwrap()),
            bits(&gxp.to_f32_vec().unwrap()),
            "gx threads {threads}"
        );
        assert_eq!(
            bits(&gw.to_f32_vec().unwrap()),
            bits(&gwp.to_f32_vec().unwrap()),
            "gw threads {threads}"
        );
    }
}

/// Neither the inner width nor the output width is a multiple of 8 or of 4.
#[test]
fn linear_widths_not_multiple_of_eight_or_four_match_scalar() {
    let cpu = wide();
    // Below the blocked grain. Column tile is 8, so nout=6 and nout=10 leave a tail.
    check_linear(&cpu, 3, 6, 3, 21);
    check_linear(&cpu, 5, 2, 6, 22);
    check_linear(&cpu, 16, 10, 10, 23);
    // Above the grain: four-row block, eight-column tile, remainder 2.
    check_linear(&cpu, 48, 10, 10, 24);
    check_linear(&cpu, 1_202, 2, 2, 25);
    check_linear(&cpu, 37, 5, 23, 26);
    // Middle of the weight row is the only nonzero, including a remainder column.
    let rows = 2usize;
    let kin = 10usize;
    let nout = 10usize;
    let mut x = vec![0.0f32; rows * kin];
    let mut w = vec![0.0f32; nout * kin];
    x[5] = 2.0;
    x[kin + 5] = -3.0;
    for col in 0..nout {
        w[col * kin + 5] = 4.0;
    }
    let y = cpu
        .linear_forward(&f32t(&cpu, &x, &[rows, kin]), &f32t(&cpu, &w, &[nout, kin]))
        .unwrap()
        .to_f32_vec()
        .unwrap();
    assert_bits(&y, &ref_forward(&x, &w, rows, kin, nout));
    assert!(y[..nout].iter().all(|v| v.to_bits() == 8.0f32.to_bits()));
    assert!(y[nout..]
        .iter()
        .all(|v| v.to_bits() == (-12.0f32).to_bits()));
}

fn strides_of(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0usize; shape.len()];
    let mut acc = 1usize;
    for i in (0..shape.len()).rev() {
        strides[i] = acc;
        if i == 0 {
            break;
        }
        acc = acc.checked_mul(shape[i]).expect("stride");
    }
    strides
}

struct Padded {
    view: Tensor,
    prefix: usize,
    payload: Vec<f32>,
    suffix: usize,
}

fn padded(cpu: &CpuBackend, data: &[f32], shape: &[usize], prefix: usize, suffix: usize) -> Padded {
    let mut raw = vec![f32::from_bits(0x7f00_0123); prefix + data.len() + suffix];
    for (i, slot) in raw.iter_mut().take(prefix).enumerate() {
        *slot = f32::from_bits(0x5a00_0001 + i as u32);
    }
    raw[prefix..prefix + data.len()].copy_from_slice(data);
    for (i, slot) in raw.iter_mut().skip(prefix + data.len()).enumerate() {
        *slot = f32::from_bits(0x6b00_00a5 + i as u32);
    }
    let storage = Tensor::from_f32(&raw, &[raw.len()], cpu.budget()).unwrap();
    let view = storage
        .narrow(prefix * 4, shape, &strides_of(shape))
        .unwrap();
    assert!(view.byte_offset() > 0);
    assert_eq!(view.to_f32_vec().unwrap(), data);
    Padded {
        view,
        prefix,
        payload: data.to_vec(),
        suffix,
    }
}

fn assert_window(view: &Tensor, prefix: usize, payload: &[f32], suffix: usize) {
    let total = prefix + payload.len() + suffix;
    let whole = view.view(&[total], &[1], 0).unwrap().to_f32_vec().unwrap();
    for (i, value) in whole.iter().take(prefix).enumerate() {
        assert_eq!(
            value.to_bits(),
            0x5a00_0001 + i as u32,
            "prefix byte {i} changed"
        );
    }
    assert_eq!(&whole[prefix..prefix + payload.len()], payload);
    for (i, value) in whole.iter().skip(prefix + payload.len()).enumerate() {
        assert_eq!(
            value.to_bits(),
            0x6b00_00a5 + i as u32,
            "suffix byte {i} changed"
        );
    }
}

fn assert_padding_intact(padded: &Padded) {
    assert_window(&padded.view, padded.prefix, &padded.payload, padded.suffix);
}

#[test]
fn linear_byte_offset_views_leave_bytes_outside_the_window_unchanged() {
    let cpu = wide();
    let rows = 5usize;
    let kin = 6usize;
    let nout = 3usize;
    let mut rng = SplitMix64(31);
    let x = rng.vec(rows * kin, 0.35);
    let w = rng.vec(nout * kin, 0.35);
    let gy = rng.vec(rows * nout, 0.35);
    let xp = padded(&cpu, &x, &[rows, kin], 7, 5);
    let wp = padded(&cpu, &w, &[nout, kin], 3, 9);
    let gp = padded(&cpu, &gy, &[rows, nout], 4, 2);

    let y = cpu.linear_forward(&xp.view, &wp.view).unwrap();
    assert_eq!(y.byte_offset(), 0);
    assert_bits(
        &y.to_f32_vec().unwrap(),
        &ref_forward(&x, &w, rows, kin, nout),
    );
    assert_padding_intact(&xp);
    assert_padding_intact(&wp);

    let (gx, gw) = cpu.linear_backward(&xp.view, &wp.view, &gp.view).unwrap();
    let (rx, rw) = ref_backward(&x, &w, &gy, rows, kin, nout);
    assert_bits(&gx.to_f32_vec().unwrap(), &rx);
    assert_bits(&gw.to_f32_vec().unwrap(), &rw);
    assert_eq!(gx.byte_offset(), 0);
    assert_eq!(gw.byte_offset(), 0);
    assert_padding_intact(&xp);
    assert_padding_intact(&wp);
    assert_padding_intact(&gp);

    // The returned gradients are placed into uniquely owned offset windows.
    // Bytes in front of and behind each window stay at the sentinel bits.
    let gx_data = gx.to_f32_vec().unwrap();
    let gw_data = gw.to_f32_vec().unwrap();
    let mut gx_dest = padded(&cpu, &vec![0.0; gx_data.len()], gx.shape(), 6, 3).view;
    let mut gw_dest = padded(&cpu, &vec![0.0; gw_data.len()], gw.shape(), 2, 8).view;
    gx_dest.write_f32(&gx_data).unwrap();
    gw_dest.write_f32(&gw_data).unwrap();
    assert_window(&gx_dest, 6, &gx_data, 3);
    assert_window(&gw_dest, 2, &gw_data, 8);
}

#[test]
fn noncontiguous_linear_and_attention_views_are_shape_errors() {
    let cpu = wide();
    let base = f32t(&cpu, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[8]);
    let before = base.to_f32_vec().unwrap();
    let gap = base.view(&[2, 2], &[4, 2], 0).unwrap();
    assert!(!gap.is_contiguous().unwrap());
    let w = f32t(&cpu, &[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    assert_shape(cpu.linear_forward(&gap, &w));
    assert_shape(cpu.linear_forward(&w, &gap));
    assert_shape(cpu.linear_backward(&gap, &w, &w));
    let offset_gap = base.view(&[2, 2], &[4, 2], 4).unwrap();
    assert_eq!(offset_gap.byte_offset(), 4);
    assert!(!offset_gap.is_contiguous().unwrap());
    assert_shape(cpu.linear_forward(&offset_gap, &w));
    assert_eq!(base.to_f32_vec().unwrap(), before);

    let q = f32t(&cpu, &[0.1; 8], &[1, 1, 2, 4]);
    let k = f32t(&cpu, &[0.2; 8], &[1, 1, 2, 4]);
    let v = f32t(&cpu, &[0.3; 8], &[1, 1, 2, 4]);
    let wide_q = f32t(&cpu, &[0.1; 16], &[16]);
    let nc_q = wide_q.view(&[1, 1, 2, 4], &[8, 8, 4, 2], 0).unwrap();
    assert!(!nc_q.is_contiguous().unwrap());
    assert_shape(cpu.causal_sdpa_forward(&nc_q, &k, &v));
    assert_shape(cpu.causal_sdpa_backward(&q, &k, &v, &nc_q));
}

#[test]
fn nan_in_the_middle_of_a_weight_row_is_nonfinite() {
    let cpu = wide();
    // Policy: any non-finite input is NonFinite before the product. A NaN
    // past the first tile lane must not become a finite partial dot.
    for (nout, kin, at) in [
        (1usize, 10usize, 5usize),
        (10, 10, 5),
        (13, 11, 5),
        (8, 17, 8),
    ] {
        let rows = if nout * kin * 4 >= 4_096 { 4 } else { 2 };
        let mut w = vec![0.25f32; nout * kin];
        w[at] = f32::NAN;
        let x = vec![0.5f32; rows * kin];
        let xt = f32t(&cpu, &x, &[rows, kin]);
        let wt = f32t(&cpu, &w, &[nout, kin]);
        assert_nonfinite(cpu.linear_forward(&xt, &wt));
        let finite_w = vec![0.25f32; nout * kin];
        let y_shape_cols = nout;
        let gy = vec![0.1f32; rows * y_shape_cols];
        let gt = f32t(&cpu, &gy, &[rows, nout]);
        let wt_ok = f32t(&cpu, &finite_w, &[nout, kin]);
        assert_nonfinite(cpu.linear_backward(&xt, &wt, &gt));
        let mut gy_nan = gy.clone();
        let mid = gy_nan.len() / 2;
        gy_nan[mid] = f32::NAN;
        assert_nonfinite(cpu.linear_backward(&xt, &wt_ok, &f32t(&cpu, &gy_nan, &[rows, nout])));
        assert_eq!(xt.to_f32_vec().unwrap(), x);
        assert!(wt.to_f32_vec().unwrap()[at].is_nan());
    }
}

#[test]
fn zero_rows_and_zero_columns_are_shape_errors() {
    let cpu = wide();
    let x = f32t(&cpu, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let w = f32t(&cpu, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3]);
    let x_bits = bits(&x.to_f32_vec().unwrap());
    let w_bits = bits(&w.to_f32_vec().unwrap());
    for (xs, ws) in [
        (&[0usize, 3][..], &[2, 3][..]),
        (&[2, 3], &[0, 3]),
        (&[2, 0], &[3, 0]),
        (&[0, 0], &[1, 0]),
        (&[4, 0, 3], &[2, 3]),
    ] {
        let empty_x = Tensor::zeros(xs, DType::F32, cpu.budget()).unwrap();
        let empty_w = if ws == [2, 3].as_slice() {
            w.clone()
        } else {
            Tensor::zeros(ws, DType::F32, cpu.budget()).unwrap()
        };
        assert_shape(cpu.linear_forward(&empty_x, &empty_w));
    }
    assert_eq!(bits(&x.to_f32_vec().unwrap()), x_bits);
    assert_eq!(bits(&w.to_f32_vec().unwrap()), w_bits);

    let q = f32t(&cpu, &[0.2; 8], &[1, 1, 2, 4]);
    let q_bits = bits(&q.to_f32_vec().unwrap());
    for shape in [[1usize, 1, 0, 4], [1, 1, 4, 0], [0, 2, 2, 4], [1, 0, 2, 4]] {
        let z = Tensor::zeros(&shape, DType::F32, cpu.budget()).unwrap();
        assert_shape(cpu.causal_sdpa_forward(&z, &z, &z));
        assert_shape(cpu.causal_sdpa_backward(&z, &z, &z, &z));
    }
    assert_eq!(bits(&q.to_f32_vec().unwrap()), q_bits);

    let cfg = ojas_core::MuonNs5Config {
        lr: 0.02,
        momentum: 0.9,
        weight_decay: 0.0,
        nesterov: false,
    };
    let mut param = Tensor::zeros(&[0, 4], DType::F32, cpu.budget()).unwrap();
    let grad = Tensor::zeros(&[0, 4], DType::F32, cpu.budget()).unwrap();
    let mut mom = Tensor::zeros(&[0, 4], DType::F32, cpu.budget()).unwrap();
    assert_shape(cpu.muon_ns5_step(&mut param, &grad, &mut mom, cfg));
    let mut param = Tensor::zeros(&[4, 0], DType::F32, cpu.budget()).unwrap();
    let grad = Tensor::zeros(&[4, 0], DType::F32, cpu.budget()).unwrap();
    let mut mom = Tensor::zeros(&[4, 0], DType::F32, cpu.budget()).unwrap();
    assert_shape(cpu.muon_ns5_step(&mut param, &grad, &mut mom, cfg));
}

fn causal_reference(q: &[f32], k: &[f32], v: &[f32], time: usize, dim: usize) -> Vec<f32> {
    let scale = 1.0 / (dim as f32).sqrt();
    let mut out = vec![0.0f32; time * dim];
    for t in 0..time {
        let mut scores = vec![0.0f32; t + 1];
        let mut max_score = f32::NEG_INFINITY;
        for j in 0..=t {
            let mut dot = 0.0f32;
            for d in 0..dim {
                dot += q[t * dim + d] * k[j * dim + d];
            }
            let score = dot * scale;
            scores[j] = score;
            if score > max_score {
                max_score = score;
            }
        }
        let mut sum = 0.0f32;
        let mut probs = vec![0.0f32; t + 1];
        for j in 0..=t {
            let e = (scores[j] - max_score).exp();
            probs[j] = e;
            sum += e;
        }
        for p in &mut probs {
            *p /= sum;
        }
        for d in 0..dim {
            let mut acc = 0.0f32;
            for j in 0..=t {
                acc += probs[j] * v[j * dim + d];
            }
            out[t * dim + d] = acc;
        }
    }
    out
}

#[test]
fn attention_head_dim_not_multiple_of_eight_or_four_matches_scalar() {
    let cpu = wide();
    for (time, dim, seed) in [
        (32usize, 6usize, 41u64),
        (10, 7, 42),
        (9, 3, 43),
        (5, 2, 44),
    ] {
        let mut rng = SplitMix64(seed);
        let n = time * dim;
        let qv = rng.vec(n, 0.2);
        let kv = rng.vec(n, 0.2);
        let vv = rng.vec(n, 0.5);
        let q = f32t(&cpu, &qv, &[1, 1, time, dim]);
        let k = f32t(&cpu, &kv, &[1, 1, time, dim]);
        let v = f32t(&cpu, &vv, &[1, 1, time, dim]);
        let y = cpu
            .causal_sdpa_forward(&q, &k, &v)
            .unwrap()
            .to_f32_vec()
            .unwrap();
        assert_bits(&y, &causal_reference(&qv, &kv, &vv, time, dim));
        let y2 = cpu
            .causal_sdpa_forward(&q, &k, &v)
            .unwrap()
            .to_f32_vec()
            .unwrap();
        assert_eq!(bits(&y), bits(&y2));
    }
}

#[test]
fn causal_t32_d64_future_key_does_not_touch_earlier_positions() {
    let cpu = wide();
    let time = 32usize;
    let dim = 64usize;
    let n = time * dim;
    let qv = vec![0.2f32; n];
    let kv = vec![0.1f32; n];
    let mut vv = vec![0.0f32; n];
    for t in 0..time {
        for d in 0..dim {
            vv[t * dim + d] = (t + 1) as f32;
        }
    }
    let q = f32t(&cpu, &qv, &[1, 1, time, dim]);
    let v = f32t(&cpu, &vv, &[1, 1, time, dim]);
    let y0 = cpu
        .causal_sdpa_forward(&q, &f32t(&cpu, &kv, &[1, 1, time, dim]), &v)
        .unwrap()
        .to_f32_vec()
        .unwrap();

    // Spike sits in the middle of the head, not on an 8-wide boundary, and
    // also at the last lane. Both are past position 0's allowed keys.
    let mut spiked = kv.clone();
    spiked[31 * dim + 30] = 40.0;
    spiked[31 * dim + 63] = -25.0;
    let y1 = cpu
        .causal_sdpa_forward(&q, &f32t(&cpu, &spiked, &[1, 1, time, dim]), &v)
        .unwrap()
        .to_f32_vec()
        .unwrap();

    for t in 0..31 {
        assert_eq!(
            bits(&y0[t * dim..(t + 1) * dim]),
            bits(&y1[t * dim..(t + 1) * dim]),
            "position {t} changed because of the key at t=31"
        );
    }
    assert_ne!(
        bits(&y0[31 * dim..32 * dim]),
        bits(&y1[31 * dim..32 * dim]),
        "position 31 ignored the key at t=31"
    );
    assert_bits(&y0, &causal_reference(&qv, &kv, &vv, time, dim));
    assert_bits(&y1, &causal_reference(&qv, &spiked, &vv, time, dim));

    // A key at index 8 is visible from there on, and invisible before it.
    let mut at8 = kv.clone();
    at8[8 * dim + 8] = 40.0;
    let y8 = cpu
        .causal_sdpa_forward(&q, &f32t(&cpu, &at8, &[1, 1, time, dim]), &v)
        .unwrap()
        .to_f32_vec()
        .unwrap();
    assert_eq!(bits(&y0[..8 * dim]), bits(&y8[..8 * dim]));
    assert_ne!(bits(&y0[8 * dim..9 * dim]), bits(&y8[8 * dim..9 * dim]));

    let gy = vec![0.05f32; n];
    let (gq0, _, _) = cpu
        .causal_sdpa_backward(
            &q,
            &f32t(&cpu, &kv, &[1, 1, time, dim]),
            &v,
            &f32t(&cpu, &gy, &[1, 1, time, dim]),
        )
        .unwrap();
    let (gq1, _, _) = cpu
        .causal_sdpa_backward(
            &q,
            &f32t(&cpu, &spiked, &[1, 1, time, dim]),
            &v,
            &f32t(&cpu, &gy, &[1, 1, time, dim]),
        )
        .unwrap();
    let gq0 = gq0.to_f32_vec().unwrap();
    let gq1 = gq1.to_f32_vec().unwrap();
    assert_eq!(bits(&gq0[..dim]), bits(&gq1[..dim]));
    assert_ne!(bits(&gq0[31 * dim..]), bits(&gq1[31 * dim..]));
}

fn rope_cache(time: usize, dim: usize) -> (Vec<f32>, Vec<f32>) {
    let half = dim / 2;
    let mut cos = vec![0.0f32; time * dim];
    let mut sin = vec![0.0f32; time * dim];
    for pos in 0..time {
        for i in 0..half {
            let inv = 10_000.0f64.powf(-((2 * i) as f64) / (dim as f64));
            let freq = (pos as f64) * inv;
            let c = freq.cos() as f32;
            let s = freq.sin() as f32;
            cos[pos * dim + i] = c;
            cos[pos * dim + half + i] = c;
            sin[pos * dim + i] = s;
            sin[pos * dim + half + i] = s;
        }
    }
    (cos, sin)
}

struct LargerBits {
    loss: u32,
    attn: Vec<u32>,
    logits: Vec<u32>,
    wq: Vec<u32>,
    norm: Vec<u32>,
}

#[allow(clippy::too_many_arguments)]
fn larger_step(
    cpu: &CpuBackend,
    x: &[f32],
    norm: &[f32],
    wq: &[f32],
    wk: &[f32],
    wv: &[f32],
    wo: &[f32],
    cos: &[f32],
    sin: &[f32],
    targets: &[u32],
) -> LargerBits {
    let (b, t, d, vocab) = (2usize, 32usize, 64usize, 128usize);
    let schedule = CosineSchedule::new(4, 20).unwrap();
    let mult = schedule.multiplier(8).unwrap();
    let x_t = f32t(cpu, x, &[b, t, d]);
    let mut norm_w = f32t(cpu, norm, &[d]);
    let mut wq_t = f32t(cpu, wq, &[d, d]);
    let wk_t = f32t(cpu, wk, &[d, d]);
    let wv_t = f32t(cpu, wv, &[d, d]);
    let wo_t = f32t(cpu, wo, &[vocab, d]);
    let cos_t = f32t(cpu, cos, &[t, d]);
    let sin_t = f32t(cpu, sin, &[t, d]);
    let targets_t = Tensor::from_u32(targets, &[b, t], cpu.budget()).unwrap();

    let h = cpu.rms_norm_forward(&x_t, &norm_w, RMS_NORM_EPS).unwrap();
    let q = cpu.linear_forward(&h, &wq_t).unwrap();
    let k = cpu.linear_forward(&h, &wk_t).unwrap();
    let v = cpu.linear_forward(&h, &wv_t).unwrap();
    let q_bt = q
        .view(&[b, t, 1, d], &strides_of(&[b, t, 1, d]), q.byte_offset())
        .unwrap();
    let k_bt = k
        .view(&[b, t, 1, d], &strides_of(&[b, t, 1, d]), k.byte_offset())
        .unwrap();
    let q_r = cpu.rope_half_split_forward(&q_bt, &cos_t, &sin_t).unwrap();
    let k_r = cpu.rope_half_split_forward(&k_bt, &cos_t, &sin_t).unwrap();
    let q_a = q_r
        .view(&[b, 1, t, d], &strides_of(&[b, 1, t, d]), q_r.byte_offset())
        .unwrap();
    let k_a = k_r
        .view(&[b, 1, t, d], &strides_of(&[b, 1, t, d]), k_r.byte_offset())
        .unwrap();
    let v_a = v
        .view(&[b, 1, t, d], &strides_of(&[b, 1, t, d]), v.byte_offset())
        .unwrap();
    let attn = cpu.causal_sdpa_forward(&q_a, &k_a, &v_a).unwrap();
    let y = attn
        .view(&[b, t, d], &strides_of(&[b, t, d]), attn.byte_offset())
        .unwrap();
    let logits = cpu.linear_forward(&y, &wo_t).unwrap();
    let loss = cpu
        .cross_entropy_mean_forward(&logits, &targets_t, None)
        .unwrap()
        .to_f32_vec()
        .unwrap()[0];
    let g_logits = cpu
        .cross_entropy_mean_backward(&logits, &targets_t, None)
        .unwrap();
    let (g_y, _) = cpu.linear_backward(&y, &wo_t, &g_logits).unwrap();
    let g_attn = g_y
        .view(&[b, 1, t, d], &strides_of(&[b, 1, t, d]), g_y.byte_offset())
        .unwrap();
    let (g_q, g_k, g_v) = cpu.causal_sdpa_backward(&q_a, &k_a, &v_a, &g_attn).unwrap();
    let g_q_bt = g_q
        .view(&[b, t, 1, d], &strides_of(&[b, t, 1, d]), g_q.byte_offset())
        .unwrap();
    let g_k_bt = g_k
        .view(&[b, t, 1, d], &strides_of(&[b, t, 1, d]), g_k.byte_offset())
        .unwrap();
    let g_q_lin = cpu
        .rope_half_split_backward(&g_q_bt, &cos_t, &sin_t)
        .unwrap()
        .view(&[b, t, d], &strides_of(&[b, t, d]), 0)
        .unwrap();
    let g_k_lin = cpu
        .rope_half_split_backward(&g_k_bt, &cos_t, &sin_t)
        .unwrap()
        .view(&[b, t, d], &strides_of(&[b, t, d]), 0)
        .unwrap();
    let g_v_lin = g_v
        .view(&[b, t, d], &strides_of(&[b, t, d]), g_v.byte_offset())
        .unwrap();
    let (gq_h, g_wq) = cpu.linear_backward(&h, &wq_t, &g_q_lin).unwrap();
    let (gk_h, _) = cpu.linear_backward(&h, &wk_t, &g_k_lin).unwrap();
    let (gv_h, _) = cpu.linear_backward(&h, &wv_t, &g_v_lin).unwrap();
    let mut g_h = vec![0.0f32; b * t * d];
    let a = gq_h.to_f32_vec().unwrap();
    let bk = gk_h.to_f32_vec().unwrap();
    let c = gv_h.to_f32_vec().unwrap();
    for i in 0..g_h.len() {
        g_h[i] = a[i] + bk[i] + c[i];
    }
    let g_h = f32t(cpu, &g_h, &[b, t, d]);
    let (_, g_norm) = cpu
        .rms_norm_backward(&x_t, &norm_w, &g_h, RMS_NORM_EPS)
        .unwrap();
    let lr = scaled_lr(6e-4, mult).unwrap();
    let mut m1 = Tensor::zeros(&[d, d], DType::F32, cpu.budget()).unwrap();
    let mut m2 = Tensor::zeros(&[d, d], DType::F32, cpu.budget()).unwrap();
    cpu.adamw_step(
        &mut wq_t,
        &g_wq,
        &mut m1,
        &mut m2,
        0,
        AdamWConfig::nanolab(lr, 0.1),
    )
    .unwrap();
    let mut nm1 = Tensor::zeros(&[d], DType::F32, cpu.budget()).unwrap();
    let mut nm2 = Tensor::zeros(&[d], DType::F32, cpu.budget()).unwrap();
    cpu.adamw_step(
        &mut norm_w,
        &g_norm,
        &mut nm1,
        &mut nm2,
        0,
        AdamWConfig::nanolab(lr, 0.0),
    )
    .unwrap();
    LargerBits {
        loss: loss.to_bits(),
        attn: bits(&attn.to_f32_vec().unwrap()),
        logits: bits(&logits.to_f32_vec().unwrap()),
        wq: bits(&wq_t.to_f32_vec().unwrap()),
        norm: bits(&norm_w.to_f32_vec().unwrap()),
    }
}

#[test]
fn two_identical_larger_steps_are_bit_identical() {
    let (b, t, d, vocab) = (2usize, 32usize, 64usize, 128usize);
    let mut rng = SplitMix64(1);
    let x = rng.vec(b * t * d, 0.02);
    let norm = rng
        .vec(d, 0.02)
        .into_iter()
        .map(|v| 1.0 + v)
        .collect::<Vec<_>>();
    let wq = rng.vec(d * d, 0.02);
    let wk = rng.vec(d * d, 0.02);
    let wv = rng.vec(d * d, 0.02);
    let wo = rng.vec(vocab * d, 0.02);
    let (cos, sin) = rope_cache(t, d);
    let targets = (0..b * t)
        .map(|_| (rng.next_u64() % (vocab as u64)) as u32)
        .collect::<Vec<_>>();

    let one = CpuBackend::new(Budget::new(64 << 20));
    let a = larger_step(&one, &x, &norm, &wq, &wk, &wv, &wo, &cos, &sin, &targets);
    let bts = larger_step(&one, &x, &norm, &wq, &wk, &wv, &wo, &cos, &sin, &targets);
    assert_eq!(a.loss, bts.loss);
    assert_eq!(a.attn, bts.attn);
    assert_eq!(a.logits, bts.logits);
    assert_eq!(a.wq, bts.wq);
    assert_eq!(a.norm, bts.norm);
    assert!(f32::from_bits(a.loss).is_finite());

    let six = CpuBackend::with_threads(Budget::new(64 << 20), 6).unwrap();
    let threaded = larger_step(&six, &x, &norm, &wq, &wk, &wv, &wo, &cos, &sin, &targets);
    assert_eq!(a.loss, threaded.loss);
    assert_eq!(a.attn, threaded.attn);
    assert_eq!(a.logits, threaded.logits);
    assert_eq!(a.wq, threaded.wq);
    assert_eq!(a.norm, threaded.norm);
}

#[test]
fn accumulation_count_and_warmup_edges() {
    assert_shape(GradAccumulator::new(0).map(|_| ()));
    let mut acc = GradAccumulator::new(2).unwrap();
    assert_shape(acc.add(&[1.0]).map(|_| ()));
    assert_eq!(acc.count(), 0);
    assert_range(acc.mean().map(|_| ()));

    acc.add(&[f32::MAX, 1.0]).unwrap();
    assert_eq!(acc.count(), 1);
    // MAX + MAX is not a finite f32. The failed add must not commit.
    assert_nonfinite(acc.add(&[f32::MAX, 0.0]).map(|_| ()));
    assert_eq!(acc.count(), 1);
    assert_eq!(acc.mean().unwrap(), vec![f32::MAX, 1.0]);
    assert_nonfinite(acc.add(&[f32::INFINITY, 0.0]).map(|_| ()));
    assert_eq!(acc.count(), 1);

    let mut at_cap = GradAccumulator::new(1).unwrap();
    for _ in 0..(1 << 24) {
        at_cap.add(&[1.0]).unwrap();
    }
    assert_eq!(at_cap.count(), 1 << 24);
    assert_eq!(at_cap.mean().unwrap(), vec![1.0]);
    assert_range(at_cap.add(&[1.0]).map(|_| ()));
    assert_eq!(at_cap.count(), 1 << 24);
    assert_eq!(at_cap.mean().unwrap(), vec![1.0]);

    assert_range(CosineSchedule::new(0, 1).map(|_| ()));
    let one = CosineSchedule::new(1, 1).unwrap();
    assert_eq!(one.multiplier(0).unwrap(), 1.0);
    assert!((one.multiplier(1).unwrap() - 1.0).abs() < 1e-12);
    assert!((one.multiplier(2).unwrap() - 0.1).abs() < 1e-12);

    let tied = CosineSchedule::new(5, 5).unwrap();
    assert_eq!(tied.multiplier(4).unwrap(), 1.0);
    assert!((tied.multiplier(5).unwrap() - 1.0).abs() < 1e-12);
    assert!((tied.multiplier(6).unwrap() - 0.1).abs() < 1e-12);

    let exact = CosineSchedule::new(1 << 53, 1 << 53).unwrap();
    assert!((exact.multiplier((1 << 53) - 1).unwrap() - 1.0).abs() < 1e-12);
    assert_range(exact.multiplier(1 << 53).map(|_| ()));
    assert_nonfinite(scaled_lr(1e308, 1e10).map(|_| ()));
    assert_eq!(scaled_lr(0.0, 0.25).unwrap(), 0.0);
}

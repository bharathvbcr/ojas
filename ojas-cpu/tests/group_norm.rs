//! `Backend::group_norm_forward` on the CPU, torch's `nn.GroupNorm`.
//!
//! - Values are held to an `f64` reference written here from the definition
//!   (statistics over the `C / G` channels and every position of one image,
//!   biased variance, then the per-channel affine).
//! - Relations to LayerNorm are bit-exact: `groups == C` is LayerNorm over
//!   each channel's `H * W` values and `groups == 1` over each image.
//! - A group whose mean is huge against its spread is held to an exact
//!   integer reference that a one-pass variance cannot meet.
//! - The bits do not depend on the thread count or on `Numerics`.
//! - Every refusal, a non-finite operand or output and an over-budget output
//!   use the variants the neighbouring ops use.

use ojas_core::{Autocast, AutocastMode, Backend, Budget, DType, Numerics, OjasError, Tensor};
use ojas_cpu::CpuBackend;

mod common;
use common::{assert_capacity, assert_nonfinite, assert_range, assert_shape, bits, SplitMix64};

const EPS: f32 = 1e-5;

fn backend(threads: usize, numerics: Numerics) -> CpuBackend {
    CpuBackend::with_threads(Budget::new(1 << 32), threads)
        .unwrap()
        .with_numerics(numerics)
}

fn t(data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_f32(data, shape, &Budget::new(u64::MAX)).unwrap()
}

fn v(t: &Tensor) -> Vec<f32> {
    t.to_f32_vec().unwrap()
}

/// torch's definition, in `f64` throughout: per `(n, group)` the mean and the
/// biased variance of the group's values, then `(x - mean) / sqrt(var + eps)`
/// scaled and shifted by the channel's `weight` and `bias`.
fn reference(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    [n, c, h, w]: [usize; 4],
    groups: usize,
    eps: f32,
) -> Vec<f64> {
    let per = c / groups;
    let plane = h * w;
    let len = per * plane;
    let mut out = vec![0.0f64; n * c * plane];
    for img in 0..n {
        for g in 0..groups {
            let start = (img * c + g * per) * plane;
            let vals: Vec<f64> = x[start..start + len]
                .iter()
                .map(|&q| f64::from(q))
                .collect();
            let mean = vals.iter().sum::<f64>() / len as f64;
            let var = vals.iter().map(|q| (q - mean) * (q - mean)).sum::<f64>() / len as f64;
            let rstd = 1.0 / (var + f64::from(eps)).sqrt();
            for (i, q) in vals.iter().enumerate() {
                let ch = g * per + i / plane;
                out[start + i] = (q - mean) * rstd * f64::from(weight[ch]) + f64::from(bias[ch]);
            }
        }
    }
    out
}

/// `max |got - want| / max |want|`, or the plain error when `want` is all 0.
fn normalized(got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0.0f64, |m, w| m.max(w.abs()));
    let err = got
        .iter()
        .zip(want)
        .fold(0.0f64, |m, (g, w)| m.max((f64::from(*g) - w).abs()));
    if scale == 0.0 {
        err
    } else {
        err / scale
    }
}

fn assert_close(what: &str, got: &[f32], want: &[f64], tol: f64) {
    let err = normalized(got, want);
    assert!(err <= tol, "{what}: normalized error {err:e} > {tol:e}");
}

/// Random `[n, c, h, w]` input and per-channel `weight` and `bias`.
fn random_case(seed: u64, shape: [usize; 4]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut rng = SplitMix64(seed);
    let [n, c, h, w] = shape;
    // A per-channel offset, so a group's channels differ in mean.
    let offsets = rng.vec(c, 3.0);
    let mut x = rng.vec(n * c * h * w, 2.0);
    for (i, q) in x.iter_mut().enumerate() {
        *q += offsets[(i / (h * w)) % c];
    }
    let weight = rng.vec(c, 1.5);
    let bias = rng.vec(c, 1.0);
    (x, weight, bias)
}

#[test]
fn matches_the_f64_reference_for_every_grouping_n_gt_1_and_non_square() {
    let cpu = backend(2, Numerics::Exact);
    // 12 channels, so groups 1, 2, 3, 4, 6 and 12 (instance norm) all divide.
    for (shape, seed) in [
        ([2, 12, 5, 7], 1u64),
        ([3, 12, 1, 9], 2),
        ([1, 12, 8, 3], 3),
    ] {
        let (x, w, b) = random_case(seed, shape);
        let input = t(&x, &shape);
        let (wt, bt) = (t(&w, &[12]), t(&b, &[12]));
        for groups in [1, 2, 3, 4, 6, 12] {
            let got = cpu
                .group_norm_forward(&input, &wt, &bt, groups, EPS)
                .unwrap();
            assert_eq!(got.shape(), shape.as_slice());
            assert_eq!(got.dtype(), DType::F32);
            assert_close(
                &format!("{shape:?} groups {groups}"),
                &v(&got),
                &reference(&x, &w, &b, shape, groups, EPS),
                5e-6,
            );
        }
    }
}

#[test]
fn a_tiny_hand_checked_case() {
    // [1, 2, 1, 2], one group: values 1, 3, 5, 7 -> mean 4, var 5.
    let x = t(&[1.0, 3.0, 5.0, 7.0], &[1, 2, 1, 2]);
    let w = t(&[2.0, -1.0], &[2]);
    let b = t(&[0.5, 10.0], &[2]);
    let eps = f64::from(1e-3f32);
    let cpu = backend(1, Numerics::Exact);
    let got = v(&cpu.group_norm_forward(&x, &w, &b, 1, 1e-3).unwrap());
    // Channel 0 has weight 2 and bias 0.5, channel 1 weight -1 and bias 10.
    let affine = |dev: f64, rstd: f64, channel: usize| {
        let (w, b) = [(2.0, 0.5), (-1.0, 10.0)][channel];
        dev * rstd * w + b
    };
    let rstd = 1.0 / (5.0f64 + eps).sqrt();
    let want = [
        affine(-3.0, rstd, 0),
        affine(-1.0, rstd, 0),
        affine(1.0, rstd, 1),
        affine(3.0, rstd, 1),
    ];
    assert_close("hand", &got, &want, 1e-6);
    // Two groups: each channel is its own pair, mean 2 and 6, var 1.
    let got = v(&cpu.group_norm_forward(&x, &w, &b, 2, 1e-3).unwrap());
    let rstd = 1.0 / (1.0f64 + eps).sqrt();
    let want = [
        affine(-1.0, rstd, 0),
        affine(1.0, rstd, 0),
        affine(-1.0, rstd, 1),
        affine(1.0, rstd, 1),
    ];
    assert_close("hand, two groups", &got, &want, 1e-6);
}

/// `group_norm(x, w, b, groups)` against `layer_norm_forward` over rows of
/// `row` values (unit weight, zero bias), then the channel's affine in the
/// same two `f32` steps.
fn assert_is_layer_norm_of_rows(shape: [usize; 4], groups: usize, seed: u64) {
    let cpu = backend(3, Numerics::Exact);
    let [n, c, h, w] = shape;
    let (x, weight, bias) = random_case(seed, shape);
    let got = v(&cpu
        .group_norm_forward(
            &t(&x, &shape),
            &t(&weight, &[c]),
            &t(&bias, &[c]),
            groups,
            EPS,
        )
        .unwrap());
    let row = (c / groups) * h * w;
    let rows = t(&x, &[n * groups, row]);
    let unit = t(&vec![1.0; row], &[row]);
    let zero = t(&vec![0.0; row], &[row]);
    let ln = v(&cpu.layer_norm_forward(&rows, &unit, &zero, EPS).unwrap());
    for (i, (&g, &l)) in got.iter().zip(&ln).enumerate() {
        let ch = (i / (h * w)) % c;
        assert_eq!(g, l * weight[ch] + bias[ch], "groups {groups} element {i}");
    }
}

#[test]
fn groups_equal_to_channels_is_layer_norm_of_each_channel_plane_bit_for_bit() {
    assert_is_layer_norm_of_rows([2, 6, 5, 7], 6, 10);
}

#[test]
fn a_single_group_is_layer_norm_of_each_image_bit_for_bit() {
    assert_is_layer_norm_of_rows([3, 4, 3, 5], 1, 11);
}

#[test]
fn an_inner_grouping_is_layer_norm_of_each_group_bit_for_bit() {
    assert_is_layer_norm_of_rows([2, 8, 3, 6], 4, 12);
}

#[test]
fn a_constant_group_has_variance_zero_and_equals_the_bias() {
    let cpu = backend(2, Numerics::Exact);
    let shape = [2, 6, 4, 5];
    let [n, c, h, w] = shape;
    let (_, weight, bias) = random_case(12, shape);
    // Each (image, group) holds its own constant, so the means differ.
    let groups = 3;
    let per = c / groups;
    let mut x = vec![0.0f32; n * c * h * w];
    for (i, q) in x.iter_mut().enumerate() {
        let (img, ch) = (i / (c * h * w), (i / (h * w)) % c);
        *q = 0.1 + 7.0 * (img * groups + ch / per) as f32;
    }
    let got = v(&cpu
        .group_norm_forward(
            &t(&x, &shape),
            &t(&weight, &[c]),
            &t(&bias, &[c]),
            groups,
            EPS,
        )
        .unwrap());
    for (i, &g) in got.iter().enumerate() {
        let ch = (i / (h * w)) % c;
        assert_eq!(g, bias[ch], "element {i}");
    }
    // The same holds for one group and for instance norm.
    for groups in [1, c] {
        let flat = t(&vec![-3.25; n * c * h * w], &shape);
        let got = v(&cpu
            .group_norm_forward(&flat, &t(&weight, &[c]), &t(&bias, &[c]), groups, EPS)
            .unwrap());
        for (i, &g) in got.iter().enumerate() {
            assert_eq!(g, bias[(i / (h * w)) % c], "groups {groups} element {i}");
        }
    }
}

/// The exact reference for `x = base + step * k` with integer `k`: the
/// centred values `step * (k - mean_k)` come from integer sums, so only the
/// final divisions round.
fn exact_offset_reference(ks: &[i64], step: f64, eps: f32) -> Vec<f64> {
    let n = ks.len() as i128;
    let sum: i128 = ks.iter().map(|&k| i128::from(k)).sum();
    let sq: i128 = ks.iter().map(|&k| i128::from(k) * i128::from(k)).sum();
    // var(k) = (n * sum(k^2) - sum(k)^2) / n^2, exact in integers until the
    // final division.
    let var_k = (n * sq - sum * sum) as f64 / (n * n) as f64;
    let mean_k = sum as f64 / n as f64;
    let var = step * step * var_k;
    let rstd = 1.0 / (var + f64::from(eps)).sqrt();
    ks.iter()
        .map(|&k| (k as f64 - mean_k) * step * rstd)
        .collect()
}

#[test]
fn a_mean_far_from_zero_keeps_its_variance_which_a_one_pass_sum_loses() {
    // Values 2e8 + 16 k are exact in f32 (spacing 16 there), the spread is a
    // few units of 16, and x^2 is about 4e16, beyond where f64 holds the
    // low bits of a sum of squares.
    let (base, step) = (200_000_000.0f32, 16.0f32);
    let shape = [1, 4, 32, 32];
    let mut rng = SplitMix64(13);
    let ks: Vec<i64> = (0..4 * 32 * 32).map(|_| rng.below(9) as i64 - 4).collect();
    let x: Vec<f32> = ks.iter().map(|&k| base + step * k as f32).collect();
    for (&q, &k) in x.iter().zip(&ks) {
        assert_eq!(f64::from(q), 2.0e8 + 16.0 * k as f64, "the input is exact");
    }
    let want = exact_offset_reference(&ks, f64::from(step), EPS);

    // What a one-pass E[x^2] - E[x]^2 in f64 gives on these values: the case
    // only tests the two-pass sums if that is far off.
    let n = x.len() as f64;
    let m = x.iter().map(|&q| f64::from(q)).sum::<f64>() / n;
    let one_pass = x.iter().map(|&q| f64::from(q) * f64::from(q)).sum::<f64>() / n - m * m;
    let exact = f64::from(step).powi(2)
        * (ks.iter().map(|&k| (k * k) as f64).sum::<f64>() / n
            - (ks.iter().map(|&k| k as f64).sum::<f64>() / n).powi(2));
    assert!(
        (one_pass - exact).abs() / exact > 1e-3,
        "one-pass variance {one_pass} is not far from {exact}: the input does not discriminate"
    );

    let cpu = backend(2, Numerics::Exact);
    let ones = t(&[1.0; 4], &[4]);
    let zeros = t(&[0.0; 4], &[4]);
    // One group holds all four channels, so the ks above are one group.
    let got = v(&cpu
        .group_norm_forward(&t(&x, &shape), &ones, &zeros, 1, EPS)
        .unwrap());
    assert_close("offset input, one group", &got, &want, 2e-6);
    // Instance norm: each channel is its own group of 1024 of the same ks.
    let per = 32 * 32;
    let got = v(&cpu
        .group_norm_forward(&t(&x, &shape), &ones, &zeros, 4, EPS)
        .unwrap());
    for ch in 0..4 {
        let want = exact_offset_reference(&ks[ch * per..(ch + 1) * per], f64::from(step), EPS);
        assert_close(
            &format!("offset input, channel {ch}"),
            &got[ch * per..(ch + 1) * per],
            &want,
            2e-6,
        );
    }
}

#[test]
fn a_non_finite_operand_or_output_is_refused() {
    let cpu = backend(2, Numerics::Exact);
    let shape = [2, 4, 3, 3];
    let (x, w, b) = random_case(14, shape);
    let (wt, bt) = (t(&w, &[4]), t(&b, &[4]));
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for at in [0, x.len() / 2, x.len() - 1] {
            let mut poisoned = x.clone();
            poisoned[at] = bad;
            assert_nonfinite(cpu.group_norm_forward(&t(&poisoned, &shape), &wt, &bt, 2, EPS));
        }
        let mut poisoned = w.clone();
        poisoned[1] = bad;
        assert_nonfinite(cpu.group_norm_forward(&t(&x, &shape), &t(&poisoned, &[4]), &bt, 2, EPS));
        let mut poisoned = b.clone();
        poisoned[3] = bad;
        assert_nonfinite(cpu.group_norm_forward(&t(&x, &shape), &wt, &t(&poisoned, &[4]), 2, EPS));
    }
    // Finite inputs whose output overflows f32 are refused, not stored.
    let huge = t(&[3.0e38; 4], &[4]);
    assert_nonfinite(cpu.group_norm_forward(&t(&x, &shape), &huge, &bt, 2, EPS));
    // f64 sums of f32 squares stay finite, so the largest inputs are still
    // normalized.
    let extreme = t(&[f32::MAX, -f32::MAX, f32::MAX, -f32::MAX], &[1, 1, 2, 2]);
    let ok = cpu
        .group_norm_forward(&extreme, &t(&[1.0], &[1]), &t(&[0.0], &[1]), 1, EPS)
        .unwrap();
    assert_close("f32::MAX inputs", &v(&ok), &[1.0, -1.0, 1.0, -1.0], 1e-6);
}

#[test]
fn every_refusal_uses_the_variant_the_neighbouring_ops_use() {
    let cpu = backend(1, Numerics::Exact);
    let x = t(&[0.5; 2 * 6 * 4 * 5], &[2, 6, 4, 5]);
    let (w, b) = (t(&[1.0; 6], &[6]), t(&[0.0; 6], &[6]));
    assert!(cpu.group_norm_forward(&x, &w, &b, 3, EPS).is_ok());

    // Rank other than 4.
    for shape in [&[6][..], &[2, 6], &[2, 6, 20], &[2, 6, 4, 5, 1]] {
        let n: usize = shape.iter().product();
        assert_shape(cpu.group_norm_forward(&t(&vec![0.5; n], shape), &w, &b, 3, EPS));
    }
    // groups == 0 is out of range; one that does not divide C is a shape.
    assert_range(cpu.group_norm_forward(&x, &w, &b, 0, EPS));
    for groups in [4, 5, 7, 12, usize::MAX] {
        assert_shape(cpu.group_norm_forward(&x, &w, &b, groups, EPS));
    }
    // weight and bias must be [C].
    for wrong in [t(&[1.0; 5], &[5]), t(&[1.0; 12], &[2, 6]), t(&[1.0], &[])] {
        assert_shape(cpu.group_norm_forward(&x, &wrong, &b, 3, EPS));
        assert_shape(cpu.group_norm_forward(&x, &w, &wrong, 3, EPS));
    }
    // eps: non-finite and non-positive.
    for eps in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_nonfinite(cpu.group_norm_forward(&x, &w, &b, 3, eps));
    }
    for eps in [0.0, -0.0, -1e-5, f32::MIN] {
        assert_range(cpu.group_norm_forward(&x, &w, &b, 3, eps));
    }
    // The smallest positive eps is accepted.
    assert!(cpu
        .group_norm_forward(&x, &w, &b, 3, f32::from_bits(1))
        .is_ok());
    // An empty extent anywhere, and a dtype that is not F32.
    for shape in [[0, 6, 4, 5], [2, 6, 0, 5], [2, 6, 4, 0]] {
        assert_shape(cpu.group_norm_forward(&t(&[], &shape), &w, &b, 3, EPS));
    }
    assert_shape(cpu.group_norm_forward(&x, &t(&[], &[0]), &t(&[], &[0]), 3, EPS));
    let ids = Tensor::from_u32(&[1; 240], &[2, 6, 4, 5], &Budget::new(1 << 12)).unwrap();
    match cpu.group_norm_forward(&ids, &w, &b, 3, EPS) {
        Err(OjasError::Dtype { op, .. }) => assert_eq!(op, "group_norm_forward"),
        other => panic!("expected Dtype, got {other:?}"),
    }
    // A transposed view is not read in storage order.
    let base = t(&[0.5; 240], &[2, 6, 5, 4]);
    let swapped = base.view(&[2, 6, 4, 5], &[120, 20, 1, 4], 0).unwrap();
    assert_shape(cpu.group_norm_forward(&swapped, &w, &b, 3, EPS));
}

#[test]
fn shape_errors_win_over_non_finite_data() {
    // Refusals are decided from shapes before any value is read: a NaN
    // input with a bad grouping reports the grouping.
    let cpu = backend(1, Numerics::Exact);
    let mut data = vec![0.5f32; 2 * 6 * 4 * 5];
    data[7] = f32::NAN;
    let x = t(&data, &[2, 6, 4, 5]);
    let (w, b) = (t(&[1.0; 6], &[6]), t(&[0.0; 6], &[6]));
    assert_range(cpu.group_norm_forward(&x, &w, &b, 0, EPS));
    assert_shape(cpu.group_norm_forward(&x, &w, &b, 4, EPS));
    assert_range(cpu.group_norm_forward(&x, &w, &b, 3, 0.0));
    assert_nonfinite(cpu.group_norm_forward(&x, &w, &b, 3, EPS));
}

#[test]
fn an_over_budget_output_is_capacity_exceeded_and_releases_its_charge() {
    let cpu = CpuBackend::new(Budget::new(4096));
    let x = t(&[0.5; 2 * 8 * 16 * 16], &[2, 8, 16, 16]);
    let (w, b) = (t(&[1.0; 8], &[8]), t(&[0.0; 8], &[8]));
    let before = cpu.budget().live_bytes().unwrap();
    assert_capacity(cpu.group_norm_forward(&x, &w, &b, 2, EPS));
    assert_eq!(cpu.budget().live_bytes().unwrap(), before);
}

#[test]
fn the_bits_do_not_depend_on_the_thread_count_or_numerics() {
    // Several (image, group) rows and enough values to be cut into chunks.
    let shape = [3, 16, 24, 20];
    let (x, w, b) = random_case(15, shape);
    let (xt, wt, bt) = (t(&x, &shape), t(&w, &[16]), t(&b, &[16]));
    let base = backend(1, Numerics::Exact)
        .group_norm_forward(&xt, &wt, &bt, 4, EPS)
        .unwrap();
    assert_close(
        "base",
        &v(&base),
        &reference(&x, &w, &b, shape, 4, EPS),
        5e-6,
    );
    for threads in [2, 7, 18] {
        for numerics in [Numerics::Exact, Numerics::Fast] {
            let got = backend(threads, numerics)
                .group_norm_forward(&xt, &wt, &bt, 4, EPS)
                .unwrap();
            assert_eq!(bits(&v(&got)), bits(&v(&base)), "x{threads} {numerics:?}");
        }
    }
}

#[test]
fn an_image_is_normalized_alone_whatever_its_batch_neighbours() {
    let cpu = backend(4, Numerics::Exact);
    let shape = [3, 8, 6, 5];
    let (x, w, b) = random_case(16, shape);
    let (wt, bt) = (t(&w, &[8]), t(&b, &[8]));
    let all = v(&cpu
        .group_norm_forward(&t(&x, &shape), &wt, &bt, 2, EPS)
        .unwrap());
    let per_image = 8 * 6 * 5;
    for img in 0..3 {
        let one = &x[img * per_image..(img + 1) * per_image];
        let alone = v(&cpu
            .group_norm_forward(&t(one, &[1, 8, 6, 5]), &wt, &bt, 2, EPS)
            .unwrap());
        assert_eq!(
            bits(&all[img * per_image..(img + 1) * per_image]),
            bits(&alone),
            "image {img}"
        );
    }
}

#[test]
fn a_bf16_region_keeps_group_norm_in_f32_like_layer_norm() {
    let wrapped = Autocast::new(backend(1, Numerics::Exact));
    let eye = t(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let _region = wrapped.autocast_region(AutocastMode::Bf16).unwrap();
    let a = wrapped
        .linear_forward(&t(&[1.0, 3.0], &[1, 2]), &eye)
        .unwrap();
    assert!(a.bf16_exact());
    let w = t(&[1.0, 1.0], &[2]);
    let z = t(&[0.0, 0.0], &[2]);
    let a4 = a.reshape(&[1, 2, 1, 1]).unwrap();
    let gn = wrapped.group_norm_forward(&a4, &w, &z, 1, EPS).unwrap();
    assert!(!gn.bf16_exact(), "group_norm keeps its output f32");
    let out = v(&gn);
    assert!(
        out.iter()
            .any(|q| ojas_core::round_f32_to_bf16(*q).to_bits() != q.to_bits()),
        "{out:?} is bf16-exact, so it was rounded"
    );
    // Refusals pass straight through the wrapper.
    assert_range(wrapped.group_norm_forward(&a4, &w, &z, 0, EPS));
}

/// SpikeTrack's pixel decoder sizes: 256 channels, 8 groups.
fn spiketrack_size(side: usize, seed: u64) {
    let shape = [1, 256, side, side];
    let (x, w, b) = random_case(seed, shape);
    let cpu = backend(4, Numerics::Exact);
    let got = cpu
        .group_norm_forward(&t(&x, &shape), &t(&w, &[256]), &t(&b, &[256]), 8, EPS)
        .unwrap();
    assert_eq!(got.shape(), shape.as_slice());
    assert_close(
        &format!("[1, 256, {side}, {side}] groups 8"),
        &v(&got),
        &reference(&x, &w, &b, shape, 8, EPS),
        5e-6,
    );
}

#[test]
fn spiketrack_size_144_matches_the_f64_reference() {
    spiketrack_size(144, 20);
}

#[test]
fn spiketrack_size_288_matches_the_f64_reference() {
    spiketrack_size(288, 21);
}

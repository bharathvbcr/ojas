//! Spatial op pins the main tests leave out: the 14x14 stride-14 patch embed
//! (never run with 14 before), padding at least the kernel, a 1x1 image,
//! bilinear to a 1x1 output, and an output-padding fringe of a transposed
//! convolution.

mod common;

use ojas_core::{Backend, Budget, Conv2dParams, ConvTranspose2dParams, Numerics, Tensor};
use ojas_cpu::CpuBackend;

fn cpu() -> CpuBackend {
    CpuBackend::with_threads(Budget::new(1 << 28), 1)
        .unwrap()
        .with_numerics(Numerics::Exact)
}

fn t(b: &CpuBackend, v: &[f64], shape: &[usize]) -> Tensor {
    let f: Vec<f32> = v.iter().map(|&x| x as f32).collect();
    Tensor::from_f32(&f, shape, b.budget()).unwrap()
}

fn near(name: &str, got: &[f32], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!((f64::from(g) - w).abs() <= tol, "{name}[{i}]: {g} vs {w}");
    }
}

/// Deterministic f32-representable values.
fn vals(seed: u64, n: usize) -> Vec<f64> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            f64::from((((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0) as f32)
        })
        .collect()
}

/// A patch-embed convolution's sizes: `c` input channels over an `h x w`
/// image to `o` channels, kernel and stride `k`.
#[derive(Clone, Copy)]
struct Patch {
    c: usize,
    h: usize,
    w: usize,
    o: usize,
    k: usize,
}

/// Patch-embed convolution (kernel = stride = k, no padding, no groups).
fn patch_ref(x: &[f64], weight: &[f64], bias: &[f64], p: Patch) -> Vec<f64> {
    let Patch { c, h, w, o, k } = p;
    let (ho, wo) = ((h - k) / k + 1, (w - k) / k + 1);
    let mut y = vec![0.0; o * ho * wo];
    for oc in 0..o {
        for oh in 0..ho {
            for ow in 0..wo {
                let mut acc = bias[oc];
                for ic in 0..c {
                    for i in 0..k {
                        for j in 0..k {
                            acc += x[(ic * h + oh * k + i) * w + ow * k + j]
                                * weight[((oc * c + ic) * k + i) * k + j];
                        }
                    }
                }
                y[(oc * ho + oh) * wo + ow] = acc;
            }
        }
    }
    y
}

#[test]
fn the_14x14_stride_14_patch_embed_matches_the_reference_and_floors_odd_sizes() {
    let be = cpu();
    let (c, o, k) = (3usize, 6usize, 14usize);
    let p = Conv2dParams {
        stride: [k, k],
        ..Conv2dParams::new()
    };
    let w = vals(1, o * c * k * k);
    let b = vals(2, o);
    // 28x42 divides by 14; 31x45 does not: torch drops the trailing 3 rows
    // and 3 columns, giving 2x3, as `(H - k) / stride + 1` does.
    for (h, wd) in [(28usize, 42usize), (31, 45), (14, 14)] {
        let x = vals(3, c * h * wd);
        let y = be
            .conv2d_forward(
                &t(&be, &x, &[1, c, h, wd]),
                &t(&be, &w, &[o, c, k, k]),
                Some(&t(&be, &b, &[o])),
                p,
            )
            .unwrap();
        let (ho, wo) = ((h - k) / k + 1, (wd - k) / k + 1);
        assert_eq!(y.shape(), &[1, o, ho, wo], "{h}x{wd}");
        near(
            &format!("{h}x{wd}"),
            &y.to_f32_vec().unwrap(),
            &patch_ref(&x, &w, &b, Patch { c, h, w: wd, o, k }),
            1e-4,
        );
    }
}

#[test]
fn padding_at_least_the_kernel_on_a_1x1_image_is_all_padding_but_one_tap() {
    let be = cpu();
    // 1x1 image holding 2.0, 3x3 kernel, padding 3: a 5x5 output. Output
    // (oh, ow) reads the pixel through tap (3 - oh, 3 - ow) when that tap
    // exists (oh, ow in 1..=3), else only the bias.
    let w: Vec<f64> = (1..=9).map(f64::from).collect();
    let p = Conv2dParams {
        padding: [3, 3],
        ..Conv2dParams::new()
    };
    let y = be
        .conv2d_forward(
            &t(&be, &[2.0], &[1, 1, 1, 1]),
            &t(&be, &w, &[1, 1, 3, 3]),
            Some(&t(&be, &[0.5], &[1])),
            p,
        )
        .unwrap();
    assert_eq!(y.shape(), &[1, 1, 5, 5]);
    let mut want = vec![0.5; 25];
    for oh in 1..=3usize {
        for ow in 1..=3usize {
            want[oh * 5 + ow] += 2.0 * w[(3 - oh) * 3 + (3 - ow)];
        }
    }
    near("pad >= kernel", &y.to_f32_vec().unwrap(), &want, 1e-6);
}

#[test]
fn bilinear_to_a_1x1_output_is_the_torch_centre_or_the_origin_sample() {
    let be = cpu();
    let (h, w) = (4usize, 5usize);
    let x = vals(5, h * w);
    let img = t(&be, &x, &[1, 1, h, w]);
    // align_corners = true with out = 1: torch's scale is 0, so the origin.
    let y = be.interpolate_bilinear_forward(&img, [1, 1], true).unwrap();
    near("align", &y.to_f32_vec().unwrap(), &[x[0]], 1e-7);
    // align_corners = false: source (h/1)*0.5 - 0.5 = 1.5 on rows (taps 1,2
    // at weight .5) and (w/1)*0.5 - 0.5 = 2.0 on columns (tap 2 exactly).
    let y = be
        .interpolate_bilinear_forward(&img, [1, 1], false)
        .unwrap();
    near(
        "centre",
        &y.to_f32_vec().unwrap(),
        &[0.5 * x[w + 2] + 0.5 * x[2 * w + 2]],
        1e-6,
    );
}

#[test]
fn a_transposed_conv_output_padding_fringe_holds_only_the_bias() {
    let be = cpu();
    let x = [1.0, 2.0, 3.0, 4.0];
    let w = [0.5, -1.0, 2.0, 0.25];
    let p = ConvTranspose2dParams {
        stride: [2, 2],
        padding: [0, 0],
        output_padding: [1, 1],
    };
    let y = be
        .conv_transpose2d_forward(
            &t(&be, &x, &[1, 1, 2, 2]),
            &t(&be, &w, &[1, 1, 2, 2]),
            Some(&t(&be, &[0.25], &[1])),
            p,
        )
        .unwrap();
    assert_eq!(y.shape(), &[1, 1, 5, 5]);
    let mut want = vec![0.25; 25];
    for ih in 0..2 {
        for iw in 0..2 {
            for i in 0..2 {
                for j in 0..2 {
                    want[(ih * 2 + i) * 5 + iw * 2 + j] += x[ih * 2 + iw] * w[i * 2 + j];
                }
            }
        }
    }
    near("fringe", &y.to_f32_vec().unwrap(), &want, 1e-6);
}

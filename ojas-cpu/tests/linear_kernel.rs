//! The tiled linear kernel: remainder widths, a 1x1 product, empty axes,
//! NaN, and a shape that is large enough to take the parallel tiles.

use ojas_core::{Backend, Budget, DType, OjasError, Tensor};
use ojas_cpu::CpuBackend;

mod common;
use common::{assert_nonfinite, assert_shape, bits, f32t, SplitMix64};

fn wide() -> CpuBackend {
    CpuBackend::new(Budget::new(8 << 20))
}

fn empty(cpu: &CpuBackend, shape: &[usize]) -> Tensor {
    Tensor::zeros(shape, DType::F32, cpu.budget()).unwrap()
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

fn check_against_reference(rows: usize, kin: usize, nout: usize, seed: u64) {
    let cpu = wide();
    let mut rng = SplitMix64(seed);
    let x = rng.vec(rows * kin, 0.5);
    let w = rng.vec(nout * kin, 0.5);
    let gy = rng.vec(rows * nout, 0.5);
    let xt = f32t(&cpu, &x, &[rows, kin]);
    let wt = f32t(&cpu, &w, &[nout, kin]);
    let y = cpu.linear_forward(&xt, &wt).unwrap();
    assert_eq!(y.shape(), &[rows, nout]);
    let yv = y.to_f32_vec().unwrap();
    assert_eq!(bits(&yv), bits(&ref_forward(&x, &w, rows, kin, nout)));
    assert_eq!(
        bits(&yv),
        bits(&cpu.linear_forward(&xt, &wt).unwrap().to_f32_vec().unwrap())
    );

    let gt = f32t(&cpu, &gy, &[rows, nout]);
    let (gx, gw) = cpu.linear_backward(&xt, &wt, &gt).unwrap();
    let (rx, rw) = ref_backward(&x, &w, &gy, rows, kin, nout);
    assert_eq!(bits(&gx.to_f32_vec().unwrap()), bits(&rx));
    assert_eq!(bits(&gw.to_f32_vec().unwrap()), bits(&rw));
    let (gx2, gw2) = cpu.linear_backward(&xt, &wt, &gt).unwrap();
    assert_eq!(
        bits(&gx.to_f32_vec().unwrap()),
        bits(&gx2.to_f32_vec().unwrap())
    );
    assert_eq!(
        bits(&gw.to_f32_vec().unwrap()),
        bits(&gw2.to_f32_vec().unwrap())
    );
    for threads in [1usize, 2, 3, 7, 16] {
        let parallel = CpuBackend::with_threads(Budget::new(8 << 20), threads).unwrap();
        let y2 = parallel
            .linear_forward(
                &f32t(&parallel, &x, &[rows, kin]),
                &f32t(&parallel, &w, &[nout, kin]),
            )
            .unwrap();
        assert_eq!(
            bits(&yv),
            bits(&y2.to_f32_vec().unwrap()),
            "threads {threads}"
        );
        let (gx_p, gw_p) = parallel
            .linear_backward(
                &f32t(&parallel, &x, &[rows, kin]),
                &f32t(&parallel, &w, &[nout, kin]),
                &f32t(&parallel, &gy, &[rows, nout]),
            )
            .unwrap();
        assert_eq!(
            bits(&gx.to_f32_vec().unwrap()),
            bits(&gx_p.to_f32_vec().unwrap()),
            "gx threads {threads}"
        );
        assert_eq!(
            bits(&gw.to_f32_vec().unwrap()),
            bits(&gw_p.to_f32_vec().unwrap()),
            "gw threads {threads}"
        );
    }
}

#[test]
fn tiled_linear_matches_scalar_reference_on_odd_and_parallel_shapes() {
    // 9 and 7 are not multiples of the 8-column tile, so the tail runs.
    check_against_reference(3, 5, 9, 11);
    check_against_reference(3, 5, 7, 15);
    // 1x1 product, the tile tail.
    check_against_reference(1, 1, 1, 12);
    // Rank-3 input: rows are the product of the prefix, width still odd.
    let cpu = wide();
    let mut rng = SplitMix64(13);
    let x = rng.vec(2 * 3 * 5, 0.25);
    let w = rng.vec(7 * 5, 0.25);
    let y = cpu
        .linear_forward(&f32t(&cpu, &x, &[2, 3, 5]), &f32t(&cpu, &w, &[7, 5]))
        .unwrap();
    assert_eq!(y.shape(), &[2, 3, 7]);
    assert_eq!(
        bits(&y.to_f32_vec().unwrap()),
        bits(&ref_forward(&x, &w, 6, 5, 7))
    );

    // Above the parallel grain, and not divisible by the 8-column tile.
    // 73*11*13 = 10439 multiply-adds. The helper also checks several thread counts.
    check_against_reference(73, 11, 13, 14);
}

#[test]
fn linear_one_by_one_is_a_product() {
    let cpu = wide();
    let x = f32t(&cpu, &[2.5], &[1]);
    let w = f32t(&cpu, &[4.0], &[1, 1]);
    let y = cpu.linear_forward(&x, &w).unwrap();
    assert_eq!(y.to_f32_vec().unwrap(), vec![10.0]);
    let gy = f32t(&cpu, &[3.0], &[1]);
    let (gx, gw) = cpu.linear_backward(&x, &w, &gy).unwrap();
    assert_eq!(gx.to_f32_vec().unwrap(), vec![12.0]);
    assert_eq!(gw.to_f32_vec().unwrap(), vec![7.5]);
}

#[test]
fn linear_empty_axes_and_mismatched_shapes_error() {
    let cpu = wide();
    let x = f32t(&cpu, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let w = f32t(&cpu, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3]);

    assert_shape(cpu.linear_forward(&empty(&cpu, &[0, 3]), &w));
    assert_shape(cpu.linear_forward(&empty(&cpu, &[2, 0]), &empty(&cpu, &[2, 0])));
    assert_shape(cpu.linear_forward(&x, &empty(&cpu, &[0, 3])));
    assert_shape(cpu.linear_backward(&empty(&cpu, &[0, 3]), &w, &empty(&cpu, &[0, 2])));

    assert_shape(cpu.linear_forward(&x, &f32t(&cpu, &[1.0, 2.0, 3.0, 4.0], &[2, 2])));
    let gy_bad = f32t(&cpu, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    assert_shape(cpu.linear_backward(&x, &w, &gy_bad));
    assert_shape(cpu.linear_backward(
        &x,
        &f32t(&cpu, &[1.0, 2.0], &[2]),
        &f32t(&cpu, &[1.0, 1.0], &[2, 1]),
    ));
}

#[test]
fn linear_nan_is_nonfinite() {
    let cpu = wide();
    let x = f32t(&cpu, &[1.0, 2.0], &[1, 2]);
    let w = f32t(&cpu, &[1.0, 0.0], &[1, 2]);
    let nan_x = f32t(&cpu, &[f32::NAN, 1.0], &[1, 2]);
    let nan_w = f32t(&cpu, &[f32::NAN, 0.0], &[1, 2]);
    assert_nonfinite(cpu.linear_forward(&nan_x, &w));
    assert_nonfinite(cpu.linear_forward(&x, &nan_w));
    let y = cpu.linear_forward(&x, &w).unwrap();
    let nan_g = f32t(&cpu, &[f32::NAN], y.shape());
    match cpu.linear_backward(&x, &w, &nan_g) {
        Err(OjasError::NonFinite { .. }) => {}
        other => panic!("expected NonFinite, got {other:?}"),
    }
}

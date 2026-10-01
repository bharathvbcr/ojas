//! The GEMM behind `linear_forward` (NT) and `linear_backward` (NN for
//! grad_x, TN for grad_w) against `CpuBackend` (`Numerics::Exact`) on shapes
//! that reach the 128x128 tile: both sides at least 128, K below, at and
//! past the 8-wide k step, ragged edges, and the nanolab projections.

mod common;

use common::*;
use ojas_core::Backend;

fn linear(rows: usize, kin: usize, nout: usize, seed: u64) {
    let c = cpu();
    let g = gpu();
    let x = host(seed, &[rows, kin]);
    let w = host(seed + 1, &[nout, kin]);
    let gy = host(seed + 2, &[rows, nout]);
    let tag = format!("linear {rows}x{kin}x{nout}");
    let (dx, dw) = (up(&x), up(&w));
    close(
        &format!("{tag} y"),
        &g.linear_forward(&dx, &dw).unwrap(),
        &c.linear_forward(&x, &w).unwrap(),
    );
    let (gx, gw) = g.linear_backward(&dx, &dw, &up(&gy)).unwrap();
    let (cx, cw) = c.linear_backward(&x, &w, &gy).unwrap();
    close(&format!("{tag} dx"), &gx, &cx);
    close(&format!("{tag} dw"), &gw, &cw);
}

#[test]
fn large_tiles_with_short_and_ragged_k_match_cpu() {
    for (i, &(r, k, n)) in [
        (128usize, 1usize, 128usize),
        (130, 7, 140),
        (128, 8, 128),
        (129, 9, 257),
        (256, 16, 128),
        (255, 17, 383),
        (300, 257, 200),
    ]
    .iter()
    .enumerate()
    {
        linear(r, k, n, 7000 + 10 * i as u64);
    }
}

#[test]
fn nanolab_projections_match_cpu() {
    linear(1024, 768, 768, 7100);
    linear(512, 768, 3072, 7110);
    linear(512, 3072, 768, 7120);
}

#[test]
fn large_tile_results_repeat_bit_for_bit() {
    let g = gpu();
    let x = up(&host(7200, &[300, 200]));
    let w = up(&host(7201, &[260, 200]));
    let run = || down(&g.linear_forward(&x, &w).unwrap());
    let first: Vec<u32> = run().iter().map(|v| v.to_bits()).collect();
    for _ in 0..3 {
        let again: Vec<u32> = run().iter().map(|v| v.to_bits()).collect();
        assert!(again == first);
    }
}

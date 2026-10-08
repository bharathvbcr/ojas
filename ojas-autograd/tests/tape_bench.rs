//! Wall time of two tape hot paths on `CpuBackend` (Fast), ignored by
//! default. One `OJAS_TAPE` line per row: the minimum and median of `N`
//! backward walks, each after its own forward (outside the timer), and the
//! walk's peak budget charge, which does not depend on load.
//!
//! - `fanin`: `h = h + h * w`, `LAYERS` times, at `[1024, 768]`. Each layer's
//!   `h` reaches its add twice (directly and through the mul), so every
//!   layer is an add backward plus a gradient fan-in.
//! - `ce_seed`: the fused linear cross-entropy over a `[50304, 768]` head at
//!   seed 1 and at seed 1/4, the trainer's seed at K = 4 micro-batches. The
//!   difference is what scaling the stored gradients by the seed costs.
//!
//! ```text
//! cargo test -p ojas-autograd --release --test tape_bench -- --ignored --nocapture --test-threads=1
//! ```

mod common;

use std::time::Instant;

use common::data;
use ojas_autograd::Tape;
use ojas_core::{Budget, CeChunk, Tensor};
use ojas_cpu::CpuBackend;

const T: usize = 1024;
const D: usize = 768;
const V: usize = 50304;
const LAYERS: usize = 8;

fn backend() -> CpuBackend {
    let cpu = CpuBackend::with_threads(Budget::new(12 << 30), 6).unwrap();
    cpu.start_workers().unwrap();
    cpu
}

/// `peak` is the most the walk charged above what was live when it began,
/// the same on every walk of a row, so it is exact where the times are noisy.
fn report(row: &str, mut ms: Vec<f64>, peak: u64) {
    ms.sort_by(f64::total_cmp);
    println!(
        "OJAS_TAPE row={row} n={} min_ms={:.4} med_ms={:.4} walk_peak_bytes={peak}",
        ms.len(),
        ms[0],
        ms[ms.len() / 2]
    );
}

/// Bytes the budget charged at its peak during `walk`, above what was live
/// before it.
fn walk_peak(budget: &Budget, walk: impl FnOnce()) -> u64 {
    let before = budget.live_bytes().unwrap();
    budget.reset_peak();
    walk();
    budget.peak_bytes() - before
}

#[test]
#[ignore]
fn tape_bench() {
    let cpu = backend();
    let budget = cpu.budget().clone();
    let x0 = Tensor::from_f32(&data(1, T * D), &[T, D], &budget).unwrap();
    let w = Tensor::from_f32(
        &data(2, T * D).iter().map(|v| v * 0.01).collect::<Vec<_>>(),
        &[T, D],
        &budget,
    )
    .unwrap();
    let mut tape = Tape::new(cpu);
    let mut fanin = Vec::new();
    let mut fanin_peak = 0;
    for i in 0..22 {
        tape.clear();
        let x = tape.leaf(x0.clone()).unwrap();
        let wv = tape.leaf(w.clone()).unwrap();
        let mut h = x;
        for _ in 0..LAYERS {
            let m = tape.mul(h, wv).unwrap();
            h = tape.add(h, m).unwrap();
        }
        let start = Instant::now();
        fanin_peak = walk_peak(&budget, || tape.backward_seeded(h, 0.25).unwrap());
        let dt = start.elapsed().as_secs_f64() * 1e3;
        assert!(tape.grad(x).is_some() && tape.grad(wv).is_some());
        if i >= 2 {
            fanin.push(dt);
        }
    }
    report(&format!("fanin_{LAYERS}x[{T},{D}]"), fanin, fanin_peak);

    let xs = Tensor::from_f32(&data(3, T * D), &[T, D], &budget).unwrap();
    let head = Tensor::from_f32(
        &data(4, V * D).iter().map(|v| v * 0.035).collect::<Vec<_>>(),
        &[V, D],
        &budget,
    )
    .unwrap();
    let ids: Vec<u32> = (0..T as u32).map(|i| (i * 7919) % V as u32).collect();
    let chunk = CeChunk {
        rows: 256,
        cols: 8192,
    };
    for (row, seed) in [("ce_seed_1", 1.0f32), ("ce_seed_0.25", 0.25)] {
        let mut ms = Vec::new();
        let mut peak = 0;
        for i in 0..9 {
            tape.clear();
            let x = tape.leaf(xs.clone()).unwrap();
            let wv = tape.leaf(head.clone()).unwrap();
            let targets = Tensor::from_u32(&ids, &[T], &budget).unwrap();
            let loss = tape
                .linear_cross_entropy(x, wv, targets, None, chunk)
                .unwrap();
            let start = Instant::now();
            peak = walk_peak(&budget, || tape.backward_seeded(loss, seed).unwrap());
            let dt = start.elapsed().as_secs_f64() * 1e3;
            assert!(tape.grad(wv).is_some());
            if i >= 2 {
                ms.push(dt);
            }
        }
        report(&format!("{row}_[{T},{D}]x[{V},{D}]"), ms, peak);
    }
}

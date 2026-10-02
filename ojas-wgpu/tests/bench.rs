//! Release benchmark: `cargo test -p ojas-wgpu --release --test bench --
//! --ignored --nocapture --test-threads=1`.
//!
//! "resident" runs the op on device tensors and synchronizes once at the end;
//! "with transfers" uploads the inputs and downloads every output on each
//! iteration. FLOPs: linear forward 2mnk, backward 4mnk; causal attention
//! forward 2·B·H·T²·D (QKᵀ and PV over the lower triangle), backward
//! 5·B·H·T²·D (S recomputed, dV, dP, dQ, dK).

mod common;

use std::time::{Duration, Instant};

use common::host;
use ojas_core::{Backend, Budget, Tensor};
use ojas_cpu::CpuBackend;
use ojas_wgpu::WgpuBackend;

const MIN_TIME: Duration = Duration::from_millis(500);
const MAX_ITERS: u32 = 50;

/// Mean seconds per call of `f`, after one warm-up call.
fn time(mut f: impl FnMut(), finish: impl Fn()) -> f64 {
    f();
    finish();
    let start = Instant::now();
    let mut n = 0u32;
    while n < 2 || (start.elapsed() < MIN_TIME && n < MAX_ITERS) {
        f();
        n += 1;
    }
    finish();
    start.elapsed().as_secs_f64() / f64::from(n)
}

struct Row {
    name: String,
    flops: f64,
    resident: f64,
    transfers: f64,
    cpu: f64,
}

fn print(rows: &[Row], adapter: &str) {
    println!("\nadapter: {adapter}; CPU: CpuBackend::with_threads(_, 6)\n");
    println!("| case | GPU resident ms | GFLOP/s | GPU with transfers ms | GFLOP/s | CPU ms | GFLOP/s | resident speedup |");
    println!("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
    for r in rows {
        let g = |s: f64| r.flops / s / 1e9;
        println!(
            "| {} | {:.3} | {:.0} | {:.3} | {:.0} | {:.3} | {:.0} | {:.1}x |",
            r.name,
            r.resident * 1e3,
            g(r.resident),
            r.transfers * 1e3,
            g(r.transfers),
            r.cpu * 1e3,
            g(r.cpu),
            r.cpu / r.resident
        );
    }
}

fn sync(g: &WgpuBackend) {
    g.sync().expect("sync");
}

fn down_all(g: &WgpuBackend, ts: &[&Tensor]) {
    for t in ts {
        let _ = g.download(t).expect("download");
    }
}

#[test]
#[ignore = "benchmark; run with --ignored --nocapture in release"]
fn bench_linear_and_attention() {
    let budget = Budget::new(16 << 30);
    let g = WgpuBackend::open(budget.clone()).expect("wgpu adapter");
    let c = CpuBackend::with_threads(budget.clone(), 6).expect("cpu backend");
    let mut rows = Vec::new();

    for &(m, n, k) in &[(512usize, 768usize, 768usize), (2048, 2048, 2048)] {
        let x = host(1, &[m, k]);
        let w = host(2, &[n, k]);
        let gy = host(3, &[m, n]);
        let (dx, dw, dgy) = (
            g.upload(&x).unwrap(),
            g.upload(&w).unwrap(),
            g.upload(&gy).unwrap(),
        );
        let fwd = 2.0 * (m * n * k) as f64;
        rows.push(Row {
            name: format!("linear fwd {m}x{n}x{k}"),
            flops: fwd,
            resident: time(|| drop(g.linear_forward(&dx, &dw).unwrap()), || sync(&g)),
            transfers: time(
                || {
                    let y = g
                        .linear_forward(&g.upload(&x).unwrap(), &g.upload(&w).unwrap())
                        .unwrap();
                    down_all(&g, &[&y]);
                },
                || sync(&g),
            ),
            cpu: time(|| drop(c.linear_forward(&x, &w).unwrap()), || {}),
        });
        rows.push(Row {
            name: format!("linear bwd {m}x{n}x{k}"),
            flops: 2.0 * fwd,
            resident: time(
                || drop(g.linear_backward(&dx, &dw, &dgy).unwrap()),
                || sync(&g),
            ),
            transfers: time(
                || {
                    let (a, b) = g
                        .linear_backward(
                            &g.upload(&x).unwrap(),
                            &g.upload(&w).unwrap(),
                            &g.upload(&gy).unwrap(),
                        )
                        .unwrap();
                    down_all(&g, &[&a, &b]);
                },
                || sync(&g),
            ),
            cpu: time(|| drop(c.linear_backward(&x, &w, &gy).unwrap()), || {}),
        });
    }

    for &t in &[128usize, 512, 2048] {
        let (b, h, d) = (4usize, 8usize, 64usize);
        let shape = [b, h, t, d];
        let (q, k, v, gy) = (
            host(10, &shape),
            host(11, &shape),
            host(12, &shape),
            host(13, &shape),
        );
        let (dq, dk, dv, dgy) = (
            g.upload(&q).unwrap(),
            g.upload(&k).unwrap(),
            g.upload(&v).unwrap(),
            g.upload(&gy).unwrap(),
        );
        let base = (b * h * t * t * d) as f64;
        rows.push(Row {
            name: format!("sdpa fwd B{b} H{h} T{t} D{d}"),
            flops: 2.0 * base,
            resident: time(
                || drop(g.causal_sdpa_forward(&dq, &dk, &dv).unwrap()),
                || sync(&g),
            ),
            transfers: time(
                || {
                    let y = g
                        .causal_sdpa_forward(
                            &g.upload(&q).unwrap(),
                            &g.upload(&k).unwrap(),
                            &g.upload(&v).unwrap(),
                        )
                        .unwrap();
                    down_all(&g, &[&y]);
                },
                || sync(&g),
            ),
            cpu: time(|| drop(c.causal_sdpa_forward(&q, &k, &v).unwrap()), || {}),
        });
        rows.push(Row {
            name: format!("sdpa bwd B{b} H{h} T{t} D{d}"),
            flops: 5.0 * base,
            resident: time(
                || drop(g.causal_sdpa_backward(&dq, &dk, &dv, &dgy).unwrap()),
                || sync(&g),
            ),
            transfers: time(
                || {
                    let (a, bb, cc) = g
                        .causal_sdpa_backward(
                            &g.upload(&q).unwrap(),
                            &g.upload(&k).unwrap(),
                            &g.upload(&v).unwrap(),
                            &g.upload(&gy).unwrap(),
                        )
                        .unwrap();
                    down_all(&g, &[&a, &bb, &cc]);
                },
                || sync(&g),
            ),
            cpu: time(
                || drop(c.causal_sdpa_backward(&q, &k, &v, &gy).unwrap()),
                || {},
            ),
        });
    }
    print(&rows, g.context().adapter_name());
}

//! Wall-time matrix for the linear and attention kernels. Ignored by default:
//! `cargo test -p ojas-cpu --release --test bench_cpu -- --ignored --nocapture --test-threads=1`.

use ojas_core::{Backend, Budget, Numerics};
use ojas_cpu::CpuBackend;

mod common;
use common::{f32t, SplitMix64};

fn median(mut ns: Vec<u128>) -> f64 {
    ns.sort_unstable();
    ns[ns.len() / 2] as f64 / 1e9
}

/// `OJAS_BENCH_THREADS=1,6` overrides the thread counts.
fn thread_counts() -> Vec<usize> {
    match std::env::var("OJAS_BENCH_THREADS") {
        Ok(list) => list
            .split(',')
            .filter_map(|t| t.trim().parse().ok())
            .collect(),
        Err(_) => vec![6, 18],
    }
}

fn backend(threads: usize, numerics: Numerics) -> CpuBackend {
    CpuBackend::with_threads(Budget::new(2 << 30), threads)
        .unwrap()
        .with_numerics(numerics)
}

const NUMERICS: [Numerics; 2] = [Numerics::Exact, Numerics::Fast];

#[test]
#[ignore]
fn linear_wall_time_matrix() {
    let shapes: &[(usize, usize, usize)] = &[
        (64, 64, 128),
        (256, 256, 256),
        (512, 768, 768),
        (2048, 2048, 2048),
    ];
    for (threads, numerics) in thread_counts()
        .into_iter()
        .flat_map(|t| NUMERICS.map(|n| (t, n)))
    {
        let cpu = backend(threads, numerics);
        for &(rows, kin, nout) in shapes {
            let mut rng = SplitMix64(0x5eed);
            let x = f32t(&cpu, &rng.vec(rows * kin, 0.1), &[rows, kin]);
            let w = f32t(&cpu, &rng.vec(nout * kin, 0.1), &[nout, kin]);
            let gy = f32t(&cpu, &rng.vec(rows * nout, 0.1), &[rows, nout]);
            let calls = if rows >= 2048 {
                4
            } else if rows >= 512 {
                8
            } else {
                30
            };
            let mut fwd = Vec::new();
            let mut bwd = Vec::new();
            for i in 0..=calls {
                let t0 = std::time::Instant::now();
                std::hint::black_box(cpu.linear_forward(&x, &w).unwrap());
                let f = t0.elapsed().as_nanos();
                let t1 = std::time::Instant::now();
                std::hint::black_box(cpu.linear_backward(&x, &w, &gy).unwrap());
                let b = t1.elapsed().as_nanos();
                if i > 0 {
                    fwd.push(f);
                    bwd.push(b);
                }
            }
            println!(
                "OJAS_LINEAR threads={threads} numerics={numerics:?} rows={rows} kin={kin} nout={nout} fwd_ms={:.4} bwd_ms={:.4}",
                median(fwd) * 1e3,
                median(bwd) * 1e3
            );
        }
    }
}

#[test]
#[ignore]
fn attention_wall_time_matrix() {
    let (b, h, d) = (4usize, 8usize, 64usize);
    for (threads, numerics) in thread_counts()
        .into_iter()
        .flat_map(|t| NUMERICS.map(|n| (t, n)))
    {
        let cpu = backend(threads, numerics);
        for t in [128usize, 512, 2048] {
            let n = b * h * t * d;
            let mut rng = SplitMix64(7);
            let shape = [b, h, t, d];
            let q = f32t(&cpu, &rng.vec(n, 0.5), &shape);
            let k = f32t(&cpu, &rng.vec(n, 0.5), &shape);
            let v = f32t(&cpu, &rng.vec(n, 0.5), &shape);
            let gy = f32t(&cpu, &rng.vec(n, 0.5), &shape);
            let calls = if t >= 2048 { 2 } else { 6 };
            let mut fwd = Vec::new();
            let mut bwd = Vec::new();
            for i in 0..=calls {
                let t0 = std::time::Instant::now();
                std::hint::black_box(cpu.causal_sdpa_forward(&q, &k, &v).unwrap());
                let f = t0.elapsed().as_nanos();
                let t1 = std::time::Instant::now();
                std::hint::black_box(cpu.causal_sdpa_backward(&q, &k, &v, &gy).unwrap());
                let bw = t1.elapsed().as_nanos();
                if i > 0 {
                    fwd.push(f);
                    bwd.push(bw);
                }
            }
            println!(
                "OJAS_ATTN threads={threads} numerics={numerics:?} B={b} H={h} T={t} D={d} fwd_ms={:.3} bwd_ms={:.3}",
                median(fwd) * 1e3,
                median(bwd) * 1e3
            );
        }
    }
}

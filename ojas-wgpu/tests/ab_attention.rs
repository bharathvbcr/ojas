//! Interleaved before/after timing of causal SDPA on one device: the
//! one-row-per-lane kernel (legacy) against the tiled one, alternating order
//! each round, min of N. Run:
//! `cargo test -p ojas-wgpu --release --test ab_attention -- --ignored --nocapture`.

mod common;

use std::time::Instant;

use common::host;
use ojas_core::{Backend, Budget};
use ojas_wgpu::WgpuBackend;

const ROUNDS: usize = 7;

#[test]
#[ignore = "benchmark; run with --ignored --nocapture in release"]
fn sdpa_legacy_vs_tiled() {
    let g = WgpuBackend::open(Budget::new(16 << 30)).unwrap();
    println!("adapter: {}", g.context().adapter_name());
    println!("| shape | pass | legacy ms (min of {ROUNDS}) | tiled ms (min of {ROUNDS}) | speedup |");
    println!("| --- | --- | ---: | ---: | ---: |");
    for shape in [[4usize, 12, 1024, 64], [4, 8, 2048, 64], [2, 8, 1024, 128]] {
        let (q, k, v, gy) = (
            g.upload(&host(1, &shape)).unwrap(),
            g.upload(&host(2, &shape)).unwrap(),
            g.upload(&host(3, &shape)).unwrap(),
            g.upload(&host(4, &shape)).unwrap(),
        );
        let fwd = || {
            let t = Instant::now();
            let y = g.causal_sdpa_forward(&q, &k, &v).unwrap();
            g.sync().unwrap();
            drop(y);
            t.elapsed().as_secs_f64()
        };
        let bwd = || {
            let t = Instant::now();
            let r = g.causal_sdpa_backward(&q, &k, &v, &gy).unwrap();
            g.sync().unwrap();
            drop(r);
            t.elapsed().as_secs_f64()
        };
        // [variant][pass] minimum seconds.
        let mut best = [[f64::INFINITY; 2]; 2];
        for legacy in [true, false] {
            g.set_legacy_attention(legacy);
            fwd();
            bwd();
        }
        for round in 0..ROUNDS {
            let order = if round % 2 == 0 { [true, false] } else { [false, true] };
            for legacy in order {
                g.set_legacy_attention(legacy);
                let i = usize::from(!legacy);
                best[i][0] = best[i][0].min(fwd());
                best[i][1] = best[i][1].min(bwd());
            }
        }
        g.set_legacy_attention(false);
        for (p, pass) in ["fwd", "bwd"].iter().enumerate() {
            println!(
                "| {shape:?} | {pass} | {:.2} | {:.2} | {:.1}x |",
                best[0][p] * 1e3,
                best[1][p] * 1e3,
                best[0][p] / best[1][p]
            );
        }
    }
}

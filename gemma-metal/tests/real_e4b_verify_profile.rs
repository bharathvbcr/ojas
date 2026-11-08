//! Where `step_verify(M)` spends its time on real E4B: per-dispatch GPU stamps for one
//! verify block at M = 1, 2, 4, summed by kernel. Stamps distort small dispatches (see
//! `tessl::dispatch_stamps`), so read the shares, not the totals; the unstamped wall time
//! per block is printed beside them.
//!
//! `GEMMA_E4B_DIR=... cargo test --release --test real_e4b_verify_profile -- --ignored --nocapture`

use gemma_metal::gpu_model::{GpuDecodeSession, GpuSynthModel};
use gemma_metal::quant::QuantScheme;
use gemma_metal::weights::{load_from_hf_dir, LoadOptions};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[test]
#[ignore = "needs the real E4B snapshot"]
fn verify_block_kernel_breakdown() {
    let e4b = PathBuf::from(std::env::var("GEMMA_E4B_DIR").expect("set GEMMA_E4B_DIR"));
    tessl::ab_flags::set_dispatch_stamps(true);
    let banks = load_from_hf_dir(
        &e4b,
        LoadOptions {
            scheme: QuantScheme::q4_mlx_default(),
            max_seq: 256,
            ..LoadOptions::default()
        },
    )
    .expect("load E4B");
    let model = GpuSynthModel::from_host_banks(banks).expect("upload E4B");
    let mut sess = GpuDecodeSession::new(model).expect("session");
    for &t in &[2u32, 818, 5279, 529, 7001, 563] {
        sess.step(t).expect("prefill");
    }
    let seeds = [3u32, 4, 5, 6, 7, 8, 9, 10];
    for m in [1usize, 2, 4] {
        let toks = &seeds[..m];
        for _ in 0..2 {
            sess.profile_verify_block(toks, false).expect("warm");
        }
        let mut walls = Vec::new();
        for _ in 0..5 {
            let w = sess.profile_verify_block(toks, false).expect("wall");
            walls.push(w.host_encode_ms + w.sync_ms);
        }
        walls.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let reps = 3;
        let mut per_kernel: BTreeMap<String, (f64, usize)> = BTreeMap::new();
        let mut span = 0.0;
        let mut n_disp = 0;
        for _ in 0..reps {
            let p = sess.profile_verify_block(toks, true).expect("stamped");
            let b = p.stamps.expect("stamps");
            if b.overflowed || b.dropped_batches != 0 || b.stamps.len() as u64 != p.dispatches {
                println!(
                    "  INCOMPLETE stamps at M={m}: {} of {} dispatches, overflowed={}, dropped={}",
                    b.stamps.len(),
                    p.dispatches,
                    b.overflowed,
                    b.dropped_batches
                );
            }
            n_disp = b.stamps.len();
            let mut last = b.t0_ns;
            for s in &b.stamps {
                let e = per_kernel.entry(s.name.clone()).or_default();
                e.0 += s.dt_ns as f64 / 1e6 / reps as f64;
                e.1 += 1;
                last = s.t_end_ns;
            }
            span += (last - b.t0_ns) as f64 / 1e6 / reps as f64;
        }
        let mut rows: Vec<_> = per_kernel.into_iter().collect();
        rows.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap());
        println!(
            "== M={m}: unstamped wall median {:.2} ms; stamped span {span:.2} ms, {n_disp} dispatches",
            walls[2]
        );
        for (name, (ms, n)) in rows.iter().take(14) {
            println!(
                "  {name:<44} {ms:7.3} ms  {:5.1}%  x{}",
                100.0 * ms / span,
                n / reps
            );
        }
    }
}

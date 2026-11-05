//! `step_verify` resets the per-step scalar arena, as the decode step (`step_inner`) does.
//!
//! Before 2026-11-04 it never called `reset_step`, so on the batched GEMM verify path the
//! arena cursor grew with every verify until `push_u32` refused ("icb_scalars u32 arena
//! exhausted"); on real E4B that was the second call. The mini graph never takes that path
//! (its GEMM needs more than 256 columns), so this runs at E4B shapes on random MLX Q4
//! weights: the values are meaningless, only that repeated verifies keep working.
//! One test body in its own process: the scalar pool flags are global.

use gemma_metal::config::Gemma4TextConfig;
use gemma_metal::gpu_model::{GpuDecodeSession, GpuSynthModel};
use gemma_metal::quant::QuantScheme;
use gemma_metal::weights::{synthetic_mlx_q4_banks, LoadOptions};

#[test]
fn repeated_verifies_do_not_exhaust_the_scalar_arena() {
    let banks = synthetic_mlx_q4_banks(
        Gemma4TextConfig::e4b_preset(),
        LoadOptions {
            scheme: QuantScheme::q4_mlx_default(),
            max_seq: 128,
            ..LoadOptions::default()
        },
        0x5eed,
    )
    .expect("synthetic E4B banks");
    let model = match GpuSynthModel::from_host_banks(banks) {
        Ok(m) => m,
        Err(e) => panic!("GPU unavailable or E4B-shaped upload failed (not a pass): {e}"),
    };
    let mut sess = GpuDecodeSession::new(model).expect("session");
    let _ = sess.generate(&[2u32, 3], 1).expect("prefill");
    let pos = sess.pos();
    for i in 0..8 {
        sess.step_verify(&[3, 4, 5, 6])
            .unwrap_or_else(|e| panic!("verify {i} failed: {e}"));
        sess.trim_kv(4).expect("trim");
        assert_eq!(sess.pos(), pos);
    }
}

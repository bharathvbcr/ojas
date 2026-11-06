//! The scalar pool does not commit per dispatch, and changes no token.
//!
//! Outside DecodeIcb and encode-once, `IcbScalarPool` passes per-dispatch
//! scalars through the binder's constant arena. Before 2026-11-04 every push
//! wrote the pool's GPU buffer, and tessl commits and waits before any host
//! mapping, so a decode token committed once per dispatch (76 on this graph,
//! 599 at E4B shapes). `set_stable_scalars(true)` restores that path as the
//! reference arm. One test body in its own process: the flags are global.

use gemma_metal::forward::SyntheticE4bGraph;
use gemma_metal::gpu_model::{GpuDecodeSession, GpuSynthModel};
use gemma_metal::kernels::set_stable_scalars;
use gemma_metal::quant::QuantScheme;

#[test]
fn arena_scalars_commit_once_per_token_and_match_the_stable_pool() {
    let model = match GpuSynthModel::from_synthetic(
        SyntheticE4bGraph::mini_parity().unwrap(),
        QuantScheme::q4_default(),
    ) {
        Ok(m) => m,
        Err(e) => panic!("GPU unavailable or mini upload failed (not a pass): {e}"),
    };
    let mut sess = GpuDecodeSession::new(model).expect("session");
    let prompt = [1u32, 2, 3, 4];

    set_stable_scalars(true);
    let stable = sess.generate(&prompt, 8).expect("generate, stable pool");
    sess.profile_prime(&prompt).expect("prime");
    let stable_tok = sess
        .profile_decode_token(false)
        .expect("token, stable pool");

    set_stable_scalars(false);
    let arena = sess.generate(&prompt, 8).expect("generate, arena");
    sess.profile_prime(&prompt).expect("prime");
    let arena_tok = sess.profile_decode_token(false).expect("token, arena");

    assert_eq!(arena, stable, "arena scalars changed the generated tokens");
    // The arena path adds the GPU store of `pos_buf` (replacing the stable
    // path's host write) and runs each of the 3 layers' attention as tessl's
    // split-KV decode, a partial and a reduce dispatch where the tiled kernel
    // was one, unless every live key fits one chunk and the single-pass kernel
    // takes it (one dispatch, as here: the prompt is 4 tokens). The stable path
    // keeps the tiled kernel: the decode binds its live values as constants,
    // which a replay cannot rewrite.
    let layers = 3;
    let split_kv_extra = if gemma_metal::kernels::decode_attn_single_pass_enabled() {
        0
    } else {
        layers
    };
    assert_eq!(
        arena_tok.dispatches,
        stable_tok.dispatches + 1 + split_kv_extra,
        "the scalar path changed what was dispatched"
    );
    // The reference arm still shows the cost being removed, so this test
    // cannot pass because nothing committed at all.
    assert!(
        stable_tok.commits * 2 > stable_tok.dispatches,
        "stable pool: {} commits for {} dispatches",
        stable_tok.commits,
        stable_tok.dispatches
    );
    assert!(
        arena_tok.commits <= 2,
        "arena: {} commits for {} dispatches (expected the token's own, at most 2)",
        arena_tok.commits,
        arena_tok.dispatches
    );
}

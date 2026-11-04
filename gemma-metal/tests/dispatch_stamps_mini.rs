//! Per-dispatch GPU stamps on the synthetic mini graph (no real weights).
//!
//! One test body in its own process: `tessl::ab_flags::set_dispatch_stamps` is
//! process-global and the stamp heap exists only for a runtime built with the
//! flag on, so the order here (flag on, build, then toggle per token) is the
//! order a profiling run uses.

use gemma_metal::forward::SyntheticE4bGraph;
use gemma_metal::gpu_model::{GpuDecodeSession, GpuSynthModel};
use gemma_metal::quant::QuantScheme;

#[test]
fn mini_token_stamps_match_dispatches_are_ordered_and_change_nothing() {
    tessl::ab_flags::set_dispatch_stamps(true);
    let model = match GpuSynthModel::from_synthetic(
        SyntheticE4bGraph::mini_parity().unwrap(),
        QuantScheme::q4_default(),
    ) {
        Ok(m) => m,
        Err(e) => panic!("GPU unavailable or mini upload failed (not a pass): {e}"),
    };
    let n_layers = 3i32; // mini_parity: SWA + global + consumer
    let mut sess = GpuDecodeSession::new(model).expect("session");
    let prompt = [1u32, 2, 3, 4];

    // Behaviour: the generated tokens are identical with the flag off and on.
    tessl::ab_flags::set_dispatch_stamps(false);
    let off = sess.generate(&prompt, 6).expect("generate off");
    tessl::ab_flags::set_dispatch_stamps(true);
    let on = sess.generate(&prompt, 6).expect("generate on");
    tessl::ab_flags::set_dispatch_stamps(false);
    assert_eq!(off, on, "stamps changed the generated tokens");

    // Stamps off: nothing recorded, but the timings and counts still come back.
    sess.profile_prime(&prompt).expect("prime");
    let cold = sess.profile_decode_token(false).expect("token off");
    assert!(cold.stamps.is_none());
    assert!(cold.dispatches > 0 && cold.barriers > 0);

    // Stamps on: one stamp per dispatch, in order, with sane ns.
    sess.profile_prime(&prompt).expect("prime");
    let tok = sess.profile_decode_token(true).expect("token on");
    let batch = tok.stamps.expect("stamps");
    // A token is one or two command buffers (tests/scalar_pool_commits.rs);
    // the batch may fold several.
    assert_eq!(tok.commits, cold.commits, "stamps changed the commit count");
    assert!(batch.command_buffers >= 1 && u64::from(batch.command_buffers) <= tok.commits);
    assert!(!batch.overflowed);
    assert_eq!(batch.dropped_batches, 0);
    assert_eq!(
        batch.stamps.len() as u64,
        tok.dispatches,
        "stamp count must equal the dispatch count tessl encoded"
    );
    assert_eq!(
        tok.dispatches, cold.dispatches,
        "stamps changed the dispatch count"
    );
    assert!(batch.freq_hz > 0);

    let mut prev = batch.t0_ns;
    let mut last_layer = -1;
    for (i, s) in batch.stamps.iter().enumerate() {
        assert_ne!(s.name, "?", "stamp {i} has no kernel name");
        assert!(s.t_end_ns >= prev, "stamp {i} ({}) went backwards", s.name);
        prev = s.t_end_ns;
        assert!(
            (-1..=n_layers).contains(&s.layer),
            "stamp {i} layer {}",
            s.layer
        );
        assert!(
            s.layer >= last_layer,
            "stamp {i} layer {} after {last_layer}",
            s.layer
        );
        last_layer = s.layer;
    }
    assert_eq!(
        last_layer, n_layers,
        "tail (final norm / lm_head / argmax) must be tagged"
    );
    // Every layer is represented, and every dispatch but the first of a command
    // buffer follows a barrier (hazard mode emits it lazily, always-on mode right after the
    // previous dispatch).
    for l in 0..n_layers {
        assert!(
            batch.stamps.iter().any(|s| s.layer == l),
            "no dispatch tagged layer {l}"
        );
    }
    // The only dispatches without one are the first of each command buffer.
    let unbarriered = batch.stamps.iter().filter(|s| !s.barrier_before).count();
    assert!(
        unbarriered <= batch.command_buffers as usize,
        "{unbarriered} dispatches had no barrier"
    );

    // The GPU span cannot exceed the host's wall time around the token.
    let span_ms = (prev - batch.t0_ns) as f64 / 1e6;
    let wall_ms = tok.host_encode_ms + tok.sync_ms;
    assert!(span_ms > 0.0, "empty span");
    assert!(
        span_ms <= wall_ms + 1.0,
        "GPU span {span_ms:.3} ms exceeds host wall {wall_ms:.3} ms"
    );

    // Flag off again: the runtime stays usable and records nothing.
    let after = sess.profile_decode_token(false).expect("token off again");
    assert!(after.stamps.is_none());
    println!(
        "mini token: {} dispatches, {} barriers, span {span_ms:.3} ms, sync {:.3} ms, encode {:.3} ms",
        tok.dispatches, tok.barriers, tok.sync_ms, tok.host_encode_ms
    );
}

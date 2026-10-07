//! `ActivationCheckpoint::Blocks`: each block is one checkpointed tape
//! segment. The loss and every parameter gradient must be the same bits as
//! the uncheckpointed forward, and past T = 2048 a budget that holds the
//! checkpointed step must refuse the uncheckpointed one.

use ojas_autograd::Tape;
use ojas_core::{Budget, CeChunk, Numerics, OjasError, Tensor};
use ojas_cpu::CpuBackend;
use ojas_model::{
    bind, forward_loss, init_params, swiglu_hidden, ActivationCheckpoint, ModelSpec, Rope,
};

const CHUNK: CeChunk = CeChunk {
    rows: 256,
    cols: 64,
};

/// Token ids and next-token targets for `batch` rows of `seq`.
fn batch(spec: &ModelSpec, batch: usize, seq: usize, budget: &Budget) -> (Tensor, Tensor) {
    let n = batch * seq;
    let ids: Vec<u32> = (0..n as u32)
        .map(|i| i.wrapping_mul(2_654_435_761) % spec.vocab as u32)
        .collect();
    let targets: Vec<u32> = ids
        .iter()
        .map(|&t| (t * 7 + 3) % spec.vocab as u32)
        .collect();
    (
        Tensor::from_u32(&ids, &[batch, seq], budget).unwrap(),
        Tensor::from_u32(&targets, &[n], budget).unwrap(),
    )
}

struct Step {
    loss: Vec<u32>,
    /// `None` for a parameter the forward never reads (layer 0's `vr_lambda`).
    grads: Vec<Option<Vec<u32>>>,
    /// Peak bytes the backend's budget held over the forward and backward.
    peak: u64,
}

fn bits(t: &Tensor) -> Vec<u32> {
    t.to_f32_vec()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// One forward and backward on a CPU backend whose budget is `cap`. The
/// parameters, ids, targets and RoPE table live on a separate host budget,
/// so `peak` counts what the step itself allocates.
fn step(
    spec: &ModelSpec,
    rows: usize,
    seq: usize,
    numerics: Numerics,
    activations: ActivationCheckpoint,
    cap: u64,
) -> Result<Step, OjasError> {
    let host = Budget::new(u64::MAX);
    let params = init_params(spec, 7, &host)?;
    let rope = Rope::new(spec, seq, &host)?;
    let (ids, targets) = batch(spec, rows, seq, &host);
    let budget = Budget::new(cap);
    let mut tape = Tape::new(CpuBackend::new(budget.clone()).with_numerics(numerics));
    let bound = bind(&mut tape, spec, &params)?;
    budget.reset_peak();
    let loss = forward_loss(
        &mut tape,
        spec,
        &bound,
        &ids,
        &targets,
        &rope,
        None,
        CHUNK,
        activations,
    )?;
    tape.backward_seeded(loss, 0.5)?;
    let loss = bits(tape.value(loss)?);
    let grads = bound
        .into_flat()
        .into_iter()
        .map(|v| tape.grad(v).map(bits))
        .collect();
    Ok(Step {
        loss,
        grads,
        peak: budget.peak_bytes(),
    })
}

fn assert_same(a: &Step, b: &Step, what: &str) {
    assert_eq!(a.loss, b.loss, "{what}: loss");
    assert_eq!(a.grads.len(), b.grads.len(), "{what}: parameter count");
    for (i, (x, y)) in a.grads.iter().zip(&b.grads).enumerate() {
        assert_eq!(x, y, "{what}: parameter {i}");
    }
    // Only layer 0's `vr_lambda` (flat index 1 + 9: tok_emb, then block 0's
    // tenth field) is never read, in either mode.
    for s in [a, b] {
        let missing: Vec<usize> = (0..s.grads.len())
            .filter(|&i| s.grads[i].is_none())
            .collect();
        assert_eq!(missing, [10], "{what}: parameters without a gradient");
    }
}

#[test]
fn blocks_give_the_uncheckpointed_loss_and_gradients_bit_for_bit() {
    let spec = ModelSpec {
        n_layer: 3,
        ..ModelSpec::tiny()
    };
    for numerics in [Numerics::Exact, Numerics::Fast] {
        let off = step(&spec, 2, 32, numerics, ActivationCheckpoint::Off, u64::MAX).unwrap();
        let blocks = step(
            &spec,
            2,
            32,
            numerics,
            ActivationCheckpoint::Blocks,
            u64::MAX,
        )
        .unwrap();
        assert_same(&off, &blocks, &format!("{numerics:?}"));
        assert!(
            blocks.peak < off.peak,
            "{numerics:?}: peak {} B checkpointed, {} B not",
            blocks.peak,
            off.peak
        );
    }
}

/// T = 2304, past 2048. The checkpointed step fits a budget the
/// uncheckpointed step does not, and still gives the uncheckpointed
/// gradients.
#[test]
fn past_2048_tokens_a_budget_that_holds_the_checkpointed_step_refuses_the_other() {
    const SEQ: usize = 2304;
    let spec = ModelSpec {
        vocab: 64,
        n_embd: 16,
        n_layer: 4,
        n_head: 1,
        n_kv_head: 1,
        head_dim: 16,
        hidden: swiglu_hidden(16),
        max_seq: SEQ,
        rope_base: 10000.0,
        rms_eps: 1e-6,
        tie_embeddings: true,
    };
    let numerics = Numerics::Exact;
    let off = step(&spec, 1, SEQ, numerics, ActivationCheckpoint::Off, u64::MAX).unwrap();
    let blocks = step(
        &spec,
        1,
        SEQ,
        numerics,
        ActivationCheckpoint::Blocks,
        u64::MAX,
    )
    .unwrap();
    assert_same(&off, &blocks, "uncapped");
    // Four blocks' intermediates against one block's plus the stream: well
    // under the uncheckpointed peak.
    assert!(
        blocks.peak * 10 < off.peak * 7,
        "peak {} B checkpointed, {} B not",
        blocks.peak,
        off.peak
    );
    let cap = blocks.peak;
    let capped = step(&spec, 1, SEQ, numerics, ActivationCheckpoint::Blocks, cap)
        .unwrap_or_else(|e| panic!("checkpointed step under its own peak {cap} B: {e:?}"));
    assert_same(&off, &capped, "checkpointed under the cap");
    match step(&spec, 1, SEQ, numerics, ActivationCheckpoint::Off, cap) {
        Err(OjasError::CapacityExceeded { .. }) => {}
        Err(other) => {
            panic!("uncheckpointed under {cap} B: expected CapacityExceeded, got {other:?}")
        }
        Ok(_) => panic!("the uncheckpointed step fit the checkpointed step's peak {cap} B"),
    }
}

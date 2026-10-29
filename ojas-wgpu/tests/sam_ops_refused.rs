//! `WgpuBackend` implements none of the SAM and encoder ops in
//! `SAM_AUTOCAST_POLICY`: each is the trait's refusing default, and
//! `docs/op-coverage.md` says "Unsupported" in the wgpu column for every one.
//! This pins that, so a device that starts answering one of them (or a wrong
//! answer taking the place of the refusal) is a test failure and not a doc
//! drift. Every operand is well formed, so a backend that implemented the op
//! would return `Ok` and fail the assertion; a refusal for a malformed
//! argument cannot pass for the default.
//!
//! Like the rest of this crate's tests it opens a real adapter and fails,
//! never skips, when there is none.

mod common;

use common::*;
use ojas_core::{
    AttentionOpts, Backend, BackendId, Conv2dParams, ConvTranspose2dParams, DType, OjasError,
    Tensor, SAM_AUTOCAST_POLICY,
};

/// A device tensor of `shape` filled from `seed`.
fn dev(seed: u64, shape: &[usize]) -> Tensor {
    up(&host(seed, shape))
}

/// The refusal an op gave, as `(op, detail)`; any other outcome is a failure
/// naming `label`.
fn refusal<T>(label: &str, got: Result<T, OjasError>) -> (&'static str, String) {
    match got {
        Err(OjasError::Unsupported { op, detail }) => {
            assert_eq!(op, label, "refused under another op's name");
            (op, detail)
        }
        Err(other) => panic!("{label}: expected Unsupported, got {other:?}"),
        Ok(_) => panic!("{label}: WgpuBackend answered an op the docs list as Unsupported"),
    }
}

#[test]
fn every_sam_op_is_the_refusing_default_on_wgpu() {
    let g = gpu();
    let x = dev(1, &[2, 8]);
    let img = dev(2, &[1, 4, 4, 4]);
    let w4 = dev(3, &[4]);
    let b4 = dev(4, &[4]);
    let w8 = dev(5, &[8]);
    let b8 = dev(6, &[8]);
    let q = dev(7, &[1, 2, 4, 8]);
    let o = dev(8, &[1, 2, 4, 8]);
    let lse = dev(9, &[1, 2, 4]);
    let tab = dev(10, &[4, 4]);
    let rope_x = dev(11, &[1, 4, 2, 8]);
    let opts = AttentionOpts::default();

    let mut seen: Vec<(&'static str, String)> = vec![
        refusal(
            "attention_forward",
            g.attention_forward(&q, &q, &q, opts).map(drop),
        ),
        refusal(
            "attention_forward_saving",
            g.attention_forward_saving(&q, &q, &q, opts).map(drop),
        ),
        refusal(
            "attention_backward",
            g.attention_backward(&q, &q, &q, &o, &lse, &o, opts, false)
                .map(drop),
        ),
        refusal(
            "linear_bias_forward",
            g.linear_bias_forward(&dev(12, &[3, 4]), &dev(13, &[5, 4]), &dev(14, &[5]))
                .map(drop),
        ),
        refusal(
            "conv2d_forward",
            g.conv2d_forward(
                &dev(15, &[1, 2, 6, 6]),
                &dev(16, &[4, 2, 3, 3]),
                None,
                Conv2dParams::new(),
            )
            .map(drop),
        ),
        refusal(
            "conv_transpose2d_forward",
            g.conv_transpose2d_forward(
                &dev(17, &[1, 2, 3, 3]),
                &dev(18, &[2, 4, 2, 2]),
                None,
                ConvTranspose2dParams::new(),
            )
            .map(drop),
        ),
        refusal("gelu_erf_forward", g.gelu_erf_forward(&x).map(drop)),
        refusal("relu_forward", g.relu_forward(&x).map(drop)),
        refusal("scale_forward", g.scale_forward(&x, 2.0).map(drop)),
        refusal(
            "broadcast_add_forward",
            g.broadcast_add_forward(&x, &w8).map(drop),
        ),
        refusal(
            "broadcast_mul_forward",
            g.broadcast_mul_forward(&x, &w8).map(drop),
        ),
        refusal(
            "max_pool2d_forward",
            g.max_pool2d_forward(&img, [2, 2], [2, 2]).map(drop),
        ),
        refusal(
            "interpolate_bilinear_forward",
            g.interpolate_bilinear_forward(&img, [8, 8], false)
                .map(drop),
        ),
        refusal(
            "rope_interleaved_forward",
            g.rope_interleaved_forward(&rope_x, &tab, &tab).map(drop),
        ),
        refusal(
            "rope_interleaved_backward",
            g.rope_interleaved_backward(&rope_x, &tab, &tab).map(drop),
        ),
        refusal(
            "layer_norm_forward",
            g.layer_norm_forward(&x, &w8, &b8, 1e-5).map(drop),
        ),
        refusal(
            "channel_layer_norm_forward",
            g.channel_layer_norm_forward(&img, &w4, &b4, 1e-5).map(drop),
        ),
        refusal(
            "group_norm_forward",
            g.group_norm_forward(&img, &w4, &b4, 2, 1e-5).map(drop),
        ),
        refusal("softmax_forward", g.softmax_forward(&x).map(drop)),
        refusal("concat", g.concat(&[&x, &x], 0).map(drop)),
        refusal("slice", g.slice(&x, 0, 0, 1).map(drop)),
        refusal("fill", g.fill(&[2, 2], DType::F32, 1.0).map(drop)),
    ];

    // The list above is the policy's list: a SAM op added to the policy
    // without a case here fails this assertion.
    let mut want: Vec<&str> = SAM_AUTOCAST_POLICY.iter().map(|(name, _)| *name).collect();
    let mut got: Vec<&str> = seen.iter().map(|(op, _)| *op).collect();
    want.sort_unstable();
    got.sort_unstable();
    assert_eq!(got, want, "the refusals cover exactly SAM_AUTOCAST_POLICY");

    // Each refusal is the default's, under the op's own name, from this
    // backend.
    for (op, detail) in seen.drain(..) {
        assert!(
            detail.contains(&format!("{:?}", BackendId::Wgpu)) && detail.contains(op),
            "{op}: detail {detail:?} is not the default's refusal"
        );
    }

    // Control: the harness distinguishes an implemented op from a refusal.
    let y = g
        .linear_forward(&dev(19, &[3, 4]), &dev(20, &[5, 4]))
        .expect("linear_forward is implemented on wgpu");
    assert_eq!(y.shape(), &[3, 5]);
}

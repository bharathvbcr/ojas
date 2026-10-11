//! Device temporaries are charged to the budget for the call that makes
//! them: the copies an op takes of operands its tessl kernels cannot read in
//! place, and the partials a split decode writes for its merge. Before, both
//! were allocated on the device past the budget's count.

#![cfg(all(target_os = "macos", feature = "metal"))]

mod common;

use common::*;
use ojas_core::{Backend, Budget, OjasError, Tensor};
use ojas_metal::MetalBackend;

const EPS: f32 = 1e-6;

/// An op of this test, its outputs in order.
fn run(m: &MetalBackend, name: &str, a: &[Tensor]) -> Result<Vec<Tensor>, OjasError> {
    match name {
        "conv" => m.causal_conv1d_silu_forward(&a[0], &a[1]).map(|y| vec![y]),
        "conv bwd" => m
            .causal_conv1d_silu_backward(&a[0], &a[1], &a[2])
            .map(|(dx, dw)| vec![dx, dw]),
        "norm" => m
            .gated_rms_norm_forward(&a[0], &a[1], &a[2], EPS)
            .map(|y| vec![y]),
        "norm bwd" => m
            .gated_rms_norm_backward(&a[0], &a[1], &a[2], &a[3], EPS)
            .map(|g| vec![g.input, g.gate, g.weight]),
        "linear" => m.linear_forward(&a[0], &a[1]).map(|y| vec![y]),
        "linear bwd" => m
            .linear_backward(&a[0], &a[1], &a[2])
            .map(|(gx, gw)| vec![gx, gw]),
        "gate" => m
            .per_head_sigmoid_gate_forward(&a[0], &a[1], &a[2], &a[3])
            .map(|y| vec![y]),
        "gate bwd" => m
            .per_head_sigmoid_gate_backward(&a[0], &a[1], &a[2], &a[3], &a[4])
            .map(|g| vec![g.input, g.weight, g.bias, g.attn_out]),
        other => panic!("no op {other}"),
    }
}

/// What one call charges past what was live before it, at its peak.
fn peak(m: &MetalBackend, call: impl FnOnce() -> Vec<Tensor>) -> u64 {
    let live = ok("live", m.budget().live_bytes());
    m.budget().reset_peak();
    drop(call());
    ok("sync", m.sync());
    m.budget().peak_bytes() - live
}

fn bytes(t: &Tensor) -> u64 {
    4 * t.shape().iter().product::<usize>() as u64
}

/// Operand `j` of a shifted call starts `lead(j)` elements into a padded
/// buffer ([`offset_view`]): never at 0 and never 16-byte aligned.
fn lead(j: usize) -> usize {
    3 + 2 * j
}

/// tessl's row-local kernels take whole buffers, so the device copies every
/// operand that does not start its buffer; its GEMMs take 16-byte-aligned
/// tensors, so the device copies a matrix operand off that alignment. The
/// copies are charged for the call like scratch: the same op over shifted
/// operands peaks higher by exactly their bytes, and a budget that holds
/// everything an aligned call charges but not the copies refuses the call
/// before the device allocates. Before, the shifted call charged what the
/// aligned one did.
#[test]
fn offset_operands_charge_the_copies_the_device_makes() {
    let (b, t, ch, k) = (2, 9, 8, 4);
    let x = rand(&[b, t, ch], 1, 2.0);
    let cw = rand(&[ch, k], 2, 1.0);
    let z = rand(&[b, t, ch], 3, 2.0);
    let nw = rand(&[ch], 4, 1.0);
    let gy = rand(&[b, t, ch], 5, 1.0);
    let (rows, din, nout, heads, dh) = (6, 8, 12, 3, 4);
    let lx = rand(&[rows, din], 6, 1.0);
    let lw = rand(&[nout, din], 7, 1.0);
    let lgy = rand(&[rows, nout], 8, 1.0);
    let gw = rand(&[heads, din], 9, 1.0);
    let gb = rand(&[heads], 10, 1.0);
    let ga = rand(&[rows, heads, dh], 11, 1.0);
    let ggy = rand(&[rows, heads, dh], 12, 1.0);
    // Each op's operands, and which of them the device copies when shifted:
    // all of a whole-buffer op's, only a GEMM's matrix operands.
    let ops: [(&str, Vec<&Tensor>, &[usize]); 8] = [
        ("conv", vec![&x, &cw], &[0, 1]),
        ("conv bwd", vec![&x, &cw, &gy], &[0, 1, 2]),
        ("norm", vec![&x, &z, &nw], &[0, 1, 2]),
        ("norm bwd", vec![&x, &z, &nw, &gy], &[0, 1, 2, 3]),
        ("linear", vec![&lx, &lw], &[0, 1]),
        ("linear bwd", vec![&lx, &lw, &lgy], &[0, 1, 2]),
        ("gate", vec![&lx, &gw, &gb, &ga], &[0, 1]),
        ("gate bwd", vec![&lx, &gw, &gb, &ga, &ggy], &[0, 1]),
    ];
    // Each operand aligned, then inside a padded buffer.
    let place = |m: &MetalBackend, operands: &[&Tensor]| {
        let aligned: Vec<Tensor> = operands.iter().map(|t| up(m, t)).collect();
        let shifted: Vec<Tensor> = (operands.iter().enumerate())
            .map(|(j, t)| offset_view(m, t, lead(j), j as u64))
            .collect();
        (aligned, shifted)
    };
    let m = metal();
    for (name, operands, copied) in ops {
        let copies: u64 = copied.iter().map(|&j| bytes(operands[j])).sum();
        let (aligned, shifted) = place(&m, &operands);
        let base = peak(&m, || ok(name, run(&m, name, &aligned)));
        assert_eq!(
            peak(&m, || ok(name, run(&m, name, &shifted))),
            base + copies,
            "{name}: shifted operands must charge their {copies} copied bytes"
        );

        // Room for the operands and all an aligned call charges; 4 bytes
        // short of the copies too.
        let held: u64 = (operands.iter().enumerate())
            .map(|(j, t)| 2 * bytes(t) + 4 * (lead(j) as u64 + 5))
            .sum();
        let tight = ok(
            "Metal",
            MetalBackend::new(Budget::new(held + base + copies - 4)),
        );
        let (aligned, shifted) = place(&tight, &operands);
        assert_eq!(ok("live", tight.budget().live_bytes()), held, "{name}");
        drop(ok(name, run(&tight, name, &aligned)));
        let r = run(&tight, name, &shifted);
        assert!(
            matches!(r, Err(OjasError::CapacityExceeded { .. })),
            "{name}: copies past the budget must be refused, got {r:?}"
        );
        ok("sync", tight.sync());
    }
}

/// A decode the device walks in splits writes each split's max, sum and
/// output for the merge: `rows * splits * (head_dim + 2)` values, charged
/// with the call. One pass writes none. Before, the partials were not
/// charged.
#[test]
fn a_split_decode_charges_its_partials() {
    let m = metal();
    // 12 heads of one query over 1024 keys: 8 splits (96 threadgroups); 63
    // keys: one pass.
    let (h, d, cap) = (12usize, 64usize, 1024usize);
    let q = up(&m, &rand(&[1, 1, h, d], 71, 1.0));
    let k = up(&m, &rand(&[1, cap, h, d], 72, 1.0));
    let v = up(&m, &rand(&[1, cap, h, d], 73, 1.0));
    let out = 4 * (h * d) as u64;
    for (kv_len, splits) in [(1024usize, 8usize), (63, 1)] {
        let parts = if splits > 1 { h * splits * (d + 2) } else { 0 };
        let got = peak(&m, || {
            vec![ok(
                "cached attention",
                m.cached_attention_forward(&q, &k, &v, kv_len, None),
            )]
        });
        assert_eq!(got, out + 4 * parts as u64, "kv_len {kv_len}");
    }
}

//! The `Tape` path of general attention with `key_lens[b] == 0` (a fully
//! masked batch row): the forward zeros, the backward skip and the bias
//! gradient pass. `attention_general.rs::lens_for` keeps every batch row
//! non-empty, so this wiring was never driven (`Rec::Attn` forward and
//! backward, `bias_grad` true for a live bias leaf).

use ojas_autograd::Tape;
use ojas_core::{Budget, Numerics, Tensor};
use ojas_cpu::CpuBackend;

fn vals(seed: u64, n: usize, amp: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * amp
        })
        .collect()
}

#[test]
fn a_batch_row_with_no_keys_has_zero_output_and_zero_gradients_on_the_tape() {
    // B 2, H 2 over 1 KV head, Tq 3, Tk 4, D 2; bias [2, 2, 3, 4] (one per batch).
    let (b, h, hkv, tq, tk, d) = (2usize, 2usize, 1usize, 3usize, 4usize, 2usize);
    let mut t = Tape::new(CpuBackend::new(Budget::new(1 << 24)).with_numerics(Numerics::Exact));
    let budget = t.backend().budget().clone();
    let f = |data: &[f32], shape: &[usize]| Tensor::from_f32(data, shape, &budget).unwrap();
    let q = t
        .leaf(f(&vals(1, b * h * tq * d, 0.8), &[b, h, tq, d]))
        .unwrap();
    let k = t
        .leaf(f(&vals(2, b * hkv * tk * d, 0.8), &[b, hkv, tk, d]))
        .unwrap();
    let v = t
        .leaf(f(&vals(3, b * hkv * tk * d, 0.8), &[b, hkv, tk, d]))
        .unwrap();
    let bias = t
        .leaf(f(&vals(4, b * h * tq * tk, 1.0), &[b, h, tq, tk]))
        .unwrap();
    let w = t
        .frozen_leaf(f(&vals(5, b * h * tq * d, 1.0), &[b, h, tq, d]))
        .unwrap();
    let lens = Tensor::from_u32(&[0, 3], &[b], &budget).unwrap();
    let y = t
        .attention(q, k, v, Some(bias), Some(lens), false, None)
        .unwrap();

    let out = t.value(y).unwrap().to_f32_vec().unwrap();
    let q_batch = h * tq * d;
    assert!(out[..q_batch].iter().all(|&x| x == 0.0), "empty row output");
    assert!(out[q_batch..].iter().any(|&x| x != 0.0), "live row output");

    let z = t.mul(y, w).unwrap();
    t.backward(z).unwrap();
    let grad = |var| t.grad(var).expect("gradient").to_f32_vec().unwrap();
    let (gq, gk, gv, gb) = (grad(q), grad(k), grad(v), grad(bias));
    let kv_batch = hkv * tk * d;
    let bias_batch = h * tq * tk;
    for (name, g, per_batch) in [
        ("q", &gq, q_batch),
        ("k", &gk, kv_batch),
        ("v", &gv, kv_batch),
        ("bias", &gb, bias_batch),
    ] {
        assert!(
            g.iter().all(|x| x.is_finite()),
            "{name}: non-finite gradient"
        );
        assert!(
            g[..per_batch].iter().all(|&x| x == 0.0),
            "{name}: the empty batch row took a gradient"
        );
        assert!(
            g[per_batch..].iter().any(|&x| x != 0.0),
            "{name}: the live batch row took none"
        );
    }
}

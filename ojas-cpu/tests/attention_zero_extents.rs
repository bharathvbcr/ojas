//! A zero extent in an attention operand, or in the cached-attention and
//! KV-cache operands, is refused with a Shape error through the public
//! `Backend`: no division by a zero head count, no panic.
//!
//! The per-operand rule (no zero extent) runs before the head-grouping
//! check, so these reach `Shape` by that rule; the layout check itself is
//! pinned by `shapes::tests::attention_layout_refuses_zero_head_counts_without_dividing`.

use ojas_core::{AttentionOpts, Backend, Budget, Tensor};
use ojas_cpu::CpuBackend;

mod common;
use common::{assert_range, assert_shape};

fn t(shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::from_f32(&vec![0.5; n], shape, &Budget::new(1 << 20)).unwrap()
}

fn lse_shape(q: &[usize]) -> Vec<usize> {
    q[..q.len() - 1].to_vec()
}

#[test]
fn attention_refuses_a_zero_extent_in_q_k_or_v_for_every_entry_point() {
    let cpu = CpuBackend::new(Budget::new(1 << 24));
    let opts = AttentionOpts::default();
    let good_q: &[usize] = &[1, 2, 4, 8];
    // (q, k and v): zero kv heads (the division the grouping check makes),
    // zero query heads, zero queries, zero keys, zero head dim, zero batch.
    let cases: [(&[usize], &[usize]); 8] = [
        (good_q, &[1, 0, 4, 8]),
        (&[1, 0, 4, 8], &[1, 0, 4, 8]),
        (&[1, 0, 4, 8], &[1, 2, 4, 8]),
        (&[1, 2, 0, 8], &[1, 2, 4, 8]),
        (good_q, &[1, 2, 0, 8]),
        (&[1, 2, 4, 0], &[1, 2, 4, 0]),
        (&[0, 2, 4, 8], &[0, 2, 4, 8]),
        (&[1, 2, 0, 8], &[1, 0, 0, 8]),
    ];
    for (qs, ks) in cases {
        let (q, k, v) = (t(qs), t(ks), t(ks));
        assert_shape(cpu.attention_forward(&q, &k, &v, opts));
        assert_shape(cpu.attention_forward_saving(&q, &k, &v, opts));
        let (y, lse, gy) = (t(qs), t(&lse_shape(qs)), t(qs));
        assert_shape(cpu.attention_backward(&q, &k, &v, &y, &lse, &gy, opts, false));
        // Causal and windowed forms take the same layout.
        let causal = AttentionOpts {
            causal: true,
            window: Some(2),
            ..AttentionOpts::default()
        };
        assert_shape(cpu.attention_forward(&q, &k, &v, causal));
    }
}

#[test]
fn a_grouping_that_does_not_divide_is_still_refused_and_a_valid_one_runs() {
    let cpu = CpuBackend::new(Budget::new(1 << 24));
    let opts = AttentionOpts::default();
    let q = t(&[1, 4, 3, 8]);
    for kv_heads in [3usize, 5, 8] {
        let k = t(&[1, kv_heads, 3, 8]);
        assert_shape(cpu.attention_forward(&q, &k, &k, opts));
    }
    for kv_heads in [1usize, 2, 4] {
        let k = t(&[1, kv_heads, 3, 8]);
        let y = cpu.attention_forward(&q, &k, &k, opts).unwrap();
        assert_eq!(y.shape(), [1, 4, 3, 8]);
    }
}

#[test]
fn the_cache_validators_refuse_a_zero_extent_before_dividing() {
    let cpu = CpuBackend::new(Budget::new(1 << 24));
    let q = t(&[1, 1, 2, 8]);
    // Zero kv heads, zero capacity and zero heads in the cache.
    for cache in [[1usize, 4, 0, 8], [1, 0, 2, 8], [0, 4, 2, 8]] {
        let (k, v) = (t(&cache), t(&cache));
        assert_shape(cpu.cached_attention_forward(&q, &k, &v, 1, None));
        assert_shape(cpu.cached_attention_forward(&q, &k, &v, 1, Some(1)));
    }
    // And a zero extent in the query.
    let cache = t(&[1, 4, 2, 8]);
    for qs in [[1usize, 1, 0, 8], [1, 0, 2, 8], [1, 1, 2, 0]] {
        assert_shape(cpu.cached_attention_forward(&t(&qs), &cache, &cache, 1, None));
    }
    // kv_cache_write: a zero extent in the cache or the source.
    let mut cache = t(&[1, 4, 2, 8]);
    for (c, s) in [
        ([1usize, 4, 0, 8], [1usize, 1, 0, 8]),
        ([1, 0, 2, 8], [1, 1, 2, 8]),
        ([1, 4, 2, 8], [1, 0, 2, 8]),
    ] {
        let (mut cache, src) = (t(&c), t(&s));
        assert_shape(cpu.kv_cache_write(&mut cache, &src, 0));
    }
    // More positions than slots is a range error, not a shape.
    assert_range(cpu.kv_cache_write(&mut cache, &t(&[1, 5, 2, 8]), 0));
}

//! `Autocast`'s hand-written attention methods (`attention_forward`,
//! `attention_forward_saving`, `attention_backward`) under a bf16 region:
//!   output = round(attention(round(q), round(k), round(v), bias = round(bias)))
//! with `key_lens` untouched and the log-sum-exp left unrounded F32. It pins
//! the three methods so a later edit that drops the bias from `prep`, rounds
//! the lse, or falls back to the refusing trait default fails here.

use ojas_core::{
    round_f32_to_bf16, AttentionOpts, Autocast, AutocastMode, Backend, Budget, Numerics, Tensor,
};
use ojas_cpu::CpuBackend;

fn raw() -> CpuBackend {
    CpuBackend::new(Budget::new(1 << 30)).with_numerics(Numerics::Exact)
}

fn f32t(values: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_f32(values, shape, &Budget::new(1 << 30)).unwrap()
}

fn values(n: usize, seed: u64, amp: f32) -> Vec<f32> {
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

fn rounded(t: &Tensor) -> Tensor {
    let v: Vec<f32> = t
        .f32_slice()
        .unwrap()
        .iter()
        .copied()
        .map(round_f32_to_bf16)
        .collect();
    f32t(&v, t.shape())
}

fn bits(t: &Tensor) -> Vec<u32> {
    t.to_f32_vec()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// q [2, 4, 5, 8], k and v [2, 2, 7, 8] (grouped), bias [1, 4, 5, 7], lens [7, 3].
struct Fixture {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    bias: Tensor,
    lens: Tensor,
    gy: Tensor,
}

fn fixture() -> Fixture {
    // Values that are not bf16-representable, so rounding changes them.
    Fixture {
        q: f32t(&values(2 * 4 * 5 * 8, 1, 1.0), &[2, 4, 5, 8]),
        k: f32t(&values(2 * 2 * 7 * 8, 2, 1.0), &[2, 2, 7, 8]),
        v: f32t(&values(2 * 2 * 7 * 8, 3, 1.0), &[2, 2, 7, 8]),
        bias: f32t(&values(4 * 5 * 7, 4, 2.0), &[1, 4, 5, 7]),
        lens: Tensor::from_u32(&[7, 3], &[2], &Budget::new(1 << 20)).unwrap(),
        gy: f32t(&values(2 * 4 * 5 * 8, 5, 1.0), &[2, 4, 5, 8]),
    }
}

#[test]
fn a_bf16_region_rounds_operands_and_output_and_keeps_lse_f32() {
    let f = fixture();
    let ac = Autocast::new(raw());
    let exact = raw();

    let rq = rounded(&f.q);
    let (rk, rv, rb) = (rounded(&f.k), rounded(&f.v), rounded(&f.bias));
    let want_opts = AttentionOpts {
        causal: false,
        window: None,
        bias: Some(&rb),
        key_lens: Some(&f.lens),
    };
    let (mid, want_lse) = exact
        .attention_forward_saving(&rq, &rk, &rv, want_opts)
        .unwrap();
    let want_y = rounded(&mid);

    let region = ac.autocast_region(AutocastMode::Bf16).unwrap();
    let got_opts = AttentionOpts {
        causal: false,
        window: None,
        bias: Some(&f.bias),
        key_lens: Some(&f.lens),
    };
    let y = ac.attention_forward(&f.q, &f.k, &f.v, got_opts).unwrap();
    let (y2, lse) = ac
        .attention_forward_saving(&f.q, &f.k, &f.v, got_opts)
        .unwrap();
    assert_eq!(bits(&y), bits(&want_y), "attention_forward output");
    assert_eq!(bits(&y2), bits(&want_y), "attention_forward_saving output");
    assert_eq!(bits(&lse), bits(&want_lse), "lse must stay unrounded F32");

    // Backward: operands rounded once, gradients emitted rounded.
    let rgy = rounded(&f.gy);
    let ry = rounded(&y2);
    let want = exact
        .attention_backward(&rq, &rk, &rv, &ry, &lse, &rgy, want_opts, true)
        .unwrap();
    let got = ac
        .attention_backward(&f.q, &f.k, &f.v, &y2, &lse, &f.gy, got_opts, true)
        .unwrap();
    drop(region);
    assert_eq!(bits(&got.q), bits(&rounded(&want.q)), "grad q");
    assert_eq!(bits(&got.k), bits(&rounded(&want.k)), "grad k");
    assert_eq!(bits(&got.v), bits(&rounded(&want.v)), "grad v");
    assert_eq!(
        bits(got.bias.as_ref().expect("bias gradient asked")),
        bits(&rounded(want.bias.as_ref().unwrap())),
        "grad bias"
    );
}

#[test]
fn outside_a_region_the_wrapper_is_the_inner_backend() {
    let f = fixture();
    let ac = Autocast::new(raw());
    let exact = raw();
    let opts = AttentionOpts {
        causal: false,
        window: None,
        bias: Some(&f.bias),
        key_lens: Some(&f.lens),
    };
    let got = ac.attention_forward(&f.q, &f.k, &f.v, opts).unwrap();
    let want = exact.attention_forward(&f.q, &f.k, &f.v, opts).unwrap();
    assert_eq!(bits(&got), bits(&want));
}

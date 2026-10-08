//! Exact linear under a bf16 region is round(linear(round(x), round(w))),
//! and every other op that rounds its operands accepts them from the CPU,
//! which takes `Bf16` storage for the matmul-class ops only.

use ojas_core::{
    round_f32_to_bf16, Autocast, AutocastMode, Backend, Budget, CeChunk, Numerics, Tensor,
};
use ojas_cpu::CpuBackend;

fn f32s(bits: &[u32], shape: &[usize], budget: &Budget) -> Tensor {
    let values: Vec<f32> = bits.iter().copied().map(f32::from_bits).collect();
    Tensor::from_f32(&values, shape, budget).unwrap()
}

fn rounded(t: &Tensor, budget: &Budget) -> Tensor {
    let values: Vec<f32> = t
        .f32_slice()
        .unwrap()
        .iter()
        .copied()
        .map(round_f32_to_bf16)
        .collect();
    Tensor::from_f32(&values, t.shape(), budget).unwrap()
}

#[test]
fn bf16_region_linear_is_the_round_of_the_rounded_product() {
    let budget = Budget::new(1 << 20);
    let x_bits = [0x3f80_0001, 0x3f81_8000, 0xbf00_0001, 0x0000_0001];
    let w_bits = [
        0x3f80_8000,
        0x4000_0001,
        0x3f00_0000,
        0x0080_0001,
        0x3f80_0000,
        0xbf80_0000,
        0x0080_0000,
        0x8000_0000,
    ];
    let x = f32s(&x_bits, &[1, 4], &budget);
    let w = f32s(&w_bits, &[2, 4], &budget);
    let exact = CpuBackend::new(Budget::new(1 << 20)).with_numerics(Numerics::Exact);
    let raw = exact.linear_forward(&x, &w).unwrap();
    let wrapped =
        Autocast::new(CpuBackend::new(Budget::new(1 << 20)).with_numerics(Numerics::Exact));
    let off = wrapped.linear_forward(&x, &w).unwrap();
    assert_eq!(
        off.f32_slice().unwrap(),
        raw.f32_slice().unwrap(),
        "an empty region leaves the exact product alone"
    );
    let region = wrapped.autocast_region(AutocastMode::Bf16).unwrap();
    let y = wrapped.linear_forward(&x, &w).unwrap();
    drop(region);
    let mid = exact
        .linear_forward(&rounded(&x, &budget), &rounded(&w, &budget))
        .unwrap();
    let want: Vec<u32> = mid
        .f32_slice()
        .unwrap()
        .iter()
        .map(|v| round_f32_to_bf16(*v).to_bits())
        .collect();
    let got: Vec<u32> = y.f32_slice().unwrap().iter().map(|v| v.to_bits()).collect();
    assert_eq!(got, want);
}

/// `n` values with low mantissa bits set, so rounding to bf16 moves them.
fn dirty(n: usize, seed: u32, budget: &Budget, shape: &[usize]) -> Tensor {
    let values: Vec<f32> = (0..n as u32)
        .map(|i| {
            let x = ((i.wrapping_mul(2_654_435_761) ^ seed) % 2001) as f32 / 1000.0 - 1.0;
            f32::from_bits(x.to_bits() | 0x1234)
        })
        .collect();
    Tensor::from_f32(&values, shape, budget).unwrap()
}

fn bits(t: &Tensor) -> Vec<u32> {
    t.to_f32_vec()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// Every non-matmul op that rounds its operands in a region (the per-head
/// gate's four entry points, the fused linear cross-entropy and cached
/// attention) accepts raw `F32` operands on the CPU, whose
/// `bf16_operands()` covers only linear and causal SDPA. Each result has
/// the bits of the same call on operands rounded beforehand
/// (`cast_bf16`), which pass through as they are.
#[test]
fn bf16_region_rounds_non_matmul_operands_in_f32_storage() {
    let budget = Budget::new(1 << 24);
    let ac = Autocast::new(CpuBackend::new(Budget::new(1 << 26)).with_numerics(Numerics::Exact));
    let _region = ac.autocast_region(AutocastMode::Bf16).unwrap();
    let r = |t: &Tensor| ac.cast_bf16(t).unwrap();
    let (n, dm, heads, hd) = (3usize, 8usize, 2usize, 4usize);
    let input = dirty(n * dm, 1, &budget, &[n, dm]);
    let weight = dirty(heads * dm, 2, &budget, &[heads, dm]);
    let bias = dirty(heads, 3, &budget, &[heads]);
    let attn = dirty(n * heads * hd, 4, &budget, &[n, heads, hd]);
    let gy = dirty(n * heads * hd, 5, &budget, &[n, heads, hd]);
    let (ri, rw, ra, rg) = (r(&input), r(&weight), r(&attn), r(&gy));

    let got = ac.per_head_sigmoid_gate_forward(&input, &weight, &bias, &attn);
    let want = ac
        .per_head_sigmoid_gate_forward(&ri, &rw, &bias, &ra)
        .unwrap();
    assert_eq!(bits(&got.expect("gate forward")), bits(&want));

    let got = ac.per_head_sigmoid_gate_backward(&input, &weight, &bias, &attn, &gy);
    let want = ac
        .per_head_sigmoid_gate_backward(&ri, &weight, &bias, &ra, &rg)
        .unwrap();
    let got = got.expect("gate backward");
    assert_eq!(bits(&got.input), bits(&want.input), "gate grad input");
    assert_eq!(bits(&got.weight), bits(&want.weight), "gate grad weight");
    assert_eq!(bits(&got.bias), bits(&want.bias), "gate grad bias");
    assert_eq!(bits(&got.attn_out), bits(&want.attn_out), "gate grad attn");

    let got = ac.per_head_sigmoid_gate_forward_saving(&input, &weight, &bias, &attn);
    let (y, scales) = got.expect("gate forward saving");
    let (wy, _) = ac
        .per_head_sigmoid_gate_forward_saving(&ri, &rw, &bias, &ra)
        .unwrap();
    assert_eq!(bits(&y), bits(&wy), "gate saving output");
    if let Some(scales) = scales {
        let got =
            ac.per_head_sigmoid_gate_backward_saved(&input, &weight, &bias, &attn, &gy, &scales);
        let want = ac
            .per_head_sigmoid_gate_backward_saved(&ri, &weight, &bias, &ra, &rg, &scales)
            .unwrap();
        let got = got.expect("gate backward saved");
        assert_eq!(bits(&got.input), bits(&want.input), "saved grad input");
        assert_eq!(bits(&got.attn_out), bits(&want.attn_out), "saved grad attn");
    }

    let (rows, d, vocab) = (4usize, 8usize, 16usize);
    let x = dirty(rows * d, 6, &budget, &[rows, d]);
    let table = dirty(vocab * d, 7, &budget, &[vocab, d]);
    let targets = Tensor::from_u32(&[1, 5, 9, 15], &[rows], &budget).unwrap();
    let chunk = CeChunk { rows: 2, cols: 8 };
    let got = ac.linear_cross_entropy_mean(&x, &table, &targets, None, chunk, true);
    let want = ac
        .linear_cross_entropy_mean(&r(&x), &r(&table), &targets, None, chunk, true)
        .unwrap();
    let got = got.expect("linear cross-entropy");
    assert_eq!(bits(&got.loss), bits(&want.loss), "ce loss");
    let pair = |a: &Option<Tensor>, b: &Option<Tensor>| {
        assert_eq!(bits(a.as_ref().unwrap()), bits(b.as_ref().unwrap()));
    };
    pair(&got.grad_input, &want.grad_input);
    pair(&got.grad_weight, &want.grad_weight);

    let (b, tq, h, hkv, dh, cap) = (1usize, 1usize, 2usize, 1usize, 4usize, 3usize);
    let q = dirty(b * tq * h * dh, 8, &budget, &[b, tq, h, dh]);
    let kc = dirty(b * cap * hkv * dh, 9, &budget, &[b, cap, hkv, dh]);
    let vc = dirty(b * cap * hkv * dh, 10, &budget, &[b, cap, hkv, dh]);
    let got = ac.cached_attention_forward(&q, &kc, &vc, 2, None);
    let want = ac
        .cached_attention_forward(&r(&q), &r(&kc), &r(&vc), 2, None)
        .unwrap();
    assert_eq!(bits(&got.expect("cached attention")), bits(&want));
}

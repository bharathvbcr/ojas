//! Exact linear under a bf16 region is round(linear(round(x), round(w))).

use ojas_core::{round_f32_to_bf16, Autocast, AutocastMode, Backend, Budget, Numerics, Tensor};
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

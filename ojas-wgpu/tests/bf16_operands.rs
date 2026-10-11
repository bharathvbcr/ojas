//! The bf16 operand contract (`ojas_oracle::bf16_operands`) on wgpu: every
//! mix of `F32` and `Bf16` operands of the matmul-class ops gives the
//! all-`F32` bits, the other ops refuse `Bf16`, a bf16 NaN or infinity is a
//! deferred `NonFinite`, and the device `to_bf16` / `to_f32` give the host
//! conversions' bits. It injects faults, so it runs on a backend of its own.

mod common;

use common::fresh;
use ojas_oracle::bf16_operands::bf16_operand_contract;

#[test]
fn wgpu_keeps_the_bf16_operand_contract() {
    let r = bf16_operand_contract(&fresh()).unwrap();
    eprintln!("wgpu: {r:?}");
    assert_eq!(r.mixed_calls, 2 * (2 * 10 + 2 * 38));
}

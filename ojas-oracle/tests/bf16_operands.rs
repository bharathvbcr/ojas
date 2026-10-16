//! The bf16 operand contract on `CpuBackend`, under both numerics tiers.
//! Device backends run the same runner from their own test crates.

use ojas_core::{Autocast, Budget, Numerics};
use ojas_cpu::CpuBackend;
use ojas_oracle::bf16_operands::bf16_operand_contract;

fn cpu(numerics: Numerics) -> CpuBackend {
    CpuBackend::new(Budget::new(1 << 30)).with_numerics(numerics)
}

#[test]
fn cpu_exact_keeps_the_bf16_operand_contract() {
    let r = bf16_operand_contract(&cpu(Numerics::Exact)).unwrap();
    eprintln!("cpu exact: {r:?}");
    // 2 linear shapes x (3 + 7) mixes, 2 attention shapes x (7 + 31) mixes,
    // each packed and at an odd element offset.
    // The tied table: one Bf16 slot (2 layouts) in embedding_forward,
    // embedding_backward and 8 fused cross-entropy calls (2 chunks, each
    // asking for each of the four `CeGrads` combinations).
    assert!(r.table_lane);
    assert_eq!(r.mixed_calls, 2 * (2 * 10 + 2 * 38) + 2 * 10);
    assert_eq!(r.refusals, 2 + 14);
    assert_eq!(r.non_finite, 2 * 5 + 2 * 8 + 3);
}

#[test]
fn cpu_fast_keeps_the_bf16_operand_contract() {
    let r = bf16_operand_contract(&cpu(Numerics::Fast)).unwrap();
    eprintln!("cpu fast: {r:?}");
    assert!(r.table_lane);
}

/// `Autocast` forwards `bf16_operands()` and every matmul-class op, so the
/// wrapper keeps the contract outside a region.
#[test]
fn autocast_over_cpu_keeps_the_bf16_operand_contract() {
    let r = bf16_operand_contract(&Autocast::new(cpu(Numerics::Exact))).unwrap();
    eprintln!("autocast cpu exact: {r:?}");
    assert!(r.table_lane);
}

//! CPU verification of Gemma Embedding ops against PyTorch goldens.

use ojas_core::{Budget, Numerics};
use ojas_cpu::CpuBackend;
use ojas_oracle::gemma_golden::check;

#[test]
fn gemma_ops_match_torch_goldens_on_cpu() {
    let be = CpuBackend::with_threads(Budget::new(1 << 33), 2)
        .unwrap()
        .with_numerics(Numerics::Exact);
    let report = check(&be).expect("gemma golden cases run on cpu");
    report.print("cpu");
}

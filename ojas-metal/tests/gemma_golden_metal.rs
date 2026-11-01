//! Metal verification of Gemma Embedding ops against PyTorch goldens.

#![cfg(all(target_os = "macos", feature = "metal"))]

mod common;

use common::metal;
use ojas_oracle::gemma_golden::check;

#[test]
fn gemma_ops_match_torch_goldens_on_metal() {
    let m = metal();
    let report = check(&m).expect("gemma golden cases run on metal");
    report.print("metal");
}

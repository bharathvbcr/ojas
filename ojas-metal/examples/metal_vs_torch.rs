//! ojas `MetalBackend` lane of the paired GPU-vs-torch benchmark.
//!
//! Run through `bench/run_paired.sh`; see `bench/README.md`. The rows live in
//! `bench/ojas_rows.rs`, shared with `ojas-wgpu/examples/wgpu_vs_torch.rs`.
//!
//! Every `MetalBackend` op ends with one device synchronize before it returns
//! (`ojas-metal/src/device.rs` module docs), so `Backend::sync` (the trait
//! default, a no-op) adds nothing and wall time around a call is
//! device-complete time.

#[path = "../../bench/ojas_rows.rs"]
#[allow(dead_code)]
mod rows;

use ojas_core::Budget;
use ojas_metal::MetalBackend;

fn main() {
    let be = match MetalBackend::new(Budget::new(48 << 30)) {
        Ok(be) => be,
        Err(e) => {
            eprintln!("MetalBackend::new: {e}");
            std::process::exit(2);
        }
    };
    let device = format!(
        "{{\"backend\":\"MetalBackend\",\"adapter\":{}}}",
        rows::json_string(be.device_name())
    );
    if let Err(e) = rows::main_with(&be, "ojas-metal", &device) {
        eprintln!("metal_vs_torch: {e}");
        std::process::exit(1);
    }
}

//! ojas `WgpuBackend` lane of the paired GPU-vs-torch benchmark.
//!
//! Run through `bench/run_paired.sh`; see `bench/README.md`. The rows live in
//! `bench/ojas_rows.rs`, shared with `ojas-metal/examples/metal_vs_torch.rs`.
//!
//! `WgpuBackend` ops record into one shared encoder and do not wait, so each
//! timed iteration is the op followed by `Backend::sync` (submit, wait,
//! and report a deferred non-finite fault).

#[path = "../../bench/ojas_rows.rs"]
#[allow(dead_code)]
mod rows;

use ojas_core::Budget;
use ojas_wgpu::WgpuBackend;

fn main() {
    let be = match WgpuBackend::open(Budget::new(48 << 30)) {
        Ok(be) => be,
        Err(e) => {
            eprintln!("WgpuBackend::open: {e}");
            std::process::exit(2);
        }
    };
    let ctx = be.context();
    let device = format!(
        "{{\"backend\":\"WgpuBackend\",\"adapter\":{},\"hal\":{},\"vendor\":{},\"binding_cap_bytes\":{}}}",
        rows::json_string(ctx.adapter_name()),
        rows::json_string(ctx.hal()),
        rows::json_string(ctx.vendor()),
        ctx.binding_cap()
    );
    if let Err(e) = rows::main_with(&be, "ojas-wgpu", &device) {
        eprintln!("wgpu_vs_torch: {e}");
        std::process::exit(1);
    }
}

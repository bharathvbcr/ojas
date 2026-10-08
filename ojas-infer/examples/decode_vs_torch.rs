//! Decode lanes of the paired benchmark: `bench/decode_rows.rs` on one
//! runtime, chosen by the first argument.
//!
//! - `cpu`: `DeviceDecoder<CpuBackend>` (runtime `ojas-cpu`).
//! - `cpu-host`: `CpuGpt`, the host fast path (runtime `ojas-cpu-host`).
//! - `metal`: `DeviceDecoder<MetalBackend>` (runtime `ojas-metal`).
//! - `wgpu`: `DeviceDecoder<WgpuBackend>` (runtime `ojas-wgpu`).
//!
//! Run through `bench/run_paired.sh`; see `bench/README.md`. Every timed
//! iteration ends with `Backend::sync`, as in the other lanes.

#[path = "../../bench/ojas_rows.rs"]
#[allow(dead_code)]
mod rows;

#[path = "../../bench/decode_rows.rs"]
mod decode;

use ojas_core::Budget;
use ojas_cpu::CpuBackend;

const BUDGET: u64 = 48 << 30;

fn fail(what: &str, e: impl std::fmt::Display) -> ! {
    eprintln!("decode_vs_torch: {what}: {e}");
    std::process::exit(2);
}

fn main() {
    let lane = std::env::args().nth(1).unwrap_or_default();
    let result = match lane.as_str() {
        "cpu" => {
            let be = CpuBackend::new(Budget::new(BUDGET));
            let device = "{\"backend\":\"CpuBackend\"}";
            rows::main_rows(&be, "ojas-cpu", device, decode::run_device)
        }
        "cpu-host" => {
            let be = CpuBackend::new(Budget::new(BUDGET));
            let device = "{\"backend\":\"CpuGpt\"}";
            rows::main_rows(&be, "ojas-cpu-host", device, decode::run_host)
        }
        "metal" => metal(),
        "wgpu" => {
            let be = ojas_wgpu::WgpuBackend::open(Budget::new(BUDGET))
                .unwrap_or_else(|e| fail("WgpuBackend::open", e));
            let ctx = be.context();
            let device = format!(
                "{{\"backend\":\"WgpuBackend\",\"adapter\":{},\"hal\":{}}}",
                rows::json_string(ctx.adapter_name()),
                rows::json_string(ctx.hal())
            );
            rows::main_rows(&be, "ojas-wgpu", &device, decode::run_device)
        }
        other => fail("lane", format!("{other:?} is not cpu, cpu-host, metal or wgpu")),
    };
    if let Err(e) = result {
        eprintln!("decode_vs_torch: {e}");
        std::process::exit(1);
    }
}

#[cfg(target_os = "macos")]
fn metal() -> Result<(), String> {
    let be = ojas_metal::MetalBackend::new(Budget::new(BUDGET))
        .unwrap_or_else(|e| fail("MetalBackend::new", e));
    let device = format!(
        "{{\"backend\":\"MetalBackend\",\"adapter\":{}}}",
        rows::json_string(be.device_name())
    );
    rows::main_rows(&be, "ojas-metal", &device, decode::run_device)
}

#[cfg(not(target_os = "macos"))]
fn metal() -> Result<(), String> {
    Err("the metal lane runs on macOS only".into())
}

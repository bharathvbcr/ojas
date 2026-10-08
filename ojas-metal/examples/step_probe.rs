//! The step-throughput probes on `MetalBackend` (`bench/step_probe.rs`).
//!
//! `OJAS_PROBE_OUT=<file> [OJAS_PROBES=adamw,link] cargo run -p ojas-metal
//! --release --example step_probe`

#[path = "../../bench/ojas_rows.rs"]
#[allow(dead_code)]
mod ojas_rows;
#[path = "../../bench/step_probe.rs"]
mod step_probe;

use ojas_core::{Budget, OjasError};
use ojas_metal::MetalBackend;

impl step_probe::Knobs for MetalBackend {
    fn flush_values(&self) -> Vec<usize> {
        Vec::new()
    }

    fn set_flush(&self, _dispatches: usize) {}

    fn round_trip(&self) -> Option<Result<(), OjasError>> {
        Some(self.memory().map(drop))
    }
}

fn main() {
    let be = match MetalBackend::new(Budget::new(48 << 30)) {
        Ok(be) => be,
        Err(e) => {
            eprintln!("MetalBackend::new: {e}");
            std::process::exit(2);
        }
    };
    if let Err(e) = step_probe::probe_main(&be, "ojas-metal") {
        eprintln!("step_probe: {e}");
        std::process::exit(1);
    }
}

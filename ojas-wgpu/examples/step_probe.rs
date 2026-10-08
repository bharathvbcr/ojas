//! The step-throughput probes on `WgpuBackend` (`bench/step_probe.rs`).
//!
//! `OJAS_PROBE_OUT=<file> [OJAS_PROBES=flush,adamw] cargo run -p ojas-wgpu
//! --release --example step_probe`

#[path = "../../bench/ojas_rows.rs"]
#[allow(dead_code)]
mod ojas_rows;
#[path = "../../bench/step_probe.rs"]
mod step_probe;

use ojas_core::{Budget, OjasError};
use ojas_wgpu::{WgpuBackend, FLUSH_AT};

impl step_probe::Knobs for WgpuBackend {
    /// The default first: the sweep leaves it set.
    fn flush_values(&self) -> Vec<usize> {
        let mut v = vec![FLUSH_AT];
        v.extend([16, 32, 64, 128, 256, 1024].into_iter().filter(|&f| f != FLUSH_AT));
        v
    }

    fn set_flush(&self, dispatches: usize) {
        self.context().set_flush_at(dispatches);
    }

    fn round_trip(&self) -> Option<Result<(), OjasError>> {
        None
    }
}

fn main() {
    let be = match WgpuBackend::open(Budget::new(48 << 30)) {
        Ok(be) => be,
        Err(e) => {
            eprintln!("WgpuBackend::open: {e}");
            std::process::exit(2);
        }
    };
    if let Err(e) = step_probe::probe_main(&be, "ojas-wgpu") {
        eprintln!("step_probe: {e}");
        std::process::exit(1);
    }
}

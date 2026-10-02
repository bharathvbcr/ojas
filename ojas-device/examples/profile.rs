//! Print what ojas can see on this machine and the plan for a budget.
//!
//! ```bash
//! cargo run --release -p ojas-device --example profile -- [budget_bytes] [--bandwidth]
//! ```
//!
//! `--bandwidth` runs the bounded copy measurement (at most 64 MiB per
//! buffer, about half a second). Without it nothing is timed.

use ojas_device::{
    measure_bandwidth, probe_system, BandwidthConfig, GemmBlocks, ResourcePlan, ResourcePolicy,
};

/// The probe list is empty: this example opens no GPU.
struct NoProbe;

impl ojas_device::MemoryProbe for NoProbe {
    fn kind(&self) -> ojas_device::Device {
        ojas_device::Device::Cpu
    }
    fn memory_bytes(&self) -> ojas_device::MemoryReport {
        ojas_device::MemoryReport::Unknown
    }
}

fn main() {
    let mut budget = u64::MAX;
    let mut bandwidth = false;
    for arg in std::env::args().skip(1) {
        if arg == "--bandwidth" {
            bandwidth = true;
        } else {
            match arg.parse::<u64>() {
                Ok(b) if b > 0 => budget = b,
                _ => {
                    eprintln!("usage: profile [budget_bytes > 0] [--bandwidth]");
                    std::process::exit(2);
                }
            }
        }
    }
    let profile = probe_system();
    println!("{profile:#?}");
    let mut plan = ResourcePlan::derive(&ResourcePolicy::new(budget), &profile, &[] as &[NoProbe]);
    if bandwidth {
        match BandwidthConfig::for_profile(&profile).and_then(|c| measure_bandwidth(&c)) {
            Ok(bw) => {
                println!("{bw:#?}");
                plan = plan.with_bandwidth(&bw);
            }
            Err(e) => println!("bandwidth not measured: {e}"),
        }
    }
    println!("{plan:#?}");
    // The CPU backend's register tile is 6 x 16 f32 (ojas-cpu `MR`, `NR`).
    println!("{:?}", GemmBlocks::derive(&plan.cache, 6, 16, 4));
}

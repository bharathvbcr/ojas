//! One snapshot of the machine: memory, CPU topology, memory architecture
//! and memory pressure. Bandwidth is measured separately and only when a
//! caller asks ([`crate::measure_bandwidth`]).

use crate::host::{probe_host, HostMemory};
use crate::topology::{probe_topology, CpuTopology};

/// Whether the CPU and the GPU draw from one physical memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryArchitecture {
    /// One pool: Apple silicon. A GPU allocation is host RAM the CPU budget
    /// can no longer use, and the reverse.
    Unified,
    /// The device has its own memory (a PCIe GPU).
    Discrete,
    /// The probe could not tell. Linux and x86 macOS hosts report this from
    /// the host probe; a GPU runtime's [`crate::MemoryProbe`] can say more.
    Unknown,
}

/// The kernel's own reading of memory pressure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryPressure {
    Normal,
    Warning,
    Critical,
    Unknown,
}

/// What [`probe_system`] could read. Each part fails on its own.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SystemProfile {
    pub memory: HostMemory,
    pub cpu: CpuTopology,
    pub architecture: MemoryArchitecture,
    pub pressure: MemoryPressure,
}

impl SystemProfile {
    /// A profile that knows only `memory`. For tests and for callers that
    /// probe memory on their own.
    pub fn from_memory(memory: HostMemory) -> Self {
        Self {
            memory,
            cpu: CpuTopology::all_unknown(),
            architecture: MemoryArchitecture::Unknown,
            pressure: MemoryPressure::Unknown,
        }
    }
}

/// Probe memory, topology, architecture and pressure. Cheap: sysctls and
/// small `/proc` and `/sys` reads, no allocation past a few KiB, no timing.
pub fn probe_system() -> SystemProfile {
    let mut cpu = probe_topology();
    let memory = probe_host();
    // One source for the usable count: the memory probe and the topology
    // probe both read `available_parallelism`; keep them equal.
    cpu.usable = memory.cpu_count;
    SystemProfile {
        memory,
        cpu,
        architecture: probe_architecture(),
        pressure: probe_pressure(),
    }
}

/// Apple silicon (`hw.optional.arm64` = 1) has unified memory on every
/// model. An x86 Mac may have a discrete GPU, an integrated one, or both,
/// and is unknown here, as is every non-macOS host.
fn probe_architecture() -> MemoryArchitecture {
    #[cfg(target_os = "macos")]
    {
        match crate::sysctl::u64_by_name(b"hw.optional.arm64\0") {
            Some(1) => MemoryArchitecture::Unified,
            _ => MemoryArchitecture::Unknown,
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        MemoryArchitecture::Unknown
    }
}

/// macOS `kern.memorystatus_vm_pressure_level`: 1 normal, 2 warning,
/// 4 critical (`DISPATCH_MEMORYPRESSURE_*`). Other values are unknown.
fn probe_pressure() -> MemoryPressure {
    #[cfg(target_os = "macos")]
    {
        pressure_from_level(crate::sysctl::u64_by_name(
            b"kern.memorystatus_vm_pressure_level\0",
        ))
    }
    #[cfg(not(target_os = "macos"))]
    {
        MemoryPressure::Unknown
    }
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn pressure_from_level(level: Option<u64>) -> MemoryPressure {
    match level {
        Some(1) => MemoryPressure::Normal,
        Some(2) => MemoryPressure::Warning,
        Some(4) => MemoryPressure::Critical,
        _ => MemoryPressure::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pressure_levels_map_and_others_are_unknown() {
        assert_eq!(pressure_from_level(Some(1)), MemoryPressure::Normal);
        assert_eq!(pressure_from_level(Some(2)), MemoryPressure::Warning);
        assert_eq!(pressure_from_level(Some(4)), MemoryPressure::Critical);
        for other in [None, Some(0), Some(3), Some(8), Some(u64::MAX)] {
            assert_eq!(pressure_from_level(other), MemoryPressure::Unknown);
        }
    }

    #[test]
    fn the_profile_agrees_with_its_parts() {
        let p = probe_system();
        assert_eq!(p.cpu.usable, p.memory.cpu_count);
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        assert_eq!(p.architecture, MemoryArchitecture::Unified);
        #[cfg(not(target_os = "macos"))]
        assert_eq!(p.architecture, MemoryArchitecture::Unknown);
        #[cfg(target_os = "macos")]
        assert_ne!(p.pressure, MemoryPressure::Unknown);
    }

    #[test]
    fn probing_from_many_threads_at_once_is_consistent() {
        let first = probe_system();
        std::thread::scope(|s| {
            let handles: Vec<_> = (0..16).map(|_| s.spawn(probe_system)).collect();
            for h in handles {
                let p = h.join().expect("probe panicked");
                // Memory figures move; the topology does not.
                assert_eq!(p.cpu, first.cpu);
                assert_eq!(p.architecture, first.architecture);
                assert_eq!(p.memory.total_bytes, first.memory.total_bytes);
            }
        });
    }
}

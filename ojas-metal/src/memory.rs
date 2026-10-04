//! What the Metal device reports about its memory, as an
//! [`ojas_device::MemoryProbe`] for [`ojas_device::ResourcePlan`].

use ojas_device::{Device, MemoryArchitecture, MemoryProbe, MemoryReport};

/// One reading of the device's memory, from [`crate::MetalBackend::memory`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MetalMemory {
    /// `MTLDevice.recommendedMaxWorkingSetSize`, read by tessl when the
    /// device opened. 0 means the device did not report one.
    pub recommended_working_set: u64,
    /// `MTLDevice.currentAllocatedSize` when this reading was taken: every
    /// buffer this process holds on the device, through every backend and
    /// session, including tessl's pools. It is not this backend's share.
    pub allocated: u64,
    /// `MTLDevice.hasUnifiedMemory`, read by tessl when the device opened:
    /// true when the GPU has no memory of its own and draws on system RAM
    /// (Apple silicon), false for a discrete GPU with its own VRAM.
    pub has_unified_memory: bool,
}

impl MemoryProbe for MetalMemory {
    fn kind(&self) -> Device {
        Device::Metal
    }

    fn memory_bytes(&self) -> MemoryReport {
        match self.recommended_working_set {
            0 => MemoryReport::Unknown,
            n => MemoryReport::Known(n),
        }
    }

    fn resident_bytes(&self) -> MemoryReport {
        MemoryReport::Known(self.allocated)
    }

    /// From the device itself rather than the host profile, so a device
    /// that shares system RAM is planned inside the shared budget even
    /// where the host profile cannot tell (an Intel Mac reports `Unknown`).
    fn architecture(&self) -> MemoryArchitecture {
        if self.has_unified_memory {
            MemoryArchitecture::Unified
        } else {
            MemoryArchitecture::Discrete
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_zero_working_set_is_unknown_and_residency_is_reported() {
        let m = MetalMemory {
            recommended_working_set: 0,
            allocated: 7,
            has_unified_memory: true,
        };
        assert_eq!(m.kind(), Device::Metal);
        assert_eq!(m.memory_bytes(), MemoryReport::Unknown);
        assert_eq!(m.resident_bytes(), MemoryReport::Known(7));
        assert_eq!(m.architecture(), MemoryArchitecture::Unified);
        let m = MetalMemory {
            recommended_working_set: 48 << 30,
            allocated: 0,
            has_unified_memory: false,
        };
        assert_eq!(m.memory_bytes(), MemoryReport::Known(48 << 30));
        assert_eq!(m.architecture(), MemoryArchitecture::Discrete);
    }

    /// The device's own answer wins over the host profile's in the plan: a
    /// unified device shares the host budget even when the host profile is
    /// `Unknown` or says `Discrete`, and a discrete one never does.
    #[test]
    fn the_device_flag_decides_whether_metal_shares_the_host_budget() {
        use ojas_device::{HostMemory, ResourcePlan, ResourcePolicy, SystemProfile};
        for host_arch in [
            MemoryArchitecture::Unknown,
            MemoryArchitecture::Unified,
            MemoryArchitecture::Discrete,
        ] {
            for unified in [true, false] {
                let mut host = HostMemory::all_unknown();
                host.available_bytes = MemoryReport::Known(32 << 30);
                let mut profile = SystemProfile::from_memory(host);
                profile.architecture = host_arch;
                let mut policy = ResourcePolicy::new(8 << 30);
                policy.devices = vec![Device::Metal, Device::Cpu];
                let m = MetalMemory {
                    recommended_working_set: 24 << 30,
                    allocated: 1 << 30,
                    has_unified_memory: unified,
                };
                let plan = ResourcePlan::derive(&policy, &profile, &[m]);
                assert_eq!(plan.device_shares_host[0], unified, "{host_arch:?}");
                assert_eq!(plan.shared_budget, unified, "{host_arch:?}");
                if unified {
                    match plan.device_room[0] {
                        MemoryReport::Known(room) => assert!(room <= plan.budget_bytes),
                        MemoryReport::Unknown => panic!("a shared device has a room"),
                    }
                }
            }
        }
    }
}

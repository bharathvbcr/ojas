//! What a wgpu device reports about its memory, as an
//! [`ojas_device::MemoryProbe`] for [`ojas_device::ResourcePlan`].
//!
//! wgpu 30.0.1 reports no device memory size: `AdapterInfo` names the
//! adapter and its type, and `Limits` bound one buffer or binding
//! (`max_buffer_size` is a per-buffer cap, not memory). So
//! [`WgpuMemory::memory_bytes`] is always [`MemoryReport::Unknown`], the
//! plan's room for a wgpu device stays unknown, and no number is stood in
//! for it. What wgpu does report is used: the adapter type gives the memory
//! architecture, and the allocator report (`Device::generate_allocator_report`,
//! kept by the Vulkan and DX12 HALs, `None` on Metal and GL) gives residency.

use ojas_device::{Device, MemoryArchitecture, MemoryProbe, MemoryReport};

/// One reading of a context's device, from [`crate::WgpuContext::memory`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WgpuMemory {
    /// `AdapterInfo::device_type` of the opened adapter.
    pub device_type: wgpu::DeviceType,
    /// `AllocatorReport::total_reserved_bytes`: the memory blocks wgpu's
    /// allocator holds on the device, live allocations and their free
    /// space. `None` when the HAL keeps no report.
    pub reserved_bytes: Option<u64>,
    /// [`crate::POOL_CAP_BYTES`]: the most freed buffers the context keeps.
    pub pool_cache_cap: u64,
}

impl MemoryProbe for WgpuMemory {
    /// The portable path's device kind (see the crate docs).
    fn kind(&self) -> Device {
        Device::Vulkan
    }

    fn memory_bytes(&self) -> MemoryReport {
        MemoryReport::Unknown
    }

    fn resident_bytes(&self) -> MemoryReport {
        self.reserved_bytes
            .map_or(MemoryReport::Unknown, MemoryReport::Known)
    }

    fn pool_cache_bytes(&self) -> MemoryReport {
        MemoryReport::Known(self.pool_cache_cap)
    }

    /// An integrated GPU and a CPU adapter draw on host RAM; a discrete
    /// GPU has its own. A virtual or unclassified adapter is unknown, and
    /// the host profile decides.
    fn architecture(&self) -> MemoryArchitecture {
        match self.device_type {
            wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::Cpu => MemoryArchitecture::Unified,
            wgpu::DeviceType::DiscreteGpu => MemoryArchitecture::Discrete,
            wgpu::DeviceType::VirtualGpu | wgpu::DeviceType::Other => MemoryArchitecture::Unknown,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ojas_device::{HostMemory, ResourcePlan, ResourcePolicy, SystemProfile};

    /// No adapter type yields a device memory figure, so the plan's room
    /// stays unknown; the type alone decides whether the device shares the
    /// host budget, and the session budget is then the host-cut or the
    /// caller's number, never a default.
    #[test]
    fn the_plan_records_an_unknown_room_and_takes_the_adapter_type() {
        for (device_type, shares) in [
            (wgpu::DeviceType::IntegratedGpu, Some(true)),
            (wgpu::DeviceType::Cpu, Some(true)),
            (wgpu::DeviceType::DiscreteGpu, Some(false)),
            (wgpu::DeviceType::VirtualGpu, None),
            (wgpu::DeviceType::Other, None),
        ] {
            for reserved_bytes in [None, Some(0), Some(7 << 20)] {
                let m = WgpuMemory {
                    device_type,
                    reserved_bytes,
                    pool_cache_cap: crate::POOL_CAP_BYTES,
                };
                assert_eq!(m.memory_bytes(), MemoryReport::Unknown);
                assert_eq!(
                    m.resident_bytes(),
                    reserved_bytes.map_or(MemoryReport::Unknown, MemoryReport::Known)
                );
                let mut host = HostMemory::all_unknown();
                host.available_bytes = MemoryReport::Known(2 << 30);
                let mut profile = SystemProfile::from_memory(host);
                profile.architecture = MemoryArchitecture::Discrete;
                let mut policy = ResourcePolicy::new(8 << 30);
                policy.devices = vec![Device::Vulkan, Device::Cpu];
                let plan = ResourcePlan::derive(&policy, &profile, &[m]);
                assert_eq!(plan.device_memory[0], MemoryReport::Unknown);
                assert_eq!(plan.device_room[0], MemoryReport::Unknown);
                assert_eq!(
                    plan.device_pool_cache[0],
                    MemoryReport::Known(crate::POOL_CAP_BYTES)
                );
                let shares = shares.unwrap_or(false);
                assert_eq!(plan.device_shares_host[0], shares, "{device_type:?}");
                assert_eq!(plan.shared_budget, shares);
                let want = if shares { 2 << 30 } else { 8 << 30 };
                assert_eq!(plan.device_budget(Device::Vulkan), Some(want));
            }
        }
        // An unclassified adapter on a host the profile calls unified (Apple
        // silicon, or a Linux host with no GPU of its own) stays on the
        // shared budget: host and cgroup limits bound it, not the caller's
        // number alone.
        let mut host = HostMemory::all_unknown();
        host.cgroup_limit_bytes = MemoryReport::Known(1 << 30);
        let mut profile = SystemProfile::from_memory(host);
        profile.architecture = MemoryArchitecture::Unified;
        let mut policy = ResourcePolicy::new(8 << 30);
        policy.devices = vec![Device::Vulkan, Device::Cpu];
        let other = WgpuMemory {
            device_type: wgpu::DeviceType::Other,
            reserved_bytes: None,
            pool_cache_cap: crate::POOL_CAP_BYTES,
        };
        let plan = ResourcePlan::derive(&policy, &profile, &[other]);
        assert!(plan.shared_budget);
        assert_eq!(plan.device_budget(Device::Vulkan), Some(1 << 30));
    }
}

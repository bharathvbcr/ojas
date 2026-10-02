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

    /// tessl does not expose `MTLDevice.hasUnifiedMemory`, so this defers
    /// to the host profile, which reports unified memory on Apple silicon
    /// and unknown on an Intel Mac.
    fn architecture(&self) -> MemoryArchitecture {
        MemoryArchitecture::Unknown
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
        };
        assert_eq!(m.kind(), Device::Metal);
        assert_eq!(m.memory_bytes(), MemoryReport::Unknown);
        assert_eq!(m.resident_bytes(), MemoryReport::Known(7));
        assert_eq!(m.architecture(), MemoryArchitecture::Unknown);
        let m = MetalMemory {
            recommended_working_set: 48 << 30,
            allocated: 0,
        };
        assert_eq!(m.memory_bytes(), MemoryReport::Known(48 << 30));
    }
}

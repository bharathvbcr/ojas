//! `SYSTEM_PROFILE`: what this machine and process can use, for a host
//! that sizes `threads` and the memory ceiling from it.
//!
//! The op reads; it changes nothing. It does not open a GPU (a Metal
//! probe needs a device thread), so device rooms are not in the record.
//!
//! Payload: empty; `budget: u64`; or `budget: u64, flags: u32`. Flag
//! [`FLAG_BANDWIDTH`] runs [`ojas_device::cached_bandwidth`] (about half a
//! second the first time in a process, a cached figure after that) and
//! fills the bandwidth fields; any other flag bit is refused.
//! The measurement runs on the job thread and is not cancellable once
//! started; it is bounded (two phases of at most 250 ms plus one repetition
//! each), and concurrent callers wait for it rather than measuring again.
//!
//! Result: `version: u32 = 1`, `count: u32`, then `count` entries of
//! `known: u8, value: u64` in [`Field`] order. `known` 0 means the probe
//! could not read it, or (for the bandwidth fields) it was not asked for or
//! the measurement was refused, and `value` is 0. A later version only
//! appends fields, so a reader takes the first `count` it knows.

use ojas_device::{
    cached_bandwidth, probe_system, Bandwidth, Device, MemoryArchitecture, MemoryPressure,
    MemoryProbe, MemoryReport, ResourcePlan, ResourcePolicy,
};

pub const PROFILE_VERSION: u32 = 1;

/// Entry order of the result. Values are the index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Field {
    /// The caller's budget cut by every known host limit.
    BudgetBytes = 0,
    TotalBytes,
    AvailableBytes,
    CgroupLimitBytes,
    /// Usable CPUs, at most `CPU_THREAD_CEILING`.
    ThreadCeiling,
    /// The fastest cluster's logical CPUs, at most the ceiling.
    FastThreads,
    LogicalCpus,
    PhysicalCpus,
    CpuQuotaMillis,
    L1dBytes,
    L2PerCoreBytes,
    /// L3, only when the OS reports one (none on Apple silicon); never the
    /// L2 in its place.
    L3Bytes,
    CacheLineBytes,
    PageBytes,
    /// 1 unified, 2 discrete; unknown otherwise.
    Architecture,
    /// 1 normal, 2 warning, 3 critical; unknown otherwise.
    Pressure,
    /// Copy bandwidth on one thread, read plus write bytes per second.
    SingleBandwidth,
    /// Copy bandwidth on `BandwidthThreads` threads.
    MultiBandwidth,
    BandwidthThreads,
    /// Threads past which memory-bound work stops scaling, from the two
    /// bandwidth figures, at most the thread ceiling.
    MemoryBoundThreads,
}

impl Field {
    /// Every field, in wire order. `Field::ALL[i] as usize == i` is tested.
    pub const ALL: [Field; 20] = [
        Field::BudgetBytes,
        Field::TotalBytes,
        Field::AvailableBytes,
        Field::CgroupLimitBytes,
        Field::ThreadCeiling,
        Field::FastThreads,
        Field::LogicalCpus,
        Field::PhysicalCpus,
        Field::CpuQuotaMillis,
        Field::L1dBytes,
        Field::L2PerCoreBytes,
        Field::L3Bytes,
        Field::CacheLineBytes,
        Field::PageBytes,
        Field::Architecture,
        Field::Pressure,
        Field::SingleBandwidth,
        Field::MultiBandwidth,
        Field::BandwidthThreads,
        Field::MemoryBoundThreads,
    ];
}

pub const FIELD_COUNT: u32 = Field::ALL.len() as u32;
pub const ENTRY_BYTES: usize = 9;
/// Payload flag: measure (or reuse) copy bandwidth.
pub const FLAG_BANDWIDTH: u32 = 1;

/// The planned budget as sent. `u64::MAX` survives the plan only when no
/// limit was known and the caller set none: that is no bound, and is sent
/// as unknown (decoded as 0) rather than as a number a host could forward
/// to its memory ceiling.
fn budget_report(bytes: u64) -> MemoryReport {
    if bytes == u64::MAX {
        MemoryReport::Unknown
    } else {
        MemoryReport::Known(bytes)
    }
}

/// Empty plans against `u64::MAX`, which reports the host limits alone.
pub fn profile_request(input: &[u8]) -> Result<Vec<u8>, String> {
    let mut r = crate::wire::Reader::new(input);
    let (caller, flags) = match input.len() {
        0 => (u64::MAX, 0),
        8 => (r.u64()?, 0),
        12 => (r.u64()?, r.u32()?),
        n => {
            return Err(format!(
                "system profile: shape: payload must be empty, `budget: u64`, or \
                 `budget: u64, flags: u32`; got {n} bytes"
            ))
        }
    };
    if flags & !FLAG_BANDWIDTH != 0 {
        return Err(format!("system profile: unknown flags {flags:#x}"));
    }
    if caller == 0 {
        return Err("system profile: budget is 0 bytes".to_string());
    }
    let profile = probe_system();
    let mut plan = ResourcePlan::derive(&ResourcePolicy::new(caller), &profile, &[] as &[NoProbe]);
    // A refused measurement leaves the bandwidth fields unknown; the rest of
    // the profile still answers.
    let bandwidth = if flags & FLAG_BANDWIDTH != 0 {
        cached_bandwidth(&profile).ok()
    } else {
        None
    };
    if let Some(bw) = &bandwidth {
        plan = plan.with_bandwidth(bw);
    }
    let known = |n: u64| MemoryReport::Known(n);
    let bw_field = |f: fn(&Bandwidth) -> u64| {
        bandwidth
            .as_ref()
            .map_or(MemoryReport::Unknown, |b| known(f(b)))
    };
    let arch = match profile.architecture {
        MemoryArchitecture::Unified => MemoryReport::Known(1),
        MemoryArchitecture::Discrete => MemoryReport::Known(2),
        MemoryArchitecture::Unknown => MemoryReport::Unknown,
    };
    let pressure = match profile.pressure {
        MemoryPressure::Normal => MemoryReport::Known(1),
        MemoryPressure::Warning => MemoryReport::Known(2),
        MemoryPressure::Critical => MemoryReport::Known(3),
        MemoryPressure::Unknown => MemoryReport::Unknown,
    };
    let mut out = Vec::with_capacity(8 + Field::ALL.len() * ENTRY_BYTES);
    out.extend_from_slice(&PROFILE_VERSION.to_le_bytes());
    out.extend_from_slice(&FIELD_COUNT.to_le_bytes());
    for field in Field::ALL {
        let entry = match field {
            Field::BudgetBytes => budget_report(plan.budget_bytes),
            Field::TotalBytes => plan.total_bytes,
            Field::AvailableBytes => plan.available_bytes,
            Field::CgroupLimitBytes => plan.cgroup_limit_bytes,
            Field::ThreadCeiling => plan.thread_ceiling,
            Field::FastThreads => plan.fast_threads,
            Field::LogicalCpus => profile.cpu.logical,
            Field::PhysicalCpus => profile.cpu.physical,
            Field::CpuQuotaMillis => plan.cpu_quota_millis,
            Field::L1dBytes => plan.cache.l1d,
            Field::L2PerCoreBytes => plan.cache.l2_per_core,
            Field::L3Bytes => plan.cache.l3,
            Field::CacheLineBytes => profile.cpu.cache_line_bytes,
            Field::PageBytes => profile.cpu.page_bytes,
            Field::Architecture => arch,
            Field::Pressure => pressure,
            Field::SingleBandwidth => bw_field(|b| b.single_bytes_per_sec),
            Field::MultiBandwidth => bw_field(|b| b.multi_bytes_per_sec),
            Field::BandwidthThreads => bw_field(|b| b.threads as u64),
            Field::MemoryBoundThreads => plan.memory_bound_threads,
        };
        let (known, value) = match entry {
            MemoryReport::Known(v) => (1u8, v),
            MemoryReport::Unknown => (0u8, 0),
        };
        out.push(known);
        out.extend_from_slice(&value.to_le_bytes());
    }
    Ok(out)
}

/// The plan's probe list is empty here; this names its element type.
pub(crate) struct NoProbe;

impl MemoryProbe for NoProbe {
    fn kind(&self) -> Device {
        Device::Cpu
    }
    fn memory_bytes(&self) -> MemoryReport {
        MemoryReport::Unknown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(out: &[u8]) -> Vec<Option<u64>> {
        assert_eq!(
            u32::from_le_bytes(out[0..4].try_into().unwrap()),
            PROFILE_VERSION
        );
        let count = u32::from_le_bytes(out[4..8].try_into().unwrap()) as usize;
        assert_eq!(out.len(), 8 + count * ENTRY_BYTES);
        let (entries, rest) = out[8..].as_chunks::<ENTRY_BYTES>();
        assert!(rest.is_empty());
        entries
            .iter()
            .map(|e| {
                let v = u64::from_le_bytes(e[1..9].try_into().unwrap());
                match e[0] {
                    0 => {
                        assert_eq!(v, 0, "an unknown entry carries 0");
                        None
                    }
                    1 => Some(v),
                    k => panic!("known byte {k}"),
                }
            })
            .collect()
    }

    #[test]
    fn the_record_has_every_field_and_never_widens_the_budget() {
        let all = decode(&profile_request(&[]).unwrap());
        assert_eq!(all.len(), FIELD_COUNT as usize);
        assert!(all[Field::BudgetBytes as usize].is_some());
        for budget in [1u64, 4096, 1 << 30, u64::MAX] {
            let got = decode(&profile_request(&budget.to_le_bytes()).unwrap());
            let b = got[Field::BudgetBytes as usize].unwrap();
            assert!(b <= budget);
            if let Some(avail) = got[Field::AvailableBytes as usize] {
                assert!(b <= avail);
            }
            if let (Some(c), Some(f)) = (
                got[Field::ThreadCeiling as usize],
                got[Field::FastThreads as usize],
            ) {
                assert!(f <= c && c <= u64::from(ojas_core::CPU_THREAD_CEILING));
            }
        }
        for f in [
            Field::SingleBandwidth,
            Field::MultiBandwidth,
            Field::BandwidthThreads,
            Field::MemoryBoundThreads,
        ] {
            assert_eq!(all[f as usize], None, "{f:?} without the flag");
        }
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        assert_eq!(all[Field::Architecture as usize], Some(1));
    }

    #[test]
    fn the_bandwidth_flag_fills_its_fields_within_the_ceiling() {
        let mut payload = (1u64 << 30).to_le_bytes().to_vec();
        payload.extend_from_slice(&FLAG_BANDWIDTH.to_le_bytes());
        let got = decode(&profile_request(&payload).unwrap());
        let single = got[Field::SingleBandwidth as usize].expect("single");
        let multi = got[Field::MultiBandwidth as usize].expect("multi");
        let threads = got[Field::BandwidthThreads as usize].expect("threads");
        let bound = got[Field::MemoryBoundThreads as usize].expect("memory-bound threads");
        assert!(single > 0 && multi > 0 && threads >= 1);
        assert!(bound >= 1 && bound <= threads);
        if let Some(c) = got[Field::ThreadCeiling as usize] {
            assert!(bound <= c);
        }
        // A second call reuses the cached figure.
        let again = decode(&profile_request(&payload).unwrap());
        assert_eq!(again[Field::SingleBandwidth as usize], Some(single));
    }

    /// On a host where nothing could be read (no `/proc`, an unknown OS),
    /// an empty request used to send `BudgetBytes` known at `u64::MAX`.
    #[test]
    fn an_unbounded_budget_is_sent_as_unknown() {
        assert_eq!(budget_report(u64::MAX), MemoryReport::Unknown);
        assert_eq!(
            budget_report(u64::MAX - 1),
            MemoryReport::Known(u64::MAX - 1)
        );
        assert_eq!(budget_report(0), MemoryReport::Known(0));
        let blind = ojas_device::SystemProfile::from_memory(ojas_device::HostMemory::all_unknown());
        let plan = ResourcePlan::derive(&ResourcePolicy::new(u64::MAX), &blind, &[] as &[NoProbe]);
        assert_eq!(budget_report(plan.budget_bytes), MemoryReport::Unknown);
    }

    #[test]
    fn field_order_is_the_wire_order() {
        for (i, f) in Field::ALL.iter().enumerate() {
            assert_eq!(*f as usize, i, "{f:?}");
        }
        assert_eq!(FIELD_COUNT, 20, "the Go decoder reads 20 fields");
    }

    #[test]
    fn bad_payloads_are_refused() {
        let mut flagged = 1u64.to_le_bytes().to_vec();
        flagged.extend_from_slice(&2u32.to_le_bytes());
        assert!(profile_request(&flagged)
            .unwrap_err()
            .contains("unknown flags"));
        flagged[8..].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(profile_request(&flagged)
            .unwrap_err()
            .contains("unknown flags"));
        assert!(profile_request(&0u64.to_le_bytes())
            .unwrap_err()
            .contains("0 bytes"));
        for n in [1usize, 7, 9, 16] {
            let err = profile_request(&vec![1u8; n]).unwrap_err();
            assert!(err.contains("shape"), "{err}");
        }
    }
}

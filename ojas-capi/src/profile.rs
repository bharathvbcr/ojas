//! `SYSTEM_PROFILE`: what this machine and process can use, for a host
//! that sizes `threads` and the memory ceiling from it.
//!
//! The op reads; it changes nothing. Without a device flag it opens no GPU
//! and the device fields are unknown.
//!
//! Payload: empty; `budget: u64`; or `budget: u64, flags: u32`. Flag
//! [`FLAG_BANDWIDTH`] runs [`ojas_device::cached_bandwidth`] (about half a
//! second the first time in a process, a cached figure after that) and
//! fills the bandwidth fields. Flag [`FLAG_DEVICE_METAL`] or
//! [`FLAG_DEVICE_WGPU`] (at most one) opens that device on a short-lived
//! thread, as a load does, with the cancel check polled while it opens,
//! reads its memory probe, closes it, and plans against the probe exactly
//! as a load on that device plans its session budget
//! ([`crate::model::device_plan`]): the device fields then say which probe
//! was used, what it reported and the room it left. A device that does not
//! open is its error (`metal:` or `wgpu:`). Any other flag bit is refused.
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
    cached_bandwidth, probe_system, Bandwidth, MemoryArchitecture, MemoryPressure, MemoryProbe,
    MemoryReport, ResourcePlan, ResourcePolicy,
};

use crate::gate::Check;
use crate::model::{self, DeviceProbe};

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
    /// The device whose probe the plan used: the load's device number,
    /// [`crate::load::DEVICE_METAL`] or [`crate::load::DEVICE_WGPU`].
    /// Unknown, as is every field through `DeviceBudgetBytes`, without a
    /// device flag.
    ProbeDevice,
    /// [`MemoryProbe::memory_bytes`]: Metal's recommended working set;
    /// never known on wgpu, which reports no memory size.
    DeviceMemoryBytes,
    /// [`MemoryProbe::resident_bytes`]: what the process holds there.
    DeviceResidentBytes,
    /// [`MemoryProbe::pool_cache_bytes`]: the uncharged pool cap.
    DevicePoolCacheBytes,
    /// The plan's room for the device.
    DeviceRoomBytes,
    /// 1 when the device draws on host memory, 0 when it has its own.
    DeviceSharesHost,
    /// 1 when the plan puts the device on the host's budget.
    SharedBudget,
    /// The session budget a load on this device with this budget gets.
    DeviceBudgetBytes,
    /// [`ojas_wgpu::DropStats::parked`]: freed wgpu contexts still
    /// holding their devices. Always known.
    WgpuDropsParked,
    /// [`ojas_wgpu::DropStats::timed_out`]. Always known.
    WgpuDropsTimedOut,
}

impl Field {
    /// Every field, in wire order. `Field::ALL[i] as usize == i` is tested.
    pub const ALL: [Field; 30] = [
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
        Field::ProbeDevice,
        Field::DeviceMemoryBytes,
        Field::DeviceResidentBytes,
        Field::DevicePoolCacheBytes,
        Field::DeviceRoomBytes,
        Field::DeviceSharesHost,
        Field::SharedBudget,
        Field::DeviceBudgetBytes,
        Field::WgpuDropsParked,
        Field::WgpuDropsTimedOut,
    ];
}

pub const FIELD_COUNT: u32 = Field::ALL.len() as u32;
pub const ENTRY_BYTES: usize = 9;
/// Payload flag: measure (or reuse) copy bandwidth.
pub const FLAG_BANDWIDTH: u32 = 1;
/// Payload flag: open the Metal device and plan against its probe.
pub const FLAG_DEVICE_METAL: u32 = 2;
/// Payload flag: open a wgpu device and plan against its probe.
pub const FLAG_DEVICE_WGPU: u32 = 4;
const KNOWN_FLAGS: u32 = FLAG_BANDWIDTH | FLAG_DEVICE_METAL | FLAG_DEVICE_WGPU;

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

/// The probe of the device `flags` names, opened as a load opens it (with
/// a budget of `caller`, which sets Metal's pool cap) and closed again.
fn probe_device(
    flags: u32,
    caller: u64,
    check: Check,
) -> Result<Option<(DeviceProbe, u32)>, String> {
    let budget = || ojas_core::Budget::new(caller);
    match flags & (FLAG_DEVICE_METAL | FLAG_DEVICE_WGPU) {
        0 => Ok(None),
        FLAG_DEVICE_METAL => {
            #[cfg(target_os = "macos")]
            {
                let opened = model::Opened::Metal(crate::owner::open_metal(budget(), check)?);
                Ok(model::device_probe(&opened)?.map(|p| (p, crate::load::DEVICE_METAL)))
            }
            #[cfg(not(target_os = "macos"))]
            {
                let _ = budget;
                crate::owner::open_metal(check).and(Err("metal: Metal requires macOS".to_string()))
            }
        }
        FLAG_DEVICE_WGPU => {
            let opened = model::Opened::Wgpu(std::sync::Arc::new(crate::owner::open_wgpu(
                budget(),
                check,
            )?));
            Ok(model::device_probe(&opened)?.map(|p| (p, crate::load::DEVICE_WGPU)))
        }
        _ => Err("system profile: at most one device flag".to_string()),
    }
}

/// Empty plans against `u64::MAX`, which reports the host limits alone.
pub fn profile_request(input: &[u8], check: Check) -> Result<Vec<u8>, String> {
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
    if flags & !KNOWN_FLAGS != 0 {
        return Err(format!("system profile: unknown flags {flags:#x}"));
    }
    if caller == 0 {
        return Err("system profile: budget is 0 bytes".to_string());
    }
    let device = probe_device(flags, caller, check)?;
    let profile = probe_system();
    let mut plan = match &device {
        Some((probe, _)) => model::device_plan(caller, probe, &profile),
        None => ResourcePlan::derive(
            &ResourcePolicy::new(caller),
            &profile,
            &[] as &[DeviceProbe],
        ),
    };
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
    let flag = |on: bool| MemoryReport::Known(u64::from(on));
    let on_device = |f: &dyn Fn(&DeviceProbe, u32) -> MemoryReport| {
        device
            .as_ref()
            .map_or(MemoryReport::Unknown, |(probe, number)| f(probe, *number))
    };
    let drops = ojas_wgpu::drop_stats();
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
            // The plan lists the probed device first.
            Field::ProbeDevice => on_device(&|_, n| known(u64::from(n))),
            Field::DeviceMemoryBytes => on_device(&|_, _| plan.device_memory[0]),
            Field::DeviceResidentBytes => on_device(&|p, _| p.resident_bytes()),
            Field::DevicePoolCacheBytes => on_device(&|_, _| plan.device_pool_cache[0]),
            Field::DeviceRoomBytes => on_device(&|_, _| plan.device_room[0]),
            Field::DeviceSharesHost => on_device(&|_, _| flag(plan.device_shares_host[0])),
            Field::SharedBudget => on_device(&|_, _| flag(plan.shared_budget)),
            Field::DeviceBudgetBytes => on_device(&|p, _| {
                plan.device_budget(p.kind())
                    .map_or(MemoryReport::Unknown, budget_report)
            }),
            Field::WgpuDropsParked => known(drops.parked as u64),
            Field::WgpuDropsTimedOut => known(drops.timed_out),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::never;

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
        let all = decode(&profile_request(&[], never()).unwrap());
        assert_eq!(all.len(), FIELD_COUNT as usize);
        assert!(all[Field::BudgetBytes as usize].is_some());
        for budget in [1u64, 4096, 1 << 30, u64::MAX] {
            let got = decode(&profile_request(&budget.to_le_bytes(), never()).unwrap());
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
            Field::ProbeDevice,
            Field::DeviceMemoryBytes,
            Field::DeviceResidentBytes,
            Field::DevicePoolCacheBytes,
            Field::DeviceRoomBytes,
            Field::DeviceSharesHost,
            Field::SharedBudget,
            Field::DeviceBudgetBytes,
        ] {
            assert_eq!(all[f as usize], None, "{f:?} without the flag");
        }
        assert!(all[Field::WgpuDropsParked as usize].is_some());
        assert!(all[Field::WgpuDropsTimedOut as usize].is_some());
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        assert_eq!(all[Field::Architecture as usize], Some(1));
    }

    #[test]
    fn the_bandwidth_flag_fills_its_fields_within_the_ceiling() {
        let mut payload = (1u64 << 30).to_le_bytes().to_vec();
        payload.extend_from_slice(&FLAG_BANDWIDTH.to_le_bytes());
        let got = decode(&profile_request(&payload, never()).unwrap());
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
        let again = decode(&profile_request(&payload, never()).unwrap());
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
        let plan = ResourcePlan::derive(
            &ResourcePolicy::new(u64::MAX),
            &blind,
            &[] as &[DeviceProbe],
        );
        assert_eq!(budget_report(plan.budget_bytes), MemoryReport::Unknown);
    }

    #[test]
    fn field_order_is_the_wire_order() {
        for (i, f) in Field::ALL.iter().enumerate() {
            assert_eq!(*f as usize, i, "{f:?}");
        }
        assert_eq!(FIELD_COUNT, 30, "the Go decoder reads 30 fields");
    }

    fn device_profile(budget: u64, flag: u32) -> Result<Vec<Option<u64>>, String> {
        let mut payload = budget.to_le_bytes().to_vec();
        payload.extend_from_slice(&flag.to_le_bytes());
        profile_request(&payload, never()).map(|out| decode(&out))
    }

    /// The Metal flag plans against the device's own probe, the path a
    /// load takes: on Apple silicon the device shares host memory, so the
    /// plan takes the shared budget path and the room is at most the host
    /// budget; the session budget is the caller's cut to that room.
    #[cfg(target_os = "macos")]
    #[test]
    fn the_metal_flag_plans_on_the_shared_budget_with_the_device_room() {
        let got = match device_profile(1 << 30, FLAG_DEVICE_METAL) {
            Ok(got) => got,
            Err(err) => return crate::tests::skip_or_fail("Metal device profile", &err),
        };
        let field = |f: Field| got[f as usize];
        assert_eq!(
            field(Field::ProbeDevice),
            Some(u64::from(crate::load::DEVICE_METAL))
        );
        let memory = field(Field::DeviceMemoryBytes).expect("the working set");
        let room = field(Field::DeviceRoomBytes).expect("a reporting device has a room");
        let pool = field(Field::DevicePoolCacheBytes).expect("the pool cap");
        assert_eq!(pool, 1 << 28, "a quarter of the 1 GiB budget");
        assert!(
            room <= memory - pool,
            "room {room} of {memory} with pool {pool}"
        );
        let budget = field(Field::DeviceBudgetBytes).expect("the session budget");
        assert!(budget <= 1 << 30 && budget <= room);
        #[cfg(target_arch = "aarch64")]
        {
            assert_eq!(field(Field::DeviceSharesHost), Some(1));
            assert_eq!(field(Field::SharedBudget), Some(1));
            assert!(room <= field(Field::BudgetBytes).unwrap());
        }
    }

    /// The wgpu flag records that wgpu reports no memory size: the room is
    /// unknown, not a default, and the session budget is the caller's cut
    /// only by what is known.
    #[test]
    fn the_wgpu_flag_records_an_unknown_room() {
        let got = match device_profile(1 << 30, FLAG_DEVICE_WGPU) {
            Ok(got) => got,
            Err(err) if err.starts_with("wgpu:") => {
                eprintln!("SKIP: no wgpu adapter: {err}");
                return;
            }
            Err(err) => panic!("{err}"),
        };
        assert_eq!(
            got[Field::ProbeDevice as usize],
            Some(u64::from(crate::load::DEVICE_WGPU))
        );
        assert_eq!(got[Field::DeviceMemoryBytes as usize], None);
        assert_eq!(got[Field::DeviceRoomBytes as usize], None);
        assert_eq!(
            got[Field::DevicePoolCacheBytes as usize],
            Some(ojas_wgpu::POOL_CAP_BYTES)
        );
        let budget = got[Field::DeviceBudgetBytes as usize].expect("the session budget");
        assert!(budget <= 1 << 30);
        assert!(got[Field::DeviceSharesHost as usize].is_some());
    }

    #[test]
    fn bad_payloads_are_refused() {
        let mut flagged = 1u64.to_le_bytes().to_vec();
        flagged.extend_from_slice(&8u32.to_le_bytes());
        assert!(profile_request(&flagged, never())
            .unwrap_err()
            .contains("unknown flags"));
        flagged[8..].copy_from_slice(&(FLAG_DEVICE_METAL | FLAG_DEVICE_WGPU).to_le_bytes());
        assert!(profile_request(&flagged, never())
            .unwrap_err()
            .contains("at most one device flag"));
        flagged[8..].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(profile_request(&flagged, never())
            .unwrap_err()
            .contains("unknown flags"));
        assert!(profile_request(&0u64.to_le_bytes(), never())
            .unwrap_err()
            .contains("0 bytes"));
        for n in [1usize, 7, 9, 16] {
            let err = profile_request(&vec![1u8; n], never()).unwrap_err();
            assert!(err.contains("shape"), "{err}");
        }
    }
}

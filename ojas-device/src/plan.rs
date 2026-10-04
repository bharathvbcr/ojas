//! Turn a caller policy and a system profile into a resource plan.
//!
//! The host probe never chooses the split factor. `allow_split` is copied
//! from the policy, where it defaults to false. Nothing in the plan widens
//! what the caller asked for: every figure is the caller's number cut by a
//! known limit, or a recommendation the caller adopts explicitly.

use crate::bandwidth::Bandwidth;
use crate::host::MemoryReport;
use crate::system::{MemoryArchitecture, MemoryPressure, SystemProfile};
use crate::tuning::CacheBudget;
use crate::Device;
use ojas_core::CPU_THREAD_CEILING;

/// What the caller will allow. The probe does not widen this.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourcePolicy {
    pub caller_budget_bytes: u64,
    /// Defaults to `false` from [`ResourcePolicy::new`].
    pub allow_split: bool,
    /// Ordered devices the caller accepts. The plan does not insert `Cpu`
    /// when a listed device has no memory number.
    pub devices: Vec<Device>,
}

impl ResourcePolicy {
    pub fn new(caller_budget_bytes: u64) -> Self {
        Self {
            caller_budget_bytes,
            allow_split: false,
            devices: vec![Device::Cpu],
        }
    }
}

/// Device memory, as reported by the runtime that owns that device.
///
/// A failed probe returns [`MemoryReport::Unknown`].
pub trait MemoryProbe {
    fn kind(&self) -> Device;
    /// Memory the device may use: total device memory, or on unified
    /// memory the recommended working set.
    fn memory_bytes(&self) -> MemoryReport;
    /// Bytes this process already holds on the device.
    fn resident_bytes(&self) -> MemoryReport {
        MemoryReport::Unknown
    }
    /// Whether the device's memory is the host's. `Unknown` defers to the
    /// host profile.
    fn architecture(&self) -> MemoryArchitecture {
        MemoryArchitecture::Unknown
    }
}

/// The numbers a caller can act on. Thread counts are advice, not spawn caps.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourcePlan {
    /// The caller budget cut by every known host and cgroup limit, and 0
    /// under [`MemoryPressure::Critical`]. `u64::MAX` only when the caller
    /// asked for that and no limit was known: it is then no bound at all.
    pub budget_bytes: u64,
    /// Usable CPUs, capped at [`CPU_THREAD_CEILING`]: the thread count for
    /// compute-bound work.
    pub thread_ceiling: MemoryReport,
    /// Threads that land on the fastest cores alone: the fastest cluster's
    /// logical count, at most `thread_ceiling`. Equal to it on a machine
    /// with one kind of core.
    pub fast_threads: MemoryReport,
    /// Threads past which a memory-bound op stops getting faster. Unknown
    /// until [`ResourcePlan::with_bandwidth`] supplies a measurement.
    pub memory_bound_threads: MemoryReport,
    pub allow_split: bool,
    pub devices: Vec<Device>,
    /// Parallel to [`ResourcePlan::devices`]. Missing probes stay unknown.
    pub device_memory: Vec<MemoryReport>,
    /// Parallel to `devices`: bytes the device can still take, its memory
    /// less what this process already holds there. For a device that shares
    /// host memory a known room is also at most `budget_bytes`, so a device
    /// figure never invites spending RAM the host budget already counts. A
    /// device that reported no memory has an `Unknown` room even when it
    /// shares host memory; `shared_budget` still bounds it. `Cpu`'s room is
    /// `budget_bytes`.
    pub device_room: Vec<MemoryReport>,
    /// Parallel to `devices`: whether that device draws from host memory.
    /// `Cpu` always does.
    pub device_shares_host: Vec<bool>,
    /// True when a listed GPU shares host memory. The caller should then
    /// charge the CPU backend and that GPU backend to one
    /// [`ojas_core::Budget`] of `budget_bytes`, not one budget each: two
    /// budgets that each fit would add past the memory they share.
    pub shared_budget: bool,
    pub architecture: MemoryArchitecture,
    pub pressure: MemoryPressure,
    pub total_bytes: MemoryReport,
    pub available_bytes: MemoryReport,
    pub cgroup_limit_bytes: MemoryReport,
    pub cpu_quota_millis: MemoryReport,
    /// Cache sizes every core can rely on, for blocked kernels
    /// ([`crate::GemmBlocks::derive`]).
    pub cache: CacheBudget,
}

impl ResourcePlan {
    /// Clamp the caller budget by known host limits. Unknown inputs are
    /// copied through and are not replaced with a number.
    pub fn derive(
        policy: &ResourcePolicy,
        profile: &SystemProfile,
        probes: &[impl MemoryProbe],
    ) -> Self {
        let host = &profile.memory;
        let mut budget = policy.caller_budget_bytes;
        budget = tighten(budget, host.total_bytes);
        budget = tighten(budget, host.available_bytes);
        budget = tighten(budget, host.cgroup_limit_bytes);
        if let (MemoryReport::Known(limit), MemoryReport::Known(current)) =
            (host.cgroup_limit_bytes, host.cgroup_current_bytes)
        {
            budget = budget.min(limit.saturating_sub(current));
        }
        // At critical pressure the kernel is already reclaiming and killing
        // to stay up; a run admitted now would be among the next killed.
        // Nothing is admitted. Warning is reported, not acted on: it is
        // routine on a busy machine and has no figure to cut to.
        if profile.pressure == MemoryPressure::Critical {
            budget = 0;
        }
        let thread_ceiling = thread_ceiling(host.cpu_count, profile.cpu.cpu_quota_millis);
        let mut device_memory = Vec::with_capacity(policy.devices.len());
        let mut device_room = Vec::with_capacity(policy.devices.len());
        let mut device_shares_host = Vec::with_capacity(policy.devices.len());
        let mut shared_budget = false;
        for device in &policy.devices {
            let probe = probes.iter().find(|probe| probe.kind() == *device);
            let memory = probe.map_or(MemoryReport::Unknown, |p| p.memory_bytes());
            let resident = probe.map_or(MemoryReport::Unknown, |p| p.resident_bytes());
            let shares = match device {
                Device::Cpu => true,
                _ => match probe.map_or(MemoryArchitecture::Unknown, |p| p.architecture()) {
                    MemoryArchitecture::Unified => true,
                    MemoryArchitecture::Discrete => false,
                    MemoryArchitecture::Unknown => {
                        profile.architecture == MemoryArchitecture::Unified
                    }
                },
            };
            let mut room = match (memory, resident) {
                (MemoryReport::Known(m), MemoryReport::Known(r)) => {
                    MemoryReport::Known(m.saturating_sub(r))
                }
                (known, _) => known,
            };
            if *device == Device::Cpu {
                // The host budget is the CPU's memory; nothing is assumed.
                room = MemoryReport::Known(budget);
            } else if shares {
                // A known room is cut to the shared budget. An unknown one
                // stays unknown: the budget is a ceiling on it, not a
                // reading of it, and `shared_budget` already says so.
                if let MemoryReport::Known(r) = room {
                    room = MemoryReport::Known(r.min(budget));
                }
                shared_budget = true;
            }
            device_memory.push(memory);
            device_room.push(room);
            device_shares_host.push(shares);
        }
        Self {
            budget_bytes: budget,
            thread_ceiling,
            fast_threads: fast_threads(profile, thread_ceiling),
            memory_bound_threads: MemoryReport::Unknown,
            allow_split: policy.allow_split,
            devices: policy.devices.clone(),
            device_memory,
            device_room,
            device_shares_host,
            shared_budget,
            architecture: profile.architecture,
            pressure: profile.pressure,
            total_bytes: host.total_bytes,
            available_bytes: host.available_bytes,
            cgroup_limit_bytes: host.cgroup_limit_bytes,
            cpu_quota_millis: profile.cpu.cpu_quota_millis,
            cache: CacheBudget::from_topology(&profile.cpu),
        }
    }

    /// Fill `memory_bound_threads` from a measurement: the threads that
    /// reached the measured multi-thread bandwidth, at most
    /// `thread_ceiling` when that is known.
    pub fn with_bandwidth(mut self, bandwidth: &Bandwidth) -> Self {
        let n = bandwidth.saturating_threads() as u64;
        self.memory_bound_threads = MemoryReport::Known(match self.thread_ceiling {
            MemoryReport::Known(c) => n.max(1).min(c),
            MemoryReport::Unknown => n.clamp(1, u64::from(CPU_THREAD_CEILING)),
        });
        self
    }
}

fn tighten(budget: u64, report: MemoryReport) -> u64 {
    match report {
        MemoryReport::Known(n) => budget.min(n),
        MemoryReport::Unknown => budget,
    }
}

/// The usable CPU count, cut to the cgroup CPU quota (whole CPUs, rounded
/// down, at least one) and to [`CPU_THREAD_CEILING`]. std's
/// `available_parallelism` already reads the quota where it can, and rounds
/// down: a 1.5-CPU container reported 1 in the Linux run. This cut holds
/// when std could not read the quota (a sandbox that hides cgroupfs from std
/// but not from the probe), and rounds the same way so both paths agree;
/// rounding up would also let a pool outrun its quota and be throttled.
fn thread_ceiling(cpu_count: MemoryReport, quota_millis: MemoryReport) -> MemoryReport {
    let MemoryReport::Known(n) = cpu_count else {
        return MemoryReport::Unknown;
    };
    let mut n = n.min(u64::from(CPU_THREAD_CEILING));
    if let MemoryReport::Known(quota) = quota_millis {
        n = n.min((quota / 1000).max(1));
    }
    MemoryReport::Known(n)
}

/// The fastest cluster's logical count when the machine has more than one
/// kind of core, else the ceiling. Never above the ceiling.
fn fast_threads(profile: &SystemProfile, ceiling: MemoryReport) -> MemoryReport {
    let MemoryReport::Known(ceiling) = ceiling else {
        return MemoryReport::Unknown;
    };
    if profile.cpu.clusters.len() < 2 {
        return MemoryReport::Known(ceiling);
    }
    match profile.cpu.clusters[0].logical {
        MemoryReport::Known(n) if n > 0 => MemoryReport::Known(n.min(ceiling)),
        _ => MemoryReport::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::HostMemory;
    use crate::topology::{CoreCluster, CpuTopology};

    /// A GPU probe that reports residency and its memory architecture.
    struct Gpu {
        kind: Device,
        memory: MemoryReport,
        resident: MemoryReport,
        arch: MemoryArchitecture,
    }

    impl MemoryProbe for Gpu {
        fn kind(&self) -> Device {
            self.kind
        }
        fn memory_bytes(&self) -> MemoryReport {
            self.memory
        }
        fn resident_bytes(&self) -> MemoryReport {
            self.resident
        }
        fn architecture(&self) -> MemoryArchitecture {
            self.arch
        }
    }

    fn reports() -> [MemoryReport; 6] {
        [
            MemoryReport::Unknown,
            MemoryReport::Known(0),
            MemoryReport::Known(1),
            MemoryReport::Known(700),
            MemoryReport::Known(5_000),
            MemoryReport::Known(u64::MAX),
        ]
    }

    #[test]
    fn unified_device_room_never_exceeds_the_shared_budget() {
        let archs = [
            MemoryArchitecture::Unified,
            MemoryArchitecture::Discrete,
            MemoryArchitecture::Unknown,
        ];
        for host_arch in archs {
            for gpu_arch in archs {
                for memory in reports() {
                    for resident in reports() {
                        for available in reports() {
                            for caller in [0u64, 1, 1_000, u64::MAX] {
                                let mut profile = SystemProfile::from_memory(HostMemory {
                                    available_bytes: available,
                                    ..HostMemory::all_unknown()
                                });
                                profile.architecture = host_arch;
                                let mut policy = ResourcePolicy::new(caller);
                                policy.devices = vec![Device::Metal, Device::Cpu];
                                let probes = [Gpu {
                                    kind: Device::Metal,
                                    memory,
                                    resident,
                                    arch: gpu_arch,
                                }];
                                let plan = ResourcePlan::derive(&policy, &profile, &probes);
                                let shares = gpu_arch == MemoryArchitecture::Unified
                                    || (gpu_arch == MemoryArchitecture::Unknown
                                        && host_arch == MemoryArchitecture::Unified);
                                assert_eq!(plan.device_shares_host, vec![shares, true]);
                                assert_eq!(plan.shared_budget, shares);
                                assert_eq!(plan.device_memory, vec![memory, MemoryReport::Unknown]);
                                assert_eq!(
                                    plan.device_room[1],
                                    MemoryReport::Known(plan.budget_bytes),
                                    "the CPU's room is the budget"
                                );
                                match plan.device_room[0] {
                                    MemoryReport::Known(room) => {
                                        assert_ne!(
                                            memory,
                                            MemoryReport::Unknown,
                                            "a room was invented for an unreported device"
                                        );
                                        if shares {
                                            assert!(room <= plan.budget_bytes, "{plan:?}");
                                        }
                                        if let MemoryReport::Known(m) = memory {
                                            assert!(room <= m, "room past the device's memory");
                                            if let MemoryReport::Known(r) = resident {
                                                assert!(room <= m.saturating_sub(r));
                                            }
                                        }
                                    }
                                    MemoryReport::Unknown => {
                                        // Only an unreported device has no room;
                                        // the budget is never substituted for it.
                                        assert_eq!(memory, MemoryReport::Unknown);
                                    }
                                }
                                assert!(plan.budget_bytes <= caller);
                                if let MemoryReport::Known(a) = available {
                                    assert!(plan.budget_bytes <= a);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn the_cpu_alone_never_sets_shared_budget() {
        let mut profile = SystemProfile::from_memory(HostMemory::all_unknown());
        profile.architecture = MemoryArchitecture::Unified;
        let plan = ResourcePlan::derive(&ResourcePolicy::new(10), &profile, &[] as &[Gpu]);
        assert!(!plan.shared_budget);
        assert_eq!(plan.device_shares_host, vec![true]);
        assert_eq!(plan.device_room, vec![MemoryReport::Known(10)]);
    }

    fn clustered(levels: &[MemoryReport], usable: MemoryReport) -> SystemProfile {
        let mut p = SystemProfile::from_memory(HostMemory {
            cpu_count: usable,
            ..HostMemory::all_unknown()
        });
        p.cpu = CpuTopology {
            usable,
            clusters: levels
                .iter()
                .map(|&logical| CoreCluster {
                    name: "c".into(),
                    physical: logical,
                    logical,
                    l1d_bytes: MemoryReport::Unknown,
                    l2_bytes: MemoryReport::Unknown,
                    cpus_per_l2: MemoryReport::Unknown,
                })
                .collect(),
            ..CpuTopology::all_unknown()
        };
        p
    }

    #[test]
    fn thread_advice_never_exceeds_the_ceiling() {
        let usables = [
            MemoryReport::Unknown,
            MemoryReport::Known(1),
            MemoryReport::Known(6),
            MemoryReport::Known(18),
            MemoryReport::Known(u64::MAX),
        ];
        let levels: [&[MemoryReport]; 5] = [
            &[],
            &[MemoryReport::Known(18)],
            &[MemoryReport::Known(6), MemoryReport::Known(12)],
            &[MemoryReport::Unknown, MemoryReport::Known(12)],
            &[MemoryReport::Known(0), MemoryReport::Known(4)],
        ];
        for usable in usables {
            for lv in levels {
                let plan = ResourcePlan::derive(
                    &ResourcePolicy::new(1),
                    &clustered(lv, usable),
                    &[] as &[Gpu],
                );
                assert_eq!(plan.memory_bound_threads, MemoryReport::Unknown);
                match (plan.thread_ceiling, plan.fast_threads) {
                    (MemoryReport::Unknown, fast) => assert_eq!(fast, MemoryReport::Unknown),
                    (MemoryReport::Known(c), MemoryReport::Known(f)) => {
                        assert!(c <= u64::from(CPU_THREAD_CEILING));
                        assert!(f >= 1 && f <= c, "fast {f} ceiling {c}");
                        if lv.len() < 2 {
                            assert_eq!(f, c);
                        }
                    }
                    (MemoryReport::Known(_), MemoryReport::Unknown) => {
                        assert!(lv.len() >= 2, "only an unreadable fast cluster is unknown");
                    }
                }
            }
        }
        let plan = ResourcePlan::derive(
            &ResourcePolicy::new(1),
            &clustered(
                &[MemoryReport::Known(6), MemoryReport::Known(12)],
                MemoryReport::Known(18),
            ),
            &[] as &[Gpu],
        );
        assert_eq!(plan.fast_threads, MemoryReport::Known(6));
    }

    #[test]
    fn bandwidth_fills_memory_bound_threads_within_the_ceiling() {
        let bw = |single: u64, multi: u64, threads: usize| Bandwidth {
            single_bytes_per_sec: single,
            multi_bytes_per_sec: multi,
            threads,
            buffer_bytes: 1 << 20,
            reps_run: (1, 1),
            load_avg: None,
        };
        let base = |usable| {
            ResourcePlan::derive(
                &ResourcePolicy::new(1),
                &clustered(&[], usable),
                &[] as &[Gpu],
            )
        };
        let plan = base(MemoryReport::Known(18)).with_bandwidth(&bw(100, 450, 18));
        assert_eq!(plan.memory_bound_threads, MemoryReport::Known(5));
        let plan = base(MemoryReport::Known(4)).with_bandwidth(&bw(1, u64::MAX, 1024));
        assert_eq!(plan.memory_bound_threads, MemoryReport::Known(4));
        let plan = base(MemoryReport::Unknown).with_bandwidth(&bw(1, u64::MAX, usize::MAX));
        assert_eq!(
            plan.memory_bound_threads,
            MemoryReport::Known(u64::from(CPU_THREAD_CEILING))
        );
        let plan = base(MemoryReport::Known(0)).with_bandwidth(&bw(1, 9, 9));
        assert_eq!(
            plan.memory_bound_threads,
            MemoryReport::Known(0),
            "a 0 ceiling holds"
        );
    }

    #[test]
    fn the_live_profile_plans_without_widening_the_caller() {
        let profile = crate::probe_system();
        for caller in [0u64, 1, 1 << 30, u64::MAX] {
            let plan = ResourcePlan::derive(&ResourcePolicy::new(caller), &profile, &[] as &[Gpu]);
            assert!(plan.budget_bytes <= caller);
            if let MemoryReport::Known(a) = profile.memory.available_bytes {
                assert!(plan.budget_bytes <= a);
            }
        }
    }

    struct Stub {
        kind: Device,
        memory: MemoryReport,
    }

    impl MemoryProbe for Stub {
        fn kind(&self) -> Device {
            self.kind
        }
        fn memory_bytes(&self) -> MemoryReport {
            self.memory
        }
    }

    fn host(
        total: MemoryReport,
        available: MemoryReport,
        limit: MemoryReport,
        current: MemoryReport,
        cpu: MemoryReport,
    ) -> HostMemory {
        HostMemory {
            total_bytes: total,
            available_bytes: available,
            cgroup_limit_bytes: limit,
            cgroup_current_bytes: current,
            cpu_count: cpu,
        }
    }

    #[test]
    fn allow_split_defaults_false_and_is_not_inferred() {
        let policy = ResourcePolicy::new(1_000);
        assert!(!policy.allow_split);
        let tiny = host(
            MemoryReport::Known(64),
            MemoryReport::Known(32),
            MemoryReport::Known(16),
            MemoryReport::Known(8),
            MemoryReport::Known(1),
        );
        let plan = ResourcePlan::derive(&policy, &SystemProfile::from_memory(tiny), &[] as &[Stub]);
        assert!(!plan.allow_split);
        let mut split = policy.clone();
        split.allow_split = true;
        assert!(
            ResourcePlan::derive(&split, &SystemProfile::from_memory(tiny), &[] as &[Stub])
                .allow_split
        );
    }

    #[test]
    fn unknown_stays_unknown_and_does_not_shrink_the_caller_budget() {
        let policy = ResourcePolicy::new(1234);
        let plan = ResourcePlan::derive(
            &policy,
            &SystemProfile::from_memory(HostMemory::all_unknown()),
            &[] as &[Stub],
        );
        assert_eq!(plan.budget_bytes, 1234);
        assert_eq!(plan.thread_ceiling, MemoryReport::Unknown);
        assert_eq!(plan.total_bytes, MemoryReport::Unknown);
        assert_eq!(plan.available_bytes, MemoryReport::Unknown);
        assert_eq!(plan.cgroup_limit_bytes, MemoryReport::Unknown);
        assert_eq!(plan.devices, vec![Device::Cpu]);
        assert_eq!(plan.device_memory, vec![MemoryReport::Unknown]);
    }

    #[test]
    fn known_cgroup_limit_is_never_exceeded() {
        let callers = [0u64, 1, 50, 100, 1_000, u64::MAX];
        let limits = [0u64, 1, 50, 100, 500];
        let currents = [0u64, 1, 40, 100, 150];
        for caller in callers {
            for limit in limits {
                for current in currents {
                    let host = host(
                        MemoryReport::Unknown,
                        MemoryReport::Unknown,
                        MemoryReport::Known(limit),
                        MemoryReport::Known(current),
                        MemoryReport::Unknown,
                    );
                    let plan = ResourcePlan::derive(
                        &ResourcePolicy::new(caller),
                        &SystemProfile::from_memory(host),
                        &[] as &[Stub],
                    );
                    assert!(plan.budget_bytes <= limit, "{plan:?}");
                    assert!(plan.budget_bytes <= caller);
                    assert!(plan.budget_bytes <= limit.saturating_sub(current));
                    assert_eq!(plan.cgroup_limit_bytes, MemoryReport::Known(limit));
                    assert_eq!(plan.thread_ceiling, MemoryReport::Unknown);
                }
            }
        }
    }

    #[test]
    fn outputs_are_monotone_in_known_inputs() {
        let budgets = [0u64, 1, 16, 128, 10_000];
        let caps = [
            MemoryReport::Unknown,
            MemoryReport::Known(0),
            MemoryReport::Known(1),
            MemoryReport::Known(64),
            MemoryReport::Known(10_000),
        ];
        let cpus = [
            MemoryReport::Unknown,
            MemoryReport::Known(1),
            MemoryReport::Known(8),
            MemoryReport::Known(2_000),
        ];
        for (i, caller_a) in budgets.iter().enumerate() {
            for caller_b in budgets.iter().skip(i) {
                for total in caps {
                    for available in caps {
                        for limit in caps {
                            for current in caps {
                                for cpu in cpus {
                                    let ha = host(total, available, limit, current, cpu);
                                    let mut hb = ha;
                                    let pa = ResourcePlan::derive(
                                        &ResourcePolicy::new(*caller_a),
                                        &SystemProfile::from_memory(ha),
                                        &[] as &[Stub],
                                    );
                                    let pb = ResourcePlan::derive(
                                        &ResourcePolicy::new(*caller_b),
                                        &SystemProfile::from_memory(hb),
                                        &[] as &[Stub],
                                    );
                                    assert!(pb.budget_bytes >= pa.budget_bytes);
                                    assert_eq!(pa.thread_ceiling, pb.thread_ceiling);
                                    assert_eq!(pa.total_bytes, total);
                                    assert_eq!(pa.available_bytes, available);
                                    assert_eq!(pa.cgroup_limit_bytes, limit);
                                    match cpu {
                                        MemoryReport::Unknown => {
                                            assert_eq!(pa.thread_ceiling, MemoryReport::Unknown)
                                        }
                                        MemoryReport::Known(n) => {
                                            let ceiling = n.min(u64::from(CPU_THREAD_CEILING));
                                            assert_eq!(
                                                pa.thread_ceiling,
                                                MemoryReport::Known(ceiling)
                                            );
                                            assert!(ceiling <= n);
                                            assert!(ceiling <= u64::from(CPU_THREAD_CEILING));
                                        }
                                    }
                                    hb.total_bytes = raise(ha.total_bytes);
                                    hb.available_bytes = raise(ha.available_bytes);
                                    hb.cgroup_limit_bytes = raise(ha.cgroup_limit_bytes);
                                    let raised = ResourcePlan::derive(
                                        &ResourcePolicy::new(*caller_b),
                                        &SystemProfile::from_memory(hb),
                                        &[] as &[Stub],
                                    );
                                    assert!(raised.budget_bytes >= pb.budget_bytes);
                                    if let MemoryReport::Known(cur) = current {
                                        hb = ha;
                                        hb.cgroup_current_bytes =
                                            MemoryReport::Known(cur.saturating_add(3));
                                        let tighter_current = ResourcePlan::derive(
                                            &ResourcePolicy::new(*caller_a),
                                            &SystemProfile::from_memory(hb),
                                            &[] as &[Stub],
                                        );
                                        assert!(tighter_current.budget_bytes <= pa.budget_bytes);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn raise(report: MemoryReport) -> MemoryReport {
        match report {
            MemoryReport::Unknown => MemoryReport::Unknown,
            MemoryReport::Known(n) => MemoryReport::Known(n.saturating_add(7)),
        }
    }

    #[test]
    fn device_order_is_kept_and_an_unprobed_device_stays_unknown() {
        let mut policy = ResourcePolicy::new(500);
        policy.devices = vec![Device::Cuda, Device::Cpu];
        let probes = [Stub {
            kind: Device::Cuda,
            memory: MemoryReport::Known(32),
        }];
        let plan = ResourcePlan::derive(
            &policy,
            &SystemProfile::from_memory(HostMemory::all_unknown()),
            &probes,
        );
        assert_eq!(plan.devices, vec![Device::Cuda, Device::Cpu]);
        assert_eq!(
            plan.device_memory,
            vec![MemoryReport::Known(32), MemoryReport::Unknown]
        );
        assert_eq!(
            plan.budget_bytes, 500,
            "device memory is reported, not substituted for the caller budget"
        );
    }

    #[test]
    fn unknown_current_still_respects_a_known_limit() {
        let host = host(
            MemoryReport::Unknown,
            MemoryReport::Known(80),
            MemoryReport::Known(100),
            MemoryReport::Unknown,
            MemoryReport::Known(4),
        );
        let plan = ResourcePlan::derive(
            &ResourcePolicy::new(1_000),
            &SystemProfile::from_memory(host),
            &[] as &[Stub],
        );
        assert_eq!(plan.budget_bytes, 80);
        assert_eq!(plan.thread_ceiling, MemoryReport::Known(4));
        assert_eq!(plan.cgroup_limit_bytes, MemoryReport::Known(100));
    }

    /// Critical pressure admits nothing: the budget is 0 and every shared
    /// device room with it; Warning, Normal and Unknown leave the plan as
    /// the limits make it. Before, pressure was copied into the plan and
    /// read by nothing.
    #[test]
    fn critical_pressure_admits_nothing_and_milder_levels_change_nothing() {
        let memory = host(
            MemoryReport::Known(64 << 30),
            MemoryReport::Known(40 << 30),
            MemoryReport::Unknown,
            MemoryReport::Unknown,
            MemoryReport::Known(18),
        );
        let gpu = [Gpu {
            kind: Device::Metal,
            memory: MemoryReport::Known(48 << 30),
            resident: MemoryReport::Known(0),
            arch: MemoryArchitecture::Unified,
        }];
        let mut policy = ResourcePolicy::new(8 << 30);
        policy.devices = vec![Device::Cpu, Device::Metal];
        for (pressure, budget) in [
            (MemoryPressure::Normal, 8 << 30),
            (MemoryPressure::Warning, 8 << 30),
            (MemoryPressure::Unknown, 8 << 30),
            (MemoryPressure::Critical, 0),
        ] {
            let mut profile = SystemProfile::from_memory(memory);
            profile.pressure = pressure;
            let plan = ResourcePlan::derive(&policy, &profile, &gpu);
            assert_eq!(plan.budget_bytes, budget, "{pressure:?}");
            assert_eq!(plan.pressure, pressure);
            for room in &plan.device_room {
                assert!(
                    matches!(room, MemoryReport::Known(r) if *r <= budget),
                    "{pressure:?}: {room:?}"
                );
            }
        }
    }

    /// A known cgroup CPU quota caps the thread ceiling (rounded down to
    /// whole CPUs, as std does, at least one) even when the CPU count did
    /// not include it.
    #[test]
    fn a_known_cpu_quota_caps_the_thread_ceiling() {
        let memory = host(
            MemoryReport::Unknown,
            MemoryReport::Unknown,
            MemoryReport::Unknown,
            MemoryReport::Unknown,
            MemoryReport::Known(64),
        );
        for (quota, ceiling) in [
            (MemoryReport::Unknown, 64),
            (MemoryReport::Known(1500), 1),
            (MemoryReport::Known(1999), 1),
            (MemoryReport::Known(2000), 2),
            (MemoryReport::Known(2001), 2),
            (MemoryReport::Known(2999), 2),
            (MemoryReport::Known(1), 1),
            (MemoryReport::Known(0), 1),
            (MemoryReport::Known(1_000_000), 64),
            (MemoryReport::Known(u64::MAX), 64),
        ] {
            let mut profile = SystemProfile::from_memory(memory);
            profile.cpu.cpu_quota_millis = quota;
            let plan = ResourcePlan::derive(&ResourcePolicy::new(1), &profile, &[] as &[Stub]);
            assert_eq!(
                plan.thread_ceiling,
                MemoryReport::Known(ceiling),
                "{quota:?}"
            );
            assert!(
                matches!(plan.fast_threads, MemoryReport::Known(f) if f <= ceiling),
                "{quota:?}"
            );
        }
        let unknown_count = host(
            MemoryReport::Unknown,
            MemoryReport::Unknown,
            MemoryReport::Unknown,
            MemoryReport::Unknown,
            MemoryReport::Unknown,
        );
        let mut profile = SystemProfile::from_memory(unknown_count);
        profile.cpu.cpu_quota_millis = MemoryReport::Known(2000);
        let plan = ResourcePlan::derive(&ResourcePolicy::new(1), &profile, &[] as &[Stub]);
        assert_eq!(plan.thread_ceiling, MemoryReport::Unknown);
    }
}

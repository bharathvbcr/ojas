//! Turn a caller policy and host reports into a resource plan.
//!
//! The host probe never chooses the split factor. `allow_split` is copied
//! from the policy, where it defaults to false.

use crate::host::{HostMemory, MemoryReport};
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
    fn memory_bytes(&self) -> MemoryReport;
}

/// The numbers a caller can act on. Thread count is advice, not a spawn cap.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourcePlan {
    pub budget_bytes: u64,
    pub thread_ceiling: MemoryReport,
    pub allow_split: bool,
    pub devices: Vec<Device>,
    /// Parallel to [`ResourcePlan::devices`]. Missing probes stay unknown.
    pub device_memory: Vec<MemoryReport>,
    pub total_bytes: MemoryReport,
    pub available_bytes: MemoryReport,
    pub cgroup_limit_bytes: MemoryReport,
}

impl ResourcePlan {
    /// Clamp the caller budget by known host limits. Unknown inputs are
    /// copied through and are not replaced with a number.
    pub fn derive(policy: &ResourcePolicy, host: &HostMemory, probes: &[impl MemoryProbe]) -> Self {
        let mut budget = policy.caller_budget_bytes;
        budget = tighten(budget, host.total_bytes);
        budget = tighten(budget, host.available_bytes);
        budget = tighten(budget, host.cgroup_limit_bytes);
        if let (MemoryReport::Known(limit), MemoryReport::Known(current)) =
            (host.cgroup_limit_bytes, host.cgroup_current_bytes)
        {
            budget = budget.min(limit.saturating_sub(current));
        }
        let device_memory = policy
            .devices
            .iter()
            .map(|device| {
                probes
                    .iter()
                    .find(|probe| probe.kind() == *device)
                    .map(|probe| probe.memory_bytes())
                    .unwrap_or(MemoryReport::Unknown)
            })
            .collect();
        Self {
            budget_bytes: budget,
            thread_ceiling: thread_ceiling(host.cpu_count),
            allow_split: policy.allow_split,
            devices: policy.devices.clone(),
            device_memory,
            total_bytes: host.total_bytes,
            available_bytes: host.available_bytes,
            cgroup_limit_bytes: host.cgroup_limit_bytes,
        }
    }
}

fn tighten(budget: u64, report: MemoryReport) -> u64 {
    match report {
        MemoryReport::Known(n) => budget.min(n),
        MemoryReport::Unknown => budget,
    }
}

fn thread_ceiling(cpu_count: MemoryReport) -> MemoryReport {
    match cpu_count {
        MemoryReport::Unknown => MemoryReport::Unknown,
        MemoryReport::Known(n) => MemoryReport::Known(n.min(u64::from(CPU_THREAD_CEILING))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let plan = ResourcePlan::derive(&policy, &tiny, &[] as &[Stub]);
        assert!(!plan.allow_split);
        let mut split = policy.clone();
        split.allow_split = true;
        assert!(ResourcePlan::derive(&split, &tiny, &[] as &[Stub]).allow_split);
    }

    #[test]
    fn unknown_stays_unknown_and_does_not_shrink_the_caller_budget() {
        let policy = ResourcePolicy::new(1234);
        let plan = ResourcePlan::derive(&policy, &HostMemory::all_unknown(), &[] as &[Stub]);
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
                    let plan =
                        ResourcePlan::derive(&ResourcePolicy::new(caller), &host, &[] as &[Stub]);
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
                                        &ha,
                                        &[] as &[Stub],
                                    );
                                    let pb = ResourcePlan::derive(
                                        &ResourcePolicy::new(*caller_b),
                                        &hb,
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
                                        &hb,
                                        &[] as &[Stub],
                                    );
                                    assert!(raised.budget_bytes >= pb.budget_bytes);
                                    if let MemoryReport::Known(cur) = current {
                                        hb = ha;
                                        hb.cgroup_current_bytes =
                                            MemoryReport::Known(cur.saturating_add(3));
                                        let tighter_current = ResourcePlan::derive(
                                            &ResourcePolicy::new(*caller_a),
                                            &hb,
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
        let plan = ResourcePlan::derive(&policy, &HostMemory::all_unknown(), &probes);
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
        let plan = ResourcePlan::derive(&ResourcePolicy::new(1_000), &host, &[] as &[Stub]);
        assert_eq!(plan.budget_bytes, 80);
        assert_eq!(plan.thread_ceiling, MemoryReport::Known(4));
        assert_eq!(plan.cgroup_limit_bytes, MemoryReport::Known(100));
    }
}

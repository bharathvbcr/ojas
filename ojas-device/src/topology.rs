//! CPU topology, core clusters, caches, and the cgroup CPU quota.
//!
//! macOS reads `hw.*` sysctls, including the per-performance-level
//! `hw.perflevelN.*` set. Linux reads `/sys/devices/system/cpu` and the
//! cgroup CPU files under a root that tests point at a fixture tree.
//! Every field fails on its own to [`MemoryReport::Unknown`]; nothing is
//! filled in from a compiled-in table.
//!
//! The Linux readers are compiled on every target so their fixture tests
//! run on macOS too; there they have no production caller.

#![cfg_attr(target_os = "macos", allow(dead_code))]

use std::path::Path;

use crate::host::{cgroup_ancestors, parse_cgroup_self, read_text, MemoryReport};

/// Most CPUs a sysfs CPU list may name. A list that names more (or a range
/// such as `0-4294967295`) is not parsed into a vector.
pub(crate) const CPU_LIST_MAX: usize = 8192;

/// Most performance levels or clusters reported.
pub(crate) const CLUSTERS_MAX: usize = 16;

/// Cores that share one performance level (macOS) or one capacity /
/// core type (Linux). The first cluster in [`CpuTopology::clusters`] is the
/// fastest the probe could tell apart.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoreCluster {
    /// `hw.perflevelN.name` on macOS (`Super`, `Performance`, `Efficiency`,
    /// depending on the chip). On Linux a description of the grouping key.
    pub name: String,
    pub physical: MemoryReport,
    pub logical: MemoryReport,
    /// L1 data cache per core, bytes.
    pub l1d_bytes: MemoryReport,
    /// One L2 instance, bytes. It may be shared; see `cpus_per_l2`.
    pub l2_bytes: MemoryReport,
    /// Logical CPUs that share one L2 instance.
    pub cpus_per_l2: MemoryReport,
}

impl CoreCluster {
    fn unknown(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            physical: MemoryReport::Unknown,
            logical: MemoryReport::Unknown,
            l1d_bytes: MemoryReport::Unknown,
            l2_bytes: MemoryReport::Unknown,
            cpus_per_l2: MemoryReport::Unknown,
        }
    }
}

/// What the topology probe could read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuTopology {
    /// Online logical CPUs on the machine.
    pub logical: MemoryReport,
    /// Physical cores on the machine.
    pub physical: MemoryReport,
    /// [`std::thread::available_parallelism`]: CPUs this process may run on,
    /// after affinity and (on Linux) the cgroup quota.
    pub usable: MemoryReport,
    /// Fastest first. Empty when the probe could not group cores.
    pub clusters: Vec<CoreCluster>,
    /// Last-level cache past L2 (L3, or a system cache the OS reports), bytes.
    /// Apple silicon does not report its system-level cache; it is unknown.
    pub l3_bytes: MemoryReport,
    pub cache_line_bytes: MemoryReport,
    pub page_bytes: MemoryReport,
    /// cgroup CPU quota in thousandths of a CPU (`cpu.max` `200000 100000`
    /// is 2000). `max` and a missing file are unknown.
    pub cpu_quota_millis: MemoryReport,
}

impl CpuTopology {
    pub fn all_unknown() -> Self {
        Self {
            logical: MemoryReport::Unknown,
            physical: MemoryReport::Unknown,
            usable: MemoryReport::Unknown,
            clusters: Vec::new(),
            l3_bytes: MemoryReport::Unknown,
            cache_line_bytes: MemoryReport::Unknown,
            page_bytes: MemoryReport::Unknown,
            cpu_quota_millis: MemoryReport::Unknown,
        }
    }

    /// Smallest known value of `field` across clusters, or unknown when no
    /// cluster reports it. The smallest is the one every core can rely on.
    pub fn min_over_clusters(&self, field: impl Fn(&CoreCluster) -> MemoryReport) -> MemoryReport {
        self.clusters
            .iter()
            .filter_map(|c| match field(c) {
                MemoryReport::Known(n) => Some(n),
                MemoryReport::Unknown => None,
            })
            .min()
            .map_or(MemoryReport::Unknown, MemoryReport::Known)
    }
}

/// Probe this machine's CPU topology. Each field fails on its own.
pub fn probe_topology() -> CpuTopology {
    probe_topology_at(Path::new("/"))
}

/// [`probe_topology`] with the Linux files read under `root`.
pub(crate) fn probe_topology_at(root: &Path) -> CpuTopology {
    #[cfg(target_os = "macos")]
    let mut topo = {
        let _ = root;
        macos::topology()
    };
    #[cfg(not(target_os = "macos"))]
    let mut topo = linux_topology(root);
    topo.usable = match std::thread::available_parallelism() {
        Ok(n) => MemoryReport::Known(n.get() as u64),
        Err(_) => MemoryReport::Unknown,
    };
    #[cfg(not(target_os = "macos"))]
    {
        topo.cpu_quota_millis = probe_cpu_quota(root);
    }
    topo
}

/// Read the sysfs CPU tree under `root`. Missing files leave fields unknown.
pub(crate) fn linux_topology(root: &Path) -> CpuTopology {
    let mut topo = CpuTopology::all_unknown();
    topo.page_bytes = page_bytes();
    let cpu_dir = root.join("sys/devices/system/cpu");
    let Some(online) = read_text(cpu_dir.join("online")).and_then(|t| parse_cpu_list(t.trim()))
    else {
        return topo;
    };
    if online.is_empty() {
        return topo;
    }
    topo.logical = MemoryReport::Known(online.len() as u64);

    struct Cpu {
        id: usize,
        core: Option<(u64, u64)>,
        capacity: Option<u64>,
    }
    let cpus: Vec<Cpu> = online
        .iter()
        .map(|&id| {
            let base = cpu_dir.join(format!("cpu{id}"));
            let package = read_u64(&base.join("topology/physical_package_id"));
            let core_id = read_u64(&base.join("topology/core_id"));
            Cpu {
                id,
                core: package.zip(core_id),
                capacity: read_u64(&base.join("cpu_capacity")),
            }
        })
        .collect();
    if cpus.iter().all(|c| c.core.is_some()) {
        let mut cores: Vec<(u64, u64)> = cpus.iter().filter_map(|c| c.core).collect();
        cores.sort_unstable();
        cores.dedup();
        topo.physical = MemoryReport::Known(cores.len() as u64);
    }

    // Group into clusters: Intel hybrid core types, else arm64 capacity,
    // else one cluster of every online CPU.
    let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
    let devices = root.join("sys/devices");
    let core = read_text(devices.join("cpu_core/cpus")).and_then(|t| parse_cpu_list(t.trim()));
    let atom = read_text(devices.join("cpu_atom/cpus")).and_then(|t| parse_cpu_list(t.trim()));
    if let (Some(core), Some(atom)) = (core, atom) {
        for (name, ids) in [("cpu_core", core), ("cpu_atom", atom)] {
            let ids: Vec<usize> = ids
                .into_iter()
                .filter(|id| online.binary_search(id).is_ok())
                .collect();
            if !ids.is_empty() {
                groups.push((name.to_string(), ids));
            }
        }
    } else if cpus.iter().all(|c| c.capacity.is_some()) {
        let mut caps: Vec<u64> = cpus.iter().filter_map(|c| c.capacity).collect();
        caps.sort_unstable_by(|a, b| b.cmp(a));
        caps.dedup();
        for cap in caps.into_iter().take(CLUSTERS_MAX) {
            let ids = cpus
                .iter()
                .filter(|c| c.capacity == Some(cap))
                .map(|c| c.id)
                .collect();
            groups.push((format!("capacity {cap}"), ids));
        }
    } else {
        groups.push(("all".to_string(), online.clone()));
    }

    for (name, ids) in groups.into_iter().take(CLUSTERS_MAX) {
        let mut cluster = CoreCluster::unknown(name);
        cluster.logical = MemoryReport::Known(ids.len() as u64);
        let members: Vec<&Cpu> = cpus
            .iter()
            .filter(|c| ids.binary_search(&c.id).is_ok())
            .collect();
        if !members.is_empty() && members.iter().all(|c| c.core.is_some()) {
            let mut cores: Vec<(u64, u64)> = members.iter().filter_map(|c| c.core).collect();
            cores.sort_unstable();
            cores.dedup();
            cluster.physical = MemoryReport::Known(cores.len() as u64);
        }
        if let Some(&first) = ids.first() {
            let caches = read_caches(&cpu_dir.join(format!("cpu{first}/cache")));
            cluster.l1d_bytes = caches.l1d;
            cluster.l2_bytes = caches.l2;
            cluster.cpus_per_l2 = caches.l2_sharing;
            if topo.cache_line_bytes == MemoryReport::Unknown {
                topo.cache_line_bytes = caches.line;
            }
            if topo.l3_bytes == MemoryReport::Unknown {
                topo.l3_bytes = caches.l3;
            }
        }
        topo.clusters.push(cluster);
    }
    topo
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Caches {
    pub l1d: MemoryReport,
    pub l2: MemoryReport,
    pub l2_sharing: MemoryReport,
    pub l3: MemoryReport,
    pub line: MemoryReport,
}

/// `cache/index*` of one CPU. An index whose `level`, `type` or `size` is
/// unreadable is skipped. At most 16 indexes are read.
pub(crate) fn read_caches(dir: &Path) -> Caches {
    let mut out = Caches {
        l1d: MemoryReport::Unknown,
        l2: MemoryReport::Unknown,
        l2_sharing: MemoryReport::Unknown,
        l3: MemoryReport::Unknown,
        line: MemoryReport::Unknown,
    };
    for index in 0..16 {
        let base = dir.join(format!("index{index}"));
        let Some(level) = read_u64(&base.join("level")) else {
            continue;
        };
        let Some(kind) = read_text(base.join("type")) else {
            continue;
        };
        let Some(size) = read_text(base.join("size")).and_then(|t| parse_cache_size(t.trim()))
        else {
            continue;
        };
        let kind = kind.trim();
        let data = kind == "Data" || kind == "Unified";
        if !data {
            continue;
        }
        if out.line == MemoryReport::Unknown {
            if let Some(line) = read_u64(&base.join("coherency_line_size")) {
                if line > 0 {
                    out.line = MemoryReport::Known(line);
                }
            }
        }
        match level {
            1 => out.l1d = MemoryReport::Known(size),
            2 => {
                out.l2 = MemoryReport::Known(size);
                out.l2_sharing = read_text(base.join("shared_cpu_list"))
                    .and_then(|t| parse_cpu_list(t.trim()))
                    .filter(|ids| !ids.is_empty())
                    .map_or(MemoryReport::Unknown, |ids| {
                        MemoryReport::Known(ids.len() as u64)
                    });
            }
            3 => out.l3 = MemoryReport::Known(size),
            _ => {}
        }
    }
    out
}

/// A sysfs CPU list such as `0-3,8,10-11`, sorted and without duplicates.
///
/// `None` for an empty token, a reversed range, a non-number, or a list
/// that names more than [`CPU_LIST_MAX`] CPUs. An empty string is an
/// empty list.
pub(crate) fn parse_cpu_list(text: &str) -> Option<Vec<usize>> {
    let mut out: Vec<usize> = Vec::new();
    if text.is_empty() {
        return Some(out);
    }
    for token in text.split(',') {
        let token = token.trim();
        let (lo, hi) = match token.split_once('-') {
            Some((lo, hi)) => (lo.parse::<usize>().ok()?, hi.parse::<usize>().ok()?),
            None => {
                let n = token.parse::<usize>().ok()?;
                (n, n)
            }
        };
        // `hi - lo < CPU_LIST_MAX` first, so `+ 1` cannot overflow.
        if lo > hi || hi - lo >= CPU_LIST_MAX || out.len() + (hi - lo + 1) > CPU_LIST_MAX {
            return None;
        }
        out.extend(lo..=hi);
    }
    out.sort_unstable();
    out.dedup();
    Some(out)
}

/// A sysfs cache size: `48K`, `1280K`, `32M`, `1G`, or a bare byte count.
pub(crate) fn parse_cache_size(text: &str) -> Option<u64> {
    let (digits, factor) = match text.as_bytes().last()? {
        b'K' | b'k' => (&text[..text.len() - 1], 1u64 << 10),
        b'M' | b'm' => (&text[..text.len() - 1], 1u64 << 20),
        b'G' | b'g' => (&text[..text.len() - 1], 1u64 << 30),
        _ => (text, 1),
    };
    let n = digits.parse::<u64>().ok()?;
    let bytes = n.checked_mul(factor)?;
    (bytes > 0).then_some(bytes)
}

fn read_u64(path: &Path) -> Option<u64> {
    read_text(path)?.trim().parse::<u64>().ok()
}

/// cgroup CPU quota for this process, in thousandths of a CPU, the
/// tightest over the cgroup and its ancestors. v2 `cpu.max`, v1
/// `cpu.cfs_quota_us` / `cpu.cfs_period_us`.
pub(crate) fn probe_cpu_quota(root: &Path) -> MemoryReport {
    let Some(text) = read_text(root.join("proc/self/cgroup")) else {
        return MemoryReport::Unknown;
    };
    let mut best: Option<u64> = None;
    let mut keep = |m: Option<u64>| {
        if let Some(m) = m {
            best = Some(best.map_or(m, |b| b.min(m)));
        }
    };
    if let Some(path) = parse_cgroup_self(&text).v2.as_deref() {
        for dir in cgroup_ancestors(path) {
            let file = root.join("sys/fs/cgroup").join(dir.trim_start_matches('/'));
            keep(read_text(file.join("cpu.max")).and_then(|t| parse_cpu_max(t.trim())));
        }
    }
    if let Some(path) = parse_cgroup_v1(&text, "cpu") {
        for dir in cgroup_ancestors(&path) {
            for mount in ["sys/fs/cgroup/cpu", "sys/fs/cgroup/cpu,cpuacct"] {
                let base = root.join(mount).join(dir.trim_start_matches('/'));
                let quota = read_text(base.join("cpu.cfs_quota_us"));
                let period = read_text(base.join("cpu.cfs_period_us"));
                if let (Some(q), Some(p)) = (quota, period) {
                    keep(quota_millis(q.trim(), p.trim()));
                }
            }
        }
    }
    best.map_or(MemoryReport::Unknown, MemoryReport::Known)
}

/// The v1 path of the hierarchy that lists `controller`, or `None`.
/// A path containing `..` or NUL is dropped.
fn parse_cgroup_v1(text: &str, controller: &str) -> Option<String> {
    for line in text.lines() {
        let mut parts = line.splitn(3, ':');
        let _id = parts.next();
        let controllers = parts.next()?;
        let path = parts.next()?;
        if path.contains("..") || path.contains('\0') {
            continue;
        }
        if controllers.split(',').any(|c| c == controller) {
            return Some(path.to_string());
        }
    }
    None
}

/// v2 `cpu.max`: `max 100000` is no quota; `200000 100000` is 2000 milli-CPUs.
pub(crate) fn parse_cpu_max(text: &str) -> Option<u64> {
    let mut parts = text.split_whitespace();
    let quota = parts.next()?;
    let period = parts.next()?;
    if parts.next().is_some() || quota == "max" {
        return None;
    }
    quota_millis(quota, period)
}

/// `quota / period * 1000`, rounded up so a fractional CPU is never 0.
/// A negative quota (v1's `-1`), a zero period, or overflow is `None`.
fn quota_millis(quota: &str, period: &str) -> Option<u64> {
    let quota = quota.parse::<u64>().ok()?;
    let period = period.parse::<u64>().ok()?;
    if quota == 0 || period == 0 {
        return None;
    }
    quota
        .checked_mul(1000)?
        .checked_add(period - 1)
        .map(|n| n / period)
}

#[cfg(not(target_os = "macos"))]
fn page_bytes() -> MemoryReport {
    #[cfg(target_os = "linux")]
    {
        sysconf_page()
    }
    #[cfg(not(target_os = "linux"))]
    {
        MemoryReport::Unknown
    }
}

#[cfg(target_os = "macos")]
fn page_bytes() -> MemoryReport {
    match crate::sysctl::u64_by_name(b"hw.pagesize\0") {
        Some(n) if n > 0 => MemoryReport::Known(n),
        _ => MemoryReport::Unknown,
    }
}

#[cfg(target_os = "linux")]
#[allow(unsafe_code)]
fn sysconf_page() -> MemoryReport {
    // SAFETY: `sysconf` takes no pointers and only reads a constant.
    let n = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if n > 0 {
        MemoryReport::Known(n as u64)
    } else {
        MemoryReport::Unknown
    }
}

#[cfg(target_os = "macos")]
mod macos {
    use super::{CoreCluster, CpuTopology, CLUSTERS_MAX};
    use crate::host::MemoryReport;
    use crate::sysctl::{string_by_name, u64_by_name};

    fn known(name: &str) -> MemoryReport {
        let mut bytes = name.as_bytes().to_vec();
        bytes.push(0);
        match u64_by_name(&bytes) {
            Some(n) if n > 0 => MemoryReport::Known(n),
            _ => MemoryReport::Unknown,
        }
    }

    pub(super) fn topology() -> CpuTopology {
        let mut topo = CpuTopology::all_unknown();
        topo.logical = known("hw.logicalcpu");
        topo.physical = known("hw.physicalcpu");
        topo.l3_bytes = known("hw.l3cachesize");
        topo.cache_line_bytes = known("hw.cachelinesize");
        topo.page_bytes = super::page_bytes();
        let levels = match known("hw.nperflevels") {
            MemoryReport::Known(n) => (n as usize).min(CLUSTERS_MAX),
            MemoryReport::Unknown => 0,
        };
        for level in 0..levels {
            let key = |field: &str| format!("hw.perflevel{level}.{field}");
            let mut name_key = key("name").into_bytes();
            name_key.push(0);
            let name = string_by_name(&name_key).unwrap_or_else(|| format!("perflevel{level}"));
            topo.clusters.push(CoreCluster {
                name,
                physical: known(&key("physicalcpu")),
                logical: known(&key("logicalcpu")),
                l1d_bytes: known(&key("l1dcachesize")),
                l2_bytes: known(&key("l2cachesize")),
                cpus_per_l2: known(&key("cpusperl2")),
            });
        }
        if topo.clusters.is_empty() {
            // Intel Macs and older releases: one machine-wide level.
            topo.clusters.push(CoreCluster {
                name: "all".to_string(),
                physical: topo.physical,
                logical: topo.logical,
                l1d_bytes: known("hw.l1dcachesize"),
                l2_bytes: known("hw.l2cachesize"),
                cpus_per_l2: MemoryReport::Unknown,
            });
        }
        topo
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::Fixture;

    fn cache(
        f: &Fixture,
        cpu: usize,
        index: usize,
        level: u32,
        kind: &str,
        size: &str,
        shared: &str,
    ) {
        let base = format!("sys/devices/system/cpu/cpu{cpu}/cache/index{index}");
        f.put(&format!("{base}/level"), &format!("{level}\n"));
        f.put(&format!("{base}/type"), &format!("{kind}\n"));
        f.put(&format!("{base}/size"), &format!("{size}\n"));
        f.put(&format!("{base}/coherency_line_size"), "64\n");
        f.put(&format!("{base}/shared_cpu_list"), &format!("{shared}\n"));
    }

    fn core(f: &Fixture, cpu: usize, package: u64, core: u64) {
        let base = format!("sys/devices/system/cpu/cpu{cpu}/topology");
        f.put(
            &format!("{base}/physical_package_id"),
            &format!("{package}\n"),
        );
        f.put(&format!("{base}/core_id"), &format!("{core}\n"));
    }

    #[test]
    fn cpu_list_parses_ranges_and_refuses_garbage() {
        assert_eq!(parse_cpu_list(""), Some(vec![]));
        assert_eq!(parse_cpu_list("0"), Some(vec![0]));
        assert_eq!(
            parse_cpu_list("0-3,8,10-11"),
            Some(vec![0, 1, 2, 3, 8, 10, 11])
        );
        assert_eq!(parse_cpu_list("3,1-2,2"), Some(vec![1, 2, 3]));
        for bad in [
            "3-1",
            "a",
            "1,,2",
            "-1",
            "1-",
            "0-4294967295",
            "0-18446744073709551615",
            "18446744073709551616",
            "1-2-3",
            " - ",
        ] {
            assert_eq!(parse_cpu_list(bad), None, "{bad:?}");
        }
        // Exactly the cap is accepted; one past it is refused.
        let cap = CPU_LIST_MAX;
        assert_eq!(
            parse_cpu_list(&format!("0-{}", cap - 1)).unwrap().len(),
            cap
        );
        assert_eq!(parse_cpu_list(&format!("0-{cap}")), None);
        // Many small tokens cannot add up past the cap either.
        let many: Vec<String> = (0..=CPU_LIST_MAX).map(|i| i.to_string()).collect();
        assert_eq!(parse_cpu_list(&many.join(",")), None);
        assert_eq!(parse_cpu_list(&many[..cap].join(",")).unwrap().len(), cap);
    }

    #[test]
    fn cache_size_units_and_overflow() {
        assert_eq!(parse_cache_size("48K"), Some(48 << 10));
        assert_eq!(parse_cache_size("32M"), Some(32 << 20));
        assert_eq!(parse_cache_size("1G"), Some(1 << 30));
        assert_eq!(parse_cache_size("4096"), Some(4096));
        for bad in [
            "",
            "K",
            "0K",
            "0",
            "-1K",
            "1T",
            "1.5M",
            "18446744073709551615K",
        ] {
            assert_eq!(parse_cache_size(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn cpu_max_and_quota_rounding() {
        assert_eq!(parse_cpu_max("max 100000"), None);
        assert_eq!(parse_cpu_max("200000 100000"), Some(2000));
        assert_eq!(parse_cpu_max("50000 100000"), Some(500));
        assert_eq!(parse_cpu_max("1 100000"), Some(1), "a sliver is not 0");
        for bad in ["", "max", "0 100000", "100 0", "a b", "1 2 3", "-1 100000"] {
            assert_eq!(parse_cpu_max(bad), None, "{bad:?}");
        }
        assert_eq!(parse_cpu_max(&format!("{} 1", u64::MAX)), None, "overflow");
    }

    #[test]
    fn an_empty_root_is_all_unknown() {
        let f = Fixture::new("empty");
        let topo = linux_topology(&f.0);
        assert_eq!(topo.logical, MemoryReport::Unknown);
        assert_eq!(topo.physical, MemoryReport::Unknown);
        assert!(topo.clusters.is_empty());
        assert_eq!(topo.l3_bytes, MemoryReport::Unknown);
        assert_eq!(probe_cpu_quota(&f.0), MemoryReport::Unknown);
    }

    #[test]
    fn a_smt_x86_tree_counts_cores_and_reads_caches() {
        let f = Fixture::new("smt");
        f.put("sys/devices/system/cpu/online", "0-3\n");
        // Two cores, two threads each.
        for (cpu, core_id) in [(0, 0), (1, 1), (2, 0), (3, 1)] {
            core(&f, cpu, 0, core_id);
            cache(&f, cpu, 0, 1, "Data", "48K", &format!("{cpu}"));
            cache(&f, cpu, 1, 1, "Instruction", "32K", &format!("{cpu}"));
            cache(&f, cpu, 2, 2, "Unified", "1280K", "0,2");
            cache(&f, cpu, 3, 3, "Unified", "30M", "0-3");
        }
        let topo = linux_topology(&f.0);
        assert_eq!(topo.logical, MemoryReport::Known(4));
        assert_eq!(topo.physical, MemoryReport::Known(2));
        assert_eq!(topo.clusters.len(), 1);
        let c = &topo.clusters[0];
        assert_eq!(c.name, "all");
        assert_eq!(c.logical, MemoryReport::Known(4));
        assert_eq!(c.physical, MemoryReport::Known(2));
        assert_eq!(
            c.l1d_bytes,
            MemoryReport::Known(48 << 10),
            "instruction cache skipped"
        );
        assert_eq!(c.l2_bytes, MemoryReport::Known(1280 << 10));
        assert_eq!(c.cpus_per_l2, MemoryReport::Known(2));
        assert_eq!(topo.l3_bytes, MemoryReport::Known(30 << 20));
        assert_eq!(topo.cache_line_bytes, MemoryReport::Known(64));
    }

    #[test]
    fn arm_capacities_split_into_clusters_fastest_first() {
        let f = Fixture::new("biglittle");
        f.put("sys/devices/system/cpu/online", "0-5\n");
        for cpu in 0..6 {
            core(&f, cpu, 0, cpu as u64);
            let (cap, l1, l2, shared) = if cpu < 4 {
                (446, "32K", "512K", "0-3")
            } else {
                (1024, "64K", "2M", "4-5")
            };
            f.put(
                &format!("sys/devices/system/cpu/cpu{cpu}/cpu_capacity"),
                &format!("{cap}\n"),
            );
            cache(&f, cpu, 0, 1, "Data", l1, &cpu.to_string());
            cache(&f, cpu, 1, 2, "Unified", l2, shared);
        }
        let topo = linux_topology(&f.0);
        assert_eq!(topo.clusters.len(), 2);
        assert_eq!(topo.clusters[0].name, "capacity 1024");
        assert_eq!(topo.clusters[0].logical, MemoryReport::Known(2));
        assert_eq!(topo.clusters[0].l2_bytes, MemoryReport::Known(2 << 20));
        assert_eq!(topo.clusters[1].name, "capacity 446");
        assert_eq!(topo.clusters[1].cpus_per_l2, MemoryReport::Known(4));
        assert_eq!(
            topo.min_over_clusters(|c| c.l1d_bytes),
            MemoryReport::Known(32 << 10)
        );
        assert_eq!(topo.l3_bytes, MemoryReport::Unknown);
    }

    #[test]
    fn intel_hybrid_types_win_over_capacity_and_offline_cpus_drop() {
        let f = Fixture::new("hybrid");
        f.put("sys/devices/system/cpu/online", "0-3,5\n");
        f.put("sys/devices/cpu_core/cpus", "0-1\n");
        f.put("sys/devices/cpu_atom/cpus", "2-5\n");
        for cpu in [0, 1, 2, 3, 5] {
            core(&f, cpu, 0, cpu as u64);
            cache(
                &f,
                cpu,
                0,
                1,
                "Data",
                if cpu < 2 { "48K" } else { "32K" },
                "0",
            );
        }
        let topo = linux_topology(&f.0);
        assert_eq!(topo.clusters.len(), 2);
        assert_eq!(topo.clusters[0].name, "cpu_core");
        assert_eq!(topo.clusters[1].name, "cpu_atom");
        assert_eq!(
            topo.clusters[1].logical,
            MemoryReport::Known(3),
            "cpu4 is offline"
        );
        assert_eq!(topo.clusters[0].l1d_bytes, MemoryReport::Known(48 << 10));
    }

    #[test]
    fn each_missing_file_only_blanks_its_own_field() {
        let f = Fixture::new("holes");
        f.put("sys/devices/system/cpu/online", "0-1\n");
        // cpu1 has no core_id: the physical count is unknown, the rest stays.
        core(&f, 0, 0, 0);
        f.put(
            "sys/devices/system/cpu/cpu1/topology/physical_package_id",
            "0\n",
        );
        cache(&f, 0, 0, 1, "Data", "garbage", "0");
        cache(&f, 0, 1, 2, "Unified", "2M", "not-a-list");
        let topo = linux_topology(&f.0);
        assert_eq!(topo.logical, MemoryReport::Known(2));
        assert_eq!(topo.physical, MemoryReport::Unknown);
        let c = &topo.clusters[0];
        assert_eq!(c.l1d_bytes, MemoryReport::Unknown, "bad size");
        assert_eq!(c.l2_bytes, MemoryReport::Known(2 << 20));
        assert_eq!(c.cpus_per_l2, MemoryReport::Unknown, "bad shared list");
    }

    #[test]
    fn hostile_online_files_are_unknown_not_huge() {
        for text in ["0-4294967295", "zzz", "5-1"] {
            let f = Fixture::new("hostile");
            f.put("sys/devices/system/cpu/online", text);
            let topo = linux_topology(&f.0);
            assert_eq!(topo.logical, MemoryReport::Unknown, "{text}");
            assert!(topo.clusters.is_empty());
        }
        // A file past the probe read cap is not read at all.
        let f = Fixture::new("big");
        let big = "0,".repeat((crate::host::PROBE_FILE_MAX as usize) / 2 + 8);
        f.put("sys/devices/system/cpu/online", &big);
        assert_eq!(linux_topology(&f.0).logical, MemoryReport::Unknown);
    }

    #[test]
    fn cpu_quota_takes_the_tightest_ancestor_in_v2_and_v1() {
        let f = Fixture::new("quota");
        f.put("proc/self/cgroup", "0::/pod/ctr\n4:cpu,cpuacct:/job\n");
        f.put("sys/fs/cgroup/cpu.max", "max 100000\n");
        f.put("sys/fs/cgroup/pod/cpu.max", "150000 100000\n");
        f.put("sys/fs/cgroup/pod/ctr/cpu.max", "400000 100000\n");
        assert_eq!(probe_cpu_quota(&f.0), MemoryReport::Known(1500));
        f.put("sys/fs/cgroup/cpu,cpuacct/job/cpu.cfs_quota_us", "50000\n");
        f.put(
            "sys/fs/cgroup/cpu,cpuacct/job/cpu.cfs_period_us",
            "100000\n",
        );
        assert_eq!(probe_cpu_quota(&f.0), MemoryReport::Known(500));
        // v1 `-1` is no quota.
        f.put("sys/fs/cgroup/cpu,cpuacct/job/cpu.cfs_quota_us", "-1\n");
        assert_eq!(probe_cpu_quota(&f.0), MemoryReport::Known(1500));
    }

    #[test]
    fn dotdot_cgroup_paths_are_not_followed_for_cpu() {
        let f = Fixture::new("dotdot");
        f.put("proc/self/cgroup", "0::/../../escape\n");
        f.put("sys/fs/cgroup/cpu.max", "100000 100000\n");
        assert_eq!(probe_cpu_quota(&f.0), MemoryReport::Unknown);
    }

    #[test]
    fn probe_topology_is_consistent_on_this_machine() {
        let topo = probe_topology();
        match std::thread::available_parallelism() {
            Ok(n) => assert_eq!(topo.usable, MemoryReport::Known(n.get() as u64)),
            Err(_) => assert_eq!(topo.usable, MemoryReport::Unknown),
        }
        if let (MemoryReport::Known(l), MemoryReport::Known(p)) = (topo.logical, topo.physical) {
            assert!(p <= l, "physical {p} logical {l}");
        }
        // Cluster logical counts never add up past the machine's.
        if let MemoryReport::Known(l) = topo.logical {
            let sum: u64 = topo
                .clusters
                .iter()
                .filter_map(|c| match c.logical {
                    MemoryReport::Known(n) => Some(n),
                    MemoryReport::Unknown => None,
                })
                .sum();
            assert!(sum <= l, "clusters {sum} machine {l}");
        }
        assert!(topo.clusters.len() <= CLUSTERS_MAX);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_reports_perf_levels_and_caches() {
        let topo = probe_topology();
        assert!(!topo.clusters.is_empty());
        for c in &topo.clusters {
            assert!(
                matches!(c.l1d_bytes, MemoryReport::Known(n) if n >= 16 << 10),
                "{c:?}"
            );
            assert!(
                matches!(c.l2_bytes, MemoryReport::Known(n) if n >= 256 << 10),
                "{c:?}"
            );
        }
        assert!(matches!(topo.page_bytes, MemoryReport::Known(n) if n.is_power_of_two()));
        assert!(matches!(topo.cache_line_bytes, MemoryReport::Known(n) if n.is_power_of_two()));
    }
}

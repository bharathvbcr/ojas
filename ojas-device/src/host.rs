//! Host memory and CPU probes.
//!
//! A failed read is [`MemoryReport::Unknown`]. This module does not
//! substitute a compiled-in guess.

#![allow(unsafe_code)]

use std::fs;
use std::path::Path;

/// A measured quantity, or a probe that did not produce a number.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryReport {
    Known(u64),
    Unknown,
}

/// What the host probe could read about this machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostMemory {
    /// Physical RAM. macOS `hw.memsize`, Linux `MemTotal`.
    pub total_bytes: MemoryReport,
    /// Available-ish memory. macOS is `(free_count + inactive_count) * page size`.
    /// Linux is `MemAvailable`. It is not a substitute for a missing total.
    pub available_bytes: MemoryReport,
    /// Tightest numeric cgroup memory limit, if one was present.
    ///
    /// cgroup v2 `memory.max` of `max`, and a cgroup v1 limit at the
    /// kernel's unlimited sentinel (`>= 1 << 62`), are [`MemoryReport::Unknown`]
    /// because they are not a finite cap.
    pub cgroup_limit_bytes: MemoryReport,
    /// The working set of the cgroup whose limit was used: `memory.current`
    /// (v1 `memory.usage_in_bytes`) less `inactive_file` (v1
    /// `total_inactive_file`) from `memory.stat`, or the raw figure when
    /// `memory.stat` is unreadable. `Unknown` when the usage file was missing.
    pub cgroup_current_bytes: MemoryReport,
    /// [`std::thread::available_parallelism`], or `Unknown` if that call fails.
    pub cpu_count: MemoryReport,
}

impl HostMemory {
    pub fn all_unknown() -> Self {
        Self {
            total_bytes: MemoryReport::Unknown,
            available_bytes: MemoryReport::Unknown,
            cgroup_limit_bytes: MemoryReport::Unknown,
            cgroup_current_bytes: MemoryReport::Unknown,
            cpu_count: MemoryReport::Unknown,
        }
    }
}

/// Read RAM, cgroup limits, and the CPU count. Each field fails on its own.
pub fn probe_host() -> HostMemory {
    probe_host_at(Path::new("/"))
}

/// [`probe_host`] with `/proc` and `/sys` read under `root`, so the Linux
/// paths can be driven from a fixture tree. macOS sysctls ignore `root`.
pub(crate) fn probe_host_at(root: &Path) -> HostMemory {
    let mut host = HostMemory::all_unknown();
    #[cfg(target_os = "macos")]
    {
        host.total_bytes = macos::total_bytes();
        host.available_bytes = macos::available_bytes();
    }
    #[cfg(not(target_os = "macos"))]
    {
        if let Some(text) = read_text(root.join("proc/meminfo")) {
            let (total, available) = parse_meminfo(&text);
            host.total_bytes = total;
            host.available_bytes = available;
        }
    }
    let (limit, current) = probe_cgroup(root);
    host.cgroup_limit_bytes = limit;
    host.cgroup_current_bytes = current;
    host.cpu_count = match std::thread::available_parallelism() {
        Ok(n) => MemoryReport::Known(n.get() as u64),
        Err(_) => MemoryReport::Unknown,
    };
    host
}

fn probe_cgroup(root: &Path) -> (MemoryReport, MemoryReport) {
    let Some(text) = read_text(root.join("proc/self/cgroup")) else {
        return (MemoryReport::Unknown, MemoryReport::Unknown);
    };
    let found = parse_cgroup_self(&text);
    let v2_mount = root.join("sys/fs/cgroup");
    let v1_mount = root.join("sys/fs/cgroup/memory");
    let mut best: Option<(u64, MemoryReport)> = None;
    // A cgroup is bounded by every ancestor's limit too, so a tighter
    // parent (a pod around a container) is read, not only the leaf.
    if let Some(path) = found.v2.as_deref() {
        for dir in cgroup_ancestors(path) {
            let limit = read_cgroup_counter(&v2_mount, dir, "memory.max", true);
            let current = read_cgroup_counter(&v2_mount, dir, "memory.current", false);
            let current = working_set(&v2_mount, dir, current, "inactive_file");
            consider(&mut best, limit, current);
        }
    }
    if let Some(path) = found.v1_memory.as_deref() {
        for dir in cgroup_ancestors(path) {
            let limit = read_cgroup_counter(&v1_mount, dir, "memory.limit_in_bytes", true);
            let current = read_cgroup_counter(&v1_mount, dir, "memory.usage_in_bytes", false);
            let current = working_set(&v1_mount, dir, current, "total_inactive_file");
            consider(&mut best, limit, current);
        }
    }
    match best {
        Some((limit, current)) => (MemoryReport::Known(limit), current),
        None => (MemoryReport::Unknown, MemoryReport::Unknown),
    }
}

/// `current` less the cgroup's inactive file pages (`memory.stat` `key`):
/// the working set, as the kubelet computes it. `memory.current` counts page
/// cache, so a process that has just read a large model sits near its limit
/// though the kernel reclaims those pages before it OOM-kills anything.
///
/// A missing or unreadable `memory.stat`, or a missing key, leaves `current`
/// as it was: the raw figure is never smaller than the working set, so the
/// fallback only ever under-states the room. So does a `memory.stat` that
/// claims more inactive pages than the usage file holds: the two disagree,
/// and a working set of 0 (the whole limit as room) is the one answer that
/// could over-state it, so the raw figure is kept.
fn working_set(mount: &Path, dir: &str, current: MemoryReport, key: &str) -> MemoryReport {
    let MemoryReport::Known(current) = current else {
        return MemoryReport::Unknown;
    };
    let inactive = cgroup_file(mount, dir, "memory.stat")
        .and_then(read_text)
        .and_then(|text| parse_stat_key(&text, key));
    MemoryReport::Known(match inactive {
        Some(inactive) if inactive <= current => current - inactive,
        Some(_) | None => current,
    })
}

/// The value of `key` in a `memory.stat` file (`key value` per line).
/// A duplicate key or a non-numeric value is `None`.
pub(crate) fn parse_stat_key(text: &str, key: &str) -> Option<u64> {
    let mut found = None;
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        if parts.next() != Some(key) {
            continue;
        }
        let value = parts.next()?.parse::<u64>().ok()?;
        if parts.next().is_some() || found.is_some() {
            return None;
        }
        found = Some(value);
    }
    found
}

/// Deepest cgroup nesting walked. Deeper paths keep their first levels
/// (the mount root and its nearest children) and the leaf.
pub(crate) const CGROUP_DEPTH_MAX: usize = 64;

/// `path` and each ancestor up to the mount root (`""`), leaf first.
///
/// `/a/b` gives `/a/b`, `/a`, `""`. Empty components (`//`) are skipped.
/// Beyond [`CGROUP_DEPTH_MAX`] levels the middle is not walked; the leaf and
/// the levels nearest the root still are.
pub(crate) fn cgroup_ancestors(path: &str) -> Vec<&str> {
    let mut ends: Vec<usize> = Vec::new();
    let mut seen_component = false;
    for (i, ch) in path.char_indices() {
        if ch == '/' {
            if seen_component {
                ends.push(i);
            }
            seen_component = false;
        } else {
            seen_component = true;
        }
    }
    if seen_component {
        ends.push(path.len());
    }
    let mut out: Vec<&str> = Vec::with_capacity(ends.len().min(CGROUP_DEPTH_MAX) + 1);
    if let Some(&leaf) = ends.last() {
        out.push(&path[..leaf]);
    }
    let keep = ends.len().saturating_sub(1).min(CGROUP_DEPTH_MAX - 1);
    for &end in ends[..keep].iter().rev() {
        out.push(&path[..end]);
    }
    out.push("");
    out
}

/// Keep the hierarchy whose usable bytes are fewer.
///
/// Usable bytes are `limit - current` when both are known, otherwise the
/// limit itself. That result is at most every known limit, so the plan
/// never goes past a known cgroup cap. A missing current is not treated as
/// zero.
fn consider(best: &mut Option<(u64, MemoryReport)>, limit: MemoryReport, current: MemoryReport) {
    let MemoryReport::Known(limit) = limit else {
        return;
    };
    let room = cgroup_room(limit, current);
    match best {
        None => *best = Some((limit, current)),
        Some((prev_limit, prev_current)) => {
            let prev_room = cgroup_room(*prev_limit, *prev_current);
            if room < prev_room || (room == prev_room && limit < *prev_limit) {
                *best = Some((limit, current));
            }
        }
    }
}

fn cgroup_room(limit: u64, current: MemoryReport) -> u64 {
    match current {
        MemoryReport::Known(current) => limit.saturating_sub(current),
        MemoryReport::Unknown => limit,
    }
}

fn read_cgroup_counter(mount: &Path, cgroup: &str, file: &str, is_limit: bool) -> MemoryReport {
    let Some(path) = cgroup_file(mount, cgroup, file) else {
        return MemoryReport::Unknown;
    };
    let Some(text) = read_text(path) else {
        return MemoryReport::Unknown;
    };
    parse_cgroup_counter(text.trim(), is_limit)
}

/// Most bytes read from one `/proc` or `/sys` file. Every file the probes
/// read is a few KiB at most; a larger one is not a kernel file and is
/// unknown rather than read whole.
pub(crate) const PROBE_FILE_MAX: u64 = 64 * 1024;

/// A small UTF-8 file, or `None` when it is missing, unreadable, not UTF-8,
/// or longer than [`PROBE_FILE_MAX`].
pub(crate) fn read_text(path: impl AsRef<Path>) -> Option<String> {
    use std::io::Read;
    let file = fs::File::open(path).ok()?;
    let mut text = String::new();
    file.take(PROBE_FILE_MAX + 1)
        .read_to_string(&mut text)
        .ok()?;
    if text.len() as u64 > PROBE_FILE_MAX {
        return None;
    }
    Some(text)
}

/// `MemTotal` and `MemAvailable`, in bytes. Missing keys stay unknown.
///
/// Compiled off macOS and for tests. macOS reads RAM through sysctl.
#[cfg(any(test, not(target_os = "macos")))]
pub(crate) fn parse_meminfo(text: &str) -> (MemoryReport, MemoryReport) {
    let mut total = MemoryReport::Unknown;
    let mut available = MemoryReport::Unknown;
    for line in text.lines() {
        let Some((key, rest)) = line.split_once(':') else {
            continue;
        };
        let report = parse_meminfo_value(rest);
        match key.trim() {
            "MemTotal" => total = report,
            "MemAvailable" => available = report,
            _ => {}
        }
    }
    (total, available)
}

#[cfg(any(test, not(target_os = "macos")))]
fn parse_meminfo_value(rest: &str) -> MemoryReport {
    let mut parts = rest.split_whitespace();
    let Some(number) = parts.next() else {
        return MemoryReport::Unknown;
    };
    let Ok(kb) = number.parse::<u64>() else {
        return MemoryReport::Unknown;
    };
    let unit = parts.next().unwrap_or("kB");
    let factor: u64 = match unit {
        "kB" => 1024,
        "B" => 1,
        _ => return MemoryReport::Unknown,
    };
    match kb.checked_mul(factor) {
        Some(bytes) => MemoryReport::Known(bytes),
        None => MemoryReport::Unknown,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CgroupPaths {
    pub v2: Option<String>,
    pub v1_memory: Option<String>,
}

/// Paths from `/proc/self/cgroup`. A line whose path contains `..` is dropped.
pub(crate) fn parse_cgroup_self(text: &str) -> CgroupPaths {
    let mut out = CgroupPaths {
        v2: None,
        v1_memory: None,
    };
    for line in text.lines() {
        let mut parts = line.splitn(3, ':');
        let _id = parts.next();
        let Some(controllers) = parts.next() else {
            continue;
        };
        let Some(path) = parts.next() else {
            continue;
        };
        if path.contains("..") || path.contains('\0') {
            continue;
        }
        if controllers.is_empty() {
            out.v2 = Some(path.to_string());
        } else if controllers.split(',').any(|c| c == "memory") {
            out.v1_memory = Some(path.to_string());
        }
    }
    out
}

/// `max` and a v1 unlimited sentinel are unknown limits. A bad token is unknown.
pub(crate) fn parse_cgroup_counter(token: &str, is_limit: bool) -> MemoryReport {
    if is_limit && token == "max" {
        return MemoryReport::Unknown;
    }
    let Ok(n) = token.parse::<u64>() else {
        return MemoryReport::Unknown;
    };
    if is_limit && n >= (1 << 62) {
        return MemoryReport::Unknown;
    }
    MemoryReport::Known(n)
}

fn cgroup_file(mount: &Path, cgroup: &str, file: &str) -> Option<std::path::PathBuf> {
    if cgroup.contains("..") || cgroup.contains('\0') {
        return None;
    }
    let rel = cgroup.trim_start_matches('/');
    Some(Path::new(mount).join(rel).join(file))
}

#[cfg(target_os = "macos")]
mod macos {
    use super::MemoryReport;

    use crate::sysctl::u64_by_name as sysctl_u64;

    pub(super) fn total_bytes() -> MemoryReport {
        match sysctl_u64(b"hw.memsize\0") {
            Some(n) => MemoryReport::Known(n),
            None => MemoryReport::Unknown,
        }
    }

    /// `(free_count + inactive_count) * hw.pagesize`. Either sysctl failing
    /// is unknown; the page size is not assumed.
    pub(super) fn available_bytes() -> MemoryReport {
        let Some(page_size) = sysctl_u64(b"hw.pagesize\0") else {
            return MemoryReport::Unknown;
        };
        if page_size == 0 {
            return MemoryReport::Unknown;
        }
        let Some((free_count, inactive_count)) = vm_free_and_inactive() else {
            return MemoryReport::Unknown;
        };
        let pages = u64::from(free_count).saturating_add(u64::from(inactive_count));
        match pages.checked_mul(page_size) {
            Some(bytes) => MemoryReport::Known(bytes),
            None => MemoryReport::Unknown,
        }
    }

    fn vm_free_and_inactive() -> Option<(u32, u32)> {
        let mut stats = unsafe { std::mem::zeroed::<libc::vm_statistics64>() };
        let mut count = libc::HOST_VM_INFO64_COUNT;
        let port = host_port();
        if port == 0 {
            return None;
        }
        let kr = unsafe {
            libc::host_statistics64(
                port,
                libc::HOST_VM_INFO64,
                &mut stats as *mut libc::vm_statistics64 as libc::host_info64_t,
                &mut count,
            )
        };
        release_host_port(port);
        if kr != libc::KERN_SUCCESS {
            return None;
        }
        Some((stats.free_count, stats.inactive_count))
    }

    #[allow(deprecated)]
    fn host_port() -> libc::mach_port_t {
        unsafe { libc::mach_host_self() }
    }

    #[allow(deprecated)]
    fn release_host_port(port: libc::mach_port_t) {
        unsafe {
            let _ = mach_port_deallocate(libc::mach_task_self(), port);
        }
    }

    unsafe extern "C" {
        fn mach_port_deallocate(
            task: libc::mach_port_t,
            name: libc::mach_port_t,
        ) -> libc::kern_return_t;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meminfo_parses_kibibytes_and_leaves_holes_unknown() {
        let (total, available) = parse_meminfo("MemTotal: 2048 kB\nMemFree: 10 kB\n");
        assert_eq!(total, MemoryReport::Known(2048 * 1024));
        assert_eq!(available, MemoryReport::Unknown);
        let (total, available) = parse_meminfo("MemAvailable: 3 kB\n");
        assert_eq!(total, MemoryReport::Unknown);
        assert_eq!(available, MemoryReport::Known(3072));
        let (total, _) = parse_meminfo("MemTotal: no kB\n");
        assert_eq!(total, MemoryReport::Unknown);
    }

    #[test]
    fn cgroup_parser_keeps_unknown_on_max_and_drops_dotdot() {
        let paths = parse_cgroup_self("0::/user.slice/pod\n5:memory:/system.slice\n");
        assert_eq!(paths.v2.as_deref(), Some("/user.slice/pod"));
        assert_eq!(paths.v1_memory.as_deref(), Some("/system.slice"));
        let paths = parse_cgroup_self("0::/../../etc\n6:memory,cpu:../x\n");
        assert_eq!(paths.v2, None);
        assert_eq!(paths.v1_memory, None);
        assert_eq!(parse_cgroup_counter("max", true), MemoryReport::Unknown);
        assert_eq!(
            parse_cgroup_counter("4096", true),
            MemoryReport::Known(4096)
        );
        assert_eq!(
            parse_cgroup_counter("9223372036854771712", true),
            MemoryReport::Unknown
        );
        assert_eq!(
            parse_cgroup_counter("not-a-number", false),
            MemoryReport::Unknown
        );
        assert_eq!(parse_cgroup_counter("12", false), MemoryReport::Known(12));
    }

    #[test]
    fn cgroup_ancestors_walk_leaf_to_root_and_stay_bounded() {
        assert_eq!(cgroup_ancestors("/"), vec![""]);
        assert_eq!(cgroup_ancestors(""), vec![""]);
        assert_eq!(cgroup_ancestors("/a/b"), vec!["/a/b", "/a", ""]);
        assert_eq!(cgroup_ancestors("a//b/"), vec!["a//b", "a", ""]);
        let deep = "/x".repeat(10 * CGROUP_DEPTH_MAX);
        let walk = cgroup_ancestors(&deep);
        assert!(walk.len() <= CGROUP_DEPTH_MAX + 1, "{}", walk.len());
        assert_eq!(walk[0], deep.as_str(), "the leaf is always read");
        assert_eq!(walk.last(), Some(&""), "the mount root is always read");
        assert_eq!(
            walk[walk.len() - 2],
            "/x",
            "levels nearest the root are kept"
        );
    }

    #[test]
    fn a_tighter_parent_cgroup_limit_is_not_missed() {
        use crate::testutil::Fixture;
        let f = Fixture::new("cg-parent");
        f.put("proc/self/cgroup", "0::/pod/ctr\n");
        f.put("sys/fs/cgroup/pod/ctr/memory.max", "max\n");
        f.put("sys/fs/cgroup/pod/ctr/memory.current", "100\n");
        f.put("sys/fs/cgroup/pod/memory.max", "4096\n");
        f.put("sys/fs/cgroup/pod/memory.current", "1000\n");
        assert_eq!(
            probe_cgroup(&f.0),
            (MemoryReport::Known(4096), MemoryReport::Known(1000)),
            "the leaf's `max` used to hide the pod limit"
        );
        // A tighter leaf wins over a looser parent.
        f.put("sys/fs/cgroup/pod/ctr/memory.max", "2048\n");
        assert_eq!(
            probe_cgroup(&f.0),
            (MemoryReport::Known(2048), MemoryReport::Known(100))
        );
        // The room, not the raw limit, decides: the parent has 96 bytes left.
        f.put("sys/fs/cgroup/pod/memory.current", "4000\n");
        assert_eq!(
            probe_cgroup(&f.0),
            (MemoryReport::Known(4096), MemoryReport::Known(4000))
        );
    }

    /// A container that has just read a 3 GiB model sits at 3.9 of its
    /// 4 GiB. Before the working-set fix its room was 0.1 GiB; the page cache
    /// is reclaimable, so the room is 3.1 GiB.
    #[test]
    fn page_cache_does_not_count_against_the_cgroup_room() {
        use crate::testutil::Fixture;
        const GIB: u64 = 1 << 30;
        let f = Fixture::new("cg-pagecache");
        f.put("proc/self/cgroup", "0::/ctr\n");
        f.put("sys/fs/cgroup/ctr/memory.max", &format!("{}\n", 4 * GIB));
        f.put(
            "sys/fs/cgroup/ctr/memory.current",
            &format!("{}\n", 4 * GIB - GIB / 10),
        );
        f.put(
            "sys/fs/cgroup/ctr/memory.stat",
            &format!(
                "anon 100\nfile {}\ninactive_file {}\nactive_file 5\n",
                3 * GIB,
                3 * GIB
            ),
        );
        let working = 4 * GIB - GIB / 10 - 3 * GIB;
        assert_eq!(
            probe_cgroup(&f.0),
            (MemoryReport::Known(4 * GIB), MemoryReport::Known(working))
        );
        // v1 subtracts the hierarchical `total_inactive_file`, not the
        // cgroup-local `inactive_file`.
        let f = Fixture::new("cg-pagecache-v1");
        f.put("proc/self/cgroup", "4:memory:/job\n");
        f.put("sys/fs/cgroup/memory/job/memory.limit_in_bytes", "1000\n");
        f.put("sys/fs/cgroup/memory/job/memory.usage_in_bytes", "900\n");
        f.put(
            "sys/fs/cgroup/memory/job/memory.stat",
            "inactive_file 100\ntotal_inactive_file 700\n",
        );
        assert_eq!(probe_cgroup(&f.0).1, MemoryReport::Known(200));
    }

    #[test]
    fn an_unreadable_memory_stat_keeps_the_raw_usage() {
        use crate::testutil::Fixture;
        let f = Fixture::new("cg-stat-holes");
        f.put("proc/self/cgroup", "0::/c\n");
        f.put("sys/fs/cgroup/c/memory.max", "1000\n");
        f.put("sys/fs/cgroup/c/memory.current", "900\n");
        assert_eq!(probe_cgroup(&f.0).1, MemoryReport::Known(900), "no stat");
        for (stat, want) in [
            ("anon 5\n", 900),                             // no key
            ("inactive_file x\n", 900),                    // not a number
            ("inactive_file 10\ninactive_file 20\n", 900), // duplicate
            ("inactive_file 10 11\n", 900),                // trailing token
            ("total_inactive_file 300\n", 900),            // v1 key on v2
            ("inactive_file 5000\n", 900),                 // more than usage: contradictory
            ("inactive_file 900\n", 0),                    // exactly the usage
            ("inactive_file 300\n", 600),
        ] {
            f.put("sys/fs/cgroup/c/memory.stat", stat);
            assert_eq!(probe_cgroup(&f.0).1, MemoryReport::Known(want), "{stat:?}");
        }
        assert_eq!(parse_stat_key("", "inactive_file"), None);
    }

    #[test]
    fn v1_hierarchy_ancestors_and_sentinels() {
        use crate::testutil::Fixture;
        let f = Fixture::new("cg-v1");
        f.put("proc/self/cgroup", "7:memory:/docker/abc\n");
        f.put(
            "sys/fs/cgroup/memory/docker/abc/memory.limit_in_bytes",
            "9223372036854771712\n",
        );
        f.put(
            "sys/fs/cgroup/memory/docker/memory.limit_in_bytes",
            "8192\n",
        );
        assert_eq!(probe_cgroup(&f.0).0, MemoryReport::Known(8192));
    }

    #[test]
    fn hostile_cgroup_inputs_stay_unknown() {
        use crate::testutil::Fixture;
        let f = Fixture::new("cg-hostile");
        assert_eq!(
            probe_cgroup(&f.0),
            (MemoryReport::Unknown, MemoryReport::Unknown),
            "no /proc/self/cgroup"
        );
        f.put("proc/self/cgroup", "0::/../../etc\n");
        f.put("sys/fs/cgroup/memory.max", "1\n");
        assert_eq!(probe_cgroup(&f.0).0, MemoryReport::Unknown);
        // An oversized cgroup file is not read.
        f.put("proc/self/cgroup", &"0::/a\n".repeat(20_000));
        assert_eq!(probe_cgroup(&f.0).0, MemoryReport::Unknown);
        f.put("proc/self/cgroup", "0::/a\n");
        for bad in ["", "-5", "1e9", "4096 4096", "\u{0}"] {
            f.put("sys/fs/cgroup/a/memory.max", bad);
            f.put("sys/fs/cgroup/memory.max", "max");
            assert_eq!(probe_cgroup(&f.0).0, MemoryReport::Unknown, "{bad:?}");
        }
    }

    #[test]
    fn read_text_refuses_past_the_cap_and_accepts_at_it() {
        use crate::testutil::Fixture;
        let f = Fixture::new("readcap");
        let at = "a".repeat(PROBE_FILE_MAX as usize);
        f.put("at", &at);
        f.put("past", &format!("{at}a"));
        assert_eq!(read_text(f.0.join("at")).map(|t| t.len()), Some(at.len()));
        assert_eq!(read_text(f.0.join("past")), None);
        assert_eq!(read_text(f.0.join("missing")), None);
        std::fs::write(f.0.join("binary"), [0xff, 0xfe, 0x00]).unwrap();
        assert_eq!(read_text(f.0.join("binary")), None);
    }

    #[test]
    fn probe_host_matches_parallelism_and_does_not_invent_the_rest() {
        let host = probe_host();
        match std::thread::available_parallelism() {
            Ok(n) => assert_eq!(host.cpu_count, MemoryReport::Known(n.get() as u64)),
            Err(_) => assert_eq!(host.cpu_count, MemoryReport::Unknown),
        }
        if let (MemoryReport::Known(total), MemoryReport::Known(available)) =
            (host.total_bytes, host.available_bytes)
        {
            assert!(available <= total, "available {available} total {total}");
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_total_ram_is_a_sysctl_value() {
        let host = probe_host();
        assert!(
            matches!(host.total_bytes, MemoryReport::Known(n) if n >= 1 << 20),
            "{:?}",
            host.total_bytes
        );
    }
}

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
    /// `memory.current` or `memory.usage_in_bytes` for the hierarchy whose
    /// limit was used, or `Unknown` when that file was missing.
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
    let mut host = HostMemory::all_unknown();
    #[cfg(target_os = "macos")]
    {
        host.total_bytes = macos::total_bytes();
        host.available_bytes = macos::available_bytes();
    }
    #[cfg(target_os = "linux")]
    {
        if let Some(text) = read_text("/proc/meminfo") {
            let (total, available) = parse_meminfo(&text);
            host.total_bytes = total;
            host.available_bytes = available;
        }
    }
    let (limit, current) = probe_cgroup();
    host.cgroup_limit_bytes = limit;
    host.cgroup_current_bytes = current;
    host.cpu_count = match std::thread::available_parallelism() {
        Ok(n) => MemoryReport::Known(n.get() as u64),
        Err(_) => MemoryReport::Unknown,
    };
    host
}

fn probe_cgroup() -> (MemoryReport, MemoryReport) {
    let Some(text) = read_text("/proc/self/cgroup") else {
        return (MemoryReport::Unknown, MemoryReport::Unknown);
    };
    let found = parse_cgroup_self(&text);
    let mut best: Option<(u64, MemoryReport)> = None;
    if let Some(path) = found.v2.as_deref() {
        let limit = read_cgroup_counter("/sys/fs/cgroup", path, "memory.max", true);
        let current = read_cgroup_counter("/sys/fs/cgroup", path, "memory.current", false);
        consider(&mut best, limit, current);
    }
    if let Some(path) = found.v1_memory.as_deref() {
        let limit =
            read_cgroup_counter("/sys/fs/cgroup/memory", path, "memory.limit_in_bytes", true);
        let current = read_cgroup_counter(
            "/sys/fs/cgroup/memory",
            path,
            "memory.usage_in_bytes",
            false,
        );
        consider(&mut best, limit, current);
    }
    match best {
        Some((limit, current)) => (MemoryReport::Known(limit), current),
        None => (MemoryReport::Unknown, MemoryReport::Unknown),
    }
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

fn read_cgroup_counter(mount: &str, cgroup: &str, file: &str, is_limit: bool) -> MemoryReport {
    let Some(path) = cgroup_file(mount, cgroup, file) else {
        return MemoryReport::Unknown;
    };
    let Some(text) = read_text(path) else {
        return MemoryReport::Unknown;
    };
    parse_cgroup_counter(text.trim(), is_limit)
}

fn read_text(path: impl AsRef<Path>) -> Option<String> {
    fs::read_to_string(path).ok()
}

/// `MemTotal` and `MemAvailable`, in bytes. Missing keys stay unknown.
///
/// Compiled for the Linux probe and for tests. macOS reads RAM through sysctl.
#[cfg(any(test, target_os = "linux"))]
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

#[cfg(any(test, target_os = "linux"))]
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

fn cgroup_file(mount: &str, cgroup: &str, file: &str) -> Option<std::path::PathBuf> {
    if cgroup.contains("..") || cgroup.contains('\0') {
        return None;
    }
    let rel = cgroup.trim_start_matches('/');
    Some(Path::new(mount).join(rel).join(file))
}

#[cfg(target_os = "macos")]
mod macos {
    use super::MemoryReport;

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

    fn sysctl_u64(name: &[u8]) -> Option<u64> {
        let mut buf = [0u8; 8];
        let mut len = buf.len();
        let rc = unsafe {
            libc::sysctlbyname(
                name.as_ptr() as *const libc::c_char,
                buf.as_mut_ptr() as *mut libc::c_void,
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        if rc != 0 || len == 0 || len > buf.len() {
            return None;
        }
        let mut tmp = [0u8; 8];
        tmp[..len].copy_from_slice(&buf[..len]);
        Some(u64::from_le_bytes(tmp))
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

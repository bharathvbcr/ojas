//! One snapshot of the machine: memory, CPU topology, memory architecture
//! and memory pressure. Bandwidth is measured separately and only when a
//! caller asks ([`crate::measure_bandwidth`]).

#[cfg(any(test, target_os = "linux"))]
use std::path::Path;

use crate::host::{probe_host, HostMemory};
use crate::topology::{probe_topology, CpuTopology};

/// Whether the CPU and the GPU draw from one physical memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryArchitecture {
    /// One pool: Apple silicon. A GPU allocation is host RAM the CPU budget
    /// can no longer use, and the reverse.
    Unified,
    /// The device has its own memory (a PCIe GPU).
    Discrete,
    /// The probe could not tell: an x86 Mac, a Linux host with an AMD GPU or
    /// with both an integrated and a discrete one, any other OS. A GPU
    /// runtime's [`crate::MemoryProbe`] can say more.
    Unknown,
}

/// The kernel's own reading of memory pressure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryPressure {
    Normal,
    Warning,
    Critical,
    Unknown,
}

/// What [`probe_system`] could read. Each part fails on its own.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SystemProfile {
    pub memory: HostMemory,
    pub cpu: CpuTopology,
    pub architecture: MemoryArchitecture,
    pub pressure: MemoryPressure,
}

impl SystemProfile {
    /// A profile that knows only `memory`. For tests and for callers that
    /// probe memory on their own.
    pub fn from_memory(memory: HostMemory) -> Self {
        Self {
            memory,
            cpu: CpuTopology::all_unknown(),
            architecture: MemoryArchitecture::Unknown,
            pressure: MemoryPressure::Unknown,
        }
    }
}

/// Probe memory, topology, architecture and pressure. Cheap: sysctls and
/// small `/proc` and `/sys` reads, no allocation past a few KiB, no timing.
pub fn probe_system() -> SystemProfile {
    let mut cpu = probe_topology();
    let memory = probe_host();
    // One source for the usable count: the memory probe and the topology
    // probe both read `available_parallelism`; keep them equal.
    cpu.usable = memory.cpu_count;
    SystemProfile {
        memory,
        cpu,
        architecture: probe_architecture(),
        pressure: probe_pressure(),
    }
}

/// Apple silicon (`hw.optional.arm64` = 1) has unified memory on every
/// model. An x86 Mac may have a discrete GPU, an integrated one, or both,
/// and is unknown here. Linux reads its GPUs from sysfs
/// ([`linux_architecture`]). Every other OS is unknown.
fn probe_architecture() -> MemoryArchitecture {
    #[cfg(target_os = "macos")]
    {
        match crate::sysctl::u64_by_name(b"hw.optional.arm64\0") {
            Some(1) => MemoryArchitecture::Unified,
            _ => MemoryArchitecture::Unknown,
        }
    }
    #[cfg(target_os = "linux")]
    {
        linux_architecture(Path::new("/"))
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        MemoryArchitecture::Unknown
    }
}

/// PCI vendor ids sysfs reports as `device/vendor`.
#[cfg(any(test, target_os = "linux"))]
const PCI_NVIDIA: &str = "0x10de";
#[cfg(any(test, target_os = "linux"))]
const PCI_INTEL: &str = "0x8086";
#[cfg(any(test, target_os = "linux"))]
const PCI_VIRTIO: &str = "0x1af4";

/// The memory architecture of the GPUs Linux lists as DRM cards
/// (`/sys/class/drm/card<N>`, connectors such as `card0-HDMI-A-1` aside).
///
/// - No card at all (an empty DRM class, or sysfs with no DRM class because
///   no GPU driver is loaded): the host has no GPU memory of its own, so
///   every device the engine can open there (the CPU, a CPU wgpu adapter)
///   draws on host RAM. That is reported as `Unified`: one pool, the host's.
///   No sysfs to read at all (not mounted, or hidden by a sandbox) is
///   absence of evidence, and `Unknown`.
/// - A card with no PCI vendor (an SoC GPU on the platform bus, as on Arm
///   boards) and a virtio GPU (a VM's, backed by guest RAM) are unified.
/// - An Intel GPU on PCI bus 0 is the integrated one (Intel places it at
///   `0000:00:02.0`); Intel's discrete cards sit behind a root port.
/// - An NVIDIA GPU is discrete: it has its own memory.
/// - Anything else (an AMD GPU, which may be an APU or a card) is unknown.
///
/// All cards unified is `Unified`, all discrete is `Discrete`, and a mix
/// or any unknown card is `Unknown`: a host with an integrated and a
/// discrete GPU has both architectures, and the GPU runtime's own
/// [`crate::MemoryProbe`] decides for the device it opened.
#[cfg(any(test, target_os = "linux"))]
pub(crate) fn linux_architecture(root: &Path) -> MemoryArchitecture {
    let Ok(entries) = std::fs::read_dir(root.join("sys/class/drm")) else {
        // With sysfs readable, no DRM class means no GPU driver is loaded;
        // without it nothing was learned.
        return if root.join("sys/class").is_dir() {
            MemoryArchitecture::Unified
        } else {
            MemoryArchitecture::Unknown
        };
    };
    let mut seen = None;
    // Bounded: a host lists a handful of cards; past this it is unknown.
    for entry in entries.take(256) {
        let Ok(entry) = entry else {
            return MemoryArchitecture::Unknown;
        };
        let name = entry.file_name();
        let Some(n) = name.to_str().and_then(|n| n.strip_prefix("card")) else {
            continue;
        };
        if n.is_empty() || !n.bytes().all(|b| b.is_ascii_digit()) {
            continue;
        }
        let card = classify_card(&entry.path().join("device"));
        match (seen, card) {
            (_, MemoryArchitecture::Unknown) => return MemoryArchitecture::Unknown,
            (None, arch) => seen = Some(arch),
            (Some(a), b) if a != b => return MemoryArchitecture::Unknown,
            _ => {}
        }
    }
    seen.unwrap_or(MemoryArchitecture::Unified)
}

/// One card's architecture from its `device` directory.
#[cfg(any(test, target_os = "linux"))]
fn classify_card(device: &Path) -> MemoryArchitecture {
    let Some(vendor) = crate::host::read_text(device.join("vendor")) else {
        // No PCI vendor: a platform (SoC) GPU, which shares system RAM.
        return MemoryArchitecture::Unified;
    };
    match vendor.trim() {
        PCI_VIRTIO => MemoryArchitecture::Unified,
        PCI_NVIDIA => MemoryArchitecture::Discrete,
        PCI_INTEL => match std::fs::read_link(device) {
            Ok(target) if on_pci_bus_zero(&target) => MemoryArchitecture::Unified,
            _ => MemoryArchitecture::Unknown,
        },
        _ => MemoryArchitecture::Unknown,
    }
}

/// Whether a sysfs device link ends in a PCI address on bus 0
/// (`DDDD:00:SS.F`).
#[cfg(any(test, target_os = "linux"))]
fn on_pci_bus_zero(target: &Path) -> bool {
    let Some(last) = target.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    let mut parts = last.split(':');
    matches!(
        (parts.next(), parts.next(), parts.next(), parts.next()),
        (Some(domain), Some("00"), Some(slot), None)
            if domain.len() == 4 && slot.contains('.')
    )
}

/// The kernel's own reading of memory pressure.
///
/// macOS: `kern.memorystatus_vm_pressure_level`: 1 normal, 2 warning,
/// 4 critical (`DISPATCH_MEMORYPRESSURE_*`); other values are unknown.
/// Linux: pressure stall information ([`linux_pressure`]). Every other OS
/// is unknown. Cheap enough to ask before each call that allocates: one
/// sysctl, or a few small `/proc` and cgroup reads.
pub fn probe_pressure() -> MemoryPressure {
    #[cfg(target_os = "macos")]
    {
        pressure_from_level(crate::sysctl::u64_by_name(
            b"kern.memorystatus_vm_pressure_level\0",
        ))
    }
    #[cfg(target_os = "linux")]
    {
        linux_pressure(Path::new("/"))
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        MemoryPressure::Unknown
    }
}

/// Share of the last 10 s that some task stalled on memory, in percent, at
/// or past which pressure is [`MemoryPressure::Warning`].
#[cfg(any(test, target_os = "linux"))]
pub(crate) const PSI_SOME_WARNING: f64 = 10.0;
/// Share of the last 10 s that every non-idle task stalled on memory at
/// once, at or past which pressure is [`MemoryPressure::Critical`]: the
/// kernel spends that time reclaiming instead of running anything.
#[cfg(any(test, target_os = "linux"))]
pub(crate) const PSI_FULL_CRITICAL: f64 = 10.0;
/// A `some` share this high is critical too, whatever `full` says.
#[cfg(any(test, target_os = "linux"))]
pub(crate) const PSI_SOME_CRITICAL: f64 = 50.0;

/// Linux memory pressure: the worst of the system's
/// `/proc/pressure/memory` and the `memory.pressure` of this process's
/// cgroup v2 and each of its ancestors (a container's own limit stalls it
/// before the host feels anything). Unknown when no PSI file can be read
/// (a kernel without `CONFIG_PSI`, or booted `psi=0`).
///
/// The thresholds are this crate's, not a kernel or systemd default:
/// Warning at a `some avg10` of [`PSI_SOME_WARNING`]%, Critical at a
/// `full avg10` of [`PSI_FULL_CRITICAL`]% or a `some avg10` of
/// [`PSI_SOME_CRITICAL`]%.
#[cfg(any(test, target_os = "linux"))]
pub(crate) fn linux_pressure(root: &Path) -> MemoryPressure {
    let mut worst = crate::host::read_text(root.join("proc/pressure/memory"))
        .map_or(MemoryPressure::Unknown, |t| pressure_from_psi(&t));
    if let Some(cgroup) = crate::host::read_text(root.join("proc/self/cgroup")) {
        if let Some(path) = crate::host::parse_cgroup_self(&cgroup).v2 {
            let mount = root.join("sys/fs/cgroup");
            for dir in crate::host::cgroup_ancestors(&path) {
                let level = crate::host::cgroup_file(&mount, dir, "memory.pressure")
                    .and_then(crate::host::read_text)
                    .map_or(MemoryPressure::Unknown, |t| pressure_from_psi(&t));
                worst = worse(worst, level);
            }
        }
    }
    worst
}

/// The more severe of two readings; a known one wins over `Unknown`.
#[cfg(any(test, target_os = "linux"))]
fn worse(a: MemoryPressure, b: MemoryPressure) -> MemoryPressure {
    let rank = |p: MemoryPressure| match p {
        MemoryPressure::Unknown => 0,
        MemoryPressure::Normal => 1,
        MemoryPressure::Warning => 2,
        MemoryPressure::Critical => 3,
    };
    if rank(b) > rank(a) {
        b
    } else {
        a
    }
}

/// A PSI file (`some avg10=1.23 avg60=... total=...`, then a `full` line)
/// as a level. A file without a parseable `some avg10` is unknown; a
/// missing `full` line (a kernel that reports only `some`) is read as
/// no full stall.
#[cfg(any(test, target_os = "linux"))]
pub(crate) fn pressure_from_psi(text: &str) -> MemoryPressure {
    let mut some = None;
    let mut full = None;
    for line in text.lines() {
        let mut fields = line.split_whitespace();
        let slot = match fields.next() {
            Some("some") => &mut some,
            Some("full") => &mut full,
            _ => continue,
        };
        let avg10 = fields
            .find_map(|f| f.strip_prefix("avg10="))
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| v.is_finite() && (0.0..=100.0).contains(v));
        match avg10 {
            Some(v) if slot.is_none() => *slot = Some(v),
            // A second line of the same kind, or a bad figure, is a file
            // this reader does not understand.
            _ => return MemoryPressure::Unknown,
        }
    }
    let Some(some) = some else {
        return MemoryPressure::Unknown;
    };
    let full = full.unwrap_or(0.0);
    if full >= PSI_FULL_CRITICAL || some >= PSI_SOME_CRITICAL {
        MemoryPressure::Critical
    } else if some >= PSI_SOME_WARNING {
        MemoryPressure::Warning
    } else {
        MemoryPressure::Normal
    }
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn pressure_from_level(level: Option<u64>) -> MemoryPressure {
    match level {
        Some(1) => MemoryPressure::Normal,
        Some(2) => MemoryPressure::Warning,
        Some(4) => MemoryPressure::Critical,
        _ => MemoryPressure::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pressure_levels_map_and_others_are_unknown() {
        assert_eq!(pressure_from_level(Some(1)), MemoryPressure::Normal);
        assert_eq!(pressure_from_level(Some(2)), MemoryPressure::Warning);
        assert_eq!(pressure_from_level(Some(4)), MemoryPressure::Critical);
        for other in [None, Some(0), Some(3), Some(8), Some(u64::MAX)] {
            assert_eq!(pressure_from_level(other), MemoryPressure::Unknown);
        }
    }

    #[test]
    fn the_profile_agrees_with_its_parts() {
        let p = probe_system();
        assert_eq!(p.cpu.usable, p.memory.cpu_count);
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        assert_eq!(p.architecture, MemoryArchitecture::Unified);
        #[cfg(target_os = "linux")]
        {
            assert_eq!(p.architecture, linux_architecture(Path::new("/")));
            // A kernel with PSI always reports a level: the probe may not
            // fall back to Unknown where the file is there to read.
            if crate::host::read_text("/proc/pressure/memory").is_some() {
                assert_ne!(p.pressure, MemoryPressure::Unknown);
            }
            eprintln!(
                "linux: architecture {:?}, pressure {:?}",
                p.architecture, p.pressure
            );
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        assert_eq!(p.architecture, MemoryArchitecture::Unknown);
        #[cfg(target_os = "macos")]
        assert_ne!(p.pressure, MemoryPressure::Unknown);
    }

    /// Live, on a real kernel: in a cgroup whose `memory.high` sits far
    /// below what one thread keeps touching, the kernel throttles that
    /// thread and charges the stall to the cgroup's PSI, and the probe must
    /// read `Critical`, the level `E_PRESSURE` admission refuses at. Run in
    /// the Linux container (`--memory=1g --cgroup-conf=memory.high=96m`)
    /// with `OJAS_EXPECT_PRESSURE=critical` and `--ignored`; the hog stops
    /// after 60 s whatever it reads, and `memory.max` bounds what it holds.
    #[cfg(target_os = "linux")]
    #[test]
    #[ignore]
    fn live_pressure_reaches_critical_under_a_throttled_cgroup() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;
        use std::time::{Duration, Instant};
        if std::env::var("OJAS_EXPECT_PRESSURE").as_deref() != Ok("critical") {
            eprintln!("SKIP: set OJAS_EXPECT_PRESSURE=critical in a throttled cgroup");
            return;
        }
        let stop = Arc::new(AtomicBool::new(false));
        let hog = {
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                const BYTES: usize = 256 << 20;
                while !stop.load(Ordering::Relaxed) {
                    let mut v = vec![0u8; BYTES];
                    for page in v.chunks_mut(4096) {
                        page[0] = 1;
                        if stop.load(Ordering::Relaxed) {
                            break;
                        }
                    }
                    std::hint::black_box(&v);
                }
            })
        };
        let deadline = Instant::now() + Duration::from_secs(60);
        let mut seen = Vec::new();
        let level = loop {
            std::thread::sleep(Duration::from_millis(500));
            let level = probe_pressure();
            seen.push(level);
            if level == MemoryPressure::Critical || Instant::now() >= deadline {
                break level;
            }
        };
        stop.store(true, Ordering::Relaxed);
        hog.join().expect("hog");
        let cgroup = crate::host::read_text("/sys/fs/cgroup/memory.pressure");
        eprintln!(
            "after {} readings: {level:?}; cgroup memory.pressure: {cgroup:?}",
            seen.len()
        );
        assert_eq!(level, MemoryPressure::Critical, "readings: {seen:?}");
    }

    fn psi(some: &str, full: Option<&str>) -> String {
        let mut text = format!("some avg10={some} avg60=0.00 avg300=0.00 total=12345\n");
        if let Some(full) = full {
            text.push_str(&format!(
                "full avg10={full} avg60=0.00 avg300=0.00 total=99\n"
            ));
        }
        text
    }

    #[test]
    fn psi_maps_to_a_level_at_the_documented_thresholds() {
        let cases = [
            ("0.00", Some("0.00"), MemoryPressure::Normal),
            ("9.99", Some("9.99"), MemoryPressure::Normal),
            ("10.00", Some("0.00"), MemoryPressure::Warning),
            ("49.99", Some("9.99"), MemoryPressure::Warning),
            ("50.00", Some("0.00"), MemoryPressure::Critical),
            ("12.00", Some("10.00"), MemoryPressure::Critical),
            ("100.00", Some("100.00"), MemoryPressure::Critical),
            // A kernel that reports only `some`.
            ("3.00", None, MemoryPressure::Normal),
            ("60.00", None, MemoryPressure::Critical),
        ];
        for (some, full, want) in cases {
            assert_eq!(pressure_from_psi(&psi(some, full)), want, "{some} {full:?}");
        }
        for bad in [
            String::new(),
            "full avg10=50.00 avg60=0 avg300=0 total=1\n".to_string(),
            psi("nan", Some("0.00")),
            psi("-1.00", Some("0.00")),
            psi("101.00", Some("0.00")),
            psi("1.00", Some("inf")),
            "some avg60=1.00 total=1\n".to_string(),
            format!("{}{}", psi("1.00", None), psi("1.00", None)),
        ] {
            assert_eq!(pressure_from_psi(&bad), MemoryPressure::Unknown, "{bad:?}");
        }
    }

    /// The system file and every cgroup ancestor's file count; the worst
    /// wins, so a container stalling on its own limit is not hidden by a
    /// calm host. With no PSI file anywhere the level is unknown.
    #[test]
    fn linux_pressure_is_the_worst_of_the_system_and_the_cgroup_chain() {
        let f = crate::testutil::Fixture::new("psi");
        assert_eq!(linux_pressure(&f.0), MemoryPressure::Unknown);
        f.put("proc/pressure/memory", &psi("0.00", Some("0.00")));
        assert_eq!(linux_pressure(&f.0), MemoryPressure::Normal);
        f.put("proc/self/cgroup", "0::/pod/ctr\n");
        f.put(
            "sys/fs/cgroup/pod/ctr/memory.pressure",
            &psi("1.00", Some("0.00")),
        );
        assert_eq!(linux_pressure(&f.0), MemoryPressure::Normal);
        f.put(
            "sys/fs/cgroup/pod/memory.pressure",
            &psi("20.00", Some("0.00")),
        );
        assert_eq!(linux_pressure(&f.0), MemoryPressure::Warning);
        f.put(
            "sys/fs/cgroup/pod/ctr/memory.pressure",
            &psi("30.00", Some("15.00")),
        );
        assert_eq!(linux_pressure(&f.0), MemoryPressure::Critical);
        // A cgroup reading alone is enough when the system file is absent.
        let g = crate::testutil::Fixture::new("psi-cg");
        g.put("proc/self/cgroup", "0::/ctr\n");
        g.put(
            "sys/fs/cgroup/ctr/memory.pressure",
            &psi("11.00", Some("0.00")),
        );
        assert_eq!(linux_pressure(&g.0), MemoryPressure::Warning);
    }

    /// The DRM rule: no card, platform or virtio cards, and Intel on bus 0
    /// are unified; NVIDIA is discrete; a mix, an AMD card or an Intel card
    /// off bus 0 is unknown. Connectors (`card0-HDMI-A-1`) are not cards.
    #[cfg(unix)]
    #[test]
    fn linux_architecture_follows_the_drm_cards() {
        use std::os::unix::fs::symlink;
        /// A card: its DRM name, its PCI vendor (none for a platform GPU) and
        /// the device's sysfs address.
        type Card<'a> = (&'a str, Option<&'a str>, &'a str);
        let tree = |cards: &[Card<'_>]| {
            let f = crate::testutil::Fixture::new("drm");
            std::fs::create_dir_all(f.0.join("sys/class/drm")).unwrap();
            for (i, (name, vendor, addr)) in cards.iter().enumerate() {
                let dev = f.0.join(format!("sys/devices/pci0000:00/{i}/{addr}"));
                std::fs::create_dir_all(&dev).unwrap();
                if let Some(v) = vendor {
                    std::fs::write(dev.join("vendor"), format!("{v}\n")).unwrap();
                }
                let card = f.0.join("sys/class/drm").join(name);
                std::fs::create_dir_all(&card).unwrap();
                symlink(&dev, card.join("device")).unwrap();
            }
            f
        };
        use MemoryArchitecture::{Discrete, Unified, Unknown};
        let empty = crate::testutil::Fixture::new("drm-none");
        assert_eq!(linux_architecture(&empty.0), Unknown, "no sysfs at all");
        std::fs::create_dir_all(empty.0.join("sys/class/net")).unwrap();
        assert_eq!(
            linux_architecture(&empty.0),
            Unified,
            "sysfs, no DRM driver"
        );
        let cases: [(&[Card<'_>], MemoryArchitecture); 9] = [
            (&[], Unified),
            (&[("card0", None, "gpu")], Unified),
            (&[("card0", Some("0x1af4"), "0000:00:01.0")], Unified),
            (&[("card1", Some("0x8086"), "0000:00:02.0")], Unified),
            (&[("card0", Some("0x10de"), "0000:01:00.0")], Discrete),
            (&[("card0", Some("0x8086"), "0000:03:00.0")], Unknown),
            (&[("card0", Some("0x1002"), "0000:00:08.1")], Unknown),
            (
                &[
                    ("card0", Some("0x8086"), "0000:00:02.0"),
                    ("card1", Some("0x10de"), "0000:01:00.0"),
                ],
                Unknown,
            ),
            (
                &[
                    ("card0", Some("0x10de"), "0000:01:00.0"),
                    ("card0-HDMI-A-1", Some("0x8086"), "0000:03:00.0"),
                    ("renderD128", Some("0x1002"), "0000:04:00.0"),
                ],
                Discrete,
            ),
        ];
        for (cards, want) in cases {
            let f = tree(cards);
            assert_eq!(linux_architecture(&f.0), want, "{cards:?}");
        }
        assert!(on_pci_bus_zero(Path::new("../../../0000:00:02.0")));
        for bad in ["0000:01:00.0", "00:02.0", "0000:00:02", "x:00:02.0:1", ""] {
            assert!(!on_pci_bus_zero(Path::new(bad)), "{bad}");
        }
    }

    #[test]
    fn probing_from_many_threads_at_once_is_consistent() {
        let first = probe_system();
        std::thread::scope(|s| {
            let handles: Vec<_> = (0..16).map(|_| s.spawn(probe_system)).collect();
            for h in handles {
                let p = h.join().expect("probe panicked");
                // Memory figures move; the topology does not.
                assert_eq!(p.cpu, first.cpu);
                assert_eq!(p.architecture, first.architecture);
                assert_eq!(p.memory.total_bytes, first.memory.total_bytes);
            }
        });
    }
}

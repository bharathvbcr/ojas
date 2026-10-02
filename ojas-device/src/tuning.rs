//! Cache-derived tuning recommendations. Pure functions of probed numbers.
//!
//! These are recommendations. No kernel in the workspace switches to them
//! on its own; a backend adopts them through an explicit constructor and a
//! benchmark that shows the change pays.

use crate::host::MemoryReport;
use crate::topology::CpuTopology;

/// The cache sizes a blocked kernel can rely on from every core it may run
/// on: the smallest over the clusters, so a block sized for a fast core
/// still fits on a slow one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheBudget {
    /// L1 data cache per core.
    pub l1d: MemoryReport,
    /// One core's share of L2: the instance size over the CPUs that share
    /// it. When the sharing count is unknown the cluster's logical count is
    /// used, which is never an overestimate.
    pub l2_per_core: MemoryReport,
    /// L3, only when the OS reports one. Apple silicon reports none (its
    /// system-level cache is not exposed), so this is unknown there; the L2
    /// is never put in its place, because the A blocks already live in it.
    pub l3: MemoryReport,
}

impl CacheBudget {
    pub fn from_topology(topo: &CpuTopology) -> Self {
        let l2_per_core = topo.min_over_clusters(|c| {
            let sharing = match (c.cpus_per_l2, c.logical) {
                (MemoryReport::Known(n), _) if n > 0 => n,
                (_, MemoryReport::Known(n)) if n > 0 => n,
                _ => return MemoryReport::Unknown,
            };
            match c.l2_bytes {
                MemoryReport::Known(b) => MemoryReport::Known(b / sharing),
                MemoryReport::Unknown => MemoryReport::Unknown,
            }
        });
        Self {
            l1d: topo.min_over_clusters(|c| c.l1d_bytes),
            l2_per_core,
            l3: topo.l3_bytes,
        }
    }
}

/// Largest `kc` recommended.
pub const GEMM_KC_MAX: usize = 4096;
/// Largest `mc` recommended, before rounding to the register tile.
pub const GEMM_MC_MAX: usize = 4096;
/// Largest `nc` recommended, before rounding to the register tile.
pub const GEMM_NC_MAX: usize = 16384;
/// `kc` is a multiple of this.
pub const GEMM_KC_STEP: usize = 8;

/// Packed-GEMM block sizes, simplified from the analytical model of Low et
/// al. (2016): an `mr x kc` A sliver and a `kc x nr` B sliver share half of
/// L1 (the paper's set-associativity constraint is not modelled), an
/// `mc x kc` A block takes half of a core's L2 share, and a `kc x nc` B
/// panel takes half of L3.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemmBlocks {
    pub kc: usize,
    pub mc: usize,
    /// `None` when no L3 is reported: the model keeps the B panel in L3,
    /// and the L2 is already the A blocks' cache. The panel is sized as one
    /// shared by every core on the L3; a kernel whose threads each pack
    /// their own panel must divide it by the threads sharing that L3.
    pub nc: Option<usize>,
}

impl GemmBlocks {
    /// `None` when L1 or the L2 share is unknown, the register tile or
    /// element size is 0, or L1 is too small for an `(mr + nr) x 8` sliver
    /// pair.
    ///
    /// `mc` never drops below `mr` and `nc` never below `nr`: a cache too
    /// small for even one tile gets the smallest legal block, not none.
    pub fn derive(cache: &CacheBudget, mr: usize, nr: usize, elem_bytes: usize) -> Option<Self> {
        if mr == 0 || nr == 0 || elem_bytes == 0 {
            return None;
        }
        let l1 = known(cache.l1d)?;
        let l2 = known(cache.l2_per_core)?;
        let sliver = mr.checked_add(nr)?.checked_mul(elem_bytes)?;
        let kc = round_down(l1 / 2 / sliver, GEMM_KC_STEP).min(GEMM_KC_MAX);
        if kc < GEMM_KC_STEP {
            return None;
        }
        let row = kc.checked_mul(elem_bytes)?;
        let mc = round_down((l2 / 2 / row).min(GEMM_MC_MAX), mr).max(mr);
        let nc = known(cache.l3).map(|l3| round_down((l3 / 2 / row).min(GEMM_NC_MAX), nr).max(nr));
        Some(Self { kc, mc, nc })
    }
}

fn known(report: MemoryReport) -> Option<usize> {
    match report {
        MemoryReport::Known(n) => Some(usize::try_from(n).unwrap_or(usize::MAX)),
        MemoryReport::Unknown => None,
    }
}

fn round_down(n: usize, step: usize) -> usize {
    n - n % step
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::CoreCluster;

    fn budget(l1: u64, l2: u64, l3: u64) -> CacheBudget {
        CacheBudget {
            l1d: MemoryReport::Known(l1),
            l2_per_core: MemoryReport::Known(l2),
            l3: MemoryReport::Known(l3),
        }
    }

    fn cluster(l1: u64, l2: u64, per_l2: MemoryReport, logical: MemoryReport) -> CoreCluster {
        CoreCluster {
            name: "t".into(),
            physical: logical,
            logical,
            l1d_bytes: MemoryReport::Known(l1),
            l2_bytes: MemoryReport::Known(l2),
            cpus_per_l2: per_l2,
        }
    }

    #[test]
    fn m5_pro_numbers_give_blocks_that_fit() {
        // `sysctl hw.perflevel*` on the M5 Pro this was written on.
        let topo = CpuTopology {
            clusters: vec![
                cluster(
                    128 << 10,
                    16 << 20,
                    MemoryReport::Known(6),
                    MemoryReport::Known(6),
                ),
                cluster(
                    64 << 10,
                    8 << 20,
                    MemoryReport::Known(6),
                    MemoryReport::Known(12),
                ),
            ],
            ..CpuTopology::all_unknown()
        };
        let cache = CacheBudget::from_topology(&topo);
        assert_eq!(cache.l1d, MemoryReport::Known(64 << 10));
        assert_eq!(cache.l2_per_core, MemoryReport::Known((8 << 20) / 6));
        assert_eq!(cache.l3, MemoryReport::Unknown, "no L3 is reported");
        let b = GemmBlocks::derive(&cache, 6, 16, 4).unwrap();
        // No nc: sizing the B panel from the L2 put six 0.67 MiB A blocks
        // and a 4 MiB panel in one 8 MiB cache.
        assert_eq!(
            b,
            GemmBlocks {
                kc: 368,
                mc: 474,
                nc: None
            }
        );
    }

    #[test]
    fn a_reported_l3_sizes_the_b_panel_and_nothing_else_does() {
        let topo = CpuTopology {
            clusters: vec![cluster(
                48 << 10,
                2 << 20,
                MemoryReport::Known(2),
                MemoryReport::Known(16),
            )],
            l3_bytes: MemoryReport::Known(32 << 20),
            ..CpuTopology::all_unknown()
        };
        let cache = CacheBudget::from_topology(&topo);
        assert_eq!(cache.l3, MemoryReport::Known(32 << 20));
        let b = GemmBlocks::derive(&cache, 6, 16, 4).unwrap();
        let nc = b.nc.expect("L3 reported");
        assert!(b.kc * nc * 4 <= (32 << 20) / 2, "{b:?}");
        // The same machine without its L3 keeps kc and mc and loses nc,
        // however large its L2 is.
        for l2 in [2u64 << 20, 64 << 20, u64::MAX] {
            let no_l3 = CacheBudget {
                l3: MemoryReport::Unknown,
                ..budget(48 << 10, l2, 0)
            };
            let c = GemmBlocks::derive(&no_l3, 6, 16, 4).unwrap();
            assert_eq!((c.kc, c.nc), (b.kc, None), "l2 {l2}");
        }
    }

    #[test]
    fn unknowns_and_zero_tiles_give_none() {
        let ok = budget(32 << 10, 1 << 20, 8 << 20);
        assert!(GemmBlocks::derive(&ok, 6, 16, 4).is_some());
        assert_eq!(GemmBlocks::derive(&ok, 0, 16, 4), None);
        assert_eq!(GemmBlocks::derive(&ok, 6, 0, 4), None);
        assert_eq!(GemmBlocks::derive(&ok, 6, 16, 0), None);
        for hole in 0..2 {
            let mut c = ok;
            match hole {
                0 => c.l1d = MemoryReport::Unknown,
                _ => c.l2_per_core = MemoryReport::Unknown,
            }
            assert_eq!(GemmBlocks::derive(&c, 6, 16, 4), None, "hole {hole}");
        }
        let no_l3 = CacheBudget {
            l3: MemoryReport::Unknown,
            ..ok
        };
        assert_eq!(GemmBlocks::derive(&no_l3, 6, 16, 4).unwrap().nc, None);
        // L1 too small for an 8-deep sliver pair.
        assert_eq!(
            GemmBlocks::derive(&budget(1024, 1 << 20, 8 << 20), 6, 16, 4),
            None
        );
        // Overflowing tile arithmetic is None, not a panic.
        assert_eq!(GemmBlocks::derive(&ok, usize::MAX, 1, 4), None);
        assert_eq!(GemmBlocks::derive(&ok, 6, 16, usize::MAX), None);
    }

    #[test]
    fn unknown_sharing_falls_back_to_the_cluster_and_then_to_unknown() {
        let topo = CpuTopology {
            clusters: vec![cluster(
                32 << 10,
                4 << 20,
                MemoryReport::Unknown,
                MemoryReport::Known(4),
            )],
            ..CpuTopology::all_unknown()
        };
        assert_eq!(
            CacheBudget::from_topology(&topo).l2_per_core,
            MemoryReport::Known(1 << 20)
        );
        let topo = CpuTopology {
            clusters: vec![cluster(
                32 << 10,
                4 << 20,
                MemoryReport::Unknown,
                MemoryReport::Unknown,
            )],
            ..CpuTopology::all_unknown()
        };
        assert_eq!(
            CacheBudget::from_topology(&topo).l2_per_core,
            MemoryReport::Unknown
        );
        assert_eq!(
            CacheBudget::from_topology(&CpuTopology::all_unknown()),
            CacheBudget {
                l1d: MemoryReport::Unknown,
                l2_per_core: MemoryReport::Unknown,
                l3: MemoryReport::Unknown,
            }
        );
    }

    /// The model's capacity constraints hold for every block it returns, and
    /// each block grows (never shrinks) with its own cache.
    #[test]
    fn derived_blocks_fit_and_are_monotone_in_their_own_cache() {
        let l1s = [
            2u64 << 10,
            16 << 10,
            32 << 10,
            48 << 10,
            64 << 10,
            128 << 10,
            1 << 20,
            u64::MAX,
        ];
        let l2s = [
            0u64,
            64 << 10,
            256 << 10,
            1 << 20,
            2 << 20,
            16 << 20,
            u64::MAX,
        ];
        let l3s = [0u64, 1 << 20, 8 << 20, 32 << 20, 256 << 20, u64::MAX];
        let tiles = [(1usize, 1usize), (4, 4), (6, 16), (8, 12), (16, 32)];
        for &(mr, nr) in &tiles {
            for eb in [2usize, 4, 8] {
                for &l1 in &l1s {
                    let mut prev_mc = 0;
                    for &l2 in &l2s {
                        let mut prev_nc = 0;
                        for &l3 in &l3s {
                            let c = budget(l1, l2, l3);
                            let Some(b) = GemmBlocks::derive(&c, mr, nr, eb) else {
                                assert!(
                                    ((l1 / 2) as usize) / ((mr + nr) * eb) < GEMM_KC_STEP,
                                    "only a tiny L1 may refuse: {c:?} {mr} {nr} {eb}"
                                );
                                continue;
                            };
                            let l1 = usize::try_from(l1).unwrap_or(usize::MAX);
                            let l2 = usize::try_from(l2).unwrap_or(usize::MAX);
                            let l3 = usize::try_from(l3).unwrap_or(usize::MAX);
                            assert!(b.kc % GEMM_KC_STEP == 0 && b.kc >= GEMM_KC_STEP);
                            assert!(b.kc <= GEMM_KC_MAX);
                            assert!((mr + nr) * b.kc * eb <= l1 / 2, "{b:?}");
                            assert!(b.mc % mr == 0 && b.mc >= mr && b.mc <= GEMM_MC_MAX.max(mr));
                            assert!(b.mc == mr || b.mc * b.kc * eb <= l2 / 2, "{b:?} l2 {l2}");
                            let nc = b.nc.expect("a known L3 always sizes nc");
                            assert!(nc % nr == 0 && nc >= nr && nc <= GEMM_NC_MAX.max(nr));
                            assert!(nc == nr || b.kc * nc * eb <= l3 / 2, "{b:?} l3 {l3}");
                            assert!(nc >= prev_nc, "nc shrank as L3 grew");
                            prev_nc = nc;
                        }
                        let b = GemmBlocks::derive(&budget(l1, l2, 8 << 20), mr, nr, eb);
                        if let Some(b) = b {
                            assert!(b.mc >= prev_mc, "mc shrank as L2 grew");
                            prev_mc = b.mc;
                        }
                    }
                }
            }
        }
    }
}

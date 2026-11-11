//! Deterministic scaling-law `e-means` clustering (`canonical_v1`) over CPU SIMD and Metal 4.

#![forbid(unsafe_code)]

pub mod cpu;
pub mod lloyd;
mod metal;
pub mod recipe;
pub mod rng;
pub mod session;

pub use cpu::{assign_cpu_bf16, assign_cpu_f32, assign_cpu_tensor, fit_cpu, CpuPool, CpuSession};
pub use lloyd::{lloyd_fit, LloydConfig, LloydFit};
pub use recipe::{resolve_recipe, round_power_of_two, scheduled_alpha};
pub use rng::{
    choose_splits, epoch_permutation, gumbel_key, prescaled_unit_normalize_f32,
    prescaled_unit_normalize_f64, sample_subset_sorted, separate_collapsed_split_bf16,
    separate_collapsed_split_f32, SplitMix64,
};
pub use session::{emeans_assign, emeans_fit, EmeansEngine, EmeansSession};

/// Sentinel cluster label assigned to rows excluded by `max_points_per_centroid` subsampling.
pub const UNASSIGNED_LABEL: u32 = u32::MAX;

/// Numerical precision lane for Apple Silicon Metal 4 GPU execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MetalPrecision {
    /// Rounds rows once to `bf16`; norms, TensorOps dot products, batch sums, and closing sums
    /// all read the `bf16`-rounded values.
    #[default]
    Bf16,
    /// Runs `bf16` TensorOps assignment while accumulating batch sums and closing means from
    /// unrounded `f32` coordinates.
    Bf16AssignF32Stats,
    /// Keeps operands, cooperative `f32` TensorOps dot products, sums, and closing means in `f32`.
    F32,
}

/// Numerical reduction and GEMM mode for CPU execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CpuMath {
    /// Uses Apple Accelerate AMX when unpinned on macOS, or L2-tiled `ojas-simd` micro-kernels.
    #[default]
    Fast,
    /// Thread-count-independent L2-tiled `ojas-simd` micro-kernels and ascending CSR gather.
    DeterministicFma,
    /// Thread-count-independent non-FMA exact `f32` dot products and ascending CSR gather.
    Exact,
}

/// Empirical mini-batch size scaling law from `canonical_v1` (`faiss/contrib/e_means`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BatchLaw {
    /// Resolves to [`BatchLaw::Quality`] when `d <= 16`, else [`BatchLaw::Throughput`].
    #[default]
    Auto,
    /// High-throughput batch scaling law (`C_B = 0.035134`, power-of-two half-`n` cap).
    Throughput,
    /// Fine-grained low-dimensional batch scaling law (`C_B = 0.00439175`, `n / 2` cap).
    Quality,
}

/// Target compute substrate for clustering. Never falls back silently across devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClusterDevice {
    /// Multi-core CPU execution (`threads: None` uses available parallelism up to ceiling).
    Cpu {
        threads: Option<usize>,
        math: CpuMath,
    },
    /// Apple Silicon Metal 4 GPU execution.
    Metal { precision: MetalPrecision },
}

impl Default for ClusterDevice {
    fn default() -> Self {
        Self::Cpu {
            threads: None,
            math: CpuMath::Fast,
        }
    }
}

/// User-facing configuration for `e-means` clustering.
#[derive(Debug, Clone, PartialEq)]
pub struct EmeansConfig {
    pub k: usize,
    pub nepochs: f64,
    pub batch_law: BatchLaw,
    pub batch_size: Option<usize>,
    pub alpha0: Option<f32>,
    pub spherical: bool,
    pub seed: u64,
    pub max_points_per_centroid: Option<usize>,
    pub assignment_chunk_size: usize,
    pub unused_threshold: Option<f32>,
    pub split_epsilon: f32,
    pub init_usage: f32,
    pub capture_assignments: bool,
}

impl EmeansConfig {
    /// Construct a default `canonical_v1` configuration for `k` clusters.
    pub fn new(k: usize) -> Self {
        Self {
            k,
            nepochs: 2.6,
            batch_law: BatchLaw::Auto,
            batch_size: None,
            alpha0: None,
            spherical: false,
            seed: 42,
            max_points_per_centroid: None,
            assignment_chunk_size: 16_384,
            unused_threshold: None,
            split_epsilon: 1.0e-4,
            init_usage: 1.0 / 65_536.0,
            capture_assignments: true,
        }
    }
}

impl Default for EmeansConfig {
    fn default() -> Self {
        Self::new(1)
    }
}

/// Fully validated and resolved `canonical_v1` execution schedule for a dataset of `[global_n, d]`.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedRecipe {
    pub global_n: usize,
    pub effective_n: usize,
    pub k: usize,
    pub d: usize,
    pub nepochs: f64,
    pub resolved_batch_law: BatchLaw,
    pub batch_size: usize,
    pub alpha0: f32,
    pub steps_per_pass: usize,
    pub epoch_rows: Vec<usize>,
    pub epoch_steps: Vec<usize>,
    pub total_steps: usize,
    pub assignment_chunk_size: usize,
    pub unused_threshold: f32,
    pub split_epsilon: f32,
    pub init_usage: f32,
    pub spherical: bool,
    pub seed: u64,
}

/// Progress snapshot returned after streaming rows into an active mini-batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccumulateStep {
    pub rows_accepted: usize,
    pub remaining_in_batch: usize,
    pub remaining_in_epoch: usize,
    pub batch_ready: bool,
}

/// Per-epoch diagnostic summary recorded at the end of each pass (or fractional pass).
#[derive(Debug, Clone, PartialEq)]
pub struct EpochStat {
    pub epoch: usize,
    pub rows_seen: usize,
    pub batch_updates: usize,
    pub mean_objective: f64,
    pub revived_clusters: usize,
}

/// Completed `e-means` clustering result.
#[derive(Debug, Clone)]
pub struct EmeansFit {
    pub centroids: ojas_core::Tensor,
    pub labels: Vec<u32>,
    pub cluster_counts: Vec<u64>,
    pub usage: Vec<f32>,
    pub epoch_stats: Vec<EpochStat>,
    pub recipe: ResolvedRecipe,
}

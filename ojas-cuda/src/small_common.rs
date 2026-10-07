//! What lane L-cuda-small's kernels share: K3 [`crate::gates_published`], K4
//! [`crate::conv1d`], K6 [`crate::qk_norm_rope`], K7 [`crate::rmsnorm`], K9
//! [`crate::embed`] and K10 [`crate::ce_rows`] (`cuda-backend-scoping.md` §3).
//!
//! # The reductions (the public API other lanes converge on)
//!
//! Every float reduction in these kernels has a fixed shape, so its bits do
//! not depend on the grid, the SM count or scheduling (§3 determinism rule):
//!
//! - **Per-thread partials.** Thread `t` of a `threads`-wide block folds
//!   elements `t, t + threads, t + 2*threads, ...` in ascending order from
//!   `+0.0` ([`thread_partials`]).
//! - **Warp butterfly** (`qd_warp_sum` / `qd_warp_max`): `v = v op
//!   shfl_xor(v, m)` for `m = 16, 8, 4, 2, 1`. Every lane ends with the same
//!   bits, because each step combines the same two values in both lanes and
//!   IEEE `+` and `max` commute ([`warp_sum`], [`warp_max`]).
//! - **Block** (`qd_block_sum` / `qd_block_max`): the warp butterfly, then
//!   lane 0 of each warp publishes its value to shared memory, then every warp
//!   folds the published values (padding past the last warp with `+0.0` or
//!   `-FLT_MAX`) with a second butterfly ([`block_sum`], [`block_max`]).
//! - **Weight gradients over rows**: fixed-size row blocks write per-block
//!   partials; `qd_col_sum_blocks_f32` ([`COL_SUM`]) then sums the blocks in
//!   ascending block order, one thread per column ([`col_sum_blocks`]). No
//!   atomics anywhere. tessl does the same on Metal
//!   (`tessl/kernels/qwen35_bwd.metal:12-15,108-124`).
//!
//! The host functions here are **bit-exact mirrors** of those device
//! sequences: the kernels are compiled with `--fmad=false`
//! ([`crate::kernels::STRICT_SM90`]), so every `+` and `*` is one IEEE
//! rounding, as in Rust. A host mirror and a device result can therefore be
//! compared bitwise wherever no libdevice transcendental is involved.
//!
//! # The device prelude
//!
//! [`small_prelude!`] is the CUDA-C text every module of this lane carries:
//! the reductions above and the `-FLT_MAX` identity. It holds **no**
//! activation function and no grid-stride loop (lead's rulings, 2026-10-01):
//! the crate's one device sigmoid / SiLU / exp / log is L-cuda-M1's
//! `crate::act_prelude!()` (`k8_act`), and the one `QD_GRID_STRIDE` is
//! `crate::device_prelude!()` (`kernels`). A module splices each it needs,
//! once.
//!
//! # Also here
//!
//! - [`Window`]: a column window `[off, off + width)` of a row-major matrix
//!   with row stride `ld`, the layout every fused-projection operand uses
//!   (tessl's `Cols`, `tessl/src/qwen35.rs`).
//! - Launch geometry for block-per-row and warp-per-unit kernels, from the
//!   shape only ([`block_grid`]).
//! - [`elementwise_check`] and [`peak_check`], the two tolerance shapes
//!   tessl's tests use (`tessl/tests/qwen35_kernels.rs:49-60`,
//!   `tessl/tests/qwen35_bwd.rs:43-66`).
//! - The source lint every module of this lane passes, and on macOS a test
//!   that parses and type-checks every source as host C++ with the system
//!   `clang++` (this host has no CUDA compiler). That check says nothing about
//!   numerics or execution; it catches typos before the box does.

use crate::check::Check;
use crate::error::CudaError;
use crate::geometry::{Launch, MAX_GRID_X};
use crate::k0_plan::ColWindow;
use crate::kernels::KernelModule;

/// Threads per block of every block-per-row kernel (the row norms, the
/// cross-entropy log-sum-exp).
pub const ROW_THREADS: u32 = 256;

/// Lanes per warp.
pub const WARP: u32 = 32;

/// Warps per block of every warp-per-unit kernel (the gated norm, the q/k
/// norm + RoPE), as tessl's `GATED_BWD_SG` / `QK_BWD_SG` (4 simdgroups).
pub const UNIT_WARPS: u32 = 4;

/// Threads per block of every warp-per-unit kernel.
pub const UNIT_THREADS: u32 = UNIT_WARPS * WARP;

/// Columns one thread may own in a weight-gradient kernel: tessl's
/// `BWD_MAX_COLS` (`tessl/kernels/qwen35_bwd.metal:25`). The kernels keep that
/// many per-thread accumulators in registers.
pub const MAX_COLS: u64 = 16;

/// The running-max identity, `-FLT_MAX` (tessl `cross_entropy.metal:51-53`).
pub const NEG_FLT_MAX: f32 = -f32::MAX;

/// Largest count a kernel converts to `float` (`(float)dim`, `(float)pos`):
/// every integer up to 2^24 is exact in f32.
pub const MAX_EXACT_F32_INT: u64 = 1 << 24;

/// The CUDA-C prelude of this lane's modules. A macro so `concat!` can splice
/// one copy into each NVRTC compilation unit (each module is its own).
macro_rules! small_prelude {
    () => {
        r#"
// ---- L-cuda-small prelude (src/small_common.rs) ----
#define QD_FULL_MASK 0xffffffffu
// -FLT_MAX, the running-max identity (tessl cross_entropy.metal:51-53).
#define QD_NEG_FLT_MAX (-3.40282347e+38f)

// Butterfly over the 32 lanes. Each step adds the same two values in both
// lanes it pairs, and IEEE addition commutes, so every lane ends with the
// same bits. small_common::warp_sum is this sequence on the host.
__device__ __forceinline__ float qd_warp_sum(float v)
{
    v = v + __shfl_xor_sync(QD_FULL_MASK, v, 16);
    v = v + __shfl_xor_sync(QD_FULL_MASK, v, 8);
    v = v + __shfl_xor_sync(QD_FULL_MASK, v, 4);
    v = v + __shfl_xor_sync(QD_FULL_MASK, v, 2);
    v = v + __shfl_xor_sync(QD_FULL_MASK, v, 1);
    return v;
}

__device__ __forceinline__ float qd_warp_max(float v)
{
    v = fmaxf(v, __shfl_xor_sync(QD_FULL_MASK, v, 16));
    v = fmaxf(v, __shfl_xor_sync(QD_FULL_MASK, v, 8));
    v = fmaxf(v, __shfl_xor_sync(QD_FULL_MASK, v, 4));
    v = fmaxf(v, __shfl_xor_sync(QD_FULL_MASK, v, 2));
    v = fmaxf(v, __shfl_xor_sync(QD_FULL_MASK, v, 1));
    return v;
}

// Sum over the whole block, returned in every thread. blockDim.x must be a
// multiple of 32 and at most 1024, and every thread must call it (it holds
// barriers). `scratch` is 32 floats of shared memory; the leading barrier
// lets consecutive calls reuse it. small_common::block_sum mirrors it.
__device__ __forceinline__ float qd_block_sum(float v, float* scratch)
{
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int nwarps = blockDim.x >> 5;
    v = qd_warp_sum(v);
    if (nwarps == 1u) {
        return v;
    }
    __syncthreads();
    if (lane == 0u) {
        scratch[warp] = v;
    }
    __syncthreads();
    return qd_warp_sum(lane < nwarps ? scratch[lane] : 0.0f);
}

__device__ __forceinline__ float qd_block_max(float v, float* scratch)
{
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int nwarps = blockDim.x >> 5;
    v = qd_warp_max(v);
    if (nwarps == 1u) {
        return v;
    }
    __syncthreads();
    if (lane == 0u) {
        scratch[warp] = v;
    }
    __syncthreads();
    return qd_warp_max(lane < nwarps ? scratch[lane] : QD_NEG_FLT_MAX);
}

// Each of N per-thread values summed over a 128-thread block (four warps);
// every thread receives all N totals. Per value: the warp butterfly, lane 0
// of each warp to `red`, then (w0 + w1) + (w2 + w3) -- not qd_block_sum's
// second butterfly, which pairs (w0 + w2) + (w1 + w3). `red` holds 4 * N
// floats. Callers alternate between two `red` buffers: the barrier here
// orders this call's writes after every thread's reads of the other buffer.
// No multiply feeds these adds, so --fmad cannot contract them; each is one
// IEEE rounding. small_common::colsum_4warps mirrors it.
template <int N>
__device__ __forceinline__ void qd_colsum_4warps(const float (&p)[N], float* red,
                                                 unsigned int warp, unsigned int lane,
                                                 float (&out)[N])
{
    #pragma unroll
    for (int n = 0; n < N; ++n) {
        const float s = qd_warp_sum(p[n]);
        if (lane == 0u) {
            red[warp * N + n] = s;
        }
    }
    __syncthreads();
    #pragma unroll
    for (int n = 0; n < N; ++n) {
        out[n] = (red[n] + red[N + n]) + (red[2 * N + n] + red[3 * N + n]);
    }
}
// ---- end L-cuda-small prelude ----
"#
    };
}
pub(crate) use small_prelude;

/// The block-order column sum: `out[c] = sum over b of part[part_off + b*dim + c]`,
/// blocks ascending from `+0.0`, one thread per column.
pub const COL_SUM: KernelModule = KernelModule {
    name: "small_col_sum",
    source: COL_SUM_SOURCE,
    entries: &[COL_SUM_ENTRY],
};

/// The entry point of [`COL_SUM`].
pub const COL_SUM_ENTRY: &str = "qd_col_sum_blocks_f32";

const COL_SUM_SOURCE: &str = concat!(
    crate::device_prelude!(),
    small_prelude!(),
    r#"
extern "C" __global__ void qd_col_sum_blocks_f32(
    const float* part, unsigned long long part_off, float* out,
    unsigned long long blocks, unsigned long long dim)
{
    QD_GRID_STRIDE(c, dim) {
        float s = 0.0f;
        for (unsigned long long b = 0; b < blocks; ++b) {
            s = s + part[part_off + b * dim + c];
        }
        out[c] = s;
    }
}
"#
);

/// Every NVRTC module of this lane, for the source lint and the compile check.
pub const SMALL_MODULES: [&KernelModule; 7] = [
    &COL_SUM,
    &crate::ce_rows::MODULE,
    &crate::conv1d::MODULE,
    &crate::embed::MODULE,
    &crate::gates_published::MODULE,
    &crate::qk_norm_rope::MODULE,
    &crate::rmsnorm::MODULE,
];

// ------------------------------------------------------------- arithmetic ---

/// `usize` to `u64`; lossless on every target this crate builds for.
pub fn to_u64(x: usize) -> u64 {
    u64::try_from(x).unwrap_or(u64::MAX)
}

/// `a * b`, refused on overflow.
pub fn mul(a: u64, b: u64, op: &str) -> Result<u64, CudaError> {
    a.checked_mul(b)
        .ok_or_else(|| CudaError::invalid(op, format!("{a} * {b} overflows u64")))
}

/// `a + b`, refused on overflow.
pub fn add(a: u64, b: u64, op: &str) -> Result<u64, CudaError> {
    a.checked_add(b)
        .ok_or_else(|| CudaError::invalid(op, format!("{a} + {b} overflows u64")))
}

/// `n` as a host index; refused where `usize` is narrower than the value.
pub fn to_usize(n: u64, op: &str) -> Result<usize, CudaError> {
    usize::try_from(n).map_err(|_| CudaError::invalid(op, format!("{n} does not fit usize")))
}

/// `ceil(n / per)`; `per` must be non-zero (every caller passes a constant).
pub fn blocks_for(n: u64, per: u64) -> u64 {
    n.div_ceil(per.max(1))
}

/// Refuse a zero size: a zero-element device buffer cannot be allocated
/// ([`crate::runtime`]'s reserve), so callers with nothing to do skip the call.
pub fn nonzero(op: &str, what: &str, v: u64) -> Result<(), CudaError> {
    if v == 0 {
        return Err(CudaError::invalid(
            op,
            format!("{what} is 0; skip the call instead"),
        ));
    }
    Ok(())
}

/// Refuse a count the kernel turns into `float` unless it is exact there.
pub fn exact_in_f32(op: &str, what: &str, v: u64) -> Result<(), CudaError> {
    if v > MAX_EXACT_F32_INT {
        return Err(CudaError::invalid(
            op,
            format!("{what} = {v} exceeds 2^24, past which (float){what} is not exact"),
        ));
    }
    Ok(())
}

/// Refuse a non-finite or non-positive epsilon.
pub fn positive_eps(op: &str, eps: f32) -> Result<(), CudaError> {
    if !(eps.is_finite() && eps > 0.0) {
        return Err(CudaError::invalid(
            op,
            format!("eps must be finite and positive, got {eps}"),
        ));
    }
    Ok(())
}

/// Check that a buffer holds exactly `want` elements.
pub fn exact_len(op: &str, name: &str, got: usize, want: u64) -> Result<(), CudaError> {
    if to_u64(got) != want {
        return Err(CudaError::invalid(
            op,
            format!("{name} has {got} elements, expected {want}"),
        ));
    }
    Ok(())
}

// ----------------------------------------------------------------- windows ---

/// Columns `[off, off + width)` of each row of a row-major matrix with row
/// stride `ld`. A window must lie inside one row: one that crosses into the
/// next row is a layout bug, not an operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Window {
    /// Row stride, in elements.
    pub ld: u64,
    /// First column.
    pub off: u64,
}

impl Window {
    /// A dense matrix of `width` columns.
    pub const fn dense(width: u64) -> Self {
        Window { ld: width, off: 0 }
    }

    /// Validate a `rows x width` window against a buffer of `len` elements:
    /// inside one row and inside the buffer. The crate's one window check is
    /// [`ColWindow::new`] (K0's); this delegates to it.
    pub fn check(
        self,
        op: &str,
        name: &str,
        rows: u64,
        width: u64,
        len: usize,
    ) -> Result<(), CudaError> {
        ColWindow::new(op, name, rows, width, (self.ld, self.off, len)).map(|_| ())
    }

    /// Host index of element `(r, c)` of the window. Only for windows that
    /// passed [`Window::check`], whose every index fits the buffer's length.
    pub fn at(self, r: usize, c: usize) -> usize {
        let ld = usize::try_from(self.ld).unwrap_or(usize::MAX);
        let off = usize::try_from(self.off).unwrap_or(usize::MAX);
        r * ld + off + c
    }
}

/// Whether column ranges `[a, a + aw)` and `[b, b + bw)` share a column.
pub fn cols_overlap((a, aw): (u64, u64), (b, bw): (u64, u64)) -> bool {
    a < b.saturating_add(bw) && b < a.saturating_add(aw)
}

// ---------------------------------------------------------------- geometry ---

/// `blocks` blocks of `threads` threads in x: the shape of every block-per-row
/// and warp-per-unit kernel here. A function of the problem only, never of the
/// SM count.
pub fn block_grid(blocks: u64, threads: u32, op: &str) -> Result<Launch, CudaError> {
    nonzero(op, "the block count", blocks)?;
    let gx = u32::try_from(blocks)
        .ok()
        .filter(|&g| g <= MAX_GRID_X)
        .ok_or_else(|| {
            CudaError::invalid(
                op,
                format!("{blocks} blocks exceed gridDim.x's {MAX_GRID_X}"),
            )
        })?;
    Ok(Launch {
        grid: (gx, 1, 1),
        block: (threads, 1, 1),
    })
}

// ------------------------------------------------------- reduction mirrors ---

/// `qd_warp_sum` on the host: the butterfly over 32 lanes. Returns lane 0's
/// value (every lane holds the same bits).
pub fn warp_sum(lanes: [f32; 32]) -> f32 {
    butterfly(lanes, |a, b| a + b)
}

/// Threads in the block [`colsum_4warps`] reduces over: four warps.
pub const COLSUM_4WARPS_THREADS: usize = 128;

/// `qd_colsum_4warps<N>` on the host: each column over 128 threads, as four
/// [`warp_sum`] butterflies then `(w0 + w1) + (w2 + w3)`.
pub fn colsum_4warps<const N: usize>(p: &[[f32; N]; COLSUM_4WARPS_THREADS]) -> [f32; N] {
    std::array::from_fn(|n| {
        let w: [f32; 4] =
            std::array::from_fn(|wi| warp_sum(std::array::from_fn(|l| p[wi * 32 + l][n])));
        (w[0] + w[1]) + (w[2] + w[3])
    })
}

/// `qd_warp_max` on the host (`fmaxf` and `f32::max` both drop a NaN operand).
pub fn warp_max(lanes: [f32; 32]) -> f32 {
    butterfly(lanes, f32::max)
}

fn butterfly(mut v: [f32; 32], op: impl Fn(f32, f32) -> f32) -> f32 {
    for m in [16usize, 8, 4, 2, 1] {
        let prev = v;
        for (i, slot) in v.iter_mut().enumerate() {
            *slot = op(prev[i], prev[i ^ m]);
        }
    }
    v[0]
}

fn block_reduce(per_thread: &[f32], pad: f32, warp: fn([f32; 32]) -> f32, what: &str) -> f32 {
    let n = per_thread.len();
    assert!(
        n > 0 && n.is_multiple_of(32) && n <= 1024,
        "{what}: a block of {n} threads is not a whole number of warps up to 1024"
    );
    // n is a multiple of 32 (asserted above), so as_chunks leaves no remainder.
    let warps: Vec<f32> = per_thread
        .as_chunks::<32>()
        .0
        .iter()
        .map(|c| warp(*c))
        .collect();
    if warps.len() == 1 {
        return warps[0];
    }
    let mut lanes = [pad; 32];
    lanes[..warps.len()].copy_from_slice(&warps);
    warp(lanes)
}

/// `qd_block_sum` on the host, given each thread's value (`blockDim.x` of them).
pub fn block_sum(per_thread: &[f32]) -> f32 {
    block_reduce(per_thread, 0.0, warp_sum, "block_sum")
}

/// `qd_block_max` on the host.
pub fn block_max(per_thread: &[f32]) -> f32 {
    block_reduce(per_thread, NEG_FLT_MAX, warp_max, "block_max")
}

/// Each thread's ascending fold of `term(i)` over `i = t, t + threads, ...`,
/// `i < n`, from `init`: the per-thread loop every reduction kernel starts with.
pub fn thread_partials(
    threads: usize,
    n: usize,
    init: f32,
    op: impl Fn(f32, f32) -> f32,
    term: impl Fn(usize) -> f32,
) -> Vec<f32> {
    (0..threads)
        .map(|t| {
            (t..n)
                .step_by(threads)
                .fold(init, |acc, i| op(acc, term(i)))
        })
        .collect()
}

/// [`thread_partials`] summed by [`block_sum`]: the whole device row sum.
pub fn row_sum(threads: usize, n: usize, term: impl Fn(usize) -> f32) -> f32 {
    block_sum(&thread_partials(threads, n, 0.0, |a, b| a + b, term))
}

/// [`thread_partials`] (max, from `-FLT_MAX`) reduced by [`block_max`].
pub fn row_max(threads: usize, n: usize, term: impl Fn(usize) -> f32) -> f32 {
    block_max(&thread_partials(threads, n, NEG_FLT_MAX, f32::max, term))
}

/// [`thread_partials`] over one warp, reduced by [`warp_sum`]: the per-unit
/// sum of the warp-per-unit kernels.
pub fn unit_sum(n: usize, term: impl Fn(usize) -> f32) -> f32 {
    let p = thread_partials(32, n, 0.0, |a, b| a + b, term);
    let mut lanes = [0.0f32; 32];
    lanes.copy_from_slice(&p);
    warp_sum(lanes)
}

/// `qd_col_sum_blocks_f32` on the host: `out[c] = sum over b < blocks of
/// part[part_off + b*dim + c]`, ascending from `+0.0`.
pub fn col_sum_blocks(part: &[f32], part_off: usize, blocks: usize, dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|c| (0..blocks).fold(0.0f32, |s, b| s + part[part_off + b * dim + c]))
        .collect()
}

// ------------------------------------------------------------------ bounds ---
//
// tessl's own bounds for the same kernels on Metal against float64, adopted
// before any CUDA run. The device tests hold the kernels to them against
// L-cuda-oracle's references; rung a's checks (`small_smoke`) hold the device
// to the mirrors with them where the mirror is not bitwise.

/// Norm and conv forwards: `|err| <= 1e-6 + 1e-5 |ref|`, as `(rel, abs)`.
pub const NORM_FWD_TOL: (f64, f64) = (1e-5, 1e-6);
/// Where [`NORM_FWD_TOL`] comes from.
pub const NORM_FWD_SOURCE: &str = "tessl/tests/qwen35_kernels.rs:1071,1118,1174";
/// The attention output gate's forward: `|err| <= 1e-7 + 1e-5 |ref|`.
pub const GATE_FWD_TOL: (f64, f64) = (1e-5, 1e-7);
/// Where [`GATE_FWD_TOL`] comes from.
pub const GATE_FWD_SOURCE: &str = "tessl/tests/qwen35_kernels.rs:1617";
/// The q/k norm + RoPE forward: `|err| <= 2e-5`.
pub const QK_FWD_ABS: f64 = 2e-5;
/// Where [`QK_FWD_ABS`] comes from.
pub const QK_FWD_SOURCE: &str = "tessl/tests/qwen35_kernels.rs:1556,1562";
/// Every backward (and the K3 gates forward): `max |err| <= 1e-4 max |ref|`.
pub const BWD_PEAK_REL: f64 = 1e-4;
/// Where [`BWD_PEAK_REL`] comes from.
pub const BWD_PEAK_SOURCE: &str = "tessl/tests/qwen35_bwd.rs:5,43-66";
/// The cross-entropy per-row loss and loss: `|err| <= 1e-5 + 1e-5 |ref|`,
/// as `(abs, rel)`.
pub const CE_LOSS_TOL: (f64, f64) = (1e-5, 1e-5);
/// The cross-entropy gradients at exact f32 operands, of the peak.
pub const CE_EXACT_GRAD_PEAK_REL: f64 = 1e-4;
/// The cross-entropy gradients at bf16 operands: `2^-7` of the peak.
pub const CE_BF16_GRAD_PEAK_REL: f64 = 0.0078125;
/// Where the three cross-entropy bounds come from.
pub const CE_SOURCE: &str = "tessl/tests/cross_entropy.rs:12-14,185,252-258";

// ------------------------------------------------------------------ checks ---

/// `x` widened to f64 (exact).
pub fn f64s(x: &[f32]) -> Vec<f64> {
    x.iter().map(|&v| f64::from(v)).collect()
}

/// tessl's elementwise bound (`tessl/tests/qwen35_kernels.rs:49-60`): every
/// output finite and `|got - want| <= abs + rel * |want|`.
pub fn elementwise_check(
    name: &str,
    got: &[f32],
    want: &[f64],
    (rel, abs): (f64, f64),
    source: &str,
) -> Check {
    if got.len() != want.len() {
        return Check::fail(
            name,
            format!("{} results against {} references", got.len(), want.len()),
        );
    }
    let (mut nonfinite, mut worst_ratio, mut worst_at, mut max_err) =
        (0usize, 0.0f64, 0usize, 0.0f64);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        if !g.is_finite() {
            nonfinite += 1;
            continue;
        }
        let e = (f64::from(g) - w).abs();
        max_err = max_err.max(e);
        let ratio = e / (abs + rel * w.abs());
        if ratio > worst_ratio {
            worst_ratio = ratio;
            worst_at = i;
        }
    }
    let detail = format!(
        "worst |err| / (abs + rel|ref|) = {worst_ratio:.3e} at {worst_at}; max |err| {max_err:.3e}; \
         bound {abs:.0e} + {rel:.0e}|ref| ({source}); {nonfinite} non-finite"
    );
    let check = if nonfinite == 0 && worst_ratio <= 1.0 {
        Check::pass(name, detail)
    } else {
        Check::fail(name, detail)
    };
    check
        .with("worst_ratio", worst_ratio)
        .with("worst_at", worst_at)
        .with("max_abs_err", max_err)
        .with("nonfinite", nonfinite)
        .with("bound_rel", rel)
        .with("bound_abs", abs)
        .with("bound_source", source)
}

/// tessl's peak-relative bound (`tessl/tests/qwen35_bwd.rs:43-66`): every
/// output finite and `max |got - want| <= rel * max |want|`. An all-zero
/// reference demands exact zeros.
pub fn peak_check(name: &str, got: &[f32], want: &[f64], rel: f64, source: &str) -> Check {
    if got.len() != want.len() {
        return Check::fail(
            name,
            format!("{} results against {} references", got.len(), want.len()),
        );
    }
    let peak = want.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let (mut nonfinite, mut max_err, mut at) = (0usize, 0.0f64, 0usize);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        if !g.is_finite() {
            nonfinite += 1;
            continue;
        }
        let e = (f64::from(g) - w).abs();
        if e > max_err {
            max_err = e;
            at = i;
        }
    }
    let measured = if peak > 0.0 { max_err / peak } else { max_err };
    let detail = format!(
        "max |err| {max_err:.3e} at {at} = {measured:.3e} of max|ref| {peak:.4e}; bound {rel:.0e} \
         ({source}); {nonfinite} non-finite"
    );
    let ok = nonfinite == 0 && max_err <= rel * peak;
    let check = if ok {
        Check::pass(name, detail)
    } else {
        Check::fail(name, detail)
    };
    check
        .with("max_abs_err", max_err)
        .with("argmax", at)
        .with("rel_of_peak", measured)
        .with("max_abs_ref", peak)
        .with("nonfinite", nonfinite)
        .with("bound_rel_of_peak", rel)
        .with("bound_source", source)
}

// ------------------------------------------------------------- source lint ---

/// Names of the `extern "C" __global__ void NAME(` kernels in `source`.
pub fn defined_entries(source: &str) -> Vec<String> {
    source
        .lines()
        .filter_map(|l| l.trim().strip_prefix("extern \"C\" __global__ void "))
        .filter_map(|rest| rest.split('(').next())
        .map(|name| name.trim().to_string())
        .collect()
}

/// The checks every module of this lane passes: it defines exactly its listed
/// entries, carries one copy of the prelude, and has no NUL (cudarc's NVRTC
/// wrapper panics on one, `nvrtc/safe.rs:138`), no `#include` (NVRTC has no
/// default include path), no atomics (one writer per element) and no
/// fast-math intrinsic (`__expf` and friends trade accuracy for speed).
pub fn lint_module(module: &KernelModule) -> Result<(), String> {
    let mut defined = defined_entries(module.source);
    defined.sort();
    let mut listed: Vec<String> = module.entries.iter().map(|s| s.to_string()).collect();
    listed.sort();
    if defined != listed {
        return Err(format!(
            "{}: defines {defined:?}, lists {listed:?}",
            module.name
        ));
    }
    let src = module.source;
    let marker = "---- L-cuda-small prelude (src/small_common.rs) ----";
    if src.matches(marker).count() != 1 {
        return Err(format!(
            "{}: the prelude must appear exactly once",
            module.name
        ));
    }
    // crate::device_prelude!() has no include guard: at most one copy.
    if src.matches("#define QD_GRID_STRIDE(").count() > 1 {
        return Err(format!(
            "{}: device_prelude!() is spliced twice",
            module.name
        ));
    }
    if src.contains("QD_GRID_STRIDE(") && !src.contains("#define QD_GRID_STRIDE(") {
        return Err(format!(
            "{}: uses QD_GRID_STRIDE without device_prelude!()",
            module.name
        ));
    }
    for (bad, why) in [
        ("\0", "an interior NUL"),
        ("#include", "an #include"),
        ("atomic", "an atomic"),
        ("__expf", "a fast-math exp"),
        ("__logf", "a fast-math log"),
        ("__sinf", "a fast-math sin"),
        ("__cosf", "a fast-math cos"),
        ("__fdividef", "a fast division"),
        ("rsqrtf", "an approximate rsqrt (use 1.0f / sqrtf)"),
    ] {
        if src.contains(bad) {
            return Err(format!("{}: source holds {why}", module.name));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::check::Status;

    #[test]
    fn every_small_module_passes_the_lint() {
        for module in SMALL_MODULES {
            if let Err(e) = lint_module(module) {
                panic!("{e}");
            }
        }
    }

    #[test]
    fn the_lint_catches_a_missing_entry_an_atomic_and_a_fast_exp() {
        static MISSING: KernelModule = KernelModule {
            name: "m",
            source: COL_SUM_SOURCE,
            entries: &["qd_col_sum_blocks_f32", "qd_not_there"],
        };
        assert!(lint_module(&MISSING).is_err());
        let with_atomic = format!("{COL_SUM_SOURCE}\n// atomicAdd(&x, 1.0f);");
        let leaked: &'static str = Box::leak(with_atomic.into_boxed_str());
        let atomic = Box::leak(Box::new(KernelModule {
            name: "a",
            source: leaked,
            entries: &["qd_col_sum_blocks_f32"],
        }));
        assert!(lint_module(atomic).unwrap_err().contains("atomic"));
        let fast = format!("{COL_SUM_SOURCE}\n// __expf(x)");
        let fast: &'static str = Box::leak(fast.into_boxed_str());
        let fast = Box::leak(Box::new(KernelModule {
            name: "f",
            source: fast,
            entries: &["qd_col_sum_blocks_f32"],
        }));
        assert!(lint_module(fast).unwrap_err().contains("fast-math exp"));
    }

    #[test]
    fn the_warp_butterfly_is_the_same_in_every_lane_and_close_to_the_true_sum() {
        let lanes: [f32; 32] = std::array::from_fn(|i| {
            (i as f32 + 0.37) * 1.0e-3 * if i % 3 == 0 { -1.0 } else { 1.0 }
        });
        // Every lane's butterfly result: run it for each starting lane.
        let mut v = lanes;
        for m in [16usize, 8, 4, 2, 1] {
            let prev = v;
            for (i, slot) in v.iter_mut().enumerate() {
                *slot = prev[i] + prev[i ^ m];
            }
        }
        assert!(v.iter().all(|x| x.to_bits() == v[0].to_bits()));
        assert_eq!(warp_sum(lanes).to_bits(), v[0].to_bits());
        let exact: f64 = lanes.iter().map(|&x| f64::from(x)).sum();
        assert!((f64::from(warp_sum(lanes)) - exact).abs() < 1e-6);
    }

    #[test]
    fn the_warp_butterfly_is_a_tree_not_a_sequential_sum() {
        // Values whose sum depends on the order: 2^24 + 1 + ... .
        let mut lanes = [1.0f32; 32];
        lanes[0] = 16_777_216.0;
        // The butterfly pairs lane 0 with lane 16 first: 2^24 + 1 rounds to
        // 2^24; the other 30 ones sum exactly to 30 in the tree; 2^24 + 30.
        assert_eq!(warp_sum(lanes), 16_777_246.0);
        // A sequential sum from lane 0 loses every 1: 2^24.
        assert_eq!(lanes.iter().fold(0.0f32, |a, &x| a + x), 16_777_216.0);
        let p: [[f32; 1]; COLSUM_4WARPS_THREADS] = std::array::from_fn(|i| [i as f32]);
        assert_eq!(colsum_4warps(&p), [8128.0]);
    }

    /// GDN's column sum folds the four warp totals as `(w0 + w1) + (w2 + w3)`,
    /// not `block_sum`'s second butterfly `(w0 + w2) + (w1 + w3)`; the two
    /// differ in bits on these totals.
    #[test]
    fn the_four_warp_colsum_pairs_adjacent_warps() {
        let mut p = [[0.0f32; 2]; COLSUM_4WARPS_THREADS];
        for (warp, total) in [16_777_216.0f32, 1.0, -16_777_216.0, 1.0]
            .into_iter()
            .enumerate()
        {
            p[warp * 32][0] = total;
            p[warp * 32 + 5][1] = -total;
        }
        // (2^24 + 1) rounds to 2^24; (-2^24 + 1) is exact: the sum is 1.
        assert_eq!(colsum_4warps(&p), [1.0, -1.0]);
        let col0: Vec<f32> = p.iter().map(|r| r[0]).collect();
        assert_eq!(block_sum(&col0), 2.0);
    }

    #[test]
    fn block_reductions_fold_warps_then_the_published_values() {
        // 64 threads: two warps. Warp 0 holds 1.0 each, warp 1 holds 2.0 each.
        let per: Vec<f32> = (0..64).map(|i| if i < 32 { 1.0 } else { 2.0 }).collect();
        assert_eq!(block_sum(&per), 96.0);
        // One warp: no second level.
        assert_eq!(block_sum(&per[..32]), 32.0);
        let mut m = vec![NEG_FLT_MAX; 256];
        m[200] = -3.0;
        m[17] = -5.0;
        assert_eq!(block_max(&m), -3.0);
        // A NaN thread value is dropped by max, as fmaxf drops it.
        m[5] = f32::NAN;
        assert_eq!(block_max(&m), -3.0);
    }

    #[test]
    #[should_panic(expected = "not a whole number of warps")]
    fn a_ragged_block_is_refused() {
        block_sum(&[1.0; 48]);
    }

    #[test]
    fn thread_partials_visit_each_index_once_in_ascending_order() {
        // With a non-commutative fold, order is visible: acc * 2 + i.
        let p = thread_partials(4, 10, 0.0, |acc, x| acc * 2.0 + x, |i| i as f32);
        // Thread 1 sees 1, 5, 9: ((0*2+1)*2+5)*2+9 = 23.
        assert_eq!(p[1], 23.0);
        // Thread 3 sees 3, 7: (0*2+3)*2+7 = 13.
        assert_eq!(p[3], 13.0);
        assert_eq!(row_sum(256, 10, |i| i as f32), 45.0);
        assert_eq!(row_max(256, 10, |i| -(i as f32)), 0.0);
        assert_eq!(unit_sum(40, |_| 1.0), 40.0);
    }

    #[test]
    fn col_sum_adds_blocks_in_order_from_the_offset() {
        // Two halves of 3 blocks x 2 columns; sum the second half.
        let part: Vec<f32> = (0..12u8).map(f32::from).collect();
        assert_eq!(
            col_sum_blocks(&part, 6, 3, 2),
            vec![6.0 + 8.0 + 10.0, 7.0 + 9.0 + 11.0]
        );
        assert_eq!(col_sum_blocks(&part, 0, 3, 2), vec![6.0, 9.0]);
    }

    #[test]
    fn windows_refuse_crossing_rows_and_running_past_the_buffer() {
        let w = Window { ld: 10, off: 4 };
        assert!(w.check("op", "x", 3, 6, 30).is_ok());
        assert!(w.check("op", "x", 3, 7, 40).is_err());
        assert!(w.check("op", "x", 3, 6, 29).is_err());
        assert!(Window {
            ld: u64::MAX,
            off: 1
        }
        .check("op", "x", 3, 1, 10)
        .is_err());
        assert_eq!(w.at(2, 5), 29);
        assert!(Window::dense(5).check("op", "x", 2, 5, 10).is_ok());
        assert!(cols_overlap((0, 4), (3, 2)));
        assert!(!cols_overlap((0, 4), (4, 2)));
    }

    #[test]
    fn grids_are_shape_only_and_bounded() {
        let l = block_grid(7, ROW_THREADS, "t").unwrap();
        assert_eq!((l.grid, l.block), ((7, 1, 1), (256, 1, 1)));
        assert!(block_grid(0, ROW_THREADS, "t").is_err());
        assert!(block_grid(u64::from(MAX_GRID_X) + 1, ROW_THREADS, "t").is_err());
        assert_eq!(blocks_for(33, 32), 2);
        assert!(exact_in_f32("t", "dim", 1 << 24).is_ok());
        assert!(exact_in_f32("t", "dim", (1 << 24) + 1).is_err());
        assert!(positive_eps("t", 0.0).is_err());
        assert!(positive_eps("t", f32::NAN).is_err());
        assert!(positive_eps("t", 1e-6).is_ok());
    }

    #[test]
    fn checks_fail_on_non_finite_and_past_the_bound() {
        let want = [1.0f64, -2.0, 0.5];
        let ok = elementwise_check("e", &[1.0, -2.0, 0.5], &want, (1e-5, 1e-6), "s");
        assert_eq!(ok.status, Status::Pass);
        let far = elementwise_check("e", &[1.0, -2.0 + 1e-3, 0.5], &want, (1e-5, 1e-6), "s");
        assert_eq!(far.status, Status::Fail);
        let nan = elementwise_check("e", &[f32::NAN, -2.0, 0.5], &want, (1.0, 1.0), "s");
        assert_eq!(nan.status, Status::Fail);
        assert_eq!(
            peak_check("p", &[1.0, -2.0, 0.5001], &want, 1e-4, "s").status,
            Status::Pass
        );
        assert_eq!(
            peak_check("p", &[1.0, -2.0, 0.501], &want, 1e-4, "s").status,
            Status::Fail
        );
        assert_eq!(
            peak_check("p", &[0.0, 0.0], &[0.0, 0.0], 1e-4, "s").status,
            Status::Pass
        );
        assert_eq!(
            peak_check("p", &[1e-30, 0.0], &[0.0, 0.0], 1e-4, "s").status,
            Status::Fail
        );
        assert_eq!(
            peak_check("p", &[1.0], &[1.0, 2.0], 1e-4, "s").status,
            Status::Fail
        );
    }

    /// Declarations that let clang++ parse NVRTC CUDA-C as host C++. Only for
    /// `-fsyntax-only`: nothing here is defined or executed.
    #[cfg(target_os = "macos")]
    const HOST_CXX_SHIM: &str = r#"
#include <cmath>
#define __global__
#define __device__
#define __forceinline__ inline
#define __shared__ static
struct qd_shim_uint3 { unsigned int x, y, z; };
extern thread_local qd_shim_uint3 threadIdx, blockIdx, blockDim, gridDim;
void __syncthreads();
float __shfl_xor_sync(unsigned int mask, float v, int lane_mask);
unsigned int __float_as_uint(float x);
float __uint_as_float(unsigned int x);
float __fadd_rn(float a, float b);
float __fsub_rn(float a, float b);
float __fmul_rn(float a, float b);
float __fdiv_rn(float a, float b);
"#;

    /// **Syntax only; says nothing about numerics or execution.** Every module
    /// of this lane parses and type-checks as host C++ (`clang++
    /// -fsyntax-only -std=c++17`, NVRTC's default dialect, `-Werror`). Not a
    /// CUDA compile: it cannot see NVRTC-specific errors. macOS only (the
    /// system `clang++` of the lane's Mac; this host has no CUDA compiler);
    /// on Linux it is compiled out, not failed. A missing `clang++` fails the
    /// test: a check that could not run is not a pass.
    #[cfg(target_os = "macos")]
    #[test]
    fn syntax_only_every_small_module_parses_as_host_cxx_says_nothing_about_numerics() {
        let dir = std::env::temp_dir().join(format!("qd-small-cxx-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let shim = dir.join("shim.h");
        std::fs::write(&shim, HOST_CXX_SHIM).expect("write shim");
        for module in SMALL_MODULES {
            let src = dir.join(format!("{}.cu.cpp", module.name));
            std::fs::write(&src, module.source).expect("write source");
            let out = std::process::Command::new("clang++")
                .args([
                    "-fsyntax-only",
                    "-std=c++17",
                    "-ffp-contract=off",
                    "-Wall",
                    "-Werror",
                    "-Wno-unused-function",
                    "-x",
                    "c++",
                    "-include",
                ])
                .arg(&shim)
                .arg(&src)
                .output()
                .unwrap_or_else(|e| {
                    panic!("clang++ could not be run ({e}); the host C++ check did not run")
                });
            assert!(
                out.status.success(),
                "{} does not parse as host C++:\n{}",
                module.name,
                String::from_utf8_lossy(&out.stderr)
            );
        }
        std::fs::remove_dir_all(&dir).expect("remove temp dir");
    }
}

//! CUDA-C kernel sources, compiled at run time by NVRTC.
//!
//! **How they are compiled** ([`STRICT_SM90`]):
//! - `--gpu-architecture=compute_90`: NVRTC emits PTX and the driver JITs it
//!   for sm_90 at module load. cudarc 0.19.10's safe NVRTC layer exposes only
//!   `nvrtcGetPTX` (`cudarc/src/nvrtc/result.rs` has no CUBIN getter), and
//!   NVIDIA does not document PTX output for a real `sm_90` target, so the
//!   virtual architecture is the documented path.
//! - `--fmad=false`: NVRTC's default is `true`. Off, plain `*` and `+` are
//!   emitted as `mul.rn`/`add.rn`, which neither NVRTC nor the driver's JIT
//!   may contract into an FMA, so each is one IEEE rounding. The FFMA GEMM's
//!   fused multiply-adds are explicit `fmaf` calls and stay fused.
//! - `--ftz=false`, `--prec-div=true`, `--prec-sqrt=true`: IEEE subnormals
//!   and division, stated rather than left to defaults.
//!
//! The JIT cannot change an `.rn` instruction's result, so the bits these
//! kernels write depend on the source and these options, not on the driver
//! version. Both still go in every report (the NVRTC and driver versions).
//!
//! **Determinism.** Every output element has exactly one writer; there are no
//! atomics and no reductions across threads. Elementwise kernels grid-stride
//! over a 64-bit index ([`crate::geometry::grid_1d`]). The GEMM walks k in
//! ascending order for every output, whatever the tile, so its bits do not
//! depend on the grid either.
//!
//! The kernels are named after their tessl seams (`tessl/kernels/utils.metal`,
//! `qwen35_bwd.metal`, `cross_entropy.metal`) with a `qd_` prefix. No
//! `#include`: NVRTC has no default include path, and the bf16 conversion is
//! the integer form of [`crate::bf16`], so `cuda_bf16.h` is not needed.

/// NVRTC options, part of the compile-cache key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CompileSpec {
    /// `--gpu-architecture`.
    pub arch: &'static str,
    /// `--fmad`.
    pub fmad: bool,
    /// `--ftz`.
    pub ftz: bool,
    /// `--prec-div`.
    pub prec_div: bool,
    /// `--prec-sqrt`.
    pub prec_sqrt: bool,
}

impl CompileSpec {
    /// The NVRTC option strings, in a fixed order.
    pub fn options(&self) -> Vec<String> {
        vec![
            format!("--gpu-architecture={}", self.arch),
            format!("--fmad={}", self.fmad),
            format!("--ftz={}", self.ftz),
            format!("--prec-div={}", self.prec_div),
            format!("--prec-sqrt={}", self.prec_sqrt),
        ]
    }
}

/// The spec every kernel in this crate is compiled with.
pub const STRICT_SM90: CompileSpec = CompileSpec {
    arch: "compute_90",
    fmad: false,
    ftz: false,
    prec_div: true,
    prec_sqrt: true,
};

/// The same, for the architecture-specific `sm_90a` features (WGMMA, TMA) a
/// later GEMM tier needs. Rung 0 compiles one module with it to show the
/// box's NVRTC accepts it; nothing launches it.
pub const STRICT_SM90A: CompileSpec = CompileSpec {
    arch: "compute_90a",
    ..STRICT_SM90
};

/// One NVRTC compilation unit and the entry points it must export.
#[derive(Debug, PartialEq, Eq)]
pub struct KernelModule {
    /// A stable name, for the cache and reports.
    pub name: &'static str,
    /// CUDA-C source.
    pub source: &'static str,
    /// `extern "C"` kernels the module defines.
    pub entries: &'static [&'static str],
}

/// K0: casts, column-window copy, deliver, zero, scatter-add, gather.
pub const K0: KernelModule = KernelModule {
    name: "k0_plumbing",
    source: K0_SOURCE,
    entries: &[
        "qd_cast_f32_to_bf16",
        "qd_cast_bf16_to_f32",
        "qd_copy_cols_f32",
        "qd_deliver_copy_f32",
        "qd_deliver_add_f32",
        "qd_zero_f32",
        "qd_scatter_add_rows_f32",
        "qd_ce_gather_rows_f32",
        "qd_ce_gather_rows_bf16",
    ],
};

/// K1 ExactF32 tier: fixed-k-order FFMA GEMM, plus the same kernel rounding
/// its operands to bf16 on load (the bf16 tier's fallback and on-device oracle).
pub const GEMM_FFMA: KernelModule = KernelModule {
    name: "k1_gemm_ffma",
    source: GEMM_FFMA_SOURCE,
    entries: &[
        "qd_gemm_ffma_nn_f32",
        "qd_gemm_ffma_tn_f32",
        "qd_gemm_ffma_nt_f32",
        "qd_gemm_ffma_nn_bf16r",
        "qd_gemm_ffma_tn_bf16r",
        "qd_gemm_ffma_nt_bf16r",
    ],
};

/// Every module this crate compiles.
pub const ALL_MODULES: [&KernelModule; 2] = [&K0, &GEMM_FFMA];

/// The FFMA GEMM entry point for a layout and operand rounding.
pub fn gemm_ffma_entry(layout: crate::gemm_plan::GemmLayout, round_bf16: bool) -> &'static str {
    use crate::gemm_plan::GemmLayout;
    match (layout, round_bf16) {
        (GemmLayout::Nn, false) => "qd_gemm_ffma_nn_f32",
        (GemmLayout::Tn, false) => "qd_gemm_ffma_tn_f32",
        (GemmLayout::Nt, false) => "qd_gemm_ffma_nt_f32",
        (GemmLayout::Nn, true) => "qd_gemm_ffma_nn_bf16r",
        (GemmLayout::Tn, true) => "qd_gemm_ffma_tn_bf16r",
        (GemmLayout::Nt, true) => "qd_gemm_ffma_nt_bf16r",
    }
}

/// Device helpers every module starts with: tessl's bf16 rounding
/// (`tessl/src/tensor.rs:641-654`) in integers, and the 64-bit grid-stride
/// loop. A macro, so `concat!` can splice one copy into each module.
///
/// Exported (`crate::device_prelude!()`, lead's ruling 2026-10-01) so every
/// kernel module in the crate splices this one copy instead of carrying its
/// own `QD_GRID_STRIDE`. It defines `qd_f32_to_bf16_bits`,
/// `qd_bf16_bits_to_f32`, `qd_round_bf16` and `QD_GRID_STRIDE`, with no
/// include guard: splice it once per module.
#[macro_export]
macro_rules! device_prelude {
    () => {
        r#"
// tessl/src/tensor.rs:641-654, as integers: NaN keeps sign and top payload and
// is quieted; finite values round to nearest, ties to even.
__device__ __forceinline__ unsigned short qd_f32_to_bf16_bits(float x) {
    const unsigned int bits = __float_as_uint(x);
    if ((bits & 0x7fffffffu) > 0x7f800000u) {
        return (unsigned short)((bits >> 16) | 0x0040u);
    }
    return (unsigned short)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

__device__ __forceinline__ float qd_bf16_bits_to_f32(unsigned short b) {
    return __uint_as_float(((unsigned int)b) << 16);
}

__device__ __forceinline__ float qd_round_bf16(float x) {
    return qd_bf16_bits_to_f32(qd_f32_to_bf16_bits(x));
}

// A 64-bit grid-stride loop: any grid visits each i in [0, n) exactly once.
#define QD_GRID_STRIDE(i, n)                                                     \
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x      \
                                + threadIdx.x;                                   \
         i < (n);                                                                \
         i += (unsigned long long)gridDim.x * blockDim.x)
"#
    };
}

const K0_SOURCE: &str = concat!(
    device_prelude!(),
    r#"
extern "C" __global__ void qd_cast_f32_to_bf16(
    const float* src, unsigned short* dst, unsigned long long n)
{
    QD_GRID_STRIDE(i, n) { dst[i] = qd_f32_to_bf16_bits(src[i]); }
}

extern "C" __global__ void qd_cast_bf16_to_f32(
    const unsigned short* src, float* dst, unsigned long long n)
{
    QD_GRID_STRIDE(i, n) { dst[i] = qd_bf16_bits_to_f32(src[i]); }
}

// dst[r, dst_off + c] = src[r, src_off + c], c < width (qwen35_bwd.metal:681).
extern "C" __global__ void qd_copy_cols_f32(
    const float* src, float* dst,
    unsigned long long rows, unsigned long long width,
    unsigned long long ld_src, unsigned long long src_off,
    unsigned long long ld_dst, unsigned long long dst_off)
{
    const unsigned long long total = rows * width;
    QD_GRID_STRIDE(i, total) {
        const unsigned long long r = i / width;
        const unsigned long long c = i - r * width;
        dst[r * ld_dst + dst_off + c] = src[r * ld_src + src_off + c];
    }
}

// tessl's deliver (qwen35_train.rs:227-243): a gradient part into a bank.
extern "C" __global__ void qd_deliver_copy_f32(
    const float* src, unsigned long long src_off,
    float* dst, unsigned long long dst_off, unsigned long long n)
{
    QD_GRID_STRIDE(i, n) { dst[dst_off + i] = src[src_off + i]; }
}

extern "C" __global__ void qd_deliver_add_f32(
    const float* src, unsigned long long src_off,
    float* dst, unsigned long long dst_off, unsigned long long n)
{
    QD_GRID_STRIDE(i, n) { dst[dst_off + i] = __fadd_rn(dst[dst_off + i], src[src_off + i]); }
}

extern "C" __global__ void qd_zero_f32(
    float* dst, unsigned long long off, unsigned long long n)
{
    QD_GRID_STRIDE(i, n) { dst[off + i] = 0.0f; }
}

// dst[pos[i] * width + c] += src[i * width + c]. Rows are distinct (checked on
// the host), so every element has one writer (qwen35_bwd.metal:705).
extern "C" __global__ void qd_scatter_add_rows_f32(
    const float* src, const unsigned int* pos, float* dst,
    unsigned long long n, unsigned long long width)
{
    const unsigned long long total = n * width;
    QD_GRID_STRIDE(i, total) {
        const unsigned long long r = i / width;
        const unsigned long long c = i - r * width;
        const unsigned long long d = (unsigned long long)pos[r] * width + c;
        dst[d] = __fadd_rn(dst[d], src[i]);
    }
}

// out[i, c] = h[rows[i] * ld + off + c] (cross_entropy.metal:28-42).
extern "C" __global__ void qd_ce_gather_rows_f32(
    const float* h, const unsigned int* rows, float* out,
    unsigned long long n_rows, unsigned long long hidden,
    unsigned long long ld, unsigned long long off)
{
    const unsigned long long total = n_rows * hidden;
    QD_GRID_STRIDE(i, total) {
        const unsigned long long r = i / hidden;
        const unsigned long long c = i - r * hidden;
        out[i] = h[(unsigned long long)rows[r] * ld + off + c];
    }
}

extern "C" __global__ void qd_ce_gather_rows_bf16(
    const unsigned short* h, const unsigned int* rows, float* out,
    unsigned long long n_rows, unsigned long long hidden,
    unsigned long long ld, unsigned long long off)
{
    const unsigned long long total = n_rows * hidden;
    QD_GRID_STRIDE(i, total) {
        const unsigned long long r = i / hidden;
        const unsigned long long c = i - r * hidden;
        out[i] = qd_bf16_bits_to_f32(h[(unsigned long long)rows[r] * ld + off + c]);
    }
}
"#
);

const GEMM_FFMA_SOURCE: &str = concat!(
    device_prelude!(),
    r#"
#define QD_TILE 16

// Row-major storage. LAYOUT 0 = nn, 1 = tn (A stored [k, m]), 2 = nt (B stored [n, k]).
template <int LAYOUT, bool ROUND>
__device__ __forceinline__ float qd_load_a(
    const float* a, unsigned long long i, unsigned long long p,
    unsigned long long m, unsigned long long k)
{
    const float v = (LAYOUT == 1) ? a[p * m + i] : a[i * k + p];
    return ROUND ? qd_round_bf16(v) : v;
}

template <int LAYOUT, bool ROUND>
__device__ __forceinline__ float qd_load_b(
    const float* b, unsigned long long p, unsigned long long j,
    unsigned long long n, unsigned long long k)
{
    const float v = (LAYOUT == 2) ? b[j * k + p] : b[p * n + j];
    return ROUND ? qd_round_bf16(v) : v;
}

// One thread per C element, 16x16 tiles staged through shared memory. Each
// thread runs acc = fmaf(a, b, acc) over p = 0, 1, ..., k-1 in that order
// (tiles ascending, then within a tile ascending), from +0.0, and stops at k:
// no padded fma(0, 0, acc), which would turn a -0.0 into +0.0. With
// accumulate, C = C + acc in one rounding. host_ref::gemm_ffma_f32 is the
// same sequence on the host.
template <int LAYOUT, bool ROUND>
__device__ __forceinline__ void qd_gemm_ffma_body(
    const float* a, const float* b, float* c,
    unsigned long long m, unsigned long long n, unsigned long long k, int accumulate)
{
    __shared__ float as[QD_TILE][QD_TILE];
    __shared__ float bs[QD_TILE][QD_TILE];
    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned long long i = (unsigned long long)blockIdx.y * QD_TILE + ty;
    const unsigned long long j = (unsigned long long)blockIdx.x * QD_TILE + tx;
    float acc = 0.0f;
    for (unsigned long long p0 = 0; p0 < k; p0 += QD_TILE) {
        const unsigned long long pa = p0 + tx;
        const unsigned long long pb = p0 + ty;
        as[ty][tx] = (i < m && pa < k) ? qd_load_a<LAYOUT, ROUND>(a, i, pa, m, k) : 0.0f;
        bs[ty][tx] = (pb < k && j < n) ? qd_load_b<LAYOUT, ROUND>(b, pb, j, n, k) : 0.0f;
        __syncthreads();
        const unsigned long long left = k - p0;
        const int lim = left < QD_TILE ? (int)left : QD_TILE;
        for (int q = 0; q < lim; ++q) {
            acc = fmaf(as[ty][q], bs[q][tx], acc);
        }
        __syncthreads();
    }
    if (i < m && j < n) {
        const unsigned long long o = i * n + j;
        c[o] = accumulate ? __fadd_rn(c[o], acc) : acc;
    }
}

#define QD_GEMM_ENTRY(NAME, LAYOUT, ROUND)                                       \
extern "C" __global__ void NAME(                                                 \
    const float* a, const float* b, float* c,                                    \
    unsigned long long m, unsigned long long n, unsigned long long k,            \
    int accumulate)                                                              \
{                                                                                \
    qd_gemm_ffma_body<LAYOUT, ROUND>(a, b, c, m, n, k, accumulate);              \
}

QD_GEMM_ENTRY(qd_gemm_ffma_nn_f32, 0, false)
QD_GEMM_ENTRY(qd_gemm_ffma_tn_f32, 1, false)
QD_GEMM_ENTRY(qd_gemm_ffma_nt_f32, 2, false)
QD_GEMM_ENTRY(qd_gemm_ffma_nn_bf16r, 0, true)
QD_GEMM_ENTRY(qd_gemm_ffma_tn_bf16r, 1, true)
QD_GEMM_ENTRY(qd_gemm_ffma_nt_bf16r, 2, true)
"#
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemm_plan::GemmLayout;

    /// The entry names of `QD_GEMM_ENTRY(...)` lines and `extern "C"` kernels.
    fn defined_entries(source: &str) -> Vec<String> {
        let mut out = Vec::new();
        for line in source.lines() {
            let line = line.trim();
            if let Some(rest) = line.strip_prefix("extern \"C\" __global__ void ") {
                if let Some(name) = rest.split('(').next() {
                    if name != "NAME" {
                        out.push(name.trim().to_string());
                    }
                }
            }
            if let Some(rest) = line.strip_prefix("QD_GEMM_ENTRY(") {
                if let Some(name) = rest.split(',').next() {
                    out.push(name.trim().to_string());
                }
            }
        }
        out
    }

    #[test]
    fn every_module_defines_exactly_its_listed_entries() {
        for module in ALL_MODULES {
            let mut defined = defined_entries(module.source);
            defined.sort();
            let mut listed: Vec<String> = module.entries.iter().map(|s| s.to_string()).collect();
            listed.sort();
            assert_eq!(defined, listed, "module {}", module.name);
        }
    }

    #[test]
    fn sources_have_no_nul_no_include_and_no_float_atomics() {
        // cudarc's NVRTC wrapper panics on an interior NUL (nvrtc/safe.rs:138).
        for module in ALL_MODULES {
            assert!(!module.source.contains('\0'), "{}", module.name);
            assert!(!module.source.contains("#include"), "{}", module.name);
            assert!(!module.source.contains("atomicAdd"), "{}", module.name);
        }
    }

    #[test]
    fn the_device_rounding_is_the_host_rounding() {
        // The integer sequence in both device modules is crate::bf16's, once each.
        let round = "(bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16";
        let nan = "((bits >> 16) | 0x0040u)";
        for module in ALL_MODULES {
            assert_eq!(module.source.matches(round).count(), 1, "{}", module.name);
            assert_eq!(module.source.matches(nan).count(), 1, "{}", module.name);
        }
    }

    #[test]
    fn strict_options_turn_fma_contraction_off() {
        let opts = STRICT_SM90.options();
        assert!(opts.contains(&"--fmad=false".to_string()));
        assert!(opts.contains(&"--gpu-architecture=compute_90".to_string()));
        assert!(STRICT_SM90A
            .options()
            .contains(&"--gpu-architecture=compute_90a".to_string()));
    }

    #[test]
    fn every_gemm_entry_is_reachable() {
        for layout in GemmLayout::ALL {
            for round in [false, true] {
                let e = gemm_ffma_entry(layout, round);
                assert!(GEMM_FFMA.entries.contains(&e), "{e}");
            }
        }
    }
}

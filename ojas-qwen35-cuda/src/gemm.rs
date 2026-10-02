//! K1: row-major GEMM `nn` / `tn` / `nt`, two numerics tiers (tessl's
//! `GemmOperands`, `tessl/src/gemm.rs:991-1047`).
//!
//! **ExactF32** — [`gemm_ffma`] with [`Operands::ExactF32`]: a hand-written
//! CUDA-core kernel, `acc = fmaf(a, b, acc)` over ascending k for every
//! output, no tensor cores (TF32 would round operands to 10 mantissa bits,
//! `cuda-backend-scoping.md` §3), no split-K. Its bits are a function of the
//! inputs only: not of the grid, the SM count or the library version, and
//! they equal [`crate::host_ref::gemm_ffma_f32`] exactly. This is the K1 row's
//! "ExactF32 (FFMA, fixed k-order)" (`cuda-backend-scoping.md:349`), and the
//! fallback the user's decision names for the bf16 tier
//! (`HANDOFF/ojas-training-2026-10-01.md:36`).
//!
//! **Bf16** — [`gemm_bf16_cublas`]: operands rounded to bf16 (RNE, the K0
//! cast), f32 accumulate, **f32 C**, through raw `cublasGemmEx` with
//! `A/B = CUDA_R_16BF`, `C = CUDA_R_32F`, `CUBLAS_COMPUTE_32F`,
//! `CUBLAS_GEMM_DEFAULT`, on a handle in `CUBLAS_DEFAULT_MATH`. cudarc's safe
//! `Gemm<bf16>` writes a bf16 C (`cudarc/src/cublas/safe/gemm.rs:176`), so it
//! is not used. Whether cuBLAS 12.8 accepts this type combination is
//! unverified (`GAP-L-CUDA-CUBLAS-BF16-F32-UNVERIFIED-2026-10-01`); rung 0
//! records the status. If it is refused, [`gemm_ffma`] with
//! [`Operands::Bf16`] computes the same tier (bf16 operands, f32 accumulate,
//! f32 C) at CUDA-core speed, bit-identical to the host emulation.
//!
//! **cuBLAS determinism scope.** NVIDIA documents bitwise-reproducible cuBLAS
//! results within one toolkit version "on GPUs with the same architecture and
//! the same number of SMs", and not with several streams or with
//! `cublasSetAtomicsMode` allowing atomics (cuBLAS docs, "Results
//! reproducibility"). So the Bf16 tier is repeatable run to run on one GH200
//! (one stream, one handle, a fixed workspace, atomics never allowed), but a
//! different SM count or cuBLAS version may change its bits. Every rung-0
//! report records the SM count and the cuBLAS, NVRTC and driver versions for
//! that reason. The FFMA tier has no such scope.

use std::ffi::c_void;

use cudarc::cublas::sys as cublas_sys;
use cudarc::driver::{DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg};

use crate::buffer::CudaBuffer;
use crate::error::CudaError;
use crate::gemm_plan::{cublas_args, GemmLayout, GemmShape, Trans};
use crate::geometry::gemm_grid;
pub use crate::host_ref::Operands;
use crate::k0::cast_f32_to_bf16;
use crate::kernels::{gemm_ffma_entry, GEMM_FFMA, STRICT_SM90};
use crate::runtime::{cublas_error, driver_error, CudaRuntime};

/// Overwrite C, or add the product into it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Accumulate {
    /// `C = op(A) op(B)`.
    Overwrite,
    /// `C = C + op(A) op(B)`.
    Add,
}

/// What one GEMM computes: tier, layout, shape, and whether it adds into C.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemmSpec {
    /// ExactF32 or bf16 operands.
    pub operands: Operands,
    /// `nn`, `tn` or `nt`.
    pub layout: GemmLayout,
    /// `m x n x k`.
    pub shape: GemmShape,
    /// Overwrite or add into C.
    pub acc: Accumulate,
}

/// The FFMA GEMM: ExactF32, or bf16-rounded operands (see the module docs).
pub fn gemm_ffma(
    rt: &CudaRuntime,
    spec: GemmSpec,
    a: &CudaBuffer<f32>,
    b: &CudaBuffer<f32>,
    c: &mut CudaBuffer<f32>,
) -> Result<(), CudaError> {
    let GemmSpec {
        operands,
        layout,
        shape,
        acc,
    } = spec;
    let entry = gemm_ffma_entry(layout, operands == Operands::Bf16);
    shape.check_lens(a.len(), b.len(), c.len(), entry)?;
    let launch = gemm_grid(shape.m, shape.n)?;
    let f = rt.function(&GEMM_FFMA, &STRICT_SM90, entry)?;
    let (m, n, k) = (
        u64::try_from(shape.m).unwrap_or(u64::MAX),
        u64::try_from(shape.n).unwrap_or(u64::MAX),
        u64::try_from(shape.k).unwrap_or(u64::MAX),
    );
    let accumulate: i32 = match acc {
        Accumulate::Overwrite => 0,
        Accumulate::Add => 1,
    };
    let mut builder = rt.stream().launch_builder(&f);
    builder
        .arg(a.slice())
        .arg(b.slice())
        .arg(c.slice_mut())
        .arg(&m)
        .arg(&n)
        .arg(&k)
        .arg(&accumulate);
    let cfg = LaunchConfig {
        grid_dim: launch.grid,
        block_dim: launch.block,
        shared_mem_bytes: 0,
    };
    // SAFETY: (const float* a, const float* b, float* c, ull m, ull n, ull k,
    // int accumulate); check_lens matched every buffer to the shape and
    // layout, and the kernel guards every index by m, n, k.
    unsafe { builder.launch(cfg) }.map_err(|e| driver_error(entry, e))?;
    Ok(())
}

fn op(t: Trans) -> cublas_sys::cublasOperation_t {
    match t {
        Trans::N => cublas_sys::cublasOperation_t::CUBLAS_OP_N,
        Trans::T => cublas_sys::cublasOperation_t::CUBLAS_OP_T,
    }
}

/// `cublasGemmEx`, bf16 A and B (as bits), f32 C, f32 compute, row-major
/// semantics through [`cublas_args`].
pub fn gemm_bf16_cublas(
    rt: &CudaRuntime,
    layout: GemmLayout,
    shape: GemmShape,
    a: &CudaBuffer<u16>,
    b: &CudaBuffer<u16>,
    c: &mut CudaBuffer<f32>,
    acc: Accumulate,
) -> Result<(), CudaError> {
    const OP: &str = "cublasGemmEx(bf16, bf16 -> f32, COMPUTE_32F)";
    shape.check_lens(a.len(), b.len(), c.len(), OP)?;
    let args = cublas_args(layout, shape)?;
    let alpha: f32 = 1.0;
    let beta: f32 = match acc {
        Accumulate::Overwrite => 0.0,
        Accumulate::Add => 1.0,
    };
    let stream = rt.stream();
    let (a_ptr, _a_sync) = a.slice().device_ptr(stream);
    let (b_ptr, _b_sync) = b.slice().device_ptr(stream);
    let (c_ptr, _c_sync) = c.slice_mut().device_ptr_mut(stream);
    // SAFETY: the handle is live and bound to `stream`; the pointers are live
    // device allocations whose lengths check_lens matched to the shape, and
    // cublas_args checked every leading dimension against BLAS's rule.
    // cuBLAS's first operand is the row-major B, its second the row-major A
    // (gemm_plan module docs). alpha and beta are host f32s, the scale type
    // of CUBLAS_COMPUTE_32F, in the default host pointer mode.
    let status = unsafe {
        cublas_sys::cublasGemmEx(
            rt.cublas_handle(),
            op(args.trans_first),
            op(args.trans_second),
            args.m,
            args.n,
            args.k,
            (&alpha as *const f32).cast::<c_void>(),
            b_ptr as *const c_void,
            cublas_sys::cudaDataType_t::CUDA_R_16BF,
            args.ld_first,
            a_ptr as *const c_void,
            cublas_sys::cudaDataType_t::CUDA_R_16BF,
            args.ld_second,
            (&beta as *const f32).cast::<c_void>(),
            c_ptr as *mut c_void,
            cublas_sys::cudaDataType_t::CUDA_R_32F,
            args.ldc,
            cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
    };
    status.result().map_err(|e| cublas_error(OP, e.0))
}

/// Which engine computes the bf16 tier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bf16Engine {
    /// `cublasGemmEx` ([`gemm_bf16_cublas`]).
    Cublas,
    /// The FFMA kernel rounding its operands on load.
    Ffma,
}

/// A GEMM on f32 operands at either tier, as tessl's `GemmOperands::{nn,tn,nt}`.
/// The bf16 tier through cuBLAS first casts A and B to bf16 with the K0 cast
/// (into temporary buffers freed in stream order).
pub fn gemm(
    rt: &CudaRuntime,
    spec: GemmSpec,
    engine: Bf16Engine,
    a: &CudaBuffer<f32>,
    b: &CudaBuffer<f32>,
    c: &mut CudaBuffer<f32>,
) -> Result<(), CudaError> {
    match (spec.operands, engine) {
        (Operands::ExactF32, _) | (Operands::Bf16, Bf16Engine::Ffma) => {
            gemm_ffma(rt, spec, a, b, c)
        }
        (Operands::Bf16, Bf16Engine::Cublas) => {
            spec.shape
                .check_lens(a.len(), b.len(), c.len(), "gemm bf16")?;
            let mut a16 = rt.alloc_zeros::<u16>(a.len(), "gemm A bf16")?;
            let mut b16 = rt.alloc_zeros::<u16>(b.len(), "gemm B bf16")?;
            cast_f32_to_bf16(rt, a, &mut a16)?;
            cast_f32_to_bf16(rt, b, &mut b16)?;
            gemm_bf16_cublas(rt, spec.layout, spec.shape, &a16, &b16, c, spec.acc)
        }
    }
}

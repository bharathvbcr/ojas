//! The device side of [`crate::small_common`]: the one way lane
//! L-cuda-small's modules are compiled and launched, the block-order column
//! sum, and the compile check rung a runs over every module of the lane.
//!
//! Every module here is compiled with [`STRICT_SM90`] and nothing else, so
//! `--fmad=false`, `--ftz=false`, `--prec-div=true` and `--prec-sqrt=true`
//! hold for every kernel: that is what makes the host mirrors bitwise
//! ([`crate::small_common`] module docs). Nothing here waits; every launch is
//! queued on the runtime's one stream.

use std::time::Instant;

use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};

use crate::buffer::CudaBuffer;
use crate::check::Check;
use crate::error::CudaError;
use crate::geometry::{grid_1d, Launch};
use crate::kernels::{KernelModule, STRICT_SM90};
use crate::runtime::{driver_error, CudaRuntime};
use crate::small_common::{add, exact_len, mul, to_u64, COL_SUM, COL_SUM_ENTRY, SMALL_MODULES};
use crate::smoke::guarded;

/// A launch shape as cudarc's config; all shared memory here is static.
pub(crate) fn cfg(l: Launch) -> LaunchConfig {
    LaunchConfig {
        grid_dim: l.grid,
        block_dim: l.block,
        shared_mem_bytes: 0,
    }
}

/// `module`'s `entry`, compiled (or fetched from the cache) with [`STRICT_SM90`].
pub(crate) fn function(
    rt: &CudaRuntime,
    module: &'static KernelModule,
    entry: &str,
) -> Result<CudaFunction, CudaError> {
    rt.function(module, &STRICT_SM90, entry)
}

/// `out[c] = sum over b < blocks of part[part_off + b*dim + c]`, blocks in
/// order (`qd_col_sum_blocks_f32`); `out` is `[dim]`.
pub fn col_sum_blocks(
    rt: &CudaRuntime,
    part: &CudaBuffer<f32>,
    part_off: u64,
    (blocks, dim): (u64, u64),
    out: &mut CudaBuffer<f32>,
) -> Result<(), CudaError> {
    const OP: &str = COL_SUM_ENTRY;
    exact_len(OP, "out", out.len(), dim)?;
    let end = add(part_off, mul(blocks, dim, OP)?, OP)?;
    if end > to_u64(part.len()) {
        return Err(CudaError::invalid(
            OP,
            format!(
                "part holds {} elements; {blocks} blocks of {dim} from {part_off} need {end}",
                part.len()
            ),
        ));
    }
    let Some(l) = grid_1d(dim) else {
        return Ok(());
    };
    let f = function(rt, &COL_SUM, OP)?;
    let mut b = rt.stream().launch_builder(&f);
    b.arg(part.slice())
        .arg(&part_off)
        .arg(out.slice_mut())
        .arg(&blocks)
        .arg(&dim);
    // SAFETY: (const float* part, ull part_off, float* out, ull blocks, ull
    // dim). part_off + blocks * dim <= part.len() and out holds dim elements,
    // checked above; the kernel reads part[part_off + b*dim + c] for b <
    // blocks, c < dim and writes out[c] once. part and out are distinct
    // allocations (one shared and one exclusive borrow).
    unsafe { b.launch(cfg(l)) }.map_err(|e| driver_error(OP, e))?;
    Ok(())
}

/// Every module of this lane and every entry compiles through NVRTC and
/// loads (rung a's first check; `crate::smoke::compile_checks` is M0's for
/// [`crate::kernels::ALL_MODULES`]).
pub fn compile_checks(rt: &CudaRuntime) -> Vec<Check> {
    let mut out = Vec::new();
    for module in SMALL_MODULES {
        let name = format!("nvrtc.{}", module.name);
        out.extend(guarded(&name, || {
            let start = Instant::now();
            for entry in module.entries {
                if let Err(e) = rt.function(module, &STRICT_SM90, entry) {
                    return vec![Check::from_error(&name, &e)];
                }
            }
            vec![Check::pass(
                &name,
                format!(
                    "{} entries compiled ({:?}) and loaded",
                    module.entries.len(),
                    STRICT_SM90.options()
                ),
            )
            .with("entries", module.entries.len())
            .with("compile_and_load_ms", start.elapsed().as_secs_f64() * 1e3)]
        }));
    }
    out
}

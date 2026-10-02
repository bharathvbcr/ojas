//! K9 on the device: [`crate::embed`]'s gather and its deterministic
//! backward. Training ids are known on the host, so both launches check them
//! there (`< vocab`) before uploading; an id is never trusted on the device.
//! The uploads are freed in stream order after the launch (cudarc's
//! `free_async`, as [`crate::k0::scatter_add_rows`]). Nothing here waits.

use cudarc::driver::PushKernelArg;

use crate::buffer::CudaBuffer;
use crate::embed::{EmbedPlan, EmbedRuns, EMBED_BWD, EMBED_FWD, MODULE};
use crate::error::CudaError;
use crate::geometry::grid_1d;
use crate::runtime::{driver_error, CudaRuntime};
use crate::small_common_cuda::{cfg, function};

/// `out[r, :] = table[ids[r], :]`, copied as bits.
pub fn embed_rows(
    rt: &CudaRuntime,
    plan: &EmbedPlan,
    ids: &[u32],
    table: &CudaBuffer<f32>,
    out: &mut CudaBuffer<f32>,
) -> Result<(), CudaError> {
    plan.check_fwd(ids, table.len(), out.len())?;
    let Some(l) = grid_1d(plan.n * plan.hidden) else {
        return Err(CudaError::invalid(EMBED_FWD, "no output elements"));
    };
    let f = function(rt, &MODULE, EMBED_FWD)?;
    let ids_dev = rt.upload(ids, "embed_rows ids")?;
    let mut b = rt.stream().launch_builder(&f);
    b.arg(ids_dev.slice())
        .arg(table.slice())
        .arg(out.slice_mut())
        .arg(&plan.n)
        .arg(&plan.hidden);
    // SAFETY: (const unsigned int* ids, const unsigned int* table, unsigned
    // int* out, ull n, ull hidden). table and out are f32 buffers passed as
    // 32-bit words: the kernel only loads and stores them, never does float
    // arithmetic, so the bits move unchanged. check_fwd checked every id
    // below vocab on the host and matched table to vocab * hidden and out to
    // n * hidden; the grid strides over out with one writer per element.
    unsafe { b.launch(cfg(l)) }.map_err(|e| driver_error(EMBED_FWD, e))?;
    Ok(())
}

/// `dw[ids[r], :] += dh[r, :]` for every row, each id's rows summed in
/// position order from `+0.0` and added once (no atomics).
pub fn embed_rows_bwd(
    rt: &CudaRuntime,
    plan: &EmbedPlan,
    ids: &[u32],
    dh: &CudaBuffer<f32>,
    dw: &mut CudaBuffer<f32>,
) -> Result<(), CudaError> {
    plan.check_bwd(ids, dh.len(), dw.len())?;
    let runs = EmbedRuns::new(plan, ids)?;
    let n_runs = runs.n_runs();
    let Some(l) = grid_1d(n_runs * plan.hidden) else {
        return Err(CudaError::invalid(EMBED_BWD, "no (run, column) pairs"));
    };
    let f = function(rt, &MODULE, EMBED_BWD)?;
    let pos = rt.upload(&runs.pos, "embed_rows_bwd positions")?;
    let run_start = rt.upload(&runs.run_start, "embed_rows_bwd run starts")?;
    let uniq = rt.upload(&runs.uniq, "embed_rows_bwd ids")?;
    let mut b = rt.stream().launch_builder(&f);
    b.arg(dh.slice())
        .arg(pos.slice())
        .arg(run_start.slice())
        .arg(uniq.slice())
        .arg(dw.slice_mut())
        .arg(&n_runs)
        .arg(&plan.hidden);
    // SAFETY: (const float* dh, const unsigned int* pos, const unsigned int*
    // run_start, const unsigned int* uniq, float* dw, ull n_runs, ull hidden).
    // EmbedRuns::new built pos as a permutation of 0..n, run_start with
    // n_runs + 1 ascending entries ending at n, and uniq with n_runs distinct
    // ids, all checked below vocab; check_bwd matched dh to n * hidden and dw
    // to vocab * hidden. Distinct ids give each dw element one writer.
    unsafe { b.launch(cfg(l)) }.map_err(|e| driver_error(EMBED_BWD, e))?;
    Ok(())
}

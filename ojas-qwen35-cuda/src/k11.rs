//! K11 on the device: AdamW in place over a parameter bank, one launch per
//! active entry, and the gradient's squared norm from per-chunk f32 partials
//! summed in f64 on the host. The scalars and the table are
//! [`crate::k11_host`]'s; so is the emulation the device must equal bit for
//! bit.

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::buffer::CudaBuffer;
use crate::error::CudaError;
use crate::geometry::{grid_1d, MAX_BLOCKS_1D};
use crate::k11_host::{sq_chunks, sum_partials_f64, AdamwBank, AdamwHyper, AdamwTable};
use crate::k11_kernels::{K11, SQ_BLOCK};
use crate::kernels::STRICT_SM90;
use crate::runtime::{driver_error, CudaRuntime};

fn len_u64(n: usize) -> u64 {
    u64::try_from(n).unwrap_or(u64::MAX)
}

fn check_bank(op: &str, table: &AdamwTable, bufs: &[(&str, usize)]) -> Result<(), CudaError> {
    for (name, len) in bufs {
        if *len != table.bank_len() {
            return Err(CudaError::invalid(
                op,
                format!(
                    "{name} holds {len} f32s; the table's bank is {}",
                    table.bank_len()
                ),
            ));
        }
    }
    Ok(())
}

/// One AdamW step on the device over every active entry of `bank`'s table,
/// then the step counts committed. Queues the launches and returns; nothing
/// waits. On an error part-way the buffers are partly stepped and the counts
/// are not committed: the bank is dead (see [`AdamwBank`]).
#[allow(clippy::too_many_arguments)]
pub fn adamw_step(
    rt: &CudaRuntime,
    bank: &mut AdamwBank,
    p: &mut CudaBuffer<f32>,
    g: &CudaBuffer<f32>,
    m: &mut CudaBuffer<f32>,
    v: &mut CudaBuffer<f32>,
    active: &[bool],
    h: &AdamwHyper,
) -> Result<(), CudaError> {
    const E: &str = "qd_adamw_window_f32";
    check_bank(
        E,
        bank.table(),
        &[
            ("p", p.len()),
            ("g", g.len()),
            ("m", m.len()),
            ("v", v.len()),
        ],
    )?;
    let plan = bank.plan_step(h, active)?;
    let s = plan.shared;
    let f = rt.function(&K11, &STRICT_SM90, E)?;
    for (e, sc) in bank.table().entries().iter().zip(&plan.per_entry) {
        let Some(sc) = sc else { continue };
        let (off, n) = (len_u64(e.offset), len_u64(e.len));
        let Some(l) = grid_1d(n) else { continue };
        let cfg = LaunchConfig {
            grid_dim: l.grid,
            block_dim: l.block,
            shared_mem_bytes: 0,
        };
        let mut b = rt.stream().launch_builder(&f);
        b.arg(p.slice_mut())
            .arg(g.slice())
            .arg(m.slice_mut())
            .arg(v.slice_mut())
            .arg(&off)
            .arg(&n)
            .arg(&sc.decay_mul)
            .arg(&sc.step_size)
            .arg(&sc.bc2_sqrt)
            .arg(&s.lerp_w)
            .arg(&s.beta2)
            .arg(&s.one_minus_beta2)
            .arg(&s.eps)
            .arg(&s.grad_scale);
        // SAFETY: the kernel takes (float* p, const float* g, float* m, float*
        // v, unsigned long long off, unsigned long long n, 8 x float) in this
        // order. The table guarantees off + n <= bank_len, and all four
        // buffers hold bank_len f32s (checked above), so every index the
        // grid-stride loop touches is in bounds. p, m and v are distinct
        // `&mut` borrows and g a shared one, so no output aliases an input.
        unsafe { b.launch(cfg) }.map_err(|err| driver_error(E, err))?;
    }
    bank.commit(plan.next)
}

/// The squared norm of `g` over the table's active entries: every active
/// window's partials in table order, downloaded after a bounded sync and
/// summed in f64. Also returns the partials, for a bitwise check.
pub fn grad_sq_norm(
    rt: &CudaRuntime,
    table: &AdamwTable,
    g: &CudaBuffer<f32>,
    active: &[bool],
) -> Result<(f64, Vec<f32>), CudaError> {
    const E: &str = "qd_sq_partials_f32";
    check_bank(E, table, &[("g", g.len())])?;
    if active.len() != table.entries().len() {
        return Err(CudaError::invalid(
            E,
            format!(
                "{} flags for {} entries",
                active.len(),
                table.entries().len()
            ),
        ));
    }
    let total: usize = table
        .entries()
        .iter()
        .zip(active)
        .filter(|(_, &on)| on)
        .map(|(e, _)| sq_chunks(e.len))
        .sum();
    if total == 0 {
        return Ok((0.0, Vec::new()));
    }
    let mut partials = rt.alloc_zeros::<f32>(total, "k11 sq partials")?;
    let f = rt.function(&K11, &STRICT_SM90, E)?;
    let mut part_off = 0u64;
    for (e, &on) in table.entries().iter().zip(active) {
        if !on {
            continue;
        }
        let chunks = len_u64(sq_chunks(e.len));
        let blocks = u32::try_from(chunks.min(u64::from(MAX_BLOCKS_1D))).unwrap_or(MAX_BLOCKS_1D);
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (SQ_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let (off, n) = (len_u64(e.offset), len_u64(e.len));
        let mut b = rt.stream().launch_builder(&f);
        b.arg(g.slice())
            .arg(&off)
            .arg(&n)
            .arg(partials.slice_mut())
            .arg(&part_off);
        // SAFETY: (const float* g, unsigned long long off, unsigned long long
        // n, float* partials, unsigned long long part_off). off + n <=
        // bank_len = g.len(); the kernel writes partials[part_off + c] for c <
        // ceil(n / 4096) = chunks, and part_off + chunks <= total =
        // partials.len() because part_off sums the earlier windows' chunks.
        // The block is SQ_BLOCK = 256 threads, the size of its shared array.
        unsafe { b.launch(cfg) }.map_err(|err| driver_error(E, err))?;
        part_off += chunks;
    }
    let host = rt.download(&partials)?;
    Ok((sum_partials_f64(&host), host))
}

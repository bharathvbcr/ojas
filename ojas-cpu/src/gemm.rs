//! The one GEMM core: `C[m, n] = A[m, k] · B[k, n]`.
//!
//! Operands are strided views ([`Mat`]), so `x·Wᵀ`, `g·W`, `gᵀ·x` and the
//! Newton-Schulz products differ only in strides. Both operands are packed
//! once: `A` into `MR`-row panels and `B` into `NR`-column panels, each panel
//! stored `k`-major so one step of the reduction is one array. The output is
//! cut into 2D tiles (multiples of `MR` x `NR`), and the pool runs tiles. A
//! tile walks `NC` columns, then `KC` reduction steps, then `MC` rows, then
//! `NR`/`MR` register tiles. The reduction axis is never split across tasks.
//!
//! The exact kernel starts each tile from `+0.0` on the first `KC` block and
//! loads the `C` tile before each later one, adds
//! `a * b` for `k` ascending without `mul_add`, and stores it back, so every
//! output is `((0 + a0*b0) + a1*b1) + ...` in ascending `k`, whatever the
//! tiling or thread count. The fast kernel uses `f32::mul_add` in the same
//! ascending order with its own blocking; its bits also do not depend on the
//! tiling or the thread count.
//!
//! A Fast product of at least [`FAST_WHOLE_CALL_MACS`] skips the packed kernel.
//! On macOS it is one `ojas_simd::sgemm_accelerate` call over the whole
//! matrix, never split across the pool; Accelerate picks its own order and
//! threads, so those bits are only as repeatable as Accelerate is on one
//! machine and OS build. Elsewhere, or when Accelerate refuses the layout, it
//! is `ojas_simd::sgemm_tile` over the same output tiles as the packed path,
//! each with the full `k`, which keeps the bits independent of the tiling.
//! The cutoff depends only on the shape, not on the thread count.
//!
//! The pool's cancel hook runs between row panels of the packed path and
//! before each `sgemm_tile` tile. An Accelerate call cannot be interrupted:
//! the hook runs once before it and is not polled while it runs.

use std::ops::Range;
use std::sync::Arc;

use ojas_core::{BackendId, Numerics, OjasError};

use crate::pool::{scoped, Exec};
use crate::validate::{product, shape};

/// Register tile rows (A panel width).
pub(crate) const MR: usize = 6;
/// Register tile columns (B panel width). 6 x 16 keeps 24 four-lane
/// accumulators in registers on aarch64 and x86-64.
pub(crate) const NR: usize = 16;

/// One register tile: `c_tile[i * ldc + j] += sum_{p < k} a_panel[p][i] * b_panel[p][j]`
/// for `i < m <= MR`, `j < n <= NR`, `p` ascending. Panel lanes at or past
/// `m` / `n` are zero padding and must not be written to `c_tile`.
///
/// `fresh` marks the tile's first reduction block: `c_tile` still holds the
/// `+0.0` it was allocated with, so the kernel starts its sums from `+0.0`
/// without reading it. The result is the same bits as loading it.
///
/// This is the plug-in boundary for a vendor or SIMD fast kernel.
pub(crate) type TileKernel = fn(
    m: usize,
    n: usize,
    k: usize,
    a_panel: &[[f32; MR]],
    b_panel: &[[f32; NR]],
    c_tile: &mut [f32],
    ldc: usize,
    fresh: bool,
);

#[derive(Clone, Copy)]
struct Blocking {
    kc: usize,
    mc: usize,
    nc: usize,
    kernel: TileKernel,
}

const EXACT: Blocking = Blocking {
    kc: 256,
    mc: 72,
    nc: 256,
    kernel: tile_exact,
};

const FAST: Blocking = Blocking {
    kc: 512,
    mc: 144,
    nc: 512,
    kernel: tile_fast,
};

const _: () = assert!(EXACT.mc.is_multiple_of(MR) && FAST.mc.is_multiple_of(MR));
const _: () = assert!(EXACT.nc.is_multiple_of(NR) && FAST.nc.is_multiple_of(NR));
const _: () = assert!(NR.is_multiple_of(4));

fn blocking(numerics: Numerics) -> Blocking {
    match numerics {
        Numerics::Exact => EXACT,
        Numerics::Fast => FAST,
    }
}

/// Multiply-adds per pool task (about 35 µs on one core). A product with
/// fewer than two tasks' worth stays on the calling thread.
pub(crate) const TASK_MACS: usize = 1 << 20;
/// Most tiles [`plan`] cuts per pool thread. More, smaller tiles let the
/// threads that run fast (P-cores, or cores other processes leave alone)
/// take work from the slow ones. At 512x768x768 Exact on a loaded 6P+12E
/// M5 Pro, 6 per thread beat 2 by about 6% (min of 6 interleaved runs) at
/// 6 threads and did not lose at 18.
const TILES_PER_THREAD: usize = 6;
/// Packed operands below this many floats are packed on the calling thread.
const PACK_PAR_MIN: usize = 1 << 18;
/// Multiply-adds at which a Fast product leaves the packed kernel for one
/// whole `ojas-simd` call.
///
/// On macOS that call is Accelerate. `bench_packed_against_accelerate`
/// (2026-10-01, M5 Pro, loaded) put the crossover between 4K and 8K
/// multiply-adds: the two paths tie at 16x16x32, Accelerate is 1.6x faster at
/// 24x24x24, 5x at 64x64x128 and 27-41x at 1x768x768, where the packed path
/// re-packs the whole weight for one row. Elsewhere the call is
/// `sgemm_tile`, which has not been measured against the packed kernel at
/// small sizes, so the cutoff stays where the packed path starts splitting
/// tiles across threads (two `TASK_MACS`).
#[cfg(target_os = "macos")]
pub const FAST_WHOLE_CALL_MACS: usize = 1 << 13;
/// See the macOS definition.
#[cfg(not(target_os = "macos"))]
pub const FAST_WHOLE_CALL_MACS: usize = 2 * TASK_MACS;

#[cfg(test)]
thread_local! {
    /// Whole calls entered on this thread: (Accelerate, `sgemm_tile`).
    static WHOLE_CALLS: std::cell::Cell<(usize, usize)> = const { std::cell::Cell::new((0, 0)) };
}

/// `a [m, k] · b [k, n]` under `numerics` bypasses the packed kernel.
pub(crate) fn whole_call(numerics: Numerics, m: usize, k: usize, n: usize) -> bool {
    numerics == Numerics::Fast && m.saturating_mul(n).saturating_mul(k) >= FAST_WHOLE_CALL_MACS
}

/// Read-only strided matrix: element `(i, j)` is `data[i * rs + j * cs]`.
///
/// It borrows its values, so a tensor's own storage
/// ([`ojas_core::Tensor::f32_slice`]) is an operand with no copy. The pool's
/// tasks are `'static`, so a pass that must read the operand on several
/// threads runs on [`crate::pool::scoped`] instead.
#[derive(Clone, Copy)]
pub(crate) struct Mat<'a> {
    pub data: &'a [f32],
    pub rows: usize,
    pub cols: usize,
    pub rs: usize,
    pub cs: usize,
}

impl<'a> Mat<'a> {
    pub(crate) fn row_major(data: &'a [f32], rows: usize, cols: usize) -> Self {
        Self {
            data,
            rows,
            cols,
            rs: cols,
            cs: 1,
        }
    }

    pub(crate) fn t(&self) -> Self {
        Self {
            data: self.data,
            rows: self.cols,
            cols: self.rows,
            rs: self.cs,
            cs: self.rs,
        }
    }

    /// Every index the view can reach is inside `data`.
    fn check(&self, op: &'static str) -> Result<(), OjasError> {
        if self.rows == 0 || self.cols == 0 {
            return Ok(());
        }
        let last = (self.rows - 1).checked_mul(self.rs).and_then(|a| {
            (self.cols - 1)
                .checked_mul(self.cs)
                .and_then(|b| a.checked_add(b))
        });
        match last {
            Some(last) if last < self.data.len() => Ok(()),
            _ => Err(shape(
                op,
                format!(
                    "gemm view {}x{} strides ({}, {}) exceeds data length {}",
                    self.rows,
                    self.cols,
                    self.rs,
                    self.cs,
                    self.data.len()
                ),
            )),
        }
    }
}

/// Panels of `W` lanes, `depth` reduction steps each, stored in chunks of
/// `per_chunk` panels so separate tasks can pack separate chunks.
struct Packed<const W: usize> {
    chunks: Vec<Vec<[f32; W]>>,
    per_chunk: usize,
    depth: usize,
}

impl<const W: usize> Packed<W> {
    fn panel(&self, index: usize, start: usize, len: usize) -> &[[f32; W]] {
        let chunk = &self.chunks[index / self.per_chunk];
        let base = (index % self.per_chunk) * self.depth + start;
        &chunk[base..base + len]
    }
}

/// Floats [`gemm`] allocates besides its `m * n` output.
pub(crate) fn scratch(
    op: &'static str,
    exec: Exec<'_>,
    m: usize,
    k: usize,
    n: usize,
) -> Result<usize, OjasError> {
    if m == 0 || n == 0 || k == 0 {
        return Ok(0);
    }
    let a = product(op, &[m.div_ceil(MR), MR, k])?;
    let b = product(op, &[n.div_ceil(NR), NR, k])?;
    let tiles = if plan(exec, m, n, k).count() > 1 {
        product(op, &[m, n])?
    } else {
        0
    };
    a.checked_add(b)
        .and_then(|v| v.checked_add(tiles))
        .ok_or_else(|| OjasError::OutOfRange {
            op,
            detail: "gemm scratch length overflows".to_string(),
        })
}

/// `A · B` as a row-major `[a.rows, b.cols]` vector.
pub(crate) fn gemm(
    op: &'static str,
    exec: Exec<'_>,
    a: &Mat,
    b: &Mat,
) -> Result<Vec<f32>, OjasError> {
    let len = operands(op, a, b)?;
    let mut c = vec![0.0f32; len];
    gemm_into(op, exec, a, b, &mut c, false)?;
    Ok(c)
}

/// `c += A · B`, `c` row-major `[a.rows, b.cols]`.
///
/// Every output continues from the value in `c` with the same steps
/// [`gemm`] takes from `+0.0`, so under [`Numerics::Exact`] a reduction
/// split into consecutive pieces of `k`, accumulated in order into a zeroed
/// `c`, gives the bits of one [`gemm`] over the whole `k`. A Fast whole call
/// passes `accumulate` to `ojas-simd`. Besides [`scratch`], a product that
/// [`plan`] splits across tasks holds an `m * n` copy of `c`.
pub(crate) fn gemm_acc(
    op: &'static str,
    exec: Exec<'_>,
    a: &Mat,
    b: &Mat,
    c: &mut [f32],
) -> Result<(), OjasError> {
    let len = operands(op, a, b)?;
    if c.len() != len {
        return Err(shape(
            op,
            format!("gemm accumulator length {} != {len}", c.len()),
        ));
    }
    gemm_into(op, exec, a, b, c, true)
}

/// Inner dimensions agree and both views stay inside their data. Returns the
/// output length.
fn operands(op: &'static str, a: &Mat, b: &Mat) -> Result<usize, OjasError> {
    if b.rows != a.cols {
        return Err(shape(
            op,
            format!(
                "gemm inner dims {}x{} · {}x{} differ",
                a.rows, a.cols, b.rows, b.cols
            ),
        ));
    }
    a.check(op)?;
    b.check(op)?;
    product(op, &[a.rows, b.cols])
}

/// `c = A · B` over a zeroed `c`, or `c += A · B` with `accumulate`.
fn gemm_into(
    op: &'static str,
    exec: Exec<'_>,
    a: &Mat,
    b: &Mat,
    c: &mut [f32],
    accumulate: bool,
) -> Result<(), OjasError> {
    let (m, k, n) = (a.rows, a.cols, b.cols);
    if c.is_empty() || k == 0 {
        return Ok(());
    }
    if whole_call(exec.numerics, m, k, n) {
        #[cfg(target_os = "macos")]
        {
            // Accelerate cannot be interrupted, so the hook is consulted
            // once before the call and not while it runs.
            exec.pool.cancel_hook()()?;
            if accelerate_into(op, a, b, c, accumulate)? {
                return Ok(());
            }
        }
        return simd_tiles_into(op, exec, a, b, c, accumulate);
    }
    packed_into(exec, a, b, c, accumulate)
}

/// The packed kernel over output tiles on the pool: [`gemm_into`] below the
/// whole-call cutoff. `c` is non-empty and `k > 0`.
fn packed_into(
    exec: Exec<'_>,
    a: &Mat,
    b: &Mat,
    c: &mut [f32],
    accumulate: bool,
) -> Result<(), OjasError> {
    let (m, k, n) = (a.rows, a.cols, b.cols);
    let bl = blocking(exec.numerics);
    let pa = Arc::new(pack::<MR>(exec, a.data, m, a.rs, k, a.cs)?);
    let pb = Arc::new(pack::<NR>(exec, b.data, n, b.cs, k, b.rs)?);
    let tiles = plan(exec, m, n, k);
    let cancel = exec.pool.cancel_hook();
    if tiles.count() == 1 {
        return macro_tile(&pa, &pb, 0..m, 0..n, k, bl, c, cancel.as_ref(), accumulate);
    }
    let init = accumulate.then(|| Arc::new(c.to_vec()));
    let parts = exec.pool.run(tiles.count(), {
        let cancel = Arc::clone(&cancel);
        move |t| {
            let (rows, cols) = tiles.tile(t);
            let mut w = window(init.as_deref().map(Vec::as_slice), n, &rows, &cols);
            macro_tile(
                &pa,
                &pb,
                rows,
                cols,
                k,
                bl,
                &mut w,
                cancel.as_ref(),
                accumulate,
            )
            .map(|()| w)
        }
    })?;
    let parts = parts.into_iter().collect::<Result<Vec<_>, _>>()?;
    assemble(tiles, &parts, c);
    Ok(())
}

/// A tile's row-major window: zeros, or a copy of `init` (row-major, `n`
/// columns) to accumulate onto.
fn window(init: Option<&[f32]>, n: usize, rows: &Range<usize>, cols: &Range<usize>) -> Vec<f32> {
    match init {
        None => vec![0.0f32; rows.len() * cols.len()],
        Some(c) => {
            let mut w = Vec::with_capacity(rows.len() * cols.len());
            for row in rows.clone() {
                w.extend_from_slice(&c[row * n + cols.start..row * n + cols.end]);
            }
            w
        }
    }
}

/// [`accelerate_into`] over a fresh output. `None` when Accelerate refused.
#[cfg(all(test, target_os = "macos"))]
fn accelerate(
    op: &'static str,
    a: &Mat,
    b: &Mat,
    len: usize,
) -> Result<Option<Vec<f32>>, OjasError> {
    let mut c = vec![0.0f32; len];
    Ok(accelerate_into(op, a, b, &mut c, false)?.then_some(c))
}

/// One `cblas_sgemm` over the whole product, on the calling thread. `false`
/// (with `c` untouched) when Accelerate cannot address a view, so the caller
/// falls back.
#[cfg(target_os = "macos")]
fn accelerate_into(
    op: &'static str,
    a: &Mat,
    b: &Mat,
    c: &mut [f32],
    accumulate: bool,
) -> Result<bool, OjasError> {
    use ojas_simd::SimdError;
    let (m, k, n) = (a.rows, a.cols, b.cols);
    match ojas_simd::sgemm_accelerate(
        m, n, k, a.data, a.rs, a.cs, b.data, b.rs, b.cs, c, n, accumulate,
    ) {
        Ok(()) => {
            #[cfg(test)]
            WHOLE_CALLS.with(|calls| calls.set((calls.get().0 + 1, calls.get().1)));
            Ok(true)
        }
        Err(SimdError::UnsupportedLayout { .. } | SimdError::DimensionTooLarge { .. }) => Ok(false),
        Err(err) => Err(simd_error(op, err)),
    }
}

/// [`simd_tiles_into`] over a fresh output.
#[cfg(test)]
fn simd_tiles(
    op: &'static str,
    exec: Exec<'_>,
    a: &Mat,
    b: &Mat,
    len: usize,
) -> Result<Vec<f32>, OjasError> {
    let mut c = vec![0.0f32; len];
    simd_tiles_into(op, exec, a, b, &mut c, false)?;
    Ok(c)
}

/// `ojas_simd::sgemm_tile` on the [`plan`] tiles, each with the full `k`.
fn simd_tiles_into(
    op: &'static str,
    exec: Exec<'_>,
    a: &Mat,
    b: &Mat,
    c: &mut [f32],
    accumulate: bool,
) -> Result<(), OjasError> {
    #[cfg(test)]
    WHOLE_CALLS.with(|calls| calls.set((calls.get().0, calls.get().1 + 1)));
    let (m, k, n) = (a.rows, a.cols, b.cols);
    let tiles = plan(exec, m, n, k);
    let cancel = exec.pool.cancel_hook();
    let tile = |rows: Range<usize>, cols: Range<usize>, c: &mut [f32]| {
        ojas_simd::sgemm_tile(
            rows.len(),
            cols.len(),
            k,
            &a.data[rows.start * a.rs..],
            a.rs,
            a.cs,
            &b.data[cols.start * b.cs..],
            b.rs,
            b.cs,
            c,
            cols.len(),
            accumulate,
        )
    };
    if tiles.count() == 1 {
        cancel()?;
        return tile(0..m, 0..n, c).map_err(|err| simd_error(op, err));
    }
    // The tiles run on scoped threads, so they read the operands and, to
    // accumulate, the current `c` where they are.
    let parts = {
        let init = accumulate.then_some(&*c);
        scoped::map(exec, tiles.count(), |t| {
            cancel()?;
            let (rows, cols) = tiles.tile(t);
            let mut w = window(init, n, &rows, &cols);
            tile(rows, cols, &mut w).map_err(|err| simd_error(op, err))?;
            Ok(w)
        })?
    };
    assemble(tiles, &parts, c);
    Ok(())
}

fn simd_error(op: &'static str, err: ojas_simd::SimdError) -> OjasError {
    OjasError::Backend {
        id: BackendId::Cpu,
        detail: format!("{op}: ojas-simd refused a validated gemm view: {err}"),
    }
}

/// Copy the per-tile row-major windows into the row-major `[m, n]` output.
fn assemble(tiles: Tiles, parts: &[Vec<f32>], c: &mut [f32]) {
    let n = tiles.n;
    for (t, part) in parts.iter().enumerate() {
        let (rows, cols) = tiles.tile(t);
        let width = cols.len();
        for (local, row) in rows.enumerate() {
            c[row * n + cols.start..row * n + cols.end]
                .copy_from_slice(&part[local * width..(local + 1) * width]);
        }
    }
}

/// Output tiles of `tm` x `tn` (multiples of `MR` x `NR`).
#[derive(Clone, Copy)]
struct Tiles {
    m: usize,
    n: usize,
    tm: usize,
    tn: usize,
}

impl Tiles {
    fn col_tiles(&self) -> usize {
        self.n.div_ceil(self.tn)
    }

    fn count(&self) -> usize {
        self.m.div_ceil(self.tm) * self.col_tiles()
    }

    fn tile(&self, t: usize) -> (Range<usize>, Range<usize>) {
        let (r, c) = (t / self.col_tiles(), t % self.col_tiles());
        (
            r * self.tm..((r + 1) * self.tm).min(self.m),
            c * self.tn..((c + 1) * self.tn).min(self.n),
        )
    }
}

/// `a [m, k] · b [k, n]` runs as one task on the calling thread.
pub(crate) fn single_task(exec: Exec<'_>, m: usize, k: usize, n: usize) -> bool {
    plan(exec, m, n, k).count() == 1
}

/// One tile on one thread or below two [`TASK_MACS`]; otherwise cache-sized
/// tiles, halved until there is one per [`TASK_MACS`], at most
/// [`TILES_PER_THREAD`] per thread. The plan never changes an
/// output's arithmetic, only which task computes it.
fn plan(exec: Exec<'_>, m: usize, n: usize, k: usize) -> Tiles {
    let one = Tiles { m, n, tm: m, tn: n };
    let threads = exec.pool.threads();
    let macs = m.saturating_mul(n).saturating_mul(k);
    let target = (macs / TASK_MACS).min(threads.saturating_mul(TILES_PER_THREAD));
    if threads <= 1 || target < 2 {
        return one;
    }
    let bl = blocking(exec.numerics);
    let mut tiles = Tiles {
        m,
        n,
        tm: bl.mc.min(round_up(m, MR)),
        tn: bl.nc.min(round_up(n, NR)),
    };
    while tiles.count() < target {
        if tiles.tm >= tiles.tn && tiles.tm > MR {
            tiles.tm = round_up(tiles.tm / 2, MR);
        } else if tiles.tn > NR {
            tiles.tn = round_up(tiles.tn / 2, NR);
        } else if tiles.tm > MR {
            tiles.tm = round_up(tiles.tm / 2, MR);
        } else {
            break;
        }
    }
    if tiles.count() <= 1 {
        one
    } else {
        tiles
    }
}

fn round_up(value: usize, unit: usize) -> usize {
    value.div_ceil(unit).max(1) * unit
}

/// `c` is the row-major `rows x cols` window of the output. Without
/// `accumulate` it holds `+0.0` and the first reduction block starts from
/// that without reading it; with `accumulate` every block loads `c`.
#[allow(clippy::too_many_arguments)]
fn macro_tile(
    pa: &Packed<MR>,
    pb: &Packed<NR>,
    rows: Range<usize>,
    cols: Range<usize>,
    k: usize,
    bl: Blocking,
    c: &mut [f32],
    cancel: &dyn Fn() -> Result<(), OjasError>,
    accumulate: bool,
) -> Result<(), OjasError> {
    let ldc = cols.len();
    let mut jc = cols.start;
    while jc < cols.end {
        let jc_end = (jc + bl.nc).min(cols.end);
        let mut pc = 0;
        while pc < k {
            let kc = bl.kc.min(k - pc);
            let mut ic = rows.start;
            while ic < rows.end {
                // Between row panels, before this panel's first reduction step.
                // Later `k` blocks of the same panel are not interrupted.
                if pc == 0 && jc == cols.start {
                    cancel()?;
                }
                let ic_end = (ic + bl.mc).min(rows.end);
                let mut jr = jc;
                while jr < jc_end {
                    let nr = NR.min(jc_end - jr);
                    let b_panel = pb.panel(jr / NR, pc, kc);
                    let mut ir = ic;
                    while ir < ic_end {
                        let mr = MR.min(ic_end - ir);
                        let a_panel = pa.panel(ir / MR, pc, kc);
                        let offset = (ir - rows.start) * ldc + (jr - cols.start);
                        let fresh = pc == 0 && !accumulate;
                        (bl.kernel)(mr, nr, kc, a_panel, b_panel, &mut c[offset..], ldc, fresh);
                        ir += MR;
                    }
                    jr += NR;
                }
                ic = ic_end;
            }
            pc += kc;
        }
        jc = jc_end;
    }
    Ok(())
}

/// Pack `groups` lanes (stride `gs`) of `depth` steps (stride `ds`) into
/// `W`-lane panels, zero-padding the last panel. A large pack splits its
/// panels across [`scoped`] threads, which read `data` where it is.
fn pack<const W: usize>(
    exec: Exec<'_>,
    data: &[f32],
    groups: usize,
    gs: usize,
    depth: usize,
    ds: usize,
) -> Result<Packed<W>, OjasError> {
    let panels = groups.div_ceil(W);
    let floats = panels.saturating_mul(W).saturating_mul(depth);
    let pieces = if floats < PACK_PAR_MIN {
        1
    } else {
        exec.split(panels, 1)
    };
    let per_chunk = panels.div_ceil(pieces).max(1);
    let chunk_count = panels.div_ceil(per_chunk);
    let chunks = if chunk_count <= 1 {
        vec![pack_range::<W>(data, groups, gs, depth, ds, 0..panels)]
    } else {
        scoped::map(exec, chunk_count, |c| {
            let start = c * per_chunk;
            let end = (start + per_chunk).min(panels);
            Ok(pack_range::<W>(data, groups, gs, depth, ds, start..end))
        })?
    };
    Ok(Packed {
        chunks,
        per_chunk,
        depth,
    })
}

/// Panels in order, `depth` slots each. Lanes at or past `groups` are zero.
fn pack_range<const W: usize>(
    data: &[f32],
    groups: usize,
    gs: usize,
    depth: usize,
    ds: usize,
    panels: Range<usize>,
) -> Vec<[f32; W]> {
    let mut out = Vec::with_capacity(panels.len() * depth);
    for panel in panels {
        let g0 = panel * W;
        let width = W.min(groups - g0);
        if gs == 1 && width == W {
            // Fixed-size copies; a slice copy of runtime length is a memmove call.
            out.extend((0..depth).map(|p| {
                let base = g0 + p * ds;
                <[f32; W]>::try_from(&data[base..base + W]).unwrap_or([0.0f32; W])
            }));
        } else if gs == 1 {
            out.extend((0..depth).map(|p| {
                let base = g0 + p * ds;
                let mut slot = [0.0f32; W];
                slot[..width].copy_from_slice(&data[base..base + width]);
                slot
            }));
        } else if ds == 1 {
            // Each lane is `depth` contiguous values, so this is a transpose:
            // four lanes by four steps at a time, four loads and four stores.
            let start = out.len();
            out.resize(start + depth, [0.0f32; W]);
            let dst = &mut out[start..];
            let quads = width / 4 * 4;
            for lane in (0..quads).step_by(4) {
                let row = |r: usize| &data[(g0 + lane + r) * gs..][..depth];
                let (r0, r1, r2, r3) = (row(0), row(1), row(2), row(3));
                let blocks = dst
                    .as_chunks_mut::<4>()
                    .0
                    .iter_mut()
                    .zip(r0.as_chunks::<4>().0)
                    .zip(r1.as_chunks::<4>().0)
                    .zip(r2.as_chunks::<4>().0)
                    .zip(r3.as_chunks::<4>().0);
                for ((((d, &a0), &a1), &a2), &a3) in blocks {
                    for (i, slot) in d.iter_mut().enumerate() {
                        slot[lane..lane + 4].copy_from_slice(&[a0[i], a1[i], a2[i], a3[i]]);
                    }
                }
                let tail = depth / 4 * 4;
                for (r, run) in [r0, r1, r2, r3].into_iter().enumerate() {
                    for (slot, &value) in dst[tail..].iter_mut().zip(&run[tail..]) {
                        slot[lane + r] = value;
                    }
                }
            }
            for lane in quads..width {
                let run = &data[(g0 + lane) * gs..][..depth];
                for (slot, &value) in dst.iter_mut().zip(run) {
                    slot[lane] = value;
                }
            }
        } else {
            out.extend((0..depth).map(|p| {
                let mut slot = [0.0f32; W];
                for (lane, value) in slot.iter_mut().enumerate().take(width) {
                    *value = data[(g0 + lane) * gs + p * ds];
                }
                slot
            }));
        }
    }
    out
}

/// Four lanes; LLVM maps each to one 128-bit vector register.
type Lanes = [f32; 4];
const GROUPS: usize = NR / 4;
type Acc = [[Lanes; GROUPS]; MR];

/// The accumulator a [`TileKernel`] starts from: `+0.0` on the first
/// reduction block (what `c` holds then), the partial sums in `c` after.
#[inline(always)]
fn start_tile(m: usize, n: usize, c: &[f32], ldc: usize, fresh: bool) -> Acc {
    if fresh {
        [[[0.0f32; 4]; GROUPS]; MR]
    } else {
        load_tile(m, n, c, ldc)
    }
}

fn load_tile(m: usize, n: usize, c: &[f32], ldc: usize) -> Acc {
    let mut acc = [[[0.0f32; 4]; GROUPS]; MR];
    if n == NR {
        for (i, row) in acc.iter_mut().enumerate().take(m) {
            for (g, lanes) in row.iter_mut().enumerate() {
                let at = i * ldc + 4 * g;
                if let Ok(src) = <[f32; 4]>::try_from(&c[at..at + 4]) {
                    *lanes = src;
                }
            }
        }
        return acc;
    }
    for (i, row) in acc.iter_mut().enumerate().take(m) {
        let src = &c[i * ldc..i * ldc + n];
        for (j, &value) in src.iter().enumerate() {
            row[j / 4][j % 4] = value;
        }
    }
    acc
}

fn store_tile(acc: &Acc, m: usize, n: usize, c: &mut [f32], ldc: usize) {
    if n == NR {
        for (i, row) in acc.iter().enumerate().take(m) {
            for (g, lanes) in row.iter().enumerate() {
                let at = i * ldc + 4 * g;
                c[at..at + 4].copy_from_slice(lanes);
            }
        }
        return;
    }
    for (i, row) in acc.iter().enumerate().take(m) {
        let dst = &mut c[i * ldc..i * ldc + n];
        for (j, slot) in dst.iter_mut().enumerate() {
            *slot = row[j / 4][j % 4];
        }
    }
}

#[inline(always)]
fn lanes(b: &[f32; NR], g: usize) -> Lanes {
    [b[4 * g], b[4 * g + 1], b[4 * g + 2], b[4 * g + 3]]
}

#[allow(clippy::too_many_arguments)]
fn tile_exact(
    m: usize,
    n: usize,
    k: usize,
    a_panel: &[[f32; MR]],
    b_panel: &[[f32; NR]],
    c_tile: &mut [f32],
    ldc: usize,
    fresh: bool,
) {
    let mut acc = start_tile(m, n, c_tile, ldc, fresh);
    steps_exact(&mut acc, &a_panel[..k], &b_panel[..k]);
    store_tile(&acc, m, n, c_tile, ldc);
}

/// Kept out of line so the accumulator is only indexed by constants and
/// stays in registers. `acc + s * b`, two roundings, no fusion.
#[inline(never)]
#[allow(clippy::needless_range_loop)]
fn steps_exact(acc: &mut Acc, a: &[[f32; MR]], b: &[[f32; NR]]) {
    let mut r = *acc;
    for (av, bv) in a.iter().zip(b) {
        for i in 0..MR {
            let s = [av[i]; 4];
            for g in 0..GROUPS {
                let bl = lanes(bv, g);
                let c = r[i][g];
                r[i][g] = [
                    c[0] + s[0] * bl[0],
                    c[1] + s[1] * bl[1],
                    c[2] + s[2] * bl[2],
                    c[3] + s[3] * bl[3],
                ];
            }
        }
    }
    *acc = r;
}

#[allow(clippy::too_many_arguments)]
fn tile_fast(
    m: usize,
    n: usize,
    k: usize,
    a_panel: &[[f32; MR]],
    b_panel: &[[f32; NR]],
    c_tile: &mut [f32],
    ldc: usize,
    fresh: bool,
) {
    let mut acc = start_tile(m, n, c_tile, ldc, fresh);
    steps_fast(&mut acc, &a_panel[..k], &b_panel[..k]);
    store_tile(&acc, m, n, c_tile, ldc);
}

/// `mul_add` lowers to a vector FMA where the target has one. Without a
/// hardware FMA it would call the libm routine per lane, so that target keeps
/// a multiply and an add (still a fixed, thread-independent order).
#[inline(never)]
#[allow(clippy::needless_range_loop)]
fn steps_fast(acc: &mut Acc, a: &[[f32; MR]], b: &[[f32; NR]]) {
    let mut r = *acc;
    for (av, bv) in a.iter().zip(b) {
        for i in 0..MR {
            let s = av[i];
            for g in 0..GROUPS {
                let bl = lanes(bv, g);
                let c = r[i][g];
                r[i][g] = [
                    fma(s, bl[0], c[0]),
                    fma(s, bl[1], c[1]),
                    fma(s, bl[2], c[2]),
                    fma(s, bl[3], c[3]),
                ];
            }
        }
    }
    *acc = r;
}

#[cfg(any(target_arch = "aarch64", target_feature = "fma"))]
#[inline(always)]
pub(crate) fn fma(a: f32, b: f32, c: f32) -> f32 {
    a.mul_add(b, c)
}

#[cfg(not(any(target_arch = "aarch64", target_feature = "fma")))]
#[inline(always)]
pub(crate) fn fma(a: f32, b: f32, c: f32) -> f32 {
    a * b + c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pool::Pool;

    fn naive(a: &Mat, b: &Mat) -> Vec<f32> {
        let mut c = vec![0.0f32; a.rows * b.cols];
        for i in 0..a.rows {
            for j in 0..b.cols {
                let mut acc = 0.0f32;
                for p in 0..a.cols {
                    acc += a.data[i * a.rs + p * a.cs] * b.data[p * b.rs + j * b.cs];
                }
                c[i * b.cols + j] = acc;
            }
        }
        c
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn exact_matches_scalar_reference_on_every_stride_layout_and_thread_count() {
        let mut seed = 0x1234_5678u64;
        let mut rand = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        };
        // Both sides of MR/NR, MC (72), NC (256) and KC (256).
        let shapes = [
            (1, 1, 1),
            (7, 9, 17),
            (71, 300, 73),
            (73, 11, 13),
            (65, 257, 63),
            (129, 255, 257),
            (64, 512, 256),
        ];
        for threads in [1usize, 2, 3, 7, 16, 18] {
            let pool = Arc::new(Pool::new(threads).unwrap());
            for numerics in [Numerics::Exact] {
                let exec = Exec {
                    pool: &pool,
                    numerics,
                };
                for &(m, k, n) in &shapes {
                    let a_data: Vec<f32> = (0..m * k).map(|_| rand()).collect();
                    let b_data: Vec<f32> = (0..k * n).map(|_| rand()).collect();
                    let a = Mat::row_major(&a_data, m, k);
                    let b = Mat::row_major(&b_data, k, n);
                    // Row-major, transposed B, transposed A, both transposed.
                    let (at_data, bt_data) = (naive_t(&a), naive_t(&b));
                    let at = Mat::row_major(&at_data, k, m).t();
                    let bt = Mat::row_major(&bt_data, n, k).t();
                    // Neither stride is 1: the general packing path.
                    let (sa, sb) = (spread(&a), spread(&b));
                    let (as_, bs) = (sa.mat(), sb.mat());
                    let want = naive(&a, &b);
                    for (x, y) in [
                        (&a, &b),
                        (&a, &bt),
                        (&at, &b),
                        (&at, &bt),
                        (&as_, &bs),
                        (&as_, &bt),
                        (&at, &bs),
                    ] {
                        let got = gemm("test", exec, x, y).unwrap();
                        assert_eq!(bits(&got), bits(&want), "{m}x{k}x{n} threads {threads}");
                    }
                }
            }
        }
    }

    /// One `mul_add` chain per output, `p` ascending from `+0.0`.
    fn naive_fma(a: &Mat, b: &Mat) -> Vec<f32> {
        let mut c = vec![0.0f32; a.rows * b.cols];
        for i in 0..a.rows {
            for j in 0..b.cols {
                let mut acc = 0.0f32;
                for p in 0..a.cols {
                    acc = a.data[i * a.rs + p * a.cs].mul_add(b.data[p * b.rs + j * b.cs], acc);
                }
                c[i * b.cols + j] = acc;
            }
        }
        c
    }

    /// The off-macOS whole-call path: `sgemm_tile` over pool tiles gives the
    /// ascending FMA chain on every stride layout and thread count.
    #[test]
    fn simd_tiles_match_the_fma_chain_on_every_layout_and_thread_count() {
        let mut seed = 0x9e37_79b9u64;
        let mut rand = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        };
        let shapes = [(1, 1, 1), (73, 11, 13), (129, 255, 257), (64, 512, 256)];
        for threads in [1usize, 2, 3, 7, 16, 18] {
            let pool = Arc::new(Pool::new(threads).unwrap());
            let exec = Exec {
                pool: &pool,
                numerics: Numerics::Fast,
            };
            for &(m, k, n) in &shapes {
                let a_data: Vec<f32> = (0..m * k).map(|_| rand()).collect();
                let b_data: Vec<f32> = (0..k * n).map(|_| rand()).collect();
                let a = Mat::row_major(&a_data, m, k);
                let b = Mat::row_major(&b_data, k, n);
                let (at_data, bt_data) = (naive_t(&a), naive_t(&b));
                let at = Mat::row_major(&at_data, k, m).t();
                let bt = Mat::row_major(&bt_data, n, k).t();
                let want = naive_fma(&a, &b);
                for (x, y) in [(&a, &b), (&a, &bt), (&at, &b), (&at, &bt)] {
                    let got = simd_tiles("test", exec, x, y, m * n).unwrap();
                    assert_eq!(bits(&got), bits(&want), "{m}x{k}x{n} threads {threads}");
                }
            }
        }
    }

    fn random(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect()
    }

    /// Below the cutoff Fast stays on `tile_fast`, which is the same chain.
    #[cfg(any(target_arch = "aarch64", target_feature = "fma"))]
    #[test]
    fn fast_below_the_whole_call_cutoff_is_the_packed_fma_chain() {
        let pool = Arc::new(Pool::new(7).unwrap());
        let exec = Exec {
            pool: &pool,
            numerics: Numerics::Fast,
        };
        // Shapes that cross MR (6), NR (16), MC (72) and KC (256) while
        // staying below the macOS cutoff come first; the larger ones run
        // only where the cutoff is higher.
        let shapes = [
            (7, 9, 17),
            (73, 7, 13),
            (1, 300, 17),
            (6, 40, 33),
            (128, 64, 96),
            (127, 127, 129),
        ];
        let below: Vec<_> = shapes
            .into_iter()
            .filter(|&(m, k, n)| !whole_call(Numerics::Fast, m, k, n))
            .collect();
        assert!(below.len() >= 4, "{below:?}");
        for (m, k, n) in below {
            let (ad, bd) = (random(m * k, 1), random(k * n, 2));
            let a = Mat::row_major(&ad, m, k);
            let b = Mat::row_major(&bd, k, n);
            let got = gemm("test", exec, &a, &b).unwrap();
            assert_eq!(bits(&got), bits(&naive_fma(&a, &b)), "{m}x{k}x{n}");
            assert_ne!(bits(&got), bits(&naive(&a, &b)), "{m}x{k}x{n}");
        }
    }

    #[test]
    fn whole_call_cutoff_is_fast_only_and_shape_only() {
        #[cfg(target_os = "macos")]
        assert_eq!(FAST_WHOLE_CALL_MACS, 1 << 13);
        #[cfg(not(target_os = "macos"))]
        assert_eq!(FAST_WHOLE_CALL_MACS, 2 * TASK_MACS);
        let c = FAST_WHOLE_CALL_MACS;
        assert!(whole_call(Numerics::Fast, c, 1, 1));
        assert!(whole_call(Numerics::Fast, 1, c, 1));
        assert!(!whole_call(Numerics::Fast, c - 1, 1, 1));
        assert!(!whole_call(Numerics::Exact, 2048, 2048, 2048));
        assert!(whole_call(Numerics::Fast, usize::MAX, usize::MAX, 2));
    }

    fn whole_calls() -> (usize, usize) {
        WHOLE_CALLS.with(std::cell::Cell::get)
    }

    /// What `ojas_core::linear_forward_dims` returns for `[rows, kin]` by
    /// `[nout, kin]`.
    fn linear_dims(rows: usize, kin: usize, nout: usize) -> ojas_core::LinearDims {
        ojas_core::LinearDims {
            rows,
            in_features: kin,
            out_features: nout,
            out_shape: vec![rows, nout],
        }
    }

    /// Fast products at or above the cutoff make exactly one `ojas-simd`
    /// entry each, on the calling thread: Accelerate on macOS, `sgemm_tile`
    /// elsewhere. Smaller Fast products and every Exact product make none.
    #[test]
    fn fast_whole_products_take_one_ojas_simd_call_and_exact_never_does() {
        let budget = ojas_core::Budget::new(1 << 30);
        for threads in [1usize, 18] {
            let pool = Arc::new(Pool::new(threads).unwrap());
            for numerics in [Numerics::Exact, Numerics::Fast] {
                let exec = Exec {
                    pool: &pool,
                    numerics,
                };
                for (rows, kin, nout) in [(7, 9, 17), (128, 64, 96), (256, 256, 256)] {
                    let x = random(rows * kin, 3);
                    let w = random(nout * kin, 4);
                    let g = random(rows * nout, 5);
                    let before = whole_calls();
                    let op = "test";
                    let dims = linear_dims(rows, kin, nout);
                    crate::linalg::linear_forward(op, &budget, exec, &x, &w, &dims).unwrap();
                    crate::linalg::linear_backward(op, &budget, exec, &x, &w, &g, &dims).unwrap();
                    let xm = Mat::row_major(&x, rows, kin);
                    gemm(op, exec, &xm, &xm.t()).unwrap();
                    let after = whole_calls();
                    let made = (after.0 - before.0, after.1 - before.1);
                    // Forward, the two backward products, then x·xᵀ.
                    let macs = [
                        rows * kin * nout,
                        rows * nout * kin,
                        nout * rows * kin,
                        rows * kin * rows,
                    ];
                    let calls = if numerics == Numerics::Fast {
                        macs.iter().filter(|&&m| m >= FAST_WHOLE_CALL_MACS).count()
                    } else {
                        0
                    };
                    let want = if cfg!(target_os = "macos") {
                        (calls, 0)
                    } else {
                        (0, calls)
                    };
                    assert_eq!(
                        made, want,
                        "{numerics:?} {rows}x{kin}x{nout} threads {threads}: (accelerate, sgemm_tile)"
                    );
                }
            }
        }
    }

    /// A whole Fast call consults the cancel hook before it starts, so
    /// `linear_backward`'s two whole calls each see it: a hook that fails on
    /// its second call stops the backward after the first product.
    #[test]
    fn whole_calls_consult_the_cancel_hook_before_each_product() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let (rows, kin, nout) = (128usize, 128usize, 128usize);
        assert!(whole_call(Numerics::Fast, rows, nout, kin));
        assert!(whole_call(Numerics::Fast, nout, rows, kin));
        let pool = Arc::new(Pool::new(3).unwrap());
        let calls = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&calls);
        pool.set_cancel(Arc::new(move || {
            if seen.fetch_add(1, Ordering::SeqCst) == 1 {
                Err(OjasError::Unsupported {
                    op: "test-cancel",
                    detail: "cancelled".to_string(),
                })
            } else {
                Ok(())
            }
        }));
        let exec = Exec {
            pool: &pool,
            numerics: Numerics::Fast,
        };
        let budget = ojas_core::Budget::new(1 << 30);
        let err = crate::linalg::linear_backward(
            "test",
            &budget,
            exec,
            &random(rows * kin, 1),
            &random(nout * kin, 2),
            &random(rows * nout, 3),
            &linear_dims(rows, kin, nout),
        )
        .err();
        assert!(
            matches!(
                err,
                Some(OjasError::Unsupported {
                    op: "test-cancel",
                    ..
                })
            ),
            "{err:?}"
        );
        let made = calls.load(Ordering::SeqCst);
        if cfg!(target_os = "macos") {
            assert_eq!(made, 2, "one check before each Accelerate call");
        } else {
            assert!(made >= 2, "{made}");
        }
    }

    /// Views Accelerate cannot address (no unit stride on either axis) fall
    /// back to `sgemm_tile`, which gives the FMA chain.
    #[cfg(target_os = "macos")]
    #[test]
    fn accelerate_refusal_falls_back_to_sgemm_tile() {
        let (m, k, n) = (128usize, 128usize, 128usize);
        assert!(whole_call(Numerics::Fast, m, k, n));
        let (ad, bd) = (random(2 * m * k, 6), random(k * n, 7));
        let a = Mat {
            data: &ad,
            rows: m,
            cols: k,
            rs: 2 * k,
            cs: 2,
        };
        let b = Mat::row_major(&bd, k, n);
        assert!(accelerate("test", &a, &b, m * n).unwrap().is_none());
        let pool = Arc::new(Pool::new(3).unwrap());
        let exec = Exec {
            pool: &pool,
            numerics: Numerics::Fast,
        };
        let before = whole_calls();
        let got = gemm("test", exec, &a, &b).unwrap();
        let after = whole_calls();
        assert_eq!((after.0 - before.0, after.1 - before.1), (0, 1));
        assert_eq!(bits(&got), bits(&naive_fma(&a, &b)));
    }

    /// A strided copy of a matrix, owned so a [`Mat`] can borrow it.
    struct Spread {
        data: Vec<f32>,
        rows: usize,
        cols: usize,
        rs: usize,
        cs: usize,
    }

    impl Spread {
        fn mat(&self) -> Mat<'_> {
            Mat {
                data: &self.data,
                rows: self.rows,
                cols: self.cols,
                rs: self.rs,
                cs: self.cs,
            }
        }
    }

    /// The same matrix with row stride `2 * cols + 3` and column stride 2.
    /// Every slot outside the view is NaN, so reading one shows in the bits.
    fn spread(a: &Mat) -> Spread {
        let (rs, cs) = (2 * a.cols + 3, 2);
        let mut data = vec![f32::NAN; a.rows * rs];
        for i in 0..a.rows {
            for j in 0..a.cols {
                data[i * rs + j * cs] = a.data[i * a.rs + j * a.cs];
            }
        }
        Spread {
            data,
            rows: a.rows,
            cols: a.cols,
            rs,
            cs,
        }
    }

    fn naive_t(a: &Mat) -> Vec<f32> {
        let mut out = vec![0.0f32; a.rows * a.cols];
        for i in 0..a.rows {
            for j in 0..a.cols {
                out[j * a.rows + i] = a.data[i * a.rs + j * a.cs];
            }
        }
        out
    }

    #[test]
    fn shapes_past_two_task_macs_use_more_than_one_tile() {
        let pool = Arc::new(Pool::new(4).unwrap());
        let exec = Exec {
            pool: &pool,
            numerics: Numerics::Exact,
        };
        // 32 * 256 * 256 = 2 * TASK_MACS. plan(m, n, k).
        assert!(plan(exec, 32, 256, 256).count() > 1);
        // rows == 1, kin = 2048, nout = 1024.
        assert!(plan(exec, 1, 1024, 2048).count() > 1);
        // 64 * 96 * 128 = 786_432, one below the old PARALLEL_AT band and still one tile.
        assert_eq!(plan(exec, 64, 96, 128).count(), 1);
    }

    #[test]
    fn short_data_and_mismatched_inner_dims_are_shape_errors() {
        let pool = Arc::new(Pool::new(1).unwrap());
        let exec = Exec {
            pool: &pool,
            numerics: Numerics::Exact,
        };
        let (five, six) = (vec![1.0f32; 5], vec![1.0f32; 6]);
        let a = Mat::row_major(&five, 2, 3);
        let b = Mat::row_major(&six, 3, 2);
        assert!(matches!(
            gemm("t", exec, &a, &b),
            Err(OjasError::Shape { .. })
        ));
        let a = Mat::row_major(&six, 2, 3);
        let b = Mat::row_major(&six, 2, 3);
        assert!(matches!(
            gemm("t", exec, &a, &b),
            Err(OjasError::Shape { .. })
        ));
        let empty = Mat::row_major(&[], 0, 3);
        let b = Mat::row_major(&six, 3, 2);
        assert!(gemm("t", exec, &empty, &b).unwrap().is_empty());
    }

    /// Fast packed kernel against one Accelerate call, per shape, on a
    /// 6-thread pool and on one thread (the attention blocks' pool-free
    /// `Exec`). Each cell is the minimum over interleaved rounds of the mean
    /// of a batch of calls, so small shapes are not timer-bound.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "timing sweep; run with --ignored --nocapture --test-threads=1"]
    fn bench_packed_against_accelerate() {
        use std::time::Instant;
        const ROUNDS: usize = 9;
        let shapes: &[(usize, usize, usize)] = &[
            (1, 64, 64),
            (1, 256, 256),
            (4, 32, 32),
            (16, 16, 32),
            (20, 20, 20),
            (24, 24, 24),
            (16, 32, 32),
            (28, 28, 28),
            (8, 16, 64),
            (1, 768, 768),
            (1, 768, 2048),
            (1, 2048, 768),
            (2, 768, 768),
            (8, 64, 64),
            (16, 16, 16),
            (32, 32, 32),
            (48, 48, 48),
            (64, 64, 64),
            (64, 64, 128),
            (64, 128, 64),
            (96, 96, 96),
            (128, 64, 128),
            (64, 64, 256),
            (64, 256, 64),
            (112, 112, 112),
            (128, 128, 127),
        ];
        for threads in [6usize, 1] {
            let pool = Arc::new(Pool::new(threads).unwrap());
            let exec = Exec {
                pool: &pool,
                numerics: Numerics::Fast,
            };
            for &(m, k, n) in shapes {
                let macs = m * k * n;
                let fill = |len: usize, salt: u32| -> Vec<f32> {
                    (0..len)
                        .map(|i| {
                            ((i as u32).wrapping_mul(2_654_435_761) ^ salt) as f32 / u32::MAX as f32
                                - 0.5
                        })
                        .collect()
                };
                let (ad, bd) = (fill(m * k, 7), fill(k * n, 11));
                let a = Mat::row_major(&ad, m, k);
                let b = Mat::row_major(&bd, k, n);
                let iters = ((1usize << 24) / macs).clamp(4, 4000);
                let mut c = vec![0.0f32; m * n];
                let (mut packed, mut accel) = (f64::MAX, f64::MAX);
                for _ in 0..ROUNDS {
                    let t0 = Instant::now();
                    for _ in 0..iters {
                        c.fill(0.0);
                        packed_into(exec, &a, &b, &mut c, false).unwrap();
                    }
                    packed = packed.min(t0.elapsed().as_secs_f64() / iters as f64);
                    std::hint::black_box(&c);
                    let t0 = Instant::now();
                    for _ in 0..iters {
                        c.fill(0.0);
                        assert!(accelerate_into("bench", &a, &b, &mut c, false).unwrap());
                    }
                    accel = accel.min(t0.elapsed().as_secs_f64() / iters as f64);
                    std::hint::black_box(&c);
                }
                println!(
                    "GEMM_SWEEP threads={threads} m={m} k={k} n={n} macs={macs} packed_us={:.2} accel_us={:.2} ratio={:.2}",
                    packed * 1e6,
                    accel * 1e6,
                    packed / accel
                );
            }
        }
    }
}

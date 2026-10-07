//! Axis permutation: values move, nothing is computed.

use ojas_core::{permute_dims, Budget, OjasError, Scratch, Tensor};
#[cfg(target_os = "macos")]
use ojas_core::{BackendId, Numerics};

use crate::pool::{scoped, Exec};

use crate::validate::{all_finite, nonfinite, shape};

/// Output floats below this stay on the calling thread.
///
/// A scoped spawn of five threads is about 0.05 ms here. The nanolab permute
/// (`[1, 1024, 12, 64]`, 786_432 floats) is in that neighborhood, so splitting
/// it is slower than the tiled copy below. Larger outputs cover the spawn.
const PARALLEL_MIN_FLOATS: usize = 1 << 22;

/// Source-address span of one tile of the inner axis, in bytes. 192 KiB is
/// one `[64 × 12 × 64]` slab of the nanolab `(0, 2, 1, 3)` permute: every
/// head of those 64 times is still in cache while the tile is written.
const TILE_SPAN_BYTES: usize = 192 * 1024;

/// Caller prefix of the 1024 time rows. The worker copies the other 384.
#[cfg(all(target_os = "macos", target_arch = "aarch64", target_feature = "neon"))]
const NANOLAB_CALLER_ROWS: usize = 640;

#[cfg(all(target_os = "macos", target_arch = "aarch64", target_feature = "neon"))]
fn permute_simd(err: ojas_simd::SimdError) -> OjasError {
    OjasError::Backend {
        id: BackendId::Cpu,
        detail: format!("permute: {err}"),
    }
}

/// Joins a handoff if the caller returns before [`crate::pool::Handoff::join`].
#[cfg(all(target_os = "macos", target_arch = "aarch64", target_feature = "neon"))]
struct HandoffJoin<T: Send> {
    handoff: Option<crate::pool::Handoff<T>>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64", target_feature = "neon"))]
impl<T: Send> Drop for HandoffJoin<T> {
    fn drop(&mut self) {
        if let Some(handoff) = self.handoff.take() {
            let _ = handoff.join();
        }
    }
}

/// `torch.permute(input, dims).contiguous()` for a host f32 tensor.
///
/// The input is read in place from its contiguous window, so no host copy
/// of it is made or charged. The output is a [`Scratch`] charged to
/// `budget` and handed to the tensor without a second copy. Values are
/// copied, never computed, so the output bits are the input bits.
///
/// An identity (and any permutation that only reorders axes of length 1)
/// is one [`Scratch::try_from_slice`] of the window. A strided permute
/// below [`PARALLEL_MIN_FLOATS`] appends each output run with
/// [`Scratch::try_extend`], so each element is written once. On macOS
/// under [`ojas_core::Numerics::Fast`], a rank-4 strided run of 64 floats with more
/// than one row is instead one `vDSP_mmov` per head into that same spare
/// capacity: still one reserve, and the move writes every output lane, so
/// the buffer is not zero-filled. The nanolab shape `[B, 1024, 12, 64]`
/// with `(0, 2, 1, 3)` is the exception: each token's two adjacent heads
/// (128 floats) are loaded once and stored into those two head blocks, in
/// tiles of 32 tokens, still with no zero-fill. One batch with a worker
/// already running hands that worker the last 384 time rows first, then the
/// caller copies the first 640, then waits. No worker stays on the one-call
/// loop. Exact, other ranks, other widths, and a single-row run stay on the
/// append. At or above the cutoff, and whenever
/// the gathered runs are already adjacent, the output is a
/// [`Scratch::try_alloc`] buffer. That buffer is zeroed first: the parallel
/// split and the adjacent-run copies write by index, and an append of a
/// shared vector cannot be split across threads.
///
/// Refused first by [`permute_dims`]: a dtype other than f32, an axis of
/// length 0, and invalid `dims`. Then device memory (from the window read),
/// a strided view, and a NaN or infinity, as every other CPU op refuses
/// one. Rank 0 is accepted. Thread count does not change bits.
pub(crate) fn permute(
    op: &'static str,
    budget: &Budget,
    exec: Exec<'_>,
    input: &Tensor,
    dims: &[usize],
) -> Result<Tensor, OjasError> {
    let in_shape = input.shape();
    let out_shape = permute_dims(input, dims)?;
    if !input.is_contiguous()? {
        return Err(shape(op, "non-contiguous view is not supported"));
    }
    let src = input.f32_slice()?;
    if !input.all_finite_cached(|w| Ok(all_finite(w)))? {
        return Err(nonfinite(op));
    }
    let gather = Gather::plan(in_shape, dims, &out_shape);
    if gather.is_whole() {
        let out = Scratch::try_from_slice(src, budget)?;
        return Tensor::from_scratch(out, &out_shape);
    }
    if gather.appends_on_caller(src.len()) {
        let out = gather.append(exec, src, budget)?;
        return Tensor::from_scratch(out, &out_shape);
    }
    let mut out = Scratch::try_alloc(src.len(), budget)?;
    gather.write(exec, src, out.as_mut_slice())?;
    Tensor::from_scratch(out, &out_shape)
}

/// How to move `src` into an output-ordered buffer.
///
/// Trailing output axes that are the same input axes in the same order are
/// one contiguous run. Axes of length 1 above that run do not move a value.
/// What remains is a head (precomputed source bases) and two inner axes:
/// `mid` varies outside a tile, `inner` varies inside it, and each inner
/// step copies `run` floats. `(0, 2, 1, 3)` on `[B, T, H, D]` has `run = D`,
/// `inner = T`, `mid = H`, and a head of `B` bases.
enum Gather {
    /// Every value already sits where the output wants it.
    Whole,
    Runs(Runs),
}

struct Runs {
    /// Source offset of each head, in output order. Empty when `head_count`
    /// is large enough that the table would dwarf the tensor; those heads
    /// are unraveled from `head_shape` / `head_steps` instead.
    bases: Vec<usize>,
    head_shape: Vec<usize>,
    head_steps: Vec<usize>,
    head_count: usize,
    mid_extent: usize,
    mid_step: usize,
    inner_extent: usize,
    inner_step: usize,
    run: usize,
    /// Input rank. The macOS Fast width-64 move is rank 4 only.
    #[cfg(target_os = "macos")]
    rank: usize,
}

impl Gather {
    fn is_whole(&self) -> bool {
        matches!(self, Gather::Whole)
    }

    fn plan(in_shape: &[usize], dims: &[usize], out_shape: &[usize]) -> Self {
        let mut in_strides = vec![1usize; in_shape.len()];
        for axis in (0..in_shape.len().saturating_sub(1)).rev() {
            in_strides[axis] = in_strides[axis + 1] * in_shape[axis + 1];
        }
        let mut keep = out_shape.len();
        while keep > 0 && dims[keep - 1] == keep - 1 {
            keep -= 1;
        }
        // An axis of length 1 contributes a source offset of 0. Dropping it
        // does not change which input element lands in which output slot.
        while keep > 0 && out_shape[keep - 1] == 1 {
            keep -= 1;
        }
        if keep == 0 {
            return Gather::Whole;
        }
        let run: usize = out_shape[keep..].iter().product();
        let inner_extent = out_shape[keep - 1];
        let inner_step = in_strides[dims[keep - 1]];
        let (mid_extent, mid_step, head_end) = if keep == 1 {
            (1, 0, 0)
        } else {
            (out_shape[keep - 2], in_strides[dims[keep - 2]], keep - 2)
        };
        let head_shape = out_shape[..head_end].to_vec();
        let head_steps: Vec<usize> = dims[..head_end]
            .iter()
            .map(|&axis| in_strides[axis])
            .collect();
        let head_count: usize = head_shape.iter().product();
        // The empty product (no head axes) is one base, at offset 0.
        let head_count = head_count.max(1);
        let bases = if head_shape.is_empty() {
            vec![0]
        } else if head_count <= 65_536 {
            precompute_bases(&head_shape, &head_steps)
        } else {
            Vec::new()
        };
        Gather::Runs(Runs {
            bases,
            head_shape,
            head_steps,
            head_count,
            mid_extent,
            mid_step,
            inner_extent,
            inner_step,
            run,
            #[cfg(target_os = "macos")]
            rank: in_shape.len(),
        })
    }

    fn write(&self, exec: Exec<'_>, src: &[f32], dst: &mut [f32]) -> Result<(), OjasError> {
        let Gather::Runs(runs) = self else {
            dst.copy_from_slice(src);
            return Ok(());
        };
        runs.write(exec, src, dst)
    }

    /// Strided, and small enough to stay on the calling thread.
    ///
    /// At or above [`PARALLEL_MIN_FLOATS`] the parallel path writes by index
    /// into a zeroed buffer; an append cannot be split across threads. Runs
    /// that already sit next to each other are copies into that same buffer.
    fn appends_on_caller(&self, n: usize) -> bool {
        match self {
            Gather::Whole => false,
            Gather::Runs(runs) => n < PARALLEL_MIN_FLOATS && runs.inner_step != runs.run,
        }
    }

    fn append(
        &self,
        exec: Exec<'_>,
        src: &[f32],
        budget: &Budget,
    ) -> Result<Scratch<f32>, OjasError> {
        let Gather::Runs(runs) = self else {
            return Scratch::try_from_slice(src, budget);
        };
        #[cfg(target_os = "macos")]
        if runs.use_mmov(exec.numerics) {
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            if runs.nanolab_pairs() {
                return runs.append_pairs(exec, src, budget);
            }
            return runs.append_mmov(src, budget);
        }
        #[cfg(not(target_os = "macos"))]
        let _ = exec;
        Scratch::try_extend(src.len(), budget, |dst| runs.append_runs(src, dst))
    }
}

impl Runs {
    /// Fast macOS, rank 4, more than one row of 64 strided floats.
    ///
    /// Each head is `inner_extent` rows of 64 columns. The source row stride
    /// is `inner_step` and the destination row stride is 64. Head `mid`
    /// starts at `mid * mid_step`, so the heads are not one matrix. One
    /// `vDSP_mmov` per head, except [`Self::nanolab_pairs`].
    #[cfg(target_os = "macos")]
    fn use_mmov(&self, numerics: Numerics) -> bool {
        numerics == Numerics::Fast
            && self.rank == 4
            && self.run == 64
            && self.inner_extent > 1
            && self.inner_step > self.run
    }

    /// `[B, 1024, 12, 64]` and `(0, 2, 1, 3)`: twelve heads, source pitch 768.
    ///
    /// Adjacent heads are contiguous in the token. The pair move loads 128
    /// floats once per token instead of twelve strided `vDSP_mmov` passes.
    #[cfg(all(target_os = "macos", target_arch = "aarch64", target_feature = "neon"))]
    fn nanolab_pairs(&self) -> bool {
        self.inner_extent == 1024
            && self.mid_extent == 12
            && self.mid_step == 64
            && self.inner_step == 768
            && self.run == 64
    }

    /// One batch at a time, two heads per load, into spare capacity.
    #[cfg(all(target_os = "macos", target_arch = "aarch64", target_feature = "neon"))]
    fn append_pairs(
        &self,
        exec: Exec<'_>,
        src: &[f32],
        budget: &Budget,
    ) -> Result<Scratch<f32>, OjasError> {
        let mut outcome: Result<(), OjasError> = Ok(());
        let scratch = Scratch::try_extend(src.len(), budget, |dst| {
            outcome = self.pairs_into(exec, src, dst);
        });
        match outcome {
            Err(err) => Err(err),
            Ok(()) => scratch,
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64", target_feature = "neon"))]
    fn pairs_into(&self, exec: Exec<'_>, src: &[f32], dst: &mut Vec<f32>) -> Result<(), OjasError> {
        const N: usize = 1024 * 768;
        if self.head_count == 1 && exec.pool.workers_ready() {
            let base = self.head_base(0);
            let end = base
                .checked_add(N)
                .ok_or_else(|| mmov_refused("source end overflow"))?;
            let window = src
                .get(base..end)
                .ok_or_else(|| mmov_refused("source window is short"))?;
            if self.write_caller_band(exec, window, dst, NANOLAB_CALLER_ROWS)? {
                return Ok(());
            }
        }
        for head in 0..self.head_count {
            let base = self.head_base(head);
            let end = base
                .checked_add(N)
                .ok_or_else(|| mmov_refused("source end overflow"))?;
            let window = src
                .get(base..end)
                .ok_or_else(|| mmov_refused("source window is short"))?;
            ojas_simd::split_nanolab_head_pairs_append(window, dst).map_err(permute_simd)?;
        }
        Ok(())
    }

    /// Hand the worker its suffix, copy the caller prefix, then wait.
    ///
    /// `Ok(false)` means no worker took the band; the caller uses the
    /// one-call loop. `caller_rows` is in `1..1024`.
    #[cfg(all(target_os = "macos", target_arch = "aarch64", target_feature = "neon"))]
    fn write_caller_band(
        &self,
        exec: Exec<'_>,
        window: &[f32],
        dst: &mut Vec<f32>,
        caller_rows: usize,
    ) -> Result<bool, OjasError> {
        ojas_simd::with_nanolab_token_bands(window, dst, caller_rows, |caller, worker| {
            let Some(handoff) = exec.pool.try_handoff(move || worker.write()) else {
                return Ok(false);
            };
            let mut guard = HandoffJoin {
                handoff: Some(handoff),
            };
            caller.write();
            let handoff = guard
                .handoff
                .take()
                .ok_or_else(|| mmov_refused("handoff dropped"))?;
            handoff.join().map(|_| true)
        })
        .map_err(permute_simd)?
    }

    /// One `vDSP_mmov` per `(head, mid)` into spare capacity. `__M` is the
    /// column count (`run`) and `__N` is the row count (`inner_extent`).
    #[cfg(target_os = "macos")]
    fn append_mmov(&self, src: &[f32], budget: &Budget) -> Result<Scratch<f32>, OjasError> {
        let mut outcome: Result<(), OjasError> = Ok(());
        let scratch = Scratch::try_extend(src.len(), budget, |dst| {
            outcome = self.mmov_into(src, dst);
        });
        match outcome {
            Err(err) => Err(err),
            Ok(()) => scratch,
        }
    }

    #[cfg(target_os = "macos")]
    fn mmov_into(&self, src: &[f32], dst: &mut Vec<f32>) -> Result<(), OjasError> {
        let rows = self.inner_extent;
        let cols = self.run;
        let stride = self.inner_step;
        let span = (rows - 1)
            .checked_mul(stride)
            .and_then(|span| span.checked_add(cols))
            .ok_or_else(|| mmov_refused("row span overflow"))?;
        for head in 0..self.head_count {
            let base = self.head_base(head);
            for mid in 0..self.mid_extent {
                let from = base
                    .checked_add(
                        mid.checked_mul(self.mid_step)
                            .ok_or_else(|| mmov_refused("mid offset overflow"))?,
                    )
                    .ok_or_else(|| mmov_refused("source offset overflow"))?;
                let end = from
                    .checked_add(span)
                    .ok_or_else(|| mmov_refused("source end overflow"))?;
                let window = src
                    .get(from..end)
                    .ok_or_else(|| mmov_refused("source window is short"))?;
                ojas_simd::vdsp_mmov_append(window, dst, rows, cols, stride).map_err(|err| {
                    OjasError::Backend {
                        id: BackendId::Cpu,
                        detail: format!("permute: {err}"),
                    }
                })?;
            }
        }
        Ok(())
    }

    fn head_base(&self, head: usize) -> usize {
        if self.bases.len() == self.head_count {
            self.bases[head]
        } else {
            unravel(head, &self.head_shape, &self.head_steps)
        }
    }

    fn write(&self, exec: Exec<'_>, src: &[f32], dst: &mut [f32]) -> Result<(), OjasError> {
        if self.inner_step == self.run {
            self.write_contiguous(src, dst);
            return Ok(());
        }
        let n = dst.len();
        if exec.pool.threads() > 1 && n >= PARALLEL_MIN_FLOATS {
            self.write_parallel(exec, src, dst)
        } else {
            self.write_tiled(src, dst);
            Ok(())
        }
    }

    /// Append each output run in row-major order: head, then mid, then inner.
    ///
    /// [`Self::write_tiled`] visits every mid inside one source tile so the
    /// slab stays in cache. That needs a destination it can index. An append
    /// only writes the next run, so one mid is finished before the next.
    fn append_runs(&self, src: &[f32], dst: &mut Vec<f32>) {
        match self.run {
            1 => self.append_n::<1>(src, dst),
            2 => self.append_n::<2>(src, dst),
            4 => self.append_n::<4>(src, dst),
            8 => self.append_n::<8>(src, dst),
            16 => self.append_n::<16>(src, dst),
            32 => self.append_n::<32>(src, dst),
            64 => self.append_n::<64>(src, dst),
            128 => self.append_n::<128>(src, dst),
            _ => self.append_dyn(src, dst),
        }
    }

    fn append_n<const N: usize>(&self, src: &[f32], dst: &mut Vec<f32>) {
        for head in 0..self.head_count {
            let base = self.head_base(head);
            for mid in 0..self.mid_extent {
                let mut from = base + mid * self.mid_step;
                for _ in 0..self.inner_extent {
                    let row: &[f32; N] = src[from..from + N].try_into().unwrap();
                    dst.extend_from_slice(row);
                    from += self.inner_step;
                }
            }
        }
    }

    fn append_dyn(&self, src: &[f32], dst: &mut Vec<f32>) {
        let run = self.run;
        for head in 0..self.head_count {
            let base = self.head_base(head);
            for mid in 0..self.mid_extent {
                let mut from = base + mid * self.mid_step;
                for _ in 0..self.inner_extent {
                    dst.extend_from_slice(&src[from..from + run]);
                    from += self.inner_step;
                }
            }
        }
    }

    /// Inner steps land on adjacent runs, so each `(head, mid)` is one copy.
    /// Adjacent mids collapse into that same copy.
    fn write_contiguous(&self, src: &[f32], dst: &mut [f32]) {
        let row = self.inner_extent * self.run;
        let mids_adjacent = self.mid_extent == 1 || self.mid_step == row;
        if mids_adjacent {
            let block = self.mid_extent * row;
            for head in 0..self.head_count {
                let at = head * block;
                let from = self.head_base(head);
                dst[at..at + block].copy_from_slice(&src[from..from + block]);
            }
            return;
        }
        for head in 0..self.head_count {
            let base = self.head_base(head);
            for mid in 0..self.mid_extent {
                let at = (head * self.mid_extent + mid) * row;
                let from = base + mid * self.mid_step;
                dst[at..at + row].copy_from_slice(&src[from..from + row]);
            }
        }
    }

    fn write_parallel(
        &self,
        exec: Exec<'_>,
        src: &[f32],
        dst: &mut [f32],
    ) -> Result<(), OjasError> {
        let n_runs = self.head_count * self.mid_extent * self.inner_extent;
        scoped::rows_into(exec, dst, n_runs, self.run, |range, part| {
            self.copy_run_range(src, part, range.start, range.end);
            Ok(())
        })?;
        Ok(())
    }

    /// Runs `[start, end)` in output order. The inner axis is a flat add of
    /// `inner_step`; the head and mid indexes are decoded only when that
    /// axis wraps.
    fn copy_run_range(&self, src: &[f32], dst: &mut [f32], start: usize, end: usize) {
        let mut run_i = start;
        let mut at = 0usize;
        let span = self.mid_extent * self.inner_extent;
        while run_i < end {
            let head = run_i / span;
            let rem = run_i % span;
            let mid = rem / self.inner_extent;
            let inner = rem % self.inner_extent;
            let n = (self.inner_extent - inner).min(end - run_i);
            let from = self.head_base(head) + mid * self.mid_step + inner * self.inner_step;
            let elems = n * self.run;
            copy_runs(
                src,
                &mut dst[at..at + elems],
                from,
                self.inner_step,
                n,
                self.run,
            );
            run_i += n;
            at += elems;
        }
    }

    fn write_tiled(&self, src: &[f32], dst: &mut [f32]) {
        match self.run {
            1 => self.tiled::<1>(src, dst),
            2 => self.tiled::<2>(src, dst),
            4 => self.tiled::<4>(src, dst),
            8 => self.tiled::<8>(src, dst),
            16 => self.tiled::<16>(src, dst),
            32 => self.tiled::<32>(src, dst),
            64 => self.tiled::<64>(src, dst),
            128 => self.tiled::<128>(src, dst),
            _ => self.tiled_dyn(src, dst),
        }
    }

    /// For each tile of the inner axis, visit every `mid` before the next
    /// tile. Those mids share a source slab (`inner_step` apart); writing
    /// them together keeps the slab in cache. Destination runs of one
    /// `(head, mid, tile)` stay contiguous.
    fn tiled<const N: usize>(&self, src: &[f32], dst: &mut [f32]) {
        let tile = tile_len(self.inner_extent, self.inner_step, self.mid_extent, N);
        for head in 0..self.head_count {
            let base = self.head_base(head);
            let mut inner0 = 0;
            while inner0 < self.inner_extent {
                let tn = tile.min(self.inner_extent - inner0);
                for mid in 0..self.mid_extent {
                    let from = base + mid * self.mid_step + inner0 * self.inner_step;
                    let at = ((head * self.mid_extent + mid) * self.inner_extent + inner0) * N;
                    copy_n::<N>(src, &mut dst[at..at + tn * N], from, self.inner_step, tn);
                }
                inner0 += tn;
            }
        }
    }

    fn tiled_dyn(&self, src: &[f32], dst: &mut [f32]) {
        let tile = tile_len(
            self.inner_extent,
            self.inner_step,
            self.mid_extent,
            self.run,
        );
        for head in 0..self.head_count {
            let base = self.head_base(head);
            let mut inner0 = 0;
            while inner0 < self.inner_extent {
                let tn = tile.min(self.inner_extent - inner0);
                for mid in 0..self.mid_extent {
                    let from = base + mid * self.mid_step + inner0 * self.inner_step;
                    let at =
                        ((head * self.mid_extent + mid) * self.inner_extent + inner0) * self.run;
                    copy_dyn(
                        src,
                        &mut dst[at..at + tn * self.run],
                        from,
                        self.inner_step,
                        tn,
                        self.run,
                    );
                }
                inner0 += tn;
            }
        }
    }
}

#[cfg(target_os = "macos")]
fn mmov_refused(detail: &str) -> OjasError {
    OjasError::Backend {
        id: BackendId::Cpu,
        detail: format!("permute: {detail}"),
    }
}

fn tile_len(inner_extent: usize, inner_step: usize, mid_extent: usize, run: usize) -> usize {
    if mid_extent <= 1 || inner_step <= run {
        return inner_extent.max(1);
    }
    let step_bytes = inner_step.saturating_mul(std::mem::size_of::<f32>());
    let tile = (TILE_SPAN_BYTES / step_bytes.max(1)).clamp(8, 128);
    tile.min(inner_extent).max(1)
}

/// Source offsets of `shape` in row-major order, each axis advancing by `steps`.
fn precompute_bases(shape: &[usize], steps: &[usize]) -> Vec<usize> {
    let n: usize = shape.iter().product();
    let mut bases = Vec::with_capacity(n);
    let mut index = vec![0usize; shape.len()];
    let mut from = 0usize;
    for _ in 0..n {
        bases.push(from);
        for axis in (0..shape.len()).rev() {
            index[axis] += 1;
            from += steps[axis];
            if index[axis] < shape[axis] {
                break;
            }
            from -= steps[axis] * shape[axis];
            index[axis] = 0;
        }
    }
    bases
}

fn unravel(mut flat: usize, shape: &[usize], steps: &[usize]) -> usize {
    let mut from = 0usize;
    for axis in (0..shape.len()).rev() {
        let extent = shape[axis];
        let idx = flat % extent;
        flat /= extent;
        from += idx * steps[axis];
    }
    from
}

fn copy_runs(src: &[f32], dst: &mut [f32], src_at: usize, step: usize, count: usize, run: usize) {
    match run {
        1 => copy_n::<1>(src, dst, src_at, step, count),
        2 => copy_n::<2>(src, dst, src_at, step, count),
        4 => copy_n::<4>(src, dst, src_at, step, count),
        8 => copy_n::<8>(src, dst, src_at, step, count),
        16 => copy_n::<16>(src, dst, src_at, step, count),
        32 => copy_n::<32>(src, dst, src_at, step, count),
        64 => copy_n::<64>(src, dst, src_at, step, count),
        128 => copy_n::<128>(src, dst, src_at, step, count),
        _ => copy_dyn(src, dst, src_at, step, count, run),
    }
}

/// `count` runs of `N` floats. `N` is a constant so the assignment is a
/// fixed-width copy, not a `memcpy` of a runtime length.
#[inline(always)]
fn copy_n<const N: usize>(
    src: &[f32],
    dst: &mut [f32],
    mut src_at: usize,
    step: usize,
    count: usize,
) {
    let mut dst_at = 0usize;
    for _ in 0..count {
        let row: &[f32; N] = src[src_at..src_at + N].try_into().unwrap();
        let out: &mut [f32; N] = (&mut dst[dst_at..dst_at + N]).try_into().unwrap();
        *out = *row;
        src_at += step;
        dst_at += N;
    }
}

fn copy_dyn(
    src: &[f32],
    dst: &mut [f32],
    mut src_at: usize,
    step: usize,
    count: usize,
    run: usize,
) {
    let mut dst_at = 0usize;
    for _ in 0..count {
        let mut s = src_at;
        let mut d = dst_at;
        let end = src_at + run;
        while s + 8 <= end {
            let row: &[f32; 8] = src[s..s + 8].try_into().unwrap();
            let out: &mut [f32; 8] = (&mut dst[d..d + 8]).try_into().unwrap();
            *out = *row;
            s += 8;
            d += 8;
        }
        while s < end {
            dst[d] = src[s];
            s += 1;
            d += 1;
        }
        src_at += step;
        dst_at += run;
    }
}

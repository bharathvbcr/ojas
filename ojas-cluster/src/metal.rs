//! Dedicated `"ojas-cluster-metal"` worker bridge over `tessl::emeans` and `OjasBudgetLedger`.

#[cfg(all(target_os = "macos", feature = "metal"))]
mod imp {
    use std::cell::Cell;
    use std::collections::HashMap;
    use std::sync::mpsc;
    use std::sync::{Arc, Mutex, PoisonError, Weak};
    use std::thread::JoinHandle;

    use ojas_core::{BackendId, Budget, BudgetTrimHandler, OjasError, Scratch, Tensor};
    use tessl::emeans::{
        assign_gpu_bf16, assign_gpu_f32, EmeansGpuBackend, EmeansGpuConfig, EmeansGpuPrecision,
        EmeansGpuRunner, EpochBatchPlan, MAX_GPU_CHUNK_ROWS, NONFINITE_ERR_PREFIX,
    };
    use tessl::pool_ledger::OjasBudgetLedger;
    use tessl::GpuRuntime;

    use crate::cpu::{resolve_init_centroids, should_check_revive, MatrixData, MatrixView};
    use crate::{
        epoch_permutation, prescaled_unit_normalize_f32, resolve_recipe, sample_subset_sorted,
        scheduled_alpha, EmeansConfig, EmeansFit, EpochStat, MetalPrecision, ResolvedRecipe,
        UNASSIGNED_LABEL,
    };

    /// Upper bound (`64 MiB`) on cached free-list buffers parked inside the Metal pool.
    const MAX_POOL_CACHE_BYTES: u64 = 64 << 20;

    thread_local! {
        static ON_EMEANS_WORKER: Cell<bool> = const { Cell::new(false) };
    }

    #[inline]
    fn on_emeans_worker() -> bool {
        ON_EMEANS_WORKER.with(Cell::get)
    }

    fn out_of_range(detail: impl Into<String>) -> OjasError {
        OjasError::OutOfRange {
            op: "emeans",
            detail: detail.into(),
        }
    }

    fn shape_err(detail: impl Into<String>) -> OjasError {
        OjasError::Shape {
            op: "emeans",
            detail: detail.into(),
        }
    }

    const fn nonfinite_err() -> OjasError {
        OjasError::NonFinite { op: "emeans" }
    }

    fn metal_backend_err(detail: impl Into<String>) -> OjasError {
        OjasError::Backend {
            id: BackendId::Metal,
            detail: detail.into(),
        }
    }

    fn map_open_err(err: String) -> OjasError {
        if let Some(cap) = OjasBudgetLedger::capacity_error(&err) {
            return cap;
        }
        OjasError::Unsupported {
            op: "emeans",
            detail: err,
        }
    }

    fn map_tessl_err(err: String) -> OjasError {
        if let Some(cap) = OjasBudgetLedger::capacity_error(&err) {
            return cap;
        }
        if err.starts_with(NONFINITE_ERR_PREFIX) {
            return nonfinite_err();
        }
        metal_backend_err(err)
    }

    #[inline]
    fn to_gpu_precision(precision: MetalPrecision) -> EmeansGpuPrecision {
        match precision {
            MetalPrecision::Bf16 => EmeansGpuPrecision::Bf16,
            MetalPrecision::Bf16AssignF32Stats => EmeansGpuPrecision::Bf16AssignF32Stats,
            MetalPrecision::F32 => EmeansGpuPrecision::F32,
        }
    }

    #[inline]
    fn assign_uses_bf16(precision: MetalPrecision) -> bool {
        matches!(
            precision,
            MetalPrecision::Bf16 | MetalPrecision::Bf16AssignF32Stats
        )
    }

    #[inline]
    fn f32_bits_to_bf16_rte(bits: u32) -> u16 {
        let mag = bits & 0x7fff_ffff;
        if mag > 0x7f80_0000 {
            return ((bits >> 16) | 0x0040) as u16;
        }
        if mag < 0x0080_0000 {
            return ((bits >> 16) & 0x8000) as u16;
        }
        ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
    }

    #[inline]
    fn bf16_bits_to_f32_clean(bits: u16) -> f32 {
        let clean = if (bits & 0x7f80) == 0 {
            bits & 0x8000
        } else {
            bits
        };
        f32::from_bits(u32::from(clean) << 16)
    }

    #[inline]
    fn tree_sum_256(mut scratch: [f32; 256]) -> f32 {
        let mut stride = 128;
        while stride > 0 {
            for tid in 0..stride {
                scratch[tid] += scratch[tid + stride];
            }
            stride >>= 1;
        }
        scratch[0]
    }

    fn validate_single_row_for_metal(
        view: &MatrixView<'_>,
        row: usize,
        precision: MetalPrecision,
    ) -> Result<(), OjasError> {
        let d = view.cols();
        let start = row * d;
        let end = start + d;
        let mut sq_lanes = [0.0_f32; 256];
        let bf16_lane = assign_uses_bf16(precision);

        match view.data() {
            MatrixData::F32(src) => {
                for (col, &raw) in src[start..end].iter().enumerate() {
                    if !raw.is_finite() {
                        return Err(nonfinite_err());
                    }
                    let nv = if bf16_lane {
                        let bf = f32_bits_to_bf16_rte(raw.to_bits());
                        if (bf & 0x7fff) >= 0x7f80 {
                            return Err(nonfinite_err());
                        }
                        bf16_bits_to_f32_clean(bf)
                    } else {
                        raw
                    };
                    sq_lanes[col & 255] += nv * nv;
                }
            }
            MatrixData::Bf16(src) => {
                for (col, &raw) in src[start..end].iter().enumerate() {
                    if (raw & 0x7fff) >= 0x7f80 {
                        return Err(nonfinite_err());
                    }
                    let nv = bf16_bits_to_f32_clean(raw);
                    sq_lanes[col & 255] += nv * nv;
                }
            }
            MatrixData::Bf16Bytes(src) => {
                for (col, pair) in src[start..end].iter().enumerate() {
                    let raw = u16::from_ne_bytes(*pair);
                    if (raw & 0x7fff) >= 0x7f80 {
                        return Err(nonfinite_err());
                    }
                    let nv = bf16_bits_to_f32_clean(raw);
                    sq_lanes[col & 255] += nv * nv;
                }
            }
        }

        if !tree_sum_256(sq_lanes).is_finite() {
            return Err(nonfinite_err());
        }
        Ok(())
    }

    /// Pre-validate every active row on the host matching `emeans_stage_rows_*` in `emeans.metal`
    /// (`NaN`/`Inf`, `bf16` range overflow, and 256-lane squared-L2 norm `f32` overflow) so
    /// streaming sessions refuse cleanly before touching GPU runner fault latches.
    pub(crate) fn validate_matrix_for_metal(
        view: &MatrixView<'_>,
        subset: Option<&[u32]>,
        precision: MetalPrecision,
    ) -> Result<(), OjasError> {
        match subset {
            Some(sub) => {
                for &r_u32 in sub {
                    let r = r_u32 as usize;
                    if r >= view.rows() {
                        return Err(out_of_range("subset row index exceeds matrix rows"));
                    }
                    validate_single_row_for_metal(view, r, precision)?;
                }
            }
            None => {
                for r in 0..view.rows() {
                    validate_single_row_for_metal(view, r, precision)?;
                }
            }
        }
        Ok(())
    }

    /// Pre-validate `[k, d]` `f32` centroids matching `emeans_prepare_centroids_*` in `emeans.metal`.
    pub(crate) fn validate_centroids_for_metal(
        centroids: &[f32],
        k: usize,
        d: usize,
        spherical: bool,
        precision: MetalPrecision,
    ) -> Result<(), OjasError> {
        let kd = k
            .checked_mul(d)
            .ok_or_else(|| out_of_range("k * d overflows usize"))?;
        if centroids.len() != kd {
            return Err(shape_err("centroids slice length mismatch"));
        }
        if centroids.iter().any(|v| !v.is_finite()) {
            return Err(nonfinite_err());
        }
        if spherical {
            return Ok(());
        }

        let bf16_lane = assign_uses_bf16(precision);
        for j in 0..k {
            let mut sq_lanes = [0.0_f32; 256];
            for (col, &raw) in centroids[j * d..(j + 1) * d].iter().enumerate() {
                let qv = if bf16_lane {
                    let bf = f32_bits_to_bf16_rte(raw.to_bits());
                    if (bf & 0x7fff) >= 0x7f80 {
                        return Err(nonfinite_err());
                    }
                    bf16_bits_to_f32_clean(bf)
                } else {
                    raw
                };
                sq_lanes[col & 255] += qv * qv;
            }
            if !tree_sum_256(sq_lanes).is_finite() {
                return Err(nonfinite_err());
            }
        }
        Ok(())
    }

    fn check_metal_extents(rows: usize, k: usize, d: usize) -> Result<(), OjasError> {
        if k == 0 {
            return Err(out_of_range("k must be >= 1"));
        }
        if k >= (UNASSIGNED_LABEL as usize) {
            return Err(out_of_range(format!(
                "k ({k}) must be < UNASSIGNED_LABEL ({UNASSIGNED_LABEL})"
            )));
        }
        if d == 0 || d > (i32::MAX as usize) {
            return Err(out_of_range(format!("d ({d}) must be in 1..={}", i32::MAX)));
        }
        if rows > (u32::MAX as usize) {
            return Err(out_of_range(format!(
                "row count ({rows}) exceeds Metal u32 indexing"
            )));
        }
        let kd = k
            .checked_mul(d)
            .ok_or_else(|| out_of_range("k * d overflows usize"))?;
        if kd > (u32::MAX as usize) {
            return Err(out_of_range(format!(
                "k * d ({kd}) exceeds Metal u32 indexing"
            )));
        }
        Ok(())
    }

    enum MatrixPayload {
        ZeroCopyTensor(Tensor),
        F32Scratch(Scratch<f32>),
        Bf16Scratch(Scratch<u16>),
    }

    fn build_full_payload(
        tensor: &Tensor,
        view: &MatrixView<'_>,
        budget: &Budget,
    ) -> Result<MatrixPayload, OjasError> {
        match view.data() {
            MatrixData::F32(_) | MatrixData::Bf16(_) => {
                Ok(MatrixPayload::ZeroCopyTensor(tensor.clone()))
            }
            MatrixData::Bf16Bytes(pairs) => {
                let scratch = Scratch::<u16>::try_extend(pairs.len(), budget, |dst| {
                    for pair in pairs {
                        dst.push(u16::from_ne_bytes(*pair));
                    }
                })?;
                Ok(MatrixPayload::Bf16Scratch(scratch))
            }
        }
    }

    fn build_subset_payload(
        view: &MatrixView<'_>,
        keep: &[u32],
        budget: &Budget,
    ) -> Result<MatrixPayload, OjasError> {
        let d = view.cols();
        let total = keep
            .len()
            .checked_mul(d)
            .ok_or_else(|| out_of_range("sampled matrix elements overflow usize"))?;
        match view.data() {
            MatrixData::F32(src) => {
                let scratch = Scratch::<f32>::try_extend(total, budget, |dst| {
                    for &row_u32 in keep {
                        let r = row_u32 as usize;
                        dst.extend_from_slice(&src[r * d..(r + 1) * d]);
                    }
                })?;
                Ok(MatrixPayload::F32Scratch(scratch))
            }
            MatrixData::Bf16(src) => {
                let scratch = Scratch::<u16>::try_extend(total, budget, |dst| {
                    for &row_u32 in keep {
                        let r = row_u32 as usize;
                        dst.extend_from_slice(&src[r * d..(r + 1) * d]);
                    }
                })?;
                Ok(MatrixPayload::Bf16Scratch(scratch))
            }
            MatrixData::Bf16Bytes(pairs) => {
                let scratch = Scratch::<u16>::try_extend(total, budget, |dst| {
                    for &row_u32 in keep {
                        let r = row_u32 as usize;
                        for pair in &pairs[r * d..(r + 1) * d] {
                            dst.push(u16::from_ne_bytes(*pair));
                        }
                    }
                })?;
                Ok(MatrixPayload::Bf16Scratch(scratch))
            }
        }
    }

    struct FitJob {
        payload: MatrixPayload,
        subset: Option<Scratch<u32>>,
        init_centroids: Scratch<f32>,
        recipe: ResolvedRecipe,
        precision: MetalPrecision,
        capture_assignments: bool,
        budget: Budget,
    }

    struct FitRawOutput {
        closing_centroids: Vec<f32>,
        labels: Vec<u32>,
        cluster_counts: Vec<u64>,
        usage: Vec<f32>,
        epoch_stats: Vec<EpochStat>,
        recipe: ResolvedRecipe,
    }

    struct AssignJob {
        payload: MatrixPayload,
        rows: u32,
        d: u32,
        centroids: Scratch<f32>,
        k: u32,
        max_chunk_rows: u32,
        precision: MetalPrecision,
        spherical: bool,
    }

    struct BeginSessionJob {
        gpu_cfg: EmeansGpuConfig,
        init_centroids: Vec<f32>,
        first_epoch_is_final: bool,
    }

    struct AccumulateSessionJob {
        id: u64,
        payload: MatrixPayload,
        chunk_rows: u32,
        capture_labels: bool,
    }

    type WorkerReply<T> = mpsc::SyncSender<Result<T, OjasError>>;

    enum WorkerCmd {
        TrimPool {
            reply: mpsc::SyncSender<u64>,
        },
        Fit {
            job: Box<FitJob>,
            reply: WorkerReply<FitRawOutput>,
        },
        Assign {
            job: Box<AssignJob>,
            reply: WorkerReply<(Vec<u32>, f64)>,
        },
        BeginSession {
            job: Box<BeginSessionJob>,
            reply: WorkerReply<u64>,
        },
        AccumulateSession {
            job: Box<AccumulateSessionJob>,
            reply: WorkerReply<Option<Vec<u32>>>,
        },
        UpdateSession {
            id: u64,
            alpha: f32,
            step: u64,
            check_revive: bool,
            reply: WorkerReply<usize>,
        },
        EndEpochSession {
            id: u64,
            next_final_epoch: Option<bool>,
            reply: WorkerReply<(f64, Vec<u64>)>,
        },
        CurrentCentroids {
            id: u64,
            reply: WorkerReply<Vec<f32>>,
        },
        FinalizeSession {
            id: u64,
            reply: WorkerReply<(Vec<f32>, Vec<f32>)>,
        },
        DropSession {
            id: u64,
            reply: mpsc::SyncSender<()>,
        },
    }

    fn trim_runtime_all(rt: &GpuRuntime) -> u64 {
        let before = rt.pool_resident().total();
        let _ = rt.synchronize();
        let _ = rt.reclaim(u64::MAX);
        rt.trim_free_list();
        let after = rt.pool_resident().total();
        before.saturating_sub(after)
    }

    fn reclaim_runtime_warm(rt: &GpuRuntime) {
        let _ = rt.synchronize();
        let _ = rt.reclaim(u64::MAX);
    }

    /// Allocate a `Budget`-charged host buffer from `"ojas-cluster-metal"`, draining any
    /// `pending_cold_recycle` or cached `free_list` buffers first if `ON_EMEANS_WORKER`
    /// suppressed `MetalWorkerTrimHandler`.
    fn alloc_on_worker<T>(
        rt: &GpuRuntime,
        mut f: impl FnMut() -> Result<T, OjasError>,
    ) -> Result<T, OjasError> {
        match f() {
            Ok(val) => Ok(val),
            Err(OjasError::CapacityExceeded { .. }) if rt.pool_resident().total() > 0 => {
                let _ = trim_runtime_all(rt);
                f()
            }
            Err(err) => Err(err),
        }
    }

    fn upload_payload_to_dataset(
        runner: &EmeansGpuRunner,
        payload: MatrixPayload,
        rows: u32,
    ) -> Result<tessl::emeans::ResidentDataset, OjasError> {
        match payload {
            MatrixPayload::ZeroCopyTensor(tensor) => {
                let view = MatrixView::open(&tensor, "emeans")?;
                let res = match view.data() {
                    MatrixData::F32(s) => runner.upload_dataset_f32(s, rows),
                    MatrixData::Bf16(s) => runner.upload_dataset_bf16(s, rows),
                    MatrixData::Bf16Bytes(_) => {
                        Err("internal: MappedBf16 must use Bf16Scratch".to_string())
                    }
                };
                drop(view);
                drop(tensor);
                res.map_err(map_tessl_err)
            }
            MatrixPayload::F32Scratch(scratch) => {
                let res = runner.upload_dataset_f32(scratch.as_slice(), rows);
                drop(scratch);
                res.map_err(map_tessl_err)
            }
            MatrixPayload::Bf16Scratch(scratch) => {
                let res = runner.upload_dataset_bf16(scratch.as_slice(), rows);
                drop(scratch);
                res.map_err(map_tessl_err)
            }
        }
    }

    fn run_fit_on_worker(rt: &Arc<GpuRuntime>, job: FitJob) -> Result<FitRawOutput, OjasError> {
        let FitJob {
            payload,
            subset,
            init_centroids,
            recipe,
            precision,
            capture_assignments,
            budget,
        } = job;

        let max_chunk_rows = recipe
            .assignment_chunk_size
            .min(recipe.batch_size)
            .clamp(1, MAX_GPU_CHUNK_ROWS as usize) as u32;

        let gpu_cfg = EmeansGpuConfig {
            k: recipe.k as u32,
            d: recipe.d as u32,
            max_chunk_rows,
            precision: to_gpu_precision(precision),
            backend: EmeansGpuBackend::Auto,
            spherical: recipe.spherical,
            unused_threshold: recipe.unused_threshold,
            split_epsilon: recipe.split_epsilon,
            init_usage: recipe.init_usage,
            seed: recipe.seed,
        };

        let mut runner =
            EmeansGpuRunner::new(rt, gpu_cfg, init_centroids.as_slice()).map_err(map_tessl_err)?;
        drop(init_centroids);

        let ds = upload_payload_to_dataset(&runner, payload, recipe.effective_n as u32)?;
        reclaim_runtime_warm(rt);

        let num_epochs = recipe.epoch_rows.len();
        let mut epoch_stats = Vec::with_capacity(num_epochs);
        let mut final_counts = vec![0_u64; recipe.k];
        let mut closing_labels = Vec::new();
        let mut global_step = 0_usize;

        for (epoch, &rows_this_epoch) in recipe.epoch_rows.iter().enumerate() {
            let final_epoch = epoch + 1 == num_epochs;
            let order = alloc_on_worker(rt, || {
                epoch_permutation(recipe.effective_n, recipe.seed, epoch, &budget)
            })?;

            let mut batches = Vec::with_capacity(rows_this_epoch.div_ceil(recipe.batch_size));
            let mut pos = 0_usize;
            while pos < rows_this_epoch {
                let b_len = (rows_this_epoch - pos).min(recipe.batch_size);
                let alpha = scheduled_alpha(recipe.alpha0, global_step, recipe.total_steps);
                let check_revive = should_check_revive(global_step, recipe.unused_threshold);
                batches.push(EpochBatchPlan {
                    row_offset: pos as u32,
                    row_count: b_len as u32,
                    alpha,
                    step: global_step as u64,
                    check_revive,
                });
                global_step += 1;
                pos += b_len;
            }

            let batch_updates = batches.len();
            let (ep_obj, revived_this_epoch, ep_counts) = if final_epoch && capture_assignments {
                let mut sampled_labels = vec![UNASSIGNED_LABEL; recipe.effective_n];
                let res = runner
                    .run_resident_epoch(
                        &ds,
                        order.as_slice(),
                        &batches,
                        true,
                        Some(&mut sampled_labels),
                    )
                    .map_err(map_tessl_err)?;
                match subset.as_ref() {
                    Some(keep) => {
                        let mut full = vec![UNASSIGNED_LABEL; recipe.global_n];
                        for (eff_idx, &global_row) in keep.as_slice().iter().enumerate() {
                            full[global_row as usize] = sampled_labels[eff_idx];
                        }
                        closing_labels = full;
                    }
                    None => {
                        closing_labels = sampled_labels;
                    }
                }
                res
            } else {
                runner
                    .run_resident_epoch(&ds, order.as_slice(), &batches, final_epoch, None)
                    .map_err(map_tessl_err)?
            };
            drop(order);

            if final_epoch {
                final_counts = ep_counts;
            }
            epoch_stats.push(EpochStat {
                epoch,
                rows_seen: rows_this_epoch,
                batch_updates,
                mean_objective: ep_obj / (rows_this_epoch as f64),
                revived_clusters: revived_this_epoch,
            });
        }

        drop(subset);
        drop(ds);

        let closing_centroids = runner.finalize_closing_centroids().map_err(map_tessl_err)?;
        let usage = runner.read_usage().map_err(map_tessl_err)?;
        drop(runner);

        Ok(FitRawOutput {
            closing_centroids,
            labels: closing_labels,
            cluster_counts: final_counts,
            usage,
            epoch_stats,
            recipe,
        })
    }

    fn run_assign_on_worker(
        rt: &Arc<GpuRuntime>,
        job: AssignJob,
    ) -> Result<(Vec<u32>, f64), OjasError> {
        let AssignJob {
            payload,
            rows,
            d,
            centroids,
            k,
            max_chunk_rows,
            precision,
            spherical,
        } = job;
        let gpu_prec = to_gpu_precision(precision);
        let c_slice = centroids.as_slice();

        let res = match payload {
            MatrixPayload::ZeroCopyTensor(tensor) => {
                let view = MatrixView::open(&tensor, "emeans")?;
                let out = match view.data() {
                    MatrixData::F32(s) => assign_gpu_f32(
                        rt,
                        s,
                        rows,
                        d,
                        c_slice,
                        k,
                        max_chunk_rows,
                        gpu_prec,
                        EmeansGpuBackend::Auto,
                        spherical,
                    ),
                    MatrixData::Bf16(s) => assign_gpu_bf16(
                        rt,
                        s,
                        rows,
                        d,
                        c_slice,
                        k,
                        max_chunk_rows,
                        gpu_prec,
                        EmeansGpuBackend::Auto,
                        spherical,
                    ),
                    MatrixData::Bf16Bytes(_) => {
                        Err("internal: MappedBf16 must use Bf16Scratch".to_string())
                    }
                };
                drop(view);
                drop(tensor);
                out
            }
            MatrixPayload::F32Scratch(scratch) => {
                let out = assign_gpu_f32(
                    rt,
                    scratch.as_slice(),
                    rows,
                    d,
                    c_slice,
                    k,
                    max_chunk_rows,
                    gpu_prec,
                    EmeansGpuBackend::Auto,
                    spherical,
                );
                drop(scratch);
                out
            }
            MatrixPayload::Bf16Scratch(scratch) => {
                let out = assign_gpu_bf16(
                    rt,
                    scratch.as_slice(),
                    rows,
                    d,
                    c_slice,
                    k,
                    max_chunk_rows,
                    gpu_prec,
                    EmeansGpuBackend::Auto,
                    spherical,
                );
                drop(scratch);
                out
            }
        };
        drop(centroids);
        res.map_err(map_tessl_err)
    }

    fn run_accumulate_on_worker(
        runner: &mut EmeansGpuRunner,
        payload: MatrixPayload,
        chunk_rows: u32,
        capture_labels: bool,
    ) -> Result<Option<Vec<u32>>, OjasError> {
        let mut labels_buf = if capture_labels {
            Some(vec![0_u32; chunk_rows as usize])
        } else {
            None
        };
        let res = match payload {
            MatrixPayload::ZeroCopyTensor(tensor) => {
                let view = MatrixView::open(&tensor, "emeans")?;
                let out = match view.data() {
                    MatrixData::F32(s) => {
                        runner.accumulate_f32_chunk(s, chunk_rows, labels_buf.as_deref_mut())
                    }
                    MatrixData::Bf16(s) => {
                        runner.accumulate_bf16_chunk(s, chunk_rows, labels_buf.as_deref_mut())
                    }
                    MatrixData::Bf16Bytes(_) => {
                        Err("internal: MappedBf16 must use Bf16Scratch".to_string())
                    }
                };
                drop(view);
                drop(tensor);
                out
            }
            MatrixPayload::F32Scratch(scratch) => {
                let out = runner.accumulate_f32_chunk(
                    scratch.as_slice(),
                    chunk_rows,
                    labels_buf.as_deref_mut(),
                );
                drop(scratch);
                out
            }
            MatrixPayload::Bf16Scratch(scratch) => {
                let out = runner.accumulate_bf16_chunk(
                    scratch.as_slice(),
                    chunk_rows,
                    labels_buf.as_deref_mut(),
                );
                drop(scratch);
                out
            }
        };
        res.map_err(map_tessl_err)?;
        Ok(labels_buf)
    }

    fn worker_main(
        budget: Budget,
        init_tx: mpsc::SyncSender<Result<(), OjasError>>,
        cmd_rx: mpsc::Receiver<WorkerCmd>,
    ) {
        ON_EMEANS_WORKER.with(|f| f.set(true));

        let rt = match GpuRuntime::new_inference() {
            Ok(rt) => rt,
            Err(err) => {
                let _ = init_tx.send(Err(map_open_err(err)));
                return;
            }
        };

        let cache_cap_u64 = MAX_POOL_CACHE_BYTES.min(budget.cap_bytes() / 4);
        rt.set_pool_cache_cap_bytes(usize::try_from(cache_cap_u64).unwrap_or(usize::MAX));
        let ledger = OjasBudgetLedger::new(&budget, cache_cap_u64);
        if let Err(err) = rt.set_pool_ledger(Some(ledger.as_pool_ledger())) {
            let _ = init_tx.send(Err(metal_backend_err(err)));
            return;
        }

        if init_tx.send(Ok(())).is_err() {
            let _ = trim_runtime_all(&rt);
            return;
        }

        let mut sessions: HashMap<u64, EmeansGpuRunner> = HashMap::new();
        let mut next_session_id: u64 = 1;

        while let Ok(cmd) = cmd_rx.recv() {
            match cmd {
                WorkerCmd::TrimPool { reply } => {
                    let freed = trim_runtime_all(&rt);
                    let _ = reply.send(freed);
                }
                WorkerCmd::Fit { job, reply } => {
                    let out = run_fit_on_worker(&rt, *job);
                    if out.is_ok() {
                        reclaim_runtime_warm(&rt);
                    } else {
                        let _ = trim_runtime_all(&rt);
                    }
                    let _ = reply.send(out);
                }
                WorkerCmd::Assign { job, reply } => {
                    let out = run_assign_on_worker(&rt, *job);
                    if out.is_ok() {
                        reclaim_runtime_warm(&rt);
                    } else {
                        let _ = trim_runtime_all(&rt);
                    }
                    let _ = reply.send(out);
                }
                WorkerCmd::BeginSession { job, reply } => {
                    let BeginSessionJob {
                        gpu_cfg,
                        init_centroids,
                        first_epoch_is_final,
                    } = *job;
                    let out = EmeansGpuRunner::new(&rt, gpu_cfg, &init_centroids)
                        .and_then(|mut runner| {
                            runner.reset_pass_accumulators(first_epoch_is_final)?;
                            Ok(runner)
                        })
                        .map_err(map_tessl_err)
                        .map(|runner| {
                            let id = next_session_id;
                            next_session_id = next_session_id.wrapping_add(1);
                            sessions.insert(id, runner);
                            id
                        });
                    if out.is_err() {
                        let _ = trim_runtime_all(&rt);
                    } else {
                        reclaim_runtime_warm(&rt);
                    }
                    let _ = reply.send(out);
                }
                WorkerCmd::AccumulateSession { job, reply } => {
                    let AccumulateSessionJob {
                        id,
                        payload,
                        chunk_rows,
                        capture_labels,
                    } = *job;
                    let out = match sessions.get_mut(&id) {
                        Some(runner) => {
                            run_accumulate_on_worker(runner, payload, chunk_rows, capture_labels)
                        }
                        None => Err(metal_backend_err(format!("unknown Metal session id {id}"))),
                    };
                    let _ = reply.send(out);
                }
                WorkerCmd::UpdateSession {
                    id,
                    alpha,
                    step,
                    check_revive,
                    reply,
                } => {
                    let out = match sessions.get_mut(&id) {
                        Some(runner) => runner
                            .apply_batch_update(alpha, step, check_revive)
                            .map_err(map_tessl_err),
                        None => Err(metal_backend_err(format!("unknown Metal session id {id}"))),
                    };
                    let _ = reply.send(out);
                }
                WorkerCmd::EndEpochSession {
                    id,
                    next_final_epoch,
                    reply,
                } => {
                    let out = match sessions.get_mut(&id) {
                        Some(runner) => runner
                            .finish_streaming_epoch()
                            .and_then(|res| {
                                if let Some(is_final) = next_final_epoch {
                                    runner.reset_pass_accumulators(is_final)?;
                                }
                                Ok(res)
                            })
                            .map_err(map_tessl_err),
                        None => Err(metal_backend_err(format!("unknown Metal session id {id}"))),
                    };
                    let _ = reply.send(out);
                }
                WorkerCmd::CurrentCentroids { id, reply } => {
                    let out = match sessions.get(&id) {
                        Some(runner) => runner.read_centroids().map_err(map_tessl_err),
                        None => Err(metal_backend_err(format!("unknown Metal session id {id}"))),
                    };
                    let _ = reply.send(out);
                }
                WorkerCmd::FinalizeSession { id, reply } => {
                    let out = match sessions.remove(&id) {
                        Some(mut runner) => {
                            let res = runner
                                .finalize_closing_centroids()
                                .and_then(|cents| {
                                    let usage = runner.read_usage()?;
                                    Ok((cents, usage))
                                })
                                .map_err(map_tessl_err);
                            drop(runner);
                            let _ = trim_runtime_all(&rt);
                            res
                        }
                        None => Err(metal_backend_err(format!("unknown Metal session id {id}"))),
                    };
                    let _ = reply.send(out);
                }
                WorkerCmd::DropSession { id, reply } => {
                    let removed = sessions.remove(&id);
                    drop(removed);
                    let _ = trim_runtime_all(&rt);
                    let _ = reply.send(());
                }
            }
        }

        sessions.clear();
        let _ = trim_runtime_all(&rt);
        drop(ledger);
        drop(rt);
    }

    struct MetalWorkerInner {
        tx: Mutex<Option<mpsc::Sender<WorkerCmd>>>,
        join: Mutex<Option<JoinHandle<()>>>,
    }

    impl std::fmt::Debug for MetalWorkerInner {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("MetalWorkerInner").finish_non_exhaustive()
        }
    }

    impl MetalWorkerInner {
        fn send_sync<R>(
            &self,
            build: impl FnOnce(mpsc::SyncSender<R>) -> WorkerCmd,
        ) -> Result<R, OjasError> {
            if on_emeans_worker() {
                return Err(metal_backend_err(
                    "ojas-cluster-metal worker re-entry refused",
                ));
            }
            let sender = {
                let guard = self.tx.lock().unwrap_or_else(PoisonError::into_inner);
                guard
                    .as_ref()
                    .cloned()
                    .ok_or_else(|| metal_backend_err("ojas-cluster-metal worker has shut down"))?
            };
            let (reply_tx, reply_rx) = mpsc::sync_channel(1);
            sender
                .send(build(reply_tx))
                .map_err(|_| metal_backend_err("ojas-cluster-metal worker exited unexpectedly"))?;
            reply_rx
                .recv()
                .map_err(|_| metal_backend_err("ojas-cluster-metal worker dropped reply channel"))
        }

        fn trim_pool_bytes(&self) -> u64 {
            if on_emeans_worker() {
                return 0;
            }
            self.send_sync(|reply| WorkerCmd::TrimPool { reply })
                .unwrap_or(0)
        }
    }

    impl Drop for MetalWorkerInner {
        fn drop(&mut self) {
            let tx = self
                .tx
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .take();
            drop(tx);
            if !on_emeans_worker() {
                if let Some(handle) = self
                    .join
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .take()
                {
                    let _ = handle.join();
                }
            }
        }
    }

    #[derive(Debug)]
    struct MetalWorkerTrimHandler {
        worker: Weak<MetalWorkerInner>,
    }

    impl BudgetTrimHandler for MetalWorkerTrimHandler {
        fn trim_memory(&self, _needed_bytes: u64) -> u64 {
            if on_emeans_worker() {
                return 0;
            }
            let Some(worker) = self.worker.upgrade() else {
                return 0;
            };
            worker.trim_pool_bytes()
        }

        fn pool_key(&self) -> Option<usize> {
            Some(Weak::as_ptr(&self.worker) as usize)
        }

        fn is_live(&self) -> bool {
            self.worker.strong_count() > 0
        }
    }

    #[derive(Debug, Clone)]
    pub(crate) struct MetalEngineHandle {
        worker: Arc<MetalWorkerInner>,
        precision: MetalPrecision,
    }

    impl MetalEngineHandle {
        pub(crate) fn new(precision: MetalPrecision, budget: &Budget) -> Result<Self, OjasError> {
            let (cmd_tx, cmd_rx) = mpsc::channel();
            let (init_tx, init_rx) = mpsc::sync_channel(1);
            let worker_budget = budget.clone();

            let handle = std::thread::Builder::new()
                .name("ojas-cluster-metal".to_string())
                .spawn(move || worker_main(worker_budget, init_tx, cmd_rx))
                .map_err(|err| {
                    metal_backend_err(format!("spawning ojas-cluster-metal failed: {err}"))
                })?;

            match init_rx.recv() {
                Ok(Ok(())) => {}
                Ok(Err(err)) => {
                    drop(cmd_tx);
                    let _ = handle.join();
                    return Err(err);
                }
                Err(_) => {
                    drop(cmd_tx);
                    let _ = handle.join();
                    return Err(metal_backend_err(
                        "ojas-cluster-metal exited during initialization",
                    ));
                }
            }

            let inner = Arc::new(MetalWorkerInner {
                tx: Mutex::new(Some(cmd_tx)),
                join: Mutex::new(Some(handle)),
            });
            budget.add_trim_handler(Arc::new(MetalWorkerTrimHandler {
                worker: Arc::downgrade(&inner),
            }));

            Ok(Self {
                worker: inner,
                precision,
            })
        }

        pub(crate) fn trim_pool(&self) -> usize {
            usize::try_from(self.worker.trim_pool_bytes()).unwrap_or(usize::MAX)
        }

        pub(crate) fn fit(
            &self,
            x: &Tensor,
            init_centroids: Option<&Tensor>,
            cfg: &EmeansConfig,
            budget: &Budget,
        ) -> Result<EmeansFit, OjasError> {
            let x_view = MatrixView::open(x, "emeans")?;
            let global_n = x_view.rows();
            let d = x_view.cols();
            let recipe = resolve_recipe(global_n, d, cfg)?;
            check_metal_extents(recipe.effective_n, recipe.k, recipe.d)?;

            let subset = if recipe.effective_n < global_n {
                Some(sample_subset_sorted(
                    global_n,
                    recipe.effective_n,
                    recipe.seed,
                    budget,
                )?)
            } else {
                None
            };
            let subset_slice = subset.as_ref().map(Scratch::as_slice);

            validate_matrix_for_metal(&x_view, subset_slice, self.precision)?;

            let init_c = resolve_init_centroids(
                &x_view,
                subset_slice,
                init_centroids,
                recipe.k,
                false,
                recipe.seed,
                budget,
            )?;
            validate_centroids_for_metal(
                init_c.as_slice(),
                recipe.k,
                recipe.d,
                recipe.spherical,
                self.precision,
            )?;

            let payload = match subset_slice {
                Some(keep) => build_subset_payload(&x_view, keep, budget)?,
                None => build_full_payload(x, &x_view, budget)?,
            };
            // Release any `SharedBuf` `HostLease` on `x` before dispatching to the worker.
            drop(x_view);

            let job = Box::new(FitJob {
                payload,
                subset,
                init_centroids: init_c,
                recipe,
                precision: self.precision,
                capture_assignments: cfg.capture_assignments,
                budget: budget.clone(),
            });

            let raw = self
                .worker
                .send_sync(|reply| WorkerCmd::Fit { job, reply })??;

            // Allocate output `Tensor` on the caller thread (`ON_EMEANS_WORKER == false`) so
            // `MetalWorkerTrimHandler` can evict cached GPU buffers if `budget` is tight.
            let centroids = Tensor::from_f32(
                &raw.closing_centroids,
                &[raw.recipe.k, raw.recipe.d],
                budget,
            )?;

            Ok(EmeansFit {
                centroids,
                labels: raw.labels,
                cluster_counts: raw.cluster_counts,
                usage: raw.usage,
                epoch_stats: raw.epoch_stats,
                recipe: raw.recipe,
            })
        }

        pub(crate) fn assign(
            &self,
            x: &Tensor,
            centroids: &Tensor,
            spherical: bool,
            chunk_size: usize,
            budget: &Budget,
        ) -> Result<(Vec<u32>, f64), OjasError> {
            if chunk_size == 0 {
                return Err(out_of_range("chunk_size must be >= 1"));
            }
            let x_view = MatrixView::open(x, "emeans")?;
            let c_view = MatrixView::open(centroids, "emeans")?;
            let n = x_view.rows();
            let d = x_view.cols();
            let k = c_view.rows();
            if k == 0 {
                return Err(out_of_range("k must be >= 1"));
            }
            if k >= (UNASSIGNED_LABEL as usize) {
                return Err(out_of_range(format!(
                    "k ({k}) must be < UNASSIGNED_LABEL ({UNASSIGNED_LABEL})"
                )));
            }
            if d == 0 {
                return Err(out_of_range("d must be >= 1"));
            }
            if c_view.cols() != d {
                return Err(shape_err(format!(
                    "centroids cols {} != x cols {d}",
                    c_view.cols()
                )));
            }
            check_metal_extents(n, k, d)?;

            let kd = k
                .checked_mul(d)
                .ok_or_else(|| out_of_range("k * d overflows usize"))?;
            let mut c_f32 = Scratch::<f32>::try_alloc(kd, budget)?;
            c_view.copy_window_f32_checked(0, k, c_f32.as_mut_slice())?;
            drop(c_view);

            validate_centroids_for_metal(c_f32.as_slice(), k, d, spherical, self.precision)?;
            validate_matrix_for_metal(&x_view, None, self.precision)?;

            let payload = build_full_payload(x, &x_view, budget)?;
            drop(x_view);

            let max_chunk_rows = chunk_size.clamp(1, MAX_GPU_CHUNK_ROWS as usize) as u32;
            let job = Box::new(AssignJob {
                payload,
                rows: n as u32,
                d: d as u32,
                centroids: c_f32,
                k: k as u32,
                max_chunk_rows,
                precision: self.precision,
                spherical,
            });

            self.worker
                .send_sync(|reply| WorkerCmd::Assign { job, reply })?
        }
    }

    #[derive(Debug)]
    pub(crate) struct MetalSessionHandle {
        worker: Arc<MetalWorkerInner>,
        id: u64,
        precision: MetalPrecision,
        closed: bool,
    }

    impl MetalSessionHandle {
        pub(crate) fn begin(
            engine: &MetalEngineHandle,
            recipe: &ResolvedRecipe,
            init_centroids_f32: &[f32],
        ) -> Result<Self, OjasError> {
            check_metal_extents(recipe.effective_n, recipe.k, recipe.d)?;
            validate_centroids_for_metal(
                init_centroids_f32,
                recipe.k,
                recipe.d,
                recipe.spherical,
                engine.precision,
            )?;

            let max_chunk_rows = recipe
                .assignment_chunk_size
                .min(recipe.batch_size)
                .clamp(1, MAX_GPU_CHUNK_ROWS as usize) as u32;

            let gpu_cfg = EmeansGpuConfig {
                k: recipe.k as u32,
                d: recipe.d as u32,
                max_chunk_rows,
                precision: to_gpu_precision(engine.precision),
                backend: EmeansGpuBackend::Auto,
                spherical: recipe.spherical,
                unused_threshold: recipe.unused_threshold,
                split_epsilon: recipe.split_epsilon,
                init_usage: recipe.init_usage,
                seed: recipe.seed,
            };

            let job = Box::new(BeginSessionJob {
                gpu_cfg,
                init_centroids: init_centroids_f32.to_vec(),
                first_epoch_is_final: recipe.epoch_rows.len() == 1,
            });

            let id = engine
                .worker
                .send_sync(|reply| WorkerCmd::BeginSession { job, reply })??;

            Ok(Self {
                worker: Arc::clone(&engine.worker),
                id,
                precision: engine.precision,
                closed: false,
            })
        }

        pub(crate) fn accumulate(
            &mut self,
            chunk: &Tensor,
            chunk_view: MatrixView<'_>,
            capture_labels: bool,
            budget: &Budget,
        ) -> Result<Option<Vec<u32>>, OjasError> {
            let chunk_rows = chunk_view.rows();
            if chunk_rows > (u32::MAX as usize) {
                return Err(out_of_range("chunk rows exceed Metal u32 indexing"));
            }

            // Validate finiteness and `bf16` / `row_norm2` bounds on host BEFORE touching
            // `EmeansGpuRunner`, keeping `runner.latched_fault == 0` so live sessions recover cleanly.
            validate_matrix_for_metal(&chunk_view, None, self.precision)?;
            let payload = build_full_payload(chunk, &chunk_view, budget)?;
            drop(chunk_view);

            let job = Box::new(AccumulateSessionJob {
                id: self.id,
                payload,
                chunk_rows: chunk_rows as u32,
                capture_labels,
            });

            self.worker
                .send_sync(|reply| WorkerCmd::AccumulateSession { job, reply })?
        }

        pub(crate) fn update(
            &mut self,
            alpha: f32,
            step: u64,
            check_revive: bool,
        ) -> Result<usize, OjasError> {
            let id = self.id;
            self.worker.send_sync(|reply| WorkerCmd::UpdateSession {
                id,
                alpha,
                step,
                check_revive,
                reply,
            })?
        }

        pub(crate) fn end_epoch(
            &mut self,
            next_final_epoch: Option<bool>,
        ) -> Result<(f64, Vec<u64>), OjasError> {
            let id = self.id;
            self.worker.send_sync(|reply| WorkerCmd::EndEpochSession {
                id,
                next_final_epoch,
                reply,
            })?
        }

        pub(crate) fn current_centroids(
            &self,
            k: usize,
            d: usize,
            spherical: bool,
        ) -> Result<Vec<f32>, OjasError> {
            let id = self.id;
            let raw = self
                .worker
                .send_sync(|reply| WorkerCmd::CurrentCentroids { id, reply })??;
            if !spherical {
                return Ok(raw);
            }
            let mut unit = vec![0.0_f32; k * d];
            for j in 0..k {
                prescaled_unit_normalize_f32(
                    &raw[j * d..(j + 1) * d],
                    &mut unit[j * d..(j + 1) * d],
                );
            }
            Ok(unit)
        }

        pub(crate) fn finalize(mut self) -> Result<(Vec<f32>, Vec<f32>), OjasError> {
            self.closed = true;
            let id = self.id;
            self.worker
                .send_sync(|reply| WorkerCmd::FinalizeSession { id, reply })?
        }
    }

    impl Drop for MetalSessionHandle {
        fn drop(&mut self) {
            if !self.closed {
                self.closed = true;
                let id = self.id;
                if !on_emeans_worker() {
                    let _ = self
                        .worker
                        .send_sync(|reply| WorkerCmd::DropSession { id, reply });
                }
            }
        }
    }
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
mod imp {
    use ojas_core::{Budget, OjasError, Tensor};

    use crate::cpu::MatrixView;
    use crate::{EmeansConfig, EmeansFit, MetalPrecision, ResolvedRecipe};

    #[derive(Debug, Clone)]
    pub(crate) enum MetalEngineHandle {}

    impl MetalEngineHandle {
        pub(crate) fn new(_precision: MetalPrecision, _budget: &Budget) -> Result<Self, OjasError> {
            Err(OjasError::Unsupported {
                op: "emeans",
                detail:
                    "ClusterDevice::Metal requires macOS with the `metal` Cargo feature enabled"
                        .to_string(),
            })
        }

        pub(crate) fn trim_pool(&self) -> usize {
            match *self {}
        }

        pub(crate) fn fit(
            &self,
            _x: &Tensor,
            _init_centroids: Option<&Tensor>,
            _cfg: &EmeansConfig,
            _budget: &Budget,
        ) -> Result<EmeansFit, OjasError> {
            match *self {}
        }

        pub(crate) fn assign(
            &self,
            _x: &Tensor,
            _centroids: &Tensor,
            _spherical: bool,
            _chunk_size: usize,
            _budget: &Budget,
        ) -> Result<(Vec<u32>, f64), OjasError> {
            match *self {}
        }
    }

    #[derive(Debug)]
    pub(crate) enum MetalSessionHandle {}

    impl MetalSessionHandle {
        pub(crate) fn begin(
            engine: &MetalEngineHandle,
            _recipe: &ResolvedRecipe,
            _init_centroids_f32: &[f32],
        ) -> Result<Self, OjasError> {
            match *engine {}
        }

        pub(crate) fn accumulate(
            &mut self,
            _chunk: &Tensor,
            _chunk_view: MatrixView<'_>,
            _capture_labels: bool,
            _budget: &Budget,
        ) -> Result<Option<Vec<u32>>, OjasError> {
            match *self {}
        }

        pub(crate) fn update(
            &mut self,
            _alpha: f32,
            _step: u64,
            _check_revive: bool,
        ) -> Result<usize, OjasError> {
            match *self {}
        }

        pub(crate) fn end_epoch(
            &mut self,
            _next_final_epoch: Option<bool>,
        ) -> Result<(f64, Vec<u64>), OjasError> {
            match *self {}
        }

        pub(crate) fn current_centroids(
            &self,
            _k: usize,
            _d: usize,
            _spherical: bool,
        ) -> Result<Vec<f32>, OjasError> {
            match *self {}
        }

        pub(crate) fn finalize(self) -> Result<(Vec<f32>, Vec<f32>), OjasError> {
            match self {}
        }
    }
}

pub(crate) use imp::{MetalEngineHandle, MetalSessionHandle};

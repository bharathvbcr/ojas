//! Persistent [`EmeansEngine`], streaming [`EmeansSession`], and convenience entry points.

use std::sync::Arc;

use ojas_core::{Budget, OjasError, Scratch, Tensor};

use crate::cpu::{
    assign_cpu_tensor, decode_bf16_row_checked, fit_cpu, should_check_revive, CpuPool, CpuSession,
    MatrixData, MatrixView,
};
use crate::metal::{MetalEngineHandle, MetalSessionHandle};
use crate::{
    resolve_recipe, scheduled_alpha, AccumulateStep, ClusterDevice, CpuMath, EmeansConfig,
    EmeansFit, EpochStat, ResolvedRecipe, UNASSIGNED_LABEL,
};

/// Default row window size (`16,384` rows) used by [`emeans_assign`].
const DEFAULT_ASSIGN_CHUNK_ROWS: usize = 16_384;

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

#[derive(Debug)]
enum EngineBackend {
    Cpu { pool: Arc<CpuPool>, math: CpuMath },
    Metal(MetalEngineHandle),
}

/// Reusable `e-means` compute engine keeping CPU worker threads and/or the `"ojas-cluster-metal"`
/// GPU runtime and buffer free-list warm across calls.
#[derive(Debug)]
pub struct EmeansEngine {
    device: ClusterDevice,
    budget: Budget,
    backend: EngineBackend,
}

impl EmeansEngine {
    /// Open a persistent clustering engine on `device`, charging runtime/scratch allocations to `budget`.
    ///
    /// Never falls back silently across substrates: requesting [`ClusterDevice::Metal`] when
    /// Metal 4 is unavailable returns [`OjasError::Unsupported`].
    pub fn new(device: ClusterDevice, budget: &Budget) -> Result<Self, OjasError> {
        let backend = match device {
            ClusterDevice::Cpu { threads, math } => {
                let pool = Arc::new(CpuPool::new(threads)?);
                EngineBackend::Cpu { pool, math }
            }
            ClusterDevice::Metal { precision } => {
                let handle = MetalEngineHandle::new(precision, budget)?;
                EngineBackend::Metal(handle)
            }
        };
        Ok(Self {
            device,
            budget: budget.clone(),
            backend,
        })
    }

    /// Target compute substrate bound to this engine.
    #[inline]
    pub fn device(&self) -> ClusterDevice {
        self.device
    }

    /// Memory [`Budget`] associated with this engine.
    #[inline]
    pub fn budget(&self) -> &Budget {
        &self.budget
    }

    /// Synchronously evict cached GPU free-list buffers, returning the number of bytes freed
    /// (`0` on [`ClusterDevice::Cpu`]).
    pub fn trim_pool(&self) -> usize {
        match &self.backend {
            EngineBackend::Cpu { .. } => 0,
            EngineBackend::Metal(metal) => metal.trim_pool(),
        }
    }

    /// Fit `k` centroids on 2D `F32`/`Bf16` (`Host` or `Mapped`) tensor `x` using deterministic
    /// `SplitMix64` centroid initialization.
    pub fn fit(&self, x: &Tensor, cfg: &EmeansConfig) -> Result<EmeansFit, OjasError> {
        if cfg.k >= (UNASSIGNED_LABEL as usize) {
            return Err(out_of_range(format!(
                "k ({}) must be < UNASSIGNED_LABEL ({UNASSIGNED_LABEL})",
                cfg.k
            )));
        }
        match &self.backend {
            EngineBackend::Cpu { pool, math } => fit_cpu(pool, x, None, cfg, *math, &self.budget),
            EngineBackend::Metal(metal) => metal.fit(x, None, cfg, &self.budget),
        }
    }

    /// Fit `k` centroids on `x` starting from explicit `[k, d]` `init_centroids`.
    pub fn fit_with_init(
        &self,
        x: &Tensor,
        init_centroids: &Tensor,
        cfg: &EmeansConfig,
    ) -> Result<EmeansFit, OjasError> {
        if cfg.k >= (UNASSIGNED_LABEL as usize) {
            return Err(out_of_range(format!(
                "k ({}) must be < UNASSIGNED_LABEL ({UNASSIGNED_LABEL})",
                cfg.k
            )));
        }
        match &self.backend {
            EngineBackend::Cpu { pool, math } => {
                fit_cpu(pool, x, Some(init_centroids), cfg, *math, &self.budget)
            }
            EngineBackend::Metal(metal) => metal.fit(x, Some(init_centroids), cfg, &self.budget),
        }
    }

    /// Assign rows of `x` `[n, d]` to nearest `centroids` `[k, d]` in windows of `chunk_size` rows,
    /// returning `(labels, total_objective)`.
    pub fn assign(
        &self,
        x: &Tensor,
        centroids: &Tensor,
        spherical: bool,
        chunk_size: usize,
    ) -> Result<(Vec<u32>, f64), OjasError> {
        match &self.backend {
            EngineBackend::Cpu { pool, math } => assign_cpu_tensor(
                pool,
                x,
                centroids,
                chunk_size,
                *math,
                spherical,
                &self.budget,
            ),
            EngineBackend::Metal(metal) => {
                metal.assign(x, centroids, spherical, chunk_size, &self.budget)
            }
        }
    }
}

enum SessionBackend {
    Cpu {
        pool: Arc<CpuPool>,
        math: CpuMath,
        session: Box<CpuSession>,
    },
    Metal(MetalSessionHandle),
}

impl std::fmt::Debug for SessionBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu { pool, math, .. } => f
                .debug_struct("SessionBackend::Cpu")
                .field("pool", pool)
                .field("math", math)
                .finish_non_exhaustive(),
            Self::Metal(metal) => f.debug_tuple("SessionBackend::Metal").field(metal).finish(),
        }
    }
}

/// Out-of-core / streaming `e-means` training session over arbitrary caller-supplied row chunks.
///
/// Driving [`EmeansSession`] over the same epoch permutations and batch boundaries as
/// [`EmeansEngine::fit_with_init`] produces a 0-ULP bit-identical [`EmeansFit`] on both CPU
/// (`CpuMath::DeterministicFma` / `Exact`) and Metal (`Bf16`, `Bf16AssignF32Stats`, `F32`).
#[derive(Debug)]
pub struct EmeansSession {
    budget: Budget,
    recipe: ResolvedRecipe,
    capture_assignments: bool,
    epoch: usize,
    step: usize,
    epoch_rows_updated: usize,
    batch_rows_accumulated: usize,
    epoch_batch_updates: usize,
    epoch_revived: usize,
    epoch_stats: Vec<EpochStat>,
    final_epoch_counts: Vec<u64>,
    closing_labels: Vec<u32>,
    backend: SessionBackend,
}

impl EmeansSession {
    /// Begin a streaming `e-means` session for an effective dataset of `global_n` rows and
    /// explicit `[k, d]` `init_centroids`.
    pub fn begin(
        engine: &EmeansEngine,
        global_n: usize,
        init_centroids: &Tensor,
        cfg: &EmeansConfig,
    ) -> Result<Self, OjasError> {
        let init_view = MatrixView::open(init_centroids, "emeans")?;
        let d = init_view.cols();
        let recipe = resolve_recipe(global_n, d, cfg)?;
        if recipe.k >= (UNASSIGNED_LABEL as usize) {
            return Err(out_of_range(format!(
                "k ({}) must be < UNASSIGNED_LABEL ({UNASSIGNED_LABEL})",
                recipe.k
            )));
        }
        if init_view.rows() != recipe.k {
            return Err(shape_err(format!(
                "init_centroids rows ({}) != k ({})",
                init_view.rows(),
                recipe.k
            )));
        }

        let kd = recipe
            .k
            .checked_mul(recipe.d)
            .ok_or_else(|| out_of_range("k * d overflows usize"))?;
        let mut init_c = Scratch::<f32>::try_alloc(kd, &engine.budget)?;
        init_view.copy_window_f32_checked(0, recipe.k, init_c.as_mut_slice())?;
        drop(init_view);

        let backend = match &engine.backend {
            EngineBackend::Cpu { pool, math } => {
                let mut cpu_sess = CpuSession::new(
                    recipe.k,
                    recipe.d,
                    recipe.spherical,
                    recipe.unused_threshold,
                    recipe.split_epsilon,
                    recipe.init_usage,
                    recipe.seed,
                    init_c.as_slice(),
                    &engine.budget,
                )?;
                drop(init_c);
                cpu_sess.reset_pass_accumulators(recipe.epoch_rows.len() == 1);
                SessionBackend::Cpu {
                    pool: Arc::clone(pool),
                    math: *math,
                    session: Box::new(cpu_sess),
                }
            }
            EngineBackend::Metal(metal) => {
                let handle = MetalSessionHandle::begin(metal, &recipe, init_c.as_slice())?;
                drop(init_c);
                SessionBackend::Metal(handle)
            }
        };

        let num_epochs = recipe.epoch_rows.len();
        let final_epoch_counts = vec![0_u64; recipe.k];

        Ok(Self {
            budget: engine.budget.clone(),
            recipe,
            capture_assignments: cfg.capture_assignments,
            epoch: 0,
            step: 0,
            epoch_rows_updated: 0,
            batch_rows_accumulated: 0,
            epoch_batch_updates: 0,
            epoch_revived: 0,
            epoch_stats: Vec::with_capacity(num_epochs),
            final_epoch_counts,
            closing_labels: Vec::new(),
            backend,
        })
    }

    /// Resolved `canonical_v1` training schedule for this session.
    #[inline]
    pub fn recipe(&self) -> &ResolvedRecipe {
        &self.recipe
    }

    /// Zero-based index of the active epoch (`== recipe.epoch_rows.len()` once all epochs finish).
    #[inline]
    pub fn epoch(&self) -> usize {
        self.epoch
    }

    /// Number of completed mini-batch [`Self::update`] steps across all epochs so far.
    #[inline]
    pub fn step(&self) -> usize {
        self.step
    }

    #[inline]
    fn current_batch_target(&self) -> usize {
        if self.epoch >= self.recipe.epoch_rows.len() {
            return 0;
        }
        let epoch_total = self.recipe.epoch_rows[self.epoch];
        epoch_total
            .saturating_sub(self.epoch_rows_updated)
            .min(self.recipe.batch_size)
    }

    /// Rows still needed to complete the current mini-batch before calling [`Self::update`].
    #[inline]
    pub fn remaining_in_batch(&self) -> usize {
        self.current_batch_target()
            .saturating_sub(self.batch_rows_accumulated)
    }

    /// Unaccumulated rows still remaining in the current epoch.
    #[inline]
    pub fn remaining_in_epoch(&self) -> usize {
        if self.epoch >= self.recipe.epoch_rows.len() {
            return 0;
        }
        let epoch_total = self.recipe.epoch_rows[self.epoch];
        epoch_total.saturating_sub(self.epoch_rows_updated + self.batch_rows_accumulated)
    }

    /// Stream a 2D `[m, d]` `F32`/`Bf16` (`Host` or `Mapped`) row chunk (`1 <= m <= remaining_in_batch()`)
    /// into the active mini-batch.
    ///
    /// Returns `(progress, Some(chunk_labels))` during the final closing epoch when
    /// `capture_assignments == true`, and `(progress, None)` otherwise. Any shape, dtype, placement,
    /// range, non-finite, or capacity error refuses cleanly before mutating session progress.
    pub fn accumulate(
        &mut self,
        chunk: &Tensor,
    ) -> Result<(AccumulateStep, Option<Vec<u32>>), OjasError> {
        let chunk_view = MatrixView::open(chunk, "emeans")?;
        if chunk_view.cols() != self.recipe.d {
            return Err(shape_err(format!(
                "chunk cols ({}) != session d ({})",
                chunk_view.cols(),
                self.recipe.d
            )));
        }
        let chunk_rows = chunk_view.rows();
        if chunk_rows == 0 {
            return Err(out_of_range("chunk must contain >= 1 row"));
        }
        let rem_batch = self.remaining_in_batch();
        if rem_batch == 0 {
            return Err(out_of_range(
                "current batch or epoch cannot accept more rows before update()/end_epoch()",
            ));
        }
        if chunk_rows > rem_batch {
            return Err(out_of_range(format!(
                "chunk rows ({chunk_rows}) exceed remaining_in_batch ({rem_batch})"
            )));
        }

        let is_final_epoch = self.epoch + 1 == self.recipe.epoch_rows.len();
        let capture_labels = is_final_epoch && self.capture_assignments;
        let d = self.recipe.d;
        let sub_cap = self.recipe.assignment_chunk_size.max(1);

        let maybe_labels = match &mut self.backend {
            SessionBackend::Cpu {
                pool,
                math,
                session,
            } => {
                chunk_view.validate_rows_finite(None)?;
                let mut out_labels = if capture_labels {
                    Some(vec![0_u32; chunk_rows])
                } else {
                    None
                };
                match chunk_view.data() {
                    MatrixData::F32(rows) => {
                        let mut r0 = 0_usize;
                        while r0 < chunk_rows {
                            let take = (chunk_rows - r0).min(sub_cap);
                            let sub_lbl = out_labels.as_mut().map(|v| &mut v[r0..r0 + take]);
                            session.accumulate_f32_chunk(
                                pool,
                                &rows[r0 * d..(r0 + take) * d],
                                take,
                                *math,
                                sub_lbl,
                                &self.budget,
                            )?;
                            r0 += take;
                        }
                    }
                    MatrixData::Bf16(rows_bf16) => {
                        let win_rows = chunk_rows.min(sub_cap);
                        let win_elems = win_rows
                            .checked_mul(d)
                            .ok_or_else(|| out_of_range("bf16 staging window overflows usize"))?;
                        let mut win = Scratch::<f32>::try_alloc(win_elems, &self.budget)?;
                        let mut r0 = 0_usize;
                        while r0 < chunk_rows {
                            let take = (chunk_rows - r0).min(sub_cap);
                            let win_slice = &mut win.as_mut_slice()[..take * d];
                            decode_bf16_row_checked(
                                &rows_bf16[r0 * d..(r0 + take) * d],
                                win_slice,
                            )?;
                            let sub_lbl = out_labels.as_mut().map(|v| &mut v[r0..r0 + take]);
                            session.accumulate_f32_chunk(
                                pool,
                                win_slice,
                                take,
                                *math,
                                sub_lbl,
                                &self.budget,
                            )?;
                            r0 += take;
                        }
                    }
                    MatrixData::Bf16Bytes(_) => {
                        let win_rows = chunk_rows.min(sub_cap);
                        let win_elems = win_rows
                            .checked_mul(d)
                            .ok_or_else(|| out_of_range("mapped bf16 window overflows usize"))?;
                        let mut win = Scratch::<f32>::try_alloc(win_elems, &self.budget)?;
                        let mut r0 = 0_usize;
                        while r0 < chunk_rows {
                            let take = (chunk_rows - r0).min(sub_cap);
                            chunk_view.copy_window_f32_checked(
                                r0,
                                take,
                                &mut win.as_mut_slice()[..take * d],
                            )?;
                            let sub_lbl = out_labels.as_mut().map(|v| &mut v[r0..r0 + take]);
                            session.accumulate_f32_chunk(
                                pool,
                                &win.as_slice()[..take * d],
                                take,
                                *math,
                                sub_lbl,
                                &self.budget,
                            )?;
                            r0 += take;
                        }
                    }
                }
                drop(chunk_view);
                out_labels
            }
            SessionBackend::Metal(metal) => {
                metal.accumulate(chunk, chunk_view, capture_labels, &self.budget)?
            }
        };

        if let Some(ref lbls) = maybe_labels {
            self.closing_labels.extend_from_slice(lbls);
        }
        self.batch_rows_accumulated += chunk_rows;

        let rem_batch_after = self.remaining_in_batch();
        let rem_epoch_after = self.remaining_in_epoch();
        Ok((
            AccumulateStep {
                rows_accepted: chunk_rows,
                remaining_in_batch: rem_batch_after,
                remaining_in_epoch: rem_epoch_after,
                batch_ready: rem_batch_after == 0,
            },
            maybe_labels,
        ))
    }

    /// Apply the scheduled cosine-decayed EMA centroid update and optional `SplitMix64` Gumbel
    /// dead-cluster split revival once [`Self::remaining_in_batch`] reaches `0`.
    pub fn update(&mut self) -> Result<usize, OjasError> {
        let target = self.current_batch_target();
        if target == 0 {
            return Err(out_of_range(
                "update() called when no active mini-batch remains in this epoch",
            ));
        }
        if self.batch_rows_accumulated != target {
            return Err(out_of_range(format!(
                "update() requires full batch ({target} rows), only {} accumulated",
                self.batch_rows_accumulated
            )));
        }

        let alpha = scheduled_alpha(self.recipe.alpha0, self.step, self.recipe.total_steps);
        let check_revive = should_check_revive(self.step, self.recipe.unused_threshold);
        let step_u64 = self.step as u64;

        let revived = match &mut self.backend {
            SessionBackend::Cpu { session, .. } => {
                session.apply_batch_update(alpha, step_u64, check_revive)?
            }
            SessionBackend::Metal(metal) => metal.update(alpha, step_u64, check_revive)?,
        };

        self.epoch_rows_updated += target;
        self.batch_rows_accumulated = 0;
        self.step += 1;
        self.epoch_batch_updates += 1;
        self.epoch_revived += revived;
        Ok(revived)
    }

    /// Close the completed epoch once all epoch rows have been accumulated and updated, returning
    /// its [`EpochStat`] and preparing accumulators for the next epoch (if any).
    pub fn end_epoch(&mut self) -> Result<EpochStat, OjasError> {
        let num_epochs = self.recipe.epoch_rows.len();
        if self.epoch >= num_epochs {
            return Err(out_of_range("all epochs have already ended"));
        }
        if self.batch_rows_accumulated != 0 {
            return Err(out_of_range(format!(
                "end_epoch() called with {} un-applied batch rows; call update() first",
                self.batch_rows_accumulated
            )));
        }
        let epoch_total = self.recipe.epoch_rows[self.epoch];
        if self.epoch_rows_updated != epoch_total {
            return Err(out_of_range(format!(
                "end_epoch() requires {epoch_total} committed rows, got {}",
                self.epoch_rows_updated
            )));
        }

        let was_final = self.epoch + 1 == num_epochs;
        let next_final_epoch = if was_final {
            None
        } else {
            Some(self.epoch + 2 == num_epochs)
        };

        let (ep_obj, ep_counts) = match &mut self.backend {
            SessionBackend::Cpu { session, .. } => {
                let res = session.finish_epoch();
                if let Some(is_final) = next_final_epoch {
                    session.reset_pass_accumulators(is_final);
                }
                res
            }
            SessionBackend::Metal(metal) => metal.end_epoch(next_final_epoch)?,
        };

        if was_final {
            self.final_epoch_counts = ep_counts;
        }

        let stat = EpochStat {
            epoch: self.epoch,
            rows_seen: epoch_total,
            batch_updates: self.epoch_batch_updates,
            mean_objective: ep_obj / (epoch_total as f64),
            revived_clusters: self.epoch_revived,
        };
        self.epoch_stats.push(stat.clone());
        self.epoch += 1;
        self.epoch_rows_updated = 0;
        self.batch_rows_accumulated = 0;
        self.epoch_batch_updates = 0;
        self.epoch_revived = 0;

        Ok(stat)
    }

    /// Read a host `[k, d]` `F32` snapshot of the current effective centroids (unit-normalized
    /// when `spherical == true`) without mutating session state.
    pub fn current_centroids(&mut self) -> Result<Tensor, OjasError> {
        let k = self.recipe.k;
        let d = self.recipe.d;
        match &self.backend {
            SessionBackend::Cpu { session, .. } => {
                Tensor::from_f32(session.effective_centroids(), &[k, d], &self.budget)
            }
            SessionBackend::Metal(metal) => {
                let cents = metal.current_centroids(k, d, self.recipe.spherical)?;
                Tensor::from_f32(&cents, &[k, d], &self.budget)
            }
        }
    }

    /// Finalize exact `f64` closing centroids after all scheduled epochs have completed via
    /// [`Self::end_epoch`], releasing all session workspaces and returning [`EmeansFit`].
    pub fn finalize(self) -> Result<EmeansFit, OjasError> {
        let num_epochs = self.recipe.epoch_rows.len();
        if self.epoch != num_epochs {
            return Err(out_of_range(format!(
                "finalize() requires all {num_epochs} epochs completed, currently at epoch {}",
                self.epoch
            )));
        }

        let Self {
            budget,
            recipe,
            epoch_stats,
            final_epoch_counts,
            closing_labels,
            backend,
            ..
        } = self;

        let (closing_vec, usage) = match backend {
            SessionBackend::Cpu { mut session, .. } => {
                let cents = session.finalize_closing_centroids()?;
                let usage = session.read_usage().to_vec();
                drop(session);
                (cents, usage)
            }
            SessionBackend::Metal(metal) => metal.finalize()?,
        };

        let centroids = Tensor::from_f32(&closing_vec, &[recipe.k, recipe.d], &budget)?;

        Ok(EmeansFit {
            centroids,
            labels: closing_labels,
            cluster_counts: final_epoch_counts,
            usage,
            epoch_stats,
            recipe,
        })
    }
}

/// One-shot convenience wrapper around [`EmeansEngine::new`] and [`EmeansEngine::fit`].
pub fn emeans_fit(
    x: &Tensor,
    cfg: &EmeansConfig,
    device: ClusterDevice,
    budget: &Budget,
) -> Result<EmeansFit, OjasError> {
    let engine = EmeansEngine::new(device, budget)?;
    engine.fit(x, cfg)
}

/// One-shot convenience wrapper around [`EmeansEngine::new`] and [`EmeansEngine::assign`]
/// (`chunk_size = 16_384`).
pub fn emeans_assign(
    x: &Tensor,
    centroids: &Tensor,
    spherical: bool,
    device: ClusterDevice,
    budget: &Budget,
) -> Result<(Vec<u32>, f64), OjasError> {
    let engine = EmeansEngine::new(device, budget)?;
    engine.assign(x, centroids, spherical, DEFAULT_ASSIGN_CHUNK_ROWS)
}

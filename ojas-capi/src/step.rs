//! One training step on the caller's payload.
//!
//! Mode [`MODE_LOGITS`] is mean cross-entropy of the supplied logits.
//! Mode [`MODE_TOKENS`] is a linear map of the u16 tokens (`weight[c] = c + 1`)
//! followed by that same cross-entropy. [`step_checked`] runs on the session's
//! backend: [`ojas_cpu::CpuBackend`] for a CPU session, the session's
//! `MetalBackend` for a Metal one, its [`ojas_wgpu::WgpuBackend`] for a wgpu
//! one. On a device, inputs are uploaded once, every intermediate stays where
//! the backend computes, and the loss is the one tensor read back;
//! `clip_grad_norm` reads its norm through the backend's own status read,
//! which is not a tensor readback. The gradient norm is `clip_grad_norm` of the
//! cross-entropy backward, and `lr` is the learning rate AdamW applied to a
//! one-element parameter whose gradient is the loss. Nothing is written into
//! the session.
//!
//! Every step borrows a child of one process budget ([`STEP_PROCESS_BUDGET_BYTES`]).
//! A call does not receive its own 1 GiB budget. A Metal or wgpu backend
//! charges a child of the same process budget.
//!
//! A CPU step runs under a [`ResourcePolicy`]. [`step`] and [`step_checked`]
//! use [`ResourcePolicy::new`], so `allow_split` is false. A caller that wants
//! row splits passes [`step_with_policy`] and sets `policy.allow_split`. The
//! split count comes from `policy.caller_budget_bytes` and the request shape.
//! It does not read the host probe.

use std::sync::LazyLock;

use ojas_core::{shape_product, AdamWConfig, Backend, Budget, DType, Tensor, CLIP_GRAD_NORM_EPS};
use ojas_cpu::CpuBackend;
use ojas_device::{Device, ResourcePolicy};

use crate::session::Compute;

pub const MODE_HEADER: u32 = 0;
pub const MODE_LOGITS: u32 = 1;
pub const MODE_TOKENS: u32 = 2;

/// Token-mode targets are u16, so no class past this index can be a target.
pub const MAX_TOKEN_CLASSES: u32 = 1 << 16;

/// Byte cap shared by every step in the process. A step that cannot reserve
/// against this cap fails; the cap is not raised to fit the request.
pub const STEP_PROCESS_BUDGET_BYTES: u64 = 1 << 30;

/// Child capped by the caller budget and by [`STEP_PROCESS_BUDGET_BYTES`].
///
/// The smaller of the two is the allocation cap. The split count uses the
/// caller budget alone, before this `min`.
fn budget_for(caller_budget_bytes: u64) -> Budget {
    static PROCESS: LazyLock<Budget> = LazyLock::new(|| Budget::new(STEP_PROCESS_BUDGET_BYTES));
    PROCESS.child(caller_budget_bytes.min(STEP_PROCESS_BUDGET_BYTES))
}

/// A full-size child of the process budget, for a step or a device backend.
pub(crate) fn step_budget() -> Budget {
    budget_for(STEP_PROCESS_BUDGET_BYTES)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepStats {
    pub loss: f32,
    pub grad_norm: f32,
    pub lr: f32,
    /// `Some(k)` when the step ran as `k` row-parts (`k >= 2`).
    /// `None` when the request was not split.
    pub split: Option<u32>,
}

pub struct StepInput<'a> {
    pub mode: u32,
    pub batch: u32,
    pub seq: u32,
    pub n_classes: u32,
    pub step: u32,
    pub lr: f32,
    pub logits: &'a [f32],
    pub tokens: &'a [u16],
    pub targets_u32: &'a [u32],
    pub targets_u16: &'a [u16],
}

/// A step on a single-threaded [`CpuBackend`].
pub fn step(input: StepInput<'_>) -> Result<StepStats, String> {
    step_checked(&Compute::Cpu { threads: 1 }, input, || Ok(()))
}

/// One step on `compute`. A CPU step does not split rows; a request that
/// does not fit the process budget is `ojas:E_CAPACITY:`.
pub fn step_checked(
    compute: &Compute,
    input: StepInput<'_>,
    mut check: impl FnMut() -> Result<(), String>,
) -> Result<StepStats, String> {
    let rows = validate(&input, &mut check)?;
    match compute {
        Compute::Cpu { threads } => {
            let policy = ResourcePolicy::new(STEP_PROCESS_BUDGET_BYTES);
            cpu_step(*threads, rows, &input, &policy, &mut check)
        }
        #[cfg(target_os = "macos")]
        Compute::Metal(metal) => run(metal, &step_budget(), rows, &input, &mut check),
        Compute::Wgpu(wgpu) => run(wgpu.as_ref(), &step_budget(), rows, &input, &mut check),
    }
}

/// One step on a single-threaded [`CpuBackend`] under `policy`.
///
/// `allow_split` defaults to false on [`ResourcePolicy::new`]. When it is
/// false, a request that does not fit the child budget is
/// `ojas:E_CAPACITY:`, the same prefix as an ordinary capacity miss.
/// When it is true, `k` is the smallest part count whose every row-part fits
/// in `policy.caller_budget_bytes` by [`part_limit_bytes`]. That count is not
/// taken from [`ojas_device::ResourcePlan`] or the host probe.
///
/// Only [`Device::Cpu`] computes here. A list that names Metal, Vulkan (wgpu),
/// CUDA, or HIP is refused. The step does not run on the CPU in that case.
/// A Metal or wgpu session steps on its device through [`step_checked`].
pub fn step_with_policy(
    input: StepInput<'_>,
    policy: &ResourcePolicy,
    mut check: impl FnMut() -> Result<(), String>,
) -> Result<StepStats, String> {
    let rows = validate(&input, &mut check)?;
    refuse_unimplemented_devices(&policy.devices)?;
    cpu_step(1, rows, &input, policy, &mut check)
}

/// Checks shared by every entry point. Returns the row count.
fn validate(
    input: &StepInput<'_>,
    check: &mut impl FnMut() -> Result<(), String>,
) -> Result<usize, String> {
    check()?;
    if input.mode == MODE_HEADER {
        return Err("step: shape: step payload is missing logits".to_string());
    }
    let rows = rows(input.batch, input.seq)?;
    check()?;
    if !input.lr.is_finite() || input.lr < 0.0 {
        return Err("step: out of range: lr must be finite and >= 0".to_string());
    }
    if input.n_classes == 0 {
        return Err("step: shape: n_classes is 0".to_string());
    }
    check_payload(input, rows)?;
    Ok(rows)
}

fn cpu_step(
    threads: usize,
    rows: usize,
    input: &StepInput<'_>,
    policy: &ResourcePolicy,
    check: &mut impl FnMut() -> Result<(), String>,
) -> Result<StepStats, String> {
    let n_classes = input.n_classes as usize;
    let cap = budget_for(policy.caller_budget_bytes);
    let cpu = if threads <= 1 {
        CpuBackend::new(cap.clone())
    } else {
        CpuBackend::with_threads(cap.clone(), threads).map_err(show)?
    };
    let k = if policy.allow_split {
        choose_k(rows, n_classes, policy.caller_budget_bytes)?
    } else {
        1
    };
    if k == 1 {
        let (loss, grad_norm) = run_part(&cpu, input, 0, rows, true, check)?.clipped()?;
        return finish(&cpu, &cap, input, loss, grad_norm, None, check);
    }
    let parts = row_parts(rows, k)?;
    gusset::ffi::log_event(&format!("Adaptation split k={k} rows={rows}"));
    let mut weighted_loss = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut offset = 0usize;
    for (index, &part_rows) in parts.iter().enumerate() {
        if index > 0 {
            check()?;
        }
        let (loss_i, grad) = run_part(&cpu, input, offset, part_rows, false, check)?.values()?;
        let rows_i = part_rows as f64;
        weighted_loss += rows_i * f64::from(loss_i);
        let scale = rows_i / (rows as f64);
        for value in grad {
            let scaled = f64::from(value) * scale;
            sum_sq += scaled * scaled;
        }
        offset += part_rows;
    }
    let loss = (weighted_loss / (rows as f64)) as f32;
    let grad_norm = clip_once(sum_sq)?;
    finish(
        &cpu,
        &cap,
        input,
        loss,
        grad_norm,
        Some(u32::try_from(k).map_err(|_| "step: shape: split k exceeds u32".to_string())?),
        check,
    )
}

/// The whole step on `backend`, unsplit. Host tensors are charged to `budget`
/// and dropped as soon as they are uploaded; `backend` charges its own budget
/// for what it holds. The payload was checked by [`check_payload`].
fn run<B: Backend>(
    backend: &B,
    budget: &Budget,
    rows: usize,
    input: &StepInput<'_>,
    check: &mut impl FnMut() -> Result<(), String>,
) -> Result<StepStats, String> {
    let n_classes = input.n_classes as usize;
    let up = |t: Tensor| backend.upload(&t).map_err(show);
    let logits = match input.mode {
        MODE_LOGITS => {
            up(Tensor::from_f32(input.logits, &[rows, n_classes], budget).map_err(show)?)?
        }
        MODE_TOKENS => {
            // The linear output is the only charge of rows * n_classes f32s.
            // A reservation held across `linear_forward` is a second copy.
            let x: Vec<f32> = input.tokens.iter().map(|&t| f32::from(t)).collect();
            let weight: Vec<f32> = (0..n_classes).map(|c| (c as f32) + 1.0).collect();
            let x = up(Tensor::from_f32(&x, &[rows, 1], budget).map_err(show)?)?;
            let weight = up(Tensor::from_f32(&weight, &[n_classes, 1], budget).map_err(show)?)?;
            backend.linear_forward(&x, &weight).map_err(show)?
        }
        other => return Err(format!("step: shape: unknown mode {other}")),
    };
    let targets: Vec<u32> = if input.mode == MODE_LOGITS {
        input.targets_u32.to_vec()
    } else {
        input.targets_u16.iter().map(|&t| u32::from(t)).collect()
    };
    check()?;
    let targets = up(Tensor::from_u32(&targets, &[rows], budget).map_err(show)?)?;
    let loss_t = backend
        .cross_entropy_mean_forward(&logits, &targets, None)
        .map_err(show)?;
    check()?;
    let mut grad = backend
        .cross_entropy_mean_backward(&logits, &targets, None)
        .map_err(show)?;
    check()?;
    let grad_norm = backend
        .clip_grad_norm(std::slice::from_mut(&mut grad), 1.0)
        .map_err(show)?;
    // The learning rate is applied to a one-element parameter whose gradient
    // is the scalar loss, viewed in place, so a non-finite lr fails inside
    // the backend.
    let grad_scalar = loss_t.narrow(0, &[1], &[1]).map_err(show)?;
    let mut param = up(Tensor::from_f32(&[0.0], &[1], budget).map_err(show)?)?;
    let mut moment1 = up(Tensor::zeros(&[1], DType::F32, budget).map_err(show)?)?;
    let mut moment2 = up(Tensor::zeros(&[1], DType::F32, budget).map_err(show)?)?;
    let config = AdamWConfig::nanolab(f64::from(input.lr), 0.0);
    check()?;
    backend
        .adamw_step(
            &mut param,
            &grad_scalar,
            &mut moment1,
            &mut moment2,
            u64::from(input.step),
            config,
        )
        .map_err(show)?;
    let loss = scalar(&backend.download(&loss_t).map_err(show)?)?;
    if !loss.is_finite() || !grad_norm.is_finite() {
        return Err(non_finite());
    }
    Ok(StepStats {
        loss,
        grad_norm,
        lr: input.lr,
        split: None,
    })
}

/// Row counts for `k` parts. The first `rows % k` parts get one extra row,
/// so a remainder makes the parts unequal. `1000` rows and `k = 3` is
/// `334 + 333 + 333`.
pub(crate) fn row_parts(rows: usize, k: usize) -> Result<Vec<usize>, String> {
    if k == 0 || rows == 0 || k > rows {
        return Err("step: shape: split k is outside 1..=rows".to_string());
    }
    let base = rows / k;
    let rem = rows % k;
    Ok((0..k)
        .map(|i| if i < rem { base + 1 } else { base })
        .collect())
}

/// Upper bound on the bytes one row-part charges: `3 Y + 2 S + 256`, with
/// `Y = rows * n_classes * 4` and `S = (rows + n_classes) * 4`.
///
/// `ojas-cpu` charges a host copy of every input it reads
/// (`ojas-cpu/src/validate.rs`), so the peaks are:
/// - cross-entropy forward or backward: the logits tensor `Y`, its copy
///   `Y`, the cross-entropy scratch `Y + 4 n_classes`, the targets tensor
///   and its copy `2 * 4 rows`, and the 4-byte loss: `3Y + 2*4*rows +
///   4*n_classes + 4`;
/// - token-mode linear: `x` and the class weight plus their copies
///   (`2 S`), the output `Y`, and the GEMM scratch, which is at most
///   `Y + 4 (rows + n_classes) + 80` bytes and so fits in the second `Y`
///   and the slack.
///
/// This bound is a function of `rows` and `n_classes` only.
pub(crate) fn part_limit_bytes(rows: usize, n_classes: usize) -> Result<u64, String> {
    let y = u64::try_from(shape_product(&[rows, n_classes]).map_err(show)?)
        .map_err(|_| "step: shape: rows * n_classes overflows".to_string())?;
    let y_bytes = y
        .checked_mul(4)
        .ok_or_else(|| "step: shape: rows * n_classes overflows".to_string())?;
    let side = (rows as u64)
        .checked_add(n_classes as u64)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| "step: shape: rows * n_classes overflows".to_string())?;
    y_bytes
        .checked_mul(3)
        .and_then(|n| side.checked_mul(2).and_then(|s| n.checked_add(s)))
        .and_then(|n| n.checked_add(256))
        .ok_or_else(|| "step: shape: rows * n_classes overflows".to_string())
}

fn choose_k(rows: usize, n_classes: usize, caller_budget_bytes: u64) -> Result<usize, String> {
    if part_limit_bytes(rows, n_classes)? <= caller_budget_bytes {
        return Ok(1);
    }
    let mut lo = 2usize;
    let mut hi = rows;
    let mut best: Option<usize> = None;
    while lo <= hi {
        let mid = lo + (hi - lo) / 2;
        if parts_fit(rows, n_classes, mid, caller_budget_bytes)? {
            best = Some(mid);
            if mid == 2 {
                break;
            }
            hi = mid - 1;
        } else if mid == rows {
            break;
        } else {
            lo = mid + 1;
        }
    }
    best.ok_or_else(|| {
        show(ojas_core::OjasError::CapacityExceeded {
            requested: part_limit_bytes(1, n_classes).unwrap_or(0),
            cap: caller_budget_bytes,
            live: 0,
        })
    })
}

fn parts_fit(
    rows: usize,
    n_classes: usize,
    k: usize,
    caller_budget_bytes: u64,
) -> Result<bool, String> {
    let parts = row_parts(rows, k)?;
    for part_rows in parts {
        if part_limit_bytes(part_rows, n_classes)? > caller_budget_bytes {
            return Ok(false);
        }
    }
    Ok(true)
}

struct PartOut {
    loss: f32,
    clipped_norm: Option<f32>,
    grad: Vec<f32>,
}

impl PartOut {
    fn clipped(self) -> Result<(f32, f32), String> {
        let norm = self
            .clipped_norm
            .ok_or_else(|| "step: shape: missing clipped gradient norm".to_string())?;
        Ok((self.loss, norm))
    }

    fn values(self) -> Result<(f32, Vec<f32>), String> {
        if self.clipped_norm.is_some() {
            return Err("step: shape: split part was clipped early".to_string());
        }
        Ok((self.loss, self.grad))
    }
}

fn run_part(
    cpu: &CpuBackend,
    input: &StepInput<'_>,
    offset: usize,
    part_rows: usize,
    clip: bool,
    check: &mut impl FnMut() -> Result<(), String>,
) -> Result<PartOut, String> {
    let n_classes = input.n_classes as usize;
    let budget = cpu.budget();
    let logits = match input.mode {
        MODE_LOGITS => {
            let start = offset
                .checked_mul(n_classes)
                .ok_or_else(|| "step: shape: rows * n_classes overflows".to_string())?;
            let width = part_rows
                .checked_mul(n_classes)
                .ok_or_else(|| "step: shape: rows * n_classes overflows".to_string())?;
            let end = start
                .checked_add(width)
                .ok_or_else(|| "step: shape: rows * n_classes overflows".to_string())?;
            Tensor::from_f32(&input.logits[start..end], &[part_rows, n_classes], budget)
                .map_err(show)?
        }
        MODE_TOKENS => {
            let end = offset + part_rows;
            // The linear's output tensor is the only charge of rows * n_classes
            // f32s. A reservation held across `linear_forward` would be a second
            // copy, and a request that fits one copy would be refused.
            let x_values: Vec<f32> = input.tokens[offset..end]
                .iter()
                .map(|&t| f32::from(t))
                .collect();
            let weight_values: Vec<f32> = (0..n_classes).map(|c| (c as f32) + 1.0).collect();
            let x = Tensor::from_f32(&x_values, &[part_rows, 1], budget).map_err(show)?;
            let weight = Tensor::from_f32(&weight_values, &[n_classes, 1], budget).map_err(show)?;
            let logits = cpu.linear_forward(&x, &weight).map_err(show)?;
            drop(weight);
            drop(x);
            logits
        }
        other => return Err(format!("step: shape: unknown mode {other}")),
    };
    let targets: Vec<u32> = if input.mode == MODE_LOGITS {
        input.targets_u32[offset..offset + part_rows].to_vec()
    } else {
        input.targets_u16[offset..offset + part_rows]
            .iter()
            .map(|&t| u32::from(t))
            .collect()
    };
    check()?;
    let targets = Tensor::from_u32(&targets, &[part_rows], budget).map_err(show)?;
    let loss_t = cpu
        .cross_entropy_mean_forward(&logits, &targets, None)
        .map_err(show)?;
    check()?;
    let mut grad = cpu
        .cross_entropy_mean_backward(&logits, &targets, None)
        .map_err(show)?;
    check()?;
    let loss = scalar(&loss_t)?;
    drop(loss_t);
    drop(logits);
    drop(targets);
    if clip {
        let grad_norm = cpu
            .clip_grad_norm(std::slice::from_mut(&mut grad), 1.0)
            .map_err(show)?;
        drop(grad);
        Ok(PartOut {
            loss,
            clipped_norm: Some(grad_norm),
            grad: Vec::new(),
        })
    } else {
        let grad = grad.to_f32_vec().map_err(show)?;
        Ok(PartOut {
            loss,
            clipped_norm: None,
            grad,
        })
    }
}

fn finish(
    cpu: &CpuBackend,
    budget: &Budget,
    input: &StepInput<'_>,
    loss: f32,
    grad_norm: f32,
    split: Option<u32>,
    check: &mut impl FnMut() -> Result<(), String>,
) -> Result<StepStats, String> {
    let mut param = Tensor::from_f32(&[0.0], &[1], budget).map_err(show)?;
    let grad_scalar = Tensor::from_f32(&[loss], &[1], budget).map_err(show)?;
    let mut moment1 = Tensor::zeros(&[1], DType::F32, budget).map_err(show)?;
    let mut moment2 = Tensor::zeros(&[1], DType::F32, budget).map_err(show)?;
    let config = AdamWConfig::nanolab(f64::from(input.lr), 0.0);
    check()?;
    cpu.adamw_step(
        &mut param,
        &grad_scalar,
        &mut moment1,
        &mut moment2,
        u64::from(input.step),
        config,
    )
    .map_err(show)?;
    if !loss.is_finite() || !grad_norm.is_finite() {
        return Err(non_finite());
    }
    Ok(StepStats {
        loss,
        grad_norm,
        lr: input.lr,
        split,
    })
}

/// One clip of the combined norm. The value returned is the norm before the
/// scale, matching [`CpuBackend::clip_grad_norm`].
fn clip_once(sum_sq: f64) -> Result<f32, String> {
    let norm = (sum_sq.sqrt()) as f32;
    if !norm.is_finite() {
        return Err(non_finite());
    }
    let coef = 1.0 / (norm + CLIP_GRAD_NORM_EPS);
    if !coef.is_finite() {
        return Err(non_finite());
    }
    Ok(norm)
}

fn refuse_unimplemented_devices(devices: &[Device]) -> Result<(), String> {
    if devices.is_empty() {
        return Err(
            "step: device: policy device list is empty; only Cpu computes a step".to_string(),
        );
    }
    let foreign: Vec<&Device> = devices
        .iter()
        .filter(|device| **device != Device::Cpu)
        .collect();
    if foreign.is_empty() {
        return Ok(());
    }
    let names = foreign
        .iter()
        .map(|device| format!("{device:?}"))
        .collect::<Vec<_>>()
        .join(", ");
    Err(format!(
        "step: device: {names} cannot compute a step through step_with_policy; refusing CPU fallback"
    ))
}

fn check_payload(input: &StepInput<'_>, total_rows: usize) -> Result<(), String> {
    let n_classes = input.n_classes as usize;
    match input.mode {
        MODE_LOGITS => {
            let n_logits = total_rows
                .checked_mul(n_classes)
                .ok_or_else(|| "step: shape: rows * n_classes overflows".to_string())?;
            if input.logits.len() != n_logits {
                return Err(format!(
                    "step: shape: logits len {} != {total_rows}*{n_classes}",
                    input.logits.len()
                ));
            }
            if input.targets_u32.len() != total_rows {
                return Err(format!(
                    "step: shape: targets len {} != {total_rows}",
                    input.targets_u32.len()
                ));
            }
        }
        MODE_TOKENS => {
            if input.n_classes > MAX_TOKEN_CLASSES {
                return Err(format!(
                    "step: shape: n_classes {} exceeds {MAX_TOKEN_CLASSES} for u16 targets",
                    input.n_classes
                ));
            }
            total_rows
                .checked_mul(n_classes)
                .ok_or_else(|| "step: shape: rows * n_classes overflows".to_string())?;
            if input.tokens.len() != total_rows || input.targets_u16.len() != total_rows {
                return Err(format!(
                    "step: shape: token buffer len {}/{} != {total_rows}",
                    input.tokens.len(),
                    input.targets_u16.len()
                ));
            }
        }
        other => return Err(format!("step: shape: unknown mode {other}")),
    }
    Ok(())
}

pub(crate) fn rows(batch: u32, seq: u32) -> Result<usize, String> {
    if batch == 0 || seq == 0 {
        return Err("step: shape: batch and seq must be non-zero".to_string());
    }
    usize::try_from(batch)
        .ok()
        .and_then(|b| usize::try_from(seq).ok().and_then(|s| b.checked_mul(s)))
        .ok_or_else(|| "step: shape: batch * seq overflows".to_string())
}

fn scalar(tensor: &Tensor) -> Result<f32, String> {
    let values = tensor.to_f32_vec().map_err(show)?;
    if values.len() != 1 {
        return Err(format!(
            "step: shape: loss rank has {} values",
            values.len()
        ));
    }
    Ok(values[0])
}

fn show(err: ojas_core::OjasError) -> String {
    crate::ojas_error("step", &err)
}

fn non_finite() -> String {
    crate::kinded(crate::ErrorKind::NonFinite, "step: non-finite value")
}

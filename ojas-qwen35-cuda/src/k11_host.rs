//! K11 on the host: AdamW in place over a parameter bank, and the gradient's
//! squared norm. These are the scalar formation, the table, and a bit-exact
//! f32 emulation of the device kernels in [`crate::k11_kernels`].
//!
//! # AdamW (`cuda-backend-scoping.md` §3, the K11 row)
//!
//! This is torch's single-tensor AdamW (`torch/optim/adam.py:416-545`, with
//! `decoupled_weight_decay`, amsgrad and maximize off), per bank entry, with a
//! **per-entry weight decay and learning-rate scale** and a **per-entry step
//! count**. In F the base lr is the schedule's, and Lappi's `apply_lr` writes
//! `lr * lr_scale` into each group (`python/qd_train/optim.py:224-241`).
//! torch counts steps per parameter, and a parameter with no gradient is
//! skipped entirely: no decay, no moments, no count. That is
//! `f-optimizer-spec.md` row D7: the span head skips letter-only batches. So
//! [`AdamwBank::step_f32`] takes an `active` flag per entry, and an inactive
//! entry is untouched.
//!
//! The scalars are formed on the host in f64 and stored as f32, as tessl
//! forms them (`tessl/src/qwen35_adamw.rs:224-247`). With `lr_i = lr *
//! lr_scale[i]` and `t` the entry's count after its increment:
//!
//! ```text
//! decay_mul = f32(1 - lr_i * wd[i])   decoupled decay scales with lr x scale (D3)
//! step_size = f32(lr_i / (1 - beta1^t))
//! bc2_sqrt  = f32((1 - beta2^t)^0.5)  bias correction outside the sqrt
//! lerp_w = f32(1 - beta1), beta2, one_minus_beta2 = f32(1 - beta2), eps, grad_scale
//! ```
//!
//! Per element, in tessl's kernel order (`kernels/qwen35_adamw.metal:58-66`),
//! every operation one IEEE f32 rounding:
//!
//! ```text
//! w  = p * decay_mul
//! gi = g * grad_scale
//! d  = gi - m
//! m' = lerp_w < 0.5 ? m + lerp_w * d : gi - d * (1 - lerp_w)   torch's two-form lerp
//! v' = v * beta2 + (one_minus_beta2 * gi) * gi
//! p' = w + (-step_size) * (m' / (sqrt(v') / bc2_sqrt + eps))
//! ```
//!
//! `grad_scale` is the clip coefficient (`clip_grad_norm_`'s `.grad` scaling,
//! in f32, as tessl applies it). A value of 1.0 leaves `g` bit-for-bit.
//!
//! # The squared norm
//!
//! `grad_sq_norm` is per-block f32 partials in a fixed order, summed in f64 on
//! the host (the K11 row). Each window (an entry with a gradient) is cut into
//! [`SQ_CHUNK`]-element chunks, one block of [`SQ_THREADS`] threads each.
//! Thread `t` accumulates `x * x` by `fmaf` over elements `t, t + 256, ...` of
//! its chunk, from +0 (`small_common::thread_partials`). The block then folds
//! the 256 values with the crate's one block reduction, `qd_block_sum`, whose
//! host mirror is `small_common::block_sum` (warp butterfly, then a butterfly
//! over the warp sums; converged onto it by the lead's ruling rather than
//! keeping a private tree). Partials are laid out window after window, chunk
//! after chunk, and summed in f64 in that order. The grid is a function of
//! the window lengths only, never of the SM count.

use crate::error::CudaError;
use crate::small_common::{block_sum, thread_partials};

/// Threads per block of the squared-norm kernel.
pub const SQ_THREADS: usize = 256;
/// Elements each thread visits per chunk.
pub const SQ_PER_THREAD: usize = 16;
/// Elements per block: one partial each.
pub const SQ_CHUNK: usize = SQ_THREADS * SQ_PER_THREAD;

/// The per-step hyperparameters shared by every entry. `lr` is the base
/// learning rate, before any entry's `lr_scale`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdamwHyper {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    /// The clip coefficient multiplied into every gradient, or 1.
    pub grad_scale: f64,
}

impl AdamwHyper {
    /// Refuse what torch would carry into every parameter: a non-finite or
    /// negative lr or grad scale, betas outside `[0, 1)`, a non-positive eps.
    /// tessl's checks (`tessl/src/qwen35_adamw.rs:200-219`).
    pub fn validate(&self) -> Result<(), CudaError> {
        const OP: &str = "AdamwHyper";
        let bad = |what: String| Err(CudaError::invalid(OP, what));
        if !(self.lr.is_finite() && self.lr >= 0.0) {
            return bad(format!("lr {} must be finite and >= 0", self.lr));
        }
        if !(self.grad_scale.is_finite() && self.grad_scale >= 0.0) {
            return bad(format!(
                "grad_scale {} must be finite and >= 0",
                self.grad_scale
            ));
        }
        for (name, b) in [("beta1", self.beta1), ("beta2", self.beta2)] {
            if !(0.0..1.0).contains(&b) {
                return bad(format!("{name} {b} must lie in [0, 1)"));
            }
        }
        if !(self.eps.is_finite() && self.eps > 0.0) {
            return bad(format!("eps {} must be finite and > 0", self.eps));
        }
        Ok(())
    }

    /// The scalars shared by every entry of one step.
    pub fn step_scalars(&self) -> StepScalars {
        StepScalars {
            lerp_w: (1.0 - self.beta1) as f32,
            beta2: self.beta2 as f32,
            one_minus_beta2: (1.0 - self.beta2) as f32,
            eps: self.eps as f32,
            grad_scale: self.grad_scale as f32,
        }
    }
}

/// Scalars shared by every entry of a step, as the kernel reads them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepScalars {
    pub lerp_w: f32,
    pub beta2: f32,
    pub one_minus_beta2: f32,
    pub eps: f32,
    pub grad_scale: f32,
}

/// One entry's scalars for one step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EntryScalars {
    pub decay_mul: f32,
    pub step_size: f32,
    pub bc2_sqrt: f32,
}

/// Form one entry's scalars in f64 for its step count `t` (after the
/// increment, 1-based) and store them as f32.
pub fn entry_scalars(
    h: &AdamwHyper,
    t: u64,
    lr_scale: f64,
    weight_decay: f64,
) -> Result<EntryScalars, CudaError> {
    const OP: &str = "entry_scalars";
    if t == 0 {
        return Err(CudaError::invalid(OP, "step counts start at 1"));
    }
    if !(lr_scale.is_finite() && lr_scale > 0.0) {
        return Err(CudaError::invalid(
            OP,
            format!("lr_scale {lr_scale} must be finite and > 0"),
        ));
    }
    if !(weight_decay.is_finite() && weight_decay >= 0.0) {
        return Err(CudaError::invalid(
            OP,
            format!("weight_decay {weight_decay} must be finite and >= 0"),
        ));
    }
    // t is exact in f64 below 2^53; a count past that is refused.
    if t > (1u64 << 53) {
        return Err(CudaError::invalid(
            OP,
            format!("step count {t} exceeds 2^53"),
        ));
    }
    let tf = t as f64;
    let lr = h.lr * lr_scale;
    let bc1 = 1.0 - h.beta1.powf(tf);
    let bc2 = 1.0 - h.beta2.powf(tf);
    Ok(EntryScalars {
        decay_mul: (1.0 - lr * weight_decay) as f32,
        step_size: (lr / bc1) as f32,
        bc2_sqrt: bc2.powf(0.5) as f32,
    })
}

/// The kernel's per-element update, in place, bit for bit. Lengths must match.
pub fn adamw_update_f32(
    p: &mut [f32],
    g: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    e: EntryScalars,
    s: StepScalars,
) -> Result<(), CudaError> {
    let n = p.len();
    if g.len() != n || m.len() != n || v.len() != n {
        return Err(CudaError::invalid(
            "adamw_update_f32",
            format!("lengths p {n}, g {}, m {}, v {}", g.len(), m.len(), v.len()),
        ));
    }
    for k in 0..n {
        let w = p[k] * e.decay_mul;
        let gi = g[k] * s.grad_scale;
        let d = gi - m[k];
        let mi = if s.lerp_w < 0.5 {
            m[k] + s.lerp_w * d
        } else {
            gi - d * (1.0 - s.lerp_w)
        };
        let vi = v[k] * s.beta2 + (s.one_minus_beta2 * gi) * gi;
        let denom = vi.sqrt() / e.bc2_sqrt + s.eps;
        p[k] = w + (-e.step_size) * (mi / denom);
        m[k] = mi;
        v[k] = vi;
    }
    Ok(())
}

/// One bank entry: a name, its window in the flat bank, and its own
/// learning-rate scale and weight decay.
#[derive(Clone, Debug, PartialEq)]
pub struct AdamwEntry {
    pub name: String,
    pub offset: usize,
    pub len: usize,
    pub lr_scale: f64,
    pub weight_decay: f64,
}

/// The per-entry table over a flat bank of `bank_len` f32s. Windows are
/// non-empty, inside the bank and disjoint; names are unique; every scale is
/// finite and positive and every decay finite and non-negative.
#[derive(Clone, Debug, PartialEq)]
pub struct AdamwTable {
    entries: Vec<AdamwEntry>,
    bank_len: usize,
}

impl AdamwTable {
    /// Validate and build.
    pub fn new(entries: Vec<AdamwEntry>, bank_len: usize) -> Result<Self, CudaError> {
        const OP: &str = "AdamwTable";
        if entries.is_empty() {
            return Err(CudaError::invalid(OP, "no entries"));
        }
        let mut spans: Vec<(usize, usize, &str)> = Vec::with_capacity(entries.len());
        let mut names = std::collections::BTreeSet::new();
        for e in &entries {
            if !names.insert(e.name.as_str()) {
                return Err(CudaError::invalid(OP, format!("entry {} repeated", e.name)));
            }
            if e.len == 0 {
                return Err(CudaError::invalid(OP, format!("entry {} is empty", e.name)));
            }
            let end = e
                .offset
                .checked_add(e.len)
                .filter(|&end| end <= bank_len)
                .ok_or_else(|| {
                    CudaError::invalid(
                        OP,
                        format!(
                            "entry {} [{}, +{}) runs past the {bank_len}-element bank",
                            e.name, e.offset, e.len
                        ),
                    )
                })?;
            if !(e.lr_scale.is_finite() && e.lr_scale > 0.0) {
                return Err(CudaError::invalid(
                    OP,
                    format!("entry {}: lr_scale {}", e.name, e.lr_scale),
                ));
            }
            if !(e.weight_decay.is_finite() && e.weight_decay >= 0.0) {
                return Err(CudaError::invalid(
                    OP,
                    format!("entry {}: weight_decay {}", e.name, e.weight_decay),
                ));
            }
            spans.push((e.offset, end, e.name.as_str()));
        }
        spans.sort_unstable();
        for pair in spans.windows(2) {
            if pair[1].0 < pair[0].1 {
                return Err(CudaError::invalid(
                    OP,
                    format!("entries {} and {} overlap", pair[0].2, pair[1].2),
                ));
            }
        }
        Ok(AdamwTable { entries, bank_len })
    }

    /// Entries in the order given.
    pub fn entries(&self) -> &[AdamwEntry] {
        &self.entries
    }

    /// Bank length in f32s.
    pub fn bank_len(&self) -> usize {
        self.bank_len
    }
}

/// The optimizer state the host keeps beside a bank: the table and each
/// entry's step count. The moments live in the bank's own buffers (host
/// vectors here; device buffers in `crate::k11`).
///
/// A step is planned ([`Self::plan_step`]), carried out, then committed
/// ([`Self::commit`]). If carrying it out fails part-way, some windows are
/// updated and the counts are not: the bank and its buffers are dead. Fail
/// closed; there is no retry and no rollback.
#[derive(Clone, Debug, PartialEq)]
pub struct AdamwBank {
    table: AdamwTable,
    steps: Vec<u64>,
}

impl AdamwBank {
    pub fn new(table: AdamwTable) -> Self {
        let n = table.entries().len();
        AdamwBank {
            table,
            steps: vec![0; n],
        }
    }

    pub fn table(&self) -> &AdamwTable {
        &self.table
    }

    /// Each entry's step count.
    pub fn steps(&self) -> &[u64] {
        &self.steps
    }

    /// The scalars every active entry takes this step, and the counts after
    /// it, without committing them. `active` has one flag per entry.
    pub fn plan_step(&self, h: &AdamwHyper, active: &[bool]) -> Result<StepPlan, CudaError> {
        h.validate()?;
        let n = self.table.entries().len();
        if active.len() != n {
            return Err(CudaError::invalid(
                "AdamwBank::plan_step",
                format!("{} active flags for {n} entries", active.len()),
            ));
        }
        let mut next = self.steps.clone();
        let mut per = Vec::with_capacity(n);
        for (i, e) in self.table.entries().iter().enumerate() {
            if !active[i] {
                per.push(None);
                continue;
            }
            next[i] = next[i].checked_add(1).ok_or_else(|| {
                CudaError::invalid("AdamwBank::plan_step", format!("{}: step overflow", e.name))
            })?;
            per.push(Some(entry_scalars(h, next[i], e.lr_scale, e.weight_decay)?));
        }
        Ok(StepPlan {
            shared: h.step_scalars(),
            per_entry: per,
            next,
        })
    }

    /// Record a step [`Self::plan_step`] planned and the device (or the host
    /// emulation) carried out: the plan's `next` counts.
    pub fn commit(&mut self, next: Vec<u64>) -> Result<(), CudaError> {
        if next.len() != self.steps.len() {
            return Err(CudaError::invalid(
                "AdamwBank::commit",
                "step counts for a different table",
            ));
        }
        self.steps = next;
        Ok(())
    }

    /// One step of the f32 emulation over host banks: what the device does,
    /// bit for bit. An inactive entry's window is untouched.
    pub fn step_f32(
        &mut self,
        p: &mut [f32],
        g: &[f32],
        m: &mut [f32],
        v: &mut [f32],
        active: &[bool],
        h: &AdamwHyper,
    ) -> Result<(), CudaError> {
        let len = self.table.bank_len();
        if p.len() != len || g.len() != len || m.len() != len || v.len() != len {
            return Err(CudaError::invalid(
                "AdamwBank::step_f32",
                format!("banks must hold {len} f32s"),
            ));
        }
        let plan = self.plan_step(h, active)?;
        for (e, sc) in self.table.entries().iter().zip(&plan.per_entry) {
            let Some(sc) = sc else { continue };
            let r = e.offset..e.offset + e.len;
            adamw_update_f32(
                &mut p[r.clone()],
                &g[r.clone()],
                &mut m[r.clone()],
                &mut v[r],
                *sc,
                plan.shared,
            )?;
        }
        self.commit(plan.next)
    }
}

/// One planned step: the scalars shared by every entry, each entry's own
/// (`None` for an entry with no gradient this step), and the step counts
/// after it.
#[derive(Clone, Debug, PartialEq)]
pub struct StepPlan {
    pub shared: StepScalars,
    pub per_entry: Vec<Option<EntryScalars>>,
    pub next: Vec<u64>,
}

/// The squared-norm kernel's partials for one window, bit for bit.
pub fn sq_partials_f32(g: &[f32]) -> Vec<f32> {
    g.chunks(SQ_CHUNK)
        .map(|chunk| {
            // Thread t folds chunk[t], chunk[t + 256], ... (at most
            // SQ_PER_THREAD of them) by fmaf from +0, ascending.
            let lanes = thread_partials(
                SQ_THREADS,
                chunk.len(),
                0.0,
                |acc, x| x.mul_add(x, acc),
                |i| chunk[i],
            );
            block_sum(&lanes)
        })
        .collect()
}

/// Chunks (partials) a window of `len` elements produces.
pub fn sq_chunks(len: usize) -> usize {
    len.div_ceil(SQ_CHUNK)
}

/// The partials summed in f64, in order.
pub fn sum_partials_f64(partials: &[f32]) -> f64 {
    partials.iter().map(|&x| f64::from(x)).sum()
}

/// The squared-norm partials of a gradient bank over its active entries
/// (torch's `clip_grad_norm_` sees only parameters with a gradient): every
/// active window's partials, in table order, as the device lays them out.
pub fn bank_sq_partials_f32(
    table: &AdamwTable,
    g: &[f32],
    active: &[bool],
) -> Result<Vec<f32>, CudaError> {
    if g.len() != table.bank_len() || active.len() != table.entries().len() {
        return Err(CudaError::invalid(
            "bank_sq_partials_f32",
            format!(
                "{} gradients and {} flags for a {}-element, {}-entry table",
                g.len(),
                active.len(),
                table.bank_len(),
                table.entries().len()
            ),
        ));
    }
    let mut partials = Vec::new();
    for (e, &on) in table.entries().iter().zip(active) {
        if on {
            partials.extend(sq_partials_f32(&g[e.offset..e.offset + e.len]));
        }
    }
    Ok(partials)
}

/// [`bank_sq_partials_f32`] summed in f64, in order.
pub fn bank_sq_norm_f32(table: &AdamwTable, g: &[f32], active: &[bool]) -> Result<f64, CudaError> {
    Ok(sum_partials_f64(&bank_sq_partials_f32(table, g, active)?))
}

/// One run of the emulation: what the device must reproduce bit for bit.
#[derive(Clone, Debug, PartialEq)]
pub struct Trajectory {
    /// The parameters after each step.
    pub w: Vec<Vec<f32>>,
    /// The moments after the last step.
    pub m: Vec<f32>,
    pub v: Vec<f32>,
    /// Each step's squared-norm partials, taken on that step's gradients
    /// before the update (the clip's order), and their f64 sums.
    pub partials: Vec<Vec<f32>>,
    pub norms: Vec<f64>,
    /// Per-entry step counts at the end.
    pub steps: Vec<u64>,
}

/// Replay `grads` (each step's gradient bank and active flags) from `init`
/// and zero moments.
pub fn replay_f32(
    table: &AdamwTable,
    h: &AdamwHyper,
    init: &[f32],
    grads: &[(Vec<f32>, Vec<bool>)],
) -> Result<Trajectory, CudaError> {
    let n = table.bank_len();
    if init.len() != n {
        return Err(CudaError::invalid(
            "replay_f32",
            format!("init holds {}, the bank {n}", init.len()),
        ));
    }
    let mut bank = AdamwBank::new(table.clone());
    let (mut p, mut m, mut v) = (init.to_vec(), vec![0.0f32; n], vec![0.0f32; n]);
    let mut out = Trajectory {
        w: Vec::with_capacity(grads.len()),
        m: Vec::new(),
        v: Vec::new(),
        partials: Vec::with_capacity(grads.len()),
        norms: Vec::with_capacity(grads.len()),
        steps: Vec::new(),
    };
    for (g, active) in grads {
        let parts = bank_sq_partials_f32(table, g, active)?;
        out.norms.push(sum_partials_f64(&parts));
        out.partials.push(parts);
        bank.step_f32(&mut p, g, &mut m, &mut v, active, h)?;
        out.w.push(p.clone());
    }
    out.m = m;
    out.v = v;
    out.steps = bank.steps().to_vec();
    Ok(out)
}

/// One bank run's inputs: a table, the hyperparameters, the initial bank and
/// each step's gradients with their active flags.
#[derive(Clone, Debug)]
pub struct BankCase {
    pub label: String,
    pub table: AdamwTable,
    pub hyper: AdamwHyper,
    pub init: Vec<f32>,
    pub grads: Vec<(Vec<f32>, Vec<bool>)>,
}

/// Window lengths of the synthetic cases: one element, both sides of a
/// 4096-element chunk, a ragged multi-chunk window and a ~1M window (245
/// chunks).
pub const SYNTHETIC_WINDOWS: [usize; 6] = [1, 4095, 4096, 4097, 3 * 4096 + 17, 1_000_003];

/// Two synthetic bank runs of three steps each, beside the goldens:
/// - every lr scale (1.0, 0.1) and decay (0.01, 0, 0.1) in turn over
///   [`SYNTHETIC_WINDOWS`], with entry 1 inactive on step 2 and entry 4 on
///   step 1 (D7);
/// - `lerp_lt_half`: F's betas at lr 1e-2, `grad_scale` 1;
/// - `lerp_ge_half_clipped`: beta1 0.3 (torch's other `lerp` form), beta2
///   0.95, lr 3e-3 and a clip coefficient of 0.37.
pub fn synthetic_cases() -> Result<Vec<BankCase>, CudaError> {
    let mut rows = Vec::new();
    let mut offset = 0usize;
    for (i, &len) in SYNTHETIC_WINDOWS.iter().enumerate() {
        rows.push(AdamwEntry {
            name: format!("w{i}_{len}"),
            offset,
            len,
            lr_scale: [1.0, 0.1][i % 2],
            weight_decay: [0.01, 0.0, 0.1][i % 3],
        });
        offset += len;
    }
    let table = AdamwTable::new(rows, offset)?;
    let n = table.bank_len();
    let entries = table.entries().len();
    let init = crate::inputs::splitmix_f32(11, n, 1.0);
    let grads: Vec<(Vec<f32>, Vec<bool>)> = (0..3u64)
        .map(|t| {
            let active: Vec<bool> = (0..entries)
                .map(|i| !((i == 1 && t == 1) || (i == 4 && t == 0)))
                .collect();
            let mut g = crate::inputs::splitmix_f32(100 + t, n, 0.05);
            for (e, &on) in table.entries().iter().zip(&active) {
                if !on {
                    g[e.offset..e.offset + e.len].fill(0.0);
                }
            }
            (g, active)
        })
        .collect();
    let hypers = [
        (
            "lerp_lt_half",
            AdamwHyper {
                lr: 1e-2,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                grad_scale: 1.0,
            },
        ),
        (
            "lerp_ge_half_clipped",
            AdamwHyper {
                lr: 3e-3,
                beta1: 0.3,
                beta2: 0.95,
                eps: 1e-8,
                grad_scale: 0.37,
            },
        ),
    ];
    hypers
        .into_iter()
        .map(|(label, hyper)| {
            hyper.validate()?;
            Ok(BankCase {
                label: label.to_string(),
                table: table.clone(),
                hyper,
                init: init.clone(),
                grads: grads.clone(),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_synthetic_cases_cover_both_lerp_forms_d7_and_the_clip() {
        let cases = synthetic_cases().unwrap();
        assert_eq!(cases.len(), 2);
        let lerps: Vec<bool> = cases
            .iter()
            .map(|c| c.hyper.step_scalars().lerp_w < 0.5)
            .collect();
        assert_eq!(lerps, [true, false]);
        assert!(cases[1].hyper.step_scalars().grad_scale != 1.0);
        for c in &cases {
            let t = replay_f32(&c.table, &c.hyper, &c.init, &c.grads).unwrap();
            assert_eq!(t.steps, [3, 2, 3, 3, 2, 3]);
            assert!(t.w.iter().flatten().all(|x| x.is_finite()));
            // The ~1M window is 245 chunks; partials cover only active windows.
            assert_eq!(sq_chunks(1_000_003), 245);
            assert_eq!(
                t.partials[0].len(),
                SYNTHETIC_WINDOWS
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| i != 4)
                    .map(|(_, &l)| sq_chunks(l))
                    .sum::<usize>()
            );
        }
    }

    fn f_hyper(lr: f64) -> AdamwHyper {
        AdamwHyper {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            grad_scale: 1.0,
        }
    }

    #[test]
    fn scalars_are_formed_in_f64_and_stored_as_f32() {
        let h = f_hyper(1e-2);
        let e = entry_scalars(&h, 1, 0.1, 0.01).unwrap();
        assert_eq!(e.decay_mul, (1.0 - 1e-2 * 0.1 * 0.01) as f32);
        assert_eq!(e.step_size, (1e-2 * 0.1 / (1.0 - 0.9f64)) as f32);
        assert_eq!(e.bc2_sqrt, (1.0 - 0.999f64).powf(0.5) as f32);
        // D3: the decay carries the entry's scale, not the base lr.
        let full = entry_scalars(&h, 1, 1.0, 0.01).unwrap();
        assert_ne!(e.decay_mul, full.decay_mul);
        let s = h.step_scalars();
        assert_eq!(s.lerp_w, (1.0 - 0.9f64) as f32);
        assert_eq!(s.one_minus_beta2, (1.0 - 0.999f64) as f32);
        assert!(entry_scalars(&h, 0, 1.0, 0.0).is_err());
        assert!(entry_scalars(&h, 1, 0.0, 0.0).is_err());
        assert!(entry_scalars(&h, 1, 1.0, -1.0).is_err());
        assert!(entry_scalars(&h, 1, f64::NAN, 0.0).is_err());
    }

    #[test]
    fn hyperparameters_are_refused_as_tessl_refuses_them() {
        assert!(f_hyper(1e-3).validate().is_ok());
        for bad in [
            AdamwHyper {
                lr: f64::NAN,
                ..f_hyper(1e-3)
            },
            AdamwHyper {
                lr: -1.0,
                ..f_hyper(1e-3)
            },
            AdamwHyper {
                beta1: 1.0,
                ..f_hyper(1e-3)
            },
            AdamwHyper {
                beta2: -0.1,
                ..f_hyper(1e-3)
            },
            AdamwHyper {
                eps: 0.0,
                ..f_hyper(1e-3)
            },
            AdamwHyper {
                grad_scale: f64::INFINITY,
                ..f_hyper(1e-3)
            },
        ] {
            assert!(bad.validate().is_err(), "{bad:?}");
        }
    }

    #[test]
    fn the_update_is_torch_order_on_a_worked_element() {
        // One element, step 1, against the same arithmetic in f64 then
        // rounded per operation by hand.
        let h = f_hyper(1e-2);
        let e = entry_scalars(&h, 1, 1.0, 0.01).unwrap();
        let s = h.step_scalars();
        let (mut p, g, mut m, mut v) = ([0.75f32], [0.5f32], [0.0f32], [0.0f32]);
        adamw_update_f32(&mut p, &g, &mut m, &mut v, e, s).unwrap();
        let w = 0.75f32 * e.decay_mul;
        let mi = 0.0f32 + s.lerp_w * (0.5f32 - 0.0);
        let vi = 0.0f32 * s.beta2 + (s.one_minus_beta2 * 0.5) * 0.5;
        let want = w + (-e.step_size) * (mi / (vi.sqrt() / e.bc2_sqrt + s.eps));
        assert_eq!(p[0].to_bits(), want.to_bits());
        assert_eq!((m[0], v[0]), (mi, vi));
        // At step 1 AdamW moves p by about lr * sign(g).
        assert!((f64::from(p[0]) - (0.75 * (1.0 - 1e-4) - 1e-2)).abs() < 1e-6);
    }

    #[test]
    fn a_grad_scale_of_one_leaves_the_gradient_bitwise() {
        for x in [0.1f32, -3.5, 1e-30, f32::MIN_POSITIVE, 7.0e30] {
            assert_eq!((x * 1.0f32).to_bits(), x.to_bits());
        }
    }

    fn table(
        spec: &[(&str, usize, usize, f64, f64)],
        bank: usize,
    ) -> Result<AdamwTable, CudaError> {
        AdamwTable::new(
            spec.iter()
                .map(|&(n, o, l, s, w)| AdamwEntry {
                    name: n.to_string(),
                    offset: o,
                    len: l,
                    lr_scale: s,
                    weight_decay: w,
                })
                .collect(),
            bank,
        )
    }

    #[test]
    fn tables_refuse_overlap_overrun_repeats_and_bad_values() {
        assert!(table(&[("a", 0, 4, 1.0, 0.01), ("b", 4, 2, 0.1, 0.01)], 6).is_ok());
        assert!(table(&[("a", 0, 4, 1.0, 0.01), ("b", 3, 2, 0.1, 0.01)], 6).is_err());
        assert!(table(&[("a", 0, 7, 1.0, 0.01)], 6).is_err());
        assert!(table(&[("a", usize::MAX, 2, 1.0, 0.01)], 6).is_err());
        assert!(table(&[("a", 0, 2, 1.0, 0.0), ("a", 2, 2, 1.0, 0.0)], 6).is_err());
        assert!(table(&[("a", 0, 0, 1.0, 0.0)], 6).is_err());
        assert!(table(&[("a", 0, 2, 0.0, 0.0)], 6).is_err());
        assert!(table(&[("a", 0, 2, 1.0, f64::NAN)], 6).is_err());
        assert!(table(&[], 6).is_err());
    }

    #[test]
    fn an_inactive_entry_is_untouched_and_keeps_its_own_count() {
        let t = table(&[("tower", 0, 3, 1.0, 0.01), ("head", 3, 2, 1.0, 0.01)], 5).unwrap();
        let mut bank = AdamwBank::new(t);
        let mut p = vec![0.5f32, -0.25, 1.0, 0.75, -0.5];
        let mut m = vec![0.0f32; 5];
        let mut v = vec![0.0f32; 5];
        let g = vec![0.1f32, 0.2, -0.3, 0.4, -0.5];
        let h = f_hyper(1e-2);
        let head0 = p[3..].to_vec();
        bank.step_f32(&mut p, &g, &mut m, &mut v, &[true, false], &h)
            .unwrap();
        assert_eq!(&p[3..], &head0[..], "an entry with no gradient moved");
        assert_eq!((&m[3..], &v[3..]), (&[0.0f32, 0.0][..], &[0.0f32, 0.0][..]));
        assert_eq!(bank.steps(), &[1, 0]);
        bank.step_f32(&mut p, &g, &mut m, &mut v, &[true, true], &h)
            .unwrap();
        assert_eq!(bank.steps(), &[2, 1]);
        // The head's first step uses its own t = 1: bias corrections of step 1.
        let mut q = head0.clone();
        let (mut mm, mut vv) = (vec![0.0f32; 2], vec![0.0f32; 2]);
        adamw_update_f32(
            &mut q,
            &g[3..],
            &mut mm,
            &mut vv,
            entry_scalars(&h, 1, 1.0, 0.01).unwrap(),
            h.step_scalars(),
        )
        .unwrap();
        assert_eq!(&p[3..], &q[..]);
        assert!(bank
            .step_f32(&mut p, &g, &mut m, &mut v, &[true], &h)
            .is_err());
    }

    #[test]
    fn squared_norm_partials_follow_the_fixed_tree() {
        // 2 chunks + a ragged tail.
        let n = 2 * SQ_CHUNK + 777;
        let g: Vec<f32> = (0..n).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
        let parts = sq_partials_f32(&g);
        assert_eq!(parts.len(), 3);
        assert_eq!(sq_chunks(n), 3);
        let exact: f64 = g.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
        let got = sum_partials_f64(&parts);
        assert!((got - exact).abs() <= 1e-6 * exact, "{got} vs {exact}");
        // The tree, by hand, for one chunk of ones: 4096.
        assert_eq!(sq_partials_f32(&vec![1.0f32; SQ_CHUNK]), vec![4096.0]);
        assert_eq!(sq_partials_f32(&[]), Vec::<f32>::new());
        assert_eq!(sq_partials_f32(&[3.0]), vec![9.0]);
    }
}

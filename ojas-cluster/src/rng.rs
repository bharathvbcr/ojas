//! Deterministic `SplitMix64` PRNG, budget-charged permutations, Gumbel donor selection,
//! split-collapse separation, and `max_abs`-prescaled spherical normalization.

use ojas_core::{Budget, OjasError, Scratch};

/// Domain separator mixed with `seed` when sampling a `max_points_per_centroid` row subset.
const SUBSET_SEED_TAG: u64 = 0x5355_4253_4554_5631;
/// Domain separator mixed with `seed` before mixing `epoch` for per-pass permutations.
const EPOCH_SEED_TAG: u64 = 0x4550_4f43_485f_5052;
/// `2^-62` (`0x2080_0000`), positive normal `f32` & `bf16` offset immune to Apple GPU FTZ.
const COLLAPSED_ZERO_SPLIT_F32_BITS: u32 = 0x2080_0000;

fn out_of_range(detail: impl Into<String>) -> OjasError {
    OjasError::OutOfRange {
        op: "emeans",
        detail: detail.into(),
    }
}

/// Deterministic 64-bit `SplitMix64` pseudo-random generator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Seed a new `SplitMix64` stream.
    #[inline]
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Stateless single-word `SplitMix64` avalanche mixer matching Metal `splitmix64_mix`.
    #[inline]
    pub fn mix(mut z: u64) -> u64 {
        z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Advance state and return the next 64-bit word.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let z = self.state;
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        Self::mix(z)
    }

    /// Unbiased uniform integer in `0..bound` via Lemire's 128-bit rejection method
    /// (returns `0` when `bound <= 1`).
    #[inline]
    pub fn next_bounded_u64(&mut self, bound: u64) -> u64 {
        if bound <= 1 {
            return 0;
        }
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let r = self.next_u64();
            let prod = u128::from(r) * u128::from(bound);
            if (prod as u64) >= threshold {
                return (prod >> 64) as u64;
            }
        }
    }
}

/// Deterministically sample `target_n` unique row indices from `0..global_n` in ascending order,
/// charging temporary and returned index storage to `budget`.
pub fn sample_subset_sorted(
    global_n: usize,
    target_n: usize,
    seed: u64,
    budget: &Budget,
) -> Result<Scratch<u32>, OjasError> {
    if target_n > global_n {
        return Err(out_of_range(format!(
            "target_n ({target_n}) exceeds global_n ({global_n})"
        )));
    }
    if global_n > (u32::MAX as usize) {
        return Err(out_of_range(format!(
            "global_n ({global_n}) exceeds u32::MAX"
        )));
    }
    let mut pool = Scratch::<u32>::try_extend(global_n, budget, |dst| {
        dst.extend(0..(global_n as u32));
    })?;
    if target_n == global_n {
        return Ok(pool);
    }

    let mut rng = SplitMix64::new(SplitMix64::mix(seed ^ SUBSET_SEED_TAG));
    let slice = pool.as_mut_slice();
    for i in 0..target_n {
        let rem = (global_n - i) as u64;
        let j = i + (rng.next_bounded_u64(rem) as usize);
        slice.swap(i, j);
    }
    slice[..target_n].sort_unstable();
    Scratch::<u32>::try_from_slice(&pool.as_slice()[..target_n], budget)
}

/// Deterministically generate a permutation of `0..n` for `(seed, epoch)`, charged to `budget`.
pub fn epoch_permutation(
    n: usize,
    seed: u64,
    epoch: usize,
    budget: &Budget,
) -> Result<Scratch<u32>, OjasError> {
    if n > (u32::MAX as usize) {
        return Err(out_of_range(format!("n ({n}) exceeds u32::MAX")));
    }
    let mut perm = Scratch::<u32>::try_extend(n, budget, |dst| {
        dst.extend(0..(n as u32));
    })?;
    let s0 = SplitMix64::mix(seed ^ EPOCH_SEED_TAG);
    let epoch_seed = SplitMix64::mix(s0 ^ (epoch as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
    let mut rng = SplitMix64::new(epoch_seed);

    let slice = perm.as_mut_slice();
    for i in 0..n.saturating_sub(1) {
        let rem = (n - i) as u64;
        let j = i + (rng.next_bounded_u64(rem) as usize);
        slice.swap(i, j);
    }
    Ok(perm)
}

/// Deterministic `SplitMix64` Gumbel donor score matching Metal `emeans_revive_prepare`:
/// `ln(usage) - ln(-ln(u))` with odd-mantissa `u in (0, 1)`.
#[inline]
pub fn gumbel_key(usage: f32, seed: u64, step: u64, cluster_id: u32) -> f32 {
    let s0 = SplitMix64::mix(seed ^ 0xd1b5_4a32_d192_ed03);
    let s1 = SplitMix64::mix(s0 ^ step.wrapping_mul(0x9e37_79b9_7f4a_7c15));
    let h = SplitMix64::mix(s1 ^ u64::from(cluster_id).wrapping_mul(0xbf58_476d_1ce4_e5b9));
    let odd24 = (((h >> 41) as u32) << 1) | 1;
    let u = (odd24 as f32) * f32::from_bits(0x3380_0000);
    usage.ln() - (-u.ln()).ln()
}

/// Select `(donor_id, target_id)` split pairs for dead-cluster revival at `step`.
///
/// Uses the same 256-lane `f32` tree reduction for `mean(usage)` and tie-breaking rules as
/// `emeans_revive_prepare` in `tessl/kernels/emeans.metal`.
pub fn choose_splits(
    usage: &[f32],
    unused_threshold: f32,
    seed: u64,
    step: u64,
) -> Vec<(u32, u32)> {
    if !unused_threshold.is_finite()
        || unused_threshold <= 0.0
        || usage.is_empty()
        || usage.len() > (u32::MAX as usize)
    {
        return Vec::new();
    }

    let mut scratch = [0.0_f32; 256];
    for (j, &u) in usage.iter().enumerate() {
        scratch[j & 255] += u;
    }
    let mut stride = 128;
    while stride > 0 {
        for tid in 0..stride {
            scratch[tid] += scratch[tid + stride];
        }
        stride >>= 1;
    }
    let mean_n = scratch[0] / (usage.len() as f32);
    let thresh = unused_threshold * mean_n;

    let mut unused = Vec::new();
    let mut donors = Vec::new();
    for (idx, &nj) in usage.iter().enumerate() {
        let j = idx as u32;
        if nj < thresh {
            unused.push(j);
        } else if nj > 0.0 {
            let raw_g = gumbel_key(nj, seed, step, j);
            let clean_g = if raw_g == 0.0 { 0.0 } else { raw_g };
            donors.push((clean_g, j));
        }
    }

    if unused.is_empty() || donors.is_empty() {
        return Vec::new();
    }

    donors.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    let nsplit = unused.len().min(donors.len());
    donors[..nsplit]
        .iter()
        .zip(&unused[..nsplit])
        .map(|(&(_, donor_id), &target_id)| (donor_id, target_id))
        .collect()
}

#[inline]
fn f32_bits_to_bf16_rte_ftz(bits: u32) -> u16 {
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

/// Ensure `donor` and `child` differ in FTZ `f32` representation after multiplicative splitting.
pub fn separate_collapsed_split_f32(donor: &[f32], child: &mut [f32]) {
    let d = donor.len().min(child.len());
    if d == 0 || !child[0].is_finite() {
        return;
    }

    let any_diff = donor[..d].iter().zip(&child[..d]).any(|(&dv, &cv)| {
        let df = if (dv.to_bits() & 0x7fff_ffff) < 0x0080_0000 {
            0.0
        } else {
            dv
        };
        let cf = if (cv.to_bits() & 0x7fff_ffff) < 0x0080_0000 {
            0.0
        } else {
            cv
        };
        df != cf
    });
    if any_diff {
        return;
    }

    let u = child[0].to_bits();
    child[0] = if (u & 0x7fff_ffff) < 0x0080_0000 {
        f32::from_bits(COLLAPSED_ZERO_SPLIT_F32_BITS)
    } else if (u & 0x8000_0000) == 0 {
        let u_step = if u >= 0x7f7f_ffff { u - 1 } else { u + 1 };
        f32::from_bits(u_step)
    } else {
        let u_step = if u == 0x8080_0000 { 0x0000_0000 } else { u - 1 };
        f32::from_bits(u_step)
    };
}

/// Ensure `donor` and `child` differ in `bf16` representation after multiplicative splitting.
pub fn separate_collapsed_split_bf16(donor: &[f32], child: &mut [f32]) {
    let d = donor.len().min(child.len());
    if d == 0 || !child[0].is_finite() {
        return;
    }

    let any_diff = donor[..d].iter().zip(&child[..d]).any(|(&dv, &cv)| {
        let dq = bf16_bits_to_f32_clean(f32_bits_to_bf16_rte_ftz(dv.to_bits()));
        let cq = bf16_bits_to_f32_clean(f32_bits_to_bf16_rte_ftz(cv.to_bits()));
        dq != cq
    });
    if any_diff {
        return;
    }

    let b = f32_bits_to_bf16_rte_ftz(child[0].to_bits());
    child[0] = if (b & 0x7fff) < 0x0080 {
        f32::from_bits(COLLAPSED_ZERO_SPLIT_F32_BITS)
    } else if (b & 0x8000) == 0 {
        let b_step = if b >= 0x7f7f { b - 1 } else { b + 1 };
        f32::from_bits(u32::from(b_step) << 16)
    } else {
        let b_step = if b == 0x8080 { 0x0000 } else { b - 1 };
        f32::from_bits(u32::from(b_step) << 16)
    };
}

/// Normalize `src` to unit L2 length in `f64` via `max_abs` prescaling and write `f32` to `dst`.
///
/// All-zero vectors write `0.0f32` everywhere without producing `NaN`.
pub fn prescaled_unit_normalize_f64(src: &[f64], dst: &mut [f32]) {
    let d = src.len().min(dst.len());
    dst[d..].fill(0.0);
    if d == 0 {
        return;
    }

    let max_abs = src[..d].iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
    if max_abs == 0.0 || !max_abs.is_finite() {
        for (i, out) in dst[..d].iter_mut().enumerate() {
            *out = if src[i].is_finite() { 0.0 } else { f32::NAN };
        }
        return;
    }

    let mut sum_sq = 0.0_f64;
    for &v in &src[..d] {
        let w = v / max_abs;
        sum_sq += w * w;
    }
    let norm = sum_sq.sqrt();
    if norm == 0.0 || !norm.is_finite() {
        dst[..d].fill(0.0);
        return;
    }

    for (i, out) in dst[..d].iter_mut().enumerate() {
        *out = ((src[i] / max_abs) / norm) as f32;
    }
}

/// Normalize an `f32` row to unit L2 length via `f64` `max_abs` prescaling and write to `dst`.
pub fn prescaled_unit_normalize_f32(src: &[f32], dst: &mut [f32]) {
    let d = src.len().min(dst.len());
    dst[d..].fill(0.0);
    if d == 0 {
        return;
    }

    let max_abs = src[..d]
        .iter()
        .fold(0.0_f64, |m, &v| m.max(f64::from(v).abs()));
    if max_abs == 0.0 || !max_abs.is_finite() {
        for (i, out) in dst[..d].iter_mut().enumerate() {
            *out = if src[i].is_finite() { 0.0 } else { f32::NAN };
        }
        return;
    }

    let mut sum_sq = 0.0_f64;
    for &v in &src[..d] {
        let w = f64::from(v) / max_abs;
        sum_sq += w * w;
    }
    let norm = sum_sq.sqrt();
    if norm == 0.0 || !norm.is_finite() {
        dst[..d].fill(0.0);
        return;
    }

    for (i, out) in dst[..d].iter_mut().enumerate() {
        *out = ((f64::from(src[i]) / max_abs) / norm) as f32;
    }
}

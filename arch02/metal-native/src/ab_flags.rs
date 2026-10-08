//! Phase-2 / Audit 4–6 latency A/B flags (env, read once).
//!
//! Recovery + Audit 4 (M5 Pro, 2026-07-12):
//! - **Root regress (kept gated):** tiled FA-2 bwd at T=256
//! - **Keep:** Muon NS5 simdgroup; packed M4 compute encoder (runtime)
//! - **Drop:** f32 GEMM interior offset tiles
//!
//! Defaults → row FA + Muon SG + packed encoder + **accum off** ≈
//! **56.6 ms / ~72k tok/s / 250 binders**; Soft EMA **2.050**.
//! Soft quality: `METAL_NATIVE_FA_TILED=1` → Soft EMA **2.037** (~69 ms).
//! Default 20k Soft: FA_TILED=1, GEMM_ACCUM off, `--warmdown 3500` → EMA **1.8969**.
//! Default 100k Soft: same + `--warmdown-start 16000 --warmdown 24000 --lr-floor 0.1
//! --final-warmdown 10000` (WSD; FINAL EMA **1.8828**; last-10% alone explodes ~21k).
//!
//! Overrides:
//!   METAL_NATIVE_FA_TILED=1           → tiled BR=BC=32 bwd — Soft EMA 2.037 KEEP;
//!                                       slower at T=256 (~69 ms); **required** for
//!                                       the default 20k Soft recipe (DECISIONS M11)
//!   METAL_NATIVE_MUON_SG=0            → hand-loop NS5 (much slower)
//!   METAL_NATIVE_GEMM_INTERIOR=1      → interior offset tiles on f32 GEMM
//!   METAL_NATIVE_HAZARD_BARRIERS=1    → skip always-on Device barrier after each
//!                                       dispatch (packed ops keep explicit barriers).
//!                                       A/B REJECT as default: NaNs by ~step 3.
//!   METAL_NATIVE_GEMM_ACCUM=1         → TensorOps multiply_accumulate for bwd
//!                                       dW/dx (Audit 6 P1a/P1a2). Default **off**:
//!                                       Soft EMA/gnorm regress with accum on
//!                                       (2.058 / late gnorm~9 vs 2.050 / ~3.4).
//!                                       Opt-in for binder/step-time A/B (211 vs 250).
//!   METAL_NATIVE_RESID_BWD_FUSE=0     → Soft-bisect: unfused rms + resid/scale bwd
//!                                       instead of Audit 6 megakernels (P1b)
//!   METAL_NATIVE_MLP_COOP_POSTFIX=1   → reserved (not wired; needs NAX util + kernel)
//!
//! Audit 7 (exact-128M backward, 2026-07-19 — A/B'd on M5 Pro, see
//! docs/optimization_map.md: CAST_ONCE+ACCUM_DX −7.2% step / −220 disp KEEP;
//! FA_TILED @128M −0.0046 BPB @500 KEEP for quality; defaults stay OFF until
//! the combined 500-step probe + champion rerun land):
//!   METAL_NATIVE_BWD_PROFILE=1        → per-section backward wall clock (syncs
//!                                       between sections — diagnostic only, do
//!                                       not read step_ms as a gate under this)
//!   METAL_NATIVE_BWD_CAST_ONCE=1      → under Bf16, cast shared f32 grad
//!                                       operands to bf16 once per site instead
//!                                       of once per GEMM (bit-identical inputs
//!                                       to each GEMM; pure dispatch/bandwidth)
//!   METAL_NATIVE_GEMM_ACCUM_DX=1      → multiply_accumulate for **dX** NT
//!                                       accums only (fresh pre-zeroed buffers).
//!                                       dW bank accum stays temp+add — the
//!                                       Audit 6 Soft regression suspect.
//!   METAL_NATIVE_FA_FAST=1            → head-dim-specialized **row** FA bwd
//!                                       (`*_d32_f32`): hoists loop-invariant
//!                                       Q/dO and K/V out of the inner loop and
//!                                       keeps accumulators in registers.
//!                                       Numerically identical to the generic
//!                                       row kernels. Requires head_dim == 32;
//!                                       ignored under FA_TILED.
//!   METAL_NATIVE_FA_BF16=1            → bf16 Q/K/V with f32
//!                                       dO/L/Delta/scores/accum/grads.
//!                                       Bandwidth approximation, **not** a
//!                                       consistency fix: `use_bf16_flash` is
//!                                       hard-coded false so the fwd is always
//!                                       f32 flash. Justified empirically (two
//!                                       seeds + 2000-step champion: no quality
//!                                       cost). Requires `PrecisionMode::Bf16`
//!                                       so f32 goldens keep the exact path.
//!                                       Works on both paths:
//!                                        - row (default): `*_row_d32_bf16`,
//!                                          implies FA_FAST, needs head_dim 32
//!                                        - `FA_TILED=1`: the Phase 4
//!                                          `flash_attn_bwd_{dq,dkv}_bf16`
//!                                          twins, any head_dim ≤ 64. These
//!                                          shipped in the metallib but were
//!                                          unreferenced until Audit 7.
//!   METAL_NATIVE_FA_BLOCKSOFT=1       → FA-2 blockwise online softmax fwd
//!                                       (rowmax over BC + one rescale; precise
//!                                       exp/log + fma). Soft@3k quality probe
//!                                       (Phase G). Not bit-identical to the
//!                                       sequential path. Default **off**.

use std::sync::OnceLock;

fn env_truthy(name: &str) -> Option<bool> {
    match std::env::var(name).ok().as_deref() {
        Some("1") | Some("true") | Some("TRUE") | Some("yes") => Some(true),
        Some("0") | Some("false") | Some("FALSE") | Some("no") => Some(false),
        _ => None,
    }
}

struct AbFlags {
    fa_tiled: bool,
    muon_sg: bool,
    gemm_interior: bool,
    hazard_barriers: bool,
    gemm_accum: bool,
    resid_bwd_fuse: bool,
    bwd_profile: bool,
    bwd_cast_once: bool,
    gemm_accum_dx: bool,
    fa_fast: bool,
    fa_bf16: bool,
    fwd_profile: bool,
    fa_fwd_fast: bool,
    fa_blocksoft: bool,
    optim_profile: bool,
    glue_rowblock: bool,
}

fn flags() -> &'static AbFlags {
    static FLAGS: OnceLock<AbFlags> = OnceLock::new();
    FLAGS.get_or_init(|| AbFlags {
        fa_tiled: env_truthy("METAL_NATIVE_FA_TILED").unwrap_or(false),
        muon_sg: env_truthy("METAL_NATIVE_MUON_SG").unwrap_or(true),
        gemm_interior: env_truthy("METAL_NATIVE_GEMM_INTERIOR").unwrap_or(false),
        hazard_barriers: env_truthy("METAL_NATIVE_HAZARD_BARRIERS").unwrap_or(false),
        // Default OFF: Audit 6 multiply_accumulate regresses Soft late gnorm / BPB.
        gemm_accum: env_truthy("METAL_NATIVE_GEMM_ACCUM").unwrap_or(false),
        resid_bwd_fuse: env_truthy("METAL_NATIVE_RESID_BWD_FUSE").unwrap_or(true),
        bwd_profile: env_truthy("METAL_NATIVE_BWD_PROFILE").unwrap_or(false),
        // Audit 7 KEEPs — **default ON** after the 2000-step champion rerun
        // (M15/M16): BPB 2.015576 vs 2.015756, +44.7% throughput, zero
        // nonfinite. Set the env var to 0 to restore the pre-Audit-7 path.
        bwd_cast_once: env_truthy("METAL_NATIVE_BWD_CAST_ONCE").unwrap_or(true),
        fa_fast: env_truthy("METAL_NATIVE_FA_FAST").unwrap_or(true),
        // bf16 FA bwd additionally requires PrecisionMode::Bf16 + head_dim 32
        // (checked in model_bwd), so f32 golden runs keep the exact path.
        fa_bf16: env_truthy("METAL_NATIVE_FA_BF16").unwrap_or(true),
        // ACCUM_DX stays OFF: two seeds could not separate it from noise
        // (sign flipped), and it inherits the Audit 6 accumulate-mode risk.
        gemm_accum_dx: env_truthy("METAL_NATIVE_GEMM_ACCUM_DX").unwrap_or(false),
        fwd_profile: env_truthy("METAL_NATIVE_FWD_PROFILE").unwrap_or(false),
        // Audit 8 KEEP / Audit 9A: default ON (flash 264→39 ms; D=32 gate in dispatch).
        fa_fwd_fast: env_truthy("METAL_NATIVE_FA_FWD_FAST").unwrap_or(true),
        // Phase G Soft quality: FA-2 blockwise softmax. Default off.
        fa_blocksoft: env_truthy("METAL_NATIVE_FA_BLOCKSOFT").unwrap_or(false),
        optim_profile: env_truthy("METAL_NATIVE_OPTIM_PROFILE").unwrap_or(false),
        // Rowblock cut atomics ~128× but A/B ≈ noise (Audit 8 glue sweep) — keep OFF.
        glue_rowblock: env_truthy("METAL_NATIVE_GLUE_ROWBLOCK").unwrap_or(false),
    })
}

/// Number of row-blocks for the glue-bwd reduction (`METAL_NATIVE_GLUE_ROWBLOCKS`).
/// Higher = more parallelism and fewer serial iterations, but more atomics
/// (C x row_blocks). 32 gives 24576 atomics/call vs 3.1M for the inline path.
pub fn glue_row_blocks() -> usize {
    std::env::var("METAL_NATIVE_GLUE_ROWBLOCKS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(32)
}

/// Use tiled BR=BC=32 flash bwd. Default: false (row wins step-time at T=256).
/// Soft quality: `METAL_NATIVE_FA_TILED=1` → Soft EMA **2.0369** (~69 ms).
pub fn fa_tiled_bwd() -> bool {
    flags().fa_tiled
}

/// Use simdgroup inner GEMM in Muon NS5. Default: true.
pub fn muon_simdgroup() -> bool {
    flags().muon_sg
}

/// Use interior offset tile extents in f32 GEMM. Default: false.
pub fn gemm_interior_offsets() -> bool {
    flags().gemm_interior
}

/// Skip always-on Dispatch→Dispatch Device barrier after every `Binder::dispatch`.
/// Packed multi-dispatch ops still insert explicit barriers. Default: false
/// (golden-safe). **Do not enable as default** — A/B produced NaNs by ~step 3.
pub fn hazard_barriers() -> bool {
    flags().hazard_barriers
}

/// Use TensorOps `multiply_accumulate` for bwd dW / dx accumulate (P1a/P1a2).
/// Default: **false** (temp GEMM + add_inplace). Soft A/B: accum-on → EMA 2.058 /
/// late gnorm ~9; accum-off → EMA 2.050 / late gnorm ~3.4. Set
/// `METAL_NATIVE_GEMM_ACCUM=1` for binder/step-time A/B (~211 vs ~250).
pub fn gemm_accum() -> bool {
    flags().gemm_accum
}

/// Use fused resid/RMS bwd megakernels (P1b). Default: true.
/// Set `METAL_NATIVE_RESID_BWD_FUSE=0` for unfused rms_norm_scale_bwd +
/// residual_scale_add_bwd / resid_mix_bwd_simple.
pub fn resid_bwd_fuse() -> bool {
    flags().resid_bwd_fuse
}

/// Per-section backward timing (syncs between sections; diagnostic only).
/// Default: false. `METAL_NATIVE_BWD_PROFILE=1` prints an aggregated table
/// to stderr after each backward call.
pub fn bwd_profile() -> bool {
    flags().bwd_profile
}

/// Under Bf16, cast shared f32 grad operands to bf16 once per site instead of
/// once per consuming GEMM. Bit-identical GEMM inputs (same cast kernel, same
/// data) — saves cast dispatches + bandwidth. Default: **true** (Audit 7 KEEP).
pub fn bwd_cast_once() -> bool {
    flags().bwd_cast_once
}

/// TensorOps `multiply_accumulate` for **dX** NT accumulates only (fresh
/// pre-zeroed activation-grad buffers). dW bank accumulation keeps temp+add.
/// Default: **false** — two-seed quality noise + Audit 6 accumulate risk.
pub fn gemm_accum_dx() -> bool {
    flags().gemm_accum_dx
}

/// Per-section **forward** timing (`METAL_NATIVE_FWD_PROFILE=1`). Syncs between
/// sections; diagnostic only, never a speed gate. Default: false.
pub fn fwd_profile() -> bool {
    flags().fwd_profile
}

/// Replace the per-element device atomics in the fused resid/RMS bwd
/// megakernels with a row-block reduction pass (`METAL_NATIVE_GLUE_ROWBLOCK=1`).
/// Cuts atomics ~128x (3.1M -> 24576 per call) at the cost of one extra
/// coalesced read pass. **Not bit-identical** — the summation order changes, so
/// gate at 1e-5. Default: **false** — Audit 8 A/B ≈ noise (REJECT as WIN).
pub fn glue_rowblock() -> bool {
    flags().glue_rowblock
}

/// Per-section **optimizer** timing (`METAL_NATIVE_OPTIM_PROFILE=1`):
/// embed AdamW / scalar AdamW / Muon banks. Diagnostic only. Default: false.
pub fn optim_profile() -> bool {
    flags().optim_profile
}

/// Head-dim-specialized **forward** flash (`flash_attn_fwd_d32_*`): compile-time
/// DH=32, full unroll, tight `[BC*DH]` threadgroup staging instead of `[BC*64]`.
/// Same math as `flash_attn_fwd_f32`. Requires head_dim == 32. Default: **true**
/// (Audit 8 KEEP / Audit 9A). Under `PrecisionMode::Bf16` this also selects the
/// bf16 twin (`use_bf16_flash` remains hard-coded false — see DECISIONS M16).
pub fn fa_fwd_fast() -> bool {
    flags().fa_fwd_fast
}

/// FA-2 blockwise online-softmax forward (`flash_attn_fwd_blocksoft_*`).
/// Soft@3k quality opt-in (`METAL_NATIVE_FA_BLOCKSOFT=1`). Default: false.
/// Not bit-identical to the sequential per-token recurrence.
pub fn fa_blocksoft() -> bool {
    flags().fa_blocksoft
}

/// Head-dim-specialized row FA bwd (`flash_attn_bwd_*_row_d32_f32`).
/// Numerically identical to the generic row kernels — removes loop-invariant
/// device reloads and register spills only. Default: **true** (Audit 7 KEEP).
/// Host must guard `head_dim == 32`; ignored when [`fa_tiled_bwd`] is set.
pub fn fa_fast_row() -> bool {
    flags().fa_fast || flags().fa_bf16
}

/// bf16 Q/K/V in FA bwd; f32 dO/L/Delta/scores/accum/grads. Default: **true**
/// (Audit 7 KEEP; requires `PrecisionMode::Bf16` + head_dim 32 in dispatch).
///
/// Row path (default): `*_row_d32_bf16`, implies [`fa_fast_row`], requires
/// `head_dim == 32`. Tiled path ([`fa_tiled_bwd`]): the Phase 4
/// `flash_attn_bwd_{dq,dkv}_bf16` twins, which are drop-in
/// signature-compatible with their f32 counterparts.
pub fn fa_bf16_row() -> bool {
    flags().fa_bf16
}

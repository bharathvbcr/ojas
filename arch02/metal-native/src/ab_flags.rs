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
//!   TESSL_HAZARD_BARRIERS=1           → skip always-on Device barrier after each
//!                                       dispatch (packed ops keep explicit barriers).
//!                                       A/B REJECT as default: NaNs by ~step 3.
//!                                       Read by tessl's runtime; [`hazard_barriers`]
//!                                       reports it. Default MLP recompute steps
//!                                       aside under it; an explicit
//!                                       MLP_RECOMPUTE=1 is refused.
//!   METAL_NATIVE_MUON_GRAM=0          → standard NS for every Muon bank (default
//!                                       ON: Gram NS on aspect >= 2, DECISIONS M20)
//!   METAL_NATIVE_MLP_RECOMPUTE=0      → keep the per-layer MLP hidden on the tape
//!                                       (default ON: recompute in bwd, M20)
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
//!   METAL_NATIVE_FA_FAST=1           → head-dim-specialized **row** FA bwd
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
//!
//! Retired (2026-10-20): `METAL_NATIVE_GEMM_ACCUM`, `METAL_NATIVE_GEMM_ACCUM_DX`,
//! `METAL_NATIVE_GEMM_INTERIOR` and `METAL_NATIVE_HAZARD_BARRIERS` were read here
//! but nothing consumed them after the import (1fc6a03), so setting them changed
//! nothing. tessl dropped its own accumulate and interior flags on 2026-10-10:
//! training accumulates now always run the TensorOps accumulate kernels
//! (`tessl::gemm::gemm_tn_accum_train`), which is the path Audit 6 measured as
//! the Soft regression suspect. The "accum off" defaults above describe the
//! pre-2026-10-10 code, not what runs now.

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
    resid_bwd_fuse: bool,
    bwd_profile: bool,
    bwd_cast_once: bool,
    fa_fast: bool,
    fa_bf16: bool,
    fwd_profile: bool,
    fa_fwd_fast: bool,
    fa_blocksoft: bool,
    optim_profile: bool,
    glue_rowblock: bool,
    muon_gram: bool,
    mlp_recompute: Option<bool>,
}

fn flags() -> &'static AbFlags {
    static FLAGS: OnceLock<AbFlags> = OnceLock::new();
    FLAGS.get_or_init(|| AbFlags {
        fa_tiled: env_truthy("METAL_NATIVE_FA_TILED").unwrap_or(false),
        muon_sg: env_truthy("METAL_NATIVE_MUON_SG").unwrap_or(true),
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
        fwd_profile: env_truthy("METAL_NATIVE_FWD_PROFILE").unwrap_or(false),
        // Audit 8 KEEP / Audit 9A: default ON (flash 264→39 ms; D=32 gate in dispatch).
        fa_fwd_fast: env_truthy("METAL_NATIVE_FA_FWD_FAST").unwrap_or(true),
        // Phase G Soft quality: FA-2 blockwise softmax. Default off.
        fa_blocksoft: env_truthy("METAL_NATIVE_FA_BLOCKSOFT").unwrap_or(false),
        optim_profile: env_truthy("METAL_NATIVE_OPTIM_PROFILE").unwrap_or(false),
        // Rowblock cut atomics ~128× but A/B ≈ noise (Audit 8 glue sweep) — keep OFF.
        glue_rowblock: env_truthy("METAL_NATIVE_GLUE_ROWBLOCK").unwrap_or(false),
        // Gram Newton-Schulz for Muon banks with aspect ratio >= 2. Same polynomial,
        // different association. Default ON (DECISIONS M20): optimizer -13 to -14%
        // at 128M, trajectory within same-config noise.
        muon_gram: env_truthy("METAL_NATIVE_MUON_GRAM").unwrap_or(true),
        // Recompute the MLP hidden activation in backward instead of keeping it on
        // the tape. Default ON (DECISIONS M20); unset means "on unless the runtime
        // skips its auto barriers", see [`mlp_recompute`].
        mlp_recompute: env_truthy("METAL_NATIVE_MLP_RECOMPUTE"),
    })
}

/// Gram Newton-Schulz Muon banks: iterate the polynomial on the small p x p Gram
/// matrix and apply it to X once, for banks whose long side is at least 2x the
/// short side. Default: **true**; `METAL_NATIVE_MUON_GRAM=0` restores standard NS.
pub fn muon_gram() -> bool {
    flags().muon_gram
}

/// MLP activation recompute: backward rebuilds `act(mlp_pre_act)` instead of
/// reading a per-layer `mlp_hidden`. Default: **true**, except when the runtime
/// skips its auto barriers (`TESSL_HAZARD_BARRIERS=1`), which the shared scratch
/// needs; then the default quietly keeps the taped hidden. An explicit
/// `METAL_NATIVE_MLP_RECOMPUTE=1` under hazard mode is refused at forward time
/// instead. `=0` restores the taped hidden.
pub fn mlp_recompute() -> bool {
    resolve_mlp_recompute(flags().mlp_recompute, hazard_barriers())
}

fn resolve_mlp_recompute(explicit: Option<bool>, skip_auto_barriers: bool) -> bool {
    explicit.unwrap_or(!skip_auto_barriers)
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

/// Whether the runtime skips the always-on Dispatch→Dispatch Device barrier
/// (`TESSL_HAZARD_BARRIERS=1`). arch02's binders pass no override, so tessl's
/// flag is the one that decides. Default: false (golden-safe). **Do not enable
/// as default** — A/B produced NaNs by ~step 3.
pub fn hazard_barriers() -> bool {
    tessl::ab_flags::hazard_barriers()
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

#[cfg(test)]
mod tests {
    use super::resolve_mlp_recompute;

    #[test]
    fn mlp_recompute_default_on_and_yields_to_hazard_mode() {
        assert!(resolve_mlp_recompute(None, false), "default is on");
        assert!(
            !resolve_mlp_recompute(None, true),
            "default steps aside in hazard mode"
        );
        // Explicit choices win; an explicit on under hazard mode is refused later
        // by the forward guard, not silently dropped here.
        assert!(resolve_mlp_recompute(Some(true), true));
        assert!(!resolve_mlp_recompute(Some(false), false));
    }
}

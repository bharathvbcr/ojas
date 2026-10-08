//! Phase 2 Gemma Metal kernels — real dispatch via the tessl Binder.
//!
//! Overlay metallib (`GEMMA_METAL_METALLIB`) is registered on a shared
//! [`GpuRuntime`]. Prefill GEMM uses tessl TensorOps / simdgroup kernels.

use std::cell::RefCell;
use std::path::Path;
use std::sync::atomic::{AtomicI8, Ordering};
use std::sync::{Arc, OnceLock};

use tessl::dispatch::{
    dispatch_1d, dispatch_2d_tg, set_f32, set_gpu_buf, set_gpu_buf_offset, set_u32,
};
// softcap_logits retained for tests / external callers; decode uses fused argmax.
use tessl::gemm::{gemm, select_backend, GemmBackend};
use tessl::runtime::GpuRuntime;
use tessl::tensor::{GpuBuffer, Tensor};

use crate::diag;
use crate::error::{Error, Result};
use crate::quant::QuantMatrix;

/// Absolute metallib path baked by `build.rs` (empty if AOT skipped / missing).
pub fn metallib_path() -> &'static str {
    option_env!("GEMMA_METAL_METALLIB").unwrap_or("")
}

/// Kernel entry points reachable from the shared [`GpuRuntime`].
///
/// Most of these now resolve from **tessl's own metallib**: the kernels were
/// promoted into `tessl/kernels/` and are no longer part of this crate's
/// overlay. `GpuRuntime::pipeline` searches the primary library before any
/// overlay, so the names resolve unchanged. Only the Gemma-specific entries
/// (`ple_*`, `persistent_interp_*`) still come from the overlay compiled by
/// this crate's `build.rs`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelId {
    GemvQ4,
    GemvQ4Mlx,
    GemvQ8,
    FlashAttnSwaH256,
    /// DFlash draft FA (head_dim=128); used by [`crate::dflash::DFlashGpuDraft`].
    FlashAttnSwaH128,
    FlashAttnGlobalH512,
    RmsQkvRope,
    /// Encode-once prototype: pos from device `u32` buffer (see `rms_qkv_rope_posbuf`).
    RmsQkvRopePosbuf,
    /// Layer-fusion: RoPE(+norms) + kv_store into cache slot (producer).
    RmsQkvRopeKvStore,
    RmsNormF32,
    /// Emit bf16 (simd GEMV act scratch) — skips cast_f32_to_bf16.
    RmsNormBf16,
    /// Fused `resid += rms_norm(x) * weight` (Gemma4 dual-norm residual tails).
    RmsNormResidualAddF32,
    PleLookup,
    PleLookupQ4Mlx,
    PleResidualAdd,
    MlpGeluTanh,
    /// `gelu(gate)*up` writing bf16 mid.
    MlpGeluTanhBf16,
    /// DFlash / Qwen3 draft MLP: `silu(gate) * up`.
    MlpSilu,
    SoftcapSample,
    SoftcapLogits,
    SoftcapArgmaxOnePass,
    ArgmaxF32,
    KvStoreTimestep,
    KvStoreTimestepPair,
    KvRingDensify,
    EmbedLookupQ4,
    EmbedLookupQ4Mlx,
    /// In-place `x *= scale` (Gemma4 embed_scale after lookup).
    ScaleF32Inplace,
    /// Layer-fusion v1: fused producer Q∥K∥V simd GEMV (one dispatch).
    GemvQ4MlxSimdQkv,
    /// Interleaved4 twin of [`Self::GemvQ4MlxSimdQkv`].
    GemvQ4MlxSimdQkvI4,
    /// Layer-fusion v1: fused PLE Q4 lookup + residual combine (one dispatch).
    PleLookupQ4MlxResidual,
    /// Thin Q4 GEMM for DFlash verify (M≤8): `Y[M,rows] = X[M,cols] @ W^T`.
    GemmQ4MlxSimd,
    GemmQ4MlxSimdI4,
    GemmQ4MlxSimdAdd,
    GemmQ4MlxSimdAddI4,
    /// Persistent-interpreter prototype: gate→down stand-in (mini only).
    PersistentInterpGateDown,
    /// Persistent-interpreter prototype: FA→o_proj stand-in (mini only).
    PersistentInterpFaOProj,
    /// Hot Q4 gate→down persistent interpreter (bounded TG, E4B opt-in).
    PersistentInterpGateDownQ4,
}

impl KernelId {
    pub fn entry_name(self) -> &'static str {
        match self {
            KernelId::GemvQ4 => "gemv_q4",
            KernelId::GemvQ4Mlx => "gemv_q4_mlx",
            KernelId::GemvQ8 => "gemv_q8",
            KernelId::FlashAttnSwaH256 => "flash_attn_swa_h256",
            KernelId::FlashAttnSwaH128 => "flash_attn_swa_h128",
            KernelId::FlashAttnGlobalH512 => "flash_attn_global_h512",
            KernelId::RmsQkvRope => "rms_qkv_rope",
            KernelId::RmsQkvRopePosbuf => "rms_qkv_rope_posbuf",
            KernelId::RmsQkvRopeKvStore => "rms_qkv_rope_kv_store",
            KernelId::RmsNormF32 => "rms_norm_f32",
            KernelId::RmsNormBf16 => "rms_norm_bf16",
            KernelId::RmsNormResidualAddF32 => "rms_norm_residual_add_f32",
            KernelId::PleLookup => "ple_lookup",
            KernelId::PleLookupQ4Mlx => "ple_lookup_q4_mlx",
            KernelId::PleResidualAdd => "ple_residual_add",
            KernelId::MlpGeluTanh => "mlp_gelu_tanh",
            KernelId::MlpGeluTanhBf16 => "mlp_gelu_tanh_bf16",
            KernelId::MlpSilu => "mlp_silu",
            KernelId::SoftcapSample => "softcap_sample",
            KernelId::SoftcapLogits => "softcap_logits",
            KernelId::SoftcapArgmaxOnePass => "softcap_argmax_one_pass",
            KernelId::ArgmaxF32 => "argmax_f32",
            KernelId::KvStoreTimestep => "kv_store_timestep",
            KernelId::KvStoreTimestepPair => "kv_store_timestep_pair",
            KernelId::KvRingDensify => "kv_ring_densify",
            KernelId::EmbedLookupQ4 => "embed_lookup_q4",
            KernelId::EmbedLookupQ4Mlx => "embed_lookup_q4_mlx",
            KernelId::ScaleF32Inplace => "scale_f32_inplace",
            KernelId::GemvQ4MlxSimdQkv => "gemv_q4_mlx_simd_qkv",
            KernelId::GemvQ4MlxSimdQkvI4 => "gemv_q4_mlx_simd_qkv_i4",
            KernelId::PleLookupQ4MlxResidual => "ple_lookup_q4_mlx_residual",
            KernelId::GemmQ4MlxSimd => "gemm_q4_mlx_simd",
            KernelId::GemmQ4MlxSimdI4 => "gemm_q4_mlx_simd_i4",
            KernelId::GemmQ4MlxSimdAdd => "gemm_q4_mlx_simd_add",
            KernelId::GemmQ4MlxSimdAddI4 => "gemm_q4_mlx_simd_add_i4",
            KernelId::PersistentInterpGateDown => "persistent_interp_gate_down",
            KernelId::PersistentInterpFaOProj => "persistent_interp_fa_o_proj",
            KernelId::PersistentInterpGateDownQ4 => "persistent_interp_gate_down_q4",
        }
    }

    pub fn all() -> &'static [KernelId] {
        &[
            KernelId::GemvQ4,
            KernelId::GemvQ4Mlx,
            KernelId::GemvQ8,
            KernelId::FlashAttnSwaH256,
            KernelId::FlashAttnSwaH128,
            KernelId::FlashAttnGlobalH512,
            KernelId::RmsQkvRope,
            KernelId::RmsQkvRopePosbuf,
            KernelId::RmsQkvRopeKvStore,
            KernelId::RmsNormF32,
            KernelId::RmsNormBf16,
            KernelId::RmsNormResidualAddF32,
            KernelId::PleLookup,
            KernelId::PleLookupQ4Mlx,
            KernelId::PleResidualAdd,
            KernelId::MlpGeluTanh,
            KernelId::MlpGeluTanhBf16,
            KernelId::MlpSilu,
            KernelId::SoftcapSample,
            KernelId::SoftcapLogits,
            KernelId::SoftcapArgmaxOnePass,
            KernelId::ArgmaxF32,
            KernelId::KvStoreTimestep,
            KernelId::KvStoreTimestepPair,
            KernelId::KvRingDensify,
            KernelId::EmbedLookupQ4,
            KernelId::EmbedLookupQ4Mlx,
            KernelId::ScaleF32Inplace,
            KernelId::GemvQ4MlxSimdQkv,
            KernelId::GemvQ4MlxSimdQkvI4,
            KernelId::PleLookupQ4MlxResidual,
            KernelId::GemmQ4MlxSimd,
            KernelId::GemmQ4MlxSimdI4,
            KernelId::GemmQ4MlxSimdAdd,
            KernelId::GemmQ4MlxSimdAddI4,
            KernelId::PersistentInterpGateDown,
            KernelId::PersistentInterpFaOProj,
        ]
    }
}

/// Instruction opcodes for [`persistent_interp_gate_down`] (must match Metal).
pub const PI_OP_HALT: u32 = 0;
pub const PI_OP_PRODUCE_MID: u32 = 1;
pub const PI_OP_BARRIER: u32 = 2;
pub const PI_OP_DOWN_PROJ: u32 = 3;

/// Canonical mini gate→down program: produce → grid barrier → down → halt.
pub fn persistent_interp_gate_down_program() -> [u32; 4] {
    [
        PI_OP_PRODUCE_MID,
        PI_OP_BARRIER,
        PI_OP_DOWN_PROJ,
        PI_OP_HALT,
    ]
}

/// Instruction opcodes for [`persistent_interp_fa_o_proj`] (must match Metal).
pub const PI_FA_OP_HALT: u32 = 0;
pub const PI_FA_OP_PRODUCE_CTX: u32 = 1;
pub const PI_FA_OP_BARRIER: u32 = 2;
pub const PI_FA_OP_O_PROJ: u32 = 3;

/// Canonical mini FA→o_proj program: produce ctx → grid barrier → o_proj → halt.
pub fn persistent_interp_fa_o_proj_program() -> [u32; 4] {
    [
        PI_FA_OP_PRODUCE_CTX,
        PI_FA_OP_BARRIER,
        PI_FA_OP_O_PROJ,
        PI_FA_OP_HALT,
    ]
}

/// Legacy tiled TG size (parity / optional `*_tiled` entries).
pub const GEMV_TG: usize = 128;

/// Blocked Hot Q4 GEMV row tile (`gemv_q4_mlx_blocked`). Must match Metal `GEMV_BN`.
pub const GEMV_BN: u32 = 16;
/// K-lanes per blocked row (`GEMV_LANES` in Metal). TG = BN × LANES = 256.
pub const GEMV_LANES: u32 = 16;
/// Must match Metal `GEMV_X_TILE` (TG x-cache cap; + static partials ≤32 KiB).
pub const GEMV_X_TILE: usize = 4096;
/// Must match Metal `SIMD_ROWS` / `SIMD_SG_PER_TG` in `gemv_q4_mlx_simd*`.
const GEMV_SIMD_ROWS: u32 = 4;
const GEMV_SIMD_SG: u32 = 2;
/// Threads per TG for simd GEMV (2 simdgroups × 32 = 64 → 8 rows/TG).
const GEMV_SIMD_TPTG: usize = 64;

/// Hot weight memory layout for decode GEMV.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HotGemvLayout {
    /// MLX / host row-major: `[row][cols/2]` packed, `[row][groups]` scales.
    RowMajor,
    /// Coalesced tile: `[row_block][group][Bn][group_bytes]` (+ matching scales).
    BlockedBn16,
    /// Simd-friendly: `[tile][uint2_pack][r0..r3]` packs + `[tile][g][r0..r3]` scales.
    Interleaved4,
}

fn gemv_blocked_enabled() -> bool {
    // Default OFF: row-major / Interleaved4 simd coalesces better than BlockedBn
    // (blocked left for A/B via GEMMA_METAL_GEMV_BLOCKED=1).
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(
        || match std::env::var("GEMMA_METAL_GEMV_BLOCKED").ok().as_deref() {
            Some("1") | Some("true") | Some("on") => true,
            _ => false,
        },
    )
}

fn gemv_interleave_enabled() -> bool {
    // Default OFF: Interleaved4 `_simd_i4` measured ~22.8 vs row-major ~23.8 E4B (bfloat2+qdot).
    // Kernels + upload path remain; opt in with `GEMMA_METAL_GEMV_INTERLEAVE=1`.
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(
        || match std::env::var("GEMMA_METAL_GEMV_INTERLEAVE").ok().as_deref() {
            Some("1") | Some("true") | Some("on") => true,
            _ => false,
        },
    )
}

fn trace_gemv_enabled() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("GEMMA_METAL_TRACE_GEMV").ok().as_deref() == Some("1"))
}

/// Prefer blocked Hot for mid-size Q4 mats; keep vocab/lm_head row-major (tied embed).
fn prefer_blocked_q4_mlx(rows: usize, cols: usize, group_size: usize) -> bool {
    gemv_blocked_enabled()
        && rows >= GEMV_BN as usize
        && rows < 65_536
        && group_size > 0
        && cols % group_size == 0
        && group_size % 8 == 0
}

/// Prefer Interleaved4 Hot for mid simd path; keep vocab/lm_head row-major (tied embed).
fn prefer_interleaved_q4_mlx(rows: usize, cols: usize, group_size: usize) -> bool {
    gemv_interleave_enabled()
        && !gemv_blocked_enabled()
        && rows >= GEMV_SIMD_ROWS as usize
        && rows < 65_536
        && cols >= 256
        && cols % 16 == 0
        && group_size > 0
        && cols % group_size == 0
        // qmv_fast K-block is 512; pointer-walk sb step needs exact divide.
        && 512 % group_size == 0
}

/// Repack row-major MLX Q4 → `BlockedBn16` (pads last block with zeros).
fn repack_q4_mlx_blocked(
    packed: &[u8],
    scales: &[f32],
    biases: &[f32],
    rows: usize,
    cols: usize,
    group_size: usize,
) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let bn = GEMV_BN as usize;
    let groups = cols / group_size;
    let bytes_per_group = group_size / 2;
    let n_blocks = (rows + bn - 1) / bn;
    let mut out_p = vec![0u8; n_blocks * groups * bn * bytes_per_group];
    let mut out_s = vec![0f32; n_blocks * groups * bn];
    let mut out_b = vec![0f32; n_blocks * groups * bn];
    for rb in 0..n_blocks {
        for g in 0..groups {
            for r in 0..bn {
                let row = rb * bn + r;
                let dst_p = rb * groups * bn * bytes_per_group
                    + g * bn * bytes_per_group
                    + r * bytes_per_group;
                let dst_s = rb * groups * bn + g * bn + r;
                if row >= rows {
                    continue;
                }
                let src_p = row * (cols / 2) + g * bytes_per_group;
                out_p[dst_p..dst_p + bytes_per_group]
                    .copy_from_slice(&packed[src_p..src_p + bytes_per_group]);
                out_s[dst_s] = scales[row * groups + g];
                out_b[dst_s] = biases[row * groups + g];
            }
        }
    }
    (out_p, out_s, out_b)
}

/// Repack row-major MLX Q4 → `Interleaved4` (pads last tile with zeros).
/// Layout: for each tile of 4 rows, for each uint2 along K, store `[r0,r1,r2,r3]`
/// contiguously; scales/biases as `[tile][group][r0..r3]`.
fn repack_q4_mlx_interleaved4(
    packed: &[u8],
    scales: &[f32],
    biases: &[f32],
    rows: usize,
    cols: usize,
    group_size: usize,
) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let rn = GEMV_SIMD_ROWS as usize;
    let groups = cols / group_size;
    let row_bytes = cols / 2;
    let packs_u2 = cols / 16;
    let n_tiles = (rows + rn - 1) / rn;
    let mut out_p = vec![0u8; n_tiles * packs_u2 * rn * 8];
    let mut out_s = vec![0f32; n_tiles * groups * rn];
    let mut out_b = vec![0f32; n_tiles * groups * rn];
    for tile in 0..n_tiles {
        for pack2 in 0..packs_u2 {
            for r in 0..rn {
                let row = tile * rn + r;
                let dst = ((tile * packs_u2 + pack2) * rn + r) * 8;
                if row >= rows {
                    continue;
                }
                let src = row * row_bytes + pack2 * 8;
                out_p[dst..dst + 8].copy_from_slice(&packed[src..src + 8]);
            }
        }
        for g in 0..groups {
            for r in 0..rn {
                let row = tile * rn + r;
                let dst = (tile * groups + g) * rn + r;
                if row >= rows {
                    continue;
                }
                out_s[dst] = scales[row * groups + g];
                out_b[dst] = biases[row * groups + g];
            }
        }
    }
    (out_p, out_s, out_b)
}

/// Pack parallel f32 scale/bias → interleaved bf16 pairs (`bfloat2` Hot bank).
fn pack_mlx_sb_bf16(scales: &[f32], biases: &[f32]) -> Vec<u16> {
    debug_assert_eq!(scales.len(), biases.len());
    let mut out = Vec::with_capacity(scales.len().saturating_mul(2));
    for (&s, &b) in scales.iter().zip(biases.iter()) {
        out.push(crate::quant::f32_to_bf16_bits(s));
        out.push(crate::quant::f32_to_bf16_bits(b));
    }
    out
}

/// Shared runtime with this crate's overlay metallib loaded.
///
/// The overlay now carries only `ple_lookup*` and `persistent_interp*`; the
/// general kernels live in tessl's primary metallib.
///
/// The stable scalar pool is an internal implementation detail: safe callers
/// cannot reset or overwrite dimensions after a wrapper validates them.
///
/// ```compile_fail
/// fn overwrite_validated_scalar(gpu: &gemma_metal::GemmaGpu) {
///     gpu.icb_scalars.reset_step();
/// }
/// ```
pub struct GemmaGpu {
    pub rt: Arc<GpuRuntime>,
    /// Reused Cold scratch for f32→bf16 activation casts feeding simd GEMV.
    act_bf16: std::sync::Mutex<Option<(GpuBuffer, usize)>>,
    /// Reused Cold scratch for bf16→f32 casts feeding classic `gemv_q4` (mini Q4
    /// banks: fuse_bf16 producers must not be read as float without this expand).
    act_f32: std::sync::Mutex<Option<(GpuBuffer, usize)>>,
    /// Stable scalar pool for ICB / encode-once binds (FA/kv/softcap). Fixed
    /// GPU addresses — contents rewritten per dispatch/step; not const-arena.
    pub(crate) icb_scalars: IcbScalarPool,
}

/// Stable `GpuBuffer` arena for per-token / per-dispatch scalars (A0 / D16).
///
/// Replaces ephemeral const-arena `set_u32`/`set_f32` for FA, kv_store, softcap,
/// GEMV/MLP/PLE/RMS dims (bind-tax cut). Host bump-allocates unique slots within
/// Hot buffers so packed async encodes keep distinct values; reset cursors at
/// step start. When the dispatch sequence is deterministic, the same logical
/// binds land at the same offsets every token → ICB-friendly stable GPU addresses.
/// Call `push_*` *before* `with_binder` so binder-nop replay still refreshes slots.
///
/// Cursors are atomics (not Mutex): decode is single-threaded per session; A2
/// cuts ~hundreds of lock/unlock pairs per binder-nop prep step.
pub(crate) struct IcbScalarPool {
    /// Softcap f32×1 (model-constant; rewritten per softcap dispatch).
    softcap: GpuBuffer,
    /// Hot u32 workspace (bump-allocated per bind within a step).
    u32s: GpuBuffer,
    /// Hot f32 workspace (FA scale + other per-dispatch floats).
    f32s: GpuBuffer,
    cursor_u32: std::sync::atomic::AtomicUsize,
    cursor_f32: std::sync::atomic::AtomicUsize,
}

impl IcbScalarPool {
    /// ~4k u32s — enough for E4B decode scalars with headroom.
    const U32_SLOTS: usize = 4096;
    /// ~1k f32s — FA scale + misc floats per step.
    const F32_SLOTS: usize = 1024;

    fn new(rt: &GpuRuntime) -> Result<Self> {
        let softcap = rt.alloc_buffer_hot(4).map_err(map_metal)?;
        softcap.write_f32(&[30.0]);
        let u32s = rt
            .alloc_buffer_hot(Self::U32_SLOTS * 4)
            .map_err(map_metal)?;
        let f32s = rt
            .alloc_buffer_hot(Self::F32_SLOTS * 4)
            .map_err(map_metal)?;
        // Zero once; subsequent pushes overwrite.
        for s in u32s.contents_u32().iter_mut() {
            *s = 0;
        }
        for s in f32s.contents_f32().iter_mut() {
            *s = 0.0;
        }
        Ok(Self {
            softcap,
            u32s,
            f32s,
            cursor_u32: std::sync::atomic::AtomicUsize::new(0),
            cursor_f32: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    pub(crate) fn set_softcap(&self, v: f32) {
        self.softcap.write_f32(&[v]);
    }

    /// Reset bump cursors (call once per decode/verify step before layer loop).
    pub(crate) fn reset_step(&self) {
        self.cursor_u32.store(0, Ordering::Relaxed);
        self.cursor_f32.store(0, Ordering::Relaxed);
    }

    /// Current bump watermarks (debug / ICB offset parity).
    pub(crate) fn cursor_snapshot(&self) -> (usize, usize) {
        (
            self.cursor_u32.load(Ordering::Relaxed),
            self.cursor_f32.load(Ordering::Relaxed),
        )
    }

    /// Force cursors (scalar-write tape apply / watermark restore).
    pub(crate) fn set_cursors(&self, u32_n: usize, f32_n: usize) {
        self.cursor_u32.store(u32_n, Ordering::Relaxed);
        self.cursor_f32.store(f32_n, Ordering::Relaxed);
    }

    /// Push a u32; returns byte offset into [`Self::u32s`] for `set_gpu_buf_offset`.
    pub(crate) fn push_u32(&self, v: u32) -> Result<usize> {
        let slot = self.cursor_u32.fetch_add(1, Ordering::Relaxed);
        if slot >= Self::U32_SLOTS {
            // Keep cursor from walking forever on repeated OOB (best-effort).
            self.cursor_u32.store(Self::U32_SLOTS, Ordering::Relaxed);
            return Err(Error::Metal(format!(
                "icb_scalars u32 arena exhausted (cap {})",
                Self::U32_SLOTS
            )));
        }
        self.u32s.contents_u32()[slot] = v;
        icb_scalar_tape_record_u32_const(v);
        Ok(slot * 4)
    }

    /// Push a u32 that must be recomputed on DecodeIcb skip-nop replay.
    pub(crate) fn push_u32_dyn(&self, v: u32, src: IcbDynSrc) -> Result<usize> {
        let slot = self.cursor_u32.fetch_add(1, Ordering::Relaxed);
        if slot >= Self::U32_SLOTS {
            self.cursor_u32.store(Self::U32_SLOTS, Ordering::Relaxed);
            return Err(Error::Metal(format!(
                "icb_scalars u32 arena exhausted (cap {})",
                Self::U32_SLOTS
            )));
        }
        self.u32s.contents_u32()[slot] = v;
        icb_scalar_tape_record_u32_dyn(src);
        Ok(slot * 4)
    }

    /// Push an f32; returns byte offset into [`Self::f32s`] for `set_gpu_buf_offset`.
    pub(crate) fn push_f32(&self, v: f32) -> Result<usize> {
        let slot = self.cursor_f32.fetch_add(1, Ordering::Relaxed);
        if slot >= Self::F32_SLOTS {
            self.cursor_f32.store(Self::F32_SLOTS, Ordering::Relaxed);
            return Err(Error::Metal(format!(
                "icb_scalars f32 arena exhausted (cap {})",
                Self::F32_SLOTS
            )));
        }
        self.f32s.contents_f32()[slot] = v;
        icb_scalar_tape_record_f32_const(v);
        Ok(slot * 4)
    }

    #[inline]
    fn bind_u32(&self, bnd: &mut tessl::dispatch::Binder<'_>, off: usize, index: usize) {
        set_gpu_buf_offset(bnd, &self.u32s, off, index);
    }

    #[inline]
    fn bind_f32(&self, bnd: &mut tessl::dispatch::Binder<'_>, off: usize, index: usize) {
        set_gpu_buf_offset(bnd, &self.f32s, off, index);
    }
}

/// Push `rows/cols/group_size` before `with_binder` (binder-nop must still refresh).
#[inline]
fn push_gemv_dims(
    gpu: &GemmaGpu,
    rows: u32,
    cols: u32,
    group_size: u32,
) -> Result<(usize, usize, usize)> {
    Ok((
        gpu.icb_scalars.push_u32(rows)?,
        gpu.icb_scalars.push_u32(cols)?,
        gpu.icb_scalars.push_u32(group_size)?,
    ))
}

// --- DecodeIcb scalar-write tape (A2 residual: skip nop layer-loop) ----------

/// Dynamic u32 sources recomputed on skip-nop replay (stable cursor shape).
#[derive(Clone, Copy, Debug)]
pub enum IcbDynSrc {
    /// Decode position (`pos as u32`).
    Pos,
    /// `peek_write_offset` before commit for a sliding ring.
    SlidingPeek(usize),
    /// `peek_write_offset` before commit for a global slot.
    GlobalPeek(usize),
    SharedSlidingPeek,
    SharedGlobalPeek,
    /// Densify/FA filled after the slot's commit: `min(seq, cap, pos+1)`.
    SlidingFilled(usize),
    SlidingStart(usize),
    SlidingKvPos(usize),
    SlidingTkv(usize),
    GlobalFilled(usize),
    GlobalTkv(usize),
    SharedSlidingFilled,
    SharedSlidingStart,
    SharedSlidingKvPos,
    SharedSlidingTkv,
    SharedGlobalFilled,
    SharedGlobalTkv,
}

/// Host KV metadata advance recorded between scalar pushes (order matters).
#[derive(Clone, Copy, Debug)]
pub enum IcbKvHostOp {
    CommitSliding(usize),
    CommitGlobal(usize),
    CommitSharedSliding,
    CommitSharedGlobal,
}

/// One host op in the scalar-write tape (push or KV commit).
#[derive(Clone, Debug)]
pub enum IcbScalarTapeOp {
    U32Const(u32),
    U32Dyn(IcbDynSrc),
    F32Const(f32),
    Kv(IcbKvHostOp),
}

/// Captured push program + KV commits for mini DecodeIcb skip-nop replay.
#[derive(Clone, Debug, Default)]
pub struct IcbScalarWriteTape {
    pub ops: Vec<IcbScalarTapeOp>,
}

impl IcbScalarWriteTape {
    pub fn op_count(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

thread_local! {
    static ICB_SCALAR_TAPE: RefCell<Option<Vec<IcbScalarTapeOp>>> =
        const { RefCell::new(None) };
    static ICB_TAPE_KV_CTX: RefCell<Option<IcbTapeKvCtx>> = const { RefCell::new(None) };
}

#[derive(Clone, Copy, Debug)]
enum IcbTapeKvCtx {
    Sliding(usize),
    Global(usize),
    SharedSliding,
    SharedGlobal,
}

/// Begin recording scalar pushes / KV commits (Binder layer-graph capture).
pub fn begin_icb_scalar_write_tape() {
    ICB_SCALAR_TAPE.with(|t| *t.borrow_mut() = Some(Vec::with_capacity(512)));
    ICB_TAPE_KV_CTX.with(|c| *c.borrow_mut() = None);
}

pub fn icb_scalar_write_tape_active() -> bool {
    ICB_SCALAR_TAPE.with(|t| t.borrow().is_some())
}

pub fn take_icb_scalar_write_tape() -> Option<IcbScalarWriteTape> {
    ICB_TAPE_KV_CTX.with(|c| *c.borrow_mut() = None);
    ICB_SCALAR_TAPE.with(|t| t.borrow_mut().take().map(|ops| IcbScalarWriteTape { ops }))
}

pub fn icb_tape_set_kv_ctx_sliding(index: usize) {
    if icb_scalar_write_tape_active() {
        ICB_TAPE_KV_CTX.with(|c| *c.borrow_mut() = Some(IcbTapeKvCtx::Sliding(index)));
    }
}

pub fn icb_tape_set_kv_ctx_global(index: usize) {
    if icb_scalar_write_tape_active() {
        ICB_TAPE_KV_CTX.with(|c| *c.borrow_mut() = Some(IcbTapeKvCtx::Global(index)));
    }
}

pub fn icb_tape_set_kv_ctx_shared_sliding() {
    if icb_scalar_write_tape_active() {
        ICB_TAPE_KV_CTX.with(|c| *c.borrow_mut() = Some(IcbTapeKvCtx::SharedSliding));
    }
}

pub fn icb_tape_set_kv_ctx_shared_global() {
    if icb_scalar_write_tape_active() {
        ICB_TAPE_KV_CTX.with(|c| *c.borrow_mut() = Some(IcbTapeKvCtx::SharedGlobal));
    }
}

pub fn icb_tape_clear_kv_ctx() {
    if icb_scalar_write_tape_active() {
        ICB_TAPE_KV_CTX.with(|c| *c.borrow_mut() = None);
    }
}

pub fn icb_tape_note_commit_sliding(index: usize) {
    icb_scalar_tape_record_kv(IcbKvHostOp::CommitSliding(index));
}

pub fn icb_tape_note_commit_global(index: usize) {
    icb_scalar_tape_record_kv(IcbKvHostOp::CommitGlobal(index));
}

pub fn icb_tape_note_commit_shared_sliding() {
    icb_scalar_tape_record_kv(IcbKvHostOp::CommitSharedSliding);
}

pub fn icb_tape_note_commit_shared_global() {
    icb_scalar_tape_record_kv(IcbKvHostOp::CommitSharedGlobal);
}

#[inline]
fn icb_scalar_tape_record_u32_const(v: u32) {
    ICB_SCALAR_TAPE.with(|t| {
        if let Some(ref mut ops) = *t.borrow_mut() {
            ops.push(IcbScalarTapeOp::U32Const(v));
        }
    });
}

#[inline]
fn icb_scalar_tape_record_u32_dyn(src: IcbDynSrc) {
    ICB_SCALAR_TAPE.with(|t| {
        if let Some(ref mut ops) = *t.borrow_mut() {
            ops.push(IcbScalarTapeOp::U32Dyn(src));
        }
    });
}

#[inline]
fn icb_scalar_tape_record_f32_const(v: f32) {
    ICB_SCALAR_TAPE.with(|t| {
        if let Some(ref mut ops) = *t.borrow_mut() {
            ops.push(IcbScalarTapeOp::F32Const(v));
        }
    });
}

#[inline]
fn icb_scalar_tape_record_kv(op: IcbKvHostOp) {
    ICB_SCALAR_TAPE.with(|t| {
        if let Some(ref mut ops) = *t.borrow_mut() {
            ops.push(IcbScalarTapeOp::Kv(op));
        }
    });
}

fn icb_tape_kv_ctx() -> Option<IcbTapeKvCtx> {
    ICB_TAPE_KV_CTX.with(|c| *c.borrow())
}

fn dyn_src_peek_from_ctx() -> Option<IcbDynSrc> {
    match icb_tape_kv_ctx()? {
        IcbTapeKvCtx::Sliding(i) => Some(IcbDynSrc::SlidingPeek(i)),
        IcbTapeKvCtx::Global(i) => Some(IcbDynSrc::GlobalPeek(i)),
        IcbTapeKvCtx::SharedSliding => Some(IcbDynSrc::SharedSlidingPeek),
        IcbTapeKvCtx::SharedGlobal => Some(IcbDynSrc::SharedGlobalPeek),
    }
}

fn dyn_src_filled_from_ctx() -> Option<IcbDynSrc> {
    match icb_tape_kv_ctx()? {
        IcbTapeKvCtx::Sliding(i) => Some(IcbDynSrc::SlidingFilled(i)),
        IcbTapeKvCtx::Global(i) => Some(IcbDynSrc::GlobalFilled(i)),
        IcbTapeKvCtx::SharedSliding => Some(IcbDynSrc::SharedSlidingFilled),
        IcbTapeKvCtx::SharedGlobal => Some(IcbDynSrc::SharedGlobalFilled),
    }
}

fn dyn_src_start_from_ctx() -> Option<IcbDynSrc> {
    match icb_tape_kv_ctx()? {
        IcbTapeKvCtx::Sliding(i) => Some(IcbDynSrc::SlidingStart(i)),
        IcbTapeKvCtx::Global(_) => None, // non-ring: no densify start push
        IcbTapeKvCtx::SharedSliding => Some(IcbDynSrc::SharedSlidingStart),
        IcbTapeKvCtx::SharedGlobal => None,
    }
}

fn dyn_src_tkv_from_ctx() -> Option<IcbDynSrc> {
    match icb_tape_kv_ctx()? {
        IcbTapeKvCtx::Sliding(i) => Some(IcbDynSrc::SlidingTkv(i)),
        IcbTapeKvCtx::Global(i) => Some(IcbDynSrc::GlobalTkv(i)),
        IcbTapeKvCtx::SharedSliding => Some(IcbDynSrc::SharedSlidingTkv),
        IcbTapeKvCtx::SharedGlobal => Some(IcbDynSrc::SharedGlobalTkv),
    }
}

fn dyn_src_kv_pos_from_ctx() -> Option<IcbDynSrc> {
    match icb_tape_kv_ctx()? {
        IcbTapeKvCtx::Sliding(i) => Some(IcbDynSrc::SlidingKvPos(i)),
        IcbTapeKvCtx::Global(_) => None, // always 0 — record as const at call site
        IcbTapeKvCtx::SharedSliding => Some(IcbDynSrc::SharedSlidingKvPos),
        IcbTapeKvCtx::SharedGlobal => None,
    }
}

/// Opt-out: `GEMMA_METAL_ICB_SKIP_NOP_LOOP=0` keeps binder-nop layer loop.
pub fn icb_skip_nop_loop_enabled() -> bool {
    match std::env::var("GEMMA_METAL_ICB_SKIP_NOP_LOOP") {
        Ok(v) => !matches!(v.as_str(), "0" | "false" | "off" | "nop"),
        Err(_) => true,
    }
}

impl GemmaGpu {
    pub fn new() -> Result<Self> {
        let path = metallib_path();
        diag::log(
            "kernels",
            format_args!("GemmaGpu::new metallib_path={path:?}"),
        );
        if path.is_empty() || !Path::new(path).exists() {
            let e = Error::Metal(
                "GEMMA_METAL_METALLIB empty or missing — build the overlay, or set \
                 GEMMA_METAL_SKIP_AOT with a valid GEMMA_METAL_PREBUILT_METALLIB"
                    .into(),
            );
            diag::err("kernels", "metallib missing", &e);
            return Err(e);
        }
        let meta = std::fs::metadata(path).ok();
        diag::log(
            "kernels",
            format_args!(
                "metallib exists size={}",
                meta.map(|m| diag::fmt_bytes(m.len()))
                    .unwrap_or_else(|| "?".into())
            ),
        );
        // Inference: no CounterHeap timestamps on the critical path.
        let rt = GpuRuntime::new_inference().map_err(|e| {
            diag::err_msg("kernels", "GpuRuntime::new_inference", &e);
            Error::Metal(e)
        })?;
        rt.add_metallib(Path::new(path)).map_err(|e| {
            diag::err_msg("kernels", "add_metallib", &e);
            Error::Metal(e)
        })?;
        // Packed Metal 4 encode: batch dispatches until synchronize (Phase 4 speed).
        rt.set_async_encode(true).map_err(|e| {
            diag::err_msg("kernels", "set_async_encode", &e);
            Error::Metal(e)
        })?;
        // Hazard mode (skip always-on Dispatch barriers; explicit RAW at phase edges).
        // Default ON for decode throughput when unset. Do NOT clobber a caller who
        // already called `set_hazard_barriers` (e.g. golden always-on / CaptureAlwaysOnGuard
        // setup) — overwriting forced 31B free-decode → 236773 while capture-on got 531.
        // Set METAL_RUNTIME_HAZARD_BARRIERS=0 to force golden always-on Device barriers.
        if !tessl::ab_flags::hazard_barriers_explicitly_set() {
            let skip_auto = match std::env::var("METAL_RUNTIME_HAZARD_BARRIERS") {
                Ok(v) if matches!(v.as_str(), "0" | "false" | "FALSE" | "no" | "off") => false,
                _ => true,
            };
            tessl::ab_flags::set_hazard_barriers(skip_auto);
        }
        let skip_auto = tessl::ab_flags::hazard_barriers();
        let ws = rt.memory_info().recommended_working_set;
        diag::log(
            "kernels",
            format_args!(
                "GemmaGpu ready async_encode=true hazard_barriers_skip_auto={skip_auto} recommendedWS≈{}",
                diag::fmt_bytes(ws as u64)
            ),
        );
        // Lazily resolve pipelines on first use — eager touch of every entry can
        // XPC-interrupt when another Metal client (e.g. training) saturates the GPU.
        let icb_scalars = IcbScalarPool::new(&rt)?;
        Ok(Self {
            rt,
            act_bf16: std::sync::Mutex::new(None),
            act_f32: std::sync::Mutex::new(None),
            icb_scalars,
        })
    }

    /// Grow-once bf16 activation scratch (≥ `n` elements).
    pub fn act_bf16_scratch(&self, n: usize) -> Result<GpuBuffer> {
        let need = n.max(1);
        let mut slot = self
            .act_bf16
            .lock()
            .map_err(|_| Error::Metal("act_bf16 scratch lock poisoned".into()))?;
        let realloc = match slot.as_ref() {
            Some((_, cap)) if *cap >= need => false,
            _ => true,
        };
        if realloc {
            let buf = self.rt.alloc_buffer(need * 2).map_err(map_metal)?;
            *slot = Some((buf, need));
        }
        Ok(slot.as_ref().unwrap().0.clone())
    }

    /// Grow-once f32 activation scratch (≥ `n` elements) for bf16→f32 expand.
    pub fn act_f32_scratch(&self, n: usize) -> Result<GpuBuffer> {
        let need = n.max(1);
        let mut slot = self
            .act_f32
            .lock()
            .map_err(|_| Error::Metal("act_f32 scratch lock poisoned".into()))?;
        let realloc = match slot.as_ref() {
            Some((_, cap)) if *cap >= need => false,
            _ => true,
        };
        if realloc {
            let buf = self.rt.alloc_buffer(need * 4).map_err(map_metal)?;
            *slot = Some((buf, need));
        }
        Ok(slot.as_ref().unwrap().0.clone())
    }

    pub fn synchronize(&self) -> Result<()> {
        let stall = diag::infer_enabled();
        if stall {
            diag::infer_stall("GemmaGpu::synchronize");
        }
        let _scope = if stall {
            Some(diag::InferScope::begin(
                "cpu_sync",
                "GemmaGpu::synchronize — potential stall",
            ))
        } else {
            None
        };
        let t0 = std::time::Instant::now();
        self.rt.synchronize().map_err(|e| {
            diag::err_msg("kernels", "synchronize", &e);
            Error::Metal(e)
        })?;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        if ms >= 5.0 || stall {
            diag::log("kernels", format_args!("synchronize took {ms:.1} ms"));
        }
        Ok(())
    }

    /// Explicit Dispatch→Dispatch Device barrier (inference hazard mode).
    pub fn barrier(&self) -> Result<()> {
        self.rt
            .with_binder(|bnd| {
                bnd.barrier();
                Ok(())
            })
            .map_err(Error::Metal)
    }
}

fn map_metal(e: String) -> Error {
    Error::Metal(e)
}

fn validate_f32_range(buffer: &GpuBuffer, offset: u32, len: u32, what: &str) -> Result<usize> {
    let start = offset as usize;
    let end = start
        .checked_add(len as usize)
        .ok_or_else(|| Error::Metal(format!("{what}: element range overflows usize")))?;
    let have = buffer.nbytes() / std::mem::size_of::<f32>();
    if end > have {
        return Err(Error::Metal(format!(
            "{what}: range {start}..{end} exceeds buffer capacity {have}"
        )));
    }
    start
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| Error::Metal(format!("{what}: byte offset overflows usize")))
}

fn validate_kv_capacity(dst: &GpuBuffer, n: u32, dst_capacity: u32, what: &str) -> Result<()> {
    if dst_capacity < n {
        return Err(Error::Metal(format!(
            "{what}: dst_capacity {dst_capacity} is smaller than n {n}"
        )));
    }
    let have = dst.nbytes() / std::mem::size_of::<f32>();
    if dst_capacity as usize > have {
        return Err(Error::Metal(format!(
            "{what}: declared capacity {dst_capacity} exceeds buffer capacity {have}"
        )));
    }
    Ok(())
}

/// One TG per `GEMV_BN` output rows; `GEMV_LANES` K-lanes/row (simd_sum).
fn dispatch_gemv_blocked(
    gpu: &GemmaGpu,
    packed: &GpuBuffer,
    scales: &GpuBuffer,
    zeros: &GpuBuffer,
    x: &GpuBuffer,
    y: &GpuBuffer,
    rows: u32,
    cols: u32,
    group_size: u32,
) -> Result<()> {
    let shape = tessl::nn::QuantShape {
        rows,
        cols,
        group_size,
    };
    let bank = tessl::nn::Q4MlxBank {
        packed,
        scales_biases: scales,
    };
    tessl::nn::validate_gemv_q4_mlx_blocked(&gpu.rt, bank, x, y, shape)
        .map_err(map_metal)?;
    if rows == 0 {
        return Ok(());
    }
    let entry = "gemv_q4_mlx_blocked";
    let tptg = (GEMV_BN * GEMV_LANES) as usize;
    let n_tg = ((rows as usize) + GEMV_BN as usize - 1) / GEMV_BN as usize;
    let tg_mem = (cols as usize).min(GEMV_X_TILE) * 4;
    if trace_gemv_enabled() {
        eprintln!(
            "[trace] gemv entry={entry} rows={rows} cols={cols} tg={tptg} groups={n_tg} tg_mem={tg_mem} layout=BlockedBn16 coop"
        );
    }
    let (rows_off, cols_off, gs_off) = push_gemv_dims(gpu, rows, cols, group_size)?;
    // The legacy `zeros` argument is an unused shader slot for MLX banks.
    let _ = zeros;
    // SAFETY: preflight validated the exact shape and buffers before scalar
    // reservation. The closure binds only those values from the stable pool.
    unsafe {
        tessl::nn::gemv_q4_mlx_blocked_with_scalars(
            &gpu.rt,
            bank,
            x,
            y,
            shape,
            |bnd| {
                gpu.icb_scalars.bind_u32(bnd, rows_off, 5);
                gpu.icb_scalars.bind_u32(bnd, cols_off, 6);
                gpu.icb_scalars.bind_u32(bnd, gs_off, 7);
            },
        )
    }
    .map_err(map_metal)
}

/// Decode GEMV: `y[rows] = W_q[rows, cols] @ x[cols]`.
pub fn gemv_q4(
    gpu: &GemmaGpu,
    packed: &GpuBuffer,
    scales: &GpuBuffer,
    zeros: &GpuBuffer,
    x: &GpuBuffer,
    y: &GpuBuffer,
    rows: u32,
    cols: u32,
    group_size: u32,
) -> Result<()> {
    let shape = tessl::nn::QuantShape {
        rows,
        cols,
        group_size,
    };
    let bank = tessl::nn::Q4Bank {
        packed,
        scales,
        zeros,
    };
    tessl::nn::validate_gemv_q4(&gpu.rt, bank, x, y, shape, false).map_err(map_metal)?;
    if rows == 0 {
        return Ok(());
    }
    let (rows_off, cols_off, gs_off) = push_gemv_dims(gpu, rows, cols, group_size)?;
    // SAFETY: the preflight above validates the exact host dimensions and
    // buffers. This closure binds only those dimensions from Gemma's stable,
    // runtime-owned scalar pool, which remains resident across ICB replays.
    unsafe {
        tessl::nn::gemv_q4_with_scalars(&gpu.rt, bank, x, y, shape, false, |bnd| {
            gpu.icb_scalars.bind_u32(bnd, rows_off, 5);
            gpu.icb_scalars.bind_u32(bnd, cols_off, 6);
            gpu.icb_scalars.bind_u32(bnd, gs_off, 7);
        })
    }
    .map_err(map_metal)
}

/// MLX affine Q4 GEMV: `y = (scale * q_u + bias) @ x` (row-major Hot).
/// `x` is f32; simd path casts to bf16 activations (MLX-style half stream).
pub fn gemv_q4_mlx(
    gpu: &GemmaGpu,
    packed: &GpuBuffer,
    scales: &GpuBuffer,
    biases: &GpuBuffer,
    x: &GpuBuffer,
    y: &GpuBuffer,
    rows: u32,
    cols: u32,
    group_size: u32,
) -> Result<()> {
    gemv_q4_mlx_with_simd_policy(
        gpu,
        packed,
        scales,
        biases,
        x,
        y,
        rows,
        cols,
        group_size,
        gemv_simd_enabled(),
    )
}

fn gemv_q4_mlx_with_simd_policy(
    gpu: &GemmaGpu,
    packed: &GpuBuffer,
    scales: &GpuBuffer,
    biases: &GpuBuffer,
    x: &GpuBuffer,
    y: &GpuBuffer,
    rows: u32,
    cols: u32,
    group_size: u32,
    allow_simd: bool,
) -> Result<()> {
    let shape = tessl::nn::QuantShape {
        rows,
        cols,
        group_size,
    };
    // MLX Hot storage interleaves each bf16 scale/bias pair in `scales`.
    // The legacy `biases` operand remains for the crate's public ABI, but none
    // of the MLX kernels reads slot 2; Tessl deliberately rebinds this bank.
    let bank = tessl::nn::Q4MlxBank {
        packed,
        scales_biases: scales,
    };
    // Validate the complete public boundary before a bf16 conversion can
    // allocate scratch, consume stable scalar slots, or mutate device state.
    tessl::nn::validate_gemv_q4_mlx_inputs(&gpu.rt, bank, x, y, shape).map_err(map_metal)?;
    if rows == 0 {
        return Ok(());
    }

    // Prefer simdgroup-cooperative row-major GEMV (MLX qmv structure).
    // Full K-blocks are 512; remainder handled when cols % 16 == 0.
    if allow_simd
        && cols >= 256
        && cols % 16 == 0
        && group_size > 0
        && cols % group_size == 0
    {
        if tessl::ab_flags::need_barrier(true) {
            gpu.barrier()?;
        }
        let x_bf16 = prepare_act_bf16(gpu, x, cols)?;
        if tessl::ab_flags::need_barrier(true) {
            gpu.barrier()?;
        }
        return dispatch_gemv_simd(
            gpu, packed, scales, biases, &x_bf16, y, rows, cols, group_size, false,
        );
    }
    let needs_tiled = (cols as usize).saturating_mul(std::mem::size_of::<f32>())
        > gpu.rt.max_threadgroup_memory();
    // Tall (lm_head) vs wide-short — same peel; Rust picks TG size from shape.
    let entry = if needs_tiled {
        "gemv_q4_mlx_tiled"
    } else if rows >= 65_536 {
        KernelId::GemvQ4Mlx.entry_name()
    } else {
        "gemv_q4_mlx_wide"
    };
    let variant = if needs_tiled {
        tessl::nn::Q4MlxRowVariant::Tiled
    } else if entry == KernelId::GemvQ4Mlx.entry_name() {
        tessl::nn::Q4MlxRowVariant::Standard
    } else {
        tessl::nn::Q4MlxRowVariant::Wide
    };
    tessl::nn::validate_gemv_q4_mlx(&gpu.rt, bank, x, y, shape, variant)
        .map_err(map_metal)?;
    if rows == 0 {
        return Ok(());
    }
    let (rows_off, cols_off, gs_off) = push_gemv_dims(gpu, rows, cols, group_size)?;
    // SAFETY: the preflight above validates the exact host dimensions and
    // buffers. This closure binds only those dimensions from Gemma's stable,
    // runtime-owned scalar pool, which remains resident across ICB replays.
    unsafe {
        tessl::nn::gemv_q4_mlx_with_scalars(
            &gpu.rt,
            bank,
            x,
            y,
            shape,
            variant,
            |bnd| {
                gpu.icb_scalars.bind_u32(bnd, rows_off, 5);
                gpu.icb_scalars.bind_u32(bnd, cols_off, 6);
                gpu.icb_scalars.bind_u32(bnd, gs_off, 7);
            },
        )
    }
    .map_err(map_metal)
}

fn gemv_simd_enabled() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(
        || match std::env::var("GEMMA_METAL_GEMV_SIMD").ok().as_deref() {
            Some("0") | Some("false") | Some("off") => false,
            _ => true,
        },
    )
}

fn dispatch_gemv_simd(
    gpu: &GemmaGpu,
    packed: &GpuBuffer,
    scales: &GpuBuffer,
    _biases: &GpuBuffer,
    x: &GpuBuffer,
    y: &GpuBuffer,
    rows: u32,
    cols: u32,
    group_size: u32,
    interleaved: bool,
) -> Result<()> {
    // `x` must already be bf16 (see `prepare_act_bf16` at call sites).
    let entry = if interleaved {
        "gemv_q4_mlx_simd_i4"
    } else {
        "gemv_q4_mlx_simd"
    };
    let shape = tessl::nn::QuantShape {
        rows,
        cols,
        group_size,
    };
    let bank = tessl::nn::Q4MlxBank {
        packed,
        scales_biases: scales,
    };
    let layout = if interleaved {
        tessl::nn::Q4MlxLayout::Interleaved4
    } else {
        tessl::nn::Q4MlxLayout::RowMajor
    };
    tessl::nn::validate_gemv_q4_mlx_simd(&gpu.rt, bank, x, y, shape, layout, None)
        .map_err(map_metal)?;
    if rows == 0 {
        return Ok(());
    }
    let rows_per_tg = GEMV_SIMD_SG * GEMV_SIMD_ROWS; // 8
    let n_tg = ((rows as usize) + rows_per_tg as usize - 1) / rows_per_tg as usize;
    if trace_gemv_enabled() {
        eprintln!(
            "[trace] gemv entry={entry} rows={rows} cols={cols} tg={} groups={n_tg} layout={}",
            GEMV_SIMD_TPTG,
            if interleaved { "SimdI4" } else { "SimdRm" }
        );
    }
    let (rows_off, cols_off, gs_off) = push_gemv_dims(gpu, rows, cols, group_size)?;
    // SAFETY: preflight validated the exact shape and buffers before scalar
    // reservation. The closure binds only those values from the stable pool.
    unsafe {
        tessl::nn::gemv_q4_mlx_simd_with_scalars(
            &gpu.rt,
            bank,
            x,
            y,
            shape,
            layout,
            None,
            |bnd| {
                gpu.icb_scalars.bind_u32(bnd, rows_off, 5);
                gpu.icb_scalars.bind_u32(bnd, cols_off, 6);
                gpu.icb_scalars.bind_u32(bnd, gs_off, 7);
            },
        )
    }
    .map_err(map_metal)
}

/// MLX Q4 GEMV on `BlockedBn16` Hot layout.
pub fn gemv_q4_mlx_blocked(
    gpu: &GemmaGpu,
    packed: &GpuBuffer,
    scales: &GpuBuffer,
    biases: &GpuBuffer,
    x: &GpuBuffer,
    y: &GpuBuffer,
    rows: u32,
    cols: u32,
    group_size: u32,
) -> Result<()> {
    dispatch_gemv_blocked(gpu, packed, scales, biases, x, y, rows, cols, group_size)
}

pub fn gemv_q8(
    gpu: &GemmaGpu,
    packed: &GpuBuffer,
    scales: &GpuBuffer,
    zeros: &GpuBuffer,
    x: &GpuBuffer,
    y: &GpuBuffer,
    rows: u32,
    cols: u32,
    group_size: u32,
) -> Result<()> {
    // Q8 stays 1-thread-per-row (less common decode path).
    tessl::nn::gemv_q8(&gpu.rt, packed, scales, zeros, x, y, rows, cols, group_size)
        .map_err(map_metal)
}

/// Store one K or V timestep into a GPU KV cache slot (`dst[offset..] = src[0..n]`).
/// `dst_capacity` is the logical f32-element bound enforced by the shader; it
/// may be smaller than the backing allocation for a suballocated cache.
pub fn kv_store_timestep(
    gpu: &GemmaGpu,
    src: &GpuBuffer,
    dst: &GpuBuffer,
    n: u32,
    dst_offset: u32,
    dst_capacity: u32,
) -> Result<()> {
    kv_store_timestep_off(gpu, src, 0, dst, n, dst_offset, dst_capacity)
}

/// Like [`kv_store_timestep`] with a float-element offset into `src`.
pub fn kv_store_timestep_off(
    gpu: &GemmaGpu,
    src: &GpuBuffer,
    src_elem_off: u32,
    dst: &GpuBuffer,
    n: u32,
    dst_offset: u32,
    dst_capacity: u32,
) -> Result<()> {
    use tessl::dispatch::set_gpu_buf_offset;
    let src_bytes = validate_f32_range(src, src_elem_off, n, "kv_store_timestep src")?;
    validate_kv_capacity(dst, n, dst_capacity, "kv_store_timestep dst")?;
    let p = gpu
        .rt
        .pipeline(KernelId::KvStoreTimestep.entry_name())
        .map_err(map_metal)?;
    let n_off = gpu.icb_scalars.push_u32(n)?;
    let dst_off = gpu.icb_scalars.push_u32(dst_offset)?;
    let capacity_off = gpu.icb_scalars.push_u32(dst_capacity)?;
    dispatch_1d(&gpu.rt, &p, n as usize, |bnd| {
        set_gpu_buf_offset(bnd, src, src_bytes, 0);
        set_gpu_buf(bnd, dst, 1);
        gpu.icb_scalars.bind_u32(bnd, n_off, 2);
        gpu.icb_scalars.bind_u32(bnd, dst_off, 3);
        gpu.icb_scalars.bind_u32(bnd, capacity_off, 4);
    })
    .map_err(map_metal)
}

/// Fuse K+V store into one dispatch (same src element offset + dst slot offset).
/// Both destinations share the explicit logical `dst_capacity` bound.
pub fn kv_store_timestep_pair_off(
    gpu: &GemmaGpu,
    src_k: &GpuBuffer,
    src_v: &GpuBuffer,
    src_elem_off: u32,
    dst_k: &GpuBuffer,
    dst_v: &GpuBuffer,
    n: u32,
    dst_offset: u32,
    dst_capacity: u32,
) -> Result<()> {
    use tessl::dispatch::set_gpu_buf_offset;
    let src_bytes = validate_f32_range(src_k, src_elem_off, n, "kv_store_timestep_pair src_k")?;
    validate_f32_range(src_v, src_elem_off, n, "kv_store_timestep_pair src_v")?;
    validate_kv_capacity(dst_k, n, dst_capacity, "kv_store_timestep_pair dst_k")?;
    validate_kv_capacity(dst_v, n, dst_capacity, "kv_store_timestep_pair dst_v")?;
    let p = gpu
        .rt
        .pipeline(KernelId::KvStoreTimestepPair.entry_name())
        .map_err(map_metal)?;
    let n_off = gpu.icb_scalars.push_u32(n)?;
    let dst_off = if let Some(src) = dyn_src_peek_from_ctx() {
        gpu.icb_scalars.push_u32_dyn(dst_offset, src)?
    } else {
        gpu.icb_scalars.push_u32(dst_offset)?
    };
    let capacity_off = gpu.icb_scalars.push_u32(dst_capacity)?;
    dispatch_1d(&gpu.rt, &p, n as usize, |bnd| {
        set_gpu_buf_offset(bnd, src_k, src_bytes, 0);
        set_gpu_buf_offset(bnd, src_v, src_bytes, 1);
        set_gpu_buf(bnd, dst_k, 2);
        set_gpu_buf(bnd, dst_v, 3);
        gpu.icb_scalars.bind_u32(bnd, n_off, 4);
        gpu.icb_scalars.bind_u32(bnd, dst_off, 5);
        gpu.icb_scalars.bind_u32(bnd, capacity_off, 6);
    })
    .map_err(map_metal)
}

/// Chronological densify of a sliding ring into a dense FA buffer.
///
/// Grid is always `capacity * n_slot` (kernel clips via `filled`) so Binder-tape
/// / DecodeIcb cmd shape and IcbScalarPool cursors stay stable across ring wrap.
pub fn kv_ring_densify(
    gpu: &GemmaGpu,
    src: &GpuBuffer,
    dst: &GpuBuffer,
    n_slot: u32,
    capacity: u32,
    filled: u32,
    start: u32,
) -> Result<()> {
    let n = (capacity as usize)
        .checked_mul(n_slot as usize)
        .ok_or_else(|| Error::Metal("kv_ring_densify: fixed grid overflows usize".into()))?;
    if n > u32::MAX as usize {
        return Err(Error::Metal(format!(
            "kv_ring_densify: fixed grid {n} exceeds Metal uint indexing"
        )));
    }
    if n == 0 {
        return Ok(());
    }
    let p = gpu
        .rt
        .pipeline(KernelId::KvRingDensify.entry_name())
        .map_err(map_metal)?;
    let n_slot_off = gpu.icb_scalars.push_u32(n_slot)?;
    let capacity_off = gpu.icb_scalars.push_u32(capacity)?;
    let filled_off = if let Some(src) = dyn_src_filled_from_ctx() {
        gpu.icb_scalars.push_u32_dyn(filled, src)?
    } else {
        gpu.icb_scalars.push_u32(filled)?
    };
    let start_off = if let Some(src) = dyn_src_start_from_ctx() {
        gpu.icb_scalars.push_u32_dyn(start, src)?
    } else {
        gpu.icb_scalars.push_u32(start)?
    };
    dispatch_1d(&gpu.rt, &p, n, |bnd| {
        set_gpu_buf(bnd, src, 0);
        set_gpu_buf(bnd, dst, 1);
        set_gpu_buf_offset(bnd, &gpu.icb_scalars.u32s, n_slot_off, 2);
        set_gpu_buf_offset(bnd, &gpu.icb_scalars.u32s, capacity_off, 3);
        set_gpu_buf_offset(bnd, &gpu.icb_scalars.u32s, filled_off, 4);
        set_gpu_buf_offset(bnd, &gpu.icb_scalars.u32s, start_off, 5);
    })
    .map_err(map_metal)
}

/// Upload a [`QuantMatrix`] Hot bank and run GEMV against host `x`.
pub fn gemv_quant_host(gpu: &GemmaGpu, w: &QuantMatrix, x: &[f32]) -> Result<Vec<f32>> {
    if x.len() != w.cols {
        return Err(Error::Metal(format!(
            "gemv x len {} != cols {}",
            x.len(),
            w.cols
        )));
    }
    let group_size = w
        .scheme
        .group_size()
        .ok_or_else(|| Error::Metal("gemv requires Q4/Q8".into()))? as u32;

    let packed = gpu
        .rt
        .alloc_buffer(w.packed.len().max(1))
        .map_err(map_metal)?;
    packed.write_bytes(&w.packed);
    let mlx = matches!(w.scheme, crate::quant::QuantScheme::Q4Mlx { .. });
    let (scales, zeros) = if mlx {
        let sb = pack_mlx_sb_bf16(&w.scales, &w.zeros);
        let scales = gpu
            .rt
            .alloc_buffer(sb.len().max(1) * 2)
            .map_err(map_metal)?;
        scales.write_bf16_bits(&sb);
        // Stub biases buffer — Q4Mlx kernels read interleaved bfloat2 from scales.
        let zeros = gpu.rt.alloc_buffer(4).map_err(map_metal)?;
        (scales, zeros)
    } else {
        let scales = gpu
            .rt
            .alloc_buffer(w.scales.len().max(1) * 4)
            .map_err(map_metal)?;
        let zeros = gpu
            .rt
            .alloc_buffer(w.zeros.len().max(1) * 4)
            .map_err(map_metal)?;
        scales.write_f32(&w.scales);
        zeros.write_f32(&w.zeros);
        (scales, zeros)
    };
    let xb = gpu.rt.alloc_buffer(x.len() * 4).map_err(map_metal)?;
    xb.write_f32(x);
    let yb = gpu.rt.alloc_buffer(w.rows * 4).map_err(map_metal)?;

    match w.scheme {
        crate::quant::QuantScheme::Q4 { .. } => gemv_q4(
            gpu,
            &packed,
            &scales,
            &zeros,
            &xb,
            &yb,
            w.rows as u32,
            w.cols as u32,
            group_size,
        )?,
        crate::quant::QuantScheme::Q4Mlx { .. } => gemv_q4_mlx(
            gpu,
            &packed,
            &scales,
            &zeros,
            &xb,
            &yb,
            w.rows as u32,
            w.cols as u32,
            group_size,
        )?,
        crate::quant::QuantScheme::Q8 { .. } => gemv_q8(
            gpu,
            &packed,
            &scales,
            &zeros,
            &xb,
            &yb,
            w.rows as u32,
            w.cols as u32,
            group_size,
        )?,
        crate::quant::QuantScheme::Bf16 => {
            return Err(Error::Metal("gemv_quant_host: Bf16 not supported".into()));
        }
    }
    gpu.synchronize()?;
    Ok(yb.read_f32())
}

/// Prefill GEMM via tessl TensorOps (preferred) / simdgroup.
pub fn gemm_prefill(a: &Tensor, b: &Tensor, c: &Tensor) -> Result<()> {
    let rt = a.runtime();
    let backend = select_backend(rt);
    gemm(a, b, c, backend).map_err(map_metal)
}

pub fn gemm_prefill_backend(
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    backend: GemmBackend,
) -> Result<()> {
    gemm(a, b, c, backend).map_err(map_metal)
}

#[allow(clippy::too_many_arguments)]
fn validate_direct_attn_storage(
    gpu: &GemmaGpu,
    q: &GpuBuffer,
    k: &GpuBuffer,
    v: &GpuBuffer,
    o: &GpuBuffer,
    b: u32,
    tq: u32,
    h: u32,
    hkv: u32,
    d: u32,
    window: u32,
    scale: f32,
    out_bf16: bool,
) -> Result<u32> {
    let dims = tessl::nn::AttnDims {
        batch: b,
        tq,
        heads: h,
        heads_kv: hkv,
        window,
        scale,
    };
    let capacity =
        tessl::nn::validate_attn_storage(&dims, d, q, k, v, o, out_bf16).map_err(map_metal)?;
    // Every direct binder below puts mutable Tkv/position values and immutable
    // dimensions in this stable pool. An output alias could overwrite values a
    // concurrent threadgroup or later ICB replay still reads.
    tessl::nn::validate_attn_output_scalar_aliases(
        &dims,
        o,
        &[
            ("ICB u32 pool", &gpu.icb_scalars.u32s),
            ("ICB f32 pool", &gpu.icb_scalars.f32s),
        ],
    )
    .map_err(map_metal)?;
    Ok(capacity)
}

/// Sliding-window FA @ D=256.
///
/// Layout: Q/O `[B,Tq,H,D]`, K/V `[B,capacity,Hkv,D]`; only the first live
/// `Tkv` positions are visited. Absolute positions are
/// `q_pos_offset + t_q` / `kv_pos_offset + t_k` (prefill: offsets 0, Tq=Tkv;
/// decode / ring densify: Tq=1, Tkv=cache_len, offsets set accordingly).
pub fn flash_attn_swa_h256(
    gpu: &GemmaGpu,
    q: &GpuBuffer,
    k: &GpuBuffer,
    v: &GpuBuffer,
    o: &GpuBuffer,
    b: u32,
    tq: u32,
    tkv: u32,
    h: u32,
    hkv: u32,
    window: u32,
    scale: f32,
    q_pos_offset: u32,
    kv_pos_offset: u32,
) -> Result<()> {
    flash_attn_swa_h256_ex(
        gpu,
        q,
        k,
        v,
        o,
        b,
        tq,
        tkv,
        h,
        hkv,
        window,
        scale,
        q_pos_offset,
        kv_pos_offset,
        false,
    )
}

pub fn flash_attn_swa_h256_ex(
    gpu: &GemmaGpu,
    q: &GpuBuffer,
    k: &GpuBuffer,
    v: &GpuBuffer,
    o: &GpuBuffer,
    b: u32,
    tq: u32,
    tkv: u32,
    h: u32,
    hkv: u32,
    window: u32,
    scale: f32,
    q_pos_offset: u32,
    kv_pos_offset: u32,
    out_bf16: bool,
) -> Result<()> {
    let kv_capacity =
        validate_direct_attn_storage(gpu, q, k, v, o, b, tq, h, hkv, 256, window, scale, out_bf16)?;
    let p = gpu
        .rt
        .pipeline(KernelId::FlashAttnSwaH256.entry_name())
        .map_err(map_metal)?;
    const BR: usize = 8;
    let groups_x = ((tq as usize) + BR - 1) / BR;
    let groups_y = (b as usize)
        .checked_mul(h as usize)
        .ok_or_else(|| Error::Metal("flash_attn_swa_h256: grid height overflows".into()))?;
    let tptg = 32usize;
    let b_off = gpu.icb_scalars.push_u32(b)?;
    let tq_off = gpu.icb_scalars.push_u32(tq)?;
    let tkv_off = if let Some(src) = dyn_src_tkv_from_ctx() {
        gpu.icb_scalars.push_u32_dyn(tkv, src)?
    } else {
        gpu.icb_scalars.push_u32(tkv)?
    };
    let h_off = gpu.icb_scalars.push_u32(h)?;
    let hkv_off = gpu.icb_scalars.push_u32(hkv)?;
    let window_off = gpu.icb_scalars.push_u32(window)?;
    let scale_off = gpu.icb_scalars.push_f32(scale)?;
    let q_pos_off = gpu.icb_scalars.push_u32_dyn(q_pos_offset, IcbDynSrc::Pos)?;
    let kv_pos_off = if let Some(src) = dyn_src_kv_pos_from_ctx() {
        gpu.icb_scalars.push_u32_dyn(kv_pos_offset, src)?
    } else {
        gpu.icb_scalars.push_u32(kv_pos_offset)?
    };
    let out_bf16_off = gpu.icb_scalars.push_u32(if out_bf16 { 1 } else { 0 })?;
    let kv_capacity_off = gpu.icb_scalars.push_u32(kv_capacity)?;
    dispatch_2d_tg(&gpu.rt, &p, groups_x, groups_y, tptg, |bnd| {
        set_gpu_buf(bnd, q, 0);
        set_gpu_buf(bnd, k, 1);
        set_gpu_buf(bnd, v, 2);
        set_gpu_buf(bnd, o, 3);
        set_gpu_buf_offset(bnd, &gpu.icb_scalars.u32s, b_off, 4);
        set_gpu_buf_offset(bnd, &gpu.icb_scalars.u32s, tq_off, 5);
        set_gpu_buf_offset(bnd, &gpu.icb_scalars.u32s, tkv_off, 6);
        set_gpu_buf_offset(bnd, &gpu.icb_scalars.u32s, h_off, 7);
        set_gpu_buf_offset(bnd, &gpu.icb_scalars.u32s, hkv_off, 8);
        set_gpu_buf_offset(bnd, &gpu.icb_scalars.u32s, window_off, 9);
        set_gpu_buf_offset(bnd, &gpu.icb_scalars.f32s, scale_off, 10);
        set_gpu_buf_offset(bnd, &gpu.icb_scalars.u32s, q_pos_off, 11);
        set_gpu_buf_offset(bnd, &gpu.icb_scalars.u32s, kv_pos_off, 12);
        set_gpu_buf_offset(bnd, &gpu.icb_scalars.u32s, out_bf16_off, 13);
        set_gpu_buf_offset(bnd, &gpu.icb_scalars.u32s, kv_capacity_off, 14);
    })
    .map_err(map_metal)
}

/// Prefill convenience: dense causal SWA with `Tq = Tkv = t`, offsets 0.
pub fn flash_attn_swa_h256_prefill(
    gpu: &GemmaGpu,
    q: &GpuBuffer,
    k: &GpuBuffer,
    v: &GpuBuffer,
    o: &GpuBuffer,
    b: u32,
    t: u32,
    h: u32,
    hkv: u32,
    window: u32,
    scale: f32,
) -> Result<()> {
    flash_attn_swa_h256(gpu, q, k, v, o, b, t, t, h, hkv, window, scale, 0, 0)
}

/// Sliding-window FA @ D=128 (DFlash draft stub — same tiled FA-2 shape as h256).
pub fn flash_attn_swa_h128(
    gpu: &GemmaGpu,
    q: &GpuBuffer,
    k: &GpuBuffer,
    v: &GpuBuffer,
    o: &GpuBuffer,
    b: u32,
    tq: u32,
    tkv: u32,
    h: u32,
    hkv: u32,
    window: u32,
    scale: f32,
    q_pos_offset: u32,
    kv_pos_offset: u32,
) -> Result<()> {
    let kv_capacity =
        validate_direct_attn_storage(gpu, q, k, v, o, b, tq, h, hkv, 128, window, scale, false)?;
    let p = gpu
        .rt
        .pipeline(KernelId::FlashAttnSwaH128.entry_name())
        .map_err(map_metal)?;
    const BR: usize = 8;
    let groups_x = ((tq as usize) + BR - 1) / BR;
    let groups_y = (b as usize)
        .checked_mul(h as usize)
        .ok_or_else(|| Error::Metal("flash_attn_swa_h128: grid height overflows".into()))?;
    let tptg = 32usize;
    let b_off = gpu.icb_scalars.push_u32(b)?;
    let tq_off = gpu.icb_scalars.push_u32(tq)?;
    let tkv_off = gpu.icb_scalars.push_u32(tkv)?;
    let h_off = gpu.icb_scalars.push_u32(h)?;
    let hkv_off = gpu.icb_scalars.push_u32(hkv)?;
    let window_off = gpu.icb_scalars.push_u32(window)?;
    let scale_off = gpu.icb_scalars.push_f32(scale)?;
    let q_pos_off = gpu.icb_scalars.push_u32(q_pos_offset)?;
    let kv_pos_off = gpu.icb_scalars.push_u32(kv_pos_offset)?;
    let kv_capacity_off = gpu.icb_scalars.push_u32(kv_capacity)?;
    dispatch_2d_tg(&gpu.rt, &p, groups_x, groups_y, tptg, |bnd| {
        set_gpu_buf(bnd, q, 0);
        set_gpu_buf(bnd, k, 1);
        set_gpu_buf(bnd, v, 2);
        set_gpu_buf(bnd, o, 3);
        set_gpu_buf_offset(bnd, &gpu.icb_scalars.u32s, b_off, 4);
        set_gpu_buf_offset(bnd, &gpu.icb_scalars.u32s, tq_off, 5);
        set_gpu_buf_offset(bnd, &gpu.icb_scalars.u32s, tkv_off, 6);
        set_gpu_buf_offset(bnd, &gpu.icb_scalars.u32s, h_off, 7);
        set_gpu_buf_offset(bnd, &gpu.icb_scalars.u32s, hkv_off, 8);
        set_gpu_buf_offset(bnd, &gpu.icb_scalars.u32s, window_off, 9);
        set_gpu_buf_offset(bnd, &gpu.icb_scalars.f32s, scale_off, 10);
        set_gpu_buf_offset(bnd, &gpu.icb_scalars.u32s, q_pos_off, 11);
        set_gpu_buf_offset(bnd, &gpu.icb_scalars.u32s, kv_pos_off, 12);
        set_gpu_buf_offset(bnd, &gpu.icb_scalars.u32s, kv_capacity_off, 13);
    })
    .map_err(map_metal)
}

/// Prefill convenience for D=128 FA stub.
pub fn flash_attn_swa_h128_prefill(
    gpu: &GemmaGpu,
    q: &GpuBuffer,
    k: &GpuBuffer,
    v: &GpuBuffer,
    o: &GpuBuffer,
    b: u32,
    t: u32,
    h: u32,
    hkv: u32,
    window: u32,
    scale: f32,
) -> Result<()> {
    flash_attn_swa_h128(gpu, q, k, v, o, b, t, t, h, hkv, window, scale, 0, 0)
}

/// Global FA @ D=512 (separate Tq/Tkv + absolute position offsets).
pub fn flash_attn_global_h512(
    gpu: &GemmaGpu,
    q: &GpuBuffer,
    k: &GpuBuffer,
    v: &GpuBuffer,
    o: &GpuBuffer,
    b: u32,
    tq: u32,
    tkv: u32,
    h: u32,
    hkv: u32,
    scale: f32,
    q_pos_offset: u32,
    kv_pos_offset: u32,
) -> Result<()> {
    flash_attn_global_h512_ex(
        gpu,
        q,
        k,
        v,
        o,
        b,
        tq,
        tkv,
        h,
        hkv,
        scale,
        q_pos_offset,
        kv_pos_offset,
        false,
    )
}

pub fn flash_attn_global_h512_ex(
    gpu: &GemmaGpu,
    q: &GpuBuffer,
    k: &GpuBuffer,
    v: &GpuBuffer,
    o: &GpuBuffer,
    b: u32,
    tq: u32,
    tkv: u32,
    h: u32,
    hkv: u32,
    scale: f32,
    q_pos_offset: u32,
    kv_pos_offset: u32,
    out_bf16: bool,
) -> Result<()> {
    let kv_capacity =
        validate_direct_attn_storage(gpu, q, k, v, o, b, tq, h, hkv, 512, 0, scale, out_bf16)?;
    let p = gpu
        .rt
        .pipeline(KernelId::FlashAttnGlobalH512.entry_name())
        .map_err(map_metal)?;
    const BR: usize = 4;
    let groups_x = ((tq as usize) + BR - 1) / BR;
    let groups_y = (b as usize)
        .checked_mul(h as usize)
        .ok_or_else(|| Error::Metal("flash_attn_global_h512: grid height overflows".into()))?;
    let tptg = 32usize;
    let b_off = gpu.icb_scalars.push_u32(b)?;
    let tq_off = gpu.icb_scalars.push_u32(tq)?;
    let tkv_off = if let Some(src) = dyn_src_tkv_from_ctx() {
        gpu.icb_scalars.push_u32_dyn(tkv, src)?
    } else {
        gpu.icb_scalars.push_u32(tkv)?
    };
    let h_off = gpu.icb_scalars.push_u32(h)?;
    let hkv_off = gpu.icb_scalars.push_u32(hkv)?;
    let scale_off = gpu.icb_scalars.push_f32(scale)?;
    let q_pos_off = gpu.icb_scalars.push_u32_dyn(q_pos_offset, IcbDynSrc::Pos)?;
    let kv_pos_off = if let Some(src) = dyn_src_kv_pos_from_ctx() {
        gpu.icb_scalars.push_u32_dyn(kv_pos_offset, src)?
    } else {
        gpu.icb_scalars.push_u32(kv_pos_offset)?
    };
    let out_bf16_off = gpu.icb_scalars.push_u32(if out_bf16 { 1 } else { 0 })?;
    let kv_capacity_off = gpu.icb_scalars.push_u32(kv_capacity)?;
    dispatch_2d_tg(&gpu.rt, &p, groups_x, groups_y, tptg, |bnd| {
        set_gpu_buf(bnd, q, 0);
        set_gpu_buf(bnd, k, 1);
        set_gpu_buf(bnd, v, 2);
        set_gpu_buf(bnd, o, 3);
        set_gpu_buf_offset(bnd, &gpu.icb_scalars.u32s, b_off, 4);
        set_gpu_buf_offset(bnd, &gpu.icb_scalars.u32s, tq_off, 5);
        set_gpu_buf_offset(bnd, &gpu.icb_scalars.u32s, tkv_off, 6);
        set_gpu_buf_offset(bnd, &gpu.icb_scalars.u32s, h_off, 7);
        set_gpu_buf_offset(bnd, &gpu.icb_scalars.u32s, hkv_off, 8);
        set_gpu_buf_offset(bnd, &gpu.icb_scalars.f32s, scale_off, 9);
        set_gpu_buf_offset(bnd, &gpu.icb_scalars.u32s, q_pos_off, 10);
        set_gpu_buf_offset(bnd, &gpu.icb_scalars.u32s, kv_pos_off, 11);
        set_gpu_buf_offset(bnd, &gpu.icb_scalars.u32s, out_bf16_off, 12);
        set_gpu_buf_offset(bnd, &gpu.icb_scalars.u32s, kv_capacity_off, 13);
    })
    .map_err(map_metal)
}

/// Prefill convenience: dense causal global with `Tq = Tkv = t`, offsets 0.
pub fn flash_attn_global_h512_prefill(
    gpu: &GemmaGpu,
    q: &GpuBuffer,
    k: &GpuBuffer,
    v: &GpuBuffer,
    o: &GpuBuffer,
    b: u32,
    t: u32,
    h: u32,
    hkv: u32,
    scale: f32,
) -> Result<()> {
    flash_attn_global_h512(gpu, q, k, v, o, b, t, t, h, hkv, scale, 0, 0)
}

pub fn rms_qkv_rope(
    gpu: &GemmaGpu,
    q: &GpuBuffer,
    k: &GpuBuffer,
    v: &GpuBuffer,
    q_w: &GpuBuffer,
    k_w: &GpuBuffer,
    v_w: &GpuBuffer,
    t: u32,
    hq: u32,
    hkv: u32,
    d: u32,
    rotary_dim: u32,
    pos_offset: u32,
    theta: f32,
    eps: f32,
) -> Result<()> {
    rms_qkv_rope_ex(
        gpu, q, k, v, q_w, k_w, v_w, t, hq, hkv, d, rotary_dim, pos_offset, theta, eps,
        /*q_only*/ false,
    )
}

/// When `q_only`, skip stale K/V norm+RoPE (consumer layers).
pub fn rms_qkv_rope_ex(
    gpu: &GemmaGpu,
    q: &GpuBuffer,
    k: &GpuBuffer,
    v: &GpuBuffer,
    q_w: &GpuBuffer,
    k_w: &GpuBuffer,
    v_w: &GpuBuffer,
    t: u32,
    hq: u32,
    hkv: u32,
    d: u32,
    rotary_dim: u32,
    pos_offset: u32,
    theta: f32,
    eps: f32,
    q_only: bool,
) -> Result<()> {
    tessl::nn::rms_qkv_rope(
        &gpu.rt,
        tessl::nn::QkvRopeVariant::PosConst,
        tessl::nn::QkvBuffers {
            q,
            k,
            v,
            q_weight: q_w,
            k_weight: k_w,
            v_weight: v_w,
        },
        tessl::nn::QkvRopeDims {
            t,
            heads_q: hq,
            heads_kv: hkv,
            head_dim: d,
            rotary_dim,
            theta,
            eps,
        },
        pos_offset,
        None,
        None,
        q_only,
    )
    .map_err(map_metal)
}

/// Like [`rms_qkv_rope_ex`], but RoPE `pos` is read from a GPU-resident `u32×1`
/// buffer (encode-once / CB-replay scaffolding). Host writes the buffer once per
/// step; layers bind the same address instead of re-packing a const-arena scalar.
pub fn rms_qkv_rope_ex_posbuf(
    gpu: &GemmaGpu,
    q: &GpuBuffer,
    k: &GpuBuffer,
    v: &GpuBuffer,
    q_w: &GpuBuffer,
    k_w: &GpuBuffer,
    v_w: &GpuBuffer,
    t: u32,
    hq: u32,
    hkv: u32,
    d: u32,
    rotary_dim: u32,
    pos_buf: &GpuBuffer,
    theta: f32,
    eps: f32,
    q_only: bool,
) -> Result<()> {
    let qkv = tessl::nn::QkvBuffers {
        q,
        k,
        v,
        q_weight: q_w,
        k_weight: k_w,
        v_weight: v_w,
    };
    let dims = tessl::nn::QkvRopeDims {
        t,
        heads_q: hq,
        heads_kv: hkv,
        head_dim: d,
        rotary_dim,
        theta,
        eps,
    };
    tessl::nn::validate_rms_qkv_rope(
        &gpu.rt,
        tessl::nn::QkvRopeVariant::PosBuffer,
        qkv,
        dims,
        Some(pos_buf),
        None,
        q_only,
    )
    .map_err(map_metal)?;
    // Resolve before mutating the stable scalar pool or its capture tape. The
    // canonical dispatch reuses this cached pipeline after the scalar writes.
    gpu.rt
        .pipeline(KernelId::RmsQkvRopePosbuf.entry_name())
        .map_err(map_metal)?;
    let t_off = gpu.icb_scalars.push_u32(t)?;
    let hq_off = gpu.icb_scalars.push_u32(hq)?;
    let hkv_off = gpu.icb_scalars.push_u32(hkv)?;
    let d_off = gpu.icb_scalars.push_u32(d)?;
    let rotary_off = gpu.icb_scalars.push_u32(rotary_dim)?;
    let theta_off = gpu.icb_scalars.push_f32(theta)?;
    let eps_off = gpu.icb_scalars.push_f32(eps)?;
    // SAFETY: the callback binds exactly the documented, prevalidated scalar
    // values from runtime-owned Hot storage that remains resident for replay.
    unsafe {
        tessl::nn::rms_qkv_rope_with_scalars(
            &gpu.rt,
            tessl::nn::QkvRopeVariant::PosBuffer,
            qkv,
            dims,
            Some(pos_buf),
            None,
            q_only,
            |bnd, _| {
                gpu.icb_scalars.bind_u32(bnd, t_off, 6);
                gpu.icb_scalars.bind_u32(bnd, hq_off, 7);
                gpu.icb_scalars.bind_u32(bnd, hkv_off, 8);
                gpu.icb_scalars.bind_u32(bnd, d_off, 9);
                gpu.icb_scalars.bind_u32(bnd, rotary_off, 10);
                gpu.icb_scalars.bind_f32(bnd, theta_off, 12);
                gpu.icb_scalars.bind_f32(bnd, eps_off, 13);
            },
        )
    }
    .map_err(map_metal)
}

/// Producer fusion: RoPE(+norms) into scratch Q/K/V **and** store K/V into the
/// cache slot at `kv_dst_offset` (replaces a follow-up `kv_store_timestep_pair`).
/// `kv_capacity` is the logical f32-element bound shared by both cache buffers.
pub fn rms_qkv_rope_kv_store(
    gpu: &GemmaGpu,
    q: &GpuBuffer,
    k: &GpuBuffer,
    v: &GpuBuffer,
    q_w: &GpuBuffer,
    k_w: &GpuBuffer,
    v_w: &GpuBuffer,
    t: u32,
    hq: u32,
    hkv: u32,
    d: u32,
    rotary_dim: u32,
    pos_buf: &GpuBuffer,
    theta: f32,
    eps: f32,
    dst_k: &GpuBuffer,
    dst_v: &GpuBuffer,
    kv_dst_offset: u32,
    kv_capacity: u32,
) -> Result<()> {
    let qkv = tessl::nn::QkvBuffers {
        q,
        k,
        v,
        q_weight: q_w,
        k_weight: k_w,
        v_weight: v_w,
    };
    let dims = tessl::nn::QkvRopeDims {
        t,
        heads_q: hq,
        heads_kv: hkv,
        head_dim: d,
        rotary_dim,
        theta,
        eps,
    };
    let target = tessl::nn::KvStoreTarget {
        dst_k,
        dst_v,
        // The stable-pool element is selected by the callback's validated
        // byte-offset rebind at slot 16 below.
        dst_offset: &gpu.icb_scalars.u32s,
        capacity: kv_capacity,
    };
    tessl::nn::validate_rms_qkv_rope(
        &gpu.rt,
        tessl::nn::QkvRopeVariant::PosBufferKvStore,
        qkv,
        dims,
        Some(pos_buf),
        Some(target),
        false,
    )
    .map_err(map_metal)?;
    gpu.rt
        .pipeline(KernelId::RmsQkvRopeKvStore.entry_name())
        .map_err(map_metal)?;
    let t_off = gpu.icb_scalars.push_u32(t)?;
    let hq_off = gpu.icb_scalars.push_u32(hq)?;
    let hkv_off = gpu.icb_scalars.push_u32(hkv)?;
    let d_off = gpu.icb_scalars.push_u32(d)?;
    let rotary_off = gpu.icb_scalars.push_u32(rotary_dim)?;
    let theta_off = gpu.icb_scalars.push_f32(theta)?;
    let eps_off = gpu.icb_scalars.push_f32(eps)?;
    let kv_dst_off = if let Some(src) = dyn_src_peek_from_ctx() {
        gpu.icb_scalars.push_u32_dyn(kv_dst_offset, src)?
    } else {
        gpu.icb_scalars.push_u32(kv_dst_offset)?
    };
    let kv_capacity_off = gpu.icb_scalars.push_u32(kv_capacity)?;
    // SAFETY: all scalar values were included in the canonical preflight.
    // Slot 16 is the documented stable-pool exception and stays within the
    // validated `target.dst_offset` allocation.
    unsafe {
        tessl::nn::rms_qkv_rope_with_scalars(
            &gpu.rt,
            tessl::nn::QkvRopeVariant::PosBufferKvStore,
            qkv,
            dims,
            Some(pos_buf),
            Some(target),
            false,
            |bnd, validated_capacity| {
                debug_assert_eq!(validated_capacity, Some(kv_capacity));
                gpu.icb_scalars.bind_u32(bnd, t_off, 6);
                gpu.icb_scalars.bind_u32(bnd, hq_off, 7);
                gpu.icb_scalars.bind_u32(bnd, hkv_off, 8);
                gpu.icb_scalars.bind_u32(bnd, d_off, 9);
                gpu.icb_scalars.bind_u32(bnd, rotary_off, 10);
                gpu.icb_scalars.bind_f32(bnd, theta_off, 12);
                gpu.icb_scalars.bind_f32(bnd, eps_off, 13);
                gpu.icb_scalars.bind_u32(bnd, kv_dst_off, 16);
                gpu.icb_scalars.bind_u32(bnd, kv_capacity_off, 17);
            },
        )
    }
    .map_err(map_metal)
}

pub fn ple_lookup(
    gpu: &GemmaGpu,
    token_ids: &GpuBuffer,
    table: &GpuBuffer,
    out: &GpuBuffer,
    dim: u32,
    vocab: u32,
    n: u32,
    scale: f32,
) -> Result<()> {
    let p = gpu
        .rt
        .pipeline(KernelId::PleLookup.entry_name())
        .map_err(map_metal)?;
    let threads = (n * dim) as usize;
    let dim_off = gpu.icb_scalars.push_u32(dim)?;
    let vocab_off = gpu.icb_scalars.push_u32(vocab)?;
    let n_off = gpu.icb_scalars.push_u32(n)?;
    let scale_off = gpu.icb_scalars.push_f32(scale)?;
    dispatch_1d(&gpu.rt, &p, threads, |bnd| {
        set_gpu_buf(bnd, token_ids, 0);
        set_gpu_buf(bnd, table, 1);
        set_gpu_buf(bnd, out, 2);
        gpu.icb_scalars.bind_u32(bnd, dim_off, 3);
        gpu.icb_scalars.bind_u32(bnd, vocab_off, 4);
        gpu.icb_scalars.bind_u32(bnd, n_off, 5);
        gpu.icb_scalars.bind_f32(bnd, scale_off, 6);
    })
    .map_err(map_metal)
}

/// MLX packed Q4 PLE: one Hot bank for all layers; `layer` selects the slice.
pub fn ple_lookup_q4_mlx(
    gpu: &GemmaGpu,
    token_ids: &GpuBuffer,
    packed: &GpuBuffer,
    scales: &GpuBuffer,
    biases: &GpuBuffer,
    out: &GpuBuffer,
    dim: u32,
    vocab: u32,
    n: u32,
    scale: f32,
    layer: u32,
    num_layers: u32,
    group_size: u32,
) -> Result<()> {
    let p = gpu
        .rt
        .pipeline(KernelId::PleLookupQ4Mlx.entry_name())
        .map_err(map_metal)?;
    let threads = (n * dim) as usize;
    let dim_off = gpu.icb_scalars.push_u32(dim)?;
    let vocab_off = gpu.icb_scalars.push_u32(vocab)?;
    let n_off = gpu.icb_scalars.push_u32(n)?;
    let scale_off = gpu.icb_scalars.push_f32(scale)?;
    let layer_off = gpu.icb_scalars.push_u32(layer)?;
    let nlayers_off = gpu.icb_scalars.push_u32(num_layers)?;
    let gs_off = gpu.icb_scalars.push_u32(group_size)?;
    dispatch_1d(&gpu.rt, &p, threads, |bnd| {
        set_gpu_buf(bnd, token_ids, 0);
        set_gpu_buf(bnd, packed, 1);
        set_gpu_buf(bnd, scales, 2);
        set_gpu_buf(bnd, biases, 3);
        set_gpu_buf(bnd, out, 4);
        gpu.icb_scalars.bind_u32(bnd, dim_off, 5);
        gpu.icb_scalars.bind_u32(bnd, vocab_off, 6);
        gpu.icb_scalars.bind_u32(bnd, n_off, 7);
        gpu.icb_scalars.bind_f32(bnd, scale_off, 8);
        gpu.icb_scalars.bind_u32(bnd, layer_off, 9);
        gpu.icb_scalars.bind_u32(bnd, nlayers_off, 10);
        gpu.icb_scalars.bind_u32(bnd, gs_off, 11);
    })
    .map_err(map_metal)
}

pub fn ple_residual_add(
    gpu: &GemmaGpu,
    dst: &GpuBuffer,
    src: &GpuBuffer,
    combine_scale: f32,
    n: u32,
) -> Result<()> {
    let p = gpu
        .rt
        .pipeline(KernelId::PleResidualAdd.entry_name())
        .map_err(map_metal)?;
    let scale_off = gpu.icb_scalars.push_f32(combine_scale)?;
    let n_off = gpu.icb_scalars.push_u32(n)?;
    dispatch_1d(&gpu.rt, &p, n as usize, |bnd| {
        set_gpu_buf(bnd, dst, 0);
        set_gpu_buf(bnd, src, 1);
        gpu.icb_scalars.bind_f32(bnd, scale_off, 2);
        gpu.icb_scalars.bind_u32(bnd, n_off, 3);
    })
    .map_err(map_metal)
}

pub fn mlp_gelu_tanh(
    gpu: &GemmaGpu,
    gate: &GpuBuffer,
    up: &GpuBuffer,
    out: &GpuBuffer,
    n: u32,
) -> Result<()> {
    let n_off = gpu.icb_scalars.push_u32(n)?;
    // SAFETY: the closure binds only documented slot 3 to the exact `n` value.
    // `gpu` owns the same-runtime scalar pool, which outlives this dispatch and
    // any ICB replay that references the stable buffer address.
    unsafe {
        tessl::nn::mlp_gelu_tanh_with_scalars(&gpu.rt, gate, up, out, n, |bnd| {
            gpu.icb_scalars.bind_u32(bnd, n_off, 3);
        })
    }
    .map_err(map_metal)
}

/// `gelu(gate)*up` writing bf16 into `out_bf16` (typically act scratch).
pub fn mlp_gelu_tanh_bf16(
    gpu: &GemmaGpu,
    gate: &GpuBuffer,
    up: &GpuBuffer,
    out_bf16: &GpuBuffer,
    n: u32,
) -> Result<()> {
    let n_off = gpu.icb_scalars.push_u32(n)?;
    // SAFETY: the closure binds only documented slot 3 to the exact `n` value.
    // `gpu` owns the same-runtime scalar pool, which outlives this dispatch and
    // any ICB replay that references the stable buffer address.
    unsafe {
        tessl::nn::mlp_gelu_tanh_bf16_with_scalars(&gpu.rt, gate, up, out_bf16, n, |bnd| {
            gpu.icb_scalars.bind_u32(bnd, n_off, 3);
        })
    }
    .map_err(map_metal)
}

/// Fused `out = silu(gate) * up` (DFlash draft MLP).
pub fn mlp_silu(
    gpu: &GemmaGpu,
    gate: &GpuBuffer,
    up: &GpuBuffer,
    out: &GpuBuffer,
    n: u32,
) -> Result<()> {
    let n_off = gpu.icb_scalars.push_u32(n)?;
    // SAFETY: the closure binds only documented slot 3 to the exact `n` value.
    // `gpu` owns the same-runtime scalar pool, which outlives this dispatch and
    // any ICB replay that references the stable buffer address.
    unsafe {
        tessl::nn::mlp_silu_with_scalars(&gpu.rt, gate, up, out, n, |bnd| {
            gpu.icb_scalars.bind_u32(bnd, n_off, 3);
        })
    }
    .map_err(map_metal)
}

/// Validate scalar values that are otherwise visible only through the raw
/// Tessl binder closure. This must run before allocating a scalar-pool slot or
/// entering an unsafe `_with_scalars` call.
fn validate_rms_scalars(entry: &str, dim: u32, eps: f32, layer_scale: Option<f32>) -> Result<()> {
    if dim == 0 {
        return Err(Error::Metal(format!("{entry}: dim must be non-zero")));
    }
    if !eps.is_finite() || eps <= 0.0 {
        return Err(Error::Metal(format!(
            "{entry}: eps must be finite and > 0, got {eps}"
        )));
    }
    if let Some(scale) = layer_scale {
        if !scale.is_finite() {
            return Err(Error::Metal(format!(
                "{entry}: layer_scale must be finite, got {scale}"
            )));
        }
    }
    Ok(())
}

/// Hidden/residual RMSNorm: `out[rows, dim] = rms(x) * weight`.
pub fn rms_norm_f32(
    gpu: &GemmaGpu,
    x: &GpuBuffer,
    weight: &GpuBuffer,
    out: &GpuBuffer,
    rows: u32,
    dim: u32,
    eps: f32,
) -> Result<()> {
    validate_rms_scalars("rms_norm_f32", dim, eps, None)?;
    let rows_off = gpu.icb_scalars.push_u32(rows)?;
    let dim_off = gpu.icb_scalars.push_u32(dim)?;
    let eps_off = gpu.icb_scalars.push_f32(eps)?;
    // SAFETY: the closure binds only documented slots 3..=5 to the exact
    // `rows`, `dim`, and `eps` values. `gpu` owns the same-runtime scalar pool,
    // which outlives this dispatch and any ICB replay using its stable buffers.
    unsafe {
        tessl::nn::rms_norm_f32_with_scalars(&gpu.rt, x, weight, out, rows, dim, |bnd| {
            gpu.icb_scalars.bind_u32(bnd, rows_off, 3);
            gpu.icb_scalars.bind_u32(bnd, dim_off, 4);
            gpu.icb_scalars.bind_f32(bnd, eps_off, 5);
        })
    }
    .map_err(map_metal)
}

/// RMSNorm writing bf16 into `out` (typically [`GemmaGpu::act_bf16_scratch`]).
pub fn rms_norm_bf16(
    gpu: &GemmaGpu,
    x: &GpuBuffer,
    weight: &GpuBuffer,
    out_bf16: &GpuBuffer,
    rows: u32,
    dim: u32,
    eps: f32,
) -> Result<()> {
    validate_rms_scalars("rms_norm_bf16", dim, eps, None)?;
    let rows_off = gpu.icb_scalars.push_u32(rows)?;
    let dim_off = gpu.icb_scalars.push_u32(dim)?;
    let eps_off = gpu.icb_scalars.push_f32(eps)?;
    // SAFETY: the closure binds only documented slots 3..=5 to the exact
    // `rows`, `dim`, and `eps` values. `gpu` owns the same-runtime scalar pool,
    // which outlives this dispatch and any ICB replay using its stable buffers.
    unsafe {
        tessl::nn::rms_norm_bf16_with_scalars(&gpu.rt, x, weight, out_bf16, rows, dim, |bnd| {
            gpu.icb_scalars.bind_u32(bnd, rows_off, 3);
            gpu.icb_scalars.bind_u32(bnd, dim_off, 4);
            gpu.icb_scalars.bind_f32(bnd, eps_off, 5);
        })
    }
    .map_err(map_metal)
}

/// RMSNorm → act bf16 scratch (fused producer path). Returns the scratch handle.
pub fn rms_norm_to_act_bf16(
    gpu: &GemmaGpu,
    x: &GpuBuffer,
    weight: &GpuBuffer,
    rows: u32,
    dim: u32,
    eps: f32,
) -> Result<GpuBuffer> {
    validate_rms_scalars("rms_norm_to_act_bf16", dim, eps, None)?;
    let n = (rows as usize).saturating_mul(dim as usize);
    let dst = gpu.act_bf16_scratch(n.max(1))?;
    rms_norm_bf16(gpu, x, weight, &dst, rows, dim, eps)?;
    Ok(dst)
}

/// Fused Gemma4 dual-norm residual: `resid[rows, dim] += rms_norm(x) * weight`.
/// When `layer_scale != 1`, folds `resid = scale * (resid + rms)`.
pub fn rms_norm_residual_add_f32(
    gpu: &GemmaGpu,
    x: &GpuBuffer,
    weight: &GpuBuffer,
    resid: &GpuBuffer,
    rows: u32,
    dim: u32,
    eps: f32,
) -> Result<()> {
    rms_norm_residual_add_f32_scaled(gpu, x, weight, resid, rows, dim, eps, 1.0)
}

/// Like [`rms_norm_residual_add_f32`] with optional end-of-layer `layer_scale`.
pub fn rms_norm_residual_add_f32_scaled(
    gpu: &GemmaGpu,
    x: &GpuBuffer,
    weight: &GpuBuffer,
    resid: &GpuBuffer,
    rows: u32,
    dim: u32,
    eps: f32,
    layer_scale: f32,
) -> Result<()> {
    validate_rms_scalars(
        "rms_norm_residual_add_f32_scaled",
        dim,
        eps,
        Some(layer_scale),
    )?;
    let rows_off = gpu.icb_scalars.push_u32(rows)?;
    let dim_off = gpu.icb_scalars.push_u32(dim)?;
    let eps_off = gpu.icb_scalars.push_f32(eps)?;
    let scale_off = gpu.icb_scalars.push_f32(layer_scale)?;
    // SAFETY: the closure binds only documented slots 3..=6 to the exact
    // `rows`, `dim`, `eps`, and `layer_scale` values. `gpu` owns the
    // same-runtime scalar pool, which outlives this dispatch and any ICB replay
    // using its stable buffers.
    unsafe {
        tessl::nn::rms_norm_residual_add_f32_with_scalars(
            &gpu.rt,
            x,
            weight,
            resid,
            rows,
            dim,
            |bnd| {
                gpu.icb_scalars.bind_u32(bnd, rows_off, 3);
                gpu.icb_scalars.bind_u32(bnd, dim_off, 4);
                gpu.icb_scalars.bind_f32(bnd, eps_off, 5);
                gpu.icb_scalars.bind_f32(bnd, scale_off, 6);
            },
        )
    }
    .map_err(map_metal)
}

/// Cast `src[0..n]` f32 → bf16 into `dst` (n bf16 elements).
pub fn cast_f32_to_bf16(gpu: &GemmaGpu, src: &GpuBuffer, dst: &GpuBuffer, n: u32) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let p = gpu.rt.pipeline("cast_f32_to_bf16").map_err(map_metal)?;
    let n_off = gpu.icb_scalars.push_u32(n)?;
    dispatch_1d(&gpu.rt, &p, n as usize, |bnd| {
        set_gpu_buf(bnd, src, 0);
        set_gpu_buf(bnd, dst, 1);
        gpu.icb_scalars.bind_u32(bnd, n_off, 2);
    })
    .map_err(map_metal)
}

/// Expand bf16 activations to f32 for classic `gemv_q4` (non-MLX mini banks).
pub fn cast_bf16_to_f32(
    gpu: &GemmaGpu,
    src_bf16: &GpuBuffer,
    dst_f32: &GpuBuffer,
    n: u32,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let p = gpu.rt.pipeline("cast_bf16_to_f32").map_err(map_metal)?;
    let n_off = gpu.icb_scalars.push_u32(n)?;
    dispatch_1d(&gpu.rt, &p, n as usize, |bnd| {
        set_gpu_buf(bnd, src_bf16, 0);
        set_gpu_buf(bnd, dst_f32, 1);
        gpu.icb_scalars.bind_u32(bnd, n_off, 2);
    })
    .map_err(map_metal)
}

/// Cast f32 activations into [`GemmaGpu::act_bf16_scratch`] for simd GEMV.
pub fn prepare_act_bf16(gpu: &GemmaGpu, x_f32: &GpuBuffer, n: u32) -> Result<GpuBuffer> {
    let dst = gpu.act_bf16_scratch(n as usize)?;
    cast_f32_to_bf16(gpu, x_f32, &dst, n)?;
    Ok(dst)
}

/// Upload a [`QuantMatrix`] into Hot-resident GPU banks (decode weights).
/// Mid-size Q4Mlx defaults to Interleaved4 (`GEMMA_METAL_GEMV_INTERLEAVE=0` → row-major;
/// `GEMMA_METAL_GEMV_BLOCKED=1` → BlockedBn16). Q4Mlx scale+bias → interleaved bfloat2.
pub fn upload_quant_hot(gpu: &GemmaGpu, w: &QuantMatrix) -> Result<HotQuantBanks> {
    let group_size =
        w.scheme
            .group_size()
            .ok_or_else(|| Error::Metal("upload_quant_hot: need Q4/Q8".into()))? as u32;

    let use_blocked = matches!(w.scheme, crate::quant::QuantScheme::Q4Mlx { .. })
        && prefer_blocked_q4_mlx(w.rows, w.cols, group_size as usize);
    let use_interleaved = !use_blocked
        && matches!(w.scheme, crate::quant::QuantScheme::Q4Mlx { .. })
        && prefer_interleaved_q4_mlx(w.rows, w.cols, group_size as usize);

    let repacked = if use_blocked {
        Some((
            HotGemvLayout::BlockedBn16,
            repack_q4_mlx_blocked(
                &w.packed,
                &w.scales,
                &w.zeros,
                w.rows,
                w.cols,
                group_size as usize,
            ),
        ))
    } else if use_interleaved {
        Some((
            HotGemvLayout::Interleaved4,
            repack_q4_mlx_interleaved4(
                &w.packed,
                &w.scales,
                &w.zeros,
                w.rows,
                w.cols,
                group_size as usize,
            ),
        ))
    } else {
        None
    };
    let layout = repacked
        .as_ref()
        .map(|(l, _)| *l)
        .unwrap_or(HotGemvLayout::RowMajor);
    let (packed_bytes, scale_f32, zero_f32): (&[u8], &[f32], &[f32]) = match &repacked {
        Some((_, (p, s, z))) => (p.as_slice(), s.as_slice(), z.as_slice()),
        None => (w.packed.as_slice(), w.scales.as_slice(), w.zeros.as_slice()),
    };

    let packed_n = packed_bytes.len().max(1);
    // Q4Mlx Hot: interleaved bfloat2 scale+bias in `scales`; `zeros` is an ABI stub.
    let scales_bf16 = matches!(w.scheme, crate::quant::QuantScheme::Q4Mlx { .. });
    let sb_bits = if scales_bf16 {
        Some(pack_mlx_sb_bf16(scale_f32, zero_f32))
    } else {
        None
    };
    let (scales_n, zeros_n) = if let Some(ref sb) = sb_bits {
        (sb.len().max(1) * 2, 4usize)
    } else {
        (scale_f32.len().max(1) * 4, zero_f32.len().max(1) * 4)
    };
    if w.nbytes_hot() >= 256 * 1024 {
        diag::log(
            "kernels",
            format_args!(
                "upload_quant_hot [{},{}] scheme={:?} layout={layout:?} packed={} scales={} zeros={} scale_dtype={}",
                w.rows,
                w.cols,
                w.scheme,
                diag::fmt_bytes(packed_n as u64),
                diag::fmt_bytes(scales_n as u64),
                diag::fmt_bytes(zeros_n as u64),
                if scales_bf16 { "bf16_sb_interleaved" } else { "f32" }
            ),
        );
    }
    let packed = gpu.rt.alloc_buffer_hot(packed_n).map_err(|e| {
        diag::err_msg(
            "kernels",
            &format!("alloc_buffer_hot packed n={packed_n}"),
            &e,
        );
        map_metal(e)
    })?;
    packed.write_bytes(packed_bytes);
    let scales = gpu.rt.alloc_buffer_hot(scales_n).map_err(|e| {
        diag::err_msg(
            "kernels",
            &format!("alloc_buffer_hot scales n={scales_n}"),
            &e,
        );
        map_metal(e)
    })?;
    let zeros = gpu.rt.alloc_buffer_hot(zeros_n).map_err(|e| {
        diag::err_msg(
            "kernels",
            &format!("alloc_buffer_hot zeros n={zeros_n}"),
            &e,
        );
        map_metal(e)
    })?;
    if let Some(sb) = sb_bits {
        scales.write_bf16_bits(&sb);
    } else {
        scales.write_f32(scale_f32);
        zeros.write_f32(zero_f32);
    }
    Ok(HotQuantBanks {
        scheme: w.scheme,
        layout,
        rows: w.rows as u32,
        cols: w.cols as u32,
        group_size,
        packed,
        scales,
        zeros,
    })
}

/// Hot-resident Q4/Q8 matrix for decode GEMV.
pub struct HotQuantBanks {
    pub scheme: crate::quant::QuantScheme,
    pub layout: HotGemvLayout,
    pub rows: u32,
    pub cols: u32,
    pub group_size: u32,
    pub packed: GpuBuffer,
    pub scales: GpuBuffer,
    pub zeros: GpuBuffer,
}

impl HotQuantBanks {
    pub fn gemv(&self, gpu: &GemmaGpu, x: &GpuBuffer, y: &GpuBuffer) -> Result<()> {
        self.gemv_impl(gpu, x, y, false)
    }

    /// Like [`Self::gemv`], but `x` is already bf16 for the simd Q4Mlx path.
    pub fn gemv_bf16_x(&self, gpu: &GemmaGpu, x_bf16: &GpuBuffer, y: &GpuBuffer) -> Result<()> {
        self.gemv_impl(gpu, x_bf16, y, true)
    }

    fn gemv_impl(
        &self,
        gpu: &GemmaGpu,
        x: &GpuBuffer,
        y: &GpuBuffer,
        x_is_bf16: bool,
    ) -> Result<()> {
        match self.scheme {
            crate::quant::QuantScheme::Q4 { .. } => {
                // Classic gemv_q4 reads `float *x`. fuse_bf16 producers pass bf16;
                // expand first so we never reinterpret / over-read the bf16 slab
                // as f32 (tape replay ~cmd 19 add_inplace blow-up, 2026-07-19).
                let x_f32;
                let x_ref = if x_is_bf16 {
                    if tessl::ab_flags::need_barrier(true) {
                        gpu.barrier()?;
                    }
                    x_f32 = gpu.act_f32_scratch(self.cols as usize)?;
                    cast_bf16_to_f32(gpu, x, &x_f32, self.cols)?;
                    if tessl::ab_flags::need_barrier(true) {
                        gpu.barrier()?;
                    }
                    &x_f32
                } else {
                    x
                };
                gemv_q4(
                    gpu,
                    &self.packed,
                    &self.scales,
                    &self.zeros,
                    x_ref,
                    y,
                    self.rows,
                    self.cols,
                    self.group_size,
                )
            }
            crate::quant::QuantScheme::Q4Mlx { .. } => match self.layout {
                // Blocked kernels read `float *x` (not bf16 like simd). Expand
                // fuse_bf16 activations — same trap as classic Q4 (2026-07-19).
                HotGemvLayout::BlockedBn16 => {
                    let x_f32;
                    let x_ref = if x_is_bf16 {
                        if tessl::ab_flags::need_barrier(true) {
                            gpu.barrier()?;
                        }
                        x_f32 = gpu.act_f32_scratch(self.cols as usize)?;
                        cast_bf16_to_f32(gpu, x, &x_f32, self.cols)?;
                        if tessl::ab_flags::need_barrier(true) {
                            gpu.barrier()?;
                        }
                        &x_f32
                    } else {
                        x
                    };
                    gemv_q4_mlx_blocked(
                        gpu,
                        &self.packed,
                        &self.scales,
                        &self.zeros,
                        x_ref,
                        y,
                        self.rows,
                        self.cols,
                        self.group_size,
                    )
                }
                HotGemvLayout::RowMajor | HotGemvLayout::Interleaved4 => {
                    let interleaved = self.layout == HotGemvLayout::Interleaved4;
                    if x_is_bf16 && gemv_simd_enabled() && self.cols >= 256 && self.cols % 16 == 0 {
                        dispatch_gemv_simd(
                            gpu,
                            &self.packed,
                            &self.scales,
                            &self.zeros,
                            x,
                            y,
                            self.rows,
                            self.cols,
                            self.group_size,
                            interleaved,
                        )
                    } else if interleaved {
                        // Interleaved4 banks require the i4 simd kernel (no row-major peel fallback).
                        if tessl::ab_flags::need_barrier(true) {
                            gpu.barrier()?;
                        }
                        let x_bf16 = if x_is_bf16 {
                            None
                        } else {
                            Some(prepare_act_bf16(gpu, x, self.cols)?)
                        };
                        let x_ref = x_bf16.as_ref().unwrap_or(x);
                        if tessl::ab_flags::need_barrier(true) {
                            gpu.barrier()?;
                        }
                        dispatch_gemv_simd(
                            gpu,
                            &self.packed,
                            &self.scales,
                            &self.zeros,
                            x_ref,
                            y,
                            self.rows,
                            self.cols,
                            self.group_size,
                            true,
                        )
                    } else {
                        // Float peel (`gemv_q4_mlx_wide` / tall). Must not feed bf16
                        // bits into `float *x` when simd is off / shape-ineligible.
                        let x_f32;
                        let x_ref = if x_is_bf16 {
                            if tessl::ab_flags::need_barrier(true) {
                                gpu.barrier()?;
                            }
                            x_f32 = gpu.act_f32_scratch(self.cols as usize)?;
                            cast_bf16_to_f32(gpu, x, &x_f32, self.cols)?;
                            if tessl::ab_flags::need_barrier(true) {
                                gpu.barrier()?;
                            }
                            &x_f32
                        } else {
                            x
                        };
                        gemv_q4_mlx(
                            gpu,
                            &self.packed,
                            &self.scales,
                            &self.zeros,
                            x_ref,
                            y,
                            self.rows,
                            self.cols,
                            self.group_size,
                        )
                    }
                }
            },
            crate::quant::QuantScheme::Q8 { .. } => gemv_q8(
                gpu,
                &self.packed,
                &self.scales,
                &self.zeros,
                x,
                y,
                self.rows,
                self.cols,
                self.group_size,
            ),
            crate::quant::QuantScheme::Bf16 => {
                Err(Error::Metal("HotQuantBanks::gemv: Bf16 unsupported".into()))
            }
        }
    }

    /// True when gate/up/down can run bounded-TG persistent gate→down (Hot Q4).
    ///
    /// Shapes: gate/up `[n_mid × n_hidden]`, down `[n_hidden × n_mid]`
    /// (`rows` = GEMV output length, `cols` = GEMV K).
    pub fn can_fuse_gate_down(&self, up: &Self, down: &Self) -> bool {
        self.can_fuse_gate_up_gelu(up)
            && matches!(self.scheme, crate::quant::QuantScheme::Q4Mlx { .. })
            && matches!(down.scheme, crate::quant::QuantScheme::Q4Mlx { .. })
            && down.layout == self.layout
            // Peel is RowMajor-only (`pi_q4_*_tile_rm`); I4 would silently
            // misread packs. Shipping fusion_ab pins GEMV_INTERLEAVE=0.
            && down.layout == HotGemvLayout::RowMajor
            && down.rows == self.cols // n_hidden
            && down.cols == self.rows // n_mid
            && down.group_size == self.group_size
            && fuse_gate_down_enabled()
            && fuse_bf16_mlp()
            && gemv_simd_enabled()
            && self.cols >= 256
            && self.cols % 16 == 0
            && self.rows % self.group_size == 0
            && self.cols % self.group_size == 0
    }

    /// True when this bank can pair with `other` for fused gate∥up→gelu.
    pub fn can_fuse_gate_up_gelu(&self, other: &Self) -> bool {
        matches!(self.scheme, crate::quant::QuantScheme::Q4Mlx { .. })
            && matches!(other.scheme, crate::quant::QuantScheme::Q4Mlx { .. })
            && self.layout == other.layout
            && matches!(
                self.layout,
                HotGemvLayout::BlockedBn16 | HotGemvLayout::RowMajor | HotGemvLayout::Interleaved4
            )
            && self.rows == other.rows
            && self.cols == other.cols
            && self.group_size == other.group_size
            && fuse_gate_up_enabled()
            && (self.layout == HotGemvLayout::BlockedBn16
                || (self.cols >= 256 && self.cols % 16 == 0 && gemv_simd_enabled()))
    }

    /// True when this bank can pair with `other` for fused producer K∥V (shared x).
    pub fn can_fuse_kv(&self, other: &Self) -> bool {
        matches!(self.scheme, crate::quant::QuantScheme::Q4Mlx { .. })
            && matches!(other.scheme, crate::quant::QuantScheme::Q4Mlx { .. })
            && matches!(
                self.layout,
                HotGemvLayout::RowMajor | HotGemvLayout::Interleaved4
            )
            && self.layout == other.layout
            && self.rows == other.rows
            && self.cols == other.cols
            && self.group_size == other.group_size
            && fuse_kv_enabled()
            && gemv_simd_enabled()
            && self.cols >= 256
            && self.cols % 16 == 0
    }

    /// Layer-fusion v1: can `self` (q_proj) run fused with `k` / `v` in one
    /// `gemv_q4_mlx_simd_qkv` / `_qkv_i4` dispatch?
    ///
    /// RowMajor → `gemv_q4_mlx_simd_qkv`; Interleaved4 → `gemv_q4_mlx_simd_qkv_i4`.
    /// All three banks must share the same layout. Rows may differ between q and
    /// k/v (GQA); cols / group_size must match because all three consume the same `x`.
    pub fn can_fuse_qkv(&self, k: &Self, v: &Self) -> bool {
        matches!(self.scheme, crate::quant::QuantScheme::Q4Mlx { .. })
            && matches!(k.scheme, crate::quant::QuantScheme::Q4Mlx { .. })
            && matches!(v.scheme, crate::quant::QuantScheme::Q4Mlx { .. })
            && matches!(
                self.layout,
                HotGemvLayout::RowMajor | HotGemvLayout::Interleaved4
            )
            && k.layout == self.layout
            && v.layout == self.layout
            && k.rows == v.rows
            && self.cols == k.cols
            && self.cols == v.cols
            && self.group_size == k.group_size
            && self.group_size == v.group_size
            && fuse_qkv_enabled()
            && gemv_simd_enabled()
            && self.cols >= 256
            && self.cols % 16 == 0
    }

    /// Accumulate residual: `resid[row] += (W @ x)[row]`.
    /// Uses fused simd when possible; otherwise `scratch` must be a distinct buffer.
    /// When `x_is_bf16`, skip the f32→bf16 cast (caller already prepared).
    pub fn gemv_add_into(
        &self,
        gpu: &GemmaGpu,
        x: &GpuBuffer,
        resid: &GpuBuffer,
        scratch: &GpuBuffer,
    ) -> Result<()> {
        self.gemv_add_into_impl(gpu, x, resid, scratch, false)
    }

    pub fn gemv_add_into_bf16_x(
        &self,
        gpu: &GemmaGpu,
        x_bf16: &GpuBuffer,
        resid: &GpuBuffer,
        scratch: &GpuBuffer,
    ) -> Result<()> {
        self.gemv_add_into_impl(gpu, x_bf16, resid, scratch, true)
    }

    /// Gemma4 dual-norm residual: `resid += post_ln(W @ x_bf16)`.
    ///
    /// Writes the raw projection into `proj_scratch`, then one fused
    /// `rms_norm_residual_add` (vs separate rms + residual_add).
    /// When `layer_scale != 1`, folds end-of-layer scale into the residual
    /// (`resid = scale * (resid + post_ln(proj))`) and the caller must skip
    /// the standalone `scale_f32_inplace`.
    pub fn gemv_postnorm_add_into_bf16_x(
        &self,
        gpu: &GemmaGpu,
        x_bf16: &GpuBuffer,
        resid: &GpuBuffer,
        proj_scratch: &GpuBuffer,
        post_norm_weight: &GpuBuffer,
        eps: f32,
    ) -> Result<()> {
        self.gemv_postnorm_add_into_bf16_x_scaled(
            gpu,
            x_bf16,
            resid,
            proj_scratch,
            post_norm_weight,
            eps,
            1.0,
        )
    }

    pub fn gemv_postnorm_add_into_bf16_x_scaled(
        &self,
        gpu: &GemmaGpu,
        x_bf16: &GpuBuffer,
        resid: &GpuBuffer,
        proj_scratch: &GpuBuffer,
        post_norm_weight: &GpuBuffer,
        eps: f32,
        layer_scale: f32,
    ) -> Result<()> {
        self.gemv_bf16_x(gpu, x_bf16, proj_scratch)?;
        if tessl::ab_flags::need_barrier(true) {
            gpu.barrier()?;
        }
        rms_norm_residual_add_f32_scaled(
            gpu,
            proj_scratch,
            post_norm_weight,
            resid,
            1,
            self.rows,
            eps,
            layer_scale,
        )
    }

    /// Thin Q4 GEMM `Y[M, rows] = X[M, cols] @ W^T` (bf16 X). Falls back to M× GEMV
    /// when simd GEMM is unavailable for this bank.
    pub fn gemm_bf16_x(
        &self,
        gpu: &GemmaGpu,
        x_bf16: &GpuBuffer,
        y: &GpuBuffer,
        m: u32,
    ) -> Result<()> {
        if m == 0 {
            return Ok(());
        }
        if self.can_gemm_simd() {
            return gemm_q4_mlx_simd(
                gpu,
                &self.packed,
                &self.scales,
                &self.zeros,
                x_bf16,
                y,
                self.rows,
                self.cols,
                self.group_size,
                m,
                self.layout == HotGemvLayout::Interleaved4,
            );
        }
        if m == 1 {
            return self.gemv_bf16_x(gpu, x_bf16, y);
        }
        for mi in 0..m {
            self.gemm_fallback_one(gpu, x_bf16, y, mi)?;
        }
        Ok(())
    }

    fn gemm_fallback_one(
        &self,
        gpu: &GemmaGpu,
        x_bf16: &GpuBuffer,
        y: &GpuBuffer,
        mi: u32,
    ) -> Result<()> {
        use tessl::dispatch::set_gpu_buf_offset;
        // Row-major peel GEMV with x/y byte offsets.
        match self.scheme {
            crate::quant::QuantScheme::Q4Mlx { .. } => {
                let interleaved = self.layout == HotGemvLayout::Interleaved4;
                let entry = if interleaved {
                    "gemv_q4_mlx_simd_i4"
                } else if matches!(self.layout, HotGemvLayout::BlockedBn16) {
                    return Err(Error::Metal(
                        "gemm fallback: BlockedBn16 needs blocked path".into(),
                    ));
                } else {
                    "gemv_q4_mlx_simd"
                };
                let p = gpu.rt.pipeline(entry).map_err(map_metal)?;
                let (tptg, n_tg) = simd_tg_geometry(self.rows);
                let x_off = (mi as usize) * (self.cols as usize) * 2;
                let y_off = (mi as usize) * (self.rows as usize) * 4;
                let (rows_off, cols_off, gs_off) =
                    push_gemv_dims(gpu, self.rows, self.cols, self.group_size)?;
                gpu.rt
                    .with_binder(|bnd| {
                        bnd.set_pipeline(&p);
                        set_gpu_buf(bnd, &self.packed, 0);
                        set_gpu_buf(bnd, &self.scales, 1);
                        set_gpu_buf(bnd, &self.zeros, 2);
                        set_gpu_buf_offset(bnd, x_bf16, x_off, 3);
                        set_gpu_buf_offset(bnd, y, y_off, 4);
                        gpu.icb_scalars.bind_u32(bnd, rows_off, 5);
                        gpu.icb_scalars.bind_u32(bnd, cols_off, 6);
                        gpu.icb_scalars.bind_u32(bnd, gs_off, 7);
                        bnd.dispatch(
                            tessl::runtime::mtl_size(n_tg, 1, 1),
                            tessl::runtime::mtl_size(tptg, 1, 1),
                        );
                        Ok(())
                    })
                    .map_err(map_metal)
            }
            _ => Err(Error::Metal("gemm fallback: need Q4Mlx".into())),
        }
    }

    pub fn can_gemm_simd(&self) -> bool {
        // Product widths only: mini H=256 loses to M×GEMV under always-on barriers.
        matches!(self.scheme, crate::quant::QuantScheme::Q4Mlx { .. })
            && matches!(
                self.layout,
                HotGemvLayout::RowMajor | HotGemvLayout::Interleaved4
            )
            && gemv_simd_enabled()
            && self.cols > 256
            && self.cols % 16 == 0
            && self.rows > 0
    }

    /// Residual-add GEMM into `resid` (same buffer in/out), M rows.
    pub fn gemm_add_into_bf16_x(
        &self,
        gpu: &GemmaGpu,
        x_bf16: &GpuBuffer,
        resid: &GpuBuffer,
        m: u32,
    ) -> Result<()> {
        if m == 0 {
            return Ok(());
        }
        if self.can_gemm_simd() {
            return gemm_q4_mlx_simd_add(
                gpu,
                &self.packed,
                &self.scales,
                &self.zeros,
                x_bf16,
                resid,
                resid,
                self.rows,
                self.cols,
                self.group_size,
                m,
                self.layout == HotGemvLayout::Interleaved4,
            );
        }
        for mi in 0..m {
            self.gemm_add_fallback_one(gpu, x_bf16, resid, mi)?;
        }
        Ok(())
    }

    /// Gemma4 dual-norm residual at M>1: `resid[m, rows] += post_ln(W @ x)`.
    pub fn gemm_postnorm_add_into_bf16_x(
        &self,
        gpu: &GemmaGpu,
        x_bf16: &GpuBuffer,
        resid: &GpuBuffer,
        proj_scratch: &GpuBuffer,
        post_norm_weight: &GpuBuffer,
        m: u32,
        eps: f32,
    ) -> Result<()> {
        self.gemm_postnorm_add_into_bf16_x_scaled(
            gpu,
            x_bf16,
            resid,
            proj_scratch,
            post_norm_weight,
            m,
            eps,
            1.0,
        )
    }

    pub fn gemm_postnorm_add_into_bf16_x_scaled(
        &self,
        gpu: &GemmaGpu,
        x_bf16: &GpuBuffer,
        resid: &GpuBuffer,
        proj_scratch: &GpuBuffer,
        post_norm_weight: &GpuBuffer,
        m: u32,
        eps: f32,
        layer_scale: f32,
    ) -> Result<()> {
        if m == 0 {
            return Ok(());
        }
        self.gemm_bf16_x(gpu, x_bf16, proj_scratch, m)?;
        if tessl::ab_flags::need_barrier(true) {
            gpu.barrier()?;
        }
        rms_norm_residual_add_f32_scaled(
            gpu,
            proj_scratch,
            post_norm_weight,
            resid,
            m,
            self.rows,
            eps,
            layer_scale,
        )
    }

    fn gemm_add_fallback_one(
        &self,
        gpu: &GemmaGpu,
        x_bf16: &GpuBuffer,
        resid: &GpuBuffer,
        mi: u32,
    ) -> Result<()> {
        use tessl::dispatch::set_gpu_buf_offset;
        let interleaved = self.layout == HotGemvLayout::Interleaved4;
        let entry = if interleaved {
            "gemv_q4_mlx_simd_add_i4"
        } else {
            "gemv_q4_mlx_simd_add"
        };
        let p = gpu.rt.pipeline(entry).map_err(map_metal)?;
        let (tptg, n_tg) = simd_tg_geometry(self.rows);
        let x_off = (mi as usize) * (self.cols as usize) * 2;
        let y_off = (mi as usize) * (self.rows as usize) * 4;
        let (rows_off, cols_off, gs_off) =
            push_gemv_dims(gpu, self.rows, self.cols, self.group_size)?;
        gpu.rt
            .with_binder(|bnd| {
                bnd.set_pipeline(&p);
                set_gpu_buf(bnd, &self.packed, 0);
                set_gpu_buf(bnd, &self.scales, 1);
                set_gpu_buf(bnd, &self.zeros, 2);
                set_gpu_buf_offset(bnd, x_bf16, x_off, 3);
                set_gpu_buf_offset(bnd, resid, y_off, 4);
                gpu.icb_scalars.bind_u32(bnd, rows_off, 5);
                gpu.icb_scalars.bind_u32(bnd, cols_off, 6);
                gpu.icb_scalars.bind_u32(bnd, gs_off, 7);
                set_gpu_buf_offset(bnd, resid, y_off, 8);
                bnd.dispatch(
                    tessl::runtime::mtl_size(n_tg, 1, 1),
                    tessl::runtime::mtl_size(tptg, 1, 1),
                );
                Ok(())
            })
            .map_err(map_metal)
    }

    fn gemv_add_into_impl(
        &self,
        gpu: &GemmaGpu,
        x: &GpuBuffer,
        resid: &GpuBuffer,
        scratch: &GpuBuffer,
        x_is_bf16: bool,
    ) -> Result<()> {
        match self.scheme {
            crate::quant::QuantScheme::Q4Mlx { .. }
                if matches!(
                    self.layout,
                    HotGemvLayout::RowMajor | HotGemvLayout::Interleaved4
                ) && gemv_simd_enabled()
                    && self.cols >= 256
                    && self.cols % 16 == 0
                    && self.rows > 0 =>
            {
                let x_bf16;
                let x_ref = if x_is_bf16 {
                    x
                } else {
                    if tessl::ab_flags::need_barrier(true) {
                        gpu.barrier()?;
                    }
                    x_bf16 = prepare_act_bf16(gpu, x, self.cols)?;
                    if tessl::ab_flags::need_barrier(true) {
                        gpu.barrier()?;
                    }
                    &x_bf16
                };
                gemv_q4_mlx_simd_add(
                    gpu,
                    &self.packed,
                    &self.scales,
                    &self.zeros,
                    x_ref,
                    resid,
                    resid,
                    self.rows,
                    self.cols,
                    self.group_size,
                    self.layout == HotGemvLayout::Interleaved4,
                )
            }
            _ => {
                if x_is_bf16 {
                    self.gemv_bf16_x(gpu, x, scratch)?;
                } else {
                    self.gemv(gpu, x, scratch)?;
                }
                // RAW: scratch → resid add (phase-edge even under coarse).
                if tessl::ab_flags::need_barrier(true) {
                    gpu.barrier()?;
                }
                add_inplace_f32(gpu, resid, scratch, self.rows)
            }
        }
    }
}

fn fuse_gate_up_enabled() -> bool {
    // Default ON after precise::tanh + inner clamp fixed fast_tanh NaNs in gelu.
    // Set GEMMA_METAL_FUSE_MLP=0 to force unfused gate∥up → mlp_gelu_tanh.
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(
        || match std::env::var("GEMMA_METAL_FUSE_MLP").ok().as_deref() {
            Some("0") | Some("false") | Some("off") => false,
            _ => true,
        },
    )
}

/// Fuse Gemma4 dual-norm tails: `proj → post_ln → resid +=` into gemv + one
/// `rms_norm_residual_add` (saves rms + residual dispatches/barriers on 31B).
/// Default ON; set `GEMMA_METAL_FUSE_DUAL_NORM=0` to restore the 3-op path.
pub fn fuse_dual_norm_enabled() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(
        || match std::env::var("GEMMA_METAL_FUSE_DUAL_NORM").ok().as_deref() {
            Some("0") | Some("false") | Some("off") => false,
            _ => true,
        },
    )
}

/// Producers emit bf16 (rms / FA / gelu → act scratch), killing per-layer cast passes.
/// Default ON; set `GEMMA_METAL_FUSE_BF16=0` to restore cast_f32_to_bf16.
/// Debug slices: `GEMMA_METAL_FUSE_BF16=rms|fa|mlp` enables only that producer.
pub fn fuse_bf16_enabled() -> bool {
    matches!(fuse_bf16_mode(), Some("all"))
}

fn fuse_bf16_mode() -> Option<&'static str> {
    static V: OnceLock<Option<&'static str>> = OnceLock::new();
    *V.get_or_init(
        || match std::env::var("GEMMA_METAL_FUSE_BF16").ok().as_deref() {
            Some("rms") => Some("rms"),
            Some("fa") => Some("fa"),
            Some("mlp") => Some("mlp"),
            Some("0") | Some("false") | Some("off") => None,
            _ => Some("all"),
        },
    )
}

#[inline]
pub fn fuse_bf16_rms() -> bool {
    matches!(fuse_bf16_mode(), Some("all") | Some("rms"))
}

#[inline]
pub fn fuse_bf16_fa() -> bool {
    matches!(fuse_bf16_mode(), Some("all") | Some("fa"))
}

#[inline]
pub fn fuse_bf16_mlp() -> bool {
    matches!(fuse_bf16_mode(), Some("all") | Some("mlp"))
}

/// Fused producer K∥V GEMV (shared x). Default on; set `GEMMA_METAL_FUSE_KV=0` to disable.
fn fuse_kv_enabled() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(
        || match std::env::var("GEMMA_METAL_FUSE_KV").ok().as_deref() {
            Some("0") | Some("false") | Some("off") => false,
            _ => true,
        },
    )
}

// --- Layer-fusion v1 (opt-in) ----------------------------------------------
//
// Rationale: `docs/audit_deep_2026-07-18.md` F2 measured a fixed cost of
// ~37 µs per dispatch (mini: 746 tok/s over ~36 dispatches, barrier mode
// ±1%), i.e. ~17 ms of a ~42 ms E4B token is launch tax over ~460 dispatches.
// These fusions cut dispatch count without changing kernel math.
//
// **Default OFF.** The unfused path stays the shipping baseline until an A/B
// on real weights shows both bit-exact tokens and a tok/s win.
//   GEMMA_METAL_FUSE_LAYER=1   enable every layer fusion below
//   GEMMA_METAL_FUSE_QKV=0|1   fused producer Q∥K∥V GEMV   (override)
//   GEMMA_METAL_FUSE_PLE=0|1   fused PLE lookup + residual (override)

fn env_on(name: &str) -> Option<bool> {
    match std::env::var(name).ok().as_deref() {
        Some("1") | Some("true") | Some("on") => Some(true),
        Some("0") | Some("false") | Some("off") => Some(false),
        _ => None,
    }
}

// Fusion flags use the settable AtomicI8 pattern (like ENCODE_ONCE /
// PERSISTENT_INTERP below), NOT OnceLock: a OnceLock freezes at the first
// call anywhere in the process, so tests that `env::set_var` after another
// test already touched the decode path read a stale `false` (2026-07-19
// failure: gemv_q4_mlx_simd_qkv_* asserts on can_fuse_qkv). Relaxed atomic
// load per call keeps the F4 host-tax win (no environ re-read).

/// -1 = read env once, 0 = off, 1 = on. Tests use [`set_fuse_layer`].
static FUSE_LAYER: AtomicI8 = AtomicI8::new(-1);
static FUSE_QKV: AtomicI8 = AtomicI8::new(-1);
static FUSE_PLE: AtomicI8 = AtomicI8::new(-1);
static FUSE_ROPE_KV: AtomicI8 = AtomicI8::new(-1);
static FUSE_GATE_DOWN: AtomicI8 = AtomicI8::new(-1);

fn flag_cached(cell: &AtomicI8, env_name: &str, default: impl Fn() -> bool) -> bool {
    let v = cell.load(Ordering::Relaxed);
    if v >= 0 {
        return v == 1;
    }
    let on = env_on(env_name).unwrap_or_else(default);
    cell.store(if on { 1 } else { 0 }, Ordering::Relaxed);
    on
}

/// Force master layer-fusion opt-in (tests / harness). Overrides env.
pub fn set_fuse_layer(on: bool) {
    FUSE_LAYER.store(if on { 1 } else { 0 }, Ordering::Relaxed);
}
/// Force fused Q∥K∥V opt-in (tests / harness). Overrides env.
pub fn set_fuse_qkv(on: bool) {
    FUSE_QKV.store(if on { 1 } else { 0 }, Ordering::Relaxed);
}
/// Force fused PLE lookup+residual opt-in (tests / harness). Overrides env.
pub fn set_fuse_ple_residual(on: bool) {
    FUSE_PLE.store(if on { 1 } else { 0 }, Ordering::Relaxed);
}
/// Force fused rope+kv_store opt-in (tests / harness). Overrides env.
pub fn set_fuse_rope_kv(on: bool) {
    FUSE_ROPE_KV.store(if on { 1 } else { 0 }, Ordering::Relaxed);
}
/// Force bounded-TG Hot gate→down fusion (tests / harness). Overrides env.
pub fn set_fuse_gate_down(on: bool) {
    FUSE_GATE_DOWN.store(if on { 1 } else { 0 }, Ordering::Relaxed);
}

/// F5 (audit 2026-07-18): E4B currently runs the legacy pre-LN residual
/// algebra (no post-attn LN on the attention branch, norm-REPLACE on the MLP
/// residual, `layer_scalar` dropped) because `use_gemma4_dual_norm` excludes
/// PLE layers. This opt-in lifts that exclusion so E4B takes the same
/// MLX-faithful dual-norm path as 31B (PLE residual still applies in its own
/// section). Default OFF — flip it for a `golden_parity` E4B run; real-weight
/// logit parity vs MLX decides whether this becomes the default.
static E4B_DUAL_NORM: AtomicI8 = AtomicI8::new(-1);

/// Force E4B dual-norm opt-in (tests / harness). Overrides env.
pub fn set_e4b_dual_norm(on: bool) {
    E4B_DUAL_NORM.store(if on { 1 } else { 0 }, Ordering::Relaxed);
}

/// Opt-in: Gemma4 dual-norm residual algebra on PLE (E4B) layers.
/// `GEMMA_METAL_E4B_DUAL_NORM=1`. Default OFF (legacy path unchanged).
pub fn e4b_dual_norm_enabled() -> bool {
    flag_cached(&E4B_DUAL_NORM, "GEMMA_METAL_E4B_DUAL_NORM", || false)
}

/// Master opt-in for layer fusion. Default OFF.
pub fn fuse_layer_enabled() -> bool {
    flag_cached(&FUSE_LAYER, "GEMMA_METAL_FUSE_LAYER", || false)
}

/// Fused producer Q∥K∥V GEMV — one dispatch for all three projections.
/// Saves 1 dispatch per producer layer.
pub fn fuse_qkv_enabled() -> bool {
    flag_cached(&FUSE_QKV, "GEMMA_METAL_FUSE_QKV", fuse_layer_enabled)
}

/// Fused PLE Q4 lookup + residual combine. Saves 1 dispatch per PLE layer
/// (E4B: every layer).
pub fn fuse_ple_residual_enabled() -> bool {
    flag_cached(&FUSE_PLE, "GEMMA_METAL_FUSE_PLE", fuse_layer_enabled)
}

/// Fused producer `rms_qkv_rope` + `kv_store_timestep_pair`. Saves 1 dispatch
/// per producer layer (shared-KV append still separate when needed).
pub fn fuse_rope_kv_enabled() -> bool {
    flag_cached(
        &FUSE_ROPE_KV,
        "GEMMA_METAL_FUSE_ROPE_KV",
        fuse_layer_enabled,
    )
}

/// Bounded-TG persistent gate→down replaces shipping gate_up_gelu + down add.
/// Default OFF; separate from [`persistent_interp_enabled`].
pub fn fuse_gate_down_enabled() -> bool {
    flag_cached(&FUSE_GATE_DOWN, "GEMMA_METAL_FUSE_GATE_DOWN", || false)
}

// --- Encode-once / CB-replay scaffolding (opt-in) ----------------------------
//
// Prerequisite for true CB/ICB replay: per-token scalars live in stable GPU
// buffers (`pos_buf`, `seed_tok`) instead of const-arena `set_u32`. The decode
// session already binds `rms_qkv_rope_posbuf`; this flag engages ping-pong
// bookkeeping (`PingPongCbReplay::mark_live_step`) + probes `try_replay_ready`
// (always NotWired — see `survey_cb_replay_api_gaps`). Metal 4 CB has no
// replay-prior-encoding API; mini ICB smoke is separate
// (`tessl::icb_smoke`, default OFF). Default OFF.
//
//   GEMMA_METAL_ENCODE_ONCE=1

/// -1 = read env once, 0 = off, 1 = on. Tests may call [`set_encode_once`].
static ENCODE_ONCE: AtomicI8 = AtomicI8::new(-1);

/// Force encode-once bookkeeping on/off (tests / harness). Overrides env.
pub fn set_encode_once(on: bool) {
    ENCODE_ONCE.store(if on { 1 } else { 0 }, Ordering::Relaxed);
}

/// Opt-in encode-once scaffold. Default OFF.
///
/// When on: session probes [`tessl::PingPongCbReplay::try_replay_ready`]
/// then advances the ledger via `mark_live_step` after each live decode encode.
/// Does **not** skip host encode (MTL4 CB has no replay API; ICB stub NotWired).
pub fn encode_once_enabled() -> bool {
    let v = ENCODE_ONCE.load(Ordering::Relaxed);
    if v >= 0 {
        return v == 1;
    }
    let on = env_on("GEMMA_METAL_ENCODE_ONCE").unwrap_or(false);
    ENCODE_ONCE.store(if on { 1 } else { 0 }, Ordering::Relaxed);
    on
}

// --- Persistent interpreter (mini / experimental, opt-in) --------------------
//
// Metal has no grid-wide forward-progress guarantee — a literal Hazy-style
// megakernel can deadlock if consumer TGs spin while producers are not
// resident. This lane prototypes the *pattern* (instruction stream + atomic
// deps) on tiny TG counts for the gate→down / FA→proj doctrine edges.
// Default OFF; never wires into 31B/E4B decode.
//
//   GEMMA_METAL_PERSISTENT_INTERP=1

/// -1 = read env once, 0 = off, 1 = on. Tests may call [`set_persistent_interp`].
static PERSISTENT_INTERP: AtomicI8 = AtomicI8::new(-1);

/// Force persistent-interpreter opt-in on/off (tests / harness). Overrides env.
pub fn set_persistent_interp(on: bool) {
    PERSISTENT_INTERP.store(if on { 1 } else { 0 }, Ordering::Relaxed);
}

/// Opt-in persistent-interpreter prototype. Default OFF.
///
/// When on **and** the graph is synthetic mini: [`crate::gpu_model::GpuDecodeSession`]
/// `step_inner` dispatches `persistent_interp_gate_down` /
/// `persistent_interp_fa_o_proj` once per layer on dedicated dense scratch
/// (doctrine edges FA→o_proj + gate→down). Shipping Q4 FA/MLP are unchanged.
/// Hot/E4B/31B: decode hook no-ops even if this returns true.
pub fn persistent_interp_enabled() -> bool {
    let v = PERSISTENT_INTERP.load(Ordering::Relaxed);
    if v >= 0 {
        return v == 1;
    }
    let on = env_on("GEMMA_METAL_PERSISTENT_INTERP").unwrap_or(false);
    PERSISTENT_INTERP.store(if on { 1 } else { 0 }, Ordering::Relaxed);
    on
}

/// Soft cap on TG count for the interpreter prototype (residency / FP safety).
pub const PERSISTENT_INTERP_MAX_TG: u32 = 8;
/// Threads per TG for the interpreter kernel.
pub const PERSISTENT_INTERP_TPTG: usize = 32;
/// Spin budget per grid barrier before setting the fail flag.
pub const PERSISTENT_INTERP_MAX_SPIN: u32 = 50_000_000;

/// Dispatch the mini gate→down persistent-interpreter stand-in.
///
/// Preconditions: `persistent_interp_enabled()`, `n_tg ∈ [1, PERSISTENT_INTERP_MAX_TG]`,
/// buffers sized for `n_mid` / `n_out`. `deps` is `u32×2` (arrival + generation),
/// zeroed by the caller; `fail` is `u32×1`, zeroed by the caller.
///
/// Returns `Err` if the flag is off or dims are out of the mini envelope.
/// After sync, check `fail` — non-zero means a barrier spin timed out
/// (Metal forward-progress caveat), not a math bug.
pub fn persistent_interp_gate_down(
    gpu: &GemmaGpu,
    insns: &GpuBuffer,
    n_insns: u32,
    gate: &GpuBuffer,
    up: &GpuBuffer,
    mid: &GpuBuffer,
    w_down: &GpuBuffer,
    out: &GpuBuffer,
    deps: &GpuBuffer,
    fail: &GpuBuffer,
    n_mid: u32,
    n_out: u32,
    n_tg: u32,
) -> Result<()> {
    if !persistent_interp_enabled() {
        return Err(Error::Metal(
            "persistent_interp_gate_down: GEMMA_METAL_PERSISTENT_INTERP off (default)".into(),
        ));
    }
    if n_mid == 0 || n_out == 0 {
        return Ok(());
    }
    if n_tg == 0 || n_tg > PERSISTENT_INTERP_MAX_TG {
        return Err(Error::Metal(format!(
            "persistent_interp: n_tg={n_tg} outside 1..={PERSISTENT_INTERP_MAX_TG} (mini only)"
        )));
    }
    if n_insns == 0 {
        return Err(Error::Metal(
            "persistent_interp: empty instruction stream".into(),
        ));
    }
    let entry = KernelId::PersistentInterpGateDown.entry_name();
    let p = gpu.rt.pipeline(entry).map_err(map_metal)?;
    gpu.rt
        .with_binder(|bnd| {
            bnd.set_pipeline(&p);
            set_gpu_buf(bnd, insns, 0);
            set_u32(bnd, n_insns, 1);
            set_gpu_buf(bnd, gate, 2);
            set_gpu_buf(bnd, up, 3);
            set_gpu_buf(bnd, mid, 4);
            set_gpu_buf(bnd, w_down, 5);
            set_gpu_buf(bnd, out, 6);
            set_gpu_buf(bnd, deps, 7);
            set_gpu_buf(bnd, fail, 8);
            set_u32(bnd, n_mid, 9);
            set_u32(bnd, n_out, 10);
            set_u32(bnd, n_tg, 11);
            set_u32(bnd, PERSISTENT_INTERP_MAX_SPIN, 12);
            bnd.dispatch(
                tessl::runtime::mtl_size(n_tg as usize, 1, 1),
                tessl::runtime::mtl_size(PERSISTENT_INTERP_TPTG, 1, 1),
            );
            Ok(())
        })
        .map_err(map_metal)
}

/// Hot Q4 bounded-TG gate→down: replaces shipping `gate_up_gelu` + `gemv_add_into`.
///
/// Opt-in [`fuse_gate_down_enabled`] (default OFF). Uses the same instruction
/// stream as the dense mini stand-in; `n_tg ∈ [1, PERSISTENT_INTERP_MAX_TG]`.
/// After sync, caller must check `fail` (barrier spin timeout).
pub fn persistent_interp_gate_down_q4(
    gpu: &GemmaGpu,
    insns: &GpuBuffer,
    n_insns: u32,
    gate: &HotQuantBanks,
    up: &HotQuantBanks,
    down: &HotQuantBanks,
    x_bf16: &GpuBuffer,
    mid: &GpuBuffer,
    x_out: &GpuBuffer,
    deps: &GpuBuffer,
    fail: &GpuBuffer,
    n_tg: u32,
    mid_as_bf16: bool,
) -> Result<()> {
    if !fuse_gate_down_enabled() {
        return Err(Error::Metal(
            "persistent_interp_gate_down_q4: GEMMA_METAL_FUSE_GATE_DOWN off (default)".into(),
        ));
    }
    if !gate.can_fuse_gate_down(up, down) {
        return Err(Error::Metal(
            "persistent_interp_gate_down_q4: gate/up/down layout mismatch".into(),
        ));
    }
    let n_mid = gate.rows;
    let n_out = down.rows;
    let cols = gate.cols;
    if n_mid == 0 || n_out == 0 {
        return Ok(());
    }
    if n_tg == 0 || n_tg > PERSISTENT_INTERP_MAX_TG {
        return Err(Error::Metal(format!(
            "persistent_interp_gate_down_q4: n_tg={n_tg} outside 1..={PERSISTENT_INTERP_MAX_TG}"
        )));
    }
    if n_insns == 0 {
        return Err(Error::Metal(
            "persistent_interp: empty instruction stream".into(),
        ));
    }
    let entry = KernelId::PersistentInterpGateDownQ4.entry_name();
    let p = gpu.rt.pipeline(entry).map_err(map_metal)?;
    let mid_bf16_flag = if mid_as_bf16 { 1u32 } else { 0u32 };
    gpu.rt
        .with_binder(|bnd| {
            bnd.set_pipeline(&p);
            set_gpu_buf(bnd, insns, 0);
            set_u32(bnd, n_insns, 1);
            set_gpu_buf(bnd, &gate.packed, 2);
            set_gpu_buf(bnd, &gate.scales, 3);
            set_gpu_buf(bnd, &gate.zeros, 4);
            set_gpu_buf(bnd, &up.packed, 5);
            set_gpu_buf(bnd, &up.scales, 6);
            set_gpu_buf(bnd, &up.zeros, 7);
            set_gpu_buf(bnd, x_bf16, 8);
            // Single mid slab (bf16 via cast in Metal) — do not dual-bind alias.
            set_gpu_buf(bnd, mid, 9);
            set_gpu_buf(bnd, &down.packed, 10);
            set_gpu_buf(bnd, &down.scales, 11);
            set_gpu_buf(bnd, &down.zeros, 12);
            set_gpu_buf(bnd, x_out, 13);
            set_gpu_buf(bnd, deps, 14);
            set_gpu_buf(bnd, fail, 15);
            set_u32(bnd, n_mid, 16);
            set_u32(bnd, n_out, 17);
            set_u32(bnd, cols, 18);
            set_u32(bnd, gate.group_size, 19);
            set_u32(bnd, n_tg, 20);
            set_u32(bnd, PERSISTENT_INTERP_MAX_SPIN, 21);
            set_u32(bnd, mid_bf16_flag, 22);
            bnd.dispatch(
                tessl::runtime::mtl_size(n_tg as usize, 1, 1),
                tessl::runtime::mtl_size(GEMV_SIMD_TPTG, 1, 1),
            );
            Ok(())
        })
        .map_err(map_metal)
}

/// Dispatch the mini FA→o_proj persistent-interpreter stand-in.
///
/// Same doctrine as [`persistent_interp_gate_down`]: instruction stream +
/// atomic grid barrier between a producer that fills `ctx` and a consumer
/// dense `o_proj` that needs *all* of `ctx`. FA itself is mocked as
/// `ctx[i] = tanh(q[i]*k[i]*scale)*v[i]` (element-local; not softmax FA).
///
/// Preconditions / fail semantics match [`persistent_interp_gate_down`].
pub fn persistent_interp_fa_o_proj(
    gpu: &GemmaGpu,
    insns: &GpuBuffer,
    n_insns: u32,
    q: &GpuBuffer,
    k: &GpuBuffer,
    v: &GpuBuffer,
    ctx: &GpuBuffer,
    w_o: &GpuBuffer,
    out: &GpuBuffer,
    deps: &GpuBuffer,
    fail: &GpuBuffer,
    n_ctx: u32,
    n_out: u32,
    n_tg: u32,
    scale: f32,
) -> Result<()> {
    if !persistent_interp_enabled() {
        return Err(Error::Metal(
            "persistent_interp_fa_o_proj: GEMMA_METAL_PERSISTENT_INTERP off (default)".into(),
        ));
    }
    if n_ctx == 0 || n_out == 0 {
        return Ok(());
    }
    if n_tg == 0 || n_tg > PERSISTENT_INTERP_MAX_TG {
        return Err(Error::Metal(format!(
            "persistent_interp: n_tg={n_tg} outside 1..={PERSISTENT_INTERP_MAX_TG} (mini only)"
        )));
    }
    if n_insns == 0 {
        return Err(Error::Metal(
            "persistent_interp: empty instruction stream".into(),
        ));
    }
    let entry = KernelId::PersistentInterpFaOProj.entry_name();
    let p = gpu.rt.pipeline(entry).map_err(map_metal)?;
    gpu.rt
        .with_binder(|bnd| {
            bnd.set_pipeline(&p);
            set_gpu_buf(bnd, insns, 0);
            set_u32(bnd, n_insns, 1);
            set_gpu_buf(bnd, q, 2);
            set_gpu_buf(bnd, k, 3);
            set_gpu_buf(bnd, v, 4);
            set_gpu_buf(bnd, ctx, 5);
            set_gpu_buf(bnd, w_o, 6);
            set_gpu_buf(bnd, out, 7);
            set_gpu_buf(bnd, deps, 8);
            set_gpu_buf(bnd, fail, 9);
            set_u32(bnd, n_ctx, 10);
            set_u32(bnd, n_out, 11);
            set_u32(bnd, n_tg, 12);
            set_u32(bnd, PERSISTENT_INTERP_MAX_SPIN, 13);
            set_f32(bnd, scale, 14);
            bnd.dispatch(
                tessl::runtime::mtl_size(n_tg as usize, 1, 1),
                tessl::runtime::mtl_size(PERSISTENT_INTERP_TPTG, 1, 1),
            );
            Ok(())
        })
        .map_err(map_metal)
}

/// Threadgroup count for one bank under the simd GEMV geometry (8 rows/TG).
/// Mirrors [`simd_tg_geometry`]'s ceil so fused banks tile identically.
fn simd_tg_count(rows: u32) -> u32 {
    let rows_per_tg = GEMV_SIMD_SG * GEMV_SIMD_ROWS;
    (rows + rows_per_tg - 1) / rows_per_tg
}

/// Fused producer Q∥K∥V simd GEMV: `q=Wq@x`, `k=Wk@x`, `v=Wv@x` in one launch.
///
/// Preconditions are checked by [`HotQuantBanks::can_fuse_qkv`]; callers must
/// gate on it. `x_bf16` is the shared post-`input_norm` activation.
///
/// RowMajor uses `gemv_q4_mlx_simd_qkv` (math ≡ `gemv_q4_mlx_simd`);
/// Interleaved4 uses `gemv_q4_mlx_simd_qkv_i4` (math ≡ `gemv_q4_mlx_simd_i4`).
/// Results must be bit-exact vs the unfused triple on the same layout.
pub fn gemv_q4_mlx_simd_qkv_bf16_x(
    gpu: &GemmaGpu,
    q: &HotQuantBanks,
    k: &HotQuantBanks,
    v: &HotQuantBanks,
    x_bf16: &GpuBuffer,
    q_out: &GpuBuffer,
    k_out: &GpuBuffer,
    v_out: &GpuBuffer,
) -> Result<()> {
    if q.rows == 0 || k.rows == 0 {
        return Ok(());
    }
    if !q.can_fuse_qkv(k, v) {
        return Err(Error::Metal(
            "simd_qkv: need matching RowMajor/Interleaved4 Q4Mlx q/k/v (or GEMMA_METAL_FUSE_QKV=0)"
                .into(),
        ));
    }
    let entry = if q.layout == HotGemvLayout::Interleaved4 {
        KernelId::GemvQ4MlxSimdQkvI4.entry_name()
    } else {
        KernelId::GemvQ4MlxSimdQkv.entry_name()
    };
    let p = gpu.rt.pipeline(entry).map_err(map_metal)?;
    let tg_q = simd_tg_count(q.rows);
    let tg_k = simd_tg_count(k.rows);
    let tg_v = simd_tg_count(v.rows);
    let n_tg = (tg_q + tg_k + tg_v) as usize;
    if trace_gemv_enabled() {
        eprintln!(
            "[trace] gemv entry={entry} rows_q={} rows_kv={} cols={} \
             tg={GEMV_SIMD_TPTG} groups={n_tg} (q={tg_q} k={tg_k} v={tg_v}) fused=qkv act=bf16",
            q.rows, k.rows, q.cols
        );
    }
    let q_rows_off = gpu.icb_scalars.push_u32(q.rows)?;
    let k_rows_off = gpu.icb_scalars.push_u32(k.rows)?;
    let cols_off = gpu.icb_scalars.push_u32(q.cols)?;
    let gs_off = gpu.icb_scalars.push_u32(q.group_size)?;
    let tg_q_off = gpu.icb_scalars.push_u32(tg_q)?;
    let tg_k_off = gpu.icb_scalars.push_u32(tg_k)?;
    gpu.rt
        .with_binder(|bnd| {
            bnd.set_pipeline(&p);
            set_gpu_buf(bnd, &q.packed, 0);
            set_gpu_buf(bnd, &q.scales, 1);
            set_gpu_buf(bnd, &k.packed, 2);
            set_gpu_buf(bnd, &k.scales, 3);
            set_gpu_buf(bnd, &v.packed, 4);
            set_gpu_buf(bnd, &v.scales, 5);
            set_gpu_buf(bnd, x_bf16, 6);
            set_gpu_buf(bnd, q_out, 7);
            set_gpu_buf(bnd, k_out, 8);
            set_gpu_buf(bnd, v_out, 9);
            gpu.icb_scalars.bind_u32(bnd, q_rows_off, 10);
            gpu.icb_scalars.bind_u32(bnd, k_rows_off, 11);
            gpu.icb_scalars.bind_u32(bnd, cols_off, 12);
            gpu.icb_scalars.bind_u32(bnd, gs_off, 13);
            gpu.icb_scalars.bind_u32(bnd, tg_q_off, 14);
            gpu.icb_scalars.bind_u32(bnd, tg_k_off, 15);
            bnd.dispatch(
                tessl::runtime::mtl_size(n_tg, 1, 1),
                tessl::runtime::mtl_size(GEMV_SIMD_TPTG, 1, 1),
            );
            Ok(())
        })
        .map_err(map_metal)
}

/// Fused PLE Q4 lookup + residual combine: `dst += combine_scale * lookup()`.
///
/// Replaces `ple_lookup_q4_mlx` + barrier + `ple_residual_add`. The caller must
/// still emit the RAW barrier *before* this call (`dst` is the residual stream,
/// written by the preceding o_proj residual).
#[allow(clippy::too_many_arguments)]
pub fn ple_lookup_q4_mlx_residual(
    gpu: &GemmaGpu,
    token_ids: &GpuBuffer,
    packed: &GpuBuffer,
    scales: &GpuBuffer,
    biases: &GpuBuffer,
    out: &GpuBuffer,
    dst: &GpuBuffer,
    dim: u32,
    vocab: u32,
    n: u32,
    scale: f32,
    combine_scale: f32,
    layer: u32,
    num_layers: u32,
    group_size: u32,
) -> Result<()> {
    let p = gpu
        .rt
        .pipeline(KernelId::PleLookupQ4MlxResidual.entry_name())
        .map_err(map_metal)?;
    let threads = (n * dim) as usize;
    let dim_off = gpu.icb_scalars.push_u32(dim)?;
    let vocab_off = gpu.icb_scalars.push_u32(vocab)?;
    let n_off = gpu.icb_scalars.push_u32(n)?;
    let scale_off = gpu.icb_scalars.push_f32(scale)?;
    let layer_off = gpu.icb_scalars.push_u32(layer)?;
    let nlayers_off = gpu.icb_scalars.push_u32(num_layers)?;
    let gs_off = gpu.icb_scalars.push_u32(group_size)?;
    let combine_off = gpu.icb_scalars.push_f32(combine_scale)?;
    dispatch_1d(&gpu.rt, &p, threads, |bnd| {
        set_gpu_buf(bnd, token_ids, 0);
        set_gpu_buf(bnd, packed, 1);
        set_gpu_buf(bnd, scales, 2);
        set_gpu_buf(bnd, biases, 3);
        set_gpu_buf(bnd, out, 4);
        gpu.icb_scalars.bind_u32(bnd, dim_off, 5);
        gpu.icb_scalars.bind_u32(bnd, vocab_off, 6);
        gpu.icb_scalars.bind_u32(bnd, n_off, 7);
        gpu.icb_scalars.bind_f32(bnd, scale_off, 8);
        gpu.icb_scalars.bind_u32(bnd, layer_off, 9);
        gpu.icb_scalars.bind_u32(bnd, nlayers_off, 10);
        gpu.icb_scalars.bind_u32(bnd, gs_off, 11);
        set_gpu_buf(bnd, dst, 12);
        gpu.icb_scalars.bind_f32(bnd, combine_off, 13);
    })
    .map_err(map_metal)
}

/// Fused gate∥up→gelu (writes `mid`). RowMajor uses simd; BlockedBn16 uses blocked.
/// `x` is f32 — cast once for the simd path.
pub fn gemv_q4_mlx_gate_up_gelu(
    gpu: &GemmaGpu,
    gate: &HotQuantBanks,
    up: &HotQuantBanks,
    x: &GpuBuffer,
    mid: &GpuBuffer,
) -> Result<()> {
    gemv_q4_mlx_gate_up_gelu_impl(gpu, gate, up, x, mid, false, false)
}

/// Like [`gemv_q4_mlx_gate_up_gelu`] but `x` is already bf16 for simd.
pub fn gemv_q4_mlx_gate_up_gelu_bf16_x(
    gpu: &GemmaGpu,
    gate: &HotQuantBanks,
    up: &HotQuantBanks,
    x_bf16: &GpuBuffer,
    mid: &GpuBuffer,
) -> Result<()> {
    gemv_q4_mlx_gate_up_gelu_impl(gpu, gate, up, x_bf16, mid, true, false)
}

/// Fused gate∥up→gelu writing bf16 mid into `mid_bf16` (act scratch).
pub fn gemv_q4_mlx_gate_up_gelu_bf16_x_out_bf16(
    gpu: &GemmaGpu,
    gate: &HotQuantBanks,
    up: &HotQuantBanks,
    x_bf16: &GpuBuffer,
    mid_bf16: &GpuBuffer,
) -> Result<()> {
    gemv_q4_mlx_gate_up_gelu_impl(gpu, gate, up, x_bf16, mid_bf16, true, true)
}

fn gemv_q4_mlx_gate_up_gelu_impl(
    gpu: &GemmaGpu,
    gate: &HotQuantBanks,
    up: &HotQuantBanks,
    x: &GpuBuffer,
    mid: &GpuBuffer,
    x_is_bf16: bool,
    mid_as_bf16: bool,
) -> Result<()> {
    if gate.rows == 0 {
        return Ok(());
    }
    if !gate.can_fuse_gate_up_gelu(up) {
        return Err(Error::Metal(
            "gate_up_gelu: need matching Q4Mlx layouts (or GEMMA_METAL_FUSE_MLP=0)".into(),
        ));
    }
    match gate.layout {
        HotGemvLayout::RowMajor | HotGemvLayout::Interleaved4 => {
            let x_bf16;
            let x_ref = if x_is_bf16 {
                x
            } else {
                if tessl::ab_flags::need_barrier(true) {
                    gpu.barrier()?;
                }
                x_bf16 = prepare_act_bf16(gpu, x, gate.cols)?;
                if tessl::ab_flags::need_barrier(true) {
                    gpu.barrier()?;
                }
                &x_bf16
            };
            gemv_q4_mlx_simd_gate_up_gelu(gpu, gate, up, x_ref, mid, mid_as_bf16)
        }
        HotGemvLayout::BlockedBn16 => {
            // `gemv_q4_mlx_blocked_gate_up_gelu` peels float x; Hot decode passes
            // bf16 under default FUSE_BF16 — expand before the fused dispatch.
            let x_f32;
            let x_ref = if x_is_bf16 {
                if tessl::ab_flags::need_barrier(true) {
                    gpu.barrier()?;
                }
                x_f32 = gpu.act_f32_scratch(gate.cols as usize)?;
                cast_bf16_to_f32(gpu, x, &x_f32, gate.cols)?;
                if tessl::ab_flags::need_barrier(true) {
                    gpu.barrier()?;
                }
                &x_f32
            } else {
                x
            };
            gemv_q4_mlx_blocked_gate_up_gelu(gpu, gate, up, x_ref, mid, mid_as_bf16)
        }
    }
}

fn simd_tg_geometry(rows: u32) -> (usize, usize) {
    let rows_per_tg = GEMV_SIMD_SG * GEMV_SIMD_ROWS;
    let n_tg = ((rows as usize) + rows_per_tg as usize - 1) / rows_per_tg as usize;
    (GEMV_SIMD_TPTG, n_tg)
}

/// MLX Q4 simd GEMV with fused residual add: `y[row] = (W@x)[row] + resid[row]`.
pub fn gemv_q4_mlx_simd_add(
    gpu: &GemmaGpu,
    packed: &GpuBuffer,
    scales: &GpuBuffer,
    biases: &GpuBuffer,
    x: &GpuBuffer,
    resid: &GpuBuffer,
    y: &GpuBuffer,
    rows: u32,
    cols: u32,
    group_size: u32,
    interleaved: bool,
) -> Result<()> {
    if rows == 0 {
        return Ok(());
    }
    // `x` must already be bf16.
    let entry = if interleaved {
        "gemv_q4_mlx_simd_add_i4"
    } else {
        "gemv_q4_mlx_simd_add"
    };
    let p = gpu.rt.pipeline(entry).map_err(map_metal)?;
    let (tptg, n_tg) = simd_tg_geometry(rows);
    if trace_gemv_enabled() {
        eprintln!(
            "[trace] gemv entry={entry} rows={rows} cols={cols} tg={tptg} groups={n_tg} fused=add act=bf16"
        );
    }
    let (rows_off, cols_off, gs_off) = push_gemv_dims(gpu, rows, cols, group_size)?;
    gpu.rt
        .with_binder(|bnd| {
            bnd.set_pipeline(&p);
            set_gpu_buf(bnd, packed, 0);
            set_gpu_buf(bnd, scales, 1);
            set_gpu_buf(bnd, biases, 2);
            set_gpu_buf(bnd, x, 3);
            set_gpu_buf(bnd, y, 4);
            gpu.icb_scalars.bind_u32(bnd, rows_off, 5);
            gpu.icb_scalars.bind_u32(bnd, cols_off, 6);
            gpu.icb_scalars.bind_u32(bnd, gs_off, 7);
            set_gpu_buf(bnd, resid, 8);
            bnd.dispatch(
                tessl::runtime::mtl_size(n_tg, 1, 1),
                tessl::runtime::mtl_size(tptg, 1, 1),
            );
            Ok(())
        })
        .map_err(map_metal)
}

/// Row-major / Interleaved4 fused gate∥up→gelu via simdgroup GEMV.
/// `x` must already be bf16 (see [`gemv_q4_mlx_gate_up_gelu_impl`]).
pub fn gemv_q4_mlx_simd_gate_up_gelu(
    gpu: &GemmaGpu,
    gate: &HotQuantBanks,
    up: &HotQuantBanks,
    x: &GpuBuffer,
    mid: &GpuBuffer,
    mid_as_bf16: bool,
) -> Result<()> {
    let entry = if gate.layout == HotGemvLayout::Interleaved4 {
        "gemv_q4_mlx_simd_gate_up_gelu_i4"
    } else {
        "gemv_q4_mlx_simd_gate_up_gelu"
    };
    let p = gpu.rt.pipeline(entry).map_err(map_metal)?;
    let (tptg, n_tg) = simd_tg_geometry(gate.rows);
    if trace_gemv_enabled() {
        eprintln!(
            "[trace] gemv entry={entry} rows={} cols={} tg={tptg} groups={n_tg} fused=gate_up_gelu_simd act=bf16 mid_bf16={mid_as_bf16}",
            gate.rows, gate.cols
        );
    }
    let (rows_off, cols_off, gs_off) = push_gemv_dims(gpu, gate.rows, gate.cols, gate.group_size)?;
    let mid_bf16_off = gpu.icb_scalars.push_u32(if mid_as_bf16 { 1 } else { 0 })?;
    gpu.rt
        .with_binder(|bnd| {
            bnd.set_pipeline(&p);
            set_gpu_buf(bnd, &gate.packed, 0);
            set_gpu_buf(bnd, &gate.scales, 1);
            set_gpu_buf(bnd, &gate.zeros, 2);
            set_gpu_buf(bnd, &up.packed, 3);
            set_gpu_buf(bnd, &up.scales, 4);
            set_gpu_buf(bnd, &up.zeros, 5);
            set_gpu_buf(bnd, x, 6);
            set_gpu_buf(bnd, mid, 7);
            gpu.icb_scalars.bind_u32(bnd, rows_off, 8);
            gpu.icb_scalars.bind_u32(bnd, cols_off, 9);
            gpu.icb_scalars.bind_u32(bnd, gs_off, 10);
            gpu.icb_scalars.bind_u32(bnd, mid_bf16_off, 11);
            bnd.dispatch(
                tessl::runtime::mtl_size(n_tg, 1, 1),
                tessl::runtime::mtl_size(tptg, 1, 1),
            );
            Ok(())
        })
        .map_err(map_metal)
}

/// Fused producer K∥V (simd). Both banks must [`HotQuantBanks::can_fuse_kv`].
/// `x` is f32 unless `x_is_bf16`.
///
/// Bank-partitioned (like QKV): `[0,tg_k)→K`, `[tg_k,2*tg_k)→V`. Each TG runs
/// a solo-style reduce so K/V are bit-exact vs separate `gemv` launches.
pub fn gemv_q4_mlx_simd_kv(
    gpu: &GemmaGpu,
    k: &HotQuantBanks,
    v: &HotQuantBanks,
    x: &GpuBuffer,
    k_out: &GpuBuffer,
    v_out: &GpuBuffer,
) -> Result<()> {
    gemv_q4_mlx_simd_kv_impl(gpu, k, v, x, k_out, v_out, false)
}

pub fn gemv_q4_mlx_simd_kv_bf16_x(
    gpu: &GemmaGpu,
    k: &HotQuantBanks,
    v: &HotQuantBanks,
    x_bf16: &GpuBuffer,
    k_out: &GpuBuffer,
    v_out: &GpuBuffer,
) -> Result<()> {
    gemv_q4_mlx_simd_kv_impl(gpu, k, v, x_bf16, k_out, v_out, true)
}

fn gemv_q4_mlx_simd_kv_impl(
    gpu: &GemmaGpu,
    k: &HotQuantBanks,
    v: &HotQuantBanks,
    x: &GpuBuffer,
    k_out: &GpuBuffer,
    v_out: &GpuBuffer,
    x_is_bf16: bool,
) -> Result<()> {
    if k.rows == 0 {
        return Ok(());
    }
    if !k.can_fuse_kv(v) {
        return Err(Error::Metal(
            "simd_kv: need matching RowMajor/Interleaved4 Q4Mlx (or GEMMA_METAL_FUSE_KV=0)".into(),
        ));
    }
    let x_bf16;
    let x_ref = if x_is_bf16 {
        x
    } else {
        if tessl::ab_flags::need_barrier(true) {
            gpu.barrier()?;
        }
        x_bf16 = prepare_act_bf16(gpu, x, k.cols)?;
        if tessl::ab_flags::need_barrier(true) {
            gpu.barrier()?;
        }
        &x_bf16
    };
    let entry = if k.layout == HotGemvLayout::Interleaved4 {
        "gemv_q4_mlx_simd_kv_i4"
    } else {
        "gemv_q4_mlx_simd_kv"
    };
    let p = gpu.rt.pipeline(entry).map_err(map_metal)?;
    let tg_k = simd_tg_count(k.rows);
    let tg_v = simd_tg_count(v.rows);
    let n_tg = (tg_k + tg_v) as usize;
    if trace_gemv_enabled() {
        eprintln!(
            "[trace] gemv entry={entry} rows={} cols={} tg={GEMV_SIMD_TPTG} \
             groups={n_tg} (k={tg_k} v={tg_v}) fused=kv act=bf16",
            k.rows, k.cols
        );
    }
    let (rows_off, cols_off, gs_off) = push_gemv_dims(gpu, k.rows, k.cols, k.group_size)?;
    let tg_k_off = gpu.icb_scalars.push_u32(tg_k)?;
    gpu.rt
        .with_binder(|bnd| {
            bnd.set_pipeline(&p);
            set_gpu_buf(bnd, &k.packed, 0);
            set_gpu_buf(bnd, &k.scales, 1);
            set_gpu_buf(bnd, &k.zeros, 2);
            set_gpu_buf(bnd, &v.packed, 3);
            set_gpu_buf(bnd, &v.scales, 4);
            set_gpu_buf(bnd, &v.zeros, 5);
            set_gpu_buf(bnd, x_ref, 6);
            set_gpu_buf(bnd, k_out, 7);
            set_gpu_buf(bnd, v_out, 8);
            gpu.icb_scalars.bind_u32(bnd, rows_off, 9);
            gpu.icb_scalars.bind_u32(bnd, cols_off, 10);
            gpu.icb_scalars.bind_u32(bnd, gs_off, 11);
            gpu.icb_scalars.bind_u32(bnd, tg_k_off, 12);
            bnd.dispatch(
                tessl::runtime::mtl_size(n_tg, 1, 1),
                tessl::runtime::mtl_size(GEMV_SIMD_TPTG, 1, 1),
            );
            Ok(())
        })
        .map_err(map_metal)
}

/// Fused BlockedBn16 gate∥up→gelu (writes `mid`). Falls back to Err if shapes mismatch.
pub fn gemv_q4_mlx_blocked_gate_up_gelu(
    gpu: &GemmaGpu,
    gate: &HotQuantBanks,
    up: &HotQuantBanks,
    x: &GpuBuffer,
    mid: &GpuBuffer,
    mid_as_bf16: bool,
) -> Result<()> {
    if gate.rows == 0 {
        return Ok(());
    }
    if gate.layout != HotGemvLayout::BlockedBn16 || up.layout != HotGemvLayout::BlockedBn16 {
        return Err(Error::Metal(
            "blocked gate_up_gelu: need BlockedBn16 Q4Mlx".into(),
        ));
    }
    let entry = "gemv_q4_mlx_blocked_gate_up_gelu";
    let p = gpu.rt.pipeline(entry).map_err(map_metal)?;
    let tptg = (GEMV_BN * GEMV_LANES) as usize;
    let n_tg = ((gate.rows as usize) + GEMV_BN as usize - 1) / GEMV_BN as usize;
    let tg_mem = (gate.cols as usize).min(GEMV_X_TILE) * 4;
    if trace_gemv_enabled() {
        eprintln!(
            "[trace] gemv entry={entry} rows={} cols={} tg={tptg} groups={n_tg} fused=gate_up_gelu mid_bf16={mid_as_bf16}",
            gate.rows, gate.cols
        );
    }
    let (rows_off, cols_off, gs_off) = push_gemv_dims(gpu, gate.rows, gate.cols, gate.group_size)?;
    let mid_bf16_off = gpu.icb_scalars.push_u32(if mid_as_bf16 { 1 } else { 0 })?;
    gpu.rt
        .with_binder(|bnd| {
            bnd.set_pipeline(&p);
            set_gpu_buf(bnd, &gate.packed, 0);
            set_gpu_buf(bnd, &gate.scales, 1);
            set_gpu_buf(bnd, &gate.zeros, 2);
            set_gpu_buf(bnd, &up.packed, 3);
            set_gpu_buf(bnd, &up.scales, 4);
            set_gpu_buf(bnd, &up.zeros, 5);
            set_gpu_buf(bnd, x, 6);
            set_gpu_buf(bnd, mid, 7);
            gpu.icb_scalars.bind_u32(bnd, rows_off, 8);
            gpu.icb_scalars.bind_u32(bnd, cols_off, 9);
            gpu.icb_scalars.bind_u32(bnd, gs_off, 10);
            gpu.icb_scalars.bind_u32(bnd, mid_bf16_off, 11);
            bnd.set_threadgroup_memory(0, tg_mem);
            bnd.dispatch(
                tessl::runtime::mtl_size(n_tg, 1, 1),
                tessl::runtime::mtl_size(tptg, 1, 1),
            );
            Ok(())
        })
        .map_err(map_metal)
}

/// Softcap + argmax for `n <= 256` (single TG). Prefer [`softcap_argmax`] for
/// full vocab (262k).
pub fn softcap_sample(
    gpu: &GemmaGpu,
    logits: &GpuBuffer,
    out_token: &GpuBuffer,
    softcap: f32,
    n: u32,
) -> Result<()> {
    if n == 0 || n > 256 {
        return Err(Error::Metal(format!(
            "softcap_sample requires 1..=256 logits, got {n}"
        )));
    }
    let p = gpu
        .rt
        .pipeline(KernelId::SoftcapSample.entry_name())
        .map_err(map_metal)?;
    let tptg = n.next_power_of_two().max(1).min(256) as usize;
    gpu.icb_scalars.set_softcap(softcap);
    gpu.rt
        .with_binder(|bnd| {
            bnd.set_pipeline(&p);
            set_gpu_buf(bnd, logits, 0);
            set_gpu_buf(bnd, out_token, 1);
            set_gpu_buf(bnd, &gpu.icb_scalars.softcap, 2);
            set_u32(bnd, n, 3);
            bnd.dispatch(
                tessl::runtime::mtl_size(1, 1, 1),
                tessl::runtime::mtl_size(tptg, 1, 1),
            );
            Ok(())
        })
        .map_err(map_metal)?;
    gpu.synchronize()?;
    if out_token.read_u32().first() == Some(&u32::MAX) {
        return Err(Error::Metal(
            "softcap_sample: logit row has no finite value".into(),
        ));
    }
    Ok(())
}

pub fn softcap_logits(gpu: &GemmaGpu, logits: &GpuBuffer, softcap: f32, n: u32) -> Result<()> {
    let p = gpu
        .rt
        .pipeline(KernelId::SoftcapLogits.entry_name())
        .map_err(map_metal)?;
    gpu.icb_scalars.set_softcap(softcap);
    dispatch_1d(&gpu.rt, &p, n as usize, |bnd| {
        set_gpu_buf(bnd, logits, 0);
        set_gpu_buf(bnd, &gpu.icb_scalars.softcap, 1);
        set_u32(bnd, n, 2);
    })
    .map_err(map_metal)
}

const ARGMAX_TG: u32 = 256;

/// Softcap then multi-pass GPU argmax — works for full E4B vocab (262_144).
///
/// First pass fuses softcap on-read (no separate write of 262k logits). Original
/// indices propagate on GPU. Scratch ping-pong buffers are reused across calls
/// when `scratch` is provided (avoids per-token Metal alloc + sync).
pub fn softcap_argmax(gpu: &GemmaGpu, logits: &GpuBuffer, softcap: f32, n: u32) -> Result<u32> {
    softcap_argmax_scratch(gpu, logits, softcap, n, None)
}

/// Encode softcap+argmax into the open CB; token lands in `out_token` (no sync).
pub fn softcap_argmax_encode(
    gpu: &GemmaGpu,
    logits: &GpuBuffer,
    softcap: f32,
    n: u32,
    scratch: &mut ArgmaxScratch,
    out_token: &GpuBuffer,
) -> Result<()> {
    softcap_argmax_encode_offset(gpu, logits, 0, softcap, n, scratch, out_token, 0)
}

/// Like [`softcap_argmax_encode`] but `logits`/`out_token` start at byte offsets
/// (for packed M-row verify logits → `verify_outs[m]`).
pub fn softcap_argmax_encode_offset(
    gpu: &GemmaGpu,
    logits: &GpuBuffer,
    logits_byte_off: usize,
    softcap: f32,
    n: u32,
    scratch: &mut ArgmaxScratch,
    out_token: &GpuBuffer,
    out_byte_off: usize,
) -> Result<()> {
    use tessl::dispatch::set_gpu_buf_offset;
    if n == 0 {
        return Err(Error::Metal("softcap_argmax: n == 0".into()));
    }
    if n <= 256 {
        let p = gpu
            .rt
            .pipeline(KernelId::SoftcapSample.entry_name())
            .map_err(map_metal)?;
        let tptg = n.next_power_of_two().max(1).min(256) as usize;
        gpu.icb_scalars.set_softcap(softcap);
        return gpu
            .rt
            .with_binder(|bnd| {
                bnd.set_pipeline(&p);
                set_gpu_buf_offset(bnd, logits, logits_byte_off, 0);
                set_gpu_buf_offset(bnd, out_token, out_byte_off, 1);
                set_gpu_buf(bnd, &gpu.icb_scalars.softcap, 2);
                set_u32(bnd, n, 3);
                bnd.dispatch(
                    tessl::runtime::mtl_size(1, 1, 1),
                    tessl::runtime::mtl_size(tptg, 1, 1),
                );
                Ok(())
            })
            .map_err(map_metal);
    }

    // Default: single-pass softcap+argmax (vocab=262k). Multipass via
    // GEMMA_METAL_ARGMAX_MULTIPASS=1 (A/B / diagnose). Offsets supported by
    // multipass only; one-pass requires aligned buffer starts.
    let want_multipass = {
        static V: OnceLock<bool> = OnceLock::new();
        *V.get_or_init(|| {
            match std::env::var("GEMMA_METAL_ARGMAX_MULTIPASS")
                .ok()
                .as_deref()
            {
                Some("1") | Some("true") | Some("on") => true,
                _ => false,
            }
        })
    };
    if !want_multipass && logits_byte_off == 0 && out_byte_off == 0 {
        let p = gpu
            .rt
            .pipeline(KernelId::SoftcapArgmaxOnePass.entry_name())
            .map_err(map_metal)?;
        // 1024 threads ≈ Metal TG cap; each lane scans n/1024 logits.
        let tptg = 1024usize;
        gpu.icb_scalars.set_softcap(softcap);
        return gpu
            .rt
            .with_binder(|bnd| {
                bnd.set_pipeline(&p);
                set_gpu_buf(bnd, logits, 0);
                set_gpu_buf(bnd, out_token, 1);
                set_gpu_buf(bnd, &gpu.icb_scalars.softcap, 2);
                set_u32(bnd, n, 3);
                bnd.dispatch(
                    tessl::runtime::mtl_size(1, 1, 1),
                    tessl::runtime::mtl_size(tptg, 1, 1),
                );
                Ok(())
            })
            .map_err(map_metal);
    }

    let mut cur_n = n;
    let mut pass = 0u32;
    let mut cur_vals = logits.clone();
    let mut cur_vals_off = logits_byte_off;
    let mut cur_idx: Option<GpuBuffer> = None;

    loop {
        let groups = (cur_n + ARGMAX_TG - 1) / ARGMAX_TG;
        scratch.ensure(gpu, groups as usize)?;
        let (idx_buf, val_buf) = {
            let (i, v) = scratch.tier(pass);
            (i.clone(), v.clone())
        };
        let sc = if pass == 0 { softcap } else { 0.0 };
        {
            let p = gpu
                .rt
                .pipeline(KernelId::ArgmaxF32.entry_name())
                .map_err(map_metal)?;
            let has_idx = if cur_idx.is_some() { 1u32 } else { 0 };
            let tptg = ARGMAX_TG as usize;
            // Multipass may need softcap=0 on later passes — use a pushed f32 via
            // softcap buffer rewrite (single softcap slot; passes are sequential).
            gpu.icb_scalars.set_softcap(sc);
            gpu.rt
                .with_binder(|bnd| {
                    bnd.set_pipeline(&p);
                    set_gpu_buf_offset(bnd, &cur_vals, cur_vals_off, 0);
                    set_gpu_buf(bnd, &idx_buf, 1);
                    set_gpu_buf(bnd, &val_buf, 2);
                    set_u32(bnd, cur_n, 3);
                    if let Some(ref idx) = cur_idx {
                        set_gpu_buf(bnd, idx, 4);
                    } else {
                        set_gpu_buf(bnd, &scratch.dummy_idx, 4);
                    }
                    set_u32(bnd, has_idx, 5);
                    set_gpu_buf(bnd, &gpu.icb_scalars.softcap, 6);
                    bnd.dispatch(
                        tessl::runtime::mtl_size(groups as usize, 1, 1),
                        tessl::runtime::mtl_size(tptg, 1, 1),
                    );
                    Ok(())
                })
                .map_err(map_metal)?;
        }
        if tessl::ab_flags::need_barrier(true) {
            gpu.barrier()?;
        }

        if groups == 1 {
            let p = gpu.rt.pipeline("copy_f32").map_err(map_metal)?;
            dispatch_1d(&gpu.rt, &p, 1, |bnd| {
                set_gpu_buf(bnd, &idx_buf, 0);
                set_gpu_buf_offset(bnd, out_token, out_byte_off, 1);
                set_u32(bnd, 1, 2);
            })
            .map_err(map_metal)?;
            return Ok(());
        }

        cur_idx = Some(idx_buf);
        cur_vals = val_buf;
        cur_vals_off = 0;
        cur_n = groups;
        pass += 1;
    }
}

/// Softcap+argmax with optional reusable scratch (two ping-pong (idx,val) tiers).
pub fn softcap_argmax_scratch(
    gpu: &GemmaGpu,
    logits: &GpuBuffer,
    softcap: f32,
    n: u32,
    mut scratch: Option<&mut ArgmaxScratch>,
) -> Result<u32> {
    if n == 0 {
        return Err(Error::Metal("softcap_argmax: n == 0".into()));
    }
    if n <= 256 {
        let tok = if let Some(ref mut sc) = scratch {
            sc.out_token.clone()
        } else {
            gpu.rt.alloc_buffer(4).map_err(map_metal)?
        };
        softcap_sample(gpu, logits, &tok, softcap, n)?;
        gpu.synchronize()?;
        let idx = tok.read_u32()[0];
        if idx == u32::MAX {
            return Err(Error::Metal(
                "softcap_argmax: logit row has no finite value".into(),
            ));
        }
        return Ok(idx);
    }

    let mut owned_scratch;
    let sc = if let Some(s) = scratch.as_mut() {
        s
    } else {
        owned_scratch = Some(ArgmaxScratch::new(gpu, n)?);
        owned_scratch.as_mut().unwrap()
    };
    softcap_argmax_encode(gpu, logits, softcap, n, sc, &sc.out_token.clone())?;
    gpu.synchronize()?;
    Ok(sc.out_token.read_u32()[0])
}

/// Copy `n` f32 elements: `dst[0..n] = src[0..n]`.
pub fn copy_f32_n(gpu: &GemmaGpu, src: &GpuBuffer, dst: &GpuBuffer, n: u32) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let p = gpu.rt.pipeline("copy_f32").map_err(map_metal)?;
    let n_off = gpu.icb_scalars.push_u32(n)?;
    dispatch_1d(&gpu.rt, &p, n as usize, |bnd| {
        set_gpu_buf(bnd, src, 0);
        set_gpu_buf(bnd, dst, 1);
        gpu.icb_scalars.bind_u32(bnd, n_off, 2);
    })
    .map_err(map_metal)
}

/// Copy `n` f32 elements into `dst[dst_elem_offset ..]`.
pub fn copy_f32_to_offset(
    gpu: &GemmaGpu,
    src: &GpuBuffer,
    dst: &GpuBuffer,
    dst_elem_offset: usize,
    n: u32,
) -> Result<()> {
    use tessl::dispatch::set_gpu_buf_offset;
    if n == 0 {
        return Ok(());
    }
    let off = dst_elem_offset.saturating_mul(4);
    let need = off + (n as usize) * 4;
    if need > dst.nbytes() {
        return Err(Error::Metal(format!(
            "copy_f32_to_offset: dst off={dst_elem_offset} n={n} OOB (buf {} B)",
            dst.nbytes()
        )));
    }
    let p = gpu.rt.pipeline("copy_f32").map_err(map_metal)?;
    let n_off = gpu.icb_scalars.push_u32(n)?;
    dispatch_1d(&gpu.rt, &p, n as usize, |bnd| {
        set_gpu_buf(bnd, src, 0);
        set_gpu_buf_offset(bnd, dst, off, 1);
        gpu.icb_scalars.bind_u32(bnd, n_off, 2);
    })
    .map_err(map_metal)
}

/// Copy `n` f32 elements from `src[src_elem_offset ..]` into `dst[0..n]`.
pub fn copy_f32_from_offset(
    gpu: &GemmaGpu,
    src: &GpuBuffer,
    src_elem_offset: usize,
    dst: &GpuBuffer,
    n: u32,
) -> Result<()> {
    copy_f32_range(gpu, src, src_elem_offset, dst, 0, n)
}

/// Copy `n` f32s from `src[src_elem_offset..]` → `dst[dst_elem_offset..]`.
pub fn copy_f32_range(
    gpu: &GemmaGpu,
    src: &GpuBuffer,
    src_elem_offset: usize,
    dst: &GpuBuffer,
    dst_elem_offset: usize,
    n: u32,
) -> Result<()> {
    use tessl::dispatch::set_gpu_buf_offset;
    if n == 0 {
        return Ok(());
    }
    let src_off = src_elem_offset.saturating_mul(4);
    let dst_off = dst_elem_offset.saturating_mul(4);
    let bytes = (n as usize) * 4;
    if src_off + bytes > src.nbytes() {
        return Err(Error::Metal(format!(
            "copy_f32_range: src off={src_elem_offset} n={n} OOB (buf {} B)",
            src.nbytes()
        )));
    }
    if dst_off + bytes > dst.nbytes() {
        return Err(Error::Metal(format!(
            "copy_f32_range: dst off={dst_elem_offset} n={n} OOB (buf {} B)",
            dst.nbytes()
        )));
    }
    let p = gpu.rt.pipeline("copy_f32").map_err(map_metal)?;
    let n_off = gpu.icb_scalars.push_u32(n)?;
    dispatch_1d(&gpu.rt, &p, n as usize, |bnd| {
        set_gpu_buf_offset(bnd, src, src_off, 0);
        set_gpu_buf_offset(bnd, dst, dst_off, 1);
        gpu.icb_scalars.bind_u32(bnd, n_off, 2);
    })
    .map_err(map_metal)
}

/// Copy one u32: `src[0]` → `dst[0]` (via 4-byte `copy_f32` elem kernel).
pub fn copy_u32_one(gpu: &GemmaGpu, src: &GpuBuffer, dst: &GpuBuffer) -> Result<()> {
    let p = gpu.rt.pipeline("copy_f32").map_err(map_metal)?; // 4-byte elem copy OK
    dispatch_1d(&gpu.rt, &p, 1, |bnd| {
        set_gpu_buf(bnd, src, 0);
        set_gpu_buf(bnd, dst, 1);
        set_u32(bnd, 1, 2);
    })
    .map_err(map_metal)
}

/// Copy `src[0]` → `dst[dst_index]` (4-byte elem) via Metal offset bind.
pub fn copy_u32_to_index(
    gpu: &GemmaGpu,
    src: &GpuBuffer,
    dst: &GpuBuffer,
    dst_index: u32,
) -> Result<()> {
    use tessl::dispatch::set_gpu_buf_offset;
    let p = gpu.rt.pipeline("copy_f32").map_err(map_metal)?;
    let off = (dst_index as usize).saturating_mul(4);
    if off + 4 > dst.nbytes() {
        return Err(Error::Metal(format!(
            "copy_u32_to_index: dst_index={dst_index} out of range (buf {} B)",
            dst.nbytes()
        )));
    }
    dispatch_1d(&gpu.rt, &p, 1, |bnd| {
        set_gpu_buf(bnd, src, 0);
        set_gpu_buf_offset(bnd, dst, off, 1);
        set_u32(bnd, 1, 2);
    })
    .map_err(map_metal)
}

/// Copy `src[src_index]` → `dst[0]` (4-byte elem) — verify seed slot → active seed.
pub fn copy_u32_from_index(
    gpu: &GemmaGpu,
    src: &GpuBuffer,
    src_index: u32,
    dst: &GpuBuffer,
) -> Result<()> {
    use tessl::dispatch::set_gpu_buf_offset;
    let p = gpu.rt.pipeline("copy_f32").map_err(map_metal)?;
    let off = (src_index as usize).saturating_mul(4);
    if off + 4 > src.nbytes() {
        return Err(Error::Metal(format!(
            "copy_u32_from_index: src_index={src_index} out of range (buf {} B)",
            src.nbytes()
        )));
    }
    dispatch_1d(&gpu.rt, &p, 1, |bnd| {
        set_gpu_buf_offset(bnd, src, off, 0);
        set_gpu_buf(bnd, dst, 1);
        set_u32(bnd, 1, 2);
    })
    .map_err(map_metal)
}

/// One hierarchical argmax pass: `ceil(n / 256)` threadgroups write partial
/// `(idx, val)` pairs into `out_idx` / `out_val` (length ≥ groups).
///
/// When `idx_in` is `Some`, indices are taken from that buffer (original vocab
/// ids propagated across multi-pass reduce). When `None`, indices are positions
/// in the current `logits` buffer.
///
/// When `softcap > 0`, softcap is applied on-read (fused first pass; no writeback).
pub fn argmax_f32_pass(
    gpu: &GemmaGpu,
    logits: &GpuBuffer,
    out_idx: &GpuBuffer,
    out_val: &GpuBuffer,
    n: u32,
    idx_in: Option<&GpuBuffer>,
    softcap: f32,
    dummy: Option<&GpuBuffer>,
) -> Result<()> {
    if n == 0 {
        return Err(Error::Metal("argmax_f32_pass: n == 0".into()));
    }
    let p = gpu
        .rt
        .pipeline(KernelId::ArgmaxF32.entry_name())
        .map_err(map_metal)?;
    let groups = ((n + ARGMAX_TG - 1) / ARGMAX_TG) as usize;
    let tptg = ARGMAX_TG as usize;
    // Prefer session-persistent dummy — never Cold-alloc on the decode hot path.
    let owned_dummy;
    let idx_buf = if let Some(b) = idx_in {
        b
    } else if let Some(d) = dummy {
        d
    } else {
        owned_dummy = Some(gpu.rt.alloc_buffer(4).map_err(map_metal)?);
        owned_dummy.as_ref().unwrap()
    };
    let has = if idx_in.is_some() { 1u32 } else { 0u32 };
    gpu.icb_scalars.set_softcap(softcap);
    gpu.rt
        .with_binder(|bnd| {
            bnd.set_pipeline(&p);
            set_gpu_buf(bnd, logits, 0);
            set_gpu_buf(bnd, out_idx, 1);
            set_gpu_buf(bnd, out_val, 2);
            set_u32(bnd, n, 3);
            set_gpu_buf(bnd, idx_buf, 4);
            set_u32(bnd, has, 5);
            set_gpu_buf(bnd, &gpu.icb_scalars.softcap, 6);
            bnd.dispatch(
                tessl::runtime::mtl_size(groups, 1, 1),
                tessl::runtime::mtl_size(tptg, 1, 1),
            );
            Ok(())
        })
        .map_err(map_metal)
}

/// Ping-pong argmax partial buffers (reused each decode step).
pub struct ArgmaxScratch {
    /// Two tiers × (idx, val). Sized for worst-case first-pass groups.
    tiers: [(GpuBuffer, GpuBuffer); 2],
    /// Persistent 4B dummy for first-pass (no idx_in) — Hot, never Cold/token.
    pub dummy_idx: GpuBuffer,
    /// GPU-resident greedy token (Hot).
    pub out_token: GpuBuffer,
    cap_groups: usize,
}

impl ArgmaxScratch {
    pub fn new(gpu: &GemmaGpu, vocab: u32) -> Result<Self> {
        let groups = ((vocab + ARGMAX_TG - 1) / ARGMAX_TG) as usize;
        let mk = |n: usize| -> Result<(GpuBuffer, GpuBuffer)> {
            let i = gpu.rt.alloc_buffer_hot(n.max(1) * 4).map_err(map_metal)?;
            let v = gpu.rt.alloc_buffer_hot(n.max(1) * 4).map_err(map_metal)?;
            Ok((i, v))
        };
        Ok(Self {
            tiers: [mk(groups)?, mk(groups)?],
            dummy_idx: gpu.rt.alloc_buffer_hot(4).map_err(map_metal)?,
            out_token: gpu.rt.alloc_buffer_hot(4).map_err(map_metal)?,
            cap_groups: groups,
        })
    }

    fn ensure(&mut self, gpu: &GemmaGpu, groups: usize) -> Result<()> {
        if groups <= self.cap_groups {
            return Ok(());
        }
        let mk = |n: usize| -> Result<(GpuBuffer, GpuBuffer)> {
            let i = gpu.rt.alloc_buffer_hot(n.max(1) * 4).map_err(map_metal)?;
            let v = gpu.rt.alloc_buffer_hot(n.max(1) * 4).map_err(map_metal)?;
            Ok((i, v))
        };
        self.tiers = [mk(groups)?, mk(groups)?];
        self.cap_groups = groups;
        Ok(())
    }

    fn tier(&self, pass: u32) -> &(GpuBuffer, GpuBuffer) {
        &self.tiers[(pass as usize) % 2]
    }
}

/// Dequant one Q4 / Q4-MLX embed row into `out` on GPU (token id in `tok_ids[0]`).
pub fn embed_lookup_quant(
    gpu: &GemmaGpu,
    banks: &HotQuantBanks,
    tok_ids: &GpuBuffer,
    out: &GpuBuffer,
    vocab: u32,
) -> Result<()> {
    embed_lookup_quant_n(gpu, banks, tok_ids, out, vocab, 1)
}

/// Embed `n_tokens` ids from `tok_ids` into `out` as `[n_tokens, hidden]` f32.
pub fn embed_lookup_quant_n(
    gpu: &GemmaGpu,
    banks: &HotQuantBanks,
    tok_ids: &GpuBuffer,
    out: &GpuBuffer,
    vocab: u32,
    n_tokens: u32,
) -> Result<()> {
    if n_tokens == 0 {
        return Ok(());
    }
    let (entry, group_size) = match banks.scheme {
        crate::quant::QuantScheme::Q4 { group_size } => {
            (KernelId::EmbedLookupQ4.entry_name(), group_size as u32)
        }
        crate::quant::QuantScheme::Q4Mlx { group_size } => {
            (KernelId::EmbedLookupQ4Mlx.entry_name(), group_size as u32)
        }
        _ => {
            return Err(Error::Metal(
                "embed_lookup_quant: only Q4 / Q4Mlx supported".into(),
            ))
        }
    };
    let p = gpu.rt.pipeline(entry).map_err(map_metal)?;
    let hidden = banks.cols as u32;
    let total = (n_tokens as usize).saturating_mul(banks.cols as usize);
    dispatch_1d(&gpu.rt, &p, total, |bnd| {
        set_gpu_buf(bnd, &banks.packed, 0);
        set_gpu_buf(bnd, &banks.scales, 1);
        set_gpu_buf(bnd, &banks.zeros, 2);
        set_gpu_buf(bnd, tok_ids, 3);
        set_gpu_buf(bnd, out, 4);
        set_u32(bnd, hidden, 5);
        set_u32(bnd, group_size, 6);
        set_u32(bnd, vocab, 7);
        set_u32(bnd, n_tokens, 8);
    })
    .map_err(map_metal)
}

/// Q4 affine thin GEMM: `Y[M, rows] = X[M, cols] @ W^T` (bf16 X, f32 Y).
pub fn gemm_q4_mlx_simd(
    gpu: &GemmaGpu,
    packed: &GpuBuffer,
    scales: &GpuBuffer,
    zeros: &GpuBuffer,
    x_bf16: &GpuBuffer,
    y: &GpuBuffer,
    rows: u32,
    cols: u32,
    group_size: u32,
    m: u32,
    interleaved: bool,
) -> Result<()> {
    if rows == 0 || m == 0 {
        return Ok(());
    }
    let entry = if interleaved {
        KernelId::GemmQ4MlxSimdI4.entry_name()
    } else {
        KernelId::GemmQ4MlxSimd.entry_name()
    };
    let p = gpu.rt.pipeline(entry).map_err(map_metal)?;
    let (tptg, n_tg) = simd_tg_geometry(rows);
    gpu.rt
        .with_binder(|bnd| {
            bnd.set_pipeline(&p);
            set_gpu_buf(bnd, packed, 0);
            set_gpu_buf(bnd, scales, 1);
            set_gpu_buf(bnd, zeros, 2);
            set_gpu_buf(bnd, x_bf16, 3);
            set_gpu_buf(bnd, y, 4);
            set_u32(bnd, rows, 5);
            set_u32(bnd, cols, 6);
            set_u32(bnd, group_size, 7);
            set_u32(bnd, m, 8);
            bnd.dispatch(
                tessl::runtime::mtl_size(n_tg, 1, 1),
                tessl::runtime::mtl_size(tptg, 1, 1),
            );
            Ok(())
        })
        .map_err(map_metal)
}

/// Like [`gemm_q4_mlx_simd`] but `y[m,r] = resid[m,r] + (X[m]·W[r])`.
pub fn gemm_q4_mlx_simd_add(
    gpu: &GemmaGpu,
    packed: &GpuBuffer,
    scales: &GpuBuffer,
    zeros: &GpuBuffer,
    x_bf16: &GpuBuffer,
    resid: &GpuBuffer,
    y: &GpuBuffer,
    rows: u32,
    cols: u32,
    group_size: u32,
    m: u32,
    interleaved: bool,
) -> Result<()> {
    if rows == 0 || m == 0 {
        return Ok(());
    }
    let entry = if interleaved {
        KernelId::GemmQ4MlxSimdAddI4.entry_name()
    } else {
        KernelId::GemmQ4MlxSimdAdd.entry_name()
    };
    let p = gpu.rt.pipeline(entry).map_err(map_metal)?;
    let (tptg, n_tg) = simd_tg_geometry(rows);
    gpu.rt
        .with_binder(|bnd| {
            bnd.set_pipeline(&p);
            set_gpu_buf(bnd, packed, 0);
            set_gpu_buf(bnd, scales, 1);
            set_gpu_buf(bnd, zeros, 2);
            set_gpu_buf(bnd, x_bf16, 3);
            set_gpu_buf(bnd, y, 4);
            set_u32(bnd, rows, 5);
            set_u32(bnd, cols, 6);
            set_u32(bnd, group_size, 7);
            set_u32(bnd, m, 8);
            set_gpu_buf(bnd, resid, 9);
            bnd.dispatch(
                tessl::runtime::mtl_size(n_tg, 1, 1),
                tessl::runtime::mtl_size(tptg, 1, 1),
            );
            Ok(())
        })
        .map_err(map_metal)
}

/// In-place `x[i] *= scale` (Gemma4 embed_scale after Hot lookup).
pub fn scale_f32_inplace(gpu: &GemmaGpu, x: &GpuBuffer, scale: f32, n: u32) -> Result<()> {
    if (scale - 1.0).abs() <= 1e-12 {
        return Ok(());
    }
    let p = gpu
        .rt
        .pipeline(KernelId::ScaleF32Inplace.entry_name())
        .map_err(map_metal)?;
    let scale_off = gpu.icb_scalars.push_f32(scale)?;
    let n_off = gpu.icb_scalars.push_u32(n)?;
    dispatch_1d(&gpu.rt, &p, n as usize, |bnd| {
        set_gpu_buf(bnd, x, 0);
        gpu.icb_scalars.bind_f32(bnd, scale_off, 1);
        gpu.icb_scalars.bind_u32(bnd, n_off, 2);
    })
    .map_err(map_metal)
}

/// `dst[i] += src[i]` via the tessl utility metallib.
pub fn add_inplace_f32(gpu: &GemmaGpu, dst: &GpuBuffer, src: &GpuBuffer, n: u32) -> Result<()> {
    let p = gpu.rt.pipeline("add_inplace_f32").map_err(map_metal)?;
    let n_off = gpu.icb_scalars.push_u32(n)?;
    dispatch_1d(&gpu.rt, &p, n as usize, |bnd| {
        set_gpu_buf(bnd, dst, 0);
        set_gpu_buf(bnd, src, 1);
        gpu.icb_scalars.bind_u32(bnd, n_off, 2);
    })
    .map_err(map_metal)
}

/// Legacy stub name — prefer typed dispatch above.
pub fn dispatch_stub(id: KernelId) -> Result<()> {
    let _ = id;
    Err(Error::Metal(
        "dispatch_stub removed — use GemmaGpu + typed kernel helpers".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{quantize_affine_f32, QuantScheme};

    fn gpu_or_fail() -> GemmaGpu {
        let g = match GemmaGpu::new() {
            Ok(g) => g,
            Err(e) => panic!("GPU test requires a Metal device: {e}"),
        };
        if let Err(e) = g.rt.pipeline(KernelId::GemvQ4.entry_name()) {
            panic!("GPU test requires a Metal pipeline: {e}");
        }
        g
    }

    #[test]
    fn registry_names_unique() {
        let mut names: Vec<_> = KernelId::all().iter().map(|k| k.entry_name()).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), KernelId::all().len());
    }

    #[test]
    fn rms_scalar_validation_rejects_invalid_domains() {
        let err = validate_rms_scalars("rms_test", 0, 1e-6, None).unwrap_err();
        assert!(err.to_string().contains("dim must be non-zero"));

        for eps in [0.0, -0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let err = validate_rms_scalars("rms_test", 1, eps, None).unwrap_err();
            assert!(
                err.to_string().contains("eps must be finite and > 0"),
                "eps={eps:?}: {err}"
            );
        }

        for scale in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let err = validate_rms_scalars("rms_test", 1, 1e-6, Some(scale)).unwrap_err();
            assert!(
                err.to_string().contains("layer_scale must be finite"),
                "layer_scale={scale:?}: {err}"
            );
        }
    }

    #[test]
    fn rms_scalar_validation_accepts_valid_boundaries() {
        for eps in [f32::MIN_POSITIVE, 1e-6, f32::MAX] {
            for scale in [None, Some(-1.0), Some(0.0), Some(f32::MAX)] {
                validate_rms_scalars("rms_test", 1, eps, scale).unwrap();
            }
        }
    }

    #[test]
    fn pipelines_resolve() {
        let gpu = gpu_or_fail();
        for id in KernelId::all() {
            // GEMM verify kernels are live in kernels/gemm_q4_mlx.metal (metallib).
            // Runtime gate remains buffer size + can_gemm_simd (cols>256).
            assert!(gpu.rt.pipeline(id.entry_name()).is_ok(), "{:?}", id);
        }
    }

    fn cpu_gemv(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut y = vec![0f32; rows];
        for r in 0..rows {
            let mut acc = 0f32;
            for c in 0..cols {
                acc += w[r * cols + c] * x[c];
            }
            y[r] = acc;
        }
        y
    }

    /// CPU oracle for the actual Hot MLX representation: the GPU stores each
    /// scale/bias as bf16, so round those values before dequantizing rows.
    fn cpu_gemv_q4_mlx_hot(q: &QuantMatrix, x: &[f32]) -> Vec<f32> {
        assert!(q.scheme.is_mlx_affine());
        assert_eq!(x.len(), q.cols);
        let mut hot = q.clone();
        for value in hot.scales.iter_mut().chain(hot.zeros.iter_mut()) {
            *value = crate::quant::bf16_bits_to_f32(crate::quant::f32_to_bf16_bits(*value));
        }
        (0..hot.rows)
            .map(|row| {
                hot.dequant_row(row)
                    .expect("validated Q4 MLX row")
                    .iter()
                    .zip(x)
                    .map(|(&w, &v)| w * v)
                    .sum()
            })
            .collect()
    }

    #[test]
    fn gemv_q4_matches_dequant() {
        let gpu = gpu_or_fail();
        let rows = 16usize;
        let cols = 64usize;
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.07)
            .collect();
        let q = quantize_affine_f32(rows, cols, &data, QuantScheme::q4_default()).unwrap();
        let w_dq = q.dequant_f32().unwrap();
        let x: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.01 - 0.2).collect();
        let expect = cpu_gemv(&w_dq, &x, rows, cols);
        let got = gemv_quant_host(&gpu, &q, &x).unwrap();
        let mut max_err = 0f32;
        for (a, b) in expect.iter().zip(got.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(max_err < 1e-4, "max_err={max_err}");
    }

    #[test]
    fn gemv_q4_mlx_rejects_invalid_boundary_before_scalar_or_gpu_work() {
        let gpu = gpu_or_fail();
        let packed = gpu.rt.alloc_buffer(32).unwrap();
        let short_packed = gpu.rt.alloc_buffer(31).unwrap();
        let scales_biases = gpu.rt.alloc_buffer(4).unwrap();
        let short_scales_biases = gpu.rt.alloc_buffer(3).unwrap();
        let ignored_biases = gpu.rt.alloc_buffer(1).unwrap();
        let x = gpu.rt.alloc_buffer(64 * std::mem::size_of::<f32>()).unwrap();
        let short_x = gpu.rt.alloc_buffer(63 * std::mem::size_of::<f32>()).unwrap();
        let y = gpu.rt.alloc_buffer(std::mem::size_of::<f32>()).unwrap();
        let short_y = gpu.rt.alloc_buffer(1).unwrap();

        let reject = |packed: &GpuBuffer,
                      scales_biases: &GpuBuffer,
                      x: &GpuBuffer,
                      y: &GpuBuffer,
                      rows: u32,
                      cols: u32,
                      group_size: u32,
                      needle: &str| {
            let cursor_before = gpu.icb_scalars.cursor_snapshot();
            let err = gemv_q4_mlx_with_simd_policy(
                &gpu,
                packed,
                scales_biases,
                &ignored_biases,
                x,
                y,
                rows,
                cols,
                group_size,
                false,
            )
            .expect_err("malformed MLX GEMV boundary must be rejected");
            assert!(
                err.to_string().contains(needle),
                "expected {needle:?}, got {err}"
            );
            assert_eq!(
                gpu.icb_scalars.cursor_snapshot(),
                cursor_before,
                "rejected inputs must not consume stable scalar slots"
            );
        };

        reject(
            &packed,
            &scales_biases,
            &x,
            &short_y,
            0,
            64,
            0,
            "group_size must be non-zero",
        );
        reject(
            &packed,
            &scales_biases,
            &x,
            &short_y,
            0,
            64,
            48,
            "is not a multiple",
        );
        reject(
            &short_packed,
            &scales_biases,
            &x,
            &y,
            1,
            64,
            64,
            "packed: buffer holds 31",
        );
        reject(
            &packed,
            &short_scales_biases,
            &x,
            &y,
            1,
            64,
            64,
            "scales_biases",
        );
        reject(
            &packed,
            &scales_biases,
            &short_x,
            &y,
            1,
            64,
            64,
            "x: buffer holds 63",
        );
        reject(
            &packed,
            &scales_biases,
            &x,
            &short_y,
            1,
            64,
            64,
            "y: buffer holds 0",
        );
        reject(
            &packed,
            &scales_biases,
            &x,
            &x,
            1,
            64,
            64,
            "overlaps read-only buffer x",
        );

        let foreign_rt = GpuRuntime::new_inference().unwrap();
        let foreign_packed = foreign_rt.alloc_buffer(32).unwrap();
        reject(
            &foreign_packed,
            &scales_biases,
            &x,
            &y,
            1,
            64,
            64,
            "belongs to another runtime",
        );

        let cursor_before = gpu.icb_scalars.cursor_snapshot();
        let err = gemv_q4(
            &gpu,
            &packed,
            &scales_biases,
            &scales_biases,
            &x,
            &short_y,
            0,
            64,
            0,
        )
        .expect_err("classic Q4 must reject group_size=0 before zero work");
        assert!(err.to_string().contains("group_size must be non-zero"));
        assert_eq!(gpu.icb_scalars.cursor_snapshot(), cursor_before);

        let cursor_before = gpu.icb_scalars.cursor_snapshot();
        let err = gemv_q4_mlx_blocked(
            &gpu,
            &packed,
            &scales_biases,
            &ignored_biases,
            &x,
            &short_y,
            0,
            64,
            0,
        )
        .expect_err("blocked Q4 MLX must reject group_size=0 before zero work");
        assert!(err.to_string().contains("group_size must be non-zero"));
        assert_eq!(gpu.icb_scalars.cursor_snapshot(), cursor_before);

        let cursor_before = gpu.icb_scalars.cursor_snapshot();
        gemv_q4_mlx_with_simd_policy(
            &gpu,
            &packed,
            &scales_biases,
            &ignored_biases,
            &x,
            &short_y,
            0,
            64,
            64,
            false,
        )
        .expect("valid zero-row GEMV is a no-op");
        assert_eq!(gpu.icb_scalars.cursor_snapshot(), cursor_before);
    }

    #[test]
    fn gemv_q4_mlx_blocked_matches_row_major() {
        let gpu = gpu_or_fail();
        if gpu.rt.pipeline("gemv_q4_mlx_blocked").is_err() {
            eprintln!("skip: gemv_q4_mlx_blocked not in metallib");
            return;
        }
        let rows = 24usize;
        let cols = 64usize;
        let group = 64usize;
        let groups = cols / group;
        let packs_per_row = cols / 8;
        let mut weight_u32 = vec![0u32; rows * packs_per_row];
        let mut scales = vec![0f32; rows * groups];
        let mut biases = vec![0f32; rows * groups];
        for r in 0..rows {
            for c in 0..cols {
                let nibble = ((r * 3 + c * 5) % 15) as u32;
                let wi = r * packs_per_row + c / 8;
                let shift = (c % 8) * 4;
                weight_u32[wi] |= nibble << shift;
            }
            for g in 0..groups {
                scales[r * groups + g] = 0.05 + (r as f32) * 0.001;
                biases[r * groups + g] = -0.2;
            }
        }
        let q = crate::quant::quant_matrix_from_mlx_q4(
            rows,
            cols,
            group,
            &weight_u32,
            &scales,
            &biases,
        )
        .unwrap();
        let x: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.01 - 0.15).collect();
        let expect = gemv_quant_host(&gpu, &q, &x).unwrap();

        let (p, s, z) =
            repack_q4_mlx_blocked(&q.packed, &q.scales, &q.zeros, q.rows, q.cols, group);
        let packed_b = gpu.rt.alloc_buffer_hot(p.len()).unwrap();
        packed_b.write_bytes(&p);
        let sb_bits = pack_mlx_sb_bf16(&s, &z);
        let scales_b = gpu.rt.alloc_buffer_hot(sb_bits.len() * 2).unwrap();
        scales_b.write_bf16_bits(&sb_bits);
        let zeros_b = gpu.rt.alloc_buffer_hot(4).unwrap();
        let banks = HotQuantBanks {
            scheme: q.scheme,
            layout: HotGemvLayout::BlockedBn16,
            rows: q.rows as u32,
            cols: q.cols as u32,
            group_size: group as u32,
            packed: packed_b,
            scales: scales_b,
            zeros: zeros_b,
        };
        let xb = gpu.rt.alloc_buffer(x.len() * 4).unwrap();
        xb.write_f32(&x);
        let yb = gpu.rt.alloc_buffer(rows * 4).unwrap();
        banks.gemv(&gpu, &xb, &yb).unwrap();
        gpu.synchronize().unwrap();
        let got = yb.read_f32();
        let mut max_err = 0f32;
        for (a, b) in expect.iter().zip(got.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(max_err < 1e-4, "blocked vs row-major max_err={max_err}");
    }

    /// Hot decode default `FUSE_BF16` feeds bf16 x into BlockedBn (float peel).
    /// Without expand, bits are reinterpreted as f32 → tok0 divergence.
    #[test]
    fn gemv_q4_mlx_blocked_bf16_x_matches_f32() {
        let gpu = gpu_or_fail();
        if gpu.rt.pipeline("gemv_q4_mlx_blocked").is_err() {
            eprintln!("skip: gemv_q4_mlx_blocked not in metallib");
            return;
        }
        let rows = 32usize;
        let cols = 256usize; // E4B-like group tiling; still ≤ GEMV_X_TILE
        let group = 64usize;
        let groups = cols / group;
        let packs_per_row = cols / 8;
        let mut weight_u32 = vec![0u32; rows * packs_per_row];
        let mut scales = vec![0f32; rows * groups];
        let mut biases = vec![0f32; rows * groups];
        for r in 0..rows {
            for c in 0..cols {
                let nibble = ((r * 3 + c * 5) % 15) as u32;
                let wi = r * packs_per_row + c / 8;
                let shift = (c % 8) * 4;
                weight_u32[wi] |= nibble << shift;
            }
            for g in 0..groups {
                scales[r * groups + g] = 0.05 + (r as f32) * 0.001;
                biases[r * groups + g] = -0.2;
            }
        }
        let q = crate::quant::quant_matrix_from_mlx_q4(
            rows,
            cols,
            group,
            &weight_u32,
            &scales,
            &biases,
        )
        .unwrap();
        let x: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.01 - 0.15).collect();
        let expect = gemv_quant_host(&gpu, &q, &x).unwrap();

        let (p, s, z) =
            repack_q4_mlx_blocked(&q.packed, &q.scales, &q.zeros, q.rows, q.cols, group);
        let packed_b = gpu.rt.alloc_buffer_hot(p.len()).unwrap();
        packed_b.write_bytes(&p);
        let sb_bits = pack_mlx_sb_bf16(&s, &z);
        let scales_b = gpu.rt.alloc_buffer_hot(sb_bits.len() * 2).unwrap();
        scales_b.write_bf16_bits(&sb_bits);
        let zeros_b = gpu.rt.alloc_buffer_hot(4).unwrap();
        let banks = HotQuantBanks {
            scheme: q.scheme,
            layout: HotGemvLayout::BlockedBn16,
            rows: q.rows as u32,
            cols: q.cols as u32,
            group_size: group as u32,
            packed: packed_b,
            scales: scales_b,
            zeros: zeros_b,
        };
        let xb = gpu.rt.alloc_buffer(x.len() * 4).unwrap();
        xb.write_f32(&x);
        let x_bf16 = prepare_act_bf16(&gpu, &xb, cols as u32).unwrap();
        let yb = gpu.rt.alloc_buffer(rows * 4).unwrap();
        banks.gemv_bf16_x(&gpu, &x_bf16, &yb).unwrap();
        gpu.synchronize().unwrap();
        let got = yb.read_f32();
        let mut max_err = 0f32;
        for (a, b) in expect.iter().zip(got.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(
            max_err < 2e-3,
            "blocked bf16_x vs f32 row-major max_err={max_err}"
        );
    }

    /// E4B-shaped + wide MLP-down (cols > GEMV_X_TILE → device-x peel).
    #[test]
    fn gemv_q4_mlx_blocked_e4b_shapes_match_cpu_oracle() {
        let gpu = gpu_or_fail();
        if gpu.rt.pipeline("gemv_q4_mlx_blocked").is_err() {
            eprintln!("skip: blocked gemv not in metallib");
            return;
        }
        let cases = [
            (32usize, 256usize), // smoke
            (64, 2560),          // Q-like cols
            (48, 2560),          // gate-like cols, non-BN multiple rows
            (32, 5120),          // > GEMV_X_TILE device-x
            (16, 10240),         // E4B down cols
        ];
        for (rows, cols) in cases {
            let group = 64usize;
            assert_eq!(cols % group, 0);
            let groups = cols / group;
            let packs_per_row = cols / 8;
            let mut weight_u32 = vec![0u32; rows * packs_per_row];
            let mut scales = vec![0f32; rows * groups];
            let mut biases = vec![0f32; rows * groups];
            for r in 0..rows {
                for c in 0..cols {
                    let nibble = ((r * 7 + c * 3) % 15) as u32;
                    let wi = r * packs_per_row + c / 8;
                    let shift = (c % 8) * 4;
                    weight_u32[wi] |= nibble << shift;
                }
                for g in 0..groups {
                    scales[r * groups + g] = 0.03 + (r as f32) * 0.0007 + (g as f32) * 0.0001;
                    biases[r * groups + g] = -0.11 - (g as f32) * 0.001;
                }
            }
            let q = crate::quant::quant_matrix_from_mlx_q4(
                rows,
                cols,
                group,
                &weight_u32,
                &scales,
                &biases,
            )
            .unwrap();
            let x: Vec<f32> = (0..cols).map(|i| ((i % 97) as f32) * 0.01 - 0.3).collect();

            // A 10,240-column row kernel would require 40,960 bytes of
            // dynamic threadgroup memory, exceeding the 32 KiB M5 limit.
            // Use an independent CPU oracle instead of an invalid GPU launch.
            let expect = cpu_gemv_q4_mlx_hot(&q, &x);
            let xb = gpu.rt.alloc_buffer(x.len() * 4).unwrap();
            xb.write_f32(&x);

            // Exercise the production non-SIMD fallback at the oversized
            // E4B-down shape. Before the fallback this exact shape reached
            // Binder with a 40,960-byte dynamic allocation and failed.
            if cols == 10_240 {
                let packed_rm = gpu.rt.alloc_buffer_hot(q.packed.len()).unwrap();
                packed_rm.write_bytes(&q.packed);
                let sb_rm = pack_mlx_sb_bf16(&q.scales, &q.zeros);
                let scales_rm = gpu.rt.alloc_buffer_hot(sb_rm.len() * 2).unwrap();
                scales_rm.write_bf16_bits(&sb_rm);
                let biases_rm = gpu.rt.alloc_buffer_hot(4).unwrap();
                let y_fallback = gpu.rt.alloc_buffer(rows * 4).unwrap();
                gemv_q4_mlx_with_simd_policy(
                    &gpu,
                    &packed_rm,
                    &scales_rm,
                    &biases_rm,
                    &xb,
                    &y_fallback,
                    rows as u32,
                    cols as u32,
                    group as u32,
                    false,
                )
                .unwrap();
                gpu.synchronize().unwrap();
                let fallback_err = max_abs_err(&expect, &y_fallback.read_f32());
                assert!(
                    fallback_err < 1e-3,
                    "tiled fallback vs CPU rows={rows} cols={cols} max_err={fallback_err}"
                );
            }

            let (p, s, z) =
                repack_q4_mlx_blocked(&q.packed, &q.scales, &q.zeros, q.rows, q.cols, group);
            let packed_b = gpu.rt.alloc_buffer_hot(p.len()).unwrap();
            packed_b.write_bytes(&p);
            let sb_b = pack_mlx_sb_bf16(&s, &z);
            let scales_b = gpu.rt.alloc_buffer_hot(sb_b.len() * 2).unwrap();
            scales_b.write_bf16_bits(&sb_b);
            let zeros_b = gpu.rt.alloc_buffer_hot(4).unwrap();
            let y_b = gpu.rt.alloc_buffer(rows * 4).unwrap();
            gemv_q4_mlx_blocked(
                &gpu,
                &packed_b,
                &scales_b,
                &zeros_b,
                &xb,
                &y_b,
                rows as u32,
                cols as u32,
                group as u32,
            )
            .unwrap();
            gpu.synchronize().unwrap();
            let got = y_b.read_f32();
            let mut max_err = 0f32;
            let mut worst = 0usize;
            for (i, (a, b)) in expect.iter().zip(got.iter()).enumerate() {
                let e = (a - b).abs();
                if e > max_err {
                    max_err = e;
                    worst = i;
                }
            }
            assert!(
                max_err < 1e-3,
                "blocked vs CPU rows={rows} cols={cols} max_err={max_err} @row={worst} expect={} got={}",
                expect[worst],
                got[worst]
            );
        }
    }

    /// Real E4B Hot weights: BlockedBn repack+kernel vs a bf16-bank CPU oracle.
    /// Opt-in (`GEMMA_METAL_BLOCKED_HOT_PARITY=1`) — loads HF cache once.
    #[test]
    fn gemv_q4_mlx_blocked_real_e4b_weights_match() {
        if std::env::var("GEMMA_METAL_BLOCKED_HOT_PARITY")
            .ok()
            .as_deref()
            != Some("1")
        {
            eprintln!("skip: set GEMMA_METAL_BLOCKED_HOT_PARITY=1 to run");
            return;
        }
        tessl::ab_flags::set_hazard_barriers(false);
        let gpu = gpu_or_fail();
        if gpu.rt.pipeline("gemv_q4_mlx_blocked").is_err() {
            eprintln!("skip: gemv_q4_mlx_blocked not in metallib");
            return;
        }
        let Some(dir) = crate::weights::resolve_default_e4b_mlx_cache() else {
            eprintln!("skip: no e4b mlx cache");
            return;
        };
        let banks = crate::weights::load_from_hf_dir(
            &dir,
            crate::weights::LoadOptions {
                scheme: QuantScheme::q4_mlx_default(),
                max_seq: 128,
                ..crate::weights::LoadOptions::default()
            },
        )
        .expect("load e4b");
        let names = [
            "layers.0.self_attn.q_proj.weight",
            "layers.0.self_attn.o_proj.weight",
            "layers.0.mlp.gate_proj.weight",
            "layers.0.mlp.up_proj.weight",
            "layers.0.mlp.down_proj.weight",
        ];
        for name in names {
            let q = banks.find(name).unwrap_or_else(|| panic!("missing {name}"));
            let group = q.scheme.group_size().expect("q4 group") as usize;
            let x: Vec<f32> = (0..q.cols)
                .map(|i| ((i * 17) % 89) as f32 * 0.01 - 0.25)
                .collect();
            let expect = cpu_gemv_q4_mlx_hot(q, &x);
            let xb = gpu.rt.alloc_buffer(x.len() * 4).unwrap();
            xb.write_f32(&x);

            let (p, s, z) =
                repack_q4_mlx_blocked(&q.packed, &q.scales, &q.zeros, q.rows, q.cols, group);
            let packed_b = gpu.rt.alloc_buffer_hot(p.len()).unwrap();
            packed_b.write_bytes(&p);
            let sb = pack_mlx_sb_bf16(&s, &z);
            let scales_b = gpu.rt.alloc_buffer_hot(sb.len() * 2).unwrap();
            scales_b.write_bf16_bits(&sb);
            let zeros_b = gpu.rt.alloc_buffer_hot(4).unwrap();
            let banks_b = HotQuantBanks {
                scheme: q.scheme,
                layout: HotGemvLayout::BlockedBn16,
                rows: q.rows as u32,
                cols: q.cols as u32,
                group_size: group as u32,
                packed: packed_b,
                scales: scales_b,
                zeros: zeros_b,
            };
            let yb = gpu.rt.alloc_buffer(q.rows * 4).unwrap();
            banks_b.gemv(&gpu, &xb, &yb).unwrap();
            gpu.synchronize().unwrap();
            let got = yb.read_f32();
            let mut max_err = 0f32;
            let mut worst = 0usize;
            for (i, (a, b)) in expect.iter().zip(got.iter()).enumerate() {
                let e = (a - b).abs();
                if e > max_err {
                    max_err = e;
                    worst = i;
                }
            }
            println!(
                "blocked_hot_parity {name} [{},{}] max_err={max_err:.6e} @row={worst}",
                q.rows, q.cols
            );
            assert!(
                max_err < 2e-3,
                "{name} blocked vs CPU Hot max_err={max_err}"
            );
        }
    }

    #[test]
    fn gemv_q4_mlx_blocked_gate_up_gelu_bf16_x_matches() {
        let gpu = gpu_or_fail();
        if gpu.rt.pipeline("gemv_q4_mlx_blocked_gate_up_gelu").is_err() {
            eprintln!("skip: fused gate_up_gelu not in metallib");
            return;
        }
        let rows = 32usize;
        let cols = 256usize;
        let group = 64usize;
        let groups = cols / group;
        let packs_per_row = cols / 8;
        let mk_q = |seed: u32| {
            let mut weight_u32 = vec![0u32; rows * packs_per_row];
            let mut scales = vec![0f32; rows * groups];
            let mut biases = vec![0f32; rows * groups];
            for r in 0..rows {
                for c in 0..cols {
                    let nibble = ((r as u32 * 3 + c as u32 * 5 + seed) % 15) as u32;
                    let wi = r * packs_per_row + c / 8;
                    let shift = (c % 8) * 4;
                    weight_u32[wi] |= nibble << shift;
                }
                for g in 0..groups {
                    scales[r * groups + g] = 0.04 + (r as f32) * 0.001 + seed as f32 * 0.01;
                    biases[r * groups + g] = -0.15 - seed as f32 * 0.01;
                }
            }
            crate::quant::quant_matrix_from_mlx_q4(rows, cols, group, &weight_u32, &scales, &biases)
                .unwrap()
        };
        let qg = mk_q(1);
        let qu = mk_q(2);
        let x: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.01 - 0.1).collect();
        let gate_y = gemv_quant_host(&gpu, &qg, &x).unwrap();
        let up_y = gemv_quant_host(&gpu, &qu, &x).unwrap();
        let expect: Vec<f32> = gate_y
            .iter()
            .zip(up_y.iter())
            .map(|(g, u)| gelu_pytorch_tanh(*g) * *u)
            .collect();

        let (p, s, z) =
            repack_q4_mlx_blocked(&qg.packed, &qg.scales, &qg.zeros, qg.rows, qg.cols, group);
        let (pu, su, zu) =
            repack_q4_mlx_blocked(&qu.packed, &qu.scales, &qu.zeros, qu.rows, qu.cols, group);
        let mk_banks = |p: Vec<u8>, s: Vec<f32>, z: Vec<f32>, q: &QuantMatrix| {
            let packed_b = gpu.rt.alloc_buffer_hot(p.len()).unwrap();
            packed_b.write_bytes(&p);
            let sb_bits = pack_mlx_sb_bf16(&s, &z);
            let scales_b = gpu.rt.alloc_buffer_hot(sb_bits.len() * 2).unwrap();
            scales_b.write_bf16_bits(&sb_bits);
            let zeros_b = gpu.rt.alloc_buffer_hot(4).unwrap();
            HotQuantBanks {
                scheme: q.scheme,
                layout: HotGemvLayout::BlockedBn16,
                rows: q.rows as u32,
                cols: q.cols as u32,
                group_size: group as u32,
                packed: packed_b,
                scales: scales_b,
                zeros: zeros_b,
            }
        };
        std::env::set_var("GEMMA_METAL_FUSE_MLP", "1");
        let gate = mk_banks(p, s, z, &qg);
        let up = mk_banks(pu, su, zu, &qu);
        assert!(gate.can_fuse_gate_up_gelu(&up));
        let xb = gpu.rt.alloc_buffer(x.len() * 4).unwrap();
        xb.write_f32(&x);
        let x_bf16 = prepare_act_bf16(&gpu, &xb, cols as u32).unwrap();
        let mid = gpu.rt.alloc_buffer(rows * 4).unwrap();
        gemv_q4_mlx_gate_up_gelu_bf16_x(&gpu, &gate, &up, &x_bf16, &mid).unwrap();
        gpu.synchronize().unwrap();
        let got = mid.read_f32();
        let mut max_err = 0f32;
        for (a, b) in expect.iter().zip(got.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(
            max_err < 2e-3,
            "blocked gate_up_gelu bf16_x max_err={max_err}"
        );
    }

    #[test]
    fn gemv_q4_mlx_blocked_gate_up_gelu_matches() {
        let gpu = gpu_or_fail();
        if gpu.rt.pipeline("gemv_q4_mlx_blocked_gate_up_gelu").is_err() {
            eprintln!("skip: fused gate_up_gelu not in metallib");
            return;
        }
        let rows = 16usize;
        let cols = 64usize;
        let group = 64usize;
        let groups = cols / group;
        let packs_per_row = cols / 8;
        let mk_q = |seed: u32| {
            let mut weight_u32 = vec![0u32; rows * packs_per_row];
            let mut scales = vec![0f32; rows * groups];
            let mut biases = vec![0f32; rows * groups];
            for r in 0..rows {
                for c in 0..cols {
                    let nibble = ((r as u32 * 3 + c as u32 * 5 + seed) % 15) as u32;
                    let wi = r * packs_per_row + c / 8;
                    let shift = (c % 8) * 4;
                    weight_u32[wi] |= nibble << shift;
                }
                for g in 0..groups {
                    scales[r * groups + g] = 0.04 + (r as f32) * 0.001 + seed as f32 * 0.01;
                    biases[r * groups + g] = -0.15 - seed as f32 * 0.01;
                }
            }
            crate::quant::quant_matrix_from_mlx_q4(rows, cols, group, &weight_u32, &scales, &biases)
                .unwrap()
        };
        let qg = mk_q(1);
        let qu = mk_q(2);
        let x: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.01 - 0.1).collect();
        let gate_y = gemv_quant_host(&gpu, &qg, &x).unwrap();
        let up_y = gemv_quant_host(&gpu, &qu, &x).unwrap();
        let expect: Vec<f32> = gate_y
            .iter()
            .zip(up_y.iter())
            .map(|(g, u)| gelu_pytorch_tanh(*g) * *u)
            .collect();

        let (p, s, z) =
            repack_q4_mlx_blocked(&qg.packed, &qg.scales, &qg.zeros, qg.rows, qg.cols, group);
        let (pu, su, zu) =
            repack_q4_mlx_blocked(&qu.packed, &qu.scales, &qu.zeros, qu.rows, qu.cols, group);
        let mk_banks = |p: Vec<u8>, s: Vec<f32>, z: Vec<f32>, q: &QuantMatrix| {
            let packed_b = gpu.rt.alloc_buffer_hot(p.len()).unwrap();
            packed_b.write_bytes(&p);
            let sb_bits = pack_mlx_sb_bf16(&s, &z);
            let scales_b = gpu.rt.alloc_buffer_hot(sb_bits.len() * 2).unwrap();
            scales_b.write_bf16_bits(&sb_bits);
            let zeros_b = gpu.rt.alloc_buffer_hot(4).unwrap();
            HotQuantBanks {
                scheme: q.scheme,
                layout: HotGemvLayout::BlockedBn16,
                rows: q.rows as u32,
                cols: q.cols as u32,
                group_size: group as u32,
                packed: packed_b,
                scales: scales_b,
                zeros: zeros_b,
            }
        };
        // Force fuse on for this unit test.
        std::env::set_var("GEMMA_METAL_FUSE_MLP", "1");
        let gate = mk_banks(p, s, z, &qg);
        let up = mk_banks(pu, su, zu, &qu);
        assert!(gate.can_fuse_gate_up_gelu(&up));
        let xb = gpu.rt.alloc_buffer(x.len() * 4).unwrap();
        xb.write_f32(&x);
        let mid = gpu.rt.alloc_buffer(rows * 4).unwrap();
        gemv_q4_mlx_blocked_gate_up_gelu(&gpu, &gate, &up, &xb, &mid, false).unwrap();
        gpu.synchronize().unwrap();
        let got = mid.read_f32();
        let mut max_err = 0f32;
        for (a, b) in expect.iter().zip(got.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(max_err < 1e-3, "fused gate_up_gelu max_err={max_err}");
    }

    #[test]
    fn gemv_q4_mlx_simd_matches_row_major() {
        let gpu = gpu_or_fail();
        if gpu.rt.pipeline("gemv_q4_mlx_simd").is_err() {
            eprintln!("skip: gemv_q4_mlx_simd not in metallib");
            return;
        }
        let rows = 32usize;
        let cols = 256usize;
        let group = 64usize;
        let groups = cols / group;
        let packs_per_row = cols / 8;
        let mut weight_u32 = vec![0u32; rows * packs_per_row];
        let mut scales = vec![0f32; rows * groups];
        let mut biases = vec![0f32; rows * groups];
        for r in 0..rows {
            for c in 0..cols {
                let nibble = ((r * 3 + c * 5) % 15) as u32;
                let wi = r * packs_per_row + c / 8;
                let shift = (c % 8) * 4;
                weight_u32[wi] |= nibble << shift;
            }
            for g in 0..groups {
                scales[r * groups + g] = 0.05 + (r as f32) * 0.001;
                biases[r * groups + g] = -0.2;
            }
        }
        let q = crate::quant::quant_matrix_from_mlx_q4(
            rows,
            cols,
            group,
            &weight_u32,
            &scales,
            &biases,
        )
        .unwrap();
        let x: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.01 - 0.15).collect();

        // Direct path calls (avoid GEMMA_METAL_GEMV_SIMD OnceLock toggle).
        let packed = gpu.rt.alloc_buffer(q.packed.len().max(1)).unwrap();
        packed.write_bytes(&q.packed);
        let sb = pack_mlx_sb_bf16(&q.scales, &q.zeros);
        let scales_b = gpu.rt.alloc_buffer(sb.len().max(1) * 2).unwrap();
        scales_b.write_bf16_bits(&sb);
        let zeros_b = gpu.rt.alloc_buffer(4).unwrap();
        let xb = gpu.rt.alloc_buffer(x.len() * 4).unwrap();
        xb.write_f32(&x);
        let y_row = gpu.rt.alloc_buffer(rows * 4).unwrap();
        let y_simd = gpu.rt.alloc_buffer(rows * 4).unwrap();
        gemv_q4_mlx_with_simd_policy(
            &gpu,
            &packed,
            &scales_b,
            &zeros_b,
            &xb,
            &y_row,
            rows as u32,
            cols as u32,
            group as u32,
            false,
        )
        .unwrap();
        let x_bf16 = prepare_act_bf16(&gpu, &xb, cols as u32).unwrap();
        dispatch_gemv_simd(
            &gpu,
            &packed,
            &scales_b,
            &zeros_b,
            &x_bf16,
            &y_simd,
            rows as u32,
            cols as u32,
            group as u32,
            false,
        )
        .unwrap();
        gpu.synchronize().unwrap();
        let expect = y_row.read_f32();
        let got = y_simd.read_f32();
        let mut max_err = 0f32;
        for (a, b) in expect.iter().zip(got.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(
            max_err < 2e-2,
            "simd (bf16 x) vs row (f32 x) max_err={max_err}"
        );
        // CPU check uses f32 x + f32 scales; allow bf16 path slack.
        let w_dq = q.dequant_f32().unwrap();
        let cpu = cpu_gemv(&w_dq, &x, rows, cols);
        let mut cpu_err = 0f32;
        for (a, b) in cpu.iter().zip(got.iter()) {
            cpu_err = cpu_err.max((a - b).abs());
        }
        assert!(cpu_err < 0.75, "simd vs cpu dequant GEMV max_err={cpu_err}");
    }

    /// Producer-shaped RowMajor Q4Mlx banks for QKV fusion parity tests.
    /// `rows_q` may exceed `rows_kv` (GQA); `cols >= 256` and `% 16 == 0` for
    /// [`HotQuantBanks::can_fuse_qkv`].
    fn mk_q4_mlx_row_major_banks(
        gpu: &GemmaGpu,
        rows: usize,
        cols: usize,
        group: usize,
        seed: u32,
    ) -> HotQuantBanks {
        let groups = cols / group;
        let packs_per_row = cols / 8;
        let mut weight_u32 = vec![0u32; rows * packs_per_row];
        let mut scales = vec![0f32; rows * groups];
        let mut biases = vec![0f32; rows * groups];
        for r in 0..rows {
            for c in 0..cols {
                let nibble = ((r as u32 * 3 + c as u32 * 5 + seed) % 15) as u32;
                let wi = r * packs_per_row + c / 8;
                let shift = (c % 8) * 4;
                weight_u32[wi] |= nibble << shift;
            }
            for g in 0..groups {
                scales[r * groups + g] = 0.04 + (r as f32) * 0.001 + seed as f32 * 0.01;
                biases[r * groups + g] = -0.15 - seed as f32 * 0.01;
            }
        }
        let q = crate::quant::quant_matrix_from_mlx_q4(
            rows,
            cols,
            group,
            &weight_u32,
            &scales,
            &biases,
        )
        .unwrap();
        let packed = gpu.rt.alloc_buffer(q.packed.len().max(1)).unwrap();
        packed.write_bytes(&q.packed);
        let sb = pack_mlx_sb_bf16(&q.scales, &q.zeros);
        let scales_b = gpu.rt.alloc_buffer(sb.len().max(1) * 2).unwrap();
        scales_b.write_bf16_bits(&sb);
        let zeros_b = gpu.rt.alloc_buffer(4).unwrap();
        HotQuantBanks {
            scheme: q.scheme,
            layout: HotGemvLayout::RowMajor,
            rows: q.rows as u32,
            cols: q.cols as u32,
            group_size: group as u32,
            packed,
            scales: scales_b,
            zeros: zeros_b,
        }
    }

    /// Same synthetic weights as [`mk_q4_mlx_row_major_banks`], uploaded as Interleaved4.
    fn mk_q4_mlx_interleaved4_banks(
        gpu: &GemmaGpu,
        rows: usize,
        cols: usize,
        group: usize,
        seed: u32,
    ) -> HotQuantBanks {
        let groups = cols / group;
        let packs_per_row = cols / 8;
        let mut weight_u32 = vec![0u32; rows * packs_per_row];
        let mut scales = vec![0f32; rows * groups];
        let mut biases = vec![0f32; rows * groups];
        for r in 0..rows {
            for c in 0..cols {
                let nibble = ((r as u32 * 3 + c as u32 * 5 + seed) % 15) as u32;
                let wi = r * packs_per_row + c / 8;
                let shift = (c % 8) * 4;
                weight_u32[wi] |= nibble << shift;
            }
            for g in 0..groups {
                scales[r * groups + g] = 0.04 + (r as f32) * 0.001 + seed as f32 * 0.01;
                biases[r * groups + g] = -0.15 - seed as f32 * 0.01;
            }
        }
        let q = crate::quant::quant_matrix_from_mlx_q4(
            rows,
            cols,
            group,
            &weight_u32,
            &scales,
            &biases,
        )
        .unwrap();
        let (packed_i4, scales_i4, biases_i4) =
            repack_q4_mlx_interleaved4(&q.packed, &q.scales, &q.zeros, rows, cols, group);
        let packed = gpu.rt.alloc_buffer(packed_i4.len().max(1)).unwrap();
        packed.write_bytes(&packed_i4);
        let sb = pack_mlx_sb_bf16(&scales_i4, &biases_i4);
        let scales_b = gpu.rt.alloc_buffer(sb.len().max(1) * 2).unwrap();
        scales_b.write_bf16_bits(&sb);
        let zeros_b = gpu.rt.alloc_buffer(4).unwrap();
        HotQuantBanks {
            scheme: q.scheme,
            layout: HotGemvLayout::Interleaved4,
            rows: q.rows as u32,
            cols: q.cols as u32,
            group_size: group as u32,
            packed,
            scales: scales_b,
            zeros: zeros_b,
        }
    }

    fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    }

    /// Layer-fusion v1: fused `gemv_q4_mlx_simd_qkv` vs unfused `gemv_q` + `gemv_kv`.
    /// Math is the same simd walk — expect bit-exact floats.
    #[test]
    fn gemv_q4_mlx_simd_qkv_matches_unfused() {
        let gpu = gpu_or_fail();
        if gpu.rt.pipeline("gemv_q4_mlx_simd_qkv").is_err() {
            eprintln!("skip: gemv_q4_mlx_simd_qkv not in metallib");
            return;
        }
        // Opt-in fusion via setter — env::set_var is unreliable here because the
        // flag cache may already be initialized by an earlier test's decode path.
        set_fuse_qkv(true);

        // Producer-shaped GQA: rows_q > rows_kv, cols meets simd fuse gate.
        let rows_q = 32usize;
        let rows_kv = 16usize;
        let cols = 256usize;
        let group = 64usize;
        let q = mk_q4_mlx_row_major_banks(&gpu, rows_q, cols, group, 1);
        let k = mk_q4_mlx_row_major_banks(&gpu, rows_kv, cols, group, 2);
        let v = mk_q4_mlx_row_major_banks(&gpu, rows_kv, cols, group, 3);
        assert!(
            q.can_fuse_qkv(&k, &v),
            "can_fuse_qkv failed (FUSE_QKV / GEMV_SIMD / RowMajor?)"
        );

        let x: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.01 - 0.12).collect();
        let xb = gpu.rt.alloc_buffer(x.len() * 4).unwrap();
        xb.write_f32(&x);
        let x_bf16 = prepare_act_bf16(&gpu, &xb, cols as u32).unwrap();

        let q_ref = gpu.rt.alloc_buffer(rows_q * 4).unwrap();
        let k_ref = gpu.rt.alloc_buffer(rows_kv * 4).unwrap();
        let v_ref = gpu.rt.alloc_buffer(rows_kv * 4).unwrap();
        q.gemv_bf16_x(&gpu, &x_bf16, &q_ref).unwrap();
        gemv_q4_mlx_simd_kv_bf16_x(&gpu, &k, &v, &x_bf16, &k_ref, &v_ref).unwrap();

        let q_fused = gpu.rt.alloc_buffer(rows_q * 4).unwrap();
        let k_fused = gpu.rt.alloc_buffer(rows_kv * 4).unwrap();
        let v_fused = gpu.rt.alloc_buffer(rows_kv * 4).unwrap();
        gemv_q4_mlx_simd_qkv_bf16_x(&gpu, &q, &k, &v, &x_bf16, &q_fused, &k_fused, &v_fused)
            .unwrap();
        gpu.synchronize().unwrap();

        let err_q = max_abs_err(&q_ref.read_f32(), &q_fused.read_f32());
        let err_k = max_abs_err(&k_ref.read_f32(), &k_fused.read_f32());
        let err_v = max_abs_err(&v_ref.read_f32(), &v_fused.read_f32());
        assert!(
            err_q == 0.0 && err_k == 0.0 && err_v == 0.0,
            "fused qkv vs gemv_q+gemv_kv max_err q={err_q} k={err_k} v={err_v}"
        );
        set_fuse_qkv(false);
    }

    /// E4B producer dims (q=2048/kv=512 and q=4096/kv=1024 @ cols=2560).
    /// Hot `fusion_ab` exactness regressed here while the tiny unit dims still pass.
    #[test]
    fn gemv_q4_mlx_simd_qkv_e4b_dims_matches_unfused() {
        let gpu = gpu_or_fail();
        if gpu.rt.pipeline("gemv_q4_mlx_simd_qkv").is_err() {
            eprintln!("skip: gemv_q4_mlx_simd_qkv not in metallib");
            return;
        }
        set_fuse_qkv(true);
        let cols = 2560usize;
        let group = 64usize;
        for (rows_q, rows_kv, seed) in [(2048usize, 512usize, 21u32), (4096, 1024, 22)] {
            let q = mk_q4_mlx_row_major_banks(&gpu, rows_q, cols, group, seed);
            let k = mk_q4_mlx_row_major_banks(&gpu, rows_kv, cols, group, seed + 1);
            let v = mk_q4_mlx_row_major_banks(&gpu, rows_kv, cols, group, seed + 2);
            assert!(
                q.can_fuse_qkv(&k, &v),
                "can_fuse_qkv failed at q={rows_q} kv={rows_kv}"
            );

            let x: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.01 - 0.12).collect();
            let xb = gpu.rt.alloc_buffer(x.len() * 4).unwrap();
            xb.write_f32(&x);
            let x_bf16 = prepare_act_bf16(&gpu, &xb, cols as u32).unwrap();

            let q_ref = gpu.rt.alloc_buffer(rows_q * 4).unwrap();
            let k_ref = gpu.rt.alloc_buffer(rows_kv * 4).unwrap();
            let v_ref = gpu.rt.alloc_buffer(rows_kv * 4).unwrap();
            q.gemv_bf16_x(&gpu, &x_bf16, &q_ref).unwrap();
            gemv_q4_mlx_simd_kv_bf16_x(&gpu, &k, &v, &x_bf16, &k_ref, &v_ref).unwrap();

            let q_fused = gpu.rt.alloc_buffer(rows_q * 4).unwrap();
            let k_fused = gpu.rt.alloc_buffer(rows_kv * 4).unwrap();
            let v_fused = gpu.rt.alloc_buffer(rows_kv * 4).unwrap();
            gemv_q4_mlx_simd_qkv_bf16_x(&gpu, &q, &k, &v, &x_bf16, &q_fused, &k_fused, &v_fused)
                .unwrap();
            gpu.synchronize().unwrap();

            let err_q = max_abs_err(&q_ref.read_f32(), &q_fused.read_f32());
            let err_k = max_abs_err(&k_ref.read_f32(), &k_fused.read_f32());
            let err_v = max_abs_err(&v_ref.read_f32(), &v_fused.read_f32());
            assert!(
                err_q == 0.0 && err_k == 0.0 && err_v == 0.0,
                "E4B-dim fused qkv q={rows_q} kv={rows_kv} max_err q={err_q} k={err_k} v={err_v}"
            );
        }
        set_fuse_qkv(false);
    }

    /// 31B global `attention_k_eq_v`: V banks share K packed/sb pointers, two outs.
    /// Fused kernel must not reject or corrupt when K/V buffers alias.
    #[test]
    fn gemv_q4_mlx_simd_qkv_tied_kv_matches() {
        let gpu = gpu_or_fail();
        if gpu.rt.pipeline("gemv_q4_mlx_simd_qkv").is_err() {
            eprintln!("skip: gemv_q4_mlx_simd_qkv not in metallib");
            return;
        }
        set_fuse_qkv(true);

        let rows_q = 32usize;
        let rows_kv = 16usize;
        let cols = 256usize;
        let group = 64usize;
        let q = mk_q4_mlx_row_major_banks(&gpu, rows_q, cols, group, 7);
        let k = mk_q4_mlx_row_major_banks(&gpu, rows_kv, cols, group, 11);
        // Mirror gpu_model.rs k_eq_v load: same packed/sb Arc, distinct HotQuantBanks.
        let v = HotQuantBanks {
            scheme: k.scheme,
            layout: k.layout,
            rows: k.rows,
            cols: k.cols,
            group_size: k.group_size,
            packed: k.packed.clone(),
            scales: k.scales.clone(),
            zeros: k.zeros.clone(),
        };
        assert!(q.can_fuse_qkv(&k, &v));
        assert!(
            std::ptr::eq(k.packed.metal(), v.packed.metal()),
            "tied k_eq_v must share packed Metal buffer"
        );

        let x: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.01 - 0.08).collect();
        let xb = gpu.rt.alloc_buffer(x.len() * 4).unwrap();
        xb.write_f32(&x);
        let x_bf16 = prepare_act_bf16(&gpu, &xb, cols as u32).unwrap();

        let q_ref = gpu.rt.alloc_buffer(rows_q * 4).unwrap();
        let k_ref = gpu.rt.alloc_buffer(rows_kv * 4).unwrap();
        let v_ref = gpu.rt.alloc_buffer(rows_kv * 4).unwrap();
        q.gemv_bf16_x(&gpu, &x_bf16, &q_ref).unwrap();
        // Unfused: two distinct outs from the same weight bank (as decode does).
        k.gemv_bf16_x(&gpu, &x_bf16, &k_ref).unwrap();
        v.gemv_bf16_x(&gpu, &x_bf16, &v_ref).unwrap();

        let q_fused = gpu.rt.alloc_buffer(rows_q * 4).unwrap();
        let k_fused = gpu.rt.alloc_buffer(rows_kv * 4).unwrap();
        let v_fused = gpu.rt.alloc_buffer(rows_kv * 4).unwrap();
        gemv_q4_mlx_simd_qkv_bf16_x(&gpu, &q, &k, &v, &x_bf16, &q_fused, &k_fused, &v_fused)
            .unwrap();
        gpu.synchronize().unwrap();

        let err_q = max_abs_err(&q_ref.read_f32(), &q_fused.read_f32());
        let err_k = max_abs_err(&k_ref.read_f32(), &k_fused.read_f32());
        let err_v = max_abs_err(&v_ref.read_f32(), &v_fused.read_f32());
        let k_host = k_ref.read_f32();
        let v_host = v_ref.read_f32();
        let kv_tie_err = max_abs_err(&k_host, &v_host);
        assert!(
            err_q == 0.0 && err_k == 0.0 && err_v == 0.0,
            "tied k_eq_v fused qkv max_err q={err_q} k={err_k} v={err_v}"
        );
        // Same weights ⇒ K and V outs must match (and fused must preserve that).
        assert!(
            kv_tie_err == 0.0,
            "tied k_eq_v: unfused k vs v outs differ max_err={kv_tie_err}"
        );
        assert!(
            max_abs_err(&k_fused.read_f32(), &v_fused.read_f32()) == 0.0,
            "tied k_eq_v: fused k vs v outs differ"
        );
        set_fuse_qkv(false);
    }

    /// Interleaved4 twin: fused `gemv_q4_mlx_simd_qkv_i4` vs unfused i4 `gemv_q` + `gemv_kv`.
    #[test]
    fn gemv_q4_mlx_simd_qkv_i4_matches_unfused() {
        let gpu = gpu_or_fail();
        if gpu.rt.pipeline("gemv_q4_mlx_simd_qkv_i4").is_err() {
            eprintln!("skip: gemv_q4_mlx_simd_qkv_i4 not in metallib");
            return;
        }
        set_fuse_qkv(true);

        let rows_q = 32usize;
        let rows_kv = 16usize;
        let cols = 256usize;
        let group = 64usize;
        let q = mk_q4_mlx_interleaved4_banks(&gpu, rows_q, cols, group, 1);
        let k = mk_q4_mlx_interleaved4_banks(&gpu, rows_kv, cols, group, 2);
        let v = mk_q4_mlx_interleaved4_banks(&gpu, rows_kv, cols, group, 3);
        assert!(
            q.can_fuse_qkv(&k, &v),
            "can_fuse_qkv failed for Interleaved4 (FUSE_QKV / GEMV_SIMD?)"
        );
        assert_eq!(q.layout, HotGemvLayout::Interleaved4);

        let x: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.01 - 0.12).collect();
        let xb = gpu.rt.alloc_buffer(x.len() * 4).unwrap();
        xb.write_f32(&x);
        let x_bf16 = prepare_act_bf16(&gpu, &xb, cols as u32).unwrap();

        let q_ref = gpu.rt.alloc_buffer(rows_q * 4).unwrap();
        let k_ref = gpu.rt.alloc_buffer(rows_kv * 4).unwrap();
        let v_ref = gpu.rt.alloc_buffer(rows_kv * 4).unwrap();
        q.gemv_bf16_x(&gpu, &x_bf16, &q_ref).unwrap();
        gemv_q4_mlx_simd_kv_bf16_x(&gpu, &k, &v, &x_bf16, &k_ref, &v_ref).unwrap();

        let q_fused = gpu.rt.alloc_buffer(rows_q * 4).unwrap();
        let k_fused = gpu.rt.alloc_buffer(rows_kv * 4).unwrap();
        let v_fused = gpu.rt.alloc_buffer(rows_kv * 4).unwrap();
        gemv_q4_mlx_simd_qkv_bf16_x(&gpu, &q, &k, &v, &x_bf16, &q_fused, &k_fused, &v_fused)
            .unwrap();
        gpu.synchronize().unwrap();

        let err_q = max_abs_err(&q_ref.read_f32(), &q_fused.read_f32());
        let err_k = max_abs_err(&k_ref.read_f32(), &k_fused.read_f32());
        let err_v = max_abs_err(&v_ref.read_f32(), &v_fused.read_f32());
        assert!(
            err_q == 0.0 && err_k == 0.0 && err_v == 0.0,
            "fused qkv_i4 vs gemv_i4+gemv_kv_i4 max_err q={err_q} k={err_k} v={err_v}"
        );
        set_fuse_qkv(false);
    }

    #[test]
    fn gemv_q8_matches_dequant() {
        let gpu = gpu_or_fail();
        let rows = 8usize;
        let cols = 32usize;
        let data: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.02 - 0.5).collect();
        let q = quantize_affine_f32(rows, cols, &data, QuantScheme::q8_default()).unwrap();
        let w_dq = q.dequant_f32().unwrap();
        let x: Vec<f32> = (0..cols).map(|i| ((i % 5) as f32) * 0.1).collect();
        let expect = cpu_gemv(&w_dq, &x, rows, cols);
        let got = gemv_quant_host(&gpu, &q, &x).unwrap();
        let mut max_err = 0f32;
        for (a, b) in expect.iter().zip(got.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(max_err < 1e-4, "max_err={max_err}");
    }

    fn gelu_pytorch_tanh(x: f32) -> f32 {
        let k = (2.0f32 / std::f32::consts::PI).sqrt();
        let xc = x.clamp(-20.0, 20.0);
        let inner = (k * (xc + 0.044715 * xc * xc * xc)).clamp(-10.0, 10.0);
        0.5 * x * (1.0 + inner.tanh())
    }

    #[test]
    fn mlp_gelu_tanh_gpu() {
        let gpu = gpu_or_fail();
        let n = 128usize;
        let gate: Vec<f32> = (0..n).map(|i| (i as f32) * 0.03 - 1.0).collect();
        let up: Vec<f32> = (0..n).map(|i| ((i % 7) as f32) * 0.2 - 0.5).collect();
        let expect: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(&g, &u)| gelu_pytorch_tanh(g) * u)
            .collect();
        let gb = gpu.rt.alloc_buffer(n * 4).unwrap();
        let ub = gpu.rt.alloc_buffer(n * 4).unwrap();
        let ob = gpu.rt.alloc_buffer(n * 4).unwrap();
        gb.write_f32(&gate);
        ub.write_f32(&up);
        mlp_gelu_tanh(&gpu, &gb, &ub, &ob, n as u32).unwrap();
        gpu.synchronize().unwrap();
        let got = ob.read_f32();
        let mut max_err = 0f32;
        for (a, b) in expect.iter().zip(got.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(max_err < 1e-5, "max_err={max_err}");
    }

    /// Large |gate| hits gelu inner ≈±300 — previously Metal `fast_tanh` NaN'd.
    #[test]
    fn mlp_gelu_tanh_gpu_large_finite() {
        let gpu = gpu_or_fail();
        let gate = vec![-25.0f32, -15.0, -8.0, -1.0, 0.0, 1.0, 8.0, 15.0, 25.0, 40.0];
        let up = vec![1.0f32, -0.5, 2.0, 0.0, 3.0, -1.0, 0.25, 1.5, -2.0, 0.1];
        let n = gate.len();
        let expect: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(&g, &u)| gelu_pytorch_tanh(g) * u)
            .collect();
        let gb = gpu.rt.alloc_buffer(n * 4).unwrap();
        let ub = gpu.rt.alloc_buffer(n * 4).unwrap();
        let ob = gpu.rt.alloc_buffer(n * 4).unwrap();
        gb.write_f32(&gate);
        ub.write_f32(&up);
        mlp_gelu_tanh(&gpu, &gb, &ub, &ob, n as u32).unwrap();
        gpu.synchronize().unwrap();
        let got = ob.read_f32();
        let nan = got.iter().filter(|v| v.is_nan()).count();
        assert_eq!(nan, 0, "gpu mid has {nan} NaNs for large gate");
        let mut max_err = 0f32;
        for (a, b) in expect.iter().zip(got.iter()) {
            assert!(b.is_finite(), "non-finite gpu mid {b}");
            max_err = max_err.max((a - b).abs());
        }
        assert!(max_err < 1e-4, "max_err={max_err}");
    }

    #[test]
    fn mlp_silu_gpu() {
        let gpu = gpu_or_fail();
        if gpu.rt.pipeline(KernelId::MlpSilu.entry_name()).is_err() {
            eprintln!("skip: mlp_silu not in metallib");
            return;
        }
        let n = 128usize;
        let gate: Vec<f32> = (0..n).map(|i| (i as f32) * 0.03 - 1.0).collect();
        let up: Vec<f32> = (0..n).map(|i| ((i % 7) as f32) * 0.2 - 0.5).collect();
        let expect: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
            .collect();
        let gb = gpu.rt.alloc_buffer(n * 4).unwrap();
        let ub = gpu.rt.alloc_buffer(n * 4).unwrap();
        let ob = gpu.rt.alloc_buffer(n * 4).unwrap();
        gb.write_f32(&gate);
        ub.write_f32(&up);
        mlp_silu(&gpu, &gb, &ub, &ob, n as u32).unwrap();
        gpu.synchronize().unwrap();
        let got = ob.read_f32();
        let mut max_err = 0f32;
        for (a, b) in expect.iter().zip(got.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(max_err < 1e-5, "max_err={max_err}");
    }

    #[test]
    fn softcap_sample_argmax() {
        let gpu = gpu_or_fail();
        let n = 64u32;
        let mut logits = vec![0.1f32; n as usize];
        logits[17] = 12.0;
        logits[3] = 8.0;
        let lb = gpu.rt.alloc_buffer((n as usize) * 4).unwrap();
        lb.write_f32(&logits);
        let tok = gpu.rt.alloc_buffer(4).unwrap();
        softcap_sample(&gpu, &lb, &tok, 30.0, n).unwrap();
        gpu.synchronize().unwrap();
        let idx = tok.read_u32()[0];
        assert_eq!(idx, 17);
        let soft = lb.read_f32();
        let expected = 30.0 * (12.0f32 / 30.0).tanh();
        assert!((soft[17] - expected).abs() < 1e-5);
        lb.write_f32(&vec![f32::NAN; n as usize]);
        let err = softcap_sample(&gpu, &lb, &tok, 30.0, n).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("finite"), "{msg}");
    }

    #[test]
    fn softcap_argmax_full_vocab_multipass() {
        let gpu = gpu_or_fail();
        // E4B vocab size — exercises multi-TG + multi-pass finish.
        let n = 262_144u32;
        let mut logits = vec![-1.0f32; n as usize];
        let winner = 200_003usize;
        logits[winner] = 40.0;
        logits[17] = 20.0;
        let lb = gpu.rt.alloc_buffer((n as usize) * 4).unwrap();
        lb.write_f32(&logits);
        let idx = softcap_argmax(&gpu, &lb, 30.0, n).unwrap();
        assert_eq!(idx as usize, winner);
    }

    #[test]
    fn flash_attn_swa_decode_vs_prefill_tail() {
        let gpu = gpu_or_fail();
        // Prefill T=8 then decode-style query Tq=1 against densified KV Tkv=8.
        let b = 1usize;
        let t = 8usize;
        let h = 2usize;
        let hkv = 1usize;
        let d = 256usize;
        let window = 4usize;
        let nq = b * t * h * d;
        let nkv = b * t * hkv * d;
        let q_h: Vec<f32> = (0..nq).map(|i| ((i % 17) as f32) * 0.01 - 0.05).collect();
        let k_h: Vec<f32> = (0..nkv).map(|i| ((i % 13) as f32) * 0.01 - 0.04).collect();
        let v_h: Vec<f32> = (0..nkv).map(|i| ((i % 11) as f32) * 0.02 - 0.1).collect();
        let expect = cpu_attn_causal(&q_h, &k_h, &v_h, b, t, h, hkv, d, Some(window), 1.0);
        // Last query row only via Tq/Tkv API.
        let q_tail = &q_h[(t - 1) * h * d..];
        let qb = gpu.rt.alloc_buffer(h * d * 4).unwrap();
        let kb = gpu.rt.alloc_buffer(nkv * 4).unwrap();
        let vb = gpu.rt.alloc_buffer(nkv * 4).unwrap();
        let ob = gpu.rt.alloc_buffer(h * d * 4).unwrap();
        qb.write_f32(q_tail);
        kb.write_f32(&k_h);
        vb.write_f32(&v_h);
        flash_attn_swa_h256(
            &gpu,
            &qb,
            &kb,
            &vb,
            &ob,
            b as u32,
            1,
            t as u32,
            h as u32,
            hkv as u32,
            window as u32,
            1.0,
            (t - 1) as u32,
            0,
        )
        .unwrap();
        gpu.synchronize().unwrap();
        let got = ob.read_f32();
        let expect_tail = &expect[(t - 1) * h * d..];
        let mut max_err = 0f32;
        for (a, c) in expect_tail.iter().zip(got.iter()) {
            max_err = max_err.max((a - c).abs());
        }
        assert!(max_err < 2e-3, "max_err={max_err}");
    }

    #[test]
    fn rms_qkv_rope_runs() {
        let gpu = gpu_or_fail();
        let t = 2u32;
        let hq = 2u32;
        let hkv = 1u32;
        let d = 8u32;
        let qn = (t * hq * d) as usize;
        let kn = (t * hkv * d) as usize;
        let q = gpu.rt.alloc_buffer(qn * 4).unwrap();
        let k = gpu.rt.alloc_buffer(kn * 4).unwrap();
        let v = gpu.rt.alloc_buffer(kn * 4).unwrap();
        let qw = gpu.rt.alloc_buffer((d as usize) * 4).unwrap();
        let kw = gpu.rt.alloc_buffer((d as usize) * 4).unwrap();
        let vw = gpu.rt.alloc_buffer((d as usize) * 4).unwrap();
        q.write_f32(&vec![0.5f32; qn]);
        k.write_f32(&vec![0.25f32; kn]);
        v.write_f32(&vec![0.125f32; kn]);
        qw.write_f32(&vec![1.0f32; d as usize]);
        kw.write_f32(&vec![1.0f32; d as usize]);
        vw.write_f32(&vec![1.0f32; d as usize]);
        rms_qkv_rope(
            &gpu, &q, &k, &v, &qw, &kw, &vw, t, hq, hkv, d, /*rotary*/ 4, 0, 10000.0, 1e-6,
        )
        .unwrap();
        gpu.synchronize().unwrap();
        let q_out = q.read_f32();
        assert!(q_out.iter().all(|x| x.is_finite()));
        // RMSNorm of constant vector → weight (all 1) → ones before RoPE on first dims.
        assert!(q_out.iter().map(|x| x.abs()).sum::<f32>() > 0.0);
    }

    /// Encode-once prerequisite: `rms_qkv_rope_posbuf` must match const-arena
    /// `rms_qkv_rope` bit-exactly (same math, pos from device u32×1).
    #[test]
    fn rms_qkv_rope_posbuf_matches_const() {
        let gpu = gpu_or_fail();
        if gpu.rt.pipeline("rms_qkv_rope_posbuf").is_err() {
            eprintln!("skip: rms_qkv_rope_posbuf not in metallib");
            return;
        }

        let t = 2u32;
        let hq = 4u32;
        let hkv = 2u32;
        let d = 16u32;
        let rotary = 8u32;
        let pos = 5u32;
        let theta = 10_000.0f32;
        let eps = 1e-6f32;
        let qn = (t * hq * d) as usize;
        let kn = (t * hkv * d) as usize;

        let mk_pat = |n: usize, seed: f32| -> Vec<f32> {
            (0..n)
                .map(|i| ((i as f32) * 0.019 + seed).sin() * 0.4 + 0.05)
                .collect()
        };
        let q0 = mk_pat(qn, 0.2);
        let k0 = mk_pat(kn, 0.4);
        let v0 = mk_pat(kn, 0.8);
        let qw0 = mk_pat(d as usize, 1.2);
        let kw0 = mk_pat(d as usize, 1.4);
        let vw0 = mk_pat(d as usize, 1.8);

        let qw = gpu.rt.alloc_buffer((d as usize) * 4).unwrap();
        let kw = gpu.rt.alloc_buffer((d as usize) * 4).unwrap();
        let vw = gpu.rt.alloc_buffer((d as usize) * 4).unwrap();
        qw.write_f32(&qw0);
        kw.write_f32(&kw0);
        vw.write_f32(&vw0);

        // --- const-arena pos ---
        let q_c = gpu.rt.alloc_buffer(qn * 4).unwrap();
        let k_c = gpu.rt.alloc_buffer(kn * 4).unwrap();
        let v_c = gpu.rt.alloc_buffer(kn * 4).unwrap();
        q_c.write_f32(&q0);
        k_c.write_f32(&k0);
        v_c.write_f32(&v0);
        rms_qkv_rope_ex(
            &gpu, &q_c, &k_c, &v_c, &qw, &kw, &vw, t, hq, hkv, d, rotary, pos, theta, eps,
            /*q_only*/ false,
        )
        .unwrap();
        gpu.synchronize().unwrap();

        // --- GPU pos_buf ---
        let pos_buf = gpu.rt.alloc_buffer(4).unwrap();
        pos_buf.write_u32(&[pos]);
        let q_p = gpu.rt.alloc_buffer(qn * 4).unwrap();
        let k_p = gpu.rt.alloc_buffer(kn * 4).unwrap();
        let v_p = gpu.rt.alloc_buffer(kn * 4).unwrap();
        q_p.write_f32(&q0);
        k_p.write_f32(&k0);
        v_p.write_f32(&v0);
        rms_qkv_rope_ex_posbuf(
            &gpu, &q_p, &k_p, &v_p, &qw, &kw, &vw, t, hq, hkv, d, rotary, &pos_buf, theta, eps,
            /*q_only*/ false,
        )
        .unwrap();
        gpu.synchronize().unwrap();

        let qe = max_abs_err(&q_c.read_f32(), &q_p.read_f32());
        let ke = max_abs_err(&k_c.read_f32(), &k_p.read_f32());
        let ve = max_abs_err(&v_c.read_f32(), &v_p.read_f32());
        assert!(
            qe == 0.0 && ke == 0.0 && ve == 0.0,
            "posbuf vs const max_abs q={qe} k={ke} v={ve}"
        );
        // Non-trivial RoPE (pos≠0) so we did not compare zeros.
        assert!(q_p.read_f32().iter().any(|&x| x != 0.0));
    }

    #[test]
    fn qkv_adapter_rejects_overflow_before_scalar_pool_mutation() {
        let gpu = GemmaGpu::new().expect("Metal 4 runtime for QKV adapter regression");
        let q = gpu.rt.alloc_buffer(4).unwrap();
        let inactive = gpu.rt.alloc_buffer(4).unwrap();
        let weight = gpu.rt.alloc_buffer(4).unwrap();
        let pos = gpu.rt.alloc_buffer(4).unwrap();
        pos.write_u32(&[0]);
        let before = gpu.icb_scalars.cursor_snapshot();

        // The former adapter multiplies these in u32 before casting: debug
        // builds panic and release builds wrap to a tiny dispatch. The Tessl
        // preflight must return an error without reserving stable scalar slots.
        let call = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            rms_qkv_rope_ex_posbuf(
                &gpu,
                &q,
                &inactive,
                &inactive,
                &weight,
                &inactive,
                &inactive,
                u32::MAX,
                u32::MAX,
                1,
                1,
                0,
                &pos,
                10_000.0,
                1e-6,
                true,
            )
        }));
        assert!(call.is_ok(), "safe adapter panicked on caller dimensions");
        assert!(call.unwrap().is_err(), "oversized shape reached dispatch");
        assert_eq!(gpu.icb_scalars.cursor_snapshot(), before);
    }

    #[test]
    fn qkv_adapter_rejects_foreign_active_buffer_before_dispatch() {
        let gpu = GemmaGpu::new().expect("Metal 4 runtime for QKV adapter regression");
        let foreign = GpuRuntime::new().expect("second runtime for ownership regression");
        let foreign_q = foreign.alloc_buffer(16).unwrap();
        let local = gpu.rt.alloc_buffer(16).unwrap();
        let weight = gpu.rt.alloc_buffer(16).unwrap();
        let pos = gpu.rt.alloc_buffer(4).unwrap();
        pos.write_u32(&[0]);
        let before_cursor = gpu.icb_scalars.cursor_snapshot();
        let before_dispatches = gpu.rt.take_dispatch_count();

        let err = rms_qkv_rope_ex_posbuf(
            &gpu, &foreign_q, &local, &local, &weight, &local, &local, 1, 1, 1, 4, 0, &pos,
            10_000.0, 1e-6, true,
        )
        .unwrap_err();
        assert!(err.to_string().contains("another runtime"), "{err}");
        assert_eq!(gpu.icb_scalars.cursor_snapshot(), before_cursor);
        assert_eq!(gpu.rt.take_dispatch_count(), 0);
        assert_eq!(before_dispatches, 0);
    }

    #[test]
    fn q_only_adapter_ignores_foreign_inactive_kv_placeholders() {
        let gpu = GemmaGpu::new().expect("Metal 4 runtime for QKV adapter regression");
        let foreign = GpuRuntime::new().expect("second runtime for inactive-operand regression");
        let q = gpu.rt.alloc_buffer(16).unwrap();
        let q_weight = gpu.rt.alloc_buffer(16).unwrap();
        let pos = gpu.rt.alloc_buffer(4).unwrap();
        let inactive = foreign.alloc_buffer(1).unwrap();
        q.write_f32(&[0.5, -0.25, 1.0, 0.75]);
        q_weight.write_f32(&[1.0; 4]);
        pos.write_u32(&[3]);

        rms_qkv_rope_ex_posbuf(
            &gpu, &q, &inactive, &inactive, &q_weight, &inactive, &inactive, 1, 1, 1, 4, 4, &pos,
            10_000.0, 1e-6, true,
        )
        .unwrap();
        gpu.synchronize().unwrap();
        assert!(q.read_f32().iter().all(|value| value.is_finite()));
    }

    /// Layer-fusion: fused `rms_qkv_rope_kv_store` vs unfused rope + `kv_store_timestep_pair`.
    /// Element-local copy after K/V norm(+RoPE) — expect bit-exact scratch and cache slots.
    #[test]
    fn rms_qkv_rope_kv_store_matches_unfused() {
        let gpu = gpu_or_fail();
        if gpu.rt.pipeline("rms_qkv_rope_kv_store").is_err() {
            eprintln!("skip: rms_qkv_rope_kv_store not in metallib");
            return;
        }

        // Decode-shaped GQA (T=1 matches producer hot path in gpu_model).
        let t = 1u32;
        let hq = 4u32;
        let hkv = 2u32;
        let d = 16u32;
        let rotary = 8u32;
        let pos = 3u32;
        let theta = 10_000.0f32;
        let eps = 1e-6f32;
        let kv_dst_offset = 7u32; // non-zero slot into a larger cache buffer
        let qn = (t * hq * d) as usize;
        let kn = (t * hkv * d) as usize;
        let cache_elems = kv_dst_offset as usize + kn;

        let mk_pat = |n: usize, seed: f32| -> Vec<f32> {
            (0..n)
                .map(|i| ((i as f32) * 0.017 + seed).sin() * 0.5 + 0.1)
                .collect()
        };
        let q0 = mk_pat(qn, 0.1);
        let k0 = mk_pat(kn, 0.3);
        let v0 = mk_pat(kn, 0.7);
        let qw0 = mk_pat(d as usize, 1.1);
        let kw0 = mk_pat(d as usize, 1.3);
        let vw0 = mk_pat(d as usize, 1.7);

        let pos_buf = gpu.rt.alloc_buffer(4).unwrap();
        pos_buf.write_u32(&[pos]);

        // --- unfused ---
        let q_u = gpu.rt.alloc_buffer(qn * 4).unwrap();
        let k_u = gpu.rt.alloc_buffer(kn * 4).unwrap();
        let v_u = gpu.rt.alloc_buffer(kn * 4).unwrap();
        let qw = gpu.rt.alloc_buffer((d as usize) * 4).unwrap();
        let kw = gpu.rt.alloc_buffer((d as usize) * 4).unwrap();
        let vw = gpu.rt.alloc_buffer((d as usize) * 4).unwrap();
        let dst_k_u = gpu.rt.alloc_buffer(cache_elems * 4).unwrap();
        let dst_v_u = gpu.rt.alloc_buffer(cache_elems * 4).unwrap();
        q_u.write_f32(&q0);
        k_u.write_f32(&k0);
        v_u.write_f32(&v0);
        qw.write_f32(&qw0);
        kw.write_f32(&kw0);
        vw.write_f32(&vw0);
        dst_k_u.write_f32(&vec![0f32; cache_elems]);
        dst_v_u.write_f32(&vec![0f32; cache_elems]);

        rms_qkv_rope_ex_posbuf(
            &gpu, &q_u, &k_u, &v_u, &qw, &kw, &vw, t, hq, hkv, d, rotary, &pos_buf, theta, eps,
            /*q_only*/ false,
        )
        .unwrap();
        // RAW: rope writes scratch; store reads it (hazard skip-auto can reorder).
        gpu.barrier().unwrap();
        kv_store_timestep_pair_off(
            &gpu,
            &k_u,
            &v_u,
            0,
            &dst_k_u,
            &dst_v_u,
            hkv * d,
            kv_dst_offset,
            cache_elems as u32,
        )
        .unwrap();
        gpu.synchronize().unwrap();

        // --- fused ---
        let q_f = gpu.rt.alloc_buffer(qn * 4).unwrap();
        let k_f = gpu.rt.alloc_buffer(kn * 4).unwrap();
        let v_f = gpu.rt.alloc_buffer(kn * 4).unwrap();
        let dst_k_f = gpu.rt.alloc_buffer(cache_elems * 4).unwrap();
        let dst_v_f = gpu.rt.alloc_buffer(cache_elems * 4).unwrap();
        q_f.write_f32(&q0);
        k_f.write_f32(&k0);
        v_f.write_f32(&v0);
        dst_k_f.write_f32(&vec![0f32; cache_elems]);
        dst_v_f.write_f32(&vec![0f32; cache_elems]);

        rms_qkv_rope_kv_store(
            &gpu,
            &q_f,
            &k_f,
            &v_f,
            &qw,
            &kw,
            &vw,
            t,
            hq,
            hkv,
            d,
            rotary,
            &pos_buf,
            theta,
            eps,
            &dst_k_f,
            &dst_v_f,
            kv_dst_offset,
            cache_elems as u32,
        )
        .unwrap();
        gpu.synchronize().unwrap();

        let q_ue = max_abs_err(&q_u.read_f32(), &q_f.read_f32());
        let k_ue = max_abs_err(&k_u.read_f32(), &k_f.read_f32());
        let v_ue = max_abs_err(&v_u.read_f32(), &v_f.read_f32());
        let dk_ue = max_abs_err(&dst_k_u.read_f32(), &dst_k_f.read_f32());
        let dv_ue = max_abs_err(&dst_v_u.read_f32(), &dst_v_f.read_f32());
        assert!(
            q_ue == 0.0 && k_ue == 0.0 && v_ue == 0.0 && dk_ue == 0.0 && dv_ue == 0.0,
            "fused vs unfused max_abs q={q_ue} k={k_ue} v={v_ue} dst_k={dk_ue} dst_v={dv_ue}"
        );
        // Cache prefix before the slot must stay zero (no overrun).
        let dk = dst_k_f.read_f32();
        let dv = dst_v_f.read_f32();
        assert!(dk[..kv_dst_offset as usize].iter().all(|&x| x == 0.0));
        assert!(dv[..kv_dst_offset as usize].iter().all(|&x| x == 0.0));
        // Written region must be non-trivial.
        let written = &dk[kv_dst_offset as usize..];
        assert!(written.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn ple_lookup_scaled() {
        let gpu = gpu_or_fail();
        let vocab = 8u32;
        let dim = 4u32;
        let n = 2u32;
        let mut table_bits = vec![0u16; (vocab * dim) as usize];
        // token 3 → bf16 ~ 2.0
        for d in 0..dim {
            table_bits[(3 * dim + d) as usize] = crate::quant::f32_to_bf16_bits(2.0);
        }
        let table = gpu.rt.alloc_buffer(table_bits.len() * 2).unwrap();
        table.write_bf16_bits(&table_bits);
        let ids_host = [3u32, 3u32];
        let ids = gpu.rt.alloc_buffer(ids_host.len() * 4).unwrap();
        ids.write_u32(&ids_host);
        let out = gpu.rt.alloc_buffer((n * dim) as usize * 4).unwrap();
        let scale = (dim as f32).sqrt();
        ple_lookup(&gpu, &ids, &table, &out, dim, vocab, n, scale).unwrap();
        gpu.synchronize().unwrap();
        let got = out.read_f32();
        for v in &got {
            assert!(
                (*v - 2.0 * scale).abs() < 0.02,
                "got={v} expect={}",
                2.0 * scale
            );
        }
    }

    fn cpu_attn_causal(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        b: usize,
        t: usize,
        h: usize,
        hkv: usize,
        d: usize,
        window: Option<usize>,
        scale: f32,
    ) -> Vec<f32> {
        let mut o = vec![0f32; b * t * h * d];
        let group = h / hkv;
        for bi in 0..b {
            for hi in 0..h {
                let hki = hi / group;
                for qi in 0..t {
                    let mut scores = vec![f32::NEG_INFINITY; t];
                    let q_off = ((bi * t + qi) * h + hi) * d;
                    for kj in 0..=qi {
                        if let Some(w) = window {
                            if qi + 1 > w && kj < qi + 1 - w {
                                continue;
                            }
                        }
                        let k_off = ((bi * t + kj) * hkv + hki) * d;
                        let mut s = 0f32;
                        for dd in 0..d {
                            s += q[q_off + dd] * k[k_off + dd];
                        }
                        scores[kj] = s * scale;
                    }
                    let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut denom = 0f32;
                    let mut weights = vec![0f32; t];
                    for kj in 0..t {
                        if scores[kj] > f32::NEG_INFINITY {
                            weights[kj] = (scores[kj] - m).exp();
                            denom += weights[kj];
                        }
                    }
                    let o_off = ((bi * t + qi) * h + hi) * d;
                    for kj in 0..t {
                        if weights[kj] == 0.0 {
                            continue;
                        }
                        let p = weights[kj] / denom;
                        let v_off = ((bi * t + kj) * hkv + hki) * d;
                        for dd in 0..d {
                            o[o_off + dd] += p * v[v_off + dd];
                        }
                    }
                }
            }
        }
        o
    }

    #[test]
    fn flash_attn_swa_h256_matches_cpu() {
        let gpu = gpu_or_fail();
        let b = 1usize;
        let t = 8usize;
        let h = 2usize;
        let hkv = 1usize;
        let d = 256usize;
        let window = 4usize;
        let nq = b * t * h * d;
        let nkv = b * t * hkv * d;
        let q_h: Vec<f32> = (0..nq).map(|i| ((i % 17) as f32) * 0.01 - 0.05).collect();
        let k_h: Vec<f32> = (0..nkv).map(|i| ((i % 13) as f32) * 0.01 - 0.04).collect();
        let v_h: Vec<f32> = (0..nkv).map(|i| ((i % 11) as f32) * 0.02 - 0.1).collect();
        let expect = cpu_attn_causal(&q_h, &k_h, &v_h, b, t, h, hkv, d, Some(window), 1.0);
        let qb = gpu.rt.alloc_buffer(nq * 4).unwrap();
        let kb = gpu.rt.alloc_buffer(nkv * 4).unwrap();
        let vb = gpu.rt.alloc_buffer(nkv * 4).unwrap();
        let ob = gpu.rt.alloc_buffer(nq * 4).unwrap();
        qb.write_f32(&q_h);
        kb.write_f32(&k_h);
        vb.write_f32(&v_h);
        flash_attn_swa_h256_prefill(
            &gpu,
            &qb,
            &kb,
            &vb,
            &ob,
            b as u32,
            t as u32,
            h as u32,
            hkv as u32,
            window as u32,
            1.0,
        )
        .unwrap();
        gpu.synchronize().unwrap();
        let got = ob.read_f32();
        let mut max_err = 0f32;
        for (a, b) in expect.iter().zip(got.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(max_err < 2e-3, "max_err={max_err}");
    }

    #[test]
    fn flash_attn_swa_h256_bf16_zero_live_kv_overwrites_output() {
        let gpu = gpu_or_fail();
        const B: usize = 2;
        const TQ: usize = 9;
        const H: usize = 4;
        const HKV: usize = 2;
        const D: usize = 256;
        const KV_CAPACITY: usize = 3;

        let output_elems = B * TQ * H * D;
        let kv_elems = B * KV_CAPACITY * HKV * D;
        let q = gpu.rt.alloc_buffer(output_elems * 4).unwrap();
        let k = gpu.rt.alloc_buffer(kv_elems * 4).unwrap();
        let v = gpu.rt.alloc_buffer(kv_elems * 4).unwrap();
        q.zero();
        k.zero();
        v.zero();

        // This is the production half-width output mode used by Gemma. Seed
        // each packed word with two bf16 1.0 values so an early-return no-op is
        // distinguishable from the required zero output.
        let output = gpu.rt.alloc_buffer(output_elems * 2).unwrap();
        output.write_u32(&vec![0x3f80_3f80; output_elems / 2]);
        flash_attn_swa_h256_ex(
            &gpu, &q, &k, &v, &output, B as u32, TQ as u32, 0, H as u32, HKV as u32, 4, 0.125, 17,
            41, true,
        )
        .unwrap();
        gpu.synchronize().unwrap();

        for (i, &bits) in output.read_u32()[..output_elems / 2].iter().enumerate() {
            assert_eq!(bits, 0, "zero-live-KV bf16 word {i} stayed {bits:#x}");
        }
    }

    #[test]
    fn flash_attn_swa_h128_matches_cpu() {
        let gpu = gpu_or_fail();
        let b = 1usize;
        let t = 8usize;
        let h = 2usize;
        let hkv = 1usize;
        let d = 128usize;
        let window = 4usize;
        let nq = b * t * h * d;
        let nkv = b * t * hkv * d;
        let q_h: Vec<f32> = (0..nq).map(|i| ((i % 17) as f32) * 0.01 - 0.05).collect();
        let k_h: Vec<f32> = (0..nkv).map(|i| ((i % 13) as f32) * 0.01 - 0.04).collect();
        let v_h: Vec<f32> = (0..nkv).map(|i| ((i % 11) as f32) * 0.02 - 0.1).collect();
        let expect = cpu_attn_causal(&q_h, &k_h, &v_h, b, t, h, hkv, d, Some(window), 1.0);
        let qb = gpu.rt.alloc_buffer(nq * 4).unwrap();
        let kb = gpu.rt.alloc_buffer(nkv * 4).unwrap();
        let vb = gpu.rt.alloc_buffer(nkv * 4).unwrap();
        let ob = gpu.rt.alloc_buffer(nq * 4).unwrap();
        qb.write_f32(&q_h);
        kb.write_f32(&k_h);
        vb.write_f32(&v_h);
        flash_attn_swa_h128_prefill(
            &gpu,
            &qb,
            &kb,
            &vb,
            &ob,
            b as u32,
            t as u32,
            h as u32,
            hkv as u32,
            window as u32,
            1.0,
        )
        .unwrap();
        gpu.synchronize().unwrap();
        let got = ob.read_f32();
        let mut max_err = 0f32;
        for (a, b) in expect.iter().zip(got.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(max_err < 2e-3, "max_err={max_err}");
    }

    #[test]
    fn flash_attn_global_h512_matches_cpu() {
        let gpu = gpu_or_fail();
        let b = 1usize;
        let t = 4usize;
        let h = 2usize;
        let hkv = 1usize;
        let d = 512usize;
        let nq = b * t * h * d;
        let nkv = b * t * hkv * d;
        let q_h: Vec<f32> = (0..nq).map(|i| ((i % 19) as f32) * 0.005 - 0.02).collect();
        let k_h: Vec<f32> = (0..nkv).map(|i| ((i % 23) as f32) * 0.005 - 0.03).collect();
        let v_h: Vec<f32> = (0..nkv).map(|i| ((i % 7) as f32) * 0.01 - 0.05).collect();
        let expect = cpu_attn_causal(&q_h, &k_h, &v_h, b, t, h, hkv, d, None, 1.0);
        let qb = gpu.rt.alloc_buffer(nq * 4).unwrap();
        let kb = gpu.rt.alloc_buffer(nkv * 4).unwrap();
        let vb = gpu.rt.alloc_buffer(nkv * 4).unwrap();
        let ob = gpu.rt.alloc_buffer(nq * 4).unwrap();
        qb.write_f32(&q_h);
        kb.write_f32(&k_h);
        vb.write_f32(&v_h);
        flash_attn_global_h512_prefill(
            &gpu, &qb, &kb, &vb, &ob, b as u32, t as u32, h as u32, hkv as u32, 1.0,
        )
        .unwrap();
        gpu.synchronize().unwrap();
        let got = ob.read_f32();
        let mut max_err = 0f32;
        for (a, c) in expect.iter().zip(got.iter()) {
            max_err = max_err.max((a - c).abs());
        }
        assert!(max_err < 2e-3, "max_err={max_err}");
    }

    #[test]
    fn gemm_prefill_small() {
        let gpu = gpu_or_fail();
        let m = 16usize;
        let k = 32usize;
        let n = 16usize;
        let a_h: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b_h: Vec<f32> = (0..k * n).map(|i| ((i % 5) as f32) * 0.02).collect();
        let a = gpu.rt.alloc_tensor_f32(&[m, k]).unwrap();
        let b = gpu.rt.alloc_tensor_f32(&[k, n]).unwrap();
        let c = gpu.rt.alloc_tensor_f32(&[m, n]).unwrap();
        a.buffer.write_f32(&a_h);
        b.buffer.write_f32(&b_h);
        gemm_prefill(&a, &b, &c).unwrap();
        gpu.synchronize().unwrap();
        let got = c.buffer.read_f32();
        let mut expect = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0f32;
                for kk in 0..k {
                    acc += a_h[i * k + kk] * b_h[kk * n + j];
                }
                expect[i * n + j] = acc;
            }
        }
        let mut max_err = 0f32;
        for (x, y) in expect.iter().zip(got.iter()) {
            max_err = max_err.max((x - y).abs());
        }
        assert!(max_err < 1e-3, "max_err={max_err}");
    }

    /// Persistent interpreter: flag defaults OFF; dispatch rejects without opt-in.
    #[test]
    fn persistent_interp_flag_default_off() {
        set_persistent_interp(false);
        assert!(!persistent_interp_enabled());
        let gpu = gpu_or_fail();
        if gpu
            .rt
            .pipeline(KernelId::PersistentInterpGateDown.entry_name())
            .is_err()
        {
            eprintln!("skip: persistent_interp_gate_down not in metallib");
            return;
        }
        let prog = persistent_interp_gate_down_program();
        let insns = gpu.rt.alloc_buffer(prog.len() * 4).unwrap();
        // write_u32 if available — else raw
        insns.write_u32(&prog);
        let gate = gpu.rt.alloc_buffer(4).unwrap();
        let up = gpu.rt.alloc_buffer(4).unwrap();
        let mid = gpu.rt.alloc_buffer(4).unwrap();
        let w = gpu.rt.alloc_buffer(4).unwrap();
        let out = gpu.rt.alloc_buffer(4).unwrap();
        let deps = gpu.rt.alloc_buffer(8).unwrap();
        let fail = gpu.rt.alloc_buffer(4).unwrap();
        let err = persistent_interp_gate_down(
            &gpu,
            &insns,
            prog.len() as u32,
            &gate,
            &up,
            &mid,
            &w,
            &out,
            &deps,
            &fail,
            1,
            1,
            1,
        );
        assert!(err.is_err(), "must reject when flag off");
    }

    fn cpu_gate_down(
        gate: &[f32],
        up: &[f32],
        w_down: &[f32],
        n_mid: usize,
        n_out: usize,
    ) -> Vec<f32> {
        let mid: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(&g, &u)| gelu_pytorch_tanh(g) * u)
            .collect();
        let mut out = vec![0f32; n_out];
        for r in 0..n_out {
            let mut acc = 0f32;
            for i in 0..n_mid {
                acc += mid[i] * w_down[r * n_mid + i];
            }
            out[r] = acc;
        }
        out
    }

    /// Persistent interpreter gate→down stand-in vs unfused gelu + dense down.
    /// Tiny TG count (mini envelope) — proves instruction stream + atomic barrier.
    #[test]
    fn persistent_interp_gate_down_matches_unfused() {
        let gpu = gpu_or_fail();
        if gpu
            .rt
            .pipeline(KernelId::PersistentInterpGateDown.entry_name())
            .is_err()
        {
            eprintln!("skip: persistent_interp_gate_down not in metallib");
            return;
        }
        set_persistent_interp(true);
        assert!(persistent_interp_enabled());

        // Mini-scale but smaller than full E4B-mini MLP (512→256) for fast unit.
        let n_mid = 64usize;
        let n_out = 16usize;
        let n_tg = 4u32;
        let gate: Vec<f32> = (0..n_mid).map(|i| (i as f32) * 0.03 - 1.0).collect();
        let up: Vec<f32> = (0..n_mid).map(|i| ((i % 7) as f32) * 0.2 - 0.5).collect();
        let w_down: Vec<f32> = (0..n_out * n_mid)
            .map(|i| ((i % 11) as f32) * 0.01 - 0.04)
            .collect();
        let expect = cpu_gate_down(&gate, &up, &w_down, n_mid, n_out);

        let prog = persistent_interp_gate_down_program();
        let insns = gpu.rt.alloc_buffer(prog.len() * 4).unwrap();
        insns.write_u32(&prog);
        let gb = gpu.rt.alloc_buffer(n_mid * 4).unwrap();
        let ub = gpu.rt.alloc_buffer(n_mid * 4).unwrap();
        let mid = gpu.rt.alloc_buffer(n_mid * 4).unwrap();
        let wb = gpu.rt.alloc_buffer(n_out * n_mid * 4).unwrap();
        let ob = gpu.rt.alloc_buffer(n_out * 4).unwrap();
        let deps = gpu.rt.alloc_buffer(8).unwrap();
        let fail = gpu.rt.alloc_buffer(4).unwrap();
        gb.write_f32(&gate);
        ub.write_f32(&up);
        wb.write_f32(&w_down);
        deps.write_u32(&[0, 0]);
        fail.write_u32(&[0]);

        persistent_interp_gate_down(
            &gpu,
            &insns,
            prog.len() as u32,
            &gb,
            &ub,
            &mid,
            &wb,
            &ob,
            &deps,
            &fail,
            n_mid as u32,
            n_out as u32,
            n_tg,
        )
        .unwrap();
        gpu.synchronize().unwrap();

        let fail_v = fail.read_u32();
        assert_eq!(
            fail_v[0], 0,
            "grid barrier spin timed out (Metal forward-progress caveat)"
        );
        let got = ob.read_f32();
        let max_err = max_abs_err(&expect, &got);
        assert!(
            max_err < 2e-4,
            "persistent_interp gate→down max_err={max_err}"
        );

        // Also check mid matches unfused gelu (RAW edge across barrier).
        let mid_expect: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(&g, &u)| gelu_pytorch_tanh(g) * u)
            .collect();
        let mid_got = mid.read_f32();
        let mid_err = max_abs_err(&mid_expect, &mid_got);
        assert!(mid_err < 2e-4, "mid after PRODUCE max_err={mid_err}");

        set_persistent_interp(false);
    }

    /// Same stand-in at E4B-mini MLP dims (512→256), still ≤ MAX_TG.
    #[test]
    fn persistent_interp_gate_down_mini_dims() {
        let gpu = gpu_or_fail();
        if gpu
            .rt
            .pipeline(KernelId::PersistentInterpGateDown.entry_name())
            .is_err()
        {
            eprintln!("skip: persistent_interp_gate_down not in metallib");
            return;
        }
        set_persistent_interp(true);

        // Matches SyntheticE4bGraph::mini_parity intermediate→hidden.
        let n_mid = 512usize;
        let n_out = 256usize;
        let n_tg = PERSISTENT_INTERP_MAX_TG;
        let gate: Vec<f32> = (0..n_mid).map(|i| ((i % 23) as f32) * 0.02 - 0.2).collect();
        let up: Vec<f32> = (0..n_mid)
            .map(|i| ((i % 19) as f32) * 0.015 - 0.1)
            .collect();
        let w_down: Vec<f32> = (0..n_out * n_mid)
            .map(|i| ((i % 29) as f32) * 0.002 - 0.02)
            .collect();
        let expect = cpu_gate_down(&gate, &up, &w_down, n_mid, n_out);

        let prog = persistent_interp_gate_down_program();
        let insns = gpu.rt.alloc_buffer(prog.len() * 4).unwrap();
        insns.write_u32(&prog);
        let gb = gpu.rt.alloc_buffer(n_mid * 4).unwrap();
        let ub = gpu.rt.alloc_buffer(n_mid * 4).unwrap();
        let mid = gpu.rt.alloc_buffer(n_mid * 4).unwrap();
        let wb = gpu.rt.alloc_buffer(n_out * n_mid * 4).unwrap();
        let ob = gpu.rt.alloc_buffer(n_out * 4).unwrap();
        let deps = gpu.rt.alloc_buffer(8).unwrap();
        let fail = gpu.rt.alloc_buffer(4).unwrap();
        gb.write_f32(&gate);
        ub.write_f32(&up);
        wb.write_f32(&w_down);
        deps.write_u32(&[0, 0]);
        fail.write_u32(&[0]);

        persistent_interp_gate_down(
            &gpu,
            &insns,
            prog.len() as u32,
            &gb,
            &ub,
            &mid,
            &wb,
            &ob,
            &deps,
            &fail,
            n_mid as u32,
            n_out as u32,
            n_tg,
        )
        .unwrap();
        gpu.synchronize().unwrap();

        assert_eq!(fail.read_u32()[0], 0, "barrier spin timeout at mini dims");
        let max_err = max_abs_err(&expect, &ob.read_f32());
        assert!(
            max_err < 5e-4,
            "mini-dims persistent_interp max_err={max_err}"
        );

        set_persistent_interp(false);
    }

    fn cpu_fa_o_proj(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        w_o: &[f32],
        n_ctx: usize,
        n_out: usize,
        scale: f32,
    ) -> Vec<f32> {
        let ctx: Vec<f32> = (0..n_ctx)
            .map(|i| {
                let s = (q[i] * k[i] * scale).clamp(-10.0, 10.0).tanh();
                s * v[i]
            })
            .collect();
        let mut out = vec![0f32; n_out];
        for r in 0..n_out {
            let mut acc = 0f32;
            for i in 0..n_ctx {
                acc += ctx[i] * w_o[r * n_ctx + i];
            }
            out[r] = acc;
        }
        out
    }

    /// Persistent interpreter FA→o_proj stand-in vs unfused mock-FA + dense o_proj.
    /// Sibling to gate→down — same instruction stream + atomic barrier doctrine.
    #[test]
    fn persistent_interp_fa_o_proj_matches_unfused() {
        let gpu = gpu_or_fail();
        if gpu
            .rt
            .pipeline(KernelId::PersistentInterpFaOProj.entry_name())
            .is_err()
        {
            eprintln!("skip: persistent_interp_fa_o_proj not in metallib");
            return;
        }
        set_persistent_interp(true);
        assert!(persistent_interp_enabled());

        let n_ctx = 64usize;
        let n_out = 16usize;
        let n_tg = 4u32;
        let scale = 0.125f32;
        let q: Vec<f32> = (0..n_ctx).map(|i| (i as f32) * 0.04 - 1.2).collect();
        let k: Vec<f32> = (0..n_ctx).map(|i| ((i % 9) as f32) * 0.15 - 0.6).collect();
        let v: Vec<f32> = (0..n_ctx).map(|i| ((i % 5) as f32) * 0.25 - 0.4).collect();
        let w_o: Vec<f32> = (0..n_out * n_ctx)
            .map(|i| ((i % 13) as f32) * 0.008 - 0.05)
            .collect();
        let expect = cpu_fa_o_proj(&q, &k, &v, &w_o, n_ctx, n_out, scale);

        let prog = persistent_interp_fa_o_proj_program();
        let insns = gpu.rt.alloc_buffer(prog.len() * 4).unwrap();
        insns.write_u32(&prog);
        let qb = gpu.rt.alloc_buffer(n_ctx * 4).unwrap();
        let kb = gpu.rt.alloc_buffer(n_ctx * 4).unwrap();
        let vb = gpu.rt.alloc_buffer(n_ctx * 4).unwrap();
        let ctx = gpu.rt.alloc_buffer(n_ctx * 4).unwrap();
        let wb = gpu.rt.alloc_buffer(n_out * n_ctx * 4).unwrap();
        let ob = gpu.rt.alloc_buffer(n_out * 4).unwrap();
        let deps = gpu.rt.alloc_buffer(8).unwrap();
        let fail = gpu.rt.alloc_buffer(4).unwrap();
        qb.write_f32(&q);
        kb.write_f32(&k);
        vb.write_f32(&v);
        wb.write_f32(&w_o);
        deps.write_u32(&[0, 0]);
        fail.write_u32(&[0]);

        persistent_interp_fa_o_proj(
            &gpu,
            &insns,
            prog.len() as u32,
            &qb,
            &kb,
            &vb,
            &ctx,
            &wb,
            &ob,
            &deps,
            &fail,
            n_ctx as u32,
            n_out as u32,
            n_tg,
            scale,
        )
        .unwrap();
        gpu.synchronize().unwrap();

        assert_eq!(
            fail.read_u32()[0],
            0,
            "grid barrier spin timed out (Metal forward-progress caveat)"
        );
        let max_err = max_abs_err(&expect, &ob.read_f32());
        assert!(
            max_err < 2e-4,
            "persistent_interp FA→o_proj max_err={max_err}"
        );

        // ctx across barrier matches unfused mock FA.
        let ctx_expect: Vec<f32> = (0..n_ctx)
            .map(|i| {
                let s = (q[i] * k[i] * scale).clamp(-10.0, 10.0).tanh();
                s * v[i]
            })
            .collect();
        let ctx_err = max_abs_err(&ctx_expect, &ctx.read_f32());
        assert!(ctx_err < 2e-4, "ctx after PRODUCE max_err={ctx_err}");

        set_persistent_interp(false);
    }

    /// Same FA→o_proj stand-in at mini attn dims (hq·d=256 → hidden=256), ≤ MAX_TG.
    #[test]
    fn persistent_interp_fa_o_proj_mini_dims() {
        let gpu = gpu_or_fail();
        if gpu
            .rt
            .pipeline(KernelId::PersistentInterpFaOProj.entry_name())
            .is_err()
        {
            eprintln!("skip: persistent_interp_fa_o_proj not in metallib");
            return;
        }
        set_persistent_interp(true);

        // Matches SyntheticE4bGraph::mini_parity: hq*head_dim → hidden (1*256→256).
        let n_ctx = 256usize;
        let n_out = 256usize;
        let n_tg = PERSISTENT_INTERP_MAX_TG;
        let scale = (n_ctx as f32).sqrt().recip();
        let q: Vec<f32> = (0..n_ctx)
            .map(|i| ((i % 17) as f32) * 0.03 - 0.25)
            .collect();
        let k: Vec<f32> = (0..n_ctx)
            .map(|i| ((i % 13) as f32) * 0.025 - 0.15)
            .collect();
        let v: Vec<f32> = (0..n_ctx).map(|i| ((i % 11) as f32) * 0.02 - 0.1).collect();
        let w_o: Vec<f32> = (0..n_out * n_ctx)
            .map(|i| ((i % 31) as f32) * 0.001 - 0.015)
            .collect();
        let expect = cpu_fa_o_proj(&q, &k, &v, &w_o, n_ctx, n_out, scale);

        let prog = persistent_interp_fa_o_proj_program();
        let insns = gpu.rt.alloc_buffer(prog.len() * 4).unwrap();
        insns.write_u32(&prog);
        let qb = gpu.rt.alloc_buffer(n_ctx * 4).unwrap();
        let kb = gpu.rt.alloc_buffer(n_ctx * 4).unwrap();
        let vb = gpu.rt.alloc_buffer(n_ctx * 4).unwrap();
        let ctx = gpu.rt.alloc_buffer(n_ctx * 4).unwrap();
        let wb = gpu.rt.alloc_buffer(n_out * n_ctx * 4).unwrap();
        let ob = gpu.rt.alloc_buffer(n_out * 4).unwrap();
        let deps = gpu.rt.alloc_buffer(8).unwrap();
        let fail = gpu.rt.alloc_buffer(4).unwrap();
        qb.write_f32(&q);
        kb.write_f32(&k);
        vb.write_f32(&v);
        wb.write_f32(&w_o);
        deps.write_u32(&[0, 0]);
        fail.write_u32(&[0]);

        persistent_interp_fa_o_proj(
            &gpu,
            &insns,
            prog.len() as u32,
            &qb,
            &kb,
            &vb,
            &ctx,
            &wb,
            &ob,
            &deps,
            &fail,
            n_ctx as u32,
            n_out as u32,
            n_tg,
            scale,
        )
        .unwrap();
        gpu.synchronize().unwrap();

        assert_eq!(fail.read_u32()[0], 0, "barrier spin timeout at mini dims");
        let max_err = max_abs_err(&expect, &ob.read_f32());
        assert!(
            max_err < 5e-4,
            "mini-dims persistent_interp FA→o_proj max_err={max_err}"
        );

        set_persistent_interp(false);
    }

    /// Flag-off rejects FA→o_proj dispatch (same opt-in as gate→down).
    #[test]
    fn persistent_interp_fa_o_proj_flag_off_rejects() {
        set_persistent_interp(false);
        assert!(!persistent_interp_enabled());
        let gpu = gpu_or_fail();
        if gpu
            .rt
            .pipeline(KernelId::PersistentInterpFaOProj.entry_name())
            .is_err()
        {
            eprintln!("skip: persistent_interp_fa_o_proj not in metallib");
            return;
        }
        let prog = persistent_interp_fa_o_proj_program();
        let insns = gpu.rt.alloc_buffer(prog.len() * 4).unwrap();
        insns.write_u32(&prog);
        let q = gpu.rt.alloc_buffer(4).unwrap();
        let k = gpu.rt.alloc_buffer(4).unwrap();
        let v = gpu.rt.alloc_buffer(4).unwrap();
        let ctx = gpu.rt.alloc_buffer(4).unwrap();
        let w = gpu.rt.alloc_buffer(4).unwrap();
        let out = gpu.rt.alloc_buffer(4).unwrap();
        let deps = gpu.rt.alloc_buffer(8).unwrap();
        let fail = gpu.rt.alloc_buffer(4).unwrap();
        let err = persistent_interp_fa_o_proj(
            &gpu,
            &insns,
            prog.len() as u32,
            &q,
            &k,
            &v,
            &ctx,
            &w,
            &out,
            &deps,
            &fail,
            1,
            1,
            1,
            1.0,
        );
        assert!(err.is_err(), "must reject when flag off");
    }

    /// `GEMMA_METAL_FUSE_GATE_DOWN` defaults OFF (separate from PERSISTENT_INTERP).
    #[test]
    fn fuse_gate_down_flag_default_off() {
        set_fuse_gate_down(false);
        assert!(!fuse_gate_down_enabled());
    }

    /// Dense stand-in stress at real E4B MLP dims (10240→2560), n_tg=MAX.
    /// Serialize Metal: `cargo test --lib persistent_interp_gate_down_e4b_dims_stress -- --test-threads=1`
    #[test]
    fn persistent_interp_gate_down_e4b_dims_stress() {
        let gpu = gpu_or_fail();
        if gpu
            .rt
            .pipeline(KernelId::PersistentInterpGateDown.entry_name())
            .is_err()
        {
            eprintln!("skip: persistent_interp_gate_down not in metallib");
            return;
        }
        set_persistent_interp(true);

        let n_mid = 10240usize;
        let n_out = 2560usize;
        let n_tg = PERSISTENT_INTERP_MAX_TG;
        // Sparse-ish deterministic fill — keeps CPU reference tractable.
        let gate: Vec<f32> = (0..n_mid)
            .map(|i| (((i * 7) % 997) as f32) * 0.001 - 0.4)
            .collect();
        let up: Vec<f32> = (0..n_mid)
            .map(|i| (((i * 11) % 503) as f32) * 0.002 - 0.3)
            .collect();
        let w_down: Vec<f32> = (0..n_out * n_mid)
            .map(|i| (((i * 13) % 251) as f32) * 0.0001 - 0.01)
            .collect();
        let expect = cpu_gate_down(&gate, &up, &w_down, n_mid, n_out);

        let prog = persistent_interp_gate_down_program();
        let insns = gpu.rt.alloc_buffer(prog.len() * 4).unwrap();
        insns.write_u32(&prog);
        let gb = gpu.rt.alloc_buffer(n_mid * 4).unwrap();
        let ub = gpu.rt.alloc_buffer(n_mid * 4).unwrap();
        let mid = gpu.rt.alloc_buffer(n_mid * 4).unwrap();
        let wb = gpu.rt.alloc_buffer(n_out * n_mid * 4).unwrap();
        let ob = gpu.rt.alloc_buffer(n_out * 4).unwrap();
        let deps = gpu.rt.alloc_buffer(8).unwrap();
        let fail = gpu.rt.alloc_buffer(4).unwrap();
        gb.write_f32(&gate);
        ub.write_f32(&up);
        wb.write_f32(&w_down);
        deps.write_u32(&[0, 0]);
        fail.write_u32(&[0]);

        persistent_interp_gate_down(
            &gpu,
            &insns,
            prog.len() as u32,
            &gb,
            &ub,
            &mid,
            &wb,
            &ob,
            &deps,
            &fail,
            n_mid as u32,
            n_out as u32,
            n_tg,
        )
        .unwrap();
        gpu.synchronize().unwrap();

        assert_eq!(
            fail.read_u32()[0],
            0,
            "barrier fail at E4B dense stress dims"
        );
        let max_err = max_abs_err(&expect, &ob.read_f32());
        assert!(
            max_err < 5e-4,
            "E4B-dims dense persistent_interp max_err={max_err}"
        );
        set_persistent_interp(false);
    }

    fn test_q4_banks(
        gpu: &GemmaGpu,
        rows: usize,
        cols: usize,
        group: usize,
        seed: u32,
    ) -> HotQuantBanks {
        let groups = cols / group;
        let packs_per_row = cols / 8;
        let mut weight_u32 = vec![0u32; rows * packs_per_row];
        let mut scales = vec![0f32; rows * groups];
        let mut biases = vec![0f32; rows * groups];
        for r in 0..rows {
            for c in 0..cols {
                let nibble = ((r * seed as usize + c * 5) % 15) as u32;
                let wi = r * packs_per_row + c / 8;
                let shift = (c % 8) * 4;
                weight_u32[wi] |= nibble << shift;
            }
            for g in 0..groups {
                scales[r * groups + g] = 0.04 + (r as f32) * 0.0005;
                biases[r * groups + g] = -0.15;
            }
        }
        let q = crate::quant::quant_matrix_from_mlx_q4(
            rows,
            cols,
            group,
            &weight_u32,
            &scales,
            &biases,
        )
        .unwrap();
        let packed = gpu.rt.alloc_buffer(q.packed.len().max(1)).unwrap();
        packed.write_bytes(&q.packed);
        let sb = pack_mlx_sb_bf16(&q.scales, &q.zeros);
        let scales_b = gpu.rt.alloc_buffer(sb.len().max(1) * 2).unwrap();
        scales_b.write_bf16_bits(&sb);
        let zeros_b = gpu.rt.alloc_buffer(4).unwrap();
        HotQuantBanks {
            scheme: q.scheme,
            layout: HotGemvLayout::RowMajor,
            rows: rows as u32,
            cols: cols as u32,
            group_size: group as u32,
            packed,
            scales: scales_b,
            zeros: zeros_b,
        }
    }

    /// Hot Q4 bounded-TG gate→down vs shipping gate_up_gelu + down resid.
    /// Serialize Metal: `--test-threads=1`
    #[test]
    fn persistent_interp_gate_down_q4_matches_mlp_fuse() {
        let gpu = gpu_or_fail();
        if gpu
            .rt
            .pipeline(KernelId::PersistentInterpGateDownQ4.entry_name())
            .is_err()
        {
            eprintln!("skip: persistent_interp_gate_down_q4 not in metallib");
            return;
        }
        std::env::set_var("GEMMA_METAL_FUSE_MLP", "1");
        std::env::set_var("GEMMA_METAL_GEMV_SIMD", "1");
        set_fuse_gate_down(true);
        assert!(fuse_gate_down_enabled());

        // Non-square MLP: mid=512, hidden=256 (SIMD_BLOCK-friendly K on down).
        let n_mid = 512usize;
        let n_hidden = 256usize;
        let group = 64usize;
        let gate = test_q4_banks(&gpu, n_mid, n_hidden, group, 3);
        let up = test_q4_banks(&gpu, n_mid, n_hidden, group, 5);
        let down = test_q4_banks(&gpu, n_hidden, n_mid, group, 7);
        assert!(
            gate.can_fuse_gate_down(&up, &down),
            "can_fuse_gate_down preconditions (mid={n_mid} hidden={n_hidden})"
        );

        // Mild x keeps MLX affine outputs in a sane range for abs-err checks.
        let x: Vec<f32> = (0..n_hidden)
            .map(|i| ((i % 17) as f32) * 0.01 - 0.05)
            .collect();
        let resid: Vec<f32> = (0..n_hidden)
            .map(|i| ((i % 9) as f32) * 0.01 - 0.04)
            .collect();
        let x_f32 = gpu.rt.alloc_buffer(x.len() * 4).unwrap();
        x_f32.write_f32(&x);
        let xb = prepare_act_bf16(&gpu, &x_f32, n_hidden as u32).unwrap();

        let mid = gpu.rt.alloc_buffer(n_mid * 2).unwrap();
        let x_ref = gpu.rt.alloc_buffer(n_hidden * 4).unwrap();
        let x_fused = gpu.rt.alloc_buffer(n_hidden * 4).unwrap();
        x_ref.write_f32(&resid);
        x_fused.write_f32(&resid);

        // hazard_barriers_skip_auto: sync between producer mid and down resid.
        gemv_q4_mlx_gate_up_gelu_bf16_x_out_bf16(&gpu, &gate, &up, &xb, &mid).unwrap();
        gpu.synchronize().unwrap();
        down.gemv_add_into_bf16_x(&gpu, &mid, &x_ref, &x_ref)
            .unwrap();
        gpu.synchronize().unwrap();
        let expect = x_ref.read_f32();

        let prog = persistent_interp_gate_down_program();
        let insns = gpu.rt.alloc_buffer(prog.len() * 4).unwrap();
        insns.write_u32(&prog);
        let deps = gpu.rt.alloc_buffer(8).unwrap();
        let fail = gpu.rt.alloc_buffer(4).unwrap();
        deps.write_u32(&[0, 0]);
        fail.write_u32(&[0]);

        persistent_interp_gate_down_q4(
            &gpu,
            &insns,
            prog.len() as u32,
            &gate,
            &up,
            &down,
            &xb,
            &mid,
            &x_fused,
            &deps,
            &fail,
            PERSISTENT_INTERP_MAX_TG,
            true,
        )
        .unwrap();
        gpu.synchronize().unwrap();
        assert_eq!(fail.read_u32()[0], 0, "Q4 gate→down barrier fail");
        let max_err = max_abs_err(&expect, &x_fused.read_f32());
        assert!(max_err < 1e-3, "Q4 persistent gate→down max_err={max_err}");
        set_fuse_gate_down(false);
    }

    /// E4B MLP dims: Q4 peel is exact with encoder sync; single-dispatch grid
    /// barrier does **not** publish `mid` to DOWN (D17/D18 hard blocker).
    /// Serialize Metal: `--test-threads=1`
    #[test]
    fn persistent_interp_gate_down_q4_e4b_dims_visibility_blocker() {
        let gpu = gpu_or_fail();
        if gpu
            .rt
            .pipeline(KernelId::PersistentInterpGateDownQ4.entry_name())
            .is_err()
        {
            eprintln!("skip: persistent_interp_gate_down_q4 not in metallib");
            return;
        }
        std::env::set_var("GEMMA_METAL_FUSE_MLP", "1");
        std::env::set_var("GEMMA_METAL_GEMV_SIMD", "1");
        set_fuse_gate_down(true);
        assert!(fuse_gate_down_enabled());

        let n_mid = 10240usize;
        let n_hidden = 2560usize;
        let group = 64usize;
        let gate = test_q4_banks(&gpu, n_mid, n_hidden, group, 3);
        let up = test_q4_banks(&gpu, n_mid, n_hidden, group, 5);
        let down = test_q4_banks(&gpu, n_hidden, n_mid, group, 7);
        assert!(
            gate.can_fuse_gate_down(&up, &down),
            "can_fuse_gate_down E4B dims"
        );

        let x: Vec<f32> = (0..n_hidden)
            .map(|i| ((i % 17) as f32) * 0.01 - 0.05)
            .collect();
        let resid: Vec<f32> = (0..n_hidden)
            .map(|i| ((i % 9) as f32) * 0.01 - 0.04)
            .collect();
        let x_f32 = gpu.rt.alloc_buffer(x.len() * 4).unwrap();
        x_f32.write_f32(&x);
        let xb = prepare_act_bf16(&gpu, &x_f32, n_hidden as u32).unwrap();

        let mid = gpu.rt.alloc_buffer(n_mid * 2).unwrap();
        let x_ref = gpu.rt.alloc_buffer(n_hidden * 4).unwrap();
        let x_fused = gpu.rt.alloc_buffer(n_hidden * 4).unwrap();
        x_ref.write_f32(&resid);
        x_fused.write_f32(&resid);

        gemv_q4_mlx_gate_up_gelu_bf16_x_out_bf16(&gpu, &gate, &up, &xb, &mid).unwrap();
        gpu.synchronize().unwrap();
        down.gemv_add_into_bf16_x(&gpu, &mid, &x_ref, &x_ref)
            .unwrap();
        gpu.synchronize().unwrap();
        let expect = x_ref.read_f32();

        let read_mid_bf16 = |buf: &GpuBuffer| -> Vec<f32> {
            buf.contents_u8()
                .chunks_exact(2)
                .take(n_mid)
                .map(|c| crate::quant::bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect()
        };
        let mid_ship = read_mid_bf16(&mid);

        // --- Two-dispatch split FIRST (encoder sync between PRODUCE and DOWN) ---
        let x_split = gpu.rt.alloc_buffer(n_hidden * 4).unwrap();
        x_split.write_f32(&resid);
        let mid_split = gpu.rt.alloc_buffer(n_mid * 2).unwrap();
        mid_split.zero();
        let produce_only = [PI_OP_PRODUCE_MID, PI_OP_HALT];
        let down_only = [PI_OP_DOWN_PROJ, PI_OP_HALT];
        let ins_p = gpu.rt.alloc_buffer(produce_only.len() * 4).unwrap();
        ins_p.write_u32(&produce_only);
        let ins_d = gpu.rt.alloc_buffer(down_only.len() * 4).unwrap();
        ins_d.write_u32(&down_only);
        let deps = gpu.rt.alloc_buffer(8).unwrap();
        let fail = gpu.rt.alloc_buffer(4).unwrap();
        deps.write_u32(&[0, 0]);
        fail.write_u32(&[0]);
        persistent_interp_gate_down_q4(
            &gpu,
            &ins_p,
            produce_only.len() as u32,
            &gate,
            &up,
            &down,
            &xb,
            &mid_split,
            &x_split,
            &deps,
            &fail,
            PERSISTENT_INTERP_MAX_TG,
            true,
        )
        .unwrap();
        gpu.synchronize().unwrap();
        let split_mid_err = max_abs_err(&mid_ship, &read_mid_bf16(&mid_split));
        deps.write_u32(&[0, 0]);
        fail.write_u32(&[0]);
        persistent_interp_gate_down_q4(
            &gpu,
            &ins_d,
            down_only.len() as u32,
            &gate,
            &up,
            &down,
            &xb,
            &mid_split,
            &x_split,
            &deps,
            &fail,
            PERSISTENT_INTERP_MAX_TG,
            true,
        )
        .unwrap();
        gpu.synchronize().unwrap();
        let split_err = max_abs_err(&expect, &x_split.read_f32());
        eprintln!(
            "[e4b_q4_gate_down] two_dispatch_split_err={split_err:.6e} split_mid_err={split_mid_err:.6e}"
        );

        // --- Single-dispatch PRODUCE→BARRIER→DOWN (shipping Hot path) ---
        let mid_f = gpu.rt.alloc_buffer(n_mid * 2).unwrap();
        mid_f.zero();
        let prog = persistent_interp_gate_down_program();
        let insns = gpu.rt.alloc_buffer(prog.len() * 4).unwrap();
        insns.write_u32(&prog);
        deps.write_u32(&[0, 0]);
        fail.write_u32(&[0]);
        persistent_interp_gate_down_q4(
            &gpu,
            &insns,
            prog.len() as u32,
            &gate,
            &up,
            &down,
            &xb,
            &mid_f,
            &x_fused,
            &deps,
            &fail,
            PERSISTENT_INTERP_MAX_TG,
            true,
        )
        .unwrap();
        gpu.synchronize().unwrap();
        assert_eq!(
            fail.read_u32()[0],
            0,
            "Q4 gate→down barrier fail at E4B dims"
        );
        let got = x_fused.read_f32();
        let max_err = max_abs_err(&expect, &got);
        let mid_fuse = read_mid_bf16(&mid_f);
        let mid_err = max_abs_err(&mid_ship, &mid_fuse);
        let mid_bad = mid_ship
            .iter()
            .zip(mid_fuse.iter())
            .filter(|(a, b)| (*a - *b).abs() > 1e-3)
            .count();
        let x_down_only = gpu.rt.alloc_buffer(n_hidden * 4).unwrap();
        x_down_only.write_f32(&resid);
        down.gemv_add_into_bf16_x(&gpu, &mid_f, &x_down_only, &x_down_only)
            .unwrap();
        gpu.synchronize().unwrap();
        let down_only_err = max_abs_err(&expect, &x_down_only.read_f32());
        eprintln!(
            "[e4b_q4_gate_down] single_resid_err={max_err:.6e} mid_err={mid_err:.6e} \
             mid_bad_rows={mid_bad}/{n_mid} shipping_down_on_fused_mid_err={down_only_err:.6e}"
        );

        assert!(
            split_mid_err < 1e-3 && split_err < 1e-3,
            "E4B two-dispatch PRODUCE|sync|DOWN should match shipping \
             (split_err={split_err} split_mid_err={split_mid_err})"
        );
        // Host-visible mid after single-dispatch is exact; in-kernel DOWN is not.
        assert!(
            mid_err < 1e-3 && down_only_err < 1e-3,
            "post-kernel mid must match shipping (mid_err={mid_err} down_only_err={down_only_err})"
        );
        assert!(
            max_err > 1.0,
            "expected single-dispatch resid blow-up (D18 visibility blocker); got max_err={max_err}"
        );
        eprintln!(
            "[e4b_q4_gate_down] HARD_BLOCKER confirmed: single-dispatch resid_err={max_err} \
             with host mid_err=0 — relaxed grid barrier does not publish mid to DOWN"
        );
        set_fuse_gate_down(false);
    }
}

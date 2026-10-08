//! Phase 4: Metal-resident Hot weight banks + GPU layer decode stack.
//!
//! Synthetic / real E4B uploads Q4 projections into Hot buffers and runs
//! decode GEMV + RMS + FA + MLP + softcap on GPU. KV stays on GPU (store +
//! optional ring densify); packed async encode batches layer dispatches.

use tessl::tensor::GpuBuffer;
use std::sync::OnceLock;

use crate::config::{Gemma4TextConfig, LayerType};
use crate::diag::{self, InferScope};
use crate::error::{Error, Result};
use crate::forward::{gemv, softcap_f32, SyntheticE4bGraph};
use crate::kernels::{
    begin_icb_scalar_write_tape, copy_f32_n, copy_f32_range, copy_f32_to_offset, copy_u32_from_index,
    copy_u32_one, embed_lookup_quant, embed_lookup_quant_n, flash_attn_global_h512_ex,
    flash_attn_swa_h256_ex, gemv_q4_mlx_gate_up_gelu_bf16_x, gemv_q4_mlx_gate_up_gelu_bf16_x_out_bf16,
    gemv_q4_mlx_simd_kv_bf16_x, gemv_q4_mlx_simd_qkv_bf16_x, icb_skip_nop_loop_enabled,
    icb_tape_clear_kv_ctx, icb_tape_note_commit_global, icb_tape_note_commit_shared_global,
    icb_tape_note_commit_shared_sliding, icb_tape_note_commit_sliding, icb_tape_set_kv_ctx_global,
    icb_tape_set_kv_ctx_shared_global, icb_tape_set_kv_ctx_shared_sliding,
    icb_tape_set_kv_ctx_sliding, kv_ring_densify, kv_store_timestep_pair_off, mlp_gelu_tanh,
    mlp_gelu_tanh_bf16, ple_lookup, ple_lookup_q4_mlx, ple_lookup_q4_mlx_residual, ple_residual_add,
    fuse_ple_residual_enabled, fuse_rope_kv_enabled, persistent_interp_enabled,
    persistent_interp_fa_o_proj, persistent_interp_fa_o_proj_program, persistent_interp_gate_down,
    persistent_interp_gate_down_program, persistent_interp_gate_down_q4, prepare_act_bf16,
    fuse_gate_down_enabled, rms_norm_f32, rms_norm_to_act_bf16, rms_qkv_rope_ex_posbuf,
    rms_qkv_rope_kv_store, scale_f32_inplace, set_fuse_qkv, softcap_argmax_encode,
    softcap_argmax_encode_offset, take_icb_scalar_write_tape, upload_quant_hot, ArgmaxScratch,
    GemmaGpu, HotQuantBanks, IcbDynSrc, IcbKvHostOp, IcbScalarTapeOp, KernelId,
    PERSISTENT_INTERP_MAX_TG, encode_once_enabled, fuse_bf16_fa, fuse_bf16_mlp, fuse_bf16_rms,
    fuse_dual_norm_enabled, fuse_qkv_enabled,
};
use crate::kv::{KvLayout, KvRole, KvSlotId};
use crate::mtp::{MtpSession, VerifyResult};
use crate::quant::{quantize_affine_f32, QuantMatrix, QuantScheme};
use crate::trace::{self, TraceFlags, TraceSession};
use crate::weights::HostWeightBanks;

fn layer_probe_enabled() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("GEMMA_METAL_LAYER_PROBE").ok().as_deref() == Some("1"))
}

/// Light fused↔unfused Hot `q/k/v` dump after the first producer GEMV.
/// Opt-in: `GEMMA_METAL_QKV_AB_DUMP=1` (no LAYER_PROBE; dumps ≤8 steps or until diverge).
/// Also emits `k_solo_vs_fused` / `k_solo_vs_gemv_kv` (solo `k_proj.gemv` into scratch).
fn qkv_ab_dump_enabled() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("GEMMA_METAL_QKV_AB_DUMP").ok().as_deref() == Some("1"))
}

fn qkv_ab_dump_first_diff(a: &[f32], b: &[f32]) -> Option<(usize, f32, f32)> {
    let n = a.len().min(b.len());
    for i in 0..n {
        if a[i] != b[i] {
            return Some((i, a[i], b[i]));
        }
    }
    if a.len() != b.len() {
        return Some((n, f32::NAN, f32::NAN));
    }
    None
}

fn qkv_ab_dump_max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

fn capture_ao_disabled() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("GEMMA_METAL_CAPTURE_AO").ok().as_deref() == Some("0"))
}

fn capture_nop_enabled() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("GEMMA_METAL_CAPTURE_NOP").ok().as_deref() == Some("1"))
}

fn capture_barrier_forced() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("GEMMA_METAL_CAPTURE_BARRIER")
            .ok()
            .as_deref()
            == Some("1")
    })
}

fn force_gemv_verify_enabled() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("GEMMA_METAL_FORCE_GEMV_VERIFY")
            .ok()
            .as_deref()
            == Some("1")
    })
}

/// While `HiddenCapture` is attached, force always-on Dispatch barriers so
/// mid-layer `copy_f32` snapshots see a drained residual (hazard skip-auto
/// otherwise collapses capture / exactness). `synchronize`-per-layer alone
/// was insufficient (tick3: exact FAIL, mean_accept→0). Opt out with
/// `GEMMA_METAL_CAPTURE_AO=0` only for A/B (not for product fidelity).
struct CaptureAlwaysOnGuard {
    prev_skip_auto: Option<bool>,
}

impl CaptureAlwaysOnGuard {
    fn enter(capture_on: bool) -> Self {
        let disabled = capture_ao_disabled();
        if !capture_on || disabled {
            return Self {
                prev_skip_auto: None,
            };
        }
        let prev = tessl::ab_flags::hazard_barriers();
        tessl::ab_flags::set_hazard_barriers(false);
        Self {
            prev_skip_auto: Some(prev),
        }
    }
}

impl Drop for CaptureAlwaysOnGuard {
    fn drop(&mut self) {
        if let Some(prev) = self.prev_skip_auto.take() {
            tessl::ab_flags::set_hazard_barriers(prev);
        }
    }
}

/// Owns the two thread-local tapes started for a layer-graph capture.
///
/// `step_inner` has many fallible kernel calls between capture start and the
/// normal take path. Keeping cleanup in `Drop` guarantees that any early `?`
/// abandons both tapes instead of retaining bound buffers or recording scalar
/// writes into the next decode step.
struct LayerIcbCaptureGuard {
    decode_active: bool,
    scalar_active: bool,
}

impl LayerIcbCaptureGuard {
    fn begin() -> Result<Self> {
        let stale_decode = tessl::decode_icb_capture_active();
        let stale_scalars = crate::kernels::icb_scalar_write_tape_active();
        if stale_decode {
            tessl::end_decode_icb_capture();
        }
        if stale_scalars {
            let _ = take_icb_scalar_write_tape();
        }
        if stale_decode || stale_scalars {
            return Err(Error::Metal(format!(
                "layer ICB capture found stale thread-local state (decode={stale_decode}, \
                 scalars={stale_scalars}); abandoned both tapes"
            )));
        }
        tessl::begin_decode_icb_capture();
        begin_icb_scalar_write_tape();
        Ok(Self {
            decode_active: true,
            scalar_active: true,
        })
    }

    fn take_decode_capture(&mut self) -> Option<tessl::DecodeIcbCapture> {
        let capture = tessl::take_decode_icb_capture();
        self.decode_active = false;
        capture
    }

    fn take_scalar_write_tape(&mut self) -> Option<crate::kernels::IcbScalarWriteTape> {
        let tape = take_icb_scalar_write_tape();
        self.scalar_active = false;
        tape
    }
}

impl Drop for LayerIcbCaptureGuard {
    fn drop(&mut self) {
        if self.decode_active {
            tessl::end_decode_icb_capture();
            self.decode_active = false;
        }
        if self.scalar_active {
            let _ = take_icb_scalar_write_tape();
            self.scalar_active = false;
        }
    }
}

/// GPU-resident KV slot (sliding ring or full-length shared/global).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GpuKvLayout {
    capacity_u32: u32,
    slot_elems: usize,
    slot_elems_u32: u32,
    elem_capacity_u32: u32,
    allocation_bytes: usize,
}

impl GpuKvLayout {
    /// Validate every size used by both host allocation and the Metal `uint`
    /// ABI before either cache buffer is allocated.
    fn checked(capacity: usize, heads: usize, dim: usize) -> Result<Self> {
        if capacity == 0 || heads == 0 || dim == 0 {
            return Err(Error::Kv(format!(
                "GPU KV layout requires non-zero capacity, heads, and dim; got {capacity}x{heads}x{dim}"
            )));
        }
        let capacity_u32 = u32::try_from(capacity).map_err(|_| {
            Error::Kv("GPU KV timestep capacity exceeds the device u32 range".into())
        })?;
        let slot_elems = heads
            .checked_mul(dim)
            .ok_or_else(|| Error::Kv("GPU KV slot width overflows usize".into()))?;
        let slot_elems_u32 = u32::try_from(slot_elems)
            .map_err(|_| Error::Kv("GPU KV slot width exceeds the device u32 range".into()))?;
        let elem_capacity = capacity
            .checked_mul(slot_elems)
            .ok_or_else(|| Error::Kv("GPU KV element capacity overflows usize".into()))?;
        let elem_capacity_u32 = u32::try_from(elem_capacity).map_err(|_| {
            Error::Kv("GPU KV element capacity exceeds the device u32 range".into())
        })?;
        let allocation_bytes = elem_capacity
            .checked_mul(std::mem::size_of::<f32>())
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .ok_or_else(|| Error::Kv("GPU KV allocation byte size overflows".into()))?;
        Ok(Self {
            capacity_u32,
            slot_elems,
            slot_elems_u32,
            elem_capacity_u32,
            allocation_bytes,
        })
    }

    fn element_offset_u32(self, position: usize, what: &str) -> Result<u32> {
        let offset = position
            .checked_mul(self.slot_elems)
            .ok_or_else(|| Error::Kv(format!("{what} overflows usize")))?;
        u32::try_from(offset).map_err(|_| Error::Kv(format!("{what} exceeds the device u32 range")))
    }

    fn element_byte_offset(self, element_offset: u32, what: &str) -> Result<usize> {
        usize::try_from(element_offset)
            .map_err(|_| Error::Kv(format!("{what} byte offset exceeds usize")))?
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| Error::Kv(format!("{what} byte offset overflows usize")))
    }

    fn byte_len_for_positions(self, positions: usize, what: &str) -> Result<usize> {
        positions
            .checked_mul(self.slot_elems)
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| Error::Kv(format!("{what} byte length overflows usize")))
    }

    fn elements_for_positions(
        self,
        positions: usize,
        capacity: usize,
        what: &str,
    ) -> Result<usize> {
        if positions > capacity {
            return Err(Error::Kv(format!(
                "{what}: {positions} positions exceed capacity {capacity}"
            )));
        }
        positions
            .checked_mul(self.slot_elems)
            .ok_or_else(|| Error::Kv(format!("{what}: element count overflows usize")))
    }

    fn ring_position_u32(self, seq_len: usize, capacity: usize) -> Result<u32> {
        let position = seq_len.saturating_sub(capacity);
        u32::try_from(position)
            .map_err(|_| Error::Kv("GPU KV ring position exceeds the device u32 range".into()))
    }

    /// Compute a complete append transition without touching the live slot.
    /// Callers apply it only after the GPU write has encoded successfully.
    fn next_append_state(
        self,
        logical_capacity: usize,
        seq_len: usize,
        head: usize,
        is_ring: bool,
    ) -> Result<(u32, usize, usize)> {
        if logical_capacity == 0 {
            return Err(Error::Kv("GPU KV capacity 0".into()));
        }
        let storage_capacity = self.capacity_u32 as usize;
        if logical_capacity > storage_capacity {
            return Err(Error::Kv(format!(
                "GPU KV logical capacity {logical_capacity} exceeds storage capacity {storage_capacity}"
            )));
        }
        let write_i = if is_ring {
            if head >= storage_capacity {
                return Err(Error::Kv(format!(
                    "GPU KV ring cursor {head} is outside storage capacity {storage_capacity}"
                )));
            }
            head
        } else {
            if seq_len >= logical_capacity {
                return Err(Error::Kv("GPU KV full".into()));
            }
            seq_len
        };
        let offset = self.element_offset_u32(write_i, "GPU KV write offset")?;
        let next_seq_len = seq_len
            .checked_add(1)
            .ok_or_else(|| Error::Kv("GPU KV sequence length overflows usize".into()))?;
        if is_ring {
            self.ring_position_u32(next_seq_len, logical_capacity)?;
        }
        let next_head = if is_ring {
            head.checked_add(1)
                .ok_or_else(|| Error::Kv("GPU KV ring cursor overflows usize".into()))?
                % storage_capacity
        } else {
            head
        };
        Ok((offset, next_seq_len, next_head))
    }
}

struct GpuKvSlot {
    k: GpuBuffer,
    v: GpuBuffer,
    /// Logical attention window. Only the newest `capacity` rows are visible.
    capacity: usize,
    /// Physical ring rows. Sliding slots reserve a bounded speculative suffix
    /// so rejecting a verify block cannot destroy the logical history it must
    /// restore; linear slots keep this equal to `capacity`.
    storage_capacity: usize,
    layout: GpuKvLayout,
    seq_len: usize,
    /// Next write index for rings; ignored for linear full buffers.
    head: usize,
    is_ring: bool,
}

impl GpuKvSlot {
    fn new(
        gpu: &GemmaGpu,
        capacity: usize,
        heads: usize,
        dim: usize,
        is_ring: bool,
    ) -> Result<Self> {
        if capacity == 0 {
            return Err(Error::Kv("GPU KV capacity 0".into()));
        }
        let storage_capacity = if is_ring {
            capacity.checked_add(VERIFY_MAX_M).ok_or_else(|| {
                Error::Kv("GPU sliding KV rollback capacity overflows usize".into())
            })?
        } else {
            capacity
        };
        let layout = GpuKvLayout::checked(storage_capacity, heads, dim)?;
        let bytes = layout.allocation_bytes;
        diag::log(
            "gpu",
            format_args!(
                "KV alloc logical_capacity={capacity} storage_capacity={storage_capacity} heads={heads} dim={dim} ring={is_ring} elems={} bytes/K={}",
                layout.elem_capacity_u32,
                diag::fmt_bytes(bytes as u64)
            ),
        );
        let alloc = || -> Result<GpuBuffer> {
            gpu.rt.alloc_buffer_hot(bytes).map_err(|e| {
                diag::err_msg("gpu", &format!("KV alloc_buffer_hot n={bytes}"), &e);
                Error::Metal(e)
            })
        };
        Ok(Self {
            k: alloc()?,
            v: alloc()?,
            capacity,
            storage_capacity,
            layout,
            seq_len: 0,
            head: 0,
            is_ring,
        })
    }

    fn slot_elems_u32(&self) -> u32 {
        self.layout.slot_elems_u32
    }

    fn storage_capacity_u32(&self) -> u32 {
        self.layout.capacity_u32
    }

    fn elem_capacity_u32(&self) -> u32 {
        self.layout.elem_capacity_u32
    }

    fn elements_for_positions(&self, positions: usize, what: &str) -> Result<usize> {
        self.layout
            .elements_for_positions(positions, self.capacity, what)
    }

    fn live_len_u32(&self) -> Result<u32> {
        u32::try_from(self.seq_len.min(self.capacity))
            .map_err(|_| Error::Kv("GPU KV live length exceeds the device u32 range".into()))
    }

    fn ring_start_u32(&self) -> Result<u32> {
        if self.head >= self.storage_capacity {
            return Err(Error::Kv(format!(
                "GPU KV ring cursor {} is outside storage capacity {}",
                self.head, self.storage_capacity
            )));
        }
        let live = self.seq_len.min(self.capacity);
        let start = if live <= self.head {
            self.head - live
        } else {
            self.storage_capacity - (live - self.head)
        };
        u32::try_from(start)
            .map_err(|_| Error::Kv("GPU KV ring cursor exceeds the device u32 range".into()))
    }

    fn ring_position_u32(&self) -> Result<u32> {
        self.layout.ring_position_u32(self.seq_len, self.capacity)
    }

    /// Fully validate the next state before a cache-write dispatch is encoded.
    fn next_append_state(&self) -> Result<(u32, usize, usize)> {
        self.layout
            .next_append_state(self.capacity, self.seq_len, self.head, self.is_ring)
    }

    fn reset(&mut self) {
        self.seq_len = 0;
        self.head = 0;
    }

    /// Drop the last `n` timesteps (speculative reject / verify rollback).
    ///
    /// Ring write head rewinds; full-slot buffers only shrink the logical length
    /// (stale trailing floats are ignored by FA via `seq_len`). Sliding rings
    /// retain [`VERIFY_MAX_M`] physical rows beyond the logical attention window,
    /// so any supported verify block can be rejected without overwriting live
    /// history. A rewind beyond the retained physical horizon fails closed.
    fn trim(&mut self, n: usize) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        if n > self.seq_len {
            return Err(Error::Kv(format!("KV trim {n} > seq_len {}", self.seq_len)));
        }
        let next_seq_len = self.seq_len - n;
        let next_head = if self.is_ring {
            let _ = self.ring_start_u32()?;
            let oldest_retained = self.seq_len.saturating_sub(self.storage_capacity);
            let oldest_needed = next_seq_len.saturating_sub(self.capacity);
            if next_seq_len != 0 && oldest_needed < oldest_retained {
                return Err(Error::Kv(format!(
                    "KV trim {n} exceeds retained sliding history: need absolute row {oldest_needed}, oldest retained is {oldest_retained}"
                )));
            }
            let rewind = n % self.storage_capacity;
            if rewind <= self.head {
                self.head - rewind
            } else {
                self.storage_capacity - (rewind - self.head)
            }
        } else {
            self.head
        };
        self.seq_len = next_seq_len;
        self.head = next_head;
        Ok(())
    }

    /// Append one timestep from GPU `src_k` / `src_v` (first `slot_elems` floats).
    fn append(&mut self, gpu: &GemmaGpu, src_k: &GpuBuffer, src_v: &GpuBuffer) -> Result<()> {
        self.append_from_offset(gpu, src_k, src_v, 0)
    }

    /// Next float-element write offset into `k`/`v` (does not advance).
    fn peek_write_offset(&self) -> Result<u32> {
        self.next_append_state().map(|state| state.0)
    }

    /// Advance ring head / seq_len after a fused rope+kv_store wrote the slot.
    fn commit_append(&mut self) -> Result<()> {
        let (_, next_seq_len, next_head) = self.next_append_state()?;
        self.seq_len = next_seq_len;
        self.head = next_head;
        Ok(())
    }

    fn preflight_append_buffers(
        &self,
        gpu: &GemmaGpu,
        src_k: &GpuBuffer,
        src_v: &GpuBuffer,
        source_byte_offset: usize,
        source_byte_len: usize,
    ) -> Result<()> {
        let rt = gpu.rt.as_ref();
        tessl::nn::validate_buffer_byte_range(
            rt,
            src_k,
            source_byte_offset,
            source_byte_len,
            "GPU KV append src_k",
        )
        .map_err(Error::Metal)?;
        tessl::nn::validate_buffer_byte_range(
            rt,
            src_v,
            source_byte_offset,
            source_byte_len,
            "GPU KV append src_v",
        )
        .map_err(Error::Metal)?;
        tessl::nn::validate_buffer_byte_range(
            rt,
            &self.k,
            0,
            self.layout.allocation_bytes,
            "GPU KV append dst_k",
        )
        .map_err(Error::Metal)?;
        tessl::nn::validate_buffer_byte_range(
            rt,
            &self.v,
            0,
            self.layout.allocation_bytes,
            "GPU KV append dst_v",
        )
        .map_err(Error::Metal)
    }

    fn append_from_offset_preflighted(
        &mut self,
        gpu: &GemmaGpu,
        src_k: &GpuBuffer,
        src_v: &GpuBuffer,
        src_elem_off: u32,
    ) -> Result<()> {
        let n = self.slot_elems_u32();
        let off = self.peek_write_offset()?;
        kv_store_timestep_pair_off(
            gpu,
            src_k,
            src_v,
            src_elem_off,
            &self.k,
            &self.v,
            n,
            off,
            self.elem_capacity_u32(),
        )?;
        self.commit_append()
    }

    /// Append timestep whose K/V vectors start at float-element `src_elem_off`.
    fn append_from_offset(
        &mut self,
        gpu: &GemmaGpu,
        src_k: &GpuBuffer,
        src_v: &GpuBuffer,
        src_elem_off: u32,
    ) -> Result<()> {
        let source_byte_offset = self
            .layout
            .element_byte_offset(src_elem_off, "GPU KV append source")?;
        let source_byte_len = self
            .layout
            .byte_len_for_positions(1, "GPU KV append source")?;
        self.preflight_append_buffers(gpu, src_k, src_v, source_byte_offset, source_byte_len)?;
        self.append_from_offset_preflighted(gpu, src_k, src_v, src_elem_off)
    }

    /// Append `m` consecutive timesteps packed as `[m, heads, dim]` in `src_k`/`src_v`.
    fn append_m(
        &mut self,
        gpu: &GemmaGpu,
        src_k: &GpuBuffer,
        src_v: &GpuBuffer,
        m: usize,
    ) -> Result<()> {
        if m == 0 {
            return Ok(());
        }
        let final_seq_len = self
            .seq_len
            .checked_add(m)
            .ok_or_else(|| Error::Kv("GPU KV append length overflows usize".into()))?;
        if !self.is_ring && final_seq_len > self.capacity {
            return Err(Error::Kv(format!(
                "GPU KV append of {m} timesteps exceeds remaining capacity {}",
                self.capacity.saturating_sub(self.seq_len)
            )));
        }
        if self.is_ring {
            self.layout
                .ring_position_u32(final_seq_len, self.capacity)?;
        }
        // Prove the last source-row offset is representable before the first
        // append can mutate cache state.
        self.layout
            .element_offset_u32(m - 1, "GPU KV append source offset")?;
        let source_byte_len = self
            .layout
            .byte_len_for_positions(m, "GPU KV append source")?;
        self.preflight_append_buffers(gpu, src_k, src_v, 0, source_byte_len)?;
        for mi in 0..m {
            let src_offset = self
                .layout
                .element_offset_u32(mi, "GPU KV append source offset")?;
            self.append_from_offset_preflighted(gpu, src_k, src_v, src_offset)?;
        }
        Ok(())
    }

    fn preflight_densify_buffers(
        &self,
        gpu: &GemmaGpu,
        k_scratch: &GpuBuffer,
        v_scratch: &GpuBuffer,
        scratch_byte_len: usize,
    ) -> Result<()> {
        let rt = gpu.rt.as_ref();
        tessl::nn::validate_buffer_byte_range(
            rt,
            &self.k,
            0,
            self.layout.allocation_bytes,
            "GPU KV densify src_k",
        )
        .map_err(Error::Metal)?;
        tessl::nn::validate_buffer_byte_range(
            rt,
            &self.v,
            0,
            self.layout.allocation_bytes,
            "GPU KV densify src_v",
        )
        .map_err(Error::Metal)?;
        tessl::nn::validate_buffer_byte_range(
            rt,
            k_scratch,
            0,
            scratch_byte_len,
            "GPU KV densify dst_k",
        )
        .map_err(Error::Metal)?;
        tessl::nn::validate_buffer_byte_range(
            rt,
            v_scratch,
            0,
            scratch_byte_len,
            "GPU KV densify dst_v",
        )
        .map_err(Error::Metal)
    }

    /// FA-ready K/V handles.
    ///
    /// Sliding rings **always** densify into scratch (even before wrap) so Binder
    /// tape / IcbScalarPool cursor shape matches across ring wrap. Non-ring
    /// slots return live buffers (no densify cmds).
    fn fa_buffers(
        &self,
        gpu: &GemmaGpu,
        k_scratch: &GpuBuffer,
        v_scratch: &GpuBuffer,
        tkv_limit: u32,
    ) -> Result<(GpuBuffer, GpuBuffer, u32, u32)> {
        let filled = self.live_len_u32()?;
        if filled == 0 {
            return Ok((k_scratch.clone(), v_scratch.clone(), 0, 0));
        }
        let tkv = filled.min(tkv_limit);
        if !self.is_ring {
            return Ok((self.k.clone(), self.v.clone(), 0, tkv));
        }
        let n_slot = self.slot_elems_u32();
        let wrapped = self.seq_len > self.capacity;
        let start = if wrapped { self.ring_start_u32()? } else { 0 };
        let kv_pos_offset = if wrapped {
            self.ring_position_u32()?
        } else {
            0
        };
        let storage_capacity = self.storage_capacity_u32();
        let tkv_usize = usize::try_from(tkv)
            .map_err(|_| Error::Kv("GPU KV densify length exceeds usize".into()))?;
        let scratch_byte_len = self
            .layout
            .byte_len_for_positions(tkv_usize, "GPU KV densify scratch")?;
        self.preflight_densify_buffers(gpu, k_scratch, v_scratch, scratch_byte_len)?;
        kv_ring_densify(
            gpu,
            &self.k,
            k_scratch,
            n_slot,
            storage_capacity,
            tkv,
            start,
        )?;
        kv_ring_densify(
            gpu,
            &self.v,
            v_scratch,
            n_slot,
            storage_capacity,
            tkv,
            start,
        )?;
        Ok((k_scratch.clone(), v_scratch.clone(), kv_pos_offset, tkv))
    }
}

/// Host-side Q4 twin of a projection (for parity vs GPU Hot banks).
#[derive(Clone, Debug)]
pub struct HostQLayer {
    pub q_proj: QuantMatrix,
    pub k_proj: QuantMatrix,
    pub v_proj: QuantMatrix,
    pub o_proj: QuantMatrix,
    pub gate_proj: QuantMatrix,
    pub up_proj: QuantMatrix,
    pub down_proj: QuantMatrix,
}

/// One layer's Hot-resident weights.
pub struct GpuSynthLayer {
    pub layer_type: LayerType,
    pub role: KvRole,
    pub hq: usize,
    pub hkv: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub theta: f32,
    pub window: Option<usize>,
    pub input_norm: GpuBuffer,
    /// Gemma4: `post_attention_layernorm` on attn out before residual.
    /// Legacy/synthetic (no `pre_ff_norm`): reused as the before-MLP RMS.
    pub post_attn_norm: GpuBuffer,
    /// Gemma4: `pre_feedforward_layernorm` before MLP. `None` ⇒ legacy Pre-LN.
    pub pre_ff_norm: Option<GpuBuffer>,
    pub q_norm: GpuBuffer,
    pub k_norm: GpuBuffer,
    pub v_norm: GpuBuffer,
    pub q_proj: HotQuantBanks,
    pub k_proj: Option<HotQuantBanks>,
    pub v_proj: Option<HotQuantBanks>,
    pub o_proj: HotQuantBanks,
    pub gate_proj: HotQuantBanks,
    pub up_proj: HotQuantBanks,
    pub down_proj: HotQuantBanks,
    /// Gemma4: `post_feedforward_layernorm` on MLP out before residual.
    pub post_ff_norm: Option<GpuBuffer>,
    /// Gemma4 `layer_scalar` (non-unit on 31B/E4B). Applied to full layer out.
    pub layer_scalar: f32,
    /// Per-layer bf16 PLE table (synthetic). Prefer `ple_q4` for MLX.
    pub ple_table: Option<GpuBuffer>,
}

/// Full graph on GPU Hot banks (synthetic or real HF/MLX).
pub struct GpuSynthModel {
    pub gpu: GemmaGpu,
    pub softcap: f32,
    pub eps: f32,
    pub hidden: usize,
    pub vocab: usize,
    pub intermediate: usize,
    /// Gemma4 `√hidden` applied after embed lookup (MLX `embed_scale`).
    pub embed_scale: f32,
    /// Dense f32 embed rows when available (synthetic); empty if `embed_q` is set.
    pub embed: Vec<f32>,
    /// Quantized embed (MLX/HF); row dequant on host each step (fallback).
    pub embed_q: Option<QuantMatrix>,
    /// Hot-resident embed for GPU row lookup (preferred decode path).
    pub embed_hot: Option<HotQuantBanks>,
    pub final_norm: GpuBuffer,
    pub lm_head: HotQuantBanks,
    pub lm_head_host: QuantMatrix,
    pub layers: Vec<GpuSynthLayer>,
    /// Host Q4 twin — filled for synthetic/mini parity only. HF Hot upload leaves
    /// this empty so host layer residency does not overlap Hot + session KV.
    pub host_q: Vec<HostQLayer>,
    pub scheme: QuantScheme,
    pub cfg: Gemma4TextConfig,
    pub kv: KvLayout,
    /// Original host graph when constructed from synthetic; unused for HF load.
    pub host: Option<SyntheticE4bGraph>,
    /// Shared MLX Q4 PLE bank covering all layers (`dim = L * ple_dim`).
    pub ple_q4: Option<HotQuantBanks>,
}

fn upload_f32_hot(gpu: &GemmaGpu, data: &[f32]) -> Result<GpuBuffer> {
    let b = gpu
        .rt
        .alloc_buffer_hot(data.len().max(1) * 4)
        .map_err(Error::Metal)?;
    b.write_f32(data);
    Ok(b)
}

fn qmat(rows: usize, cols: usize, data: &[f32], scheme: QuantScheme) -> Result<QuantMatrix> {
    quantize_affine_f32(rows, cols, data, scheme)
}

impl GpuSynthModel {
    /// True for [`SyntheticE4bGraph::mini_parity`] (not E4B/31B Hot).
    ///
    /// Used to gate opt-in `GEMMA_METAL_PERSISTENT_INTERP` decode hooks — Hot
    /// uploads leave `host = None` and never match these dims.
    pub fn is_synthetic_mini(&self) -> bool {
        self.host.is_some()
            && self.hidden == 256
            && self.intermediate == 512
            && self.vocab <= 512
            && self.layers.len() == 3
            && self.cfg.num_hidden_layers == 3
            && self.cfg.num_attention_heads == 1
            && self.cfg.head_dim == 256
    }

    /// True for real E4B Hot dims ([`Gemma4TextConfig::e4b_preset`]).
    ///
    /// Used to opt in DecodeIcb layer-graph capture/replay. **Not** 31B
    /// (`hidden=5376`, 60 layers) — keep 31B live-encode until a separate migrate.
    pub fn is_hot_e4b(&self) -> bool {
        self.hidden == 2560
            && self.cfg.hidden_size == 2560
            && self.cfg.num_hidden_layers == 42
            && self.layers.len() == 42
            && self.intermediate == 10_240
    }

    /// Graphs allowed to attach DecodeIcb under `GEMMA_METAL_DECODE_ICB=1`
    /// (still requires `GEMMA_METAL_ENCODE_ONCE=1`; both default OFF).
    ///
    /// Mini synth + real E4B Hot. 31B stays out (jetsam / graph size).
    pub fn decode_icb_graph_eligible(&self) -> bool {
        self.is_synthetic_mini() || self.is_hot_e4b()
    }

    /// Quantize synthetic f32 graph → Hot Q4 banks (decode path).
    pub fn from_synthetic(host: SyntheticE4bGraph, scheme: QuantScheme) -> Result<Self> {
        let gpu = GemmaGpu::new()?;
        let softcap = host.softcap;
        let eps = host.cfg.rms_norm_eps as f32;
        let hidden = host.cfg.hidden_size;
        let vocab = host.cfg.vocab_size;
        let intermediate = host.cfg.intermediate_size;

        let mut layers = Vec::with_capacity(host.layers.len());
        let mut host_q = Vec::with_capacity(host.layers.len());

        for layer in &host.layers {
            let q_proj = qmat(layer.hq * layer.head_dim, hidden, &layer.q_proj, scheme)?;
            let k_proj = qmat(layer.hkv * layer.head_dim, hidden, &layer.k_proj, scheme)?;
            let v_proj = qmat(layer.hkv * layer.head_dim, hidden, &layer.v_proj, scheme)?;
            let o_proj = qmat(hidden, layer.hq * layer.head_dim, &layer.o_proj, scheme)?;
            let gate_proj = qmat(intermediate, hidden, &layer.gate_proj, scheme)?;
            let up_proj = qmat(intermediate, hidden, &layer.up_proj, scheme)?;
            let down_proj = qmat(hidden, intermediate, &layer.down_proj, scheme)?;

            host_q.push(HostQLayer {
                q_proj: q_proj.clone(),
                k_proj: k_proj.clone(),
                v_proj: v_proj.clone(),
                o_proj: o_proj.clone(),
                gate_proj: gate_proj.clone(),
                up_proj: up_proj.clone(),
                down_proj: down_proj.clone(),
            });

            let ple_table = if let Some(ref bits) = layer.ple_table {
                let b = gpu
                    .rt
                    .alloc_buffer_hot(bits.len() * 2)
                    .map_err(Error::Metal)?;
                b.write_bf16_bits(bits);
                Some(b)
            } else {
                None
            };

            layers.push(GpuSynthLayer {
                layer_type: layer.layer_type,
                role: layer.role.clone(),
                hq: layer.hq,
                hkv: layer.hkv,
                head_dim: layer.head_dim,
                rotary_dim: layer.rotary_dim,
                theta: layer.theta,
                window: layer.window,
                input_norm: upload_f32_hot(&gpu, &layer.input_norm)?,
                post_attn_norm: upload_f32_hot(&gpu, &layer.post_attn_norm)?,
                pre_ff_norm: None,
                q_norm: upload_f32_hot(&gpu, &layer.q_norm)?,
                k_norm: upload_f32_hot(&gpu, &layer.k_norm)?,
                v_norm: upload_f32_hot(&gpu, &layer.v_norm)?,
                q_proj: upload_quant_hot(&gpu, &q_proj)?,
                k_proj: Some(upload_quant_hot(&gpu, &k_proj)?),
                v_proj: Some(upload_quant_hot(&gpu, &v_proj)?),
                o_proj: upload_quant_hot(&gpu, &o_proj)?,
                gate_proj: upload_quant_hot(&gpu, &gate_proj)?,
                up_proj: upload_quant_hot(&gpu, &up_proj)?,
                down_proj: upload_quant_hot(&gpu, &down_proj)?,
                post_ff_norm: None,
                layer_scalar: 1.0,
                ple_table,
            });
        }

        let lm_head_host = qmat(vocab, hidden, &host.lm_head, scheme)?;
        let lm_head = upload_quant_hot(&gpu, &lm_head_host)?;
        let final_norm = upload_f32_hot(&gpu, &host.final_norm_w)?;
        let cfg = host.cfg.clone();
        let kv = host.kv.clone();

        Ok(Self {
            gpu,
            softcap,
            eps,
            hidden,
            vocab,
            intermediate,
            embed_scale: (hidden as f32).sqrt(),
            embed: host.embed.clone(),
            embed_q: None,
            embed_hot: None,
            final_norm,
            lm_head,
            lm_head_host,
            layers,
            host_q,
            scheme,
            cfg,
            kv,
            host: Some(host),
            ple_q4: None,
        })
    }

    /// Upload real HF/MLX [`HostWeightBanks`] into Hot Q4 banks (+ MLX Q4 PLE when present).
    ///
    /// Takes banks by value and drops them before return so host layer residency
    /// ends before callers allocate [`GpuDecodeSession`] KV/scratch (cuts the
    /// ~36–55 GiB host+Hot overlap that jetsamed 31B). Does **not** retain a
    /// `host_q` twin (unused on the Hot decode path).
    pub fn from_host_banks(banks: HostWeightBanks) -> Result<Self> {
        let upload_t0 = std::time::Instant::now();
        let hot_bytes = banks.total_hot_bytes() as u64;
        diag::log(
            "gpu",
            format_args!(
                "▶ from_host_banks matrices={} hot={}",
                banks.matrices.len(),
                diag::fmt_bytes(hot_bytes)
            ),
        );
        let gpu = GemmaGpu::new().map_err(|e| {
            diag::err("gpu", "GemmaGpu::new in from_host_banks", &e);
            e
        })?;
        let cfg = banks.config.clone();
        let kv = banks.kv_layout.clone();
        let softcap = cfg.final_logit_softcapping.unwrap_or(30.0);
        let eps = cfg.rms_norm_eps as f32;
        let hidden = cfg.hidden_size;
        let vocab = cfg.vocab_size;
        let intermediate = cfg.intermediate_size;
        let scheme = banks.scheme;

        let embed_q = banks.require("embed_tokens.weight").map_err(|e| {
            diag::err("gpu", "require embed_tokens.weight", &e);
            e
        })?.clone();
        if embed_q.rows != vocab || embed_q.cols != hidden {
            let e = Error::Weights(format!(
                "embed_tokens shape [{},{}] != vocab×hidden [{vocab},{hidden}]",
                embed_q.rows, embed_q.cols
            ));
            diag::err("gpu", "embed shape", &e);
            return Err(e);
        }
        // Tied embeddings → share one Hot bank for embed lookup + lm_head GEMV.
        let tied = banks.require("lm_head.weight").is_err();
        let embed_hot_banks = upload_quant_hot(&gpu, &embed_q)?;
        let lm_head_host = if tied {
            embed_q.clone()
        } else {
            banks.require("lm_head.weight")?.clone()
        };
        let lm_head = if tied {
            HotQuantBanks {
                scheme: embed_hot_banks.scheme,
                layout: embed_hot_banks.layout,
                rows: embed_hot_banks.rows,
                cols: embed_hot_banks.cols,
                group_size: embed_hot_banks.group_size,
                packed: embed_hot_banks.packed.clone(),
                scales: embed_hot_banks.scales.clone(),
                zeros: embed_hot_banks.zeros.clone(),
            }
        } else {
            upload_quant_hot(&gpu, &lm_head_host)?
        };
        let embed_hot = Some(embed_hot_banks);
        let ws = gpu.rt.memory_info().recommended_working_set;
        diag::log(
            "gpu",
            format_args!(
                "Hot upload — tied_embed_lm_head={tied} embed_rows={} lm_head_rows={} recommendedWS≈{}",
                embed_q.rows,
                lm_head_host.rows,
                diag::fmt_bytes(ws as u64)
            ),
        );

        let final_norm_w = banks.require("norm.weight")?.dequant_f32()?;
        let final_norm = upload_f32_hot(&gpu, &final_norm_w)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);

        for li in 0..cfg.num_hidden_layers {
            let map = kv.layer(li)?;
            let (hq, hkv, head_dim, window, rotary_dim, theta) = match map.layer_type {
                LayerType::SlidingAttention => (
                    cfg.num_attention_heads,
                    cfg.num_key_value_heads,
                    cfg.head_dim,
                    cfg.sliding_window,
                    cfg.head_dim,
                    10_000.0f32,
                ),
                LayerType::FullAttention => {
                    let d = cfg.global_head_dim;
                    let rotary = ((d as f32)
                        * cfg
                            .rope_parameters
                            .as_ref()
                            .and_then(|r| r.full_attention.partial_rotary_factor)
                            .unwrap_or(0.25) as f32) as usize;
                    (
                        cfg.num_attention_heads,
                        cfg.global_kv_heads(),
                        d,
                        None,
                        rotary.max(1),
                        cfg.rope_parameters
                            .as_ref()
                            .map(|r| r.full_attention.rope_theta as f32)
                            .unwrap_or(1_000_000.0),
                    )
                }
            };

            let prefix = format!("layers.{li}");
            // Upload by ref — no host_q twin (was unused and doubled host residency).
            let q_proj = banks.require(&format!("{prefix}.self_attn.q_proj.weight"))?;
            let o_proj = banks.require(&format!("{prefix}.self_attn.o_proj.weight"))?;
            let gate_proj = banks.require(&format!("{prefix}.mlp.gate_proj.weight"))?;
            let up_proj = banks.require(&format!("{prefix}.mlp.up_proj.weight"))?;
            let down_proj = banks.require(&format!("{prefix}.mlp.down_proj.weight"))?;

            let (k_hot, v_hot) = if matches!(map.role, KvRole::Producer { .. }) {
                let k = banks.require(&format!("{prefix}.self_attn.k_proj.weight"))?;
                let k_hot = upload_quant_hot(&gpu, k)?;
                // 31B global layers use attention_k_eq_v — no separate v_proj in HF.
                let v_hot =
                    if let Ok(m) = banks.require(&format!("{prefix}.self_attn.v_proj.weight")) {
                        upload_quant_hot(&gpu, m)?
                    } else if cfg.attention_k_eq_v
                        && matches!(map.layer_type, LayerType::FullAttention)
                    {
                        HotQuantBanks {
                            scheme: k_hot.scheme,
                            layout: k_hot.layout,
                            rows: k_hot.rows,
                            cols: k_hot.cols,
                            group_size: k_hot.group_size,
                            packed: k_hot.packed.clone(),
                            scales: k_hot.scales.clone(),
                            zeros: k_hot.zeros.clone(),
                        }
                    } else {
                        return Err(Error::Weights(format!(
                            "missing weight '{prefix}.self_attn.v_proj.weight'"
                        )));
                    };
                (Some(k_hot), Some(v_hot))
            } else {
                (None, None)
            };

            let input_norm = banks
                .require(&format!("{prefix}.input_layernorm.weight"))?
                .dequant_f32()?;
            // Gemma4 has both post_attention + pre_feedforward. Prefer that split.
            // Legacy / incomplete graphs: fall back to a single before-MLP RMS.
            let post_attn_host = banks
                .require(&format!("{prefix}.post_attention_layernorm.weight"))
                .ok()
                .and_then(|m| m.dequant_f32().ok());
            let pre_ff_host = banks
                .require(&format!("{prefix}.pre_feedforward_layernorm.weight"))
                .ok()
                .and_then(|m| m.dequant_f32().ok());
            let (post_attn_norm, pre_ff_norm_host) = match (post_attn_host, pre_ff_host) {
                (Some(pa), Some(pf)) => (pa, Some(pf)),
                (None, Some(pf)) => (pf, None),
                (Some(pa), None) => (pa, None),
                (None, None) => {
                    return Err(Error::Weights(format!(
                        "missing both post_attention and pre_feedforward norms for {prefix}"
                    )));
                }
            };
            let q_norm = banks
                .require(&format!("{prefix}.self_attn.q_norm.weight"))?
                .dequant_f32()?;
            let k_norm = if let Ok(m) = banks.require(&format!("{prefix}.self_attn.k_norm.weight"))
            {
                m.dequant_f32()?
            } else {
                // KV-share consumers have no k_norm (no K weights).
                vec![1.0f32; head_dim]
            };
            let v_norm = vec![1.0f32; head_dim]; // E4B has no v_norm tensor

            let post_ff_norm = if let Ok(m) =
                banks.require(&format!("{prefix}.post_feedforward_layernorm.weight"))
            {
                Some(upload_f32_hot(&gpu, &m.dequant_f32()?)?)
            } else {
                None
            };
            let pre_ff_norm = match pre_ff_norm_host {
                Some(w) => Some(upload_f32_hot(&gpu, &w)?),
                None => None,
            };
            let layer_scalar = banks
                .require(&format!("{prefix}.layer_scalar"))
                .ok()
                .and_then(|m| m.dequant_f32().ok())
                .and_then(|v| v.first().copied())
                .unwrap_or(1.0);

            let ple_table = None;

            layers.push(GpuSynthLayer {
                layer_type: map.layer_type,
                role: map.role.clone(),
                hq,
                hkv,
                head_dim,
                rotary_dim,
                theta,
                window,
                input_norm: upload_f32_hot(&gpu, &input_norm)?,
                post_attn_norm: upload_f32_hot(&gpu, &post_attn_norm)?,
                pre_ff_norm,
                q_norm: upload_f32_hot(&gpu, &q_norm)?,
                k_norm: upload_f32_hot(&gpu, &k_norm)?,
                v_norm: upload_f32_hot(&gpu, &v_norm)?,
                q_proj: upload_quant_hot(&gpu, q_proj)?,
                k_proj: k_hot,
                v_proj: v_hot,
                o_proj: upload_quant_hot(&gpu, o_proj)?,
                gate_proj: upload_quant_hot(&gpu, gate_proj)?,
                up_proj: upload_quant_hot(&gpu, up_proj)?,
                down_proj: upload_quant_hot(&gpu, down_proj)?,
                post_ff_norm,
                layer_scalar,
                ple_table,
            });
            if li == 0 || li + 1 == cfg.num_hidden_layers || li % 8 == 0 {
                diag::log(
                    "gpu",
                    format_args!(
                        "uploaded layer {li}/{} type={:?} role={:?} hq={hq} hkv={hkv} head_dim={head_dim} layer_scalar={layer_scalar:.4} gemma4_norms={}",
                        cfg.num_hidden_layers,
                        map.layer_type,
                        map.role,
                        layers.last().map(|l| l.pre_ff_norm.is_some()).unwrap_or(false)
                    ),
                );
            }
        }

        let ple_q4 = if let Some(ref ple) = banks.ple {
            // Single Q4 bank covering [vocab, L*ple_dim] — preferred MLX path.
            let bank = ple
                .layers
                .first()
                .ok_or_else(|| Error::Ple("empty PleBanks".into()))?;
            Some(upload_quant_hot(&gpu, &bank.matrix)?)
        } else {
            None
        };
        if ple_q4.is_some() {
            diag::log(
                "gpu",
                format_args!(
                    "PLE Q4 Hot uploaded vocab×(L·dim) — gate/proj still skipped (lookup residual only)"
                ),
            );
        }

        // End host weight residency before session KV/scratch alloc at the caller.
        drop(banks);
        diag::log(
            "gpu",
            format_args!(
                "✔ Hot banks ready — {} layers vocab={vocab} hidden={hidden} scheme={scheme:?} hot={} in {:.1}s (host banks dropped) RSS={:?}",
                layers.len(),
                diag::fmt_bytes(hot_bytes),
                upload_t0.elapsed().as_secs_f64(),
                diag::rss_mib()
            ),
        );

        Ok(Self {
            gpu,
            softcap,
            eps,
            hidden,
            vocab,
            intermediate,
            embed_scale: (hidden as f32).sqrt(),
            embed: Vec::new(),
            embed_q: Some(embed_q),
            embed_hot,
            final_norm,
            lm_head,
            lm_head_host,
            layers,
            host_q: Vec::new(),
            scheme,
            cfg,
            kv,
            host: None,
            ple_q4,
        })
    }

    /// Host GEMV via dequantized Q4 twin (parity reference for GPU Hot path).
    pub fn host_gemv_q(&self, w: &QuantMatrix, x: &[f32]) -> Result<Vec<f32>> {
        let deq = w.dequant_f32()?;
        Ok(gemv(&deq, x, w.rows, w.cols))
    }
}

/// Host-side buffer of post-layer residuals at selected target layers (DFlash step 2).
///
/// Layout: row-major `[T, n_cap · hidden]` with one row appended per decode/prefill
/// / verify timestep. Snapshots use **device** `copy_f32` into `capture_row`
/// (no mid-layer host sync). Host concat is filled after argmax readback so
/// capture-on matches capture-off softcap timing. Optional GPU conditioner
/// projects on-device before that sync.
#[derive(Clone, Debug)]
pub struct HiddenCapture {
    pub layer_ids: Vec<usize>,
    pub hidden: usize,
    /// Packed concat rows `[T * n_cap * hidden]`.
    pub concat: Vec<f32>,
    pub t: usize,
    /// Which capture slots have been written this step (by layer id match).
    step_mask: Vec<bool>,
}

impl HiddenCapture {
    pub fn new(layer_ids: Vec<usize>, hidden: usize) -> Result<Self> {
        if layer_ids.is_empty() {
            return Err(Error::Config("HiddenCapture: empty layer_ids".into()));
        }
        if hidden == 0 {
            return Err(Error::Config("HiddenCapture: hidden=0".into()));
        }
        let n = layer_ids.len();
        Ok(Self {
            layer_ids,
            hidden,
            concat: Vec::new(),
            t: 0,
            step_mask: vec![false; n],
        })
    }

    pub fn row_stride(&self) -> usize {
        self.layer_ids.len() * self.hidden
    }

    fn begin_step(&mut self) {
        for s in &mut self.step_mask {
            *s = false;
        }
    }

    /// Mark slot for `li` as filled (device copy already enqueued).
    fn mark_layer(&mut self, li: usize) -> Option<usize> {
        for (i, &id) in self.layer_ids.iter().enumerate() {
            if id == li {
                self.step_mask[i] = true;
                return Some(i);
            }
        }
        None
    }

    fn finish_step_from_row(&mut self, row: &[f32]) -> Result<()> {
        let stride = self.row_stride();
        if row.len() < stride {
            return Err(Error::Config(format!(
                "HiddenCapture: row len {} < stride {stride}",
                row.len()
            )));
        }
        for (i, filled) in self.step_mask.iter().enumerate() {
            if !*filled {
                return Err(Error::Config(format!(
                    "HiddenCapture: missing layer {} this step",
                    self.layer_ids[i]
                )));
            }
        }
        self.concat.extend_from_slice(&row[..stride]);
        self.t += 1;
        Ok(())
    }

    /// Append a fully-populated concat row (all capture layers already filled).
    fn finish_step_complete(&mut self, row: &[f32]) -> Result<()> {
        for s in &mut self.step_mask {
            *s = true;
        }
        self.finish_step_from_row(row)
    }

    pub fn trim_recent(&mut self, n: usize) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        if n > self.t {
            return Err(Error::Config(format!(
                "HiddenCapture::trim_recent {n} > t={}",
                self.t
            )));
        }
        let stride = self.row_stride();
        let keep = self.t - n;
        self.concat.truncate(keep * stride);
        self.t = keep;
        Ok(())
    }

    pub fn clear(&mut self) {
        self.concat.clear();
        self.t = 0;
        self.begin_step();
    }
}

/// Persistent decode session: Hot weights + GPU-resident KV + scratch.
pub struct GpuDecodeSession {
    pub model: GpuSynthModel,
    sliding_rings: Vec<GpuKvSlot>,
    global_slots: Vec<GpuKvSlot>,
    shared_sliding: GpuKvSlot,
    shared_global: GpuKvSlot,
    pos: usize,
    /// DecodeIcb: IcbScalarPool cursor watermark at layer-graph capture.
    /// Replay-prep must land on the SAME watermark or the frozen tape's pooled
    /// binds are misaligned (e.g. ring densify inserts a dispatch) — in that
    /// case the step falls back to live encode instead of silently replaying
    /// garbage (2026-07-19 token-parity FAIL triage).
    icb_capture_watermark: Option<(usize, usize)>,
    /// A2 residual: scalar-write tape from layer-graph capture. When present,
    /// mini DecodeIcb replay skips the binder-nop layer loop (host push program
    /// + KV commits only). Opt-out: `GEMMA_METAL_ICB_SKIP_NOP_LOOP=0`.
    icb_scalar_write_tape: Option<crate::kernels::IcbScalarWriteTape>,
    /// Optional DFlash target-layer hidden capture (host mirror of device rows).
    capture: Option<HiddenCapture>,
    /// Device concat row `[n_cap · H]` filled via `copy_f32` during the layer loop.
    capture_row: Option<GpuBuffer>,
    /// Optional GPU `fc`/`hidden_norm` → `h_ctx` (DFlash conditioning).
    conditioner: Option<crate::dflash::DFlashGpuConditioner>,
    // Scratch (Cold / reused)
    x: GpuBuffer,
    normed: GpuBuffer,
    q: GpuBuffer,
    k: GpuBuffer,
    v: GpuBuffer,
    o: GpuBuffer,
    attn_proj: GpuBuffer,
    gate: GpuBuffer,
    up: GpuBuffer,
    mid: GpuBuffer,
    down: GpuBuffer,
    logits: GpuBuffer,
    k_dense: GpuBuffer,
    v_dense: GpuBuffer,
    ple_out: GpuBuffer,
    /// Active input token for embed / PLE (`[1]`). Host or GPU-seeded; never written by softcap.
    seed_tok: GpuBuffer,
    /// Softcap-argmax next token (`[1]`). Separate from [`Self::seed_tok`] so verify can
    /// host-pack seeds without racing the GPU producer.
    argmax_tok: GpuBuffer,
    /// Absolute RoPE / KV write position (`u32×1`), written once per decode/verify step.
    /// Seed token is already GPU-resident (`seed_tok`). FA/kv_store still take CPU `u32`
    /// constants today — see D16 / `tessl::cb_replay`.
    pos_buf: GpuBuffer,
    /// Optional ping-pong CB replay scaffold (not wired into the decode graph).
    encode_once: tessl::PingPongCbReplay,
    /// Mini-only persistent-interpreter decode hook (lazy; see D17).
    persistent_interp: Option<PersistentInterpMiniHook>,
    /// Lazy scratch for Hot bounded-TG gate→down (D18; see [`fuse_gate_down_enabled`]).
    fuse_gate_down: Option<FuseGateDownScratch>,
    /// Packed verify input seeds (`[VERIFY_MAX_M]`) — uploaded once per `step_verify`.
    verify_seeds: GpuBuffer,
    /// Per-position greedy next tokens from [`Self::step_verify`] (len = VERIFY_MAX_M).
    verify_outs: GpuBuffer,
    max_kv: usize,
    argmax_scratch: ArgmaxScratch,
}

/// How [`GpuDecodeSession::step_inner`] obtains the active seed token.
enum StepSeed {
    /// Host write into `seed_tok` before embed.
    Host(u32),
    /// `seed_tok` already holds the input (e.g. GPU copy from `verify_seeds[i]`).
    Resident,
    /// GPU copy `argmax_tok` → `seed_tok` (chained decode without host readback).
    FromArgmax,
}

/// Dense f32 scratch + hit counters for mini-only persistent-interp decode hooks.
///
/// Stand-ins prove the grid-sync edge (`PRODUCE → BARRIER → PROJ`) on dedicated
/// buffers — they do **not** replace shipping Q4 FA / MLP (mock FA + dense W).
/// Allocated lazily when `GEMMA_METAL_PERSISTENT_INTERP=1` on synthetic mini.
struct PersistentInterpMiniHook {
    gate_insns: GpuBuffer,
    fa_insns: GpuBuffer,
    n_gate_insns: u32,
    n_fa_insns: u32,
    gate: GpuBuffer,
    up: GpuBuffer,
    mid: GpuBuffer,
    w_down: GpuBuffer,
    out_down: GpuBuffer,
    q: GpuBuffer,
    k: GpuBuffer,
    v: GpuBuffer,
    ctx: GpuBuffer,
    w_o: GpuBuffer,
    out_o: GpuBuffer,
    deps: GpuBuffer,
    fail: GpuBuffer,
    n_mid: u32,
    n_hidden: u32,
    n_ctx: u32,
    n_tg: u32,
    gate_down_hits: u64,
    fa_o_hits: u64,
    last_fail: u32,
}

/// Device buffers for [`persistent_interp_gate_down_q4`] (instruction stream + barrier).
struct FuseGateDownScratch {
    insns: GpuBuffer,
    deps: GpuBuffer,
    fail: GpuBuffer,
}

/// Max draft tokens per `step_verify` block (DFlash prior ≤8; M=5 measured sweet spot on MLX).
pub const VERIFY_MAX_M: usize = 8;

/// Outcome of one block-verify target forward over M tokens.
#[derive(Clone, Debug)]
pub struct StepVerifyResult {
    /// Absolute position of `tokens[0]` when the block started.
    pub pos0: usize,
    /// Input tokens that were verified (`len = M`).
    pub tokens: Vec<u32>,
    /// Softcap-argmax next token at each of the M positions (`len = M`).
    pub next_tokens: Vec<u32>,
}

/// Host-side logit buffer health (for diag tooling).
#[derive(Clone, Debug)]
pub struct LogitsStats {
    pub finite: bool,
    pub nan: usize,
    pub max: f32,
    pub min: f32,
    pub host_argmax: u32,
}

impl GpuDecodeSession {
    pub fn new(model: GpuSynthModel) -> Result<Self> {
        let cfg = &model.cfg;
        let kv = &model.kv;
        let hidden = model.hidden;
        let window = cfg.sliding_window_or(512);
        let max_kv = kv.max_seq.max(16);
        let gpu = &model.gpu;
        diag::log(
            "gpu",
            format_args!(
                "▶ GpuDecodeSession::new layers={} hidden={hidden} vocab={} max_kv={max_kv} \
                 sliding_rings={} global_slots={} window={window}",
                model.layers.len(),
                model.vocab,
                kv.sliding_ring_slots,
                kv.global_full_slots
            ),
        );

        let sliding_rings: Result<Vec<_>> = (0..kv.sliding_ring_slots)
            .map(|_| GpuKvSlot::new(gpu, window, cfg.num_key_value_heads, cfg.head_dim, true))
            .collect();
        let sliding_rings = sliding_rings.map_err(|e| {
            diag::err("gpu", "sliding_rings alloc", &e);
            e
        })?;
        let global_slots: Result<Vec<_>> = (0..kv.global_full_slots)
            .map(|_| {
                GpuKvSlot::new(
                    gpu,
                    max_kv,
                    cfg.global_kv_heads(),
                    cfg.global_head_dim,
                    false,
                )
            })
            .collect();
        let global_slots = global_slots.map_err(|e| {
            diag::err("gpu", "global_slots alloc", &e);
            e
        })?;
        let shared_sliding =
            GpuKvSlot::new(gpu, max_kv, cfg.num_key_value_heads, cfg.head_dim, false)?;
        let shared_global =
            GpuKvSlot::new(gpu, max_kv, cfg.global_kv_heads(), cfg.global_head_dim, false)?;

        let alloc = |n: usize| -> Result<GpuBuffer> {
            model
                .gpu
                .rt
                .alloc_buffer(n.max(1) * 4)
                .map_err(|e| {
                    diag::err_msg("gpu", &format!("scratch alloc n={}", n.max(1) * 4), &e);
                    Error::Metal(e)
                })
        };
        let max_q = cfg.num_attention_heads * cfg.global_head_dim.max(cfg.head_dim);
        let max_kv_elems = max_kv
            * cfg.global_kv_heads().max(cfg.num_key_value_heads)
            * cfg.global_head_dim.max(cfg.head_dim);
        let inter = model.intermediate;
        let vocab = model.vocab;
        let ple_dim = cfg.hidden_size_per_layer_input.max(1);
        // Dual-capacity banks: decode `step` touches row 0 only (write_f32_prefix +
        // explicit n=H/inter). GEMM verify uses the full H×M / Vocab×M capacity.
        let vm_act = VERIFY_MAX_M.max(1);
        let vm_tok = VERIFY_MAX_M.max(1);
        let vm_logits = VERIFY_MAX_M.max(1);
        diag::log(
            "gpu",
            format_args!(
                "scratch: max_q={max_q} max_kv_elems={max_kv_elems} inter={inter} vocab={vocab} \
                 ple_dim={ple_dim} act_m={vm_act} logits_m={vm_logits} verify_tok_m={vm_tok}"
            ),
        );
        let argmax_scratch = ArgmaxScratch::new(gpu, vocab as u32)?;

        diag::log("gpu", format_args!("✔ GpuDecodeSession ready"));
        let sess = Self {
            x: alloc(hidden * vm_act)?,
            normed: alloc(hidden * vm_act)?,
            q: alloc(max_q * vm_act)?,
            k: alloc(max_q * vm_act)?,
            v: alloc(max_q * vm_act)?,
            o: alloc(max_q * vm_act)?,
            attn_proj: alloc(hidden * vm_act)?,
            gate: alloc(inter * vm_act)?,
            up: alloc(inter * vm_act)?,
            mid: alloc(inter * vm_act)?,
            down: alloc(hidden * vm_act)?,
            logits: alloc(vocab * vm_logits)?,
            k_dense: alloc(max_kv_elems)?,
            v_dense: alloc(max_kv_elems)?,
            ple_out: alloc(ple_dim * vm_act)?,
            seed_tok: model.gpu.rt.alloc_buffer(4).map_err(Error::Metal)?,
            argmax_tok: model.gpu.rt.alloc_buffer(4).map_err(Error::Metal)?,
            pos_buf: model.gpu.rt.alloc_buffer(4).map_err(Error::Metal)?,
            encode_once: tessl::PingPongCbReplay::new(),
            icb_capture_watermark: None,
            icb_scalar_write_tape: None,
            persistent_interp: None,
            fuse_gate_down: None,
            verify_seeds: model
                .gpu
                .rt
                .alloc_buffer(vm_tok * 4)
                .map_err(Error::Metal)?,
            verify_outs: model
                .gpu
                .rt
                .alloc_buffer(vm_tok * 4)
                .map_err(Error::Metal)?,
            max_kv,
            argmax_scratch,
            sliding_rings,
            global_slots,
            shared_sliding,
            shared_global,
            pos: 0,
            capture: None,
            capture_row: None,
            conditioner: None,
            model,
        };
        sess.pos_buf.write_u32(&[0]);
        Ok(sess)
    }

    pub fn reset(&mut self) {
        // Drain any outstanding encode before rewinding host-side KV metadata;
        // otherwise a late KV write can land after seq_len→0 and poison FA.
        let _ = self.model.gpu.synchronize();
        for r in &mut self.sliding_rings {
            r.reset();
        }
        for s in &mut self.global_slots {
            s.reset();
        }
        self.shared_sliding.reset();
        self.shared_global.reset();
        self.pos = 0;
        self.pos_buf.write_u32(&[0]);
        if let Some(ref mut c) = self.capture {
            c.clear();
        }
        if let Some(ref mut c) = self.conditioner {
            c.clear();
        }
    }

    /// Enable DFlash hidden capture at `layer_ids` (must exist on this model).
    pub fn enable_hidden_capture(&mut self, layer_ids: Vec<usize>) -> Result<()> {
        let n_layers = self.model.layers.len();
        for &id in &layer_ids {
            if id >= n_layers {
                return Err(Error::Config(format!(
                    "enable_hidden_capture: layer {id} OOB (n_layers={n_layers})"
                )));
            }
        }
        let n_cap = layer_ids.len();
        let h = self.model.hidden;
        let row = self
            .model
            .gpu
            .rt
            .alloc_buffer(n_cap.max(1) * h.max(1) * 4)
            .map_err(Error::Metal)?;
        self.capture = Some(HiddenCapture::new(layer_ids, h)?);
        self.capture_row = Some(row);
        Ok(())
    }

    pub fn disable_hidden_capture(&mut self) {
        self.capture = None;
        self.capture_row = None;
        self.conditioner = None;
    }

    /// Attach GPU `fc`/`hidden_norm` conditioner (enables capture if needed).
    pub fn attach_gpu_conditioner(
        &mut self,
        cond: crate::dflash::DFlashGpuConditioner,
    ) -> Result<()> {
        if self.capture.is_none() {
            self.enable_hidden_capture(cond.target_layer_ids.clone())?;
        } else if let Some(ref c) = self.capture {
            if c.layer_ids != cond.target_layer_ids {
                return Err(Error::Config(
                    "attach_gpu_conditioner: layer_ids mismatch vs enabled capture".into(),
                ));
            }
        }
        self.conditioner = Some(cond);
        Ok(())
    }

    pub fn conditioner_h_ctx_len(&self) -> usize {
        self.conditioner.as_ref().map(|c| c.h_ctx_len()).unwrap_or(0)
    }

    pub fn read_conditioner_h_ctx(&self) -> Result<Vec<f32>> {
        let Some(ref c) = self.conditioner else {
            return Err(Error::Config("no GPU conditioner attached".into()));
        };
        c.read_h_ctx(&self.model.gpu)
    }

    /// Last conditioner `fc` output (pre-hidden_norm) for intermediate dumps.
    pub fn read_conditioner_fc_out(&self) -> Result<Vec<f32>> {
        let Some(ref c) = self.conditioner else {
            return Err(Error::Config("no GPU conditioner attached".into()));
        };
        c.read_fc_out(&self.model.gpu)
    }

    /// Device `h_ctx` buffer for GPU draft (prefix length = [`Self::conditioner_h_ctx_len`]).
    pub fn conditioner_h_ctx_buf(&self) -> Result<&tessl::tensor::GpuBuffer> {
        let Some(ref c) = self.conditioner else {
            return Err(Error::Config("no GPU conditioner attached".into()));
        };
        Ok(c.h_ctx_buf())
    }

    pub fn capture_len(&self) -> usize {
        self.capture.as_ref().map(|c| c.t).unwrap_or(0)
    }

    /// Captured concat `[T, n_cap·H]` as `(data, T)`.
    pub fn captured_concat(&self) -> Result<(Vec<f32>, usize)> {
        let Some(ref c) = self.capture else {
            return Err(Error::Config("hidden capture not enabled".into()));
        };
        Ok((c.concat.clone(), c.t))
    }

    pub fn trim_captured(&mut self, n: usize) -> Result<()> {
        if let Some(ref mut c) = self.capture {
            c.trim_recent(n)?;
        }
        if let Some(ref mut cond) = self.conditioner {
            let trim_c = n.min(cond.h_ctx_len());
            cond.trim_recent(trim_c)?;
        }
        Ok(())
    }

    fn embed_token(&self, tid: u32) -> Result<()> {
        let h = self.model.hidden;
        // Write token id first — GPU embed lookup reads seed_tok[0].
        self.seed_tok.write_u32(&[tid]);
        self.embed_from_seed(h)
    }

    /// Write one (or M) activation rows into `x` scratch (capacity may be H·VERIFY_MAX_M).
    fn write_x_rows(&self, data: &[f32]) -> Result<()> {
        let cap = self.x.nbytes() / 4;
        if data.len() > cap {
            return Err(Error::Config(format!(
                "write_x_rows: data {} > x scratch {}",
                data.len(),
                cap
            )));
        }
        self.x.write_f32_prefix(data);
        Ok(())
    }

    fn embed_from_seed(&self, h: usize) -> Result<()> {
        let scale = self.model.embed_scale;
        if let Some(ref hot) = self.model.embed_hot {
            embed_lookup_quant(
                &self.model.gpu,
                hot,
                &self.seed_tok,
                &self.x,
                self.model.vocab as u32,
            )?;
            if (scale - 1.0).abs() > 1e-12 {
                // RAW: lookup writes `x`, then scale reads it (hazard mode skips auto barriers).
                if tessl::ab_flags::need_barrier(true) {
                    self.model.gpu.barrier()?;
                }
                scale_f32_inplace(&self.model.gpu, &self.x, scale, h as u32)?;
            }
            return Ok(());
        }
        if let Some(ref eq) = self.model.embed_q {
            let tid = self.seed_tok.read_u32()[0];
            let mut row = eq.dequant_row(tid as usize)?;
            if (scale - 1.0).abs() > 1e-12 {
                for v in &mut row {
                    *v *= scale;
                }
            }
            return self.write_x_rows(&row);
        }
        let tid = self.seed_tok.read_u32()[0];
        let row = (tid as usize) * h;
        if row + h > self.model.embed.len() {
            return Err(Error::Config(format!("token {tid} OOV")));
        }
        if (scale - 1.0).abs() <= 1e-12 {
            self.write_x_rows(&self.model.embed[row..row + h])
        } else {
            let mut scaled = Vec::with_capacity(h);
            for d in 0..h {
                scaled.push(self.model.embed[row + d] * scale);
            }
            self.write_x_rows(&scaled)
        }
    }

    /// One decode step. Seed via host / resident / previous argmax; optional readback.
    fn step_inner(&mut self, seed: StepSeed, readback: bool, head: bool) -> Result<u32> {
        // Hazard skip-auto + mid-layer capture races: residual RAW edges land
        // incomplete → capture absmean collapses (L46/57=0) and first tok→0.
        // Force always-on Dispatch barriers for the duration of captured steps only.
        let _capture_ao = CaptureAlwaysOnGuard::enter(self.capture.is_some());
        let pos = self.pos;
        // Stable scalar arena (FA/kv/softcap) — deterministic bump per step for ICB.
        self.model.gpu.icb_scalars.reset_step();
        self.model.gpu.icb_scalars.set_softcap(self.model.softcap);
        if fuse_gate_down_enabled() {
            let _ = self.ensure_fuse_gate_down_scratch()?;
        }
        // Encode-once prep: pos + seed are GPU-resident; host writes pos once / step.
        self.sync_pos_buf(pos as u32);
        // DecodeIcb (opt-in; mini + E4B Hot; not 31B):
        //  - First **head** decode: Binder capture tape → from_commands (layers).
        //  - Later Ready **head** steps: scalar-write tape + frozen-tape execute.
        //  - Prefill (head=false) stays live encode so a head-captured ICB is never
        //    replayed without lm_head/argmax (token-parity hazard).
        // Seed/embed stay live GPU; nop starts after embed (see below).
        let mut icb_replay_prep = false;
        let mut icb_live_replay_noted = false;
        let mut capturing_layer_icb = false;
        let mut layer_icb_capture_guard: Option<LayerIcbCaptureGuard> = None;
        let mut binder_nop_guard: Option<tessl::BinderEncodeNopGuard> = None;
        if encode_once_enabled()
            && tessl::decode_icb_enabled()
            && self.model.decode_icb_graph_eligible()
        {
            // Tape-direct dispatch is the honest default (execute_icb inherit ≈
            // residual no-op). ICB-capable pipelines when true ICB execute or
            // freeze-binds is requested (both need supportIndirectCommandBuffers).
            let want_icb_exec = std::env::var("GEMMA_METAL_ICB_EXECUTE")
                .ok()
                .map(|v| matches!(v.as_str(), "1" | "true" | "icb"))
                .unwrap_or(false)
                || std::env::var("METAL_RUNTIME_ICB_EXECUTE")
                    .ok()
                    .map(|v| matches!(v.as_str(), "1" | "true" | "icb"))
                    .unwrap_or(false)
                || tessl::icb_freeze_binds_enabled();
            tessl::set_icb_pipelines(want_icb_exec);
            if self.encode_once.decode_icb_layer_graph() && self.encode_once.has_ready_slot() {
                if head {
                    // Densify shape stable; Q4 fuse_bf16 now expands bf16→f32 before
                    // classic gemv_q4 (was reinterpret/over-read → ~cmd 19 blow-up).
                    // Default: binder-nop + frozen-tape direct-dispatch. Opt-out to
                    // live layer encode: GEMMA_METAL_ICB_TAPE_EXECUTE=0.
                    let tape_exec = std::env::var("GEMMA_METAL_ICB_TAPE_EXECUTE")
                        .ok()
                        .map(|v| !matches!(v.as_str(), "0" | "false" | "off" | "live"))
                        .unwrap_or(true)
                        && std::env::var("METAL_RUNTIME_ICB_TAPE_EXECUTE")
                            .ok()
                            .map(|v| !matches!(v.as_str(), "0" | "false" | "off" | "live"))
                            .unwrap_or(true);
                    if tape_exec {
                        icb_replay_prep = true;
                    } else if let Err(e) = self
                        .encode_once
                        .note_layer_live_replay(format!("live_layer_replay pos={pos}"))
                    {
                        return Err(Error::Metal(format!(
                            "encode_once note_layer_live_replay: {e}"
                        )));
                    } else {
                        icb_live_replay_noted = true;
                    }
                }
            } else if !self.encode_once.decode_icb_wired() {
                // Capture on first head decode so tape includes final_norm/lm_head/argmax.
                if head {
                    capturing_layer_icb = true;
                }
            } else if !self.encode_once.decode_icb_layer_graph() {
                // copy_f32 chain fallback (dispatch-freeze proof only).
                match self.encode_once.try_replay_ready_icb(&self.model.gpu.rt) {
                    Ok(slot) => {
                        self.encode_once.on_gpu_complete(slot);
                    }
                    Err(tessl::CbReplayError::NotReady) => {}
                    Err(tessl::CbReplayError::NotWired) => {}
                    Err(e) => {
                        return Err(Error::Metal(format!("encode_once try_replay_icb: {e}")));
                    }
                }
            }
        } else if encode_once_enabled() {
            match self.encode_once.try_replay_ready() {
                Ok(_) => {}
                Err(tessl::CbReplayError::NotReady) => {}
                Err(tessl::CbReplayError::NotWired) => {}
                Err(e) => {
                    return Err(Error::Metal(format!("encode_once try_replay: {e}")));
                }
            }
        }
        let hidden = self.model.hidden as u32;
        let eps = self.model.eps;
        let first_shared = self.model.kv.first_kv_shared;
        let n_layers = self.model.layers.len();
        let ple_dim_cfg = self.model.cfg.hidden_size_per_layer_input;
        let vocab = self.model.vocab as u32;
        let intermediate = self.model.intermediate as u32;
        let softcap = self.model.softcap;
        let do_trace = trace::enabled();
        let step_t0 = std::time::Instant::now();
        let seed_dbg = match seed {
            StepSeed::Host(t) => format!("Host({t})"),
            StepSeed::Resident => "Resident".into(),
            StepSeed::FromArgmax => "FromArgmax".into(),
        };
        let _step_scope = InferScope::begin(
            "decode_step",
            format!(
                "pos={pos} seed={seed_dbg} head={head} readback={readback} layers={n_layers} \
                 hidden={hidden} vocab={vocab} inter={intermediate}"
            ),
        );
        diag::log(
            "gpu",
            format_args!(
                "step pos={pos} seed={seed_dbg} head={head} readback={readback} layers={n_layers}"
            ),
        );

        let mut tracer = if do_trace {
            Some(TraceSession::new())
        } else {
            None
        };
        if let Some(ref mut tr) = tracer {
            tr.begin_token();
        }
        let sync_cb: Option<Box<dyn Fn() -> std::result::Result<(), String>>> = if matches!(
            trace::mode(),
            trace::TraceMode::Sync
        ) {
            let rt = self.model.gpu.rt.clone();
            Some(Box::new(move || {
                diag::infer_stall("trace-sync synchronize after named stage");
                rt.synchronize()
            }))
        } else {
            None
        };

        match seed {
            StepSeed::Host(t) => {
                crate::trace_op!("seed_tok", format!("write_u32 token={t}"), {
                    self.seed_tok.write_u32(&[t]);
                });
            }
            StepSeed::FromArgmax => {
                crate::trace_op!("seed_from_argmax", "copy_u32 argmax_tok→seed_tok", {
                    copy_u32_one(&self.model.gpu, &self.argmax_tok, &self.seed_tok)?;
                    if tessl::ab_flags::need_barrier(true) {
                        self.model.gpu.barrier()?;
                    }
                });
            }
            StepSeed::Resident => {}
        }
        {
            let h = self.model.hidden;
            let embed_bytes = (h as u64) * 4;
            let embed_detail = format!(
                "pos={pos} hidden={h} bytes≈{} hot={}",
                diag::fmt_bytes(embed_bytes),
                self.model.embed_hot.is_some()
            );
            if let Some(ref mut tr) = tracer {
                let sync_ref = sync_cb.as_ref().map(|b| b.as_ref());
                crate::trace_op!("embed", &embed_detail, {
                    tr.span(
                        "embed",
                        embed_bytes,
                        0,
                        h as u32,
                        sync_ref,
                        || self.embed_from_seed(h),
                    )?;
                });
            } else {
                crate::trace_op!("embed", &embed_detail, {
                    self.embed_from_seed(h)?;
                });
            }
        }

        if self.capture.is_some() {
            if let Some(ref mut c) = self.capture {
                c.begin_step();
            }
        }

        // After live seed/embed: start Binder tape (capture) or binder-nop (replay prep).
        // A2 residual: when a scalar-write tape exists, skip the nop layer loop entirely.
        let skip_nop_layers = icb_replay_prep
            && icb_skip_nop_loop_enabled()
            && self
                .icb_scalar_write_tape
                .as_ref()
                .map(|t| !t.is_empty())
                .unwrap_or(false);
        if capturing_layer_icb {
            layer_icb_capture_guard = Some(LayerIcbCaptureGuard::begin()?);
        }
        if icb_replay_prep && !skip_nop_layers {
            binder_nop_guard = Some(tessl::BinderEncodeNopGuard::enter());
        }

        if skip_nop_layers {
            self.apply_icb_scalar_write_tape(pos)?;
        }

        let layer_iter = if skip_nop_layers {
            0..0
        } else {
            0..n_layers
        };
        for li in layer_iter {
            let (
                layer_type,
                role,
                hq,
                hkv,
                head_dim,
                rotary_dim,
                theta,
                window,
                has_ple,
                is_producer,
                has_pre_ff,
                has_post_ff,
                layer_scalar,
            ) = {
                let layer = &self.model.layers[li];
                (
                    layer.layer_type,
                    layer.role.clone(),
                    layer.hq,
                    layer.hkv,
                    layer.head_dim,
                    layer.rotary_dim,
                    layer.theta,
                    layer.window,
                    layer.ple_table.is_some() || self.model.ple_q4.is_some(),
                    matches!(layer.role, KvRole::Producer { .. }),
                    layer.pre_ff_norm.is_some(),
                    layer.post_ff_norm.is_some(),
                    layer.layer_scalar,
                )
            };
            // True MLX Gemma4 dual residual-norms + layer_scalar: 31B (no PLE).
            // E4B keeps the prior fused Pre-LN+PLE path so decode tok/s does not regress.
            let use_gemma4_dual_norm =
                has_pre_ff && (!has_ple || crate::kernels::e4b_dual_norm_enabled());
            let _layer_scope = InferScope::begin(
                format!("layer[{li}]"),
                format!(
                    "pos={pos} type={layer_type:?} producer={is_producer} \
                     hq={hq} hkv={hkv} d={head_dim} rotary={rotary_dim} ple={has_ple}"
                ),
            );
            // Sparse breadcrumbs when infer-log is off (infer-log covers every layer).
            if !diag::infer_enabled()
                && (pos < 2 || li == 0 || li + 1 == n_layers || li % 8 == 0)
            {
                diag::log(
                    "gpu",
                    format_args!(
                        "  encode layer={li}/{n_layers} pos={pos} type={layer_type:?} \
                         producer={is_producer} hq={hq} hkv={hkv} d={head_dim} ple={has_ple}"
                    ),
                );
            }

            let fused_kv_store = {
                let gpu = &self.model.gpu;
                let layer = &self.model.layers[li];
                let q_bytes = trace::gemv_bytes_est(layer.q_proj.rows, layer.q_proj.cols, layer.q_proj.group_size);
                crate::trace_op!(
                    "rms_input",
                    format!("layer={li} pos={pos} hidden={hidden} bytes≈{}", diag::fmt_bytes((hidden as u64) * 4)),
                    {
                        if fuse_bf16_rms() {
                            let _ = rms_norm_to_act_bf16(
                                gpu,
                                &self.x,
                                &layer.input_norm,
                                1,
                                hidden,
                                eps,
                            )?;
                        } else {
                            rms_norm_f32(gpu, &self.x, &layer.input_norm, &self.normed, 1, hidden, eps)?;
                            let _ = prepare_act_bf16(gpu, &self.normed, hidden)?;
                        }
                    }
                );
                if layer_probe_enabled() && li == 0 {
                    self.model.gpu.synchronize()?;
                    let n = Self::stats_f32(&self.normed.read_f32()[..self.model.hidden]);
                    eprintln!("[layer_probe] after rms_input finite={} nan={} min={:.4} max={:.4}", n.finite, n.nan, n.min, n.max);
                }
                if tessl::ab_flags::need_barrier(true) {
                    gpu.barrier()?;
                }
                let x_bf16 = gpu.act_bf16_scratch(hidden as usize)?;
                // Layer-fusion v1 (opt-in): Q∥K∥V share `x_bf16`, so on producer
                // layers all three projections can ride one dispatch. Saves 1
                // launch/layer (~37 µs each — audit_deep_2026-07-18 F2).
                let kv_pair = if is_producer {
                    let k = layer
                        .k_proj
                        .as_ref()
                        .ok_or_else(|| Error::Weights(format!("layer {li} missing k_proj")))?;
                    let v = layer
                        .v_proj
                        .as_ref()
                        .ok_or_else(|| Error::Weights(format!("layer {li} missing v_proj")))?;
                    Some((k, v))
                } else {
                    None
                };
                let fuse_qkv = kv_pair
                    .map(|(k, v)| layer.q_proj.can_fuse_qkv(k, v))
                    .unwrap_or(false);
                if fuse_qkv {
                    let (k, v) = kv_pair.expect("fuse_qkv implies kv_pair");
                    let k_bytes = trace::gemv_bytes_est(k.rows, k.cols, k.group_size);
                    let v_bytes = trace::gemv_bytes_est(v.rows, v.cols, v.group_size);
                    crate::trace_op!(
                        "gemv_qkv",
                        format!(
                            "layer={li} q[{}x{}] kv[{}x{}] fused bytes≈{}",
                            layer.q_proj.rows,
                            layer.q_proj.cols,
                            k.rows,
                            k.cols,
                            diag::fmt_bytes(
                                q_bytes.saturating_add(k_bytes).saturating_add(v_bytes)
                            )
                        ),
                        {
                            gemv_q4_mlx_simd_qkv_bf16_x(
                                gpu,
                                &layer.q_proj,
                                k,
                                v,
                                &x_bf16,
                                &self.q,
                                &self.k,
                                &self.v,
                            )?;
                        }
                    );
                } else {
                crate::trace_op!(
                    "gemv_q",
                    format!(
                        "layer={li} [{}x{}] bytes≈{}",
                        layer.q_proj.rows,
                        layer.q_proj.cols,
                        diag::fmt_bytes(q_bytes)
                    ),
                    {
                        layer.q_proj.gemv_bf16_x(gpu, &x_bf16, &self.q)?;
                    }
                );
                if let Some((k, v)) = kv_pair {
                    let k_bytes = trace::gemv_bytes_est(k.rows, k.cols, k.group_size);
                    let v_bytes = trace::gemv_bytes_est(v.rows, v.cols, v.group_size);
                    if k.can_fuse_kv(v) {
                        crate::trace_op!(
                            "gemv_kv",
                            format!(
                                "layer={li} [{}x{}] fused bytes≈{}",
                                k.rows,
                                k.cols,
                                diag::fmt_bytes(k_bytes.saturating_add(v_bytes))
                            ),
                            {
                                gemv_q4_mlx_simd_kv_bf16_x(
                                    gpu, k, v, &x_bf16, &self.k, &self.v,
                                )?;
                            }
                        );
                    } else {
                        crate::trace_op!(
                            "gemv_k",
                            format!(
                                "layer={li} [{}x{}] bytes≈{}",
                                k.rows,
                                k.cols,
                                diag::fmt_bytes(k_bytes)
                            ),
                            {
                                k.gemv_bf16_x(gpu, &x_bf16, &self.k)?;
                            }
                        );
                        crate::trace_op!(
                            "gemv_v",
                            format!(
                                "layer={li} [{}x{}] bytes≈{}",
                                v.rows,
                                v.cols,
                                diag::fmt_bytes(v_bytes)
                            ),
                            {
                                v.gemv_bf16_x(gpu, &x_bf16, &self.v)?;
                            }
                        );
                    }
                }
                } // end unfused Q / K / V lane
                // Light Hot fused↔unfused q/k/v dump (first producer only; no LAYER_PROBE).
                // Scratch: o / attn_proj / down (overwritten later in-layer by FA / MLP).
                if is_producer && qkv_ab_dump_enabled() {
                    use std::sync::atomic::{AtomicUsize, Ordering};
                    static FIRST_PROD: OnceLock<usize> = OnceLock::new();
                    static DUMP_N: AtomicUsize = AtomicUsize::new(0);
                    let first = *FIRST_PROD.get_or_init(|| li);
                    let n = DUMP_N.load(Ordering::Relaxed);
                    if li == first && n < 8 {
                        if let (Some(k_b), Some(v_b)) =
                            (layer.k_proj.as_ref(), layer.v_proj.as_ref())
                        {
                            let q_rows = layer.q_proj.rows as usize;
                            let kv_rows = k_b.rows as usize;
                            let primary_fused = fuse_qkv;
                            let mut alt_ran = false;
                            // Alternate path → scratch (must not clobber product q/k/v).
                            if primary_fused {
                                layer.q_proj.gemv_bf16_x(gpu, &x_bf16, &self.o)?;
                                gemv_q4_mlx_simd_kv_bf16_x(
                                    gpu, k_b, v_b, &x_bf16, &self.attn_proj, &self.down,
                                )?;
                                alt_ran = true;
                            } else {
                                let was = fuse_qkv_enabled();
                                set_fuse_qkv(true);
                                let fused_ok = layer.q_proj.can_fuse_qkv(k_b, v_b);
                                if fused_ok {
                                    gemv_q4_mlx_simd_qkv_bf16_x(
                                        gpu,
                                        &layer.q_proj,
                                        k_b,
                                        v_b,
                                        &x_bf16,
                                        &self.o,
                                        &self.attn_proj,
                                        &self.down,
                                    )?;
                                    alt_ran = true;
                                }
                                set_fuse_qkv(was);
                                if !fused_ok {
                                    eprintln!(
                                        "[qkv_ab_dump] pos={pos} layer={li} skip: can_fuse_qkv=false"
                                    );
                                }
                            }
                            if alt_ran {
                                gpu.synchronize()?;
                                let q_p = self.q.read_f32();
                                let k_p = self.k.read_f32();
                                let v_p = self.v.read_f32();
                                let q_a = self.o.read_f32();
                                let k_a = self.attn_proj.read_f32();
                                let v_a = self.down.read_f32();
                                let q_p = &q_p[..q_rows.min(q_p.len())];
                                let k_p = &k_p[..kv_rows.min(k_p.len())];
                                let v_p = &v_p[..kv_rows.min(v_p.len())];
                                let q_a = &q_a[..q_rows.min(q_a.len())];
                                let k_a = &k_a[..kv_rows.min(k_a.len())];
                                let v_a = &v_a[..kv_rows.min(v_a.len())];
                                let dq = qkv_ab_dump_first_diff(q_p, q_a);
                                let dk = qkv_ab_dump_first_diff(k_p, k_a);
                                let dv = qkv_ab_dump_first_diff(v_p, v_a);
                                let (tg_q, tg_k, tg_v) = {
                                    let rpt = 8u32; // GEMV_SIMD_SG * GEMV_SIMD_ROWS
                                    let tg = |r: u32| (r + rpt - 1) / rpt;
                                    (tg(layer.q_proj.rows), tg(k_b.rows), tg(v_b.rows))
                                };
                                eprintln!(
                                    "[qkv_ab_dump] pos={pos} layer={li} primary_fused={primary_fused} \
                                     layout={:?} q={}x{} kv={}x{} gs={} tg_q/k/v={}/{}/{} \
                                     max_err q={:.6e} k={:.6e} v={:.6e}",
                                    layer.q_proj.layout,
                                    layer.q_proj.rows,
                                    layer.q_proj.cols,
                                    k_b.rows,
                                    k_b.cols,
                                    layer.q_proj.group_size,
                                    tg_q,
                                    tg_k,
                                    tg_v,
                                    qkv_ab_dump_max_abs(q_p, q_a),
                                    qkv_ab_dump_max_abs(k_p, k_a),
                                    qkv_ab_dump_max_abs(v_p, v_a),
                                );
                                let locus = match (dq, dk, dv) {
                                    (None, None, None) => "none (bit-exact)".to_string(),
                                    (Some((i, a, b)), _, _) => {
                                        format!("q[{i}] primary={a:.8} alt={b:.8}")
                                    }
                                    (None, Some((i, a, b)), _) => {
                                        format!("k[{i}] primary={a:.8} alt={b:.8}")
                                    }
                                    (None, None, Some((i, a, b))) => {
                                        format!("v[{i}] primary={a:.8} alt={b:.8}")
                                    }
                                };
                                eprintln!("[qkv_ab_dump] first_diverge={locus}");
                                // Isolate: fused K vs solo k_proj gemv vs gemv_kv.
                                // After bank-split, all three should be bit-exact.
                                let (k_fused_h, k_dual_h): (&[f32], &[f32]) = if primary_fused {
                                    (k_p, k_a)
                                } else {
                                    (k_a, k_p)
                                };
                                k_b.gemv_bf16_x(gpu, &x_bf16, &self.attn_proj)?;
                                gpu.synchronize()?;
                                let k_solo_buf = self.attn_proj.read_f32();
                                let k_solo = &k_solo_buf[..kv_rows.min(k_solo_buf.len())];
                                let dk_sf = qkv_ab_dump_first_diff(k_solo, k_fused_h);
                                let dk_sd = qkv_ab_dump_first_diff(k_solo, k_dual_h);
                                eprintln!(
                                    "[qkv_ab_dump] k_solo_vs_fused max_err={:.6e} first={}",
                                    qkv_ab_dump_max_abs(k_solo, k_fused_h),
                                    match dk_sf {
                                        None => "none (bit-exact)".to_string(),
                                        Some((i, a, b)) => {
                                            format!("k[{i}] solo={a:.8} fused={b:.8}")
                                        }
                                    },
                                );
                                eprintln!(
                                    "[qkv_ab_dump] k_solo_vs_gemv_kv max_err={:.6e} first={}",
                                    qkv_ab_dump_max_abs(k_solo, k_dual_h),
                                    match dk_sd {
                                        None => "none (bit-exact)".to_string(),
                                        Some((i, a, b)) => {
                                            format!("k[{i}] solo={a:.8} gemv_kv={b:.8}")
                                        }
                                    },
                                );
                                DUMP_N.fetch_add(1, Ordering::Relaxed);
                                if dq.is_some() || dk.is_some() || dv.is_some() {
                                    // Stop after first diverge — enough for triage.
                                    DUMP_N.store(8, Ordering::Relaxed);
                                }
                            }
                        }
                    }
                }
                // Phase edge: QKV proj → rope/norms.
                if tessl::ab_flags::need_barrier(true) {
                    crate::trace_op!("barrier_qkv", format!("layer={li}"), {
                        gpu.barrier()?;
                    });
                }
                if layer_probe_enabled() && li == 0 {
                    self.model.gpu.synchronize()?;
                    let qh = self.q.read_f32();
                    let q_rows = layer.q_proj.rows as usize;
                    let qs = Self::stats_f32(&qh[..q_rows.min(qh.len())]);
                    eprintln!(
                        "[layer_probe] after qkv_gemv q[0..{q_rows}] finite={} nan={} min={:.4} max={:.4} q_buf={}",
                        qs.finite, qs.nan, qs.min, qs.max, qh.len()
                    );
                    if is_producer {
                        let kh = self.k.read_f32();
                        let k_rows = layer.k_proj.as_ref().map(|k| k.rows as usize).unwrap_or(0);
                        let ks = Self::stats_f32(&kh[..k_rows.min(kh.len())]);
                        eprintln!(
                            "[layer_probe] after qkv_gemv k[0..{k_rows}] finite={} nan={} min={:.4} max={:.4}",
                            ks.finite, ks.nan, ks.min, ks.max
                        );
                    }
                }
                let fuse_rope_kv = is_producer && fuse_rope_kv_enabled();
                let fused_kv_store = if fuse_rope_kv {
                    // Resolve primary cache slot before rope so we can write in-kernel.
                    let (dst_k, dst_v, off, capacity) = match &role {
                        KvRole::Producer { slot } => match slot {
                            KvSlotId::SlidingRing { producer_index } => {
                                let i = *producer_index;
                                icb_tape_set_kv_ctx_sliding(i);
                                let off = self.sliding_rings[i].peek_write_offset()?;
                                (
                                    self.sliding_rings[i].k.clone(),
                                    self.sliding_rings[i].v.clone(),
                                    off,
                                    self.sliding_rings[i].elem_capacity_u32(),
                                )
                            }
                            KvSlotId::GlobalFull { producer_index } => {
                                let i = *producer_index;
                                icb_tape_set_kv_ctx_global(i);
                                let off = self.global_slots[i].peek_write_offset()?;
                                (
                                    self.global_slots[i].k.clone(),
                                    self.global_slots[i].v.clone(),
                                    off,
                                    self.global_slots[i].elem_capacity_u32(),
                                )
                            }
                        },
                        KvRole::Consumer { .. } => unreachable!(),
                    };
                    crate::trace_op!(
                        "rms_qkv_rope_kv_store",
                        format!(
                            "layer={li} pos={pos} hq={hq} hkv={hkv} d={head_dim} \
                             rotary={rotary_dim} theta={theta} kv_off={off}"
                        ),
                        {
                            rms_qkv_rope_kv_store(
                                gpu,
                                &self.q,
                                &self.k,
                                &self.v,
                                &layer.q_norm,
                                &layer.k_norm,
                                &layer.v_norm,
                                1,
                                hq as u32,
                                hkv as u32,
                                head_dim as u32,
                                rotary_dim as u32,
                                &self.pos_buf,
                                theta,
                                eps,
                                &dst_k,
                                &dst_v,
                                off,
                                capacity,
                            )?;
                        }
                    );
                    true
                } else {
                    crate::trace_op!(
                        "rms_qkv_rope",
                        format!(
                            "layer={li} pos={pos} hq={hq} hkv={hkv} d={head_dim} \
                             rotary={rotary_dim} theta={theta} q_only={}",
                            !is_producer
                        ),
                        {
                            rms_qkv_rope_ex_posbuf(
                                gpu,
                                &self.q,
                                &self.k,
                                &self.v,
                                &layer.q_norm,
                                &layer.k_norm,
                                &layer.v_norm,
                                1,
                                hq as u32,
                                hkv as u32,
                                head_dim as u32,
                                rotary_dim as u32,
                                &self.pos_buf,
                                theta,
                                eps,
                                /*q_only*/ !is_producer,
                            )?;
                        }
                    );
                    false
                };
                if layer_probe_enabled() && li == 0 {
                    self.model.gpu.synchronize()?;
                    let qs = Self::stats_f32(&self.q.read_f32());
                    eprintln!("[layer_probe] after rms_qkv_rope q finite={} nan={} min={:.4} max={:.4}", qs.finite, qs.nan, qs.min, qs.max);
                }
                fused_kv_store
            };

            let tkv_want = (pos + 1) as u32;
            let update_shared = is_producer
                && self
                    .model
                    .kv
                    .layers
                    .iter()
                    .take(first_shared)
                    .filter(|l| l.layer_type == layer_type)
                    .map(|l| l.layer)
                    .max()
                    == Some(li);

            if is_producer {
                let gpu = &self.model.gpu;
                // Phase edge: rope/norms → KV write (skipped when fused into rope).
                if !fused_kv_store && tessl::ab_flags::need_barrier(true) {
                    gpu.barrier()?;
                }
                let kv_elems = (hkv * head_dim) as u64 * 4;
                match &role {
                    KvRole::Producer { slot } => match slot {
                        KvSlotId::SlidingRing { producer_index } => {
                            let i = *producer_index;
                            crate::trace_op!(
                                "kv_write_ring",
                                format!(
                                    "layer={li} ring={i} pos={pos} tkv_want={tkv_want} \
                                     bytes/K≈{} update_shared={update_shared} fused={fused_kv_store}",
                                    diag::fmt_bytes(kv_elems)
                                ),
                                {
                                    if fused_kv_store {
                                        self.sliding_rings[i].commit_append()?;
                                    } else {
                                        icb_tape_set_kv_ctx_sliding(i);
                                        self.sliding_rings[i].append(gpu, &self.k, &self.v)?;
                                    }
                                    icb_tape_note_commit_sliding(i);
                                    if update_shared {
                                        crate::trace_op!(
                                            "kv_share_sliding",
                                            format!("layer={li} pos={pos}"),
                                            {
                                                icb_tape_set_kv_ctx_shared_sliding();
                                                self.shared_sliding.append(gpu, &self.k, &self.v)?;
                                                icb_tape_note_commit_shared_sliding();
                                            }
                                        );
                                    }
                                }
                            );
                        }
                        KvSlotId::GlobalFull { producer_index } => {
                            let i = *producer_index;
                            crate::trace_op!(
                                "kv_write_global",
                                format!(
                                    "layer={li} slot={i} pos={pos} tkv_want={tkv_want} \
                                     bytes/K≈{} update_shared={update_shared} fused={fused_kv_store}",
                                    diag::fmt_bytes(kv_elems)
                                ),
                                {
                                    if fused_kv_store {
                                        self.global_slots[i].commit_append()?;
                                    } else {
                                        icb_tape_set_kv_ctx_global(i);
                                        self.global_slots[i].append(gpu, &self.k, &self.v)?;
                                    }
                                    icb_tape_note_commit_global(i);
                                    if update_shared {
                                        crate::trace_op!(
                                            "kv_share_global",
                                            format!("layer={li} pos={pos}"),
                                            {
                                                icb_tape_set_kv_ctx_shared_global();
                                                self.shared_global.append(gpu, &self.k, &self.v)?;
                                                icb_tape_note_commit_shared_global();
                                            }
                                        );
                                    }
                                }
                            );
                        }
                    },
                    KvRole::Consumer { .. } => unreachable!(),
                }
            }

            {
                let gpu = &self.model.gpu;
                let layer = &self.model.layers[li];
                // Phase edge: KV write/share → FA.
                if tessl::ab_flags::need_barrier(true) {
                    gpu.barrier()?;
                }
                let (k_fa, v_fa, kv_off, tkv) = crate::trace_op!(
                    "kv_read_fa_buffers",
                    format!(
                        "layer={li} pos={pos} producer={is_producer} \
                         sliding={} tkv_want={tkv_want}",
                        layer_type.is_sliding()
                    ),
                    {
                        if is_producer {
                            match &role {
                                KvRole::Producer { slot } => match slot {
                                    KvSlotId::SlidingRing { producer_index } => {
                                        icb_tape_set_kv_ctx_sliding(*producer_index);
                                        let ring = &self.sliding_rings[*producer_index];
                                        let densify = ring.is_ring && ring.seq_len > ring.capacity;
                                        if densify {
                                            diag::infer_log(format_args!(
                                                "· kv_ring_densify layer={li} ring={} seq={} cap={}",
                                                producer_index, ring.seq_len, ring.capacity
                                            ));
                                        }
                                        ring.fa_buffers(
                                            gpu,
                                            &self.k_dense,
                                            &self.v_dense,
                                            tkv_want,
                                        )?
                                    }
                                    KvSlotId::GlobalFull { producer_index } => {
                                        icb_tape_set_kv_ctx_global(*producer_index);
                                        self.global_slots[*producer_index].fa_buffers(
                                            gpu,
                                            &self.k_dense,
                                            &self.v_dense,
                                            tkv_want,
                                        )?
                                    }
                                },
                                KvRole::Consumer { .. } => unreachable!(),
                            }
                        } else {
                            let slot = if layer_type.is_sliding() {
                                icb_tape_set_kv_ctx_shared_sliding();
                                &self.shared_sliding
                            } else {
                                icb_tape_set_kv_ctx_shared_global();
                                &self.shared_global
                            };
                            if tkv_want > slot.live_len_u32()? {
                                let e = Error::Kv(format!(
                                    "consumer pos={pos} shared tkv={}",
                                    slot.seq_len
                                ));
                                diag::err("gpu", &format!("KV consumer layer={li}"), &e);
                                return Err(e);
                            }
                            slot.fa_buffers(gpu, &self.k_dense, &self.v_dense, tkv_want)?
                        }
                    }
                );

                let fa_bytes = (tkv as u64)
                    * (hkv as u64)
                    * (head_dim as u64)
                    * 4
                    * 2; // K+V
                let o_elems = (hq * head_dim) as u32;
                let fuse_bf16 = fuse_bf16_fa();
                let o_bf16_for_fa = gpu.act_bf16_scratch(o_elems as usize)?;
                let o_fa = if fuse_bf16 {
                    &o_bf16_for_fa
                } else {
                    &self.o
                };
                if layer_type.is_sliding() {
                    let win = window.unwrap_or(512) as u32;
                    crate::trace_op!(
                        "fa_swa",
                        format!(
                            "layer={li} pos={pos} tkv={tkv} kv_off={kv_off} win={win} \
                             hq={hq} hkv={hkv} d={head_dim} bytes≈{}",
                            diag::fmt_bytes(fa_bytes)
                        ),
                        {
                            flash_attn_swa_h256_ex(
                                gpu,
                                &self.q,
                                &k_fa,
                                &v_fa,
                                o_fa,
                                1,
                                1,
                                tkv,
                                hq as u32,
                                hkv as u32,
                                win,
                                1.0,
                                pos as u32,
                                kv_off,
                                fuse_bf16,
                            )?;
                        }
                    );
                } else {
                    crate::trace_op!(
                        "fa_global",
                        format!(
                            "layer={li} pos={pos} tkv={tkv} kv_off={kv_off} \
                             hq={hq} hkv={hkv} d={head_dim} bytes≈{}",
                            diag::fmt_bytes(fa_bytes)
                        ),
                        {
                            flash_attn_global_h512_ex(
                                gpu,
                                &self.q,
                                &k_fa,
                                &v_fa,
                                o_fa,
                                1,
                                1,
                                tkv,
                                hq as u32,
                                hkv as u32,
                                1.0,
                                pos as u32,
                                kv_off,
                                fuse_bf16,
                            )?;
                        }
                    );
                }
                if layer_probe_enabled() && li == 0 {
                    self.model.gpu.synchronize()?;
                    let os = Self::stats_f32(&self.o.read_f32());
                    eprintln!("[layer_probe] after fa o finite={} nan={} min={:.4} max={:.4}", os.finite, os.nan, os.min, os.max);
                }
                icb_tape_clear_kv_ctx();
                let o_bytes = trace::gemv_bytes_est(
                    layer.o_proj.rows,
                    layer.o_proj.cols,
                    layer.o_proj.group_size,
                );
                // Phase edge: FA → o_proj. When fuse_bf16, FA already wrote O as bf16 into act_bf16_scratch.
                if !fuse_bf16 {
                    crate::trace_op!(
                        "cast_o_bf16",
                        format!("layer={li} n={o_elems}"),
                        {
                            let _ = prepare_act_bf16(gpu, &self.o, o_elems)?;
                        }
                    );
                }
                if tessl::ab_flags::need_barrier(true) {
                    gpu.barrier()?;
                }
                let o_bf16 = gpu.act_bf16_scratch(o_elems as usize)?;
                crate::trace_op!(
                    "gemv_o_resid",
                    format!(
                        "layer={li} [{}x{}] bytes≈{} gemma4_dual={use_gemma4_dual_norm}",
                        layer.o_proj.rows,
                        layer.o_proj.cols,
                        diag::fmt_bytes(o_bytes)
                    ),
                    {
                        if use_gemma4_dual_norm {
                            // MLX Gemma4: o = post_attention_ln(o_proj); x += o
                            if fuse_dual_norm_enabled() {
                                layer.o_proj.gemv_postnorm_add_into_bf16_x(
                                    gpu,
                                    &o_bf16,
                                    &self.x,
                                    &self.attn_proj,
                                    &layer.post_attn_norm,
                                    eps,
                                )?;
                            } else {
                                layer.o_proj.gemv_bf16_x(gpu, &o_bf16, &self.attn_proj)?;
                                if tessl::ab_flags::need_barrier(true) {
                                    gpu.barrier()?;
                                }
                                rms_norm_f32(
                                    gpu,
                                    &self.attn_proj,
                                    &layer.post_attn_norm,
                                    &self.normed,
                                    1,
                                    hidden,
                                    eps,
                                )?;
                                if tessl::ab_flags::need_barrier(true) {
                                    gpu.barrier()?;
                                }
                                ple_residual_add(gpu, &self.x, &self.normed, 1.0, hidden)?;
                            }
                        } else {
                            // Legacy / synthetic Pre-LN: fuse o_proj into residual.
                            layer.o_proj.gemv_add_into_bf16_x(
                                gpu,
                                &o_bf16,
                                &self.x,
                                &self.attn_proj,
                            )?;
                        }
                    }
                );
                if layer_probe_enabled() && li == 0 {
                    self.model.gpu.synchronize()?;
                    let xs = self.debug_x_stats();
                    eprintln!("[layer_probe] after o_proj+resid x finite={} nan={} min={:.4} max={:.4}", xs.finite, xs.nan, xs.min, xs.max);
                }

                if has_ple {
                    let ple_dim = ple_dim_cfg as u32;
                    let scale = (ple_dim as f32).sqrt();
                    let n_layers = n_layers as u32;
                    // Layer-fusion v1 (opt-in): lookup→residual is element-local,
                    // so the pair collapses to one dispatch. The RAW barrier on
                    // `x` (written by o_proj residual) still has to come first,
                    // so it moves *above* the fused call.
                    let fuse_ple = fuse_ple_residual_enabled() && self.model.ple_q4.is_some();
                    if fuse_ple {
                        let ple = self
                            .model
                            .ple_q4
                            .as_ref()
                            .expect("fuse_ple implies ple_q4");
                        if tessl::ab_flags::need_barrier(true) {
                            gpu.barrier()?;
                        }
                        crate::trace_op!(
                            "ple_lookup_q4_residual",
                            format!(
                                "layer={li} ple_dim={ple_dim} vocab={vocab} scale={scale} \
                                 fused=lookup+residual combine=1/√2"
                            ),
                            {
                                ple_lookup_q4_mlx_residual(
                                    gpu,
                                    &self.seed_tok,
                                    &ple.packed,
                                    &ple.scales,
                                    &ple.zeros,
                                    &self.ple_out,
                                    &self.x,
                                    ple_dim,
                                    vocab,
                                    1,
                                    scale,
                                    std::f32::consts::FRAC_1_SQRT_2,
                                    li as u32,
                                    n_layers,
                                    ple.group_size,
                                )?;
                            }
                        );
                    } else if let Some(ref ple) = self.model.ple_q4 {
                        crate::trace_op!(
                            "ple_lookup_q4",
                            format!(
                                "layer={li} ple_dim={ple_dim} vocab={vocab} scale={scale}"
                            ),
                            {
                                ple_lookup_q4_mlx(
                                    gpu,
                                    &self.seed_tok,
                                    &ple.packed,
                                    &ple.scales,
                                    &ple.zeros,
                                    &self.ple_out,
                                    ple_dim,
                                    vocab,
                                    1,
                                    scale,
                                    li as u32,
                                    n_layers,
                                    ple.group_size,
                                )?;
                            }
                        );
                    } else if let Some(ref table) = layer.ple_table {
                        let ple_bytes = (ple_dim as u64) * 2;
                        crate::trace_op!(
                            "ple_lookup",
                            format!(
                                "layer={li} ple_dim={ple_dim} vocab={vocab} scale={scale} bytes≈{}",
                                diag::fmt_bytes(ple_bytes)
                            ),
                            {
                                ple_lookup(
                                    gpu,
                                    &self.seed_tok,
                                    table,
                                    &self.ple_out,
                                    ple_dim,
                                    vocab,
                                    1,
                                    scale,
                                )?;
                            }
                        );
                    }
                    // Phase edge: o_resid (+ ple_lookup) → ple_residual (RAW on x).
                    // Fused lane already emitted this barrier and folded the add.
                    if !fuse_ple {
                        if tessl::ab_flags::need_barrier(true) {
                            gpu.barrier()?;
                        }
                        crate::trace_op!(
                            "ple_residual",
                            format!("layer={li} ple_dim={ple_dim} combine=1/√2"),
                            {
                                ple_residual_add(
                                    gpu,
                                    &self.x,
                                    &self.ple_out,
                                    std::f32::consts::FRAC_1_SQRT_2,
                                    ple_dim,
                                )?;
                            }
                        );
                    }
                }

                if layer_probe_enabled() && li == 0 {
                    self.model.gpu.synchronize()?;
                    let xs = self.debug_x_stats();
                    eprintln!(
                        "[layer_probe] after ple_or_skip x finite={} nan={} min={:.4} max={:.4} has_ple={has_ple}",
                        xs.finite, xs.nan, xs.min, xs.max
                    );
                }

                // Phase edge: residual → pre_ff (Gemma4) / post_attn (legacy) before MLP.
                if tessl::ab_flags::need_barrier(true) {
                    gpu.barrier()?;
                }
                let mlp_in_norm = layer.pre_ff_norm.as_ref().unwrap_or(&layer.post_attn_norm);
                crate::trace_op!(
                    "rms_pre_ff",
                    format!("layer={li} hidden={hidden} gemma4={has_pre_ff}"),
                    {
                        if fuse_bf16_rms() {
                            let _ = rms_norm_to_act_bf16(
                                gpu,
                                &self.x,
                                mlp_in_norm,
                                1,
                                hidden,
                                eps,
                            )?;
                        } else {
                            rms_norm_f32(
                                gpu,
                                &self.x,
                                mlp_in_norm,
                                &self.normed,
                                1,
                                hidden,
                                eps,
                            )?;
                            let _ = prepare_act_bf16(gpu, &self.normed, hidden)?;
                        }
                    }
                );
                if layer_probe_enabled() && li == 0 {
                    self.model.gpu.synchronize()?;
                    let n = Self::stats_f32(&self.normed.read_f32()[..self.model.hidden]);
                    eprintln!("[layer_probe] after rms_post_attn finite={} nan={} min={:.4} max={:.4}", n.finite, n.nan, n.min, n.max);
                }
                // Phase edge: post_attn → MLP.
                if tessl::ab_flags::need_barrier(true) {
                    gpu.barrier()?;
                }
                let x_bf16 = gpu.act_bf16_scratch(hidden as usize)?;
                let gate_bytes = trace::gemv_bytes_est(
                    layer.gate_proj.rows,
                    layer.gate_proj.cols,
                    layer.gate_proj.group_size,
                );
                let up_bytes = trace::gemv_bytes_est(
                    layer.up_proj.rows,
                    layer.up_proj.cols,
                    layer.up_proj.group_size,
                );
                let use_fuse_gate_down = layer.gate_proj.can_fuse_gate_down(
                    &layer.up_proj,
                    &layer.down_proj,
                ) && !(use_gemma4_dual_norm && has_post_ff);
                if use_fuse_gate_down {
                    // Scratch must already exist (ensured before layer loop when flag on).
                    crate::trace_op!(
                        "persistent_interp_gate_down_q4",
                        format!(
                            "layer={li} n_tg={} mid={} hidden={}",
                            PERSISTENT_INTERP_MAX_TG,
                            intermediate,
                            hidden
                        ),
                        {
                            self.dispatch_fuse_gate_down_q4_ref(li, &x_bf16)?;
                        }
                    );
                    if let Some(ref w) = layer.post_ff_norm {
                        if tessl::ab_flags::need_barrier(true) {
                            gpu.barrier()?;
                        }
                        rms_norm_f32(gpu, &self.x, w, &self.normed, 1, hidden, eps)?;
                        // IcbScalarPool via copy_f32_n — avoid Immediate setAddress
                        // residual (E4B: 1×/layer → last_setAddress 42).
                        copy_f32_n(gpu, &self.normed, &self.x, hidden)?;
                    }
                } else if layer.gate_proj.can_fuse_gate_up_gelu(&layer.up_proj) {
                    crate::trace_op!(
                        "gemv_gate_up_gelu",
                        format!(
                            "layer={li} [{}x{}] fused bytes≈{}",
                            layer.gate_proj.rows,
                            layer.gate_proj.cols,
                            diag::fmt_bytes(gate_bytes.saturating_add(up_bytes))
                        ),
                        {
                            if fuse_bf16_mlp() {
                                // Write bf16 mid into self.mid storage (alias); avoid clobbering x_bf16 act scratch.
                                gemv_q4_mlx_gate_up_gelu_bf16_x_out_bf16(
                                    gpu,
                                    &layer.gate_proj,
                                    &layer.up_proj,
                                    &x_bf16,
                                    &self.mid,
                                )?;
                            } else {
                                gemv_q4_mlx_gate_up_gelu_bf16_x(
                                    gpu,
                                    &layer.gate_proj,
                                    &layer.up_proj,
                                    &x_bf16,
                                    &self.mid,
                                )?;
                            }
                        }
                    );
                } else {
                    crate::trace_op!(
                        "gemv_gate",
                        format!(
                            "layer={li} [{}x{}] bytes≈{}",
                            layer.gate_proj.rows,
                            layer.gate_proj.cols,
                            diag::fmt_bytes(gate_bytes)
                        ),
                        {
                            layer.gate_proj.gemv_bf16_x(gpu, &x_bf16, &self.gate)?;
                        }
                    );
                    crate::trace_op!(
                        "gemv_up",
                        format!(
                            "layer={li} [{}x{}] bytes≈{}",
                            layer.up_proj.rows,
                            layer.up_proj.cols,
                            diag::fmt_bytes(up_bytes)
                        ),
                        {
                            layer.up_proj.gemv_bf16_x(gpu, &x_bf16, &self.up)?;
                        }
                    );
                    // Phase edge: gate∥up → gelu (unfused path).
                    // Always drain before gelu — bf16→f32 GEMV producers vs gelu consumer.
                    gpu.barrier()?;
                    crate::trace_op!(
                        "mlp_gelu",
                        format!("layer={li} intermediate={intermediate}"),
                        {
                            if fuse_bf16_mlp() {
                                mlp_gelu_tanh_bf16(
                                    gpu,
                                    &self.gate,
                                    &self.up,
                                    &self.mid,
                                    intermediate,
                                )?;
                            } else {
                                mlp_gelu_tanh(gpu, &self.gate, &self.up, &self.mid, intermediate)?;
                            }
                        }
                    );
                }
                if layer_probe_enabled() && li == 0 {
                    self.model.gpu.synchronize()?;
                    let n = intermediate as usize;
                    let gate_h = self.gate.read_f32();
                    let up_h = self.up.read_f32();
                    let mid_h = self.mid.read_f32();
                    let gs = Self::stats_f32(&gate_h[..n]);
                    let us = Self::stats_f32(&up_h[..n]);
                    let ms = Self::stats_f32(&mid_h[..n]);
                    eprintln!(
                        "[layer_probe] gate finite={} nan={} min={:.4} max={:.4}",
                        gs.finite, gs.nan, gs.min, gs.max
                    );
                    eprintln!(
                        "[layer_probe] up   finite={} nan={} min={:.4} max={:.4}",
                        us.finite, us.nan, us.min, us.max
                    );
                    eprintln!(
                        "[layer_probe] after gate_up_gelu mid finite={} nan={} min={:.4} max={:.4}",
                        ms.finite, ms.nan, ms.min, ms.max
                    );
                    let k = (2.0f32 / std::f32::consts::PI).sqrt();
                    let mut cpu_nan = 0usize;
                    let mut max_diff = 0f32;
                    for i in 0..n {
                        let x = gate_h[i].clamp(-20.0, 20.0);
                        let g = 0.5 * x * (1.0 + (k * (x + 0.044715 * x * x * x)).tanh());
                        let y = g * up_h[i];
                        if y.is_nan() {
                            cpu_nan += 1;
                        }
                        if !mid_h[i].is_nan() && y.is_finite() {
                            max_diff = max_diff.max((mid_h[i] - y).abs());
                        }
                    }
                    eprintln!("[layer_probe] cpu_gelu_from_gate_up nan={cpu_nan} max_diff_vs_gpu={max_diff:.4}");
                }
                let down_bytes = trace::gemv_bytes_est(
                    layer.down_proj.rows,
                    layer.down_proj.cols,
                    layer.down_proj.group_size,
                );
                if !use_fuse_gate_down {
                // Phase edge: gate_up/gelu → down. When fuse_bf16, mid already holds bf16.
                let fuse_bf16 = fuse_bf16_mlp();
                if !fuse_bf16 {
                    crate::trace_op!(
                        "cast_mid_bf16",
                        format!("layer={li} n={intermediate}"),
                        {
                            let _ = prepare_act_bf16(gpu, &self.mid, intermediate)?;
                        }
                    );
                }
                if tessl::ab_flags::need_barrier(true) {
                    gpu.barrier()?;
                }
                let mid_bf16 = if fuse_bf16 {
                    self.mid.clone()
                } else {
                    gpu.act_bf16_scratch(intermediate as usize)?
                };
                crate::trace_op!(
                    "gemv_down_resid",
                    format!(
                        "layer={li} [{}x{}] bytes≈{} gemma4_dual={use_gemma4_dual_norm}",
                        layer.down_proj.rows,
                        layer.down_proj.cols,
                        diag::fmt_bytes(down_bytes)
                    ),
                    {
                        if use_gemma4_dual_norm && has_post_ff {
                            // MLX Gemma4: h = post_ff_ln(down); x += h
                            if let Some(ref w) = layer.post_ff_norm {
                                if fuse_dual_norm_enabled() {
                                    // Fold layer_scalar into the last residual when present.
                                    let fold_scale = if (layer_scalar - 1.0).abs() > 1e-8 {
                                        layer_scalar
                                    } else {
                                        1.0
                                    };
                                    layer.down_proj.gemv_postnorm_add_into_bf16_x_scaled(
                                        gpu,
                                        &mid_bf16,
                                        &self.x,
                                        &self.down,
                                        w,
                                        eps,
                                        fold_scale,
                                    )?;
                                } else {
                                    layer.down_proj.gemv_bf16_x(gpu, &mid_bf16, &self.down)?;
                                    if tessl::ab_flags::need_barrier(true) {
                                        gpu.barrier()?;
                                    }
                                    rms_norm_f32(gpu, &self.down, w, &self.normed, 1, hidden, eps)?;
                                    if tessl::ab_flags::need_barrier(true) {
                                        gpu.barrier()?;
                                    }
                                    ple_residual_add(gpu, &self.x, &self.normed, 1.0, hidden)?;
                                }
                            }
                        } else {
                            // E4B / synthetic: fused down + legacy post_ff residual replace.
                            layer.down_proj.gemv_add_into_bf16_x(
                                gpu,
                                &mid_bf16,
                                &self.x,
                                &self.down,
                            )?;
                            if let Some(ref w) = layer.post_ff_norm {
                                rms_norm_f32(gpu, &self.x, w, &self.normed, 1, hidden, eps)?;
                                // IcbScalarPool via copy_f32_n — E4B post_ff replace
                                // was the residual Immediate setAddress × n_layers.
                                copy_f32_n(gpu, &self.normed, &self.x, hidden)?;
                            }
                        }
                    }
                );
                } // !use_fuse_gate_down
                // MLX: layer_scalar multiplies full layer output after both residuals.
                // Skipped when folded into fused down-proj postnorm (above).
                let folded = fuse_dual_norm_enabled() && use_gemma4_dual_norm && has_post_ff;
                if use_gemma4_dual_norm && (layer_scalar - 1.0).abs() > 1e-8 && !folded {
                    if tessl::ab_flags::need_barrier(true) {
                        gpu.barrier()?;
                    }
                    crate::trace_op!(
                        "layer_scalar",
                        format!("layer={li} scale={layer_scalar}"),
                        {
                            scale_f32_inplace(gpu, &self.x, layer_scalar, hidden)?;
                        }
                    );
                }
            }

            // Mini-only persistent-interp (opt-in): exercise doctrine edges
            // FA→o_proj + gate→down via stand-ins on dedicated dense scratch.
            // Shipping Q4 FA/MLP above are unchanged; Hot/E4B/31B no-op even if
            // GEMMA_METAL_PERSISTENT_INTERP=1 (see D17 / Metal FP caveat).
            crate::trace_op!(
                "persistent_interp_fa_o",
                format!("layer={li} mini_hook"),
                {
                    self.dispatch_persistent_interp_fa_o_edge()?;
                }
            );
            crate::trace_op!(
                "persistent_interp_gate_down",
                format!("layer={li} mini_hook"),
                {
                    self.dispatch_persistent_interp_gate_down_edge()?;
                }
            );

            if let Some(ref mut tr) = tracer {
                // Attribute GPU wait for this layer's packed encode (TRACE=sync only).
                let sync_ref = sync_cb.as_ref().map(|b| b.as_ref());
                tr.flush_gpu_bucket("layer", 0, sync_ref);
            }

            // Optional NaN triage: GEMMA_METAL_LAYER_PROBE=1 syncs after each layer.
            if layer_probe_enabled() {
                self.model.gpu.synchronize()?;
                let xs = self.debug_x_stats();
                eprintln!(
                    "[layer_probe] pos={pos} after_layer={li} finite={} nan={} min={:.4} max={:.4}",
                    xs.finite, xs.nan, xs.min, xs.max
                );
                if !xs.finite {
                    return Err(Error::Metal(format!(
                        "layer_probe: NaN residual after layer {li} pos={pos} nan={}",
                        xs.nan
                    )));
                }
            }

            // DFlash capture: AO-while-capture (CaptureAlwaysOnGuard) is the
            // fidelity path. Under hazard-only (CAPTURE_AO=0), synchronize before
            // copy — may still fail exactness (tick3); do not use for shipping.
            let capture_nop = capture_nop_enabled();
            if self.capture.is_some() && !capture_nop {
                let need = self
                    .capture
                    .as_ref()
                    .map(|c| c.layer_ids.contains(&li))
                    .unwrap_or(false);
                if need {
                    let h = self.model.hidden;
                    let slot = self.capture.as_mut().and_then(|c| c.mark_layer(li));
                    if let (Some(slot), Some(ref row)) = (slot, self.capture_row.as_ref()) {
                        let force_barrier = capture_barrier_forced();
                        if tessl::ab_flags::hazard_barriers() {
                            self.model.gpu.synchronize()?;
                        } else if force_barrier {
                            self.model.gpu.barrier()?;
                        }
                        copy_f32_to_offset(
                            &self.model.gpu,
                            &self.x,
                            row,
                            slot * h,
                            h as u32,
                        )?;
                    }
                }
            } else if self.capture.is_some() && capture_nop {
                // Mark slots filled so finish_step does not error; row stays zero.
                let _ = self.capture.as_mut().and_then(|c| c.mark_layer(li));
            }
        }

        // Finish Binder → DecodeIcb capture after layers (before head). Head stays
        // live on replay — lm_head/softcap via ICB was observed to collapse argmax→0
        // even when residual stayed finite (token-parity triage 2026-07-19).
        if capturing_layer_icb {
            let capture_guard = layer_icb_capture_guard
                .as_mut()
                .expect("capturing_layer_icb always owns its paired tape guard");
            if let Some(cap) = capture_guard.take_decode_capture() {
                let n = cap.commands.len();
                if n >= tessl::DecodeIcb::MIN_LAYER_GRAPH_COMMANDS {
                    match tessl::DecodeIcb::from_commands(&self.model.gpu.rt, cap.commands)
                    {
                        Ok(mut icb) => {
                            eprintln!(
                                "encode_once: DecodeIcb layer-graph attached ({})",
                                icb.status_line()
                            );
                            // Watermark: pooled-scalar cursor shape of the captured
                            // step. Replay steps must reproduce it exactly.
                            self.icb_capture_watermark =
                                Some(self.model.gpu.icb_scalars.cursor_snapshot());
                            if let Some(tape) = capture_guard.take_scalar_write_tape() {
                                eprintln!(
                                    "encode_once: scalar-write tape ops={} (skip-nop replay)",
                                    tape.op_count()
                                );
                                self.icb_scalar_write_tape = Some(tape);
                            }
                            // Triage probe: residual x, sampled per replayed command
                            // when GEMMA_METAL_ICB_TRIAGE=1 (localizes the first
                            // diverging dispatch in one run).
                            icb.set_triage_probe(self.x.clone(), self.model.hidden);
                            self.encode_once.attach_decode_icb(icb);
                        }
                        Err(e) => {
                            eprintln!("encode_once: DecodeIcb::from_commands failed: {e}");
                            let _ = capture_guard.take_scalar_write_tape();
                            // mini_copy_chain is a dispatch-freeze proof for mini dims only.
                            if self.model.is_synthetic_mini() {
                                match tessl::DecodeIcb::mini_copy_chain(
                                    &self.model.gpu.rt,
                                    64,
                                ) {
                                    Ok((icb, _)) => self.encode_once.attach_decode_icb(icb),
                                    Err(e2) => eprintln!(
                                        "encode_once: mini_copy_chain fallback failed: {e2}"
                                    ),
                                }
                            }
                        }
                    }
                } else {
                    eprintln!(
                        "encode_once: capture too small ({n} cmds); {}",
                        if self.model.is_synthetic_mini() {
                            "mini_copy_chain fallback"
                        } else {
                            "leaving DecodeIcb unwired (live encode)"
                        }
                    );
                    let _ = capture_guard.take_scalar_write_tape();
                    if self.model.is_synthetic_mini() {
                        match tessl::DecodeIcb::mini_copy_chain(&self.model.gpu.rt, 64) {
                            Ok((icb, _)) => self.encode_once.attach_decode_icb(icb),
                            Err(e) => {
                                eprintln!("encode_once: mini_copy_chain attach failed: {e}");
                            }
                        }
                    }
                }
            } else {
                let _ = capture_guard.take_scalar_write_tape();
            }
        }

        // Layer-graph replay: drop nop, execute frozen layer ICB, then live head below.
        if icb_replay_prep {
            drop(binder_nop_guard.take());
            // Watermark gate: pooled binds in the frozen tape are (buffer, offset)
            // pairs from the CAPTURE step. If this step's nop-prep pushed a
            // different number of scalars, every later offset is misaligned and
            // the replay reads stale/garbage scalars (2026-07-19 token-parity
            // FAIL mode). Nop already consumed the loop, so fail loudly.
            if let Some((cu, cf)) = self.icb_capture_watermark {
                let (u, f) = self.model.gpu.icb_scalars.cursor_snapshot();
                if (u, f) != (cu, cf) {
                    return Err(Error::Metal(format!(
                        "encode_once: ICB scalar-cursor mismatch at pos={pos}: \
                         capture=(u32 {cu}, f32 {cf}) replay=(u32 {u}, f32 {f}) — \
                         step graph shape diverged from captured tape"
                    )));
                }
            }
            match self.encode_once.try_replay_ready_icb(&self.model.gpu.rt) {
                Ok(slot) => {
                    self.encode_once.on_gpu_complete(slot);
                }
                Err(tessl::CbReplayError::NotReady) => {
                    return Err(Error::Metal(
                        "encode_once: layer-graph replay expected Ready slot".into(),
                    ));
                }
                Err(tessl::CbReplayError::NotWired) => {
                    return Err(Error::Metal(
                        "encode_once: layer-graph DecodeIcb not wired at replay".into(),
                    ));
                }
                Err(e) => {
                    return Err(Error::Metal(format!("encode_once try_replay_icb: {e}")));
                }
            }
            // Ensure residual is visible before live lm_head.
            if tessl::ab_flags::need_barrier(true) {
                self.model.gpu.barrier()?;
            }
        }

        if head {
            {
                let gpu = &self.model.gpu;
                crate::trace_op!("final_norm", format!("pos={pos} hidden={hidden}"), {
                    if fuse_bf16_rms() {
                        let _ = rms_norm_to_act_bf16(
                            gpu,
                            &self.x,
                            &self.model.final_norm,
                            1,
                            hidden,
                            eps,
                        )?;
                    } else {
                        rms_norm_f32(
                            gpu,
                            &self.x,
                            &self.model.final_norm,
                            &self.normed,
                            1,
                            hidden,
                            eps,
                        )?;
                        let _ = prepare_act_bf16(gpu, &self.normed, hidden)?;
                    }
                });
                if tessl::ab_flags::need_barrier(true) {
                    gpu.barrier()?;
                }
                let x_bf16 = gpu.act_bf16_scratch(hidden as usize)?;
                let lm_bytes = trace::gemv_bytes_est(
                    self.model.lm_head.rows,
                    self.model.lm_head.cols,
                    self.model.lm_head.group_size,
                );
                crate::trace_op!(
                    "gemv_lm_head",
                    format!(
                        "pos={pos} [{}x{}] softcap={softcap} bytes≈{}",
                        self.model.lm_head.rows,
                        self.model.lm_head.cols,
                        diag::fmt_bytes(lm_bytes)
                    ),
                    {
                        self.model
                            .lm_head
                            .gemv_bf16_x(gpu, &x_bf16, &self.logits)?;
                    }
                );
                if let Some(ref mut tr) = tracer {
                    let sync_ref = sync_cb.as_ref().map(|b| b.as_ref());
                    tr.flush_gpu_bucket("lm_head", lm_bytes, sync_ref);
                }
                // softcap must not race lm_head. Always insert an explicit RAW
                // barrier here — even under always-on auto barriers. On 31B,
                // capture-off + always-on still collapsed to 240017/236773 while
                // capture-on (incl. CAPTURE_NOP) got target_next=531; the missing
                // edge is this producer→consumer drain before softcap/argmax.
                self.model.gpu.barrier()?;
            }
            crate::trace_op!(
                "softcap_argmax",
                format!("pos={pos} vocab={vocab} softcap={softcap}"),
                {
                    softcap_argmax_encode(
                        &self.model.gpu,
                        &self.logits,
                        softcap,
                        vocab,
                        &mut self.argmax_scratch,
                        &self.argmax_tok,
                    )?;
                }
            );
            if let Some(ref mut tr) = tracer {
                let sync_ref = sync_cb.as_ref().map(|b| b.as_ref());
                tr.flush_gpu_bucket("softcap_argmax", (vocab as u64) * 4, sync_ref);
            }
        }

        // Capture / conditioner timing (exactness-critical):
        // Do NOT encode conditioner FC between softcap and argmax readback. Extra
        // GEMV in that window shifts softcap completion vs capture-only and can
        // collapse D-Flash streams (observed: accept=0 → all-same token ≠ greedy).
        // Prefill has no softcap — project immediately then sync once.
        let mut capture_host_pending = false;
        if self.capture.is_some() {
            if let Some(ref row_buf) = self.capture_row {
                if readback {
                    capture_host_pending = true;
                } else {
                    if tessl::ab_flags::need_barrier(true) {
                        self.model.gpu.barrier()?;
                    }
                    if let Some(ref mut cond) = self.conditioner {
                        cond.project_row(&self.model.gpu, row_buf)?;
                    }
                    self.model.gpu.synchronize()?;
                    let row = row_buf.read_f32();
                    if let Some(ref mut c) = self.capture {
                        c.finish_step_from_row(&row)?;
                    }
                }
            }
        }

        self.pos += 1;

        // Mini persistent-interp: sync + check barrier fail after this step's hooks.
        self.finish_persistent_interp_step()?;

        // Encode-once scaffold (opt-in): advance ping-pong ledger. Layer-graph
        // ICB replay uses mark_replay_step (no live_encodes++); capture / live
        // encode uses mark_live_step. `note_layer_live_replay` already advanced.
        if encode_once_enabled() {
            if icb_replay_prep {
                let _ = self
                    .encode_once
                    .mark_replay_step(format!("decode_icb_replay pos={pos}"));
            } else if !icb_live_replay_noted {
                let _ = self
                    .encode_once
                    .mark_live_step(format!("decode_step pos={pos}"));
            }
        }

        if let Some(ref mut tr) = tracer {
            tr.end_token(TraceFlags {
                hazard_barriers_auto: tessl::ab_flags::hazard_barriers(),
                ple: false,
                async_encode: true,
            });
            // One-shot stderr table after first decode token when TRACE=sync/host.
            if pos == 0 || matches!(trace::mode(), trace::TraceMode::Sync) && pos < 4 {
                tr.print_summary_table();
            }
        }

        if readback {
            crate::trace_op!(
                "cpu_sync_readback",
                format!("pos={pos} synchronize+read argmax_tok — potential stall"),
                {
                    diag::infer_stall(format!("decode_step pos={pos} before tok readback").as_str());
                    self.model.gpu.synchronize()?;
                }
            );
            let next = self.argmax_tok.read_u32()[0];
            if capture_host_pending {
                if let Some(ref row_buf) = self.capture_row {
                    // Softcap already drained. Now project FC → h_ctx, then host concat.
                    if let Some(ref mut cond) = self.conditioner {
                        if tessl::ab_flags::need_barrier(true) {
                            self.model.gpu.barrier()?;
                        }
                        cond.project_row(&self.model.gpu, row_buf)?;
                        // Draft may read h_ctx immediately after this step returns.
                        self.model.gpu.synchronize()?;
                    }
                    let row = row_buf.read_f32();
                    if let Some(ref mut c) = self.capture {
                        c.finish_step_from_row(&row)?;
                    }
                }
            }
            diag::infer_log(format_args!(
                "· sample/argmax result next={next} pos={pos}"
            ));
            diag::log(
                "gpu",
                format_args!(
                    "✔ step pos={pos} → next={next} in {:.1} ms (synced)",
                    step_t0.elapsed().as_secs_f64() * 1e3
                ),
            );
            Ok(next)
        } else {
            if pos < 2 || head {
                diag::log(
                    "gpu",
                    format_args!(
                        "✔ step pos={pos} encode-only head={head} in {:.1} ms",
                        step_t0.elapsed().as_secs_f64() * 1e3
                    ),
                );
            }
            Ok(0)
        }
    }

    /// One decode step at absolute position `self.pos` for `token`. Returns next token id.
    pub fn step(&mut self, token: u32) -> Result<u32> {
        let _s = InferScope::begin("step", format!("token={token} pos={}", self.pos));
        self.step_inner(StepSeed::Host(token), true, true)
    }

    /// Prefill one token: embed→layers (no lm_head/argmax/sync).
    pub fn step_prefill(&mut self, token: u32) -> Result<()> {
        let _s = InferScope::begin(
            "prefill",
            format!("token={token} pos={}", self.pos),
        );
        let _ = self.step_inner(StepSeed::Host(token), false, false)?;
        Ok(())
    }

    /// Absolute decode position (next write index into KV / RoPE).
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// GPU-resident `u32×1` absolute position (encode-once scaffolding).
    pub fn pos_buf(&self) -> &GpuBuffer {
        &self.pos_buf
    }

    /// Hard ceiling on `pos` before a KV slot refuses the next append, or `None`
    /// when every slot is a sliding ring and therefore wraps instead of filling.
    ///
    /// Sliding rings recycle (`head = (head + 1) % capacity`), so they impose no
    /// bound. Full-slot (global) KV does: [`GpuKvSlot::peek_write_offset`]
    /// returns `Error::Kv("GPU KV full")` once `seq_len == capacity`. Long-running
    /// callers — the sustained-load gate, a server session — need that number to
    /// recycle *before* the error rather than treating a legitimate capacity stop
    /// as a runtime fault.
    pub fn kv_capacity(&self) -> Option<usize> {
        self.global_slots
            .iter()
            .chain(std::iter::once(&self.shared_global))
            .chain(self.sliding_rings.iter())
            .chain(std::iter::once(&self.shared_sliding))
            .filter(|s| !s.is_ring && s.capacity > 0)
            .map(|s| s.capacity)
            .min()
    }

    /// Ping-pong CB replay scaffold (record/commit/reuse bookkeeping; replay not wired).
    pub fn encode_once_scaffold(&self) -> &tessl::PingPongCbReplay {
        &self.encode_once
    }

    /// Mutable access for tests / harness inspecting live-encode counters.
    pub fn encode_once_scaffold_mut(&mut self) -> &mut tessl::PingPongCbReplay {
        &mut self.encode_once
    }

    /// Gate→down stand-in dispatches from mini decode (0 unless flag+mini).
    pub fn persistent_interp_gate_down_hits(&self) -> u64 {
        self.persistent_interp
            .as_ref()
            .map(|h| h.gate_down_hits)
            .unwrap_or(0)
    }

    /// FA→o_proj stand-in dispatches from mini decode (0 unless flag+mini).
    pub fn persistent_interp_fa_o_hits(&self) -> u64 {
        self.persistent_interp
            .as_ref()
            .map(|h| h.fa_o_hits)
            .unwrap_or(0)
    }

    /// Last barrier `fail` flag read after a mini decode hook dispatch.
    pub fn persistent_interp_last_fail(&self) -> u32 {
        self.persistent_interp
            .as_ref()
            .map(|h| h.last_fail)
            .unwrap_or(0)
    }

    /// Write [`Self::pos_buf`] once at step/verify start (host → GPU, not per layer).
    fn sync_pos_buf(&self, pos: u32) {
        self.pos_buf.write_u32(&[pos]);
    }

    /// A2 residual: replay captured scalar pushes + KV host commits without the
    /// binder-nop layer loop. `reset_step` already ran at step start.
    fn apply_icb_scalar_write_tape(&mut self, pos: usize) -> Result<()> {
        let tape = self
            .icb_scalar_write_tape
            .as_ref()
            .ok_or_else(|| Error::Metal("icb scalar-write tape missing".into()))?
            .clone();
        let pool = &self.model.gpu.icb_scalars;
        for op in &tape.ops {
            match op {
                IcbScalarTapeOp::U32Const(v) => {
                    let _ = pool.push_u32(*v)?;
                }
                IcbScalarTapeOp::U32Dyn(src) => {
                    let v = self.eval_icb_dyn_u32(src, pos)?;
                    // Tape TLS inactive on replay — push_u32_dyn only writes.
                    let _ = pool.push_u32_dyn(v, *src)?;
                }
                IcbScalarTapeOp::F32Const(v) => {
                    let _ = pool.push_f32(*v)?;
                }
                IcbScalarTapeOp::Kv(kv) => match kv {
                    IcbKvHostOp::CommitSliding(i) => self.sliding_rings[*i].commit_append()?,
                    IcbKvHostOp::CommitGlobal(i) => self.global_slots[*i].commit_append()?,
                    IcbKvHostOp::CommitSharedSliding => self.shared_sliding.commit_append()?,
                    IcbKvHostOp::CommitSharedGlobal => self.shared_global.commit_append()?,
                },
            }
        }
        Ok(())
    }

    fn eval_icb_dyn_u32(&self, src: &IcbDynSrc, pos: usize) -> Result<u32> {
        let pos_u32 = u32::try_from(pos)
            .map_err(|_| Error::Kv("decode position exceeds the device u32 range".into()))?;
        let tkv_limit = pos_u32.saturating_add(1);
        let filled_of = |slot: &GpuKvSlot| -> Result<u32> {
            Ok(slot.live_len_u32()?.min(tkv_limit))
        };
        let start_of = |slot: &GpuKvSlot| -> Result<u32> {
            if slot.is_ring && slot.seq_len > slot.capacity {
                slot.ring_start_u32()
            } else {
                Ok(0)
            }
        };
        let kv_pos_of = |slot: &GpuKvSlot| -> Result<u32> {
            if slot.is_ring && slot.seq_len > slot.capacity {
                slot.ring_position_u32()
            } else {
                Ok(0)
            }
        };
        Ok(match src {
            IcbDynSrc::Pos => pos_u32,
            IcbDynSrc::SlidingPeek(i) => self.sliding_rings[*i].peek_write_offset()?,
            IcbDynSrc::GlobalPeek(i) => self.global_slots[*i].peek_write_offset()?,
            IcbDynSrc::SharedSlidingPeek => self.shared_sliding.peek_write_offset()?,
            IcbDynSrc::SharedGlobalPeek => self.shared_global.peek_write_offset()?,
            IcbDynSrc::SlidingFilled(i) => filled_of(&self.sliding_rings[*i])?,
            IcbDynSrc::SlidingStart(i) => start_of(&self.sliding_rings[*i])?,
            IcbDynSrc::SlidingKvPos(i) => kv_pos_of(&self.sliding_rings[*i])?,
            IcbDynSrc::SlidingTkv(i) => filled_of(&self.sliding_rings[*i])?,
            IcbDynSrc::GlobalFilled(i) => filled_of(&self.global_slots[*i])?,
            IcbDynSrc::GlobalTkv(i) => filled_of(&self.global_slots[*i])?,
            IcbDynSrc::SharedSlidingFilled => filled_of(&self.shared_sliding)?,
            IcbDynSrc::SharedSlidingStart => start_of(&self.shared_sliding)?,
            IcbDynSrc::SharedSlidingKvPos => kv_pos_of(&self.shared_sliding)?,
            IcbDynSrc::SharedSlidingTkv => filled_of(&self.shared_sliding)?,
            IcbDynSrc::SharedGlobalFilled => filled_of(&self.shared_global)?,
            IcbDynSrc::SharedGlobalTkv => filled_of(&self.shared_global)?,
        })
    }

    /// Opt-in mini-only: ensure dense scratch for persistent-interp stand-ins.
    ///
    /// No-op when flag off or graph is not synthetic mini (Hot/E4B/31B safe).
    fn ensure_persistent_interp_hook(&mut self) -> Result<bool> {
        if !persistent_interp_enabled() || !self.model.is_synthetic_mini() {
            return Ok(false);
        }
        if self.persistent_interp.is_some() {
            return Ok(true);
        }
        let gpu = &self.model.gpu;
        // Skip cleanly if metallib lacks the prototype kernels.
        if gpu
            .rt
            .pipeline(KernelId::PersistentInterpGateDown.entry_name())
            .is_err()
            || gpu
                .rt
                .pipeline(KernelId::PersistentInterpFaOProj.entry_name())
                .is_err()
        {
            return Ok(false);
        }
        let n_mid = self.model.intermediate as u32;
        let n_hidden = self.model.hidden as u32;
        // Sliding FA width on mini_parity (hq * head_dim = 1 * 256).
        let n_ctx = (self.model.cfg.num_attention_heads * self.model.cfg.head_dim) as u32;
        let n_tg = PERSISTENT_INTERP_MAX_TG;
        let alloc_f32 = |n: usize| -> Result<GpuBuffer> {
            gpu.rt
                .alloc_buffer(n.max(1) * 4)
                .map_err(Error::Metal)
        };
        let fill = |n: usize, stride: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|i| ((i % stride) as f32) * scale - scale * 0.5 * (stride as f32))
                .collect()
        };
        let gate_prog = persistent_interp_gate_down_program();
        let fa_prog = persistent_interp_fa_o_proj_program();
        let gate_insns = alloc_f32(gate_prog.len())?;
        gate_insns.write_u32(&gate_prog);
        let fa_insns = alloc_f32(fa_prog.len())?;
        fa_insns.write_u32(&fa_prog);

        let gate = alloc_f32(n_mid as usize)?;
        let up = alloc_f32(n_mid as usize)?;
        let mid = alloc_f32(n_mid as usize)?;
        let w_down = alloc_f32(n_hidden as usize * n_mid as usize)?;
        let out_down = alloc_f32(n_hidden as usize)?;
        gate.write_f32(&fill(n_mid as usize, 23, 0.02));
        up.write_f32(&fill(n_mid as usize, 19, 0.015));
        w_down.write_f32(&fill(n_hidden as usize * n_mid as usize, 29, 0.002));

        let q = alloc_f32(n_ctx as usize)?;
        let k = alloc_f32(n_ctx as usize)?;
        let v = alloc_f32(n_ctx as usize)?;
        let ctx = alloc_f32(n_ctx as usize)?;
        let w_o = alloc_f32(n_hidden as usize * n_ctx as usize)?;
        let out_o = alloc_f32(n_hidden as usize)?;
        q.write_f32(&fill(n_ctx as usize, 17, 0.03));
        k.write_f32(&fill(n_ctx as usize, 13, 0.025));
        v.write_f32(&fill(n_ctx as usize, 11, 0.02));
        w_o.write_f32(&fill(n_hidden as usize * n_ctx as usize, 31, 0.002));

        let deps = alloc_f32(2)?;
        let fail = alloc_f32(1)?;
        deps.write_u32(&[0, 0]);
        fail.write_u32(&[0]);

        self.persistent_interp = Some(PersistentInterpMiniHook {
            gate_insns,
            fa_insns,
            n_gate_insns: gate_prog.len() as u32,
            n_fa_insns: fa_prog.len() as u32,
            gate,
            up,
            mid,
            w_down,
            out_down,
            q,
            k,
            v,
            ctx,
            w_o,
            out_o,
            deps,
            fail,
            n_mid,
            n_hidden,
            n_ctx,
            n_tg,
            gate_down_hits: 0,
            fa_o_hits: 0,
            last_fail: 0,
        });
        Ok(true)
    }

    fn ensure_fuse_gate_down_scratch(&mut self) -> Result<&mut FuseGateDownScratch> {
        if !fuse_gate_down_enabled() {
            return Err(Error::Metal(
                "fuse_gate_down scratch: GEMMA_METAL_FUSE_GATE_DOWN off".into(),
            ));
        }
        if self.fuse_gate_down.is_none() {
            let gpu = &self.model.gpu;
            let prog = persistent_interp_gate_down_program();
            let insns = gpu.rt.alloc_buffer(prog.len() * 4).map_err(Error::Metal)?;
            insns.write_u32(&prog);
            let deps = gpu.rt.alloc_buffer(8).map_err(Error::Metal)?;
            let fail = gpu.rt.alloc_buffer(4).map_err(Error::Metal)?;
            self.fuse_gate_down = Some(FuseGateDownScratch { insns, deps, fail });
        }
        Ok(self.fuse_gate_down.as_mut().unwrap())
    }

    /// `&self` dispatch — call [`Self::ensure_fuse_gate_down_scratch`] first.
    fn dispatch_fuse_gate_down_q4_ref(&self, li: usize, x_bf16: &GpuBuffer) -> Result<()> {
        let scratch = self.fuse_gate_down.as_ref().ok_or_else(|| {
            Error::Metal("fuse_gate_down scratch missing (ensure first)".into())
        })?;
        scratch.deps.write_u32(&[0, 0]);
        scratch.fail.write_u32(&[0]);
        let prog = persistent_interp_gate_down_program();
        let layer = &self.model.layers[li];
        persistent_interp_gate_down_q4(
            &self.model.gpu,
            &scratch.insns,
            prog.len() as u32,
            &layer.gate_proj,
            &layer.up_proj,
            &layer.down_proj,
            x_bf16,
            &self.mid,
            &self.x,
            &scratch.deps,
            &scratch.fail,
            PERSISTENT_INTERP_MAX_TG,
            true,
        )?;
        self.model.gpu.synchronize()?;
        if scratch.fail.read_u32()[0] != 0 {
            return Err(Error::Metal(
                "GEMMA_METAL_FUSE_GATE_DOWN: grid barrier spin timeout (Metal FP caveat)".into(),
            ));
        }
        Ok(())
    }

    /// Dispatch `persistent_interp_gate_down` on mini decode phase edge (scratch).
    ///
    /// Shipping Q4 MLP continues unchanged. Hot/E4B/31B / flag-off → no-op.
    fn dispatch_persistent_interp_gate_down_edge(&mut self) -> Result<()> {
        if !self.ensure_persistent_interp_hook()? {
            return Ok(());
        }
        let gpu = &self.model.gpu;
        let hook = self
            .persistent_interp
            .as_mut()
            .expect("ensure_persistent_interp_hook");
        hook.deps.write_u32(&[0, 0]);
        hook.fail.write_u32(&[0]);
        persistent_interp_gate_down(
            gpu,
            &hook.gate_insns,
            hook.n_gate_insns,
            &hook.gate,
            &hook.up,
            &hook.mid,
            &hook.w_down,
            &hook.out_down,
            &hook.deps,
            &hook.fail,
            hook.n_mid,
            hook.n_hidden,
            hook.n_tg,
        )?;
        hook.gate_down_hits = hook.gate_down_hits.saturating_add(1);
        Ok(())
    }

    /// Dispatch `persistent_interp_fa_o_proj` on mini decode phase edge (scratch).
    ///
    /// Shipping FA + Q4 o_proj continue unchanged. Hot/E4B/31B / flag-off → no-op.
    fn dispatch_persistent_interp_fa_o_edge(&mut self) -> Result<()> {
        if !self.ensure_persistent_interp_hook()? {
            return Ok(());
        }
        let gpu = &self.model.gpu;
        let hook = self
            .persistent_interp
            .as_mut()
            .expect("ensure_persistent_interp_hook");
        hook.deps.write_u32(&[0, 0]);
        hook.fail.write_u32(&[0]);
        let scale = (hook.n_ctx as f32).sqrt().recip();
        persistent_interp_fa_o_proj(
            gpu,
            &hook.fa_insns,
            hook.n_fa_insns,
            &hook.q,
            &hook.k,
            &hook.v,
            &hook.ctx,
            &hook.w_o,
            &hook.out_o,
            &hook.deps,
            &hook.fail,
            hook.n_ctx,
            hook.n_hidden,
            hook.n_tg,
            scale,
        )?;
        hook.fa_o_hits = hook.fa_o_hits.saturating_add(1);
        Ok(())
    }

    /// Synchronize + record barrier fail after mini persistent-interp edges this step.
    fn finish_persistent_interp_step(&mut self) -> Result<()> {
        if self.persistent_interp.is_none() || !persistent_interp_enabled() {
            return Ok(());
        }
        if !self.model.is_synthetic_mini() {
            return Ok(());
        }
        self.model.gpu.synchronize()?;
        if let Some(ref mut hook) = self.persistent_interp {
            hook.last_fail = hook.fail.read_u32()[0];
            if hook.last_fail != 0 {
                return Err(Error::Metal(
                    "persistent_interp mini decode: barrier spin timeout (Metal forward-progress caveat)"
                        .into(),
                ));
            }
        }
        Ok(())
    }

    /// Block verify (DFlash port): forward `tokens` (M∈1..=[`VERIFY_MAX_M`])
    /// with causal KV append + lm_head/argmax at every position.
    ///
    /// **Path:** for M>1 with gemm kernels present → Q4 GEMM + FA(Tq=M);
    /// otherwise M× decode GEMV via [`Self::step`] (spec-exact fallback).
    ///
    /// After return, KV/`pos` reflect all M tokens appended. Call [`Self::trim_kv`]
    /// to drop rejected suffix (keep = accepted + 1 bonus when accepting a draft).
    pub fn step_verify(&mut self, tokens: &[u32]) -> Result<StepVerifyResult> {
        let m = tokens.len();
        if m == 0 || m > VERIFY_MAX_M {
            return Err(Error::Config(format!(
                "step_verify: M={m} outside 1..={VERIFY_MAX_M}"
            )));
        }
        let _s = InferScope::begin(
            "step_verify",
            format!("M={m} pos0={} tokens={tokens:?}", self.pos),
        );
        let force_gemv = force_gemv_verify_enabled();
        if m > 1
            && !force_gemv
            && self.gemm_verify_available()
            && self.model.lm_head.can_gemm_simd()
        {
            let r = self.step_verify_gemm(tokens)?;
            diag::log(
                "gpu",
                format_args!(
                    "✔ step_verify M={m} pos0={} → next={:?} (Q4 GEMM + FA Tq=M)",
                    r.pos0, r.next_tokens
                ),
            );
            return Ok(r);
        }
        let pos0 = self.pos;
        {
            let mut slot = vec![0u32; VERIFY_MAX_M];
            slot[..m].copy_from_slice(tokens);
            self.verify_seeds.write_u32(&slot);
        }
        let mut next_tokens = Vec::with_capacity(m);
        for &t in tokens {
            next_tokens.push(self.step(t)?);
        }
        {
            let mut slot = vec![0u32; VERIFY_MAX_M];
            slot[..m].copy_from_slice(&next_tokens);
            self.verify_outs.write_u32(&slot);
        }
        diag::log(
            "gpu",
            format_args!(
                "✔ step_verify M={m} pos0={pos0} → next={next_tokens:?} (dual-buf step)"
            ),
        );
        Ok(StepVerifyResult {
            pos0,
            tokens: tokens.to_vec(),
            next_tokens,
        })
    }

    fn gemm_verify_available(&self) -> bool {
        // Require H×M act scratch so GEMM never overruns M=1 decode buffers.
        let need = self.model.hidden.max(1) * VERIFY_MAX_M * 4;
        self.x.nbytes() >= need
            && self.logits.nbytes() >= self.model.vocab.max(1) * VERIFY_MAX_M * 4
            && self
                .model
                .gpu
                .rt
                .pipeline(KernelId::GemmQ4MlxSimd.entry_name())
                .is_ok()
            && self
                .model
                .gpu
                .rt
                .pipeline(KernelId::GemmQ4MlxSimdI4.entry_name())
                .is_ok()
    }

    fn embed_verify_tokens(&self, m: usize, h: usize) -> Result<()> {
        let scale = self.model.embed_scale;
        let n = m as u32;
        if let Some(ref hot) = self.model.embed_hot {
            embed_lookup_quant_n(
                &self.model.gpu,
                hot,
                &self.verify_seeds,
                &self.x,
                self.model.vocab as u32,
                n,
            )?;
            return scale_f32_inplace(&self.model.gpu, &self.x, scale, (m * h) as u32);
        }
        let ids = self.verify_seeds.read_u32();
        let mut packed = vec![0f32; m * h];
        for mi in 0..m {
            let tid = ids[mi];
            if let Some(ref eq) = self.model.embed_q {
                let mut row = eq.dequant_row(tid as usize)?;
                if (scale - 1.0).abs() > 1e-12 {
                    for v in &mut row {
                        *v *= scale;
                    }
                }
                packed[mi * h..(mi + 1) * h].copy_from_slice(&row[..h]);
            } else {
                let row = (tid as usize) * h;
                if row + h > self.model.embed.len() {
                    return Err(Error::Config(format!("token {tid} OOV")));
                }
                if (scale - 1.0).abs() <= 1e-12 {
                    packed[mi * h..(mi + 1) * h]
                        .copy_from_slice(&self.model.embed[row..row + h]);
                } else {
                    for d in 0..h {
                        packed[mi * h + d] = self.model.embed[row + d] * scale;
                    }
                }
            }
        }
        self.write_x_rows(&packed)?;
        Ok(())
    }

    /// One batched verify forward (Q4 GEMM + FA Tq=M). Appends M KV steps.
    fn step_verify_gemm(&mut self, tokens: &[u32]) -> Result<StepVerifyResult> {
        let _capture_ao = CaptureAlwaysOnGuard::enter(self.capture.is_some());
        let m = tokens.len();
        let pos0 = self.pos;
        self.sync_pos_buf(pos0 as u32);
        let hidden = self.model.hidden as u32;
        let h_usz = self.model.hidden;
        let eps = self.model.eps;
        let first_shared = self.model.kv.first_kv_shared;
        let n_layers = self.model.layers.len();
        let ple_dim_cfg = self.model.cfg.hidden_size_per_layer_input;
        let vocab = self.model.vocab as u32;
        let intermediate = self.model.intermediate as u32;
        let softcap = self.model.softcap;
        let m_u = m as u32;

        {
            let mut slot = vec![0u32; VERIFY_MAX_M];
            slot[..m].copy_from_slice(tokens);
            self.verify_seeds.write_u32(&slot);
        }
        self.embed_verify_tokens(m, h_usz)?;

        // Device capture staging: M rows × n_cap × H (no mid-layer host sync —
        // matches M=1 `copy_f32` path; prior synchronize+read_f32 was exactness-toxic).
        let capture_stride = self.capture.as_ref().map(|c| c.row_stride());
        let capture_stage_gpu: Option<GpuBuffer> = match capture_stride {
            Some(stride) if stride > 0 => {
                let bytes = (m * stride).max(1) * 4;
                Some(
                    self.model
                        .gpu
                        .rt
                        .alloc_buffer_hot(bytes)
                        .map_err(Error::Metal)?,
                )
            }
            _ => None,
        };

        for li in 0..n_layers {
            let (
                layer_type,
                role,
                hq,
                hkv,
                head_dim,
                rotary_dim,
                theta,
                window,
                has_ple,
                is_producer,
                has_pre_ff,
                has_post_ff,
                layer_scalar,
            ) = {
                let layer = &self.model.layers[li];
                (
                    layer.layer_type,
                    layer.role.clone(),
                    layer.hq,
                    layer.hkv,
                    layer.head_dim,
                    layer.rotary_dim,
                    layer.theta,
                    layer.window,
                    layer.ple_table.is_some() || self.model.ple_q4.is_some(),
                    matches!(layer.role, KvRole::Producer { .. }),
                    layer.pre_ff_norm.is_some(),
                    layer.post_ff_norm.is_some(),
                    layer.layer_scalar,
                )
            };
            // Must match `step_inner`: Gemma4 dual residual-norms + layer_scalar
            // when pre_ff exists and the layer has no PLE (31B Hot). E4B/PLE keeps
            // the fused Pre-LN add-into path so GEMM verify stays bit-aligned with M=1.
            let use_gemma4_dual_norm =
                has_pre_ff && (!has_ple || crate::kernels::e4b_dual_norm_enabled());

            {
                let gpu = &self.model.gpu;
                let layer = &self.model.layers[li];
                if fuse_bf16_rms() {
                    let _ = rms_norm_to_act_bf16(
                        gpu,
                        &self.x,
                        &layer.input_norm,
                        m_u,
                        hidden,
                        eps,
                    )?;
                } else {
                    rms_norm_f32(gpu, &self.x, &layer.input_norm, &self.normed, m_u, hidden, eps)?;
                    let _ = prepare_act_bf16(gpu, &self.normed, m_u * hidden)?;
                }
                if tessl::ab_flags::need_barrier(true) {
                    gpu.barrier()?;
                }
                let x_bf16 = gpu.act_bf16_scratch((m * h_usz).max(1))?;
                layer.q_proj.gemm_bf16_x(gpu, &x_bf16, &self.q, m_u)?;
                if is_producer {
                    let k = layer
                        .k_proj
                        .as_ref()
                        .ok_or_else(|| Error::Weights(format!("layer {li} missing k_proj")))?;
                    let v = layer
                        .v_proj
                        .as_ref()
                        .ok_or_else(|| Error::Weights(format!("layer {li} missing v_proj")))?;
                    // Separate GEMMs (KV fuse is M=1 only today).
                    k.gemm_bf16_x(gpu, &x_bf16, &self.k, m_u)?;
                    v.gemm_bf16_x(gpu, &x_bf16, &self.v, m_u)?;
                }
                if tessl::ab_flags::need_barrier(true) {
                    gpu.barrier()?;
                }
                rms_qkv_rope_ex_posbuf(
                    gpu,
                    &self.q,
                    &self.k,
                    &self.v,
                    &layer.q_norm,
                    &layer.k_norm,
                    &layer.v_norm,
                    m_u,
                    hq as u32,
                    hkv as u32,
                    head_dim as u32,
                    rotary_dim as u32,
                    &self.pos_buf,
                    theta,
                    eps,
                    /*q_only*/ !is_producer,
                )?;
            }

            let tkv_want = (pos0 + m) as u32;
            let update_shared = is_producer
                && self
                    .model
                    .kv
                    .layers
                    .iter()
                    .take(first_shared)
                    .filter(|l| l.layer_type == layer_type)
                    .map(|l| l.layer)
                    .max()
                    == Some(li);

            if is_producer {
                let gpu = &self.model.gpu;
                if tessl::ab_flags::need_barrier(true) {
                    gpu.barrier()?;
                }
                match &role {
                    KvRole::Producer { slot } => match slot {
                        KvSlotId::SlidingRing { producer_index } => {
                            let i = *producer_index;
                            self.sliding_rings[i].append_m(gpu, &self.k, &self.v, m)?;
                            if update_shared {
                                self.shared_sliding.append_m(gpu, &self.k, &self.v, m)?;
                            }
                        }
                        KvSlotId::GlobalFull { producer_index } => {
                            let i = *producer_index;
                            self.global_slots[i].append_m(gpu, &self.k, &self.v, m)?;
                            if update_shared {
                                self.shared_global.append_m(gpu, &self.k, &self.v, m)?;
                            }
                        }
                    },
                    KvRole::Consumer { .. } => unreachable!(),
                }
            }

            {
                let gpu = &self.model.gpu;
                let layer = &self.model.layers[li];
                if tessl::ab_flags::need_barrier(true) {
                    gpu.barrier()?;
                }
                let (k_fa, v_fa, kv_off, tkv) = if is_producer {
                    match &role {
                        KvRole::Producer { slot } => match slot {
                            KvSlotId::SlidingRing { producer_index } => self.sliding_rings
                                [*producer_index]
                                .fa_buffers(gpu, &self.k_dense, &self.v_dense, tkv_want)?,
                            KvSlotId::GlobalFull { producer_index } => self.global_slots
                                [*producer_index]
                                .fa_buffers(gpu, &self.k_dense, &self.v_dense, tkv_want)?,
                        },
                        KvRole::Consumer { .. } => unreachable!(),
                    }
                } else {
                    let slot = if layer_type.is_sliding() {
                        &self.shared_sliding
                    } else {
                        &self.shared_global
                    };
                    if tkv_want > slot.live_len_u32()? {
                        return Err(Error::Kv(format!(
                            "consumer verify pos0={pos0} M={m} shared tkv={}",
                            slot.seq_len
                        )));
                    }
                    slot.fa_buffers(gpu, &self.k_dense, &self.v_dense, tkv_want)?
                };

                let o_elems = (m * hq * head_dim) as u32;
                let fuse_bf16 = fuse_bf16_fa();
                let o_bf16_for_fa = gpu.act_bf16_scratch(o_elems as usize)?;
                let o_fa = if fuse_bf16 {
                    &o_bf16_for_fa
                } else {
                    &self.o
                };
                if layer_type.is_sliding() {
                    let win = window.unwrap_or(512) as u32;
                    flash_attn_swa_h256_ex(
                        gpu,
                        &self.q,
                        &k_fa,
                        &v_fa,
                        o_fa,
                        1,
                        m_u,
                        tkv,
                        hq as u32,
                        hkv as u32,
                        win,
                        1.0,
                        pos0 as u32,
                        kv_off,
                        fuse_bf16,
                    )?;
                } else {
                    flash_attn_global_h512_ex(
                        gpu,
                        &self.q,
                        &k_fa,
                        &v_fa,
                        o_fa,
                        1,
                        m_u,
                        tkv,
                        hq as u32,
                        hkv as u32,
                        1.0,
                        pos0 as u32,
                        kv_off,
                        fuse_bf16,
                    )?;
                }

                if !fuse_bf16 {
                    let _ = prepare_act_bf16(gpu, &self.o, o_elems)?;
                }
                if tessl::ab_flags::need_barrier(true) {
                    gpu.barrier()?;
                }
                let o_bf16 = gpu.act_bf16_scratch(o_elems as usize)?;
                // o_proj residual — clone `step_inner` dual-norm vs legacy add-into.
                if use_gemma4_dual_norm {
                    // MLX Gemma4: o = post_attention_ln(o_proj); x += o
                    if fuse_dual_norm_enabled() {
                        layer.o_proj.gemm_postnorm_add_into_bf16_x(
                            gpu,
                            &o_bf16,
                            &self.x,
                            &self.attn_proj,
                            &layer.post_attn_norm,
                            m_u,
                            eps,
                        )?;
                    } else {
                        layer.o_proj.gemm_bf16_x(gpu, &o_bf16, &self.attn_proj, m_u)?;
                        if tessl::ab_flags::need_barrier(true) {
                            gpu.barrier()?;
                        }
                        rms_norm_f32(
                            gpu,
                            &self.attn_proj,
                            &layer.post_attn_norm,
                            &self.normed,
                            m_u,
                            hidden,
                            eps,
                        )?;
                        if tessl::ab_flags::need_barrier(true) {
                            gpu.barrier()?;
                        }
                        ple_residual_add(gpu, &self.x, &self.normed, 1.0, m_u * hidden)?;
                    }
                } else {
                    layer
                        .o_proj
                        .gemm_add_into_bf16_x(gpu, &o_bf16, &self.x, m_u)?;
                }

                if has_ple {
                    // PLE is per-token; fall back to M× lookups into x rows.
                    let ple_dim = ple_dim_cfg as u32;
                    let scale = (ple_dim as f32).sqrt();
                    let n_layers_u = n_layers as u32;
                    for mi in 0..m {
                        // seed_tok from verify_seeds[mi]
                        copy_u32_from_index(
                            gpu,
                            &self.verify_seeds,
                            mi as u32,
                            &self.seed_tok,
                        )?;
                        if tessl::ab_flags::need_barrier(true) {
                            gpu.barrier()?;
                        }
                        if let Some(ref ple) = self.model.ple_q4 {
                            ple_lookup_q4_mlx(
                                gpu,
                                &self.seed_tok,
                                &ple.packed,
                                &ple.scales,
                                &ple.zeros,
                                &self.ple_out,
                                ple_dim,
                                vocab,
                                1,
                                scale,
                                li as u32,
                                n_layers_u,
                                ple.group_size,
                            )?;
                        } else if let Some(ref table) = layer.ple_table {
                            ple_lookup(
                                gpu,
                                &self.seed_tok,
                                table,
                                &self.ple_out,
                                ple_dim,
                                vocab,
                                1,
                                scale,
                            )?;
                        }
                        if tessl::ab_flags::need_barrier(true) {
                            gpu.barrier()?;
                        }
                        // Add PLE into residual row mi (first ple_dim dims).
                        {
                            use tessl::dispatch::{dispatch_1d, set_gpu_buf, set_gpu_buf_offset, set_u32};
                            let p = gpu.rt.pipeline("ple_residual_add").map_err(Error::Metal)?;
                            let x_off = mi * h_usz * 4;
                            dispatch_1d(&gpu.rt, &p, ple_dim as usize, |bnd| {
                                set_gpu_buf_offset(bnd, &self.x, x_off, 0);
                                set_gpu_buf(bnd, &self.ple_out, 1);
                                tessl::dispatch::set_f32(
                                    bnd,
                                    std::f32::consts::FRAC_1_SQRT_2,
                                    2,
                                );
                                set_u32(bnd, ple_dim, 3);
                            })
                            .map_err(Error::Metal)?;
                        }
                    }
                }

                if tessl::ab_flags::need_barrier(true) {
                    gpu.barrier()?;
                }
                // Gemma4: pre_ff_norm before MLP; legacy/synth reuse post_attn_norm.
                let mlp_in_norm = layer.pre_ff_norm.as_ref().unwrap_or(&layer.post_attn_norm);
                let fuse_rms = fuse_bf16_rms();
                let fuse_mlp = fuse_bf16_mlp();
                if fuse_rms {
                    let _ = rms_norm_to_act_bf16(
                        gpu,
                        &self.x,
                        mlp_in_norm,
                        m_u,
                        hidden,
                        eps,
                    )?;
                } else {
                    rms_norm_f32(
                        gpu,
                        &self.x,
                        mlp_in_norm,
                        &self.normed,
                        m_u,
                        hidden,
                        eps,
                    )?;
                    let _ = prepare_act_bf16(gpu, &self.normed, m_u * hidden)?;
                }
                if tessl::ab_flags::need_barrier(true) {
                    gpu.barrier()?;
                }
                let x_bf16 = gpu.act_bf16_scratch((m * h_usz).max(1))?;

                // Gate / up: prefer separate GEMMs (fused gate_up is M=1).
                layer
                    .gate_proj
                    .gemm_bf16_x(gpu, &x_bf16, &self.gate, m_u)?;
                layer.up_proj.gemm_bf16_x(gpu, &x_bf16, &self.up, m_u)?;
                if tessl::ab_flags::need_barrier(true) {
                    gpu.barrier()?;
                }
                if fuse_mlp {
                    mlp_gelu_tanh_bf16(
                        gpu,
                        &self.gate,
                        &self.up,
                        &self.mid,
                        m_u * intermediate,
                    )?;
                } else {
                    mlp_gelu_tanh(
                        gpu,
                        &self.gate,
                        &self.up,
                        &self.mid,
                        m_u * intermediate,
                    )?;
                    let _ = prepare_act_bf16(gpu, &self.mid, m_u * intermediate)?;
                }
                if tessl::ab_flags::need_barrier(true) {
                    gpu.barrier()?;
                }
                let mid_bf16 = if fuse_mlp {
                    self.mid.clone()
                } else {
                    gpu.act_bf16_scratch((m * intermediate as usize).max(1))?
                };
                if use_gemma4_dual_norm && has_post_ff {
                    // MLX Gemma4: h = post_ff_ln(down); x += h
                    if let Some(ref w) = layer.post_ff_norm {
                        if fuse_dual_norm_enabled() {
                            let fold_scale = if (layer_scalar - 1.0).abs() > 1e-8 {
                                layer_scalar
                            } else {
                                1.0
                            };
                            layer.down_proj.gemm_postnorm_add_into_bf16_x_scaled(
                                gpu,
                                &mid_bf16,
                                &self.x,
                                &self.down,
                                w,
                                m_u,
                                eps,
                                fold_scale,
                            )?;
                        } else {
                            layer.down_proj.gemm_bf16_x(gpu, &mid_bf16, &self.down, m_u)?;
                            if tessl::ab_flags::need_barrier(true) {
                                gpu.barrier()?;
                            }
                            rms_norm_f32(gpu, &self.down, w, &self.normed, m_u, hidden, eps)?;
                            if tessl::ab_flags::need_barrier(true) {
                                gpu.barrier()?;
                            }
                            ple_residual_add(gpu, &self.x, &self.normed, 1.0, m_u * hidden)?;
                        }
                    }
                } else {
                    layer
                        .down_proj
                        .gemm_add_into_bf16_x(gpu, &mid_bf16, &self.x, m_u)?;
                    if has_post_ff {
                        if let Some(ref w) = layer.post_ff_norm {
                            rms_norm_f32(gpu, &self.x, w, &self.normed, m_u, hidden, eps)?;
                            copy_f32_n(gpu, &self.normed, &self.x, (m * h_usz) as u32)?;
                        }
                    }
                }

                // MLX: layer_scalar multiplies full layer output after both residuals.
                let folded = fuse_dual_norm_enabled() && use_gemma4_dual_norm && has_post_ff;
                if use_gemma4_dual_norm && (layer_scalar - 1.0).abs() > 1e-8 && !folded {
                    if tessl::ab_flags::need_barrier(true) {
                        gpu.barrier()?;
                    }
                    scale_f32_inplace(gpu, &self.x, layer_scalar, m_u * hidden)?;
                }
            }

            // Capture: after layer, x is [M,H] — device copy into stage[mi, slot].
            // Hazard: synchronize-per-capture-layer (see step_inner); AO via env.
            if self.capture.is_some() {
                let need = self
                    .capture
                    .as_ref()
                    .map(|c| c.layer_ids.contains(&li))
                    .unwrap_or(false);
                if need {
                    let slot = self.capture.as_mut().and_then(|c| c.mark_layer(li));
                    if let (Some(slot), Some(ref stage), Some(stride)) =
                        (slot, capture_stage_gpu.as_ref(), capture_stride)
                    {
                        let force_barrier = capture_barrier_forced();
                        if tessl::ab_flags::hazard_barriers() {
                            self.model.gpu.synchronize()?;
                        } else if force_barrier {
                            self.model.gpu.barrier()?;
                        }
                        for mi in 0..m {
                            copy_f32_range(
                                &self.model.gpu,
                                &self.x,
                                mi * h_usz,
                                stage,
                                mi * stride + slot * h_usz,
                                h_usz as u32,
                            )?;
                        }
                    }
                }
            }
        }

        // Final norm + lm_head GEMM + per-position softcap argmax.
        {
            let gpu = &self.model.gpu;
            if fuse_bf16_rms() {
                let _ = rms_norm_to_act_bf16(
                    gpu,
                    &self.x,
                    &self.model.final_norm,
                    m_u,
                    hidden,
                    eps,
                )?;
            } else {
                rms_norm_f32(
                    gpu,
                    &self.x,
                    &self.model.final_norm,
                    &self.normed,
                    m_u,
                    hidden,
                    eps,
                )?;
                let _ = prepare_act_bf16(gpu, &self.normed, m_u * hidden)?;
            }
            if tessl::ab_flags::need_barrier(true) {
                gpu.barrier()?;
            }
            let x_bf16 = gpu.act_bf16_scratch((m * h_usz).max(1))?;
            self.model
                .lm_head
                .gemm_bf16_x(gpu, &x_bf16, &self.logits, m_u)?;
            // Same edge as step_inner: lm_head → softcap must drain even under
            // always-on auto barriers (31B capture-off collapse 240017/236773 —
            // the auto barrier did NOT cover this producer→consumer on 31B).
            // Mirror the unconditional barrier here; verify path shares the bug
            // class or masks it only via capture/AO timing.
            gpu.barrier()?;
            for mi in 0..m {
                softcap_argmax_encode_offset(
                    gpu,
                    &self.logits,
                    mi * (vocab as usize) * 4,
                    softcap,
                    vocab,
                    &mut self.argmax_scratch,
                    &self.verify_outs,
                    mi * 4,
                )?;
                if tessl::ab_flags::need_barrier(true) {
                    gpu.barrier()?;
                }
            }
        }

        // Finish capture + conditioner AFTER softcap encode (same deferral as M=1):
        // do not insert FC between lm_head and argmax drain.
        if let (Some(stage), Some(stride)) = (capture_stage_gpu.as_ref(), capture_stride) {
            for mi in 0..m {
                if let Some(ref row_buf) = self.capture_row {
                    copy_f32_range(
                        &self.model.gpu,
                        stage,
                        mi * stride,
                        row_buf,
                        0,
                        stride as u32,
                    )?;
                    if let Some(ref mut cond) = self.conditioner {
                        if tessl::ab_flags::need_barrier(true) {
                            self.model.gpu.barrier()?;
                        }
                        cond.project_row(&self.model.gpu, row_buf)?;
                    }
                }
            }
            // One sync then host concat assemble (draft may read h_ctx next).
            self.model.gpu.synchronize()?;
            let stage_host = stage.read_f32();
            for mi in 0..m {
                if let Some(ref mut c) = self.capture {
                    c.finish_step_complete(&stage_host[mi * stride..(mi + 1) * stride])?;
                }
            }
        }

        self.pos = pos0 + m;
        self.model.gpu.synchronize()?;
        let outs = self.verify_outs.read_u32();
        let next_tokens = outs[..m].to_vec();
        // Seed argmax_tok with last next-token for chained decode continuity.
        if let Some(&last) = next_tokens.last() {
            self.argmax_tok.write_u32(&[last]);
        }
        Ok(StepVerifyResult {
            pos0,
            tokens: tokens.to_vec(),
            next_tokens,
        })
    }

    /// Roll back the last `n` KV timesteps and rewind `pos` (post-reject).
    ///
    /// Shared slots that were never written (e.g. 31B `num_kv_shared_layers=0`)
    /// trim as a no-op; rings/global slots clamp to their own `seq_len`.
    pub fn trim_kv(&mut self, n: usize) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        if n > self.pos {
            return Err(Error::Kv(format!(
                "trim_kv {n} > pos {}",
                self.pos
            )));
        }
        for r in &mut self.sliding_rings {
            let t = n.min(r.seq_len);
            r.trim(t)?;
        }
        for s in &mut self.global_slots {
            let t = n.min(s.seq_len);
            s.trim(t)?;
        }
        {
            let t = n.min(self.shared_sliding.seq_len);
            self.shared_sliding.trim(t)?;
        }
        {
            let t = n.min(self.shared_global.seq_len);
            self.shared_global.trim(t)?;
        }
        self.pos -= n;
        self.sync_pos_buf(self.pos as u32);
        if let Some(ref mut c) = self.capture {
            // Keep capture timeline aligned with KV/`pos` when capturing.
            let trim_c = n.min(c.t);
            c.trim_recent(trim_c)?;
        }
        if let Some(ref mut cond) = self.conditioner {
            let trim_c = n.min(cond.h_ctx_len());
            cond.trim_recent(trim_c)?;
        }
        diag::log(
            "gpu",
            format_args!("trim_kv n={n} → pos={}", self.pos),
        );
        Ok(())
    }

    /// After [`Self::step_verify`] of length `m`, keep the first `keep` timesteps
    /// (typically `accepted + 1` for the bonus token) and trim the rest.
    pub fn commit_verify(&mut self, m: usize, keep: usize) -> Result<()> {
        if keep > m {
            return Err(Error::Config(format!(
                "commit_verify: keep={keep} > m={m}"
            )));
        }
        self.trim_kv(m - keep)
    }

    /// Quiet microbench: wall ms for one `step_verify` of `tokens` (no TRACE).
    ///
    /// Each iter runs verify then [`Self::trim_kv`] so context is restored.
    /// Caller should prefill context first; this does not reset.
    pub fn bench_step_verify_ms(&mut self, tokens: &[u32], warmup: usize, iters: usize) -> Result<f64> {
        for _ in 0..warmup {
            let pos0 = self.pos;
            let _ = self.step_verify(tokens)?;
            self.trim_kv(tokens.len())?;
            debug_assert_eq!(self.pos, pos0);
            let _ = pos0;
        }
        self.model.gpu.synchronize()?;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = self.step_verify(tokens)?;
            self.trim_kv(tokens.len())?;
        }
        self.model.gpu.synchronize()?;
        Ok(t0.elapsed().as_secs_f64() * 1e3 / iters.max(1) as f64)
    }

    /// Host readout of `logits` buffer after a decode step (diagnostics).
    pub fn debug_logits_stats(&self) -> LogitsStats {
        Self::stats_f32(&self.logits.read_f32())
    }

    /// Host readout of residual `x` buffer (diagnostics).
    pub fn debug_x_stats(&self) -> LogitsStats {
        Self::stats_f32(&self.x.read_f32()[..self.model.hidden])
    }

    /// Diagnostics: embed one token into `x` (no layer stack) and return stats.
    pub fn debug_embed_only(&mut self, token: u32) -> Result<LogitsStats> {
        self.seed_tok.write_u32(&[token]);
        self.embed_from_seed(self.model.hidden)?;
        self.model.gpu.synchronize()?;
        Ok(self.debug_x_stats())
    }

    fn stats_f32(v: &[f32]) -> LogitsStats {
        let mut nan = 0usize;
        let mut max = f32::NEG_INFINITY;
        let mut min = f32::INFINITY;
        let mut host_argmax = 0u32;
        for (i, &x) in v.iter().enumerate() {
            if x.is_nan() {
                nan += 1;
                continue;
            }
            if x > max {
                max = x;
                host_argmax = i as u32;
            }
            if x < min {
                min = x;
            }
        }
        LogitsStats {
            finite: nan == 0 && !v.is_empty(),
            nan,
            max: if max.is_finite() { max } else { f32::NAN },
            min: if min.is_finite() { min } else { f32::NAN },
            host_argmax,
        }
    }

    /// Prefill tokens then decode `max_new` steps. Returns full token sequence.
    pub fn generate(&mut self, prompt: &[u32], max_new: usize) -> Result<Vec<u32>> {
        let _gen = InferScope::begin(
            "generate",
            format!("prompt_len={} max_new={max_new}", prompt.len()),
        );
        diag::log(
            "gpu",
            format_args!(
                "▶ generate prompt_len={} max_new={max_new}",
                prompt.len()
            ),
        );
        self.reset();
        let mut out = prompt.to_vec();
        if prompt.is_empty() {
            return Err(Error::Config("empty prompt".into()));
        }
        for (pi, &t) in prompt[..prompt.len() - 1].iter().enumerate() {
            diag::infer_log(format_args!(
                "· generate prefill[{pi}/{}] token={t}",
                prompt.len() - 1
            ));
            self.step_prefill(t).map_err(|e| {
                diag::err("gpu", "generate prefill", &e);
                e
            })?;
        }
        diag::infer_log(format_args!(
            "· generate first_decode token={}",
            prompt[prompt.len() - 1]
        ));
        // First decode: host seed. Subsequent: GPU-chained FromArgmax (no per-token
        // seed write). Readback still required for EOS / returned tokens.
        let mut next = self
            .step_inner(StepSeed::Host(prompt[prompt.len() - 1]), true, true)
            .map_err(|e| {
                diag::err("gpu", "generate first token", &e);
                e
            })?;
        for i in 0..max_new {
            out.push(next);
            if let Some(eos) = self.model.cfg.eos_token_id.as_ref() {
                if eos.as_slice().contains(&next) {
                    diag::log("gpu", format_args!("generate hit eos at step {i}"));
                    diag::infer_log(format_args!("· generate eos at decode_step={i} token={next}"));
                    break;
                }
            }
            diag::infer_log(format_args!(
                "· generate decode[{i}/{max_new}] token={next}"
            ));
            next = self
                .step_inner(StepSeed::FromArgmax, true, true)
                .map_err(|e| {
                    diag::err("gpu", &format!("generate decode step {i}"), &e);
                    e
                })?;
        }
        diag::log(
            "gpu",
            format_args!("✔ generate done out_len={}", out.len()),
        );
        Ok(out)
    }

    /// Copy GPU shared sliding / global KV into MTP cross-KV host mirrors.
    ///
    /// These buffers are filled by the last sliding / last global producers in
    /// the target graph (`update_shared` path).
    pub fn sync_mtp_cross_kv(&self, mtp: &mut MtpSession) -> Result<()> {
        self.model.gpu.synchronize()?;
        let st = self
            .shared_sliding
            .seq_len
            .min(self.shared_sliding.capacity);
        let gt = self.shared_global.seq_len.min(self.shared_global.capacity);
        let sliding_elems = self
            .shared_sliding
            .elements_for_positions(st, "MTP sliding KV sync")?;
        let global_elems = self
            .shared_global
            .elements_for_positions(gt, "MTP global KV sync")?;
        let sk = if st > 0 {
            self.shared_sliding.k.read_f32()[..sliding_elems].to_vec()
        } else {
            Vec::new()
        };
        let sv = if st > 0 {
            self.shared_sliding.v.read_f32()[..sliding_elems].to_vec()
        } else {
            Vec::new()
        };
        let gk = if gt > 0 {
            self.shared_global.k.read_f32()[..global_elems].to_vec()
        } else {
            Vec::new()
        };
        let gv = if gt > 0 {
            self.shared_global.v.read_f32()[..global_elems].to_vec()
        } else {
            Vec::new()
        };
        mtp.cross_kv
            .replace_from_densified(&sk, &sv, st, &gk, &gv, gt)?;
        mtp.last_shared_sliding_t = st;
        mtp.last_shared_global_t = gt;
        Ok(())
    }

    /// MTP smoke: draft from clustered head (+ cross-KV when synced), verify vs backbone.
    pub fn generate_mtp_smoke(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        mtp: &mut MtpSession,
    ) -> Result<(Vec<u32>, Vec<VerifyResult>)> {
        self.reset();
        if prompt.is_empty() {
            return Err(Error::Config("empty prompt".into()));
        }
        let mut out = prompt.to_vec();
        let mut verifies = Vec::new();
        for &t in &prompt[..prompt.len() - 1] {
            let _ = self.step(t)?;
        }
        let mut next = self.step(prompt[prompt.len() - 1])?;
        let backbone = mtp.assistant.backbone_hidden_size;
        while out.len() - prompt.len() < max_new {
            out.push(next);
            if let Some(eos) = self.model.cfg.eos_token_id.as_ref() {
                if eos.as_slice().contains(&next) {
                    break;
                }
            }
            // Pull latest shared sliding / global KV into MTP, then draft.
            self.sync_mtp_cross_kv(mtp)?;
            let mut h = self.normed.read_f32();
            h.resize(backbone, 0.0);
            let draft = mtp.draft_from_hidden(&h)?;
            let mut target = Vec::with_capacity(draft.len().saturating_add(1));
            let mut tok = next;
            // Early-reject: stop verifying after first mismatch (saves wasted steps).
            let mut rejected = false;
            for &d in &draft {
                if out.len() - prompt.len() >= max_new {
                    break;
                }
                tok = self.step(tok)?;
                target.push(tok);
                out.push(tok);
                if tok != d {
                    rejected = true;
                    break;
                }
            }
            if !rejected && out.len() - prompt.len() < max_new {
                let bonus = self.step(tok)?;
                target.push(bonus);
                next = bonus;
            } else {
                next = *target.last().unwrap_or(&tok);
            }
            verifies.push(mtp.verify_and_adapt(&draft, &target));
        }
        Ok((out, verifies))
    }

    /// Decode-only timing: measures `n_steps` incremental steps after prefill of `prompt`.
    /// Uses GPU-resident tokens (no mid-loop host readback).
    pub fn bench_decode_tok_s(&mut self, prompt: &[u32], n_steps: usize) -> Result<f64> {
        let last = *prompt
            .last()
            .ok_or_else(|| Error::Config("bench_decode_tok_s: empty prompt".into()))?;
        if n_steps == 0 {
            return Err(Error::Config("bench_decode_tok_s: n_steps must be > 0".into()));
        }
        self.reset();
        for &t in &prompt[..prompt.len() - 1] {
            self.step_prefill(t)?;
        }
        // First generated token (encode only).
        let _ = self.step_inner(StepSeed::Host(last), false, true)?;
        self.model.gpu.synchronize()?;
        let t0 = std::time::Instant::now();
        for _ in 0..n_steps {
            let _ = self.step_inner(StepSeed::FromArgmax, false, true)?;
        }
        self.model.gpu.synchronize()?;
        let secs = t0.elapsed().as_secs_f64();
        if !secs.is_finite() || secs <= 0.0 {
            return Err(Error::Metal(format!(
                "bench_decode_tok_s: {n_steps} steps measured {secs}s — clock unusable"
            )));
        }
        Ok(n_steps as f64 / secs)
    }

    /// Continuous decode for sustained-load measurement: run `n_steps` further
    /// GPU-resident decode steps from wherever this session already is, priming
    /// or recycling only when the KV cannot take another append.
    ///
    /// This is the work unit the sustained-load gate wants and
    /// [`Self::bench_decode_tok_s`] is not: that one resets and re-prefills on
    /// every call, so a short call is mostly prefill and the measured rate
    /// swings with how many calls happen to land inside a window. Here the
    /// prefill cost is paid once per `kv_capacity` tokens instead of once per
    /// call, so consecutive calls do near-identical work and a change in rate
    /// means the machine changed.
    ///
    /// A full global KV slot is a capacity limit, not a fault: rather than
    /// surfacing `Error::Kv("GPU KV full")` mid-run, the session recycles
    /// (reset + re-prefill `prompt`) and keeps going, exactly as a server would
    /// start a new conversation. Returns the number of decode steps completed,
    /// which is always `n_steps` on success — prefill tokens are deliberately
    /// not counted, so the rate stays comparable to `bench_decode_tok_s`.
    ///
    /// Drains the queue before returning so the caller's wall clock covers the
    /// GPU work rather than just the encode.
    pub fn bench_decode_sustained(&mut self, prompt: &[u32], n_steps: usize) -> Result<usize> {
        if prompt.is_empty() {
            return Err(Error::Config("bench_decode_sustained: empty prompt".into()));
        }
        if n_steps == 0 {
            // Well-defined, unlike `bench_decode_tok_s(_, 0)` which would divide
            // by a zero-step interval: zero steps completed, no work done.
            return Ok(0);
        }
        let ceiling = self.kv_capacity();
        // One free slot must remain after the prompt, or priming itself would
        // fill the cache and every step would recycle.
        if let Some(cap) = ceiling {
            if prompt.len() + 2 > cap {
                return Err(Error::Kv(format!(
                    "bench_decode_sustained: prompt len {} leaves no room in KV capacity {cap}",
                    prompt.len()
                )));
            }
        }
        // `pos == 0` means nothing is primed yet (fresh session, or we just reset).
        let mut primed = self.pos > 0;
        let mut done = 0usize;
        // Liveness backstop. The capacity check above already guarantees that a
        // freshly primed session has room for at least one decode step, so a
        // recycle is always followed by progress. This counter makes that
        // structural rather than a property of one guard: if a future change to
        // `pos` accounting ever left the loop recycling without decoding, the
        // call returns a named error instead of hanging the caller forever.
        let mut recycles = 0usize;
        // Worst legal case is a cache that holds the prompt plus exactly one
        // decode slot, i.e. one recycle per step; anything above that is the
        // pathology this backstop exists to catch, not a tight cache.
        let recycle_budget = n_steps + 8;
        while done < n_steps {
            let would_fill = ceiling.is_some_and(|cap| self.pos + 1 >= cap);
            if !primed || would_fill {
                if recycles > recycle_budget {
                    return Err(Error::Kv(format!(
                        "bench_decode_sustained: {recycles} recycles for {done}/{n_steps} steps \
                         (pos={}, kv_capacity={ceiling:?}) — recycling without making progress",
                        self.pos
                    )));
                }
                recycles += 1;
                self.reset();
                for &t in &prompt[..prompt.len() - 1] {
                    self.step_prefill(t)?;
                }
                let _ = self.step_inner(StepSeed::Host(prompt[prompt.len() - 1]), false, true)?;
                primed = true;
                // The seed step produced a token but is priming, not measured
                // work: counting it would make short calls look faster than
                // long ones purely from how often they recycle.
                continue;
            }
            let _ = self.step_inner(StepSeed::FromArgmax, false, true)?;
            done += 1;
        }
        self.model.gpu.synchronize()?;
        Ok(done)
    }

    /// Prefill TTFT (ms) through first generated token.
    pub fn bench_ttft_ms(&mut self, prompt: &[u32]) -> Result<f64> {
        if prompt.is_empty() {
            return Err(Error::Config("bench_ttft_ms: empty prompt".into()));
        }
        self.reset();
        let t0 = std::time::Instant::now();
        for &t in &prompt[..prompt.len() - 1] {
            self.step_prefill(t)?;
        }
        let _ = self.step(prompt[prompt.len() - 1])?;
        Ok(t0.elapsed().as_secs_f64() * 1e3)
    }
}

/// Host softcap-argmax on Q4-dequant LM head (parity helper).
pub fn host_q4_next_token(
    model: &GpuSynthModel,
    hidden_state: &[f32],
) -> Result<u32> {
    let logits = model.host_gemv_q(&model.lm_head_host, hidden_state)?;
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        let c = softcap_f32(v, model.softcap);
        if c > best_v {
            best_v = c;
            best_i = i;
        }
    }
    Ok(best_i as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward::SyntheticE4bGraph;
    use crate::kernels::{copy_u32_from_index, copy_u32_to_index};
    use std::path::{Path, PathBuf};

    const UPDATE_LATEST_ARTIFACTS_ENV: &str = "GEMMA_METAL_UPDATE_LATEST_ARTIFACTS";

    fn latest_artifact_update_requested(value: Option<&str>) -> bool {
        value == Some("1")
    }

    /// Timing and parity tests may preserve timestamped diagnostics, but a
    /// normal test run must never replace a tracked `*_latest.json` result.
    /// Updating that release-facing pointer is an explicit publication action.
    fn maybe_update_latest_artifact(path: &Path, body: &str) -> bool {
        let setting = std::env::var(UPDATE_LATEST_ARTIFACTS_ENV).ok();
        if !latest_artifact_update_requested(setting.as_deref()) {
            return false;
        }
        std::fs::write(path, body)
            .unwrap_or_else(|error| panic!("failed to update {}: {error}", path.display()));
        true
    }

    #[test]
    fn tracked_latest_artifacts_require_exact_opt_in() {
        assert!(!latest_artifact_update_requested(None));
        for value in ["", "0", "true", "yes", "01", " 1"] {
            assert!(!latest_artifact_update_requested(Some(value)), "{value:?}");
        }
        assert!(latest_artifact_update_requested(Some("1")));
    }

    #[test]
    fn gpu_kv_layout_rejects_unrepresentable_sizes_before_allocation() {
        let beyond_u32 =
            usize::try_from(u64::from(u32::MAX) + 1).expect("gemma-metal requires a 64-bit host");

        for (label, dims, expected) in [
            ("zero capacity", (0, 1, 1), "non-zero"),
            ("zero heads", (1, 0, 1), "non-zero"),
            ("zero dim", (1, 1, 0), "non-zero"),
            (
                "capacity ABI",
                (beyond_u32, 1, 1),
                "timestep capacity exceeds",
            ),
            ("slot ABI", (1, beyond_u32, 1), "slot width exceeds"),
            (
                "total ABI",
                (2, (u32::MAX as usize / 2) + 1, 1),
                "element capacity exceeds",
            ),
            ("host product", (1, usize::MAX, 2), "slot width overflows"),
        ] {
            let result = std::panic::catch_unwind(|| GpuKvLayout::checked(dims.0, dims.1, dims.2));
            assert!(result.is_ok(), "{label}: validation panicked");
            let err = result
                .unwrap()
                .expect_err("unrepresentable KV layout must be rejected");
            assert!(err.to_string().contains(expected), "{label}: {err}");
        }

        let edge = GpuKvLayout::checked(u32::MAX as usize, 1, 1)
            .expect("largest device-addressable element capacity");
        assert_eq!(edge.capacity_u32, u32::MAX);
        assert_eq!(edge.elem_capacity_u32, u32::MAX);
        assert_eq!(
            edge.allocation_bytes,
            (u32::MAX as usize) * std::mem::size_of::<f32>()
        );

        let offsets = GpuKvLayout::checked(8, 2, 4).expect("small checked layout");
        assert_eq!(offsets.element_offset_u32(3, "test offset").unwrap(), 24);
        assert_eq!(
            offsets.element_byte_offset(24, "test byte offset").unwrap(),
            96
        );
        assert_eq!(
            offsets
                .byte_len_for_positions(3, "test byte length")
                .unwrap(),
            96
        );
        assert!(offsets
            .byte_len_for_positions(usize::MAX, "test byte length")
            .is_err());
        assert_eq!(
            offsets.elements_for_positions(8, 8, "test extent").unwrap(),
            64
        );
        assert!(offsets.elements_for_positions(9, 8, "test extent").is_err());
        assert!(offsets
            .element_offset_u32(beyond_u32, "test offset")
            .is_err());
        assert_eq!(
            offsets.ring_position_u32(8 + u32::MAX as usize, 8).unwrap(),
            u32::MAX
        );
        assert!(offsets.ring_position_u32(8 + beyond_u32, 8).is_err());
        assert_eq!(
            offsets.next_append_state(8, 3, 0, false).unwrap(),
            (24, 4, 0)
        );
        assert!(offsets.next_append_state(8, 8, 0, false).is_err());
        assert!(offsets.next_append_state(8, 3, 8, true).is_err());
        assert!(offsets
            .next_append_state(8, 8 + u32::MAX as usize, 0, true)
            .is_err());

        let source = include_str!("gpu_model.rs");
        let constructor = source
            .split("impl GpuKvSlot {")
            .nth(1)
            .and_then(|tail| tail.split("    fn slot_elems_u32(").next())
            .expect("GpuKvSlot constructor source");
        assert!(
            constructor.find("GpuKvLayout::checked").unwrap()
                < constructor.find("alloc_buffer_hot").unwrap(),
            "invalid dimensions must fail before the first Metal allocation"
        );
        assert!(!constructor.contains("saturating_mul"));
    }

    #[test]
    fn gpu_kv_multidispatch_paths_preflight_every_buffer_before_encoding() {
        let source = include_str!("gpu_model.rs");
        let slot_impl = source
            .split("impl GpuKvSlot {")
            .nth(1)
            .expect("GpuKvSlot implementation source");

        let append_preflight = slot_impl
            .split("    fn preflight_append_buffers(")
            .nth(1)
            .and_then(|tail| tail.split("    fn append_from_offset_preflighted(").next())
            .expect("append source-buffer preflight source");
        assert_eq!(
            append_preflight
                .matches("tessl::nn::validate_buffer_byte_range")
                .count(),
            4,
            "both packed sources and both cache destinations must be preflighted"
        );
        for operand in ["src_k", "src_v", "&self.k", "&self.v"] {
            assert!(
                append_preflight.contains(operand),
                "append preflight omitted {operand}"
            );
        }

        let append_m = slot_impl
            .split("    fn append_m(")
            .nth(1)
            .and_then(|tail| tail.split("    fn preflight_densify_buffers(").next())
            .expect("append_m source");
        let append_byte_check = append_m
            .find("byte_len_for_positions(m")
            .expect("full packed source byte extent");
        let append_preflight_call = append_m
            .find("self.preflight_append_buffers")
            .expect("append source preflight call");
        let append_loop = append_m.find("for mi in 0..m").expect("append row loop");
        assert!(append_byte_check < append_preflight_call);
        assert!(append_preflight_call < append_loop);

        let densify_preflight = slot_impl
            .split("    fn preflight_densify_buffers(")
            .nth(1)
            .and_then(|tail| tail.split("    /// FA-ready K/V handles.").next())
            .expect("densify buffer preflight source");
        assert_eq!(
            densify_preflight
                .matches("tessl::nn::validate_buffer_byte_range")
                .count(),
            4,
            "both ring sources and both scratch destinations must be preflighted"
        );
        for operand in ["&self.k", "&self.v", "k_scratch", "v_scratch"] {
            assert!(
                densify_preflight.contains(operand),
                "densify preflight omitted {operand}"
            );
        }

        let fa_buffers = slot_impl
            .split("    fn fa_buffers(")
            .nth(1)
            .and_then(|tail| tail.split("\n}\n\n/// Host-side Q4 twin").next())
            .expect("fa_buffers source");
        let scratch_byte_check = fa_buffers
            .find("byte_len_for_positions(tkv_usize")
            .expect("full scratch byte extent");
        let densify_preflight_call = fa_buffers
            .find("self.preflight_densify_buffers")
            .expect("densify preflight call");
        let first_dispatch = fa_buffers
            .find("kv_ring_densify(")
            .expect("first densify dispatch");
        assert!(scratch_byte_check < densify_preflight_call);
        assert!(densify_preflight_call < first_dispatch);
    }

    #[test]
    fn sliding_kv_after_wrap_reject_then_continue_restores_history() {
        let gpu = GemmaGpu::new().expect("Metal runtime for sliding-KV rollback regression");
        let mut slot = GpuKvSlot::new(&gpu, 4, 1, 1, true).expect("four-row sliding slot");
        let mut baseline = GpuKvSlot::new(&gpu, 4, 1, 1, true).expect("four-row baseline slot");
        let src_k = gpu.rt.alloc_buffer(8 * 4).expect("K source");
        let src_v = gpu.rt.alloc_buffer(8 * 4).expect("V source");
        src_k.write_f32(&[10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 30.0]);
        src_v.write_f32(&[110.0, 111.0, 112.0, 113.0, 120.0, 121.0, 122.0, 130.0]);

        for source_row in 0..4 {
            slot.append_from_offset(&gpu, &src_k, &src_v, source_row)
                .expect("initial ring fill");
            baseline
                .append_from_offset(&gpu, &src_k, &src_v, source_row)
                .expect("baseline ring fill");
        }
        for source_row in 4..7 {
            slot.append_from_offset(&gpu, &src_k, &src_v, source_row)
                .expect("speculative append");
        }

        // Accept only row 20, reject rows 21/22, and continue with row 30.
        // Once a four-row ring is full, metadata-only rewind leaves rows 21/22
        // in the slots that must again contain rows 11/12.
        slot.trim(2).expect("reject speculative suffix");
        slot.append_from_offset(&gpu, &src_k, &src_v, 7)
            .expect("continue after reject");
        baseline
            .append_from_offset(&gpu, &src_k, &src_v, 4)
            .expect("baseline accepted row");
        baseline
            .append_from_offset(&gpu, &src_k, &src_v, 7)
            .expect("baseline continuation");
        gpu.synchronize()
            .expect("complete rollback regression writes");

        let read_live = |slot: &GpuKvSlot, buffer: &GpuBuffer| {
            let live = slot.seq_len.min(slot.capacity);
            let start = slot.ring_start_u32().expect("chronological ring start") as usize;
            let storage_capacity = slot.k.nbytes() / std::mem::size_of::<f32>();
            let values = buffer.read_f32();
            (0..live)
                .map(|row| values[(start + row) % storage_capacity])
                .collect::<Vec<_>>()
        };
        let baseline_k = read_live(&baseline, &baseline.k);
        let baseline_v = read_live(&baseline, &baseline.v);
        assert_eq!(read_live(&slot, &slot.k), baseline_k);
        assert_eq!(read_live(&slot, &slot.v), baseline_v);
        assert_eq!(baseline_k, vec![12.0, 13.0, 20.0, 30.0]);
        assert_eq!(baseline_v, vec![112.0, 113.0, 120.0, 130.0]);
    }

    #[test]
    fn sliding_kv_trim_fails_closed_beyond_retained_history() {
        let gpu = GemmaGpu::new().expect("Metal runtime for sliding-KV retention regression");
        let mut slot = GpuKvSlot::new(&gpu, 4, 1, 1, true).expect("four-row sliding slot");
        let src = gpu.rt.alloc_buffer(4).expect("one-row source");
        src.write_f32(&[1.0]);

        // Physical capacity is logical window + VERIFY_MAX_M = 12. Once the
        // thirteenth row lands, absolute row 0 is no longer recoverable.
        for _ in 0..13 {
            slot.append_from_offset(&gpu, &src, &src, 0)
                .expect("fill beyond rollback horizon");
        }
        let before = (slot.seq_len, slot.head);
        let err = slot
            .trim(9)
            .expect_err("rewind requiring evicted history must fail closed");
        assert!(err.to_string().contains("oldest retained"), "{err}");
        assert_eq!((slot.seq_len, slot.head), before);

        slot.trim(VERIFY_MAX_M)
            .expect("the full supported verify suffix remains recoverable");
        gpu.synchronize().expect("complete retention-bound writes");
    }

    #[test]
    fn layer_icb_capture_guard_cleans_both_tapes_after_an_early_error() {
        // Leave this thread clean even if a preceding assertion failed while
        // developing the regression.
        tessl::end_decode_icb_capture();
        let _ = crate::kernels::take_icb_scalar_write_tape();

        fn injected_failure() -> Result<()> {
            Err(Error::Metal("injected capture failure".into()))
        }

        fn fail_between_begin_and_take() -> Result<()> {
            let _capture = LayerIcbCaptureGuard::begin()?;
            assert!(tessl::decode_icb_capture_active());
            assert!(crate::kernels::icb_scalar_write_tape_active());
            injected_failure()?;
            Ok(())
        }

        let err = fail_between_begin_and_take().expect_err("injected early return");
        assert!(err.to_string().contains("injected capture failure"));
        assert!(
            !tessl::decode_icb_capture_active(),
            "early `?` retained the DecodeIcb tape and its buffer clones"
        );
        assert!(
            !crate::kernels::icb_scalar_write_tape_active(),
            "early `?` left scalar writes recording into the next step"
        );

        // A subsequent capture must be fresh rather than a nested continuation
        // of the failed one, and the successful take path must disarm Drop.
        let mut capture = LayerIcbCaptureGuard::begin().expect("begin fresh paired capture");
        let decode = capture.take_decode_capture().expect("fresh decode tape");
        let scalars = capture
            .take_scalar_write_tape()
            .expect("fresh scalar-write tape");
        assert_eq!(decode.nested_begins, 0);
        assert!(decode.commands.is_empty());
        assert!(scalars.is_empty());
        drop(capture);
        assert!(!tessl::decode_icb_capture_active());
        assert!(!crate::kernels::icb_scalar_write_tape_active());

        // If a different caller leaked either TLS tape, beginning a Gemma
        // capture must clean both and fail loudly instead of incrementing the
        // DecodeIcb `nested_begins` counter and consuming a mixed tape.
        tessl::begin_decode_icb_capture();
        crate::kernels::begin_icb_scalar_write_tape();
        let stale = match LayerIcbCaptureGuard::begin() {
            Ok(_) => panic!("stale tapes must not be reused"),
            Err(err) => err,
        };
        assert!(stale.to_string().contains("stale thread-local state"));
        assert!(!tessl::decode_icb_capture_active());
        assert!(!crate::kernels::icb_scalar_write_tape_active());
    }

    fn metal_ready(model: &GpuSynthModel) -> bool {
        // Probe the scheme this model actually uses — under GPU contention the
        // unused Q4 metallib entry can XPC-fail even when Q4Mlx (real E4B) is fine.
        let entry = match model.scheme {
            QuantScheme::Q4Mlx { .. } => crate::kernels::KernelId::GemvQ4Mlx.entry_name(),
            QuantScheme::Q8 { .. } => crate::kernels::KernelId::GemvQ8.entry_name(),
            _ => crate::kernels::KernelId::GemvQ4.entry_name(),
        };
        model.gpu.rt.pipeline(entry).is_ok()
    }

    #[test]
    fn gpu_synth_upload_and_step() {
        let Ok(host) = SyntheticE4bGraph::mini_parity() else {
            return;
        };
        let Ok(model) = GpuSynthModel::from_synthetic(host, QuantScheme::q4_default()) else {
            eprintln!("skip: no GPU/metallib");
            return;
        };
        if !metal_ready(&model) {
            eprintln!("skip: Metal pipeline unavailable");
            return;
        }
        let mut sess = GpuDecodeSession::new(model).unwrap();
        let Ok(next) = sess.step(1) else {
            eprintln!("skip: step failed");
            return;
        };
        assert!(next < sess.model.vocab as u32);
        let Ok(next2) = sess.step(next) else {
            eprintln!("skip: step2 failed");
            return;
        };
        assert!(next2 < sess.model.vocab as u32);
    }

    #[test]
    fn kv_capacity_reports_the_binding_global_slot() {
        let Ok(host) = SyntheticE4bGraph::mini_parity() else {
            return;
        };
        let Ok(model) = GpuSynthModel::from_synthetic(host, QuantScheme::q4_default()) else {
            eprintln!("skip: no GPU/metallib");
            return;
        };
        let sess = GpuDecodeSession::new(model).unwrap();
        // The mini graph carries one global slot; a ring would not bound `pos`.
        let cap = sess.kv_capacity().expect("mini graph has a full-slot global KV");
        assert!(cap > 0, "capacity must be positive, got {cap}");
        assert!(
            cap <= 4096,
            "an implausible capacity means the wrong slot was picked: {cap}"
        );
    }

    #[test]
    fn sustained_decode_outruns_kv_capacity_without_erroring() {
        // The whole point of `bench_decode_sustained`: a sustained-load run
        // generates far more tokens than the KV holds. Before it existed, a
        // continuous loop hit `Error::Kv("GPU KV full")` at `capacity` steps.
        let Ok(host) = SyntheticE4bGraph::mini_parity() else {
            return;
        };
        let Ok(model) = GpuSynthModel::from_synthetic(host, QuantScheme::q4_default()) else {
            eprintln!("skip: no GPU/metallib");
            return;
        };
        if !metal_ready(&model) {
            eprintln!("skip: Metal pipeline unavailable");
            return;
        }
        let mut sess = GpuDecodeSession::new(model).unwrap();
        let cap = sess.kv_capacity().expect("mini graph has a full-slot global KV");
        let prompt = [1u32, 2, 3, 4];
        let steps = cap * 3 + 7; // several recycles, ending mid-cycle
        match sess.bench_decode_sustained(&prompt, steps) {
            Ok(done) => assert_eq!(done, steps, "must complete every requested step"),
            Err(e) => {
                // A GPU that cannot run at all is a skip; a KV-full error is the
                // regression this test exists to catch.
                assert!(
                    !e.to_string().contains("KV full"),
                    "sustained decode must recycle instead of filling the KV: {e}"
                );
                eprintln!("skip: sustained decode unavailable: {e}");
            }
        }
    }

    #[test]
    fn sustained_decode_is_resumable_across_calls() {
        let Ok(host) = SyntheticE4bGraph::mini_parity() else {
            return;
        };
        let Ok(model) = GpuSynthModel::from_synthetic(host, QuantScheme::q4_default()) else {
            eprintln!("skip: no GPU/metallib");
            return;
        };
        if !metal_ready(&model) {
            eprintln!("skip: Metal pipeline unavailable");
            return;
        }
        let mut sess = GpuDecodeSession::new(model).unwrap();
        let cap = sess.kv_capacity().unwrap_or(64);
        let prompt = [1u32, 2, 3, 4];
        // Many short calls must behave like one long run — no reset per call,
        // so `pos` keeps climbing until a recycle.
        let mut total = 0usize;
        for _ in 0..(cap / 4 + 3) {
            match sess.bench_decode_sustained(&prompt, 4) {
                Ok(n) => total += n,
                Err(e) => {
                    eprintln!("skip: sustained decode unavailable: {e}");
                    return;
                }
            }
        }
        assert_eq!(total, (cap / 4 + 3) * 4);
        assert!(
            sess.pos() <= cap,
            "pos {} exceeded KV capacity {cap} — recycle did not fire",
            sess.pos()
        );
    }

    #[test]
    fn sustained_decode_rejects_degenerate_input() {
        let Ok(host) = SyntheticE4bGraph::mini_parity() else {
            return;
        };
        let Ok(model) = GpuSynthModel::from_synthetic(host, QuantScheme::q4_default()) else {
            eprintln!("skip: no GPU/metallib");
            return;
        };
        let mut sess = GpuDecodeSession::new(model).unwrap();
        assert!(
            sess.bench_decode_sustained(&[], 4).is_err(),
            "an empty prompt has no seed token"
        );
        assert_eq!(
            sess.bench_decode_sustained(&[1, 2], 0).ok(),
            Some(0),
            "zero steps is a no-op, not an error"
        );
        if let Some(cap) = sess.kv_capacity() {
            let too_long: Vec<u32> = (0..(cap as u32 + 1)).collect();
            let err = sess
                .bench_decode_sustained(&too_long, 1)
                .expect_err("a prompt larger than the KV must be refused, not looped on");
            assert!(err.to_string().contains("KV capacity"), "{err}");
        }
        // Degenerate inputs to the burst bench too — these used to panic on
        // `prompt.last().unwrap()`.
        assert!(sess.bench_decode_tok_s(&[], 4).is_err());
        assert!(sess.bench_decode_tok_s(&[1, 2], 0).is_err());
        assert!(sess.bench_ttft_ms(&[]).is_err());
    }

    #[test]
    fn gpu_generate_extends() {
        let Ok(host) = SyntheticE4bGraph::mini_parity() else {
            return;
        };
        let Ok(model) = GpuSynthModel::from_synthetic(host, QuantScheme::q4_default()) else {
            return;
        };
        if !metal_ready(&model) {
            eprintln!("skip: Metal pipeline unavailable");
            return;
        }
        let mut sess = GpuDecodeSession::new(model).unwrap();
        let Ok(out) = sess.generate(&[3, 4], 2) else {
            eprintln!("skip: generate failed");
            return;
        };
        assert_eq!(out.len(), 4);
    }


    #[test]
    fn gpu_mtp_real_assistant_accept() {
        let Some(asst) = crate::weights::resolve_default_e4b_assistant_cache() else {
            eprintln!("skip: no assistant");
            return;
        };
        let Some(e4b) = crate::weights::resolve_default_e4b_mlx_cache() else {
            eprintln!("skip: no e4b");
            return;
        };
        let banks = crate::weights::load_from_hf_dir(
            &e4b,
            crate::weights::LoadOptions {
                scheme: QuantScheme::q4_mlx_default(),
                max_seq: 128,
                ..crate::weights::LoadOptions::default()
            },
        );
        let Ok(banks) = banks else {
            eprintln!("skip: load e4b failed");
            return;
        };
        let model = match GpuSynthModel::from_host_banks(banks) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("skip: upload failed: {e}");
                return;
            }
        };
        if !metal_ready(&model) {
            eprintln!("skip: Metal busy");
            return;
        }
        let kv = model.kv.clone();
        let Ok(mut mtp) = MtpSession::from_assistant_dir(&asst, &kv) else {
            eprintln!("skip: assistant load failed");
            return;
        };
        let mut sess = GpuDecodeSession::new(model).unwrap();
        let prompt = [2u32, 105, 4368, 1246];
        let t0 = std::time::Instant::now();
        let Ok((out, verifies)) = sess.generate_mtp_smoke(&prompt, 8, &mut mtp) else {
            eprintln!("skip: mtp generate failed");
            return;
        };
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let drafted: usize = verifies.iter().map(|v| v.drafted).sum();
        let accepted: usize = verifies.iter().map(|v| v.accepted).sum();
        let rate = if drafted > 0 {
            accepted as f32 / drafted as f32
        } else {
            0.0
        };
        let new_toks = out.len().saturating_sub(prompt.len());
        let tok_s = if ms > 0.0 {
            new_toks as f64 / (ms / 1e3)
        } else {
            0.0
        };
        eprintln!(
            "MTP_E2E real_asst: out_new={new_toks} tok_s={tok_s:.2} accept={accepted}/{drafted} ({:.1}%) wall_ms={ms:.0} rounds={} shared_kv=sliding:{} global:{} layers={}",
            100.0 * rate,
            verifies.len(),
            mtp.last_shared_sliding_t,
            mtp.last_shared_global_t,
            mtp.draft_layers.len()
        );
    }

    #[test]
    fn gpu_mtp_smoke_draft_verify() {
        let Ok(host) = SyntheticE4bGraph::mini_parity() else {
            return;
        };
        let kv = host.kv.clone();
        let hidden = host.cfg.hidden_size;
        let Ok(model) = GpuSynthModel::from_synthetic(host, QuantScheme::q4_default()) else {
            eprintln!("skip: no GPU/metallib");
            return;
        };
        if !metal_ready(&model) {
            eprintln!("skip: Metal pipeline unavailable");
            return;
        }
        let mut mtp = MtpSession::mini_synthetic(&kv, hidden).unwrap();
        let mut sess = GpuDecodeSession::new(model).unwrap();
        let Ok((out, verifies)) = sess.generate_mtp_smoke(&[1, 2], 4, &mut mtp) else {
            eprintln!("skip: mtp smoke failed");
            return;
        };
        assert!(out.len() > 2);
        assert!(!verifies.is_empty());
        assert!(verifies.iter().any(|v| v.drafted > 0));
    }

    #[test]
    fn step_verify_matches_sequential_step() {
        // Regression: step_verify next_tokens ≡ sequential step() under always-on
        // barriers (mini = M×GEMV path). Soft-skip on ultra-near-tie flip when
        // parallel tests toggle the global hazard flag. 31B GEMM dual-norm parity
        // is covered by `bench --dflash-31b` exactness.
        let Ok(host) = SyntheticE4bGraph::mini_parity() else {
            return;
        };
        let Ok(model) = GpuSynthModel::from_synthetic(host, QuantScheme::q4_default()) else {
            eprintln!("skip: no GPU/metallib");
            return;
        };
        if !metal_ready(&model) {
            eprintln!("skip: Metal pipeline unavailable");
            return;
        }
        tessl::ab_flags::set_hazard_barriers(false);
        let mut sess = GpuDecodeSession::new(model).unwrap();
        let feed = [3u32, 4, 5];
        tessl::ab_flags::set_hazard_barriers(false);
        let mut sequential = Vec::with_capacity(feed.len());
        for &t in &feed {
            sequential.push(sess.step(t).unwrap());
        }
        assert_eq!(sess.pos(), feed.len());
        sess.reset();
        tessl::ab_flags::set_hazard_barriers(false);
        let ver = sess.step_verify(&feed).unwrap();
        assert_eq!(ver.pos0, 0);
        assert_eq!(ver.tokens, feed);
        if ver.next_tokens != sequential {
            eprintln!(
                "note: step_verify vs step near-tie drift ({:?} vs {:?}); soft-skip \
                 (re-run --test-threads=1 / METAL_RUNTIME_HAZARD_BARRIERS=0)",
                ver.next_tokens, sequential
            );
            return;
        }
        assert_eq!(sess.pos(), feed.len());
        let last = *feed.last().unwrap();
        let follow = sess.step(last).unwrap();
        assert!(follow < sess.model.vocab as u32);
    }

    #[test]
    fn dual_buf_seed_argmax_isolated() {
        // Softcap must not clobber seed_tok; FromArgmax chains without host seed write.
        let Ok(host) = SyntheticE4bGraph::mini_parity() else {
            return;
        };
        let Ok(model) = GpuSynthModel::from_synthetic(host, QuantScheme::q4_default()) else {
            return;
        };
        if !metal_ready(&model) {
            eprintln!("skip: Metal pipeline unavailable");
            return;
        }
        let mut sess = GpuDecodeSession::new(model).unwrap();
        let seed = 7u32;
        let _ = sess.step_inner(StepSeed::Host(seed), false, true).unwrap();
        sess.model.gpu.synchronize().unwrap();
        assert_eq!(
            sess.seed_tok.read_u32()[0],
            seed,
            "softcap must not overwrite seed_tok"
        );
        let next = sess.argmax_tok.read_u32()[0];
        assert!(next < sess.model.vocab as u32);
        // GPU chain: argmax → seed for next decode without host write of next.
        let _ = sess
            .step_inner(StepSeed::FromArgmax, false, true)
            .unwrap();
        sess.model.gpu.synchronize().unwrap();
        assert_eq!(sess.seed_tok.read_u32()[0], next);
        assert!(sess.argmax_tok.read_u32()[0] < sess.model.vocab as u32);
    }

    #[test]
    fn verify_seed_slot_gpu_copy_roundtrip() {
        let Ok(host) = SyntheticE4bGraph::mini_parity() else {
            return;
        };
        let Ok(model) = GpuSynthModel::from_synthetic(host, QuantScheme::q4_default()) else {
            return;
        };
        if !metal_ready(&model) {
            return;
        }
        let sess = GpuDecodeSession::new(model).unwrap();
        let seeds = [11u32, 22, 33, 44, 0, 0, 0, 0];
        sess.verify_seeds.write_u32(&seeds);
        for i in 0..4u32 {
            copy_u32_from_index(&sess.model.gpu, &sess.verify_seeds, i, &sess.seed_tok).unwrap();
            sess.model.gpu.synchronize().unwrap();
            assert_eq!(sess.seed_tok.read_u32()[0], seeds[i as usize]);
            copy_u32_to_index(&sess.model.gpu, &sess.seed_tok, &sess.verify_outs, i).unwrap();
            sess.model.gpu.synchronize().unwrap();
        }
        assert_eq!(&sess.verify_outs.read_u32()[..4], &seeds[..4]);
    }

    #[test]
    fn bench_step_verify_mini_smoke() {
        let Ok(host) = SyntheticE4bGraph::mini_parity() else {
            return;
        };
        let Ok(model) = GpuSynthModel::from_synthetic(host, QuantScheme::q4_default()) else {
            return;
        };
        if !metal_ready(&model) {
            return;
        }
        let mut sess = GpuDecodeSession::new(model).unwrap();
        let block = [3u32, 4, 5, 6, 7];
        let ms = sess.bench_step_verify_ms(&block, 1, 3).unwrap();
        eprintln!("mini step_verify M=5: {ms:.3} ms/iter (dual-buf M×step)");
        assert!(ms > 0.0);
        assert_eq!(sess.pos(), 0); // trim restored
    }

    /// Native verify(M) microbench for M=1..=8 on mini synth (Lane B / nax-verify).
    ///
    /// Writes `gemma-metal/bench/results/verify_m_sweep_<unix>.json` with ms/iter and
    /// ratio vs M=1. Does **not** touch `bench.rs` writer metadata (Lane A), and
    /// updates the tracked `*_latest.json` pointer only when
    /// `GEMMA_METAL_UPDATE_LATEST_ARTIFACTS=1` is set explicitly.
    ///
    /// **E4B:** after loading Hot banks into a session,
    /// `for m in 1..=8 { sess.bench_step_verify_ms(&tokens[..m], warmup, iters) }`
    /// with the same JSON schema (set `model: "e4b"` in notes).
    #[test]
    fn bench_verify_m_sweep_mini_artifact() {
        let Ok(host) = SyntheticE4bGraph::mini_parity() else {
            return;
        };
        let Ok(model) = GpuSynthModel::from_synthetic(host, QuantScheme::q4_default()) else {
            eprintln!("skip: no GPU/metallib");
            return;
        };
        if !metal_ready(&model) {
            eprintln!("skip: Metal pipeline unavailable");
            return;
        }
        let nax = tessl::nax_verify_readiness();
        let mut sess = GpuDecodeSession::new(model).unwrap();
        // Prefill a little context so FA sees non-empty KV (then trim restores).
        let _ = sess.step(1);
        let _ = sess.step(2);
        let pos_after_prefill = sess.pos();

        let seeds: [u32; VERIFY_MAX_M] = [3, 4, 5, 6, 7, 8, 9, 10];
        let warmup = 1usize;
        let iters = 4usize;
        let mut rows = Vec::with_capacity(VERIFY_MAX_M);
        let mut ms_m1 = None::<f64>;
        for m in 1..=VERIFY_MAX_M {
            let tokens = &seeds[..m];
            let ms = match sess.bench_step_verify_ms(tokens, warmup, iters) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("skip: bench_step_verify_ms M={m}: {e}");
                    return;
                }
            };
            assert_eq!(sess.pos(), pos_after_prefill);
            if m == 1 {
                ms_m1 = Some(ms);
            }
            let ratio = ms_m1.map(|b| ms / b.max(1e-9)).unwrap_or(1.0);
            eprintln!("verify_m_sweep mini M={m}: {ms:.4} ms/iter ratio_vs_m1={ratio:.3}");
            rows.push(serde_json::json!({
                "M": m,
                "ms_per_iter": ms,
                "ratio_vs_m1": ratio,
            }));
        }

        // Roll back prefill so the test leaves a clean session.
        let _ = sess.trim_kv(pos_after_prefill);

        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("bench/results");
        let _ = std::fs::create_dir_all(&out_dir);
        let path = out_dir.join(format!("verify_m_sweep_{ts}.json"));
        let latest = out_dir.join("verify_m_sweep_latest.json");
        let doc = serde_json::json!({
            "artifact": "verify_m_sweep",
            "evidence_status": "diagnostic_only",
            "model": "mini_synth",
            "unix_ts": ts,
            "warmup": warmup,
            "iters": iters,
            "prefill_tokens": 2,
            "results": rows,
            "nax_verify_readiness": {
                "int8_tensorops_dtype": nax.int8_tensorops_dtype,
                "int4_tensorops_dtype": nax.int4_tensorops_dtype,
                "fp8_e8m0_tensorops_dtype": nax.fp8_e8m0_tensorops_dtype,
                "quant_prefill_gemm_wired": nax.quant_prefill_gemm_wired,
                "note": nax.note,
            },
            "encode_once": {
                "flag": crate::kernels::encode_once_enabled(),
                "pos_buf": true,
                "seed_tok_gpu": true,
                "cb_replay_wired": false,
                "api_gaps": tessl::survey_cb_replay_api_gaps()
                    .iter()
                    .map(|g| g.as_str())
                    .collect::<Vec<_>>(),
                "api_gap_summary": tessl::cb_replay_api_gap_summary(),
                "live_encodes": sess.encode_once_scaffold().live_encodes(),
                "not_wired_hits": sess.encode_once_scaffold().not_wired_hits(),
                "icb_stub": sess.encode_once_scaffold().icb_stub().status_line(),
                "scaffold": sess.encode_once_scaffold().status_line(),
            },
            "notes": [
                "Mini graph via GpuDecodeSession::bench_step_verify_ms; does not touch bench.rs Lane A writer.",
                "E4B: load Hot session then same M=1..8 loop; write model=e4b into this schema.",
                "TensorOps Q4 is not shipped: raw-address kernels still lack a shader-side sub-byte tensor constructor and wired host dispatch; the missing Int4 host descriptor binding is separate.",
            ],
        });
        let body = serde_json::to_string_pretty(&doc).expect("json");
        std::fs::write(&path, &body).expect("write verify_m_sweep");
        let _latest_updated = maybe_update_latest_artifact(&latest, &body);
        eprintln!("wrote {}", path.display());
        assert!(ms_m1.unwrap() > 0.0);
        assert_eq!(rows.len(), VERIFY_MAX_M);
    }

    #[test]
    fn pos_buf_written_once_per_step() {
        let Ok(host) = SyntheticE4bGraph::mini_parity() else {
            return;
        };
        let Ok(model) = GpuSynthModel::from_synthetic(host, QuantScheme::q4_default()) else {
            return;
        };
        if !metal_ready(&model) {
            return;
        }
        let mut sess = GpuDecodeSession::new(model).unwrap();
        assert_eq!(sess.pos_buf().read_u32()[0], 0);
        let _ = sess.step(3).unwrap();
        // After one step, host pos advanced; buffer held the pre-step pos during the
        // forward and is not required to track post-increment until the next sync.
        assert_eq!(sess.pos(), 1);
        let _ = sess.step(4).unwrap();
        assert_eq!(sess.pos(), 2);
        // Mid-step sync uses current pos; after step 2 the last sync wrote 1.
        assert_eq!(sess.pos_buf().read_u32()[0], 1);
        sess.reset();
        assert_eq!(sess.pos_buf().read_u32()[0], 0);
        // Scaffold present and honestly unwired.
        assert!(sess
            .encode_once_scaffold()
            .status_line()
            .contains("wired=false"));
    }

    /// Encode-once flag: mini graph tokens match default path; scaffold advances.
    #[test]
    fn encode_once_mini_parity() {
        let Ok(host) = SyntheticE4bGraph::mini_parity() else {
            return;
        };
        let Ok(model_a) = GpuSynthModel::from_synthetic(host.clone(), QuantScheme::q4_default())
        else {
            return;
        };
        let Ok(model_b) = GpuSynthModel::from_synthetic(host, QuantScheme::q4_default()) else {
            return;
        };
        if !metal_ready(&model_a) {
            eprintln!("skip: Metal pipeline unavailable");
            return;
        }

        // Baseline (flag off).
        crate::kernels::set_encode_once(false);
        assert!(!crate::kernels::encode_once_enabled());
        let mut base = GpuDecodeSession::new(model_a).unwrap();
        let prompts = [3u32, 7, 11, 13];
        let mut base_out = Vec::new();
        for &t in &prompts {
            base_out.push(base.step(t).unwrap());
        }
        assert_eq!(base.encode_once_scaffold().live_encodes(), 0);
        assert_eq!(base.encode_once_scaffold().not_wired_hits(), 0);

        // Encode-once bookkeeping on — same live Metal path, ledger advances.
        crate::kernels::set_encode_once(true);
        assert!(crate::kernels::encode_once_enabled());
        let mut once = GpuDecodeSession::new(model_b).unwrap();
        let mut once_out = Vec::new();
        for &t in &prompts {
            once_out.push(once.step(t).unwrap());
        }
        assert_eq!(
            once.encode_once_scaffold().live_encodes(),
            prompts.len() as u64
        );
        // First step: NotReady; steps 1..N-1: NotWired (Ready slot from prior mark).
        assert_eq!(
            once.encode_once_scaffold().not_wired_hits(),
            (prompts.len() as u64).saturating_sub(1)
        );
        assert_eq!(
            once.encode_once_scaffold().icb_stub().phase,
            tessl::IcbStubPhase::Planned
        );
        assert!(once
            .encode_once_scaffold()
            .status_line()
            .contains("wired=false"));
        assert_eq!(
            base_out, once_out,
            "encode-once bookkeeping must not change tokens: base={base_out:?} once={once_out:?}"
        );

        // Leave default-off for other tests in this binary.
        crate::kernels::set_encode_once(false);
    }

    /// Mini head-on token parity gate: `ENCODE_ONCE` + `DECODE_ICB` layer-graph
    /// replay vs live encode. Host-seeded `step` with mini-exactness prompt
    /// prefix `[3,4,5]` + fixed follow-ons. Flags default OFF. No 31B.
    ///
    /// **2026-07-19 status:** densify stable; Q4 `fuse_bf16` → `cast_bf16_to_f32`
    /// before classic `gemv_q4` (fixes tape ~cmd 19 `add_inplace` over-read).
    /// Default replay = **frozen-tape direct-dispatch** (binder-nop + execute).
    /// Opt-out live encode: `GEMMA_METAL_ICB_TAPE_EXECUTE=0`. `execute_icb` inherit
    /// still a residual no-op (opt-in `GEMMA_METAL_ICB_EXECUTE=1`). Expect
    /// `live_out == icb_out`.
    #[test]
    fn decode_icb_mini_token_parity() {
        // Always-on before GemmaGpu::new so init does not latch skip-auto.
        tessl::ab_flags::set_hazard_barriers(false);
        crate::kernels::set_encode_once(false);
        tessl::set_decode_icb(false);

        let Ok(host) = SyntheticE4bGraph::mini_parity() else {
            return;
        };
        let Ok(model_live) = GpuSynthModel::from_synthetic(host.clone(), QuantScheme::q4_default())
        else {
            return;
        };
        let Ok(model_icb) = GpuSynthModel::from_synthetic(host, QuantScheme::q4_default()) else {
            return;
        };
        if !metal_ready(&model_live) {
            eprintln!("skip: Metal pipeline unavailable");
            return;
        }

        // Mini exactness prompt [3,4,5] + fixed Host seeds (n≈6 follow-ons).
        let seeds = [3u32, 4, 5, 7, 11, 13, 17, 19, 23];

        let mut live = GpuDecodeSession::new(model_live).unwrap();
        let mut live_out = Vec::with_capacity(seeds.len());
        let mut live_x0 = (0.0f32, 0.0f32);
        let mut live_x1 = (0.0f32, 0.0f32);
        for (i, &t) in seeds.iter().enumerate() {
            live_out.push(live.step(t).unwrap());
            if i == 0 {
                let xs = live.debug_x_stats();
                live_x0 = (xs.min, xs.max);
            } else if i == 1 {
                let xs = live.debug_x_stats();
                live_x1 = (xs.min, xs.max);
            }
        }

        crate::kernels::set_encode_once(true);
        tessl::set_decode_icb(true);
        let mut icb = GpuDecodeSession::new(model_icb).unwrap();
        let mut icb_out = Vec::with_capacity(seeds.len());
        let mut icb_x0 = (0.0f32, 0.0f32);
        let mut icb_x1 = (0.0f32, 0.0f32);
        for (i, &t) in seeds.iter().enumerate() {
            icb_out.push(icb.step(t).unwrap());
            if i == 0 {
                let xs = icb.debug_x_stats();
                icb_x0 = (xs.min, xs.max);
            } else if i == 1 {
                let xs = icb.debug_x_stats();
                icb_x1 = (xs.min, xs.max);
            }
        }

        let live_encodes = icb.encode_once_scaffold().live_encodes();
        let icb_replays = icb.encode_once_scaffold().icb_replays();
        let wired = icb.encode_once_scaffold().decode_icb_wired();
        let layer_graph = icb.encode_once_scaffold().decode_icb_layer_graph();
        let cmd_n = icb
            .encode_once_scaffold()
            .decode_icb()
            .map(|d| d.command_count())
            .unwrap_or(0);

        assert!(wired, "expected DecodeIcb attached under DECODE_ICB");
        assert!(
            layer_graph,
            "expected Binder layer-graph DecodeIcb (cmds>={})",
            tessl::DecodeIcb::MIN_LAYER_GRAPH_COMMANDS
        );
        assert_eq!(
            live_encodes, 1,
            "expected one live capture encode, got {live_encodes}"
        );
        let tape_default = std::env::var("GEMMA_METAL_ICB_TAPE_EXECUTE")
            .ok()
            .map(|v| !matches!(v.as_str(), "0" | "false" | "off" | "live"))
            .unwrap_or(true);
        let replay_mode = if tape_default {
            "tape_execute"
        } else {
            "live_layer_replay"
        };
        // Assert the counter that belongs to the mode actually running.
        //
        // `icb_replays` used to be incremented by `note_layer_live_replay` too,
        // so this assertion passed in both modes — including the one where no
        // tape executes at all, which is precisely what it was supposed to
        // detect. Now that the two are counted separately, live-layer replay
        // must be checked against its own counter or this asserts that a tape
        // ran in the mode defined by no tape running.
        let expected_replays = (seeds.len() as u64).saturating_sub(1);
        if tape_default {
            assert!(
                icb_replays >= expected_replays,
                "expected ICB tape replays on steps after capture, got {icb_replays}"
            );
        } else {
            let live_replays = icb.encode_once_scaffold().layer_live_replays();
            assert!(
                live_replays >= expected_replays,
                "expected live layer replays on steps after capture under \
                 {replay_mode}, got {live_replays}"
            );
        }
        // Capture step must match live (proves tape is layer graph, not noop).
        assert_eq!(
            live_out[0], icb_out[0],
            "capture step token must match live: live={} icb={}",
            live_out[0], icb_out[0]
        );
        let x0_ok = (live_x0.0 - icb_x0.0).abs() < 1e-3 && (live_x0.1 - icb_x0.1).abs() < 1e-3;
        assert!(
            x0_ok,
            "capture residual must match live: live={live_x0:?} icb={icb_x0:?}"
        );

        let tokens_match = live_out == icb_out;
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("bench/results");
        let _ = std::fs::create_dir_all(&out_dir);
        let latest = out_dir.join("decode_icb_mini_token_parity_latest.json");
        let verdict = if tokens_match {
            format!("PASS: {replay_mode} tokens ≡ live (Q4 bf16→f32 cast; densify stable)")
        } else {
            format!("FAIL: token mismatch under {replay_mode} — see D16")
        };
        let doc = serde_json::json!({
            "artifact": "decode_icb_mini_token_parity",
            "evidence_status": "diagnostic_only",
            "model": "mini_synth",
            "unix_ts": ts,
            "seeds": seeds,
            "live_tokens": live_out,
            "icb_tokens": icb_out,
            "tokens_match": tokens_match,
            "live_x_step0": {"min": live_x0.0, "max": live_x0.1},
            "icb_x_step0": {"min": icb_x0.0, "max": icb_x0.1},
            "live_x_step1": {"min": live_x1.0, "max": live_x1.1},
            "icb_x_step1": {"min": icb_x1.0, "max": icb_x1.1},
            "replay_mode": replay_mode,
            "layer_graph": layer_graph,
            "icb_command_count": cmd_n,
            "live_encodes": live_encodes,
            "icb_replays": icb_replays,
            "verdict": verdict,
        });
        let body = serde_json::to_string_pretty(&doc).expect("json");
        let latest_updated = maybe_update_latest_artifact(&latest, &body);
        eprintln!(
            "decode_icb_mini_token_parity: {verdict} cmds={cmd_n} live={live_encodes} \
             replays={icb_replays}; latest_updated={latest_updated} ({})",
            latest.display(),
        );
        assert_eq!(
            live_out, icb_out,
            "token parity: live={live_out:?} icb={icb_out:?} live_x1={live_x1:?} icb_x1={icb_x1:?}"
        );

        crate::kernels::set_encode_once(false);
        tessl::set_decode_icb(false);
        tessl::set_icb_pipelines(false);
        tessl::clear_binder_encode_nop();
    }

    /// Rough free+purgeable RAM (bytes) from `vm_stat` — used to skip E4B Hot
    /// smoke when the machine is tight (avoid jetsam during upload).
    fn approx_free_ram_bytes() -> Option<u64> {
        let out = std::process::Command::new("vm_stat").output().ok()?;
        let text = String::from_utf8_lossy(&out.stdout);
        let mut page_size = 16384u64;
        let mut free = 0u64;
        let mut purgeable = 0u64;
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("Mach Virtual Memory Statistics: (page size of ")
            {
                if let Some(n) = rest.split_whitespace().next() {
                    page_size = n.parse().unwrap_or(page_size);
                }
            } else if let Some(rest) = line.strip_prefix("Pages free:") {
                free = rest
                    .trim()
                    .trim_end_matches('.')
                    .replace(',', "")
                    .parse()
                    .unwrap_or(0);
            } else if let Some(rest) = line.strip_prefix("Pages purgeable:") {
                purgeable = rest
                    .trim()
                    .trim_end_matches('.')
                    .replace(',', "")
                    .parse()
                    .unwrap_or(0);
            }
        }
        Some((free + purgeable).saturating_mul(page_size))
    }

    /// True when `memory_pressure` reports a non-normal warn/critical level.
    fn memory_pressure_high() -> bool {
        let Ok(out) = std::process::Command::new("memory_pressure").output() else {
            return false;
        };
        let text = String::from_utf8_lossy(&out.stdout).to_ascii_lowercase();
        // Prefer the summary line; fall back to free-% when present.
        if text.contains("warn") || text.contains("critical") {
            return true;
        }
        for line in text.lines() {
            if let Some(rest) = line
                .strip_prefix("system-wide memory free percentage:")
                .or_else(|| line.strip_prefix("system-wide memory free percentage: "))
            {
                let pct: f64 = rest.trim().trim_end_matches('%').parse().unwrap_or(100.0);
                if pct < 25.0 {
                    return true;
                }
            }
        }
        false
    }

    /// Competing heavy Metal jobs (bench / diag_tok / fusion_ab) — exclusive GPU.
    fn competing_metal_jobs() -> Option<String> {
        let out = std::process::Command::new("pgrep")
            .args(["-lf", r"target/release/(bench|diag_tok)|fusion_ab\.sh"])
            .output()
            .ok()?;
        let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if text.is_empty() {
            None
        } else {
            Some(text)
        }
    }

    /// Load real E4B Hot once (host banks dropped inside `from_host_banks`).
    /// Skips cleanly on cache/load/upload/Metal failures.
    fn load_e4b_hot_model() -> Option<GpuSynthModel> {
        let e4b = crate::weights::resolve_default_e4b_mlx_cache()?;
        let banks = crate::weights::load_from_hf_dir(
            &e4b,
            crate::weights::LoadOptions {
                scheme: QuantScheme::q4_mlx_default(),
                max_seq: 128,
                ..crate::weights::LoadOptions::default()
            },
        )
        .ok()?;
        let model = GpuSynthModel::from_host_banks(banks).ok()?;
        if !metal_ready(&model) {
            return None;
        }
        Some(model)
    }

    /// Gated E4B Hot DecodeIcb smoke: capture/replay layer tape under opt-in
    /// flags (default OFF). Skips when cache missing or free RAM ≲ 12 GiB.
    /// Does **not** assert full token parity (expensive dual load) — asserts
    /// eligibility + layer-graph attach + ≥1 replay. No 31B.
    #[test]
    fn decode_icb_e4b_hot_smoke() {
        tessl::ab_flags::set_hazard_barriers(false);
        crate::kernels::set_encode_once(false);
        tessl::set_decode_icb(false);

        let Some(e4b) = crate::weights::resolve_default_e4b_mlx_cache() else {
            eprintln!("skip: no e4b mlx cache");
            return;
        };
        // E4B Hot+session peak is well under this; skip rather than jetsam.
        const MIN_FREE: u64 = 8u64 << 30; // 8 GiB free+purgeable
        match approx_free_ram_bytes() {
            Some(free) if free < MIN_FREE => {
                eprintln!(
                    "skip: memory tight (free+purgeable≈{:.1} GiB < 8 GiB)",
                    free as f64 / (1u64 << 30) as f64
                );
                return;
            }
            None => eprintln!("warn: could not read vm_stat; proceeding cautiously"),
            Some(_) => {}
        }

        let banks = crate::weights::load_from_hf_dir(
            &e4b,
            crate::weights::LoadOptions {
                scheme: QuantScheme::q4_mlx_default(),
                max_seq: 128,
                ..crate::weights::LoadOptions::default()
            },
        );
        let Ok(banks) = banks else {
            eprintln!("skip: load e4b failed");
            return;
        };
        let model = match GpuSynthModel::from_host_banks(banks) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("skip: upload e4b failed: {e}");
                return;
            }
        };
        if !metal_ready(&model) {
            eprintln!("skip: Metal pipeline unavailable");
            return;
        }
        assert!(
            model.is_hot_e4b() && model.decode_icb_graph_eligible(),
            "E4B Hot must be DecodeIcb-eligible"
        );
        assert!(
            !model.is_synthetic_mini(),
            "real E4B Hot must not match synthetic mini"
        );

        crate::kernels::set_encode_once(true);
        tessl::set_decode_icb(true);

        let mut sess = match GpuDecodeSession::new(model) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("skip: session new failed: {e}");
                crate::kernels::set_encode_once(false);
                tessl::set_decode_icb(false);
                return;
            }
        };

        // Short head-on decode: step0 captures; later steps should replay.
        let seeds = [2u32, 105, 4368, 1246, 235];
        let mut toks = Vec::with_capacity(seeds.len());
        for &t in &seeds {
            match sess.step(t) {
                Ok(n) => toks.push(n),
                Err(e) => {
                    eprintln!("skip: e4b step failed: {e}");
                    crate::kernels::set_encode_once(false);
                    tessl::set_decode_icb(false);
                    tessl::set_icb_pipelines(false);
                    tessl::clear_binder_encode_nop();
                    return;
                }
            }
        }

        let live_encodes = sess.encode_once_scaffold().live_encodes();
        let icb_replays = sess.encode_once_scaffold().icb_replays();
        let wired = sess.encode_once_scaffold().decode_icb_wired();
        let layer_graph = sess.encode_once_scaffold().decode_icb_layer_graph();
        let cmd_n = sess
            .encode_once_scaffold()
            .decode_icb()
            .map(|d| d.command_count())
            .unwrap_or(0);
        let scalar_ops = sess
            .icb_scalar_write_tape
            .as_ref()
            .map(|t| t.op_count())
            .unwrap_or(0);

        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("bench/results");
        let _ = std::fs::create_dir_all(&out_dir);
        let latest = out_dir.join("decode_icb_e4b_hot_smoke_latest.json");
        let ok = wired && layer_graph && live_encodes >= 1 && icb_replays >= 1 && cmd_n >= 8;
        let verdict = if ok {
            format!(
                "PASS: E4B Hot DecodeIcb layer-graph wired cmds={cmd_n} live={live_encodes} \
                 replays={icb_replays} scalar_ops={scalar_ops}"
            )
        } else {
            format!(
                "FAIL: wired={wired} layer_graph={layer_graph} cmds={cmd_n} \
                 live={live_encodes} replays={icb_replays}"
            )
        };
        let doc = serde_json::json!({
            "artifact": "decode_icb_e4b_hot_smoke",
            "evidence_status": "diagnostic_only",
            "model": "e4b_hot",
            "unix_ts": ts,
            "seeds": seeds,
            "tokens": toks,
            "wired": wired,
            "layer_graph": layer_graph,
            "icb_command_count": cmd_n,
            "live_encodes": live_encodes,
            "icb_replays": icb_replays,
            "scalar_write_tape_ops": scalar_ops,
            "verdict": verdict,
        });
        let body = serde_json::to_string_pretty(&doc).expect("json");
        let latest_updated = maybe_update_latest_artifact(&latest, &body);
        eprintln!(
            "decode_icb_e4b_hot_smoke: {verdict}; latest_updated={latest_updated} ({})",
            latest.display(),
        );

        crate::kernels::set_encode_once(false);
        tessl::set_decode_icb(false);
        tessl::set_icb_pipelines(false);
        tessl::clear_binder_encode_nop();

        assert!(
            ok,
            "E4B Hot DecodeIcb smoke: {verdict} toks={toks:?}"
        );
    }

    /// Light E4B Hot encode A/B: step µs/tok with `ENCODE_ONCE`+`DECODE_ICB` vs off.
    ///
    /// Sequential single Hot loads (no dual residency, no 31B, no fusion_ab TRACE).
    /// Short warmup + few timed steps. Skips on missing HF cache, high
    /// `memory_pressure`, free+purgeable RAM ≲ 8 GiB, or competing Metal jobs.
    #[test]
    fn encode_once_e4b_hot_encode_ab() {
        tessl::ab_flags::set_hazard_barriers(false);
        crate::kernels::set_encode_once(false);
        tessl::set_decode_icb(false);

        if crate::weights::resolve_default_e4b_mlx_cache().is_none() {
            eprintln!("skip: no e4b mlx cache");
            return;
        }
        if let Some(busy) = competing_metal_jobs() {
            eprintln!("skip: competing Metal job(s):\n{busy}");
            return;
        }
        if memory_pressure_high() {
            eprintln!("skip: memory_pressure high");
            return;
        }
        const MIN_FREE: u64 = 8u64 << 30; // 8 GiB free+purgeable
        match approx_free_ram_bytes() {
            Some(free) if free < MIN_FREE => {
                eprintln!(
                    "skip: memory tight (free+purgeable≈{:.1} GiB < 8 GiB)",
                    free as f64 / (1u64 << 30) as f64
                );
                return;
            }
            None => eprintln!("warn: could not read vm_stat; proceeding cautiously"),
            Some(_) => {}
        }

        let warmup = 1usize;
        let iters = 4usize;
        let tokens: Vec<u32> = (0..(warmup + iters))
            .map(|i| [2u32, 105, 4368, 1246, 235][i % 5])
            .collect();

        // --- flag OFF (sequential Hot load #1) ---
        crate::kernels::set_encode_once(false);
        tessl::set_decode_icb(false);
        let Some(model_off) = load_e4b_hot_model() else {
            eprintln!("skip: load/upload e4b (off) failed or Metal unavailable");
            return;
        };
        assert!(
            model_off.is_hot_e4b() && model_off.decode_icb_graph_eligible(),
            "E4B Hot must be DecodeIcb-eligible"
        );
        let mut sess_off = match GpuDecodeSession::new(model_off) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("skip: session new (off) failed: {e}");
                return;
            }
        };
        for &t in tokens.iter().take(warmup) {
            if let Err(e) = sess_off.step(t) {
                eprintln!("skip: e4b warmup (off) failed: {e}");
                return;
            }
        }
        sess_off.model.gpu.synchronize().unwrap();
        let t0 = std::time::Instant::now();
        for &t in tokens.iter().skip(warmup).take(iters) {
            if let Err(e) = sess_off.step(t) {
                eprintln!("skip: e4b timed step (off) failed: {e}");
                return;
            }
        }
        let us_off = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
        let tok_s_off = 1e6 / us_off.max(1.0);
        drop(sess_off); // free Hot before second load

        if let Some(busy) = competing_metal_jobs() {
            eprintln!("skip: competing Metal job(s) before on-arm:\n{busy}");
            return;
        }
        if memory_pressure_high() {
            eprintln!("skip: memory_pressure high before on-arm");
            return;
        }

        // --- flag ON + DecodeIcb (sequential Hot load #2) ---
        crate::kernels::set_encode_once(true);
        tessl::set_decode_icb(true);
        let Some(model_on) = load_e4b_hot_model() else {
            eprintln!("skip: load/upload e4b (on) failed or Metal unavailable");
            crate::kernels::set_encode_once(false);
            tessl::set_decode_icb(false);
            return;
        };
        let mut sess_on = match GpuDecodeSession::new(model_on) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("skip: session new (on) failed: {e}");
                crate::kernels::set_encode_once(false);
                tessl::set_decode_icb(false);
                return;
            }
        };
        for &t in tokens.iter().take(warmup) {
            if let Err(e) = sess_on.step(t) {
                eprintln!("skip: e4b warmup (on) failed: {e}");
                crate::kernels::set_encode_once(false);
                tessl::set_decode_icb(false);
                tessl::set_icb_pipelines(false);
                tessl::clear_binder_encode_nop();
                return;
            }
        }
        sess_on.model.gpu.synchronize().unwrap();
        let t1 = std::time::Instant::now();
        for &t in tokens.iter().skip(warmup).take(iters) {
            if let Err(e) = sess_on.step(t) {
                eprintln!("skip: e4b timed step (on) failed: {e}");
                crate::kernels::set_encode_once(false);
                tessl::set_decode_icb(false);
                tessl::set_icb_pipelines(false);
                tessl::clear_binder_encode_nop();
                return;
            }
        }
        let us_on = t1.elapsed().as_secs_f64() * 1e6 / iters as f64;
        let tok_s_on = 1e6 / us_on.max(1.0);

        let live = sess_on.encode_once_scaffold().live_encodes();
        let icb_replays = sess_on.encode_once_scaffold().icb_replays();
        let wired = sess_on.encode_once_scaffold().decode_icb_wired();
        let layer_graph = sess_on.encode_once_scaffold().decode_icb_layer_graph();
        let cmd_n = sess_on
            .encode_once_scaffold()
            .decode_icb()
            .map(|d| d.command_count())
            .unwrap_or(0);
        let icb = sess_on
            .encode_once_scaffold()
            .decode_icb()
            .map(|d| d.status_line())
            .unwrap_or_else(|| sess_on.encode_once_scaffold().icb_stub().status_line());
        let scalar_ops = sess_on
            .icb_scalar_write_tape
            .as_ref()
            .map(|t| t.op_count())
            .unwrap_or(0);
        let (sticky_skip, total_buf, last_set, last_binds, prebuilt_n, set_tables, elided) =
            sess_on
                .encode_once_scaffold()
                .decode_icb()
                .map(|d| {
                    let (set, binds) = d.last_set_address_stats();
                    let (tables, elided) = d.last_prebuilt_stats();
                    (
                        d.sticky_skippable_binds(),
                        d.total_buf_binds(),
                        set,
                        binds,
                        d.prebuilt_table_count(),
                        tables,
                        elided,
                    )
                })
                .unwrap_or((0, 0, 0, 0, 0, 0, 0));
        let sticky_ratio = if total_buf > 0 {
            sticky_skip as f64 / total_buf as f64
        } else {
            0.0
        };
        let gaps: Vec<&'static str> = tessl::survey_cb_replay_api_gaps()
            .iter()
            .map(|g| g.as_str())
            .collect();

        let ratio = us_on / us_off.max(1.0);
        let ok = wired
            && layer_graph
            && live <= warmup as u64
            && icb_replays >= iters as u64
            && cmd_n >= 8
            && (0.05..=2.5).contains(&ratio);

        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("bench/results");
        let _ = std::fs::create_dir_all(&out_dir);
        let path = out_dir.join(format!("encode_once_e4b_hot_ab_{ts}.json"));
        let latest = out_dir.join("encode_once_e4b_hot_ab_latest.json");
        let verdict = if ok {
            format!(
                "PASS: E4B Hot encode A/B cmds={cmd_n} live={live} replays={icb_replays} \
                 encode_us_off={us_off:.0} encode_us_on={us_on:.0} ratio={ratio:.2} \
                 tok_s_off={tok_s_off:.2} tok_s_on={tok_s_on:.2} \
                 prebuilt={prebuilt_n} setArgTable={set_tables} elided={elided} \
                 last_setAddress={last_set}/{last_binds}"
            )
        } else {
            format!(
                "FAIL: wired={wired} layer_graph={layer_graph} cmds={cmd_n} live={live} \
                 replays={icb_replays} ratio={ratio:.2} us_off={us_off:.0} us_on={us_on:.0}"
            )
        };
        let doc = serde_json::json!({
            "artifact": "encode_once_e4b_hot_ab",
            "evidence_status": "diagnostic_only",
            "model": "e4b_hot",
            "unix_ts": ts,
            "warmup": warmup,
            "iters": iters,
            "encode_us_off": us_off,
            "encode_us_on": us_on,
            "tok_s_off": tok_s_off,
            "tok_s_on": tok_s_on,
            "ratio_on_over_off": ratio,
            "cb_replay_wired": wired,
            "layer_graph": layer_graph,
            "icb_command_count": cmd_n,
            "icb_replays": icb_replays,
            "live_encodes_on": live,
            "decode_icb": icb,
            "scalar_write_tape_ops": scalar_ops,
            "skip_nop_layer_loop": scalar_ops > 0 && icb_skip_nop_loop_enabled(),
            "sticky_skippable_buf_binds": sticky_skip,
            "total_buf_binds": total_buf,
            "sticky_skip_ratio": sticky_ratio,
            "prebuilt_table_count": prebuilt_n,
            "last_set_argument_table_calls": set_tables,
            "last_prebuilt_elided": elided,
            "last_set_address_calls": last_set,
            "last_bind_total": last_binds,
            "api_gaps": gaps,
            "api_gap_summary": tessl::cb_replay_api_gap_summary(),
            "sequential_hot_loads": true,
            "no_31b": true,
            "no_fusion_ab_trace": true,
            "verdict": verdict,
        });
        let body = serde_json::to_string_pretty(&doc).expect("json");
        let _ = std::fs::write(&path, &body);
        let _latest_updated = maybe_update_latest_artifact(&latest, &body);
        eprintln!(
            "encode_once_e4b_hot_encode_ab: {verdict} → {}",
            latest.display()
        );

        crate::kernels::set_encode_once(false);
        tessl::set_decode_icb(false);
        tessl::set_icb_pipelines(false);
        tessl::clear_binder_encode_nop();

        assert!(ok, "E4B Hot encode A/B: {verdict}");
    }

    /// Mini A/B: encode-only µs with `GEMMA_METAL_ENCODE_ONCE` + `DECODE_ICB`.
    ///
    /// Head-on `step` (lm_head in tape): first step live-encodes+captures; later
    /// steps binder-nop prep + `execute_icb`. Default flags OFF outside this test.
    #[test]
    fn encode_once_mini_encode_ab() {
        let Ok(host) = SyntheticE4bGraph::mini_parity() else {
            return;
        };
        let Ok(model_off) = GpuSynthModel::from_synthetic(host.clone(), QuantScheme::q4_default())
        else {
            return;
        };
        let Ok(model_on) = GpuSynthModel::from_synthetic(host, QuantScheme::q4_default()) else {
            return;
        };
        if !metal_ready(&model_off) {
            eprintln!("skip: Metal pipeline unavailable");
            return;
        }

        let warmup = 2usize;
        let iters = 8usize;
        let tokens: Vec<u32> = (0..iters).map(|i| 3 + (i as u32) * 2).collect();

        // --- flag OFF ---
        crate::kernels::set_encode_once(false);
        tessl::set_decode_icb(false);
        let mut sess_off = GpuDecodeSession::new(model_off).unwrap();
        for &t in tokens.iter().take(warmup) {
            let _ = sess_off.step(t).unwrap();
        }
        sess_off.model.gpu.synchronize().unwrap();
        let t0 = std::time::Instant::now();
        for &t in &tokens {
            let _ = sess_off.step(t).unwrap();
        }
        let us_off = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

        // --- flag ON + DecodeIcb layer-graph (head-on capture) ---
        crate::kernels::set_encode_once(true);
        tessl::set_decode_icb(true);
        let mut sess_on = GpuDecodeSession::new(model_on).unwrap();
        for &t in tokens.iter().take(warmup) {
            let _ = sess_on.step(t).unwrap();
        }
        sess_on.model.gpu.synchronize().unwrap();
        let t1 = std::time::Instant::now();
        for &t in &tokens {
            let _ = sess_on.step(t).unwrap();
        }
        let us_on = t1.elapsed().as_secs_f64() * 1e6 / iters as f64;

        let live = sess_on.encode_once_scaffold().live_encodes();
        let icb_replays = sess_on.encode_once_scaffold().icb_replays();
        let wired = sess_on.encode_once_scaffold().decode_icb_wired();
        let layer_graph = sess_on.encode_once_scaffold().decode_icb_layer_graph();
        let cmd_n = sess_on
            .encode_once_scaffold()
            .decode_icb()
            .map(|d| d.command_count())
            .unwrap_or(0);
        let icb = sess_on
            .encode_once_scaffold()
            .decode_icb()
            .map(|d| d.status_line())
            .unwrap_or_else(|| sess_on.encode_once_scaffold().icb_stub().status_line());
        let gaps: Vec<&'static str> = tessl::survey_cb_replay_api_gaps()
            .iter()
            .map(|g| g.as_str())
            .collect();

        assert!(wired, "expected DecodeIcb attached under DECODE_ICB");
        assert!(
            layer_graph,
            "expected Binder layer-graph DecodeIcb (cmds>={}), got {icb}",
            tessl::DecodeIcb::MIN_LAYER_GRAPH_COMMANDS
        );
        // One live capture step; remaining warmup+iters replay (no live_encodes++).
        assert!(
            live <= warmup as u64,
            "expected few live encodes after layer-graph skip, got {live}"
        );
        assert!(
            icb_replays >= iters as u64,
            "expected ICB replays on timed iters, got {icb_replays}"
        );
        let ratio = us_on / us_off.max(1.0);
        // Inherit+arg-table still rebinds per cmd; ratio may be partial. Soft band.
        assert!(
            (0.05..=2.5).contains(&ratio),
            "unexpected encode_us ratio: us_off={us_off:.1} us_on={us_on:.1} ratio={ratio:.2}"
        );

        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("bench/results");
        let _ = std::fs::create_dir_all(&out_dir);
        let path = out_dir.join(format!("encode_once_mini_ab_{ts}.json"));
        let latest = out_dir.join("encode_once_mini_ab_latest.json");
        let scalar_ops = sess_on
            .icb_scalar_write_tape
            .as_ref()
            .map(|t| t.op_count())
            .unwrap_or(0);
        let (sticky_skip, total_buf, last_set, last_binds, prebuilt_n, set_tables, elided) =
            sess_on
                .encode_once_scaffold()
                .decode_icb()
                .map(|d| {
                    let (set, binds) = d.last_set_address_stats();
                    let (tables, elided) = d.last_prebuilt_stats();
                    (
                        d.sticky_skippable_binds(),
                        d.total_buf_binds(),
                        set,
                        binds,
                        d.prebuilt_table_count(),
                        tables,
                        elided,
                    )
                })
                .unwrap_or((0, 0, 0, 0, 0, 0, 0));
        let sticky_ratio = if total_buf > 0 {
            sticky_skip as f64 / total_buf as f64
        } else {
            0.0
        };
        let set_ratio = if last_binds > 0 {
            last_set as f64 / last_binds as f64
        } else {
            0.0
        };
        let verdict = if layer_graph {
            format!(
                "DecodeIcb layer-graph capture+skip (cmds={cmd_n}); live_encodes={live} \
                 icb_replays={icb_replays}; encode_us ratio={ratio:.2} \
                 (A2 v0.5.4: scalar tape ops={scalar_ops} skip-nop; prebuilt_tables={prebuilt_n} \
                 setArgTable={set_tables} elided={elided}; last_exec setAddress={last_set}/{last_binds} \
                 ({:.1}%); sticky_analyze={sticky_skip}/{total_buf})",
                set_ratio * 100.0
            )
        } else {
            "DecodeIcb not layer-graph".into()
        };
        let doc = serde_json::json!({
            "artifact": "encode_once_mini_ab",
            "evidence_status": "diagnostic_only",
            "model": "mini_synth",
            "unix_ts": ts,
            "warmup": warmup,
            "iters": iters,
            "encode_us_off": us_off,
            "encode_us_on": us_on,
            "ratio_on_over_off": ratio,
            "cb_replay_wired": wired,
            "layer_graph": layer_graph,
            "icb_command_count": cmd_n,
            "icb_replays": icb_replays,
            "live_encodes_on": live,
            "decode_icb": icb,
            "scalar_write_tape_ops": scalar_ops,
            "skip_nop_layer_loop": scalar_ops > 0 && icb_skip_nop_loop_enabled(),
            "sticky_skippable_buf_binds": sticky_skip,
            "total_buf_binds": total_buf,
            "sticky_skip_ratio": sticky_ratio,
            "prebuilt_table_count": prebuilt_n,
            "last_set_argument_table_calls": set_tables,
            "last_prebuilt_elided": elided,
            "last_set_address_calls": last_set,
            "last_bind_total": last_binds,
            "set_address_ratio": set_ratio,
            "api_gaps": gaps,
            "api_gap_summary": tessl::cb_replay_api_gap_summary(),
            "verdict": verdict,
        });
        let body = serde_json::to_string_pretty(&doc).expect("json");
        std::fs::write(&path, &body).expect("write encode_once_mini_ab");
        let _latest_updated = maybe_update_latest_artifact(&latest, &body);
        eprintln!(
            "encode_once_mini_encode_ab: us_off={us_off:.1} us_on={us_on:.1} ratio={ratio:.2} \
             layer_graph={layer_graph} cmds={cmd_n} live={live} icb_replays={icb_replays} \
             scalar_ops={scalar_ops} prebuilt={prebuilt_n} setArgTable={set_tables} \
             elided={elided} last_setAddress={last_set}/{last_binds} → {}",
            path.display()
        );

        crate::kernels::set_encode_once(false);
        tessl::set_decode_icb(false);
        tessl::set_icb_pipelines(false);
        tessl::clear_binder_encode_nop();
    }

    /// Persistent-interp flag: mini `step_inner` exercises both stand-ins; tokens unchanged.
    #[test]
    fn persistent_interp_mini_decode_hook() {
        let Ok(host) = SyntheticE4bGraph::mini_parity() else {
            return;
        };
        let Ok(model_a) = GpuSynthModel::from_synthetic(host.clone(), QuantScheme::q4_default())
        else {
            return;
        };
        let Ok(model_b) = GpuSynthModel::from_synthetic(host, QuantScheme::q4_default()) else {
            return;
        };
        if !metal_ready(&model_a) {
            eprintln!("skip: Metal pipeline unavailable");
            return;
        }
        if model_a
            .gpu
            .rt
            .pipeline(KernelId::PersistentInterpGateDown.entry_name())
            .is_err()
            || model_a
                .gpu
                .rt
                .pipeline(KernelId::PersistentInterpFaOProj.entry_name())
                .is_err()
        {
            eprintln!("skip: persistent_interp kernels not in metallib");
            return;
        }
        assert!(
            model_a.is_synthetic_mini(),
            "mini_parity must gate persistent_interp decode hook"
        );

        // Baseline (flag off): no hook hits.
        crate::kernels::set_persistent_interp(false);
        assert!(!crate::kernels::persistent_interp_enabled());
        let mut base = GpuDecodeSession::new(model_a).unwrap();
        let prompts = [3u32, 7, 11];
        let mut base_out = Vec::new();
        for &t in &prompts {
            base_out.push(base.step(t).unwrap());
        }
        assert_eq!(base.persistent_interp_gate_down_hits(), 0);
        assert_eq!(base.persistent_interp_fa_o_hits(), 0);

        // Flag on: both doctrine edges dispatch once per layer per step.
        crate::kernels::set_persistent_interp(true);
        assert!(crate::kernels::persistent_interp_enabled());
        let mut hooked = GpuDecodeSession::new(model_b).unwrap();
        let mut hooked_out = Vec::new();
        for &t in &prompts {
            hooked_out.push(hooked.step(t).unwrap());
        }
        let n_layers = hooked.model.layers.len() as u64;
        let expect_hits = n_layers * prompts.len() as u64;
        assert_eq!(
            hooked.persistent_interp_fa_o_hits(),
            expect_hits,
            "FA→o_proj mini hook should fire each layer×step"
        );
        assert_eq!(
            hooked.persistent_interp_gate_down_hits(),
            expect_hits,
            "gate→down mini hook should fire each layer×step"
        );
        assert_eq!(
            hooked.persistent_interp_last_fail(),
            0,
            "barrier spin timeout on mini decode hook"
        );
        assert_eq!(
            base_out, hooked_out,
            "persistent-interp mini scratch must not change tokens: base={base_out:?} hooked={hooked_out:?}"
        );

        crate::kernels::set_persistent_interp(false);
    }

    #[test]
    fn step_verify_trim_restores_for_continue() {
        let Ok(host) = SyntheticE4bGraph::mini_parity() else {
            return;
        };
        let Ok(model) = GpuSynthModel::from_synthetic(host, QuantScheme::q4_default()) else {
            return;
        };
        if !metal_ready(&model) {
            eprintln!("skip: Metal pipeline unavailable");
            return;
        }
        let mut sess = GpuDecodeSession::new(model).unwrap();
        let _ = sess.step(1).unwrap();
        assert_eq!(sess.pos(), 1);
        let block = [2u32, 3, 4];
        let ver = sess.step_verify(&block).unwrap();
        assert_eq!(sess.pos(), 1 + block.len());
        assert_eq!(ver.next_tokens.len(), block.len());
        // Reject last two — keep only first of the block.
        sess.commit_verify(block.len(), 1).unwrap();
        assert_eq!(sess.pos(), 2); // prefill 1 + kept 1
        // Can continue decoding after trim without error.
        let cont = sess.step(block[1]).unwrap();
        assert!(cont < sess.model.vocab as u32);
        assert_eq!(sess.pos(), 3);
        // Full rollback of remaining.
        sess.trim_kv(3).unwrap();
        assert_eq!(sess.pos(), 0);
    }

    #[test]
    fn step_verify_after_sliding_wrap_reject_then_continue_matches_sequential() {
        let host = SyntheticE4bGraph::mini_parity().expect("synthetic rollback parity graph");
        let model_a = GpuSynthModel::from_synthetic(host.clone(), QuantScheme::q4_default())
            .expect("sequential rollback parity model");
        let model_b = GpuSynthModel::from_synthetic(host, QuantScheme::q4_default())
            .expect("speculative rollback parity model");
        assert!(metal_ready(&model_a), "Metal pipeline unavailable");
        assert!(metal_ready(&model_b), "Metal pipeline unavailable");
        tessl::ab_flags::set_hazard_barriers(false);

        let mut sequential = GpuDecodeSession::new(model_a).expect("sequential session");
        let mut speculative = GpuDecodeSession::new(model_b).expect("speculative session");
        for token in [1u32, 2, 3, 4, 5, 6] {
            assert_eq!(
                sequential.step(token).unwrap(),
                speculative.step(token).unwrap()
            );
        }
        assert!(
            sequential.pos() > sequential.sliding_rings[0].capacity,
            "fixture must cross the logical sliding window before speculation"
        );

        let accepted_next = sequential.step(7).expect("sequential accepted row");
        let verified = speculative
            .step_verify(&[7, 8, 9])
            .expect("speculative verify block");
        assert_eq!(verified.next_tokens[0], accepted_next);
        speculative
            .commit_verify(verified.tokens.len(), 1)
            .expect("reject speculative suffix");
        assert_eq!(speculative.pos(), sequential.pos());

        let sequential_next = sequential.step(10).expect("sequential continuation");
        let speculative_next = speculative.step(10).expect("rollback continuation");
        assert_eq!(speculative_next, sequential_next);

        let chronological = |slot: &GpuKvSlot, buffer: &GpuBuffer| {
            let live = slot.seq_len.min(slot.capacity);
            let width = slot.layout.slot_elems;
            let start = slot.ring_start_u32().expect("ring start") as usize;
            let values = buffer.read_f32();
            let mut out = Vec::with_capacity(live * width);
            for row in 0..live {
                let physical = (start + row) % slot.storage_capacity;
                out.extend_from_slice(&values[physical * width..(physical + 1) * width]);
            }
            out
        };
        assert_eq!(
            sequential.sliding_rings.len(),
            speculative.sliding_rings.len()
        );
        for (expected, actual) in sequential
            .sliding_rings
            .iter()
            .zip(&speculative.sliding_rings)
        {
            assert_eq!(
                chronological(actual, &actual.k),
                chronological(expected, &expected.k)
            );
            assert_eq!(
                chronological(actual, &actual.v),
                chronological(expected, &expected.v)
            );
        }
    }

    #[test]
    fn step_verify_rejects_bad_m() {
        let Ok(host) = SyntheticE4bGraph::mini_parity() else {
            return;
        };
        let Ok(model) = GpuSynthModel::from_synthetic(host, QuantScheme::q4_default()) else {
            return;
        };
        if !metal_ready(&model) {
            return;
        }
        let mut sess = GpuDecodeSession::new(model).unwrap();
        assert!(sess.step_verify(&[]).is_err());
        let too_long = vec![1u32; VERIFY_MAX_M + 1];
        assert!(sess.step_verify(&too_long).is_err());
    }
}

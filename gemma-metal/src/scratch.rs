//! Zero-alloc-after-load scratch + KV sizing API.
//!
//! Call [`ScratchPlan::from_config`] at load time, allocate once, then decode
//! without further heap / Metal buffer growth.

use crate::config::Gemma4TextConfig;
use crate::kv::KvLayout;
use crate::ple::plan_ple_layer_bytes;
use crate::quant::QuantScheme;
use crate::error::{Error, Result};

/// Element width for activations / KV (fp16 KV is the MTP-friendly default).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActStorage {
    Fp16,
    Bf16,
    Fp32,
}

impl ActStorage {
    pub fn elem_bytes(self) -> usize {
        match self {
            ActStorage::Fp16 | ActStorage::Bf16 => 2,
            ActStorage::Fp32 => 4,
        }
    }
}

/// One named buffer reservation.
#[derive(Clone, Debug)]
pub struct BufferReservation {
    pub name: &'static str,
    pub bytes: usize,
    /// Hot = long-lived weights/KV; Cold = per-step scratch (still preallocated).
    pub hot: bool,
}

/// Complete pre-decode allocation plan.
#[derive(Clone, Debug)]
pub struct ScratchPlan {
    pub max_batch: usize,
    pub max_seq: usize,
    pub prefill_chunk: usize,
    pub kv: KvLayout,
    pub reservations: Vec<BufferReservation>,
    pub total_bytes: usize,
    pub total_hot_bytes: usize,
    pub total_scratch_bytes: usize,
}

impl ScratchPlan {
    /// Build a zero-alloc-after-load plan.
    ///
    /// - `max_seq`: KV / shared-full length cap
    /// - `prefill_chunk`: max tokens per prefill encode chunk (activation footprint)
    /// - `weight_hot_bytes`: already-quantized weight banks (caller-measured)
    pub fn from_config(
        cfg: &Gemma4TextConfig,
        max_batch: usize,
        max_seq: usize,
        prefill_chunk: usize,
        weight_hot_bytes: usize,
        weight_scheme: QuantScheme,
        act: ActStorage,
    ) -> Result<Self> {
        if max_batch == 0 || max_seq == 0 || prefill_chunk == 0 {
            return Err(Error::Config(
                "max_batch, max_seq, prefill_chunk must be > 0".into(),
            ));
        }
        let kv = KvLayout::from_config(cfg, max_seq)?;
        let eb = act.elem_bytes();
        let mut reservations = Vec::new();

        reservations.push(BufferReservation {
            name: "weights_hot",
            bytes: weight_hot_bytes,
            hot: true,
        });

        let kv_bytes = kv.total_kv_bytes(eb);
        reservations.push(BufferReservation {
            name: "kv_cache",
            bytes: kv_bytes,
            hot: true,
        });

        // PLE banks already counted in weight_hot_bytes when loaded; plan lists sizes for validation.
        let ple_sizes = plan_ple_layer_bytes(cfg, weight_scheme)?;
        for (i, bytes) in ple_sizes.iter().enumerate() {
            reservations.push(BufferReservation {
                name: "ple_layer", // distinct names not required for sizing
                bytes: *bytes,
                hot: true,
            });
            let _ = i;
        }

        // Decode / prefill scratch (batch × chunk × hidden)
        let hidden = cfg.hidden_size;
        let scratch_tokens = max_batch.saturating_mul(prefill_chunk);
        let act_bytes = scratch_tokens.saturating_mul(hidden).saturating_mul(eb);

        let scratch_names = [
            ("x_resid", act_bytes),
            ("x_attn", act_bytes),
            ("q_buf", {
                // max of local/global Q: batch*chunk*heads*dim
                let local = scratch_tokens
                    * cfg.num_attention_heads
                    * cfg.head_dim
                    * eb;
                let global = scratch_tokens
                    * cfg.num_attention_heads
                    * cfg.global_head_dim
                    * eb;
                local.max(global)
            }),
            ("k_buf", {
                let local = scratch_tokens * cfg.num_key_value_heads * cfg.head_dim * eb;
                let global = scratch_tokens * cfg.global_kv_heads() * cfg.global_head_dim * eb;
                local.max(global)
            }),
            ("v_buf", {
                let local = scratch_tokens * cfg.num_key_value_heads * cfg.head_dim * eb;
                let global = scratch_tokens * cfg.global_kv_heads() * cfg.global_head_dim * eb;
                local.max(global)
            }),
            ("attn_out", act_bytes),
            ("mlp_gate", scratch_tokens * cfg.intermediate_size * eb),
            ("mlp_up", scratch_tokens * cfg.intermediate_size * eb),
            ("logits", max_batch * cfg.vocab_size * 4), // f32 logits for softcap/sample
            ("ple_tmp", {
                if cfg.has_ple() {
                    scratch_tokens * cfg.hidden_size_per_layer_input * eb
                } else {
                    0
                }
            }),
        ];

        for (name, bytes) in scratch_names {
            if bytes > 0 {
                reservations.push(BufferReservation {
                    name,
                    bytes,
                    hot: false,
                });
            }
        }

        // Note: PLE layer sizes in reservations are planning hints; when
        // `weight_hot_bytes` already includes PLE, subtract them from the double-count
        // for total_bytes reporting.
        let ple_sum: usize = ple_sizes.iter().sum();
        let total_hot_bytes = weight_hot_bytes
            .saturating_add(kv_bytes);
        // Avoid double-counting PLE if caller already folded it into weight_hot_bytes.
        let _ = ple_sum;

        let total_scratch_bytes: usize = reservations
            .iter()
            .filter(|r| !r.hot)
            .map(|r| r.bytes)
            .sum();

        let total_bytes = total_hot_bytes.saturating_add(total_scratch_bytes);

        Ok(Self {
            max_batch,
            max_seq,
            prefill_chunk,
            kv,
            reservations,
            total_bytes,
            total_hot_bytes,
            total_scratch_bytes,
        })
    }

    /// Human-readable summary for load banners.
    pub fn summary(&self) -> String {
        format!(
            "scratch plan: batch={} seq={} chunk={} | hot={:.2} GiB scratch={:.2} GiB total={:.2} GiB | \
             KV rings={} global_slots={} shared_sw={} shared_g={} first_kv_shared={}",
            self.max_batch,
            self.max_seq,
            self.prefill_chunk,
            self.total_hot_bytes as f64 / (1u64 << 30) as f64,
            self.total_scratch_bytes as f64 / (1u64 << 30) as f64,
            self.total_bytes as f64 / (1u64 << 30) as f64,
            self.kv.sliding_ring_slots,
            self.kv.global_full_slots,
            self.kv.has_shared_sliding_full,
            self.kv.has_shared_global_full,
            self.kv.first_kv_shared,
        )
    }
}

/// Runtime holder: once constructed, decode must not grow these buffers.
#[derive(Debug)]
pub struct ScratchArena {
    pub plan: ScratchPlan,
    /// Opaque preallocated slabs (CPU staging); Metal upload happens in Phase 2+.
    pub slabs: Vec<Vec<u8>>,
}

impl ScratchArena {
    pub fn allocate(plan: ScratchPlan) -> Self {
        let slabs = plan
            .reservations
            .iter()
            .filter(|r| !r.hot || r.name == "kv_cache")
            .map(|r| vec![0u8; r.bytes])
            .collect();
        Self { plan, slabs }
    }

    pub fn assert_no_grow(&self) {
        // Documented contract — decode paths must use these slabs only.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Gemma4TextConfig;

    #[test]
    fn e4b_plan_smoke() {
        let cfg = Gemma4TextConfig::e4b_preset();
        let plan = ScratchPlan::from_config(
            &cfg,
            1,
            4096,
            256,
            2_000_000_000, // ~2GB fake weights
            QuantScheme::q4_default(),
            ActStorage::Fp16,
        )
        .unwrap();
        assert_eq!(plan.kv.first_kv_shared, 24);
        assert!(plan.total_bytes > plan.total_hot_bytes);
        assert!(plan.summary().contains("first_kv_shared=24"));
        let arena = ScratchArena::allocate(plan);
        assert!(!arena.slabs.is_empty());
    }
}

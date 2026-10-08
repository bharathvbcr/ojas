//! Dual KV layouts: sliding ring + global full, with E4B KV-share consumer map.
//!
//! E4B: `first_kv_shared = num_hidden_layers - num_kv_shared_layers` (42−18=24).
//! Layers `[0, first_kv_shared)` are producers with K/V weights.
//! Layers `[first_kv_shared, L)` are consumers — no K/V weights; they read
//! `shared_kv_states[layer_type]` (full-length for both types — not the SWA ring).

use crate::config::{Gemma4TextConfig, LayerType};
use crate::error::{Error, Result};

/// How a layer participates in KV cache.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KvRole {
    /// Owns a cache slot (has K/V projections).
    Producer { slot: KvSlotId },
    /// Reuses shared state for its layer type (no K/V weights).
    Consumer { shared: SharedKvId },
}

/// Per-producer cache slot kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KvSlotId {
    /// Ring buffer of width `sliding_window`, head_dim = local (256).
    SlidingRing { producer_index: usize },
    /// Full-context buffer, head_dim = global (512).
    GlobalFull { producer_index: usize },
}

/// Shared KV used by consumer layers (and updated by the last producer of that type).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SharedKvId {
    /// Full-length shared sliding source (not truncated to window).
    SlidingFull,
    /// Full-length shared global source.
    GlobalFull,
}

/// One layer's KV wiring.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvLayerMap {
    pub layer: usize,
    pub layer_type: LayerType,
    pub role: KvRole,
    /// Last producer layer of the same type at or before this layer (for debugging).
    pub source_producer_layer: Option<usize>,
}

/// Full model KV consumer/producer map + layout sizes.
#[derive(Clone, Debug)]
pub struct KvLayout {
    pub first_kv_shared: usize,
    pub layers: Vec<KvLayerMap>,
    pub sliding_ring_slots: usize,
    pub global_full_slots: usize,
    /// Always true when any consumer exists (E4B).
    pub has_shared_sliding_full: bool,
    pub has_shared_global_full: bool,
    pub sliding_window: usize,
    pub max_seq: usize,
    pub local_head_dim: usize,
    pub global_head_dim: usize,
    pub local_kv_heads: usize,
    pub global_kv_heads: usize,
}

impl KvLayout {
    /// Build dual-KV map from text config.
    ///
    /// `max_seq` caps global / shared-full buffers (batch=1 assumed for sizing).
    pub fn from_config(cfg: &Gemma4TextConfig, max_seq: usize) -> Result<Self> {
        if max_seq == 0 {
            return Err(Error::Kv("max_seq must be > 0".into()));
        }
        let first_kv_shared = cfg.first_kv_shared();
        let sliding_window = cfg.sliding_window_or(512);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let mut sliding_ring_slots = 0usize;
        let mut global_full_slots = 0usize;
        let mut last_sliding_producer: Option<usize> = None;
        let mut last_global_producer: Option<usize> = None;

        for layer in 0..cfg.num_hidden_layers {
            let layer_type = cfg.layer_type(layer)?;
            let is_producer = layer < first_kv_shared;
            let (role, source_producer_layer) = if is_producer {
                let role = match layer_type {
                    LayerType::SlidingAttention => {
                        let slot = KvSlotId::SlidingRing {
                            producer_index: sliding_ring_slots,
                        };
                        sliding_ring_slots += 1;
                        last_sliding_producer = Some(layer);
                        KvRole::Producer { slot }
                    }
                    LayerType::FullAttention => {
                        let slot = KvSlotId::GlobalFull {
                            producer_index: global_full_slots,
                        };
                        global_full_slots += 1;
                        last_global_producer = Some(layer);
                        KvRole::Producer { slot }
                    }
                };
                let src = match layer_type {
                    LayerType::SlidingAttention => last_sliding_producer,
                    LayerType::FullAttention => last_global_producer,
                };
                (role, src)
            } else {
                let shared = match layer_type {
                    LayerType::SlidingAttention => SharedKvId::SlidingFull,
                    LayerType::FullAttention => SharedKvId::GlobalFull,
                };
                let src = match layer_type {
                    LayerType::SlidingAttention => last_sliding_producer,
                    LayerType::FullAttention => last_global_producer,
                };
                (KvRole::Consumer { shared }, src)
            };
            layers.push(KvLayerMap {
                layer,
                layer_type,
                role,
                source_producer_layer,
            });
        }

        let has_consumers = first_kv_shared < cfg.num_hidden_layers;
        // Shared full buffers only needed when consumers exist.
        let has_shared_sliding_full =
            has_consumers && layers.iter().any(|l| {
                matches!(
                    l.role,
                    KvRole::Consumer {
                        shared: SharedKvId::SlidingFull
                    }
                )
            });
        let has_shared_global_full = has_consumers
            && layers.iter().any(|l| {
                matches!(
                    l.role,
                    KvRole::Consumer {
                        shared: SharedKvId::GlobalFull
                    }
                )
            });

        Ok(Self {
            first_kv_shared,
            layers,
            sliding_ring_slots,
            global_full_slots,
            has_shared_sliding_full,
            has_shared_global_full,
            sliding_window,
            max_seq,
            local_head_dim: cfg.head_dim,
            global_head_dim: cfg.global_head_dim,
            local_kv_heads: cfg.num_key_value_heads,
            global_kv_heads: cfg.global_kv_heads(),
        })
    }

    pub fn layer(&self, i: usize) -> Result<&KvLayerMap> {
        self.layers
            .get(i)
            .ok_or_else(|| Error::Kv(format!("layer {i} out of range")))
    }

    pub fn is_producer(&self, layer: usize) -> bool {
        layer < self.first_kv_shared
    }

    pub fn is_consumer(&self, layer: usize) -> bool {
        layer >= self.first_kv_shared
    }

    /// Bytes for one K or V tensor at `seq` × heads × dim × elem_bytes (batch=1).
    fn kv_tensor_bytes(seq: usize, heads: usize, dim: usize, elem_bytes: usize) -> usize {
        seq.saturating_mul(heads)
            .saturating_mul(dim)
            .saturating_mul(elem_bytes)
    }

    /// Total KV cache bytes for K+V (fp16 → 2 bytes). Does not include scratch.
    pub fn total_kv_bytes(&self, elem_bytes: usize) -> usize {
        let mut total = 0usize;
        // Sliding rings: each slot holds K and V of width=window
        let ring = Self::kv_tensor_bytes(
            self.sliding_window,
            self.local_kv_heads,
            self.local_head_dim,
            elem_bytes,
        ) * 2;
        total = total.saturating_add(ring.saturating_mul(self.sliding_ring_slots));

        let global = Self::kv_tensor_bytes(
            self.max_seq,
            self.global_kv_heads,
            self.global_head_dim,
            elem_bytes,
        ) * 2;
        total = total.saturating_add(global.saturating_mul(self.global_full_slots));

        if self.has_shared_sliding_full {
            // Full-length shared sliding (consumers) — plan: not truncated SWA ring.
            let shared_sw = Self::kv_tensor_bytes(
                self.max_seq,
                self.local_kv_heads,
                self.local_head_dim,
                elem_bytes,
            ) * 2;
            total = total.saturating_add(shared_sw);
        }
        if self.has_shared_global_full {
            let shared_g = Self::kv_tensor_bytes(
                self.max_seq,
                self.global_kv_heads,
                self.global_head_dim,
                elem_bytes,
            ) * 2;
            total = total.saturating_add(shared_g);
        }
        total
    }
}

/// Host-side sliding KV ring (producer path). Chronological densify feeds FA.
#[derive(Clone, Debug)]
pub struct KvRingBuffer {
    pub capacity: usize,
    pub heads: usize,
    pub dim: usize,
    /// Next write slot in `[0, capacity)`.
    pub head: usize,
    /// Total tokens appended (may exceed capacity).
    pub seq_len: usize,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}

impl KvRingBuffer {
    pub fn new(capacity: usize, heads: usize, dim: usize) -> Self {
        let slot = capacity.saturating_mul(heads).saturating_mul(dim);
        Self {
            capacity,
            heads,
            dim,
            head: 0,
            seq_len: 0,
            k: vec![0.0; slot],
            v: vec![0.0; slot],
        }
    }

    fn slot_elems(&self) -> usize {
        self.heads * self.dim
    }

    /// Append one timestep of K/V (`heads * dim` each).
    pub fn append(&mut self, k_t: &[f32], v_t: &[f32]) -> Result<()> {
        let n = self.slot_elems();
        if k_t.len() != n || v_t.len() != n {
            return Err(Error::Kv(format!(
                "ring append len k={} v={} expected {}",
                k_t.len(),
                v_t.len(),
                n
            )));
        }
        if self.capacity == 0 {
            return Err(Error::Kv("ring capacity 0".into()));
        }
        let off = self.head * n;
        self.k[off..off + n].copy_from_slice(k_t);
        self.v[off..off + n].copy_from_slice(v_t);
        self.head = (self.head + 1) % self.capacity;
        self.seq_len += 1;
        Ok(())
    }

    /// Chronological densify for FA. Returns `(k_dense, v_dense, kv_pos_offset, tkv)`.
    pub fn densify(&self) -> (Vec<f32>, Vec<f32>, u32, u32) {
        let filled = self.seq_len.min(self.capacity);
        if filled == 0 {
            return (Vec::new(), Vec::new(), 0, 0);
        }
        let n = self.slot_elems();
        let mut k_out = vec![0f32; filled * n];
        let mut v_out = vec![0f32; filled * n];
        let kv_pos_offset = if self.seq_len <= self.capacity {
            0u32
        } else {
            (self.seq_len - self.capacity) as u32
        };
        let start = if self.seq_len <= self.capacity {
            0
        } else {
            self.head // oldest slot after wrap
        };
        for i in 0..filled {
            let src = (start + i) % self.capacity;
            let s = src * n;
            let d = i * n;
            k_out[d..d + n].copy_from_slice(&self.k[s..s + n]);
            v_out[d..d + n].copy_from_slice(&self.v[s..s + n]);
        }
        (k_out, v_out, kv_pos_offset, filled as u32)
    }
}

/// Full-length shared KV (E4B consumer source — not truncated to SWA window).
#[derive(Clone, Debug)]
pub struct SharedKvBuffer {
    pub max_seq: usize,
    pub heads: usize,
    pub dim: usize,
    pub seq_len: usize,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}

impl SharedKvBuffer {
    pub fn new(max_seq: usize, heads: usize, dim: usize) -> Self {
        let n = max_seq.saturating_mul(heads).saturating_mul(dim);
        Self {
            max_seq,
            heads,
            dim,
            seq_len: 0,
            k: vec![0.0; n],
            v: vec![0.0; n],
        }
    }

    fn slot_elems(&self) -> usize {
        self.heads * self.dim
    }

    pub fn append(&mut self, k_t: &[f32], v_t: &[f32]) -> Result<()> {
        let n = self.slot_elems();
        if k_t.len() != n || v_t.len() != n {
            return Err(Error::Kv(format!(
                "shared append len k={} v={} expected {}",
                k_t.len(),
                v_t.len(),
                n
            )));
        }
        if self.seq_len >= self.max_seq {
            return Err(Error::Kv("shared KV full".into()));
        }
        let off = self.seq_len * n;
        self.k[off..off + n].copy_from_slice(k_t);
        self.v[off..off + n].copy_from_slice(v_t);
        self.seq_len += 1;
        Ok(())
    }

    /// View densified `[0, seq_len)` for consumer FA (kv_pos_offset = 0).
    pub fn densify(&self) -> (Vec<f32>, Vec<f32>, u32, u32) {
        let n = self.slot_elems();
        let t = self.seq_len;
        (
            self.k[..t * n].to_vec(),
            self.v[..t * n].to_vec(),
            0,
            t as u32,
        )
    }
}

/// Resolve which densified K/V a layer should attend over (producer ring vs shared).
pub fn consumer_kv_alias(
    role: &KvRole,
    sliding_shared: &SharedKvBuffer,
    global_shared: &SharedKvBuffer,
) -> Result<(Vec<f32>, Vec<f32>, u32, u32)> {
    match role {
        KvRole::Consumer {
            shared: SharedKvId::SlidingFull,
        } => Ok(sliding_shared.densify()),
        KvRole::Consumer {
            shared: SharedKvId::GlobalFull,
        } => Ok(global_shared.densify()),
        KvRole::Producer { .. } => Err(Error::Kv(
            "consumer_kv_alias: layer is a producer, not a consumer".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Gemma4TextConfig;

    #[test]
    fn e4b_first_kv_shared_24() {
        let cfg = Gemma4TextConfig::e4b_preset();
        let layout = KvLayout::from_config(&cfg, 4096).unwrap();
        assert_eq!(layout.first_kv_shared, 24);
        assert!(layout.is_producer(23));
        assert!(layout.is_consumer(24));
        assert!(layout.is_consumer(41));
        // 42 layers, pattern 5 sliding + 1 full → 35 sliding + 7 global total
        // Producers are first 24 layers: indices 0..23
        // Among 0..23: full at 5,11,17,23 → 4 global producers; 20 sliding producers
        assert_eq!(layout.sliding_ring_slots, 20);
        assert_eq!(layout.global_full_slots, 4);
        assert!(layout.has_shared_sliding_full);
        assert!(layout.has_shared_global_full);

        let consumer = layout.layer(24).unwrap();
        assert!(matches!(
            consumer.role,
            KvRole::Consumer {
                shared: SharedKvId::SlidingFull
            }
        ));
        // Layer 24 is sliding (25th layer, index 24): 24%6 = 0 → not full → sliding
        assert_eq!(consumer.layer_type, LayerType::SlidingAttention);
        assert_eq!(consumer.source_producer_layer, Some(22)); // last sliding producer before share

        let last = layout.layer(41).unwrap();
        assert_eq!(last.layer_type, LayerType::FullAttention);
        assert!(matches!(
            last.role,
            KvRole::Consumer {
                shared: SharedKvId::GlobalFull
            }
        ));
    }

    #[test]
    fn b31_all_producers() {
        let cfg = Gemma4TextConfig::b31_preset();
        let layout = KvLayout::from_config(&cfg, 8192).unwrap();
        assert_eq!(layout.first_kv_shared, 60);
        assert!(!layout.has_shared_sliding_full);
        assert!(!layout.has_shared_global_full);
        // 60 layers: 50 sliding + 10 global
        assert_eq!(layout.sliding_ring_slots, 50);
        assert_eq!(layout.global_full_slots, 10);
        assert!(layout.layers.iter().all(|l| matches!(l.role, KvRole::Producer { .. })));
    }

    #[test]
    fn kv_bytes_positive() {
        let cfg = Gemma4TextConfig::e4b_preset();
        let layout = KvLayout::from_config(&cfg, 4096).unwrap();
        let n = layout.total_kv_bytes(2);
        assert!(n > 0);
    }

    #[test]
    fn ring_densify_no_wrap() {
        let mut ring = KvRingBuffer::new(4, 1, 2);
        ring.append(&[1.0, 2.0], &[10.0, 20.0]).unwrap();
        ring.append(&[3.0, 4.0], &[30.0, 40.0]).unwrap();
        let (k, v, off, t) = ring.densify();
        assert_eq!(t, 2);
        assert_eq!(off, 0);
        assert_eq!(k, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(v, vec![10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn ring_densify_wraps_chronological() {
        let mut ring = KvRingBuffer::new(2, 1, 1);
        ring.append(&[1.0], &[10.0]).unwrap();
        ring.append(&[2.0], &[20.0]).unwrap();
        ring.append(&[3.0], &[30.0]).unwrap(); // drops 1
        let (k, v, off, t) = ring.densify();
        assert_eq!(t, 2);
        assert_eq!(off, 1); // absolute positions 1,2
        assert_eq!(k, vec![2.0, 3.0]);
        assert_eq!(v, vec![20.0, 30.0]);
    }

    #[test]
    fn shared_kv_consumer_alias() {
        let cfg = Gemma4TextConfig::e4b_preset();
        let layout = KvLayout::from_config(&cfg, 16).unwrap();
        let mut shared = SharedKvBuffer::new(16, 2, 4);
        shared
            .append(&[1.0; 8], &[2.0; 8])
            .unwrap();
        let role = &layout.layer(24).unwrap().role;
        let (k, _v, off, t) = consumer_kv_alias(role, &shared, &SharedKvBuffer::new(16, 2, 8)).unwrap();
        assert_eq!(t, 1);
        assert_eq!(off, 0);
        assert_eq!(k.len(), 8);
    }
}

//! Per-layer PLE split — Metal single-buffer hard limit is 4 GiB.
//!
//! HF packs `embed_tokens_per_layer` as
//! `[vocab_size_per_layer_input, num_hidden_layers * hidden_size_per_layer_input]`.
//! E4B bf16 packed ≈ 5.6 GiB → must split into per-layer Hot banks at load.

use crate::config::Gemma4TextConfig;
use crate::diag;
use crate::error::{Error, Result};
use crate::quant::{quantize_affine_bf16_bits, QuantMatrix, QuantScheme};

/// Apple Metal practical single-buffer ceiling (bytes).
pub const METAL_MAX_BUFFER_BYTES: usize = 4 * 1024 * 1024 * 1024;

/// One layer's PLE embedding table: `[vocab, ple_dim]`.
#[derive(Clone, Debug)]
pub struct PleLayerBank {
    pub layer: usize,
    pub vocab: usize,
    pub dim: usize,
    pub matrix: QuantMatrix,
}

/// Host-side PLE after mandatory per-layer split.
#[derive(Clone, Debug)]
pub struct PleBanks {
    pub layers: Vec<PleLayerBank>,
}

impl PleBanks {
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn total_hot_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.matrix.nbytes_hot()).sum()
    }

    /// Assert every layer bank is under the Metal limit.
    pub fn validate_metal_limit(&self) -> Result<()> {
        for layer in &self.layers {
            let n = layer.matrix.nbytes_hot();
            if n > METAL_MAX_BUFFER_BYTES {
                return Err(Error::Ple(format!(
                    "PLE layer {} bank {n} bytes exceeds Metal {METAL_MAX_BUFFER_BYTES} limit",
                    layer.layer
                )));
            }
        }
        Ok(())
    }
}

/// Bytes for a packed PLE table at `elem_bytes` (2 = bf16, 4 = f32).
pub fn packed_ple_nbytes(
    vocab: usize,
    num_layers: usize,
    ple_dim: usize,
    elem_bytes: usize,
) -> usize {
    vocab
        .saturating_mul(num_layers)
        .saturating_mul(ple_dim)
        .saturating_mul(elem_bytes)
}

/// Bytes for one layer slice.
pub fn layer_ple_nbytes(vocab: usize, ple_dim: usize, elem_bytes: usize) -> usize {
    vocab.saturating_mul(ple_dim).saturating_mul(elem_bytes)
}

/// Whether packed PLE must be split for Metal.
pub fn must_split_packed_ple(cfg: &Gemma4TextConfig, elem_bytes: usize) -> bool {
    if !cfg.has_ple() {
        return false;
    }
    packed_ple_nbytes(
        cfg.vocab_size_per_layer_input,
        cfg.num_hidden_layers,
        cfg.hidden_size_per_layer_input,
        elem_bytes,
    ) > METAL_MAX_BUFFER_BYTES
}

/// Split packed row-major `[vocab, L * dim]` bf16 bits into per-layer `[vocab, dim]`.
///
/// Layout: for token `t`, packed columns `[layer0 | layer1 | … | layerL-1]` each of width `dim`.
pub fn split_packed_ple_bf16(
    packed: &[u16],
    vocab: usize,
    num_layers: usize,
    ple_dim: usize,
) -> Result<Vec<Vec<u16>>> {
    let expect = vocab * num_layers * ple_dim;
    if packed.len() != expect {
        let e = Error::Ple(format!(
            "packed PLE len {} != vocab*L*dim {}",
            packed.len(),
            expect
        ));
        diag::err("ple", "split_packed_ple_bf16", &e);
        return Err(e);
    }
    diag::log(
        "ple",
        format_args!(
            "split interleaved vocab={vocab} layers={num_layers} dim={ple_dim} packed_elems={}",
            packed.len()
        ),
    );
    let mut out = vec![vec![0u16; vocab * ple_dim]; num_layers];
    for t in 0..vocab {
        let row_base = t * num_layers * ple_dim;
        for layer in 0..num_layers {
            let src = row_base + layer * ple_dim;
            let dst = t * ple_dim;
            out[layer][dst..dst + ple_dim]
                .copy_from_slice(&packed[src..src + ple_dim]);
        }
    }
    Ok(out)
}

/// Alternate layout: packed as `[vocab * L, dim]` with layer-major blocks
/// `layer * vocab * dim ..`. Used by some exporters — detect via `layout`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlePackLayout {
    /// `[vocab, L*dim]` — HF / Transformers default.
    InterleavedPerToken,
    /// Contiguous `[layer][vocab, dim]` blocks.
    ContiguousPerLayer,
}

pub fn split_packed_ple_bf16_layout(
    packed: &[u16],
    vocab: usize,
    num_layers: usize,
    ple_dim: usize,
    layout: PlePackLayout,
) -> Result<Vec<Vec<u16>>> {
    match layout {
        PlePackLayout::InterleavedPerToken => {
            split_packed_ple_bf16(packed, vocab, num_layers, ple_dim)
        }
        PlePackLayout::ContiguousPerLayer => {
            let expect = vocab * num_layers * ple_dim;
            if packed.len() != expect {
                return Err(Error::Ple(format!(
                    "packed PLE len {} != expected {}",
                    packed.len(),
                    expect
                )));
            }
            let mut out = Vec::with_capacity(num_layers);
            for layer in 0..num_layers {
                let start = layer * vocab * ple_dim;
                let end = start + vocab * ple_dim;
                out.push(packed[start..end].to_vec());
            }
            Ok(out)
        }
    }
}

/// Build quantized per-layer PLE banks from packed bf16.
pub fn build_ple_banks_from_packed_bf16(
    cfg: &Gemma4TextConfig,
    packed: &[u16],
    layout: PlePackLayout,
    scheme: QuantScheme,
) -> Result<PleBanks> {
    if !cfg.has_ple() {
        return Err(Error::Ple(
            "config has no PLE (hidden_size_per_layer_input == 0)".into(),
        ));
    }
    let vocab = cfg.vocab_size_per_layer_input;
    let num_layers = cfg.num_hidden_layers;
    let dim = cfg.hidden_size_per_layer_input;
    let packed_bf16 = packed_ple_nbytes(vocab, num_layers, dim, 2);
    diag::log(
        "ple",
        format_args!(
            "build_ple_banks layout={layout:?} scheme={scheme:?} vocab={vocab} L={num_layers} dim={dim} \
             packed_bf16≈{} must_split={}",
            diag::fmt_bytes(packed_bf16 as u64),
            must_split_packed_ple(cfg, 2)
        ),
    );

    // Always split — even if packed would fit — so load path is uniform and future-proof.
    let _ = must_split_packed_ple(cfg, 2);

    let layers_bf16 = split_packed_ple_bf16_layout(packed, vocab, num_layers, dim, layout)?;
    let mut layers = Vec::with_capacity(num_layers);
    for (layer, bits) in layers_bf16.into_iter().enumerate() {
        // `ple_lookup` Metal kernel reads bf16 tables — keep host banks as Bf16
        // regardless of weight quant scheme (Q4 weights stay Q4).
        let _ = scheme;
        let matrix = quantize_affine_bf16_bits(vocab, dim, &bits, QuantScheme::Bf16)?;
        let bank = PleLayerBank {
            layer,
            vocab,
            dim,
            matrix,
        };
        if bank.matrix.nbytes_hot() > METAL_MAX_BUFFER_BYTES {
            let e = Error::Ple(format!(
                "PLE layer {layer} still exceeds Metal 4GiB after split"
            ));
            diag::err("ple", "per-layer bank too large", &e);
            return Err(e);
        }
        if layer == 0 || layer + 1 == num_layers || layer % 8 == 0 {
            diag::log(
                "ple",
                format_args!(
                    "PLE layer {layer} hot={}",
                    diag::fmt_bytes(bank.matrix.nbytes_hot() as u64)
                ),
            );
        }
        layers.push(bank);
    }
    let banks = PleBanks { layers };
    banks.validate_metal_limit()?;
    diag::log(
        "ple",
        format_args!(
            "PLE banks ready: {} layers total_hot={}",
            banks.num_layers(),
            diag::fmt_bytes(banks.total_hot_bytes() as u64)
        ),
    );
    Ok(banks)
}

/// Synthetic empty PLE banks sized from config (for scratch planning without weights).
pub fn plan_ple_layer_bytes(cfg: &Gemma4TextConfig, scheme: QuantScheme) -> Result<Vec<usize>> {
    if !cfg.has_ple() {
        return Ok(Vec::new());
    }
    let vocab = cfg.vocab_size_per_layer_input;
    let dim = cfg.hidden_size_per_layer_input;
    let mut out = Vec::with_capacity(cfg.num_hidden_layers);
    for _ in 0..cfg.num_hidden_layers {
        let nbytes = match scheme {
            QuantScheme::Bf16 => layer_ple_nbytes(vocab, dim, 2),
            QuantScheme::Q4 { .. } | QuantScheme::Q4Mlx { .. } => {
                // packed nibbles + scales/zeros approx
                let packed = (vocab * dim + 1) / 2;
                let groups = vocab * (dim / scheme.group_size().unwrap_or(32));
                packed + groups * 8
            }
            QuantScheme::Q8 { .. } => {
                let packed = vocab * dim;
                let groups = vocab * (dim / scheme.group_size().unwrap_or(32));
                packed + groups * 8
            }
        };
        if nbytes > METAL_MAX_BUFFER_BYTES {
            return Err(Error::Ple(format!(
                "planned PLE layer {nbytes} exceeds Metal limit"
            )));
        }
        out.push(nbytes);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Gemma4TextConfig;

    #[test]
    fn e4b_packed_exceeds_4gb_bf16() {
        let cfg = Gemma4TextConfig::e4b_preset();
        let n = packed_ple_nbytes(
            cfg.vocab_size_per_layer_input,
            cfg.num_hidden_layers,
            cfg.hidden_size_per_layer_input,
            2,
        );
        assert!(n > METAL_MAX_BUFFER_BYTES, "packed={n}");
        assert!(must_split_packed_ple(&cfg, 2));
        let per = layer_ple_nbytes(
            cfg.vocab_size_per_layer_input,
            cfg.hidden_size_per_layer_input,
            2,
        );
        assert!(per < METAL_MAX_BUFFER_BYTES);
    }

    #[test]
    fn split_roundtrip_tiny() {
        let vocab = 3;
        let layers = 2;
        let dim = 4;
        // Build interleaved [vocab, L*dim]
        let mut packed = vec![0u16; vocab * layers * dim];
        for t in 0..vocab {
            for l in 0..layers {
                for d in 0..dim {
                    packed[t * layers * dim + l * dim + d] = (t * 100 + l * 10 + d) as u16;
                }
            }
        }
        let split = split_packed_ple_bf16(&packed, vocab, layers, dim).unwrap();
        assert_eq!(split.len(), 2);
        assert_eq!(split[0][0], 0); // t=0,l=0,d=0
        assert_eq!(split[1][0], 10); // t=0,l=1,d=0
        assert_eq!(split[0][dim], 100); // t=1,l=0,d=0
    }

    #[test]
    fn build_banks_respects_limit() {
        let mut cfg = Gemma4TextConfig::e4b_preset();
        // Tiny synthetic to keep test fast
        cfg.vocab_size_per_layer_input = 32;
        cfg.num_hidden_layers = 4;
        cfg.hidden_size_per_layer_input = 32;
        cfg.layer_types = cfg.layer_types[..4].to_vec();
        let n = cfg.vocab_size_per_layer_input
            * cfg.num_hidden_layers
            * cfg.hidden_size_per_layer_input;
        let packed: Vec<u16> = (0..n as u16).collect();
        let banks = build_ple_banks_from_packed_bf16(
            &cfg,
            &packed,
            PlePackLayout::InterleavedPerToken,
            QuantScheme::q4_default(),
        )
        .unwrap();
        assert_eq!(banks.num_layers(), 4);
        banks.validate_metal_limit().unwrap();
    }

    #[test]
    fn b31_no_ple() {
        let cfg = Gemma4TextConfig::b31_preset();
        assert!(!must_split_packed_ple(&cfg, 2));
        assert!(plan_ple_layer_bytes(&cfg, QuantScheme::q4_default())
            .unwrap()
            .is_empty());
    }
}

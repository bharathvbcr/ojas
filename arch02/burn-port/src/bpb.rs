//! Bits-per-byte (BPB) evaluation, including the sliding-window protocol.
//!
//!   bpb = (mean_nll_nats / ln 2) * (total_tokens / total_bytes)
//!
//! Byte count per TARGET token (python parity):
//!   bytes  = base_bytes[target]
//!   bytes += 1 if has_leading_space[target] && !is_boundary[prev]
//!
//! The three LUTs come from the SentencePiece model; export them once with
//! scripts/export_token_bytes.py → token_bytes.json.

use std::path::Path;

use burn::prelude::*;
use serde::Deserialize;

use crate::data::bigram_hash;
use crate::model::Gpt;

#[derive(Debug, Clone, Deserialize)]
pub struct TokenByteLut {
    pub base_bytes: Vec<u32>,
    pub has_leading_space: Vec<bool>,
    pub is_boundary_token: Vec<bool>,
}

impl TokenByteLut {
    pub fn load(path: &Path) -> Result<Self, String> {
        let s = std::fs::read_to_string(path).map_err(|e| format!("{path:?}: {e}"))?;
        serde_json::from_str(&s).map_err(|e| format!("{path:?}: {e}"))
    }

    /// Bytes attributed to predicting `target` given previous token `prev`.
    pub fn token_bytes(&self, target: u16, prev: u16) -> u64 {
        let mut b = self.base_bytes[target as usize] as u64;
        if self.has_leading_space[target as usize] && !self.is_boundary_token[prev as usize] {
            b += 1;
        }
        b
    }
}

pub struct BpbAccumulator {
    pub nll_sum: f64,   // sum of per-token nll (nats)
    pub token_count: u64,
    pub byte_count: u64,
}

impl BpbAccumulator {
    pub fn new() -> Self {
        Self { nll_sum: 0.0, token_count: 0, byte_count: 0 }
    }
    pub fn bpb(&self) -> f64 {
        let mean_nll = self.nll_sum / self.token_count as f64;
        let bits_per_token = mean_nll / std::f64::consts::LN_2;
        bits_per_token * (self.token_count as f64 / self.byte_count as f64)
    }
}

impl Default for BpbAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone)]
struct Window {
    start: usize,
    wlen: usize,
    score_from: usize, // first scored index within the window
}

/// Sliding-window eval over a flat validation token stream.
///
/// Windows start every `stride` tokens; each window is up to `seq_len` long;
/// only the last `stride` tokens of a window are scored (the first window
/// scores everything), so every target is scored exactly once with maximal
/// left context. Zero right-padding is excluded from scoring.
///
/// Mac-optimization: the first draft ran one batch-1 forward per window with a
/// host readback each (~1000 sequential round-trips per eval). This batches
/// `eval_batch` equal-length windows into a single `[B, T]` forward with one
/// readback per batch. Windows of identical length are grouped (all but the
/// possibly-shorter tail window are full `seq_len`).
pub fn eval_sliding<B: Backend>(
    model: &Gpt<B>,
    val_tokens: &[u16],
    seq_len: usize,
    stride: usize,
    lut: &TokenByteLut,
    bigram_vocab_size: usize,
    eval_batch: usize,
    device: &B::Device,
) -> f64 {
    let total = val_tokens.len() - 1; // number of (input → target) pairs
    let mut acc = BpbAccumulator::new();
    let batch_cap = eval_batch.max(1);

    // Enumerate windows.
    let mut windows: Vec<Window> = Vec::new();
    let mut start = 0usize;
    while start < total {
        let wlen = (start + seq_len).min(total) - start;
        let score_from = if start == 0 { 0 } else { wlen.saturating_sub(stride) };
        windows.push(Window { start, wlen, score_from });
        start += stride;
    }

    // Group consecutive same-length windows into batches.
    let mut i = 0usize;
    while i < windows.len() {
        let wlen = windows[i].wlen;
        let mut batch: Vec<Window> = Vec::with_capacity(batch_cap);
        while batch.len() < batch_cap && i < windows.len() && windows[i].wlen == wlen {
            batch.push(windows[i].clone());
            i += 1;
        }
        let bsz = batch.len();

        let mut xs = Vec::with_capacity(bsz * wlen);
        let mut ys = Vec::with_capacity(bsz * wlen);
        let mut bgs = Vec::with_capacity(bsz * wlen);
        for w in &batch {
            let inputs = &val_tokens[w.start..w.start + wlen];
            let targets = &val_tokens[w.start + 1..w.start + 1 + wlen];
            bgs.extend(bigram_hash(inputs, bigram_vocab_size));
            xs.extend(inputs.iter().map(|&v| v as i32));
            ys.extend(targets.iter().map(|&v| v as i32));
        }
        let shape = [bsz, wlen];
        let x = Tensor::<B, 1, Int>::from_ints(xs.as_slice(), device).reshape(shape);
        let bgt = Tensor::<B, 1, Int>::from_ints(bgs.as_slice(), device).reshape(shape);
        let y = Tensor::<B, 1, Int>::from_ints(ys.as_slice(), device).reshape(shape);

        let nll = model.nll_per_token(x, bgt, y); // [bsz, wlen]
        let nll_data = nll.into_data();
        let nll_v = nll_data.as_slice::<f32>().unwrap();

        for (bi, w) in batch.iter().enumerate() {
            let targets = &val_tokens[w.start + 1..w.start + 1 + wlen];
            for j in w.score_from..wlen {
                acc.nll_sum += nll_v[bi * wlen + j] as f64;
                acc.token_count += 1;
                let prev = if w.start + j == 0 {
                    val_tokens[0]
                } else {
                    val_tokens[w.start + j]
                };
                acc.byte_count += lut.token_bytes(targets[j], prev);
            }
        }
    }
    acc.bpb()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lut3() -> TokenByteLut {
        TokenByteLut {
            // tok 0: boundary/control (0 bytes), tok 1: "▁the" (3 bytes + space),
            // tok 2: "x" (1 byte, no space)
            base_bytes: vec![0, 3, 1],
            has_leading_space: vec![false, true, false],
            is_boundary_token: vec![true, false, false],
        }
    }

    #[test]
    fn byte_attribution() {
        let l = lut3();
        // "▁the" after boundary token: no +1
        assert_eq!(l.token_bytes(1, 0), 3);
        // "▁the" after a normal token: +1 for the space
        assert_eq!(l.token_bytes(1, 2), 4);
        // "x" never gets +1
        assert_eq!(l.token_bytes(2, 1), 1);
    }

    #[test]
    fn bpb_formula() {
        let mut a = BpbAccumulator::new();
        // 100 tokens, mean nll = ln(2) → 1 bit/token; 200 bytes → 0.5 bpb
        a.nll_sum = 100.0 * std::f64::consts::LN_2;
        a.token_count = 100;
        a.byte_count = 200;
        assert!((a.bpb() - 0.5).abs() < 1e-12);
    }
}

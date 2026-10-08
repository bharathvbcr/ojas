//! Bits-per-byte evaluation (sliding-window), ported from burn-port without Burn.
//!
//! **Weight source:** mid-run BPB (`--eval-every`) uses **live** training weights;
//! the final reported BPB copies **EMA** shadows into live weights first
//! (`copy_ema_into_weights`). Do not compare mid-run live numbers to the final EMA figure.

use std::path::Path;

use crate::model_fwd::forward_infer_f32;
use crate::runtime::GpuRuntime;
use crate::weights::Weights;
use objc2_metal::MTLBuffer;
use std::sync::Arc;

#[derive(Debug, Clone, serde::Deserialize)]
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

    pub fn token_bytes(&self, target: u16, prev: u16) -> u64 {
        let mut b = self.base_bytes[target as usize] as u64;
        if self.has_leading_space[target as usize] && !self.is_boundary_token[prev as usize] {
            b += 1;
        }
        b
    }
}

pub struct BpbAccumulator {
    pub nll_sum: f64,
    pub token_count: u64,
    pub byte_count: u64,
}

impl BpbAccumulator {
    pub fn new() -> Self {
        Self {
            nll_sum: 0.0,
            token_count: 0,
            byte_count: 0,
        }
    }
    pub fn bpb(&self) -> f64 {
        let mean_nll = self.nll_sum / self.token_count as f64;
        let bits_per_token = mean_nll / std::f64::consts::LN_2;
        bits_per_token * (self.token_count as f64 / self.byte_count as f64)
    }
}

#[derive(Clone)]
struct Window {
    start: usize,
    wlen: usize,
    score_from: usize,
}

/// Sliding-window BPB via no-tape inference forward + a single per-row CE pass.
///
/// Short tail windows (`wlen < seq_len`) are **right-padded** to `cfg.seq_len` so
/// fixed rope / kernel shapes still apply; only real positions `score_from..wlen`
/// are scored (padding excluded), matching burn-port coverage of the val tail.
pub fn eval_sliding(
    rt: &Arc<GpuRuntime>,
    w: &Weights,
    val_tokens: &[u16],
    seq_len: usize,
    stride: usize,
    lut: &TokenByteLut,
    bigram_vocab_size: usize,
    eval_batch: usize,
    val_cap: usize,
) -> Result<f64, String> {
    let mut tokens = val_tokens;
    if val_cap > 0 && tokens.len() > val_cap {
        tokens = &tokens[..val_cap];
    }
    let total = tokens.len().saturating_sub(1);
    if total == 0 {
        return Err("val set empty".into());
    }
    let mut acc = BpbAccumulator::new();
    let batch_cap = eval_batch.max(1);
    let cfg_b = w.cfg.batch;
    let cfg_t = w.cfg.seq_len;
    if seq_len != cfg_t {
        return Err(format!(
            "eval seq_len {seq_len} != model cfg.seq_len {cfg_t} (rope buffers)"
        ));
    }
    let _ = bigram_vocab_size; // stem recomputes bigram from ids

    let mut windows: Vec<Window> = Vec::new();
    let mut start = 0usize;
    while start < total {
        let wlen = (start + seq_len).min(total) - start;
        let score_from = if start == 0 {
            0
        } else {
            wlen.saturating_sub(stride)
        };
        windows.push(Window {
            start,
            wlen,
            score_from,
        });
        start += stride;
    }

    let mut i = 0usize;
    while i < windows.len() {
        let wlen = windows[i].wlen;
        let mut batch: Vec<Window> = Vec::with_capacity(batch_cap);
        while batch.len() < batch_cap && i < windows.len() && windows[i].wlen == wlen {
            batch.push(windows[i].clone());
            i += 1;
        }
        let bsz = batch.len();

        // Always feed [cfg_b, cfg_t]: right-pad short windows; pad unused batch slots.
        let mut xs = vec![0i32; cfg_b * cfg_t];
        let mut ys = vec![0i32; cfg_b * cfg_t];
        for (bi, win) in batch.iter().enumerate() {
            if bi >= cfg_b {
                break;
            }
            let inputs = &tokens[win.start..win.start + win.wlen];
            let targets = &tokens[win.start + 1..win.start + 1 + win.wlen];
            for t in 0..win.wlen {
                xs[bi * cfg_t + t] = inputs[t] as i32;
                ys[bi * cfg_t + t] = targets[t] as i32;
            }
            // positions win.wlen..cfg_t remain 0 (right-pad); not scored below
        }
        if bsz < cfg_b && bsz > 0 {
            let last = &batch[bsz - 1];
            let inputs = &tokens[last.start..last.start + last.wlen];
            let targets = &tokens[last.start + 1..last.start + 1 + last.wlen];
            for bi in bsz..cfg_b {
                for t in 0..last.wlen {
                    xs[bi * cfg_t + t] = inputs[t] as i32;
                    ys[bi * cfg_t + t] = targets[t] as i32;
                }
            }
        }

        // No-tape infer forward (no training activation copies / no mean-CE sync).
        let logits = forward_infer_f32(rt, w, &xs)?;
        // Single CE pass: per-row NLL only (no separate ce_mean).
        let row_nll = row_ce_host(rt, &logits, &ys, cfg_b, cfg_t, w.cfg.vocab_size)?;

        for (bi, win) in batch.iter().enumerate() {
            if bi >= cfg_b {
                break;
            }
            let targets = &tokens[win.start + 1..win.start + 1 + win.wlen];
            for j in win.score_from..win.wlen {
                acc.nll_sum += row_nll[bi * cfg_t + j] as f64;
                acc.token_count += 1;
                let prev = if win.start + j == 0 {
                    tokens[0]
                } else {
                    tokens[win.start + j]
                };
                acc.byte_count += lut.token_bytes(targets[j], prev);
            }
        }
    }

    if acc.token_count == 0 {
        return Err("no tokens scored in sliding eval".into());
    }
    Ok(acc.bpb())
}

fn row_ce_host(
    rt: &Arc<GpuRuntime>,
    logits: &crate::tensor::Tensor,
    targets: &[i32],
    b: usize,
    t: usize,
    v: usize,
) -> Result<Vec<f32>, String> {
    use crate::dispatch::{dispatch_1d, set_tensor, set_u32};
    let rows = b * t;
    let tgts = {
        let tensor = rt.alloc_tensor_f32(&[rows])?;
        // store i32 bit patterns
        let ptr = tensor.buffer.metal().contents().as_ptr() as *mut i32;
        unsafe {
            std::ptr::copy_nonoverlapping(targets.as_ptr(), ptr, rows);
        }
        tensor
    };
    let flat = crate::tensor::Tensor::from_buffer(
        rt,
        logits.buffer.clone(),
        &[rows, v],
        logits.dtype,
        logits.byte_offset(),
    )?;
    let row_loss = rt.alloc_tensor_f32(&[rows])?;
    let p = rt.pipeline("ce_row_f32")?;
    dispatch_1d(rt, &p, rows, |bnd| {
        set_tensor(bnd, &flat, 0);
        set_tensor(bnd, &tgts, 1);
        set_tensor(bnd, &row_loss, 2);
        set_u32(bnd, rows as u32, 3);
        set_u32(bnd, v as u32, 4);
    })?;
    rt.synchronize()?;
    Ok(row_loss.buffer.read_f32())
}

//! Sequential shard loader — exact port of the python pipeline.
//!
//! Shard format: 256×int32 LE header:
//!   header[0] = 20240520 (magic), header[1] = 1 (version), header[2] = num_tokens
//! Payload: num_tokens × uint16 LE token ids at byte offset 1024.
//! File size must equal 1024 + 2*num_tokens.
//!
//! Sampling is purely sequential over sorted shards with wrap-around:
//! take span of (batch_tokens + 1); x = span[:-1], y = span[1:], reshaped
//! to [num_seqs, seq_len]. No shuffling.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{sync_channel, Receiver};
use std::thread;

pub const SHARD_MAGIC: i32 = 20240520;
pub const SHARD_VERSION: i32 = 1;

pub fn load_shard(path: &Path) -> Result<Vec<u16>, String> {
    let bytes = fs::read(path).map_err(|e| format!("{path:?}: {e}"))?;
    if bytes.len() < 1024 {
        return Err(format!("{path:?}: too small for header"));
    }
    let rd_i32 = |i: usize| i32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
    let magic = rd_i32(0);
    let version = rd_i32(1);
    let num_tokens = rd_i32(2) as usize;
    if magic != SHARD_MAGIC {
        return Err(format!("{path:?}: bad magic {magic}"));
    }
    if version != SHARD_VERSION {
        return Err(format!("{path:?}: bad version {version}"));
    }
    if bytes.len() != 1024 + 2 * num_tokens {
        return Err(format!(
            "{path:?}: size mismatch (expected {}, got {})",
            1024 + 2 * num_tokens,
            bytes.len()
        ));
    }
    let mut tokens = Vec::with_capacity(num_tokens);
    for i in 0..num_tokens {
        let o = 1024 + 2 * i;
        tokens.push(u16::from_le_bytes([bytes[o], bytes[o + 1]]));
    }
    Ok(tokens)
}

/// Bigram hash — must match model/gpt.rs docs:
///   idx[0] = mod_ (= bigram_vocab_size - 1 = 2047)
///   idx[t] = (36313*tok[t] ^ 27191*tok[t-1]) mod 2047   (wrapping i32)
pub fn bigram_hash(tokens: &[u16], bigram_vocab_size: usize) -> Vec<i32> {
    let m = (bigram_vocab_size - 1) as i32; // 2047
    let mut out = Vec::with_capacity(tokens.len());
    for (t, &tok) in tokens.iter().enumerate() {
        if t == 0 {
            out.push(m);
        } else {
            let a = (36313i32).wrapping_mul(tok as i32);
            let b = (27191i32).wrapping_mul(tokens[t - 1] as i32);
            out.push((a ^ b).rem_euclid(m));
        }
    }
    out
}

/// Sequential stream over shards with wrap-around.
pub struct ShardStream {
    shards: Vec<PathBuf>,
    shard_idx: usize,
    pos: usize,
    current: Vec<u16>,
}

impl ShardStream {
    pub fn new(dir: &Path, pattern_contains: &str) -> Result<Self, String> {
        let mut shards: Vec<PathBuf> = fs::read_dir(dir)
            .map_err(|e| format!("{dir:?}: {e}"))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.file_name()
                    .map(|n| n.to_string_lossy().contains(pattern_contains))
                    .unwrap_or(false)
            })
            .collect();
        shards.sort();
        if shards.is_empty() {
            return Err(format!("no shards matching '{pattern_contains}' in {dir:?}"));
        }
        let current = load_shard(&shards[0])?;
        Ok(Self { shards, shard_idx: 0, pos: 0, current })
    }

    /// Next contiguous span of `n` tokens (crosses shard boundaries, wraps).
    pub fn next_span(&mut self, n: usize) -> Vec<u16> {
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            let avail = self.current.len() - self.pos;
            let take = avail.min(n - out.len());
            out.extend_from_slice(&self.current[self.pos..self.pos + take]);
            self.pos += take;
            if self.pos == self.current.len() {
                self.shard_idx = (self.shard_idx + 1) % self.shards.len();
                self.current = load_shard(&self.shards[self.shard_idx])
                    .expect("shard became unreadable mid-run");
                self.pos = 0;
            }
        }
        out
    }

    /// One micro-batch: (x, y, bigram_idx) each of len num_seqs*seq_len,
    /// row-major [num_seqs, seq_len] — upload these to Int tensors.
    pub fn next_batch(
        &mut self,
        num_seqs: usize,
        seq_len: usize,
        bigram_vocab_size: usize,
    ) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
        let span = self.next_span(num_seqs * seq_len + 1);
        let x: Vec<i32> = span[..span.len() - 1].iter().map(|&t| t as i32).collect();
        let y: Vec<i32> = span[1..].iter().map(|&t| t as i32).collect();
        // bigram hash computed per ROW (each sequence independently, position
        // 0 of every row gets the boundary index) — matches python, which
        // hashes after reshape to [num_seqs, seq_len].
        let mut bg = Vec::with_capacity(x.len());
        for r in 0..num_seqs {
            let row = &span[r * seq_len..(r + 1) * seq_len];
            bg.extend(bigram_hash(row, bigram_vocab_size));
        }
        (x, y, bg)
    }
}

/// One micro-batch of host-side integer data ready to upload.
pub struct HostBatch {
    pub x: Vec<i32>,
    pub y: Vec<i32>,
    pub bigram: Vec<i32>,
}

/// Background prefetching wrapper around `ShardStream`.
///
/// Shard IO + span assembly + the bigram XOR hash are pure CPU work that the
/// first draft did inline, on the critical path, between GPU micro-steps. This
/// moves that work onto a dedicated thread feeding a bounded channel, so the
/// next batch is (usually) ready by the time the GPU asks for it — overlapping
/// host prep with device compute.
pub struct PrefetchLoader {
    rx: Receiver<HostBatch>,
    _handle: thread::JoinHandle<()>,
}

impl PrefetchLoader {
    pub fn new(
        dir: &Path,
        pattern_contains: &str,
        micro_batch: usize,
        seq_len: usize,
        bigram_vocab_size: usize,
        depth: usize,
    ) -> Result<Self, String> {
        let mut stream = ShardStream::new(dir, pattern_contains)?;
        let (tx, rx) = sync_channel::<HostBatch>(depth.max(1));
        let handle = thread::spawn(move || loop {
            let (x, y, bigram) = stream.next_batch(micro_batch, seq_len, bigram_vocab_size);
            // Stop when the consumer is dropped.
            if tx.send(HostBatch { x, y, bigram }).is_err() {
                break;
            }
        });
        Ok(Self {
            rx,
            _handle: handle,
        })
    }

    /// Blocks until the next prefetched micro-batch is available.
    pub fn next(&self) -> HostBatch {
        self.rx
            .recv()
            .expect("prefetch worker terminated unexpectedly")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bigram_hash_matches_python_semantics() {
        // python: out[...,0]=2047; out[...,1:]=(36313*t[1:] ^ 27191*t[:-1]) % 2047
        let toks = [5u16, 7, 1023];
        let h = bigram_hash(&toks, 2048);
        assert_eq!(h[0], 2047);
        let e1 = ((36313i32 * 7) ^ (27191i32 * 5)).rem_euclid(2047);
        let e2 = ((36313i32 * 1023) ^ (27191i32 * 7)).rem_euclid(2047);
        assert_eq!(h[1], e1);
        assert_eq!(h[2], e2);
        for v in h {
            assert!((0..2048).contains(&v));
        }
    }

    #[test]
    fn shard_roundtrip() {
        let dir = std::env::temp_dir().join("arch02_shard_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_train_000.bin");
        let tokens: Vec<u16> = (0..1000u16).collect();
        let mut bytes = vec![0u8; 1024];
        bytes[0..4].copy_from_slice(&SHARD_MAGIC.to_le_bytes());
        bytes[4..8].copy_from_slice(&SHARD_VERSION.to_le_bytes());
        bytes[8..12].copy_from_slice(&(tokens.len() as i32).to_le_bytes());
        for t in &tokens {
            bytes.extend_from_slice(&t.to_le_bytes());
        }
        std::fs::write(&path, &bytes).unwrap();

        let loaded = load_shard(&path).unwrap();
        assert_eq!(loaded, tokens);

        let mut stream = ShardStream::new(&dir, "train").unwrap();
        let span = stream.next_span(1500); // forces wrap-around
        assert_eq!(span[0], 0);
        assert_eq!(span[999], 999);
        assert_eq!(span[1000], 0); // wrapped
        std::fs::remove_dir_all(&dir).ok();
    }
}

//! Sequential FineWeb shard loader — port of burn-port `data.rs` (leave burn-port untouched).

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

/// A shard token is a `u16`. Against a vocab of 1024 anything at or above that
/// vocab must be refused, not cast through and indexed later.
pub fn require_u16_vocab(tokens: &[u16], vocab_size: usize) -> Result<(), String> {
    for (i, &tok) in tokens.iter().enumerate() {
        if tok as usize >= vocab_size {
            return Err(format!(
                "token[{i}] = {tok} is outside 0..{vocab_size}"
            ));
        }
    }
    Ok(())
}

pub fn bigram_hash(tokens: &[u16], bigram_vocab_size: usize) -> Vec<i32> {
    let m = (bigram_vocab_size - 1) as i32;
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
            return Err(format!(
                "no shards matching '{pattern_contains}' in {dir:?}"
            ));
        }
        let current = load_shard(&shards[0])?;
        Ok(Self {
            shards,
            shard_idx: 0,
            pos: 0,
            current,
        })
    }

    /// Discard `n` tokens (wraps shards). Used for seed-derived FineWeb offsets.
    pub fn skip_tokens(&mut self, mut n: usize) {
        while n > 0 {
            let avail = self.current.len() - self.pos;
            if avail == 0 {
                self.shard_idx = (self.shard_idx + 1) % self.shards.len();
                self.current = load_shard(&self.shards[self.shard_idx])
                    .expect("shard became unreadable mid-skip");
                self.pos = 0;
                continue;
            }
            let take = avail.min(n);
            self.pos += take;
            n -= take;
            if self.pos == self.current.len() {
                self.shard_idx = (self.shard_idx + 1) % self.shards.len();
                self.current = load_shard(&self.shards[self.shard_idx])
                    .expect("shard became unreadable mid-skip");
                self.pos = 0;
            }
        }
    }

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

pub fn next_batch(
        &mut self,
        num_seqs: usize,
        seq_len: usize,
        _bigram_vocab_size: usize,
        vocab_size: usize,
    ) -> Result<(Vec<i32>, Vec<i32>), String> {
        let span = self.next_span(num_seqs * seq_len + 1);
        require_u16_vocab(&span, vocab_size)?;
        let x: Vec<i32> = span[..span.len() - 1].iter().map(|&t| t as i32).collect();
        let y: Vec<i32> = span[1..].iter().map(|&t| t as i32).collect();
        Ok((x, y))
    }
}

pub struct HostBatch {
    pub x: Vec<i32>,
    pub y: Vec<i32>,
}

pub struct PrefetchLoader {
    rx: Receiver<Result<HostBatch, String>>,
    _handle: thread::JoinHandle<()>,
}

impl PrefetchLoader {
    pub fn new(
        dir: &Path,
        pattern_contains: &str,
        micro_batch: usize,
        seq_len: usize,
        bigram_vocab_size: usize,
        vocab_size: usize,
        depth: usize,
    ) -> Result<Self, String> {
        Self::new_with_seed(
            dir,
            pattern_contains,
            micro_batch,
            seq_len,
            bigram_vocab_size,
            vocab_size,
            depth,
            None,
        )
    }

    /// Like `new`, optionally skipping `token_skip` tokens before the first batch
    /// (seed-derived FineWeb cursor; see `init::fineweb_token_skip`).
    pub fn new_with_seed(
        dir: &Path,
        pattern_contains: &str,
        micro_batch: usize,
        seq_len: usize,
        bigram_vocab_size: usize,
        vocab_size: usize,
        depth: usize,
        token_skip: Option<usize>,
    ) -> Result<Self, String> {
        let mut stream = ShardStream::new(dir, pattern_contains)?;
        if let Some(skip) = token_skip {
            if skip > 0 {
                stream.skip_tokens(skip);
            }
        }
        let (tx, rx) = sync_channel(depth.max(1));
        let handle = thread::spawn(move || loop {
            let batch = match stream.next_batch(micro_batch, seq_len, bigram_vocab_size, vocab_size) {
                Ok((x, y)) => HostBatch { x, y },
                Err(error) => {
                    let _ = tx.send(Err(error));
                    break;
                }
            };
            if tx.send(Ok(batch)).is_err() {
                break;
            }
        });
        Ok(Self {
            rx,
            _handle: handle,
        })
    }

    pub fn next(&self) -> Result<HostBatch, String> {
        self.rx
            .recv()
            .map_err(|_| "prefetch worker terminated unexpectedly".to_string())?
    }
}

#[cfg(test)]
mod tests {
    use super::require_u16_vocab;

    #[test]
    fn shard_tokens_outside_a_1024_vocab_are_refused() {
        let err = require_u16_vocab(&[0, 1024], 1024).unwrap_err();
        assert!(err.contains("1024"), "{err}");
        require_u16_vocab(&[0, 1023], 1024).unwrap();
    }
}

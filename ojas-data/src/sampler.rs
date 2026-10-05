//! Seeded, epoch-based `(x, y)` batches over a [`TokenBin`].
//!
//! Windows do not overlap. Window `k` covers tokens `[k*T, k*T + T]`
//! (`T + 1` tokens: `x` is the first `T`, `y` the last `T`), for
//! `k < W = (len - 1) / T`. Each epoch visits every window exactly once,
//! in an order that is a pseudorandom permutation of `0..W` keyed by
//! `(seed, epoch)` through [`CounterRng`]. A batch takes the next `B`
//! windows of that stream and continues into the next epoch when one ends,
//! so the window sequence does not depend on `B`.
//!
//! The permutation is a 4-round Feistel network on the smallest even
//! number of bits covering `W`, cycle-walked back into `0..W`. It needs no
//! per-epoch table, so a cursor resumes in O(1) memory whatever `W` is.
//!
//! Resume uses [`DataCursor`] from the checkpoint schema: `shard` is the
//! epoch and `token_index` is the ordinal of the next window inside that
//! epoch (not a token offset). The seed, `T` and `B` are configuration, not
//! cursor state; resuming with a different seed or `T` is a different
//! stream.
//!
//! nanolab's `Batcher.batch` samples starts uniformly with replacement
//! (`torch.randint`), overlapping and with no epochs. This sampler is the
//! without-replacement, exactly-once alternative; it does not reproduce
//! torch's stream.

use crate::error::DataError;
use crate::rng::CounterRng;
use crate::tokens::TokenBin;
use ojas_core::DataCursor;

/// Feistel rounds. Four rounds of a keyed bijection give a pseudorandom
/// permutation (Luby-Rackoff); the round function is SplitMix64.
const ROUNDS: usize = 4;

/// Per-epoch key spacing, an odd 64-bit constant (the SplitMix64 gamma's
/// sibling from Steele et al.), so `(seed, epoch)` pairs do not collide by
/// simple addition.
const EPOCH_STEP: u64 = 0xD1B5_4A32_D192_ED03;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SamplerConfig {
    /// `T`, tokens per row of `x` and of `y`.
    pub seq_len: usize,
    /// `B`, rows per batch.
    pub batch: usize,
    pub seed: u64,
}

/// One batch, row-major `[batch, seq_len]`. `y[r][i] == x[r][i + 1]` for
/// `i < seq_len - 1`, and `y[r][seq_len - 1]` is the token after the window.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Batch {
    pub x: Vec<u32>,
    pub y: Vec<u32>,
    pub batch: usize,
    pub seq_len: usize,
}

#[derive(Debug)]
pub struct BatchSampler<'a> {
    bin: &'a TokenBin,
    cfg: SamplerConfig,
    windows: u64,
    /// Bits per Feistel half.
    half_bits: u32,
    epoch: u64,
    ordinal: u64,
}

impl<'a> BatchSampler<'a> {
    /// Start at epoch 0, window 0. Refuses `T == 0`, `B == 0`, and a bin
    /// with fewer than `T + 1` tokens.
    pub fn new(bin: &'a TokenBin, cfg: SamplerConfig) -> Result<Self, DataError> {
        Self::resume(bin, cfg, DataCursor::default())
    }

    /// Continue from `cursor`. `cursor.token_index` must be below
    /// [`Self::windows_per_epoch`].
    pub fn resume(
        bin: &'a TokenBin,
        cfg: SamplerConfig,
        cursor: DataCursor,
    ) -> Result<Self, DataError> {
        if cfg.seq_len == 0 || cfg.batch == 0 {
            return Err(DataError::new(format!(
                "sampler: seq_len {} and batch {} must be non-zero",
                cfg.seq_len, cfg.batch
            )));
        }
        let t = u64::try_from(cfg.seq_len)
            .map_err(|_| DataError::new("sampler: seq_len exceeds u64"))?;
        let need = t
            .checked_add(1)
            .ok_or_else(|| DataError::new("sampler: seq_len + 1 overflows"))?;
        if bin.len() < need {
            return Err(DataError::new(format!(
                "sampler: a window needs {need} tokens, the bin has {}",
                bin.len()
            )));
        }
        // Row buffers are allocated per batch; refuse sizes that cannot be.
        cfg.batch
            .checked_mul(cfg.seq_len)
            .and_then(|n| n.checked_add(cfg.seq_len))
            .ok_or_else(|| DataError::new("sampler: batch * seq_len overflows"))?;
        let windows = (bin.len() - 1) / t;
        if cursor.token_index >= windows {
            return Err(DataError::new(format!(
                "sampler: cursor window {} is past the {windows} windows of an epoch",
                cursor.token_index
            )));
        }
        let bits = u64::BITS - (windows - 1).leading_zeros();
        let half_bits = bits.div_ceil(2).max(1);
        Ok(Self {
            bin,
            cfg,
            windows,
            half_bits,
            epoch: cursor.shard,
            ordinal: cursor.token_index,
        })
    }

    /// `W`, the windows in one epoch.
    pub fn windows_per_epoch(&self) -> u64 {
        self.windows
    }

    /// Where the next batch starts. Store it in the checkpoint.
    pub fn cursor(&self) -> DataCursor {
        DataCursor {
            shard: self.epoch,
            token_index: self.ordinal,
        }
    }

    /// Token offset of the `ordinal`-th window of `epoch`.
    pub fn window_start(&self, epoch: u64, ordinal: u64) -> Result<u64, DataError> {
        if ordinal >= self.windows {
            return Err(DataError::new(format!(
                "sampler: window {ordinal} past {} per epoch",
                self.windows
            )));
        }
        let k = self.permute(epoch, ordinal);
        Ok(k * self.cfg.seq_len as u64)
    }

    /// The next `B` windows. On error the cursor does not move.
    pub fn next_batch(&mut self) -> Result<Batch, DataError> {
        let (t, b) = (self.cfg.seq_len, self.cfg.batch);
        let mut epoch = self.epoch;
        let mut ordinal = self.ordinal;
        let mut starts = Vec::new();
        starts
            .try_reserve_exact(b)
            .map_err(|_| DataError::new("sampler: allocation of the batch refused"))?;
        for _ in 0..b {
            starts.push(self.window_start(epoch, ordinal)?);
            ordinal += 1;
            if ordinal == self.windows {
                ordinal = 0;
                epoch = epoch
                    .checked_add(1)
                    .ok_or_else(|| DataError::new("sampler: epoch counter overflows"))?;
            }
        }
        let n = b * t;
        let mut x = Vec::new();
        let mut y = Vec::new();
        x.try_reserve_exact(n)
            .and_then(|_| y.try_reserve_exact(n))
            .map_err(|_| DataError::new(format!("sampler: allocation of 2 x {n} ids refused")))?;
        let mut row = vec![0u32; t + 1];
        for start in starts {
            self.bin.read_into_u32(start, &mut row)?;
            x.extend_from_slice(&row[..t]);
            y.extend_from_slice(&row[1..]);
        }
        self.epoch = epoch;
        self.ordinal = ordinal;
        Ok(Batch {
            x,
            y,
            batch: b,
            seq_len: t,
        })
    }

    /// A bijection on `0..windows`, keyed by `(seed, epoch)`.
    fn permute(&self, epoch: u64, ordinal: u64) -> u64 {
        let keys = self.round_keys(epoch);
        let mut v = ordinal;
        // Cycle-walk: the Feistel network permutes `0..2^(2h)` and
        // `2^(2h) < 4 * windows`, so this ends after a few steps on
        // average and always ends, because `ordinal` is on its own cycle.
        loop {
            v = self.feistel(v, &keys);
            if v < self.windows {
                return v;
            }
        }
    }

    fn round_keys(&self, epoch: u64) -> [u64; ROUNDS] {
        let mut rng = CounterRng::new(self.cfg.seed);
        rng.set_state(self.cfg.seed ^ epoch.wrapping_mul(EPOCH_STEP));
        let mut keys = [0u64; ROUNDS];
        for key in keys.iter_mut() {
            *key = rng.next_u64();
        }
        keys
    }

    fn feistel(&self, v: u64, keys: &[u64; ROUNDS]) -> u64 {
        let h = self.half_bits;
        let mask = (1u64 << h) - 1;
        let mut left = (v >> h) & mask;
        let mut right = v & mask;
        for &key in keys {
            let f = CounterRng::new(right ^ key).next_u64() & mask;
            let next = left ^ f;
            left = right;
            right = next;
        }
        (left << h) | right
    }
}

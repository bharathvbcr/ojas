//! Seeded, epoch-based `(x, y)` batches over a token bin.

use ojas_core::DataCursor;
use ojas_data::{
    BatchSampler, SamplerConfig, SamplerRngState, TokenBin, RNG_GENERATOR_FEISTEL4_SPLITMIX64,
    RNG_STATE_BYTES, RNG_STATE_VERSION,
};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static SEQ: AtomicU64 = AtomicU64::new(0);

struct Tmp(PathBuf);

impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// A headerless bin whose token `i` is `i % 65521`, so a token names its
/// own position.
fn bin(len: usize) -> (Tmp, TokenBin) {
    let path = std::env::temp_dir().join(format!(
        "ojas-sampler-{}-{}.bin",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let bytes: Vec<u8> = (0..len)
        .flat_map(|i| ((i % 65521) as u16).to_le_bytes())
        .collect();
    std::fs::write(&path, bytes).unwrap();
    let tokens = TokenBin::open_headerless(&path).unwrap();
    (Tmp(path), tokens)
}

fn cfg(seq_len: usize, batch: usize, seed: u64) -> SamplerConfig {
    SamplerConfig {
        seq_len,
        batch,
        seed,
    }
}

/// Every row: `y` is `x` shifted by one, and both are the file's own tokens.
fn starts(x: &[u32], y: &[u32], seq_len: usize) -> Vec<u64> {
    assert_eq!(x.len(), y.len());
    x.chunks_exact(seq_len)
        .zip(y.chunks_exact(seq_len))
        .map(|(xr, yr)| {
            let s = xr[0];
            for i in 0..seq_len {
                assert_eq!(xr[i], s + i as u32, "x is not a contiguous window");
                assert_eq!(yr[i], s + i as u32 + 1, "y is not x shifted by one");
            }
            u64::from(s)
        })
        .collect()
}

#[test]
fn every_window_start_is_visited_once_per_epoch_and_y_is_x_shifted() {
    // 103 tokens, T = 8: windows start at 0, 8, ..., 88 (12 windows; the
    // 13th would need token 104).
    let (_tmp, tokens) = bin(103);
    let t = 8;
    let mut s = BatchSampler::new(&tokens, cfg(t, 5, 7)).unwrap();
    assert_eq!(s.windows_per_epoch(), 12);
    let mut seen: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
    // 12 batches of 5 rows = 60 windows = exactly 5 epochs.
    for _ in 0..12 {
        let before = s.cursor();
        let b = s.next_batch().unwrap();
        assert_eq!((b.batch, b.seq_len), (5, t));
        let rows = starts(&b.x, &b.y, t);
        let mut epoch = before.shard;
        let mut ordinal = before.token_index;
        for start in rows {
            seen.entry(epoch).or_default().push(start);
            ordinal += 1;
            if ordinal == 12 {
                epoch += 1;
                ordinal = 0;
            }
        }
    }
    assert_eq!(
        s.cursor(),
        DataCursor {
            shard: 5,
            token_index: 0
        }
    );
    let all: Vec<u64> = (0..12).map(|k| k * t as u64).collect();
    let mut orders = Vec::new();
    for epoch in 0..5 {
        let got = &seen[&epoch];
        let mut sorted = got.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, all, "epoch {epoch} did not visit each window once");
        orders.push(got.clone());
    }
    assert!(orders.iter().any(|o| o != &all), "no epoch was shuffled");
    assert!(
        orders.windows(2).any(|w| w[0] != w[1]),
        "every epoch had the same order"
    );
}

#[test]
fn same_seed_same_batches_other_seed_other_order() {
    let (_tmp, tokens) = bin(4000);
    let run = |seed: u64| {
        let mut s = BatchSampler::new(&tokens, cfg(16, 4, seed)).unwrap();
        (0..20)
            .flat_map(|_| {
                let b = s.next_batch().unwrap();
                starts(&b.x, &b.y, 16)
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(run(42), run(42));
    assert_ne!(run(42), run(43));
}

/// Stop after `k` batches, resume from the cursor, and get the same rows
/// as an uninterrupted run, including batches that straddle an epoch.
#[test]
fn resume_from_any_cursor_reproduces_the_uninterrupted_stream() {
    let (_tmp, tokens) = bin(211);
    let c = cfg(6, 7, 0xC0FFEE);
    let mut whole = BatchSampler::new(&tokens, c.clone()).unwrap();
    let w = whole.windows_per_epoch();
    assert_eq!(w, 35);
    assert_ne!(w % 7, 1, "pick sizes so a batch straddles an epoch");
    let reference: Vec<(DataCursor, Vec<u32>, Vec<u32>)> = (0..16)
        .map(|_| {
            let at = whole.cursor();
            let b = whole.next_batch().unwrap();
            (at, b.x, b.y)
        })
        .collect();
    assert!(reference.iter().any(|(at, _, _)| at.shard > 0));
    for k in 0..reference.len() {
        let mut resumed = BatchSampler::resume(&tokens, c.clone(), reference[k].0).unwrap();
        for (j, (at, x, y)) in reference.iter().enumerate().skip(k) {
            assert_eq!(
                resumed.cursor(),
                *at,
                "cursor before batch {j} (resumed at {k})"
            );
            let b = resumed.next_batch().unwrap();
            assert_eq!(&b.x, x, "x of batch {j} (resumed at {k})");
            assert_eq!(&b.y, y, "y of batch {j} (resumed at {k})");
        }
    }
    // A mid-batch cursor is just as valid: resume at window 3 of epoch 1.
    let mid = DataCursor {
        shard: 1,
        token_index: 3,
    };
    let mut a = BatchSampler::resume(&tokens, c.clone(), mid).unwrap();
    let mut b = BatchSampler::new(&tokens, cfg(6, 1, 0xC0FFEE)).unwrap();
    for _ in 0..(w + 3) {
        b.next_batch().unwrap();
    }
    let first = a.next_batch().unwrap();
    let mut one_at_a_time = Vec::new();
    for _ in 0..7 {
        one_at_a_time.extend(b.next_batch().unwrap().x);
    }
    assert_eq!(first.x, one_at_a_time);
}

#[test]
fn shapes_that_cannot_produce_a_window_are_refused() {
    let (_tmp, tokens) = bin(9);
    assert!(
        BatchSampler::new(&tokens, cfg(8, 1, 0)).is_ok(),
        "T + 1 == len fits"
    );
    assert!(
        BatchSampler::new(&tokens, cfg(9, 1, 0)).is_err(),
        "T + 1 > len"
    );
    assert!(BatchSampler::new(&tokens, cfg(0, 1, 0)).is_err(), "T == 0");
    assert!(BatchSampler::new(&tokens, cfg(4, 0, 0)).is_err(), "B == 0");
    assert!(BatchSampler::new(&tokens, cfg(usize::MAX, 1, 0)).is_err());
    let (_tmp, empty) = bin(0);
    assert!(BatchSampler::new(&empty, cfg(1, 1, 0)).is_err());
    let (_tmp, tokens) = bin(100);
    let c = cfg(9, 2, 0);
    let w = BatchSampler::new(&tokens, c.clone())
        .unwrap()
        .windows_per_epoch();
    assert_eq!(w, 11);
    let past = DataCursor {
        shard: 0,
        token_index: w,
    };
    assert!(BatchSampler::resume(&tokens, c.clone(), past).is_err());
    let last = DataCursor {
        shard: u64::MAX,
        token_index: w - 1,
    };
    let mut s = BatchSampler::resume(&tokens, c, last).unwrap();
    assert!(
        s.next_batch().is_err(),
        "epoch counter overflow must not wrap"
    );
}

/// One window per epoch: every batch row is that window.
#[test]
fn a_single_window_repeats_every_epoch() {
    let (_tmp, tokens) = bin(5);
    let mut s = BatchSampler::new(&tokens, cfg(4, 3, 9)).unwrap();
    assert_eq!(s.windows_per_epoch(), 1);
    let b = s.next_batch().unwrap();
    assert_eq!(b.x, vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]);
    assert_eq!(b.y, vec![1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]);
    assert_eq!(
        s.cursor(),
        DataCursor {
            shard: 3,
            token_index: 0
        }
    );
}

/// Window starts for fixed `(seed, W = 100, epoch, ordinal)`, computed by the
/// independent Python oracle `target-robust/sampler_oracle.py`. A change to
/// the permutation that moves any of them must also change
/// `RNG_GENERATOR_FEISTEL4_SPLITMIX64`, or a checkpoint's `rng_state` would
/// name a stream the sampler no longer produces.
#[test]
fn permutation_is_pinned() {
    let (_tmp, tokens) = bin(1001);
    let cases: [(u64, u64, [u64; 8]); 6] = [
        (42, 0, [460, 450, 170, 260, 80, 430, 60, 970]),
        (42, 1, [310, 260, 960, 460, 810, 110, 470, 30]),
        (42, 7, [830, 50, 290, 530, 740, 820, 630, 260]),
        (0xDEAD_BEEF, 0, [570, 750, 980, 610, 430, 990, 840, 310]),
        (0xDEAD_BEEF, 1, [410, 710, 390, 780, 250, 940, 800, 140]),
        (0xDEAD_BEEF, 7, [630, 990, 250, 200, 900, 170, 360, 130]),
    ];
    for (seed, epoch, want) in cases {
        let s = BatchSampler::new(&tokens, cfg(10, 1, seed)).unwrap();
        assert_eq!(s.windows_per_epoch(), 100);
        let got: Vec<u64> = (0..8).map(|o| s.window_start(epoch, o).unwrap()).collect();
        assert_eq!(got, want, "seed {seed:#x} epoch {epoch}");
    }
}

#[test]
fn rng_state_layout_is_exact_and_refuses_anything_else() {
    let state = SamplerRngState {
        seed: 0x0123_4567_89AB_CDEF,
    };
    let bytes = state.encode();
    assert_eq!(bytes.len(), RNG_STATE_BYTES);
    assert_eq!(&bytes[0..4], &RNG_STATE_VERSION.to_le_bytes());
    assert_eq!(
        &bytes[4..8],
        &RNG_GENERATOR_FEISTEL4_SPLITMIX64.to_le_bytes()
    );
    assert_eq!(&bytes[8..16], &0x0123_4567_89AB_CDEFu64.to_le_bytes());
    assert_eq!(SamplerRngState::decode(&bytes).unwrap(), state);
    let top = SamplerRngState { seed: u64::MAX };
    assert_eq!(SamplerRngState::decode(&top.encode()).unwrap(), top);

    // Every length but the layout's, including the empty pre-v1 state.
    for n in [0, 1, 8, 15, 17, 32] {
        let mut v = bytes.to_vec();
        v.resize(n, 0);
        let err = SamplerRngState::decode(&v).unwrap_err();
        assert!(err.detail().contains("bytes"), "{n}: {err}");
    }
    for version in [0u32, 2, u32::MAX] {
        let mut v = bytes;
        v[0..4].copy_from_slice(&version.to_le_bytes());
        let err = SamplerRngState::decode(&v).unwrap_err();
        assert!(err.detail().contains("version"), "{version}: {err}");
    }
    for generator in [0u32, 2, u32::MAX] {
        let mut v = bytes;
        v[4..8].copy_from_slice(&generator.to_le_bytes());
        let err = SamplerRngState::decode(&v).unwrap_err();
        assert!(err.detail().contains("generator"), "{generator}: {err}");
    }
}

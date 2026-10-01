use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static SEQ: AtomicU64 = AtomicU64::new(0);

fn unique(tag: &str) -> PathBuf {
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("ojas-io-{}-{tag}-{n}", std::process::id()))
}

pub(crate) struct Tmp(pub(crate) PathBuf);

impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

pub(crate) fn tmp(tag: &str) -> Tmp {
    Tmp(unique(tag))
}

pub(crate) struct TmpDir(pub(crate) PathBuf);

impl Drop for TmpDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

pub(crate) fn tmp_dir(tag: &str) -> TmpDir {
    let dir = unique(tag);
    std::fs::create_dir_all(&dir).unwrap();
    TmpDir(dir)
}

/// SplitMix64. Deterministic stream for the mutation tests.
pub(crate) struct Mix(u64);

impl Mix {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub(crate) fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform-ish in `0..n`. `n` must be nonzero.
    pub(crate) fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

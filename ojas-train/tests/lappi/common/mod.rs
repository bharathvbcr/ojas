//! Shared support for `qd-train`'s integration tests: fixture paths, a byte-counting
//! [`ShardFiles`], and scratch directories built without a `tempfile` dependency.

#![allow(dead_code)]

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use qd_train::files::{OsFiles, RangedRead, ShardFiles};
use qd_train::held_out::DataConfig;
use qd_train::shards::{ShardError, ShardOpen, ShardReader};

/// `crates/qd-train`.
pub fn crate_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// The repository root.
pub fn repo_root() -> PathBuf {
    crate_dir().join("..").join("..")
}

/// `python/qd_data`: the sources a header's `code_fingerprint` is compared with.
pub fn qd_data_dir() -> PathBuf {
    repo_root().join("python").join("qd_data")
}

/// The oracle's fixture: `tools/qd_train_oracle_shards.py --out` this directory.
pub fn fixture() -> PathBuf {
    crate_dir()
        .join("tests")
        .join("fixtures")
        .join("shards-tiny")
}

/// A JSON file from the oracle's dump.
pub fn oracle(name: &str) -> serde_json::Value {
    let path = fixture().join("oracle").join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The oracle's `corpus_rev`, so the rev check runs.
pub fn corpus_rev() -> String {
    oracle("meta.json")["corpus_rev"]
        .as_str()
        .expect("corpus_rev")
        .to_owned()
}

/// Options for the fixture: its own root anchors the held-out roots, the revision is the one
/// it was written at, and `allow_stale_code` is set because the fixture is a frozen artifact
/// whose `qd_data` fingerprint will drift as the package changes (the check still runs and
/// is recorded; `door.rs` tests the refusal itself).
pub fn fixture_open(files: Arc<dyn ShardFiles>) -> ShardOpen {
    let mut opts = ShardOpen::new(DataConfig::default(), fixture(), qd_data_dir());
    opts.expect_rev = Some(corpus_rev());
    opts.allow_stale_code = true;
    opts.files = files;
    opts
}

/// The fixture's train shard set, opened through the door on the real filesystem.
pub fn open_train() -> ShardReader {
    ShardReader::open(
        &fixture().join("shards").join("train"),
        &fixture().join("data").join("pool").join("train.json"),
        &fixture_open(Arc::new(OsFiles)),
    )
    .expect("the fixture's train shard set opens through the door")
}

/// A fresh, empty scratch directory unique to this test process and `tag`.
pub fn scratch(tag: &str) -> PathBuf {
    static N: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "qd-train-test-{}-{}-{tag}",
        std::process::id(),
        N.fetch_add(1, Ordering::SeqCst)
    ));
    if dir.exists() {
        std::fs::remove_dir_all(&dir).expect("clear a stale scratch dir");
    }
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

/// Copy a flat directory (a shard set is one).
pub fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).expect("create copy target");
    for entry in std::fs::read_dir(from).expect("read fixture dir") {
        let entry = entry.expect("dir entry");
        std::fs::copy(entry.path(), to.join(entry.file_name())).expect("copy fixture file");
    }
}

/// A [`ShardFiles`] that counts every byte it hands out, per path, and every file it opens.
#[derive(Default)]
pub struct CountingFiles {
    inner: OsFiles,
    bytes: Arc<Mutex<HashMap<PathBuf, u64>>>,
}

impl CountingFiles {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    fn add(map: &Mutex<HashMap<PathBuf, u64>>, path: &Path, n: u64) {
        *map.lock()
            .expect("counter lock")
            .entry(path.to_path_buf())
            .or_insert(0) += n;
    }

    /// Bytes read from exactly `path`.
    pub fn bytes_read(&self, path: &Path) -> u64 {
        self.bytes
            .lock()
            .expect("counter lock")
            .get(path)
            .copied()
            .unwrap_or(0)
    }

    /// Bytes read from any file under `dir`.
    pub fn bytes_read_under(&self, dir: &Path) -> u64 {
        self.bytes
            .lock()
            .expect("counter lock")
            .iter()
            .filter(|(p, _)| p.starts_with(dir))
            .map(|(_, n)| *n)
            .sum()
    }

    /// Bytes read from anywhere.
    pub fn total(&self) -> u64 {
        self.bytes.lock().expect("counter lock").values().sum()
    }
}

impl ShardFiles for CountingFiles {
    fn read(&self, path: &Path, max_bytes: u64) -> io::Result<Vec<u8>> {
        let out = self.inner.read(path, max_bytes)?;
        Self::add(&self.bytes, path, out.len() as u64);
        Ok(out)
    }

    fn len(&self, path: &Path) -> io::Result<u64> {
        self.inner.len(path)
    }

    fn is_file(&self, path: &Path) -> bool {
        self.inner.is_file(path)
    }

    fn open_ranged(&self, path: &Path) -> io::Result<Box<dyn RangedRead>> {
        Ok(Box::new(CountingRanged {
            inner: self.inner.open_ranged(path)?,
            path: path.to_path_buf(),
            bytes: Arc::clone(&self.bytes),
        }))
    }
}

struct CountingRanged {
    inner: Box<dyn RangedRead>,
    path: PathBuf,
    bytes: Arc<Mutex<HashMap<PathBuf, u64>>>,
}

impl RangedRead for CountingRanged {
    fn read_exact_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        self.inner.read_exact_at(offset, buf)?;
        CountingFiles::add(&self.bytes, &self.path, buf.len() as u64);
        Ok(())
    }
}

/// Open, and if the door admitted the set, read every sequence -- so a door that wrongly
/// admits shows up as token bytes read, not only as a missing error.
pub fn open_and_drain(
    root: &Path,
    manifest: &Path,
    opts: &ShardOpen,
) -> Result<ShardReader, ShardError> {
    let reader = ShardReader::open(root, manifest, opts)?;
    for i in 0..reader.len() {
        reader.sequence(i)?;
    }
    Ok(reader)
}

// --- The trainer half (L-trainer): the trainer oracle's loader and the loop's test doubles, a
// toy FT batch and a toy pointer head, both real implementations of the crate's small
// interfaces (`FtBatch`, `SpanHead`).

use qd_train::objective::{fold_batch_parts, FtBatch, LetterTarget, SpanHead};
use qd_train::pyjson::float_fromhex;
use qd_train::run_control::{ConsumedPrefix, RunControlError};
use qd_train::step::ParamSpec;
use qd_train::trainer::{ConsumedBatch, HostParams};

/// `tools/qd_train_oracle_trainer.py --out` this file.
pub fn trainer_oracle() -> serde_json::Value {
    let path = crate_dir().join("tests").join("fixtures").join("trainer-oracle.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

pub fn fhex(v: &serde_json::Value) -> f64 {
    float_fromhex(v.as_str().expect("a float.hex string")).expect("parse float.hex")
}

pub fn fhex_list(v: &serde_json::Value) -> Vec<f64> {
    v.as_array().expect("a list").iter().map(fhex).collect()
}

/// A span row for [`ToySpanHead`]: the candidate positions and the gold start/end as indices
/// into the candidates, `candidates.len()` meaning "abstain".
#[derive(Debug, Clone, PartialEq)]
pub struct ToySpan {
    pub candidates: Vec<u32>,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone)]
pub struct ToyRow {
    pub tokens: Vec<u32>,
    pub letter: Option<LetterTarget>,
    pub span: Option<ToySpan>,
}

#[derive(Debug, Clone)]
pub struct ToyBatch {
    pub index: u64,
    pub bucket: u64,
    pub width: usize,
    pub rows: Vec<ToyRow>,
}

impl ConsumedBatch for ToyBatch {
    fn index(&self) -> u64 {
        self.index
    }

    fn fold_into(&self, prefix: &mut ConsumedPrefix) -> Result<(), RunControlError> {
        let mut tokens = Vec::new();
        let mut lengths = Vec::new();
        for r in &self.rows {
            for c in 0..self.width {
                tokens.extend_from_slice(&r.tokens.get(c).copied().unwrap_or(0).to_le_bytes());
            }
            lengths.extend_from_slice(&(r.tokens.len() as i32).to_le_bytes());
        }
        let target: Vec<u8> = self
            .rows
            .iter()
            .flat_map(|r| i64::from(r.letter.map_or(-1, |l| l.position as i32)).to_le_bytes())
            .collect();
        fold_batch_parts(prefix, self.index, self.bucket, &tokens, &lengths, [None, Some(&target), None, None])
    }
}

impl FtBatch for ToyBatch {
    type Span = ToySpan;

    fn n_rows(&self) -> usize {
        self.rows.len()
    }

    fn padded_width(&self) -> usize {
        self.width
    }

    fn row_tokens(&self, row: usize) -> &[u32] {
        &self.rows[row].tokens
    }

    fn letter(&self, row: usize) -> Option<LetterTarget> {
        self.rows[row].letter
    }

    fn span(&self, row: usize) -> Option<&ToySpan> {
        self.rows[row].span.as_ref()
    }
}

/// A pointer head: score(c) = u . h[c] for each candidate, abstain score b; start and end are
/// two cross-entropies over the same scores (the toy shares one scorer for both).
pub struct ToySpanHead {
    pub u: Vec<f32>,
    pub b: Vec<f32>,
    pub gu: Vec<f32>,
    pub gb: Vec<f32>,
}

impl ToySpanHead {
    pub fn new(hidden: usize) -> Self {
        Self {
            u: (0..hidden).map(|i| 0.3 - 0.07 * i as f32).collect(),
            b: vec![0.2],
            gu: vec![0.0; hidden],
            gb: vec![0.0],
        }
    }

    fn scores(&self, hidden: &[f32], width: usize) -> Vec<f64> {
        let mut s: Vec<f64> = hidden
            .chunks(width)
            .map(|h| h.iter().zip(&self.u).map(|(a, b)| f64::from(a * b)).sum())
            .collect();
        s.push(f64::from(self.b[0]));
        s
    }

    /// The row's unscaled loss with the head as it is (for central differences).
    pub fn row_loss(&self, span: &ToySpan, hidden: &[f32], width: usize) -> f64 {
        let s = self.scores(hidden, width);
        let max = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let lse = max + s.iter().map(|x| (x - max).exp()).sum::<f64>().ln();
        (lse - s[span.start]) + (lse - s[span.end])
    }
}

impl HostParams for ToySpanHead {
    fn entries(&self) -> Vec<ParamSpec> {
        vec![ParamSpec::new("pointer.weight", &[self.u.len()]), ParamSpec::new("abstain", &[1])]
    }

    fn zero_grads(&mut self) {
        self.gu.iter_mut().for_each(|x| *x = 0.0);
        self.gb[0] = 0.0;
    }

    fn grads(&self) -> Vec<&[f32]> {
        vec![&self.gu, &self.gb]
    }

    fn values(&self) -> Vec<&[f32]> {
        vec![&self.u, &self.b]
    }

    fn values_and_grads(&mut self) -> Vec<(&mut [f32], &[f32])> {
        vec![(&mut self.u, &self.gu), (&mut self.b, &self.gb)]
    }
}

impl SpanHead for ToySpanHead {
    type Span = ToySpan;

    fn positions(&self, span: &ToySpan) -> Result<Vec<u32>, String> {
        let mut p = span.candidates.clone();
        p.sort_unstable();
        p.dedup();
        if p.len() != span.candidates.len() || p != span.candidates {
            return Err("candidates must be distinct and ascending".into());
        }
        Ok(p)
    }

    fn loss_and_grad(
        &mut self,
        span: &ToySpan,
        positions: &[u32],
        hidden: &[f32],
        width: usize,
        scale: f32,
    ) -> Result<(f64, Vec<f32>), String> {
        if positions.len() * width != hidden.len() {
            return Err("hidden does not fit the positions".into());
        }
        let s = self.scores(hidden, width);
        let max = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let lse = max + s.iter().map(|x| (x - max).exp()).sum::<f64>().ln();
        let loss = (lse - s[span.start]) + (lse - s[span.end]);
        // d loss / d s_j = 2 softmax_j - [j == start] - [j == end]
        let ds: Vec<f64> = (0..s.len())
            .map(|j| 2.0 * (s[j] - lse).exp() - f64::from(u8::from(j == span.start)) - f64::from(u8::from(j == span.end)))
            .collect();
        let sc = f64::from(scale);
        let mut dh = vec![0.0f32; hidden.len()];
        for (k, h) in hidden.chunks(width).enumerate() {
            let d = (ds[k] * sc) as f32;
            for i in 0..width {
                dh[k * width + i] = d * self.u[i];
                self.gu[i] += d * h[i];
            }
        }
        self.gb[0] += (ds[s.len() - 1] * sc) as f32;
        Ok((loss, dh))
    }
}

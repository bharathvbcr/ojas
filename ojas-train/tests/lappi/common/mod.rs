//! Shared test doubles for the loop: a toy FT batch and a toy pointer head, both real
//! implementations of the crate's small interfaces (`FtBatch`, `SpanHead`), and the oracle
//! fixture loader.

#![allow(dead_code)]

use qd_train::objective::{fold_batch_parts, FtBatch, LetterTarget, SpanHead};
use qd_train::pyjson::float_fromhex;
use qd_train::run_control::{ConsumedPrefix, RunControlError};
use qd_train::step::ParamSpec;
use qd_train::trainer::{ConsumedBatch, HostParams};

pub fn oracle() -> serde_json::Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/trainer-oracle.json");
    serde_json::from_str(&std::fs::read_to_string(path).expect("read the oracle fixture")).expect("parse it")
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

//! GPT-2 tokenizer throughput: `Bpe::encode_ordinary` and `Bpe::decode_ordinary`
//! on a real GPT-2 rank table over a fixed multi-MB English text.
//!
//! Compiled by `#[path]` into `ojas-data/examples/bpe_throughput.rs`. It uses
//! only `load_hf_gpt2`, `encode_ordinary`, `decode_ordinary` and `TokenBin`,
//! so the same file builds against an older tree for an A/B.
//!
//! The text is nanolab's FineWeb-Edu `val.bin` (headerless `u16`, 2.5M
//! tokens), which `nanolab/prep_fineweb.py` wrote as tiktoken 0.12
//! `encode_ordinary(doc) + [eot]` per document. The stream is split at
//! `<|endoftext|>` (50256); the first and last pieces can be partial slices of
//! a document and are dropped. Each remaining document is decoded once,
//! outside the timer, and that text is the input.
//!
//! Rows, each timed once per round, min and median over rounds:
//! - `encode_docs`: `encode_ordinary` once per document, as a data pipeline
//!   calls it.
//! - `encode_joined`: one `encode_ordinary` over every document joined by
//!   `"\n\n"` (one call of about 10 MB).
//! - `decode_docs`: `decode_ordinary` once per document.
//! - `decode_joined`: one `decode_ordinary` over the `encode_joined` ids.
//!
//! Before timing, every document's re-encode is compared with its tiktoken ids
//! from the bin (a byte-identity check over the whole text, not twenty
//! strings) and its decode with its text; the run refuses to time on a
//! mismatch. `digest` is FNV-1a over every `encode_joined` id, so two trees
//! that print the same digest produced the same ids.
//!
//! Usage: `bpe_throughput VOCAB_DIR CORPUS_BIN [ROUNDS]` where `VOCAB_DIR`
//! holds `vocab.json` and `merges.txt`.

use ojas_data::{load_hf_gpt2, Bpe, TokenBin};
use std::path::Path;
use std::time::{Duration, Instant};

const EOT: u32 = 50256;

struct Corpus {
    docs: Vec<String>,
    doc_ids: Vec<Vec<u32>>,
    joined: String,
}

fn load_corpus(bpe: &Bpe, bin: &Path) -> Corpus {
    let bin = TokenBin::open_headerless(bin).unwrap_or_else(|e| panic!("corpus: {e}"));
    let n = usize::try_from(bin.len()).expect("token count fits usize");
    let mut stream = vec![0u32; n];
    bin.read_into_u32(0, &mut stream)
        .unwrap_or_else(|e| panic!("corpus read: {e}"));
    let mut pieces: Vec<&[u32]> = stream.split(|&t| t == EOT).collect();
    // The slice can start and end inside a document.
    pieces.remove(0);
    pieces.pop();
    let mut docs = Vec::new();
    let mut doc_ids = Vec::new();
    for ids in pieces.into_iter().filter(|p| !p.is_empty()) {
        let text = bpe
            .decode_ordinary(ids)
            .unwrap_or_else(|e| panic!("corpus document does not decode: {e}"));
        docs.push(text);
        doc_ids.push(ids.to_vec());
    }
    let joined = docs.join("\n\n");
    Corpus {
        docs,
        doc_ids,
        joined,
    }
}

fn fnv1a(ids: &[u32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for id in ids {
        for b in id.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

fn time<T>(f: impl FnOnce() -> T) -> (Duration, T) {
    let start = Instant::now();
    let out = f();
    (start.elapsed(), out)
}

struct Row {
    name: &'static str,
    bytes: usize,
    tokens: usize,
    times: Vec<Duration>,
}

impl Row {
    fn new(name: &'static str, bytes: usize, tokens: usize) -> Self {
        Self {
            name,
            bytes,
            tokens,
            times: Vec::new(),
        }
    }

    fn min_med(&self) -> (f64, f64) {
        let mut s: Vec<f64> = self.times.iter().map(Duration::as_secs_f64).collect();
        s.sort_by(f64::total_cmp);
        (s[0], s[s.len() / 2])
    }
}

pub fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: {} VOCAB_DIR CORPUS_BIN [ROUNDS]", args[0]);
        std::process::exit(2);
    }
    let dir = Path::new(&args[1]);
    let rounds: usize = args.get(3).map_or(5, |r| r.parse().expect("ROUNDS"));
    assert!(rounds >= 1, "ROUNDS must be at least 1");

    let (load_t, bpe) = time(|| {
        load_hf_gpt2(&dir.join("vocab.json"), &dir.join("merges.txt"))
            .unwrap_or_else(|e| panic!("rank table: {e}"))
    });
    // Fingerprint: the GPT-2 table, not some other 50257-piece file.
    assert_eq!(bpe.decode_ordinary(&[31373]).unwrap(), "hello");
    assert_eq!(bpe.decode_ordinary(&[EOT]).unwrap(), "<|endoftext|>");

    let corpus = load_corpus(&bpe, Path::new(&args[2]));
    let doc_bytes: usize = corpus.docs.iter().map(String::len).sum();
    let doc_tokens: usize = corpus.doc_ids.iter().map(Vec::len).sum();

    // Parity before timing.
    let mut mismatched = 0usize;
    for (i, (text, want)) in corpus.docs.iter().zip(&corpus.doc_ids).enumerate() {
        let got = bpe.encode_ordinary(text).unwrap();
        if &got != want {
            if mismatched < 3 {
                let at = got.iter().zip(want).position(|(a, b)| a != b);
                eprintln!(
                    "document {i}: re-encode differs from tiktoken at {at:?} (got {} ids, want {})",
                    got.len(),
                    want.len()
                );
            }
            mismatched += 1;
        }
    }
    if mismatched != 0 {
        eprintln!(
            "{mismatched} of {} documents differ from tiktoken; not timed",
            corpus.docs.len()
        );
        std::process::exit(1);
    }
    let joined_ids = bpe.encode_ordinary(&corpus.joined).unwrap();
    assert_eq!(bpe.decode_ordinary(&joined_ids).unwrap(), corpus.joined);

    let mut rows = [
        Row::new("encode_docs", doc_bytes, doc_tokens),
        Row::new("encode_joined", corpus.joined.len(), joined_ids.len()),
        Row::new("decode_docs", doc_bytes, doc_tokens),
        Row::new("decode_joined", corpus.joined.len(), joined_ids.len()),
    ];
    // One untimed warm-up of every row.
    for round in 0..=rounds {
        let (t, out) = time(|| {
            corpus
                .docs
                .iter()
                .map(|d| bpe.encode_ordinary(d).unwrap().len())
                .sum::<usize>()
        });
        assert_eq!(out, doc_tokens);
        let t0 = t;
        let (t1, ids) = time(|| bpe.encode_ordinary(&corpus.joined).unwrap());
        assert_eq!(ids.len(), joined_ids.len());
        let (t2, out) = time(|| {
            corpus
                .doc_ids
                .iter()
                .map(|ids| bpe.decode_ordinary(ids).unwrap().len())
                .sum::<usize>()
        });
        assert_eq!(out, doc_bytes);
        let (t3, text) = time(|| bpe.decode_ordinary(&joined_ids).unwrap());
        assert_eq!(text.len(), corpus.joined.len());
        if round > 0 {
            for (row, t) in rows.iter_mut().zip([t0, t1, t2, t3]) {
                row.times.push(t);
            }
        }
    }

    println!("vocab_dir={}", dir.display());
    println!("corpus={}", args[2]);
    println!("load_hf_gpt2_s={:.4}", load_t.as_secs_f64());
    println!(
        "documents={} doc_bytes={doc_bytes} doc_tokens={doc_tokens} joined_bytes={} joined_tokens={}",
        corpus.docs.len(),
        corpus.joined.len(),
        joined_ids.len()
    );
    println!(
        "tiktoken_parity=ok documents_matched={}/{}",
        corpus.docs.len(),
        corpus.docs.len()
    );
    println!("digest={:016x}", fnv1a(&joined_ids));
    println!("rounds={rounds}");
    println!("| row | min s | median s | MB/s (min) | Mtok/s (min) |");
    println!("| :-- | --: | --: | --: | --: |");
    for row in &rows {
        let (min, med) = row.min_med();
        println!(
            "| {} | {min:.4} | {med:.4} | {:.2} | {:.3} |",
            row.name,
            row.bytes as f64 / 1e6 / min,
            row.tokens as f64 / 1e6 / min
        );
    }
}

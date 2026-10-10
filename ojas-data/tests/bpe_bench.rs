//! Tokenizer throughput: `encode_ordinary` MB/s and tokens/s over a fixed
//! generated English-like text, plus `decode_ordinary`. Ignored by default:
//! `cargo test -p ojas-data --release --test bpe_bench -- --ignored --nocapture`.
//!
//! `OJAS_GPT2_DIR` names a directory holding a Hugging Face `vocab.json` and
//! `merges.txt`. Without it the vocabulary is synthetic: the 256 pieces of
//! `bytes_to_unicode` plus up to 20k merges learned greedily from the text.
//! `OJAS_BPE_BENCH_RUNS` sets the run count (default 5); each row is the
//! fastest run. `ids_fnv` hashes the encoded ids, so two builds can be
//! compared for bit-identical output.

use ojas_data::{bytes_to_unicode, gpt2_split, load_hf_gpt2, Bpe, BpeBuilder, CounterRng};
use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap, HashMap};
use std::time::Instant;

const TEXT_BYTES: usize = 4 << 20;
const SYNTHETIC_MERGES: usize = 20_000;

const WORDS: &[&str] = &[
    "the",
    "of",
    "and",
    "to",
    "a",
    "in",
    "is",
    "that",
    "it",
    "was",
    "for",
    "on",
    "are",
    "as",
    "with",
    "his",
    "they",
    "at",
    "be",
    "this",
    "from",
    "have",
    "or",
    "by",
    "one",
    "had",
    "not",
    "but",
    "what",
    "all",
    "were",
    "when",
    "we",
    "there",
    "can",
    "an",
    "your",
    "which",
    "their",
    "said",
    "if",
    "do",
    "will",
    "each",
    "about",
    "how",
    "up",
    "out",
    "them",
    "then",
    "she",
    "many",
    "some",
    "so",
    "these",
    "would",
    "other",
    "into",
    "has",
    "more",
    "her",
    "two",
    "like",
    "him",
    "see",
    "time",
    "could",
    "no",
    "make",
    "than",
    "first",
    "been",
    "its",
    "who",
    "now",
    "people",
    "my",
    "made",
    "over",
    "did",
    "down",
    "only",
    "way",
    "find",
    "use",
    "may",
    "water",
    "long",
    "little",
    "very",
    "after",
    "words",
    "called",
    "just",
    "where",
    "most",
    "know",
    "don't",
    "it's",
    "we'll",
    "they're",
    "I've",
    "government",
    "between",
    "important",
    "understand",
    "information",
    "development",
    "community",
    "café",
    "naïve",
    "—",
    "“quoted”",
];

const SYLLABLES: &[&str] = &[
    "ba", "ter", "ing", "con", "ver", "sa", "tion", "pre", "dis", "ment", "al", "ly", "ro", "ma",
    "ne", "ti", "cal", "ex", "per", "for", "un", "der", "lo", "gy", "mi", "cro", "na", "ture",
    "or", "ous", "ble", "pro", "re", "si", "de", "qu", "est", "an", "on", "ic",
];

fn pick<'a>(rng: &mut CounterRng, items: &[&'a str]) -> &'a str {
    items[(rng.next_u64() % items.len() as u64) as usize]
}

/// Deterministic English-like prose: frequent words skewed to the front of
/// [`WORDS`], a long tail of made-up words, numbers, punctuation, and
/// paragraphs. Stops at the first sentence end past `bytes`.
fn english_like(bytes: usize) -> String {
    let mut rng = CounterRng::new(0x7E47);
    let mut out = String::with_capacity(bytes + 256);
    while out.len() < bytes {
        for _ in 0..3 + rng.next_u64() % 5 {
            let words = 6 + rng.next_u64() % 15;
            for w in 0..words {
                let roll = rng.next_u64() % 100;
                let word = if roll < 78 {
                    let a = rng.next_u64() % WORDS.len() as u64;
                    let b = rng.next_u64() % WORDS.len() as u64;
                    WORDS[a.min(b) as usize].to_string()
                } else if roll < 96 {
                    let n = 2 + rng.next_u64() % 3;
                    (0..n).map(|_| pick(&mut rng, SYLLABLES)).collect()
                } else {
                    (rng.next_u64() % 10_000).to_string()
                };
                if w > 0 {
                    out.push(' ');
                    out.push_str(&word);
                } else {
                    let mut chars = word.chars();
                    if let Some(c) = chars.next() {
                        out.extend(c.to_uppercase());
                        out.push_str(chars.as_str());
                    }
                }
                if w + 1 < words && rng.next_u64().is_multiple_of(9) {
                    out.push(',');
                }
            }
            out.push_str(pick(&mut rng, &[".", ".", ".", "?", "!"]));
            out.push(' ');
        }
        out.pop();
        out.push_str("\n\n");
    }
    out
}

/// Greedy BPE training over the text's pre-tokens: merge the most frequent
/// adjacent pair (smallest ids on ties) until `merges` are learned or no
/// pair occurs twice. Ids 0..256 are the byte pieces in byte order.
fn learn_bpe(text: &str, merges: usize) -> Bpe {
    let map = bytes_to_unicode();
    let mut pieces: Vec<String> = map.iter().map(|c| c.to_string()).collect();
    let mut piece_ids: HashMap<String, u32> = pieces
        .iter()
        .enumerate()
        .map(|(i, p)| (p.clone(), i as u32))
        .collect();
    let mut counts: HashMap<&str, i64> = HashMap::new();
    for word in gpt2_split(text) {
        *counts.entry(word).or_default() += 1;
    }
    let mut words: Vec<(Vec<u32>, i64)> = counts
        .into_iter()
        .map(|(w, c)| (w.bytes().map(u32::from).collect(), c))
        .collect();
    words.sort_unstable();
    let mut pair_count: HashMap<(u32, u32), i64> = HashMap::new();
    let mut holders: HashMap<(u32, u32), BTreeSet<usize>> = HashMap::new();
    for (wi, (w, c)) in words.iter().enumerate() {
        for p in w.windows(2) {
            *pair_count.entry((p[0], p[1])).or_default() += c;
            holders.entry((p[0], p[1])).or_default().insert(wi);
        }
    }
    let mut heap: BinaryHeap<(i64, Reverse<(u32, u32)>)> =
        pair_count.iter().map(|(&p, &c)| (c, Reverse(p))).collect();
    let mut learned = Vec::new();
    while learned.len() < merges {
        let Some((c, Reverse(pair))) = heap.pop() else {
            break;
        };
        if pair_count.get(&pair) != Some(&c) {
            continue;
        }
        if c < 2 {
            break;
        }
        pair_count.remove(&pair);
        let text = format!("{}{}", pieces[pair.0 as usize], pieces[pair.1 as usize]);
        let merged = *piece_ids.entry(text.clone()).or_insert_with(|| {
            pieces.push(text);
            pieces.len() as u32 - 1
        });
        learned.push((pair.0, pair.1, merged));
        let mut touched = BTreeSet::new();
        for wi in holders.remove(&pair).unwrap_or_default() {
            let (w, c) = &mut words[wi];
            for p in w.windows(2) {
                if (p[0], p[1]) != pair {
                    *pair_count.entry((p[0], p[1])).or_default() -= *c;
                    touched.insert((p[0], p[1]));
                }
            }
            let mut merged_word = Vec::with_capacity(w.len());
            let mut i = 0;
            while i < w.len() {
                if i + 1 < w.len() && (w[i], w[i + 1]) == pair {
                    merged_word.push(merged);
                    i += 2;
                } else {
                    merged_word.push(w[i]);
                    i += 1;
                }
            }
            *w = merged_word;
            for p in w.windows(2) {
                *pair_count.entry((p[0], p[1])).or_default() += *c;
                holders.entry((p[0], p[1])).or_default().insert(wi);
                touched.insert((p[0], p[1]));
            }
        }
        for p in touched {
            if p != pair {
                heap.push((pair_count[&p], Reverse(p)));
            }
        }
    }
    let mut b = BpeBuilder::new();
    for (id, piece) in pieces.iter().enumerate() {
        b = b.token(id as u32, piece).unwrap();
    }
    for (rank, &(l, r, merged)) in learned.iter().enumerate() {
        b = b.merge(l, r, rank as u32, merged).unwrap();
    }
    b.build().unwrap()
}

fn fnv(ids: &[u32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in ids.iter().flat_map(|id| id.to_le_bytes()) {
        h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// Fastest of `runs` timed calls, in seconds, and the last result.
fn fastest<T>(runs: usize, mut f: impl FnMut() -> T) -> (f64, T) {
    let mut best = f64::INFINITY;
    let mut last = None;
    for _ in 0..runs {
        let t0 = Instant::now();
        let out = std::hint::black_box(f());
        best = best.min(t0.elapsed().as_secs_f64());
        last = Some(out);
    }
    (best, last.expect("at least one run"))
}

#[test]
#[ignore]
fn encode_ordinary_and_decode_throughput() {
    let runs: usize = std::env::var("OJAS_BPE_BENCH_RUNS")
        .ok()
        .and_then(|r| r.parse().ok())
        .unwrap_or(5)
        .max(1);
    let text = english_like(TEXT_BYTES);
    let (vocab, bpe) = match std::env::var_os("OJAS_GPT2_DIR") {
        Some(dir) => {
            let dir = std::path::PathBuf::from(dir);
            let bpe = load_hf_gpt2(&dir.join("vocab.json"), &dir.join("merges.txt"))
                .unwrap_or_else(|e| panic!("OJAS_GPT2_DIR {}: {e}", dir.display()));
            ("gpt2", bpe)
        }
        None => ("synthetic", learn_bpe(&text, SYNTHETIC_MERGES)),
    };
    let mb = text.len() as f64 / 1e6;

    let (split_s, _) = fastest(runs, || gpt2_split(&text).len());
    let (enc_s, ids) = fastest(runs, || bpe.encode_ordinary(&text).unwrap());
    let paragraphs: Vec<&str> = text.split_inclusive("\n\n").collect();
    let (para_s, para_tokens) = fastest(runs, || {
        paragraphs
            .iter()
            .map(|p| bpe.encode_ordinary(p).unwrap().len())
            .sum::<usize>()
    });
    let (dec_s, decoded) = fastest(runs, || bpe.decode_ordinary(&ids).unwrap());
    assert_eq!(decoded, text);
    let tokens = ids.len() as f64;
    println!(
        "OJAS_BPE vocab={vocab} runs={runs} bytes={} tokens={} ids_fnv={:#018x} \
         split_mb_s={:.2} encode_mb_s={:.2} encode_tok_s={:.0} \
         para_calls={} para_tokens={para_tokens} para_encode_mb_s={:.2} \
         decode_mb_s={:.2} decode_tok_s={:.0}",
        text.len(),
        ids.len(),
        fnv(&ids),
        mb / split_s,
        mb / enc_s,
        tokens / enc_s,
        paragraphs.len(),
        mb / para_s,
        mb / dec_s,
        tokens / dec_s,
    );
}

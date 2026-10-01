//! Byte-pair encoder.
//!
//! [`fixture_bpe`] round-trips a hand-written vocabulary. It does not use
//! tiktoken ranks.
//!
//! [`load_hf_gpt2`] reads a Hugging Face `vocab.json` and `merges.txt`.
//! [`Bpe::encode_ordinary`] applies the GPT-2 regex split, the byte alphabet,
//! then those merges. Twenty strings match tiktoken 0.12.0 `encode_ordinary`
//! ([`TIKTOKEN_GPT2_BYTE_IDENTITY`]). That is not a million-line check.
//!
//! [`bytes_to_unicode`] is the GPT-2 byte alphabet (a fixed bijection from
//! bytes to Unicode scalars). It is not a merge table and it does not encode text.

use crate::error::DataError;
use crate::gpt2_class::{is_letter, is_number, is_space};
use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};
use std::path::Path;

/// `(rank, left position, right position, left id, right id)`. The ids let a
/// popped entry be checked against the current list; stale entries are skipped.
type HeapPair = Reverse<(u32, usize, usize, u32, u32)>;

/// Byte-identity of [`Bpe::encode_ordinary`] with tiktoken `gpt2` `encode_ordinary`.
///
/// The twenty strings in the encoder test match tiktoken 0.12.0
/// `Encoding.encode_ordinary` using the gpt2 pattern (`r50k_pat_str`) and the
/// ranks in the local Hugging Face `vocab.json`. The system `python3` does not
/// import tiktoken; the check used
/// `Step-Audio-EditX/.venv` (tiktoken 0.12.0) and did not download vocab files.
/// This is not a million-line check.
pub const TIKTOKEN_GPT2_BYTE_IDENTITY: &str = "verified-20-strings";

#[derive(Clone, Debug)]
pub struct Bpe {
    tokens: Vec<String>,
    encoder: BTreeMap<String, u32>,
    /// `(left, right) -> (rank, merged id)`. Lower rank merges first.
    merges: BTreeMap<(u32, u32), (u32, u32)>,
}

pub struct BpeBuilder {
    tokens: BTreeMap<u32, String>,
    merges: Vec<(u32, u32, u32, u32)>,
}

impl BpeBuilder {
    pub fn new() -> Self {
        Self {
            tokens: BTreeMap::new(),
            merges: Vec::new(),
        }
    }

    pub fn token(mut self, id: u32, piece: &str) -> Result<Self, DataError> {
        if piece.is_empty() {
            return Err(DataError::new("empty BPE piece"));
        }
        if self.tokens.insert(id, piece.to_string()).is_some() {
            return Err(DataError::new(format!("duplicate token id {id}")));
        }
        Ok(self)
    }

    /// `merged` must already be a token whose text is `left` concatenated with `right`.
    /// `rank` is this fixture's order. It is not a tiktoken rank.
    pub fn merge(
        mut self,
        left: u32,
        right: u32,
        rank: u32,
        merged: u32,
    ) -> Result<Self, DataError> {
        self.merges.push((left, right, rank, merged));
        Ok(self)
    }

    pub fn build(self) -> Result<Bpe, DataError> {
        if self.tokens.is_empty() {
            return Err(DataError::new("BPE vocabulary is empty"));
        }
        let mut tokens = Vec::new();
        for (i, (id, piece)) in self.tokens.iter().enumerate() {
            if *id != i as u32 {
                return Err(DataError::new(format!(
                    "token ids must be 0..n without holes, missing {i}"
                )));
            }
            tokens.push(piece.clone());
        }
        let mut encoder = BTreeMap::new();
        for (id, piece) in tokens.iter().enumerate() {
            if encoder.insert(piece.clone(), id as u32).is_some() {
                return Err(DataError::new(format!("duplicate piece {piece:?}")));
            }
        }
        let mut merges = BTreeMap::new();
        for (left, right, rank, merged) in self.merges {
            let l = tokens.get(left as usize).ok_or_else(|| {
                DataError::new(format!("merge left id {left} is not in the vocabulary"))
            })?;
            let r = tokens.get(right as usize).ok_or_else(|| {
                DataError::new(format!("merge right id {right} is not in the vocabulary"))
            })?;
            let m = tokens.get(merged as usize).ok_or_else(|| {
                DataError::new(format!("merge result id {merged} is not in the vocabulary"))
            })?;
            if *m != format!("{l}{r}") {
                return Err(DataError::new(format!(
                    "merge {left}+{right} -> {merged} text is {m:?}, not the concatenation"
                )));
            }
            if merges.insert((left, right), (rank, merged)).is_some() {
                return Err(DataError::new(format!(
                    "duplicate merge pair ({left}, {right})"
                )));
            }
        }
        Ok(Bpe {
            tokens,
            encoder,
            merges,
        })
    }
}

impl Default for BpeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Bpe {
    /// Encode `text` by splitting into vocabulary characters, then applying
    /// merges from lowest rank, leftmost first among equal ranks. A character
    /// that is not a vocabulary piece is an error. This split is one Unicode
    /// scalar per piece. GPT-2 regex splitting is [`Self::encode_ordinary`].
    /// Cost is O(n log n) in the character count.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, DataError> {
        let mut ids = Vec::new();
        let mut buf = [0u8; 4];
        for ch in text.chars() {
            let piece: &str = ch.encode_utf8(&mut buf);
            let id = self
                .encoder
                .get(piece)
                .ok_or_else(|| DataError::new(format!("no vocabulary piece for {piece:?}")))?;
            ids.push(*id);
        }
        let n = ids.len();
        // A doubly linked list over character positions. A merge keeps the
        // left node, so list order and position order agree, and the heap key
        // (rank, left position) is the leftmost lowest-rank pair.
        let mut next: Vec<usize> = (1..=n).collect();
        let mut prev: Vec<usize> = (0..n).map(|i| i.wrapping_sub(1)).collect();
        let mut alive = vec![true; n];
        let mut heap = BinaryHeap::new();
        for i in 1..n {
            self.push_pair(&mut heap, &ids, i - 1, i);
        }
        while let Some(Reverse((_, l, r, left_id, right_id))) = heap.pop() {
            if !alive[l] || next[l] != r || ids[l] != left_id || ids[r] != right_id {
                continue;
            }
            let Some(&(_, merged)) = self.merges.get(&(left_id, right_id)) else {
                return Err(DataError::new("merge pair vanished"));
            };
            ids[l] = merged;
            alive[r] = false;
            next[l] = next[r];
            if next[l] < n {
                prev[next[l]] = l;
            }
            if prev[l] < n {
                self.push_pair(&mut heap, &ids, prev[l], l);
            }
            if next[l] < n {
                self.push_pair(&mut heap, &ids, l, next[l]);
            }
        }
        Ok(ids
            .into_iter()
            .zip(alive)
            .filter_map(|(id, live)| live.then_some(id))
            .collect())
    }

    fn push_pair(&self, heap: &mut BinaryHeap<HeapPair>, ids: &[u32], l: usize, r: usize) {
        if let Some(&(rank, _)) = self.merges.get(&(ids[l], ids[r])) {
            heap.push(Reverse((rank, l, r, ids[l], ids[r])));
        }
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, DataError> {
        let mut out = String::new();
        for &id in ids {
            let piece = self.tokens.get(id as usize).ok_or_else(|| {
                DataError::new(format!("token id {id} is outside the vocabulary"))
            })?;
            out.push_str(piece);
        }
        Ok(out)
    }

    /// GPT-2 ordinary encode: regex pre-tokenize, map each piece's bytes through
    /// [`bytes_to_unicode`], then merge. Special tokens are ordinary text.
    ///
    /// Byte-identity with tiktoken is [`TIKTOKEN_GPT2_BYTE_IDENTITY`].
    pub fn encode_ordinary(&self, text: &str) -> Result<Vec<u32>, DataError> {
        let map = bytes_to_unicode();
        let mut ids = Vec::new();
        let mut mapped = String::new();
        for piece in gpt2_split(text) {
            mapped.clear();
            for &b in piece.as_bytes() {
                mapped.push(map[b as usize]);
            }
            ids.extend(self.encode(&mapped)?);
        }
        Ok(ids)
    }

    /// Invert [`Self::encode_ordinary`]: token pieces to bytes via the inverse
    /// byte alphabet, then UTF-8. A piece character outside that alphabet, or
    /// bytes that are not UTF-8, is an error.
    pub fn decode_ordinary(&self, ids: &[u32]) -> Result<String, DataError> {
        let rendered = self.decode(ids)?;
        let map = bytes_to_unicode();
        let mut inv = [None; 512];
        for (b, ch) in map.iter().enumerate() {
            let u = u32::from(*ch) as usize;
            if u >= inv.len() {
                return Err(DataError::new("byte alphabet exceeds the inverse table"));
            }
            inv[u] = Some(b as u8);
        }
        let mut bytes = Vec::with_capacity(rendered.len());
        for ch in rendered.chars() {
            let u = u32::from(ch) as usize;
            let Some(b) = inv.get(u).copied().flatten() else {
                return Err(DataError::new(format!(
                    "token text has no GPT-2 byte for {ch:?}"
                )));
            };
            bytes.push(b);
        }
        String::from_utf8(bytes)
            .map_err(|e| DataError::new(format!("decoded bytes are not utf-8: {e}")))
    }

    fn from_ranked_pieces(
        pieces: Vec<String>,
        merges: Vec<(String, String)>,
    ) -> Result<Bpe, DataError> {
        if pieces.is_empty() {
            return Err(DataError::new("BPE vocabulary is empty"));
        }
        let mut encoder = BTreeMap::new();
        for (id, piece) in pieces.iter().enumerate() {
            if piece.is_empty() {
                return Err(DataError::new(format!("empty BPE piece at id {id}")));
            }
            if encoder.insert(piece.clone(), id as u32).is_some() {
                return Err(DataError::new(format!("duplicate piece {piece:?}")));
            }
        }
        let mut merge_map = BTreeMap::new();
        for (rank, (left, right)) in merges.into_iter().enumerate() {
            let left_id = *encoder.get(&left).ok_or_else(|| {
                DataError::new(format!("merge left {left:?} is not in the vocabulary"))
            })?;
            let right_id = *encoder.get(&right).ok_or_else(|| {
                DataError::new(format!("merge right {right:?} is not in the vocabulary"))
            })?;
            let merged_text = format!("{left}{right}");
            let merged_id = *encoder.get(&merged_text).ok_or_else(|| {
                DataError::new(format!(
                    "merge {left:?}+{right:?} is {merged_text:?}, which is not in the vocabulary"
                ))
            })?;
            let rank_u = u32::try_from(rank)
                .map_err(|_| DataError::new(format!("merge rank {rank} exceeds u32")))?;
            if merge_map
                .insert((left_id, right_id), (rank_u, merged_id))
                .is_some()
            {
                return Err(DataError::new(format!(
                    "duplicate merge pair ({left_id}, {right_id})"
                )));
            }
        }
        Ok(Bpe {
            tokens: pieces,
            encoder,
            merges: merge_map,
        })
    }
}

/// GPT-2 pre-tokenizer pieces, in order.
///
/// Pattern: `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`.
/// The optional space is one U+0020. The classes are in [`crate::gpt2_class`].
pub fn gpt2_split(text: &str) -> Vec<&str> {
    const CONTRACTIONS: [&str; 7] = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"];
    let mut out = Vec::new();
    let mut i = 0;
    while i < text.len() {
        let rest = &text[i..];
        let mut end = 0;
        for c in CONTRACTIONS {
            if rest.starts_with(c) {
                end = i + c.len();
                break;
            }
        }
        if end <= i {
            if let Some(n) = opt_space_class(text, i, is_letter) {
                end = n;
            } else if let Some(n) = opt_space_class(text, i, is_number) {
                end = n;
            } else if let Some(n) = opt_space_class(text, i, is_other) {
                end = n;
            } else if let Some(n) = whitespace_token_end(text, i) {
                end = n;
            }
        }
        if end <= i {
            let Some(ch) = rest.chars().next() else {
                break;
            };
            end = i + ch.len_utf8();
        }
        out.push(&text[i..end]);
        i = end;
    }
    out
}

fn is_other(c: char) -> bool {
    !is_letter(c) && !is_number(c) && !is_space(c)
}

fn class_run(text: &str, i: usize, class: fn(char) -> bool) -> usize {
    let mut j = i;
    for (off, ch) in text[i..].char_indices() {
        if !class(ch) {
            break;
        }
        j = i + off + ch.len_utf8();
    }
    j
}

/// Optional single ASCII space, then one or more characters of `class`.
fn opt_space_class(text: &str, i: usize, class: fn(char) -> bool) -> Option<usize> {
    if text[i..].starts_with(' ') {
        let after = class_run(text, i + 1, class);
        if after > i + 1 {
            return Some(after);
        }
    }
    let end = class_run(text, i, class);
    (end > i).then_some(end)
}

/// `\s+(?!\S)` when that alternative matches, otherwise `\s+`.
fn whitespace_token_end(text: &str, i: usize) -> Option<usize> {
    let run_end = class_run(text, i, is_space);
    if run_end == i {
        return None;
    }
    let followed_by_non_ws = text[run_end..].chars().next().is_some_and(|c| !is_space(c));
    if !followed_by_non_ws {
        return Some(run_end);
    }
    let (off, _) = text[i..run_end].char_indices().next_back()?;
    let last = i + off;
    if last > i {
        Some(last)
    } else {
        Some(run_end)
    }
}

/// Load a Hugging Face GPT-2 `vocab.json` (string to id) and `merges.txt`.
///
/// The first line is skipped when it starts with `#version`. Later lines that
/// start with `#` are merges of the `#` piece. Rank is the remaining line order,
/// starting at 0.
pub fn load_hf_gpt2(vocab_json: &Path, merges_txt: &Path) -> Result<Bpe, DataError> {
    let vocab = std::fs::read_to_string(vocab_json)
        .map_err(|e| DataError::new(format!("{}: {e}", vocab_json.display())))?;
    let merges = std::fs::read_to_string(merges_txt)
        .map_err(|e| DataError::new(format!("{}: {e}", merges_txt.display())))?;
    let pieces = parse_vocab_json(&vocab)?;
    let pairs = parse_merges(&merges)?;
    Bpe::from_ranked_pieces(pieces, pairs)
}

fn parse_vocab_json(text: &str) -> Result<Vec<String>, DataError> {
    let bytes = text.as_bytes();
    let mut i = 0;
    if text.starts_with('\u{FEFF}') {
        i = '\u{FEFF}'.len_utf8();
    }
    skip_ws(bytes, &mut i);
    if bytes.get(i) != Some(&b'{') {
        return Err(DataError::new("vocab json must be an object"));
    }
    i += 1;
    let mut pairs = Vec::new();
    let mut after_comma = false;
    loop {
        skip_ws(bytes, &mut i);
        if bytes.get(i) == Some(&b'}') {
            if after_comma {
                return Err(DataError::new("trailing comma in vocab json"));
            }
            i += 1;
            break;
        }
        let key = parse_json_string(text, &mut i)?;
        skip_ws(bytes, &mut i);
        if bytes.get(i) != Some(&b':') {
            return Err(DataError::new("expected ':' after a vocab key"));
        }
        i += 1;
        skip_ws(bytes, &mut i);
        let id = parse_u32(text, &mut i)?;
        pairs.push((id, key));
        skip_ws(bytes, &mut i);
        match bytes.get(i) {
            Some(&b',') => {
                i += 1;
                after_comma = true;
            }
            Some(&b'}') => {
                i += 1;
                break;
            }
            _ => return Err(DataError::new("expected ',' or '}' in vocab json")),
        }
    }
    skip_ws(bytes, &mut i);
    if i != bytes.len() {
        return Err(DataError::new("trailing data after vocab json"));
    }
    if pairs.is_empty() {
        return Err(DataError::new("BPE vocabulary is empty"));
    }
    let n = pairs.len();
    let mut pieces: Vec<Option<String>> = vec![None; n];
    for (id, key) in pairs {
        if key.is_empty() {
            return Err(DataError::new(format!("empty BPE piece at id {id}")));
        }
        let Some(slot) = pieces.get_mut(id as usize) else {
            return Err(DataError::new(format!("token id {id} is outside 0..{n}")));
        };
        if slot.is_some() {
            return Err(DataError::new(format!("duplicate token id {id}")));
        }
        *slot = Some(key);
    }
    let mut out = Vec::with_capacity(n);
    for (id, piece) in pieces.into_iter().enumerate() {
        let Some(piece) = piece else {
            return Err(DataError::new(format!(
                "token ids must be 0..n without holes, missing {id}"
            )));
        };
        out.push(piece);
    }
    Ok(out)
}

fn parse_merges(text: &str) -> Result<Vec<(String, String)>, DataError> {
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        if n == 0 && line.starts_with("#version") {
            continue;
        }
        if line.is_empty() {
            continue;
        }
        let Some((left, right)) = line.split_once(' ') else {
            return Err(DataError::new(format!(
                "merge line {} has no space: {line:?}",
                n + 1
            )));
        };
        if left.is_empty() || right.is_empty() || right.contains(' ') {
            return Err(DataError::new(format!(
                "merge line {} is not two pieces: {line:?}",
                n + 1
            )));
        }
        out.push((left.to_string(), right.to_string()));
    }
    Ok(out)
}

fn skip_ws(bytes: &[u8], i: &mut usize) {
    while bytes
        .get(*i)
        .is_some_and(|c| matches!(c, b' ' | b'\t' | b'\n' | b'\r'))
    {
        *i += 1;
    }
}

fn parse_u32(text: &str, i: &mut usize) -> Result<u32, DataError> {
    let bytes = text.as_bytes();
    let start = *i;
    if start >= bytes.len() || !bytes[start].is_ascii_digit() {
        return Err(DataError::new("expected an integer token id"));
    }
    while bytes.get(*i).is_some_and(|c| c.is_ascii_digit()) {
        *i += 1;
    }
    let digits = &text[start..*i];
    digits
        .parse::<u32>()
        .map_err(|_| DataError::new(format!("token id {digits} does not fit in u32")))
}

fn parse_json_string(text: &str, i: &mut usize) -> Result<String, DataError> {
    let bytes = text.as_bytes();
    if bytes.get(*i) != Some(&b'"') {
        return Err(DataError::new("expected a JSON string"));
    }
    *i += 1;
    let mut out = Vec::new();
    while *i < bytes.len() {
        let c = bytes[*i];
        if c == b'"' {
            *i += 1;
            return String::from_utf8(out).map_err(|_| DataError::new("vocab key is not UTF-8"));
        }
        if c < 0x20 {
            return Err(DataError::new("raw control character in a vocab key"));
        }
        if c == b'\\' {
            *i += 1;
            let Some(esc) = bytes.get(*i).copied() else {
                return Err(DataError::new("truncated escape in a vocab key"));
            };
            *i += 1;
            match esc {
                b'"' | b'\\' | b'/' => out.push(esc),
                b'b' => out.push(0x08),
                b'f' => out.push(0x0C),
                b'n' => out.push(b'\n'),
                b'r' => out.push(b'\r'),
                b't' => out.push(b'\t'),
                b'u' => {
                    if *i + 4 > bytes.len() {
                        return Err(DataError::new("truncated unicode escape in a vocab key"));
                    }
                    let hex = std::str::from_utf8(&bytes[*i..*i + 4])
                        .map_err(|_| DataError::new("unicode escape is not ASCII"))?;
                    *i += 4;
                    let cp = u32::from_str_radix(hex, 16).map_err(|_| {
                        DataError::new(format!("unicode escape \\u{hex} is not hex"))
                    })?;
                    if (0xD800..=0xDFFF).contains(&cp) {
                        return Err(DataError::new("surrogate unicode escape in a vocab key"));
                    }
                    let Some(ch) = char::from_u32(cp) else {
                        return Err(DataError::new(format!("invalid codepoint U+{cp:04X}")));
                    };
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
                _ => return Err(DataError::new(format!("unknown JSON escape \\{esc}"))),
            }
        } else {
            out.push(c);
            *i += 1;
        }
    }
    Err(DataError::new("unterminated vocab key"))
}

/// Custom vocabulary used by tests. Ranks are local to this fixture.
///
/// Pieces: `a=0`, `b=1`, `c=2`, `ab=3` (rank 0), `abc=4` (rank 1), `bc=5` (rank 2).
pub fn fixture_bpe() -> Bpe {
    BpeBuilder::new()
        .token(0, "a")
        .and_then(|b| b.token(1, "b"))
        .and_then(|b| b.token(2, "c"))
        .and_then(|b| b.token(3, "ab"))
        .and_then(|b| b.token(4, "abc"))
        .and_then(|b| b.token(5, "bc"))
        .and_then(|b| b.merge(0, 1, 0, 3))
        .and_then(|b| b.merge(3, 2, 1, 4))
        .and_then(|b| b.merge(1, 2, 2, 5))
        .expect("fixture vocabulary is consistent")
        .build()
        .expect("fixture vocabulary builds")
}

/// GPT-2 `bytes_to_unicode` map. Index `b` holds the Unicode scalar for byte `b`.
///
/// This table is the published byte alphabet. It does not assign token ranks.
pub fn bytes_to_unicode() -> [char; 256] {
    let mut bs: Vec<u32> = Vec::new();
    bs.extend(u32::from(b'!')..=u32::from(b'~'));
    bs.extend(0xA1u32..=0xACu32);
    bs.extend(0xAEu32..=0xFFu32);
    let mut cs = bs.clone();
    let mut n = 0u32;
    for b in 0u32..256 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut out = ['\0'; 256];
    for (b, cp) in bs.into_iter().zip(cs) {
        out[b as usize] = char::from_u32(cp).unwrap_or('\u{FFFD}');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::CounterRng;

    #[test]
    fn fixture_round_trips_and_is_not_a_tiktoken_claim() {
        assert_eq!(TIKTOKEN_GPT2_BYTE_IDENTITY, "verified-20-strings");
        let bpe = fixture_bpe();
        for text in ["", "a", "ab", "abc", "bc", "cba", "aa"] {
            let ids = bpe.encode(text).unwrap();
            assert_eq!(bpe.decode(&ids).unwrap(), text, "{text}");
        }
        assert_eq!(bpe.encode("abc").unwrap(), vec![4]);
        assert_eq!(bpe.encode("cba").unwrap(), vec![2, 1, 0]);
        assert!(bpe.encode("z").is_err());
        assert!(bpe.decode(&[99]).is_err());
    }

    /// The rescan-everything merge loop: lowest rank, leftmost on ties.
    fn naive_encode(bpe: &Bpe, text: &str) -> Option<Vec<u32>> {
        let mut seq: Vec<u32> = text
            .chars()
            .map(|c| bpe.encoder.get(&c.to_string()).copied())
            .collect::<Option<_>>()?;
        loop {
            let mut best: Option<(u32, usize)> = None;
            for i in 0..seq.len().saturating_sub(1) {
                if let Some(&(rank, _)) = bpe.merges.get(&(seq[i], seq[i + 1])) {
                    if best.is_none_or(|(b, _)| rank < b) {
                        best = Some((rank, i));
                    }
                }
            }
            let Some((_, i)) = best else {
                return Some(seq);
            };
            seq[i] = bpe.merges[&(seq[i], seq[i + 1])].1;
            seq.remove(i + 1);
        }
    }

    /// Random vocabulary over `alphabet`, with merges of existing pieces and
    /// ranks drawn from a small range so equal ranks occur.
    fn random_bpe(rng: &mut CounterRng, alphabet: &[char], merges: usize) -> Bpe {
        let mut pieces: Vec<String> = alphabet.iter().map(|c| c.to_string()).collect();
        let mut b = BpeBuilder::new();
        for (i, p) in pieces.iter().enumerate() {
            b = b.token(i as u32, p).unwrap();
        }
        let mut pairs = std::collections::BTreeSet::new();
        for _ in 0..merges {
            let l = (rng.next_u64() % pieces.len() as u64) as u32;
            let r = (rng.next_u64() % pieces.len() as u64) as u32;
            let text = format!("{}{}", pieces[l as usize], pieces[r as usize]);
            if text.chars().count() > 6 || pieces.contains(&text) || !pairs.insert((l, r)) {
                continue;
            }
            let id = pieces.len() as u32;
            b = b.token(id, &text).unwrap();
            let rank = (rng.next_u64() % 8) as u32;
            b = b.merge(l, r, rank, id).unwrap();
            pieces.push(text);
        }
        b.build().unwrap()
    }

    #[test]
    fn heap_encoder_matches_the_naive_merge_loop() {
        let mut rng = CounterRng::new(0xB9E);
        let alphabet = ['a', 'b', 'c', 'é', '字'];
        let mut checked = 0;
        for _ in 0..60 {
            let bpe = random_bpe(&mut rng, &alphabet, 40);
            for _ in 0..50 {
                let len = (rng.next_u64() % 40) as usize;
                let text: String = (0..len)
                    .map(|_| alphabet[(rng.next_u64() % alphabet.len() as u64) as usize])
                    .collect();
                let got = bpe.encode(&text).unwrap();
                assert_eq!(Some(got.clone()), naive_encode(&bpe, &text), "{text:?}");
                assert_eq!(bpe.decode(&got).unwrap(), text);
                checked += 1;
            }
        }
        assert_eq!(checked, 3000);
    }

    #[test]
    fn hostile_text_and_ids_return_err_without_panicking() {
        let bpe = fixture_bpe();
        let mut rng = CounterRng::new(7);
        let soup = [
            'a',
            'b',
            'c',
            'z',
            '\0',
            '\u{FFFD}',
            '\u{10FFFF}',
            ' ',
            '\u{301}',
        ];
        for _ in 0..2000 {
            let len = (rng.next_u64() % 24) as usize;
            let text: String = (0..len)
                .map(|_| soup[(rng.next_u64() % soup.len() as u64) as usize])
                .collect();
            let ids: Vec<u32> = (0..len)
                .map(|_| match rng.next_u64() % 3 {
                    0 => (rng.next_u64() % 6) as u32,
                    1 => u32::MAX,
                    _ => rng.next_u64() as u32,
                })
                .collect();
            let caught = std::panic::catch_unwind(|| (bpe.encode(&text), bpe.decode(&ids)));
            let (enc, dec) = caught.unwrap_or_else(|_| panic!("panicked on {text:?} / {ids:?}"));
            let known = text.chars().all(|c| "abc".contains(c));
            assert_eq!(enc.is_ok(), known, "{text:?}");
            assert_eq!(dec.is_ok(), ids.iter().all(|&i| i < 6), "{ids:?}");
        }
        assert!(BpeBuilder::new().build().is_err());
        assert!(BpeBuilder::new().token(0, "").is_err());
        assert!(BpeBuilder::new().token(1, "a").unwrap().build().is_err());
        let merge_to_missing = BpeBuilder::new()
            .token(0, "a")
            .and_then(|b| b.merge(0, 0, 0, 9))
            .and_then(BpeBuilder::build);
        assert!(merge_to_missing.is_err());
    }

    #[test]
    fn long_input_encodes_in_near_linear_time() {
        let bpe = fixture_bpe();
        let text = "ab".repeat(20_000);
        let start = std::time::Instant::now();
        let ids = bpe.encode(&text).unwrap();
        let took = start.elapsed();
        assert_eq!(ids, vec![3; 20_000]);
        assert!(took.as_secs_f64() < 2.0, "40k chars took {took:?}");
    }

    #[test]
    fn gpt2_byte_alphabet_maps_space_and_is_bijective() {
        let map = bytes_to_unicode();
        assert_eq!(map[0x00], '\u{0100}');
        assert_eq!(map[0x20], '\u{0120}');
        assert_eq!(map[0x21], '!');
        assert_eq!(map[0x7E], '~');
        let mut uniq = map.to_vec();
        uniq.sort_unstable();
        uniq.dedup();
        assert_eq!(uniq.len(), 256);
    }

    #[test]
    fn gpt2_pretokenizer_splits_the_twenty_strings() {
        let cases: &[(&str, &[&str])] = &[
            ("", &[]),
            (" ", &[" "]),
            ("\n", &["\n"]),
            ("hello", &["hello"]),
            ("hello world", &["hello", " world"]),
            ("don't", &["don", "'t"]),
            ("123", &["123"]),
            ("a\nb", &["a", "\n", "b"]),
            ("  trailing  ", &[" ", " trailing", "  "]),
            ("café", &["café"]),
            ("你好", &["你好"]),
            ("🙂", &["🙂"]),
            ("Hello, world!", &["Hello", ",", " world", "!"]),
            ("\t\t", &["\t\t"]),
            ("'s", &["'s"]),
            ("end.", &["end", "."]),
            ("a  b", &["a", " ", " b"]),
            ("line\r\n", &["line", "\r\n"]),
            ("  ", &["  "]),
            ("I\u{2019}m", &["I", "\u{2019}", "m"]),
        ];
        assert_eq!(cases.len(), 20);
        for (text, parts) in cases {
            assert_eq!(gpt2_split(text).as_slice(), *parts, "{text:?}");
        }
        assert_eq!(gpt2_split(" 12").as_slice(), [" 12"]);
        assert_eq!(gpt2_split("  12").as_slice(), [" ", " 12"]);
        assert_eq!(gpt2_split("a \nb").as_slice(), ["a", " ", "\n", "b"]);
        for cp in [0u32, 0x20, 0x41, 0x2019, 0x4F60, 0x1F642, 0xA0, 0x3000] {
            let s = char::from_u32(cp).unwrap().to_string();
            assert_eq!(gpt2_split(&s).as_slice(), [s.as_str()]);
        }
    }

    fn tmp_dir(tag: &str) -> std::path::PathBuf {
        let n = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("ojas-bpe-{tag}-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    struct RmDir(std::path::PathBuf);
    impl Drop for RmDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn hf_loader_keeps_hash_merges_and_unicode_escapes() {
        let dir = RmDir(tmp_dir("hash"));
        let vocab = dir.0.join("vocab.json");
        let merges = dir.0.join("merges.txt");
        std::fs::write(&vocab, "{\"#\":0,\"##\":1}").unwrap();
        std::fs::write(&merges, "#version: 0.2\n# #\n").unwrap();
        let bpe = load_hf_gpt2(&vocab, &merges).unwrap();
        assert_eq!(bpe.encode_ordinary("###").unwrap(), vec![1, 0]);
        assert_eq!(bpe.decode_ordinary(&[1, 0]).unwrap(), "###");

        std::fs::write(&vocab, r#"{"\u0120":0,"t":1,"\u0120t":2}"#).unwrap();
        std::fs::write(&merges, "#version: 0.2\n\u{0120} t\n").unwrap();
        let bpe = load_hf_gpt2(&vocab, &merges).unwrap();
        assert_eq!(bpe.encode_ordinary(" t").unwrap(), vec![2]);
        assert_eq!(bpe.decode_ordinary(&[2]).unwrap(), " t");

        std::fs::write(&vocab, r#"{"\"":0}"#).unwrap();
        std::fs::write(&merges, "#version: 0.2\n").unwrap();
        let bpe = load_hf_gpt2(&vocab, &merges).unwrap();
        assert_eq!(bpe.decode(&[0]).unwrap(), "\"");

        std::fs::write(&vocab, r#"{"a":0,}"#).unwrap();
        assert!(load_hf_gpt2(&vocab, &merges).is_err());
        std::fs::write(&vocab, "{}").unwrap();
        assert!(load_hf_gpt2(&vocab, &merges).is_err());
        assert!(load_hf_gpt2(std::path::Path::new("/no/such/ojas-vocab.json"), &merges).is_err());
    }

    /// Ids that match tiktoken 0.12.0 `encode_ordinary` on the gpt2 pattern.
    fn twenty_gpt2_ids() -> [(&'static str, &'static [u32]); 20] {
        [
            ("", &[]),
            (" ", &[220]),
            ("\n", &[198]),
            ("hello", &[31373]),
            ("hello world", &[31373, 995]),
            ("don't", &[9099, 470]),
            ("123", &[10163]),
            ("a\nb", &[64, 198, 65]),
            ("  trailing  ", &[220, 25462, 220, 220]),
            ("café", &[66, 1878, 2634]),
            ("你好", &[19526, 254, 25001, 121]),
            ("🙂", &[8582, 25081]),
            ("Hello, world!", &[15496, 11, 995, 0]),
            ("\t\t", &[197, 197]),
            ("'s", &[338]),
            ("end.", &[437, 13]),
            ("a  b", &[64, 220, 275]),
            ("line\r\n", &[1370, 201, 198]),
            ("  ", &[220, 220]),
            ("I\u{2019}m", &[40, 447, 247, 76]),
        ]
    }

    fn rank_table_dir() -> Option<std::path::PathBuf> {
        let candidates = [
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata/gpt2"),
            std::path::PathBuf::from(
                "/Users/bharath/Code/research/MLSystemsLab/Step-Audio-EditX/funasr_detach/models/whisper/utils/assets/gpt2",
            ),
        ];
        candidates
            .into_iter()
            .find(|dir| dir.join("vocab.json").is_file() && dir.join("merges.txt").is_file())
    }

    #[test]
    fn verified_20_strings_match_tiktoken_encode_ordinary() {
        assert_eq!(TIKTOKEN_GPT2_BYTE_IDENTITY, "verified-20-strings");
        let Some(dir) = rank_table_dir() else {
            return;
        };
        let bpe = load_hf_gpt2(&dir.join("vocab.json"), &dir.join("merges.txt"))
            .unwrap_or_else(|e| panic!("rank table at {} failed: {e}", dir.display()));
        let piece = |id: u32, text: &str| bpe.decode(&[id]).ok().as_deref() == Some(text);
        let gpt2 = bpe.tokens.len() == 50257
            && bpe.merges.len() == 50_000
            && piece(0, "!")
            && piece(1, "\"")
            && piece(59, "\\")
            && piece(31373, "hello")
            && piece(50256, "<|endoftext|>");
        if bpe.tokens.len() == 50257 {
            assert!(
                gpt2,
                "50257-piece table did not match the GPT-2 fingerprint at {}",
                dir.display()
            );
        } else {
            for (text, _) in twenty_gpt2_ids() {
                if let Ok(ids) = bpe.encode_ordinary(text) {
                    assert_eq!(bpe.decode_ordinary(&ids).unwrap(), text);
                }
            }
            return;
        }
        assert_eq!(bpe.encode_ordinary(" t").unwrap(), vec![256]);
        assert_eq!(bpe.encode_ordinary("###").unwrap(), vec![21017]);
        let special = bpe.encode_ordinary("<|endoftext|>").unwrap();
        assert_ne!(special, vec![50256]);
        let rows = twenty_gpt2_ids();
        assert_eq!(rows.len(), 20);
        for (text, ids) in rows {
            let got = bpe.encode_ordinary(text).unwrap();
            assert_eq!(got, ids, "{text:?}");
            assert_eq!(bpe.decode_ordinary(&got).unwrap(), text);
        }
    }
}

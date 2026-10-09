//! TOKENIZER and TOKENIZE: the GPT-2 byte-level BPE of `ojas_data`, held by
//! the session.
//!
//! TOKENIZER loads a Hugging Face `vocab.json` and `merges.txt` from under
//! the root (each opened with `O_NOFOLLOW` and handed to
//! [`ojas_data::load_hf_gpt2`] by descriptor; that loader refuses a file over
//! [`ojas_data::HF_TEXT_CAP`] before reading it). TOKENIZE encodes text with
//! `encode_ordinary`, decodes ids with `decode_ordinary` (non-UTF-8 output is
//! an error) or `decode_ordinary_lossy` (each invalid sequence becomes
//! U+FFFD), or looks up the id of one vocabulary piece such as
//! `<|endoftext|>`. Encode refuses input text over the same cap. Both
//! decodes refuse more decoded bytes than that cap before allocating them;
//! the lossy one's U+FFFD replacements can then make its text longer.

use crate::gate::Check;
use crate::load::Verified;
use crate::session;
use crate::wire::{field, required, tag, Fields, Kind, Reader};

pub const TOKENIZE_ENCODE: u32 = 1;
pub const TOKENIZE_DECODE: u32 = 2;
pub const TOKENIZE_DECODE_LOSSY: u32 = 3;
pub const TOKENIZE_PIECE_ID: u32 = 4;

/// TOKENIZER: `id: u64`, then `{vocab_json, merges_txt}`. Replaces the
/// session's tokenizer only once the new one has loaded.
pub fn load_request(bytes: &[u8], mut check: Check) -> Result<(), String> {
    check()?;
    let mut r = Reader::new(bytes);
    let id = r.u64()?;
    let f = Fields::read(
        &mut r,
        &[
            field(tag::VOCAB_JSON, Kind::Str),
            field(tag::MERGES_TXT, Kind::Str),
        ],
    )?;
    r.finish()?;
    let session = session::require(id)?;
    let vocab = Verified::open(required(f.str(tag::VOCAB_JSON), tag::VOCAB_JSON)?)?;
    let merges = Verified::open(required(f.str(tag::MERGES_TXT), tag::MERGES_TXT)?)?;
    let mut state = session.lock_state()?;
    check()?;
    let bpe = ojas_data::load_hf_gpt2(&vocab.fd_path(), &merges.fd_path())
        .map_err(|e| format!("tokenizer: {}", merges.explain(&vocab.explain(e.detail()))))?;
    state.tokenizer = Some(bpe);
    Ok(())
}

/// TOKENIZE: `id: u64, mode: u32`, then UTF-8 text ([`TOKENIZE_ENCODE`],
/// returns `u32` ids), `u32` ids ([`TOKENIZE_DECODE`] and
/// [`TOKENIZE_DECODE_LOSSY`], return UTF-8), or one vocabulary piece as UTF-8
/// ([`TOKENIZE_PIECE_ID`], returns its `u32` id; a piece the vocabulary does
/// not hold is an error).
pub fn tokenize_request(bytes: &[u8], mut check: Check) -> Result<Vec<u8>, String> {
    check()?;
    let mut r = Reader::new(bytes);
    let id = r.u64()?;
    let mode = r.u32()?;
    let session = session::require(id)?;
    let state = session.lock_state()?;
    let bpe = state
        .tokenizer
        .as_ref()
        .ok_or_else(|| "tokenize: no tokenizer is loaded on this model".to_string())?;
    let show = |e: ojas_data::DataError| format!("tokenize: {}", e.detail());
    match mode {
        TOKENIZE_ENCODE => {
            let text = r.rest_str()?;
            let ids = bpe.encode_ordinary(text).map_err(show)?;
            Ok(ids.iter().flat_map(|id| id.to_le_bytes()).collect())
        }
        TOKENIZE_DECODE | TOKENIZE_DECODE_LOSSY => {
            let rest = r.rest();
            if !rest.len().is_multiple_of(4) {
                return Err("tokenize: shape: ids are not a whole number of u32".to_string());
            }
            let ids: Vec<u32> = rest
                .as_chunks::<4>()
                .0
                .iter()
                .copied()
                .map(u32::from_le_bytes)
                .collect();
            let text = if mode == TOKENIZE_DECODE {
                bpe.decode_ordinary(&ids)
            } else {
                bpe.decode_ordinary_lossy(&ids)
            };
            Ok(text.map_err(show)?.into_bytes())
        }
        TOKENIZE_PIECE_ID => {
            let piece = r.rest_str()?;
            let id = bpe.piece_id(piece).ok_or_else(|| {
                format!(
                    "tokenize: no vocabulary piece is the {}-byte text given",
                    piece.len()
                )
            })?;
            Ok(id.to_le_bytes().to_vec())
        }
        other => Err(format!("tokenize: shape: unknown mode {other}")),
    }
}

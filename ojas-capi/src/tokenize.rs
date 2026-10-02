//! TOKENIZER and TOKENIZE: the GPT-2 byte-level BPE of `ojas_data`, held by
//! the session.
//!
//! TOKENIZER loads a Hugging Face `vocab.json` and `merges.txt` from under
//! the root (each opened with `O_NOFOLLOW` and handed to
//! [`ojas_data::load_hf_gpt2`] by descriptor; that loader refuses a file over
//! [`ojas_data::HF_TEXT_CAP`] before reading it). TOKENIZE encodes text with
//! `encode_ordinary` or decodes ids with `decode_ordinary`; both refuse text
//! over the same cap.

use crate::gate::Check;
use crate::load::Verified;
use crate::session;
use crate::wire::{field, required, tag, Fields, Kind, Reader};

pub const TOKENIZE_ENCODE: u32 = 1;
pub const TOKENIZE_DECODE: u32 = 2;

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
/// returns `u32` ids) or `u32` ids ([`TOKENIZE_DECODE`], returns UTF-8).
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
        TOKENIZE_DECODE => {
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
            let text = bpe.decode_ordinary(&ids).map_err(show)?;
            Ok(text.into_bytes())
        }
        other => Err(format!("tokenize: shape: unknown mode {other}")),
    }
}

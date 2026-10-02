//! Next tokens.
//!
//! GENERATE mode [`GEN_LOGITS`] is [`ojas_infer::argmax_token`] of the
//! caller's host logits; it reads no model. A non-finite logit is an error
//! from that function, never token 0.
//!
//! SAMPLE runs the session's model: an [`ojas_infer::DeviceDecoder`] over
//! the session's current parameters (the trainer's, once one is open) on the
//! session's device, with a KV cache sized to the request, then
//! [`ojas_infer::DeviceDecoder::generate`] with temperature, top-k, top-p, a
//! seed, a token budget and stop tokens. Each forward reads back one
//! `[vocab]` logit row.

use ojas_core::{Backend, OjasError};
use ojas_infer::{argmax_token, DeviceDecoder, GenerateConfig, SamplingConfig};
use ojas_model::ModelParams;

use crate::gate::Check;
use crate::model::{on_model, Model};
use crate::session;
use crate::wire::{field, required, tag, Fields, Kind, Reader};

pub const GEN_LOGITS: u32 = 1;

/// Argmax of `logits`.
pub fn argmax(logits: &[f32]) -> Result<u32, String> {
    argmax_token(logits).map_err(|e| crate::ojas_error("generate", &e))
}

/// SAMPLE: `id: u64`, then `{temperature, top_k?, top_p?, seed,
/// max_new_tokens, stop?, prompt}`. Returns the new ids (a stop token, when
/// emitted, is the last).
pub fn sample_request(bytes: &[u8], mut check: Check) -> Result<Vec<u32>, String> {
    check()?;
    let mut r = Reader::new(bytes);
    let id = r.u64()?;
    let f = Fields::read(
        &mut r,
        &[
            field(tag::TEMPERATURE, Kind::F32),
            field(tag::TOP_K, Kind::U32),
            field(tag::TOP_P, Kind::F32),
            field(tag::SEED, Kind::U64),
            field(tag::MAX_NEW, Kind::U32),
            field(tag::STOP, Kind::U32s),
            field(tag::PROMPT, Kind::U32s),
        ],
    )?;
    r.finish()?;
    let to_usize =
        |v: u32| usize::try_from(v).map_err(|_| "sample: count exceeds usize".to_string());
    let cfg = GenerateConfig {
        sampling: SamplingConfig {
            temperature: required(f.f32(tag::TEMPERATURE), tag::TEMPERATURE)?,
            top_k: f.u32(tag::TOP_K).map(to_usize).transpose()?,
            top_p: f.f32(tag::TOP_P),
        },
        seed: required(f.u64(tag::SEED), tag::SEED)?,
        max_new_tokens: to_usize(required(f.u32(tag::MAX_NEW), tag::MAX_NEW)?)?,
        stop_tokens: f.u32s(tag::STOP).unwrap_or_default(),
    };
    let prompt = required(f.u32s(tag::PROMPT), tag::PROMPT)?;
    let session = session::require(id)?;
    let mut guard = session.lock_state()?;
    let state = &mut *guard;
    let armed = state.slot.arm(check);
    on_model!(&state.engine, m => {
        let ids = sample_on(m, &prompt, &cfg).map_err(|e| armed.report("sample", &e));
        crate::settled("sample", &m.backend, ids)
    })
}

fn sample_on<B: Backend + Clone>(
    m: &Model<B>,
    prompt: &[u32],
    cfg: &GenerateConfig,
) -> Result<Vec<u32>, OjasError> {
    cfg.sampling.validate()?;
    if prompt.is_empty() {
        return Err(OjasError::Shape {
            op: "sample",
            detail: "prompt is empty".into(),
        });
    }
    // The decoder forwards the prompt and every emitted token but the last.
    // A request past `max_seq` gets a cache of `max_seq`, and the decoder
    // refuses it as `CapacityExceeded` before any forward.
    let needed = prompt
        .len()
        .saturating_add(cfg.max_new_tokens.saturating_sub(1));
    let capacity = needed.clamp(1, m.spec.max_seq);
    let params = ModelParams::from_flat(&m.spec, m.params()?)?;
    let mut decoder = DeviceDecoder::new(m.backend.clone(), &m.spec, &params, capacity)?;
    decoder.generate(prompt, cfg)
}

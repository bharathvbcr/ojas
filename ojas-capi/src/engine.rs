//! Opcode handlers. Opcode 0 is not registered.

use gusset::JobContext;

#[cfg(test)]
use std::sync::{atomic::AtomicBool, Arc};

#[cfg(test)]
use gusset::CallHeader;

use crate::generate::{self, GenerateBody, GEN_GREEDY, GEN_LOGITS};
use crate::load;
use crate::session;
use crate::step::{self, StepInput, MODE_HEADER, MODE_LOGITS, MODE_TOKENS};

pub const OP_LOAD: u32 = 1;
pub const OP_STEP: u32 = 2;
pub const OP_GENERATE: u32 = 3;
pub const OP_FREE: u32 = 4;
/// Panics so the Go tests can observe a poisoned handle. Not a model op.
pub const OP_PANIC: u32 = 5;

pub fn install_engine() -> Result<(), String> {
    gusset::clear_engine_handlers();
    gusset::register_engine(OP_LOAD, dispatch)?;
    gusset::register_engine(OP_STEP, dispatch)?;
    gusset::register_engine(OP_GENERATE, dispatch)?;
    gusset::register_engine(OP_FREE, dispatch)?;
    gusset::register_engine(OP_PANIC, dispatch)?;
    Ok(())
}

pub fn dispatch(ctx: &JobContext, input: &[u8]) -> Result<Vec<u8>, String> {
    ctx.check()
        .map_err(|reason| format!("cancelled: {reason:?}"))?;
    match ctx.opcode() {
        OP_LOAD => op_load(input),
        OP_STEP => op_step(input),
        OP_GENERATE => op_generate(input),
        OP_FREE => op_free(input),
        OP_PANIC => panic!("ojas: induced panic"),
        other => Err(format!("unknown opcode {other}")),
    }
}

fn op_load(input: &[u8]) -> Result<Vec<u8>, String> {
    let text = std::str::from_utf8(input).map_err(|_| "load: path is not utf-8".to_string())?;
    let session = load::load_path(text)?;
    let mut out = Vec::with_capacity(12);
    out.extend_from_slice(&session.id.to_le_bytes());
    out.extend_from_slice(&session.tensors.to_le_bytes());
    Ok(out)
}

fn op_free(input: &[u8]) -> Result<Vec<u8>, String> {
    if input.len() != 8 {
        return Err("free: shape: payload must be one u64 id".to_string());
    }
    let mut cur = input;
    let id = take_u64(&mut cur)?;
    session::try_free(id)?;
    Ok(Vec::new())
}

fn op_step(input: &[u8]) -> Result<Vec<u8>, String> {
    let mut cur = input;
    let id = take_u64(&mut cur)?;
    let _session = session::require(id)?;
    let mode = take_u32(&mut cur)?;
    let batch = take_u32(&mut cur)?;
    let seq = take_u32(&mut cur)?;
    if mode == MODE_HEADER {
        if !cur.is_empty() {
            let _step = take_u32(&mut cur)?;
            if !cur.is_empty() {
                return Err("step: shape: step payload is missing logits".to_string());
            }
        }
        return Err("step: shape: step payload is missing logits".to_string());
    }
    let n_classes = take_u32(&mut cur)?;
    let step_index = take_u32(&mut cur)?;
    let lr = f32::from_le_bytes(take_array(&mut cur)?);
    let stats = match mode {
        MODE_LOGITS => {
            let rows = step::rows(batch, seq)?;
            let n_logits = rows
                .checked_mul(n_classes as usize)
                .ok_or_else(|| "step: shape: logits size overflows".to_string())?;
            let logits = take_vec(&mut cur, n_logits, f32::from_le_bytes)?;
            let targets = take_vec(&mut cur, rows, u32::from_le_bytes)?;
            if !cur.is_empty() {
                return Err("shape: trailing bytes".to_string());
            }
            step::step(StepInput {
                mode,
                batch,
                seq,
                n_classes,
                step: step_index,
                lr,
                logits: &logits,
                tokens: &[],
                targets_u32: &targets,
                targets_u16: &[],
            })?
        }
        MODE_TOKENS => {
            let rows = step::rows(batch, seq)?;
            let tokens = take_vec(&mut cur, rows, u16::from_le_bytes)?;
            let targets = take_vec(&mut cur, rows, u16::from_le_bytes)?;
            if !cur.is_empty() {
                return Err("shape: trailing bytes".to_string());
            }
            step::step(StepInput {
                mode,
                batch,
                seq,
                n_classes,
                step: step_index,
                lr,
                logits: &[],
                tokens: &tokens,
                targets_u32: &[],
                targets_u16: &targets,
            })?
        }
        other => return Err(format!("step: shape: unknown mode {other}")),
    };
    let mut out = Vec::with_capacity(12);
    out.extend_from_slice(&stats.loss.to_le_bytes());
    out.extend_from_slice(&stats.grad_norm.to_le_bytes());
    out.extend_from_slice(&stats.lr.to_le_bytes());
    Ok(out)
}

fn op_generate(input: &[u8]) -> Result<Vec<u8>, String> {
    let mut cur = input;
    let id = take_u64(&mut cur)?;
    let _session = session::require(id)?;
    let mode = take_u32(&mut cur)?;
    let token = match mode {
        GEN_LOGITS => {
            let n = take_u32(&mut cur)? as usize;
            let logits = take_vec(&mut cur, n, f32::from_le_bytes)?;
            if !cur.is_empty() {
                return Err("shape: trailing bytes".to_string());
            }
            generate::generate(
                mode,
                GenerateBody {
                    logits: &logits,
                    prompt: &[],
                },
            )?
        }
        GEN_GREEDY => {
            let n = take_u32(&mut cur)? as usize;
            let prompt = take_vec(&mut cur, n, u32::from_le_bytes)?;
            if !cur.is_empty() {
                return Err("shape: trailing bytes".to_string());
            }
            generate::generate(
                mode,
                GenerateBody {
                    logits: &[],
                    prompt: &prompt,
                },
            )?
        }
        other => return Err(format!("generate: shape: unknown mode {other}")),
    };
    Ok(token.to_le_bytes().to_vec())
}

fn take_u64(input: &mut &[u8]) -> Result<u64, String> {
    take_array(input).map(u64::from_le_bytes)
}

fn take_u32(input: &mut &[u8]) -> Result<u32, String> {
    take_array(input).map(u32::from_le_bytes)
}

fn take_array<const N: usize>(input: &mut &[u8]) -> Result<[u8; N], String> {
    if input.len() < N {
        return Err("shape: truncated payload".to_string());
    }
    let (head, tail) = input.split_at(N);
    *input = tail;
    let mut out = [0u8; N];
    out.copy_from_slice(head);
    Ok(out)
}

/// `n` little-endian values of `N` bytes each. The length is checked against
/// the payload before anything is allocated.
fn take_vec<T, const N: usize>(
    input: &mut &[u8],
    n: usize,
    decode: fn([u8; N]) -> T,
) -> Result<Vec<T>, String> {
    let bytes = n
        .checked_mul(N)
        .ok_or_else(|| "shape: payload size overflows".to_string())?;
    if input.len() < bytes {
        return Err("shape: truncated payload".to_string());
    }
    let (head, tail) = input.split_at(bytes);
    *input = tail;
    Ok(head
        .chunks_exact(N)
        .map(|chunk| {
            let mut b = [0u8; N];
            b.copy_from_slice(chunk);
            decode(b)
        })
        .collect())
}

/// Used by the cancel test to build a context without a live pool.
#[cfg(test)]
pub(crate) fn context_for(opcode: u32, cancelled: bool) -> JobContext {
    let header = CallHeader {
        reserved: opcode,
        ..CallHeader::default()
    };
    JobContext::new(header, Arc::new(AtomicBool::new(cancelled)))
}

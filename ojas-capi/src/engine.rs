//! Opcode handlers. Opcode 0 is not registered.
//!
//! | Op | Payload | Result |
//! | :--- | :--- | :--- |
//! | 1 LOAD | `{path, device?, threads?, budget?, numerics?}` | `id: u64, tensors: u32` |
//! | 2 | retired (the payload-only Step); not registered | |
//! | 3 GENERATE | `id: u64, mode: u32 = 1, n: u32, logits: [f32; n]` | `token: u32` |
//! | 4 FREE | `id: u64` | empty |
//! | 5 PANIC | anything | (panics) |
//! | 6 NEW | spec fields, `seed`, placement fields | `id: u64, tensors: u32` |
//! | 7 TRAIN_OPEN | `id: u64`, train fields | empty |
//! | 8 TRAIN_STEP | `id: u64, mode: u32`, caller tokens for mode 1 | [`crate::train::StepOut`], 40 bytes |
//! | 9 SAVE | `id: u64`, directory | empty |
//! | 10 RESUME | `{path}`, placement fields, train fields | `id: u64, tensors: u32` |
//! | 11 TOKENIZER | `id: u64, {vocab_json, merges_txt}` | empty |
//! | 12 TOKENIZE | `id: u64, mode: u32`, text (1 encode, 4 piece id) or ids (2 decode, 3 lossy decode) | ids, text, or one id |
//! | 13 SAMPLE | `id: u64, {temperature, top_k?, top_p?, seed, max_new_tokens, stop?, prompt}` | `[u32]` |
//! | 14 INSPECT | relative path | `tensors: u32` |
//! | 15 SET_MEMORY_CEILING | `bytes: u64` | empty |
//! | 16 SYSTEM_PROFILE | empty, `budget: u64`, or `budget: u64, flags: u32` | [`crate::profile`] record |
//!
//! `{...}` is a [`crate::wire::Fields`] option record. Each handler polls the
//! job's cancel flag before it starts, and a model op polls it before every
//! device op that cannot commit (`crate::gate`).

use gusset::{JobContext, JobOutput};

#[cfg(test)]
use std::sync::{atomic::AtomicBool, Arc};

#[cfg(test)]
use gusset::CallHeader;

use std::sync::Mutex;

use crate::gate::Check;
use crate::generate::{self, GEN_LOGITS};
use crate::{load, session, tokenize, train};

pub const OP_LOAD: u32 = 1;
/// Retired with the payload-only Step that computed a loss of the caller's
/// logits without a model. Not registered; the number is not reused.
pub const OP_STEP_RETIRED: u32 = 2;
pub const OP_GENERATE: u32 = 3;
pub const OP_FREE: u32 = 4;
/// Panics so the Go tests can observe a poisoned handle. Not a model op.
///
/// This opcode stays in the default engine. `TestPoisonDropsSession` reaches
/// it through the staticlib (`cargo build -p ojas-gusset-engine`), which is
/// not built with `cfg(test)`. A `panic-hook` feature that is off by default
/// would leave opcode 5 unregistered there, and `poisonHandle` could not
/// poison the handle. Gating it would break that test, so it stays.
pub const OP_PANIC: u32 = 5;
pub const OP_NEW: u32 = 6;
pub const OP_TRAIN_OPEN: u32 = 7;
pub const OP_TRAIN_STEP: u32 = 8;
pub const OP_SAVE: u32 = 9;
pub const OP_RESUME: u32 = 10;
pub const OP_TOKENIZER: u32 = 11;
pub const OP_TOKENIZE: u32 = 12;
pub const OP_SAMPLE: u32 = 13;
pub const OP_INSPECT: u32 = 14;
/// Replace the process-wide memory ceiling every session budget draws from
/// ([`session::set_memory_ceiling`]). Refused for 0 and while any model is
/// open.
pub const OP_SET_MEMORY_CEILING: u32 = 15;
/// Read the host profile and the plan for a caller budget
/// ([`crate::profile::profile_request`]). Reads only.
pub const OP_SYSTEM_PROFILE: u32 = 16;

const OPCODES: [u32; 15] = [
    OP_LOAD,
    OP_GENERATE,
    OP_FREE,
    OP_PANIC,
    OP_NEW,
    OP_TRAIN_OPEN,
    OP_TRAIN_STEP,
    OP_SAVE,
    OP_RESUME,
    OP_TOKENIZER,
    OP_TOKENIZE,
    OP_SAMPLE,
    OP_INSPECT,
    OP_SET_MEMORY_CEILING,
    OP_SYSTEM_PROFILE,
];

pub fn install_engine() -> Result<(), String> {
    static INSTALLED: Mutex<bool> = Mutex::new(false);
    let mut installed = INSTALLED
        .lock()
        .map_err(|_| "engine install lock poisoned".to_string())?;
    if *installed {
        return Ok(());
    }
    gusset::clear_engine_handlers();
    for op in OPCODES {
        gusset::register_engine(op, dispatch)?;
    }
    *installed = true;
    Ok(())
}

/// Errors cross unchanged. A kind prefix, when there is one, was added where
/// the typed error was produced ([`crate::ojas_error`], [`crate::kinded`]).
pub fn dispatch(ctx: &JobContext, input: &[u8]) -> Result<JobOutput, String> {
    dispatch_bytes(ctx, input).map(stage)
}

/// The job's cancel check, owned so a model op can hold it for the call.
/// The message is gusset's own (`"cancelled: Explicit"`), which its Go side
/// maps to `context.Canceled`.
pub(crate) fn cancel_check(ctx: &JobContext) -> Check {
    let ctx = ctx.clone();
    Box::new(move || {
        ctx.check()
            .map_err(|reason| format!("cancelled: {reason:?}"))
    })
}

fn session_result(session: session::Session) -> Vec<u8> {
    let mut out = Vec::with_capacity(12);
    out.extend_from_slice(&session.id.to_le_bytes());
    out.extend_from_slice(&session.tensors.to_le_bytes());
    out
}

/// Calls that allocate model-sized memory. Under critical memory pressure
/// they are refused before any work; FREE, SAVE and the queries still run,
/// so a host can always release memory and checkpoint.
const ALLOCATING_OPS: [u32; 7] = [
    OP_LOAD,
    OP_NEW,
    OP_TRAIN_OPEN,
    OP_TRAIN_STEP,
    OP_RESUME,
    OP_SAMPLE,
    OP_GENERATE,
];

/// Refuse `opcode` when it allocates and the kernel reports critical
/// pressure: a run started then is among the next the OS kills, and a
/// training step's partial work is lost with the process. The trainer and
/// every session are left as they were.
pub(crate) fn admit(opcode: u32, pressure: ojas_device::MemoryPressure) -> Result<(), String> {
    if pressure == ojas_device::MemoryPressure::Critical && ALLOCATING_OPS.contains(&opcode) {
        return Err(crate::kinded(
            crate::ErrorKind::Pressure,
            format!(
                "memory pressure: opcode {opcode} refused: the system reports critical memory \
                 pressure; retry once it eases (Save and Free still run)"
            ),
        ));
    }
    Ok(())
}

fn dispatch_bytes(ctx: &JobContext, input: &[u8]) -> Result<Vec<u8>, String> {
    ctx.check()
        .map_err(|reason| format!("cancelled: {reason:?}"))?;
    if ALLOCATING_OPS.contains(&ctx.opcode()) {
        admit(ctx.opcode(), ojas_device::probe_pressure())?;
    }
    let check = cancel_check(ctx);
    match ctx.opcode() {
        OP_LOAD => load::load_request(input, check).map(session_result),
        OP_GENERATE => op_generate(input),
        OP_FREE => op_free(input),
        OP_PANIC => panic!("ojas: induced panic"),
        OP_NEW => load::new_request(input, check).map(session_result),
        OP_TRAIN_OPEN => train::open_request(input, check).map(|()| Vec::new()),
        OP_TRAIN_STEP => train::step_request(input, check).map(train::StepOut::to_bytes),
        OP_SAVE => train::save_request(input, check).map(|()| Vec::new()),
        OP_RESUME => train::resume_request(input, check).map(session_result),
        OP_TOKENIZER => tokenize::load_request(input, check).map(|()| Vec::new()),
        OP_TOKENIZE => tokenize::tokenize_request(input, check),
        OP_SAMPLE => generate::sample_request(input, check)
            .map(|ids| ids.iter().flat_map(|id| id.to_le_bytes()).collect()),
        OP_INSPECT => {
            let path = std::str::from_utf8(input).map_err(|_| "inspect: path is not utf-8")?;
            load::inspect(path).map(|n| n.to_le_bytes().to_vec())
        }
        OP_SET_MEMORY_CEILING => op_set_memory_ceiling(input),
        OP_SYSTEM_PROFILE => crate::profile::profile_request(input, check),
        other => Err(format!("unknown opcode {other}")),
    }
}

/// Results under 4 KiB stay `JobOutput::Bytes`, which is the path that
/// exists before Rust 1.100. At or above that size, and only when gusset's
/// own probe set `DEP_GUSSET_ALLOCATOR_API=1`, the bytes are copied into
/// `BufferAlloc` and returned as `JobOutput::Allocated`.
pub fn stage(bytes: Vec<u8>) -> JobOutput {
    #[cfg(gusset_allocator_api)]
    {
        if bytes.len() >= 4096 {
            let mut out: Vec<u8, gusset::BufferAlloc> = Vec::new_in(gusset::BufferAlloc);
            if out.try_reserve_exact(bytes.len()).is_ok() {
                out.extend_from_slice(&bytes);
                return JobOutput::from(out);
            }
        }
    }
    JobOutput::from(bytes)
}

#[cfg(test)]
pub fn bytes_of(out: JobOutput) -> Vec<u8> {
    match out {
        JobOutput::Bytes(bytes) => bytes,
        #[cfg(gusset_allocator_api)]
        JobOutput::Allocated(bytes) => bytes.to_vec(),
        _ => Vec::new(),
    }
}

fn op_free(input: &[u8]) -> Result<Vec<u8>, String> {
    if input.len() != 8 {
        return Err("free: shape: payload must be one u64 id".to_string());
    }
    let id = crate::wire::Reader::new(input).u64()?;
    session::try_free(id)?;
    Ok(Vec::new())
}

fn op_set_memory_ceiling(input: &[u8]) -> Result<Vec<u8>, String> {
    if input.len() != 8 {
        return Err("memory ceiling: shape: payload must be one u64 byte count".to_string());
    }
    let bytes = crate::wire::Reader::new(input).u64()?;
    session::set_memory_ceiling(bytes)?;
    Ok(Vec::new())
}

/// `id: u64, mode: u32, n: u32, logits: [f32; n]`. Mode [`GEN_LOGITS`] is
/// the only mode; the model is not read, but the id must be live.
fn op_generate(input: &[u8]) -> Result<Vec<u8>, String> {
    let mut r = crate::wire::Reader::new(input);
    let id = r.u64()?;
    session::require(id)?;
    match r.u32()? {
        GEN_LOGITS => {
            let n = usize::try_from(r.u32()?).map_err(|_| "generate: shape: n".to_string())?;
            let logits = r.values(n, f32::from_le_bytes)?;
            r.finish()?;
            generate::argmax(&logits).map(|t| t.to_le_bytes().to_vec())
        }
        other => Err(format!("generate: shape: unknown mode {other}")),
    }
}

/// Used by the cancel test to build a context without a live pool.
#[cfg(test)]
pub(crate) fn context_for(opcode: u32, cancelled: bool) -> JobContext {
    context_with(opcode, Arc::new(AtomicBool::new(cancelled)))
}

/// A context whose cancel flag the test keeps.
#[cfg(test)]
pub(crate) fn context_with(opcode: u32, flag: Arc<AtomicBool>) -> JobContext {
    let header = CallHeader {
        reserved: opcode,
        ..CallHeader::default()
    };
    JobContext::new(header, flag)
}

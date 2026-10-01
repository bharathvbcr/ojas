//! In-process load, step, generate, and free for the Go API.
//!
//! The numeric step calls [`ojas_cpu::CpuBackend`]. Generate calls
//! [`ojas_infer::argmax_token`], directly or after a one-layer linear of
//! [`ojas_infer::embed`]; see the `generate` module.
//! This crate does not launch Metal.

#![forbid(unsafe_code)]

mod engine;
mod generate;
mod load;
mod owner;
mod session;
mod step;

pub use engine::{dispatch, install_engine, OP_FREE, OP_GENERATE, OP_LOAD, OP_PANIC, OP_STEP};
pub use generate::{generate, GEN_GREEDY, GEN_LOGITS};
pub use load::{resolve_under_root, INLINE_PATH_MAX};
pub use session::{
    clear_sessions, last_error, load_model, reset_sessions, session_count, set_last_error,
    set_model_root, try_free, Session,
};
pub use step::{step, StepStats, MODE_HEADER, MODE_LOGITS, MODE_TOKENS};

pub const SESSION_CAP: usize = session::SESSION_CAP;

#[cfg(test)]
mod tests;

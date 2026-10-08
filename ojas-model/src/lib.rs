//! The nanolab GPT, written once (`docs/framework-design.md` §1-§3).
//!
//! - [`ModelSpec`]: the shape, with [`ModelSpec::nanolab_124m`] and the CI
//!   [`ModelSpec::tiny`].
//! - [`param_table`]: nanolab `state_dict` names, shapes, init rule and
//!   optimizer group; [`ModelParams`] is the typed view of that table.
//! - [`init_params`]: order-independent init, one `CounterRng` per name.
//!   [`load_model`] reads a safetensors file with the same names and its
//!   `ojas.spec` metadata.
//! - [`Graph`]: one op vocabulary with two executors, `ojas_autograd::Tape`
//!   (training) and [`Eval`] (eager). [`block`] is the nanolab block over
//!   any `Graph`; [`forward_loss`] ends in the fused tied-head CE, with
//!   each block optionally a checkpointed segment ([`ActivationCheckpoint`]).
//! - [`qwen35`]: the Qwen3.5 hybrid text tower (gated delta net and gated
//!   attention layers) over the same [`Graph`], loaded from a Hugging Face
//!   checkpoint.
//! - [`Trainer`]: the §3 step on any `Backend`; [`Trainer::save`] and
//!   [`Trainer::resume_from`] for the §4 checkpoint directory.

#![forbid(unsafe_code)]

mod block;
mod checkpoint;
mod graph;
mod init;
mod json;
mod load;
mod names;
pub mod qwen35;
mod spec;
mod trainer;

pub use block::{
    bind, block, block_with, causal_attention, forward_hidden, forward_logits, forward_loss,
    ActivationCheckpoint, BlockOut, Rope,
};
pub use checkpoint::{
    MODEL_FILE, OPTIM_FILE, RUN_METADATA_KEY, STATE_FILE, STATE_FORMAT, STEP_METADATA_KEY,
};
pub use graph::{Eval, Graph};
pub use init::{fnv1a, init_params, init_values};
pub use load::{load_model, load_params, load_spec, COMPILED_PREFIX, LM_HEAD};
pub use names::{
    param_bytes, param_count, param_table, BlockParams, Init, ModelParams, ParamInfo, BLOCK_PARAMS,
    INIT_STD,
};
pub use spec::{
    swiglu_hidden, ModelSpec, MAX_LAYERS, SPEC_ARCH, SPEC_FORMAT, SPEC_FORMAT_V2, SPEC_METADATA_KEY,
};
pub use trainer::{
    MomentsRef, NonFinitePolicy, StepReport, TrainConfig, TrainState, Trainer, DEFAULT_CE_CHUNK,
    NANOLAB_ADAM_LR, NANOLAB_GRAD_CLIP, NANOLAB_MATRIX_LR,
};

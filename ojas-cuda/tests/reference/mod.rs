//! Float64 host references for the CUDA Qwen3.5 training kernels (the parity
//! ladder's rung a: every kernel is held to one of these before it is timed).
//!
//! Each integration test that needs a reference declares `mod reference;` and
//! uses the module for its kernel. Nothing here touches a device, so every
//! reference and its self-validation run on any host, the Mac included.
//!
//! What each reference is validated against, and how strongly, is stated in its
//! own module doc under **Validation**, in three grades:
//!
//! - **golden**: an independent committed fixture (tessl's numpy-generated
//!   published GDN corpus, transformers' RoPE output, or a float64 torch
//!   forward/autograd golden from `tests/fixtures/gen_goldens.py`), within a
//!   bound written before the run;
//! - **tessl in process**: tessl's own host reference, compiled into the test
//!   unmodified and run on the same inputs;
//! - **derivative only**: central finite differences of the reference's own
//!   forward. That checks a backward against its forward; it says nothing
//!   about whether the forward is the right operator.
//!
//! Rule 9 of the Lappi repo binds here: every GDN reference, fixture and test
//! names the `published` rule. nanolab's default `rule="repo"` is a different
//! operator, and [`gdn_published::Rule::Repo`] exists only so a test can show
//! that it does not reproduce the published golden.
//!
//! Each test target uses a subset of these modules, so unused-item warnings
//! are expected per target and silenced here once.
#![allow(dead_code)]

pub mod adamw;
pub mod attn_pieces;
pub mod ce;
pub mod conv1d;
pub mod embed;
pub mod fd;
pub mod gates_published;
pub mod gdn_published;
pub mod goldens;
pub mod norms;
pub mod npy;
pub mod rng;
pub mod sha256;

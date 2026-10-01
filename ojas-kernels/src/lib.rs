//! Shared kernels for every GPU backend.
//!
//! This crate depends only on `ojas-core`. It does not name wgpu, Metal, CUDA,
//! or HIP types. A future out-of-process backend can use the same geometry,
//! source text, and parity harness.

#![forbid(unsafe_code)]

mod geometry;
mod harness;
mod source;

pub use geometry::{
    attention_tiles, cover_1d, fold_grid, gemm_grid, gemm_tile, grid_1d, AttentionTiles, Grid, Limits,
    ATTENTION_PARTS,
    ATTENTION_MAX_HEAD_DIM, GEMM_BIG_TILE, GEMM_TILE,
};
pub use harness::{linear_close, max_abs, splitmix_f32};
pub use source::{affine_cuda, affine_wgsl, wgsl_module, WgslModule};

//! The ops of SpikeTrack's SAM 3.1 frame that the frame ledger
//! (`bench/results/2026-10-27-frame-ledger/ledger.csv`, 7a4a547) had no Ojas
//! number for, measured on Metal at the ledger's exact shapes: layer_norm,
//! channel_layer_norm, residual_add, broadcast add and mul, scale, gelu, relu,
//! rope_interleaved, conv2d, conv_transpose2d, interpolate_bilinear, linear
//! (`linear_bias_forward` and `linear_forward`, f32 and under Autocast) and
//! the attention shapes the ledger lists without a bench. `permute` is
//! Lane A's (`memory_bound_bench`) and `softmax` is not a SpikeTrack call.
//!
//! `cargo run -p ojas-metal --release --example frame_ops_bench [warmups] [runs] [only]`
//!
//! Defaults 5 and 25; `only` is a substring of a row label. Every row is its
//! own run of arms: the op (two arms for a linear, f32 and Autocast) and an f32
//! linear control (fc1, `[5184,1024]x[4736,1024]`), each run twice per round,
//! once in each of two passes (the repeat is the arm's A'); odd rounds walk the
//! arms in reverse. Per arm: min, p10, median and p90 of `runs` timed rounds in
//! milliseconds, GB/s over the bytes the op reads and writes, and the gap of
//! the arm to its repeat against the larger p90 - p10 of the pair (`ok` or
//! `NOISY`). A row attempt is clean when the 1-minute load average at its start
//! and end is at most `MAX_LOAD`, the control repeats `CONTROL_TFLOPS` within
//! `CONTROL_TOL` and no pair is NOISY; a row that is not clean is repeated, up
//! to `ATTEMPTS` times, and every attempt is printed. The rows are generated
//! from ledger.csv and written out below, with the key of the ledger row each
//! one fills; see rules.md for how.

#[cfg(target_os = "macos")]
mod bench {
    use std::process::Command;
    use std::time::Instant;

    use ojas_core::{
        AttentionOpts, Autocast, AutocastMode, Backend, Budget, Conv2dParams,
        ConvTranspose2dParams, Tensor,
    };
    use ojas_metal::MetalBackend;

    /// The 1-minute load above which a row attempt is repeated.
    const MAX_LOAD: f64 = 6.0;
    /// What the fc1 f32 control must repeat: the GEMM study's 6.3 TFLOP/s.
    const CONTROL_TFLOPS: f64 = 6.3;
    const CONTROL_TOL: f64 = 0.03;
    const ATTEMPTS: usize = 3;

    #[derive(Clone, Copy)]
    enum Op {
        LayerNorm,
        ChannelLayerNorm,
        Residual,
        BroadcastAdd,
        BroadcastMul,
        Scale,
        Gelu,
        Relu,
        Rope,
        Conv,
        ConvTranspose,
        Interpolate,
        LinearBias,
        Linear,
        Attention,
    }

    /// One ledger row: the op, the ledger's shape string (the key), the shapes
    /// of its tensor arguments in the order the ledger prints them, and the
    /// extra parameters (conv: stride, padding, groups; interpolate: the output
    /// size).
    struct Row {
        op: Op,
        key: &'static str,
        shapes: &'static [&'static [usize]],
        extra: &'static [usize],
    }

    /// Generated from ledger.csv (7a4a547): every row without an Ojas number, minus `permute`, plus the
    /// memory-dependent rows at each F (m = 5184 F + 32).
    const ROWS: &[Row] = &[
        Row {
            op: Op::Attention,
            key: "[1, 8, 1, 32] x [1, 8, 1, 32] x [1, 8, 1, 32]",
            shapes: &[&[1, 8, 1, 32], &[1, 8, 1, 32], &[1, 8, 1, 32]],
            extra: &[],
        },
        Row {
            op: Op::Attention,
            key: "[1, 8, 1, 32] x [1, 8, 5184, 32] x [1, 8, 5184, 32]",
            shapes: &[&[1, 8, 1, 32], &[1, 8, 5184, 32], &[1, 8, 5184, 32]],
            extra: &[],
        },
        Row {
            op: Op::Attention,
            key: "[1, 8, 201, 32] x [1, 8, 4, 32] x [1, 8, 4, 32]",
            shapes: &[&[1, 8, 201, 32], &[1, 8, 4, 32], &[1, 8, 4, 32]],
            extra: &[],
        },
        Row {
            op: Op::Attention,
            key: "[1, 8, 5184, 16] x [1, 8, 80, 16] x [1, 8, 80, 16]",
            shapes: &[&[1, 8, 5184, 16], &[1, 8, 80, 16], &[1, 8, 80, 16]],
            extra: &[],
        },
        Row {
            op: Op::Attention,
            key: "[1, 8, 5184, 32] x [1, 8, 4, 32] x [1, 8, 4, 32]",
            shapes: &[&[1, 8, 5184, 32], &[1, 8, 4, 32], &[1, 8, 4, 32]],
            extra: &[],
        },
        Row {
            op: Op::Attention,
            key: "[1, 8, 80, 16] x [1, 8, 5184, 16] x [1, 8, 5184, 16]",
            shapes: &[&[1, 8, 80, 16], &[1, 8, 5184, 16], &[1, 8, 5184, 16]],
            extra: &[],
        },
        Row {
            op: Op::Attention,
            key: "[1, 8, 80, 32] x [1, 8, 80, 32] x [1, 8, 80, 32]",
            shapes: &[&[1, 8, 80, 32], &[1, 8, 80, 32], &[1, 8, 80, 32]],
            extra: &[],
        },
        Row {
            op: Op::BroadcastAdd,
            key: "[1, 256, 144, 1, 144, 1] x [1, 1, 1, 2, 1, 2]",
            shapes: &[&[1, 256, 144, 1, 144, 1], &[1, 1, 1, 2, 1, 2]],
            extra: &[],
        },
        Row {
            op: Op::BroadcastAdd,
            key: "[1, 256, 144, 144] x [1, 256, 1, 1]",
            shapes: &[&[1, 256, 144, 144], &[1, 256, 1, 1]],
            extra: &[],
        },
        Row {
            op: Op::BroadcastAdd,
            key: "[1, 256, 288, 288] x [1, 256, 1, 1]",
            shapes: &[&[1, 256, 288, 288], &[1, 256, 1, 1]],
            extra: &[],
        },
        Row {
            op: Op::BroadcastAdd,
            key: "[1, 256, 72, 1, 72, 1] x [1, 1, 1, 2, 1, 2]",
            shapes: &[&[1, 256, 72, 1, 72, 1], &[1, 1, 1, 2, 1, 2]],
            extra: &[],
        },
        Row {
            op: Op::BroadcastMul,
            key: "[1, 256, 144, 144] x [1, 256, 1, 1]",
            shapes: &[&[1, 256, 144, 144], &[1, 256, 1, 1]],
            extra: &[],
        },
        Row {
            op: Op::BroadcastMul,
            key: "[1, 256, 288, 288] x [1, 256, 1, 1]",
            shapes: &[&[1, 256, 288, 288], &[1, 256, 1, 1]],
            extra: &[],
        },
        Row {
            op: Op::BroadcastMul,
            key: "[1, 72, 72, 256] x [256]",
            shapes: &[&[1, 72, 72, 256], &[256]],
            extra: &[],
        },
        Row {
            op: Op::ChannelLayerNorm,
            key: "[1, 1024, 72, 72] x [1024] x [1024]",
            shapes: &[&[1, 1024, 72, 72], &[1024], &[1024]],
            extra: &[],
        },
        Row {
            op: Op::ChannelLayerNorm,
            key: "[1, 16, 576, 576] x [16] x [16]",
            shapes: &[&[1, 16, 576, 576], &[16], &[16]],
            extra: &[],
        },
        Row {
            op: Op::ChannelLayerNorm,
            key: "[1, 256, 144, 144] x [256] x [256]",
            shapes: &[&[1, 256, 144, 144], &[256], &[256]],
            extra: &[],
        },
        Row {
            op: Op::ChannelLayerNorm,
            key: "[1, 256, 72, 72] x [256] x [256]",
            shapes: &[&[1, 256, 72, 72], &[256], &[256]],
            extra: &[],
        },
        Row {
            op: Op::ChannelLayerNorm,
            key: "[1, 64, 144, 144] x [64] x [64]",
            shapes: &[&[1, 64, 144, 144], &[64], &[64]],
            extra: &[],
        },
        Row {
            op: Op::ChannelLayerNorm,
            key: "[1, 64, 288, 288] x [64] x [64]",
            shapes: &[&[1, 64, 288, 288], &[64], &[64]],
            extra: &[],
        },
        Row {
            op: Op::Conv,
            key: "[1, 1024, 72, 72] x [256, 1024, 1, 1] x [256]",
            shapes: &[&[1, 1024, 72, 72], &[256, 1024, 1, 1], &[256]],
            extra: &[1, 0, 1],
        },
        Row {
            op: Op::Conv,
            key: "[1, 16, 576, 576] x [64, 16, 3, 3] x [64]",
            shapes: &[&[1, 16, 576, 576], &[64, 16, 3, 3], &[64]],
            extra: &[2, 1, 1],
        },
        Row {
            op: Op::Conv,
            key: "[1, 256, 144, 144] x [1024, 256, 3, 3] x [1024]",
            shapes: &[&[1, 256, 144, 144], &[1024, 256, 3, 3], &[1024]],
            extra: &[2, 1, 1],
        },
        Row {
            op: Op::Conv,
            key: "[1, 256, 144, 144] x [64, 256, 1, 1] x [64]",
            shapes: &[&[1, 256, 144, 144], &[64, 256, 1, 1], &[64]],
            extra: &[1, 0, 1],
        },
        Row {
            op: Op::Conv,
            key: "[1, 256, 288, 288] x [1, 256, 1, 1] x [1]",
            shapes: &[&[1, 256, 288, 288], &[1, 256, 1, 1], &[1]],
            extra: &[1, 0, 1],
        },
        Row {
            op: Op::Conv,
            key: "[1, 256, 288, 288] x [256, 256, 1, 1] x [256]",
            shapes: &[&[1, 256, 288, 288], &[256, 256, 1, 1], &[256]],
            extra: &[1, 0, 1],
        },
        Row {
            op: Op::Conv,
            key: "[1, 256, 288, 288] x [32, 256, 1, 1] x [32]",
            shapes: &[&[1, 256, 288, 288], &[32, 256, 1, 1], &[32]],
            extra: &[1, 0, 1],
        },
        Row {
            op: Op::Conv,
            key: "[1, 256, 72, 72] x [256, 1, 7, 7] x [256]",
            shapes: &[&[1, 256, 72, 72], &[256, 1, 7, 7], &[256]],
            extra: &[1, 3, 256],
        },
        Row {
            op: Op::Conv,
            key: "[1, 256, 72, 72] x [256, 256, 1, 1] x [256]",
            shapes: &[&[1, 256, 72, 72], &[256, 256, 1, 1], &[256]],
            extra: &[1, 0, 1],
        },
        Row {
            op: Op::Conv,
            key: "[1, 32, 1152, 1152] x [16, 32, 3, 3] x [16]",
            shapes: &[&[1, 32, 1152, 1152], &[16, 32, 3, 3], &[16]],
            extra: &[2, 1, 1],
        },
        Row {
            op: Op::Conv,
            key: "[1, 512, 144, 144] x [256, 512, 1, 1] x [256]",
            shapes: &[&[1, 512, 144, 144], &[256, 512, 1, 1], &[256]],
            extra: &[1, 0, 1],
        },
        Row {
            op: Op::Conv,
            key: "[1, 64, 288, 288] x [256, 64, 3, 3] x [256]",
            shapes: &[&[1, 64, 288, 288], &[256, 64, 3, 3], &[256]],
            extra: &[2, 1, 1],
        },
        Row {
            op: Op::ConvTranspose,
            key: "[1, 1024, 72, 72] x [1024, 512, 2, 2] x [512]",
            shapes: &[&[1, 1024, 72, 72], &[1024, 512, 2, 2], &[512]],
            extra: &[],
        },
        Row {
            op: Op::ConvTranspose,
            key: "[1, 256, 72, 72] x [256, 64, 2, 2] x [64]",
            shapes: &[&[1, 256, 72, 72], &[256, 64, 2, 2], &[64]],
            extra: &[],
        },
        Row {
            op: Op::ConvTranspose,
            key: "[1, 512, 144, 144] x [512, 256, 2, 2] x [256]",
            shapes: &[&[1, 512, 144, 144], &[512, 256, 2, 2], &[256]],
            extra: &[],
        },
        Row {
            op: Op::ConvTranspose,
            key: "[1, 64, 144, 144] x [64, 32, 2, 2] x [32]",
            shapes: &[&[1, 64, 144, 144], &[64, 32, 2, 2], &[32]],
            extra: &[],
        },
        Row {
            op: Op::Gelu,
            key: "[1, 1024, 72, 72]",
            shapes: &[&[1, 1024, 72, 72]],
            extra: &[],
        },
        Row {
            op: Op::Gelu,
            key: "[1, 16, 576, 576]",
            shapes: &[&[1, 16, 576, 576]],
            extra: &[],
        },
        Row {
            op: Op::Gelu,
            key: "[1, 256, 144, 144]",
            shapes: &[&[1, 256, 144, 144]],
            extra: &[],
        },
        Row {
            op: Op::Gelu,
            key: "[1, 32, 288, 288]",
            shapes: &[&[1, 32, 288, 288]],
            extra: &[],
        },
        Row {
            op: Op::Gelu,
            key: "[1, 512, 144, 144]",
            shapes: &[&[1, 512, 144, 144]],
            extra: &[],
        },
        Row {
            op: Op::Gelu,
            key: "[1, 5184, 2048]",
            shapes: &[&[1, 5184, 2048]],
            extra: &[],
        },
        Row {
            op: Op::Gelu,
            key: "[1, 64, 144, 144]",
            shapes: &[&[1, 64, 144, 144]],
            extra: &[],
        },
        Row {
            op: Op::Gelu,
            key: "[1, 64, 288, 288]",
            shapes: &[&[1, 64, 288, 288]],
            extra: &[],
        },
        Row {
            op: Op::Gelu,
            key: "[1, 72, 72, 1024]",
            shapes: &[&[1, 72, 72, 1024]],
            extra: &[],
        },
        Row {
            op: Op::Gelu,
            key: "[1, 72, 72, 4736]",
            shapes: &[&[1, 72, 72, 4736]],
            extra: &[],
        },
        Row {
            op: Op::Interpolate,
            key: "[1, 32, 1008, 1008]",
            shapes: &[&[1, 32, 1008, 1008]],
            extra: &[1152, 1152],
        },
        Row {
            op: Op::LayerNorm,
            key: "[1, 1, 256] x [256] x [256]",
            shapes: &[&[1, 1, 256], &[256], &[256]],
            extra: &[],
        },
        Row {
            op: Op::LayerNorm,
            key: "[1, 201, 256] x [256] x [256]",
            shapes: &[&[1, 201, 256], &[256], &[256]],
            extra: &[],
        },
        Row {
            op: Op::LayerNorm,
            key: "[1, 5184, 256] x [256] x [256]",
            shapes: &[&[1, 5184, 256], &[256], &[256]],
            extra: &[],
        },
        Row {
            op: Op::LayerNorm,
            key: "[1, 72, 72, 1024] x [1024] x [1024]",
            shapes: &[&[1, 72, 72, 1024], &[1024], &[1024]],
            extra: &[],
        },
        Row {
            op: Op::LayerNorm,
            key: "[1, 8, 2654208] x [2654208] x [2654208]",
            shapes: &[&[1, 8, 2654208], &[2654208], &[2654208]],
            extra: &[],
        },
        Row {
            op: Op::LayerNorm,
            key: "[1, 8, 663552] x [663552] x [663552]",
            shapes: &[&[1, 8, 663552], &[663552], &[663552]],
            extra: &[],
        },
        Row {
            op: Op::LayerNorm,
            key: "[1, 80, 256] x [256] x [256]",
            shapes: &[&[1, 80, 256], &[256], &[256]],
            extra: &[],
        },
        Row {
            op: Op::LinearBias,
            key: "[1, 1, 2048] x [256, 2048] x [256]",
            shapes: &[&[1, 1, 2048], &[256, 2048], &[256]],
            extra: &[],
        },
        Row {
            op: Op::LinearBias,
            key: "[1, 1, 256] x [2048, 256] x [2048]",
            shapes: &[&[1, 1, 256], &[2048, 256], &[2048]],
            extra: &[],
        },
        Row {
            op: Op::LinearBias,
            key: "[1, 1, 256] x [256, 256] x [256]",
            shapes: &[&[1, 1, 256], &[256, 256], &[256]],
            extra: &[],
        },
        Row {
            op: Op::LinearBias,
            key: "[1, 5216, 256] x [256, 256] x [256]",
            shapes: &[&[1, 5216, 256], &[256, 256], &[256]],
            extra: &[],
        },
        Row {
            op: Op::LinearBias,
            key: "[1, 10400, 256] x [256, 256] x [256]",
            shapes: &[&[1, 10400, 256], &[256, 256], &[256]],
            extra: &[],
        },
        Row {
            op: Op::LinearBias,
            key: "[1, 15584, 256] x [256, 256] x [256]",
            shapes: &[&[1, 15584, 256], &[256, 256], &[256]],
            extra: &[],
        },
        Row {
            op: Op::LinearBias,
            key: "[1, 36320, 256] x [256, 256] x [256]",
            shapes: &[&[1, 36320, 256], &[256, 256], &[256]],
            extra: &[],
        },
        Row {
            op: Op::LinearBias,
            key: "[1, 201, 2048] x [256, 2048] x [256]",
            shapes: &[&[1, 201, 2048], &[256, 2048], &[256]],
            extra: &[],
        },
        Row {
            op: Op::LinearBias,
            key: "[1, 201, 256] x [2048, 256] x [2048]",
            shapes: &[&[1, 201, 256], &[2048, 256], &[2048]],
            extra: &[],
        },
        Row {
            op: Op::LinearBias,
            key: "[1, 5184, 128] x [256, 128] x [256]",
            shapes: &[&[1, 5184, 128], &[256, 128], &[256]],
            extra: &[],
        },
        Row {
            op: Op::LinearBias,
            key: "[1, 5184, 256] x [128, 256] x [128]",
            shapes: &[&[1, 5184, 256], &[128, 256], &[128]],
            extra: &[],
        },
        Row {
            op: Op::LinearBias,
            key: "[1, 72, 72, 1024] x [256, 1024] x [256]",
            shapes: &[&[1, 72, 72, 1024], &[256, 1024], &[256]],
            extra: &[],
        },
        Row {
            op: Op::LinearBias,
            key: "[1, 72, 72, 256] x [1024, 256] x [1024]",
            shapes: &[&[1, 72, 72, 256], &[1024, 256], &[1024]],
            extra: &[],
        },
        Row {
            op: Op::LinearBias,
            key: "[1, 80, 128] x [256, 128] x [256]",
            shapes: &[&[1, 80, 128], &[256, 128], &[256]],
            extra: &[],
        },
        Row {
            op: Op::LinearBias,
            key: "[1, 80, 2048] x [256, 2048] x [256]",
            shapes: &[&[1, 80, 2048], &[256, 2048], &[256]],
            extra: &[],
        },
        Row {
            op: Op::LinearBias,
            key: "[1, 80, 256] x [128, 256] x [128]",
            shapes: &[&[1, 80, 256], &[128, 256], &[128]],
            extra: &[],
        },
        Row {
            op: Op::LinearBias,
            key: "[1, 80, 256] x [2048, 256] x [2048]",
            shapes: &[&[1, 80, 256], &[2048, 256], &[2048]],
            extra: &[],
        },
        Row {
            op: Op::LinearBias,
            key: "[1, 80, 256] x [256, 256] x [256]",
            shapes: &[&[1, 80, 256], &[256, 256], &[256]],
            extra: &[],
        },
        Row {
            op: Op::LinearBias,
            key: "[200, 256] x [256, 256] x [256]",
            shapes: &[&[200, 256], &[256, 256], &[256]],
            extra: &[],
        },
        Row {
            op: Op::Linear,
            key: "[200, 256] x [82944, 256]",
            shapes: &[&[200, 256], &[82944, 256]],
            extra: &[],
        },
        Row {
            op: Op::Linear,
            key: "[48, 32] x [82944, 32]",
            shapes: &[&[48, 32], &[82944, 32]],
            extra: &[],
        },
        Row {
            op: Op::Relu,
            key: "[1, 1, 2048]",
            shapes: &[&[1, 1, 2048]],
            extra: &[],
        },
        Row {
            op: Op::Relu,
            key: "[1, 201, 2048]",
            shapes: &[&[1, 201, 2048]],
            extra: &[],
        },
        Row {
            op: Op::Relu,
            key: "[1, 256, 144, 144]",
            shapes: &[&[1, 256, 144, 144]],
            extra: &[],
        },
        Row {
            op: Op::Relu,
            key: "[1, 256, 288, 288]",
            shapes: &[&[1, 256, 288, 288]],
            extra: &[],
        },
        Row {
            op: Op::Relu,
            key: "[1, 5184, 2048]",
            shapes: &[&[1, 5184, 2048]],
            extra: &[],
        },
        Row {
            op: Op::Relu,
            key: "[1, 80, 2048]",
            shapes: &[&[1, 80, 2048]],
            extra: &[],
        },
        Row {
            op: Op::Relu,
            key: "[200, 256]",
            shapes: &[&[200, 256]],
            extra: &[],
        },
        Row {
            op: Op::Residual,
            key: "[1, 1, 256] x [1, 1, 256]",
            shapes: &[&[1, 1, 256], &[1, 1, 256]],
            extra: &[],
        },
        Row {
            op: Op::Residual,
            key: "[1, 5216, 256] x [1, 5216, 256]",
            shapes: &[&[1, 5216, 256], &[1, 5216, 256]],
            extra: &[],
        },
        Row {
            op: Op::Residual,
            key: "[1, 10400, 256] x [1, 10400, 256]",
            shapes: &[&[1, 10400, 256], &[1, 10400, 256]],
            extra: &[],
        },
        Row {
            op: Op::Residual,
            key: "[1, 15584, 256] x [1, 15584, 256]",
            shapes: &[&[1, 15584, 256], &[1, 15584, 256]],
            extra: &[],
        },
        Row {
            op: Op::Residual,
            key: "[1, 36320, 256] x [1, 36320, 256]",
            shapes: &[&[1, 36320, 256], &[1, 36320, 256]],
            extra: &[],
        },
        Row {
            op: Op::Residual,
            key: "[1, 201, 256] x [1, 201, 256]",
            shapes: &[&[1, 201, 256], &[1, 201, 256]],
            extra: &[],
        },
        Row {
            op: Op::Residual,
            key: "[1, 256, 144, 144] x [1, 256, 144, 144]",
            shapes: &[&[1, 256, 144, 144], &[1, 256, 144, 144]],
            extra: &[],
        },
        Row {
            op: Op::Residual,
            key: "[1, 256, 288, 288] x [1, 256, 288, 288]",
            shapes: &[&[1, 256, 288, 288], &[1, 256, 288, 288]],
            extra: &[],
        },
        Row {
            op: Op::Residual,
            key: "[1, 256, 72, 72] x [1, 256, 72, 72]",
            shapes: &[&[1, 256, 72, 72], &[1, 256, 72, 72]],
            extra: &[],
        },
        Row {
            op: Op::Residual,
            key: "[1, 32, 288, 288] x [1, 32, 288, 288]",
            shapes: &[&[1, 32, 288, 288], &[1, 32, 288, 288]],
            extra: &[],
        },
        Row {
            op: Op::Residual,
            key: "[1, 5184, 256] x [1, 5184, 256]",
            shapes: &[&[1, 5184, 256], &[1, 5184, 256]],
            extra: &[],
        },
        Row {
            op: Op::Residual,
            key: "[1, 64, 144, 144] x [1, 64, 144, 144]",
            shapes: &[&[1, 64, 144, 144], &[1, 64, 144, 144]],
            extra: &[],
        },
        Row {
            op: Op::Residual,
            key: "[1, 72, 72, 1024] x [1, 72, 72, 1024]",
            shapes: &[&[1, 72, 72, 1024], &[1, 72, 72, 1024]],
            extra: &[],
        },
        Row {
            op: Op::Residual,
            key: "[1, 80, 256] x [1, 80, 256]",
            shapes: &[&[1, 80, 256], &[1, 80, 256]],
            extra: &[],
        },
        Row {
            op: Op::Rope,
            key: "[1, 5216, 8, 32] x [5216, 16] x [5216, 16]",
            shapes: &[&[1, 5216, 8, 32], &[5216, 16], &[5216, 16]],
            extra: &[],
        },
        Row {
            op: Op::Rope,
            key: "[1, 10400, 8, 32] x [10400, 16] x [10400, 16]",
            shapes: &[&[1, 10400, 8, 32], &[10400, 16], &[10400, 16]],
            extra: &[],
        },
        Row {
            op: Op::Rope,
            key: "[1, 15584, 8, 32] x [15584, 16] x [15584, 16]",
            shapes: &[&[1, 15584, 8, 32], &[15584, 16], &[15584, 16]],
            extra: &[],
        },
        Row {
            op: Op::Rope,
            key: "[1, 36320, 8, 32] x [36320, 16] x [36320, 16]",
            shapes: &[&[1, 36320, 8, 32], &[36320, 16], &[36320, 16]],
            extra: &[],
        },
        Row {
            op: Op::Rope,
            key: "[1, 5184, 16, 64] x [5184, 32] x [5184, 32]",
            shapes: &[&[1, 5184, 16, 64], &[5184, 32], &[5184, 32]],
            extra: &[],
        },
        Row {
            op: Op::Rope,
            key: "[1, 5184, 8, 32] x [5184, 16] x [5184, 16]",
            shapes: &[&[1, 5184, 8, 32], &[5184, 16], &[5184, 16]],
            extra: &[],
        },
        Row {
            op: Op::Rope,
            key: "[9, 576, 16, 64] x [576, 32] x [576, 32]",
            shapes: &[&[9, 576, 16, 64], &[576, 32], &[576, 32]],
            extra: &[],
        },
        Row {
            op: Op::Scale,
            key: "[1, 5184, 256]",
            shapes: &[&[1, 5184, 256]],
            extra: &[],
        },
    ];

    fn values(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let u = (s >> 40) as f32 / (1u64 << 24) as f32;
                (2.0 * u - 1.0) * scale
            })
            .collect()
    }

    /// `(1-minute, 5-minute, 15-minute)` load averages from `sysctl vm.loadavg`.
    fn load() -> [f64; 3] {
        let out = Command::new("sysctl")
            .args(["-n", "vm.loadavg"])
            .output()
            .expect("sysctl vm.loadavg");
        let text = String::from_utf8_lossy(&out.stdout).into_owned();
        let v: Vec<f64> = text
            .split_whitespace()
            .filter_map(|t| t.parse().ok())
            .collect();
        assert!(v.len() >= 3, "unparsable vm.loadavg: {text:?}");
        [v[0], v[1], v[2]]
    }

    type Arm<'a> = (String, u64, Box<dyn Fn() + 'a>);

    struct Stat {
        min: f64,
        p10: f64,
        med: f64,
        p90: f64,
    }

    fn stat(mut ms: Vec<f64>) -> Stat {
        ms.sort_by(f64::total_cmp);
        let at = |q: f64| ms[((ms.len() - 1) as f64 * q).round() as usize];
        Stat {
            min: ms[0],
            p10: at(0.10),
            med: at(0.50),
            p90: at(0.90),
        }
    }

    /// The control: an f32 `linear_forward` at fc1.
    struct Control<'a> {
        x: Tensor,
        w: Tensor,
        metal: &'a MetalBackend,
    }

    const FC1: (usize, usize, usize) = (5184, 1024, 4736);

    /// One attempt at a row: every arm plus the control, each twice per round.
    /// Returns whether the attempt was clean.
    fn attempt(
        metal: &MetalBackend,
        ctl: &Control<'_>,
        label: &str,
        attempt: usize,
        arms: &[Arm<'_>],
        warmups: usize,
        runs: usize,
    ) -> bool {
        let control: Arm<'_> = (
            "control: f32 linear fc1".to_string(),
            4 * (FC1.0 * FC1.1 + FC1.2 * FC1.1 + FC1.0 * FC1.2) as u64,
            Box::new(move || drop(ctl.metal.linear_forward(&ctl.x, &ctl.w).expect("control"))),
        );
        let all: Vec<&Arm<'_>> = arms.iter().chain(std::iter::once(&control)).collect();
        let count = all.len();
        let load_start = load();
        let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(runs); 2 * count];
        for round in 0..warmups + runs {
            for pass in 0..2 {
                for step in 0..count {
                    let i = if round % 2 == 0 {
                        step
                    } else {
                        count - 1 - step
                    };
                    metal.sync().expect("sync");
                    let start = Instant::now();
                    (all[i].2)();
                    metal.sync().expect("sync");
                    let ms = start.elapsed().as_secs_f64() * 1e3;
                    if round >= warmups {
                        samples[pass * count + i].push(ms);
                    }
                }
            }
        }
        let load_end = load();
        let mut noisy = 0;
        let mut control_tflops = 0.0;
        for (i, (name, bytes, _)) in all.iter().map(|a| (&a.0, a.1, ())).enumerate() {
            let a = stat(samples[i].clone());
            let b = stat(samples[count + i].clone());
            let gap = (a.med - b.med).abs();
            let spread = (a.p90 - a.p10).max(b.p90 - b.p10);
            let ok = gap <= spread;
            if !ok {
                noisy += 1;
            }
            if i == count - 1 {
                let flops = 2.0 * FC1.0 as f64 * FC1.1 as f64 * FC1.2 as f64;
                control_tflops = flops / (a.med * 1e-3) / 1e12;
            }
            for (r, tag) in [(&a, ""), (&b, "'")] {
                let gbs = bytes as f64 / (r.med * 1e-3) / 1e9;
                println!(
                    "{label}\t{attempt}\t{name}{tag}\t{bytes}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{gbs:.1}\t{gap:.4}\t{spread:.4}\t{}",
                    r.min,
                    r.p10,
                    r.med,
                    r.p90,
                    if ok { "ok" } else { "NOISY" }
                );
            }
        }
        let load_ok = load_start[0] <= MAX_LOAD && load_end[0] <= MAX_LOAD;
        let control_ok = (control_tflops / CONTROL_TFLOPS - 1.0).abs() <= CONTROL_TOL;
        let clean = load_ok && control_ok && noisy == 0;
        println!(
            "# attempt\t{label}\t{attempt}\tload_start {:.2} {:.2} {:.2}\tload_end {:.2} {:.2} {:.2}\tcontrol_tflops {control_tflops:.2}\tcontrol_ok {control_ok}\tnoisy_pairs {noisy}\tverdict {}",
            load_start[0],
            load_start[1],
            load_start[2],
            load_end[0],
            load_end[1],
            load_end[2],
            if clean { "clean" } else { "repeat" }
        );
        clean
    }

    fn numel(shape: &[usize]) -> usize {
        shape.iter().product()
    }

    fn bytes_of(shapes: &[&[usize]], out: &[usize]) -> u64 {
        4 * (shapes.iter().map(|s| numel(s)).sum::<usize>() + numel(out)) as u64
    }

    pub fn run() {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let num = |i: usize, d: usize| args.get(i).and_then(|a| a.parse().ok()).unwrap_or(d);
        let (warmups, runs) = (num(0, 5), num(1, 25));
        let only = args.get(2).cloned().unwrap_or_default();
        let budget = Budget::new(24 << 30);
        let metal_owner = MetalBackend::new(budget.clone()).expect("Metal device");
        let metal = &metal_owner;
        assert!(
            metal.has_tensorops(),
            "the control and the Autocast arm need TensorOps: this device has none"
        );
        let host =
            |data: &[f32], shape: &[usize]| Tensor::from_f32(data, shape, &budget).expect("host");
        let up = |t: &Tensor| metal.upload(t).expect("upload");
        let dev = |shape: &[usize], seed: u64, scale: f32| {
            up(&host(&values(numel(shape), seed, scale), shape))
        };
        let ctl = Control {
            x: dev(&[FC1.0, FC1.1], 91, 1.0),
            w: dev(&[FC1.2, FC1.1], 92, 0.05),
            metal,
        };
        let start_load = load();
        println!(
            "# load_start {:.2} {:.2} {:.2}",
            start_load[0], start_load[1], start_load[2]
        );
        println!(
            "# columns: row\tattempt\tarm\tbytes\tmin_ms\tp10_ms\tmedian_ms\tp90_ms\tgb_per_s\tgap_ms\tspread_ms\tpair"
        );
        let (mut rows_clean, mut rows_total) = (0usize, 0usize);

        for row in ROWS.iter().filter(|r| r.key.contains(&only)) {
            rows_total += 1;
            let label = format!(
                "{}\t{}",
                match row.op {
                    Op::LayerNorm => "layer_norm_forward",
                    Op::ChannelLayerNorm => "channel_layer_norm_forward",
                    Op::Residual => "residual_add_forward",
                    Op::BroadcastAdd => "broadcast_add_forward",
                    Op::BroadcastMul => "broadcast_mul_forward",
                    Op::Scale => "scale_forward",
                    Op::Gelu => "gelu_erf_forward",
                    Op::Relu => "relu_forward",
                    Op::Rope => "rope_interleaved_forward",
                    Op::Conv => "conv2d_forward",
                    Op::ConvTranspose => "conv_transpose2d_forward",
                    Op::Interpolate => "interpolate_bilinear_forward",
                    Op::LinearBias => "linear_bias_forward",
                    Op::Linear => "linear_forward",
                    Op::Attention => "attention_forward",
                },
                row.key
            );
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let s = row.shapes;
                // Device operands, one per shape; weights small, activations of order one.
                let t: Vec<Tensor> = s
                    .iter()
                    .enumerate()
                    .map(|(i, sh)| dev(sh, 10 + i as u64, if i == 0 { 1.0 } else { 0.05 }))
                    .collect();
                let t = &t;
                let ac = Autocast::new(metal);
                let acr = &ac;
                let _region = ac.autocast_region(AutocastMode::Bf16).expect("region");
                let arms: Vec<Arm<'_>> = match row.op {
                    Op::LayerNorm => {
                        let out = metal
                            .layer_norm_forward(&t[0], &t[1], &t[2], 1e-5)
                            .expect("layer_norm");
                        vec![(
                            "op".into(),
                            bytes_of(s, out.shape()),
                            Box::new(move || {
                                drop(
                                    metal
                                        .layer_norm_forward(&t[0], &t[1], &t[2], 1e-5)
                                        .expect("layer_norm"),
                                )
                            }),
                        )]
                    }
                    Op::ChannelLayerNorm => {
                        let out = metal
                            .channel_layer_norm_forward(&t[0], &t[1], &t[2], 1e-6)
                            .expect("channel ln");
                        vec![(
                            "op".into(),
                            bytes_of(s, out.shape()),
                            Box::new(move || {
                                drop(
                                    metal
                                        .channel_layer_norm_forward(&t[0], &t[1], &t[2], 1e-6)
                                        .expect("channel ln"),
                                )
                            }),
                        )]
                    }
                    Op::Residual => {
                        let out = metal.residual_add_forward(&t[0], &t[1]).expect("residual");
                        vec![(
                            "op".into(),
                            bytes_of(s, out.shape()),
                            Box::new(move || {
                                drop(metal.residual_add_forward(&t[0], &t[1]).expect("residual"))
                            }),
                        )]
                    }
                    Op::BroadcastAdd => {
                        let out = metal
                            .broadcast_add_forward(&t[0], &t[1])
                            .expect("broadcast add");
                        vec![(
                            "op".into(),
                            bytes_of(s, out.shape()),
                            Box::new(move || {
                                drop(
                                    metal
                                        .broadcast_add_forward(&t[0], &t[1])
                                        .expect("broadcast add"),
                                )
                            }),
                        )]
                    }
                    Op::BroadcastMul => {
                        let out = metal
                            .broadcast_mul_forward(&t[0], &t[1])
                            .expect("broadcast mul");
                        vec![(
                            "op".into(),
                            bytes_of(s, out.shape()),
                            Box::new(move || {
                                drop(
                                    metal
                                        .broadcast_mul_forward(&t[0], &t[1])
                                        .expect("broadcast mul"),
                                )
                            }),
                        )]
                    }
                    Op::Scale => {
                        let out = metal.scale_forward(&t[0], 0.1).expect("scale");
                        vec![(
                            "op".into(),
                            bytes_of(s, out.shape()),
                            Box::new(move || drop(metal.scale_forward(&t[0], 0.1).expect("scale"))),
                        )]
                    }
                    Op::Gelu => {
                        let out = metal.gelu_erf_forward(&t[0]).expect("gelu");
                        vec![(
                            "op".into(),
                            bytes_of(s, out.shape()),
                            Box::new(move || drop(metal.gelu_erf_forward(&t[0]).expect("gelu"))),
                        )]
                    }
                    Op::Relu => {
                        let out = metal.relu_forward(&t[0]).expect("relu");
                        vec![(
                            "op".into(),
                            bytes_of(s, out.shape()),
                            Box::new(move || drop(metal.relu_forward(&t[0]).expect("relu"))),
                        )]
                    }
                    Op::Rope => {
                        let out = metal
                            .rope_interleaved_forward(&t[0], &t[1], &t[2])
                            .expect("rope");
                        vec![(
                            "op".into(),
                            bytes_of(s, out.shape()),
                            Box::new(move || {
                                drop(
                                    metal
                                        .rope_interleaved_forward(&t[0], &t[1], &t[2])
                                        .expect("rope"),
                                )
                            }),
                        )]
                    }
                    Op::Conv => {
                        let mut p = Conv2dParams::new();
                        p.stride = [row.extra[0], row.extra[0]];
                        p.padding = [row.extra[1], row.extra[1]];
                        p.groups = row.extra[2];
                        let out = metal
                            .conv2d_forward(&t[0], &t[1], Some(&t[2]), p)
                            .expect("conv2d");
                        vec![(
                            "op".into(),
                            bytes_of(s, out.shape()),
                            Box::new(move || {
                                drop(
                                    metal
                                        .conv2d_forward(&t[0], &t[1], Some(&t[2]), p)
                                        .expect("conv2d"),
                                )
                            }),
                        )]
                    }
                    Op::ConvTranspose => {
                        let mut p = ConvTranspose2dParams::new();
                        p.stride = [2, 2];
                        let out = metal
                            .conv_transpose2d_forward(&t[0], &t[1], Some(&t[2]), p)
                            .expect("convT");
                        vec![(
                            "op".into(),
                            bytes_of(s, out.shape()),
                            Box::new(move || {
                                drop(
                                    metal
                                        .conv_transpose2d_forward(&t[0], &t[1], Some(&t[2]), p)
                                        .expect("convT"),
                                )
                            }),
                        )]
                    }
                    Op::Interpolate => {
                        let size = [row.extra[0], row.extra[1]];
                        let out = metal
                            .interpolate_bilinear_forward(&t[0], size, false)
                            .expect("interpolate");
                        vec![(
                            "op".into(),
                            bytes_of(s, out.shape()),
                            Box::new(move || {
                                drop(
                                    metal
                                        .interpolate_bilinear_forward(&t[0], size, false)
                                        .expect("interpolate"),
                                )
                            }),
                        )]
                    }
                    Op::LinearBias => {
                        let out = metal
                            .linear_bias_forward(&t[0], &t[1], &t[2])
                            .expect("linear_bias");
                        let bytes = bytes_of(s, out.shape());
                        vec![
                            (
                                "f32".into(),
                                bytes,
                                Box::new(move || {
                                    drop(
                                        metal
                                            .linear_bias_forward(&t[0], &t[1], &t[2])
                                            .expect("linear_bias"),
                                    )
                                }),
                            ),
                            (
                                "autocast".into(),
                                bytes,
                                Box::new(move || {
                                    drop(
                                        acr.linear_bias_forward(&t[0], &t[1], &t[2])
                                            .expect("autocast linear_bias"),
                                    )
                                }),
                            ),
                        ]
                    }
                    Op::Linear => {
                        let out = metal.linear_forward(&t[0], &t[1]).expect("linear");
                        let bytes = bytes_of(s, out.shape());
                        vec![
                            (
                                "f32".into(),
                                bytes,
                                Box::new(move || {
                                    drop(metal.linear_forward(&t[0], &t[1]).expect("linear"))
                                }),
                            ),
                            (
                                "autocast".into(),
                                bytes,
                                Box::new(move || {
                                    drop(acr.linear_forward(&t[0], &t[1]).expect("autocast linear"))
                                }),
                            ),
                        ]
                    }
                    Op::Attention => {
                        let opts = AttentionOpts::default();
                        let out = metal
                            .attention_forward(&t[0], &t[1], &t[2], opts)
                            .expect("attention");
                        vec![(
                            "op".into(),
                            bytes_of(s, out.shape()),
                            Box::new(move || {
                                drop(
                                    metal
                                        .attention_forward(&t[0], &t[1], &t[2], opts)
                                        .expect("attention"),
                                )
                            }),
                        )]
                    }
                };
                let mut clean = false;
                for n in 1..=ATTEMPTS {
                    if attempt(metal, &ctl, &label, n, &arms, warmups, runs) {
                        clean = true;
                        break;
                    }
                }
                clean
            }));
            match outcome {
                Ok(clean) => {
                    if clean {
                        rows_clean += 1;
                    }
                    println!("# row\t{label}\tclean {clean}");
                }
                Err(e) => {
                    // The backend refused the op at this shape (the example's expect); the row has no number.
                    let why = e
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                        .unwrap_or_default();
                    println!("# row\t{label}\trefused: {}", why.replace('\n', " "));
                }
            }
        }

        let end_load = load();
        println!(
            "# load_end {:.2} {:.2} {:.2}",
            end_load[0], end_load[1], end_load[2]
        );
        println!("# rows_clean {rows_clean} of {rows_total}");
    }
}

#[cfg(target_os = "macos")]
fn main() {
    bench::run();
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("frame_ops_bench needs macOS");
}

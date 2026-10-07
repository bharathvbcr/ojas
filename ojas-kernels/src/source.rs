//! Kernel source text. Backends compile these strings. HIP is not registered:
//! the CUDA-C source is written so a future hiprtc path can reuse it unchanged.

use ojas_core::OjasError;

use crate::geometry::{AttentionTiles, ATTENTION_MAX_HEAD_DIM, ATTENTION_PARTS};

pub const fn affine_wgsl() -> &'static str {
    r#"
struct Params {
    scale: f32,
    bias: f32,
}

@group(0) @binding(0) var<storage, read> input_values: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_values: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let index = gid.y * groups.x * 64u + gid.x;
    if (index >= arrayLength(&input_values)) {
        return;
    }
    output_values[index] = input_values[index] * params.scale + params.bias;
}
"#
}

pub const fn affine_cuda() -> &'static str {
    r#"
extern "C" __global__ void affine_f32(
    float* out,
    const float* inp,
    float scale,
    float bias,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = inp[i] * scale + bias;
    }
}
"#
}

const COMMON: &str = include_str!("wgsl/common.wgsl");
const TREE: &str = include_str!("wgsl/tree.wgsl");
const GEMM: &str = include_str!("wgsl/gemm.wgsl");
const POINTWISE: &str = include_str!("wgsl/pointwise.wgsl");
const REDUCE: &str = include_str!("wgsl/reduce.wgsl");
const NORM: &str = include_str!("wgsl/norm.wgsl");
const LOSS: &str = include_str!("wgsl/loss.wgsl");
const OPTIM: &str = include_str!("wgsl/optim.wgsl");
const ATTENTION: &str = include_str!("wgsl/attention.wgsl");
const CACHED_ATTENTION: &str = include_str!("wgsl/cached_attention.wgsl");
const LAYOUT: &str = include_str!("wgsl/layout.wgsl");
const FAULT: &str = include_str!("wgsl/fault.wgsl");

/// WGSL module families for the device-resident wgpu backend.
///
/// Every module starts with the shared header: binding 0 is sixteen u32
/// parameter words (word 15 is the launching op's fault id, its index + 1)
/// and binding 1 is the fault words: a 64-bit op mask in words 0 and 1, the
/// first faulting op in word 2. Kernel bindings start at 2.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WgslModule {
    Gemm,
    Pointwise,
    Reduce,
    Norm,
    Loss,
    Optim,
    /// Bit-exact moves (permute) over `u32` words.
    Layout,
    /// The host hand-off of the fault words (`fault_hold`, `fault_release`).
    Fault,
    /// Tiled flash attention for one padded head width and block plan.
    Attention(AttentionTiles),
    /// Split-key causal attention of new queries against a time-major KV
    /// cache (decode, and prefill onto a non-empty cache).
    CachedAttention,
}

/// The tiled attention template for `tiles`. A plan the template cannot
/// realise (padded width not 16/32/64/128/256, a block not a multiple of
/// [`ATTENTION_PARTS`]) is refused, never adjusted.
fn attention_source(tiles: AttentionTiles) -> Result<String, OjasError> {
    let AttentionTiles {
        padded_dim: dp,
        fwd_rows: fq,
        fwd_keys: fk,
        bwd_block: bb,
        ..
    } = tiles;
    let parts = ATTENTION_PARTS;
    let ok_dim = matches!(dp, 16 | 32 | 64 | 128 | 256) && dp <= ATTENTION_MAX_HEAD_DIM;
    let ok_block = |b: u32| (parts..=32).contains(&b) && b.is_multiple_of(parts);
    if !(ok_dim && ok_block(fq) && ok_block(fk) && ok_block(bb)) {
        return Err(OjasError::OutOfRange {
            op: "wgsl_module",
            detail: format!("attention plan {tiles:?} is not a valid template"),
        });
    }
    let dp4 = dp / 4;
    let sp = dp4 + 1;
    let words: [(&str, u32); 21] = [
        ("{{DP4}}", dp4),
        ("{{SP}}", sp),
        ("{{CV}}", dp4 / parts),
        ("{{FQ}}", fq),
        ("{{FKP}}", fk + 1),
        ("{{FKPT}}", fk / parts),
        ("{{FK}}", fk),
        ("{{FWG}}", fq * parts),
        ("{{BBP}}", bb + 1),
        ("{{BKPT}}", bb / parts),
        ("{{BWG}}", bb * parts),
        ("{{FQ_SP}}", fq * sp),
        ("{{FK_SP}}", fk * sp),
        ("{{FQ_FKP}}", fq * (fk + 1)),
        ("{{FQ_PARTS}}", fq * parts),
        ("{{BB_SP}}", bb * sp),
        ("{{BB_BBP}}", bb * (bb + 1)),
        ("{{BB2}}", 2 * bb),
        ("{{BB}}", bb),
        ("{{PARTS}}", parts),
        ("{{DP}}", dp),
    ];
    let mut body = ATTENTION.to_string();
    for (name, value) in words {
        body = body.replace(name, &value.to_string());
    }
    Ok(body)
}

/// Full source for `module`. The attention template is checked against the
/// head-dimension cap and its own tile rule; it is never truncated to fit.
pub fn wgsl_module(module: WgslModule) -> Result<String, OjasError> {
    let mut src = String::from(COMMON);
    match module {
        WgslModule::Gemm => src.push_str(GEMM),
        WgslModule::Pointwise => src.push_str(POINTWISE),
        WgslModule::Reduce => {
            src.push_str(TREE);
            src.push_str(REDUCE);
        }
        WgslModule::Norm => {
            src.push_str(TREE);
            src.push_str(NORM);
        }
        WgslModule::Loss => {
            src.push_str(TREE);
            src.push_str(LOSS);
        }
        WgslModule::Optim => {
            src.push_str(TREE);
            src.push_str(OPTIM);
        }
        WgslModule::Layout => src.push_str(LAYOUT),
        WgslModule::Fault => src.push_str(FAULT),
        WgslModule::Attention(tiles) => src.push_str(&attention_source(tiles)?),
        WgslModule::CachedAttention => src.push_str(CACHED_ATTENTION),
    }
    Ok(src)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::attention_tiles;

    #[test]
    fn every_module_carries_the_header_and_no_placeholder() {
        let tiles = attention_tiles(64, 32 * 1024).unwrap();
        for module in [
            WgslModule::Gemm,
            WgslModule::Pointwise,
            WgslModule::Reduce,
            WgslModule::Norm,
            WgslModule::Loss,
            WgslModule::Optim,
            WgslModule::Layout,
            WgslModule::Fault,
            WgslModule::Attention(tiles),
            WgslModule::CachedAttention,
        ] {
            let src = wgsl_module(module).unwrap();
            assert!(
                src.contains("@binding(1) var<storage, read_write> fault"),
                "{module:?}"
            );
            assert!(!src.contains("{{"), "{module:?} left a placeholder");
        }
    }

    #[test]
    fn cached_attention_kernel_agrees_with_the_host_plan() {
        use crate::geometry::{CACHED_ATTENTION_LANES, CACHED_ATTENTION_MAX_SPLIT};
        let src = wgsl_module(WgslModule::CachedAttention).unwrap();
        for needle in [
            format!("const LANES: u32 = {CACHED_ATTENTION_LANES}u;"),
            format!("const MAX_SPLIT: u32 = {CACHED_ATTENTION_MAX_SPLIT}u;"),
            format!("const MAX_D: u32 = {ATTENTION_MAX_HEAD_DIM}u;"),
            format!("@workgroup_size({CACHED_ATTENTION_LANES}, 1, 1)"),
            format!("@workgroup_size({ATTENTION_MAX_HEAD_DIM}, 1, 1)"),
        ] {
            assert!(src.contains(&needle), "missing `{needle}`");
        }
    }

    #[test]
    fn attention_template_refuses_bad_shapes() {
        let good = attention_tiles(64, 32 * 1024).unwrap();
        for bad in [
            AttentionTiles {
                padded_dim: 48,
                ..good
            },
            AttentionTiles {
                padded_dim: 512,
                ..good
            },
            AttentionTiles {
                fwd_rows: 6,
                ..good
            },
            AttentionTiles {
                fwd_keys: 0,
                ..good
            },
            AttentionTiles {
                bwd_block: 64,
                ..good
            },
        ] {
            assert!(wgsl_module(WgslModule::Attention(bad)).is_err(), "{bad:?}");
        }
        for d in [16u32, 32, 64, 128, 129, 192, 256] {
            for cap in [16 * 1024, 32 * 1024] {
                let t = attention_tiles(d, cap).unwrap();
                let src = wgsl_module(WgslModule::Attention(t)).unwrap();
                assert!(!src.contains("{{"), "{t:?}");
            }
        }
    }
}

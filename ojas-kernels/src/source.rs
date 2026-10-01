//! Kernel source text. Backends compile these strings. HIP is not registered:
//! the CUDA-C source is written so a future hiprtc path can reuse it unchanged.

/// Ops the parity harness knows about. Each one is a name, not a GPU type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelId {
    Affine,
    Gemm,
    ReduceSum,
}

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

/// Tiled GEMM source for a future launch. `k` runs in ascending order.
/// `fmad` is left off by the caller. This text is not registered for HIP.
pub fn gemm_cuda() -> &'static str {
    r#"
extern "C" __global__ void gemm_nt_f32(
    float* out,
    const float* a,
    const float* b,
    unsigned int m,
    unsigned int k,
    unsigned int n
) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) {
        return;
    }
    float acc = 0.0f;
    for (unsigned int inner = 0; inner < k; ++inner) {
        acc += a[row * k + inner] * b[inner * n + col];
    }
    out[row * n + col] = acc;
}
"#
}

pub fn reduce_sum_wgsl() -> &'static str {
    r#"
@group(0) @binding(0) var<storage, read> input_values: array<f32>;
@group(0) @binding(1) var<storage, read_write> partial: array<f32>;

var<workgroup> tile: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let n = arrayLength(&input_values);
    var acc = 0.0;
    var index = wgid.x * 64u + lid.x;
    let stride = 64u * 65535u;
    loop {
        if (index >= n) {
            break;
        }
        acc = acc + input_values[index];
        let next = index + stride;
        if (next <= index) {
            break;
        }
        index = next;
    }
    tile[lid.x] = acc;
    workgroupBarrier();
    var offset = 32u;
    loop {
        if (offset == 0u) {
            break;
        }
        if (lid.x < offset) {
            tile[lid.x] = tile[lid.x] + tile[lid.x + offset];
        }
        workgroupBarrier();
        offset = offset / 2u;
    }
    if (lid.x == 0u) {
        partial[wgid.x] = tile[0];
    }
}
"#
}

/// Math kernels for the portable GPU path. `gemm_tiled` walks k ascending.
/// Row reductions use a 64-thread tree. The source names no GPU API type.
pub const fn math_wgsl() -> &'static str {
    include_str!("math.wgsl")
}

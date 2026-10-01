//! One adapter, one device, cached pipelines, and a size-keyed buffer pool.
//!
//! [`WgpuContext`] is the performance path: device buffers stay resident, a
//! [`Pass`] records several dispatches, and `poll` runs only when the pass
//! submits, with that submit's [`wgpu::SubmissionIndex`].
//!
//! [`crate::WgpuBackend`] uploads and reads back on every trait method. That
//! matches the host-tensor `Backend` signature. It is the parity path, not
//! this one.
//!
//! wgpu does not disable Metal's default contraction, so CPU parity uses a
//! tolerance. Two runs of the same dispatch on this context are bit-identical.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use ojas_device::{Device, DeviceError};
use ojas_kernels::math_wgsl;

use crate::{bytes_to_f32, f32_bytes, gpu_error, hal_backends, info_of};

struct Cache {
    math: wgpu::ShaderModule,
    pipelines: HashMap<&'static str, wgpu::ComputePipeline>,
    /// Exact-size storage buffers waiting to be reused.
    free: HashMap<(u64, u32), Vec<wgpu::Buffer>>,
    compiles: u64,
    reuses: u64,
}

/// Persistent portable GPU context.
#[derive(Clone)]
pub struct WgpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pub adapter_name: String,
    pub hal: String,
    pub vendor: String,
    limits: wgpu::Limits,
    inner: std::sync::Arc<Mutex<Cache>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheStats {
    pub compiles: u64,
    pub reuses: u64,
}

/// Device-resident f32 buffer. The bytes stay on the GPU until a pass reads them.
pub struct DeviceTensor {
    buf: wgpu::Buffer,
    len: usize,
    usage_bits: u32,
    bytes: u64,
}

impl WgpuContext {
    pub fn open() -> Result<Self, DeviceError> {
        let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle();
        instance_desc.backends = hal_backends();
        let instance = wgpu::Instance::new(instance_desc);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        }))
        .map_err(|err| gpu_error(format!("request_adapter: {err}")))?;
        let info = adapter.get_info();
        if info.device_type == wgpu::DeviceType::Cpu {
            return Err(gpu_error(format!(
                "refusing CPU adapter {} ({:?})",
                info.name, info.backend
            )));
        }
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("ojas-wgpu"),
            required_features: wgpu::Features::empty(),
            required_limits: limits.clone(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .map_err(|err| gpu_error(format!("request_device: {err}")))?;
        let math = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ojas-math"),
            source: wgpu::ShaderSource::Wgsl(math_wgsl().into()),
        });
        let (adapter_name, hal, vendor) = info_of(&info);
        Ok(Self {
            device,
            queue,
            adapter_name,
            hal,
            vendor,
            limits,
            inner: std::sync::Arc::new(Mutex::new(Cache {
                math,
                pipelines: HashMap::new(),
                free: HashMap::new(),
                compiles: 0,
                reuses: 0,
            })),
        })
    }

    /// Process-wide context. The first caller opens the device; later callers clone handles.
    pub fn shared() -> Result<Self, DeviceError> {
        static CELL: OnceLock<Result<WgpuContext, String>> = OnceLock::new();
        match CELL.get_or_init(|| Self::open().map_err(|err| err.to_string())) {
            Ok(ctx) => Ok(ctx.clone()),
            Err(err) => Err(gpu_error(err.clone())),
        }
    }

    pub fn stats(&self) -> CacheStats {
        let guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        CacheStats {
            compiles: guard.compiles,
            reuses: guard.reuses,
        }
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn limits(&self) -> &wgpu::Limits {
        &self.limits
    }

    fn pipeline(&self, entry: &'static str) -> Result<wgpu::ComputePipeline, DeviceError> {
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(hit) = guard.pipelines.get(entry) {
            return Ok(hit.clone());
        }
        let validation = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &guard.math,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        if let Some(err) = pollster::block_on(validation.pop()) {
            return Err(DeviceError::Compile {
                kind: Device::Vulkan,
                detail: format!("{entry}: {err}"),
            });
        }
        guard.compiles += 1;
        guard.pipelines.insert(entry, pipeline.clone());
        Ok(pipeline)
    }

    pub fn alloc_f32(
        &self,
        len: usize,
        usage: wgpu::BufferUsages,
    ) -> Result<DeviceTensor, DeviceError> {
        let bytes = u64::try_from(len.saturating_mul(4)).unwrap_or(u64::MAX);
        if bytes == 0 {
            return Err(DeviceError::Capacity {
                kind: Device::Vulkan,
                detail: "refusing a zero-length device buffer".to_string(),
            });
        }
        let cap = self
            .limits
            .max_storage_buffer_binding_size
            .min(self.limits.max_buffer_size);
        if bytes > cap {
            return Err(DeviceError::Capacity {
                kind: Device::Vulkan,
                detail: format!("{bytes} bytes exceed the storage limit {cap}"),
            });
        }
        let bits = usage.bits();
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let buf = if let Some(pool) = guard.free.get_mut(&(bytes, bits)) {
            if let Some(buf) = pool.pop() {
                guard.reuses += 1;
                buf
            } else {
                drop(guard);
                self.make_buffer(bytes, usage)
            }
        } else {
            drop(guard);
            self.make_buffer(bytes, usage)
        };
        Ok(DeviceTensor {
            buf,
            len,
            usage_bits: bits,
            bytes,
        })
    }

    fn make_buffer(&self, bytes: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ojas-pool"),
            size: bytes,
            usage,
            mapped_at_creation: false,
        })
    }

    pub fn recycle(&self, tensor: DeviceTensor) {
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        guard
            .free
            .entry((tensor.bytes, tensor.usage_bits))
            .or_default()
            .push(tensor.buf);
    }

    pub fn upload(&self, tensor: &DeviceTensor, values: &[f32]) -> Result<(), DeviceError> {
        if values.len() != tensor.len {
            return Err(DeviceError::Capacity {
                kind: Device::Vulkan,
                detail: format!(
                    "upload {} values into a buffer of {}",
                    values.len(),
                    tensor.len
                ),
            });
        }
        self.queue.write_buffer(&tensor.buf, 0, &f32_bytes(values));
        Ok(())
    }

    pub fn begin(&self) -> Pass<'_> {
        Pass {
            ctx: self,
            encoder: self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("ojas-pass"),
                }),
            kept_groups: Vec::new(),
            kept_buffers: Vec::new(),
        }
    }

    pub(crate) fn wait(&self, index: &wgpu::SubmissionIndex) -> Result<(), DeviceError> {
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(index.clone()),
                timeout: Some(std::time::Duration::from_secs(30)),
            })
            .map(|_| ())
            .map_err(|err| gpu_error(format!("poll: {err}")))
    }
}

impl DeviceTensor {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// A sequence of dispatches submitted together. Bind groups stay alive until submit.
pub struct Pass<'a> {
    ctx: &'a WgpuContext,
    encoder: wgpu::CommandEncoder,
    kept_groups: Vec<wgpu::BindGroup>,
    kept_buffers: Vec<wgpu::Buffer>,
}

impl Pass<'_> {
    pub fn math(
        &mut self,
        entry: &'static str,
        a: &DeviceTensor,
        b: &DeviceTensor,
        out: &DeviceTensor,
        params: [u32; 4],
        groups: (u32, u32, u32),
    ) -> Result<(), DeviceError> {
        let pipeline = self.ctx.pipeline(entry)?;
        let uniform = self.ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut raw = [0u8; 16];
        for (i, word) in params.iter().enumerate() {
            raw[i * 4..i * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        self.ctx.queue().write_buffer(&uniform, 0, &raw);
        let bind = self
            .ctx
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(entry),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: a.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: b.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: out.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: uniform.as_entire_binding(),
                    },
                ],
            });
        {
            let mut compute = self
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(entry),
                    timestamp_writes: None,
                });
            compute.set_pipeline(&pipeline);
            compute.set_bind_group(0, &bind, &[]);
            compute.dispatch_workgroups(groups.0, groups.1, groups.2);
        }
        self.kept_groups.push(bind);
        self.kept_buffers.push(uniform);
        Ok(())
    }

    /// Submit once and read `out`. The wait is this submission's index.
    pub fn submit_read(self, out: &DeviceTensor) -> Result<Vec<f32>, DeviceError> {
        let staging = self.ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: out.bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.encoder;
        encoder.copy_buffer_to_buffer(&out.buf, 0, &staging, 0, out.bytes);
        let index = self.ctx.queue().submit(std::iter::once(encoder.finish()));
        drop(self.kept_groups);
        drop(self.kept_buffers);
        self.ctx.wait(&index)?;
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.ctx.wait(&index)?;
        receiver
            .try_recv()
            .map_err(|_| gpu_error("map callback did not run"))?
            .map_err(|err| gpu_error(format!("map_async: {err}")))?;
        let mapped = slice
            .get_mapped_range()
            .map_err(|err| gpu_error(format!("get_mapped_range: {err}")))?;
        let values = bytes_to_f32(&mapped)?;
        drop(mapped);
        staging.unmap();
        if values.len() != out.len {
            return Err(gpu_error(format!(
                "mapped {} values, expected {}",
                values.len(),
                out.len
            )));
        }
        Ok(values)
    }
}

/// `C = A @ B` with `A` row-major `[m, k]` and `B` row-major `[k, n]`.
pub fn gemm(
    ctx: &WgpuContext,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, DeviceError> {
    if a.len() != m * k || b.len() != k * n {
        return Err(gpu_error("gemm length does not match shape"));
    }
    let storage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let a_t = ctx.alloc_f32(a.len(), storage)?;
    let b_t = ctx.alloc_f32(b.len().max(1), storage)?;
    let c_t = ctx.alloc_f32(m * n, storage)?;
    ctx.upload(&a_t, a)?;
    if b.is_empty() {
        return Err(gpu_error("gemm k is 0"));
    }
    ctx.upload(&b_t, b)?;
    let groups = ((n as u32).div_ceil(16), (m as u32).div_ceil(16), 1);
    let mut pass = ctx.begin();
    pass.math(
        "gemm_tiled",
        &a_t,
        &b_t,
        &c_t,
        [m as u32, k as u32, n as u32, 0],
        groups,
    )?;
    let out = pass.submit_read(&c_t)?;
    ctx.recycle(a_t);
    ctx.recycle(b_t);
    ctx.recycle(c_t);
    Ok(out)
}

fn element_groups(n: usize) -> (u32, u32, u32) {
    ((n as u32).div_ceil(64), 1, 1)
}

pub fn silu(ctx: &WgpuContext, x: &[f32]) -> Result<Vec<f32>, DeviceError> {
    unary(ctx, "silu", x)
}

pub fn mul(ctx: &WgpuContext, a: &[f32], b: &[f32]) -> Result<Vec<f32>, DeviceError> {
    binary(ctx, "mul", a, b)
}

pub fn residual(ctx: &WgpuContext, a: &[f32], b: &[f32]) -> Result<Vec<f32>, DeviceError> {
    binary(ctx, "residual", a, b)
}

pub fn softmax_rows(
    ctx: &WgpuContext,
    rows: usize,
    cols: usize,
    x: &[f32],
) -> Result<Vec<f32>, DeviceError> {
    if x.len() != rows * cols || cols > 4096 {
        return Err(DeviceError::Capacity {
            kind: Device::Vulkan,
            detail: "softmax shape is not a dense matrix within 4096 columns".to_string(),
        });
    }
    unary_shaped(
        ctx,
        "softmax_row",
        x,
        x,
        [rows as u32, cols as u32, 0, 0],
        (rows as u32, 1, 1),
    )
}

pub fn row_sum(
    ctx: &WgpuContext,
    rows: usize,
    cols: usize,
    x: &[f32],
) -> Result<Vec<f32>, DeviceError> {
    if x.len() != rows * cols {
        return Err(gpu_error("row sum shape mismatch"));
    }
    let storage = store_usage();
    let a_t = ctx.alloc_f32(x.len(), storage)?;
    let dummy = ctx.alloc_f32(1, storage)?;
    ctx.upload(&dummy, &[0.0])?;
    let out_t = ctx.alloc_f32(rows, storage)?;
    ctx.upload(&a_t, x)?;
    let mut pass = ctx.begin();
    pass.math(
        "row_sum",
        &a_t,
        &dummy,
        &out_t,
        [rows as u32, cols as u32, 0, 0],
        (rows as u32, 1, 1),
    )?;
    let out = pass.submit_read(&out_t)?;
    ctx.recycle(a_t);
    ctx.recycle(dummy);
    ctx.recycle(out_t);
    Ok(out)
}

pub fn rms_norm(
    ctx: &WgpuContext,
    rows: usize,
    cols: usize,
    x: &[f32],
    weight: &[f32],
    eps: f32,
) -> Result<Vec<f32>, DeviceError> {
    if x.len() != rows * cols || weight.len() != cols {
        return Err(gpu_error("rms shape mismatch"));
    }
    unary_shaped(
        ctx,
        "rms_norm",
        x,
        weight,
        [rows as u32, cols as u32, eps.to_bits(), 0],
        element_groups(rows),
    )
}

fn store_usage() -> wgpu::BufferUsages {
    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST
}

fn unary(ctx: &WgpuContext, entry: &'static str, x: &[f32]) -> Result<Vec<f32>, DeviceError> {
    unary_shaped(
        ctx,
        entry,
        x,
        x,
        [x.len() as u32, 0, 0, 0],
        element_groups(x.len()),
    )
}

fn binary(
    ctx: &WgpuContext,
    entry: &'static str,
    a: &[f32],
    b: &[f32],
) -> Result<Vec<f32>, DeviceError> {
    if a.len() != b.len() {
        return Err(gpu_error("binary operands differ in length"));
    }
    unary_shaped(
        ctx,
        entry,
        a,
        b,
        [a.len() as u32, 0, 0, 0],
        element_groups(a.len()),
    )
}

fn unary_shaped(
    ctx: &WgpuContext,
    entry: &'static str,
    a: &[f32],
    b: &[f32],
    params: [u32; 4],
    groups: (u32, u32, u32),
) -> Result<Vec<f32>, DeviceError> {
    let storage = store_usage();
    let a_t = ctx.alloc_f32(a.len(), storage)?;
    let b_t = ctx.alloc_f32(b.len().max(1), storage)?;
    let out_t = ctx.alloc_f32(a.len(), storage)?;
    ctx.upload(&a_t, a)?;
    if !b.is_empty() {
        ctx.upload(&b_t, b)?;
    }
    let mut pass = ctx.begin();
    pass.math(entry, &a_t, &b_t, &out_t, params, groups)?;
    let out = pass.submit_read(&out_t)?;
    ctx.recycle(a_t);
    ctx.recycle(b_t);
    ctx.recycle(out_t);
    Ok(out)
}

/// Record GEMM then SiLU and read once. The buffers stay on the device between the two ops.
pub fn gemm_then_silu(
    ctx: &WgpuContext,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, DeviceError> {
    let storage = store_usage();
    let a_t = ctx.alloc_f32(a.len(), storage)?;
    let b_t = ctx.alloc_f32(b.len(), storage)?;
    let c_t = ctx.alloc_f32(m * n, storage)?;
    let s_t = ctx.alloc_f32(m * n, storage)?;
    ctx.upload(&a_t, a)?;
    ctx.upload(&b_t, b)?;
    let mut pass = ctx.begin();
    pass.math(
        "gemm_tiled",
        &a_t,
        &b_t,
        &c_t,
        [m as u32, k as u32, n as u32, 0],
        ((n as u32).div_ceil(16), (m as u32).div_ceil(16), 1),
    )?;
    pass.math(
        "silu",
        &c_t,
        &c_t,
        &s_t,
        [(m * n) as u32, 0, 0, 0],
        element_groups(m * n),
    )?;
    let out = pass.submit_read(&s_t)?;
    ctx.recycle(a_t);
    ctx.recycle(b_t);
    ctx.recycle(c_t);
    ctx.recycle(s_t);
    Ok(out)
}

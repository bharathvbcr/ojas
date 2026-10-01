//! Portable GPU backend.
//!
//! The shader is ours. wgpu only compiles WGSL and submits the dispatch.
//! On Apple the HAL is Metal. The device kind stays [`Device::Vulkan`]
//! so this path does not stand in for tessl. A CPU adapter is refused.
//! A failed adapter request is [`DeviceError::NoDevice`], never a CPU result.

#![forbid(unsafe_code)]

use ojas_device::{require_kind, DeviceError, DeviceInfo, Device};

const AFFINE_SHADER: &str = r#"
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
"#;

/// Lanes per workgroup in `AFFINE_SHADER`.
const WORKGROUP: u32 = 64;

/// Workgroup grid `(x, y)` covering `n` lanes when each dimension is capped
/// at `max_per_dim`. Rows past the first are full `x` rows; the shader guards
/// the tail.
fn workgroup_grid(n: u32, max_per_dim: u32) -> Result<(u32, u32), DeviceError> {
    let groups = n.div_ceil(WORKGROUP);
    if max_per_dim == 0 {
        return Err(gpu_error(
            "device reports a workgroup-per-dimension limit of 0",
        ));
    }
    if groups <= max_per_dim {
        return Ok((groups, 1));
    }
    let rows = groups.div_ceil(max_per_dim);
    if rows > max_per_dim {
        return Err(gpu_error(format!(
            "{n} lanes need {rows} workgroup rows, past the device limit {max_per_dim}"
        )));
    }
    Ok((max_per_dim, rows))
}

/// Result of `y = x * scale + bias` on the selected adapter.
#[derive(Clone, Debug, PartialEq)]
pub struct AffineF32 {
    pub values: Vec<f32>,
    pub adapter_name: String,
    pub hal: String,
    pub vendor: String,
    /// False when the buffer was empty and no compute pass was submitted.
    pub dispatched: bool,
}

/// Byte length of `len` f32 values.
///
/// `len == 0` is 0. A product that does not fit in `u32` is
/// [`DeviceError::NoDevice`]: the dispatch index is a `u32` and is not clamped.
pub fn f32_byte_len(len: usize) -> Result<u32, DeviceError> {
    let bytes = len.checked_mul(4).ok_or_else(|| {
        gpu_error(format!(
            "byte length of {len} f32 values does not fit in u32"
        ))
    })?;
    u32::try_from(bytes).map_err(|_| {
        gpu_error(format!(
            "byte length {bytes} of {len} f32 values does not fit in u32"
        ))
    })
}

/// HAL wgpu is asked to open.
///
/// Apple uses the Metal HAL so this machine can run the shader.
/// Other hosts use Vulkan. DX12 is compiled into wgpu's default features
/// and is not requested here.
fn hal_backends() -> wgpu::Backends {
    if cfg!(target_os = "macos") || cfg!(target_os = "ios") {
        wgpu::Backends::METAL
    } else {
        wgpu::Backends::VULKAN
    }
}

fn vendor_name(vendor: u32, adapter_name: &str) -> String {
    match vendor {
        0x106b => "Apple".to_string(),
        0x10de => "NVIDIA".to_string(),
        0x1002 => "AMD".to_string(),
        0x8086 => "Intel".to_string(),
        // wgpu's Metal HAL left the PCI vendor field at 0 on this machine.
        // The adapter name is still the name wgpu returned.
        0 if adapter_name.starts_with("Apple") => "Apple (wgpu vendor field 0)".to_string(),
        0 => "vendor field 0".to_string(),
        other => format!("pci 0x{other:04x}"),
    }
}

fn gpu_error(detail: impl std::fmt::Display) -> DeviceError {
    DeviceError::NoDevice {
        kind: Device::Vulkan,
        detail: detail.to_string(),
    }
}

struct Session {
    device: wgpu::Device,
    queue: wgpu::Queue,
    info: wgpu::AdapterInfo,
}

fn open_session() -> Result<Session, DeviceError> {
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
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("ojas-wgpu"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .map_err(|err| gpu_error(format!("request_device: {err}")))?;
    Ok(Session { device, queue, info })
}

fn info_of(info: &wgpu::AdapterInfo) -> (String, String, String) {
    (
        info.name.clone(),
        format!("{:?}", info.backend),
        vendor_name(info.vendor, &info.name),
    )
}

/// Adapters wgpu can open for the portable path.
///
/// Failure is an error. An empty success would look like "no GPU, carry on".
pub fn probe() -> Result<Vec<DeviceInfo>, DeviceError> {
    let session = open_session()?;
    let (name, _hal, vendor) = info_of(&session.info);
    Ok(vec![DeviceInfo {
        name,
        vendor,
        backend: Device::Vulkan,
    }])
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn bytes_to_f32(bytes: &[u8]) -> Result<Vec<f32>, DeviceError> {
    if bytes.len() % 4 != 0 {
        return Err(gpu_error(format!(
            "mapped buffer length {} is not a multiple of 4",
            bytes.len()
        )));
    }
    let mut values = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let mut le = [0u8; 4];
        le.copy_from_slice(chunk);
        values.push(f32::from_le_bytes(le));
    }
    Ok(values)
}

/// `y[i] = x[i] * scale + bias` on `kind`.
///
/// `kind` must be [`Device::Vulkan`]. Any other kind is
/// [`DeviceError::DeviceMismatch`] and does not touch a GPU.
/// A NaN `scale` or `bias` propagates. An empty buffer returns an empty
/// vector and does not submit a compute pass. A byte length that does not
/// fit in `u32`, or that exceeds the device's storage binding limit, returns
/// [`DeviceError::NoDevice`] and does not allocate. A wgpu validation,
/// internal, or out-of-memory error is returned, not raised as a panic.
pub fn affine_f32(
    kind: Device,
    input: &[f32],
    scale: f32,
    bias: f32,
) -> Result<AffineF32, DeviceError> {
    require_kind(Device::Vulkan, kind)?;
    let bytes_u32 = f32_byte_len(input.len())?;
    let session = open_session()?;
    let (adapter_name, hal, vendor) = info_of(&session.info);
    if bytes_u32 == 0 {
        return Ok(AffineF32 {
            values: Vec::new(),
            adapter_name,
            hal,
            vendor,
            dispatched: false,
        });
    }
    let values = dispatch_affine(&session, input, scale, bias, bytes_u32)?;
    Ok(AffineF32 {
        values,
        adapter_name,
        hal,
        vendor,
        dispatched: true,
    })
}

fn dispatch_affine(
    session: &Session,
    input: &[f32],
    scale: f32,
    bias: f32,
    bytes_u32: u32,
) -> Result<Vec<f32>, DeviceError> {
    let n = u32::try_from(input.len()).map_err(|_| gpu_error("input length does not fit in u32"))?;
    let byte_len = u64::from(bytes_u32);
    let limits = session.device.limits();
    let cap = limits
        .max_storage_buffer_binding_size
        .min(limits.max_buffer_size);
    if byte_len > cap {
        return Err(gpu_error(format!(
            "{byte_len} bytes exceed the device storage binding limit {cap}"
        )));
    }
    let (groups_x, groups_y) = workgroup_grid(n, limits.max_compute_workgroups_per_dimension)?;
    let oom_scope = session
        .device
        .push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let internal_scope = session.device.push_error_scope(wgpu::ErrorFilter::Internal);
    let validation_scope = session
        .device
        .push_error_scope(wgpu::ErrorFilter::Validation);
    let shader = session
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ojas-affine-f32"),
            source: wgpu::ShaderSource::Wgsl(AFFINE_SHADER.into()),
        });
    let pipeline = session
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ojas-affine-f32"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
    let input_buf = session.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("input"),
        size: byte_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let output_buf = session.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output"),
        size: byte_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut params = [0u8; 16];
    params[..4].copy_from_slice(&scale.to_le_bytes());
    params[4..8].copy_from_slice(&bias.to_le_bytes());
    let params_buf = session.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("params"),
        size: params.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let staging = session.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: byte_len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    session
        .queue
        .write_buffer(&input_buf, 0, &f32_bytes(input));
    session.queue.write_buffer(&params_buf, 0, &params);

    let bind_group = session
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scale"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
    let mut encoder = session
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("scale"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("scale"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(groups_x, groups_y, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buf, 0, &staging, 0, byte_len);
    session.queue.submit(std::iter::once(encoder.finish()));
    for (kind, scope) in [
        ("validation", validation_scope),
        ("internal", internal_scope),
        ("out of memory", oom_scope),
    ] {
        if let Some(err) = pollster::block_on(scope.pop()) {
            return Err(gpu_error(format!("wgpu {kind} error: {err}")));
        }
    }

    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    session
        .device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(30)),
        })
        .map_err(|err| gpu_error(format!("poll: {err}")))?;
    receiver
        .try_recv()
        .map_err(|_| gpu_error("map callback did not run after the poll returned"))?
        .map_err(|err| gpu_error(format!("map_async: {err}")))?;
    let mapped = slice
        .get_mapped_range()
        .map_err(|err| gpu_error(format!("get_mapped_range: {err}")))?;
    let values = bytes_to_f32(&mapped)?;
    drop(mapped);
    staging.unmap();
    if values.len() != input.len() {
        return Err(gpu_error(format!(
            "mapped {} values, expected {}",
            values.len(),
            input.len()
        )));
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metal_kind_is_not_this_backend() {
        let err = affine_f32(Device::Metal, &[1.0], 2.0, 0.0).unwrap_err();
        assert!(matches!(
            err,
            DeviceError::DeviceMismatch {
                expected: Device::Vulkan,
                actual: Device::Metal,
            }
        ));
    }

    #[test]
    fn cpu_kind_is_refused() {
        let err = affine_f32(Device::Cpu, &[1.0, 2.0], 3.0, 0.0).unwrap_err();
        assert!(matches!(err, DeviceError::DeviceMismatch { .. }));
    }

    #[test]
    fn cuda_kind_is_not_a_wgpu_run() {
        let err = affine_f32(Device::Cuda, &[1.0], 2.0, 1.0).unwrap_err();
        assert!(matches!(
            err,
            DeviceError::DeviceMismatch {
                expected: Device::Vulkan,
                actual: Device::Cuda,
            }
        ));
    }

    #[test]
    fn byte_len_past_u32_is_an_error() {
        let lens = [
            (u32::MAX as usize / 4) + 1,
            1usize << 32,
            (1usize << 32) + 3,
            usize::MAX / 2,
            usize::MAX,
        ];
        for len in lens {
            let err = f32_byte_len(len).expect_err("length that cannot fit in u32 returned Ok");
            let text = err.to_string();
            assert!(text.contains("u32"), "{len}: {text}");
            assert!(matches!(err, DeviceError::NoDevice { .. }), "{len}: {err}");
            assert_ne!(f32_byte_len(len).ok(), Some(0), "{len} collapsed to 0");
        }
        assert_eq!(f32_byte_len(0).unwrap(), 0);
        assert_eq!(f32_byte_len(1).unwrap(), 4);
    }

    #[test]
    fn affine_matches_cpu_formula_on_the_adapter() {
        let input: Vec<f32> = (0..128).map(|i| (i as f32) - 8.0).collect();
        let scale = 2.0f32;
        let bias = -0.5f32;
        let ran = affine_f32(Device::Vulkan, &input, scale, bias)
            .expect("wgpu adapter request failed; this test must fail rather than skip");
        eprintln!(
            "ojas-wgpu adapter: {} vendor={} hal={}",
            ran.adapter_name, ran.vendor, ran.hal
        );
        assert!(ran.dispatched);
        assert!(!ran.adapter_name.is_empty(), "adapter name was empty");
        assert_eq!(ran.values.len(), input.len());
        for (index, (got, src)) in ran.values.iter().zip(input.iter()).enumerate() {
            let expect = src * scale + bias;
            let delta = (got - expect).abs();
            assert!(
                delta <= 1e-6,
                "index {index}: gpu {got} vs cpu {expect} (adapter {}, hal {})",
                ran.adapter_name, ran.hal
            );
        }
    }

    #[test]
    fn empty_buffer_is_ok_and_does_not_dispatch() {
        let ran = affine_f32(Device::Vulkan, &[], 2.0, 1.0)
            .expect("missing adapter must fail, not skip");
        eprintln!(
            "ojas-wgpu empty adapter: {} hal={}",
            ran.adapter_name, ran.hal
        );
        assert!(ran.values.is_empty());
        assert!(!ran.dispatched);
    }

    fn ramp(n: usize, seed: u32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let x = (i as u32).wrapping_mul(2654435761).wrapping_add(seed) % 2001;
                x as f32 / 1000.0 - 1.0
            })
            .collect()
    }

    fn assert_affine(input: &[f32], scale: f32, bias: f32) {
        let ran = affine_f32(Device::Vulkan, input, scale, bias)
            .unwrap_or_else(|err| panic!("len {}: {err}", input.len()));
        assert!(ran.dispatched, "len {}", input.len());
        assert_eq!(ran.values.len(), input.len());
        for (index, (got, src)) in ran.values.iter().zip(input).enumerate() {
            let expect = src * scale + bias;
            assert!(
                (got - expect).abs() <= 1e-6,
                "len {} index {index}: gpu {got} vs cpu {expect}",
                input.len()
            );
        }
    }

    #[test]
    fn affine_matches_cpu_at_odd_lengths() {
        for (seed, n) in [1usize, 2, 63, 64, 65, 1023, 1024, 1025, 100_003]
            .into_iter()
            .enumerate()
        {
            assert_affine(&ramp(n, seed as u32), 1.5, -0.25);
        }
    }

    #[test]
    fn affine_past_one_row_of_workgroups_matches_cpu() {
        // 65_535 workgroups of 64 is the default per-dimension cap.
        let n = 65_535 * 64 + 65;
        assert_affine(&ramp(n, 7), -3.0, 0.5);
    }

    #[test]
    fn affine_at_the_binding_limit_matches_cpu() {
        let n = (wgpu::Limits::default().max_storage_buffer_binding_size / 4) as usize;
        assert_affine(&ramp(n, 9), 0.5, 2.0);
    }

    #[test]
    fn affine_past_the_binding_limit_is_an_error() {
        let n = (wgpu::Limits::default().max_storage_buffer_binding_size / 4) as usize + 1;
        let input = vec![1.0f32; n];
        let err = affine_f32(Device::Vulkan, &input, 1.0, 0.0).unwrap_err();
        assert!(matches!(err, DeviceError::NoDevice { .. }), "{err}");
        assert!(err.to_string().contains("limit"), "{err}");
    }

    #[test]
    fn workgroup_grid_covers_every_lane_within_the_cap() {
        assert_eq!(workgroup_grid(1, 65_535).unwrap(), (1, 1));
        assert_eq!(workgroup_grid(64, 65_535).unwrap(), (1, 1));
        assert_eq!(workgroup_grid(65, 65_535).unwrap(), (2, 1));
        assert_eq!(workgroup_grid(64 * 4, 2).unwrap(), (2, 2));
        assert!(workgroup_grid(64 * 4 + 1, 2)
            .unwrap_err()
            .to_string()
            .contains("limit"));
        assert!(workgroup_grid(1, 0).is_err());
        for n in [1u32, 63, 65, 1023, 1025, 4_194_241, u32::MAX / 4] {
            let (x, y) = workgroup_grid(n, 65_535).unwrap();
            assert!(x <= 65_535 && y <= 65_535, "{n}");
            assert!(u64::from(x) * u64::from(y) * 64 >= u64::from(n), "{n}");
            assert!(u64::from(x) * u64::from(y - 1) * 64 < u64::from(n), "{n}");
        }
    }

    #[test]
    fn non_finite_inputs_propagate() {
        let input = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX, -0.0, 1.0];
        let ran = affine_f32(Device::Vulkan, &input, 2.0, 0.0).unwrap();
        assert!(ran.values[0].is_nan());
        assert_eq!(ran.values[1], f32::INFINITY);
        assert_eq!(ran.values[2], f32::NEG_INFINITY);
        assert_eq!(ran.values[3], f32::INFINITY);
        assert_eq!(ran.values[4].to_bits(), 0.0f32.to_bits());
        assert_eq!(ran.values[5], 2.0);
    }

    #[test]
    fn repeated_dispatches_are_deterministic() {
        let input = ramp(1025, 3);
        let first = affine_f32(Device::Vulkan, &input, 1.25, -0.75).unwrap().values;
        assert!(first.iter().all(|v| v.is_finite()));
        for i in 0..1000 {
            let got = affine_f32(Device::Vulkan, &input, 1.25, -0.75).unwrap().values;
            assert!(
                got.iter().zip(&first).all(|(a, b)| a.to_bits() == b.to_bits()),
                "dispatch {i} differs"
            );
        }
    }

    #[test]
    fn concurrent_calls_from_threads_match_cpu() {
        let handles: Vec<_> = (0..8u32)
            .map(|t| {
                std::thread::spawn(move || {
                    for round in 0..8u32 {
                        let n = 63 + 1000 * t as usize + round as usize;
                        assert_affine(&ramp(n, t * 31 + round), t as f32 - 3.5, round as f32);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().expect("worker thread panicked");
        }
    }

    fn assert_real_adapter(ran: &AffineF32) {
        assert!(
            !ran.adapter_name.is_empty(),
            "adapter name was empty; a missing adapter must fail the test"
        );
        assert!(
            ran.adapter_name.contains("Apple"),
            "expected the Apple GPU, got {:?} vendor={} hal={}",
            ran.adapter_name,
            ran.vendor,
            ran.hal
        );
        eprintln!(
            "ojas-wgpu adapter: {} vendor={} hal={}",
            ran.adapter_name, ran.vendor, ran.hal
        );
    }

    #[test]
    fn nan_scale_propagates() {
        let input = [1.0f32, -4.0, 0.0, f32::MIN_POSITIVE, -0.0];
        let ran = affine_f32(Device::Vulkan, &input, f32::NAN, 0.25)
            .expect("missing adapter must fail, not skip");
        assert_real_adapter(&ran);
        assert!(ran.dispatched);
        for (index, value) in ran.values.iter().enumerate() {
            assert!(value.is_nan(), "index {index}: {value} is not NaN");
        }
        let bias = affine_f32(Device::Vulkan, &[1.0, -2.0, 0.0], 3.0, f32::NAN)
            .expect("missing adapter must fail, not skip");
        assert!(bias.dispatched);
        assert!(bias.values.iter().all(|value| value.is_nan()));
    }

    #[test]
    fn zero_length_does_not_dispatch_even_with_nan_params() {
        let ran = affine_f32(Device::Vulkan, &[], f32::NAN, f32::NAN)
            .expect("missing adapter must fail, not skip");
        assert_real_adapter(&ran);
        assert!(ran.values.is_empty());
        assert!(!ran.dispatched);
    }
}

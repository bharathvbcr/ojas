//! `Tape` on a device backend. `Resident` reports `BackendId::Wgpu`, keeps
//! every tensor in its own `DeviceBuffer`, and refuses any input that is not
//! one of those with `Placement`. Its kernels are the CPU ones, so its
//! gradients must equal a CPU tape's bit for bit.
//!
//! Readbacks are counted on the tape backend's own budget
//! (`Budget::device_readbacks`), so tests running in parallel cannot disturb
//! them. Both the double and the reference run `Numerics::Exact` (the CPU
//! default is `Fast`); the comparison is bit for bit.

use std::any::Any;
use std::sync::{Arc, Mutex, MutexGuard};

use ojas_autograd::{Tape, Var};
use ojas_core::{
    AdamWConfig, Backend, BackendId, Budget, DType, DeviceBuffer, MuonNs5Config, Numerics,
    OjasError, PerHeadGateGrad, Tensor, ValueResidualGrad,
};
use ojas_cpu::CpuBackend;

static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[derive(Debug)]
struct Bytes {
    id: BackendId,
    bytes: Vec<u8>,
}

impl DeviceBuffer for Bytes {
    fn backend(&self) -> BackendId {
        self.id
    }
    fn byte_len(&self) -> usize {
        self.bytes.len()
    }
    fn read_bytes(&self, offset: usize, len: usize) -> Result<Vec<u8>, OjasError> {
        Ok(self.bytes[offset..offset + len].to_vec())
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn on(id: BackendId, host: &Tensor, budget: &Budget) -> Result<Tensor, OjasError> {
    let bytes = host.contiguous_bytes()?.to_vec();
    Tensor::from_device(
        Arc::new(Bytes { id, bytes }),
        host.shape(),
        host.dtype(),
        budget,
    )
}

struct Resident {
    cpu: CpuBackend,
}

impl Resident {
    fn new() -> Self {
        Self {
            cpu: CpuBackend::new(Budget::new(1 << 26)).with_numerics(Numerics::Exact),
        }
    }

    /// Host copy of one of our tensors for the CPU kernel. Reads the fake
    /// buffer directly, so it is not counted as a readback.
    fn host(&self, op: &'static str, t: &Tensor) -> Result<Tensor, OjasError> {
        let placement = OjasError::Placement {
            op,
            expected: Some(BackendId::Wgpu),
            found: t.device(),
        };
        if t.device() != Some(BackendId::Wgpu) {
            return Err(placement);
        }
        let buf = t
            .device_buffer()
            .and_then(|b| b.as_any().downcast_ref::<Bytes>())
            .ok_or(placement)?;
        if !t.is_contiguous()? {
            return Err(OjasError::Shape {
                op,
                detail: "non-contiguous".to_string(),
            });
        }
        let start = t.byte_offset();
        let window = &buf.bytes[start..start + t.num_elements()? * 4];
        let words = window.as_chunks::<4>().0.iter().copied();
        match t.dtype() {
            DType::F32 => {
                let data: Vec<f32> = words.map(f32::from_ne_bytes).collect();
                Tensor::from_f32(&data, t.shape(), self.cpu.budget())
            }
            DType::U32 => {
                let data: Vec<u32> = words.map(u32::from_ne_bytes).collect();
                Tensor::from_u32(&data, t.shape(), self.cpu.budget())
            }
            other => Err(OjasError::Dtype {
                op,
                expected: DType::F32,
                got: other,
            }),
        }
    }

    fn dev(&self, t: Tensor) -> Result<Tensor, OjasError> {
        on(BackendId::Wgpu, &t, self.cpu.budget())
    }
}

fn unsupported(op: &'static str) -> OjasError {
    OjasError::Unsupported {
        op,
        detail: "test double".to_string(),
    }
}

impl Backend for Resident {
    fn id(&self) -> BackendId {
        BackendId::Wgpu
    }
    fn budget(&self) -> &Budget {
        self.cpu.budget()
    }
    fn upload(&self, tensor: &Tensor) -> Result<Tensor, OjasError> {
        match tensor.device() {
            None => self.dev(tensor.clone()),
            Some(BackendId::Wgpu) => Ok(tensor.clone()),
            found => Err(OjasError::Placement {
                op: "Resident::upload",
                expected: Some(BackendId::Wgpu),
                found,
            }),
        }
    }
    fn embedding_forward(&self, table: &Tensor, ids: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "embedding_forward";
        let y = self
            .cpu
            .embedding_forward(&self.host(OP, table)?, &self.host(OP, ids)?)?;
        self.dev(y)
    }
    fn embedding_backward(
        &self,
        table: &Tensor,
        ids: &Tensor,
        grad: &Tensor,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "embedding_backward";
        let g = self.cpu.embedding_backward(
            &self.host(OP, table)?,
            &self.host(OP, ids)?,
            &self.host(OP, grad)?,
        )?;
        self.dev(g)
    }
    fn linear_forward(&self, x: &Tensor, w: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "linear_forward";
        let y = self
            .cpu
            .linear_forward(&self.host(OP, x)?, &self.host(OP, w)?)?;
        self.dev(y)
    }
    fn linear_backward(
        &self,
        x: &Tensor,
        w: &Tensor,
        grad: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        const OP: &str = "linear_backward";
        let (gx, gw) = self.cpu.linear_backward(
            &self.host(OP, x)?,
            &self.host(OP, w)?,
            &self.host(OP, grad)?,
        )?;
        Ok((self.dev(gx)?, self.dev(gw)?))
    }
    fn rms_norm_forward(&self, _: &Tensor, _: &Tensor, _: f32) -> Result<Tensor, OjasError> {
        Err(unsupported("rms_norm_forward"))
    }
    fn rms_norm_backward(
        &self,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
        _: f32,
    ) -> Result<(Tensor, Tensor), OjasError> {
        Err(unsupported("rms_norm_backward"))
    }
    fn rope_half_split_forward(
        &self,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
    ) -> Result<Tensor, OjasError> {
        Err(unsupported("rope_half_split_forward"))
    }
    fn rope_half_split_backward(
        &self,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
    ) -> Result<Tensor, OjasError> {
        Err(unsupported("rope_half_split_backward"))
    }
    fn rms_qk_norm_forward(
        &self,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
        _: f32,
    ) -> Result<(Tensor, Tensor), OjasError> {
        Err(unsupported("rms_qk_norm_forward"))
    }
    fn rms_qk_norm_backward(
        &self,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
        _: f32,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor), OjasError> {
        Err(unsupported("rms_qk_norm_backward"))
    }
    fn causal_sdpa_forward(&self, _: &Tensor, _: &Tensor, _: &Tensor) -> Result<Tensor, OjasError> {
        Err(unsupported("causal_sdpa_forward"))
    }
    fn causal_sdpa_backward(
        &self,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor), OjasError> {
        Err(unsupported("causal_sdpa_backward"))
    }
    fn per_head_sigmoid_gate_forward(
        &self,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
    ) -> Result<Tensor, OjasError> {
        Err(unsupported("per_head_sigmoid_gate_forward"))
    }
    fn per_head_sigmoid_gate_backward(
        &self,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
    ) -> Result<PerHeadGateGrad, OjasError> {
        Err(unsupported("per_head_sigmoid_gate_backward"))
    }
    fn value_residual_blend_forward(
        &self,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
    ) -> Result<Tensor, OjasError> {
        Err(unsupported("value_residual_blend_forward"))
    }
    fn value_residual_blend_backward(
        &self,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
        _: &Tensor,
    ) -> Result<ValueResidualGrad, OjasError> {
        Err(unsupported("value_residual_blend_backward"))
    }
    fn silu_forward(&self, x: &Tensor) -> Result<Tensor, OjasError> {
        let y = self.cpu.silu_forward(&self.host("silu_forward", x)?)?;
        self.dev(y)
    }
    fn silu_backward(&self, x: &Tensor, grad: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "silu_backward";
        let g = self
            .cpu
            .silu_backward(&self.host(OP, x)?, &self.host(OP, grad)?)?;
        self.dev(g)
    }
    fn mul_forward(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "mul_forward";
        let y = self
            .cpu
            .mul_forward(&self.host(OP, a)?, &self.host(OP, b)?)?;
        self.dev(y)
    }
    fn mul_backward(
        &self,
        a: &Tensor,
        b: &Tensor,
        grad: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        const OP: &str = "mul_backward";
        let (ga, gb) = self.cpu.mul_backward(
            &self.host(OP, a)?,
            &self.host(OP, b)?,
            &self.host(OP, grad)?,
        )?;
        Ok((self.dev(ga)?, self.dev(gb)?))
    }
    fn residual_add_forward(&self, x: &Tensor, y: &Tensor) -> Result<Tensor, OjasError> {
        const OP: &str = "residual_add_forward";
        let z = self
            .cpu
            .residual_add_forward(&self.host(OP, x)?, &self.host(OP, y)?)?;
        self.dev(z)
    }
    fn residual_add_backward(
        &self,
        x: &Tensor,
        y: &Tensor,
        grad: &Tensor,
    ) -> Result<(Tensor, Tensor), OjasError> {
        const OP: &str = "residual_add_backward";
        let (gx, gy) = self.cpu.residual_add_backward(
            &self.host(OP, x)?,
            &self.host(OP, y)?,
            &self.host(OP, grad)?,
        )?;
        Ok((self.dev(gx)?, self.dev(gy)?))
    }
    fn cross_entropy_mean_forward(
        &self,
        logits: &Tensor,
        targets: &Tensor,
        ignore: Option<u32>,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "cross_entropy_mean_forward";
        let y = self.cpu.cross_entropy_mean_forward(
            &self.host(OP, logits)?,
            &self.host(OP, targets)?,
            ignore,
        )?;
        self.dev(y)
    }
    fn cross_entropy_mean_backward(
        &self,
        logits: &Tensor,
        targets: &Tensor,
        ignore: Option<u32>,
    ) -> Result<Tensor, OjasError> {
        const OP: &str = "cross_entropy_mean_backward";
        let g = self.cpu.cross_entropy_mean_backward(
            &self.host(OP, logits)?,
            &self.host(OP, targets)?,
            ignore,
        )?;
        self.dev(g)
    }
    fn clip_grad_norm(&self, _: &mut [Tensor], _: f32) -> Result<f32, OjasError> {
        Err(unsupported("clip_grad_norm"))
    }
    fn adamw_step(
        &self,
        _: &mut Tensor,
        _: &Tensor,
        _: &mut Tensor,
        _: &mut Tensor,
        _: u64,
        _: AdamWConfig,
    ) -> Result<(), OjasError> {
        Err(unsupported("adamw_step"))
    }
    fn muon_ns5_step(
        &self,
        _: &mut Tensor,
        _: &Tensor,
        _: &mut Tensor,
        _: MuonNs5Config,
    ) -> Result<(), OjasError> {
        Err(unsupported("muon_ns5_step"))
    }
}

fn data(seed: u32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let v = (i as u32)
                .wrapping_mul(2_654_435_761)
                .wrapping_add(seed * 97)
                % 1000;
            v as f32 / 500.0 - 1.0
        })
        .collect()
}

const VOCAB: usize = 7;
const DIM: usize = 5;
const ROWS: usize = 3;

struct Graph {
    table: Var,
    w1: Var,
    w2: Var,
    loss: Var,
    root: Var,
}

/// Embedding, two linears with a residual, silu, two reshapes, cross-entropy.
/// `h` feeds both the second linear and the add, so its gradient is
/// accumulated. With `doubled`, the root is `loss + loss` and the
/// cross-entropy gradient is 2, not the seed.
fn record<B: Backend>(tape: &mut Tape<B>, doubled: bool) -> Result<Graph, OjasError> {
    let budget = Budget::new(1 << 20);
    let table = tape.leaf(Tensor::from_f32(
        &data(1, VOCAB * DIM),
        &[VOCAB, DIM],
        &budget,
    )?)?;
    let w1 = tape.leaf(Tensor::from_f32(&data(2, DIM * DIM), &[DIM, DIM], &budget)?)?;
    let w2 = tape.leaf(Tensor::from_f32(
        &data(3, VOCAB * DIM),
        &[VOCAB, DIM],
        &budget,
    )?)?;
    let ids = Tensor::from_u32(&[1, 4, 6], &[ROWS], &budget)?;
    let targets = Tensor::from_u32(&[2, 0, 5], &[ROWS], &budget)?;
    let e = tape.embedding(table, ids)?;
    let h = tape.linear(e, w1)?;
    let s = tape.silu(h)?;
    let r = tape.add(h, s)?;
    let r3 = tape.reshape(r, &[1, ROWS, DIM])?;
    let r2 = tape.reshape(r3, &[ROWS, DIM])?;
    let logits = tape.linear(r2, w2)?;
    let loss = tape.cross_entropy(logits, targets, None)?;
    let root = if doubled { tape.add(loss, loss)? } else { loss };
    Ok(Graph {
        table,
        w1,
        w2,
        loss,
        root,
    })
}

fn bits(t: &Tensor) -> Vec<u32> {
    t.to_f32_vec()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn reference(doubled: bool) -> (Vec<u32>, [Vec<u32>; 3]) {
    let mut tape = Tape::new(CpuBackend::new(Budget::new(1 << 26)).with_numerics(Numerics::Exact));
    let g = record(&mut tape, doubled).unwrap();
    tape.backward(g.root).unwrap();
    let grads = [g.table, g.w1, g.w2].map(|v| bits(tape.grad(v).unwrap()));
    (bits(tape.value(g.loss).unwrap()), grads)
}

fn run_on_device(doubled: bool) {
    let _serial = serial();
    let (want_loss, want_grads) = reference(doubled);
    let mut tape = Tape::new(Resident::new());
    let before = tape.backend().budget().device_readbacks();
    let g = record(&mut tape, doubled).expect("forward on a device backend");
    tape.backward(g.root).expect("backward on a device backend");
    assert_eq!(
        tape.backend().budget().device_readbacks(),
        before,
        "forward or backward read a tensor back to the host"
    );
    for (name, var, want) in [
        ("table", g.table, &want_grads[0]),
        ("w1", g.w1, &want_grads[1]),
        ("w2", g.w2, &want_grads[2]),
    ] {
        let grad = tape.grad(var).expect("leaf gradient");
        assert_eq!(
            grad.device(),
            Some(BackendId::Wgpu),
            "{name} grad left the device"
        );
        assert_eq!(
            &bits(&tape.backend().download(grad).unwrap()),
            want,
            "{name} grad"
        );
    }
    let mid = tape.backend().budget().device_readbacks();
    let loss = tape
        .backend()
        .download(tape.value(g.loss).unwrap())
        .unwrap();
    assert_eq!(
        tape.backend().budget().device_readbacks(),
        (mid.0 + 1, mid.1 + 4),
        "the loss is one 4-byte read"
    );
    assert_eq!(bits(&loss), want_loss);
}

#[test]
fn device_tape_backward_reads_nothing_back() {
    run_on_device(false);
}

#[test]
fn device_tape_scales_a_non_root_cross_entropy_on_the_device() {
    run_on_device(true);
}

#[test]
fn device_tape_refuses_another_devices_tensor() {
    let _serial = serial();
    let mut tape = Tape::new(Resident::new());
    let budget = Budget::new(1 << 10);
    let host = Tensor::from_f32(&[1.0, 2.0], &[2], &budget).unwrap();
    let metal = on(BackendId::Metal, &host, &budget).unwrap();
    assert!(matches!(
        tape.leaf(metal),
        Err(OjasError::Placement {
            expected: Some(BackendId::Wgpu),
            found: Some(BackendId::Metal),
            ..
        })
    ));
}

//! ft-757cb6b2 criterion 8: a GEMM-backed op whose operand or output holds
//! more than `i32::MAX` elements is refused on the host as `Unsupported`,
//! before any reservation, allocation or dispatch. tessl's own
//! `require_i32_extent` (../tessl/src/gemm.rs) refuses the same quantity
//! later, on the device thread, after work was charged; the host bound makes
//! that refusal early and typed.
//!
//! Every case here uses a budget far below the oversize output, so nothing
//! of that size can be allocated: the only way to return `Unsupported` is
//! to refuse before the reservation, and the only way to return
//! `CapacityExceeded` is to have reached it. The other ops' over-bound
//! matrix is itself an operand, so those cases use [`Phantom`] buffers that
//! hold no memory; the in-crate unit test in `backend.rs` pins the bound
//! itself on the dims.

#![cfg(target_os = "macos")]

use std::any::Any;
use std::sync::{Arc, Mutex, MutexGuard};

use ojas_core::{
    BackendCore, BackendId, BackendModule, Budget, CeChunk, CeGrads, Conv2dParams,
    ConvTranspose2dParams, DType, DeviceBuffer, MuonNs5Config, OjasError, Tensor,
};
use ojas_metal::MetalBackend;

/// The device's allocated-bytes gauge is shared by every backend in the
/// process, so tests that compare it before and after a call run one at a
/// time: a sibling's upload would move it.
fn serial() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// A Metal-owned buffer that holds no memory: only a byte length. A tensor
/// over it passes `ojas_core`'s shape validators, which read shapes and
/// dtypes, so an op can be driven to its host bound with an operand of any
/// size and nothing allocated. The backend never reaches the buffer: it
/// refuses before placement is checked, and the bound under test is what
/// refuses. (Were the bound missing, the op would fail on placement, as
/// `Backend`, not `Unsupported`.)
#[derive(Debug)]
struct Phantom(usize);

impl DeviceBuffer for Phantom {
    fn backend(&self) -> BackendId {
        BackendId::Metal
    }
    fn byte_len(&self) -> usize {
        self.0
    }
    fn read_bytes(&self, _: usize, _: usize) -> Result<Vec<u8>, OjasError> {
        unreachable!("a phantom buffer is never read")
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// An f32 tensor of `shape` over a [`Phantom`]. Its reservation is on its
/// own accounting budget, not the backend's.
fn phantom(shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::from_device(
        Arc::new(Phantom(n * 4)),
        shape,
        DType::F32,
        &Budget::new(1 << 40),
    )
    .expect("phantom tensor")
}

/// Runs `call` and requires `Unsupported` naming `op` and the i32 limit,
/// with nothing dispatched, charged or allocated by the backend.
fn refused_before_any_work(
    op: &'static str,
    call: impl FnOnce(&MetalBackend) -> Option<OjasError>,
) {
    let _serial = serial();
    let m = MetalBackend::new(Budget::new(1 << 30)).expect("Metal device");
    m.sync().expect("settle");
    let dispatches = m.dispatches();
    let live = m.budget().live_bytes().expect("live");
    let allocated = m.memory().expect("memory").allocated;
    match call(&m) {
        Some(OjasError::Unsupported { op: got, detail }) => {
            assert_eq!(got, op);
            assert!(detail.contains("i32"), "{op}: {detail}");
        }
        other => panic!("{op}: expected Unsupported naming the i32 limit, got {other:?}"),
    }
    assert_eq!(m.dispatches(), dispatches, "{op}: nothing dispatched");
    assert_eq!(
        m.budget().live_bytes().expect("live"),
        live,
        "{op}: nothing charged"
    );
    assert_eq!(
        m.memory().expect("memory").allocated,
        allocated,
        "{op}: nothing allocated"
    );
    m.sync().expect("clean");
}

/// `x [65537, 32768]` holds 2^31 + 2^15 elements; `gy [65537, 1]` and
/// `w [1, 32768]` are small.
#[test]
fn linear_backward_input_operand_past_i32_is_unsupported_before_any_work() {
    refused_before_any_work("linear_backward_input", |m| {
        m.linear_backward_input(
            &phantom(&[65537, 32768]),
            &phantom(&[1, 32768]),
            &phantom(&[65537, 1]),
        )
        .err()
    });
}

#[test]
fn linear_backward_operand_past_i32_is_unsupported_before_any_work() {
    refused_before_any_work("linear_backward", |m| {
        m.linear_backward(
            &phantom(&[1, 32768]),
            &phantom(&[65537, 32768]),
            &phantom(&[1, 65537]),
        )
        .err()
    });
}

/// `weight [heads, d_model]`, `attn_out [rows, heads, head_dim]`; `[rows,
/// heads]` = 65537 * 32768 elements is the logits matrix.
#[test]
fn gate_forward_logits_past_i32_is_unsupported_before_any_work() {
    refused_before_any_work("per_head_sigmoid_gate_forward", |m| {
        m.per_head_sigmoid_gate_forward(
            &phantom(&[65537, 1]),
            &phantom(&[32768, 1]),
            &phantom(&[32768]),
            &phantom(&[65537, 32768, 1]),
        )
        .err()
    });
}

#[test]
fn gate_backward_logits_past_i32_is_unsupported_before_any_work() {
    refused_before_any_work("per_head_sigmoid_gate_backward", |m| {
        m.per_head_sigmoid_gate_backward(
            &phantom(&[65537, 1]),
            &phantom(&[32768, 1]),
            &phantom(&[32768]),
            &phantom(&[65537, 32768, 1]),
            &phantom(&[65537, 32768, 1]),
        )
        .err()
    });
}

#[test]
fn muon_parameter_past_i32_is_unsupported_before_any_work() {
    refused_before_any_work("muon_ns5_step", |m| {
        let mut param = phantom(&[65537, 32768]);
        let mut momentum = phantom(&[65537, 32768]);
        m.muon_ns5_step(
            &mut param,
            &phantom(&[65537, 32768]),
            &mut momentum,
            MuonNs5Config::nanolab_default(),
        )
        .err()
    });
}

fn up(m: &MetalBackend, shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    let host = Tensor::from_f32(&vec![0.5; n], shape, &Budget::new(1 << 26)).expect("host");
    m.upload(&host).expect("upload")
}

/// `[65537, 1] x [32768, 1]^T` outputs 65537 * 32768 = 2^31 + 2^15 elements.
#[test]
fn linear_forward_output_past_i32_is_unsupported_before_any_work() {
    let _serial = serial();
    let m = MetalBackend::new(Budget::new(1 << 30)).expect("Metal device");
    let x = up(&m, &[65537, 1]);
    let w = up(&m, &[32768, 1]);
    m.sync().expect("settle");
    let dispatches = m.dispatches();
    let live = m.budget().live_bytes().expect("live");
    let allocated = m.memory().expect("memory").allocated;
    match m.linear_forward(&x, &w) {
        Err(OjasError::Unsupported { op, detail }) => {
            assert_eq!(op, "linear_forward");
            assert!(detail.contains("i32"), "{detail}");
        }
        other => panic!("expected Unsupported naming the i32 limit, got {other:?}"),
    }
    assert_eq!(m.dispatches(), dispatches, "nothing dispatched");
    assert_eq!(
        m.budget().live_bytes().expect("live"),
        live,
        "nothing charged"
    );
    assert_eq!(
        m.memory().expect("memory").allocated,
        allocated,
        "nothing allocated"
    );
    m.sync().expect("clean");
}

/// Control: an output under the bound (2^30 + 2^14 elements, 4 GiB) is not
/// refused as `Unsupported`; with this 1 GiB budget it is a capacity
/// refusal. The host bound is tessl's, not tighter.
#[test]
fn linear_forward_output_under_i32_is_not_refused_as_unsupported() {
    let _serial = serial();
    let m = MetalBackend::new(Budget::new(1 << 30)).expect("Metal device");
    let x = up(&m, &[65537, 1]);
    let w = up(&m, &[16384, 1]);
    match m.linear_forward(&x, &w) {
        Err(OjasError::CapacityExceeded { .. }) => {}
        other => panic!("expected CapacityExceeded, got {other:?}"),
    }
}

/// `[65537, 1] x [32768, 1]^T` plus a `[32768]` bias: the fused path's
/// output is 2^31 + 2^15 elements.
#[test]
fn linear_bias_forward_output_past_i32_is_unsupported_before_any_work() {
    refused_before_any_work("linear_bias_forward", |m| {
        m.linear_bias_forward(
            &phantom(&[65537, 1]),
            &phantom(&[32768, 1]),
            &phantom(&[32768]),
        )
        .err()
    });
}

/// `linear_bias_backward` takes its gradients from `linear_backward`, whose
/// bound it inherits; the bias gradient adds no matrix.
#[test]
fn linear_bias_backward_inherits_the_linear_backward_bound() {
    refused_before_any_work("linear_backward", |m| {
        m.linear_bias_backward(
            &phantom(&[1, 32768]),
            &phantom(&[65537, 32768]),
            &phantom(&[65537]),
            &phantom(&[1, 65537]),
        )
        .err()
    });
}

/// 1 x 1 convolution: `W [32768, 1] @ X [1, 256 * 256]` is one image's
/// 2^31-element output, whichever route the operand's alignment picks.
#[test]
fn conv2d_pointwise_image_past_i32_is_unsupported_before_any_work() {
    refused_before_any_work("conv2d_forward", |m| {
        m.conv2d_forward(
            &phantom(&[1, 1, 256, 256]),
            &phantom(&[32768, 1, 1, 1]),
            None,
            Conv2dParams::new(),
        )
        .err()
    });
}

/// 3 x 3: `[65536, 9] @ [9, 32768]` is a 2^31-element product (the im2col
/// route, or the TensorOps route's 2^31-element output).
#[test]
fn conv2d_3x3_product_past_i32_is_unsupported_before_any_work() {
    refused_before_any_work("conv2d_forward", |m| {
        m.conv2d_forward(
            &phantom(&[1, 1, 258, 258]),
            &phantom(&[32768, 1, 3, 3]),
            None,
            Conv2dParams::new(),
        )
        .err()
    });
}

/// The reference arm, which never takes the pointwise route, on a 1 x 1.
#[test]
fn conv2d_via_im2col_product_past_i32_is_unsupported_before_any_work() {
    refused_before_any_work("conv2d_forward", |m| {
        m.conv2d_via_im2col(
            &phantom(&[1, 1, 256, 256]),
            &phantom(&[32768, 1, 1, 1]),
            None,
            Conv2dParams::new(),
        )
        .err()
    });
}

fn stride2() -> ConvTranspose2dParams {
    ConvTranspose2dParams {
        stride: [2, 2],
        ..ConvTranspose2dParams::new()
    }
}

/// `[65536, 1] @ [1, 32768]` columns: 2^31 elements.
#[test]
fn conv_transpose2d_columns_past_i32_is_unsupported_before_any_work() {
    refused_before_any_work("conv_transpose2d_forward", |m| {
        m.conv_transpose2d_forward(
            &phantom(&[1, 1, 256, 256]),
            &phantom(&[1, 8192, 2, 2]),
            None,
            stride2(),
        )
        .err()
    });
}

#[test]
fn conv_transpose2d_backward_columns_past_i32_is_unsupported_before_any_work() {
    refused_before_any_work("conv_transpose2d_backward", |m| {
        m.conv_transpose2d_backward(
            &phantom(&[1, 1, 256, 256]),
            &phantom(&[1, 8192, 2, 2]),
            None,
            &phantom(&[1, 8192, 512, 512]),
            stride2(),
        )
        .err()
    });
}

#[test]
fn conv2d_backward_product_past_i32_is_unsupported_before_any_work() {
    refused_before_any_work("conv2d_backward", |m| {
        m.conv2d_backward(
            &phantom(&[1, 1, 258, 258]),
            &phantom(&[32768, 1, 3, 3]),
            None,
            &phantom(&[1, 32768, 256, 256]),
            Conv2dParams::new(),
        )
        .err()
    });
}

/// One row of the model dim, `[1, 2^31]`, is past the limit: no tile fits.
#[test]
fn linear_cross_entropy_model_dim_past_i32_is_unsupported_before_any_work() {
    refused_before_any_work("linear_cross_entropy_mean", |m| {
        let targets = Tensor::from_device(
            Arc::new(Phantom(4)),
            &[1],
            DType::U32,
            &Budget::new(1 << 40),
        )
        .expect("phantom targets");
        m.linear_cross_entropy_mean(
            &phantom(&[1, 1 << 31]),
            &phantom(&[1, 1 << 31]),
            &targets,
            None,
            CeChunk { rows: 1, cols: 1 },
            CeGrads {
                input: true,
                weight: true,
            },
        )
        .err()
    });
}

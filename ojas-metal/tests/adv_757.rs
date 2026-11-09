//! Adversarial probes for ft-757cb6b2 criterion 8 (lane 757madv): boundary, zero and
//! wrapping dims, bf16 and numerics lanes, and the fused and convolution ops.
#![cfg(target_os = "macos")]

use std::any::Any;
use std::sync::{Arc, Mutex, MutexGuard};

use ojas_core::{
    bf16_to_f32, f32_to_bf16, BackendCore, BackendId, BackendModule, Budget, Conv2dParams, DType,
    DeviceBuffer, MuonNs5Config, Numerics, OjasError, Tensor,
};
use ojas_metal::MetalBackend;

fn serial() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|p| p.into_inner())
}

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
        unreachable!()
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}
fn phantom(shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::from_device(
        Arc::new(Phantom(n * 4)),
        shape,
        DType::F32,
        &Budget::new(1 << 44),
    )
    .expect("phantom tensor")
}

fn is_i32_unsupported(e: &Option<OjasError>) -> bool {
    matches!(e, Some(OjasError::Unsupported { detail, .. }) if detail.contains("i32"))
}

const MAX: usize = i32::MAX as usize;

#[test]
fn boundary_exact_max_accepted_max_plus_one_refused() {
    let _s = serial();
    let m = MetalBackend::new(Budget::new(1 << 30)).expect("Metal");
    let mut rows: Vec<(String, bool, Option<OjasError>)> = Vec::new();
    for (tag, k, over) in [("max", MAX, false), ("max+1", MAX + 1, true)] {
        // input [1,k] weight [1,k] (each exactly the bound at max)
        rows.push((
            format!("linear_forward in/w k={tag}"),
            over,
            m.linear_forward(&phantom(&[1, k]), &phantom(&[1, k])).err(),
        ));
        rows.push((
            format!("linear_backward_input k={tag}"),
            over,
            m.linear_backward_input(&phantom(&[1, k]), &phantom(&[1, k]), &phantom(&[1, 1]))
                .err(),
        ));
        rows.push((
            format!("linear_backward k={tag}"),
            over,
            m.linear_backward(&phantom(&[1, k]), &phantom(&[1, k]), &phantom(&[1, 1]))
                .err(),
        ));
        rows.push((
            format!("gate_fwd k={tag}"),
            over,
            m.per_head_sigmoid_gate_forward(
                &phantom(&[1, k]),
                &phantom(&[1, k]),
                &phantom(&[1]),
                &phantom(&[1, 1, 1]),
            )
            .err(),
        ));
        rows.push((
            format!("gate_bwd k={tag}"),
            over,
            m.per_head_sigmoid_gate_backward(
                &phantom(&[1, k]),
                &phantom(&[1, k]),
                &phantom(&[1]),
                &phantom(&[1, 1, 1]),
                &phantom(&[1, 1, 1]),
            )
            .err(),
        ));
        let mut p = phantom(&[1, k]);
        let mut mo = phantom(&[1, k]);
        rows.push((
            format!("muon k={tag}"),
            over,
            m.muon_ns5_step(
                &mut p,
                &phantom(&[1, k]),
                &mut mo,
                MuonNs5Config::nanolab_default(),
            )
            .err(),
        ));
        // output exactly at bound: [rows,k]x[n,k]: rows*n
        rows.push((
            format!("linear_forward out n={tag}"),
            over,
            m.linear_forward(&phantom(&[1, 1]), &phantom(&[k, 1])).err(),
        ));
    }
    let mut bad = vec![];
    for (name, over, e) in &rows {
        eprintln!("{name}: over={over} -> {e:?}");
        if is_i32_unsupported(e) != *over {
            bad.push(name.clone());
        }
    }
    assert!(bad.is_empty(), "boundary disagreements: {bad:?}");
}

#[test]
fn zero_dims_and_wrap_shapes() {
    let _s = serial();
    let m = MetalBackend::new(Budget::new(1 << 30)).expect("Metal");
    // zero rows, huge k: the product is 0; must not be an i32 refusal.
    for shape in [[0usize, 1usize << 40], [1usize << 40, 0]] {
        let r = std::panic::catch_unwind(|| ());
        let _ = r;
        let t = Tensor::from_device(
            Arc::new(Phantom(0)),
            &shape,
            DType::F32,
            &Budget::new(1 << 44),
        );
        eprintln!("{shape:?} -> tensor {:?}", t.as_ref().map(|_| ()));
        if let Ok(t) = t {
            let e = m.linear_forward(&t, &t).err();
            eprintln!("zero-dim linear_forward {shape:?}: {e:?}");
            assert!(!is_i32_unsupported(&e), "{shape:?}");
        }
    }
    // 2^33 * 2^33 wraps usize; the tensor constructor decides.
    let t = Tensor::from_device(
        Arc::new(Phantom(0)),
        &[1usize << 33, 1usize << 33],
        DType::F32,
        &Budget::new(1 << 44),
    );
    eprintln!("wrap shape tensor: {:?}", t.as_ref().map(|_| ()));
    // 2^16 x 2^16 = 2^32 wraps u32.
    let e = m
        .linear_forward(&phantom(&[1 << 16, 1 << 16]), &phantom(&[1, 1 << 16]))
        .err();
    assert!(is_i32_unsupported(&e), "{e:?}");
    // 65536*32768 = 2^31 output (i32::MAX+1)
    let e = m
        .linear_forward(&phantom(&[65536, 1]), &phantom(&[32768, 1]))
        .err();
    assert!(is_i32_unsupported(&e), "{e:?}");
}

fn bf16_up(m: &MetalBackend, shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    let b: Vec<u16> = (0..n).map(|_| f32_to_bf16(0.5)).collect();
    let _ = bf16_to_f32(b[0]);
    let host = Tensor::from_bf16_bits(&b, shape, &Budget::new(1 << 28)).expect("bf16 host");
    m.upload(&host).expect("upload")
}
fn f32_up(m: &MetalBackend, shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    let host = Tensor::from_f32(&vec![0.5; n], shape, &Budget::new(1 << 28)).expect("host");
    m.upload(&host).expect("upload")
}

fn snapshot(m: &MetalBackend) -> (u64, u64, u64) {
    m.sync().expect("settle");
    (
        m.dispatches(),
        m.budget().live_bytes().expect("live"),
        m.memory().expect("mem").allocated,
    )
}

#[test]
fn ordering_bf16_and_numerics_lanes_refuse_before_any_side_effect() {
    let _s = serial();
    for (label, numerics, accum) in [
        ("default", Numerics::Fast, false),
        ("exact", Numerics::Exact, false),
        ("accum", Numerics::Fast, true),
    ] {
        let m = MetalBackend::new(Budget::new(1 << 30))
            .expect("Metal")
            .with_numerics(numerics);
        m.set_bf16_accumulation(accum);
        for (kind, x, w) in [
            (
                "bf16/bf16",
                bf16_up(&m, &[65537, 1]),
                bf16_up(&m, &[32768, 1]),
            ),
            (
                "bf16/f32",
                bf16_up(&m, &[65537, 1]),
                f32_up(&m, &[32768, 1]),
            ),
            (
                "f32/bf16",
                f32_up(&m, &[65537, 1]),
                bf16_up(&m, &[32768, 1]),
            ),
        ] {
            let before = snapshot(&m);
            let e = m.linear_forward(&x, &w).err();
            let after = snapshot(&m);
            eprintln!("{label} {kind}: {e:?} before {before:?} after {after:?}");
            assert!(is_i32_unsupported(&e), "{label} {kind}: {e:?}");
            assert_eq!(before, after, "{label} {kind}: side effect before refusal");
        }
    }
}

/// linear_bias_forward (f32 and bf16 operands, which would widen first) and
/// conv2d_forward (1x1 pointwise, 3x3) with a 2^31-element output: each is
/// `Unsupported` naming i32, with nothing dispatched, charged or allocated.
/// Real small operands, so these run the whole host path, widening included.
#[test]
fn linear_bias_forward_and_conv2d_refuse_an_over_i32_output_with_no_side_effect() {
    let _s = serial();
    let m = MetalBackend::new(Budget::new(1 << 30)).expect("Metal");
    let bias = f32_up(&m, &[32768]);
    for (kind, x, w) in [
        ("f32", f32_up(&m, &[65537, 1]), f32_up(&m, &[32768, 1])),
        ("bf16", bf16_up(&m, &[65537, 1]), bf16_up(&m, &[32768, 1])),
    ] {
        let before = snapshot(&m);
        let e = m.linear_bias_forward(&x, &w, &bias).err();
        let after = snapshot(&m);
        assert!(is_i32_unsupported(&e), "linear_bias_forward {kind}: {e:?}");
        assert_eq!(before, after, "linear_bias_forward {kind}: side effect");
    }
    // 1x1: one image's [32768, 65536] output.
    let x = f32_up(&m, &[1, 1, 256, 256]);
    let w = f32_up(&m, &[32768, 1, 1, 1]);
    let before = snapshot(&m);
    let e = m.conv2d_forward(&x, &w, None, Conv2dParams::new()).err();
    let after = snapshot(&m);
    assert!(is_i32_unsupported(&e), "conv2d_forward 1x1: {e:?}");
    assert_eq!(before, after, "conv2d_forward 1x1: side effect");
    // 3x3: [65536, 9] @ [9, 32768].
    let x = f32_up(&m, &[1, 1, 258, 258]);
    let w = f32_up(&m, &[32768, 1, 3, 3]);
    let before = snapshot(&m);
    let e = m.conv2d_forward(&x, &w, None, Conv2dParams::new()).err();
    let after = snapshot(&m);
    assert!(is_i32_unsupported(&e), "conv2d_forward 3x3: {e:?}");
    assert_eq!(before, after, "conv2d_forward 3x3: side effect");
}

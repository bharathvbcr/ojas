//! Op inputs are read in place and never charged (since 2026-10-01).
//!
//! Every op reads its tensor operands where they are (a borrowed slice, or
//! a clone of the tensor that its pool tasks share), so the backend's
//! budget sees only what the op itself builds: its output, and the scratch
//! its kernel documents. Each case below states that exact peak, and the op
//! must succeed with exactly that much room and refuse with one f32 less.
//! Before, each operand was copied and the copy charged (audit F7), so the
//! peak was this plus the input bytes, and these bounds fail against it.
//!
//! The inputs are built under their own budget, so the backend's budget
//! sees only what the op itself charges. The backend is single-threaded
//! (`CpuBackend::new`), so the per-task terms are one task's.

use ojas_core::{
    Backend, Budget, CeChunk, CeGrads, GdnInputs, OjasError, Tensor, GDN_CHECKPOINT_TOKENS,
    RMS_NORM_EPS,
};
use ojas_cpu::CpuBackend;

const F32: u64 = 4;

type Op = Box<dyn Fn(&CpuBackend) -> Result<Tensor, OjasError>>;

struct Case {
    name: &'static str,
    /// Bytes of every tensor input the op reads; never charged.
    input_bytes: u64,
    /// Bytes of the tensor the op returns.
    output_bytes: u64,
    /// The smallest room the op runs in: its output plus its own scratch.
    peak_bytes: u64,
    run: Op,
}

fn f32t(budget: &Budget, n: usize, shape: &[usize]) -> Tensor {
    let data: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.125).collect();
    Tensor::from_f32(&data, shape, budget).unwrap()
}

fn cases(inputs: &Budget) -> Vec<Case> {
    let x = f32t(inputs, 64, &[4, 16]);
    let y = f32t(inputs, 64, &[4, 16]);
    let w16 = f32t(inputs, 16, &[16]);
    let logits = f32t(inputs, 32, &[4, 8]);
    let targets = Tensor::from_u32(&[1, 7, 0, 3], &[4], inputs).unwrap();
    let qkv = f32t(inputs, 32, &[1, 2, 4, 4]);
    let lin_x = f32t(inputs, 32, &[4, 8]);
    let lin_w = f32t(inputs, 24, &[3, 8]);
    let conv_x = f32t(inputs, 40, &[2, 5, 4]);
    let conv_w = f32t(inputs, 12, &[4, 3]);
    let rope_x = f32t(inputs, 64, &[1, 4, 2, 8]);
    let rope_cs = f32t(inputs, 16, &[4, 4]);
    let decay_a = f32t(inputs, 32, &[8, 4]);
    let decay_h = f32t(inputs, 4, &[4]);
    vec![
        // Written in place, row chunk by row chunk, into the output tensor
        // (`scoped::rows_into`). Until 2026-10-02 the peak was the output
        // twice: the task parts joined into a vector, then the tensor
        // copied from it.
        Case {
            name: "silu_forward",
            input_bytes: 64 * F32,
            output_bytes: 64 * F32,
            peak_bytes: 64 * F32,
            run: {
                let x = x.clone();
                Box::new(move |cpu| cpu.silu_forward(&x))
            },
        },
        // Written once, straight into the output tensor (validate.rs
        // `fill_out`).
        Case {
            name: "mul_forward",
            input_bytes: 128 * F32,
            output_bytes: 64 * F32,
            peak_bytes: 64 * F32,
            run: {
                let (x, y) = (x.clone(), y.clone());
                Box::new(move |cpu| cpu.mul_forward(&x, &y))
            },
        },
        Case {
            name: "residual_add_forward",
            input_bytes: 128 * F32,
            output_bytes: 64 * F32,
            peak_bytes: 64 * F32,
            run: {
                let (x, y) = (x.clone(), y.clone());
                Box::new(move |cpu| cpu.residual_add_forward(&x, &y))
            },
        },
        // As SiLU, plus one rstd per row (4 rows) while the kernel runs.
        Case {
            name: "rms_norm_forward",
            input_bytes: 80 * F32,
            output_bytes: 64 * F32,
            peak_bytes: (64 + 4) * F32,
            run: {
                let (x, w) = (x.clone(), w16.clone());
                Box::new(move |cpu| cpu.rms_norm_forward(&x, &w, RMS_NORM_EPS))
            },
        },
        Case {
            name: "cross_entropy_mean_forward",
            input_bytes: 32 * F32 + 4 * 4,
            output_bytes: F32,
            peak_bytes: F32,
            run: {
                let (l, t) = (logits.clone(), targets.clone());
                Box::new(move |cpu| cpu.cross_entropy_mean_forward(&l, &t, None))
            },
        },
        Case {
            name: "cross_entropy_mean_backward",
            input_bytes: 32 * F32 + 4 * 4,
            output_bytes: 32 * F32,
            peak_bytes: 32 * F32,
            run: {
                let (l, t) = (logits.clone(), targets.clone());
                Box::new(move |cpu| cpu.cross_entropy_mean_backward(&l, &t, None))
            },
        },
        // The per-row kernel's one task holds `time * dim + 2 * time` = 24
        // floats beside the output, which it writes in place: no per-task
        // result and no copy into the tensor (until 2026-10-02 the peak was
        // the output twice, its charge and the tensor copied from it). The
        // forward's second output, the 8-float row log-sum-exp, is charged
        // with the first; the case keeps only the output.
        Case {
            name: "causal_sdpa_forward",
            input_bytes: 96 * F32,
            output_bytes: 32 * F32,
            peak_bytes: (32 + 8 + 24) * F32,
            run: {
                let q = qkv.clone();
                Box::new(move |cpu| cpu.causal_sdpa_forward(&q, &q, &q, None).map(|(y, _)| y))
            },
        },
        // The hybrid-layer ops (`ojas-cpu/src/hybrid.rs`) are row passes
        // written in place into their output (`validate::fill_rows`), and
        // none keeps a per-row scratch vector: the RMS scale is a local
        // scalar and the sigmoid and the decay are pointwise. The peak is
        // the output alone.
        Case {
            name: "causal_conv1d_silu_forward",
            input_bytes: (40 + 12) * F32,
            output_bytes: 40 * F32,
            peak_bytes: 40 * F32,
            run: {
                let (x, w) = (conv_x.clone(), conv_w.clone());
                Box::new(move |cpu| cpu.causal_conv1d_silu_forward(&x, &w))
            },
        },
        // Unlike `rms_norm_forward` there is no rstd vector to charge.
        Case {
            name: "gated_rms_norm_forward",
            input_bytes: (64 + 64 + 16) * F32,
            output_bytes: 64 * F32,
            peak_bytes: 64 * F32,
            run: {
                let (x, z, w) = (x.clone(), y.clone(), w16.clone());
                Box::new(move |cpu| cpu.gated_rms_norm_forward(&x, &z, &w, RMS_NORM_EPS))
            },
        },
        Case {
            name: "rope_partial_forward",
            input_bytes: (64 + 16 + 16) * F32,
            output_bytes: 64 * F32,
            peak_bytes: 64 * F32,
            run: {
                let (x, cs) = (rope_x.clone(), rope_cs.clone());
                Box::new(move |cpu| cpu.rope_partial_forward(&x, &cs, &cs))
            },
        },
        Case {
            name: "rope_partial_backward",
            input_bytes: (64 + 16 + 16) * F32,
            output_bytes: 64 * F32,
            peak_bytes: 64 * F32,
            run: {
                let (gy, cs) = (rope_x.clone(), rope_cs.clone());
                Box::new(move |cpu| cpu.rope_partial_backward(&gy, &cs, &cs))
            },
        },
        Case {
            name: "sigmoid_forward",
            input_bytes: 64 * F32,
            output_bytes: 64 * F32,
            peak_bytes: 64 * F32,
            run: {
                let x = x.clone();
                Box::new(move |cpu| cpu.sigmoid_forward(&x))
            },
        },
        Case {
            name: "sigmoid_backward",
            input_bytes: 128 * F32,
            output_bytes: 64 * F32,
            peak_bytes: 64 * F32,
            run: {
                let (x, gy) = (x.clone(), y.clone());
                Box::new(move |cpu| cpu.sigmoid_backward(&x, &gy))
            },
        },
        // The per-head rate `-exp(a_log)` is a `heads`-long host vector
        // (4 floats here) that is not charged; the output alone is.
        Case {
            name: "gdn_log_decay_forward",
            input_bytes: (32 + 4 + 4) * F32,
            output_bytes: 32 * F32,
            peak_bytes: 32 * F32,
            run: {
                let (a, al, dt) = (decay_a.clone(), decay_h.clone(), decay_h.clone());
                Box::new(move |cpu| cpu.gdn_log_decay_forward(&a, &al, &dt))
            },
        },
        // `[4, 8] · [3, 8]ᵀ` is 96 multiply-adds, under the Fast whole-call
        // cutoff, so it is packed: A in one 6-row panel and B in one 16-wide
        // panel, 8 deep each (6·8 + 16·8 = 176 floats), held with the
        // 12-float output's charge.
        Case {
            name: "linear_forward",
            input_bytes: 56 * F32,
            output_bytes: 12 * F32,
            peak_bytes: (12 + 176) * F32,
            run: Box::new(move |cpu| cpu.linear_forward(&lin_x, &lin_w)),
        },
    ]
}

/// The exact peak: room for the output and the op's own scratch succeeds
/// and leaves only the output charged; one f32 less refuses and leaves
/// nothing charged. The input bytes are not part of either.
#[test]
fn every_op_charges_its_output_and_scratch_never_its_inputs() {
    let inputs = Budget::new(1 << 20);
    for case in cases(&inputs) {
        let with_copies = case.peak_bytes + case.input_bytes;
        let fits = CpuBackend::new(Budget::new(case.peak_bytes));
        let out = (case.run)(&fits).unwrap_or_else(|e| {
            panic!(
                "{}: refused with {} bytes, its output and scratch: {e:?}",
                case.name, case.peak_bytes
            )
        });
        assert_eq!(
            fits.budget().live_bytes().unwrap(),
            case.output_bytes,
            "{}",
            case.name
        );
        drop(out);
        let tight = CpuBackend::new(Budget::new(case.peak_bytes - F32));
        match (case.run)(&tight) {
            Err(OjasError::CapacityExceeded { .. }) => {}
            other => panic!(
                "{}: expected CapacityExceeded one f32 under its peak {} \
                 (copying the inputs would need {with_copies}), got {:?}",
                case.name,
                case.peak_bytes,
                other.map(|_| ())
            ),
        }
        assert_eq!(
            tight.budget().live_bytes().unwrap(),
            0,
            "{}: a refused op left a charge behind",
            case.name
        );
    }
}

#[test]
fn only_the_output_stays_charged_when_the_op_returns() {
    let inputs = Budget::new(1 << 20);
    for case in cases(&inputs) {
        let cpu = CpuBackend::new(Budget::new(1 << 20));
        let out = (case.run)(&cpu).unwrap_or_else(|e| panic!("{}: {e:?}", case.name));
        assert_eq!(
            cpu.budget().live_bytes().unwrap(),
            case.output_bytes,
            "{}: only the returned tensor stays charged",
            case.name
        );
        drop(out);
        assert_eq!(cpu.budget().live_bytes().unwrap(), 0, "{}", case.name);
    }
}

/// Every operand is validated before anything is charged, so a NaN in a
/// later operand is `NonFinite` under a budget with no room at all, not a
/// `CapacityExceeded` for the output.
#[test]
fn an_invalid_later_operand_is_reported_before_any_charge() {
    let inputs = Budget::new(1 << 20);
    let good = f32t(&inputs, 8, &[2, 4]);
    let bad = Tensor::from_f32(
        &[1.0, 2.0, f32::NAN, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[2, 4],
        &inputs,
    )
    .unwrap();
    // `linear_backward` of `[2, 4]` by `[2, 4]` takes a `[2, 2]` gradient. A
    // `[2, 4]` NaN gradient is malformed, and a shape error is reported
    // before any NaN scan (docs/shape-contract.md D3).
    let bad_grad = Tensor::from_f32(&[1.0, f32::NAN, 3.0, 4.0], &[2, 2], &inputs).unwrap();
    let empty = CpuBackend::new(Budget::new(0));
    for (name, result) in [
        ("mul_forward", empty.mul_forward(&good, &bad).map(|_| ())),
        (
            "linear_backward",
            empty.linear_backward(&good, &good, &bad_grad).map(|_| ()),
        ),
        (
            "clip_grad_norm",
            empty
                .clip_grad_norm(&mut [good.clone(), bad.clone()], 1.0)
                .map(|_| ()),
        ),
    ] {
        match result {
            Err(OjasError::NonFinite { .. }) => {}
            other => panic!("{name}: expected NonFinite, got {other:?}"),
        }
    }
    let ids = Tensor::from_f32(&[0.0, 1.0], &[2], &inputs).unwrap();
    match empty.cross_entropy_mean_forward(&good, &ids, None) {
        Err(OjasError::Dtype { .. }) => {}
        other => panic!("cross_entropy targets: expected Dtype, got {other:?}"),
    }
}

/// Causal SDPA backward returns three tensors, so it is not a [`Case`]. Its
/// per-row kernel writes the three gradients into their output tensors (3 ×
/// 32 floats) and holds `2 * time` = 8 floats of row scratch beside them
/// (probabilities and their gradients; the forward's lse replaces the
/// score row): that is the exact peak. Until 2026-10-02 every head's three
/// gradients were held as parts, joined into three vectors, then copied
/// into the tensors, so it needed about twice the outputs and this bound
/// fails.
#[test]
fn sdpa_backward_charges_its_three_gradients_and_its_row_scratch_only() {
    let inputs = Budget::new(1 << 20);
    let qkv = f32t(&inputs, 32, &[1, 2, 4, 4]);
    let gy = f32t(&inputs, 32, &[1, 2, 4, 4]);
    let (out, lse) = CpuBackend::new(inputs.clone())
        .causal_sdpa_forward(&qkv, &qkv, &qkv, None)
        .unwrap();
    let outputs = 3 * 32 * F32;
    exact_peak("causal_sdpa_backward", outputs + 8 * F32, outputs, |cpu| {
        cpu.causal_sdpa_backward(&qkv, &qkv, &qkv, &out, &lse, &gy, None)
    });
}

/// The gate backward writes its four gradients into their output tensors
/// (`[4, 8]` input, `[2, 8]` weight, `[2]` bias, `[4, 2, 2]` attn: 66
/// floats) and holds 280 floats beside them on one thread under
/// `Numerics::Fast`: the gate logits (8) and their product's packed
/// operands (6 × 8 + 16 × 8), `gz` (8), and the larger gradient product's
/// packed operands (`gzᵀ · input`: 6 × 4 + 16 × 4). Until 2026-10-02 the
/// gradient-attn and `gz` task parts, a joined copy of `gz` and the input
/// and weight gradients' product results were held as well, 74 floats more,
/// so this bound fails against that build.
#[test]
fn gate_backward_charges_its_four_gradients_and_its_scratch_only() {
    let inputs = Budget::new(1 << 20);
    let x = f32t(&inputs, 32, &[4, 8]);
    let w = f32t(&inputs, 16, &[2, 8]);
    let b = f32t(&inputs, 2, &[2]);
    let attn = f32t(&inputs, 16, &[4, 2, 2]);
    let gy = f32t(&inputs, 16, &[4, 2, 2]);
    let outputs = 66 * F32;
    exact_peak(
        "per_head_sigmoid_gate_backward",
        outputs + 280 * F32,
        outputs,
        |cpu| cpu.per_head_sigmoid_gate_backward(&x, &w, &b, &attn, &gy),
    );
}

/// The value-residual backward writes both gradients and `lambda`'s into
/// their output tensors (64 + 64 + 1 floats) and holds one `f64` sum per
/// 4096-value block beside them (two floats). Until 2026-10-02 both
/// gradients' task parts were held too, then joined and copied, so this
/// bound fails against that build.
#[test]
fn value_residual_backward_charges_its_gradients_and_its_block_sums_only() {
    let inputs = Budget::new(1 << 20);
    let v = f32t(&inputs, 64, &[4, 16]);
    let v0 = f32t(&inputs, 64, &[4, 16]);
    let lam = f32t(&inputs, 1, &[1]);
    let gy = f32t(&inputs, 64, &[4, 16]);
    let outputs = 129 * F32;
    exact_peak(
        "value_residual_blend_backward",
        outputs + 2 * F32,
        outputs,
        |cpu| cpu.value_residual_blend_backward(&v, &v0, &lam, &gy),
    );
}

/// A uniquely owned accumulator is added to in place: no buffer, no
/// charge, so it runs on a backend with no room at all. Until 2026-10-02
/// the sum was built in a charged buffer first, and this refused.
#[test]
fn accumulate_into_a_unique_acc_charges_nothing() {
    let inputs = Budget::new(1 << 20);
    let mut acc = f32t(&inputs, 64, &[4, 16]);
    let grad = f32t(&inputs, 64, &[4, 16]);
    let want: Vec<f32> = acc
        .f32_slice()
        .unwrap()
        .iter()
        .zip(grad.f32_slice().unwrap())
        .map(|(a, g)| a + g)
        .collect();
    let empty = CpuBackend::new(Budget::new(0));
    empty.accumulate_grad(&mut acc, &grad).unwrap();
    assert_eq!(acc.f32_slice().unwrap(), &want[..]);
    assert_eq!(empty.budget().live_bytes().unwrap(), 0);
}

/// An accumulator whose storage is shared is replaced by a new tensor,
/// written straight into it: the peak is that tensor (64 floats). Until
/// 2026-10-02 the sum was built in a charged buffer and then copied into
/// the tensor, twice that.
#[test]
fn accumulate_into_a_shared_acc_charges_the_new_tensor_only() {
    let inputs = Budget::new(1 << 20);
    let acc = f32t(&inputs, 64, &[4, 16]);
    let grad = f32t(&inputs, 64, &[4, 16]);
    exact_peak("accumulate_grad", 64 * F32, 64 * F32, |cpu| {
        // A clone shares `acc`'s storage, so `acc` cannot be written.
        let mut shared = acc.clone();
        cpu.accumulate_grad(&mut shared, &grad)?;
        Ok(shared)
    });
}

/// The fused cross-entropy writes both gradients straight into their output
/// tensors. `[4, 8]` input, `[64, 8]` weight, chunk `4 × 8`, one thread: the
/// gradients (32 + 512 floats) and the tile scratch (284 floats: the
/// `4 × 8` logit tile 32, row statistics 12, the largest product's packed
/// operands 6 × 8 + 16 × 8, and the larger gradient product 8 × 8) are the
/// peak; the loss (1 float) comes after the scratch is freed. Until
/// 2026-10-02 each gradient was built in a charged buffer and copied into
/// its tensor, so the weight gradient was held twice: 1057 floats.
#[test]
fn fused_cross_entropy_charges_its_gradients_and_tile_scratch_only() {
    let inputs = Budget::new(1 << 20);
    let x = f32t(&inputs, 32, &[4, 8]);
    let w = f32t(&inputs, 512, &[64, 8]);
    let t = Tensor::from_u32(&[1, 63, 0, 17], &[4], &inputs).unwrap();
    let chunk = CeChunk { rows: 4, cols: 8 };
    exact_peak(
        "linear_cross_entropy_mean",
        (544 + 284) * F32,
        545 * F32,
        |cpu| cpu.linear_cross_entropy_mean(&x, &w, &t, None, chunk, CeGrads::BOTH),
    );
}

/// The conv backward writes `grad_input` (`[2, 5, 4]`, 40 floats) and
/// `grad_weight` (`[4, 3]`, 12) into their output tensors and charges, beside
/// them, `da = gy * silu'(a)` (one value per input, 40 floats) and the
/// weight gradient's tap-major sums (`K * channels in the piece` = 3 * 4 =
/// 12 floats; one thread and 4 channels is one piece), the two terms the
/// kernel documents (`hybrid.rs`, `conv1d_silu_backward`).
#[test]
fn conv1d_backward_charges_its_two_gradients_da_and_tap_sums_only() {
    let inputs = Budget::new(1 << 20);
    let x = f32t(&inputs, 40, &[2, 5, 4]);
    let w = f32t(&inputs, 12, &[4, 3]);
    let gy = f32t(&inputs, 40, &[2, 5, 4]);
    let outputs = (40 + 12) * F32;
    exact_peak(
        "causal_conv1d_silu_backward",
        outputs + (40 + 3 * 4) * F32,
        outputs,
        |cpu| cpu.causal_conv1d_silu_backward(&x, &w, &gy),
    );
}

/// The gated RMSNorm backward writes `dx` and `dz` (`[4, 16]`, 64 floats
/// each) and `dw` (16) into their output tensors and holds one `rstd` per
/// row (4 floats) while the row pass runs (`gated_rms_backward`'s
/// `room_for(rows)`).
#[test]
fn gated_rms_backward_charges_its_three_gradients_and_one_rstd_per_row_only() {
    let inputs = Budget::new(1 << 20);
    let x = f32t(&inputs, 64, &[4, 16]);
    let z = f32t(&inputs, 64, &[4, 16]);
    let w = f32t(&inputs, 16, &[16]);
    let gy = f32t(&inputs, 64, &[4, 16]);
    let outputs = (64 + 64 + 16) * F32;
    exact_peak(
        "gated_rms_norm_backward",
        outputs + 4 * F32,
        outputs,
        |cpu| cpu.gated_rms_norm_backward(&x, &z, &w, &gy, RMS_NORM_EPS),
    );
}

/// The decay backward writes `da` (`[8, 4]`, 32 floats), `da_log` and
/// `ddt_bias` (4 each) and charges nothing else.
#[test]
fn gdn_log_decay_backward_charges_its_three_gradients_only() {
    let inputs = Budget::new(1 << 20);
    let a = f32t(&inputs, 32, &[8, 4]);
    let h = f32t(&inputs, 4, &[4]);
    let gy = f32t(&inputs, 32, &[8, 4]);
    let outputs = (32 + 4 + 4) * F32;
    exact_peak("gdn_log_decay_backward", outputs, outputs, |cpu| {
        cpu.gdn_log_decay_backward(&a, &h, &h, &gy)
    });
}

/// `B = 1, T = 5, H = 2, Dk = Dv = 4`, so `rows = B * T * H = 10`, one
/// 64-token checkpoint chunk, and a per-head state of `Dk * Dv = 16`.
struct Gdn {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    s0: Tensor,
}

impl Gdn {
    fn new(inputs: &Budget) -> Self {
        Self {
            q: f32t(inputs, 40, &[1, 5, 2, 4]),
            k: f32t(inputs, 40, &[1, 5, 2, 4]),
            v: f32t(inputs, 40, &[1, 5, 2, 4]),
            g: f32t(inputs, 10, &[1, 5, 2]),
            beta: f32t(inputs, 10, &[1, 5, 2]),
            s0: f32t(inputs, 32, &[1, 2, 4, 4]),
        }
    }

    fn inputs(&self) -> GdnInputs<'_> {
        GdnInputs {
            q: &self.q,
            k: &self.k,
            v: &self.v,
            g: &self.g,
            beta: &self.beta,
            initial_state: Some(&self.s0),
        }
    }
}

/// The delta rule's forward writes the output (`[1, 5, 2, 4]`, 40 floats),
/// the final state (32) and the checkpoints (`[1, 2, 1, 4, 4]`, 32) into
/// their tensors and holds, from `gdn::forward`'s own sum: the normalized q
/// and k (`rows * 2 * Dk` = 80 floats), the head-major output because H > 1
/// (`rows * Dv` = 40) and one task's state, `u` and `delta`
/// (`Dk * Dv + 2 * Dv` = 24).
#[test]
fn gdn_forward_charges_its_three_outputs_and_the_documented_scratch_only() {
    let inputs = Budget::new(1 << 20);
    let gdn = Gdn::new(&inputs);
    let outputs = (40 + 32 + 32) * F32;
    exact_peak(
        "chunked_gdn_forward",
        outputs + (80 + 40 + 24) * F32,
        outputs,
        |cpu| cpu.chunked_gdn_forward(gdn.inputs()),
    );
}

/// The delta rule's backward writes dq, dk, dv (40 floats each), dg and
/// dbeta (10 each) and the initial-state gradient (32) and holds
/// (`gdn::backward`): `rows * (4 * Dk + Dv + 4)` = 240 floats for the
/// normalized q and k, their norms and the head-major gradients, and one
/// task's `(64 + 4) * Dk * Dv + 4 * Dv + 3 * Dk` = 1116 floats for a chunk of
/// states, the running state and its gradients, and the per-token vectors.
#[test]
fn gdn_backward_charges_its_six_gradients_and_the_documented_scratch_only() {
    let inputs = Budget::new(1 << 20);
    let gdn = Gdn::new(&inputs);
    assert_eq!(
        GDN_CHECKPOINT_TOKENS, 64,
        "the 1116 above assumes 64-token chunks"
    );
    let fwd = CpuBackend::new(inputs.clone())
        .chunked_gdn_forward(gdn.inputs())
        .unwrap();
    let gy = f32t(&inputs, 40, &[1, 5, 2, 4]);
    let outputs = (3 * 40 + 2 * 10 + 32) * F32;
    exact_peak(
        "chunked_gdn_backward",
        outputs + (240 + 1116) * F32,
        outputs,
        |cpu| cpu.chunked_gdn_backward(gdn.inputs(), &fwd.checkpoints, &gy, None),
    );
}

/// `run` succeeds with exactly `peak` bytes of room and leaves `outputs`
/// charged while its result lives, then nothing; one f32 less refuses with
/// `CapacityExceeded` and leaves nothing charged.
fn exact_peak<T>(
    name: &str,
    peak: u64,
    outputs: u64,
    run: impl Fn(&CpuBackend) -> Result<T, OjasError>,
) {
    let fits = CpuBackend::new(Budget::new(peak));
    let got = run(&fits).unwrap_or_else(|e| panic!("{name}: refused with {peak} bytes: {e:?}"));
    assert_eq!(fits.budget().live_bytes().unwrap(), outputs, "{name}");
    drop(got);
    assert_eq!(fits.budget().live_bytes().unwrap(), 0, "{name}");
    let tight = CpuBackend::new(Budget::new(peak - F32));
    match run(&tight) {
        Err(OjasError::CapacityExceeded { .. }) => {}
        other => panic!(
            "{name}: expected CapacityExceeded one f32 under {peak}, got {:?}",
            other.map(|_| ())
        ),
    }
    assert_eq!(tight.budget().live_bytes().unwrap(), 0, "{name}");
}

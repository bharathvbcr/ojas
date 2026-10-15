//! The bf16 operand contract of [`Backend::bf16_operands`], checked on any
//! backend that reports it.
//!
//! The matmul-class ops (`linear_forward`, `linear_backward`,
//! `causal_sdpa_forward`, `causal_sdpa_backward` except its `lse`) take each
//! operand as `F32` or `Bf16`, mixed freely, and a `Bf16` operand is widened
//! exactly into `f32` accumulation. So for operands whose `f32` values are
//! bf16-exact, every mix of `F32` and `Bf16` slots must give the bits the
//! all-`F32` call gives, under the backend's own [`ojas_core::Numerics`].
//! [`bf16_operand_contract`] runs every mix of every op on a few shapes
//! (grouped-query and windowed attention, a `Bf16` view at an odd element
//! offset), then checks the edges of the contract:
//!
//! - the tied table (`embedding_forward`'s and `embedding_backward`'s
//!   `table`, `linear_cross_entropy_mean`'s `weight`) takes the same rule
//!   on a backend with the table lane, checked the same way, with the fused
//!   cross-entropy over one vocabulary chunk and over several; a backend
//!   without the lane must refuse every `Bf16` table with
//!   [`OjasError::Dtype`] ([`Bf16Report::table_lane`] says which ran);
//! - every other op, and every other operand of the table ops, refuses a
//!   `Bf16` operand with [`OjasError::Dtype`];
//! - a `Bf16` NaN or infinity in a matmul-class operand is
//!   [`OjasError::NonFinite`], from the op or, on a backend that defers
//!   faults, from the readback or [`Backend::sync`] after it;
//! - [`Backend::to_bf16`] and [`Backend::to_f32`] on this backend's own
//!   tensors give the host conversions' bits ([`f32_to_bf16`],
//!   [`bf16_to_f32`]), NaN payloads included: the NaN rule is ojas's one rule
//!   (sign, the top payload bits and the quiet bit `0x0040`), not torch's
//!   canonical `0x7fc0`.
//!
//! Operands go up with [`Backend::upload`] and results come back with
//! [`Backend::download`], so a device backend runs the same runner as the CPU.

use ojas_core::{bf16_to_f32, f32_to_bf16, round_f32_to_bf16, Backend, DType, OjasError, Tensor};

use crate::parity::ParityFailure;

/// What one [`bf16_operand_contract`] run covered.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Bf16Report {
    /// Calls with at least one `Bf16` operand, each compared bit for bit
    /// with the all-`F32` call.
    pub mixed_calls: usize,
    /// Non-matmul calls refused with [`OjasError::Dtype`].
    pub refusals: usize,
    /// Matmul-class and table calls whose `Bf16` NaN or infinity was
    /// refused.
    pub non_finite: usize,
    /// Whether the backend took the tied-table lane. Without it, every
    /// table call was refused with [`OjasError::Dtype`] and counted in
    /// `refusals`.
    pub table_lane: bool,
}

fn fail(check: &str, detail: impl Into<String>) -> ParityFailure {
    ParityFailure {
        check: check.to_string(),
        detail: detail.into(),
    }
}

fn ojas(check: &str) -> impl Fn(OjasError) -> ParityFailure + '_ {
    move |e| fail(check, format!("backend error: {e}"))
}

/// Deterministic bf16-exact values in `[-scale, scale)`.
fn values(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = (state >> 40) as f32 / (1u64 << 24) as f32;
            round_f32_to_bf16((2.0 * unit - 1.0) * scale)
        })
        .collect()
}

/// One operand: its bf16-exact values, its shape, and whether it may be `Bf16`.
struct Slot {
    data: Vec<f32>,
    shape: Vec<usize>,
    bf16: bool,
}

impl Slot {
    fn matmul(data: Vec<f32>, shape: &[usize]) -> Self {
        Slot {
            data,
            shape: shape.to_vec(),
            bf16: true,
        }
    }

    fn f32_only(data: Vec<f32>, shape: &[usize]) -> Self {
        Slot {
            data,
            shape: shape.to_vec(),
            bf16: false,
        }
    }
}

/// A host `Bf16` tensor of `data` (bf16-exact, so the narrowing is exact).
fn host_bf16<B: Backend + ?Sized>(
    b: &B,
    data: &[f32],
    shape: &[usize],
) -> Result<Tensor, OjasError> {
    let bits: Vec<u16> = data.iter().map(|&v| f32_to_bf16(v)).collect();
    Tensor::from_bf16_bits(&bits, shape, b.budget())
}

/// A `Bf16` operand that is a contiguous view one element into a larger
/// allocation (byte offset 2), so a backend cannot assume a 4-byte-aligned
/// bf16 window.
fn host_bf16_offset<B: Backend + ?Sized>(
    b: &B,
    data: &[f32],
    shape: &[usize],
) -> Result<Tensor, OjasError> {
    let mut bits = vec![f32_to_bf16(7.0)];
    bits.extend(data.iter().map(|&v| f32_to_bf16(v)));
    bits.push(f32_to_bf16(-7.0));
    let whole = Tensor::from_bf16_bits(&bits, &[bits.len()], b.budget())?;
    whole.narrow(2, shape, &contiguous_strides(shape))
}

fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// How the `Bf16` slots of a call are built.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Layout {
    Packed,
    Offset,
}

fn operands<B: Backend + ?Sized>(
    b: &B,
    slots: &[Slot],
    mask: u32,
    layout: Layout,
) -> Result<Vec<Tensor>, OjasError> {
    slots
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let host = if mask & (1 << i) != 0 {
                match layout {
                    Layout::Packed => host_bf16(b, &s.data, &s.shape)?,
                    Layout::Offset => host_bf16_offset(b, &s.data, &s.shape)?,
                }
            } else {
                Tensor::from_f32(&s.data, &s.shape, b.budget())?
            };
            b.upload(&host)
        })
        .collect()
}

fn bits_of<B: Backend + ?Sized>(b: &B, outs: &[Tensor]) -> Result<Vec<Vec<u32>>, OjasError> {
    outs.iter()
        .map(|t| {
            if t.dtype() != DType::F32 {
                return Err(OjasError::Dtype {
                    op: "bf16 operand contract",
                    expected: DType::F32,
                    got: t.dtype(),
                });
            }
            Ok(b.download(t)?
                .to_f32_vec()?
                .iter()
                .map(|v| v.to_bits())
                .collect())
        })
        .collect()
}

type Call<'a, B> = &'a dyn Fn(&B, &[Tensor]) -> Result<Vec<Tensor>, OjasError>;

/// Every mix of `F32` and `Bf16` over the bf16-capable slots, against the
/// all-`F32` call. Returns the number of mixed calls.
fn every_mix<B: Backend + ?Sized>(
    b: &B,
    name: &str,
    slots: &[Slot],
    call: Call<'_, B>,
) -> Result<usize, ParityFailure> {
    let check = format!("bf16 operands {name}");
    let reference = operands(b, slots, 0, Layout::Packed).map_err(ojas(&check))?;
    let want = call(b, &reference)
        .and_then(|outs| bits_of(b, &outs))
        .map_err(ojas(&check))?;
    let capable: u32 = slots
        .iter()
        .enumerate()
        .filter(|(_, s)| s.bf16)
        .fold(0, |m, (i, _)| m | (1 << i));
    let mut calls = 0;
    for layout in [Layout::Packed, Layout::Offset] {
        let mut mask = capable;
        while mask != 0 {
            let ts = operands(b, slots, mask, layout).map_err(ojas(&check))?;
            let got = call(b, &ts)
                .and_then(|outs| bits_of(b, &outs))
                .map_err(|e| fail(&check, format!("bf16 slots {mask:#b}: backend error: {e}")))?;
            for (out, (g, w)) in got.iter().zip(&want).enumerate() {
                if let Some(i) = g.iter().zip(w).position(|(g, w)| g != w) {
                    let offset = if layout == Layout::Offset {
                        " (offset view)"
                    } else {
                        ""
                    };
                    return Err(fail(
                        &check,
                        format!(
                            "bf16 slots {mask:#b}{offset}: output {out} element {i} is {:#010x}, \
                             the all-F32 call gives {:#010x}",
                            g[i], w[i]
                        ),
                    ));
                }
            }
            calls += 1;
            mask = (mask - 1) & capable;
        }
    }
    Ok(calls)
}

/// The op's one call with operand `slot` made `Bf16` and holding `bad`
/// (a NaN or an infinity) at element 0 must fail [`OjasError::NonFinite`],
/// from the op, its readback, or the sync after it.
fn refuses_non_finite<B: Backend + ?Sized>(
    b: &B,
    name: &str,
    slots: &[Slot],
    slot: usize,
    bad: f32,
    call: Call<'_, B>,
) -> Result<(), ParityFailure> {
    let check = format!("bf16 non-finite {name}");
    let mut poisoned: Vec<Slot> = slots
        .iter()
        .map(|s| Slot {
            data: s.data.clone(),
            shape: s.shape.clone(),
            bf16: s.bf16,
        })
        .collect();
    poisoned[slot].data[0] = bad;
    let ts = operands(b, &poisoned, 1 << slot, Layout::Packed).map_err(ojas(&check))?;
    let outcome = call(b, &ts).and_then(|outs| {
        bits_of(b, &outs)?;
        b.sync()
    });
    match outcome {
        Err(OjasError::NonFinite { .. }) => {
            // A deferring backend has reported its fault; nothing is left.
            b.sync().map_err(ojas(&check))
        }
        Err(e) => Err(fail(
            &check,
            format!("slot {slot} holding {bad}: expected NonFinite, got {e}"),
        )),
        Ok(()) => Err(fail(
            &check,
            format!("slot {slot} holding {bad}: the call succeeded"),
        )),
    }
}

fn two(pair: (Tensor, Tensor)) -> Vec<Tensor> {
    vec![pair.0, pair.1]
}

fn three(t: (Tensor, Tensor, Tensor)) -> Vec<Tensor> {
    vec![t.0, t.1, t.2]
}

/// Linear forward and backward at `rows x kin -> nout`, input shaped
/// `lead ++ [kin]`.
fn linear_cases<B: Backend + ?Sized>(
    b: &B,
    lead: &[usize],
    kin: usize,
    nout: usize,
    seed: u64,
    report: &mut Bf16Report,
) -> Result<(), ParityFailure> {
    let rows: usize = lead.iter().product();
    let mut x_shape = lead.to_vec();
    x_shape.push(kin);
    let mut y_shape = lead.to_vec();
    y_shape.push(nout);
    let fwd = [
        Slot::matmul(values(rows * kin, seed, 2.0), &x_shape),
        Slot::matmul(values(nout * kin, seed + 1, 0.5), &[nout, kin]),
    ];
    let call_fwd: Call<'_, B> = &|b, t| Ok(vec![b.linear_forward(&t[0], &t[1])?]);
    let name = format!("linear_forward {x_shape:?} x [{nout}, {kin}]");
    report.mixed_calls += every_mix(b, &name, &fwd, call_fwd)?;
    let bwd = [
        Slot::matmul(values(rows * kin, seed + 2, 2.0), &x_shape),
        Slot::matmul(values(nout * kin, seed + 3, 0.5), &[nout, kin]),
        Slot::matmul(values(rows * nout, seed + 4, 1.0), &y_shape),
    ];
    let call_bwd: Call<'_, B> = &|b, t| Ok(two(b.linear_backward(&t[0], &t[1], &t[2])?));
    let name = format!("linear_backward {x_shape:?} x [{nout}, {kin}]");
    report.mixed_calls += every_mix(b, &name, &bwd, call_bwd)?;
    for slot in 0..2 {
        refuses_non_finite(b, "linear_forward", &fwd, slot, f32::INFINITY, call_fwd)?;
        report.non_finite += 1;
    }
    for slot in 0..3 {
        refuses_non_finite(b, "linear_backward", &bwd, slot, f32::NAN, call_bwd)?;
        report.non_finite += 1;
    }
    Ok(())
}

/// Causal SDPA forward and backward on `[batch, heads, time, dim]` queries
/// against `kv_heads` KV heads, optionally windowed. The backward's
/// `output` is the forward's, rounded to bf16 so it may be a `Bf16` slot;
/// its `lse` is the forward's, `F32` only.
#[allow(clippy::too_many_arguments)]
fn sdpa_cases<B: Backend + ?Sized>(
    b: &B,
    batch: usize,
    heads: usize,
    kv_heads: usize,
    time: usize,
    dim: usize,
    window: Option<usize>,
    seed: u64,
    report: &mut Bf16Report,
) -> Result<(), ParityFailure> {
    let q_shape = [batch, heads, time, dim];
    let kv_shape = [batch, kv_heads, time, dim];
    let qn = batch * heads * time * dim;
    let kvn = batch * kv_heads * time * dim;
    let fwd = [
        Slot::matmul(values(qn, seed, 1.0), &q_shape),
        Slot::matmul(values(kvn, seed + 1, 1.0), &kv_shape),
        Slot::matmul(values(kvn, seed + 2, 1.0), &kv_shape),
    ];
    let call_fwd: Call<'_, B> =
        &move |b, t| Ok(two(b.causal_sdpa_forward(&t[0], &t[1], &t[2], window)?));
    let name = format!("causal_sdpa_forward q {q_shape:?} kv {kv_shape:?} window {window:?}");
    report.mixed_calls += every_mix(b, &name, &fwd, call_fwd)?;
    let check = format!("bf16 operands {name}");
    let reference = operands(b, &fwd, 0, Layout::Packed).map_err(ojas(&check))?;
    let (o, lse) = b
        .causal_sdpa_forward(&reference[0], &reference[1], &reference[2], window)
        .map_err(ojas(&check))?;
    let o: Vec<f32> = b
        .download(&o)
        .and_then(|o| o.to_f32_vec())
        .map_err(ojas(&check))?
        .into_iter()
        .map(round_f32_to_bf16)
        .collect();
    let lse = b
        .download(&lse)
        .and_then(|l| l.to_f32_vec())
        .map_err(ojas(&check))?;
    let bwd = [
        Slot::matmul(fwd[0].data.clone(), &q_shape),
        Slot::matmul(fwd[1].data.clone(), &kv_shape),
        Slot::matmul(fwd[2].data.clone(), &kv_shape),
        Slot::matmul(o, &q_shape),
        Slot::f32_only(lse, &[batch, heads, time]),
        Slot::matmul(values(qn, seed + 3, 1.0), &q_shape),
    ];
    let call_bwd: Call<'_, B> = &move |b, t| {
        Ok(three(b.causal_sdpa_backward(
            &t[0], &t[1], &t[2], &t[3], &t[4], &t[5], window,
        )?))
    };
    let name = format!("causal_sdpa_backward q {q_shape:?} kv {kv_shape:?} window {window:?}");
    report.mixed_calls += every_mix(b, &name, &bwd, call_bwd)?;
    for slot in 0..3 {
        refuses_non_finite(
            b,
            "causal_sdpa_forward",
            &fwd,
            slot,
            f32::INFINITY,
            call_fwd,
        )?;
        report.non_finite += 1;
    }
    for slot in [0, 1, 2, 3, 5] {
        refuses_non_finite(b, "causal_sdpa_backward", &bwd, slot, f32::NAN, call_bwd)?;
        report.non_finite += 1;
    }
    // The lse slot is F32 only.
    let mut ts = operands(b, &bwd, 0, Layout::Packed).map_err(ojas(&check))?;
    ts[4] = host_bf16(b, &bwd[4].data, &bwd[4].shape)
        .and_then(|t| b.upload(&t))
        .map_err(ojas(&check))?;
    match call_bwd(b, &ts) {
        Err(OjasError::Dtype { .. }) => report.refusals += 1,
        Err(e) => return Err(fail(&check, format!("Bf16 lse: expected Dtype, got {e}"))),
        Ok(_) => return Err(fail(&check, "Bf16 lse was accepted")),
    }
    Ok(())
}

/// A named call that must be refused.
type Refusal<'a> = (&'a str, Box<dyn Fn() -> Result<(), OjasError> + 'a>);

/// The tied table at `[vocab, dim]`: `embedding_forward` and
/// `embedding_backward` over `ids`, and `linear_cross_entropy_mean` of an
/// `[rows, dim]` input over every target, at a chunk of one vocabulary
/// block and at one of several, with and without gradients. On a backend
/// with the lane every mix of the table slot must give the `F32` table's
/// bits and a `Bf16` NaN or infinity must be refused; on one without it,
/// every `Bf16` table must be [`OjasError::Dtype`].
fn table_cases<B: Backend + ?Sized>(
    b: &B,
    vocab: usize,
    dim: usize,
    rows: usize,
    seed: u64,
    report: &mut Bf16Report,
) -> Result<(), ParityFailure> {
    const CHECK: &str = "bf16 tied table";
    let up_u32 = |data: &[u32], shape: &[usize]| {
        Tensor::from_u32(data, shape, b.budget()).and_then(|t| b.upload(&t))
    };
    let picks: Vec<u32> = (0..rows as u32)
        .map(|i| (i * 7 + 3) % vocab as u32)
        .collect();
    let ids = up_u32(&picks, &[rows]).map_err(ojas(CHECK))?;
    let targets: Vec<u32> = picks.iter().map(|&i| (i + 5) % vocab as u32).collect();
    let targets = up_u32(&targets, &[rows]).map_err(ojas(CHECK))?;
    let table = || Slot::matmul(values(vocab * dim, seed, 1.0), &[vocab, dim]);
    let embed = [table()];
    let call_embed: Call<'_, B> = &|b, t| Ok(vec![b.embedding_forward(&t[0], &ids)?]);
    let embed_bwd = [
        table(),
        Slot::f32_only(values(rows * dim, seed + 1, 1.0), &[rows, dim]),
    ];
    let call_embed_bwd: Call<'_, B> = &|b, t| Ok(vec![b.embedding_backward(&t[0], &ids, &t[1])?]);
    let lce = [
        Slot::f32_only(values(rows * dim, seed + 2, 2.0), &[rows, dim]),
        table(),
    ];
    let chunks = [
        ojas_core::CeChunk {
            rows: 2,
            cols: vocab.div_ceil(3),
        },
        ojas_core::CeChunk {
            rows: rows + 1,
            cols: vocab + 1,
        },
    ];
    let targets = &targets;
    let lce_call = |chunk: ojas_core::CeChunk, want_grad: bool| {
        move |b: &B, t: &[Tensor]| -> Result<Vec<Tensor>, OjasError> {
            let out = b.linear_cross_entropy_mean(&t[0], &t[1], targets, None, chunk, want_grad)?;
            let mut outs = vec![out.loss];
            outs.extend(out.grad_input);
            outs.extend(out.grad_weight);
            Ok(outs)
        }
    };
    let probe = operands(b, &embed, 1, Layout::Packed).map_err(ojas(CHECK))?;
    match b.embedding_forward(&probe[0], &ids) {
        Err(OjasError::Dtype { .. }) => {
            // No table lane: every Bf16 table must be refused the same way.
            report.refusals += 1;
            let bwd = operands(b, &embed_bwd, 1, Layout::Packed).map_err(ojas(CHECK))?;
            let ce = operands(b, &lce, 1 << 1, Layout::Packed).map_err(ojas(CHECK))?;
            let rest: [Refusal<'_>; 2] = [
                (
                    "embedding_backward table",
                    Box::new(|| b.embedding_backward(&bwd[0], &ids, &bwd[1]).map(|_| ())),
                ),
                (
                    "linear_cross_entropy_mean weight",
                    Box::new(|| {
                        b.linear_cross_entropy_mean(&ce[0], &ce[1], targets, None, chunks[0], true)
                            .map(|_| ())
                    }),
                ),
            ];
            for (name, call) in rest {
                match call() {
                    Err(OjasError::Dtype { .. }) => report.refusals += 1,
                    Err(e) => return Err(fail(CHECK, format!("{name}: expected Dtype, got {e}"))),
                    Ok(()) => {
                        return Err(fail(
                            CHECK,
                            format!("{name}: accepted a Bf16 table that embedding_forward refused"),
                        ))
                    }
                }
            }
            return Ok(());
        }
        Err(e) => return Err(fail(CHECK, format!("embedding_forward table: {e}"))),
        Ok(_) => report.table_lane = true,
    }
    let name = format!("embedding_forward [{vocab}, {dim}]");
    report.mixed_calls += every_mix(b, &name, &embed, call_embed)?;
    let name = format!("embedding_backward [{vocab}, {dim}]");
    report.mixed_calls += every_mix(b, &name, &embed_bwd, call_embed_bwd)?;
    for chunk in chunks {
        for want_grad in [false, true] {
            let call = lce_call(chunk, want_grad);
            let name = format!(
                "linear_cross_entropy_mean [{rows}, {dim}] x [{vocab}, {dim}] {chunk:?} \
                 grad {want_grad}"
            );
            report.mixed_calls += every_mix(b, &name, &lce, &call)?;
        }
    }
    refuses_non_finite(b, "embedding_forward", &embed, 0, f32::NAN, call_embed)?;
    refuses_non_finite(
        b,
        "embedding_backward",
        &embed_bwd,
        0,
        f32::INFINITY,
        call_embed_bwd,
    )?;
    let call = lce_call(chunks[0], true);
    refuses_non_finite(b, "linear_cross_entropy_mean", &lce, 1, f32::NAN, &call)?;
    report.non_finite += 3;
    Ok(())
}

/// Every op outside the matmul class and the table lane refuses a `Bf16`
/// operand with [`OjasError::Dtype`], before it reads anything; so do the
/// table ops' other operands.
fn other_ops_refuse<B: Backend + ?Sized>(
    b: &B,
    report: &mut Bf16Report,
) -> Result<(), ParityFailure> {
    const CHECK: &str = "bf16 refused outside the matmul class";
    let up_f32 = |data: &[f32], shape: &[usize]| {
        Tensor::from_f32(data, shape, b.budget()).and_then(|t| b.upload(&t))
    };
    let up_bf16 =
        |data: &[f32], shape: &[usize]| host_bf16(b, data, shape).and_then(|t| b.upload(&t));
    let x = values(4 * 8, 11, 1.0);
    let w = values(8, 12, 1.0);
    let ids = Tensor::from_u32(&[0, 3, 1, 2], &[4], b.budget())
        .and_then(|t| b.upload(&t))
        .map_err(ojas(CHECK))?;
    let table = up_f32(&values(4 * 8, 16, 1.0), &[4, 8]).map_err(ojas(CHECK))?;
    let chunk = ojas_core::CeChunk { rows: 2, cols: 2 };
    let xb = up_bf16(&x, &[4, 8]).map_err(ojas(CHECK))?;
    let xf = up_f32(&x, &[4, 8]).map_err(ojas(CHECK))?;
    let wb = up_bf16(&w, &[8]).map_err(ojas(CHECK))?;
    let wf = up_f32(&w, &[8]).map_err(ojas(CHECK))?;
    let rope = [1usize, 4, 2, 8];
    let rb = up_bf16(&values(64, 13, 1.0), &rope).map_err(ojas(CHECK))?;
    let cos = up_f32(&values(16, 14, 1.0), &[4, 4]).map_err(ojas(CHECK))?;
    let sin = up_f32(&values(16, 15, 1.0), &[4, 4]).map_err(ojas(CHECK))?;
    let lambda = up_f32(&[0.5], &[]).map_err(ojas(CHECK))?;
    let calls: Vec<Refusal<'_>> = vec![
        (
            "embedding_backward grad_output",
            Box::new(|| b.embedding_backward(&table, &ids, &xb).map(|_| ())),
        ),
        (
            "linear_cross_entropy_mean input",
            Box::new(|| {
                b.linear_cross_entropy_mean(&xb, &table, &ids, None, chunk, true)
                    .map(|_| ())
            }),
        ),
        (
            "rms_norm_forward input",
            Box::new(|| b.rms_norm_forward(&xb, &wf, 1e-6).map(|_| ())),
        ),
        (
            "rms_norm_forward weight",
            Box::new(|| b.rms_norm_forward(&xf, &wb, 1e-6).map(|_| ())),
        ),
        (
            "rope_half_split_forward x",
            Box::new(|| b.rope_half_split_forward(&rb, &cos, &sin).map(|_| ())),
        ),
        ("silu_forward", Box::new(|| b.silu_forward(&xb).map(|_| ()))),
        (
            "silu_backward",
            Box::new(|| b.silu_backward(&xf, &xb).map(|_| ())),
        ),
        (
            "mul_forward",
            Box::new(|| b.mul_forward(&xf, &xb).map(|_| ())),
        ),
        (
            "residual_add_forward",
            Box::new(|| b.residual_add_forward(&xb, &xf).map(|_| ())),
        ),
        (
            "residual_add_backward",
            Box::new(|| b.residual_add_backward(&xf, &xf, &xb).map(|_| ())),
        ),
        (
            "value_residual_blend_forward",
            Box::new(|| {
                b.value_residual_blend_forward(&xb, &xf, &lambda)
                    .map(|_| ())
            }),
        ),
        (
            "cross_entropy_mean_forward logits",
            Box::new(|| b.cross_entropy_mean_forward(&xb, &ids, None).map(|_| ())),
        ),
        ("permute", Box::new(|| b.permute(&xb, &[1, 0]).map(|_| ()))),
        ("cast_bf16", Box::new(|| b.cast_bf16(&xb).map(|_| ()))),
    ];
    for (name, call) in calls {
        match call() {
            Err(OjasError::Dtype { .. }) => report.refusals += 1,
            Err(e) => return Err(fail(CHECK, format!("{name}: expected Dtype, got {e}"))),
            Ok(()) => return Err(fail(CHECK, format!("{name}: a Bf16 operand was accepted"))),
        }
    }
    Ok(())
}

/// [`Backend::to_bf16`] and [`Backend::to_f32`] on this backend's tensors
/// against the host conversions, bit for bit: round to nearest even, ties
/// to even, overflow to infinity, and the NaN rule (sign, top payload bits,
/// quiet bit).
fn conversions_match_the_host<B: Backend + ?Sized>(b: &B) -> Result<(), ParityFailure> {
    const CHECK: &str = "bf16 conversions";
    let mut src: Vec<f32> = [
        0x0000_0000u32,
        0x8000_0000,
        0x3f80_8000, // tie, rounds down to even
        0x3f81_8000, // tie, rounds up to even
        0x3f80_8001,
        0x7f7f_ffff, // largest finite: rounds to infinity
        0xff7f_ffff,
        0x7f80_0000,
        0xff80_0000,
        0x0000_0001, // smallest subnormal
        0x807f_ffff,
        0x7fc0_0000,
        0xffd2_b4c6, // signed NaN with a payload
        0x7f80_0001, // signalling NaN, payload only in the dropped bits
        0x7fa1_2345,
    ]
    .iter()
    .map(|&bits| f32::from_bits(bits))
    .collect();
    src.extend(values(37, 21, 3.0).iter().map(|v| v * 1.000_1));
    let n = src.len();
    let host = Tensor::from_f32(&src, &[n], b.budget()).map_err(ojas(CHECK))?;
    let dev = b.upload(&host).map_err(ojas(CHECK))?;
    let narrow = b.to_bf16(&dev).map_err(ojas(CHECK))?;
    if narrow.dtype() != DType::Bf16 {
        return Err(fail(
            CHECK,
            format!("to_bf16 returned {:?}", narrow.dtype()),
        ));
    }
    let got = b
        .download(&narrow)
        .and_then(|t| t.bf16_slice().map(<[u16]>::to_vec))
        .map_err(ojas(CHECK))?;
    if got.len() != n {
        return Err(fail(
            CHECK,
            format!("to_bf16 gave {} values for {n}", got.len()),
        ));
    }
    for (&g, &v) in got.iter().zip(&src) {
        let want = f32_to_bf16(v);
        if g != want {
            return Err(fail(
                CHECK,
                format!(
                    "to_bf16 of {:#010x}: {g:#06x}, the host gives {want:#06x}",
                    v.to_bits()
                ),
            ));
        }
    }
    let wide = b.to_f32(&narrow).map_err(ojas(CHECK))?;
    let back = b
        .download(&wide)
        .and_then(|t| t.to_f32_vec())
        .map_err(ojas(CHECK))?;
    for (&w, &bits) in back.iter().zip(&got) {
        if w.to_bits() != bf16_to_f32(bits).to_bits() {
            return Err(fail(
                CHECK,
                format!("to_f32 of {bits:#06x}: {:#010x}", w.to_bits()),
            ));
        }
    }
    b.sync().map_err(ojas(CHECK))
}

/// Run the whole contract on `b`. A backend that does not report
/// [`Backend::bf16_operands`] fails at once: it has no contract to check.
pub fn bf16_operand_contract<B: Backend + ?Sized>(b: &B) -> Result<Bf16Report, ParityFailure> {
    if !b.bf16_operands() {
        return Err(fail(
            "bf16 operands",
            format!("{:?} does not report bf16_operands()", b.id()),
        ));
    }
    let mut report = Bf16Report::default();
    conversions_match_the_host(b)?;
    linear_cases(b, &[2, 3], 40, 24, 100, &mut report)?;
    linear_cases(b, &[64], 96, 80, 200, &mut report)?;
    sdpa_cases(b, 1, 2, 1, 7, 64, None, 300, &mut report)?;
    sdpa_cases(b, 2, 2, 2, 9, 64, Some(3), 400, &mut report)?;
    table_cases(b, 11, 8, 5, 500, &mut report)?;
    other_ops_refuse(b, &mut report)?;
    b.sync().map_err(ojas("bf16 operands"))?;
    Ok(report)
}

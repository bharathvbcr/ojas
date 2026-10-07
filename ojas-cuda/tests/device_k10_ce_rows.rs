//! K10: cross-entropy over supplied `(row, target)` pairs of the hidden
//! states against the tied head, in vocabulary chunks with an online
//! log-sum-exp, and its gradients `dh` (per supplied row) and `dW`.
//!
//! - **Host** (run everywhere): the host mirror of the whole device walk
//!   (`ce_rows::ce_rows_mirror`, GEMMs on the FFMA engine's exact sequence)
//!   against L-cuda-oracle's float64 one-pass reference on the three torch
//!   goldens' f32-rounded inputs (mean, sum at scale 0.7, and a planted
//!   dominant logit in the last column), at chunks of 7, 64 and 300 (so the
//!   carried sum is rescaled across chunk boundaries, and once not at all),
//!   with f32 and bf16 operands; and at 16,384 and the 2B's 248,320 columns
//!   in production chunks of 8,192 (a 2,560-wide tail).
//! - **Device** (`--features cuda`, `#[ignore]`: an sm_90 GPU; on the Mac they
//!   compile and are NOT RUN): the same cases on the device, bit for bit
//!   against the mirror on the FFMA engine and against a repeat run, cuBLAS's
//!   bf16 engine within the bf16 bounds, and rung a's checks.
//!
//! Bounds, written before any run (`tessl/tests/cross_entropy.rs:12-14,185,
//! 252-258`): every per-row loss and the loss within `1e-5 + 1e-5|ref|`;
//! `dh` and `dW` within `1e-4` of `max|ref|` at f32 operands and `2^-7` at
//! bf16 operands, the reference then evaluated on the bf16-rounded inputs.

mod device_small_common;
mod reference;

use device_small_common::{
    f32s, golden, wide, Goldens, CE_BF16_GRAD_PEAK_REL, CE_EXACT_GRAD_PEAK_REL, CE_LOSS_TOL,
    CE_SOURCE,
};
use ojas_cuda::bf16::round_to_bf16;
use ojas_cuda::ce_rows::{ce_rows_mirror, CeMirror, CeOutput, CeRowsPlan, Reduction, CE_CHUNK};
use ojas_cuda::check::Check;
use ojas_cuda::host_ref::Operands;
use ojas_cuda::inputs::splitmix_f32;
use ojas_cuda::small_common::{peak_check, Window};
use reference::ce::{cross_entropy_rows_f64, CeRows, Reduction as RefReduction};

const GOLDENS: Goldens = Goldens(&[
    golden!("ce_rows_c0_h"),
    golden!("ce_rows_c0_w"),
    golden!("ce_rows_c0_rows"),
    golden!("ce_rows_c0_targets"),
    golden!("ce_rows_c0_scale"),
    golden!("ce_rows_c1_h"),
    golden!("ce_rows_c1_w"),
    golden!("ce_rows_c1_rows"),
    golden!("ce_rows_c1_targets"),
    golden!("ce_rows_c1_scale"),
    golden!("ce_rows_c2_h"),
    golden!("ce_rows_c2_w"),
    golden!("ce_rows_c2_rows"),
    golden!("ce_rows_c2_targets"),
    golden!("ce_rows_c2_scale"),
]);

/// The inputs of one cross-entropy problem, `h` dense `[t_rows, hidden]`.
struct Problem {
    tag: String,
    h: Vec<f32>,
    w: Vec<f32>,
    rows: Vec<u32>,
    targets: Vec<u32>,
    hidden: usize,
    vocab: usize,
    reduction: Reduction,
    scale: f32,
}

impl Problem {
    fn golden(c: usize) -> Self {
        let name = format!("ce_rows_c{c}");
        let (hs, h) = GOLDENS.shaped(&format!("{name}_h"));
        let (ws, w) = GOLDENS.shaped(&format!("{name}_w"));
        let idx = |n: &str| -> Vec<u32> {
            GOLDENS
                .indices(&format!("{name}_{n}"))
                .iter()
                .map(|&i| u32::try_from(i).expect("index fits u32"))
                .collect()
        };
        Problem {
            tag: name.clone(),
            h: f32s(&h),
            w: f32s(&w),
            rows: idx("rows"),
            targets: idx("targets"),
            hidden: hs[1],
            vocab: ws[0],
            // c1 is the golden's sum case (its manifest entry and L-cuda-oracle's test).
            reduction: if c == 1 {
                Reduction::Sum
            } else {
                Reduction::Mean
            },
            scale: GOLDENS.f64s(&format!("{name}_scale"))[0] as f32,
        }
    }

    /// `vocab` columns at a small hidden size, a dominant logit planted in the
    /// last column for the first row's target.
    fn wide_vocab(vocab: usize, hidden: usize, n: usize) -> Self {
        let t_rows = n + 4;
        let h = splitmix_f32(0x10c1, t_rows * hidden, 1.0);
        let mut w = splitmix_f32(0x10c2, vocab * hidden, 0.25);
        let rows: Vec<u32> = (0..n).map(|i| ((i * 3) % t_rows) as u32).collect();
        let mut targets: Vec<u32> = (0..n).map(|i| ((i * 7919 + 11) % vocab) as u32).collect();
        targets[0] = (vocab - 1) as u32;
        let src = rows[0] as usize * hidden;
        for k in 0..hidden {
            w[(vocab - 1) * hidden + k] = 4.0 * h[src + k];
        }
        Problem {
            tag: format!("v{vocab}_h{hidden}_n{n}"),
            h,
            w,
            rows,
            targets,
            hidden,
            vocab,
            reduction: Reduction::Mean,
            scale: 1.0,
        }
    }

    fn t_rows(&self) -> usize {
        self.h.len() / self.hidden
    }

    /// `h` in a window of stride `hidden + 5` at column 2.
    fn h_window(&self) -> (Window, Vec<f32>) {
        let (ld, off) = (self.hidden + 5, 2);
        let mut buf = splitmix_f32(0x10c3, self.t_rows() * ld, 7.0);
        for r in 0..self.t_rows() {
            buf[r * ld + off..][..self.hidden]
                .copy_from_slice(&self.h[r * self.hidden..][..self.hidden]);
        }
        (
            Window {
                ld: ld as u64,
                off: off as u64,
            },
            buf,
        )
    }

    fn plan(&self, chunk: u64) -> CeRowsPlan {
        let (win, _) = self.h_window();
        CeRowsPlan::new(
            (
                self.rows.len() as u64,
                self.hidden as u64,
                self.vocab as u64,
            ),
            chunk,
            (self.t_rows() as u64, win),
            self.reduction,
        )
        .expect("plan")
    }

    /// The float64 reference on the inputs the kernels see at `operands`.
    fn reference(&self, operands: Operands) -> CeRows {
        let round = |x: &[f32]| -> Vec<f64> {
            match operands {
                Operands::ExactF32 => wide(x),
                Operands::Bf16 => x.iter().map(|&v| f64::from(round_to_bf16(v))).collect(),
            }
        };
        let to_usize = |v: &[u32]| -> Vec<usize> { v.iter().map(|&i| i as usize).collect() };
        let red = match self.reduction {
            Reduction::Mean => RefReduction::Mean,
            Reduction::Sum => RefReduction::Sum,
        };
        cross_entropy_rows_f64(
            &round(&self.h),
            &round(&self.w),
            &to_usize(&self.rows),
            &to_usize(&self.targets),
            self.hidden,
            self.vocab,
            red,
            f64::from(self.scale),
        )
    }

    fn mirror(&self, chunk: u64, operands: Operands) -> CeMirror {
        let (_, hbuf) = self.h_window();
        ce_rows_mirror(
            &self.plan(chunk),
            &hbuf,
            &self.w,
            (&self.rows, &self.targets),
            operands,
            Some(self.scale),
        )
        .expect("mirror")
    }
}

fn loss_check(label: &str, got: &CeOutput, want: &CeRows) -> Check {
    let (abs, rel) = CE_LOSS_TOL;
    let pairs: Vec<(f64, f64)> = got
        .per_row
        .iter()
        .copied()
        .zip(want.per_row.iter().copied())
        .chain([(got.loss, want.loss)])
        .collect();
    let worst = pairs
        .iter()
        .map(|(g, w)| (g - w).abs() / (abs + rel * w.abs()))
        .fold(0.0f64, f64::max);
    let ok = got.per_row.len() == want.per_row.len()
        && pairs.iter().all(|(g, _)| g.is_finite())
        && worst <= 1.0;
    let detail = format!(
        "{} per-row losses and the loss: worst |err| / (1e-5 + 1e-5|ref|) = {worst:.3e} ({CE_SOURCE})",
        want.per_row.len()
    );
    if ok {
        Check::pass(label, detail)
    } else {
        Check::fail(label, detail)
    }
}

/// The loss, `dh` and `dW` against the reference at `operands`' bound.
fn judge(
    label: &str,
    (out, dh, dw): (&CeOutput, &[f32], &[f32]),
    want: &CeRows,
    operands: Operands,
) -> Vec<Check> {
    let rel = match operands {
        Operands::ExactF32 => CE_EXACT_GRAD_PEAK_REL,
        Operands::Bf16 => CE_BF16_GRAD_PEAK_REL,
    };
    vec![
        loss_check(&format!("{label} loss"), out, want),
        peak_check(&format!("{label} dh"), dh, &want.dh, rel, CE_SOURCE),
        peak_check(&format!("{label} dW"), dw, &want.dw, rel, CE_SOURCE),
    ]
}

/// Every (problem, chunk, operands) the suites run.
fn cases() -> Vec<(Problem, u64, Operands)> {
    let mut out = Vec::new();
    for c in 0..3 {
        for chunk in [7u64, 64, 300] {
            for operands in [Operands::ExactF32, Operands::Bf16] {
                out.push((Problem::golden(c), chunk, operands));
            }
        }
    }
    out.push((
        Problem::wide_vocab(16_384, 32, 4),
        CE_CHUNK,
        Operands::ExactF32,
    ));
    out.push((
        Problem::wide_vocab(248_320, 16, 3),
        CE_CHUNK,
        Operands::ExactF32,
    ));
    out
}

#[test]
fn the_embedded_goldens_are_l_cuda_oracles_pinned_bytes() {
    assert_eq!(GOLDENS.verify(), 15);
}

#[test]
fn the_mirror_matches_the_one_pass_reference_whatever_the_chunk_and_operands() {
    let mut checks = Vec::new();
    for (p, chunk, operands) in cases() {
        let label = format!("mirror {} chunk {chunk} {}", p.tag, operands.name());
        let m = p.mirror(chunk, operands);
        let (dh, dw) = (m.dh.clone().expect("dh"), m.dw.clone().expect("dW"));
        checks.extend(judge(
            &label,
            (&m.out, &dh, &dw),
            &p.reference(operands),
            operands,
        ));
    }
    device_small_common::assert_pass(&checks);
}

#[cfg(feature = "cuda")]
mod device {
    use super::*;
    use device_small_common::{assert_bits, assert_pass, runtime};
    use ojas_cuda::ce_rows_cuda::{ce_rows, CeGrads, CeWorkspace};
    use ojas_cuda::gemm::Bf16Engine;
    use ojas_cuda::k0::GatherSource;
    use ojas_cuda::runtime::CudaRuntime;
    use ojas_cuda::small_smoke::k10_checks;

    fn run(
        rt: &CudaRuntime,
        p: &Problem,
        chunk: u64,
        (operands, engine): (Operands, Bf16Engine),
    ) -> (CeOutput, Vec<f32>, Vec<f32>) {
        let plan = p.plan(chunk);
        let (_, hbuf) = p.h_window();
        let mut ws = CeWorkspace::new(rt, plan).expect("workspace");
        let h = rt.upload(&hbuf, "h").expect("h");
        let w = rt.upload(&p.w, "w").expect("w");
        let mut dh = rt
            .upload(&vec![f32::NAN; p.rows.len() * p.hidden], "dh")
            .expect("dh");
        let mut dw = rt.upload(&vec![f32::NAN; p.w.len()], "dW").expect("dW");
        let grads = CeGrads {
            scale: p.scale,
            dh: &mut dh,
            dw: &mut dw,
        };
        let out = ce_rows(
            rt,
            &mut ws,
            GatherSource::F32(&h),
            &w,
            (&p.rows, &p.targets),
            (operands, engine),
            Some(grads),
        )
        .expect("ce_rows");
        (
            out,
            rt.download(&dh).expect("dh"),
            rt.download(&dw).expect("dW"),
        )
    }

    fn loss_bits(label: &str, got: &CeOutput, want: &CeOutput) {
        let bits = |o: &CeOutput| -> Vec<u64> {
            o.per_row
                .iter()
                .chain([&o.loss])
                .map(|v| v.to_bits())
                .collect()
        };
        assert_eq!(
            bits(got),
            bits(want),
            "{label}: losses differ: {got:?} vs {want:?}"
        );
    }

    #[test]
    #[ignore = "needs an sm_90 NVIDIA GPU; run on the box with --ignored"]
    fn the_device_walk_on_the_ffma_engine_matches_the_reference_and_the_mirror_bitwise() {
        let rt = runtime();
        let mut checks = Vec::new();
        for (p, chunk, operands) in cases() {
            let label = format!("device ffma {} chunk {chunk} {}", p.tag, operands.name());
            let got = run(&rt, &p, chunk, (operands, Bf16Engine::Ffma));
            let again = run(&rt, &p, chunk, (operands, Bf16Engine::Ffma));
            let m = p.mirror(chunk, operands);
            loss_bits(&format!("{label} vs mirror"), &got.0, &m.out);
            assert_bits(
                &format!("{label} dh vs mirror"),
                &got.1,
                m.dh.as_deref().expect("dh"),
            );
            assert_bits(
                &format!("{label} dW vs mirror"),
                &got.2,
                m.dw.as_deref().expect("dW"),
            );
            loss_bits(&format!("{label} repeat"), &again.0, &got.0);
            assert_bits(&format!("{label} dh repeat"), &again.1, &got.1);
            assert_bits(&format!("{label} dW repeat"), &again.2, &got.2);
            checks.extend(judge(
                &label,
                (&got.0, &got.1, &got.2),
                &p.reference(operands),
                operands,
            ));
        }
        assert_pass(&checks);
    }

    #[test]
    #[ignore = "needs an sm_90 NVIDIA GPU; run on the box with --ignored"]
    fn the_device_walk_on_cublas_bf16_matches_the_reference_within_the_bf16_bounds() {
        let rt = runtime();
        let mut checks = Vec::new();
        for c in 0..3 {
            let p = Problem::golden(c);
            for chunk in [7u64, 64, 300] {
                let label = format!("device cublas {} chunk {chunk} bf16", p.tag);
                let got = run(&rt, &p, chunk, (Operands::Bf16, Bf16Engine::Cublas));
                let again = run(&rt, &p, chunk, (Operands::Bf16, Bf16Engine::Cublas));
                loss_bits(&format!("{label} repeat"), &again.0, &got.0);
                assert_bits(&format!("{label} dh repeat"), &again.1, &got.1);
                assert_bits(&format!("{label} dW repeat"), &again.2, &got.2);
                checks.extend(judge(
                    &label,
                    (&got.0, &got.1, &got.2),
                    &p.reference(Operands::Bf16),
                    Operands::Bf16,
                ));
            }
        }
        assert_pass(&checks);
    }

    #[test]
    #[ignore = "needs an sm_90 NVIDIA GPU; run on the box with --ignored"]
    fn rung_a_k10_checks_pass() {
        assert_pass(&k10_checks(&runtime()));
    }
}

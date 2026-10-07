//! K2(i) on the device: the published-rule GDN scan, forward and backward,
//! against L-cuda-oracle's float64 reference and bit for bit against the host
//! mirror (`gdn_host`). Every test needs an sm_90 NVIDIA GPU, so each is
//! `#[ignore]`: on the Mac they compile and are reported NOT RUN. On the box:
//! `flock /home/ubuntu/queue/gpu.lock timeout 900 env LD_LIBRARY_PATH=<cublas lib>:<nvrtc lib>
//! ./device_gdn_published-<hash> --ignored --test-threads=1`.
//!
//! **Bounds, written before any run** (`gdn_host::published_bounds`):
//! - every output (o, s_fin, ckpt, dq, dk, dv, dg, dbeta, ds0) within
//!   `1e-4 * max |ref|` of the float64 reference, tessl's own Metal-vs-f64
//!   bound (`tessl/tests/gdn_train.rs:227-249`, `rel <= 1e-4` at `:240-241`),
//!   argued against the f32 accumulation width in `published_bounds`;
//! - the corpus forward within `1e-4 * max |golden|` of tessl's float64
//!   golden after undoing the seam's q normalization;
//! - every output **bit-identical** to the host mirror (the kernels round
//!   every operation explicitly; the mirror replays them), and to a repeat run;
//! - a batched variable-length launch bit-identical to its sequences' `B = 1`
//!   launches;
//! - no output element left at the sentinel (tessl `tests/gdn_train.rs:181-182`).
//!
//! The corpus is embedded in this binary (`device_gdn_published_common`) and
//! its bytes checked against tessl's pinned sums, so the binary needs no
//! checkout on the box. The timing test is report-only: it prints a JSON
//! line and asserts only that it measured.
#![cfg(feature = "cuda")]

mod device_gdn_published_common;
mod reference;

use device_gdn_published_common::{
    judge, judge_corpus_golden, load_corpus, reference_outputs, verify_embedded_corpus,
    PUBLISHED_CORPUS,
};
use ojas_cuda::check::{Check, Status};
use ojas_cuda::gdn::{
    gdn_published_backward, gdn_published_forward, GdnPublishedDeviceGrads, GdnPublishedLayout,
    GdnPublishedWorkspace,
};
use ojas_cuda::gdn_host::{
    gdn_published_mirror, GdnPublishedCase, GdnPublishedOutputs, TESSL_EDGES,
};
use ojas_cuda::gdn_plan::GdnPublishedPlan;
use ojas_cuda::gdn_smoke::{
    bitwise_outputs, gdn_published_checks, gdn_published_timing, run_published_case,
    run_published_forward, DeviceCase, SENTINEL,
};
use ojas_cuda::runtime::{CudaRuntime, RuntimeConfig};
use ojas_cuda::CudaError;

fn runtime(budget_bytes: u64) -> CudaRuntime {
    CudaRuntime::open(RuntimeConfig {
        budget_bytes,
        ..RuntimeConfig::default()
    })
    .unwrap_or_else(|e| panic!("CudaRuntime::open: {e}"))
}

fn assert_all_pass(checks: &[Check]) {
    assert!(!checks.is_empty(), "no checks ran");
    let bad: Vec<String> = checks
        .iter()
        .filter(|c| c.status != Status::Pass)
        .map(|c| format!("{} {}: {}", c.status.name(), c.name, c.detail))
        .collect();
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

fn device_outputs(rt: &CudaRuntime, case: &GdnPublishedCase) -> GdnPublishedOutputs {
    let dc = DeviceCase::upload(rt, case).unwrap_or_else(|e| panic!("{}: upload: {e}", case.label));
    run_published_case(rt, &dc).unwrap_or_else(|e| panic!("{}: {e}", case.label))
}

/// The float64 bound, bit equality with the mirror, and two repeat runs
/// bit-identical to the first. Returns the worst ratio to the bound's base.
fn check_case(rt: &CudaRuntime, case: &GdnPublishedCase) -> f64 {
    let label = format!("device {}", case.label);
    let first = device_outputs(rt, case);
    let worst = judge(&label, &first, &reference_outputs(case), Some(SENTINEL));
    let mirror = gdn_published_mirror(case).unwrap_or_else(|e| panic!("{label}: mirror: {e}"));
    assert_all_pass(&bitwise_outputs(
        &format!("{label} vs mirror"),
        &first,
        &mirror,
    ));
    for run in 1..3 {
        let again = device_outputs(rt, case);
        assert_all_pass(&bitwise_outputs(
            &format!("{label} repeat {run}"),
            &again,
            &first,
        ));
    }
    worst
}

#[test]
#[ignore = "needs an sm_90 NVIDIA GPU; run on the box with --ignored"]
fn published_gdn_device_checks_pass() {
    let rt = runtime(4 << 30);
    assert_all_pass(&gdn_published_checks(&rt));
}

#[test]
#[ignore = "needs an sm_90 NVIDIA GPU; run on the box with --ignored"]
fn published_gdn_matches_the_f64_reference_across_chunk_edges_b2() {
    let rt = runtime(4 << 30);
    for (t, s0, dfin) in TESSL_EDGES {
        check_case(
            &rt,
            &GdnPublishedCase::tessl(2, t, 3, 32, 100 + t as u64, s0, dfin).unwrap(),
        );
    }
}

#[test]
#[ignore = "needs an sm_90 NVIDIA GPU; run on the box with --ignored"]
fn published_gdn_matches_the_f64_reference_across_chunk_edges_b1() {
    let rt = runtime(4 << 30);
    for (t, s0, dfin) in TESSL_EDGES {
        check_case(
            &rt,
            &GdnPublishedCase::tessl(1, t, 3, 32, 200 + t as u64, s0, dfin).unwrap(),
        );
    }
}

#[test]
#[ignore = "needs an sm_90 NVIDIA GPU; run on the box with --ignored"]
fn published_gdn_matches_the_f64_reference_at_qwen35_2b_heads() {
    let rt = runtime(4 << 30);
    check_case(
        &rt,
        &GdnPublishedCase::tessl(1, 200, 16, 128, 7, false, false).unwrap(),
    );
}

/// One batched launch of variable lengths: within the bound of each
/// sequence's float64 reference, and bit-identical to launching each
/// sequence alone at `B = 1`.
#[test]
#[ignore = "needs an sm_90 NVIDIA GPU; run on the box with --ignored"]
fn published_gdn_varlen_batch_matches_references_and_its_b1_launches_bitwise() {
    let rt = runtime(4 << 30);
    for case in [
        GdnPublishedCase::varlen(&[1, 63, 64, 65, 130], 3, 32, 500, true, true).unwrap(),
        GdnPublishedCase::varlen(&[130, 1, 65, 64, 63], 2, 48, 77, true, true).unwrap(),
        GdnPublishedCase::varlen(&[200, 64, 1, 130], 16, 128, 31, false, false).unwrap(),
    ] {
        check_case(&rt, &case);
        let all = device_outputs(&rt, &case);
        let p = &case.plan;
        let bits = |x: &[f32]| x.iter().map(|v| v.to_bits()).collect::<Vec<u32>>();
        for b in 0..p.batch() {
            let one = device_outputs(&rt, &case.sequence(b).unwrap());
            let label = format!("{} sequence {b}", case.label);
            for (name, got, width) in [
                ("o", &one.o, p.v_dim()),
                ("dv", &one.dv, p.v_dim()),
                ("dq", &one.dq, 128),
                ("dk", &one.dk, 128),
                ("dg", &one.dg, 1),
                ("dbeta", &one.dbeta, 1),
            ] {
                let all_t = match name {
                    "o" => &all.o,
                    "dv" => &all.dv,
                    "dq" => &all.dq,
                    "dk" => &all.dk,
                    "dg" => &all.dg,
                    _ => &all.dbeta,
                };
                assert_eq!(
                    bits(got),
                    bits(p.split_tokens(all_t, width).unwrap()[b]),
                    "{label}: {name}"
                );
            }
            assert_eq!(
                bits(&one.ckpt),
                bits(p.split_ckpt(&all.ckpt).unwrap()[b]),
                "{label}: ckpt"
            );
            assert_eq!(
                bits(&one.s_fin),
                bits(p.split_states(&all.s_fin).unwrap()[b]),
                "{label}: s_fin"
            );
            if let (Some(o1), Some(oa)) = (&one.ds0, &all.ds0) {
                assert_eq!(
                    bits(o1),
                    bits(p.split_states(oa).unwrap()[b]),
                    "{label}: ds0"
                );
            }
        }
    }
}

/// tessl's 13 published cases embedded in the kernels' shape: forward and
/// backward against the float64 seam reference, the forward against the
/// corpus's own golden, and every output against the mirror bit for bit.
#[test]
#[ignore = "needs an sm_90 NVIDIA GPU; run on the box with --ignored"]
fn published_gdn_matches_the_published_corpus() {
    assert_eq!(verify_embedded_corpus(), 13 * 6);
    let rt = runtime(4 << 30);
    for (i, files) in PUBLISHED_CORPUS.iter().enumerate() {
        let cc = load_corpus(files, 9000 + i as u64);
        let label = format!("device {}", cc.case.label);
        let want = reference_outputs(&cc.case);
        let got = device_outputs(&rt, &cc.case);
        judge(&label, &got, &want, Some(SENTINEL));
        judge_corpus_golden(&cc, &got.o, &want);
        let mirror = gdn_published_mirror(&cc.case).unwrap();
        assert_all_pass(&bitwise_outputs(
            &format!("{label} vs mirror"),
            &got,
            &mirror,
        ));
    }
}

/// The final state is optional: without it the forward writes the same `o`
/// and checkpoints, bit for bit.
#[test]
#[ignore = "needs an sm_90 NVIDIA GPU; run on the box with --ignored"]
fn published_gdn_forward_without_s_fin_writes_the_same_o_and_ckpt() {
    let rt = runtime(4 << 30);
    let case = GdnPublishedCase::tessl(2, 130, 3, 32, 230, true, true).unwrap();
    let dc = DeviceCase::upload(&rt, &case).unwrap();
    let (o1, s1, c1) = run_published_forward(&rt, &dc, true).unwrap();
    let (o2, s2, c2) = run_published_forward(&rt, &dc, false).unwrap();
    assert!(s1.is_some() && s2.is_none());
    let bits = |x: &[f32]| x.iter().map(|v| v.to_bits()).collect::<Vec<u32>>();
    assert_eq!(bits(&o1), bits(&o2));
    assert_eq!(bits(&c1), bits(&c2));
}

/// Repeat-run bit equality at Qwen3.5-2B's heads, batched: five runs from
/// fresh buffers.
#[test]
#[ignore = "needs an sm_90 NVIDIA GPU; run on the box with --ignored"]
fn published_gdn_is_bit_identical_over_five_runs_at_2b_heads_batched() {
    let rt = runtime(4 << 30);
    let case = GdnPublishedCase::varlen(&[200, 64, 1, 130], 16, 128, 41, true, true).unwrap();
    let first = device_outputs(&rt, &case);
    for run in 1..5 {
        assert_all_pass(&bitwise_outputs(
            &format!("{} run {run}", case.label),
            &device_outputs(&rt, &case),
            &first,
        ));
    }
}

/// What the device entry points refuse before queuing anything: a workspace
/// or layout for another plan, an unpaired `s0` / `ds0`, a wrong length.
#[test]
#[ignore = "needs an sm_90 NVIDIA GPU; run on the box with --ignored"]
fn published_gdn_refuses_what_the_kernels_cannot_run() {
    let rt = runtime(1 << 30);
    let case = GdnPublishedCase::tessl(1, 5, 2, 32, 3, true, false).unwrap();
    let p = &case.plan;
    let dc = DeviceCase::upload(&rt, &case).unwrap();
    let z = |n: usize| rt.alloc_zeros::<f32>(n, "refusal").unwrap();
    let refused = |r: Result<(), CudaError>, needle: &str| {
        let e = r.expect_err(needle).to_string();
        assert!(e.contains(needle), "{e:?} lacks {needle:?}");
    };
    let (mut o, mut ckpt) = (z(p.v_len()), z(p.ckpt_len()));
    gdn_published_forward(&rt, &dc.layout, dc.inputs(), &mut o, None, &mut ckpt)
        .expect("valid forward");
    let mut short = z(p.v_len() - 1);
    refused(
        gdn_published_forward(&rt, &dc.layout, dc.inputs(), &mut short, None, &mut ckpt),
        "o has",
    );
    let mut sf = z(p.state_len() + 1);
    refused(
        gdn_published_forward(
            &rt,
            &dc.layout,
            dc.inputs(),
            &mut o,
            Some(&mut sf),
            &mut ckpt,
        ),
        "s_fin has",
    );

    let mut ws = GdnPublishedWorkspace::new(&rt, p).unwrap();
    let (mut dq, mut dk, mut dv, mut dg, mut db) = (
        z(p.qk_len()),
        z(p.qk_len()),
        z(p.v_len()),
        z(p.gate_len()),
        z(p.gate_len()),
    );
    let mut ds0 = z(p.state_len());
    let d_o = z(p.v_len());
    let mut bwd = |ws: &mut GdnPublishedWorkspace, with_ds0: bool, layout: &GdnPublishedLayout| {
        gdn_published_backward(
            &rt,
            layout,
            dc.inputs(),
            &ckpt,
            &d_o,
            None,
            ws,
            GdnPublishedDeviceGrads {
                dq: &mut dq,
                dk: &mut dk,
                dv: &mut dv,
                dg: &mut dg,
                dbeta: &mut db,
                ds0: if with_ds0 { Some(&mut ds0) } else { None },
            },
        )
    };
    bwd(&mut ws, true, &dc.layout).expect("valid backward");
    refused(bwd(&mut ws, false, &dc.layout), "ds0 is required");
    let other = GdnPublishedPlan::dense(1, 6, 2, 128, 32).unwrap();
    let mut ws_other = GdnPublishedWorkspace::new(&rt, &other).unwrap();
    refused(bwd(&mut ws_other, true, &dc.layout), "the workspace is for");
    let layout_other = GdnPublishedLayout::upload(&rt, &other).unwrap();
    refused(bwd(&mut ws_other, true, &layout_other), "q has");

    let no_s0 = GdnPublishedCase::tessl(1, 5, 2, 32, 3, false, false).unwrap();
    let dc2 = DeviceCase::upload(&rt, &no_s0).unwrap();
    let mut ds0b = z(p.state_len());
    refused(
        gdn_published_backward(
            &rt,
            &dc2.layout,
            dc2.inputs(),
            &ckpt,
            &d_o,
            None,
            &mut ws,
            GdnPublishedDeviceGrads {
                dq: &mut dq,
                dk: &mut dk,
                dv: &mut dv,
                dg: &mut dg,
                dbeta: &mut db,
                ds0: Some(&mut ds0b),
            },
        ),
        "no initial state",
    );
}

/// Report-only (Fable's decision 6 trigger): one GDN layer at Qwen3.5-2B's
/// heads (H = 16, Dv = 128), the step's 4 sequences as `B = 1` launches in
/// series against one batched launch, at 4 x 2048 and 4 x 8192 tokens. Prints
/// one JSON object per shape with the projection `18 x (2 fwd + bwd)`
/// beside PyTorch's whole step (1.2-1.9 s). Asserts only that it measured,
/// and that the batched outputs equal the `B = 1` outputs bit for bit.
#[test]
#[ignore = "needs an sm_90 NVIDIA GPU (about 15 GiB free); run on the box with --ignored"]
fn published_gdn_timing_b1_vs_batched_report_only() {
    let rt = runtime(24 << 30);
    let info = rt.info().clone();
    for lens in [vec![2048usize; 4], vec![8192usize; 4]] {
        let t = gdn_published_timing(&rt, &lens, 16, 128, 3).unwrap_or_else(|e| {
            panic!(
                "timing NOT RUN for lens {lens:?}: {} ({e}); this is not a measurement",
                e.kind()
            )
        });
        let json = ojas_cuda::json::JsonObj::new()
            .with("device", info.name.as_str())
            .with("sm_count", info.sm_count)
            .with("driver_version", info.driver_version)
            .with("timing", t.to_json());
        println!(
            "GDN_PUBLISHED_TIMING {}",
            ojas_cuda::json::Json::from(json).render()
        );
        for (name, ms) in [
            ("batched_fwd_ms", t.batched_fwd_ms),
            ("batched_bwd_ms", t.batched_bwd_ms),
            ("serial_fwd_ms", t.serial_fwd_ms),
            ("serial_bwd_ms", t.serial_bwd_ms),
        ] {
            assert!(
                ms.is_finite() && ms > 0.0,
                "{name} = {ms}: not a measurement"
            );
        }
        assert!(
            t.batched_equals_serial_bitwise,
            "batched outputs differ from the B = 1 outputs"
        );
    }
}

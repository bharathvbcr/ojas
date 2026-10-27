//! The CPU backend against SpikeTrack's real SAM 3.1 fixtures
//! (`ojas_oracle::sam_fixtures`). Every test needs the 3.4 GB file named by
//! `OJAS_SAM_FIXTURES`, so they are ignored by default; run them with
//!
//! ```text
//! OJAS_SAM_FIXTURES=.../SpikeTrack/fixtures/backbone/visem_22_f0.safetensors \
//!   cargo test -p ojas-oracle --test sam_fixtures_cpu -- --ignored --nocapture
//! ```
//!
//! The GroupNorm test reads a different file, named by
//! `OJAS_SAM_GROUPNORM_FIXTURES` (`.../SpikeTrack/fixtures/groupnorm/
//! pixel_decoder.safetensors`, 212 MB); run it in release:
//!
//! ```text
//! OJAS_SAM_GROUPNORM_FIXTURES=.../groupnorm/pixel_decoder.safetensors \
//!   cargo test --release -p ojas-oracle --test sam_fixtures_cpu \
//!   group_norm -- --ignored --nocapture
//! ```
//!
//! The weight-bearing ops (LayerNorm with its affine, linear with bias,
//! conv2d, conv_transpose2d) also need the converted SAM 3.1 checkpoint named
//! by `OJAS_SAM_WEIGHTS` (3.5 GB, read lazily, one tensor at a time). Run them
//! in release with both variables set:
//!
//! ```text
//! OJAS_SAM_FIXTURES=.../SpikeTrack/fixtures/backbone/visem_22_f0.safetensors \
//! OJAS_SAM_WEIGHTS=.../models/sam3.1/sam3.1_multiplex.safetensors \
//!   cargo test --release -p ojas-oracle --test sam_fixtures_cpu \
//!   -- --ignored --nocapture --test-threads=1
//! ```
//!
//! With the variable unset an ignored test FAILS: an unrun check must never
//! report the same result as a passing one. Each test prints its rows as a
//! Markdown table.

use ojas_core::{Budget, Numerics};
use ojas_cpu::CpuBackend;
use ojas_oracle::sam_fixtures::{
    attention, axial_tables, conv2d_rows, conv_transpose2d_rows, gelu, group_norm,
    layer_norm_affine_recovered, layer_norm_rows, linear_bias_rows, rope_interleaved, skipped_ops,
    weighted_ops, Fixture, OpReport, ENV_FIXTURES, ENV_GROUPNORM_FIXTURES, ENV_WEIGHTS, VIT_LN_EPS,
};

fn fixture() -> Fixture<'static> {
    Fixture::from_env()
        .unwrap_or_else(|e| panic!("{ENV_FIXTURES}: {e}"))
        .unwrap_or_else(|| panic!("{ENV_FIXTURES} is not set: this test cannot run"))
}

fn weights() -> Fixture<'static> {
    Fixture::from_env_var(ENV_WEIGHTS)
        .unwrap_or_else(|e| panic!("{ENV_WEIGHTS}: {e}"))
        .unwrap_or_else(|| panic!("{ENV_WEIGHTS} is not set: this test cannot run"))
}

/// 16 GiB. The largest call is the neck's 3x3 convolution at 288x288, whose
/// im2col matrix is 82,944 x 2,304 f32 (764 MB) beside its 85 MB input and
/// output; the tensors read and uploaded for a row come to a few hundred MB.
fn cpu() -> CpuBackend {
    CpuBackend::new(Budget::new(1 << 34)).with_numerics(Numerics::Exact)
}

fn show(rows: &[OpReport]) {
    println!("{}", OpReport::header());
    for r in rows {
        println!("{r}");
    }
    let failed: Vec<String> = rows
        .iter()
        .filter(|r| !r.pass)
        .map(|r| format!("{} [{}]", r.op, r.detail))
        .collect();
    assert!(failed.is_empty(), "outside tolerance: {failed:#?}");
}

#[test]
#[ignore = "needs OJAS_SAM_FIXTURES (SpikeTrack fixtures)"]
fn rope_tables_match_the_recorded_freqs_cis() {
    let (fx, be) = (fixture(), cpu());
    show(&[
        axial_tables(&be, &fx, "trunk.blocks.0", [24, 24], 1.0).unwrap(),
        // 24/72 as python computes it (a double), rounded once to f32.
        axial_tables(
            &be,
            &fx,
            "trunk.blocks.7",
            [72, 72],
            (24.0f64 / 72.0) as f32,
        )
        .unwrap(),
    ]);
}

#[test]
#[ignore = "needs OJAS_SAM_FIXTURES (SpikeTrack fixtures)"]
fn rope_interleaved_matches_torch() {
    let (fx, be) = (fixture(), cpu());
    let mut rows = Vec::new();
    for block in ["trunk.blocks.0", "trunk.blocks.7"] {
        for stream in ["q", "k"] {
            rows.push(rope_interleaved(&be, &fx, block, stream).unwrap());
        }
    }
    show(&rows);
}

#[test]
#[ignore = "needs OJAS_SAM_FIXTURES (SpikeTrack fixtures)"]
fn gelu_erf_matches_torch() {
    let (fx, be) = (fixture(), cpu());
    show(&[
        gelu(
            &be,
            &fx,
            "trunk.blocks.0.mlp.fc1.out",
            "trunk.blocks.0.mlp.gelu.out",
        )
        .unwrap(),
        gelu(
            &be,
            &fx,
            "trunk.blocks.7.mlp.fc1.out",
            "trunk.blocks.7.mlp.gelu.out",
        )
        .unwrap(),
        gelu(&be, &fx, "neck.convs.0.gelu.in", "neck.convs.0.gelu").unwrap(),
    ]);
}

#[test]
#[ignore = "needs OJAS_SAM_FIXTURES (SpikeTrack fixtures)"]
fn windowed_attention_matches_torch() {
    let (fx, be) = (fixture(), cpu());
    show(&[attention(&be, &fx, "trunk.blocks.0", None).unwrap()]);
}

#[test]
#[ignore = "needs OJAS_SAM_FIXTURES (SpikeTrack fixtures)"]
fn global_attention_matches_torch() {
    let (fx, be) = (fixture(), cpu());
    let started = std::time::Instant::now();
    let row = attention(&be, &fx, "trunk.blocks.7", None).unwrap();
    println!("global attention, all heads: {:?}", started.elapsed());
    show(&[row]);
}

#[test]
#[ignore = "needs OJAS_SAM_FIXTURES (SpikeTrack fixtures)"]
fn layer_norm_rows_match_torch_with_the_affine_part_recovered() {
    let (fx, be) = (fixture(), cpu());
    let mut rows = vec![layer_norm_affine_recovered(
        &be,
        &fx,
        "trunk.ln_pre.out.in",
        "trunk.ln_pre.out",
        VIT_LN_EPS,
    )
    .unwrap()];
    for block in ["trunk.blocks.0", "trunk.blocks.7"] {
        for norm in ["norm1", "norm2"] {
            rows.push(
                layer_norm_affine_recovered(
                    &be,
                    &fx,
                    &format!("{block}.{norm}.in"),
                    &format!("{block}.{norm}"),
                    VIT_LN_EPS,
                )
                .unwrap(),
            );
        }
    }
    show(&rows);
}

#[test]
#[ignore = "needs OJAS_SAM_FIXTURES (SpikeTrack fixtures)"]
fn the_ops_the_fixture_cannot_check_are_recorded() {
    let fx = fixture();
    let skipped = skipped_ops(&fx);
    assert!(
        !skipped.is_empty(),
        "the fixture now holds weights: add the weight-bearing ops"
    );
    // With the weights file named, each of those ops is checked (by the tests
    // below) and is reported as CHECKED; without it, SKIPPED as before.
    let Some(w) =
        Fixture::from_env_var(ENV_WEIGHTS).unwrap_or_else(|e| panic!("{ENV_WEIGHTS}: {e}"))
    else {
        for s in &skipped {
            println!("SKIPPED {}: {} ({})", s.op, s.tensors, s.reason);
        }
        return;
    };
    let checked = weighted_ops(&w).unwrap();
    for s in &skipped {
        match checked.iter().find(|c| c.op == s.op) {
            Some(c) => println!("CHECKED {}: {} (from {ENV_WEIGHTS})", c.op, c.tensors),
            None => panic!(
                "SKIPPED {}: {}: the weights file does not cover it",
                s.op, s.tensors
            ),
        }
    }
    assert_eq!(
        checked.len(),
        skipped.len(),
        "an op checked but never skipped"
    );
}

#[test]
#[ignore = "needs OJAS_SAM_FIXTURES and OJAS_SAM_WEIGHTS"]
fn layer_norm_with_the_true_affine_matches_torch() {
    let (fx, w, be) = (fixture(), weights(), cpu());
    show(&layer_norm_rows(&be, &fx, &w).unwrap());
}

#[test]
#[ignore = "needs OJAS_SAM_FIXTURES and OJAS_SAM_WEIGHTS"]
fn linear_bias_matches_torch() {
    let (fx, w, be) = (fixture(), weights(), cpu());
    show(&linear_bias_rows(&be, &fx, &w).unwrap());
}

#[test]
#[ignore = "needs OJAS_SAM_FIXTURES and OJAS_SAM_WEIGHTS"]
fn conv2d_matches_torch() {
    let (fx, w, be) = (fixture(), weights(), cpu());
    show(&conv2d_rows(&be, &fx, &w).unwrap());
}

#[test]
#[ignore = "needs OJAS_SAM_FIXTURES and OJAS_SAM_WEIGHTS"]
fn conv_transpose2d_matches_torch() {
    let (fx, w, be) = (fixture(), weights(), cpu());
    show(&conv_transpose2d_rows(&be, &fx, &w).unwrap());
}

#[test]
#[ignore = "needs OJAS_SAM_GROUPNORM_FIXTURES (SpikeTrack groupnorm fixture)"]
fn group_norm_matches_torch_at_both_pixel_decoder_sizes() {
    let fx = Fixture::from_env_var(ENV_GROUPNORM_FIXTURES)
        .unwrap_or_else(|e| panic!("{ENV_GROUPNORM_FIXTURES}: {e}"))
        .unwrap_or_else(|| panic!("{ENV_GROUPNORM_FIXTURES} is not set: this test cannot run"));
    // The values the fixture's header must record, asserted rather than
    // taken from it silently: `group_norm` refuses a header that differs.
    let (groups, eps) = ojas_oracle::sam_fixtures::group_norm_header(&fx).unwrap();
    assert_eq!((groups, eps), (8, 1e-5));
    let be = cpu();
    let rows: Vec<OpReport> = ["gn0", "gn1"]
        .iter()
        .map(|prefix| {
            let started = std::time::Instant::now();
            let row = group_norm(&be, &fx, prefix, groups, eps).unwrap();
            println!("{prefix}: {:?}", started.elapsed());
            row
        })
        .collect();
    show(&rows);
}

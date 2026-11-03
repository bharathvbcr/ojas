//! The tiled forward's output bits, pinned. The bf16-accumulation forward
//! moved out of the f32 body to its own entry (`ojas_attn_fwd_bf16`), a change
//! that must not move a bit in either mode: SpikeTrack's parity gates compare
//! its outputs byte for byte, and so does any later retile of the f32 entries.
//! So each case here hashes O then LSE (FNV-1a 64 over the f32 bit patterns)
//! and holds the hash to the one the kernels before the split produced
//! (`286a444`, through the probe `ojas-metal/examples/flash_fwd_probe.rs`:
//! `bench/results/2026-11-03-flash-fwd-specialize/hash-A.txt`), on the same
//! inputs.
//!
//! The matrix units' rounding is the device's, so the hashes hold on the
//! device they were recorded on. Another device fails with its own hashes
//! printed, to be recorded from a build of `286a444` there, not accepted.
//!
//! Also checked: under bf16 accumulation the forward encodes the bf16 entry
//! and no f32 one, and the other way round without it.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use ojas_core::{AttentionOpts, Backend, Tensor};
use ojas_metal::MetalBackend;

/// The device the hashes were recorded on.
const RECORDED_ON: &str = "Apple M5 Pro";

/// `hash-A.txt`, case by case, in the probe's order.
const GOLDEN: [(&str, u64); 12] = [
    ("spiketrack_x36400_fwd", 0xfead_90be_d0b2_6b30),
    ("spiketrack_x36400_saving", 0x298e_da49_5830_9793),
    ("x_d32_q100_k333", 0x2eb2_7248_9f66_c109),
    ("x_d32_q5184_k15600", 0x8b29_e6e8_83f0_5dc8),
    ("x_d64_q65_k129_b2h3", 0x5653_7be8_99a6_ae80),
    ("x_d128_q40_k70", 0xc733_bb23_73f7_23c4),
    ("x_d32_q100_k333_bias", 0x5728_9d21_2706_2208),
    ("causal_d32_t200", 0x0d3e_2166_ed76_5932),
    ("causal_d64_t150_w37", 0x7b35_8372_d7e3_aeb3),
    ("bf16_x_d32_q100_k333", 0x51fa_8580_92bf_555a),
    ("bf16_x_d32_q100_k333_bias", 0xc670_0098_5942_4277),
    ("bf16_causal_d32_t200", 0xf975_f211_e1ce_5364),
];

const F32_D32: &str = "ojas_attn_fwd_d32";
const BF16: &str = "ojas_attn_fwd_bf16";

fn fnv(ts: &[&Tensor]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for t in ts {
        for x in down(t) {
            for byte in x.to_bits().to_le_bytes() {
                h ^= u64::from(byte);
                h = h.wrapping_mul(0x0000_0100_0000_01B3);
            }
        }
    }
    h
}

fn up_seeded(m: &MetalBackend, shape: &[usize], seed: u64, scale: f32) -> Tensor {
    up(m, &rand(shape, seed, scale))
}

/// Forced flash cross-attention, as the probe's `cross`.
fn cross(m: &MetalBackend, qs: [usize; 4], tk: usize, bias: bool) -> u64 {
    let [b, h, tq, d] = qs;
    let q = up_seeded(m, &qs, 11, 1.0);
    let k = up_seeded(m, &[b, h, tk, d], 12, 1.0);
    let v = up_seeded(m, &[b, h, tk, d], 13, 1.0);
    let bt = bias.then(|| up_seeded(m, &[b, h, tq, tk], 14, 0.5));
    let opts = AttentionOpts {
        causal: false,
        bias: bt.as_ref(),
        ..Default::default()
    };
    let (o, lse) = m
        .attention_forward_saving_flash(&q, &k, &v, opts)
        .expect("flash");
    fnv(&[&o, &lse])
}

/// The causal forward, as the probe's `causal`.
fn causal(m: &MetalBackend, s: [usize; 4], window: Option<usize>) -> u64 {
    let q = up_seeded(m, &s, 21, 1.0);
    let k = up_seeded(m, &s, 22, 1.0);
    let v = up_seeded(m, &s, 23, 1.0);
    let (o, lse) = m.causal_sdpa_forward(&q, &k, &v, window).expect("causal");
    fnv(&[&o, &lse])
}

/// Every case of the probe's `hash` mode, in its order, with its hash.
fn hashes(m: &MetalBackend) -> Vec<(&'static str, u64)> {
    let (b, h, tq, tk, d) = (1usize, 8usize, 5184usize, 36400usize, 32usize);
    let q = up_seeded(m, &[b, h, tq, d], 101, 1.0);
    let k = up_seeded(m, &[b, h, tk, d], 102, 1.0);
    let v = up_seeded(m, &[b, h, tk, d], 103, 1.0);
    let opts = || AttentionOpts {
        causal: false,
        ..Default::default()
    };
    let o = m.attention_forward(&q, &k, &v, opts()).expect("fwd");
    let fwd = fnv(&[&o]);
    let (o2, lse2) = m
        .attention_forward_saving(&q, &k, &v, opts())
        .expect("saving");
    let saving = fnv(&[&o2, &lse2]);
    drop((q, k, v, o, o2, lse2));

    let mut out = vec![
        ("spiketrack_x36400_fwd", fwd),
        ("spiketrack_x36400_saving", saving),
        ("x_d32_q100_k333", cross(m, [1, 2, 100, 32], 333, false)),
        (
            "x_d32_q5184_k15600",
            cross(m, [1, 8, 5184, 32], 15600, false),
        ),
        ("x_d64_q65_k129_b2h3", cross(m, [2, 3, 65, 64], 129, false)),
        ("x_d128_q40_k70", cross(m, [1, 2, 40, 128], 70, false)),
        ("x_d32_q100_k333_bias", cross(m, [1, 2, 100, 32], 333, true)),
        ("causal_d32_t200", causal(m, [1, 2, 200, 32], None)),
        ("causal_d64_t150_w37", causal(m, [1, 2, 150, 64], Some(37))),
    ];
    m.set_bf16_accumulation(true);
    out.push((
        "bf16_x_d32_q100_k333",
        cross(m, [1, 2, 100, 32], 333, false),
    ));
    out.push((
        "bf16_x_d32_q100_k333_bias",
        cross(m, [1, 2, 100, 32], 333, true),
    ));
    out.push(("bf16_causal_d32_t200", causal(m, [1, 2, 200, 32], None)));
    m.set_bf16_accumulation(false);
    out
}

#[test]
fn flash_forward_bits_match_the_kernels_before_the_retile() {
    let m = metal();
    assert!(
        m.has_tensorops(),
        "the f32 flash forward runs on the TensorOps matrix units; this device has none"
    );
    let got = hashes(&m);
    let printed: Vec<String> = got.iter().map(|(n, h)| format!("{n}\t{h:016x}")).collect();
    assert_eq!(
        m.device_name(),
        RECORDED_ON,
        "golden hashes were recorded on {RECORDED_ON}; record this device's from a build of 286a444. This build's:\n{}",
        printed.join("\n")
    );
    for ((name, h), (want_name, want)) in got.iter().zip(GOLDEN) {
        assert_eq!(name, want_name, "case order differs from GOLDEN");
        assert_eq!(
            *h, *want,
            "{name}: output bits moved ({h:016x}, recorded {want:016x})"
        );
    }
    assert_eq!(got.len(), GOLDEN.len());
}

#[test]
fn bf16_accumulation_runs_its_own_forward_entry() {
    let m = metal();
    let count = |m: &MetalBackend| (m.kernel_dispatches(F32_D32), m.kernel_dispatches(BF16));
    for bf16 in [false, true] {
        m.set_bf16_accumulation(bf16);
        let before = count(&m);
        let _ = causal(&m, [1, 2, 64, 32], None);
        let _ = cross(&m, [1, 2, 100, 32], 333, false);
        let after = count(&m);
        m.set_bf16_accumulation(false);
        let ran = (after.0 - before.0, after.1 - before.1);
        let want = if bf16 { (0, 2) } else { (2, 0) };
        assert_eq!(
            ran, want,
            "bf16 accumulation {bf16}: ({F32_D32}, {BF16}) dispatches"
        );
    }
}

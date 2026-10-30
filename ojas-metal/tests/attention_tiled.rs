//! The split-K tiled general-attention kernel (`ojas_xattn_part_dD` and
//! `ojas_xattn_combine_dD`, head dim 16 and 32, non-causal) against the CPU
//! reference: cross-attention (`Tq != Tk`), every bias broadcast layout, key
//! lengths including 0, grouped heads, tile and split edges (a split count
//! that does not divide `Tk`, `Tq = 1`), the log-sum-exp the autograd tape
//! reads, and the contracts around it (a non-finite operand faults under the
//! name that was called, a budget that holds only the outputs runs the scalar
//! kernel).
//!
//! The kernel is observed through `MetalBackend::dispatches`: the tiled path
//! is two dispatches (partials and combine) where the scalar kernel is one,
//! so a call must cost exactly one dispatch more than the same call made
//! `causal` (which only the scalar kernel runs). `attention_forward_saving`
//! never routes to tessl's `encoder_attn`, so it is the tiled kernel's entry
//! at shapes the pad route would otherwise take.
//!
//! Head dim 32 with no bias and no key lengths takes the TensorOps flash
//! kernel instead of these (`flash_cross_route`) once a call launches
//! `FLASH_MIN_GROUPS` threadgroups, 32 query rows of a plane each. The shapes
//! here are small: the flash-eligible ones (head dim 32, no bias or lengths) reach
//! 10 groups, and only head-dim-16 and bias cases go higher (up to 1024), which
//! never route to flash. They are split-K at the floor in
//! force when this was written; `flash_dispatched` derives the route from
//! `MetalBackend::flash_min_groups`, so a lower floor moves a case to the
//! flash kernel and the planned kernel and the dispatch count are asserted for
//! whichever kernel the floor gives it. The flash route is `attention_flash.rs`.

#![cfg(target_os = "macos")]

mod common;

use std::time::Instant;

use common::*;
use ojas_core::{AttentionOpts, Backend, Budget, Tensor};
use ojas_metal::MetalBackend;

fn tol(tk: usize) -> f32 {
    4e-5 * (tk as f32).sqrt().max(1.0)
}

#[derive(Clone, Debug)]
struct Case {
    b: usize,
    h: usize,
    hkv: usize,
    tq: usize,
    tk: usize,
    d: usize,
    bias: Option<[usize; 3]>,
    lens: Option<Vec<u32>>,
    /// A query row whose bias is `MASKED` on every key, in every bias plane.
    masked_row: Option<usize>,
}

/// A finite bias that buries a score: the row it covers is not "sees no key"
/// (that is a key length of 0), it is a softmax over scores that all round to
/// the same value.
const MASKED: f32 = -1e30;

impl Case {
    fn new(b: usize, h: usize, hkv: usize, tq: usize, tk: usize, d: usize) -> Self {
        Case {
            b,
            h,
            hkv,
            tq,
            tk,
            d,
            bias: None,
            lens: None,
            masked_row: None,
        }
    }
    fn masked_row(mut self, row: usize) -> Self {
        self.masked_row = Some(row);
        self
    }
    fn bias(mut self, s: [usize; 3]) -> Self {
        self.bias = Some(s);
        self
    }
    fn lens(mut self, l: &[u32]) -> Self {
        self.lens = Some(l.to_vec());
        self
    }
}

struct Operands {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    bias: Option<Tensor>,
    lens: Option<Tensor>,
}

impl Operands {
    fn host(c: &Case, seed: u64) -> Self {
        Operands {
            q: rand(&[c.b, c.h, c.tq, c.d], seed, 1.0),
            k: rand(&[c.b, c.hkv, c.tk, c.d], seed + 1, 1.0),
            v: rand(&[c.b, c.hkv, c.tk, c.d], seed + 2, 1.0),
            bias: c.bias.map(|[b, h, t]| {
                let shape = [b, h, t, c.tk];
                let mut data = values(shape.iter().product(), seed + 3, 2.0);
                if let Some(row) = c.masked_row {
                    for plane in 0..b * h {
                        let at = (plane * t + row) * c.tk;
                        data[at..at + c.tk].fill(MASKED);
                    }
                }
                host(&data, &shape)
            }),
            lens: c.lens.as_ref().map(|l| host_u32(l, &[l.len()])),
        }
    }
    fn up(&self, m: &MetalBackend) -> Operands {
        Operands {
            q: up(m, &self.q),
            k: up(m, &self.k),
            v: up(m, &self.v),
            bias: self.bias.as_ref().map(|t| up(m, t)),
            lens: self.lens.as_ref().map(|t| up(m, t)),
        }
    }
    fn opts(&self, causal: bool) -> AttentionOpts<'_> {
        AttentionOpts {
            causal,
            window: None,
            bias: self.bias.as_ref(),
            key_lens: self.lens.as_ref(),
        }
    }
}

/// Whether the dispatcher runs the flash kernel for `c`: head dim 32, no bias
/// or key lengths, on a TensorOps device, with at least `flash_min_groups`
/// threadgroups of 32 query rows.
fn flash_dispatched(m: &MetalBackend, c: &Case) -> bool {
    let groups = (c.tq.div_ceil(32) * c.b * c.h) as u64;
    m.has_tensorops()
        && c.d == 32
        && c.bias.is_none()
        && c.lens.is_none()
        && groups >= MetalBackend::flash_min_groups()
}

/// Dispatches one `attention_forward_saving` call encodes, and how many of
/// them were the flash kernel and the scalar kernel: `(dispatches, flash,
/// scalar)`. The flash and scalar kernels are both one dispatch, so the count
/// alone cannot tell a scalar misroute from the flash route at a lowered
/// floor; the kernel names can.
fn observed_of(m: &MetalBackend, o: &Operands, causal: bool) -> (u64, u64, u64) {
    ok("sync", m.sync());
    let before = m.dispatches();
    let kernels = (
        m.kernel_dispatches("ojas_attn_fwd_d32"),
        m.kernel_dispatches("ojas_attn_gen_fwd"),
    );
    let out = ok(
        "saving",
        m.attention_forward_saving(&o.q, &o.k, &o.v, o.opts(causal)),
    );
    ok("sync", m.sync());
    drop(out);
    (
        m.dispatches() - before,
        m.kernel_dispatches("ojas_attn_fwd_d32") - kernels.0,
        m.kernel_dispatches("ojas_attn_gen_fwd") - kernels.1,
    )
}

/// Dispatches one `attention_forward_saving` call encodes.
fn dispatches_of(m: &MetalBackend, o: &Operands, causal: bool) -> u64 {
    observed_of(m, o, causal).0
}

/// The tiled kernel agrees with the CPU on the output and the log-sum-exp,
/// ran (one dispatch more than the scalar kernel's and planned as split-K, or
/// as many and planned as flash where `flash_dispatched` says the floor gives
/// the case to the flash kernel), and repeats bit for bit.
fn check(m: &MetalBackend, c: &Case, seed: u64) {
    let cpu = cpu();
    let host = Operands::host(c, seed);
    {
        let probe = host.up(m);
        assert_eq!(
            ok(
                "planned",
                m.attention_saving_kernel(&probe.q, &probe.k, &probe.v, probe.opts(false))
            ),
            if flash_dispatched(m, c) {
                "flash"
            } else {
                "splitk"
            },
            "{c:?}: planned kernel"
        );
    }
    let dev = host.up(m);
    let tag = format!("{c:?}");
    let (want, want_lse) = ok(
        &tag,
        cpu.attention_forward_saving(&host.q, &host.k, &host.v, host.opts(false)),
    );
    let (got, got_lse) = ok(
        &tag,
        m.attention_forward_saving(&dev.q, &dev.k, &dev.v, dev.opts(false)),
    );
    ok("sync", m.sync());
    same_tensor(&format!("{tag} output"), &got, &want, tol(c.tk), 1e-4);
    same_tensor(&format!("{tag} lse"), &got_lse, &want_lse, tol(c.tk), 1e-4);
    let (again, again_lse) = ok(
        &tag,
        m.attention_forward_saving(&dev.q, &dev.k, &dev.v, dev.opts(false)),
    );
    ok("sync", m.sync());
    assert_eq!(down(&again), down(&got), "{tag}: not repeatable");
    assert_eq!(
        down(&again_lse),
        down(&got_lse),
        "{tag}: lse not repeatable"
    );
    // Fresh uploads for each call, so no operand scan is skipped in one.
    let (tiled, tiled_flash, tiled_scalar) = observed_of(m, &host.up(m), false);
    let (scalar, causal_flash, causal_scalar) = observed_of(m, &host.up(m), true);
    assert_eq!(
        (tiled_flash, tiled_scalar),
        (u64::from(flash_dispatched(m, c)), 0),
        "{tag}: kernels of the non-causal call (flash, scalar)"
    );
    assert_eq!(
        (causal_flash, causal_scalar),
        (0, 1),
        "{tag}: kernels of the causal call (flash, scalar)"
    );
    assert_eq!(
        tiled,
        if flash_dispatched(m, c) {
            scalar
        } else {
            scalar + 1
        },
        "{tag}: the tiled kernel is two dispatches, the scalar and the flash kernel one"
    );
}

#[test]
fn cross_attention_matches_cpu_at_head_dim_32_and_16() {
    let m = metal();
    for (i, c) in [
        Case::new(2, 3, 3, 17, 40, 32),
        Case::new(1, 2, 2, 9, 33, 16),
        Case::new(2, 4, 2, 9, 33, 32),
        Case::new(2, 4, 2, 9, 33, 16),
        Case::new(1, 2, 2, 100, 100, 32),
    ]
    .iter()
    .enumerate()
    {
        check(&m, c, 10 + 5 * i as u64);
    }
}

/// `Tq > Tk` is allowed for the non-causal form.
#[test]
fn a_query_longer_than_the_key_set_matches_cpu() {
    let (m, c) = (metal(), cpu());
    let case = Case::new(1, 2, 2, 50, 7, 32);
    let host = Operands::host(&case, 9);
    let dev = host.up(&m);
    let (want, want_lse) = ok(
        "cpu",
        c.attention_forward_saving(&host.q, &host.k, &host.v, host.opts(false)),
    );
    let (got, got_lse) = ok(
        "metal",
        m.attention_forward_saving(&dev.q, &dev.k, &dev.v, dev.opts(false)),
    );
    ok("sync", m.sync());
    same_tensor("output", &got, &want, tol(7), 1e-4);
    same_tensor("lse", &got_lse, &want_lse, tol(7), 1e-4);
}

#[test]
fn the_detector_bias_case_matches_cpu() {
    let m = metal();
    // [1, 8, 201, 32] over [1, 8, 5184, 32] with a [1, 8, 201, 5184] bias.
    check(&m, &Case::new(1, 8, 8, 201, 5184, 32).bias([1, 8, 201]), 1);
}

#[test]
fn a_short_query_over_a_long_key_set_matches_cpu() {
    let m = metal();
    // The mask decoder's shape: 16 tokens over 4096 image keys, head dim 16.
    check(&m, &Case::new(1, 8, 8, 16, 4096, 16), 2);
    check(&m, &Case::new(1, 8, 8, 4096, 16, 16).bias([1, 8, 4096]), 3);
    // Tq = 1.
    check(&m, &Case::new(1, 2, 2, 1, 700, 32), 4);
    check(&m, &Case::new(2, 2, 2, 1, 1, 32), 5);
}

#[test]
fn tile_and_split_edges_match_cpu() {
    let m = metal();
    // Tq around the 128-row threadgroup, Tk around the 64-key tile, and
    // key counts that no split count divides.
    for (i, (tq, tk)) in [
        (127usize, 63usize),
        (128, 64),
        (129, 65),
        (130, 1000),
        (3, 1),
        (4, 20000),
        (5, 129),
    ]
    .into_iter()
    .enumerate()
    {
        for d in [16usize, 32] {
            check(
                &m,
                &Case::new(1, 2, 2, tq, tk, d),
                100 + 7 * i as u64 + d as u64,
            );
        }
    }
}

#[test]
fn every_bias_broadcast_layout_matches_cpu() {
    let m = metal();
    let (b, h, tq, tk) = (3usize, 4usize, 50usize, 200usize);
    for (i, layout) in [
        [1usize, 1, 1],
        [b, 1, 1],
        [1, h, 1],
        [1, 1, tq],
        [b, h, 1],
        [1, h, tq],
        [b, 1, tq],
        [b, h, tq],
    ]
    .into_iter()
    .enumerate()
    {
        check(
            &m,
            &Case::new(b, h, h, tq, tk, 32).bias(layout),
            200 + 3 * i as u64,
        );
    }
}

#[test]
fn key_lengths_including_zero_match_cpu() {
    let m = metal();
    check(&m, &Case::new(3, 2, 2, 20, 300, 32).lens(&[300, 5, 0]), 30);
    check(&m, &Case::new(3, 2, 2, 20, 300, 16).lens(&[1, 64, 65]), 31);
    check(
        &m,
        &Case::new(2, 4, 2, 9, 130, 32)
            .lens(&[130, 70])
            .bias([2, 1, 9]),
        32,
    );
    // A batch whose length is 0 writes zeros and an lse of 0, as the CPU does.
    let c = Case::new(2, 2, 2, 6, 100, 32).lens(&[0, 100]);
    let host = Operands::host(&c, 33);
    let dev = host.up(&m);
    let (out, lse) = ok(
        "saving",
        m.attention_forward_saving(&dev.q, &dev.k, &dev.v, dev.opts(false)),
    );
    ok("sync", m.sync());
    let (out, lse) = (down(&out), down(&lse));
    let per_batch = 2 * 6 * 32;
    assert!(out[..per_batch].iter().all(|&x| x == 0.0), "zeros");
    assert!(lse[..2 * 6].iter().all(|&x| x == 0.0), "lse 0");
}

/// A key length past `Tk` reads every key, as the CPU and the D64 scalar test
/// (`attention_general.rs`, `key_lengths_match_cpu_including_empty_and_oversized`)
/// do: the kernel clamps it with `min(Tk, len)`. One key tile and several
/// splits, so the clamp is held at the split bounds too.
#[test]
fn a_key_length_beyond_tk_reads_every_key_as_the_cpu_does() {
    let m = metal();
    for (i, d) in [32usize, 16].into_iter().enumerate() {
        check(
            &m,
            &Case::new(2, 2, 2, 6, 40, d).lens(&[99, 7]),
            70 + 2 * i as u64,
        );
        check(
            &m,
            &Case::new(2, 2, 2, 6, 300, d).lens(&[999, 65]),
            71 + 2 * i as u64,
        );
    }
}

/// A query row biased by -1e30 on every key, the other rows normal: the
/// scores of that row all round to one value, so it averages `V` and its
/// log-sum-exp is about -1e30, as on the CPU.
#[test]
fn a_query_row_biased_to_minus_1e30_on_every_key_matches_cpu() {
    let m = metal();
    for (i, d) in [32usize, 16].into_iter().enumerate() {
        check(
            &m,
            &Case::new(2, 3, 3, 5, 200, d).bias([2, 3, 5]).masked_row(2),
            90 + i as u64,
        );
    }
}

#[test]
fn a_non_finite_operand_faults_under_the_name_that_was_called() {
    let m = metal();
    let c = Case::new(1, 2, 2, 5, 130, 32).bias([1, 2, 5]);
    for which in ["q", "k", "v", "bias"] {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut ops = Operands::host(&c, 40);
            let target = match which {
                "q" => &mut ops.q,
                "k" => &mut ops.k,
                "v" => &mut ops.v,
                _ => ops.bias.as_mut().expect("bias"),
            };
            let mut data = ok("host", target.to_f32_vec());
            data[7] = bad;
            *target = host(&data, target.shape());
            let dev = ops.up(&m);
            let r = m.attention_forward(&dev.q, &dev.k, &dev.v, dev.opts(false));
            deferred(&m, &format!("{bad} in {which}"), r, "attention_forward");
            let r = m.attention_forward_saving(&dev.q, &dev.k, &dev.v, dev.opts(false));
            deferred(
                &m,
                &format!("{bad} in {which} (saving)"),
                r,
                "attention_forward_saving",
            );
        }
    }
}

/// A budget with room for the inputs, the output and the log-sum-exp but not
/// for the split partials runs the scalar kernel (one dispatch, as a causal
/// call costs), matches the CPU and charges nothing it keeps.
#[test]
fn a_budget_without_room_for_the_partials_runs_the_scalar_kernel() {
    let c = Case::new(1, 2, 2, 40, 3000, 32).bias([1, 2, 40]);
    let host = Operands::host(&c, 50);
    let (want, want_lse) = ok(
        "cpu",
        cpu().attention_forward_saving(&host.q, &host.k, &host.v, host.opts(false)),
    );
    let planes = c.b * c.h;
    let q_n = planes * c.tq * c.d;
    let kv_n = planes * c.tk * c.d;
    let bias_n = c.h * c.tq * c.tk;
    // Inputs, output and lse, and 1 KiB over; the partials need
    // splits * planes * Tq * (D + 2) * 4 bytes, far more.
    let room = ((q_n + 2 * kv_n + bias_n + q_n + planes * c.tq) * 4 + 1024) as u64;
    let tight = MetalBackend::new(Budget::new(room))
        .unwrap_or_else(|e| panic!("Metal device unavailable: {e:?}"));
    let dev = host.up(&tight);
    let baseline = ok("live", tight.budget().live_bytes());
    let fell_back = dispatches_of(&tight, &dev, false);
    let scalar = dispatches_of(&tight, &dev, true);
    assert_eq!(fell_back, scalar, "the scalar kernel ran");
    let (got, got_lse) = ok(
        "tight",
        tight.attention_forward_saving(&dev.q, &dev.k, &dev.v, dev.opts(false)),
    );
    ok("sync", tight.sync());
    same_tensor("tight output", &got, &want, tol(c.tk), 1e-4);
    same_tensor("tight lse", &got_lse, &want_lse, tol(c.tk), 1e-4);
    drop((got, got_lse));
    assert_eq!(ok("live", tight.budget().live_bytes()), baseline);
}

/// Causal and head dims other than 16 and 32 stay on the scalar kernel: the
/// same call costs the scalar kernel's one dispatch.
#[test]
fn causal_and_other_head_dims_stay_on_the_scalar_kernel() {
    let m = metal();
    let base_case = Case::new(1, 2, 2, 40, 40, 32);
    let base = Operands::host(&base_case, 60).up(&m);
    let (causal, causal_flash, causal_scalar) = observed_of(&m, &base, true);
    assert_eq!((causal_flash, causal_scalar), (0, 1), "causal is scalar");
    let other = Operands::host(&Case::new(1, 2, 2, 40, 90, 64), 61).up(&m);
    let d64 = dispatches_of(&m, &other, false);
    let (tiled, tiled_flash, tiled_scalar) = observed_of(&m, &base, false);
    assert_eq!(
        (tiled_flash, tiled_scalar),
        (u64::from(flash_dispatched(&m, &base_case)), 0),
        "head dim 32 non-causal: flash or split-K, never the scalar kernel"
    );
    assert_eq!(
        tiled,
        if flash_dispatched(&m, &base_case) {
            causal
        } else {
            causal + 1
        },
        "head dim 32 non-causal is tiled (one dispatch where the floor gives it to flash)"
    );
    assert_eq!(
        d64, causal,
        "head dim 64 with Tq != Tk is the scalar kernel"
    );
}

/// The detector's bias case and the memory encoder's cross-attention, the
/// split-K tiled kernel (`attention_forward_saving_splitk`; the memory rows'
/// dispatcher call is the flash kernel at these 1296 threadgroups, timed by
/// `attention_flash_bench`) against the pad route (`attention_forward`, which
/// routes at these shapes): minimum of 5 after 2 warm-up calls. Run with
/// `--ignored --nocapture`.
#[test]
#[ignore = "timing, not a correctness check"]
fn tiled_kernel_wall_clock() {
    let m = metal();
    let time = |f: &dyn Fn()| {
        let mut best = f64::MAX;
        for i in 0..7 {
            ok("sync", m.sync());
            let start = Instant::now();
            f();
            ok("sync", m.sync());
            if i >= 2 {
                best = best.min(start.elapsed().as_secs_f64() * 1e3);
            }
        }
        best
    };
    // Detector: [1,8,201,32] x [1,8,5184,32] with a bias.
    let det = Operands::host(&Case::new(1, 8, 8, 201, 5184, 32).bias([1, 8, 201]), 1).up(&m);
    let t = time(&|| {
        drop(ok(
            "det",
            m.attention_forward_saving(&det.q, &det.k, &det.v, det.opts(false)),
        ));
    });
    eprintln!("TILED detector bias [1,8,201,32]x[1,8,5184,32]: {t:.2} ms");
    // Mask decoder row (d): [1,8,16,16] x [1,8,4096,16].
    let dec = Operands::host(&Case::new(1, 8, 8, 16, 4096, 16), 2).up(&m);
    let t = time(&|| {
        drop(ok(
            "dec",
            m.attention_forward_saving(&dec.q, &dec.k, &dec.v, dec.opts(false)),
        ));
    });
    eprintln!("TILED decoder [1,8,16,16]x[1,8,4096,16]: {t:.2} ms");
    // Memory cross-attention: q [1,8,5184,32] over m keys.
    for tk in [5200usize, 10400, 15616, 36544] {
        let mem = Operands::host(&Case::new(1, 8, 8, 5184, tk, 32), 3).up(&m);
        let tiled = time(&|| {
            drop(ok(
                "tiled",
                m.attention_forward_saving_splitk(&mem.q, &mem.k, &mem.v, mem.opts(false)),
            ));
        });
        let padded = time(&|| {
            drop(ok(
                "padded",
                m.attention_forward(&mem.q, &mem.k, &mem.v, mem.opts(false)),
            ));
        });
        // `attention_forward` takes the pad route up to Tk = 2.25 Tq (11664
        // keys here); beyond it, at head dim 32 with these 1296 threadgroups,
        // it runs the flash kernel, which is what that column times.
        let ran = if tk as f64 <= 2.25 * 5184.0 {
            "pad route"
        } else {
            "flash kernel (beyond the pad ratio)"
        };
        eprintln!(
            "TILED memory m={tk}: split-K {tiled:.1} ms, attention_forward {padded:.1} ms ({ran})"
        );
    }
}

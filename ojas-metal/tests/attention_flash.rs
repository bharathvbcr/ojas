//! The TensorOps flash kernel on cross-attention: `ojas_attn_fwd_d32` run with
//! `cross` (`Tq != Tk`, every key visible, no window; the generalisation of the
//! causal forward `attn_fwd_tiled_body`), taken by `attention_forward_saving`
//! and `attention_forward` for head dim 32 with no bias and no key lengths once
//! a call launches `FLASH_MIN_GROUPS` threadgroups of 32 query rows; a shorter
//! Q stays on split-K, whose threadgroups cut the keys instead.
//!
//! Every output is held to the CPU reference and to both device references the
//! dispatcher used to pick between (`attention_forward_saving_splitk` and
//! `attention_forward_saving_scalar`) on the same inputs, at stage 1's
//! tolerance: the ragged last 64-key block (`Tk` = 1, a block plus or minus
//! one), `Tq` above and below `Tk`, `Tq` = 1, grouped heads, batches, scores
//! whose maximum moves between key blocks (a dropped rescale shows there), a
//! budget with room for the outputs only, and the contracts around the route.
//!
//! What ran is read from `MetalBackend::kernel_dispatches`, the dispatches
//! the device thread encoded by pipeline name, not from the planner
//! (`attention_saving_kernel`), which a test only holds to agree. The edge
//! cases run the flash kernel forced (`attention_forward_saving_flash`), since
//! most of them are too short for the dispatcher to route; the routing test
//! pins both sides of the threadgroup floor. A device without TensorOps has no
//! flash kernel: there every call is asserted to run split-K, and every CPU
//! comparison still runs.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use ojas_core::{AttentionOpts, Backend, Budget, Tensor};
use ojas_metal::MetalBackend;

const FLASH: &str = "ojas_attn_fwd_d32";
const SCALAR: &str = "ojas_attn_gen_fwd";
/// Split-K's pipelines, one per head dim it covers.
const PART: [&str; 2] = ["ojas_xattn_part_d32", "ojas_xattn_part_d16"];
const COMBINE: [&str; 2] = ["ojas_xattn_combine_d32", "ojas_xattn_combine_d16"];

/// The attention kernels one call encoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Ran {
    flash: u64,
    scalar: u64,
    part: u64,
    combine: u64,
}

impl Ran {
    const FLASH: Ran = Ran {
        flash: 1,
        scalar: 0,
        part: 0,
        combine: 0,
    };
    const SCALAR: Ran = Ran {
        flash: 0,
        scalar: 1,
        part: 0,
        combine: 0,
    };
    const SPLITK: Ran = Ran {
        flash: 0,
        scalar: 0,
        part: 1,
        combine: 1,
    };
}

fn snapshot(m: &MetalBackend) -> Ran {
    Ran {
        flash: m.kernel_dispatches(FLASH),
        scalar: m.kernel_dispatches(SCALAR),
        part: PART.iter().map(|n| m.kernel_dispatches(n)).sum(),
        combine: COMBINE.iter().map(|n| m.kernel_dispatches(n)).sum(),
    }
}

fn since(m: &MetalBackend, before: Ran) -> Ran {
    let now = snapshot(m);
    Ran {
        flash: now.flash - before.flash,
        scalar: now.scalar - before.scalar,
        part: now.part - before.part,
        combine: now.combine - before.combine,
    }
}

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
}

impl Case {
    fn new(b: usize, h: usize, hkv: usize, tq: usize, tk: usize) -> Self {
        Case { b, h, hkv, tq, tk }
    }
    /// Threadgroups the flash kernel launches: 32 query rows of each plane.
    fn groups(&self) -> u64 {
        (self.tq.div_ceil(32) * self.b * self.h) as u64
    }
}

const D: usize = 32;

struct Operands {
    q: Tensor,
    k: Tensor,
    v: Tensor,
}

impl Operands {
    fn host(c: &Case, seed: u64) -> Self {
        Operands {
            q: rand(&[c.b, c.h, c.tq, D], seed, 1.0),
            k: rand(&[c.b, c.hkv, c.tk, D], seed + 1, 1.0),
            v: rand(&[c.b, c.hkv, c.tk, D], seed + 2, 1.0),
        }
    }
    fn with(c: &Case, q: Vec<f32>, k: Vec<f32>, v: Vec<f32>) -> Self {
        Operands {
            q: host(&q, &[c.b, c.h, c.tq, D]),
            k: host(&k, &[c.b, c.hkv, c.tk, D]),
            v: host(&v, &[c.b, c.hkv, c.tk, D]),
        }
    }
    fn up(&self, m: &MetalBackend) -> Operands {
        Operands {
            q: up(m, &self.q),
            k: up(m, &self.k),
            v: up(m, &self.v),
        }
    }
}

fn opts() -> AttentionOpts<'static> {
    AttentionOpts::default()
}

#[derive(Clone, Copy, Debug)]
enum Which {
    /// `attention_forward_saving`: the dispatcher.
    Auto,
    /// `attention_forward_saving_flash`: the flash kernel wherever it covers.
    Flash,
    SplitK,
    Scalar,
}

fn call(m: &MetalBackend, o: &Operands, which: Which) -> ((Tensor, Tensor), Ran) {
    ok("sync", m.sync());
    let before = snapshot(m);
    let out = match which {
        Which::Auto => m.attention_forward_saving(&o.q, &o.k, &o.v, opts()),
        Which::Flash => m.attention_forward_saving_flash(&o.q, &o.k, &o.v, opts()),
        Which::SplitK => m.attention_forward_saving_splitk(&o.q, &o.k, &o.v, opts()),
        Which::Scalar => m.attention_forward_saving_scalar(&o.q, &o.k, &o.v, opts()),
    };
    let out = ok(&format!("{which:?}"), out);
    ok("sync", m.sync());
    (out, since(m, before))
}

/// Dispatches one call encodes on fresh uploads.
fn dispatches(m: &MetalBackend, host: &Operands, which: Which) -> u64 {
    let o = host.up(m);
    ok("sync", m.sync());
    let before = m.dispatches();
    let out = call(m, &o, which);
    drop(out);
    m.dispatches() - before
}

/// What the dispatcher runs for `c`: the flash kernel on a TensorOps device
/// when the call launches at least `flash_min_groups` threadgroups, else
/// split-K.
fn dispatcher_flash(m: &MetalBackend, c: &Case) -> bool {
    m.has_tensorops() && c.groups() >= MetalBackend::flash_min_groups()
}

fn expected(flash: bool) -> Ran {
    if flash {
        Ran::FLASH
    } else {
        Ran::SPLITK
    }
}

/// The dispatcher's, the forced flash, split-K's and the scalar kernel's
/// outputs agree with the CPU's and with each other (output and log-sum-exp),
/// each ran the kernels it should (observed), the flash call repeats bit for
/// bit and, when `count`, costs the dispatches it should.
fn check(m: &MetalBackend, c: &Case, host: &Operands, count: bool) {
    let cpu = cpu();
    let dev = host.up(m);
    let tag = format!("{c:?}");
    let tensorops = m.has_tensorops();
    let planned = ok(
        &tag,
        m.attention_saving_kernel(&dev.q, &dev.k, &dev.v, opts()),
    );
    let flash_dispatched = dispatcher_flash(m, c);
    assert_eq!(
        planned,
        if flash_dispatched { "flash" } else { "splitk" },
        "{tag}: planned kernel"
    );
    let (want, want_lse) = ok(
        &tag,
        cpu.attention_forward_saving(&host.q, &host.k, &host.v, opts()),
    );
    let ((auto, auto_lse), ran_auto) = call(m, &dev, Which::Auto);
    let ((got, got_lse), ran_flash) = call(m, &dev, Which::Flash);
    let ((sk, sk_lse), ran_sk) = call(m, &dev, Which::SplitK);
    let ((sc, sc_lse), ran_sc) = call(m, &dev, Which::Scalar);
    assert_eq!(
        ran_auto,
        expected(flash_dispatched),
        "{tag}: the dispatcher's kernel"
    );
    assert_eq!(
        ran_flash,
        expected(tensorops),
        "{tag}: the forced flash call (split-K without TensorOps)"
    );
    assert_eq!(ran_sk, Ran::SPLITK, "{tag}: split-K");
    assert_eq!(ran_sc, Ran::SCALAR, "{tag}: scalar");
    same_tensor(&format!("{tag} output"), &got, &want, tol(c.tk), 1e-4);
    same_tensor(&format!("{tag} lse"), &got_lse, &want_lse, tol(c.tk), 1e-4);
    for (name, o, l) in [
        ("dispatcher", &auto, &auto_lse),
        ("split-K", &sk, &sk_lse),
        ("scalar", &sc, &sc_lse),
    ] {
        close(
            &format!("{tag} output vs {name}"),
            &down(&got),
            &down(o),
            tol(c.tk),
            1e-4,
        );
        close(
            &format!("{tag} lse vs {name}"),
            &down(&got_lse),
            &down(l),
            tol(c.tk),
            1e-4,
        );
    }
    let ((again, again_lse), _) = call(m, &dev, Which::Flash);
    assert_eq!(down(&again), down(&got), "{tag}: not repeatable");
    assert_eq!(
        down(&again_lse),
        down(&got_lse),
        "{tag}: lse not repeatable"
    );
    if count {
        let scalar = dispatches(m, host, Which::Scalar);
        assert_eq!(
            dispatches(m, host, Which::Flash),
            if tensorops { scalar } else { scalar + 1 },
            "{tag}: the flash kernel is one kernel, as the scalar call"
        );
        assert_eq!(
            dispatches(m, host, Which::SplitK),
            scalar + 1,
            "{tag}: split-K is partials and combine"
        );
    }
}

#[test]
fn query_and_key_counts_around_the_key_block_and_the_query_block_match() {
    let m = metal();
    // Key blocks are 64 keys, query blocks 32 rows: ragged last blocks on
    // both, a single key, a single query, and each side longer than the other.
    let tqs = [1usize, 7, 31, 32, 33, 100, 129];
    let tks = [1usize, 31, 63, 64, 65, 127, 129, 200];
    let mut seed = 100;
    for &tq in &tqs {
        for &tk in &tks {
            seed += 7;
            let c = Case::new(1, 2, 1, tq, tk);
            check(&m, &c, &Operands::host(&c, seed), tq == 33 && tk == 65);
        }
    }
}

#[test]
fn the_sam_shapes_and_batches_and_grouped_heads_match() {
    let m = metal();
    for (i, c) in [
        Case::new(1, 8, 8, 5184, 5200),
        Case::new(1, 8, 8, 5184, 10400),
        Case::new(1, 8, 8, 201, 5184),
        Case::new(2, 4, 4, 33, 130),
        Case::new(2, 4, 2, 33, 130),
        Case::new(1, 6, 2, 40, 77),
        Case::new(3, 2, 2, 1, 1000),
        Case::new(1, 2, 2, 150, 40),
    ]
    .iter()
    .enumerate()
    {
        check(&m, c, &Operands::host(c, 300 + 5 * i as u64), i % 2 == 0);
    }
}

/// Scores whose maximum moves between key blocks: keys grow along the key
/// axis (the last block holds the maximum), shrink along it (the first
/// does), and a query with large logits throughout. A rescale of the running
/// output that is dropped or applied to the wrong row shows as a large error.
#[test]
fn a_row_maximum_that_moves_between_key_blocks_is_rescaled() {
    let m = metal();
    let c = Case::new(1, 2, 2, 33, 300);
    let base = Operands::host(&c, 700);
    let q = ok("q", base.q.to_f32_vec());
    let k = ok("k", base.k.to_f32_vec());
    let v = ok("v", base.v.to_f32_vec());
    let per_key = |scale: &dyn Fn(usize) -> f32| -> Vec<f32> {
        k.iter()
            .enumerate()
            .map(|(i, x)| x * scale((i / D) % c.tk))
            .collect()
    };
    let loud_q: Vec<f32> = q.iter().map(|x| x * 6.0).collect();
    for (what, qs, ks) in [
        ("growing", q.clone(), per_key(&|j| 1.0 + j as f32 / 20.0)),
        (
            "shrinking",
            q.clone(),
            per_key(&|j| 1.0 + (c.tk - j) as f32 / 20.0),
        ),
        (
            "large logits",
            loud_q.clone(),
            per_key(&|j| 1.0 + (j % 7) as f32),
        ),
        (
            "alternating blocks",
            loud_q,
            per_key(&|j| if (j / 64) % 2 == 0 { 0.2 } else { 3.0 }),
        ),
    ] {
        let case = Operands::with(&c, qs, ks, v.clone());
        eprintln!("{what}");
        check(&m, &c, &case, false);
    }
}

/// At the SAM scale the output of random operands is about 1e-2 and the
/// stage-1 tolerance (3e-3 at 5200 keys) would hide a masking or rescale
/// error. Here V grows along the key axis (`v[j][c] = j / Tk + c / 100`), so
/// the output is the softmax-weighted mean position, about 0.5, and a key
/// dropped or masked wrongly in the ragged last block, or a block's weight
/// mis-scaled, moves it by far more than the 5e-5 allowed here; the lse is
/// held to the same bound.
#[test]
fn a_tight_tolerance_at_the_sam_scale_cannot_hide_a_mask_or_rescale_error() {
    let m = metal();
    let c = Case::new(1, 2, 2, 5184, 5200);
    let base = Operands::host(&c, 740);
    let v: Vec<f32> = (0..c.b * c.hkv * c.tk * D)
        .map(|i| {
            let (j, ch) = ((i / D) % c.tk, i % D);
            j as f32 / c.tk as f32 + ch as f32 / 100.0
        })
        .collect();
    let host_ops = Operands::with(
        &c,
        ok("q", base.q.to_f32_vec()),
        ok("k", base.k.to_f32_vec()),
        v,
    );
    let dev = host_ops.up(&m);
    let (want, want_lse) = ok(
        "cpu",
        cpu().attention_forward_saving(&host_ops.q, &host_ops.k, &host_ops.v, opts()),
    );
    let ((got, got_lse), ran) = call(&m, &dev, Which::Flash);
    assert_eq!(ran, expected(m.has_tensorops()), "the flash kernel ran");
    close(
        "output",
        &down(&got),
        &ok("want", want.to_f32_vec()),
        5e-5,
        5e-5,
    );
    close(
        "lse",
        &down(&got_lse),
        &ok("want lse", want_lse.to_f32_vec()),
        5e-5,
        5e-5,
    );
    // Not a degenerate comparison: the output is large.
    let peak = down(&got).iter().fold(0.0f32, |a, x| a.max(x.abs()));
    assert!(peak > 0.4, "the structured output is about 0.5, got {peak}");
}

/// The flash kernel needs nothing beyond its two outputs, so a budget with
/// room for the operands and the outputs only still runs it, one dispatch,
/// where split-K's partials would not fit and the scalar kernel would run.
#[test]
fn a_budget_that_holds_only_the_outputs_still_runs_the_flash_kernel() {
    let c = Case::new(1, 2, 2, 40, 3000);
    let host_ops = Operands::host(&c, 800);
    let (want, want_lse) = ok(
        "cpu",
        cpu().attention_forward_saving(&host_ops.q, &host_ops.k, &host_ops.v, opts()),
    );
    let planes = c.b * c.h;
    let q_n = planes * c.tq * D;
    let kv_n = planes * c.tk * D;
    // Operands, output and lse, and 1 KiB over.
    let room = ((q_n + 2 * kv_n + q_n + planes * c.tq) * 4 + 1024) as u64;
    let tight = MetalBackend::new(Budget::new(room))
        .unwrap_or_else(|e| panic!("Metal device unavailable: {e:?}"));
    let dev = host_ops.up(&tight);
    let baseline = ok("live", tight.budget().live_bytes());
    let ((got, got_lse), ran) = call(&tight, &dev, Which::Flash);
    if tight.has_tensorops() {
        assert_eq!(ran, Ran::FLASH, "flash needs no partials, no combine");
    } else {
        // Split-K's partials do not fit: the scalar kernel runs.
        assert_eq!(ran, Ran::SCALAR);
    }
    same_tensor("tight output", &got, &want, tol(c.tk), 1e-4);
    same_tensor("tight lse", &got_lse, &want_lse, tol(c.tk), 1e-4);
    drop((got, got_lse));
    assert_eq!(ok("live", tight.budget().live_bytes()), baseline);
    // The dispatcher sends this call to split-K unless the floor is 4 or
    // less: it is 4 flash threadgroups (two 32-row blocks of each of 2
    // planes; split-K's own launch is 2 groups of 128 rows). Split-K's
    // partials do not fit this budget, so the scalar kernel runs, as in stage
    // 1; the first call's outputs are freed, so the budget has room for these.
    let ((got, got_lse), ran_auto) = call(&tight, &dev, Which::Auto);
    assert_eq!(
        ran_auto,
        if dispatcher_flash(&tight, &c) {
            Ran::FLASH
        } else {
            Ran::SCALAR
        },
        "split-K's budget fallback"
    );
    same_tensor("fallback output", &got, &want, tol(c.tk), 1e-4);
    same_tensor("fallback lse", &got_lse, &want_lse, tol(c.tk), 1e-4);
    drop((got, got_lse));
    assert_eq!(ok("live", tight.budget().live_bytes()), baseline);
}

/// `attention_forward` (the inference form) takes the flash kernel too where
/// the pad route does not (`Tk` beyond 2.25 `Tq`), and agrees with the CPU: a
/// call of 3 operand scans, the kernel and 2 output checks, 6 dispatches, not
/// the pad route's 9 or split-K's 7.
#[test]
fn attention_forward_beyond_the_pad_ratio_runs_the_flash_kernel() {
    let m = metal();
    let c = Case::new(1, 8, 8, 5184, 15584);
    let host_ops = Operands::host(&c, 900);
    let dev = host_ops.up(&m);
    let want = ok(
        "cpu",
        cpu().attention_forward(&host_ops.q, &host_ops.k, &host_ops.v, opts()),
    );
    ok("sync", m.sync());
    let (before, kernels) = (m.dispatches(), snapshot(&m));
    let got = ok("metal", m.attention_forward(&dev.q, &dev.k, &dev.v, opts()));
    ok("sync", m.sync());
    let n = m.dispatches() - before;
    let ran = since(&m, kernels);
    if m.has_tensorops() {
        assert_eq!(ran, Ran::FLASH, "the flash kernel ran, not the pad route");
        assert_eq!(n, 6, "3 scans, the kernel, 2 output checks");
    } else {
        assert_eq!(ran, Ran::SPLITK);
        assert_eq!(n, 7, "split-K: partials and combine");
    }
    same_tensor("15584", &got, &want, tol(c.tk), 1e-4);
}

/// The threadgroup floor, both sides of whatever `flash_min_groups` is: a call
/// just under it stays on split-K, a call at it runs flash, observed by the
/// kernels encoded; the planner agrees; both match the CPU.
#[test]
fn the_dispatcher_routes_on_the_threadgroup_floor() {
    let m = metal();
    let floor = MetalBackend::flash_min_groups();
    // Enough planes that a call one query block shorter still has a row, for
    // any floor of 2 or more: 8 planes from a floor of 16, else 1.
    assert!(floor >= 2, "a floor of {floor} leaves no call below it");
    let planes = if floor >= 16 { 8usize } else { 1 };
    let over_rows = 32 * (floor as usize).div_ceil(planes);
    let under_rows = over_rows - 32;
    assert!(
        under_rows >= 1,
        "a floor of {floor} leaves no call below it"
    );
    for (what, tq, flash) in [
        ("under", under_rows, false),
        ("at or over", over_rows, true),
    ] {
        let c = Case::new(1, planes, planes, tq, 100);
        assert_eq!(
            c.groups() >= floor,
            flash,
            "{what}: the test's own arithmetic"
        );
        let host_ops = Operands::host(&c, 960 + tq as u64);
        let dev = host_ops.up(&m);
        let planned = ok(
            what,
            m.attention_saving_kernel(&dev.q, &dev.k, &dev.v, opts()),
        );
        let route_flash = flash && m.has_tensorops();
        assert_eq!(
            planned,
            if route_flash { "flash" } else { "splitk" },
            "{what}: planned"
        );
        let ((got, got_lse), ran) = call(&m, &dev, Which::Auto);
        assert_eq!(ran, expected(route_flash), "{what}: observed");
        let (want, want_lse) = ok(
            "cpu",
            cpu().attention_forward_saving(&host_ops.q, &host_ops.k, &host_ops.v, opts()),
        );
        same_tensor(&format!("{what} output"), &got, &want, tol(c.tk), 1e-4);
        same_tensor(&format!("{what} lse"), &got_lse, &want_lse, tol(c.tk), 1e-4);
    }
    // The extreme the review named: one query row over 5184 keys is 8
    // threadgroups, split-K's 384 are not matched, and it stays on split-K
    // (flash only for a floor of 8 or less).
    let c = Case::new(1, 8, 8, 1, 5184);
    let dev = Operands::host(&c, 970).up(&m);
    let one_row = dispatcher_flash(&m, &c);
    assert_eq!(
        ok(
            "one row",
            m.attention_saving_kernel(&dev.q, &dev.k, &dev.v, opts())
        ),
        if one_row { "flash" } else { "splitk" }
    );
    let (_, ran) = call(&m, &dev, Which::Auto);
    assert_eq!(ran, expected(one_row), "one query row: observed");
}

/// A NaN or infinity anywhere in an operand, including the ragged last key
/// and the last query row, is a deferred fault under the name that was called,
/// on the flash kernel and on the dispatcher alike.
#[test]
fn a_non_finite_operand_faults_under_the_name_that_was_called() {
    let m = metal();
    let c = Case::new(1, 2, 2, 33, 65);
    for (what, at) in [
        ("q last row", 0usize),
        ("k last key", 1),
        ("v last key", 2),
        ("v first", 3),
    ] {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let host_ops = Operands::host(&c, 950);
            let mut q = ok("q", host_ops.q.to_f32_vec());
            let mut k = ok("k", host_ops.k.to_f32_vec());
            let mut v = ok("v", host_ops.v.to_f32_vec());
            match at {
                0 => *q.last_mut().expect("q") = bad,
                1 => *k.last_mut().expect("k") = bad,
                2 => *v.last_mut().expect("v") = bad,
                _ => v[0] = bad,
            }
            let ops = Operands::with(&c, q, k, v).up(&m);
            let r = m.attention_forward_saving_flash(&ops.q, &ops.k, &ops.v, opts());
            let _kept = deferred(
                &m,
                &format!("flash {what} {bad}"),
                r,
                "attention_forward_saving",
            );
            let r = m.attention_forward_saving(&ops.q, &ops.k, &ops.v, opts());
            let _kept = deferred(
                &m,
                &format!("dispatcher {what} {bad}"),
                r,
                "attention_forward_saving",
            );
        }
    }
    // The inference form names itself.
    let ops = Operands::host(&c, 951);
    let mut v = ok("v", ops.v.to_f32_vec());
    v[7] = f32::NAN;
    let ops = Operands::with(
        &c,
        ok("q", ops.q.to_f32_vec()),
        ok("k", ops.k.to_f32_vec()),
        v,
    )
    .up(&m);
    let r = m.attention_forward(&ops.q, &ops.k, &ops.v, opts());
    let _kept = deferred(&m, "attention_forward", r, "attention_forward");
}

/// A call the flash route does not cover stays on its kernel: a bias, key
/// lengths and head dim 16 are split-K, causal and head dim 64 the scalar one,
/// whatever its threadgroup count (the planner and the kernels encoded agree).
/// The shapes are sized from the floor, so the covered call is at it.
#[test]
fn the_calls_the_route_does_not_cover_are_planned_and_run_elsewhere() {
    let m = metal();
    // 8 planes of enough 32-row query blocks to reach the floor: flash if the
    // call were covered.
    let rows = 32 * (MetalBackend::flash_min_groups() as usize).div_ceil(8);
    let big = Case::new(1, 8, 8, rows, 64);
    assert!(big.groups() >= MetalBackend::flash_min_groups());
    let host_ops = Operands::host(&big, 970);
    let dev = host_ops.up(&m);
    let bias = up(&m, &rand(&[1, 8, rows, 64], 971, 1.0));
    let lens = up(&m, &host_u32(&[50], &[1]));
    let kernel = |ops: &Operands, o: AttentionOpts<'_>| {
        ok(
            "kernel",
            m.attention_saving_kernel(&ops.q, &ops.k, &ops.v, o),
        )
    };
    let observed = |ops: &Operands, o: AttentionOpts<'_>| {
        ok("sync", m.sync());
        let before = snapshot(&m);
        let out = ok(
            "call",
            m.attention_forward_saving(&ops.q, &ops.k, &ops.v, o),
        );
        ok("sync", m.sync());
        drop(out);
        since(&m, before)
    };
    assert_eq!(
        kernel(&dev, opts()),
        if m.has_tensorops() { "flash" } else { "splitk" }
    );
    assert_eq!(
        observed(&dev, opts()),
        expected(m.has_tensorops()),
        "the covered call at the floor"
    );
    let with_bias = AttentionOpts {
        bias: Some(&bias),
        ..opts()
    };
    assert_eq!(kernel(&dev, with_bias), "splitk");
    assert_eq!(observed(&dev, with_bias), Ran::SPLITK);
    let with_lens = AttentionOpts {
        key_lens: Some(&lens),
        ..opts()
    };
    assert_eq!(kernel(&dev, with_lens), "splitk");
    assert_eq!(observed(&dev, with_lens), Ran::SPLITK);
    let causal = AttentionOpts {
        causal: true,
        ..opts()
    };
    assert_eq!(kernel(&dev, causal), "scalar");
    assert_eq!(observed(&dev, causal), Ran::SCALAR);
    let d16 = Operands {
        q: up(&m, &rand(&[1, 8, rows, 16], 972, 1.0)),
        k: up(&m, &rand(&[1, 8, 64, 16], 973, 1.0)),
        v: up(&m, &rand(&[1, 8, 64, 16], 974, 1.0)),
    };
    assert_eq!(kernel(&d16, opts()), "splitk");
    assert_eq!(observed(&d16, opts()), Ran::SPLITK, "head dim 16: observed");
    let d64 = Operands {
        q: up(&m, &rand(&[1, 8, rows, 64], 975, 1.0)),
        k: up(&m, &rand(&[1, 8, 90, 64], 976, 1.0)),
        v: up(&m, &rand(&[1, 8, 90, 64], 977, 1.0)),
    };
    assert_eq!(kernel(&d64, opts()), "scalar");
    assert_eq!(observed(&d64, opts()), Ran::SCALAR, "head dim 64: observed");
}

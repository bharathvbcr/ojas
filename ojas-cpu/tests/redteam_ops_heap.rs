//! Red-team gate: every non-linear CPU op allocates no more on the Rust heap
//! than it charges to its `Budget`, and its budget boundary is sharp.
//!
//! `redteam_linear_heap.rs` holds the linear ops to this; this file holds
//! every other op `CpuBackend` implements, including the in-place optimizer
//! steps and the gradient clip.
//!
//! A counting global allocator (this test binary only) records the heap
//! high-water mark while one op runs. The charged peak is the smallest room
//! the op accepts, found by binary search with a filler reservation that
//! leaves exactly that many bytes free. The gate is
//! `heap peak <= charged peak + SLACK`.
//!
//! `SLACK` is a fixed 64 KiB, not a fraction of the op: it covers pool batch
//! bookkeeping, result slots, shape and range vectors, `Arc` headers and
//! error strings, which do not grow with the tensors. Every shape below makes
//! the smallest output or temporary at least 512 KiB, so a buffer the budget
//! misses cannot hide in it.
//!
//! The gap this file was written against: each op's kernel dropped its
//! output hold when it returned, then `alloc_f32` (validate.rs) copied the
//! still-live output `Vec` into a new tensor, so the output was on the heap
//! twice while the budget held it once. The optimizer steps reserved one
//! parameter's length of headroom while AdamW built three new vectors and
//! Muon built two plus its Newton-Schulz temporaries.

mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use common::SplitMix64;
use ojas_core::{AdamWConfig, Backend, Budget, MuonNs5Config, Numerics, OjasError, Tensor};
use ojas_cpu::CpuBackend;

struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn grow(bytes: usize) {
    let now = LIVE.fetch_add(bytes, Ordering::SeqCst) + bytes;
    PEAK.fetch_max(now, Ordering::SeqCst);
}

fn shrink(bytes: usize) {
    LIVE.fetch_sub(bytes, Ordering::SeqCst);
}

// SAFETY: every method forwards to `System` with the caller's arguments
// unchanged and only updates two counters on success.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            grow(layout.size());
        }
        p
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            grow(layout.size());
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        shrink(layout.size());
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            if new_size >= layout.size() {
                grow(new_size - layout.size());
            } else {
                shrink(layout.size() - new_size);
            }
        }
        p
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// Heap measurements need the whole process to themselves.
static SERIAL: Mutex<()> = Mutex::new(());

const HUGE: u64 = 1 << 40;
/// Fixed allowance for bookkeeping that does not scale with the tensors
/// (see the module comment).
const SLACK: usize = 64 << 10;

/// Inputs and in-place targets live on their own unbounded budget, so the
/// backend's budget sees only what the op itself charges.
fn input(rng: &mut SplitMix64, shape: &[usize]) -> Tensor {
    let n = shape.iter().product();
    Tensor::from_f32(&rng.vec(n, 1.0), shape, &Budget::new(u64::MAX)).unwrap()
}

fn ids(rng: &mut SplitMix64, n: usize, below: usize) -> Tensor {
    let v: Vec<u32> = (0..n).map(|_| rng.below(below) as u32).collect();
    Tensor::from_u32(&v, &[n], &Budget::new(u64::MAX)).unwrap()
}

fn fresh(values: &[(Vec<f32>, Vec<usize>)]) -> Vec<Tensor> {
    values
        .iter()
        .map(|(v, s)| Tensor::from_f32(v, s, &Budget::new(u64::MAX)).unwrap())
        .collect()
}

fn bits(t: &Tensor) -> Vec<u32> {
    t.to_f32_vec().unwrap().iter().map(|x| x.to_bits()).collect()
}

type Run = Box<dyn Fn(&CpuBackend, &mut [Tensor]) -> Result<Vec<Tensor>, OjasError>>;

/// One op on fixed inputs. `targets` are the tensors it updates in place
/// (none for an op that returns new tensors); each run gets fresh copies.
struct Case {
    name: &'static str,
    targets: Vec<(Vec<f32>, Vec<usize>)>,
    run: Run,
}

impl Case {
    fn new(
        name: &'static str,
        run: impl Fn(&CpuBackend, &mut [Tensor]) -> Result<Vec<Tensor>, OjasError> + 'static,
    ) -> Self {
        Self {
            name,
            targets: Vec::new(),
            run: Box::new(run),
        }
    }

    fn in_place(
        name: &'static str,
        targets: Vec<Tensor>,
        run: impl Fn(&CpuBackend, &mut [Tensor]) -> Result<Vec<Tensor>, OjasError> + 'static,
    ) -> Self {
        Self {
            name,
            targets: targets
                .iter()
                .map(|t| (t.to_f32_vec().unwrap(), t.shape().to_vec()))
                .collect(),
            run: Box::new(run),
        }
    }
}

fn cases() -> Vec<Case> {
    let mut rng = SplitMix64(0x0b5e_55ed);
    let mut out = Vec::new();

    // Output 2048 x 512 from a 64-row table: the output dominates.
    let (table, tok) = (input(&mut rng, &[64, 512]), ids(&mut rng, 2048, 64));
    out.push(Case::new("embedding_forward", move |be, _| {
        be.embedding_forward(&table, &tok).map(|y| vec![y])
    }));
    // Output is the 8192 x 128 table gradient; 16 ids.
    let (table, tok, gy) = (
        input(&mut rng, &[8192, 128]),
        ids(&mut rng, 16, 8192),
        input(&mut rng, &[16, 128]),
    );
    out.push(Case::new("embedding_backward", move |be, _| {
        be.embedding_backward(&table, &tok, &gy).map(|y| vec![y])
    }));

    let (x, w, gy) = (
        input(&mut rng, &[512, 512]),
        input(&mut rng, &[512]),
        input(&mut rng, &[512, 512]),
    );
    let (x2, w2, gy2) = (x.clone(), w.clone(), gy.clone());
    out.push(Case::new("rms_norm_forward", move |be, _| {
        be.rms_norm_forward(&x, &w, 1e-6).map(|y| vec![y])
    }));
    out.push(Case::new("rms_norm_backward", move |be, _| {
        be.rms_norm_backward(&x2, &w2, &gy2, 1e-6)
            .map(|(a, b)| vec![a, b])
    }));

    // [batch, time, heads, dim] with [time, dim] tables.
    let (x, cos, sin) = (
        input(&mut rng, &[2, 128, 4, 256]),
        input(&mut rng, &[128, 256]),
        input(&mut rng, &[128, 256]),
    );
    let (gy, cos2, sin2) = (x.clone(), cos.clone(), sin.clone());
    out.push(Case::new("rope_half_split_forward", move |be, _| {
        be.rope_half_split_forward(&x, &cos, &sin).map(|y| vec![y])
    }));
    out.push(Case::new("rope_half_split_backward", move |be, _| {
        be.rope_half_split_backward(&gy, &cos2, &sin2)
            .map(|y| vec![y])
    }));

    // time 128 is the per-row path; time 320 takes the blocked path in Fast.
    for (name_f, name_b, shape) in [
        (
            "causal_sdpa_forward t128",
            "causal_sdpa_backward t128",
            [2usize, 8, 128, 64],
        ),
        (
            "causal_sdpa_forward t320",
            "causal_sdpa_backward t320",
            [1, 8, 320, 64],
        ),
    ] {
        let (q, k, v, gy) = (
            input(&mut rng, &shape),
            input(&mut rng, &shape),
            input(&mut rng, &shape),
            input(&mut rng, &shape),
        );
        let (q2, k2, v2) = (q.clone(), k.clone(), v.clone());
        out.push(Case::new(name_f, move |be, _| {
            be.causal_sdpa_forward(&q, &k, &v).map(|y| vec![y])
        }));
        out.push(Case::new(name_b, move |be, _| {
            be.causal_sdpa_backward(&q2, &k2, &v2, &gy)
                .map(|(a, b, c)| vec![a, b, c])
        }));
    }

    let (gx, gw, gb, ga, ggy) = (
        input(&mut rng, &[1024, 16]),
        input(&mut rng, &[8, 16]),
        input(&mut rng, &[8]),
        input(&mut rng, &[1024, 8, 64]),
        input(&mut rng, &[1024, 8, 64]),
    );
    let (gx2, gw2, gb2, ga2) = (gx.clone(), gw.clone(), gb.clone(), ga.clone());
    out.push(Case::new("per_head_sigmoid_gate_forward", move |be, _| {
        be.per_head_sigmoid_gate_forward(&gx, &gw, &gb, &ga)
            .map(|y| vec![y])
    }));
    out.push(Case::new("per_head_sigmoid_gate_backward", move |be, _| {
        be.per_head_sigmoid_gate_backward(&gx2, &gw2, &gb2, &ga2, &ggy)
            .map(|g| vec![g.input, g.weight, g.bias, g.attn_out])
    }));

    let (v, v0, lam, gy) = (
        input(&mut rng, &[256, 1024]),
        input(&mut rng, &[256, 1024]),
        input(&mut rng, &[1]),
        input(&mut rng, &[256, 1024]),
    );
    let (v2, v02, lam2) = (v.clone(), v0.clone(), lam.clone());
    out.push(Case::new("value_residual_blend_forward", move |be, _| {
        be.value_residual_blend_forward(&v, &v0, &lam)
            .map(|y| vec![y])
    }));
    out.push(Case::new("value_residual_blend_backward", move |be, _| {
        be.value_residual_blend_backward(&v2, &v02, &lam2, &gy)
            .map(|g| vec![g.value, g.value0, g.lambda])
    }));

    let (a, b, gy) = (
        input(&mut rng, &[256, 1024]),
        input(&mut rng, &[256, 1024]),
        input(&mut rng, &[256, 1024]),
    );
    {
        let (a, gy) = (a.clone(), gy.clone());
        out.push(Case::new("silu_forward", {
            let a = a.clone();
            move |be, _| be.silu_forward(&a).map(|y| vec![y])
        }));
        out.push(Case::new("silu_backward", move |be, _| {
            be.silu_backward(&a, &gy).map(|y| vec![y])
        }));
    }
    for (name, kind) in [
        ("mul_forward", 0u8),
        ("mul_backward", 1),
        ("residual_add_forward", 2),
        ("residual_add_backward", 3),
    ] {
        let (a, b, gy) = (a.clone(), b.clone(), gy.clone());
        out.push(Case::new(name, move |be, _| match kind {
            0 => be.mul_forward(&a, &b).map(|y| vec![y]),
            1 => be.mul_backward(&a, &b, &gy).map(|(x, y)| vec![x, y]),
            2 => be.residual_add_forward(&a, &b).map(|y| vec![y]),
            _ => be.residual_add_backward(&a, &b, &gy).map(|(x, y)| vec![x, y]),
        }));
    }

    let (logits, targets) = (input(&mut rng, &[512, 512]), ids(&mut rng, 512, 512));
    let (logits2, targets2) = (logits.clone(), targets.clone());
    out.push(Case::new("cross_entropy_mean_forward", move |be, _| {
        be.cross_entropy_mean_forward(&logits, &targets, None)
            .map(|y| vec![y])
    }));
    out.push(Case::new("cross_entropy_mean_backward", move |be, _| {
        be.cross_entropy_mean_backward(&logits2, &targets2, None)
            .map(|y| vec![y])
    }));

    let p = input(&mut rng, &[64, 128, 64]);
    out.push(Case::new("permute", move |be, _| {
        be.permute(&p, &[2, 0, 1]).map(|y| vec![y])
    }));

    // Norm far above 1e-3, so the clip rewrites every value.
    out.push(Case::in_place(
        "clip_grad_norm",
        vec![input(&mut rng, &[256, 1024]), input(&mut rng, &[1000])],
        |be, grads| be.clip_grad_norm(grads, 1e-3).map(|_| Vec::new()),
    ));

    let g = input(&mut rng, &[256, 1024]);
    let m2: Vec<f32> = rng.vec(256 * 1024, 1.0).iter().map(|x| x * x).collect();
    let m2 = Tensor::from_f32(&m2, &[256, 1024], &Budget::new(u64::MAX)).unwrap();
    out.push(Case::in_place(
        "adamw_step",
        vec![input(&mut rng, &[256, 1024]), input(&mut rng, &[256, 1024]), m2],
        move |be, t| {
            let [p, m1, m2] = t else {
                panic!("adamw_step takes three targets")
            };
            be.adamw_step(p, &g, m1, m2, 3, AdamWConfig::nanolab(1e-3, 0.1))
                .map(|()| Vec::new())
        },
    ));

    // Wide (rows < cols) and tall (rows > cols, transposed Newton-Schulz).
    for (name, shape) in [
        ("muon_ns5_step wide", [256usize, 384]),
        ("muon_ns5_step tall", [384, 256]),
    ] {
        let g = input(&mut rng, &shape);
        out.push(Case::in_place(
            name,
            vec![input(&mut rng, &shape), input(&mut rng, &shape)],
            move |be, t| {
                let [p, m] = t else {
                    panic!("muon_ns5_step takes two targets")
                };
                be.muon_ns5_step(p, &g, m, MuonNs5Config::nanolab_default())
                    .map(|()| Vec::new())
            },
        ));
    }
    out
}

/// What one run left behind: output bits, then in-place target bits.
type Seen = (Vec<Vec<u32>>, Vec<Vec<u32>>);

/// Run `case` with exactly `room` bytes free on `be`'s budget. Asserts the
/// budget is balanced either way: success holds exactly the outputs, and a
/// refusal holds nothing and leaves every in-place target unchanged.
fn with_room(be: &CpuBackend, case: &Case, room: u64) -> Result<Seen, OjasError> {
    let what = case.name;
    let budget = be.budget();
    assert_eq!(budget.live_bytes().unwrap(), 0, "{what}: budget not idle");
    let mut targets = fresh(&case.targets);
    let before: Vec<Vec<u32>> = targets.iter().map(bits).collect();
    let held = HUGE - room;
    let filler = budget.try_reserve(held).unwrap();
    let result = match (case.run)(be, &mut targets) {
        Ok(outputs) => {
            let out_bytes: u64 = outputs
                .iter()
                .map(|t| 4 * t.num_elements().unwrap() as u64)
                .sum();
            assert_eq!(
                budget.live_bytes().unwrap(),
                held + out_bytes,
                "{what}: room {room}: success must hold exactly the outputs"
            );
            let out_bits = outputs.iter().map(bits).collect();
            drop(outputs);
            Ok((out_bits, targets.iter().map(bits).collect()))
        }
        Err(err) => {
            let after: Vec<Vec<u32>> = targets.iter().map(bits).collect();
            assert!(after == before, "{what}: room {room}: a refusal wrote a target");
            Err(err)
        }
    };
    assert_eq!(
        budget.live_bytes().unwrap(),
        held,
        "{what}: room {room}: budget not released"
    );
    drop(filler);
    assert_eq!(budget.live_bytes().unwrap(), 0);
    result
}

fn accepts(be: &CpuBackend, case: &Case, room: u64) -> bool {
    match with_room(be, case, room) {
        Ok(_) => true,
        Err(OjasError::CapacityExceeded { .. }) => false,
        Err(err) => panic!("{}: room {room}: unexpected {err:?}", case.name),
    }
}

/// Smallest room `case` accepts.
fn charged_peak(be: &CpuBackend, case: &Case) -> u64 {
    let mut hi = 1u64 << 20;
    while !accepts(be, case, hi) {
        hi *= 2;
        assert!(hi < HUGE, "{}: refused with {hi} bytes of room", case.name);
    }
    let mut lo = 0u64;
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if accepts(be, case, mid) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    hi
}

/// Heap high-water growth while `case` runs once, outputs included. The
/// in-place targets are made before the count starts.
fn heap_peak(be: &CpuBackend, case: &Case) -> usize {
    let mut targets = fresh(&case.targets);
    let base = LIVE.load(Ordering::SeqCst);
    PEAK.store(base, Ordering::SeqCst);
    let out = (case.run)(be, &mut targets).unwrap();
    let peak = PEAK.load(Ordering::SeqCst) - base;
    drop(out);
    peak
}

fn backends() -> Vec<(String, CpuBackend)> {
    let mut out = Vec::new();
    for threads in [1usize, 7] {
        for numerics in [Numerics::Exact, Numerics::Fast] {
            let be = CpuBackend::with_threads(Budget::new(HUGE), threads)
                .unwrap()
                .with_numerics(numerics);
            out.push((format!("threads {threads} {numerics:?}"), be));
        }
    }
    out
}

/// Heap peak <= charged peak + `SLACK` for every op, thread count and
/// numerics contract.
#[test]
fn every_op_heap_peak_stays_within_its_charged_peak() {
    let _serial = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    let cases = cases();
    let mut failures = Vec::new();
    for (label, be) in backends() {
        for case in &cases {
            // Warm the pool so worker spawns are not counted.
            drop(with_room(&be, case, HUGE).unwrap());
            let charged = charged_peak(&be, case);
            let heap = heap_peak(&be, case);
            println!(
                "{} {label}: heap peak {heap} bytes, charged peak {charged} bytes",
                case.name
            );
            if heap > charged as usize + SLACK {
                failures.push(format!(
                    "{} {label}: heap {heap} > charged {charged} (+{} over)",
                    case.name,
                    heap - charged as usize
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "uncharged heap growth:\n{}",
        failures.join("\n")
    );
}

/// `peak - 1` refuses with `CapacityExceeded` and leaves the budget and the
/// in-place targets as they were; `peak` and more succeed with the same bits.
#[test]
fn every_op_budget_boundary_is_sharp_and_balanced() {
    let _serial = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    let cases = cases();
    for (label, be) in backends() {
        for case in &cases {
            let what = format!("{} {label}", case.name);
            let full = with_room(&be, case, HUGE)
                .unwrap_or_else(|err| panic!("{what}: refused with the whole budget: {err:?}"));
            let peak = charged_peak(&be, case);
            match with_room(&be, case, peak - 1) {
                Err(OjasError::CapacityExceeded { .. }) => {}
                other => panic!(
                    "{what}: peak - 1 = {} expected CapacityExceeded, got {:?}",
                    peak - 1,
                    other.map(|_| ())
                ),
            }
            for room in [peak, peak + 1, peak + 4096] {
                let got = with_room(&be, case, room)
                    .unwrap_or_else(|err| panic!("{what}: refused at {room} >= peak {peak}: {err:?}"));
                assert!(got == full, "{what}: bits depend on the budget");
            }
        }
    }
}

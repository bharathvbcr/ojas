//! Causal SDPA under `Numerics::Fast`, where more than 256 positions take the
//! blocked GEMM kernel (`ojas-cpu/src/attn/flash.rs`): parity with `Exact`
//! and with an f64 reference, thread-count invariance, refusal of NaN and
//! infinity before any budget charge, and a budget that is taken before the
//! kernel allocates, refuses cleanly one byte short, and covers the heap.
//!
//! A counting global allocator (this test binary only) measures the heap
//! high-water mark, as in `redteam_linear_heap.rs`, so every test here holds
//! [`SERIAL`]: a parity test allocating on another thread would show up in a
//! heap measurement.

mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};

use common::{bits, SplitMix64};
use ojas_core::{Backend, Budget, Numerics, OjasError, Tensor};
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

static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|p| p.into_inner())
}

const HUGE: u64 = 1 << 40;
/// Pool batch bookkeeping, result slots, `Arc` headers, per-row scale
/// vectors and error strings, which do not grow with the tensors.
const SLACK: usize = 64 << 10;

fn backend(threads: usize, numerics: Numerics) -> CpuBackend {
    CpuBackend::with_threads(Budget::new(HUGE), threads)
        .unwrap()
        .with_numerics(numerics)
}

/// Inputs live on their own unbounded budget, so a backend's budget sees
/// only what the op charges.
fn tensor(values: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_f32(values, shape, &Budget::new(u64::MAX)).unwrap()
}

/// `[q, k, v, grad_y]` values and their `[B, H, T, D]` shape.
struct Fixture {
    shape: [usize; 4],
    values: [Vec<f32>; 4],
}

impl Fixture {
    fn new(shape: [usize; 4], seed: u64, amplitude: f32) -> Self {
        let n: usize = shape.iter().product();
        let mut rng = SplitMix64(seed);
        let values = [
            rng.vec(n, amplitude),
            rng.vec(n, amplitude),
            rng.vec(n, 1.0),
            rng.vec(n, 1.0),
        ];
        Self { shape, values }
    }

    fn tensors(&self) -> [Tensor; 4] {
        [0, 1, 2, 3].map(|i| tensor(&self.values[i], &self.shape))
    }

    fn tag(&self) -> String {
        format!("{:?}", self.shape)
    }
}

/// `[y, grad_q, grad_k, grad_v]`.
type Outs = [Vec<f32>; 4];

fn run(be: &CpuBackend, t: &[Tensor; 4]) -> Outs {
    let y = be.causal_sdpa_forward(&t[0], &t[1], &t[2]).unwrap();
    let (gq, gk, gv) = be.causal_sdpa_backward(&t[0], &t[1], &t[2], &t[3]).unwrap();
    [y, gq, gk, gv].map(|x| x.to_f32_vec().unwrap())
}

/// f64 causal SDPA forward and backward of one `[time, dim]` head, written
/// from the definition: `S = scale·QKᵀ` masked to `j <= t`, `P = softmax(S)`,
/// `Y = PV`, `dP = dY·Vᵀ`, `dS = P∘(dP - rowsum(P∘dP))`,
/// `dQ = scale·dS·K`, `dK = scale·dSᵀ·Q`, `dV = Pᵀ·dY`. Also returns the
/// largest `|S|` over the causal pairs.
fn reference_head(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    time: usize,
    dim: usize,
) -> ([Vec<f64>; 4], f64) {
    let scale = 1.0 / (dim as f64).sqrt();
    let at = |x: &[f32], r: usize, c: usize| f64::from(x[r * dim + c]);
    let mut y = vec![0.0; time * dim];
    let mut gq = vec![0.0; time * dim];
    let mut gk = vec![0.0; time * dim];
    let mut gv = vec![0.0; time * dim];
    let mut p = vec![0.0f64; time];
    let mut dp = vec![0.0f64; time];
    let mut spread = 0.0f64;
    for t in 0..time {
        let mut max = f64::NEG_INFINITY;
        for (j, pj) in p.iter_mut().enumerate().take(t + 1) {
            let s: f64 = (0..dim).map(|c| at(q, t, c) * at(k, j, c)).sum::<f64>() * scale;
            *pj = s;
            max = max.max(s);
            spread = spread.max(s.abs());
        }
        let mut sum = 0.0;
        for pj in p.iter_mut().take(t + 1) {
            *pj = (*pj - max).exp();
            sum += *pj;
        }
        for pj in p.iter_mut().take(t + 1) {
            *pj /= sum;
        }
        let mut delta = 0.0;
        for (j, (dpj, &pj)) in dp.iter_mut().zip(&p).enumerate().take(t + 1) {
            *dpj = (0..dim).map(|c| at(g, t, c) * at(v, j, c)).sum();
            delta += pj * *dpj;
        }
        for j in 0..=t {
            let ds = p[j] * (dp[j] - delta) * scale;
            for c in 0..dim {
                y[t * dim + c] += p[j] * at(v, j, c);
                gq[t * dim + c] += ds * at(k, j, c);
                gk[j * dim + c] += ds * at(q, t, c);
                gv[j * dim + c] += p[j] * at(g, t, c);
            }
        }
    }
    ([y, gq, gk, gv], spread)
}

/// [`reference_head`] for every head, heads spread over OS threads.
fn reference(f: &Fixture) -> ([Vec<f64>; 4], f64) {
    let [b, h, time, dim] = f.shape;
    let stride = time * dim;
    let heads = b * h;
    let per_head: Vec<([Vec<f64>; 4], f64)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..heads)
            .map(|head| {
                let span = head * stride..(head + 1) * stride;
                let [q, k, v, g] = [0, 1, 2, 3].map(|i| &f.values[i][span.clone()]);
                scope.spawn(move || reference_head(q, k, v, g, time, dim))
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut out: [Vec<f64>; 4] = Default::default();
    let mut spread = 0.0f64;
    for (head, s) in per_head {
        spread = spread.max(s);
        for (o, part) in out.iter_mut().zip(head) {
            o.extend(part);
        }
    }
    (out, spread)
}

/// `max |got - want| / max |want|`.
fn rel_err(got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let scale = want
        .iter()
        .fold(0.0f64, |m, v| m.max(v.abs()))
        .max(f64::MIN_POSITIVE);
    let err = got
        .iter()
        .zip(want)
        .fold(0.0f64, |m, (&a, &b)| m.max((f64::from(a) - b).abs()));
    err / scale
}

const NAMES: [&str; 4] = ["y", "grad_q", "grad_k", "grad_v"];

/// Bound for Fast against the f64 reference, relative to the largest
/// reference magnitude of each output, plus [`PER_SCORE`] per unit of the
/// largest `|score|`. Measured on aarch64 with scores within about 2: at
/// most 9.1e-7 (the per-row kernel at T = 256), 6.3e-7 on the blocked one.
const TOL_F64: f64 = 2e-6;
/// Both kernels round each score to f32, an error of `|s|·2^-24` that the
/// exponential turns into a relative weight error of the same size; the
/// f64 reference does not. Measured 4.5e-6 for Fast and Exact alike at
/// `[1, 2, 1024, 64]` with scores up to about 27.
const PER_SCORE: f64 = 2e-7;
/// Exact's own distance from f64 grows with `T` (one f32 sum over up to
/// `T` keys); measured at most 2.1e-6, at `[1, 12, 1024, 128]` grad_k.
const TOL_EXACT_F64: f64 = 4e-6;
/// Where summation rather than score rounding dominates (scores within
/// [`NO_WORSE_MAX_SCORE`]), Fast may be less accurate than Exact by at most
/// this much (relative to the f64 magnitude); measured at most 8e-8. With
/// peaked scores both sit at the rounding floor (Fast 5.7e-6, Exact 4.2e-6
/// at `[2, 3, 300, 128]`, scores up to 30) and only the bound above holds.
const NO_WORSE: f64 = 2e-7;
const NO_WORSE_MAX_SCORE: f64 = 4.0;
/// Bound for Fast against Exact, relative to the largest Exact magnitude:
/// the bound `numerics.rs` states for attention, over the lengths it covers
/// (`T <= 520`). At `T = 1024` the gap is Exact's own drift (Fast is within
/// [`TOL_F64`] of f64 and closer to it than Exact), measured at most 2.1e-6.
const TOL_EXACT: f64 = 2e-6;
const TOL_EXACT_MAX_T: usize = 520;

#[test]
fn fast_matches_exact_and_f64_across_the_kernel_cutoff() {
    let _serial = serial();
    let fast = backend(18, Numerics::Fast);
    let exact = backend(18, Numerics::Exact);
    let mut cases: Vec<Fixture> = Vec::new();
    let mut seed = 0xa77e_0000u64;
    for &time in &[1usize, 17, 255, 256, 257, 1024] {
        for &dim in &[64usize, 128] {
            for &heads in &[1usize, 12] {
                seed += 1;
                cases.push(Fixture::new([1, heads, time, dim], seed, 1.0));
            }
        }
    }
    // Peaked rows: scores spread over tens of units, so most weights are
    // deep in the exponential's tail and the row maximum dominates.
    cases.push(Fixture::new([1, 2, 1024, 64], 0xbeef, 4.0));
    cases.push(Fixture::new([2, 3, 300, 128], 0xfeed, 4.0));
    let mut worst = [0.0f64; 2];
    for f in &cases {
        let t = f.tensors();
        let (want, spread) = reference(f);
        let got_fast = run(&fast, &t);
        let got_exact = run(&exact, &t);
        let rounding = PER_SCORE * spread;
        for i in 0..4 {
            let what = format!("{} {}", f.tag(), NAMES[i]);
            assert!(
                got_fast[i].iter().all(|v| v.is_finite()),
                "{what}: non-finite Fast output"
            );
            let fast_f64 = rel_err(&got_fast[i], &want[i]);
            let exact_f64 = rel_err(&got_exact[i], &want[i]);
            let exact_as_f64: Vec<f64> = got_exact[i].iter().map(|&v| f64::from(v)).collect();
            let fast_exact = rel_err(&got_fast[i], &exact_as_f64);
            worst[0] = worst[0].max(fast_f64);
            worst[1] = worst[1].max(fast_exact);
            println!(
                "{what}: fast-f64 {fast_f64:.2e} exact-f64 {exact_f64:.2e} \
                 fast-exact {fast_exact:.2e} max|s| {spread:.1}"
            );
            let tol = TOL_F64 + rounding;
            assert!(
                fast_f64 <= tol,
                "{what}: Fast vs f64 {fast_f64:e} > {tol:e}"
            );
            let tol = TOL_EXACT_F64 + rounding;
            assert!(
                exact_f64 <= tol,
                "{what}: Exact vs f64 {exact_f64:e} > {tol:e}"
            );
            if spread <= NO_WORSE_MAX_SCORE {
                assert!(
                    fast_f64 <= exact_f64 + NO_WORSE,
                    "{what}: Fast {fast_f64:e} is further from f64 than Exact {exact_f64:e}"
                );
            }
            if f.shape[2] <= TOL_EXACT_MAX_T {
                let tol = TOL_EXACT + rounding;
                assert!(
                    fast_exact <= tol,
                    "{what}: Fast vs Exact {fast_exact:e} > {tol:e}"
                );
            }
        }
    }
    println!(
        "worst fast-f64 {:.2e} fast-exact {:.2e}",
        worst[0], worst[1]
    );
}

/// At or below 256 positions Fast keeps the per-row kernel; above it the
/// blocked kernel must actually run (its bits differ from Exact's).
#[test]
fn fast_takes_the_blocked_kernel_only_above_256_positions() {
    let _serial = serial();
    let fast = backend(7, Numerics::Fast);
    let exact = backend(7, Numerics::Exact);
    for (time, blocked) in [(256usize, false), (257, true), (1024, true)] {
        let f = Fixture::new([1, 2, time, 64], 0x5eed + time as u64, 1.0);
        let t = f.tensors();
        let (a, b) = (run(&fast, &t), run(&exact, &t));
        let same = (0..4).all(|i| bits(&a[i]) == bits(&b[i]));
        assert_eq!(same, !blocked, "T = {time}: Fast == Exact bits is {same}");
    }
}

#[test]
fn fast_bits_do_not_depend_on_the_thread_count() {
    let _serial = serial();
    let fixtures = [
        Fixture::new([1, 12, 1024, 64], 0x7001, 1.0),
        Fixture::new([1, 1, 257, 128], 0x7002, 1.0),
        Fixture::new([2, 3, 300, 64], 0x7003, 4.0),
        Fixture::new([1, 2, 640, 128], 0x7004, 1.0),
    ];
    for f in &fixtures {
        let t = f.tensors();
        let want = run(&backend(1, Numerics::Fast), &t);
        for threads in [2usize, 7, 18] {
            let got = run(&backend(threads, Numerics::Fast), &t);
            for i in 0..4 {
                assert!(
                    bits(&got[i]) == bits(&want[i]),
                    "{} {}: {threads} threads differ from 1",
                    f.tag(),
                    NAMES[i]
                );
            }
        }
        // And the same backend twice.
        let be = backend(18, Numerics::Fast);
        assert!(run(&be, &t)
            .iter()
            .zip(&want)
            .all(|(a, b)| bits(a) == bits(b)));
    }
}

fn assert_nonfinite<T: std::fmt::Debug>(what: &str, r: Result<T, OjasError>) {
    assert!(
        matches!(r, Err(OjasError::NonFinite { .. })),
        "{what}: {r:?}"
    );
}

/// A NaN or infinity in any operand is refused as `NonFinite` on a budget
/// with no room at all: validation runs before the first charge.
#[test]
fn nan_and_inf_are_refused_before_any_charge() {
    let _serial = serial();
    let shape = [1usize, 2, 300, 64];
    let f = Fixture::new(shape, 0x0bad, 1.0);
    let n: usize = shape.iter().product();
    for threads in [1usize, 7] {
        let be = CpuBackend::with_threads(Budget::new(0), threads).unwrap();
        assert_eq!(be.numerics(), Numerics::Fast);
        for which in 0..4 {
            for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut values = f.values.clone();
                values[which][n / 2 + which] = bad;
                let t = values.each_ref().map(|v| tensor(v, &shape));
                let what = format!("threads {threads} operand {which} = {bad}");
                if which < 3 {
                    assert_nonfinite(&what, be.causal_sdpa_forward(&t[0], &t[1], &t[2]));
                }
                assert_nonfinite(&what, be.causal_sdpa_backward(&t[0], &t[1], &t[2], &t[3]));
                assert_eq!(be.budget().live_bytes().unwrap(), 0, "{what}");
            }
        }
    }
}

/// Finite inputs whose scores overflow are refused by the blocked kernel
/// itself, and the refusal releases every charge.
#[test]
fn overflowing_scores_are_refused_and_release_their_charge() {
    let _serial = serial();
    let shape = [1usize, 3, 300, 64];
    let n: usize = shape.iter().product();
    let huge = tensor(&vec![3.0e19f32; n], &shape);
    let small = tensor(&SplitMix64(9).vec(n, 1.0), &shape);
    for threads in [1usize, 7] {
        let be = backend(threads, Numerics::Fast);
        assert_nonfinite("fwd", be.causal_sdpa_forward(&huge, &huge, &small));
        assert_nonfinite("bwd", be.causal_sdpa_backward(&huge, &huge, &small, &small));
        assert_eq!(be.budget().live_bytes().unwrap(), 0);
    }
}

type Op<'a> = &'a dyn Fn(&CpuBackend) -> Result<Vec<Tensor>, OjasError>;

/// `op` with exactly `room` bytes free on the backend budget. A success
/// leaves exactly its outputs charged; a refusal is `CapacityExceeded` and
/// leaves nothing charged.
fn with_room(be: &CpuBackend, room: u64, op: Op<'_>) -> Option<Vec<Vec<u32>>> {
    let budget = be.budget();
    assert_eq!(budget.live_bytes().unwrap(), 0, "budget not idle");
    let held = HUGE - room;
    let filler = budget.try_reserve(held).unwrap();
    let out = match op(be) {
        Ok(outputs) => {
            let bytes: u64 = outputs
                .iter()
                .map(|t| 4 * t.num_elements().unwrap() as u64)
                .sum();
            assert_eq!(budget.live_bytes().unwrap(), held + bytes, "room {room}");
            Some(
                outputs
                    .iter()
                    .map(|t| {
                        t.to_f32_vec()
                            .unwrap()
                            .iter()
                            .map(|v| v.to_bits())
                            .collect()
                    })
                    .collect(),
            )
        }
        Err(OjasError::CapacityExceeded { .. }) => {
            assert_eq!(
                budget.live_bytes().unwrap(),
                held,
                "room {room}: refusal leaked"
            );
            None
        }
        Err(err) => panic!("room {room}: unexpected {err:?}"),
    };
    drop(filler);
    assert_eq!(budget.live_bytes().unwrap(), 0);
    out
}

/// Smallest room `op` accepts.
fn charged_peak(be: &CpuBackend, op: Op<'_>) -> u64 {
    let mut hi = 1u64 << 20;
    while with_room(be, hi, op).is_none() {
        hi *= 2;
        assert!(hi < HUGE, "refused with {hi} bytes of room");
    }
    let mut lo = 0u64;
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if with_room(be, mid, op).is_some() {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    hi
}

/// Heap high-water growth while `op` runs once, outputs included.
fn heap_peak(be: &CpuBackend, op: Op<'_>) -> usize {
    let base = LIVE.load(Ordering::SeqCst);
    PEAK.store(base, Ordering::SeqCst);
    let out = op(be).unwrap();
    let peak = PEAK.load(Ordering::SeqCst) - base;
    drop(out);
    peak
}

/// For forward and backward, Exact and Fast, 1 and 7 threads: the heap
/// peak stays within the charged peak (plus [`SLACK`]), one byte less room
/// is a clean refusal, and every room from the peak up gives the same bits.
#[test]
fn sdpa_budget_is_sharp_balanced_and_covers_the_heap() {
    let _serial = serial();
    let mut failures = Vec::new();
    for shape in [[1usize, 4, 300, 64], [1, 12, 1024, 64]] {
        let f = Fixture::new(shape, 0x4ea9 + shape[2] as u64, 1.0);
        let t = f.tensors();
        let fwd = |be: &CpuBackend| be.causal_sdpa_forward(&t[0], &t[1], &t[2]).map(|y| vec![y]);
        let bwd = |be: &CpuBackend| {
            be.causal_sdpa_backward(&t[0], &t[1], &t[2], &t[3])
                .map(|(a, b, c)| vec![a, b, c])
        };
        for threads in [1usize, 7] {
            for numerics in [Numerics::Exact, Numerics::Fast] {
                let be = backend(threads, numerics);
                for (dir, op) in [("forward", &fwd as Op<'_>), ("backward", &bwd as Op<'_>)] {
                    let what = format!("{dir} {} threads {threads} {numerics:?}", f.tag());
                    // Warm the pool so worker spawns are not counted.
                    let full = with_room(&be, HUGE, op).expect("refused with the whole budget");
                    let peak = charged_peak(&be, op);
                    assert!(
                        with_room(&be, peak - 1, op).is_none(),
                        "{what}: peak - 1 accepted"
                    );
                    for room in [peak, peak + 1, peak + 4096] {
                        let got = with_room(&be, room, op)
                            .unwrap_or_else(|| panic!("{what}: refused at {room} >= {peak}"));
                        assert!(got == full, "{what}: bits depend on the budget");
                    }
                    let heap = heap_peak(&be, op);
                    println!("{what}: heap peak {heap} bytes, charged peak {peak} bytes");
                    if heap > peak as usize + SLACK {
                        failures.push(format!("{what}: heap {heap} > charged {peak}"));
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "uncharged heap growth:\n{}",
        failures.join("\n")
    );
}

//! `Backend::adamw_step` against `torch.optim.AdamW` over 100 steps, on the
//! LoRA recipe's hyperparameters (Lappi campaign/lora-2b-preregistered.json:
//! betas (0.9, 0.999), eps 1e-8, decoupled weight decay 0.01, lr 2e-4 for the
//! adapters and 1e-4 for D17 and the span head).
//!
//! Five scaled-down tensors, two lr groups, one tensor without weight decay,
//! and whole-tensor zero gradients at some steps. The gradient is closed-form
//! integer arithmetic that `ojas-oracle/python/adamw_oracle.py` computes with
//! the same bits, so no data crosses the language boundary except the
//! fixture's torch results at steps 1, 10, 50 and 100 (parameters) and 100
//! (both moments). The bound is the fixture's, derived from torch's own
//! float32-vs-float64 gap (see the generator's docstring).
//!
//! The CPU runs under both `Numerics`. Metal (macOS) runs the same driver; a
//! missing device fails unless `OJAS_ALLOW_NO_GPU=1`. A missing fixture
//! fails loudly.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ojas_core::{AdamWConfig, Backend, Budget, Numerics, Tensor};
use ojas_cpu::CpuBackend;
use ojas_io::SafeTensors;

const STEPS: u64 = 100;
const PARAM_CHECKPOINTS: [u64; 4] = [1, 10, 50, 100];
const BETA1: f64 = 0.9;
const BETA2: f64 = 0.999;
const EPS: f64 = 1e-8;

struct Spec {
    name: &'static str,
    shape: &'static [usize],
    lr: f64,
    weight_decay: f64,
}

/// The generator's TENSORS, in its tensor-id order.
static TENSORS: [Spec; 5] = [
    Spec {
        name: "lora_A",
        shape: &[16, 64],
        lr: 2e-4,
        weight_decay: 0.01,
    },
    Spec {
        name: "lora_B",
        shape: &[64, 16],
        lr: 2e-4,
        weight_decay: 0.01,
    },
    Spec {
        name: "d17",
        shape: &[17, 64],
        lr: 1e-4,
        weight_decay: 0.01,
    },
    Spec {
        name: "span_head",
        shape: &[2, 64],
        lr: 1e-4,
        weight_decay: 0.01,
    },
    Spec {
        name: "norm_nodecay",
        shape: &[64],
        lr: 1e-4,
        weight_decay: 0.0,
    },
];
/// The generator's ZERO_GRAD: (tensor id, first step, last step), 1-based.
static ZERO_GRAD: [(u32, u64, u64); 2] = [(1, 30, 39), (4, 1, 3)];

const SALT_INIT: u32 = 1;
const SALT_GRAD: u32 = 2;
const SALT_OFFSET: u32 = 3;

fn numel(spec: &Spec) -> usize {
    spec.shape.iter().product()
}

fn hash32(salt: u32, step: u32, tensor: u32, index: u32) -> u32 {
    let mut x = salt
        .wrapping_mul(0x27D4_EB2F)
        .wrapping_add(step.wrapping_mul(0x9E37_79B1))
        .wrapping_add(tensor.wrapping_mul(0x1656_67B1))
        .wrapping_add(index.wrapping_mul(0xC2B2_AE35))
        .wrapping_add(0x85EB_CA6B);
    x ^= x >> 16;
    x = x.wrapping_mul(0x85EB_CA6B);
    x ^= x >> 13;
    x = x.wrapping_mul(0xC2B2_AE35);
    x ^ (x >> 16)
}

fn signed24(h: u32) -> i64 {
    i64::from(h >> 8) - (1 << 23)
}

/// `2^exp` as an f32, exact (`exp` in the normal range).
fn pow2(exp: i32) -> f32 {
    f32::from_bits(((127 + exp) as u32) << 23)
}

fn zero_grad(step: u64, tensor: u32) -> bool {
    ZERO_GRAD
        .iter()
        .any(|&(t, first, last)| t == tensor && (first..=last).contains(&step))
}

fn init_int(tensor: u32, index: u32) -> i64 {
    signed24(hash32(SALT_INIT, 0, tensor, index))
}

fn grad_int(step: u64, tensor: u32, index: u32) -> i64 {
    if zero_grad(step, tensor) {
        return 0;
    }
    let noise = signed24(hash32(SALT_GRAD, step as u32, tensor, index));
    let offset = signed24(hash32(SALT_OFFSET, 0, tensor, index));
    noise + (offset >> 1)
}

/// `value` is below 2^24 in magnitude, so the f32 is exact.
fn exact_f32(value: i64) -> f32 {
    assert!(
        value.unsigned_abs() < 1 << 24,
        "{value} is not exact in f32"
    );
    value as f32
}

fn init_value(tensor: u32, index: u32) -> f32 {
    exact_f32(init_int(tensor, index)) * pow2(-27)
}

fn grad_value(step: u64, tensor: u32, index: u32) -> f32 {
    exact_f32(grad_int(step, tensor, index)) * pow2(-33)
}

fn init_tensor(tensor: u32) -> Vec<f32> {
    (0..numel(&TENSORS[tensor as usize]) as u32)
        .map(|i| init_value(tensor, i))
        .collect()
}

fn grad_tensor(step: u64, tensor: u32) -> Vec<f32> {
    (0..numel(&TENSORS[tensor as usize]) as u32)
        .map(|i| grad_value(step, tensor, i))
        .collect()
}

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/adamw_100")
}

/// A missing or unreadable fixture panics with its path.
fn open_fixture(dir: &Path) -> SafeTensors<'static> {
    let path = dir.join("adamw_100.safetensors");
    match SafeTensors::open(&path) {
        Ok(file) => file,
        Err(err) => panic!(
            "adamw_100 fixture {} is missing or unreadable ({err:?}); generate it with \
             python3.14 ojas-oracle/python/adamw_oracle.py",
            path.display()
        ),
    }
}

fn read_f32(file: &SafeTensors<'_>, name: &str, shape: &[usize]) -> Vec<f32> {
    let (dims, values) = file
        .read_f32(name)
        .unwrap_or_else(|e| panic!("{name}: {e:?}"));
    let want: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
    assert_eq!(dims, want, "{name}: shape");
    values
}

fn read_i64(file: &SafeTensors<'_>, name: &str, cols: u64) -> Vec<i64> {
    let (dims, values) = file
        .read_i64(name)
        .unwrap_or_else(|e| panic!("{name}: {e:?}"));
    assert!(
        dims.len() == 1 || (dims.len() == 2 && dims[1] == cols),
        "{name}: shape {dims:?}"
    );
    values
}

fn bound(file: &SafeTensors<'_>) -> f64 {
    let bound = f64::from(read_f32(file, "bound", &[1])[0]);
    let gap = f64::from(read_f32(file, "gap", &[1])[0]);
    // The bound is derived from the gap (4 x, rounded up to two digits). A
    // fixture whose bound drifted from that, or that is loose, is refused.
    assert!(gap > 0.0 && gap < 1e-3, "gap {gap:e}");
    assert!(
        bound >= 4.0 * gap * 0.999 && bound <= 4.0 * gap * 1.1,
        "bound {bound:e} vs gap {gap:e}"
    );
    bound
}

/// `max|x - ref| / max|ref|`, in f64. A non-finite value or an all-zero
/// reference fails.
fn peak_rel(what: &str, got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len(), "{what}: length");
    assert!(
        got.iter().all(|x| x.is_finite()),
        "{what}: non-finite value in the result"
    );
    let peak = want.iter().map(|x| f64::from(x.abs())).fold(0.0, f64::max);
    assert!(
        peak > 0.0 && peak.is_finite(),
        "{what}: reference peak {peak}"
    );
    let worst = got
        .iter()
        .zip(want)
        .map(|(g, w)| (f64::from(*g) - f64::from(*w)).abs())
        .fold(0.0, f64::max);
    worst / peak
}

type Checkpoints = BTreeMap<String, Vec<f32>>;

/// Where tensors live and how they come back; the driver is otherwise the
/// same on every backend.
struct Placement<'a> {
    put: &'a dyn Fn(&[f32], &[usize]) -> Tensor,
    get: &'a dyn Fn(&Tensor) -> Vec<f32>,
}

/// What the run passes to `adamw_step` for tensor `t` at 1-based `step`: the
/// step count before the update and the config. The honest one is
/// [`honest`]; the sensitivity tests bend it.
type Call = dyn Fn(usize, u64) -> (u64, AdamWConfig);

fn honest(t: usize, step: u64) -> (u64, AdamWConfig) {
    let spec = &TENSORS[t];
    let config = AdamWConfig {
        lr: spec.lr,
        beta1: BETA1,
        beta2: BETA2,
        eps: EPS,
        weight_decay: spec.weight_decay,
    };
    (step - 1, config)
}

/// The 100-step run on `backend`: `p<step>.<name>` at the parameter
/// checkpoints, `m100.<name>` and `v100.<name>` at the end.
fn run<B: Backend>(backend: &B, place: &Placement<'_>, call: &Call) -> Checkpoints {
    let mut params: Vec<Tensor> = Vec::new();
    let mut m1: Vec<Tensor> = Vec::new();
    let mut m2: Vec<Tensor> = Vec::new();
    for (t, spec) in TENSORS.iter().enumerate() {
        let zeros = vec![0.0f32; numel(spec)];
        params.push((place.put)(&init_tensor(t as u32), spec.shape));
        m1.push((place.put)(&zeros, spec.shape));
        m2.push((place.put)(&zeros, spec.shape));
    }
    let mut out = Checkpoints::new();
    for step in 1..=STEPS {
        for (t, spec) in TENSORS.iter().enumerate() {
            let grad = (place.put)(&grad_tensor(step, t as u32), spec.shape);
            let (before, config) = call(t, step);
            backend
                .adamw_step(
                    &mut params[t],
                    &grad,
                    &mut m1[t],
                    &mut m2[t],
                    before,
                    config,
                )
                .unwrap_or_else(|e| panic!("{} step {step}: {e:?}", spec.name));
        }
        if PARAM_CHECKPOINTS.contains(&step) {
            backend.sync().expect("sync");
            for (t, spec) in TENSORS.iter().enumerate() {
                out.insert(format!("p{step}.{}", spec.name), (place.get)(&params[t]));
            }
        }
    }
    backend.sync().expect("sync");
    for (t, spec) in TENSORS.iter().enumerate() {
        out.insert(format!("m100.{}", spec.name), (place.get)(&m1[t]));
        out.insert(format!("v100.{}", spec.name), (place.get)(&m2[t]));
    }
    out
}

/// Every checkpoint's peak-relative error against the fixture, worst first
/// asserted against `bound`. Returns the worst error.
fn compare(label: &str, got: &Checkpoints, file: &SafeTensors<'_>, bound: f64) -> f64 {
    let mut worst = 0.0f64;
    let mut checked = 0;
    for (name, values) in got {
        let spec = TENSORS
            .iter()
            .find(|s| name.ends_with(&format!(".{}", s.name)))
            .unwrap_or_else(|| panic!("{name}: no tensor spec"));
        let want = read_f32(file, name, spec.shape);
        let err = peak_rel(&format!("{label} {name}"), values, &want);
        eprintln!("{label} {name}: peak-relative {err:.3e} (bound {bound:.1e})");
        assert!(
            err <= bound,
            "{label} {name}: {err:.3e} exceeds the bound {bound:.3e}"
        );
        worst = worst.max(err);
        checked += 1;
    }
    // 5 tensors x (4 parameter checkpoints + 2 moments).
    assert_eq!(
        checked,
        TENSORS.len() * (PARAM_CHECKPOINTS.len() + 2),
        "{label}: checkpoints"
    );
    worst
}

fn cpu_placement(budget: &Budget) -> impl Fn(&[f32], &[usize]) -> Tensor + '_ {
    move |data, shape| Tensor::from_f32(data, shape, budget).expect("host tensor")
}

fn cpu_run(numerics: Numerics, call: &Call) -> Checkpoints {
    let budget = Budget::new(1 << 30);
    let backend = CpuBackend::new(budget.clone()).with_numerics(numerics);
    let put = cpu_placement(&budget);
    let get = |t: &Tensor| t.to_f32_vec().expect("read back");
    run(
        &backend,
        &Placement {
            put: &put,
            get: &get,
        },
        call,
    )
}

#[test]
fn the_rust_formula_reproduces_the_generators_probes() {
    let file = open_fixture(&fixture_dir());
    let keys = read_i64(&file, "probe.key", 3);
    let ints = read_i64(&file, "probe.gi", 1);
    let grads = read_f32(&file, "probe.grad", &[ints.len()]);
    assert!(
        ints.len() >= 50 && keys.len() == 3 * ints.len(),
        "probe count {}",
        ints.len()
    );
    let mut zeros = 0;
    for (n, key) in keys.chunks(3).enumerate() {
        let (step, tensor, index) = (key[0] as u64, key[1] as u32, key[2] as u32);
        assert_eq!(
            grad_int(step, tensor, index),
            ints[n],
            "grad int at {key:?}"
        );
        assert_eq!(
            grad_value(step, tensor, index).to_bits(),
            grads[n].to_bits(),
            "grad bits at {key:?}"
        );
        zeros += usize::from(ints[n] == 0);
    }
    assert!(zeros > 0, "no probe covers a zero-gradient step");

    let init_keys = read_i64(&file, "probe.init_key", 2);
    let inits = read_f32(&file, "probe.init", &[init_keys.len() / 2]);
    for (n, key) in init_keys.chunks(2).enumerate() {
        let (tensor, index) = (key[0] as u32, key[1] as u32);
        assert_eq!(
            init_value(tensor, index).to_bits(),
            inits[n].to_bits(),
            "init bits at {key:?}"
        );
    }
}

#[test]
fn the_zero_gradient_steps_are_exactly_zero_and_the_rest_are_not() {
    for &(tensor, first, last) in &ZERO_GRAD {
        for step in [first, last] {
            assert!(grad_tensor(step, tensor).iter().all(|g| g.to_bits() == 0));
        }
        let next = grad_tensor(last + 1, tensor);
        assert!(
            next.iter().any(|g| *g != 0.0),
            "tensor {tensor} step {}",
            last + 1
        );
    }
    for (t, spec) in TENSORS.iter().enumerate() {
        assert!(
            init_tensor(t as u32).iter().any(|p| *p != 0.0),
            "{}",
            spec.name
        );
    }
}

fn cpu_matches_torch(numerics: Numerics) {
    let file = open_fixture(&fixture_dir());
    let bound = bound(&file);
    let got = cpu_run(numerics, &honest);
    let label = format!("cpu {numerics:?}");
    compare(&label, &got, &file, bound);
    // Torch leaves a parameter alone when its gradient and moments are all
    // zero and it has no decay (`norm_nodecay`, steps 1 to 3): the first
    // update of that tensor is the identity, so step 1 is the init, to the bit.
    let step1 = &got["p1.norm_nodecay"];
    let init = init_tensor(4);
    assert!(
        step1
            .iter()
            .zip(&init)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "{label}: a zero first step moved norm_nodecay"
    );
}

#[test]
fn cpu_exact_matches_torch_adamw_over_100_steps() {
    cpu_matches_torch(Numerics::Exact);
}

#[test]
fn cpu_fast_matches_torch_adamw_over_100_steps() {
    cpu_matches_torch(Numerics::Fast);
}

/// The comparison is not vacuous: the two ways to get the recipe wrong (the
/// weight decay on the decay-free tensor, and the step count passed after
/// the update instead of before) both land outside the bound.
#[test]
fn a_wrong_weight_decay_or_step_count_is_outside_the_bound() {
    let file = open_fixture(&fixture_dir());
    let bound = bound(&file);
    let decays_everything: &Call = &|t, step| {
        let (before, config) = honest(t, step);
        (
            before,
            AdamWConfig {
                weight_decay: 0.01,
                ..config
            },
        )
    };
    let step_after: &Call = &|t, step| {
        let (_, config) = honest(t, step);
        (step, config)
    };
    for (what, call) in [
        ("decay on norm_nodecay", decays_everything),
        ("step after", step_after),
    ] {
        let got = cpu_run(Numerics::Exact, call);
        let outside = got.iter().any(|(name, values)| {
            let spec = TENSORS
                .iter()
                .find(|s| name.ends_with(&format!(".{}", s.name)))
                .unwrap();
            let want = read_f32(&file, name, spec.shape);
            peak_rel(name, values, &want) > bound
        });
        assert!(
            outside,
            "{what}: every checkpoint stayed inside the bound {bound:e}"
        );
    }
}

#[test]
#[should_panic(expected = "adamw_100 fixture")]
fn a_missing_fixture_fails_loudly() {
    let _ = open_fixture(&fixture_dir().join("no-such-dir"));
}

#[cfg(target_os = "macos")]
mod metal {
    use super::*;
    use ojas_metal::MetalBackend;

    fn backend() -> Option<MetalBackend> {
        match MetalBackend::new(Budget::new(8 << 30)) {
            Ok(m) => Some(m),
            Err(e) if std::env::var_os("OJAS_ALLOW_NO_GPU").is_some() => {
                eprintln!("no Metal device: {e:?}");
                None
            }
            Err(e) => panic!("no Metal device (set OJAS_ALLOW_NO_GPU=1 to skip): {e:?}"),
        }
    }

    #[test]
    fn metal_matches_torch_adamw_over_100_steps() {
        let Some(m) = backend() else { return };
        let file = open_fixture(&fixture_dir());
        let bound = bound(&file);
        let host_budget = Budget::new(1 << 30);
        let put = |data: &[f32], shape: &[usize]| {
            let host = Tensor::from_f32(data, shape, &host_budget).expect("host tensor");
            m.upload(&host).expect("upload")
        };
        let get = |t: &Tensor| {
            t.to_host(&host_budget)
                .expect("download")
                .to_f32_vec()
                .expect("decode")
        };
        let got = run(
            &m,
            &Placement {
                put: &put,
                get: &get,
            },
            &honest,
        );
        compare("metal", &got, &file, bound);
    }
}

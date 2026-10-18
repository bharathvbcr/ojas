//! `adamw_all_or_nothing`: with every gradient finite it is bit for bit
//! `Backend::adamw_step` called per tensor; with a bad one in tensor `k > 0`
//! no tensor and no moment moves and the error names `k`. The last test of
//! the CPU half shows the per-tensor loop is not all or nothing, which is
//! why the two passes exist.

use ojas_core::{AdamWConfig, Backend, Budget, DType, Numerics, Tensor};
use ojas_cpu::CpuBackend;
use ojas_model::qwen35::optim::{adamw_all_or_nothing, AdamWEntry, OptimError, GRAD_ABS_LIMIT};

const NAMES: [&str; 4] = ["a.lora_A", "a.lora_B", "d17", "span_head"];
const SHAPES: [&[usize]; 4] = [&[3, 2], &[5], &[2, 2], &[7]];

fn cpu() -> CpuBackend {
    CpuBackend::new(Budget::new(1 << 28)).with_numerics(Numerics::Exact)
}

fn budget() -> Budget {
    Budget::new(1 << 28)
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn config(i: usize) -> AdamWConfig {
    AdamWConfig {
        lr: 1e-2 * (i as f64 + 1.0) / 2.0,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: if i.is_multiple_of(2) { 0.01 } else { 0.0 },
    }
}

fn values(n: usize, salt: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 7 + salt * 13) % 11) as f32 - 5.0) * scale)
        .collect()
}

struct State {
    p: Tensor,
    m1: Tensor,
    m2: Tensor,
}

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Parameters, and moments that are not zero (a step in progress): the
/// second moment is a square so it is non-negative.
fn states(b: &Budget) -> Vec<State> {
    SHAPES
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let n = numel(s);
            let m2: Vec<f32> = values(n, i + 40, 0.01).iter().map(|x| x * x).collect();
            State {
                p: Tensor::from_f32(&values(n, i, 0.1), s, b).unwrap(),
                m1: Tensor::from_f32(&values(n, i + 20, 0.01), s, b).unwrap(),
                m2: Tensor::from_f32(&m2, s, b).unwrap(),
            }
        })
        .collect()
}

fn grads(b: &Budget, salt: usize) -> Vec<Tensor> {
    SHAPES
        .iter()
        .enumerate()
        .map(|(i, s)| Tensor::from_f32(&values(numel(s), i + salt, 0.5), s, b).unwrap())
        .collect()
}

fn snapshot(states: &[State]) -> Vec<Vec<u32>> {
    states
        .iter()
        .flat_map(|s| [&s.p, &s.m1, &s.m2].map(|t| bits(&t.to_f32_vec().unwrap())))
        .collect()
}

fn entries<'a>(states: &'a mut [State], grads: &'a [Tensor]) -> Vec<AdamWEntry<'a>> {
    states
        .iter_mut()
        .zip(grads)
        .enumerate()
        .map(|(i, (s, g))| AdamWEntry {
            name: NAMES[i],
            param: &mut s.p,
            grad: g,
            moment1: &mut s.m1,
            moment2: &mut s.m2,
            config: config(i),
        })
        .collect()
}

fn with_grad_value(b: &Budget, salt: usize, at: usize, position: usize, v: f32) -> Vec<Tensor> {
    let mut all: Vec<Vec<f32>> = SHAPES
        .iter()
        .enumerate()
        .map(|(i, s)| values(numel(s), i + salt, 0.5))
        .collect();
    all[at][position] = v;
    all.iter()
        .zip(SHAPES)
        .map(|(v, s)| Tensor::from_f32(v, s, b).unwrap())
        .collect()
}

#[test]
fn all_finite_equals_per_tensor_stepping_bit_for_bit() {
    let (backend, b) = (cpu(), budget());
    let mut wrapped = states(&b);
    let mut direct = states(&b);
    assert_eq!(snapshot(&wrapped), snapshot(&direct));
    for step in 0..3u64 {
        let g = grads(&b, step as usize * 3);
        adamw_all_or_nothing(&backend, &mut entries(&mut wrapped, &g), step, &b).unwrap();
        for (i, (s, g)) in direct.iter_mut().zip(&g).enumerate() {
            backend
                .adamw_step(&mut s.p, g, &mut s.m1, &mut s.m2, step, config(i))
                .unwrap();
        }
        assert_eq!(snapshot(&wrapped), snapshot(&direct), "step {step}");
    }
    // The steps moved things at all.
    assert_ne!(snapshot(&wrapped), snapshot(&states(&b)));
}

#[test]
fn a_bad_gradient_in_tensor_k_leaves_everything_unchanged() {
    let (backend, b) = (cpu(), budget());
    // NaN, both infinities, a value whose square overflows f32, and f32::MAX.
    let bad = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        1.0e30,
        -1.0e30,
        f32::MAX,
    ];
    for at in [1usize, 2, 3] {
        for v in bad {
            let mut s = states(&b);
            let before = snapshot(&s);
            let g = with_grad_value(&b, 5, at, 1, v);
            let err = adamw_all_or_nothing(&backend, &mut entries(&mut s, &g), 4, &b).unwrap_err();
            match &err {
                OptimError::NonFiniteGrad {
                    entry,
                    name,
                    position,
                    ..
                } => {
                    assert_eq!(
                        (*entry, name.as_str(), *position),
                        (at, NAMES[at], 1),
                        "{v}"
                    );
                }
                other => panic!("tensor {at}, value {v}: {other}"),
            }
            assert!(err.to_string().contains(NAMES[at]), "{err}");
            assert_eq!(snapshot(&s), before, "tensor {at}, value {v}");
        }
    }
}

#[test]
fn the_first_bad_tensor_is_the_one_named_and_tensor_zero_is_covered_too() {
    let (backend, b) = (cpu(), budget());
    let mut s = states(&b);
    let before = snapshot(&s);
    let mut g = with_grad_value(&b, 7, 0, 0, f32::NAN);
    g[3] = with_grad_value(&b, 7, 3, 2, f32::INFINITY).remove(3);
    let err = adamw_all_or_nothing(&backend, &mut entries(&mut s, &g), 0, &b).unwrap_err();
    assert!(
        matches!(
            err,
            OptimError::NonFiniteGrad {
                entry: 0,
                position: 0,
                ..
            }
        ),
        "{err}"
    );
    assert_eq!(snapshot(&s), before);
}

#[test]
fn the_largest_squarable_gradient_is_accepted() {
    let (backend, b) = (cpu(), budget());
    let mut s = states(&b);
    let before = snapshot(&s);
    let g = with_grad_value(&b, 9, 2, 0, GRAD_ABS_LIMIT);
    adamw_all_or_nothing(&backend, &mut entries(&mut s, &g), 0, &b).unwrap();
    assert_ne!(snapshot(&s), before);
    for t in s.iter().flat_map(|s| [&s.p, &s.m1, &s.m2]) {
        assert!(t.to_f32_vec().unwrap().iter().all(|x| x.is_finite()));
    }
}

#[test]
fn operands_config_and_step_are_refused_before_any_tensor_moves() {
    let (backend, b) = (cpu(), budget());
    let ok = grads(&b, 1);

    // A gradient of the wrong shape in tensor 1.
    let mut s = states(&b);
    let before = snapshot(&s);
    let mut g = grads(&b, 1);
    g[1] = Tensor::from_f32(&[0.0; 4], &[4], &b).unwrap();
    let err = adamw_all_or_nothing(&backend, &mut entries(&mut s, &g), 0, &b).unwrap_err();
    assert!(matches!(err, OptimError::Refused { entry: 1, .. }), "{err}");
    assert_eq!(snapshot(&s), before);

    // A bf16 gradient in tensor 2.
    let mut g = grads(&b, 1);
    g[2] = Tensor::zeros(SHAPES[2], DType::Bf16, &b).unwrap();
    let err = adamw_all_or_nothing(&backend, &mut entries(&mut s, &g), 0, &b).unwrap_err();
    assert!(matches!(err, OptimError::Refused { entry: 2, .. }), "{err}");
    assert_eq!(snapshot(&s), before);

    // A NaN learning rate on tensor 3: refused at 3, tensors 0..3 unmoved.
    let mut es = entries(&mut s, &ok);
    es[3].config.lr = f64::NAN;
    let err = adamw_all_or_nothing(&backend, &mut es, 0, &b).unwrap_err();
    assert!(matches!(err, OptimError::Refused { entry: 3, .. }), "{err}");
    drop(es);
    assert_eq!(snapshot(&s), before);

    // A step counter that cannot advance.
    let err = adamw_all_or_nothing(&backend, &mut entries(&mut s, &ok), u64::MAX, &b).unwrap_err();
    assert!(matches!(err, OptimError::Refused { entry: 0, .. }), "{err}");
    assert_eq!(snapshot(&s), before);
}

#[test]
fn no_entries_is_a_no_op() {
    let (backend, b) = (cpu(), budget());
    adamw_all_or_nothing(&backend, &mut [], 0, &b).unwrap();
}

#[test]
fn per_tensor_stepping_alone_is_not_all_or_nothing() {
    // The behaviour the wrapper exists to prevent: Backend::adamw_step on
    // each tensor in turn moves tensors 0 and 1 before tensor 2 refuses.
    let (backend, b) = (cpu(), budget());
    let mut s = states(&b);
    let before = snapshot(&s);
    let g = with_grad_value(&b, 5, 2, 1, f32::NAN);
    let mut refused_at = None;
    for (i, (s, g)) in s.iter_mut().zip(&g).enumerate() {
        if backend
            .adamw_step(&mut s.p, g, &mut s.m1, &mut s.m2, 0, config(i))
            .is_err()
        {
            refused_at = Some(i);
            break;
        }
    }
    assert_eq!(refused_at, Some(2));
    let after = snapshot(&s);
    assert_ne!(&after[..6], &before[..6], "tensors 0 and 1 were stepped");
    assert_eq!(&after[6..], &before[6..], "tensors 2 and 3 were not");
}

#[cfg(target_os = "macos")]
mod metal {
    use super::*;
    use ojas_metal::MetalBackend;

    fn metal() -> Option<MetalBackend> {
        match MetalBackend::new(Budget::new(1 << 30)) {
            Ok(m) => Some(m),
            Err(e) if std::env::var_os("OJAS_ALLOW_NO_GPU").is_some() => {
                eprintln!("no Metal device: {e:?}");
                None
            }
            Err(e) => panic!("no Metal device (set OJAS_ALLOW_NO_GPU=1 to skip): {e:?}"),
        }
    }

    fn upload(m: &MetalBackend, s: State) -> State {
        State {
            p: m.upload(&s.p).unwrap(),
            m1: m.upload(&s.m1).unwrap(),
            m2: m.upload(&s.m2).unwrap(),
        }
    }

    fn download(m: &MetalBackend, states: &[State], b: &Budget) -> Vec<State> {
        m.sync().unwrap();
        states
            .iter()
            .map(|s| State {
                p: s.p.to_host(b).unwrap(),
                m1: s.m1.to_host(b).unwrap(),
                m2: s.m2.to_host(b).unwrap(),
            })
            .collect()
    }

    /// On Metal the gradients are device tensors, read back once each for
    /// the scan; a bad one in tensor 2 leaves every device tensor unchanged
    /// bit for bit, and a good step matches the CPU's within f32 rounding.
    #[test]
    fn metal_twin_all_or_nothing_and_close_to_cpu() {
        let Some(m) = metal() else { return };
        let (host, b) = (cpu(), budget());

        let mut on_gpu: Vec<State> = states(&b).into_iter().map(|s| upload(&m, s)).collect();
        let before = snapshot(&download(&m, &on_gpu, &b));
        let g_bad: Vec<Tensor> = with_grad_value(&b, 5, 2, 1, f32::NAN)
            .iter()
            .map(|t| m.upload(t).unwrap())
            .collect();
        let err = adamw_all_or_nothing(&m, &mut entries(&mut on_gpu, &g_bad), 0, &b).unwrap_err();
        assert!(
            matches!(err, OptimError::NonFiniteGrad { entry: 2, .. }),
            "{err}"
        );
        assert_eq!(snapshot(&download(&m, &on_gpu, &b)), before);

        let g_host = grads(&b, 2);
        let g_gpu: Vec<Tensor> = g_host.iter().map(|t| m.upload(t).unwrap()).collect();
        adamw_all_or_nothing(&m, &mut entries(&mut on_gpu, &g_gpu), 0, &b).unwrap();
        let mut on_cpu = states(&b);
        adamw_all_or_nothing(&host, &mut entries(&mut on_cpu, &g_host), 0, &b).unwrap();
        let (gpu, cpu_) = (snapshot(&download(&m, &on_gpu, &b)), snapshot(&on_cpu));
        for (tg, tc) in gpu.iter().zip(&cpu_) {
            for (x, y) in tg.iter().zip(tc) {
                let (x, y) = (f32::from_bits(*x), f32::from_bits(*y));
                assert!((x - y).abs() <= 1e-6 * y.abs().max(1e-3), "{x} vs {y}");
            }
        }
    }
}

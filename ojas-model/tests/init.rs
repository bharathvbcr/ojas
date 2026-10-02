//! G11: init statistics at real size, exact zeros and ones, and order
//! independence (`docs/framework-design.md` §2, §8).

use ojas_core::Budget;
use ojas_model::{init_params, init_values, param_table, Init, ModelSpec, INIT_STD};

const SEED: u64 = 1337;

struct Moments {
    n: f64,
    mean: f64,
    std: f64,
    kurtosis: f64,
}

fn moments(values: &[f32]) -> Moments {
    let n = values.len() as f64;
    let mean = values.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
    let (mut m2, mut m4) = (0.0, 0.0);
    for &v in values {
        let d = f64::from(v) - mean;
        m2 += d * d;
        m4 += d * d * d * d;
    }
    let var = m2 / n;
    Moments {
        n,
        mean,
        std: var.sqrt(),
        kurtosis: (m4 / n) / (var * var),
    }
}

/// Five standard errors for a normal sample of `n` with std `sigma`.
fn check_normal(name: &str, values: &[f32], sigma: f64) {
    let m = moments(values);
    let se_mean = sigma / m.n.sqrt();
    assert!(
        m.mean.abs() < 5.0 * se_mean,
        "{name}: mean {} vs 5 se {}",
        m.mean,
        5.0 * se_mean
    );
    let rel = (m.std / sigma - 1.0).abs();
    let se_std = 1.0 / (2.0 * m.n).sqrt();
    assert!(rel < 5.0 * se_std, "{name}: std {} rel err {rel}", m.std);
    // A uniform with the same std has kurtosis 1.8; a normal has 3.
    let se_kurt = (24.0 / m.n).sqrt();
    assert!(
        (m.kurtosis - 3.0).abs() < 5.0 * se_kurt,
        "{name}: kurtosis {} vs 3 (5 se {})",
        m.kurtosis,
        5.0 * se_kurt
    );
}

#[test]
fn g11_real_size_statistics_and_exact_constants() {
    let spec = ModelSpec::nanolab_124m();
    let table = param_table(&spec).unwrap();
    let params = init_params(&spec, SEED, &Budget::new(4 << 30)).unwrap();
    assert_eq!(params.len(), table.len());
    let mut pooled = Vec::new();
    for (info, t) in table.iter().zip(&params) {
        assert_eq!(t.shape(), info.shape.as_slice(), "{}", info.name);
        let v = t.to_f32_vec().unwrap();
        match info.init {
            Init::Zeros => assert!(
                v.iter().all(|x| x.to_bits() == 0),
                "{}: not exactly +0.0",
                info.name
            ),
            Init::Ones => assert!(
                v.iter().all(|x| x.to_bits() == 1.0f32.to_bits()),
                "{}: not exactly 1.0",
                info.name
            ),
            Init::Normal { std } => {
                assert_eq!(std, INIT_STD);
                check_normal(&info.name, &v, f64::from(std));
                if info.name.starts_with("blocks.") {
                    pooled.extend_from_slice(&v);
                }
            }
        }
    }
    check_normal("every hidden normal init", &pooled, 0.02);
    let emb = params[0].to_f32_vec().unwrap();
    assert_eq!(emb.len(), 50304 * 768);
    check_normal("tok_emb.weight", &emb, 0.02);
}

#[test]
fn g11_streams_do_not_depend_on_order_and_differ_by_name_and_seed() {
    let spec = ModelSpec::tiny();
    let budget = Budget::new(1 << 30);
    let table = param_table(&spec).unwrap();
    let full = init_params(&spec, SEED, &budget).unwrap();
    // Each parameter alone, in reverse order, equals its value in the full init.
    for (info, t) in table.iter().zip(&full).rev() {
        let alone = init_values(info, SEED).unwrap();
        let bits: Vec<u32> = alone.iter().map(|v| v.to_bits()).collect();
        let want: Vec<u32> = t
            .to_f32_vec()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(bits, want, "{}", info.name);
    }
    // At real size too, for one layer deep in the table.
    let big = param_table(&ModelSpec::nanolab_124m()).unwrap();
    let q7 = big
        .iter()
        .find(|p| p.name == "blocks.7.mixer.q_proj.weight")
        .unwrap();
    let small_twin = table
        .iter()
        .find(|p| p.name == "blocks.1.mixer.q_proj.weight")
        .unwrap();
    assert_ne!(
        init_values(q7, SEED).unwrap()[..64],
        init_values(small_twin, SEED).unwrap()[..64],
        "different names drew the same stream"
    );
    // Same shape, different names: uncorrelated streams.
    let q = table
        .iter()
        .find(|p| p.name == "blocks.0.mixer.q_proj.weight")
        .unwrap();
    let k = table
        .iter()
        .find(|p| p.name == "blocks.0.mixer.k_proj.weight")
        .unwrap();
    let (a, b) = (init_values(q, SEED).unwrap(), init_values(k, SEED).unwrap());
    let n = a.len() as f64;
    let dot: f64 = a
        .iter()
        .zip(&b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    let corr = dot / (n * 0.02 * 0.02);
    assert!(corr.abs() < 5.0 / n.sqrt(), "q/k correlation {corr}");
    // Another seed is another stream; the same seed repeats.
    assert_ne!(
        init_values(q, SEED).unwrap(),
        init_values(q, SEED + 1).unwrap()
    );
    assert_eq!(init_values(q, SEED).unwrap(), init_values(q, SEED).unwrap());
}

//! GPU parity. Every test here opens a Metal runtime, so every test is
//! `#[ignore]` and named `gpu_*`: `cargo test` never runs one. Run them in
//! the agreed GPU window, serialized:
//!
//! ```text
//! cargo test --manifest-path <ojas>/ojas-qwen35/Cargo.toml --release --test gpu_parity \
//!     -- --ignored --test-threads=1 gpu_tiny
//! cargo test --manifest-path <ojas>/ojas-qwen35/Cargo.toml --release --test gpu_parity \
//!     -- --ignored --test-threads=1 gpu_real_2b
//! ```
//!
//! The tiny tests use tessl's committed fixture (`tessl/tests/fixtures/qwen35_train`:
//! a random `Qwen3_5ForCausalLM` of the 2B's shape family with transformers'
//! float32 loss and every gradient) and reuse tessl's own bounds for it. The
//! wiring tests compare this crate's provider with tessl driven directly, bit
//! for bit. Bounds were written before the first run.

#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};

use ojas_core::{Budget, Tensor};
use ojas_qwen35::{
    clip_coefficient, tower_tensors, AdamWHyper, BankState, ExternalGrad, GroupSpec, LrRule,
    Numerics, OptimizerPlan, Qwen35Step, Select, Sequence, Snapshot, WdRule, Which,
};
use tessl::gemm::GemmOperands;
use tessl::npy::read_npy;
use tessl::qwen35_model::{Precision, Qwen35Config, Qwen35Model};
use tessl::qwen35_train::{Qwen35Grads, Supervise};

fn tessl_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tessl")
}

fn tiny_dir() -> PathBuf {
    tessl_root().join("tests/fixtures/qwen35_train")
}

fn npy(path: &Path) -> Vec<f64> {
    let a = read_npy(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    if let Ok(s) = a.f32_slice() {
        s.iter().map(|&x| f64::from(x)).collect()
    } else if let Ok(s) = a.f64_slice() {
        s.to_vec()
    } else {
        a.i64_slice().unwrap().iter().map(|&x| x as f64).collect()
    }
}

fn ids(dir: &Path) -> Vec<u32> {
    npy(&dir.join("ids.npy"))
        .iter()
        .map(|&x| x as u32)
        .collect()
}

fn open_tiny(numerics: Numerics) -> Qwen35Step {
    let snap = Snapshot::from_dir(&tiny_dir()).unwrap();
    Qwen35Step::open(&snap, numerics, Budget::new(1 << 30)).unwrap()
}

/// Worst `|got - want| / max|want|`.
fn rel(got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let peak = want.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-30);
    got.iter()
        .zip(want)
        .map(|(&g, w)| (f64::from(g) - w).abs())
        .fold(0.0, f64::max)
        / peak
}

fn bits(t: &Tensor) -> Vec<u32> {
    t.to_f32_vec()
        .unwrap()
        .iter()
        .map(|x| x.to_bits())
        .collect()
}

/// Every position scored against the next token at `1 / (T - 1)`: the causal
/// loss as letter rows. Returns `scale * sum`, the mean cross-entropy.
fn step_all_rows(s: &mut Qwen35Step, ids: &[u32]) -> f64 {
    let n = ids.len() - 1;
    let rows: Vec<u32> = (0..n as u32).collect();
    let pending = s
        .forward(&Sequence {
            ids,
            letter_rows: &rows,
            letter_targets: &ids[1..],
            letter_scale: 1.0 / n as f32,
            span_positions: &[],
        })
        .unwrap();
    let loss = pending.letter_ce_sum() / n as f64;
    s.backward(pending, None).unwrap();
    loss
}

fn reference_files(dir: &Path) -> usize {
    std::fs::read_dir(dir)
        .unwrap()
        .filter(|e| {
            e.as_ref()
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with("grad.")
        })
        .count()
}

/// Every reference gradient `grad.model.<name>.npy` in `dir` against the
/// provider's bank, within `bound` of each parameter's peak.
fn compare_to_fixture(s: &Qwen35Step, dir: &Path, bound: f64) -> f64 {
    let mut worst = 0.0f64;
    let mut seen = 0;
    for g in s.read_table(Which::Gradients).unwrap() {
        let path = dir.join(format!("grad.model.{}.npy", g.name));
        if !path.exists() {
            continue;
        }
        let r = rel(&g.tensor.to_f32_vec().unwrap(), &npy(&path));
        eprintln!("{}: {r:.2e}", g.name);
        assert!(r <= bound, "{}: {r:.3e} > {bound:e}", g.name);
        worst = worst.max(r);
        seen += 1;
    }
    assert_eq!(
        seen,
        reference_files(dir),
        "every reference gradient must be compared"
    );
    worst
}

#[test]
#[ignore]
fn gpu_name_map_is_tessls_parameter_table() {
    let s = open_tiny(Numerics::ExactF32);
    assert_eq!(s.parameter_table(), tower_tensors(s.config()).as_slice());
    assert_eq!(*s.bank_state(), BankState::Empty);
    assert_eq!(s.step_count(), 0);
}

/// tessl's `tiny_step_matches_transformers_autograd` bounds (loss 1e-5
/// relative, each gradient 1e-4 of its peak), through this crate's provider:
/// letter rows at every position, read back in transformers' layout. A rerun
/// is the same bits.
#[test]
#[ignore]
fn gpu_tiny_rows_step_matches_the_transformers_fixture() {
    let dir = tiny_dir();
    let ids = ids(&dir);
    let mut s = open_tiny(Numerics::ExactF32);
    let loss = step_all_rows(&mut s, &ids);
    let want = npy(&dir.join("loss.npy"))[0];
    eprintln!("loss {loss:.8} vs transformers {want:.8}");
    assert!((loss - want).abs() <= 1e-5 * want.abs(), "{loss} vs {want}");
    let worst = compare_to_fixture(&s, &dir, 1e-4);
    eprintln!("worst gradient {worst:.2e}");
    let first: Vec<Vec<u32>> = s
        .read_table(Which::Gradients)
        .unwrap()
        .iter()
        .map(|g| bits(&g.tensor))
        .collect();
    s.discard_gradients();
    let again = step_all_rows(&mut s, &ids);
    assert_eq!(again.to_bits(), loss.to_bits());
    let second: Vec<Vec<u32>> = s
        .read_table(Which::Gradients)
        .unwrap()
        .iter()
        .map(|g| bits(&g.tensor))
        .collect();
    assert_eq!(first, second, "a rerun changed the gradients");
}

/// tessl's bf16-operand bounds against the same float32 reference (loss
/// 2^-8, gradients 2^-5), and not the exact step's bits.
#[test]
#[ignore]
fn gpu_tiny_bf16_operands_stay_near_the_fixture() {
    let dir = tiny_dir();
    let ids = ids(&dir);
    let mut s = open_tiny(Numerics::Bf16Operands);
    let loss = step_all_rows(&mut s, &ids);
    let want = npy(&dir.join("loss.npy"))[0];
    eprintln!("bf16 operands: loss {loss:.8} vs {want:.8}");
    assert!((loss - want).abs() <= 2f64.powi(-8) * want.abs());
    compare_to_fixture(&s, &dir, 2f64.powi(-5));
    let mut exact = open_tiny(Numerics::ExactF32);
    let exact_loss = step_all_rows(&mut exact, &ids);
    let a: Vec<Vec<u32>> = s
        .read_table(Which::Gradients)
        .unwrap()
        .iter()
        .map(|g| bits(&g.tensor))
        .collect();
    let b: Vec<Vec<u32>> = exact
        .read_table(Which::Gradients)
        .unwrap()
        .iter()
        .map(|g| bits(&g.tensor))
        .collect();
    assert!(
        loss.to_bits() != exact_loss.to_bits() || a != b,
        "the bf16-operand step is the exact step's bits: nothing was rounded"
    );
}

/// One sequence of the wiring test: its letter rows, its span positions
/// (repeats allowed), and the distinct positions its external gradient
/// lands on.
struct WiringCase {
    rows: &'static [u32],
    spans: &'static [u32],
    dpos: &'static [u32],
}

/// The provider's hidden rows, letter loss and bank are tessl's own, bit for
/// bit, when tessl is driven directly with the same sequence, the same
/// external gradient and the same accumulation: the wiring adds nothing.
#[test]
#[ignore]
fn gpu_tiny_hidden_rows_external_grad_and_accumulation_are_tessls_own() {
    let dir = tiny_dir();
    let ids = ids(&dir);
    let h = 64usize;
    let budget = Budget::new(1 << 30);
    let mut s = open_tiny(Numerics::ExactF32);

    // tessl, directly.
    let rt = tessl::GpuRuntime::new().unwrap();
    let st = tessl::safetensors::SafeTensors::open(&dir.join("model.safetensors")).unwrap();
    let cfg = Qwen35Config::from_config_file(&dir.join("config.json")).unwrap();
    let model = Qwen35Model::load(&rt, &st, "model.", cfg, Precision::F32).unwrap();
    let bank = Qwen35Grads::zeros_like(&model).unwrap();

    let cases = [
        WiringCase {
            rows: &[1, 4, 20],
            spans: &[2, 9, 9, 30],
            dpos: &[2, 9, 30],
        },
        WiringCase {
            rows: &[3, 7],
            spans: &[0, 5],
            dpos: &[0, 5],
        },
    ];
    let scale = &0.25f32;
    for (k, WiringCase { rows, spans, dpos }) in cases.iter().enumerate() {
        let targets: Vec<u32> = rows.iter().map(|&p| ids[p as usize + 1]).collect();
        let len = if k == 0 { ids.len() } else { ids.len() - 4 };
        let seq_ids = &ids[..len];
        let pending = s
            .forward(&Sequence {
                ids: seq_ids,
                letter_rows: rows,
                letter_targets: &targets,
                letter_scale: *scale,
                span_positions: spans,
            })
            .unwrap();
        let p = model
            .train_forward(
                seq_ids,
                GemmOperands::ExactF32,
                Supervise::Rows {
                    positions: rows,
                    targets: &targets,
                    scale: *scale,
                },
            )
            .unwrap();
        assert_eq!(
            pending.letter_ce_sum().to_bits(),
            p.loss().to_bits(),
            "case {k}: letter loss"
        );
        let out = rt.alloc_tensor_f32(&[spans.len(), h]).unwrap();
        p.hidden(spans, &out).unwrap();
        let theirs: Vec<u32> = out
            .read_f32()
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect();
        assert_eq!(bits(pending.hidden()), theirs, "case {k}: hidden rows");

        let dh: Vec<f32> = (0..dpos.len() * h)
            .map(|i| ((i % 11) as f32 - 5.0) * 1e-2 + k as f32 * 1e-3)
            .collect();
        let dh_ojas = Tensor::from_f32(&dh, &[dpos.len(), h], &budget).unwrap();
        s.backward(
            pending,
            Some(ExternalGrad {
                positions: dpos,
                dh: &dh_ojas,
            }),
        )
        .unwrap();
        let dh_tessl = rt.alloc_tensor_f32(&[dpos.len(), h]).unwrap();
        dh_tessl.write_f32(&dh).unwrap();
        model
            .train_backward_into(p, Some((dpos, &dh_tessl)), &bank, k > 0)
            .unwrap();
        assert_eq!(*s.bank_state(), BankState::Holds(k + 1));

        let table = model.parameter_table().unwrap();
        let ts: Vec<tessl::Tensor> = table
            .iter()
            .map(|q| rt.alloc_tensor_f32(&q.storage_shape()).unwrap())
            .collect();
        model.read_gradients(&bank, &ts).unwrap();
        let ours = s.read_table(Which::Gradients).unwrap();
        for ((q, t), o) in table.iter().zip(&ts).zip(&ours) {
            assert_eq!(q.name, o.name);
            let v = t.read_f32().unwrap();
            let v = if q.transposed {
                ojas_qwen35::transpose_2d(&v, q.shape[1], q.shape[0]).unwrap()
            } else {
                v
            };
            let theirs: Vec<u32> = v.iter().map(|x| x.to_bits()).collect();
            assert_eq!(
                bits(&o.tensor),
                theirs,
                "case {k}: {} differs from tessl's bank",
                q.name
            );
        }
    }
}

fn no_decay() -> Select {
    Select::AnyOf(vec![
        Select::Contains("bias".into()),
        Select::Contains("layernorm".into()),
        Select::Segment("norm".into()),
        Select::SegmentSuffix("_norm".into()),
    ])
}

fn plan(s: &Qwen35Step, lr_scale: f64, wd: f32) -> OptimizerPlan {
    OptimizerPlan::build(
        s.parameter_table(),
        &GroupSpec {
            lr: vec![LrRule {
                label: "all".into(),
                select: Select::All,
                lr_scale,
            }],
            weight_decay: vec![
                WdRule {
                    label: "no_decay".into(),
                    select: no_decay(),
                    weight_decay: 0.0,
                },
                WdRule {
                    label: "decay".into(),
                    select: Select::Not(Box::new(no_decay())),
                    weight_decay: wd,
                },
            ],
        },
    )
    .unwrap()
}

/// One AdamW step through the provider against torch's single-tensor AdamW
/// formed in f64 on the host from the same f32 inputs (decoupled decay, the
/// moments, bias correction outside the root, the clip scale on the
/// gradient). Bounds before the first run: each moment within 1e-6 of its
/// tensor's peak; each parameter within 2e-7 x max(1, its peak). The norm
/// tessl reports is the host's sum of squares to 1e-5. Then the refusals an
/// open provider makes.
#[test]
#[ignore]
fn gpu_tiny_adamw_step_is_torchs_update() {
    let dir = tiny_dir();
    let ids = ids(&dir);
    let mut s = open_tiny(Numerics::ExactF32);
    let before = s.read_table(Which::Parameters).unwrap();
    step_all_rows(&mut s, &ids);
    let grads = s.read_table(Which::Gradients).unwrap();
    let sq = s.grad_sq_norm().unwrap();
    let host_sq: f64 = grads
        .iter()
        .flat_map(|g| g.tensor.to_f32_vec().unwrap())
        .map(|x| f64::from(x) * f64::from(x))
        .sum();
    assert!((sq - host_sq).abs() <= 1e-5 * host_sq, "{sq} vs {host_sq}");
    let clip = clip_coefficient(0.5, sq, 0.0).unwrap();
    eprintln!("grad norm {:.6}, clip {clip}", sq.sqrt());
    let hyper = AdamWHyper {
        lr: 1e-3,
        beta1: 0.9,
        beta2: 0.95,
        eps: 1e-8,
        grad_scale: clip,
    };
    let p = plan(&s, 1.0, 0.1);
    assert_eq!(s.adamw_step(&hyper, &p).unwrap(), 1);
    assert_eq!(*s.bank_state(), BankState::Empty);
    let after = s.read_table(Which::Parameters).unwrap();
    let m = s.read_table(Which::ExpAvg).unwrap();
    let v = s.read_table(Which::ExpAvgSq).unwrap();
    let (bc1, bc2) = (1.0 - hyper.beta1, 1.0 - hyper.beta2);
    for i in 0..before.len() {
        let wd = f64::from(p.weight_decay()[i]);
        let (p0, g) = (
            before[i].tensor.to_f32_vec().unwrap(),
            grads[i].tensor.to_f32_vec().unwrap(),
        );
        let (p1, m1, v1) = (
            after[i].tensor.to_f32_vec().unwrap(),
            m[i].tensor.to_f32_vec().unwrap(),
            v[i].tensor.to_f32_vec().unwrap(),
        );
        let (mut want_p, mut want_m, mut want_v) = (Vec::new(), Vec::new(), Vec::new());
        for j in 0..p0.len() {
            let gs = f64::from(g[j] * clip);
            let mm = (1.0 - hyper.beta1) * gs;
            let vv = (1.0 - hyper.beta2) * gs * gs;
            let denom = vv.sqrt() / bc2.sqrt() + hyper.eps;
            want_p.push(f64::from(p0[j]) * (1.0 - hyper.lr * wd) - hyper.lr / bc1 * mm / denom);
            want_m.push(mm);
            want_v.push(vv);
        }
        let name = &before[i].name;
        assert!(
            rel(&m1, &want_m) <= 1e-6,
            "{name}: first moment {:.3e}",
            rel(&m1, &want_m)
        );
        assert!(
            rel(&v1, &want_v) <= 1e-6,
            "{name}: second moment {:.3e}",
            rel(&v1, &want_v)
        );
        let scale = p0.iter().fold(1.0f64, |a, &x| a.max(f64::from(x).abs()));
        let worst = p1
            .iter()
            .zip(&want_p)
            .map(|(&a, b)| (f64::from(a) - b).abs())
            .fold(0.0, f64::max);
        assert!(
            worst <= 2e-7 * scale,
            "{name}: parameter off by {worst:.3e}"
        );
    }

    // Refusals on an open provider, each before device work.
    let e = s.adamw_step(&hyper, &p).unwrap_err().to_string();
    assert!(e.contains("holds no sequence"), "{e}");
    assert!(s.grad_sq_norm().is_err());
    let rows: Vec<u32> = vec![2];
    let stale = s
        .forward(&Sequence {
            ids: &ids,
            letter_rows: &rows,
            letter_targets: &ids[3..4],
            letter_scale: 1.0,
            span_positions: &[],
        })
        .unwrap();
    step_all_rows(&mut s, &ids);
    let e = s
        .adamw_step(&hyper, &plan(&s, 0.1, 0.1))
        .unwrap_err()
        .to_string();
    assert!(e.contains("lappi-train-lrscale-mrope"), "{e}");
    assert_eq!(s.step_count(), 1, "a refused step moved the count");
    s.adamw_step(&hyper, &p).unwrap();
    let e = s.backward(stale, None).unwrap_err().to_string();
    assert!(e.contains("weights changed since this forward"), "{e}");
}

fn scratch(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("ojas-qwen35-gpu-{name}-{}", std::process::id()));
    if d.exists() {
        std::fs::remove_dir_all(&d).unwrap();
    }
    d
}

/// Save after a step, load into a fresh provider: masters, both moments and
/// the step count are the same bits, and the next step from each is the
/// same bits. An existing directory is never overwritten, and a symbolic
/// link is never written through.
#[test]
#[ignore]
fn gpu_tiny_state_round_trip_is_bit_exact() {
    let dir = tiny_dir();
    let ids = ids(&dir);
    let hyper = AdamWHyper {
        lr: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        grad_scale: 1.0,
    };
    let mut a = open_tiny(Numerics::ExactF32);
    step_all_rows(&mut a, &ids);
    let pa = plan(&a, 1.0, 0.01);
    a.adamw_step(&hyper, &pa).unwrap();
    let state = scratch("state");
    a.save_state(&state).unwrap();
    let e = a.save_state(&state).unwrap_err().to_string();
    assert!(e.contains("never overwritten"), "{e}");
    let (link, target) = (scratch("state-link"), scratch("state-link-target"));
    std::fs::create_dir_all(&target).unwrap();
    std::os::unix::fs::symlink(&target, &link).unwrap();
    let e = a.save_state(&link).unwrap_err().to_string();
    assert!(e.contains("symbolic link"), "{e}");
    assert_eq!(std::fs::read_dir(&target).unwrap().count(), 0);

    let mut b = open_tiny(Numerics::ExactF32);
    b.load_state(&state).unwrap();
    assert_eq!(b.step_count(), 1);
    for which in [Which::Parameters, Which::ExpAvg, Which::ExpAvgSq] {
        let (x, y) = (a.read_table(which).unwrap(), b.read_table(which).unwrap());
        for (p, q) in x.iter().zip(&y) {
            assert_eq!(bits(&p.tensor), bits(&q.tensor), "{which:?} {}", p.name);
        }
    }
    let pb = plan(&b, 1.0, 0.01);
    for (s, p) in [(&mut a, &pa), (&mut b, &pb)] {
        step_all_rows(s, &ids[..ids.len() - 2]);
        s.adamw_step(&hyper, p).unwrap();
    }
    let (x, y) = (
        a.read_table(Which::Parameters).unwrap(),
        b.read_table(Which::Parameters).unwrap(),
    );
    for (p, q) in x.iter().zip(&y) {
        assert_eq!(
            bits(&p.tensor),
            bits(&q.tensor),
            "after the next step: {}",
            p.name
        );
    }
}

fn snapshot_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("QWEN35_2B_SNAPSHOT") {
        return PathBuf::from(d);
    }
    let root = PathBuf::from(std::env::var_os("HOME").expect("HOME"))
        .join(".cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots");
    let dirs: Vec<PathBuf> = std::fs::read_dir(&root)
        .unwrap_or_else(|e| panic!("{}: {e}", root.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| p.join("config.json").is_file())
        .collect();
    assert_eq!(
        dirs.len(),
        1,
        "expected one snapshot under {}",
        root.display()
    );
    dirs[0].clone()
}

/// The real Qwen3.5-2B-Base, opened from its snapshot directory (the
/// snapshot's own `config.json`, MRoPE fields included), one step on
/// `make_train_fixture.py 2b`'s sequence against its transformers float32
/// reference at tessl's own bounds for the real 2B
/// (`real_2b_step_matches_transformers`: loss 1e-4, each compared gradient
/// 1e-2 of its peak), then the clip coefficient and one AdamW step.
///
/// Needs `tessl/target/qwen35_train_ref` (or `QWEN35_TRAIN_REF_DIR`) and the
/// snapshot (`QWEN35_2B_SNAPSHOT` or the HF cache). Holds about 40 GB of
/// unified memory: f32 weights, bank, two moments, and one table staged for
/// read-back.
#[test]
#[ignore]
fn gpu_real_2b_single_step_matches_the_transformers_reference() {
    let ref_dir = std::env::var_os("QWEN35_TRAIN_REF_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| tessl_root().join("target/qwen35_train_ref"));
    let snap = Snapshot::from_dir(&snapshot_dir()).unwrap();
    let mut s = Qwen35Step::open(&snap, Numerics::ExactF32, Budget::new(16 << 30)).unwrap();
    let ids = ids(&ref_dir);
    let loss = step_all_rows(&mut s, &ids);
    let want = npy(&ref_dir.join("loss.npy"))[0];
    let r = (loss - want).abs() / want.abs();
    eprintln!(
        "2B: loss {loss:.8} vs transformers {want:.8} (rel {r:.2e}) over {} tokens",
        ids.len()
    );
    assert!(r <= 1e-4, "{loss} vs {want}");

    let names: Vec<String> = s
        .parameter_table()
        .iter()
        .map(|t| t.name.clone())
        .filter(|n| ref_dir.join(format!("grad.model.{n}.npy")).exists())
        .collect();
    let refs: Vec<&str> = names
        .iter()
        .map(String::as_str)
        .chain(["embed_tokens.weight"])
        .collect();
    let got = s.read_entries(Which::Gradients, &refs).unwrap();
    let mut seen = 0;
    for g in &got {
        let r = if g.name == "embed_tokens.weight" {
            let rows = npy(&ref_dir.join("embed_rows.npy"));
            let want = npy(&ref_dir.join("grad.model.embed_tokens.weight.rows.npy"));
            let all = g.tensor.to_f32_vec().unwrap();
            let h = s.config().hidden as usize;
            let kept: Vec<f32> = rows
                .iter()
                .flat_map(|&r| all[r as usize * h..][..h].to_vec())
                .collect();
            rel(&kept, &want)
        } else {
            rel(
                &g.tensor.to_f32_vec().unwrap(),
                &npy(&ref_dir.join(format!("grad.model.{}.npy", g.name))),
            )
        };
        eprintln!("{}: {r:.2e}", g.name);
        assert!(r <= 1e-2, "{}: {r:.3e}", g.name);
        seen += 1;
    }
    assert_eq!(
        seen,
        reference_files(&ref_dir),
        "every reference gradient must be compared"
    );

    let sq = s.grad_sq_norm().unwrap();
    assert!(sq.is_finite() && sq > 0.0);
    let clip = clip_coefficient(1.0, sq, 0.0).unwrap();
    eprintln!("2B: grad norm {:.6}, clip {clip}", sq.sqrt());
    let p = plan(&s, 1.0, 0.01);
    let hyper = AdamWHyper {
        lr: 1e-5,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        grad_scale: clip,
    };
    assert_eq!(s.adamw_step(&hyper, &p).unwrap(), 1);
    assert_eq!(*s.bank_state(), BankState::Empty);
}

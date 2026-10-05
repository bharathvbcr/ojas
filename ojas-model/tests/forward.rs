//! The block and the forward: `Eval` against `Tape` bit for bit under
//! `Numerics::Exact`, an f64 gradcheck of one block through the tape, the
//! RoPE table, the cached-attention refusal on a tape, and the safetensors
//! loader's name rules.

use ojas_autograd::{central_diff, gradients_match, Tape};
use ojas_core::{Backend, Budget, CeChunk, Numerics, OjasError, Tensor};
use ojas_cpu::CpuBackend;
use ojas_io::{encode_safetensors, SafeTensors, StDtype, TensorOut};
use ojas_model::{
    bind, block, forward_hidden, forward_logits, forward_loss, init_params, load_model,
    load_params, load_spec, param_table, BlockParams, Eval, Graph, Init, ModelSpec, Rope,
    COMPILED_PREFIX, LM_HEAD, SPEC_METADATA_KEY,
};

fn exact() -> CpuBackend {
    CpuBackend::new(Budget::new(1 << 30)).with_numerics(Numerics::Exact)
}

/// SplitMix64 uniform in `[-1, 1)`, exact in f32.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        f64::from(((z >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0)
    }

    fn vec(&mut self, n: usize, scale: f64, offset: f64) -> Vec<f64> {
        (0..n)
            .map(|_| f64::from((offset + scale * self.next()) as f32))
            .collect()
    }
}

/// Every parameter of `spec` with non-degenerate values: matrices in
/// `[-0.3, 0.3)`, norm weights in `[0.8, 1.2)`, biases and lambdas in
/// `[-0.5, 0.5)`. The zero inits would hide the attention path.
fn random_params(spec: &ModelSpec, seed: u64, budget: &Budget) -> Vec<Tensor> {
    let mut rng = Rng(seed);
    param_table(spec)
        .unwrap()
        .iter()
        .map(|info| {
            let n = info.numel();
            let v = match (info.init, info.shape.len()) {
                (Init::Ones, _) => rng.vec(n, 0.2, 1.0),
                (_, 2) => rng.vec(n, 0.3, 0.0),
                _ => rng.vec(n, 0.5, 0.0),
            };
            let f: Vec<f32> = v.iter().map(|&x| x as f32).collect();
            Tensor::from_f32(&f, &info.shape, budget).unwrap()
        })
        .collect()
}

fn ids(spec: &ModelSpec, batch: usize, seq: usize, salt: u32, budget: &Budget) -> Tensor {
    let v: Vec<u32> = (0..batch * seq)
        .map(|i| (i as u32 * 37 + salt * 11 + 5) % spec.vocab as u32)
        .collect();
    Tensor::from_u32(&v, &[batch, seq], budget).unwrap()
}

fn tensor_bits(t: &Tensor) -> Vec<u32> {
    t.to_f32_vec()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

const CHUNK: CeChunk = CeChunk { rows: 5, cols: 37 };

#[test]
fn eval_and_tape_forward_are_bit_identical_under_exact() {
    let spec = ModelSpec::tiny();
    let cpu = exact();
    let budget = cpu.budget().clone();
    let params = random_params(&spec, 7, &budget);
    let (b, t) = (2, 9);
    let rope = Rope::new(&spec, t, &budget).unwrap();
    let x = ids(&spec, b, t, 1, &budget);
    let y = ids(&spec, b, t, 2, &budget);

    let mut eval = Eval::new(cpu.clone());
    let ep = bind(&mut eval, &spec, &params).unwrap();
    let e_hidden = forward_hidden(&mut eval, &spec, &ep, &x, &rope).unwrap();
    let e_logits = forward_logits(&mut eval, &spec, &ep, &x, &rope).unwrap();
    let e_loss = forward_loss(&mut eval, &spec, &ep, &x, &y, &rope, None, CHUNK).unwrap();

    let mut tape = Tape::new(cpu.clone());
    let tp = bind(&mut tape, &spec, &params).unwrap();
    let t_hidden = forward_hidden(&mut tape, &spec, &tp, &x, &rope).unwrap();
    let t_logits = forward_logits(&mut tape, &spec, &tp, &x, &rope).unwrap();
    let t_loss = forward_loss(&mut tape, &spec, &tp, &x, &y, &rope, None, CHUNK).unwrap();

    assert_eq!(e_hidden.shape(), &[b, t, spec.n_embd]);
    assert_eq!(e_logits.shape(), &[b, t, spec.vocab]);
    assert_eq!(e_loss.shape(), &[] as &[usize]);
    assert_eq!(
        tensor_bits(&e_hidden),
        tensor_bits(tape.value(t_hidden).unwrap())
    );
    assert_eq!(
        tensor_bits(&e_logits),
        tensor_bits(tape.value(t_logits).unwrap())
    );
    // The tape asks the fused CE for gradients and Eval does not; the loss
    // must not depend on that.
    assert_eq!(
        tensor_bits(&e_loss),
        tensor_bits(tape.value(t_loss).unwrap())
    );
    let loss = e_loss.to_f32_vec().unwrap()[0];
    assert!(loss.is_finite() && loss > 0.0);
    // The fused loss equals the unfused composition over the same logits.
    let flat = Tensor::from_f32(
        &e_logits.to_f32_vec().unwrap(),
        &[b * t, spec.vocab],
        &budget,
    )
    .unwrap();
    let y_flat = Tensor::from_u32(&y.to_u32_vec().unwrap(), &[b * t], &budget).unwrap();
    let unfused = cpu
        .cross_entropy_mean_forward(&flat, &y_flat, None)
        .unwrap();
    let unfused = unfused.to_f32_vec().unwrap()[0];
    assert!(
        (unfused - loss).abs() <= 1e-6 * loss.abs(),
        "fused {loss} unfused {unfused}"
    );
}

#[test]
fn every_layer_changes_the_output() {
    // A forward that skipped a block, or let layer 0 blend, would leave the
    // hidden state unchanged by that layer's weights.
    let spec = ModelSpec::tiny();
    let cpu = exact();
    let budget = cpu.budget().clone();
    let params = random_params(&spec, 9, &budget);
    let rope = Rope::new(&spec, 4, &budget).unwrap();
    let x = ids(&spec, 1, 4, 3, &budget);
    let hidden = |p: &[Tensor]| {
        let mut eval = Eval::new(cpu.clone());
        let bound = bind(&mut eval, &spec, p).unwrap();
        tensor_bits(&forward_hidden(&mut eval, &spec, &bound, &x, &rope).unwrap())
    };
    let base = hidden(&params);
    let table = param_table(&spec).unwrap();
    for (i, info) in table.iter().enumerate() {
        let mut p = params.clone();
        let mut v = p[i].to_f32_vec().unwrap();
        v.iter_mut().for_each(|x| *x += 0.25);
        p[i] = Tensor::from_f32(&v, &info.shape, &budget).unwrap();
        let changed = hidden(&p) != base;
        // Layer 0's vr_lambda is never read: there is nothing to blend.
        let read = info.name != "blocks.0.mixer.vr_lambda";
        assert_eq!(changed, read, "{}", info.name);
    }
}

#[test]
fn every_later_layer_blends_with_layer_zeros_raw_value() {
    // Three layers, so "v0 is the previous layer's v" and "v0 is layer
    // 0's v" give different results at layer 2.
    let spec = ModelSpec {
        n_layer: 3,
        ..ModelSpec::tiny()
    };
    let cpu = exact();
    let budget = cpu.budget().clone();
    let params = random_params(&spec, 13, &budget);
    let rope = Rope::new(&spec, 6, &budget).unwrap();
    let x = ids(&spec, 2, 6, 4, &budget);
    let mut eval = Eval::new(cpu);
    let p = bind(&mut eval, &spec, &params).unwrap();
    let got = forward_hidden(&mut eval, &spec, &p, &x, &rope).unwrap();
    let mut h = eval.embedding(&p.tok_emb, &x).unwrap();
    let first = block(&mut eval, &spec, &p.blocks[0], &h, None, &rope, 2).unwrap();
    let v0 = first.raw_v;
    h = first.x;
    for blk in &p.blocks[1..] {
        h = block(&mut eval, &spec, blk, &h, Some(&v0), &rope, 2)
            .unwrap()
            .x;
    }
    let want = eval.rms_norm(&h, &p.norm_f, spec.eps()).unwrap();
    assert_eq!(tensor_bits(&got), tensor_bits(&want));
}

#[test]
fn a_tape_refuses_cached_attention_and_records_nothing() {
    let cpu = exact();
    let budget = cpu.budget().clone();
    let mut tape = Tape::new(cpu);
    let q = Tensor::from_f32(&[0.5; 8], &[1, 1, 2, 4], &budget).unwrap();
    let cache = Tensor::from_f32(&[0.5; 16], &[1, 2, 2, 4], &budget).unwrap();
    let qv = Graph::param(&mut tape, &q).unwrap();
    let cv = Graph::param(&mut tape, &cache).unwrap();
    match tape.cached_attn(&qv, &cv, &cv, 1) {
        Err(OjasError::Unsupported { .. }) => {}
        other => panic!("expected Unsupported, got {other:?}"),
    }
    // Only the two leaves are on the tape.
    let next = Graph::param(&mut tape, &q).unwrap();
    assert_eq!(next.0, 2);
}

#[test]
fn eval_cached_attention_equals_causal_attention_on_a_full_cache() {
    // The decode op Eval exposes for item 11: with kv_len == Tq it is the
    // block's causal SDPA after the layout permute.
    let cpu = exact();
    let budget = cpu.budget().clone();
    let mut rng = Rng(4);
    let (t, h, d) = (5, 2, 4);
    let mk = |rng: &mut Rng| {
        let v: Vec<f32> = rng
            .vec(t * h * d, 1.0, 0.0)
            .iter()
            .map(|&x| x as f32)
            .collect();
        Tensor::from_f32(&v, &[1, t, h, d], &budget).unwrap()
    };
    let (q, k, v) = (mk(&mut rng), mk(&mut rng), mk(&mut rng));
    let mut eval = Eval::new(cpu);
    let cached = eval.cached_attn(&q, &k, &v, t).unwrap();
    let swap = [0, 2, 1, 3];
    let (qh, kh, vh) = (
        eval.permute(&q, &swap).unwrap(),
        eval.permute(&k, &swap).unwrap(),
        eval.permute(&v, &swap).unwrap(),
    );
    let y = eval.sdpa(&qh, &kh, &vh).unwrap();
    let y = eval.permute(&y, &swap).unwrap();
    let (a, b) = (cached.to_f32_vec().unwrap(), y.to_f32_vec().unwrap());
    for (x, y) in a.iter().zip(&b) {
        assert!((x - y).abs() <= 1e-6, "cached {x} causal {y}");
    }
}

#[test]
fn rope_rows_are_the_nanolab_table_at_any_offset() {
    let spec = ModelSpec::tiny();
    let budget = Budget::new(1 << 20);
    let full = Rope::new(&spec, 12, &budget).unwrap();
    let tail = Rope::rows(&spec, 5, 7, &budget).unwrap();
    let dim = spec.head_dim;
    let (fc, tc) = (
        full.cos.to_f32_vec().unwrap(),
        tail.cos.to_f32_vec().unwrap(),
    );
    let (fs, ts) = (
        full.sin.to_f32_vec().unwrap(),
        tail.sin.to_f32_vec().unwrap(),
    );
    assert_eq!(f32s(&fc[5 * dim..]), f32s(&tc));
    assert_eq!(f32s(&fs[5 * dim..]), f32s(&ts));
    for pos in 0..12 {
        for i in 0..dim / 2 {
            let angle = pos as f64 * 10000f64.powf(-((2 * i) as f64) / dim as f64);
            for slot in [i, i + dim / 2] {
                assert_eq!(fc[pos * dim + slot], angle.cos() as f32);
                assert_eq!(fs[pos * dim + slot], angle.sin() as f32);
            }
        }
    }
}

fn f32s(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

// --------------------------------------------------------------- gradcheck

const B: usize = 2;
const T: usize = 3;
const DM: usize = 8;
const NH: usize = 2;
const HD: usize = 4;
const HID: usize = 6;

fn small_spec() -> ModelSpec {
    ModelSpec {
        n_layer: 1,
        n_embd: DM,
        n_head: NH,
        n_kv_head: NH,
        head_dim: HD,
        hidden: HID,
        max_seq: T,
        vocab: 16,
        rope_base: 10000.0,
        rms_eps: 1e-6,
        tie_embeddings: true,
    }
}

/// Block inputs in f64: index 0 is `x`, 1 is `v0`, then the 14 block
/// parameters in field order, then the loss weight `r`.
fn shapes() -> Vec<Vec<usize>> {
    let q = NH * HD;
    vec![
        vec![B, T, DM],
        vec![B, T, NH, HD],
        vec![DM],
        vec![q, DM],
        vec![q, DM],
        vec![q, DM],
        vec![DM, q],
        vec![HD],
        vec![HD],
        vec![NH, DM],
        vec![NH],
        vec![1],
        vec![DM],
        vec![HID, DM],
        vec![HID, DM],
        vec![DM, HID],
        vec![B, T, DM],
    ]
}

fn sig(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// RMSNorm of each `dim`-row of `x` times `w`.
fn rms(x: &[f64], w: &[f64], dim: usize) -> Vec<f64> {
    let mut y = vec![0.0; x.len()];
    for (row, out) in x.chunks(dim).zip(y.chunks_mut(dim)) {
        let ms = row.iter().map(|v| v * v).sum::<f64>() / dim as f64;
        let inv = 1.0 / (ms + 1e-6).sqrt();
        for i in 0..dim {
            out[i] = row[i] * inv * w[i];
        }
    }
    y
}

fn lin(x: &[f64], w: &[f64], kin: usize, nout: usize) -> Vec<f64> {
    let rows = x.len() / kin;
    let mut y = vec![0.0; rows * nout];
    for r in 0..rows {
        for c in 0..nout {
            y[r * nout + c] = (0..kin).map(|i| x[r * kin + i] * w[c * kin + i]).sum();
        }
    }
    y
}

fn at(b: usize, t: usize, h: usize, d: usize) -> usize {
    ((b * T + t) * NH + h) * HD + d
}

/// The nanolab block in f64 (`mixers.py` Attention.forward with a `v0`,
/// `model.py` Block.forward), then `sum(out * r)`.
fn block_ref(p: &[Vec<f64>], cos: &[f64], sin: &[f64]) -> f64 {
    let (x, v0) = (&p[0], &p[1]);
    let (n1, wq, wk, wv, wo) = (&p[2], &p[3], &p[4], &p[5], &p[6]);
    let (qn, kn, gw, gb, lam) = (&p[7], &p[8], &p[9], &p[10], p[11][0]);
    let (n2, wg, wu, wd, r) = (&p[12], &p[13], &p[14], &p[15], &p[16]);
    let q_w = NH * HD;
    let h = rms(x, n1, DM);
    let q = rms(&lin(&h, wq, DM, q_w), qn, HD);
    let k = rms(&lin(&h, wk, DM, q_w), kn, HD);
    let raw_v = lin(&h, wv, DM, q_w);
    let rope = |x: &[f64]| {
        let mut y = vec![0.0; x.len()];
        let half = HD / 2;
        for b in 0..B {
            for t in 0..T {
                for hh in 0..NH {
                    for i in 0..half {
                        let lo = x[at(b, t, hh, i)];
                        let hi = x[at(b, t, hh, i + half)];
                        y[at(b, t, hh, i)] = lo * cos[t * HD + i] - hi * sin[t * HD + i];
                        y[at(b, t, hh, i + half)] =
                            hi * cos[t * HD + i + half] + lo * sin[t * HD + i + half];
                    }
                }
            }
        }
        y
    };
    let (q, k) = (rope(&q), rope(&k));
    let s = sig(lam);
    let v: Vec<f64> = raw_v
        .iter()
        .zip(v0.iter())
        .map(|(a, b)| (1.0 - s) * a + s * b)
        .collect();
    let scale = 1.0 / (HD as f64).sqrt();
    let mut y = vec![0.0; q.len()];
    for b in 0..B {
        for hh in 0..NH {
            for t in 0..T {
                let scores: Vec<f64> = (0..=t)
                    .map(|j| {
                        (0..HD)
                            .map(|d| q[at(b, t, hh, d)] * k[at(b, j, hh, d)])
                            .sum::<f64>()
                            * scale
                    })
                    .collect();
                let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
                let z: f64 = e.iter().sum();
                let mut zg = gb[hh];
                for c in 0..DM {
                    zg += h[(b * T + t) * DM + c] * gw[hh * DM + c];
                }
                let g = sig(zg);
                for d in 0..HD {
                    let a: f64 = (0..=t).map(|j| e[j] / z * v[at(b, j, hh, d)]).sum();
                    y[at(b, t, hh, d)] = a * g;
                }
            }
        }
    }
    let o = lin(&y, wo, q_w, DM);
    let x1: Vec<f64> = x.iter().zip(&o).map(|(a, b)| a + b).collect();
    let h2 = rms(&x1, n2, DM);
    let a = lin(&h2, wg, DM, HID);
    let u = lin(&h2, wu, DM, HID);
    let m: Vec<f64> = a.iter().zip(&u).map(|(a, u)| a * sig(*a) * u).collect();
    let dn = lin(&m, wd, HID, DM);
    x1.iter()
        .zip(&dn)
        .zip(r.iter())
        .map(|((a, b), r)| (a + b) * r)
        .sum()
}

#[test]
fn one_block_gradients_match_f64_central_differences() {
    let spec = small_spec();
    let cpu = exact();
    let budget = cpu.budget().clone();
    let mut rng = Rng(21);
    let inputs: Vec<Vec<f64>> = shapes()
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let n: usize = s.iter().product();
            match i {
                2 | 7 | 8 | 12 => rng.vec(n, 0.2, 1.0),
                11 => vec![0.3],
                _ => rng.vec(n, 0.5, 0.0),
            }
        })
        .collect();
    let rope = Rope::new(&spec, T, &budget).unwrap();
    let cos: Vec<f64> = rope
        .cos
        .to_f32_vec()
        .unwrap()
        .iter()
        .map(|&v| f64::from(v))
        .collect();
    let sin: Vec<f64> = rope
        .sin
        .to_f32_vec()
        .unwrap()
        .iter()
        .map(|&v| f64::from(v))
        .collect();

    let mut tape = Tape::new(cpu);
    let vars: Vec<_> = inputs
        .iter()
        .zip(shapes())
        .map(|(v, s)| {
            let f: Vec<f32> = v.iter().map(|&x| x as f32).collect();
            tape.leaf(Tensor::from_f32(&f, &s, &budget).unwrap())
                .unwrap()
        })
        .collect();
    let p = BlockParams {
        norm1: vars[2],
        q_proj: vars[3],
        k_proj: vars[4],
        v_proj: vars[5],
        o_proj: vars[6],
        q_norm: vars[7],
        k_norm: vars[8],
        gate_w: vars[9],
        gate_b: vars[10],
        vr_lambda: vars[11],
        norm2: vars[12],
        ffn_gate: vars[13],
        ffn_up: vars[14],
        ffn_down: vars[15],
    };
    let out = block(&mut tape, &spec, &p, &vars[0], Some(&vars[1]), &rope, B).unwrap();
    let loss = tape.mul(out.x, vars[16]).unwrap();
    tape.backward(loss).unwrap();

    // The f64 reference agrees with the f32 forward.
    let got: f64 = tape
        .value(loss)
        .unwrap()
        .to_f32_vec()
        .unwrap()
        .iter()
        .map(|&v| f64::from(v))
        .sum();
    let want = block_ref(&inputs, &cos, &sin);
    assert!(
        (got - want).abs() < 1e-4 * want.abs().max(1.0),
        "{got} vs {want}"
    );

    let names = [
        "x",
        "v0",
        "norm1",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "q_norm",
        "k_norm",
        "gate_w",
        "gate_b",
        "vr_lambda",
        "norm2",
        "ffn_gate",
        "ffn_up",
        "ffn_down",
    ];
    for (i, name) in names.iter().enumerate() {
        let numeric = central_diff(&inputs[i], 1e-3, |point| {
            let mut all = inputs.clone();
            all[i] = point.to_vec();
            Ok(block_ref(&all, &cos, &sin))
        })
        .unwrap();
        let analytic = tape.grad(vars[i]).unwrap().to_f32_vec().unwrap();
        gradients_match(&analytic, &numeric, 2e-3, 2e-2).unwrap_or_else(|e| panic!("{name}: {e}"));
    }
}

// ------------------------------------------------------------------ loader

fn encode(
    spec: &ModelSpec,
    params: &[Tensor],
    edit: impl Fn(&mut Vec<(String, Vec<u64>, Vec<u8>)>),
) -> Vec<u8> {
    encode_with(spec, params, edit, &[])
}

fn encode_with(
    spec: &ModelSpec,
    params: &[Tensor],
    edit: impl Fn(&mut Vec<(String, Vec<u64>, Vec<u8>)>),
    metadata: &[(&str, &str)],
) -> Vec<u8> {
    let table = param_table(spec).unwrap();
    let mut items: Vec<(String, Vec<u64>, Vec<u8>)> = table
        .iter()
        .zip(params)
        .map(|(info, t)| {
            let shape = info.shape.iter().map(|&d| d as u64).collect();
            let bytes = t
                .to_f32_vec()
                .unwrap()
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            (info.name.clone(), shape, bytes)
        })
        .collect();
    edit(&mut items);
    let outs: Vec<TensorOut<'_>> = items
        .iter()
        .map(|(name, shape, data)| TensorOut {
            name,
            dtype: StDtype::F32,
            shape,
            data,
        })
        .collect();
    encode_safetensors(&outs, metadata).unwrap()
}

#[test]
fn loader_accepts_compiled_names_and_a_bit_equal_head() {
    let spec = ModelSpec::tiny();
    let budget = Budget::new(1 << 28);
    let params = init_params(&spec, 3, &budget).unwrap();
    let bytes = encode(&spec, &params, |items| {
        let head = items[0].clone();
        for item in items.iter_mut() {
            item.0 = format!("{COMPILED_PREFIX}{}", item.0);
        }
        items.push((LM_HEAD.to_string(), head.1, head.2));
    });
    let file = SafeTensors::parse(&bytes).unwrap();
    let loaded = load_params(&spec, &file, &budget).unwrap();
    assert_eq!(loaded.len(), params.len());
    for (a, b) in loaded.iter().zip(&params) {
        assert_eq!(a.shape(), b.shape());
        assert_eq!(tensor_bits(a), tensor_bits(b));
    }
}

#[test]
fn the_spec_rides_in_metadata_and_is_required() {
    let spec = ModelSpec::tiny();
    let budget = Budget::new(1 << 28);
    let params = init_params(&spec, 3, &budget).unwrap();
    // Without `ojas.spec` the shape is never guessed.
    let bare = encode(&spec, &params, |_| {});
    let file = SafeTensors::parse(&bare).unwrap();
    assert!(load_model(&file, &budget).is_err());
    assert!(load_spec(&file).is_err());
    // With the writer's JSON it loads, and the spec decides the table.
    let json = spec.to_json().unwrap();
    let tagged = encode_with(&spec, &params, |_| {}, &[(SPEC_METADATA_KEY, &json)]);
    let file = SafeTensors::parse(&tagged).unwrap();
    let (got, loaded) = load_model(&file, &budget).unwrap();
    assert_eq!(got, spec);
    for (a, b) in loaded.iter().zip(&params) {
        assert_eq!(tensor_bits(a), tensor_bits(b));
    }
    // A spec that disagrees with the tensors is refused.
    let wrong = ModelSpec {
        hidden: 128,
        ..spec
    }
    .to_json()
    .unwrap();
    let mismatched = encode_with(&spec, &params, |_| {}, &[(SPEC_METADATA_KEY, &wrong)]);
    let file = SafeTensors::parse(&mismatched).unwrap();
    assert!(load_model(&file, &budget).is_err());
}

#[test]
fn loader_refuses_every_mismatch() {
    let spec = ModelSpec::tiny();
    let budget = Budget::new(1 << 28);
    let params = init_params(&spec, 3, &budget).unwrap();
    type Edit = Box<dyn Fn(&mut Vec<(String, Vec<u64>, Vec<u8>)>)>;
    let cases: Vec<(&str, Edit)> = vec![
        (
            "missing tensor",
            Box::new(|items| {
                items.retain(|i| i.0 != "blocks.1.ffn.up.weight");
            }),
        ),
        (
            "extra tensor",
            Box::new(|items| {
                items.push(("pos_emb.weight".into(), vec![1], vec![0; 4]));
            }),
        ),
        (
            "wrong shape",
            Box::new(|items| {
                let i = items.iter().position(|i| i.0 == "norm_f.weight").unwrap();
                items[i].1 = vec![32, 2];
            }),
        ),
        (
            "head differs by one bit",
            Box::new(|items| {
                let mut head = items[0].clone();
                head.0 = LM_HEAD.into();
                head.2[0] ^= 1;
                items.push(head);
            }),
        ),
        (
            "same parameter twice",
            Box::new(|items| {
                let mut dup = items[1].clone();
                dup.0 = format!("{COMPILED_PREFIX}{}", dup.0);
                items.push(dup);
            }),
        ),
    ];
    for (what, edit) in cases {
        let bytes = encode(&spec, &params, edit);
        let file = SafeTensors::parse(&bytes).unwrap();
        assert!(
            load_params(&spec, &file, &budget).is_err(),
            "{what} accepted"
        );
    }
}

/// Refusals made from the header, before any tensor data is read or any
/// byte is charged: a budget smaller than the parameters is
/// `CapacityExceeded` with nothing left charged, and a missing tensor is
/// found before an earlier one is read.
#[test]
fn loader_checks_room_and_names_before_reading() {
    let spec = ModelSpec::tiny();
    let budget = Budget::new(1 << 28);
    let params = init_params(&spec, 3, &budget).unwrap();
    let total: u64 = params
        .iter()
        .map(|p| p.num_elements().unwrap() as u64 * 4)
        .sum();
    let bytes = encode(&spec, &params, |_| {});
    let file = SafeTensors::parse(&bytes).unwrap();
    // One byte short: refused up front, nothing charged.
    let short = Budget::new(total - 1);
    let err = load_params(&spec, &file, &short).unwrap_err();
    assert!(matches!(err, OjasError::CapacityExceeded { .. }), "{err:?}");
    assert_eq!(short.peak_bytes(), 0, "a tensor was read before the check");
    // The decode reads in chunks beside each tensor; the parameters alone
    // and their chunk buffers fit here.
    let room = Budget::new(total + (1 << 20));
    assert_eq!(
        load_params(&spec, &file, &room).unwrap().len(),
        params.len()
    );
    // The last tensor of the table missing: refused with nothing charged,
    // so no earlier tensor was decoded first.
    let last = param_table(&spec).unwrap().last().unwrap().name.clone();
    let missing = encode(&spec, &params, |items| items.retain(|i| i.0 != last));
    let file = SafeTensors::parse(&missing).unwrap();
    let fresh = Budget::new(1 << 28);
    let err = load_params(&spec, &file, &fresh).unwrap_err();
    assert!(err.to_string().contains("missing tensor"), "{err}");
    assert_eq!(fresh.peak_bytes(), 0, "a tensor was read before the check");
}

/// A spec cannot size the name table freely: one block past
/// `MAX_LAYERS` is refused by `validate`, so `param_table` never builds it.
#[test]
fn a_spec_past_max_layers_is_refused_before_any_table() {
    let spec = ModelSpec {
        n_layer: ojas_model::MAX_LAYERS + 1,
        ..ModelSpec::tiny()
    };
    assert!(matches!(spec.validate(), Err(OjasError::OutOfRange { .. })));
    assert!(param_table(&spec).is_err());
    let huge = ModelSpec {
        n_layer: usize::MAX,
        ..ModelSpec::tiny()
    };
    assert!(param_table(&huge).is_err());
    let at = ModelSpec {
        n_layer: ojas_model::MAX_LAYERS,
        ..ModelSpec::tiny()
    };
    at.validate().unwrap();
}

#[test]
fn the_forward_refuses_mismatched_inputs() {
    let spec = ModelSpec::tiny();
    let cpu = exact();
    let budget = cpu.budget().clone();
    let params = random_params(&spec, 1, &budget);
    let rope = Rope::new(&spec, 4, &budget).unwrap();
    let mut eval = Eval::new(cpu);
    let p = bind(&mut eval, &spec, &params).unwrap();
    let wrong_len = ids(&spec, 1, 5, 0, &budget);
    assert!(forward_hidden(&mut eval, &spec, &p, &wrong_len, &rope).is_err());
    let short = ModelSpec { max_seq: 3, ..spec };
    let x4 = ids(&spec, 1, 4, 0, &budget);
    assert!(forward_hidden(&mut eval, &short, &p, &x4, &rope).is_err());
    let flat = Tensor::from_u32(&[1, 2, 3, 4], &[4], &budget).unwrap();
    assert!(forward_hidden(&mut eval, &spec, &p, &flat, &rope).is_err());
    let x = ids(&spec, 1, 4, 0, &budget);
    let bad_targets = Tensor::from_u32(&[1, 2, 3], &[3], &budget).unwrap();
    assert!(forward_loss(&mut eval, &spec, &p, &x, &bad_targets, &rope, None, CHUNK).is_err());
    let out_of_vocab = Tensor::from_u32(&[1, 2, 3, 256], &[1, 4], &budget).unwrap();
    assert!(forward_hidden(&mut eval, &spec, &p, &out_of_vocab, &rope).is_err());
    // A grouped-query spec runs on Eval, but these are multi-head weights.
    let gqa = ModelSpec {
        n_kv_head: 2,
        ..spec
    };
    assert!(forward_hidden(&mut eval, &gqa, &p, &x, &rope).is_err());
}

/// Grouped-query weights as multi-head ones: query head `h` gets the k and v
/// rows of KV head `group(h)`.
fn expand_kv(
    gqa: &ModelSpec,
    params: &[Tensor],
    group: impl Fn(usize) -> usize,
    budget: &Budget,
) -> Vec<Tensor> {
    let (d, dh) = (gqa.n_embd, gqa.head_dim);
    param_table(gqa)
        .unwrap()
        .iter()
        .zip(params)
        .map(|(info, t)| {
            if !(info.name.ends_with("k_proj.weight") || info.name.ends_with("v_proj.weight")) {
                return t.clone();
            }
            let rows = t.to_f32_vec().unwrap();
            let mut out = Vec::with_capacity(gqa.n_head * dh * d);
            for h in 0..gqa.n_head {
                let j = group(h);
                out.extend_from_slice(&rows[j * dh * d..(j + 1) * dh * d]);
            }
            Tensor::from_f32(&out, &[gqa.n_head * dh, d], budget).unwrap()
        })
        .collect()
}

#[test]
fn eval_and_tape_run_grouped_query_attention() {
    let mha = ModelSpec::tiny();
    let gqa = ModelSpec {
        n_kv_head: 2,
        ..mha
    };
    let cpu = exact();
    let budget = cpu.budget().clone();
    let params = random_params(&gqa, 9, &budget);
    let rope = Rope::new(&gqa, 8, &budget).unwrap();
    let x = ids(&gqa, 2, 8, 3, &budget);
    let logits = |spec: &ModelSpec, params: &[Tensor]| {
        let mut eval = Eval::new(exact());
        let p = bind(&mut eval, spec, params).unwrap();
        forward_logits(&mut eval, spec, &p, &x, &rope)
            .unwrap()
            .to_f32_vec()
            .unwrap()
    };
    let max_diff = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };
    let got = logits(&gqa, &params);
    // Head h reads KV head h / (n_head / n_kv_head): the same model as
    // multi-head attention with each KV head's rows repeated.
    let rep = gqa.n_head / gqa.n_kv_head;
    let same = logits(&mha, &expand_kv(&gqa, &params, |h| h / rep, &budget));
    let d = max_diff(&got, &same);
    assert!(d <= 1e-4, "GQA vs repeated-KV MHA: {d}");
    // Any other grouping is a different model.
    let other = logits(
        &mha,
        &expand_kv(&gqa, &params, |h| h % gqa.n_kv_head, &budget),
    );
    assert!(max_diff(&got, &other) > 1e-3, "the grouping is not checked");

    // The tape records the same grouped-query forward and a k-projection
    // gradient of the parameter's shape.
    let mut tape = Tape::new(exact());
    let p = bind(&mut tape, &gqa, &params).unwrap();
    let y = forward_logits(&mut tape, &gqa, &p, &x, &rope).unwrap();
    let tape_logits = tape.value(y).unwrap().to_f32_vec().unwrap();
    assert_eq!(tape_logits, got);
    let k_proj = p.blocks[0].k_proj;
    tape.backward(y).unwrap();
    let gk = tape.grad(k_proj).unwrap();
    assert_eq!(gk.shape(), tape.value(k_proj).unwrap().shape());
    assert!(gk.to_f32_vec().unwrap().iter().any(|v| *v != 0.0));
}

#[test]
fn block_with_hands_attention_the_cache_layout() {
    // A KV-cache attention through block_with: post-RoPE k and blended v
    // written to caches, then cached attention, matches the causal block.
    // Layer 1 blends with layer 0's v, so a raw v would not match.
    let spec = ModelSpec::tiny();
    let cpu = exact();
    let budget = cpu.budget().clone();
    let params = random_params(&spec, 6, &budget);
    let (b, t) = (2, 6);
    let rope = Rope::new(&spec, t, &budget).unwrap();
    let mut eval = Eval::new(cpu);
    let p = bind(&mut eval, &spec, &params).unwrap();
    let x = eval
        .embedding(&p.tok_emb, &ids(&spec, b, t, 1, &budget))
        .unwrap();
    let l0 = block(&mut eval, &spec, &p.blocks[0], &x, None, &rope, b).unwrap();
    let want = block(
        &mut eval,
        &spec,
        &p.blocks[1],
        &l0.x,
        Some(&l0.raw_v),
        &rope,
        b,
    )
    .unwrap();

    let kv = [b, t, spec.n_kv_head, spec.head_dim];
    let mut keys = Tensor::zeros(&kv, ojas_core::DType::F32, &budget).unwrap();
    let mut values = Tensor::zeros(&kv, ojas_core::DType::F32, &budget).unwrap();
    let mut seen = Vec::new();
    let mut handed_v = None;
    let got = ojas_model::block_with(
        &mut eval,
        &spec,
        &p.blocks[1],
        &l0.x,
        Some(&l0.raw_v),
        &rope,
        b,
        |g, q, k, v| {
            seen = vec![q.shape().to_vec(), k.shape().to_vec(), v.shape().to_vec()];
            handed_v = Some(v.clone());
            g.kv_cache_write(&mut keys, k, 0)?;
            g.kv_cache_write(&mut values, v, 0)?;
            g.cached_attn(q, &keys, &values, t)
        },
    )
    .unwrap();
    assert_eq!(
        seen,
        [
            vec![b, t, spec.n_head, spec.head_dim],
            vec![b, t, spec.n_kv_head, spec.head_dim],
            vec![b, t, spec.n_kv_head, spec.head_dim],
        ]
    );
    let (a, w) = (got.x.to_f32_vec().unwrap(), want.x.to_f32_vec().unwrap());
    let d = a
        .iter()
        .zip(&w)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(d <= 1e-5, "cached block vs causal block: {d}");
    assert_eq!(tensor_bits(&got.raw_v), tensor_bits(&want.raw_v));
    // The v handed to attention is the blend with layer 0's raw v.
    let blended = eval
        .vres(&want.raw_v, &l0.raw_v, &p.blocks[1].vr_lambda)
        .unwrap();
    assert_eq!(
        tensor_bits(&handed_v.unwrap()),
        tensor_bits(&blended),
        "attention got the raw v"
    );
}

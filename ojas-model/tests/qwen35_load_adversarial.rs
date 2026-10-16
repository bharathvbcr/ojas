//! Adversarial attack on `load_hf`'s header-sized budget check and on
//! `load_runs`: bit-identity against the pre-G-IOREAD staging (the old
//! `read` / `read_as` / `split_rows` / `deinterleave`, copied here as the
//! oracle), exact Budget accounting for every `table` x `linears` storage,
//! chunk-boundary run lengths, failure after open, and parent budgets.
//!
//! The oracle reads through `read_bytes` and decodes by hand, so it does not
//! lean on `read_typed`, which the change also rewrote.

use std::fs::OpenOptions;
use std::path::PathBuf;

use ojas_core::{f32_to_bf16, Budget, DType, OjasError, Tensor};
use ojas_io::{
    bf16_to_f32, encode_safetensors, f16_to_f32, f32_to_f16, write_safetensors, SafeTensors,
    StDtype, TensorOut,
};
use ojas_model::qwen35::{hf_tensors, load_hf, Qwen35Mixer, Qwen35Params, Qwen35Spec};

const PREFIX: &str = "model.";

// ---- a tiny seeded generator (no dependency) -----------------------------

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + self.below((hi - lo + 1) as u64) as usize
    }
    fn value(&mut self) -> f32 {
        match self.below(40) {
            0 => 1e30,
            1 => -0.0,
            2 => 1e-40,
            3 => f32::NAN,
            4 => 70000.0,
            _ => ((self.next() & 0xff_ffff) as f32 / 8_388_608.0 - 1.0) * 3.0,
        }
    }
}

// ---- spec and file construction ------------------------------------------

#[derive(Clone, Copy, Debug)]
enum Src {
    F32,
    Bf16,
    F16,
    Mixed,
}

fn base_spec() -> Qwen35Spec {
    Qwen35Spec {
        vocab: 7,
        hidden: 5,
        intermediate: 3,
        layers: vec![Qwen35Mixer::GatedDeltaNet, Qwen35Mixer::Attention],
        q_heads: 3,
        kv_heads: 1,
        head_dim: 2,
        rotary_dim: 2,
        rope_theta: 1e4,
        gdn_heads: 3,
        gdn_key_dim: 2,
        gdn_value_dim: 3,
        conv_width: 4,
        eps: 1e-6,
    }
}

fn random_spec(rng: &mut Lcg) -> Qwen35Spec {
    let n_layers = rng.range(1, 4);
    let layers = (0..n_layers)
        .map(|_| {
            if rng.below(2) == 0 {
                Qwen35Mixer::GatedDeltaNet
            } else {
                Qwen35Mixer::Attention
            }
        })
        .collect();
    let kv = rng.range(1, 3);
    let head_dim = rng.range(2, 6);
    Qwen35Spec {
        vocab: rng.range(1, 20),
        hidden: rng.range(1, 9),
        intermediate: rng.range(1, 7),
        layers,
        q_heads: kv * rng.range(1, 3),
        kv_heads: kv,
        head_dim,
        rotary_dim: 2 * rng.range(1, head_dim / 2),
        rope_theta: 1e4,
        gdn_heads: rng.range(1, 4),
        gdn_key_dim: rng.range(1, 5),
        gdn_value_dim: rng.range(1, 5),
        conv_width: rng.range(1, 4),
        eps: 1e-6,
    }
}

/// The file's bytes: every tensor `hf_tensors` names, random values, each in
/// `src`'s dtype (`Mixed`: a random one per tensor).
fn build(spec: &Qwen35Spec, prefix: &str, src: Src, rng: &mut Lcg) -> Vec<u8> {
    let tensors = hf_tensors(spec, prefix);
    let mut kept: Vec<(String, StDtype, Vec<u64>, Vec<u8>)> = Vec::new();
    for t in &tensors {
        let n: usize = t.shape.iter().product();
        let dtype = match src {
            Src::F32 => StDtype::F32,
            Src::Bf16 => StDtype::BF16,
            Src::F16 => StDtype::F16,
            Src::Mixed => [StDtype::F32, StDtype::BF16, StDtype::F16][rng.below(3) as usize],
        };
        let mut data = Vec::new();
        for _ in 0..n {
            let v = rng.value();
            match dtype {
                StDtype::F32 => data.extend_from_slice(&v.to_le_bytes()),
                StDtype::BF16 => data.extend_from_slice(&f32_to_bf16(v).to_le_bytes()),
                _ => data.extend_from_slice(&f32_to_f16(v).to_le_bytes()),
            }
        }
        kept.push((
            t.name.clone(),
            dtype,
            t.shape.iter().map(|&d| d as u64).collect(),
            data,
        ));
    }
    let items: Vec<TensorOut<'_>> = kept
        .iter()
        .map(|(name, dtype, shape, data)| TensorOut {
            name,
            dtype: *dtype,
            shape,
            data,
        })
        .collect();
    encode_safetensors(&items, &[]).unwrap()
}

// ---- the oracle: the pre-change staging ----------------------------------

#[derive(Clone, Debug, PartialEq)]
enum Vals {
    F32(Vec<u32>),
    Bf16(Vec<u16>),
}

fn split_rows<T: Copy>(v: &[T], cols: usize, counts: &[usize]) -> Vec<Vec<T>> {
    let mut at = 0;
    counts
        .iter()
        .map(|&n| {
            let part = v[at * cols..(at + n) * cols].to_vec();
            at += n;
            part
        })
        .collect()
}

fn deinterleave<T: Copy>(v: &[T], heads: usize, d: usize, cols: usize) -> (Vec<T>, Vec<T>) {
    let block = d * cols;
    let (mut q, mut g) = (
        Vec::with_capacity(heads * block),
        Vec::with_capacity(heads * block),
    );
    for h in 0..heads {
        let at = h * 2 * block;
        q.extend_from_slice(&v[at..at + block]);
        g.extend_from_slice(&v[at + block..at + 2 * block]);
    }
    (q, g)
}

impl Vals {
    fn split(&self, cols: usize, counts: &[usize]) -> Vec<Vals> {
        match self {
            Vals::F32(v) => split_rows(v, cols, counts)
                .into_iter()
                .map(Vals::F32)
                .collect(),
            Vals::Bf16(v) => split_rows(v, cols, counts)
                .into_iter()
                .map(Vals::Bf16)
                .collect(),
        }
    }
    fn deinterleave(&self, heads: usize, d: usize, cols: usize) -> (Vals, Vals) {
        match self {
            Vals::F32(v) => {
                let (q, g) = deinterleave(v, heads, d, cols);
                (Vals::F32(q), Vals::F32(g))
            }
            Vals::Bf16(v) => {
                let (q, g) = deinterleave(v, heads, d, cols);
                (Vals::Bf16(q), Vals::Bf16(g))
            }
        }
    }
    fn bytes(&self) -> usize {
        match self {
            Vals::F32(v) => v.len() * 4,
            Vals::Bf16(v) => v.len() * 2,
        }
    }
}

fn widen(st: &SafeTensors<'_>, name: &str) -> Vec<f32> {
    let bytes = st.read_bytes(name).unwrap();
    match st.info(name).unwrap().dtype {
        StDtype::F32 => bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        StDtype::BF16 => bytes
            .chunks_exact(2)
            .map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]])))
            .collect(),
        StDtype::F16 => bytes
            .chunks_exact(2)
            .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
            .collect(),
        other => panic!("{name}: {other:?}"),
    }
}

fn read_as(st: &SafeTensors<'_>, name: &str, storage: DType) -> Vals {
    if storage == DType::F32 {
        return Vals::F32(widen(st, name).iter().map(|v| v.to_bits()).collect());
    }
    if st.info(name).unwrap().dtype == StDtype::BF16 {
        let bytes = st.read_bytes(name).unwrap();
        return Vals::Bf16(
            bytes
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect(),
        );
    }
    Vals::Bf16(widen(st, name).into_iter().map(f32_to_bf16).collect())
}

type Out = (Vec<usize>, Vals);

/// What the old `load_hf` produced, in `Qwen35Params::values` order.
fn old_load(
    spec: &Qwen35Spec,
    st: &SafeTensors<'_>,
    prefix: &str,
    linears: DType,
    table: DType,
) -> Vec<Out> {
    let (d, f) = (spec.hidden, spec.intermediate);
    let f32v =
        |name: String, shape: &[usize]| -> Out { (shape.to_vec(), read_as(st, &name, DType::F32)) };
    let lin =
        |name: String, shape: &[usize]| -> Out { (shape.to_vec(), read_as(st, &name, linears)) };
    let mut out = vec![(
        vec![spec.vocab, d],
        read_as(st, &format!("{prefix}embed_tokens.weight"), table),
    )];
    for (i, kind) in spec.layers.iter().enumerate() {
        let l = format!("{prefix}layers.{i}.");
        out.push(f32v(format!("{l}input_layernorm.weight"), &[d]));
        match kind {
            Qwen35Mixer::GatedDeltaNet => {
                let m = format!("{l}linear_attn.");
                let chans = spec.gdn_qkv_channels();
                let (h, dv, k) = (spec.gdn_heads, spec.gdn_value_dim, spec.conv_width);
                let w = read_as(st, &format!("{m}in_proj_qkv.weight"), linears).split(d, &chans);
                let Vals::F32(cv) = read_as(st, &format!("{m}conv1d.weight"), DType::F32) else {
                    unreachable!()
                };
                let c = split_rows(&cv, k, &chans);
                for i in 0..3 {
                    out.push((vec![chans[i], d], w[i].clone()));
                }
                for i in 0..3 {
                    out.push((vec![chans[i], k], Vals::F32(c[i].clone())));
                }
                out.push(lin(format!("{m}in_proj_z.weight"), &[h * dv, d]));
                out.push(lin(format!("{m}in_proj_a.weight"), &[h, d]));
                out.push(lin(format!("{m}in_proj_b.weight"), &[h, d]));
                out.push(f32v(format!("{m}A_log"), &[h]));
                out.push(f32v(format!("{m}dt_bias"), &[h]));
                out.push(f32v(format!("{m}norm.weight"), &[dv]));
                out.push(lin(format!("{m}out_proj.weight"), &[d, h * dv]));
            }
            Qwen35Mixer::Attention => {
                let m = format!("{l}self_attn.");
                let (hq, hkv, hd) = (spec.q_heads, spec.kv_heads, spec.head_dim);
                let (q, g) =
                    read_as(st, &format!("{m}q_proj.weight"), linears).deinterleave(hq, hd, d);
                out.push((vec![hq * hd, d], q));
                out.push((vec![hq * hd, d], g));
                out.push(lin(format!("{m}k_proj.weight"), &[hkv * hd, d]));
                out.push(lin(format!("{m}v_proj.weight"), &[hkv * hd, d]));
                out.push(f32v(format!("{m}q_norm.weight"), &[hd]));
                out.push(f32v(format!("{m}k_norm.weight"), &[hd]));
                out.push(lin(format!("{m}o_proj.weight"), &[d, hq * hd]));
            }
        }
        out.push(f32v(format!("{l}post_attention_layernorm.weight"), &[d]));
        out.push(lin(format!("{l}mlp.gate_proj.weight"), &[f, d]));
        out.push(lin(format!("{l}mlp.up_proj.weight"), &[f, d]));
        out.push(lin(format!("{l}mlp.down_proj.weight"), &[d, f]));
    }
    out.push(f32v(format!("{prefix}norm.weight"), &[d]));
    out
}

fn vals_of(t: &Tensor) -> Vals {
    match t.dtype() {
        DType::F32 => Vals::F32(t.f32_slice().unwrap().iter().map(|v| v.to_bits()).collect()),
        DType::Bf16 => Vals::Bf16(t.bf16_slice().unwrap().to_vec()),
        other => panic!("{other:?}"),
    }
}

const STORAGES: [(DType, DType); 4] = [
    (DType::F32, DType::F32),
    (DType::Bf16, DType::F32),
    (DType::F32, DType::Bf16),
    (DType::Bf16, DType::Bf16),
];

/// `load_hf` against the oracle, plus exact accounting. Returns the total.
fn check(spec: &Qwen35Spec, bytes: &[u8], prefix: &str, ctx: &str) -> u64 {
    let st = SafeTensors::parse(bytes).unwrap();
    let mut total_seen = 0;
    for (linears, table) in STORAGES {
        let want = old_load(spec, &st, prefix, linears, table);
        let total: u64 = want.iter().map(|(_, v)| v.bytes() as u64).sum();
        total_seen = total;

        let budget = Budget::new(total);
        let params = load_hf(spec, &st, prefix, linears, table, &budget)
            .unwrap_or_else(|e| panic!("{ctx} {linears:?}/{table:?}: exact-total budget: {e}"));
        let got = params.values();
        assert_eq!(got.len(), want.len(), "{ctx}: parameter count");
        for (i, (t, (shape, vals))) in got.iter().zip(&want).enumerate() {
            assert_eq!(
                t.shape(),
                &shape[..],
                "{ctx} {linears:?}/{table:?} param {i} shape"
            );
            assert_eq!(
                &vals_of(t),
                vals,
                "{ctx} {linears:?}/{table:?} param {i} values"
            );
        }
        // total is exactly what was charged; nothing else was held at the peak
        // beyond one decode chunk (the orchestrator knows the chunk is not
        // charged today; a future charged chunk may not exceed 128 KiB).
        assert_eq!(budget.live_bytes().unwrap(), total, "{ctx} live");
        assert!(
            budget.peak_bytes() >= total && budget.peak_bytes() <= total + (128 << 10),
            "{ctx}: peak {} vs total {total}",
            budget.peak_bytes()
        );
        drop(params);
        assert_eq!(budget.live_bytes().unwrap(), 0, "{ctx}: leak after drop");

        // one byte short refuses with CapacityExceeded and charges nothing
        if total > 0 {
            let tight = Budget::new(total - 1);
            let err = load_hf(spec, &st, prefix, linears, table, &tight).unwrap_err();
            assert!(
                matches!(err, OjasError::CapacityExceeded { .. }),
                "{ctx}: total-1 gave {err:?}"
            );
            assert_eq!(tight.live_bytes().unwrap(), 0);
            assert_eq!(
                tight.peak_bytes(),
                0,
                "{ctx}: refusal must precede any charge"
            );
        }
    }
    total_seen
}

#[test]
fn every_source_dtype_and_storage_is_bit_identical_to_the_old_staging() {
    let spec = base_spec();
    for (seed, src) in [Src::F32, Src::Bf16, Src::F16, Src::Mixed]
        .into_iter()
        .enumerate()
    {
        let mut rng = Lcg(100 + seed as u64);
        let bytes = build(&spec, PREFIX, src, &mut rng);
        check(&spec, &bytes, PREFIX, &format!("{src:?}"));
    }
}

#[test]
fn runs_that_straddle_or_exactly_fill_a_decode_chunk_match_the_old_staging() {
    // The decode chunk is 64 KiB: 16384 f32 or 32768 half-width elements. A
    // run of `head_dim * hidden` (and a GDN block of `key_dim * hidden`) is
    // exactly one chunk, one element short, or one element over, for both widths.
    for (hd, d) in [
        (64, 256),
        (64, 255),
        (64, 257),
        (128, 256),
        (128, 255),
        (128, 257),
    ] {
        let spec = Qwen35Spec {
            vocab: 2,
            hidden: d,
            intermediate: 1,
            layers: vec![Qwen35Mixer::GatedDeltaNet, Qwen35Mixer::Attention],
            q_heads: 3,
            kv_heads: 1,
            head_dim: hd,
            rotary_dim: 2,
            rope_theta: 1e4,
            gdn_heads: 1,
            gdn_key_dim: hd,
            gdn_value_dim: hd,
            conv_width: 3,
            eps: 1e-6,
        };
        for (seed, src) in [Src::F32, Src::Bf16, Src::F16].into_iter().enumerate() {
            let mut rng = Lcg(7 * hd as u64 + d as u64 + seed as u64);
            let bytes = build(&spec, PREFIX, src, &mut rng);
            check(&spec, &bytes, PREFIX, &format!("hd={hd} d={d} {src:?}"));
        }
    }
}

#[test]
fn an_odd_head_count_spec_with_runs_past_two_chunks_matches() {
    let spec = Qwen35Spec {
        vocab: 130,
        hidden: 257,
        intermediate: 3,
        layers: vec![Qwen35Mixer::GatedDeltaNet, Qwen35Mixer::Attention],
        q_heads: 3,
        kv_heads: 1,
        head_dim: 130,
        rotary_dim: 4,
        rope_theta: 1e4,
        gdn_heads: 3,
        gdn_key_dim: 70,
        gdn_value_dim: 90,
        conv_width: 5,
        eps: 1e-6,
    };
    let mut rng = Lcg(2024);
    let bytes = build(&spec, PREFIX, Src::Mixed, &mut rng);
    check(&spec, &bytes, PREFIX, "straddle");
}

#[test]
fn seeded_random_specs_match_the_oracle_with_exact_budgets() {
    for seed in 0..60u64 {
        let mut rng = Lcg(seed * 7919 + 1);
        let spec = random_spec(&mut rng);
        spec.validate().unwrap();
        let src = [Src::F32, Src::Bf16, Src::F16, Src::Mixed][rng.below(4) as usize];
        let prefix = ["model.", "model.language_model."][rng.below(2) as usize];
        let bytes = build(&spec, prefix, src, &mut rng);
        check(
            &spec,
            &bytes,
            prefix,
            &format!("seed {seed} {src:?} {spec:?}"),
        );
    }
}

/// Which outputs follow `linears` / `table` / F32, from the oracle's own
/// classification, against what `load_hf` produced: a dtype, not only values.
#[test]
fn every_tensor_lands_in_the_storage_the_old_code_gave_it() {
    let spec = base_spec();
    let mut rng = Lcg(5);
    let bytes = build(&spec, PREFIX, Src::F32, &mut rng);
    let st = SafeTensors::parse(&bytes).unwrap();
    for (linears, table) in STORAGES {
        let budget = Budget::new(1 << 30);
        let params = load_hf(&spec, &st, PREFIX, linears, table, &budget).unwrap();
        let want = old_load(&spec, &st, PREFIX, linears, table);
        for (i, (t, (_, vals))) in params.values().iter().zip(&want).enumerate() {
            let expect = match vals {
                Vals::F32(_) => DType::F32,
                Vals::Bf16(_) => DType::Bf16,
            };
            assert_eq!(t.dtype(), expect, "param {i} under {linears:?}/{table:?}");
        }
    }
}

// ---- classification fragility --------------------------------------------

/// `is_linear` (params.rs) is `name.ends_with(".weight") && name.contains("_proj")`
/// over the FULL name, prefix included. A checkpoint prefix containing
/// `_proj` makes every `.weight` norm a "linear", so under `Bf16` linears the
/// norms silently become Bf16 where the old code always kept them F32.
#[test]
fn real_bug_a_prefix_containing_proj_must_not_change_norm_storage() {
    let spec = base_spec();
    let prefix = "vision_proj.";
    let mut rng = Lcg(9);
    let bytes = build(&spec, prefix, Src::F32, &mut rng);
    let st = SafeTensors::parse(&bytes).unwrap();
    let budget = Budget::new(1 << 30);
    let params: Qwen35Params<Tensor> =
        load_hf(&spec, &st, prefix, DType::Bf16, DType::F32, &budget).unwrap();
    let want = old_load(&spec, &st, prefix, DType::Bf16, DType::F32);
    for (i, (t, (_, vals))) in params.values().iter().zip(&want).enumerate() {
        let expect = match vals {
            Vals::F32(_) => DType::F32,
            Vals::Bf16(_) => DType::Bf16,
        };
        assert_eq!(t.dtype(), expect, "param {i}");
    }
}

// ---- hostile files through load_hf ---------------------------------------

#[test]
fn a_tensor_the_loader_cannot_decode_is_refused_before_any_charge() {
    let spec = base_spec();
    let tensors = hf_tensors(&spec, PREFIX);
    let mut datas: Vec<Vec<u8>> = Vec::new();
    let mut dtypes = Vec::new();
    for (i, t) in tensors.iter().enumerate() {
        let n: usize = t.shape.iter().product();
        if i == 3 {
            dtypes.push(StDtype::C64);
            datas.push(vec![0u8; n * 8]);
        } else {
            dtypes.push(StDtype::F32);
            datas.push(vec![0u8; n * 4]);
        }
    }
    let shapes: Vec<Vec<u64>> = tensors
        .iter()
        .map(|t| t.shape.iter().map(|&d| d as u64).collect())
        .collect();
    let items: Vec<TensorOut<'_>> = (0..tensors.len())
        .map(|i| TensorOut {
            name: &tensors[i].name,
            dtype: dtypes[i],
            shape: &shapes[i],
            data: &datas[i],
        })
        .collect();
    let bytes = encode_safetensors(&items, &[]).unwrap();
    let st = SafeTensors::parse(&bytes).unwrap();
    let budget = Budget::new(1 << 30);
    let err = load_hf(&spec, &st, PREFIX, DType::F32, DType::F32, &budget).unwrap_err();
    assert!(err.to_string().contains("C64"), "{err}");
    assert_eq!(budget.peak_bytes(), 0);
}

#[test]
fn a_zero_row_embedding_or_a_missing_tensor_is_refused() {
    let spec = base_spec();
    let tensors = hf_tensors(&spec, PREFIX);
    let build_with = |skip: Option<usize>, zero_embed: bool| -> Vec<u8> {
        let mut kept = Vec::new();
        for (i, t) in tensors.iter().enumerate() {
            if Some(i) == skip {
                continue;
            }
            let mut shape: Vec<u64> = t.shape.iter().map(|&d| d as u64).collect();
            if zero_embed && i == 0 {
                shape = vec![0, spec.hidden as u64];
            }
            let n: u64 = shape.iter().product();
            kept.push((t.name.clone(), shape, vec![0u8; n as usize * 4]));
        }
        let items: Vec<TensorOut<'_>> = kept
            .iter()
            .map(|(name, shape, data)| TensorOut {
                name,
                dtype: StDtype::F32,
                shape,
                data,
            })
            .collect();
        encode_safetensors(&items, &[]).unwrap()
    };
    for bytes in [build_with(None, true), build_with(Some(5), false)] {
        let st = SafeTensors::parse(&bytes).unwrap();
        let budget = Budget::new(1 << 30);
        assert!(load_hf(&spec, &st, PREFIX, DType::F32, DType::F32, &budget).is_err());
        assert_eq!(budget.live_bytes().unwrap(), 0);
    }
}

// ---- budgets -------------------------------------------------------------

#[test]
fn a_parent_with_less_room_than_the_child_refuses_before_charging_either() {
    let spec = base_spec();
    let mut rng = Lcg(11);
    let bytes = build(&spec, PREFIX, Src::Bf16, &mut rng);
    let st = SafeTensors::parse(&bytes).unwrap();
    for (linears, table) in STORAGES {
        let total = old_load(&spec, &st, PREFIX, linears, table)
            .iter()
            .map(|(_, v)| v.bytes() as u64)
            .sum::<u64>();
        let parent = Budget::new(total - 1);
        let child = parent.child(total * 4);
        let err = load_hf(&spec, &st, PREFIX, linears, table, &child).unwrap_err();
        assert!(matches!(err, OjasError::CapacityExceeded { .. }), "{err:?}");
        assert_eq!(parent.live_bytes().unwrap(), 0);
        assert_eq!(child.live_bytes().unwrap(), 0);

        // a child cap one under the total refuses on the child
        let parent = Budget::new(total * 4);
        let child = parent.child(total - 1);
        assert!(load_hf(&spec, &st, PREFIX, linears, table, &child).is_err());
        assert_eq!(parent.live_bytes().unwrap(), 0);

        // exactly enough on both loads, and both account the total
        let parent = Budget::new(total);
        let child = parent.child(total);
        let params = load_hf(&spec, &st, PREFIX, linears, table, &child).unwrap();
        assert_eq!(parent.live_bytes().unwrap(), total);
        assert_eq!(child.live_bytes().unwrap(), total);
        drop(params);
        assert_eq!(parent.live_bytes().unwrap(), 0);
    }
}

#[test]
fn room_already_held_counts_against_the_check() {
    let spec = base_spec();
    let mut rng = Lcg(12);
    let bytes = build(&spec, PREFIX, Src::F32, &mut rng);
    let st = SafeTensors::parse(&bytes).unwrap();
    check(&spec, &bytes, PREFIX, "held");
    let (linears, table) = (DType::Bf16, DType::Bf16);
    let want: u64 = old_load(&spec, &st, PREFIX, linears, table)
        .iter()
        .map(|(_, v)| v.bytes() as u64)
        .sum();
    let budget = Budget::new(want + 100);
    let held = budget.try_reserve(101).unwrap();
    assert!(load_hf(&spec, &st, PREFIX, linears, table, &budget).is_err());
    drop(held);
    let held = budget.try_reserve(100).unwrap();
    let params = load_hf(&spec, &st, PREFIX, linears, table, &budget).unwrap();
    assert_eq!(budget.live_bytes().unwrap(), want + 100);
    drop((held, params));
}

// ---- failure after open --------------------------------------------------

fn tmp(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "ojas-w12-adv-{}-{tag}.safetensors",
        std::process::id()
    ))
}

/// The file is valid at `open` and then loses its tail, as a copy still in
/// flight or a truncating writer would leave it: the header check passes, the
/// reads fail mid-load.
#[test]
fn a_file_truncated_after_open_fails_the_load_and_leaks_no_charge() {
    let spec = base_spec();
    let mut rng = Lcg(77);
    let bytes = build(&spec, PREFIX, Src::Bf16, &mut rng);
    let st0 = SafeTensors::parse(&bytes).unwrap();
    let names: Vec<String> = st0.names().map(str::to_string).collect();
    let datas: Vec<Vec<u8>> = names.iter().map(|n| st0.read_bytes(n).unwrap()).collect();
    let shapes: Vec<Vec<u64>> = names
        .iter()
        .map(|n| st0.info(n).unwrap().shape.clone())
        .collect();
    let items: Vec<TensorOut<'_>> = names
        .iter()
        .enumerate()
        .map(|(i, n)| TensorOut {
            name: n,
            dtype: StDtype::BF16,
            shape: &shapes[i],
            data: &datas[i],
        })
        .collect();
    let probe = tmp("probe");
    write_safetensors(&probe, &items, &[]).unwrap();
    let disk = std::fs::read(&probe).unwrap();
    let _ = std::fs::remove_file(&probe);
    let header_len = u64::from_le_bytes(disk[..8].try_into().unwrap());
    let data_start = 8 + header_len;
    let len = disk.len() as u64;
    for (tag, cut) in [
        ("tail1", len - 1),
        ("half", data_start + (len - data_start) / 2),
        ("embed", data_start + 10),
        ("quarter", data_start + (len - data_start) / 4 + 1),
    ] {
        let path = tmp(tag);
        write_safetensors(&path, &items, &[]).unwrap();
        let on_disk = std::fs::metadata(&path).unwrap().len();
        assert!(cut < on_disk, "{tag}");

        let opened = SafeTensors::open(&path).unwrap();
        OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(cut)
            .unwrap();

        for (linears, table) in STORAGES {
            let budget = Budget::new(1 << 30);
            let before = budget.live_bytes().unwrap();
            let res = load_hf(&spec, &opened, PREFIX, linears, table, &budget);
            assert!(
                res.is_err(),
                "{tag} {linears:?}/{table:?}: truncated file loaded"
            );
            assert_eq!(budget.live_bytes().unwrap(), before, "{tag}: leaked charge");
        }
        // The size check precedes every read: too small a budget is a
        // CapacityExceeded even though the file is unreadable.
        let tiny = Budget::new(16);
        let err = load_hf(&spec, &opened, PREFIX, DType::F32, DType::F32, &tiny).unwrap_err();
        assert!(
            matches!(err, OjasError::CapacityExceeded { .. }),
            "{tag}: {err:?}"
        );
        let _ = std::fs::remove_file(&path);
    }
}

//! The Qwen3.5 hybrid tower on the CPU tape against transformers' autograd.
//!
//! The fixture is `tests/fixtures/qwen35_tiny` (see `qwen35_common`): a random
//! `Qwen3_5ForCausalLM` of the 2B's shape family (head_dim 256, two GDN heads
//! of 128, grouped KV heads, tied embeddings, vocab 8320), one gated delta net
//! layer and one gated attention layer, as a bf16 checkpoint, run at batch 2
//! with `-100` labels on a prompt and on right padding, with transformers'
//! float32 loss and every parameter's gradient. Every parameter is non-trivial
//! in the fixture, so a weight applied in the wrong place (a `1 + w` norm, a
//! fused row block) shows in that tensor's gradient.
//!
//! Bounds are tessl's for the same fixture against the same reference
//! (`tests/qwen35_train.rs`): loss within 1e-5 relative, and each gradient
//! within 1e-4 of its own peak.

mod qwen35_common;

use ojas_core::{bf16_to_f32, f32_to_bf16, Budget, DType, OjasError, Tensor};
use ojas_io::{encode_f32_as, encode_safetensors, SafeTensors, StDtype, TensorOut};
use ojas_model::qwen35::{
    bind, forward_loss, hf_tensors, load_hf, HfKind, MixerParams, Qwen35Mixer, Qwen35Params,
    Qwen35Tables,
};
use ojas_model::{ActivationCheckpoint, Eval, DEFAULT_CE_CHUNK};
use qwen35_common::*;

#[test]
fn the_name_table_is_the_checkpoints_tower() {
    let model = SafeTensors::open(&dir().join("model.safetensors")).unwrap();
    let mut want: Vec<String> = model.names().map(str::to_string).collect();
    let mut got: Vec<String> = hf_tensors(&spec(), PREFIX)
        .into_iter()
        .map(|t| t.name)
        .collect();
    want.sort();
    got.sort();
    assert_eq!(got, want);
}

#[test]
fn the_cpu_tape_matches_transformers() {
    let r = run(cpu(), Tensor::clone, ActivationCheckpoint::Off);
    let want = reference_loss();
    let rel = (f64::from(r.loss) - want).abs() / want.abs();
    eprintln!("loss {} vs transformers {want} (rel {rel:.2e})", r.loss);
    assert!(rel <= LOSS_BOUND, "loss {} vs {want}", r.loss);
    let worst = compare_to_reference(&r.grads, GRAD_BOUND);
    eprintln!("worst gradient {worst:.3e} of its peak");
}

fn bits(p: &Qwen35Params<Tensor>) -> Vec<Vec<u32>> {
    p.values()
        .into_iter()
        .map(|t| {
            t.to_f32_vec()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect()
        })
        .collect()
}

#[test]
fn checkpointed_layers_give_the_same_bits_on_the_cpu() {
    let direct = run(cpu(), Tensor::clone, ActivationCheckpoint::Off);
    let ckpt = run(cpu(), Tensor::clone, ActivationCheckpoint::Blocks);
    assert_eq!(direct.loss.to_bits(), ckpt.loss.to_bits());
    assert_eq!(bits(&direct.grads), bits(&ckpt.grads));
}

#[test]
fn eval_gives_the_tapes_loss_bits() {
    let s = spec();
    let model = SafeTensors::open(&dir().join("model.safetensors")).unwrap();
    let reference = SafeTensors::open(&dir().join("reference.safetensors")).unwrap();
    let params = load_hf(&s, &model, PREFIX, DType::F32, DType::F32, &budget()).unwrap();
    let (ids, targets) = ids_and_targets(&reference);
    let tables = Qwen35Tables::new(&s, ids.shape()[1], &budget()).unwrap();
    let mut eval = Eval::new(cpu());
    let bound = bind(&mut eval, &params).unwrap();
    let loss = forward_loss(
        &mut eval,
        &s,
        &bound,
        &ids,
        &targets,
        &tables,
        Some(IGNORE),
        DEFAULT_CE_CHUNK,
        ActivationCheckpoint::Off,
    )
    .unwrap();
    let tape = run(cpu(), Tensor::clone, ActivationCheckpoint::Off);
    assert_eq!(loss.to_f32_vec().unwrap()[0].to_bits(), tape.loss.to_bits());
}

#[test]
fn a_mismatched_checkpoint_is_refused_by_name() {
    let model = SafeTensors::open(&dir().join("model.safetensors")).unwrap();
    let mut s = spec();
    s.layers = vec![Qwen35Mixer::Attention, Qwen35Mixer::GatedDeltaNet];
    let err = load_hf(&s, &model, PREFIX, DType::F32, DType::F32, &budget()).unwrap_err();
    assert!(
        format!("{err:?}").contains("model.layers.0.self_attn.q_proj.weight"),
        "{err:?}"
    );
    let mut s = spec();
    s.head_dim = 128;
    s.rotary_dim = 32;
    let err = load_hf(&s, &model, PREFIX, DType::F32, DType::F32, &budget()).unwrap_err();
    assert!(format!("{err:?}").contains("shape"), "{err:?}");
}

/// Every weight a `linear` reads, in one fixed order.
fn linears(p: &Qwen35Params<Tensor>) -> Vec<&Tensor> {
    let mut out = Vec::new();
    for l in &p.layers {
        match &l.mixer {
            MixerParams::GatedDeltaNet(m) => {
                out.extend([&m.wq, &m.wk, &m.wv, &m.wz, &m.wa, &m.wb, &m.out])
            }
            MixerParams::Attention(m) => out.extend([&m.wq, &m.wgate, &m.wk, &m.wv, &m.wo]),
        }
        out.extend([&l.gate, &l.up, &l.down]);
    }
    out
}

fn elems(t: &Tensor) -> u64 {
    t.num_elements().unwrap() as u64
}

#[test]
fn a_bf16_frozen_base_holds_its_linears_at_two_bytes_and_keeps_the_rest_f32() {
    let s = spec();
    let model = SafeTensors::open(&dir().join("model.safetensors")).unwrap();
    let (fb, hb) = (Budget::new(1 << 30), Budget::new(1 << 30));
    let full = load_hf(&s, &model, PREFIX, DType::F32, DType::F32, &fb).unwrap();
    let half = load_hf(&s, &model, PREFIX, DType::Bf16, DType::F32, &hb).unwrap();

    let lin_half = linears(&half);
    let lin_elems: u64 = lin_half.iter().map(|t| elems(t)).sum();
    let all_elems: u64 = half.values().iter().map(|t| elems(t)).sum();
    assert_eq!(lin_half.len(), 7 + 5 + 2 * 3, "every linear of both layers");
    for t in &lin_half {
        assert_eq!(t.dtype(), DType::Bf16);
    }
    // The embedding, the norms, the convolutions, A_log and dt_bias: f32.
    let lin_ptrs: Vec<*const Tensor> = lin_half.iter().map(|t| *t as *const Tensor).collect();
    let rest: Vec<&Tensor> = half
        .values()
        .into_iter()
        .filter(|t| !lin_ptrs.contains(&(*t as *const Tensor)))
        .collect();
    assert_eq!(rest.len(), half.values().len() - lin_half.len());
    for t in &rest {
        assert_eq!(t.dtype(), DType::F32, "shape {:?}", t.shape());
    }
    assert_eq!(half.embed.dtype(), DType::F32);

    // Resident bytes: 2 per linear element, 4 for the rest.
    assert_eq!(fb.live_bytes().unwrap(), 4 * all_elems);
    assert_eq!(
        hb.live_bytes().unwrap(),
        2 * lin_elems + 4 * (all_elems - lin_elems)
    );

    // The checkpoint is bf16: its bits are kept, and widen to the f32 load.
    for (h, f) in lin_half.iter().zip(linears(&full)) {
        let widened: Vec<u32> = h
            .bf16_slice()
            .unwrap()
            .iter()
            .map(|&b| bf16_to_f32(b).to_bits())
            .collect();
        let want: Vec<u32> = f
            .to_f32_vec()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(widened, want);
    }

    let err = load_hf(&s, &model, PREFIX, DType::U32, DType::F32, &budget()).unwrap_err();
    assert!(
        matches!(
            err,
            ojas_core::OjasError::Dtype {
                got: DType::U32,
                ..
            }
        ),
        "{err:?}"
    );
}

#[test]
fn a_bf16_frozen_base_gives_the_f32_loss_and_gradient_bits_on_the_cpu() {
    let full = run(cpu(), Tensor::clone, ActivationCheckpoint::Blocks);
    let half = run_with(
        cpu(),
        Tensor::clone,
        ActivationCheckpoint::Blocks,
        DType::Bf16,
        DType::F32,
    );
    assert_eq!(half.loss.to_bits(), full.loss.to_bits());
    assert_eq!(bits(&half.grads), bits(&full.grads));
}

#[test]
fn a_bf16_tied_table_holds_two_bytes_per_element_and_keeps_its_bits() {
    let s = spec();
    let model = SafeTensors::open(&dir().join("model.safetensors")).unwrap();
    let (fb, hb) = (Budget::new(1 << 30), Budget::new(1 << 30));
    let full = load_hf(&s, &model, PREFIX, DType::F32, DType::F32, &fb).unwrap();
    let half = load_hf(&s, &model, PREFIX, DType::Bf16, DType::Bf16, &hb).unwrap();

    assert_eq!(half.embed.dtype(), DType::Bf16);
    assert_eq!(half.embed.shape(), full.embed.shape());
    let table = elems(&half.embed);
    let lin: u64 = linears(&half).iter().map(|t| elems(t)).sum();
    let all: u64 = half.values().iter().map(|t| elems(t)).sum();
    // Resident bytes: 2 per table and linear element, 4 for the rest.
    assert_eq!(fb.live_bytes().unwrap(), 4 * all);
    assert_eq!(
        hb.live_bytes().unwrap(),
        2 * (table + lin) + 4 * (all - table - lin)
    );

    // The checkpoint is bf16: the table's bits are kept, and widen to the
    // f32 load.
    let widened: Vec<u32> = half
        .embed
        .bf16_slice()
        .unwrap()
        .iter()
        .map(|&b| bf16_to_f32(b).to_bits())
        .collect();
    let want: Vec<u32> = full
        .embed
        .to_f32_vec()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect();
    assert_eq!(widened, want);

    // The table alone may be Bf16 too; any other table storage is refused.
    let only = load_hf(&s, &model, PREFIX, DType::F32, DType::Bf16, &budget()).unwrap();
    assert_eq!(only.embed.dtype(), DType::Bf16);
    assert!(linears(&only).iter().all(|t| t.dtype() == DType::F32));
    for bad in [DType::U32, DType::F16] {
        let err = load_hf(&s, &model, PREFIX, DType::F32, bad, &budget()).unwrap_err();
        assert!(
            matches!(err, OjasError::Dtype { got, .. } if got == bad),
            "{err:?}"
        );
    }
}

#[test]
fn a_bf16_tied_table_gives_the_f32_loss_and_gradient_bits_on_the_cpu() {
    for activations in [ActivationCheckpoint::Off, ActivationCheckpoint::Blocks] {
        let full = run(cpu(), Tensor::clone, activations);
        let half = run_with(cpu(), Tensor::clone, activations, DType::Bf16, DType::Bf16);
        assert_eq!(half.loss.to_bits(), full.loss.to_bits());
        assert_eq!(bits(&half.grads), bits(&full.grads));
        assert_eq!(half.grads.embed.dtype(), DType::F32);
    }
}

fn loss_with(
    ignore: Option<u32>,
    targets_of: impl Fn(&Tensor) -> Tensor,
) -> Result<f32, OjasError> {
    let s = spec();
    let model = SafeTensors::open(&dir().join("model.safetensors")).unwrap();
    let reference = SafeTensors::open(&dir().join("reference.safetensors")).unwrap();
    let params = load_hf(&s, &model, PREFIX, DType::F32, DType::F32, &budget()).unwrap();
    let (ids, targets) = ids_and_targets(&reference);
    let targets = targets_of(&targets);
    let tables = Qwen35Tables::new(&s, ids.shape()[1], &budget()).unwrap();
    let mut eval = Eval::new(cpu());
    let bound = bind(&mut eval, &params).unwrap();
    let loss = forward_loss(
        &mut eval,
        &s,
        &bound,
        &ids,
        &targets,
        &tables,
        ignore,
        DEFAULT_CE_CHUNK,
        ActivationCheckpoint::Off,
    )?;
    Ok(loss.to_f32_vec().unwrap()[0])
}

/// Without an ignore index the `-100` labels are targets outside the
/// vocabulary: refused, never trained on or silently dropped.
#[test]
fn masked_labels_without_an_ignore_index_are_refused() {
    let err = loss_with(None, Tensor::clone).unwrap_err();
    assert!(matches!(err, OjasError::OutOfRange { .. }), "{err:?}");
}

/// The ignore index is whatever the caller says, and only it masks: the same
/// positions under another ignore value give the same bits, and scoring them
/// as token 5 instead moves the loss.
#[test]
fn only_the_ignore_index_masks_a_position() {
    let masked = loss_with(Some(IGNORE), Tensor::clone).unwrap();
    let remap = |to: u32| {
        move |t: &Tensor| {
            let v: Vec<u32> = t
                .to_u32_vec()
                .unwrap()
                .into_iter()
                .map(|x| if x == IGNORE { to } else { x })
                .collect();
            Tensor::from_u32(&v, t.shape(), &budget()).unwrap()
        }
    };
    let other = loss_with(Some(8319), remap(8319)).unwrap();
    assert_eq!(masked.to_bits(), other.to_bits());
    let scored = loss_with(None, remap(5)).unwrap();
    assert!((scored - masked).abs() > 1e-3, "{scored} vs {masked}");
}

/// A batch whose every target is ignored has no mean: refused, not zero.
#[test]
fn a_batch_with_every_target_ignored_is_refused() {
    let err = loss_with(Some(IGNORE), |t| {
        let n = t.num_elements().unwrap();
        Tensor::from_u32(&vec![IGNORE; n], t.shape(), &budget()).unwrap()
    })
    .unwrap_err();
    assert!(matches!(err, OjasError::NonFinite { .. }), "{err:?}");
}

/// The fixture's tower re-encoded as one checkpoint of `source` dtype, with
/// the first elements of every tensor replaced by awkward bit patterns (a
/// subnormal, a NaN with a payload, negative zero) that a decode must carry
/// through unchanged.
fn source_file(source: StDtype) -> Vec<u8> {
    let model = SafeTensors::open(&dir().join("model.safetensors")).unwrap();
    let (special, width): ([u32; 3], usize) = match source {
        StDtype::F32 => ([0x0000_0001, 0x7FC0_0001, 0x8000_0000], 4),
        StDtype::BF16 => ([0x0001, 0x7FC1, 0x8000], 2),
        _ => ([0x0001, 0x7E01, 0x8000], 2),
    };
    let tensors: Vec<(String, Vec<u64>, Vec<u8>)> = hf_tensors(&spec(), PREFIX)
        .into_iter()
        .map(|t| {
            let (shape, values) = model.read_f32_widened(&t.name).unwrap();
            let mut bytes = encode_f32_as(source, &values).unwrap();
            for (i, bits) in special.iter().enumerate().take(values.len()) {
                bytes[i * width..(i + 1) * width].copy_from_slice(&bits.to_le_bytes()[..width]);
            }
            (t.name, shape, bytes)
        })
        .collect();
    let items: Vec<TensorOut<'_>> = tensors
        .iter()
        .map(|(name, shape, data)| TensorOut {
            name,
            dtype: source,
            shape,
            data,
        })
        .collect();
    encode_safetensors(&items, &[]).unwrap()
}

/// Every parameter's stored bits in [`Qwen35Params::values`] order, read the
/// way `load_hf` read them before it streamed: each tensor whole through
/// `read_f32_widened` (or `read_bf16_bits`), then cut into its parts.
fn whole_tensor_oracle(file: &SafeTensors<'_>, linears: DType, table: DType) -> Vec<Vec<u32>> {
    let s = spec();
    let get = |name: &str, storage: DType| -> Vec<u32> {
        let stored = file.info(name).unwrap().dtype;
        match (storage, stored) {
            (DType::Bf16, StDtype::BF16) => file
                .read_bf16_bits(name)
                .unwrap()
                .1
                .into_iter()
                .map(u32::from)
                .collect(),
            (DType::Bf16, _) => file
                .read_f32_widened(name)
                .unwrap()
                .1
                .into_iter()
                .map(|v| u32::from(f32_to_bf16(v)))
                .collect(),
            _ => file
                .read_f32_widened(name)
                .unwrap()
                .1
                .iter()
                .map(|v| v.to_bits())
                .collect(),
        }
    };
    let rows = |v: &[u32], cols: usize, first: usize, n: usize| {
        v[first * cols..(first + n) * cols].to_vec()
    };
    let d = s.hidden;
    let mut out = vec![get(&format!("{PREFIX}embed_tokens.weight"), table)];
    for (i, kind) in s.layers.iter().enumerate() {
        let l = format!("{PREFIX}layers.{i}.");
        out.push(get(&format!("{l}input_layernorm.weight"), DType::F32));
        match kind {
            Qwen35Mixer::GatedDeltaNet => {
                let m = format!("{l}linear_attn.");
                let chans = s.gdn_qkv_channels();
                let w = get(&format!("{m}in_proj_qkv.weight"), linears);
                let c = get(&format!("{m}conv1d.weight"), DType::F32);
                let (mut at, mut parts) = (0, Vec::new());
                for &n in &chans {
                    parts.push((at, n));
                    at += n;
                }
                for &(first, n) in &parts {
                    out.push(rows(&w, d, first, n));
                }
                for &(first, n) in &parts {
                    out.push(rows(&c, s.conv_width, first, n));
                }
                for name in ["in_proj_z.weight", "in_proj_a.weight", "in_proj_b.weight"] {
                    out.push(get(&format!("{m}{name}"), linears));
                }
                for name in ["A_log", "dt_bias", "norm.weight"] {
                    out.push(get(&format!("{m}{name}"), DType::F32));
                }
                out.push(get(&format!("{m}out_proj.weight"), linears));
            }
            Qwen35Mixer::Attention => {
                let m = format!("{l}self_attn.");
                let q = get(&format!("{m}q_proj.weight"), linears);
                let block = s.head_dim * d;
                let (mut query, mut gate) = (Vec::new(), Vec::new());
                for h in 0..s.q_heads {
                    query.extend_from_slice(&q[2 * h * block..(2 * h + 1) * block]);
                    gate.extend_from_slice(&q[(2 * h + 1) * block..(2 * h + 2) * block]);
                }
                out.push(query);
                out.push(gate);
                out.push(get(&format!("{m}k_proj.weight"), linears));
                out.push(get(&format!("{m}v_proj.weight"), linears));
                out.push(get(&format!("{m}q_norm.weight"), DType::F32));
                out.push(get(&format!("{m}k_norm.weight"), DType::F32));
                out.push(get(&format!("{m}o_proj.weight"), linears));
            }
        }
        out.push(get(
            &format!("{l}post_attention_layernorm.weight"),
            DType::F32,
        ));
        for name in ["gate_proj.weight", "up_proj.weight", "down_proj.weight"] {
            out.push(get(&format!("{l}mlp.{name}"), linears));
        }
    }
    out.push(get(&format!("{PREFIX}norm.weight"), DType::F32));
    out
}

fn stored_bits(t: &Tensor) -> Vec<u32> {
    match t.dtype() {
        DType::Bf16 => t
            .bf16_slice()
            .unwrap()
            .iter()
            .map(|&b| u32::from(b))
            .collect(),
        _ => t.f32_slice().unwrap().iter().map(|v| v.to_bits()).collect(),
    }
}

/// F32, BF16 and F16 checkpoints into F32 and Bf16 storage give the bits the
/// whole-tensor read gave, subnormals, NaN payloads and negative zero
/// included.
#[test]
fn load_hf_bits_match_the_whole_tensor_read_for_every_source_and_storage() {
    for source in [StDtype::F32, StDtype::BF16, StDtype::F16] {
        let bytes = source_file(source);
        let file = SafeTensors::parse(&bytes).unwrap();
        for (linears, table) in [
            (DType::F32, DType::F32),
            (DType::Bf16, DType::Bf16),
            (DType::Bf16, DType::F32),
            (DType::F32, DType::Bf16),
        ] {
            let got = load_hf(&spec(), &file, PREFIX, linears, table, &budget()).unwrap();
            let got: Vec<Vec<u32>> = got.values().into_iter().map(stored_bits).collect();
            let want = whole_tensor_oracle(&file, linears, table);
            assert_eq!(got.len(), want.len());
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    g == w,
                    "{source:?} -> {linears:?}/{table:?}: parameter {i} differs"
                );
            }
        }
    }
}

/// The tower's charged bytes for `linears` and `table`, from the header.
fn tower_bytes(linears: DType, table: DType) -> u64 {
    hf_tensors(&spec(), PREFIX)
        .iter()
        .map(|t| {
            let storage = match t.kind {
                HfKind::Table => table,
                HfKind::Linear => linears,
                HfKind::Plain => DType::F32,
            };
            (t.shape.iter().product::<usize>() * storage.size()) as u64
        })
        .sum()
}

/// A budget one byte short of the tower is refused from the header sizes
/// alone: nothing was ever charged, so no tensor was allocated and no byte
/// was read. A budget of exactly the tower loads it.
#[test]
fn a_budget_one_byte_short_is_refused_before_any_read() {
    let bytes = source_file(StDtype::BF16);
    let file = SafeTensors::parse(&bytes).unwrap();
    for (linears, table) in [(DType::F32, DType::F32), (DType::Bf16, DType::Bf16)] {
        let total = tower_bytes(linears, table);
        let short = Budget::new(total - 1);
        let err = load_hf(&spec(), &file, PREFIX, linears, table, &short).unwrap_err();
        assert!(matches!(err, OjasError::CapacityExceeded { .. }), "{err:?}");
        assert_eq!(
            short.peak_bytes(),
            0,
            "a tensor was allocated before the refusal"
        );
        let exact = Budget::new(total);
        let params = load_hf(&spec(), &file, PREFIX, linears, table, &exact).unwrap();
        assert_eq!(exact.live_bytes().unwrap(), total);
        assert_eq!(exact.peak_bytes(), total);
        drop(params);
    }
}

/// Wall time of loading the tiny fixture, the best of five loads per storage:
/// `cargo test --release --test qwen35_fixture -- --ignored --nocapture
/// load_hf_wall_time`. Not a gate.
#[test]
#[ignore = "timing report, run by hand"]
fn load_hf_wall_time() {
    let s = spec();
    let model = SafeTensors::open(&dir().join("model.safetensors")).unwrap();
    for (linears, table) in [(DType::F32, DType::F32), (DType::Bf16, DType::Bf16)] {
        let best = (0..5)
            .map(|_| {
                let start = std::time::Instant::now();
                let p = load_hf(&s, &model, PREFIX, linears, table, &budget()).unwrap();
                let took = start.elapsed();
                drop(p);
                took
            })
            .min()
            .unwrap();
        eprintln!(
            "load_hf {linears:?}/{table:?} best of 5: {} us",
            best.as_micros()
        );
    }
}

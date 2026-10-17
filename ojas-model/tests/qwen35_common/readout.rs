//! Shared by `qwen35_readout.rs` (CPU) and `qwen35_readout_metal.rs`: the
//! readout seam composed the way Lappi's host-side answer head uses it, and
//! the torch fixture it is pinned against
//! (`tests/fixtures/qwen35_tiny_readout`, made by
//! `ojas-oracle/python/qwen35_readout_fixture.py`).
//!
//! One step ([`step`]) is: `forward_hidden_lora` to the final-norm hidden
//! state `h` `[B, T, D]`, `gather_rows` at one answer position per example,
//! `W17 = embedding(frozen_embed, answer_ids) + D17` with `D17` a trainable
//! `[17, D]` leaf, and the 17-way mean cross-entropy
//! `lin_ce(rows, W17, targets)`. Beside it, recorded as values only: each
//! example's own 17-way loss (the other examples' targets ignored) and the
//! full-vocabulary cross-entropy at the same positions with both operands
//! frozen. `h` is read back through a zero probe leaf added to it, so `dH` is
//! the probe's gradient.
#![allow(dead_code)]

use std::path::PathBuf;

use ojas_autograd::Tape;
use ojas_core::{Backend, Tensor};
use ojas_io::SafeTensors;
use ojas_model::lora::{LoraConfig, LoraSite};
use ojas_model::qwen35::{
    bind, bind_frozen, forward_hidden_lora, gather_rows, LoraForward, Qwen35Lora, Qwen35Params,
    Qwen35Tables, RightPadded,
};
use ojas_model::{ActivationCheckpoint, Graph, DEFAULT_CE_CHUNK};

use super::*;

pub const ANSWERS: usize = 17;

/// A tape value or gradient on the host, read through `Backend::download`: on
/// a GPU backend that is a sync point, so a non-finite value an op produced
/// surfaces here as `NonFinite` naming the op. Raw `Tensor::to_host` is not
/// a fault point on Metal. On CPU `download` is `to_host`.
pub fn read<B: Backend>(tape: &Tape<B>, t: &Tensor) -> Vec<f32> {
    tape.backend().download(t).unwrap().to_f32_vec().unwrap()
}

pub fn fixture() -> SafeTensors<'static> {
    let path: PathBuf = dir()
        .parent()
        .unwrap()
        .join("qwen35_tiny_readout")
        .join("readout.safetensors");
    SafeTensors::open(&path).unwrap()
}

pub fn i64s(f: &SafeTensors<'_>, name: &str) -> Vec<i64> {
    f.read_i64(name).unwrap().1
}

pub fn usizes(f: &SafeTensors<'_>, name: &str) -> Vec<usize> {
    i64s(f, name).iter().map(|&v| v as usize).collect()
}

pub fn u32s(f: &SafeTensors<'_>, name: &str) -> Vec<u32> {
    i64s(f, name).iter().map(|&v| v as u32).collect()
}

pub fn f32s(f: &SafeTensors<'_>, name: &str) -> Vec<f32> {
    f.read_f32(name).unwrap().1
}

/// What the base model is during a step.
pub enum Base<'a> {
    /// Every base parameter trainable; its gradients are the step's
    /// `param_grads`. `W17` reads a separate frozen copy of the table.
    Trainable,
    /// Every base parameter frozen ([`bind_frozen`]), `W17` reads the frozen
    /// tied table itself: only `D17` and the probe train.
    Frozen,
    /// A frozen base with these adapters (scaling from the fixture's config);
    /// the adapters' gradients are the step's `param_grads`.
    Lora(&'a Qwen35Lora<LoraSite<Tensor>>),
}

/// The head: `D17` `[17, D]`, the answer token ids, and each example's class.
pub struct Head {
    pub d17: Vec<f32>,
    pub answer_ids: Vec<u32>,
    pub targets: Vec<u32>,
}

/// Right-padded `rows`' tokens into `[B, seq]` with `pad(b, column)` after
/// each row's length.
pub fn pad_batch(
    rows: &[&[u32]],
    seq: usize,
    pad: impl Fn(usize, usize) -> u32,
) -> (Tensor, RightPadded) {
    let mut flat = Vec::new();
    for (b, r) in rows.iter().enumerate() {
        flat.extend_from_slice(r);
        flat.extend((r.len()..seq).map(|c| pad(b, c)));
    }
    let lengths = rows.iter().map(|r| r.len()).collect();
    (
        Tensor::from_u32(&flat, &[rows.len(), seq], &budget()).unwrap(),
        RightPadded::new(lengths, seq).unwrap(),
    )
}

pub struct Step {
    /// The 17-way mean cross-entropy over the batch.
    pub loss: f32,
    /// Each example's own 17-way loss.
    pub loss_each: Vec<f32>,
    /// Full-vocabulary cross-entropy at the same positions, report-only.
    pub full_loss: f32,
    /// `h`, `[B, T, D]`, and the gathered rows `[B, D]`.
    pub hidden: Vec<f32>,
    pub rows: Vec<f32>,
    /// `dH`, `[B, T, D]`.
    pub dh: Vec<f32>,
    pub d17_grad: Vec<f32>,
    /// Trainable base parameters (`values()` order) or adapters (`A`, then
    /// each `B` part, per site), by [`Base`].
    pub param_grads: Vec<Vec<f32>>,
}

/// One forward and backward on `backend`'s tape. `record_full` adds the
/// full-vocabulary node (its value is `full_loss`; `NaN` when off).
#[allow(clippy::too_many_arguments)]
pub fn step<B: Backend + Clone>(
    backend: B,
    up: &impl Fn(&Tensor) -> Tensor,
    host: &Qwen35Params<Tensor>,
    base: &Base<'_>,
    ids: &Tensor,
    rows: &RightPadded,
    positions: &[usize],
    head: &Head,
    activations: ActivationCheckpoint,
    record_full: bool,
) -> Step {
    let s = spec();
    let bud = budget();
    let (batch, seq) = (rows.batch(), rows.seq());
    let tables = Qwen35Tables::new(&s, seq, &bud).unwrap();
    let tables = Qwen35Tables {
        cos: up(&tables.cos),
        sin: up(&tables.sin),
        ones_hidden: up(&tables.ones_hidden),
        ones_head: up(&tables.ones_head),
    };
    let device = host.try_map(&mut |t| Ok::<_, ()>(up(t))).unwrap();
    let mut tape = Tape::new(backend);
    let vars = match base {
        Base::Trainable => bind(&mut tape, &device),
        _ => bind_frozen(&mut tape, &device),
    }
    .unwrap();
    let lora_fwd = match base {
        Base::Lora(sites) => {
            let bound = sites
                .try_map(&mut |site| site.try_map(&mut |t| tape.leaf(t.clone())))
                .unwrap();
            let cfg: LoraConfig = super::lora::config();
            let mut rng = ojas_data::CounterRng::new(0);
            Some(LoraForward::new(&mut tape, bound, &cfg, batch, seq, &mut rng, &bud).unwrap())
        }
        _ => None,
    };
    let h = forward_hidden_lora(
        &mut tape,
        &s,
        &vars,
        lora_fwd.as_ref(),
        ids,
        &tables,
        activations,
    )
    .unwrap();
    let hidden = read(&tape, tape.value(h).unwrap());
    let zeros = Tensor::from_f32(&vec![0.0; hidden.len()], &[batch, seq, s.hidden], &bud).unwrap();
    let probe = Graph::param(&mut tape, &up(&zeros)).unwrap();
    let hp = Graph::add(&mut tape, &h, &probe).unwrap();
    let picked = gather_rows(&mut tape, &hp, rows, positions, &bud).unwrap();
    let table = match base {
        Base::Trainable => Graph::frozen(&mut tape, &device.embed).unwrap(),
        _ => vars.embed,
    };
    let answers = Tensor::from_u32(&head.answer_ids, &[ANSWERS], &bud).unwrap();
    let fe = Graph::embedding(&mut tape, &table, &answers).unwrap();
    let d17_host = Tensor::from_f32(&head.d17, &[ANSWERS, s.hidden], &bud).unwrap();
    let d17 = Graph::param(&mut tape, &up(&d17_host)).unwrap();
    let w17 = Graph::add(&mut tape, &fe, &d17).unwrap();
    let targets = Tensor::from_u32(&head.targets, &[batch], &bud).unwrap();
    let loss = Graph::lin_ce(&mut tape, &picked, &w17, &targets, None, DEFAULT_CE_CHUNK).unwrap();
    // Each example's own loss: every other target ignored.
    let each: Vec<_> = (0..batch)
        .map(|b| {
            let only: Vec<u32> = (0..batch)
                .map(|i| if i == b { head.targets[b] } else { IGNORE })
                .collect();
            let only = Tensor::from_u32(&only, &[batch], &bud).unwrap();
            Graph::lin_ce(
                &mut tape,
                &picked,
                &w17,
                &only,
                Some(IGNORE),
                DEFAULT_CE_CHUNK,
            )
            .unwrap()
        })
        .collect();
    // Full vocabulary: both operands frozen leaves holding the same values.
    let full = record_full.then(|| {
        let rows_value = tape.value(picked).unwrap().clone();
        let frozen_rows = Graph::frozen(&mut tape, &rows_value).unwrap();
        let frozen_table = Graph::frozen(&mut tape, &device.embed).unwrap();
        let class: Vec<u32> = head
            .targets
            .iter()
            .map(|&t| head.answer_ids[t as usize])
            .collect();
        let class = Tensor::from_u32(&class, &[batch], &bud).unwrap();
        Graph::lin_ce(
            &mut tape,
            &frozen_rows,
            &frozen_table,
            &class,
            None,
            DEFAULT_CE_CHUNK,
        )
        .unwrap()
    });
    tape.backward(loss).unwrap();
    let scalar = |tape: &Tape<B>, v| read(tape, tape.value(v).unwrap())[0];
    let param_grads = match (base, &lora_fwd) {
        (Base::Trainable, _) => vars
            .values()
            .into_iter()
            .map(|v| {
                read(
                    &tape,
                    tape.grad(*v)
                        .expect("a trainable base parameter has a gradient"),
                )
            })
            .collect(),
        (Base::Lora(_), Some(fwd)) => fwd
            .sites()
            .sites()
            .iter()
            .flat_map(|site| std::iter::once(&site.a).chain(&site.b))
            .map(|v| read(&tape, tape.grad(*v).expect("an adapter has a gradient")))
            .collect(),
        _ => {
            for v in vars.values() {
                assert!(tape.grad(*v).is_none(), "a frozen parameter has a gradient");
            }
            Vec::new()
        }
    };
    Step {
        loss: scalar(&tape, loss),
        loss_each: each.iter().map(|&v| scalar(&tape, v)).collect(),
        full_loss: full.map_or(f32::NAN, |v| scalar(&tape, v)),
        hidden,
        rows: read(&tape, tape.value(picked).unwrap()),
        dh: read(&tape, tape.grad(probe).expect("dH")),
        d17_grad: read(&tape, tape.grad(d17).expect("dD17")),
        param_grads,
    }
}

pub fn host() -> Qwen35Params<Tensor> {
    super::lora::host_params()
}

/// The fixture batch: padded `ids`, its `RightPadded`, positions and head.
pub struct Case {
    pub ids: Tensor,
    pub rows: RightPadded,
    pub tokens: Vec<Vec<u32>>,
    pub positions: Vec<usize>,
    pub head: Head,
}

pub fn case(d17_on: bool) -> Case {
    let f = fixture();
    let ids = u32s(&f, "ids");
    let lengths = usizes(&f, "lengths");
    let seq = ids.len() / lengths.len();
    let tokens = lengths
        .iter()
        .enumerate()
        .map(|(b, &n)| ids[b * seq..b * seq + n].to_vec())
        .collect();
    let d17 = f32s(&f, "d17");
    Case {
        ids: Tensor::from_u32(&ids, &[lengths.len(), seq], &budget()).unwrap(),
        rows: RightPadded::new(lengths, seq).unwrap(),
        tokens,
        positions: usizes(&f, "positions"),
        head: Head {
            d17: if d17_on {
                d17.clone()
            } else {
                vec![0.0; d17.len()]
            },
            answer_ids: u32s(&f, "answer_ids"),
            targets: u32s(&f, "targets"),
        },
    }
}

fn peak(v: &[f32]) -> f32 {
    v.iter().fold(0.0f32, |p, x| p.max(x.abs()))
}

/// The worst `|got - want|` over the peak of `want`.
pub fn peak_ratio(got: &[f32], want: &[f32], what: &str) -> f32 {
    assert_eq!(got.len(), want.len(), "{what}");
    let p = peak(want);
    assert!(p > 0.0, "{what}: an all-zero reference checks nothing");
    got.iter()
        .zip(want)
        .map(|(a, b)| {
            assert!(a.is_finite(), "{what}: non-finite");
            (a - b).abs()
        })
        .fold(0.0f32, f32::max)
        / p
}

fn rel(got: f32, want: f32) -> f64 {
    f64::from((got - want).abs()) / f64::from(want.abs())
}

/// `gather_rows` of the fixture's random `[B, T, D]` at its positions is
/// torch's `index_select` of the flat rows, bit for bit, and its backward is
/// the scattered gradient of `sum(out * up)`, bit for bit, with the rows
/// nothing read exactly zero.
pub fn check_gather<B: Backend + Clone>(backend: B, up: &impl Fn(&Tensor) -> Tensor) {
    let f = fixture();
    let bud = budget();
    let c = case(true);
    let (batch, seq, d) = (c.rows.batch(), c.rows.seq(), spec().hidden);
    let mut tape = Tape::new(backend);
    let h_host = Tensor::from_f32(&f32s(&f, "gather.h"), &[batch, seq, d], &bud).unwrap();
    let h = Graph::param(&mut tape, &up(&h_host)).unwrap();
    let pick = gather_rows(&mut tape, &h, &c.rows, &c.positions, &bud).unwrap();
    let up_host = Tensor::from_f32(&f32s(&f, "gather.up"), &[batch, d], &bud).unwrap();
    let upstream = Graph::frozen(&mut tape, &up(&up_host)).unwrap();
    let prod = Graph::mul(&mut tape, &pick, &upstream).unwrap();
    tape.backward(prod).unwrap();
    let out = read(&tape, tape.value(pick).unwrap());
    let want = f32s(&f, "gather.out");
    assert_eq!(out.len(), batch * d);
    assert!(
        out.iter()
            .zip(&want)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "forward"
    );
    let dh = read(&tape, tape.grad(h).expect("dH"));
    let want = f32s(&f, "gather.dh");
    assert_eq!(dh.len(), want.len());
    assert!(
        dh.iter()
            .zip(&want)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "scattered gradient"
    );
    assert!(want.contains(&0.0) && want.iter().any(|&v| v != 0.0));
}

/// Every example reading the same position: each gets its own row of the
/// gradient, none aliases another's, and the rest stays zero.
pub fn check_same_position<B: Backend + Clone>(backend: B, up: &impl Fn(&Tensor) -> Tensor) {
    let bud = budget();
    let (batch, seq, d) = (3, 6, 8);
    let vals: Vec<f32> = (0..batch * seq * d)
        .map(|i| i as f32 * 0.25 - 7.0)
        .collect();
    let rows = RightPadded::full(batch, seq).unwrap();
    let mut tape = Tape::new(backend);
    let h = Graph::param(
        &mut tape,
        &up(&Tensor::from_f32(&vals, &[batch, seq, d], &bud).unwrap()),
    )
    .unwrap();
    let pick = gather_rows(&mut tape, &h, &rows, &[5, 5, 5], &bud).unwrap();
    let w: Vec<f32> = (0..batch * d).map(|i| 1.0 + i as f32).collect();
    let w = Graph::frozen(
        &mut tape,
        &up(&Tensor::from_f32(&w, &[batch, d], &bud).unwrap()),
    )
    .unwrap();
    let prod = Graph::mul(&mut tape, &pick, &w).unwrap();
    tape.backward(prod).unwrap();
    let got = read(&tape, tape.value(pick).unwrap());
    for b in 0..batch {
        let at = (b * seq + 5) * d;
        assert_eq!(got[b * d..(b + 1) * d], vals[at..at + d], "row {b}");
    }
    let dh = read(&tape, tape.grad(h).unwrap());
    for b in 0..batch {
        for t in 0..seq {
            let row = &dh[(b * seq + t) * d..(b * seq + t + 1) * d];
            if t == 5 {
                let want: Vec<f32> = (0..d).map(|k| 1.0 + (b * d + k) as f32).collect();
                assert_eq!(row, &want[..], "example {b}");
            } else {
                assert!(row.iter().all(|&v| v == 0.0), "({b}, {t}) is not zero");
            }
        }
    }
}

/// The composed answer head (see the module docs) against torch, on the
/// fixture batch at the fixture's non-zero `D17` and at `D17 = 0`.
pub fn check_against_torch<B: Backend + Clone>(
    backend: B,
    up: &impl Fn(&Tensor) -> Tensor,
    loss_bound: f64,
    grad_bound: f32,
) {
    let f = fixture();
    let host = host();
    let d = spec().hidden;
    let run = |c: &Case| {
        step(
            backend.clone(),
            up,
            &host,
            &Base::Frozen,
            &c.ids,
            &c.rows,
            &c.positions,
            &c.head,
            ActivationCheckpoint::Off,
            true,
        )
    };
    let on = run(&case(true));
    let off = run(&case(false));
    let (want_loss, want_off) = (f32s(&f, "row.loss")[0], f32s(&f, "row.loss_d0")[0]);
    let (e_loss, e_off) = (rel(on.loss, want_loss), rel(off.loss, want_off));
    let e_full = rel(on.full_loss, f32s(&f, "row.full_loss")[0]);
    let e_each = on
        .loss_each
        .iter()
        .zip(f32s(&f, "row.loss_each"))
        .map(|(&a, b)| rel(a, b))
        .fold(0.0f64, f64::max);
    let e_rows = peak_ratio(&on.rows, &f32s(&f, "row.h"), "h at the answer positions");
    let e_d17 = peak_ratio(&on.d17_grad, &f32s(&f, "row.d17_grad"), "dD17");
    let c = case(true);
    let (seq, want_dh) = (c.rows.seq(), f32s(&f, "row.dh"));
    let mut dh_rows = Vec::new();
    for (b, &p) in c.positions.iter().enumerate() {
        dh_rows.extend_from_slice(&on.dh[(b * seq + p) * d..(b * seq + p + 1) * d]);
        for t in (0..seq).filter(|&t| t != p) {
            assert!(on.dh[(b * seq + t) * d..(b * seq + t + 1) * d]
                .iter()
                .all(|&v| v == 0.0));
        }
    }
    let e_dh = peak_ratio(&dh_rows, &want_dh, "dH at the answer positions");
    eprintln!(
        "loss {e_loss:.2e} (D17 = 0: {e_off:.2e}), per example {e_each:.2e}, full vocab \
         {e_full:.2e}; h {e_rows:.2e}, dD17 {e_d17:.2e}, dH {e_dh:.2e} of peak"
    );
    for (what, e) in [
        ("loss", e_loss),
        ("loss at D17 = 0", e_off),
        ("full-vocab loss", e_full),
        ("per-example loss", e_each),
    ] {
        assert!(e <= loss_bound, "{what}: {e:.2e}");
    }
    for (what, e) in [("h", e_rows), ("dD17", e_d17), ("dH", e_dh)] {
        assert!(e <= grad_bound, "{what}: {e:.2e}");
    }
    // A sign error or a dropped D17 would move the loss by this much.
    assert!(
        f64::from((on.loss - off.loss).abs()) > 0.1,
        "D17 must matter"
    );
    assert!(peak(&on.d17_grad) > 0.0 && peak(&off.d17_grad) > 0.0);
}

/// The full-vocabulary cross-entropy over two frozen operands: its value is
/// torch's, and a backward from it leaves no gradient on any node.
pub fn check_full_vocab_is_inert<B: Backend + Clone>(backend: B, up: &impl Fn(&Tensor) -> Tensor) {
    let f = fixture();
    let bud = budget();
    let host = host();
    let mut tape = Tape::new(backend);
    let rows = Tensor::from_f32(&f32s(&f, "row.h"), &[4, spec().hidden], &bud).unwrap();
    let rows = Graph::frozen(&mut tape, &up(&rows)).unwrap();
    let table = Graph::frozen(&mut tape, &up(&host.embed)).unwrap();
    let answers = u32s(&f, "answer_ids");
    let class: Vec<u32> = u32s(&f, "targets")
        .iter()
        .map(|&t| answers[t as usize])
        .collect();
    let class = Tensor::from_u32(&class, &[4], &bud).unwrap();
    let loss = Graph::lin_ce(&mut tape, &rows, &table, &class, None, DEFAULT_CE_CHUNK).unwrap();
    let e = rel(
        read(&tape, tape.value(loss).unwrap())[0],
        f32s(&f, "row.full_loss")[0],
    );
    eprintln!("full-vocab loss {e:.2e} relative to torch");
    assert!(e <= LOSS_BOUND, "{e:.2e}");
    tape.backward(loss).unwrap();
    for i in 0..3 {
        assert!(
            tape.grad(ojas_autograd::Var(i)).is_none(),
            "node {i} has a gradient"
        );
    }
}

/// How close a padded batch must come to each example alone. `values` bounds
/// what the ops compute row by row (hidden states, the readout, a loss, dH);
/// `sums` bounds what is summed over the batch (parameter and `D17`
/// gradients). A bound of 0 is bit equality.
#[derive(Clone, Copy, Debug)]
pub struct Tol {
    pub values: f32,
    pub sums: f32,
}

/// Row-wise values bit-equal; a batch sum differs from the mean of the
/// examples' only by the order of the f32 additions, bounded relative to the
/// tensor's peak.
pub const SUM_ORDER_BOUND: f32 = 1e-5;
pub const EXACT: Tol = Tol {
    values: 0.0,
    sums: SUM_ORDER_BOUND,
};

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// `got == want` to `bound` of `want`'s peak, or bit for bit at 0.
pub fn assert_close(got: &[f32], want: &[f32], bound: f32, what: &str) {
    if bound == 0.0 {
        assert_eq!(bits(got), bits(want), "{what}");
    } else {
        let e = peak_ratio(got, want, what);
        assert!(e <= bound, "{what}: {e:.3e} of its peak");
    }
}

pub fn run_on<B: Backend + Clone>(
    backend: B,
    up: &impl Fn(&Tensor) -> Tensor,
    c: &Case,
    base: &Base<'_>,
    act: ActivationCheckpoint,
) -> Step {
    step(
        backend,
        up,
        &host(),
        base,
        &c.ids,
        &c.rows,
        &c.positions,
        &c.head,
        act,
        true,
    )
}

/// Example `b` of `c` run on its own, at its own length, no padding.
pub fn alone<B: Backend + Clone>(
    backend: B,
    up: &impl Fn(&Tensor) -> Tensor,
    c: &Case,
    b: usize,
    base: &Base<'_>,
) -> Step {
    let tokens = &c.tokens[b];
    let (ids, rows) = pad_batch(&[tokens], tokens.len(), |_, _| unreachable!());
    let head = Head {
        d17: c.head.d17.clone(),
        answer_ids: c.head.answer_ids.clone(),
        targets: vec![c.head.targets[b]],
    };
    step(
        backend,
        up,
        &host(),
        base,
        &ids,
        &rows,
        &[c.positions[b]],
        &head,
        ActivationCheckpoint::Off,
        false,
    )
}

/// The padded batch of the fixture against each example alone: each
/// example's loss, hidden states at every real position and readout row
/// within `tol.values`; the batch-mean loss's dH at the read position the
/// example's own dH over `B` (exact, `B = 4`) and exactly zero at every other
/// position, padding included; `D17` and parameter gradients the mean of the
/// examples' within `tol.sums`. Prints the observed worst ratios.
pub fn check_batch_against_alone<B: Backend + Clone>(
    backend: B,
    up: &impl Fn(&Tensor) -> Tensor,
    base: &Base<'_>,
    tol: Tol,
) {
    let c = case(true);
    let batch = run_on(backend.clone(), up, &c, base, ActivationCheckpoint::Off);
    let alones: Vec<Step> = (0..c.tokens.len())
        .map(|b| alone(backend.clone(), up, &c, b, base))
        .collect();
    let (seq, d) = (c.rows.seq(), spec().hidden);
    let n = c.tokens.len() as f32;
    assert_eq!(n, 4.0, "1 / B is exact, so a row's dH scales by it exactly");
    let (mut e_hidden, mut e_dh) = (0.0f32, 0.0f32);
    for (b, a) in alones.iter().enumerate() {
        let len = c.rows.lengths()[b];
        let span = b * seq * d..(b * seq + len) * d;
        assert_close(
            &batch.hidden[span.clone()],
            &a.hidden,
            tol.values,
            &format!("hidden of {b}"),
        );
        e_hidden = e_hidden.max(peak_ratio(&batch.hidden[span], &a.hidden, "h"));
        assert_close(
            &batch.rows[b * d..(b + 1) * d],
            &a.rows,
            tol.values,
            &format!("row of {b}"),
        );
        if tol.values == 0.0 {
            assert_eq!(
                batch.loss_each[b].to_bits(),
                a.loss.to_bits(),
                "loss of {b}"
            );
        } else {
            assert!(
                rel(batch.loss_each[b], a.loss) <= f64::from(tol.values),
                "loss of {b}"
            );
        }
        let p = c.positions[b];
        let want: Vec<f32> = a.dh[p * d..(p + 1) * d].iter().map(|v| v / n).collect();
        let got = &batch.dh[(b * seq + p) * d..(b * seq + p + 1) * d];
        assert_close(got, &want, tol.values, &format!("dH of {b}"));
        e_dh = e_dh.max(peak_ratio(got, &want, "dH"));
        for t in (0..seq).filter(|&t| t != p) {
            assert!(
                batch.dh[(b * seq + t) * d..(b * seq + t + 1) * d]
                    .iter()
                    .all(|&v| v == 0.0),
                "dH of {b} at {t} is not zero"
            );
        }
    }
    let mean = |pick: &dyn Fn(&Step) -> &Vec<f32>| -> Vec<f32> {
        let len = pick(&batch).len();
        assert_eq!(len, pick(&alones[0]).len());
        (0..len)
            .map(|k| alones.iter().map(|a| pick(a)[k]).sum::<f32>() / n)
            .collect()
    };
    let e_d17 = peak_ratio(&batch.d17_grad, &mean(&|s| &s.d17_grad), "dD17");
    let mut e_params = 0.0f32;
    assert!(base_has_grads(base) == !batch.param_grads.is_empty());
    for i in 0..batch.param_grads.len() {
        let want = mean(&|s| &s.param_grads[i]);
        if peak(&want) == 0.0 {
            assert!(
                batch.param_grads[i].iter().all(|&v| v == 0.0),
                "parameter {i}"
            );
            continue;
        }
        e_params = e_params.max(peak_ratio(&batch.param_grads[i], &want, "parameter"));
    }
    let mean_loss = batch.loss_each.iter().sum::<f32>() / n;
    let e_loss = rel(batch.loss, mean_loss);
    eprintln!(
        "padded batch vs alone: hidden {e_hidden:.1e}, dH {e_dh:.1e} (of peak); sums: \
         dD17 {e_d17:.3e}, parameters {e_params:.3e}; batch loss vs mean {e_loss:.1e} rel"
    );
    assert!(
        e_loss <= 1e-6,
        "batch loss {} vs mean of the examples' {mean_loss}",
        batch.loss
    );
    assert!(e_d17 <= tol.sums, "dD17 {e_d17:.3e}");
    assert!(e_params <= tol.sums, "parameter gradient {e_params:.3e}");
}

fn base_has_grads(base: &Base<'_>) -> bool {
    !matches!(base, Base::Frozen)
}

/// The same real tokens under three other paddings (zeros, the last token
/// id, and a mix): every output a row can read is unchanged to `tol.values`
/// (hidden states at every real position, rows, losses, dH, `D17` and
/// parameter gradients; at 0, bit for bit). The padding is never read, so
/// this is the causality of the whole tower, measured.
pub fn check_padding_is_inert<B: Backend + Clone>(
    backend: B,
    up: &impl Fn(&Tensor) -> Tensor,
    base: &Base<'_>,
    tol: Tol,
) {
    let c = case(true);
    let first = run_on(backend.clone(), up, &c, base, ActivationCheckpoint::Off);
    let seq = c.rows.seq();
    let refs: Vec<&[u32]> = c.tokens.iter().map(Vec::as_slice).collect();
    let fills: [Box<dyn Fn(usize, usize) -> u32>; 3] = [
        Box::new(|_, _| 0),
        Box::new(|_, _| 8319),
        Box::new(|b, col| ((b * 7919 + col * 104_729) % 8320) as u32),
    ];
    let d = spec().hidden;
    for (i, fill) in fills.into_iter().enumerate() {
        let (ids, rows) = pad_batch(&refs, seq, fill);
        let other = step(
            backend.clone(),
            up,
            &host(),
            base,
            &ids,
            &rows,
            &c.positions,
            &c.head,
            ActivationCheckpoint::Off,
            true,
        );
        let what = |s: &str| format!("{s} under padding {i}");
        assert_close(&[first.loss], &[other.loss], tol.values, &what("loss"));
        assert_close(
            &first.loss_each,
            &other.loss_each,
            tol.values,
            &what("losses"),
        );
        assert_close(&first.rows, &other.rows, tol.values, &what("rows"));
        assert_close(&first.dh, &other.dh, tol.values, &what("dH"));
        assert_close(&first.d17_grad, &other.d17_grad, tol.values, &what("dD17"));
        for (k, (a, b)) in first.param_grads.iter().zip(&other.param_grads).enumerate() {
            assert_close(a, b, tol.values, &what(&format!("parameter {k}")));
        }
        for (b, &n) in c.rows.lengths().iter().enumerate() {
            let span = b * seq * d..(b * seq + n) * d;
            assert_close(
                &first.hidden[span.clone()],
                &other.hidden[span],
                tol.values,
                &what("hidden"),
            );
        }
    }
}

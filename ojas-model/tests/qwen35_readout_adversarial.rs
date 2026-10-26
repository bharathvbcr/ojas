//! Adversarial tests for the Qwen3.5 readout seam (card ft-9c4ae3c8):
//! `RightPadded` against a reference classifier, size edges, mismatches
//! between the padded batch and the tensors it is used with, and padding
//! tokens chosen to make a non-causal leak visible. A test marked
//! `#[ignore = "BREAK: ..."]` asserts what the code should do and fails today.

mod qwen35_common;

use std::panic::catch_unwind;

use ojas_core::{Backend, OjasError, Tensor};
use ojas_cpu::CpuBackend;
use ojas_model::qwen35::{bind, forward_hidden, gather_rows, Qwen35Tables, RightPadded};
use ojas_model::{ActivationCheckpoint, Eval, Graph};
use qwen35_common::lora::{fixture_sites, lora_file, to_vec};
use qwen35_common::readout::*;
use qwen35_common::*;

// ---- RightPadded::from_attention_mask against a reference --------------------

struct Xs(u64);
impl Xs {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// The spec in the doc comment, written independently: a row is valid iff it
/// is `n >= 1` ones followed only by zeros; every value is 0 or 1.
fn reference(mask: &[i64], batch: usize, seq: usize) -> Option<Vec<usize>> {
    if mask.len() != batch * seq || mask.iter().any(|&v| v != 0 && v != 1) {
        return None;
    }
    let mut out = Vec::new();
    for row in mask.chunks(seq) {
        let ones: usize = row.iter().filter(|&&v| v == 1).count();
        let shape_ok = row.iter().enumerate().all(|(i, &v)| (v == 1) == (i < ones));
        if ones == 0 || !shape_ok {
            return None;
        }
        out.push(ones);
    }
    Some(out)
}

#[test]
fn from_attention_mask_agrees_with_a_reference_on_seeded_random_masks() {
    let mut rng = Xs(0x9E37_79B9_7F4A_7C15);
    let (mut accepted, mut refused) = (0, 0);
    for case in 0..20_000 {
        let batch = 1 + rng.below(4) as usize;
        let seq = 1 + rng.below(9) as usize;
        let mut mask = Vec::with_capacity(batch * seq);
        for _ in 0..batch {
            // Mostly valid right-padded rows, then flips to stress each refusal.
            let n = rng.below(seq as u64 + 1) as usize;
            let mut row: Vec<i64> = (0..seq).map(|i| i64::from(i < n)).collect();
            match rng.below(8) {
                0 => {
                    let i = rng.below(seq as u64) as usize;
                    row[i] ^= 1;
                }
                1 => row.reverse(),
                2 => {
                    let i = rng.below(seq as u64) as usize;
                    row[i] = [2, -1, i64::MAX, i64::MIN][rng.below(4) as usize];
                }
                _ => {}
            }
            mask.extend(row);
        }
        let bad_len = rng.below(50) == 0;
        let given_batch = if bad_len { batch + 1 } else { batch };
        let want = reference(&mask, given_batch, seq);
        let got = RightPadded::from_attention_mask(&mask, given_batch, seq);
        match (&want, &got) {
            (Some(l), Ok(r)) => {
                assert_eq!(r.lengths(), &l[..], "case {case}: {mask:?}");
                assert_eq!((r.batch(), r.seq()), (given_batch, seq));
                accepted += 1;
            }
            (None, Err(_)) => refused += 1,
            _ => panic!(
                "case {case}: mask {mask:?} {given_batch}x{seq}: reference {want:?}, got {got:?}"
            ),
        }
    }
    assert!(
        accepted > 1_000 && refused > 1_000,
        "{accepted} accepted, {refused} refused"
    );
}

#[test]
fn every_mask_of_up_to_6_columns_and_2_rows_is_classified_like_the_reference() {
    for seq in 1..=6usize {
        for code in 0..(1u32 << (2 * seq)) {
            let mask: Vec<i64> = (0..2 * seq).map(|i| i64::from((code >> i) & 1)).collect();
            let want = reference(&mask, 2, seq);
            let got = RightPadded::from_attention_mask(&mask, 2, seq);
            assert_eq!(want.is_some(), got.is_ok(), "{mask:?}");
            if let (Some(l), Ok(r)) = (want, got) {
                assert_eq!(r.lengths(), &l[..]);
            }
        }
    }
}

#[test]
fn degenerate_shapes_are_errors_not_panics() {
    for (mask, batch, seq) in [
        (vec![], 0, 0),
        (vec![], 0, 5),
        (vec![], 5, 0),
        (vec![1], 0, 1),
        (vec![1], 1, 0),
        (vec![1, 1], 1, 1),
    ] {
        let r = catch_unwind(|| RightPadded::from_attention_mask(&mask, batch, seq));
        assert!(matches!(r, Ok(Err(_))), "{mask:?} {batch}x{seq}: {r:?}");
    }
}

#[test]
fn the_flat_index_edges_are_exact() {
    // B * T = u32::MAX is accepted, one more is refused; no row is allocated.
    for (b, t) in [
        (3usize, 1_431_655_765usize),
        (5, 858_993_459),
        (1, u32::MAX as usize),
    ] {
        assert_eq!(b as u64 * t as u64, u32::MAX as u64);
        let r = RightPadded::new(vec![1; b], t).unwrap();
        // The last example's last position: flat index B*T - 1 fits.
        let mut pos = vec![0; b];
        pos[b - 1] = 0;
        assert_eq!(r.flat_ids(&pos).unwrap()[b - 1], ((b - 1) * t) as u32);
    }
    for (b, t) in [
        (3usize, 1_431_655_766usize),
        (2, 1usize << 31),
        (1, u32::MAX as usize + 1),
        (2, usize::MAX),
        (usize::MAX / 2 + 1, 2),
    ] {
        if b > 1_000 {
            continue; // a Vec of that many lengths would itself be the OOM
        }
        let r = RightPadded::new(vec![1; b], t);
        assert!(matches!(r, Err(OjasError::Shape { .. })), "{b}x{t}: {r:?}");
    }
    // The tail row at its last real position at the u32 edge.
    let t = 1_431_655_765usize;
    let r = RightPadded::new(vec![t; 3], t).unwrap();
    assert_eq!(r.flat_ids(&[0, 0, t - 1]).unwrap()[2], u32::MAX - 1);
    assert!(r.flat_ids(&[0, 0, t]).is_err());
}

/// BREAK: `RightPadded::new` documents "that is checked before anything is
/// allocated for it", but `full(batch, seq)` builds `vec![seq; batch]` first,
/// so a hostile batch size is a panic (or an allocation of `8 * batch` bytes
/// that is touched, past the 32 GiB memguard at `batch` = 2^32), not an error.
#[test]
fn full_refuses_a_huge_batch_before_allocating() {
    let r = catch_unwind(|| RightPadded::full(usize::MAX, 1));
    assert!(matches!(r, Ok(Err(OjasError::Shape { .. }))), "{r:?}");
}

/// BREAK: `from_attention_mask` calls `Vec::with_capacity(batch)` before
/// `new` checks `B * T`. With `seq == 0` the mask is empty for any `batch`, so
/// the capacity is attacker-chosen: `usize::MAX` panics with "capacity
/// overflow"; 2^40 asks malloc for 8 TiB and aborts the process.
#[test]
fn from_attention_mask_refuses_a_huge_batch_with_zero_seq_before_allocating() {
    let r = catch_unwind(|| RightPadded::from_attention_mask(&[], usize::MAX, 0));
    assert!(matches!(r, Ok(Err(_))), "{r:?}");
}

// ---- mismatches between the padded batch and the tensors -----------------------

#[test]
fn a_padded_batch_that_does_not_match_the_hidden_state_is_refused() {
    let s = spec();
    let params = host();
    let tables = Qwen35Tables::new(&s, 8, &budget()).unwrap();
    let mut eval = Eval::new(cpu());
    let bound = bind(&mut eval, &params).unwrap();
    let ids = Tensor::from_u32(&[1; 16], &[2, 8], &budget()).unwrap();
    let h = forward_hidden(
        &mut eval,
        &s,
        &bound,
        &ids,
        &tables,
        ActivationCheckpoint::Off,
    )
    .unwrap();
    let b = &budget();
    let ok = RightPadded::new(vec![8, 3], 8).unwrap();
    assert!(gather_rows(&mut eval, &h, &ok, &[7, 2], b).is_ok());
    for (what, rows, pos) in [
        (
            "batch 3",
            RightPadded::new(vec![8, 8, 8], 8).unwrap(),
            vec![0, 0, 0],
        ),
        ("batch 1", RightPadded::new(vec![8], 8).unwrap(), vec![0]),
        (
            "seq 7",
            RightPadded::new(vec![7, 7], 7).unwrap(),
            vec![0, 0],
        ),
        (
            "seq 9",
            RightPadded::new(vec![9, 9], 9).unwrap(),
            vec![0, 0],
        ),
        // Same product B*T, transposed: a flat reshape would accept it.
        ("4x4", RightPadded::new(vec![4; 4], 4).unwrap(), vec![0; 4]),
        ("1x16", RightPadded::new(vec![16], 16).unwrap(), vec![0]),
    ] {
        let r = gather_rows(&mut eval, &h, &rows, &pos, b);
        assert!(
            matches!(r, Err(OjasError::Shape { .. })),
            "{what}: {:?}",
            r.map(|_| ())
        );
    }
    // Positions: one short, one over, at and past the row length, usize::MAX.
    for pos in [
        vec![0],
        vec![0, 0, 0],
        vec![8, 0],
        vec![0, 3],
        vec![usize::MAX, 0],
        vec![],
    ] {
        assert!(gather_rows(&mut eval, &h, &ok, &pos, b).is_err(), "{pos:?}");
    }
    // A rank-2 hidden state.
    let flat = eval.reshape(&h, &[16, s.hidden]).unwrap();
    assert!(gather_rows(&mut eval, &flat, &ok, &[0, 0], b).is_err());
}

// ---- padding tokens that make a non-causal leak visible --------------------------

/// A hostile embedding row for padding id `pad`: `fill` everywhere.
fn hostile_host(pad: u32, fill: f32) -> ojas_model::qwen35::Qwen35Params<Tensor> {
    let mut p = host();
    let d = spec().hidden;
    let shape = p.embed.shape().to_vec();
    let mut v = to_vec(&p.embed);
    v[pad as usize * d..(pad as usize + 1) * d].fill(fill);
    p.embed = Tensor::from_f32(&v, &shape, &budget()).unwrap();
    p
}

fn unused_ids(c: &Case) -> (u32, u32) {
    let used: std::collections::HashSet<u32> = c
        .tokens
        .iter()
        .flatten()
        .chain(&c.head.answer_ids)
        .copied()
        .collect();
    let mut free = (0..8320u32).rev().filter(|i| !used.contains(i));
    (free.next().unwrap(), free.next().unwrap())
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Run the fixture batch padded with `pad` after each row, hostile row
/// `fill` for `pad` (the baseline pads with `calm`, whose row is untouched),
/// and require every output a row can read to be bit-identical.
fn leak_check<B: Backend + Clone>(
    backend: B,
    up: &impl Fn(&Tensor) -> Tensor,
    base: &Base<'_>,
    act: ActivationCheckpoint,
    fill: f32,
) -> (bool, bool) {
    let c = case(true);
    let (pad, calm) = unused_ids(&c);
    let seq = c.rows.seq();
    let d = spec().hidden;
    let refs: Vec<&[u32]> = c.tokens.iter().map(Vec::as_slice).collect();
    let run = |host_params, fill_id| {
        let (ids, rows) = pad_batch(&refs, seq, |_, _| fill_id);
        step(
            backend.clone(),
            up,
            &host_params,
            base,
            &ids,
            &rows,
            &c.positions,
            &c.head,
            act,
            false,
        )
    };
    let calm_run = run(host(), calm);
    let hot = run(hostile_host(pad, fill), pad);
    let what = format!("pad row {fill:e}");
    for (b, &n) in c.rows.lengths().iter().enumerate() {
        let span = b * seq * d..(b * seq + n) * d;
        assert_eq!(
            bits(&hot.hidden[span.clone()]),
            bits(&calm_run.hidden[span]),
            "{what}: hidden of row {b} at its real positions"
        );
    }
    assert_eq!(bits(&hot.rows), bits(&calm_run.rows), "{what}: rows");
    assert_eq!(hot.loss.to_bits(), calm_run.loss.to_bits(), "{what}: loss");
    assert_eq!(
        bits(&hot.d17_grad),
        bits(&calm_run.d17_grad),
        "{what}: dD17"
    );
    // dH: the gathered row's gradient must be zero at every padding position.
    for (b, &n) in c.rows.lengths().iter().enumerate() {
        for t in n..seq {
            let row = &hot.dh[(b * seq + t) * d..(b * seq + t + 1) * d];
            assert!(
                row.iter().all(|&v| v == 0.0),
                "{what}: dH at padding ({b},{t})"
            );
        }
    }
    assert_eq!(bits(&hot.dh), bits(&calm_run.dh), "{what}: dH");
    let finite = hot.param_grads.iter().flatten().all(|v| v.is_finite());
    let same = hot.param_grads.len() == calm_run.param_grads.len()
        && hot
            .param_grads
            .iter()
            .zip(&calm_run.param_grads)
            .all(|(a, b)| bits(a) == bits(b));
    (finite, same)
}

fn cpu_backend() -> CpuBackend {
    cpu()
}

#[test]
fn a_large_norm_padding_row_does_not_reach_the_readout_on_cpu() {
    let sites = fixture_sites(&lora_file(), true);
    for act in [ActivationCheckpoint::Off, ActivationCheckpoint::Blocks] {
        for base in [Base::Frozen, Base::Trainable, Base::Lora(&sites)] {
            for fill in [1.0e3f32, 1.0e18] {
                let (finite, same) = leak_check(cpu_backend(), &Tensor::clone, &base, act, fill);
                assert!(
                    finite && same,
                    "{fill:e} {act:?}: finite {finite}, equal {same}"
                );
            }
        }
    }
}

/// A padding row of f32::MAX scale overflows `rms_norm` and makes the CPU step fail
/// loudly (`NonFinite`) for the whole batch rather than reach a real row:
/// fail closed. (`leak_check` unwraps `step`, so the error is the panic.)
#[test]
#[should_panic(expected = "NonFinite")]
fn an_overflowing_padding_row_fails_closed_on_cpu() {
    leak_check(
        cpu_backend(),
        &Tensor::clone,
        &Base::Frozen,
        ActivationCheckpoint::Off,
        3.0e38,
    );
}

#[test]
#[should_panic(expected = "NonFinite")]
fn a_nan_padding_row_fails_closed_on_cpu() {
    leak_check(
        cpu_backend(),
        &Tensor::clone,
        &Base::Trainable,
        ActivationCheckpoint::Blocks,
        f32::NAN,
    );
}

#[cfg(target_os = "macos")]
mod metal {
    use super::*;
    use ojas_core::Budget;
    use ojas_metal::MetalBackend;

    fn device() -> Option<MetalBackend> {
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
    fn a_large_norm_padding_row_does_not_reach_the_readout_on_metal() {
        let Some(m) = device() else { return };
        let up = |t: &Tensor| m.upload(t).unwrap();
        let sites = fixture_sites(&lora_file(), true);
        for fill in [1.0e3f32, 1.0e18] {
            for act in [ActivationCheckpoint::Off, ActivationCheckpoint::Blocks] {
                for base in [Base::Frozen, Base::Lora(&sites)] {
                    let (finite, same) = leak_check(m.clone(), &up, &base, act, fill);
                    assert!(
                        finite && same,
                        "{fill:e} {act:?}: finite {finite}, equal {same}"
                    );
                }
            }
        }
    }

    /// Runs the leak check with a non-finite padding row and asserts it panics
    /// with `NonFinite`. A `#[should_panic]` test cannot do this: when the
    /// runner has no Metal 4 device (`OJAS_ALLOW_NO_GPU=1`) the test skips by
    /// returning early, which `#[should_panic]` reports as a failure.
    fn fails_closed_on_metal(fill: f32) {
        let Some(m) = device() else { return };
        let up = |t: &Tensor| m.upload(t).unwrap();
        let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            leak_check(
                m.clone(),
                &up,
                &Base::Frozen,
                ActivationCheckpoint::Off,
                fill,
            )
        }))
        .expect_err("a non-finite padding row must fail closed");
        let msg = err
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| err.downcast_ref::<&str>().map(|s| (*s).to_owned()))
            .unwrap_or_default();
        assert!(msg.contains("NonFinite"), "unexpected panic: {msg}");
    }

    /// A NaN embedding row at a padding id fails closed on Metal as on CPU:
    /// `embedding_forward` records the fault, and it surfaces as `NonFinite`
    /// naming the op at the first `Backend::download` (a sync point) the step
    /// reads through. Raw `Tensor::to_host` would not report it.
    #[test]
    fn a_nan_padding_row_fails_closed_on_metal() {
        fails_closed_on_metal(f32::NAN);
    }

    #[test]
    fn an_infinite_padding_row_fails_closed_on_metal() {
        fails_closed_on_metal(f32::INFINITY);
    }
}

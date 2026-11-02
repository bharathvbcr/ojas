//! The mask features no `AttentionOpts` field reaches yet: the score `scale`,
//! the symmetric `radius`, and dead query rows. They are driven here on the
//! device command, against the CPU backward under an equivalent call: a scale
//! by pre-scaling Q, a radius by a finite bias of -1e4 on the hidden entries
//! (its probability is exactly 0 in f32), and dead rows by a zeroed
//! gradient on the CPU's side. The CPU supports all of those today.

use super::*;
use crate::link::Waits;
use ojas_core::{sdpa_scale, AttentionOpts, Backend, Budget, Numerics, Tensor};
use std::sync::Arc;

const B: usize = 2;
const H: usize = 4;
const HKV: usize = 2;
const D: usize = 32;

fn pattern(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        })
        .collect()
}

fn host(v: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_f32(v, shape, &Budget::new(1 << 30)).unwrap()
}

fn upload(w: &mut Worker, src: Tensor, n: usize) -> Arg {
    match w.run(Cmd::Upload { src }) {
        Ok(Reply::Bufs(b)) if b.len() == 1 => Arg {
            id: b[0].id,
            off: 0,
            n,
        },
        other => panic!("upload: {other:?}"),
    }
}

fn up_f32(w: &mut Worker, v: &[f32]) -> Arg {
    upload(w, host(v, &[v.len()]), v.len())
}

fn read(w: &mut Worker, id: u64, n: usize) -> Vec<f32> {
    match w.run(Cmd::Read {
        id,
        off: 0,
        len: n * 4,
    }) {
        Ok(Reply::Bytes(b)) => b
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_ne_bytes(*c))
            .collect(),
        other => panic!("read: {other:?}"),
    }
}

fn cpu() -> ojas_cpu::CpuBackend {
    ojas_cpu::CpuBackend::new(Budget::new(8 << 30)).with_numerics(Numerics::Exact)
}

fn close(what: &str, got: &[f32], want: &[f32], t: usize) {
    let atol = 2e-5 * (t as f32).sqrt();
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, e)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - e).abs() <= atol + 1e-3 * e.abs(),
            "{what}[{i}]: got {g} want {e}"
        );
    }
}

/// What the CPU gives for one call: the forward's output and log-sum-exp,
/// then the gradients.
struct Reference {
    out: Vec<f32>,
    lse: Vec<f32>,
    dq: Vec<f32>,
    dk: Vec<f32>,
    dv: Vec<f32>,
    dbias: Option<Vec<f32>>,
}

fn reference(
    [q, k, v, gy]: [&[f32]; 4],
    bias: Option<(&[f32], [usize; 4])>,
    lens: Option<&[u32]>,
    [tq, tk]: [usize; 2],
) -> Reference {
    let c = cpu();
    let budget = Budget::new(1 << 30);
    let (qt, kt, vt, gt) = (
        host(q, &[B, H, tq, D]),
        host(k, &[B, HKV, tk, D]),
        host(v, &[B, HKV, tk, D]),
        host(gy, &[B, H, tq, D]),
    );
    let bias_t = bias.map(|(b, shape)| host(b, &shape));
    let lens_t = lens.map(|l| Tensor::from_u32(l, &[B], &budget).unwrap());
    let opts = AttentionOpts {
        bias: bias_t.as_ref(),
        key_lens: lens_t.as_ref(),
        ..Default::default()
    };
    let (out, lse) = c.attention_forward_saving(&qt, &kt, &vt, opts).unwrap();
    let g = c
        .attention_backward(&qt, &kt, &vt, &out, &lse, &gt, opts, bias.is_some())
        .unwrap();
    let f = |t: &Tensor| t.to_f32_vec().unwrap();
    Reference {
        out: f(&out),
        lse: f(&lse),
        dq: f(&g.q),
        dk: f(&g.k),
        dv: f(&g.v),
        dbias: g.bias.as_ref().map(f),
    }
}

struct Device {
    w: Worker,
}

impl Device {
    fn new() -> Self {
        Device {
            w: Worker::open(Arc::new(Waits::default()), ojas_core::Budget::new(8 << 30))
                .expect("Metal device"),
        }
    }

    /// Run the device backward and read dQ, dK, dV and the bias gradient.
    fn backward(
        &mut self,
        [q, k, v, o, lse, gy]: [&[f32]; 6],
        bias: Option<(&[f32], [u32; 3])>,
        lens: Option<&[u32]>,
        [tq, tk]: [usize; 2],
        tweak: impl FnOnce(&mut AttnBwdGeom),
    ) -> Res<Vec<Vec<f32>>> {
        let (qa, ka, va) = (
            up_f32(&mut self.w, q),
            up_f32(&mut self.w, k),
            up_f32(&mut self.w, v),
        );
        let (oa, la, ga) = (
            up_f32(&mut self.w, o),
            up_f32(&mut self.w, lse),
            up_f32(&mut self.w, gy),
        );
        let bias_arg = bias.map(|(b, _)| up_f32(&mut self.w, b));
        let lens_arg = lens.map(|l| {
            let src = Tensor::from_u32(l, &[l.len()], &Budget::new(1 << 20)).unwrap();
            upload(&mut self.w, src, l.len())
        });
        let mut geom = AttnBwdGeom {
            attn: AttnGeom {
                bh: (B * H) as u32,
                heads: H as u32,
                rep: (H / HKV) as u32,
                q_len: tq as u32,
                kv_len: tk as u32,
                d: D as u32,
                causal: false,
                window: 0,
                bias: bias.map(|(_, shape)| shape),
                has_lens: lens.is_some(),
                splits: 0,
                per_split: 0,
                flash: false,
            },
            scale: sdpa_scale(D as u32).unwrap(),
            radius: 0,
            dead_rows: false,
            bias_grad: bias.is_some(),
        };
        tweak(&mut geom);
        let outs = match self.w.run(Cmd::AttnBwd {
            op: "attention_backward",
            args: [qa, ka, va, oa, la, ga],
            bias: bias_arg,
            lens: lens_arg,
            geom,
        })? {
            Reply::Bufs(b) => b,
            other => panic!("attn bwd: {other:?}"),
        };
        let sizes = [
            B * H * tq * D,
            B * HKV * tk * D,
            B * HKV * tk * D,
            bias.map_or(0, |(b, _)| b.len()),
        ];
        Ok(outs
            .iter()
            .zip(sizes)
            .map(|(buf, n)| read(&mut self.w, buf.id, n))
            .collect())
    }
}

/// A score scale other than `D^-0.5`: the CPU runs Q pre-scaled by the ratio,
/// which gives the same scores; dQ scales back by it.
#[test]
fn the_score_scale_is_the_calls() {
    let mut dev = Device::new();
    let (tq, tk) = (45usize, 70usize);
    let q = pattern(B * H * tq * D, 1);
    let k = pattern(B * HKV * tk * D, 2);
    let v = pattern(B * HKV * tk * D, 3);
    let gy = pattern(B * H * tq * D, 4);
    let ratio = 1.0 / sdpa_scale(D as u32).unwrap();
    let qs: Vec<f32> = q.iter().map(|x| x * ratio).collect();
    // Scale 1: the CPU's scores are (q * ratio) . k * D^-0.5 = q . k.
    let want = reference([&qs, &k, &v, &gy], None, None, [tq, tk]);
    let got = dev
        .backward(
            [&q, &k, &v, &want.out, &want.lse, &gy],
            None,
            None,
            [tq, tk],
            |g| g.scale = 1.0,
        )
        .unwrap();
    let t = tq.max(tk);
    // dQ of the unscaled Q is the CPU's (with respect to the scaled one) times
    // the ratio.
    let dq: Vec<f32> = want.dq.iter().map(|x| x * ratio).collect();
    close("dq", &got[0], &dq, t);
    close("dk", &got[1], &want.dk, t);
    close("dv", &got[2], &want.dv, t);
}

/// `|i - j| <= R` is visible: the CPU hides the rest with a bias of -1e4.
#[test]
fn a_radius_keeps_the_keys_within_it() {
    let mut dev = Device::new();
    let t = 70usize;
    for radius in [1usize, 5, 33, 69] {
        let q = pattern(B * H * t * D, 10 + radius as u64);
        let k = pattern(B * HKV * t * D, 11);
        let v = pattern(B * HKV * t * D, 12);
        let gy = pattern(B * H * t * D, 13);
        // A real bias [B, 1, T, T] plus the mask, so the bias gradient is
        // checked under the radius too.
        let real = pattern(B * t * t, 14);
        let mut total = real.clone();
        for b in 0..B {
            for i in 0..t {
                for j in 0..t {
                    if i.abs_diff(j) > radius {
                        total[(b * t + i) * t + j] -= 1e4;
                    }
                }
            }
        }
        let want = reference(
            [&q, &k, &v, &gy],
            Some((&total, [B, 1, t, t])),
            None,
            [t, t],
        );
        let got = dev
            .backward(
                [&q, &k, &v, &want.out, &want.lse, &gy],
                Some((&real, [B as u32, 1, t as u32])),
                None,
                [t, t],
                |g| g.radius = radius as u32,
            )
            .unwrap();
        close(&format!("R{radius} dq"), &got[0], &want.dq, t);
        close(&format!("R{radius} dk"), &got[1], &want.dk, t);
        close(&format!("R{radius} dv"), &got[2], &want.dv, t);
        close(
            &format!("R{radius} dbias"),
            &got[3],
            want.dbias.as_deref().unwrap(),
            t,
        );
    }
}

/// A query row past its batch row's key length is dead: no key, no gradient.
/// The CPU gets dO zeroed there; the device gets it whole.
#[test]
fn query_rows_past_the_key_length_have_no_gradient() {
    let mut dev = Device::new();
    let t = 70usize;
    let lens = [40u32, 70];
    let q = pattern(B * H * t * D, 21);
    let k = pattern(B * HKV * t * D, 22);
    let v = pattern(B * HKV * t * D, 23);
    let gy = pattern(B * H * t * D, 24);
    let dead = |plane: usize, i: usize| i >= lens[plane / H] as usize;
    let mut gy_live = gy.clone();
    for p in 0..B * H {
        for i in 0..t {
            if dead(p, i) {
                gy_live[(p * t + i) * D..(p * t + i + 1) * D].fill(0.0);
            }
        }
    }
    let want = reference([&q, &k, &v, &gy_live], None, Some(&lens), [t, t]);
    // The contract's forward: a dead row has output 0 and lse 0.
    let (mut out, mut lse) = (want.out.clone(), want.lse.clone());
    for p in 0..B * H {
        for i in 0..t {
            if dead(p, i) {
                out[(p * t + i) * D..(p * t + i + 1) * D].fill(0.0);
                lse[p * t + i] = 0.0;
            }
        }
    }
    let got = dev
        .backward(
            [&q, &k, &v, &out, &lse, &gy],
            None,
            Some(&lens),
            [t, t],
            |g| g.dead_rows = true,
        )
        .unwrap();
    close("dq", &got[0], &want.dq, t);
    close("dk", &got[1], &want.dk, t);
    close("dv", &got[2], &want.dv, t);
}

/// A radius or dead rows outside their contract are refused before anything
/// runs: a radius needs bidirectional self-attention, dead rows need key
/// lengths and `Tq == Tk`.
#[test]
fn a_radius_or_dead_rows_outside_their_contract_are_refused() {
    let mut dev = Device::new();
    let (tq, tk) = (6usize, 8usize);
    let q = pattern(B * H * tq * D, 1);
    let kv = pattern(B * HKV * tk * D, 2);
    let lse = vec![0.0; B * H * tq];
    let mut call = |lens: Option<&[u32]>, shape: [usize; 2], tweak: fn(&mut AttnBwdGeom)| {
        let (q, kv) = (&q[..B * H * shape[0] * D], &kv[..B * HKV * shape[1] * D]);
        dev.backward(
            [q, kv, kv, q, &lse[..B * H * shape[0]], q],
            None,
            lens,
            shape,
            tweak,
        )
    };
    // Tq != Tk.
    assert!(call(None, [tq, tk], |g| g.radius = 2).is_err());
    assert!(call(Some(&[3, 4]), [tq, tk], |g| g.dead_rows = true).is_err());
    // Tq == Tk, but no key lengths.
    assert!(call(None, [tq, tq], |g| g.dead_rows = true).is_err());
}

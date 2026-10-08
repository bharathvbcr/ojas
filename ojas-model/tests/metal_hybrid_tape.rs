//! A Qwen3.5 linear-attention layer's ops on the tape, on Metal: causal
//! conv1d + SiLU in front of q, k and v, the gated delta rule, the gated
//! RMSNorm on its output, and partial RoPE with text-only MRoPE tables. The
//! forward and every leaf gradient must match the CPU tape, and a
//! checkpointed segment must give the direct recording's bits on Metal.
//!
//! This is the layer at test size (the kernels' key dim 128, one value
//! block), not the 2B model. A missing device fails unless
//! `OJAS_ALLOW_NO_GPU=1`.

#![cfg(target_os = "macos")]

use ojas_autograd::{Tape, Var};
use ojas_core::{
    mrope_text_tables, Backend, Budget, MropeSection, Numerics, OjasError, Tensor,
    METAL_GDN_KEY_DIM,
};
use ojas_cpu::CpuBackend;
use ojas_metal::MetalBackend;

const B: usize = 1;
const T: usize = 70;
const H: usize = 2;
const DK: usize = METAL_GDN_KEY_DIM;
const DV: usize = 16;
const K: usize = 4;
const R: usize = 8;
const EPS: f32 = 1e-6;
/// `max |metal - cpu| / max |cpu|` per tensor: the gated delta rule's bound
/// in `ojas-metal/tests/gdn.rs`.
const BOUND: f32 = 2e-4;

fn vals(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            lo + (hi - lo) * ((s >> 40) as f32 / (1u64 << 24) as f32)
        })
        .collect()
}

fn f(seed: u64, shape: &[usize], lo: f32, hi: f32) -> Tensor {
    let n = shape.iter().product();
    Tensor::from_f32(&vals(seed, n, lo, hi), shape, &Budget::new(1 << 30)).unwrap()
}

/// Host leaves, in order: xq, wq, xk, wk, xv, wv, g, beta, z, nw.
fn leaves() -> Vec<Tensor> {
    let (cq, cv) = (H * DK, H * DV);
    vec![
        f(1, &[B, T, cq], -1.0, 1.0),
        f(2, &[cq, K], -0.5, 0.5),
        f(3, &[B, T, cq], -1.0, 1.0),
        f(4, &[cq, K], -0.5, 0.5),
        f(5, &[B, T, cv], -1.0, 1.0),
        f(6, &[cv, K], -0.5, 0.5),
        f(7, &[B, T, H], -1.0, -0.05),
        f(8, &[B, T, H], 0.1, 0.9),
        f(9, &[B, T, H, DV], -2.0, 2.0),
        f(10, &[DV], 0.5, 1.5),
    ]
}

fn tables() -> (Tensor, Tensor) {
    let qwen = MropeSection {
        section: [2, 1, 1],
        interleaved: true,
    };
    mrope_text_tables(0, T, qwen, R, 1e7, &Budget::new(1 << 30)).unwrap()
}

fn layer<Bk: Backend>(t: &mut Tape<Bk>, v: &[Var]) -> Result<Var, OjasError> {
    let q = t.causal_conv1d_silu(v[0], v[1])?;
    let q = t.reshape(q, &[B, T, H, DK])?;
    let k = t.causal_conv1d_silu(v[2], v[3])?;
    let k = t.reshape(k, &[B, T, H, DK])?;
    let vv = t.causal_conv1d_silu(v[4], v[5])?;
    let vv = t.reshape(vv, &[B, T, H, DV])?;
    let o = t.chunked_gdn(q, k, vv, v[6], v[7])?;
    let o = t.gated_rms_norm(o, v[8], v[9], EPS)?;
    let (cos, sin) = tables();
    t.rope_partial(o, cos, sin)
}

/// The forward and every leaf gradient, on the host.
fn run<Bk: Backend>(be: Bk, up: impl Fn(&Tensor) -> Tensor, checkpoint: bool) -> Vec<Vec<f32>> {
    let mut t = Tape::new(be);
    let v: Vec<Var> = leaves().iter().map(|x| t.leaf(up(x)).unwrap()).collect();
    let y = if checkpoint {
        t.checkpoint(|t| layer(t, &v).map(|y| vec![y])).unwrap()[0]
    } else {
        layer(&mut t, &v).unwrap()
    };
    let r = t.leaf(up(&f(11, &[B, T, H, DV], -1.0, 1.0))).unwrap();
    let loss = t.mul(y, r).unwrap();
    t.backward(loss).unwrap();
    let host = |x: &Tensor| {
        x.to_host(&Budget::new(1 << 30))
            .unwrap()
            .to_f32_vec()
            .unwrap()
    };
    let mut out = vec![host(t.value(y).unwrap())];
    out.extend(v.iter().map(|&v| host(t.grad(v).unwrap())));
    out
}

fn metal() -> Option<MetalBackend> {
    match MetalBackend::new(Budget::new(8 << 30)) {
        Ok(m) => Some(m),
        Err(e) if std::env::var_os("OJAS_ALLOW_NO_GPU").is_some() => {
            eprintln!("no Metal device: {e:?}");
            None
        }
        Err(e) => panic!("no Metal device (set OJAS_ALLOW_NO_GPU=1 to skip): {e:?}"),
    }
}

const NAMES: [&str; 11] = [
    "y", "dxq", "dwq", "dxk", "dwk", "dxv", "dwv", "dg", "dbeta", "dz", "dnw",
];

#[test]
fn the_layer_on_metal_matches_the_cpu_tape() {
    let Some(m) = metal() else { return };
    let cpu = CpuBackend::new(Budget::new(1 << 30)).with_numerics(Numerics::Exact);
    let want = run(cpu, Tensor::clone, false);
    let got = run(m.clone(), |x| m.upload(x).unwrap(), false);
    for ((name, g), w) in NAMES.iter().zip(&got).zip(&want) {
        let peak = w.iter().fold(0.0f32, |p, x| p.max(x.abs()));
        assert!(peak > 0.0, "{name}: an all-zero reference checks nothing");
        let worst = g
            .iter()
            .zip(w)
            .map(|(a, b)| {
                assert!(a.is_finite(), "{name}: non-finite");
                (a - b).abs()
            })
            .fold(0.0f32, f32::max);
        eprintln!("{name}: {:.3e} of peak {peak:.3e}", worst / peak);
        assert!(
            worst / peak <= BOUND,
            "{name}: {:.3e} of peak",
            worst / peak
        );
    }
}

#[test]
fn a_checkpointed_layer_on_metal_gives_the_same_bits() {
    let Some(m) = metal() else { return };
    let bits = |r: Vec<Vec<f32>>| -> Vec<Vec<u32>> {
        r.into_iter()
            .map(|v| v.iter().map(|x| x.to_bits()).collect())
            .collect()
    };
    let direct = bits(run(m.clone(), |x| m.upload(x).unwrap(), false));
    let ckpt = bits(run(m.clone(), |x| m.upload(x).unwrap(), true));
    for ((name, a), b) in NAMES.iter().zip(&ckpt).zip(&direct) {
        assert!(a == b, "{name}: checkpointing changed the bits");
    }
}

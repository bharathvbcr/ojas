//! One MLP training step on an MNIST batch that comes from the ojas-data
//! reader and sampler, on Metal and on the CPU reference. The idx fixture is
//! generated here; nothing is downloaded. Only existing ops run: linear,
//! SiLU, cross-entropy, AdamW.

#![cfg(target_os = "macos")]

mod common;

use common::*;
use ojas_core::{AdamWConfig, Backend, BackendId, Budget, OjasError, Tensor};
use ojas_data::{Augment, ImageBatchSampler, ImageSamplerConfig, ImageSet, Normalization};
use std::io::Cursor;

const N: usize = 96;
const BATCH: usize = 32;
const PIXELS: usize = 28 * 28;
const HIDDEN: usize = 64;
const CLASSES: usize = 10;
/// Relative loss tolerance for Metal's `Numerics::Fast` against the CPU
/// reference: the figure `training_step.rs` uses for a loss that has passed
/// through backward and an optimizer update.
const LOSS_RTOL: f32 = 1e-4;

/// An idx image/label pair of `N` 28 x 28 images with noisy pixels.
fn idx_fixture() -> (Vec<u8>, Vec<u8>) {
    let mut img = vec![0, 0, 8, 3];
    for d in [N as u32, 28, 28] {
        img.extend_from_slice(&d.to_be_bytes());
    }
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    for _ in 0..N * PIXELS {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        img.push((s >> 33) as u8);
    }
    let mut lab = vec![0, 0, 8, 1];
    lab.extend_from_slice(&(N as u32).to_be_bytes());
    lab.extend((0..N).map(|i| (i % CLASSES) as u8));
    (img, lab)
}

struct Net {
    w1: Tensor,
    w2: Tensor,
    m: [(Tensor, Tensor); 2],
}

fn zeros(shape: &[usize]) -> Tensor {
    host(&vec![0.0; shape.iter().product()], shape)
}

impl Net {
    fn host() -> Self {
        Self {
            w1: rand(&[HIDDEN, PIXELS], 1, 0.05),
            w2: rand(&[CLASSES, HIDDEN], 2, 0.2),
            m: [
                (zeros(&[HIDDEN, PIXELS]), zeros(&[HIDDEN, PIXELS])),
                (zeros(&[CLASSES, HIDDEN]), zeros(&[CLASSES, HIDDEN])),
            ],
        }
    }

    fn on(&self, f: impl Fn(&Tensor) -> Tensor) -> Self {
        Self {
            w1: f(&self.w1),
            w2: f(&self.w2),
            m: [
                (f(&self.m[0].0), f(&self.m[0].1)),
                (f(&self.m[1].0), f(&self.m[1].1)),
            ],
        }
    }
}

fn loss<Bk: Backend>(b: &Bk, n: &Net, x: &Tensor, y: &Tensor) -> Result<f32, OjasError> {
    let h = b.silu_forward(&b.linear_forward(x, &n.w1)?)?;
    let logits = b.linear_forward(&h, &n.w2)?;
    let l = b.cross_entropy_mean_forward(&logits, y, None)?;
    Ok(b.download(&l)?.to_f32_vec()?[0])
}

/// Forward, backward and one AdamW update. Returns the loss before it.
fn step<Bk: Backend>(b: &Bk, n: &mut Net, x: &Tensor, y: &Tensor) -> Result<f32, OjasError> {
    let pre = b.linear_forward(x, &n.w1)?;
    let h = b.silu_forward(&pre)?;
    let logits = b.linear_forward(&h, &n.w2)?;
    let before = b.cross_entropy_mean_forward(&logits, y, None)?;
    let glogits = b.cross_entropy_mean_backward(&logits, y, None)?;
    let (gh, gw2) = b.linear_backward(&h, &n.w2, &glogits)?;
    let gpre = b.silu_backward(&pre, &gh)?;
    let (_, gw1) = b.linear_backward(x, &n.w1, &gpre)?;
    let cfg = AdamWConfig::nanolab(1e-2, 0.0);
    let [(m1a, m2a), (m1b, m2b)] = &mut n.m;
    b.adamw_step(&mut n.w1, &gw1, m1a, m2a, 0, cfg)?;
    b.adamw_step(&mut n.w2, &gw2, m1b, m2b, 0, cfg)?;
    if b.id() == BackendId::Metal {
        for t in [&pre, &h, &logits, &gh, &gw1, &n.w1] {
            assert_eq!(t.device(), Some(BackendId::Metal), "left the device");
        }
    }
    Ok(b.download(&before)?.to_f32_vec()?[0])
}

#[test]
fn an_mlp_step_on_a_reader_batch_matches_cpu_within_the_fast_tolerance() {
    let (img, lab) = idx_fixture();
    let set =
        ImageSet::from_mnist(Cursor::new(img), Cursor::new(lab), &Budget::new(1 << 24)).unwrap();
    let cfg = ImageSamplerConfig {
        batch: BATCH,
        seed: 7,
        drop_last: true,
        normalization: Normalization::MNIST,
        augment: Augment::NONE,
    };
    let mut sampler = ImageBatchSampler::new(&set, cfg).unwrap();
    let (mut xs, mut ys) = (vec![0.0f32; BATCH * PIXELS], vec![0u32; BATCH]);
    assert_eq!(sampler.next_batch(&mut xs, &mut ys).unwrap(), BATCH);
    let (x, y) = (host(&xs, &[BATCH, PIXELS]), host_u32(&ys, &[BATCH]));

    let (m, c) = (metal(), cpu());
    let mut hn = Net::host();
    let mut dn = hn.on(|t| up(&m, t));
    let (dx, dy) = (up(&m, &x), up(&m, &y));

    let cpu_before = ok("cpu step", step(&c, &mut hn, &x, &y));
    let metal_before = ok("metal step", step(&m, &mut dn, &dx, &dy));
    let cpu_after = ok("cpu loss", loss(&c, &hn, &x, &y));
    let metal_after = ok("metal loss", loss(&m, &dn, &dx, &dy));
    close(
        "loss before the step",
        &[metal_before],
        &[cpu_before],
        0.0,
        LOSS_RTOL,
    );
    close(
        "loss after the step",
        &[metal_after],
        &[cpu_after],
        0.0,
        LOSS_RTOL,
    );
    assert!(
        cpu_after < cpu_before && metal_after < metal_before,
        "the step did not lower the loss: cpu {cpu_before} -> {cpu_after}, metal {metal_before} -> {metal_after}"
    );
    eprintln!("cpu {cpu_before} -> {cpu_after}, metal {metal_before} -> {metal_after}");
}

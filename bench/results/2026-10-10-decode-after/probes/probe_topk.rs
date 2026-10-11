//! Throwaway probe: where device top-k stops paying on one `[1, 50304]`
//! logit row. For each `k`, interleaved, min of `ITERS`:
//! (a) `topk_rows(k)`, download values and ids, `sync`;
//! (b) download the row, `sync`, then `sample_token` at temperature 0.7
//!     with `top_k = k` on the host (the path `forward_top` replaced).
//! The row is resident on the device before either starts, as the logits
//! are in decode.

use std::time::Instant;

use ojas_core::{Backend, Budget, Tensor};
use ojas_infer::{sample_token, SamplingConfig, SplitMix64};

const VOCAB: usize = 50304;
const ITERS: usize = 50;
const KS: [usize; 10] = [1, 2, 4, 8, 16, 32, 50, 128, 512, 1024];

fn probe<B: Backend>(name: &str, be: &B) {
    let mut r = SplitMix64::new(77);
    let host: Vec<f32> = (0..VOCAB)
        .map(|_| (r.next_f64() * 16.0 - 8.0) as f32)
        .collect();
    let x = be
        .upload(&Tensor::from_f32(&host, &[1, VOCAB], be.budget()).unwrap())
        .unwrap();
    be.sync().unwrap();
    println!("| backend | k | device topk + read k us | read row + host sample us | device / host |");
    println!("| :-- | --: | --: | --: | --: |");
    for k in KS {
        let cfg = SamplingConfig {
            temperature: 0.7,
            top_k: Some(k),
            top_p: None,
        };
        let (mut dev, mut hst) = (f64::INFINITY, f64::INFINITY);
        let mut rng = SplitMix64::new(1);
        for i in 0..ITERS + 3 {
            let t = Instant::now();
            let (v, ids) = be.topk_rows(&x, k).unwrap();
            let v = be.download(&v).unwrap();
            let ids = be.download(&ids).unwrap();
            be.sync().unwrap();
            std::hint::black_box((v.to_f32_vec().unwrap(), ids.u32_slice().unwrap().len()));
            let a = t.elapsed().as_secs_f64() * 1e6;
            let t = Instant::now();
            let row = be.download(&x).unwrap();
            be.sync().unwrap();
            let id = sample_token(&row.to_f32_vec().unwrap(), &cfg, &mut rng).unwrap();
            std::hint::black_box(id);
            let b = t.elapsed().as_secs_f64() * 1e6;
            if i >= 3 {
                dev = dev.min(a);
                hst = hst.min(b);
            }
        }
        println!("| {name} | {k} | {dev:.1} | {hst:.1} | {:.2} |", dev / hst);
    }
}

fn main() {
    let budget = || Budget::new(1 << 30);
    #[cfg(target_os = "macos")]
    probe("metal", &ojas_metal::MetalBackend::new(budget()).unwrap());
    probe("wgpu", &ojas_wgpu::WgpuBackend::open(budget()).unwrap());
}

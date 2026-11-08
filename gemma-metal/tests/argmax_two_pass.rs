//! `softcap_argmax_two_pass_encode` against the one-pass kernel it replaces in greedy decode
//! (`softcap_argmax_encode` with `GEMMA_METAL_ARGMAX_TWO_PASS=0`): same token for ties,
//! non-finite logits, softcap saturation and lengths that do not divide the 256 spans.
//! One test body in its own process: the env flag is read once.

use gemma_metal::kernels::{softcap_argmax_encode, softcap_argmax_two_pass_encode, ArgmaxScratch};
use gemma_metal::GemmaGpu;

fn values(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 40.0
        })
        .collect()
}

#[test]
fn two_pass_argmax_matches_the_one_pass_kernel() {
    std::env::set_var("GEMMA_METAL_ARGMAX_TWO_PASS", "0");
    let gpu = match GemmaGpu::new() {
        Ok(g) => g,
        Err(e) => panic!("Metal runtime unavailable (not a pass): {e}"),
    };
    let vocab = 262_144usize;
    let mut cases: Vec<(&str, Vec<f32>, f32, Option<u32>)> = Vec::new();
    cases.push(("random", values(vocab, 7), 30.0, None));
    let mut v = values(vocab, 8);
    v[200_000] = 99.0;
    v[3] = 99.0;
    cases.push(("tie across spans", v, 0.0, Some(3)));
    let mut v = values(vocab, 9);
    v[1030] = 99.0;
    v[1025] = 99.0;
    cases.push(("tie within a span", v, 0.0, Some(1025)));
    let mut v = values(vocab, 10);
    v[10] = f32::NAN;
    v[11] = f32::INFINITY;
    v[12] = f32::NEG_INFINITY;
    v[77_777] = 50.0;
    cases.push(("non-finite skipped", v, 30.0, Some(77_777)));
    let mut v = values(vocab, 11);
    v[150_000] = 1.0e6;
    v[90_000] = 2.0e6;
    cases.push(("softcap saturation tie", v, 30.0, None));
    cases.push(("all NaN", vec![f32::NAN; vocab], 30.0, Some(u32::MAX)));
    cases.push(("odd length", values(262_143, 12), 30.0, None));
    cases.push(("short", values(1000, 13), 30.0, None));

    for (name, logits, softcap, want) in cases {
        let n = logits.len() as u32;
        let buf = gpu.rt.alloc_buffer(logits.len() * 4).expect("logits");
        buf.write_f32(&logits);
        let mut scratch = ArgmaxScratch::new(&gpu, n).expect("scratch");
        let one = gpu.rt.alloc_buffer(4).expect("one-pass token");
        let two = gpu.rt.alloc_buffer(4).expect("two-pass token");
        one.write_f32(&[f32::from_bits(0x1234_5678)]);
        two.write_f32(&[f32::from_bits(0x1234_5678)]);
        softcap_argmax_encode(&gpu, &buf, softcap, n, &mut scratch, &one).expect("one-pass");
        gpu.barrier().expect("barrier");
        softcap_argmax_two_pass_encode(&gpu, &buf, softcap, n, &mut scratch, &two)
            .expect("two-pass");
        gpu.synchronize().expect("complete");
        let (a, b) = (one.read_f32()[0].to_bits(), two.read_f32()[0].to_bits());
        assert_eq!(a, b, "{name}: one-pass {a} vs two-pass {b}");
        assert_ne!(a, 0x1234_5678, "{name}: kernel wrote nothing");
        if let Some(w) = want {
            assert_eq!(b, w, "{name}");
        }
    }
}
